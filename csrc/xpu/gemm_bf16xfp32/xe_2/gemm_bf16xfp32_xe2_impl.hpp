/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * BF16xFP32 GEMM implementation using DualGemm infrastructure from sycl-tla.
 *
 * Emulates FP32-precision GEMM using two BF16 DPAS operations:
 *   result = X * W_high + (X * W_low) * scale
 *
 * where W_high and W_low are BF16 decompositions of an FP32 weight matrix W.
 **************************************************************************************************/
#pragma once

#include <torch/all.h>
#include "csrc/utils.h"

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"

// DualGemm infrastructure from applications/
#include "dual_gemm/kernel/xe_dual_gemm.hpp"
#include "dual_gemm/collective/xe_dual_gemm_mma.hpp"
#include "dual_gemm/collective/xe_dual_gemm_epilogue.hpp"

// Local epilogue
#include "bf16xfp32_epilogue.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace gemm_bf16xfp32 {
using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Tile policies for different M ranges
///////////////////////////////////////////////////////////////////////////////////////////////////

// Default policy: good for medium/large M (prefill).
// TileN = 64 (not 128) because N=192 = 3x64 divides evenly, whereas 128 splits
// 192 into 128+64 and wastes half of the second N-tile.
//   SG_M = 128/8 = 16, SG_N = 64/2 = 32 -> 16 subgroups per workgroup.
// NOTE: PipelineStages must stay at 2 here. V2 tried 3 to hide the mainloop
// stall (M=4096: XVE_ACTIVE ~37%, XVE_STALL ~52%, SHARED_FUNCTION_ACCESS_HOLD
// ~53%), but it REGRESSED large M (M=1024 84->98us, M=4096 197->221us): GRF is
// already maxed (256/thread) and the <128,64,64> tile + dual-DPAS accumulators
// leave no headroom, so a 3rd stage spills registers and makes stall worse.
// The real fix for the stall is lower per-WG work + more parallelism (Split-K),
// not deeper prefetch.
struct policy_default {
  using TileShape = Shape<_128, _64, _64>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
      Layout<TileShape>,
      Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>>::TiledMMA;
  static constexpr int PipelineStages = 2;
};

// Small M policy: for decode (M=1~32)
// Constraints:
//   - per-subgroup N (TileN / SGLayoutN) must be 32 (B-copy
//   XE_2D_U16x32x32_LD_V)
//   - per-subgroup M (TileM / SGLayoutM) must be 16 (A-copy
//   XE_2D_U16x16x32_LD_N)
//   - DualGemm prefetch needs Num_SGs >= 4 so the B prefetch block_non_contig
//     (TileK-tile / sgs_non_contig) stays <= 32.
// Here: SG_M = 32/2 = 16, SG_N = 64/2 = 32 -> 4 subgroups per workgroup.
struct policy_small_m {
  using TileShape = Shape<_32, _64, _64>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
      Layout<TileShape>,
      Layout<Shape<_2, _2, _1>, Stride<_2, _1, _0>>>::TiledMMA;
  static constexpr int PipelineStages = 2;
};

// Medium M policy: for small batch prefill (M=17~64)
struct policy_medium_m {
  using TileShape = Shape<_64, _64, _64>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
      Layout<TileShape>,
      Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>>::TiledMMA;
  static constexpr int PipelineStages = 2;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel type assembly
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Policy>
struct DualGemmKernelType {
  using ElementA = bfloat16_t;
  using ElementB = bfloat16_t;
  using ElementAccumulator = float;
  using ElementOutput = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using StrideA = cutlass::gemm::TagToStrideA_t<LayoutA>;
  using StrideB = cutlass::gemm::TagToStrideB_t<LayoutB>;
  using StrideC = cutlass::gemm::TagToStrideC_t<LayoutC>;
  using StrideD = cutlass::gemm::TagToStrideC_t<LayoutD>;

  using TileShape = typename Policy::TileShape;
  using TiledMma = typename Policy::TiledMma;
  static constexpr int PipelineStages = Policy::PipelineStages;

  using GmemTiledCopyA = XE_2D_U16x16x32_LD_N;
  using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;

  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  // Individual epilogues (pass-through, do NOT write output)
  using EpilogueOp = cutlass::epilogue::fusion::LinCombEltAct<
      cutlass::epilogue::thread::Identity,
      ElementOutput,
      float,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape,
      decltype(tile_shape(TiledMma()))>;

  static constexpr bool WriteEpilogueOutput = false;

  using CollectiveEpilogue = cutlass::epilogue::collective::DualGemmEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      ElementAccumulator,
      StrideC,
      ElementOutput,
      StrideD,
      FusionCallBacks,
      XE_2D_U32x8x16_LD_N,
      XE_2D_U32x8x16_ST_N,
      WriteEpilogueOutput>;

  // BF16xFP32 combining epilogue: D = acc_high + acc_low * scale
  using CollectiveBF16xFP32Epilogue =
      cutlass::epilogue::collective::BF16xFP32Epilogue<
          EpilogueDispatchPolicy,
          TileShape,
          void,  // No source C
          StrideC,
          ElementOutput,
          StrideD,
          void,  // No load op
          XE_2D_U32x8x16_ST_N>;

  // Mainloop: shared A, dual B (W_high, W_low)
  using CollectiveMainloop = cutlass::gemm::collective::DualGemmMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementA,
      StrideA,
      ElementB,
      StrideB,
      TiledMma,
      GmemTiledCopyA,
      GmemTiledCopyB>;

  // Full kernel
  using GemmKernel = cutlass::gemm::kernel::DualGemm<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      CollectiveEpilogue,
      CollectiveBF16xFP32Epilogue>;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// SYCL kernel name tags
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Policy>
class GemmBF16xFP32KernelName;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Launch function
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Policy>
void launch_gemm_bf16xfp32(
    sycl::queue& queue,
    const bfloat16_t* A,       // [M, K]
    const bfloat16_t* B_high,  // [K, N]
    const bfloat16_t* B_low,   // [K, N]
    float* D,                  // [M, N]
    int M,
    int N,
    int K,
    float scale) {
  using Kernel = typename DualGemmKernelType<Policy>::GemmKernel;
  using StrideA = typename DualGemmKernelType<Policy>::StrideA;
  using StrideB = typename DualGemmKernelType<Policy>::StrideB;
  using StrideD = typename DualGemmKernelType<Policy>::StrideD;

  auto stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
  auto stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
  auto stride_D =
      cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));

  using EpilogueArguments = typename Kernel::EpilogueArguments0;
  EpilogueArguments epilogue_args0{
      {1.0f, 0.0f}, nullptr, stride_D, nullptr, stride_D};
  EpilogueArguments epilogue_args1{
      {1.0f, 0.0f}, nullptr, stride_D, nullptr, stride_D};

  using BF16xFP32EpilogueArguments =
      typename Kernel::DualGemmElemActEpilogueArguments;
  BF16xFP32EpilogueArguments bf16xfp32_args{D, stride_D, scale};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  typename Kernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {A, stride_A, B_high, stride_B, B_low, stride_B},
      epilogue_args0,
      epilogue_args1,
      bf16xfp32_args,
      hw_info};

  TORCH_CHECK(
      Kernel::can_implement(arguments),
      "gemm_bf16xfp32: problem size not supported by kernel. "
      "M=",
      M,
      " N=",
      N,
      " K=",
      K);

  size_t workspace_size = Kernel::get_workspace_size(arguments);
  TORCH_CHECK(
      workspace_size == 0, "gemm_bf16xfp32: unexpected workspace requirement");

  typename Kernel::Params params =
      Kernel::to_underlying_arguments(arguments, nullptr);

  dim3 block = Kernel::get_block_shape();
  dim3 grid = Kernel::get_grid_shape(params);

  sycl::range<3> local(1, 1, block.x);
  sycl::range<3> global(grid.z, grid.y, grid.x);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{
      syclex::sub_group_size<Kernel::DispatchPolicy::SubgroupSize>,
      intelex::grf_size<256>};

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<GemmBF16xFP32KernelName<Policy>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          // SharedStorageSize is 0 for DualGemm
          char* smem_buf = nullptr;
          Kernel kernel_op;
          kernel_op(params, smem_buf);
        });
  });
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Top-level dispatch (selects tile policy based on M)
///////////////////////////////////////////////////////////////////////////////////////////////////

inline at::Tensor gemm_bf16xfp32_xe2_impl(
    at::Tensor& A,       // [M, K] bf16
    at::Tensor& B_high,  // [K, N] bf16 (transposed weight)
    at::Tensor& B_low,   // [K, N] bf16 (transposed weight)
    double scale) {
  TORCH_CHECK(A.dim() == 2, "A must be 2D [M, K]");
  TORCH_CHECK(B_high.dim() == 2, "B_high must be 2D [K, N]");
  TORCH_CHECK(B_low.dim() == 2, "B_low must be 2D [K, N]");
  TORCH_CHECK(A.dtype() == at::kBFloat16, "A must be BF16");
  TORCH_CHECK(B_high.dtype() == at::kBFloat16, "B_high must be BF16");
  TORCH_CHECK(B_low.dtype() == at::kBFloat16, "B_low must be BF16");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B_high.is_contiguous(), "B_high must be contiguous");
  TORCH_CHECK(B_low.is_contiguous(), "B_low must be contiguous");

  int M = A.size(0);
  int K = A.size(1);
  int N = B_high.size(1);

  TORCH_CHECK(B_high.size(0) == K, "B_high.size(0) must match A.size(1) (K)");
  TORCH_CHECK(B_low.size(0) == K, "B_low.size(0) must match A.size(1) (K)");
  TORCH_CHECK(
      B_low.size(1) == N, "B_low.size(1) must match B_high.size(1) (N)");
  TORCH_CHECK(N % 4 == 0, "N must be divisible by 4");

  // Allocate output [M, N] fp32
  auto D = at::empty({M, N}, A.options().dtype(at::kFloat));

  auto& queue = at::xpu::getCurrentXPUStream(A.device().index()).queue();

  const bfloat16_t* ptr_A = reinterpret_cast<const bfloat16_t*>(A.data_ptr());
  const bfloat16_t* ptr_B_high =
      reinterpret_cast<const bfloat16_t*>(B_high.data_ptr());
  const bfloat16_t* ptr_B_low =
      reinterpret_cast<const bfloat16_t*>(B_low.data_ptr());
  float* ptr_D = reinterpret_cast<float*>(D.data_ptr());

  float scale_f = static_cast<float>(scale);

  // Select tile policy based on M
  if (M <= 32) {
    launch_gemm_bf16xfp32<policy_small_m>(
        queue, ptr_A, ptr_B_high, ptr_B_low, ptr_D, M, N, K, scale_f);
  } else if (M <= 64) {
    launch_gemm_bf16xfp32<policy_medium_m>(
        queue, ptr_A, ptr_B_high, ptr_B_low, ptr_D, M, N, K, scale_f);
  } else {
    launch_gemm_bf16xfp32<policy_default>(
        queue, ptr_A, ptr_B_high, ptr_B_low, ptr_D, M, N, K, scale_f);
  }

  return D;
}

}  // namespace gemm_bf16xfp32
