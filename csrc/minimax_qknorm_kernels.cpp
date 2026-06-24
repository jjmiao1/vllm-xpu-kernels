/*
 * MiniMax QK Norm SYCL kernels (split variant, optimized).
 *
 * Two independent custom ops designed to be called from Python with a
 * Python-level allreduce in between:
 *
 *   qk_var = torch.ops._C.minimax_qk_local_variance(qkv, q_size, kv_size)
 *   qk_var = tensor_model_parallel_all_reduce(qk_var)
 *   q, k = torch.ops._C.minimax_qk_rms_norm(qkv, qk_var, q_w, k_w,
 *                                            q_size, kv_size, tp_world, eps)
 *
 * Key optimizations:
 *  - One WG per token (processes both Q and K, halves launch overhead)
 *  - WG_SIZE=128, optimal for B60 EU utilization
 *  - Var: reqd_sub_group_size(32) + combined Q+K loop + sub-group pre-reduction
 *  - Norm: explicit sycl::vec<uint16_t,8> loads/stores for guaranteed coalescing
 *  - bf16 round-to-nearest-even via bit manipulation (avoids scalar cast overhead)
 */

#include <sycl/sycl.hpp>

#include <ATen/DeviceGuard.h>

#include "dispatch_utils.h"
#include "quantization/utils.h"
#include "utils.h"

namespace vllm {

// ============================================================================
// Kernel 1: Local variance (mean of squares) for Q and K
// ============================================================================
// ONE work-group per token. Combined Q+K loop with reqd_sub_group_size(32)
// for wider memory coalescing (32×16 = 512 bytes per sub-group access).
// Sub-group pre-reduction minimizes SLM traffic in final reduce_over_group.

template <typename scalar_t, int WG_SIZE, int VEC = 8>
struct minimax_qk_local_var_kernel {
  minimax_qk_local_var_kernel(
      const scalar_t* __restrict__ qkv_,
      float* __restrict__ var_out_,
      const int num_tokens_,
      const int q_size_,
      const int kv_size_,
      const int qkv_stride_)
      : qkv(qkv_),
        var_out(var_out_),
        num_tokens(num_tokens_),
        q_size(q_size_),
        kv_size(kv_size_),
        qkv_stride(qkv_stride_) {}

  [[sycl::reqd_sub_group_size(32)]]
  void operator()(sycl::nd_item<1> item) const {
    const int token_idx = item.get_group(0);
    if (token_idx >= num_tokens) return;

    const int lid = item.get_local_id(0);
    const uint16_t* row =
        reinterpret_cast<const uint16_t*>(qkv + token_idx * qkv_stride);

    // Combined Q+K loop: all threads stay busy across full (q+kv)/VEC chunks
    const int total_vec = (q_size + kv_size) / VEC;
    const int q_vec_boundary = q_size / VEC;

    float q_sum = 0.0f;
    float k_sum = 0.0f;

    for (int chunk = lid; chunk < total_vec; chunk += WG_SIZE) {
      const uint16_t* p = row + chunk * VEC;
      uint16_t raw[VEC];
      *reinterpret_cast<sycl::vec<uint16_t, VEC>*>(raw) =
          *reinterpret_cast<const sycl::vec<uint16_t, VEC>*>(p);

      float local_sum = 0.0f;
#pragma unroll
      for (int v = 0; v < VEC; v++) {
        uint32_t bits = static_cast<uint32_t>(raw[v]) << 16;
        float val;
        __builtin_memcpy(&val, &bits, 4);
        local_sum += val * val;
      }

      if (chunk < q_vec_boundary)
        q_sum += local_sum;
      else
        k_sum += local_sum;
    }

    // Sub-group reduction (hardware shuffle, no barrier)
    auto sg = item.get_sub_group();
    float sg_q = sycl::reduce_over_group(sg, q_sum, sycl::plus<float>());
    float sg_k = sycl::reduce_over_group(sg, k_sum, sycl::plus<float>());

    // Final reduction: only sg leaders contribute non-zero
    float q_total = sycl::reduce_over_group(
        item.get_group(),
        sg.get_local_id()[0] == 0 ? sg_q : 0.0f, sycl::plus<float>());
    float k_total = sycl::reduce_over_group(
        item.get_group(),
        sg.get_local_id()[0] == 0 ? sg_k : 0.0f, sycl::plus<float>());

    if (lid == 0) {
      var_out[token_idx * 2 + 0] = q_total / static_cast<float>(q_size);
      var_out[token_idx * 2 + 1] = k_total / static_cast<float>(kv_size);
    }
  }

 private:
  const scalar_t* __restrict__ qkv;
  float* __restrict__ var_out;
  const int num_tokens;
  const int q_size;
  const int kv_size;
  const int qkv_stride;
};

// ============================================================================
// Kernel 2: Post-allreduce RMS norm application
// ============================================================================
// ONE work-group per token. Separate Q/K loops (no branch divergence).
// Explicit sycl::vec<uint16_t,8> loads/stores guarantee 16-byte coalesced
// memory transactions. bf16 rounding via bit manipulation avoids scalar
// cast overhead.

template <typename scalar_t, int WG_SIZE, int VEC = 8>
struct minimax_qk_rms_norm_kernel {
  minimax_qk_rms_norm_kernel(
      const scalar_t* __restrict__ qkv_,
      const float* __restrict__ var_,
      const scalar_t* __restrict__ q_weight_,
      const scalar_t* __restrict__ k_weight_,
      scalar_t* __restrict__ q_out_,
      scalar_t* __restrict__ k_out_,
      float eps_,
      float inv_nranks_,
      const int num_tokens_,
      const int q_size_,
      const int kv_size_,
      const int qkv_stride_)
      : qkv(qkv_),
        var(var_),
        q_weight(q_weight_),
        k_weight(k_weight_),
        q_out(q_out_),
        k_out(k_out_),
        eps(eps_),
        inv_nranks(inv_nranks_),
        num_tokens(num_tokens_),
        q_size(q_size_),
        kv_size(kv_size_),
        qkv_stride(qkv_stride_) {}

  void operator()(sycl::nd_item<1> item) const {
    const int token_idx = item.get_group(0);
    if (token_idx >= num_tokens) return;

    const int lid = item.get_local_id(0);
    const uint16_t* row =
        reinterpret_cast<const uint16_t*>(qkv + token_idx * qkv_stride);

    float q_var = var[token_idx * 2 + 0];
    float k_var = var[token_idx * 2 + 1];
    float inv_rms_q = sycl::rsqrt(q_var * inv_nranks + eps);
    float inv_rms_k = sycl::rsqrt(k_var * inv_nranks + eps);

    // --- Normalize Q: explicit vec load/store ---
    {
      const uint16_t* q_w_raw = reinterpret_cast<const uint16_t*>(q_weight);
      uint16_t* q_dst =
          reinterpret_cast<uint16_t*>(q_out + token_idx * q_size);
      const int q_vec_elems = q_size / VEC;

      for (int chunk = lid; chunk < q_vec_elems; chunk += WG_SIZE) {
        const uint16_t* src_p = row + chunk * VEC;
        const uint16_t* wt_p = q_w_raw + chunk * VEC;

        uint16_t src_raw[VEC], wt_raw[VEC], out_raw[VEC];
        *reinterpret_cast<sycl::vec<uint16_t, VEC>*>(src_raw) =
            *reinterpret_cast<const sycl::vec<uint16_t, VEC>*>(src_p);
        *reinterpret_cast<sycl::vec<uint16_t, VEC>*>(wt_raw) =
            *reinterpret_cast<const sycl::vec<uint16_t, VEC>*>(wt_p);

#pragma unroll
        for (int v = 0; v < VEC; v++) {
          uint32_t xbits = static_cast<uint32_t>(src_raw[v]) << 16;
          uint32_t wbits = static_cast<uint32_t>(wt_raw[v]) << 16;
          float x, w;
          __builtin_memcpy(&x, &xbits, 4);
          __builtin_memcpy(&w, &wbits, 4);
          float y = x * inv_rms_q * w;
          uint32_t ybits;
          __builtin_memcpy(&ybits, &y, 4);
          uint32_t rounding = ((ybits >> 16) & 1) + 0x7FFF;
          out_raw[v] = static_cast<uint16_t>((ybits + rounding) >> 16);
        }
        *reinterpret_cast<sycl::vec<uint16_t, VEC>*>(q_dst + chunk * VEC) =
            *reinterpret_cast<sycl::vec<uint16_t, VEC>*>(out_raw);
      }
    }

    // --- Normalize K: explicit vec load/store ---
    {
      const uint16_t* k_row = row + q_size;
      const uint16_t* k_w_raw = reinterpret_cast<const uint16_t*>(k_weight);
      uint16_t* k_dst =
          reinterpret_cast<uint16_t*>(k_out + token_idx * kv_size);
      const int k_vec_elems = kv_size / VEC;

      for (int chunk = lid; chunk < k_vec_elems; chunk += WG_SIZE) {
        const uint16_t* src_p = k_row + chunk * VEC;
        const uint16_t* wt_p = k_w_raw + chunk * VEC;

        uint16_t src_raw[VEC], wt_raw[VEC], out_raw[VEC];
        *reinterpret_cast<sycl::vec<uint16_t, VEC>*>(src_raw) =
            *reinterpret_cast<const sycl::vec<uint16_t, VEC>*>(src_p);
        *reinterpret_cast<sycl::vec<uint16_t, VEC>*>(wt_raw) =
            *reinterpret_cast<const sycl::vec<uint16_t, VEC>*>(wt_p);

#pragma unroll
        for (int v = 0; v < VEC; v++) {
          uint32_t xbits = static_cast<uint32_t>(src_raw[v]) << 16;
          uint32_t wbits = static_cast<uint32_t>(wt_raw[v]) << 16;
          float x, w;
          __builtin_memcpy(&x, &xbits, 4);
          __builtin_memcpy(&w, &wbits, 4);
          float y = x * inv_rms_k * w;
          uint32_t ybits;
          __builtin_memcpy(&ybits, &y, 4);
          uint32_t rounding = ((ybits >> 16) & 1) + 0x7FFF;
          out_raw[v] = static_cast<uint16_t>((ybits + rounding) >> 16);
        }
        *reinterpret_cast<sycl::vec<uint16_t, VEC>*>(k_dst + chunk * VEC) =
            *reinterpret_cast<sycl::vec<uint16_t, VEC>*>(out_raw);
      }
    }
  }

 private:
  const scalar_t* __restrict__ qkv;
  const float* __restrict__ var;
  const scalar_t* __restrict__ q_weight;
  const scalar_t* __restrict__ k_weight;
  scalar_t* __restrict__ q_out;
  scalar_t* __restrict__ k_out;
  float eps;
  float inv_nranks;
  const int num_tokens;
  const int q_size;
  const int kv_size;
  const int qkv_stride;
};

// ============================================================================
// Op 1: minimax_qk_local_variance
// ============================================================================

torch::Tensor minimax_qk_local_variance(
    torch::Tensor qkv,
    int64_t q_size,
    int64_t kv_size) {
  const at::DeviceGuard device_guard(qkv.device());

  TORCH_CHECK(qkv.dim() == 2, "minimax_qk_local_variance: qkv must be 2D");
  TORCH_CHECK(qkv.is_contiguous(),
              "minimax_qk_local_variance: qkv must be contiguous");

  int64_t num_tokens = qkv.size(0);
  int64_t qkv_stride = qkv.stride(0);

  auto var_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(qkv.device());
  auto qk_var = torch::empty({num_tokens, 2}, var_options);

  sycl::queue& queue = vllm::xpu::vllmGetQueue(qkv.device().index());

  // One WG per token (handles both Q and K)
  constexpr int WG_SIZE = 128;
  int num_groups = static_cast<int>(num_tokens);

  VLLM_DISPATCH_HALF_TYPES(
      qkv.scalar_type(), "minimax_qk_local_variance", [&] {
        queue.parallel_for(
            sycl::nd_range<1>(num_groups * WG_SIZE, WG_SIZE),
            minimax_qk_local_var_kernel<scalar_t, WG_SIZE>(
                qkv.data_ptr<scalar_t>(),
                qk_var.data_ptr<float>(),
                static_cast<int>(num_tokens),
                static_cast<int>(q_size),
                static_cast<int>(kv_size),
                static_cast<int>(qkv_stride)));
      });

  return qk_var;
}

// ============================================================================
// Op 2: minimax_qk_rms_norm
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor> minimax_qk_rms_norm(
    torch::Tensor qkv,
    torch::Tensor qk_var,
    torch::Tensor norm_weight_q,
    torch::Tensor norm_weight_k,
    int64_t q_size,
    int64_t kv_size,
    int64_t tp_world,
    double eps) {
  const at::DeviceGuard device_guard(qkv.device());

  TORCH_CHECK(qkv.dim() == 2, "minimax_qk_rms_norm: qkv must be 2D");
  TORCH_CHECK(qkv.is_contiguous(),
              "minimax_qk_rms_norm: qkv must be contiguous");
  TORCH_CHECK(qk_var.dim() == 2 && qk_var.size(1) == 2,
              "minimax_qk_rms_norm: qk_var must be [num_tokens, 2]");

  int64_t num_tokens = qkv.size(0);
  int64_t qkv_stride = qkv.stride(0);

  auto q_out = torch::empty({num_tokens, q_size}, qkv.options());
  auto k_out = torch::empty({num_tokens, kv_size}, qkv.options());

  sycl::queue& queue = vllm::xpu::vllmGetQueue(qkv.device().index());

  // One WG per token (handles both Q and K)
  constexpr int WG_SIZE = 128;
  int num_groups = static_cast<int>(num_tokens);
  float inv_nranks = 1.0f / static_cast<float>(tp_world);

  VLLM_DISPATCH_HALF_TYPES(
      qkv.scalar_type(), "minimax_qk_rms_norm", [&] {
        queue.parallel_for(
            sycl::nd_range<1>(num_groups * WG_SIZE, WG_SIZE),
            minimax_qk_rms_norm_kernel<scalar_t, WG_SIZE>(
                qkv.data_ptr<scalar_t>(),
                qk_var.data_ptr<float>(),
                norm_weight_q.data_ptr<scalar_t>(),
                norm_weight_k.data_ptr<scalar_t>(),
                q_out.data_ptr<scalar_t>(),
                k_out.data_ptr<scalar_t>(),
                static_cast<float>(eps),
                inv_nranks,
                static_cast<int>(num_tokens),
                static_cast<int>(q_size),
                static_cast<int>(kv_size),
                static_cast<int>(qkv_stride)));
      });

  return {q_out, k_out};
}

} // namespace vllm
