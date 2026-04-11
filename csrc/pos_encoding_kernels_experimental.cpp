#include <sycl/sycl.hpp>
#include "utils.h"
#include "dispatch_utils.h"
#include "ops.h"

#include <algorithm>
#include <cmath>
#include <c10/macros/Macros.h>

namespace vllm {

// Non-SLM kernel: restructured loop with cos/sin from global memory.
// Each work-item has a fixed vec_dim index and iterates over heads,
// eliminating per-iteration integer div/mod.  cos/sin vectors are loaded
// once from global memory before the head loop (L1-cached for small
// token counts typical of decode).
template <typename scalar_t, int VEC_SIZE>
class rotary_embedding_neox_vec_kernel {
 public:
  rotary_embedding_neox_vec_kernel(
      const int64_t* __restrict__ positions_,
      scalar_t* __restrict__ query_,
      scalar_t* __restrict__ key_,
      const scalar_t* __restrict__ cos_sin_cache_,
      const int rot_dim_,
      const int64_t query_stride_,
      const int64_t key_stride_,
      const int64_t head_stride_,
      const int num_heads_,
      const int num_kv_heads_,
      const int num_vec_dims_,
      const int heads_per_block_)
      : positions(positions_),
        query(query_),
        key(key_),
        cos_sin_cache(cos_sin_cache_),
        rot_dim(rot_dim_),
        query_stride(query_stride_),
        key_stride(key_stride_),
        head_stride(head_stride_),
        num_heads(num_heads_),
        num_kv_heads(num_kv_heads_),
        num_vec_dims(num_vec_dims_),
        heads_per_block(heads_per_block_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    using vec_t = sycl::vec<scalar_t, VEC_SIZE>;

    const int embed_dim = rot_dim / 2;
    const int token_idx = item_ct1.get_group(2);
    const int local_id = item_ct1.get_local_id(2);

    const int64_t pos = positions[token_idx];
    const scalar_t* cos_ptr = cos_sin_cache + pos * rot_dim;
    const scalar_t* sin_ptr = cos_ptr + embed_dim;

    // Each work-item maps to a unique (start_head, vec_idx) pair.
    const int vec_idx = local_id % num_vec_dims;
    const int dim_idx = vec_idx * VEC_SIZE;
    const int start_head = local_id / num_vec_dims;

    // Load cos/sin once from global memory (L1-cached for decode).
    const vec_t cos_vec =
        *reinterpret_cast<const vec_t*>(cos_ptr + dim_idx);
    const vec_t sin_vec =
        *reinterpret_cast<const vec_t*>(sin_ptr + dim_idx);

    for (int head_idx = start_head; head_idx < num_heads;
         head_idx += heads_per_block) {
      const int64_t q_offset =
          token_idx * query_stride + head_idx * head_stride;

      vec_t q1 =
          *reinterpret_cast<const vec_t*>(&query[q_offset + dim_idx]);
      vec_t q2 = *reinterpret_cast<const vec_t*>(
          &query[q_offset + embed_dim + dim_idx]);
      vec_t out_q1;
      vec_t out_q2;
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        out_q1[v] = q1[v] * cos_vec[v] - q2[v] * sin_vec[v];
        out_q2[v] = q2[v] * cos_vec[v] + q1[v] * sin_vec[v];
      }
      *reinterpret_cast<vec_t*>(&query[q_offset + dim_idx]) = out_q1;
      *reinterpret_cast<vec_t*>(
          &query[q_offset + embed_dim + dim_idx]) = out_q2;
    }

    if (key != nullptr) {
      for (int head_idx = start_head; head_idx < num_kv_heads;
           head_idx += heads_per_block) {
        const int64_t k_offset =
            token_idx * key_stride + head_idx * head_stride;

        vec_t k1 =
            *reinterpret_cast<const vec_t*>(&key[k_offset + dim_idx]);
        vec_t k2 = *reinterpret_cast<const vec_t*>(
            &key[k_offset + embed_dim + dim_idx]);
        vec_t out_k1;
        vec_t out_k2;
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          out_k1[v] = k1[v] * cos_vec[v] - k2[v] * sin_vec[v];
          out_k2[v] = k2[v] * cos_vec[v] + k1[v] * sin_vec[v];
        }
        *reinterpret_cast<vec_t*>(&key[k_offset + dim_idx]) = out_k1;
        *reinterpret_cast<vec_t*>(
            &key[k_offset + embed_dim + dim_idx]) = out_k2;
      }
    }
  }

 private:
  const int64_t* __restrict__ positions;
  scalar_t* __restrict__ query;
  scalar_t* __restrict__ key;
  const scalar_t* __restrict__ cos_sin_cache;
  const int rot_dim;
  const int64_t query_stride;
  const int64_t key_stride;
  const int64_t head_stride;
  const int num_heads;
  const int num_kv_heads;
  const int num_vec_dims;
  const int heads_per_block;
};

// SLM kernel: cos/sin values are loaded into Shared Local Memory once per
// work-group and reused across all heads.  For nh=96 this eliminates ~95
// redundant global memory reads per cos/sin element.  Beneficial when many
// tokens compete for L1 cache space (prefill / prefill_session_cache).
template <typename scalar_t, int VEC_SIZE>
class rotary_embedding_neox_slm_kernel {
 public:
  rotary_embedding_neox_slm_kernel(
      const int64_t* __restrict__ positions_,
      scalar_t* __restrict__ query_,
      scalar_t* __restrict__ key_,
      const scalar_t* __restrict__ cos_sin_cache_,
      sycl::local_accessor<scalar_t, 1> slm_,
      const int rot_dim_,
      const int64_t query_stride_,
      const int64_t key_stride_,
      const int64_t head_stride_,
      const int num_heads_,
      const int num_kv_heads_,
      const int num_vec_dims_,
      const int heads_per_block_)
      : positions(positions_),
        query(query_),
        key(key_),
        cos_sin_cache(cos_sin_cache_),
        slm(slm_),
        rot_dim(rot_dim_),
        query_stride(query_stride_),
        key_stride(key_stride_),
        head_stride(head_stride_),
        num_heads(num_heads_),
        num_kv_heads(num_kv_heads_),
        num_vec_dims(num_vec_dims_),
        heads_per_block(heads_per_block_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    using vec_t = sycl::vec<scalar_t, VEC_SIZE>;

    const int embed_dim = rot_dim / 2;
    const int token_idx = item_ct1.get_group(2);
    const int local_id = item_ct1.get_local_id(2);
    const int block_size = item_ct1.get_local_range(2);

    // Phase 1: Cooperatively load cos/sin from global memory into SLM.
    const int64_t pos = positions[token_idx];
    const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
    for (int i = local_id; i < rot_dim; i += block_size) {
      slm[i] = cache_ptr[i];
    }
    sycl::group_barrier(item_ct1.get_group());

    // Get raw SLM pointer for vectorized access.
    scalar_t* slm_ptr =
        slm.template get_multi_ptr<sycl::access::decorated::no>().get();
    const scalar_t* cos_slm = slm_ptr;
    const scalar_t* sin_slm = slm_ptr + embed_dim;

    // Phase 2: Process with fixed vec_dim, iterating over heads.
    const int vec_idx = local_id % num_vec_dims;
    const int dim_idx = vec_idx * VEC_SIZE;
    const int start_head = local_id / num_vec_dims;

    // Load cos/sin from SLM once for all heads at this dim.
    const vec_t cos_vec =
        *reinterpret_cast<const vec_t*>(cos_slm + dim_idx);
    const vec_t sin_vec =
        *reinterpret_cast<const vec_t*>(sin_slm + dim_idx);

    for (int head_idx = start_head; head_idx < num_heads;
         head_idx += heads_per_block) {
      const int64_t q_offset =
          token_idx * query_stride + head_idx * head_stride;

      vec_t q1 =
          *reinterpret_cast<const vec_t*>(&query[q_offset + dim_idx]);
      vec_t q2 = *reinterpret_cast<const vec_t*>(
          &query[q_offset + embed_dim + dim_idx]);
      vec_t out_q1;
      vec_t out_q2;
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        out_q1[v] = q1[v] * cos_vec[v] - q2[v] * sin_vec[v];
        out_q2[v] = q2[v] * cos_vec[v] + q1[v] * sin_vec[v];
      }
      *reinterpret_cast<vec_t*>(&query[q_offset + dim_idx]) = out_q1;
      *reinterpret_cast<vec_t*>(
          &query[q_offset + embed_dim + dim_idx]) = out_q2;
    }

    if (key != nullptr) {
      for (int head_idx = start_head; head_idx < num_kv_heads;
           head_idx += heads_per_block) {
        const int64_t k_offset =
            token_idx * key_stride + head_idx * head_stride;

        vec_t k1 =
            *reinterpret_cast<const vec_t*>(&key[k_offset + dim_idx]);
        vec_t k2 = *reinterpret_cast<const vec_t*>(
            &key[k_offset + embed_dim + dim_idx]);
        vec_t out_k1;
        vec_t out_k2;
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          out_k1[v] = k1[v] * cos_vec[v] - k2[v] * sin_vec[v];
          out_k2[v] = k2[v] * cos_vec[v] + k1[v] * sin_vec[v];
        }
        *reinterpret_cast<vec_t*>(&key[k_offset + dim_idx]) = out_k1;
        *reinterpret_cast<vec_t*>(
            &key[k_offset + embed_dim + dim_idx]) = out_k2;
      }
    }
  }

 private:
  const int64_t* __restrict__ positions;
  scalar_t* __restrict__ query;
  scalar_t* __restrict__ key;
  const scalar_t* __restrict__ cos_sin_cache;
  sycl::local_accessor<scalar_t, 1> slm;
  const int rot_dim;
  const int64_t query_stride;
  const int64_t key_stride;
  const int64_t head_stride;
  const int num_heads;
  const int num_kv_heads;
  const int num_vec_dims;
  const int heads_per_block;
};

}  // namespace vllm

template <typename scalar_t>
void call_rotary_embedding_kernel_experimental(
    torch::Tensor& positions,
    torch::Tensor& query,
    std::optional<torch::Tensor> key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox) {
  // GPT-J interleaved layout doesn't benefit from vectorization — the
  // natural unit is a single (x,y) pair which matches the scalar reference.
  if (!is_neox) {
    rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox);
    return;
  }

  constexpr int VEC_SIZE = 4;
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  const int64_t num_tokens = positions.numel();
  if (num_tokens == 0) return;
  const int positions_ndim = positions.dim();

  TORCH_CHECK(
      positions_ndim == 1 || positions_ndim == 2,
      "positions must have shape [num_tokens] or [batch_size, seq_len]");
  if (positions_ndim == 1) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) &&
            (!key.has_value() || key->size(0) == positions.size(0)),
        "query, key and positions must have the same number of tokens");
  }
  if (positions_ndim == 2) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) &&
            (!key.has_value() || key->size(0) == positions.size(0)) &&
            query.size(1) == positions.size(1) &&
            (!key.has_value() || key->size(1) == positions.size(1)),
        "query, key and positions must have the same batch_size and seq_len");
  }

  const int query_hidden_size = query.numel() / num_tokens;
  const int key_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;
  TORCH_CHECK(query_hidden_size % head_size == 0);
  TORCH_CHECK(key_hidden_size % head_size == 0);

  const int num_heads = query_hidden_size / head_size;
  const int num_kv_heads =
      key.has_value() ? key_hidden_size / head_size : num_heads;
  TORCH_CHECK(num_heads % num_kv_heads == 0);

  const int rot_dim = cos_sin_cache.size(1);
  const int embed_dim = rot_dim / 2;
  const int seq_dim_idx = positions_ndim - 1;
  const int64_t query_stride = query.stride(seq_dim_idx);
  const int64_t key_stride = key.has_value() ? key->stride(seq_dim_idx) : 0;
  const int query_ndim = query.dim();
  const int64_t head_stride =
      (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;

  const bool aligned_embed_dim = embed_dim % VEC_SIZE == 0;
  const bool query_last_dim_contig = query.stride(-1) == 1;
  const bool key_last_dim_contig = !key.has_value() || key->stride(-1) == 1;
  const bool cache_last_dim_contig = cos_sin_cache.stride(1) == 1;
  if (!aligned_embed_dim || !query_last_dim_contig || !key_last_dim_contig ||
      !cache_last_dim_contig) {
    rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox);
    return;
  }

  auto positions_ptr = positions.data_ptr<int64_t>();
  auto query_ptr = query.data_ptr<scalar_t>();
  auto key_ptr = key.has_value() ? key->data_ptr<scalar_t>() : nullptr;
  auto cos_sin_cache_ptr = cos_sin_cache.data_ptr<scalar_t>();

  const int num_vec_dims = embed_dim / VEC_SIZE;

  // Choose block_size = heads_per_block * num_vec_dims such that
  // heads_per_block evenly divides num_heads (no tail waste).
  int heads_per_block = std::min<int>(num_heads, 512 / num_vec_dims);
  while (heads_per_block > 1 && num_heads % heads_per_block != 0) {
    --heads_per_block;
  }
  int block_size = heads_per_block * num_vec_dims;

  // Ensure block_size >= 32 for reqd_sub_group_size(32) compatibility.
  // If too small (e.g. nh=1 with small rot_dim), fall back to reference.
  if (block_size < 32) {
    rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox);
    return;
  }

  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, block_size);

  at::DeviceGuard device_guard(query.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  // Use SLM kernel for large token counts (prefill) where many work-groups
  // compete for L1 cache space.  For small token counts (decode), L1 is
  // sufficient and the SLM barrier overhead would hurt.
  constexpr int SLM_TOKEN_THRESHOLD = 256;

  if (num_tokens >= SLM_TOKEN_THRESHOLD) {
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<sycl_t, 1> slm(sycl::range<1>(rot_dim), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          vllm::rotary_embedding_neox_slm_kernel<sycl_t, VEC_SIZE>(
              positions_ptr,
              (sycl_t*)query_ptr,
              (sycl_t*)key_ptr,
              (sycl_t*)cos_sin_cache_ptr,
              slm,
              rot_dim,
              query_stride,
              key_stride,
              head_stride,
              num_heads,
              num_kv_heads,
              num_vec_dims,
              heads_per_block));
    });
  } else {
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          vllm::rotary_embedding_neox_vec_kernel<sycl_t, VEC_SIZE>(
              positions_ptr,
              (sycl_t*)query_ptr,
              (sycl_t*)key_ptr,
              (sycl_t*)cos_sin_cache_ptr,
              rot_dim,
              query_stride,
              key_stride,
              head_stride,
              num_heads,
              num_kv_heads,
              num_vec_dims,
              heads_per_block));
    });
  }
}

void rotary_embedding_experimental(
    torch::Tensor& positions,
    torch::Tensor& query,
    std::optional<torch::Tensor> key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox) {
  VLLM_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "rotary_embedding_experimental", [&] {
        call_rotary_embedding_kernel_experimental<scalar_t>(
            positions, query, key, head_size, cos_sin_cache, is_neox);
      });
}