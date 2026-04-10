#include <sycl/sycl.hpp>
#include "utils.h"
#include "dispatch_utils.h"
#include "ops.h"

#include <algorithm>
#include <cmath>
#include <c10/macros/Macros.h>

namespace vllm {

// Optimized NeoX rotary embedding kernel: one work-group per token,
// work-items are distributed across (head, vec_dim) pairs so head work is
// fully parallelized instead of being serialized in a loop.  Uses
// VEC_SIZE-wide vector loads/stores to reduce iteration count by VEC_SIZE
// compared to the scalar reference kernel.
//
// Only applicable to NeoX-style rotary embedding where x-half and y-half
// are stored in separate contiguous regions, enabling clean vectorization.
// GPT-J style (interleaved pairs) falls back to the reference kernel.
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
      const int head_size_)
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
        head_size(head_size_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    using vec_t = sycl::vec<scalar_t, VEC_SIZE>;

    const int embed_dim = rot_dim / 2;
    const int num_vec_dims = embed_dim / VEC_SIZE;

    const int token_idx = item_ct1.get_group(2);
    const int local_id = item_ct1.get_local_id(2);
    const int block_size = item_ct1.get_local_range(2);

    const int64_t pos = positions[token_idx];
    const scalar_t* cos_ptr = cos_sin_cache + pos * rot_dim;
    const scalar_t* sin_ptr = cos_ptr + embed_dim;

    // Process query: work-items iterate over (head, vec_dim) pairs.
    const int total_q_vecs = num_heads * num_vec_dims;
    for (int i = local_id; i < total_q_vecs; i += block_size) {
      const int head_idx = i / num_vec_dims;
      const int vec_idx = i % num_vec_dims;
      const int dim_idx = vec_idx * VEC_SIZE;

      const int64_t q_offset =
          token_idx * query_stride + head_idx * head_stride;

      const vec_t cos_vec =
          *reinterpret_cast<const vec_t*>(cos_ptr + dim_idx);
      const vec_t sin_vec =
          *reinterpret_cast<const vec_t*>(sin_ptr + dim_idx);

      // NeoX: x-half and y-half are embed_dim apart.
      vec_t q1 =
          *reinterpret_cast<const vec_t*>(&query[q_offset + dim_idx]);
      vec_t q2 = *reinterpret_cast<const vec_t*>(
          &query[q_offset + embed_dim + dim_idx]);
      vec_t out_q1;
      vec_t out_q2;
      for (int v = 0; v < VEC_SIZE; ++v) {
        out_q1[v] = q1[v] * cos_vec[v] - q2[v] * sin_vec[v];
        out_q2[v] = q2[v] * cos_vec[v] + q1[v] * sin_vec[v];
      }
      *reinterpret_cast<vec_t*>(&query[q_offset + dim_idx]) = out_q1;
      *reinterpret_cast<vec_t*>(
          &query[q_offset + embed_dim + dim_idx]) = out_q2;
    }

    // Process key: same pattern with num_kv_heads.
    if (key != nullptr) {
      const int total_k_vecs = num_kv_heads * num_vec_dims;
      for (int i = local_id; i < total_k_vecs; i += block_size) {
        const int head_idx = i / num_vec_dims;
        const int vec_idx = i % num_vec_dims;
        const int dim_idx = vec_idx * VEC_SIZE;

        const int64_t k_offset =
            token_idx * key_stride + head_idx * head_stride;

        const vec_t cos_vec =
            *reinterpret_cast<const vec_t*>(cos_ptr + dim_idx);
        const vec_t sin_vec =
            *reinterpret_cast<const vec_t*>(sin_ptr + dim_idx);

        vec_t k1 =
            *reinterpret_cast<const vec_t*>(&key[k_offset + dim_idx]);
        vec_t k2 = *reinterpret_cast<const vec_t*>(
            &key[k_offset + embed_dim + dim_idx]);
        vec_t out_k1;
        vec_t out_k2;
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
  const int head_size;
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
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(
      1, 1, std::min<int64_t>(num_heads * num_vec_dims, 512));

  at::DeviceGuard device_guard(query.device());
  auto& queue = vllm::xpu::vllmGetQueue();
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
            head_size));
  });
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