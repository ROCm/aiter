#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include "fmla.h"

std::vector<torch::Tensor> get_mla_metadata(
    const torch::Tensor& p_seqlens_k,               // [batch size]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k)
{
    assert(false);

    auto opts = p_seqlens_k.options();
    auto ret  = torch::empty({4}, opts);
    return {ret};
}

std::vector<torch::Tensor> flash_mla_fwd_with_kvcache_impl(
    torch::Tensor&       query,                     // [batch size,  seqlen of q, head count of q,  head dim of qk]
    const torch::Tensor& key_cache,                 // [block count, block size,  head count of kv, head dim of qk]
    const torch::Tensor& value_cache,               // [block count, block size,  head count of kv, head dim of v ]
    const int32_t        head_size_v,
    const torch::Tensor& seqlens_k,                 // [batch size]
    const torch::Tensor& block_table,               // [batch size, max blocks per seq]
    const float          softmax_scale,
    const bool           is_causal,
    const torch::Tensor& tile_scheduler_metadata,   // [num cu parts, metadata size]
    const torch::Tensor& num_splits)                // [batch size + 1]
{
    const int32_t seqlen_q = query.size(1);

    if (seqlen_q < 32)
    {
        return flash_mla_fwd_decode_with_kvcache_impl(
            query, key_cache, value_cache,
            head_size_v, 
            seqlens_k, block_table,
            softmax_scale, is_causal,
            tile_scheduler_metadata, num_splits);
    }
    else
    {
        return flash_mla_fwd_prefill_with_kvcache_impl(
            query, key_cache, value_cache,
            head_size_v, 
            seqlens_k, block_table,
            softmax_scale, is_causal);
    }
}