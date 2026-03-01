#pragma once

#include <torch/extension.h>

using namespace at;

namespace aiter {

void fused_qk_norm_rope_cache_quant_shuffle(
    at::Tensor& qkv,                   // Combined QKV tensor [num_tokens,
                                       // (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int64_t num_heads_q,               // Number of query heads
    int64_t num_heads_k,               // Number of key heads
    int64_t num_heads_v,               // Number of value heads
    int64_t head_dim,                  // Dimension per head
    double eps,                        // Epsilon for RMS normalization
    at::Tensor& q_weight,              // RMSNorm weights for query [head_dim]
    at::Tensor& k_weight,              // RMSNorm weights for key [head_dim]
    at::Tensor& cos_sin_cache,         // Cos/sin cache [max_position, head_dim]
    bool is_neox,                      // Whether RoPE is applied in Neox style
    at::Tensor& position_ids,          // Position IDs for RoPE [num_tokens]
    at::Tensor& k_cache,               // k cache
    at::Tensor& v_cache,               // v cache
    at::Tensor& slot_mapping,          // slot mapping
    const std::string& kv_cache_dtype, // kv cache data type
    std::optional<at::Tensor> k_scale, // k scale tensor for quantized k cache
    std::optional<at::Tensor> v_scale  // v scale tensor for quantized v cache
);

void fused_qk_norm_rope_cache_pts_quant_shuffle(at::Tensor& qkv,
                                                at::Tensor& qw,
                                                at::Tensor& kw,
                                                at::Tensor& cos_sin,
                                                at::Tensor& positions,
                                                int64_t num_tokens,
                                                int64_t num_heads_q,
                                                int64_t num_heads_k,
                                                int64_t num_heads_v,
                                                int64_t head_size,
                                                bool is_neox_style,
                                                double eps,
                                                at::Tensor& q_out,
                                                at::Tensor& k_cache,
                                                at::Tensor& v_cache,
                                                at::Tensor& slot_mapping,
                                                at::Tensor& per_tensor_k_scale,
                                                at::Tensor& per_tensor_v_scale,
                                                std::optional<at::Tensor> k_out,
                                                std::optional<at::Tensor> v_out,
                                                bool return_kv,
                                                bool use_shuffle_layout,
                                                int64_t block_size,
                                                int64_t x);

} // namespace aiter
