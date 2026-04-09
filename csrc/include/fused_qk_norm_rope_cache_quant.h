#pragma once

#include "aiter_tensor.h"
#include <optional>
#include <string>

namespace aiter {

void fused_qk_norm_rope_cache_quant_shuffle(
    const aiter_tensor_t& qkv,                   // Combined QKV tensor [num_tokens,
                                       // (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int64_t num_heads_q,               // Number of query heads
    int64_t num_heads_k,               // Number of key heads
    int64_t num_heads_v,               // Number of value heads
    int64_t head_dim,                  // Dimension per head
    double eps,                        // Epsilon for RMS normalization
    const aiter_tensor_t& q_weight,              // RMSNorm weights for query [head_dim]
    const aiter_tensor_t& k_weight,              // RMSNorm weights for key [head_dim]
    const aiter_tensor_t& cos_sin_cache,         // Cos/sin cache [max_position, head_dim]
    bool is_neox,                      // Whether RoPE is applied in Neox style
    const aiter_tensor_t& position_ids,          // Position IDs for RoPE [num_tokens]
    const aiter_tensor_t& k_cache,               // [num_blocks, num_kv_heads, head_dim//x, page_size, x]
    const aiter_tensor_t& v_cache,               // 4D [num_blocks, num_heads_v, head_dim, page_size] or 5D shuffle
                                       // [num_blocks, num_heads_v, page_size//x, head_dim, x]
    const aiter_tensor_t& slot_mapping,          // slot mapping
    const std::string& kv_cache_dtype, // kv cache data type
    std::optional<aiter_tensor_t> k_scale, // k scale tensor for quantized k cache
    std::optional<aiter_tensor_t> v_scale  // v scale tensor for quantized v cache
);

void fused_qk_norm_rope_cache_pts_quant_shuffle(const aiter_tensor_t& qkv,
                                                const aiter_tensor_t& qw,
                                                const aiter_tensor_t& kw,
                                                const aiter_tensor_t& cos_sin,
                                                const aiter_tensor_t& positions,
                                                int64_t num_tokens,
                                                int64_t num_heads_q,
                                                int64_t num_heads_k,
                                                int64_t num_heads_v,
                                                int64_t head_size,
                                                bool is_neox_style,
                                                double eps,
                                                const aiter_tensor_t& q_out,
                                                const aiter_tensor_t& k_cache,
                                                const aiter_tensor_t& v_cache,
                                                const aiter_tensor_t& slot_mapping,
                                                const aiter_tensor_t& per_tensor_k_scale,
                                                const aiter_tensor_t& per_tensor_v_scale,
                                                std::optional<aiter_tensor_t> k_out,
                                                std::optional<aiter_tensor_t> v_out,
                                                bool return_kv,
                                                bool use_shuffle_layout,
                                                int64_t block_size,
                                                int64_t x,
                                                int64_t rotary_dim = 0);

void fused_qk_norm_rope_2way(const aiter_tensor_t& q0,
                             const aiter_tensor_t& k0,
                             const aiter_tensor_t& q1,
                             const aiter_tensor_t& k1,
                             const aiter_tensor_t& w_q0,
                             const aiter_tensor_t& w_k0,
                             const aiter_tensor_t& w_q1,
                             const aiter_tensor_t& w_k1,
                             const aiter_tensor_t& cos_sin0,
                             const aiter_tensor_t& cos_sin1,
                             int64_t batch_size,
                             int64_t num_tokens0,
                             int64_t num_tokens1,
                             int64_t num_heads_q,
                             int64_t num_heads_k,
                             int64_t head_size,
                             bool is_interleaved,
                             double eps,
                             const aiter_tensor_t& out_q01,
                             const aiter_tensor_t& out_k01);

void fused_qk_rmsnorm(const aiter_tensor_t& q,
                      const aiter_tensor_t& q_weight,
                      double q_eps,
                      const aiter_tensor_t& k,
                      const aiter_tensor_t& k_weight,
                      double k_eps,
                      const aiter_tensor_t& q_out,
                      const aiter_tensor_t& k_out);

void fused_qk_norm_rope_cache_block_quant_shuffle(
    const aiter_tensor_t& qkv,                   // Combined QKV tensor [num_tokens,
                                       // (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int64_t num_heads_q,               // Number of query heads
    int64_t num_heads_k,               // Number of key heads
    int64_t num_heads_v,               // Number of value heads
    int64_t head_dim,                  // Dimension per head
    double eps,                        // Epsilon for RMS normalization
    const aiter_tensor_t& q_weight,              // RMSNorm weights for query [head_dim]
    const aiter_tensor_t& k_weight,              // RMSNorm weights for key [head_dim]
    const aiter_tensor_t& cos_sin_cache,         // Cos/sin cache [max_position, head_dim]
    bool is_neox,                      // Whether RoPE is applied in Neox style
    const aiter_tensor_t& position_ids,          // Position IDs for RoPE [num_tokens]
    const aiter_tensor_t& k_cache,               // k cache
    const aiter_tensor_t& v_cache,               // v cache
    const aiter_tensor_t& slot_mapping,          // slot mapping
    const aiter_tensor_t& cu_q_len,              // cu q len tensor
    const std::string& kv_cache_dtype, // kv cache data type
    std::optional<aiter_tensor_t> k_scale, // k scale tensor for quantized k cache
    std::optional<aiter_tensor_t> v_scale, // v scale tensor for quantized v cache
    int64_t max_tokens_per_batch = 0   // max tokens in any single batch (0 = use avg)
);

} // namespace aiter
