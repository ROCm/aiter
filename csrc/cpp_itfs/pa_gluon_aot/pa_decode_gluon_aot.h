#pragma once

#include <torch/all.h>
#include <optional>

namespace aiter {

void pa_decode_gluon_aot(
    torch::Tensor& output,                 // [num_seqs * query_length, num_query_heads, head_size]
    torch::Tensor& output_gluon,           // [num_seqs, num_kv_heads * query_length * query_group_size, head_size]
    torch::Tensor& query,                  // [num_seqs * query_length, num_query_heads, head_size]
    torch::Tensor& query_gluon,            // [num_seqs, num_kv_heads * query_length * query_group_size, head_size]
    torch::Tensor& query_scale_gluon,      // [num_seqs, num_kv_heads * query_length * query_group_size, 1] or [1]
    torch::Tensor& key_cache,              // [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    torch::Tensor& value_cache,            // [num_blocks, num_kv_heads, head_size, kv_block_size] or [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
    torch::Tensor& context_lengths,        // [num_seqs]
    torch::Tensor& block_tables,           // [num_seqs, max_num_blocks_per_seq]

    float softmax_scale,
    int query_length,
    int max_context_length,
    int context_partition_size,

    std::string compute_type,
    torch::Tensor& query_scale,            // [num_seqs * query_length, num_query_heads, 1] or [1]
    torch::Tensor& key_scale,              // [num_blocks, num_kv_heads, kv_block_size, 1]
    torch::Tensor& value_scale,            // [num_blocks, num_kv_heads, kv_block_size, 1]
    torch::Tensor& exp_sums,               // [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
    torch::Tensor& max_logits,             // [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
    torch::Tensor& temporary_output,       // [num_seqs, num_kv_heads, max_context_partition_num, query_group_size, head_size]

    std::optional<torch::Tensor> alibi_slopes = std::nullopt
);

} // namespace aiter
