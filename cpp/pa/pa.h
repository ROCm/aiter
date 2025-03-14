#pragma once

#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <torch/all.h>

void paged_attention_rocm_torch(
    torch::Tensor& out,  // [num_seqs, num_heads, head_size]
    torch::Tensor& workspace_buffer,
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, block_size, head_size] or
                    // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, block_size, head_size] or
                      // [num_blocks, block_size, num_heads, head_size]
    double scale,
    torch::Tensor& kv_indptr,                         // [num_seqs + 1]
    torch::Tensor& kv_page_indices,                   // [max_num_blocks]
    std::optional<torch::Tensor>& kv_last_page_lens,  // [num_seqs]
    int64_t block_size, int64_t max_num_partitions,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, const std::string& kv_cache_layout,
    float logits_soft_cap, torch::Tensor& k_scale, torch::Tensor& v_scale,
    const c10::optional<torch::Tensor>& fp8_out_scale);

void paged_attention_rocm(
    int num_seqs, int num_kv_heads, int num_heads, int max_num_partitions,
    int q_stride, int kv_block_stride, int kv_head_stride, int kv_seq_stride,
    int gqa_ratio, int head_size, std::string dtype, std::string kv_dtype,
    std::string kv_cache_dtype, std::string out_dtype, int block_size,
    std::string alibi_enabled, void* query_ptr, void* key_cache_ptr,
    void* value_cache_ptr, void* workspace_buffer_ptr, int* kv_indptr_ptr,
    int* kv_page_indices_ptr, int* kv_last_page_lens_ptr,
    const float* k_scale_ptr, const float* v_scale_ptr,
    const float* fp8_out_scale_ptr, void* out_ptr,
    const float* alibi_slopes_ptr, float logits_soft_cap, double scale,
    const void* stream);