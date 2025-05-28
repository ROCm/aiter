#pragma once

#include <hip/hip_runtime.h>
#include <optional>

namespace aiter {
void paged_attention_ragged(std::optional<std::string> folder,
                            int num_seqs,
                            int num_kv_heads,
                            int num_heads,
                            int max_num_partitions,
                            int q_stride,
                            int kv_block_stride,
                            int kv_head_stride,
                            int kv_seq_stride,
                            int gqa_ratio,
                            int head_size,
                            std::string dtype,
                            std::string kv_dtype,
                            std::string kv_cache_dtype,
                            std::string out_dtype,
                            int block_size,
                            std::string alibi_enabled,
                            void* query_ptr,
                            void* key_cache_ptr,
                            void* value_cache_ptr,
                            void* workspace_buffer_ptr,
                            int* kv_indptr_ptr,
                            int* kv_page_indices_ptr,
                            int* kv_last_page_lens_ptr,
                            const float* k_scale_ptr,
                            const float* v_scale_ptr,
                            const float* fp8_out_scale_ptr,
                            void* out_ptr,
                            const float* alibi_slopes_ptr,
                            float logits_soft_cap,
                            double scale,
                            const hipStream_t stream);
}
