#pragma once

#include <hip/hip_runtime.h>
#include <optional>

namespace aiter {
void paged_attention_ragged(std::optional<std::string> folder,
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
                            const float scale,
                            const int num_seqs,
                            const int num_kv_heads,
                            const int num_heads,
                            const int max_num_partitions,
                            const int head_size,
                            const int block_size,
                            const float logits_soft_cap,
                            const int q_stride,
                            const int kv_block_stride,
                            const int kv_head_stride,
                            const int kv_seq_stride,
                            const std::string dtype,
                            const std::string kv_dtype,
                            const std::string kv_cache_dtype,
                            const std::string out_dtype,
                            const hipStream_t stream);
}
