#pragma once

#include <optional>
#include <hip/hip_runtime.h>

namespace aiter {
void asm_mla_decode_fwd(std::optional<std::string> folder,
                        void* q,                 //   [num_seqs, num_heads, head_size]
                        void* kv_buffer,         //   [num_page, page_size, num_kv_heads, head_size]
                        void* qo_indptr,         //   [batch_size+1]
                        void* kv_indptr,         //   [batch_size+1]
                        void* kv_page_indices,   //   [num_page_used]
                        void* kv_last_page_lens, //   [batch_size]
                        int max_seqlen_q,
                        float softmax_scale,
                        // following are output
                        void* logits,   //[batch_size, num_kv_splits, num_heads, v_head_dim]
                        void* attn_lse, //[batch_size, num_kv_splits, num_heads,  1]
                        void* output,
                        int num_seqs,
                        int num_heads,
                        int num_kv_heads,
                        int q_stride_0,
                        int kv_buffer_stride_0,
                        int attn_lse_stride_0,
                        int attn_lse_stride_1,
                        int attn_lse_stride_2,
                        int output_stride_0,
                        int output_stride_1,
                        const int page_size,
                        const std::string q_dtype,
                        const std::string kv_dtype,
                        const int num_kv_splits,
                        const int v_head_dim,
                        const hipStream_t stream);
}
