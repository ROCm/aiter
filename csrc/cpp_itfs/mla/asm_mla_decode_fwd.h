#pragma once

#include <optional>
#include <hip/hip_runtime.h>

namespace aiter {
/**
 * @brief Performs forward pass decoding for Multi-head Latent Attention (MLA) using
 * assembly-optimized implementation
 *
 * This function executes the forward pass of MLA decoding with paged key-value cache support.
 * It dynamically compiles and runs optimized assembly code for the specific configuration.
 *
 * @param folder Optional folder name for the compiled kernel. If not provided, uses auto-generated
 * name
 * @param q Pointer to query tensor [num_seqs, num_heads, head_size]
 * @param kv_buffer Pointer to key-value cache buffer [num_page, page_size, num_kv_heads, head_size]
 * @param qo_indptr Pointer to query tensor indices [batch_size+1]
 * @param kv_indptr Pointer to KV cache indices [batch_size+1]
 * @param kv_page_indices Pointer to used page indices [num_page_used]
 * @param kv_last_page_lens Pointer to last page lengths [batch_size]
 * @param max_seqlen_q Maximum sequence length of query
 * @param softmax_scale Scaling factor for softmax computation
 * @param logits Output pointer for attention logits [batch_size, num_kv_splits, num_heads,
 * v_head_dim]
 * @param attn_lse Output pointer for attention log-sum-exp [batch_size, num_kv_splits, num_heads,
 * 1]
 * @param output Output pointer for final results
 * @param num_seqs Number of sequences in the batch
 * @param num_heads Number of attention heads
 * @param num_kv_heads Number of key-value heads
 * @param q_stride_0 Stride for first dimension of query tensor
 * @param kv_buffer_stride_0 Stride for first dimension of KV buffer
 * @param attn_lse_stride_0 First dimension stride for attention LSE tensor
 * @param attn_lse_stride_1 Second dimension stride for attention LSE tensor
 * @param attn_lse_stride_2 Third dimension stride for attention LSE tensor
 * @param output_stride_0 First dimension stride for output tensor
 * @param output_stride_1 Second dimension stride for output tensor
 * @param page_size Size of each page in the KV cache
 * @param q_dtype Data type string for query tensor
 * @param kv_dtype Data type string for key-value tensor
 * @param num_kv_splits Number of key-value splits
 * @param v_head_dim Dimension of value head
 * @param stream HIP stream for GPU execution
 *
 * @note Requires HIP runtime; compiles the kernel on first use with the given parameters.
 * @throws May throw if kernel compilation fails or GPU execution errors.
 * @warning All input/output pointers must be allocated and aligned; compilation needs write access.
 */
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
