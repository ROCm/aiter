// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include "attention_ragged.h"
#include "attention_common.cuh"

#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
#define __HIP__MI300_MI250__
#endif

#if defined(__HIP__MI300_MI250__) // TODO: Add NAVI support

///////////////////////////////////////
// grid (num_seqs, num_partitions,num_kv_heads)
// block (256)
template <typename scalar_t,
          typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE,
          typename OUTT,
          int BLOCK_SIZE,
          int HEAD_SIZE,
          int NUM_THREADS,
          bool ALIBI_ENABLED,
          int GQA_RATIO,
          typename AttentionVariant>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma16_kernel(
    const scalar_t* __restrict__ q,      // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache, // [num_blocks, block_size, num_kv_heads,
                                         // head_size]
    const cache_t* __restrict__ v_cache, // [num_blocks, block_size, num_kv_heads,
                                         // head_size]
    const float scale,
    const int* __restrict__ kv_indptr,         // [num_seqs + 1]
    const int* __restrict__ kv_page_indices,   // [max_num_blocks]
    const int* __restrict__ kv_last_page_lens, // [num_seqs]
    const float* __restrict__ alibi_slopes,    // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const int kv_seq_stride,
    float* __restrict__ exp_sums,   // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits, // [num_seqs, num_heads,
                                    // max_num_partitions]
    scalar_t* __restrict__ out,     // [num_seqs, num_heads, max_num_partitions,
                                    // head_size]
    OUTT* __restrict__ final_out,   // [num_seqs, num_heads, head_size]
    float logits_soft_cap,
    float logits_soft_cap_rcp,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    const AttentionVariant* variant)
{
    const int seq_idx       = blockIdx.x;
    const int partition_idx = blockIdx.y;

    constexpr int T_PAR_SIZE = 256; // token partition size set to 256

    int context_len;
    if constexpr(BLOCK_SIZE > 1)
    {
        context_len = (kv_indptr[seq_idx + 1] - kv_indptr[seq_idx] - 1) * BLOCK_SIZE +
                      kv_last_page_lens[seq_idx];
    }
    else
    {
        context_len = kv_indptr[seq_idx + 1] - kv_indptr[seq_idx];
    }

    const int partition_start_token_idx = partition_idx * T_PAR_SIZE; // partition_size;
    // exit if partition is out of context for seq
    if(partition_start_token_idx >= context_len)
    {
        return;
    }
    const int64_t query_loc = static_cast<int64_t>(seq_idx);
    const int* block_table_seq = kv_page_indices + kv_indptr[seq_idx];
    _paged_attention_kernel<scalar_t, cache_t, KV_DTYPE, OUTT, BLOCK_SIZE, HEAD_SIZE, NUM_THREADS, ALIBI_ENABLED, GQA_RATIO, AttentionVariant>(block_table_seq, query_loc, context_len, partition_start_token_idx, q, k_cache, v_cache, scale, alibi_slopes, q_stride, kv_block_stride, kv_head_stride, kv_seq_stride, exp_sums, max_logits, out, final_out, logits_soft_cap, logits_soft_cap_rcp, k_scale_ptr, v_scale_ptr, variant);
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t,
          typename OUTT,
          int HEAD_SIZE,
          int NUM_THREADS,
          int PARTITION_SIZE,
          int NPAR_LOOPS,
          bool ENABLE_LAST_PAGE_LENS>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    OUTT* __restrict__ out,                    // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,        // [num_seqs, num_heads,
                                               // max_num_partitions]
    const float* __restrict__ max_logits,      // [num_seqs, num_heads,
                                               // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,      // [num_seqs, num_heads,
                                               // max_num_partitions, head_size]
    const int* __restrict__ kv_indptr,         // [num_seqs + 1]
    const int* __restrict__ kv_last_page_lens, // [num_seqs]
    const int block_size,
    const int max_num_partitions,
    const float* __restrict__ fp8_out_scale_ptr)
{
    const int num_heads = gridDim.x;
    const int head_idx  = blockIdx.x;
    const int seq_idx   = blockIdx.y;
    int context_len;
    if constexpr(ENABLE_LAST_PAGE_LENS)
    {
        context_len = (kv_indptr[seq_idx + 1] - kv_indptr[seq_idx] - 1) * block_size +
                      kv_last_page_lens[seq_idx];
    }
    else
    {
        context_len = kv_indptr[seq_idx + 1] - kv_indptr[seq_idx];
    }
    const int64_t query_loc = static_cast<int64_t>(seq_idx);
    _paged_attention_ll4mi_reduce_kernel<scalar_t, OUTT, HEAD_SIZE, NUM_THREADS, PARTITION_SIZE, NPAR_LOOPS>(query_loc, context_len, out, exp_sums, max_logits, tmp_out, max_num_partitions, fp8_out_scale_ptr);
}

#else // !defined(__HIP__MI300_MI250__) TODO: Add NAVI support


template <typename scalar_t,
          typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE,
          typename OUTT,
          int BLOCK_SIZE,
          int HEAD_SIZE,
          int NUM_THREADS,
          bool ALIBI_ENABLED,
          int GQA_RATIO,
          typename AttentionVariant>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma16_kernel(
    const scalar_t* __restrict__ q,      // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache, // [num_blocks, num_kv_heads,
                                         // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache, // [num_blocks, num_kv_heads,
                                         // head_size, block_size]
    const float scale,
    const int* __restrict__ kv_indptr,         // [num_seqs + 1]
    const int* __restrict__ kv_page_indices,   // [max_num_blocks]
    const int* __restrict__ kv_last_page_lens, // [num_seqs]
    const float* __restrict__ alibi_slopes,    // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const int kv_seq_stride,
    float* __restrict__ exp_sums,   // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits, // [num_seqs, num_heads,
                                    // max_num_partitions]
    scalar_t* __restrict__ out,     // [num_seqs, num_heads, max_num_partitions,
                                    // head_size]
    OUTT* __restrict__ final_out,   // [num_seqs, num_heads, head_size]
    float logits_soft_cap,
    float logits_soft_cap_rcp,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    const AttentionVariant* variant)
{
    UNREACHABLE_CODE
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t,
          typename OUTT,
          int HEAD_SIZE,
          int NUM_THREADS,
          int PARTITION_SIZE,
          int NPAR_LOOPS,
          bool ENABLE_LAST_PAGE_LENS>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    OUTT* __restrict__ out,                    // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,        // [num_seqs, num_heads,
                                               // max_num_partitions]
    const float* __restrict__ max_logits,      // [num_seqs, num_heads,
                                               // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,      // [num_seqs, num_heads,
                                               // max_num_partitions, head_size]
    const int* __restrict__ kv_indptr,         // [num_seqs + 1]
    const int* __restrict__ kv_last_page_lens, // [num_seqs]
    const int block_size,
    const int max_num_partitions,
    const float* __restrict__ fp8_out_scale_ptr)
{
    UNREACHABLE_CODE
}

#endif // defined(__HIP__MI300_MI250__) TODO: Add NAVI support

#define LAUNCH_CUSTOM_ATTENTION_MFMA16(GQA_RATIO)           \
    paged_attention_ll4mi_QKV_mfma16_kernel<T,              \
                                            KVT,            \
                                            KV_DTYPE,       \
                                            OUTT,           \
                                            BLOCK_SIZE,     \
                                            HEAD_SIZE,      \
                                            NTHR,           \
                                            ALIBI_ENABLED,  \
                                            GQA_RATIO>      \
        <<<grid, block, 0, stream>>>(query_ptr,             \
                                     key_cache_ptr,         \
                                     value_cache_ptr,       \
                                     scale,                 \
                                     kv_indptr_ptr,         \
                                     kv_page_indices_ptr,   \
                                     kv_last_page_lens_ptr, \
                                     alibi_slopes_ptr,      \
                                     q_stride,              \
                                     kv_block_stride,       \
                                     kv_head_stride,        \
                                     kv_seq_stride,         \
                                     exp_sums_ptr,          \
                                     max_logits_ptr,        \
                                     tmp_out_ptr,           \
                                     out_ptr,               \
                                     logits_soft_cap,       \
                                     logits_soft_cap_rcp,   \
                                     k_scale_ptr,           \
                                     v_scale_ptr,           \
                                     &variant);

#define LAUNCH_CUSTOM_REDUCTION(NPAR_LOOPS)                               \
    paged_attention_ll4mi_reduce_kernel<T,                                \
                                        OUTT,                             \
                                        HEAD_SIZE,                        \
                                        HEAD_SIZE,                        \
                                        PARTITION_SIZE,                   \
                                        NPAR_LOOPS,                       \
                                        ENABLE_LAST_PAGE_LENS>            \
        <<<reduce_grid, reduce_block, 0, stream>>>(out_ptr,               \
                                                   exp_sums_ptr,          \
                                                   max_logits_ptr,        \
                                                   tmp_out_ptr,           \
                                                   kv_indptr_ptr,         \
                                                   kv_last_page_lens_ptr, \
                                                   BLOCK_SIZE,            \
                                                   max_num_partitions,    \
                                                   fp8_out_scale_ptr);


template <typename T,
          typename KVT,
          vllm::Fp8KVCacheDataType KV_DTYPE,
          int BLOCK_SIZE,
          int HEAD_SIZE,
          typename OUTT,
          int PARTITION_SIZE_OLD,
          bool ALIBI_ENABLED,
          bool LOGITS_SOFT_CAP_ENABLED>
void paged_attention_custom_launcher(torch::Tensor& out,
                                     torch::Tensor& workspace_buffer,
                                     torch::Tensor& query,
                                     torch::Tensor& key_cache,
                                     torch::Tensor& value_cache,
                                     float scale,
                                     torch::Tensor& kv_indptr,
                                     torch::Tensor& kv_page_indices,
                                     std::optional<torch::Tensor>& kv_last_page_lens,
                                     int max_num_partitions,
                                     const std::optional<torch::Tensor>& alibi_slopes,
                                     const std::string& kv_cache_layout,
                                     float logits_soft_cap,
                                     torch::Tensor& k_scale,
                                     torch::Tensor& v_scale,
                                     const std::optional<torch::Tensor>& fp8_out_scale)
{
    const int num_kv_heads = kv_cache_layout == "HND" ? key_cache.size(1) : key_cache.size(2);
    int num_seqs           = query.size(0);
    int num_heads          = query.size(1);
    int head_size          = query.size(2);
    int q_stride           = query.stride(0);
    int kv_block_stride    = key_cache.stride(0);
    int kv_head_stride     = kv_cache_layout == "HND" ? key_cache.stride(1) : key_cache.stride(2);
    int kv_seq_stride      = kv_cache_layout == "HND" ? key_cache.stride(2) : key_cache.stride(1);

    // NOTE: alibi_slopes is optional.
    const float* alibi_slopes_ptr =
        alibi_slopes ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr()) : nullptr;

    T* query_ptr             = reinterpret_cast<T*>(query.data_ptr());
    KVT* key_cache_ptr       = reinterpret_cast<KVT*>(key_cache.data_ptr());
    KVT* value_cache_ptr     = reinterpret_cast<KVT*>(value_cache.data_ptr());
    int* kv_indptr_ptr       = kv_indptr.data_ptr<int>();
    int* kv_page_indices_ptr = kv_page_indices.data_ptr<int>();
    int* kv_last_page_lens_ptr =
        BLOCK_SIZE > 1 ? kv_last_page_lens.value().data_ptr<int>() : nullptr;

    const float* k_scale_ptr = reinterpret_cast<const float*>(k_scale.data_ptr());
    const float* v_scale_ptr = reinterpret_cast<const float*>(v_scale.data_ptr());
    // NOTE: fp8_out_scale is optional.
    const float* fp8_out_scale_ptr =
        fp8_out_scale ? reinterpret_cast<const float*>(fp8_out_scale.value().data_ptr()) : nullptr;
    OUTT* out_ptr = reinterpret_cast<OUTT*>(out.data_ptr());

    const float logits_soft_cap_rcp = (LOGITS_SOFT_CAP_ENABLED ? 1.f / logits_soft_cap : 0.f);

    // partition size is fixed at 256 since both mfma4 and mfma16 kernels support it
    // mfma4 kernel also supports partition size 512
    constexpr int PARTITION_SIZE = 256;
    const int gqa_ratio          = num_heads / num_kv_heads;
    assert(num_heads % num_kv_heads == 0);
    assert(head_size == HEAD_SIZE);

    // split workspace into 3 intermediate tensors
    float* exp_sums_ptr   = reinterpret_cast<float*>(workspace_buffer.data_ptr());
    float* max_logits_ptr = exp_sums_ptr + (num_seqs * num_heads * max_num_partitions);
    T* tmp_out_ptr =
        reinterpret_cast<T*>(max_logits_ptr + (num_seqs * num_heads * max_num_partitions));

    ck_tile::ComposedAttention<LOGITS_SOFT_CAP_ENABLED * ck_tile::LOGITS_SOFT_CAP> variant;

    constexpr int NTHR = 256;
    dim3 grid(num_seqs, max_num_partitions, num_kv_heads);
    dim3 block(NTHR);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(query));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    // mfma4 kernel is faster than mfma16 for gqa_ratio <= 4
    switch(gqa_ratio)
    {
    case 1: LAUNCH_CUSTOM_ATTENTION_MFMA16(1); break;
    case 2: LAUNCH_CUSTOM_ATTENTION_MFMA16(2); break;
    case 3: LAUNCH_CUSTOM_ATTENTION_MFMA16(3); break;
    case 4: LAUNCH_CUSTOM_ATTENTION_MFMA16(4); break;
    case 5: LAUNCH_CUSTOM_ATTENTION_MFMA16(5); break;
    case 6: LAUNCH_CUSTOM_ATTENTION_MFMA16(6); break;
    case 7: LAUNCH_CUSTOM_ATTENTION_MFMA16(7); break;
    case 8: LAUNCH_CUSTOM_ATTENTION_MFMA16(8); break;
    case 9: LAUNCH_CUSTOM_ATTENTION_MFMA16(9); break;
    case 10: LAUNCH_CUSTOM_ATTENTION_MFMA16(10); break;
    case 11: LAUNCH_CUSTOM_ATTENTION_MFMA16(11); break;
    case 12: LAUNCH_CUSTOM_ATTENTION_MFMA16(12); break;
    case 13: LAUNCH_CUSTOM_ATTENTION_MFMA16(13); break;
    case 14: LAUNCH_CUSTOM_ATTENTION_MFMA16(14); break;
    case 15: LAUNCH_CUSTOM_ATTENTION_MFMA16(15); break;
    case 16: LAUNCH_CUSTOM_ATTENTION_MFMA16(16); break;
    default: TORCH_CHECK(false, "Unsupported gqa ratio: ", gqa_ratio); break;
    }

    dim3 reduce_grid(num_heads, num_seqs);
    dim3 reduce_block(head_size);
    const int npar_loops = DIVIDE_ROUND_UP(max_num_partitions, WARP_SIZE);
    // reduction kernel supports upto 8 NPAR_loops * 64 (warp_size) * 256 (partition size) = 128K
    // context length
    constexpr bool ENABLE_LAST_PAGE_LENS = BLOCK_SIZE > 1;
    switch(npar_loops)
    {
    case 1: LAUNCH_CUSTOM_REDUCTION(1); break;
    case 2: LAUNCH_CUSTOM_REDUCTION(2); break;
    case 3: LAUNCH_CUSTOM_REDUCTION(3); break;
    case 4: LAUNCH_CUSTOM_REDUCTION(4); break;
    case 5: LAUNCH_CUSTOM_REDUCTION(5); break;
    case 6: LAUNCH_CUSTOM_REDUCTION(6); break;
    case 7: LAUNCH_CUSTOM_REDUCTION(7); break;
    case 8: LAUNCH_CUSTOM_REDUCTION(8); break;
    default: TORCH_CHECK(false, "Unsupported npar_loops: ", npar_loops); break;
    }
}
  
#define CALL_CUSTOM_LAUNCHER(                                                                \
    T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, PSIZE, ALIBI_ENABLED, LOGITS_SOFT_CAP_ENABLED) \
    paged_attention_custom_launcher<T,                                                          \
                                    KVT,                                                        \
                                    KV_DTYPE,                                                   \
                                    BLK_SIZE,                                                   \
                                    HEAD_SIZE,                                                  \
                                    OUTT,                                                       \
                                    PSIZE,                                                      \
                                    ALIBI_ENABLED,                                              \
                                    LOGITS_SOFT_CAP_ENABLED>(out,                               \
                                                             workspace_buffer,                  \
                                                             query,                             \
                                                             key_cache,                         \
                                                             value_cache,                       \
                                                             scale,                             \
                                                             kv_indptr,                         \
                                                             kv_page_indices,                   \
                                                             kv_last_page_lens,                 \
                                                             max_num_partitions,                \
                                                             alibi_slopes,                      \
                                                             kv_cache_layout,                   \
                                                             logits_soft_cap,                   \
                                                             k_scale,                           \
                                                             v_scale,                           \
                                                             fp8_out_scale);

#define CALL_CUSTOM_LAUNCHER_SOFT_CAP(                                                 \
    T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, PSIZE, ALIBI_ENABLED)                 \
    if(0.f < logits_soft_cap)                                                          \
    {                                                                                  \
        CALL_CUSTOM_LAUNCHER(                                                          \
            T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, PSIZE, ALIBI_ENABLED, true);  \
    }                                                                                  \
    else if(logits_soft_cap == 0.f)                                                    \
    {                                                                                  \
        CALL_CUSTOM_LAUNCHER(                                                          \
            T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, PSIZE, ALIBI_ENABLED, false); \
    }                                                                                  \
    else                                                                               \
    {                                                                                  \
        TORCH_CHECK(false, "logits_soft_cap must be non-negative");                    \
    }


#define CALL_CUSTOM_LAUNCHER_ALIBI(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, PSIZE)            \
    if(alibi_slopes)                                                                              \
    {                                                                                             \
        CALL_CUSTOM_LAUNCHER_SOFT_CAP(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, PSIZE, true);  \
    }                                                                                             \
    else                                                                                          \
    {                                                                                             \
        CALL_CUSTOM_LAUNCHER_SOFT_CAP(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, PSIZE, false); \
    }

#define CALL_CUSTOM_LAUNCHER_PSIZE(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT)                    \
    switch(partition_size)                                                                         \
    {                                                                                              \
    case 256: CALL_CUSTOM_LAUNCHER_ALIBI(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, 256); break; \
    default: TORCH_CHECK(false, "Unsupported partition size: ", partition_size); break;            \
    }

#if defined(__HIPCC__) && defined(__gfx90a__)
#define CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE)       \
    if(fp8_out_scale)                                                         \
    {                                                                         \
        TORCH_CHECK(false, "fp8 out scale unsupported for gfx90a");           \
    }                                                                         \
    else                                                                      \
    {                                                                         \
        CALL_CUSTOM_LAUNCHER_PSIZE(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, T); \
    }
#else
#define CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE)             \
    if(fp8_out_scale)                                                               \
    {                                                                               \
        CALL_CUSTOM_LAUNCHER_PSIZE(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, uint8_t); \
    }                                                                               \
    else                                                                            \
    {                                                                               \
        CALL_CUSTOM_LAUNCHER_PSIZE(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, T);       \
    }
#endif
#define CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, HEAD_SIZE)                   \
    switch(block_size)                                                          \
    {                                                                           \
    case 1: CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, 1, HEAD_SIZE); break;    \
    case 16: CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, 16, HEAD_SIZE); break;  \
    case 32: CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, 32, HEAD_SIZE); break;  \
    default: TORCH_CHECK(false, "Unsupported block size: ", block_size); break; \
    }

#define CALL_CUSTOM_LAUNCHER_BLK_HEAD(T, KVT, KV_DTYPE)                       \
    switch(head_size)                                                         \
    {                                                                         \
    case 64: CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, 64); break;           \
    case 128: CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, 128); break;         \
    default: TORCH_CHECK(false, "Unsupported head size: ", head_size); break; \
    }

void paged_attention_ragged(
    torch::Tensor& out, // [num_seqs, num_heads, head_size]
    torch::Tensor& workspace_buffer,
    torch::Tensor& query,       // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,   // [num_blocks, num_heads, block_size, head_size] or
                                // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& value_cache, // [num_blocks, num_heads, block_size, head_size] or
                                // [num_blocks, block_size, num_heads, head_size]
    double scale,
    torch::Tensor& kv_indptr,                        // [num_seqs + 1]
    torch::Tensor& kv_page_indices,                  // [max_num_blocks]
    std::optional<torch::Tensor>& kv_last_page_lens, // [num_seqs]
    int64_t block_size,
    int64_t max_num_partitions,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype,
    const std::string& kv_cache_layout,
    float logits_soft_cap,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale,
    const std::optional<torch::Tensor>& fp8_out_scale,
    int64_t partition_size)
{
    const int head_size = query.size(2);
    if(kv_cache_dtype == "auto")
    {
        if(query.dtype() == at::ScalarType::Half)
        {
            CALL_CUSTOM_LAUNCHER_BLK_HEAD(_Float16, _Float16, vllm::Fp8KVCacheDataType::kAuto);
        }
        else if(query.dtype() == at::ScalarType::BFloat16)
        {
            CALL_CUSTOM_LAUNCHER_BLK_HEAD(
                __hip_bfloat16, __hip_bfloat16, vllm::Fp8KVCacheDataType::kAuto);
        }
        else
        {
            TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
        }
    }
    else if(kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3")
    {
        if(query.dtype() == at::ScalarType::Half)
        {
            CALL_CUSTOM_LAUNCHER_BLK_HEAD(_Float16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
        }
        else if(query.dtype() == at::ScalarType::BFloat16)
        {
            CALL_CUSTOM_LAUNCHER_BLK_HEAD(
                __hip_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
        }
        else
        {
            TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
        }
    }
    else
    {
        TORCH_CHECK(false, "Unsupported KV cache dtype: ", kv_cache_dtype);
    }
}

#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
