// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "common/arithmetic.hpp"

namespace aiter::mxfp4_moe::moe_3stage_sort {

template <int NE, int TOPK, int N_SORT_CTAS, int THREADS_PER_CTA>
__global__ void sort_count_kernel_impl(
    int M,
    const int32_t *__restrict__ topk_ids,
    int32_t *__restrict__ block_offsets) {

    __shared__ int local_count[NE];

    const int tid = threadIdx.x;
    const int cta = blockIdx.x;
    const int total_pairs = M * TOPK;

    #pragma unroll
    for (int i = tid; i < NE; i += THREADS_PER_CTA)
        local_count[i] = 0;
    __syncthreads();

    const int per_cta = ceil_div(total_pairs, N_SORT_CTAS);
    const int start   = cta * per_cta;
    const int end     = min(start + per_cta, total_pairs);

    for (int i = start + tid; i < end; i += THREADS_PER_CTA) {
        atomicAdd(&local_count[topk_ids[i]], 1);
    }
    __syncthreads();

    #pragma unroll
    for (int e = tid; e < NE; e += THREADS_PER_CTA)
        block_offsets[e * N_SORT_CTAS + cta] = local_count[e];
}

template <int NE, int N_SORT_CTAS, int MB, int THREADS_PER_CTA>
__global__ void sort_cumsum_kernel_impl(
    int M,
    int32_t *__restrict__ block_offsets,
    int32_t *__restrict__ real_counts,
    int32_t *__restrict__ cumsum_tensor,
    int32_t *__restrict__ sorted_expert_ids) {

    static_assert(NE > 0, "NE must be positive");
    static_assert(THREADS_PER_CTA >= 64, "need at least one wave for the scan");

    __shared__ int total_count[NE];
    __shared__ int padded_count[NE];
    __shared__ int expert_starts[NE + 1];

    const int tid = threadIdx.x;

    for (int e = tid; e < NE; e += THREADS_PER_CTA) {
        int sum = 0;
        #pragma unroll
        for (int c = 0; c < N_SORT_CTAS; ++c)
            sum += block_offsets[e * N_SORT_CTAS + c];
        total_count[e]     = sum;
        padded_count[e]    = round_up(sum, MB);
        real_counts[e]     = sum;
    }
    __syncthreads();

    if (tid == 0) {
        int acc = 0;
        for (int e = 0; e < NE; ++e) { expert_starts[e] = acc; acc += padded_count[e]; }
        expert_starts[NE] = acc;
        cumsum_tensor[0] = acc;
        cumsum_tensor[1] = M;  // num_valid_ids[1] = valid tokens (non-EP == M)
    }
    __syncthreads();

    for (int e = tid; e < NE; e += THREADS_PER_CTA) {
        int acc = expert_starts[e];
        #pragma unroll
        for (int c = 0; c < N_SORT_CTAS; ++c) {
            int cnt = block_offsets[e * N_SORT_CTAS + c];
            block_offsets[e * N_SORT_CTAS + c] = acc;
            acc += cnt;
        }
    }

    for (int e = tid; e < NE; e += THREADS_PER_CTA) {
        int b0 = expert_starts[e]     / MB;
        int b1 = expert_starts[e + 1] / MB;
        for (int b = b0; b < b1; ++b)
            sorted_expert_ids[b] = e;
    }
}

template <int NE, int TOPK, int N_SORT_CTAS, int MB, int THREADS_PER_CTA>
__global__ void sort_place_pad_kernel_impl(
    int M,
    const int32_t *__restrict__ topk_ids,
    const float   *__restrict__ topk_weight,
    const int32_t *__restrict__ block_offsets,
    const int32_t *__restrict__ real_counts,
    const int32_t *__restrict__ cumsum_tensor,
    int32_t *__restrict__ sorted_token_ids,
    int32_t *__restrict__ reverse_sorted,
    float   *__restrict__ sorted_weights,
    int32_t *__restrict__ m_indices) {

    __shared__ int local_offsets[NE];
    __shared__ int row_starts[NE + 1];

    const int tid = threadIdx.x;
    const int cta = blockIdx.x;
    const int total_pairs = M * TOPK;

    for (int e = tid; e < NE; e += THREADS_PER_CTA) {
        local_offsets[e] = block_offsets[e * N_SORT_CTAS + cta];
        row_starts[e]    = block_offsets[e * N_SORT_CTAS];
    }
    if (tid == 0) row_starts[NE] = cumsum_tensor[0];
    __syncthreads();

    const int per_cta = ceil_div(total_pairs, N_SORT_CTAS);
    const int start   = cta * per_cta;
    const int end     = min(start + per_cta, total_pairs);

    for (int i = start + tid; i < end; i += THREADS_PER_CTA) {
        int eid = topk_ids[i];
        int sp  = atomicAdd(&local_offsets[eid], 1);
        int token_id = i / TOPK;
        int topk_id  = i % TOPK;
        sorted_token_ids[sp] = (token_id & 0x00FFFFFF) | ((topk_id & 0xFF) << 24);
        sorted_weights[sp]   = topk_weight[i];
        m_indices[sp]        = token_id & 0x00FFFFFF;
        reverse_sorted[i]    = sp;
    }

    __syncthreads();

    const int experts_per_cta = ceil_div(NE, N_SORT_CTAS);
    const int e_lo = cta * experts_per_cta;
    const int e_hi = min(e_lo + experts_per_cta, NE);
    // pad m_indices = M makes buffer_load voff exceed A_q's rsrc extent ⇒ HW
    // drops the load; sorted_token_ids gets the same value.
    const int pad_val = M & 0x00FFFFFF;

    for (int e = e_lo; e < e_hi; ++e) {
        int real_end   = row_starts[e] + real_counts[e];
        int padded_end = row_starts[e + 1];
        for (int j = real_end + tid; j < padded_end; j += THREADS_PER_CTA) {
            sorted_token_ids[j] = pad_val;
            m_indices[j]        = pad_val;
            sorted_weights[j]   = 0.0f;
        }
    }
}

template <int NE, int TOPK, int MB, int N_SORT_CTAS, int THREADS_PER_CTA>
inline void launch(
    hipStream_t stream, int M,
    const int32_t *topk_ids, const float *topk_weight,
    int32_t *sorted_token_ids, int32_t *sorted_expert_ids, int32_t *cumsum_tensor,
    int32_t *reverse_sorted, float *sorted_weights,
    int32_t *m_indices,
    int32_t *block_offsets,
    int32_t *real_counts)
{
    sort_count_kernel_impl<NE, TOPK, N_SORT_CTAS, THREADS_PER_CTA>
        <<<N_SORT_CTAS, THREADS_PER_CTA, 0, stream>>>(
            M, topk_ids, block_offsets);

    sort_cumsum_kernel_impl<NE, N_SORT_CTAS, MB, THREADS_PER_CTA>
        <<<1, THREADS_PER_CTA, 0, stream>>>(
            M, block_offsets, real_counts, cumsum_tensor, sorted_expert_ids);

    sort_place_pad_kernel_impl<NE, TOPK, N_SORT_CTAS, MB, THREADS_PER_CTA>
        <<<N_SORT_CTAS, THREADS_PER_CTA, 0, stream>>>(
            M, topk_ids, topk_weight, block_offsets,
            real_counts, cumsum_tensor,
            sorted_token_ids, reverse_sorted, sorted_weights, m_indices);
}

} // namespace aiter::mxfp4_moe::moe_3stage_sort
