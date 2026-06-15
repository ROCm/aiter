// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "common/arithmetic.hpp"

namespace aiter::mxfp4_moe::moe_sort_scales {

constexpr int MASK_TOKEN_ID = 0x00FFFFFF;

template <int BM, int NUM_EXPERTS, int D_INTER, int D_HIDDEN, int BK,
          int N_CTAS, int THREADS_PER_CTA>
__global__ void sort_scales_kernel_impl(
    const int M, const int MAX_SORTED,
    const uint8_t *a_scale, const int32_t *sorted_token_ids, const int32_t *cumsum_tensor,
    uint8_t *a_scale_sorted_shuffled) {

    static_assert(BM == 32 || BM == 64 || BM == 128 || BM == 256,
                  "BM ∈ {32, 64, 128, 256}");
    static_assert(BK == 128 || BK == 256, "BK ∈ {128, 256}");

    constexpr int A_SCALE_COLS = exact_div<D_HIDDEN, 32>();
    constexpr int MN_PACK = 2;
    constexpr int K_PACK  = BK / 128;
    constexpr int C_M1    = BM / (16 * MN_PACK);
    constexpr int C_K1    = (D_HIDDEN / 32) / (4 * K_PACK);
    constexpr int K_LANE  = 4;
    constexpr int N_LANE  = 16;
    constexpr int DWORDS_PER_CHUNK = C_M1 * C_K1 * K_LANE * N_LANE;

    static_assert(C_M1 >= 1, "BM too small for mn_pack=2 (need BM ≥ 32)");

    const int n_chunks       = MAX_SORTED / BM;
    const int actual_sorted  = cumsum_tensor[0];
    const int actual_n_chunks = (actual_sorted + BM - 1) / BM;

    const int total_work = n_chunks * DWORDS_PER_CHUNK;
    constexpr int total_threads = N_CTAS * THREADS_PER_CTA;
    const int global_tid = blockIdx.x * THREADS_PER_CTA + threadIdx.x;

    for (int work_id = global_tid; work_id < total_work; work_id += total_threads) {
        int r = work_id;
        const int n_lane = r % N_LANE;  r /= N_LANE;
        const int k_lane = r % K_LANE;  r /= K_LANE;
        const int ku     = r % C_K1;    r /= C_K1;
        const int mi     = r % C_M1;    r /= C_M1;
        const int chunk  = r;

        uint8_t bytes[4] = {0, 0, 0, 0};
        if (chunk < actual_n_chunks) {
            int tok_ids[MN_PACK];
            #pragma unroll
            for (int im_a = 0; im_a < MN_PACK; im_a++) {
                const int sorted_row = chunk * BM + (mi * MN_PACK + im_a) * 16 + n_lane;
                int tid = 0;
                if (sorted_row < actual_sorted) {
                    const int sti_val = sorted_token_ids[sorted_row] & MASK_TOKEN_ID;
                    tid = (sti_val < M) ? sti_val : 0;
                }
                tok_ids[im_a] = tid;
            }
            #pragma unroll
            for (int ikxdl = 0; ikxdl < K_PACK; ikxdl++) {
                #pragma unroll
                for (int im_a = 0; im_a < MN_PACK; im_a++) {
                    const int k_idx = ku * K_PACK * 4 + ikxdl * 4 + k_lane;
                    bytes[ikxdl * MN_PACK + im_a] =
                        a_scale[(long long)tok_ids[im_a] * A_SCALE_COLS + k_idx];
                }
            }
        }

        const long long out_offset = (long long)work_id * 4;
        *reinterpret_cast<uint32_t*>(&a_scale_sorted_shuffled[out_offset]) =
            *reinterpret_cast<const uint32_t*>(bytes);
    }
}

template <int BM, int NE, int D_INTER, int D_HIDDEN, int BK,
          int N_CTAS, int THREADS_PER_CTA>
inline void launch(
    hipStream_t stream, int M, int max_sorted,
    const uint8_t *a_scale, const int32_t *sorted_token_ids, const int32_t *cumsum,
    uint8_t *a_scale_sorted_shuffled)
{
    sort_scales_kernel_impl<BM, NE, D_INTER, D_HIDDEN, BK, N_CTAS, THREADS_PER_CTA>
        <<<N_CTAS, THREADS_PER_CTA, 0, stream>>>(
            M, max_sorted, a_scale, sorted_token_ids, cumsum, a_scale_sorted_shuffled);
}

} // namespace aiter::mxfp4_moe::moe_sort_scales
