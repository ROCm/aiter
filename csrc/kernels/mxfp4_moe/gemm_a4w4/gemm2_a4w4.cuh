// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp4.h>
#include <hip/hip_ext_ocp.h>
#include <hip/hip_bf16.h>

#include "ck_tile/core/utility/functional.hpp"

#include "common/mxfp4_gemm_common.hpp"
#include "gemm_a4w4/common/mfma_f4f4.hpp"
#include "gemm_a4w4/common/mxfp4_epilogs.hpp"
#include "gemm_a4w4/common/xcd_remap.hpp"

namespace aiter::mxfp4_moe::gemm2 {

using namespace aiter::mxfp4_moe::gemm_common;

enum class EpilogPolicy : int { Atomic = 0, Nonatomic = 1 };

constexpr int NUM_CU = 256;

template <EpilogPolicy kEpilog>
constexpr bool is_atomic_v = (kEpilog == EpilogPolicy::Atomic);
template <EpilogPolicy kEpilog>
constexpr bool is_nonatomic_v = (kEpilog == EpilogPolicy::Nonatomic);

template <bool kAtomic, int BM, int BK, int BN, int kStages, int kAS_LDS_slot_bytes>
struct LDSLayout;

template <int BM, int BK, int BN, int kStages, int kAS_LDS_slot_bytes>
struct alignas(16) LDSLayout</*kAtomic=*/true, BM, BK, BN, kStages, kAS_LDS_slot_bytes> {
    union {
        alignas(16) __hip_fp4x2_storage_t s_Aq[kStages][BM][BK / 2];
        alignas(16) float lds_acc[BM * BN];
    };
};

template <int BM, int BK, int BN, int kStages, int kAS_LDS_slot_bytes>
struct alignas(16) LDSLayout</*kAtomic=*/false, BM, BK, BN, kStages, kAS_LDS_slot_bytes> {
    union {
        alignas(16) __hip_fp4x2_storage_t s_Aq[kStages][BM][BK / 2];
        alignas(16) float                 lds_acc[BM * BN];
    };
    alignas(16) uint8_t               s_Ascale[kStages][kAS_LDS_slot_bytes];
};

template <int NUM_EXPERTS, int K, int N_OUT, int TOPK, int BM,
          EpilogPolicy kEpilog,
          bool kUseNT = false,
          int kXcdSwizzle = 0,
          bool kMxfp4Out = false>
__global__ void
__launch_bounds__(256,
                  is_nonatomic_v<kEpilog> ? 1 :
                  ((BM == 16) ? 4 : 2))
kernel(
    const __hip_fp4x2_storage_t* __restrict__ A_q,
    const __amd_scale_t*         __restrict__ A_scale,
    const __hip_fp4x2_storage_t* __restrict__ B_q,
    const __amd_scale_t*         __restrict__ B_scale,
    const int*                   __restrict__ sorted_expert_ids,
    const int*                   __restrict__ cumsum_tensor,
    const int*                   __restrict__ sorted_token_ids,
    const float*                 __restrict__ sorted_weights,
    int                                       M,
    int                                       max_sorted,
    __hip_bfloat16*              __restrict__ out_bf16,
    uint8_t*                     __restrict__ flat_out_scale)
{
    static_assert(K == 512);
    static_assert(N_OUT % 256 == 0);
    constexpr bool kAtomic     = is_atomic_v<kEpilog>;
    constexpr bool kNonatomic  = is_nonatomic_v<kEpilog>;
    constexpr bool kUseAGPR    = kNonatomic;
    constexpr bool kPersistent = kNonatomic;
    static_assert(
        (kAtomic    && (BM == 16 || BM == 32 || BM == 64)) ||
        (kNonatomic && BM == 128),
        "Atomic supports BM ∈ {16,32,64}; Nonatomic supports BM == 128");

    constexpr int BN     = 256;
    constexpr int BK     = 256;
    constexpr int K_HALF = K / 2;

    constexpr int K_TILES_TOTAL = K / BK;
    constexpr int kStages       = 2;
    constexpr int kLoopIter     = K_TILES_TOTAL - kStages;
    constexpr int kUnroll       = kLoopIter;
    constexpr int kSubBlocks    = (BM < 32) ? 1 : BM / 32;
    constexpr int kMChunks      = (BM == 16) ? 1 : BM / 16;
    constexpr int BM_GRID       = BM;
    constexpr int kCachedRows   = (BM == 16) ? 2 : kSubBlocks;

    constexpr int kBS_c_n1            = N_OUT / 16 / 2;
    constexpr int kBS_c_k1            = (K / 32) / 4 / 2;
    constexpr int kBS_stride_k0_dw    = 64;
    constexpr int kBS_stride_n0_dw    = kBS_c_k1 * 64;
    constexpr int kBS_per_expert_dw   = kBS_c_n1 * kBS_stride_n0_dw;

    constexpr int kAS_c_k1            = (K / 32) / 4 / 2;
    constexpr int kAS_per_chunk_dw    = 1 * kAS_c_k1 * 64;

    constexpr int kAS_LDS_slot_bytes  = kSubBlocks * 256;

    const int pid    = blockIdx.x;
    const int tid    = threadIdx.x;
    __builtin_assume(0 <= tid && tid < 256);
    const int wave   = __builtin_amdgcn_readfirstlane(tid / 64);
    const int wave_n = wave;
    const int lane   = tid % 64;

    const buffer_rsrc_t A_q_rsrc =
        make_buffer_rsrc(A_q, (uint32_t)((long long)max_sorted * K_HALF * sizeof(__hip_fp4x2_storage_t)));
    const buffer_rsrc_t B_q_rsrc =
        make_buffer_rsrc(B_q,
            (uint32_t)((long long)NUM_EXPERTS * N_OUT * K_HALF * sizeof(__hip_fp4x2_storage_t)));
    constexpr int kAS_bound_div = (BM == 16) ? BM_GRID : 32;
    const buffer_rsrc_t A_scale_rsrc =
        make_buffer_rsrc(A_scale,
            (uint32_t)((long long)(max_sorted / kAS_bound_div) * kAS_per_chunk_dw * 4));
    const buffer_rsrc_t B_scale_rsrc =
        make_buffer_rsrc(B_scale, (uint32_t)((long long)NUM_EXPERTS * kBS_per_expert_dw * 4));

    __shared__ LDSLayout<kAtomic, BM, BK, BN, kStages, kAS_LDS_slot_bytes> lds;
    auto&        s_Aq    = lds.s_Aq;

    i32x4 a[kMChunks][2];
    i32x4 b[kStages][4][2];
    int   b_load_s_base[4];
    int   a_scale_s_base[kSubBlocks];
    int   b_scale_s_base[2];
    int   a_scale_aiter[kSubBlocks];
    int   a_scale_v[kSubBlocks][2];
    int   b_scale_v[kStages][2];
    f32x4 accm[kMChunks][4];
    f32x4 c_zero;

    auto issue_a_load_lds = [&](int slot, int kt,
                                const int car[kCachedRows]) {
        constexpr int kRowsPerChunk = 8;
        constexpr int kLanesPerRow  = 8;
        const int row_off = lane / kLanesPerRow;
        if constexpr (BM == 16) {
            if (wave < 2) {
                const int lds_row = wave * 8;
                const int mask    = lds_swizzle_mask<BK / 2>(lds_row + row_off);
                const int voffset = (((lane % kLanesPerRow) * 16) ^ mask)
                                  + car[wave] * (K / 2);
                buffer_load_lds(A_q_rsrc, &s_Aq[slot][lds_row][0],
                                /*size=*/16, voffset, kt * (BK / 2), 0, 0);
            }
        } else {
            #pragma unroll
            for (int sub = 0; sub < kSubBlocks; sub++) {
                const int lds_row = wave * (BM / 4) + sub * kRowsPerChunk;
                const int mask    = lds_swizzle_mask<BK / 2>(lds_row + row_off);
                const int voffset = (((lane % kLanesPerRow) * 16) ^ mask)
                                  + car[sub] * (K / 2);
                buffer_load_lds(A_q_rsrc, &s_Aq[slot][lds_row][0],
                                /*size=*/16, voffset, kt * (BK / 2), 0, 0);
            }
        }
    };

    auto issue_a_ds_read = [&](int lds_slot) {
        const int lane_row = lane % 16;
        const int lane_col = (lane / 16) * 16;
        const int mask     = lds_swizzle_mask<BK / 2>(lane_row);
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            const int lds_col = (lane_col + k * 64) ^ mask;
            #pragma unroll
            for (int i = 0; i < kMChunks; i++) {
                const int lds_row = lane_row + i * 16;
                *reinterpret_cast<i32x4*>(&a[i][k]) =
                    *reinterpret_cast<i32x4*>(&s_Aq[lds_slot][lds_row][lds_col]);
            }
        }
    };

    auto issue_a_scale_load_atomic = [&]() {
        const int v_voff = ((lane / 16) * 16 + (lane % 16)) * 4;
        #pragma unroll
        for (int sub = 0; sub < kSubBlocks; sub++) {
            a_scale_v[sub][0] = buffer_load_b32_imm<  0>(
                A_scale_rsrc, v_voff, a_scale_s_base[sub]);
            a_scale_v[sub][1] = buffer_load_b32_imm<256>(
                A_scale_rsrc, v_voff, a_scale_s_base[sub]);
        }
    };

    auto issue_a_scale_ds_read_ku_atomic = [&]<int KU>() {
        #pragma unroll
        for (int sub = 0; sub < kSubBlocks; sub++) {
            a_scale_aiter[sub] = a_scale_v[sub][KU];
        }
    };

    auto issue_a_scale_load_nonatomic = [&](int slot, int kt) {
        if constexpr (!kNonatomic) return;
        const int v_voff = ((lane / 16) * 16 + (lane % 16)) * 4;
        const int mi = wave_n;
        if (mi >= kSubBlocks) return;
        const int s_voff = __builtin_amdgcn_readfirstlane(
            a_scale_s_base[mi] + kt * (64 * 4));
        if constexpr (kNonatomic) {
            buffer_load_lds(A_scale_rsrc, &lds.s_Ascale[slot][mi * 256],
                            /*size=*/4, v_voff, s_voff, 0, 0);
        }
    };

    auto issue_a_scale_ds_read_nonatomic = [&](int slot) {
        if constexpr (!kNonatomic) return;
        #pragma unroll
        for (int sub = 0; sub < kSubBlocks; sub++) {
            if constexpr (kNonatomic) {
                const int lds_off = sub * 256
                                  + (lane / 16) * 64
                                  + (lane % 16) * 4;
                a_scale_aiter[sub] = *reinterpret_cast<int*>(&lds.s_Ascale[slot][lds_off]);
            }
        }
    };

    auto issue_b_load_j = [&]<int K_C>(auto& b_sub, int j) {
        constexpr int K_BYTE = K_C * 2048;
        const int v_voff = (lane / 16) * 256
                         + (lane % 16) * 16
                         + K_BYTE;
        constexpr int kBQ_AUX = (kAtomic && kUseNT) ? 2 : 0;
        buffer_load_b128_imm_inplace<   0, kBQ_AUX>(
            b_sub[j][0], B_q_rsrc, v_voff, b_load_s_base[j]);
        buffer_load_b128_imm_inplace<1024, kBQ_AUX>(
            b_sub[j][1], B_q_rsrc, v_voff, b_load_s_base[j]);
    };

    auto issue_b_scale_load_ku = [&]<int KU>(auto& bs_sub) {
        const int v_voff = ((lane / 16) * 16 + (lane % 16)) * 4;
        constexpr int IMM = KU * (kBS_stride_k0_dw * 4);
        #pragma unroll
        for (int mw = 0; mw < 2; mw++) {
            bs_sub[mw] = buffer_load_b32_imm<IMM>(
                B_scale_rsrc, v_voff, b_scale_s_base[mw]);
        }
    };

    auto issue_mfma_cluster = [&]<int J, bool kInit>(int slot) {
        constexpr int mni  = J / 2;
        constexpr int in_b = J % 2;
        const int sb = b_scale_v[slot][mni];
        #pragma unroll
        for (int sub = 0; sub < kSubBlocks; sub++) {
            const int sa = a_scale_aiter[sub];
            const int i0 = sub * 2 + 0;
            [[maybe_unused]] const int i1 = sub * 2 + 1;
            if constexpr (kInit) {
                if constexpr (kUseAGPR) mfma_f4f4_agpr_init_zero<0, 0 + in_b>(accm[i0][J], a[i0][0], b[slot][J][0], sa, sb);
                else                    mfma_f4f4_vgpr_init<0, 0 + in_b>(accm[i0][J], a[i0][0], b[slot][J][0], c_zero, sa, sb);
            } else {
                if constexpr (kUseAGPR) mfma_f4f4_agpr<0, 0 + in_b>(accm[i0][J], a[i0][0], b[slot][J][0], sa, sb);
                else                    mfma_f4f4_vgpr<0, 0 + in_b>(accm[i0][J], a[i0][0], b[slot][J][0], sa, sb);
            }
            if constexpr (BM != 16) {
                if constexpr (kInit) {
                    if constexpr (kUseAGPR) mfma_f4f4_agpr_init_zero<1, 0 + in_b>(accm[i1][J], a[i1][0], b[slot][J][0], sa, sb);
                    else                    mfma_f4f4_vgpr_init<1, 0 + in_b>(accm[i1][J], a[i1][0], b[slot][J][0], c_zero, sa, sb);
                } else {
                    if constexpr (kUseAGPR) mfma_f4f4_agpr<1, 0 + in_b>(accm[i1][J], a[i1][0], b[slot][J][0], sa, sb);
                    else                    mfma_f4f4_vgpr<1, 0 + in_b>(accm[i1][J], a[i1][0], b[slot][J][0], sa, sb);
                }
            }
            if constexpr (kUseAGPR) mfma_f4f4_agpr<2, 2 + in_b>(accm[i0][J], a[i0][1], b[slot][J][1], sa, sb);
            else                    mfma_f4f4_vgpr<2, 2 + in_b>(accm[i0][J], a[i0][1], b[slot][J][1], sa, sb);
            if constexpr (BM != 16) {
                if constexpr (kUseAGPR) mfma_f4f4_agpr<3, 2 + in_b>(accm[i1][J], a[i1][1], b[slot][J][1], sa, sb);
                else                    mfma_f4f4_vgpr<3, 2 + in_b>(accm[i1][J], a[i1][1], b[slot][J][1], sa, sb);
            }
        }
    };

    auto run_one = [&](int m_block_idx, int n_block_idx, int e) {
        const int m_row = m_block_idx * BM_GRID;
        if constexpr (kAtomic) {
            c_zero = f32x4{0.f, 0.f, 0.f, 0.f};
        }
        __builtin_assume(0 <= e && e < NUM_EXPERTS);

        int cached_actual_row[kCachedRows];
        if constexpr (kAtomic) {
            const int row_off = lane / 8;
            if constexpr (BM == 16) {
                if (wave < 2) {
                    cached_actual_row[wave] = m_row + wave * 8 + row_off;
                }
            } else {
                const int lds_row = wave * (BM / 4);
                #pragma unroll
                for (int sub = 0; sub < kSubBlocks; sub++) {
                    cached_actual_row[sub] = m_row + lds_row + sub * 8 + row_off;
                }
            }
        } else {
            const int row_off = lane / 8;
            const int lds_row = wave * (BM / 4);
            #pragma unroll
            for (int sub = 0; sub < kSubBlocks; sub++) {
                cached_actual_row[sub] = m_row + lds_row + sub * 8 + row_off;
            }
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            b_load_s_base[j] = __builtin_amdgcn_readfirstlane(
                ((long long)e * N_OUT + n_block_idx * BN + wave_n * (BN / 4) + j * 16)
                * (K / 2));
        }

        {
            const int mni_base = n_block_idx * (BN / 16 / 2)
                               + wave_n     * (BN / 64 / 2);
            #pragma unroll
            for (int mw = 0; mw < 2; mw++) {
                b_scale_s_base[mw] = __builtin_amdgcn_readfirstlane(
                    ((long long)e               * kBS_per_expert_dw
                   + (mni_base + mw) * kBS_stride_n0_dw) * 4);
            }
        }

        {
            const int chunk_base = (BM == 16) ? (m_row / BM_GRID) : (m_row / 32);
            #pragma unroll
            for (int sub = 0; sub < kSubBlocks; sub++) {
                a_scale_s_base[sub] = __builtin_amdgcn_readfirstlane(
                    (chunk_base + sub) * kAS_per_chunk_dw * 4);
            }
        }

        if constexpr (kNonatomic) {
            // iter-boundary fence: persistent-grid only, LDS-slot reuse race.
            __syncthreads();

            issue_a_load_lds(0, 0, cached_actual_row);
            issue_a_scale_load_nonatomic(/*slot=*/0, /*kt=*/0);
            issue_a_load_lds(1, 1, cached_actual_row);
            issue_a_scale_load_nonatomic(/*slot=*/1, /*kt=*/1);
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for (int j = 0; j < 4; j++)
                issue_b_load_j.template operator()<0>(b[0], j);
            issue_b_scale_load_ku.template operator()<0>(b_scale_v[0]);
            #pragma unroll
            for (int j = 0; j < 4; j++)
                issue_b_load_j.template operator()<1>(b[1], j);
            issue_b_scale_load_ku.template operator()<1>(b_scale_v[1]);

            ck_tile::static_for<0, kStages, 1>{}([&](auto ss) {
                constexpr int S     = ss.value;
                constexpr int kt    = K_TILES_TOTAL - kStages + S;
                constexpr int slot_ = kt % kStages;
                __syncthreads();
                issue_a_ds_read(/*lds_slot=*/slot_);
                issue_a_scale_ds_read_nonatomic(/*slot=*/slot_);
                ck_tile::static_for<0, 4, 1>{}([&](auto jj) {
                    constexpr int J = jj.value;
                    issue_mfma_cluster.template operator()<J, /*kInit=*/(S == 0)>(slot_);
                });
            });

            if constexpr (kMxfp4Out) {
                apply_mxfp4_flat_epilog_bm128<N_OUT>(
                    accm, reinterpret_cast<uint8_t*>(out_bf16), flat_out_scale,
                    m_row, n_block_idx, wave_n, lane, tid, lds.lds_acc);
            } else {
                apply_bf16_flat_epilog_bm128<N_OUT>(
                    accm, out_bf16, m_row, n_block_idx, wave_n, lane);
            }
        } else {
            issue_a_load_lds(0, 0, cached_actual_row);
            issue_a_load_lds(1, 1, cached_actual_row);
            __builtin_amdgcn_sched_barrier(0);
            issue_a_scale_load_atomic();
            issue_b_scale_load_ku.template operator()<0>(b_scale_v[0]);
            issue_b_scale_load_ku.template operator()<1>(b_scale_v[1]);
            #pragma unroll
            for (int j = 0; j < 4; j++)
                issue_b_load_j.template operator()<0>(b[0], j);
            #pragma unroll
            for (int j = 0; j < 4; j++)
                issue_b_load_j.template operator()<1>(b[1], j);

            ck_tile::static_for<0, kStages, 1>{}([&](auto ss) {
                constexpr int S     = ss.value;
                constexpr int kt    = K_TILES_TOTAL - kStages + S;
                constexpr int slot_ = kt % kStages;
                // vmcnt(23/22): cross-wave correctness fence (loads land before ds_read), not a perf knob.
                if constexpr (S == 0) {
                    asm volatile("s_waitcnt vmcnt(23)" ::: "memory");
                } else {
                    asm volatile("s_waitcnt vmcnt(22)" ::: "memory");
                }
                __builtin_amdgcn_s_barrier();
                issue_a_ds_read(/*lds_slot=*/slot_);
                issue_a_scale_ds_read_ku_atomic.template operator()<kt>();
                ck_tile::static_for<0, 4, 1>{}([&](auto jj) {
                    constexpr int J = jj.value;
                    issue_mfma_cluster.template operator()<J, /*kInit=*/(S == 0)>(slot_);
                });
            });

            __syncthreads();
            apply_atomic_bf16_epilog<N_OUT, BM>(
                accm, out_bf16, sorted_token_ids, sorted_weights,
                m_row, n_block_idx, wave_n, lane, tid, M, lds.lds_acc);
        }
    };

    constexpr int num_n_blocks_local = N_OUT / 256;
    const int total_m_blocks = __ldg(cumsum_tensor) / BM_GRID;
    if constexpr (kPersistent) {
        const int total_work = total_m_blocks * num_n_blocks_local;
        const int grid_x     = gridDim.x;
        for (int wu = pid; wu < total_work; wu += grid_x) {
            int m_block_idx, n_block_idx;
            if constexpr (kXcdSwizzle != 0) {
                remap_xcd_grouped</*NUM_XCDS=*/8, kXcdSwizzle>(
                    wu, total_m_blocks, num_n_blocks_local,
                    m_block_idx, n_block_idx);
            } else {
                m_block_idx = wu / num_n_blocks_local;
                n_block_idx = wu % num_n_blocks_local;
            }
            const int e = __ldg(sorted_expert_ids + m_block_idx);
            run_one(m_block_idx, n_block_idx, e);
        }
    } else {
        if (pid >= total_m_blocks * num_n_blocks_local) return;
        int m_block_idx, n_block_idx;
        if constexpr (kXcdSwizzle != 0) {
            remap_xcd_grouped</*NUM_XCDS=*/8, kXcdSwizzle>(
                pid, total_m_blocks, num_n_blocks_local,
                m_block_idx, n_block_idx);
        } else {
            m_block_idx = pid / num_n_blocks_local;
            n_block_idx = pid % num_n_blocks_local;
        }
        const int e = __ldg(sorted_expert_ids + m_block_idx);
        run_one(m_block_idx, n_block_idx, e);
    }
}

template <int NUM_EXPERTS, int K, int N_OUT, int TOPK, int BM,
          bool kUseNT = false, int kXcdSwizzle = 0>
inline void launch_atomic(
    hipStream_t stream,
    const void* A_q,    const void* A_scale,
    const void* B_q,    const void* B_scale,
    const int*  sorted_expert_ids, const int* cumsum_tensor,
    const int*  sorted_token_ids, const float* sorted_weights,
    int M,
    void*       out)
{
    static_assert(BM == 16 || BM == 32 || BM == 64, "BM must be 16, 32, or 64");
    constexpr int BM_GRID = BM;
    constexpr int num_n_blocks = N_OUT / 256;
    const int max_m_blocks =
        (M * TOPK + NUM_EXPERTS * (BM_GRID - 1) + BM_GRID - 1) / BM_GRID;
    const int grid = max_m_blocks * num_n_blocks;
    const int max_sorted = max_m_blocks * BM;  // runtime A_q/A_scale bound (replaces MAX_M)
    kernel<NUM_EXPERTS, K, N_OUT, TOPK, BM,
           EpilogPolicy::Atomic, kUseNT, kXcdSwizzle>
        <<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __hip_fp4x2_storage_t*>(A_q),
            reinterpret_cast<const __amd_scale_t*>(A_scale),
            reinterpret_cast<const __hip_fp4x2_storage_t*>(B_q),
            reinterpret_cast<const __amd_scale_t*>(B_scale),
            sorted_expert_ids, cumsum_tensor,
            sorted_token_ids, sorted_weights,
            M, max_sorted,
            reinterpret_cast<__hip_bfloat16*>(out),
            /*flat_out_scale=*/nullptr);
}

template <int NUM_EXPERTS, int K, int N_OUT, int kXcdSwizzle = 0>
inline void launch_nonatomic(
    hipStream_t stream,
    const void* A_q,    const void* A_scale,
    const void* B_q,    const void* B_scale,
    const int*  sorted_expert_ids, const int* cumsum_tensor,
    int max_sorted,
    void*       flat_out)
{
    constexpr int BM = 128;
    constexpr int num_n_blocks = N_OUT / 256;
    const int max_m_blocks = (max_sorted + BM - 1) / BM;
    const int total_work   = max_m_blocks * num_n_blocks;
    const int grid = (total_work < NUM_CU) ? total_work : NUM_CU;
    kernel<NUM_EXPERTS, K, N_OUT, /*TOPK=*/9, BM,
           EpilogPolicy::Nonatomic, /*kUseNT=*/false,
           kXcdSwizzle>
        <<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __hip_fp4x2_storage_t*>(A_q),
            reinterpret_cast<const __amd_scale_t*>(A_scale),
            reinterpret_cast<const __hip_fp4x2_storage_t*>(B_q),
            reinterpret_cast<const __amd_scale_t*>(B_scale),
            sorted_expert_ids, cumsum_tensor,
            /*sorted_token_ids=*/nullptr, /*sorted_weights=*/nullptr,
            /*M=*/0, max_sorted,
            reinterpret_cast<__hip_bfloat16*>(flat_out),
            /*flat_out_scale=*/nullptr);
}

template <int NUM_EXPERTS, int K, int N_OUT, int kXcdSwizzle = 0>
inline void launch_nonatomic_mxfp4(
    hipStream_t stream,
    const void* A_q,    const void* A_scale,
    const void* B_q,    const void* B_scale,
    const int*  sorted_expert_ids, const int* cumsum_tensor,
    int max_sorted,
    void*       flat_out_q,
    void*       flat_out_scale)
{
    constexpr int BM = 128;
    constexpr int num_n_blocks = N_OUT / 256;
    const int max_m_blocks = (max_sorted + BM - 1) / BM;
    const int total_work   = max_m_blocks * num_n_blocks;
    const int grid = (total_work < NUM_CU) ? total_work : NUM_CU;
    kernel<NUM_EXPERTS, K, N_OUT, /*TOPK=*/9, BM,
           EpilogPolicy::Nonatomic, /*kUseNT=*/false,
           kXcdSwizzle, /*kMxfp4Out=*/true>
        <<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __hip_fp4x2_storage_t*>(A_q),
            reinterpret_cast<const __amd_scale_t*>(A_scale),
            reinterpret_cast<const __hip_fp4x2_storage_t*>(B_q),
            reinterpret_cast<const __amd_scale_t*>(B_scale),
            sorted_expert_ids, cumsum_tensor,
            /*sorted_token_ids=*/nullptr, /*sorted_weights=*/nullptr,
            /*M=*/0, max_sorted,
            reinterpret_cast<__hip_bfloat16*>(flat_out_q),
            reinterpret_cast<uint8_t*>(flat_out_scale));
}

}
