#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// a16w4 DECODE GEMM (bf16 x int4 -> bf16) for gfx1201 -- grid-level split-K.
// Tile: BM=16, BN=128, BK=64, WM=16, WN=32 | 4 waves/WG | 128 threads
// grid = (ceil(M/BM), N/BN, SPLIT_K)
//
// =====================================================================
// WHY GRID-LEVEL SPLIT-K, AND WHY S=2 (not the "balanced" S=4)
// =====================================================================
// With BM = WM = 16 the wave count is fixed by N and WN alone:
//
//     total waves = (N/BN) x (BN/WN) = N/WN
//
// BN cancels. At N=5120, WN=32 that is 160 waves for ANY BN -- which is
// exactly why BN=64, 128 and 256 all landed near 400 GB/s. K is the only
// free variable left, and the split MUST be at grid level: splitting inside
// the workgroup leaves the grid at 40 WGs and merely moves the problem
// inside the WG.
//
// Cold GEMM at N=5120 K=5120:
//
//     S   waves/SIMD32   "balanced"   GEMM      GEMM+reduce
//     1       1.25           no       37.36 us  37.27 us
//     2       2.50           no       27.33 us  30.35 us   <-- BEST
//     4       5.00          YES       28.51 us  31.51 us
//     8      10.00          YES       29.17 us  32.08 us
//
// S=2 wins and BOTH "balanced" configs lose to it. The mechanism is
// RESIDENCY, not divisibility:
//
//   - LDS is 39424 B/WG against 128 KB/WGP, so only floor(128/38.5) = 3 WGs
//     fit per WGP = 3 waves/SIMD32. That is a hard cap.
//   - S=4 asks for 5 waves/SIMD32 and S=8 asks for 10. Those surplus waves
//     are QUEUED across dispatch rounds, not co-resident, so they hide no
//     additional memory latency.
//   - Meanwhile every WG pays ~2 K-steps of pipeline fill, and the WG count
//     scales with S while useful work is fixed:
//         fill = 2S/80 -> 2.5% / 5% / 10% / 20% for S = 1 / 2 / 4 / 8
//     Measured degradation past S=2 is +4.3% and +6.7%, tracking that.
//
// So: occupancy benefit SATURATES at the resident cap, fill cost grows
// LINEARLY with S. Pick the largest S still under the cap -- here, 2. THE
// RIGHT S DEPENDS ON LDS_TOTAL, so changing the tile changes the best S.
//
// =====================================================================
// OUTPUT PATH
// =====================================================================
// Each split writes an fp32 partial to workspace[split][M][N] with PLAIN
// STORES -- no atomics, so the result is bit-deterministic run to run. A
// second kernel sums the SPLIT_K partials and narrows to bf16.
//
// The reduce costs ~3.0 us at M=1 N=5120, and it cost the same ~3.0 us at
// S=2, 4 and 8. Constant in S means it is LAUNCH LATENCY, not bandwidth --
// it moves 170 KB against 13.95 MB of weights (1.2%). Fusing it into the
// GEMM epilogue is the next ~10% and is left undone deliberately: it needs
// a cross-workgroup completion protocol.
//
// =====================================================================
// TWO THINGS THAT WERE MEASURED AND LOST -- DO NOT RETRY BLIND
// =====================================================================
//  - LDS-only barrier instead of __syncthreads: 30.40 us vs 28.45 us,
//    spread 0.25 us. The barrier semantics were never the problem; the
//    "memory" clobber on the inline asm is opaque to the scheduler, which
//    then loses track of outstanding global loads and waits before every
//    use (15 s_wait_loadcnt vs 1). Diff the codegen wait counts first.
//  - Live dequant straight into the WMMA operand instead of staging through
//    LDS: raised VALU per element from 3.875 to 4.48 and created WAR
//    hazards against the 320-cycle WMMA latency.

#include "gfx1201/gemm_a16w4_common_gfx1201.cuh"

// FINAL. Measured best at N=5120 K=5120 (see the sweep above). This is NOT a
// free knob: S changes the resident-WG count and the pipeline fill cost, and
// the optimum moves with LDS_TOTAL. If you change the tile, re-sweep.
#define SPLIT_K 2

#define BM 16
#define BN 128
#define BK 64
#define WM 16
#define WN 32
#define NUM_WAVES ((BM / WM) * (BN / WN)) // 4
#define THREADS (NUM_WAVES * 32)          // 128
#define GROUP_SIZE 128
#define PACKS_PER_THR (BK / 8)          // 8
#define A_PER_THR ((BM * BK) / THREADS) // 8

#define LDS_A_STRIDE (BK + 8)  // 72 elems, b128 aligned
#define LDS_WK_STRIDE (BK + 4) // 68 elems, b64 aligned

#define LDS_BUF_A_ELEMS (BM * LDS_A_STRIDE)
#define LDS_BUF_W_ELEMS (BN * LDS_WK_STRIDE)
#define LDS_BUF_ELEMS (LDS_BUF_A_ELEMS + LDS_BUF_W_ELEMS)
#define LDS_TOTAL (2 * LDS_BUF_ELEMS * 2) // 39424 B

namespace aiter {
namespace a16w4 {
namespace decode_bf16 {

static_assert(GROUP_SIZE % BK == 0, "GROUP_SIZE must be a multiple of BK");

// ─── Pipeline stage macros ──────────────────────────────────────────────────

// Nothing loaded here may be USED here -- touching a just-issued load forces
// an s_waitcnt mid-prefetch and collapses the pipeline.
#define PREFETCH_TILE(R, KS_LOAD)                                           \
    {                                                                       \
        const int nk_ = (KS_LOAD)*BK;                                       \
        const int ng_ = nk_ / GROUP_SIZE;                                   \
        rsc##R        = sc_base[(size_t)ng_ * N + my_col];                  \
        rzr##R        = zr_base[(size_t)ng_ * N + my_col];                  \
        if(a_ok)                                                            \
            rA##R = *(const u16x8_t*)(a_src + nk_);                         \
        const unsigned int* wsrc_ = W_col + (size_t)(nk_ / 8) * N;          \
        _Pragma("unroll") for(int p = 0; p < PACKS_PER_THR; p++) rW##R[p] = \
            wsrc_[(size_t)p * N];                                           \
    }

// Staged dequant: registers -> LDS.
#define DEQUANT_TO_LDS(R, B)                                                    \
    {                                                                           \
        const float sc_ = bf16_to_f32(rsc##R);                                  \
        const float zs_ = -bf16_to_f32(rzr##R) * sc_;                           \
        *(u16x8_t*)(&lds_buf[B][a_lds_off]) = rA##R;                            \
        bf16_raw_t* wcol_ = &lds_buf[B][LDS_BUF_A_ELEMS + tid * LDS_WK_STRIDE]; \
        _Pragma("unroll") for(int p = 0; p < PACKS_PER_THR; p++)                \
        {                                                                       \
            const unsigned packed_ = rW##R[p];                                  \
            u16x4_t lo_, hi_;                                                   \
            _Pragma("unroll") for(int j = 0; j < 4; j++)                        \
            {                                                                   \
                lo_[j] = f32_to_bf16(                                           \
                    fmaf((float)((packed_ >> (j * 4)) & 0xF), sc_, zs_));        \
                hi_[j] = f32_to_bf16(                                           \
                    fmaf((float)((packed_ >> ((j + 4) * 4)) & 0xF), sc_, zs_));  \
            }                                                                   \
            *(u16x4_t*)(wcol_ + p * 8)     = lo_;                               \
            *(u16x4_t*)(wcol_ + p * 8 + 4) = hi_;                               \
        }                                                                       \
    }

#define WMMA_FROM_LDS(B)                                                             \
    {                                                                                \
        const bf16_raw_t* srcA_ = &lds_buf[B][0];                                    \
        const bf16_raw_t* srcW_ = &lds_buf[B][LDS_BUF_A_ELEMS];                      \
        _Pragma("unroll") for(int s = 0; s < BK / 16; s++)                           \
        {                                                                            \
            const int koff_    = s * 16 + laneGroup * 8;                             \
            frag_bf16x8 af_    = as_bf16x8(                                          \
                *(const u16x8_t*)(srcA_ + laneWrapped * LDS_A_STRIDE + koff_));      \
            const bf16_raw_t* w0_ = srcW_ + wcol0 * LDS_WK_STRIDE + koff_;           \
            const bf16_raw_t* w1_ = srcW_ + wcol1 * LDS_WK_STRIDE + koff_;           \
            frag_bf16x8 bf0_      = as_bf16x8(__builtin_shufflevector(               \
                *(const u16x4_t*)(w0_), *(const u16x4_t*)(w0_ + 4), 0, 1, 2, 3, 4, 5, 6, 7)); \
            frag_bf16x8 bf1_      = as_bf16x8(__builtin_shufflevector(               \
                *(const u16x4_t*)(w1_), *(const u16x4_t*)(w1_ + 4), 0, 1, 2, 3, 4, 5, 6, 7)); \
            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(af_, bf0_, acc0); \
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(af_, bf1_, acc1); \
        }                                                                            \
    }

#define PIPE_STEP(RPF, RDQ, CURB, NXTB, KS) \
    {                                       \
        const int ks_ = (KS);               \
        if(ks_ + 2 < k_steps)               \
            PREFETCH_TILE(RPF, ks_ + 2)     \
        WMMA_FROM_LDS(CURB)                 \
        if(ks_ + 1 < k_steps)               \
            DEQUANT_TO_LDS(RDQ, NXTB)       \
        __syncthreads();                    \
    }

// ─── GEMM kernel ────────────────────────────────────────────────────────────

__global__ __launch_bounds__(THREADS) void gemm_a16w4_decode_bf16_kernel(
    const bf16_raw_t* __restrict__ A,      // [M, K] bf16
    const unsigned int* __restrict__ W_q,  // [K/8, N] int32, packed along K
    const bf16_raw_t* __restrict__ scales, // [K/G, N] bf16
    const bf16_raw_t* __restrict__ zeros,  // [K/G, N] bf16
    float* __restrict__ workspace,         // [SPLIT_K, M, N] fp32 partials
    int M,
    int N,
    int K)
{
    AITER_A16W4_REQUIRE_GFX1201();
#if AITER_A16W4_DEVICE_SUPPORTED
    const int bm          = blockIdx.x * BM;
    const int bn          = blockIdx.y * BN;
    const int split       = blockIdx.z;
    const int tid         = threadIdx.x;
    const int wave_n      = tid / 32;
    const int lane_id     = tid % 32;
    const int laneWrapped = lane_id % 16;
    const int laneGroup   = lane_id / 16;
    const int wn_start    = wave_n * WN;
    const int my_col      = bn + tid;

    const int wcol0 = wn_start + laneWrapped;
    const int wcol1 = wcol0 + 16;

    // This WG owns K range [k_begin, k_begin + K/SPLIT_K). The bases are
    // advanced once so the pipeline macros are identical to the SPLIT_K=1
    // case -- the split costs nothing in the inner loop.
    const int k_span  = K / SPLIT_K;
    const int k_begin = split * k_span;
    const int k_steps = k_span / BK;

    __shared__ bf16_raw_t lds_buf[2][LDS_BUF_ELEMS];

    frag_f32x8 acc0 = {0}, acc1 = {0};

    const unsigned int* W_col = W_q + (size_t)(k_begin / 8) * N + bn + tid;
    const bf16_raw_t* sc_base = scales + (size_t)(k_begin / GROUP_SIZE) * N;
    const bf16_raw_t* zr_base = zeros + (size_t)(k_begin / GROUP_SIZE) * N;

    // Activation addressing, hoisted out of the loop.
    const int a_row            = (tid * A_PER_THR) / BK;
    const int a_col            = (tid * A_PER_THR) % BK;
    const bool a_ok            = (bm + a_row) < M;
    const int a_row_clamped    = a_ok ? (bm + a_row) : 0;
    const bf16_raw_t* a_src    = A + (long)a_row_clamped * K + k_begin + a_col;
    const int a_lds_off        = a_row * LDS_A_STRIDE + a_col;

    unsigned rW0[PACKS_PER_THR], rW1[PACKS_PER_THR];
    u16x8_t rA0, rA1;
    bf16_raw_t rsc0, rzr0, rsc1, rzr1;

    rA0 = (u16x8_t)(bf16_raw_t)0;
    rA1 = (u16x8_t)(bf16_raw_t)0;

    PREFETCH_TILE(0, 0)
    DEQUANT_TO_LDS(0, 0)
    if(k_steps > 1)
        PREFETCH_TILE(1, 1)
    __syncthreads();

    for(int ks = 0; ks < k_steps; ks += 2)
    {
        PIPE_STEP(0, 1, 0, 1, ks)
        PIPE_STEP(1, 0, 1, 0, ks + 1)
    }

    // Epilogue: plain fp32 stores into this split's slice. No atomics, so
    // the result is bit-deterministic run to run.
    float* out = workspace + (size_t)split * M * N;
#pragma unroll
    for(int e = 0; e < 8; e++)
    {
        const int row = bm + e + laneGroup * 8;
        if(row >= M)
            continue;
        const int c0 = bn + wcol0, c1 = bn + wcol1;
        if(c0 < N)
            out[(size_t)row * N + c0] = acc0[e];
        if(c1 < N)
            out[(size_t)row * N + c1] = acc1[e];
    }
#endif // AITER_A16W4_DEVICE_SUPPORTED
}

// Sum the SPLIT_K fp32 partials and narrow to bf16. SPLIT_K is a
// compile-time constant so the loop fully unrolls.
__global__ void gemm_a16w4_decode_bf16_reduce_kernel(const float* __restrict__ wsp,
                                                     bf16_raw_t* __restrict__ C,
                                                     int MN)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= MN)
        return;
    float s = 0.f;
#pragma unroll
    for(int k = 0; k < SPLIT_K; k++)
        s += wsp[(size_t)k * MN + i];
    C[i] = f32_to_bf16(s);
}

constexpr int kBlockM  = BM;
constexpr int kBlockN  = BN;
constexpr int kBlockK  = BK;
constexpr int kThreads = THREADS;
constexpr int kSplitK  = SPLIT_K;

} // namespace decode_bf16
} // namespace a16w4
} // namespace aiter

#undef PIPE_STEP
#undef WMMA_FROM_LDS
#undef DEQUANT_TO_LDS
#undef PREFETCH_TILE
#undef LDS_TOTAL
#undef LDS_BUF_ELEMS
#undef LDS_BUF_W_ELEMS
#undef LDS_BUF_A_ELEMS
#undef LDS_WK_STRIDE
#undef LDS_A_STRIDE
#undef A_PER_THR
#undef PACKS_PER_THR
#undef GROUP_SIZE
#undef THREADS
#undef NUM_WAVES
#undef WN
#undef WM
#undef BK
#undef BN
#undef BM
#undef SPLIT_K
