#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// a16w4 PREFILL GEMM (bf16 x int4 -> bf16) for gfx1201 -- 128x32 wave tile.
// Tile: BM=128, BN=512, BK=64 | wave 128x32 | 16 waves | 512 threads
// grid = (M/BM, N/BN)
//
// =====================================================================
// WHY THE WAVE SHAPE IS 128x32 AND NOT 64x64
// =====================================================================
// Both shapes have a*b = 16, so both need 128 accumulator VGPRs and both
// give the same WMMA pipe occupancy per K-step:
//
//     wmma_cy = a*b*BK = 16*64 = 1024   (identical for w64x64 and w128x32)
//
// What differs is DEQUANT, which scales with WN and with the number of
// wave ROWS:
//
//   w64x64  : BM/WM = 2 wave rows x 8 wave cols. Both rows cover the SAME
//             512 columns, so both dequantise the SAME B tile -- the work
//             is done twice. WN=64 => 128 elem/lane => 496 VALU/K-step.
//
//   w128x32 : BM/WM = 1 wave row x 16 wave cols. WM == BM, so there is no
//             second row to repeat anything, and each wave owns a
//             distinct 32-column slice. WN=32 => 64 elem/lane
//             => 248 VALU/K-step.
//
// Halving the live B fragments (NT 4 -> 2) also drops register pressure
// below the cliff the 64x64 version fell off. MEASURED, ROCm 7.2 / clang 21:
//
//     w64x64 : VGPR 256, spill 17, private 40 B, 14 scratch insns
//     w128x32: VGPR 210, spill  0, private  0 B,  0 scratch insns
//
// On the older ROCm 6.1 the 64x64 shape showed VGPR 239 spill 0 and looked
// fine -- which is exactly how it shipped and then ran at 2.53x the floor.
//
// =====================================================================
// RESIDENCY DEPENDS ON -fno-slp-vectorize
// =====================================================================
// Two WGs need VGPR <= 192 (8 waves/SIMD sharing 1536). This tile measured
// 210 and missed by 18. But 30 of those registers were SLP splat copies of
// the dequant scalars. Rebuilt with -fno-slp-vectorize:
//
//     w128x32:  VGPR 210 -> 180, spill 0 both ways
//
// 180 rounds to 184 by the allocation granule, and 8 * 184 = 1472 <= 1536,
// so this reaches 2 WG/WGP. THE FLAG IS MANDATORY, NOT A TUNING CHOICE --
// it is set in flags_extra_hip for module_gemm_a16w4. Dropping it is a
// measured 12% regression.
//
// =====================================================================
// REGISTER DISCIPLINE (load-bearing)
// =====================================================================
// The KSUB loop MUST stay rolled. Unrolled, clang 21/22 hoists every
// substep's LDS loads and dequants and spills ~100 registers to scratch,
// reloading the A fragment between WMMA groups at ~200 cy each -- 27% of
// all stall in an ATT trace of the 64x64 version. ROCm 6.1 does NOT do
// this, so a local 6.1 build proves nothing.
//
// ALWAYS CHECK ON THE TARGET TOOLCHAIN, AND CHECK ALL THREE:
//   grep -E 'vgpr_count|vgpr_spill_count|private_segment_fixed_size' x.s
//   grep -c 'scratch_' x.s
// vgpr_spill_count alone reads 0 on a build that is spilling into
// private_segment.

#include "gfx1201/gemm_a16w4_common_gfx1201.cuh"

#define BM 128
#define BN 512
#define BK 64
#define WM 128
#define WN 32

#define WAVES_M (BM / WM)             // 1  <- the point: no repeated dequant
#define WAVES_N (BN / WN)             // 16
#define NUM_WAVES (WAVES_M * WAVES_N) // 16
#define THREADS (NUM_WAVES * 32)      // 512
#define GROUP_SIZE 128

#define MT (WM / 16)  // 8 m-tiles per wave
#define NT (WN / 16)  // 2 n-tiles per wave
#define KSUB (BK / 16) // 4 k-substeps per K-step
#define KPACK (BK / 8) // 8 int32 per column per K-step

// Padded LDS strides, SINGLE buffered.
//   A: stride 72 bf16 = 144 B = 9 x 16 B. gcd(9,8)=1 so the b128 fragment
//      reads across the 16 lanes of a group hit distinct wide banks.
//   W: stride 9 int32. 9n mod 32 is distinct for n = 0..15.
#define LDS_A_STRIDE (BK + 8)           // 72 bf16
#define LDS_W_STRIDE (KPACK + 1)        // 9 int32
#define LDS_A_ELEMS (BM * LDS_A_STRIDE) // 9216 bf16 = 18432 B
#define LDS_W_ELEMS (BN * LDS_W_STRIDE) // 4608 int32 = 18432 B
// Total = 36864 B = 36 KB (double buffering would be 72 KB and NOT fit)

#define A_PER_THR ((BM * BK) / THREADS) // 16 bf16 = 32 B = 2 x b128

// AGROUP = how many A fragments are issued as a batch before their single
// s_wait_dscnt drain. MEASURED, 4096x5120x5120, gfx1201:
//
//   AGROUP  VGPR  drains  WG/WGP     time     TFLOPS
//     1      184      19       2   1752.2 us   122.6   <- DEFAULT, use this
//     2      190      13       2   1737.9 us   123.6      +0.8%, inside spread
//     4      206       9       1   1964.7 us   109.3      -12%
//     8      214       3       1   1882.1 us   114.1      -7%
//
// GROUPING IS A SUBSTITUTE FOR OCCUPANCY, NOT AN ADDITION TO IT. The second
// resident workgroup already runs through the drain stalls, so removing them
// wins only where there is no second workgroup. THE BUDGET IS 8 REGISTERS,
// NOT 72: AGROUP=2 at 190 leaves 2 registers of margin against the 192
// ceiling, which will not survive a compiler update, for a gain inside the
// measurement spread.
#ifndef AGROUP
#define AGROUP 1
#endif

namespace aiter {
namespace a16w4 {
namespace prefill_bf16 {

static_assert(MT % AGROUP == 0, "AGROUP must divide MT");
static_assert(GROUP_SIZE % BK == 0, "GROUP_SIZE must be a multiple of BK");

// One packed int32 == 8 consecutive k for one column, which is EXACTLY the
// 8 elements a lane needs for a 16x16x16 B fragment.
__device__ __forceinline__ frag_bf16x8 dequant8(unsigned packed, float sc, float zs)
{
    u16x8_t o;
#pragma unroll
    for(int j = 0; j < 8; j++)
        o[j] = f32_to_bf16(fmaf((float)((packed >> (j * 4)) & 0xF), sc, zs));
    return as_bf16x8(o);
}

// ─── Pipeline stages ────────────────────────────────────────────────────────

// Issued at the top of the read phase, consumed by STORE_LDS at the bottom
// of the SAME iteration -- one whole WMMA phase later. Nothing loaded here
// may be used here.
#define PREFETCH(KS)                                             \
    {                                                            \
        const int k0_             = (KS)*BK;                     \
        const bf16_raw_t* as_     = A + (size_t)a_row * K + k0_ + a_col; \
        rA0                       = *(const u16x8_t*)(as_);      \
        rA1                       = *(const u16x8_t*)(as_ + 8);  \
        const unsigned* ws_       = W_q + (size_t)(k0_ / 8) * N + w_col; \
        _Pragma("unroll") for(int p = 0; p < KPACK; p++) rW[p] = ws_[(size_t)p * N]; \
    }

// W lands in LDS STILL PACKED. Dequant happens per wave, in registers,
// inside WMMA_STEP -- that is what keeps LDS at 36 KB.
#define STORE_LDS()                                                     \
    {                                                                   \
        *(u16x8_t*)(&lds_a[a_lds_off])     = rA0;                       \
        *(u16x8_t*)(&lds_a[a_lds_off + 8]) = rA1;                       \
        unsigned* wd_                      = &lds_w[w_lds_off];         \
        _Pragma("unroll") for(int p = 0; p < KPACK; p++) wd_[p] = rW[p]; \
    }

// At AGROUP = 1 the scheduling hints are pure cost: there is nothing to
// group, but they still pin the schedule and measured +10 VGPR.
// 0x100 = DS read, 0x008 = MFMA/WMMA.
#if AGROUP > 1
#define SCHED_A_GROUP()                                             \
    do                                                              \
    {                                                               \
        __builtin_amdgcn_sched_group_barrier(0x100, AGROUP, 0);     \
        __builtin_amdgcn_sched_group_barrier(0x008, AGROUP* NT, 0); \
    } while(0)
#else
#define SCHED_A_GROUP() \
    do                  \
    {                   \
    } while(0)
#endif

// THE KSUB LOOP MUST STAY ROLLED -- see the header. Rolled, only ONE
// substep is live: NT B-fragments + AGROUP streamed A-fragments. B is the
// hoisted operand because it is the one that costs a dequant; re-deriving
// it per i would multiply VALU by MT = 8.
#define WMMA_STEP(SC)                                                              \
    {                                                                              \
        _Pragma("unroll 1") for(int s = 0; s < KSUB; s++)                          \
        {                                                                          \
            frag_bf16x8 bf_[NT];                                                   \
            _Pragma("unroll") for(int j = 0; j < NT; j++) bf_[j] = dequant8(       \
                lds_w[(n_base + j * 16) * LDS_W_STRIDE + 2 * s + laneGroup],       \
                sc_f[SC][j],                                                       \
                zs_f[SC][j]);                                                      \
            _Pragma("unroll") for(int g = 0; g < MT / AGROUP; g++)                 \
            {                                                                      \
                frag_bf16x8 af_[AGROUP];                                           \
                _Pragma("unroll") for(int a = 0; a < AGROUP; a++) af_[a] =         \
                    as_bf16x8(*(const u16x8_t*)(&lds_a[(m_base + (g * AGROUP + a) * 16) * \
                                                           LDS_A_STRIDE +          \
                                                       s * 16 + laneGroup * 8]));  \
                _Pragma("unroll") for(int a = 0; a < AGROUP; a++)                  \
                    _Pragma("unroll") for(int j = 0; j < NT; j++) acc[g * AGROUP + a][j] = \
                        __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(         \
                            af_[a], bf_[j], acc[g * AGROUP + a][j]);               \
                SCHED_A_GROUP();                                                   \
            }                                                                      \
        }                                                                          \
    }

// SC is the buffer being FILLED, not the one consumed now. One base pointer
// plus immediate offsets -- an address per n-tile costs ~16 VGPRs of pure
// addressing, which shows up in a trace as v_add_co_u32 pairs before every
// scale load.
#define LOAD_SCALES(SC, KS)                                                     \
    {                                                                           \
        const size_t goff_     = (size_t)(((KS)*BK) / GROUP_SIZE) * N + n_col0; \
        const bf16_raw_t* sp_  = scales + goff_;                                \
        const bf16_raw_t* zp_  = zeros + goff_;                                 \
        _Pragma("unroll") for(int j = 0; j < NT; j++)                           \
        {                                                                       \
            const float s_ = bf16_to_f32(sp_[j * 16]);                          \
            sc_f[SC][j]    = s_;                                                \
            zs_f[SC][j]    = -bf16_to_f32(zp_[j * 16]) * s_;                    \
        }                                                                       \
    }

// Single-buffered step: read phase, drain readers, write phase, drain
// writers. Both barriers are mandatory with one buffer.
// NOTE both __syncthreads() are UNCONDITIONAL. STORE_LDS() expands to a
// braced block, so `if(more_) STORE_LDS()` ends at the closing brace and the
// barrier below it is reached by every thread. Making the second barrier
// conditional would deadlock on the final K-step.
#define PIPE_STEP(SC_CUR, SC_NXT, KS)                      \
    {                                                      \
        const int ks_    = (KS);                           \
        const bool more_ = (ks_ + 1 < k_steps);            \
        if(more_)                                          \
        {                                                  \
            PREFETCH(ks_ + 1) LOAD_SCALES(SC_NXT, ks_ + 1) \
        }                                                  \
        WMMA_STEP(SC_CUR)                                  \
        __syncthreads();                                   \
        if(more_)                                          \
            STORE_LDS()                                    \
        __syncthreads();                                   \
    }

// ─── Kernel ─────────────────────────────────────────────────────────────────

__global__ __launch_bounds__(THREADS) void gemm_a16w4_prefill_bf16_kernel(
    const bf16_raw_t* __restrict__ A,     // [M, K] bf16
    const unsigned int* __restrict__ W_q, // [K/8, N] int32, packed along K
    const bf16_raw_t* __restrict__ scales, // [K/G, N] bf16
    const bf16_raw_t* __restrict__ zeros,  // [K/G, N] bf16
    bf16_raw_t* __restrict__ C,            // [M, N] bf16
    int M,
    int N,
    int K)
{
    AITER_A16W4_REQUIRE_GFX1201();
#if AITER_A16W4_DEVICE_SUPPORTED
    // M unused: supported() enforces M % BM == 0, so every row this workgroup
    // writes is in range. A ragged-M path would need a predicate.
    (void)M;
    const int bm          = blockIdx.x * BM;
    const int bn          = blockIdx.y * BN;
    const int tid         = threadIdx.x;
    const int wave        = tid / 32;
    const int lane        = tid % 32;
    const int laneWrapped = lane % 16;
    const int laneGroup   = lane / 16;

    // WAVES_M == 1, so wave_m is always 0 and every wave owns the full BM
    // rows and a distinct 32-column slice.
    const int wave_n = wave % WAVES_N; // 0..15

    // BOTH bases carry laneWrapped. For wmma_*_w32 the lane's free index is
    // lane%16 on BOTH operands: A gives lane A[m][k] with m = lane%16, B
    // gives lane B[k][n] with n = lane%16, and laneGroup picks the k-half on
    // both. Dropping it from B collapses the B tile to 16 identical columns
    // and reads as cosine ~0.51 rather than as a crash.
    const int m_base = laneWrapped;
    const int n_base = wave_n * WN + laneWrapped;
    // First global column this lane owns; the other n-tile is +16. Must
    // agree with n_base, or the right scale lands on the wrong weights.
    const int n_col0 = bn + n_base;

    // ---- staging assignments, hoisted out of the loop ----
    // A: 128 rows x 4 chunks of 16 bf16 = 512 threads, 2 x b128 each.
    // LDS byte offset 144*a_m + 2*a_col is 16 B aligned since 144 = 9*16.
    const int a_m       = tid / (BK / A_PER_THR);
    const int a_col     = (tid % (BK / A_PER_THR)) * A_PER_THR;
    const int a_row     = bm + a_m;
    const int a_lds_off = a_m * LDS_A_STRIDE + a_col;
    // W: BN == THREADS, so thread t owns exactly column t. Each of the KPACK
    // loads is strided by N in global, which is coalesced ACROSS the wave
    // (32 lanes x 4 B = one 128 B line).
    const int w_col     = bn + tid;
    const int w_lds_off = tid * LDS_W_STRIDE;

    __shared__ bf16_raw_t lds_a[LDS_A_ELEMS];
    __shared__ unsigned lds_w[LDS_W_ELEMS];

    frag_f32x8 acc[MT][NT];
#pragma unroll
    for(int i = 0; i < MT; i++)
#pragma unroll
        for(int j = 0; j < NT; j++)
            acc[i][j] = (frag_f32x8)0.f;

    float sc_f[2][NT], zs_f[2][NT];
    u16x8_t rA0, rA1;
    unsigned rW[KPACK];

    const int k_steps = K / BK;

    PREFETCH(0)
    STORE_LDS()
    LOAD_SCALES(0, 0)
    __syncthreads();

    // Unrolled x2 only so the scale buffer index is a compile-time constant.
    for(int ks = 0; ks < k_steps; ks += 2)
    {
        PIPE_STEP(0, 1, ks)
        PIPE_STEP(1, 0, ks + 1)
    }

    // ---- epilogue ----
    // For a 16x16 D tile in w32: lane holds D[m][n] with n = lane%16 and
    // m = (lane/16)*8 + e. NOTE this INVERTS the input convention -- the m
    // that came from lane%16 on the A operand arrives from lane/16 here.
#pragma unroll
    for(int i = 0; i < MT; i++)
    {
        const int row = bm + i * 16 + laneGroup * 8;
#pragma unroll
        for(int j = 0; j < NT; j++)
        {
            const int col = n_col0 + j * 16;
#pragma unroll
            for(int e = 0; e < 8; e++)
                C[(size_t)(row + e) * N + col] = f32_to_bf16(acc[i][j][e]);
        }
    }
#endif // AITER_A16W4_DEVICE_SUPPORTED
}

// Tile geometry, re-exported so the host side can keep using it after the
// macros below are undefined.
constexpr int kBlockM  = BM;
constexpr int kBlockN  = BN;
constexpr int kBlockK  = BK;
constexpr int kThreads = THREADS;

} // namespace prefill_bf16
} // namespace a16w4
} // namespace aiter

// Undefined so several a16w4 kernel headers can coexist in one translation
// unit: every one of these names is redefined with a different value by the
// decode header.
#undef PIPE_STEP
#undef LOAD_SCALES
#undef WMMA_STEP
#undef SCHED_A_GROUP
#undef STORE_LDS
#undef PREFETCH
// AGROUP is deliberately NOT undefined: it is an #ifndef override knob, and
// dropping it here would silently ignore a -DAGROUP passed to the build for
// every header included after this one.
#undef A_PER_THR
#undef LDS_W_ELEMS
#undef LDS_A_ELEMS
#undef LDS_W_STRIDE
#undef LDS_A_STRIDE
#undef KPACK
#undef KSUB
#undef NT
#undef MT
#undef GROUP_SIZE
#undef THREADS
#undef NUM_WAVES
#undef WAVES_N
#undef WAVES_M
#undef WN
#undef WM
#undef BK
#undef BN
#undef BM
