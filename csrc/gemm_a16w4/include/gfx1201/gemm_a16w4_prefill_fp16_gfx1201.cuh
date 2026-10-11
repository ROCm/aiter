#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// a16w4 PREFILL GEMM (fp16 x int4 -> fp16) for gfx1201 -- bit-magic dequant.
// Tile: BM=128, BN=512, BK=64 | wave 128x32 | 16 waves | 512 threads
// grid = (M/BM, N/BN)
//
// Same tile and same pipeline as the bf16 prefill kernel; the only change is
// the dequant, which uses the fp16 bit-magic instead of an int->float convert
// plus fma. gfx1201 has NO bf16 ALU (v_pk_add_bf16 / v_pk_mul_bf16 /
// v_pk_fma_bf16 are all absent from the ISA), so the packed-math path only
// exists for fp16 -- which is also what the AWQ checkpoints this targets
// actually ship (Qwen3-32B-AWQ config.json says torch_dtype: float16).
//
// TWO HOST-SIDE INVARIANTS, BOTH OF WHICH FAIL SILENTLY IF VIOLATED:
//   * the nibble for row k must sit at bit p(k) = 4*(k>>1) + 16*(k&1), so the
//     four extractions j = 0..3 yield k = 0..7 in order. This is a DIFFERENT
//     packing from the bf16 kernels and the two are NOT interchangeable;
//   * `zeros` must arrive with +1024 already added.
// aiter/ops/gemm_op_a16w4.py owns both conversions.
//
// BUILD FLAG: -fno-slp-vectorize is kept for parity with the bf16 prefill
// build. It is less critical here -- the packed-fp16 dequant has no scalar
// broadcast for SLP to splat -- but this kernel sits at 184 VGPR against a
// 192 ceiling for 2 workgroups per WGP, so there is no headroom to gamble
// with and the flag costs nothing.

#include "gfx1201/gemm_a16w4_common_gfx1201.cuh"

#define BM 128
#define BN 512
#define BK 64
#define WM 128
#define WN 32

#define WAVES_M (BM / WM)             // 1  <- no repeated dequant
#define WAVES_N (BN / WN)             // 16
#define NUM_WAVES (WAVES_M * WAVES_N) // 16
#define THREADS (NUM_WAVES * 32)      // 512
#define GROUP_SIZE 128

#define MT (WM / 16)   // 8 m-tiles per wave
#define NT (WN / 16)   // 2 n-tiles per wave
#define KSUB (BK / 16) // 4 k-substeps per K-step
#define KPACK (BK / 8) // 8 int32 per column per K-step

#define LDS_A_STRIDE (BK + 8)           // 72 fp16
#define LDS_W_STRIDE (KPACK + 1)        // 9 int32
#define LDS_A_ELEMS (BM * LDS_A_STRIDE) // 9216 fp16 = 18432 B
#define LDS_W_ELEMS (BN * LDS_W_STRIDE) // 4608 int32 = 18432 B
// Total = 36864 B = 36 KB

#define A_PER_THR ((BM * BK) / THREADS) // 16 fp16 = 32 B = 2 x b128

#ifndef AGROUP
#define AGROUP 1
#endif

namespace aiter {
namespace a16w4 {
namespace prefill_fp16 {

static_assert(MT % AGROUP == 0, "AGROUP must divide MT");
// A K-step must lie inside one quantisation group, or one WMMA_STEP would
// need two different scales for the same B fragment.
static_assert(GROUP_SIZE % BK == 0, "GROUP_SIZE must be a multiple of BK");

// One packed int32 (8 pre-permuted nibbles) -> 8 dequantised fp16, in k order.
// No cvt, no perm: the OR with 0x6400 IS the int->float conversion.
//
// SUBTRACT BEFORE SCALING. THIS ORDER IS NOT NEGOTIABLE.
//   fused form   (1024+n)*s + (-(1024+z)*s)   -- ONE v_pk_fma_f16, BUT WRONG
//   correct form ((1024+n) - (1024+z)) * s    -- pk_add + pk_mul
// The fused form subtracts two quantities of magnitude ~1024*s to produce a
// result of magnitude ~15*s, amplifying fp16's 2^-11 relative error by
// 1024/15 ~= 68 to about 3.3%. Measured: cosine 0.99944, max_err 0.53,
// 37% of elements over 5% error. It looks "almost right", which is the
// worst way for a kernel to be wrong.
//
// The correct form is EXACT in the subtraction: both operands lie in the
// binade [1024, 2048), so by Sterbenz's lemma their fp16 difference is
// representable with no rounding at all. The integer n - z in [-15, 15]
// comes out bit-exact, and the single multiply by s is the ONLY rounding.
__device__ __forceinline__ f16x8 dequant8_magic(unsigned packed, f16x2 sc2, f16x2 zm2)
{
    u32x4_t o;
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        const unsigned bits = ((packed >> (4 * j)) & kNibMask) | kF16Magic;
        // v_pk_add_f16 (exact) then v_pk_mul_f16. Written as (a-b)*c so that
        // -ffp-contract=fast cannot refactor it back into the fused form.
        o[j] = as_u32((as_f16x2(bits) - zm2) * sc2);
    }
    return as_f16x8_u32x4(o);
}

// ─── Pipeline stages ────────────────────────────────────────────────────────

#define PREFETCH(KS)                                                                 \
    {                                                                                \
        const int k0_         = (KS)*BK;                                             \
        const fp16_raw_t* as_ = A + (size_t)a_row * K + k0_ + a_col;                 \
        rA0                   = *(const u16x8_t*)(as_);                              \
        rA1                   = *(const u16x8_t*)(as_ + 8);                          \
        const unsigned* ws_   = W_q + (size_t)(k0_ / 8) * N + w_col;                 \
        _Pragma("unroll") for(int p = 0; p < KPACK; p++) rW[p] = ws_[(size_t)p * N]; \
    }

// W lands in LDS STILL PACKED; dequant is per wave, in registers, inside
// WMMA_STEP. That is what keeps LDS at 36 KB rather than 4x that.
#define STORE_LDS()                                                     \
    {                                                                   \
        *(u16x8_t*)(&lds_a[a_lds_off])     = rA0;                       \
        *(u16x8_t*)(&lds_a[a_lds_off + 8]) = rA1;                       \
        unsigned* wd_                      = &lds_w[w_lds_off];         \
        _Pragma("unroll") for(int p = 0; p < KPACK; p++) wd_[p] = rW[p]; \
    }

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

// THE KSUB LOOP MUST STAY ROLLED. Unrolled, clang 21/22 hoists every
// substep's fragments and spills ~100 registers to scratch. ROCm 6.1 does
// NOT reproduce this, so a 6.1 build proves nothing.
#define WMMA_STEP(SC)                                                                      \
    {                                                                                      \
        _Pragma("unroll 1") for(int s = 0; s < KSUB; s++)                                  \
        {                                                                                  \
            f16x8 bf_[NT];                                                                 \
            _Pragma("unroll") for(int j = 0; j < NT; j++) bf_[j] = dequant8_magic(         \
                lds_w[(n_base + j * 16) * LDS_W_STRIDE + 2 * s + laneGroup],               \
                sc2_f[SC][j],                                                              \
                zm2_f[SC][j]);                                                             \
            _Pragma("unroll") for(int g = 0; g < MT / AGROUP; g++)                         \
            {                                                                              \
                f16x8 af_[AGROUP];                                                         \
                _Pragma("unroll") for(int a = 0; a < AGROUP; a++) af_[a] =                 \
                    as_f16x8(*(const u16x8_t*)(&lds_a[(m_base + (g * AGROUP + a) * 16) *   \
                                                          LDS_A_STRIDE +                   \
                                                      s * 16 + laneGroup * 8]));           \
                _Pragma("unroll") for(int a = 0; a < AGROUP; a++)                          \
                    _Pragma("unroll") for(int j = 0; j < NT; j++) acc[g * AGROUP + a][j] = \
                        __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(                  \
                            af_[a], bf_[j], acc[g * AGROUP + a][j]);                       \
                SCHED_A_GROUP();                                                           \
            }                                                                              \
        }                                                                                  \
    }

// SC is the buffer being FILLED, not the one consumed now. The scale and the
// (already +1024-biased) zero are broadcast into both halves ONCE per K-step,
// so the inner loop sees them as ready packed operands. Both are used as raw
// fp16 -- no unpack to f32, no folding of the scale into the zero (see the
// Sterbenz note on dequant8_magic for why that folding is forbidden).
#define LOAD_SCALES(SC, KS)                                                            \
    {                                                                                  \
        const size_t goff_    = (size_t)(((KS)*BK) / GROUP_SIZE) * N + n_col0;         \
        const fp16_raw_t* sp_ = scales + goff_;                                        \
        const fp16_raw_t* zp_ = zeros + goff_;                                         \
        _Pragma("unroll") for(int j = 0; j < NT; j++)                                  \
        {                                                                              \
            sc2_f[SC][j] = as_f16x2((unsigned)sp_[j * 16] * 0x00010001u);              \
            zm2_f[SC][j] = as_f16x2((unsigned)zp_[j * 16] * 0x00010001u);              \
        }                                                                              \
    }

// Both __syncthreads() are UNCONDITIONAL -- STORE_LDS() expands to a braced
// block, so the barrier after it is reached by every thread.
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

__global__ __launch_bounds__(THREADS) void gemm_a16w4_prefill_fp16_kernel(
    const fp16_raw_t* __restrict__ A,      // [M, K] fp16
    const unsigned int* __restrict__ W_q,  // [K/8, N] int32, MAGIC-permuted
    const fp16_raw_t* __restrict__ scales, // [K/G, N] fp16
    const fp16_raw_t* __restrict__ zeros,  // [K/G, N] fp16, BIASED by +1024
    fp16_raw_t* __restrict__ C,            // [M, N] fp16
    int M,
    int N,
    int K)
{
    AITER_A16W4_REQUIRE_GFX1201();
#if AITER_A16W4_DEVICE_SUPPORTED
    (void)M; // supported() enforces M % BM == 0
    const int bm          = blockIdx.x * BM;
    const int bn          = blockIdx.y * BN;
    const int tid         = threadIdx.x;
    const int wave        = tid / 32;
    const int lane        = tid % 32;
    const int laneWrapped = lane % 16;
    const int laneGroup   = lane / 16;

    const int wave_n = wave % WAVES_N;

    // BOTH bases carry laneWrapped. For wmma_*_w32 the lane's free index is
    // lane%16 on BOTH operands. Dropping it from B collapses the B tile to 16
    // identical columns and reads as cosine ~0.51 rather than as a crash.
    const int m_base = laneWrapped;
    const int n_base = wave_n * WN + laneWrapped;
    const int n_col0 = bn + n_base;

    const int a_m       = tid / (BK / A_PER_THR);
    const int a_col     = (tid % (BK / A_PER_THR)) * A_PER_THR;
    const int a_row     = bm + a_m;
    const int a_lds_off = a_m * LDS_A_STRIDE + a_col;
    const int w_col     = bn + tid;
    const int w_lds_off = tid * LDS_W_STRIDE;

    __shared__ fp16_raw_t lds_a[LDS_A_ELEMS];
    __shared__ unsigned lds_w[LDS_W_ELEMS];

    frag_f32x8 acc[MT][NT];
#pragma unroll
    for(int i = 0; i < MT; i++)
#pragma unroll
        for(int j = 0; j < NT; j++)
            acc[i][j] = (frag_f32x8)0.f;

    f16x2 sc2_f[2][NT], zm2_f[2][NT];
    u16x8_t rA0, rA1;
    unsigned rW[KPACK];

    const int k_steps = K / BK;

    PREFETCH(0)
    STORE_LDS()
    LOAD_SCALES(0, 0)
    __syncthreads();

    for(int ks = 0; ks < k_steps; ks += 2)
    {
        PIPE_STEP(0, 1, ks)
        PIPE_STEP(1, 0, ks + 1)
    }

    // For a 16x16 D tile in w32: lane holds D[m][n] with n = lane%16 and
    // m = (lane/16)*8 + e. NOTE this INVERTS the input convention.
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
                C[(size_t)(row + e) * N + col] = f32_to_fp16(acc[i][j][e]);
        }
    }
#endif // AITER_A16W4_DEVICE_SUPPORTED
}

constexpr int kBlockM  = BM;
constexpr int kBlockN  = BN;
constexpr int kBlockK  = BK;
constexpr int kThreads = THREADS;

} // namespace prefill_fp16
} // namespace a16w4
} // namespace aiter

#undef PIPE_STEP
#undef LOAD_SCALES
#undef WMMA_STEP
#undef SCHED_A_GROUP
#undef STORE_LDS
#undef PREFETCH
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
