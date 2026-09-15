#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// a16w4 DECODE GEMM (fp16 x int4 -> fp16) for gfx1201 -- grid-level split-K
// with the fp16 bit-magic dequant.
// Tile: BM=16, BN=128, BK=64, WM=16, WN=32 | 4 waves/WG | 128 threads
// grid = (ceil(M/BM), N/BN, SPLIT_K)
//
// Same split-K structure as the bf16 decode kernel -- see that file for why
// S=2 and why the dequant is staged through LDS rather than fed live into the
// WMMA operand. Two things differ:
//
//   * the dequant is the fp16 bit-magic (see dequant8_magic below), which
//     needs MAGIC-permuted nibbles and +1024-biased zeros from the host;
//   * LDS_WK_STRIDE is 72, NOT the bf16 kernel's 68. See the note on the
//     define -- it is a 2.4x difference and it is invisible to any
//     numerical test.

#include "gfx1201/gemm_a16w4_common_gfx1201.cuh"

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

#define LDS_A_STRIDE (BK + 8) // 72 elems, b128 aligned
// 72 elems = 144 B. MUST be a multiple of 8 fp16 (16 B), NOT the bf16
// kernel's 68. This kernel reads and writes the W region with 16-byte LDS
// accesses (ds_load_b128 / ds_store_b128) because the dequantised fragment
// is a u32x4/f16x8. At stride 68 (136 B) those accesses are only 8-byte
// aligned and the hardware serialises them:
//     stride 68 -> 60.96 us / 183 GB/s
//     stride 72 -> 25.05 us / 446 GB/s   <-- 2.4x, from alignment alone
//     stride 80 -> 34.89 us / 320 GB/s   (aligned, but 8-way bank conflict)
//     stride 88 -> 30.40 us / 367 GB/s   (aligned, 2-way conflict)
// The bf16 kernel gets away with 68 only because clang splits its W accesses
// into ds_load_2addr_b64, which needs just 8-byte alignment. Correctness is
// IDENTICAL at every stride, so this is invisible to any numerical test.
#define LDS_WK_STRIDE (BK + 8) // 72 elems, b128 aligned

#define LDS_BUF_A_ELEMS (BM * LDS_A_STRIDE)
#define LDS_BUF_W_ELEMS (BN * LDS_WK_STRIDE)
#define LDS_BUF_ELEMS (LDS_BUF_A_ELEMS + LDS_BUF_W_ELEMS)
#define LDS_TOTAL (2 * LDS_BUF_ELEMS * 2) // 41472 B

namespace aiter {
namespace a16w4 {
namespace decode_fp16 {

static_assert(GROUP_SIZE % BK == 0, "GROUP_SIZE must be a multiple of BK");

// ─── The bit-magic ──────────────────────────────────────────────────────────
// One packed int32 (8 pre-permuted nibbles) -> 8 dequantised fp16, written as
// 4 dwords. sc2 is the scale and zm2 is the MAGIC-BIASED zero (1024 + z),
// both broadcast into the two halves so each op handles two weights.
//
// SUBTRACT BEFORE SCALING. THIS ORDER IS NOT NEGOTIABLE.
//   fused form   (1024+n)*s + (-(1024+z)*s)   -- ONE v_pk_fma_f16, BUT WRONG
//   correct form ((1024+n) - (1024+z)) * s    -- pk_add + pk_mul
// The fused form subtracts two quantities of magnitude ~1024*s to produce a
// result of magnitude ~15*s, amplifying fp16's 2^-11 relative error by
// 1024/15 ~= 68 to about 3.3%. Measured: cosine 0.99947, max_err 0.40,
// 36% of elements over 5% error. It looks "almost right", which is the
// worst way for a kernel to be wrong.
//
// The correct form is EXACT in the subtraction: both operands lie in the
// binade [1024, 2048), so by Sterbenz's lemma their fp16 difference is
// representable with no rounding at all. The integer n - z in [-15, 15]
// comes out bit-exact, and the single multiply by s is the ONLY rounding in
// the whole dequant.
__device__ __forceinline__ u32x4_t dequant8_magic(unsigned packed, f16x2 sc2, f16x2 zm2)
{
    u32x4_t out;
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        // (1024 + n) for two nibbles, by OR alone -- no cvt instruction.
        const unsigned bits = ((packed >> (4 * j)) & kNibMask) | kF16Magic;
        // v_pk_add_f16 (exact) then v_pk_mul_f16. Written as (a-b)*c so that
        // -ffp-contract=fast cannot refactor it back into the fused form.
        out[j] = as_u32((as_f16x2(bits) - zm2) * sc2);
    }
    return out;
}

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

// Staged dequant: registers -> LDS, exactly as the bf16 kernel.
//
// The zero point arrives ALREADY BIASED by 1024 from the host, so it is used
// as a raw fp16 broadcast: no unpack to f32, no folding into the scale.
#define DEQUANT_TO_LDS(R, B)                                                    \
    {                                                                           \
        const f16x2 sc2_ = as_f16x2((unsigned)rsc##R * 0x00010001u);            \
        const f16x2 zm2_ = as_f16x2((unsigned)rzr##R * 0x00010001u);            \
        *(u16x8_t*)(&lds_buf[B][a_lds_off]) = rA##R;                            \
        fp16_raw_t* wcol_ = &lds_buf[B][LDS_BUF_A_ELEMS + tid * LDS_WK_STRIDE]; \
        _Pragma("unroll") for(int p = 0; p < PACKS_PER_THR; p++)                \
        {                                                                       \
            /* 8 fp16 in k order, because the host pre-permuted the nibbles */  \
            *(u32x4_t*)(wcol_ + p * 8) = dequant8_magic(rW##R[p], sc2_, zm2_);  \
        }                                                                       \
    }

#define WMMA_FROM_LDS(B)                                                              \
    {                                                                                 \
        const fp16_raw_t* srcA_ = &lds_buf[B][0];                                     \
        const fp16_raw_t* srcW_ = &lds_buf[B][LDS_BUF_A_ELEMS];                       \
        _Pragma("unroll") for(int s = 0; s < BK / 16; s++)                            \
        {                                                                             \
            const int koff_ = s * 16 + laneGroup * 8;                                 \
            f16x8 af_       = as_f16x8(                                               \
                *(const u16x8_t*)(srcA_ + laneWrapped * LDS_A_STRIDE + koff_));       \
            f16x8 bf0_ =                                                              \
                as_f16x8(*(const u16x8_t*)(srcW_ + wcol0 * LDS_WK_STRIDE + koff_));   \
            f16x8 bf1_ =                                                              \
                as_f16x8(*(const u16x8_t*)(srcW_ + wcol1 * LDS_WK_STRIDE + koff_));   \
            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af_, bf0_, acc0); \
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af_, bf1_, acc1); \
        }                                                                             \
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

__global__ __launch_bounds__(THREADS) void gemm_a16w4_decode_fp16_kernel(
    const fp16_raw_t* __restrict__ A,      // [M, K] fp16
    const unsigned int* __restrict__ W_q,  // [K/8, N] int32, MAGIC-permuted
    const fp16_raw_t* __restrict__ scales, // [K/G, N] fp16
    const fp16_raw_t* __restrict__ zeros,  // [K/G, N] fp16, BIASED by +1024
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

    const int k_span  = K / SPLIT_K;
    const int k_begin = split * k_span;
    const int k_steps = k_span / BK;

    __shared__ fp16_raw_t lds_buf[2][LDS_BUF_ELEMS];

    frag_f32x8 acc0 = {0}, acc1 = {0};

    const unsigned int* W_col = W_q + (size_t)(k_begin / 8) * N + bn + tid;
    const fp16_raw_t* sc_base = scales + (size_t)(k_begin / GROUP_SIZE) * N;
    const fp16_raw_t* zr_base = zeros + (size_t)(k_begin / GROUP_SIZE) * N;

    const int a_row         = (tid * A_PER_THR) / BK;
    const int a_col         = (tid * A_PER_THR) % BK;
    const bool a_ok         = (bm + a_row) < M;
    const int a_row_clamped = a_ok ? (bm + a_row) : 0;
    const fp16_raw_t* a_src = A + (long)a_row_clamped * K + k_begin + a_col;
    const int a_lds_off     = a_row * LDS_A_STRIDE + a_col;

    unsigned rW0[PACKS_PER_THR], rW1[PACKS_PER_THR];
    u16x8_t rA0, rA1;
    fp16_raw_t rsc0, rzr0, rsc1, rzr1;

    rA0 = (u16x8_t)(fp16_raw_t)0;
    rA1 = (u16x8_t)(fp16_raw_t)0;

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

__global__ void gemm_a16w4_decode_fp16_reduce_kernel(const float* __restrict__ wsp,
                                                     fp16_raw_t* __restrict__ C,
                                                     int MN)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= MN)
        return;
    float s = 0.f;
#pragma unroll
    for(int k = 0; k < SPLIT_K; k++)
        s += wsp[(size_t)k * MN + i];
    C[i] = f32_to_fp16(s);
}

constexpr int kBlockM  = BM;
constexpr int kBlockN  = BN;
constexpr int kBlockK  = BK;
constexpr int kThreads = THREADS;
constexpr int kSplitK  = SPLIT_K;

} // namespace decode_fp16
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
