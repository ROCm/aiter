// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 path of fused_qk_norm_rope_group_quant for the DeepSeek-V4 shape:
// bf16 in, head_dim 512 with a 64-wide GPT-J RoPE tail, fp8 e4m3 Q and K NoPE
// with a 1x64 e8m0 scale written inline (twice), bf16 RoPE outputs, and the
// optional SWA copy of the K row. Its outputs are bit-identical to the coarse
// kernel in fused_qk_norm_rope_cache_quant.cu: same row-sum order, same e8m0
// rounding, same RoPE fma shape.
//
// Why a separate kernel. The coarse kernel gives each wave one token and walks
// its heads one row at a time, so every row pays a full load -> 32-lane reduce
// -> quant -> store chain before the next starts; at decode sizes that latency
// is the whole kernel. Here one wave32 takes HPP heads of one token, the TDM
// engine copies all HPP rows into LDS at once, and RPS rows go through each
// step together, so their reductions overlap. The layout follows the Gluon
// kernel aiter/ops/triton/_gluon_kernels/gfx1250/fusions/
// fused_qk_norm_rope_group_quant.py; measured against it, this one is faster
// from T=1 up (e.g. H=128: 0.93x at T=1, 0.60x at T=512, 0.95x at T=16384).
//
// Grid: (1 + num_heads / HPP, num_tokens), one wave32 per workgroup. x == 0 is
// the token's K row (and its SWA copy); x >= 1 is a group of HPP Q heads.
// Lane l owns row elements [16l, 16l + 16): lanes 0..27 the NoPE part, 28..31
// the RoPE part. RoPE reloads the 64 RoPE columns two per lane (one GPT-J
// pair), so all 32 lanes share it instead of four.
#pragma once

#include <cstdint>

#include "opus/opus.hpp"

namespace aiter {
namespace qknr_gfx1250 {

constexpr int kHead          = 512;
constexpr int kRope          = 64;
constexpr int kNope          = kHead - kRope; // 448
constexpr int kGroup         = 64;
constexpr int kLanes         = 32;            // wave32
constexpr int kVec           = kHead / kLanes; // 16 elements per lane per row
constexpr int kNopeLanes     = kNope / kVec;   // 28
constexpr int kLanesPerGroup = kGroup / kVec;  // 4
constexpr int kRowDwords     = kHead / 2;      // one bf16 row in dwords
// TDM cache policy: regular temporal hint, CU scope. opus defaults to device
// scope, which measured ~3% slower at T=4096 H=128.
constexpr int kTdmCachePolicy = 0;

typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef float f32x16 __attribute__((ext_vector_type(16)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef __bf16 bf16x2_t __attribute__((ext_vector_type(2)));

// The launch never uses clusters. Saying so lets gfx1250 read the workgroup id
// straight from ttmp9/ttmp7 instead of rebuilding it from the cluster id. The
// attribute is an error on other targets, hence device-pass only.
#if defined(__gfx1250__)
#define AITER_QKNR_GFX1250_NO_CLUSTER __attribute__((cluster_dims(1, 1, 1)))
#else
#define AITER_QKNR_GFX1250_NO_CLUSTER
#endif

#if defined(__gfx1250__)
// Every DPP pattern below reads an in-range lane, so bound_ctrl and the zero
// `old` change no value. They let the move fold into the consuming VALU op.
template <int CTRL>
__device__ __forceinline__ float mov_dpp_f(float v)
{
    return __builtin_bit_cast(
        float,
        __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL, 0xf, 0xf, true));
}

// Sum over the 32 lanes, in every lane. The same butterfly as wave_reduce in
// hip_reduce.h, so the row sums round exactly as the coarse kernel's do.
__device__ __forceinline__ float wave_sum(float v)
{
    v = mov_dpp_f<0xb1>(v) + v;  // quad_perm [1,0,3,2]
    v = mov_dpp_f<0x4e>(v) + v;  // quad_perm [2,3,0,1]
    v = mov_dpp_f<0x167>(v) + v; // row_xmask:7
    v = mov_dpp_f<0x16f>(v) + v; // row_xmask:15
    v = __builtin_bit_cast(float,
                           __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, v),
                                                        __builtin_bit_cast(int, v),
                                                        0x76543210u,
                                                        0xfedcba98u,
                                                        true,
                                                        true)) +
        v;
    return v;
}

// Max over each aligned group of 4 lanes, i.e. one 64-element quant group.
__device__ __forceinline__ float max_quad(float v)
{
    v = fmaxf(v, mov_dpp_f<0xb1>(v));
    v = fmaxf(v, mov_dpp_f<0x4e>(v));
    return v;
}

__device__ __forceinline__ float bf16_lo(uint32_t w) { return __builtin_bit_cast(float, w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w)
{
    return __builtin_bit_cast(float, w & 0xffff0000u);
}

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b)
{
    return __builtin_bit_cast(uint32_t, __builtin_convertvector(f32x2{a, b}, bf16x2_t));
}

// fp_f32_to_e8m0_block_scale<RoundUp, FP8_E4M3>(amax).byte, without its
// branches: amax >= 0 leaves the sign bit clear, so adding 0x7fffff carries
// into the exponent exactly when the mantissa is non-zero, and the min keeps an
// Inf/NaN exponent at 0xff the way its `e < 0xff` guard does.
__device__ __forceinline__ uint32_t e8m0_byte(float amax)
{
    const uint32_t u = __builtin_bit_cast(uint32_t, amax * (1.0f / 448.0f));
    const uint32_t e = (u + 0x7fffffu) >> 23;
    return e < 0xffu ? e : 0xffu;
}

// RMSNorm, e8m0 fp8 NoPE with the inline scale, GPT-J RoPE, and the stores for
// RPS rows already in LDS. K passes its weight; Q runs weightless.
//   lds:          first of the RPS rows (bf16 pairs)
//   nope, rope:   first destination row; rows are kHead bytes / kRope bf16 apart
//   nope2, rope2: the SWA copy, written only when write2
template <int RPS, bool K_ROW>
__device__ __forceinline__ void step_rows(const uint32_t* __restrict__ lds,
                                          int lane,
                                          float eps,
                                          const float (&w)[kVec],
                                          float wr0,
                                          float wr1,
                                          float c,
                                          float s,
                                          uint8_t* __restrict__ nope,
                                          uint32_t* __restrict__ rope,
                                          uint8_t* __restrict__ nope2,
                                          uint32_t* __restrict__ rope2,
                                          bool write2)
{
    f32x16 x[RPS];
    uint32_t xr[RPS];
#pragma unroll
    for(int r = 0; r < RPS; ++r)
    {
        const u32x4* src =
            reinterpret_cast<const u32x4*>(lds + r * kRowDwords + lane * (kVec / 2));
        const u32x4 a = src[0];
        const u32x4 b = src[1];
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            x[r][2 * i]         = bf16_lo(a[i]);
            x[r][2 * i + 1]     = bf16_hi(a[i]);
            x[r][8 + 2 * i]     = bf16_lo(b[i]);
            x[r][8 + 2 * i + 1] = bf16_hi(b[i]);
        }
        xr[r] = lds[r * kRowDwords + kNope / 2 + lane];
    }

    // ---- RMSNorm: the RPS row sums reduce together ----
    // Per-lane partials in the coarse kernel's order: Q accumulates adjacent
    // pairs (v_pk_fma), K adds its 16 squares one by one.
    float inv[RPS];
#pragma unroll
    for(int r = 0; r < RPS; ++r)
    {
        float part;
        if constexpr(K_ROW)
        {
            part = 0.f;
#pragma unroll
            for(int i = 0; i < kVec; ++i)
                part += x[r][i] * x[r][i];
        }
        else
        {
            f32x2 acc = {0.f, 0.f};
#pragma unroll
            for(int i = 0; i < kVec; i += 2)
            {
                const f32x2 v = {x[r][i], x[r][i + 1]};
                acc += v * v;
            }
            part = acc.x + acc.y;
        }
        inv[r] = wave_sum(part);
    }
#pragma unroll
    for(int r = 0; r < RPS; ++r)
    {
        const float ss = __builtin_bit_cast(
            float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, inv[r])));
        inv[r] = rsqrtf(ss / static_cast<float>(kHead) + eps);
    }

    // ---- NoPE: group amax -> e8m0 -> fp8 ----
    // Lanes 28..31 quantise their RoPE columns too and never store them; that
    // costs nothing, the wave issues the instructions either way.
    // The hardware convert divides by the scale. The scale is a power of two,
    // so (x * rstd) / scale is the coarse kernel's x * (rstd / scale) exactly.
    f32x16 wv;
    if constexpr(K_ROW)
    {
#pragma unroll
        for(int i = 0; i < kVec; ++i)
            wv[i] = w[i];
    }
    u32x4 q8[RPS];
    uint32_t scale_pair[RPS];
#pragma unroll
    for(int r = 0; r < RPS; ++r)
    {
        f32x16 n = x[r] * inv[r];
        if constexpr(K_ROW)
            n *= wv;
        float amax = 1e-10f; // kAmaxFloor (quant_amax_floor.h)
#pragma unroll
        for(int i = 0; i < kVec; i += 2)
            amax = fmaxf(amax, fmaxf(fabsf(n[i]), fabsf(n[i + 1])));
        amax                = max_quad(amax);
        const uint32_t byte = e8m0_byte(amax);
        const float dq      = __builtin_bit_cast(float, byte << 23);
        const u32x2 c0      = __builtin_amdgcn_cvt_scalef32_pk8_fp8_f32(n.s01234567, dq);
        const u32x2 c1      = __builtin_amdgcn_cvt_scalef32_pk8_fp8_f32(n.s89abcdef, dq);
        q8[r]               = u32x4{c0.x, c0.y, c1.x, c1.y};
        scale_pair[r]       = byte | (byte << 8);
    }

    // ---- RoPE: lane l rotates pair l of the 32 ----
    // The fma shape and the empty asm are the coarse kernel's K RoPE: each
    // result is materialised once, so the main row and the SWA copy agree.
    uint32_t rot[RPS];
#pragma unroll
    for(int r = 0; r < RPS; ++r)
    {
        float e = bf16_lo(xr[r]) * inv[r];
        float o = bf16_hi(xr[r]) * inv[r];
        if constexpr(K_ROW)
        {
            e *= wr0;
            o *= wr1;
        }
        float r0 = __builtin_fmaf(e, c, -(o * s));
        float r1 = __builtin_fmaf(o, c, e * s);
        asm("" : "+v"(r0), "+v"(r1));
        rot[r] = pack_bf16x2(r0, r1);
    }

    // ---- stores ----
    if(lane < kNopeLanes)
    {
#pragma unroll
        for(int r = 0; r < RPS; ++r)
        {
            *reinterpret_cast<u32x4*>(nope + r * kHead + lane * kVec) = q8[r];
            if(write2)
                *reinterpret_cast<u32x4*>(nope2 + r * kHead + lane * kVec) = q8[r];
        }
        // Each scale byte twice, (s0,s0,..,s6,s6), at [kNope, kNope + 14).
        if(lane % kLanesPerGroup == 0)
        {
            const int off = kNope + 2 * (lane / kLanesPerGroup);
#pragma unroll
            for(int r = 0; r < RPS; ++r)
            {
                *reinterpret_cast<uint16_t*>(nope + r * kHead + off) =
                    static_cast<uint16_t>(scale_pair[r]);
                if(write2)
                    *reinterpret_cast<uint16_t*>(nope2 + r * kHead + off) =
                        static_cast<uint16_t>(scale_pair[r]);
            }
        }
    }
#pragma unroll
    for(int r = 0; r < RPS; ++r)
    {
        rope[r * (kRope / 2) + lane] = rot[r];
        if(write2)
            rope2[r * (kRope / 2) + lane] = rot[r];
    }
}
#endif // __gfx1250__

// SWA_MODE: 0 no SWA copy, 1 direct (swa_dest_row[t]), 2 paged (block table).
//
// Kernarg order matters: only the first 30 dwords are preloaded into SGPRs and
// the rest cost an s_load. Everything a program needs before its first memory
// access -- including the SWA row and batch-id arrays, whose loads the K program
// starts up front -- sits in that window. What is only read at the stores (the
// SWA planes and their strides, the paged-SWA table) sits after it.
template <int HPP, int RPS, int WPE, int SWA_MODE>
__global__ void __launch_bounds__(kLanes) __attribute__((amdgpu_waves_per_eu(WPE, WPE)))
AITER_QKNR_GFX1250_NO_CLUSTER
qk_norm_rope_group_quant_kernel(const uint16_t* __restrict__ q,
                                const int64_t* __restrict__ positions,
                                const uint16_t* __restrict__ cos_cache,
                                const uint16_t* __restrict__ sin_cache,
                                const uint16_t* __restrict__ kv,
                                const uint16_t* __restrict__ k_weight,
                                const int32_t* __restrict__ swa_dest_row,
                                const int32_t* __restrict__ batch_id,
                                float eps,
                                int max_position,
                                int q_stride_0,
                                int kv_stride_0,
                                int num_heads,
                                int swa_num_rows,
                                uint8_t* __restrict__ q_nope,
                                uint32_t* __restrict__ q_rope,
                                uint8_t* __restrict__ k_nope,
                                uint32_t* __restrict__ k_rope,
                                // ---- past the preload window ----
                                uint8_t* __restrict__ swa_nope,
                                uint32_t* __restrict__ swa_rope,
                                int swa_nope_row_stride,
                                int swa_rope_row_stride,
                                const int32_t* __restrict__ swa_block_tables,
                                int swa_block_size,
                                int swa_bt_stride,
                                int swa_bt_blocks)
{
#if defined(__gfx1250__)
    static_assert(HPP % RPS == 0, "HPP must be a multiple of RPS");
    constexpr int kSteps = HPP / RPS;
    __shared__ __attribute__((aligned(1024))) uint32_t smem[HPP * kRowDwords];

    const uint32_t pit   = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const int lane       = threadIdx.x;
    const opus::u32_t lds_base =
        static_cast<opus::u32_t>(reinterpret_cast<__UINTPTR_TYPE__>(smem));

    if(pit == 0)
    {
        // ---- K row (+ SWA copy) ----
        // Every load is issued up front, before any is waited on. A load inside
        // a branch gets drained inside that branch, which put two memory round
        // trips back to back on this program's path; the sched_barrier keeps
        // the scheduler from sinking the weight loads past the scalar wait.
        using KWin = opus::tdm<opus::bf16_t,
                               opus::seq<kHead, 1>,
                               opus::tdm_traits::cache<kTdmCachePolicy>>;
        auto win = opus::make_tdm<KWin>(
            lds_base, kv + static_cast<int64_t>(token) * kv_stride_0, kHead, 1, kHead);
        win.async_load();

        const u32x4* wp    = reinterpret_cast<const u32x4*>(k_weight + lane * kVec);
        const u32x4 wa     = wp[0];
        const u32x4 wb     = wp[1];
        const uint32_t wrp = reinterpret_cast<const uint32_t*>(k_weight + kNope)[lane];
        const int64_t pos_raw = positions[token];
        int32_t bid = 0, dest = 0;
        if constexpr(SWA_MODE != 0)
            bid = batch_id[token];
        if constexpr(SWA_MODE == 1)
            dest = swa_dest_row[token];
        __builtin_amdgcn_sched_barrier(0);

        // A stale / out-of-range position on a pad token is clamped before it
        // indexes cos/sin.
        int pos = static_cast<int>(pos_raw);
        if(max_position > 0)
            pos = pos < 0 ? 0 : (pos >= max_position ? max_position - 1 : pos);
        const float c = bf16_lo(cos_cache[pos * (kRope / 2) + lane]);
        const float s = bf16_lo(sin_cache[pos * (kRope / 2) + lane]);
        float w[kVec];
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            w[2 * i]         = bf16_lo(wa[i]);
            w[2 * i + 1]     = bf16_hi(wa[i]);
            w[8 + 2 * i]     = bf16_lo(wb[i]);
            w[8 + 2 * i + 1] = bf16_hi(wb[i]);
        }

        bool write2     = false;
        uint8_t* nope2  = k_nope;
        uint32_t* rope2 = k_rope;
        if constexpr(SWA_MODE != 0)
        {
            // swa_row_for_token: skip pad tokens, stale positions, unmapped
            // blocks and rows the pool does not have.
            int64_t row = -1;
            if(bid >= 0 && pos_raw >= 0)
            {
                if constexpr(SWA_MODE == 1)
                    row = dest;
                else
                {
                    const uint32_t upos = static_cast<uint32_t>(pos_raw);
                    const uint32_t ubs  = static_cast<uint32_t>(swa_block_size);
                    const uint32_t ublk = upos / ubs;
                    const uint32_t urem = upos - ublk * ubs;
                    if(static_cast<int32_t>(ublk) < swa_bt_blocks)
                    {
                        const int32_t phys =
                            swa_block_tables[static_cast<int64_t>(bid) * swa_bt_stride + ublk];
                        if(phys >= 0)
                            row = static_cast<int64_t>(phys) * swa_block_size + urem;
                    }
                }
            }
            if(row >= 0 && row < swa_num_rows)
            {
                write2 = true;
                nope2  = swa_nope + row * swa_nope_row_stride;
                rope2  = reinterpret_cast<uint32_t*>(reinterpret_cast<uint16_t*>(swa_rope) +
                                                    row * swa_rope_row_stride);
            }
        }
        opus::s_wait_tensorcnt<0>();
        // The TDM write into smem is invisible to the compiler; keep the LDS
        // reads below the wait.
        asm volatile("" ::: "memory");
        step_rows<1, true>(smem,
                           lane,
                           eps,
                           w,
                           bf16_lo(wrp),
                           bf16_hi(wrp),
                           c,
                           s,
                           k_nope + static_cast<int64_t>(token) * kHead,
                           k_rope + static_cast<int64_t>(token) * (kRope / 2),
                           nope2,
                           rope2,
                           write2);
    }
    else
    {
        // ---- HPP Q heads, RPS rows per step ----
        // One TDM copy per step, all issued before the first wait; they finish
        // in order, so step st waits for the copies of steps <= st only.
        const int first_head = static_cast<int>(pit - 1) * HPP;
        using QWin = opus::tdm<opus::bf16_t,
                               opus::seq<kHead, RPS>,
                               opus::tdm_traits::cache<kTdmCachePolicy>>;
        auto win = opus::make_tdm<QWin>(lds_base,
                                        q + static_cast<int64_t>(token) * q_stride_0 +
                                            static_cast<int64_t>(first_head) * kHead,
                                        kHead,
                                        HPP,
                                        kHead);
#pragma unroll
        for(int st = 0; st < kSteps; ++st)
        {
            win.async_load(st * RPS * kHead);
            if(st + 1 < kSteps)
                win.move(opus::number<0>{}, RPS);
        }

        int pos = static_cast<int>(positions[token]);
        if(max_position > 0)
            pos = pos < 0 ? 0 : (pos >= max_position ? max_position - 1 : pos);
        const float c = bf16_lo(cos_cache[pos * (kRope / 2) + lane]);
        const float s = bf16_lo(sin_cache[pos * (kRope / 2) + lane]);

        const int64_t row0       = static_cast<int64_t>(token) * num_heads + first_head;
        uint8_t* nope            = q_nope + row0 * kHead;
        uint32_t* rope           = q_rope + row0 * (kRope / 2);
        const float no_w[kVec]   = {};
        opus::static_for<kSteps>([&](auto I) {
            constexpr int st = decltype(I)::value;
            opus::s_wait_tensorcnt<kSteps - 1 - st>();
            asm volatile("" ::: "memory");
            step_rows<RPS, false>(smem + st * RPS * kRowDwords,
                                  lane,
                                  eps,
                                  no_w,
                                  0.f,
                                  0.f,
                                  c,
                                  s,
                                  nope + st * RPS * kHead,
                                  rope + st * RPS * (kRope / 2),
                                  nullptr,
                                  nullptr,
                                  false);
        });
    }
#endif // __gfx1250__
}

#undef AITER_QKNR_GFX1250_NO_CLUSTER

struct LaunchArgs
{
    const void* q; // bf16 [T, H, 512], heads packed (row pitch 512)
    const void* positions; // int64 [T]
    const void* cos_cache; // bf16 [max_position, 32]
    const void* sin_cache;
    const void* kv;        // bf16 [T, 512] rows, pitch kv_stride_0
    const void* k_weight;  // bf16 [512]
    void* q_nope;          // fp8 [T, H, 512]
    void* q_rope;          // bf16 [T, H, 64]
    void* k_nope;          // fp8 [T, 512]
    void* k_rope;          // bf16 [T, 64]
    const int32_t* swa_dest_row;     // direct SWA mode
    const int32_t* batch_id;         // [T], any SWA mode
    void* swa_nope;                  // fp8 [rows, *], pitch swa_nope_row_stride
    void* swa_rope;                  // bf16 [rows, *], pitch swa_rope_row_stride
    const int32_t* swa_block_tables; // paged SWA mode
    float eps;
    int num_tokens, num_heads, q_stride_0, kv_stride_0, max_position;
    int swa_mode; // 0 none, 1 direct, 2 paged
    int swa_num_rows, swa_nope_row_stride, swa_rope_row_stride;
    int swa_block_size, swa_bt_stride, swa_bt_blocks;
};

// (HPP, RPS, WPE) per size, from sweeps of T in [1, 16384] x H in {16, 32, 64,
// 128} on MI450 (interleaved runs, median launch time):
//  - T*H < 64: (8, 8, 5) -- fewest programs; at T=1 H<=32 the extra wave
//    dispatches of the narrower tile cost more than they save.
//  - T <= 384 / 128 / 64 for H >= 128 / >= 64 / below: (4, 4, 8) -- still
//    latency-bound, so twice the programs of half the rows. The crossover moves
//    down with H because the same T is fewer programs.
//  - T <= 2048: (8, 8, 5).
//  - above: (8, 8, 4); the extra VGPRs keep all 8 rows in registers.
// H % 8 != 0 (but % 4 == 0) always takes (4, 4, 8).
//
// Tried and dropped: folding the K row into each token's first Q program (one
// program fewer per token). The K row's own chain -- its TDM copy, the SWA row
// lookup, its math and stores -- then runs in front of that program's Q rows
// instead of beside them on another wave, and the longer programs cost more
// than the dropped ones save: 1.3x slower at H=16 T=512, 1.6x at H=128 T=512.
template <int HPP, int RPS, int WPE, int SWA_MODE>
inline void launch_tier(const LaunchArgs& a, hipStream_t stream)
{
    // grid.y is 16 bits wide: launch the tokens in chunks.
    constexpr int kMaxGridY = 65535;
    for(int t0 = 0; t0 < a.num_tokens; t0 += kMaxGridY)
    {
        const int nt = a.num_tokens - t0 < kMaxGridY ? a.num_tokens - t0 : kMaxGridY;
        const int64_t qrow = static_cast<int64_t>(t0) * a.num_heads;
        dim3 grid(static_cast<unsigned>(1 + a.num_heads / HPP), static_cast<unsigned>(nt));
        qk_norm_rope_group_quant_kernel<HPP, RPS, WPE, SWA_MODE><<<grid, kLanes, 0, stream>>>(
            static_cast<const uint16_t*>(a.q) + static_cast<int64_t>(t0) * a.q_stride_0,
            static_cast<const int64_t*>(a.positions) + t0,
            static_cast<const uint16_t*>(a.cos_cache),
            static_cast<const uint16_t*>(a.sin_cache),
            static_cast<const uint16_t*>(a.kv) + static_cast<int64_t>(t0) * a.kv_stride_0,
            static_cast<const uint16_t*>(a.k_weight),
            a.swa_dest_row != nullptr ? a.swa_dest_row + t0 : nullptr,
            a.batch_id != nullptr ? a.batch_id + t0 : nullptr,
            a.eps,
            a.max_position,
            a.q_stride_0,
            a.kv_stride_0,
            a.num_heads,
            a.swa_num_rows,
            static_cast<uint8_t*>(a.q_nope) + qrow * kHead,
            static_cast<uint32_t*>(a.q_rope) + qrow * (kRope / 2),
            static_cast<uint8_t*>(a.k_nope) + static_cast<int64_t>(t0) * kHead,
            static_cast<uint32_t*>(a.k_rope) + static_cast<int64_t>(t0) * (kRope / 2),
            static_cast<uint8_t*>(a.swa_nope),
            static_cast<uint32_t*>(a.swa_rope),
            a.swa_nope_row_stride,
            a.swa_rope_row_stride,
            a.swa_block_tables,
            a.swa_block_size,
            a.swa_bt_stride,
            a.swa_bt_blocks);
    }
}

template <int HPP, int RPS, int WPE>
inline void launch_swa_mode(const LaunchArgs& a, hipStream_t stream)
{
    if(a.swa_mode == 1)
        launch_tier<HPP, RPS, WPE, 1>(a, stream);
    else if(a.swa_mode == 2)
        launch_tier<HPP, RPS, WPE, 2>(a, stream);
    else
        launch_tier<HPP, RPS, WPE, 0>(a, stream);
}

inline void launch(const LaunchArgs& a, hipStream_t stream)
{
    if(a.num_tokens == 0)
        return;
    const int64_t rows   = static_cast<int64_t>(a.num_tokens) * a.num_heads;
    const int narrow_max = a.num_heads >= 128 ? 384 : (a.num_heads >= 64 ? 128 : 64);
    if(a.num_heads % 8 != 0)
        launch_swa_mode<4, 4, 8>(a, stream);
    else if(rows < 64)
        launch_swa_mode<8, 8, 5>(a, stream);
    else if(a.num_tokens <= narrow_max)
        launch_swa_mode<4, 4, 8>(a, stream);
    else if(a.num_tokens <= 2048)
        launch_swa_mode<8, 8, 5>(a, stream);
    else
        launch_swa_mode<8, 8, 4>(a, stream);
}

} // namespace qknr_gfx1250
} // namespace aiter
