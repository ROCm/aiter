// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// R1 NVFP4 MLA decode prep (gfx1250): in one launch, for each decode token,
//
// Q blocks (2 per batch row, 64 heads each): the RoPE part of q_aux. q_aux row:
//   0..31 QR codes, 32..63 QE codes, 64..95 the NoPE scales (left as written),
//   96..99 QR scales, 100..103 QE scales, 104..127 0x7F. A head's 64 RoPE
//   channels are 8 lanes, 16 bytes each; a wave's 8 heads come into LDS as one
//   tensor load.
// K blocks (1 per token, launched first): the new token into its row of the
//   caller's BF16 tail tile, that 16-token tile requantized into the page
//   (NoPE: one scale per 16 x 16 block, codes over amax / 6), and its RoPE part
//   (per token, plus a residual) into the N16 AUX tile. A thread owns one tile
//   row's 32 channels (two blocks); the 16 rows of a block are 16 lanes of a
//   wave, and a wave's 64 channels of the older rows come in as one tensor load.
//
// The bytes are those of mla_nvfp4_quantize_q (bytes 0..63, 96..127) and
// mla_nvfp4_update_cache on the tail tile. RoPE (optional) rounds as aiter's
// Triton RoPE (forward_triton): v_pk_mul_bf16 / v_pk_fma_bf16 on GPT-J pairs.
// Quantization: x / div
// correctly rounded (Markstein, exact for these operands), then the hardware
// FP32 -> E2M1 conversion, which matches round-to-nearest-even for every input;
// dequantized values come back through the hardware E2M1 -> FP32 conversion.
// Build with -ffp-contract=off.
// The translation unit is torch-free (AITER_NO_TORCH_TYPES before any aiter
// header).
#define AITER_NO_TORCH_TYPES
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "mla_nvfp4_decode_prep.h"
#include "opus/opus.hpp"

#include <cstdint>
#include <initializer_list>
#include <utility>

namespace aiter {
namespace mla_nvfp4_prep {

struct Args
{
    const uint16_t* q_rope;
    uint8_t* qa;
    int64_t stride_qrb, stride_qrh;
    int q_rows, q_blocks, k_blocks;
    uint16_t* ring;
    int64_t ring_rows, stride_req, stride_slot;
    const uint16_t* k_nope;
    const uint16_t* k_rope;
    int64_t stride_knope, stride_krope;
    const int32_t* req;
    uint8_t* kv_nope;
    uint8_t* kv_aux;
    int64_t pages;
    const int64_t* pos;
    const int64_t* slot;
    const uint16_t* cos;
    const uint16_t* sin;
    int64_t stride_cos;
};

constexpr int kQBlocksPerRow = 2;           // 128 heads, 64 per block
constexpr int kQtHeads       = 8;           // a Q wave's heads
constexpr int kLds           = 8 * 16 * 64; // K: 8 waves x [16 rows, 64]; Q: 8 x [8 heads, 64]

#if defined(__gfx1250__)
typedef float v8f __attribute__((ext_vector_type(8)));
using QtWin = opus::tdm<uint16_t, opus::seq<64, kQtHeads>>;
using KtWin = opus::tdm<uint16_t, opus::seq<64, 16>>;

struct QGain
{ static constexpr float g = 6.72f, r = 1.0f / 6.72f; };
struct KvGain
{ static constexpr float g = 89.6f, r = 1.0f / 89.6f; };

__device__ __forceinline__ float as_f(uint32_t b) { return __uint_as_float(b); }
__device__ __forceinline__ uint32_t as_u(float f) { return __float_as_uint(f); }
__device__ __forceinline__ float bf(uint32_t h) { return as_f(h << 16); }

// E4M3, nearest even on [2**-9, 448] (no saturation or underflow cases there).
__device__ __forceinline__ float e4m3(float v, uint32_t& code)
{
    const int pk = __builtin_amdgcn_cvt_pk_fp8_f32(v, v, 0, false);
    code         = uint32_t(pk) & 0xFF;
    return __builtin_amdgcn_cvt_f32_fp8(pk, 0);
}

// RN(a / gain).
template <class Gn>
__device__ __forceinline__ float div_gain(float a)
{
    const float q = a * Gn::r;
    return __fmaf_rn(__fmaf_rn(-Gn::g, q, a), Gn::r, q);
}

__device__ __forceinline__ uint32_t pk_mul_bf16(uint32_t a, uint32_t b)
{
    uint32_t d;
    asm("v_pk_mul_bf16 %0, %1, %2" : "=v"(d) : "v"(a), "v"(b));
    return d;
}

__device__ __forceinline__ uint32_t pk_fma_bf16(uint32_t a, uint32_t b, uint32_t c)
{
    uint32_t d;
    asm("v_pk_fma_bf16 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    return d;
}

// RN(amax / 6): a multiply and an FMA correction, exact for every FP32
// significand; the division itself where the correction could underflow.
// EXACT_TINY = false: below 1e-30 only some value below 1e-30, enough for a
// scale that clamps to 2**-9 there.
template <bool EXACT_TINY>
__device__ __forceinline__ float div6(float a)
{
    constexpr float r6 = 1.0f / 6.0f;
    if(EXACT_TINY && a < 1e-30f)
        return __fdiv_rn(a, 6.0f);
    const float q = a * r6;
    return __fmaf_rn(__fmaf_rn(-6.0f, q, a), r6, q);
}

// One 16-value block's quantizer at the gain: IDEAL divides by amax / 6 itself,
// the others by the E4M3 scale over the gain. RN(1 / div) is v_rcp_f32 and one
// Newton step, equal to it for every divisor these blocks can produce (checked
// exhaustively) except ideal ones below 1e-30, which keep the division.
template <class Gn, bool IDEAL>
struct Block
{
    float scale, div, rdiv;
    uint32_t scode;
    __device__ __forceinline__ explicit Block(float amax)
    {
        const float d0 = amax == 0.0f ? 1.0f : div6<IDEAL>(amax);
        scale          = e4m3(fminf(fmaxf(d0 * Gn::g, 1.0f / 512), 448.0f), scode);
        div            = IDEAL ? d0 : div_gain<Gn>(scale);
        if(!IDEAL || div >= 1e-30f)
        {
            const float r0 = __builtin_amdgcn_rcpf(div);
            rdiv           = __fmaf_rn(__fmaf_rn(div, -r0, 1.0f), r0, r0);
        }
        else
        {
            rdiv = __fdiv_rn(1.0f, div);
        }
    }
    // 8 codes, low nibble first: x / div correctly rounded (signed, so -0
    // stays -0), then FP32 -> E2M1 at scale 2**0.
    __device__ __forceinline__ uint32_t codes(const float* x) const
    {
        // IDEAL can underflow to zero with the module's FTZ build flags.
        // Match the producer's scalar FP4 conversion for the resulting Inf/NaN:
        // its nearest-code loop selects even code 6 for Inf and code 0 for NaN,
        // retaining the quotient sign. Markstein's finite-divisor correction
        // and the hardware FP4 converter are not equivalent on this boundary.
        if constexpr(IDEAL)
        {
            if(div == 0.0f)
            {
                uint32_t packed = 0;
#pragma unroll
                for(int i = 0; i < 8; ++i)
                {
                    const float value = x[i] / div;
                    const uint32_t code = ((as_u(value) >> 28) & 8u) |
                                          (fabsf(value) > 0.0f ? 6u : 0u);
                    packed |= code << (4 * i);
                }
                return packed;
            }
        }
        v8f q;
#pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const float q0 = x[i] * rdiv;
            q[i]           = __fmaf_rn(-__fmaf_rn(div, q0, -x[i]), rdiv, q0);
        }
        return __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32(q, 1.0f);
    }
    // The signed dequantized values of 8 codes.
    __device__ __forceinline__ void deq(uint32_t c, float* out) const
    {
        // E8M0 scale 2**0 in every byte: the lane halves read different bytes.
        const v8f qv = __builtin_amdgcn_cvt_scale_pk8_f32_fp4(c, 0x7F7F7F7Fu, 0);
#pragma unroll
        for(int i = 0; i < 8; ++i)
            out[i] = copysignf(div_gain<Gn>(fabsf(qv[i]) * scale), qv[i]);
    }
};

// 8 RoPE channels (4 GPT-J pairs, BF16 words) at cos/sin index 4 j + p:
// bf16(x * cos + bf16(rot * sin)), as aiter's Triton RoPE computes them.
__device__ __forceinline__ void rope8(uint32_t (&w)[4], uint2 cs, uint2 sn)
{
#pragma unroll
    for(int p = 0; p < 4; ++p)
    {
        const uint32_t cw = p < 2 ? cs.x : cs.y, sw = p < 2 ? sn.x : sn.y;
        const uint32_t c   = ((p & 1) ? cw >> 16 : cw & 0xFFFFu) * 0x10001u;
        const uint32_t s   = ((p & 1) ? sw >> 16 : sw & 0xFFFFu) * 0x10001u;
        const uint32_t rot = ((w[p] >> 16) | (w[p] << 16)) ^ 0x8000u; // (-odd, even)
        w[p]               = pk_fma_bf16(w[p], c, pk_mul_bf16(rot, s));
    }
}

// Select the packed coefficient and (-odd, even) directly in the BF16
// instructions. Preserve rope8's BF16 multiply followed by BF16 FMA rounding.
template <bool HIGH>
__device__ __forceinline__ uint32_t rope_pair_select(uint32_t w, uint32_t cs, uint32_t sn)
{
    uint32_t t, out;
    if constexpr(HIGH)
    {
        asm("v_pk_mul_bf16 %0, %1, %2 op_sel:[1,1] op_sel_hi:[0,1] neg_lo:[1,0]"
            : "=v"(t) : "v"(w), "v"(sn));
        asm("v_pk_fma_bf16 %0, %1, %2, %3 op_sel:[0,1,0] op_sel_hi:[1,1,1]"
            : "=v"(out) : "v"(w), "v"(cs), "v"(t));
    }
    else
    {
        asm("v_pk_mul_bf16 %0, %1, %2 op_sel:[1,0] op_sel_hi:[0,0] neg_lo:[1,0]"
            : "=v"(t) : "v"(w), "v"(sn));
        asm("v_pk_fma_bf16 %0, %1, %2, %3 op_sel:[0,0,0] op_sel_hi:[1,0,1]"
            : "=v"(out) : "v"(w), "v"(cs), "v"(t));
    }
    return out;
}

__device__ __forceinline__ void rope8_select(uint32_t (&w)[4], uint2 cs, uint2 sn)
{
#pragma unroll
    for(int p = 0; p < 4; ++p)
    {
        const uint32_t cw = p < 2 ? cs.x : cs.y, sw = p < 2 ? sn.x : sn.y;
        w[p] = (p & 1) ? rope_pair_select<true>(w[p], cw, sw)
                       : rope_pair_select<false>(w[p], cw, sw);
    }
}

__device__ __forceinline__ void unpack8(const uint32_t (&w)[4], float* x)
{
#pragma unroll
    for(int p = 0; p < 4; ++p)
    {
        x[2 * p]     = bf(w[p] & 0xFFFFu);
        x[2 * p + 1] = bf(w[p] >> 16);
    }
}

__device__ __forceinline__ float pair_max(float v) { return fmaxf(v, __shfl_xor(v, 1)); }

// v from lane i ^ MASK of the same 16-lane row (DPP row_xmask).
template <int MASK>
__device__ __forceinline__ float row_xor(float v)
{
    return as_f(
        uint32_t(__builtin_amdgcn_update_dpp(0, int(as_u(v)), 0x160 + MASK, 0xF, 0xF, false)));
}

// Lanes j < 8 hold 8 channels of a 64-channel RoPE vector (blocks are lane
// pairs): base and residual codes (4 bytes each per lane), and the packed
// scale words for lanes 0 / 1 of the group starting at lane g: four base
// scales, four residual scales.
template <class Gn>
__device__ __forceinline__ void
rope_quant(const float (&x)[8], int g, int j, uint32_t& bc, uint32_t& ec, uint32_t& word)
{
    float m = 0.0f;
#pragma unroll
    for(int i = 0; i < 8; ++i)
        m = fmaxf(m, fabsf(x[i]));
    const Block<Gn, false> base(pair_max(m));
    bc = base.codes(x);
    float res[8];
    base.deq(bc, res);
    float me = 0.0f;
#pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        res[i] = x[i] - res[i];
        me     = fmaxf(me, fabsf(res[i]));
    }
    const Block<Gn, false> resq(pair_max(me));
    ec                = resq.codes(res);
    const uint32_t sc = base.scode | (resq.scode << 8);
    const uint32_t k0 = __shfl(sc, g), k1 = __shfl(sc, g + 2);
    const uint32_t k2 = __shfl(sc, g + 4), k3 = __shfl(sc, g + 6);
    const int sh = j == 1 ? 8 : 0;
    word         = ((k0 >> sh) & 0xFF) | (((k1 >> sh) & 0xFF) << 8) | (((k2 >> sh) & 0xFF) << 16) |
                   (((k3 >> sh) & 0xFF) << 24);
}

template <bool ROPE>
__device__ __forceinline__ void cos_sin(const Args& a, int64_t p, int j, uint2& cs, uint2& sn)
{
    if(ROPE)
    {
        const int64_t t = p * a.stride_cos + 4 * j;
        cs              = *reinterpret_cast<const uint2*>(a.cos + t);
        sn              = *reinterpret_cast<const uint2*>(a.sin + t);
    }
}

// Q block qb: row qb / 2, heads 64 (qb % 2) ..; wave w's 8 heads as one [8, 64]
// tile of the [rows x 128, 64] q_rope window, issued before the padding test.
// The block leaves only after the load landed (the next block reuses the LDS).
template <bool ROPE>
__device__ __forceinline__ void q_block(const Args& a, int qb, uint16_t* lds)
{
    const int lane = threadIdx.x & 31;
    const int wave = __builtin_amdgcn_readfirstlane(int(threadIdx.x >> 5));
    const int j = lane & 7, hl = lane >> 3; // 8 lanes a head; heads hl and hl + 4 of the wave's 8
    const int b    = qb / kQBlocksPerRow;
    const int h8   = (qb % kQBlocksPerRow) * 64 + wave * kQtHeads;
    uint16_t* tile = lds + wave * kQtHeads * 64;
    QtWin win      = opus::make_tdm<QtWin>(uint32_t(reinterpret_cast<uintptr_t>(tile)),
                                           a.q_rope,
                                           64u,
                                           uint32_t(a.q_rows) * 128u,
                                           uint64_t(a.stride_qrh),
                                           0u,
                                           uint32_t(b) * 128u + uint32_t(h8));
    win.async_load(0u);
    uint8_t* out            = a.qa + (int64_t(b) * 128 + h8 + hl) * 128;
    constexpr int kHeadStep = 4 * 128; // heads hl and hl + 4
    if(a.slot[b] < 0)
    {
        // A padding row: the decoder never reads its Q; zero codes and scales.
        const uint32_t pad = j < 2 ? 0u : 0x7F7F7F7Fu;
#pragma unroll
        for(int r = 0; r < 2; ++r)
        {
            *reinterpret_cast<uint2*>(out + r * kHeadStep + 8 * j)         = make_uint2(0, 0);
            *reinterpret_cast<uint32_t*>(out + r * kHeadStep + 96 + 4 * j) = pad;
        }
        opus::s_wait_tensorcnt<0>();
        return;
    }
    uint2 cs = make_uint2(0, 0), sn = make_uint2(0, 0);
    cos_sin<ROPE>(a, a.pos[b], j, cs, sn);
    opus::s_wait_tensorcnt<0>();
#pragma unroll
    for(int r = 0; r < 2; ++r)
    {
        const uint4 raw = *reinterpret_cast<const uint4*>(tile + (hl + 4 * r) * 64 + 8 * j);
        uint32_t w[4]   = {raw.x, raw.y, raw.z, raw.w};
        if(ROPE)
            rope8(w, cs, sn);
        float x[8];
        unpack8(w, x);
        const int g = lane & ~7;
        uint32_t bc, ec, word;
        rope_quant<QGain>(x, g, j, bc, ec, word);
        // Bytes 0..63: lane j writes 8 at 8 j, the codes of lanes 2j, 2j + 1
        // (QR for j < 4, QE after).
        const int s0      = g + 2 * (j & 3);
        const uint32_t bl = __shfl(bc, s0), bh = __shfl(bc, s0 + 1);
        const uint32_t el = __shfl(ec, s0), eh = __shfl(ec, s0 + 1);
        *reinterpret_cast<uint2*>(out + r * kHeadStep + 8 * j) =
            j < 4 ? make_uint2(bl, bh) : make_uint2(el, eh);
        *reinterpret_cast<uint32_t*>(out + r * kHeadStep + 96 + 4 * j) = j < 2 ? word : 0x7F7F7F7Fu;
    }
}

// K block n: thread t has row r = t % 16 of the token's tile and channels c0 ..
// c0 + 31; wave w's 64 channels of rows 0 .. rn - 1 come from the tail tile as
// one [16, 64] tensor load (rows from rn on clamped away).
template <bool ROPE>
__device__ __forceinline__ void k_rope_write(const Args& a, int64_t p, int rn,
                                            int t, uint4 kr, uint16_t* ring_row0,
                                            uint8_t* aux_tile)
{
    // RoPE: lanes 0..7, the new row only (per-token scales).
    if(t < 8)
    {
        const int j   = t;
        uint32_t w[4] = {kr.x, kr.y, kr.z, kr.w};
        if(ROPE)
        {
            uint2 cs, sn;
            cos_sin<ROPE>(a, p, j, cs, sn);
            rope8(w, cs, sn);
        }
        *reinterpret_cast<uint4*>(ring_row0 + int64_t(rn) * a.stride_slot + 512 + 8 * j) =
            make_uint4(w[0], w[1], w[2], w[3]);
        float x[8];
        unpack8(w, x);
        uint32_t bc, ec, word;
        rope_quant<KvGain>(x, 0, j, bc, ec, word);
        // Codes 0 / 512: two 16 x 16-byte bricks each, row rn; scales at 1536 + 16 rn.
        const int off                                      = (j >> 2) * 256 + rn * 16 + (j & 3) * 4;
        *reinterpret_cast<uint32_t*>(aux_tile + off)       = bc;
        *reinterpret_cast<uint32_t*>(aux_tile + 512 + off) = ec;
        if(j < 2)
            *reinterpret_cast<uint32_t*>(aux_tile + 1536 + rn * 16 + 4 * j) = word;
    }
}
template <bool ROPE, int TBASE = 0, bool OVERLAP = false>
__device__ __forceinline__ void k_block(const Args& a, int n, uint16_t* lds)
{
    const int t  = int(threadIdx.x) - TBASE;
    const int r  = t & 15;
    const int c0 = (t >> 4) * 32;
    // Read with the slot, before the padding test (every row of k_nope /
    // k_rope exists, padding rows included): the new row's NoPE slice, by
    // every lane of its channel group, and its RoPE part.
    const int64_t slot = a.slot[n];
    const int64_t p    = a.pos[n];
    const int64_t req  = a.req[n];
    uint4 kn[4];
    const uint4* ks = reinterpret_cast<const uint4*>(a.k_nope + int64_t(n) * a.stride_knope + c0);
#pragma unroll
    for(int i = 0; i < 4; ++i)
        kn[i] = ks[i];
    uint4 kr = make_uint4(0, 0, 0, 0);
    if(t < 8)
        kr = *reinterpret_cast<const uint4*>(a.k_rope + int64_t(n) * a.stride_krope + 8 * t);
    // A padding row has nothing to keep (no tail write, no tile); nor has a
    // token outside the cache or the tail tiles.
    if(slot < 0 || (slot >> 6) >= a.pages || req < 0 || req >= a.ring_rows)
        return;
    const int rn        = int(p & 15); // the new token's tile row; valid = rn + 1
    const int tile      = int((p & 63) >> 4);
    uint16_t* ring_row0 = a.ring + req * a.stride_req;
    uint8_t* nope_tile  = a.kv_nope + (slot >> 6) * 16384 + tile * 4096;
    uint8_t* aux_tile   = a.kv_aux + (slot >> 6) * 8192 + tile * 2048;

    {
        const int wave = __builtin_amdgcn_readfirstlane(int(t >> 5));
        uint16_t* old  = lds + wave * 16 * 64;
        KtWin win      = opus::make_tdm<KtWin>(uint32_t(reinterpret_cast<uintptr_t>(old)),
                                               ring_row0 + wave * 64,
                                               64u,
                                               uint32_t(rn),
                                               uint64_t(a.stride_slot),
                                               0u,
                                               0u);
        win.async_load(0u);
        if constexpr(OVERLAP)
            k_rope_write<ROPE>(a, p, rn, t, kr, ring_row0, aux_tile);
        opus::s_wait_tensorcnt<0>();
        uint4 v[4] = {};
        if(r < rn)
        {
            const uint16_t* s = old + r * 64 + ((t >> 4) & 1) * 32;
#pragma unroll
            for(int i = 0; i < 4; ++i)
                v[i] = *reinterpret_cast<const uint4*>(s + 8 * i);
        }
        else if(r == rn)
        {
            uint4* d = reinterpret_cast<uint4*>(ring_row0 + int64_t(rn) * a.stride_slot + c0);
#pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                v[i] = kn[i];
                d[i] = v[i];
            }
        }
        float x[32];
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            const uint32_t w[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
            unpack8(w, x + 8 * i);
        }
        uint32_t c[4], sc[2];
#pragma unroll
        for(int k = 0; k < 2; ++k)
        {
            float m = 0.0f;
#pragma unroll
            for(int i = 0; i < 16; ++i)
                m = fmaxf(m, fabsf(x[16 * k + i]));
            // Over the 16 rows: the 16 lanes of a DPP row, all active.
            m = fmaxf(m, row_xor<1>(m));
            m = fmaxf(m, row_xor<2>(m));
            m = fmaxf(m, row_xor<4>(m));
            m = fmaxf(m, row_xor<8>(m));
            const Block<KvGain, true> blk(m);
            // The producer writes zero codes for every invalid tile row,
            // including when an underflowed divisor makes zero / div non-finite.
            c[2 * k]     = r <= rn ? blk.codes(x + 16 * k) : 0u;
            c[2 * k + 1] = r <= rn ? blk.codes(x + 16 * k + 8) : 0u;
            sc[k]        = r <= rn ? blk.scode : 127u;
        }
        // 16-token x 32-channel bricks: channel group, token, byte. Two
        // 16 x 16-byte scale bricks: k * 256 + row * 16 + s for index k * 16 + s.
        const int b0 = c0 / 2, s0 = c0 / 16;
        *reinterpret_cast<uint4*>(nope_tile + (b0 >> 4) * 256 + r * 16 + (b0 & 15)) =
            make_uint4(c[0], c[1], c[2], c[3]);
        *reinterpret_cast<uint16_t*>(aux_tile + 1024 + (s0 >> 4) * 256 + r * 16 + (s0 & 15)) =
            uint16_t(sc[0] | (sc[1] << 8));
    }

    if constexpr(!OVERLAP)
        k_rope_write<ROPE>(a, p, rn, t, kr, ring_row0, aux_tile);

    // A tile started here: its later rows and the padding as the cache writer
    // leaves them (zero codes, scale 0x7F).
    if(rn == 0)
    {
        const uint32_t f7 = 0x7F7F7F7Fu;
        const uint4 z = make_uint4(0, 0, 0, 0), f = make_uint4(f7, f7, f7, f7);
        if(t < 60) // base / residual code bricks (0, 256, 512, 768), rows 1..15
            *reinterpret_cast<uint4*>(aux_tile + (t / 15) * 256 + (1 + t % 15) * 16) = z;
        else if(t < 75) // rows 1..15 of the scale rows: scales and padding
            *reinterpret_cast<uint4*>(aux_tile + 1536 + (t - 59) * 16) = f;
        else if(t == 75) // row 0's padding
            *reinterpret_cast<uint2*>(aux_tile + 1536 + 8) = make_uint2(f7, f7);
        else if(t < 92) // 1792 .. 2047
            *reinterpret_cast<uint4*>(aux_tile + 1792 + (t - 76) * 16) = f;
    }
}
// Joint Q/K path: four Q waves cover 128 heads, eight K waves
// cover the tail tile. Each Q wave loads 32 heads and computes four passes.
// Q and K have disjoint LDS and output regions; no CTA barrier is needed.
template<bool ROPE, bool PACKED_ROPE = false>
__device__ __forceinline__ void q_block_joint(const Args& a,int qb,uint16_t* lds){
 const int lane=threadIdx.x&31;
 const int wave=__builtin_amdgcn_readfirstlane(int(threadIdx.x>>5));
 const int j=lane&3,hl=lane>>2,b=qb,h8=wave*32;
 if(b>=a.q_rows)return;
 int64_t meta_slot=a.slot[b],meta_pos=a.pos[b];
 // Keep the slot and position loads together, before constructing the TDM.
 asm volatile("" : "+v"(meta_slot), "+v"(meta_pos));
 uint16_t* tile=lds+wave*32*64;
 opus::tdm<uint16_t,opus::seq<64,32>> win=opus::make_tdm<opus::tdm<uint16_t,opus::seq<64,32>>>(uint32_t(reinterpret_cast<uintptr_t>(tile)),a.q_rope,64u,uint32_t(a.q_rows)*128u,uint64_t(a.stride_qrh),0u,uint32_t(b)*128u+uint32_t(h8));
 win.async_load(0u);
 uint8_t* out=a.qa+(int64_t(b)*128+h8+hl)*128;
 if(meta_slot<0){
 // Packed RoPE leaves register headroom for unrolled padding writes.
 #pragma clang loop unroll_count(PACKED_ROPE ? 4 : 1)
 for(int pass=0;pass<4;pass++){
  *reinterpret_cast<uint4*>(out+pass*8*128+16*j)=make_uint4(0,0,0,0);
  uint32_t v=j==0?0:0x7f7f7f7f;
  *reinterpret_cast<uint2*>(out+pass*8*128+96+8*j)=make_uint2(v,v);
 }
  opus::s_wait_tensorcnt<0>();return;
 }
 uint2 cs0{},sn0{},cs1{},sn1{};
 cos_sin<ROPE>(a,meta_pos,2*j,cs0,sn0);
 cos_sin<ROPE>(a,meta_pos,2*j+1,cs1,sn1);
 opus::s_wait_tensorcnt<0>();
 // Retain the loop: unrolling extends live ranges and reduces residency.
 #pragma unroll 1
 for(int pass=0;pass<4;pass++){
 const uint4 raw0=*reinterpret_cast<const uint4*>(tile+(hl+8*pass)*64+16*j);
 const uint4 raw1=*reinterpret_cast<const uint4*>(tile+(hl+8*pass)*64+16*j+8);
 uint32_t w0[4]={raw0.x,raw0.y,raw0.z,raw0.w},w1[4]={raw1.x,raw1.y,raw1.z,raw1.w};
 if(ROPE){
  if constexpr(PACKED_ROPE){rope8_select(w0,cs0,sn0);rope8_select(w1,cs1,sn1);}
  else {rope8(w0,cs0,sn0);rope8(w1,cs1,sn1);}
 }
 float x[16],res[16];unpack8(w0,x);unpack8(w1,x+8);
 float m=0;
 #pragma unroll
 for(int i=0;i<16;i++)m=fmaxf(m,fabsf(x[i]));
 const Block<QGain,false> base(m);
 const uint2 bc=make_uint2(base.codes(x),base.codes(x+8));
 base.deq(bc.x,res);base.deq(bc.y,res+8);
 float me=0;
 #pragma unroll
 for(int i=0;i<16;i++){res[i]=x[i]-res[i];me=fmaxf(me,fabsf(res[i]));}
 const Block<QGain,false> resq(me);
 const uint2 ec=make_uint2(resq.codes(res),resq.codes(res+8));
 *reinterpret_cast<uint2*>(out+pass*8*128+8*j)=bc;
 *reinterpret_cast<uint2*>(out+pass*8*128+32+8*j)=ec;
 const uint32_t sc=base.scode|(resq.scode<<8);
 // Pair adjacent lanes, then collect lanes 2/3 inside each four-lane head.
 // Each lane contributes one base byte and one residual byte.
 const uint32_t adj=uint32_t(__builtin_amdgcn_update_dpp(0,int(sc),0xb1,0xF,0xF,false));
 const uint32_t pair=__builtin_amdgcn_perm(adj,sc,0x05040100);
 const uint32_t other=uint32_t(__builtin_amdgcn_update_dpp(0,int(pair),0xaa,0xF,0xF,false));
 const uint32_t wb=__builtin_amdgcn_perm(other,pair,0x06040200);
 const uint32_t we=__builtin_amdgcn_perm(other,pair,0x07050301);
 *reinterpret_cast<uint2*>(out+pass*8*128+96+8*j)=j==0?make_uint2(wb,we):make_uint2(0x7f7f7f7f,0x7f7f7f7f);
 }
}

#endif

} // namespace mla_nvfp4_prep

// The K blocks (one per token) first, then the Q blocks (two per row).
template <bool ROPE>
__global__ __launch_bounds__(256) void mla_nvfp4_decode_prep_kernel(mla_nvfp4_prep::Args a)
{
#if defined(__gfx1250__)
    using namespace mla_nvfp4_prep;
    __shared__ uint16_t lds[kLds];
    const int bid = blockIdx.x;
    if(bid < a.k_blocks)
        k_block<ROPE>(a, bid, lds);
    else
        q_block<ROPE>(a, bid - a.k_blocks, lds);
#endif
}

template<bool ROPE>
__global__ __launch_bounds__(384) void mla_nvfp4_decode_prep_joint_kernel(int p_k_blocks,int p_q_rows,const int64_t* p_slot,const int64_t* p_pos,const uint16_t* p_cos,const uint16_t* p_sin,int64_t p_stride_cos,const uint16_t* p_q_rope,uint8_t* p_qa,int64_t p_stride_qrh,uint16_t* p_ring,const int32_t* p_req,const uint16_t* p_k_nope,const uint16_t* p_k_rope,int64_t p_stride_knope,int64_t p_stride_krope,int64_t p_ring_rows,int64_t p_stride_req,int64_t p_stride_slot,uint8_t* p_kv_nope,uint8_t* p_kv_aux,int64_t p_pages) {
#if defined(__gfx1250__)
using namespace mla_nvfp4_prep;
Args a{};
a.k_blocks = p_k_blocks;
a.q_rows = p_q_rows;
a.slot = p_slot;
a.pos = p_pos;
a.cos = p_cos;
a.sin = p_sin;
a.stride_cos = p_stride_cos;
a.q_rope = p_q_rope;
a.qa = p_qa;
a.stride_qrh = p_stride_qrh;
a.ring = p_ring;
a.req = p_req;
a.k_nope = p_k_nope;
a.k_rope = p_k_rope;
a.stride_knope = p_stride_knope;
a.stride_krope = p_stride_krope;
a.ring_rows = p_ring_rows;
a.stride_req = p_stride_req;
a.stride_slot = p_stride_slot;
a.kv_nope = p_kv_nope;
a.kv_aux = p_kv_aux;
a.pages = p_pages;
__shared__ uint16_t lds[2*kLds];
const int bid=blockIdx.x;
if(threadIdx.x<128)
    q_block_joint<ROPE>(a,bid,lds);
else
    k_block<ROPE,128,true>(a,bid,lds+kLds);
#endif
}

// Fixed-stride DSR1 entry. The host checks every specialized stride.
template<bool ROPE, bool PACKED_ROPE = false>
__global__ __launch_bounds__(384) void mla_nvfp4_decode_prep_dsr1_kernel(const uint16_t* p_q_rope,uint8_t* p_qa,uint16_t* p_ring,const uint16_t* p_k_nope,const uint16_t* p_k_rope,const int32_t* p_req,uint8_t* p_kv_nope,uint8_t* p_kv_aux,const int64_t* p_pos,const int64_t* p_slot,const uint16_t* p_cos,const uint16_t* p_sin,int64_t p_ring_rows,int64_t p_pages,int p_batch) {
#if defined(__gfx1250__)
using namespace mla_nvfp4_prep;
Args a{};
a.k_blocks = p_batch;
a.q_rows = p_batch;
a.slot = p_slot;
a.pos = p_pos;
a.cos = p_cos;
a.sin = p_sin;
a.stride_cos = 32;
a.q_rope = p_q_rope;
a.qa = p_qa;
a.stride_qrh = 192;
a.ring = p_ring;
a.req = p_req;
a.k_nope = p_k_nope;
a.k_rope = p_k_rope;
a.stride_knope = 512;
a.stride_krope = 2112;
a.ring_rows = p_ring_rows;
a.stride_req = 562176;
a.stride_slot = 576;
a.kv_nope = p_kv_nope;
a.kv_aux = p_kv_aux;
a.pages = p_pages;
__shared__ uint16_t lds[16384];
const int bid=blockIdx.x; if(threadIdx.x<128) q_block_joint<ROPE,PACKED_ROPE>(a,bid,lds); else k_block<ROPE,128,true>(a,bid,lds+8192);
#endif
}

namespace {
void check_tensor(
    const aiter_tensor_t& t, const char* name, AiterDtype dtype, int dev, size_t align)
{
    AITER_CHECK(t.is_gpu() && t.device_id == dev, name, " must be on q_rope's device");
    AITER_CHECK(t.dtype() == dtype, name, " has unsupported dtype");
    AITER_CHECK(t.numel() == 0 || (t.data_ptr() && uintptr_t(t.data_ptr()) % align == 0),
                name,
                " requires ",
                align,
                "-byte pointer alignment");
}
void check_shape(const aiter_tensor_t& t, const char* name, std::initializer_list<int64_t> dims)
{
    AITER_CHECK(t.dim() == int(dims.size()), name, " has wrong rank");
    int i = 0;
    for(auto dim : dims)
        AITER_CHECK(t.size(i++) == dim, name, " has wrong shape");
}
// Unit last stride, 16-byte rows (8 BF16 elements) along every other axis.
void check_rows16(const aiter_tensor_t& t, const char* name)
{
    AITER_CHECK(t.stride(-1) == 1, name, " needs a unit last stride");
    for(int d = 0; d + 1 < t.dim(); ++d)
        AITER_CHECK(t.stride(d) % 8 == 0, name, " needs 16-byte aligned rows");
}
// Whether the rows of a [R,16,W] unit-last-stride tensor are disjoint: taken
// by stride, each axis longer than 1 must step past everything below it.
bool rows_disjoint(const aiter_tensor_t& t)
{
    int64_t size[2] = {t.size(0), t.size(1)}, stride[2] = {t.stride(0), t.stride(1)};
    if(stride[0] > stride[1])
    {
        std::swap(size[0], size[1]);
        std::swap(stride[0], stride[1]);
    }
    int64_t span = t.size(2);
    for(int d = 0; d < 2; ++d)
    {
        if(size[d] == 1)
            continue;
        if(stride[d] < span)
            return false;
        span += (size[d] - 1) * stride[d];
    }
    return true;
}
} // namespace

void mla_nvfp4_decode_prep(aiter_tensor_t& q_rope,
                           aiter_tensor_t& q_aux,
                           aiter_tensor_t& ring,
                           aiter_tensor_t& req_ids,
                           aiter_tensor_t& k_nope,
                           aiter_tensor_t& k_rope,
                           aiter_tensor_t& positions,
                           aiter_tensor_t& slots,
                           aiter_tensor_t& kv_nope,
                           aiter_tensor_t& kv_aux,
                           std::optional<aiter_tensor_t> cos_cache,
                           std::optional<aiter_tensor_t> sin_cache)
{
    // A failed check raises in Python instead of aborting the process.
    aiter_detail::AiterThrowGuard throw_guard;
    AITER_CHECK(q_rope.is_gpu(), "q_rope must be a GPU tensor");
    const int dev = q_rope.device_id;
    HipDeviceGuard device_guard(dev);
    AITER_CHECK(get_gpu_arch() == "gfx1250", "the NVFP4 MLA decode prep requires gfx1250");

    check_tensor(q_rope, "q_rope", AITER_DTYPE_bf16, dev, 16);
    AITER_CHECK(q_rope.dim() == 3, "q_rope must be [B,128,64]");
    const int64_t b = q_rope.size(0);
    AITER_CHECK(b < (1LL << 25), "batch exceeds uint32 head indexing");
    check_shape(q_rope, "q_rope", {b, 128, 64});
    check_rows16(q_rope, "q_rope");
    // Rows and heads are one axis of the Q tile loads: a contiguous q_rope or
    // the RoPE slice of a [B,128,D] Q.
    AITER_CHECK(q_rope.stride(0) == 128 * q_rope.stride(1),
                "q_rope rows must be 128 head strides apart");
    check_tensor(q_aux, "q_aux", AITER_DTYPE_u8, dev, 16);
    AITER_CHECK(q_aux.is_contiguous(), "q_aux must be contiguous");
    check_shape(q_aux, "q_aux", {b, 128, 128});

    check_tensor(ring, "ring", AITER_DTYPE_bf16, dev, 16);
    AITER_CHECK(ring.dim() == 3 && ring.size(0) > 0, "ring must be [R,16,576]");
    check_shape(ring, "ring", {ring.size(0), 16, 576});
    check_rows16(ring, "ring");
    // Each request's tile rows are written, so no two may share memory.
    AITER_CHECK(rows_disjoint(ring), "ring rows must not overlap");
    check_tensor(req_ids, "req_ids", AITER_DTYPE_i32, dev, 4);
    AITER_CHECK(req_ids.is_contiguous(), "req_ids must be contiguous");
    check_shape(req_ids, "req_ids", {b});
    check_tensor(k_nope, "k_nope", AITER_DTYPE_bf16, dev, 16);
    check_shape(k_nope, "k_nope", {b, 512});
    check_rows16(k_nope, "k_nope");
    check_tensor(k_rope, "k_rope", AITER_DTYPE_bf16, dev, 16);
    check_shape(k_rope, "k_rope", {b, 64});
    check_rows16(k_rope, "k_rope");
    for(auto [t, name] : {std::pair<aiter_tensor_t*, const char*>{&positions, "positions"},
                          std::pair<aiter_tensor_t*, const char*>{&slots, "slots"}})
    {
        check_tensor(*t, name, AITER_DTYPE_i64, dev, 8);
        AITER_CHECK(t->is_contiguous(), name, " must be contiguous");
        check_shape(*t, name, {b});
    }

    check_tensor(kv_nope, "kv_nope", AITER_DTYPE_u8, dev, 256);
    check_tensor(kv_aux, "kv_aux", AITER_DTYPE_u8, dev, 256);
    AITER_CHECK(kv_nope.is_contiguous() && kv_aux.is_contiguous(),
                "cache planes must be contiguous");
    AITER_CHECK(kv_nope.dim() == 2 && kv_nope.size(0) > 0, "cache needs physical pages");
    const int64_t pages = kv_nope.size(0);
    check_shape(kv_nope, "kv_nope", {pages, 16384});
    check_shape(kv_aux, "kv_aux", {pages, 8192});

    AITER_CHECK(cos_cache.has_value() == sin_cache.has_value(), "cos and sin come together");
    const bool rope = cos_cache.has_value();
    if(rope)
    {
        const aiter_tensor_t &cos = *cos_cache, &sin = *sin_cache;
        check_tensor(cos, "cos_cache", AITER_DTYPE_bf16, dev, 8);
        check_tensor(sin, "sin_cache", AITER_DTYPE_bf16, dev, 8);
        AITER_CHECK(cos.dim() == 2 && cos.size(-1) == 32, "cos_cache must be [max_pos,32]");
        check_shape(sin, "sin_cache", {cos.size(0), 32});
        AITER_CHECK(cos.stride(-1) == 1 && sin.stride(-1) == 1 && cos.stride(0) % 4 == 0 &&
                        sin.stride(0) == cos.stride(0),
                    "cos/sin caches need equal strides and 8-byte aligned rows");
    }
    if(!b)
        return;

    mla_nvfp4_prep::Args a{};
    a.q_rope                 = static_cast<const uint16_t*>(q_rope.data_ptr());
    a.qa                     = static_cast<uint8_t*>(q_aux.data_ptr());
    a.stride_qrb             = q_rope.stride(0);
    a.stride_qrh             = q_rope.stride(1);
    a.q_rows                 = int(b);
    a.q_blocks               = int(b) * mla_nvfp4_prep::kQBlocksPerRow;
    a.k_blocks               = int(b);
    a.ring                   = static_cast<uint16_t*>(ring.data_ptr());
    a.ring_rows              = ring.size(0);
    a.stride_req             = ring.stride(0);
    a.stride_slot            = ring.stride(1);
    a.k_nope                 = static_cast<const uint16_t*>(k_nope.data_ptr());
    a.k_rope                 = static_cast<const uint16_t*>(k_rope.data_ptr());
    a.stride_knope           = k_nope.stride(0);
    a.stride_krope           = k_rope.stride(0);
    a.req                    = static_cast<const int32_t*>(req_ids.data_ptr());
    a.kv_nope                = static_cast<uint8_t*>(kv_nope.data_ptr());
    a.kv_aux                 = static_cast<uint8_t*>(kv_aux.data_ptr());
    a.pages                  = pages;
    a.pos                    = static_cast<const int64_t*>(positions.data_ptr());
    a.slot                   = static_cast<const int64_t*>(slots.data_ptr());
    const hipStream_t stream = getCurrentHIPStream();
    const int blocks         = a.k_blocks + a.q_blocks;
    if(rope)
    {
        a.cos        = static_cast<const uint16_t*>(cos_cache->data_ptr());
        a.sin        = static_cast<const uint16_t*>(sin_cache->data_ptr());
        a.stride_cos = cos_cache->stride(0);
        // Keep small batches on the separate Q/K-block path. Large batches
        // share one joint block per request, including graph-padding rows.
        // Fixed model strides keep the runtime-batch argument list compact
        // (116 bytes), within gfx1250's kernarg preload window.
        if(b >= 1024 && a.stride_qrh == 192 && a.stride_knope == 512 &&
            a.stride_krope == 2112 && a.stride_req == 61 * 16 * 576 &&
            a.stride_slot == 576 && a.stride_cos == 32)
        {
            // Packed Q RoPE reduces VGPRs from 76 to 60. Its measured benefit
            // starts at larger batches; preserve the smaller-batch schedule.
            if(b >= 2048)
                mla_nvfp4_decode_prep_dsr1_kernel<true, true><<<int(b), 384, 0, stream>>>(
                    a.q_rope, a.qa, a.ring, a.k_nope, a.k_rope, a.req,
                    a.kv_nope, a.kv_aux, a.pos, a.slot, a.cos, a.sin, a.ring_rows, a.pages,
                    int(b));
            else
                mla_nvfp4_decode_prep_dsr1_kernel<true><<<int(b), 384, 0, stream>>>(
                    a.q_rope, a.qa, a.ring, a.k_nope, a.k_rope, a.req,
                    a.kv_nope, a.kv_aux, a.pos, a.slot, a.cos, a.sin, a.ring_rows, a.pages,
                    int(b));
        }
        else if(b >= 1024)
            mla_nvfp4_decode_prep_joint_kernel<true><<<int(b), 384, 0, stream>>>(
                a.k_blocks,a.q_rows,a.slot,a.pos,a.cos,a.sin,a.stride_cos,a.q_rope,a.qa,a.stride_qrh,a.ring,a.req,a.k_nope,a.k_rope,a.stride_knope,a.stride_krope,a.ring_rows,a.stride_req,a.stride_slot,a.kv_nope,a.kv_aux,a.pages);
        else
            mla_nvfp4_decode_prep_kernel<true><<<blocks, 256, 0, stream>>>(a);
    }
    else
    {
        mla_nvfp4_decode_prep_kernel<false><<<blocks, 256, 0, stream>>>(a);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

} // namespace aiter
