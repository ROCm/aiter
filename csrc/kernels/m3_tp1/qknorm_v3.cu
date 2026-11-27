// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// fused_qknorm_idxrqknorm's arithmetic, operation for operation, for the MiniMax-M3.1 TP1 layout
// (m3_tp1_qknorm.h): 16-lane rows of 16 B per lane, DPP reductions, whole-page K/V stores.

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "m3_tp1_qknorm.h"

#include <algorithm>
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace aiter {
namespace m3_tp1_qknorm_ops {

constexpr int kHeadDim     = 128;
constexpr int kNumHeads    = 64;
constexpr int kNumKvHeads  = 4;
constexpr int kNumIdxHeads = 4;
constexpr int kRotaryDim   = 64;
constexpr int kRows        = kNumHeads + 2 * kNumKvHeads + kNumIdxHeads + 1; // q | k | v | iq | ik
constexpr int kQkvRow      = kRows * kHeadDim;                                // 9856
constexpr int kPage        = 16; // SHUFFLE page (tokens) == fp8 interleave
constexpr int kLanesPerRow = 16;
constexpr int kElems       = 8; // per lane
constexpr int kWavesPerToken = 2;
constexpr int kSteps         = 20; // 77 rows, 4 per wave step
constexpr int kStepsPerWave  = kSteps / kWavesPerToken;
// per 16-token group: the token waves, then one page writer per (kv head, K | V)
constexpr int kGroupWaves = kPage * kWavesPerToken + 2 * kNumKvHeads;
static_assert(kSteps % kWavesPerToken == 0 && kGroupWaves % 4 == 0, "4 waves per block");

// hip_bfloat16(float) as the reference compiles it: RNE for finite values and Inf,
// NaN keeps its payload with bit 16 forced.
__device__ __forceinline__ uint32_t f32_to_bf16_bits(float f)
{
    const uint32_t u             = __builtin_bit_cast(uint32_t, f);
    const uint32_t rne           = u + 0x7fffu + ((u >> 16) & 1u);
    const uint32_t nan_keep      = ((u & 0xffffu) != 0u) ? (u | 0x10000u) : u;
    const bool finite_or_inf     = (u & 0x7f800000u) != 0x7f800000u;
    return (finite_or_inf ? rne : nan_keep) >> 16;
}
__device__ __forceinline__ float bf16_bits_to_f32(uint32_t b) { return __builtin_bit_cast(float, b << 16); }

__device__ __forceinline__ uint32_t fp8x4(float a, float b, float c, float d)
{
    int w = 0;
    w     = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, w, false);
    w     = __builtin_amdgcn_cvt_pk_fp8_f32(c, d, w, true);
    return static_cast<uint32_t>(w);
}

template <int CTRL>
__device__ __forceinline__ float dpp(float v)
{
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL, 0xf, 0xf, false));
}
constexpr int kRor8 = 0x128, kRor4 = 0x124, kRor12 = 0x12C, kXor2 = 0x4E, kXor1 = 0xB1;
// Butterfly partner for masks 8, 4, 2, 1; after the mask-8 level every value is symmetric
// under xor 8, so row_ror:4 equals xor 4.
template <int M>
__device__ __forceinline__ float bfly(float v)
{
    if constexpr(M == 8) return dpp<kRor8>(v);
    else if constexpr(M == 4) return dpp<kRor4>(v);
    else if constexpr(M == 2) return dpp<kXor2>(v);
    else return dpp<kXor1>(v);
}

// The reference's RMSNorm, lane j = its lanes 2j (e[0..3]) and 2j + 1 (e[4..7]): its 32-lane
// butterfly (masks 16..1) is two 16-lane chains (masks 8..1), then even + odd.
__device__ __forceinline__ void norm8(float (&e)[kElems], float eps, const float (&w1)[kElems])
{
    float se = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(e[0], e[0]), __fmul_rn(e[1], e[1])), __fmul_rn(e[2], e[2])),
                         __fmul_rn(e[3], e[3]));
    float so = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(e[4], e[4]), __fmul_rn(e[5], e[5])), __fmul_rn(e[6], e[6])),
                         __fmul_rn(e[7], e[7]));
    se = __fadd_rn(se, bfly<8>(se)); so = __fadd_rn(so, bfly<8>(so));
    se = __fadd_rn(se, bfly<4>(se)); so = __fadd_rn(so, bfly<4>(so));
    se = __fadd_rn(se, bfly<2>(se)); so = __fadd_rn(so, bfly<2>(so));
    se = __fadd_rn(se, bfly<1>(se)); so = __fadd_rn(so, bfly<1>(so));
    const float sum = __fadd_rn(se, so);
    const float rcp = rsqrtf(__fmaf_rn(sum, 0.0078125f, eps));
#pragma unroll
    for(int i = 0; i < kElems; ++i)
        e[i] = __fmul_rn(__fmul_rn(e[i], rcp), w1[i]); // (e * rcp) * (1 + w)
}

// Partial NeoX RoPE, rotary 64: lanes 0..3 (dims 0..31) pair with lanes 4..7, which
// row_ror:12 / row_ror:4 deliver; ps = partner * sin, negated on the first half.
__device__ __forceinline__ void rope8(float (&e)[kElems], int j, const float (&c)[kElems], const float (&s)[kElems])
{
    float p[kElems];
#pragma unroll
    for(int i = 0; i < kElems; ++i)
    {
        const float from_minus4 = dpp<kRor4>(e[i]);
        const float from_plus4  = dpp<kRor12>(e[i]);
        p[i]                    = (j & 4) ? from_minus4 : from_plus4;
    }
    if(j < 8)
    {
        const bool first = j < 4;
#pragma unroll
        for(int i = 0; i < kElems; ++i)
        {
            float ps = __fmul_rn(p[i], s[i]);
            ps       = first ? -ps : ps;
            e[i]     = __fmaf_rn(e[i], c[i], ps);
        }
    }
}

__device__ __forceinline__ void unpack8(const uint4 v, float (&e)[kElems])
{
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        e[2 * i]     = __builtin_bit_cast(float, w[i] << 16);
        e[2 * i + 1] = __builtin_bit_cast(float, w[i] & 0xffff0000u);
    }
}

__device__ __forceinline__ uint4 pack_bf16x8(const float (&e)[kElems])
{
    uint32_t b[kElems];
#pragma unroll
    for(int i = 0; i < kElems; ++i)
        b[i] = f32_to_bf16_bits(e[i]);
    return make_uint4(b[0] | (b[1] << 16), b[2] | (b[3] << 16), b[4] | (b[5] << 16), b[6] | (b[7] << 16));
}

__device__ __forceinline__ float row_max16(float v)
{
    v = fmaxf(v, dpp<kRor8>(v));
    v = fmaxf(v, dpp<kRor4>(v));
    v = fmaxf(v, dpp<kXor2>(v));
    v = fmaxf(v, dpp<kXor1>(v));
    return v;
}

__device__ __forceinline__ void unit_weight8(const uint16_t* __restrict__ w, int j, float (&w1)[kElems])
{
    unpack8(*reinterpret_cast<const uint4*>(w + j * kElems), w1);
#pragma unroll
    for(int i = 0; i < kElems; ++i)
        w1[i] = __fadd_rn(w1[i], 1.0f);
}

// Whether the 16 tokens of group g fill one page in order; its K/V pages are then
// written whole by the group's page writers, and the token waves skip them.
__device__ __forceinline__ bool group_aligned(const int64_t* __restrict__ slot_mapping, int g, int num_tokens,
                                              int lane)
{
    const int t0 = g * kPage;
    if(t0 + kPage - 1 >= num_tokens)
        return false;
    const int64_t base = slot_mapping[t0];
    const int64_t mine = lane < kPage ? slot_mapping[t0 + lane] : base + lane;
    return __all((base >= 0) && (base % kPage == 0) && (mine == base + lane));
}

// One page of (kv head, K | V) for group g; lane = tq * 4 + q holds token tq's dims
// 32q..32q+31, i.e. the reference's lanes 8q..8q+7 (chain c = its lane 8q + c).
__device__ __forceinline__ void kv_group(bool is_k, int g, int head, const uint16_t* __restrict__ qkv,
                                         const uint16_t* __restrict__ kw, const uint16_t* __restrict__ cos_sin,
                                         const int64_t* __restrict__ positions,
                                         const int64_t* __restrict__ slot_mapping, uint8_t* __restrict__ k_cache,
                                         uint8_t* __restrict__ v_cache, float* __restrict__ k_scale,
                                         float* __restrict__ v_scale, float eps, uint8_t* __restrict__ lds)
{
    const int lane  = threadIdx.x % 64;
    const int tq    = lane / 4, q = lane % 4;
    const int token = g * kPage + tq;
    const int col   = (is_k ? kNumHeads + head : kNumHeads + kNumKvHeads + head) * kHeadDim + q * 32;
    const uint4* src = reinterpret_cast<const uint4*>(qkv + static_cast<int64_t>(token) * kQkvRow + col);
    float e[32];
    {
        const uint4 r[4] = {src[0], src[1], src[2], src[3]};
#pragma unroll
        for(int k = 0; k < 4; ++k)
        {
            float t8[kElems];
            unpack8(r[k], t8);
#pragma unroll
            for(int i = 0; i < kElems; ++i)
                e[k * kElems + i] = t8[i];
        }
    }
    if(is_k)
    {
        float sc_[8];
#pragma unroll
        for(int c = 0; c < 8; ++c)
            sc_[c] = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(e[4 * c], e[4 * c]), __fmul_rn(e[4 * c + 1], e[4 * c + 1])),
                                         __fmul_rn(e[4 * c + 2], e[4 * c + 2])),
                               __fmul_rn(e[4 * c + 3], e[4 * c + 3]));
        // the reference's masks 16 and 8 are q bits 1 and 0, then 4, 2, 1 the chain bits
#pragma unroll
        for(int c = 0; c < 8; ++c) sc_[c] = __fadd_rn(sc_[c], dpp<kXor2>(sc_[c]));
#pragma unroll
        for(int c = 0; c < 8; ++c) sc_[c] = __fadd_rn(sc_[c], dpp<kXor1>(sc_[c]));
        float l3[8], l4[8];
#pragma unroll
        for(int c = 0; c < 8; ++c) l3[c] = __fadd_rn(sc_[c], sc_[c ^ 4]);
#pragma unroll
        for(int c = 0; c < 8; ++c) l4[c] = __fadd_rn(l3[c], l3[c ^ 2]);
        const float sum = __fadd_rn(l4[0], l4[1]);
        const float rcp = rsqrtf(__fmaf_rn(sum, 0.0078125f, eps));
#pragma unroll
        for(int i = 0; i < 32; ++i)
        {
            const float w1 = __fadd_rn(bf16_bits_to_f32(kw[q * 32 + i]), 1.0f);
            e[i]           = __fmul_rn(__fmul_rn(e[i], rcp), w1);
        }
        // RoPE: q 0 (dims 0..31) pairs with q 1 (dims 32..63); element i uses cos[i], sin[i]
        const uint16_t* cs = cos_sin + positions[token] * kRotaryDim;
#pragma unroll
        for(int i = 0; i < 32; ++i)
        {
            const float p = dpp<kXor1>(e[i]);
            if(q < 2)
            {
                const float c  = bf16_bits_to_f32(cs[i]);
                const float sn = bf16_bits_to_f32(cs[32 + i]);
                float ps       = __fmul_rn(p, sn);
                ps             = (q == 0) ? -ps : ps;
                e[i]           = __fmaf_rn(e[i], c, ps);
            }
        }
    }
    float am = 0.0f;
#pragma unroll
    for(int i = 0; i < 32; ++i)
        am = fmaxf(am, fabsf(e[i]));
    am             = fmaxf(am, dpp<kXor1>(am));
    am             = fmaxf(am, dpp<kXor2>(am));
    const float sc = (am > 0.0f) ? am / 448.0f : 1.0f;
    uint32_t wd[8];
#pragma unroll
    for(int k = 0; k < 8; ++k)
    {
        float v4[4];
#pragma unroll
        for(int t = 0; t < 4; ++t)
            v4[t] = (is_k ? bf16_bits_to_f32(f32_to_bf16_bits(e[4 * k + t])) : e[4 * k + t]) / sc;
        wd[k] = fp8x4(v4[0], v4[1], v4[2], v4[3]);
    }
    const int64_t b   = slot_mapping[g * kPage] / kPage;
    const int64_t blk = b * (static_cast<int64_t>(kNumKvHeads) * kHeadDim * kPage) + head * (kHeadDim * kPage);
    float* scl        = is_k ? k_scale : v_scale;
    if(q == 0)
        scl[b * (kNumKvHeads * kPage) + head * kPage + tq] = sc;
    if(is_k)
    { // K page [d / 16][t][16]: chunks (2q, tq) and (2q + 1, tq)
        *reinterpret_cast<uint4*>(k_cache + blk + (2 * q) * 256 + tq * 16) = make_uint4(wd[0], wd[1], wd[2], wd[3]);
        *reinterpret_cast<uint4*>(k_cache + blk + (2 * q + 1) * 256 + tq * 16) =
            make_uint4(wd[4], wd[5], wd[6], wd[7]);
    }
    else
    { // V page [d][t]: transpose the bytes through LDS, then 16 B rows
#pragma unroll
        for(int k = 0; k < 8; ++k)
#pragma unroll
            for(int t = 0; t < 4; ++t)
                lds[(q * 32 + 4 * k + t) * 16 + tq] = static_cast<uint8_t>((wd[k] >> (8 * t)) & 0xff);
        __builtin_amdgcn_s_waitcnt(0xc07f); // lgkmcnt(0)
        __builtin_amdgcn_wave_barrier();
        const uint4 r0 = *reinterpret_cast<const uint4*>(lds + lane * 16);
        const uint4 r1 = *reinterpret_cast<const uint4*>(lds + (lane + 64) * 16);
        *reinterpret_cast<uint4*>(v_cache + blk + lane * 16)        = r0;
        *reinterpret_cast<uint4*>(v_cache + blk + (lane + 64) * 16) = r1;
    }
}

// Rows per token: q 0..63 | k 64..67 | v 68..71 | iq 72..75 | ik 76, four per wave step; the
// grid holds kGroupWaves waves per 16-token group, 4 per block.
__global__ void __launch_bounds__(256) qknorm_v3_kernel(
    const uint16_t* __restrict__ qkv, uint16_t* __restrict__ q_out, uint16_t* __restrict__ iq_out,
    const uint16_t* __restrict__ qw, const uint16_t* __restrict__ kw, const uint16_t* __restrict__ iqw,
    const uint16_t* __restrict__ ikw, const uint16_t* __restrict__ cos_sin, const int64_t* __restrict__ positions,
    const int64_t* __restrict__ slot_mapping, const int64_t* __restrict__ index_slot_mapping,
    uint8_t* __restrict__ k_cache, uint8_t* __restrict__ v_cache, uint8_t* __restrict__ index_cache,
    float* __restrict__ k_scale, float* __restrict__ v_scale, float eps, int num_tokens)
{
#if defined(__gfx950__)
    const int wave = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
    const int lane = threadIdx.x % 64;
    __shared__ uint8_t lds_all[4][2048];
    const int G = wave / kGroupWaves;
    const int r = wave % kGroupWaves;
    if(r >= kPage * kWavesPerToken)
    {
        const int gw = r - kPage * kWavesPerToken;
        if(group_aligned(slot_mapping, G, num_tokens, lane))
            kv_group(gw % 2 == 0, G, gw / 2, qkv, kw, cos_sin, positions, slot_mapping, k_cache, v_cache, k_scale,
                     v_scale, eps, lds_all[threadIdx.x / 64]);
        return;
    }
    const int token = G * kPage + r / kWavesPerToken;
    if(token >= num_tokens)
        return;
    const int rg = lane / kLanesPerRow; // row group 0..3
    const int j  = lane % kLanesPerRow; // dims 8j..8j+7
    const int s0 = (r % kWavesPerToken) * kStepsPerWave;
    const bool kv_by_group = s0 <= 17 && s0 + kStepsPerWave > 16 && group_aligned(slot_mapping, G, num_tokens, lane);

    const uint16_t* trow = qkv + static_cast<int64_t>(token) * kQkvRow;
    uint4 raw[kStepsPerWave]; // every load of the wave first
#pragma unroll
    for(int k = 0; k < kStepsPerWave; ++k)
    {
        const int row = (s0 + k) * 4 + rg;
        raw[k] = row < kRows ? *reinterpret_cast<const uint4*>(trow + row * kHeadDim + j * kElems)
                             : make_uint4(0, 0, 0, 0);
    }
    const uint16_t* cs = cos_sin + positions[token] * kRotaryDim;
    const int ib       = (j < 4 ? j : j - 4) * kElems; // valid for j < 8
    float c[kElems], s[kElems];
    unpack8(*reinterpret_cast<const uint4*>(cs + (ib & 31)), c);
    unpack8(*reinterpret_cast<const uint4*>(cs + 32 + (ib & 31)), s);
    float wq[kElems];
    unit_weight8(qw, j, wq);

#pragma unroll
    for(int k = 0; k < kStepsPerWave; ++k)
    {
        const int step = s0 + k;
        const int row  = step * 4 + rg;
        float e[kElems];
        unpack8(raw[k], e);
        if(step < 16)
        { // q heads
            norm8(e, eps, wq);
            rope8(e, j, c, s);
            *reinterpret_cast<uint4*>(q_out + static_cast<int64_t>(token) * kNumHeads * kHeadDim + row * kHeadDim +
                                      j * kElems) = pack_bf16x8(e);
        }
        else if(step == 16 && !kv_by_group)
        { // k heads: norm, RoPE, per-token FP8 into the SHUFFLE K page
            float w1[kElems];
            unit_weight8(kw, j, w1);
            norm8(e, eps, w1);
            rope8(e, j, c, s);
            const int64_t slot = slot_mapping[token];
            float am           = 0.0f;
#pragma unroll
            for(int i = 0; i < kElems; ++i)
                am = fmaxf(am, fabsf(e[i]));
            am             = row_max16(am);
            const float sc = (am > 0.0f) ? am / 448.0f : 1.0f;
            if(slot >= 0)
            {
                const int64_t b = slot / kPage, t = slot % kPage;
                if(j == 0)
                    k_scale[b * (kNumKvHeads * kPage) + rg * kPage + t] = sc;
                float q8[kElems];
#pragma unroll
                for(int i = 0; i < kElems; ++i)
                    q8[i] = bf16_bits_to_f32(f32_to_bf16_bits(e[i])) / sc;
                const int d0      = j * kElems;
                const int64_t off = b * (static_cast<int64_t>(kNumKvHeads) * kHeadDim * kPage) +
                                    rg * (kHeadDim * kPage) + (d0 / 16) * (kPage * 16) + t * 16 + (d0 % 16);
                *reinterpret_cast<uint2*>(k_cache + off) =
                    make_uint2(fp8x4(q8[0], q8[1], q8[2], q8[3]), fp8x4(q8[4], q8[5], q8[6], q8[7]));
            }
        }
        else if(step == 17 && !kv_by_group)
        { // v heads: raw row, per-token FP8 into the SHUFFLE V page
            const int64_t slot = slot_mapping[token];
            float am           = 0.0f;
#pragma unroll
            for(int i = 0; i < kElems; ++i)
                am = fmaxf(am, fabsf(e[i]));
            am             = row_max16(am);
            const float sc = (am > 0.0f) ? am / 448.0f : 1.0f;
            if(slot >= 0)
            {
                const int64_t b = slot / kPage, t = slot % kPage;
                if(j == 0)
                    v_scale[b * (kNumKvHeads * kPage) + rg * kPage + t] = sc;
                const int64_t base = b * (static_cast<int64_t>(kNumKvHeads) * kHeadDim * kPage) +
                                     rg * kHeadDim * kPage + static_cast<int64_t>(j * kElems) * 16 + t;
#pragma unroll
                for(int i = 0; i < kElems; ++i)
                    v_cache[base + i * 16] = static_cast<uint8_t>(fp8x4(e[i] / sc, 0.0f, 0.0f, 0.0f) & 0xff);
            }
        }
        else if(step == 18)
        { // index q heads
            float w1[kElems];
            unit_weight8(iqw, j, w1);
            norm8(e, eps, w1);
            rope8(e, j, c, s);
            *reinterpret_cast<uint4*>(iq_out + static_cast<int64_t>(token) * kNumIdxHeads * kHeadDim +
                                      rg * kHeadDim + j * kElems) = pack_bf16x8(e);
        }
        else if(step == 19 && rg == 0)
        { // index k (row 76): unit-scale FP8 into the index cache
            float w1[kElems];
            unit_weight8(ikw, j, w1);
            norm8(e, eps, w1);
            rope8(e, j, c, s);
            const int64_t islot = index_slot_mapping[token];
            if(islot >= 0)
            {
                float q8[kElems];
#pragma unroll
                for(int i = 0; i < kElems; ++i)
                    q8[i] = bf16_bits_to_f32(f32_to_bf16_bits(e[i]));
                *reinterpret_cast<uint2*>(index_cache + islot * kHeadDim + j * kElems) =
                    make_uint2(fp8x4(q8[0], q8[1], q8[2], q8[3]), fp8x4(q8[4], q8[5], q8[6], q8[7]));
            }
        }
    }
#else
    // gfx950 only (e4m3fn conversion); the host wrapper never launches it elsewhere
    __builtin_trap();
#endif
}

} // namespace m3_tp1_qknorm_ops

void m3_tp1_qknorm_v3(const aiter_tensor_t& qkv,
                      aiter_tensor_t& q_out,
                      aiter_tensor_t& index_q_out,
                      const aiter_tensor_t& q_norm_weight,
                      const aiter_tensor_t& k_norm_weight,
                      const aiter_tensor_t& index_q_norm_weight,
                      const aiter_tensor_t& index_k_norm_weight,
                      const aiter_tensor_t& cos_sin_cache,
                      const aiter_tensor_t& positions,
                      const aiter_tensor_t& slot_mapping,
                      const aiter_tensor_t& index_slot_mapping,
                      aiter_tensor_t& k_cache,
                      aiter_tensor_t& v_cache,
                      aiter_tensor_t& index_cache,
                      aiter_tensor_t& k_scale,
                      aiter_tensor_t& v_scale,
                      double eps,
                      int64_t block_size)
{
    using namespace m3_tp1_qknorm_ops;
    // a failed argument check raises a Python RuntimeError instead of aborting
    aiter_detail::AiterThrowGuard throw_guard;
    AITER_CHECK(qkv.dtype() == AITER_DTYPE_bf16 && qkv.is_contiguous() && qkv.dim() == 2 && qkv.size(1) == kQkvRow,
                "qkv must be contiguous bf16 [num_tokens, 9856] (64 q / 4 kv / 4 index heads of 128)");
    const int64_t T = qkv.size(0);
    const auto bf16_of = [](const aiter_tensor_t& t, int64_t n) {
        return t.dtype() == AITER_DTYPE_bf16 && t.is_contiguous() && static_cast<int64_t>(t.numel()) == n;
    };
    AITER_CHECK(bf16_of(q_out, T * kNumHeads * kHeadDim) && bf16_of(index_q_out, T * kNumIdxHeads * kHeadDim),
                "q_out / index_q_out must be contiguous bf16 with num_tokens * 64 / 4 * 128 elements");
    AITER_CHECK(bf16_of(q_norm_weight, kHeadDim) && bf16_of(k_norm_weight, kHeadDim) &&
                    bf16_of(index_q_norm_weight, kHeadDim) && bf16_of(index_k_norm_weight, kHeadDim),
                "the four norm weights must be contiguous bf16 [128]");
    AITER_CHECK(cos_sin_cache.dtype() == AITER_DTYPE_bf16 && cos_sin_cache.is_contiguous() &&
                    cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == kRotaryDim,
                "cos_sin_cache must be contiguous bf16 [max_pos, 64]");
    for(const aiter_tensor_t* m : {&positions, &slot_mapping, &index_slot_mapping})
        AITER_CHECK(m->dtype() == AITER_DTYPE_i64 && m->is_contiguous() && m->dim() == 1 && m->size(0) >= T,
                    "positions and slot mappings must be contiguous int64 [>= num_tokens]");
    AITER_CHECK(block_size == kPage, "block_size must be 16 (page-16 SHUFFLE K/V)");
    const auto fp8 = [](const aiter_tensor_t& t) {
        return (t.dtype() == AITER_DTYPE_fp8 || t.dtype() == AITER_DTYPE_u8) && t.is_contiguous();
    };
    AITER_CHECK(fp8(k_cache) && k_cache.dim() == 5 && k_cache.size(1) == kNumKvHeads &&
                    k_cache.size(2) == kHeadDim / kPage && k_cache.size(3) == kPage && k_cache.size(4) == kPage,
                "k_cache must be contiguous fp8 [num_blocks, 4, 8, 16, 16]");
    AITER_CHECK(fp8(v_cache) && v_cache.dim() == 5 && v_cache.size(1) == kNumKvHeads && v_cache.size(2) == 1 &&
                    v_cache.size(3) == kHeadDim && v_cache.size(4) == kPage,
                "v_cache must be contiguous fp8 [num_blocks, 4, 1, 128, 16]");
    const int64_t scales = std::min(k_cache.size(0), v_cache.size(0)) * kNumKvHeads * kPage;
    for(const aiter_tensor_t* sc : {&k_scale, &v_scale})
        AITER_CHECK(sc->dtype() == AITER_DTYPE_fp32 && sc->is_contiguous() &&
                        static_cast<int64_t>(sc->numel()) >= scales,
                    "k_scale / v_scale must be contiguous fp32 [num_blocks, 4, 16]");
    AITER_CHECK(fp8(index_cache) && index_cache.numel() % kHeadDim == 0,
                "index_cache must be contiguous fp8 rows of 128");
    for(const aiter_tensor_t* t : std::initializer_list<const aiter_tensor_t*>{
            &q_out, &index_q_out, &q_norm_weight, &k_norm_weight, &index_q_norm_weight, &index_k_norm_weight,
            &cos_sin_cache, &positions, &slot_mapping, &index_slot_mapping, &k_cache, &v_cache, &index_cache,
            &k_scale, &v_scale})
        AITER_CHECK(qkv.is_gpu() && t->device_id == qkv.device_id, "every tensor must be on qkv's GPU");
    if(T == 0)
        return;

    HipDeviceGuard device_guard(qkv.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const int groups         = static_cast<int>((T + kPage - 1) / kPage);
    qknorm_v3_kernel<<<groups * kGroupWaves / 4, 256, 0, stream>>>(
        static_cast<const uint16_t*>(qkv.data_ptr()),
        static_cast<uint16_t*>(q_out.data_ptr()),
        static_cast<uint16_t*>(index_q_out.data_ptr()),
        static_cast<const uint16_t*>(q_norm_weight.data_ptr()),
        static_cast<const uint16_t*>(k_norm_weight.data_ptr()),
        static_cast<const uint16_t*>(index_q_norm_weight.data_ptr()),
        static_cast<const uint16_t*>(index_k_norm_weight.data_ptr()),
        static_cast<const uint16_t*>(cos_sin_cache.data_ptr()),
        static_cast<const int64_t*>(positions.data_ptr()),
        static_cast<const int64_t*>(slot_mapping.data_ptr()),
        static_cast<const int64_t*>(index_slot_mapping.data_ptr()),
        static_cast<uint8_t*>(k_cache.data_ptr()),
        static_cast<uint8_t*>(v_cache.data_ptr()),
        static_cast<uint8_t*>(index_cache.data_ptr()),
        static_cast<float*>(k_scale.data_ptr()),
        static_cast<float*>(v_scale.data_ptr()),
        static_cast<float>(eps),
        static_cast<int>(T));
    HIP_CALL_LAUNCH(hipGetLastError());
}

} // namespace aiter
