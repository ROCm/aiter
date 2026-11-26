// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <cstdint>
#include <hip/hip_runtime.h>

namespace mla_nvfp4 {
__device__ __forceinline__ float bf16(const uint16_t* p)
{
    return __uint_as_float(uint32_t(*p) << 16);
}
__device__ __forceinline__ float sf_value(uint8_t c)
{
    return c < 8 ? c * 0x1p-9f
                 : __uint_as_float(((uint32_t(c >> 3) + 120) << 23) | ((c & 7) << 20));
}
__device__ __forceinline__ uint8_t sf_code(float x)
{
    x      = fminf(448.0f, fmaxf(0x1p-9f, x));
    int lo = 0, hi = 126;
    while(lo < hi)
    {
        int mid = (lo + hi) / 2;
        if(sf_value(mid) < x)
            lo = mid + 1;
        else
            hi = mid;
    }
    int prev = lo > 0 ? lo - 1 : 0;
    float a = x - sf_value(prev), b = sf_value(lo) - x;
    return a < b || (a == b && !(prev & 1)) ? prev : lo;
}
__device__ __forceinline__ float f4_value(uint8_t c)
{
    constexpr float values[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return (c & 8) ? -values[c & 7] : values[c & 7];
}
__device__ __forceinline__ uint8_t f4_code(float v)
{
    float x = fabsf(v), best = x;
    int code = 0;
    // Finite values past 5 are nearest to 6.0. From 2**25 up the FP32 distances
    // to 4.0 and 6.0 below round alike, and the even-code tie would keep 4.0.
    if(x > 5.0f && x <= 3.40282347e38f)
        code = 7;
    else
        for(int i = 1; i < 8; ++i)
        {
            float d = fabsf(x - f4_value(i));
            if(d < best || (d == best && !(i & 1)))
            {
                best = d;
                code = i;
            }
        }
    return code | ((__float_as_uint(v) >> 28) & 8);
}
__device__ __forceinline__ uint8_t quantize16(float (&v)[16], float gain, uint8_t (&codes)[16])
{
    float amax = 0;
    for(int i = 0; i < 16; ++i)
        amax = fmaxf(amax, fabsf(v[i]));
    uint8_t sf    = sf_code(gain * (amax > 0 ? amax / 6.0f : 1.0f));
    float divisor = sf_value(sf) / gain;
    for(int i = 0; i < 16; ++i)
        codes[i] = f4_code(v[i] / divisor);
    return sf;
}
__device__ __forceinline__ void
residual16(float (&v)[16], float gain, uint8_t sf, const uint8_t (&codes)[16])
{
    for(int i = 0; i < 16; ++i)
        v[i] -= f4_value(codes[i]) * sf_value(sf) / gain;
}
__device__ __forceinline__ int c_offset(int t, int d)
{
    return (t / 16) * 4096 + (d / 32) * 256 + (t % 16) * 16 + (d % 32) / 2;
}
__device__ __forceinline__ int r_offset(int t, int byte)
{
    return (t / 16) * 2048 + (byte / 16) * 256 + (t % 16) * 16 + byte % 16;
}
} // namespace mla_nvfp4

// Alignment applies to the instruction entry, not just its descriptor.
extern "C" __global__ __attribute__((aligned(16384)))
__launch_bounds__(64) void mla_nvfp4_quantize_q_kernel(const uint16_t* q, uint8_t* qn, uint8_t* qa)
{
    using namespace mla_nvfp4;
    const uint64_t row = blockIdx.x;
    const int g        = threadIdx.x;
    qa += row * 128;
    if(g >= 36)
    {
        if(g < 60)
            qa[104 + g - 36] = 0x7f;
        return;
    }
    float v[16];
    uint8_t code[16];
    for(int i = 0; i < 16; ++i)
        v[i] = bf16(q + row * 576 + g * 16 + i);
    uint8_t sf   = quantize16(v, 6.72f, code);
    qa[64 + g]   = sf;
    uint8_t* dst = g < 32 ? qn + row * 256 + g * 8 : qa + (g - 32) * 8;
    for(int i = 0; i < 8; ++i)
        dst[i] = code[2 * i] | (code[2 * i + 1] << 4);
    if(g >= 32)
    {
        residual16(v, 6.72f, sf, code);
        sf               = quantize16(v, 6.72f, code);
        qa[100 + g - 32] = sf;
        for(int i = 0; i < 8; ++i)
            qa[32 + (g - 32) * 8 + i] = code[2 * i] | (code[2 * i + 1] << 4);
    }
}

extern "C" __global__ __attribute__((aligned(16384)))
__launch_bounds__(64) void mla_nvfp4_update_cache_kernel(const uint16_t* tiles,
                                                         const int32_t* pages,
                                                         const int32_t* groups,
                                                         const int32_t* lengths,
                                                         uint8_t* cn,
                                                         uint8_t* ca,
                                                         int64_t npages)
{
    using namespace mla_nvfp4;
    const int64_t update = blockIdx.x;
    const int page = pages[update], tile = groups[update], valid = lengths[update];
    // Data-dependent validation stays on device; callers supply valid, unique updates.
    if(page < 0 || page >= npages || tile < 0 || tile >= 4 || valid < 0 || valid > 16)
        return;
    cn += uint64_t(page) * 16384;
    ca += uint64_t(page) * 8192;
    tiles += update * 16 * 576;
    const int lane = threadIdx.x;
    if(lane < 32)
    {
        float amax = 0;
        for(int t = 0; t < valid; ++t)
            for(int d = 0; d < 16; ++d)
                amax = fmaxf(amax, fabsf(bf16(tiles + t * 576 + lane * 16 + d)));
        float ideal = amax > 0 ? amax / 6.0f : 1.0f;
        uint8_t sf  = sf_code(89.6f * ideal);
        for(int t = 0; t < 16; ++t)
        {
            int pt = tile * 16 + t;
            for(int d = 0; d < 16; d += 2)
            {
                uint8_t a = t < valid ? f4_code(bf16(tiles + t * 576 + lane * 16 + d) / ideal) : 0;
                uint8_t b =
                    t < valid ? f4_code(bf16(tiles + t * 576 + lane * 16 + d + 1) / ideal) : 0;
                cn[c_offset(pt, lane * 16 + d)] = a | (b << 4);
            }
            int offset = tile * 2048 + 1024 + (lane / 16) * 256 + t * 16 + lane % 16;
            ca[offset] = t < valid ? sf : 0x7f;
        }
    }
    else
        for(int g = lane - 32; g < 64; g += 32)
        {
            int t = g / 4, rb = g % 4, pt = tile * 16 + t;
            float v[16];
            uint8_t code[16];
            for(int i = 0; i < 16; ++i)
                v[i] = t < valid ? bf16(tiles + t * 576 + 512 + rb * 16 + i) : 0;
            uint8_t sf       = quantize16(v, 89.6f, code);
            int scale_offset = tile * 2048 + 1536 + t * 16 + rb;
            ca[scale_offset] = t < valid ? sf : 0x7f;
            for(int i = 0; i < 8; ++i)
                ca[r_offset(pt, rb * 8 + i)] = code[2 * i] | (code[2 * i + 1] << 4);
            residual16(v, 89.6f, sf, code);
            sf                   = quantize16(v, 89.6f, code);
            ca[scale_offset + 4] = t < valid ? sf : 0x7f;
            for(int i = 0; i < 8; ++i)
                ca[512 + r_offset(pt, rb * 8 + i)] = code[2 * i] | (code[2 * i + 1] << 4);
        }
    for(int i = lane; i < 16 * 24; i += 64)
    {
        int offset = tile * 2048 + (i < 128 ? 1536 + (i / 8) * 16 + 8 + i % 8 : 1792 + i - 128);
        ca[offset] = 0x7f;
    }
}
