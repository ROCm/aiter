// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "fused_dsv41/cache_pipeline.hpp"
#include "fused_dsv41/decode_prefetch.hpp"
#include <cmath>
#include <hip/hip_runtime.h>
using B = opus::dtypes::bf16;
#include "fused_dsv41/kernel_args.hpp"
using B8 = opus::vector_t<B, 8>;
using F4 = opus::vector_t<float, 4>;

__device__ __forceinline__ float f32(B x)
{
    return __uint_as_float(uint32_t(__builtin_bit_cast(uint16_t, x)) << 16);
}
__device__ __forceinline__ float packed_row_max(float x)
{
    auto a = __builtin_amdgcn_permlane32_swap(__float_as_uint(x), __float_as_uint(x), false, true);
    x      = fmaxf(__uint_as_float(a[0]), __uint_as_float(a[1]));
    auto b = __builtin_amdgcn_permlane16_swap(__float_as_uint(x), __float_as_uint(x), false, true);
    return fmaxf(__uint_as_float(b[0]), __uint_as_float(b[1]));
}

// H64 bounds FP8 decode live ranges and shares CSR validity in LDS.
// H128 uses eight waves so all heads share one decoded KV tile.
// Both keep head-major Q and the original output layout.
template <bool RotateQ>
__global__ __launch_bounds__(256, 2) void fused_dsv41_mxfp8_kernel_64(FusedDsv41Args a)
{
#if defined(__gfx950__)
    using T = opus_mla_v4_prefill_a16w16_16mx8_32nx1_traits<16, 32, 512, 4>;
    using namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, group = lane / 16;
    const int hl = wave * 16 + lane % 16, h = blockIdx.y * 64 + hl, t = blockIdx.x;
    const int hload = min(h, a.heads - 1);
    __shared__ __align__(16) B kv_tile[2 * T::smem_kv_tile_elems];
    const int64_t pos = a.positions64 ? static_cast<const int64_t*>(a.positions)[t]
                                      : static_cast<const int*>(a.positions)[t];
    const B* cs       = a.cos + pos * 32;
    const B* sn       = a.sin + pos * 32;
    opus::vector_t<B, 128> q;
    opus::vector_t<float, 8> o[16];
    opus::static_for<16>([&](auto chunk) {
        opus::clear(o[int(chunk)]);
        int d = int(chunk) * 32 + group * 8;
        B8 x  = *reinterpret_cast<const B8*>(a.q + (int64_t(t) * a.heads + hload) * 512 + d);
        if constexpr(RotateQ && int(chunk) >= 14)
        {
            opus::static_for<4>([&](auto pair) {
                int i = int(pair) * 2, ri = (d + i - 448) / 2;
                float x0 = f32(x[i]), x1 = f32(x[i + 1]), c = f32(cs[ri]), s = f32(sn[ri]);
                x[i]     = static_cast<B>(fmaf(x0, c, -x1 * s));
                x[i + 1] = static_cast<B>(fmaf(x0, s, x1 * c));
            });
        }
        opus::set_slice(q, x, opus::number<int(chunk) * 8>{}, opus::number<int(chunk) * 8 + 8>{});
    });
    constexpr float log2e = 1.4426950408889634f;
    float m = -__FLT_MAX__, actual_max = -INFINITY, l = 0;
    int begin = max(0, min(a.prefix_capacity, a.prefix_indptr[t]));
    int end   = max(begin, min(a.prefix_capacity, a.prefix_indptr[t + 1]));
    if(end > begin)
    {
        PackedCacheSource64<true> src{a.pool, a.prefix + begin, end - begin, a.pool_bytes};
        packed_prefill_accum64<T>(src,
                                  end - begin,
                                  (end - begin + 31) / 32,
                                  reinterpret_cast<char*>(kv_tile),
                                  q,
                                  o,
                                  actual_max,
                                  m,
                                  l,
                                  a.softmax_scale * log2e);
    }
    __syncthreads();
    if(a.extend_indptr)
    {
        begin = max(0, min(a.extend_capacity, a.extend_indptr[t]));
        end   = max(begin, min(a.extend_capacity, a.extend_indptr[t + 1]));
        if(end > begin)
        {
            PackedCacheSource64<false> src{
                a.extend_kv, a.extend + begin, end - begin, a.extend_rows};
            packed_prefill_accum64<T>(src,
                                      end - begin,
                                      (end - begin + 31) / 32,
                                      reinterpret_cast<char*>(kv_tile),
                                      q,
                                      o,
                                      actual_max,
                                      m,
                                      l,
                                      a.softmax_scale * log2e);
        }
    }
    __syncthreads();
    auto* staging = reinterpret_cast<uint8_t*>(kv_tile);
    if(h < a.heads)
    {
        const float sink    = a.sink ? a.sink[h] * log2e : -INFINITY;
        const float final_m = fmaxf(m, sink);
        const float rescale = __builtin_amdgcn_exp2f(m - final_m);
        const float denom   = l * rescale + __builtin_amdgcn_exp2f(sink - final_m);
        const float inv     = denom > 0 ? rescale / denom : 0;
        if(group == 0)
        {
            a.lse[int64_t(t) * a.heads + h]        = l > 0 ? m / log2e + logf(l) : INFINITY;
            a.max_logits[int64_t(t) * a.heads + h] = actual_max / log2e;
        }
        opus::static_for<16>([&](auto chunk) {
            F4 x[2] = {opus::slice(o[int(chunk)], opus::number<0>{}, opus::number<4>{}),
                       opus::slice(o[int(chunk)], opus::number<4>{}, opus::number<8>{})};
            // ATOM's attention returns BF16 before inverse_rope_group_quant.
            opus::static_for<2>([&](auto n) {
                opus::static_for<4>([&](auto i) {
                    x[int(n)][int(i)] = f32(static_cast<B>(x[int(n)][int(i)] * inv));
                });
                if constexpr(int(chunk) >= 14)
                {
                    opus::static_for<2>([&](auto pair) {
                        int i   = int(pair) * 2,
                            ri  = (int(chunk) * 32 + int(n) * 16 + group * 4 + i - 448) / 2;
                        float c = f32(cs[ri]), s = f32(sn[ri]), x0 = x[int(n)][i],
                              x1         = x[int(n)][i + 1];
                        x[int(n)][i]     = x0 * c + x1 * s;
                        x[int(n)][i + 1] = x1 * c - x0 * s;
                    });
                }
            });
            float amax = 0;
            opus::static_for<2>([&](auto n) {
                opus::static_for<4>([&](auto i) { amax = fmaxf(amax, fabsf(x[int(n)][int(i)])); });
            });
            amax          = fmaxf(packed_row_max(amax), 1e-8f);
            unsigned bits = __float_as_uint(amax / 448.0f);
            int exponent  = int((bits - 1) >> 23) + 1 - 127;
            float mul     = __uint_as_float(uint32_t(127 - exponent) << 23);
            if(group == 0)
                a.scales[(int64_t(t) * a.heads + h) * 16 + int(chunk)] = exponent + 127;
            opus::static_for<2>([&](auto n) {
                int d      = int(chunk) * 32 + int(n) * 16 + group * 4;
                auto quant = opus::cast<opus::fp8_t>(x[int(n)] * mul);
                *reinterpret_cast<uint32_t*>(staging + hl * 528 + d) =
                    __builtin_bit_cast(uint32_t, quant);
                if(a.debug_out)
                    *reinterpret_cast<F4*>(a.debug_out + (int64_t(t) * a.heads + h) * 512 + d) =
                        x[int(n)];
            });
        });
    }
    __syncthreads();
    using U4 = opus::vector_t<uint32_t, 4>;
    for(int i = tid; i < min(64, a.heads - int(blockIdx.y) * 64) * 32; i += 256)
        *reinterpret_cast<U4*>(a.out + (int64_t(t) * a.heads + blockIdx.y * 64) * 512 + i * 16) =
            *reinterpret_cast<const U4*>(staging + (i / 32) * 528 + (i % 32) * 16);
#endif
}

template <bool RotateQ, bool LateLoad>
__global__ __launch_bounds__(512, 1) void fused_dsv41_mxfp8_kernel_128(FusedDsv41Args a)
{
#if defined(__gfx950__)
    using T = opus_mla_v4_prefill_a16w16_16mx8_32nx1_traits<16, 32, 512, 8>;
    using namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, group = lane / 16;
    const int hl = wave * 16 + lane % 16, h = blockIdx.y * 128 + hl, t = blockIdx.x;
    const int hload = min(h, a.heads - 1);
    __shared__ __align__(16) B kv_tile[2 * T::smem_kv_tile_elems];
    const int64_t pos = a.positions64 ? static_cast<const int64_t*>(a.positions)[t]
                                      : static_cast<const int*>(a.positions)[t];
    const B* cs       = a.cos + pos * 32;
    const B* sn       = a.sin + pos * 32;
    opus::vector_t<B, 128> q;
    opus::vector_t<float, 8> o[16];
    opus::static_for<16>([&](auto chunk) {
        opus::clear(o[int(chunk)]);
        int d = int(chunk) * 32 + group * 8;
        B8 x  = *reinterpret_cast<const B8*>(a.q + (int64_t(t) * a.heads + hload) * 512 + d);
        if constexpr(RotateQ && int(chunk) >= 14)
        {
            opus::static_for<4>([&](auto pair) {
                int i = int(pair) * 2, ri = (d + i - 448) / 2;
                float x0 = f32(x[i]), x1 = f32(x[i + 1]), c = f32(cs[ri]), s = f32(sn[ri]);
                x[i]     = static_cast<B>(fmaf(x0, c, -x1 * s));
                x[i + 1] = static_cast<B>(fmaf(x0, s, x1 * c));
            });
        }
        opus::set_slice(q, x, opus::number<int(chunk) * 8>{}, opus::number<int(chunk) * 8 + 8>{});
    });
    constexpr float log2e = 1.4426950408889634f;
    float m = -__FLT_MAX__, actual_max = -INFINITY, l = 0;
    int begin = max(0, min(a.prefix_capacity, a.prefix_indptr[t]));
    int end   = max(begin, min(a.prefix_capacity, a.prefix_indptr[t + 1]));
    if(end > begin)
    {
        PackedCacheSource128<true, LateLoad> src{
            a.pool, a.prefix + begin, end - begin, a.pool_bytes};
        packed_prefill_accum128<T, LateLoad>(src,
                                             end - begin,
                                             (end - begin + 31) / 32,
                                             reinterpret_cast<char*>(kv_tile),
                                             q,
                                             o,
                                             actual_max,
                                             m,
                                             l,
                                             a.softmax_scale * log2e);
    }
    __syncthreads();
    if(a.extend_indptr)
    {
        begin = max(0, min(a.extend_capacity, a.extend_indptr[t]));
        end   = max(begin, min(a.extend_capacity, a.extend_indptr[t + 1]));
        if(end > begin)
        {
            PackedCacheSource128<false, LateLoad> src{
                a.extend_kv, a.extend + begin, end - begin, a.extend_rows};
            packed_prefill_accum128<T, LateLoad>(src,
                                                 end - begin,
                                                 (end - begin + 31) / 32,
                                                 reinterpret_cast<char*>(kv_tile),
                                                 q,
                                                 o,
                                                 actual_max,
                                                 m,
                                                 l,
                                                 a.softmax_scale * log2e);
        }
    }
    __syncthreads();
    auto* staging = reinterpret_cast<uint8_t*>(kv_tile);
    if(h < a.heads)
    {
        const float sink    = a.sink ? a.sink[h] * log2e : -INFINITY;
        const float final_m = fmaxf(m, sink);
        const float rescale = __builtin_amdgcn_exp2f(m - final_m);
        const float denom   = l * rescale + __builtin_amdgcn_exp2f(sink - final_m);
        const float inv     = denom > 0 ? rescale / denom : 0;
        if(group == 0)
        {
            a.lse[int64_t(t) * a.heads + h]        = l > 0 ? m / log2e + logf(l) : INFINITY;
            a.max_logits[int64_t(t) * a.heads + h] = actual_max / log2e;
        }
        opus::static_for<16>([&](auto chunk) {
            F4 x[2] = {opus::slice(o[int(chunk)], opus::number<0>{}, opus::number<4>{}),
                       opus::slice(o[int(chunk)], opus::number<4>{}, opus::number<8>{})};
            // ATOM's attention returns BF16 before inverse_rope_group_quant.
            opus::static_for<2>([&](auto n) {
                opus::static_for<4>([&](auto i) {
                    x[int(n)][int(i)] = f32(static_cast<B>(x[int(n)][int(i)] * inv));
                });
                if constexpr(int(chunk) >= 14)
                {
                    opus::static_for<2>([&](auto pair) {
                        int i   = int(pair) * 2,
                            ri  = (int(chunk) * 32 + int(n) * 16 + group * 4 + i - 448) / 2;
                        float c = f32(cs[ri]), s = f32(sn[ri]), x0 = x[int(n)][i],
                              x1         = x[int(n)][i + 1];
                        x[int(n)][i]     = x0 * c + x1 * s;
                        x[int(n)][i + 1] = x1 * c - x0 * s;
                    });
                }
            });
            float amax = 0;
            opus::static_for<2>([&](auto n) {
                opus::static_for<4>([&](auto i) { amax = fmaxf(amax, fabsf(x[int(n)][int(i)])); });
            });
            amax          = fmaxf(packed_row_max(amax), 1e-8f);
            unsigned bits = __float_as_uint(amax / 448.0f);
            int exponent  = int((bits - 1) >> 23) + 1 - 127;
            float mul     = __uint_as_float(uint32_t(127 - exponent) << 23);
            if(group == 0)
                a.scales[(int64_t(t) * a.heads + h) * 16 + int(chunk)] = exponent + 127;
            opus::static_for<2>([&](auto n) {
                int d      = int(chunk) * 32 + int(n) * 16 + group * 4;
                auto quant = opus::cast<opus::fp8_t>(x[int(n)] * mul);
                *reinterpret_cast<uint32_t*>(staging + hl * 528 + d) =
                    __builtin_bit_cast(uint32_t, quant);
                if(a.debug_out)
                    *reinterpret_cast<F4*>(a.debug_out + (int64_t(t) * a.heads + h) * 512 + d) =
                        x[int(n)];
            });
        });
    }
    __syncthreads();
    using U4 = opus::vector_t<uint32_t, 4>;
    for(int i = tid; i < min(128, a.heads - int(blockIdx.y) * 128) * 32; i += 512)
        *reinterpret_cast<U4*>(a.out + (int64_t(t) * a.heads + blockIdx.y * 128) * 512 + i * 16) =
            *reinterpret_cast<const U4*>(staging + (i / 32) * 528 + (i % 32) * 16);
#endif
}

// Small-head kernels reuse a common KV panel and compute disjoint PV
// dimensions. QK repeats for each output partition, as it did for padded heads.
// No weight/channel permutation or extra launch is required.
template <bool RotateQ, int Parts = 1>
__global__ __launch_bounds__(256, 2) void fused_dsv41_mxfp8_kernel_small(FusedDsv41Args a)
{
#if defined(__gfx950__)
    using T = opus_mla_v4_prefill_a16w16_16mx8_32nx1_traits<16, 32, 512, 4>;
    using namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, group = lane / 16;
    constexpr int BH = 64 / Parts, OSlices = 16 / Parts;
    const int part = wave % Parts;
    const int hl = (wave / Parts) * 16 + lane % 16, h = blockIdx.y * BH + hl, t = blockIdx.x;
    const int hload = min(h, a.heads - 1);
    __shared__ __align__(16) B kv_tile[2 * T::smem_kv_tile_elems];
    const int64_t pos = a.positions64 ? static_cast<const int64_t*>(a.positions)[t]
                                      : static_cast<const int*>(a.positions)[t];
    const B* cs       = a.cos + pos * 32;
    const B* sn       = a.sin + pos * 32;
    opus::vector_t<B, 128> q;
    opus::vector_t<float, 8> o[OSlices];
    opus::static_for<OSlices>([&](auto i) { opus::clear(o[int(i)]); });
    opus::static_for<16>([&](auto chunk) {
        int d = int(chunk) * 32 + group * 8;
        B8 x  = *reinterpret_cast<const B8*>(a.q + (int64_t(t) * a.heads + hload) * 512 + d);
        if constexpr(RotateQ && int(chunk) >= 14)
        {
            opus::static_for<4>([&](auto pair) {
                int i = int(pair) * 2, ri = (d + i - 448) / 2;
                float x0 = f32(x[i]), x1 = f32(x[i + 1]), c = f32(cs[ri]), s = f32(sn[ri]);
                x[i]     = static_cast<B>(fmaf(x0, c, -x1 * s));
                x[i + 1] = static_cast<B>(fmaf(x0, s, x1 * c));
            });
        }
        opus::set_slice(q, x, opus::number<int(chunk) * 8>{}, opus::number<int(chunk) * 8 + 8>{});
    });
    constexpr float log2e = 1.4426950408889634f;
    float m = -__FLT_MAX__, actual_max = -INFINITY, l = 0;
    int begin = max(0, min(a.prefix_capacity, a.prefix_indptr[t]));
    int end   = max(begin, min(a.prefix_capacity, a.prefix_indptr[t + 1]));
    if(end > begin)
    {
        PackedCacheSourceSmall<true> src{a.pool, a.prefix + begin, end - begin, a.pool_bytes};
        packed_prefill_accum_small<T, Parts>(src,
                                             end - begin,
                                             (end - begin + 31) / 32,
                                             reinterpret_cast<char*>(kv_tile),
                                             q,
                                             o,
                                             actual_max,
                                             m,
                                             l,
                                             a.softmax_scale * log2e);
    }
    __syncthreads();
    if(a.extend_indptr)
    {
        begin = max(0, min(a.extend_capacity, a.extend_indptr[t]));
        end   = max(begin, min(a.extend_capacity, a.extend_indptr[t + 1]));
        if(end > begin)
        {
            PackedCacheSourceSmall<false> src{
                a.extend_kv, a.extend + begin, end - begin, a.extend_rows};
            packed_prefill_accum_small<T, Parts>(src,
                                                 end - begin,
                                                 (end - begin + 31) / 32,
                                                 reinterpret_cast<char*>(kv_tile),
                                                 q,
                                                 o,
                                                 actual_max,
                                                 m,
                                                 l,
                                                 a.softmax_scale * log2e);
        }
    }
    __syncthreads();
    auto* staging = reinterpret_cast<uint8_t*>(kv_tile);
    if(h < a.heads)
    {
        const float sink    = a.sink ? a.sink[h] * log2e : -INFINITY;
        const float final_m = fmaxf(m, sink);
        const float rescale = __builtin_amdgcn_exp2f(m - final_m);
        const float denom   = l * rescale + __builtin_amdgcn_exp2f(sink - final_m);
        const float inv     = denom > 0 ? rescale / denom : 0;
        if(group == 0 && part == 0)
        {
            a.lse[int64_t(t) * a.heads + h]        = l > 0 ? m / log2e + logf(l) : INFINITY;
            a.max_logits[int64_t(t) * a.heads + h] = actual_max / log2e;
        }
        opus::static_for<OSlices>([&](auto local_chunk) {
            const int chunk = int(local_chunk) + part * OSlices;
            F4 x[2] = {opus::slice(o[int(local_chunk)], opus::number<0>{}, opus::number<4>{}),
                       opus::slice(o[int(local_chunk)], opus::number<4>{}, opus::number<8>{})};
            // ATOM's attention returns BF16 before inverse_rope_group_quant.
            opus::static_for<2>([&](auto n) {
                opus::static_for<4>([&](auto i) {
                    x[int(n)][int(i)] = f32(static_cast<B>(x[int(n)][int(i)] * inv));
                });
                if(chunk >= 14)
                {
                    opus::static_for<2>([&](auto pair) {
                        int i   = int(pair) * 2,
                            ri  = (int(chunk) * 32 + int(n) * 16 + group * 4 + i - 448) / 2;
                        float c = f32(cs[ri]), s = f32(sn[ri]), x0 = x[int(n)][i],
                              x1         = x[int(n)][i + 1];
                        x[int(n)][i]     = x0 * c + x1 * s;
                        x[int(n)][i + 1] = x1 * c - x0 * s;
                    });
                }
            });
            float amax = 0;
            opus::static_for<2>([&](auto n) {
                opus::static_for<4>([&](auto i) { amax = fmaxf(amax, fabsf(x[int(n)][int(i)])); });
            });
            amax          = fmaxf(packed_row_max(amax), 1e-8f);
            unsigned bits = __float_as_uint(amax / 448.0f);
            int exponent  = int((bits - 1) >> 23) + 1 - 127;
            float mul     = __uint_as_float(uint32_t(127 - exponent) << 23);
            if(group == 0)
                a.scales[(int64_t(t) * a.heads + h) * 16 + int(chunk)] = exponent + 127;
            opus::static_for<2>([&](auto n) {
                int d      = int(chunk) * 32 + int(n) * 16 + group * 4;
                auto quant = opus::cast<opus::fp8_t>(x[int(n)] * mul);
                *reinterpret_cast<uint32_t*>(staging + hl * 528 + d) =
                    __builtin_bit_cast(uint32_t, quant);
                if(a.debug_out)
                    *reinterpret_cast<F4*>(a.debug_out + (int64_t(t) * a.heads + h) * 512 + d) =
                        x[int(n)];
            });
        });
    }
    __syncthreads();
    using U4 = opus::vector_t<uint32_t, 4>;
    for(int i = tid; i < min(BH, a.heads - int(blockIdx.y) * BH) * 32; i += 256)
        *reinterpret_cast<U4*>(a.out + (int64_t(t) * a.heads + blockIdx.y * BH) * 512 + i * 16) =
            *reinterpret_cast<const U4*>(staging + (i / 32) * 528 + (i % 32) * 16);
#endif
}

template <bool RotateQ>
__global__ __launch_bounds__(256, 2) void fused_dsv41_mxfp8_decode_64(FusedDsv41Args a)
{
#if defined(__gfx950__)
    using T = opus_mla_v4_prefill_a16w16_16mx8_32nx1_traits<16, 32, 512, 4>;
    using namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, group = lane / 16;
    const int hl = wave * 16 + lane % 16, h = blockIdx.y * 64 + hl, t = blockIdx.x;
    const int hload = min(h, a.heads - 1);
    __shared__ __align__(16) B kv_tile[2 * T::smem_kv_tile_elems];
    const int64_t pos = a.positions64 ? static_cast<const int64_t*>(a.positions)[t]
                                      : static_cast<const int*>(a.positions)[t];
    const B* cs       = a.cos + pos * 32;
    const B* sn       = a.sin + pos * 32;
    opus::vector_t<B, 128> q;
    opus::vector_t<float, 8> o[16];
    opus::static_for<16>([&](auto chunk) {
        opus::clear(o[int(chunk)]);
        int d = int(chunk) * 32 + group * 8;
        B8 x  = *reinterpret_cast<const B8*>(a.q + (int64_t(t) * a.heads + hload) * 512 + d);
        if constexpr(RotateQ && int(chunk) >= 14)
        {
            opus::static_for<4>([&](auto pair) {
                int i = int(pair) * 2, ri = (d + i - 448) / 2;
                float x0 = f32(x[i]), x1 = f32(x[i + 1]), c = f32(cs[ri]), s = f32(sn[ri]);
                x[i]     = static_cast<B>(fmaf(x0, c, -x1 * s));
                x[i + 1] = static_cast<B>(fmaf(x0, s, x1 * c));
            });
        }
        opus::set_slice(q, x, opus::number<int(chunk) * 8>{}, opus::number<int(chunk) * 8 + 8>{});
    });
    constexpr float log2e = 1.4426950408889634f;
    float m = -__FLT_MAX__, actual_max = -INFINITY, l = 0;
    int begin = max(0, min(a.prefix_capacity, a.prefix_indptr[t]));
    int end   = max(begin, min(a.prefix_capacity, a.prefix_indptr[t + 1]));
    if(end > begin)
    {
        PackedDecodeSource64<true> src{a.pool, a.prefix + begin, end - begin, a.pool_bytes};
        packed_decode_accum64<T>(src,
                                 end - begin,
                                 (end - begin + 31) / 32,
                                 reinterpret_cast<char*>(kv_tile),
                                 q,
                                 o,
                                 actual_max,
                                 m,
                                 l,
                                 a.softmax_scale * log2e);
    }
    __syncthreads();
    if(a.extend_indptr)
    {
        begin = max(0, min(a.extend_capacity, a.extend_indptr[t]));
        end   = max(begin, min(a.extend_capacity, a.extend_indptr[t + 1]));
        if(end > begin)
        {
            PackedDecodeSource64<false> src{
                a.extend_kv, a.extend + begin, end - begin, a.extend_rows};
            packed_decode_accum64<T>(src,
                                     end - begin,
                                     (end - begin + 31) / 32,
                                     reinterpret_cast<char*>(kv_tile),
                                     q,
                                     o,
                                     actual_max,
                                     m,
                                     l,
                                     a.softmax_scale * log2e);
        }
    }
    __syncthreads();
    auto* staging = reinterpret_cast<uint8_t*>(kv_tile);
    if(h < a.heads)
    {
        const float sink    = a.sink ? a.sink[h] * log2e : -INFINITY;
        const float final_m = fmaxf(m, sink);
        const float rescale = __builtin_amdgcn_exp2f(m - final_m);
        const float denom   = l * rescale + __builtin_amdgcn_exp2f(sink - final_m);
        const float inv     = denom > 0 ? rescale / denom : 0;
        if(group == 0)
        {
            a.lse[int64_t(t) * a.heads + h]        = l > 0 ? m / log2e + logf(l) : INFINITY;
            a.max_logits[int64_t(t) * a.heads + h] = actual_max / log2e;
        }
        opus::static_for<16>([&](auto chunk) {
            F4 x[2] = {opus::slice(o[int(chunk)], opus::number<0>{}, opus::number<4>{}),
                       opus::slice(o[int(chunk)], opus::number<4>{}, opus::number<8>{})};
            // ATOM's attention returns BF16 before inverse_rope_group_quant.
            opus::static_for<2>([&](auto n) {
                opus::static_for<4>([&](auto i) {
                    x[int(n)][int(i)] = f32(static_cast<B>(x[int(n)][int(i)] * inv));
                });
                if constexpr(int(chunk) >= 14)
                {
                    opus::static_for<2>([&](auto pair) {
                        int i   = int(pair) * 2,
                            ri  = (int(chunk) * 32 + int(n) * 16 + group * 4 + i - 448) / 2;
                        float c = f32(cs[ri]), s = f32(sn[ri]), x0 = x[int(n)][i],
                              x1         = x[int(n)][i + 1];
                        x[int(n)][i]     = x0 * c + x1 * s;
                        x[int(n)][i + 1] = x1 * c - x0 * s;
                    });
                }
            });
            float amax = 0;
            opus::static_for<2>([&](auto n) {
                opus::static_for<4>([&](auto i) { amax = fmaxf(amax, fabsf(x[int(n)][int(i)])); });
            });
            amax          = fmaxf(packed_row_max(amax), 1e-8f);
            unsigned bits = __float_as_uint(amax / 448.0f);
            int exponent  = int((bits - 1) >> 23) + 1 - 127;
            float mul     = __uint_as_float(uint32_t(127 - exponent) << 23);
            if(group == 0)
                a.scales[(int64_t(t) * a.heads + h) * 16 + int(chunk)] = exponent + 127;
            opus::static_for<2>([&](auto n) {
                int d      = int(chunk) * 32 + int(n) * 16 + group * 4;
                auto quant = opus::cast<opus::fp8_t>(x[int(n)] * mul);
                *reinterpret_cast<uint32_t*>(staging + hl * 528 + d) =
                    __builtin_bit_cast(uint32_t, quant);
                if(a.debug_out)
                    *reinterpret_cast<F4*>(a.debug_out + (int64_t(t) * a.heads + h) * 512 + d) =
                        x[int(n)];
            });
        });
    }
    __syncthreads();
    using U4 = opus::vector_t<uint32_t, 4>;
    for(int i = tid; i < min(64, a.heads - int(blockIdx.y) * 64) * 32; i += 256)
        *reinterpret_cast<U4*>(a.out + (int64_t(t) * a.heads + blockIdx.y * 64) * 512 + i * 16) =
            *reinterpret_cast<const U4*>(staging + (i / 32) * 528 + (i % 32) * 16);
#endif
}

template <bool RotateQ, bool LateLoad>
__global__ __launch_bounds__(512, 1) void fused_dsv41_mxfp8_decode_128(FusedDsv41Args a)
{
#if defined(__gfx950__)
    using T = opus_mla_v4_prefill_a16w16_16mx8_32nx1_traits<16, 32, 512, 8>;
    using namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, group = lane / 16;
    const int hl = wave * 16 + lane % 16, h = blockIdx.y * 128 + hl, t = blockIdx.x;
    const int hload = min(h, a.heads - 1);
    __shared__ __align__(16) B kv_tile[2 * T::smem_kv_tile_elems];
    const int64_t pos = a.positions64 ? static_cast<const int64_t*>(a.positions)[t]
                                      : static_cast<const int*>(a.positions)[t];
    const B* cs       = a.cos + pos * 32;
    const B* sn       = a.sin + pos * 32;
    opus::vector_t<B, 128> q;
    opus::vector_t<float, 8> o[16];
    opus::static_for<16>([&](auto chunk) {
        opus::clear(o[int(chunk)]);
        int d = int(chunk) * 32 + group * 8;
        B8 x  = *reinterpret_cast<const B8*>(a.q + (int64_t(t) * a.heads + hload) * 512 + d);
        if constexpr(RotateQ && int(chunk) >= 14)
        {
            opus::static_for<4>([&](auto pair) {
                int i = int(pair) * 2, ri = (d + i - 448) / 2;
                float x0 = f32(x[i]), x1 = f32(x[i + 1]), c = f32(cs[ri]), s = f32(sn[ri]);
                x[i]     = static_cast<B>(fmaf(x0, c, -x1 * s));
                x[i + 1] = static_cast<B>(fmaf(x0, s, x1 * c));
            });
        }
        opus::set_slice(q, x, opus::number<int(chunk) * 8>{}, opus::number<int(chunk) * 8 + 8>{});
    });
    constexpr float log2e = 1.4426950408889634f;
    float m = -__FLT_MAX__, actual_max = -INFINITY, l = 0;
    int begin = max(0, min(a.prefix_capacity, a.prefix_indptr[t]));
    int end   = max(begin, min(a.prefix_capacity, a.prefix_indptr[t + 1]));
    if(end > begin)
    {
        PackedDecodeSource128<true, LateLoad> src{
            a.pool, a.prefix + begin, end - begin, a.pool_bytes};
        packed_decode_accum128<T, LateLoad>(src,
                                            end - begin,
                                            (end - begin + 31) / 32,
                                            reinterpret_cast<char*>(kv_tile),
                                            q,
                                            o,
                                            actual_max,
                                            m,
                                            l,
                                            a.softmax_scale * log2e);
    }
    __syncthreads();
    if(a.extend_indptr)
    {
        begin = max(0, min(a.extend_capacity, a.extend_indptr[t]));
        end   = max(begin, min(a.extend_capacity, a.extend_indptr[t + 1]));
        if(end > begin)
        {
            PackedDecodeSource128<false, LateLoad> src{
                a.extend_kv, a.extend + begin, end - begin, a.extend_rows};
            packed_decode_accum128<T, LateLoad>(src,
                                                end - begin,
                                                (end - begin + 31) / 32,
                                                reinterpret_cast<char*>(kv_tile),
                                                q,
                                                o,
                                                actual_max,
                                                m,
                                                l,
                                                a.softmax_scale * log2e);
        }
    }
    __syncthreads();
    auto* staging = reinterpret_cast<uint8_t*>(kv_tile);
    if(h < a.heads)
    {
        const float sink    = a.sink ? a.sink[h] * log2e : -INFINITY;
        const float final_m = fmaxf(m, sink);
        const float rescale = __builtin_amdgcn_exp2f(m - final_m);
        const float denom   = l * rescale + __builtin_amdgcn_exp2f(sink - final_m);
        const float inv     = denom > 0 ? rescale / denom : 0;
        if(group == 0)
        {
            a.lse[int64_t(t) * a.heads + h]        = l > 0 ? m / log2e + logf(l) : INFINITY;
            a.max_logits[int64_t(t) * a.heads + h] = actual_max / log2e;
        }
        opus::static_for<16>([&](auto chunk) {
            F4 x[2] = {opus::slice(o[int(chunk)], opus::number<0>{}, opus::number<4>{}),
                       opus::slice(o[int(chunk)], opus::number<4>{}, opus::number<8>{})};
            // ATOM's attention returns BF16 before inverse_rope_group_quant.
            opus::static_for<2>([&](auto n) {
                opus::static_for<4>([&](auto i) {
                    x[int(n)][int(i)] = f32(static_cast<B>(x[int(n)][int(i)] * inv));
                });
                if constexpr(int(chunk) >= 14)
                {
                    opus::static_for<2>([&](auto pair) {
                        int i   = int(pair) * 2,
                            ri  = (int(chunk) * 32 + int(n) * 16 + group * 4 + i - 448) / 2;
                        float c = f32(cs[ri]), s = f32(sn[ri]), x0 = x[int(n)][i],
                              x1         = x[int(n)][i + 1];
                        x[int(n)][i]     = x0 * c + x1 * s;
                        x[int(n)][i + 1] = x1 * c - x0 * s;
                    });
                }
            });
            float amax = 0;
            opus::static_for<2>([&](auto n) {
                opus::static_for<4>([&](auto i) { amax = fmaxf(amax, fabsf(x[int(n)][int(i)])); });
            });
            amax          = fmaxf(packed_row_max(amax), 1e-8f);
            unsigned bits = __float_as_uint(amax / 448.0f);
            int exponent  = int((bits - 1) >> 23) + 1 - 127;
            float mul     = __uint_as_float(uint32_t(127 - exponent) << 23);
            if(group == 0)
                a.scales[(int64_t(t) * a.heads + h) * 16 + int(chunk)] = exponent + 127;
            opus::static_for<2>([&](auto n) {
                int d      = int(chunk) * 32 + int(n) * 16 + group * 4;
                auto quant = opus::cast<opus::fp8_t>(x[int(n)] * mul);
                *reinterpret_cast<uint32_t*>(staging + hl * 528 + d) =
                    __builtin_bit_cast(uint32_t, quant);
                if(a.debug_out)
                    *reinterpret_cast<F4*>(a.debug_out + (int64_t(t) * a.heads + h) * 512 + d) =
                        x[int(n)];
            });
        });
    }
    __syncthreads();
    using U4 = opus::vector_t<uint32_t, 4>;
    for(int i = tid; i < min(128, a.heads - int(blockIdx.y) * 128) * 32; i += 512)
        *reinterpret_cast<U4*>(a.out + (int64_t(t) * a.heads + blockIdx.y * 128) * 512 + i * 16) =
            *reinterpret_cast<const U4*>(staging + (i / 32) * 528 + (i % 32) * 16);
#endif
}

hipError_t launch_fused_dsv41_mxfp8(const FusedDsv41Args* a, int rotate_q, hipStream_t stream)
{
    if(!a->tokens)
        return hipSuccess;
    // Full-FP8 decode: prefetch raw bytes before QK; commit after QK (H64)
    // or after PV (H128). Other shapes retain the validated v4 kernels.
    if(!a->extend_indptr && a->tokens <= 1024 && (a->heads == 64 || a->heads == 128))
    {
        dim3 grid(a->tokens);
        if(a->heads == 64)
        {
            if(rotate_q)
                fused_dsv41_mxfp8_decode_64<true><<<grid, 256, 0, stream>>>(*a);
            else
                fused_dsv41_mxfp8_decode_64<false><<<grid, 256, 0, stream>>>(*a);
        }
        else
        {
            if(rotate_q)
                fused_dsv41_mxfp8_decode_128<true, false><<<grid, 512, 0, stream>>>(*a);
            else
                fused_dsv41_mxfp8_decode_128<false, false><<<grid, 512, 0, stream>>>(*a);
        }
        return hipGetLastError();
    }
    if(a->heads <= 16)
    {
        dim3 grid(a->tokens);
        if(rotate_q)
            fused_dsv41_mxfp8_kernel_small<true, 4><<<grid, 256, 0, stream>>>(*a);
        else
            fused_dsv41_mxfp8_kernel_small<false, 4><<<grid, 256, 0, stream>>>(*a);
    }
    else if(a->heads == 32)
    {
        dim3 grid(a->tokens);
        if(rotate_q)
            fused_dsv41_mxfp8_kernel_small<true, 2><<<grid, 256, 0, stream>>>(*a);
        else
            fused_dsv41_mxfp8_kernel_small<false, 2><<<grid, 256, 0, stream>>>(*a);
    }
    else if(a->heads == 64)
    {
        dim3 grid(a->tokens);
        if(rotate_q)
            fused_dsv41_mxfp8_kernel_64<true><<<grid, 256, 0, stream>>>(*a);
        else
            fused_dsv41_mxfp8_kernel_64<false><<<grid, 256, 0, stream>>>(*a);
    }
    else
    {
        dim3 grid(a->tokens);
        // H128 early loading wins at T<=1024; the late loader wins at
        // T2048 and T4096. Keep the unmeasured interval on the original path.
        if(a->tokens >= 2048)
        {
            if(rotate_q)
                fused_dsv41_mxfp8_kernel_128<true, true><<<grid, 512, 0, stream>>>(*a);
            else
                fused_dsv41_mxfp8_kernel_128<false, true><<<grid, 512, 0, stream>>>(*a);
        }
        else
        {
            if(rotate_q)
                fused_dsv41_mxfp8_kernel_128<true, false><<<grid, 512, 0, stream>>>(*a);
            else
                fused_dsv41_mxfp8_kernel_128<false, false><<<grid, 512, 0, stream>>>(*a);
        }
    }
    return hipGetLastError();
}
