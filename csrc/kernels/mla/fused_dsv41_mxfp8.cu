// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "fused_dsv41/cache_pipeline.hpp"
#include "fused_dsv41/decode_fp8_qk.hpp"
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
template <bool RotateQ, int PageSize>
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
        PackedCacheSource64<(PageSize != -2), PageSize> src{
            a.pool,
            a.prefix_at<PageSize>(begin),
            end - begin,
            (PageSize == -2 ? a.cache.num_rows : a.pool_bytes),
            a.cache};
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

template <bool RotateQ, bool LateLoad, int PageSize>
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
        PackedCacheSource128<(PageSize != -2), LateLoad, PageSize> src{
            a.pool,
            a.prefix_at<PageSize>(begin),
            end - begin,
            (PageSize == -2 ? a.cache.num_rows : a.pool_bytes),
            a.cache};
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
template <bool RotateQ, int Parts, int PageSize>
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
        PackedCacheSourceSmall<(PageSize != -2), PageSize> src{
            a.pool,
            a.prefix_at<PageSize>(begin),
            end - begin,
            (PageSize == -2 ? a.cache.num_rows : a.pool_bytes),
            a.cache};
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

template <bool RotateQ, int PageSize>
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
        PackedDecodeSource64<(PageSize != -2), PageSize> src{
            a.pool,
            a.prefix_at<PageSize>(begin),
            end - begin,
            (PageSize == -2 ? a.cache.num_rows : a.pool_bytes),
            a.cache};
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

template <bool RotateQ, int PageSize, bool PreparedQ = false, bool LateCommit = false>
__global__ __launch_bounds__(256, 1) void fused_dsv41_fp8_qk_decode_64(FusedDsv41Args a)
{
#if defined(__gfx950__)
    using T = opus_mla_v4_prefill_a16w16_16mx8_32nx1_traits<16, 32, 512, 4>;
    using namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, group = lane / 16;
    const int hl = wave * 16 + lane % 16, h = blockIdx.y * 64 + hl, t = blockIdx.x;
    const int hload = min(h, a.heads - 1);
    __shared__ __align__(16) B kv_tile[T::smem_kv_tile_elems];
    const int64_t pos = a.positions64 ? static_cast<const int64_t*>(a.positions)[t]
                                      : static_cast<const int*>(a.positions)[t];
    const B* cs       = a.cos + pos * 32;
    const B* sn       = a.sin + pos * 32;
    __shared__ __align__(32) uint8_t raw_k[2 * (32 * 528 + 128)];
    opus::vector_t<opus::fp8_t, 32> q[4];
    int q_scales[4];
    opus::vector_t<float, 8> o[16];
    opus::static_for<16>([&](auto chunk) { opus::clear(o[int(chunk)]); });
    if constexpr(PreparedQ)
    {
        opus::static_for<4>([&](auto step) {
            const int d  = int(step) * 128 + group * 32;
            q[int(step)] = *reinterpret_cast<const opus::vector_t<opus::fp8_t, 32>*>(
                reinterpret_cast<const uint8_t*>(a.q) + (int64_t(t) * a.heads + hload) * 512 + d);
            q[int(step)]        = pack_fp8_qk_group(q[int(step)], lane);
            q_scales[int(step)] = a.q_scale[(int64_t(t) * a.heads + hload) * 16 + d / 32];
        });
    }
    else
    {
        // One lane owns all 32 elements of a quantization group: no reduction barrier.
        // BF16 rounding after forward RoPE matches the existing attention boundary.
        opus::static_for<4>([&](auto step) {
            opus::vector_t<float, 32> values;
            const int d = int(step) * 128 + group * 32;
            opus::static_for<4>([&](auto sub) {
                B8 x = *reinterpret_cast<const B8*>(a.q + (int64_t(t) * a.heads + hload) * 512 + d +
                                                    int(sub) * 8);
                if constexpr(RotateQ && int(step) == 3)
                {
                    if(group >= 2)
                        opus::static_for<4>([&](auto pair) {
                            int i = int(pair) * 2, ri = (d + int(sub) * 8 + i - 448) / 2;
                            float x0 = f32(x[i]), x1 = f32(x[i + 1]), c = f32(cs[ri]),
                                  snv = f32(sn[ri]);
                            x[i]      = static_cast<B>(fmaf(x0, c, -x1 * snv));
                            x[i + 1]  = static_cast<B>(fmaf(x0, snv, x1 * c));
                        });
                }
                opus::static_for<8>(
                    [&](auto j) { values[int(sub) * 8 + int(j)] = f32(x[int(j)]); });
            });
            float amax = 1e-30f;
            opus::static_for<32>([&](auto j) { amax = fmaxf(amax, fabsf(values[int(j)])); });
            const unsigned bits = __float_as_uint(amax / 448.0f);
            const int exponent  = int((bits - 1) >> 23) + 1 - 127;
            const float mul     = __uint_as_float(uint32_t(127 - exponent) << 23);
            q_scales[int(step)] = exponent + 127;
            q[int(step)]        = pack_fp8_qk_group(opus::cast<opus::fp8_t>(values * mul), lane);
        });
    }
    constexpr float log2e = 1.4426950408889634f;
    float m = -__FLT_MAX__, actual_max = -INFINITY, l = 0;
    int begin = max(0, min(a.prefix_capacity, a.prefix_indptr[t]));
    int end   = max(begin, min(a.prefix_capacity, a.prefix_indptr[t + 1]));
    if(end > begin)
    {
        Fp8QkSource64<PageSize> src{{a.pool,
                                     a.prefix_at<PageSize>(begin),
                                     end - begin,
                                     (PageSize == -2 ? a.cache.num_rows : a.pool_bytes),
                                     a.cache}};
        fp8_qk_decode_accum64<T, LateCommit>(src,
                                             end - begin,
                                             (end - begin + 31) / 32,
                                             reinterpret_cast<char*>(kv_tile),
                                             q,
                                             q_scales,
                                             raw_k,
                                             o,
                                             actual_max,
                                             m,
                                             l,
                                             a.softmax_scale * log2e);
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

template <bool RotateQ, bool LateLoad, int PageSize>
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
        PackedDecodeSource128<(PageSize != -2), LateLoad, PageSize> src{
            a.pool,
            a.prefix_at<PageSize>(begin),
            end - begin,
            (PageSize == -2 ? a.cache.num_rows : a.pool_bytes),
            a.cache};
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

template <int PageSize>
hipError_t launch_cache_mode(const FusedDsv41Args* a, int rotate_q, hipStream_t stream)
{
    if(!a->tokens)
        return hipSuccess;
    if constexpr(PageSize != -2)
    {
        if(a->fp8_qk)
        {
            dim3 grid(a->tokens, a->heads / 64);
            // H64 benefits from overlapping V dequantization with the next raw
            // tile fetch. H128 retains the earlier commit to avoid its regression.
            if(a->heads == 64)
            {
                if(a->q_scale)
                    fused_dsv41_fp8_qk_decode_64<false, PageSize, true, true>
                        <<<grid, 256, 0, stream>>>(*a);
                else if(rotate_q)
                    fused_dsv41_fp8_qk_decode_64<true, PageSize, false, true>
                        <<<grid, 256, 0, stream>>>(*a);
                else
                    fused_dsv41_fp8_qk_decode_64<false, PageSize, false, true>
                        <<<grid, 256, 0, stream>>>(*a);
            }
            else
            {
                if(a->q_scale)
                    fused_dsv41_fp8_qk_decode_64<false, PageSize, true>
                        <<<grid, 256, 0, stream>>>(*a);
                else if(rotate_q)
                    fused_dsv41_fp8_qk_decode_64<true, PageSize><<<grid, 256, 0, stream>>>(*a);
                else
                    fused_dsv41_fp8_qk_decode_64<false, PageSize><<<grid, 256, 0, stream>>>(*a);
            }
            return hipGetLastError();
        }
    }
    // Pure-cache attention uses the specialized pipeline for decode and prefill.
    // FP8 prefetches raw bytes before QK; BF16 loads directly to LDS.
    // Commit/load is after QK (H64) or PV (H128). Mixed sources keep v4.
    if(!a->extend_indptr && (a->heads == 64 || a->heads == 128))
    {
        dim3 grid(a->tokens, PageSize == -2 ? 1 : a->heads / 64);
        if(a->heads == 64 || (a->heads == 128 && PageSize != -2))
        {
            if(rotate_q)
                fused_dsv41_mxfp8_decode_64<true, PageSize><<<grid, 256, 0, stream>>>(*a);
            else
                fused_dsv41_mxfp8_decode_64<false, PageSize><<<grid, 256, 0, stream>>>(*a);
        }
        else
        {
            if(rotate_q)
                fused_dsv41_mxfp8_decode_128<true, false, PageSize><<<grid, 512, 0, stream>>>(*a);
            else
                fused_dsv41_mxfp8_decode_128<false, false, PageSize><<<grid, 512, 0, stream>>>(*a);
        }
        return hipGetLastError();
    }
    if(a->heads <= 16)
    {
        dim3 grid(a->tokens);
        if(rotate_q)
            fused_dsv41_mxfp8_kernel_small<true, 4, PageSize><<<grid, 256, 0, stream>>>(*a);
        else
            fused_dsv41_mxfp8_kernel_small<false, 4, PageSize><<<grid, 256, 0, stream>>>(*a);
    }
    else if(a->heads == 32)
    {
        dim3 grid(a->tokens);
        if(rotate_q)
            fused_dsv41_mxfp8_kernel_small<true, 2, PageSize><<<grid, 256, 0, stream>>>(*a);
        else
            fused_dsv41_mxfp8_kernel_small<false, 2, PageSize><<<grid, 256, 0, stream>>>(*a);
    }
    else if(a->heads == 64)
    {
        dim3 grid(a->tokens);
        if(rotate_q)
            fused_dsv41_mxfp8_kernel_64<true, PageSize><<<grid, 256, 0, stream>>>(*a);
        else
            fused_dsv41_mxfp8_kernel_64<false, PageSize><<<grid, 256, 0, stream>>>(*a);
    }
    else
    {
        dim3 grid(a->tokens);
        // H128 early loading wins at T<=1024; the late loader wins at
        // T2048 and T4096. Keep the unmeasured interval on the original path.
        if(a->tokens >= 2048)
        {
            if(rotate_q)
                fused_dsv41_mxfp8_kernel_128<true, true, PageSize><<<grid, 512, 0, stream>>>(*a);
            else
                fused_dsv41_mxfp8_kernel_128<false, true, PageSize><<<grid, 512, 0, stream>>>(*a);
        }
        else
        {
            if(rotate_q)
                fused_dsv41_mxfp8_kernel_128<true, false, PageSize><<<grid, 512, 0, stream>>>(*a);
            else
                fused_dsv41_mxfp8_kernel_128<false, false, PageSize><<<grid, 512, 0, stream>>>(*a);
        }
    }
    return hipGetLastError();
}

hipError_t launch_fused_dsv41_mxfp8(const FusedDsv41Args* a, int rotate_q, hipStream_t stream)
{
    switch(a->cache.page_size)
    {
    case -2: return launch_cache_mode<-2>(a, rotate_q, stream); // Contiguous BF16 rows.
    case 0: return launch_cache_mode<0>(a, rotate_q, stream);
    case 32: return launch_cache_mode<32>(a, rotate_q, stream);
    case 128: return launch_cache_mode<128>(a, rotate_q, stream);
    default: return launch_cache_mode<-1>(a, rotate_q, stream);
    }
}
