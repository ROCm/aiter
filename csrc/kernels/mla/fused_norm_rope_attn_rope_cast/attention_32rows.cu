// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// HIP device kernels and launches for 32-row/shared-KV attention and legacy/BF16 paths.
// Public ctypes entrypoints and tensor validation are in fused_norm_rope_attn_rope_cast.cu.
#include "attention_32rows.cuh"
#include "dispatch_policy.hpp"
#include "params.hpp"
#include "q_norm.cuh"
#include <cmath>
#include <hip/hip_runtime.h>

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
template <bool RotateQ, int KvLayout>
__global__ __launch_bounds__(256, 2) void fused_norm_rope_attn_rope_cast_kernel_64(
    FusedNormRopeAttnRopeCastArgs a)
{
#if defined(__gfx950__)
    using T = FusedMla32RowsTraits<16, 32, 512, 4>;
    using namespace fused_mla_32rows;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, group = lane / 16;
    const int hl = wave * 16 + lane % 16, h = blockIdx.y * 64 + hl, t = blockIdx.x;
    const int hload   = min(h, a.heads - 1);
    const float q_inv = fused_q_norm::inv_rms<4, 16>(
        a.q + (int64_t(t) * a.heads + hload) * 512, group, a.q_norm, a.norm_eps);
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
        if(a.q_norm)
            fused_q_norm::apply(x, q_inv);
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
        PackedCacheSource64<(KvLayout != fused_kv_layout::contiguous_bf16_rows), KvLayout> src{
            a.pool,
            a.prefix_at<KvLayout>(begin),
            end - begin,
            (KvLayout == fused_kv_layout::contiguous_bf16_rows ? a.cache.num_rows : a.pool_bytes),
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

template <bool RotateQ, bool LateLoad, int KvLayout>
__global__ __launch_bounds__(512, 1) void fused_norm_rope_attn_rope_cast_kernel_128(
    FusedNormRopeAttnRopeCastArgs a)
{
#if defined(__gfx950__)
    using T = FusedMla32RowsTraits<16, 32, 512, 8>;
    using namespace fused_mla_32rows;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, group = lane / 16;
    const int hl = wave * 16 + lane % 16, h = blockIdx.y * 128 + hl, t = blockIdx.x;
    const int hload   = min(h, a.heads - 1);
    const float q_inv = fused_q_norm::inv_rms<4, 16>(
        a.q + (int64_t(t) * a.heads + hload) * 512, group, a.q_norm, a.norm_eps);
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
        if(a.q_norm)
            fused_q_norm::apply(x, q_inv);
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
        PackedCacheSource128<(KvLayout != fused_kv_layout::contiguous_bf16_rows),
                             LateLoad,
                             KvLayout>
            src{a.pool,
                a.prefix_at<KvLayout>(begin),
                end - begin,
                (KvLayout == fused_kv_layout::contiguous_bf16_rows ? a.cache.num_rows
                                                                   : a.pool_bytes),
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
template <bool RotateQ, int Parts, int KvLayout>
__global__ __launch_bounds__(256, 2) void fused_norm_rope_attn_rope_cast_kernel_small(
    FusedNormRopeAttnRopeCastArgs a)
{
#if defined(__gfx950__)
    using T = FusedMla32RowsTraits<16, 32, 512, 4>;
    using namespace fused_mla_32rows;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, group = lane / 16;
    constexpr int BH = 64 / Parts, OSlices = 16 / Parts;
    const int part = wave % Parts;
    const int hl = (wave / Parts) * 16 + lane % 16, h = blockIdx.y * BH + hl, t = blockIdx.x;
    const int hload   = min(h, a.heads - 1);
    const float q_inv = fused_q_norm::inv_rms<4, 16>(
        a.q + (int64_t(t) * a.heads + hload) * 512, group, a.q_norm, a.norm_eps);
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
        if(a.q_norm)
            fused_q_norm::apply(x, q_inv);
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
        PackedCacheSourceSmall<(KvLayout != fused_kv_layout::contiguous_bf16_rows), KvLayout> src{
            a.pool,
            a.prefix_at<KvLayout>(begin),
            end - begin,
            (KvLayout == fused_kv_layout::contiguous_bf16_rows ? a.cache.num_rows : a.pool_bytes),
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

template <bool RotateQ, int KvLayout>
__global__ __launch_bounds__(256, 2) void fused_norm_rope_attn_rope_cast_decode_64(
    FusedNormRopeAttnRopeCastArgs a)
{
#if defined(__gfx950__)
    using T = FusedMla32RowsTraits<16, 32, 512, 4>;
    using namespace fused_mla_32rows;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, group = lane / 16;
    const int hl = wave * 16 + lane % 16, h = blockIdx.y * 64 + hl, t = blockIdx.x;
    const int hload   = min(h, a.heads - 1);
    const float q_inv = fused_q_norm::inv_rms<4, 16>(
        a.q + (int64_t(t) * a.heads + hload) * 512, group, a.q_norm, a.norm_eps);
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
        if(a.q_norm)
            fused_q_norm::apply(x, q_inv);
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
        PackedDecodeSource64<(KvLayout != fused_kv_layout::contiguous_bf16_rows), KvLayout> src{
            a.pool,
            a.prefix_at<KvLayout>(begin),
            end - begin,
            (KvLayout == fused_kv_layout::contiguous_bf16_rows ? a.cache.num_rows : a.pool_bytes),
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

template <bool RotateQ,
          int KvLayout,
          bool PreparedQ   = false,
          bool H64Schedule = false,
          bool QNorm       = false,
          bool TwoSources  = false,
          int HeadTile     = 64>
__global__ __launch_bounds__(HeadTile * 4, 1) void fused_norm_rope_attn_rope_cast_fp8_qk_decode_64(
    FusedNormRopeAttnRopeCastArgs a)
{
#if defined(__gfx950__)
    using T = FusedMla32RowsTraits<16, 32, 512, 4>;
    using namespace fused_mla_32rows;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, group = lane / 16;
    const int hl = wave * 16 + lane % 16, h = blockIdx.y * HeadTile + hl, t = blockIdx.x;
    const int hload = min(h, a.heads - 1);
    static_assert(!QNorm || (RotateQ && !PreparedQ));
    float q_inv = 1.0f;
    if constexpr(QNorm)
        q_inv = fused_q_norm::inv_rms<4, 16>(
            a.q + (int64_t(t) * a.heads + hload) * 512, group, true, a.norm_eps);
    static_assert(HeadTile == 64 || HeadTile == 128);
    // Paired waves share V across head groups for raw and prepared Q.
    constexpr int PVReuse       = HeadTile == 128 ? 2 : 1;
    constexpr int RawPitch      = PVReuse > 1 ? 544 : 528;
    constexpr int PVSlices      = 16 / PVReuse;
    constexpr int STAGING_ELEMS = HeadTile * 528 / sizeof(B);
    constexpr int LDS_ELEMS =
        T::smem_kv_tile_elems > STAGING_ELEMS ? T::smem_kv_tile_elems : STAGING_ELEMS;
    static_assert(PVReuse == 1 || LDS_ELEMS >= T::smem_kv_tile_elems + 8 * 64 * 8 + 128 * 2);
    __shared__ __align__(16) B kv_tile[LDS_ELEMS];
    const int64_t pos = a.positions64 ? static_cast<const int64_t*>(a.positions)[t]
                                      : static_cast<const int*>(a.positions)[t];
    const B* cs       = a.cos + pos * 32;
    const B* sn       = a.sin + pos * 32;
    __shared__ __align__(32) uint8_t raw_k[2 * (32 * RawPitch + 128)];
    opus::vector_t<opus::fp8_t, 32> q[4];
    int q_scales[4];
    opus::vector_t<float, 8> o[16];
    opus::static_for<16>([&](auto chunk) { opus::clear(o[int(chunk)]); });
    // Each lane reads the two 16-byte pieces consumed by scaled MFMA.
    // Only two scalar group maxima and one scale code cross lanes per step.
    opus::static_for<4>([&](auto step) {
        if constexpr(PreparedQ)
        {
            opus::static_for<2>([&](auto part) {
                const int d =
                    int(step) * 128 + (group / 2) * 32 + (group % 2) * 16 + int(part) * 64;
                const auto values = *reinterpret_cast<const opus::vector_t<opus::fp8_t, 16>*>(
                    reinterpret_cast<const uint8_t*>(a.q) + (int64_t(t) * a.heads + hload) * 512 +
                    d);
                opus::set_slice(q[int(step)],
                                values,
                                opus::number<int(part) * 16>{},
                                opus::number<int(part) * 16 + 16>{});
            });
            q_scales[int(step)] =
                a.q_scale[(int64_t(t) * a.heads + hload) * 16 + int(step) * 4 + group];
        }
        else
        {
            int codes[2];
            opus::static_for<2>([&](auto part) {
                const int d =
                    int(step) * 128 + (group / 2) * 32 + (group % 2) * 16 + int(part) * 64;
                opus::vector_t<float, 16> values;
                opus::static_for<2>([&](auto sub) {
                    B8 x = *reinterpret_cast<const B8*>(a.q + (int64_t(t) * a.heads + hload) * 512 +
                                                        d + int(sub) * 8);
                    if constexpr(QNorm)
                        fused_q_norm::apply(x, q_inv);
                    if constexpr(RotateQ && int(step) == 3 && int(part) == 1)
                        opus::static_for<4>([&](auto pair) {
                            const int i    = int(pair) * 2;
                            const int ri   = (d + int(sub) * 8 + i - 448) / 2;
                            const float x0 = f32(x[i]), x1 = f32(x[i + 1]), cv = f32(cs[ri]),
                                        sv = f32(sn[ri]);
                            x[i]           = static_cast<B>(fmaf(x0, cv, -x1 * sv));
                            x[i + 1]       = static_cast<B>(fmaf(x0, sv, x1 * cv));
                        });
                    opus::static_for<8>(
                        [&](auto j) { values[int(sub) * 8 + int(j)] = f32(x[int(j)]); });
                });
                float amax = 1e-30f;
                opus::static_for<16>([&](auto j) { amax = fmaxf(amax, fabsf(values[int(j)])); });
                const uint32_t other =
                    __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, __float_as_uint(amax));
                amax                = fmaxf(amax, __uint_as_float(other));
                const unsigned bits = __float_as_uint(amax / 448.0f);
                const int exponent  = int((bits - 1) >> 23) + 1 - 127;
                const float mul     = __uint_as_float(uint32_t(127 - exponent) << 23);
                codes[int(part)]    = exponent + 127;
                const auto quant    = opus::cast<opus::fp8_t>(values * mul);
                opus::set_slice(q[int(step)],
                                quant,
                                opus::number<int(part) * 16>{},
                                opus::number<int(part) * 16 + 16>{});
            });
            const uint32_t code = (group & 1) ? codes[1] : codes[0];
            const int src       = lane % 16 + ((group % 2) * 2 + group / 2) * 16;
            q_scales[int(step)] = __builtin_amdgcn_ds_bpermute(src * 4, code);
        }
    });
    constexpr float log2e = 1.4426950408889634f;
    float m = -__FLT_MAX__, actual_max = -INFINITY, l = 0;
    if constexpr(KvLayout == fused_kv_layout::strided_fp8_rows && TwoSources && PVReuse == 1)
    {
        const int begin  = max(0, min(a.prefix_capacity, a.prefix_indptr[t]));
        const int end    = max(begin, min(a.prefix_capacity, a.prefix_indptr[t + 1]));
        int extend_begin = 0, extend_end = 0;
        if(a.extend_indptr)
        {
            extend_begin = max(0, min(a.extend_capacity, a.extend_indptr[t]));
            extend_end   = max(extend_begin, min(a.extend_capacity, a.extend_indptr[t + 1]));
        }
        const int prefix_tiles = (end - begin + 31) / 32;
        const int tiles        = prefix_tiles + (extend_end - extend_begin + 31) / 32;
        if(tiles)
        {
            Fp8QkSource64<fused_kv_layout::strided_fp8_rows> prefix{
                {a.pool,
                 a.prefix ? a.prefix_at<fused_kv_layout::strided_fp8_rows>(begin) : nullptr,
                 end - begin,
                 a.cache.num_rows,
                 a.cache}};
            Fp8QkSource64<fused_kv_layout::strided_fp8_rows> extend{
                {reinterpret_cast<const uint8_t*>(a.extend_kv),
                 a.extend ? a.extend + extend_begin : nullptr,
                 extend_end - extend_begin,
                 a.extend_rows,
                 a.extend_cache}};
            Fp8QkTwoSource64 src{{}, prefix, extend, prefix_tiles};
            fp8_qk_decode_accum64<T, H64Schedule, HeadTile / 64, PVReuse>(
                src,
                end - begin + extend_end - extend_begin,
                tiles,
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
    }
    else
    {
        int begin = max(0, min(a.prefix_capacity, a.prefix_indptr[t]));
        int end   = max(begin, min(a.prefix_capacity, a.prefix_indptr[t + 1]));
        if(end > begin)
        {
            Fp8QkSource64<KvLayout, RawPitch> src{
                {a.pool,
                 a.prefix_at<KvLayout>(begin),
                 end - begin,
                 (KvLayout == fused_kv_layout::contiguous_bf16_rows ? a.cache.num_rows
                                                                    : a.pool_bytes),
                 a.cache}};
            fp8_qk_decode_accum64<T, H64Schedule, HeadTile / 64, PVReuse>(
                src,
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
        // Both sources contribute to one online softmax; normalize only after the
        // second.
        if constexpr(KvLayout == fused_kv_layout::strided_fp8_rows)
        {
            if(a.extend_indptr)
            {
                begin = max(0, min(a.extend_capacity, a.extend_indptr[t]));
                end   = max(begin, min(a.extend_capacity, a.extend_indptr[t + 1]));
                if(end > begin)
                {
                    Fp8QkSource64<fused_kv_layout::strided_fp8_rows, RawPitch> src{
                        {reinterpret_cast<const uint8_t*>(a.extend_kv),
                         a.extend + begin,
                         end - begin,
                         a.extend_rows,
                         a.extend_cache}};
                    fp8_qk_decode_accum64<T, H64Schedule, HeadTile / 64, PVReuse>(
                        src,
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
            }
        }
    }
    if constexpr(PVReuse == 1)
    {
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
                if(a.lse)
                    a.lse[int64_t(t) * a.heads + h] = l > 0 ? m / log2e + logf(l) : INFINITY;
                if(a.max_logits)
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
                    opus::static_for<4>(
                        [&](auto i) { amax = fmaxf(amax, fabsf(x[int(n)][int(i)])); });
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
    }
    else
    {
        // Raw K is dead after the final PV barrier. Use it for normalization
        // factors while kv_tile becomes the packed-output staging buffer.
        auto* inv_lds       = reinterpret_cast<float*>(raw_k);
        const float sink    = a.sink ? a.sink[h] * log2e : -INFINITY;
        const float final_m = fmaxf(m, sink);
        const float rescale = __builtin_amdgcn_exp2f(m - final_m);
        const float denom   = l * rescale + __builtin_amdgcn_exp2f(sink - final_m);
        if(group == 0)
        {
            inv_lds[hl] = denom > 0 ? rescale / denom : 0;
            if(a.lse)
                a.lse[int64_t(t) * a.heads + h] = l > 0 ? m / log2e + logf(l) : INFINITY;
            if(a.max_logits)
                a.max_logits[int64_t(t) * a.heads + h] = actual_max / log2e;
        }
        __syncthreads();
        auto* staging = reinterpret_cast<uint8_t*>(kv_tile);
        opus::static_for<PVReuse>([&](auto hg) {
            const int hl    = ((wave / PVReuse) * PVReuse + int(hg)) * 16 + lane % 16;
            const int h     = blockIdx.y * HeadTile + hl;
            const float inv = inv_lds[hl];
            opus::static_for<PVSlices>([&](auto ds) {
                constexpr int index = int(hg) * PVSlices + int(ds);
                const int chunk     = (wave % PVReuse) * PVSlices + int(ds);
                F4 x[2]             = {opus::slice(o[index], opus::number<0>{}, opus::number<4>{}),
                                       opus::slice(o[index], opus::number<4>{}, opus::number<8>{})};
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
                    opus::static_for<4>(
                        [&](auto i) { amax = fmaxf(amax, fabsf(x[int(n)][int(i)])); });
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
        });
    }
    auto* staging = reinterpret_cast<uint8_t*>(kv_tile);
    __syncthreads();
    using U4 = opus::vector_t<uint32_t, 4>;
    for(int i = tid; i < min(HeadTile, a.heads - int(blockIdx.y) * HeadTile) * 32;
        i += HeadTile * 4)
        *reinterpret_cast<U4*>(a.out + (int64_t(t) * a.heads + blockIdx.y * HeadTile) * 512 +
                               i * 16) =
            *reinterpret_cast<const U4*>(staging + (i / 32) * 528 + (i % 32) * 16);
#endif
}

template <bool RotateQ, bool LateLoad, int KvLayout>
__global__ __launch_bounds__(512, 1) void fused_norm_rope_attn_rope_cast_decode_128(
    FusedNormRopeAttnRopeCastArgs a)
{
#if defined(__gfx950__)
    using T = FusedMla32RowsTraits<16, 32, 512, 8>;
    using namespace fused_mla_32rows;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, group = lane / 16;
    const int hl = wave * 16 + lane % 16, h = blockIdx.y * 128 + hl, t = blockIdx.x;
    const int hload   = min(h, a.heads - 1);
    const float q_inv = fused_q_norm::inv_rms<4, 16>(
        a.q + (int64_t(t) * a.heads + hload) * 512, group, a.q_norm, a.norm_eps);
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
        if(a.q_norm)
            fused_q_norm::apply(x, q_inv);
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
        PackedDecodeSource128<(KvLayout != fused_kv_layout::contiguous_bf16_rows),
                              LateLoad,
                              KvLayout>
            src{a.pool,
                a.prefix_at<KvLayout>(begin),
                end - begin,
                (KvLayout == fused_kv_layout::contiguous_bf16_rows ? a.cache.num_rows
                                                                   : a.pool_bytes),
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

// Launch fused attention after the caller has selected the KV layout.
// KvLayout is a fused_kv_layout storage tag.
// HasPrefixAndExtend refers to KV sources; apply_q_rope refers to forward Q RoPE.
template <int KvLayout, bool HasPrefixAndExtend = false>
hipError_t launch_attention_for_kv_layout(const FusedNormRopeAttnRopeCastArgs* args,
                                          int apply_q_rope,
                                          hipStream_t stream)
{
    namespace policy = fused_norm_rope_attn_rope_cast_policy;
    if(!args->tokens)
        return hipSuccess;
    if constexpr(KvLayout != fused_kv_layout::contiguous_bf16_rows)
    {
        if(args->fp8_qk)
        {
            if constexpr(KvLayout == fused_kv_layout::strided_fp8_rows)
            {
                const int64_t capacity = int64_t(args->prefix_capacity) + args->extend_capacity;
                if(args->heads == 128 && !args->q_norm && (args->q_scale || apply_q_rope) &&
                   policy::prefer_h128_shared_kv(args->tokens, capacity))
                {
                    const dim3 head_grid(args->tokens);
                    if(args->q_scale)
                        fused_norm_rope_attn_rope_cast_fp8_qk_decode_64<
                            false,
                            fused_kv_layout::strided_fp8_rows,
                            true,
                            false,
                            false,
                            HasPrefixAndExtend,
                            128><<<head_grid, 512, 0, stream>>>(*args);
                    else
                        fused_norm_rope_attn_rope_cast_fp8_qk_decode_64<
                            true,
                            fused_kv_layout::strided_fp8_rows,
                            false,
                            false,
                            false,
                            HasPrefixAndExtend,
                            128><<<head_grid, 512, 0, stream>>>(*args);
                    return hipGetLastError();
                }
            }
            dim3 grid(args->tokens, args->heads / 64);
            // H64 benefits from overlapping V dequantization with the next raw
            // tile fetch. H128 retains the earlier commit to avoid its regression.
            if(args->heads == 64)
            {
                if(args->q_scale)
                    fused_norm_rope_attn_rope_cast_fp8_qk_decode_64<false,
                                                                    KvLayout,
                                                                    true,
                                                                    true,
                                                                    false,
                                                                    HasPrefixAndExtend>
                        <<<grid, 256, 0, stream>>>(*args);
                else if(args->q_norm)
                    fused_norm_rope_attn_rope_cast_fp8_qk_decode_64<true,
                                                                    KvLayout,
                                                                    false,
                                                                    true,
                                                                    true,
                                                                    HasPrefixAndExtend>
                        <<<grid, 256, 0, stream>>>(*args);
                else if(apply_q_rope)
                    fused_norm_rope_attn_rope_cast_fp8_qk_decode_64<true,
                                                                    KvLayout,
                                                                    false,
                                                                    true,
                                                                    false,
                                                                    HasPrefixAndExtend>
                        <<<grid, 256, 0, stream>>>(*args);
                else
                    fused_norm_rope_attn_rope_cast_fp8_qk_decode_64<false,
                                                                    KvLayout,
                                                                    false,
                                                                    true,
                                                                    false,
                                                                    HasPrefixAndExtend>
                        <<<grid, 256, 0, stream>>>(*args);
            }
            else
            {
                if(args->q_scale)
                    fused_norm_rope_attn_rope_cast_fp8_qk_decode_64<false,
                                                                    KvLayout,
                                                                    true,
                                                                    false,
                                                                    false,
                                                                    HasPrefixAndExtend>
                        <<<grid, 256, 0, stream>>>(*args);
                else if(args->q_norm)
                    fused_norm_rope_attn_rope_cast_fp8_qk_decode_64<true,
                                                                    KvLayout,
                                                                    false,
                                                                    false,
                                                                    true,
                                                                    HasPrefixAndExtend>
                        <<<grid, 256, 0, stream>>>(*args);
                else if(apply_q_rope)
                    fused_norm_rope_attn_rope_cast_fp8_qk_decode_64<true,
                                                                    KvLayout,
                                                                    false,
                                                                    false,
                                                                    false,
                                                                    HasPrefixAndExtend>
                        <<<grid, 256, 0, stream>>>(*args);
                else
                    fused_norm_rope_attn_rope_cast_fp8_qk_decode_64<false,
                                                                    KvLayout,
                                                                    false,
                                                                    false,
                                                                    false,
                                                                    HasPrefixAndExtend>
                        <<<grid, 256, 0, stream>>>(*args);
            }
            return hipGetLastError();
        }
    }
    // Pure-cache attention uses the specialized pipeline for decode and prefill.
    // FP8 prefetches raw bytes before QK; BF16 loads directly to LDS.
    // Commit/load is after QK (H64) or PV (H128). Mixed sources use the prefill pipeline.
    if(!args->extend_indptr && (args->heads == 64 || args->heads == 128))
    {
        dim3 grid(args->tokens,
                  KvLayout == fused_kv_layout::contiguous_bf16_rows ? 1 : args->heads / 64);
        if(args->heads == 64 ||
           (args->heads == 128 && KvLayout != fused_kv_layout::contiguous_bf16_rows))
        {
            if(apply_q_rope)
                fused_norm_rope_attn_rope_cast_decode_64<true, KvLayout>
                    <<<grid, 256, 0, stream>>>(*args);
            else
                fused_norm_rope_attn_rope_cast_decode_64<false, KvLayout>
                    <<<grid, 256, 0, stream>>>(*args);
        }
        else
        {
            if(apply_q_rope)
                fused_norm_rope_attn_rope_cast_decode_128<true, false, KvLayout>
                    <<<grid, 512, 0, stream>>>(*args);
            else
                fused_norm_rope_attn_rope_cast_decode_128<false, false, KvLayout>
                    <<<grid, 512, 0, stream>>>(*args);
        }
        return hipGetLastError();
    }
    if(args->heads <= 16)
    {
        dim3 grid(args->tokens);
        if(apply_q_rope)
            fused_norm_rope_attn_rope_cast_kernel_small<true, 4, KvLayout>
                <<<grid, 256, 0, stream>>>(*args);
        else
            fused_norm_rope_attn_rope_cast_kernel_small<false, 4, KvLayout>
                <<<grid, 256, 0, stream>>>(*args);
    }
    else if(args->heads == 32)
    {
        dim3 grid(args->tokens);
        if(apply_q_rope)
            fused_norm_rope_attn_rope_cast_kernel_small<true, 2, KvLayout>
                <<<grid, 256, 0, stream>>>(*args);
        else
            fused_norm_rope_attn_rope_cast_kernel_small<false, 2, KvLayout>
                <<<grid, 256, 0, stream>>>(*args);
    }
    else if(args->heads == 64)
    {
        dim3 grid(args->tokens);
        if(apply_q_rope)
            fused_norm_rope_attn_rope_cast_kernel_64<true, KvLayout>
                <<<grid, 256, 0, stream>>>(*args);
        else
            fused_norm_rope_attn_rope_cast_kernel_64<false, KvLayout>
                <<<grid, 256, 0, stream>>>(*args);
    }
    else
    {
        dim3 grid(args->tokens);
        if(policy::prefer_h128_late_bf16_load(args->tokens))
        {
            if(apply_q_rope)
                fused_norm_rope_attn_rope_cast_kernel_128<true, true, KvLayout>
                    <<<grid, 512, 0, stream>>>(*args);
            else
                fused_norm_rope_attn_rope_cast_kernel_128<false, true, KvLayout>
                    <<<grid, 512, 0, stream>>>(*args);
        }
        else
        {
            if(apply_q_rope)
                fused_norm_rope_attn_rope_cast_kernel_128<true, false, KvLayout>
                    <<<grid, 512, 0, stream>>>(*args);
            else
                fused_norm_rope_attn_rope_cast_kernel_128<false, false, KvLayout>
                    <<<grid, 512, 0, stream>>>(*args);
        }
    }
    return hipGetLastError();
}

hipError_t launch_fused_norm_rope_attn_rope_cast(const FusedNormRopeAttnRopeCastArgs* args,
                                                 int apply_q_rope,
                                                 hipStream_t stream)
{
    switch(args->cache.layout)
    {
    case fused_kv_layout::strided_fp8_rows:
        // Capacities are host metadata; single-source requests keep their original
        // kernel.
        if(args->prefix_capacity > 0 && args->extend_capacity > 0)
            return launch_attention_for_kv_layout<fused_kv_layout::strided_fp8_rows, true>(
                args, apply_q_rope, stream);
        return launch_attention_for_kv_layout<fused_kv_layout::strided_fp8_rows>(
            args, apply_q_rope, stream);
    case fused_kv_layout::contiguous_bf16_rows:
        return launch_attention_for_kv_layout<fused_kv_layout::contiguous_bf16_rows>(
            args, apply_q_rope, stream); // Contiguous BF16 rows.
    case fused_kv_layout::legacy_tagged_fp8:
        return launch_attention_for_kv_layout<fused_kv_layout::legacy_tagged_fp8>(
            args, apply_q_rope, stream);
    default: return hipErrorInvalidValue;
    }
}
