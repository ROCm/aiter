// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "hip_reduce.h"
#include "opus.hpp"
#include "vec_convert.h"

namespace aiter {
using namespace opus;
#define RT 0
#define GROUP_NT 3

using index_t = int;

// Load a large vector (vec_size elements of type T) from gmem buffer in chunks.
// Each chunk issues one buffer_load instruction of chunk_bytes bytes (4/8/16 ->
// dword/dwordx2/dwordx4). Total loads = vec_size * sizeof(T) / chunk_bytes.
//
// interleave=false: chunks are contiguous in GMEM.
//   GMEM layout (per thread):
//     base + row_offset
//     |<-- chunk_bytes -->|<-- chunk_bytes -->|<-- chunk_bytes -->|<-- chunk_bytes -->|
//     [     chunk 0      ][     chunk 1      ][     chunk 2      ][     chunk 3      ]
//
// interleave=true: chunks are strided by interleave_thread_size * chunk_bytes in GMEM.
//   GMEM layout (thread 0 loads marked with *, other threads fill the gaps):
//     base + row_offset
//     |<- chunk_bytes ->|<- (interleave_thread_size-1)*chunk_bytes gap ->|<- chunk_bytes ->|...
//     [ *chunk 0 (t0)* ][ chunk 0 (t1) ]...[ chunk 0 (tN-1) ]         [ *chunk 1 (t0)* ]...
//
//   Each thread's chunks are interleaved with other threads' data,
//   stride = interleave_thread_size * chunk_bytes bytes between chunks.
//
// Example: T=bf16(2B), vec_size=32, chunk_bytes=16, interleave_thread_size=256
//   total = 64B -> 4x buffer_load_dwordx4, each loading 8 bf16 elements.
//   interleave stride = 256 * 16 = 4096 bytes between chunks.
template <typename T,
          int vec_size,
          int chunk_bytes,
          int aux                    = 0,
          bool interleave            = false,
          int interleave_thread_size = WARP_SIZE>
__device__ opus::vector_t<T, vec_size> load_vector_nbytes(opus::gmem<T>& buffer, int row_offset)
{
    static_assert(vec_size * sizeof(T) % chunk_bytes == 0,
                  "vec_size * sizeof(T) must be a multiple of chunk_bytes");
    static constexpr index_t num_chunks   = vec_size * sizeof(T) / chunk_bytes;
    constexpr index_t chunk_size_elements = chunk_bytes / sizeof(T);
    constexpr index_t interleave_bytes    = interleave_thread_size * chunk_bytes;

    opus::vector_t<T, vec_size> result;
    T* result_ptr = reinterpret_cast<T*>(&result);

    opus::static_for<num_chunks>([&](auto i) {
        constexpr index_t chunk_offset_bytes =
            interleave ? i.value * interleave_bytes : i.value * chunk_bytes;
        constexpr index_t chunk_offset_elements = chunk_offset_bytes / sizeof(T);

        opus::vector_t<T, chunk_size_elements>* chunk_ptr =
            reinterpret_cast<opus::vector_t<T, chunk_size_elements>*>(
                result_ptr + i.value * chunk_size_elements);
        *chunk_ptr =
            buffer.template load<chunk_size_elements, aux>(row_offset, chunk_offset_elements);
    });

    return result;
}

// Store a vector (vec_size elements of DTYPE_I) to gmem buffer in chunks, with optional type
// conversion. Mirror of load_vector_nbytes but for writing. Each chunk issues one buffer_store of
// chunk_bytes bytes.
//
// Template params:
//   T          : buffer element type (storage type in GMEM)
//   DTYPE_I    : input element type in registers (e.g. float)
//   vec_size   : number of input elements
//   chunk_bytes: bytes per buffer_store instruction (4/8/16 -> dword/dwordx2/dwordx4)
//   T_R        : target conversion type before storing (default = T)
//               if T_R != DTYPE_I, data is converted per-chunk before store.
//   interleave : same strided layout as load_vector_nbytes
//                (stride = interleave_thread_size * chunk_bytes)
//
// interleave=false: chunks are contiguous in GMEM.
//   GMEM layout (per thread):
//     base + row_offset
//     |<-- chunk_bytes -->|<-- chunk_bytes -->|<-- chunk_bytes -->|<-- chunk_bytes -->|
//     [     chunk 0      ][     chunk 1      ][     chunk 2      ][     chunk 3      ]
//
// interleave=true: chunks are strided by interleave_thread_size * chunk_bytes in GMEM.
//   GMEM layout (thread 0 stores marked with *, other threads fill the gaps):
//     base + row_offset
//     |<- chunk_bytes ->|<- (interleave_thread_size-1)*chunk_bytes gap ->|<- chunk_bytes ->|...
//     [ *chunk 0 (t0)* ][ chunk 0 (t1) ]...[ chunk 0 (tN-1) ]         [ *chunk 1 (t0)* ]...
//
//   Each thread's chunks are interleaved with other threads' data,
//   stride = interleave_thread_size * chunk_bytes bytes between chunks.
//
// Conversion paths (when T_R != DTYPE_I):
//   - T_R is bf16/fp16: per-element type_convert (scalar loop)
//   - otherwise:        vec_convert with inverted_scale (e.g. float -> fp8/fp4)
// When T_R == DTYPE_I: direct store, no conversion.
template <typename T,
          typename DTYPE_I,
          int vec_size,
          int chunk_bytes,
          int aux                    = 0,
          bool interleave            = false,
          int interleave_thread_size = WARP_SIZE,
          typename T_R               = T>
__device__ void store_vector_nbytes(opus::gmem<T>& buffer,
                                    const opus::vector_t<DTYPE_I, vec_size>& vec,
                                    int row_offset,
                                    float inverted_scale = 1.0f)
{
    static constexpr int32_t store_vec_size =
        std::is_same_v<T_R, ck_tile::fp4x2_t> ? vec_size / 2 : vec_size;
    static_assert(store_vec_size * sizeof(T) % chunk_bytes == 0,
                  "store_vec_size * sizeof(T) must be a multiple of chunk_bytes");
    static constexpr index_t num_chunks                = store_vec_size * sizeof(T) / chunk_bytes;
    static constexpr index_t chunk_size_elements       = vec_size / num_chunks;
    static constexpr index_t store_chunk_size_elements = store_vec_size / num_chunks;
    static constexpr index_t interleave_bytes          = interleave_thread_size * chunk_bytes;
    const DTYPE_I* vec_ptr                             = reinterpret_cast<const DTYPE_I*>(&vec);
    using chunk_type   = ck_tile::vec_t<DTYPE_I, chunk_size_elements>;
    using convert_type = std::conditional_t<std::is_same_v<T_R, ck_tile::fp4x2_t>,
                                            ck_tile::vec_t<T_R, chunk_size_elements / 2>,
                                            ck_tile::vec_t<T_R, chunk_size_elements>>;
    using store_type   = opus::vector_t<T, store_chunk_size_elements>;

    opus::static_for<num_chunks>([&](auto i) {
        constexpr index_t chunk_offset_bytes =
            interleave ? i.value * interleave_bytes : i.value * chunk_bytes;
        constexpr index_t chunk_offset_elements = chunk_offset_bytes / sizeof(T);

        const chunk_type* chunk_ptr =
            reinterpret_cast<const chunk_type*>(vec_ptr + i.value * chunk_size_elements);
        if constexpr(!std::is_same_v<T_R, DTYPE_I>)
        {
            convert_type chunk_convert;
            if constexpr(std::is_same_v<T_R, ck_tile::bf16_t> ||
                         std::is_same_v<T_R, ck_tile::fp16_t>)
            {
                for(int j = 0; j < chunk_size_elements; j++)
                {
                    chunk_convert[j] = ck_tile::type_convert<T_R>((*chunk_ptr)[j]);
                }
            }
            else
            {
                chunk_convert = ck_tile::vec_convert<T_R, DTYPE_I, chunk_size_elements>(
                    *chunk_ptr, inverted_scale);
            }
            store_type& chunk_store = reinterpret_cast<store_type&>(chunk_convert);
            buffer.template store<store_chunk_size_elements, store_type, aux>(
                chunk_store, row_offset, chunk_offset_elements);
            // Workaround: compiler may not insert s_nop after the last buffer_store, causing a
            // WAR hazard where vdata VGPRs are overwritten before buffer_store finishes reading
            // them.
            asm volatile("s_nop 0");
        }
        else
        {
            const store_type* chunk_store_ptr = reinterpret_cast<const store_type*>(chunk_ptr);
            buffer.template store<store_chunk_size_elements, store_type, aux>(
                *chunk_store_ptr, row_offset, chunk_offset_elements);
        }
    });
}

// High-level store API: automatically selects the best chunk_bytes (16/8/4) for
// store_vector_nbytes. Picks the largest chunk size that evenly divides the total store bytes.
//
// When interleave=true, num_repeat controls how many interleaved repeats per thread,
// which affects the effective store size used to choose chunk_bytes.
template <typename T,
          typename DTYPE_I,
          int vec_size,
          int aux                    = 0,
          bool interleave            = false,
          int interleave_thread_size = WARP_SIZE,
          int num_repeat             = 1,
          typename T_R               = T>
__device__ void store_vector(opus::gmem<T>& buffer,
                             const opus::vector_t<DTYPE_I, vec_size>& vec,
                             int row_offset,
                             float inverted_scale = 1.0f)
{
    static constexpr int32_t num_store_repeat = interleave ? num_repeat : 1;
    static constexpr int32_t store_vec_size =
        std::is_same_v<T_R, ck_tile::fp4x2_t> ? vec_size / 2 : vec_size;
    if constexpr((store_vec_size * sizeof(T) / num_store_repeat) % 16 == 0)
    {
        store_vector_nbytes<T, DTYPE_I, vec_size, 16, aux, interleave, interleave_thread_size, T_R>(
            buffer, vec, row_offset, inverted_scale);
    }
    else if constexpr((store_vec_size * sizeof(T) / num_store_repeat) % 8 == 0)
    {
        store_vector_nbytes<T, DTYPE_I, vec_size, 8, aux, interleave, interleave_thread_size, T_R>(
            buffer, vec, row_offset, inverted_scale);
    }
    else if constexpr((store_vec_size * sizeof(T) / num_store_repeat) % 4 == 0)
    {
        store_vector_nbytes<T, DTYPE_I, vec_size, 4, aux, interleave, interleave_thread_size, T_R>(
            buffer, vec, row_offset, inverted_scale);
    }
    else
    {
        static_assert(false, "vec_size * sizeof(T) must be a multiple of 16, 8, or 4");
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// scaled type conversion: v_pk_mul_f32 + v_med3_f32 + v_cvt_pk_{fp8,bf8}_f32
// Identical ISA to ck_tile::vec_convert for performance parity

OPUS_D fp32x2_t pk_mul_f32(fp32x2_t a, fp32x2_t b)
{
    fp32x2_t c;
    asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));
    return c;
}

// fp32x2 -> fp8x2 with scale + saturation clamp (E4M3)
// ISA: v_pk_mul_f32 + v_med3_f32 x2 + v_cvt_pk_fp8_f32
template <typename S, std::enable_if_t<std::is_same_v<S, fp32x2_t>, bool> = true>
OPUS_D decltype(auto) fp32_to_fp8_scaled_x2(const S& s, float inverted_scale)
{
    fp32x2_t tmp = pk_mul_f32(s, fp32x2_t{inverted_scale, inverted_scale});
#if defined(__gfx950__)
    constexpr float hi = 448.0f, lo = -448.0f;
#else
    constexpr float hi = 240.0f, lo = -240.0f;
#endif
    float a = tmp[0], b = tmp[1];
    int w;
    asm volatile("v_med3_f32 %1, %1, %3, %4\n"
                 "v_med3_f32 %2, %2, %3, %4\n"
                 "v_cvt_pk_fp8_f32 %0, %1, %2"
                 : "=v"(w), "+v"(a), "+v"(b)
                 : "v"(lo), "v"(hi));
    return __builtin_bit_cast(fp8x2_t, static_cast<int16_t>(w));
}

template <typename S, std::enable_if_t<std::is_same_v<S, fp32x4_t>, bool> = true>
OPUS_D decltype(auto) fp32_to_fp8_scaled_x4(const S& s, float inverted_scale)
{
    auto lo = fp32_to_fp8_scaled_x2(fp32x2_t{s[0], s[1]}, inverted_scale);
    auto hi = fp32_to_fp8_scaled_x2(fp32x2_t{s[2], s[3]}, inverted_scale);
    return fp8x4_t{lo[0], lo[1], hi[0], hi[1]};
}

// fp32x2 -> bf8x2 with scale + saturation clamp (E5M2)
// ISA: v_pk_mul_f32 + v_med3_f32 x2 + v_cvt_pk_bf8_f32
template <typename S, std::enable_if_t<std::is_same_v<S, fp32x2_t>, bool> = true>
OPUS_D decltype(auto) fp32_to_bf8_scaled_x2(const S& s, float inverted_scale)
{
    fp32x2_t tmp       = pk_mul_f32(s, fp32x2_t{inverted_scale, inverted_scale});
    constexpr float hi = 57344.0f, lo = -57344.0f;
    float a = tmp[0], b = tmp[1];
    int w;
    asm volatile("v_med3_f32 %1, %1, %3, %4\n"
                 "v_med3_f32 %2, %2, %3, %4\n"
                 "v_cvt_pk_bf8_f32 %0, %1, %2"
                 : "=v"(w), "+v"(a), "+v"(b)
                 : "v"(lo), "v"(hi));
    return __builtin_bit_cast(bf8x2_t, static_cast<int16_t>(w));
}

template <typename S, std::enable_if_t<std::is_same_v<S, fp32x4_t>, bool> = true>
OPUS_D decltype(auto) fp32_to_bf8_scaled_x4(const S& s, float inverted_scale)
{
    auto lo = fp32_to_bf8_scaled_x2(fp32x2_t{s[0], s[1]}, inverted_scale);
    auto hi = fp32_to_bf8_scaled_x2(fp32x2_t{s[2], s[3]}, inverted_scale);
    return bf8x4_t{lo[0], lo[1], hi[0], hi[1]};
}

// fp32x2 -> i8x2 with scale
// ISA: v_pk_mul_f32 + v_cvt_i32_f32 x2
template <typename S, std::enable_if_t<std::is_same_v<S, fp32x2_t>, bool> = true>
OPUS_D decltype(auto) fp32_to_i8_scaled_x2(const S& s, float inverted_scale)
{
    fp32x2_t tmp = pk_mul_f32(s, fp32x2_t{inverted_scale, inverted_scale});
    return i8x2_t{static_cast<i8_t>(tmp[0]), static_cast<i8_t>(tmp[1])};
}

template <typename S, std::enable_if_t<std::is_same_v<S, fp32x4_t>, bool> = true>
OPUS_D decltype(auto) fp32_to_i8_scaled_x4(const S& s, float inverted_scale)
{
    fp32x2_t tmp0 = pk_mul_f32(fp32x2_t{s[0], s[1]}, fp32x2_t{inverted_scale, inverted_scale});
    fp32x2_t tmp1 = pk_mul_f32(fp32x2_t{s[2], s[3]}, fp32x2_t{inverted_scale, inverted_scale});
    return i8x4_t{static_cast<i8_t>(tmp0[0]),
                  static_cast<i8_t>(tmp0[1]),
                  static_cast<i8_t>(tmp1[0]),
                  static_cast<i8_t>(tmp1[1])};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// fp16x2 -> fp4 with scale (v_cvt_scalef32_pk_fp4_f16, gfx950 only)
// opus.hpp has fp32->fp4 and bf16->fp4 but NOT fp16->fp4
#if defined(__gfx950__)
template <typename S, index_t sel = 0, std::enable_if_t<std::is_same_v<S, fp16x2_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp16_to_fp4_scaled_x2(const S& s, float scale, number<sel> = {})
{
    u32_t w;
    w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(w, s, scale, sel);
    return __builtin_bit_cast(array<fp4_t, 1>, static_cast<u8_t>(w));
}
template <typename S, std::enable_if_t<std::is_same_v<S, fp16x4_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp16_to_fp4_scaled_x4(const S& s, float scale)
{
    u32_t w;
    w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(w, fp16x2_t{s[0], s[1]}, scale, 0);
    w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(w, fp16x2_t{s[2], s[3]}, scale, 1);
    return __builtin_bit_cast(array<fp4_t, 2>, static_cast<u16_t>(w));
}
template <typename S, std::enable_if_t<std::is_same_v<S, fp16x8_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp16_to_fp4_scaled_x8(const S& s, float scale)
{
    u32_t w;
    w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(w, fp16x2_t{s[0], s[1]}, scale, 0);
    w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(w, fp16x2_t{s[2], s[3]}, scale, 1);
    w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(w, fp16x2_t{s[4], s[5]}, scale, 2);
    w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(w, fp16x2_t{s[6], s[7]}, scale, 3);
    return __builtin_bit_cast(array<fp4_t, 4>, w);
}
#else
template <typename S, std::enable_if_t<std::is_same_v<S, fp16x2_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp16_to_fp4_scaled_x2(const S&, float)
{
    return array<fp4_t, 1>{};
}
template <typename S, std::enable_if_t<std::is_same_v<S, fp16x4_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp16_to_fp4_scaled_x4(const S&, float)
{
    return array<fp4_t, 2>{};
}
template <typename S, std::enable_if_t<std::is_same_v<S, fp16x8_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp16_to_fp4_scaled_x8(const S&, float)
{
    return array<fp4_t, 4>{};
}
#endif

// bf16 -> fp4 larger vectors (bf16x4/x8) using opus bf16_to_fp4_packed_x2
template <typename S, std::enable_if_t<std::is_same_v<S, bf16x4_t>, bool> = true>
OPUS_D constexpr decltype(auto) bf16_to_fp4_scaled_x4(const S& s, float scale)
{
    auto lo = bf16_to_fp4_packed_x2(bf16x2_t{s[0], s[1]}, scale);
    auto hi = bf16_to_fp4_packed_x2(bf16x2_t{s[2], s[3]}, scale);
    return array<fp4_t, 2>{lo, hi};
}
template <typename S, std::enable_if_t<std::is_same_v<S, bf16x8_t>, bool> = true>
OPUS_D constexpr decltype(auto) bf16_to_fp4_scaled_x8(const S& s, float scale)
{
    auto a = bf16_to_fp4_packed_x2(bf16x2_t{s[0], s[1]}, scale);
    auto b = bf16_to_fp4_packed_x2(bf16x2_t{s[2], s[3]}, scale);
    auto c = bf16_to_fp4_packed_x2(bf16x2_t{s[4], s[5]}, scale);
    auto d = bf16_to_fp4_packed_x2(bf16x2_t{s[6], s[7]}, scale);
    return array<fp4_t, 4>{a, b, c, d};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// scaled_cast: type conversion with scale multiplication (ck_tile::vec_convert equivalent)
// Usage: aiter::scaled_cast<fp8_t>(fp32_vec, inverted_scale)

// --- 8-bit targets (fp8, bf8, i8): fp32 source x2/x4 ---
template <typename D,
          typename S,
          std::enable_if_t<std::is_same_v<S, fp32x2_t> && std::is_same_v<D, fp8_t>, bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    return fp32_to_fp8_scaled_x2(s, inverted_scale);
}
template <typename D,
          typename S,
          std::enable_if_t<std::is_same_v<S, fp32x2_t> && std::is_same_v<D, bf8_t>, bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    return fp32_to_bf8_scaled_x2(s, inverted_scale);
}
template <typename D,
          typename S,
          std::enable_if_t<std::is_same_v<S, fp32x2_t> && std::is_same_v<D, i8_t>, bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    return fp32_to_i8_scaled_x2(s, inverted_scale);
}
template <typename D,
          typename S,
          std::enable_if_t<std::is_same_v<S, fp32x4_t> && std::is_same_v<D, fp8_t>, bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    return fp32_to_fp8_scaled_x4(s, inverted_scale);
}
template <typename D,
          typename S,
          std::enable_if_t<std::is_same_v<S, fp32x4_t> && std::is_same_v<D, bf8_t>, bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    return fp32_to_bf8_scaled_x4(s, inverted_scale);
}
template <typename D,
          typename S,
          std::enable_if_t<std::is_same_v<S, fp32x4_t> && std::is_same_v<D, i8_t>, bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    return fp32_to_i8_scaled_x4(s, inverted_scale);
}

// --- fp4 target: fp32 source (delegates to opus cast<fp4_t>) ---
template <typename D,
          typename S,
          std::enable_if_t<std::is_same_v<S, fp32x2_t> && std::is_same_v<D, fp4_t>, bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    return fp32_to_fp4_packed_x2(s, inverted_scale);
}
template <typename D,
          typename S,
          std::enable_if_t<std::is_same_v<S, fp32x4_t> && std::is_same_v<D, fp4_t>, bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    return fp32_to_fp4_packed_x4(s, inverted_scale);
}
template <typename D,
          typename S,
          std::enable_if_t<std::is_same_v<S, fp32x8_t> && std::is_same_v<D, fp4_t>, bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    return fp32_to_fp4_packed_x8(s, inverted_scale);
}

// --- fp4 target: bf16 source ---
template <typename D,
          typename S,
          std::enable_if_t<std::is_same_v<S, bf16x2_t> && std::is_same_v<D, fp4_t>, bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    return bf16_to_fp4_packed_x2(s, inverted_scale);
}
template <typename D,
          typename S,
          std::enable_if_t<std::is_same_v<S, bf16x4_t> && std::is_same_v<D, fp4_t>, bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    return bf16_to_fp4_scaled_x4(s, inverted_scale);
}
template <typename D,
          typename S,
          std::enable_if_t<std::is_same_v<S, bf16x8_t> && std::is_same_v<D, fp4_t>, bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    return bf16_to_fp4_scaled_x8(s, inverted_scale);
}

// --- fp4 target: fp16 source ---
template <typename D,
          typename S,
          std::enable_if_t<std::is_same_v<S, fp16x2_t> && std::is_same_v<D, fp4_t>, bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    return fp16_to_fp4_scaled_x2(s, inverted_scale);
}
template <typename D,
          typename S,
          std::enable_if_t<std::is_same_v<S, fp16x4_t> && std::is_same_v<D, fp4_t>, bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    return fp16_to_fp4_scaled_x4(s, inverted_scale);
}
template <typename D,
          typename S,
          std::enable_if_t<std::is_same_v<S, fp16x8_t> && std::is_same_v<D, fp4_t>, bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    return fp16_to_fp4_scaled_x8(s, inverted_scale);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// auto-fold: build flat output vector using x2 primitives in a loop

// 8-bit targets (fp8, bf8, i8): any fp32 vector size via x2 loop
template <typename D,
          typename S,
          std::enable_if_t<is_vector_v<S> && std::is_same_v<get_value_t<S>, fp32_t> &&
                               !is_any_of_v<S, fp32x2_t, fp32x4_t> &&
                               (std::is_same_v<D, fp8_t> || std::is_same_v<D, bf8_t> ||
                                std::is_same_v<D, i8_t>),
                           bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    constexpr index_t N = size<S>();
    static_assert(N % 2 == 0);
    vector_t<D, N> out;
    static_for<N / 2>([&](auto i) {
        auto pair = scaled_cast<D>(fp32x2_t{s[i.value * 2], s[i.value * 2 + 1]}, inverted_scale);
        out[i.value * 2]     = pair[0];
        out[i.value * 2 + 1] = pair[1];
    });
    return out;
}

// two-hop: non-fp32 source -> convert to fp32 via static_cast -> scaled_cast to 8-bit target
// Uses static_cast<float> instead of opus::cast to handle _Float16/__fp16 mismatch
template <typename D,
          typename S,
          std::enable_if_t<is_vector_v<S> && !std::is_same_v<get_value_t<S>, fp32_t> &&
                               (std::is_same_v<D, fp8_t> || std::is_same_v<D, bf8_t> ||
                                std::is_same_v<D, i8_t>),
                           bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    constexpr index_t N = size<S>();
    vector_t<fp32_t, N> fp32_vec;
    static_for<N>([&](auto i) { fp32_vec[i.value] = static_cast<float>(s[i.value]); });
    return scaled_cast<D>(fp32_vec, inverted_scale);
}

// general fallback: fp32 source -> any non-quantized target with scale
template <typename D,
          typename S,
          std::enable_if_t<is_vector_v<S> && std::is_same_v<get_value_t<S>, fp32_t> &&
                               !is_any_of_v<D, fp8_t, bf8_t, i8_t, fp4_t>,
                           bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    constexpr index_t N = size<S>();
    S tmp;
    static_for<N>([&](auto i) { tmp[i.value] = s[i.value] * inverted_scale; });
    if constexpr(std::is_same_v<D, fp32_t>)
    {
        return tmp;
    }
    else
    {
        return cast<D>(tmp);
    }
}

// general fallback: non-fp32 source -> any non-quantized target with scale (two-hop via fp32)
template <typename D,
          typename S,
          std::enable_if_t<is_vector_v<S> && !std::is_same_v<get_value_t<S>, fp32_t> &&
                               !is_any_of_v<D, fp8_t, bf8_t, i8_t, fp4_t>,
                           bool> = true>
OPUS_D decltype(auto) scaled_cast(const S& s, float inverted_scale)
{
    constexpr index_t N = size<S>();
    vector_t<fp32_t, N> fp32_vec;
    static_for<N>([&](auto i) { fp32_vec[i.value] = static_cast<float>(s[i.value]); });
    return scaled_cast<D>(fp32_vec, inverted_scale);
}

} // namespace aiter
