// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Self-contained coalesced vectorized IO for the opus rmsnorm rewrite: copies of the
// generic load/store_vector device functions (chunked buffer_load/store), with the
// output conversion expressed via opus internal API (opus::cast, opus::med3,
// opus::fp32_to_fp8_packed / fp32_to_i8 / fp32_to_fp4_packed). Depends only on opus.hpp
// -- no aiter_opus_plus.h (torch/c10) and no scaled_cast helper layer.
#pragma once
#include "opus/opus.hpp"
#include <cstdint>

namespace aiter {
using namespace opus;
using index_t = int;

// buffer-resource aux codes passed to load/store (RT: regular, GROUP_NT: grouped no-cache).
#define RT 0
#define GROUP_NT 3

// Packed fp32x2 multiply (v_pk_mul_f32 on CDNA; element-wise elsewhere).
OPUS_D fp32x2_t pk_mul_f32(fp32x2_t a, fp32x2_t b)
{
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || \
    defined(__gfx950__)
    fp32x2_t c;
    asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));
    return c;
#else
    return fp32x2_t{a[0] * b[0], a[1] * b[1]};
#endif
}

// fp32 -> quant (i8/fp8) with scale, using opus primitives (replaces scaled_cast).
// i8: truncate(x*inv) -- matches the CK reference (yscale keeps values in [-qmax,qmax]).
// fp8: v_med3 clamp to +/-max then packed v_cvt_pk_fp8 (same ISA as the reference).
template <typename T_R, int N>
OPUS_D vector_t<T_R, N> scaled_to_quant(const vector_t<float, N>& s, float inv)
{
    vector_t<T_R, N> out;
    if constexpr(std::is_same_v<T_R, i8_t>)
    {
#pragma unroll
        for(int j = 0; j < N; ++j)
            out[j] = fp32_to_i8(s[j] * inv);
    }
    else // fp8_t
    {
#if defined(__gfx942__)
        constexpr float mx = 240.0f; // e4m3 fnuz max
#else
        constexpr float mx = 448.0f; // e4m3 ocp max
#endif
#pragma unroll
        for(int j = 0; j < N; j += 2)
        {
            auto pk = fp32_to_fp8_packed_x2(
                fp32x2_t{med3(s[j] * inv, -mx, mx), med3(s[j + 1] * inv, -mx, mx)});
            out[j]     = pk[0];
            out[j + 1] = pk[1];
        }
    }
    return out;
}

// fp32 -> fp4 (packed 2/byte) with scale, via opus's packed cast in a loop so any vec size
// works (opus::cast<fp4_t> only covers x2/x4/x8). Returns array<fp4_t,N/2> (N/2 bytes).
template <int N>
OPUS_D array<fp4_t, N / 2> scaled_to_fp4(const vector_t<float, N>& s, float scale)
{
    array<fp4_t, N / 2> out;
    const float* sp = reinterpret_cast<const float*>(&s);
    fp4_t* op       = reinterpret_cast<fp4_t*>(&out);
    constexpr int G = (N % 8 == 0) ? 8 : (N % 4 == 0 ? 4 : 2); // opus packs x2/x4/x8
#pragma unroll
    for(int i = 0; i < N; i += G)
    {
        vector_t<float, G> chunk;
#pragma unroll
        for(int k = 0; k < G; ++k)
            chunk[k] = sp[i + k];
        auto packed = opus::cast<fp4_t>(chunk, scale); // array<fp4_t, G/2>
#pragma unroll
        for(int k = 0; k < G / 2; ++k)
            op[i / 2 + k] = packed[k];
    }
    return out;
}

// Load vec_size elements of T from a gmem buffer in chunks (one buffer_load per chunk;
// chunk_bytes 4/8/16 -> dword/x2/x4). interleave=true strides chunks by
// interleave_thread_size*chunk_bytes for coalesced per-thread layout.
template <typename T,
          int vec_size,
          int chunk_bytes,
          int aux                    = 0,
          bool interleave            = false,
          int interleave_thread_size = opus::get_warp_size()>
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
            load<chunk_size_elements>(buffer, row_offset, chunk_offset_elements, opus::number<aux>{});
    });

    return result;
}

// Store vec_size DTYPE_I elements to gmem as T_R (default T), converting per chunk:
// bf16/fp16 via opus::cast; fp4 via opus::cast<fp4_t>(.,scale); i8/fp8 via scaled_to_quant.
template <typename T,
          typename DTYPE_I,
          int vec_size,
          int chunk_bytes,
          int aux                    = 0,
          bool interleave            = false,
          int interleave_thread_size = opus::get_warp_size(),
          typename T_R               = T>
__device__ void store_vector_nbytes(opus::gmem<T>& buffer,
                                    const opus::vector_t<DTYPE_I, vec_size>& vec,
                                    int row_offset,
                                    float inverted_scale = 1.0f)
{
    static constexpr int32_t store_vec_size =
        std::is_same_v<T_R, opus::fp4_t> ? vec_size / 2 : vec_size;
    static_assert(store_vec_size * sizeof(T) % chunk_bytes == 0,
                  "store_vec_size * sizeof(T) must be a multiple of chunk_bytes");
    static constexpr index_t num_chunks                = store_vec_size * sizeof(T) / chunk_bytes;
    static constexpr index_t chunk_size_elements       = vec_size / num_chunks;
    static constexpr index_t store_chunk_size_elements = store_vec_size / num_chunks;
    static constexpr index_t interleave_bytes          = interleave_thread_size * chunk_bytes;
    const DTYPE_I* vec_ptr                             = reinterpret_cast<const DTYPE_I*>(&vec);
    using chunk_type = opus::vector_t<DTYPE_I, chunk_size_elements>;
    using store_type = opus::vector_t<T, store_chunk_size_elements>;

    opus::static_for<num_chunks>([&](auto i) {
        constexpr index_t chunk_offset_bytes =
            interleave ? i.value * interleave_bytes : i.value * chunk_bytes;
        constexpr index_t chunk_offset_elements = chunk_offset_bytes / sizeof(T);

        const chunk_type* chunk_ptr =
            reinterpret_cast<const chunk_type*>(vec_ptr + i.value * chunk_size_elements);
        if constexpr(!std::is_same_v<T_R, DTYPE_I>)
        {
            if constexpr(std::is_same_v<T_R, opus::bf16_t> || std::is_same_v<T_R, opus::fp16_t>)
            {
                opus::vector_t<T_R, chunk_size_elements> chunk_convert;
                for(int j = 0; j < chunk_size_elements; j++)
                    chunk_convert[j] = opus::cast<T_R>((*chunk_ptr)[j]);
                store_type& chunk_store = reinterpret_cast<store_type&>(chunk_convert);
                store<store_chunk_size_elements>(
                    buffer, chunk_store, row_offset, chunk_offset_elements, opus::number<aux>{});
            }
            else if constexpr(std::is_same_v<T_R, opus::fp4_t>)
            {
                auto chunk_convert = scaled_to_fp4<chunk_size_elements>(*chunk_ptr, inverted_scale);
                store_type& chunk_store = reinterpret_cast<store_type&>(chunk_convert);
                store<store_chunk_size_elements>(
                    buffer, chunk_store, row_offset, chunk_offset_elements, opus::number<aux>{});
            }
            else // i8 / fp8
            {
                auto chunk_convert =
                    scaled_to_quant<T_R, chunk_size_elements>(*chunk_ptr, inverted_scale);
                store_type& chunk_store = reinterpret_cast<store_type&>(chunk_convert);
                store<store_chunk_size_elements>(
                    buffer, chunk_store, row_offset, chunk_offset_elements, opus::number<aux>{});
            }
            // s_nop guards a WAR hazard: the compiler may not fence vdata after the last
            // buffer_store before those VGPRs are reused.
            asm volatile("s_nop 0");
        }
        else
        {
            const store_type* chunk_store_ptr = reinterpret_cast<const store_type*>(chunk_ptr);
            store<store_chunk_size_elements>(
                buffer, *chunk_store_ptr, row_offset, chunk_offset_elements, opus::number<aux>{});
        }
    });
}

// Picks the largest chunk (16/8/4 bytes) that divides the store, then store_vector_nbytes.
template <typename T,
          typename DTYPE_I,
          int vec_size,
          int aux                    = 0,
          bool interleave            = false,
          int interleave_thread_size = opus::get_warp_size(),
          int num_repeat             = 1,
          typename T_R               = T>
__device__ void store_vector(opus::gmem<T>& buffer,
                             const opus::vector_t<DTYPE_I, vec_size>& vec,
                             int row_offset,
                             float inverted_scale = 1.0f)
{
    static constexpr int32_t num_store_repeat = interleave ? num_repeat : 1;
    static constexpr int32_t store_vec_size =
        std::is_same_v<T_R, opus::fp4_t> ? vec_size / 2 : vec_size;
    if constexpr((store_vec_size * sizeof(T) / num_store_repeat) % 16 == 0)
        store_vector_nbytes<T, DTYPE_I, vec_size, 16, aux, interleave, interleave_thread_size, T_R>(
            buffer, vec, row_offset, inverted_scale);
    else if constexpr((store_vec_size * sizeof(T) / num_store_repeat) % 8 == 0)
        store_vector_nbytes<T, DTYPE_I, vec_size, 8, aux, interleave, interleave_thread_size, T_R>(
            buffer, vec, row_offset, inverted_scale);
    else if constexpr((store_vec_size * sizeof(T) / num_store_repeat) % 4 == 0)
        store_vector_nbytes<T, DTYPE_I, vec_size, 4, aux, interleave, interleave_thread_size, T_R>(
            buffer, vec, row_offset, inverted_scale);
    else
        static_assert(false, "vec_size * sizeof(T) must be a multiple of 16, 8, or 4");
}

} // namespace aiter
