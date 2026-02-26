// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "hip_reduce.h"
#include "opus.hpp"
#include "vec_convert.h"

namespace aiter {
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
} // namespace aiter
