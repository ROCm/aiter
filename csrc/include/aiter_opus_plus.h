// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "vec_convert.h"
#include "opus.hpp"
#include "hip_reduce.h"

namespace aiter {
    #define RT 0
    #define GROUP_NT 3

    using index_t = int;

    template <typename T, int vec_size, int chunk_bytes, int aux = 0, bool interleave = false>
    __device__ opus::vector_t<T, vec_size> load_vector_nbytes(opus::gmem<T>& buffer, int row_offset) {
        static_assert(vec_size * sizeof(T) % chunk_bytes == 0, "vec_size * sizeof(T) must be a multiple of chunk_bytes");
        static constexpr index_t num_chunks = vec_size * sizeof(T) / chunk_bytes;
        constexpr index_t chunk_size_elements = chunk_bytes / sizeof(T);

        opus::vector_t<T, vec_size> result;
        T* result_ptr = reinterpret_cast<T*>(&result);

        opus::static_for<num_chunks>([&](auto i) {
            constexpr index_t chunk_offset_bytes = interleave ? i.value * WARP_SIZE * chunk_bytes : i.value * chunk_bytes;
            constexpr index_t chunk_offset_elements = chunk_offset_bytes / sizeof(T);

            opus::vector_t<T, chunk_size_elements> *chunk_ptr = reinterpret_cast<opus::vector_t<T, chunk_size_elements> *>(result_ptr + i.value * chunk_size_elements);
            *chunk_ptr = buffer.template load<chunk_size_elements, aux>(row_offset, chunk_offset_elements);
        });

        return result;
    }

    template <typename T, typename DTYPE_I, int vec_size, int chunk_bytes, int aux = 0, bool interleave = false, typename T_R = T>
    __device__ void store_vector_nbytes(opus::gmem<T>& buffer, const opus::vector_t<DTYPE_I, vec_size>& vec, int row_offset, float inverted_scale=1.0f) {
        static constexpr int32_t store_vec_size = std::is_same_v<T_R, ck_tile::fp4x2_t> ? vec_size / 2 : vec_size;
        static_assert(store_vec_size * sizeof(T) % chunk_bytes == 0, "store_vec_size * sizeof(T) must be a multiple of chunk_bytes");
        static constexpr index_t num_chunks = store_vec_size * sizeof(T) / chunk_bytes;
        static constexpr index_t chunk_size_elements = vec_size / num_chunks;
        static constexpr index_t store_chunk_size_elements = store_vec_size / num_chunks;
        const DTYPE_I* vec_ptr = reinterpret_cast<const DTYPE_I*>(&vec);
        using chunk_type = ck_tile::vec_t<DTYPE_I, chunk_size_elements>;
        using convert_type = std::conditional_t<std::is_same_v<T_R, ck_tile::fp4x2_t>, ck_tile::vec_t<T_R, chunk_size_elements/2>, ck_tile::vec_t<T_R, chunk_size_elements>>;
        using store_type = opus::vector_t<T, store_chunk_size_elements>;

        opus::static_for<num_chunks>([&](auto i) {
            constexpr index_t chunk_offset_bytes = interleave ? i.value * WARP_SIZE * chunk_bytes : i.value * chunk_bytes;
            constexpr index_t chunk_offset_elements = chunk_offset_bytes / sizeof(T);

            const chunk_type* chunk_ptr = reinterpret_cast<const chunk_type*>(vec_ptr + i.value * chunk_size_elements);
            if constexpr(!std::is_same_v<T_R, DTYPE_I>)
            {
                convert_type chunk_convert;
                if constexpr(std::is_same_v<T_R, ck_tile::bf16_t> || std::is_same_v<T_R, ck_tile::fp16_t>)
                {
                    for(int j = 0; j < chunk_size_elements; j++)
                    {
                        chunk_convert[j] = ck_tile::type_convert<T_R>((*chunk_ptr)[j]);
                    }
                }
                else
                {
                    chunk_convert = ck_tile::vec_convert<T_R, DTYPE_I, chunk_size_elements>(*chunk_ptr, inverted_scale);
                }
                store_type& chunk_store = reinterpret_cast<store_type&>(chunk_convert);
                buffer.template store<store_chunk_size_elements, store_type, aux>(chunk_store, row_offset, chunk_offset_elements);
            }
            else
            {
                store_type& chunk_store = reinterpret_cast<store_type&>(*chunk_ptr);
                buffer.template store<store_chunk_size_elements, store_type, aux>(chunk_store, row_offset, chunk_offset_elements);
            }
        });
    }

    template <typename T, typename DTYPE_I, int vec_size, int aux = 0, bool interleave = false, int num_repeat = 1, typename T_R = T>
    __device__ void store_vector(opus::gmem<T>& buffer, const opus::vector_t<DTYPE_I, vec_size>& vec, int row_offset, float inverted_scale=1.0f) {
        static constexpr int32_t num_store_repeat = interleave ? num_repeat : 1;
        static constexpr int32_t store_vec_size = std::is_same_v<T_R, ck_tile::fp4x2_t> ? vec_size / 2 : vec_size;
        if constexpr((store_vec_size * sizeof(T) / num_store_repeat) % 16 == 0) {
            store_vector_nbytes<T, DTYPE_I, vec_size, 16, aux, interleave, T_R>(buffer, vec, row_offset, inverted_scale);
        } else if constexpr((store_vec_size * sizeof(T) / num_store_repeat) % 8 == 0) {
            store_vector_nbytes<T, DTYPE_I, vec_size, 8, aux, interleave, T_R>(buffer, vec, row_offset, inverted_scale);
        } else if constexpr((store_vec_size * sizeof(T) / num_store_repeat) % 4 == 0) {
            store_vector_nbytes<T, DTYPE_I, vec_size, 4, aux, interleave, T_R>(buffer, vec, row_offset, inverted_scale);
        } else {
            static_assert(false, "vec_size * sizeof(T) must be a multiple of 16, 8, or 4");
        }
    }
}
