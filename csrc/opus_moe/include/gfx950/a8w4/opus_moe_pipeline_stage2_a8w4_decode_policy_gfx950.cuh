// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_traits_stage2_a8w4_decode_gfx950.cuh"
#include "../../opus_moe_common.cuh"
#include "opus/opus.hpp"

#include <hip/hip_bfloat16.h>

// Shape-derived pipeline policies.
enum class OpusMoeStage2A8W4DecodeMainloopSchedule
{
    Baseline,
    SplitALoadByNWave,
};

enum class OpusMoeStage2A8W4DecodeCShuffleSchedule
{
    Baseline,
    WideRows,
};

template<typename T>
struct OpusMoeStage2A8W4DecodeSchedule
{
    using MainloopSchedule = OpusMoeStage2A8W4DecodeMainloopSchedule;
    using CShuffleSchedule = OpusMoeStage2A8W4DecodeCShuffleSchedule;

    static constexpr MainloopSchedule Mainloop =
        (T::IS_BM32_BN256 && T::M_MFMA_PER_WAVE == 1)
            ? MainloopSchedule::SplitALoadByNWave
            : MainloopSchedule::Baseline;

    static constexpr bool MainloopSplitsALoadByNWave =
        Mainloop == MainloopSchedule::SplitALoadByNWave;
    static constexpr bool MainloopEndsWithSmemBarrier = MainloopSplitsALoadByNWave;

    static constexpr CShuffleSchedule CShuffle =
        (T::IS_BM32_BN256 || T::IS_BM64_BN256)
            ? CShuffleSchedule::WideRows
            : CShuffleSchedule::Baseline;

    static constexpr int CShuffleNLane =
        CShuffle == CShuffleSchedule::WideRows
            ? 2 * opus::get_warp_size()
            : opus::get_warp_size() / 2;
};

#ifdef __HIP_DEVICE_COMPILE__

// Layout helpers shared by the A8W4 decode pipeline.
template<typename T>
inline __device__ auto opus_moe_stage2_a8w4_layout_ga(int lane_id, int wave_id_m)
{
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::T_M>{},
        opus::number<T::M_MFMA_PER_WAVE>{},
        opus::number<T::MMA_M>{},
        opus::number<T::THREADS_K>{},
        opus::number<T::VEC_A>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        block_shape,
        opus::unfold_x_stride(
            block_dim,
            block_shape,
            opus::tuple{opus::number<T::K_STEP_PACKED>{}, opus::number<1>{}}),
        opus::unfold_p_coord(
            block_dim,
            opus::tuple{wave_id_m, lane_id % T::MMA_M, lane_id / T::MMA_M}));
}

template<typename T>
inline __device__ int opus_moe_stage2_a8w4_a_local_m(int ga_offset)
{
    static_assert((T::K_STEP_PACKED & (T::K_STEP_PACKED - 1)) == 0);
    return ga_offset / T::K_STEP_PACKED;
}

template<typename T>
inline __device__ int opus_moe_stage2_a8w4_a_k_byte(int ga_offset)
{
    static_assert((T::K_STEP_PACKED & (T::K_STEP_PACKED - 1)) == 0);
    constexpr int kStepMask = T::K_STEP_PACKED - 1;
    return ga_offset & kStepMask;
}

template<typename T>
inline __device__ int opus_moe_stage2_a8w4_a_lane_k(int ga_offset)
{
    return opus_moe_stage2_a8w4_a_k_byte<T>(ga_offset) / T::VEC_A;
}

template<typename T>
inline __device__ int opus_moe_stage2_a8w4_a_payload_byte_offset(int a_base,
                                                                 int k_base,
                                                                 int ga_offset)
{
    constexpr auto block_shape = opus::make_tuple(
        opus::number<1>{},
        opus::number<1>{},
        opus::number<T::K_STEP_PACKED>{});
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}));
    constexpr auto u = opus::make_layout<-1>(
        block_shape,
        opus::unfold_x_stride(
            block_dim,
            block_shape,
            opus::tuple{opus::number<1>{}, opus::number<1>{}, opus::number<1>{}}));
    return static_cast<int>(u(a_base, k_base, opus_moe_stage2_a8w4_a_k_byte<T>(ga_offset)));
}

template<typename T>
inline __device__ constexpr auto opus_moe_stage2_a8w4_layout_scale_word()
{
    constexpr auto block_shape = opus::make_tuple(
        opus::number<1>{},
        opus::number<T::THREADS_K>{},
        opus::number<T::MMA_M>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}));

    return opus::make_layout<-1>(
        block_shape,
        opus::unfold_x_stride(
            block_dim,
            block_shape,
            opus::tuple{opus::number<T::SCALE_WORDS_PER_ROW_PACK>{},
                        opus::number<T::MMA_M>{},
                        opus::number<1>{}}));
}

template<typename T>
inline __device__ int opus_moe_stage2_a8w4_a_scale_base_word_offset(int route_base,
                                                                    int ga_offset)
{
    const int local_m = opus_moe_stage2_a8w4_a_local_m<T>(ga_offset);
    const int row_pack =
        route_base / T::SCALE_ROWS_PER_ROW_PACK +
        local_m / T::SCALE_ROWS_PER_ROW_PACK;
    const int row_lane = local_m % T::MMA_M;
    constexpr auto u = opus_moe_stage2_a8w4_layout_scale_word<T>();
    return static_cast<int>(u(row_pack, opus_moe_stage2_a8w4_a_lane_k<T>(ga_offset), row_lane));
}

template<typename T>
inline __device__ auto opus_moe_stage2_a8w4_layout_sa(int lane_id, int wave_id_m)
{
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::T_M>{},
        opus::number<T::M_MFMA_PER_WAVE>{},
        opus::number<2>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_A>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        block_shape,
        opus::unfold_x_stride(
            block_dim,
            block_shape,
            opus::tuple{opus::number<opus::get_warp_size() * T::VEC_A>{},
                        opus::number<1>{}}),
        opus::unfold_p_coord(
            block_dim,
            opus::tuple{wave_id_m, lane_id}));
}

template<typename T>
inline __device__ auto opus_moe_stage2_a8w4_layout_gb(int lane_id, int wave_id_n)
{
    constexpr int packed_col_group_bytes =
        T::MMA_M * T::B_PAYLOAD_ROW_STRIDE_BYTES;
    constexpr int packed_k_group_bytes = T::B_PAYLOAD_KLANE_STRIDE_BYTES;

    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::N_MFMA_PER_WAVE>{},
        opus::number<T::THREADS_K>{},
        opus::number<T::MMA_M>{},
        opus::number<T::B_BYTES_PER_VEC>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        block_shape,
        opus::unfold_x_stride(
            block_dim,
            block_shape,
            opus::tuple{opus::number<packed_col_group_bytes>{},
                        opus::number<packed_k_group_bytes>{},
                        opus::number<1>{}}),
        opus::unfold_p_coord(
            block_dim,
            opus::tuple{wave_id_n, lane_id / T::MMA_M, lane_id % T::MMA_M}));
}

template<typename T>
inline __device__ int opus_moe_stage2_a8w4_b_payload_tile_base_byte_offset(int col_base,
                                                                           int k_base)
{
    constexpr auto block_shape = opus::make_tuple(opus::number<1>{}, opus::number<1>{});
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}));
    constexpr auto u = opus::make_layout<-1>(
        block_shape,
        opus::unfold_x_stride(
            block_dim,
            block_shape,
            opus::tuple{opus::number<T::B_PAYLOAD_ROW_STRIDE_BYTES>{},
                        opus::number<T::B_PAYLOAD_K_STRIDE_BYTES>{}}));
    return static_cast<int>(u(col_base, k_base));
}

template<typename T>
inline __device__ int opus_moe_stage2_a8w4_b_lane_m(int gb_offset)
{
    constexpr int col_byte_mask = T::B_PAYLOAD_KLANE_STRIDE_BYTES - 1;
    const int col_byte = gb_offset & col_byte_mask;
    return col_byte / T::B_BYTES_PER_VEC;
}

template<typename T>
inline __device__ int opus_moe_stage2_a8w4_b_lane_k(int gb_offset)
{
    constexpr int thread_k_mask = T::THREADS_K - 1;
    return (gb_offset / T::B_PAYLOAD_KLANE_STRIDE_BYTES) & thread_k_mask;
}

template<typename T>
inline __device__ int opus_moe_stage2_a8w4_b_wave_id_n(int gb_offset)
{
    constexpr int b_wave_stride_bytes =
        T::N_MFMA_PER_WAVE * T::MMA_N * T::B_PAYLOAD_ROW_STRIDE_BYTES;
    return gb_offset / b_wave_stride_bytes;
}

template<typename T>
inline __device__ int opus_moe_stage2_a8w4_b_scale_base_word_offset(int scale_row_col_base,
                                                                    int gb_offset)
{
    const int row_pack =
        scale_row_col_base / T::SCALE_ROWS_PER_ROW_PACK +
        opus_moe_stage2_a8w4_b_wave_id_n<T>(gb_offset) *
            T::HALF_N_MFMA_PER_WAVE;
    constexpr auto u = opus_moe_stage2_a8w4_layout_scale_word<T>();
    return static_cast<int>(u(row_pack,
                              opus_moe_stage2_a8w4_b_lane_k<T>(gb_offset),
                              opus_moe_stage2_a8w4_b_lane_m<T>(gb_offset)));
}

template<typename T>
inline __device__ int opus_moe_stage2_a8w4_b_scale_word_offset(int base_word_offset,
                                                               int k_group_word_base,
                                                               int pair)
{
    constexpr auto block_shape = opus::make_tuple(
        opus::number<1>{},
        opus::number<1>{},
        opus::number<T::HALF_N_MFMA_PER_WAVE>{});
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}));
    constexpr auto u = opus::make_layout<-1>(
        block_shape,
        opus::unfold_x_stride(
            block_dim,
            block_shape,
            opus::tuple{opus::number<1>{},
                        opus::number<1>{},
                        opus::number<T::SCALE_WORDS_PER_ROW_PACK>{}}));
    return static_cast<int>(u(base_word_offset, k_group_word_base, pair));
}

template<typename T>
struct OpusMoeStage2A8W4CShuffleLayout
{
    static constexpr int CSHUFFLE_NLANE =
        OpusMoeStage2A8W4DecodeSchedule<T>::CShuffleNLane;
    static constexpr int ELEM_PER_ATOMIC = T::ELEM_PER_ATOMIC;
    static constexpr int ELEM_PER_ATOMIC_MASK = ELEM_PER_ATOMIC - 1;
    static constexpr int CSHUFFLE_MLANE = T::BLOCK_SIZE / CSHUFFLE_NLANE;
    static constexpr int PAIRS_PER_ROW = T::C_LDS_N / ELEM_PER_ATOMIC;

    int tid;
    int lane_id;
    int wave_id_m;
    int wave_id_n;

    inline __device__ static int smem_pair_index(int local_m, int elem_col)
    {
        static_assert((PAIRS_PER_ROW & (PAIRS_PER_ROW - 1)) == 0);
        const int pair_col = elem_col / ELEM_PER_ATOMIC;
        if constexpr(T::DIRECT_ATOMIC_OUT)
            return local_m * PAIRS_PER_ROW + pair_col;
        else
        {
            // Route-out stores read four rows per wave; offset row groups by
            // eight pairs to avoid the compiler's high-VGPR row-major path.
            constexpr int ROW_SWIZZLE_STEP = PAIRS_PER_ROW / 8;
            static_assert(ROW_SWIZZLE_STEP > 0);
            const int row_swizzle = (local_m & 3) * ROW_SWIZZLE_STEP;
            return local_m * PAIRS_PER_ROW + (pair_col ^ row_swizzle);
        }
    }

    inline __device__ static int smem_scalar_index(int local_m, int elem_col)
    {
        static_assert((ELEM_PER_ATOMIC & ELEM_PER_ATOMIC_MASK) == 0);
        const int elem_col0 = elem_col & ~ELEM_PER_ATOMIC_MASK;
        return smem_pair_index(local_m, elem_col0) * ELEM_PER_ATOMIC +
               (elem_col & ELEM_PER_ATOMIC_MASK);
    }

    inline __device__ static int
    output_byte_offset(int token, int64_t row_stride, int col_base, int col)
    {
        return static_cast<int>((static_cast<int64_t>(token) * row_stride +
                                 col_base + col) *
                                static_cast<int>(sizeof(hip_bfloat16)));
    }

    inline __device__ int acc_local_m(int mi, int elem_in_vec) const
    {
        return wave_id_m * T::M_MFMA_PER_WAVE * T::MMA_M +
               mi * T::MMA_M + (lane_id / T::MMA_M) * T::VEC_C + elem_in_vec;
    }

    inline __device__ int acc_local_col(int ni) const
    {
        return wave_id_n * T::N_MFMA_PER_WAVE * T::MMA_N +
               ni * T::MMA_N + (lane_id % T::MMA_N);
    }

    inline __device__ int atomic_local_m(int mr) const
    {
        return mr * CSHUFFLE_MLANE + tid / CSHUFFLE_NLANE;
    }

    inline __device__ int atomic_col0() const
    {
        return (tid % CSHUFFLE_NLANE) * ELEM_PER_ATOMIC;
    }
};

template<typename T>
inline __device__ OpusMoeStage2A8W4CShuffleLayout<T>
opus_moe_stage2_a8w4_layout_c(int wave_id_m, int wave_id_n)
{
    const int tid = static_cast<int>(opus::thread_id_x());
    return {tid, tid % opus::get_warp_size(), wave_id_m, wave_id_n};
}

#endif // __HIP_DEVICE_COMPILE__
