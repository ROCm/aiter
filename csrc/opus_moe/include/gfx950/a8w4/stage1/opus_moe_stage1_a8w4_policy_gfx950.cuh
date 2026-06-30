// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_stage1_a8w4_traits_gfx950.cuh"

#include "opus/opus.hpp"

#include <hip/hip_runtime.h>

namespace opus_moe
{
namespace stage1_a8w4
{

#ifdef __HIP_DEVICE_COMPILE__

// ============================================================================
// Layout helpers: operand coordinates, scale offsets, and output coordinates.
// Keep this section side-effect free. It mirrors stage2's layout helpers and
// opus_gemm's make_layout_* convention, but stays local to this pipeline file.
// ============================================================================

typedef uint32_t stage1_u32x4_t __attribute__((ext_vector_type(4)));
typedef uint32_t stage1_u32x8_t __attribute__((ext_vector_type(8)));

template<typename Traits>
inline __device__ auto make_layout_ga(int lane_id, int wave_id_m)
{
    (void)wave_id_m;

    constexpr auto block_shape = opus::make_tuple(
        opus::number<Traits::M_MFMA_PER_WAVE>{},
        opus::number<Traits::MMA_M>{},
        opus::number<Traits::THREADS_K>{},
        opus::number<Traits::BYTES_PER_VEC>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<Traits::BYTES_PER_VEC>(
        block_shape,
        opus::unfold_x_stride(
            block_dim,
            block_shape,
            opus::tuple{opus::number<Traits::H>{}, opus::number<1>{}}),
        opus::unfold_p_coord(
            block_dim,
            opus::tuple{lane_id % Traits::MMA_M,
                        lane_id / Traits::MMA_M}));
}

template<typename Traits>
inline __device__ auto make_layout_gb(int lane_id, int wave_id_n)
{
    (void)wave_id_n;

    constexpr auto block_shape = opus::make_tuple(
        opus::number<Traits::N_MFMA_PER_WAVE>{},
        opus::number<Traits::THREADS_K>{},
        opus::number<Traits::MMA_N>{},
        opus::number<Traits::BYTES_PER_VEC>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<Traits::BYTES_PER_VEC>(
        block_shape,
        opus::unfold_x_stride(
            block_dim,
            block_shape,
            opus::tuple{
                opus::number<Traits::MMA_N * Traits::BYTES_PER_VEC>{},
                opus::number<1>{}}),
        opus::unfold_p_coord(
            block_dim,
            opus::tuple{lane_id / Traits::MMA_N,
                        lane_id % Traits::MMA_N}));
}

template<typename Traits>
inline __device__ constexpr auto make_layout_scale_word()
{
    constexpr auto block_shape = opus::make_tuple(
        opus::number<1>{},
        opus::number<1>{},
        opus::number<Traits::THREADS_K>{},
        opus::number<Traits::MMA_M>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}));

    return opus::make_layout<-1>(
        block_shape,
        opus::unfold_x_stride(
            block_dim,
            block_shape,
            opus::tuple{opus::number<Traits::SCALE_LAYOUT_STRIDE_N0>{},
                        opus::number<Traits::SCALE_LAYOUT_STRIDE_K0>{},
                        opus::number<Traits::MMA_M>{},
                        opus::number<1>{}}));
}

template<typename Traits>
inline __device__ int a_local_m(int ga_offset)
{
    return ga_offset / Traits::H;
}

template<typename Traits>
inline __device__ int a_k_byte(int ga_offset)
{
    return ga_offset - a_local_m<Traits>(ga_offset) * Traits::H;
}

template<typename Traits>
struct AOperandCoord
{
    int local_m;
    int k_byte;
    int lane_div_16;
    int lane_mod_16;
};

template<typename Traits>
inline __device__ AOperandCoord<Traits> make_a_operand_coord(int ga_offset)
{
    const int local_m = a_local_m<Traits>(ga_offset);
    const int k_byte = ga_offset - local_m * Traits::H;
    return {local_m,
            k_byte,
            k_byte / Traits::BYTES_PER_VEC,
            local_m & (Traits::MMA_M - 1)};
}

template<typename Traits>
inline __device__ int b_lane_k(int gb_offset)
{
    return gb_offset / (Traits::MMA_N * Traits::BYTES_PER_VEC);
}

template<typename Traits>
inline __device__ int b_lane_n(int gb_offset)
{
    return (gb_offset / Traits::BYTES_PER_VEC) & (Traits::MMA_N - 1);
}

template<typename Traits, typename LayoutA>
inline __device__ void collect_a_offsets(
    const LayoutA& u_ga,
    int (&ga)[Traits::M_MFMA_PER_WAVE])
{
    using opus::operator""_I;

    ga[0] = static_cast<int>(u_ga(0_I, 0_I));
    if constexpr(Traits::M_MFMA_PER_WAVE > 1)
        ga[1] = static_cast<int>(u_ga(1_I, 0_I));
    if constexpr(Traits::M_MFMA_PER_WAVE > 2)
        ga[2] = static_cast<int>(u_ga(2_I, 0_I));
    if constexpr(Traits::M_MFMA_PER_WAVE > 3)
        ga[3] = static_cast<int>(u_ga(3_I, 0_I));
    if constexpr(Traits::M_MFMA_PER_WAVE > 4)
        ga[4] = static_cast<int>(u_ga(4_I, 0_I));
    if constexpr(Traits::M_MFMA_PER_WAVE > 5)
        ga[5] = static_cast<int>(u_ga(5_I, 0_I));
    if constexpr(Traits::M_MFMA_PER_WAVE > 6)
        ga[6] = static_cast<int>(u_ga(6_I, 0_I));
    if constexpr(Traits::M_MFMA_PER_WAVE > 7)
        ga[7] = static_cast<int>(u_ga(7_I, 0_I));
}

template<typename Traits, typename LayoutA>
inline __device__ void collect_a_coords(
    const LayoutA& u_ga,
    AOperandCoord<Traits> (&coord)[Traits::M_MFMA_PER_WAVE])
{
    int ga[Traits::M_MFMA_PER_WAVE];
    collect_a_offsets<Traits>(u_ga, ga);

    #pragma unroll
    for(int mi = 0; mi < Traits::M_MFMA_PER_WAVE; ++mi)
    {
        coord[mi] = make_a_operand_coord<Traits>(ga[mi]);
    }
}

template<typename Traits>
inline __device__ int64_t a_payload_base_byte_offset(int token,
                                                     int ga_offset,
                                                     int64_t stride_hidden_t)
{
    return static_cast<int64_t>(token) * stride_hidden_t +
           a_k_byte<Traits>(ga_offset);
}

template<typename Traits>
inline __device__ constexpr int w1_payload_k_step_stride_bytes()
{
    constexpr int kW1NLane = Traits::MMA_N;
    constexpr int kW1KLane = 64 / kW1NLane;
    constexpr int kW1KPack = Traits::BYTES_PER_VEC;
    return kW1KLane * kW1NLane * kW1KPack;
}

template<typename Traits>
inline __device__ int64_t w1_payload_group_base_byte_offset(
    int expert_id,
    int output_col_base,
    int gate_up,
    int gb_offset,
    int64_t stride_w1_e)
{
    const int w1_n0 = output_col_base / Traits::MMA_N;
    return static_cast<int64_t>(expert_id) * stride_w1_e +
           (static_cast<int64_t>(w1_n0) * 2 + gate_up) *
               Traits::MFMA_K_STEPS *
               w1_payload_k_step_stride_bytes<Traits>() +
           gb_offset;
}

template<typename Traits>
inline __device__ int scale_word_byte_offset(int mn_pack,
                                             int k_tile_256,
                                             int lane_div_16,
                                             int lane_mod_16)
{
    constexpr auto u = make_layout_scale_word<Traits>();
    const int dword_offset =
        static_cast<int>(u(mn_pack, k_tile_256, lane_div_16, lane_mod_16));
    return dword_offset * static_cast<int>(sizeof(uint32_t));
}

template<typename Traits>
inline __device__ int a_scale_base_byte_offset(int route_base,
                                               int m_scale_pack,
                                               int gb_offset)
{
    const int mn_pack =
        route_base / (Traits::SCALE_MN_PACK * Traits::MMA_M) +
        m_scale_pack;
    return scale_word_byte_offset<Traits>(
        mn_pack, 0, b_lane_k<Traits>(gb_offset), b_lane_n<Traits>(gb_offset));
}

template<typename Traits>
inline __device__ int w1_scale_base_byte_offset(int expert_id,
                                                int output_col_base,
                                                int gb_offset)
{
    const int logical_row_pair_base =
        expert_id * Traits::GATE_UP_LOGICAL_DIM +
        (output_col_base / Traits::MMA_N) *
            (Traits::SCALE_MN_PACK * Traits::MMA_M);
    const int mn_pack =
        logical_row_pair_base / (Traits::SCALE_MN_PACK * Traits::MMA_M);
    return scale_word_byte_offset<Traits>(
        mn_pack, 0, b_lane_k<Traits>(gb_offset), b_lane_n<Traits>(gb_offset));
}

inline __device__ int mx_scale_shuffle_idx(int scale_n_pad,
                                           int row,
                                           int scale_col)
{
    return (row / 32 * scale_n_pad) * 32 + (scale_col / 8) * 256 +
           (scale_col % 4) * 64 + (row % 16) * 4 +
           (scale_col % 8) / 4 * 2 + (row % 32) / 16;
}

template<typename Traits>
inline __device__ int gate_up_smem_offset(int smem_row,
                                          int group,
                                          int gate_up,
                                          int local_col)
{
    return smem_row * Traits::EPILOGUE_SMEM_COLS +
           group * Traits::SCALE_GROUP_LOGICAL_K * 2 +
           gate_up * Traits::SCALE_GROUP_LOGICAL_K + local_col;
}

template<typename Traits>
inline __device__ int activated_smem_offset(int smem_row,
                                            int group,
                                            int local_col)
{
    if constexpr(Traits::GATE_UP_GROUP_SPLIT && Traits::B_M == 64 &&
                 Traits::B_N == 256)
    {
        local_col ^= (smem_row & 7) << 2;
    }
    return smem_row *
               (Traits::OUTPUT_SCALE_GROUPS_PER_TILE *
                Traits::SCALE_GROUP_LOGICAL_K) +
           group * Traits::SCALE_GROUP_LOGICAL_K + local_col;
}

template<typename Traits>
inline __device__ int output_byte_offset(int token,
                                         int slot,
                                         int effective_col,
                                         int64_t stride_out_t,
                                         int64_t stride_out_k)
{
    return static_cast<int>(static_cast<int64_t>(token) * stride_out_t +
                            static_cast<int64_t>(slot) * stride_out_k +
                            effective_col);
}

template<typename Traits>
inline __device__ int output_scale_byte_offset(int route_row,
                                               int scale_col,
                                               int64_t stride_scale_route)
{
    return mx_scale_shuffle_idx(
        static_cast<int>(stride_scale_route), route_row, scale_col);
}

// ============================================================================
// Prologue: block mapping and route metadata. This stays small and explicit.
// ============================================================================

template<typename Traits>
struct Tile
{
    int tid;
    int route_base;
    int col_tile;
    int out_col_base;
    int valid_rows;
    int expert_id;
};

template<typename Traits>
inline __device__ Tile<Traits> make_tile(const OpusMoeStage1A8W4Kargs& kargs)
{
    const int route_tile = static_cast<int>(blockIdx.y);
    const int col_tile = static_cast<int>(blockIdx.x);
    constexpr int route_subtiles = Traits::SORT_BLOCK_M / Traits::B_M;
    const int sorted_block = route_tile / route_subtiles;
    const int route_subtile = route_tile - sorted_block * route_subtiles;
    const int expert_id = kargs.sorted_expert_ids[sorted_block];
    return {static_cast<int>(threadIdx.x),
            sorted_block * Traits::SORT_BLOCK_M +
                route_subtile * Traits::B_M,
            col_tile,
            col_tile * Traits::OUTPUT_COLS_PER_TILE,
            kargs.num_valid_ids[0],
            expert_id};
}

inline __device__ int token_id(int32_t packed)
{
    return packed & 0x00ffffff;
}

inline __device__ int topk_slot(int32_t packed)
{
    return static_cast<uint32_t>(packed) >> 24;
}

template<typename Traits>
inline __device__ bool route_is_valid(const OpusMoeStage1A8W4Kargs& kargs,
                                      int route_row,
                                      int& token,
                                      int& slot)
{
    token = 0;
    slot = 0;
    if(route_row >= kargs.num_valid_ids[0])
        return false;

    const int32_t packed = kargs.sorted_token_ids[route_row];
    token = token_id(packed);
    slot = topk_slot(packed);
    return token < kargs.token_num && slot < kargs.topk;
}

template<typename Traits>
inline __device__ bool route_is_valid_fast(const OpusMoeStage1A8W4Kargs& kargs,
                                           bool tile_full_valid,
                                           int route_row,
                                           int& token,
                                           int& slot)
{
    if(!tile_full_valid)
        return route_is_valid<Traits>(kargs, route_row, token, slot);

    const int32_t packed = kargs.sorted_token_ids[route_row];
    token = token_id(packed);
    slot = topk_slot(packed);
    return token < kargs.token_num && slot < kargs.topk;
}

template<typename Traits>
inline __device__ bool load_route_metadata_to_smem(
    const OpusMoeStage1A8W4Kargs& kargs,
    const Tile<Traits>& tile,
    int* __restrict__ smem_token,
    int* __restrict__ smem_slot,
    uint8_t* __restrict__ smem_route_valid)
{
    const bool tile_full_valid =
        tile.route_base + Traits::B_M <= tile.valid_rows;
    int has_route = 0;

    #pragma unroll 1
    for(int local_m = tile.tid; local_m < Traits::B_M;
        local_m += Traits::BLOCK_SIZE)
    {
        int token = 0;
        int slot = 0;
        const bool valid = route_is_valid_fast<Traits>(
            kargs, tile_full_valid, tile.route_base + local_m, token, slot);
        smem_token[local_m] = token;
        smem_slot[local_m] = slot;
        smem_route_valid[local_m] = valid ? 1 : 0;
        has_route |= valid ? 1 : 0;
    }

    return __syncthreads_or(has_route) != 0;
}

inline __device__ bool route_from_smem(int local_m,
                                       const int* __restrict__ smem_token,
                                       const int* __restrict__ smem_slot,
                                       const uint8_t* __restrict__ smem_route_valid,
                                       int& token,
                                       int& slot)
{
    token = smem_token[local_m];
    slot = smem_slot[local_m];
    return smem_route_valid[local_m] != 0;
}

#endif // __HIP_DEVICE_COMPILE__
} // namespace stage1_a8w4
} // namespace opus_moe
