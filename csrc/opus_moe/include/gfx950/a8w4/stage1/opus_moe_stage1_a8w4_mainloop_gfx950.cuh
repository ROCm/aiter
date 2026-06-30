// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_stage1_a8w4_policy_gfx950.cuh"

namespace opus_moe
{
namespace stage1_a8w4
{

#ifdef __HIP_DEVICE_COMPILE__

// ============================================================================
// Mainloop: hidden/W1/scale loads and scaled MFMA accumulation.
// ============================================================================

template<typename Reg>
inline __device__ void pack_a_mfma_reg(stage1_u32x4_t lo,
                                       stage1_u32x4_t hi,
                                       Reg& reg)
{
    stage1_u32x8_t packed{};
    packed[0] = lo[0];
    packed[1] = lo[1];
    packed[2] = lo[2];
    packed[3] = lo[3];
    packed[4] = hi[0];
    packed[5] = hi[1];
    packed[6] = hi[2];
    packed[7] = hi[3];
    reg = __builtin_bit_cast(opus::remove_cvref_t<Reg>, packed);
}

template<typename Reg>
inline __device__ void unpack_b_mfma_reg(stage1_u32x4_t value, Reg& reg)
{
    stage1_u32x8_t packed{};
    packed[0] = value[0];
    packed[1] = value[1];
    packed[2] = value[2];
    packed[3] = value[3];
    reg = __builtin_bit_cast(opus::remove_cvref_t<Reg>, packed);
}

template<typename Mma, typename AReg, typename BReg, typename CReg>
inline __device__ auto scaled_mfma_select(Mma& mma,
                                          const AReg& a,
                                          const BReg& b,
                                          const CReg& c,
                                          int scale_a,
                                          int scale_b,
                                          int selector_a,
                                          int selector_b)
    -> typename Mma::vtype_c
{
    switch(selector_a * 4 + selector_b)
    {
    case 0:
        return mma(a, b, c, scale_a, scale_b, opus::number<0>{}, opus::number<0>{});
    case 1:
        return mma(a, b, c, scale_a, scale_b, opus::number<0>{}, opus::number<1>{});
    case 2:
        return mma(a, b, c, scale_a, scale_b, opus::number<0>{}, opus::number<2>{});
    case 3:
        return mma(a, b, c, scale_a, scale_b, opus::number<0>{}, opus::number<3>{});
    case 4:
        return mma(a, b, c, scale_a, scale_b, opus::number<1>{}, opus::number<0>{});
    case 5:
        return mma(a, b, c, scale_a, scale_b, opus::number<1>{}, opus::number<1>{});
    case 6:
        return mma(a, b, c, scale_a, scale_b, opus::number<1>{}, opus::number<2>{});
    case 7:
        return mma(a, b, c, scale_a, scale_b, opus::number<1>{}, opus::number<3>{});
    case 8:
        return mma(a, b, c, scale_a, scale_b, opus::number<2>{}, opus::number<0>{});
    case 9:
        return mma(a, b, c, scale_a, scale_b, opus::number<2>{}, opus::number<1>{});
    case 10:
        return mma(a, b, c, scale_a, scale_b, opus::number<2>{}, opus::number<2>{});
    case 11:
        return mma(a, b, c, scale_a, scale_b, opus::number<2>{}, opus::number<3>{});
    case 12:
        return mma(a, b, c, scale_a, scale_b, opus::number<3>{}, opus::number<0>{});
    case 13:
        return mma(a, b, c, scale_a, scale_b, opus::number<3>{}, opus::number<1>{});
    case 14:
        return mma(a, b, c, scale_a, scale_b, opus::number<3>{}, opus::number<2>{});
    default:
        return mma(a, b, c, scale_a, scale_b, opus::number<3>{}, opus::number<3>{});
    }
}

template<typename Traits>
inline __device__ constexpr int a_reg_lds_stage_bytes()
{
    return Traits::M_MFMA_PER_WAVE * Traits::WAVE_SIZE * 2 *
           Traits::BYTES_PER_VEC;
}

template<typename Traits>
inline __device__ int a_reg_lds_offset(int pair_buffer,
                                       int kk,
                                       int mi,
                                       int lane_id,
                                       int half)
{
    constexpr int kStageBytes = a_reg_lds_stage_bytes<Traits>();
    const int stage_base =
        (pair_buffer * Traits::SCALE_K_PACK + kk) * kStageBytes;
    if constexpr(Traits::B_M >= 128)
    {
        return stage_base +
               ((mi * 2 + half) * Traits::WAVE_SIZE + lane_id) *
                   Traits::BYTES_PER_VEC;
    }
    else
    {
        return stage_base +
               ((mi * Traits::WAVE_SIZE + lane_id) * 2 + half) *
                   Traits::BYTES_PER_VEC;
    }
}

template<typename Traits>
inline __device__ void stage_a_reg_kstep_to_lds(
    const OpusMoeStage1A8W4Kargs& kargs,
    opus::gmem<uint8_t>& hidden_gmem,
    bool route_valid,
    int64_t a_payload_base,
    int k_step,
    int pair_buffer,
    int kk,
    int mi,
    int lane_id,
    uint8_t* __restrict__ smem_a_reg)
{
    if(!route_valid)
        return;

    const int64_t a_offset =
        a_payload_base + static_cast<int64_t>(k_step) * Traits::MMA_K;

    const int smem_lo =
        a_reg_lds_offset<Traits>(pair_buffer, kk, mi, lane_id, 0);
    const int smem_hi =
        a_reg_lds_offset<Traits>(pair_buffer, kk, mi, lane_id, 1);
    if constexpr(Traits::B_M >= 128)
    {
        const int a_offset_i32 = static_cast<int>(a_offset);
        hidden_gmem.template async_load<Traits::BYTES_PER_VEC>(
            smem_a_reg + smem_lo, a_offset_i32);
        hidden_gmem.template async_load<Traits::BYTES_PER_VEC>(
            smem_a_reg + smem_hi,
            a_offset_i32 + 4 * Traits::BYTES_PER_VEC);
    }
    else
    {
        const auto a_lo = *reinterpret_cast<const stage1_u32x4_t*>(
            kargs.hidden_fp8 + a_offset);
        const auto a_hi = *reinterpret_cast<const stage1_u32x4_t*>(
            kargs.hidden_fp8 + a_offset + 4 * Traits::BYTES_PER_VEC);
        *reinterpret_cast<stage1_u32x4_t*>(smem_a_reg + smem_lo) = a_lo;
        *reinterpret_cast<stage1_u32x4_t*>(smem_a_reg + smem_hi) = a_hi;
    }
}

template<typename Traits>
inline __device__ void stage_a_reg_kpair_to_lds(
    const OpusMoeStage1A8W4Kargs& kargs,
    int wave_id,
    int lane_id,
    int k_pair,
    int pair_buffer,
    const bool (&route_valid)[Traits::M_MFMA_PER_WAVE],
    const int64_t (&a_payload_base)[Traits::M_MFMA_PER_WAVE],
    uint8_t* __restrict__ smem_a_reg)
{
    constexpr int kProducerWaves = Traits::BLOCK_SIZE / Traits::WAVE_SIZE;
    auto hidden_gmem = opus::gmem<uint8_t>(kargs.hidden_fp8);
    #pragma unroll
    for(int mi = wave_id; mi < Traits::M_MFMA_PER_WAVE;
        mi += kProducerWaves)
    {
        #pragma unroll
        for(int kk = 0; kk < Traits::SCALE_K_PACK; ++kk)
        {
            const int k_step = k_pair + kk;
            stage_a_reg_kstep_to_lds<Traits>(
                kargs,
                hidden_gmem,
                route_valid[mi],
                a_payload_base[mi],
                k_step,
                pair_buffer,
                kk,
                mi,
                lane_id,
                smem_a_reg);
        }
    }
}

template<typename Traits, int MiBegin>
inline __device__ void stage_a_reg_kpair_4mi_to_lds(
    const OpusMoeStage1A8W4Kargs& kargs,
    int wave_id,
    int lane_id,
    int k_pair,
    int pair_buffer,
    const bool (&route_valid)[Traits::M_MFMA_PER_WAVE],
    const int64_t (&a_payload_base)[Traits::M_MFMA_PER_WAVE],
    uint8_t* __restrict__ smem_a_reg)
{
    constexpr int kProducerWaves = Traits::BLOCK_SIZE / Traits::WAVE_SIZE;
    static_assert(kProducerWaves == 4);
    static_assert(MiBegin >= 0);
    static_assert(MiBegin + kProducerWaves <= Traits::M_MFMA_PER_WAVE);

    auto hidden_gmem = opus::gmem<uint8_t>(kargs.hidden_fp8);
    const int mi = MiBegin + wave_id;
    #pragma unroll
    for(int kk = 0; kk < Traits::SCALE_K_PACK; ++kk)
    {
        const int k_step = k_pair + kk;
        stage_a_reg_kstep_to_lds<Traits>(
            kargs,
            hidden_gmem,
            route_valid[mi],
            a_payload_base[mi],
            k_step,
            pair_buffer,
            kk,
            mi,
            lane_id,
            smem_a_reg);
    }
}

template<typename Traits>
inline __device__ void wait_a_reg_kpair_to_lds()
{
    if constexpr(Traits::B_M >= 128)
        opus::s_waitcnt_vmcnt(opus::number<0>{});
}

template<typename Traits, typename Mma>
inline __device__ void load_a_mfma_reg_from_lds(
    const uint8_t* __restrict__ smem_a_reg,
    typename Mma::mfma_type::vtype_a& ra,
    int pair_buffer,
    int kk,
    int mi,
    int lane_id)
{
    const int smem_lo =
        a_reg_lds_offset<Traits>(pair_buffer, kk, mi, lane_id, 0);
    const int smem_hi =
        a_reg_lds_offset<Traits>(pair_buffer, kk, mi, lane_id, 1);
    const auto a_lo = *reinterpret_cast<const stage1_u32x4_t*>(
        smem_a_reg + smem_lo);
    const auto a_hi = *reinterpret_cast<const stage1_u32x4_t*>(
        smem_a_reg + smem_hi);
    pack_a_mfma_reg(a_lo, a_hi, ra);
}

template<typename Traits, typename Mma>
inline __device__ void accumulate_loaded_a_with_b(
    Mma& mma,
    const typename Mma::mfma_type::vtype_a& ra,
    const typename Mma::mfma_type::vtype_b& rb,
    typename Mma::vtype_c& rc,
    int mi_idx,
    bool route_valid,
    int a_scale,
    int k_step,
    int selector_b,
    int b_scale)
{
    if(!route_valid)
        return;

    const int selector_a =
        (k_step & 1) * Traits::SCALE_MN_PACK +
        (mi_idx & (Traits::SCALE_MN_PACK - 1));
    rc = scaled_mfma_select(
        mma, ra, rb, rc, a_scale, b_scale, selector_a, selector_b);
}

template<typename Traits, typename Mma, typename LayoutA, typename LayoutB>
inline __device__ void mainloop(
    Mma& mma,
    const LayoutA& u_ga,
    const LayoutB& u_gb,
    const OpusMoeStage1A8W4Kargs& kargs,
    const Tile<Traits>& tile,
    int wave_id,
    int lane_id,
    const int* __restrict__ smem_token,
    const int* __restrict__ smem_slot,
    const uint8_t* __restrict__ smem_route_valid,
    uint8_t* __restrict__ smem_a_reg,
    typename Mma::vtype_c (&rc)[Traits::M_MFMA_PER_WAVE]
                              [Traits::OUTPUT_SCALE_GROUPS_PER_TILE])
{
    using opus::operator""_I;

    static_assert(Traits::B_N == 384 || Traits::B_N == 256);
    static_assert(Traits::SORT_BLOCK_M % Traits::B_M == 0);
    static_assert(Traits::OUTPUT_SCALE_GROUPS_PER_TILE == 6 ||
                  Traits::OUTPUT_SCALE_GROUPS_PER_TILE == 4);
    static_assert(Traits::SCALE_GROUP_LOGICAL_K == 32);
    static_assert(Traits::MMA_M == 16);
    static_assert(Traits::MMA_N == 16);
    static_assert(Traits::MMA_K == 128);
    static_assert(Traits::M_MFMA_PER_WAVE == Traits::B_M / Traits::MMA_M);
    static_assert(Traits::M_MFMA_PER_WAVE <= 8);
    static_assert(a_reg_lds_stage_bytes<Traits>() == Traits::B_M * Traits::MMA_K);
    static_assert(2 * Traits::SCALE_K_PACK * a_reg_lds_stage_bytes<Traits>() <=
                  Traits::EPILOGUE_SMEM_ROWS * Traits::EPILOGUE_SMEM_COLS *
                      static_cast<int>(sizeof(float)));

    using V_A = typename Mma::mfma_type::vtype_a;
    using V_B = typename Mma::mfma_type::vtype_b;

    const int out_half = wave_id / 2;
    const int gate_up = wave_id & 1;
    if(tile.route_base >= tile.valid_rows || tile.expert_id < 0 ||
       tile.expert_id >= Traits::EXPERTS)
        return;

    int ga[Traits::M_MFMA_PER_WAVE];
    collect_a_offsets<Traits>(u_ga, ga);
    const int gb = static_cast<int>(u_gb(0_I, 0_I));

    int token[Traits::M_MFMA_PER_WAVE];
    int slot[Traits::M_MFMA_PER_WAVE];
    bool route_valid[Traits::M_MFMA_PER_WAVE];
    int64_t a_payload_base[Traits::M_MFMA_PER_WAVE];
    #pragma unroll
    for(int mi = 0; mi < Traits::M_MFMA_PER_WAVE; ++mi)
    {
        route_valid[mi] = route_from_smem(
            a_local_m<Traits>(ga[mi]),
            smem_token,
            smem_slot,
            smem_route_valid,
            token[mi],
            slot[mi]);
        (void)slot[mi];
        a_payload_base[mi] = a_payload_base_byte_offset<Traits>(
            token[mi], ga[mi], kargs.stride_hidden_t);
    }

    constexpr int kW1KStepBytes =
        w1_payload_k_step_stride_bytes<Traits>();
    int a_scale_base[Traits::M_SCALE_PACKS];
    #pragma unroll
    for(int mp = 0; mp < Traits::M_SCALE_PACKS; ++mp)
    {
        a_scale_base[mp] =
            a_scale_base_byte_offset<Traits>(tile.route_base, mp, gb);
    }

    int64_t b_payload_base[Traits::OUTPUT_SCALE_GROUPS_PER_TILE];
    int b_scale_base[Traits::OUTPUT_SCALE_GROUPS_PER_TILE];
    #pragma unroll
    for(int group = 0; group < Traits::OUTPUT_SCALE_GROUPS_PER_TILE; ++group)
    {
        const int group_col_base =
            tile.out_col_base + group * Traits::SCALE_GROUP_LOGICAL_K;
        const int output_col_base =
            group_col_base + out_half * Traits::MMA_N;
        b_payload_base[group] =
            w1_payload_group_base_byte_offset<Traits>(
                tile.expert_id,
                output_col_base,
                gate_up,
                gb,
                kargs.stride_w1_e);
        b_scale_base[group] =
            w1_scale_base_byte_offset<Traits>(
                tile.expert_id, output_col_base, gb);
    }

    stage_a_reg_kpair_to_lds<Traits>(
        kargs,
        wave_id,
        lane_id,
        0,
        0,
        route_valid,
        a_payload_base,
        smem_a_reg);
    wait_a_reg_kpair_to_lds<Traits>();
    __syncthreads();

    for(int k_pair = 0; k_pair < Traits::MFMA_K_STEPS;
        k_pair += Traits::SCALE_K_PACK)
    {
        const int k_pair_idx = k_pair / Traits::SCALE_K_PACK;
        const int pair_buffer = k_pair_idx & 1;
        const int next_k_pair = k_pair + Traits::SCALE_K_PACK;
        const int a_scale_step_offset =
            k_pair_idx * Traits::SCALE_LAYOUT_STRIDE_K0 *
            static_cast<int>(sizeof(uint32_t));
        int a_scale[Traits::M_SCALE_PACKS];
        #pragma unroll
        for(int mp = 0; mp < Traits::M_SCALE_PACKS; ++mp)
        {
            a_scale[mp] = static_cast<int>(
                *reinterpret_cast<const uint32_t*>(
                    kargs.hidden_scale_e8m0 + a_scale_base[mp] +
                    a_scale_step_offset));
        }

        int b_scale[Traits::OUTPUT_SCALE_GROUPS_PER_TILE];
        #pragma unroll
        for(int group = 0; group < Traits::OUTPUT_SCALE_GROUPS_PER_TILE;
            ++group)
        {
            b_scale[group] = static_cast<int>(
                *reinterpret_cast<const uint32_t*>(
                    kargs.w1_scale_e8m0 + b_scale_base[group] +
                    a_scale_step_offset));
        }

        stage1_u32x4_t b_raw_kk0[Traits::OUTPUT_SCALE_GROUPS_PER_TILE];
        #pragma unroll
        for(int group = 0; group < Traits::OUTPUT_SCALE_GROUPS_PER_TILE;
            ++group)
        {
            const int64_t b_offset = b_payload_base[group] +
                                     static_cast<int64_t>(k_pair) *
                                         kW1KStepBytes;
            b_raw_kk0[group] = *reinterpret_cast<const stage1_u32x4_t*>(
                kargs.w1_fp4 + b_offset);
        }

        if(next_k_pair < Traits::MFMA_K_STEPS)
        {
            stage_a_reg_kpair_to_lds<Traits>(
                kargs,
                wave_id,
                lane_id,
                next_k_pair,
                pair_buffer ^ 1,
                route_valid,
                a_payload_base,
                smem_a_reg);
        }

        #pragma unroll
        for(int kk = 0; kk < Traits::SCALE_K_PACK; ++kk)
        {
            const int k_step = k_pair + kk;
            stage1_u32x4_t b_raw[Traits::OUTPUT_SCALE_GROUPS_PER_TILE];
            #pragma unroll
            for(int group = 0; group < Traits::OUTPUT_SCALE_GROUPS_PER_TILE;
                ++group)
            {
                if(kk == 0)
                {
                    b_raw[group] = b_raw_kk0[group];
                }
                else
                {
                    const int64_t b_offset =
                        b_payload_base[group] +
                        static_cast<int64_t>(k_step) * kW1KStepBytes;
                    b_raw[group] =
                        *reinterpret_cast<const stage1_u32x4_t*>(
                            kargs.w1_fp4 + b_offset);
                }
            }

            V_A ra[Traits::M_MFMA_PER_WAVE]{};
            #pragma unroll
            for(int mi = 0; mi < Traits::M_MFMA_PER_WAVE; ++mi)
            {
                if(route_valid[mi])
                {
                    load_a_mfma_reg_from_lds<Traits, Mma>(
                        smem_a_reg, ra[mi], pair_buffer, kk, mi, lane_id);
                }
            }
            const int selector_b =
                (k_step & 1) * Traits::SCALE_MN_PACK + gate_up;

            #pragma unroll
            for(int group = 0; group < Traits::OUTPUT_SCALE_GROUPS_PER_TILE;
                ++group)
            {
                V_B rb{};
                unpack_b_mfma_reg(b_raw[group], rb);
                #pragma unroll
                for(int mi = 0; mi < Traits::M_MFMA_PER_WAVE; ++mi)
                {
                    accumulate_loaded_a_with_b<Traits, Mma>(
                        mma,
                        ra[mi],
                        rb,
                        rc[mi][group],
                        mi,
                        route_valid[mi],
                        a_scale[mi / Traits::SCALE_MN_PACK],
                        k_step,
                        selector_b,
                        b_scale[group]);
                }
            }
        }

        if(next_k_pair < Traits::MFMA_K_STEPS)
        {
            wait_a_reg_kpair_to_lds<Traits>();
            __syncthreads();
        }
    }
}

template<typename Traits, typename Mma, typename LayoutA, typename LayoutB>
inline __device__ void mainloop_gate_up_group_split(
    Mma& mma,
    const LayoutA& u_ga,
    const LayoutB& u_gb,
    const OpusMoeStage1A8W4Kargs& kargs,
    const Tile<Traits>& tile,
    int wave_id,
    int lane_id,
    const int* __restrict__ smem_token,
    const int* __restrict__ smem_slot,
    const uint8_t* __restrict__ smem_route_valid,
    uint8_t* __restrict__ smem_a_reg,
    typename Mma::vtype_c (&rc)[Traits::M_MFMA_PER_WAVE]
                              [Traits::OUTPUT_SCALE_GROUPS_PER_TILE])
{
    using opus::operator""_I;

    static_assert(Traits::GATE_UP_GROUP_SPLIT);
    static_assert(Traits::B_N == 384 || Traits::B_N == 256);
    static_assert(Traits::SORT_BLOCK_M % Traits::B_M == 0);
    static_assert(Traits::OUTPUT_SCALE_GROUPS_PER_TILE == 6 ||
                  Traits::OUTPUT_SCALE_GROUPS_PER_TILE == 4);
    static_assert(Traits::GATE_UP_GROUP_SPLIT_GROUPS * 2 ==
                  Traits::OUTPUT_SCALE_GROUPS_PER_TILE);
    static_assert(Traits::SCALE_GROUP_LOGICAL_K == 32);
    static_assert(Traits::MMA_M == 16);
    static_assert(Traits::MMA_N == 16);
    static_assert(Traits::MMA_K == 128);
    static_assert(Traits::M_MFMA_PER_WAVE == Traits::B_M / Traits::MMA_M);
    static_assert(Traits::M_MFMA_PER_WAVE <= 8);
    static_assert(a_reg_lds_stage_bytes<Traits>() == Traits::B_M * Traits::MMA_K);
    static_assert(2 * Traits::SCALE_K_PACK * a_reg_lds_stage_bytes<Traits>() <=
                  Traits::EPILOGUE_SMEM_ROWS * Traits::EPILOGUE_SMEM_COLS *
                      static_cast<int>(sizeof(float)));

    using V_A = typename Mma::mfma_type::vtype_a;
    using V_B = typename Mma::mfma_type::vtype_b;

    constexpr int kGroupsPerWave = Traits::GATE_UP_GROUP_SPLIT_GROUPS;
    const int out_half = wave_id & 1;
    const int group_block = wave_id / 2;
    if(tile.route_base >= tile.valid_rows || tile.expert_id < 0 ||
       tile.expert_id >= Traits::EXPERTS || group_block >= 2)
        return;

    AOperandCoord<Traits> a_coord[Traits::M_MFMA_PER_WAVE];
    collect_a_coords<Traits>(u_ga, a_coord);
    const int gb = static_cast<int>(u_gb(0_I, 0_I));

    int token[Traits::M_MFMA_PER_WAVE];
    int slot[Traits::M_MFMA_PER_WAVE];
    bool route_valid[Traits::M_MFMA_PER_WAVE];
    int64_t a_payload_base[Traits::M_MFMA_PER_WAVE];
    #pragma unroll
    for(int mi = 0; mi < Traits::M_MFMA_PER_WAVE; ++mi)
    {
        route_valid[mi] = route_from_smem(
            a_coord[mi].local_m,
            smem_token,
            smem_slot,
            smem_route_valid,
            token[mi],
            slot[mi]);
        (void)slot[mi];
        a_payload_base[mi] =
            static_cast<int64_t>(token[mi]) * kargs.stride_hidden_t +
            a_coord[mi].k_byte;
    }

    constexpr int kW1KStepBytes = w1_payload_k_step_stride_bytes<Traits>();
    int a_scale_base[Traits::M_SCALE_PACKS];
    #pragma unroll
    for(int mp = 0; mp < Traits::M_SCALE_PACKS; ++mp)
    {
        a_scale_base[mp] =
            a_scale_base_byte_offset<Traits>(tile.route_base, mp, gb);
    }

    int64_t b_payload_base_gate[kGroupsPerWave];
    int64_t b_payload_base_up[kGroupsPerWave];
    int b_scale_base[kGroupsPerWave];
    #pragma unroll
    for(int local_group = 0; local_group < kGroupsPerWave; ++local_group)
    {
        const int group = group_block * kGroupsPerWave + local_group;
        const int group_col_base =
            tile.out_col_base + group * Traits::SCALE_GROUP_LOGICAL_K;
        const int output_col_base =
            group_col_base + out_half * Traits::MMA_N;
        b_payload_base_gate[local_group] =
            w1_payload_group_base_byte_offset<Traits>(
                tile.expert_id, output_col_base, 0, gb, kargs.stride_w1_e);
        b_payload_base_up[local_group] =
            b_payload_base_gate[local_group] +
            Traits::MFMA_K_STEPS * kW1KStepBytes;
        b_scale_base[local_group] =
            w1_scale_base_byte_offset<Traits>(
                tile.expert_id, output_col_base, gb);
    }

    stage_a_reg_kpair_to_lds<Traits>(
        kargs,
        wave_id,
        lane_id,
        0,
        0,
        route_valid,
        a_payload_base,
        smem_a_reg);
    wait_a_reg_kpair_to_lds<Traits>();
    __syncthreads();

    for(int k_pair = 0; k_pair < Traits::MFMA_K_STEPS;
        k_pair += Traits::SCALE_K_PACK)
    {
        const int k_pair_idx = k_pair / Traits::SCALE_K_PACK;
        const int pair_buffer = k_pair_idx & 1;
        const int next_k_pair = k_pair + Traits::SCALE_K_PACK;
        const int a_scale_step_offset =
            k_pair_idx * Traits::SCALE_LAYOUT_STRIDE_K0 *
            static_cast<int>(sizeof(uint32_t));
        int a_scale[Traits::M_SCALE_PACKS];
        #pragma unroll
        for(int mp = 0; mp < Traits::M_SCALE_PACKS; ++mp)
        {
            a_scale[mp] = static_cast<int>(
                *reinterpret_cast<const uint32_t*>(
                    kargs.hidden_scale_e8m0 + a_scale_base[mp] +
                    a_scale_step_offset));
        }

        int b_scale[kGroupsPerWave];
        #pragma unroll
        for(int local_group = 0; local_group < kGroupsPerWave; ++local_group)
        {
            b_scale[local_group] = static_cast<int>(
                *reinterpret_cast<const uint32_t*>(
                    kargs.w1_scale_e8m0 + b_scale_base[local_group] +
                    a_scale_step_offset));
        }

        stage1_u32x4_t b_gate_kk0[kGroupsPerWave];
        stage1_u32x4_t b_up_kk0[kGroupsPerWave];
        #pragma unroll
        for(int local_group = 0; local_group < kGroupsPerWave; ++local_group)
        {
            const int gate_offset = static_cast<int>(
                b_payload_base_gate[local_group] +
                static_cast<int64_t>(k_pair) * kW1KStepBytes);
            const int up_offset = static_cast<int>(
                b_payload_base_up[local_group] +
                static_cast<int64_t>(k_pair) * kW1KStepBytes);
            b_gate_kk0[local_group] =
                *reinterpret_cast<const stage1_u32x4_t*>(
                    kargs.w1_fp4 + gate_offset);
            b_up_kk0[local_group] =
                *reinterpret_cast<const stage1_u32x4_t*>(
                    kargs.w1_fp4 + up_offset);
        }

        if(next_k_pair < Traits::MFMA_K_STEPS)
        {
            if constexpr(Traits::M_MFMA_PER_WAVE >= 8)
            {
                stage_a_reg_kpair_4mi_to_lds<Traits, 0>(
                    kargs,
                    wave_id,
                    lane_id,
                    next_k_pair,
                    pair_buffer ^ 1,
                    route_valid,
                    a_payload_base,
                    smem_a_reg);
            }
            else
            {
                stage_a_reg_kpair_to_lds<Traits>(
                    kargs,
                    wave_id,
                    lane_id,
                    next_k_pair,
                    pair_buffer ^ 1,
                    route_valid,
                    a_payload_base,
                    smem_a_reg);
            }
        }

        #pragma unroll
        for(int kk = 0; kk < Traits::SCALE_K_PACK; ++kk)
        {
            const int k_step = k_pair + kk;
            stage1_u32x4_t b_gate[kGroupsPerWave];
            stage1_u32x4_t b_up[kGroupsPerWave];
            #pragma unroll
            for(int local_group = 0; local_group < kGroupsPerWave; ++local_group)
            {
                if(kk == 0)
                {
                    b_gate[local_group] = b_gate_kk0[local_group];
                    b_up[local_group] = b_up_kk0[local_group];
                }
                else
                {
                    const int gate_offset = static_cast<int>(
                        b_payload_base_gate[local_group] +
                        static_cast<int64_t>(k_step) * kW1KStepBytes);
                    const int up_offset = static_cast<int>(
                        b_payload_base_up[local_group] +
                        static_cast<int64_t>(k_step) * kW1KStepBytes);
                    b_gate[local_group] =
                        *reinterpret_cast<const stage1_u32x4_t*>(
                            kargs.w1_fp4 + gate_offset);
                    b_up[local_group] =
                        *reinterpret_cast<const stage1_u32x4_t*>(
                            kargs.w1_fp4 + up_offset);
                }
            }

            V_A ra[Traits::M_MFMA_PER_WAVE]{};
            #pragma unroll
            for(int mi = 0; mi < Traits::M_MFMA_PER_WAVE; ++mi)
            {
                load_a_mfma_reg_from_lds<Traits, Mma>(
                    smem_a_reg, ra[mi], pair_buffer, kk, mi, lane_id);
            }
            const int selector_gate = (k_step & 1) * Traits::SCALE_MN_PACK;
            const int selector_up = selector_gate + 1;

            #pragma unroll
            for(int local_group = 0; local_group < kGroupsPerWave; ++local_group)
            {
                V_B rb_gate{};
                unpack_b_mfma_reg(b_gate[local_group], rb_gate);
                #pragma unroll
                for(int mi = 0; mi < Traits::M_MFMA_PER_WAVE; ++mi)
                {
                    accumulate_loaded_a_with_b<Traits, Mma>(
                        mma,
                        ra[mi],
                        rb_gate,
                        rc[mi][local_group],
                        mi,
                        route_valid[mi],
                        a_scale[mi / Traits::SCALE_MN_PACK],
                        k_step,
                        selector_gate,
                        b_scale[local_group]);
                }

                V_B rb_up{};
                unpack_b_mfma_reg(b_up[local_group], rb_up);
                #pragma unroll
                for(int mi = 0; mi < Traits::M_MFMA_PER_WAVE; ++mi)
                {
                    accumulate_loaded_a_with_b<Traits, Mma>(
                        mma,
                        ra[mi],
                        rb_up,
                        rc[mi][local_group + kGroupsPerWave],
                        mi,
                        route_valid[mi],
                        a_scale[mi / Traits::SCALE_MN_PACK],
                        k_step,
                        selector_up,
                        b_scale[local_group]);
                }
            }

            if constexpr(Traits::M_MFMA_PER_WAVE >= 8)
            {
                if(kk == 0 && next_k_pair < Traits::MFMA_K_STEPS)
                {
                    stage_a_reg_kpair_4mi_to_lds<Traits, 4>(
                        kargs,
                        wave_id,
                        lane_id,
                        next_k_pair,
                        pair_buffer ^ 1,
                        route_valid,
                        a_payload_base,
                        smem_a_reg);
                }
            }
        }

        if(next_k_pair < Traits::MFMA_K_STEPS)
        {
            wait_a_reg_kpair_to_lds<Traits>();
            __syncthreads();
        }
    }
}

#endif // __HIP_DEVICE_COMPILE__
} // namespace stage1_a8w4
} // namespace opus_moe
