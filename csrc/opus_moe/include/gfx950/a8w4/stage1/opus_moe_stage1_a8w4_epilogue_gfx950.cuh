// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_stage1_a8w4_policy_gfx950.cuh"
#include "mx_quant_utils.h"

namespace opus_moe
{
namespace stage1_a8w4
{

#ifdef __HIP_DEVICE_COMPILE__

inline __device__ float clamp_e4m3fn(float v)
{
    constexpr float kFp8E4M3FnMax = 448.0f;
    if(!(v == v))
        return 0.0f;
    v = v > kFp8E4M3FnMax ? kFp8E4M3FnMax : v;
    v = v < -kFp8E4M3FnMax ? -kFp8E4M3FnMax : v;
    return v;
}

template<typename Traits>
inline __device__ uint32_t fp32x4_to_fp8x4_word(float v0,
                                                float v1,
                                                float v2,
                                                float v3)
{
    int packed = 0;
    if constexpr(Traits::CLAMP_OUTPUT_FP8)
    {
        packed = __builtin_amdgcn_cvt_pk_fp8_f32(
            clamp_e4m3fn(v0), clamp_e4m3fn(v1), packed, 0);
        packed = __builtin_amdgcn_cvt_pk_fp8_f32(
            clamp_e4m3fn(v2), clamp_e4m3fn(v3), packed, 1);
    }
    else
    {
        packed = __builtin_amdgcn_cvt_pk_fp8_f32(v0, v1, packed, 0);
        packed = __builtin_amdgcn_cvt_pk_fp8_f32(v2, v3, packed, 1);
    }
    return static_cast<uint32_t>(packed);
}

// ============================================================================
// Epilogue: rc -> shared gate/up -> SiLU -> MXFP8 quant -> output.
// ============================================================================

inline __device__ float silu_mul(float gate, float up)
{
    constexpr float kNegLog2E = -1.4426950408889634f;
    const float emu = __builtin_amdgcn_exp2f(gate * kNegLog2E);
    const float sig = __builtin_amdgcn_rcpf(1.0f + emu);
    return gate * sig * up;
}

template<bool AssumeFinite>
inline __device__ uint8_t mxfp8_scale_byte_from_amax(float amax)
{
    // Exact RoundUp/RCEIL for FP8_E4M3:
    //   ceil_pow2(amax / 448), where 448 = 1.75 * 2^8.
    const uint32_t bits = __builtin_bit_cast(uint32_t, amax);
    const uint32_t exp = (bits >> 23) & 0xFFu;
    const uint32_t mant = bits & 0x7FFFFFu;
    if constexpr(!AssumeFinite)
    {
        if(exp == 0xFFu)
            return 0xFFu;
    }
    if(exp <= 8u)
    {
        const uint32_t scaled_bits =
            __builtin_bit_cast(uint32_t, amax * (1.0f / 448.0f));
        uint32_t scaled_exp = (scaled_bits >> 23) & 0xFFu;
        if(scaled_exp < 0xFFu && (scaled_bits & 0x7FFFFFu))
            scaled_exp += 1u;
        return static_cast<uint8_t>(scaled_exp);
    }
    uint32_t scale_exp = exp > 8u ? exp - 8u : 0u;
    if(mant > 0x600000u)
        scale_exp += 1u;
    return static_cast<uint8_t>(scale_exp);
}

inline __device__ float mxfp8_quant_scale(uint8_t e8m0_biased)
{
    const uint32_t quant_exp = 254u - static_cast<uint32_t>(e8m0_biased);
    return __builtin_bit_cast(float, quant_exp << 23);
}

inline __device__ float pair_amax(float local_amax, int lane_id)
{
    const int peer_lane = lane_id ^ 1;
    const int local_bits = __builtin_bit_cast(int, local_amax);
    const int peer_bits = __builtin_amdgcn_ds_bpermute(peer_lane << 2, local_bits);
    const float peer_amax = __builtin_bit_cast(float, peer_bits);
    return local_amax > peer_amax ? local_amax : peer_amax;
}

template<typename Traits, typename Acc>
inline __device__ void store_group_accum_to_smem(
    float* __restrict__ smem_gate_up,
    const Acc& acc,
    int ga_offset,
    int group,
    int gate_up,
    int out_half,
    int row_pass_base)
{
    const int local_m = a_local_m<Traits>(ga_offset);
    const int lane_k_byte = a_k_byte<Traits>(ga_offset);
    const int lane_div_16 = lane_k_byte / Traits::BYTES_PER_VEC;
    const int lane_mod_16 = local_m & (Traits::MMA_M - 1);

    #pragma unroll
    for(int ii = 0; ii < 4; ++ii)
    {
        const int row = (local_m & ~(Traits::MMA_M - 1)) +
                        lane_div_16 * 4 + ii;
        int smem_row = row;
        if constexpr(Traits::EPILOGUE_ROW_SPLIT != 1)
        {
            smem_row = row - row_pass_base;
            if(smem_row < 0 || smem_row >= Traits::EPILOGUE_ROWS_PER_PASS)
                continue;
        }
        const int local_col = out_half * Traits::MMA_N + lane_mod_16;
        smem_gate_up[gate_up_smem_offset<Traits>(
            smem_row, group, gate_up, local_col)] = static_cast<float>(acc[ii]);
    }
}

template<typename Traits, typename Acc>
inline __device__ void store_group_activated_to_smem(
    float* __restrict__ smem_values,
    const Acc& gate,
    const Acc& up,
    const AOperandCoord<Traits>& a_coord,
    int group,
    int out_half,
    int row_pass_base)
{
    #pragma unroll
    for(int ii = 0; ii < 4; ++ii)
    {
        const int row = (a_coord.local_m & ~(Traits::MMA_M - 1)) +
                        a_coord.lane_div_16 * 4 + ii;
        int smem_row = row;
        if constexpr(Traits::EPILOGUE_ROW_SPLIT != 1)
        {
            smem_row = row - row_pass_base;
            if(smem_row < 0 || smem_row >= Traits::EPILOGUE_ROWS_PER_PASS)
                continue;
        }

        const int local_col = out_half * Traits::MMA_N + a_coord.lane_mod_16;
        const float value =
            silu_mul(static_cast<float>(gate[ii]), static_cast<float>(up[ii]));
        smem_values[activated_smem_offset<Traits>(
            smem_row, group, local_col)] = value;
    }
}

template<typename Traits, typename Acc, int RowPassBase>
inline __device__ void store_group_activated_to_smem_static_rowpass(
    float* __restrict__ smem_values,
    const Acc& gate,
    const Acc& up,
    const AOperandCoord<Traits>& a_coord,
    int group,
    int out_half)
{
    #pragma unroll
    for(int ii = 0; ii < 4; ++ii)
    {
        const int row = (a_coord.local_m & ~(Traits::MMA_M - 1)) +
                        a_coord.lane_div_16 * 4 + ii;
        const int smem_row = row - RowPassBase;
        const int local_col = out_half * Traits::MMA_N + a_coord.lane_mod_16;
        const float value =
            silu_mul(static_cast<float>(gate[ii]), static_cast<float>(up[ii]));
        smem_values[activated_smem_offset<Traits>(
            smem_row, group, local_col)] = value;
    }
}

template<typename Traits, typename CAcc>
inline __device__ void epilogue_store_row_pass(
    float* __restrict__ smem_gate_up,
    CAcc (&rc)[Traits::M_MFMA_PER_WAVE]
             [Traits::ACC_SCALE_GROUPS_PER_TILE],
    const int (&ga)[Traits::M_MFMA_PER_WAVE],
    int wave_id,
    int row_pass_base)
{
    if constexpr(Traits::K_WAVE > 1)
    {
        if(wave_id >= Traits::KWAVE_BASE_WAVES)
            return;
    }
    const int epilogue_wave = wave_id % Traits::KWAVE_BASE_WAVES;
    const int out_half =
        Traits::PAIR_GATE_UP_SINGLE_GROUP ? epilogue_wave : epilogue_wave / 2;
    const int gate_up = epilogue_wave & 1;

    #pragma unroll
    for(int mi = 0; mi < Traits::M_MFMA_PER_WAVE; ++mi)
    {
        #pragma unroll
        for(int group = 0; group < Traits::OUTPUT_SCALE_GROUPS_PER_TILE; ++group)
        {
            if constexpr(Traits::PAIR_GATE_UP_SINGLE_GROUP)
            {
                store_group_accum_to_smem<Traits>(
                    smem_gate_up,
                    rc[mi][0],
                    ga[mi],
                    group,
                    0,
                    out_half,
                    row_pass_base);
                store_group_accum_to_smem<Traits>(
                    smem_gate_up,
                    rc[mi][1],
                    ga[mi],
                    group,
                    1,
                    out_half,
                    row_pass_base);
            }
            else
            {
                store_group_accum_to_smem<Traits>(
                    smem_gate_up,
                    rc[mi][group],
                    ga[mi],
                    group,
                    gate_up,
                    out_half,
                    row_pass_base);
            }
        }
    }
}

template<typename Traits>
inline __device__ void epilogue_quantize_row_pass(
    const OpusMoeStage1A8W4Kargs& kargs,
    const Tile<Traits>& tile,
    int row_pass_base,
    const int* __restrict__ smem_token,
    const int* __restrict__ smem_slot,
    const uint8_t* __restrict__ smem_route_valid,
    float* __restrict__ smem_gate_up)
{
    const bool epilogue_active = tile.tid < Traits::EPILOGUE_THREADS;
    const int smem_m = tile.tid >> 1;
    const int local_m = row_pass_base + smem_m;
    const int half = tile.tid & 1;
    const int local_col_base = half * (Traits::SCALE_GROUP_LOGICAL_K / 2);
    const int route_row = tile.route_base + local_m;
    if(!epilogue_active)
        return;

    int token = 0;
    int slot = 0;
    const bool route_valid = route_from_smem(
        local_m, smem_token, smem_slot, smem_route_valid, token, slot);
    if(!route_valid)
        return;

    for(int group = 0; group < Traits::OUTPUT_SCALE_GROUPS_PER_TILE; ++group)
    {
        const int group_col_base =
            tile.out_col_base + group * Traits::SCALE_GROUP_LOGICAL_K;
        float values[Traits::SCALE_GROUP_LOGICAL_K / 2];
        float local_amax = 0.0f;

        for(int local_col = 0;
            local_col < Traits::SCALE_GROUP_LOGICAL_K / 2;
            ++local_col)
        {
            const int smem_base = gate_up_smem_offset<Traits>(
                smem_m, group, 0, local_col_base + local_col);
            const float gate = smem_gate_up[smem_base];
            const float up =
                smem_gate_up[smem_base + Traits::SCALE_GROUP_LOGICAL_K];
            const float value = silu_mul(gate, up);
            values[local_col] = value;
            const float abs_value = __builtin_fabsf(value);
            local_amax = abs_value > local_amax ? abs_value : local_amax;
        }

        const int lane_id = tile.tid % opus::get_warp_size();
        const float amax = pair_amax(local_amax, lane_id);

        const uint8_t scale_byte =
            mxfp8_scale_byte_from_amax<!Traits::CLAMP_OUTPUT_FP8>(amax);
        const float quant_scale = mxfp8_quant_scale(scale_byte);
        const int scale_col =
            group_col_base / Traits::SCALE_GROUP_LOGICAL_K;
        if(half == 0)
        {
            const int scale_offset =
                output_scale_byte_offset<Traits>(
                    route_row, scale_col, kargs.stride_out_scale_route);
            kargs.inter_states_scale_e8m0[scale_offset] = scale_byte;
        }

        stage1_u32x4_t packed{};
        #pragma unroll
        for(int word = 0;
            word < (Traits::SCALE_GROUP_LOGICAL_K / 2) /
                       static_cast<int>(sizeof(uint32_t));
            ++word)
        {
            const int base = word * static_cast<int>(sizeof(uint32_t));
            packed[word] = fp32x4_to_fp8x4_word<Traits>(
                values[base + 0] * quant_scale,
                values[base + 1] * quant_scale,
                values[base + 2] * quant_scale,
                values[base + 3] * quant_scale);
        }

        const int64_t output_offset = output_byte_offset<Traits>(
            token,
            slot,
            group_col_base,
            kargs.stride_out_t,
            kargs.stride_out_k);
        *reinterpret_cast<stage1_u32x4_t*>(
            kargs.inter_states_fp8 + output_offset + local_col_base) =
            packed;
    }
}

template<typename Traits>
inline __device__ void epilogue_quantize_activated_row_pass(
    const OpusMoeStage1A8W4Kargs& kargs,
    const Tile<Traits>& tile,
    int row_pass_base,
    const int* __restrict__ smem_token,
    const int* __restrict__ smem_slot,
    const uint8_t* __restrict__ smem_route_valid,
    float* __restrict__ smem_values)
{
    constexpr bool kSplitQuantGroupBlocks2 =
        Traits::GATE_UP_GROUP_SPLIT &&
        Traits::OUTPUT_SCALE_GROUPS_PER_TILE == 4 &&
        Traits::EPILOGUE_THREADS * 2 == Traits::BLOCK_SIZE;
    constexpr bool kSplitQuantGroupBlocks3 =
        Traits::GATE_UP_GROUP_SPLIT &&
        (Traits::B_M == 32 || Traits::B_M == 64) &&
        Traits::EPILOGUE_ROW_SPLIT == 2 &&
        Traits::OUTPUT_SCALE_GROUPS_PER_TILE == 6 &&
        Traits::EPILOGUE_THREADS * 3 <= Traits::BLOCK_SIZE;
    constexpr bool kSplitQuantGroupBlocks6x2 =
        Traits::GATE_UP_GROUP_SPLIT &&
        Traits::B_M == 128 &&
        Traits::EPILOGUE_ROW_SPLIT == 2 &&
        Traits::OUTPUT_SCALE_GROUPS_PER_TILE == 6 &&
        Traits::EPILOGUE_THREADS * 2 <= Traits::BLOCK_SIZE;

    int smem_m = tile.tid >> 1;
    if constexpr(kSplitQuantGroupBlocks2 || kSplitQuantGroupBlocks6x2)
        smem_m = tile.tid >> 2;
    if constexpr(kSplitQuantGroupBlocks3)
        smem_m = tile.tid / 6;
    const int half = tile.tid & 1;
    const int local_m = row_pass_base + smem_m;
    const int local_col_base = half * (Traits::SCALE_GROUP_LOGICAL_K / 2);
    const int route_row = tile.route_base + local_m;
    bool epilogue_active = tile.tid < Traits::EPILOGUE_THREADS;
    if constexpr(kSplitQuantGroupBlocks2 || kSplitQuantGroupBlocks6x2)
        epilogue_active = tile.tid < Traits::EPILOGUE_THREADS * 2;
    if constexpr(kSplitQuantGroupBlocks3)
        epilogue_active = tile.tid < Traits::EPILOGUE_THREADS * 3;
    if(!epilogue_active)
        return;

    int token = 0;
    int slot = 0;
    const bool route_valid = route_from_smem(
        local_m, smem_token, smem_slot, smem_route_valid, token, slot);
    if(!route_valid)
        return;

    constexpr int kGroupsPerThread =
        kSplitQuantGroupBlocks2 ? 2 :
        kSplitQuantGroupBlocks6x2 ? 3 :
        kSplitQuantGroupBlocks3 ? 2 :
        Traits::OUTPUT_SCALE_GROUPS_PER_TILE;
    int group_base = 0;
    if constexpr(kSplitQuantGroupBlocks2)
        group_base = ((tile.tid >> 1) & 1) * 2;
    if constexpr(kSplitQuantGroupBlocks6x2)
        group_base = ((tile.tid >> 1) & 1) * 3;
    if constexpr(kSplitQuantGroupBlocks3)
        group_base = ((tile.tid >> 1) % 3) * 2;
    for(int local_group = 0; local_group < kGroupsPerThread; ++local_group)
    {
        const int group = group_base + local_group;
        const int group_col_base =
            tile.out_col_base + group * Traits::SCALE_GROUP_LOGICAL_K;
        float values[Traits::SCALE_GROUP_LOGICAL_K / 2];
        float local_amax = 0.0f;

        for(int local_col = 0;
            local_col < Traits::SCALE_GROUP_LOGICAL_K / 2;
            ++local_col)
        {
            const float value =
                smem_values[activated_smem_offset<Traits>(
                    smem_m, group, local_col_base + local_col)];
            values[local_col] = value;
            const float abs_value = __builtin_fabsf(value);
            local_amax = abs_value > local_amax ? abs_value : local_amax;
        }

        const int lane_id = tile.tid % opus::get_warp_size();
        const float amax = pair_amax(local_amax, lane_id);

        const uint8_t scale_byte =
            mxfp8_scale_byte_from_amax<!Traits::CLAMP_OUTPUT_FP8>(amax);
        const float quant_scale = mxfp8_quant_scale(scale_byte);
        const int scale_col =
            group_col_base / Traits::SCALE_GROUP_LOGICAL_K;
        if(half == 0)
        {
            const int scale_offset =
                output_scale_byte_offset<Traits>(
                    route_row, scale_col, kargs.stride_out_scale_route);
            kargs.inter_states_scale_e8m0[scale_offset] = scale_byte;
        }

        stage1_u32x4_t packed{};
        #pragma unroll
        for(int word = 0;
            word < (Traits::SCALE_GROUP_LOGICAL_K / 2) /
                       static_cast<int>(sizeof(uint32_t));
            ++word)
        {
            const int base = word * static_cast<int>(sizeof(uint32_t));
            packed[word] = fp32x4_to_fp8x4_word<Traits>(
                values[base + 0] * quant_scale,
                values[base + 1] * quant_scale,
                values[base + 2] * quant_scale,
                values[base + 3] * quant_scale);
        }

        const int64_t output_offset = output_byte_offset<Traits>(
            token,
            slot,
            group_col_base,
            kargs.stride_out_t,
            kargs.stride_out_k);
        *reinterpret_cast<stage1_u32x4_t*>(
            kargs.inter_states_fp8 + output_offset + local_col_base) =
            packed;
    }
}

template<typename Traits, typename CAcc, typename LayoutA>
inline __device__ void quant_epilogue(
    const OpusMoeStage1A8W4Kargs& kargs,
    const Tile<Traits>& tile,
    const LayoutA& u_ga,
    CAcc (&rc)[Traits::M_MFMA_PER_WAVE]
             [Traits::ACC_SCALE_GROUPS_PER_TILE],
    int wave_id,
    const int* __restrict__ smem_token,
    const int* __restrict__ smem_slot,
    const uint8_t* __restrict__ smem_route_valid,
    float* __restrict__ smem_gate_up)
{
    int ga[Traits::M_MFMA_PER_WAVE];
    collect_a_offsets<Traits>(u_ga, ga);

    #pragma unroll
    for(int row_pass = 0; row_pass < Traits::EPILOGUE_ROW_SPLIT; ++row_pass)
    {
        const int row_pass_base = row_pass * Traits::EPILOGUE_ROWS_PER_PASS;
        epilogue_store_row_pass<Traits>(
            smem_gate_up, rc, ga, wave_id, row_pass_base);
        __syncthreads();
        epilogue_quantize_row_pass<Traits>(
            kargs,
            tile,
            row_pass_base,
            smem_token,
            smem_slot,
            smem_route_valid,
            smem_gate_up);

        if(row_pass + 1 < Traits::EPILOGUE_ROW_SPLIT)
            __syncthreads();
    }
}

template<typename Traits, typename CAcc>
inline __device__ void epilogue_store_activated_row_pass_gate_up_group_split(
    float* __restrict__ smem_values,
    CAcc (&rc)[Traits::M_MFMA_PER_WAVE]
             [Traits::ACC_SCALE_GROUPS_PER_TILE],
    const AOperandCoord<Traits> (&a_coord)[Traits::M_MFMA_PER_WAVE],
    int wave_id,
    int row_pass_base)
{
    static_assert(Traits::GATE_UP_GROUP_SPLIT);
    constexpr int kGroupsPerWave = Traits::GATE_UP_GROUP_SPLIT_GROUPS;
    static_assert(kGroupsPerWave * 2 == Traits::OUTPUT_SCALE_GROUPS_PER_TILE);

    const int out_half = wave_id & 1;
    const int group_block = wave_id / 2;
    if(group_block >= 2)
        return;

    #pragma unroll
    for(int mi = 0; mi < Traits::M_MFMA_PER_WAVE; ++mi)
    {
        #pragma unroll
        for(int local_group = 0; local_group < kGroupsPerWave; ++local_group)
        {
            const int group = group_block * kGroupsPerWave + local_group;
            store_group_activated_to_smem<Traits>(
                smem_values,
                rc[mi][local_group],
                rc[mi][local_group + kGroupsPerWave],
                a_coord[mi],
                group,
                out_half,
                row_pass_base);
        }
    }
}

template<typename Traits, typename CAcc, int RowPass>
inline __device__ void
epilogue_store_activated_row_pass_gate_up_group_split_static(
    float* __restrict__ smem_values,
    CAcc (&rc)[Traits::M_MFMA_PER_WAVE]
             [Traits::ACC_SCALE_GROUPS_PER_TILE],
    const AOperandCoord<Traits> (&a_coord)[Traits::M_MFMA_PER_WAVE],
    int wave_id)
{
    static_assert(Traits::GATE_UP_GROUP_SPLIT);
    static_assert(Traits::EPILOGUE_ROW_SPLIT == 2);
    static_assert(Traits::B_M == 128);
    static_assert(Traits::MMA_M == 16);
    static_assert(Traits::EPILOGUE_ROWS_PER_PASS == 64);

    constexpr int kGroupsPerWave = Traits::GATE_UP_GROUP_SPLIT_GROUPS;
    constexpr int kMiPerPass = Traits::EPILOGUE_ROWS_PER_PASS / Traits::MMA_M;
    constexpr int kMiBegin = RowPass * kMiPerPass;
    constexpr int kMiEnd = kMiBegin + kMiPerPass;
    constexpr int kRowPassBase = RowPass * Traits::EPILOGUE_ROWS_PER_PASS;

    static_assert(kGroupsPerWave * 2 == Traits::OUTPUT_SCALE_GROUPS_PER_TILE);
    static_assert(kMiEnd <= Traits::M_MFMA_PER_WAVE);

    const int out_half = wave_id & 1;
    const int group_block = wave_id / 2;
    if(group_block >= 2)
        return;

    #pragma unroll
    for(int mi = kMiBegin; mi < kMiEnd; ++mi)
    {
        #pragma unroll
        for(int local_group = 0; local_group < kGroupsPerWave; ++local_group)
        {
            const int group = group_block * kGroupsPerWave + local_group;
            store_group_activated_to_smem_static_rowpass<Traits, CAcc, kRowPassBase>(
                smem_values,
                rc[mi][local_group],
                rc[mi][local_group + kGroupsPerWave],
                a_coord[mi],
                group,
                out_half);
        }
    }
}

template<typename Traits, typename CAcc, typename LayoutA>
inline __device__ void quant_epilogue_gate_up_group_split(
    const OpusMoeStage1A8W4Kargs& kargs,
    const Tile<Traits>& tile,
    const LayoutA& u_ga,
    CAcc (&rc)[Traits::M_MFMA_PER_WAVE]
             [Traits::ACC_SCALE_GROUPS_PER_TILE],
    int wave_id,
    const int* __restrict__ smem_token,
    const int* __restrict__ smem_slot,
    const uint8_t* __restrict__ smem_route_valid,
    float* __restrict__ smem_gate_up)
{
    static_assert(Traits::GATE_UP_GROUP_SPLIT);

    AOperandCoord<Traits> a_coord[Traits::M_MFMA_PER_WAVE];
    collect_a_coords<Traits>(u_ga, a_coord);

    if constexpr(Traits::EPILOGUE_ROW_SPLIT == 2 && Traits::B_M == 128)
    {
        epilogue_store_activated_row_pass_gate_up_group_split_static<Traits,
                                                                     CAcc,
                                                                     0>(
            smem_gate_up, rc, a_coord, wave_id);
        __syncthreads();
        epilogue_quantize_activated_row_pass<Traits>(
            kargs,
            tile,
            0,
            smem_token,
            smem_slot,
            smem_route_valid,
            smem_gate_up);

        __syncthreads();

        epilogue_store_activated_row_pass_gate_up_group_split_static<Traits,
                                                                     CAcc,
                                                                     1>(
            smem_gate_up, rc, a_coord, wave_id);
        __syncthreads();
        epilogue_quantize_activated_row_pass<Traits>(
            kargs,
            tile,
            Traits::EPILOGUE_ROWS_PER_PASS,
            smem_token,
            smem_slot,
            smem_route_valid,
            smem_gate_up);
    }
    else
    {
        #pragma unroll
        for(int row_pass = 0; row_pass < Traits::EPILOGUE_ROW_SPLIT; ++row_pass)
        {
            const int row_pass_base = row_pass * Traits::EPILOGUE_ROWS_PER_PASS;
            epilogue_store_activated_row_pass_gate_up_group_split<Traits>(
                smem_gate_up, rc, a_coord, wave_id, row_pass_base);
            __syncthreads();
            epilogue_quantize_activated_row_pass<Traits>(
                kargs,
                tile,
                row_pass_base,
                smem_token,
                smem_slot,
                smem_route_valid,
                smem_gate_up);

            if(row_pass + 1 < Traits::EPILOGUE_ROW_SPLIT)
                __syncthreads();
        }
    }
}

#endif // __HIP_DEVICE_COMPILE__
} // namespace stage1_a8w4
} // namespace opus_moe
