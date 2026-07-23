// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_stage1_a8w4_traits_gfx950.cuh"
#include "mx_quant_utils.h"

#include "opus/hip_minimal.hpp"

namespace opus_moe
{
namespace stage1_a8w4
{
namespace pipeline_common
{

#ifdef __HIP_DEVICE_COMPILE__

using u32x4_t = uint32_t __attribute__((ext_vector_type(4)));
using stage1_u32x4_t = u32x4_t;
typedef uint32_t stage1_u32x8_t __attribute__((ext_vector_type(8)));

struct Route { int token, slot; bool valid; };
struct Tile
{
    int tid, wave_id, route_base, out_col_base, out_half, group_base;
};

template<typename ScaleGmem>
__device__ __forceinline__ int load_scale_word(ScaleGmem& g_scale, int byte_offset)
{
    const auto raw =
        g_scale.template load<static_cast<int>(sizeof(uint32_t))>(
            byte_offset);
    return static_cast<int>(__builtin_bit_cast(uint32_t, raw));
}

template<typename Traits>
__device__ __forceinline__ int scale_word_byte_offset(int base_word_offset,
                                                      int k_pair)
{
    const int k_pair_idx = k_pair / Traits::SCALE_K_PACK;
    const int word_offset = base_word_offset +
                            k_pair_idx * Traits::SCALE_LAYOUT_STRIDE_K0;
    return word_offset * static_cast<int>(sizeof(uint32_t));
}

template<typename ScaleLayout>
__device__ __forceinline__ int scale_base_word_stage1(
    const ScaleLayout& u_sf, int mn_pack)
{
    return static_cast<int>(u_sf(mn_pack));
}

template<typename Traits, typename ScaleGmem>
__device__ __forceinline__ void load_sfa_frag_stage1(
    ScaleGmem& g_scale,
    const int (&a_scale_base_word)[Traits::M_SCALE_PACKS],
    int k_pair,
    int (&a_scale)[Traits::M_SCALE_PACKS])
{
    #pragma unroll
    for(int mp = 0; mp < Traits::M_SCALE_PACKS; ++mp)
    {
        a_scale[mp] = load_scale_word(
            g_scale,
            scale_word_byte_offset<Traits>(
                a_scale_base_word[mp], k_pair));
    }
}

template<typename Traits, typename ScaleGmem>
__device__ __forceinline__ void load_sfb_frag_stage1(
    ScaleGmem& g_scale,
    const int (&b_scale_base_word)[Traits::B_GROUPS_PER_WAVE],
    int k_pair,
    int (&b_scale)[Traits::B_GROUPS_PER_WAVE])
{
    #pragma unroll
    for(int local_group = 0; local_group < Traits::B_GROUPS_PER_WAVE;
        ++local_group)
    {
        b_scale[local_group] = load_scale_word(
            g_scale,
            scale_word_byte_offset<Traits>(
                b_scale_base_word[local_group], k_pair));
    }
}

__device__ __forceinline__ Route decode_route(int32_t packed)
{
    return {opus_moe_token_id(packed), opus_moe_topk_slot(packed), packed >= 0};
}

__device__ __forceinline__ Route route_from_smem(
    int local_m, const int* __restrict__ smem_route)
{
    const int packed = smem_route[local_m];
    return packed < 0 ? Route{0, 0, false} : decode_route(packed);
}

template<typename Traits>
__device__ __forceinline__ void load_routes_stage1(
    const OpusMoeStage1A8W4Kargs& kargs,
    int route_base,
    int valid_rows,
    int tid,
    int* __restrict__ smem_route)
{
    const bool tile_full_valid = route_base + Traits::B_M <= valid_rows;
    for(int local_m = tid; local_m < Traits::B_M;
        local_m += Traits::BLOCK_SIZE)
    {
        const int route_row = route_base + local_m;
        int packed = (!tile_full_valid && route_row >= valid_rows) ?
            -1 : kargs.sorted_token_ids[route_row];
        const int token = opus_moe_token_id(packed);
        const int slot = opus_moe_topk_slot(packed);
        if(token >= kargs.token_num || slot >= kargs.topk)
            packed = -1;
        smem_route[local_m] = packed;
    }
}

template<typename Traits>
__device__ __forceinline__ int64_t a_payload_offset(int64_t payload_base,
                                                    int k_step)
{
    return payload_base + static_cast<int64_t>(k_step) * Traits::MMA_K;
}

template<typename Traits>
__device__ __forceinline__ int64_t b_payload_offset(int64_t payload_base,
                                                    int item,
                                                    int k_step,
                                                    int k_steps)
{
    return payload_base +
           static_cast<int64_t>(item * k_steps + k_step) *
               Traits::W1_PAYLOAD_K_STEP_STRIDE_BYTES;
}

template<typename Traits>
__device__ __forceinline__ int epilogue_smem_offset(int smem_row,
                                                    int group,
                                                    int local_col)
{
    return smem_row * Traits::EPILOGUE_SMEM_COLS +
           group * Traits::SCALE_GROUP_LOGICAL_K + local_col;
}

template<typename Traits>
__device__ __forceinline__ int a_reg_lds_stage_byte_offset(int k_step)
{
    return (k_step & (2 * Traits::SCALE_K_PACK - 1)) *
           Traits::A_REG_LDS_STAGE_BYTES;
}

template<typename Traits, int Mi>
__device__ __forceinline__ int select_sfa_stage1(
    const bool (&a_route_valid)[Traits::M_MFMA_PER_WAVE],
    const int (&a_scale)[Traits::M_SCALE_PACKS])
{
    constexpr int kNoScaleWord = 0x7f7f7f7f;
    const bool route_valid =
        (Traits::GATE_UP_GROUP_SPLIT && Traits::SKIP_INVALID_A_SCALE_GUARD) ||
        a_route_valid[Mi];
    return route_valid ? a_scale[Mi / Traits::SCALE_MN_PACK] : kNoScaleWord;
}

template<typename Traits, int Item, int Kk>
__device__ __forceinline__ constexpr auto selector_b()
{
    return opus::number<Kk * Traits::SCALE_MN_PACK + Item>{};
}

template<typename Traits, int Mi, int Kk, typename Fn>
__device__ __forceinline__ void with_selector_a(int route_base, const Fn& run)
{
    if constexpr(Traits::M_MFMA_PER_WAVE == 1)
    {
        if(((route_base / Traits::MMA_M) & 1) == 0)
            run(opus::number<Kk * Traits::SCALE_MN_PACK>{});
        else
            run(opus::number<Kk * Traits::SCALE_MN_PACK + 1>{});
    }
    else
    {
        run(opus::number<Kk * Traits::SCALE_MN_PACK +
                         (Mi & (Traits::SCALE_MN_PACK - 1))>{});
    }
}

template<typename V_A>
__device__ __forceinline__ V_A make_a_reg(stage1_u32x4_t a_lo,
                                          stage1_u32x4_t a_hi)
{
    stage1_u32x8_t a_packed{
        a_lo[0], a_lo[1], a_lo[2], a_lo[3],
        a_hi[0], a_hi[1], a_hi[2], a_hi[3]};
    return __builtin_bit_cast(V_A, a_packed);
}

template<typename V_A, typename Traits, typename HiddenGmem>
__device__ __forceinline__ V_A load_a_reg_stage1(
    HiddenGmem& g_a,
    int64_t payload_base,
    int k_step)
{
    const int64_t a_offset =
        a_payload_offset<Traits>(payload_base, k_step);
    const auto a_lo = __builtin_bit_cast(
        stage1_u32x4_t,
        g_a.template load<Traits::BYTES_PER_VEC>(
            static_cast<int>(a_offset)));
    const auto a_hi = __builtin_bit_cast(
        stage1_u32x4_t,
        g_a.template load<Traits::BYTES_PER_VEC>(
            static_cast<int>(a_offset + 4 * Traits::BYTES_PER_VEC)));
    return make_a_reg<V_A>(a_lo, a_hi);
}

template<typename V_B>
__device__ __forceinline__ V_B make_b_reg(stage1_u32x4_t b_raw)
{
    stage1_u32x8_t b_packed{
        b_raw[0], b_raw[1], b_raw[2], b_raw[3], 0, 0, 0, 0};
    return __builtin_bit_cast(V_B, b_packed);
}

template<typename Traits, typename WeightGmem>
__device__ __forceinline__ stage1_u32x4_t load_b_raw_stage1(
    WeightGmem& g_b,
    int64_t payload_base,
    int item,
    int k_step,
    int k_steps)
{
    const int64_t b_offset =
        b_payload_offset<Traits>(payload_base, item, k_step, k_steps);
    // aux=2 (stream/SLC) vs aux=0 (L2 cache) per WEIGHT_LOAD_STREAM (see traits).
    return __builtin_bit_cast(
        stage1_u32x4_t,
        g_b.template load<Traits::BYTES_PER_VEC>(
            static_cast<int>(b_offset), 0,
            opus::number<Traits::WEIGHT_LOAD_STREAM ? 2 : 0>{}));
}

template<typename Traits, typename ALayout>
__device__ __forceinline__ int a_lane_byte_stage1(
    const ALayout& u_ga, int mi, int stride_hidden_t)
{
    const int a_lane_base = static_cast<int>(u_ga(0, 0, 0, 0));
    return mi * Traits::MMA_M * stride_hidden_t + a_lane_base;
}

__device__ __forceinline__ int a_lane_local_m_stage1(int a_lane_byte,
                                                     int stride_hidden_t)
{
    return a_lane_byte / stride_hidden_t;
}

__device__ __forceinline__ int64_t a_payload_base_stage1(
    const OpusMoeStage1A8W4Kargs& kargs, const Route& route, int lane_k_byte)
{
    return static_cast<int64_t>(route.token) * kargs.stride_hidden_t +
           lane_k_byte;
}

template<typename Traits>
__device__ __forceinline__ int a_scale_mn_pack_stage1(int route_base, int mp)
{
    return route_base / (Traits::SCALE_MN_PACK * Traits::MMA_M) + mp;
}

template<typename Traits, typename ScaleLayout>
__device__ __forceinline__ int a_scale_base_word_stage1(
    const ScaleLayout& u_sf, int route_base, int mp)
{
    return scale_base_word_stage1(
        u_sf, a_scale_mn_pack_stage1<Traits>(route_base, mp));
}

template<typename Traits>
__device__ __forceinline__ int w1_n0_stage1(
    int out_col_base, int out_half, int group_base, int local_group)
{
    const int group_col_base =
        out_col_base +
        (group_base + local_group) * Traits::SCALE_GROUP_LOGICAL_K;
    const int output_col_base = group_col_base + out_half * Traits::MMA_N;
    return output_col_base / Traits::MMA_N;
}

template<typename Traits>
__device__ __forceinline__ int64_t b_payload_base_stage1(
    const OpusMoeStage1A8W4Kargs& kargs,
    int expert_id,
    int w1_n0,
    int b_lane_byte,
    int k_steps)
{
    return static_cast<int64_t>(expert_id) * kargs.stride_w1_e +
           static_cast<int64_t>(w1_n0) * 2 * k_steps *
               Traits::W1_PAYLOAD_K_STEP_STRIDE_BYTES +
           b_lane_byte;
}

template<typename Traits>
__device__ __forceinline__ int b_scale_mn_pack_stage1(
    const OpusMoeStage1A8W4Kargs& kargs, int expert_id, int w1_n0)
{
    return expert_id * (2 * kargs.inter_dim) /
               (Traits::SCALE_MN_PACK * Traits::MMA_M) +
           w1_n0;
}

template<typename Traits, typename ScaleLayout>
__device__ __forceinline__ int b_scale_base_word_stage1(
    const ScaleLayout& u_sf,
    const OpusMoeStage1A8W4Kargs& kargs,
    int expert_id,
    int w1_n0)
{
    return scale_base_word_stage1(
        u_sf, b_scale_mn_pack_stage1<Traits>(kargs, expert_id, w1_n0));
}

template<typename Traits>
__device__ __forceinline__ int a_reg_lds_load_byte_offset(
    int k_step, int mi)
{
    return a_reg_lds_stage_byte_offset<Traits>(k_step) +
           mi * Traits::MMA_M * Traits::MMA_K;
}

template<typename V_A, typename Traits, typename SmemA, typename RegLayout>
__device__ __forceinline__ V_A load_a_reg_lds_stage1(
    SmemA& s_a,
    const RegLayout& u_ra,
    int k_step,
    int mi)
{
    const int smem_base =
        a_reg_lds_load_byte_offset<Traits>(k_step, mi);
    return __builtin_bit_cast(
        V_A,
        opus::load<Traits::BYTES_PER_VEC>(
            s_a, u_ra + smem_base));
}

template<typename Traits, typename WeightGmem>
__device__ __forceinline__ void load_b_kpair_stage1(
    WeightGmem& g_b,
    const int64_t (&b_group_payload_base)[Traits::B_GROUPS_PER_WAVE],
    int k_pair,
    int k_steps,
    stage1_u32x4_t (&b_kk0)[Traits::B_ITEMS_PER_WAVE])
{
    #pragma unroll
    for(int flat_item = 0; flat_item < Traits::B_ITEMS_PER_WAVE; ++flat_item)
    {
        const int local_group = flat_item / Traits::B_ITEMS_PER_GROUP;
        const int item = flat_item - local_group * Traits::B_ITEMS_PER_GROUP;
        b_kk0[flat_item] = load_b_raw_stage1<Traits>(
            g_b, b_group_payload_base[local_group],
            item, k_pair, k_steps);
    }
}

template<typename Traits, typename WeightGmem>
__device__ __forceinline__ void load_b_full_kpair_stage1(
    WeightGmem& g_b,
    const int64_t (&b_group_payload_base)[Traits::B_GROUPS_PER_WAVE],
    int k_pair,
    int k_steps,
    stage1_u32x4_t (&b_kpair)[Traits::SCALE_K_PACK]
                             [Traits::B_ITEMS_PER_WAVE])
{
    opus::static_for<Traits::SCALE_K_PACK>([&](auto kk_id) {
        constexpr int kk = kk_id.value;
        load_b_kpair_stage1<Traits>(
            g_b, b_group_payload_base, k_pair + kk,
            k_steps, b_kpair[kk]);
    });
}

template<typename Traits, typename SmemLayout>
__device__ __forceinline__ int a_reg_lds_store_byte_offset(
    const SmemLayout& u_sa, int k_step, int mi, int half)
{
    return a_reg_lds_stage_byte_offset<Traits>(k_step) +
           static_cast<int>(u_sa(mi, half, opus::number<0>{}));
}

template<typename Traits, typename HiddenGmem, typename SmemA, typename SmemLayout>
__device__ __forceinline__ void stage_a_reg_kpair_to_lds(
    HiddenGmem& hidden_gmem,
    SmemA& s_a,
    const SmemLayout& u_sa,
    const Tile& tile,
    int k_pair,
    const int64_t (&a_payload_base)[Traits::M_MFMA_PER_WAVE])
{
    constexpr int kProducerWaves = Traits::BLOCK_SIZE / Traits::WAVE_SIZE;

    opus::static_for<Traits::M_MFMA_PER_WAVE>([&](auto mi_id) {
        constexpr int mi = mi_id.value;
        if(mi % kProducerWaves != tile.wave_id)
            return;

        #pragma unroll
        for(int kk = 0; kk < Traits::SCALE_K_PACK; ++kk)
        {
            const int k_step = k_pair + kk;
            const int64_t a_offset =
                a_payload_offset<Traits>(a_payload_base[mi], k_step);
            const int smem_lo = a_reg_lds_store_byte_offset<Traits>(
                u_sa, k_step, mi, 0);
            const int a_offset_i32 = static_cast<int>(a_offset);
            hidden_gmem.template async_load<Traits::BYTES_PER_VEC>(
                s_a.ptr + smem_lo, a_offset_i32);
            const int smem_hi = a_reg_lds_store_byte_offset<Traits>(
                u_sa, k_step, mi, 1);
            hidden_gmem.template async_load<Traits::BYTES_PER_VEC>(
                s_a.ptr + smem_hi,
                a_offset_i32 + 4 * Traits::BYTES_PER_VEC);
        }
    });
}

template<typename Traits, typename CReg>
__device__ __forceinline__ void reduce_single_group_kwave(
    int wave_id,
    int lane_id,
    uint8_t* __restrict__ smem_reduce,
    CReg (&rc)[Traits::M_MFMA_PER_WAVE]
             [Traits::ACC_SCALE_GROUPS_PER_TILE])
{
    if constexpr(Traits::K_WAVE > 1)
    {
        constexpr int kAccItems =
            Traits::M_MFMA_PER_WAVE * Traits::ACC_SCALE_GROUPS_PER_TILE;
        const int base_wave = wave_id % Traits::KWAVE_BASE_WAVES;
        const int wave_k = wave_id / Traits::KWAVE_BASE_WAVES;
        auto* smem_c = reinterpret_cast<CReg*>(smem_reduce);

        opus::static_for<kAccItems>([&](auto item_id) {
            constexpr int item = item_id.value;
            constexpr int mi = item / Traits::ACC_SCALE_GROUPS_PER_TILE;
            constexpr int acc_group =
                item - mi * Traits::ACC_SCALE_GROUPS_PER_TILE;
            const int store_idx =
                (((wave_k * Traits::KWAVE_BASE_WAVES + base_wave) *
                      kAccItems +
                  item) *
                     Traits::WAVE_SIZE +
                 lane_id);
            smem_c[store_idx] = rc[mi][acc_group];
        });
        opus::sync_threads();

        if(wave_k == 0)
        {
            opus::static_for<kAccItems>([&](auto item_id) {
                constexpr int item = item_id.value;
                constexpr int mi = item / Traits::ACC_SCALE_GROUPS_PER_TILE;
                constexpr int acc_group =
                    item - mi * Traits::ACC_SCALE_GROUPS_PER_TILE;
                const int base_idx =
                    ((base_wave * kAccItems + item) *
                         Traits::WAVE_SIZE +
                     lane_id);
                CReg sum = smem_c[base_idx];
                opus::static_for<Traits::K_WAVE - 1>([&](auto kw_id) {
                    constexpr int kw = kw_id.value + 1;
                    const int load_idx =
                        (((kw * Traits::KWAVE_BASE_WAVES + base_wave) *
                              kAccItems +
                          item) *
                             Traits::WAVE_SIZE +
                         lane_id);
                    sum = sum + smem_c[load_idx];
                });
                rc[mi][acc_group] = sum;
            });
        }
        opus::sync_threads();
    }
}

template<Stage1Activation Act>
__device__ __forceinline__ float activation_mul_stage1(float gate_raw, float up_raw)
{
    if constexpr(Act == Stage1Activation::Swiglu)
    {
        constexpr float kLimit = 7.0f;
        constexpr float kNegAlphaLog2E = -1.4426950408889634f * 1.702f;
        const float gate = gate_raw < kLimit ? gate_raw : kLimit;
        const float up_hi = up_raw < kLimit ? up_raw : kLimit;
        const float linear = up_hi > -kLimit ? up_hi : -kLimit;
        const float emu = __builtin_amdgcn_exp2f(gate * kNegAlphaLog2E);
        const float sig = __builtin_amdgcn_rcpf(1.0f + emu);
        return gate * sig * (linear + 1.0f);
    }
    else
    {
        constexpr float kNegLog2E = -1.4426950408889634f;
        const float emu = __builtin_amdgcn_exp2f(gate_raw * kNegLog2E);
        const float sig = __builtin_amdgcn_rcpf(1.0f + emu);
        return gate_raw * sig * up_raw;
    }
}

__device__ __forceinline__ float load_w1_bias_stage1(
    const OpusMoeStage1A8W4Kargs& kargs, int expert_id, int col)
{
    return kargs.w1_bias[
        static_cast<int64_t>(expert_id) * kargs.stride_w1_bias_e +
        col];
}

template<typename Traits, typename Acc, typename CLayoutM, typename CLayoutN>
__device__ __forceinline__ void epilogue_store_group_to_smem(
    const OpusMoeStage1A8W4Kargs& kargs,
    float* __restrict__ smem_values,
    const Acc& gate_or_acc, const Acc& up,
    const CLayoutM& u_c_m,
    const CLayoutN& u_c_n,
    const Tile& tile, int expert_id, int group, int row_pass_base)
{
    opus::static_for<4>([&](auto ii_id) {
        constexpr int ii = ii_id.value;
        const int smem_row =
            static_cast<int>(u_c_m(0, 0, ii, 0)) - row_pass_base;
        const int local_col =
            tile.out_half * Traits::MMA_N +
            static_cast<int>(u_c_n(0, 0, ii, 0));
        float gate = static_cast<float>(gate_or_acc[ii]);
        float up_value = static_cast<float>(up[ii]);
        if constexpr(Traits::ACTIVATION == Stage1Activation::Swiglu)
        {
            const int output_col =
                tile.out_col_base +
                group * Traits::SCALE_GROUP_LOGICAL_K + local_col;
            gate += load_w1_bias_stage1(kargs, expert_id, output_col);
            up_value += load_w1_bias_stage1(
                kargs, expert_id, kargs.inter_dim + output_col);
        }
        const float value = activation_mul_stage1<Traits::ACTIVATION>(
            gate, up_value);
        const int offset = epilogue_smem_offset<Traits>(
            smem_row, group, local_col);
        smem_values[offset] = value;
    });
}

template<typename Traits,
         int RowPass,
         typename CAcc,
         typename CLayoutM,
         typename CLayoutN>
__device__ __forceinline__ void epilogue_store_row_pass(
    const OpusMoeStage1A8W4Kargs& kargs,
    float* __restrict__ smem_values,
    CAcc (&rc)[Traits::M_MFMA_PER_WAVE]
             [Traits::ACC_SCALE_GROUPS_PER_TILE],
    const CLayoutM& u_c_m,
    const CLayoutN& u_c_n,
    const Tile& tile,
    int expert_id)
{
    if constexpr(!Traits::GATE_UP_GROUP_SPLIT && Traits::K_WAVE > 1)
        if(tile.wave_id >= Traits::KWAVE_BASE_WAVES)
            return;

    constexpr int kRowPassBase = RowPass * Traits::EPILOGUE_ROWS_PER_PASS;
    constexpr int kMiPerPass =
        Traits::GATE_UP_GROUP_SPLIT ? Traits::EPILOGUE_ROWS_PER_PASS / Traits::MMA_M :
                                      Traits::M_MFMA_PER_WAVE;
    constexpr int kMiBegin =
        Traits::GATE_UP_GROUP_SPLIT ? kRowPassBase / Traits::MMA_M : 0;
    constexpr int kStoreGroups =
        Traits::GATE_UP_GROUP_SPLIT ? Traits::B_GROUPS_PER_WAVE : 1;
    constexpr int kUpOffset = Traits::B_GROUPS_PER_WAVE;

    opus::static_for<kMiPerPass>([&](auto mi_local_id) {
        constexpr int mi = kMiBegin + mi_local_id.value;
        opus::static_for<kStoreGroups>([&](auto local_group_id) {
            constexpr int local_group = local_group_id.value;
            const int group = Traits::GATE_UP_GROUP_SPLIT ?
                tile.group_base + local_group : 0;
            epilogue_store_group_to_smem<Traits, CAcc>(
                kargs, smem_values, rc[mi][local_group],
                rc[mi][local_group + kUpOffset], u_c_m, u_c_n,
                tile, expert_id, group, kRowPassBase - mi * Traits::MMA_M);
        });
    });
}

template<typename Traits, typename OutGmem, typename OutScaleGmem>
inline __device__ void epilogue_quantize_row_pass(
    const OpusMoeStage1A8W4Kargs& kargs,
    OutGmem& g_out,
    OutScaleGmem& g_out_scale,
    const Tile& tile, int row_pass_base,
    const int* __restrict__ smem_route,
    float* __restrict__ smem_values)
{
    if(tile.tid >= Traits::QUANT_ACTIVE_THREADS)
        return;

    const int smem_m = tile.tid / (2 * Traits::QUANT_GROUP_BLOCKS);
    const int half = tile.tid & 1;
    const int local_m = row_pass_base + smem_m;
    const int local_col_base = half * Traits::HALF_SCALE_GROUP;
    const int route_row = tile.route_base + local_m;

    const auto route = route_from_smem(local_m, smem_route);
    if(!route.valid)
        return;

    const int group_base =
        ((tile.tid >> 1) % Traits::QUANT_GROUP_BLOCKS) *
        Traits::QUANT_GROUPS_PER_THREAD;
    constexpr int kWordsPerHalfGroup =
        Traits::HALF_SCALE_GROUP / static_cast<int>(sizeof(uint32_t));
    opus::static_for<Traits::QUANT_GROUPS_PER_THREAD>([&](auto local_group_id) {
        constexpr int local_group = local_group_id.value;
        const int group = group_base + local_group;
        const int group_col_base =
            tile.out_col_base + group * Traits::SCALE_GROUP_LOGICAL_K;
        float values[Traits::HALF_SCALE_GROUP];
        float local_amax = 0.0f;

        opus::static_for<Traits::HALF_SCALE_GROUP>([&](auto local_col_id) {
            constexpr int local_col = local_col_id.value;
            const float value =
                smem_values[epilogue_smem_offset<Traits>(
                    smem_m, group, local_col_base + local_col)];
            values[local_col] = value;
            const float abs_value = __builtin_fabsf(value);
            local_amax = abs_value > local_amax ? abs_value : local_amax;
        });

        const int peer_lane = (tile.tid % opus::get_warp_size()) ^ 1;
        const float amax = opus::max(local_amax, opus::shfl(local_amax, peer_lane));
        const auto block_scale =
            aiter::fp_f32_to_e8m0_block_scale<aiter::kDefaultMxScaleRoundMode,
                                              aiter::MxDtype::FP8_E4M3>(amax);
        const uint8_t scale_byte = block_scale.byte;
        const float quant_scale = __builtin_bit_cast(
            float, (254u - static_cast<uint32_t>(scale_byte)) << 23);
        const int scale_col = group_col_base / Traits::SCALE_GROUP_LOGICAL_K;
        if(half == 0)
        {
            const int scale_offset = aiter::mx_scale_shuffle_idx(
                static_cast<int>(kargs.stride_out_scale_route),
                route_row,
                scale_col);
            g_out_scale.template store<1>(scale_byte, scale_offset);
        }

        stage1_u32x4_t packed{};
        opus::static_for<kWordsPerHalfGroup>([&](auto word_id) {
            constexpr int word = word_id.value;
            constexpr int base = word * static_cast<int>(sizeof(uint32_t));
            int packed_word = __builtin_amdgcn_cvt_pk_fp8_f32(
                values[base + 0] * quant_scale,
                values[base + 1] * quant_scale,
                0,
                0);
            packed_word = __builtin_amdgcn_cvt_pk_fp8_f32(
                values[base + 2] * quant_scale,
                values[base + 3] * quant_scale,
                packed_word,
                1);
            packed[word] = static_cast<uint32_t>(packed_word);
        });

        const int64_t output_offset =
            static_cast<int64_t>(route.token) * kargs.stride_out_t +
            static_cast<int64_t>(route.slot) * kargs.stride_out_k +
            group_col_base;
        const int output_word_offset =
            static_cast<int>(
                (output_offset + local_col_base) /
                static_cast<int64_t>(sizeof(uint32_t)));
        g_out.template store<4>(packed, output_word_offset);
    });
}

template<typename Traits,
         typename CAcc,
         typename CLayoutM,
         typename CLayoutN,
         typename OutGmem,
         typename OutScaleGmem>
inline __device__ void quant_epilogue(
    const OpusMoeStage1A8W4Kargs& kargs,
    OutGmem& g_out,
    OutScaleGmem& g_out_scale,
    const Tile& tile,
    int expert_id,
    const CLayoutM& u_c_m,
    const CLayoutN& u_c_n,
    CAcc (&rc)[Traits::M_MFMA_PER_WAVE]
             [Traits::ACC_SCALE_GROUPS_PER_TILE],
    const int* __restrict__ smem_route,
    float* __restrict__ smem_gate_up)
{
    opus::static_for<Traits::EPILOGUE_ROW_SPLIT>([&](auto row_pass_id) {
        constexpr int row_pass = row_pass_id.value;
        constexpr int row_pass_base =
            row_pass * Traits::EPILOGUE_ROWS_PER_PASS;
        epilogue_store_row_pass<Traits, row_pass, CAcc>(
            kargs, smem_gate_up, rc, u_c_m, u_c_n, tile, expert_id);
        opus::sync_threads();
        epilogue_quantize_row_pass<Traits>(
            kargs, g_out, g_out_scale, tile, row_pass_base,
            smem_route, smem_gate_up);

        if constexpr(row_pass + 1 < Traits::EPILOGUE_ROW_SPLIT)
            opus::sync_threads();
    });
}

template<typename T, typename Mma>
inline __device__ auto make_layout_ga_stage1(Mma& mma,
                                             int lane_id,
                                             int stride_a)
{
    const auto p_coord_a = opus::make_tuple(
        0,
        lane_id % Mma::grpm_a,
        0,
        lane_id / Mma::grpm_a);
    return opus::partition_layout_a<T::BYTES_PER_VEC>(
        mma,
        opus::make_tuple(stride_a, opus::number<1>{}),
        p_coord_a);
}

template<typename T>
inline __device__ auto make_layout_sa_stage1(int lane_id, int wave_id_m)
{
    constexpr int threads_k = T::MMA_K / T::BYTES_PER_VEC;
    constexpr int threads_m_per_wave = opus::get_warp_size() / threads_k;
    static_assert(T::MMA_K % T::BYTES_PER_VEC == 0);
    static_assert(T::MMA_M % threads_m_per_wave == 0);

    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::M_MFMA_PER_WAVE>{},
        opus::number<1>{},
        opus::number<T::MMA_M / threads_m_per_wave>{},
        opus::number<threads_m_per_wave>{},
        opus::number<threads_k>{},
        opus::number<T::BYTES_PER_VEC>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{},
                         opus::p_dim{},
                         opus::y_dim{},
                         opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::BYTES_PER_VEC>(
        block_shape,
        opus::unfold_x_stride(
            block_dim,
            block_shape,
            opus::tuple{
                opus::number<T::MMA_K>{},
                opus::number<1>{}}),
        opus::unfold_p_coord(
            block_dim,
            opus::tuple{
                wave_id_m, lane_id / threads_k, lane_id % threads_k}));
}

template<typename T>
inline __device__ auto make_layout_ra_stage1(int lane_id)
{
    constexpr int threads_k = T::MMA_K / T::BYTES_PER_VEC;
    constexpr int threads_m_per_wave = opus::get_warp_size() / threads_k;
    static_assert(T::MMA_K % T::BYTES_PER_VEC == 0);
    static_assert(T::MMA_M % threads_m_per_wave == 0);

    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::MMA_M / threads_m_per_wave>{},
        opus::number<threads_m_per_wave>{},
        opus::number<threads_k>{},
        opus::number<T::BYTES_PER_VEC>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::BYTES_PER_VEC>(
        block_shape,
        opus::unfold_x_stride(
            block_dim,
            block_shape,
            opus::tuple{
                opus::number<T::MMA_K>{},
                opus::number<1>{}}),
        opus::unfold_p_coord(
            block_dim,
            opus::tuple{lane_id / threads_k, lane_id % threads_k}));
}

template<typename T, typename Mma>
inline __device__ auto make_layout_gb_stage1(Mma& mma, int lane_id)
{
    const auto p_coord_b = opus::make_tuple(
        0,
        lane_id / Mma::grpn_b,
        0,
        lane_id % Mma::grpn_b);
    return opus::partition_layout_b<T::BYTES_PER_VEC>(
        mma,
        opus::make_tuple(
            opus::number<T::MMA_N * T::BYTES_PER_VEC>{},
            opus::number<1>{}),
        p_coord_b);
}

template<typename T>
__device__ __forceinline__ int scale_layout_stride_n0_stage1(
    int hidden_scale_cols)
{
    return ((hidden_scale_cols / 4) / T::SCALE_K_PACK) *
           T::SCALE_LAYOUT_STRIDE_K0;
}

template<typename T>
inline __device__ auto make_layout_scale_word_stage1(int scale_lane_byte,
                                                     int stride_n0)
{
    const int scale_lane = scale_lane_byte / T::BYTES_PER_VEC;
    const int lane_k = scale_lane / T::MMA_N;
    const int lane_m = scale_lane & (T::MMA_N - 1);

    constexpr auto block_shape = opus::make_tuple(
        opus::number<1>{},
        opus::number<T::MMA_N>{},
        opus::number<T::MMA_M>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}));

    return opus::make_layout(
        block_shape,
        opus::unfold_x_stride(
            block_dim,
            block_shape,
            opus::tuple{stride_n0,
                        opus::number<T::MMA_M>{},
                        opus::number<1>{}}),
        opus::unfold_p_coord(
            block_dim,
            opus::tuple{lane_k, lane_m}));
}

template<typename T, typename Mma>
inline __device__ auto make_layout_c_m_stage1(Mma& mma, int lane_id)
{
    const auto p_coord_c = opus::make_tuple(
        0,
        lane_id / Mma::grpn_c,
        0,
        lane_id % Mma::grpn_c);
    return opus::partition_layout_c<1>(
        mma,
        opus::make_tuple(opus::number<1>{}, opus::number<0>{}),
        p_coord_c);
}

template<typename T, typename Mma>
inline __device__ auto make_layout_c_n_stage1(Mma& mma, int lane_id)
{
    const auto p_coord_c = opus::make_tuple(
        0,
        lane_id / Mma::grpn_c,
        0,
        lane_id % Mma::grpn_c);
    return opus::partition_layout_c<1>(
        mma,
        opus::make_tuple(opus::number<0>{}, opus::number<1>{}),
        p_coord_c);
}

#endif // __HIP_DEVICE_COMPILE__

} // namespace pipeline_common
} // namespace stage1_a8w4
} // namespace opus_moe
