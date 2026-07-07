// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_pipeline_stage2_a8w4_decode_policy_gfx950.cuh"
#include "../opus_moe_stage2_utils_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#include "opus/opus.hpp"
#endif

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)

// Mainloop: A/B/scale loads and MFMA accumulation.
using opus_moe_stage2_a8w4_decode_u32x4_t = opus::vector_t<uint32_t, 4>;

template<typename Reg>
inline __device__ void opus_moe_stage2_a8w4_decode_pack_a_mfma_reg(
    opus_moe_stage2_a8w4_decode_u32x4_t lo,
    opus_moe_stage2_a8w4_decode_u32x4_t hi,
    Reg& reg)
{
    const auto packed = opus::concat_vector(lo, hi);
    reg = __builtin_bit_cast(opus::remove_cvref_t<Reg>, packed);
}

template<typename Reg>
inline __device__ void opus_moe_stage2_a8w4_decode_unpack_b_mfma_reg(
    opus_moe_stage2_a8w4_decode_u32x4_t value,
    Reg& reg)
{
    const opus_moe_stage2_a8w4_decode_u32x4_t zero{};
    const auto packed = opus::concat_vector(value, zero);
    reg = __builtin_bit_cast(opus::remove_cvref_t<Reg>, packed);
}

template<typename D_A, int StageElems, int... Stages>
inline __device__ auto opus_moe_stage2_a8w4_decode_make_smem_a_stages(
    char* smem_scratch,
    opus::seq<Stages...>)
{
    return opus::make_array(
        opus::make_smem(reinterpret_cast<D_A*>(
            smem_scratch + Stages * StageElems * static_cast<int>(sizeof(D_A))))...);
}

// A payload flat helpers; caller owns layouts/memory state and init fills base offsets.
template<typename T, typename LayoutA>
inline __device__ void opus_moe_stage2_a8w4_decode_init_a_payload(
    const LayoutA& u_ga,
    const int32_t* __restrict__ smem_a_base,
    int route_base,
    int (&a_base)[T::M_MFMA_PER_WAVE],
    int (&a_scale_base_word)[T::M_MFMA_PER_WAVE])
{
    opus::static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
        const int ga = static_cast<int>(u_ga(mi, opus::number<0>{}));
        const int local_m = opus_moe_stage2_a8w4_a_local_m<T>(ga);
        a_base[mi.value] = smem_a_base[local_m];
        a_scale_base_word[mi.value] =
            opus_moe_stage2_a8w4_a_scale_base_word_offset<T>(route_base, ga);
    });
}

template<typename T,
         typename Stage,
         typename Mi,
         typename LayoutA,
         typename LayoutASmem,
         typename SmemA,
         typename GmemA>
inline __device__ void opus_moe_stage2_a8w4_decode_issue_one_a(
    const LayoutA& u_ga,
    const LayoutASmem& u_sa,
    SmemA& s_a,
    GmemA& g_a,
    const int (&a_base)[T::M_MFMA_PER_WAVE],
    int wave_id_n,
    Stage stage,
    Mi mi,
    int k_base)
{
    constexpr int StageValue = decltype(stage)::value;
    static_assert(StageValue >= 0 && StageValue < T::A_LDS_STAGES);

    using Schedule = OpusMoeStage2A8W4DecodeSchedule<T>;
    using MainloopSchedule = OpusMoeStage2A8W4DecodeMainloopSchedule;

    const int ga = static_cast<int>(u_ga(mi, opus::number<0>{}));
    auto* smem_lo =
        s_a[StageValue].ptr + static_cast<int>(u_sa(mi, opus::number<0>{}, opus::number<0>{}));
    auto* smem_hi =
        s_a[StageValue].ptr + static_cast<int>(u_sa(mi, opus::number<1>{}, opus::number<0>{}));
    const int a_offset_lo =
        a_base[mi.value] + k_base + opus_moe_stage2_a8w4_a_k_byte<T>(ga);
    auto issue_async = [&](auto* smem, int offset) {
        g_a.template async_load<T::VEC_A>(
            reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(smem)),
            offset,
            0,
            opus::number<T::CACHECTL_A>{});
    };
    if constexpr(Schedule::Mainloop == MainloopSchedule::SplitALoadByNWave)
    {
        if(wave_id_n == 0)
            issue_async(smem_lo, a_offset_lo);
        else
            issue_async(smem_hi, a_offset_lo + T::K_STEP_PACKED / 2);
    }
    else
    {
        issue_async(smem_lo, a_offset_lo);
        issue_async(smem_hi, a_offset_lo + T::K_STEP_PACKED / 2);
    }
}

template<typename T,
         typename Stage,
         typename LayoutA,
         typename LayoutASmem,
         typename SmemA,
         typename GmemA>
inline __device__ void opus_moe_stage2_a8w4_decode_issue_a(
    const LayoutA& u_ga,
    const LayoutASmem& u_sa,
    SmemA& s_a,
    GmemA& g_a,
    const int (&a_base)[T::M_MFMA_PER_WAVE],
    int wave_id_n,
    Stage stage,
    int k_base)
{
    using Schedule = OpusMoeStage2A8W4DecodeSchedule<T>;
    using MainloopSchedule = OpusMoeStage2A8W4DecodeMainloopSchedule;

    if(wave_id_n > 1)
        return;

    if constexpr(Schedule::Mainloop == MainloopSchedule::SplitALoadByNWave)
    {
        opus_moe_stage2_a8w4_decode_issue_one_a<T>(
            u_ga, u_sa, s_a, g_a, a_base, wave_id_n, stage, opus::number<0>{}, k_base);
    }
    else
    {
        // M_MFMA_PER_WAVE is 1 (bm32/bm16) or 4 (bm64) for all live shapes.
        opus::static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
            if((mi.value & 1) == wave_id_n)
                opus_moe_stage2_a8w4_decode_issue_one_a<T>(
                    u_ga, u_sa, s_a, g_a, a_base, wave_id_n, stage, mi, k_base);
        });
    }
}

template<typename PendingALoads>
inline __device__ void opus_moe_stage2_a8w4_decode_wait_a(
    int wave_id_n, PendingALoads pending_a_loads)
{
    if(wave_id_n <= 1)
    {
        opus::s_waitcnt_vmcnt(pending_a_loads);
    }
    opus::sync_threads();
}

template<typename T,
         typename Stage,
         typename LayoutASmem,
         typename SmemA,
         typename V_A>
inline __device__ void opus_moe_stage2_a8w4_decode_load_a(
    const LayoutASmem& u_sa,
    SmemA& s_a,
    Stage stage,
    V_A (&v_a)[T::M_MFMA_PER_WAVE])
{
    constexpr int StageValue = decltype(stage)::value;
    static_assert(StageValue >= 0 && StageValue < T::A_LDS_STAGES);

    opus::static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
        if constexpr(T::K_TILES == 3)
        {
            auto packed = s_a[StageValue].template load<T::VEC_A>(
                opus_moe_stage2_a8w4_layout_ra<T>(
                    static_cast<int>(u_sa(mi, opus::number<0>{}, opus::number<0>{}))));
            v_a[mi.value] = __builtin_bit_cast(V_A, packed);
        }
        else
        {
            auto lo = s_a[StageValue].template load<T::VEC_A>(
                static_cast<int>(u_sa(mi, opus::number<0>{}, opus::number<0>{})));
            auto hi = s_a[StageValue].template load<T::VEC_A>(
                static_cast<int>(u_sa(mi, opus::number<1>{}, opus::number<0>{})));
            opus_moe_stage2_a8w4_decode_pack_a_mfma_reg(
                __builtin_bit_cast(opus_moe_stage2_a8w4_decode_u32x4_t, lo),
                __builtin_bit_cast(opus_moe_stage2_a8w4_decode_u32x4_t, hi),
                v_a[mi.value]);
        }
    });
}

// B FP4x2 tile load for one N-half; caller precomputes lane_offset and wave_scalar_base.
template<typename T, typename NHalf, typename GmemB, typename V_B>
inline __device__ void opus_moe_stage2_a8w4_decode_load_b_half(
    GmemB& g_b,
    int lane_offset,
    int wave_scalar_base,
    NHalf n_half,
    int tile_base,
    V_B (&v_b)[T::HALF_N_MFMA_PER_WAVE])
{
    constexpr int NHalfValue = decltype(n_half)::value;
    opus::static_for<T::HALF_N_MFMA_PER_WAVE>([&](auto local_ni) {
        const auto rb_offset =
            opus_moe_stage2_a8w4_layout_rb_half<T, NHalfValue>(
                lane_offset, local_ni);
        const int b_scalar_offset =
            tile_base + wave_scalar_base + opus::get<1>(rb_offset);
        auto value = g_b.template load<T::B_BYTES_PER_VEC>(
            opus::get<0>(rb_offset),
            b_scalar_offset,
            opus::number<T::CACHECTL_B>{});
        opus_moe_stage2_a8w4_decode_unpack_b_mfma_reg(
            __builtin_bit_cast(opus_moe_stage2_a8w4_decode_u32x4_t, value),
            v_b[local_ni.value]);
    });
}

// A/W MXFP8 scale word loads; caller precomputes A and B scale base words.
template<typename T, typename GmemAScale>
inline __device__ void opus_moe_stage2_a8w4_decode_load_a_scale(
    GmemAScale& g_a_scale,
    const int* __restrict__ a_scale_base_word,
    int k_group_word_base,
    int (&a_scale)[T::M_MFMA_PER_WAVE])
{
    opus::static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
        const int word_offset = a_scale_base_word[mi.value] + k_group_word_base;
        const auto word = g_a_scale.template load<sizeof(uint32_t)>(
            word_offset * static_cast<int>(sizeof(uint32_t)),
            0,
            opus::number<T::CACHECTL_A>{});
        a_scale[mi.value] = static_cast<int>(__builtin_bit_cast(uint32_t, word));
    });
}

template<typename T, typename GmemWScale>
inline __device__ void opus_moe_stage2_a8w4_decode_load_b_scale(
    GmemWScale& g_w_scale,
    int b_scale_base_word,
    int k_group_word_base,
    int (&b_scale)[T::HALF_N_MFMA_PER_WAVE])
{
    opus::static_for<T::HALF_N_MFMA_PER_WAVE>([&](auto pair) {
        const int word_offset =
            b_scale_base_word + k_group_word_base +
            pair.value * T::SCALE_WORDS_PER_ROW_PACK;
        const auto word = g_w_scale.template load<sizeof(uint32_t)>(
            word_offset * static_cast<int>(sizeof(uint32_t)),
            0,
            opus::number<T::CACHECTL_W_SCALE>{});
        b_scale[pair.value] = static_cast<int>(__builtin_bit_cast(uint32_t, word));
    });
}

// Scaled tiled MFMA over one N-half of the wave tile.
template<typename T,
         typename Mma,
         typename ScalePair,
         typename NHalf,
         typename V_A,
         typename V_B>
inline __device__ void opus_moe_stage2_a8w4_decode_compute_scaled_half(
    Mma& mma,
    int wave_id_m,
    typename Mma::vtype_c (&v_c)[T::M_MFMA_PER_WAVE][T::N_MFMA_PER_WAVE],
    ScalePair scale_pair,
    NHalf n_half,
    const V_A (&v_a)[T::M_MFMA_PER_WAVE],
    const int (&a_scale)[T::M_MFMA_PER_WAVE],
    const int (&b_scale)[T::HALF_N_MFMA_PER_WAVE],
    const V_B (&v_b)[T::HALF_N_MFMA_PER_WAVE])
{
    using namespace opus;

    constexpr int ScalePairValue = decltype(scale_pair)::value;
    constexpr int NHalfValue = decltype(n_half)::value;
    static_assert(ScalePairValue == 0 || ScalePairValue == 1);
    static_assert(NHalfValue == 0 || NHalfValue == 1);

    // Half-N sub-tile: expand M/N only; tile dims are unused by for_each_sub.
    auto tiled = make_tiled_mma(mma,
                                number<T::M_MFMA_PER_WAVE>{},
                                number<T::HALF_N_MFMA_PER_WAVE>{},
                                number<1>{},
                                number<1>{},
                                number<1>{},
                                number<1>{});
    tiled.for_each_sub(v_a, v_b,
        [&](auto i_m, auto i_n, auto, const auto& s_a, const auto& s_b) {
            constexpr int mi = i_m.value;
            constexpr int ni = NHalfValue * T::HALF_N_MFMA_PER_WAVE + i_n.value;
            constexpr int b_sel = ScalePairValue * 2 + (ni & 1);
            constexpr int b_scale_index = ni / 2;
            if constexpr(T::M_MFMA_PER_WAVE == 1 && T::T_M == 2)
            {
                constexpr int a_sel_base = ScalePairValue * 2;
                if(wave_id_m == 0)
                    v_c[mi][ni] = mma(s_a, s_b, v_c[mi][ni], a_scale[mi],
                                      b_scale[b_scale_index],
                                      number<a_sel_base>{}, number<b_sel>{});
                else
                    v_c[mi][ni] = mma(s_a, s_b, v_c[mi][ni], a_scale[mi],
                                      b_scale[b_scale_index],
                                      number<a_sel_base + 1>{}, number<b_sel>{});
            }
            else
            {
                constexpr int a_sel = ScalePairValue * 2 + (mi & 1);
                v_c[mi][ni] = mma(s_a, s_b, v_c[mi][ni], a_scale[mi],
                                  b_scale[b_scale_index],
                                  number<a_sel>{}, number<b_sel>{});
            }
        });
}

// K5 generic schedule is included after the shared helpers it calls.
#include "opus_moe_pipeline_stage2_a8w4_decode_k5_gfx950.cuh"

// Tile mapping: baseline m-fast by default, with optional SWIZZLE_C windowed XCD swizzle.
template<typename T>
inline __device__ void opus_moe_stage2_a8w4_decode_tile_ids(int wgid,
                                                            int route_blocks,
                                                            int num_tiles_n,
                                                            int& tile_m_id,
                                                            int& tile_n_id)
{
    const int num_tiles_m = route_blocks;

    if constexpr(T::NUM_XCD > 1)
    {
        // Windowed XCD swizzle (W/C): partial w2 L2 reuse while keeping XCDs interleaved (preserves MLP).
        constexpr int nXCD = T::NUM_XCD;
        constexpr int W = T::SWIZZLE_W;
        constexpr int C = T::SWIZZLE_C;
        if(C > 0)
        {
            const int total_wgs = num_tiles_m * num_tiles_n;
            const int blocks_per_cycle = nXCD * C;
            const int tiles_per_group = W * num_tiles_n;
            const int limit = (total_wgs / blocks_per_cycle) * blocks_per_cycle;
            if(wgid >= limit)
            {
                const int full_groups = limit / tiles_per_group;
                const int covered_cols = (limit - full_groups * tiles_per_group) / W;
                const int partial_first_row = full_groups * W;
                int partial_row_extent = num_tiles_m - partial_first_row;
                if(partial_row_extent > W)
                    partial_row_extent = W;
                const int tail = wgid - limit;
                const int partial_tiles =
                    (partial_row_extent > 0) ? (num_tiles_n - covered_cols) * partial_row_extent
                                             : 0;
                if(tail < partial_tiles)
                {
                    tile_m_id = partial_first_row + (tail % partial_row_extent);
                    tile_n_id = covered_cols + (tail / partial_row_extent);
                    return;
                }
                const int rest = tail - partial_tiles;
                tile_m_id = partial_first_row + partial_row_extent + rest / num_tiles_n;
                tile_n_id = rest % num_tiles_n;
                return;
            }

            const int xcd = wgid % nXCD;
            const int local = wgid / nXCD;
            const int chunk_idx = local / C;
            const int pos = local % C;
            const int swizzled = xcd * C + chunk_idx * blocks_per_cycle + pos;
            const int group_id = swizzled / tiles_per_group;
            const int first_row = group_id * W;
            int win_h = num_tiles_m - first_row;
            if(win_h > W)
                win_h = W;
            const int in_group = swizzled % tiles_per_group;
            tile_m_id = first_row + (in_group % win_h);
            tile_n_id = in_group / win_h;
            if(tile_n_id < num_tiles_n)
                return;
        } // if(C > 0)
    }

    // Baseline m-fast mapping: NUM_XCD<=1 (atomic) or traits SWIZZLE_C<=0.
    tile_n_id = wgid / route_blocks;
    tile_m_id = wgid - tile_n_id * route_blocks;
}

template<typename T>
inline __device__ void opus_moe_stage2_a8w4_decode_make_tile(
    const opus_moe_stage2_a8w4_kargs& kargs,
    int& sorted_rows,
    int& route_base,
    int& col_base)
{
    constexpr int BM = T::B_M;
    constexpr int BN = T::B_N;

    sorted_rows = kargs.num_valid_ids[0];
    int tile_m_id;
    int tile_n_id;
    const int route_blocks = kargs.sorted_blocks;
    if constexpr(T::DIRECT_ATOMIC_OUT)
    {
        tile_n_id = static_cast<int>(opus::block_id_x());
        tile_m_id = static_cast<int>(opus::block_id_y());
    }
    else
    {
        const int num_tiles_n = kargs.model_dim / BN;
        const int wgid = static_cast<int>(opus::block_id_y()) * num_tiles_n +
                         static_cast<int>(opus::block_id_x());
        opus_moe_stage2_a8w4_decode_tile_ids<T>(
            wgid, route_blocks, num_tiles_n, tile_m_id, tile_n_id);
    }

    route_base = tile_m_id * T::ROUTE_M_STRIDE;
    col_base = tile_n_id * BN;
}

template<typename T>
inline __device__ bool opus_moe_stage2_a8w4_decode_load_route_metadata(
    const opus_moe_stage2_a8w4_kargs& kargs,
    int route_base,
    int sorted_rows,
    int tid,
    int32_t* __restrict__ smem_a_base,
    int32_t* __restrict__ smem_route_base,
    float* __restrict__ smem_weight)
{
    const int token_num = kargs.token_num;
    const int topk = kargs.topk;
    const bool has_sorted_weights = kargs.sorted_weights != nullptr;
    const int stride_a_t = static_cast<int>(kargs.stride_a_t);
    const int stride_a_k = static_cast<int>(kargs.stride_a_k);
    int has_route = 0;
    for(int local_m = tid; local_m < T::B_M; local_m += T::BLOCK_SIZE)
    {
        const int row = route_base + local_m;
        int32_t a_base = 0;
        int32_t route_row = -1;
        float weight = 0.0f;
        if(row < sorted_rows)
        {
            const int32_t packed = kargs.sorted_token_ids[row];
            const int token = opus_moe_token_id(packed);
            const int slot = opus_moe_topk_slot(packed);
            const bool valid_route = token < token_num && slot < topk;
            if(valid_route)
            {
                a_base = token * stride_a_t + slot * stride_a_k;
                weight = has_sorted_weights ? kargs.sorted_weights[row] : 1.0f;
                if constexpr(T::DIRECT_ATOMIC_OUT)
                    route_row = static_cast<int32_t>(token);
                else
                {
                    route_row = static_cast<int32_t>(token * topk + slot);
                    has_route = 1;
                }
            }
        }

        smem_a_base[local_m] = a_base;
        smem_route_base[local_m] = route_row;
        smem_weight[local_m] = weight;
    }
    if constexpr(T::DIRECT_ATOMIC_OUT)
    {
        opus::sync_threads();
        return true;
    }
    else
    {
        if constexpr(T::IS_BM32_BN256)
        {
            // Fast path for full sorted tiles. Store-side route_row guards are
            // still kept so malformed metadata cannot write a negative route row.
            if(route_base + T::B_M <= sorted_rows)
            {
                opus::sync_threads();
                return true;
            }
        }
        const int route_count = opus::sync_threads_count(has_route);
        return route_count != 0;
    }
}

// Epilogue: direct atomic output or route-out store.
using opus_moe_stage2_a8w4_decode_u32x4_store_t = opus::vector_t<uint32_t, 4>;

struct OpusMoeStage2A8W4DecodeCShuffleSmem
{
    uint32_t* __restrict__ pair;

    inline __device__ void store_bf16(int scalar_idx, hip_bfloat16 value)
    {
        reinterpret_cast<hip_bfloat16*>(pair)[scalar_idx] = value;
    }

    inline __device__ uint32_t load_pair(int pair_idx)
    {
        return pair[pair_idx];
    }

    inline __device__ opus_moe_stage2_a8w4_decode_u32x4_store_t load_pair4(
        int pair_idx)
    {
        return {pair[pair_idx + 0],
                pair[pair_idx + 1],
                pair[pair_idx + 2],
                pair[pair_idx + 3]};
    }
};

template<typename T, typename CAcc>
inline __device__ void opus_moe_stage2_a8w4_decode_write_acc_to_smem(
    CAcc (&v_c)[T::M_MFMA_PER_WAVE][T::N_MFMA_PER_WAVE],
    const OpusMoeStage2A8W4CShuffleLayout<T>& c_layout,
    const float* __restrict__ smem_weight,
    OpusMoeStage2A8W4DecodeCShuffleSmem& c_smem)
{
    using namespace opus;

    static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
        static_for<T::VEC_C>([&](auto ii) {
            const int local_m = c_layout.acc_local_m(mi.value, ii.value);
            const float weight = smem_weight[local_m];
            static_for<T::N_MFMA_PER_WAVE>([&](auto ni) {
                const int local_col = c_layout.acc_local_col(ni.value);
                c_smem.store_bf16(
                    c_layout.smem_scalar_index(local_m, local_col),
                    opus_moe_gfx950_cvt_bf16_f32(
                        static_cast<float>(v_c[mi.value][ni.value][ii.value]) *
                        weight));
            });
        });
    });
}

template<typename T, typename CAcc>
inline __device__ void opus_moe_stage2_a8w4_decode_store_acc_to_cshuffle(
    CAcc (&v_c)[T::M_MFMA_PER_WAVE][T::N_MFMA_PER_WAVE],
    const OpusMoeStage2A8W4CShuffleLayout<T>& c_layout,
    const float* __restrict__ smem_weight,
    uint32_t* __restrict__ smem_c_pair)
{
    OpusMoeStage2A8W4DecodeCShuffleSmem c_smem{smem_c_pair};
    opus_moe_stage2_a8w4_decode_write_acc_to_smem<T>(
        v_c,
        c_layout,
        smem_weight,
        c_smem);
    opus::s_waitcnt_lgkmcnt(opus::number<0>{});
    opus::sync_threads();
}

template<typename T, typename OutputGmem>
inline __device__ void opus_moe_stage2_a8w4_decode_atomic_smem_to_out(
    uint32_t* __restrict__ smem_c_pair,
    const int32_t* __restrict__ smem_route_base,
    const OpusMoeStage2A8W4CShuffleLayout<T>& c_layout,
    int col_base,
    int64_t output_row_stride,
    OutputGmem& output_gmem)
{
    static_assert(T::B_N == T::C_LDS_N);
    static_assert(T::DIRECT_ATOMIC_OUT);

    constexpr int CSHUFFLE_NLANE =
        OpusMoeStage2A8W4CShuffleLayout<T>::CSHUFFLE_NLANE;
    constexpr int CSHUFFLE_MLANE =
        OpusMoeStage2A8W4CShuffleLayout<T>::CSHUFFLE_MLANE;
    constexpr int PAIRS_PER_ROW =
        OpusMoeStage2A8W4CShuffleLayout<T>::PAIRS_PER_ROW;
    constexpr int ATOMIC_GROUPS = PAIRS_PER_ROW / CSHUFFLE_NLANE;
    static_assert(T::BLOCK_SIZE % CSHUFFLE_NLANE == 0);
    static_assert(T::B_M % CSHUFFLE_MLANE == 0);
    static_assert((PAIRS_PER_ROW & (PAIRS_PER_ROW - 1)) == 0);
    static_assert(PAIRS_PER_ROW % CSHUFFLE_NLANE == 0);

    OpusMoeStage2A8W4DecodeCShuffleSmem c_smem{smem_c_pair};
    const int col0 = c_layout.atomic_col0();

    #pragma unroll
    for(int mr = 0; mr < T::B_M / CSHUFFLE_MLANE; ++mr)
    {
        const int local_m = c_layout.atomic_local_m(mr);
        if(smem_route_base[local_m] >= 0)
        {
            const int token = smem_route_base[local_m];
            const int pair_base = c_layout.smem_pair_index(local_m, col0);
            const int elem_offset = static_cast<int>(
                static_cast<int64_t>(token) * output_row_stride + col_base + col0);
            #pragma unroll
            for(int group = 0; group < ATOMIC_GROUPS; ++group)
            {
                const int pair_delta = group * CSHUFFLE_NLANE;
                const int elem_delta = pair_delta * T::ELEM_PER_ATOMIC;
                const opus::bf16x2_t data = __builtin_bit_cast(
                    opus::bf16x2_t, c_smem.load_pair(pair_base + pair_delta));
                output_gmem.template atomic_add<2>(
                    data,
                    elem_offset + elem_delta);
            }
        }
    }
}

template<typename T>
struct OpusMoeStage2A8W4DecodeRouteOutTile
{
    static constexpr int PAIRS_PER_ROW =
        OpusMoeStage2A8W4CShuffleLayout<T>::PAIRS_PER_ROW;
    static constexpr int PAIRS_PER_VECTOR = 4;
    static constexpr int THREADS_PER_ROW = PAIRS_PER_ROW / PAIRS_PER_VECTOR;
    static constexpr int ROWS_PER_ITER = T::BLOCK_SIZE / THREADS_PER_ROW;
    static constexpr int ROW_ITERS = T::B_M / ROWS_PER_ITER;
    static_assert(PAIRS_PER_ROW % PAIRS_PER_VECTOR == 0);
    static_assert(T::BLOCK_SIZE % THREADS_PER_ROW == 0);
    static_assert(T::B_M % ROWS_PER_ITER == 0);

    int row_in_iter;
    int col0;

    inline __device__ explicit OpusMoeStage2A8W4DecodeRouteOutTile(
        const OpusMoeStage2A8W4CShuffleLayout<T>& c_layout)
    {
        row_in_iter = c_layout.tid / THREADS_PER_ROW;
        const int lane_in_row = c_layout.tid - row_in_iter * THREADS_PER_ROW;
        const int pair_col = lane_in_row * PAIRS_PER_VECTOR;
        col0 = pair_col * T::ELEM_PER_ATOMIC;
    }

    inline __device__ int local_m(int row_iter) const
    {
        return row_iter * ROWS_PER_ITER + row_in_iter;
    }
};

template<typename T, typename StoreGroup>
inline __device__ void opus_moe_stage2_a8w4_decode_store_route_out_groups(
    uint32_t* __restrict__ smem_c_pair,
    const int32_t* __restrict__ smem_route_base,
    const OpusMoeStage2A8W4CShuffleLayout<T>& c_layout,
    StoreGroup&& store_group)
{
    static_assert(T::B_N == T::C_LDS_N);
    static_assert(!T::DIRECT_ATOMIC_OUT);

    OpusMoeStage2A8W4DecodeCShuffleSmem c_smem{smem_c_pair};
    const OpusMoeStage2A8W4DecodeRouteOutTile<T> route_tile(c_layout);

    #pragma unroll
    for(int row_iter = 0;
        row_iter < OpusMoeStage2A8W4DecodeRouteOutTile<T>::ROW_ITERS;
        ++row_iter)
    {
        const int local_m = route_tile.local_m(row_iter);
        const int route_row = smem_route_base[local_m];
        if(route_row >= 0)
        {
            const int pair_base =
                c_layout.smem_pair_index(local_m, route_tile.col0);
            store_group(route_row, route_tile.col0, c_smem.load_pair4(pair_base));
        }
    }
}

template<typename T>
inline __device__ void opus_moe_stage2_a8w4_decode_store_smem_to_route_out(
    uint32_t* __restrict__ smem_c_pair,
    const int32_t* __restrict__ smem_route_base,
    const OpusMoeStage2A8W4CShuffleLayout<T>& c_layout,
    int col_base,
    hip_bfloat16* __restrict__ out,
    int64_t output_row_stride)
{
    opus_moe_stage2_a8w4_decode_store_route_out_groups<T>(
        smem_c_pair,
        smem_route_base,
        c_layout,
        [&](int route_row,
            int col0,
            opus_moe_stage2_a8w4_decode_u32x4_store_t pairs) {
            hip_bfloat16* row_ptr =
                out + static_cast<int64_t>(route_row) * output_row_stride +
                col_base + col0;
            auto* row_pair = reinterpret_cast<uint32_t*>(row_ptr);
            __builtin_nontemporal_store(
                pairs,
                reinterpret_cast<opus_moe_stage2_a8w4_decode_u32x4_store_t*>(
                    row_pair));
        });
}

struct OpusMoeStage2A8W4DecodeMxFp8Group
{
    opus::u8x8_t data;
    opus::u8_t e8m0;
};

inline __device__ uint32_t opus_moe_stage2_a8w4_decode_bf16_pair4_amax_bits(
    opus_moe_stage2_a8w4_decode_u32x4_store_t pairs)
{
    uint32_t amax_bits = 0; // 15-bit bf16 magnitude
    amax_bits = max(amax_bits, max(pairs[0] & 0x7fffu, (pairs[0] >> 16) & 0x7fffu));
    amax_bits = max(amax_bits, max(pairs[1] & 0x7fffu, (pairs[1] >> 16) & 0x7fffu));
    amax_bits = max(amax_bits, max(pairs[2] & 0x7fffu, (pairs[2] >> 16) & 0x7fffu));
    amax_bits = max(amax_bits, max(pairs[3] & 0x7fffu, (pairs[3] >> 16) & 0x7fffu));
    return amax_bits;
}

inline __device__ int opus_moe_stage2_a8w4_decode_mxfp8_e8m0_from_amax_bits(
    uint32_t amax_bits)
{
    const int ax_e = (amax_bits >> 7) & 0xff; // biased bf16 exponent
    return amax_bits == 0 ? 0 : (ax_e > 8 ? ax_e - 7 : 1);
}

inline __device__ float opus_moe_stage2_a8w4_decode_mxfp8_scale_from_e8m0(
    int e8m0)
{
    // fp8 = bf16 / 2^(E-127); power-of-2 scaling is exact.
    return e8m0 == 0
        ? 1.0f
        : opus_moe_gfx950_e8m0_to_float_scale(static_cast<uint32_t>(e8m0));
}

inline __device__ opus::u8x8_t
opus_moe_stage2_a8w4_decode_bf16_pair4_to_fp8x8(
    opus_moe_stage2_a8w4_decode_u32x4_store_t pairs,
    float scale)
{
    const opus::u64_t bf16_0123_bits =
        static_cast<opus::u64_t>(pairs[0]) |
        (static_cast<opus::u64_t>(pairs[1]) << 32);
    const opus::u64_t bf16_4567_bits =
        static_cast<opus::u64_t>(pairs[2]) |
        (static_cast<opus::u64_t>(pairs[3]) << 32);
    const auto fp8_0123 = opus::cast<opus::fp8_t>(
        __builtin_bit_cast(opus::bf16x4_t, bf16_0123_bits), scale);
    const auto fp8_4567 = opus::cast<opus::fp8_t>(
        __builtin_bit_cast(opus::bf16x4_t, bf16_4567_bits), scale);
    const opus::u64_t packed8 =
        static_cast<opus::u64_t>(__builtin_bit_cast(uint32_t, fp8_0123)) |
        (static_cast<opus::u64_t>(__builtin_bit_cast(uint32_t, fp8_4567)) << 32);
    return __builtin_bit_cast(opus::u8x8_t, packed8);
}

inline __device__ OpusMoeStage2A8W4DecodeMxFp8Group
opus_moe_stage2_a8w4_decode_bf16_pair4_to_mxfp8(
    opus_moe_stage2_a8w4_decode_u32x4_store_t pairs)
{
    const uint32_t amax_bits =
        opus_moe_stage2_a8w4_decode_bf16_pair4_amax_bits(pairs);
    const int E =
        opus_moe_stage2_a8w4_decode_mxfp8_e8m0_from_amax_bits(amax_bits);
    const float scale = opus_moe_stage2_a8w4_decode_mxfp8_scale_from_e8m0(E);
    return {opus_moe_stage2_a8w4_decode_bf16_pair4_to_fp8x8(pairs, scale),
            static_cast<opus::u8_t>(E)};
}

// MXFP8 route_out: fp8 e4m3 data + per-8-col e8m0 scale in one row.
template<typename T, typename RouteOutGmem>
inline __device__ void opus_moe_stage2_a8w4_decode_store_smem_to_route_out_fp8(
    uint32_t* __restrict__ smem_c_pair,
    const int32_t* __restrict__ smem_route_base,
    const OpusMoeStage2A8W4CShuffleLayout<T>& c_layout,
    int col_base,
    RouteOutGmem& route_out_gmem,
    int64_t row_stride_bytes,
    int scale_col_off)
{
    static_assert(T::B_N == T::C_LDS_N);
    static_assert(!T::DIRECT_ATOMIC_OUT);
    static_assert(T::ELEM_PER_ATOMIC == 2);

    opus_moe_stage2_a8w4_decode_store_route_out_groups<T>(
        smem_c_pair,
        smem_route_base,
        c_layout,
        [&](int route_row,
            int col0,
            opus_moe_stage2_a8w4_decode_u32x4_store_t pairs) {
            const int data_col = col_base + col0;
            const int scale_col = scale_col_off + (data_col >> 3);
            const auto mxfp8_group =
                opus_moe_stage2_a8w4_decode_bf16_pair4_to_mxfp8(pairs);
            const int64_t row_base =
                static_cast<int64_t>(route_row) * row_stride_bytes;
            route_out_gmem.template store<8>(
                mxfp8_group.data,
                static_cast<int>(row_base + data_col));
            route_out_gmem.store(
                mxfp8_group.e8m0,
                static_cast<int>(row_base + scale_col));
        });
}

#endif // __HIP_DEVICE_COMPILE__ && __gfx950__

// Kernel entry.
template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_BLOCKS_PER_CU) void
opus_moe_stage2_a8w4_decode_kernel_gfx950(opus_moe_stage2_a8w4_kargs kargs)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_MFMA_A = typename T::D_MFMA_A;
    using D_MFMA_B = typename T::D_MFMA_B;
    using D_ACC = typename T::D_ACC;
    using Schedule = OpusMoeStage2A8W4DecodeSchedule<T>;

    int sorted_rows;
    int route_base;
    int col_base;
    opus_moe_stage2_a8w4_decode_make_tile<T>(kargs, sorted_rows, route_base, col_base);
    if(route_base >= sorted_rows)
        return;
    const int token_num = kargs.token_num;
    const int sorted_block_id = route_base / T::SORT_BLOCK_M;
    const int expert_id = kargs.sorted_expert_ids[sorted_block_id];

    const int tid = static_cast<int>(thread_id_x());
    const int lane_id = tid % get_warp_size();
    const int wave_id = __builtin_amdgcn_readfirstlane(tid / get_warp_size());
    const int wave_id_m = wave_id / T::T_N;
    const int wave_id_n = wave_id % T::T_N;
    const int64_t w2_expert_base = static_cast<int64_t>(expert_id) * kargs.stride_w_e;
    const int scale_row_base = expert_id * kargs.model_dim;
    const int scale_row_col_base = scale_row_base + col_base;

    __shared__ int32_t smem_a_base[T::B_M];
    __shared__ int32_t smem_route_base[T::B_M];
    __shared__ float smem_weight[T::B_M];
    constexpr int A_LDS_BYTES =
        T::A_LDS_STAGES * T::A_LDS_STAGE_ELEMS * static_cast<int>(sizeof(D_A));
    constexpr int C_LDS_BYTES =
        T::B_M * T::C_LDS_N / T::ELEM_PER_ATOMIC *
        static_cast<int>(sizeof(uint32_t));
    constexpr int SCRATCH_BYTES =
        (A_LDS_BYTES > C_LDS_BYTES) ? A_LDS_BYTES : C_LDS_BYTES;
    alignas(T::BYTES_PER_VEC) __shared__ char smem_scratch[SCRATCH_BYTES];
    auto* smem_c_pair = reinterpret_cast<uint32_t*>(smem_scratch);
    auto s_a = opus_moe_stage2_a8w4_decode_make_smem_a_stages<
        D_A,
        T::A_LDS_STAGE_ELEMS>(smem_scratch,
                              opus::make_index_seq<T::A_LDS_STAGES>{});

    const bool has_route = opus_moe_stage2_a8w4_decode_load_route_metadata<T>(
        kargs,
        route_base,
        sorted_rows,
        tid,
        smem_a_base,
        smem_route_base,
        smem_weight);
    if(!has_route)
        return;

    auto mma = make_mfma<D_MFMA_A, D_MFMA_B, D_ACC>(
        number<T::MMA_M>{},
        number<T::MMA_N>{},
        number<T::MMA_K>{});

    const D_A* __restrict__ inter_states =
        reinterpret_cast<const D_A*>(kargs.inter_states_fp8);
    const uint8_t* __restrict__ w2 = kargs.w2_fp4;
    const uint8_t* __restrict__ a2_scale = kargs.a2_scale_e8m0;
    const uint8_t* __restrict__ w2_scale = kargs.w2_scale_e8m0;
    const unsigned int a_size_bytes =
        static_cast<unsigned int>(static_cast<unsigned long long>(token_num) *
                                  static_cast<unsigned long long>(kargs.stride_a_t));
    const unsigned int a_scale_size_bytes =
        static_cast<unsigned int>(static_cast<unsigned long long>(sorted_rows) *
                                  static_cast<unsigned long long>(kargs.stride_a_scale_route));
    auto g_a = make_gmem(inter_states, a_size_bytes);
    auto g_a_scale = make_gmem(a2_scale, a_scale_size_bytes);
    auto g_b = make_gmem(w2 + w2_expert_base, static_cast<unsigned int>(kargs.stride_w_e));
    const unsigned int w_scale_size_bytes = static_cast<unsigned int>(
        static_cast<unsigned long long>(kargs.num_experts) *
        static_cast<unsigned long long>(kargs.model_dim) *
        static_cast<unsigned long long>(kargs.stride_w_scale_row));
    auto g_w_scale = make_gmem(w2_scale, w_scale_size_bytes);
    auto u_ga = opus_moe_stage2_a8w4_layout_ga<T>(lane_id, wave_id_m);
    auto u_sa = opus_moe_stage2_a8w4_layout_sa<T>(lane_id, wave_id_m);
    auto u_gb = opus_moe_stage2_a8w4_layout_gb<T>(lane_id, wave_id_n);
    auto u_c = opus_moe_stage2_a8w4_layout_c<T>(wave_id_m, wave_id_n);

    typename decltype(mma)::vtype_c v_c[T::M_MFMA_PER_WAVE][T::N_MFMA_PER_WAVE];
    static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
        static_for<T::N_MFMA_PER_WAVE>([&](auto ni) {
            clear(v_c[mi.value][ni.value]);
        });
    });

    // Mainloop: flat memory views plus explicitly scheduled load/compute stages.
    using opus::operator""_I;
    using V_A = typename decltype(mma)::mfma_type::vtype_a;
    using V_B = typename decltype(mma)::mfma_type::vtype_b;

    static_assert(T::DECODE_EFFECTIVE_INTER_DIM == T::K_TILES * T::K_STEP_PACKED);
    static_assert(T::N_MFMA_PER_WAVE >= 2 && (T::N_MFMA_PER_WAVE % 2) == 0);

    int a_base[T::M_MFMA_PER_WAVE];
    int a_scale_base_word[T::M_MFMA_PER_WAVE];
    opus_moe_stage2_a8w4_decode_init_a_payload<T>(
        u_ga, smem_a_base, route_base, a_base, a_scale_base_word);

    // B payload addressing is computed once and reused by the scheduled loads.
    const int b_gb_offset0 =
        static_cast<int>(u_gb(opus::number<0>{}, opus::number<0>{}));
    const int b_lane_offset = b_gb_offset0 & (T::B_THREADGROUP_STRIDE_BYTES - 1);
    const int b_wave_scalar_base =
        wave_id_n * T::N_MFMA_PER_WAVE * (T::MMA_N * T::B_PAYLOAD_ROW_STRIDE_BYTES);

    const int b_scale_base_word =
        opus_moe_stage2_a8w4_b_scale_base_word_offset<T>(
            scale_row_col_base, b_gb_offset0);

    // Local helpers capture repeated state while keeping the schedule explicit.
    auto issue_a_gmem_to_smem = [&](auto stage, int k_base) {
        opus_moe_stage2_a8w4_decode_issue_a<T>(
            u_ga, u_sa, s_a, g_a, a_base, wave_id_n, stage, k_base);
    };
    auto load_a_smem_to_reg = [&](auto stage, auto& v_a) {
        opus_moe_stage2_a8w4_decode_load_a<T>(u_sa, s_a, stage, v_a);
    };
    auto load_b_half_gmem_to_reg = [&](auto n_half, int b_tile_base, auto& v_b) {
        opus_moe_stage2_a8w4_decode_load_b_half<T>(
            g_b, b_lane_offset, b_wave_scalar_base, n_half, b_tile_base, v_b);
    };
    auto compute_mma_half_scaled = [&](auto scale_pair, auto n_half,
                                       const auto& v_a,
                                       const int (&a_scale)[T::M_MFMA_PER_WAVE],
                                       const int (&b_scale)[T::HALF_N_MFMA_PER_WAVE],
                                       const auto& v_b) {
        opus_moe_stage2_a8w4_decode_compute_scaled_half<T>(
            mma, wave_id_m, v_c, scale_pair, n_half, v_a, a_scale, b_scale, v_b);
    };

    // Load one K-tile's A and both B halves, then compute both halves.
    auto compute_k_tile_both_n_halves = [&](auto scale_pair,
                                            auto stage,
                                            int b_tile_base,
                                            const int (&b_scale)[T::HALF_N_MFMA_PER_WAVE],
                                            const int (&a_scale)[T::M_MFMA_PER_WAVE]) {
        V_A v_a[T::M_MFMA_PER_WAVE];
        V_B v_b0[T::HALF_N_MFMA_PER_WAVE];
        V_B v_b1[T::HALF_N_MFMA_PER_WAVE];
        load_a_smem_to_reg(stage, v_a);
        load_b_half_gmem_to_reg(0_I, b_tile_base, v_b0);
        load_b_half_gmem_to_reg(1_I, b_tile_base, v_b1);
        __builtin_amdgcn_s_setprio(1);
        compute_mma_half_scaled(scale_pair, 0_I, v_a, a_scale, b_scale, v_b0);
        compute_mma_half_scaled(scale_pair, 1_I, v_a, a_scale, b_scale, v_b1);
        __builtin_amdgcn_s_setprio(0);
    };

    if constexpr(T::K_TILES == 3)
    {
        // K3 (3 K-tiles) fully-unrolled software pipeline.
        using MainloopSchedule = OpusMoeStage2A8W4DecodeMainloopSchedule;
        constexpr bool kSplitALoadByNWave =
            Schedule::Mainloop == MainloopSchedule::SplitALoadByNWave;

        const int b_tile_base0 = col_base * T::B_PAYLOAD_ROW_STRIDE_BYTES;
        const int b_tile_stride = T::K_STEP_PACKED * T::B_PAYLOAD_K_STRIDE_BYTES;
        const int b_tile_base1 = b_tile_base0 + b_tile_stride;
        const int b_tile_base2 = b_tile_base0 + 2 * b_tile_stride;

        int b_scale0[T::HALF_N_MFMA_PER_WAVE];
        int b_scale1[T::HALF_N_MFMA_PER_WAVE];
        int a_scale0[T::M_MFMA_PER_WAVE];
        int a_scale1[T::M_MFMA_PER_WAVE];

        issue_a_gmem_to_smem(0_I, 0);
        issue_a_gmem_to_smem(1_I, T::K_STEP_PACKED);
        issue_a_gmem_to_smem(2_I, 2 * T::K_STEP_PACKED);

        if constexpr(kSplitALoadByNWave)
        {
            V_B v_b0h0[T::HALF_N_MFMA_PER_WAVE];
            V_B v_b0h1[T::HALF_N_MFMA_PER_WAVE];
            load_b_half_gmem_to_reg(0_I, b_tile_base0, v_b0h0);
            load_b_half_gmem_to_reg(1_I, b_tile_base0, v_b0h1);
            opus_moe_stage2_a8w4_decode_load_b_scale<T>(
                g_w_scale, b_scale_base_word, 0, b_scale0);
            opus_moe_stage2_a8w4_decode_load_a_scale<T>(
                g_a_scale, a_scale_base_word, 0, a_scale0);
            opus_moe_stage2_a8w4_decode_wait_a(
                wave_id_n,
                opus::number<T::A_LDS_BUFFER_LOAD_INSTS +
                             T::HALF_N_MFMA_PER_WAVE +
                             T::M_MFMA_PER_WAVE +
                             2 * T::HALF_N_MFMA_PER_WAVE>{});

            V_A v_a0[T::M_MFMA_PER_WAVE];
            load_a_smem_to_reg(0_I, v_a0);
            __builtin_amdgcn_s_setprio(1);
            compute_mma_half_scaled(0_I, 0_I, v_a0, a_scale0, b_scale0, v_b0h0);
            compute_mma_half_scaled(0_I, 1_I, v_a0, a_scale0, b_scale0, v_b0h1);
            __builtin_amdgcn_s_setprio(0);
        }
        else
        {
            opus_moe_stage2_a8w4_decode_wait_a(
                wave_id_n, opus::number<T::A_LDS_BUFFER_LOAD_INSTS>{});
            opus_moe_stage2_a8w4_decode_load_b_scale<T>(
                g_w_scale, b_scale_base_word, 0, b_scale0);
            opus_moe_stage2_a8w4_decode_load_a_scale<T>(
                g_a_scale, a_scale_base_word, 0, a_scale0);
            compute_k_tile_both_n_halves(0_I, 0_I, b_tile_base0, b_scale0, a_scale0);
        }

        V_A v_a1[T::M_MFMA_PER_WAVE];
        V_A v_a2[T::M_MFMA_PER_WAVE];
        V_B v_b1h0[T::HALF_N_MFMA_PER_WAVE];
        V_B v_b1h1[T::HALF_N_MFMA_PER_WAVE];
        if constexpr(kSplitALoadByNWave)
            opus_moe_stage2_a8w4_decode_wait_a(wave_id_n, 1_I);
        else
            opus_moe_stage2_a8w4_decode_wait_a(wave_id_n, 0_I);
        opus::s_waitcnt_vmcnt(opus::number<T::HALF_N_MFMA_PER_WAVE>{});

        load_a_smem_to_reg(1_I, v_a1);
        load_b_half_gmem_to_reg(0_I, b_tile_base1, v_b1h0);
        load_b_half_gmem_to_reg(1_I, b_tile_base1, v_b1h1);

        __builtin_amdgcn_s_setprio(1);
        compute_mma_half_scaled(1_I, 0_I, v_a1, a_scale0, b_scale0, v_b1h0);
        opus::s_waitcnt_vmcnt(0_I);
        load_a_smem_to_reg(2_I, v_a2);
        if constexpr(kSplitALoadByNWave)
            opus::sync_threads();
        compute_mma_half_scaled(1_I, 1_I, v_a1, a_scale0, b_scale0, v_b1h1);
        __builtin_amdgcn_s_setprio(0);

        opus_moe_stage2_a8w4_decode_load_b_scale<T>(
            g_w_scale, b_scale_base_word, T::SCALE_WORDS_PER_GROUP_PACK, b_scale1);
        opus_moe_stage2_a8w4_decode_load_a_scale<T>(
            g_a_scale, a_scale_base_word, T::SCALE_WORDS_PER_GROUP_PACK, a_scale1);

        V_B v_b2h0[T::HALF_N_MFMA_PER_WAVE];
        V_B v_b2h1[T::HALF_N_MFMA_PER_WAVE];
        load_b_half_gmem_to_reg(0_I, b_tile_base2, v_b2h0);
        load_b_half_gmem_to_reg(1_I, b_tile_base2, v_b2h1);

        __builtin_amdgcn_s_setprio(1);
        compute_mma_half_scaled(0_I, 0_I, v_a2, a_scale1, b_scale1, v_b2h0);
        opus::s_waitcnt_vmcnt(0_I);
        compute_mma_half_scaled(0_I, 1_I, v_a2, a_scale1, b_scale1, v_b2h1);
        __builtin_amdgcn_s_setprio(0);
    }
    else
    {
        opus_moe_stage2_a8w4_decode_run_generic_schedule_gfx950<T, V_A, V_B>(
            col_base,
            u_ga,
            u_sa,
            s_a,
            g_a,
            a_base,
            wave_id_n,
            g_a_scale,
            a_scale_base_word,
            g_w_scale,
            b_scale_base_word,
            compute_k_tile_both_n_halves);
    }

    if constexpr(!Schedule::MainloopEndsWithSmemBarrier)
    {
        opus::sync_threads();
    }
    if constexpr(T::DIRECT_ATOMIC_OUT)
    {
        constexpr int output_rows_per_token = 1;
        const unsigned int output_size_bytes = static_cast<unsigned int>(
            static_cast<unsigned long long>(token_num) *
            static_cast<unsigned long long>(output_rows_per_token) *
            static_cast<unsigned long long>(kargs.stride_o_t) *
            static_cast<unsigned long long>(sizeof(hip_bfloat16)));
        auto output_gmem = opus::make_gmem(
            reinterpret_cast<opus::bf16_t*>(kargs.out_bf16),
            output_size_bytes);
        opus_moe_stage2_a8w4_decode_store_acc_to_cshuffle<T>(
            v_c, u_c, smem_weight, smem_c_pair);
        opus_moe_stage2_a8w4_decode_atomic_smem_to_out<T>(
            smem_c_pair, smem_route_base, u_c, col_base,
            kargs.stride_o_t, output_gmem);
    }
    else if(kargs.route_out_fp8)
    {
        const unsigned int route_out_size_bytes = static_cast<unsigned int>(
            static_cast<unsigned long long>(token_num) *
            static_cast<unsigned long long>(kargs.topk) *
            static_cast<unsigned long long>(kargs.route_out_row_bytes));
        auto route_out_gmem = opus::make_gmem(
            reinterpret_cast<opus::u8_t*>(kargs.out_bf16),
            route_out_size_bytes);
        opus_moe_stage2_a8w4_decode_store_acc_to_cshuffle<T>(
            v_c, u_c, smem_weight, smem_c_pair);
        opus_moe_stage2_a8w4_decode_store_smem_to_route_out_fp8<T>(
            smem_c_pair, smem_route_base, u_c, col_base,
            route_out_gmem,
            kargs.route_out_row_bytes, kargs.model_dim);
    }
    else
    {
        opus_moe_stage2_a8w4_decode_store_acc_to_cshuffle<T>(
            v_c, u_c, smem_weight, smem_c_pair);
        opus_moe_stage2_a8w4_decode_store_smem_to_route_out<T>(
            smem_c_pair, smem_route_base, u_c, col_base,
            kargs.out_bf16, kargs.stride_o_t);
    }
#endif // __HIP_DEVICE_COMPILE__ && __gfx950__
}
