// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "../opus_moe_stage2_utils_gfx950.cuh"
#include "opus_moe_pipeline_stage2_a8w4_decode_policy_gfx950.cuh"

#include "opus/opus.hpp"

#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)

// Tile mapping: baseline m-fast mapping by default. Windowed XCD swizzle is
// available when traits opt in with SWIZZLE_C > 0.
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
        tile_n_id = static_cast<int>(blockIdx.x);
        tile_m_id = static_cast<int>(blockIdx.y);
    }
    else
    {
        const int num_tiles_n = kargs.model_dim / BN;
        const int wgid = static_cast<int>(blockIdx.y) * num_tiles_n +
                         static_cast<int>(blockIdx.x);
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
                a_base = static_cast<int32_t>(
                    static_cast<int64_t>(token) * kargs.stride_a_t +
                    static_cast<int64_t>(slot) * kargs.stride_a_k);
                weight = (kargs.sorted_weights == nullptr) ? 1.0f : kargs.sorted_weights[row];
                if constexpr(T::DIRECT_ATOMIC_OUT)
                    route_row = static_cast<int32_t>(token);
                else
                    route_row = static_cast<int32_t>(token * topk + slot);
                has_route = 1;
            }
        }

        smem_a_base[local_m] = a_base;
        smem_route_base[local_m] = route_row;
        smem_weight[local_m] = weight;
    }
    if constexpr(T::DIRECT_ATOMIC_OUT)
    {
        if(route_base + T::B_M <= sorted_rows)
        {
            __syncthreads();
            return true;
        }
        return __syncthreads_or(has_route) != 0;
    }
    else
    {
        if constexpr(T::IS_BM32_BN256 || T::IS_BM64_BN256)
        {
            // Full sorted tiles do not need a route-count reduction.
            if(route_base + T::B_M <= sorted_rows)
            {
                __syncthreads();
                return true;
            }
        }
        const int route_count = __syncthreads_count(has_route);
        return route_count != 0;
    }
}

template<typename T,
         typename V_A,
         typename V_B,
         typename IssueAPayload,
         typename WaitAPayload,
         typename LoadAPayload,
         typename LoadBHalf,
         typename LoadBScale,
         typename LoadAScale,
         typename ComputeHalf,
         typename ComputeTile>
inline __device__ void opus_moe_stage2_a8w4_decode_run_k_scheduler_gfx950(
    int col_base,
    int b_payload_row_stride_bytes,
    IssueAPayload& issue_a_payload,
    WaitAPayload& wait_a_payload,
    LoadAPayload& load_a_payload,
    LoadBHalf& load_b_half,
    LoadBScale& load_b_scale,
    LoadAScale& load_a_scale,
    ComputeHalf& compute_half,
    ComputeTile& compute_tile)
{
    using namespace opus;

    static_assert(T::K_TILES >= 2);
    using Schedule = OpusMoeStage2A8W4DecodeSchedule<T>;
    using MainloopSchedule = OpusMoeStage2A8W4DecodeMainloopSchedule;

    constexpr bool HasOddTail = (T::K_TILES & 1) != 0;
    constexpr int PairCount = T::K_TILES / 2;
    static_assert(PairCount >= 1);

    constexpr int k0 = 0;
    constexpr int k1 = T::K_STEP_PACKED;
    constexpr int k2 = 2 * T::K_STEP_PACKED;
    const int b_tile_base0 = opus_moe_stage2_a8w4_b_payload_tile_base_byte_offset<T>(
        col_base, k0, b_payload_row_stride_bytes);
    const int b_tile_base1 = opus_moe_stage2_a8w4_b_payload_tile_base_byte_offset<T>(
        col_base, k1, b_payload_row_stride_bytes);
    const int b_tile_base2 = opus_moe_stage2_a8w4_b_payload_tile_base_byte_offset<T>(
        col_base, k2, b_payload_row_stride_bytes);

    auto b_tile_base_for = [&](auto stage) {
        constexpr int Stage = decltype(stage)::value;
        if constexpr(Stage == 0)
            return b_tile_base0;
        else if constexpr(Stage == 1)
            return b_tile_base1;
        else if constexpr(Stage == 2)
            return b_tile_base2;
        else
            return static_cast<int>(opus_moe_stage2_a8w4_b_payload_tile_base_byte_offset<T>(
                col_base,
                Stage * T::K_STEP_PACKED,
                b_payload_row_stride_bytes));
    };

    auto load_scale_pair = [&](auto pair,
                               int (&b_scale)[T::HALF_N_MFMA_PER_WAVE],
                               int (&a_scale)[T::M_MFMA_PER_WAVE]) {
        constexpr int scale_word_base =
            decltype(pair)::value * T::SCALE_WORDS_PER_GROUP_PACK;
        load_b_scale(scale_word_base, b_scale);
        load_a_scale(scale_word_base, a_scale);
    };

    auto wait_a_tile_and_pending_b = [&]() {
        if constexpr(Schedule::Mainloop == MainloopSchedule::SplitALoadByNWave)
            wait_a_payload(1_I);
        else
            wait_a_payload(0_I);
        s_waitcnt_vmcnt(number<T::HALF_N_MFMA_PER_WAVE>{});
    };

    auto compute_prefetched_even_tile = [&](int tile_base,
                                            const V_A (&v_a_even)[T::M_MFMA_PER_WAVE],
                                            const int (&a_scale)[T::M_MFMA_PER_WAVE],
                                            const int (&b_scale)[T::HALF_N_MFMA_PER_WAVE]) {
        V_B v_b_half0[T::HALF_N_MFMA_PER_WAVE];
        V_B v_b_half1[T::HALF_N_MFMA_PER_WAVE];
        load_b_half(0, tile_base, v_b_half0);
        load_b_half(1, tile_base, v_b_half1);

        __builtin_amdgcn_s_setprio(1);
        compute_half(0_I, 0_I, v_a_even, a_scale, b_scale, v_b_half0);
        s_waitcnt_vmcnt(0_I);
        compute_half(0_I, 1_I, v_a_even, a_scale, b_scale, v_b_half1);
        __builtin_amdgcn_s_setprio(0);
    };

    if constexpr(T::A_LDS_STAGES < T::K_TILES)
    {
        static_assert(T::A_LDS_STAGES == T::A_LDS_STREAM_STAGES);

        auto issue_streamed_a_tile = [&](auto stage_slot, int k_tile) {
            const int k_base = k_tile * T::K_STEP_PACKED;
            issue_a_payload(stage_slot, k_base);
        };

        auto compute_streamed_tile = [&](auto scale_pair,
                                         auto stage_slot,
                                         int k_tile,
                                         const int (&b_scale)[T::HALF_N_MFMA_PER_WAVE],
                                         const int (&a_scale)[T::M_MFMA_PER_WAVE]) {
            const int k_base = k_tile * T::K_STEP_PACKED;
            const int b_tile_base =
                opus_moe_stage2_a8w4_b_payload_tile_base_byte_offset<T>(
                    col_base, k_base, b_payload_row_stride_bytes);
            compute_tile(
                scale_pair,
                0_I,
                stage_slot,
                b_tile_base,
                b_scale,
                a_scale);
            __builtin_amdgcn_s_barrier();
        };

        #pragma unroll 1
        for(int pair = 0; pair < T::K_TILES / 2; ++pair)
        {
            int b_scale[T::HALF_N_MFMA_PER_WAVE];
            int a_scale[T::M_MFMA_PER_WAVE];
            const int scale_word_base = pair * T::SCALE_WORDS_PER_GROUP_PACK;
            issue_streamed_a_tile(0_I, 2 * pair);
            issue_streamed_a_tile(1_I, 2 * pair + 1);
            load_b_scale(scale_word_base, b_scale);
            load_a_scale(scale_word_base, a_scale);
            wait_a_payload(0_I);
            compute_streamed_tile(0_I, 0_I, 2 * pair, b_scale, a_scale);
            compute_streamed_tile(1_I, 1_I, 2 * pair + 1, b_scale, a_scale);
        }

        if constexpr(HasOddTail)
        {
            constexpr int TailTile = T::K_TILES - 1;
            constexpr int TailPair = TailTile / 2;
            int b_scale[T::HALF_N_MFMA_PER_WAVE];
            int a_scale[T::M_MFMA_PER_WAVE];
            issue_streamed_a_tile(0_I, TailTile);
            load_scale_pair(number<TailPair>{}, b_scale, a_scale);
            wait_a_payload(0_I);
            compute_streamed_tile(0_I, 0_I, TailTile, b_scale, a_scale);
        }
    }

    if constexpr(T::A_LDS_STAGES >= T::K_TILES)
    {
        V_A v_a_prefetched_even[T::M_MFMA_PER_WAVE];

        static_for<T::K_TILES>([&](auto kt) {
            issue_a_payload(kt, kt.value * T::K_STEP_PACKED);
        });

        auto run_odd_tile = [&](auto stage,
                                auto prefetch_next,
                                auto wait_for_pending_b_half0,
                                int tile_base,
                                const int (&a_scale)[T::M_MFMA_PER_WAVE],
                                const int (&b_scale)[T::HALF_N_MFMA_PER_WAVE]) {
            constexpr int Stage = decltype(stage)::value;
            constexpr bool PrefetchNext = decltype(prefetch_next)::value != 0;
            constexpr bool WaitForPendingBHalf0 =
                decltype(wait_for_pending_b_half0)::value != 0;
            static_assert((Stage & 1) == 1);
            static_assert(!PrefetchNext || Stage + 1 < T::K_TILES);

            V_A v_a_odd[T::M_MFMA_PER_WAVE];
            V_B v_b_half0[T::HALF_N_MFMA_PER_WAVE];
            V_B v_b_half1[T::HALF_N_MFMA_PER_WAVE];
            wait_a_tile_and_pending_b();

            load_a_payload(stage, v_a_odd);
            load_b_half(0, tile_base, v_b_half0);
            load_b_half(1, tile_base, v_b_half1);

            if constexpr(WaitForPendingBHalf0)
                s_waitcnt_vmcnt(number<T::HALF_N_MFMA_PER_WAVE>{});

            __builtin_amdgcn_s_setprio(1);
            compute_half(1_I, 0_I, v_a_odd, a_scale, b_scale, v_b_half0);
            s_waitcnt_vmcnt(0_I);
            if constexpr(PrefetchNext)
            {
                load_a_payload(number<Stage + 1>{}, v_a_prefetched_even);
                if constexpr(Schedule::Mainloop == MainloopSchedule::SplitALoadByNWave)
                    __builtin_amdgcn_s_barrier();
            }
            compute_half(1_I, 1_I, v_a_odd, a_scale, b_scale, v_b_half1);
            __builtin_amdgcn_s_setprio(0);
        };

        auto run_first_pair = [&](auto prefetch_next) {
            constexpr bool PrefetchNext = decltype(prefetch_next)::value != 0;
            int b_scale[T::HALF_N_MFMA_PER_WAVE];
            int a_scale[T::M_MFMA_PER_WAVE];

            if constexpr(Schedule::Mainloop == MainloopSchedule::SplitALoadByNWave)
            {
                V_B v_b_half0[T::HALF_N_MFMA_PER_WAVE];
                V_B v_b_half1[T::HALF_N_MFMA_PER_WAVE];
                load_b_half(0, b_tile_base0, v_b_half0);
                load_b_half(1, b_tile_base0, v_b_half1);

                load_scale_pair(0_I, b_scale, a_scale);
                if constexpr(PrefetchNext)
                    wait_a_payload(number<T::A_LDS_BUFFER_LOAD_INSTS +
                                          T::HALF_N_MFMA_PER_WAVE +
                                          T::M_MFMA_PER_WAVE +
                                          2 * T::HALF_N_MFMA_PER_WAVE>{});
                else
                    wait_a_payload(0_I);

                V_A v_a_even[T::M_MFMA_PER_WAVE];
                load_a_payload(0_I, v_a_even);

                if constexpr(!PrefetchNext)
                    s_waitcnt_vmcnt(0_I);

                __builtin_amdgcn_s_setprio(1);
                compute_half(0_I, 0_I, v_a_even, a_scale, b_scale, v_b_half0);
                compute_half(0_I, 1_I, v_a_even, a_scale, b_scale, v_b_half1);
                __builtin_amdgcn_s_setprio(0);
            }
            else
            {
                if constexpr(PrefetchNext)
                    wait_a_payload(number<T::A_LDS_BUFFER_LOAD_INSTS>{});
                else
                    wait_a_payload(0_I);
                load_scale_pair(0_I, b_scale, a_scale);
                compute_tile(
                    0_I,
                    number<!PrefetchNext>{},
                    0_I,
                    b_tile_base0,
                    b_scale,
                    a_scale);
            }

            run_odd_tile(
                1_I,
                number<PrefetchNext>{},
                number<!PrefetchNext>{},
                b_tile_base1,
                a_scale,
                b_scale);
        };

        auto run_middle_pair = [&](auto pair) {
            constexpr int Pair = decltype(pair)::value;
            constexpr int EvenStage = 2 * Pair;
            constexpr int OddStage = EvenStage + 1;
            constexpr bool PrefetchNext = OddStage + 1 < T::K_TILES;
            static_assert(Pair > 0);
            static_assert(OddStage < T::K_TILES);

            int b_scale[T::HALF_N_MFMA_PER_WAVE];
            int a_scale[T::M_MFMA_PER_WAVE];
            load_scale_pair(pair, b_scale, a_scale);
            compute_prefetched_even_tile(
                b_tile_base_for(number<EvenStage>{}),
                v_a_prefetched_even,
                a_scale,
                b_scale);
            run_odd_tile(
                number<OddStage>{},
                number<PrefetchNext>{},
                0_I,
                b_tile_base_for(number<OddStage>{}),
                a_scale,
                b_scale);
        };

        constexpr bool HasRemainingAfterFirstPair = T::K_TILES > 2;
        run_first_pair(number<HasRemainingAfterFirstPair>{});
        static_for<PairCount - 1>([&](auto local_pair) {
            run_middle_pair(number<local_pair.value + 1>{});
        });
        if constexpr(HasOddTail)
        {
            constexpr int TailStage = 2 * PairCount;
            constexpr int TailPair = PairCount;
            int b_scale[T::HALF_N_MFMA_PER_WAVE];
            int a_scale[T::M_MFMA_PER_WAVE];
            load_scale_pair(number<TailPair>{}, b_scale, a_scale);
            compute_prefetched_even_tile(
                b_tile_base_for(number<TailStage>{}),
                v_a_prefetched_even,
                a_scale,
                b_scale);
        }
    }
}

// Mainloop: A/B/scale loads and MFMA accumulation.
typedef uint32_t opus_moe_stage2_a8w4_decode_u32x4_t __attribute__((ext_vector_type(4)));
typedef uint32_t opus_moe_stage2_a8w4_decode_u32x8_t __attribute__((ext_vector_type(8)));

template<typename Reg>
inline __device__ void opus_moe_stage2_a8w4_decode_pack_a_mfma_reg(
    opus_moe_stage2_a8w4_decode_u32x4_t lo,
    opus_moe_stage2_a8w4_decode_u32x4_t hi,
    Reg& reg)
{
    opus_moe_stage2_a8w4_decode_u32x8_t packed{};
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
inline __device__ void opus_moe_stage2_a8w4_decode_unpack_b_mfma_reg(
    opus_moe_stage2_a8w4_decode_u32x4_t value,
    Reg& reg)
{
    opus_moe_stage2_a8w4_decode_u32x8_t packed{};
    packed[0] = value[0];
    packed[1] = value[1];
    packed[2] = value[2];
    packed[3] = value[3];
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

template<typename T,
         typename Mma,
         typename LayoutA,
         typename LayoutASmem,
         typename SmemA,
         typename LayoutB,
         typename GmemA,
         typename GmemAScale,
         typename GmemB,
         typename GmemWScale>
inline __device__ void opus_moe_stage2_a8w4_decode_mainloop(
    Mma& mma,
    const LayoutA& u_ga,
    const LayoutASmem& u_sa,
    SmemA& s_a,
    const LayoutB& u_gb,
    GmemA& g_a,
    GmemAScale& g_a_scale,
    GmemB& g_b,
    GmemWScale& g_w_scale,
    const int32_t* __restrict__ smem_a_base,
    int route_base,
    int col_base,
    int b_payload_row_stride_bytes,
    int wave_id_m,
    int wave_id_n,
    int scale_row_col_base,
    typename Mma::vtype_c (&v_c)[T::M_MFMA_PER_WAVE][T::N_MFMA_PER_WAVE])
{
    using namespace opus;
    using opus::operator""_I;

    using V_A = typename Mma::mfma_type::vtype_a;
    using V_B = typename Mma::mfma_type::vtype_b;

    static_assert(T::DECODE_EFFECTIVE_INTER_DIM == T::K_TILES * T::K_STEP_PACKED);
    static_assert(T::N_MFMA_PER_WAVE >= 2 && (T::N_MFMA_PER_WAVE % 2) == 0);
    using Schedule = OpusMoeStage2A8W4DecodeSchedule<T>;
    using MainloopSchedule = OpusMoeStage2A8W4DecodeMainloopSchedule;

    auto ga_offset = [&](auto mi) {
        return static_cast<int>(u_ga(mi, 0_I));
    };
    auto gb_offset = [&](auto ni) {
        return static_cast<int>(u_gb(ni, 0_I));
    };
    auto sa_offset = [&](auto mi, auto half) {
        return static_cast<int>(u_sa(mi, half, 0_I));
    };

    auto a_base_for_ga = [&](int ga) {
        const int local_m = opus_moe_stage2_a8w4_a_local_m<T>(ga);
        return smem_a_base[local_m];
    };

    int a_base[T::M_MFMA_PER_WAVE];
    int a_scale_base_word[T::M_MFMA_PER_WAVE];
    static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
        const int ga = ga_offset(mi);
        a_base[mi.value] = a_base_for_ga(ga);
        a_scale_base_word[mi.value] =
            opus_moe_stage2_a8w4_a_scale_base_word_offset<T>(route_base, ga_offset(mi));
    });
    const int b_scale_base_word =
        opus_moe_stage2_a8w4_b_scale_base_word_offset<T>(scale_row_col_base,
                                                         gb_offset(0_I),
                                                         wave_id_n);
    const int b_ni_stride_bytes = T::MMA_N * b_payload_row_stride_bytes;
    constexpr int b_lane_offset_mask = T::B_THREADGROUP_STRIDE_BYTES - 1;
    const int b_lane_offset = gb_offset(0_I) & b_lane_offset_mask;
    const int b_wave_scalar_base =
        wave_id_n * T::N_MFMA_PER_WAVE * b_ni_stride_bytes;

    auto issue_a_payload = [&](auto stage, int k_base) {
        constexpr int Stage = decltype(stage)::value;
        static_assert(Stage >= 0 && Stage < T::A_LDS_STAGES);

        if(wave_id_n > 1)
            return;

        auto issue_one_mi = [&](auto mi) {
            auto* smem_lo = s_a[Stage].ptr + sa_offset(mi, 0_I);
            auto* smem_hi = s_a[Stage].ptr + sa_offset(mi, 1_I);
            const int a_offset_lo = opus_moe_stage2_a8w4_a_payload_byte_offset<T>(
                a_base[mi.value],
                k_base,
                ga_offset(mi));
            if constexpr(Schedule::Mainloop ==
                         MainloopSchedule::SplitALoadByNWave)
            {
                if(wave_id_n == 0)
                {
                    g_a.template async_load<T::VEC_A>(
                        reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(smem_lo)),
                        a_offset_lo,
                        0,
                        opus::number<T::CACHECTL_A>{});
                }
                else
                {
                    g_a.template async_load<T::VEC_A>(
                        reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(smem_hi)),
                        a_offset_lo + T::K_STEP_PACKED / 2,
                        0,
                        opus::number<T::CACHECTL_A>{});
                }
            }
            else
            {
                g_a.template async_load<T::VEC_A>(
                    reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(smem_lo)),
                    a_offset_lo,
                    0,
                    opus::number<T::CACHECTL_A>{});
                g_a.template async_load<T::VEC_A>(
                    reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(smem_hi)),
                    a_offset_lo + T::K_STEP_PACKED / 2,
                    0,
                    opus::number<T::CACHECTL_A>{});
            }
        };

        if constexpr(Schedule::Mainloop ==
                     MainloopSchedule::SplitALoadByNWave)
        {
            issue_one_mi(0_I);
        }
        else
        {
            // M_MFMA_PER_WAVE is 1 (bm32/bm16) or 4 (bm64) for all live shapes.
            static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
                if((mi.value & 1) == wave_id_n)
                    issue_one_mi(mi);
            });
        }
    };

    auto wait_a_payload = [&](auto pending_a_loads) {
        if(wave_id_n <= 1) {
            s_waitcnt_vmcnt(pending_a_loads);
        }
        __builtin_amdgcn_s_barrier();
    };

    auto load_a_payload = [&](auto stage, V_A (&v_a)[T::M_MFMA_PER_WAVE]) {
        constexpr int Stage = decltype(stage)::value;
        static_assert(Stage >= 0 && Stage < T::A_LDS_STAGES);

        static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
            auto lo = s_a[Stage].template load<T::VEC_A>(sa_offset(mi, 0_I));
            auto hi = s_a[Stage].template load<T::VEC_A>(sa_offset(mi, 1_I));
            opus_moe_stage2_a8w4_decode_pack_a_mfma_reg(
                __builtin_bit_cast(opus_moe_stage2_a8w4_decode_u32x4_t, lo),
                __builtin_bit_cast(opus_moe_stage2_a8w4_decode_u32x4_t, hi),
                v_a[mi.value]);
        });
    };

    auto load_b_half = [&](int n_half,
                           int tile_base,
                           V_B (&v_b)[T::HALF_N_MFMA_PER_WAVE]) {
        static_for<T::HALF_N_MFMA_PER_WAVE>([&](auto local_ni) {
            const int ni = n_half * T::HALF_N_MFMA_PER_WAVE + local_ni.value;
            const int b_scalar_offset =
                tile_base + b_wave_scalar_base +
                ni * b_ni_stride_bytes;
            auto value = g_b.template load<T::B_BYTES_PER_VEC>(
                b_lane_offset, b_scalar_offset, opus::number<T::CACHECTL_B>{});
            opus_moe_stage2_a8w4_decode_unpack_b_mfma_reg(
                __builtin_bit_cast(opus_moe_stage2_a8w4_decode_u32x4_t, value),
                v_b[local_ni.value]);
        });
    };

    auto load_a_scale = [&](int k_group_word_base,
                            int (&a_scale)[T::M_MFMA_PER_WAVE]) {
        static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
            const int word_offset = a_scale_base_word[mi.value] + k_group_word_base;
            const auto word = g_a_scale.template load<sizeof(uint32_t)>(
                word_offset * static_cast<int>(sizeof(uint32_t)),
                0,
                opus::number<T::CACHECTL_A>{});
            a_scale[mi.value] = static_cast<int>(__builtin_bit_cast(uint32_t, word));
        });
    };

    auto load_b_scale = [&](int k_group_word_base,
                            int (&b_scale)[T::HALF_N_MFMA_PER_WAVE]) {
        static_for<T::HALF_N_MFMA_PER_WAVE>([&](auto pair) {
            const int word_offset = opus_moe_stage2_a8w4_b_scale_word_offset<T>(
                b_scale_base_word,
                k_group_word_base,
                pair.value);
            const auto word = g_w_scale.template load<sizeof(uint32_t)>(
                word_offset * static_cast<int>(sizeof(uint32_t)),
                0,
                opus::number<T::CACHECTL_W_SCALE>{});
            b_scale[pair.value] = static_cast<int>(__builtin_bit_cast(uint32_t, word));
        });
    };

    auto compute_half = [&](auto scale_pair,
                            auto n_half,
                            const V_A (&v_a)[T::M_MFMA_PER_WAVE],
                            const int (&a_scale)[T::M_MFMA_PER_WAVE],
                            const int (&b_scale)[T::HALF_N_MFMA_PER_WAVE],
                            const V_B (&v_b)[T::HALF_N_MFMA_PER_WAVE]) {
        constexpr int ScalePair = decltype(scale_pair)::value;
        constexpr int NHalf = decltype(n_half)::value;
        static_assert(ScalePair == 0 || ScalePair == 1);
        static_assert(NHalf == 0 || NHalf == 1);

        static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
            static_for<T::HALF_N_MFMA_PER_WAVE>([&](auto local_ni) {
                constexpr int ni = NHalf * T::HALF_N_MFMA_PER_WAVE + local_ni.value;
                constexpr int b_sel = ScalePair * 2 + (ni & 1);
                constexpr int b_scale_index = ni / 2;
                if constexpr(T::M_MFMA_PER_WAVE == 1 && T::T_M == 2)
                {
                    constexpr int a_sel_base = ScalePair * 2;
                    if(wave_id_m == 0)
                    {
                        v_c[mi.value][ni] = mma(v_a[mi.value],
                                                v_b[local_ni.value],
                                                v_c[mi.value][ni],
                                                a_scale[mi.value],
                                                b_scale[b_scale_index],
                                                number<a_sel_base>{},
                                                number<b_sel>{});
                    }
                    else
                    {
                        v_c[mi.value][ni] = mma(v_a[mi.value],
                                                v_b[local_ni.value],
                                                v_c[mi.value][ni],
                                                a_scale[mi.value],
                                                b_scale[b_scale_index],
                                                number<a_sel_base + 1>{},
                                                number<b_sel>{});
                    }
                }
                else
                {
                    constexpr int a_sel = ScalePair * 2 + (mi.value & 1);
                    v_c[mi.value][ni] = mma(v_a[mi.value],
                                            v_b[local_ni.value],
                                            v_c[mi.value][ni],
                                            a_scale[mi.value],
                                            b_scale[b_scale_index],
                                            number<a_sel>{},
                                            number<b_sel>{});
                }
            });
        });
    };

    auto compute_tile = [&](auto scale_pair,
                            auto wait_for_pending_b_half1,
                            auto stage,
                            int b_tile_base,
                            const int (&b_scale)[T::HALF_N_MFMA_PER_WAVE],
                            const int (&a_scale)[T::M_MFMA_PER_WAVE]) {
        constexpr bool WaitForPendingBHalf1 =
            decltype(wait_for_pending_b_half1)::value != 0;

        V_A v_a[T::M_MFMA_PER_WAVE];
        V_B v_b_half0[T::HALF_N_MFMA_PER_WAVE];
        V_B v_b_half1[T::HALF_N_MFMA_PER_WAVE];

        load_a_payload(stage, v_a);
        load_b_half(0, b_tile_base, v_b_half0);
        load_b_half(1, b_tile_base, v_b_half1);

        if constexpr(WaitForPendingBHalf1)
            s_waitcnt_vmcnt(number<T::HALF_N_MFMA_PER_WAVE>{});

        __builtin_amdgcn_s_setprio(1);
        compute_half(scale_pair, 0_I, v_a, a_scale, b_scale, v_b_half0);

        if constexpr(WaitForPendingBHalf1)
            s_waitcnt_vmcnt(0_I);

        compute_half(scale_pair, 1_I, v_a, a_scale, b_scale, v_b_half1);
        __builtin_amdgcn_s_setprio(0);
    };

    opus_moe_stage2_a8w4_decode_run_k_scheduler_gfx950<T, V_A, V_B>(
        col_base,
        b_payload_row_stride_bytes,
        issue_a_payload,
        wait_a_payload,
        load_a_payload,
        load_b_half,
        load_b_scale,
        load_a_scale,
        compute_half,
        compute_tile);
}

// Epilogue: direct atomic output or route-out store.
typedef uint32_t opus_moe_stage2_a8w4_decode_u32x4_store_t
    __attribute__((ext_vector_type(4)));

inline __device__ void opus_moe_stage2_a8w4_decode_atomic_add_bf16x2(
    opus::bf16x2_t data,
    opus::i32x4_t out_rsrc,
    int byte_offset)
{
#if OPUS_HAS_BUFFER_ATOMIC_PK_ADD_BF16
    (void)opus::llvm_amdgcn_raw_buffer_atomic_fadd_v2bf16(
        data, out_rsrc, byte_offset, 0, 0);
#else
    (void)data;
    (void)out_rsrc;
    (void)byte_offset;
    __builtin_trap();
#endif
}

template<typename T, typename CAcc>
inline __device__ void opus_moe_stage2_a8w4_decode_write_direct_acc_to_smem(
    CAcc (&v_c)[T::M_MFMA_PER_WAVE][T::N_MFMA_PER_WAVE],
    const OpusMoeStage2A8W4CShuffleLayout<T>& c_layout,
    const int32_t* __restrict__ smem_route_base,
    const float* __restrict__ smem_weight,
    uint32_t* __restrict__ smem_c_pair)
{
    using namespace opus;

    static_assert(T::DIRECT_ATOMIC_OUT);

    auto* smem_c_bf16 = reinterpret_cast<hip_bfloat16*>(smem_c_pair);

    static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
        static_for<T::VEC_C>([&](auto ii) {
            const int local_m = c_layout.acc_local_m(mi.value, ii.value);
            if(smem_route_base[local_m] >= 0)
            {
                const float weight = smem_weight[local_m];
                static_for<T::N_MFMA_PER_WAVE>([&](auto ni) {
                    const int local_col = c_layout.acc_local_col(ni.value);
                    smem_c_bf16[c_layout.smem_scalar_index(local_m, local_col)] =
                        opus_moe_gfx950_cvt_bf16_f32(
                            static_cast<float>(v_c[mi.value][ni.value][ii.value]) *
                            weight);
                });
            }
        });
    });
}

template<typename T, typename CAcc>
inline __device__ void opus_moe_stage2_a8w4_decode_write_route_out_acc_to_smem(
    CAcc (&v_c)[T::M_MFMA_PER_WAVE][T::N_MFMA_PER_WAVE],
    const OpusMoeStage2A8W4CShuffleLayout<T>& c_layout,
    const float* __restrict__ smem_weight,
    uint32_t* __restrict__ smem_c_pair)
{
    using namespace opus;

    static_assert(!T::DIRECT_ATOMIC_OUT);

    auto* smem_c_bf16 = reinterpret_cast<hip_bfloat16*>(smem_c_pair);

    static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
        static_for<T::VEC_C>([&](auto ii) {
            const int local_m = c_layout.acc_local_m(mi.value, ii.value);
            const float weight = smem_weight[local_m];
            static_for<T::N_MFMA_PER_WAVE>([&](auto ni) {
                const int local_col = c_layout.acc_local_col(ni.value);
                smem_c_bf16[c_layout.smem_scalar_index(local_m, local_col)] =
                    opus_moe_gfx950_cvt_bf16_f32(
                        static_cast<float>(v_c[mi.value][ni.value][ii.value]) *
                        weight);
            });
        });
    });
}

template<typename T>
inline __device__ void opus_moe_stage2_a8w4_decode_atomic_smem_to_out(
    const uint32_t* __restrict__ smem_c_pair,
    const int32_t* __restrict__ smem_route_base,
    const OpusMoeStage2A8W4CShuffleLayout<T>& c_layout,
    int col_base,
    int64_t output_row_stride,
    opus::i32x4_t out_rsrc)
{
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

    const int col0 = c_layout.atomic_col0();

    #pragma unroll
    for(int mr = 0; mr < T::B_M / CSHUFFLE_MLANE; ++mr)
    {
        const int local_m = c_layout.atomic_local_m(mr);
        if(smem_route_base[local_m] >= 0)
        {
            const int token = smem_route_base[local_m];
            const int pair_base = c_layout.smem_pair_index(local_m, col0);
            const int byte_offset =
                c_layout.output_byte_offset(token, output_row_stride, col_base, col0);
            opus::static_for<ATOMIC_GROUPS>([&](auto group) {
                constexpr int pair_delta = group.value * CSHUFFLE_NLANE;
                constexpr int byte_delta = pair_delta * static_cast<int>(sizeof(uint32_t));
                const auto data = __builtin_bit_cast(
                    opus::bf16x2_t, smem_c_pair[pair_base + pair_delta]);
                opus_moe_stage2_a8w4_decode_atomic_add_bf16x2(
                    data, out_rsrc, byte_offset + byte_delta);
            });
        }
    }
}

template<typename T>
inline __device__ void opus_moe_stage2_a8w4_decode_store_smem_to_route_out(
    const uint32_t* __restrict__ smem_c_pair,
    const int32_t* __restrict__ smem_route_base,
    const OpusMoeStage2A8W4CShuffleLayout<T>& c_layout,
    int col_base,
    hip_bfloat16* __restrict__ out,
    int64_t output_row_stride)
{
    constexpr int PAIRS_PER_ROW =
        OpusMoeStage2A8W4CShuffleLayout<T>::PAIRS_PER_ROW;
    constexpr int PAIRS_PER_VECTOR = 4;
    constexpr int THREADS_PER_ROW = PAIRS_PER_ROW / PAIRS_PER_VECTOR;
    constexpr int ROWS_PER_ITER = T::BLOCK_SIZE / THREADS_PER_ROW;
    static_assert(PAIRS_PER_ROW % PAIRS_PER_VECTOR == 0);
    static_assert(T::BLOCK_SIZE % THREADS_PER_ROW == 0);
    static_assert(T::B_M % ROWS_PER_ITER == 0);

    const int tid = c_layout.tid;
    const int row_in_iter = tid / THREADS_PER_ROW;
    const int pair_col = (tid - row_in_iter * THREADS_PER_ROW) * PAIRS_PER_VECTOR;
    const int col0 = pair_col * T::ELEM_PER_ATOMIC;

    auto store_row = [&](int local_m, int route_row) {
        const int pair_base = c_layout.smem_pair_index(local_m, col0);
        const opus_moe_stage2_a8w4_decode_u32x4_store_t data{
            smem_c_pair[pair_base + 0],
            smem_c_pair[pair_base + 1],
            smem_c_pair[pair_base + 2],
            smem_c_pair[pair_base + 3]};
        hip_bfloat16* row_ptr =
            out + static_cast<int64_t>(route_row) * output_row_stride + col_base + col0;
        auto* row_pair = reinterpret_cast<uint32_t*>(row_ptr);
        __builtin_nontemporal_store(
            data,
            reinterpret_cast<opus_moe_stage2_a8w4_decode_u32x4_store_t*>(
                row_pair));
    };

    #pragma unroll
    for(int row_iter = 0; row_iter < T::B_M / ROWS_PER_ITER; ++row_iter)
    {
        const int local_m = row_iter * ROWS_PER_ITER + row_in_iter;
        const int route_row = smem_route_base[local_m];
        if(route_row >= 0)
            store_row(local_m, route_row);
    }
}

// MXFP8 route_out store: fp8 e4m3 + per-8col e8m0 (1-byte) scale. Per-thread
// scale (each thread owns 8 cols) -> NO cross-lane reduction. Row layout:
// [model_dim fp8 | model_dim/8 e8m0 scale bytes]. 0.56x of bf16 -> store ~ -44%.
template<typename T>
inline __device__ void opus_moe_stage2_a8w4_decode_store_smem_to_route_out_fp8(
    const uint32_t* __restrict__ smem_c_pair,
    const int32_t* __restrict__ smem_route_base,
    const OpusMoeStage2A8W4CShuffleLayout<T>& c_layout,
    int col_base,
    uint8_t* __restrict__ out_base,
    int64_t row_stride_bytes,
    int scale_col_off)
{
    constexpr int PAIRS_PER_ROW =
        OpusMoeStage2A8W4CShuffleLayout<T>::PAIRS_PER_ROW;
    constexpr int PAIRS_PER_VECTOR = 4;
    constexpr int THREADS_PER_ROW = PAIRS_PER_ROW / PAIRS_PER_VECTOR;
    constexpr int ROWS_PER_ITER = T::BLOCK_SIZE / THREADS_PER_ROW;
    static_assert(T::ELEM_PER_ATOMIC == 2);

    const int tid = c_layout.tid;
    const int row_in_iter = tid / THREADS_PER_ROW;
    const int lane_in_row = tid % THREADS_PER_ROW;
    const int pair_col = lane_in_row * PAIRS_PER_VECTOR;
    const int col0 = pair_col * T::ELEM_PER_ATOMIC; // 8 cols owned by this thread

    auto store_row = [&](int local_m, int route_row) {
        const int pair_base = c_layout.smem_pair_index(local_m, col0);
        uint32_t p[PAIRS_PER_VECTOR];
        uint32_t amax_bits = 0; // 15-bit bf16 magnitude
        #pragma unroll
        for(int i = 0; i < PAIRS_PER_VECTOR; ++i)
        {
            p[i] = smem_c_pair[pair_base + i];
            amax_bits = max(amax_bits, max(p[i] & 0x7fffu, (p[i] >> 16) & 0x7fffu));
        }
        // e8m0 scale: pick power-of-2 s=2^(E-127) so amax/s in (128,256] < 448(e4m3 max).
        const int ax_e = (amax_bits >> 7) & 0xff;        // biased bf16 exponent
        int E = ax_e - 7;
        if(amax_bits == 0)
            E = 0;
        else if(E < 1)
            E = 1;
        // gfx950 HW scaled-cvt: fp8 = bf16 / s, s = 2^(E-127) = bitcast(E<<23). Power-of-2
        // scaling is exact, so this is bit-identical to (bf16->f32)*2^(127-E)->fp8 but skips
        // 8x(bf16->f32) + 8x(*inv) per row -> fewer VALU.
        const float scale = (amax_bits == 0)
                                ? 1.0f
                                : __builtin_bit_cast(float, static_cast<uint32_t>(E) << 23);
        typedef unsigned short ush2 __attribute__((ext_vector_type(2)));
        typedef short s2 __attribute__((ext_vector_type(2)));
        s2 a0{0, 0}, a1{0, 0};
        a0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(a0, __builtin_bit_cast(ush2, p[0]), scale, false);
        a0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(a0, __builtin_bit_cast(ush2, p[1]), scale, true);
        a1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(a1, __builtin_bit_cast(ush2, p[2]), scale, false);
        a1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(a1, __builtin_bit_cast(ush2, p[3]), scale, true);
        const uint32_t w0 = __builtin_bit_cast(uint32_t, a0);
        const uint32_t w1 = __builtin_bit_cast(uint32_t, a1);
        uint8_t* rowp = out_base + static_cast<int64_t>(route_row) * row_stride_bytes;
        const uint64_t packed8 =
            static_cast<uint64_t>(w0) | (static_cast<uint64_t>(w1) << 32);
        __builtin_nontemporal_store(
            packed8, reinterpret_cast<uint64_t*>(rowp + col_base + col0));
        rowp[scale_col_off + ((col_base + col0) >> 3)] = static_cast<uint8_t>(E);
    };

    #pragma unroll
    for(int row_iter = 0; row_iter < T::B_M / ROWS_PER_ITER; ++row_iter)
    {
        const int local_m = row_iter * ROWS_PER_ITER + row_in_iter;
        const int route_row = smem_route_base[local_m];
        if(route_row >= 0)
            store_row(local_m, route_row);
    }
}

template<typename T, typename CAcc>
inline __device__ void opus_moe_stage2_a8w4_decode_direct_epilogue(
    CAcc (&v_c)[T::M_MFMA_PER_WAVE][T::N_MFMA_PER_WAVE],
    const OpusMoeStage2A8W4CShuffleLayout<T>& c_layout,
    const int32_t* __restrict__ smem_route_base,
    const float* __restrict__ smem_weight,
    uint32_t* __restrict__ smem_c_pair,
    int col_base,
    int64_t output_row_stride,
    opus::i32x4_t output_rsrc)
{
    using namespace opus;
    using opus::operator""_I;

    static_assert(T::B_N == T::C_LDS_N);
    static_assert(T::DIRECT_ATOMIC_OUT);

    opus_moe_stage2_a8w4_decode_write_direct_acc_to_smem<T>(
        v_c,
        c_layout,
        smem_route_base,
        smem_weight,
        smem_c_pair);
    s_waitcnt_lgkmcnt(0_I);
    __syncthreads();
    opus_moe_stage2_a8w4_decode_atomic_smem_to_out<T>(
        smem_c_pair,
        smem_route_base,
        c_layout,
        col_base,
        output_row_stride,
        output_rsrc);
}

template<typename T, typename CAcc>
inline __device__ void opus_moe_stage2_a8w4_decode_route_out_epilogue(
    CAcc (&v_c)[T::M_MFMA_PER_WAVE][T::N_MFMA_PER_WAVE],
    const OpusMoeStage2A8W4CShuffleLayout<T>& c_layout,
    const int32_t* __restrict__ smem_route_base,
    const float* __restrict__ smem_weight,
    uint32_t* __restrict__ smem_c_pair,
    int col_base,
    hip_bfloat16* __restrict__ out,
    int64_t output_row_stride)
{
    using namespace opus;
    using opus::operator""_I;

    static_assert(T::B_N == T::C_LDS_N);
    static_assert(!T::DIRECT_ATOMIC_OUT);

    opus_moe_stage2_a8w4_decode_write_route_out_acc_to_smem<T>(
        v_c,
        c_layout,
        smem_weight,
        smem_c_pair);
    s_waitcnt_lgkmcnt(0_I);
    __syncthreads();
    opus_moe_stage2_a8w4_decode_store_smem_to_route_out<T>(
        smem_c_pair,
        smem_route_base,
        c_layout,
        col_base,
        out,
        output_row_stride);
}

#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__

// Kernel entry.
template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_BLOCKS_PER_CU) void
opus_moe_stage2_a8w4_decode_kernel_gfx950(opus_moe_stage2_a8w4_kargs kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
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
    __shared__ __align__(T::BYTES_PER_VEC) char smem_scratch[SCRATCH_BYTES];
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
        static_cast<unsigned int>(static_cast<unsigned long long>(kargs.a_scale_rows) *
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
    const int b_payload_row_stride_bytes = static_cast<int>(kargs.stride_w_h);
    auto u_gb = opus_moe_stage2_a8w4_layout_gb<T>(
        lane_id, wave_id_n, b_payload_row_stride_bytes);
    auto u_c = opus_moe_stage2_a8w4_layout_c<T>(wave_id_m, wave_id_n);

    typename decltype(mma)::vtype_c v_c[T::M_MFMA_PER_WAVE][T::N_MFMA_PER_WAVE];
    static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
        static_for<T::N_MFMA_PER_WAVE>([&](auto ni) {
            clear(v_c[mi.value][ni.value]);
        });
    });

    opus_moe_stage2_a8w4_decode_mainloop<T>(mma,
                                            u_ga,
                                            u_sa,
                                            s_a,
                                            u_gb,
                                            g_a,
                                            g_a_scale,
                                            g_b,
                                            g_w_scale,
                                            smem_a_base,
                                            route_base,
                                            col_base,
                                            b_payload_row_stride_bytes,
                                            wave_id_m,
                                            wave_id_n,
                                            scale_row_col_base,
                                            v_c);

    if constexpr(!Schedule::MainloopEndsWithSmemBarrier)
    {
        __syncthreads();
    }
    if constexpr(T::DIRECT_ATOMIC_OUT)
    {
        constexpr int output_rows_per_token = 1;
        const unsigned int output_size_bytes = static_cast<unsigned int>(
            static_cast<unsigned long long>(token_num) *
            static_cast<unsigned long long>(output_rows_per_token) *
            static_cast<unsigned long long>(kargs.stride_o_t) *
            static_cast<unsigned long long>(sizeof(hip_bfloat16)));
        const auto output_ptr_bits = reinterpret_cast<__UINTPTR_TYPE__>(kargs.out_bf16);
        const opus::i32x4_t output_rsrc{
            static_cast<int>(static_cast<unsigned int>(output_ptr_bits)),
            static_cast<int>((static_cast<unsigned long long>(output_ptr_bits) >> 32) &
                             0xffffu),
            static_cast<int>(output_size_bytes),
            static_cast<int>(opus::buffer_default_config())};
        opus_moe_stage2_a8w4_decode_direct_epilogue<T>(
            v_c,
            u_c,
            smem_route_base,
            smem_weight,
            smem_c_pair,
            col_base,
            kargs.stride_o_t,
            output_rsrc);
    }
    else if(kargs.route_out_fp8)
    {
        opus_moe_stage2_a8w4_decode_write_route_out_acc_to_smem<T>(
            v_c, u_c, smem_weight, smem_c_pair);
        s_waitcnt_lgkmcnt(0_I);
        __syncthreads();
        opus_moe_stage2_a8w4_decode_store_smem_to_route_out_fp8<T>(
            smem_c_pair, smem_route_base, u_c, col_base,
            reinterpret_cast<uint8_t*>(kargs.out_bf16),
            kargs.route_out_row_bytes, kargs.model_dim);
    }
    else
    {
        opus_moe_stage2_a8w4_decode_route_out_epilogue<T>(
            v_c,
            u_c,
            smem_route_base,
            smem_weight,
            smem_c_pair,
            col_base,
            kargs.out_bf16,
            kargs.stride_o_t);
    }
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}
