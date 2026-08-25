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

template<typename T, int ScalePair, int Mi, typename Fn>
inline __device__ void opus_moe_stage2_a8w4_decode_with_a_selector(
    int route_base,
    int wave_id_m,
    const Fn& run)
{
    constexpr int a_sel_base = ScalePair * 2;
    if constexpr(T::M_MFMA_PER_WAVE == 1)
    {
        const int a_half = T::T_M == 1
                               ? ((route_base / T::MMA_M) & 1)
                               : wave_id_m;
        if(a_half == 0)
            run(opus::number<a_sel_base>{});
        else
            run(opus::number<a_sel_base + 1>{});
    }
    else
    {
        run(opus::number<a_sel_base + (Mi & 1)>{});
    }
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
                a_base = kargs.stride_a_k == 0
                             ? static_cast<int32_t>(
                                   static_cast<int64_t>(row) * kargs.stride_a_t)
                             : static_cast<int32_t>(
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

// Runtime-K pipeline with a fixed-size LDS ring and compile-time slot indices.
template<typename T,
         typename IssueAPayload,
         typename WaitAPayload,
         typename LoadBScale,
         typename LoadAScale,
         typename ComputeTile>
inline __device__ void opus_moe_stage2_a8w4_decode_run_k_pipeline_gfx950(
    int k_tiles,
    int col_base,
    int b_payload_row_stride_bytes,
    IssueAPayload& issue_a_payload,
    WaitAPayload& wait_a_payload,
    LoadBScale& load_b_scale,
    LoadAScale& load_a_scale,
    ComputeTile& compute_tile)
{
    using namespace opus;

    static_assert(T::A_LDS_STAGES == 2 * T::PAIR_SLOTS);
    constexpr int TILES_PER_CHUNK = T::A_LDS_STAGES;
    constexpr int B_TILE_BYTE_STEP =
        T::K_STEP_PACKED * T::B_PAYLOAD_K_STRIDE_BYTES;

    // Seed the guaranteed first K pair, then predicate optional ring slots.
    issue_a_payload(0_I, 0);
    issue_a_payload(1_I, T::K_STEP_PACKED);
    if constexpr(T::A_LDS_STAGES == 4)
    {
        if(k_tiles > 2)
        {
            issue_a_payload(2_I, 2 * T::K_STEP_PACKED);
            if(k_tiles > 3)
                issue_a_payload(3_I, 3 * T::K_STEP_PACKED);
        }
    }

    // Drain directly when K fits the ring and no slot is reused.
    if(__builtin_expect(k_tiles <= TILES_PER_CHUNK, 1))
    {
        const int first_b_tile_base = col_base * b_payload_row_stride_bytes;
        static_for<T::PAIR_SLOTS>([&](auto pair_slot) {
            constexpr int EvenStage = 2 * pair_slot.value;
            constexpr int OddStage = EvenStage + 1;
            if(k_tiles <= EvenStage)
                return;

            int b_scale[T::HALF_N_MFMA_PER_WAVE];
            int a_scale[T::M_MFMA_PER_WAVE];
            constexpr int ScaleWordBase =
                pair_slot.value * T::SCALE_WORDS_PER_GROUP_PACK;
            if constexpr(T::OVERLAP_SHORT_A_STAGES && pair_slot.value == 0)
            {
                constexpr int PendingAStageLoads =
                    (T::A_LDS_STAGES - 1) * T::M_MFMA_PER_WAVE;
                if(k_tiles < TILES_PER_CHUNK)
                    wait_a_payload(0_I);
                else
                    wait_a_payload(number<PendingAStageLoads>{});
            }
            load_b_scale(ScaleWordBase, b_scale);
            load_a_scale(ScaleWordBase, a_scale);
            if constexpr(pair_slot.value == 0 && !T::OVERLAP_SHORT_A_STAGES)
            {
                constexpr int PendingScaleLoads =
                    T::HALF_N_MFMA_PER_WAVE + T::M_MFMA_PER_WAVE;
                wait_a_payload(number<PendingScaleLoads>{});
            }

            compute_tile(0_I,
                         number<pair_slot.value == 0>{},
                         0_I,
                         number<EvenStage>{},
                         first_b_tile_base + EvenStage * B_TILE_BYTE_STEP,
                         b_scale,
                         a_scale);
            if constexpr(T::OVERLAP_SHORT_A_STAGES && pair_slot.value == 0)
                wait_a_payload(0_I);
            if(k_tiles > OddStage)
            {
                compute_tile(1_I,
                             0_I,
                             0_I,
                             number<OddStage>{},
                             first_b_tile_base + OddStage * B_TILE_BYTE_STEP,
                             b_scale,
                             a_scale);
            }
        });
        return;
    }

    if constexpr(T::STEADY_PAIR_SLOTS == 1)
    {
        const int pair_count = (k_tiles + 1) / 2;
        int b_tile_base = col_base * b_payload_row_stride_bytes;
        int scale_word_base = 0;
        #pragma unroll 1
        for(int pair = 0; pair < pair_count;
            ++pair,
            b_tile_base += 2 * B_TILE_BYTE_STEP,
            scale_word_base += T::SCALE_WORDS_PER_GROUP_PACK)
        {
            int b_scale[T::HALF_N_MFMA_PER_WAVE];
            int a_scale[T::M_MFMA_PER_WAVE];
            load_b_scale(scale_word_base, b_scale);
            load_a_scale(scale_word_base, a_scale);
            constexpr int PendingScaleLoads =
                T::HALF_N_MFMA_PER_WAVE + T::M_MFMA_PER_WAVE;
            wait_a_payload(number<PendingScaleLoads>{});

            compute_tile(0_I, 0_I, 0_I, 0_I, b_tile_base, b_scale, a_scale);
            const int next_even_tile = 2 * pair + 2;
            if(next_even_tile < k_tiles)
            {
                __builtin_amdgcn_s_barrier();
                issue_a_payload(0_I, next_even_tile * T::K_STEP_PACKED);
            }

            const int odd_tile = 2 * pair + 1;
            if(odd_tile < k_tiles)
            {
                compute_tile(1_I,
                             0_I,
                             0_I,
                             1_I,
                             b_tile_base + B_TILE_BYTE_STEP,
                             b_scale,
                             a_scale);
                const int next_odd_tile = next_even_tile + 1;
                if(next_odd_tile < k_tiles)
                {
                    __builtin_amdgcn_s_barrier();
                    issue_a_payload(1_I, next_odd_tile * T::K_STEP_PACKED);
                }
            }
        }
    }
    else
    {
        int chunk_tile_base = 0;
        int chunk_k_base = 0;
        int chunk_b_tile_base = col_base * b_payload_row_stride_bytes;
        int chunk_scale_word_base = 0;

        #pragma unroll 1
        for(; chunk_tile_base < k_tiles;
            chunk_tile_base += TILES_PER_CHUNK,
            chunk_k_base += TILES_PER_CHUNK * T::K_STEP_PACKED,
            chunk_b_tile_base += TILES_PER_CHUNK * B_TILE_BYTE_STEP,
            chunk_scale_word_base +=
                T::PAIR_SLOTS * T::SCALE_WORDS_PER_GROUP_PACK)
        {
            static_for<T::PAIR_SLOTS>([&](auto pair_slot) {
                constexpr int EvenStage = 2 * pair_slot.value;
                constexpr int OddStage = EvenStage + 1;
                const int even_tile = chunk_tile_base + EvenStage;
                if(even_tile >= k_tiles)
                    return;

                int b_scale[T::HALF_N_MFMA_PER_WAVE];
                int a_scale[T::M_MFMA_PER_WAVE];
                const int scale_word_base =
                    chunk_scale_word_base +
                    pair_slot.value * T::SCALE_WORDS_PER_GROUP_PACK;
                load_b_scale(scale_word_base, b_scale);
                load_a_scale(scale_word_base, a_scale);
                if constexpr(pair_slot.value == 0)
                {
                    constexpr int PendingScaleLoads =
                        T::HALF_N_MFMA_PER_WAVE + T::M_MFMA_PER_WAVE;
                    wait_a_payload(number<PendingScaleLoads>{});
                }

                const int even_b_tile_base =
                    chunk_b_tile_base + EvenStage * B_TILE_BYTE_STEP;
                compute_tile(0_I,
                             0_I,
                             0_I,
                             number<EvenStage>{},
                             even_b_tile_base,
                             b_scale,
                             a_scale);
                const int next_even_tile = even_tile + TILES_PER_CHUNK;
                if(next_even_tile < k_tiles)
                {
                    __builtin_amdgcn_s_barrier();
                    issue_a_payload(number<EvenStage>{},
                                    chunk_k_base +
                                        (TILES_PER_CHUNK + EvenStage) *
                                            T::K_STEP_PACKED);
                }

                const int odd_tile = even_tile + 1;
                if(odd_tile < k_tiles)
                {
                    const int odd_b_tile_base =
                        chunk_b_tile_base + OddStage * B_TILE_BYTE_STEP;
                    compute_tile(1_I,
                                 0_I,
                                 0_I,
                                 number<OddStage>{},
                                 odd_b_tile_base,
                                 b_scale,
                                 a_scale);
                    const int next_odd_tile = odd_tile + TILES_PER_CHUNK;
                    if(next_odd_tile < k_tiles)
                    {
                        __builtin_amdgcn_s_barrier();
                        issue_a_payload(number<OddStage>{},
                                        chunk_k_base +
                                            (TILES_PER_CHUNK + OddStage) *
                                                T::K_STEP_PACKED);
                    }
                }
            });
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

template<typename T,
         typename Mma,
         typename LayoutA,
         typename LayoutASmem,
         typename LayoutB,
         typename GmemA,
         typename GmemAScale,
         typename GmemB,
         typename GmemWScale>
inline __device__ void opus_moe_stage2_a8w4_decode_mainloop(
    Mma& mma,
    const LayoutA& u_ga,
    const LayoutASmem& u_sa,
    char* smem_a_scratch,
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
    const opus_moe_stage2_a8w4_kargs& kargs,
    typename Mma::vtype_c (&v_c)[T::M_MFMA_PER_WAVE][T::N_MFMA_PER_WAVE])
{
    using namespace opus;
    using opus::operator""_I;

    using V_A = typename Mma::mfma_type::vtype_a;
    using V_B = typename Mma::mfma_type::vtype_b;
    using D_A = typename T::D_A;

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
    auto make_smem_a_stage = [&](auto stage) {
        constexpr int Stage = decltype(stage)::value;
        static_assert(Stage >= 0 && Stage < T::A_LDS_STAGES);
        return opus::make_smem(reinterpret_cast<D_A*>(
            smem_a_scratch + Stage * T::A_LDS_STAGE_ELEMS *
                static_cast<int>(sizeof(D_A))));
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
            opus_moe_stage2_a8w4_a_scale_base_word_offset<T>(
                route_base,
                ga,
                kargs.a_scale_words_per_row_pack);
    });
    const int b_scale_base_word =
        opus_moe_stage2_a8w4_b_scale_base_word_offset<T>(
            scale_row_col_base,
            gb_offset(0_I),
            wave_id_n,
            kargs.w_scale_words_per_row_pack);
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
            auto s_a_stage = make_smem_a_stage(stage);
            auto* smem_lo = s_a_stage.ptr + sa_offset(mi, 0_I);
            auto* smem_hi = s_a_stage.ptr + sa_offset(mi, 1_I);
            if constexpr(T::IS_BM64_BN256 && T::BLOCK_SIZE == 256)
            {
                // Materialize BM64's lane-0 LDS base directly to avoid runtime-K address spills.
                constexpr int HalfStrideBytes =
                    opus::get_warp_size() * T::VEC_A * sizeof(D_A);
                constexpr int MiStrideBytes = 2 * HalfStrideBytes;
                smem_lo = s_a_stage.ptr + mi.value * MiStrideBytes;
                smem_hi = smem_lo + HalfStrideBytes;
            }
            const int ga = ga_offset(mi);
            const int a_offset_lo = opus_moe_stage2_a8w4_a_payload_byte_offset<T>(
                a_base[mi.value],
                k_base,
                ga);
            if constexpr(Schedule::Mainloop ==
                         MainloopSchedule::SplitALoadByNWave)
            {
                if(wave_id_n == 0)
                {
                    g_a.template async_load<T::VEC_A>(
                        smem_lo,
                        a_offset_lo,
                        0,
                        opus::number<T::CACHECTL_A>{});
                }
                else
                {
                    g_a.template async_load<T::VEC_A>(
                        smem_hi,
                        a_offset_lo + T::K_STEP_PACKED / 2,
                        0,
                        opus::number<T::CACHECTL_A>{});
                }
            }
            else
            {
                g_a.template async_load<T::VEC_A>(
                    smem_lo,
                    a_offset_lo,
                    0,
                    opus::number<T::CACHECTL_A>{});
                g_a.template async_load<T::VEC_A>(
                    smem_hi,
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

    auto load_a_fragment = [&](auto stage, auto mi, V_A& v_a) {
        constexpr int Stage = decltype(stage)::value;
        static_assert(Stage >= 0 && Stage < T::A_LDS_STAGES);
        auto s_a_stage = make_smem_a_stage(stage);

        auto lo = s_a_stage.template load<T::VEC_A>(sa_offset(mi, 0_I));
        auto hi = s_a_stage.template load<T::VEC_A>(sa_offset(mi, 1_I));
        opus_moe_stage2_a8w4_decode_pack_a_mfma_reg(
            __builtin_bit_cast(opus_moe_stage2_a8w4_decode_u32x4_t, lo),
            __builtin_bit_cast(opus_moe_stage2_a8w4_decode_u32x4_t, hi),
            v_a);
    };

    auto load_a_payload = [&](auto stage, V_A (&v_a)[T::M_MFMA_PER_WAVE]) {
        static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
            load_a_fragment(stage, mi, v_a[mi.value]);
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

    auto load_b_packed_fragment = [&](auto n_half,
                                      auto local_ni,
                                      int tile_base) {
        constexpr int NHalf = decltype(n_half)::value;
        constexpr int LocalNi = decltype(local_ni)::value;
        static_assert(NHalf == 0 || NHalf == 1);
        static_assert(LocalNi >= 0 && LocalNi < T::HALF_N_MFMA_PER_WAVE);
        constexpr int ni = NHalf * T::HALF_N_MFMA_PER_WAVE + LocalNi;
        const int b_scalar_offset =
            tile_base + b_wave_scalar_base + ni * b_ni_stride_bytes;
        auto value = g_b.template load<T::B_BYTES_PER_VEC>(
            b_lane_offset, b_scalar_offset, opus::number<T::CACHECTL_B>{});
        return __builtin_bit_cast(opus_moe_stage2_a8w4_decode_u32x4_t, value);
    };

    auto load_a_scale = [&](int k_group_word_base,
                            int (&a_scale)[T::M_MFMA_PER_WAVE]) {
        static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
            const int word_offset =
                a_scale_base_word[mi.value] + k_group_word_base;
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
                pair.value,
                kargs.w_scale_words_per_row_pack);
            const auto word = g_w_scale.template load<sizeof(uint32_t)>(
                word_offset * static_cast<int>(sizeof(uint32_t)),
                0,
                opus::number<T::CACHECTL_W_SCALE>{});
            b_scale[pair.value] = static_cast<int>(__builtin_bit_cast(uint32_t, word));
        });
    };

    auto compute_half = [&](auto scale_pair,
                            auto initialize_acc,
                            auto n_half,
                            const V_A (&v_a)[T::M_MFMA_PER_WAVE],
                            const int (&a_scale)[T::M_MFMA_PER_WAVE],
                            const int (&b_scale)[T::HALF_N_MFMA_PER_WAVE],
                            const V_B (&v_b)[T::HALF_N_MFMA_PER_WAVE]) {
        constexpr int ScalePair = decltype(scale_pair)::value;
        constexpr bool InitializeAcc =
            decltype(initialize_acc)::value != 0;
        constexpr int NHalf = decltype(n_half)::value;
        static_assert(ScalePair == 0 || ScalePair == 1);
        static_assert(NHalf == 0 || NHalf == 1);

        static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
            static_for<T::HALF_N_MFMA_PER_WAVE>([&](auto local_ni) {
                constexpr int ni = NHalf * T::HALF_N_MFMA_PER_WAVE + local_ni.value;
                constexpr int b_sel = ScalePair * 2 + (ni & 1);
                constexpr int b_scale_index = ni / 2;
                opus_moe_stage2_a8w4_decode_with_a_selector<
                    T, ScalePair, mi.value>(
                    route_base, wave_id_m, [&](auto a_sel) {
                        if constexpr(InitializeAcc)
                        {
                            typename Mma::vtype_c zero_acc{};
                            v_c[mi.value][ni] = mma(v_a[mi.value],
                                                    v_b[local_ni.value],
                                                    zero_acc,
                                                    a_scale[mi.value],
                                                    b_scale[b_scale_index],
                                                    a_sel,
                                                    number<b_sel>{});
                        }
                        else
                        {
                            v_c[mi.value][ni] = mma(v_a[mi.value],
                                                    v_b[local_ni.value],
                                                    v_c[mi.value][ni],
                                                    a_scale[mi.value],
                                                    b_scale[b_scale_index],
                                                    a_sel,
                                                    number<b_sel>{});
                        }
                    });
            });
        });
    };

    auto compute_fragment = [&](auto scale_pair,
                                auto initialize_acc,
                                auto n_half,
                                auto local_ni,
                                auto mi,
                                const V_A& v_a,
                                const int (&a_scale)[T::M_MFMA_PER_WAVE],
                                const int (&b_scale)[T::HALF_N_MFMA_PER_WAVE],
                                const V_B& v_b) {
        constexpr int ScalePair = decltype(scale_pair)::value;
        constexpr bool InitializeAcc =
            decltype(initialize_acc)::value != 0;
        constexpr int NHalf = decltype(n_half)::value;
        constexpr int LocalNi = decltype(local_ni)::value;
        constexpr int Mi = decltype(mi)::value;
        static_assert(ScalePair == 0 || ScalePair == 1);
        static_assert(NHalf == 0 || NHalf == 1);
        static_assert(LocalNi >= 0 && LocalNi < T::HALF_N_MFMA_PER_WAVE);
        static_assert(Mi >= 0 && Mi < T::M_MFMA_PER_WAVE);
        constexpr int ni = NHalf * T::HALF_N_MFMA_PER_WAVE + LocalNi;
        constexpr int b_sel = ScalePair * 2 + (ni & 1);
        constexpr int b_scale_index = ni / 2;

        opus_moe_stage2_a8w4_decode_with_a_selector<T, ScalePair, Mi>(
            route_base, wave_id_m, [&](auto a_sel) {
                if constexpr(InitializeAcc)
                {
                    typename Mma::vtype_c zero_acc{};
                    v_c[Mi][ni] = mma(v_a,
                                      v_b,
                                      zero_acc,
                                      a_scale[Mi],
                                      b_scale[b_scale_index],
                                      a_sel,
                                      number<b_sel>{});
                }
                else
                {
                    v_c[Mi][ni] = mma(v_a,
                                      v_b,
                                      v_c[Mi][ni],
                                      a_scale[Mi],
                                      b_scale[b_scale_index],
                                      a_sel,
                                      number<b_sel>{});
                }
            });
    };

    auto compute_tile = [&](auto scale_pair,
                            auto initialize_acc,
                            auto wait_for_pending_b_half1,
                            auto stage,
                            int b_tile_base,
                            const int (&b_scale)[T::HALF_N_MFMA_PER_WAVE],
                            const int (&a_scale)[T::M_MFMA_PER_WAVE]) {
        constexpr bool WaitForPendingBHalf1 =
            decltype(wait_for_pending_b_half1)::value != 0;

        if constexpr(T::IS_BM64_BN256)
        {
            // Reuse one expanded BM64 B operand to reduce VGPR pressure.
            V_A v_a[T::M_MFMA_PER_WAVE];
            opus_moe_stage2_a8w4_decode_u32x4_t
                b_packed[2][T::HALF_N_MFMA_PER_WAVE];
            V_B v_b_fragment;
            static_for<2>([&](auto n_half) {
                static_for<T::HALF_N_MFMA_PER_WAVE>([&](auto local_ni) {
                    b_packed[n_half.value][local_ni.value] =
                        load_b_packed_fragment(n_half, local_ni, b_tile_base);
                });
            });
            load_a_payload(stage, v_a);
            __builtin_amdgcn_s_setprio(1);
            static_for<2>([&](auto n_half) {
                static_for<T::HALF_N_MFMA_PER_WAVE>([&](auto local_ni) {
                    opus_moe_stage2_a8w4_decode_unpack_b_mfma_reg(
                        b_packed[n_half.value][local_ni.value], v_b_fragment);
                    static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
                        compute_fragment(scale_pair,
                                         initialize_acc,
                                         n_half,
                                         local_ni,
                                         mi,
                                         v_a[mi.value],
                                         a_scale,
                                         b_scale,
                                         v_b_fragment);
                    });
                });
                __builtin_amdgcn_sched_barrier(0);
            });
            __builtin_amdgcn_s_setprio(0);
        }
        else
        {
            V_A v_a[T::M_MFMA_PER_WAVE];
            V_B v_b_half0[T::HALF_N_MFMA_PER_WAVE];
            V_B v_b_half1[T::HALF_N_MFMA_PER_WAVE];
            load_a_payload(stage, v_a);
            load_b_half(0, b_tile_base, v_b_half0);
            load_b_half(1, b_tile_base, v_b_half1);

            if constexpr(WaitForPendingBHalf1)
                s_waitcnt_vmcnt(number<T::HALF_N_MFMA_PER_WAVE>{});

            __builtin_amdgcn_s_setprio(1);
            compute_half(
                scale_pair, initialize_acc, 0_I, v_a, a_scale, b_scale, v_b_half0);

            if constexpr(WaitForPendingBHalf1)
                s_waitcnt_vmcnt(0_I);

            compute_half(
                scale_pair, initialize_acc, 1_I, v_a, a_scale, b_scale, v_b_half1);
            __builtin_amdgcn_s_setprio(0);
        }
    };

    // Streaming paths require explicit accumulator initialization.
    if(kargs.k_tiles > T::A_LDS_STAGES)
    {
        static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
            static_for<T::N_MFMA_PER_WAVE>([&](auto ni) {
                clear(v_c[mi.value][ni.value]);
            });
        });
    }

    opus_moe_stage2_a8w4_decode_run_k_pipeline_gfx950<T>(
        kargs.k_tiles,
        col_base,
        b_payload_row_stride_bytes,
        issue_a_payload,
        wait_a_payload,
        load_b_scale,
        load_a_scale,
        compute_tile);

    // Synchronize before reusing the A-ring LDS allocation for the C-shuffle epilogue.
    __builtin_amdgcn_s_barrier();
}

// Epilogue: direct atomic output or route-out store.
typedef uint32_t opus_moe_stage2_a8w4_decode_u32x4_store_t
    __attribute__((ext_vector_type(4)));

template<typename T, bool CheckRoute, typename CAcc>
inline __device__ void opus_moe_stage2_a8w4_decode_write_acc_to_smem(
    CAcc (&v_c)[T::M_MFMA_PER_WAVE][T::N_MFMA_PER_WAVE],
    const OpusMoeStage2A8W4CShuffleLayout<T>& c_layout,
    const int32_t* __restrict__ smem_route_base,
    const float* __restrict__ smem_weight,
    uint32_t* __restrict__ smem_c_pair)
{
    using namespace opus;

    auto* smem_c_bf16 = reinterpret_cast<hip_bfloat16*>(smem_c_pair);

    static_for<T::M_MFMA_PER_WAVE>([&](auto mi) {
        static_for<T::VEC_C>([&](auto ii) {
            const int local_m = c_layout.acc_local_m(mi.value, ii.value);
            if constexpr(CheckRoute)
            {
                if(smem_route_base[local_m] < 0)
                    return;
            }
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
            const int byte0 =
                c_layout.output_elem_offset(token, output_row_stride, col_base, col0) *
                static_cast<int>(sizeof(opus::bf16_t));
            opus::static_for<ATOMIC_GROUPS>([&](auto group) {
                constexpr int pair_delta = group.value * CSHUFFLE_NLANE;
                constexpr int byte_delta =
                    pair_delta * T::ELEM_PER_ATOMIC * static_cast<int>(sizeof(opus::bf16_t));
                const auto data = __builtin_bit_cast(
                    opus::bf16x2_t, smem_c_pair[pair_base + pair_delta]);
                opus::llvm_amdgcn_raw_buffer_atomic_fadd_v2bf16(
                    data, out_rsrc, byte0 + byte_delta, 0, 0);
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
    opus::i32x4_t out_rsrc)
{
    using namespace opus;
    using opus::operator""_I;

    static_assert(T::B_N == T::C_LDS_N);
    static_assert(T::DIRECT_ATOMIC_OUT);

    opus_moe_stage2_a8w4_decode_write_acc_to_smem<T, true>(
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
        out_rsrc);
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

    opus_moe_stage2_a8w4_decode_write_acc_to_smem<T, false>(
        v_c,
        c_layout,
        smem_route_base,
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

template<typename T, typename CAcc>
inline __device__ void opus_moe_stage2_a8w4_decode_route_out_fp8_epilogue(
    CAcc (&v_c)[T::M_MFMA_PER_WAVE][T::N_MFMA_PER_WAVE],
    const OpusMoeStage2A8W4CShuffleLayout<T>& c_layout,
    const int32_t* __restrict__ smem_route_base,
    const float* __restrict__ smem_weight,
    uint32_t* __restrict__ smem_c_pair,
    int col_base,
    uint8_t* __restrict__ out_base,
    int64_t row_stride_bytes,
    int scale_col_off)
{
    using namespace opus;
    using opus::operator""_I;

    static_assert(T::B_N == T::C_LDS_N);
    static_assert(!T::DIRECT_ATOMIC_OUT);

    opus_moe_stage2_a8w4_decode_write_acc_to_smem<T, false>(
        v_c,
        c_layout,
        smem_route_base,
        smem_weight,
        smem_c_pair);
    s_waitcnt_lgkmcnt(0_I);
    __syncthreads();
    opus_moe_stage2_a8w4_decode_store_smem_to_route_out_fp8<T>(
        smem_c_pair,
        smem_route_base,
        c_layout,
        col_base,
        out_base,
        row_stride_bytes,
        scale_col_off);
}

#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__

// Kernel entry.
template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_BLOCKS_PER_CU) void
opus_moe_stage2_a8w4_decode_kernel_gfx950(
    opus_moe_stage2_a8w4_kargs kargs)
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
    // Use shifts for the supported power-of-two sort blocks.
    const int sorted_block_id =
        route_base >> __builtin_ctz(static_cast<unsigned>(kargs.sort_block_m));
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

    auto mma = make_tiled_mma<D_MFMA_A, D_MFMA_B, D_ACC>(
        seq<1, 1, 1>{},
        seq<1, 1, 1>{},
        seq<T::MMA_M, T::MMA_N, T::MMA_K>{});

    const D_A* __restrict__ inter_states =
        reinterpret_cast<const D_A*>(kargs.inter_states_fp8);
    const uint8_t* __restrict__ w2 = kargs.w2_fp4;
    const uint8_t* __restrict__ a2_scale = kargs.a2_scale_e8m0;
    const uint8_t* __restrict__ w2_scale = kargs.w2_scale_e8m0;
    const unsigned int a_size_bytes = static_cast<unsigned int>(
        static_cast<unsigned long long>(kargs.stride_a_k == 0 ? sorted_rows : token_num) *
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

    typename decltype(mma)::vtype_c v_c[T::M_MFMA_PER_WAVE][T::N_MFMA_PER_WAVE];

    opus_moe_stage2_a8w4_decode_mainloop<T>(mma,
                                            u_ga,
                                            u_sa,
                                            smem_scratch,
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
                                            kargs,
                                            v_c);

    if constexpr(!Schedule::MainloopEndsWithSmemBarrier)
    {
        __syncthreads();
    }
    auto u_c = opus_moe_stage2_a8w4_layout_c<T>(wave_id_m, wave_id_n);
    if constexpr(T::DIRECT_ATOMIC_OUT)
    {
        constexpr int output_rows_per_token = 1;
        const unsigned int output_size_bytes = static_cast<unsigned int>(
            static_cast<unsigned long long>(token_num) *
            static_cast<unsigned long long>(output_rows_per_token) *
            static_cast<unsigned long long>(kargs.stride_o_t) *
            static_cast<unsigned long long>(sizeof(hip_bfloat16)));
        // Build the output buffer descriptor as a plain i32x4 rather than via
        // make_gmem/make_buffer_rsrc: the __builtin_amdgcn_make_buffer_rsrc
        // intrinsic result is held across the atomic loop at +2 VGPR, which tips
        // the zero-headroom atomic decode kernels into 32 B/lane scratch (~4-7%
        // regression). The plain descriptor packs into cheap registers. The
        // atomic itself still uses the opus raw-buffer primitive.
        const auto output_ptr_bits = reinterpret_cast<__UINTPTR_TYPE__>(kargs.out_bf16);
        const opus::i32x4_t out_rsrc{
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
            out_rsrc);
    }
    else if(kargs.route_out_fp8)
    {
        opus_moe_stage2_a8w4_decode_route_out_fp8_epilogue<T>(
            v_c,
            u_c,
            smem_route_base,
            smem_weight,
            smem_c_pair,
            col_base,
            reinterpret_cast<uint8_t*>(kargs.out_bf16),
            kargs.route_out_row_bytes,
            kargs.model_dim);
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
