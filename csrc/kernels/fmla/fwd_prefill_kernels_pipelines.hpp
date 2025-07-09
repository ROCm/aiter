// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "fwd_prefill_kernels_policy.hpp"

// =====================================================================================================================
// Kernel Pipeline Functions
//

// This function get the range of seqlen for the specified `split_idx`. `granularity` is the granularity of group of
// workload which cannot be further subdivded.
// The workload is divided as evenly as possible. When the workload cannot be evenly divided by num_splits, the
// high-ranking splits will get 1 additional `granularity` of tasks.
// E.g. when `num_seqlen` is `28`, `granularity` is `2` and `num_splits` is `3`, the 3 splits will be assigned the
// following tasks:
// split.0: [0, 10)  // 10 workloads
// split.1: [10, 20) // 10 workloads
// split.2: [20, 28) //  8 workloads
// split.3: [28, 36) // Note that this may not be what you're expecting. upper_bound may be helpful in this case.
CK_TILE_DEVICE static auto GetSeqlenRange(
    const int32_t num_seqlen,
    const int32_t granularity,
    const int32_t num_splits,
    const int32_t split_idx,
    const int32_t lower_bound,
    const int32_t upper_bound)
{
    const int32_t num_workload = ck_tile::integer_divide_ceil(num_seqlen, granularity);
    const int32_t base_workload = ck_tile::integer_divide_floor(num_workload, num_splits);
    const int32_t addition_threshold = num_workload % num_splits;
    const int32_t start = base_workload * split_idx + ck_tile::min(addition_threshold, split_idx);
    const int32_t count = base_workload + ((split_idx < addition_threshold) ? 1 : 0);
    const int32_t end = start + count;

    return ck_tile::make_tuple(ck_tile::max(lower_bound, start * granularity),
                               ck_tile::min(upper_bound, end * granularity));
}

template <typename Policy, typename SaccType>
CK_TILE_DEVICE static auto ShuffleSacc(
    const SaccType& s_acc,
    uint8_t*        p_smem)
{
    using Traits = typename Policy::Traits;

    constexpr int32_t kWarpGemmM = Traits::QKWarpTile::at(ck_tile::number<0>{});
    constexpr int32_t ColWarps = Traits::kBlockM / kWarpGemmM;
    constexpr int32_t RowWarps = Traits::kNumWarps / ColWarps;

    if constexpr (RowWarps > 1)
    {
        // P shuffle LDS windows
        auto p_shuffle_lds = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            reinterpret_cast<SaccType::DataType*>(p_smem), Policy::MakePShuffleLdsBlockDescriptor());
        auto p_st_lds_window =
            ck_tile::make_tile_window(
                p_shuffle_lds,
                ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN0>{}),
                {0, 0});
        auto p_ld_lds_window =
            ck_tile::make_tile_window(
                p_shuffle_lds,
                ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN0>{}),
                {0, 0},
                Policy::GetKVBlockGemm().MakeABlockTileDistribution());

        if (Policy::HandleGemm0())
        {
            ck_tile::store_tile(p_st_lds_window, s_acc);
        }
        ck_tile::block_sync_lds();
        return ck_tile::load_tile(p_ld_lds_window);
    }
    else
    {
        return s_acc;
    }
}

template <bool IsMasking, typename acc_t>
CK_TILE_DEVICE static auto GetValidatedMax(acc_t raw_m)
{
    if constexpr (IsMasking)
    {
        return (raw_m == -ck_tile::numeric<acc_t>::infinity()) ? ck_tile::type_convert<acc_t>(0.f) : raw_m;
    }
    else
    {
        return raw_m;
    }
}

// =====================================================================================================================
// Kernel Pipelines
//

template<typename Traits,
         typename scalar_t,
         typename acc_t,
         typename out_t,
         typename LseDramBlockWindow,
         typename OutDramBlockWindow,
         typename OaccBlockTileType,
         typename MLBlockTileType,
         typename Mask>
CK_TILE_DEVICE static inline void kn_fmla_fwd_splitkv_prefill_tile_epilogue(
    LseDramBlockWindow& lse_dram_window_,
    OutDramBlockWindow& out_dram_window_,
    OaccBlockTileType*  o_acc,
    MLBlockTileType&    m,
    MLBlockTileType&    l,
    Mask&               mask)
{
    constexpr int32_t n1_loops = Traits::kSizeDV / Traits::kBlockN1;

    // 7. Store LSE
    //
    auto lse_acc = ck_tile::make_static_distributed_tensor<acc_t>(m.get_tile_distribution());
    constexpr auto lse_acc_spans = decltype(lse_acc)::get_distributed_spans();
    ck_tile::sweep_tile_span(lse_acc_spans[ck_tile::number<0>{}], [&, m_ = m, l_ = l](auto id0) {
        constexpr auto i = make_tuple(id0);
#if FMLA_FWD_FAST_EXP2
#ifndef C_LOG2E
#define C_LOG2E 1.44269504088896340736 // log2(e)
#endif
        lse_acc(i) = m_[i] / C_LOG2E  + log(l_[i]);
#else
        lse_acc(i) = m_[i] + log(l_[i]);
#endif
    });
    ck_tile::store_tile(lse_dram_window_, lse_acc);

    // 8. Adjust and output
    //
    constexpr auto o_spans = ck_tile::remove_cvref_t<decltype(o_acc[0])>::get_distributed_spans();;
    ck_tile::sweep_tile_span(o_spans[ck_tile::number<0>{}], [&](auto id0) {
        constexpr auto i = ck_tile::make_tuple(id0);
        const auto tmp   = [&]() {
            if constexpr (Mask::IsMasking)
            {
                return l[i] == 0.f ? 0.f : 1 / l[i];
            }
            else
            {
                return 1 / l[i];
            }
        }();
        ck_tile::sweep_tile_span(o_spans[ck_tile::number<1>{}], [&](auto id1) {
            constexpr auto ij = ck_tile::make_tuple(id0, id1);
            ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id)
            {
#if 0
                acc_t o_acc_v = o_acc[n1_id](ij);
                asm volatile("v_mul_f32 %[v_o_acc], %[v_tmp], %[v_o_acc]\n"
                             : [v_o_acc] "+v"(o_acc_v)
                             : [v_tmp] "v"(tmp));
                o_acc[n1_id](ij) = o_acc_v;
#else
                o_acc[n1_id](ij) *= tmp;
#endif
            });
        });
    });

    ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id)
    {
        ck_tile::store_tile(out_dram_window_, ck_tile::cast_tile<out_t>(o_acc[n1_id]));
        if constexpr (n1_id < (n1_loops - 1))
        {
            ck_tile::move_tile_window(out_dram_window_, {0, Traits::kBlockN1});
        }
    });
}

template<typename Traits,
         typename scalar_t,
         typename acc_t,
         typename out_t,
         typename QDramBlockWindow,
         typename KDramBlockWindow,
         typename VDramBlockWindow,
         typename LseDramBlockWindow,
         typename OutDramBlockWindow,
         typename Mask>
CK_TILE_DEVICE static void kn_fmla_fwd_splitkv_prefill_tile(
    const QDramBlockWindow& q_dram_window_,
    const KDramBlockWindow& k_dram_window_raw,
    const VDramBlockWindow& v_dram_window_raw,
    LseDramBlockWindow&     lse_dram_window_,
    OutDramBlockWindow&     out_dram_window_,
    const int32_t*          p_block_table,
    const int32_t           page_block_size,
    const int32_t           stride_s_k,
    const int32_t           stride_s_v,
    int32_t                 seqlen_k,
    int32_t                 num_splits,
    int32_t                 split_id,
    Mask                    mask,
    float                   scale_s,
    uint8_t*                p_smem)
{
    using Policy = FlashMlaPrefillPolicy<Traits, scalar_t, acc_t>;


    // 1. Allocate LDS
    //
    auto k_lds = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
        reinterpret_cast<scalar_t*>(p_smem), Policy::MakeKLdsBlockDescriptor());
    auto v_lds = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
        reinterpret_cast<scalar_t*>(p_smem), Policy::MakeVLdsBlockDescriptor());

    auto k_lds_window = ck_tile::make_tile_window(
        k_lds, ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, ck_tile::number<Traits::kBlockK0>{}), {0, 0});
    auto v_lds_window = ck_tile::make_tile_window(
        v_lds, Policy::MakeVLdsBlockDescriptor().get_lengths(), {0, 0});


    // 2. Misc. preparation
    //

    // Loop counts
    constexpr int32_t k0_loops = Traits::kSizeD / Traits::kBlockK0;      // #loop for Q in reg
    constexpr int32_t k1_loops = Traits::kBlockN0 / Traits::kBlockK1;
    constexpr int32_t n1_loops = Traits::kSizeDV / Traits::kBlockN1;
    static_assert(k0_loops >= 2);
    static_assert(k1_loops >= 1);
    static_assert(n1_loops >= 1);
    static_assert((Traits::kSizeD   % Traits::kBlockK0) == 0);
    static_assert((Traits::kBlockN0 % Traits::kBlockK1) == 0);
    static_assert((Traits::kSizeDV  % Traits::kBlockN1) == 0);
    // TODO: make it tunable once slice tile issue is fixed.
    static_assert(!((k1_loops > 1) && (Traits::kNumWarps == 8)),
                  "k1_loops > 1 is not support by wave 8 for now due to the issue in tile_slice.");

    // Block GEMMs
    constexpr auto gemm_0 = Policy::GetQKBlockGemm();
    constexpr auto gemm_1 = Policy::GetKVBlockGemm();

    // Reduction funtions for softmax
    using BlockShape           = ck_tile::remove_cvref_t<decltype(gemm_0.GetCBlockShape())>;
    using BlockReduce2dProblem = ck_tile::BlockReduce2dProblem<acc_t, acc_t, BlockShape>;
    auto block_reduce_2d       = ck_tile::BlockReduce2d<BlockReduce2dProblem>();        // In-thread
    auto block_reduce_2d_sync  = ck_tile::BlockReduce2dSync<BlockReduce2dProblem>();    // In-warp
    auto reduce_sum_func = ck_tile::ReduceOp::Add{};
    auto reduce_max_func = ck_tile::ReduceOp::Max{};

    // sacc, S, P, M, L, Oacc
    using SaccBlockTileType    = decltype(gemm_0.MakeCBlockTile());
    auto s_acc                 = SaccBlockTileType{};
    using SaccShuffledTileType = decltype(ShuffleSacc<Policy>(s_acc, nullptr));
    auto s_acc_shuffled        = SaccShuffledTileType{};
    using MLBlockTileType      = decltype(block_reduce_2d(s_acc_shuffled,
                                                          ck_tile::numeric<acc_t>::min(), 
                                                          reduce_max_func));
    using OaccBlockTileType    = decltype(gemm_1.MakeCBlockTile());
    OaccBlockTileType o_acc[n1_loops];
    auto m = MLBlockTileType{};
    auto l = MLBlockTileType{};
    ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id){ck_tile::clear_tile(o_acc[n1_id]);});
    ck_tile::set_tile(m, -ck_tile::numeric<acc_t>::infinity());
    ck_tile::clear_tile(l);

    const auto q_origin = q_dram_window_.get_window_origin();
    auto [origin_start, origin_end] =
        mask.GetTileRangeAlongX(q_origin.at(ck_tile::number<0>{}),
                                ck_tile::number<Traits::kBlockM>{},
                                ck_tile::number<Traits::kBlockN0>{});
    auto [seqlen_k_start, seqlen_k_end] =
        GetSeqlenRange(
            seqlen_k,
            Traits::kBlockN0,
            num_splits,
            split_id,
            origin_start,
            ck_tile::min(origin_end, seqlen_k));


    // 3. Quick exit if no work to do
    //
    const int32_t num_total_loop =
        ck_tile::integer_divide_ceil(seqlen_k_end - seqlen_k_start, Traits::kBlockN0);

    if (num_total_loop <= 0)
    {
        auto lse_acc = ck_tile::make_static_distributed_tensor<acc_t>(m.get_tile_distribution());
        ck_tile::set_tile(lse_acc, -ck_tile::numeric<acc_t>::infinity());
        ck_tile::store_tile(lse_dram_window_, lse_acc);
        ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id)
        {
            ck_tile::store_tile(out_dram_window_, ck_tile::cast_tile<out_t>(o_acc[n1_id]));
            if constexpr (n1_id < (n1_loops - 1))
            {
                ck_tile::move_tile_window(out_dram_window_, {0, Traits::kBlockN1});
            }
        });
        return;
    }


    // 4. Load Q to lds and reg
    //
    auto q_dram_window =
        ck_tile::make_tile_window(q_dram_window_.get_bottom_tensor_view(),
                                  q_dram_window_.get_window_lengths(),
                                  q_dram_window_.get_window_origin(),
                                  Policy::MakeQRegTileDistribution());
    using QTile = decltype(ck_tile::load_tile(q_dram_window));
    QTile q_regs[2];


    // 5. Prepare KV
    //
    auto k_dram_window_origin = ck_tile::make_tile_window(
        k_dram_window_raw.get_bottom_tensor_view(),
        k_dram_window_raw.get_window_lengths(),
        {seqlen_k_start, 0});
    auto k_dist = Policy::MakeKDramTileDistribution();
    auto k_coord = k_dist.calculate_index();
    constexpr auto kKNumRepeat = Policy::GetNumRepeatOfKDramTileDistribution();
    constexpr auto kKPageIdxDim = ck_tile::number<0>{};
    const int32_t seqlen_k_base_idx = k_coord[kKPageIdxDim] + seqlen_k_start;
    ck_tile::statically_indexed_array<int32_t, kKNumRepeat> k_offsets;
    ck_tile::statically_indexed_array<bool, kKNumRepeat> k_valids;
    ck_tile::static_for<0, kKNumRepeat, 1>{}([&](auto rid)
    {
        const int32_t seqlen_idx = seqlen_k_base_idx + Traits::kBlockN0 / kKNumRepeat * rid.value;
        const int32_t page_idx   = seqlen_idx / page_block_size;
        const int32_t inside_idx = seqlen_idx % page_block_size;
        k_offsets[rid] = (p_block_table[page_idx] * page_block_size + inside_idx) * stride_s_k;
        k_valids[rid] = (seqlen_idx < seqlen_k_end);
    });
    auto k_dram_window = ck_tile::make_tile_scatter_gather(
        k_dram_window_origin.get_bottom_tensor_view(),
        k_dram_window_origin.get_window_lengths(),
        k_dram_window_origin.get_window_origin(),
        k_dist,
        k_offsets,
        k_valids,
        kKPageIdxDim);
    k_dram_window.init_raw();

    auto v_dram_window_origin = ck_tile::make_tile_window(
        v_dram_window_raw.get_bottom_tensor_view(),
        v_dram_window_raw.get_window_lengths(),
        {0, seqlen_k_start});
    auto v_dist = Policy::MakeVDramTileDistribution();
    auto v_coord = v_dist.calculate_index();
    constexpr auto kVNumRepeat = Policy::GetNumRepeatOfVDramTileDistribution();
    constexpr auto kVPageIdxDim = ck_tile::number<1>{};
    const int32_t seqlen_v_base_idx = v_coord[kVPageIdxDim] + seqlen_k_start;
    ck_tile::statically_indexed_array<int32_t, kVNumRepeat> v_offsets;
    ck_tile::statically_indexed_array<bool, kVNumRepeat> v_valids;
    ck_tile::static_for<0, kVNumRepeat, 1>{}([&](auto rid)
    {
        const int32_t seqlen_idx = seqlen_v_base_idx + rid.value;
        const int32_t page_idx   = seqlen_idx / page_block_size;
        const int32_t inside_idx = seqlen_idx % page_block_size;
        v_offsets[rid] = (p_block_table[page_idx] * page_block_size + inside_idx) * stride_s_v;
        v_valids[rid] = (seqlen_idx < seqlen_k_end);
    });
    using VDramWindow = decltype(ck_tile::make_tile_scatter_gather(
        v_dram_window_origin.get_bottom_tensor_view(),
        v_dram_window_origin.get_window_lengths(),
        v_dram_window_origin.get_window_origin(),
        v_dist,
        v_offsets,
        v_valids,
        kVPageIdxDim));
    VDramWindow v_dram_windows[n1_loops];
    ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id)
    {
        constexpr ck_tile::array<int32_t, 2> origin_col = {n1_id * Traits::kBlockN1, 0};
        v_dram_windows[n1_id] = ck_tile::make_tile_scatter_gather(
            v_dram_window_origin.get_bottom_tensor_view(),
            v_dram_window_origin.get_window_lengths(),
            v_dram_window_origin.get_window_origin() + origin_col,
            v_dist,
            v_offsets,
            v_valids,
            kVPageIdxDim);
        v_dram_windows[n1_id].init_raw();
    });


    // 6. Main loop
    //
    for(int32_t loop_idx = 0; loop_idx < num_total_loop; loop_idx += 1)
    {
        const bool is_even_loop = (loop_idx % 2 == 0);

        if (Policy::HandleGemm0())
        {
            ck_tile::clear_tile(s_acc);

            // I. QK GEMM
            //
            ck_tile::array<int32_t, 2> qk_origin =
                {0, is_even_loop ? 0 : Traits::kBlockK0 * (k0_loops - 1)};
            ck_tile::array<int32_t, 2> qk_direction =
                {0, is_even_loop ? Traits::kBlockK0 : -Traits::kBlockK0};
            auto q_dram_window = ck_tile::make_tile_window(
                q_dram_window_.get_bottom_tensor_view(),
                q_dram_window_.get_window_lengths(),
                q_dram_window_.get_window_origin() + qk_origin,
                Policy::MakeQRegTileDistribution());
            q_regs[0] = ck_tile::load_tile(q_dram_window);
            ck_tile::move_tile_window(q_dram_window, qk_direction);
            q_regs[1] = ck_tile::load_tile(q_dram_window);

            // Load 1st K tile from DRAM to SMEM and start loading the 2nd
            // k_dram_window moves along K0 and only moves within page block.
            auto k_block_tile = ck_tile::load_tile(k_dram_window);

            ck_tile::move_tile_window(k_dram_window, qk_direction);
            ck_tile::store_tile(k_lds_window, k_block_tile);
            k_block_tile = ck_tile::load_tile(k_dram_window);

            // Main part of QK GEMM_0: conduct GEMM and load K tiles 
            if constexpr (k0_loops > 2)
            {
                ck_tile::static_for<0, k0_loops - 2, 1>{}([&](auto k0_id)
                {
                    ck_tile::block_sync_lds();
                    gemm_0(s_acc, q_regs[k0_id % 2], k_lds_window);
                    ck_tile::block_sync_lds();
                    // pre-load q
                    ck_tile::move_tile_window(q_dram_window, qk_direction);
                    q_regs[k0_id % 2] = ck_tile::load_tile(q_dram_window);
                    // pre-load k
                    ck_tile::move_tile_window(k_dram_window, qk_direction);
                    ck_tile::store_tile(k_lds_window, k_block_tile);
                    k_block_tile = ck_tile::load_tile(k_dram_window);
                });
            }

            // Tailing 2 tiles of QK GEMM_0
            ck_tile::block_sync_lds();
            gemm_0(s_acc, q_regs[(k0_loops - 2) % 2], k_lds_window);

            ck_tile::block_sync_lds();
            ck_tile::store_tile(k_lds_window, k_block_tile);

            ck_tile::block_sync_lds();
            gemm_0(s_acc, q_regs[(k0_loops - 1) % 2], k_lds_window);

            ck_tile::tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
        }
        else
        {
            if constexpr (k0_loops > 2)
            {
                ck_tile::static_for<0, k0_loops - 2, 1>{}([&](auto k0_id)
                {
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_s_barrier();
                });
            }
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_barrier();
        }

        // prefetch load V tile
        auto v_prefetch = ck_tile::load_tile(v_dram_windows[0]);
        
        if (Policy::HandleGemm0())
        {
            // II. scale_s, mask, softmax
            //

            // Masking
            // Note that masking is also required when k is padded
            const auto k_origin = k_dram_window_origin.get_window_origin();
            const bool need_perpixel_check = mask.IsEdgeTile(
                __builtin_amdgcn_readfirstlane(q_origin.at(ck_tile::number<0>{})),
                __builtin_amdgcn_readfirstlane(k_origin.at(ck_tile::number<0>{})),
                ck_tile::number<Traits::kBlockM>{},
                ck_tile::number<Traits::kBlockN0>{});

            if (need_perpixel_check)
            {
                ck_tile::set_tile_if(
                    s_acc, -ck_tile::numeric<acc_t>::infinity(),
                    [&](auto ids)
                    {
                        const auto row = q_origin.at(ck_tile::number<0>{}) + ids.at(ck_tile::number<0>{});
                        const auto col = k_origin.at(ck_tile::number<0>{}) + ids.at(ck_tile::number<1>{});
                        return mask.IsOutOfBound(row, col);
                    });
            }
        }

        s_acc_shuffled = ShuffleSacc<Policy>(
            s_acc, p_smem + Policy::GetPSmemStart());

        // Get max of row
        auto m_local = block_reduce_2d(s_acc_shuffled, reduce_max_func.GetIdentityValue<acc_t>(), reduce_max_func);
        block_reduce_2d_sync(m_local, reduce_max_func);

        const auto m_old = m;
        ck_tile::tile_elementwise_inout(
            [](auto& e0, auto e1, auto e2) { e0 = ck_tile::max(e1, e2); }, m, m_old, m_local);

        // Compute exp(x_i - m)
        auto p_intermedia = ck_tile::make_static_distributed_tensor<acc_t>(s_acc_shuffled.get_tile_distribution());
        const auto p_spans = decltype(p_intermedia)::get_distributed_spans();
        ck_tile::sweep_tile_span(
            p_spans[ck_tile::number<0>{}],
            [&](auto id0)
            {
                constexpr auto i = ck_tile::make_tuple(id0);
                auto row_max  = GetValidatedMax<Mask::IsMasking>(m[i]);
                ck_tile::sweep_tile_span(
                    p_spans[ck_tile::number<1>{}],
                    [&](auto id1)
                    {
                        constexpr auto ij = ck_tile::make_tuple(id0, id1);
#if FMLA_FWD_FAST_EXP2
                        p_intermedia(ij) = ck_tile::exp2(s_acc_shuffled[ij] - row_max);
#else
                        p_intermedia(ij) = ck_tile::exp(s_acc_shuffled[ij] - row_max);
#endif
                    });
            });

        // Compute row sum of exp(x_i - m)
        auto rowsum_p = block_reduce_2d(p_intermedia, reduce_sum_func.GetIdentityValue<acc_t>(), reduce_sum_func);
        block_reduce_2d_sync(rowsum_p, reduce_sum_func);

        // Calculate new l and adjust old output acc
        constexpr auto o_spans = ck_tile::remove_cvref_t<decltype(o_acc[0])>::get_distributed_spans();
        ck_tile::sweep_tile_span(
            o_spans[ck_tile::number<0>{}],
            [&](auto id0)
            {
                constexpr auto i = ck_tile::make_tuple(id0);
                const auto row_max = GetValidatedMax<Mask::IsMasking>(m[i]);
#if FMLA_FWD_FAST_EXP2
                const auto temp_i  = ck_tile::exp2(m_old[i] - row_max);
#else
                const auto temp_i  = ck_tile::exp(m_old[i] - row_max);
#endif
                l(i) = temp_i * l[i] + rowsum_p[i];
                ck_tile::sweep_tile_span(
                    o_spans[ck_tile::number<1>{}],
                    [&](auto id1)
                    {
                        constexpr auto ij = ck_tile::make_tuple(id0, id1);
                        ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id)
                        {
#if 0
                            acc_t o_acc_v = o_acc[n1_id](ij);
                            asm volatile("v_mul_f32 %[v_o_acc], %[v_tmp], %[v_o_acc]\n"
                                        : [v_o_acc] "+v"(o_acc_v)
                                        : [v_tmp] "v"(temp_i));
                            o_acc[n1_id](ij) = o_acc_v;
#else
                            o_acc[n1_id](ij) *= temp_i;
#endif
                        });
                    });
            });


        // III. GEMM for PV
        //

        // Store V tile to LDS. V is expected as row-major so it needs to be shuffled before store.
        ck_tile::block_sync_lds();
        const auto p = ck_tile::cast_tile<scalar_t>(p_intermedia);
        ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id)
        {
            auto v_shuffled = ck_tile::make_static_distributed_tensor<scalar_t>(Policy::MakeShuffledVRegBlockDescriptor());
            ck_tile::shuffle_tile(v_shuffled, v_prefetch);
            ck_tile::store_tile(v_lds_window, v_shuffled);
            __builtin_amdgcn_sched_barrier(0); // This barrier decrease the useage of vgprs, but
                                               // negatively affects performance for small seqlen.
            if constexpr(k1_loops > 1)
            {
                ck_tile::static_for<0, k1_loops - 1, 1>{}([&](auto k1_id)
                {
                    // ck_tile::move_tile_window(v_dram_windows[n1_id], {0, Traits::kBlockK1});
                    ck_tile::static_for<0, kVNumRepeat, 1>{}([&](auto rid)
                    {
                        const int32_t seqlen_idx = seqlen_v_base_idx + rid.value +
                            loop_idx * Traits::kBlockN0 + (k1_id + 1) * Traits::kBlockK1;
                        const int32_t page_idx   = seqlen_idx / page_block_size;
                        const int32_t inside_idx = seqlen_idx % page_block_size;
                        v_offsets[rid] = (p_block_table[page_idx] * page_block_size + inside_idx) * stride_s_v;
                        v_valids[rid] = (seqlen_idx < seqlen_k_end);
                    });
                    v_dram_windows[n1_id].update_page_idx_and_valids(v_offsets, v_valids);

                    auto v = ck_tile::load_tile(v_dram_windows[n1_id]); // load next v
                    ck_tile::block_sync_lds();

                    gemm_1(o_acc[n1_id],
                           ck_tile::get_slice_tile(
                               p,
                               ck_tile::sequence<0, k1_id * Traits::kBlockK1>{},
                               ck_tile::sequence<Traits::kBlockM, (k1_id + 1) * Traits::kBlockK1>{}),
                           v_lds_window);
                    ck_tile::block_sync_lds();

                    auto v_shuffled = ck_tile::make_static_distributed_tensor<scalar_t>(
                        Policy::MakeShuffledVRegBlockDescriptor());
                        ck_tile::shuffle_tile(v_shuffled, v);
                        ck_tile::store_tile(v_lds_window, v_shuffled); // store the prefetch
                });
            }
            // Output tail
            ck_tile::block_sync_lds();

            if constexpr (n1_id < (n1_loops-1))
            {
                v_prefetch = ck_tile::load_tile(v_dram_windows[n1_id + 1]);
                // TODO: The following code is not necessary but it positively affects performance
                // for unknown reason. Remove the following code once it no longer affects.
                ck_tile::set_tile_if(
                    v_prefetch, ck_tile::numeric<scalar_t>::zero(),
                    [&](auto ids)
                    {
                        const auto col = ids.at(kVPageIdxDim);
                        return (loop_idx * Traits::kBlockN0 + col) >= seqlen_k;
                    });
            }
            if constexpr (k1_loops > 1)
            {
                gemm_1(o_acc[n1_id],
                   ck_tile::get_slice_tile(
                        p,
                        ck_tile::sequence<0, (k1_loops - 1) * Traits::kBlockK1>{},
                        ck_tile::sequence<Traits::kBlockM, Traits::kBlockN0>{}),
                   v_lds_window);
            }
            else
            {
                gemm_1(o_acc[n1_id], p, v_lds_window);
            }
            ck_tile::block_sync_lds();
        });

        if ((loop_idx + 1) < num_total_loop)
        {
            // Move K to next block of column
            ck_tile::array<int32_t, 2> next_qk_origin =
                {0, is_even_loop ? Traits::kBlockK0 * (k0_loops - 1) : 0};
            ck_tile::move_tile_window(k_dram_window_origin, {Traits::kBlockN0, 0});
            k_dram_window.set_window_origin(k_dram_window_origin.get_window_origin() + next_qk_origin);
            // Recalculate offsets
            ck_tile::static_for<0, kKNumRepeat, 1>{}([&](auto rid)
            {
                const int32_t seqlen_idx = seqlen_k_base_idx + (loop_idx + 1) * Traits::kBlockN0 +
                                           Traits::kBlockN0 / kKNumRepeat * rid.value;
                const int32_t page_idx   = seqlen_idx / page_block_size;
                const int32_t inside_idx = seqlen_idx % page_block_size;
                k_offsets[rid] = (p_block_table[page_idx] * page_block_size + inside_idx) * stride_s_k;
                k_valids[rid] = (seqlen_idx < seqlen_k_end);
            });
            k_dram_window.update_page_idx_and_valids(k_offsets, k_valids);
            // Move to next V block
            ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id)
            {
                // ck_tile::move_tile_window(v_dram_windows[n1_id], {0, -Traits::kBlockK1 * (k1_loops - 1)});
                ck_tile::static_for<0, kVNumRepeat, 1>{}([&](auto rid)
                {
                    const int32_t seqlen_idx = seqlen_v_base_idx + rid.value + (loop_idx + 1) * Traits::kBlockN0;
                    const int32_t page_idx   = seqlen_idx / page_block_size;
                    const int32_t inside_idx = seqlen_idx % page_block_size;
                    v_offsets[rid] = (p_block_table[page_idx] * page_block_size + inside_idx) * stride_s_v;
                    v_valids[rid] = (seqlen_idx < seqlen_k_end);
                });
                v_dram_windows[n1_id].update_page_idx_and_valids(v_offsets, v_valids);
            });
        }
    }

    // 7. tile epilogue: store LSE, Adjust and output
    kn_fmla_fwd_splitkv_prefill_tile_epilogue<Traits, scalar_t, acc_t, out_t>(
        lse_dram_window_,
        out_dram_window_,
        o_acc,
        m,
        l,
        mask);
}

// this function work for mla which load key&value tensor once, 
// transpose the k-nope tensor into v tensor, and never load v tensor from dram again.
// TODO: 1. async load from dram to lds. use double lds buffer to load two kv blocks.
//       2. transpose v value while gemm_0.
template<typename Traits,
         typename scalar_t,
         typename acc_t,
         typename out_t,
         typename QNopeDramRegBlockWindow,
         typename QRopeDramRegBlockWindow,
         typename KVDramTensorView,
         typename LseDramBlockWindow,
         typename OutDramBlockWindow,
         typename Mask>
CK_TILE_DEVICE static void kn_fmla_fwd_splitkv_prefill_load_once_tile(
    const QNopeDramRegBlockWindow&  q_nope_dram_window_,
    const QRopeDramRegBlockWindow&  q_rope_dram_window_,
    const KVDramTensorView&         k_dram_tensor_view_,
    LseDramBlockWindow&             lse_dram_window_,
    OutDramBlockWindow&             out_dram_window_,
    const int32_t*                  p_block_table,
    const int32_t                   page_block_size,
    const int32_t                   stride_s_k,
    const int32_t                   stride_s_v,
    int32_t                         seqlen_k,
    int32_t                         num_splits,
    int32_t                         split_id,
    Mask                            mask,
    float                           scale_s,
    uint8_t*                        p_smem
    )
{
    using Policy = FlashMlaPrefillPolicy<Traits, scalar_t, acc_t>;

    // 1. Allocate LDS
    //
    const auto k_rope_smem_offset = (Traits::kNumPrefetchK * Policy::template GetSingleKSpaceSize<false>());

    const auto k_nope_repeat_smem_offset = Policy::GetKNopeSingleRepeatSize();
    constexpr auto k_oob_ck = ck_tile::bool_constant<true>{};
    constexpr auto k_pre_np = ck_tile::bool_constant<false>{};

    auto k_nope_lds_ptr = reinterpret_cast<scalar_t*>(p_smem);
    auto k_nope_st_lds_windows = ck_tile::generate_tuple(
        [&](auto i_buf) {
            return ck_tile::make_tile_window(
                ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                    k_nope_lds_ptr, Policy::template MakeKLdsStoreBlockDescriptor<false>(i_buf)),
                Policy::template MakeKLdsStoreBlockDescriptor<false>(i_buf).get_lengths(),
                // ck_tile::make_tuple(ck_tile::number<4>{}, ck_tile::number<4>{}, ck_tile::number<128>{}),
                {0, 0, 0});
        },
        ck_tile::number<Traits::kNumPrefetchK>{});
    auto k_nope_ld_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
        k_nope_lds_ptr, Policy::template MakeKLdsLoadBlockDescriptor<false>());
    auto k_nope_ld_lds_window = ck_tile::make_tile_window(
        k_nope_ld_lds_view,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, ck_tile::number<Traits::kSizeNope>{}),
        {0, 0});

    auto k_rope_lds_ptr = reinterpret_cast<scalar_t*>(p_smem + k_rope_smem_offset);
    auto k_rope_st_lds_windows = ck_tile::generate_tuple(
        [&](auto i_buf) {
            return ck_tile::make_tile_window(
                ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                    k_rope_lds_ptr, Policy::template MakeKLdsStoreBlockDescriptor<true>(i_buf)),
                Policy::template MakeKLdsStoreBlockDescriptor<true>(i_buf).get_lengths(),
                // ck_tile::make_tuple(ck_tile::number<2>{}, ck_tile::number<4>{}, ck_tile::number<128>{}),
                {0, 0, 0});
        },
        ck_tile::number<Traits::kNumPrefetchK>{});
    auto k_rope_ld_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
        k_rope_lds_ptr, Policy::template MakeKLdsLoadBlockDescriptor<true>());
    auto k_rope_ld_lds_window = ck_tile::make_tile_window(
        k_rope_ld_lds_view,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, ck_tile::number<Traits::kSizeRope>{}),
        {0, 0});

    auto vt_lds_ptr = reinterpret_cast<scalar_t*>(p_smem + Traits::kNumPrefetchK *
        Policy::GetSmemSizeK());
    auto v_ld_lds_window = ck_tile::make_tile_window(
        k_nope_ld_lds_view,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, ck_tile::number<Traits::kSizeNope>{}),
        {0, 0},
        Policy::MakeVTileDistribution());
    auto vt_st_lds_window = ck_tile::make_tile_window(
        ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            vt_lds_ptr,
            Policy::MakeVLdsStoreBlockDescriptor()),
        Policy::MakeVLdsStoreBlockDescriptor().get_lengths(), {0, 0});
    auto vt_ld_lds_window = ck_tile::make_tile_window(
        ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            vt_lds_ptr,
            Policy::MakeVLdsStoreBlockDescriptor()),
        Policy::MakeVLdsStoreBlockDescriptor().get_lengths(), {0, 0});


    // 2. Misc. preparation
    //

    // Loop counts
    constexpr int32_t k0_loops = Traits::kSizeD / Traits::kBlockK0;      // #loop for Q in reg
    constexpr int32_t k1_loops = Traits::kBlockN0 / Traits::kBlockK1;
    constexpr int32_t n1_loops = Traits::kSizeDV / Traits::kBlockN1;
    static_assert(k0_loops >= 1);
    static_assert(k1_loops >= 1);
    static_assert(n1_loops >= 1);
    // static_assert((Traits::kSizeD   % Traits::kBlockK0) == 0);
    static_assert((Traits::kBlockN0 % Traits::kBlockK1) == 0);
    static_assert((Traits::kSizeDV  % Traits::kBlockN1) == 0);

    // Block GEMMs
    constexpr auto gemm_0      = Policy::template GetQKBlockGemm<false>();
    constexpr auto gemm_0_rope = Policy::template GetQKBlockGemm<true>();
    constexpr auto gemm_1      = Policy::GetKVBlockGemm();

    // Reduction funtions for softmax
    const auto f_max = [](auto e0, auto e1) { return ck_tile::max(e0, e1); };
    const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

    // Reduction funtions for softmax
    using BlockShape           = ck_tile::remove_cvref_t<decltype(gemm_0.GetCBlockShape())>;
    using BlockReduce2dProblem = ck_tile::BlockReduce2dProblem<acc_t, acc_t, BlockShape>;
    auto block_reduce_2d       = ck_tile::BlockReduce2d<BlockReduce2dProblem>();        // In-thread
    auto block_reduce_2d_sync  = ck_tile::BlockReduce2dSync<BlockReduce2dProblem>();    // In-warp
    auto reduce_sum_func = ck_tile::ReduceOp::Add{};
    auto reduce_max_func = ck_tile::ReduceOp::Max{};

    // sacc, S, P, M, L, Oacc
    using SaccBlockTileType    = decltype(gemm_0.MakeCBlockTile());
    auto s_acc                 = SaccBlockTileType{};
    using SaccShuffledTileType = decltype(ShuffleSacc<Policy>(s_acc, nullptr));
    auto s_acc_shuffled        = SaccShuffledTileType{};
    using MLBlockTileType      = decltype(block_reduce_2d(s_acc_shuffled,
                                                          ck_tile::numeric<acc_t>::min(), 
                                                          reduce_max_func));
    using OaccBlockTileType    = decltype(gemm_1.MakeCBlockTile());
    OaccBlockTileType o_acc[n1_loops];
    auto m = MLBlockTileType{};
    auto l = MLBlockTileType{};
    ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id){ck_tile::clear_tile(o_acc[n1_id]);});
    ck_tile::set_tile(m, -ck_tile::numeric<acc_t>::infinity());
    ck_tile::clear_tile(l);


    const auto q_origin = q_nope_dram_window_.get_window_origin();
    auto [origin_start, origin_end] =
        mask.GetTileRangeAlongX(q_origin.at(ck_tile::number<0>{}),
                                ck_tile::number<Traits::kBlockM>{},
                                ck_tile::number<Traits::kBlockN0>{});
    auto [seqlen_k_start, seqlen_k_end] =
        GetSeqlenRange(seqlen_k, Traits::kBlockN0, num_splits, split_id, origin_start, origin_end);


    // 3. Quick exit if no work to do
    //
    const int32_t num_total_loop =
        ck_tile::integer_divide_ceil(seqlen_k_end - seqlen_k_start, Traits::kBlockN0);

    if (num_total_loop <= 0)
    {
        auto lse_acc = ck_tile::make_static_distributed_tensor<acc_t>(m.get_tile_distribution());
        ck_tile::set_tile(lse_acc, -ck_tile::numeric<acc_t>::infinity());
        ck_tile::store_tile(lse_dram_window_, lse_acc);
        ck_tile::static_for<0, n1_loops, 1>{}(
            [&](auto n1_id){
                ck_tile::store_tile(out_dram_window_, ck_tile::cast_tile<out_t>(o_acc[n1_id]));
                if constexpr (n1_id < (n1_loops - 1))
                {
                    ck_tile::move_tile_window(out_dram_window_, {0, Traits::kBlockN1});
                }
            }
        );
        return;
    }

    // 4. Load Q to lds and reg
    //
    auto q_dram_window_nope = ck_tile::make_tile_window(
        q_nope_dram_window_.get_bottom_tensor_view(),
        q_nope_dram_window_.get_window_lengths(),
        q_nope_dram_window_.get_window_origin(),
        Policy::template MakeQRegTileDistribution<false>());
    auto q_dram_window_rope = ck_tile::make_tile_window(
        q_rope_dram_window_.get_bottom_tensor_view(),
        q_rope_dram_window_.get_window_lengths(),
        q_rope_dram_window_.get_window_origin(),
        Policy::template MakeQRegTileDistribution<true>());

    auto q_reg_nope = ck_tile::load_tile(q_dram_window_nope);
    auto q_reg_rope = ck_tile::load_tile(q_dram_window_rope);

    // 5. Prepare KV
    //
    constexpr auto k_nope_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, ck_tile::number<Traits::kSizeNope>{});
    constexpr auto k_rope_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, ck_tile::number<Traits::kSizeRope>{});

    auto k_dist = Policy::template MakeKDramTileDistribution<false>();
    auto k_coord = k_dist.calculate_index();
    constexpr auto kKNumRepeat = Policy::template GetNumRepeatOfKDramTileDistribution<false>();
    constexpr auto kKPageIdxDim = ck_tile::number<0>{};
    const int32_t seqlen_k_base_idx = k_coord[kKPageIdxDim] + seqlen_k_start;
    ck_tile::statically_indexed_array<int32_t, kKNumRepeat> k_offsets;
    ck_tile::statically_indexed_array<bool, kKNumRepeat> k_valids;
    ck_tile::static_for<0, kKNumRepeat, 1>{}([&](auto rid)
    {
        const int32_t seqlen_idx = seqlen_k_base_idx + Traits::kBlockN0 / kKNumRepeat * rid.value;
        const int32_t page_idx   = seqlen_idx / page_block_size;
        const int32_t inside_idx = seqlen_idx % page_block_size;
        k_offsets[rid] = (p_block_table[page_idx] * page_block_size + inside_idx) * stride_s_k;
        k_valids[rid] = (seqlen_idx < seqlen_k_end);
    });
    auto k_dram_window = ck_tile::make_tile_scatter_gather(
        k_dram_tensor_view_,
        k_nope_dram_window_lengths,
        {seqlen_k_start, 0},
        k_dist,
        k_offsets,
        k_valids,
        kKPageIdxDim);
    k_dram_window.init_raw();

    ck_tile::static_for<0, Traits::kKNumRepeat, 1>{}([&](auto rid)
    {
        ck_tile::async_load_tile_raw(
            k_nope_st_lds_windows.at(ck_tile::number<0>{}), k_dram_window, ck_tile::number<-1>{}, k_oob_ck, k_pre_np, k_nope_repeat_smem_offset * rid);
        __builtin_amdgcn_sched_barrier(0);
        ck_tile::async_load_fence(k_dram_window.get_num_of_access());
        __builtin_amdgcn_s_barrier();
        ck_tile::move_tile_window(k_dram_window, {0, Traits::kMaxSplits});
    });
    ck_tile::move_tile_window(k_dram_window, {0, -Traits::kSizeNope});

    auto k_rope_dist = Policy::template MakeKDramTileDistribution<true>();
    auto k_rope_coord = k_rope_dist.calculate_index();
    constexpr auto kKRopeNumRepeat = Policy::template GetNumRepeatOfKDramTileDistribution<true>();
    constexpr auto kKRopePageIdxDim = ck_tile::number<0>{};
    const int32_t seqlen_k_rope_base_idx = k_rope_coord[kKRopePageIdxDim] + seqlen_k_start;
    ck_tile::statically_indexed_array<int32_t, kKRopeNumRepeat> k_rope_offsets;
    ck_tile::statically_indexed_array<bool, kKRopeNumRepeat> k_rope_valids;
    ck_tile::static_for<0, kKRopeNumRepeat, 1>{}([&](auto rid)
    {
        const int32_t seqlen_idx = seqlen_k_rope_base_idx + Traits::kBlockN0 / kKRopeNumRepeat * rid.value;
        const int32_t page_idx   = seqlen_idx / page_block_size;
        const int32_t inside_idx = seqlen_idx % page_block_size;
        k_rope_offsets[rid] = (p_block_table[page_idx] * page_block_size + inside_idx) * stride_s_k;
        k_rope_valids[rid] = (seqlen_idx < seqlen_k_end);
    });
    auto k_rope_dram_window = ck_tile::make_tile_scatter_gather(
        k_dram_tensor_view_,
        k_rope_dram_window_lengths,
        {seqlen_k_start, Traits::kSizeNope},
        k_rope_dist,
        k_rope_offsets,
        k_rope_valids,
        kKRopePageIdxDim);
    k_rope_dram_window.init_raw();

    ck_tile::async_load_tile_raw(
        k_rope_st_lds_windows(ck_tile::number<0>{}), k_rope_dram_window, ck_tile::number<-1>{}, k_oob_ck, k_pre_np, k_rope_smem_offset);
    __builtin_amdgcn_sched_barrier(0);
    ck_tile::async_load_fence(k_rope_dram_window.get_num_of_access());
    __builtin_amdgcn_s_barrier();

    static constexpr auto I0    = ck_tile::number<0>{};
    static constexpr auto I1    = ck_tile::number<1>{};
    static constexpr auto I3    = ck_tile::number<3>{};

    static constexpr uint32_t s_perm0 = 0x07060302;
    static constexpr uint32_t s_perm1 = 0x05040100;
    static constexpr uint32_t VKRepeats = Policy::GetVTileDistributionStride();
    // 6. Main loop
    //
    // Define main loop
    auto main_loop = [&]<bool IsEvenLoop>(int32_t loop_idx)
    {
        ck_tile::clear_tile(s_acc);
        ck_tile::block_sync_lds();

        using k_st_idx            = std::conditional_t<IsEvenLoop, ck_tile::number<1>, ck_tile::number<0>>;
        constexpr auto k_ld_begin = IsEvenLoop ? ck_tile::number<0>{} : ck_tile::number<Traits::kBlockN0>{};
        constexpr auto k_ld_end   = ck_tile::number<k_ld_begin + Traits::kBlockN0>{};
        constexpr ck_tile::array<int32_t, 2> v_lds_direction = { IsEvenLoop ? Traits::kBlockN0 : -Traits::kBlockN0, -Traits::kSizeNope };

        auto vt_tile = ck_tile::make_static_distributed_tensor<scalar_t>(Policy::MakeVTTileDistribution());
#pragma unroll 2
        for (int32_t vidx = 0; vidx < Traits::kKNumRepeat; ++vidx)
        {
            auto v_tile  = ck_tile::load_tile(v_ld_lds_window);
            ck_tile::move_tile_window(v_ld_lds_window, {0, Traits::kMaxSplits});
            auto vt_thread_buffer = vt_tile.get_thread_buffer().get();

            auto k_thread_buffer = v_tile.get_thread_buffer().get();
            ck_tile::static_for<0, VKRepeats, 1>{}([&](auto rid){
                constexpr auto dw_offset   = ck_tile::number<rid * 2>{};
                constexpr auto dw_offset_1 = ck_tile::number<rid * 2 + 1>{};
                constexpr auto vt_dw_offset = rid + VKRepeats;
                reinterpret_cast<uint32_t*>(vt_thread_buffer)[rid] = __builtin_amdgcn_perm(
                    reinterpret_cast<uint32_t*>(k_thread_buffer)[dw_offset_1],
                    reinterpret_cast<uint32_t*>(k_thread_buffer)[dw_offset],
                    s_perm1
                );
                reinterpret_cast<uint32_t*>(vt_thread_buffer)[vt_dw_offset] = __builtin_amdgcn_perm(
                    reinterpret_cast<uint32_t*>(k_thread_buffer)[dw_offset_1],
                    reinterpret_cast<uint32_t*>(k_thread_buffer)[dw_offset],
                    s_perm0
                );
            });
            // ck_tile::block_sync_lds();
            ck_tile::store_tile(vt_st_lds_window, vt_tile);
            ck_tile::move_tile_window(vt_st_lds_window, {Traits::kMaxSplits, 0});
            ck_tile::block_sync_lds();
        }
        ck_tile::move_tile_window(vt_st_lds_window, {-Traits::kSizeNope, 0});
        ck_tile::move_tile_window(v_ld_lds_window, v_lds_direction);

        // I. QK GEMM
        if (Policy::HandleGemm0())
        {
            ck_tile::clear_tile(s_acc);

            // I. QK GEMM
            //
            if constexpr (k0_loops > 1)
            {
                constexpr ck_tile::array<int32_t, 2> qk_direction = {0, Traits::kBlockK0};
                ck_tile::static_for<0, k0_loops, 1>{}(
                    [&](auto k0_id)
                    {
                        ck_tile::block_sync_lds();
                        gemm_0(s_acc,
                               ck_tile::get_slice_tile(q_reg_nope,
                                   ck_tile::sequence<0, k0_id * Traits::kBlockK0>{},
                                   ck_tile::sequence<Traits::kBlockM, (k0_id + 1) * Traits::kBlockK0>{}),
                               k_nope_ld_lds_window);

                        // pre-load k
                        ck_tile::move_tile_window(k_nope_ld_lds_window, qk_direction);
                    });
            }
            if constexpr (k0_loops == 1)
            {
                ck_tile::block_sync_lds();
                __builtin_amdgcn_s_waitcnt(3952);
                gemm_0(s_acc,
                       q_reg_nope,
                       ck_tile::get_slice_tile(k_nope_ld_lds_window,
                            ck_tile::sequence<k_ld_begin, 0>{},
                            ck_tile::sequence<k_ld_end, Traits::kSizeNope>{}));
                ck_tile::block_sync_lds();
            }

            // QK rope tail
            ck_tile::block_sync_lds();
            // __builtin_amdgcn_s_waitcnt(3952);
            gemm_0_rope(s_acc,
                        q_reg_rope,
                        ck_tile::get_slice_tile(k_rope_ld_lds_window,
                            ck_tile::sequence<k_ld_begin, 0>{},
                            ck_tile::sequence<k_ld_end, Traits::kSizeRope>{}));

            ck_tile::tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
        }
        else
        {
            if constexpr (k0_loops > 2)
            {
                ck_tile::static_for<0, k0_loops - 2, 1>{}([&](auto k0_id)
                {
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_s_barrier();
                });
            }
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_barrier();
        }

        // II. scale_s, mask, softmax
        //

        // Masking
        // Note that masking is also required when k is padded
        const auto k_origin = k_dram_window.get_window_origin();
        const bool need_perpixel_check = mask.IsEdgeTile(
            __builtin_amdgcn_readfirstlane(q_origin.at(ck_tile::number<0>{})),
            __builtin_amdgcn_readfirstlane(k_origin.at(ck_tile::number<0>{})),
            ck_tile::number<Traits::kBlockM>{},
            ck_tile::number<Traits::kBlockN0>{});

        if (need_perpixel_check)
        {
            ck_tile::set_tile_if(
                s_acc, -ck_tile::numeric<acc_t>::infinity(),
                [&](auto ids)
                {
                    const auto row = q_origin.at(ck_tile::number<0>{}) + ids.at(ck_tile::number<0>{});
                    const auto col = k_origin.at(ck_tile::number<0>{}) + ids.at(ck_tile::number<1>{});
                    return mask.IsOutOfBound(row, col);
                });
        }

        s_acc_shuffled = ShuffleSacc<Policy>(
            s_acc, p_smem + Policy::GetPSmemStart());

        // Get max of row
        auto m_local = block_reduce_2d(s_acc_shuffled, reduce_max_func.GetIdentityValue<acc_t>(), reduce_max_func);
        block_reduce_2d_sync(m_local, reduce_max_func);

        const auto m_old = m;
        ck_tile::tile_elementwise_inout(
            [](auto& e0, auto e1, auto e2) { e0 = ck_tile::max(e1, e2); }, m, m_old, m_local);

        // Compute exp(x_i - m)
        auto p_intermedia = ck_tile::make_static_distributed_tensor<acc_t>(s_acc_shuffled.get_tile_distribution());
        const auto p_spans = decltype(p_intermedia)::get_distributed_spans();
        ck_tile::sweep_tile_span(
            p_spans[ck_tile::number<0>{}],
            [&](auto id0)
            {
                constexpr auto i = ck_tile::make_tuple(id0);
                auto row_max  = GetValidatedMax<Mask::IsMasking>(m[i]);
                ck_tile::sweep_tile_span(
                    p_spans[ck_tile::number<1>{}],
                    [&](auto id1)
                    {
                        constexpr auto ij = ck_tile::make_tuple(id0, id1);
#if FMLA_FWD_FAST_EXP2
                        p_intermedia(ij) = ck_tile::exp2(s_acc_shuffled[ij] - row_max);
#else
                        p_intermedia(ij) = ck_tile::exp(s_acc_shuffled[ij] - row_max);
#endif
                    });
            });

        // Compute row sum of exp(x_i - m)
        auto rowsum_p = block_reduce_2d(p_intermedia, reduce_sum_func.GetIdentityValue<acc_t>(), reduce_sum_func);
        block_reduce_2d_sync(rowsum_p, reduce_sum_func);

        // Calculate new l and adjust old output acc
        constexpr auto o_spans = ck_tile::remove_cvref_t<decltype(o_acc[0])>::get_distributed_spans();
        ck_tile::sweep_tile_span(
            o_spans[ck_tile::number<0>{}],
            [&](auto id0)
            {
                constexpr auto i = ck_tile::make_tuple(id0);
                const auto row_max = GetValidatedMax<Mask::IsMasking>(m[i]);
#if FMLA_FWD_FAST_EXP2
                const auto temp_i  = ck_tile::exp2(m_old[i] - row_max);
#else
                const auto temp_i  = ck_tile::exp(m_old[i] - row_max);
#endif
                l(i) = temp_i * l[i] + rowsum_p[i];
                ck_tile::sweep_tile_span(
                    o_spans[ck_tile::number<1>{}],
                    [&](auto id1)
                    {
                        constexpr auto ij = ck_tile::make_tuple(id0, id1);
                        ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id)
                        {
#if 0
                            acc_t o_acc_v = o_acc[n1_id](ij);
                            asm volatile("v_mul_f32 %[v_o_acc], %[v_tmp], %[v_o_acc]\n"
                                        : [v_o_acc] "+v"(o_acc_v)
                                        : [v_tmp] "v"(temp_i));
                            o_acc[n1_id](ij) = o_acc_v;
#else
                            o_acc[n1_id](ij) *= temp_i;
#endif
                        });
                    });
            });


        // III. GEMM for PV and load next k block
        //
        if (loop_idx != num_total_loop - 1) {
            // Load next K
            ck_tile::move_tile_window(k_dram_window, {Traits::kBlockN0, 0});
            ck_tile::move_tile_window(k_rope_dram_window, {Traits::kBlockN0, 0});

            // Recalculate offsets
            ck_tile::static_for<0, kKNumRepeat, 1>{}([&](auto rid)
            {
                const int32_t seqlen_idx = seqlen_k_base_idx + (loop_idx + 1) * Traits::kBlockN0 +
                                           Traits::kBlockN0 / kKNumRepeat * rid.value;
                const int32_t page_idx   = seqlen_idx / page_block_size;
                const int32_t inside_idx = seqlen_idx % page_block_size;
                k_offsets[rid] = (p_block_table[page_idx] * page_block_size + inside_idx) * stride_s_k;
                k_valids[rid] = (seqlen_idx < seqlen_k_end);
            });
            k_dram_window.update_page_idx_and_valids(k_offsets, k_valids);

            ck_tile::static_for<0, kKRopeNumRepeat, 1>{}([&](auto rid)
            {
                const int32_t seqlen_idx = seqlen_k_rope_base_idx + (loop_idx + 1) * Traits::kBlockN0 +
                                           Traits::kBlockN0 / kKRopeNumRepeat * rid.value;
                const int32_t page_idx   = seqlen_idx / page_block_size;
                const int32_t inside_idx = seqlen_idx % page_block_size;
                k_rope_offsets[rid] = (p_block_table[page_idx] * page_block_size + inside_idx) * stride_s_k;
                k_rope_valids[rid] = (seqlen_idx < seqlen_k_end);
            });
            k_rope_dram_window.update_page_idx_and_valids(k_rope_offsets, k_rope_valids);

            ck_tile::static_for<0, 4, 1>{}([&](auto rid)
            {
                ck_tile::async_load_tile_raw(
                    k_nope_st_lds_windows.at(k_st_idx{}), k_dram_window, ck_tile::number<-1>{}, k_oob_ck, k_pre_np, k_nope_repeat_smem_offset * rid);
                __builtin_amdgcn_sched_barrier(0);
                ck_tile::async_load_fence(k_dram_window.get_num_of_access());
                __builtin_amdgcn_s_barrier();
                ck_tile::move_tile_window(k_dram_window, {0, Traits::kMaxSplits});
            });
            ck_tile::move_tile_window(k_dram_window, {0, -Traits::kSizeNope});

            ck_tile::async_load_tile_raw(
                k_rope_st_lds_windows.at(k_st_idx{}), k_rope_dram_window, ck_tile::number<-1>{}, k_oob_ck, k_pre_np, k_rope_smem_offset);
            __builtin_amdgcn_sched_barrier(0);
            ck_tile::async_load_fence(k_rope_dram_window.get_num_of_access());
            __builtin_amdgcn_s_barrier();
        }

        const auto p = ck_tile::cast_tile<scalar_t>(p_intermedia);
        if constexpr (k1_loops == 1)
        {

            ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id) {
                ck_tile::block_sync_lds();
                gemm_1(o_acc[n1_id],
                       p,
                       vt_ld_lds_window);
            });
        }
        else
        {
            ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id) {
                ck_tile::static_for<0, k1_loops, 1>{}([&](auto k1_id) {
                    ck_tile::block_sync_lds();
                    gemm_1(o_acc[n1_id],
                           ck_tile::get_slice_tile(
                               p,
                               ck_tile::sequence<0, k1_id * Traits::kBlockK1>{},
                               ck_tile::sequence<Traits::kBlockM, (k1_id + 1) * Traits::kBlockK1>{}),
                           vt_ld_lds_window);
                    ck_tile::block_sync_lds();
                    ck_tile::move_tile_window(vt_ld_lds_window, {0, Traits::kBlockK1});
                });
                vt_ld_lds_window.set_window_origin({(n1_id + 1) * Traits::kBlockN1, 0});
            });
        }
    };


    // Execute the loop
    main_loop.template operator()<true>(0);
    for (int32_t loop_idx = 1; (loop_idx + 1) < num_total_loop; (loop_idx += 2))
    {
        main_loop.template operator()<false>(loop_idx);
        main_loop.template operator()<true>(loop_idx + 1);
    }
    if ((num_total_loop % 2) == 0)
    {
        main_loop.template operator()<false>(num_total_loop - 1);
    }

    // 7. tile epilogue: store LSE, Adjust and output
    kn_fmla_fwd_splitkv_prefill_tile_epilogue<Traits, scalar_t, acc_t, out_t>(
        lse_dram_window_,
        out_dram_window_,
        o_acc,
        m,
        l,
        mask);
}

