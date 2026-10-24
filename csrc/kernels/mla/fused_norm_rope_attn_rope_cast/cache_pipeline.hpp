// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// ATOM CSR pipelines with per-head-count load scheduling.
// H64 loads after QK; H128 large grids load after PV. Smaller H128 grids
// retain early loading, which wins below the measured throughput crossover.
// Both retain double panels and ordered LDS waits. Moving loads earlier can
// extend register lifetimes enough to spill, despite more apparent overlap.
#pragma once
#include "cache_source.hpp"
namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1 {
template <class Traits, class Source>
__device__ void packed_prefill_accum64(
    const Source& source,
    int valid_kv_len,
    int num_kv_tiles,
    char* smem_kv_buf,
    opus::vector_t<typename Traits::D_ATTN,
                   Traits::Q_TILE_SIZE * Traits::D_TILE_SIZE / Traits::WARP_SIZE>& v_q,
    opus::vector_t<typename Traits::D_ACC, 8> (&v_o)[16],
    typename Traits::D_ACC& actual_max,
    typename Traits::D_ACC& m_row,
    typename Traits::D_ACC& l_row,
    typename Traits::D_ACC temperature_scale)
{
    using namespace opus;
    using T      = opus::remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    int lane_id = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id)); // break CSE
    const int warp_id =
        __builtin_amdgcn_readfirstlane((thread_id_x() / T::WARP_SIZE) % T::NUM_WARPS);

    // CSR bounds and int64 packed byte addresses are owned by Source.

    auto s_kv = make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf));

    auto mma0 =
        make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(seq<T::GEMM0_E_M, T::GEMM0_E_N, T::GEMM0_E_K>{},
                                              seq<T::T_M, T::T_N, T::T_K>{},
                                              seq<T::W_M, T::W_N, T::W_K>{},
                                              mfma_adaptor_swap_ab{});
    auto mma1 =
        make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
                                              seq<T::T_M, T::T_N, T::T_K>{},
                                              seq<T::W_M, T::W_N, T::W_K>{},
                                              mfma_adaptor_swap_ab{});

    auto u_gkv        = make_layout_gkv<T>(warp_id, lane_id);
    auto u_skv        = make_layout_skv<T>(warp_id);
    auto u_rk         = make_layout_rk<T>(lane_id);
    auto u_rv         = make_layout_rv<T>(lane_id);
    auto u_kv_indices = make_layout_kv_indices<T>(warp_id, lane_id);

    typename decltype(mma0)::vtype_b v_k[2];
    typename decltype(mma0)::vtype_c v_s;
    typename decltype(mma1)::vtype_a v_p;
    typename decltype(mma1)::vtype_b v_v[2];
    auto v_q_slices =
        reinterpret_cast<vector_t<D_ATTN, T::Q_TILE_SIZE * T::SLICE_D / T::WARP_SIZE>*>(&v_q);
    auto* v_o_slices = v_o;

    constexpr index_t s_len = vector_traits<typename decltype(mma0)::vtype_c>::size();

    auto skv_slice = [](auto slice_idx) {
        constexpr int s = decltype(slice_idx)::value;
        return number<(s / 2) * T::smem_n_rpt*(T::smem_linear_wave + T::smem_padding_32B) +
                      (s % 2) * T::SLICE_D>{};
    };

    int qk_tile      = 0;
    auto mask_scores = [&](auto& s) {
        static_for<2>([&](auto n) {
            int first = int(n) * 16 + (lane_id / 16) * 4;
            static_for<4>([&](auto j) {
                int k          = first + int(j);
                auto* flag_lds = reinterpret_cast<OPUS_LDS_ADDR const D_ATTN*>(
                    reinterpret_cast<uintptr_t>(s_kv.ptr));
                if(!reinterpret_cast<OPUS_LDS_ADDR const uint32_t*>(flag_lds + (k % 4) * 528 +
                                                                    512)[k / 4])
                    s[int(n) * 4 + int(j)] = -INFINITY;
            });
        });
        ++qk_tile;
    };
    auto compute_qk = [&](auto& s, const auto& q, auto& k) {
        clear(s);
        static_for<T::NUM_D_SLICES>([&](auto i) {
            constexpr int idx  = i.value;
            constexpr int slot = idx & 1;
            s                  = mma0(q[idx], k[slot], s);
            if constexpr(idx + 2 < T::NUM_D_SLICES)
            {
                k[slot] = load<T::VEC_KV>(s_kv, u_rk + skv_slice(number<idx + 2>{}));
                s_waitcnt_lgkmcnt(number<T::k_ds_read_insts>{});
            }
            else if constexpr(idx + 1 < T::NUM_D_SLICES)
            {
                s_waitcnt_lgkmcnt(0_I);
            }
        });
        mask_scores(s);
    };

    auto compute_pv = [&](const auto& p, auto& v, auto& o, int kv_tile_idx) {
        static_for<T::NUM_D_SLICES - 2>([&](auto i) {
            constexpr int idx  = i.value;
            constexpr int slot = idx & 1;
            o[idx]             = mma1(p, v[slot], o[idx]);
            v[slot]            = tr_load<T::VEC_TR_V>(s_kv, u_rv + skv_slice(number<idx + 2>{}));
            s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
            // Partial waits rely on LDS issue order and on which MFMA consumes
            // the completed reads. Allow VALU across this boundary, but keep
            // LDS, MFMA and SALU (including s_waitcnt) ordered. Allowing MFMA
            // or SALU here makes a later slice consume still-pending reads.
            // See experiments/20261014/PV_SCHEDULING.md for ISA and regressions.
            __builtin_amdgcn_sched_barrier(VALU_MASK);
        });
        o[T::NUM_D_SLICES - 2] = mma1(p, v[(T::NUM_D_SLICES - 2) & 1], o[T::NUM_D_SLICES - 2]);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_sched_barrier(0);
        o[T::NUM_D_SLICES - 1] = mma1(p, v[(T::NUM_D_SLICES - 1) & 1], o[T::NUM_D_SLICES - 1]);
    };

    const opus::u32_t neg_inf_v =
        std::bit_cast<opus::u32_t>(-opus::numeric_limits<D_ACC>::infinity());

    auto other_kv = make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf) + T::smem_kv_tile_elems);
    source.template load_tile<T>(0, s_kv.ptr, u_gkv, u_skv, warp_id, lane_id);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    for(int tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx)
    {
        v_k[0] = load<T::VEC_KV>(s_kv, u_rk);
        v_k[1] = load<T::VEC_KV>(s_kv, u_rk + skv_slice(1_I));
        s_waitcnt_lgkmcnt(number<T::k_ds_read_insts>{});
        compute_qk(v_s, v_q_slices, v_k);
        if(tile_idx + 1 < num_kv_tiles)
        {
            source.template load_tile<T>(
                tile_idx + 1, other_kv.ptr, u_gkv, u_skv, warp_id, lane_id);
        }

        D_ACC row_max = track_max<T>(v_s, temperature_scale, actual_max);
        bool below =
            (__builtin_amdgcn_ballot_w64(row_max - m_row <= 8.0f) == __builtin_amdgcn_read_exec());
        row_max         = below ? m_row : max(m_row, row_max);
        D_ACC rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
        m_row           = row_max;
        attn_row_scale_sub<T>(v_s, temperature_scale, row_max);
        attn_exp2_slice<T, 0, s_len>(v_s);
        l_row *= rescale_m;
        l_row += attn_row_sum<T>(v_s);
        v_p = cast<D_ATTN>(v_s);
        if(!below)
            scale_output_tile<T>(v_o, rescale_m);

        v_v[0] = tr_load<T::VEC_TR_V>(s_kv, u_rv);
        v_v[1] = tr_load<T::VEC_TR_V>(s_kv, u_rv + skv_slice(1_I));
        s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
        __builtin_amdgcn_sched_barrier(0);
        compute_pv(v_p, v_v, v_o_slices, tile_idx);
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        std::swap(s_kv, other_kv);
    }
}

} // namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1

namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1 {
template <class Traits, bool LateLoad, class Source>
__device__ void packed_prefill_accum128(
    const Source& source,
    int valid_kv_len,
    int num_kv_tiles,
    char* smem_kv_buf,
    opus::vector_t<typename Traits::D_ATTN,
                   Traits::Q_TILE_SIZE * Traits::D_TILE_SIZE / Traits::WARP_SIZE>& v_q,
    opus::vector_t<typename Traits::D_ACC, 8> (&v_o)[16],
    typename Traits::D_ACC& actual_max,
    typename Traits::D_ACC& m_row,
    typename Traits::D_ACC& l_row,
    typename Traits::D_ACC temperature_scale)
{
    using namespace opus;
    using T      = opus::remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    int lane_id = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id)); // break CSE
    const int warp_id =
        __builtin_amdgcn_readfirstlane((thread_id_x() / T::WARP_SIZE) % T::NUM_WARPS);

    // CSR bounds and int64 packed byte addresses are owned by Source.

    auto s_kv = make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf));

    auto mma0 =
        make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(seq<T::GEMM0_E_M, T::GEMM0_E_N, T::GEMM0_E_K>{},
                                              seq<T::T_M, T::T_N, T::T_K>{},
                                              seq<T::W_M, T::W_N, T::W_K>{},
                                              mfma_adaptor_swap_ab{});
    auto mma1 =
        make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
                                              seq<T::T_M, T::T_N, T::T_K>{},
                                              seq<T::W_M, T::W_N, T::W_K>{},
                                              mfma_adaptor_swap_ab{});

    auto u_gkv        = make_layout_gkv<T>(warp_id, lane_id);
    auto u_skv        = make_layout_skv<T>(warp_id);
    auto u_rk         = make_layout_rk<T>(lane_id);
    auto u_rv         = make_layout_rv<T>(lane_id);
    auto u_kv_indices = make_layout_kv_indices<T>(warp_id, lane_id);

    typename decltype(mma0)::vtype_b v_k[2];
    typename decltype(mma0)::vtype_c v_s;
    typename decltype(mma1)::vtype_a v_p;
    typename decltype(mma1)::vtype_b v_v[2];
    auto v_q_slices =
        reinterpret_cast<vector_t<D_ATTN, T::Q_TILE_SIZE * T::SLICE_D / T::WARP_SIZE>*>(&v_q);
    auto* v_o_slices = v_o;

    constexpr index_t s_len = vector_traits<typename decltype(mma0)::vtype_c>::size();

    auto skv_slice = [](auto slice_idx) {
        constexpr int s = decltype(slice_idx)::value;
        return number<(s / 2) * T::smem_n_rpt*(T::smem_linear_wave + T::smem_padding_32B) +
                      (s % 2) * T::SLICE_D>{};
    };

    int qk_tile      = 0;
    auto mask_scores = [&](auto& s) {
        static_for<2>([&](auto n) {
            int first = qk_tile * 32 + int(n) * 16 + (lane_id / 16) * 4;
            static_for<4>([&](auto j) {
                if(!source.valid(first + int(j)))
                    s[int(n) * 4 + int(j)] = -INFINITY;
            });
        });
        ++qk_tile;
    };
    auto compute_qk = [&](auto& s, const auto& q, auto& k) {
        clear(s);
        static_for<T::NUM_D_SLICES>([&](auto i) {
            constexpr int idx  = i.value;
            constexpr int slot = idx & 1;
            s                  = mma0(q[idx], k[slot], s);
            if constexpr(idx + 2 < T::NUM_D_SLICES)
            {
                k[slot] = load<T::VEC_KV>(s_kv, u_rk + skv_slice(number<idx + 2>{}));
                s_waitcnt_lgkmcnt(number<T::k_ds_read_insts>{});
            }
            else if constexpr(idx + 1 < T::NUM_D_SLICES)
            {
                s_waitcnt_lgkmcnt(0_I);
            }
        });
        mask_scores(s);
    };

    auto compute_pv = [&](const auto& p, auto& v, auto& o, int kv_tile_idx) {
        static_for<T::NUM_D_SLICES - 2>([&](auto i) {
            constexpr int idx  = i.value;
            constexpr int slot = idx & 1;
            o[idx]             = mma1(p, v[slot], o[idx]);
            v[slot]            = tr_load<T::VEC_TR_V>(s_kv, u_rv + skv_slice(number<idx + 2>{}));
            s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
            // Partial waits rely on LDS issue order and on which MFMA consumes
            // the completed reads. Allow VALU across this boundary, but keep
            // LDS, MFMA and SALU (including s_waitcnt) ordered. Allowing MFMA
            // or SALU here makes a later slice consume still-pending reads.
            // See experiments/20261014/PV_SCHEDULING.md for ISA and regressions.
            __builtin_amdgcn_sched_barrier(VALU_MASK);
        });
        o[T::NUM_D_SLICES - 2] = mma1(p, v[(T::NUM_D_SLICES - 2) & 1], o[T::NUM_D_SLICES - 2]);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_sched_barrier(0);
        o[T::NUM_D_SLICES - 1] = mma1(p, v[(T::NUM_D_SLICES - 1) & 1], o[T::NUM_D_SLICES - 1]);
    };

    const opus::u32_t neg_inf_v =
        std::bit_cast<opus::u32_t>(-opus::numeric_limits<D_ACC>::infinity());

    auto other_kv = make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf) + T::smem_kv_tile_elems);
    source.template load_tile<T>(0, s_kv.ptr, u_gkv, u_skv, warp_id, lane_id);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    for(int tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx)
    {
        if constexpr(!LateLoad)
        {
            if(tile_idx + 1 < num_kv_tiles)
            {
                source.template load_tile<T>(
                    tile_idx + 1, other_kv.ptr, u_gkv, u_skv, warp_id, lane_id);
            }
        }
        v_k[0] = load<T::VEC_KV>(s_kv, u_rk);
        v_k[1] = load<T::VEC_KV>(s_kv, u_rk + skv_slice(1_I));
        s_waitcnt_lgkmcnt(number<T::k_ds_read_insts>{});
        compute_qk(v_s, v_q_slices, v_k);

        D_ACC row_max = track_max<T>(v_s, temperature_scale, actual_max);
        bool below =
            (__builtin_amdgcn_ballot_w64(row_max - m_row <= 8.0f) == __builtin_amdgcn_read_exec());
        row_max         = below ? m_row : max(m_row, row_max);
        D_ACC rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
        m_row           = row_max;
        attn_row_scale_sub<T>(v_s, temperature_scale, row_max);
        attn_exp2_slice<T, 0, s_len>(v_s);
        l_row *= rescale_m;
        l_row += attn_row_sum<T>(v_s);
        v_p = cast<D_ATTN>(v_s);
        if(!below)
            scale_output_tile<T>(v_o, rescale_m);

        v_v[0] = tr_load<T::VEC_TR_V>(s_kv, u_rv);
        v_v[1] = tr_load<T::VEC_TR_V>(s_kv, u_rv + skv_slice(1_I));
        s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
        __builtin_amdgcn_sched_barrier(0);
        compute_pv(v_p, v_v, v_o_slices, tile_idx);
        if constexpr(LateLoad)
        {
            if(tile_idx + 1 < num_kv_tiles)
            {
                source.template load_tile<T>(
                    tile_idx + 1, other_kv.ptr, u_gkv, u_skv, warp_id, lane_id);
            }
        }

        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        std::swap(s_kv, other_kv);
    }
}

} // namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1

namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1 {
template <class Traits, int OutParts = 1, class Source>
__device__ void packed_prefill_accum_small(
    const Source& source,
    int valid_kv_len,
    int num_kv_tiles,
    char* smem_kv_buf,
    opus::vector_t<typename Traits::D_ATTN,
                   Traits::Q_TILE_SIZE * Traits::D_TILE_SIZE / Traits::WARP_SIZE>& v_q,
    opus::vector_t<typename Traits::D_ACC, 8> (&v_o)[16 / OutParts],
    typename Traits::D_ACC& actual_max,
    typename Traits::D_ACC& m_row,
    typename Traits::D_ACC& l_row,
    typename Traits::D_ACC temperature_scale)
{
    using namespace opus;
    using T      = opus::remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    int lane_id = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id)); // break CSE
    const int warp_id =
        __builtin_amdgcn_readfirstlane((thread_id_x() / T::WARP_SIZE) % T::NUM_WARPS);

    // CSR bounds and int64 packed byte addresses are owned by Source.

    auto s_kv = make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf));

    auto mma0 =
        make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(seq<T::GEMM0_E_M, T::GEMM0_E_N, T::GEMM0_E_K>{},
                                              seq<T::T_M, T::T_N, T::T_K>{},
                                              seq<T::W_M, T::W_N, T::W_K>{},
                                              mfma_adaptor_swap_ab{});
    auto mma1 =
        make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
                                              seq<T::T_M, T::T_N, T::T_K>{},
                                              seq<T::W_M, T::W_N, T::W_K>{},
                                              mfma_adaptor_swap_ab{});

    auto u_gkv            = make_layout_gkv<T>(warp_id, lane_id);
    auto u_skv            = make_layout_skv<T>(warp_id);
    auto u_rk             = make_layout_rk<T>(lane_id);
    constexpr int OSlices = 16 / OutParts;
    const int out_part    = warp_id % OutParts;
    auto u_rv             = make_layout_rv<T>(lane_id) + out_part * (OSlices / 2) * T::smem_n_rpt *
                                                 (T::smem_linear_wave + T::smem_padding_32B);
    auto u_kv_indices = make_layout_kv_indices<T>(warp_id, lane_id);

    typename decltype(mma0)::vtype_b v_k[2];
    typename decltype(mma0)::vtype_c v_s;
    typename decltype(mma1)::vtype_a v_p;
    typename decltype(mma1)::vtype_b v_v[2];
    auto v_q_slices =
        reinterpret_cast<vector_t<D_ATTN, T::Q_TILE_SIZE * T::SLICE_D / T::WARP_SIZE>*>(&v_q);
    auto* v_o_slices = v_o;

    constexpr index_t s_len = vector_traits<typename decltype(mma0)::vtype_c>::size();

    auto skv_slice = [](auto slice_idx) {
        constexpr int s = decltype(slice_idx)::value;
        return number<(s / 2) * T::smem_n_rpt*(T::smem_linear_wave + T::smem_padding_32B) +
                      (s % 2) * T::SLICE_D>{};
    };

    int qk_tile      = 0;
    auto mask_scores = [&](auto& s) {
        static_for<2>([&](auto n) {
            int first = int(n) * 16 + (lane_id / 16) * 4;
            static_for<4>([&](auto j) {
                int k          = first + int(j);
                auto* flag_lds = reinterpret_cast<OPUS_LDS_ADDR const D_ATTN*>(
                    reinterpret_cast<uintptr_t>(s_kv.ptr));
                if(!reinterpret_cast<OPUS_LDS_ADDR const uint32_t*>(flag_lds + (k % 4) * 528 +
                                                                    512)[k / 4])
                    s[int(n) * 4 + int(j)] = -INFINITY;
            });
        });
        ++qk_tile;
    };
    auto compute_qk = [&](auto& s, const auto& q, auto& k) {
        clear(s);
        static_for<T::NUM_D_SLICES>([&](auto i) {
            constexpr int idx  = i.value;
            constexpr int slot = idx & 1;
            s                  = mma0(q[idx], k[slot], s);
            if constexpr(idx + 2 < T::NUM_D_SLICES)
            {
                k[slot] = load<T::VEC_KV>(s_kv, u_rk + skv_slice(number<idx + 2>{}));
                s_waitcnt_lgkmcnt(number<T::k_ds_read_insts>{});
            }
            else if constexpr(idx + 1 < T::NUM_D_SLICES)
            {
                s_waitcnt_lgkmcnt(0_I);
            }
        });
        mask_scores(s);
    };

    auto compute_pv = [&](const auto& p, auto& v, auto& o, int kv_tile_idx) {
        static_for<OSlices - 2>([&](auto i) {
            constexpr int idx  = i.value;
            constexpr int slot = idx & 1;
            o[idx]             = mma1(p, v[slot], o[idx]);
            v[slot]            = tr_load<T::VEC_TR_V>(s_kv, u_rv + skv_slice(number<idx + 2>{}));
            s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
            // Partial waits rely on LDS issue order and on which MFMA consumes
            // the completed reads. Allow VALU across this boundary, but keep
            // LDS, MFMA and SALU (including s_waitcnt) ordered. Allowing MFMA
            // or SALU here makes a later slice consume still-pending reads.
            // See experiments/20261014/PV_SCHEDULING.md for ISA and regressions.
            __builtin_amdgcn_sched_barrier(VALU_MASK);
        });
        o[OSlices - 2] = mma1(p, v[(OSlices - 2) & 1], o[OSlices - 2]);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_sched_barrier(0);
        o[OSlices - 1] = mma1(p, v[(OSlices - 1) & 1], o[OSlices - 1]);
    };

    const opus::u32_t neg_inf_v =
        std::bit_cast<opus::u32_t>(-opus::numeric_limits<D_ACC>::infinity());

    typename Source::RawTile raw;
    auto other_kv = make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf) + T::smem_kv_tile_elems);
    source.template load_tile<T>(0, s_kv.ptr, u_gkv, u_skv, warp_id, lane_id);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    for(int tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx)
    {
        if(tile_idx + 1 < num_kv_tiles)
        {
            if constexpr(Source::is_packed)
                raw = source.prefetch(tile_idx + 1, warp_id, lane_id);
            else
                source.template load_tile<T>(
                    tile_idx + 1, other_kv.ptr, u_gkv, u_skv, warp_id, lane_id);
        }

        v_k[0] = load<T::VEC_KV>(s_kv, u_rk);
        v_k[1] = load<T::VEC_KV>(s_kv, u_rk + skv_slice(1_I));
        s_waitcnt_lgkmcnt(number<T::k_ds_read_insts>{});
        compute_qk(v_s, v_q_slices, v_k);

        D_ACC row_max = track_max<T>(v_s, temperature_scale, actual_max);
        bool below =
            (__builtin_amdgcn_ballot_w64(row_max - m_row <= 8.0f) == __builtin_amdgcn_read_exec());
        row_max         = below ? m_row : max(m_row, row_max);
        D_ACC rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
        m_row           = row_max;
        attn_row_scale_sub<T>(v_s, temperature_scale, row_max);
        attn_exp2_slice<T, 0, s_len>(v_s);
        l_row *= rescale_m;
        l_row += attn_row_sum<T>(v_s);
        v_p = cast<D_ATTN>(v_s);
        if(!below)
            static_for<OSlices>([&](auto i) { v_o[int(i)] *= rescale_m; });

        v_v[0] = tr_load<T::VEC_TR_V>(s_kv, u_rv);
        v_v[1] = tr_load<T::VEC_TR_V>(s_kv, u_rv + skv_slice(1_I));
        s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
        __builtin_amdgcn_sched_barrier(0);
        compute_pv(v_p, v_v, v_o_slices, tile_idx);
        if constexpr(Source::is_packed)
        {
            if(tile_idx + 1 < num_kv_tiles)
                source.commit(raw, other_kv.ptr, warp_id, lane_id);
        }
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        std::swap(s_kv, other_kv);
    }
}

} // namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1
