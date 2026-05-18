// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "hk_mla_buffer_managers.cuh"
#include "hk_mla_softmax.cuh"
#include "mla.h"
#include <assert.h>

using namespace hk_mla;

// V4.0 mi35x m16x8 decode kernel: separate FP8 NOPE + BF16 ROPE buffers for
// both Q and KV. End-to-end body (Phases 4a..4g) in place: prologue (Q load +
// first KV tile) -> per-warp dispatch ladder over mla_main (QK GEMM + softmax
// + PV GEMM + epilogue, with online-softmax rescale across K-tile iters).
#if defined(__gfx950__)
template <typename T>
__global__ __launch_bounds__(T::kNumThreads, T::kOccupancy)
    __attribute__((amdgpu_num_vgpr(68))) void
    kn_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16(HkMlaV40DecodeFwdParams<T> params)
{
    using q_nope_t  = T::q_nope_t;
    using q_rope_t  = T::q_rope_t;
    using kv_nope_t = T::kv_nope_t;
    using kv_rope_t = T::kv_rope_t;
    using out_t     = T::out_t;
    using comp_t    = float;
    using split_t   = float; // format of temp split output and lse.
    // All MFMA operands live in bf16 after the QManager/KvManager cvt step.
    using mfma_ab_t = hk::bf16;

    using G = hk::group<T::kNumWarps>;

    constexpr comp_t log2e = 1.4426950408889634;

    const int32_t worker_idx     = blockIdx.x;
    const int32_t work_start_idx = __builtin_amdgcn_readfirstlane(params.p_work_indptr[worker_idx]);
    const int32_t work_end_idx =
        __builtin_amdgcn_readfirstlane(params.p_work_indptr[worker_idx + 1]);
    if(work_start_idx >= work_end_idx)
    {
        return;
    }

    // ---- VGPR layout (per-lane, per spec hk_mla_v4_gen1.md §4.2) ----
    //
    //   v128..v255  : oaccu     (128 fp32 = full kVoHeadDim 512 / 16 cols-per-mfma * kTileM 16 / 64)
    //   v120..v127  : p_comp    (8  fp32, kBlockN=32 x kTileM=16 / 64)
    //     v120..v123: p_mfma    (4  bf16, OVERLAYS LOW HALF of p_comp; safe by low-to-high pack)
    //   v112..v119  : kv        (8  bf16, single 32x16 KV tile -- no kv_alt; see spec §4.2)
    //   v104..v111  : pv_v_aux  (8  bf16, second V-tile staging during PV)
    //   v72 ..v103  : q_vgpr    (32 bf16, Q[:, 0:256] in mfma A layout)
    //   v68 ..v71   : q_rope    (4  bf16, Q[:, 448:512] kept in VGPR for QK RoPE pass)
    //   v0  ..v67   : free / scratch (cvt staging, scale dwords, ds_read_b64_tr, etc.)
    //
    // Pinned total = 184 (matches spec). Compiler is constrained to v0..v67
    // for scratch via amdgpu_num_vgpr(68) on the __global__ -- without this,
    // scratch leaks into v68..v255 and clobbers pinned q_vgpr/kv/p_comp/oaccu.
    constexpr uint32_t k_o_sz        = 128;
    constexpr uint32_t k_p_comp_sz   = 8;
    constexpr uint32_t k_p_mfma_sz   = 4;
    constexpr uint32_t k_kv_sz       = 8;
    constexpr uint32_t k_pv_v_aux_sz = 8;
    constexpr uint32_t k_q_vgpr_sz   = 32;
    constexpr uint32_t k_q_rope_sz   = 4;

    constexpr uint32_t k_o_end          = 255;
    constexpr uint32_t k_o_begin        = k_o_end - k_o_sz + 1;             // 128
    constexpr uint32_t k_p_comp_end     = k_o_begin - 1;                    // 127
    constexpr uint32_t k_p_comp_begin   = k_p_comp_end - k_p_comp_sz + 1;   // 120
    constexpr uint32_t k_p_mfma_begin   = k_p_comp_begin + 0;               // 120 (overlay)
    constexpr uint32_t k_p_mfma_end     = k_p_mfma_begin + k_p_mfma_sz - 1; // 123
    constexpr uint32_t k_kv_end         = k_p_comp_begin - 1;               // 119
    constexpr uint32_t k_kv_begin       = k_kv_end - k_kv_sz + 1;           // 112
    constexpr uint32_t k_pv_v_aux_end   = k_kv_begin - 1;                   // 111
    constexpr uint32_t k_pv_v_aux_begin = k_pv_v_aux_end - k_pv_v_aux_sz + 1; // 104
    constexpr uint32_t k_q_vgpr_end     = k_pv_v_aux_begin - 1;             // 103
    constexpr uint32_t k_q_vgpr_begin   = k_q_vgpr_end - k_q_vgpr_sz + 1;   // 72
    constexpr uint32_t k_q_rope_end     = k_q_vgpr_begin - 1;               // 71
    constexpr uint32_t k_q_rope_begin   = k_q_rope_end - k_q_rope_sz + 1;   // 68

    // ---- art (auto-register-tile) range views ----
    //
    // q_vgpr holds Q[:, 0:256] in mfma A-operand layout: 8 mfma A-tiles total
    // (256 cols / 32 cols-per-mfma), each 4 vgprs/lane = 32 vgprs.
    using q_vgpr_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_q_vgpr_begin, k_q_vgpr_end>>,
                             4>; // 32 vgprs -> 8 ranges of 4 (8 16x32 base tiles, bf16)
    using q_rope_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_q_rope_begin, k_q_rope_end>>,
                             4>; // 4 vgprs -> 1 range of 4 (1 16x32 base tile, bf16)
    // split_many_t<list, N> splits each range into chunks of N vgprs each. N is
    // registers_per_thread per base tile for the chosen rt_shape + elem_t.
    //   rt_16x16_s + fp32 -> 4 vgprs/base
    //   rt_16x16_s + bf16 -> 2 vgprs/base
    //   rt_16x32_s + bf16 -> 4 vgprs/base
    using kv_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_kv_begin, k_kv_end>>,
                             4>; // 8 vgprs -> 2 ranges of 4: 2 base tiles (16x32 bf16) = 32x32
    using pv_v_aux_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_pv_v_aux_begin, k_pv_v_aux_end>>,
                             4>; // same shape as kv
    using p_comp_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_p_comp_begin, k_p_comp_end>>,
                             4>; // 8 vgprs -> 2 ranges of 4: 2 base tiles (16x16 fp32)
    // p_comp lo/hi halves over the same vgprs (each is 16 N-rows = 1 base tile).
    using p_comp_lo_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_p_comp_begin + 0,
                                                             k_p_comp_begin + 3>>,
                             4>;
    using p_comp_hi_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_p_comp_begin + 4,
                                                             k_p_comp_begin + 7>>,
                             4>;
    // kv top/bot halves over the same vgprs (each is 16 K-rows = 1 base tile).
    using kv_top_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_kv_begin + 0,
                                                             k_kv_begin + 3>>,
                             4>;
    using kv_bot_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_kv_begin + 4,
                                                             k_kv_begin + 7>>,
                             4>;
    using p_mfma_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_p_mfma_begin, k_p_mfma_end>>,
                             4>; // 4 vgprs -> 1 range of 4: 1 base tile (16x32 bf16)
    using o_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_o_begin, k_o_end>>, 4>; // 128 vgprs

    hkdart::clobber<q_vgpr_ranges>();
    hkdart::clobber<q_rope_ranges>();
    hkdart::clobber<kv_ranges>();
    hkdart::clobber<pv_v_aux_ranges>();
    hkdart::clobber<p_comp_ranges>();
    hkdart::clobber<p_mfma_ranges>();
    hkdart::clobber<o_ranges>();

    // ---- Managers ----
    QManager8to16bitsV1<T> q_manager;
    KvManager8to16bitsV1<T> kv_manager;
    OManager16bitsV2<T, out_t> o_manager;
    OManager32bitsV2<T, split_t> split_o_manager;

    // ---- art tile declarations ----
    // q_vgpr: Q[:, 0:256] held bf16 in VGPR, mfma A-operand layout.
    //   shape = (kTileM=16, 256), row_l, rt_16x32_s -> 8 base tiles x 4 vgprs = 32 vgprs.
    hk::art<mfma_ab_t, T::kTileM, 256, hk::row_l, hk::rt_16x32_s, q_vgpr_ranges> q_vgpr;
    // kv: K (QK) / V (PV) tile, mfma A-operand, bf16 in 16x32 base tiles.
    //   shape = (kBlockK=32, 32) -> 2 base tiles x 4 vgprs = 8 vgprs.
    hk::art<mfma_ab_t, T::kBlockK, 32, hk::row_l, hk::rt_16x32_s, kv_ranges> kv;
    hk::art<mfma_ab_t, T::kBlockK, 32, hk::row_l, hk::rt_16x32_s, pv_v_aux_ranges> pv_v_aux;
    // p_comp: kBlockN=32 N-cols x kTileM=16 M-rows in col_l mfma layout (= 2 base tiles fp32).
    hk::art<comp_t, T::kBlockN, T::kTileM, hk::col_l, hk::rt_16x16_s, p_comp_ranges> p_comp;
    // p_comp lo/hi: alternate views over the same vgprs, each (16, 16) = 1 base tile.
    // Lo covers N=0..15 (the kv_top mma writes), hi covers N=16..31 (the kv_bot mma writes).
    hk::art<comp_t, 16, T::kTileM, hk::col_l, hk::rt_16x16_s, p_comp_lo_ranges> p_comp_lo;
    hk::art<comp_t, 16, T::kTileM, hk::col_l, hk::rt_16x16_s, p_comp_hi_ranges> p_comp_hi;
    // kv top/bot: 16 K-rows each = 1 base tile of (16, 32) bf16, used for QK B-operand.
    hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, kv_top_ranges> kv_top;
    hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, kv_bot_ranges> kv_bot;
    // p_mfma: bf16 P-operand for PV mfma, row_l 16x32 (4 vgprs/lane = 1 base tile).
    hk::art<mfma_ab_t, T::kTileM, T::kBlockN, hk::row_l, hk::rt_16x32_s, p_mfma_ranges> p_mfma;
    // oaccu: full kVoHeadDim=512 wide, kTileM=16 rows, row_l 16x16 sub-tiles (fp32).
    hk::art<comp_t, T::kTileM, T::kVoHeadDim, hk::row_l, hk::rt_16x16_s, o_ranges> oaccu;

    // ---- Runtime constants ----
    const uint32_t warp_idx = __builtin_amdgcn_readfirstlane(threadIdx.x / opus::get_warp_size());
    const uint32_t lane_idx = opus::lane_id();

    // Causal mask: compute per-warp kv_end offset for MTP.
    // num_wave_group = qseqlen = kBlockM / num_qheads
    // waves_per_head = num_qheads / kTileM
    // causal_offset = num_wave_group - 1 - (warp_idx / waves_per_head)
    const int32_t log2_num_qheads     = __builtin_amdgcn_readfirstlane(params.log2_num_qheads);
    const int32_t num_qheads          = 1 << log2_num_qheads;
    const int32_t num_wave_group      = T::kBlockM >> log2_num_qheads; // qseqlen
    const int32_t log2_waves_per_head = log2_num_qheads - 4;           // log2(kTileM) = 4
    const int32_t qpos_off_from_last  = num_wave_group - 1 - (warp_idx >> log2_waves_per_head);

    const uintptr_t out_as_int       = reinterpret_cast<uintptr_t>(params.final_output.raw_ptr);
    const uint64_t out_as_u64        = static_cast<uint64_t>(out_as_int);
    const hk::buffer_resource out_br = hk::make_buffer_resource(out_as_u64, 0xFFFFFFFF, 0x00020000);
    const uintptr_t split_out_as_int = reinterpret_cast<uintptr_t>(params.split_output.raw_ptr);
    const uint64_t split_out_as_u64  = static_cast<uint64_t>(split_out_as_int);
    const hk::buffer_resource split_out_br =
        hk::make_buffer_resource(split_out_as_u64, 0xFFFFFFFF, 0x00020000);

    // ---- LDS layout ----
    //
    // p_lds_q          : 64 KB - QManager region. Phase 1 of load_q transiently
    //                    aliases the first 16 KB as per-warp staging; Phase 2
    //                    overwrites those bytes as part of the 64 KB final
    //                    bf16-Q residency. After load_q returns, the +0..16 KB
    //                    window is free (kernel reuses it as O bounce buffer).
    // p_lds_kv_curr/   : 36 KB each (32 rows * 512 bf16 cols + RoPE * 2-buf).
    //  p_lds_kv_next
    //
    // Total (occupancy=1): 64 KB Q + 2 * KvLds <= 160 KB.
    extern __shared__ int32_t p_lds[];

    constexpr uint32_t kSzLdsQ  = q_manager.get_lds_size_in_byte();
    constexpr uint32_t kSzLdsKv = kv_manager.get_lds_size_in_byte();
    constexpr uint32_t kSzLdsO =
        (o_manager.get_lds_size_in_byte() > split_o_manager.get_lds_size_in_byte())
            ? o_manager.get_lds_size_in_byte()
            : split_o_manager.get_lds_size_in_byte();

    // O bounce overlays the +0..16 KB window of p_lds_q (free after load_q).
    static_assert(kSzLdsO <= kSzLdsQ,
                  "kSzLdsO must fit within p_lds_q so the O bounce can overlay it.");
    static_assert(kSzLdsQ + 2u * kSzLdsKv <= 160u * 1024u,
                  "V4.0 LDS budget exceeds 160 KB at kOccupancy=1.");

    const uintptr_t p_lds_q       = reinterpret_cast<uintptr_t>(p_lds);
    uintptr_t       p_lds_kv_curr = p_lds_q + kSzLdsQ;
    uintptr_t       p_lds_kv_next = p_lds_kv_curr + kSzLdsKv;

    // ---- Work loop ----
    // Phase 4b is in place: per work item, read work_info, resolve kv extents,
    // load Q (vmem -> VGPR + bf16 LDS), and prefetch+cvt+store the first KV
    // tile into the curr pong. The mla_main lambda + dispatch ladder still TODO
    // (Phases 4c-4f); kernel still hits assert(false) at the bottom of the loop.
    const uint32_t kv_ld_row_base_idx = kv_manager.get_kv_ld_row_base_idx(warp_idx);

    for(int32_t work_idx = work_start_idx; work_idx < work_end_idx; ++work_idx)
    {
        const int32_t partial_qo_loc = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 1]);
        const int32_t qo_start = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 2]);
        const int32_t qo_end = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 3]);
        const int32_t kv_start_page = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 4]);
        const int32_t kv_end_page = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 5]);
        // kv_offset == 0 iff this work item ends at the batch tail (kPageSize > 1).
        const int32_t kv_offset = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 6]);

        // Convert work_info page bounds to TOKEN space. When kPageSize == 1
        // pages == tokens. When kPageSize > 1 and this is the batch tail
        // (kv_offset == 0), clip the last page with kv_last_page_lens[batch].
        // The (kPageSize == 1) check folds at compile time so the load is
        // dead-code-eliminated for kPageSize == 1.
        const int32_t kv_start = kv_start_page * T::kPageSize;
        int32_t       kv_end;
        if((T::kPageSize == 1) || (kv_offset != 0))
        {
            kv_end = kv_end_page * T::kPageSize;
        }
        else
        {
            const int32_t batch_idx = __builtin_amdgcn_readfirstlane(
                params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 0]);
            const int32_t last_page_len =
                __builtin_amdgcn_readfirstlane(params.p_kv_last_page_lens[batch_idx]);
            kv_end = (kv_end_page - 1) * T::kPageSize + last_page_len;
        }
        // Per-warp causal mask: qpos i sees kv_end - max(0, qpos_off_from_last - kv_offset).
        const int32_t causal_offset = opus::max(qpos_off_from_last - kv_offset, 0);
        const int32_t kv_end_eff    = kv_end - causal_offset;
        const int32_t kv_len        = kv_end - kv_start;
        const int32_t kv_len_eff    = kv_end_eff - kv_start;

        // Online-softmax running stats. Each warp owns one (16-row) M-tile; the
        // values are lane-private (each lane holds the stats for its 1/64th
        // share of the tile, established by the warp_reduce inside softmax_p0).
        comp_t row_max;
        comp_t row_sum_e;

        // Helper: resolve the physical KV row for the 32-row tile that begins
        // at tile_start. Returns -1 if the tile is entirely OOB.
        auto resolve_row_kv_ld = [&](const int32_t tile_start) -> int32_t {
            const int32_t tile_end = tile_start + T::kBlockN;
            int32_t       row_kv_ld;
            if(tile_end <= kv_end)
            {
                row_kv_ld = get_kv_ld_row<false, T::kPageSize>(
                    params.p_kv_indices, kv_ld_row_base_idx, tile_start, tile_end);
            }
            else if(tile_start < kv_end)
            {
                row_kv_ld = get_kv_ld_row<true, T::kPageSize>(
                    params.p_kv_indices, kv_ld_row_base_idx, tile_start, kv_end);
            }
            else
            {
                row_kv_ld = -1;
            }
            return row_kv_ld;
        };

        // Tile 0's KV row goes to the prologue; tile 1's seed row goes to the
        // first lambda call's prefetch. `row_kv_ld_next_next` is a one-deep
        // carry: each lambda call snapshots it for its prefetch and updates it
        // for the call after (matching V32's per-warp dispatch pattern).
        const int32_t row_kv_ld_first = resolve_row_kv_ld(kv_start);
        int32_t       row_kv_ld_next_next =
            (kv_len > T::kBlockN) ? resolve_row_kv_ld(kv_start + T::kBlockN) : -1;

        // Load Q: Q[:, 0:256] -> VGPR pinned at k_q_vgpr_begin (32 vgprs/lane).
        //         Q[:, 256:512] -> bf16 final LDS region inside p_lds_q.
        // Q rope/nope buffers are separate tensors in V4.0.
        q_manager.template load_q<k_q_vgpr_begin>(
            params.query, params.query_rope, warp_idx, qo_start, p_lds_q);
        __builtin_amdgcn_sched_barrier(0);

        // Prologue: prefetch + cvt+store the first KV tile into the curr pong.
        // kCheckBoundary is true when the tile straddles the batch tail.
        if(kv_len < T::kBlockN)
        {
            kv_manager.template async_load_k<true>(p_lds_kv_curr,
                                                   warp_idx,
                                                   params.kv_buffer,
                                                   params.kv_buffer_rope,
                                                   row_kv_ld_first);
        }
        else
        {
            kv_manager.template async_load_k<false>(p_lds_kv_curr,
                                                    warp_idx,
                                                    params.kv_buffer,
                                                    params.kv_buffer_rope,
                                                    row_kv_ld_first);
        }

        // ---- mla_main lambda (Phase 4g) ----
        //
        // One K-tile iter. Templates:
        //   kIsFirstIter      : this is the warp's first compute iter (oaccu
        //                       gets initialized by PV's 3-arg mfma, no
        //                       rescale needed against prior row_max/oaccu).
        //   kSkipCompute      : warp is idle on this tile (e.g., causal-masked
        //                       trailing iter); only barriers + KV cooperative
        //                       work run. Implies !kIsFirstIter.
        //   kEpilogueType     : None (continue) / OutputFinal / OutputSplit.
        //   kCheckBoundaryNext: the NEXT tile may be OOB (partial last tile);
        //                       prefetch uses kCheckBoundary=true.
        //
        // Derived: kDoEpilogue = (kEpilogueType != None);
        //          kIsGlobalLast = kSkipCompute || kDoEpilogue.
        // kIsGlobalLast means no next tile to load -- skip prefetch, wait, swap.
        auto mla_main = [&]<bool kIsFirstIter,
                            bool kSkipCompute,
                            PvGemmEpilogueType kEpilogueType,
                            bool kCheckBoundaryNext>(const int32_t kv_tile_start,
                                                     const int32_t kv_tile_end) {
            constexpr bool kDoEpilogue   = (kEpilogueType != PvGemmEpilogueType::None);
            constexpr bool kIsGlobalLast = kSkipCompute || kDoEpilogue;
            (void)kv_tile_end;

            static_assert((kSkipCompute == false) || (kIsFirstIter == false),
                          "A skipped iter cannot be the warp's first compute iter.");
            static_assert((kIsGlobalLast == false) || (kCheckBoundaryNext == false),
                          "kIsGlobalLast == true means no next tile, so kCheckBoundaryNext must be false.");

            // Drain prior iter's vmem+LDS, cross-warp barrier so all KV LDS
            // sub-blocks (each warp writes its own 16x64 patch) are visible to
            // QK reads. This also gates the prologue's KV writes on iter 0.
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Snapshot next-tile KV row (set by prior call or prologue).
            int32_t row_kv_ld_next = 0;
            if constexpr(kIsGlobalLast == false)
            {
                row_kv_ld_next = row_kv_ld_next_next;
            }

            // ---- Phase A: prefetch NEXT tile into the next-pong ----
            // 2 halves (kColOffset 0 + 256) per tile. Carriers live in VGPRs
            // until the wait+cvt+store sequence below; the gap in between
            // hides vmcnt latency under QK MFMAs.
            typename KvManager8to16bitsV1<T>::KvTilePrefetch p0, p1;
            if constexpr(kIsGlobalLast == false)
            {
                constexpr uint32_t kTileCols = 256u;
                kv_manager.template prefetch_kv_tile<0u, 0u, kCheckBoundaryNext>(
                    p_lds_kv_next, warp_idx, params.kv_buffer, params.kv_buffer_rope,
                    row_kv_ld_next, p0);
                kv_manager.template prefetch_kv_tile<0u, kTileCols, kCheckBoundaryNext>(
                    p_lds_kv_next, warp_idx, params.kv_buffer, params.kv_buffer_rope,
                    row_kv_ld_next, p1);
            }

            // ---- QK GEMM ----
            constexpr uint32_t kNumQkVgprIter = 8;
            if constexpr(kSkipCompute == false)
            {
                // Phase A: 8 iters over Q[:, 0:256] which is pinned in q_vgpr.
                // Each iter k loads 32 K-cols from KV LDS into kv_top (rows 0..15)
                // + kv_bot (rows 16..31), then runs 2 x mma_ABt against q_k --
                // one per N-half of p_comp (lo = N rows 0..15, hi = N rows 16..31).
                // p_comp is initialized by iter 0's 3-arg mma; later iters accumulate.
                opus::static_for<kNumQkVgprIter>([&](auto idx) {
                    constexpr uint32_t kColOffset = idx.value * T::kBlockK;
                    constexpr uint32_t kQReg      = k_q_vgpr_begin + idx.value * 4u;
                    using q_range_k               = hkdart::split_many_t<
                        hkdart::type_list<hkdart::range<kQReg, kQReg + 3u>>, 4>;
                    hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, q_range_k>
                        q_k;

                    kv_manager.template load_k_to_gpr<0u,  kColOffset>(kv_top, p_lds_kv_curr);
                    kv_manager.template load_k_to_gpr<16u, kColOffset>(kv_bot, p_lds_kv_curr);

                    // ds_read + mfma asm wrappers are opaque to the compiler, so
                    // the lgkmcnt drain isn't inserted automatically. Drain to <=1
                    // (kv_top ready, kv_bot still in flight) -- overlaps the bot
                    // ds_read with the top mma -- then drain fully before the bot
                    // mma.
                    __builtin_amdgcn_s_waitcnt(
                        hk_mla::encode_s_waitcnt(/*lgkmcnt=*/1, /*vmcnt=*/-1));
                    if constexpr(idx.value == 0)
                    {
                        hk::mma_ABt(p_comp_lo, kv_top, q_k);
                        __builtin_amdgcn_s_waitcnt(
                            hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/-1));
                        hk::mma_ABt(p_comp_hi, kv_bot, q_k);
                    }
                    else
                    {
                        hk::mma_ABt(p_comp_lo, kv_top, q_k, p_comp_lo);
                        __builtin_amdgcn_s_waitcnt(
                            hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/-1));
                        hk::mma_ABt(p_comp_hi, kv_bot, q_k, p_comp_hi);
                    }
                });

                // Phase B: 8 iters over Q[:, 256:512] which lives in the bf16
                // final Q LDS region (col_tile 0..5 = NoPE Q cols 256..447,
                // col_tile 6,7 = RoPE Q cols 448..511). K-loads continue at
                // kColOffset 256..480. Q-from-LDS lands in a 4-vgpr scratch
                // range overlaying pv_v_aux's first 4 vgprs -- safe because
                // pv_v_aux is dead until the PV phase.
                constexpr uint32_t kQLdsScratchReg = k_pv_v_aux_begin;
                opus::static_for<kNumQkVgprIter>([&](auto idx) {
                    constexpr uint32_t kColTile   = idx.value; // 0..7
                    constexpr uint32_t kColOffset = (kNumQkVgprIter + idx.value) * T::kBlockK;
                    using q_range_k               = hkdart::split_many_t<
                        hkdart::type_list<hkdart::range<kQLdsScratchReg, kQLdsScratchReg + 3u>>, 4>;
                    hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, q_range_k>
                        q_k;

                    q_manager.template load_q_lds_to_gpr<kColTile>(q_k, p_lds_q, warp_idx);
                    kv_manager.template load_k_to_gpr<0u,  kColOffset>(kv_top, p_lds_kv_curr);
                    kv_manager.template load_k_to_gpr<16u, kColOffset>(kv_bot, p_lds_kv_curr);

                    // 3 ds_reads issued (q_k, kv_top, kv_bot). lgkmcnt(1) lets
                    // the first 2 (q_k + kv_top) settle while kv_bot overlaps
                    // the top mma; lgkmcnt(0) drains kv_bot before the bot mma.
                    __builtin_amdgcn_s_waitcnt(
                        hk_mla::encode_s_waitcnt(/*lgkmcnt=*/1, /*vmcnt=*/-1));
                    hk::mma_ABt(p_comp_lo, kv_top, q_k, p_comp_lo);
                    __builtin_amdgcn_s_waitcnt(
                        hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/-1));
                    hk::mma_ABt(p_comp_hi, kv_bot, q_k, p_comp_hi);
                });
            }

            // ---- Phase B+C: wait + cvt + store NEXT tile to LDS ----
            // Sequenced after QK so the QK ds_reads from p_lds_kv_curr aren't
            // delayed by the cvt+store traffic on p_lds_kv_next.
            if constexpr(kIsGlobalLast == false)
            {
                constexpr uint32_t kTileCols = 256u;
                kv_manager.template wait_kv_loads<0u, 0u>(warp_idx);
                kv_manager.template cvt_and_store_kv_tile<0u, 0u>(p_lds_kv_next, warp_idx, p0);
                kv_manager.template wait_kv_loads<0u, kTileCols>(warp_idx);
                kv_manager.template cvt_and_store_kv_tile<0u, kTileCols>(
                    p_lds_kv_next, warp_idx, p1);
            }

            // ---- Update row_kv_ld_next_next for the call AFTER this one ----
            // Only meaningful when (a) this iter has a next tile (kIsGlobalLast
            // == false) and (b) the next tile is not the partial last
            // (kCheckBoundaryNext == false). In the kCheckBoundaryNext case
            // there's no tile-after-next, so leave the carry untouched.
            if constexpr((kIsGlobalLast == false) && (kCheckBoundaryNext == false))
            {
                row_kv_ld_next_next = resolve_row_kv_ld(kv_tile_start + 2 * T::kBlockN);
            }

            // ---- Softmax + fp32->bf16 pack ----
            //
            // p_comp is 8 fp32 lanes (kBlockN=32 N-cols x kTileM=16 rows / 64
            // lanes = 8 elems/lane), laid out per softmax_scale_p_8: lane's
            // col_0 group covers vgprs +0..+3 (N-cols [col_0_idx*4, +4)) and
            // col_1 group covers +4..+7 (N-cols [col_0_idx*4+16, +20)).
            const uint32_t col_0_idx = lane_idx >> 4;
            comp_t         local_max{};
            comp_t         rescale = 1.0f;
            if constexpr(kSkipCompute == false)
            {
                const uint32_t kv_tile_start_u = static_cast<uint32_t>(kv_tile_start);
                if((kv_tile_start_u + T::kBlockN) > static_cast<uint32_t>(kv_end_eff))
                {
                    softmax_scale_p<true, k_p_comp_begin>(
                        col_0_idx * 4u + kv_tile_start_u,
                        static_cast<uint32_t>(kv_end_eff),
                        params.softmax_scale);
                }
                else
                {
                    softmax_scale_p<false, k_p_comp_begin>(
                        col_0_idx * 4u + kv_tile_start_u,
                        static_cast<uint32_t>(kv_end_eff),
                        params.softmax_scale);
                }

                // Row-wise max across 8 p_comp vgprs, then across the 4-lane
                // M-group via warp_reduce (matches softmax_p0's reduction).
                local_max = max_8<k_p_comp_begin, comp_t>();
                {
                    constexpr int32_t reduce_range = opus::get_warp_size();
                    constexpr int32_t stop_stride  = opus::get_warp_size() / 4 - 1;
                    local_max = warp_reduce<aiter::MaxFunctor,
                                            decltype(local_max),
                                            reduce_range,
                                            stop_stride>(local_max);
                }
                const comp_t new_row_max =
                    kIsFirstIter ? local_max : opus::max(local_max, row_max);
                rescale =
                    kIsFirstIter ? 1.0f
                                 : __builtin_amdgcn_exp2f((row_max - new_row_max) * log2e);
                row_max = new_row_max;

                // exp + sum + warp_reduce(add) -> row_sum_e. Updates p_comp in
                // place to hold exp(p_comp - new_row_max).
                softmax_p1<kIsFirstIter, k_p_comp_begin>(&row_sum_e, row_max, rescale);

                // ---- fp32->bf16 pack (p_comp -> p_mfma overlay) ----
                // 8 fp32 (v120..v127) -> 4 bf16x2 dwords (v120..v123 overlay).
                // Low-to-high pack order is hazard-free: each v_cvt_pk_bf16_f32
                // is atomic (reads sources before writing dst), and no later
                // pack reads a vgpr that an earlier pack overwrote.
                pack_2f32_to_bf16_pair_pinned<k_p_mfma_begin + 0, k_p_comp_begin + 0>();
                pack_2f32_to_bf16_pair_pinned<k_p_mfma_begin + 1, k_p_comp_begin + 2>();
                pack_2f32_to_bf16_pair_pinned<k_p_mfma_begin + 2, k_p_comp_begin + 4>();
                pack_2f32_to_bf16_pair_pinned<k_p_mfma_begin + 3, k_p_comp_begin + 6>();
            }

            // ---- Rescale accumulated oaccu before PV mfma (non-first iter) ----
            // Online-softmax: oaccu_new = oaccu_prev * exp(row_max_prev - row_max_new) + P_new @ V_new.
            // Apply the scalar rescale to oaccu BEFORE the 4-arg accum mfmas below.
            if constexpr((kSkipCompute == false) && (kIsFirstIter == false))
            {
                hk::mul_vgpr(oaccu, oaccu, rescale);
            }

            // ---- PV GEMM ----
            //
            // O = P @ V, computed as oaccu^T = V^T @ P^T via
            // mma_ABt(oaccu, kv, p_mfma). For V4.0 kBlockN=32, each iter
            // covers 32 V-cols (= 2 mfma A-tiles = both base tiles of kv) and
            // writes 2 oaccu base tiles. With kVoHeadDim=512 we run 16 iters.
            //
            // Per iter: 4 ds_read_b64_tr_b16 to fill kv (8 vgprs = 2 A-tiles),
            // wait lgkmcnt(0), 2 mfmas (3-arg init when kIsFirstIter, else
            // 4-arg accum).
            //
            // Single-buffered (pv_v_aux unused in Gen.1 -- deferred).
            constexpr uint32_t num_pv_iter = T::kVoHeadDim / T::kBlockN; // 16
            if constexpr(kSkipCompute == false)
            {
                opus::static_for<num_pv_iter>([&](auto i) {
                    constexpr uint32_t iter       = i.value;
                    constexpr uint32_t kColOffset = iter * T::kBlockN;

                    kv_manager
                        .template load_transposed_v_to_gpr<0u, kColOffset + 0u, k_kv_begin + 0>(
                            p_lds_kv_curr);
                    kv_manager
                        .template load_transposed_v_to_gpr<16u, kColOffset + 0u, k_kv_begin + 2>(
                            p_lds_kv_curr);
                    kv_manager
                        .template load_transposed_v_to_gpr<0u, kColOffset + 16u, k_kv_begin + 4>(
                            p_lds_kv_curr);
                    kv_manager
                        .template load_transposed_v_to_gpr<16u, kColOffset + 16u, k_kv_begin + 6>(
                            p_lds_kv_curr);

                    // Per-iter oaccu views: 2 adjacent 16x16 col_l base tiles
                    // (vgprs k_o_begin + iter*8 .. +7).
                    constexpr uint32_t oaccu_base = k_o_begin + iter * 8u;
                    using oaccu_a_r               = hkdart::split_many_t<
                        hkdart::type_list<hkdart::range<oaccu_base + 0, oaccu_base + 3>>, 4>;
                    using oaccu_b_r = hkdart::split_many_t<
                        hkdart::type_list<hkdart::range<oaccu_base + 4, oaccu_base + 7>>, 4>;
                    hk::art<comp_t, T::kTileM, T::kTileM, hk::col_l, hk::rt_16x16_s, oaccu_a_r>
                        oaccu_a;
                    hk::art<comp_t, T::kTileM, T::kTileM, hk::col_l, hk::rt_16x16_s, oaccu_b_r>
                        oaccu_b;

                    __builtin_amdgcn_s_waitcnt(
                        hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/-1));
                    if constexpr(kIsFirstIter)
                    {
                        hk::mma_ABt(oaccu_a, kv_top, p_mfma);
                        hk::mma_ABt(oaccu_b, kv_bot, p_mfma);
                    }
                    else
                    {
                        hk::mma_ABt(oaccu_a, kv_top, p_mfma, oaccu_a);
                        hk::mma_ABt(oaccu_b, kv_bot, p_mfma, oaccu_b);
                    }
                });
            }

            // ---- Epilogue ----
            //
            // Rescale oaccu by 1/row_sum_e (single mul_vgpr over full 128-vgpr
            // tile), then write 16-row x kVoHeadDim tile to vmem.
            //   partial_qo_loc < 0 -> final_output via OManager16bitsV2 (bf16).
            //   partial_qo_loc >= 0 -> split_output via OManager32bitsV2 (fp32)
            //                          + per-warp LSE row (lanes 0..15).
            // O LDS bounce overlays p_lds_q's first kSzLdsO bytes (free after
            // load_q).
            if constexpr(kDoEpilogue)
            {
                const comp_t reci_row_sum_e = 1.0f / row_sum_e;
                hk::mul_vgpr(oaccu, oaccu, reci_row_sum_e);

                const uintptr_t p_lds_o = p_lds_q;
                if constexpr(kEpilogueType == PvGemmEpilogueType::OutputFinal)
                {
                    opus::static_for<num_pv_iter>([&](auto i) {
                        constexpr uint32_t iter       = i.value;
                        constexpr uint32_t kOaccuBase = k_o_begin + iter * 8u;
                        constexpr uint32_t kColOff    = iter * T::kBlockN;
                        o_manager.template output_to_vram<kOaccuBase, kColOff>(
                            params.final_output.raw_ptr,
                            warp_idx,
                            qo_start,
                            p_lds_o,
                            num_qheads);
                    });
                }
                else
                {
                    opus::static_for<num_pv_iter>([&](auto i) {
                        constexpr uint32_t iter       = i.value;
                        constexpr uint32_t kOaccuBase = k_o_begin + iter * 8u;
                        constexpr uint32_t kColOff    = iter * T::kBlockN;
                        split_o_manager.template output_to_vram<kOaccuBase, kColOff>(
                            params.split_output.raw_ptr,
                            warp_idx,
                            static_cast<uint32_t>(partial_qo_loc),
                            p_lds_o,
                            num_qheads);
                    });

                    // LSE: row_max + ln(row_sum_e). Lanes 0..15 own the M-rows
                    // after warp_reduce; lanes 16..63 hold redundant copies.
                    constexpr uint32_t kMfmaResultRows = 16;
                    if(lane_idx < kMfmaResultRows)
                    {
                        constexpr comp_t inv_log2e = 1.0f / log2e;
                        const uint32_t   row_idx =
                            lane_idx + warp_idx * kMfmaResultRows +
                            static_cast<uint32_t>(partial_qo_loc) * num_qheads;
                        const comp_t lse =
                            row_max + __builtin_amdgcn_logf(row_sum_e) * inv_log2e;
                        params.split_lse.raw_ptr[row_idx] = lse;
                    }
                }
            }

            // ---- Swap pongs ----
            // No-op on the global last iter (the swap-target is not consumed).
            if constexpr(kIsGlobalLast == false)
            {
                std::swap(p_lds_kv_curr, p_lds_kv_next);
            }
        };

        // ---- Per-warp dispatch ladder ----
        //
        // All warps execute the same number of global tiles. On tiles past
        // this warp's effective end (kv_end_eff), the warp dispatches mla_main
        // with kSkipCompute=true: still participates in barriers + cooperative
        // KV cvt+store but skips QK/softmax/PV. Epilogue fires only on the
        // global last tile and is synchronized across all working warps.
        //
        // Per-warp causal_offset < kBlockN (qseqlen <= 8, kBlockN = 32) means
        // num_iters_eff in {0, num_iters - 1, num_iters}: at most 1 trailing
        // skip iter. Same ladder shape as V32 m16x8.
        if(kv_len_eff <= 0)
        {
            // Warp fully idle. num_iters == 1. One skip iter on the global
            // last tile, no epilogue (no oaccu state).
            mla_main.template operator()<false, true, PvGemmEpilogueType::None, false>(
                kv_start, kv_end);
        }
        else if(kv_len_eff < T::kBlockN)
        {
            // Warp has exactly 1 partial real tile.
            if(kv_len < T::kBlockN)
            {
                // num_iters == 1: single real iter, also the epilogue iter.
                if(partial_qo_loc < 0)
                {
                    mla_main.template operator()<true,
                                                 false,
                                                 PvGemmEpilogueType::OutputFinal,
                                                 false>(kv_start, kv_end);
                }
                else
                {
                    mla_main.template operator()<true,
                                                 false,
                                                 PvGemmEpilogueType::OutputSplit,
                                                 false>(kv_start, kv_end);
                }
            }
            else
            {
                // num_iters == 2: real (partial) iter on tile 0, then
                // skip+epilogue on tile 1.
                mla_main.template operator()<true, false, PvGemmEpilogueType::None, true>(
                    kv_start, kv_start + T::kBlockN);
                if(partial_qo_loc < 0)
                {
                    mla_main.template operator()<false,
                                                 true,
                                                 PvGemmEpilogueType::OutputFinal,
                                                 false>(kv_start + T::kBlockN, kv_end);
                }
                else
                {
                    mla_main.template operator()<false,
                                                 true,
                                                 PvGemmEpilogueType::OutputSplit,
                                                 false>(kv_start + T::kBlockN, kv_end);
                }
            }
        }
        else if(kv_len_eff == T::kBlockN)
        {
            // Warp has exactly 1 exact (full) real tile.
            if(kv_len == T::kBlockN)
            {
                if(partial_qo_loc < 0)
                {
                    mla_main.template operator()<true,
                                                 false,
                                                 PvGemmEpilogueType::OutputFinal,
                                                 false>(kv_start, kv_end);
                }
                else
                {
                    mla_main.template operator()<true,
                                                 false,
                                                 PvGemmEpilogueType::OutputSplit,
                                                 false>(kv_start, kv_end);
                }
            }
            else
            {
                // num_iters == 2: exact real iter on tile 0, then skip+epilogue
                // on tile 1. kCheckBoundaryNext iff global last tile is partial.
                const bool boundary_next = (kv_len % T::kBlockN) != 0;
                if(boundary_next)
                {
                    mla_main.template operator()<true, false, PvGemmEpilogueType::None, true>(
                        kv_start, kv_start + T::kBlockN);
                }
                else
                {
                    mla_main.template operator()<true, false, PvGemmEpilogueType::None, false>(
                        kv_start, kv_start + T::kBlockN);
                }
                if(partial_qo_loc < 0)
                {
                    mla_main.template operator()<false,
                                                 true,
                                                 PvGemmEpilogueType::OutputFinal,
                                                 false>(kv_start + T::kBlockN, kv_end);
                }
                else
                {
                    mla_main.template operator()<false,
                                                 true,
                                                 PvGemmEpilogueType::OutputSplit,
                                                 false>(kv_start + T::kBlockN, kv_end);
                }
            }
        }
        else // kv_len_eff > kBlockN: warp has >= 2 real tiles
        {
            const int32_t kv_1st_end = kv_start + T::kBlockN;

            // First real tile (kIsFirstIter=true). Next-tile boundary check
            // iff the tile being prefetched (tile 1) is the global last AND
            // partial.
            if((kv_1st_end + T::kBlockN - 1) < kv_end)
            {
                mla_main.template operator()<true, false, PvGemmEpilogueType::None, false>(
                    kv_start, kv_1st_end);
            }
            else
            {
                mla_main.template operator()<true, false, PvGemmEpilogueType::None, true>(
                    kv_start, kv_1st_end);
            }

            int32_t kv_idx = kv_1st_end;
            // Middle real tiles: while next tile is not warp's last real.
            while((kv_idx + T::kBlockN) < kv_end_eff)
            {
                if((kv_idx + 2 * T::kBlockN - 1) < kv_end)
                {
                    mla_main
                        .template operator()<false, false, PvGemmEpilogueType::None, false>(
                            kv_idx, kv_idx + T::kBlockN);
                }
                else
                {
                    mla_main.template operator()<false, false, PvGemmEpilogueType::None, true>(
                        kv_idx, kv_idx + T::kBlockN);
                }
                kv_idx += T::kBlockN;
            }

            // Warp's last real tile starts at kv_idx. It may or may not
            // coincide with the global last tile.
            const bool tile_is_global_last = ((kv_idx + T::kBlockN) >= kv_end);

            if(tile_is_global_last)
            {
                // Warp's last real == global last -> real iter with epilogue.
                if(partial_qo_loc < 0)
                {
                    mla_main.template operator()<false,
                                                 false,
                                                 PvGemmEpilogueType::OutputFinal,
                                                 false>(kv_idx, kv_end);
                }
                else
                {
                    mla_main.template operator()<false,
                                                 false,
                                                 PvGemmEpilogueType::OutputSplit,
                                                 false>(kv_idx, kv_end);
                }
            }
            else
            {
                // Warp's last real is NOT the global last; one trailing skip
                // iter does the epilogue. Real iter prefetches K for the
                // global last tile.
                const bool boundary_next = (kv_len % T::kBlockN) != 0;
                if(boundary_next)
                {
                    mla_main.template operator()<false, false, PvGemmEpilogueType::None, true>(
                        kv_idx, kv_idx + T::kBlockN);
                }
                else
                {
                    mla_main
                        .template operator()<false, false, PvGemmEpilogueType::None, false>(
                            kv_idx, kv_idx + T::kBlockN);
                }
                // Skip + epilogue on the global last tile.
                if(partial_qo_loc < 0)
                {
                    mla_main.template operator()<false,
                                                 true,
                                                 PvGemmEpilogueType::OutputFinal,
                                                 false>(kv_idx + T::kBlockN, kv_end);
                }
                else
                {
                    mla_main.template operator()<false,
                                                 true,
                                                 PvGemmEpilogueType::OutputSplit,
                                                 false>(kv_idx + T::kBlockN, kv_end);
                }
            }
        }

        (void)qo_end;
        (void)out_br;
        (void)split_out_br;
    }
}
#else
template <typename T>
__global__ __launch_bounds__(T::kNumThreads, T::kOccupancy) void
kn_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16(HkMlaV40DecodeFwdParams<T> params)
{
    (void)params;
    assert(false);
}
#endif

template <typename Traits>
void mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16(aiter_tensor_t& query,
                                                    aiter_tensor_t& query_rope,
                                                    aiter_tensor_t& kv_buffer,
                                                    aiter_tensor_t& kv_buffer_rope,
                                                    const aiter_tensor_t& qo_indptr,
                                                    const aiter_tensor_t& kv_indptr,
                                                    const aiter_tensor_t& kv_page_indices,
                                                    const aiter_tensor_t& kv_last_page_lens,
                                                    const aiter_tensor_t& work_indptr,
                                                    const aiter_tensor_t& work_info_set,
                                                    const int max_seqlen_q,
                                                    const float softmax_scale,
                                                    aiter_tensor_t& split_output,
                                                    aiter_tensor_t& split_lse,
                                                    aiter_tensor_t& final_output)
{
    const int32_t num_qheads = query.size(1);
    AITER_CHECK((num_qheads & (num_qheads - 1)) == 0 && num_qheads >= 16 && num_qheads <= 128,
                "num_qheads must be a power of 2 in [16, 128], got ",
                num_qheads);
    AITER_CHECK(num_qheads * max_seqlen_q == Traits::kBlockM,
                "num_qheads * max_seqlen_q must equal ",
                Traits::kBlockM,
                ", got ",
                num_qheads,
                " * ",
                max_seqlen_q,
                " = ",
                num_qheads * max_seqlen_q);
    const int32_t log2_num_qheads = __builtin_ctz(num_qheads);

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const hipStream_t stream = aiter::getCurrentHIPStream();

    HkMlaV40DecodeFwdParams<Traits> params = {
        hk::make_gl<typename Traits::gl_q_nope>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(query.data_ptr())),
            query.size(0),
            num_qheads / Traits::kTileM,
            Traits::kTileM,
            Traits::kQkPackedNopeQElems),
        hk::make_gl<typename Traits::gl_q_rope>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(query_rope.data_ptr())),
            query_rope.size(0),
            num_qheads / Traits::kTileM,
            Traits::kTileM,
            Traits::kQkRopeHeadDim),
        hk::make_gl<typename Traits::gl_kv_nope>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kv_buffer.data_ptr())),
            kv_buffer.size(0),
            Traits::kPageSize,
            Traits::kKvNumHead,
            Traits::kQkPackedNopeKvElems),
        hk::make_gl<typename Traits::gl_kv_rope>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kv_buffer_rope.data_ptr())),
            kv_buffer_rope.size(0),
            Traits::kPageSize,
            Traits::kKvNumHead,
            Traits::kQkRopeHeadDim),
        // kv_indices
        reinterpret_cast<int32_t*>(kv_page_indices.data_ptr()),
        // kv_last_page_lens (only read by kernel when kPageSize > 1)
        reinterpret_cast<int32_t*>(kv_last_page_lens.data_ptr()),
        // metadata
        reinterpret_cast<int32_t*>(work_indptr.data_ptr()),
        reinterpret_cast<int32_t*>(work_info_set.data_ptr()),
        hk::make_gl<typename Traits::gl_o>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(final_output.data_ptr())),
            1,
            final_output.size(0),
            Traits::kBlockM,
            Traits::kVoHeadDim),
        hk::make_gl<typename Traits::gl_so>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(split_output.data_ptr())),
            1,
            split_output.size(0),
            Traits::kBlockM,
            Traits::kVoHeadDim),
        hk::make_gl<typename Traits::gl_slse>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(split_lse.data_ptr())),
            1,
            split_lse.size(0),
            Traits::kBlockM,
            1),
        // parameters
        softmax_scale,
        log2_num_qheads};

    const dim3 grid        = dim3(dev_prop.multiProcessorCount);
    const int32_t lds_size = dev_prop.maxSharedMemoryPerMultiProcessor / Traits::kOccupancy;

    kn_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16<Traits>
        <<<grid, Traits::kNumThreads, lds_size, stream>>>(params);
}

void hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16(aiter_tensor_t& query,
                                                       aiter_tensor_t& query_rope,
                                                       aiter_tensor_t& kv_buffer,
                                                       aiter_tensor_t& kv_buffer_rope,
                                                       const aiter_tensor_t& qo_indptr,
                                                       const aiter_tensor_t& kv_indptr,
                                                       const aiter_tensor_t& kv_page_indices,
                                                       const aiter_tensor_t& kv_last_page_lens,
                                                       const aiter_tensor_t& work_indptr,
                                                       const aiter_tensor_t& work_info_set,
                                                       const int max_seqlen_q,
                                                       const float softmax_scale,
                                                       aiter_tensor_t& split_output,
                                                       aiter_tensor_t& split_lse,
                                                       aiter_tensor_t& final_output)
{
    HipDeviceGuard device_guard(final_output.device_id);

    const bool q_nope_is_fp8  = (query.dtype() == AITER_DTYPE_fp8);
    const bool kv_nope_is_fp8 = (kv_buffer.dtype() == AITER_DTYPE_fp8);
    const bool q_rope_is_bf16  = (query_rope.dtype() == AITER_DTYPE_bf16);
    const bool kv_rope_is_bf16 = (kv_buffer_rope.dtype() == AITER_DTYPE_bf16);

    AITER_CHECK(q_nope_is_fp8 && kv_nope_is_fp8,
                "hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16 requires FP8 NOPE; got q=",
                AiterDtype_to_str(query.dtype()),
                ", kv=",
                AiterDtype_to_str(kv_buffer.dtype()));
    AITER_CHECK(q_rope_is_bf16 && kv_rope_is_bf16,
                "hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16 requires BF16 ROPE; got q_rope=",
                AiterDtype_to_str(query_rope.dtype()),
                ", kv_rope=",
                AiterDtype_to_str(kv_buffer_rope.dtype()));

    const int32_t page_size = kv_buffer.size(1);

#define DISPATCH_PAGE_SIZE(PageSize)                                                 \
    case PageSize: {                                                                 \
        using Traits = HkMlaV40DecodeFwdTraits<hk::fp8e4m3,                          \
                                               hk::bf16,                             \
                                               hk::fp8e4m3,                          \
                                               hk::bf16,                             \
                                               hk::bf16,                             \
                                               /*kBlockN_=*/32,                      \
                                               /*kNumWarps_=*/8,                     \
                                               /*kOccupancy_=*/1,                    \
                                               /*kBlockM_=*/128,                     \
                                               /*kPageSize_=*/PageSize>;             \
        mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16<Traits>(query,                \
                                                              query_rope,            \
                                                              kv_buffer,             \
                                                              kv_buffer_rope,        \
                                                              qo_indptr,             \
                                                              kv_indptr,             \
                                                              kv_page_indices,       \
                                                              kv_last_page_lens,     \
                                                              work_indptr,           \
                                                              work_info_set,         \
                                                              max_seqlen_q,          \
                                                              softmax_scale,         \
                                                              split_output,          \
                                                              split_lse,             \
                                                              final_output);         \
        break;                                                                       \
    }

    // Only page_size in {1, 64} are instantiated -- same pattern as v32.
    switch(page_size)
    {
        DISPATCH_PAGE_SIZE(1)
        DISPATCH_PAGE_SIZE(64)
    default:
        AITER_CHECK(false,
                    "hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16: unsupported page_size ",
                    page_size,
                    " (supported: 1, 64).");
    }

#undef DISPATCH_PAGE_SIZE
}
