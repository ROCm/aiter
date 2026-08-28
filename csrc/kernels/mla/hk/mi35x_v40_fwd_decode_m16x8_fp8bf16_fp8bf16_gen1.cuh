// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "hk_mla_softmax.cuh"
#include "hk_mla_v40_buffer_managers_gen1.cuh"
#include "mla.h"
#include <assert.h>
#include <limits>
#include <optional>

using namespace hk_mla;

// Toggle the slim dispatch ladder (fewer mla_main instantiations, always
// boundary-checked prefetch). Comment out to fall back to the full ladder.
#define MLA_SLIM_DISPATCH 1

// Warp-index group. PV is always at call end for every warp; the only
// compile-time distinction that matters is whether the warp is a RoPE owner
// (warps 5,7) -- those run the buffer_load_lds RoPE prefetch path. The NoPE
// split (Lo/Hi) is vestigial but kept so the per-warp-type dispatch and the
// SIMD-pairing comments stay stable.
enum class WarpTypeM16x8 : uint8_t
{
    LoNoPEWarp, // warps 0-3: tile 0 (cols 0-255), pure NoPE
    HiRoPEWarp  // warps 4-7: tile 1 (cols 256-511), NoPE + RoPE
};

#include "hk_mla_v40_fwd_decode_gen1_common.cuh"

// V4.0 mi35x m16x8 decode kernel: separate FP8 NOPE + BF16 ROPE buffers for
// both Q and KV. End-to-end body (Phases 4a..4g) in place: prologue (Q load +
// first KV tile) -> per-warp dispatch ladder over mla_main (QK GEMM + softmax
// + PV GEMM + epilogue, with online-softmax rescale across K-tile iters).
#if defined(__gfx950__)
template <WarpTypeM16x8 kWarpType, typename T>
__device__ void
mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1_impl(HkMlaV40DecodeFwdParams<T> params,
                                                         const uint32_t warp_idx)
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

    // Compile-time warp type (chosen by the __global__ wrapper's entry divergence).
    constexpr bool kIsRopeWarp = (kWarpType == WarpTypeM16x8::HiRoPEWarp);

    // Deferred-PV grouping: defers its PV by one tile if false;
    constexpr bool kPvAtEnd = (kWarpType == WarpTypeM16x8::LoNoPEWarp);

    // Run softmax WITHOUT packed-ALU (v_pk_*) ops if false
    constexpr bool kSoftmaxUsePk      = kPvAtEnd;
    constexpr bool kFinalRescaleUsePk = (kPvAtEnd == false);

    constexpr comp_t log2e = 1.4426950408889634;

    const int32_t worker_idx     = blockIdx.x;
    const int32_t work_start_idx = __builtin_amdgcn_readfirstlane(params.p_work_indptr[worker_idx]);
    const int32_t work_end_idx =
        __builtin_amdgcn_readfirstlane(params.p_work_indptr[worker_idx + 1]);
    if(work_start_idx >= work_end_idx)
    {
        return;
    }

    // ---- VGPR layout (per-lane) ----
    // Compiler scratch constrained to v0..v43 (budget=44 on the __global__).
    // Full-Q-VGPR map (chunks 0-4 populated this step):
    //   255:128 oaccu
    //   127: 64 q_vgpr (full Q; v64..v103 = chunks 0-4, v104..v111 = q_lds scratch)
    //    63: 56 p_comp (60:63 = HI half -> PV V tile v_0); 56:59 p_mfma overlay
    //    55: 52 k_0    (also PV V tile v_1)
    //    51: 48 k_1
    //    47: 44 k_2
    // Register layout single source of truth (see HkMlaV40Regs at file scope).
    // Re-alias the constants/ranges locally so the body below keeps its existing
    // k_* / *_ranges names; both orchestrators + every stage bind the same regs.
    using R                            = HkMlaV40Regs<T>;
    constexpr uint32_t mfma_tile_sz    = R::mfma_tile_sz;
    constexpr uint32_t k_o_begin       = R::k_o_begin;
    constexpr uint32_t k_o_end         = R::k_o_end;
    constexpr uint32_t k_p_comp_begin  = R::k_p_comp_begin;
    constexpr uint32_t k_p_comp_end    = R::k_p_comp_end;
    constexpr uint32_t k_p_mfma_begin  = R::k_p_mfma_begin;
    constexpr uint32_t k_p_mfma_end    = R::k_p_mfma_end;
    constexpr uint32_t k_v0_begin      = R::k_v0_begin;
    constexpr uint32_t k_v0_end        = R::k_v0_end;
    constexpr uint32_t k_q_vgpr_begin  = R::k_q_vgpr_begin;
    constexpr uint32_t k_q_vgpr_end    = R::k_q_vgpr_end;
    constexpr uint32_t k_k0_begin      = R::k_k0_begin;
    constexpr uint32_t k_k1_begin      = R::k_k1_begin;
    constexpr uint32_t k_k2_begin      = R::k_k2_begin;
    constexpr uint32_t k_q_lds_1_begin = R::k_q_lds_1_begin;
    constexpr uint32_t k_q_lds_0_begin = R::k_q_lds_0_begin;
    constexpr uint32_t k_q_lds_begin   = R::k_q_lds_begin;
    (void)mfma_tile_sz;
    (void)k_o_end;
    (void)k_p_comp_end;
    (void)k_p_mfma_end;
    (void)k_v0_end;
    (void)k_q_vgpr_end;
    (void)k_k2_begin;
    (void)k_q_lds_1_begin;
    (void)k_q_lds_0_begin;

    using q_vgpr_ranges     = typename R::q_vgpr_ranges;
    using p_comp_ranges     = typename R::p_comp_ranges;
    using p_comp_lo_ranges  = typename R::p_comp_lo_ranges;
    using p_comp_hi_ranges  = typename R::p_comp_hi_ranges;
    using kv_top_ranges     = typename R::kv_top_ranges;
    using kv_bot_ranges     = typename R::kv_bot_ranges;
    using kv_alt_top_ranges = typename R::kv_alt_top_ranges;
    using p_mfma_ranges     = typename R::p_mfma_ranges;
    using o_ranges          = typename R::o_ranges;
    using pv_v_0_ranges     = typename R::pv_v_0_ranges;
    using pv_v_1_ranges     = typename R::pv_v_1_ranges;
    using pv_v_2_ranges     = typename R::pv_v_2_ranges;
    using q_lds_ranges      = typename R::q_lds_ranges;

    hkdart::clobber<q_vgpr_ranges>();
    hkdart::clobber<p_comp_ranges>();
    hkdart::clobber<p_mfma_ranges>();
    hkdart::clobber<o_ranges>();
    hkdart::clobber<q_lds_ranges>();

    // ---- Managers ----
    QManager8to16bitsV1<T> q_manager;
    KvManager8to16bitsV2<T> kv_manager;
    OManager16bitsV4Gen1Swizzle<T, out_t> o_manager;
    OManager32bitsV4Gen1Swizzle<T, split_t> split_o_manager;

    // ---- art tile declarations (bound to the shared register ranges) ----
    typename R::q_vgpr_t q_vgpr;
    typename R::p_comp_t p_comp;
    typename R::p_comp_lo_t p_comp_lo;     // sub-tile A, N-cols 0:16
    typename R::p_comp_hi_t p_comp_hi;     // sub-tile A, N-cols 16:32
    typename R::p_comp_b_lo_t p_comp_b_lo; // sub-tile B, N-cols 32:48
    typename R::p_comp_b_hi_t p_comp_b_hi; // sub-tile B, N-cols 48:64
    typename R::k_0_t k_0;
    typename R::k_1_t k_1;
    typename R::k_2_t k_2;
    typename R::pv_v_0_t pv_v_0;
    typename R::pv_v_1_t pv_v_1;
    typename R::pv_v_2_t pv_v_2;
    typename R::p_mfma_t p_mfma;
    typename R::oaccu_t oaccu;

    // ---- Runtime constants ----
    const uint32_t lane_idx = opus::lane_id();

    // Causal mask: compute per-warp kv_end offset for MTP.
    // num_wave_group = qseqlen = kBlockM / num_qheads
    // waves_per_head = num_qheads / kTileM
    // causal_offset = num_wave_group - 1 - (warp_idx / waves_per_head)
    const int32_t log2_num_qheads = __builtin_amdgcn_readfirstlane(params.log2_num_qheads);
    const int32_t num_qheads      = 1 << log2_num_qheads;

    // Per-lane attention sink logit. Loaded once at kernel entry: it depends
    // only on (warp_idx, lane_idx), not on work_idx, so it lives in a VGPR
    // for the kernel's lifetime. When p_attn_sink is null, substitute -inf
    // so exp(sink - row_max) = 0 -> the epilogue's row_sum_e += sink_term
    // becomes a no-op. num_qheads is a power of 2 in {16,32,64,128} (see
    // outer wrapper check).
    const uint32_t head_idx =
        (warp_idx * 16u + (lane_idx & 15u)) & (static_cast<uint32_t>(num_qheads) - 1u);
    const float attn_sink = (params.p_attn_sink == nullptr)
                                ? -std::numeric_limits<float>::infinity()
                                : params.p_attn_sink[head_idx];

    const int32_t num_wave_group      = T::kBlockM >> log2_num_qheads; // qseqlen
    const int32_t log2_waves_per_head = log2_num_qheads - 4;           // log2(kTileM) = 4
    const int32_t qpos_off_from_last  = num_wave_group - 1 - (warp_idx >> log2_waves_per_head);

    // ---- LDS layout ----
    //
    // p_lds_kv_curr/   : 32 KB each (32 rows * 512 bf16 cols, 2 pongs).
    //  p_lds_kv_next     Placed FIRST so they cover the +0 LDS base.
    // O bounce         : overlays p_lds_kv_next (the next pong is DEAD on the
    //                    global last iter, where the epilogue runs, since the
    //                    swap is a no-op). Per-warp strides differ between
    //                    QManager and OManager V3 (2112 B bf16 / 4352 B fp32),
    //                    so placing the O bounce inside p_lds_q creates
    //                    cross-warp aliasing with the next work_idx's load_q --
    //                    racy when a fast warp's load_q lands while a slow warp's
    //                    epilogue is still in flight. Overlaying KV-next instead
    //                    keeps the O bounce in a region whose next consumer (next
    //                    work_idx's KV prologue) writes to p_lds_kv_curr, not next.
    // p_lds_q          : 16 KB - QManager region (NoPE all in VGPR -> the final
    //                    LDS holds RoPE only, reusing the staging bytes). Placed
    //                    AFTER both KV pongs + max(O, KV) so the O bounce never
    //                    overlaps Q, and so warp 0's Phase-1 staging (at p_lds_q
    //                    + 0) starts well above 0 in m0 -- this lets
    //                    p1_vmem_to_staging_chunk pre-subtract up to 192 B
    //                    (kColInRecord = 0/64/128/192) from the LDS dst without
    //                    m0 underflowing mod 2^32.
    //
    // Total (occupancy=1): KvLds + max(KvLds, O) + 16 KB Q.
    extern __shared__ int32_t p_lds[];

    // opus::max is device-only / non-constexpr; use inline ternary in constexpr
    // contexts.
    constexpr uint32_t kSzLdsQ  = q_manager.get_lds_size_in_byte();
    constexpr uint32_t kSzLdsKv = kv_manager.get_lds_size_in_byte();
    constexpr uint32_t kSzLdsO =
        (o_manager.get_lds_size_in_byte() > split_o_manager.get_lds_size_in_byte())
            ? o_manager.get_lds_size_in_byte()
            : split_o_manager.get_lds_size_in_byte();

    // Sub-tile B raw-fp8 staging (kBlockN=64): 32 rows x 512 packed bytes = 16 KB.
    constexpr uint32_t kSzLdsStage =
        (T::kBlockN == 64) ? (32u * static_cast<uint32_t>(T::kQkPackedNopeBytes)) : 0u;
    static_assert(kSzLdsQ + kSzLdsKv + (kSzLdsO > kSzLdsKv ? kSzLdsO : kSzLdsKv) + kSzLdsStage <=
                      160u * 1024u,
                  "V4.0 LDS budget exceeds 160 KB at kOccupancy=1.");
    // QManager pre-subtracts up to kLdsHeadPadBytes from p_lds_q in
    // p1_vmem_to_staging_chunk. Placing Q after both KV pongs gives that
    // subtraction enough headroom (m0 lands in KV-pong region, still valid LDS).
    static_assert(kSzLdsKv + (kSzLdsO > kSzLdsKv ? kSzLdsO : kSzLdsKv) >=
                      QManager8to16bitsV1<T>::kLdsHeadPadBytes,
                  "KV pongs must precede Q LDS with enough bytes to absorb the "
                  "QManager P1 pre-subtract.");

    uintptr_t p_lds_kv_curr        = reinterpret_cast<uintptr_t>(p_lds);
    uintptr_t p_lds_kv_next        = p_lds_kv_curr + kSzLdsKv;
    const uintptr_t p_lds_q        = p_lds_kv_next + (kSzLdsO > kSzLdsKv ? kSzLdsO : kSzLdsKv);
    const uintptr_t p_lds_kv_stage = p_lds_q + kSzLdsQ; // B raw-fp8 staging

    // ---- Work loop ----
    // Phase 4b is in place: per work item, read work_info, resolve kv extents,
    // load Q (vmem -> VGPR + bf16 LDS), and prefetch+cvt+store the first KV
    // tile into the curr pong. The mla_main lambda + dispatch ladder still TODO
    // (Phases 4c-4f); kernel still hits assert(false) at the bottom of the loop.
    const uint32_t kv_ld_row_base_idx = kv_manager.get_kv_ld_row_base_idx(warp_idx);

    for(int32_t work_idx = work_start_idx; work_idx < work_end_idx; ++work_idx)
    {
        const int32_t batch_idx = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 0]);
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

        // "Last split of this batch" -- the planner sets
        // kv_offset = curr_kv_end - work_info.kv_end (metadata/v1_2_device.cuh
        // L202/L214), so kv_offset == 0 iff this split's kv_end coincides
        // with the batch tail. Used by the epilogue sink fold: only the
        // LAST split inflates row_sum_e with the sink term, so the reducer
        // routes the sink contribution into the global denominator exactly
        // once. Last-vs-first is mathematically equivalent (reducer combines
        // lses commutatively); last-split is cheaper -- no extra kv_indptr
        // load (kv_offset is already in scope above).
        const bool is_last_split = (kv_offset == 0);

        // Convert work_info page bounds to TOKEN space. When kPageSize == 1
        // pages == tokens. When kPageSize > 1 and this is the batch tail
        // (kv_offset == 0), clip the last page with kv_last_page_lens[batch].
        // The (kPageSize == 1) check folds at compile time so the load is
        // dead-code-eliminated for kPageSize == 1.
        const int32_t kv_start = kv_start_page * T::kPageSize;
        int32_t kv_end;
        if((T::kPageSize == 1) || (kv_offset != 0))
        {
            kv_end = kv_end_page * T::kPageSize;
        }
        else
        {
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
        // rescale / do_rescale are produced by each tile's softmax and consumed
        // by its PV at call end (same call), so these could be call-local; kept
        // at work-item scope for stable codegen.
        comp_t rescale  = 1.0f;
        bool do_rescale = false;

        // Cross-iter deferred strip-3 (lo warps only): Phase A stages strip 3's NoPE
        // into private LDS + its e8m0 scale into this carried VGPR (an async load that
        // lands here). The consume (ds_read staging + cvt + store into the KV pong) is
        // deferred to the NEXT mla_main call's top -- spreading the cvt work off the
        // busy Phase B. Every non-first lo call runs it (its predecessor always staged
        // strip 3); the first call's tile-0 strip 3 came from the prologue.
        uint32_t s3_scale = 0u;

        // Helper: resolve the physical KV row for the 32-row tile that begins
        // at tile_start. Returns -1 if the tile is entirely OOB.
        //
        // ALERT: this call issues EITHER one buffer_load OR none (the -1 path,
        // and get_kv_ld_row's own per-warp boundary skip). Its in-flight load
        // count is therefore data/lane-dependent, so any s_waitcnt vmcnt placed
        // where this load is still outstanding cannot account for it with a
        // static count -- only issue resolve AFTER the relevant vmcnt waits.
        // (Issuing it before the cvt+store waits was the resolve-front bug.)
        // Resolves ONE 32-row KV sub-tile (kBlockN=64 = two 32-row sub-tiles A,B;
        // kv_ld_row_base_idx is the lane's row in a 32-row tile). Returns -1 if the
        // sub-tile is entirely OOB.
        auto resolve_row_kv_ld = [&](const int32_t tile_start) -> int32_t {
            const int32_t tile_end = tile_start + 16;
            int32_t row_kv_ld;
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

        // Bytes of one 32-row KV sub-tile in a pong (sub-tile B lives at +kSubPong).
        constexpr uint32_t kSubPong = 32u * T::kQkHeadDim * sizeof(hk::bf16); // 32768

        // V2 band remap: each warp owns ONE 16-row band of the 64-row tile.
        //   band_off = this warp's band token offset (0/16/32/48)
        //   sub_off  = its 32-row sub-tile's LDS base offset (A=0, B=kSubPong)
        // -> ONE resolved KV row per lane (no A/B pair).
        const int32_t band_off  = static_cast<int32_t>((warp_idx & 3u) * 16u);
        const uintptr_t sub_off = static_cast<uintptr_t>(((warp_idx >> 1) & 1u)) * kSubPong;

        // Tile 0's band rows go to the prologue; the next 64-tile's band rows seed the
        // first lambda call. `row_kv_ld_next_next` is a one-deep carry.
        const int32_t row_kv_ld_first = resolve_row_kv_ld(kv_start + band_off);
        int32_t row_kv_ld_next_next   = resolve_row_kv_ld(kv_start + T::kBlockN + band_off);

        // Load Q: Q[:, 0:256] -> VGPR pinned at k_q_vgpr_begin (32 vgprs/lane).
        //         Q[:, 256:512] -> bf16 final LDS region inside p_lds_q.
        // Q rope/nope buffers are separate tensors in V4.0.
        // Fold the softmax temperature AND the natural->log2 conversion into Q
        // once here, so the per-KV-tile softmax drops both its sm_scale multiply
        // (softmax_scale_p -> softmax_mask_p) and its log2e multiply
        // (softmax_p1 -> softmax_p1_prescaled). Scores then arrive in log2 units.
        const float q_scale_log2 = params.softmax_scale * static_cast<float>(log2e);
        q_manager.template load_q<k_q_vgpr_begin>(params.p_query,
                                                  params.p_query_rope,
                                                  num_qheads,
                                                  warp_idx,
                                                  qo_start,
                                                  p_lds_q,
                                                  q_scale_log2);
        __builtin_amdgcn_sched_barrier(0);

        // ---- RoPE (col-tiles 14,15) -> pinned q_vgpr (v120:127), prologue read ----
        // load_q staged RoPE into Q-LDS col-tiles 0,1; read it ONCE into q_vgpr so
        // the QK loop is all-VGPR. lgkmcnt drain orders the stage's ds_writes here.
        if constexpr(R::kRopeInVgpr)
        {
            __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/-1));
            __builtin_amdgcn_sched_barrier(0);
            constexpr uint32_t q_rope_0_base = k_q_vgpr_begin + 14u * 4u; // 120
            constexpr uint32_t q_rope_1_base = k_q_vgpr_begin + 15u * 4u; // 124
            using q_rope_0_range             = hkdart::split_many_t<
                hkdart::type_list<hkdart::range<q_rope_0_base, q_rope_0_base + 3u>>,
                4>;
            using q_rope_1_range = hkdart::split_many_t<
                hkdart::type_list<hkdart::range<q_rope_1_base, q_rope_1_base + 3u>>,
                4>;
            hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, q_rope_0_range>
                q_rope_0;
            hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, q_rope_1_range>
                q_rope_1;
            q_manager.template load_q_lds_to_gpr<QManager8to16bitsV1<T>::kRopeColTileLo>(
                q_rope_0, p_lds_q, warp_idx);
            q_manager.template load_q_lds_to_gpr<QManager8to16bitsV1<T>::kRopeColTileHi>(
                q_rope_1, p_lds_q, warp_idx);
            __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/-1));
            __builtin_amdgcn_sched_barrier(0);
        }

        // Prologue: prefetch + cvt+store this warp's 16x256 band of the first 64-row
        // KV tile into its sub-tile base (curr + sub_off). One boundary check on the
        // band's own rows (row_kv_ld_first == -1 when the whole band is past kv_len).
        // kCheckBoundary=true always (cold path).
        kv_manager.template async_load_k<true, kIsRopeWarp>(p_lds_kv_curr + sub_off,
                                                            warp_idx,
                                                            params.p_kv_buffer,
                                                            params.p_kv_buffer_rope,
                                                            row_kv_ld_first);

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

            // Deferred PV: a PV of the PREVIOUS tile runs at call start (block
            // after Phase A) for every Lo call with a not-yet-PV'd prior tile --
            // all but the warp's first compute call and the fully-idle skip call.
            constexpr bool kHasPv =
                (!kPvAtEnd) && (!kIsFirstIter) && ((!kSkipCompute) || kDoEpilogue);
            // Lo's deferred first PV takes the accumulate path (the 3-arg init
            // mfma is the Hi/at-end path), so Lo zeroes oaccu once on its first
            // compute call. row_max/row_sum_e are still initialised by the
            // kIsFirstIter softmax below (shared with Hi).
            if constexpr((kPvAtEnd == false) && kIsFirstIter)
            {
                hk::zero(oaccu);
            }

            static_assert((kSkipCompute == false) || (kIsFirstIter == false),
                          "A skipped iter cannot be the warp's first compute iter.");
            static_assert(
                (kIsGlobalLast == false) || (kCheckBoundaryNext == false),
                "kIsGlobalLast == true means no next tile, so kCheckBoundaryNext must be false.");

            // Snapshot this warp's NEXT-tile band KV row (set by prior call/prologue).
            int32_t row_kv_ld_next = 0;
            if constexpr(kIsGlobalLast == false)
            {
                row_kv_ld_next = row_kv_ld_next_next;
                // Force the (already-resolved) index into a stable VGPR so the compiler
                // does NOT rematerialise the get_kv_ld_row buffer_load into the prefetch
                // address path -- that rematerialisation makes the prefetch address
                // depend on a just-issued load, forcing an s_waitcnt vmcnt(0) that drains
                // all in-flight vmem. With it held, this iter's index load is long done.
                asm volatile("" : "+v"(row_kv_ld_next));

                // Lo warps use 2x PV GEMM to hide latency of loading next-next rows.
                if constexpr(kWarpType == WarpTypeM16x8::LoNoPEWarp)
                {
                    row_kv_ld_next_next =
                        resolve_row_kv_ld(kv_tile_start + 2 * T::kBlockN + band_off); // +128
                }
            }

            // ---- Phase A: prefetch this warp's NEXT-tile band into the next-pong ----
            // V2 band: 4 col-strips of 16x64. strips 0,1 -> VGPR carriers (hidden under
            // QK, 2-carrier ceiling); strip 2 (+ strip 3 for lo) -> staging LDS; tile-1
            // strip 3 (RoPE) DMAs to LDS in Phase B. Single row index (row_kv_ld_next).
            constexpr uint32_t kTile = kIsRopeWarp ? 1u : 0u;
            typename KvManager8to16bitsV2<T>::KvTilePrefetch p0, p1;
            // Staging: buffer_load_lds dst is wave-uniform m0; HW adds laneId*16. Two
            // slots (strip 2 @ +0, strip 3 @ +kStageTileBytes), each per-warp base
            // p_lds_kv_stage + slot*8192 + warp*1024. Read matches at the same base.
            constexpr uint32_t kStageWarpBytes = 64u * 16u;            // 1024
            constexpr uint32_t kStageTileBytes = 8u * kStageWarpBytes; // 8192
            const uintptr_t stage_t0           = p_lds_kv_stage + warp_idx * kStageWarpBytes;
            const uintptr_t stage_t1 =
                p_lds_kv_stage + kStageTileBytes + warp_idx * kStageWarpBytes;
            uint32_t scale_s0 = 0u;

            // ---- Deferred strip-3 consume (lo, non-first calls) ----
            // The PREVIOUS call staged strip 3 (NoPE in slot 1, scale in s3_scale).
            // Consume it now -- ds_read staging + cvt + store into THIS tile's KV pong
            // (p_lds_kv_curr, already swapped in) -- so it completes strips 0-3 before
            // this iter's QK. Runs even on the last/skip iter (still filling curr KV).
            // Must precede Phase A, which re-stages slot 1 for the next tile.
            if constexpr((!kIsRopeWarp) && (!kIsFirstIter))
            {
                const uintptr_t curr_sub = p_lds_kv_curr + sub_off;
                hk::u32x4 dw3, dw4;
                // skip resolve_row_kv_ld
                kv_manager.template wait_kv_loads<false, /*kVmCnt=*/1>(warp_idx);
                const hk::u32x4 s3 = kv_manager.template load_staged_kv_carrier<1u>(stage_t0);
                const float sf3    = hk_mla::e8m0_to_f32(s3_scale);
                __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/-1));
                kv_manager.template cvt_kv_tile_step<0>(dw3, s3, sf3);
                kv_manager.template cvt_kv_tile_step<1>(dw3, s3, sf3);
                kv_manager.template cvt_kv_tile_step<2>(dw4, s3, sf3);
                kv_manager.template cvt_kv_tile_step<3>(dw4, s3, sf3);
                kv_manager.template store_kv_tile_step<3u, 0u, 0>(curr_sub, warp_idx, dw3);
                kv_manager.template store_kv_tile_step<3u, 0u, 1>(curr_sub, warp_idx, dw4);
            }

            // Issue NoPE carriers + staging before the barrier so their vmem latency
            // overlaps the barrier wait. NoPE carriers are VGPR-landing; staging DMAs
            // to the (private) stage LDS -- both safe ahead of the p_lds_kv_next barrier.
            if constexpr(kIsGlobalLast == false)
            {
                kv_manager.template prefetch_kv_nope<0u, kTile, kCheckBoundaryNext, kIsRopeWarp>(
                    warp_idx, params.p_kv_buffer, row_kv_ld_next, p0);
                kv_manager.template prefetch_kv_nope<1u, kTile, kCheckBoundaryNext, kIsRopeWarp>(
                    warp_idx, params.p_kv_buffer, row_kv_ld_next, p1);

                kv_manager
                    .template prefetch_kv_nope_lds<2u, kTile, kCheckBoundaryNext, kIsRopeWarp>(
                        warp_idx, params.p_kv_buffer, row_kv_ld_next, stage_t0, scale_s0);
                // Strip 3 (lo only): stage NoPE into slot 1; its scale lands in the
                // carried s3_scale. The consume is deferred to the NEXT call's top.
                if constexpr(!kIsRopeWarp)
                    kv_manager
                        .template prefetch_kv_nope_lds<3u, kTile, kCheckBoundaryNext, kIsRopeWarp>(
                            warp_idx, params.p_kv_buffer, row_kv_ld_next, stage_t1, s3_scale);
            }

            // ---- Lo deferred PV (of the PREVIOUS tile) ----
            // Issued after this call's prefetch so its MFMAs + V ds_reads overlap
            // the prefetch vmem latency and the barrier wait. Reads the previous
            // tile's V from p_lds_kv_next (still alive -- this call's cvt+store has
            // not overwritten it yet) and the previous tile's p_mfma / rescale /
            // do_rescale (pinned regs + carried scalars). Accumulate path only
            // (oaccu pre-zeroed on the first iter above). Lo warps only.
            // Deferred PV of the PREVIOUS 64-tile: sub-tile A (rescale folded) then B
            // (rescale already applied by A -> false). A from next[0], B from next[+kSubPong].
            if constexpr(kHasPv)
            {
                // One PV call contracts BOTH KV sub-tiles (kBlockN=64) -> single prologue.
                hk_mla_v40_pv_stage<false, T>(kv_manager, p_lds_kv_next, rescale, do_rescale);
            }

            __builtin_amdgcn_s_setprio(3);

            // Keep this warp's band loads in flight across the barrier + QK (hidden).
            // Lo: 2 carriers (dwx4+ubyte=4) + 2 staged (lds+ubyte=4) = 8.
            // Hi: 2 carriers (4) + 1 staged (2) = 6 (strip-3 RoPE issued in Phase B).
            if constexpr(kIsRopeWarp)
            {
                __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/6));
            }
            else
            {
                __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/9));
            }
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // ---- QK GEMM (simple generic full-drain; correctness-first) ----
            // One mfma-pair per global col-tile gct in [0,16): K both row-halves
            // from KV-LDS, Q from q_vgpr (gct < kQkGemmTiles) or from Q-LDS
            // (gct >= kQkGemmTiles). Full lgkmcnt(0) drain before each pair -> no
            // round-robin / wait tuning. kQkGemmTiles is the only migration knob.
            constexpr uint32_t kQkGemmTiles = R::kQkGemmTiles;
            if constexpr(kSkipCompute == false)
            {
                constexpr uint32_t kBK             = T::kBlockK;
                constexpr uint32_t kNumColTilesAll = T::kQkHeadDim / T::kBlockK; // 16
                // All 16 col-tiles of Q are in VGPR (kQkGemmTiles==16), so every
                // QK A-operand is read straight from the pinned q_vgpr block and
                // lgkmcnt counts ONLY the K ds_reads below.
                static_assert(kQkGemmTiles == kNumColTilesAll,
                              "QK loop assumes all col-tiles in VGPR (kQkGemmTiles==16).");

                // 3-deep software-pipelined QK over the 64 K sub-tiles (16 col-tiles
                // x 4 row-groups). Sub-tile S -> col c=S/4, row-group rg=S%4:
                //   rg 0,1 = sub-tile A (rows 0:16, 16:32) from curr[0]     -> p_comp_lo/hi
                //   rg 2,3 = sub-tile B (rows 0:16, 16:32) from curr[+kSubPong] -> p_comp_b_lo/hi
                // ds_read_b128 K into a 3-slot ring (k_0/k_1/k_2), mma vs q_vgpr[c].
                // Init (3-arg) the first touch of each of the 4 N-groups (S<4, c==0).
                constexpr uint32_t kNumSub = 4u * kNumColTilesAll; // 64
                constexpr uint32_t kDepth  = 3u;

                auto k_ring_base = [](uint32_t s) constexpr -> uint32_t {
                    return (s % 3u == 0u)   ? R::k_k0_begin
                           : (s % 3u == 1u) ? R::k_k1_begin
                                            : R::k_k2_begin;
                };
                auto issue_k = [&]<uint32_t S>() {
                    constexpr uint32_t c    = S / 4u;
                    constexpr uint32_t rg   = S % 4u;
                    constexpr uint32_t base = k_ring_base(S);
                    using kr =
                        hkdart::split_many_t<hkdart::type_list<hkdart::range<base, base + 3u>>, 4>;
                    hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, kr> kt;
                    // kRowOffset = rg*16 (0/16 = sub-tile A, 32/48 = sub-tile B); the
                    // sub-tile offset is folded into load_k_to_gpr's ds_read imm, so the
                    // base ptr is always p_lds_kv_curr (no +kSubPong address VGPR).
                    kv_manager.template load_k_to_gpr<rg * 16u, c * kBK>(kt, p_lds_kv_curr);
                };
                auto mma_k = [&]<uint32_t S>() {
                    constexpr uint32_t c    = S / 4u;
                    constexpr uint32_t rg   = S % 4u;
                    constexpr uint32_t base = k_ring_base(S);
                    using kr =
                        hkdart::split_many_t<hkdart::type_list<hkdart::range<base, base + 3u>>, 4>;
                    hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, kr> kt;
                    constexpr uint32_t qb = k_q_vgpr_begin + c * 4u;
                    using qr =
                        hkdart::split_many_t<hkdart::type_list<hkdart::range<qb, qb + 3u>>, 4>;
                    hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, qr> qt;
                    constexpr bool kInit = (S < 4u); // first touch of N-group rg -> 3-arg init
                    if constexpr(rg == 0u)
                    {
                        if constexpr(kInit)
                            hk::mma_ABt(p_comp_lo, kt, qt);
                        else
                            hk::mma_ABt(p_comp_lo, kt, qt, p_comp_lo);
                    }
                    else if constexpr(rg == 1u)
                    {
                        if constexpr(kInit)
                            hk::mma_ABt(p_comp_hi, kt, qt);
                        else
                            hk::mma_ABt(p_comp_hi, kt, qt, p_comp_hi);
                    }
                    else if constexpr(rg == 2u)
                    {
                        if constexpr(kInit)
                            hk::mma_ABt(p_comp_b_lo, kt, qt);
                        else
                            hk::mma_ABt(p_comp_b_lo, kt, qt, p_comp_b_lo);
                    }
                    else
                    {
                        if constexpr(kInit)
                            hk::mma_ABt(p_comp_b_hi, kt, qt);
                        else
                            hk::mma_ABt(p_comp_b_hi, kt, qt, p_comp_b_hi);
                    }
                };

                // Prologue: preload kDepth ds_reads.
                opus::static_for<kDepth>([&](auto p_) { issue_k.template operator()<p_.value>(); });
                // Steady + drain: wait until sub-tile S is the oldest still queued
                // (lgkmcnt = min(reads-behind-S, kDepth-1)), mma it, then refill
                // its slot with S+kDepth.
                opus::static_for<kNumSub>([&](auto s_) {
                    constexpr uint32_t S    = s_.value;
                    constexpr uint32_t tail = kNumSub - 1u - S;
                    constexpr uint32_t w    = (tail < kDepth - 1u) ? tail : (kDepth - 1u);
                    __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/w, -1));
                    mma_k.template operator()<S>();
                    if constexpr(S + kDepth < kNumSub)
                        issue_k.template operator()<S + kDepth>();
                });
            }

            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(2);
            __builtin_amdgcn_sched_barrier(0);

            // ---- Phase B+C: wait + cvt + store NEXT tile to LDS ----
            // Sequenced after QK so the QK ds_reads from p_lds_kv_curr aren't
            // delayed by the cvt+store traffic on p_lds_kv_next.
            if constexpr(kIsGlobalLast == false)
            {
                // This warp's band goes into its own sub-tile base (next + sub_off).
                const uintptr_t next_sub = p_lds_kv_next + sub_off;
                hk::u32x4 dw;

                // RoPE (hi warps only): strip 3 (cols 448-511) DMAs vmem->LDS.
                kv_manager.template prefetch_kv_rope<kCheckBoundaryNext, kIsRopeWarp>(
                    next_sub, warp_idx, params.p_kv_buffer_rope, row_kv_ld_next);

                // Carrier strip 0.
                kv_manager.template wait_kv_loads<kIsRopeWarp, /*kVmCnt=*/6>(warp_idx);
                const float scale_f0 = kv_manager.kv_tile_scale_f(p0);
                kv_manager.template cvt_kv_tile_step<0>(dw, p0.nope_dw, scale_f0);
                kv_manager.template cvt_kv_tile_step<1>(dw, p0.nope_dw, scale_f0);
                kv_manager.template store_kv_tile_step<0u, kTile, 0>(next_sub, warp_idx, dw);
                kv_manager.template cvt_kv_tile_step<2>(dw, p0.nope_dw, scale_f0);
                kv_manager.template cvt_kv_tile_step<3>(dw, p0.nope_dw, scale_f0);
                kv_manager.template store_kv_tile_step<0u, kTile, 1>(next_sub, warp_idx, dw);

                // Carrier strip 1.
                kv_manager.template wait_kv_loads<kIsRopeWarp, /*kVmCnt=*/4>(warp_idx);
                const float scale_f1 = kv_manager.kv_tile_scale_f(p1);
                kv_manager.template cvt_kv_tile_step<0>(dw, p1.nope_dw, scale_f1);
                kv_manager.template cvt_kv_tile_step<1>(dw, p1.nope_dw, scale_f1);
                kv_manager.template store_kv_tile_step<1u, kTile, 0>(next_sub, warp_idx, dw);
                kv_manager.template cvt_kv_tile_step<2>(dw, p1.nope_dw, scale_f1);
                kv_manager.template cvt_kv_tile_step<3>(dw, p1.nope_dw, scale_f1);
                kv_manager.template store_kv_tile_step<1u, kTile, 1>(next_sub, warp_idx, dw);

                // Staged strip 2 (lo & hi).
                {
                    kv_manager.template wait_kv_loads<kIsRopeWarp, /*kVmCnt=*/2>(warp_idx);
                    const hk::u32x4 s2 = kv_manager.template load_staged_kv_carrier<0u>(stage_t0);
                    const float sf2    = hk_mla::e8m0_to_f32(scale_s0);
                    __builtin_amdgcn_s_waitcnt(
                        hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/-1));
                    kv_manager.template cvt_kv_tile_step<0>(dw, s2, sf2);
                    kv_manager.template cvt_kv_tile_step<1>(dw, s2, sf2);
                    kv_manager.template store_kv_tile_step<2u, kTile, 0>(next_sub, warp_idx, dw);
                    kv_manager.template cvt_kv_tile_step<2>(dw, s2, sf2);
                    kv_manager.template cvt_kv_tile_step<3>(dw, s2, sf2);
                    kv_manager.template store_kv_tile_step<2u, kTile, 1>(next_sub, warp_idx, dw);
                }

                // Strip 3 (lo) is deferred to the NEXT call's top (see above); hi's
                // strip 3 is the RoPE DMA issued at Phase B start.
            }

            // ---- Update row_kv_ld_next_next for the call AFTER this one ----
            // Hi warps use softmax + PV GEMM to hide latency of loading next-next rows.
            if constexpr((kIsGlobalLast == false) && (kWarpType != WarpTypeM16x8::LoNoPEWarp))
            {
                row_kv_ld_next_next =
                    resolve_row_kv_ld(kv_tile_start + 2 * T::kBlockN + band_off); // +128
            }

            // ---- Softmax + fp32->bf16 pack ----
            //
            // p_comp is 8 fp32 lanes (kBlockN=32 N-cols x kTileM=16 rows / 64
            // lanes = 8 elems/lane), laid out per softmax_scale_p_8: lane's
            // col_0 group covers vgprs +0..+3 (N-cols [col_0_idx*4, +4)) and
            // col_1 group covers +4..+7 (N-cols [col_0_idx*4+16, +20)).
            const uint32_t col_0_idx = lane_idx >> 4;
            comp_t local_max{};
            rescale = 1.0f;
            // Wave-uniform: does the running max move enough this tile to
            // require rescaling the prior oaccu / row_sum_e? Decided by ballot
            // so the whole wave agrees (oaccu rescale is a per-wave op, but
            // each lane owns different rows). Stays false on kIsFirstIter (no
            // prior oaccu) and whenever every active lane's new max is within
            // T::kRescaleThreshold of the stale max -- in which case row_max is
            // kept stale and the rescale multiplies (all == 1) are skipped.
            do_rescale = false;
            if constexpr(kSkipCompute == false)
            {
                // Q was pre-scaled by sm_scale*log2e in load_q, so only mask
                // OOB columns here (no per-tile multiply). Two 32-col sub-tiles:
                // A = p_comp dwords 0:7 (N-cols [start, +32)), B = dwords 8:15
                // (N-cols [start+32, +64)). Each gets its own boundary check.
                const uint32_t kv_tile_start_u = static_cast<uint32_t>(kv_tile_start);
                const uint32_t kv_end_eff_u    = static_cast<uint32_t>(kv_end_eff);
                if((kv_tile_start_u + 32u) > kv_end_eff_u)
                {
                    softmax_mask_p<true, k_p_comp_begin, kSoftmaxUsePk>(
                        col_0_idx * 4u + kv_tile_start_u, kv_end_eff_u);
                }
                else
                {
                    softmax_mask_p<false, k_p_comp_begin, kSoftmaxUsePk>(
                        col_0_idx * 4u + kv_tile_start_u, kv_end_eff_u);
                }
                if((kv_tile_start_u + 64u) > kv_end_eff_u)
                {
                    softmax_mask_p<true, k_p_comp_begin + 8u, kSoftmaxUsePk>(
                        col_0_idx * 4u + kv_tile_start_u + 32u, kv_end_eff_u);
                }
                else
                {
                    softmax_mask_p<false, k_p_comp_begin + 8u, kSoftmaxUsePk>(
                        col_0_idx * 4u + kv_tile_start_u + 32u, kv_end_eff_u);
                }

                // Row-wise max across 16 p_comp vgprs (both sub-tiles), then across
                // the 4-lane M-group via warp_reduce (matches softmax_p0's reduction).
                local_max = max_16<k_p_comp_begin, comp_t>();
                {
                    constexpr int32_t reduce_range = opus::get_warp_size();
                    constexpr int32_t stop_stride  = opus::get_warp_size() / 4 - 1;
                    local_max                      = warp_reduce<aiter::MaxFunctor,
                                                                 decltype(local_max),
                                                                 reduce_range,
                                                                 stop_stride>(local_max);
                }
                if constexpr(kIsFirstIter)
                {
                    row_max = local_max;
                    rescale = 1.0f;
                }
                else
                {
                    // Lane-private: would this lane's rows need a rescale?
                    const bool lane_needs =
                        (local_max - row_max) > static_cast<comp_t>(T::kRescaleThreshold);
                    // Promote to a wave-uniform decision: rescale iff ANY active
                    // lane needs it (ballot != 0). When no lane needs it, keep
                    // row_max stale so exp(p_comp - row_max) accumulates into the
                    // existing oaccu reference (rescale stays 1.0, mults skipped).
                    do_rescale = (__builtin_amdgcn_ballot_w64(lane_needs) != 0ull);
                    if(do_rescale)
                    {
                        const comp_t new_row_max = opus::max(local_max, row_max);
                        // row_max is already in log2 units (Q pre-scaled), so no
                        // * log2e here.
                        rescale = __builtin_amdgcn_exp2f(row_max - new_row_max);
                        row_max = new_row_max;
                    }
                }

                // exp + sum + warp_reduce(add) -> row_sum_e. Updates p_comp in
                // place to hold exp(p_comp - row_max). rescale==1.0 when the
                // running max was kept stale, so the prior row_sum_e carries
                // forward unscaled.
                softmax_p1_prescaled_16<kIsFirstIter, k_p_comp_begin, comp_t, kSoftmaxUsePk>(
                    &row_sum_e, row_max, rescale);

                // ---- fp32->bf16 pack (p_comp -> p_mfma overlay) ----
                // 16 fp32 -> 8 bf16x2 dwords. A: p_comp 0:7 -> p_mfma 0:3;
                // B: p_comp 8:15 -> p_mfma 4:7. Low-to-high pack order hazard-free.
                pack_2f32_to_bf16_pair_pinned<k_p_mfma_begin + 0, k_p_comp_begin + 0>();
                pack_2f32_to_bf16_pair_pinned<k_p_mfma_begin + 1, k_p_comp_begin + 2>();
                pack_2f32_to_bf16_pair_pinned<k_p_mfma_begin + 2, k_p_comp_begin + 4>();
                pack_2f32_to_bf16_pair_pinned<k_p_mfma_begin + 3, k_p_comp_begin + 6>();
                pack_2f32_to_bf16_pair_pinned<k_p_mfma_begin + 4, k_p_comp_begin + 8>();
                pack_2f32_to_bf16_pair_pinned<k_p_mfma_begin + 5, k_p_comp_begin + 10>();
                pack_2f32_to_bf16_pair_pinned<k_p_mfma_begin + 6, k_p_comp_begin + 12>();
                pack_2f32_to_bf16_pair_pinned<k_p_mfma_begin + 7, k_p_comp_begin + 14>();
            }

            __builtin_amdgcn_s_setprio(1);

            // ---- oaccu rescale + PV GEMM ----
            //
            if constexpr(kPvAtEnd && (kSkipCompute == false))
            {
                // One PV call contracts BOTH KV sub-tiles (kBlockN=64) -> single prologue.
                hk_mla_v40_pv_stage<kIsFirstIter, T>(
                    kv_manager, p_lds_kv_curr, rescale, do_rescale);
            }

            // ---- Epilogue ----
            //
            // Rescale oaccu by 1/row_sum_e (single mul_vgpr over full 128-vgpr
            // tile), then write 16-row x kVoHeadDim tile to vmem.
            //   partial_qo_loc < 0 -> final_output via OManager16bitsV4Gen1Swizzle (bf16).
            //   partial_qo_loc >= 0 -> split_output via OManager32bitsV4Gen1Swizzle (fp32)
            //                          + per-warp LSE row (lanes 0..15).
            // O LDS bounce overlays p_lds_kv_next (the next pong is dead on
            // the global last iter -- the swap is a no-op and the next
            // work_idx's KV prologue writes to p_lds_kv_curr).
            if constexpr(kDoEpilogue)
            {
                // Lo drain: on the global-last call that ran a QK (!kSkipCompute),
                // THIS tile's PV is still pending (Lo defers PV by one tile). Run
                // it now, reading V from p_lds_kv_curr (this tile; no swap on the
                // global last). The kSkipCompute global-last case already drained
                // the warp's last real tile via the deferred PV at call start
                // (kHasPv), so it needs nothing here. Hi: PV already ran at end.
                if constexpr((!kPvAtEnd) && (kSkipCompute == false))
                {
                    // One PV call contracts BOTH KV sub-tiles (kBlockN=64).
                    hk_mla_v40_pv_stage<false, T>(kv_manager, p_lds_kv_curr, rescale, do_rescale);
                }

                // ---- Attention-sink fold ----
                // Apply on OutputFinal (single-split == global) OR on the
                // LAST split of this batch element. By inflating exactly
                // one split's row_sum_e (and thus its lse), the reducer's
                // sum_k exp(lse_k - global_lse) * out_k formula naturally
                // routes exp(sink) into the global denominator exactly once
                // while contributing 0 to the V numerator.
                //
                // attn_sink is a per-lane VGPR loaded once at kernel entry
                // (-inf if p_attn_sink is null, so exp(...)=0 -> no-op).
                if(kEpilogueType == PvGemmEpilogueType::OutputFinal || is_last_split)
                {
                    // row_max is in log2 units (Q pre-scaled), attn_sink is a raw
                    // logit, so convert the sink to log2 units before the diff.
                    const float sink_term = __builtin_amdgcn_exp2f(attn_sink * log2e - row_max);
                    row_sum_e += sink_term;
                }

                const comp_t reci_row_sum_e = 1.0f / row_sum_e;
                // hk::mul_vgpr generates v_pk_mul; the de-packed sweep generates v_mul.
                if constexpr(kFinalRescaleUsePk)
                {
                    hk::mul_vgpr(oaccu, oaccu, reci_row_sum_e);
                }
                else
                {
                    opus::static_for<R::k_o_sz>([reci_row_sum_e](auto i) {
                        asm volatile("v_mul_f32_e32 v[%0], %1, v[%0]"
                                     :
                                     : "n"(k_o_begin + i.value), "v"(reci_row_sum_e));
                    });
                }

                const uintptr_t p_lds_o = p_lds_kv_next;
                // Output is 512 cols regardless of kBlockN: each pair writes 16 oaccu
                // dwords (2x 16x16 = 64 cols). Bind to 64, not 2*kBlockN.
                constexpr uint32_t kOutPairCols     = 64u;
                constexpr uint32_t num_pv_pair_iter = T::kVoHeadDim / kOutPairCols; // 8
                if constexpr(kEpilogueType == PvGemmEpilogueType::OutputFinal)
                {
                    opus::static_for<num_pv_pair_iter>([&](auto i) {
                        constexpr uint32_t iter       = i.value;
                        constexpr uint32_t kOaccuBase = k_o_begin + iter * 16u;
                        constexpr uint32_t kColOff    = iter * kOutPairCols;
                        o_manager.template output_to_vram_pair<kOaccuBase, kColOff, true>(
                            params.p_final_output, warp_idx, qo_start, qo_end, p_lds_o, num_qheads);
                        // Block LLVM from fusing adjacent OMgr calls' ds_reads
                        // (caps in-flight depth, keeps OMgr targets at v[58:69]).
                        __builtin_amdgcn_sched_barrier(0);
                    });
                }
                else
                {
                    opus::static_for<num_pv_pair_iter>([&](auto i) {
                        constexpr uint32_t iter       = i.value;
                        constexpr uint32_t kOaccuBase = k_o_begin + iter * 16u;
                        constexpr uint32_t kColOff    = iter * kOutPairCols;
                        split_o_manager.template output_to_vram_pair<kOaccuBase, kColOff, false>(
                            params.p_split_output,
                            warp_idx,
                            static_cast<uint32_t>(partial_qo_loc),
                            0,
                            p_lds_o,
                            num_qheads);
                        __builtin_amdgcn_sched_barrier(0);
                    });

                    // LSE: row_max + ln(row_sum_e). Lanes 0..15 own the M-rows
                    // after warp_reduce; lanes 16..63 hold redundant copies.
                    constexpr uint32_t kMfmaResultRows = 16;
                    if(lane_idx < kMfmaResultRows)
                    {
                        constexpr comp_t inv_log2e = 1.0f / log2e;
                        const uint32_t row_idx = lane_idx + warp_idx * kMfmaResultRows +
                                                 static_cast<uint32_t>(partial_qo_loc) * num_qheads;
                        // row_max is now in log2 units (Q pre-scaled).
                        // __builtin_amdgcn_logf == v_log_f32 == LOG2 in HW.
                        // lse_nat = row_max_nat + ln(sum)
                        //         = row_max_log2*inv_log2e + log2(sum)*inv_log2e
                        //         = (row_max + log2(sum)) * inv_log2e.
                        const comp_t lse = (row_max + __builtin_amdgcn_logf(row_sum_e)) * inv_log2e;
                        params.p_split_lse[row_idx] = lse;
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
#if !defined(MLA_SLIM_DISPATCH)
        if(kv_len_eff <= 0)
        {
            // Warp fully idle. num_iters == 1. One skip iter on the global
            // last tile, no epilogue (no oaccu state).
            mla_main.template operator()<false, true, PvGemmEpilogueType::None, false>(kv_start,
                                                                                       kv_end);
        }
        else if(kv_len_eff < T::kBlockN)
        {
            // Warp has exactly 1 partial real tile.
            if(kv_len < T::kBlockN)
            {
                // num_iters == 1: single real iter, also the epilogue iter.
                if(partial_qo_loc < 0)
                {
                    mla_main
                        .template operator()<true, false, PvGemmEpilogueType::OutputFinal, false>(
                            kv_start, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<true, false, PvGemmEpilogueType::OutputSplit, false>(
                            kv_start, kv_end);
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
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputFinal, false>(
                            kv_start + T::kBlockN, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputSplit, false>(
                            kv_start + T::kBlockN, kv_end);
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
                    mla_main
                        .template operator()<true, false, PvGemmEpilogueType::OutputFinal, false>(
                            kv_start, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<true, false, PvGemmEpilogueType::OutputSplit, false>(
                            kv_start, kv_end);
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
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputFinal, false>(
                            kv_start + T::kBlockN, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputSplit, false>(
                            kv_start + T::kBlockN, kv_end);
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
            // Middle real tiles. Split the range so the inner loop only
            // contains iters whose NEXT tile is fully in bounds
            // (kCheckBoundaryNext=false, cheap). Any final middle iter
            // whose next tile may straddle the global end is handled
            // outside the loop with kCheckBoundaryNext=true. This avoids
            // a per-iter branch inside the hot middle loop (~2-3% perf
            // gain measured via thread trace).
            while((kv_idx + T::kBlockN) < kv_end_eff && (kv_idx + 2 * T::kBlockN) <= kv_end)
            {
                mla_main.template operator()<false, false, PvGemmEpilogueType::None, false>(
                    kv_idx, kv_idx + T::kBlockN);
                kv_idx += T::kBlockN;
            }
            // Trailing middle iter (if any): its next tile is the global
            // last (possibly partial) -> boundary-checked prefetch.
            if((kv_idx + T::kBlockN) < kv_end_eff)
            {
                mla_main.template operator()<false, false, PvGemmEpilogueType::None, true>(
                    kv_idx, kv_idx + T::kBlockN);
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
                    mla_main
                        .template operator()<false, false, PvGemmEpilogueType::OutputFinal, false>(
                            kv_idx, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<false, false, PvGemmEpilogueType::OutputSplit, false>(
                            kv_idx, kv_end);
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
                    mla_main.template operator()<false, false, PvGemmEpilogueType::None, false>(
                        kv_idx, kv_idx + T::kBlockN);
                }
                // Skip + epilogue on the global last tile.
                if(partial_qo_loc < 0)
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputFinal, false>(
                            kv_idx + T::kBlockN, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputSplit, false>(
                            kv_idx + T::kBlockN, kv_end);
                }
            }
        }
#else  // MLA_SLIM_DISPATCH
       // Slim dispatch: always use kCheckBoundaryNext=true. This drops the
       // kv_len%kBlockN==0 / kv_len_eff%kBlockN==0 fast-path
       // instantiations (rare in practice with random kv seqlens), halving
       // the number of template instantiations of mla_main. Cost: 1 cmp +
       // 1 cmov per K-iter for in_bounds check inside prefetch_kv_tile.
        if(kv_len_eff <= 0)
        {
            // Warp fully idle. Single skip iter, no epilogue.
            mla_main.template operator()<false, true, PvGemmEpilogueType::None, false>(kv_start,
                                                                                       kv_end);
        }
        else if(kv_len_eff <= T::kBlockN)
        {
            // Warp has exactly 1 real tile (full or partial).
            const bool tile_is_global_last = (kv_start + T::kBlockN) >= kv_end;
            if(tile_is_global_last)
            {
                // Real iter is also the epilogue iter; no next tile.
                if(partial_qo_loc < 0)
                {
                    mla_main
                        .template operator()<true, false, PvGemmEpilogueType::OutputFinal, false>(
                            kv_start, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<true, false, PvGemmEpilogueType::OutputSplit, false>(
                            kv_start, kv_end);
                }
            }
            else
            {
                // Real iter prefetches the global last tile (boundary-checked).
                mla_main.template operator()<true, false, PvGemmEpilogueType::None, true>(
                    kv_start, kv_start + T::kBlockN);
                // Trailing skip + epilogue.
                if(partial_qo_loc < 0)
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputFinal, false>(
                            kv_start + T::kBlockN, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputSplit, false>(
                            kv_start + T::kBlockN, kv_end);
                }
            }
        }
        else // kv_len_eff > kBlockN: >= 2 real tiles
        {
            const int32_t kv_1st_end = kv_start + T::kBlockN;

            // First real iter; next prefetch boundary-checked.
            mla_main.template operator()<true, false, PvGemmEpilogueType::None, true>(kv_start,
                                                                                      kv_1st_end);

            int32_t kv_idx = kv_1st_end;
            // Middle real tiles. Split the range so the inner loop only
            // contains iters whose NEXT tile is fully in bounds
            // (kCheckBoundaryNext=false, cheap). Any final middle iter
            // whose next tile may straddle the global end is handled
            // outside the loop with kCheckBoundaryNext=true.
            while((kv_idx + T::kBlockN) < kv_end_eff && (kv_idx + 2 * T::kBlockN) <= kv_end)
            {
                mla_main.template operator()<false, false, PvGemmEpilogueType::None, false>(
                    kv_idx, kv_idx + T::kBlockN);
                kv_idx += T::kBlockN;
            }
            // Trailing middle iter (if any): its next tile is the global
            // last (possibly partial) -> boundary-checked prefetch.
            if((kv_idx + T::kBlockN) < kv_end_eff)
            {
                mla_main.template operator()<false, false, PvGemmEpilogueType::None, true>(
                    kv_idx, kv_idx + T::kBlockN);
                kv_idx += T::kBlockN;
            }

            // Warp's last real tile starts at kv_idx.
            const bool tile_is_global_last = ((kv_idx + T::kBlockN) >= kv_end);
            if(tile_is_global_last)
            {
                // Warp's last real == global last -> real iter with epilogue.
                if(partial_qo_loc < 0)
                {
                    mla_main
                        .template operator()<false, false, PvGemmEpilogueType::OutputFinal, false>(
                            kv_idx, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<false, false, PvGemmEpilogueType::OutputSplit, false>(
                            kv_idx, kv_end);
                }
            }
            else
            {
                // Last real iter prefetches the global last tile (boundary-checked).
                mla_main.template operator()<false, false, PvGemmEpilogueType::None, true>(
                    kv_idx, kv_idx + T::kBlockN);
                // Skip + epilogue on the global last tile.
                if(partial_qo_loc < 0)
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputFinal, false>(
                            kv_idx + T::kBlockN, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputSplit, false>(
                            kv_idx + T::kBlockN, kv_end);
                }
            }
        }
#endif // MLA_SLIM_DISPATCH
    }
}

template <typename T>
// scratch budget = HkMlaV40Regs::k_scratch_budget = k_p_comp_begin-12 = 36 for
// kBlockN=64 (pinned region grows down to v36 to fit the doubled p_comp).
__global__ __launch_bounds__(T::kNumThreads, T::kOccupancy) __attribute__((amdgpu_num_vgpr(
    36))) void kn_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1(HkMlaV40DecodeFwdParams<T>
                                                                          params)
{
    const uint32_t warp_idx = __builtin_amdgcn_readfirstlane(threadIdx.x / opus::get_warp_size());

    // Diverge on warp type ONCE at kernel entry so each type compiles as its own
    // body (compile-time kWarpType). Band remap (V2): warps 0-3 own tile 0 (pure
    // NoPE) = LoNoPEWarp; warps 4-7 own tile 1 (NoPE + RoPE) = HiRoPEWarp. The band
    // (warp&3) stays runtime inside each type.
    if(warp_idx < 4u)
    {
        mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1_impl<WarpTypeM16x8::LoNoPEWarp>(
            params, warp_idx);
    }
    else
    {
        mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1_impl<WarpTypeM16x8::HiRoPEWarp>(
            params, warp_idx);
    }
}

#else
template <typename T>
__global__ __launch_bounds__(
    T::kNumThreads,
    T::kOccupancy) void kn_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1(HkMlaV40DecodeFwdParams<T>
                                                                                   params)
{
    (void)params;
    assert(false);
}
#endif

template <typename Traits>
void mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1(aiter_tensor_t& query,
                                                         aiter_tensor_t& query_rope,
                                                         aiter_tensor_t& kv_buffer,
                                                         aiter_tensor_t& kv_buffer_rope,
                                                         const aiter_tensor_t& qo_indptr,
                                                         const aiter_tensor_t& kv_page_indices,
                                                         const aiter_tensor_t& kv_last_page_lens,
                                                         const aiter_tensor_t& work_indptr,
                                                         const aiter_tensor_t& work_info_set,
                                                         const int max_seqlen_q,
                                                         const float softmax_scale,
                                                         aiter_tensor_t& split_output,
                                                         aiter_tensor_t& split_lse,
                                                         aiter_tensor_t& final_output,
                                                         const float* p_attn_sink)
{
    // Shape / dtype / rank checks live ONCE in the outer dispatcher
    // (hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1) so we don't
    // pay for them per page_size template instantiation.
    const int32_t num_qheads      = query.size(1);
    const int32_t log2_num_qheads = __builtin_ctz(num_qheads);

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const hipStream_t stream = aiter::getCurrentHIPStream();

    HkMlaV40DecodeFwdParams<Traits> params = {
        reinterpret_cast<typename Traits::q_nope_t const*>(query.data_ptr()),
        reinterpret_cast<typename Traits::q_rope_t const*>(query_rope.data_ptr()),
        reinterpret_cast<typename Traits::kv_nope_t const*>(kv_buffer.data_ptr()),
        reinterpret_cast<typename Traits::kv_rope_t const*>(kv_buffer_rope.data_ptr()),
        // kv_indices
        reinterpret_cast<int32_t*>(kv_page_indices.data_ptr()),
        // kv_last_page_lens (only read by kernel when kPageSize > 1)
        reinterpret_cast<int32_t*>(kv_last_page_lens.data_ptr()),
        // metadata
        reinterpret_cast<int32_t*>(work_indptr.data_ptr()),
        reinterpret_cast<int32_t*>(work_info_set.data_ptr()),
        // optional per-head attention sink ([num_qheads] fp32, or nullptr)
        p_attn_sink,
        // outputs
        reinterpret_cast<typename Traits::out_t*>(final_output.data_ptr()),
        reinterpret_cast<float*>(split_output.data_ptr()),
        reinterpret_cast<float*>(split_lse.data_ptr()),
        // parameters
        softmax_scale,
        log2_num_qheads};

    const dim3 grid        = dim3(dev_prop.multiProcessorCount);
    const int32_t lds_size = dev_prop.maxSharedMemoryPerMultiProcessor / Traits::kOccupancy;

    kn_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1<Traits>
        <<<grid, Traits::kNumThreads, lds_size, stream>>>(params);
}

void hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1(aiter_tensor_t& query,
                                                            aiter_tensor_t& query_rope,
                                                            aiter_tensor_t& kv_buffer,
                                                            aiter_tensor_t& kv_buffer_rope,
                                                            const aiter_tensor_t& qo_indptr,
                                                            const aiter_tensor_t& kv_page_indices,
                                                            const aiter_tensor_t& kv_last_page_lens,
                                                            const aiter_tensor_t& work_indptr,
                                                            const aiter_tensor_t& work_info_set,
                                                            const int max_seqlen_q,
                                                            const float softmax_scale,
                                                            aiter_tensor_t& split_output,
                                                            aiter_tensor_t& split_lse,
                                                            aiter_tensor_t& final_output,
                                                            std::optional<aiter_tensor_t> attn_sink)
{
    HipDeviceGuard device_guard(final_output.device_id);

    const bool q_nope_is_fp8   = (query.dtype() == AITER_DTYPE_fp8);
    const bool kv_nope_is_fp8  = (kv_buffer.dtype() == AITER_DTYPE_fp8);
    const bool q_rope_is_bf16  = (query_rope.dtype() == AITER_DTYPE_bf16);
    const bool kv_rope_is_bf16 = (kv_buffer_rope.dtype() == AITER_DTYPE_bf16);

    AITER_CHECK(q_nope_is_fp8 && kv_nope_is_fp8,
                "hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1 requires FP8 NOPE; got q=",
                AiterDtype_to_str(query.dtype()),
                ", kv=",
                AiterDtype_to_str(kv_buffer.dtype()));
    AITER_CHECK(
        q_rope_is_bf16 && kv_rope_is_bf16,
        "hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1 requires BF16 ROPE; got q_rope=",
        AiterDtype_to_str(query_rope.dtype()),
        ", kv_rope=",
        AiterDtype_to_str(kv_buffer_rope.dtype()));

    // ---- Shape / rank checks ----
    // The kernel takes raw device pointers (no HK gl_* shape carrier), so
    // every shape MUST be validated here against the V4 layout constants:
    // any mismatch silently OOBs the kernel. Checks live ONCE in the outer
    // dispatcher (page_size-independent constants only) so the rank/size
    // logic doesn't bloat per page-size instantiation.
    //
    // Pull constants from a dummy traits instantiation so the values stay in
    // sync with HkMlaV40DecodeFwdTraits without duplication. kPageSize_=1 is
    // arbitrary -- only page_size-independent constants are used below.
    using DummyTraits = HkMlaV40DecodeFwdTraits<hk::fp8e4m3,
                                                hk::bf16,
                                                hk::fp8e4m3,
                                                hk::bf16,
                                                hk::bf16,
                                                /*kBlockN_=*/32,
                                                /*kNumWarps_=*/8,
                                                /*kOccupancy_=*/1,
                                                /*kBlockM_=*/128,
                                                /*kPageSize_=*/1>;

    const int64_t num_qheads = query.size(1);
    AITER_CHECK((num_qheads & (num_qheads - 1)) == 0 && num_qheads >= 16 && num_qheads <= 128,
                "num_qheads must be a power of 2 in [16, 128], got ",
                num_qheads);
    AITER_CHECK(num_qheads * max_seqlen_q == DummyTraits::kBlockM,
                "num_qheads * max_seqlen_q must equal ",
                DummyTraits::kBlockM,
                ", got ",
                num_qheads,
                " * ",
                max_seqlen_q,
                " = ",
                num_qheads * max_seqlen_q);

    AITER_CHECK(query.dim() == 3,
                "query must be 3-D [total_q, num_qheads, kQkPackedNopeQElems], got rank ",
                query.dim());
    AITER_CHECK(query.size(2) == DummyTraits::kQkPackedNopeQElems,
                "query.size(2) must equal kQkPackedNopeQElems=",
                DummyTraits::kQkPackedNopeQElems,
                ", got ",
                query.size(2));

    AITER_CHECK(query_rope.dim() == 3,
                "query_rope must be 3-D [total_q, num_qheads, kQkRopeHeadDim], got rank ",
                query_rope.dim());
    AITER_CHECK(query_rope.size(0) == query.size(0) && query_rope.size(1) == num_qheads,
                "query_rope dims 0,1 must match query: query=[",
                query.size(0),
                ",",
                query.size(1),
                "] vs query_rope=[",
                query_rope.size(0),
                ",",
                query_rope.size(1),
                "]");
    AITER_CHECK(query_rope.size(2) == DummyTraits::kQkRopeHeadDim,
                "query_rope.size(2) must equal kQkRopeHeadDim=",
                DummyTraits::kQkRopeHeadDim,
                ", got ",
                query_rope.size(2));

    const int32_t page_size = kv_buffer.size(1);

    AITER_CHECK(kv_buffer.dim() == 4,
                "kv_buffer must be 4-D [num_page, page_size, kKvNumHead, kQkPackedNopeKvElems], "
                "got rank ",
                kv_buffer.dim());
    AITER_CHECK(kv_buffer.size(2) == DummyTraits::kKvNumHead,
                "kv_buffer.size(2) must equal kKvNumHead=",
                DummyTraits::kKvNumHead,
                ", got ",
                kv_buffer.size(2));
    AITER_CHECK(kv_buffer.size(3) == DummyTraits::kQkPackedNopeKvElems,
                "kv_buffer.size(3) must equal kQkPackedNopeKvElems=",
                DummyTraits::kQkPackedNopeKvElems,
                ", got ",
                kv_buffer.size(3));

    AITER_CHECK(
        kv_buffer_rope.dim() == 4, "kv_buffer_rope must be 4-D, got rank ", kv_buffer_rope.dim());
    AITER_CHECK(
        kv_buffer_rope.size(0) == kv_buffer.size(0) && kv_buffer_rope.size(1) == page_size &&
            kv_buffer_rope.size(2) == DummyTraits::kKvNumHead,
        "kv_buffer_rope dims 0..2 must match kv_buffer's [num_page, page_size, kKvNumHead]=[",
        kv_buffer.size(0),
        ",",
        page_size,
        ",",
        DummyTraits::kKvNumHead,
        "], got [",
        kv_buffer_rope.size(0),
        ",",
        kv_buffer_rope.size(1),
        ",",
        kv_buffer_rope.size(2),
        "]");
    AITER_CHECK(kv_buffer_rope.size(3) == DummyTraits::kQkRopeHeadDim,
                "kv_buffer_rope.size(3) must equal kQkRopeHeadDim=",
                DummyTraits::kQkRopeHeadDim,
                ", got ",
                kv_buffer_rope.size(3));

    AITER_CHECK(final_output.dim() == 3,
                "final_output must be 3-D [total_q, num_qheads, kVoHeadDim], got rank ",
                final_output.dim());
    AITER_CHECK(final_output.size(0) == query.size(0) && final_output.size(1) == num_qheads &&
                    final_output.size(2) == DummyTraits::kVoHeadDim,
                "final_output shape must be [",
                query.size(0),
                ",",
                num_qheads,
                ",",
                DummyTraits::kVoHeadDim,
                "], got [",
                final_output.size(0),
                ",",
                final_output.size(1),
                ",",
                final_output.size(2),
                "]");

    AITER_CHECK(split_output.dim() >= 2 &&
                    split_output.size(split_output.dim() - 1) == DummyTraits::kVoHeadDim,
                "split_output trailing dim must equal kVoHeadDim=",
                DummyTraits::kVoHeadDim,
                ", got ",
                split_output.size(split_output.dim() - 1));
    AITER_CHECK(split_lse.dim() >= 1, "split_lse must have rank >= 1");

    AITER_CHECK(work_indptr.dim() == 1, "work_indptr must be 1-D, got rank ", work_indptr.dim());
    AITER_CHECK(kv_page_indices.dim() == 1,
                "kv_page_indices must be 1-D, got rank ",
                kv_page_indices.dim());

    // Optional attention sink: [num_qheads] fp32. Disabled when absent.
    const float* p_attn_sink = nullptr;
    if(attn_sink.has_value())
    {
        const aiter_tensor_t& s = attn_sink.value();
        AITER_CHECK(s.dtype() == AITER_DTYPE_fp32,
                    "attn_sink must be fp32, got ",
                    AiterDtype_to_str(s.dtype()));
        AITER_CHECK(s.dim() == 1, "attn_sink must be 1-D, got rank ", s.dim());
        AITER_CHECK(s.size(0) == num_qheads,
                    "attn_sink.size(0) must equal num_qheads=",
                    num_qheads,
                    ", got ",
                    s.size(0));
        p_attn_sink = reinterpret_cast<const float*>(s.data_ptr());
    }

#define DISPATCH_PAGE_SIZE(PageSize)                                                   \
    case PageSize: {                                                                   \
        using Traits = HkMlaV40DecodeFwdTraits<hk::fp8e4m3,                            \
                                               hk::bf16,                               \
                                               hk::fp8e4m3,                            \
                                               hk::bf16,                               \
                                               hk::bf16,                               \
                                               /*kBlockN_=*/64,                        \
                                               /*kNumWarps_=*/8,                       \
                                               /*kOccupancy_=*/1,                      \
                                               /*kBlockM_=*/128,                       \
                                               /*kPageSize_=*/PageSize>;               \
        mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1<Traits>(query,             \
                                                                    query_rope,        \
                                                                    kv_buffer,         \
                                                                    kv_buffer_rope,    \
                                                                    qo_indptr,         \
                                                                    kv_page_indices,   \
                                                                    kv_last_page_lens, \
                                                                    work_indptr,       \
                                                                    work_info_set,     \
                                                                    max_seqlen_q,      \
                                                                    softmax_scale,     \
                                                                    split_output,      \
                                                                    split_lse,         \
                                                                    final_output,      \
                                                                    p_attn_sink);      \
        break;                                                                         \
    }

    // Only page_size in {1, 64} are instantiated -- same pattern as v32.
    switch(page_size)
    {
        DISPATCH_PAGE_SIZE(1)
        DISPATCH_PAGE_SIZE(64)
    default:
        AITER_CHECK(
            false,
            "hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1: unsupported page_size ",
            page_size,
            " (supported: 1, 64).");
    }

#undef DISPATCH_PAGE_SIZE
}
