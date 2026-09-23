// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// topk_index_score.hpp -- __global__ entry, per-variant host launchers,
// and the explicit instantiation table for the bf16-Q x fp8-K decode
// lightning-indexer scoring path.
//
//   device template  topk_index_score_kernels.cuh (ported from
//                    the accepted configuration of the porting study)
//   this file        P2, launcher half: __global__ + variant table + host wave
//                    geometry. NOT the ctypes entry point, NOT the Python
//                    dispatch, NOT the grid lever -- those are later files.
//
// THIS IS AN ADDITION. The incumbent fp8-Q x fp8-K path
// (pa_sparse_block_select.hpp, SPARSE_DECODE_TABLE) is not referenced, not
// included, and not modified by this header: its variants stay byte-identical
// (T1). The two operators differ in Q dtype, scale handling and sentinels --
// matching the incumbent's signature here would be how a wrong dispatch gets
// written, so this launcher deliberately does not.
//
// CERTIFICATION, and why the table and the dispatch differ in size
// (integration.md section 8.1, lead scope ruling):
//
//   BUILD every (H, Q) x AUX_K cell -- so any evidence gap is VISIBLE and
//   closable, and so per-cell code objects exist for a reviewer to compare.
//
//   DISPATCH only cells with per-cell BITWISE evidence at config 3. Today that
//   is (H=1, Q=4) only -- the deployed cell: TP4 and TP8 both land on H=1
//   (integration.md section 8.6), and phase-1 Tier-2 evidence at config 3 covers
//   {(H=1,Q=4) x the traced shape} and nothing else. Every other cell is BUILT
//   BUT NOT CERTIFIED and must be refused with that reason by the entry point
//   (adversarial case N16). "Class A by the envelope's own logic" is an argument
//   that those cells should behave; the envelope's logic has been correctly
//   derived and then refuted before. A built-but-undispatched variant makes the
//   owed evidence visible; a dispatched-on-argument variant makes it invisible.
//
// This header exposes the certification predicate as DATA
// (opus_idx_score_cell_certified). It performs no dispatch of its own.

#include "aiter_hip_common.h"
#include "topk_index_score_kernels.cuh"

// Build constants, fixed like the incumbent's (SPARSE_HEAD_DIM /
// SPARSE_BLOCK_SIZE). head_dim = 128 is not a default: invariant I5's 1.00 L1
// accesses per 128 B line is derived for 128-byte rows.
#define OPUS_IDX_HEAD_DIM 128
#define OPUS_IDX_BLOCK_SIZE 128

// Launcher signature. It carries sm_scale (ledger D1: the frozen contract takes
// it and applies sm_scale*log2e in-kernel) and whole-tensor element counts,
// which become the buffer-descriptor extents -- that is what buys the hardware
// bounds check the page offset relies on (invariant I3).
#define OPUS_IDX_SCORE_PARAMS                                                                 \
    const void *q_idx, const void *key_cache_idx, float *score, const int *block_table,       \
        const int *seq_lens, long long q_numel, long long key_cache_numel,                    \
        long long score_numel, long long block_table_numel, int batch, int num_chunks,        \
        int chunk_blocks, long long stride_q_n, long long stride_q_h, long long stride_ik_blk, \
        long long stride_s_h, long long stride_s_b, long long stride_bt_b, float sm_scale,    \
        hipStream_t stream

#define OPUS_IDX_SCORE_ARGS                                                                    \
    q_idx, key_cache_idx, score, block_table, seq_lens, q_numel, key_cache_numel, score_numel, \
        block_table_numel, batch, num_chunks, chunk_blocks, stride_q_n, stride_q_h,            \
        stride_ik_blk, stride_s_h, stride_s_b, stride_bt_b, sm_scale, stream

namespace aiter {
namespace sparse_attn {

// ---------------------------------------------------------------------------
// __global__ entry
// ---------------------------------------------------------------------------
// __launch_bounds__(256, 2) is carried over verbatim from the accepted build:
// it is part of the register-allocation context in which VGPR 61 / 8 waves per
// SIMD was measured, not a decoration. Changing it invalidates the occupancy
// premise of the grid lever (integration.md section 8.4).
//
// The args struct is passed BY VALUE, as in the prototype.
//
// K_T is fixed to opus fp8 here by the template's own Layer-K static_assert;
// the kernel takes the dtype as a parameter only so the bf16 exclusion stays a
// refusal rather than a deletion (ledger D0). On a non-gfx950 device pass this
// body calls the refusal stub, which traps -- it is never absent.
template <int H, int Q, int AUX_K>
__global__ __launch_bounds__(kOpusNumWarps* kOpusWarpSize, 2) void opus_decode_index_score_kernel(
    const opus_decode_score_args a)
{
    opus_index_score::decode_index_score_impl<
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
        opus::fp8_t,
#else
        uint8_t, // stub pass: 1 byte, so the sizeof(K_T) == 1 refusal still holds
#endif
        OPUS_IDX_HEAD_DIM,
        OPUS_IDX_BLOCK_SIZE,
        H,
        Q,
        AUX_K>(a);
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
// Grid is (batch, num_chunks) -- the PROTOTYPE's axis order, which is the order
// the accepted measurement was taken in, and deliberately NOT the incumbent's
// (q_tiles, reqs, chunks). num_chunks and chunk_blocks are computed by the
// caller from capture-time constants only (cudagraph invariants G1/G2); this
// launcher never reads a device tensor and never allocates.
template <int H, int Q, int AUX_K>
void launch_opus_decode_index_score(OPUS_IDX_SCORE_PARAMS)
{
    // G4: no zero-extent grid may be captured. The entry point also rejects
    // num_reqs == 0 earlier; this is the defense-in-depth layer, matching the
    // incumbent's convention of re-checking across layers.
    if(batch <= 0 || num_chunks <= 0)
        return;

    opus_decode_score_args a{};
    a.q_ptr         = q_idx;
    a.ik_ptr        = key_cache_idx;
    a.score_ptr     = score;
    a.bt_ptr        = block_table;
    a.seq_lens_ptr  = seq_lens;
    a.q_numel       = q_numel;
    a.ik_numel      = key_cache_numel;
    a.score_numel   = score_numel;
    a.bt_numel      = block_table_numel;
    a.batch         = batch;
    a.chunk_blocks  = chunk_blocks;
    a.stride_q_n    = stride_q_n;
    a.stride_q_h    = stride_q_h;
    a.stride_ik_blk = stride_ik_blk;
    a.stride_s_h    = stride_s_h;
    a.stride_s_b    = stride_s_b;
    a.stride_bt_b   = stride_bt_b;
    a.sm_scale      = sm_scale;

    dim3 grid((unsigned)batch, (unsigned)num_chunks, 1);
    dim3 block((unsigned)(kOpusNumWarps * kOpusWarpSize), 1, 1);

    opus_decode_index_score_kernel<H, Q, AUX_K><<<grid, block, 0, stream>>>(a);
}

// Host-side wave geometry, so the entry point and any future host-side sizing
// read the same numbers the kernel was built with instead of restating them.
constexpr int kOpusIdxScoreThreads = kOpusNumWarps * kOpusWarpSize; // 256

// ---------------------------------------------------------------------------
// Certification data (NOT dispatch)
// ---------------------------------------------------------------------------
// True only for cells with per-cell BITWISE evidence at config 3. The entry
// point turns a false here into an explicit "variant built but not certified"
// refusal (N16); it must never fall through to a launch. AUX_K is not part of
// the predicate: both (1,4) legs -- AUX_K = 0 (the control) and AUX_K = 3 (the
// accepted point) -- carry phase-1 Tier-2 correctness evidence.
//
// Editing this predicate is how a cell becomes dispatchable, and it may only be
// edited on certification evidence produced by something that did not author
// the change. It is not the author's to widen.
//
// WIDENED TO ALL FOUR BUILT CELLS on the per-cell certification sweep under the amended
// sweep spec: 32/32 per cell, 16 constructions x 2 aux legs, non-vacuous,
// bit-pattern equality against the frozen Triton reference. The built set and
// the certified set are now IDENTICAL, so the N16 arm below is unreachable --
// it stays, because it becomes reachable again the moment a fifth cell is
// built, and a gate that is deleted when it stops firing is a gate that is
// absent when it should fire.
//
// Superseded note, kept for the chain:
// WIDENED to (1,4) and (4,4) on the per-cell certification sweep, spec r2:
// 32/32 each, 16 constructions x 2 legs, non-vacuous, bit-pattern equality
// against the frozen Triton reference, the untouched-tail contract asserted,
// and the NaN-payload divergence escalated under the pre-registered carve-out.
// CERTIFIED UNDER BYPASS: every record carries certified_under_bypass with its
// receipt, which is exactly what the hook exists for -- once per cell, and it
// retires for a cell the moment this predicate covers it.
//
// (1,1) and (4,1) are PENDING one clause: their only failures were N4's
// query_len-dependent expect, ruled case-side by the sweep's owner (spec r3).
// They are NOT here until their re-run evidence is read.
//
// The Python mirror OPUS_CERTIFIED_CELLS must be edited TOGETHER with this.
// SIX cells, on the certification sweep against the refactored build.
// (1,8) is BUILT and deliberately absent here until its re-run, which keeps the
// N16 arm reachable -- the arm that stopped meaning anything the last time the
// built set and the certified set coincided.
// ALL SEVEN BUILT CELLS, on the certification sweep. This is now
// exactly opus_idx_score_cell_built: (H in {1,4}) x (Q in {1,2,4,8}) minus
// (4,8), which H*Q <= 16 rejects first. Keep the two predicates SEPARATE even
// though they coincide today -- they diverge the moment a cell is built ahead
// of its evidence, which is the state this path was in for most of its life.
constexpr bool opus_idx_score_cell_certified(int H, int Q) {
  return (H == 1 && (Q == 1 || Q == 2 || Q == 4 || Q == 8))
         || (H == 4 && (Q == 1 || Q == 2 || Q == 4));
}

} // namespace sparse_attn
} // namespace aiter

// ---------------------------------------------------------------------------
// Per-variant launchers
// ---------------------------------------------------------------------------
#define OPUS_IDX_SCORE_FN(H, Q, A) opus_idx_score_h##H##_q##Q##_aux##A

#define OPUS_IDX_SCORE_DECLARE(H, Q, A) void OPUS_IDX_SCORE_FN(H, Q, A)(OPUS_IDX_SCORE_PARAMS);

#define OPUS_IDX_SCORE_DEFINE(H, Q, A)                                  \
    void OPUS_IDX_SCORE_FN(H, Q, A)(OPUS_IDX_SCORE_PARAMS)              \
    {                                                                   \
        launch_opus_decode_index_score<H, Q, A>(OPUS_IDX_SCORE_ARGS);    \
    }

// ---------------------------------------------------------------------------
// Instantiation table -- (index heads, query tokens, index-K load cache policy)
// ---------------------------------------------------------------------------
// FOURTEEN specialisations are BUILT: seven (H, Q) cells, each on both AUX_K
// legs. (H, Q) is shape-determined by the call, never tuned; AUX_K in {0, 3} is
// the one tuned axis; B = 2 and config 3 are pinned in the kernel template and
// static_asserted there.
//
// Coverage note, executed against the model config rather than assumed
// (integration.md section 8.6): per-rank H is 4 // TP with a replication guard,
// total index heads = 4, so H in {1, 2, 4} across all TP values and H = 8 never
// occurs. H = 2 is a deliberate refusal (N15), not a gap: no bf16-Q
// implementation exists for it. H = 4 is the TP1 / indexer-CP surface; H = 1 is
// production (TP4 and TP8 both).
//
// ALL SEVEN cells are dispatchable today: the built set and the certified set
// coincide, so the N16 bypass is unused. The two predicates stay separate
// anyway, because they diverge the moment a cell is built ahead of its
// evidence.
//
// This block said "eight cells, exactly two dispatchable" until a review caught
// it (finding D10); it had been stale since the table widened, and the widening
// note directly below it was already correct. Two comments about the same fact,
// one of them wrong, is worse than one.
// WIDENED 2026-09-23 on the configuration-coverage measurement
// measured against the model config and the shard geometry: Q = 2 is
// the DEFAULT MTP width and refused on every TP, and Q = 8 is the model's full
// num_mtp_modules = 7. Both are inside the 16-column budget at the H values
// MiniMax-M3 can produce. H = 2 (TP2) is still deliberately absent (N15, a
// human decision). (4, 8) is H*Q = 32 and is NOT a table gap -- it exceeds the
// MFMA column budget and needs a second column tile in the kernel.
//
// The six instantiations added by that widening were built-but-not-certified
// until the sweep produced their per-cell evidence; all seven cells are
// certified now.
#define OPUS_IDX_SCORE_TABLE(F)                                        \
    F(1, 1, 0) F(1, 1, 3) F(1, 2, 0) F(1, 2, 3) F(1, 4, 0) F(1, 4, 3)  \
    F(1, 8, 0) F(1, 8, 3) F(4, 1, 0) F(4, 1, 3) F(4, 2, 0) F(4, 2, 3)  \
    F(4, 4, 0) F(4, 4, 3)

namespace aiter {
namespace sparse_attn {
OPUS_IDX_SCORE_TABLE(OPUS_IDX_SCORE_DECLARE)
} // namespace sparse_attn
} // namespace aiter
