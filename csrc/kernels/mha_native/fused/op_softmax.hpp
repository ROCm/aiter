#pragma once
#include "op_gemm.hpp"

// ================================================================
// op_softmax.hpp — the online-softmax stage of the D64 FMHA fwd kernel
// ================================================================
//
// ROLE IN THE PIPELINE
//   Sits between GEMM0 (S_acc = Q.K^T) and GEMM1 (O_acc += P.V). For each
//   64-column K/V tile the kernel walks, this stage:
//     1. masks out-of-bounds / causal-future scores      (softmax_mask)
//     2. tracks the running per-row max across tiles      (softmax_row_max)
//     3. rescales the carried O_acc when the max grows     (rescale_o_acc)
//     4. turns scores into probabilities P = exp2(...)     (softmax_exp2)
//     5. accumulates the running per-row sum               (softmax_row_sum)
//   Steps 2/3/5 implement Milakov-style online softmax: max and sum are kept
//   incrementally so the kernel never materializes the full score row.
//
// DEFERRED SCALE
//   GEMM0 emits RAW (unscaled) scores. The softmax scale (1/sqrt(d) folded into
//   log2e) is NOT applied here — it is fused into the exp2 argument in
//   softmax_exp2 as exp2(scale*S - scale*m). Carrying S unscaled keeps mask /
//   max bookkeeping in raw units and saves a multiply pass over S_acc.
//
// S_acc distribution (TransposedC, groups-of-8 via SwizzleA):
//   m_row = (lane%32) + 32*warp  — each lane owns ONE M-row
//   n_col(i, k_sub, n_tile) = n_tile*32 + (i/8)*16 + k_sub*8 + (i%8)
//   where i=0..15 (register index), k_sub = lane/32, n_tile=0 or 1
//
// Reduction: 32 lanes in the same k_sub half hold the SAME 32 N-columns.
// k_sub=0: N-cols {0-7, 16-23, 32-39, 48-55}
// k_sub=1: N-cols {8-15, 24-31, 40-47, 56-63}  (complementary)
// So reduction is:
//   1. Intra-lane: reduce over 32 registers (covers one half of N-columns)
//   2. Cross-half: 1 ds_bpermute with lane^32 merges the complementary half
// NO butterfly needed.

// ---- Mask ----
//
// Applies masking to S_acc in-place (scale deferred to exp).
// Boundary mask: if n_col >= seqlen_k (absolute): -INFINITY
// Causal mask: if n_col > m_row + shift: -INFINITY
//
// n_col = kv_offset + (i/8)*16 + k_sub*8 + (i%8)      [for n0 tile]
//       = kv_offset + 32 + (i/8)*16 + k_sub*8 + (i%8)  [for n1 tile]
//
// softmax_mask<HasMask>: set masked S_acc entries to -INF in place.
//   The -INF survives the deferred scale (fmaf(scale, -INF, ...) = -INF) and
//   becomes exp2(-INF) = 0 in softmax_exp2, so masked columns contribute nothing
//   to either the row max or the row sum.
//
//   HasMask is a COMPILE-TIME switch: false = boundary masking only (non-causal),
//   true = causal + boundary. Templating it lets the dead branch fold away.
//
//   Per-lane derivation. This lane owns columns at absolute index
//   col_base + off, where col_base = kv_offset + k_sub*8 and off ranges over the
//   16 register offsets {0..7, 16..23} (n1 adds a further +32). Rather than
//   compare each absolute column to two bounds, we fold both bounds into a single
//   scalar `limit` measured in `off` units, so the per-element test is just
//   off >= limit:
//     - boundary: off < seqlen_k - col_base
//     - causal  : off < (m_row + mask_shift - col_base + 1)
//   `limit` = min of the two (causal only when HasMask).
//
//   Params:
//     s_acc_n0/n1 : the two 32-wide score halves (N-tile 0 / 1), modified in place
//     seqlen_k    : valid K length (boundary bound)
//     kv_offset   : absolute column of this tile's first N-tile-0 entry
//     m_row       : this lane's absolute query row (causal bound)
//     mask_shift  : seqlen_k - seqlen_q, aligns the causal diagonal
//     m_tile_base : min query row of this M-tile (wave-uniform; full-tile guard)

template <bool HasMask>
__device__ __forceinline__ void softmax_mask(
    v16f& s_acc_n0, v16f& s_acc_n1,
    int seqlen_k,
    int kv_offset,
    int m_row,         // this thread's M-row index
    int mask_shift,    // seqlen_k - seqlen_q (for causal)
    int m_tile_base)   // m_tile_idx * kM0 (wave-uniform min row of this M-tile)
{
    const int k_sub = (threadIdx.x & 63) >> 5;

    // Fold both bounds into one `off`-space threshold (see doc block): an element
    // is valid iff its register offset `off` < limit.
    const int col_base = kv_offset + k_sub * 8;
    int limit = seqlen_k - col_base;          // boundary bound in off-units
    if constexpr (HasMask) {
        int causal = m_row + mask_shift - col_base + 1;  // causal bound in off-units
        limit = (causal < limit) ? causal : limit;
    }

    // Full-tile fast path: when the whole 64-column tile is in-bounds, no element
    // is masked. Skipping it removes the per-iteration 32 v_cmp + 30 s_or + 32
    // v_cndmask the compiler emits otherwise. The guard is wave-uniform so this is
    // a single scalar branch. The boundary mask only does work on the last tile
    // (and, for causal, the diagonal tiles), which still take the slow path.
    if constexpr (!HasMask) {
        // Max absolute column in this tile = kv_offset + 63 (k_sub=1, n1, off=23).
        if (kv_offset + kN0 <= seqlen_k)
            return;
    } else {
        // Causal: a tile fully BELOW the diagonal needs no masking. The tightest
        // causal limit is at the topmost row (m_tile_base). If the whole tile is
        // left of that row's diagonal AND in-bounds, every element is valid.
        // Only the diagonal/last edge tile takes the slow path (CK's IsEdgeTile).
        if (kv_offset + kN0 <= m_tile_base + mask_shift + 1 &&
            kv_offset + kN0 <= seqlen_k)
            return;
    }

    // Slow path: one compare per element against `limit` (no scale — deferred to
    // exp). The 16 offsets are the n_col free-dim pattern for this lane's half:
    // (i/8)*16 + (i%8) = {0..7, 16..23}. n1 is the same columns shifted by +32.
    //
    // NOTE: because `limit` is a runtime scalar and `off` a compile-time constant,
    // LLVM lowers this as a serial OR-scan of the per-element predicates rather
    // than a vector compare. That is INTENTIONALLY left simple — the scalar VALU
    // work is hidden behind the neighboring MFMA pipeline and was verified to be a
    // non-issue. The real causal performance lever is block load-balance (heavy
    // M-tiles launched first), handled in the entry .cu files, NOT here.
    constexpr int offsets[16] = {0,1,2,3,4,5,6,7, 16,17,18,19,20,21,22,23};
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        const int off = offsets[i];
        if (off >= limit)
            s_acc_n0[i] = -INFINITY;
        if (off + 32 >= limit)          // n1 column = n0 column + 32
            s_acc_n1[i] = -INFINITY;
    }
}

// ---- Row max: intra-lane max + 1 ds_bpermute cross-half ----
//
// softmax_row_max(v16f&, v16f&, rmax): reduce this row's 64 masked scores to a
// single fp32 max, seeded with the running `rmax` from prior tiles (online
// softmax). Returns the same scalar on every lane of the row (both k_sub halves).
//
// WHY no butterfly. By the TransposedC layout, all 32 lanes of one k_sub half
// already hold the SAME 32 N-columns (they differ only in m_row). So the 64-column
// max for this row is: (intra-lane max over this half's 32 registers) combined
// with (the other half's 32 registers). The complementary half lives on partner
// lane^32, so a SINGLE ds_bpermute exchange suffices — no log2(32) shuffle tree.
//
//   s_acc_n0/n1 : the two masked score halves (read-only)
//   rmax        : running max carried in from previous K/V tiles (-INF at start)

__device__ __forceinline__ float softmax_row_max(
    const v16f& s_acc_n0,
    const v16f& s_acc_n1,
    float rmax = -INFINITY)
{
    // Intra-lane max over 32 registers + previous rmax (avoids -INF when all masked)
    float local_max = rmax;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        local_max = fmaxf(local_max, s_acc_n0[i]);
        local_max = fmaxf(local_max, s_acc_n1[i]);
    }

    // Cross-half exchange: merge the complementary 32 columns held by lane^32.
    // partner = same warp, k_sub flipped (bit 5 of the 6-bit lane id toggled).
    int partner = (threadIdx.x & ~63) | ((threadIdx.x & 63) ^ 32);
    float other = bpermute_f32(partner, local_max);
    return fmaxf(local_max, other);
}

// ---- Exp2: P = exp2(scale * S - scale * m_new) ----
//
// softmax_exp2: convert raw masked scores S into probabilities P, in place.
// This is where the DEFERRED softmax scale is finally applied, fused with the
// max-subtraction into one FMA: arg = fmaf(scale, S, -scale_m) = scale*(S - m).
// Then P = exp2(arg) via the hardware exp2.
//   - scale is log2e-based (the 1/sqrt(d) factor folded into log2e), so exp2 of
//     a log2-domain argument yields the natural-base softmax weight.
//   - scale_m = scale * m_new is precomputed by the caller (max already in
//     log2 domain), so each element costs one v_fma_f32 (1-cycle, guaranteed
//     fused) + one v_exp_f32 — no separate subtract/multiply pass over S.
//   - Masked entries: fmaf(scale, -INF, -scale_m) = -INF, exp2(-INF) = 0, so
//     they drop out of the subsequent row sum / P.V GEMM automatically.
//
//   s_acc_n0/n1 : scores in, probabilities P out (in place)
//   scale       : log2e-based softmax scale
//   scale_m     : scale * running_max (the shift, precomputed by caller)

__device__ __forceinline__ void softmax_exp2(
    v16f& s_acc_n0, v16f& s_acc_n1,
    float scale, float scale_m)
{
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        s_acc_n0[i] = __builtin_amdgcn_exp2f(fmaf(scale, s_acc_n0[i], -scale_m));
        s_acc_n1[i] = __builtin_amdgcn_exp2f(fmaf(scale, s_acc_n1[i], -scale_m));
    }
}

// ---- Row sum: intra-lane sum + 1 ds_bpermute cross-half ----
//
// softmax_row_sum: reduce this row's 64 probabilities P to a single fp32 sum,
// returned identically on every lane of the row. Same layout argument as
// softmax_row_max (one ds_bpermute, no butterfly) — see that doc block. The
// caller adds this into the running denominator across tiles; the final O is
// divided by it in the epilogue. Masked entries are already 0 (from exp2(-INF)),
// so they add nothing.
//
//   p_n0/n1 : the two probability halves (read-only)

__device__ __forceinline__ float softmax_row_sum(
    const v16f& p_n0,
    const v16f& p_n1)
{
    // Intra-lane sum over 32 P values
    float local_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        local_sum += p_n0[i];
        local_sum += p_n1[i];
    }

    // Cross-half exchange: add in the complementary 32 columns held by lane^32.
    int partner = (threadIdx.x & ~63) | ((threadIdx.x & 63) ^ 32);
    float other = bpermute_f32(partner, local_sum);
    return local_sum + other;
}

// ---- Rescale O_acc when max changes between tiles ----
//
// rescale_o_acc: the online-softmax correction. When a new K/V tile raises the
// running row max from old_max to new_max, every probability computed against the
// OLD max is too large by exp2(old_max - new_max); the carried O_acc (= sum of
// P_old . V) inherits that same factor and must be scaled down before this tile's
// P.V is added. (The running sum is corrected the same way by the caller.)
// new_max >= old_max, so 0 < factor <= 1.
//
//   o_acc_d0/d1 : the two hdim halves of the output accumulator, scaled in place
//
// Overload 1: factor already computed (lets the caller share exp2 across uses).
__device__ __forceinline__ void rescale_o_acc(
    v16f& o_acc_d0, v16f& o_acc_d1,
    float factor)
{
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        o_acc_d0[i] *= factor;
        o_acc_d1[i] *= factor;
    }
}

// Overload 2: compute the factor from old/new max (log2 domain) then rescale.
__device__ __forceinline__ void rescale_o_acc(
    v16f& o_acc_d0, v16f& o_acc_d1,
    float old_max, float new_max)
{
    rescale_o_acc(o_acc_d0, o_acc_d1, __builtin_amdgcn_exp2f(old_max - new_max));
}

// ================================================================
// Legacy functions — DEAD CODE, kept for reference (no live caller).
//
// These are the PRE-online-softmax variants. They differ from the live path
// above in two ways: (1) they produce/consume a per-register array (one max/sum
// per register slot instead of a single per-row scalar), and (2) they reduce with
// a full log2(32) butterfly of ds_bpermutes instead of the single lane^32
// exchange the TransposedC layout makes sufficient. They were called by an old
// `_device.hpp` entry that has since been removed; nothing in this kernel
// references them now (the live pipeline.hpp uses the scalar softmax_row_max /
// softmax_row_sum / softmax_exp2 above). `inline` so they emit no code while
// uncalled. New readers can skip this block.
// ================================================================

// Old softmax_row_max with array output (no live caller; see banner above)
__device__ __forceinline__ void softmax_row_max(float (&row_max)[16],
                                       const v16f& s_acc_n0,
                                       const v16f& s_acc_n1,
                                       int lane_id) {
    int k_sub = lane_id >> 5;
    int base_lane = k_sub * 32;

    for (int i = 0; i < 16; i++) {
        float local_max = fmaxf(s_acc_n0[i], s_acc_n1[i]);
        for (int offset = 16; offset >= 1; offset >>= 1) {
            int src = base_lane | ((lane_id & 31) ^ offset);
            float other = bpermute_f32(src, local_max);
            local_max = fmaxf(local_max, other);
        }
        row_max[i] = local_max;
    }
}

// Old softmax_exp with array input
__device__ __forceinline__ void softmax_exp(v16f& s_acc_n0,
                                   v16f& s_acc_n1,
                                   const float (&row_max)[16]) {
    for (int i = 0; i < 16; i++) {
        if (row_max[i] == -INFINITY) {
            s_acc_n0[i] = 0.0f;
            s_acc_n1[i] = 0.0f;
        } else {
            s_acc_n0[i] = __builtin_amdgcn_exp2f(s_acc_n0[i] - row_max[i]);
            s_acc_n1[i] = __builtin_amdgcn_exp2f(s_acc_n1[i] - row_max[i]);
        }
    }
}

// Old softmax_row_sum with array output
__device__ __forceinline__ void softmax_row_sum(float (&row_sum)[16],
                                       const v16f& s_acc_n0,
                                       const v16f& s_acc_n1,
                                       int lane_id) {
    int k_sub = lane_id >> 5;
    int base_lane = k_sub * 32;

    for (int i = 0; i < 16; i++) {
        float local_sum = s_acc_n0[i] + s_acc_n1[i];
        for (int offset = 16; offset >= 1; offset >>= 1) {
            int src = base_lane | ((lane_id & 31) ^ offset);
            float other = bpermute_f32(src, local_sum);
            local_sum += other;
        }
        row_sum[i] = local_sum;
    }
}

// Old rescale_o_acc with array input
__device__ __forceinline__ void rescale_o_acc(v16f& o_acc_n0, v16f& o_acc_n1,
                                     const float (&old_max)[16],
                                     const float (&new_max)[16]) {
    for (int i = 0; i < 16; i++) {
        float factor = __builtin_amdgcn_exp2f(old_max[i] - new_max[i]);
        o_acc_n0[i] *= factor;
        o_acc_n1[i] *= factor;
    }
}
