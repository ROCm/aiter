#pragma once
#include <hip/hip_runtime.h>
#include "runner/params.hpp"
#include "op_lds.hpp"
#include "op_gemm.hpp"
#include "op_softmax.hpp"
#include "op_epilog.hpp"

// pipeline.hpp — the per-block forward pass (heart of the kernel)
//
// fmha_fwd_d64_device<HasMask,IsVarlen,IsSplit> IS the whole FMHA forward pass for
// one M-tile (kM0=128 query rows of one batch/head). The split __global__ entries
// decode blockIdx and call this. It orchestrates op_lds/op_gemm/op_softmax/op_epilog
// into a software-pipelined loop over KV tiles.
//
// END-TO-END FLOW (one block):
//   1. SETUP   — decode geometry; resolve Q/K/V/O bases + seqlens (dense/varlen);
//                build buffer SRDs; derive causal loop bound (seqlen_k_end).
//   2. Q LOAD  — each thread loads its slice of the 128xkHeadDim Q tile ONCE
//                (q_regs[4]); Q is reused for every KV tile.
//   3. PROLOGUE— issue the first K sub-tile async copy so GEMM0 has data.
//   4. TILE LOOP over KV tiles (kN0=64 keys each):
//        GEMM0   S_acc = Q . K^T          (reads K from LDS, Q from reg)
//        SOFTMAX mask -> row_max -> exp2 -> row_sum  (online)
//        V STAGE DRAM -> regs -> v_perm shuffle -> LDS
//        ONLINE  rescale carried O_acc by exp2(scale*(old_max-new_max)) when the
//                running max grew; correct the running sum likewise
//        GEMM1   O_acc += P . V           (reads V from LDS, P from reg)
//      while prefetching the next tile's K copy and the second V half to overlap HBM.
//   5. EPILOGUE— normalize O_acc by the final sum, bf16-truncate, store O, opt LSE.
//
// ONLINE SOFTMAX (Milakov), carried across tiles in three per-row scalars:
//     rmax — running max of scaled scores
//     rsum — running denominator (sum of exp2 probs)
//     o_acc_d0/d1 — running numerator (sum of P.V), TransposedC layout
//   When a tile raises the max rmax->m_new, earlier contributions are too large by
//   exp2(scale*(rmax-m_new)); rescale o_acc and rsum by it BEFORE adding this tile,
//   so the result equals a single global softmax.
//
// LDS DOUBLE/TRIPLE BUFFERING via LdsSeq[] — see the constant's comment below.
//
// sched_barrier() CALLS mirror CK's barrier structure one-for-one; their purpose is
//   codegen/parity (pin the compiler's scheduling to CK's so the ISA/numerics line
//   up), NOT perf — verified ~0% on their own. The mask restricts what may move
//   across the barrier (0 = full barrier; 0x1/0x7/0x7F progressively allow certain
//   instruction classes, fencing MFMA vs VALU as CK does). Not a tuning knob.
//
// THREAD GEOMETRY (shared with op_*.hpp):
//   warp_id = threadIdx.x>>6 (0..3); lane_id = threadIdx.x&63 (0..63);
//   k_sub = lane_id>>5 (0/1, 32-lane half); m_row = (lane_id&31)+32*warp_id is this
//   lane's query row within the M-tile (TransposedC: one M-row per lane).

// Build an untyped byte buffer SRD over a DRAM tensor base. num_records is the VALID
// byte extent; the HW bounds-check returns 0 for any access at or beyond it.
// Load-bearing for the partial tail tile (seqlen_k % kN0 != 0): the loop walks a
// full kN0-wide tile, so padding rows (row >= seqlen_k) read PAST the tensor.
// Unbounded, those reads return adjacent memory (e.g. a freed NaN block) and the
// masked-but-still-summed P(=0)*V term computes 0*NaN = NaN in GEMM1, poisoning
// O_acc; bounding makes them return 0. 0x00027000 is the CDNA raw-byte-buffer
// data-format word for the raw_buffer_load builtins.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_buffer_resource(const void* base,
                                                                       unsigned num_records) {
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<void*>(base), 0, num_records, 0x00027000);
}

// Clamp a 64-bit byte extent to the 32-bit num_records field. Real test tensors
// are far under 4 GiB per (b,h) region; the clamp only matters for pathologically
// large tensors, where it degrades back to "no bounds check" rather than truncating
// a valid access.
__device__ __forceinline__ unsigned clamp_num_records(int64_t bytes) {
    return (bytes < (int64_t)0xFFFFFFFFu) ? (unsigned)bytes : 0xFFFFFFFFu;
}

// LDS buffer rotation for the four staging slots of one tile iteration (3-buffer
// scheme, buf_idx in {0,1,2}). LdsSeq maps each logical slot to a physical buffer:
//   LdsSeq[0] = K sub-tile 0 (consumed by GEMM0.0; also where the NEXT tile's
//               prefetched K lands)
//   LdsSeq[1] = K sub-tile 1 (consumed by GEMM0.1)
//   LdsSeq[2] = V half 0     (staged for GEMM1.0)
//   LdsSeq[3] = V half 1     (staged for GEMM1.1)
// {1,2,1,0} keeps GEMM0's K and GEMM1's V in different buffers so producer and
// consumer never alias within an iteration (reusing buffer 1 for both K halves is
// safe: GEMM0 finishes sub-tile 0 before sub-tile 1 is needed).
constexpr int LdsSeq[4] = {1, 2, 1, 0};

// One block's full FMHA forward pass over its M-tile.
//   HasMask  : compile-time. false = boundary mask only; true = causal+boundary.
//   IsVarlen : compile-time. false = dense batch tensors; true = group/varlen.
//   IsSplit  : compile-time. false (DEFAULT) = ordinary full forward (split branches
//              if-constexpr-discarded). true = split-K: walk only this split's
//              disjoint KV sub-range and write a normalized fp32 partial (O_g, LSE_g)
//              to split-major scratch via epilog_store_split.
//   params   : tensor pointers, strides, scale, optional LSE/seqstart arrays.
//   lds      : this block's __shared__ scratch (kLdsBytes; the 3 rotating buffers).
//   batch_idx/head_idx/m_tile_idx : tile coordinates (from blockIdx; causal M-tile
//                                   reversal already applied in the entry .cu files).
//   --- TRAILING split-only args (defaulted so a non-split call can omit them) ---
//   scratch_o   : split-major fp32 partial-O scratch base (IsSplit only).
//   scratch_lse : split-major fp32 LSE scratch base (IsSplit only).
//   num_splits  : G — the KV axis is partitioned into G disjoint ranges.
//   split_idx   : which of the G splits this block handles (0..G-1).
template <bool HasMask, bool IsVarlen, bool IsSplit = false>
__device__ __forceinline__ void fmha_fwd_d64_device(const FmhaFwdParams& params,
                                    char* lds,
                                    int batch_idx,
                                    int head_idx,
                                    int m_tile_idx,
                                    float* scratch_o = nullptr,
                                    float* scratch_lse = nullptr,
                                    int num_splits = 1,
                                    int split_idx = 0) {
    // ---- Thread geometry (TransposedC; see file header / op_gemm.hpp) ----
    const int lane_id = threadIdx.x & 63;
    const int warp_id = threadIdx.x >> 6;
    const int k_sub   = lane_id >> 5;                  // 32-lane half (0/1)
    const int m_row   = (lane_id & 31) + 32 * warp_id; // this lane's query row in tile

    // GQA/MQA: several Q heads can share one K/V head. Map this Q head to its KV head (nhead_ratio==1 for full MHA).
    const int nhead_ratio = params.nhead_q / params.nhead_k;
    const int kv_head_idx = head_idx / nhead_ratio;

    // ---- Resolve per-sequence lengths and the row offset into the tensors ----
    // Varlen (group mode): sequences are packed back-to-back; seqstart_*[b] is the
    // running row offset and the length is the gap to the next start. Dense mode
    // uses uniform seqlens and addresses by batch stride later.
    int seqlen_q, seqlen_k;
    int offset_q = 0, offset_k = 0;
    if constexpr (IsVarlen) {
        offset_q = params.seqstart_q[batch_idx];
        offset_k = params.seqstart_k[batch_idx];
        seqlen_q = params.seqstart_q[batch_idx + 1] - offset_q;
        seqlen_k = params.seqstart_k[batch_idx + 1] - offset_k;
        // This M-tile starts past the end of this (short) sequence: nothing to do.
        // Cheap early-out; dense mode cannot hit it (m_tiles sized to seqlen_q).
        if (m_tile_idx * kM0 >= seqlen_q) return;
    } else {
        seqlen_q = params.seqlen_q;
        seqlen_k = params.seqlen_k;
    }

    // ---- Base pointers for this (batch, head) ----
    // Varlen indexes rows via offset_* (no batch stride; sequences are packed).
    // Dense indexes via batch_stride_* then nhead_stride_*. K/V use kv_head_idx
    // (GQA); Q/O use the full head_idx. int64 math avoids overflow on big tensors.
    const __hip_bfloat16* q_base;
    const __hip_bfloat16* k_base;
    const __hip_bfloat16* v_base;
    __hip_bfloat16* o_base;
    if constexpr (IsVarlen) {
        q_base = params.q + static_cast<int64_t>(head_idx)    * params.nhead_stride_q
                           + static_cast<int64_t>(offset_q)   * params.stride_q;
        k_base = params.k + static_cast<int64_t>(kv_head_idx) * params.nhead_stride_k
                           + static_cast<int64_t>(offset_k)   * params.stride_k;
        v_base = params.v + static_cast<int64_t>(kv_head_idx) * params.nhead_stride_v
                           + static_cast<int64_t>(offset_k)   * params.stride_v;
        o_base = params.o + static_cast<int64_t>(head_idx)    * params.nhead_stride_o
                           + static_cast<int64_t>(offset_q)   * params.stride_o;
    } else {
        q_base = params.q + static_cast<int64_t>(batch_idx) * params.batch_stride_q
                           + static_cast<int64_t>(head_idx)  * params.nhead_stride_q;
        k_base = params.k + static_cast<int64_t>(batch_idx) * params.batch_stride_k
                           + static_cast<int64_t>(kv_head_idx) * params.nhead_stride_k;
        v_base = params.v + static_cast<int64_t>(batch_idx) * params.batch_stride_v
                           + static_cast<int64_t>(kv_head_idx) * params.nhead_stride_v;
        o_base = params.o + static_cast<int64_t>(batch_idx) * params.batch_stride_o
                           + static_cast<int64_t>(head_idx)  * params.nhead_stride_o;
    }

    // Buffer SRDs the raw_buffer_load builtins read through (O's SRD is built in the
    // epilogue). Each is bounded to the valid byte extent of this (b,h) region so
    // partial-tail-tile padding rows read 0, not OOB garbage. Extent = #rows *
    // row_stride(elements) * 2 bytes.
    auto srd_q = make_buffer_resource(
        q_base, clamp_num_records((int64_t)seqlen_q * params.stride_q * 2));
    auto srd_k = make_buffer_resource(
        k_base, clamp_num_records((int64_t)seqlen_k * params.stride_k * 2));
    auto srd_v = make_buffer_resource(
        v_base, clamp_num_records((int64_t)seqlen_k * params.stride_v * 2));

    // ---- KV loop bounds ----
    // mask_shift aligns the causal diagonal when seqlen_k != seqlen_q: query row r
    // may attend keys with column <= r + mask_shift (CK convention: the last query
    // attends the last key). Non-causal walks all of seqlen_k.
    int seqlen_k_start = 0;
    int seqlen_k_end   = seqlen_k;
    int mask_shift = seqlen_k - seqlen_q;

    if constexpr (HasMask) {
        // Causal: skip every KV tile entirely PAST this M-tile's diagonal (all
        // masked). Derivation:
        //   last_q_row  = highest query row this M-tile owns (clamped to seqlen_q)
        //   raw_end     = last column that row may attend = last_q_row+mask_shift+1
        //   seqlen_k_end= raw_end rounded UP to a whole kN0 tile (diagonal tile still
        //                 processed; softmax_mask handles its partial masking),
        //                 clamped to seqlen_k.
        // With heavy-first M-tile reversal (entry .cu), causal cost is ~linear in m_tile.
        int last_q_row = m_tile_idx * kM0 + kM0 - 1;
        if (last_q_row >= seqlen_q) last_q_row = seqlen_q - 1;
        int raw_end = last_q_row + mask_shift + 1;
        if (raw_end > seqlen_k) raw_end = seqlen_k;
        seqlen_k_end = ((raw_end + kN0 - 1) / kN0) * kN0;
        if (seqlen_k_end > seqlen_k) seqlen_k_end = seqlen_k;
        seqlen_k_start = 0;
    }

    // Number of kN0(=64)-key tiles this block walks.
    int num_total_loop = (seqlen_k_end - seqlen_k_start + kN0 - 1) / kN0;

    // ---- SPLIT-K KV-range narrowing (IsSplit=true ONLY) ----
    // Partition the FULL tile count into G contiguous chunks, keep THIS split's:
    //   T (tiles per split) = ceil(num_total_loop_full / num_splits)
    //   this split owns tiles [split_idx*T, min((split_idx+1)*T, full))
    // Translating tiles -> keys (×kN0) narrows WITHIN the already-causal-clamped
    // [seqlen_k_start, seqlen_k_end) range, so causally-excluded future tiles stay
    // excluded (A4). The kv_offset / kv_v_byte / kv_k_byte IVs init from
    // seqlen_k_start, so they pick up the split's start key automatically. An empty
    // split (start >= full) leaves num_total_loop <= 0 -> the degenerate path below
    // fires (writing the fp32 -inf/0 sentinel plane via epilog_store_split).
    if constexpr (IsSplit) {
        int num_total_loop_full = num_total_loop;
        int tiles_per_split = (num_total_loop_full + num_splits - 1) / num_splits;
        int tile_lo = split_idx * tiles_per_split;
        int tile_hi = tile_lo + tiles_per_split;
        if (tile_hi > num_total_loop_full) tile_hi = num_total_loop_full;
        // Narrow within the existing (possibly causal-clamped) range.
        int base_start = seqlen_k_start;
        seqlen_k_start = base_start + tile_lo * kN0;
        seqlen_k_end   = base_start + tile_hi * kN0;
        if (seqlen_k_end > seqlen_k) seqlen_k_end = seqlen_k;
        if (seqlen_k_start > seqlen_k_end) seqlen_k_start = seqlen_k_end; // empty split
        num_total_loop = (seqlen_k_end - seqlen_k_start + kN0 - 1) / kN0;
    }

    // O accumulator (numerator of online softmax): two kHeadDim/2 halves in the
    // TransposedC layout, carried across all KV tiles. Start at zero.
    v16f o_acc_d0, o_acc_d1;
    clear_acc(o_acc_d0);
    clear_acc(o_acc_d1);

    // ---- SPLIT-K scratch row-plane base pointers (IsSplit=true ONLY) ----
    // Resolve, for THIS (split_idx, b, h), the Sq×64 fp32 partial-O plane and Sq
    // fp32 LSE plane in the split-major scratch:
    //   scratch_o_base  = scratch_o  + (((split_idx*B + b)*Hq + h)*Sq)*64
    //   scratch_lse_base= scratch_lse + (((split_idx*B + b)*Hq + h)*Sq)
    // epilog_store_split adds the in-plane row/col. Hq == params.nhead_q; B is not a
    // kernarg field (split grid z-axis is batch*num_splits, so B = gridDim.z /
    // num_splits).
    float* scratch_o_base   = nullptr;
    float* scratch_lse_base = nullptr;
    if constexpr (IsSplit) {
        const int Hq = params.nhead_q;
        const int Sq = params.seqlen_q;
        const int B  = gridDim.z / num_splits;
        const int64_t plane = (((static_cast<int64_t>(split_idx) * B + batch_idx)
                                 * Hq + head_idx) * Sq);
        scratch_o_base   = scratch_o   + plane * kHeadDim;
        scratch_lse_base = scratch_lse + plane;
    }

    // Degenerate tile (causal M-tile fully masked, or varlen tail): no KV work.
    // Emit a zeroed O row with LSE=-inf and return. For a split this is ALSO the
    // empty-split path and must write the fp32 -inf/0 sentinel plane via
    // epilog_store_split (A4: a masked-future split still owns its scratch plane),
    // NOT the bf16 epilog_store.
    if (num_total_loop <= 0) {
        if constexpr (IsSplit) {
            epilog_store_split(o_acc_d0, o_acc_d1, 0.0f, -INFINITY, params.scale,
                               seqlen_q, m_tile_idx, scratch_o_base, scratch_lse_base);
        } else {
            float* lse_base = nullptr;
            if (params.lse) {
                if constexpr (IsVarlen) {
                    int nhead_stride_lse = params.nhead_stride_q / params.stride_q;
                    lse_base = params.lse + static_cast<int64_t>(head_idx) * nhead_stride_lse + offset_q;
                } else {
                    lse_base = params.lse
                        + static_cast<int64_t>(batch_idx) * (params.nhead_q * params.seqlen_q)
                        + static_cast<int64_t>(head_idx) * params.seqlen_q;
                }
            }
            epilog_store(o_acc_d0, o_acc_d1, 0.0f, -INFINITY, params.scale,
                         params.stride_o, lse_base, seqlen_q, m_tile_idx, o_base);
        }
        return;
    }

    // ---- Q LOAD (once; reused for every KV tile) ----
    // This lane's absolute query row, and Q's row stride in bytes.
    const int abs_m_row = m_tile_idx * kM0 + m_row;
    const int q_stride_bytes = params.stride_q * 2;

    // Load this lane's full kHeadDim(=64) Q slice as 4x b128 (4 dwords = 8 bf16).
    // Per TransposedC this lane owns headdim hd = kstep*16 + k_sub*8 + (0..7) in
    // q_regs[kstep]; slice_q() hands the right pair to each GEMM0 sub-tile.
    // Out-of-range query rows (last M-tile padding) load zeros.
    v4i q_regs[4];
    if (abs_m_row < seqlen_q) {
        #pragma unroll
        for (int kstep = 0; kstep < 4; ++kstep) {
            int hd = kstep * 16 + k_sub * 8;
            int voff = abs_m_row * q_stride_bytes + hd * 2;
            q_regs[kstep] = __builtin_amdgcn_raw_buffer_load_b128(srd_q, voff, 0, 0);
        }
    } else {
        #pragma unroll
        for (int kstep = 0; kstep < 4; ++kstep)
            q_regs[kstep] = v4i{0, 0, 0, 0};
    }

    // Finite rmax seed below any realizable raw score, so a real score always wins
    // the running max; -inf would make a fully-masked row NaN instead of O=0/LSE=-inf.
    float rmax = -1e30f;
    float rsum = 0.0f;

    // kv_offset = absolute key row of the current tile's first key.
    // k_col_offset = which kK0(=32) headdim half of K to stage next (0 then 32).
    int kv_offset = seqlen_k_start;
    int k_col_offset = 0;

    // V byte-base induction variable: kv_offset pre-multiplied into the V row stride
    // (bytes) so per-tile address math is a constant add, not a multiply. Wave-uniform
    // -> stays in an SGPR.
    const int v_stride_bytes = params.stride_v * 2;
    int kv_v_byte = kv_offset * v_stride_bytes;

    // K byte-base induction variable: same transform, for the async DRAM->LDS K copies. Also wave-uniform -> SGPR.
    const int k_stride_bytes = params.stride_k * 2;
    int kv_k_byte = kv_offset * k_stride_bytes;

    // After Q load, before K prefetch — match CK prologue barriers 1-2.
    // (sched_barriers are codegen/parity fences, ~0% perf — see file header.)
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_sched_barrier(0);

    // ---- PROLOGUE: kick off the first K sub-tile copy (headdim half 0) so the
    // first GEMM0 has data. Async (vmcnt only); the loop fences it before reading.
    async_copy_k_subtile(lds, srd_k, params.stride_k, kv_k_byte, k_col_offset, LdsSeq[0]);
    k_col_offset += kK0;

    __builtin_amdgcn_sched_barrier(0); // prologue barrier 3

    // ================= TILE LOOP over KV tiles =================
    int i_total_loops = 0;
    __builtin_amdgcn_sched_barrier(0); // prologue barrier 4
    do {
        // ---- GEMM0: S_acc = Q . K^T for this tile (two 32-wide N halves) ----
        v16f s_acc_n0, s_acc_n1;
        clear_acc(s_acc_n0);
        clear_acc(s_acc_n1);

        {
            // Prefetch K headdim half 1 (buffer LdsSeq[1]) while half 0 is still
            // in flight, then drain to <=4 outstanding and barrier so half 0 is
            // visible, and run GEMM0 sub-tile 0 (consumes half 0 from LdsSeq[0]).
            async_copy_k_subtile(lds, srd_k, params.stride_k, kv_k_byte, k_col_offset, LdsSeq[1]);
            k_col_offset += kK0;
            async_load_fence(4);
            s_barrier();
            __builtin_amdgcn_sched_barrier(0); // hot-loop barrier 5 — GEMM0 entry
            gemm0_subtile(s_acc_n0, s_acc_n1, slice_q(q_regs, 0), lds, LdsSeq[0]);
        }

        {
            // Drain K half 1 and barrier so it is visible, then start V loading
            // from DRAM into registers (overlapping GEMM0 sub-tile 1's MFMA) and
            // run GEMM0 sub-tile 1 (consumes half 1 from LdsSeq[1]) to finish S_acc.
            async_load_fence(0);
            s_barrier();
            __builtin_amdgcn_sched_barrier(0); // CK barrier 2 — after s_barrier, before V-load + GEMM0.1
            v2i v_k3_0, v_k3_1;
            load_v_from_dram(v_k3_0, v_k3_1, srd_v, params.stride_v, kv_v_byte);
            __builtin_amdgcn_sched_barrier(0); // CK barrier 3 — after V-load, before GEMM0.1
            gemm0_subtile(s_acc_n0, s_acc_n1, slice_q(q_regs, 1), lds, LdsSeq[1]);
            __builtin_amdgcn_sched_barrier(0x1); // CK barrier 4 — GEMM0 exit, VALU-only

            // ---- SOFTMAX part 1: mask + running row max ----
            // scale is deferred: GEMM0 emitted RAW scores; mask/max work in raw
            // units and the scale is fused into the exp2 below (see op_softmax.hpp).
            // softmax_mask sets out-of-bounds / causal-future entries to -INF.
            // softmax_row_max folds this tile's masked scores into the running rmax,
            // returning the new per-row max m_new (>= rmax).
            float scale_s = params.scale;
            softmax_mask<HasMask>(s_acc_n0, s_acc_n1,
                                  seqlen_k, kv_offset, abs_m_row, mask_shift,
                                  m_tile_idx * kM0);
            float m_new = softmax_row_max(s_acc_n0, s_acc_n1, rmax);
            __builtin_amdgcn_sched_barrier(0x7F); // CK barrier 5 — after bpermute, all non-MFMA

            // ---- V STAGING (slotted between row_max and exp so its LDS write +
            // the next half's DRAM load overlap the upcoming exp/sum/GEMM1) ----
            // Drain the V regs, shuffle+store V half 0 into LDS (LdsSeq[2]) for GEMM1,
            // then start loading V half 1 (rows +32).
            s_waitcnt_vmcnt_0();
            store_v_to_lds(v_k3_0, v_k3_1, lds, LdsSeq[2]);
            v2i v1_k3_0, v1_k3_1;
            load_v_from_dram(v1_k3_0, v1_k3_1, srd_v, params.stride_v, kv_v_byte + 32 * v_stride_bytes);
            // v1 load left in flight: consumed by store_v_to_lds at the end of GEMM1
            // (guarded there). Draining now would expose HBM latency instead of
            // overlapping it with the exp2 / row_sum / rescale / GEMM1 that follows.

            __builtin_amdgcn_sched_barrier(0); // CK barrier 6 — after V-staging, before O-rescale + GEMM1

            // ---- SOFTMAX part 2 + ONLINE update + GEMM1 (one scheduling region
            // so the compiler interleaves the VALU exp/pack with GEMM1's MFMA) ----
            // exp2 turns scores into probabilities P = exp2(scale*(S - m_new)),
            // applying the deferred scale; row_sum reduces this tile's P to l_new.
            float scale_m = scale_s * m_new;
            softmax_exp2(s_acc_n0, s_acc_n1, scale_s, scale_m);
            float l_new = softmax_row_sum(s_acc_n0, s_acc_n1);

            // ONLINE-SOFTMAX correction: if the running max grew (rmax -> m_new),
            // every prior contribution used too-large probabilities by the factor
            // exp2(scale*(rmax-m_new)) (in (0,1]). Rescale the carried numerator
            // o_acc and denominator rsum by it BEFORE folding in this tile, then
            // advance the running max. (m_new==rmax => factor 1 => no-op.)
            float rescale = __builtin_amdgcn_exp2f(scale_s * (rmax - m_new));
            rescale_o_acc(o_acc_d0, o_acc_d1, rescale);
            rsum = rescale * rsum + l_new;
            rmax = m_new;

            // (P fp32->bf16 truncation is done inline by gemm1_subtile's v_perm_b32.)

            // ---- GEMM1 sub-tile 0: O_acc += P_n0 . V_half0 ----
            // block_sync_lds() makes V half 0 (just stored) visible to all waves.
            // After the MFMA, shuffle+store V half 1 into LDS (LdsSeq[3]) for sub-tile 1.
            {
                block_sync_lds();
                gemm1_subtile(o_acc_d0, o_acc_d1, s_acc_n0, lds, LdsSeq[2]);
                s_waitcnt_vmcnt_0();
                store_v_to_lds(v1_k3_0, v1_k3_1, lds, LdsSeq[3]);
            }

            // Advance to the next tile and PREFETCH its K half 0 (into LdsSeq[0],
            // the buffer GEMM0 reads first next iteration) so the copy overlaps
            // this iteration's remaining GEMM1. Skipped on the last iteration.
            i_total_loops++;
            if (i_total_loops < num_total_loop) {
                kv_offset += kN0;
                kv_v_byte += kN0 * v_stride_bytes;   // advance V byte-base by a loop constant
                kv_k_byte += kN0 * k_stride_bytes;   // advance K byte-base (used by the prefetch below)
                k_col_offset = 0;
                s_barrier();
                async_copy_k_subtile(lds, srd_k, params.stride_k, kv_k_byte, k_col_offset, LdsSeq[0]);
                k_col_offset += kK0;
            }

            // ---- GEMM1 sub-tile 1: O_acc += P_n1 . V_half1 (LdsSeq[3]) ----
            {
                block_sync_lds();
                gemm1_subtile(o_acc_d0, o_acc_d1, s_acc_n1, lds, LdsSeq[3]);
            }


        }

    } while (i_total_loops < num_total_loop);

    // ---- EPILOGUE: normalize O_acc by rsum, store O (+LSE) ----
    // For a split (IsSplit=true) write the NORMALIZED fp32 partial (O_g, LSE_g) to
    // this split's scratch plane via epilog_store_split; the combine pass folds the
    // G partials later. For IsSplit=false the else-branch is the bf16 epilogue.
    if constexpr (IsSplit) {
        epilog_store_split(o_acc_d0, o_acc_d1, rsum, rmax, params.scale,
                           seqlen_q, m_tile_idx, scratch_o_base, scratch_lse_base);
    } else {
        // Resolve the LSE output base for this (batch/varlen, head). Varlen: packed
        // like Q (nhead_stride from Q's element strides + seq offset); dense:
        // [batch][head][seqlen_q]. nullptr if LSE not requested.
        float* lse_base = nullptr;
        if (params.lse) {
            if constexpr (IsVarlen) {
                int nhead_stride_lse = params.nhead_stride_q / params.stride_q;
                lse_base = params.lse + static_cast<int64_t>(head_idx) * nhead_stride_lse + offset_q;
            } else {
                lse_base = params.lse
                    + static_cast<int64_t>(batch_idx) * (params.nhead_q * params.seqlen_q)
                    + static_cast<int64_t>(head_idx) * params.seqlen_q;
            }
        }

        // Hand the final running numerator (o_acc), denominator (rsum) and max (rmax)
        // to the epilogue, which divides, truncates to bf16, and writes O/LSE to DRAM.
        epilog_store(o_acc_d0, o_acc_d1, rsum, rmax, params.scale,
                     params.stride_o, lse_base, seqlen_q, m_tile_idx, o_base);
    }
}
