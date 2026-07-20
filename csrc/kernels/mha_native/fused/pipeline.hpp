#pragma once
#include <hip/hip_runtime.h>
#include "runner/params.hpp"
#include "op_lds.hpp"
#include "op_gemm.hpp"
#include "op_softmax.hpp"
#include "op_epilog.hpp"

// ================================================================
// pipeline.hpp — the per-block forward pass (heart of the kernel)
// ================================================================
//
// ROLE IN THE PIPELINE
//   fmha_fwd_d64_device<HasMask,IsVarlen,IsSplit> IS the whole FMHA forward pass
//   for one M-tile (kM0=128 query rows of one batch/head). The split __global__
//   entries (fmha_fwd_d64_bf16_msk{0,1}_split) are thin shells that decode
//   blockIdx and call this. Everything below orchestrates the helpers in
//   op_lds.hpp / op_gemm.hpp / op_softmax.hpp / op_epilog.hpp into a
//   software-pipelined loop over the KV tiles.
//
// END-TO-END FLOW (one block):
//   1. SETUP   — decode lane/warp geometry; resolve Q/K/V/O base pointers and
//                seqlens for dense vs varlen; build buffer SRDs; derive the causal
//                loop bound (seqlen_k_end).
//   2. Q LOAD  — each thread loads its slice of the 128xkHeadDim Q tile into
//                registers ONCE (q_regs[4]); Q is reused for every KV tile.
//   3. PROLOGUE— issue the first K sub-tile async copy into LDS so GEMM0 of the
//                first iteration has data to read.
//   4. TILE LOOP over KV tiles (kN0=64 keys each):
//        GEMM0   S_acc = Q . K^T          (op_gemm: reads K from LDS, Q from reg)
//        SOFTMAX mask -> row_max -> exp2 -> row_sum  (online; op_softmax)
//        V STAGE DRAM -> regs -> v_perm shuffle -> LDS   (op_lds)
//        ONLINE  rescale carried O_acc by exp2(scale*(old_max-new_max)) when the
//                running max grew this tile; correct the running sum likewise
//        GEMM1   O_acc += P . V           (op_gemm: reads V from LDS, P from reg)
//      All while prefetching the NEXT tile's K copy and the second V half so HBM
//      latency overlaps compute.
//   5. EPILOGUE— normalize O_acc by the final row sum, bf16-truncate, store O to
//                DRAM, optionally write LSE (op_epilog).
//
// ONLINE SOFTMAX (Milakov), carried across tiles in three scalars per row:
//     rmax — running max of scaled scores seen so far
//     rsum — running denominator (sum of exp2 probabilities)
//     o_acc_d0/d1 — running numerator (sum of P.V), in TransposedC layout
//   When a tile raises the max from rmax to m_new, every earlier contribution is
//   too large by exp2(scale*(rmax-m_new)); we rescale o_acc and rsum by that
//   factor BEFORE adding this tile, so the result equals a single global softmax.
//
// LDS DOUBLE/TRIPLE BUFFERING via LdsSeq[] — see the constant's comment below.
//
// sched_barrier() CALLS: the __builtin_amdgcn_sched_barrier(mask) calls scattered
//   through the loop mirror CK's barrier structure one-for-one. Their PURPOSE here
//   is codegen/parity: pinning the compiler's instruction scheduling to match CK's
//   so the generated ISA (and thus numerics/behavior) lines up — a CORRECTNESS /
//   parity goal, not a perf lever. VERIFIED: barrier-for-barrier matching by
//   itself moved performance ~0%. The mask argument restricts what the scheduler
//   may move across the barrier (0 = full barrier / no reordering across it;
//   0x1, 0x7, 0x7F = progressively allow only certain instruction classes, used
//   to fence MFMA vs VALU regions exactly as CK does). Do not read them as a
//   tuning knob.
//
// THREAD GEOMETRY (shared with the op_*.hpp files):
//   warp_id = threadIdx.x>>6 (0..3); lane_id = threadIdx.x&63 (0..63);
//   k_sub = lane_id>>5 (0/1, the 32-lane half); m_row = (lane_id&31)+32*warp_id
//   is this lane's query row within the M-tile (TransposedC: one M-row per lane).

// Build an untyped byte buffer SRD over a DRAM tensor base. num_records is the
// VALID byte extent of the region from `base`: the hardware bounds-check then
// returns 0 for any access at or beyond it. This is load-bearing for the partial
// tail tile (seqlen_k % kN0 != 0): the K/V tile loop walks a full kN0-wide tile
// even when the last tile has fewer real keys, so the padding rows (row >=
// seqlen_k) read PAST the tensor. Without a real bound those reads return whatever
// is in adjacent memory (e.g. a freed NaN block), and the masked-but-still-summed
// P(=0)*V term computes 0*NaN = NaN in GEMM1, poisoning O_acc. Bounding the SRD
// makes those OOB reads return 0 instead. 0x00027000 is the CDNA data-format word
// for a raw byte buffer used by the raw_buffer_load builtins.
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

// LDS buffer rotation for the four staging slots used within one tile iteration.
// The kernel runs a 3-buffer rotating LDS scheme (op_lds.hpp: buf_idx in {0,1,2}).
// LdsSeq encodes which physical buffer each logical slot of a tile maps to:
//   LdsSeq[0] = K sub-tile 0 (consumed by GEMM0 sub-tile 0, and where the NEXT
//               tile's prefetched K lands)
//   LdsSeq[1] = K sub-tile 1 (consumed by GEMM0 sub-tile 1)
//   LdsSeq[2] = V half 0     (staged for GEMM1 sub-tile 0)
//   LdsSeq[3] = V half 1     (staged for GEMM1 sub-tile 1)
// The values {1,2,1,0} keep the K tile being read by GEMM0 in a different physical
// buffer from the V tile being written for GEMM1, so producer and consumer never
// alias the same buffer within an iteration (the reuse of buffer 1 for both K
// halves is safe because GEMM0 finishes sub-tile 0 before sub-tile 1 is needed).
constexpr int LdsSeq[4] = {1, 2, 1, 0};

// One block's full FMHA forward pass over its M-tile.
//   HasMask  : compile-time. false = boundary mask only; true = causal+boundary.
//   IsVarlen : compile-time. false = dense batch tensors; true = group/varlen.
//   IsSplit  : compile-time. false (DEFAULT) = ordinary full forward pass: every
//              split-specific branch below is `if constexpr`-discarded. true =
//              split-K: walk only this split's disjoint KV sub-range and write a
//              normalized fp32 partial (O_g, LSE_g) to the split-major scratch via
//              epilog_store_split (see op_epilog.hpp).
//   params   : tensor pointers, strides, scale, optional LSE/seqstart arrays.
//   lds      : this block's __shared__ scratch (kLdsBytes; the 3 rotating buffers).
//   batch_idx/head_idx/m_tile_idx : the tile coordinates (from blockIdx; the
//                                   causal M-tile reversal already applied in
//                                   the entry .cu files for the masked entries).
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

    // GQA/MQA: several Q heads can share one K/V head. Map this Q head to its KV
    // head (nhead_ratio==1 for full MHA).
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

    // Buffer SRDs the raw_buffer_load builtins read through (O's SRD is built
    // separately inside the epilogue). Each SRD is bounded to the valid byte extent
    // of this (b,h) tensor region so the partial-tail-tile padding rows (row >=
    // seqlen_q / seqlen_k) read 0 rather than out-of-bounds garbage. Q rows are
    // already guarded (OOB rows load zeros), but bounding it too costs nothing and
    // keeps the three paths uniform. Extent = #rows * row_stride(elements) * 2 bytes.
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
        // Causal: skip every KV tile that lies entirely PAST this M-tile's
        // diagonal (those keys are all masked, so they'd add nothing). Derivation:
        //   last_q_row  = highest query row this M-tile owns (clamped to seqlen_q)
        //   raw_end     = last column that row may attend = last_q_row+mask_shift+1
        //   seqlen_k_end= raw_end rounded UP to a whole kN0 tile (so the diagonal
        //                 tile itself is still processed; softmax_mask handles the
        //                 partial masking within it), clamped to seqlen_k.
        // Combined with the heavy-first M-tile reversal in the entry .cu files, this is what
        // makes causal cost ~linear in m_tile.
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
    // Discarded entirely when IsSplit=false. For a split, partition the FULL tile count
    // computed above into G contiguous chunks and keep only THIS split's chunk:
    //   T (tiles per split) = ceil(num_total_loop_full / num_splits)
    //   this split owns tiles [split_idx*T, min((split_idx+1)*T, full))
    // Translating tiles -> keys (×kN0) and adding to the existing seqlen_k_start
    // narrows WITHIN the already-causal-clamped [seqlen_k_start, seqlen_k_end)
    // range, so masked-future tiles a causal M-tile already excluded stay excluded
    // (A4 causal-correctness). The kv_offset / kv_v_byte / kv_k_byte induction
    // vars below initialize from seqlen_k_start, so narrowing it here makes them
    // pick up the split's start key automatically (no separate fix-up needed).
    // An empty split (start tile >= full) leaves num_total_loop <= 0, so the
    // degenerate sentinel path below fires (and, for IsSplit, writes the fp32
    // -inf/0 sentinel plane via epilog_store_split — see that path).
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
    // Resolve, for THIS (split_idx, b, h), the base of the Sq×64 fp32 partial-O
    // plane and the Sq fp32 LSE plane in the split-major scratch buffer:
    //   scratch_o_base  = scratch_o  + (((split_idx*B + b)*Hq + h)*Sq)*64
    //   scratch_lse_base= scratch_lse + (((split_idx*B + b)*Hq + h)*Sq)
    // epilog_store_split then just adds the in-plane row/col (abs_m_row*64 + col /
    // abs_m_row). Hq == params.nhead_q. B is not a kernarg field: the split grid's
    // z-axis is batch*num_splits, so B = gridDim.z / num_splits (documented in
    // FmhaFwdSplitParams). The whole block is if-constexpr-discarded when
    // IsSplit=false.
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

    // Degenerate tile (e.g. a causal M-tile whose every key is masked, or a varlen
    // tail): no KV work. Emit a zeroed O row with LSE=-inf and return. The LSE base
    // resolution mirrors the epilogue's (kept inline to avoid carrying it down).
    // For a split this is ALSO the empty-split path (narrowed range empty); it must
    // write the fp32 -inf/0 sentinel plane via epilog_store_split (A4: a mask1
    // split entirely in the masked-future region still owns its scratch plane), NOT
    // the bf16 epilog_store. The else-branch is the EXISTING code, unchanged.
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

    // Load this lane's full kHeadDim(=64) Q slice as 4x b128 (4 dwords = 8 bf16
    // each). Per the TransposedC mapping, this lane owns headdim
    // hd = kstep*16 + k_sub*8 + (0..7) in q_regs[kstep]; slice_q() (op_gemm.hpp)
    // hands the right pair of these to each GEMM0 sub-tile. Out-of-range query rows
    // (the last M-tile's padding) load zeros so masked rows contribute nothing.
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

    // V byte-base induction variable: kv_offset pre-multiplied into the V row
    // stride (bytes). load_v_from_dram consumes this directly so the per-tile
    // address math is a constant add, not a multiply. kv_offset is wave-uniform
    // (from blockIdx) so this stays in an SGPR — no VGPR-budget cost.
    const int v_stride_bytes = params.stride_v * 2;
    int kv_v_byte = kv_offset * v_stride_bytes;

    // K byte-base induction variable: same transform as the V one above, for the
    // async DRAM->LDS K copies. Also wave-uniform -> SGPR.
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
            // Drain the V regs loaded above, shuffle+store V half 0 into LDS
            // (LdsSeq[2]) for GEMM1, then start loading V half 1 (rows +32).
            s_waitcnt_vmcnt_0();
            store_v_to_lds(v_k3_0, v_k3_1, lds, LdsSeq[2]);
            v2i v1_k3_0, v1_k3_1;
            load_v_from_dram(v1_k3_0, v1_k3_1, srd_v, params.stride_v, kv_v_byte + 32 * v_stride_bytes);
            // v1 load left in flight: its only consumer is store_v_to_lds at the
            // end of GEMM1 (already guarded by s_waitcnt_vmcnt_0 there). Draining
            // here would expose the V-load HBM latency instead of overlapping it
            // with the exp2 / row_sum / rescale / GEMM1 compute that follows.

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

            // P fp32->bf16 truncation is done inline by gemm1_subtile's
            // v_perm_b32 (selector 0x07060302 extracts the high 16 bits of
            // each fp32). A separate &=0xFFFF0000 pass would be redundant.

            // ---- GEMM1 sub-tile 0: O_acc += P_n0 . V_half0 ----
            // block_sync_lds() makes V half 0 (just stored) visible to all waves.
            // P is packed to bf16 inline inside gemm1_subtile. After the MFMA,
            // shuffle+store V half 1 into LDS (LdsSeq[3]) for sub-tile 1.
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
        // Resolve the LSE output base for this (batch/varlen, head). For varlen the
        // LSE tensor is packed like Q (nhead_stride derived from Q's element strides
        // + the sequence offset); for dense it is [batch][head][seqlen_q]. nullptr
        // if the caller did not request LSE.
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
