// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026 Page_Attetion_GQA_fp8 project
//
// FP8 paged-attention decode main kernel (v2) — full wide-load variant for bs >= 32.
//
// Both QK and PV use kWidth=16 (16 fp8 per lane, `buffer_load_dwordx4`)
// and each wide load feeds 2 sequential `v_mfma_f32_16x16x32_fp8_fp8`
// calls.  Halves both K and V VMEM instruction counts vs v1's
// `buffer_load_dwordx2` (kWidth=8) path while keeping MFMA instruction
// count unchanged — matching gluon's wide-tile load pattern.
//
// PMC sketch (bs=64, ctx=64k, nf=5):
//
//   |  load type   | v1   | v2 K-wide only | v2 full-wide |
//   |--------------|------|----------------|--------------|
//   | K dwordx2    |  32  |    0           |    0         |
//   | K dwordx4    |   0  |   16           |   16         |
//   | V dwordx2    |  32  |   32           |    0         |
//   | V dwordx4    |   0  |    0           |   16         |
//   | MFMA fp8     |  64  |   64           |   64         |
//
// QK chain lane mapping (unchanged from K-wide-only v2)
// -----------------------------------------------------
//   K cache memory layout: [num_blocks, kv_heads, hd//16, slots, 16] fp8.
//   Lane (rowid, lane16id) reads 16 contiguous fp8 = ONE full intra-chunk
//   row at chunk = (qkhe*4 + rowid), slot = lane16id.  For one MFMA call
//   hardware sees K[k=8*rowid..+7, n=lane16id] as src0.  Feeding the lo 8
//   fp8 to MFMA #1 and the hi 8 fp8 to MFMA #2 assigns the lane's 16 fp8
//   to head_dim subsets that, summed across rowid, cover head_dim 0..63
//   (qkhe=0) and 64..127 (qkhe=1).  Q is loaded with the SAME lane→
//   head_dim mapping, so the dot product is numerically identical to v1's.
//
// PV chain lane mapping (new)
// ---------------------------
//   V cache layout: [num_blocks, kv_heads, head_size, block_size] fp8.
//   Innermost stride = block_size = 16 (= slot axis); block_size = 16 fp8
//   per (block, head_dim) row → exactly one dwordx4.
//
//   Per outer kbi iter the partition has 16 KV blocks split as 4 groups
//   of 4 blocks (v_group=0..3 ≡ source warp index).  Lane (rowid, lane16id)
//   issues 2 dwordx4 V loads per v_group (one per `vhe` head_dim chunk),
//   each reading 16 slots × 1 head_dim = ONE full block × ONE head_dim of
//   the block at `pbs + v_group*4 + rowid`.  4 rowid lanes therefore read
//   4 DIFFERENT blocks within a v_group.
//
//   PV MFMA src0 hardware layout: lane (rowid, lane16id) provides
//   V[K=8*rowid..+7, N=lane16id].  Feeding the lo 8 fp8 (slots 0..7) to
//   MFMA #1 and the hi 8 fp8 (slots 8..15) to MFMA #2 means a single
//   MFMA call's K=0..31 covers `slots 0..7 of 4 different blocks` (not
//   32 contiguous tokens).  Numerically identical to v1's
//   "32 contiguous tokens" mapping because PV accumulates across all
//   16 MFMA calls covering the full 256-token partition × 128 head_dim.
//
//   P→LDS reading: QK^T LDS write index is `(warpid, t, lane16id, rowid)`
//   (UNCHANGED from v1 — saves a writeback rewrite).  Wide-PV's lane
//   (rowid, lane16id) reads P at `(v_group, rowid_self, lane16id,
//   qk_subrow=0..3)` instead — the `t` axis of LDS is naturally indexed
//   by the new PV lane's `rowid` because QK^T's block at index
//   (pbs + warpid*4 + t) is exactly the block PV wants when v_group=warpid
//   and rowid_self=t.  Pairs of qk_subrow rows are stitched into the lo/
//   hi int64 src1 operand per MFMA.
//
//   Net effect (PV alone):
//     - 2 dwordx4 V loads per v_group × 4 v_groups = 8 dwordx4 V loads
//       per warp per outer iter (was 16 dwordx2; VMEM count halved)
//     - 16 PV MFMA calls per warp per outer iter (same as v1)

#pragma once

#include "pa_fp8_common.h"
#include "pa_fp8_main_kernel.cuh"  // re-uses v0 namespace constants

namespace pa_fp8_gqa {

namespace v2 {

// QK chain constants for the wide-load (kWidth=16) variant.  These are
// LOCAL to the QK MFMA section — V/PV constants in v0:: are reused as-is.
//
// kWideQkheLoop * kK_PER_WIDE_QKHE = kHeadSize = 128.
constexpr int kFp8PerLaneWide      = 16;                            // 16 fp8 = 2 longs
constexpr int kK_PER_WIDE_QKHE     = v0::kRowsPerWarp * kFp8PerLaneWide;   // 4 rowid lanes * 16 = 64 head_dim per qkhe
constexpr int kWideQkheLoop        = v0::kHeadSize / kK_PER_WIDE_QKHE;     // 2
constexpr int kBytesPerChunkAllSlot = v0::kBlockSize * v0::kElems16B_fp8;  // 256 (= 16 slots * 16 fp8)
constexpr int kBytesPerWideQkhe    = v0::kRowsPerWarp * kBytesPerChunkAllSlot; // 1024 (= 4 chunks * 256)

} // namespace v2

// ---------------------------------------------------------------------------
// pa_fp8_main_kernel_v2 — bs >= 32 wide-load variant.
//
// QK uses kWidth=16 (16 fp8/lane) + buffer_load_dwordx4 + paired MFMA;
// everything else (V load, PV MFMA, LDS layout, softmax, output) is the
// same as v1.  See block-comment at top of this file for the lane mapping.
//
// Template parameters mirror v1: `QIn` selects fp8-direct vs bf16-with-in-
// kernel-quantisation Q loading, and `HasPScale` toggles the FlyDSL-style
// P fp8 quantisation rescale (p_scale before pack, p_scale_inv folded into
// the post-PV correction).
// ---------------------------------------------------------------------------
// `EnablePrefetch` toggles the FlyDSL-aligned cross-kbi K + K-scale
// pipeline.  Launcher picks it via `num_kblocks_per_fat_part >= 2`:
//   - true:  prologue prefetches K(kbi_start) parallel to Q load; each
//            iter consumes loop-carried K and issues next-kbi prefetch
//            during PV MFMA → +20-60% wins at long-ctx low-bs.
//   - false: legacy in-iter K-load path (load_k_tile interleaved with
//            QK MFMA t-loop) → avoids prologue + wasted-last-iter
//            prefetch overhead in 1-iter-per-CTA short-ctx cases.
template <typename output_t, int Mtp,
          typename QIn = __hip_fp8_e4m3_fnuz, bool HasPScale = false,
          bool EnablePrefetch = true>
__global__ __launch_bounds__(v0::kNumThreads, 2)
void pa_fp8_main_kernel_v2(
    const QIn* __restrict__                  q,
    const __hip_fp8_e4m3_fnuz* __restrict__ k_cache,
    const __hip_fp8_e4m3_fnuz* __restrict__ v_cache,
    const float                              softmax_scale,
    const float* __restrict__                q_scale_ptr,   // unused when QIn=bf16
    const float* __restrict__                k_scale_ptr,
    const float* __restrict__                v_scale_ptr,
    const float* __restrict__                p_scale_ptr,     // [total_num_heads]
    const float* __restrict__                p_scale_inv_ptr, // [total_num_heads]
    const float                              p_scale_value,   // scalar fallback when p_scale_ptr is null
    const float                              p_scale_inv_value,
    const int* __restrict__                  block_tables,
    const int* __restrict__                  context_lens,
    const int                                max_num_blocks_per_seq,
    const int                                q_stride,
    const int                                kv_block_stride,
    const int                                kv_head_stride,
    const int                                ks_block_stride, // k_scale per-block stride (fp32 elems) = k_scale.stride(0). Read from the tensor's REAL stride so strided/padded views (e.g. FlyDSL packed [nb,1,nkv,hd/4] embedded in the padded KV cache) are consumed zero-copy without a .contiguous() materialization.
    const int                                ks_head_stride,  // k_scale per-kv-head stride (fp32 elems) = k_scale.stride(dim-2): block_size for flat [nb,nkv,bs], head_dim/4 for packed [nb,1,nkv,hd/4]
    float* __restrict__                      exp_sums,
    float* __restrict__                      max_logits,
    output_t* __restrict__                   out,
    const int                                num_kblocks_per_fat_part)
{
    using namespace v0;
    constexpr int kMtp = Mtp;
    static_assert(Mtp == 1 || Mtp == 2, "v2 supports Mtp in {1, 2}");

    constexpr float kLog2E    = 1.4426950408889634f;
    constexpr float kInvLog2E = 0.6931471805599453f;

    const auto seq_idx     = blockIdx.x;
    const auto fp_idx      = blockIdx.y;
    const auto kv_head_idx = blockIdx.z;

    const int warpid   = threadIdx.x / WARP_SIZE;
    const int laneid   = threadIdx.x % WARP_SIZE;
    const int lane16id = laneid % 16;
    const int rowid    = laneid / 16;

    const int num_fat_partitions = gridDim.y;
    const int total_num_heads    = gridDim.z * kGqaRatio;
    const int context_len        = context_lens[seq_idx];
    const int total_num_kblocks  = PAGQA_DIVUP(context_len, kTParSize);

    // FlyDSL-aligned RUNTIME page size.  Distribute the ACTUAL context's
    // kblocks across the launched fat partitions (grid.y) at runtime,
    // instead of the capture-time `num_kblocks_per_fat_part` (sized off the
    // worst-case max_seq_len).  The static value left every partition
    // covering a fixed `cdiv(total_kblocks(max_seq_len), nf)` slice, which
    // at ctx << max_seq_len both (a) under-parallelizes the real work
    // (few partitions cover all of it serially) and (b) leaves the rest as
    // dead CTAs.  The runtime page size mirrors
    // `page_size_partitions = cdiv(num_total_partitions, max_context_partition_num)`
    // in FlyDSL's pa_decode_ps_kernel, so the real work always spreads over
    // `min(nf, total_num_kblocks)` partitions.  `num_kblocks_per_fat_part`
    // is now consumed ONLY by the launcher (prefetch-variant selection).
    const int kbpfp_rt     = PAGQA_DIVUP(total_num_kblocks, num_fat_partitions);
    const int kbi_start    = fp_idx * kbpfp_rt;
    const int kbi_stop_raw = kbi_start + kbpfp_rt;
    const int kbi_stop     = (kbi_stop_raw < total_num_kblocks)
                                 ? kbi_stop_raw : total_num_kblocks;

    if (kbi_start >= total_num_kblocks) {
        // Graph-capture safety: this partition has no work for the
        // current `context_lens[seq_idx]`.  Put exp_sums/max_logits
        // into "contributes-zero" state so the reduce kernel ignores
        // this slot:
        //   max_logits = -inf  →  rescale weight = exp(-inf - max) = 0
        //   exp_sums   = 0     →  prevents `0 * inf = NaN` from stale data
        //
        // We deliberately DO NOT clear tmp_out here (it would cost ~1us
        // per skipped CTA, which dominates at short ctx + large nf).
        // Instead the caller's `make_workspace` zero-initializes
        // tmp_out once; subsequent writes by valid partitions leave it
        // with finite real values.  Either way, the reduce's
        // `0 * tmp_out` term is `0 * finite = 0` — safe.  This depends
        // on the caller using `make_workspace` (or any zero-init alloc);
        // calling decode with a `torch.empty` tmp_out is UB.
        if (warpid == 0 && rowid == 0 && lane16id < kMtp * kGqaRatio) {
            const int q_token_for_lane_ee = (kMtp == 1) ? 0 : (lane16id >> 3);
            const int head_idx_ee = lane16id & (kGqaRatio - 1);
            const int total_num_heads_ee = gridDim.z * kGqaRatio;
            const int64_t maxp_ee = static_cast<int64_t>(num_fat_partitions);
            const int64_t offset_ee =
                  (static_cast<int64_t>(seq_idx) * kMtp + q_token_for_lane_ee)
                      * static_cast<int64_t>(total_num_heads_ee) * maxp_ee
                + (static_cast<int64_t>(kv_head_idx) * kGqaRatio + head_idx_ee) * maxp_ee
                + static_cast<int64_t>(fp_idx);
            max_logits[offset_ee] = -FLT_MAX;
            exp_sums[offset_ee]   = 0.f;
        }
        return;
    }

    const int wg_start_head_idx    = kv_head_idx * kGqaRatio;
    const int wg_start_kv_head_idx = kv_head_idx;
    const int num_context_blocks   = PAGQA_DIVUP(context_len, kBlockSize);
    const int last_ctx_block       = num_context_blocks - 1;
    const int* block_table_seq     = block_tables + seq_idx * max_num_blocks_per_seq;

    __shared__ _T8x8 shared_logits[kNWarps * kTLoop * kSlotsPerWarpT];
    __shared__ float shared_qk[kNWarps * 16 * 2];
    // Dedicated bf16-quant staging — see v1 for full layout commentary.
    __shared__ int64_t q_stage_lds[16 * 16];
    __shared__ float   q_scale_lds[16];
    // K-scale LDS staging (FlyDSL-aligned).  Each warp stages
    //   ks_lds[warpid][t][slot]  ← k_scale[kphys[warpid][t]][kv_head][slot]
    // Per kbi the warp does 1 buffer_load_dword (each lane loads 1 fp32,
    // covering 4 kphys × 16 slots = 64 fp32 → 1 VMEM op vs the previous
    // 4 dwordx4 loads inside the apply step).  Lane (rowid, lane16id)
    // loads kphys[rowid]'s slot lane16id; the apply step then does a
    // ds_read_b128 (4 contiguous fp32) per t.  Layout is warp-local so
    // no cross-warp sync needed; the compiler-emitted `s_waitcnt lgkmcnt`
    // handles the intra-wave RAW hazard between write and read.
    __shared__ float ks_lds[kNWarps * kTLoop * kBlockSize];

    // ── Per-kbi phys-block-table LDS staging (FlyDSL-aligned) ───────────
    // Profiling (rocprof, bs=64 ctx=16384 mtp=2) showed our K/V buffer_loads
    // serialise behind the per-lane `block_table_seq[...]` global_loads that
    // feed their addresses: same MFMA + L2 traffic as FlyDSL, yet 2.46x the
    // SQ_WAIT_ANY and 5.67x the VMEM-active cycles.  We break that dependency
    // chain by cooperatively staging this kbi's `kBlocksPerKbi` block-table
    // entries into LDS once (16 threads, 1 coalesced load), so both the K
    // prefetch and the inline V load read phys blocks from LDS (ds_read,
    // ~20cy) instead of each issuing dependent narrow global_loads.
    // Double-buffered on `kbi & 1` so iter `kbi`'s V can read BT(kbi) while
    // the same iter's K-prefetch reads the just-staged BT(kbi+1); the write
    // is published by the EXISTING post-prob-pack barrier (no new sync).
    constexpr int kBlocksPerKbi = kTParSize / kBlockSize;
    __shared__ int bt_lds[2][kBlocksPerKbi];

    const int64_t kv_head_base_offset =
        static_cast<int64_t>(wg_start_kv_head_idx) * kv_head_stride;

    const int q_token_for_lane = (kMtp == 1) ? 0 : (lane16id >> 3);
    const int head_for_lane    = lane16id & (kGqaRatio - 1);
    const int q_head_idx       = wg_start_head_idx + head_for_lane;

    float qk_base_log2;
    if constexpr (std::is_same<QIn, __hip_fp8_e4m3_fnuz>::value)
    {
        const int64_t q_scale_idx =
              (static_cast<int64_t>(seq_idx) * kMtp + q_token_for_lane)
            * static_cast<int64_t>(total_num_heads) + q_head_idx;
        qk_base_log2 = softmax_scale * q_scale_ptr[q_scale_idx] * kLog2E;
    }
    else
    {
        qk_base_log2 = 0.f;  // filled in after Q load
    }
    const float v_scale_perhead = v_scale_ptr[kv_head_idx];

    float p_scale_lane     = 1.f;
    float p_scale_inv_lane = 1.f;
    if constexpr (HasPScale)
    {
        if (p_scale_ptr != nullptr)
        {
            p_scale_lane     = p_scale_ptr    [q_head_idx];
            p_scale_inv_lane = p_scale_inv_ptr[q_head_idx];
        }
        else
        {
            p_scale_lane     = p_scale_value;
            p_scale_inv_lane = p_scale_inv_value;
        }
    }

    // Wide Q load: 16 fp8 / lane per qkhe step, covering head_dim
    // (qkhe*64 + rowid*16) .. (+15) — one head_dim chunk worth.  Held as
    // a pair of int64 so the lo/hi halves can be fed to two consecutive
    // MFMA calls (mirrors gluon's `v[N+0:N+1]` + `v[N+2:N+3]` pattern).
    //
    // For QIn=bf16 we still produce the same `(lo, hi)` int64 packed-fp8
    // pair after per-lane bf16→fp32→fp8 conversion + cross-rowid max-abs
    // reduce, so the downstream wide-load MFMA chain is byte-identical.
    struct PaWide { int64_t lo; int64_t hi; };
    PaWide Qlocal[v2::kWideQkheLoop];

    // ── Loop-carried K-data + K-scale (FlyDSL-aligned cross-kbi prefetch) ──
    //
    // When `EnablePrefetch=true`: `Klocal` / `my_ks_carried` are refreshed
    // at the END of each iter (after QK MFMA consumes them, before PV
    // MFMA starts) with the NEXT kbi's K + K-scale, so the buffer_load
    // latency overlaps with PV MFMA + the accumulator update of the
    // current iter — mirrors FlyDSL's `k_flat` / `k_scale_next` loop-
    // carry pattern in `pa_decode_ps_kernel`
    // (FlyDSL/kernels/pa_decode_fp8.py lines 2497-2595).
    //
    // When `EnablePrefetch=false`: K is loaded inside each iter via the
    // legacy `load_k_tile` lambda (interleaved with QK MFMA t-loop) and
    // these VGPRs are unused.  Compiler DCE removes the allocation.
    //
    // VGPR cost (EnablePrefetch=true): +17 VGPR for K-data + K-scale.
    // Stays at occupancy 4 waves/SIMD; baseline (no prefetch) is at 3.
    PaWide Klocal_carried[kTLoop][v2::kWideQkheLoop];
    float  my_ks_carried;

    const unsigned int k_chunk_row_off =
        (unsigned int)rowid * (unsigned int)v2::kBytesPerChunkAllSlot;

    // Cooperative single-shot load of kbi_in's block table into LDS buffer
    // `kbi_in & 1`.  Caller MUST issue a __syncthreads() before any thread
    // reads bt_lds[kbi_in & 1] (the cross-kbi-prefetch path reuses the
    // existing post-prob-pack barrier for this; the prologue / no-prefetch
    // path adds its own).
    auto stage_bt_to_lds = [&](int kbi_in) __attribute__((always_inline))
    {
        if (warpid == 0 && rowid == 0)
        {
            const int g_idx      = kbi_in * kBlocksPerKbi + lane16id;
            const int g_idx_safe = (g_idx < num_context_blocks)
                                       ? g_idx : last_ctx_block;
            bt_lds[kbi_in & 1][lane16id] = block_table_seq[g_idx_safe];
        }
    };

    auto stage_kbi_prefetch = [&](int kbi_in) __attribute__((always_inline))
    {
        // Issues K-data + K-scale buffer_loads for `kbi_in`, writing into
        // `Klocal_carried` and `my_ks_carried` (the loop-carried VGPRs).
        // The block_table index is clamped to `last_ctx_block` to keep
        // OOB-speculative loads (final iter's prefetch) safe.
        const int partition_start_token_idx_x = kbi_in * kTParSize;

        int kphys_local[kTLoop];
        int kphys_off_local[kTLoop];
        #pragma unroll
        for (int t = 0; t < kTLoop; t++)
        {
            const int klocal_token_idx  = kTokensPerWarp * warpid + t * 16 + lane16id;
            const int kglobal_token_idx = partition_start_token_idx_x + klocal_token_idx;
            // Phys block from the LDS-staged block table (stage_bt_to_lds),
            // not a per-lane global_load — breaks the block_table_seq[...] ->
            // K-address dependency chain that serialised the K buffer_loads.
            kphys_local[t]              = bt_lds[kbi_in & 1][warpid * kTLoop + t];
            kphys_off_local[t]          = kglobal_token_idx % kBlockSize;
        }

        // Per-thread K-scale fetch (mirrors FlyDSL's `_load_my_k_scale_from_vgpr`):
        // each thread gets its `kphys[rowid]` directly from VGPR (avoids
        // the LDS round-trip needed by the K-data path's wider broadcast).
        int my_kphys;
        if (rowid == 0)      my_kphys = kphys_local[0];
        else if (rowid == 1) my_kphys = kphys_local[1];
        else if (rowid == 2) my_kphys = kphys_local[2];
        else                 my_kphys = kphys_local[3];
        // K-scale element offset using the tensor's REAL strides (fp32 elems):
        //   ks_off = kphys*ks_block_stride + kv_head*ks_head_stride + slot
        // This reads whatever layout k_scale actually has, so a strided/padded
        // view is consumed zero-copy (no host .contiguous() needed):
        //   flat   [nb, nkv, bs]            → stride(0)=nkv*bs, stride(-2)=bs
        //   packed [nb, 1, nkv, head_dim/4] → stride(0)=padded-block stride,
        //                                     stride(-2)=head_dim/4
        // The legacy formula (kphys*gridDim.z + kv_head)*ks_head_stride hard-wired
        // the block stride to gridDim.z*ks_head_stride, which only held when the
        // scale buffer was densely packed; it silently mis-addressed the padded
        // FlyDSL view on the block axis.  For the packed layout scale_rows==1
        // (block_size*4 <= head_dim at v2's fixed bs=16/hd=128) so slot stays in
        // row 0 and the padding tail is never read.
        const int64_t ks_off =
              static_cast<int64_t>(my_kphys) * ks_block_stride
            + static_cast<int64_t>(kv_head_idx) * ks_head_stride
            + lane16id;
        my_ks_carried = k_scale_ptr[ks_off];

        // Issue all K-data buffer_loads for this kbi (8 dwordx4 / lane).
        #pragma unroll
        for (int t = 0; t < kTLoop; t++)
        {
            const int64_t kblock_number = static_cast<int64_t>(kphys_local[t]);
            const __amdgpu_buffer_rsrc_t k_block_rsrc =
                pa_make_buffer_rsrc(
                    k_cache + kv_head_base_offset
                    + kblock_number * static_cast<int64_t>(kv_block_stride));
            const unsigned int k_base_voffset =
                (unsigned int)kphys_off_local[t] * kElems16B_fp8
                + k_chunk_row_off;
            #pragma unroll
            for (int qkhe = 0; qkhe < v2::kWideQkheLoop; qkhe++)
            {
                const unsigned int voff =
                    k_base_voffset
                    + (unsigned int)qkhe * (unsigned int)v2::kBytesPerWideQkhe;
                const pa_u32x4 v = pa_buffer_load_b128(k_block_rsrc, voff);
                Klocal_carried[t][qkhe].lo = pa_u32x4_low_long(v);
                Klocal_carried[t][qkhe].hi = pa_u32x4_high_long(v);
            }
        }
    };

    if constexpr (EnablePrefetch) {
        // PROLOGUE: stage iter 0's block table to LDS, then issue
        // K(kbi_start) + K-scale(kbi_start) buffer_loads BEFORE the Q load.
        // The one-shot barrier here is amortised over the whole CTA; the K
        // HBM latency (~400 cy) then overlaps with the Q load + Q LDS staging
        // + Q register prep that follows (~500-800 cy on the bf16-Q path),
        // so iter 0's QK MFMA finds K already in VGPR with no extra wait.
        stage_bt_to_lds(kbi_start);
        __syncthreads();
        stage_kbi_prefetch(kbi_start);
    }

    {
        const int64_t query_row_off =
            (static_cast<int64_t>(seq_idx) * kMtp + q_token_for_lane) * q_stride
            + (wg_start_head_idx + head_for_lane) * kHeadSize;
        const QIn* q_row = q + query_row_off;
        if constexpr (std::is_same<QIn, __hip_fp8_e4m3_fnuz>::value)
        {
            #pragma unroll
            for (int qkhe = 0; qkhe < v2::kWideQkheLoop; qkhe++)
            {
                const int hd_off = qkhe * v2::kK_PER_WIDE_QKHE
                                 + rowid * v2::kFp8PerLaneWide;
                const int64_t* p =
                    reinterpret_cast<const int64_t*>(q_row + hd_off);
                Qlocal[qkhe].lo = p[0];
                Qlocal[qkhe].hi = p[1];
            }
        }
        else
        {
            // bf16 Q in-kernel quant — same FlyDSL-aligned LDS staging as v1
            // (dedicated q_stage_lds buffer).  Producer writes 8 fp8 per
            // (qhead, segment), consumer reads back pairs of i64 cells for
            // v2's wide-load (16 fp8 / lane / qkhe) layout.
            const int q_quant_qhead_raw = warpid * kRowsPerWarp + rowid;
            const int q_quant_qhead = (kMtp == 1)
                                         ? (q_quant_qhead_raw & (kGqaRatio - 1))
                                         : q_quant_qhead_raw;
            const int q_quant_qtoken =
                (kMtp == 1) ? 0 : (q_quant_qhead_raw >> 3);
            const int q_quant_qhead_in_token =
                (kMtp == 1) ? q_quant_qhead : (q_quant_qhead & (kGqaRatio - 1));

            const int64_t q_quant_row_off =
                  (static_cast<int64_t>(seq_idx) * kMtp + q_quant_qtoken) * q_stride
                + (wg_start_head_idx + q_quant_qhead_in_token) * kHeadSize;
            const __hip_bfloat16* q_q_row =
                reinterpret_cast<const __hip_bfloat16*>(q + q_quant_row_off);
            const _B16x8 q_bf =
                *reinterpret_cast<const _B16x8*>(q_q_row + lane16id * 8);

            const floatx4 qf_lo = pa_to_floatx4<__hip_bfloat16>(q_bf.xy[0]);
            const floatx4 qf_hi = pa_to_floatx4<__hip_bfloat16>(q_bf.xy[1]);
            float lm = fmaxf(
                fmaxf(fmaxf(fabsf(qf_lo[0]), fabsf(qf_lo[1])),
                      fmaxf(fabsf(qf_lo[2]), fabsf(qf_lo[3]))),
                fmaxf(fmaxf(fabsf(qf_hi[0]), fabsf(qf_hi[1])),
                      fmaxf(fabsf(qf_hi[2]), fabsf(qf_hi[3]))));
            lm = fmaxf(lm, pa_shfl_xor_within_32<8>(lm));
            lm = fmaxf(lm, pa_shfl_xor_within_32<4>(lm));
            lm = fmaxf(lm, pa_shfl_xor_within_32<2>(lm));
            lm = fmaxf(lm, pa_shfl_xor_within_32<1>(lm));

            const float q_scale_lane =
                (lm > 0.f) ? (lm * (1.f / PA_FP8_MAX)) : 1.f;
            const float inv_q = __builtin_amdgcn_rcpf(q_scale_lane);

            const uint32_t pk_lo = pa_pk_fp8x4(
                qf_lo[0] * inv_q, qf_lo[1] * inv_q,
                qf_lo[2] * inv_q, qf_lo[3] * inv_q);
            const uint32_t pk_hi = pa_pk_fp8x4(
                qf_hi[0] * inv_q, qf_hi[1] * inv_q,
                qf_hi[2] * inv_q, qf_hi[3] * inv_q);

            q_stage_lds[q_quant_qhead * 16 + lane16id] =
                  static_cast<int64_t>(pk_lo)
                | (static_cast<int64_t>(pk_hi) << 32);
            if (lane16id == 0)
                q_scale_lds[q_quant_qhead] = q_scale_lane;

            __syncthreads();

            // Consumer (v2 wide-load): lane (R, L) needs 16 fp8 per qkhe
            //   = (qhead=L, hd[qkhe*64 + R*16 .. qkhe*64 + R*16 + 15])
            // split into two consecutive i64 cells:
            //   lo  = q_stage_lds[L][qkhe*8 + R*2 + 0]   (hd_seg = qkhe*8 + R*2)
            //   hi  = q_stage_lds[L][qkhe*8 + R*2 + 1]   (hd_seg = qkhe*8 + R*2 + 1)
            const int consumer_qhead = (kMtp == 1)
                                         ? (lane16id & (kGqaRatio - 1))
                                         : lane16id;
            qk_base_log2 = softmax_scale * q_scale_lds[consumer_qhead] * kLog2E;

            #pragma unroll
            for (int qkhe = 0; qkhe < v2::kWideQkheLoop; qkhe++)
            {
                const int seg_base = qkhe * 8 + rowid * 2;
                Qlocal[qkhe].lo = q_stage_lds[consumer_qhead * 16 + seg_base + 0];
                Qlocal[qkhe].hi = q_stage_lds[consumer_qhead * 16 + seg_base + 1];
            }
        }
    }

    float   m_running = -FLT_MAX;
    float   l_running = 0.f;
    floatx4 o_running[kVheLoop];
    #pragma unroll
    for (int vhe = 0; vhe < kVheLoop; vhe++)
        o_running[vhe] = floatx4{0.f, 0.f, 0.f, 0.f};

    // EnablePrefetch=true: K + K-scale for iter 0 were prefetched BEFORE
    // the Q load above so the HBM latency overlaps with Q load.  See
    // `stage_kbi_prefetch(kbi_start)` near `PaWide Qlocal`.
    // EnablePrefetch=false: K + K-scale for iter 0 are loaded inline at
    // iter start below (no prologue).
    constexpr unsigned int kVBytesPerVhe = (unsigned int)(kNWarps * 16 * kBlockSize);

    for (int kbi = kbi_start; kbi < kbi_stop; kbi++)
    {
        const int partition_start_token_idx = kbi * kTParSize;

        if (kbi != kbi_start) __syncthreads();

        // Baseline path: K + K-scale are NOT prefetched cross-kbi.  Issue
        // the buffer_loads here at iter start (compiler inserts implicit
        // waitcnt before QK MFMA reads `Klocal_carried`).  This is the
        // 1-iter-per-CTA short-ctx friendly path: no prologue overhead
        // (Q-load is the long pole anyway) and no wasted final-iter
        // prefetch.  At ≥2 iters per CTA, the cross-kbi prefetch path
        // wins (selected by `EnablePrefetch=true` in launcher).
        if constexpr (!EnablePrefetch) {
            stage_bt_to_lds(kbi);
            __syncthreads();
            stage_kbi_prefetch(kbi);
        }

        floatx4 d_out[kTLoop];
        // V_wide[v_group][vhe]: per-(warp's-tile-group, head_dim_chunk)
        // dwordx4 V slice.  Per lane the slice covers 16 slots (one full
        // block) × 1 head_dim of the block at
        //   partition_block_start + v_group * 4 + rowid.
        // Lo half (slots 0..7) → PV MFMA #1, hi half (slots 8..15) → MFMA #2.
        pa_u32x4 V_wide[kNWarps][kVheLoop];
        {
            // ── K-scale LDS staging (from loop-carried VGPR) ────────────
            // Source of `my_ks_carried`:
            //   EnablePrefetch=true  → loaded by previous iter's
            //     `stage_kbi_prefetch` (or prologue for iter 0), HBM
            //     latency already paid for.
            //   EnablePrefetch=false → loaded inline at iter start (just
            //     above), compiler inserts implicit waitcnt here.
            ks_lds[warpid * (kTLoop * kBlockSize) + rowid * kBlockSize + lane16id] =
                my_ks_carried;

            // Wide V phys-block table: rowid now selects which of the 4
            // blocks within a v_group (= warp's tile of 4 blocks).  Each
            // PV MFMA pair covers slots 0..7 (lo) + slots 8..15 (hi) of
            // those 4 blocks → 64 tokens per (v_group, vhe).
            unsigned int v_phys_block_wide[kNWarps];
            #pragma unroll
            for (int v_group = 0; v_group < kNWarps; v_group++)
            {
                // Phys block from the LDS-staged block table (stage_bt_to_lds
                // for THIS kbi, published by the previous iter's post-prob-pack
                // barrier / the prologue barrier) — removes the per-lane
                // block_table_seq[...] global_load on the V-address path.
                v_phys_block_wide[v_group] =
                    (unsigned int)bt_lds[kbi & 1][v_group * kRowsPerWarp + rowid];
            }

            // Wide V load: 1 dwordx4 per (v_group, vhe), reading 16 slots
            // (= full block) × 1 head_dim per lane.  Issued interleaved
            // with QK MFMA so V's load latency hides behind QK compute.
            auto load_v_slice_wide = [&](int v_group) __attribute__((always_inline))
            {
                const unsigned int v_phys = v_phys_block_wide[v_group];
                const __amdgpu_buffer_rsrc_t v_block_rsrc =
                    pa_make_buffer_rsrc(
                        v_cache + kv_head_base_offset
                        + static_cast<int64_t>(v_phys)
                              * static_cast<int64_t>(kv_block_stride));
                const unsigned int v_base_voffset =
                    (unsigned int)(warpid * 16 + lane16id) * kBlockSize;
                // Non-temporal (L2-bypass) V loads.  Each V byte is read
                // exactly once per workgroup (paged-attention has no
                // intra-WG V reuse), so caching V in L2 would only thrash
                // K cache lines.  Matches gluon's `global_load nt` policy.
                V_wide[v_group][0] = pa_buffer_load_b128_nt(
                    v_block_rsrc, v_base_voffset);
                V_wide[v_group][1] = pa_buffer_load_b128_nt(
                    v_block_rsrc, v_base_voffset + kVBytesPerVhe);
            };

            #pragma unroll
            for (int t = 0; t < kTLoop; t++)
            {
                // v_group index aligns with t: warp `myself` issues
                // load_v_slice_wide(0..3) interleaved with QK MFMA #0..3.
                load_v_slice_wide(t);

                d_out[t] = floatx4{0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int qkhe = 0; qkhe < v2::kWideQkheLoop; qkhe++)
                {
                    // Lo half: head_dim subset {rowid*16 + qkhe*64 + 0..7}
                    d_out[t] = pa_mfma16x16x32_fp8_fp8(
                        Klocal_carried[t][qkhe].lo, Qlocal[qkhe].lo, d_out[t]);
                    // Hi half: head_dim subset {rowid*16 + qkhe*64 + 8..15}
                    d_out[t] = pa_mfma16x16x32_fp8_fp8(
                        Klocal_carried[t][qkhe].hi, Qlocal[qkhe].hi, d_out[t]);
                }
                // K-scale apply via LDS-staged values: lane (rowid, lane16id)
                // reads 4 contiguous fp32 from ks_lds[warpid][t][rowid*4 .. +3]
                // (one ds_read_b128 per t).  The HBM K-scale load latency
                // was hidden by prev iter's PV MFMA (cross-kbi prefetch).
                const float* ks_row =
                    &ks_lds[warpid * (kTLoop * kBlockSize) + t * kBlockSize + rowid * 4];
                const float4 ks4 = *reinterpret_cast<const float4*>(ks_row);
                d_out[t][0] *= qk_base_log2 * ks4.x;
                d_out[t][1] *= qk_base_log2 * ks4.y;
                d_out[t][2] *= qk_base_log2 * ks4.z;
                d_out[t][3] *= qk_base_log2 * ks4.w;
            }
        }

        const int qkout_token_idx = partition_start_token_idx
                                    + kTokensPerWarp * warpid + rowid * 4;
        float qk_max  = -FLT_MAX;
        float exp_sum = 0.f;
        {
            // MTP causal mask: query token `q_token_for_lane` (0..kMtp-1) is the
            // (kMtp-1-q_token_for_lane)-th token BEFORE the sequence end, so it
            // must NOT attend to the speculative tokens that follow it within
            // the same mtp group.  Its causal KV length is therefore
            //   context_len - (kMtp - 1 - q_token_for_lane).
            // (kMtp==1 collapses to context_len.)  Without this every mtp token
            // shared `context_len`, letting token0 attend to token1's KV slot —
            // a future-token leak that corrupts decode output under MTP.
            const int valid_upper =
                context_len - (kMtp - 1 - q_token_for_lane);
            // `interior_partition` gates whether the (per-lane) mask loop runs
            // at all.  Different lanes now have different `valid_upper`, so the
            // gate must use the SMALLEST causal length in the group
            // (= context_len - (kMtp - 1), token0's) — otherwise a lane that
            // still needs masking could be skipped.
            const int valid_upper_min = context_len - (kMtp - 1);
            const bool interior_partition =
                (partition_start_token_idx + kTParSize) <= valid_upper_min;

            if (!interior_partition)
            {
                #pragma unroll
                for (int t = 0; t < kTLoop; t++)
                {
                    const int local_token_idx = qkout_token_idx + t * 16;
                    #pragma unroll
                    for (int i = 0; i < 4; i++)
                    {
                        if ((local_token_idx + i) >= valid_upper)
                            d_out[t][i] = -FLT_MAX;
                    }
                }
            }

            #pragma unroll
            for (int t = 0; t < kTLoop; t++)
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    qk_max = fmaxf(qk_max, d_out[t][i]);
            qk_max = fmaxf(qk_max, pa_shfl_xor_32(qk_max));
            qk_max = fmaxf(qk_max, pa_shfl_xor_within_32<16>(qk_max));

            const floatx4 nqk_max{-qk_max, -qk_max, -qk_max, -qk_max};
            floatx4 exp_sum_v4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int t = 0; t < kTLoop; t++)
            {
                const floatx4 diff = d_out[t] + nqk_max;
                floatx4 v;
                v[0] = __builtin_amdgcn_exp2f(diff[0]);
                v[1] = __builtin_amdgcn_exp2f(diff[1]);
                v[2] = __builtin_amdgcn_exp2f(diff[2]);
                v[3] = __builtin_amdgcn_exp2f(diff[3]);
                d_out[t]    = v;
                exp_sum_v4 += v;
            }
            exp_sum = exp_sum_v4[0] + exp_sum_v4[1] + exp_sum_v4[2] + exp_sum_v4[3];
            exp_sum = exp_sum + pa_shfl_xor_32(exp_sum);
            exp_sum = exp_sum + pa_shfl_xor_within_32<16>(exp_sum);
        }

        if (laneid < 16)
        {
            const int slot = lane16id * (kNWarps * 2) + warpid * 2;
            shared_qk[slot + 0] = qk_max;
            shared_qk[slot + 1] = exp_sum;
        }
        __syncthreads();

        float partition_qk_max  = -FLT_MAX;
        float partition_exp_sum = 0.f;
        float warp_scale;
        {
            float warp_qk_max[kNWarps];
            float warp_exp_sum[kNWarps];
            const int base = lane16id * (kNWarps * 2);
            #pragma unroll
            for (int w = 0; w < kNWarps; w++)
            {
                warp_qk_max[w]   = shared_qk[base + w * 2 + 0];
                warp_exp_sum[w]  = shared_qk[base + w * 2 + 1];
                partition_qk_max = fmaxf(partition_qk_max, warp_qk_max[w]);
            }
            float warp_qk_max_exp[kNWarps];
            #pragma unroll
            for (int w = 0; w < kNWarps; w++)
            {
                warp_qk_max_exp[w] = __builtin_amdgcn_exp2f(
                    warp_qk_max[w] - partition_qk_max);
                partition_exp_sum += warp_exp_sum[w] * warp_qk_max_exp[w];
            }
            warp_scale = warp_qk_max_exp[warpid];
        }

        // FlyDSL-style P-scale: pre-multiply attention weights by
        // `p_scale_lane` so the cvt_pk_fp8_f32 input is centred near 1.0
        // (caller-chosen), preserving fp8 dynamic range.  Compensated by
        // `p_scale_inv_lane` in `post_scale` below.
        const float p_pack_scale = HasPScale ? (warp_scale * p_scale_lane)
                                             :  warp_scale;
        #pragma unroll
        for (int t = 0; t < kTLoop; t++)
        {
            d_out[t] *= p_pack_scale;
            const uint32_t pk = pa_pk_fp8x4(
                d_out[t][0], d_out[t][1], d_out[t][2], d_out[t][3]);
            const int idx = v0::shared_logits_index(warpid, t, lane16id, rowid);
            shared_logits[idx].i64 = static_cast<int64_t>(pk);
        }
        // Stage NEXT kbi's block table to LDS NOW, so the existing
        // post-prob-pack barrier below publishes it for both this iter's
        // K-prefetch (which reads BT(kbi+1)) and next iter's V load — i.e.
        // no new barrier is introduced.  Clamp mirrors the K-prefetch's
        // `kbi_safe` so the final iter re-stages the current kbi harmlessly.
        if constexpr (EnablePrefetch) {
            const int kbi_bt_next = kbi + 1;
            const int kbi_bt_safe = (kbi_bt_next < kbi_stop)
                                        ? kbi_bt_next : (kbi_stop - 1);
            stage_bt_to_lds(kbi_bt_safe);
        }
        __syncthreads();

        // ── Cross-kbi K + K-scale prefetch (FlyDSL-aligned) ─────────────
        // Issue next iter's K-data and K-scale buffer_loads here, BEFORE
        // PV MFMA, so the ~400-cy HBM latency is hidden behind PV MFMA +
        // accumulator update (~512+ cycles of compute on the
        // critical path).  By the time next iter's QK MFMA needs them,
        // the loads have completed.
        //
        // Clamp `kbi+1` to `kbi_stop-1` so the FINAL iter's prefetch
        // re-loads the current iter's K (a harmless no-op-after-quiesce
        // since QK MFMA consumed those values already and the data is
        // discarded post-loop).  Using a select instead of a branch keeps
        // the compiler from materializing both Klocal_carried & a "next"
        // buffer at the iter boundary — VGPR stays at 126 / occupancy 4
        // vs 150 / occupancy 3 we'd get with `if (kbi+1<kbi_stop) prefetch`.
        // Mirrors FlyDSL's `arith.select` pattern in pa_decode_ps_kernel.
        //
        // We overwrite `Klocal_carried` / `my_ks_carried` in place —
        // current iter's QK MFMA already consumed them and PV MFMA reads
        // `V_wide` and `P_lo/P_hi_per_g` (unrelated VGPRs), so K's old
        // values are dead.
        if constexpr (EnablePrefetch) {
            const int kbi_next = kbi + 1;
            const int kbi_safe = (kbi_next < kbi_stop)
                                     ? kbi_next : (kbi_stop - 1);
            stage_kbi_prefetch(kbi_safe);
        }

        floatx4 pv_acc[kVheLoop];
        #pragma unroll
        for (int vhe = 0; vhe < kVheLoop; vhe++)
            pv_acc[vhe] = floatx4{0.f, 0.f, 0.f, 0.f};

        // Wide PV: 4 MFMA per v_group (2 vhe × {lo, hi}), 16 total.
        //
        // P→LDS write index is `(warpid, t, lane16id, rowid)` (unchanged
        // from v1).  Since QK^T computed block (pbs + warpid*4 + t) at
        // LDS slot (warpid, t, ...) and wide-V reads block
        // (pbs + v_group*4 + rowid) for lane (rowid, *), the LDS read
        // index trivially becomes (v_group, rowid, lane16id, qk_subrow).
        // No LDS layout change needed.
        //
        // Per v_group, lane (rowid, lane16id) needs:
        //   P_lo (for MFMA #1, slots 0..7 of block (v_group, rowid))
        //     = LDS[v_group, rowid, lane16id, 0] (slots 0..3)
        //     ++ LDS[v_group, rowid, lane16id, 1] (slots 4..7)
        //   P_hi (for MFMA #2, slots 8..15)
        //     = LDS[v_group, rowid, lane16id, 2] (slots 8..11)
        //     ++ LDS[v_group, rowid, lane16id, 3] (slots 12..15)
        int64_t P_lo_per_g[kNWarps];
        int64_t P_hi_per_g[kNWarps];
        #pragma unroll
        for (int v_group = 0; v_group < kNWarps; v_group++)
        {
            _T8x8 P_lo_pack, P_hi_pack;
            P_lo_pack.b8x4[0] = static_cast<uint32_t>(
                shared_logits[v0::shared_logits_index(v_group, rowid, lane16id, 0)].i64);
            P_lo_pack.b8x4[1] = static_cast<uint32_t>(
                shared_logits[v0::shared_logits_index(v_group, rowid, lane16id, 1)].i64);
            P_hi_pack.b8x4[0] = static_cast<uint32_t>(
                shared_logits[v0::shared_logits_index(v_group, rowid, lane16id, 2)].i64);
            P_hi_pack.b8x4[1] = static_cast<uint32_t>(
                shared_logits[v0::shared_logits_index(v_group, rowid, lane16id, 3)].i64);
            P_lo_per_g[v_group] = P_lo_pack.i64;
            P_hi_per_g[v_group] = P_hi_pack.i64;
        }

        #pragma unroll
        for (int v_group = 0; v_group < kNWarps; v_group++)
        {
            const int64_t P_lo = P_lo_per_g[v_group];
            const int64_t P_hi = P_hi_per_g[v_group];
            #pragma unroll
            for (int vhe = 0; vhe < kVheLoop; vhe++)
            {
                const pa_u32x4 V_chunk = V_wide[v_group][vhe];
                const int64_t V_lo = pa_u32x4_low_long(V_chunk);
                const int64_t V_hi = pa_u32x4_high_long(V_chunk);
                // MFMA #1: slots 0..7 of block (v_group, rowid)
                pv_acc[vhe] = pa_mfma16x16x32_fp8_fp8(V_lo, P_lo, pv_acc[vhe]);
                // MFMA #2: slots 8..15 of same block
                pv_acc[vhe] = pa_mfma16x16x32_fp8_fp8(V_hi, P_hi, pv_acc[vhe]);
            }
        }

        {
            const float m_new = fmaxf(m_running, partition_qk_max);
            const float alpha = (m_running > -FLT_MAX)
                ? __builtin_amdgcn_exp2f(m_running - m_new) : 0.f;
            const float beta  = __builtin_amdgcn_exp2f(partition_qk_max - m_new);
            l_running = alpha * l_running + beta * partition_exp_sum;
            #pragma unroll
            for (int vhe = 0; vhe < kVheLoop; vhe++)
                o_running[vhe] = alpha * o_running[vhe] + beta * pv_acc[vhe];
            m_running = m_new;
        }
    }

    const float inv_l = __fdividef(1.f, l_running + 1e-6f);
    // Fold p_scale_inv into post_scale when HasPScale (compensates the
    // P-pack pre-scale).  Net o_running math is identical to !HasPScale.
    const float post_scale = HasPScale
        ? (inv_l * v_scale_perhead * p_scale_inv_lane)
        : (inv_l * v_scale_perhead);
    #pragma unroll
    for (int vhe = 0; vhe < kVheLoop; vhe++)
        o_running[vhe] *= post_scale;

    if (warpid == 0 && rowid == 0 && lane16id < kMtp * kGqaRatio)
    {
        const int head_idx = lane16id & (kGqaRatio - 1);
        const int64_t query_start_off = static_cast<int64_t>(seq_idx) * kMtp;
        const int64_t maxp = static_cast<int64_t>(num_fat_partitions);
        const int64_t offset =
              static_cast<int64_t>(query_start_off + q_token_for_lane)
                  * static_cast<int64_t>(total_num_heads) * maxp
            + (static_cast<int64_t>(wg_start_head_idx) + head_idx) * maxp
            + static_cast<int64_t>(fp_idx);
        max_logits[offset] = m_running * kInvLog2E;
        exp_sums[offset]   = l_running;
    }

    _B16x4 outelems[kVheLoop];
    #pragma unroll
    for (int vhe = 0; vhe < kVheLoop; vhe++)
        outelems[vhe] = pa_from_floatx4<output_t>(o_running[vhe]);

    __syncthreads();
    #pragma unroll
    for (int vhe = 0; vhe < kVheLoop; vhe++)
    {
        const int idx = v0::shared_logits_index(warpid, vhe, lane16id, rowid);
        _T8x8 cell;
        cell.b16x4 = outelems[vhe];
        shared_logits[idx] = cell;
    }
    __syncthreads();

    if (warpid == 0)
    {
        const int64_t query_start_off = static_cast<int64_t>(seq_idx) * kMtp;
        constexpr int kGqa4_  = (kGqaRatio + 3) / 4;
        constexpr int kRowsHere = kMtp * kGqa4_;

        const int head_elem_idx = lane16id * 8;
        if (head_elem_idx < kHeadSize)
        {
            const int64_t hsz_maxp_mult =
                static_cast<int64_t>(kHeadSize)
              * static_cast<int64_t>(num_fat_partitions);
            #pragma unroll
            for (int local_row_idx = 0; local_row_idx < kRowsHere; local_row_idx++)
            {
                const int q_tok_w  = (kMtp == 1) ? 0 : (local_row_idx >> 1);
                const int head_quad = local_row_idx & 1;
                const int local_head_idx_in_quad = rowid;
                const int packed_lane = q_tok_w * 8 + head_quad * 4 + local_head_idx_in_quad;
                _B16x8 vout;
                const int offset1 = (head_elem_idx / 16) % 4;
                const int offset2 = head_elem_idx / 16 / kNWarps;
                const int offset3 = (head_elem_idx / 4) % 4;
                #pragma unroll
                for (int i = 0; i < 2; i++)
                {
                    const int idx =
                        v0::shared_logits_index(offset1, offset2, packed_lane, offset3 + i);
                    vout.xy[i] = shared_logits[idx].b16x4;
                }
                const int head_idx = head_quad * 4 + local_head_idx_in_quad;
                if (head_idx < kGqaRatio)
                {
                    const int64_t out_head_idx =
                        static_cast<int64_t>(wg_start_head_idx + head_idx);
                    output_t* out_ptr = out
                        + (query_start_off + q_tok_w) * total_num_heads * hsz_maxp_mult
                        + out_head_idx * hsz_maxp_mult
                        + fp_idx * kHeadSize
                        + head_elem_idx;
                    *reinterpret_cast<_B16x8*>(out_ptr) = vout;
                }
            }
        }
    }
}

} // namespace pa_fp8_gqa
