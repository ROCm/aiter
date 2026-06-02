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
// ---------------------------------------------------------------------------
template <typename output_t, int Mtp>
__global__ __launch_bounds__(v0::kNumThreads, 2)
void pa_fp8_main_kernel_v2(
    const __hip_fp8_e4m3_fnuz* __restrict__ q,
    const __hip_fp8_e4m3_fnuz* __restrict__ k_cache,
    const __hip_fp8_e4m3_fnuz* __restrict__ v_cache,
    const float                              softmax_scale,
    const float* __restrict__                q_scale_ptr,
    const float* __restrict__                k_scale_ptr,
    const float* __restrict__                v_scale_ptr,
    const float* __restrict__                p_scale_ptr,
    const float* __restrict__                p_scale_inv_ptr,
    const bool                               has_p_scale,
    const int* __restrict__                  block_tables,
    const int* __restrict__                  context_lens,
    const int                                max_num_blocks_per_seq,
    const int                                q_stride,
    const int                                kv_block_stride,
    const int                                kv_head_stride,
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

    const int kbi_start    = fp_idx * num_kblocks_per_fat_part;
    const int kbi_stop_raw = kbi_start + num_kblocks_per_fat_part;
    const int kbi_stop     = (kbi_stop_raw < total_num_kblocks)
                                 ? kbi_stop_raw : total_num_kblocks;

    if (kbi_start >= total_num_kblocks) return;

    const int wg_start_head_idx    = kv_head_idx * kGqaRatio;
    const int wg_start_kv_head_idx = kv_head_idx;
    const int num_context_blocks   = PAGQA_DIVUP(context_len, kBlockSize);
    const int last_ctx_block       = num_context_blocks - 1;
    const int* block_table_seq     = block_tables + seq_idx * max_num_blocks_per_seq;

    __shared__ _T8x8 shared_logits[kNWarps * kTLoop * kSlotsPerWarpT];
    __shared__ float shared_qk[kNWarps * 16 * 2];

    const __amdgpu_buffer_rsrc_t k_rsrc =
        pa_make_buffer_rsrc(k_cache + wg_start_kv_head_idx * kv_head_stride);
    const __amdgpu_buffer_rsrc_t v_rsrc =
        pa_make_buffer_rsrc(v_cache + wg_start_kv_head_idx * kv_head_stride);

    const int q_token_for_lane = (kMtp == 1) ? 0 : (lane16id >> 3);
    const int head_for_lane    = lane16id & (kGqaRatio - 1);
    const int q_head_idx       = wg_start_head_idx + head_for_lane;
    const int64_t q_scale_idx =
          (static_cast<int64_t>(seq_idx) * kMtp + q_token_for_lane)
        * static_cast<int64_t>(total_num_heads) + q_head_idx;
    const float qk_base_log2 = softmax_scale * q_scale_ptr[q_scale_idx] * kLog2E;
    const float v_scale_perhead = v_scale_ptr[kv_head_idx];
    const float p_scale_perhead = has_p_scale ? p_scale_ptr[q_head_idx] : 1.f;
    const float p_scale_inv_perhead =
        has_p_scale ? p_scale_inv_ptr[q_head_idx] : 1.f;

    // Wide Q load: 16 fp8 / lane per qkhe step, covering head_dim
    // (qkhe*64 + rowid*16) .. (+15) — one head_dim chunk worth.  Held as
    // a pair of int64 so the lo/hi halves can be fed to two consecutive
    // MFMA calls (mirrors gluon's `v[N+0:N+1]` + `v[N+2:N+3]` pattern).
    struct PaWide { int64_t lo; int64_t hi; };
    PaWide Qlocal[v2::kWideQkheLoop];
    {
        const int64_t query_row_off =
            (static_cast<int64_t>(seq_idx) * kMtp + q_token_for_lane) * q_stride
            + (wg_start_head_idx + head_for_lane) * kHeadSize;
        const __hip_fp8_e4m3_fnuz* q_row = q + query_row_off;
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

    float   m_running = -FLT_MAX;
    float   l_running = 0.f;
    floatx4 o_running[kVheLoop];
    #pragma unroll
    for (int vhe = 0; vhe < kVheLoop; vhe++)
        o_running[vhe] = floatx4{0.f, 0.f, 0.f, 0.f};

    for (int kbi = kbi_start; kbi < kbi_stop; kbi++)
    {
        const int partition_start_token_idx = kbi * kTParSize;
        const int partition_block_start     = kbi * (kTParSize / kBlockSize);

        if (kbi != kbi_start) __syncthreads();

        floatx4 d_out[kTLoop];
        // V_wide[v_group][vhe]: per-(warp's-tile-group, head_dim_chunk)
        // dwordx4 V slice.  Per lane the slice covers 16 slots (one full
        // block) × 1 head_dim of the block at
        //   partition_block_start + v_group * 4 + rowid.
        // Lo half (slots 0..7) → PV MFMA #1, hi half (slots 8..15) → MFMA #2.
        pa_u32x4 V_wide[kNWarps][kVheLoop];
        {
            int kphysical_block_number[kTLoop];
            int kphysical_block_offset[kTLoop];
            #pragma unroll
            for (int t = 0; t < kTLoop; t++)
            {
                const int klocal_token_idx  = kTokensPerWarp * warpid + t * 16 + lane16id;
                const int kglobal_token_idx = partition_start_token_idx + klocal_token_idx;
                const int kblock_idx        = warpid * kTLoop + t;
                const int bt_g_idx          = partition_block_start + kblock_idx;
                const int bt_g_idx_safe     = (bt_g_idx < num_context_blocks)
                                                  ? bt_g_idx : last_ctx_block;
                kphysical_block_number[t]   = block_table_seq[bt_g_idx_safe];
                kphysical_block_offset[t]   = kglobal_token_idx % kBlockSize;
            }

            PaWide Klocal[kTLoop][v2::kWideQkheLoop];

            // Wide V phys-block table: rowid now selects which of the 4
            // blocks within a v_group (= warp's tile of 4 blocks).  Each
            // PV MFMA pair covers slots 0..7 (lo) + slots 8..15 (hi) of
            // those 4 blocks → 64 tokens per (v_group, vhe).
            unsigned int v_phys_block_wide[kNWarps];
            #pragma unroll
            for (int v_group = 0; v_group < kNWarps; v_group++)
            {
                const int v_bt_g_idx = partition_block_start
                                       + v_group * kRowsPerWarp + rowid;
                const int v_bt_g_idx_safe =
                    (v_bt_g_idx < num_context_blocks) ? v_bt_g_idx : last_ctx_block;
                v_phys_block_wide[v_group] =
                    (unsigned int)block_table_seq[v_bt_g_idx_safe];
            }

            // Wide K base offset: lane (rowid, lane16id) reads its chunk's
            // intra_chunk row of 16 fp8.  Stride for qkhe is
            // `kBytesPerWideQkhe` (= 1024 B = 4 head_dim chunks).
            const unsigned int k_chunk_row_off =
                (unsigned int)rowid * (unsigned int)v2::kBytesPerChunkAllSlot;

            auto load_k_tile = [&](int t) __attribute__((always_inline))
            {
                const unsigned int kblock_number =
                    (unsigned int)kphysical_block_number[t];
                const unsigned int k_base_voffset =
                    kblock_number * (unsigned int)kv_block_stride
                    + (unsigned int)kphysical_block_offset[t] * kElems16B_fp8
                    + k_chunk_row_off;
                #pragma unroll
                for (int qkhe = 0; qkhe < v2::kWideQkheLoop; qkhe++)
                {
                    const unsigned int voff =
                        k_base_voffset
                        + (unsigned int)qkhe * (unsigned int)v2::kBytesPerWideQkhe;
                    const pa_u32x4 v = pa_buffer_load_b128(k_rsrc, voff);
                    Klocal[t][qkhe].lo = pa_u32x4_low_long(v);
                    Klocal[t][qkhe].hi = pa_u32x4_high_long(v);
                }
            };

            constexpr unsigned int kVBytesPerVhe = (unsigned int)(kNWarps * 16 * kBlockSize);
            // Wide V load: 1 dwordx4 per (v_group, vhe), reading 16 slots
            // (= full block) × 1 head_dim per lane.  Same total bytes as
            // v1's 2 dwordx2 per (v, t_pv, vhe) but half the VMEM
            // instruction count.
            auto load_v_slice_wide = [&](int v_group) __attribute__((always_inline))
            {
                const unsigned int v_phys = v_phys_block_wide[v_group];
                const unsigned int v_base_voffset =
                    v_phys * (unsigned int)kv_block_stride
                    + (unsigned int)(warpid * 16 + lane16id) * kBlockSize;
                // Non-temporal (L2-bypass) V loads.  Each V byte is read
                // exactly once per workgroup (paged-attention has no
                // intra-WG V reuse), so caching V in L2 would only thrash
                // K cache lines.  Matches gluon's `global_load nt` policy.
                V_wide[v_group][0] = pa_buffer_load_b128_nt(v_rsrc, v_base_voffset);
                V_wide[v_group][1] = pa_buffer_load_b128_nt(
                    v_rsrc, v_base_voffset + kVBytesPerVhe);
            };

            load_k_tile(0);
            #pragma unroll
            for (int t = 0; t < kTLoop; t++)
            {
                if (t + 1 < kTLoop) load_k_tile(t + 1);
                // v_group index aligns with t: warp `myself` issues
                // load_v_slice_wide(0..3) interleaved with QK MFMA #0..3,
                // mirroring v1's load_v_slice(t) cadence.
                load_v_slice_wide(t);

                d_out[t] = floatx4{0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int qkhe = 0; qkhe < v2::kWideQkheLoop; qkhe++)
                {
                    // Lo half: head_dim subset {rowid*16 + qkhe*64 + 0..7}
                    d_out[t] = pa_mfma16x16x32_fp8_fp8(
                        Klocal[t][qkhe].lo, Qlocal[qkhe].lo, d_out[t]);
                    // Hi half: head_dim subset {rowid*16 + qkhe*64 + 8..15}
                    d_out[t] = pa_mfma16x16x32_fp8_fp8(
                        Klocal[t][qkhe].hi, Qlocal[qkhe].hi, d_out[t]);
                }
                pa_apply_qk_token_scales_for_block(
                    d_out[t], k_scale_ptr, kphysical_block_number[t], rowid * 4,
                    gridDim.z, kv_head_idx, kBlockSize, qk_base_log2);
            }
            // NOTE: NO `__builtin_amdgcn_sched_group_barrier` here.  v1 had
            // a `sched_group_barrier(MFMA, 4, 0)` which forced "4 MFMA in a
            // row, then VMEM" — this BURSTS the memory pipeline and creates
            // both consumer-side stalls (waiting for V) and issue-side
            // stalls (4 dwordx4 in a row).  Removing it lets the LLVM
            // AMDGPU scheduler freely interleave the 16 K+V dwordx4 loads
            // with the 16 QK MFMAs (1:1 ratio), which beats every
            // explicit IGLP pattern we tested (incl. gluon's 1:4 pattern).
            //
            // Measured on bs=64..256 ctx=128k:
            //   with    sched_group_barrier(MFMA, 4, 0) : v2 = 818 us (bs=64)
            //   without sched_group_barrier            : v2 = 770 us (bs=64)
            // — a 6% main-kernel win from a single deletion.
        }

        const int qkout_token_idx = partition_start_token_idx
                                    + kTokensPerWarp * warpid + rowid * 4;
        float qk_max  = -FLT_MAX;
        float exp_sum = 0.f;
        {
            const int valid_upper = context_len;
            const bool interior_partition =
                (partition_start_token_idx + kTParSize) <= valid_upper;

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

        #pragma unroll
        for (int t = 0; t < kTLoop; t++)
        {
            d_out[t] *= warp_scale * p_scale_perhead;
            const uint32_t pk = pa_pk_fp8x4(
                d_out[t][0], d_out[t][1], d_out[t][2], d_out[t][3]);
            const int idx = v0::shared_logits_index(warpid, t, lane16id, rowid);
            shared_logits[idx].i64 = static_cast<int64_t>(pk);
        }
        __syncthreads();

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
    const float post_scale = inv_l * v_scale_perhead * p_scale_inv_perhead;
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
