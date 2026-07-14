// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026 Page_Attetion_GQA_fp8 project
//
// FP8 paged-attention decode — SINGLE-PASS multi-(mtp)-group kernel.
//
// Ported from Page_Attetion_GQA_fp8_mtp_geak/csrc/pa_fp8_main_kernel_v2_mg.cuh.
// Handles query_length (== tx-aiter `mtp`) in {3,4} (fp8 query) by loading K/V
// (and K-scale) ONCE per 256-token partition and looping `NumGroups` mtp-groups
// internally (each group == the v2 kMtp=2 / 16-pair compute with its own Q,
// online-softmax accumulators and output rows).  K/V HBM traffic is therefore
// independent of ql, vs the ceil(ql/2)-pass v2 fallback.
//
// tx-aiter delta vs geak source:
//   * K-scale addressing uses the SAME two-stride scheme as pa_fp8_main_kernel_v2
//     (k_scale[kphys*ks_block_stride + kv_head*ks_head_stride + slot]) so it
//     consumes the exact same k_scale tensor tx-aiter feeds pa_fp8_decode_v2
//     (dense flat OR FlyDSL packed/padded fp32 view), zero-copy.
//   * Per-token MTP causal mask (context_len-(ql-1-gtoken(g))), matching
//     tx-aiter's context_len-includes-speculative-tokens convention.
//
// Scope / simplifications vs v2 (kept deliberately narrow + low-risk):
//   * fp8 query only (bf16-Q in-kernel quant stays on the v2 multi-pass path).
//   * no cross-kbi K prefetch beyond the one carried buffer.
//   * block_size 16/64 via the SAME unified addressing as v2.

#pragma once

#include "pa_fp8_common.h"
#include "pa_fp8_main_kernel.cuh"       // v0 constants
#include "pa_fp8_main_kernel_v2.cuh"    // v2:: wide-load constants

namespace pa_fp8_gqa {

// Each group handles kMtp=2 query tokens × kGqaRatio heads = 16 pairs.
//
// Static mode (DynSched=false, default):
//   grid = (num_seqs, nf, num_kv_heads), task_map_* / per_seq_nf ignored.
template <typename output_t, int NumGroups, typename QIn = __hip_fp8_e4m3_fnuz,
          bool HasPScale = false, int BlockSz = 16, bool DynSched = false>
__global__ __launch_bounds__(v0::kNumThreads, 2)
void pa_fp8_main_kernel_v2_mg(
    const QIn* __restrict__                  q,
    const __hip_fp8_e4m3_fnuz* __restrict__ k_cache,
    const __hip_fp8_e4m3_fnuz* __restrict__ v_cache,
    const float                              softmax_scale,
    const float* __restrict__                q_scale_ptr,
    const float* __restrict__                k_scale_ptr,
    const float* __restrict__                v_scale_ptr,
    const float* __restrict__                p_scale_ptr,
    const float* __restrict__                p_scale_inv_ptr,
    const int* __restrict__                  block_tables,
    const int* __restrict__                  context_lens,
    const int                                max_num_blocks_per_seq,
    const int                                q_stride,
    const int                                kv_block_stride,
    const int                                kv_head_stride,
    const int                                ks_block_stride,
    const int                                ks_head_stride,
    float* __restrict__                      exp_sums,
    float* __restrict__                      max_logits,
    output_t* __restrict__                   out,
    const int                                query_length,
    const int                                qtoken_base,
    // Dynamic scheduling parameters (only used when DynSched=true):
    const int* __restrict__                  task_map_seq = nullptr,
    const int* __restrict__                  task_map_fp  = nullptr,
    const int* __restrict__                  per_seq_nf   = nullptr,
    const int                                max_num_partitions = 0,
    const int* __restrict__                  total_tasks_ptr = nullptr)
{
    using namespace v0;
    constexpr int kMtp = 2;                 // tokens per group
    constexpr int kBlockSize = BlockSz;
    static_assert(BlockSz == 16 || BlockSz == 64, "block_size 16 or 64");
    constexpr float kLog2E    = 1.4426950408889634f;
    constexpr float kInvLog2E = 0.6931471805599453f;

    constexpr unsigned int kBytesPerChunkAllSlot_l =
        (unsigned int)(kBlockSize * kElems16B_fp8);
    constexpr unsigned int kBytesPerWideQkhe_l =
        (unsigned int)(kRowsPerWarp * kBlockSize * kElems16B_fp8);
    constexpr unsigned int kVBytesPerVhe =
        (unsigned int)(kNWarps * 16 * kBlockSize);

    // --- Index resolution: static vs dynamic scheduling ---
    int seq_idx, fp_idx, num_fat_partitions, ws_maxp;
    if constexpr (DynSched) {
        if (total_tasks_ptr != nullptr && static_cast<int>(blockIdx.x) >= *total_tasks_ptr)
            return;
        seq_idx             = task_map_seq[blockIdx.x];
        fp_idx              = task_map_fp[blockIdx.x];
        num_fat_partitions  = per_seq_nf[seq_idx];
        ws_maxp             = max_num_partitions;
    } else {
        seq_idx             = static_cast<int>(blockIdx.x);
        fp_idx              = static_cast<int>(blockIdx.y);
        num_fat_partitions  = static_cast<int>(gridDim.y);
        ws_maxp             = static_cast<int>(gridDim.y);
    }
    const auto kv_head_idx = blockIdx.z;

    const int warpid   = threadIdx.x / WARP_SIZE;
    const int laneid   = threadIdx.x % WARP_SIZE;
    const int lane16id = laneid % 16;
    const int rowid    = laneid / 16;

    const int total_num_heads    = gridDim.z * kGqaRatio;
    const int context_len        = context_lens[seq_idx];
    const int total_num_kblocks  = PAGQA_DIVUP(context_len, kTParSize);

    const int kbpfp_rt     = PAGQA_DIVUP(total_num_kblocks, num_fat_partitions);
    const int kbi_start    = fp_idx * kbpfp_rt;
    const int kbi_stop_raw = kbi_start + kbpfp_rt;
    const int kbi_stop     = (kbi_stop_raw < total_num_kblocks)
                                 ? kbi_stop_raw : total_num_kblocks;

    const int wg_start_head_idx    = kv_head_idx * kGqaRatio;
    const int wg_start_kv_head_idx = kv_head_idx;
    const int num_context_blocks   = PAGQA_DIVUP(context_len, kBlockSize);
    const int last_ctx_block       = num_context_blocks - 1;
    const int* block_table_seq     = block_tables + seq_idx * max_num_blocks_per_seq;

    const int q_token_for_lane = lane16id >> 3;          // 0 or 1 within group
    const int head_for_lane    = lane16id & (kGqaRatio - 1);
    const int q_head_idx       = wg_start_head_idx + head_for_lane;

    // seq row of (group g, local token q_token_for_lane) in the [seqs*ql,...]
    // tensors:  seq*ql + qtoken_base + g*kMtp + q_token_for_lane.
    const int64_t seq_qrow_base = static_cast<int64_t>(seq_idx) * query_length
                                + qtoken_base;

    // global token index (within sequence's ql tokens) for group g.
    auto gtoken = [&](int g) { return g * kMtp + q_token_for_lane; };
    auto valid_tok = [&](int g) { return gtoken(g) < query_length; };

    // ── Empty-partition early exit (graph-capture safety) ───────────────
    if (kbi_start >= total_num_kblocks) {
        if (warpid == 0 && rowid == 0 && lane16id < kMtp * kGqaRatio) {
            const int head_idx_ee = lane16id & (kGqaRatio - 1);
            const int64_t maxp = static_cast<int64_t>(ws_maxp);
            #pragma unroll
            for (int g = 0; g < NumGroups; g++) {
                if (!valid_tok(g)) continue;
                const int64_t off =
                      (seq_qrow_base + gtoken(g)) * static_cast<int64_t>(total_num_heads) * maxp
                    + (static_cast<int64_t>(wg_start_head_idx) + head_idx_ee) * maxp
                    + static_cast<int64_t>(fp_idx);
                max_logits[off] = -FLT_MAX;
                exp_sums[off]   = 0.f;
            }
        }
        return;
    }

    __shared__ _T8x8 shared_logits[NumGroups][kNWarps * kTLoop * kSlotsPerWarpT];
    constexpr int kSharedQkStride = kNWarps * 2 + 1;  // 9 (bank-conflict pad)
    __shared__ float shared_qk[NumGroups][16 * kSharedQkStride];
    __shared__ float ks_lds[kNWarps * kTokensPerWarp];
    // bf16-Q in-kernel quant staging (only used when QIn=bf16; DCE'd for fp8).
    // Mirrors pa_fp8_main_kernel_v2's q_stage_lds/q_scale_lds, re-used per group.
    __shared__ int64_t q_stage_lds[16 * 16];
    __shared__ float   q_scale_lds[16];
    constexpr int kBlocksPerKbi = kTParSize / kBlockSize;
    __shared__ int  bt_lds[2][kBlocksPerKbi];   // double-buffered for prefetch

    const __amdgpu_buffer_rsrc_t k_rsrc =
        pa_make_buffer_rsrc(k_cache + wg_start_kv_head_idx * kv_head_stride);
    const __amdgpu_buffer_rsrc_t v_rsrc =
        pa_make_buffer_rsrc(v_cache + wg_start_kv_head_idx * kv_head_stride);

    const float v_scale_perhead = v_scale_ptr[kv_head_idx];
    float p_scale_lane = 1.f, p_scale_inv_lane = 1.f;
    if constexpr (HasPScale) {
        p_scale_lane     = p_scale_ptr[q_head_idx];
        p_scale_inv_lane = p_scale_inv_ptr[q_head_idx];
    }

    struct PaWide { int64_t lo; int64_t hi; };

    const unsigned int k_chunk_row_off = (unsigned int)rowid * kBytesPerChunkAllSlot_l;

    auto stage_bt = [&](int kbi_in) __attribute__((always_inline)) {
        if (warpid == 0 && rowid == 0 && lane16id < kBlocksPerKbi) {
            const int g_idx  = kbi_in * kBlocksPerKbi + lane16id;
            const int g_safe = (g_idx < num_context_blocks) ? g_idx : last_ctx_block;
            bt_lds[kbi_in & 1][lane16id] = block_table_seq[g_safe];
        }
    };

    PaWide Klocal_carried[kTLoop][v2::kWideQkheLoop];
    float  my_ks_carried;
    int          blk_pf[kTLoop];
    unsigned int k_lane_off[kTLoop];
    #pragma unroll
    for (int t = 0; t < kTLoop; t++) {
        const int klocal = kTokensPerWarp * warpid + t * 16 + lane16id;
        blk_pf[t]        = (warpid * kTokensPerWarp + t * 16) / kBlockSize;
        k_lane_off[t]    = (unsigned int)(klocal % kBlockSize) * kElems16B_fp8
                         + k_chunk_row_off;
    }
    const int ks_tok     = warpid * kTokensPerWarp + rowid * 16 + lane16id;
    const int ks_blk_idx = ks_tok / kBlockSize;   // bt_lds index (kbi-invariant)
    const int ks_slot    = ks_tok % kBlockSize;
    auto prefetch_k = [&](int kbi_in) __attribute__((always_inline)) {
        {
            const int ks_blk = bt_lds[kbi_in & 1][ks_blk_idx];
            // Two-stride K-scale addressing (identical to pa_fp8_main_kernel_v2):
            //   k_scale[kphys*ks_block_stride + kv_head*ks_head_stride + slot]
            // consumes dense flat [nb,nkv,bs] AND FlyDSL packed/padded fp32
            // views zero-copy (strides read from the tensor by the launcher).
            my_ks_carried = k_scale_ptr[
                static_cast<int64_t>(ks_blk) * ks_block_stride
                + static_cast<int64_t>(kv_head_idx) * ks_head_stride
                + ks_slot];
        }
        #pragma unroll
        for (int t = 0; t < kTLoop; t++) {
            const unsigned int kbn   = (unsigned int)bt_lds[kbi_in & 1][blk_pf[t]];
            const unsigned int kbase = kbn * (unsigned int)kv_block_stride + k_lane_off[t];
            #pragma unroll
            for (int qkhe = 0; qkhe < v2::kWideQkheLoop; qkhe++) {
                const pa_u32x4 v = pa_buffer_load_b128(
                    k_rsrc, kbase + (unsigned int)qkhe * kBytesPerWideQkhe_l);
                Klocal_carried[t][qkhe].lo = pa_u32x4_low_long(v);
                Klocal_carried[t][qkhe].hi = pa_u32x4_high_long(v);
            }
        }
    };

    // Prologue: stage + prefetch iter 0's K BEFORE Q load.
    stage_bt(kbi_start);
    __syncthreads();
    prefetch_k(kbi_start);

    // ── Per-group Q + qk base scale ─────────────────────────────────────
    PaWide Qlocal[NumGroups][v2::kWideQkheLoop];
    float  qk_base_log2[NumGroups];
    #pragma unroll
    for (int g = 0; g < NumGroups; g++) {
        const int gt        = gtoken(g);
        const int gt_safe   = valid_tok(g) ? gt : (query_length - 1);
        if constexpr (std::is_same<QIn, __hip_fp8_e4m3_fnuz>::value) {
            // fp8 Q: external per-(token,head) q_scale, direct wide fp8 load.
            const int64_t q_row_off =
                (seq_qrow_base + gt_safe) * q_stride
                + (wg_start_head_idx + head_for_lane) * kHeadSize;
            const __hip_fp8_e4m3_fnuz* q_row = q + q_row_off;
            const int64_t q_scale_idx =
                (seq_qrow_base + gt_safe) * static_cast<int64_t>(total_num_heads) + q_head_idx;
            qk_base_log2[g] = softmax_scale * q_scale_ptr[q_scale_idx] * kLog2E;
            #pragma unroll
            for (int qkhe = 0; qkhe < v2::kWideQkheLoop; qkhe++) {
                const int hd_off = qkhe * v2::kK_PER_WIDE_QKHE + rowid * v2::kFp8PerLaneWide;
                const int64_t* p = reinterpret_cast<const int64_t*>(q_row + hd_off);
                Qlocal[g][qkhe].lo = p[0];
                Qlocal[g][qkhe].hi = p[1];
            }
        } else {
            // bf16 Q: in-kernel max|Q| quant to fp8 (FlyDSL-aligned), same LDS
            // staging as pa_fp8_main_kernel_v2, done per group (q_scale ignored).
            if (g > 0) __syncthreads();  // protect q_stage_lds reuse across groups
            const int q_quant_qhead_raw = warpid * kRowsPerWarp + rowid;  // 0..15
            const int qq_token = q_quant_qhead_raw >> 3;                  // 0/1 in group
            const int qq_head  = q_quant_qhead_raw & (kGqaRatio - 1);
            const int pgt      = g * kMtp + qq_token;
            const int pgt_safe = (pgt < query_length) ? pgt : (query_length - 1);
            const int64_t q_quant_row_off =
                  (seq_qrow_base + pgt_safe) * q_stride
                + (wg_start_head_idx + qq_head) * kHeadSize;
            const __hip_bfloat16* q_q_row =
                reinterpret_cast<const __hip_bfloat16*>(q + q_quant_row_off);
            const _B16x8 q_bf = *reinterpret_cast<const _B16x8*>(q_q_row + lane16id * 8);
            const floatx4 qf_lo = pa_to_floatx4<__hip_bfloat16>(q_bf.xy[0]);
            const floatx4 qf_hi = pa_to_floatx4<__hip_bfloat16>(q_bf.xy[1]);
            float lm = fmaxf(
                fmaxf(fmaxf(fabsf(qf_lo[0]), fabsf(qf_lo[1])), fmaxf(fabsf(qf_lo[2]), fabsf(qf_lo[3]))),
                fmaxf(fmaxf(fabsf(qf_hi[0]), fabsf(qf_hi[1])), fmaxf(fabsf(qf_hi[2]), fabsf(qf_hi[3]))));
            lm = fmaxf(lm, pa_shfl_xor_within_32<8>(lm));
            lm = fmaxf(lm, pa_shfl_xor_within_32<4>(lm));
            lm = fmaxf(lm, pa_shfl_xor_within_32<2>(lm));
            lm = fmaxf(lm, pa_shfl_xor_within_32<1>(lm));
            const float q_scale_lane = (lm > 0.f) ? (lm * (1.f / PA_FP8_MAX)) : 1.f;
            const float inv_q = __builtin_amdgcn_rcpf(q_scale_lane);
            const uint32_t pk_lo = pa_pk_fp8x4(qf_lo[0]*inv_q, qf_lo[1]*inv_q, qf_lo[2]*inv_q, qf_lo[3]*inv_q);
            const uint32_t pk_hi = pa_pk_fp8x4(qf_hi[0]*inv_q, qf_hi[1]*inv_q, qf_hi[2]*inv_q, qf_hi[3]*inv_q);
            q_stage_lds[q_quant_qhead_raw * 16 + lane16id] =
                  static_cast<int64_t>(pk_lo) | (static_cast<int64_t>(pk_hi) << 32);
            if (lane16id == 0) q_scale_lds[q_quant_qhead_raw] = q_scale_lane;
            __syncthreads();
            // consumer_qhead == q_token_for_lane*8 + head_for_lane == lane16id.
            const int consumer_qhead = lane16id;
            qk_base_log2[g] = softmax_scale * q_scale_lds[consumer_qhead] * kLog2E;
            #pragma unroll
            for (int qkhe = 0; qkhe < v2::kWideQkheLoop; qkhe++) {
                const int seg_base = qkhe * 8 + rowid * 2;
                Qlocal[g][qkhe].lo = q_stage_lds[consumer_qhead * 16 + seg_base + 0];
                Qlocal[g][qkhe].hi = q_stage_lds[consumer_qhead * 16 + seg_base + 1];
            }
        }
    }

    // ── Per-group online-softmax accumulators ───────────────────────────
    float   m_running[NumGroups], l_running[NumGroups];
    floatx4 o_running[NumGroups][kVheLoop];
    #pragma unroll
    for (int g = 0; g < NumGroups; g++) {
        m_running[g] = -FLT_MAX;
        l_running[g] = 0.f;
        #pragma unroll
        for (int vhe = 0; vhe < kVheLoop; vhe++)
            o_running[g][vhe] = floatx4{0.f, 0.f, 0.f, 0.f};
    }

    unsigned int v_lane_off[kTLoop];
    #pragma unroll
    for (int t = 0; t < kTLoop; t++) {
        const unsigned int v_slot_base =
            (unsigned int)((t * kTokensPerWarp + rowid * 16) % kBlockSize);
        v_lane_off[t] = (unsigned int)(warpid * 16 + lane16id) * kBlockSize + v_slot_base;
    }

    for (int kbi = kbi_start; kbi < kbi_stop; kbi++)
    {
        const int partition_start_token_idx = kbi * kTParSize;
        if (kbi != kbi_start) __syncthreads();

        ks_lds[warpid * kTokensPerWarp + rowid * 16 + lane16id] = my_ks_carried;

        // ── Load V once (shared by all groups) ──────────────────────────
        pa_u32x4 V_wide[kNWarps][kVheLoop];
        unsigned int v_phys_block_wide[kNWarps];
        #pragma unroll
        for (int v_group = 0; v_group < kNWarps; v_group++) {
            const int v_blk_in_part = (v_group * kTokensPerWarp + rowid * 16) / kBlockSize;
            v_phys_block_wide[v_group] = (unsigned int)bt_lds[kbi & 1][v_blk_in_part];
        }

        const int qkout_token_idx = partition_start_token_idx
                                    + kTokensPerWarp * warpid + rowid * 4;
        // MTP causal mask: query token `gtoken(g)` (0..ql-1) is the
        // (ql-1-gtoken(g))-th token before the sequence end, so it must NOT
        // attend to the speculative tokens that follow it within the same mtp
        // group.  Its causal KV length is context_len-(ql-1-gtoken(g)), applied
        // per-group below.  The interior gate uses the SMALLEST causal length in
        // the group (= gtoken=0's = context_len-(ql-1)) so no masking lane is
        // skipped.  (ql==1 collapses to context_len -> no masking.)
        const int valid_upper_min = context_len - (query_length - 1);
        const bool interior_partition =
            (partition_start_token_idx + kTParSize) <= valid_upper_min;

        floatx4 d_out[NumGroups][kTLoop];
        floatx4 pv_acc[kVheLoop];
        int64_t P_lo_per_g[kNWarps];
        int64_t P_hi_per_g[kNWarps];
        float   partition_qk_max_g[NumGroups];
        float   partition_exp_sum_g[NumGroups];

        // ══ Phase A: QK → softmax for BOTH groups (independent → ILP) ════
        #pragma unroll
        for (int g = 0; g < NumGroups; g++)
        {
            #pragma unroll
            for (int t = 0; t < kTLoop; t++) {
                if (g == 0) {
                    const unsigned int v_phys = v_phys_block_wide[t];
                    const unsigned int v_base_voffset =
                        v_phys * (unsigned int)kv_block_stride + v_lane_off[t];
                    V_wide[t][0] = pa_buffer_load_b128_nt(v_rsrc, v_base_voffset);
                    V_wide[t][1] = pa_buffer_load_b128_nt(v_rsrc, v_base_voffset + kVBytesPerVhe);
                }

                d_out[g][t] = floatx4{0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int qkhe = 0; qkhe < v2::kWideQkheLoop; qkhe++) {
                    d_out[g][t] = pa_mfma16x16x32_fp8_fp8(
                        Klocal_carried[t][qkhe].lo, Qlocal[g][qkhe].lo, d_out[g][t]);
                    d_out[g][t] = pa_mfma16x16x32_fp8_fp8(
                        Klocal_carried[t][qkhe].hi, Qlocal[g][qkhe].hi, d_out[g][t]);
                }
                const float* ks_row =
                    &ks_lds[warpid * kTokensPerWarp + t * 16 + rowid * 4];
                const floatx4 ks4 = *reinterpret_cast<const floatx4*>(ks_row);
                d_out[g][t] *= ks4;
            }

            if (!interior_partition) {
                // Per-(group,lane) causal KV cutoff for this lane's query token.
                const int valid_upper_g = context_len - (query_length - 1 - gtoken(g));
                #pragma unroll
                for (int t = 0; t < kTLoop; t++) {
                    const int local_token_idx = qkout_token_idx + t * 16;
                    #pragma unroll
                    for (int i = 0; i < 4; i++)
                        if ((local_token_idx + i) >= valid_upper_g) d_out[g][t][i] = -FLT_MAX;
                }
            }

            float qk_max = -FLT_MAX;
            #pragma unroll
            for (int t = 0; t < kTLoop; t++)
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    qk_max = fmaxf(qk_max, d_out[g][t][i]);
            qk_max = fmaxf(qk_max, pa_shfl_xor_32(qk_max));
            qk_max = fmaxf(qk_max, pa_shfl_xor_within_32<16>(qk_max));

            const float qbase         = qk_base_log2[g];
            const float qk_max_scaled = qbase * qk_max;
            const float neg_qk_max_sc = -qk_max_scaled;
            floatx4 exp_sum_v4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int t = 0; t < kTLoop; t++) {
                floatx4 v;
                v[0] = __builtin_amdgcn_exp2f(__builtin_fmaf(qbase, d_out[g][t][0], neg_qk_max_sc));
                v[1] = __builtin_amdgcn_exp2f(__builtin_fmaf(qbase, d_out[g][t][1], neg_qk_max_sc));
                v[2] = __builtin_amdgcn_exp2f(__builtin_fmaf(qbase, d_out[g][t][2], neg_qk_max_sc));
                v[3] = __builtin_amdgcn_exp2f(__builtin_fmaf(qbase, d_out[g][t][3], neg_qk_max_sc));
                d_out[g][t] = v;
                exp_sum_v4 += v;
            }
            float exp_sum = exp_sum_v4[0] + exp_sum_v4[1] + exp_sum_v4[2] + exp_sum_v4[3];
            exp_sum = exp_sum + pa_shfl_xor_32(exp_sum);
            exp_sum = exp_sum + pa_shfl_xor_within_32<16>(exp_sum);

            if (laneid < 16) {
                const int slot = lane16id * kSharedQkStride + warpid * 2;
                shared_qk[g][slot + 0] = qk_max_scaled;
                shared_qk[g][slot + 1] = exp_sum;
            }
        }

        const int kbi_next_pf = (kbi + 1 < kbi_stop) ? (kbi + 1) : (kbi_stop - 1);
        stage_bt(kbi_next_pf);
        __syncthreads();                 // Barrier 1/3: publish shared_qk[*] + bt_lds
        prefetch_k(kbi_next_pf);

        // ══ Phase B: cross-warp softmax reduce + P-pack for BOTH groups ══
        #pragma unroll
        for (int g = 0; g < NumGroups; g++)
        {
            float partition_qk_max  = -FLT_MAX;
            float partition_exp_sum = 0.f;
            float warp_scale;
            {
                float warp_qk_max[kNWarps];
                float warp_exp_sum[kNWarps];
                const int base = lane16id * kSharedQkStride;
                #pragma unroll
                for (int w = 0; w < kNWarps; w++) {
                    warp_qk_max[w]   = shared_qk[g][base + w * 2 + 0];
                    warp_exp_sum[w]  = shared_qk[g][base + w * 2 + 1];
                    partition_qk_max = fmaxf(partition_qk_max, warp_qk_max[w]);
                }
                float warp_qk_max_exp[kNWarps];
                #pragma unroll
                for (int w = 0; w < kNWarps; w++) {
                    warp_qk_max_exp[w] = __builtin_amdgcn_exp2f(warp_qk_max[w] - partition_qk_max);
                    partition_exp_sum += warp_exp_sum[w] * warp_qk_max_exp[w];
                }
                warp_scale = warp_qk_max_exp[warpid];
            }
            partition_qk_max_g[g]  = partition_qk_max;   // carried to Phase C
            partition_exp_sum_g[g] = partition_exp_sum;

            const float p_pack_scale = HasPScale ? (warp_scale * p_scale_lane) : warp_scale;
            uint32_t* __restrict__ shared_p32 =
                reinterpret_cast<uint32_t*>(shared_logits[g]);
            #pragma unroll
            for (int t = 0; t < kTLoop; t++) {
                d_out[g][t] *= p_pack_scale;
                const uint32_t pk = pa_pk_fp8x4(d_out[g][t][0], d_out[g][t][1], d_out[g][t][2], d_out[g][t][3]);
                const int idx = v0::shared_p32_index(warpid, t, lane16id, rowid);
                shared_p32[idx] = pk;
            }
        }
        __syncthreads();                 // Barrier 2/3: publish P[*]

        // ══ Phase C: PV MFMA + online-softmax accumulate for BOTH groups ══
        #pragma unroll
        for (int g = 0; g < NumGroups; g++)
        {
            {
                const uint32_t* __restrict__ shared_p32_r =
                    reinterpret_cast<const uint32_t*>(shared_logits[g]);
                #pragma unroll
                for (int v_group = 0; v_group < kNWarps; v_group++) {
                    _T8x8 P_lo_pack, P_hi_pack;
                    P_lo_pack.b8x4[0] =
                        shared_p32_r[v0::shared_p32_index(v_group, rowid, lane16id, 0)];
                    P_lo_pack.b8x4[1] =
                        shared_p32_r[v0::shared_p32_index(v_group, rowid, lane16id, 1)];
                    P_hi_pack.b8x4[0] =
                        shared_p32_r[v0::shared_p32_index(v_group, rowid, lane16id, 2)];
                    P_hi_pack.b8x4[1] =
                        shared_p32_r[v0::shared_p32_index(v_group, rowid, lane16id, 3)];
                    P_lo_per_g[v_group] = P_lo_pack.i64;
                    P_hi_per_g[v_group] = P_hi_pack.i64;
                }
            }

            #pragma unroll
            for (int vhe = 0; vhe < kVheLoop; vhe++)
                pv_acc[vhe] = floatx4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int v_group = 0; v_group < kNWarps; v_group++) {
                const int64_t P_lo = P_lo_per_g[v_group];
                const int64_t P_hi = P_hi_per_g[v_group];
                #pragma unroll
                for (int vhe = 0; vhe < kVheLoop; vhe++) {
                    const pa_u32x4 V_chunk = V_wide[v_group][vhe];
                    pv_acc[vhe] = pa_mfma16x16x32_fp8_fp8(
                        pa_u32x4_low_long(V_chunk), P_lo, pv_acc[vhe]);
                    pv_acc[vhe] = pa_mfma16x16x32_fp8_fp8(
                        pa_u32x4_high_long(V_chunk), P_hi, pv_acc[vhe]);
                }
            }

            const float partition_qk_max  = partition_qk_max_g[g];
            const float partition_exp_sum = partition_exp_sum_g[g];
            if (partition_qk_max <= m_running[g] && m_running[g] > -FLT_MAX) {
                const float beta = __builtin_amdgcn_exp2f(partition_qk_max - m_running[g]);
                l_running[g] += beta * partition_exp_sum;
                #pragma unroll
                for (int vhe = 0; vhe < kVheLoop; vhe++)
                    o_running[g][vhe] += beta * pv_acc[vhe];
            } else {
                const float m_new = fmaxf(m_running[g], partition_qk_max);
                const float alpha = (m_running[g] > -FLT_MAX)
                    ? __builtin_amdgcn_exp2f(m_running[g] - m_new) : 0.f;
                const float beta  = __builtin_amdgcn_exp2f(partition_qk_max - m_new);
                l_running[g] = alpha * l_running[g] + beta * partition_exp_sum;
                #pragma unroll
                for (int vhe = 0; vhe < kVheLoop; vhe++)
                    o_running[g][vhe] = alpha * o_running[g][vhe] + beta * pv_acc[vhe];
                m_running[g] = m_new;
            }
        }
    }

    // ── Finalize + write exp_sums/max_logits + output, per group ────────
    constexpr int kGqa4_   = (kGqaRatio + 3) / 4;
    constexpr int kRowsHere = kMtp * kGqa4_;
    const int64_t maxp = static_cast<int64_t>(ws_maxp);
    const int64_t hsz_maxp_mult =
        static_cast<int64_t>(kHeadSize) * static_cast<int64_t>(ws_maxp);

    #pragma unroll
    for (int g = 0; g < NumGroups; g++)
    {
        const float inv_l = __fdividef(1.f, l_running[g] + 1e-6f);
        const float post_scale = HasPScale
            ? (inv_l * v_scale_perhead * p_scale_inv_lane)
            : (inv_l * v_scale_perhead);
        _B16x4 outelems[kVheLoop];
        #pragma unroll
        for (int vhe = 0; vhe < kVheLoop; vhe++)
            outelems[vhe] = pa_from_floatx4<output_t>(o_running[g][vhe] * post_scale);

        if (warpid == 0 && rowid == 0 && lane16id < kMtp * kGqaRatio && valid_tok(g)) {
            const int head_idx = lane16id & (kGqaRatio - 1);
            const int64_t off =
                  (seq_qrow_base + gtoken(g)) * static_cast<int64_t>(total_num_heads) * maxp
                + (static_cast<int64_t>(wg_start_head_idx) + head_idx) * maxp
                + static_cast<int64_t>(fp_idx);
            max_logits[off] = m_running[g] * kInvLog2E;
            exp_sums[off]   = l_running[g];
        }

        __syncthreads();
        #pragma unroll
        for (int vhe = 0; vhe < kVheLoop; vhe++) {
            const int idx = v0::shared_logits_index(warpid, vhe, lane16id, rowid);
            _T8x8 cell;
            cell.b16x4 = outelems[vhe];
            shared_logits[0][idx] = cell;   // finalize scratch: serial-per-g w/ barriers
        }
        __syncthreads();

        if (warpid == 0) {
            const int head_elem_idx = lane16id * 8;
            if (head_elem_idx < kHeadSize) {
                #pragma unroll
                for (int local_row_idx = 0; local_row_idx < kRowsHere; local_row_idx++) {
                    const int q_tok_w  = local_row_idx >> 1;
                    const int head_quad = local_row_idx & 1;
                    const int packed_lane = q_tok_w * 8 + head_quad * 4 + rowid;
                    const int offset1 = (head_elem_idx / 16) % 4;
                    const int offset2 = head_elem_idx / 16 / kNWarps;
                    const int offset3 = (head_elem_idx / 4) % 4;
                    _B16x8 vout;
                    #pragma unroll
                    for (int i = 0; i < 2; i++) {
                        const int idx =
                            v0::shared_logits_index(offset1, offset2, packed_lane, offset3 + i);
                        vout.xy[i] = shared_logits[0][idx].b16x4;
                    }
                    const int head_idx = head_quad * 4 + rowid;
                    const int gtok_w = g * kMtp + q_tok_w;
                    if (head_idx < kGqaRatio && gtok_w < query_length) {
                        const int64_t out_head_idx =
                            static_cast<int64_t>(wg_start_head_idx + head_idx);
                        output_t* out_ptr = out
                            + (seq_qrow_base + gtok_w) * total_num_heads * hsz_maxp_mult
                            + out_head_idx * hsz_maxp_mult
                            + fp_idx * kHeadSize
                            + head_elem_idx;
                        *reinterpret_cast<_B16x8*>(out_ptr) = vout;
                    }
                }
            }
        }
    }
}

} // namespace pa_fp8_gqa
