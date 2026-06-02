// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026 Page_Attetion_GQA_fp8 project
//
// FP8 paged-attention decode main kernel (v0).
//
// Target config (must match `_check_default_config`):
//   - dtype       : fp8 (e4m3 fnuz on gfx942/MI308)
//   - num_q_heads : 8       (GQA_RATIO = 8)
//   - num_kv_heads: 1
//   - head_size   : 128
//   - block_size  : 16
//   - mtp / q_len : 1       (1 query token per sequence)
//   - output      : bf16 (or fp16)
//
// MFMA instruction: `mfma_f32_16x16x32_fp8_fp8`
//   - A = 16x32 fp8 (8 fp8 per lane = 1 long)
//   - B = 32x16 fp8 (8 fp8 per lane = 1 long)
//   - C = 16x16 fp32 (4 fp32 per lane = floatx4)
//
// Grid : (num_seqs, max_num_partitions, num_kv_heads)
// Block: 256 threads = 4 warps, 64 KV tokens per warp (kTParSize=256).
//
// FP8 vs BF16 v5 tile differences:
//   - 4 QK MFMAs/warp covering 256 K-elements head_dim each (was 16/warp in BF16)
//     -> Actually same count (4 t × 4 qkhe = 16 BF16 MFMAs; 4 t × 4 qkhe = 16
//        FP8 MFMAs).  But each FP8 MFMA covers 2x the head_dim (32 vs 16), so
//        4 FP8 MFMAs/warp/N-tile cover head_dim=128 exactly.
//   - x = 16 (vs 8 in bf16): kElems16B = 16 fp8 per 16-byte chunk
//   - M = 8 effective (vs 16 in v5): only lane16id ∈ [0..7] is real Q;
//     lanes 8..15 hold duplicate rows.  Wasted compute on duplicates but
//     MFMA latency unchanged.  Soft/write only emit rows 0..7.
//   - PV MFMA K = 32 kv tokens (vs 16 in BF16): need to pack 2 t-tiles
//     worth of P per lane in LDS for one PV MFMA.
//
// Per-tensor scaling:
//   d_out[t] *= q_scale * k_scale * softmax_scale (folded into log2 domain)
//   pv_acc   *= v_scale (post-MFMA, scalar broadcast)

#pragma once

#include "pa_fp8_common.h"

namespace pa_fp8_gqa {

namespace v0 {

// -------- compile-time tile / pack constants ----------------------------
constexpr int kBlockSize     = 16;
constexpr int kHeadSize      = 128;
constexpr int kElems16B_fp8  = 16;                // 16 fp8 per 16-B chunk

constexpr int kNumThreads    = 256;
constexpr int kNWarps        = kNumThreads / WARP_SIZE;          // 4
constexpr int kRowsPerWarp   = WARP_SIZE / 16;                   // 4
constexpr int kGqaRatio      = 8;                                // num_q / num_kv
constexpr int kMtpMax        = 2;                                // supported {1, 2}
constexpr int kTParSize      = 256;
constexpr int kTokensPerWarp = kTParSize / kNWarps;              // 64
constexpr int kTLoop         = kTokensPerWarp / 16;              // 4 (16 N tokens per MFMA)

// Each QK MFMA covers 32 head_dim values along K-dim.
constexpr int kKPerMfma      = 32;
constexpr int kQkheLoop      = kHeadSize / kKPerMfma;            // 4
constexpr int kFp8PerLanePerMfma = 8;                            // 8 fp8 = 1 long

// PV MFMA covers 32 kv tokens per call.  Per warp: 64/32 = 2 calls.
constexpr int kPvTLoop       = kTokensPerWarp / kKPerMfma;       // 2

// LDS layout for P (rescaled softmax, fp8).
// We use 4 fp8 per cell (lower 32 bits of a 64-bit slot — slot reused for
// the bf16 PV write-back later).  Indexed by (warp, t, lane16, rowid).
//   - warp ∈ [0, 4)
//   - t    ∈ [0, kTLoop)
//   - lane16 ∈ [0, 16)
//   - rowid  ∈ [0, kRowsPerWarp)
//
// Banking trick (mtp-safe):
//   slot = (warp*kTLoop + t) * kSlotsPerWarpT + lane16*4 + (rowid ^ lane16)
//
//   - lane16*4 gives each lane16 its own 4-slot block (no collision across
//     lane16 within a (warp, t)).
//   - (rowid ^ lane16) ∈ [0, 16) is the XOR-swizzle for bank-conflict
//     avoidance on 16-wide reads — since this can reach 15, each (warp, t)
//     needs kSlotsPerWarpT = 16*4 + 15 + 1 = 80 slots (vs the 64 you'd get
//     from kRowsPerWarp=4 alone), which fixes the silent slot collision
//     between t=0 lane16∈{13..15} and t=1 lane16∈{0..2} that breaks any Mtp
//     where lanes 8..15 carry real (non-duplicate) data.
//
// Total cells = kNWarps * kTLoop * kSlotsPerWarpT = 4 * 4 * 80 = 1280;
// 8 bytes per cell -> 10 KiB LDS.
constexpr int kSlotsPerWarpT = 80;

__host__ __device__ __forceinline__ constexpr int
shared_logits_index(int warp, int t, int lane16, int rowid)
{
    return (warp * kTLoop + t) * kSlotsPerWarpT
         + lane16 * kRowsPerWarp + (rowid ^ lane16);
}

// V tile bookkeeping (V cache layout
// [num_blocks, num_kv_heads, head_size, block_size] fp8).
//   - kVtLoop : warps-worth (4) of kv-token sub-tiles in head_dim dimension
//   - kVheLoop: 2 head_dim split (128 / (4*16) = 2 — each PV "v" tile
//     handles 16 head_dim positions per warp)
//   - kVtLaneLoop : 2 — covers 2 PV MFMA K=32 chunks per (v, vhe)
//
// For one PV MFMA call (A=V[head_dim,kv]/16x32, B=P[kv,q_row]/32x16):
//   A per lane = 8 fp8 = one (head_dim=lane16id) row's 8 contiguous kv values.
constexpr int kVtLoop      = kNWarps;             // 4 — warps for kv block
constexpr int kVheLoop     = kHeadSize / (kNWarps * 16);    // 2
constexpr int kVtLaneLoop  = kKPerMfma / kFp8PerLanePerMfma; // 4 (8-fp8 lanes per K=32)

} // namespace v0

// ---------------------------------------------------------------------------
// FP8 paged-attention main kernel (v0)
//
// Computes per-partition (exp_sums, max_logits, tmp_out).  Cross-partition
// reduce in `pa_fp8_reduce_kernel`.
//
// Template params:
//   output_t : bf16 / fp16  (NOT fp8; output is the higher-precision dtype)
//
// Layout of inputs (matches `pa_decode_gluon` Python contract):
//   q        : [num_seqs * mtp, num_q_heads, head_size]                fp8
//   k_cache  : [num_blocks, num_kv_heads, head_size/16, block_size, 16] fp8
//   v_cache  : [num_blocks, num_kv_heads, head_size, block_size]        fp8
//   block_tables : [num_seqs, max_num_blocks_per_seq]                   int32
//   context_lens : [num_seqs]                                           int32
//
// Outputs:
//   exp_sums   : [num_seqs * mtp, num_q_heads, max_num_partitions]      fp32
//   max_logits : same shape                                              fp32
//   tmp_out    : [num_seqs * mtp, num_q_heads, max_num_partitions,
//                 head_size]                                             output_t (bf16)
//
// Per-tensor scales (v0):
//   q_scale, k_scale, v_scale — scalar fp32.
// ---------------------------------------------------------------------------
template <typename output_t, int Mtp>
__global__ __launch_bounds__(v0::kNumThreads, 2)
void pa_fp8_main_kernel_v0(
    const __hip_fp8_e4m3_fnuz* __restrict__ q,
    const __hip_fp8_e4m3_fnuz* __restrict__ k_cache,
    const __hip_fp8_e4m3_fnuz* __restrict__ v_cache,
    const float                              softmax_scale,
    const float* __restrict__                q_scale_ptr,   // [1] device fp32
    const float* __restrict__                k_scale_ptr,   // [1]
    const float* __restrict__                v_scale_ptr,   // [1]
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
    output_t* __restrict__                   out
)
{
    using namespace v0;
    constexpr int kMtp = Mtp;
    static_assert(Mtp == 1 || Mtp == 2, "v0 supports Mtp in {1, 2}");

    constexpr float kLog2E    = 1.4426950408889634f;
    constexpr float kInvLog2E = 0.6931471805599453f;
    const auto seq_idx       = blockIdx.x;
    const auto partition_idx = blockIdx.y;
    const auto kv_head_idx   = blockIdx.z;

    const int warpid   = threadIdx.x / WARP_SIZE;
    const int laneid   = threadIdx.x % WARP_SIZE;
    const int lane16id = laneid % 16;
    const int rowid    = laneid / 16;

    const int max_num_partitions = gridDim.y;
    const int total_num_heads    = gridDim.z * kGqaRatio;
    const int context_len               = context_lens[seq_idx];
    const int partition_start_token_idx = partition_idx * kTParSize;
    if (partition_start_token_idx >= context_len)
        return;

    const int wg_start_head_idx    = kv_head_idx * kGqaRatio;
    const int wg_start_kv_head_idx = kv_head_idx;
    const int num_context_blocks   = PAGQA_DIVUP(context_len, kBlockSize);
    const int last_ctx_block       = num_context_blocks - 1;
    const int* block_table_seq     = block_tables + seq_idx * max_num_blocks_per_seq;

    // ----- LDS ------------------------------------------------------------
    // shared_logits: 10 KiB.  Two phases:
    //   Phase A (P staging, post-softmax): 4 fp8 per cell (lower 32 bits).
    //   Phase B (PV result bf16): 8 bytes per cell (_B16x4 of bf16).
    __shared__ _T8x8 shared_logits[kNWarps * kTLoop * kSlotsPerWarpT];

    // Cross-warp qk_max / exp_sum (disjoint from shared_logits).
    __shared__ float shared_qk[kNWarps * 16 * 2];

    // Block-table prefetch.
    __shared__ int bt_lds[kNWarps * kTLoop];
    {
        const int partition_block_start = partition_idx * (kTParSize / kBlockSize);
        if (threadIdx.x < kNWarps * kTLoop)
        {
            const int b = partition_block_start + threadIdx.x;
            bt_lds[threadIdx.x] =
                (b < num_context_blocks) ? block_table_seq[b] : block_table_seq[last_ctx_block];
        }
        __syncthreads();
    }

    // For Mtp=1: q_token = 0; head_for_lane = lane16id & 7 (real for 0..7,
    //            duplicate for 8..15).
    // For Mtp=2: q_token = lane16id >> 3 (= 0 for lanes 0..7, 1 for lanes 8..15);
    //            head_for_lane = lane16id & 7.  All 16 lanes carry real Q
    //            rows, so MFMA's M=16 is fully utilized.
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

    // -----------------------------------------------------------------------
    // Q load — 8 fp8 = 1 long per lane per qkhe; 4 longs total = 32 fp8.
    //
    // Layout: q[seq_idx, head_for_lane, head_dim_offset .. +8] for
    //   head_dim_offset = qkhe * 32 + rowid * 8.
    // -----------------------------------------------------------------------
    int64_t Qlocal[kQkheLoop];
    {
        const int64_t query_row_off =
            (static_cast<int64_t>(seq_idx) * kMtp + q_token_for_lane) * q_stride
            + (wg_start_head_idx + head_for_lane) * kHeadSize;
        const __hip_fp8_e4m3_fnuz* q_row = q + query_row_off;
        #pragma unroll
        for (int qkhe = 0; qkhe < kQkheLoop; qkhe++)
        {
            const int k_off = qkhe * kKPerMfma + rowid * kFp8PerLanePerMfma;
            Qlocal[qkhe] = *reinterpret_cast<const int64_t*>(q_row + k_off);
        }
    }

    // -----------------------------------------------------------------------
    // V load + K load + QK^T MFMA with K[t+1] prefetch + V[v=t] streaming.
    //
    // Strategy (v0.2): interleave V loads with QK MFMA iters so each V
    // tile's gmem latency overlaps the MFMA compute that follows.  Per QK
    // tile `t` we issue 4 V loads (= V[v=t][t_pv=0,1][vhe=0,1]); by the time
    // softmax + sync finishes, all 16 V loads are in flight and overlapped
    // with ~600 cycles of QK compute + softmax.
    //
    // Lane (rowid, lane16id) for MFMA t, qkhe:
    //   loads K[block_for_t, kv_head, head_dim_chunk = (qkhe*32 + rowid*8)/16,
    //          slot = lane16id, intra16 = (qkhe*32 + rowid*8) % 16 .. +8].
    // 8 contiguous fp8 bytes per load.
    // -----------------------------------------------------------------------
    floatx4 d_out[kTLoop];
    int64_t Vlocal[kVtLoop][kPvTLoop][kVheLoop];
    {
        int kphysical_block_number[kTLoop];
        int kphysical_block_offset[kTLoop];
        #pragma unroll
        for (int t = 0; t < kTLoop; t++)
        {
            const int klocal_token_idx  = kTokensPerWarp * warpid + t * 16 + lane16id;
            const int kglobal_token_idx = partition_start_token_idx + klocal_token_idx;
            const int kblock_idx        = warpid * kTLoop + t;
            kphysical_block_number[t]   = bt_lds[kblock_idx];
            kphysical_block_offset[t]   = kglobal_token_idx % kBlockSize;
        }

        // K/V via buffer_load_dwordx2 with IMMEDIATE qkhe offsets — see
        // load_k_tile / load_v_slice comments inside pa_fp8_main_kernel_v1
        // for the per-instruction layout rationale.
        const __amdgpu_buffer_rsrc_t k_rsrc =
            pa_make_buffer_rsrc(k_cache + wg_start_kv_head_idx * kv_head_stride);
        const __amdgpu_buffer_rsrc_t v_rsrc =
            pa_make_buffer_rsrc(v_cache + wg_start_kv_head_idx * kv_head_stride);
        const unsigned int v_slot_in_block =
            (unsigned int)(rowid & 1) * kFp8PerLanePerMfma;

        constexpr unsigned int kKBytesPerQkhe =
            (kKPerMfma / kElems16B_fp8) * (kBlockSize * kElems16B_fp8); // 512
        constexpr unsigned int kVBytesPerVhe =
            (unsigned int)(kNWarps * 16 * kBlockSize);                  // 1024
        const unsigned int rowid_byte_off =
            ((unsigned int)rowid >> 1) * (kBlockSize * kElems16B_fp8)
            + ((unsigned int)rowid & 1u) * kFp8PerLanePerMfma;

        int64_t Klocal[kTLoop][kQkheLoop];

        auto load_k_tile = [&](int t) __attribute__((always_inline))
        {
            const unsigned int kblock_number =
                (unsigned int)kphysical_block_number[t];
            const unsigned int k_base_voffset =
                kblock_number * (unsigned int)kv_block_stride
                + (unsigned int)kphysical_block_offset[t] * kElems16B_fp8
                + rowid_byte_off;
            Klocal[t][0] = pa_buffer_load_b64(k_rsrc, k_base_voffset);
            Klocal[t][1] = pa_buffer_load_b64(k_rsrc, k_base_voffset +     kKBytesPerQkhe);
            Klocal[t][2] = pa_buffer_load_b64(k_rsrc, k_base_voffset + 2 * kKBytesPerQkhe);
            Klocal[t][3] = pa_buffer_load_b64(k_rsrc, k_base_voffset + 3 * kKBytesPerQkhe);
        };

        // Issue 4 V loads at "v slice = v" — V[v][t_pv=0,1][vhe=0,1].
        auto load_v_slice = [&](int v) __attribute__((always_inline))
        {
            #pragma unroll
            for (int t_pv = 0; t_pv < kPvTLoop; t_pv++)
            {
                const unsigned int v_phys_block =
                    (unsigned int)bt_lds[v * kRowsPerWarp + t_pv * 2 + rowid / 2];
                const unsigned int v_base_voffset =
                    v_phys_block * (unsigned int)kv_block_stride
                    + (unsigned int)(warpid * 16 + lane16id) * kBlockSize
                    + v_slot_in_block;
                Vlocal[v][t_pv][0] = pa_buffer_load_b64(v_rsrc, v_base_voffset);
                Vlocal[v][t_pv][1] = pa_buffer_load_b64(v_rsrc, v_base_voffset + kVBytesPerVhe);
            }
        };

        // IGLP-style scheduling hint: interleave VMEM_READ with MFMA so
        // K[t+1]+V[t] loads overlap with MFMA[t] (mirrors gluon).  Hard
        // sched_barrier(0) clusters loads/MFMAs separately → ~600 cyc
        // s_waitcnt stall every tile.  Group barrier biases the scheduler
        // to mix them.
        load_k_tile(0);

        #pragma unroll
        for (int t = 0; t < kTLoop; t++)
        {
            if (t + 1 < kTLoop) {
                load_k_tile(t + 1);
            }
            load_v_slice(t);

            d_out[t] = floatx4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int qkhe = 0; qkhe < kQkheLoop; qkhe++)
            {
                d_out[t] = pa_mfma16x16x32_fp8_fp8(
                    Klocal[t][qkhe], Qlocal[qkhe], d_out[t]);
            }
            pa_apply_qk_token_scales_for_block(
                d_out[t], k_scale_ptr, kphysical_block_number[t], rowid * 4,
                gridDim.z, kv_head_idx, kBlockSize, qk_base_log2);

            __builtin_amdgcn_sched_group_barrier(/*MFMA*/0x008, 4, 0);
            __builtin_amdgcn_sched_group_barrier(/*VMEM_READ*/0x020, 4, 0);
        }
    } // Klocal scope

    // -----------------------------------------------------------------------
    // Per-warp softmax (mtp=1 → single qk_max / exp_sum per lane).
    // Lanes 8..15 produce duplicate values; we mask them out at write time.
    // -----------------------------------------------------------------------
    const int qkout_token_idx = partition_start_token_idx
                                + kTokensPerWarp * warpid + rowid * 4;
    float qk_max  = -FLT_MAX;
    float exp_sum = 0.f;
    {
        const int valid_upper = context_len;
        const bool interior_partition =
            (partition_start_token_idx + kTParSize) <= valid_upper;

        // Unified path (see v1 for the same trick): pre-mask out-of-range
        // d_out values to -FLT_MAX in the boundary case so a single set of
        // 16 v_exp_f32 instructions is shared by both branches.
        if (!interior_partition) {
            #pragma unroll
            for (int t = 0; t < kTLoop; t++) {
                const int local_token_idx = qkout_token_idx + t * 16;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    if ((local_token_idx + i) >= valid_upper)
                        d_out[t][i] = -FLT_MAX;
                }
            }
        }
        {
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
                d_out[t]     = v;
                exp_sum_v4  += v;
            }
            exp_sum = exp_sum_v4[0] + exp_sum_v4[1] + exp_sum_v4[2] + exp_sum_v4[3];
            exp_sum = exp_sum + pa_shfl_xor_32(exp_sum);
            exp_sum = exp_sum + pa_shfl_xor_within_32<16>(exp_sum);
        }
    }

    // -----------------------------------------------------------------------
    // Cross-warp merge qk_max / exp_sum.
    // -----------------------------------------------------------------------
    // shared_qk layout repacked for ds_read_b128 (per-lane contiguous):
    //   shared_qk[lane16id * (kNWarps * 2) + warp * 2 + 0] = qk_max
    //   shared_qk[lane16id * (kNWarps * 2) + warp * 2 + 1] = exp_sum
    if (laneid < 16)
    {
        const int slot = lane16id * (kNWarps * 2) + warpid * 2;
        shared_qk[slot + 0] = qk_max;
        shared_qk[slot + 1] = exp_sum;
    }
    __syncthreads();

    float partition_qk_max = -FLT_MAX;
    float partition_exp_sum = 0.f;
    float inv_sum_scale;
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
            warp_qk_max_exp[w] = __builtin_amdgcn_exp2f(warp_qk_max[w] - partition_qk_max);
            partition_exp_sum += warp_exp_sum[w] * warp_qk_max_exp[w];
        }
        inv_sum_scale = __fdividef(1.f, partition_exp_sum + 1e-6f)
                        * warp_qk_max_exp[warpid];
    }

    // -----------------------------------------------------------------------
    // Rescale d_out by inv_sum_scale and convert to FP8.  Stage 4 fp8 per
    // (warp, t, lane16, rowid) cell (lower 32 bits of slot; upper 32 unused).
    // -----------------------------------------------------------------------
    #pragma unroll
    for (int t = 0; t < kTLoop; t++)
    {
        d_out[t] *= inv_sum_scale * p_scale_perhead;
        const uint32_t pk = pa_pk_fp8x4(
            d_out[t][0], d_out[t][1], d_out[t][2], d_out[t][3]);
        const int idx = v0::shared_logits_index(warpid, t, lane16id, rowid);
        // Write into the lower 32 bits of the slot.  Use full 64-bit store
        // (upper bits zeroed) — simpler, no LDS partial-write hazards.
        shared_logits[idx].i64 = static_cast<int64_t>(pk);
    }

    // Per-(q_token, head) max_logits / exp_sums to gmem (one thread per row).
    //   Mtp=1: lanes 0..7 (q_token=0, head=lane16id).
    //   Mtp=2: lanes 0..15 (q_token=lane16id>>3, head=lane16id&7).
    // Constrain warpid==0 so each (q_token, head, partition) is written by a
    // single warp.
    if (warpid == 0 && rowid == 0 && lane16id < kMtp * kGqaRatio)
    {
        const int head_idx = lane16id & (kGqaRatio - 1);
        const int64_t query_start_off = static_cast<int64_t>(seq_idx) * kMtp;
        const int64_t maxp = static_cast<int64_t>(max_num_partitions);
        const int64_t offset =
              static_cast<int64_t>(query_start_off + q_token_for_lane)
                  * static_cast<int64_t>(total_num_heads) * maxp
            + (static_cast<int64_t>(wg_start_head_idx) + head_idx) * maxp
            + static_cast<int64_t>(partition_idx);
        max_logits[offset] = partition_qk_max * kInvLog2E;
        exp_sums[offset]   = partition_exp_sum;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // PV MFMA.
    //
    // For each (v=warp_src, t_pv ∈ [0, kPvTLoop), vhe ∈ [0, kVheLoop)):
    //   pv_acc[vhe] += mfma_f32_16x16x32_fp8_fp8(A=V, B=P)
    //
    // A=V at lane (rowid_r, lane16id_r) holds 8 fp8 of
    //   V[head_dim = vhe*64 + warpid*16 + lane16id_r,
    //     kv_token = v*64 + t_pv*32 + rowid_r*8 + 0..7]
    //   (pre-loaded into Vlocal[v][t_pv][vhe] before softmax).
    //
    // B=P at lane (rowid_r, lane16id_r) holds 8 fp8 of
    //   P[q_row = lane16id_r, kv_token = v*64 + t_pv*32 + rowid_r*8 + 0..7]
    //   sourced from shared_logits (staged after softmax).  Specifically:
    //     d_out[t_s][i=0..3] at producer lane (rowid_s, lane16id_s=L)
    //       = P[q_row=L, kv_token = warpid_src*64 + t_s*16 + rowid_s*4 + i]
    //     For warpid_src = v and target kv_token = v*64 + t_pv*32 + rowid_r*8 + j:
    //       t_s        = t_pv*2 + rowid_r/2
    //       rowid_s_lo = (rowid_r & 1) * 2       (j=0..3)
    //       rowid_s_hi = (rowid_r & 1) * 2 + 1   (j=4..7)
    //     lane16id_s = L = lane16id_r
    // -----------------------------------------------------------------------
    floatx4 pv_acc[kVheLoop];
    #pragma unroll
    for (int vhe = 0; vhe < kVheLoop; vhe++)
        pv_acc[vhe] = floatx4{0.f, 0.f, 0.f, 0.f};

    // Batch all P LDS reads up-front (single lgkmcnt drain via progressive
    // lgkmcnt(N) waits during the MFMA chain).
    int64_t P_pack_all[kVtLoop][kPvTLoop];
    #pragma unroll
    for (int v = 0; v < kVtLoop; v++)
    {
        #pragma unroll
        for (int t_pv = 0; t_pv < kPvTLoop; t_pv++)
        {
            const int t_s   = t_pv * 2 + rowid / 2;
            const int rs_lo = (rowid & 1) * 2;
            const int rs_hi = rs_lo + 1;
            _T8x8 P_pack;
            P_pack.b8x4[0] = static_cast<uint32_t>(
                shared_logits[v0::shared_logits_index(v, t_s, lane16id, rs_lo)].i64);
            P_pack.b8x4[1] = static_cast<uint32_t>(
                shared_logits[v0::shared_logits_index(v, t_s, lane16id, rs_hi)].i64);
            P_pack_all[v][t_pv] = P_pack.i64;
        }
    }

    #pragma unroll
    for (int v = 0; v < kVtLoop; v++)
    {
        #pragma unroll
        for (int t_pv = 0; t_pv < kPvTLoop; t_pv++)
        {
            #pragma unroll
            for (int vhe = 0; vhe < kVheLoop; vhe++)
            {
                pv_acc[vhe] = pa_mfma16x16x32_fp8_fp8(
                    Vlocal[v][t_pv][vhe], P_pack_all[v][t_pv], pv_acc[vhe]);
            }
        }
    }

    // Apply V dequant scale (per KV head).
    #pragma unroll
    for (int vhe = 0; vhe < kVheLoop; vhe++)
        pv_acc[vhe] *= v_scale_perhead * p_scale_inv_perhead;

    // -----------------------------------------------------------------------
    // Convert PV to output_t and stage to LDS for warp-0 gmem write.
    //
    // After PV MFMA: lane (rowid_r, lane16id_r) holds pv_acc[vhe] (floatx4)
    //   = PV[head_dim=lane16id_r ?, q_row = rowid_r*4 + 0..3 ?]
    //
    // Actually per v5 convention (which we mirror): lane16id encodes q_row
    // post-PV-MFMA (because MFMA output mapping puts the "M" of C = the
    // first matrix's M = our A=V's M = head_dim ... but v5 treats this as
    // q_row, so it's just a convention mismatch with my derivation).
    //
    // Mirror v5: lane16id encodes (q_token, head); rowid*4 + i covers
    // head_dim chunks.
    // -----------------------------------------------------------------------
    _B16x4 outelems[kVheLoop];
    #pragma unroll
    for (int vhe = 0; vhe < kVheLoop; vhe++)
        outelems[vhe] = pa_from_floatx4<output_t>(pv_acc[vhe]);

    __syncthreads();

    // Stage outelems to LDS (re-use shared_logits, 8 bytes per cell).
    #pragma unroll
    for (int vhe = 0; vhe < kVheLoop; vhe++)
    {
        const int idx = v0::shared_logits_index(warpid, vhe, lane16id, rowid);
        _T8x8 cell;
        cell.b16x4 = outelems[vhe];
        shared_logits[idx] = cell;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Output write-back (warp 0).
    //
    // tmp_out gmem: [num_seqs*Mtp, num_heads, max_num_partitions, head_size]
    //   Mtp=1, kGqaRatio=8, kGqa4_=2: kRowsHere = 2 head-quads per partition.
    //   Mtp=2, kGqaRatio=8, kGqa4_=2: kRowsHere = 4 (q_token, head_quad) pairs.
    //
    // packed_lane maps (q_token, head_quad, local_head_idx_in_quad=rowid) to the
    // lane16id slot in LDS where pv_acc[vhe] was staged.  For Mtp=2 the upper
    // 8 lane16id slots [8..15] hold q_token=1 outputs (vs duplicates in Mtp=1).
    // -----------------------------------------------------------------------
    if (warpid == 0)
    {
        const int64_t query_start_off = static_cast<int64_t>(seq_idx) * kMtp;
        constexpr int kGqa4_  = (kGqaRatio + 3) / 4;
        constexpr int kRowsHere = kMtp * kGqa4_;  // Mtp=1 -> 2; Mtp=2 -> 4

        const int head_elem_idx = lane16id * 8;
        if (head_elem_idx < kHeadSize)
        {
            const int64_t hsz_maxp_mult =
                static_cast<int64_t>(kHeadSize) * static_cast<int64_t>(max_num_partitions);
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
                        + partition_idx * kHeadSize
                        + head_elem_idx;
                    *reinterpret_cast<_B16x8*>(out_ptr) = vout;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// pa_fp8_main_kernel_v1 — fat-CTA variant.
//
// Same compute as v0 but each CTA processes `num_kblocks_per_fat_part`
// consecutive 256-token sub-blocks instead of exactly one.  Online-softmax
// accumulator state (m_running, l_running, o_running) is kept across the
// outer loop, so the CTA emits ONE (max_logit, exp_sum, tmp_out) per fat
// partition — matching gluon's `get_recommended_splits` scheduling (≤8
// splits per seq instead of cdiv(ctx,256)).
//
// Grid : (num_seqs, num_fat_partitions, num_kv_heads)
// Block: 256 threads, same LDS / register layout as v0.
//
// Extra runtime args:
//   num_kblocks_per_fat_part : ceil_div(total_256_blocks, num_fat_partitions)
//                              — every CTA loops over up to this many
//                              256-token sub-blocks (last fat may stop early).
//
// Output tensor layout uses `max_num_partitions = num_fat_partitions`,
// which the host launcher passes through; the reduce kernel reads the same
// shape and now reduces over a much smaller partition axis.
// ---------------------------------------------------------------------------
// v1 (fat-CTA, long-ctx primary path).  launch_bounds(256, 2) → 2 CTAs/CU.
// VGPR cap is 256/2 = 128 per wave on gfx942 (256 SGPRs, 512 VGPRs/CU /2).
// At 2 CTAs/CU each CTA holds ~half the HBM bandwidth (~1.45 TB/s) but the
// second CTA hides single-CTA HBM-latency gaps, mirroring gluon's per-CU
// load-issue rate while still respecting the 128-VGPR target.
template <typename output_t, int Mtp>
__global__ __launch_bounds__(v0::kNumThreads, 2)
void pa_fp8_main_kernel_v1(
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
    const int                                num_kblocks_per_fat_part
)
{
    using namespace v0;
    constexpr int kMtp = Mtp;
    static_assert(Mtp == 1 || Mtp == 2, "v1 supports Mtp in {1, 2}");

    constexpr float kLog2E    = 1.4426950408889634f;
    constexpr float kInvLog2E = 0.6931471805599453f;

    const auto seq_idx     = blockIdx.x;
    const auto fp_idx      = blockIdx.y;       // fat-partition index
    const auto kv_head_idx = blockIdx.z;

    const int warpid   = threadIdx.x / WARP_SIZE;
    const int laneid   = threadIdx.x % WARP_SIZE;
    const int lane16id = laneid % 16;
    const int rowid    = laneid / 16;

    const int num_fat_partitions = gridDim.y;
    const int total_num_heads    = gridDim.z * kGqaRatio;
    const int context_len        = context_lens[seq_idx];
    const int total_num_kblocks  = PAGQA_DIVUP(context_len, kTParSize);

    const int kbi_start = fp_idx * num_kblocks_per_fat_part;
    const int kbi_stop_raw = kbi_start + num_kblocks_per_fat_part;
    const int kbi_stop  = (kbi_stop_raw < total_num_kblocks)
                              ? kbi_stop_raw : total_num_kblocks;

    if (kbi_start >= total_num_kblocks) return;

    const int wg_start_head_idx    = kv_head_idx * kGqaRatio;
    const int wg_start_kv_head_idx = kv_head_idx;
    const int num_context_blocks   = PAGQA_DIVUP(context_len, kBlockSize);
    const int last_ctx_block       = num_context_blocks - 1;
    const int* block_table_seq     = block_tables + seq_idx * max_num_blocks_per_seq;

    __shared__ _T8x8 shared_logits[kNWarps * kTLoop * kSlotsPerWarpT];
    __shared__ float shared_qk[kNWarps * 16 * 2];

    // ----- Buffer V# resource descriptors (uniform across wave) ----------
    // Built once per CTA so K/V hot-path loads can use buffer_load_dwordx2
    // (uniform base in SGPR + 32-bit per-lane voffset) instead of flat
    // global_load (per-lane 64-bit address).  This is the single largest
    // lever vs gluon: their kernel uses buffer_load while ours (before this
    // change) had ZERO buffer_load instructions in the inner loop.
    //
    // Base = k_cache + kv_head_idx * kv_head_stride; voffset per lane =
    //   kblock_number * kv_block_stride + (intra-block byte offset).
    // For the largest config (bs=256, ctx=131072, fp8) the max kblock_number
    // is ~2M and kv_block_stride is 2KB, so voffset stays within uint32.
    const __amdgpu_buffer_rsrc_t k_rsrc =
        pa_make_buffer_rsrc(k_cache + wg_start_kv_head_idx * kv_head_stride);
    const __amdgpu_buffer_rsrc_t v_rsrc =
        pa_make_buffer_rsrc(v_cache + wg_start_kv_head_idx * kv_head_stride);

    // ----- Q load (once per CTA, reused across all outer iterations) ------
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

    int64_t Qlocal[kQkheLoop];
    {
        const int64_t query_row_off =
            (static_cast<int64_t>(seq_idx) * kMtp + q_token_for_lane) * q_stride
            + (wg_start_head_idx + head_for_lane) * kHeadSize;
        const __hip_fp8_e4m3_fnuz* q_row = q + query_row_off;
        #pragma unroll
        for (int qkhe = 0; qkhe < kQkheLoop; qkhe++)
        {
            const int k_off = qkhe * kKPerMfma + rowid * kFp8PerLanePerMfma;
            Qlocal[qkhe] = *reinterpret_cast<const int64_t*>(q_row + k_off);
        }
    }

    // ----- Online softmax accumulator state -------------------------------
    float   m_running = -FLT_MAX;
    float   l_running = 0.f;
    floatx4 o_running[kVheLoop];
    #pragma unroll
    for (int vhe = 0; vhe < kVheLoop; vhe++)
        o_running[vhe] = floatx4{0.f, 0.f, 0.f, 0.f};

    // =====================================================================
    // OUTER LOOP — process kblocks [kbi_start, kbi_stop)
    // =====================================================================
    for (int kbi = kbi_start; kbi < kbi_stop; kbi++)
    {
        const int partition_start_token_idx = kbi * kTParSize;
        const int partition_block_start = kbi * (kTParSize / kBlockSize);

        // Wait for previous iter's PV MFMA to be done reading shared_logits.
        if (kbi != kbi_start) __syncthreads();

        // ----- K load + V prefetch + QK MFMA (mirrors v0) -----------------
        floatx4 d_out[kTLoop];
        int64_t Vlocal[kVtLoop][kPvTLoop][kVheLoop];
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

            const unsigned int v_slot_in_block =
                (unsigned int)(rowid & 1) * kFp8PerLanePerMfma;

            int64_t Klocal[kTLoop][kQkheLoop];

            // Hoist ALL V block_table reads to top of outer iter so they
            // batch with K block_table reads → single lgkmcnt drain.  This
            // lets the global_load_dword burst at the start of the outer
            // iter, instead of being interleaved with QK MFMA (which would
            // cause repeated lgkmcnt(0) waits inside the t-loop).
            unsigned int v_phys_block_arr[kVtLoop][kPvTLoop];
            #pragma unroll
            for (int v = 0; v < kVtLoop; v++)
            {
                #pragma unroll
                for (int t_pv = 0; t_pv < kPvTLoop; t_pv++)
                {
                    const int v_bt_g_idx = partition_block_start
                                           + v * kRowsPerWarp + t_pv * 2 + rowid / 2;
                    const int v_bt_g_idx_safe =
                        (v_bt_g_idx < num_context_blocks) ? v_bt_g_idx : last_ctx_block;
                    v_phys_block_arr[v][t_pv] =
                        (unsigned int)block_table_seq[v_bt_g_idx_safe];
                }
            }

            // K/V loads via buffer_load_dwordx2 with IMMEDIATE offsets.
            constexpr unsigned int kKBytesPerQkhe =
                (kKPerMfma / kElems16B_fp8) * (kBlockSize * kElems16B_fp8); // 2*256 = 512
            const unsigned int rowid_byte_off =
                ((unsigned int)rowid >> 1) * (kBlockSize * kElems16B_fp8)
                + ((unsigned int)rowid & 1u) * kFp8PerLanePerMfma;

            auto load_k_tile = [&](int t) __attribute__((always_inline))
            {
                const unsigned int kblock_number =
                    (unsigned int)kphysical_block_number[t];
                const unsigned int k_base_voffset =
                    kblock_number * (unsigned int)kv_block_stride
                    + (unsigned int)kphysical_block_offset[t] * kElems16B_fp8
                    + rowid_byte_off;
                Klocal[t][0] = pa_buffer_load_b64(k_rsrc, k_base_voffset);
                Klocal[t][1] = pa_buffer_load_b64(k_rsrc, k_base_voffset +     kKBytesPerQkhe);
                Klocal[t][2] = pa_buffer_load_b64(k_rsrc, k_base_voffset + 2 * kKBytesPerQkhe);
                Klocal[t][3] = pa_buffer_load_b64(k_rsrc, k_base_voffset + 3 * kKBytesPerQkhe);
            };

            constexpr unsigned int kVBytesPerVhe = (unsigned int)(kNWarps * 16 * kBlockSize); // 1024
            auto load_v_slice = [&](int v) __attribute__((always_inline))
            {
                #pragma unroll
                for (int t_pv = 0; t_pv < kPvTLoop; t_pv++)
                {
                    const unsigned int v_phys_block = v_phys_block_arr[v][t_pv];
                    const unsigned int v_base_voffset =
                        v_phys_block * (unsigned int)kv_block_stride
                        + (unsigned int)(warpid * 16 + lane16id) * kBlockSize
                        + v_slot_in_block;
                    Vlocal[v][t_pv][0] = pa_buffer_load_b64(v_rsrc, v_base_voffset);
                    Vlocal[v][t_pv][1] = pa_buffer_load_b64(v_rsrc, v_base_voffset + kVBytesPerVhe);
                }
            };

            // K load: burst N at start of outer iter (gluon-style upfront
            // load).  Interleave V loads with QK MFMA so V latency hides
            // under QK MFMA's ~16-cycle critical path.  PV MFMA later waits
            // for V to be ready via natural vmcnt-based progressive drain.
            //
            // Two-stage prefetch: load_k_tile(0) eagerly so the first
            // QK MFMA has K ready immediately, then issue load_k_tile(t+1)
            // INSIDE the t-loop (one iteration ahead) to keep VMEM queue
            // full while QK MFMA executes.  V loads are issued AFTER the
            // K[t+1] prefetch so the compiler sees a steady stream of VMEM
            // ops; the IGLP MFMA barrier prevents the scheduler from
            // collapsing all loads to the front and forcing an early
            // vmcnt(0).
            load_k_tile(0);
            #pragma unroll
            for (int t = 0; t < kTLoop; t++)
            {
                if (t + 1 < kTLoop) {
                    load_k_tile(t + 1);
                }
                load_v_slice(t);

                d_out[t] = floatx4{0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int qkhe = 0; qkhe < kQkheLoop; qkhe++)
                {
                    d_out[t] = pa_mfma16x16x32_fp8_fp8(
                        Klocal[t][qkhe], Qlocal[qkhe], d_out[t]);
                }
                pa_apply_qk_token_scales_for_block(
                    d_out[t], k_scale_ptr, kphysical_block_number[t], rowid * 4,
                    gridDim.z, kv_head_idx, kBlockSize, qk_base_log2);

                __builtin_amdgcn_sched_group_barrier(/*MFMA*/0x008, 4, 0);
            }
        }

        // ----- Per-warp softmax max + exp_sum (mirrors v0) ----------------
        const int qkout_token_idx = partition_start_token_idx
                                    + kTokensPerWarp * warpid + rowid * 4;
        float qk_max  = -FLT_MAX;
        float exp_sum = 0.f;
        {
            const int valid_upper = context_len;
            const bool interior_partition =
                (partition_start_token_idx + kTParSize) <= valid_upper;

            // Boundary tiles: pre-mask out-of-range elements to -FLT_MAX so
            // the downstream max-reduce and exp loop can be a single unified
            // code path (saves ~16-20 static v_exp_f32 instructions vs the
            // earlier interior/boundary split — see gluon asm: ~9 vs ours 75).
            // exp2(-FLT_MAX - qk_max) underflows to 0, so masked positions
            // contribute zero to exp_sum and never become qk_max.
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

        // ----- Cross-warp merge → partition_qk_max / partition_exp_sum ----
        //
        // LDS layout repacked for b128 reads:
        //   shared_qk[lane16id * (kNWarps * 2) + warp * 2 + 0] = qk_max
        //   shared_qk[lane16id * (kNWarps * 2) + warp * 2 + 1] = exp_sum
        //
        // Per-lane read pattern then becomes a single contiguous 32-byte
        // span (= 2 ds_read_b128 = 8 dwords for kNWarps=4), which the
        // compiler can fuse into 2 ds_read_b128 instead of the 8 scattered
        // b32 / 2-offset ds_read2 it had to emit on the previous layout.
        if (laneid < 16)
        {
            const int slot = lane16id * (kNWarps * 2) + warpid * 2;
            shared_qk[slot + 0] = qk_max;
            shared_qk[slot + 1] = exp_sum;
        }
        __syncthreads();

        float partition_qk_max  = -FLT_MAX;
        float partition_exp_sum = 0.f;
        float warp_scale;      // = exp2(this_warp_qk_max - partition_qk_max)
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

        // Stage P to LDS: d_out[t] *= warp_scale (rescales each warp's
        // exp(qk - warp_max) → exp(qk - partition_max)).  NOTE: unlike v0,
        // we do NOT divide by partition_exp_sum here — the cross-partition
        // online softmax merge below handles the global normalisation, and
        // the final write divides by l_running once.
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

        // ----- PV MFMA (mirrors v0; produces unnormalised pv_acc[vhe]) ----
        // Optimisation: batch ALL P LDS reads up-front so the compiler can
        // issue them back-to-back (one lgkmcnt batch), then drain via
        // progressive lgkmcnt(N) during the MFMA chain (gluon-style).  This
        // converts the per-(v,t_pv) lgkmcnt(0) hard stalls into single
        // lgkmcnt(N) progressive waits.  VGPR cost = 8 P_packs × 8B / lane
        // = 16 VGPRs/lane, well within budget (60 VGPRs total).
        floatx4 pv_acc[kVheLoop];
        #pragma unroll
        for (int vhe = 0; vhe < kVheLoop; vhe++)
            pv_acc[vhe] = floatx4{0.f, 0.f, 0.f, 0.f};

        int64_t P_pack_all[kVtLoop][kPvTLoop];
        #pragma unroll
        for (int v = 0; v < kVtLoop; v++)
        {
            #pragma unroll
            for (int t_pv = 0; t_pv < kPvTLoop; t_pv++)
            {
                const int t_s   = t_pv * 2 + rowid / 2;
                const int rs_lo = (rowid & 1) * 2;
                const int rs_hi = rs_lo + 1;
                _T8x8 P_pack;
                P_pack.b8x4[0] = static_cast<uint32_t>(
                    shared_logits[v0::shared_logits_index(v, t_s, lane16id, rs_lo)].i64);
                P_pack.b8x4[1] = static_cast<uint32_t>(
                    shared_logits[v0::shared_logits_index(v, t_s, lane16id, rs_hi)].i64);
                P_pack_all[v][t_pv] = P_pack.i64;
            }
        }

        #pragma unroll
        for (int v = 0; v < kVtLoop; v++)
        {
            #pragma unroll
            for (int t_pv = 0; t_pv < kPvTLoop; t_pv++)
            {
                #pragma unroll
                for (int vhe = 0; vhe < kVheLoop; vhe++)
                {
                    pv_acc[vhe] = pa_mfma16x16x32_fp8_fp8(
                        Vlocal[v][t_pv][vhe], P_pack_all[v][t_pv], pv_acc[vhe]);
                }
            }
        }
        // pv_acc[vhe] now holds sum(exp(qk - partition_max) * V) (unscaled).

        // ----- ONLINE SOFTMAX MERGE ---------------------------------------
        // Per-lane: m_new = max(m_running, partition_qk_max); rescale o, l.
        // First iter (m_running=-inf): alpha=0, beta=1 → state initialised
        // from this iter's (partition_qk_max, partition_exp_sum, pv_acc).
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

    }  // === end outer loop ===

    // ----- Normalise o_running and apply v_scale --------------------------
    const float inv_l = __fdividef(1.f, l_running + 1e-6f);
    const float post_scale = inv_l * v_scale_perhead * p_scale_inv_perhead;
    #pragma unroll
    for (int vhe = 0; vhe < kVheLoop; vhe++)
        o_running[vhe] *= post_scale;

    // ----- Write max_logits / exp_sums (one (q_tok, head, fat_part) each)
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

    // ----- Convert o_running → output_t, stage to LDS, warp-0 gmem write --
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
