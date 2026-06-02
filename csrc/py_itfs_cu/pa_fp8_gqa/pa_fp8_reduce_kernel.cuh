// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026 Page_Attetion_GQA_fp8 project
//
// Reduce kernel for FP8 paged-attention decode.  Identical algorithm to the
// pa_gqa v5 reduce (cross-partition log-sum-exp + weighted bf16 sum), only
// the wrapper namespace differs.  Output dtype is bf16/fp16.
//
// Grid : (num_heads, num_seqs, mtp)
// Block: 128

#pragma once

#include "pa_fp8_common.h"

namespace pa_fp8_gqa {

namespace v_red {
constexpr int kReduceHeadSize    = 128;
constexpr int kReduceNumThreads  = 128;
constexpr int kReduceNumWarps    = kReduceNumThreads / WARP_SIZE; // 2
constexpr int kReducePartSize    = 256;
constexpr int kReduceMaxPart     = 64;
constexpr int kReduceJChunk      = 16;
} // namespace v_red

template <typename scalar_t, int kNL, int kPartSize = v_red::kReducePartSize>
__global__ __launch_bounds__(v_red::kReduceNumThreads)
void pa_fp8_reduce_kernel(
    scalar_t* __restrict__ out,            // [num_seqs * mtp, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs * mtp, num_heads, max_num_partitions]
    const float* __restrict__ max_logits,  // same shape
    const scalar_t* __restrict__ tmp_out,  // [num_seqs * mtp, num_heads, max_num_partitions, head_size]
    const int* __restrict__ context_lens,
    const int max_num_partitions,
    const int fixed_num_partitions  // > 0 = fat mode (use directly); <= 0 = auto from ctx
)
{
    using namespace v_red;
    constexpr int kHead = kReduceHeadSize;
    constexpr int kPar  = kPartSize;
    constexpr int kMxP  = kReduceMaxPart;
    constexpr int kJ    = kReduceJChunk;

    const int num_heads = gridDim.x;
    const int head_idx  = blockIdx.x;
    const int seq_idx   = blockIdx.y;
    const int MTP       = gridDim.z;
    const int mtp       = blockIdx.z;

    const int context_len    = context_lens[seq_idx];
    const int num_partitions = (fixed_num_partitions > 0)
                                   ? fixed_num_partitions
                                   : PAGQA_DIVUP(context_len, kPar);

    const int warpid = threadIdx.x / WARP_SIZE;

    __shared__ float shared_global_exp_sum;
    __shared__ float shared_exp_sums[kNL * WARP_SIZE];

    if (warpid == 0)
    {
        const float* max_logits_ptr =
            max_logits + (seq_idx * MTP + mtp) * num_heads * max_num_partitions
            + head_idx * max_num_partitions;
        const float* exp_sums_ptr =
            exp_sums + (seq_idx * MTP + mtp) * num_heads * max_num_partitions
            + head_idx * max_num_partitions;

        const int last_valid_partition = num_partitions - 1;
        int   valid_partition[kNL];
        float reg_max_logit[kNL];

        #pragma unroll
        for (int i = 0; i < kNL; i++)
        {
            const int partition_no = i * WARP_SIZE + threadIdx.x;
            valid_partition[i] =
                (partition_no < num_partitions) ? partition_no : last_valid_partition;
        }
        #pragma unroll
        for (int i = 0; i < kNL; i++)
            reg_max_logit[i] = max_logits_ptr[valid_partition[i]];

        float max_logit = reg_max_logit[0];
        #pragma unroll
        for (int i = 1; i < kNL; i++)
            max_logit = fmaxf(max_logit, reg_max_logit[i]);
        max_logit = fmaxf(max_logit, pa_shfl_xor_32(max_logit));
        max_logit = fmaxf(max_logit, pa_shfl_xor_within_32<16>(max_logit));
        max_logit = fmaxf(max_logit, pa_shfl_xor_within_32< 8>(max_logit));
        max_logit = fmaxf(max_logit, pa_shfl_xor_within_32< 4>(max_logit));
        max_logit = fmaxf(max_logit, pa_shfl_xor_within_32< 2>(max_logit));
        max_logit = fmaxf(max_logit, pa_shfl_xor_within_32< 1>(max_logit));

        float rescaled_exp_sum[kNL];
        #pragma unroll
        for (int i = 0; i < kNL; i++)
            rescaled_exp_sum[i] = exp_sums_ptr[valid_partition[i]];
        #pragma unroll
        for (int i = 0; i < kNL; i++)
        {
            const int partition_no = i * WARP_SIZE + threadIdx.x;
            rescaled_exp_sum[i] *=
                (partition_no < num_partitions) ? __expf(reg_max_logit[i] - max_logit) : 0.f;
        }
        float global_exp_sum = rescaled_exp_sum[0];
        #pragma unroll
        for (int i = 1; i < kNL; i++)
            global_exp_sum += rescaled_exp_sum[i];
        #pragma unroll
        for (int i = 0; i < kNL; i++)
        {
            const int partition_no        = i * WARP_SIZE + threadIdx.x;
            shared_exp_sums[partition_no] = rescaled_exp_sum[i];
        }
        global_exp_sum = global_exp_sum + pa_shfl_xor_32(global_exp_sum);
        global_exp_sum = global_exp_sum + pa_shfl_xor_within_32<16>(global_exp_sum);
        global_exp_sum = global_exp_sum + pa_shfl_xor_within_32< 8>(global_exp_sum);
        global_exp_sum = global_exp_sum + pa_shfl_xor_within_32< 4>(global_exp_sum);
        global_exp_sum = global_exp_sum + pa_shfl_xor_within_32< 2>(global_exp_sum);
        global_exp_sum = global_exp_sum + pa_shfl_xor_within_32< 1>(global_exp_sum);
        if (threadIdx.x == 0)
            shared_global_exp_sum = global_exp_sum;
    }

    const scalar_t* tmp_out_ptr =
        tmp_out + (seq_idx * MTP + mtp) * num_heads * max_num_partitions * kHead
        + head_idx * max_num_partitions * kHead + threadIdx.x;

    scalar_t tmps[kMxP];
    #pragma unroll
    for (int j = 0; j < kMxP; j++)
        tmps[j] = pa_from_float<scalar_t>(0.f);

    const int last_partition_offset = (num_partitions - 1) * kHead;
    const int num_partition_offset  = num_partitions * kHead;
    int idx = 0;

    #pragma unroll
    for (int j = 0; j < kJ * kHead; j += kHead)
    {
        const int lastj_offset = (j < num_partition_offset) ? j : last_partition_offset;
        tmps[idx++] = tmp_out_ptr[lastj_offset];
    }
    __syncthreads();

    if (num_partitions > kJ)
    {
        #pragma unroll
        for (int j = kJ * kHead; j < 2 * kJ * kHead; j += kHead)
        {
            const int lastj_offset = (j < num_partition_offset) ? j : last_partition_offset;
            tmps[idx++] = tmp_out_ptr[lastj_offset];
        }
        if (num_partitions > 2 * kJ)
        {
            #pragma unroll
            for (int j = 2 * kJ * kHead; j < kMxP * kHead; j += kHead)
            {
                const int lastj_offset = (j < num_partition_offset) ? j : last_partition_offset;
                tmps[idx++] = tmp_out_ptr[lastj_offset];
            }
        }
    }

    static_assert(kMxP % 2 == 0, "kMxP must be even for packed FMA pairs");
    static_assert(kJ   % 2 == 0, "kJ must be even for packed FMA pairs");

    floatx2 acc2{0.f, 0.f};

    auto fma_pairs = [&](int base_j, int base_s, int count) __attribute__((always_inline)) {
        const _B16x2* tmp_pairs = reinterpret_cast<const _B16x2*>(&tmps[base_j]);
        const float*  spt       = &shared_exp_sums[base_s];
        #pragma unroll
        for (int p2 = 0; p2 < (count / 2); p2++)
        {
            const floatx2 fp = pa_bf16x2_to_floatx2(tmp_pairs[p2]);
            const floatx2 sp{spt[2*p2], spt[2*p2 + 1]};
            acc2 += fp * sp;
        }
    };

    fma_pairs(0, 0, kJ);
    if (num_partitions > kJ)
    {
        fma_pairs(kJ, kJ, kJ);
        if (num_partitions > 2 * kJ)
            fma_pairs(2 * kJ, 2 * kJ, kMxP - 2 * kJ);
    }

    #pragma unroll
    for (int p = 1; p < kNL; p++)
    {
        if (num_partitions > p * kMxP)
        {
            idx = 0;
            #pragma unroll
            for (int j = p * kMxP * kHead; j < (p + 1) * kMxP * kHead; j += kHead)
            {
                const int lastj_offset = (j < num_partition_offset) ? j : last_partition_offset;
                tmps[idx++] = tmp_out_ptr[lastj_offset];
            }
            fma_pairs(0, p * kMxP, kMxP);
        }
    }

    const float acc = acc2[0] + acc2[1];

    const float inv_global_exp_sum = __fdividef(1.f, shared_global_exp_sum + 1e-6f);
    const float scaled_acc         = acc * inv_global_exp_sum;

    scalar_t* out_ptr =
        out + static_cast<int64_t>(seq_idx * MTP + mtp) * num_heads * kHead
        + static_cast<int64_t>(head_idx) * kHead;
    out_ptr[threadIdx.x] = pa_from_float<scalar_t>(scaled_acc);
}

// ---------------------------------------------------------------------------
// pa_fp8_reduce_kernel_v2 — gluon-style HIP reduce.
//
// Replaces the v1 kernel for small/medium num_partitions.  Differences:
//   * Block = HEAD_SIZE (= 128) threads — all threads participate in the FMA
//     loop, one head_dim element per thread.  v1 used 128 threads but only
//     warp 0 ran the FMA core; here all 2 warps FMA in lockstep.
//   * Warp 0 alone does the max/sum reduce; weights are staged into LDS
//     (NUM_CHUNKS * WARP_SIZE = ≤ 512 floats = ≤ 2 KiB LDS) for the FMA phase.
//   * Tight inner loop (`for p in 0..N: fma(load, weight)`) compiles to a
//     dense schedule of buffer_load + v_fmac without the multi-stage register
//     caching v1 uses.  Substantially fewer cycles for small N and avoids
//     the LDS-only weight reads that limited v1's IPC.
//
// Grid : (num_seqs, num_heads, mtp)   ← identical to v1
// Block: 128 threads (1 wave + 1 wave; second wave only joins the FMA phase)
// LDS  : NumChunks * 64 floats (256–2 048 B)
//
// Supports up to NumChunks * WARP_SIZE = 64, 128, 256 or 512 partitions per
// reduce call (NumChunks ∈ {1, 2, 4, 8}).
// ---------------------------------------------------------------------------
namespace v_red2 {
constexpr int kHeadSize    = 128;
constexpr int kNumThreads  = 128;
constexpr int kPartSize    = 256;
constexpr int kWarp        = WARP_SIZE;
} // namespace v_red2

template <typename scalar_t, int NumChunks,
          int kPartSize = v_red2::kPartSize>
__global__ __launch_bounds__(v_red2::kNumThreads)
void pa_fp8_reduce_kernel_v2(
    scalar_t* __restrict__ out,            // [num_seqs * mtp, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs * mtp, num_heads, max_num_partitions]
    const float* __restrict__ max_logits,  // same shape
    const scalar_t* __restrict__ tmp_out,  // [num_seqs * mtp, num_heads, max_num_partitions, head_size]
    const int* __restrict__ context_lens,
    const int max_num_partitions,
    const int fixed_num_partitions
)
{
    using namespace v_red2;
    constexpr int kHead       = kHeadSize;
    constexpr int kPar        = kPartSize;
    constexpr int kMaxPart    = NumChunks * kWarp;
    constexpr float kLog2E    = 1.4426950408889634f;
    constexpr float kInvLog2E = 0.6931471805599453f;

    const int num_heads = gridDim.x;
    const int MTP       = gridDim.z;
    const int head_idx  = blockIdx.x;
    const int seq_idx   = blockIdx.y;
    const int mtp       = blockIdx.z;
    const int tid       = threadIdx.x;
    const int warp_id   = tid / kWarp;
    const int lane      = tid % kWarp;

    const int context_len    = context_lens[seq_idx];
    const int num_partitions = (fixed_num_partitions > 0)
                                   ? fixed_num_partitions
                                   : PAGQA_DIVUP(context_len, kPar);
    const int ns_mtp_idx     = seq_idx * MTP + mtp;

    __shared__ float shared_weights[kMaxPart > 0 ? kMaxPart : 1];

    // ============ Warp 0: load + max/sum + write weights to LDS ============
    if (warp_id == 0)
    {
        const int64_t base = static_cast<int64_t>(ns_mtp_idx) * num_heads
                                * max_num_partitions
                           + static_cast<int64_t>(head_idx) * max_num_partitions;
        const float* ml_ptr = max_logits + base;
        const float* es_ptr = exp_sums   + base;

        float ml_local[NumChunks];
        float es_local[NumChunks];
        float ml_max = -FLT_MAX;
        #pragma unroll
        for (int c = 0; c < NumChunks; c++)
        {
            const int p = lane + c * kWarp;
            const bool valid = (p < num_partitions);
            ml_local[c] = valid ? ml_ptr[p] : -FLT_MAX;
            es_local[c] = valid ? es_ptr[p] : 0.f;
            ml_max = fmaxf(ml_max, ml_local[c]);
        }
        // 64-lane max
        ml_max = fmaxf(ml_max, pa_shfl_xor_within_32<16>(ml_max));
        ml_max = fmaxf(ml_max, pa_shfl_xor_within_32< 8>(ml_max));
        ml_max = fmaxf(ml_max, pa_shfl_xor_within_32< 4>(ml_max));
        ml_max = fmaxf(ml_max, pa_shfl_xor_within_32< 2>(ml_max));
        ml_max = fmaxf(ml_max, pa_shfl_xor_within_32< 1>(ml_max));
        ml_max = fmaxf(ml_max, pa_shfl_xor_32(ml_max));

        float sum_local = 0.f;
        #pragma unroll
        for (int c = 0; c < NumChunks; c++)
        {
            const float scale = (ml_local[c] > -FLT_MAX)
                ? __builtin_amdgcn_exp2f((ml_local[c] - ml_max) * kLog2E)
                : 0.f;
            es_local[c] *= scale;
            sum_local   += es_local[c];
        }
        sum_local += pa_shfl_xor_within_32<16>(sum_local);
        sum_local += pa_shfl_xor_within_32< 8>(sum_local);
        sum_local += pa_shfl_xor_within_32< 4>(sum_local);
        sum_local += pa_shfl_xor_within_32< 2>(sum_local);
        sum_local += pa_shfl_xor_within_32< 1>(sum_local);
        sum_local += pa_shfl_xor_32(sum_local);

        const float inv_sum = __fdividef(1.f, sum_local + 1e-6f);
        #pragma unroll
        for (int c = 0; c < NumChunks; c++)
        {
            const int p = lane + c * kWarp;
            const float w = es_local[c] * inv_sum;
            // Only write the lanes that map to valid partitions; the rest of
            // the LDS slots are never read by the FMA phase below.
            if (p < num_partitions) shared_weights[p] = w;
        }
    }
    __syncthreads();

    // ============ All threads: FMA loop (one head_dim element per thread) ============
    const int64_t logits_base =
          static_cast<int64_t>(ns_mtp_idx) * num_heads * max_num_partitions * kHead
        + static_cast<int64_t>(head_idx)   * max_num_partitions * kHead;
    const scalar_t* logits_ptr = tmp_out + logits_base + tid;

    float acc = 0.f;
    // Outer loop over chunks, inner loop unrolled — keeps the load/fma
    // pipeline saturated even for max-chunk (512-partition) cases.
    int partitions_remaining = num_partitions;
    #pragma unroll
    for (int c = 0; c < NumChunks; c++)
    {
        const int chunk_end = (partitions_remaining < kWarp) ? partitions_remaining : kWarp;
        #pragma unroll
        for (int p = 0; p < kWarp; p++)
        {
            if (p >= chunk_end) break;
            const int p_idx = c * kWarp + p;
            const float v = pa_to_float<scalar_t>(logits_ptr[p_idx * kHead]);
            acc = fmaf(v, shared_weights[p_idx], acc);
        }
        partitions_remaining -= kWarp;
        if (partitions_remaining <= 0) break;
    }

    scalar_t* out_ptr = out
        + static_cast<int64_t>(ns_mtp_idx) * num_heads * kHead
        + static_cast<int64_t>(head_idx) * kHead;
    out_ptr[tid] = pa_from_float<scalar_t>(acc);
}

// ---------------------------------------------------------------------------
// pa_fp8_reduce_kernel_v3 — gluon-style PS-reduce (ds_bpermute weight bcast).
//
// Tier-1 fast path: only used when num_partitions <= 64.  Beats v1/v2 for
// small partition counts by:
//   * 0 LDS, 0 __syncthreads.
//   * Single-instruction `ds_bpermute` weight broadcast — no LDS round-trip.
//   * Tight, fully-unrolled inner loop over CONTEXT_PARTITION_NUM partitions.
//   * One CTA per (seq*mtp, kv_head, head) — minimal launch parameters.
//
// Per-CTA work:
//   * lane ∈ [0, NumPart): reads (exp_sum, max_logit) for its partition.
//   * Wave-reduce max & sum across first 64 lanes (within warp 0; lanes
//     [NumPart..64) hold neutral values).
//   * Per-lane normalised weight w[lane] = part_sum[lane] / global_exp_sum.
//   * Then for each part_idx in 0..NumPart, broadcast weight from lane=
//     part_idx via ds_bpermute, load tmp_out[part_idx, head_dim=tid], FMA.
//
// Both warps of the CTA run the partition-reduce code in parallel (they
// produce identical weights in their lanes), then they FMA over disjoint
// head_dim halves [tid=0..63 and tid=64..127] of the output.  This mirrors
// gluon's `pa_decode_ps_reduce_hip_kernel` and avoids any cross-warp sync.
//
// Grid: (num_seqs, num_kv_heads, mtp * num_q_heads)
// Block: HEAD_SIZE = 128 threads (2 warps on gfx942)
// ---------------------------------------------------------------------------
template <typename scalar_t, int NumPart, int kPartSize = v_red2::kPartSize>
__global__ __launch_bounds__(v_red2::kHeadSize)
void pa_fp8_reduce_kernel_v3(
    scalar_t* __restrict__ out,            // [num_seqs * mtp, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs * mtp, num_heads, max_num_partitions]
    const float* __restrict__ max_logits,  // same shape
    const scalar_t* __restrict__ tmp_out,  // [num_seqs * mtp, num_heads, max_num_partitions, head_size]
    const int* __restrict__ context_lens,
    const int max_num_partitions,
    const int fixed_num_partitions
)
{
    using namespace v_red2;
    constexpr int kHead       = kHeadSize;
    constexpr int kPar        = kPartSize;
    constexpr float kLog2E    = 1.4426950408889634f;
    static_assert(NumPart > 0 && NumPart <= 64,
                  "v3 reduce supports 1..64 partitions per call");

    const int num_heads = gridDim.x;
    const int MTP       = gridDim.z;
    const int head_idx  = blockIdx.x;
    const int seq_idx   = blockIdx.y;
    const int mtp       = blockIdx.z;
    const int tid       = threadIdx.x;
    const int lane      = tid & (kWarp - 1);

    const int context_len    = context_lens[seq_idx];
    const int num_partitions = (fixed_num_partitions > 0)
                                   ? fixed_num_partitions
                                   : PAGQA_DIVUP(context_len, kPar);
    const int ns_mtp_idx     = seq_idx * MTP + mtp;

    const int64_t base =
          static_cast<int64_t>(ns_mtp_idx) * num_heads * max_num_partitions
        + static_cast<int64_t>(head_idx) * max_num_partitions;

    // Load partition's (max, sum) for lane < num_partitions; -inf/0 otherwise.
    // Note: NumPart is the compile-time *bound* (= reduce_width); the actual
    // num_partitions may be smaller (e.g. NumPart=8 with num_partitions=5).
    const bool lane_in_range = (lane < NumPart) && (lane < num_partitions);
    float part_max = -FLT_MAX;
    float part_sum = 0.f;
    if (lane_in_range)
    {
        part_max = max_logits[base + lane];
        part_sum = exp_sums  [base + lane];
    }

    // Wave-reduce max (within reduce_width = NumPart, rounded up to pow2).
    constexpr int kRW = (NumPart <= 1)  ? 1
                       : (NumPart <= 2)  ? 2
                       : (NumPart <= 4)  ? 4
                       : (NumPart <= 8)  ? 8
                       : (NumPart <= 16) ? 16
                       : (NumPart <= 32) ? 32 : 64;

    float global_max = part_max;
    if constexpr (kRW > 32) global_max = fmaxf(global_max, pa_shfl_xor_32(global_max));
    if constexpr (kRW > 16) global_max = fmaxf(global_max, pa_shfl_xor_within_32<16>(global_max));
    if constexpr (kRW >  8) global_max = fmaxf(global_max, pa_shfl_xor_within_32< 8>(global_max));
    if constexpr (kRW >  4) global_max = fmaxf(global_max, pa_shfl_xor_within_32< 4>(global_max));
    if constexpr (kRW >  2) global_max = fmaxf(global_max, pa_shfl_xor_within_32< 2>(global_max));
    if constexpr (kRW >  1) global_max = fmaxf(global_max, pa_shfl_xor_within_32< 1>(global_max));

    const bool valid_part = lane_in_range && (part_max > -FLT_MAX);
    const float safe_max  = (global_max > -FLT_MAX) ? global_max : 0.f;
    const float part_scale =
        valid_part ? __builtin_amdgcn_exp2f((part_max - safe_max) * kLog2E) : 0.f;
    const float scaled_sum = part_sum * part_scale;

    float global_exp_sum = scaled_sum;
    if constexpr (kRW > 32) global_exp_sum = global_exp_sum + pa_shfl_xor_32(global_exp_sum);
    if constexpr (kRW > 16) global_exp_sum = global_exp_sum + pa_shfl_xor_within_32<16>(global_exp_sum);
    if constexpr (kRW >  8) global_exp_sum = global_exp_sum + pa_shfl_xor_within_32< 8>(global_exp_sum);
    if constexpr (kRW >  4) global_exp_sum = global_exp_sum + pa_shfl_xor_within_32< 4>(global_exp_sum);
    if constexpr (kRW >  2) global_exp_sum = global_exp_sum + pa_shfl_xor_within_32< 2>(global_exp_sum);
    if constexpr (kRW >  1) global_exp_sum = global_exp_sum + pa_shfl_xor_within_32< 1>(global_exp_sum);

    const float inv_sum = __fdividef(1.f, global_exp_sum + 1e-6f);
    const float weight_local = scaled_sum * inv_sum;

    // FMA loop — two-stage software pipeline.
    //
    // The naive `for p in 0..NumPart: load; fma` loop, even fully unrolled,
    // is bottlenecked at NumPart=64 by the s_waitcnt inserted before each
    // FMA: with only ~10 VGPRs of live state the compiler can keep at most
    // 2-3 loads outstanding, serialising the global_load latency (~400 cy
    // each).  Splitting into PREFETCH of `kBatch` loads followed by their
    // FMAs lets all `kBatch` loads be in flight at once before any FMA
    // s_waitcnts.
    //
    // Bigger kBatch → more outstanding loads → better latency hiding, at
    // the cost of more VGPRs.  At NumPart=64, kBatch=16 keeps the loop
    // body at 43 VGPRs (8 waves/SIMD on gfx942) while saturating the
    // global_load throughput.
    constexpr int kBatch = (NumPart <= 4)  ? NumPart
                          : (NumPart <= 8) ? NumPart
                          : 16;
    static_assert(NumPart % kBatch == 0, "NumPart must be a multiple of kBatch");

    const int64_t logits_base =
          static_cast<int64_t>(ns_mtp_idx) * num_heads * max_num_partitions * kHead
        + static_cast<int64_t>(head_idx)   * max_num_partitions * kHead
        + tid;
    const scalar_t* logits_ptr = tmp_out + logits_base;

    float acc = 0.f;
    #pragma unroll
    for (int p0 = 0; p0 < NumPart; p0 += kBatch)
    {
        float v_batch[kBatch];
        #pragma unroll
        for (int k = 0; k < kBatch; k++)
            v_batch[k] = pa_to_float<scalar_t>(logits_ptr[(p0 + k) * kHead]);
        #pragma unroll
        for (int k = 0; k < kBatch; k++)
        {
            const float w = pa_lane_bcast(weight_local, p0 + k);
            if ((p0 + k) < num_partitions)
                acc = fmaf(v_batch[k], w, acc);
        }
    }

    scalar_t* out_ptr = out
        + static_cast<int64_t>(ns_mtp_idx) * num_heads * kHead
        + static_cast<int64_t>(head_idx) * kHead;
    out_ptr[tid] = pa_from_float<scalar_t>(acc);
}

} // namespace pa_fp8_gqa
