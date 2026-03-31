// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// This file is a design skeleton for the split-K MHA stage1 kernel.
// It is intentionally not wired into pybind/JIT yet. The goal is to capture
// the agreed launch contract and the intended opus-based compute structure.

#include "opus/opus.hpp"
#include "mha_prefill_splitk_metadata.h"

namespace aiter::mha_splitk_stage1 {

struct WorkInfoView {
    MhaPrefillSplitKWorkInfo wi;

    OPUS_H_D int batch_idx() const { return wi.batch_idx; }
    OPUS_H_D int qo_start() const { return wi.qo_start; }
    OPUS_H_D int qo_end() const { return wi.qo_end; }
    OPUS_H_D int kv_start() const { return wi.kv_start; }
    OPUS_H_D int kv_end() const { return wi.kv_end; }
    OPUS_H_D int kv_offset() const { return wi.kv_offset; }
    OPUS_H_D int partial_base() const { return wi.partial_qo_loc; }
    OPUS_H_D int qo_len() const { return wi.qo_end - wi.qo_start; }
    OPUS_H_D int kv_len() const { return wi.kv_end - wi.kv_start; }
};

template <typename T, int NumHead, int HeadDim>
struct ContiguousTHDView {
    const T* base;

    OPUS_H_D const T* ptr(int token_idx, int head_idx) const {
        return base + ((token_idx * NumHead + head_idx) * HeadDim);
    }
};

template <typename T, int HeadDim>
struct ContiguousKDView {
    const T* base;

    OPUS_H_D const T* ptr(int token_idx) const {
        return base + token_idx * HeadDim;
    }
};

template <
    typename QType,
    typename KVType,
    typename AccType,
    int HeadDim,
    int NumHeads,
    int NumKvHeads,
    int QTileSize,
    int NTile,
    bool IsCausal>
__global__ void mha_prefill_splitk_stage1_opus_kernel(
    const QType* q,                  // [sum_q, NumHeads, HeadDim]
    const KVType* k,                 // [sum_kv, NumKvHeads, HeadDim]
    const KVType* v,                 // [sum_kv, NumKvHeads, HeadDim]
    const int32_t* kv_indptr,        // reserved for future paged/cache path
    const int32_t* kv_page_indices,  // reserved for future paged/cache path
    int32_t page_size,               // reserved, fixed to 1 in future paged path
    const MhaPrefillSplitKWorkInfo* work_info_set,
    int32_t num_work,
    float softmax_scale,
    AccType* split_o,                // [partial_q_tokens, NumHeads, HeadDim]
    float* split_lse)                // [partial_q_tokens, NumHeads]
{
    using namespace opus;

    constexpr int kWaveSize = 64;
    constexpr int kNumWaves = 4;
    constexpr int kBlockSize = kWaveSize * kNumWaves;
    constexpr int kRowsPerWave = QTileSize / kNumWaves;
    constexpr float kNegInf = -3.402823466e+38F;

    static_assert(QTileSize == 128, "v1 skeleton assumes QTileSize=128");
    static_assert(kRowsPerWave == 32, "v1 skeleton assumes 4 waves over M=128");

    const int work_id = blockIdx.x;
    const int head_idx = blockIdx.y;

    if(work_id >= num_work || head_idx >= NumHeads)
    {
        return;
    }

    const int lane_id = threadIdx.x & (kWaveSize - 1);
    const int wave_id = threadIdx.x / kWaveSize;

    WorkInfoView work{work_info_set[work_id]};

    const int qo_start = work.qo_start();
    const int qo_len = work.qo_len();
    const int kv_start = work.kv_start();
    const int kv_end = work.kv_end();
    const int partial_base = work.partial_base();

    const int local_q_begin = wave_id * kRowsPerWave;
    const int local_q_end = local_q_begin + kRowsPerWave < qo_len
                                ? local_q_begin + kRowsPerWave
                                : qo_len;

    // Current agreed simplification:
    // - Q/K/V come from contiguous buffers.
    // - page-table style arguments stay in the signature but are not used yet.
    (void)kv_indptr;
    (void)kv_page_indices;
    (void)page_size;
    (void)IsCausal;

    ContiguousTHDView<QType, NumHeads, HeadDim> q_view{q};
    ContiguousTHDView<KVType, NumKvHeads, HeadDim> k_view{k};
    ContiguousTHDView<KVType, NumKvHeads, HeadDim> v_view{v};

    extern __shared__ char smem_raw[];
    KVType* smem_k_ptr = reinterpret_cast<KVType*>(smem_raw);
    KVType* smem_v_ptr = smem_k_ptr + NTile * HeadDim;

    auto s_k = make_smem(smem_k_ptr);
    auto s_v = make_smem(smem_v_ptr);
    auto smem_k_layout = make_layout(make_tuple(number<NTile>{}, number<HeadDim>{}));
    auto smem_v_layout = make_layout(make_tuple(number<NTile>{}, number<HeadDim>{}));

    // Q wave-local tile: [32, HeadDim]
    const QType* q_head_base = q_view.ptr(qo_start, head_idx);
    const QType* q_wave_base = q_head_base + local_q_begin * NumHeads * HeadDim;
    auto g_q = make_gmem(q_wave_base);
    auto q_vec_layout = make_layout(
        make_tuple(number<kRowsPerWave>{}, number<HeadDim / 4>{}),
        make_tuple(number<NumHeads * HeadDim>{}, 4_I));
    auto q_frag = g_q.load<4>(q_vec_layout);

    float row_max[kRowsPerWave];
    float row_sum[kRowsPerWave];
    float row_acc[kRowsPerWave][HeadDim];

    for(int r = 0; r < kRowsPerWave; ++r)
    {
        row_max[r] = kNegInf;
        row_sum[r] = 0.0f;
        for(int d = 0; d < HeadDim; ++d)
        {
            row_acc[r][d] = 0.0f;
        }
    }

    for(int n0 = kv_start; n0 < kv_end; n0 += NTile)
    {
        const int n_len = (n0 + NTile < kv_end) ? NTile : (kv_end - n0);

        // Stage K/V into LDS. Current version uses direct contiguous copies.
        for(int linear = threadIdx.x; linear < n_len * HeadDim; linear += kBlockSize)
        {
            const int d = linear % HeadDim;
            const int n_local = linear / HeadDim;
            const int global_kv = n0 + n_local;

            smem_k_ptr[linear] = *(k_view.ptr(global_kv, head_idx) + d);
            smem_v_ptr[linear] = *(v_view.ptr(global_kv, head_idx) + d);
        }
        __builtin_amdgcn_s_barrier();

        // ------------------------------------------------------------------
        // TODO(opus-compute):
        // Replace the scalar QK/PV reference loops below with opus MFMA.
        //
        // Target decomposition per wave:
        //   Q stripe : [32, 128]
        //   K tile   : [NTile, 128], with NTile expected to be 32 in v1
        //   Score    : [32, NTile]
        //
        // Recommended MFMA micro-kernel:
        //   mma_qk = make_mfma<bf16/fp16, bf16/fp16, float>(16_I, 16_I, 16_I)
        //   -> score_frag[2][2], q_frag_micro[2], k_frag_micro[2], k-loop = 8
        //
        // PV stage:
        //   P tile   : [32, NTile]
        //   V tile   : [NTile, 128]
        //   O tile   : [32, 128]
        //   -> o_frag[2][8], p_frag[2][2], v_frag[8][2]
        //
        // The intent is:
        //   1. QK uses K tile directly as B:[N,K]
        //   2. PV uses a logical transpose view of V as B:[D,K]
        // ------------------------------------------------------------------
        for(int q_local = local_q_begin; q_local < local_q_end; ++q_local)
        {
            const int row = q_local - local_q_begin;

            float scores[NTile];
            for(int ni = 0; ni < NTile; ++ni)
            {
                scores[ni] = kNegInf;
            }

            // Reference QK path: row vector x current K tile.
            for(int ni = 0; ni < n_len; ++ni)
            {
                float dot = 0.0f;
                for(int d = lane_id; d < HeadDim; d += kWaveSize)
                {
                    const float qv =
                        static_cast<float>(*(q_view.ptr(qo_start + q_local, head_idx) + d));
                    const float kv = static_cast<float>(smem_k_ptr[ni * HeadDim + d]);
                    dot += qv * kv;
                }

                for(int offset = 32; offset > 0; offset >>= 1)
                {
                    dot += shfl(dot, lane_id + offset, 64);
                }

                if(lane_id == 0)
                {
                    scores[ni] = dot * softmax_scale;
                }
            }

            if(lane_id == 0)
            {
                float old_max = row_max[row];
                float old_sum = row_sum[row];

                float tile_max = old_max;
                for(int ni = 0; ni < n_len; ++ni)
                {
                    tile_max = tile_max > scores[ni] ? tile_max : scores[ni];
                }

                const float old_scale = (old_sum == 0.0f) ? 0.0f : expf(old_max - tile_max);
                for(int d = 0; d < HeadDim; ++d)
                {
                    row_acc[row][d] *= old_scale;
                }

                float new_sum = old_sum * old_scale;
                for(int ni = 0; ni < n_len; ++ni)
                {
                    const float p = expf(scores[ni] - tile_max);
                    new_sum += p;

                    const int v_base = ni * HeadDim;
                    for(int d = 0; d < HeadDim; ++d)
                    {
                        row_acc[row][d] += p * static_cast<float>(smem_v_ptr[v_base + d]);
                    }
                }

                row_max[row] = tile_max;
                row_sum[row] = new_sum;
            }
        }

        __builtin_amdgcn_s_barrier();
    }

    if(lane_id == 0)
    {
        for(int q_local = local_q_begin; q_local < local_q_end; ++q_local)
        {
            const int row = q_local - local_q_begin;
            const int out_token = partial_base + q_local;
            const int lse_idx = out_token * NumHeads + head_idx;

            split_lse[lse_idx] = row_max[row] + logf(row_sum[row] > 1e-20f ? row_sum[row] : 1e-20f);

            const float inv_sum = 1.0f / (row_sum[row] > 1e-20f ? row_sum[row] : 1e-20f);
            const int out_base = (out_token * NumHeads + head_idx) * HeadDim;
            for(int d = 0; d < HeadDim; ++d)
            {
                split_o[out_base + d] = static_cast<AccType>(row_acc[row][d] * inv_sum);
            }
        }
    }
}

} // namespace aiter::mha_splitk_stage1
