// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// This file is a design skeleton for the split-K MHA stage1 kernel.
// It is intentionally not wired into pybind/JIT yet. The goal is to capture
// the agreed launch contract and the intended opus-based compute structure.

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/extension.h>

#include "opus/opus.hpp"
#include "mha_prefill_splitk_metadata.h"

namespace aiter::mha_splitk_stage1 {

#ifndef AITER_SPLITK_STAGE1_USE_MFMA_QK
#define AITER_SPLITK_STAGE1_USE_MFMA_QK 1
#endif

#ifndef AITER_SPLITK_STAGE1_DEBUG_SCORE_DUMP
#define AITER_SPLITK_STAGE1_DEBUG_SCORE_DUMP 0
#endif

#ifndef AITER_SPLITK_STAGE1_USE_MFMA_PV
#define AITER_SPLITK_STAGE1_USE_MFMA_PV 1
#endif

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

template <typename ScoreMem, typename ScoreFrag, typename LayoutC>
OPUS_D void scatter_score_frag_to_smem(
    ScoreMem& score_mem,
    const ScoreFrag& score_frag,
    const LayoutC& layout_c)
{
    // `layout_c` is the lane-aware opus mapping from MFMA accumulator lanes
    // to the logical [32, 32] score tile in shared memory.
    score_mem.template store<4>(score_frag, layout_c);
}

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
    float* split_lse,                // [partial_q_tokens, NumHeads]
    float* debug_qk_scores = nullptr) // optional [num_work, NumHeads, 4, 32, 32]
{
    using namespace opus;
    using opus::operator""_I;

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
    QType* smem_p_ptr = reinterpret_cast<QType*>(smem_k_ptr);

    auto s_k = make_smem(smem_k_ptr);
    auto s_v = make_smem(smem_v_ptr);
    auto smem_k_layout = make_layout(opus::make_tuple(number<NTile>{}, number<HeadDim>{}));
    auto smem_v_layout = make_layout(opus::make_tuple(number<NTile>{}, number<HeadDim>{}));

    // Q wave-local tile: [32, HeadDim]
    const QType* q_head_base = q_view.ptr(qo_start, head_idx);
    const QType* q_wave_base = q_head_base + local_q_begin * NumHeads * HeadDim;
    auto g_q = make_gmem(q_wave_base);

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
        // QK compute path
        // ------------------------------------------------------------------
        float* smem_scores =
            reinterpret_cast<float*>(smem_v_ptr + NTile * HeadDim);
        float* wave_score_base = smem_scores + wave_id * (32 * 32);

#if AITER_SPLITK_STAGE1_USE_MFMA_QK
        static_assert(NTile == 32, "MFMA QK draft currently assumes NTile=32");

        constexpr int E_M = 1;
        constexpr int E_N = 1;
        constexpr int E_K = 1;
        constexpr int T_M = 1;
        constexpr int T_N = 1;
        constexpr int T_K = 1;
        constexpr int ELEM_A = 32 * 8 / 64;
        constexpr int ELEM_B = 32 * 8 / 64;

        auto mma_qk = make_tiled_mma<QType, KVType, float>(
            seq<E_M, E_N, E_K>{},
            seq<T_M, T_N, T_K>{},
            seq<32, 32, 8>{},
            mfma_adaptor_swap_ab{});

        auto u_a = partition_layout_a<ELEM_A>(
            mma_qk,
            opus::make_tuple(number<NumHeads * HeadDim>{}, 1_I),
            opus::make_tuple(0_I, lane_id % mma_qk.grpm_a, 0_I, lane_id / mma_qk.grpm_a));
        auto u_b = partition_layout_b<ELEM_B>(
            mma_qk,
            opus::make_tuple(number<HeadDim>{}, 1_I),
            opus::make_tuple(0_I, lane_id % mma_qk.grpn_b, 0_I, lane_id / mma_qk.grpn_b));
        auto u_c = partition_layout_c(
            mma_qk,
            opus::make_tuple(32_I, 1_I),
            opus::make_tuple(0_I, lane_id % mma_qk.grpn_c, 0_I, lane_id / mma_qk.grpn_c));

        auto s_scores = make_smem(wave_score_base);
        typename decltype(mma_qk)::vtype_c score_frag;
        clear(score_frag);

        auto g_q_sub = make_gmem(q_wave_base);
        auto s_k_sub = make_smem(smem_k_ptr);

        for(int k0 = 0; k0 < HeadDim; k0 += 8)
        {
            auto q_frag_mfma = g_q_sub.template load<ELEM_A>(u_a);
            auto k_frag_mfma = s_k_sub.template load<ELEM_B>(u_b);
            score_frag = mma_qk(q_frag_mfma, k_frag_mfma, score_frag);
            u_a += 8;
            u_b += 8;
        }

        scatter_score_frag_to_smem(s_scores, score_frag, u_c);
        __builtin_amdgcn_s_barrier();

#if AITER_SPLITK_STAGE1_DEBUG_SCORE_DUMP
        if(debug_qk_scores != nullptr)
        {
            if(lane_id == 0)
            {
                float* dbg_base =
                    debug_qk_scores +
                    (((work_id * NumHeads + head_idx) * kNumWaves + wave_id) * 32 * 32);
                for(int i = 0; i < 32 * 32; ++i)
                {
                    dbg_base[i] = wave_score_base[i];
                }
            }
            __builtin_amdgcn_s_barrier();
        }
#endif

        for(int q_local = local_q_begin; q_local < local_q_end; ++q_local)
        {
            const int row = q_local - local_q_begin;
            const int row_in_wave = q_local - local_q_begin;

            if(lane_id == 0)
            {
                float old_max = row_max[row];
                float old_sum = row_sum[row];

                float tile_max = old_max;
                for(int ni = 0; ni < n_len; ++ni)
                {
                    const float s = wave_score_base[row_in_wave * 32 + ni] * softmax_scale;
                    tile_max = tile_max > s ? tile_max : s;
                }

                const float old_scale = (old_sum == 0.0f) ? 0.0f : expf(old_max - tile_max);
                for(int d = 0; d < HeadDim; ++d)
                {
                    row_acc[row][d] *= old_scale;
                }

                float new_sum = old_sum * old_scale;
                for(int ni = 0; ni < n_len; ++ni)
                {
                    const float s = wave_score_base[row_in_wave * 32 + ni] * softmax_scale;
                    const float p = expf(s - tile_max);
                    new_sum += p;
                    smem_p_ptr[row_in_wave * NTile + ni] = static_cast<QType>(p);
                }
                for(int ni = n_len; ni < NTile; ++ni)
                {
                    smem_p_ptr[row_in_wave * NTile + ni] = static_cast<QType>(0);
                }

                row_max[row] = tile_max;
                row_sum[row] = new_sum;
            }
        }
        if(lane_id == 0)
        {
            for(int row = local_q_end - local_q_begin; row < kRowsPerWave; ++row)
            {
                for(int ni = 0; ni < NTile; ++ni)
                {
                    smem_p_ptr[row * NTile + ni] = static_cast<QType>(0);
                }
            }
        }
        __builtin_amdgcn_s_barrier();

#if AITER_SPLITK_STAGE1_USE_MFMA_PV
        static_assert(NTile == 32, "MFMA PV draft currently assumes NTile=32");

        constexpr int PV_E_M = 1;
        constexpr int PV_E_N = 1;
        constexpr int PV_E_K = 1;
        constexpr int PV_T_M = 1;
        constexpr int PV_T_N = 1;
        constexpr int PV_T_K = 1;
        constexpr int PV_ELEM_A = 32 * 8 / 64;
        constexpr int PV_ELEM_B = 32 * 8 / 64;

        auto mma_pv = make_tiled_mma<QType, KVType, float>(
            seq<PV_E_M, PV_E_N, PV_E_K>{},
            seq<PV_T_M, PV_T_N, PV_T_K>{},
            seq<32, 32, 8>{},
            mfma_adaptor_swap_ab{});

        auto s_p = make_smem(smem_p_ptr);
        auto s_v_sub = make_smem(smem_v_ptr);
        auto s_o = make_smem(wave_score_base);

        for(int d0 = 0; d0 < HeadDim; d0 += 32)
        {
            auto u_p = partition_layout_a<PV_ELEM_A>(
                mma_pv,
                opus::make_tuple(32_I, 1_I),
                opus::make_tuple(0_I, lane_id % mma_pv.grpm_a, 0_I, lane_id / mma_pv.grpm_a));
            auto u_v = partition_layout_b<PV_ELEM_B>(
                mma_pv,
                opus::make_tuple(1_I, number<HeadDim>{}),
                opus::make_tuple(0_I, lane_id % mma_pv.grpn_b, 0_I, lane_id / mma_pv.grpn_b));
            auto u_o = partition_layout_c(
                mma_pv,
                opus::make_tuple(32_I, 1_I),
                opus::make_tuple(0_I, lane_id % mma_pv.grpn_c, 0_I, lane_id / mma_pv.grpn_c));

            typename decltype(mma_pv)::vtype_c o_frag;
            clear(o_frag);

            auto s_v_d = make_smem(smem_v_ptr + d0);
            for(int k0 = 0; k0 < NTile; k0 += 8)
            {
                auto p_frag_mfma = s_p.template load<PV_ELEM_A>(u_p);
                auto v_frag_mfma = s_v_d.template load<PV_ELEM_B>(u_v);
                o_frag = mma_pv(p_frag_mfma, v_frag_mfma, o_frag);
                u_p += 8;
                u_v += 8;
            }

            scatter_score_frag_to_smem(s_o, o_frag, u_o);
            __builtin_amdgcn_s_barrier();

            if(lane_id == 0)
            {
                for(int q_local = local_q_begin; q_local < local_q_end; ++q_local)
                {
                    const int row = q_local - local_q_begin;
                    const int row_in_wave = q_local - local_q_begin;
                    for(int di = 0; di < 32; ++di)
                    {
                        row_acc[row][d0 + di] += wave_score_base[row_in_wave * 32 + di];
                    }
                }
            }
            __builtin_amdgcn_s_barrier();
        }
#else
        for(int q_local = local_q_begin; q_local < local_q_end; ++q_local)
        {
            const int row = q_local - local_q_begin;
            const int row_in_wave = q_local - local_q_begin;
            if(lane_id == 0)
            {
                for(int ni = 0; ni < n_len; ++ni)
                {
                    const float p = static_cast<float>(smem_p_ptr[row_in_wave * NTile + ni]);
                    const int v_base = ni * HeadDim;
                    for(int d = 0; d < HeadDim; ++d)
                    {
                        row_acc[row][d] += p * static_cast<float>(smem_v_ptr[v_base + d]);
                    }
                }
            }
        }
#endif
#else
        // Scalar fallback path kept for bring-up and comparison.
        for(int q_local = local_q_begin; q_local < local_q_end; ++q_local)
        {
            const int row = q_local - local_q_begin;

            float scores[NTile];
            for(int ni = 0; ni < NTile; ++ni)
            {
                scores[ni] = kNegInf;
            }

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
#endif

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

void mha_prefill_splitk_stage1_opus(
    const torch::Tensor& q,                 // [sum_q, num_heads, head_dim]
    const torch::Tensor& k,                 // [sum_kv, num_kv_heads, head_dim]
    const torch::Tensor& v,                 // [sum_kv, num_kv_heads, head_dim]
    const torch::Tensor& kv_indptr,         // reserved for future paged path
    const torch::Tensor& kv_page_indices,   // reserved for future paged path
    const int32_t page_size,
    const torch::Tensor& work_indptr,       // [2], simplified [0, num_work]
    const torch::Tensor& work_info_set,     // [max_work, 8]
    const float softmax_scale,
    torch::Tensor& split_o,                 // [partial_tokens, num_heads, head_dim]
    torch::Tensor& split_lse,               // [partial_tokens, num_heads]
    std::optional<torch::Tensor> debug_qk_scores = std::nullopt)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(q));

    TORCH_CHECK(q.is_cuda(), __func__, ": q must be CUDA tensor");
    TORCH_CHECK(k.is_cuda(), __func__, ": k must be CUDA tensor");
    TORCH_CHECK(v.is_cuda(), __func__, ": v must be CUDA tensor");
    TORCH_CHECK(work_indptr.is_cuda(), __func__, ": work_indptr must be CUDA tensor");
    TORCH_CHECK(work_info_set.is_cuda(), __func__, ": work_info_set must be CUDA tensor");
    TORCH_CHECK(split_o.is_cuda(), __func__, ": split_o must be CUDA tensor");
    TORCH_CHECK(split_lse.is_cuda(), __func__, ": split_lse must be CUDA tensor");

    TORCH_CHECK(q.dim() == 3, __func__, ": q must be [sum_q, num_heads, head_dim]");
    TORCH_CHECK(k.dim() == 3, __func__, ": k must be [sum_kv, num_kv_heads, head_dim]");
    TORCH_CHECK(v.dim() == 3, __func__, ": v must be [sum_kv, num_kv_heads, head_dim]");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16,
                __func__,
                ": v1 wrapper only supports bf16 q");
    TORCH_CHECK(k.scalar_type() == at::ScalarType::BFloat16,
                __func__,
                ": v1 wrapper only supports bf16 k");
    TORCH_CHECK(v.scalar_type() == at::ScalarType::BFloat16,
                __func__,
                ": v1 wrapper only supports bf16 v");
    TORCH_CHECK(split_o.scalar_type() == at::ScalarType::Float,
                __func__,
                ": split_o must be fp32 in v1");
    TORCH_CHECK(split_lse.scalar_type() == at::ScalarType::Float,
                __func__,
                ": split_lse must be fp32 in v1");

    TORCH_CHECK(q.size(1) == 1, __func__, ": v1 wrapper currently fixes num_heads=1");
    TORCH_CHECK(k.size(1) == 1, __func__, ": v1 wrapper currently fixes num_kv_heads=1");
    TORCH_CHECK(v.size(1) == 1, __func__, ": v1 wrapper currently fixes num_kv_heads=1");
    TORCH_CHECK(q.size(2) == 128, __func__, ": v1 wrapper currently fixes head_dim=128");
    TORCH_CHECK(k.size(2) == 128, __func__, ": v1 wrapper currently fixes head_dim=128");
    TORCH_CHECK(v.size(2) == 128, __func__, ": v1 wrapper currently fixes head_dim=128");
    TORCH_CHECK(work_info_set.size(-1) == 8, __func__, ": work_info_set must be [..., 8]");
    TORCH_CHECK(work_indptr.numel() >= 2, __func__, ": work_indptr must have at least 2 elements");

    auto work_indptr_cpu = work_indptr.to(torch::kCPU);
    const int32_t num_work = work_indptr_cpu.data_ptr<int32_t>()[1];
    TORCH_CHECK(num_work >= 0, __func__, ": invalid num_work");

    dim3 block(256);
    dim3 grid(num_work, q.size(1), 1);

    const size_t smem_bytes =
        sizeof(opus::bf16_t) * 2 * 32 * 128 + sizeof(float) * 4 * 32 * 32;

    float* debug_ptr = nullptr;
    if(debug_qk_scores.has_value())
    {
        TORCH_CHECK(debug_qk_scores->is_cuda(),
                    __func__,
                    ": debug_qk_scores must be CUDA tensor when provided");
        TORCH_CHECK(debug_qk_scores->scalar_type() == at::ScalarType::Float,
                    __func__,
                    ": debug_qk_scores must be fp32");
        debug_ptr = debug_qk_scores->data_ptr<float>();
    }

    hipLaunchKernelGGL(
        (aiter::mha_splitk_stage1::mha_prefill_splitk_stage1_opus_kernel<
            opus::bf16_t,
            opus::bf16_t,
            float,
            128,
            1,
            1,
            128,
            32,
            true>),
        grid,
        block,
        smem_bytes,
        0,
        reinterpret_cast<const opus::bf16_t*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const opus::bf16_t*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const opus::bf16_t*>(v.data_ptr<at::BFloat16>()),
        kv_indptr.data_ptr<int32_t>(),
        kv_page_indices.data_ptr<int32_t>(),
        page_size,
        reinterpret_cast<const MhaPrefillSplitKWorkInfo*>(work_info_set.data_ptr<int32_t>()),
        num_work,
        softmax_scale,
        split_o.data_ptr<float>(),
        split_lse.data_ptr<float>(),
        debug_ptr);

    auto err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess,
                __func__,
                ": kernel launch failed with error ",
                hipGetErrorString(err));
}
