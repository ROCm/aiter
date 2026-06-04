#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused softmax -> topk -> counting-sort kernel for the MoE decode path.
//
// Combines the work of `topk_softmax` (csrc/kernels/topk_softmax_kernels.cu)
// and the opus oneshot `moe_sorting_opus_fwd` (csrc/include/moe_sorting_opus.h)
// into a single launch. The intermediate topk_ids / topk_weights are kept in
// LDS instead of being materialised in global memory.
//
// Only the oneshot-feasible decode shapes are supported (the full token set
// must fit in one CU's LDS). Larger problems must fall back to the separate
// kernels. The produced sorted_ids / sorted_weights / sorted_expert_ids /
// num_valid_ids and the zeroed moe_buf match the separate chain exactly.

#include <torch/extension.h>

namespace aiter {

// gating_output : [num_tokens, >=num_experts]  (fp32 / fp16 / bf16)
// sorted_ids        : [max_num_tokens_padded]   int32
// sorted_weights    : [max_num_tokens_padded]   fp32
// sorted_expert_ids : [max_num_m_blocks]         int32
// num_valid_ids     : [2]                        int32
// moe_buf           : [num_tokens, model_dim]    (any dtype, zeroed)
void fused_topk_moe_sorting_fwd(torch::Tensor& gating_output,
                                torch::Tensor& sorted_ids,
                                torch::Tensor& sorted_weights,
                                torch::Tensor& sorted_expert_ids,
                                torch::Tensor& num_valid_ids,
                                torch::Tensor& moe_buf,
                                int num_experts,
                                int topk,
                                int unit_size,
                                bool need_renorm);

// Largest token count that still fits the fused oneshot LDS budget for the
// given (num_experts, topk). Returns <= 0 when even a single token cannot fit.
int fused_topk_moe_sorting_max_tokens(int num_experts, int topk);

} // namespace aiter

#ifdef FUSED_TOPK_MOE_SORTING_IMPL
// ============================================================================
// Implementation (only compiled in the kernels translation unit)
// ============================================================================

#include "dispatch_utils.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <algorithm>
#include <hip/hip_runtime.h>

namespace aiter {

namespace fused_topk_sort {

static constexpr int kBlockSize = 256;
// Upper bound on topk; bounds the per-thread register scratch in phase 1.
static constexpr int kMaxTopk = 16;
// gfx942 / gfx950 expose 64 KiB of LDS per workgroup.
static constexpr int kLdsBytesLimit = 64 * 1024;

// token id (low 24 bit) + topk slot (high 8 bit); mirrors
// MOE_SORTING_MOCK_ID in moe_sorting_opus.h so downstream kernels decode it.
__device__ __forceinline__ int mock_id(int token_id, int topk_id)
{
    return static_cast<int>((static_cast<uint32_t>(token_id) & 0x00ffffffu) |
                            ((static_cast<uint32_t>(topk_id) & 0xffu) << 24));
}

// LDS budget for a (tokens, experts, topk) problem, in bytes.
__host__ __device__ __forceinline__ size_t lds_bytes(int tokens, int num_experts, int topk)
{
    const size_t cols = static_cast<size_t>(num_experts) + 1;
    // smem_tokens[tokens][cols] + cumsum[cols] (int) + weights[tokens][topk] (float)
    return ((static_cast<size_t>(tokens) + 1) * cols + static_cast<size_t>(tokens) * topk) *
           sizeof(int);
}

template <typename scalar_t>
__global__ void __launch_bounds__(kBlockSize)
    fused_topk_moe_sorting_kernel(const scalar_t* __restrict__ gating,
                                  int M,
                                  int E,
                                  int K,
                                  int row_stride,
                                  int unit_size,
                                  bool need_renorm,
                                  int* __restrict__ sorted_ids,
                                  float* __restrict__ sorted_weights,
                                  int* __restrict__ sorted_expert_ids,
                                  int* __restrict__ num_valid_ids,
                                  void* __restrict__ moe_buf,
                                  long moe_buf_bytes)
{
    const int tid  = static_cast<int>(threadIdx.x);
    const int cols = E + 1;

    // Phase 3: every block except block 0 zeroes the fused-moe output buffer.
    if(blockIdx.x != 0)
    {
        const long nvec   = moe_buf_bytes >> 4; // 16-byte chunks
        const long stride = static_cast<long>(gridDim.x - 1) * kBlockSize;
        const long base   = static_cast<long>(blockIdx.x - 1) * kBlockSize + tid;
        uint4* pv         = reinterpret_cast<uint4*>(moe_buf);
        const uint4 z     = make_uint4(0u, 0u, 0u, 0u);
        for(long i = base; i < nvec; i += stride)
            pv[i] = z;
        const long tail_start = nvec << 4;
        char* pc              = reinterpret_cast<char*>(moe_buf);
        for(long i = tail_start + base; i < moe_buf_bytes; i += stride)
            pc[i] = 0;
        return;
    }

    extern __shared__ char smem_raw[];
    int* smem_tokens   = reinterpret_cast<int*>(smem_raw);          // [M][cols]
    int* smem_cumsum   = smem_tokens + static_cast<size_t>(M) * cols; // [cols]
    float* smem_weight = reinterpret_cast<float*>(smem_cumsum + cols); // [M][K]

    for(int i = tid; i < M * cols; i += kBlockSize)
        smem_tokens[i] = 0;
    __syncthreads();

    // Phase 1: softmax + topk selection, one thread per token.
    for(int t = tid; t < M; t += kBlockSize)
    {
        const scalar_t* row = gating + static_cast<long>(t) * row_stride;

        float row_max = -INFINITY;
        for(int e = 0; e < E; ++e)
        {
            const float v = static_cast<float>(row[e]);
            row_max       = v > row_max ? v : row_max;
        }

        int sel_e[kMaxTopk];
        float sel_numer[kMaxTopk];
        for(int k = 0; k < K; ++k)
        {
            float best_v = -INFINITY;
            int best_e   = 0;
            for(int e = 0; e < E; ++e)
            {
                const float v = static_cast<float>(row[e]);
                bool used     = false;
                for(int j = 0; j < k; ++j)
                    used |= (sel_e[j] == e);
                // strict '>' keeps the lowest index on ties (matches topk_softmax)
                if(!used && v > best_v)
                {
                    best_v = v;
                    best_e = e;
                }
            }
            sel_e[k]     = best_e;
            sel_numer[k] = expf(best_v - row_max);
        }

        float denom = 0.f;
        if(need_renorm)
        {
            for(int k = 0; k < K; ++k)
                denom += sel_numer[k];
        }
        else
        {
            for(int e = 0; e < E; ++e)
                denom += expf(static_cast<float>(row[e]) - row_max);
        }
        const float inv = denom != 0.f ? 1.f / denom : 1.f;

        for(int k = 0; k < K; ++k)
        {
            smem_weight[t * K + k]            = sel_numer[k] * inv;
            smem_tokens[t * cols + sel_e[k]]  = k + 1; // topk slot, 1-based
        }
    }
    __syncthreads();

    // Phase 2a: per-expert token counts.
    for(int e = tid; e < E; e += kBlockSize)
    {
        int c = 0;
        for(int t = 0; t < M; ++t)
            c += (smem_tokens[t * cols + e] != 0);
        smem_cumsum[e + 1] = c;
    }
    if(tid == 0)
        smem_cumsum[0] = 0;
    __syncthreads();

    // Phase 2b: exclusive prefix sum of unit-size-padded counts.
    if(tid == 0)
    {
        int run = 0;
        for(int e = 0; e < E; ++e)
        {
            const int c      = smem_cumsum[e + 1];
            const int blocks = (c + unit_size - 1) / unit_size; // 0 when c == 0
            run += blocks * unit_size;
            smem_cumsum[e + 1] = run;
        }
        num_valid_ids[0] = run;
        num_valid_ids[1] = M;
    }
    __syncthreads();

    // Phase 2c: expert id per unit-size block.
    for(int e = tid; e < E; e += kBlockSize)
    {
        const int e_start = smem_cumsum[e];
        const int e_end   = smem_cumsum[e + 1];
        if(e_start == e_end) // expert with zero tokens is skipped
            continue;
        for(int i = e_start; i < e_end; i += unit_size)
            sorted_expert_ids[i / unit_size] = e;
    }
    __syncthreads();

    // Phase 2d: scatter tokens (increasing token order) then pad each expert.
    for(int e = tid; e < E; e += kBlockSize)
    {
        int pos           = smem_cumsum[e];
        const int e_end   = smem_cumsum[e + 1];
        for(int t = 0; t < M; ++t)
        {
            const int x = smem_tokens[t * cols + e];
            if(x != 0)
            {
                const int slot         = x - 1;
                sorted_ids[pos]        = mock_id(t, slot);
                sorted_weights[pos]    = smem_weight[t * K + slot];
                ++pos;
            }
        }
        while(pos < e_end)
        {
            sorted_ids[pos]     = mock_id(M, K);
            sorted_weights[pos] = 0.f;
            ++pos;
        }
    }
}

} // namespace fused_topk_sort

int fused_topk_moe_sorting_max_tokens(int num_experts, int topk)
{
    using namespace fused_topk_sort;
    int lo = 0, hi = 0;
    // exponential search for the largest token count that still fits
    for(int t = 1;; t <<= 1)
    {
        if(lds_bytes(t, num_experts, topk) > static_cast<size_t>(kLdsBytesLimit))
        {
            hi = t;
            break;
        }
        lo = t;
        if(t > (1 << 20))
        {
            hi = t;
            break;
        }
    }
    while(lo + 1 < hi)
    {
        const int mid = (lo + hi) / 2;
        if(lds_bytes(mid, num_experts, topk) > static_cast<size_t>(kLdsBytesLimit))
            hi = mid;
        else
            lo = mid;
    }
    return lo;
}

void fused_topk_moe_sorting_fwd(torch::Tensor& gating_output,
                                torch::Tensor& sorted_ids,
                                torch::Tensor& sorted_weights,
                                torch::Tensor& sorted_expert_ids,
                                torch::Tensor& num_valid_ids,
                                torch::Tensor& moe_buf,
                                int num_experts,
                                int topk,
                                int unit_size,
                                bool need_renorm)
{
    using namespace fused_topk_sort;

    TORCH_CHECK(gating_output.dim() == 2, "gating_output must be 2D [tokens, experts]");
    TORCH_CHECK(topk > 0 && topk <= kMaxTopk, "fused topk must be in (0, 16]");
    TORCH_CHECK(gating_output.size(-1) >= num_experts,
                "gating_output last dim smaller than num_experts");
    TORCH_CHECK(sorted_ids.scalar_type() == at::ScalarType::Int, "sorted_ids must be int32");
    TORCH_CHECK(sorted_weights.scalar_type() == at::ScalarType::Float,
                "sorted_weights must be fp32");
    TORCH_CHECK(sorted_expert_ids.scalar_type() == at::ScalarType::Int,
                "sorted_expert_ids must be int32");
    TORCH_CHECK(num_valid_ids.scalar_type() == at::ScalarType::Int, "num_valid_ids must be int32");

    const int M          = static_cast<int>(gating_output.size(0));
    const int E          = num_experts;
    const int row_stride = static_cast<int>(gating_output.stride(0));

    TORCH_CHECK(lds_bytes(M, E, topk) <= static_cast<size_t>(kLdsBytesLimit),
                "fused_topk_moe_sorting: problem exceeds LDS budget, use the separate kernels");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(gating_output));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    hipDeviceProp_t prop;
    int dev = 0;
    (void)hipGetDevice(&dev);
    (void)hipGetDeviceProperties(&prop, dev);
    const int grid = std::max(prop.multiProcessorCount, 2);

    const long moe_buf_bytes =
        static_cast<long>(moe_buf.numel()) * static_cast<long>(moe_buf.element_size());
    const unsigned smem = static_cast<unsigned>(lds_bytes(M, E, topk));

    VLLM_DISPATCH_FLOATING_TYPES(gating_output.scalar_type(), "fused_topk_moe_sorting", [&] {
        fused_topk_moe_sorting_kernel<scalar_t><<<grid, kBlockSize, smem, stream>>>(
            gating_output.data_ptr<scalar_t>(),
            M,
            E,
            topk,
            row_stride,
            unit_size,
            need_renorm,
            sorted_ids.data_ptr<int>(),
            sorted_weights.data_ptr<float>(),
            sorted_expert_ids.data_ptr<int>(),
            num_valid_ids.data_ptr<int>(),
            moe_buf.data_ptr(),
            moe_buf_bytes);
    });
}

} // namespace aiter

#endif // FUSED_TOPK_MOE_SORTING_IMPL
