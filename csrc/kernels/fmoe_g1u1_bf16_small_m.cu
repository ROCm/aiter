// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_hip_common.h"
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

AITER_CTYPES_ERROR_DEF

namespace aiter {

// Direct BF16 G1U1 kernel for gfx1201 small-token-count MoE cases where
// M in {1,2,3,4,8,16}, model_dim=2048, inter_dim in {128,256},
// num_experts=256, and topk=8.
template <int group_size, bool gate_is_shuffled>
__global__ void fmoe_g1u1_bf16_small_m_stage1_kernel(
    const hip_bfloat16* __restrict__ input,
    const hip_bfloat16* __restrict__ gate,
    const int32_t* __restrict__ topk_ids,
    hip_bfloat16* __restrict__ act_out,
    int token_cnt,
    int model_dim,
    int inter_dim,
    int expert_cnt,
    int topk)
{
    int route = blockIdx.x;
    int i_base = blockIdx.y * group_size;
    if(route >= token_cnt * topk || i_base >= inter_dim)
        return;

    int token = route / topk;
    int topk_idx = route - token * topk;
    int expert = topk_ids[token * topk + topk_idx];
    // Treat invalid expert ids as no-op routes and avoid out-of-bounds weight reads.
    if(expert < 0 || expert >= expert_cnt)
    {
        for(int g = 0; g < group_size; ++g)
        {
            int i = i_base + g;
            if(i < inter_dim)
                act_out[route * inter_dim + i] = hip_bfloat16(0.0f);
        }
        return;
    }

    const hip_bfloat16* x = input + token * model_dim;
    __shared__ float gate_smem[group_size][256];
    __shared__ float up_smem[group_size][256];
    float gate_acc[group_size] = {};
    float up_acc[group_size] = {};
    int gate_nblock_stride = (model_dim / 32) * 512;

    for(int k = threadIdx.x; k < model_dim; k += blockDim.x)
    {
        float xv = static_cast<float>(x[k]);
        for(int g = 0; g < group_size; ++g)
        {
            int i = i_base + g;
            if(i < inter_dim)
            {
                if constexpr(gate_is_shuffled)
                {
                    int kblock = k / 32;
                    int k_in_32 = k - kblock * 32;
                    int klane = k_in_32 / 8;
                    int k8 = k_in_32 - klane * 8;
                    int gate_row = expert * (2 * inter_dim) + i;
                    int up_row = gate_row + inter_dim;
                    int gate_nblock = gate_row / 16;
                    int gate_nintra = gate_row - gate_nblock * 16;
                    int up_nblock = up_row / 16;
                    int up_nintra = up_row - up_nblock * 16;
                    int gate_idx = gate_nblock * gate_nblock_stride + kblock * 512 +
                                   klane * 128 + gate_nintra * 8 + k8;
                    int up_idx = up_nblock * gate_nblock_stride + kblock * 512 +
                                 klane * 128 + up_nintra * 8 + k8;
                    gate_acc[g] += xv * static_cast<float>(gate[gate_idx]);
                    up_acc[g] += xv * static_cast<float>(gate[up_idx]);
                }
                else
                {
                    const hip_bfloat16* w_gate =
                        gate + expert * (2 * inter_dim * model_dim) + i * model_dim;
                    const hip_bfloat16* w_up = gate + expert * (2 * inter_dim * model_dim) +
                                               (inter_dim + i) * model_dim;
                    gate_acc[g] += xv * static_cast<float>(w_gate[k]);
                    up_acc[g] += xv * static_cast<float>(w_up[k]);
                }
            }
        }
    }

    for(int g = 0; g < group_size; ++g)
    {
        gate_smem[g][threadIdx.x] = gate_acc[g];
        up_smem[g][threadIdx.x] = up_acc[g];
    }
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(threadIdx.x < stride)
        {
            for(int g = 0; g < group_size; ++g)
            {
                gate_smem[g][threadIdx.x] += gate_smem[g][threadIdx.x + stride];
                up_smem[g][threadIdx.x] += up_smem[g][threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if(threadIdx.x != 0)
        return;

    for(int g = 0; g < group_size; ++g)
    {
        int i = i_base + g;
        if(i < inter_dim)
        {
            float gate_val = gate_smem[g][0];
            float up_val = up_smem[g][0];
            float silu = gate_val / (1.0f + expf(-gate_val));
            act_out[route * inter_dim + i] = hip_bfloat16(silu * up_val);
        }
    }
}

template <int group_size, bool down_is_shuffled>
__global__ void fmoe_g1u1_bf16_small_m_stage2_kernel(
    const hip_bfloat16* __restrict__ act,
    const hip_bfloat16* __restrict__ down,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    hip_bfloat16* __restrict__ out,
    int token_cnt,
    int model_dim,
    int inter_dim,
    int expert_cnt,
    int topk)
{
    int token = blockIdx.x;
    int h_base = blockIdx.y * group_size;
    if(token >= token_cnt || h_base >= model_dim)
        return;

    __shared__ float smem[group_size][256];
    float acc[group_size] = {};
    int total = topk * inter_dim;
    int down_nblock_stride = (inter_dim / 32) * 512;
    for(int idx = threadIdx.x; idx < total; idx += blockDim.x)
    {
        int topk_idx = idx / inter_dim;
        int i = idx - topk_idx * inter_dim;
        int route = token * topk + topk_idx;
        int expert = topk_ids[route];
        if(expert < 0 || expert >= expert_cnt)
            continue;

        float route_weight = topk_weights[route];
        const hip_bfloat16* act_route = act + route * inter_dim;
        float act_val = static_cast<float>(act_route[i]) * route_weight;
        for(int g = 0; g < group_size; ++g)
        {
            int h = h_base + g;
            if(h < model_dim)
            {
                if constexpr(down_is_shuffled)
                {
                    int iblock = i / 32;
                    int i_in_32 = i - iblock * 32;
                    int ilane = i_in_32 / 8;
                    int i8 = i_in_32 - ilane * 8;
                    int down_row = expert * model_dim + h;
                    int down_nblock = down_row / 16;
                    int down_nintra = down_row - down_nblock * 16;
                    int down_idx = down_nblock * down_nblock_stride + iblock * 512 +
                                   ilane * 128 + down_nintra * 8 + i8;
                    acc[g] += act_val * static_cast<float>(down[down_idx]);
                }
                else
                {
                    const hip_bfloat16* w2 =
                        down + expert * (model_dim * inter_dim) + h * inter_dim;
                    acc[g] += act_val * static_cast<float>(w2[i]);
                }
            }
        }
    }

    for(int g = 0; g < group_size; ++g)
        smem[g][threadIdx.x] = acc[g];
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(threadIdx.x < stride)
        {
            for(int g = 0; g < group_size; ++g)
                smem[g][threadIdx.x] += smem[g][threadIdx.x + stride];
        }
        __syncthreads();
    }

    if(threadIdx.x != 0)
        return;

    for(int g = 0; g < group_size; ++g)
    {
        int h = h_base + g;
        if(h < model_dim)
            out[token * model_dim + h] = hip_bfloat16(smem[g][0]);
    }
}

static void launch_fmoe_g1u1_bf16_small_m(
    hip_bfloat16* out,
    const hip_bfloat16* input,
    const hip_bfloat16* gate,
    const hip_bfloat16* down,
    const int32_t* topk_ids,
    const float* topk_weights,
    hip_bfloat16* act_workspace,
    int token_cnt,
    int model_dim,
    int inter_dim,
    int expert_cnt,
    int topk,
    bool gate_is_shuffled,
    bool down_is_shuffled,
    hipStream_t stream)
{
    constexpr int threads = 256;
    constexpr int group_size = 8;

    dim3 grid1(token_cnt * topk, (inter_dim + group_size - 1) / group_size);
    if(gate_is_shuffled)
    {
        hipLaunchKernelGGL((fmoe_g1u1_bf16_small_m_stage1_kernel<group_size, true>),
                           grid1,
                           dim3(threads),
                           0,
                           stream,
                           input,
                           gate,
                           topk_ids,
                           act_workspace,
                           token_cnt,
                           model_dim,
                           inter_dim,
                           expert_cnt,
                           topk);
    }
    else
    {
        hipLaunchKernelGGL((fmoe_g1u1_bf16_small_m_stage1_kernel<group_size, false>),
                           grid1,
                           dim3(threads),
                           0,
                           stream,
                           input,
                           gate,
                           topk_ids,
                           act_workspace,
                           token_cnt,
                           model_dim,
                           inter_dim,
                           expert_cnt,
                           topk);
    }
    HIP_CALL_LAUNCH(hipGetLastError());

    dim3 grid2(token_cnt, (model_dim + group_size - 1) / group_size);
    if(down_is_shuffled)
    {
        hipLaunchKernelGGL((fmoe_g1u1_bf16_small_m_stage2_kernel<group_size, true>),
                           grid2,
                           dim3(threads),
                           0,
                           stream,
                           act_workspace,
                           down,
                           topk_ids,
                           topk_weights,
                           out,
                           token_cnt,
                           model_dim,
                           inter_dim,
                           expert_cnt,
                           topk);
    }
    else
    {
        hipLaunchKernelGGL((fmoe_g1u1_bf16_small_m_stage2_kernel<group_size, false>),
                           grid2,
                           dim3(threads),
                           0,
                           stream,
                           act_workspace,
                           down,
                           topk_ids,
                           topk_weights,
                           out,
                           token_cnt,
                           model_dim,
                           inter_dim,
                           expert_cnt,
                           topk);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

} // namespace aiter

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    fmoe_g1u1_bf16_small_m,
    (
    aiter_tensor_t* out,
    aiter_tensor_t* input,
    aiter_tensor_t* gate,
    aiter_tensor_t* down,
    aiter_tensor_t* topk_ids,
    aiter_tensor_t* topk_weights,
    aiter_tensor_t* act_workspace,
    int topk,
    bool gate_is_shuffled,
    bool down_is_shuffled,
    hipStream_t stream),
    (out,
     input,
     gate,
     down,
     topk_ids,
     topk_weights,
     act_workspace,
     topk,
     gate_is_shuffled,
     down_is_shuffled,
     stream))
{
    const HipDeviceGuard device_guard(input->device_id);
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16, __func__, " only supports bf16 output");
    AITER_CHECK(input->dtype() == AITER_DTYPE_bf16 && gate->dtype() == AITER_DTYPE_bf16 &&
                    down->dtype() == AITER_DTYPE_bf16,
                __func__,
                " only supports bf16 input and weights");
    AITER_CHECK(topk_ids->dtype() == AITER_DTYPE_i32, __func__, " topk_ids must be int32");
    AITER_CHECK(topk_weights->dtype() == AITER_DTYPE_fp32, __func__, " topk_weights must be fp32");
    AITER_CHECK(act_workspace->dtype() == AITER_DTYPE_bf16,
                __func__,
                " act_workspace must be bf16");

    int token_cnt = input->size(0);
    int model_dim = input->size(1);
    int expert_cnt = gate->size(0);
    int inter_dim = down->size(2);
    AITER_CHECK(token_cnt == 1 || token_cnt == 2 || token_cnt == 3 || token_cnt == 4 ||
                    token_cnt == 8 || token_cnt == 16,
                __func__,
                " only supports M in {1,2,3,4,8,16}");
    AITER_CHECK(model_dim == 2048, __func__, " only supports model_dim=2048");
    AITER_CHECK(inter_dim == 128 || inter_dim == 256,
                __func__,
                " only supports inter_dim in {128,256}");
    AITER_CHECK(expert_cnt == 256, __func__, " only supports 256 experts");
    AITER_CHECK(topk == 8, __func__, " only supports topk=8");
    AITER_CHECK(out->size(0) == token_cnt && out->size(1) == model_dim,
                __func__,
                " output shape mismatch");
    AITER_CHECK(gate->size(1) == inter_dim * 2,
                __func__,
                " expects G1U1 gate shape [E, 2*inter_dim, model_dim]");
    AITER_CHECK(down->size(0) == expert_cnt,
                __func__,
                " gate/down expert count mismatch");
    AITER_CHECK(gate->size(2) == model_dim && down->size(1) == model_dim,
                __func__,
                " model_dim mismatch");
    AITER_CHECK(topk_ids->size(0) == token_cnt && topk_ids->size(1) == topk,
                __func__,
                " topk_ids shape mismatch");
    AITER_CHECK(topk_weights->size(0) == token_cnt && topk_weights->size(1) == topk,
                __func__,
                " topk_weights shape mismatch");
    AITER_CHECK(act_workspace->size(0) == token_cnt * topk && act_workspace->size(1) == inter_dim,
                __func__,
                " act_workspace shape mismatch");

    aiter::launch_fmoe_g1u1_bf16_small_m(reinterpret_cast<hip_bfloat16*>(out->ptr),
                                         reinterpret_cast<const hip_bfloat16*>(input->ptr),
                                         reinterpret_cast<const hip_bfloat16*>(gate->ptr),
                                         reinterpret_cast<const hip_bfloat16*>(down->ptr),
                                         reinterpret_cast<const int32_t*>(topk_ids->ptr),
                                         reinterpret_cast<const float*>(topk_weights->ptr),
                                         reinterpret_cast<hip_bfloat16*>(act_workspace->ptr),
                                         token_cnt,
                                         model_dim,
                                         inter_dim,
                                         expert_cnt,
                                         topk,
                                         gate_is_shuffled,
                                         down_is_shuffled,
                                         stream);
}
