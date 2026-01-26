// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "py_itfs_common.h"
#include "topk_softmax_api.hpp"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

namespace aiter {

constexpr int DPP_QUAD_PERM_1032 = 0xb1;  // quad_perm: [1,0,3,2]
constexpr int DPP_QUAD_PERM_2301 = 0x4e;  // quad_perm: [2,3,0,1]
constexpr int DPP_ROW_SHR_4      = 0x114; // row_shr:4
constexpr int DPP_ROW_SHR_8      = 0x118; // row_shr:8
constexpr int DPP_ROW_BCAST_15   = 0x142; // row_bcast:15
constexpr int DPP_ROW_BCAST_31   = 0x143; // row_bcast:31

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + __expf(-x)); }

#define DPP_REDUCE_STEP(dpp_code)                                                              \
    do                                                                                         \
    {                                                                                          \
        float remote_val = __builtin_bit_cast(                                                 \
            float,                                                                             \
            __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, val), dpp_code, 0xf, 0xf, true)); \
        int remote_idx = __builtin_amdgcn_mov_dpp(idx, dpp_code, 0xf, 0xf, true);              \
        idx            = val > remote_val ? idx : remote_idx;                                  \
        val            = val > remote_val ? val : remote_val;                                  \
    } while(0)

__device__ __forceinline__ void warp_reduce_argmax(float& val_o, int& idx)
{
    float val = val_o;

    DPP_REDUCE_STEP(0xb1);
    DPP_REDUCE_STEP(0x4e);
    DPP_REDUCE_STEP(0x114);
    DPP_REDUCE_STEP(0x118);
    DPP_REDUCE_STEP(0x142);
    DPP_REDUCE_STEP(0x143);

    val_o = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, val), 63));
    idx   = __builtin_amdgcn_readlane(idx, 63);
}

#undef DPP_REDUCE_STEP

template <typename scalar_t, int BLOCK_SIZE = 256, int EXPERTS_PER_THREAD = 2>
__global__ void topk_sigmoid_kernel(const scalar_t* __restrict__ gating_output,
                                    float* __restrict__ topk_weights,
                                    int32_t* __restrict__ topk_indices,
                                    const int num_tokens,
                                    const int num_experts,
                                    const int topk)
{
    constexpr int WARP_SIZE   = 64;
    const int warp_id         = threadIdx.x / WARP_SIZE;
    const int lane_id         = threadIdx.x % WARP_SIZE;
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    const int token_id        = blockIdx.x * warps_per_block + warp_id;

    if(token_id >= num_tokens)
        return;

    const scalar_t* token_gating = gating_output + token_id * num_experts;
    float* token_weights         = topk_weights + token_id * topk;
    int32_t* token_indices       = topk_indices + token_id * topk;

    float local_vals[EXPERTS_PER_THREAD];
    int local_indices[EXPERTS_PER_THREAD];

#pragma unroll
    for(int i = 0; i < EXPERTS_PER_THREAD; i++)
    {
        int expert_id = lane_id + i * WARP_SIZE;
        if(expert_id < num_experts)
        {
            float val        = static_cast<float>(token_gating[expert_id]);
            local_vals[i]    = sigmoid_f(val);
            local_indices[i] = expert_id;
        }
        else
        {
            local_vals[i]    = -INFINITY;
            local_indices[i] = -1;
        }
    }

    for(int k = 0; k < topk; k++)
    {
        float max_val = -INFINITY;
        int max_idx   = -1;

#pragma unroll
        for(int i = 0; i < EXPERTS_PER_THREAD; i++)
        {
            if(local_vals[i] > max_val)
            {
                max_val = local_vals[i];
                max_idx = local_indices[i];
            }
        }

        warp_reduce_argmax(max_val, max_idx);

        if(lane_id == 0)
        {
            token_weights[k] = max_val;
            token_indices[k] = max_idx;
        }

#pragma unroll
        for(int i = 0; i < EXPERTS_PER_THREAD; i++)
        {
            if(local_indices[i] == max_idx)
            {
                local_vals[i] = -INFINITY;
            }
        }
    }
}

#define LAUNCH_KERNEL(scalar_type, experts_per_thread)               \
    topk_sigmoid_kernel<scalar_type, BLOCK_SIZE, experts_per_thread> \
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(                     \
            reinterpret_cast<const scalar_type*>(gating_output),     \
            reinterpret_cast<float*>(topk_weights),                  \
            reinterpret_cast<int32_t*>(topk_indices),                \
            num_tokens,                                              \
            num_experts,                                             \
            topk)

static void topk_sigmoid_gfx9(const void* gating_output,
                              void* topk_weights,
                              void* topk_indices,
                              int num_tokens,
                              int num_experts,
                              int topk,
                              torch::ScalarType dtype,
                              hipStream_t stream)
{
    constexpr int BLOCK_SIZE      = 256;
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 64;
    const int num_blocks          = (num_tokens + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    if(num_experts <= 64)
    {
        if(dtype == torch::kFloat16)
        {
            LAUNCH_KERNEL(__half, 1);
        }
        else if(dtype == torch::kBFloat16)
        {
            LAUNCH_KERNEL(__hip_bfloat16, 1);
        }
        else
        {
            LAUNCH_KERNEL(float, 1);
        }
    }
    else if(num_experts <= 128)
    {
        if(dtype == torch::kFloat16)
        {
            LAUNCH_KERNEL(__half, 2);
        }
        else if(dtype == torch::kBFloat16)
        {
            LAUNCH_KERNEL(__hip_bfloat16, 2);
        }
        else
        {
            LAUNCH_KERNEL(float, 2);
        }
    }
    else if(num_experts <= 256)
    {
        if(dtype == torch::kFloat16)
        {
            LAUNCH_KERNEL(__half, 4);
        }
        else if(dtype == torch::kBFloat16)
        {
            LAUNCH_KERNEL(__hip_bfloat16, 4);
        }
        else
        {
            LAUNCH_KERNEL(float, 4);
        }
    }
    else if(num_experts <= 512)
    {
        if(dtype == torch::kFloat16)
        {
            LAUNCH_KERNEL(__half, 8);
        }
        else if(dtype == torch::kBFloat16)
        {
            LAUNCH_KERNEL(__hip_bfloat16, 8);
        }
        else
        {
            LAUNCH_KERNEL(float, 8);
        }
    }
    else
    {
        TORCH_CHECK(false, "topk_sigmoid_gfx9 supports up to 512 experts, got ", num_experts);
    }
}

#undef LAUNCH_KERNEL

static void
topk_sigmoid_ck(torch::Tensor topk_weights, torch::Tensor topk_indices, torch::Tensor gating_output)
{
    const int tokens  = gating_output.size(0);
    const int experts = gating_output.size(1);
    const int topk    = topk_weights.size(1);

    // Assume default strides
    const int stride_input  = experts;
    const int stride_output = topk;

    // Determine datatypes
    auto dtype_to_string = [](const auto dtype) -> std::string {
        if(dtype == torch::kFloat16)
        {
            return "fp16";
        }
        else if(dtype == torch::kBFloat16)
        {
            return "bf16";
        }
        else if(dtype == torch::kFloat32)
        {
            return "fp32";
        }
        else
        {
            throw std::runtime_error("invalid datatype for topk_sigmoid: only fp16/bf16/fp32!");
        }
    };
    std::string input_prec  = dtype_to_string(gating_output.dtype());
    std::string weight_prec = dtype_to_string(topk_weights.dtype());

    // Prepare kernel arguments
    static const std::string activation = "sigmoid";
    topk_softmax_trait trait{input_prec, weight_prec, experts, activation};

    topk_softmax_kargs karg{gating_output.data_ptr(),
                            topk_weights.data_ptr(),
                            topk_indices.data_ptr(),
                            tokens,
                            experts,
                            topk,
                            stride_input,
                            stride_output};

    ck_tile::stream_config sc{at::hip::getCurrentHIPStream()};

    topk_softmax(trait, karg, sc);
}

static bool is_gfx9_arch() { return isGPUArch({"gfx9"}); }

void topk_sigmoid(torch::Tensor topk_weights,
                  torch::Tensor topk_indices,
                  torch::Tensor gating_output)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(at::device_of(gating_output));

    const int num_tokens  = gating_output.size(0);
    const int num_experts = gating_output.size(1);
    const int topk        = topk_weights.size(1);
    const auto dtype      = gating_output.scalar_type();

    if(is_gfx9_arch())
    {
        TORCH_CHECK(num_experts <= 512, "topk_sigmoid supports up to 512 experts");
        TORCH_CHECK(topk <= 32, "topk_sigmoid supports up to 32 top-k");

        topk_sigmoid_gfx9(gating_output.data_ptr(),
                          topk_weights.data_ptr(),
                          topk_indices.data_ptr(),
                          num_tokens,
                          num_experts,
                          topk,
                          dtype,
                          at::hip::getCurrentHIPStream());
        return;
    }

    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "CK fallback only supports fp16/bf16 (use GFX9 for fp32)");
    TORCH_CHECK(num_experts <= 192, "CK fallback supports up to 192 experts (use GFX9 for more)");

    topk_sigmoid_ck(topk_weights, topk_indices, gating_output);
}

} // namespace aiter
