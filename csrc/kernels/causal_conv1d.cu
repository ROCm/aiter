// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/extension.h>

#include "aiter_hip_common.h"
#include "ck_tile/core.hpp"
#include "dispatch_utils.h"
#include "hip_compat.h"
#include "py_itfs_common.h"

// Helper macros
#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA/HIP tensor")

#define HIP_CHECK(err)                                                      \
    do {                                                                    \
        hipError_t err_ = (err);                                            \
        if (err_ != hipSuccess) {                                           \
            throw std::runtime_error(                                       \
                std::string("HIP error: ") + hipGetErrorString(err_) +      \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));       \
        }                                                                   \
    } while (0)

namespace aiter {

// Simple causal conv1d kernel
// Each thread processes one output element
template <typename DTYPE>
__global__ void causal_conv1d_fwd_kernel(
    DTYPE* __restrict__ out,
    const DTYPE* __restrict__ x,
    const DTYPE* __restrict__ weight,
    const DTYPE* __restrict__ bias,
    const int32_t batch,
    const int32_t dim,
    const int32_t seqlen,
    const int32_t width,
    const bool use_silu)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t total_elements = batch * dim * seqlen;
    
    if(idx >= total_elements)
        return;
    
    // Decode indices
    const int32_t t = idx % seqlen;
    const int32_t d = (idx / seqlen) % dim;
    const int32_t b = idx / (dim * seqlen);
    
    // Compute convolution
    float acc = 0.0f;
    
    #pragma unroll
    for(int32_t i = 0; i < width; ++i)
    {
        const int32_t input_t = t - width + 1 + i;
        if(input_t >= 0)
        {
            const int32_t input_idx = b * (dim * seqlen) + d * seqlen + input_t;
            acc += ck_tile::type_convert<float>(x[input_idx]) * 
                   ck_tile::type_convert<float>(weight[d * width + i]);
        }
    }
    
    // Add bias
    if(bias != nullptr)
    {
        acc += ck_tile::type_convert<float>(bias[d]);
    }
    
    // Apply SiLU activation
    if(use_silu)
    {
        acc = acc / (1.0f + expf(-acc));
    }
    
    out[idx] = ck_tile::type_convert<DTYPE>(acc);
}

// Launch function
void causal_conv1d_fwd(
    torch::Tensor& out,           // [batch, dim, seqlen]
    const torch::Tensor& x,       // [batch, dim, seqlen]
    const torch::Tensor& weight,  // [dim, width]
    const torch::Tensor& bias,    // [dim] or empty
    bool use_silu)
{
    CHECK_INPUT(out);
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    
    const int32_t batch = x.size(0);
    const int32_t dim = x.size(1);
    const int32_t seqlen = x.size(2);
    const int32_t width = weight.size(1);
    
    TORCH_CHECK(out.size(0) == batch && out.size(1) == dim && out.size(2) == seqlen,
                "Output shape mismatch");
    TORCH_CHECK(weight.size(0) == dim, "Weight shape mismatch");
    
    const void* bias_ptr = nullptr;
    if(bias.defined() && bias.numel() > 0)
    {
        CHECK_INPUT(bias);
        TORCH_CHECK(bias.size(0) == dim, "Bias shape mismatch");
        bias_ptr = bias.data_ptr();
    }
    
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(x));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    
    const int32_t total_elements = batch * dim * seqlen;
    const int32_t threads = 256;
    const int32_t blocks = (total_elements + threads - 1) / threads;
    
    AITER_DISPATCH_FLOATING16_TYPES(x.scalar_type(), "causal_conv1d_fwd", [&] {
        using dtype = typename t2ck<scalar_t>::type;
        
        const dtype* bias_ptr_typed = bias_ptr ? reinterpret_cast<const dtype*>(bias_ptr) : nullptr;
        
        aiter::causal_conv1d_fwd_kernel<dtype>
            <<<blocks, threads, 0, stream>>>(
                reinterpret_cast<dtype*>(out.data_ptr()),
                reinterpret_cast<const dtype*>(x.data_ptr()),
                reinterpret_cast<const dtype*>(weight.data_ptr()),
                bias_ptr_typed,
                batch,
                dim,
                seqlen,
                width,
                use_silu);
    });
    
    HIP_CHECK(hipGetLastError());
}

} // namespace aiter
