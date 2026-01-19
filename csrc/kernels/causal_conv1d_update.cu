// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Tri Dao.
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// 
// Causal 1D Convolution Update Kernel for AIter Framework
// Adapted for AMD MI308 GPU (ROCm/HIP)
//
// This kernel implements causal 1D convolution update for autoregressive generation,
// designed for Mamba-style models. It processes one or a few new tokens at a time
// while maintaining a sliding window state buffer.
//
// Key Features:
// - Supports both circular and non-circular buffer modes
// - Continuous batching support with flexible state indexing
// - SiLU activation option
// - Optimized for AMD MI308 GPU (64-thread wavefront)
// - Supports fp16, bf16, and fp32 data types
// - Convolution widths: 2, 3, 4

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

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float)  {                                   \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

#define DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(WTYPE, NAME, ...)                     \
    if (WTYPE == at::ScalarType::Half) {                                             \
        using weight_t = at::Half;                                                   \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == at::ScalarType::BFloat16) {                                  \
        using weight_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == at::ScalarType::Float)  {                                    \
        using weight_t = float;                                                      \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
    }

namespace aiter {

// ============================================================================
// ConvParamsBaseUpdate - Kernel Parameters Structure
// ============================================================================
// Contains all parameters needed for the causal_conv1d_update kernel
// Optimized for efficient GPU memory access and minimal register pressure

struct ConvParamsBaseUpdate {
    using index_t = uint32_t;

    // Tensor dimensions
    int batch, dim, seqlen, width;
    bool silu_activation;

    // Input tensor strides (for flexible memory layouts)
    index_t x_batch_stride;    // Stride between batches in x
    index_t x_c_stride;        // Stride between channels in x
    index_t x_l_stride;        // Stride between sequence positions in x
    
    // Weight tensor strides
    index_t weight_c_stride;       // Stride between channels in weight
    index_t weight_width_stride;   // Stride within convolution width
    
    // Output tensor strides
    index_t out_batch_stride;  // Stride between batches in output
    index_t out_c_stride;      // Stride between channels in output
    index_t out_l_stride;      // Stride between sequence positions in output

    // Convolution state dimensions and strides
    int conv_state_len;               // Length of state buffer (>= width-1)
    index_t conv_state_batch_stride;  // Stride between batches in state
    index_t conv_state_c_stride;      // Stride between channels in state
    index_t conv_state_l_stride;      // Stride within state buffer

    // Data pointers
    void *__restrict__ x_ptr;          // Input data [batch, dim, seqlen]
    void *__restrict__ weight_ptr;     // Convolution weights [dim, width]
    void *__restrict__ bias_ptr;       // Bias [dim] (nullable)
    void *__restrict__ out_ptr;        // Output data [batch, dim, seqlen]

    void *__restrict__ conv_state_ptr; // State buffer [batch, dim, state_len]
    int32_t *__restrict__ cache_seqlens;  // Sequence lengths for circular buffer (nullable)

    // Continuous batching support
    int32_t *__restrict__ conv_state_indices_ptr;  // Batch-to-state mapping (nullable)
    int pad_slot_id;  // Slot ID indicating padding (skip processing if matched)
};

// ============================================================================
// Kernel Traits - Template Configuration
// ============================================================================
// Defines compile-time constants for kernel specialization
// Allows the compiler to optimize for specific configurations

template<int kNThreads_, int kWidth_, typename input_t_, typename weight_t_>
struct Causal_conv1d_update_kernel_traits {
    using input_t = input_t_;    // Input/output data type (float, fp16, bf16)
    using weight_t = weight_t_;  // Weight data type (usually same as input_t)
    static constexpr int kNThreads = kNThreads_;  // Threads per block (typically 64 for AMD)
    static constexpr int kWidth = kWidth_;        // Convolution kernel width (2, 3, or 4)
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4, "Only 2-byte or 4-byte types supported");
};

// ============================================================================
// Update Kernel
// ============================================================================
// Implements causal 1D convolution update for autoregressive generation
// Processes one or few new tokens at a time while maintaining a sliding window state

template<typename Ktraits, bool kIsCircularBuffer>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_update_kernel(ConvParamsBaseUpdate params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y * kNThreads + tidx;
    
    // Early exit for out-of-bounds channels
    if (channel_id >= params.dim) return;

    // Input pointer for this batch and channel
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + channel_id * params.x_c_stride;

    // Handle continuous batching: If conv_state_indices is set, gather conv_state from non-contiguous locations
    // Otherwise, conv_state coordinate is the same as batch_id
    const int conv_state_batch_coord = params.conv_state_indices_ptr == nullptr
        ? batch_id
        : params.conv_state_indices_ptr[batch_id];
    
    // Skip processing if this is a padding slot
    if (conv_state_batch_coord == params.pad_slot_id){
        return;
    }
    
    // Conv state pointer for this channel
    input_t *conv_state = reinterpret_cast<input_t *>(params.conv_state_ptr)
        + conv_state_batch_coord * params.conv_state_batch_stride
        + channel_id * params.conv_state_c_stride;
    
    // Weight and output pointers
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + channel_id * params.out_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);

    // State management variables
    int state_len = params.conv_state_len;
    int advance_len = params.seqlen;
    int cache_seqlen = kIsCircularBuffer ? params.cache_seqlens[batch_id] % state_len : 0;
    int update_idx = cache_seqlen - (kWidth - 1);
    update_idx = update_idx < 0 ? update_idx + state_len : update_idx;

    // Load weights into registers for fast access
    float weight_vals[kWidth] = {0};
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { weight_vals[i] = float(weight[i * params.weight_width_stride]); }

    // Sliding window buffer for input values
    float x_vals[kWidth] = {0};
    
    // Initialize x_vals with historical state values
    if constexpr (!kIsCircularBuffer) {
        // Non-circular mode: Shift old data to make room for new data
        #pragma unroll 2
        for (int i = 0; i < state_len - advance_len - (kWidth - 1); ++i) {
            conv_state[i * params.conv_state_l_stride] = conv_state[(i + advance_len) * params.conv_state_l_stride];
        }
        
        // Load the most recent (kWidth-1) historical states into x_vals
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) {
            input_t state_val = conv_state[(state_len - (kWidth - 1) + i) * params.conv_state_l_stride];
            if (i < advance_len + (kWidth - 1) && state_len - advance_len - (kWidth - 1) + i >= 0) {
                conv_state[(state_len - advance_len - (kWidth - 1) + i) * params.conv_state_l_stride] = state_val;
            }
            x_vals[i] = float(state_val);
        }
    } else {
        // Circular mode: Load (kWidth-1) historical values in circular order
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i, update_idx = update_idx + 1 >= state_len ? update_idx + 1 - state_len : update_idx + 1) {
            input_t state_val = conv_state[update_idx * params.conv_state_l_stride];
            x_vals[i] = float(state_val);
        }
    }
    
    // Main convolution loop: Process each new input token
    #pragma unroll 2
    for (int i = 0; i < params.seqlen; ++i) {
        // Read new input
        input_t x_val = x[i * params.x_l_stride];
        
        // Update conv_state with new input
        if constexpr (!kIsCircularBuffer) {
            // Non-circular: Write to the end of the buffer
            if (i < advance_len && state_len - advance_len + i >= 0) {
                conv_state[(state_len - advance_len + i) * params.conv_state_l_stride] = x_val;
            }
        } else {
            // Circular: Write at current index and advance
            conv_state[update_idx * params.conv_state_l_stride] = x_val;
            ++update_idx;
            update_idx = update_idx >= state_len ? update_idx - state_len : update_idx;
        }
        
        // Add new input to the sliding window
        x_vals[kWidth - 1] = float(x_val);
        
        // Compute convolution output
        float out_val = bias_val;
        #pragma unroll
        for (int j = 0; j < kWidth; ++j) { out_val += weight_vals[j] * x_vals[j]; }
        
        // Apply SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
        if (params.silu_activation) { out_val = out_val / (1 + expf(-out_val)); }
        
        // Write output
        out[i * params.out_l_stride] = input_t(out_val);
        
        // Shift the sliding window left by 1 position
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) { x_vals[i] = x_vals[i + 1]; }
    }
}

// ============================================================================
// Launch Functions
// ============================================================================
// Helper functions to configure and launch the kernel with appropriate settings

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_update_launch(ConvParamsBaseUpdate &params, hipStream_t stream) {
    using Ktraits = Causal_conv1d_update_kernel_traits<kNThreads, kWidth, input_t, weight_t>;

    // Grid configuration: one block per batch, channels distributed across blocks
    dim3 grid(params.batch, (params.dim + kNThreads - 1) / kNThreads);

    // Select kernel variant based on buffer mode
    auto kernel = params.cache_seqlens == nullptr
        ? &causal_conv1d_update_kernel<Ktraits, false>  // Non-circular buffer
        : &causal_conv1d_update_kernel<Ktraits, true>;  // Circular buffer

    // Launch kernel
    hipLaunchKernelGGL(kernel, grid, Ktraits::kNThreads, 0, stream, params);
}

// Dispatch based on convolution width
template<typename input_t, typename weight_t>
void causal_conv1d_update_dispatch(ConvParamsBaseUpdate &params, hipStream_t stream) {
    constexpr int kNThreads = 64;  // Optimized for AMD wavefront size
    
    if (params.width == 2) {
        causal_conv1d_update_launch<kNThreads, 2, input_t, weight_t>(params, stream);
    } else if (params.width == 3) {
        causal_conv1d_update_launch<kNThreads, 3, input_t, weight_t>(params, stream);
    } else if (params.width == 4) {
        causal_conv1d_update_launch<kNThreads, 4, input_t, weight_t>(params, stream);
    }
}

// ============================================================================
// PyTorch Interface
// ============================================================================
// Main entry point for Python/PyTorch
// Handles tensor validation, parameter setup, and kernel dispatch

void causal_conv1d_update(
    torch::Tensor& x,                          // [batch, dim, seqlen] - new input (typically seqlen=1 for decoding)
    torch::Tensor& conv_state,                 // [batch, dim, state_len] - state buffer (updated in-place)
    const torch::Tensor& weight,               // [dim, width] - convolution weights
    const torch::Tensor& bias,                 // [dim] - bias (or empty)
    torch::Tensor& out,                        // [batch, dim, seqlen] - output
    bool use_silu,                             // Whether to apply SiLU activation
    const torch::Tensor& cache_seqlens,        // [batch] - for circular buffer mode (or empty)
    const torch::Tensor& conv_state_indices,   // [batch] - for continuous batching (or empty)
    int pad_slot_id)                           // Padding slot ID (-1 = no padding)
{
    CHECK_INPUT(x);
    CHECK_INPUT(conv_state);
    CHECK_INPUT(weight);
    CHECK_INPUT(out);

    // Extract dimensions
    const int32_t batch = x.size(0);
    const int32_t dim = x.size(1);
    const int32_t seqlen = x.size(2);
    const int32_t width = weight.size(1);
    const int32_t conv_state_len = conv_state.size(2);

    // Validate tensor shapes
    TORCH_CHECK(conv_state.size(0) == batch || conv_state_indices.defined(), "conv_state batch mismatch");
    TORCH_CHECK(conv_state.size(1) == dim, "conv_state dim mismatch");
    TORCH_CHECK(conv_state_len >= width - 1, "conv_state_len must be >= width - 1");
    TORCH_CHECK(out.size(0) == batch && out.size(1) == dim && out.size(2) == seqlen, "Output shape mismatch");
    TORCH_CHECK(weight.size(0) == dim, "Weight shape mismatch");
    TORCH_CHECK(width >= 2 && width <= 4, "Width must be 2, 3, or 4");

    // Setup kernel parameters
    // Setup kernel parameters
    ConvParamsBaseUpdate params;
    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;
    params.silu_activation = use_silu;

    // Input tensor strides
    params.x_batch_stride = x.stride(0);
    params.x_c_stride = x.stride(1);
    params.x_l_stride = x.stride(2);

    // Weight tensor strides
    params.weight_c_stride = weight.stride(0);
    params.weight_width_stride = weight.stride(1);

    // Output tensor strides
    params.out_batch_stride = out.stride(0);
    params.out_c_stride = out.stride(1);
    params.out_l_stride = out.stride(2);

    // Conv state dimensions and strides
    params.conv_state_len = conv_state_len;
    params.conv_state_batch_stride = conv_state.stride(0);
    params.conv_state_c_stride = conv_state.stride(1);
    params.conv_state_l_stride = conv_state.stride(2);

    // Data pointers
    params.x_ptr = x.data_ptr();
    params.weight_ptr = weight.data_ptr();
    params.out_ptr = out.data_ptr();
    params.conv_state_ptr = conv_state.data_ptr();

    // Optional bias
    if(bias.defined() && bias.numel() > 0)
    {
        CHECK_INPUT(bias);
        TORCH_CHECK(bias.size(0) == dim, "Bias shape mismatch");
        params.bias_ptr = bias.data_ptr();
    } else {
        params.bias_ptr = nullptr;
    }

    // Padding slot ID for continuous batching
    params.pad_slot_id = pad_slot_id;

    // Optional: cache_seqlens for circular buffer mode
    if (cache_seqlens.defined() && cache_seqlens.numel() > 0) {
        CHECK_INPUT(cache_seqlens);
        TORCH_CHECK(cache_seqlens.scalar_type() == torch::kInt32, "cache_seqlens must be int32");
        TORCH_CHECK(cache_seqlens.size(0) == batch, "cache_seqlens batch mismatch");
        params.cache_seqlens = cache_seqlens.data_ptr<int32_t>();
    } else {
        params.cache_seqlens = nullptr;
    }

    // Optional: conv_state_indices for continuous batching
    if (conv_state_indices.defined() && conv_state_indices.numel() > 0) {
        CHECK_INPUT(conv_state_indices);
        TORCH_CHECK(conv_state_indices.scalar_type() == torch::kInt32, "conv_state_indices must be int32");
        TORCH_CHECK(conv_state_indices.size(0) == batch, "conv_state_indices batch mismatch");
        params.conv_state_indices_ptr = conv_state_indices.data_ptr<int32_t>();
    } else {
        params.conv_state_indices_ptr = nullptr;
    }

    // Get HIP device and stream
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(x));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    // Dispatch to appropriate kernel based on data types
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "causal_conv1d_update", [&] {
        DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(weight.scalar_type(), "causal_conv1d_update", [&] {
            causal_conv1d_update_dispatch<input_t, weight_t>(params, stream);
        });
    });

    // Check for kernel launch errors
    HIP_CHECK(hipGetLastError());
}

} // namespace aiter

