// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Tri Dao.
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// Adapted for AMD MI308 GPU (ROCm/HIP) and integrated into AIter framework

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

// ============================================================================
// ConvParamsBase structure (matching causal_conv1d.h)
// ============================================================================

struct ConvParamsBaseUpdate {
    using index_t = uint32_t;

    int batch, dim, seqlen, width;
    bool silu_activation;

    index_t x_batch_stride;
    index_t x_c_stride;
    index_t x_l_stride;
    index_t weight_c_stride;
    index_t weight_width_stride;
    index_t out_batch_stride;
    index_t out_c_stride;
    index_t out_l_stride;

    int conv_state_len;
    index_t conv_state_batch_stride;
    index_t conv_state_c_stride;
    index_t conv_state_l_stride;

    // Common data pointers
    void *__restrict__ x_ptr;
    void *__restrict__ weight_ptr;
    void *__restrict__ bias_ptr;
    void *__restrict__ out_ptr;

    void *__restrict__ conv_state_ptr;
    int32_t *__restrict__ cache_seqlens;

    // For continuous batching
    int32_t *__restrict__ conv_state_indices_ptr;
};

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// ============================================================================
// Kernel Traits
// ============================================================================

template<int kNThreads_, int kWidth_, typename input_t_, typename weight_t_>
struct Causal_conv1d_update_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4, "Only 2-byte or 4-byte types supported");
};

// ============================================================================
// Update Kernel
// ============================================================================

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float silu_activation(float x) {
    return x / (1.0f + expf(-x));
}

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

    // Input and output pointers for this batch and channel
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + channel_id * params.x_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + channel_id * params.out_c_stride;

    // Handle continuous batching: gather conv state from potentially non-contiguous locations
    const int conv_state_batch_coord = params.conv_state_indices_ptr == nullptr
        ? batch_id
        : params.conv_state_indices_ptr[batch_id];

    // Skip padding tokens (negative indices indicate padding)
    if (conv_state_batch_coord < 0) {
        #pragma unroll 2
        for (int i = 0; i < params.seqlen; ++i) {
            out[i * params.out_l_stride] = input_t(0.f);
        }
        return;
    }

    // Conv state pointer for this channel
    input_t *conv_state = reinterpret_cast<input_t *>(params.conv_state_ptr)
        + conv_state_batch_coord * params.conv_state_batch_stride
        + channel_id * params.conv_state_c_stride;
    
    // Weight and bias for this channel
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);

    // State management variables
    int state_len = params.conv_state_len;
    int advance_len = params.seqlen;
    int cache_seqlen = kIsCircularBuffer ? params.cache_seqlens[batch_id] % state_len : 0;
    int update_idx = cache_seqlen - (kWidth - 1);
    update_idx = update_idx < 0 ? update_idx + state_len : update_idx;

    // Load weights into registers (AMD MI308: fast register file access)
    float weight_vals[kWidth];
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { 
        weight_vals[i] = float(weight[i * params.weight_width_stride]); 
    }

    // Sliding window buffer for input values
    float x_vals[kWidth];
    
    // Initialize x_vals with zeros
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) {
        x_vals[i] = 0.0f;
    }

    // Mode A: Non-circular buffer (shift data in conv_state)
    if constexpr (!kIsCircularBuffer) {
        // Shift old data to make room for new data
        // AMD MI308: optimize for coalesced memory access
        #pragma unroll 2
        for (int i = 0; i < state_len - advance_len - (kWidth - 1); ++i) {
            conv_state[i * params.conv_state_l_stride] = conv_state[(i + advance_len) * params.conv_state_l_stride];
        }
        
        // Load the most recent kWidth-1 historical states into x_vals
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) {
            input_t state_val = conv_state[(state_len - (kWidth - 1) + i) * params.conv_state_l_stride];
            // Update conv_state with shifted data
            if (i < advance_len + (kWidth - 1) && state_len - advance_len - (kWidth - 1) + i >= 0) {
                conv_state[(state_len - advance_len - (kWidth - 1) + i) * params.conv_state_l_stride] = state_val;
            }
            x_vals[i] = float(state_val);
        }
    } 
    // Mode B: Circular buffer (only update index, no data movement)
    else {
        // Load kWidth-1 historical values in circular order
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) {
            input_t state_val = conv_state[update_idx * params.conv_state_l_stride];
            x_vals[i] = float(state_val);
            // Circular index increment
            update_idx = update_idx + 1 >= state_len ? update_idx + 1 - state_len : update_idx + 1;
        }
    }

    // Main convolution loop: process each new input token
    // AMD MI308: optimize with instruction-level parallelism
    #pragma unroll 2
    for (int i = 0; i < params.seqlen; ++i) {
        // Read new input
        input_t x_val = x[i * params.x_l_stride];
        
        // Update conv_state with new input
        if constexpr (!kIsCircularBuffer) {
            // Non-circular: write to the end of the buffer
            if (i < advance_len && state_len - advance_len + i >= 0) {
                conv_state[(state_len - advance_len + i) * params.conv_state_l_stride] = x_val;
            }
        } else {
            // Circular: write at current index and advance
            conv_state[update_idx * params.conv_state_l_stride] = x_val;
            ++update_idx;
            update_idx = update_idx >= state_len ? update_idx - state_len : update_idx;
        }
        
        // Add new input to the sliding window
        x_vals[kWidth - 1] = float(x_val);
        
        // Compute convolution output
        // AMD MI308: FMA (fused multiply-add) optimization
        float out_val = bias_val;
        #pragma unroll
        for (int j = 0; j < kWidth; ++j) { 
            out_val += weight_vals[j] * x_vals[j]; 
        }
        
        // Apply SiLU activation if requested
        if (params.silu_activation) { 
            out_val = silu(out_val); 
        }
        
        // Write output
        out[i * params.out_l_stride] = input_t(out_val);
        
        // Shift sliding window left by 1 position
        #pragma unroll
        for (int k = 0; k < kWidth - 1; ++k) { 
            x_vals[k] = x_vals[k + 1]; 
        }
    }
}


// template<typename Ktraits, bool kIsCircularBuffer>
// __global__ __launch_bounds__(Ktraits::kNThreads)
// void causal_conv1d_update_kernel(ConvParamsBaseUpdate params) {
//     constexpr int kWidth = Ktraits::kWidth;
//     constexpr int kNThreads = Ktraits::kNThreads;
//     using input_t = typename Ktraits::input_t;
//     using weight_t = typename Ktraits::weight_t;

//     const int tidx = threadIdx.x;
//     const int batch_id = blockIdx.x;
//     const int channel_id = blockIdx.y * kNThreads + tidx;

//     // Early exit for out-of-bounds channels
//     if (channel_id >= params.dim) return;

//     // Input and output pointers for this batch and channel
//     input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
//         + channel_id * params.x_c_stride;
//     input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
//         + channel_id * params.out_c_stride;

//     // Handle continuous batching: gather conv state from potentially non-contiguous locations
//     const int conv_state_batch_coord = params.conv_state_indices_ptr == nullptr
//         ? batch_id
//         : params.conv_state_indices_ptr[batch_id];

//     // Skip padding tokens (negative indices indicate padding)
//     if (conv_state_batch_coord < 0) {
//         #pragma unroll 2
//         for (int i = 0; i < params.seqlen; ++i) {
//             out[i * params.out_l_stride] = input_t(0.f);
//         }
//         return;
//     }

//     // Conv state pointer for this channel
//     input_t *conv_state = reinterpret_cast<input_t *>(params.conv_state_ptr)
//         + conv_state_batch_coord * params.conv_state_batch_stride
//         + channel_id * params.conv_state_c_stride;

//     // Weight and bias for this channel
//     weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
//     float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);

//     // State management variables
//     int state_len = params.conv_state_len;
//     int advance_len = params.seqlen;
//     int cache_seqlen = kIsCircularBuffer ? params.cache_seqlens[batch_id] % state_len : 0;
//     int update_idx = cache_seqlen - (kWidth - 1);
//     update_idx = update_idx < 0 ? update_idx + state_len : update_idx;

//     // Load weights into registers
//     float weight_vals[kWidth];
//     #pragma unroll
//     for (int i = 0; i < kWidth; ++i) {
//         weight_vals[i] = float(weight[i * params.weight_width_stride]);
//     }

//     // Sliding window buffer for input values
//     float x_vals[kWidth];

//     // Initialize x_vals with zeros
//     #pragma unroll
//     for (int i = 0; i < kWidth; ++i) {
//         x_vals[i] = 0.0f;
//     }

//     // Mode A: Non-circular buffer (shift data in conv_state)
//     if constexpr (!kIsCircularBuffer) {
//         // Shift old data to make room for new data
//         #pragma unroll 2
//         for (int i = 0; i < state_len - advance_len - (kWidth - 1); ++i) {
//             conv_state[i * params.conv_state_l_stride] = conv_state[(i + advance_len) * params.conv_state_l_stride];
//         }

//         // Load the most recent kWidth-1 historical states into x_vals
//         #pragma unroll
//         for (int i = 0; i < kWidth - 1; ++i) {
//             input_t state_val = conv_state[(state_len - (kWidth - 1) + i) * params.conv_state_l_stride];
//             // Update conv_state with shifted data
//             if (i < advance_len + (kWidth - 1) && state_len - advance_len - (kWidth - 1) + i >= 0) {
//                 conv_state[(state_len - advance_len - (kWidth - 1) + i) * params.conv_state_l_stride] = state_val;
//             }
//             x_vals[i] = float(state_val);
//         }
//     }
//     // Mode B: Circular buffer (only update index, no data movement)
//     else {
//         // Load kWidth-1 historical values in circular order
//         #pragma unroll
//         for (int i = 0; i < kWidth - 1; ++i) {
//             input_t state_val = conv_state[update_idx * params.conv_state_l_stride];
//             x_vals[i] = float(state_val);
//             // Circular index increment
//             update_idx = update_idx + 1 >= state_len ? update_idx + 1 - state_len : update_idx + 1;
//         }
//     }

//     // Main convolution loop: process each new input token
//     #pragma unroll 2
//     for (int i = 0; i < params.seqlen; ++i) {
//         // Read new input
//         input_t x_val = x[i * params.x_l_stride];

//         // Update conv_state with new input
//         if constexpr (!kIsCircularBuffer) {
//             // Non-circular: write to the end of the buffer
//             if (i < advance_len && state_len - advance_len + i >= 0) {
//                 conv_state[(state_len - advance_len + i) * params.conv_state_l_stride] = x_val;
//             }
//         } else {
//             // Circular: write at current index and advance
//             conv_state[update_idx * params.conv_state_l_stride] = x_val;
//             ++update_idx;
//             update_idx = update_idx >= state_len ? update_idx - state_len : update_idx;
//         }

//         // Add new input to the sliding window
//         x_vals[kWidth - 1] = float(x_val);

//         // Compute convolution output
//         float out_val = bias_val;
//         #pragma unroll
//         for (int j = 0; j < kWidth; ++j) {
//             out_val += weight_vals[j] * x_vals[j];
//         }

//         // Apply SiLU activation if requested
//         if (params.silu_activation) {
//             out_val = silu_activation(out_val);
//         }

//         // Write output
//         out[i * params.out_l_stride] = input_t(out_val);

//         // Shift sliding window left by 1 position
//         #pragma unroll
//         for (int k = 0; k < kWidth - 1; ++k) {
//             x_vals[k] = x_vals[k + 1];
//         }
//     }
// }

// ============================================================================
// Launch Functions
// ============================================================================

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_update_launch(ConvParamsBaseUpdate &params, hipStream_t stream) {
    using Ktraits = Causal_conv1d_update_kernel_traits<kNThreads, kWidth, input_t, weight_t>;

    // Grid configuration
    dim3 grid(params.batch, (params.dim + kNThreads - 1) / kNThreads);

    // Select kernel based on whether circular buffer is used
    auto kernel = params.cache_seqlens == nullptr
        ? &causal_conv1d_update_kernel<Ktraits, false>   // Non-circular mode
        : &causal_conv1d_update_kernel<Ktraits, true>;   // Circular buffer mode

    // Launch kernel
    hipLaunchKernelGGL(kernel, grid, Ktraits::kNThreads, 0, stream, params);
}

template<typename input_t, typename weight_t>
void causal_conv1d_update_dispatch(ConvParamsBaseUpdate &params, hipStream_t stream) {
    // Dispatch based on convolution width
    constexpr int kNThreads = 64;  // AMD MI308 wavefront size
    
    switch (params.width) {
        case 2:
            causal_conv1d_update_launch<kNThreads, 2, input_t, weight_t>(params, stream);
            break;
        case 3:
            causal_conv1d_update_launch<kNThreads, 3, input_t, weight_t>(params, stream);
            break;
        case 4:
            causal_conv1d_update_launch<kNThreads, 4, input_t, weight_t>(params, stream);
            break;
        default:
            TORCH_CHECK(false, "Unsupported width. Only 2, 3, 4 are supported.");
    }
}

// ============================================================================
// PyTorch Interface
// ============================================================================

void causal_conv1d_update(
    torch::Tensor& x,              // [batch, dim, seqlen] - new input (typically seqlen=1)
    torch::Tensor& conv_state,     // [batch, dim, state_len] - state buffer (will be updated in-place)
    const torch::Tensor& weight,   // [dim, width]
    const torch::Tensor& bias,     // [dim] or empty
    torch::Tensor& out,            // [batch, dim, seqlen] - output
    bool use_silu,
    const torch::Tensor& cache_seqlens,      // [batch] - optional, for circular buffer
    const torch::Tensor& conv_state_indices) // [batch] - optional, for continuous batching
{
    CHECK_INPUT(x);
    CHECK_INPUT(conv_state);
    CHECK_INPUT(weight);
    CHECK_INPUT(out);

    const int32_t batch = x.size(0);
    const int32_t dim = x.size(1);
    const int32_t seqlen = x.size(2);
    const int32_t width = weight.size(1);
    const int32_t conv_state_len = conv_state.size(2);

    TORCH_CHECK(conv_state.size(0) == batch || conv_state_indices.defined(), "conv_state batch mismatch");
    TORCH_CHECK(conv_state.size(1) == dim, "conv_state dim mismatch");
    TORCH_CHECK(conv_state_len >= width - 1, "conv_state_len must be >= width - 1");
    TORCH_CHECK(out.size(0) == batch && out.size(1) == dim && out.size(2) == seqlen, "Output shape mismatch");
    TORCH_CHECK(weight.size(0) == dim, "Weight shape mismatch");
    TORCH_CHECK(width >= 2 && width <= 4, "Width must be 2, 3, or 4");

    // Setup parameters
    ConvParamsBaseUpdate params;
    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;
    params.silu_activation = use_silu;

    // Strides
    params.x_batch_stride = x.stride(0);
    params.x_c_stride = x.stride(1);
    params.x_l_stride = x.stride(2);

    params.weight_c_stride = weight.stride(0);
    params.weight_width_stride = weight.stride(1);

    params.out_batch_stride = out.stride(0);
    params.out_c_stride = out.stride(1);
    params.out_l_stride = out.stride(2);

    // Conv state
    params.conv_state_len = conv_state_len;
    params.conv_state_batch_stride = conv_state.stride(0);
    params.conv_state_c_stride = conv_state.stride(1);
    params.conv_state_l_stride = conv_state.stride(2);

    // Pointers
    params.x_ptr = x.data_ptr();
    params.weight_ptr = weight.data_ptr();
    params.out_ptr = out.data_ptr();
    params.conv_state_ptr = conv_state.data_ptr();

    if(bias.defined() && bias.numel() > 0)
    {
        CHECK_INPUT(bias);
        TORCH_CHECK(bias.size(0) == dim, "Bias shape mismatch");
        params.bias_ptr = bias.data_ptr();
    } else {
        params.bias_ptr = nullptr;
    }

    // Optional: cache_seqlens (for circular buffer)
    if (cache_seqlens.defined() && cache_seqlens.numel() > 0) {
        CHECK_INPUT(cache_seqlens);
        TORCH_CHECK(cache_seqlens.scalar_type() == torch::kInt32, "cache_seqlens must be int32");
        TORCH_CHECK(cache_seqlens.size(0) == batch, "cache_seqlens batch mismatch");
        params.cache_seqlens = cache_seqlens.data_ptr<int32_t>();
    } else {
        params.cache_seqlens = nullptr;
    }

    // Optional: conv_state_indices (for continuous batching)
    if (conv_state_indices.defined() && conv_state_indices.numel() > 0) {
        CHECK_INPUT(conv_state_indices);
        TORCH_CHECK(conv_state_indices.scalar_type() == torch::kInt32, "conv_state_indices must be int32");
        TORCH_CHECK(conv_state_indices.size(0) == batch, "conv_state_indices batch mismatch");
        params.conv_state_indices_ptr = conv_state_indices.data_ptr<int32_t>();
    } else {
        params.conv_state_indices_ptr = nullptr;
    }

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(x));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    // Type dispatching
    VLLM_DISPATCH_FLOATING_TYPES(x.scalar_type(), "causal_conv1d_update", [&] {
        using dtype = typename t2ck<scalar_t>::type;
        causal_conv1d_update_dispatch<dtype, dtype>(params, stream);
    });

    HIP_CHECK(hipGetLastError());
}

} // namespace aiter

