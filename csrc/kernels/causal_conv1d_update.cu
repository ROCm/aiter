// SPDX-License-Identifier: MIT
// Copyright (C) 2023-2026, Tri Dao.
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Causal 1D Convolution Update Kernel for AIter Framework
// Adapted for AMD MI308 GPU (ROCm/HIP)

#include "aiter_hip_common.h"
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include "causal_conv1d.h"
#include "ck_tile/core.hpp"

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
// ConvParamsBaseUpdate - Kernel Parameters Structure
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

    void *__restrict__ x_ptr;
    void *__restrict__ weight_ptr;
    void *__restrict__ bias_ptr;
    void *__restrict__ out_ptr;

    void *__restrict__ conv_state_ptr;
    int32_t *__restrict__ cache_seqlens;

    int32_t *__restrict__ conv_state_indices_ptr;
    int pad_slot_id;
};

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

    if (channel_id >= params.dim) return;

    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + channel_id * params.x_c_stride;

    const int conv_state_batch_coord = params.conv_state_indices_ptr == nullptr
        ? batch_id
        : params.conv_state_indices_ptr[batch_id];

    if (conv_state_batch_coord == params.pad_slot_id){
        return;
    }

    input_t *conv_state = reinterpret_cast<input_t *>(params.conv_state_ptr)
        + conv_state_batch_coord * params.conv_state_batch_stride
        + channel_id * params.conv_state_c_stride;

    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + channel_id * params.out_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);

    int state_len = params.conv_state_len;
    int advance_len = params.seqlen;
    int cache_seqlen = kIsCircularBuffer ? params.cache_seqlens[batch_id] % state_len : 0;
    int update_idx = cache_seqlen - (kWidth - 1);
    update_idx = update_idx < 0 ? update_idx + state_len : update_idx;

    float weight_vals[kWidth] = {0};
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { weight_vals[i] = float(weight[i * params.weight_width_stride]); }

    float x_vals[kWidth] = {0};

    if constexpr (!kIsCircularBuffer) {
        #pragma unroll 2
        for (int i = 0; i < state_len - advance_len - (kWidth - 1); ++i) {
            conv_state[i * params.conv_state_l_stride] = conv_state[(i + advance_len) * params.conv_state_l_stride];
        }

        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) {
            input_t state_val = conv_state[(state_len - (kWidth - 1) + i) * params.conv_state_l_stride];
            if (i < advance_len + (kWidth - 1) && state_len - advance_len - (kWidth - 1) + i >= 0) {
                conv_state[(state_len - advance_len - (kWidth - 1) + i) * params.conv_state_l_stride] = state_val;
            }
            x_vals[i] = float(state_val);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i, update_idx = update_idx + 1 >= state_len ? update_idx + 1 - state_len : update_idx + 1) {
            input_t state_val = conv_state[update_idx * params.conv_state_l_stride];
            x_vals[i] = float(state_val);
        }
    }

    #pragma unroll 2
    for (int i = 0; i < params.seqlen; ++i) {
        input_t x_val = x[i * params.x_l_stride];

        if constexpr (!kIsCircularBuffer) {
            if (i < advance_len && state_len - advance_len + i >= 0) {
                conv_state[(state_len - advance_len + i) * params.conv_state_l_stride] = x_val;
            }
        } else {
            conv_state[update_idx * params.conv_state_l_stride] = x_val;
            ++update_idx;
            update_idx = update_idx >= state_len ? update_idx - state_len : update_idx;
        }

        x_vals[kWidth - 1] = float(x_val);

        float out_val = bias_val;
        #pragma unroll
        for (int j = 0; j < kWidth; ++j) { out_val += weight_vals[j] * x_vals[j]; }

        if (params.silu_activation) { out_val = out_val / (1 + expf(-out_val)); }

        out[i * params.out_l_stride] = input_t(out_val);

        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) { x_vals[i] = x_vals[i + 1]; }
    }
}

// ============================================================================
// Launch Functions
// ============================================================================

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_update_launch(ConvParamsBaseUpdate &params, hipStream_t stream) {
    using Ktraits = Causal_conv1d_update_kernel_traits<kNThreads, kWidth, input_t, weight_t>;

    dim3 grid(params.batch, (params.dim + kNThreads - 1) / kNThreads);

    auto kernel = params.cache_seqlens == nullptr
        ? &causal_conv1d_update_kernel<Ktraits, false>
        : &causal_conv1d_update_kernel<Ktraits, true>;

    hipLaunchKernelGGL(kernel, grid, Ktraits::kNThreads, 0, stream, params);
}

template<typename input_t, typename weight_t>
void causal_conv1d_update_dispatch(ConvParamsBaseUpdate &params, hipStream_t stream) {
    constexpr int kNThreads = 64;

    if (params.width == 2) {
        causal_conv1d_update_launch<kNThreads, 2, input_t, weight_t>(params, stream);
    } else if (params.width == 3) {
        causal_conv1d_update_launch<kNThreads, 3, input_t, weight_t>(params, stream);
    } else if (params.width == 4) {
        causal_conv1d_update_launch<kNThreads, 4, input_t, weight_t>(params, stream);
    }
}

// ============================================================================
// Host Interface
// ============================================================================

void causal_conv1d_update(
    aiter_tensor_t& x,
    aiter_tensor_t& conv_state,
    aiter_tensor_t& weight,
    aiter_tensor_t& bias,
    aiter_tensor_t& out,
    bool use_silu,
    aiter_tensor_t& cache_seqlens,
    aiter_tensor_t& conv_state_indices,
    int pad_slot_id)
{
    const int32_t batch = x.size(0);
    const int32_t dim = x.size(1);
    const int32_t seqlen = x.size(2);
    const int32_t width = weight.size(1);
    const int32_t conv_state_len = conv_state.size(2);

    AITER_CHECK(conv_state.size(0) == batch || conv_state_indices.numel() > 0, "conv_state batch mismatch");
    AITER_CHECK(conv_state.size(1) == dim, "conv_state dim mismatch");
    AITER_CHECK(conv_state_len >= width - 1, "conv_state_len must be >= width - 1");
    AITER_CHECK(out.size(0) == batch && out.size(1) == dim && out.size(2) == seqlen, "Output shape mismatch");
    AITER_CHECK(weight.size(0) == dim, "Weight shape mismatch");
    AITER_CHECK(width >= 2 && width <= 4, "Width must be 2, 3, or 4");

    ConvParamsBaseUpdate params;
    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;
    params.silu_activation = use_silu;

    params.x_batch_stride = x.stride(0);
    params.x_c_stride = x.stride(1);
    params.x_l_stride = x.stride(2);

    params.weight_c_stride = weight.stride(0);
    params.weight_width_stride = weight.stride(1);

    params.out_batch_stride = out.stride(0);
    params.out_c_stride = out.stride(1);
    params.out_l_stride = out.stride(2);

    params.conv_state_len = conv_state_len;
    params.conv_state_batch_stride = conv_state.stride(0);
    params.conv_state_c_stride = conv_state.stride(1);
    params.conv_state_l_stride = conv_state.stride(2);

    params.x_ptr = x.data_ptr();
    params.weight_ptr = weight.data_ptr();
    params.out_ptr = out.data_ptr();
    params.conv_state_ptr = conv_state.data_ptr();

    if(bias.numel() > 0)
    {
        AITER_CHECK(bias.size(0) == dim, "Bias shape mismatch");
        params.bias_ptr = bias.data_ptr();
    } else {
        params.bias_ptr = nullptr;
    }

    params.pad_slot_id = pad_slot_id;

    if (cache_seqlens.numel() > 0) {
        AITER_CHECK(cache_seqlens.dtype() == AITER_DTYPE_i32, "cache_seqlens must be int32");
        AITER_CHECK(cache_seqlens.size(0) == batch, "cache_seqlens batch mismatch");
        params.cache_seqlens = reinterpret_cast<int32_t*>(cache_seqlens.data_ptr());
    } else {
        params.cache_seqlens = nullptr;
    }

    if (conv_state_indices.numel() > 0) {
        AITER_CHECK(conv_state_indices.dtype() == AITER_DTYPE_i32, "conv_state_indices must be int32");
        AITER_CHECK(conv_state_indices.size(0) == batch, "conv_state_indices batch mismatch");
        params.conv_state_indices_ptr = reinterpret_cast<int32_t*>(conv_state_indices.data_ptr());
    } else {
        params.conv_state_indices_ptr = nullptr;
    }

    HipDeviceGuard device_guard(x.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    if (x.dtype() == AITER_DTYPE_fp16) {
        using input_t = _Float16;
        if (weight.dtype() == AITER_DTYPE_fp16) {
            using weight_t = _Float16;
            causal_conv1d_update_dispatch<input_t, weight_t>(params, stream);
        } else if (weight.dtype() == AITER_DTYPE_bf16) {
            using weight_t = hip_bfloat16;
            causal_conv1d_update_dispatch<input_t, weight_t>(params, stream);
        } else if (weight.dtype() == AITER_DTYPE_fp32) {
            using weight_t = float;
            causal_conv1d_update_dispatch<input_t, weight_t>(params, stream);
        } else {
            AITER_CHECK(false, "causal_conv1d_update not implemented for weight type");
        }
    } else if (x.dtype() == AITER_DTYPE_bf16) {
        using input_t = hip_bfloat16;
        if (weight.dtype() == AITER_DTYPE_fp16) {
            using weight_t = _Float16;
            causal_conv1d_update_dispatch<input_t, weight_t>(params, stream);
        } else if (weight.dtype() == AITER_DTYPE_bf16) {
            using weight_t = hip_bfloat16;
            causal_conv1d_update_dispatch<input_t, weight_t>(params, stream);
        } else if (weight.dtype() == AITER_DTYPE_fp32) {
            using weight_t = float;
            causal_conv1d_update_dispatch<input_t, weight_t>(params, stream);
        } else {
            AITER_CHECK(false, "causal_conv1d_update not implemented for weight type");
        }
    } else if (x.dtype() == AITER_DTYPE_fp32) {
        using input_t = float;
        if (weight.dtype() == AITER_DTYPE_fp16) {
            using weight_t = _Float16;
            causal_conv1d_update_dispatch<input_t, weight_t>(params, stream);
        } else if (weight.dtype() == AITER_DTYPE_bf16) {
            using weight_t = hip_bfloat16;
            causal_conv1d_update_dispatch<input_t, weight_t>(params, stream);
        } else if (weight.dtype() == AITER_DTYPE_fp32) {
            using weight_t = float;
            causal_conv1d_update_dispatch<input_t, weight_t>(params, stream);
        } else {
            AITER_CHECK(false, "causal_conv1d_update not implemented for weight type");
        }
    } else {
        AITER_CHECK(false, "causal_conv1d_update not implemented for input type");
    }

    HIP_CHECK(hipGetLastError());
}

} // namespace aiter
