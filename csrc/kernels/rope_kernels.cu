// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <c10/cuda/CUDAGuard.h>
#include "dispatch_utils.h"

// =====================================================================================================================
// Kernel Functionalities
//

template <typename scalar_t>
__device__
void kn_rope_group_fwd(
    scalar_t* __restrict__       p_output,
    const scalar_t* __restrict__ p_input,
    const float* __restrict__    p_freqs,
    const int32_t size_h, const int32_t size_d, const int32_t size_f,
    const int32_t stride_i_h, const int32_t stride_i_d,
    const int32_t stride_o_h, const int32_t stride_o_d)
{
    const int32_t size_half_f   = size_f >> 1;
    const int32_t offset_half_f = size_half_f * stride_i_d;

    #pragma unroll
    for (int32_t did = threadIdx.x; did < size_f; did += blockDim.x)
    {
        const int32_t offset_i_d = did * stride_i_d;
        const int32_t offset_o_d = did * stride_o_d;

        float cos, sin;
        sincosf(p_freqs[did], &sin, &cos);

        #pragma unroll
        for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
        {
            const int32_t offset_i = hid * stride_i_h + offset_i_d;
            const int32_t offset_o = hid * stride_o_h + offset_o_d;

            const float input = float(p_input[offset_i]);
            const float input_rotate =
                (did < size_half_f) ? float(-p_input[offset_i + offset_half_f]):
                                      float( p_input[offset_i - offset_half_f]);

            p_output[offset_o] = scalar_t(input * cos + input_rotate * sin);
        }
    }

    // the rest are just forwarded
    if (size_d > size_f)
    {
        #pragma unroll
        for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
        {
            const int32_t offset_i = hid * stride_i_h;
            const int32_t offset_o = hid * stride_o_h;

            #pragma unroll
            for (int32_t did = threadIdx.x + size_f; did < size_d; did += blockDim.x)
            {
                p_output[offset_o + did * stride_o_d] = p_input[offset_i + did * stride_i_d];
            }
        }
    }
}

template <typename scalar_t>
__device__
void kn_rope_group_bwd(
    scalar_t* __restrict__       p_input_grads,
    const scalar_t* __restrict__ p_output_grads,
    const float* __restrict__    p_freqs,
    const int32_t size_h, const int32_t size_d, const int32_t size_f,
    const int32_t stride_o_h, const int32_t stride_o_d,
    const int32_t stride_i_h, const int32_t stride_i_d)
{
    const int32_t size_half_f   = size_f >> 1;
    const int32_t offset_half_f = size_half_f * stride_i_d;

    #pragma unroll
    for (int32_t did = threadIdx.x; did < size_f; did += blockDim.x)
    {
        const int32_t offset_o_d = did * stride_o_d;
        const int32_t offset_i_d = did * stride_i_d;

        const float cos = cosf(p_freqs[did]);
        const float sin =
            (did < size_half_f) ? sinf(p_freqs[did + size_half_f]) :
                                 -sinf(p_freqs[did - size_half_f]);

        #pragma unroll
        for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
        {
            const int32_t offset_o = hid * stride_o_h + offset_o_d;
            const int32_t offset_i = hid * stride_i_h + offset_i_d;

            const float output_grad = float(p_output_grads[offset_o]);
            const float output_grad_rotate =
                (did < size_half_f) ? float(p_output_grads[offset_o + offset_half_f]):
                                      float(p_output_grads[offset_o - offset_half_f]);

            p_input_grads[offset_i] = scalar_t(output_grad * cos + output_grad_rotate * sin);
        }
    }

    // the rest are just forwarded
    if (size_d > size_f)
    {
        #pragma unroll
        for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
        {
            const int32_t offset_o = hid * stride_o_h;
            const int32_t offset_i = hid * stride_i_h;

            #pragma unroll
            for (int32_t did = threadIdx.x + size_f; did < size_d; did += blockDim.x)
            {
                p_input_grads[offset_i + did * stride_i_d] = p_output_grads[offset_o + did * stride_o_d];
            }
        }
    }
}

// =====================================================================================================================
// Kernel Entries
//

template <typename scalar_t>
__global__
void kn_rope_fwd(
    scalar_t* __restrict__       p_output,
    const scalar_t* __restrict__ p_input,
    const float* __restrict__    p_freqs,
    const int32_t size_s, const int32_t size_b, const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_i_s, const int32_t stride_i_b, const int32_t stride_i_h, const int32_t stride_i_d,
    const int32_t stride_o_s, const int32_t stride_o_b, const int32_t stride_o_h, const int32_t stride_o_d)
{
    const int32_t seq_idx   = blockIdx.x;
    const int32_t batch_idx = blockIdx.y;
    const int32_t seq_i_idx = seq_idx * stride_i_s + batch_idx * stride_i_b;
    const int32_t seq_o_idx = seq_idx * stride_o_s + batch_idx * stride_o_b;
    const int32_t seq_f_idx = seq_idx * size_f;

    kn_rope_group_fwd(
        p_output + seq_o_idx,
        p_input + seq_i_idx,
        p_freqs + seq_f_idx,
        size_h, size_d, size_f,
        stride_i_h, stride_i_d,
        stride_o_h, stride_o_d);
}

template <typename scalar_t>
__global__
void kn_rope_bwd(
    scalar_t* __restrict__       p_input_grads,
    const scalar_t* __restrict__ p_output_grads,
    const float* __restrict__    p_freqs,
    const int32_t size_s, const int32_t size_b, const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_o_s, const int32_t stride_o_b, const int32_t stride_o_h, const int32_t stride_o_d,
    const int32_t stride_i_s, const int32_t stride_i_b, const int32_t stride_i_h, const int32_t stride_i_d)
{
    const int32_t seq_idx   = blockIdx.x;
    const int32_t batch_idx = blockIdx.y;
    const int32_t seq_o_idx = seq_idx * stride_o_s + batch_idx * stride_o_b;
    const int32_t seq_i_idx = seq_idx * stride_i_s + batch_idx * stride_i_b;
    const int32_t seq_f_idx = seq_idx * size_f;

    kn_rope_group_bwd(
        p_input_grads + seq_i_idx,
        p_output_grads + seq_o_idx,
        p_freqs + seq_f_idx,
        size_h, size_d, size_f,
        stride_o_h, stride_o_d,
        stride_i_h, stride_i_d);
}

// =====================================================================================================================
// Dispatches
//

template <typename scalar_t>
void dispatch_rope_fwd(
    scalar_t* __restrict__       p_output,
    const scalar_t* __restrict__ p_input,
    const float* __restrict__    p_freqs,
    const int32_t size_s, const int32_t size_b, const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_i_s, const int32_t stride_i_b, const int32_t stride_i_h, const int32_t stride_i_d,
    const int32_t stride_o_s, const int32_t stride_o_b, const int32_t stride_o_h, const int32_t stride_o_d)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 block(size_s, size_b);
    const dim3 grid(C10_WARP_SIZE, size_h < 16 ? 4 : 8);

    kn_rope_fwd<<<grid, block, 0, stream>>>(
        p_output,
        p_input,
        p_freqs,
        size_s, size_b, size_h, size_d,
        size_f,
        stride_i_s, stride_i_b, stride_i_h, stride_i_d,
        stride_o_s, stride_o_b, stride_o_h, stride_o_d);
}

template <typename scalar_t>
void dispatch_rope_bwd(
    scalar_t* __restrict__       p_input_grads,
    const scalar_t* __restrict__ p_output_grads,
    const float* __restrict__    p_freqs,
    const int32_t size_s, const int32_t size_b, const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_o_s, const int32_t stride_o_b, const int32_t stride_o_h, const int32_t stride_o_d,
    const int32_t stride_i_s, const int32_t stride_i_b, const int32_t stride_i_h, const int32_t stride_i_d)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 block(size_s, size_b);
    const dim3 grid(C10_WARP_SIZE, size_h < 16 ? 4 : 8);

    kn_rope_bwd<<<grid, block, 0, stream>>>(
        p_input_grads,
        p_output_grads,
        p_freqs,
        size_s, size_b, size_h, size_d,
        size_f,
        stride_o_s, stride_o_b, stride_o_h, stride_o_d,
        stride_i_s, stride_i_b, stride_i_h, stride_i_d);
}

// =====================================================================================================================
// Interfaces
//

void rope_fwd(
    torch::Tensor&       output,        // [s, b, h, d]
    const torch::Tensor& input,         // [s, b, h, d]
    const torch::Tensor& freqs)         // [s, 1, 1, d]
{
    // Get sizes of input and output
    const int32_t size_s = input.size(0);
    const int32_t size_b = input.size(1);
    const int32_t size_h = input.size(2);
    const int32_t size_d = input.size(3);
    const int32_t size_f = freqs.size(3);
    // Get strides of input
    const int32_t stride_i_s = input.stride(0);
    const int32_t stride_i_b = input.stride(1);
    const int32_t stride_i_h = input.stride(2);
    const int32_t stride_i_d = input.stride(3);
    // Get strides of output
    const int32_t stride_o_s = output.stride(0);
    const int32_t stride_o_b = output.stride(1);
    const int32_t stride_o_h = output.stride(2);
    const int32_t stride_o_d = output.stride(3);

    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "kn_rope_fwd",
        [&] {
            dispatch_rope_fwd<scalar_t>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                freqs.data_ptr<float>(),
                size_s, size_b, size_h, size_d,
                size_f, // size of last dimension of freqs.
                stride_i_s, stride_i_b, stride_i_h, stride_i_d,
                stride_o_s, stride_o_b, stride_o_h, stride_o_d);
        });
}

void rope_bwd(
    torch::Tensor&       input_grads,   // [s, b, h, d]
    const torch::Tensor& output_grads,  // [s, b, h, d]
    const torch::Tensor& freqs)         // [s, 1, 1, d]
{
    // Get sizes of input and output
    const int32_t size_s = output_grads.size(0);
    const int32_t size_b = output_grads.size(1);
    const int32_t size_h = output_grads.size(2);
    const int32_t size_d = output_grads.size(3);
    const int32_t size_f = freqs.size(3);
    // Get strides of output_grads
    const int32_t stride_o_s = output_grads.stride(0);
    const int32_t stride_o_b = output_grads.stride(1);
    const int32_t stride_o_h = output_grads.stride(2);
    const int32_t stride_o_d = output_grads.stride(3);
    // Get strides of input_grads
    const int32_t stride_i_s = input_grads.stride(0);
    const int32_t stride_i_b = input_grads.stride(1);
    const int32_t stride_i_h = input_grads.stride(2);
    const int32_t stride_i_d = input_grads.stride(3);

    VLLM_DISPATCH_FLOATING_TYPES(
        output_grads.scalar_type(),
        "kn_rope_bwd",
        [&] {
            dispatch_rope_bwd<scalar_t>(
                input_grads.data_ptr<scalar_t>(),
                output_grads.data_ptr<scalar_t>(),
                freqs.data_ptr<float>(),
                size_s, size_b, size_h, size_d,
                size_f, // size of last dimension of freqs.
                stride_o_s, stride_o_b, stride_o_h, stride_o_d,
                stride_i_s, stride_i_b, stride_i_h, stride_i_d);
        });
}

void rope_cached_fwd(
    torch::Tensor&       output,        // [s, b, h, d]
    const torch::Tensor& input,         // [s, b, h, d]
    const torch::Tensor& cos,           // [s, 1, 1, d]
    const torch::Tensor& sin)           // [s, 1, 1, d]
{

}

void rope_cached_bwd(
    torch::Tensor&       input_grads,   // [s, b, h, d]
    const torch::Tensor& output_grads,  // [s, b, h, d]
    const torch::Tensor& cos,           // [s, 1, 1, d]
    const torch::Tensor& sin)           // [s, 1, 1, d]
{

}

void rope_thd_fwd(
    torch::Tensor&       output,        // [t, h, d]
    const torch::Tensor& input,         // [t, h, d]
    const torch::Tensor& cu_seqlens,    // [t]
    const torch::Tensor& freqs)         // [t, d]
{

}

void rope_thd_bwd(
    torch::Tensor&       input_grads,   // [t, h, d]
    const torch::Tensor& output_grads,  // [t, h, d]
    const torch::Tensor& cu_seqlens,    // [t]
    const torch::Tensor& freqs)         // [t, d]
{

}

void rope_2d_fwd(
    torch::Tensor&       output,
    const torch::Tensor& input,
    const torch::Tensor& cos_height,
    const torch::Tensor& sin_height,
    const torch::Tensor& cos_width,
    const torch::Tensor& sin_width)
{

}

void rope_2d_bwd(
    torch::Tensor&       input_grads,
    const torch::Tensor& output_grads,
    const torch::Tensor& cos_height,
    const torch::Tensor& sin_height,
    const torch::Tensor& cos_width,
    const torch::Tensor& sin_width)
{

}
