#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace aiter {
__global__ void ParsePhiloxCudaState(at::PhiloxCudaState arg, uint64_t* rng_state);

inline int
num_splits_heuristic_ck(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits)
{
    for(int num_splits = 1; num_splits <= max_splits; num_splits *= 2)
    {
        if(num_SMs < batch_nheads_mblocks * (num_splits * 2))
        {
            return num_splits;
        }
    }

    return max_splits;
}

inline int override_num_splits_if_necessary(
    int batch, int nhead, int max_seqlen_q, int hdim_v, float p_drop, int num_splits, int kM0 = 64)
{
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
        return num_splits;

    hipDeviceProp_t props{};
    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
        return num_splits;

    // TODO - tile size should match the TileFmhaShape, hardcode for now
    const int kN1 = hdim_v;

    const int num_m_blocks = (max_seqlen_q + kM0 - 1) / kM0;
    const int num_n_blocks = (hdim_v + kN1 - 1) / kN1;

    if(num_splits < 1 && p_drop == 0.0f)
        return num_splits_heuristic_ck(
            batch * nhead * num_m_blocks, props.multiProcessorCount, num_n_blocks, 8);

    return num_splits;
}

template<typename ARG>
inline void print_fmha_fwd_args(ARG args)
{
    printf("seqlen_q = %d\n", args.seqlen_q);
    printf("seqlen_k = %d\n", args.seqlen_k);
    printf("batch = %d\n", args.batch);
    printf("max_seqlen_q = %d\n", args.max_seqlen_q);
    printf("hdim_q = %d\n", args.hdim_q);
    printf("hdim_v = %d\n", args.hdim_v);
    printf("nhead_q = %d\n", args.nhead_q);
    printf("nhead_k = %d\n", args.nhead_k);
    printf("scale_s = %f\n", args.scale_s);
    printf("scale_p = %f\n", args.scale_p);
    printf("scale_o = %f\n", args.scale_o);
    printf("stride_q = %d\n", args.stride_q);
    printf("stride_k = %d\n", args.stride_k);
    printf("stride_v = %d\n", args.stride_v);
    printf("stride_bias = %d\n", args.stride_bias);
    printf("stride_randval = %d\n", args.stride_randval);
    printf("stride_o = %d\n", args.stride_o);
    printf("nhead_stride_q = %d\n", args.nhead_stride_q);
    printf("nhead_stride_k = %d\n", args.nhead_stride_k);
    printf("nhead_stride_v = %d\n", args.nhead_stride_v);
    printf("nhead_stride_bias = %d\n", args.nhead_stride_bias);
    printf("nhead_stride_randval = %d\n", args.nhead_stride_randval);
    printf("nhead_stride_lse = %d\n", args.nhead_stride_lse);
    printf("nhead_stride_o = %d\n", args.nhead_stride_o);
    printf("batch_stride_q = %d\n", args.batch_stride_q);
    printf("batch_stride_k = %d\n", args.batch_stride_k);
    printf("batch_stride_v = %d\n", args.batch_stride_v);
    printf("batch_stride_bias = %d\n", args.batch_stride_bias);
    printf("batch_stride_randval = %d\n", args.batch_stride_randval);
    printf("batch_stride_lse = %d\n", args.batch_stride_lse);
    printf("batch_stride_o = %d\n", args.batch_stride_o);
    printf("window_size_left = %d\n", args.window_size_left);
    printf("window_size_right = %d\n", args.window_size_right);
    printf("mask_type = %d\n", args.mask_type);
    printf("p_drop = %f\n", args.p_drop);
    printf("s_randval = %d\n", args.s_randval);
}

} // namespace aiter