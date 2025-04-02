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

inline int num_splits_heuristic_ck(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

inline int override_num_splits_if_necessary(int batch, int nhead, int max_seqlen_q, int hdim_v, float p_drop, int num_splits)
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
    const int kM0 = 128;
    const int kN1 = hdim_v;

    const int num_m_blocks = (max_seqlen_q + kM0 - 1) / kM0;
    const int num_n_blocks = (hdim_v + kN1 - 1) / kN1;

    if(num_splits < 1 && p_drop == 0.0f)
        return num_splits_heuristic_ck(
            batch * nhead * num_m_blocks, props.multiProcessorCount * 2, num_n_blocks, 128);

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