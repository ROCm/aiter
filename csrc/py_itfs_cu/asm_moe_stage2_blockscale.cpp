// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"
#include "hip_float8.h"

struct __attribute__((packed)) KernelArgs
{
        const int NumTokens;
        const int TopK;
        const int M;
        const int N;
        const int K;
        const int StrideA;
        const int StrideB;
        const int StrideDs;
        const int StrideC;
        const int KBatch;
        const int MPadded;
        const int NPadded;
        const int KRead;
        const int KPadded;
        const int AK0;
        const int BK0;
        const int MBlock;
        const int NBlock;
        const int BN0Shuffled;
        const int BK0Shuffled;
        const void* p_sorted_token_ids;
        const void* p_sorted_expert_ids;
        const void* p_max_token_id;
        const void* p_a_grid;
        const void* p_b_grid;
        const void* p_ds_grid;
        void* p_c_grid;
        void* p_a_scale_grid;
        void* p_b_scale_grid;
};

using namespace hip_fp8_impl;
torch::Tensor moe_stage2_blockscale_asm(
    at::Tensor &inter_states,      // [M, Top, K]
    at::Tensor &w1,      // [E, N*2, K] -> [N/128, K*128]
    at::Tensor &w2, // [E , N, K]
    at::Tensor &sorted_token_ids,
    at::Tensor &sorted_expert_ids,
    at::Tensor &sorted_weights,
    at::Tensor &num_valid_ids,
    at::Tensor &out,    // [M, N]
    const int  &topk,
    at::Tensor &w2_scale, //[E, N / 128, K / 128]
    at::Tensor &a2_scale, //[M, TopK, K]
): ...
{
    constexpr int MPerBlock = 132;
    constexpr int NPerBlock = 128;
    constexpr int KPerBlock = 128;
    constexpr int AK1 = 16;
    constexpr int BK1 = 16;
    constexpr int K_Batch = 1;

    int tokens = inter_states.sze(0);
    int m = sorted_token_ids.size(0);
    int n = out.size(1);
    int k = w2.size(2);

    TORCH_CHECK(out.dtype() == torch::ScalarType::Half,
                "asm_moe_stage2_blockscale only support fp8!");
    TORCH_CHECK(n % TileN == 0 && k % TileK == 0, 
                "flatmm a8w8 blockscale asm only suuport 128x256x128 tile now!");

    KernelArgs args;
    size_t arg_size = sizeof(args);

   
    args.NumTokens           = tokens;
    args.TopK                = topk;
    args.M                   = m;
    args.N                   = n;
    args.K                   = k;
    args.StrideA             = k;
    args.StrideB             = k;
    args.StrideDs            = 0;
    args.StrideC             = n;
    args.KBatch;
    args.MPadded             = math::integer_least_multiple(M, MPerBlock);
    args.NPadded             = math::integer_least_multiple(N, NPerBlock);
    args.KRead               = [&](){
        constexpr auto KReadVec = math::lcm(AK1, BK1);
        auto K_t                = K_Batch * KReadVec;
        return (K + K_t - 1) / K_t * KReadVec;
    }();
    args.KPadded             = math::integer_divide_ceil(K, KPerBlock) * KPerBlock;
    args.AK0                 = [&](){
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * (KPerBlock / AK1Value);
    }();
    args.BK0                 = [&](){
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * (KPerBlock / BK1Value);
    }();
    args.MBlock              = math::integer_divide_ceil(M, MPerBlock);
    args.NBlock              = math::integer_divide_ceil(N, NPerBlock);
    args.BN0Shuffled         = math::integer_divide_ceil(N, NLane);
    args.BK0Shuffled         = math::integer_divide_ceil(K, KLane * KPack);
    args.p_sorted_token_ids  = (void *)sorted_token_ids.data_ptr();
    args.p_sorted_expert_ids = (void *)sorted_expert_ids.data_ptr();
    args.p_max_token_id      = (void *)num_valid_ids.data_ptr();
    args.p_a_grid            = (void *)inter_states.data_ptr();
    args.p_b_grid            = (void *)w2.data_ptr();
    args.p_ds_grid           = (void *)sorted_weights.data_ptr();
    args.p_c_grid            = nullptr;
    args.p_a_scale_grid      = (void *)a2_scale.data_ptr();
    args.p_b_scale_grid      = (void *)w2_scale.data_ptr();


    const at::cuda::OptionalCUDAGuard device_guard(device_of(inter_states));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AiterAsmKernel *impl_ptr = nullptr;
    static AiterAsmKernel impl_kenrel("moe_gemm2_xdl_fp8_block", "flatmm_uk_gfx9_f16f8_128x256x128_1x4x1_16x16x32.co");
    impl_ptr = &impl_kenrel;

    int gdx = (n + NPerBlock - 1) / NPerBlock;
    int gdy = (m + MPerBlock - 1) / MPerBlock;

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx,   // gdx
                             gdy,   // gdy
                             1,     // gdz
                             256,   // bdx: 4 wv64
                             1,     // bdy
                             1,     // bdz
                             stream});                                 

    return out;
}
