// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "asm_pa_configs.hpp"
#include "py_itfs_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

struct __attribute__((packed)) KernelArgs
{
    void* ptr_O;
    p2 _p0;
    void* ptr_Q;
    p2 _p1;
    void* ptr_K;
    p2 _p2;
    void* ptr_V;
    p2 _p3;
    void* ptr_BT;
    p2 _p4;
    void* ptr_CL;
    p2 _p5;
    void* ptr_KQ;
    p2 _p6;
    void* ptr_VQ;
    p2 _p7;
    float sclg2e;
    p3 _p12;
    unsigned int mblk;
    p3 _p13;
    unsigned int kv_nheads;
    p3 _p14;
    unsigned int Qs;
    p3 _p15;
    unsigned int Bs;
    p3 _p16;
    unsigned int KVs;
    p3 _p17;
    unsigned int GQA;
    p3 _p18;
    void* ptr_QTP;
    p2 _p19;
};

std::string get_heuristic_kernel(
    std::string q_type, std::string kv_type, int gqa, int mtp, int msk, int hp, CFG* cfgs)
{
    for(const auto& el : *cfgs)
    {
        const auto& cfg = el.second;
        if(cfg.q_type == q_type && cfg.kv_type == kv_type && cfg.gqa == gqa && cfg.mtp == mtp &&
           cfg.msk == msk && cfg.hp == hp)
        {
            return el.first;
        }
    }
    TORCH_CHECK(false,
                __func__,
                ": cannot get heuristic kernel!"
                " q_type:",
                q_type,
                " kv_type:",
                kv_type,
                " gqa:",
                gqa,
                " mtp:",
                mtp,
                " msk:",
                msk,
                " hp:",
                hp);
    return "";
}
const float f_log2E = log2f(expf(1));

torch::Tensor pa_fwd(torch::Tensor& Q, //   [num_seqs, num_heads, head_size]
                     torch::Tensor& K, //   [num_blocks, num_kv_heads, head_size/x, block_size, x]
                     torch::Tensor& V, //   [num_blocks, num_kv_heads, block_size/X, head_size, X]
                     torch::Tensor& block_tables, //   [num_seqs, max_num_blocks_per_seq]
                     torch::Tensor& context_lens, //   [num_seqs]
                     int max_num_blocks,
                     std::optional<torch::Tensor> K_QScale  = std::nullopt,
                     std::optional<torch::Tensor> V_QScale  = std::nullopt,
                     std::optional<torch::Tensor> out_      = std::nullopt,
                     std::optional<torch::Tensor> qo_indptr = std::nullopt,
                     std::string kernelName                 = "",
                     std::optional<int> high_precision      = 1,
                     std::optional<int> max_seqlen_q        = 1)
{
    torch::Tensor output = out_.value_or(torch::empty_like(Q));
    int batch            = context_lens.size(0);
    // int max_num_blocks = block_tables.size(1);
    int num_heads       = Q.size(1);
    int head_size       = Q.size(2);
    int num_kv_heads    = K.size(1);
    int block_size      = K.size(3);
    const int gqa_ratio = num_heads / num_kv_heads;
    TORCH_CHECK(block_size == 16, __func__, " for now only support block_size == 16");

    int dim            = head_size;
    int stride_Q       = Q.stride(0) * Q.itemsize();
    int stride_KV_head = block_size * dim * K.itemsize();
    int stride_KV_blk  = stride_KV_head * num_kv_heads;
    float k_log2e      = f_log2E;
    float k_scalar     = sqrt(dim);
    k_scalar           = (float)((double)k_log2e / (double)k_scalar);

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_O      = output.data_ptr();
    args.ptr_Q      = Q.data_ptr();
    args.ptr_K      = K.data_ptr();
    args.ptr_V      = V.data_ptr();
    args.ptr_BT     = block_tables.data_ptr();
    args.ptr_CL     = context_lens.data_ptr();
    if(K_QScale)
    {
        args.ptr_KQ = K_QScale.value().data_ptr();
        args.ptr_VQ = V_QScale.value().data_ptr();
    }
    else
    {
        args.ptr_KQ = nullptr;
        args.ptr_VQ = nullptr;
    }
    args.sclg2e    = k_scalar;
    args.mblk      = max_num_blocks;
    args.kv_nheads = num_kv_heads;
    args.Qs        = stride_Q;
    args.Bs        = stride_KV_blk;
    args.KVs       = stride_KV_head;
    args.GQA       = gqa_ratio;
    args.ptr_QTP   = qo_indptr ? qo_indptr.value().data_ptr() : nullptr;
    // std::cout << "sclg2e: " << args.sclg2e << " mblk:" << args.mblk << "
    // kv_nheads:" << args.kv_nheads << " Qs:" << args.Qs << " Bs:" << args.Bs <<
    // " KVs:" << args.KVs << std::endl;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    std::string q_type;
    std::string kv_type;
    int gqa;
    int mtp;
    int msk;
    int hp;
    // 1. "q_type"
    if(Q.dtype() == at::ScalarType::Half)
        q_type = "fp16";
    else if(Q.dtype() == at::ScalarType::BFloat16)
        q_type = "bf16";
    else
        TORCH_CHECK(false, __func__, ": unsupport Q dtype:", Q.scalar_type());

    // 2. "kv_type"
    if(K.dtype() == at::ScalarType::Half)
        kv_type = "fp16";
    else if(K.dtype() == at::ScalarType::BFloat16)
        kv_type = "bf16";
    else if(K.dtype() == at::ScalarType::Byte || K.dtype() == at::ScalarType::Char) //?
        kv_type = "int8";
    else if(K.dtype() == torch_fp8)
        kv_type = "fp8";
    else
        TORCH_CHECK(false, __func__, ": unsupport K dtype:", K.scalar_type());

    // 3. "gqa_ratio"
    gqa = (gqa_ratio <= 8) ? 8 : 16;

    // 4. "mtp" , 5. "mask"
    if(qo_indptr && max_seqlen_q > 1)
    {
        mtp = 1;
        msk = 1;
    }
    else
    {
        mtp = 0;
        msk = 0;
    }
    // 6. "high_precision" , 7. "ultra_precision"
    switch(high_precision.value())
    {
    case 1: hp = 1; break;
    case 2: hp = 2; break;
    default: hp = 0; break;
    };

    CFG* config_map = &cfg_pa_asm; // only one config csv in hsa/<arch>/pa, now
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;

    kernelName = get_heuristic_kernel(q_type, kv_type, gqa, mtp, msk, hp, config_map);
    if(kernelName.empty())
    {
        TORCH_CHECK(false, __func__, "not supported this kernel now! ");
    }

    AiterAsmKernel* impl_ptr = nullptr;

    auto it = config_map->find(kernelName);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.name.c_str();
        const char* co_name = cfg.co_name.c_str();
        auto result         = impl_ptr_map.emplace(name, nullptr);
        if(result.second)
        {
            result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
        }
        impl_ptr = result.first->second.get();
    }
    else
        TORCH_CHECK(false, __func__, " not find kernel " + kernelName);

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             num_kv_heads, // gdx
                             batch,        // gdy
                             1,            // gdz
                             256,          // bdx: 4 wv64
                             1,            // bdy
                             1,            // bdz
                             stream});
    return output;
}
