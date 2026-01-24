// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "asm_pa_configs.hpp"
#include "py_itfs_common.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>
#include "aiter_enum.h"

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


struct __attribute__((packed)) PsKernelArgs
{
    void* ptr_O;
    p2 _p0;
    void* ptr_Q;
    p2 _p1;
    void* ptr_K;
    p2 _p2;
    void* ptr_V;
    p2 _p3;
    void *ptr_KVIndices;
    p2 _p4;
    void *ptr_CL;
    p2 _p5;
    void *ptr_KQ;
    p2 _p6;
    void *ptr_VQ;
    p2 _p7;
    float sclg2e;
    p3 _p12;
    unsigned int kv_nheads;
    p3 _p14;
    unsigned int Qs;
    p3 _p15;
    unsigned int Bs;
    p3 _p16;
    unsigned int KVs;
    p3 _p17;
    unsigned int mtp;
    p3 _p18;
    unsigned int GQA;
    p3 _p19;
    void *ptr_QOPtr;
    p2 _p20;
    void *ptr_KVPtr;
    p2 _p21;
    void *ptr_WorkPtr;
    p2 _p22;
    void *ptr_WorkInfo;
    p2 _p23;
    void *ptr_SplitO;
    p2 _p24;
    void *ptr_SplitLSE;
    p2 _p25;
};


std::string get_heuristic_kernel(std::string q_type,
                                 std::string kv_type,
                                 int gqa,
                                 int mtp,
                                 int msk,
                                 int hp,
                                 int block_size,
                                 std::string arch_id,
                                 int ps,
                                 int qTile,
                                 int quant_type,
                                 CFG* cfgs)
{
    // # mtp * gqa <= 16
    // # gpa = 16, mtp 1
    // # qlen = mtp + 1
    // # qlen * gqa <=16

    const std::vector<int> mtp_flags = (mtp > 0) ? std::vector<int>{mtp, 1} : std::vector<int>{0};
    const std::vector<int> gqa_flags = {gqa, (gqa + 7) / 8 * 8};
    for(int mtp_ : mtp_flags)
    {
        for(int gqa_ : gqa_flags)
        {
            // find exact match
            for(const auto& el : *cfgs)
            {
                if (el.first.find(arch_id) != 0)
                    continue;
                const auto& cfg = el.second;
                // hp is just distinct from uhp
                if(cfg.qType == q_type && cfg.kvType == kv_type && cfg.Gqa == gqa_ &&
                   cfg.Mtp == mtp_ && cfg.Msk == msk && (cfg.Hp == hp || hp == 1) &&
                   cfg.blkSz == block_size && cfg.ps == ps && cfg.qTile == qTile && cfg.quant_type == quant_type)

                    return el.first;
            }
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
                hp,
                " block_size:",
                block_size,
                " ps:",
                ps,
                " qTile:",
                qTile,
                " quant_type:",
                quant_type);
    return "";
}
const float f_log2E = log2f(expf(1));

torch::Tensor pa_fwd(torch::Tensor& Q, //   [num_seqs, num_heads, head_size]
                     torch::Tensor& K, //   [num_blocks, num_kv_heads, head_size/x, block_size, x]
                     torch::Tensor& V, //   [num_blocks, num_kv_heads, block_size/X, head_size, X]
                     torch::Tensor& block_tables, //   [num_seqs, max_num_blocks_per_seq]
                     torch::Tensor& context_lens, //   [num_seqs]
                     int block_tables_stride0,
                     int max_qlen                           = 1,
                     std::optional<torch::Tensor> K_QScale  = std::nullopt,
                     std::optional<torch::Tensor> V_QScale  = std::nullopt,
                     std::optional<torch::Tensor> out_      = std::nullopt,
                     std::optional<torch::Tensor> qo_indptr = std::nullopt,
                     std::optional<int> high_precision      = 1,
                     std::optional<std::string> kernelName_ = std::nullopt)
{
    torch::Tensor output = out_.value_or(torch::empty_like(Q));
    int batch            = context_lens.size(0);
    std::string arch_id = get_gpu_arch();
    // int block_tables_stride0 = block_tables.size(1);
    int num_heads       = Q.size(1);
    int head_size       = Q.size(2);
    int num_kv_heads    = K.size(1);
    int block_size      = K.size(3);
    const int gqa_ratio = num_heads / num_kv_heads;

    int dim            = head_size;
    int stride_Q       = Q.stride(0) * Q.itemsize();
    int stride_KV_head = K.stride(1) * K.itemsize();
    int stride_KV_blk  = K.stride(0) * K.itemsize();
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
    args.mblk      = block_tables_stride0;
    args.kv_nheads = num_kv_heads;
    args.Qs        = stride_Q;
    args.Bs        = stride_KV_blk;
    args.KVs       = stride_KV_head;
    args.GQA       = gqa_ratio;
    args.ptr_QTP   = qo_indptr ? qo_indptr.value().data_ptr() : nullptr;
    // std::cout << "sclg2e: " << args.sclg2e << " mblk:" << args.mblk
    //           << " kv_nheads:" << args.kv_nheads << " Qs:" << args.Qs << " Bs:" << args.Bs
    //           << " KVs:" << args.KVs << std::endl;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(Q));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

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
    // gqa = (gqa_ratio <= 8) ? 8 : 16;

    // 4. "mtp" , 5. "mask"
    if(qo_indptr && max_qlen > 1)
    {
        mtp = max_qlen + 10; // for kernels only support qlen=3, we encode it as 3+10=13
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
    int qTile = 0;
    CFG* config_map = &cfg_pa_asm; // only one config csv in hsa/<arch>/pa, now
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    std::string kernelName = kernelName_.has_value() ? arch_id + kernelName_.value() : "";
    int ps = 0;
    if (kernelName.empty())
        kernelName = get_heuristic_kernel(q_type, kv_type, gqa_ratio, mtp, msk, hp, block_size, arch_id, ps, qTile, 0, config_map);
    if(kernelName.empty())
    {
        TORCH_CHECK(false, __func__, "not supported this kernel now! ");
    }

    AiterAsmKernel* impl_ptr = nullptr;

    auto it = config_map->find(kernelName);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.knl_name.c_str();
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

torch::Tensor pa_ps_fwd(torch::Tensor& Q, //   [num_seqs, num_heads, head_size]
    torch::Tensor& K, //   [num_blocks, num_kv_heads, head_size/x, block_size, x]
    torch::Tensor& V, //   [num_blocks, num_kv_heads, block_size/X, head_size, X]
    torch::Tensor& kv_indptr, //   [batch_size+1], kvlen prefix sum
    torch::Tensor& kv_indices, //   [sum_kvlen], packed kv ids
    torch::Tensor& context_lens, //   [batch_size]
    float softmax_scale,
    int max_qlen                           = 1,
    std::optional<torch::Tensor> K_QScale  = std::nullopt,
    std::optional<torch::Tensor> V_QScale  = std::nullopt,
    std::optional<torch::Tensor> out_      = std::nullopt,
    std::optional<torch::Tensor> qo_indptr = std::nullopt,
    std::optional<torch::Tensor> work_indptr = std::nullopt,
    std::optional<torch::Tensor> work_info = std::nullopt,
    // std::optional<torch::Tensor> work_meta_data = std::nullopt,
    std::optional<torch::Tensor> splitData = std::nullopt,
    std::optional<torch::Tensor> splitLse = std::nullopt,
    int mask                               = 0,
    std::optional<int> high_precision      = 1,
    std::optional<std::string> kernelName_ = std::nullopt,
    QuantType quant_type = QuantType::per_Token)
{
    torch::Tensor output = out_.value_or(torch::empty_like(Q));
    int batch           = qo_indptr->size(0) - 1;
    // int block_tables_stride0 = block_tables.size(1);
    int num_heads       = Q.size(1);
    int head_size       = Q.size(2);
    int num_kv_heads    = K.size(1);
    int block_size      = K.size(3);
    const int gqa_ratio = num_heads / num_kv_heads;    

    int dim            = head_size;
    int stride_Q       = Q.stride(0) * Q.itemsize();
    int stride_KV_head = K.stride(1) * K.itemsize();
    int stride_KV_blk  = K.stride(0) * K.itemsize();
    float k_log2e      = f_log2E;
    float k_scalar     = sqrt(dim);
    k_scalar           = (float)((double)k_log2e / (double)k_scalar);

    PsKernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_O      = output.data_ptr();
    args.ptr_Q      = Q.data_ptr();
    args.ptr_K      = K.data_ptr();
    args.ptr_V      = V.data_ptr();

    args.ptr_KVIndices     = kv_indices.data_ptr();
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
    args.sclg2e       = k_scalar;
    // args.mblk         = 1; // fix
    args.kv_nheads    = num_kv_heads;
    args.Qs           = stride_Q;
    args.Bs           = stride_KV_blk;
    args.KVs          = stride_KV_head;
    args.GQA          = gqa_ratio;
    args.ptr_QOPtr      = qo_indptr ? qo_indptr.value().data_ptr() : nullptr;
    args.ptr_KVPtr     = kv_indptr.data_ptr();
    // args.ptr_Metadata = work_meta_data ? work_meta_data.value().data_ptr() : nullptr;
    args.ptr_WorkPtr  = work_indptr ? work_indptr.value().data_ptr() : nullptr;
    args.ptr_WorkInfo = work_info ? work_info.value().data_ptr() : nullptr;
    args.ptr_SplitO   = work_info ? splitData.value().data_ptr() : nullptr;
    args.ptr_SplitLSE = work_info ? splitLse.value().data_ptr() : nullptr;
    args.mtp          = max_qlen - 1;

    // std::cout << "sclg2e: " << args.sclg2e << " mblk:" << args.mblk
    //           << " kv_nheads:" << args.kv_nheads << " Qs:" << args.Qs << " Bs:" << args.Bs
    //           << " KVs:" << args.KVs << std::endl;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(Q));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    std::string q_type;
    std::string kv_type;
    int gqa;
    int mtp;
    int msk;
    int hp;
    int ps = work_indptr.has_value() ? 1 : 0;
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
    // 4. "mtp" , 5. "mask"
    // We make mtp=0, gqa=0 to dispatch kernel, since we only focus on qTile
    msk = mask;
    gqa = 0;
    mtp = 0;

    // 6. "high_precision" , 7. "ultra_precision"
    switch(high_precision.value())
    {
    case 1: hp = 1; break;
    case 2: hp = 2; break;
    default: hp = 0; break;
    };
    
    // gqa_ratio * max_qlen <= qTile
    int required_qTile = gqa_ratio * max_qlen;
    std::vector<int> available_qTiles = {16, 32, 40, 48, 64};
    int qTile = -1;
    
    for (int tile : available_qTiles) {
        if (required_qTile <= tile) {
            qTile = tile;
            break;
        }
    }
    
    TORCH_CHECK(qTile != -1, 
                __func__, 
                ": required qTile (gqa_ratio * max_qlen = ", gqa_ratio, " * ", max_qlen, 
                " = ", required_qTile, 
                ") exceeds maximum available qTile. Please reduce gqa_ratio or max_qlen.");

    CFG* config_map = &cfg_pa_asm; // only one config csv in hsa/<arch>/pa, now
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    std::string arch_id = get_gpu_arch();
    std::string kernelName = kernelName_.value_or(
        get_heuristic_kernel(q_type, kv_type, gqa, mtp, msk, hp, block_size, arch_id, ps, qTile, int(quant_type), config_map));
    if(kernelName.empty())
    {
        TORCH_CHECK(false, __func__, "not supported this kernel now! ");
    }

    AiterAsmKernel* impl_ptr = nullptr;
    int gdx, gdy;

    auto it = config_map->find(kernelName);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();
        auto result         = impl_ptr_map.emplace(name, nullptr);
        if(result.second)
        {
            result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
        }
        impl_ptr = result.first->second.get();
        if(cfg.ps)
        {
            gdx = get_num_cu_func();
            gdy = 1;
        }
        else
        {
            gdx = num_kv_heads;
            gdy = batch;
        }
    }
    else
        TORCH_CHECK(false, __func__, " not find kernel " + kernelName);

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx, // gdx
                             gdy, // gdy
                             1,   // gdz
                             256, // bdx: 4 wv64
                             1,   // bdy
                             1,   // bdz
                             stream});
    return output;
}