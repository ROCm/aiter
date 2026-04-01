// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "asm_pa_configs.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>

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
    unsigned int stride_scale_blk;
    p3 _p26;
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

    AITER_CHECK(false,
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

AITER_C_ITFS
void pa_fwd(aiter_tensor_t* Q,              //   [num_seqs, num_heads, head_size]
            aiter_tensor_t* K,              //   [num_blocks, num_kv_heads, head_size/x, block_size, x]
            aiter_tensor_t* V,              //   [num_blocks, num_kv_heads, block_size/X, head_size, X]
            aiter_tensor_t* block_tables,   //   [num_seqs, max_num_blocks_per_seq]
            aiter_tensor_t* context_lens,   //   [num_seqs]
            int block_tables_stride0,
            int max_qlen,
            aiter_tensor_t* K_QScale,       //   nullable
            aiter_tensor_t* V_QScale,       //   nullable
            aiter_tensor_t* out_,           //   output tensor (pre-allocated by caller)
            aiter_tensor_t* qo_indptr,      //   nullable
            int high_precision,
            const char* kernelName_,     //   nullable
            hipStream_t stream)
{
    int batch            = context_lens->size(0);
    std::string arch_id = get_gpu_arch();
    int num_heads       = Q->size(1);
    int head_size       = Q->size(2);
    AITER_CHECK(head_size == 128,
        __func__,
        ": ASM PA only supports head_size=128, got ",
        head_size);
    int num_kv_heads    = K->size(1);
    int block_size      = K->size(3);
    const int gqa_ratio = num_heads / num_kv_heads;

    int dim            = head_size;
    int stride_Q       = Q->stride(0) * Q->element_size();
    int stride_KV_head = K->stride(1) * K->element_size();
    int stride_KV_blk  = K->stride(0) * K->element_size();
    float k_log2e      = f_log2E;
    float k_scalar     = sqrt(dim);
    k_scalar           = (float)((double)k_log2e / (double)k_scalar);

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_O      = out_->data_ptr();
    args.ptr_Q      = Q->data_ptr();
    args.ptr_K      = K->data_ptr();
    args.ptr_V      = V->data_ptr();
    args.ptr_BT     = block_tables->data_ptr();
    args.ptr_CL     = context_lens->data_ptr();
    if(K_QScale != nullptr)
    {
        args.ptr_KQ = K_QScale->data_ptr();
        args.ptr_VQ = V_QScale->data_ptr();
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
    args.ptr_QTP   = (qo_indptr != nullptr) ? qo_indptr->data_ptr() : nullptr;

    const HipDeviceGuard device_guard(Q->device_id);

    std::string q_type;
    std::string kv_type;
    int gqa;
    int mtp;
    int msk;
    int hp;
    // 1. "q_type"
    auto q_dtype = Q->dtype();
    auto kv_dtype = K->dtype();
    if(q_dtype == AITER_DTYPE_fp16)
        q_type = "fp16";
    else if(q_dtype == AITER_DTYPE_bf16)
        q_type = "bf16";
    else
        AITER_CHECK(false, __func__, ": unsupport Q dtype:", AiterDtype_to_str(q_dtype));

    // 2. "kv_type"
    if(kv_dtype == AITER_DTYPE_fp16)
        kv_type = "fp16";
    else if(kv_dtype == AITER_DTYPE_bf16)
        kv_type = "bf16";
    else if(kv_dtype == AITER_DTYPE_i8 || kv_dtype == AITER_DTYPE_u8)
        kv_type = "int8";
    else if(kv_dtype == AITER_DTYPE_fp8)
        kv_type = "fp8";
    else
        AITER_CHECK(false, __func__, ": unsupport K dtype:", AiterDtype_to_str(kv_dtype));

    if(qo_indptr != nullptr && max_qlen > 1)
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
    switch(high_precision)
    {
    case 1: hp = 1; break;
    case 2: hp = 2; break;
    default: hp = 0; break;
    };
    int qTile = 0;
    CFG* config_map = &cfg_pa_asm; // only one config csv in hsa/<arch>/pa, now
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    std::string kernelName = (kernelName_ != nullptr) ? arch_id + std::string(kernelName_) : "";
    int ps = 0;
    if (kernelName.empty())
        kernelName = get_heuristic_kernel(q_type, kv_type, gqa_ratio, mtp, msk, hp, block_size, arch_id, ps, qTile, 0, config_map);
    if(kernelName.empty())
    {
        AITER_CHECK(false, __func__, "not supported this kernel now! ");
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
        AITER_CHECK(false, __func__, " not find kernel ", kernelName);

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             num_kv_heads, // gdx
                             batch,        // gdy
                             1,            // gdz
                             256,          // bdx: 4 wv64
                             1,            // bdy
                             1,            // bdz
                             stream});
}

AITER_C_ITFS
void pa_ps_fwd(aiter_tensor_t* Q,            //   [num_seqs, num_heads, head_size]
               aiter_tensor_t* K,            //   [num_blocks, num_kv_heads, head_size/x, block_size, x]
               aiter_tensor_t* V,            //   [num_blocks, num_kv_heads, block_size/X, head_size, X]
               aiter_tensor_t* kv_indptr,    //   [batch_size+1], kvlen prefix sum
               aiter_tensor_t* kv_indices,   //   [sum_kvlen], packed kv ids
               aiter_tensor_t* context_lens, //   [batch_size]
               float softmax_scale,
               int max_qlen,
               aiter_tensor_t* K_QScale,     //   nullable
               aiter_tensor_t* V_QScale,     //   nullable
               aiter_tensor_t* out_,         //   output (pre-allocated by caller)
               aiter_tensor_t* qo_indptr,    //   nullable
               aiter_tensor_t* work_indptr,  //   nullable
               aiter_tensor_t* work_info,    //   nullable
               aiter_tensor_t* splitData,    //   nullable
               aiter_tensor_t* splitLse,     //   nullable
               int mask,
               int high_precision,
               const char* kernelName_,   //   nullable
               int quant_type,            //   QuantType enum value
               hipStream_t stream)
{
    int batch           = qo_indptr->size(0) - 1;
    int num_heads       = Q->size(1);
    int head_size       = Q->size(2);
    int num_kv_heads    = K->size(1);
    int block_size      = K->size(3);
    const int gqa_ratio = num_heads / num_kv_heads;

    int dim            = head_size;
    int stride_Q       = Q->stride(0) * Q->element_size();
    int stride_KV_head = K->stride(1) * K->element_size();
    int stride_KV_blk  = K->stride(0) * K->element_size();
    int stride_scale_blk = (K_QScale != nullptr)
                               ? (K_QScale->stride(1) * K_QScale->element_size())
                               : (block_size * sizeof(float));
    float k_log2e      = f_log2E;
    float k_scalar     = sqrt(dim);
    k_scalar           = (float)((double)k_log2e / (double)k_scalar);

    PsKernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_O      = out_->data_ptr();
    args.ptr_Q      = Q->data_ptr();
    args.ptr_K      = K->data_ptr();
    args.ptr_V      = V->data_ptr();

    args.ptr_KVIndices     = kv_indices->data_ptr();
    args.ptr_CL     = context_lens->data_ptr();
    if(K_QScale != nullptr)
    {
        args.ptr_KQ = K_QScale->data_ptr();
        args.ptr_VQ = V_QScale->data_ptr();
    }
    else
    {
        args.ptr_KQ = nullptr;
        args.ptr_VQ = nullptr;
    }
    args.sclg2e       = k_scalar;
    args.kv_nheads    = num_kv_heads;
    args.Qs           = stride_Q;
    args.Bs           = stride_KV_blk;
    args.KVs          = stride_KV_head;
    args.GQA          = gqa_ratio;
    args.ptr_QOPtr      = (qo_indptr != nullptr) ? qo_indptr->data_ptr() : nullptr;
    args.ptr_KVPtr     = kv_indptr->data_ptr();
    args.ptr_WorkPtr  = (work_indptr != nullptr) ? work_indptr->data_ptr() : nullptr;
    args.ptr_WorkInfo = (work_info != nullptr) ? work_info->data_ptr() : nullptr;
    args.ptr_SplitO   = (work_info != nullptr) ? splitData->data_ptr() : nullptr;
    args.ptr_SplitLSE = (work_info != nullptr) ? splitLse->data_ptr() : nullptr;
    args.stride_scale_blk = stride_scale_blk;
    args.mtp          = max_qlen - 1;

    const HipDeviceGuard device_guard(Q->device_id);

    std::string q_type;
    std::string kv_type;
    int gqa;
    int mtp;
    int msk;
    int hp;
    int ps = (work_indptr != nullptr) ? 1 : 0;
    // 1. "q_type"
    auto q_dtype = Q->dtype();
    auto kv_dtype = K->dtype();
    if(q_dtype == AITER_DTYPE_fp16)
        q_type = "fp16";
    else if(q_dtype == AITER_DTYPE_bf16)
        q_type = "bf16";
    else
        AITER_CHECK(false, __func__, ": unsupport Q dtype:", AiterDtype_to_str(q_dtype));

    // 2. "kv_type"
    if(kv_dtype == AITER_DTYPE_fp16)
        kv_type = "fp16";
    else if(kv_dtype == AITER_DTYPE_bf16)
        kv_type = "bf16";
    else if(kv_dtype == AITER_DTYPE_i8 || kv_dtype == AITER_DTYPE_u8)
        kv_type = "int8";
    else if(kv_dtype == AITER_DTYPE_fp8)
        kv_type = "fp8";
    else
        AITER_CHECK(false, __func__, ": unsupport K dtype:", AiterDtype_to_str(kv_dtype));

    // 3. "gqa_ratio"
    // 4. "mtp" , 5. "mask"
    // We make mtp=0, gqa=0 to dispatch kernel, since we only focus on qTile
    msk = mask;
    gqa = 0;
    mtp = 0;

    // 6. "high_precision" , 7. "ultra_precision"
    switch(high_precision)
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

    AITER_CHECK(qTile != -1,
                __func__,
                ": required qTile (gqa_ratio * max_qlen = ", gqa_ratio, " * ", max_qlen,
                " = ", required_qTile,
                ") exceeds maximum available qTile. Please reduce gqa_ratio or max_qlen.");

    CFG* config_map = &cfg_pa_asm; // only one config csv in hsa/<arch>/pa, now
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    std::string arch_id = get_gpu_arch();
    std::string kernelName = (kernelName_ != nullptr) ? std::string(kernelName_) :
        get_heuristic_kernel(q_type, kv_type, gqa, mtp, msk, hp, block_size, arch_id, ps, qTile, quant_type, config_map);
    if(kernelName.empty())
    {
        AITER_CHECK(false, __func__, "not supported this kernel now! ");
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
        AITER_CHECK(false, __func__, " not find kernel ", kernelName);

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx, // gdx
                             gdy, // gdy
                             1,   // gdz
                             256, // bdx: 4 wv64
                             1,   // bdy
                             1,   // bdz
                             stream});
}
