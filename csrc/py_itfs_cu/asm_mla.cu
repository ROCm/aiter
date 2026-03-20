// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "asm_mla_configs.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>

struct __attribute__((packed)) KernelArgs
{
    void* ptr_R;
    p2 _p0;
    void* ptr_LSE;
    p2 _p1;
    void* ptr_Q;
    p2 _p2;
    void* ptr_KV;
    p2 _p3;
    void* ptr_LTP;
    p2 _p4;
    void* ptr_LTD;
    p2 _p5;
    void* ptr_LTL;
    p2 _p6;
    float scalar;
    p3 _p12;
    unsigned int s_MQA;
    p3 _p13;
    unsigned int s_kv_split;
    p3 _p14;
    unsigned int s_Q_Bs;
    p3 _p15;
    unsigned int s_Bs;
    p3 _p16;
    unsigned int s_log2_plen;
    p3 _p17;
    void* ptr_QTP;
    p2 _p18;
    void* ptr_STP;
    p2 _p19;
    void* ptr_RP;
    p2 _p20;
    void* ptr_QSCALE;
    p2 _p21;
    void* ptr_KVSCALE;
    p2 _p22;
    unsigned int out_16_nosplit;
    p3 _p23;
    void* ptr_LSEP;
    p2 _p24;
};

std::string get_heuristic_kernel_mla(std::string q_type,
                                     std::string kv_type,
                                     int gqa,
                                     int ps,
                                     int prefill,
                                     int causal,
                                     int qseqlen,
                                     std::string arch_id,
                                     CFG* cfgs)
{
    for(const auto& el : *cfgs)
    {
        if (el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        
        if (cfg.qType != q_type || cfg.kvType != kv_type)
            continue;
        if (cfg.Gqa != gqa || cfg.ps != ps || cfg.prefill != prefill)
            continue;
        if (cfg.causal != causal || cfg.qSeqLen != qseqlen)
            continue;
        return el.first;
    }
    
    AITER_CHECK(false,
                __func__,
                ": cannot get heuristic kernel!"
                " q_type:", q_type,
                " kv_type:", kv_type,
                " gqa:", gqa,
                " ps:", ps,
                " prefill:", prefill,
                " causal:", causal,
                " qseqlen:", qseqlen);
    return "";
}

extern "C" __attribute__((visibility("default")))
void mla_decode_stage1_asm_fwd(
    AiterTensor* Q,                    //   [num_seqs, num_heads, head_size]
    AiterTensor* KV,                   //   [num_page, page_size, num_kv_heads, head_size] or [num_page, page_size*(nhead_kv*(kv_lora_rank+scale_dim+qk_rope_head_dim))]
    AiterTensor* qo_indptr,            //   [batch_size+1]
    AiterTensor* kv_indptr,            //   [batch_size+1]
    AiterTensor* kv_page_indices,      //   [num_page_used]
    AiterTensor* kv_last_page_lens,    //   [batch_size]
    AiterTensor* num_kv_splits_indptr, //   metadata (nullable)
    AiterTensor* work_meta_data,       //   metadata addr (nullable)
    AiterTensor* work_indptr,          //   metadata (nullable)
    AiterTensor* work_info_set,        //   [batch_size+1] (nullable)
    int max_seqlen_q,
    int page_size,
    int nhead_kv,
    float softmax_scale,
    // following are output
    AiterTensor* splitData,            //   [batch_size, num_kv_splits, num_heads, v_head_dim]
    AiterTensor* splitLse,             //   [batch_size, num_kv_splits, num_heads,  1]
    AiterTensor* output,               //   [batch_size, num_heads, v_head_dim]
    AiterTensor* lse,                  //   [batch_size, num_heads] (nullable)
    AiterTensor* q_scale,              //   [1] (nullable)
    AiterTensor* kv_scale,             //   [1] (nullable)
    hipStream_t stream)
{    
    int batch           = qo_indptr->size(0) - 1;
    int num_heads       = Q->size(1);
    int head_size       = Q->size(2);
    int num_kv_heads    = nhead_kv;
    int kv_split        = splitData->size(1);
    const int gqa_ratio = num_heads / num_kv_heads;

    bool persistent = (num_kv_splits_indptr == nullptr);

    int stride_Q       = Q->stride(0) * Q->element_size() * max_seqlen_q;
    int stride_Page    = KV->stride(0) * KV->element_size();
    uint32_t log2_page = (uint32_t)log2f(page_size);

    KernelArgs args;
    size_t arg_size  = sizeof(args);
    args.ptr_R       = splitData->data_ptr();
    args.ptr_LSE     = splitLse->data_ptr();
    args.ptr_Q       = Q->data_ptr();
    args.ptr_KV      = KV->data_ptr();
    args.ptr_LTP     = kv_indptr->data_ptr();
    args.ptr_LTD     = kv_page_indices->data_ptr();
    args.ptr_LTL     = kv_last_page_lens->data_ptr();
    args.ptr_QTP     = qo_indptr->data_ptr();
    args.scalar      = softmax_scale;
    args.s_MQA       = gqa_ratio * max_seqlen_q;
    args.s_kv_split  = kv_split;
    args.s_Q_Bs      = stride_Q;
    args.s_Bs        = stride_Page;
    args.s_log2_plen = log2_page;
    args.out_16_nosplit = kv_split;

    if (persistent)
    {
        if (work_meta_data != nullptr)
        {
            args.ptr_STP = work_meta_data->data_ptr();
        }
        else
        {
            AITER_CHECK(work_indptr != nullptr && work_info_set != nullptr,
                        __func__, ": work_indptr and work_info_set must be provided");
            AITER_CHECK(work_indptr->data_ptr() != nullptr && work_info_set->data_ptr() != nullptr,
                        __func__, ": work_indptr and work_info_set data_ptr must not be null");

            uint64_t* persistent_meta_data = new uint64_t[10];
            persistent_meta_data[0] = (uint64_t)work_indptr->data_ptr();
            persistent_meta_data[1] = (uint64_t)work_info_set->data_ptr();
            uint32_t* dev_PS_META_DATA;

            unsigned long buf_size_META = 10 * sizeof(uint64_t);
            hipMalloc(&dev_PS_META_DATA, buf_size_META);
            hipMemcpy(dev_PS_META_DATA, persistent_meta_data, buf_size_META, hipMemcpyHostToDevice);

            args.ptr_STP = dev_PS_META_DATA;
        }
    }
    else
    {
        args.ptr_STP = num_kv_splits_indptr->data_ptr();
    }
    args.ptr_RP = output->data_ptr(); //final output
    args.ptr_LSEP = nullptr;
    if (lse != nullptr)
    {
        args.ptr_LSEP = lse->data_ptr(); //final lse
    }

    // std::cout << "mla args" << std::endl;
    // std::cout << "ptr_R: " << args.ptr_R << std::endl;
    // std::cout << "ptr_LSE: " << args.ptr_LSE << std::endl;
    // std::cout << "ptr_Q: " << args.ptr_Q << std::endl;
    // std::cout << "ptr_KV: " << args.ptr_KV << std::endl;
    // std::cout << "ptr_LTP: " << args.ptr_LTP << std::endl;
    // std::cout << "ptr_LTD: " << args.ptr_LTD << std::endl;
    // std::cout << "ptr_LTL: " << args.ptr_LTL << std::endl;
    // std::cout << "scalar: " << args.scalar << std::endl;
    // std::cout << "s_MQA: " << args.s_MQA << std::endl;
    // std::cout << "s_kv_split: " << args.s_kv_split << std::endl;
    // std::cout << "s_Q_Bs: " << args.s_Q_Bs << std::endl;
    // std::cout << "s_Bs: " << args.s_Bs << std::endl;
    // std::cout << "s_log2_plen: " << args.s_log2_plen << std::endl;
    // std::cout << "ptr_RP: " << args.ptr_RP << std::endl;
    // std::cout << "ptr_QTP: " << args.ptr_QTP << std::endl;
    // std::cout << "ptr_STP: " << args.ptr_STP << std::endl;
    // std::cout << "out_16_nosplit: " << args.out_16_nosplit << std::endl;
    // std::cout << "ptr_LSEP: " << args.ptr_LSEP << std::endl;

    int prev_device;
    HIP_CALL(hipGetDevice(&prev_device));
    HIP_CALL(hipSetDevice(Q->device_id));

    AITER_CHECK(Q->is_contiguous(), __func__, ":only support Q.is_contiguous() for now");
    AITER_CHECK(num_kv_heads == 1, __func__, ":only support num_kv_heads==1 for now");

    auto q_dtype = Q->dtype();
    auto kv_dtype = KV->dtype();

    if (kv_dtype != AITER_DTYPE_i8 && kv_dtype != AITER_DTYPE_u8) {
        AITER_CHECK(head_size == KV->size(3), __func__, ":only support head_size == KV.size(3) for now");
    }
    
    if(q_dtype == AITER_DTYPE_fp8)
    {
        AITER_CHECK(q_scale != nullptr && kv_scale != nullptr,
                    __func__, ": fp8 Q requires q_scale and kv_scale");
        AITER_CHECK(q_scale->data_ptr() != nullptr && kv_scale->data_ptr() != nullptr,
                    __func__, ": q_scale and kv_scale data_ptr must not be null");
        args.ptr_QSCALE  = q_scale->data_ptr();
        args.ptr_KVSCALE = kv_scale->data_ptr();
    }
    else if(kv_dtype == AITER_DTYPE_fp8 && kv_scale != nullptr)
    {
        AITER_CHECK(kv_scale->data_ptr() != nullptr,
                    __func__, ": kv_scale data_ptr must not be null");
        args.ptr_KVSCALE = kv_scale->data_ptr();
    }

    // Determine data types
    std::string q_type, kv_type;
    if(q_dtype == AITER_DTYPE_bf16)
        q_type = "bf16";
    else if(q_dtype == AITER_DTYPE_fp8)
        q_type = "fp8";
    else
        AITER_CHECK(false, __func__, ": unsupport Q dtype:", AiterDtype_to_str(q_dtype));

    if(kv_dtype == AITER_DTYPE_bf16)
        kv_type = "bf16";
    else if(kv_dtype == AITER_DTYPE_fp8)
        kv_type = "fp8";
    else if(kv_dtype == AITER_DTYPE_i8 || kv_dtype == AITER_DTYPE_u8)
        kv_type = "byte";
    else
        AITER_CHECK(false, __func__, ": unsupport KV dtype:", AiterDtype_to_str(kv_dtype));

    // Get kernel using config dispatch
    std::string arch_id = get_gpu_arch();
    CFG* config_map = &cfg_mla_asm;
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    
    int ps = persistent ? 1 : 0;
    int prefill = 0; // decode stage
    int causal = 0;
    int config_max_seqlen_q = max_seqlen_q;
    int sub_Q = 128; // default value
    
    if(gqa_ratio == 128){
        config_max_seqlen_q = 0;
        sub_Q = 128;
        if (q_type == "bf16" && kv_type == "bf16"){
            ps = 0; // not use ps
        }
    }
    else if(gqa_ratio == 16){
        sub_Q = 128;
        if (q_type == "bf16" && kv_type == "bf16"){
            if(persistent){
                if (max_seqlen_q <= 4){
                    config_max_seqlen_q = 4; // padding it
                }
            }else{
                if(max_seqlen_q == 1){
                    config_max_seqlen_q = 1;
                    sub_Q = 16;
                }else if(max_seqlen_q <= 4){
                    config_max_seqlen_q = 4;
                }else{
                    config_max_seqlen_q = 8;
                }
            }
        }else if ((q_type == "bf16" && kv_type == "fp8") || (q_type == "bf16" && kv_type == "byte")){
            if(persistent){
                if(max_seqlen_q <= 4){
                    config_max_seqlen_q = 4;
                }
            }
        }else if (q_type == "fp8"){
            if(max_seqlen_q == 1){
                config_max_seqlen_q = 1;
            }else if(max_seqlen_q == 2){
                config_max_seqlen_q = 2;
            }else if(max_seqlen_q <= 4){
                sub_Q = 64;
                config_max_seqlen_q = 4;
            }else if (max_seqlen_q > 4){
                AITER_CHECK(false, __func__, ":only support fp8 mla decoding for qo_len <= 4");
            }
        }
    } else if (gqa_ratio == 32){
        if (q_type == "bf16" && kv_type == "bf16"){
            if(!persistent){
                config_max_seqlen_q = 0;
                sub_Q = 64;
            }
        }else if (q_type == "fp8" && kv_type == "fp8"){
            if((max_seqlen_q == 4) && persistent){
                config_max_seqlen_q = 4;
                sub_Q = 128;
            } else {
                AITER_CHECK(false, __func__, 
                    ": fp8/fp8 with gqa_ratio=32 only supports decode_qlen=4 in persistent mode");
            }
        }
    } else if (gqa_ratio == 64){
        if (q_type == "bf16" && kv_type == "bf16"){
            if(!persistent){
                config_max_seqlen_q = 0;
                sub_Q = 64;
            }
        }
    }

    std::string kernelName = get_heuristic_kernel_mla(q_type, kv_type, gqa_ratio, ps, prefill, causal, config_max_seqlen_q, arch_id, config_map);
    
    AITER_CHECK(!kernelName.empty(), __func__, ": cannot find suitable kernel");
    
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

    AITER_CHECK(impl_ptr != nullptr, __func__,
        ": unsupport current data type or shape. please refer to asm_mla.cu");

    int bdx = 256;
    int gdx = (max_seqlen_q * gqa_ratio + sub_Q - 1) / sub_Q;
    int gdy = batch;
    int gdz = kv_split;

    if(persistent)
    {
        gdx = work_indptr->size(0) - 1;
        gdy = 1;
        gdz = 1;
    }

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx,       // gdx
                             gdy,       // gdy
                             gdz,       // gdz
                             256,       // bdx: 4 wv64
                             1,         // bdy
                             1,         // bdz
                             stream});
    HIP_CALL(hipSetDevice(prev_device));
}

struct __attribute__((packed)) PsKernelArgs
{
    void *ptr_Q;
    p2 _p0;
    void *ptr_K;
    p2 _p1;
    void *ptr_V;
    p2 _p2;
    void *ptr_O;
    p2 _p3;
    void *ptr_PartialO;
    p2 _p4;
    void *ptr_PartialLSE;
    p2 _p5;
    void *ptr_WorkIndptr;
    p2 _p6;
    void *ptr_WorkInfo;
    p2 _p7;
    void *ptr_QOIndptr;
    p2 _p8;
    void *ptr_KVIndptr;
    p2 _p9;
    void *ptr_KVPageIndices;
    p2 _p10;
    void *ptr_QScale;
    p2 _p11;
    void *ptr_KScale;
    p2 _p12;
    void *ptr_VScale;
    p2 _p13;
    float scalar;
    p3 _p14;
    unsigned int num_q_tokens;
    p3 _p15;
    unsigned int num_head_q;
    p3 _p16;
    unsigned int num_page;
    p3 _p17;
    unsigned int num_used_page;
    p3 _p18;
};


extern "C" __attribute__((visibility("default")))
void mla_prefill_ps_asm_fwd(
    AiterTensor* Q,                    //  [num_seqs, num_q_heads, qk_hetad_size], fp8
    AiterTensor* K,                    //   [num_page, num_kv_heads, qk_head_size], fp8
    AiterTensor* V,                    //   [num_page, num_kv_heads, v_head_size], fp8
    AiterTensor* qo_indptr,            //   [batch_size+1], int
    AiterTensor* kv_indptr,            //   [batch_size+1], int
    AiterTensor* kv_page_indices,      //   [num_page_used], int
    AiterTensor* work_indptr,          //   [available_tgs+1], int (nullable)
    AiterTensor* work_info_set,        //   [max_works], int (nullable)
    int max_seqlen_q,
    float softmax_scale,
    int is_causal,
    AiterTensor* splitData,            //   [num_q_heads, num_seqs, max_kv_split, v_head_dim], fp32
    AiterTensor* splitLse,             //   [num_q_heads, num_seqs, max_kv_split,  1], fp32
    AiterTensor* output,               //   [num_seqs, num_q_heads, v_head_dim], bf16
    AiterTensor* q_scale,              //   fp32, scalar (nullable)
    AiterTensor* k_scale,              //   fp32, scalar (nullable)
    AiterTensor* v_scale,              //   fp32, scalar (nullable)
    hipStream_t stream)
{
    int num_q_tokens  = Q->size(0);
    int num_head_q    = Q->size(1);
    int num_page      = K->size(0);
    int num_kv_heads  = K->size(1);
    int num_used_page = kv_page_indices->size(0);
    int available_tgs = 1;
    const int gqa_ratio = num_head_q / num_kv_heads;

    PsKernelArgs args;
    size_t arg_size = sizeof(args);
    
    float k_scalar = softmax_scale;
    
    args.ptr_Q             = Q->data_ptr();
    args.ptr_K             = K->data_ptr();
    args.ptr_V             = V->data_ptr();
    args.ptr_O             = output->data_ptr();
    args.ptr_PartialO      = splitData->data_ptr();
    args.ptr_PartialLSE    = splitLse->data_ptr();
    args.ptr_WorkIndptr    = work_indptr != nullptr ? work_indptr->data_ptr() : nullptr;
    args.ptr_WorkInfo      = work_info_set != nullptr ? work_info_set->data_ptr() : nullptr;
    args.ptr_QOIndptr      = qo_indptr->data_ptr();
    args.ptr_KVIndptr      = kv_indptr->data_ptr();
    args.ptr_KVPageIndices = kv_page_indices->data_ptr();
    args.ptr_QScale        = q_scale != nullptr ? q_scale->data_ptr() : nullptr;
    args.ptr_KScale        = k_scale != nullptr ? k_scale->data_ptr() : nullptr;
    args.ptr_VScale        = v_scale != nullptr ? v_scale->data_ptr() : nullptr;
    args.scalar            = k_scalar;
    args.num_q_tokens      = num_q_tokens;
    args.num_head_q        = num_head_q;
    args.num_page          = num_page;
    args.num_used_page     = num_used_page;
    
    int prev_device;
    HIP_CALL(hipGetDevice(&prev_device));
    HIP_CALL(hipSetDevice(Q->device_id));

    auto q_dtype = Q->dtype();
    auto k_dtype = K->dtype();

    std::string q_type, k_type;
    if(q_dtype == AITER_DTYPE_fp8)
        q_type = "fp8";
    else
        AITER_CHECK(false, __func__, ": unsupport Q dtype:", AiterDtype_to_str(q_dtype));

    if(k_dtype == AITER_DTYPE_fp8)
        k_type = "fp8";
    else
        AITER_CHECK(false, __func__, ": unsupport K dtype:", AiterDtype_to_str(k_dtype));

    std::string arch_id = get_gpu_arch();
    if(arch_id == "gfx942"){
        AITER_CHECK(false, __func__, ": fp8 mla persistent prefill is not supported on gfx942");
    }
    CFG* config_map = &cfg_mla_asm;
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    
    int ps = 1; // ps_prefill always uses persistent scheduling
    int prefill = 1; // prefill stage
    int causal_flag = is_causal ? 1 : 0;
    int qseqlen = 0; // not used for prefill
    
    std::string kernelName = get_heuristic_kernel_mla(q_type, k_type, gqa_ratio, ps, prefill, causal_flag, qseqlen, arch_id, config_map);
    
    AITER_CHECK(!kernelName.empty(), __func__, ": cannot find suitable kernel");
    
    AiterAsmKernel* impl_ptr = nullptr;
    int wave_per_tg = 8;
    
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
    
    int block_size_x = wave_per_tg * 64;
    int grid_size_x = work_indptr->size(0) - 1;
    
    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             grid_size_x,  // gdx
                             1,            // gdy
                             1,            // gdz
                             block_size_x, // bdx
                             1,            // bdy
                             1,            // bdz
                             stream});
    HIP_CALL(hipSetDevice(prev_device));
}


extern "C" __attribute__((visibility("default")))
void mla_prefill_asm_fwd(
    AiterTensor* Q,                    //   [num_seqs, num_heads, head_size]
    AiterTensor* KV,                   //   [num_page, page_size, num_kv_heads, head_size]
    AiterTensor* qo_indptr,            //   [batch_size+1]
    AiterTensor* kv_indptr,            //   [batch_size+1]
    AiterTensor* kv_page_indices,      //   [num_page_used]
    AiterTensor* kv_last_page_lens,    //   [batch_size]
    int max_seqlen_q,
    float softmax_scale,
    AiterTensor* splitData,            //   [batch_size, num_kv_splits, num_heads, v_head_dim]
    AiterTensor* splitLse,             //   [batch_size, num_kv_splits, num_heads,  1]
    hipStream_t stream)
{
    int sub_Q           = 128;
    int batch           = kv_indptr->size(0) - 1;
    int num_heads       = Q->size(1);
    int head_size       = Q->size(2);
    int page_size       = KV->size(1);
    int num_kv_heads    = KV->size(2);
    int kv_split        = splitData->size(1);
    const int gqa_ratio = num_heads / num_kv_heads;

    int stride_Q       = Q->stride(0) * Q->element_size();
    int stride_Page    = KV->stride(0) * KV->element_size();
    uint32_t log2_page = (uint32_t)log2f(page_size);

    KernelArgs args;
    size_t arg_size  = sizeof(args);
    args.ptr_R       = splitData->data_ptr();
    args.ptr_LSE     = splitLse->data_ptr();
    args.ptr_Q       = Q->data_ptr();
    args.ptr_KV      = KV->data_ptr();
    args.ptr_LTP     = kv_indptr->data_ptr();
    args.ptr_LTD     = kv_page_indices->data_ptr();
    args.ptr_LTL     = kv_last_page_lens->data_ptr();
    args.ptr_QTP     = qo_indptr->data_ptr();
    args.scalar      = softmax_scale;
    args.s_MQA       = gqa_ratio;
    args.s_kv_split  = kv_split;
    args.s_Q_Bs      = stride_Q;
    args.s_Bs        = stride_Page;
    args.s_log2_plen = log2_page;

    int prev_device;
    HIP_CALL(hipGetDevice(&prev_device));
    HIP_CALL(hipSetDevice(Q->device_id));

    AITER_CHECK(Q->is_contiguous(), __func__, ":only support Q.is_contiguous() for now");
    AITER_CHECK(gqa_ratio == 16 || gqa_ratio == 128,
                __func__,
                ":only support num_q_heads/num_kv_heads==16 or 128 for now");
    AITER_CHECK(num_kv_heads == 1, __func__, ":only support num_kv_heads==1 for now");
    AITER_CHECK(head_size == KV->size(3), __func__, ":only support head_size == KV.size(3) for now");
    
    auto q_dtype = Q->dtype();
    auto kv_dtype = KV->dtype();

    std::string q_type, kv_type;
    if(q_dtype == AITER_DTYPE_bf16)
        q_type = "bf16";
    else 
        AITER_CHECK(false, __func__, ": unsupport Q dtype:", AiterDtype_to_str(q_dtype));

    if(kv_dtype == AITER_DTYPE_bf16)
        kv_type = "bf16";
    else
        AITER_CHECK(false, __func__, ": unsupport KV dtype:", AiterDtype_to_str(kv_dtype));

    std::string arch_id = get_gpu_arch();
    CFG* config_map = &cfg_mla_asm;
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    
    int ps = 0; // prefill without persistent scheduling
    int prefill = 1; // prefill stage
    int causal_flag = 0;
    int qseqlen = 0;
    std::string kernelName = get_heuristic_kernel_mla(q_type, kv_type, gqa_ratio, ps, prefill, causal_flag, qseqlen, arch_id, config_map);
    
    AITER_CHECK(!kernelName.empty(), __func__, ": cannot find suitable kernel");
    
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

    AITER_CHECK(impl_ptr != nullptr, __func__, ": unsupport current Q_type:", AiterDtype_to_str(q_dtype));
    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             (max_seqlen_q * gqa_ratio + sub_Q - 1) / sub_Q, // gdx
                             batch,                                          // gdy
                             kv_split,                                       // gdz
                             256,                                            // bdx: 4 wv64
                             1,                                              // bdy
                             1,                                              // bdz
                             stream});
    HIP_CALL(hipSetDevice(prev_device));
}
