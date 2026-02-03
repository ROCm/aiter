// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "asm_mla_configs.hpp"
#include "py_itfs_common.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

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
    
    TORCH_CHECK(false,
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

void mla_decode_stage1_asm_fwd(
    torch::Tensor& Q,                    //   [num_seqs, num_heads, head_size]
    torch::Tensor& KV,                   //   [num_page, page_size, num_kv_heads, head_size] or [num_page, page_size*(nhead_kv*(kv_lora_rank+scale_dim+qk_rope_head_dim))]
    torch::Tensor& qo_indptr,            //   [batch_size+1]
    torch::Tensor& kv_indptr,            //   [batch_size+1]
    torch::Tensor& kv_page_indices,      //   [num_page_used]
    torch::Tensor& kv_last_page_lens,    //   [batch_size]
    std::optional<torch::Tensor>& num_kv_splits_indptr,   //   metadata
    std::optional<torch::Tensor>& work_meta_data,         //   metadata addr
    std::optional<torch::Tensor>& work_indptr,            //   metadata
    std::optional<torch::Tensor>& work_info_set,          //   [batch_size+1]
    int max_seqlen_q,
    int page_size,
    int nhead_kv,
    float softmax_scale,
    // following are output
    torch::Tensor& splitData, //[batch_size, num_kv_splits, num_heads, v_head_dim]
    torch::Tensor& splitLse,  //[batch_size, num_kv_splits, num_heads,  1]
    torch::Tensor& output,    //[batch_size, num_heads, v_head_dim]
    std::optional<torch::Tensor> q_scale  = std::nullopt, //   [1]
    std::optional<torch::Tensor> kv_scale = std::nullopt  //   [1]
)
{    
    int batch           = qo_indptr.size(0) - 1;
    int num_heads       = Q.size(1);
    int head_size       = Q.size(2);
    int num_kv_heads    = nhead_kv;
    int kv_split        = splitData.size(1);
    const int gqa_ratio = num_heads / num_kv_heads;

    bool persistent = !num_kv_splits_indptr.has_value();

    int stride_Q       = Q.stride(0) * Q.itemsize() * max_seqlen_q;
    int stride_Page    = KV.stride(0) * KV.itemsize();
    uint32_t log2_page = (uint32_t)log2f(page_size);

    KernelArgs args;
    size_t arg_size  = sizeof(args);
    args.ptr_R       = splitData.data_ptr();
    args.ptr_LSE     = splitLse.data_ptr();
    args.ptr_Q       = Q.data_ptr();
    args.ptr_KV      = KV.data_ptr();
    args.ptr_LTP     = kv_indptr.data_ptr();
    args.ptr_LTD     = kv_page_indices.data_ptr();
    args.ptr_LTL     = kv_last_page_lens.data_ptr();
    args.ptr_QTP     = qo_indptr.data_ptr();
    args.scalar      = softmax_scale;
    args.s_MQA       = gqa_ratio * max_seqlen_q;
    args.s_kv_split  = kv_split;
    args.s_Q_Bs      = stride_Q;
    args.s_Bs        = stride_Page;
    args.s_log2_plen = log2_page;
    args.out_16_nosplit = kv_split;

    if (persistent)
    {
        if (work_meta_data.has_value())
        {
            args.ptr_STP = work_meta_data.value().data_ptr();
        }
        else
        {
            assert(work_indptr.has_value() && work_info_set.has_value());
            assert(work_indptr.value().data_ptr() != nullptr && work_info_set.value().data_ptr() != nullptr);

            uint64_t* persistent_meta_data = new uint64_t[10];
            persistent_meta_data[0] = (uint64_t)work_indptr.value().data_ptr();
            persistent_meta_data[1] = (uint64_t)work_info_set.value().data_ptr();
            uint32_t* dev_PS_META_DATA;

            unsigned long buf_size_META = 10 * sizeof(uint64_t);
            hipMalloc(&dev_PS_META_DATA, buf_size_META);
            hipMemcpy(dev_PS_META_DATA, persistent_meta_data, buf_size_META, hipMemcpyHostToDevice);

            args.ptr_STP = dev_PS_META_DATA;
        }
    }
    else
    {
        args.ptr_STP = num_kv_splits_indptr.value().data_ptr();
    }
    args.ptr_RP = output.data_ptr(); //final output
    

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

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(Q));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    TORCH_CHECK(Q.is_contiguous(), __func__, ":only support Q.is_contiguous() for now");
    TORCH_CHECK(num_kv_heads == 1, __func__, ":only support num_kv_heads==1 for now");
    if (KV.dtype() != at::ScalarType::Byte && KV.dtype() != at::ScalarType::Char) {
        TORCH_CHECK(head_size == KV.size(3), __func__, ":only support head_size == KV.size(3) for now");
    }
    
    if(Q.dtype() == at::ScalarType::Float8_e4m3fnuz || Q.dtype() == at::ScalarType::Float8_e4m3fn)
    {
        assert(q_scale.has_value() && kv_scale.has_value());
        assert(q_scale.value().data_ptr() != nullptr && kv_scale.value().data_ptr() != nullptr);
        args.ptr_QSCALE  = q_scale.value().data_ptr();
        args.ptr_KVSCALE = kv_scale.value().data_ptr();
    }
    else if((KV.dtype() == at::ScalarType::Float8_e4m3fnuz || KV.dtype() == at::ScalarType::Float8_e4m3fn) && kv_scale.has_value())
    {
        assert(kv_scale.value().data_ptr() != nullptr);
        args.ptr_KVSCALE = kv_scale.value().data_ptr();
    }

    // Determine data types
    std::string q_type, kv_type;
    if(Q.dtype() == at::ScalarType::BFloat16)
        q_type = "bf16";
    else if(Q.dtype() == at::ScalarType::Float8_e4m3fnuz || Q.dtype() == at::ScalarType::Float8_e4m3fn)
        q_type = "fp8";
    else
        TORCH_CHECK(false, __func__, ": unsupport Q dtype:", Q.scalar_type());

    if(KV.dtype() == at::ScalarType::BFloat16)
        kv_type = "bf16";
    else if(KV.dtype() == at::ScalarType::Float8_e4m3fnuz || KV.dtype() == at::ScalarType::Float8_e4m3fn)
        kv_type = "fp8";
    else if(KV.dtype() == at::ScalarType::Byte || KV.dtype() == at::ScalarType::Char)
        kv_type = "byte";
    else
        TORCH_CHECK(false, __func__, ": unsupport KV dtype:", KV.scalar_type());

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
                TORCH_CHECK(false, __func__, ":only support fp8 mla decoding for qo_len <= 4");
            }
        }
    } else if (gqa_ratio == 32){
        if (q_type == "bf16" && kv_type == "bf16"){
            if(!persistent){
                config_max_seqlen_q = 0;
                sub_Q = 64;
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
    
    TORCH_CHECK(!kernelName.empty(), __func__, ": cannot find suitable kernel");
    
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

    TORCH_CHECK(impl_ptr != nullptr, __func__,
        ": unsupport current data type or shape. please refer to asm_mla.cu");

    int bdx = 256;
    int gdx = (max_seqlen_q * gqa_ratio + sub_Q - 1) / sub_Q;
    int gdy = batch;
    int gdz = kv_split;

    if(persistent)
    {
        gdx = work_indptr.value().size(0) - 1;
        gdy = 1;
        gdz = 1;
    }
    // printf("gdz: %d \n", gdz);

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx,       // gdx
                             gdy,       // gdy
                             gdz,       // gdz
                             256,       // bdx: 4 wv64
                             1,         // bdy
                             1,         // bdz
                             stream});
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


void mla_prefill_ps_asm_fwd(
    torch::Tensor& Q,                    //  [num_seqs, num_q_heads, qk_hetad_size], fp8
    torch::Tensor& K,                   //   [num_page, num_kv_heads, qk_head_size], fp8
    torch::Tensor& V,                   //   [num_page, num_kv_heads, v_head_size], fp8
    torch::Tensor& qo_indptr,            //   [batch_size+1], int
    torch::Tensor& kv_indptr,            //   [batch_size+1], int
    torch::Tensor& kv_page_indices,      //   [num_page_used], int
    std::optional<torch::Tensor>& work_indptr,            //   [available_tgs+1], int
    std::optional<torch::Tensor>& work_info_set,          //   [max_works], int
    int max_seqlen_q,
    float softmax_scale, // following are output
    bool is_causal,
    torch::Tensor& splitData, //[num_q_heads, num_seqs, max_kv_split, v_head_dim], fp32
    torch::Tensor& splitLse,  //[num_q_heads, num_seqs, max_kv_split,  1], fp32
    torch::Tensor& output,    //[num_seqs, num_q_heads, v_head_dim], bf16
    std::optional<torch::Tensor> q_scale  = std::nullopt,// fp32, scalar
    std::optional<torch::Tensor> k_scale = std::nullopt, // fp32, scalar
    std::optional<torch::Tensor> v_scale = std::nullopt  // fp32, scalar
){
    // TORCH_CHECK(Q.dtype() == at::ScalarType::Float8_e4m3fnuz || Q.dtype() == at::ScalarType::Float8_e4m3fn,
    //             __func__, ": only support fp8 Q for now");
    // TORCH_CHECK(K.dtype() == at::ScalarType::Float8_e4m3fnuz || K.dtype() == at::ScalarType::Float8_e4m3fn,
    //             __func__, ": only support fp8 K for now");
    // TORCH_CHECK(V.dtype() == at::ScalarType::Float8_e4m3fnuz || V.dtype() == at::ScalarType::Float8_e4m3fn,
    //             __func__, ": only support fp8 V for now");
    // TORCH_CHECK(q_scale.has_value() && k_scale.has_value() && v_scale.has_value(),
    //             __func__, ": fp8 requires scale tensors");
    // TORCH_CHECK(work_indptr.has_value() && work_info_set.has_value(),
    //             __func__, ": persistent scheduling requires work_indptr and work_info_set");
    
    int num_q_tokens  = Q.size(0);
    int num_head_q    = Q.size(1);
    int num_page      = K.size(0);
    int num_kv_heads = K.size(1);
    int num_used_page = kv_page_indices.size(0);
    int available_tgs = 1;
    const int gqa_ratio = num_head_q / num_kv_heads;

    // Setup kernel arguments
    PsKernelArgs args;
    size_t arg_size = sizeof(args);
    
    float k_scalar = softmax_scale;
    
    // Setup all pointers
    args.ptr_Q             = Q.data_ptr();
    args.ptr_K             = K.data_ptr();
    args.ptr_V             = V.data_ptr();
    args.ptr_O             = output.data_ptr();
    args.ptr_PartialO      = splitData.data_ptr();
    args.ptr_PartialLSE    = splitLse.data_ptr();
    args.ptr_WorkIndptr    = work_indptr.has_value() ? work_indptr.value().data_ptr() : nullptr;
    args.ptr_WorkInfo      = work_info_set.has_value() ? work_info_set.value().data_ptr() : nullptr;
    args.ptr_QOIndptr      = qo_indptr.data_ptr();
    args.ptr_KVIndptr      = kv_indptr.data_ptr();
    args.ptr_KVPageIndices = kv_page_indices.data_ptr();
    args.ptr_QScale        = q_scale.has_value() ? q_scale.value().data_ptr() : nullptr;
    args.ptr_KScale        = k_scale.has_value() ? k_scale.value().data_ptr() : nullptr;
    args.ptr_VScale        = v_scale.has_value() ? v_scale.value().data_ptr() : nullptr;
    args.scalar            = k_scalar;
    args.num_q_tokens      = num_q_tokens;
    args.num_head_q        = num_head_q;
    args.num_page          = num_page;
    args.num_used_page     = num_used_page;
    
    // Get CUDA/HIP stream and device guard
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(Q));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    
    // Determine data types
    std::string q_type, k_type;
    if(Q.dtype() == at::ScalarType::Float8_e4m3fnuz || Q.dtype() == at::ScalarType::Float8_e4m3fn)
        q_type = "fp8";
    else
        TORCH_CHECK(false, __func__, ": unsupport Q dtype:", Q.scalar_type());

    if(K.dtype() == at::ScalarType::Float8_e4m3fnuz || K.dtype() == at::ScalarType::Float8_e4m3fn)
        k_type = "fp8";
    else
        TORCH_CHECK(false, __func__, ": unsupport K dtype:", K.scalar_type());

    // Get kernel using config dispatch
    std::string arch_id = get_gpu_arch();
    if(arch_id == "gfx942"){
        TORCH_CHECK(false, __func__, ": fp8 mla persistent prefill is not supported on gfx942");
    }
    CFG* config_map = &cfg_mla_asm;
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    
    int ps = 1; // ps_prefill always uses persistent scheduling
    int prefill = 1; // prefill stage
    int causal_flag = is_causal ? 1 : 0;
    int qseqlen = 0; // not used for prefill
    
    std::string kernelName = get_heuristic_kernel_mla(q_type, k_type, gqa_ratio, ps, prefill, causal_flag, qseqlen, arch_id, config_map);
    
    TORCH_CHECK(!kernelName.empty(), __func__, ": cannot find suitable kernel");
    
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
        TORCH_CHECK(false, __func__, " not find kernel " + kernelName);
    
    // Launch kernel
    int block_size_x = wave_per_tg * 64;  // 8 * 64 = 512
    // int grid_size_x  = get_num_cu_func();
    // std::cout << "grid_size_x" << grid_size_x << std::endl;
    int grid_size_x = work_indptr.value().size(0) - 1;
    // std::cout << "grid_size_x" << grid_size_x << std::endl;
    
    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             grid_size_x,  // gdx
                             1,            // gdy
                             1,            // gdz
                             block_size_x, // bdx
                             1,            // bdy
                             1,            // bdz
                             stream});
}


void mla_prefill_asm_fwd(
    torch::Tensor& Q,                 //   [num_seqs, num_heads, head_size]
    torch::Tensor& KV,                //   [num_page, page_size, num_kv_heads, head_size]
    torch::Tensor& qo_indptr,         //   [batch_size+1]
    torch::Tensor& kv_indptr,         //   [batch_size+1]
    torch::Tensor& kv_page_indices,   //   [num_page_used]
    torch::Tensor& kv_last_page_lens, //   [batch_size]
    int max_seqlen_q,
    float softmax_scale,
    // following are output
    torch::Tensor& splitData, //[batch_size, num_kv_splits, num_heads, v_head_dim]
    torch::Tensor& splitLse   //[batch_size, num_kv_splits, num_heads,  1]

)
{
    int sub_Q           = 128;
    int batch           = kv_indptr.size(0) - 1;
    int num_heads       = Q.size(1);
    int head_size       = Q.size(2);
    int page_size       = KV.size(1);
    int num_kv_heads    = KV.size(2);
    int kv_split        = splitData.size(1);
    const int gqa_ratio = num_heads / num_kv_heads;

    int stride_Q       = Q.stride(0) * Q.itemsize();
    int stride_Page    = KV.stride(0) * KV.itemsize();
    uint32_t log2_page = (uint32_t)log2f(page_size);

    KernelArgs args;
    size_t arg_size  = sizeof(args);
    args.ptr_R       = splitData.data_ptr();
    args.ptr_LSE     = splitLse.data_ptr();
    args.ptr_Q       = Q.data_ptr();
    args.ptr_KV      = KV.data_ptr();
    args.ptr_LTP     = kv_indptr.data_ptr();
    args.ptr_LTD     = kv_page_indices.data_ptr();
    args.ptr_LTL     = kv_last_page_lens.data_ptr();
    args.ptr_QTP     = qo_indptr.data_ptr();
    args.scalar      = softmax_scale;
    args.s_MQA       = gqa_ratio;
    args.s_kv_split  = kv_split;
    args.s_Q_Bs      = stride_Q;
    args.s_Bs        = stride_Page;
    args.s_log2_plen = log2_page;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(Q));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    TORCH_CHECK(Q.is_contiguous(), __func__, ":only support Q.is_contiguous() for now");
    TORCH_CHECK(gqa_ratio == 16 || gqa_ratio == 128,
                __func__,
                ":only support num_q_heads/num_kv_heads==16 or 128 for now");
    TORCH_CHECK(num_kv_heads == 1, __func__, ":only support num_kv_heads==1 for now");
    TORCH_CHECK(head_size == KV.size(3), __func__, ":only support head_size == KV.size(3) for now");
    
    // Determine data types
    std::string q_type, kv_type;
    if(Q.dtype() == at::ScalarType::BFloat16)
        q_type = "bf16";
    else 
        TORCH_CHECK(false, __func__, ": unsupport Q dtype:", Q.scalar_type());

    if(KV.dtype() == at::ScalarType::BFloat16)
        kv_type = "bf16";
    else
        TORCH_CHECK(false, __func__, ": unsupport KV dtype:", KV.scalar_type());

    // Get kernel using config dispatch
    std::string arch_id = get_gpu_arch();
    CFG* config_map = &cfg_mla_asm;
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    
    int ps = 0; // prefill without persistent scheduling
    int prefill = 1; // prefill stage
    int causal_flag = 0;
    int qseqlen = 0;
    std::string kernelName = get_heuristic_kernel_mla(q_type, kv_type, gqa_ratio, ps, prefill, causal_flag, qseqlen, arch_id, config_map);
    
    TORCH_CHECK(!kernelName.empty(), __func__, ": cannot find suitable kernel");
    
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

    TORCH_CHECK(impl_ptr != nullptr, __func__, ": unsupport current Q_type:", Q.scalar_type());
    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             (max_seqlen_q * gqa_ratio + sub_Q - 1) / sub_Q, // gdx
                             batch,                                          // gdy
                             kv_split,                                       // gdz
                             256,                                            // bdx: 4 wv64
                             1,                                              // bdy
                             1,                                              // bdz
                             stream});
}