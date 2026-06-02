// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include "asm_mla_configs.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <unordered_map>

// Debug instrumentation (host prints + post-launch sync/error checks + raw
// buffer dumps) for the gfx1250 mi400 MLA dispatch is compiled ONLY when
// ASM_DEBUG is defined (e.g. JIT build with `-DASM_DEBUG`, toggled by
// AITER_ASM_DEBUG=1). When ASM_DEBUG is not defined, none of the debug code
// below is compiled, so the normal build performs a pure async launch and is
// completely unaffected. This mirrors poc_kl/mi400/mla/mla_execute_v3_hip.inl,
// which keeps its prints and dumps in the C++ host under its own MLA_DEBUG
// macro. ASM_DEBUG is shared across asm-kernel ports; the stage1 raw buffer
// dump is additionally gated at runtime by the op-specific
// AITER_MLA_DEBUG_DUMP_DIR environment variable. Python-only artifacts that
// the asm kernel never sees (final output, final_lse, token-level kv_indptr)
// are still dumped from aiter/mla.py.
#ifdef ASM_DEBUG
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <vector>
#endif

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

struct __attribute__((packed)) MlaMi400KernelArgs
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
    p3 _p7;
    unsigned int q_seq_lens;
    p3 _p8;
    unsigned int passes;
    p3 _p9;
    // Patched .args names this slot total_kv, but the v3 kernel uses stride_Q.
    unsigned int stride_Q;
    p3 _p10;
    unsigned int stride_page;
    p3 _p11;
    unsigned int log2_page;
    p3 _p12;
    void* ptr_QTP;
    p2 _p13;
    void* ptr_STP;
    p2 _p14;
    unsigned int out_16_nosplit;
    p3 _p15;
    void* ptr_QROPE;
    p2 _p16;
    void* ptr_KVROPE;
    p2 _p17;
};

static_assert(sizeof(MlaMi400KernelArgs) == 288, "MLA mi400 packed args must be 18*16=288B");

std::string get_heuristic_kernel_mla(std::string q_type,
                                     std::string kv_type,
                                     int gqa,
                                     int ps,
                                     int prefill,
                                     int causal,
                                     int qseqlen,
                                     std::string arch_id,
                                     CFG* cfgs,
                                     int lse = 0)
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
        if (cfg.lse != lse)
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
                " qseqlen:", qseqlen,
                " lse:", lse);
    return "";
}

#ifdef ASM_DEBUG
// Dump a device tensor's contiguous raw bytes plus a .meta.txt, matching the
// poc_kl dump format (name, dtype, shape, stride, element_size, numel, nbytes,
// layout). Copies device->host first since asm dispatch tensors live on GPU.
static void mla_dump_mi400_debug_buffer(const std::string& dump_dir,
                                        const char* name,
                                        const aiter_tensor_t* t)
{
    if(dump_dir.empty() || t == nullptr || t->data_ptr() == nullptr)
    {
        return;
    }
    mkdir(dump_dir.c_str(), 0777);

    const size_t nbytes = t->numel() * t->element_size();
    std::vector<char> host(nbytes);
    if(nbytes > 0)
    {
        hipError_t err = hipMemcpy(host.data(), t->data_ptr(), nbytes, hipMemcpyDeviceToHost);
        if(err != hipSuccess)
        {
            std::printf("[aiter][mi400][debug] dump %s: hipMemcpy D2H failed: %s\n",
                        name,
                        hipGetErrorString(err));
            return;
        }
    }

    const std::string bin_path = dump_dir + "/" + name + ".bin";
    if(FILE* bin = std::fopen(bin_path.c_str(), "wb"))
    {
        std::fwrite(host.data(), 1, nbytes, bin);
        std::fclose(bin);
    }
    else
    {
        std::printf("[aiter][mi400][debug] failed to open dump file %s\n", bin_path.c_str());
    }

    std::string shape  = "(";
    std::string stride = "(";
    for(int i = 0; i < t->ndim; ++i)
    {
        shape += std::to_string(t->size(i));
        stride += std::to_string(t->stride(i));
        if(i + 1 < t->ndim)
        {
            shape += ",";
            stride += ",";
        }
    }
    shape += ")";
    stride += ")";

    const std::string meta_path = dump_dir + "/" + name + ".meta.txt";
    if(FILE* meta = std::fopen(meta_path.c_str(), "w"))
    {
        std::fprintf(meta, "name=%s\n", name);
        std::fprintf(meta, "dtype=%s\n", AiterDtype_to_str(t->dtype()).c_str());
        std::fprintf(meta, "shape=%s\n", shape.c_str());
        std::fprintf(meta, "stride=%s\n", stride.c_str());
        std::fprintf(meta, "element_size=%zu\n", t->element_size());
        std::fprintf(meta, "numel=%zu\n", t->numel());
        std::fprintf(meta, "nbytes=%zu\n", nbytes);
        std::fprintf(meta, "layout=contiguous raw tensor bytes (device->host copy)\n");
        std::fclose(meta);
    }
}
#endif

static void mla_decode_mi400_dispatch(
    aiter_tensor_t* Q,
    aiter_tensor_t* KV,
    aiter_tensor_t* qo_indptr,
    aiter_tensor_t* kv_indptr,
    aiter_tensor_t* kv_page_indices,
    aiter_tensor_t* kv_last_page_lens,
    aiter_tensor_t* num_kv_splits_indptr,
    aiter_tensor_t* work_meta_data,
    aiter_tensor_t* work_indptr,
    aiter_tensor_t* work_info_set,
    int max_seqlen_q,
    int page_size,
    int nhead_kv,
    float softmax_scale,
    aiter_tensor_t* splitData,
    aiter_tensor_t* splitLse,
    aiter_tensor_t* output,
    aiter_tensor_t* lse,
    aiter_tensor_t* q_scale,
    aiter_tensor_t* kv_scale,
    hipStream_t stream)
{
    (void)lse;
    const std::string arch_id = get_gpu_arch();
    AITER_CHECK(arch_id == "gfx1250", __func__, ": only supports gfx1250, got ", arch_id);

    AITER_CHECK(Q != nullptr && KV != nullptr && qo_indptr != nullptr && kv_indptr != nullptr &&
                    kv_page_indices != nullptr && kv_last_page_lens != nullptr &&
                    splitData != nullptr && splitLse != nullptr && output != nullptr,
                __func__,
                ": required tensor argument is null");
    AITER_CHECK(work_meta_data == nullptr && work_indptr == nullptr && work_info_set == nullptr &&
                    num_kv_splits_indptr != nullptr,
                __func__,
                ": gfx1250 MLA minimal smoke only supports non-persistent decode");
    AITER_CHECK(Q->is_contiguous(), __func__, ": only support Q.is_contiguous() for now");
    AITER_CHECK(Q->dtype() == AITER_DTYPE_fp8 && KV->dtype() == AITER_DTYPE_fp8,
                __func__,
                ": only supports fp8/fp8 for minimal smoke");

    const int batch        = qo_indptr->size(0) - 1;
    const int num_heads    = Q->size(1);
    const int gqa_ratio    = num_heads / nhead_kv;
    const int kv_split     = splitData->size(1);
    constexpr int bdx      = 128;
    constexpr int bdy      = 1;
    constexpr int bdz      = 1;

    AITER_CHECK(nhead_kv == 1, __func__, ": only support nhead_kv == 1 for minimal smoke");
    // Supported gfx1250 mi400 variants, matching hsa/gfx1250/mla/mla_asm.csv and
    // the poc_kl 1-threadgroup (`1tg`) kernels. Each kernel processes the full
    // flattened extent M = gqa_ratio * max_seqlen_q in a single workgroup, so
    // sub_Q == M and the grid is exactly one tile along the M (x) dimension
    // (mirrors make_launch_geometry() in poc_kl/mi400/mla/mla_helper.h). The CSV
    // heuristic lookup is keyed by (gqa_ratio, max_seqlen_q), so reject any combo
    // that has no registered kernel before touching the dispatch path.
    const bool supported_variant =
        (gqa_ratio == 16 && (max_seqlen_q == 1 || max_seqlen_q == 2 || max_seqlen_q == 4)) ||
        (gqa_ratio == 64 && max_seqlen_q == 1);
    AITER_CHECK(supported_variant,
                __func__,
                ": unsupported (gqa_ratio, max_seqlen_q) combo for gfx1250 mi400 MLA; got gqa_ratio=",
                gqa_ratio,
                " max_seqlen_q=",
                max_seqlen_q,
                " (supported: gqa16 x qSeqLen{1,2,4}, gqa64 x qSeqLen1)");
    const int sub_Q = gqa_ratio * max_seqlen_q;
    AITER_CHECK(page_size == 64, __func__, ": only support page_size == 64 for minimal smoke");
    AITER_CHECK(kv_split == 1, __func__, ": only support num_kv_splits == 1 for minimal smoke");
    AITER_CHECK(Q->size(2) == 576, __func__, ": only support Q head dim 576 for minimal smoke");
    AITER_CHECK(output->size(2) == 512, __func__, ": only support output head dim 512 for minimal smoke");
    AITER_CHECK(q_scale != nullptr && kv_scale != nullptr,
                __func__,
                ": q_scale and kv_scale are required for fp8 minimal smoke");
    AITER_CHECK(q_scale->dtype() == AITER_DTYPE_fp32 && kv_scale->dtype() == AITER_DTYPE_fp32,
                __func__,
                ": q_scale and kv_scale must be fp32");
    AITER_CHECK(static_cast<int64_t>(q_scale->numel()) >= batch &&
                    static_cast<int64_t>(kv_scale->numel()) >= batch,
                __func__,
                ": q_scale and kv_scale must have at least batch elements");
    // ABI contract with the mi400 .co: kernel writes R (split partial output)
    // as fp32 and LSE as fp32. Earlier versions of mla.py shared splitData with
    // a bf16 `o` view when num_kv_splits==1, which silently made R only half
    // the required size and produced GPU page faults. Reject that here so the
    // upstream caller is forced to allocate a real fp32 R buffer.
    AITER_CHECK(splitData->dtype() == AITER_DTYPE_fp32,
                __func__,
                ": gfx1250 mi400 MLA requires splitData (R) to be fp32; got ",
                AiterDtype_to_str(splitData->dtype()));
    AITER_CHECK(splitLse->dtype() == AITER_DTYPE_fp32,
                __func__,
                ": gfx1250 mi400 MLA requires splitLse to be fp32; got ",
                AiterDtype_to_str(splitLse->dtype()));

    CFG* config_map = &cfg_mla_asm;
    std::string kernelName =
        get_heuristic_kernel_mla("fp8", "fp8", gqa_ratio, 0, 0, 0, max_seqlen_q, arch_id, config_map, 0);
    AITER_CHECK(!kernelName.empty(), __func__, ": cannot find suitable gfx1250 kernel");

    static SynchronizedCache<std::string_view, AiterAsmKernel> mi400_impl_ptr_map;
    AiterAsmKernel* impl_ptr = nullptr;
    auto it                  = config_map->find(kernelName);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();
        impl_ptr =
            &mi400_impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });
    }
    else
    {
        AITER_CHECK(false, __func__, " not find kernel ", kernelName);
    }

    // gfx1250 mi400 dispatch: fill kernarg pack to match the layout produced by
    // poc_kl/mi400/mla/mla_execute_v3_hip.inl::execute_v3_kernel (struct
    // MlaV3HipKernelArgs). poc_kl multiplies CLI q_seq_lens by gqa_ratio
    // before filling these slots, so the kernel sees the flattened Q/head
    // extent rather than the user-visible token count.
    const int q_elem_size = Q->element_size();
    const int qk_head_dim = Q->size(2);
    const int q_seq_lens_kernel = max_seqlen_q * gqa_ratio;
    MlaMi400KernelArgs args = {};
    size_t arg_size         = sizeof(args);
    args.ptr_R              = splitData->data_ptr();
    args.ptr_LSE            = splitLse->data_ptr();
    args.ptr_Q              = Q->data_ptr();
    args.ptr_KV             = KV->data_ptr();
    args.ptr_LTP            = kv_indptr->data_ptr();
    args.ptr_LTD            = kv_page_indices->data_ptr();
    args.ptr_LTL            = kv_last_page_lens->data_ptr();
    args.scalar             = softmax_scale;
    // poc_kl host: kargs.q_seq_lens_a = (cl_int)params.q_seq_lens.
    args.q_seq_lens         = static_cast<unsigned int>(q_seq_lens_kernel);
    args.passes             = kv_split;
    // poc_kl host: stride_Q = num_kv_heads * q_seq_lens * dim_qk * sizeof(TQ).
    args.stride_Q = static_cast<unsigned int>(nhead_kv * q_seq_lens_kernel * qk_head_dim * q_elem_size);
    args.stride_page = static_cast<unsigned int>(KV->stride(0) * KV->element_size());
    args.log2_page          = static_cast<unsigned int>(log2f(static_cast<float>(page_size)));
    args.ptr_QTP            = qo_indptr->data_ptr();
    args.ptr_STP            = num_kv_splits_indptr->data_ptr();
    args.out_16_nosplit     = 0;
    args.ptr_QROPE          = q_scale->data_ptr();
    args.ptr_KVROPE         = kv_scale->data_ptr();

    const int gdx = (max_seqlen_q * gqa_ratio + sub_Q - 1) / sub_Q;
    const int gdy = batch;
    const int gdz = kv_split;

#ifdef ASM_DEBUG
    std::printf("[aiter][mi400][debug] kernelName=%s\n", kernelName.c_str());
    if(it != config_map->end())
    {
        const auto& cfg = it->second;
        std::printf("[aiter][mi400][debug] knl_name=%s co_name=%s\n",
                    cfg.knl_name.c_str(),
                    cfg.co_name.c_str());
    }
    std::printf("[aiter][mi400][debug] inputs: arch=%s batch=%d num_heads=%d nhead_kv=%d "
                "gqa_ratio=%d max_seqlen_q=%d page_size=%d qk_head_dim=%d q_elem_size=%d "
                "kv_split=%d softmax_scale=%g\n",
                arch_id.c_str(),
                batch,
                num_heads,
                nhead_kv,
                gqa_ratio,
                max_seqlen_q,
                page_size,
                qk_head_dim,
                q_elem_size,
                kv_split,
                softmax_scale);
    std::printf("[aiter][mi400][debug] tensor shapes: Q=(%ld,%ld,%ld) KV=(%ld,%ld,%ld,%ld) "
                "splitData=(%ld,%ld,%ld,%ld) splitLse=(%ld,%ld,%ld,%ld) output=(%ld,%ld,%ld)\n",
                Q->size(0),
                Q->size(1),
                Q->size(2),
                KV->size(0),
                KV->size(1),
                KV->size(2),
                KV->size(3),
                splitData->size(0),
                splitData->size(1),
                splitData->size(2),
                splitData->size(3),
                splitLse->size(0),
                splitLse->size(1),
                splitLse->size(2),
                splitLse->size(3),
                output->size(0),
                output->size(1),
                output->size(2));
    std::printf("[aiter][mi400][debug] tensor strides: Q=(%ld,%ld,%ld) KV=(%ld,%ld,%ld,%ld) "
                "splitData=(%ld,%ld,%ld,%ld) splitLse=(%ld,%ld,%ld,%ld) output=(%ld,%ld,%ld)\n",
                Q->stride(0),
                Q->stride(1),
                Q->stride(2),
                KV->stride(0),
                KV->stride(1),
                KV->stride(2),
                KV->stride(3),
                splitData->stride(0),
                splitData->stride(1),
                splitData->stride(2),
                splitData->stride(3),
                splitLse->stride(0),
                splitLse->stride(1),
                splitLse->stride(2),
                splitLse->stride(3),
                output->stride(0),
                output->stride(1),
                output->stride(2));
    std::printf("[aiter][mi400][debug] ptrs: R=%p LSE=%p Q=%p KV=%p LTP=%p LTD=%p LTL=%p "
                "QTP=%p STP=%p QROPE=%p KVROPE=%p output=%p final_lse=%p stream=%p\n",
                args.ptr_R,
                args.ptr_LSE,
                args.ptr_Q,
                args.ptr_KV,
                args.ptr_LTP,
                args.ptr_LTD,
                args.ptr_LTL,
                args.ptr_QTP,
                args.ptr_STP,
                args.ptr_QROPE,
                args.ptr_KVROPE,
                output->data_ptr(),
                lse == nullptr ? nullptr : lse->data_ptr(),
                stream);
    std::printf("[aiter][mi400][debug] kernargs: arg_size=%zu scalar=%g q_seq_lens=%u "
                "passes=%u stride_Q=%u stride_page=%u log2_page=%u out_16_nosplit=%u\n",
                arg_size,
                args.scalar,
                args.q_seq_lens,
                args.passes,
                args.stride_Q,
                args.stride_page,
                args.log2_page,
                args.out_16_nosplit);
    std::printf("[aiter][mi400][debug][arg00] offset=%zu name=ptr_R value=%p\n",
                offsetof(MlaMi400KernelArgs, ptr_R),
                args.ptr_R);
    std::printf("[aiter][mi400][debug][arg01] offset=%zu name=ptr_LSE value=%p\n",
                offsetof(MlaMi400KernelArgs, ptr_LSE),
                args.ptr_LSE);
    std::printf("[aiter][mi400][debug][arg02] offset=%zu name=ptr_Q value=%p\n",
                offsetof(MlaMi400KernelArgs, ptr_Q),
                args.ptr_Q);
    std::printf("[aiter][mi400][debug][arg03] offset=%zu name=ptr_KV value=%p\n",
                offsetof(MlaMi400KernelArgs, ptr_KV),
                args.ptr_KV);
    std::printf("[aiter][mi400][debug][arg04] offset=%zu name=ptr_LTP value=%p\n",
                offsetof(MlaMi400KernelArgs, ptr_LTP),
                args.ptr_LTP);
    std::printf("[aiter][mi400][debug][arg05] offset=%zu name=ptr_LTD value=%p\n",
                offsetof(MlaMi400KernelArgs, ptr_LTD),
                args.ptr_LTD);
    std::printf("[aiter][mi400][debug][arg06] offset=%zu name=ptr_LTL value=%p\n",
                offsetof(MlaMi400KernelArgs, ptr_LTL),
                args.ptr_LTL);
    std::printf("[aiter][mi400][debug][arg07] offset=%zu name=scalar value=%g\n",
                offsetof(MlaMi400KernelArgs, scalar),
                args.scalar);
    std::printf("[aiter][mi400][debug][arg08] offset=%zu name=q_seq_lens value=%u\n",
                offsetof(MlaMi400KernelArgs, q_seq_lens),
                args.q_seq_lens);
    std::printf("[aiter][mi400][debug][arg09] offset=%zu name=passes value=%u\n",
                offsetof(MlaMi400KernelArgs, passes),
                args.passes);
    std::printf("[aiter][mi400][debug][arg10] offset=%zu name=stride_Q value=%u\n",
                offsetof(MlaMi400KernelArgs, stride_Q),
                args.stride_Q);
    std::printf("[aiter][mi400][debug][arg11] offset=%zu name=stride_page value=%u\n",
                offsetof(MlaMi400KernelArgs, stride_page),
                args.stride_page);
    std::printf("[aiter][mi400][debug][arg12] offset=%zu name=log2_page value=%u\n",
                offsetof(MlaMi400KernelArgs, log2_page),
                args.log2_page);
    std::printf("[aiter][mi400][debug][arg13] offset=%zu name=ptr_QTP value=%p\n",
                offsetof(MlaMi400KernelArgs, ptr_QTP),
                args.ptr_QTP);
    std::printf("[aiter][mi400][debug][arg14] offset=%zu name=ptr_STP value=%p\n",
                offsetof(MlaMi400KernelArgs, ptr_STP),
                args.ptr_STP);
    std::printf("[aiter][mi400][debug][arg15] offset=%zu name=out_16_nosplit value=%u\n",
                offsetof(MlaMi400KernelArgs, out_16_nosplit),
                args.out_16_nosplit);
    std::printf("[aiter][mi400][debug][arg16] offset=%zu name=ptr_QROPE value=%p\n",
                offsetof(MlaMi400KernelArgs, ptr_QROPE),
                args.ptr_QROPE);
    std::printf("[aiter][mi400][debug][arg17] offset=%zu name=ptr_KVROPE value=%p\n",
                offsetof(MlaMi400KernelArgs, ptr_KVROPE),
                args.ptr_KVROPE);
    std::printf("[aiter][mi400][debug] launch: grid=(%d,%d,%d) block=(%d,%d,%d)\n",
                gdx,
                gdy,
                gdz,
                bdx,
                bdy,
                bdz);
    const char* skip_kernel_env = std::getenv("AITER_MLA_DEBUG_SKIP_KERNEL");
    const bool skip_kernel =
        skip_kernel_env != nullptr && skip_kernel_env[0] != '\0' &&
        !(skip_kernel_env[0] == '0' && skip_kernel_env[1] == '\0');
    if(skip_kernel)
    {
        std::printf("[aiter][mi400][debug] skipping kernel launch because AITER_MLA_DEBUG_SKIP_KERNEL=%s\n",
                    skip_kernel_env);
    }
    else
    {
        std::printf("[aiter][mi400][debug] launching kernel.\n");
    }
    std::fflush(stdout);
#endif

#ifdef ASM_DEBUG
    if(!skip_kernel)
    {
#endif
    impl_ptr->launch_kernel({&args, &arg_size, gdx, gdy, gdz, bdx, bdy, bdz, stream});
#ifdef ASM_DEBUG
        hipError_t launch_status = hipGetLastError();
        std::printf("[aiter][mi400][debug] after launch enqueue: hipGetLastError=%s (%d)\n",
                    hipGetErrorString(launch_status),
                    static_cast<int>(launch_status));
        std::printf("[aiter][mi400][debug] before hipDeviceSynchronize after stage1 launch.\n");
        std::fflush(stdout);
        hipError_t sync_status = hipDeviceSynchronize();
        std::printf("[aiter][mi400][debug] after hipDeviceSynchronize: status=%s (%d)\n",
                    hipGetErrorString(sync_status),
                    static_cast<int>(sync_status));
        std::fflush(stdout);
    }

    // Stage1 raw buffer dump (the buffers the asm kernel consumes/produces).
    // Runtime-gated by AITER_MLA_DEBUG_DUMP_DIR and dumped after the kernel
    // finished so splitData/splitLse hold real results. Matches the poc_kl
    // dump set in mla_execute_v3_hip.inl. Python-only output/final_lse and
    // token-level kv_indptr are still dumped from aiter/mla.py.
    const char* dump_env = std::getenv("AITER_MLA_DEBUG_DUMP_DIR");
    if(dump_env != nullptr && dump_env[0] != '\0')
    {
        const std::string dump_dir(dump_env);
        mla_dump_mi400_debug_buffer(dump_dir, "q", Q);
        mla_dump_mi400_debug_buffer(dump_dir, "kv_buffer", KV);
        mla_dump_mi400_debug_buffer(dump_dir, "qo_indptr", qo_indptr);
        mla_dump_mi400_debug_buffer(dump_dir, "kv_indptr", kv_indptr);
        mla_dump_mi400_debug_buffer(dump_dir, "kv_page_indices", kv_page_indices);
        mla_dump_mi400_debug_buffer(dump_dir, "kv_last_page_lens", kv_last_page_lens);
        mla_dump_mi400_debug_buffer(dump_dir, "num_kv_splits_indptr", num_kv_splits_indptr);
        mla_dump_mi400_debug_buffer(dump_dir, "q_scale", q_scale);
        mla_dump_mi400_debug_buffer(dump_dir, "kv_scale", kv_scale);
        if(!skip_kernel)
        {
            mla_dump_mi400_debug_buffer(dump_dir, "splitData", splitData);
            mla_dump_mi400_debug_buffer(dump_dir, "splitLse", splitLse);
        }
        std::printf("[aiter][mi400][debug] dumped raw stage1 buffers to %s\n", dump_dir.c_str());
        std::fflush(stdout);
    }
#endif
}

AITER_C_ITFS
void mla_decode_stage1_asm_fwd(
    aiter_tensor_t* Q,                    //   [num_seqs, num_heads, head_size]
    aiter_tensor_t* KV,                   //   [num_page, page_size, num_kv_heads, head_size] or [num_page, page_size*(nhead_kv*(kv_lora_rank+scale_dim+qk_rope_head_dim))]
    aiter_tensor_t* qo_indptr,            //   [batch_size+1]
    aiter_tensor_t* kv_indptr,            //   [batch_size+1]
    aiter_tensor_t* kv_page_indices,      //   [num_page_used]
    aiter_tensor_t* kv_last_page_lens,    //   [batch_size]
    aiter_tensor_t* num_kv_splits_indptr, //   metadata (nullable)
    aiter_tensor_t* work_meta_data,       //   metadata addr (nullable)
    aiter_tensor_t* work_indptr,          //   metadata (nullable)
    aiter_tensor_t* work_info_set,        //   [batch_size+1] (nullable)
    int max_seqlen_q,
    int page_size,
    int nhead_kv,
    float softmax_scale,
    // following are output
    aiter_tensor_t* splitData,            //   [batch_size, num_kv_splits, num_heads, v_head_dim]
    aiter_tensor_t* splitLse,             //   [batch_size, num_kv_splits, num_heads,  1]
    aiter_tensor_t* output,               //   [batch_size, num_heads, v_head_dim]
    aiter_tensor_t* lse,                  //   [batch_size, num_heads] (nullable)
    aiter_tensor_t* q_scale,              //   [1] (nullable)
    aiter_tensor_t* kv_scale,             //   [1] (nullable)
    hipStream_t stream)
{    
    int batch           = qo_indptr->size(0) - 1;
    int num_heads       = Q->size(1);
    int head_size       = Q->size(2);
    int num_kv_heads    = nhead_kv;
    int kv_split        = splitData->size(1);
    const int gqa_ratio = num_heads / num_kv_heads;

    bool persistent = (num_kv_splits_indptr == nullptr);

    const HipDeviceGuard device_guard(Q->device_id);

    std::string arch_id = get_gpu_arch();
    if(arch_id == "gfx1250")
    {
        return mla_decode_mi400_dispatch(Q,
                                         KV,
                                         qo_indptr,
                                         kv_indptr,
                                         kv_page_indices,
                                         kv_last_page_lens,
                                         num_kv_splits_indptr,
                                         work_meta_data,
                                         work_indptr,
                                         work_info_set,
                                         max_seqlen_q,
                                         page_size,
                                         nhead_kv,
                                         softmax_scale,
                                         splitData,
                                         splitLse,
                                         output,
                                         lse,
                                         q_scale,
                                         kv_scale,
                                         stream);
    }

    int stride_Q       = Q->stride(0) * Q->element_size() * max_seqlen_q;
    int stride_Page    = KV->stride(0) * KV->element_size();
    uint32_t log2_page = (uint32_t)log2f(page_size);

    KernelArgs args = {};
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
    args.s_Q_Bs      =  stride_Q;
    args.s_Bs        = stride_Page;
    args.s_log2_plen = log2_page;
    args.ptr_LSEP = nullptr;
    if (lse != nullptr)
    {
        args.ptr_LSEP = lse->data_ptr();
    }

    if (persistent)
    {
        args.out_16_nosplit = kv_split;
        args.ptr_RP = output->data_ptr();

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
        args.out_16_nosplit = 0;
        args.ptr_RP = nullptr;
        args.ptr_STP = num_kv_splits_indptr->data_ptr();
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
    CFG* config_map = &cfg_mla_asm;
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    
    int ps = persistent ? 1 : 0;
    int prefill = 0; // decode stage
    int causal = 0;
    int config_max_seqlen_q = max_seqlen_q;
    int config_gqa_ratio = gqa_ratio;
    int sub_Q = 128; // default value
    
    if(gqa_ratio == 128){
        config_max_seqlen_q = 0;
        sub_Q = 128;
        if (q_type == "bf16" && kv_type == "bf16" && arch_id == "gfx942"){
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
            } else if(persistent && max_seqlen_q == 1){
                config_max_seqlen_q = 4;
                config_gqa_ratio = 32;
                args.s_MQA = gqa_ratio;
            }
        }else if (q_type == "fp8" && kv_type == "fp8"){
            if((max_seqlen_q == 4) && persistent){
                config_max_seqlen_q = 4;
                sub_Q = 128;
            } else if((max_seqlen_q == 2) && persistent){
                config_max_seqlen_q = 2;
                sub_Q = 128;
            } else {
                AITER_CHECK(false, __func__,
                    ": fp8/fp8 with gqa_ratio=32 only supports decode_qlen=2,4 in persistent mode");
            }
        }
    } else if (gqa_ratio == 64){
        if (q_type == "bf16" && kv_type == "bf16"){
            if(!persistent){
                if(max_seqlen_q == 1){
                    config_max_seqlen_q = 1;
                } else {
                    config_max_seqlen_q = 0;
                }
                sub_Q = 64;
            }
        } else if (q_type == "fp8" && kv_type == "fp8"){
            if (persistent){
                if(max_seqlen_q == 1){
                    config_max_seqlen_q = 1;
                } else {
                    config_max_seqlen_q = 4;
                }
            } else {
                AITER_CHECK(false, __func__,
                    ": fp8/fp8 with gqa_ratio=64 only supports persistent mode");
            }
        }
    } else if (gqa_ratio == 8){
        if (q_type == "bf16" && kv_type == "bf16"){
            if(!persistent){
                config_max_seqlen_q = 1;
                sub_Q = 8;
            }
        } else if (q_type == "fp8" && kv_type == "fp8"){
            if(!persistent && max_seqlen_q == 1){
                config_max_seqlen_q = 1;
                sub_Q = 8;
            }
        }
    }

    if (arch_id == "gfx950" && q_type == "bf16" && kv_type == "bf16" && persistent && (gqa_ratio * max_seqlen_q >= 128 || gqa_ratio > 64) && gqa_ratio != 48){
        config_max_seqlen_q = 4;
        config_gqa_ratio = 32;
        args.s_MQA = gqa_ratio;
    } else if (arch_id == "gfx950" && q_type == "bf16" && kv_type == "bf16" && persistent && (gqa_ratio * max_seqlen_q >= 64 || gqa_ratio > 16) && (gqa_ratio * max_seqlen_q != 32)){
        config_max_seqlen_q = 1;
        config_gqa_ratio = 64;
        args.s_MQA = gqa_ratio;
    } else if (arch_id == "gfx950" && q_type == "fp8" && kv_type == "fp8" && persistent
               && ((gqa_ratio == 32 && max_seqlen_q == 4)
                   || (gqa_ratio == 64 && max_seqlen_q >= 2 && max_seqlen_q <= 4)
                   || (gqa_ratio == 128))){
        config_max_seqlen_q = 4;
        config_gqa_ratio = 32;
        args.s_MQA = gqa_ratio;
    }
    int lse_flag = (lse != nullptr) ? 1 : 0;
    std::string kernelName = get_heuristic_kernel_mla(q_type, kv_type, config_gqa_ratio, ps, prefill, causal, config_max_seqlen_q, arch_id, config_map, lse_flag);
    AITER_CHECK(!kernelName.empty(), __func__, ": cannot find suitable kernel");
    
    AiterAsmKernel* impl_ptr = nullptr;
    
    auto it = config_map->find(kernelName);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();

        impl_ptr =
            &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });
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


AITER_C_ITFS
void mla_prefill_ps_asm_fwd(
    aiter_tensor_t* Q,                    //  [num_seqs, num_q_heads, qk_hetad_size], fp8
    aiter_tensor_t* K,                    //   [num_page, num_kv_heads, qk_head_size], fp8
    aiter_tensor_t* V,                    //   [num_page, num_kv_heads, v_head_size], fp8
    aiter_tensor_t* qo_indptr,            //   [batch_size+1], int
    aiter_tensor_t* kv_indptr,            //   [batch_size+1], int
    aiter_tensor_t* kv_page_indices,      //   [num_page_used], int
    aiter_tensor_t* work_indptr,          //   [available_tgs+1], int (nullable)
    aiter_tensor_t* work_info_set,        //   [max_works], int (nullable)
    int max_seqlen_q,
    float softmax_scale,
    int is_causal,
    aiter_tensor_t* splitData,            //   [num_q_heads, num_seqs, max_kv_split, v_head_dim], fp32
    aiter_tensor_t* splitLse,             //   [num_q_heads, num_seqs, max_kv_split,  1], fp32
    aiter_tensor_t* output,               //   [num_seqs, num_q_heads, v_head_dim], bf16
    aiter_tensor_t* q_scale,              //   fp32, scalar (nullable)
    aiter_tensor_t* k_scale,              //   fp32, scalar (nullable)
    aiter_tensor_t* v_scale,              //   fp32, scalar (nullable)
    hipStream_t stream)
{
    int num_q_tokens  = Q->size(0);
    int num_head_q    = Q->size(1);
    int num_page      = K->size(0);
    int num_kv_heads  = K->size(1);
    int num_used_page = kv_page_indices->size(0);
    int available_tgs = 1;
    const int gqa_ratio = num_head_q / num_kv_heads;

    const HipDeviceGuard device_guard(Q->device_id);

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
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    
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

        impl_ptr =
            &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });
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
}


AITER_C_ITFS
void mla_prefill_asm_fwd(
    aiter_tensor_t* Q,                    //   [num_seqs, num_heads, head_size]
    aiter_tensor_t* KV,                   //   [num_page, page_size, num_kv_heads, head_size]
    aiter_tensor_t* qo_indptr,            //   [batch_size+1]
    aiter_tensor_t* kv_indptr,            //   [batch_size+1]
    aiter_tensor_t* kv_page_indices,      //   [num_page_used]
    aiter_tensor_t* kv_last_page_lens,    //   [batch_size]
    int max_seqlen_q,
    float softmax_scale,
    aiter_tensor_t* splitData,            //   [batch_size, num_kv_splits, num_heads, v_head_dim]
    aiter_tensor_t* splitLse,             //   [batch_size, num_kv_splits, num_heads,  1]
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

    const HipDeviceGuard device_guard(Q->device_id);

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
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    
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

        impl_ptr =
            &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });
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
}
