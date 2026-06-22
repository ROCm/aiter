// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ASM Paged-Attention decode (gfx1250) — persistent / split-KV variant.
//
// Wraps the SP3 kernel `PA_DECODE_D64_1TG_4W_PS` (see the reference host file
// sched2/pa_ps.cpp).  Key properties of this kernel, distinct from the
// MI300/MI350 PA kernels in asm_pa.cu:
//   * HeadDim = 64, PageSize = 256, GQA = 8, TileQ = 32.
//   * FP8 Q **and** FP8 paged KV cache, bf16 output.
//   * Per-tensor scalar dequant scales for Q/K/V (query_scale / key_scale /
//     value_scale), passed as 1-element fp32 device tensors — NOT the
//     per-token / per-block scale tensors used by asm_pa.cu.  The attention
//     softmax scale (1/sqrt(d)) must be pre-folded into one of these scales by
//     the caller (the kernel forms scl_log2e = query_scale*key_scale*log2e).
//   * Single thread-group per work item, 4 waves of 32 lanes → bdx = 128
//     (wave32), launched persistently with grid.x = CU count.
//   * GPT-OSS style attention sink: per-Q-head fp32 sink logits in the SCALED-
//     logit domain (compared directly to (q.k)*softmax_scale, i.e. exp(sink); the
//     kernel divides by s_eff internally). Sink slot is always read by this kernel.
//
// Memory-allocation policy: every tensor is caller-allocated.  This entry point
// does only pointer + stride bookkeeping and kernel launch — no GPU memory
// allocation, no torch dependency (mirrors asm_fmha_fwd_with_sink.cu).
//
// Kernel argument block — 16-byte-slot padded ABI, 0x170 bytes total.  Offsets
// must match the s_load_dword offsets in the SP3 main (see sched2/pa_ps.cpp
// "Kernel argument layout").  softmax_scale is a by-value f32 kernarg at 0x60.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "aiter_hip_common.h"   // HipDeviceGuard, AiterAsmKernel, p2, p3, get_num_cu_func, ...
#include "asm_pa_decode_bf16_configs.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <string>

#pragma pack(push, 1)
struct KernelArgs
{
    void* ptr_O;          p2 _p0;    // 0x000  R_addr (output, bf16)
    void* ptr_Q;          p2 _p1;    // 0x010  Q_addr (FP8)
    void* ptr_K;          p2 _p2;    // 0x020  K_addr (FP8 paged)
    void* ptr_V;          p2 _p3;    // 0x030  V_addr (FP8 paged)
    void* ptr_KVIndices;  p2 _p4;    // 0x040  flattened physical page ids
    void* ptr_CL;         p2 _p5;    // 0x050  context lengths
    float softmax_scale;  p3 _p6;    // 0x060  attention softmax scale (by value)
    void* ptr_QScale;     p2 _p7;    // 0x070  per-tensor Q scale (scalar)
    void* ptr_KScale;     p2 _p8;    // 0x080  per-tensor K scale (scalar)
    void* ptr_VScale;     p2 _p9;    // 0x090  per-tensor V scale (scalar)
    unsigned int kv_nheads; p3 _p10; // 0x0A0  kv_head_num
    unsigned int Qs;      p3 _p11;   // 0x0B0  bytes per MTP layer in FP8 Q
    unsigned int Bs;      p3 _p12;   // 0x0C0  K_blk_stride
    unsigned int KVs;     p3 _p13;   // 0x0D0  K_head_stride
    unsigned int mtp;     p3 _p14;   // 0x0E0  mtp
    unsigned int GQA;     p3 _p15;   // 0x0F0  gqa_ratio
    void* ptr_QOIndptr;   p2 _p16;   // 0x100  QOIndptr (accepted but unused)
    void* ptr_KVIndptr;   p2 _p17;   // 0x110  KVIndptr
    void* ptr_WorkPtr;    p2 _p18;   // 0x120  WorkPtr
    void* ptr_WorkInfo;   p2 _p19;   // 0x130  WorkInfo
    void* ptr_SplitO;     p2 _p20;   // 0x140  SplitO
    void* ptr_SplitLSE;   p2 _p21;   // 0x150  SplitLSE
    void* ptr_Sink;       p2 _p22;   // 0x160  SinkBuffer (scaled-domain logits, exp(sink))
};
#pragma pack(pop)
static_assert(sizeof(KernelArgs) == 0x170,
              "asm_pa_decode_bf16: KernelArgs must be 0x170 B (matches SP3 s_load offsets; "
              "softmax_scale by-value at 0x60)");

// Kernel-side constants (must match SP3).
static constexpr int PA_HEAD_DIM  = 64;
static constexpr int PA_PAGE_SIZE = 256;
static constexpr int PA_GQA_RATIO = 8;
static constexpr int PA_BDX       = 128;   // 4 waves * 32 lanes (wave32)

// Kernel selection on the small set of keys captured by the csv manifest.
static std::string get_heuristic_kernel_pa_decode_bf16(const std::string& qdtype,
                                                       const std::string& kvdtype,
                                                       int hdim,
                                                       int page_size,
                                                       int gqa,
                                                       const std::string& arch_id,
                                                       CFG* cfgs)
{
    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(cfg.qdtype != qdtype)        continue;
        if(cfg.kvdtype != kvdtype)      continue;
        if(cfg.hdim != hdim)            continue;
        if(cfg.page_size != page_size)  continue;
        if(cfg.gqa != gqa)              continue;
        return el.first;
    }
    AITER_CHECK(false,
                "asm_pa_decode_bf16: no kernel for qdtype=", qdtype,
                " kvdtype=", kvdtype, " hdim=", hdim,
                " page_size=", page_size, " gqa=", gqa,
                " arch=", arch_id);
    return "";
}

AITER_CTYPES_ERROR_DEF

// C ABI: all tensors caller-allocated; no GPU allocation, no torch dependency.
//
// Q       : FP8, layout [batch, mtp_layers, kv_head, gqa, head_dim].
// K/V     : FP8 paged cache (head_dim 64, page_size 256), see pa_ps.cpp.
// out     : bf16, same logical layout as Q.
// kv_indices / kv_indptr / context_lens / qo_indptr : persistent metadata.
// work_indptr / work_info / split_o / split_lse      : persistent work split.
// q_scale / k_scale / v_scale : 1-element fp32 device tensors (per-tensor
//           dequant; per-tensor scales only).
// softmax_scale : attention scale (e.g. 1/sqrt(head_dim)), passed BY VALUE at
//           kernarg 0x60.  The kernel folds query_scale*key_scale*softmax_scale
//           *log2(e) into scl_log2e (caller does NOT pre-fold it).
// sink    : fp32 [kv_head*gqa] per-Q-head sink logits (scaled-logit domain); the
//           kernel always reads this slot, so it must be non-null.
AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    pa_decode_bf16_asm,
    (aiter_tensor_t* Q,
     aiter_tensor_t* K,
     aiter_tensor_t* V,
     aiter_tensor_t* kv_indices,
     aiter_tensor_t* context_lens,
     float           softmax_scale,
     aiter_tensor_t* q_scale,
     aiter_tensor_t* k_scale,
     aiter_tensor_t* v_scale,
     aiter_tensor_t* out,
     aiter_tensor_t* qo_indptr,
     aiter_tensor_t* kv_indptr,
     aiter_tensor_t* work_indptr,
     aiter_tensor_t* work_info,
     aiter_tensor_t* split_o,
     aiter_tensor_t* split_lse,
     aiter_tensor_t* sink,
     int             gqa,
     int             mtp,
     const char*     kernelName_,
     hipStream_t     stream),
    (Q, K, V, kv_indices, context_lens, softmax_scale, q_scale, k_scale, v_scale, out,
     qo_indptr, kv_indptr, work_indptr, work_info, split_o, split_lse, sink,
     gqa, mtp, kernelName_, stream))
{
    // ---- null safety (validate before touching the device) ----------------
    AITER_CHECK(Q && K && V && kv_indices && context_lens && q_scale && k_scale &&
                    v_scale && out && kv_indptr && sink,
                "pa_decode_bf16_asm: Q/K/V/kv_indices/context_lens/q_scale/k_scale/"
                "v_scale/out/kv_indptr/sink must all be non-null");

    HipDeviceGuard device_guard{Q->device_id};

    // ---- arch + dtype validation ------------------------------------------
    const std::string arch_id = get_gpu_arch();
    AITER_CHECK(arch_id == "gfx1250",
                "pa_decode_bf16_asm: only supported on gfx1250, got ", arch_id);

    AITER_CHECK(Q->dtype() == AITER_DTYPE_fp8, "pa_decode_bf16_asm: Q must be fp8");
    AITER_CHECK(K->dtype() == AITER_DTYPE_fp8 && V->dtype() == AITER_DTYPE_fp8,
                "pa_decode_bf16_asm: K/V must be fp8");
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16, "pa_decode_bf16_asm: out must be bf16");
    AITER_CHECK(q_scale->dtype() == AITER_DTYPE_fp32 &&
                    k_scale->dtype() == AITER_DTYPE_fp32 &&
                    v_scale->dtype() == AITER_DTYPE_fp32,
                "pa_decode_bf16_asm: q/k/v scales must be fp32");
    AITER_CHECK(sink->dtype() == AITER_DTYPE_fp32, "pa_decode_bf16_asm: sink must be fp32");

    // ---- dimensions -------------------------------------------------------
    // head_dim comes from Q's innermost dim ([batch, mtp, kv_head, gqa, head_dim]).
    // Do NOT derive it from K/V: those use the tiled paged layout
    //   K[num_pages, kv_head, head_dim/16, page, 16] / V[..., page/16, head_dim, 16]
    // (see pa_ps.cpp), whose last dim is the 16-element tile, not head_dim.
    const int kv_head_num = (int)K->size(1);
    const int head_dim    = (int)Q->size(-1);
    const int page_size   = PA_PAGE_SIZE;

    AITER_CHECK(head_dim == PA_HEAD_DIM,
                "pa_decode_bf16_asm: kernel requires head_dim=", PA_HEAD_DIM,
                ", got ", head_dim);
    AITER_CHECK(gqa == PA_GQA_RATIO,
                "pa_decode_bf16_asm: kernel requires gqa=", PA_GQA_RATIO, ", got ", gqa);

    // ---- strides (bytes) --------------------------------------------------
    const int elem_q = (int)Q->element_size();   // fp8 -> 1
    const int elem_k = (int)K->element_size();   // fp8 -> 1

    // Q_mtp_stride: bytes per MTP layer = (kv_head*gqa) * head_dim * sizeof(QT).
    const int stride_Q       = kv_head_num * gqa * head_dim * elem_q;
    // Paged K/V: K[num_pages, kv_heads, head_dim/16, page, 16] (contiguous).
    const int stride_KV_blk  = (int)K->stride(0) * elem_k;
    const int stride_KV_head = (int)K->stride(1) * elem_k;

    // ---- kernel args ------------------------------------------------------
    KernelArgs args;
    memset(&args, 0, sizeof(args));
    args.ptr_O         = out->data_ptr();
    args.ptr_Q         = Q->data_ptr();
    args.ptr_K         = K->data_ptr();
    args.ptr_V         = V->data_ptr();
    args.ptr_KVIndices = kv_indices->data_ptr();
    args.ptr_CL        = context_lens->data_ptr();
    args.softmax_scale = softmax_scale;                 // by-value f32 @ 0x60
    args.ptr_QScale    = q_scale->data_ptr();
    args.ptr_KScale    = k_scale->data_ptr();
    args.ptr_VScale    = v_scale->data_ptr();
    args.kv_nheads     = (unsigned int)kv_head_num;
    args.Qs            = (unsigned int)stride_Q;
    args.Bs            = (unsigned int)stride_KV_blk;
    args.KVs           = (unsigned int)stride_KV_head;
    args.mtp           = (unsigned int)mtp;
    args.GQA           = (unsigned int)gqa;
    args.ptr_QOIndptr  = (qo_indptr != nullptr) ? qo_indptr->data_ptr() : nullptr;
    args.ptr_KVIndptr  = kv_indptr->data_ptr();
    args.ptr_WorkPtr   = (work_indptr != nullptr) ? work_indptr->data_ptr() : nullptr;
    args.ptr_WorkInfo  = (work_info != nullptr) ? work_info->data_ptr() : nullptr;
    args.ptr_SplitO    = (split_o != nullptr) ? split_o->data_ptr() : nullptr;
    args.ptr_SplitLSE  = (split_lse != nullptr) ? split_lse->data_ptr() : nullptr;
    args.ptr_Sink      = sink->data_ptr();

    size_t arg_size = sizeof(args);

    // ---- kernel selection -------------------------------------------------
    CFG* config_map = &cfg_pa_decode_bf16;
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;

    std::string kernel_key = (kernelName_ != nullptr)
                                 ? (arch_id + std::string(kernelName_))
                                 : get_heuristic_kernel_pa_decode_bf16(
                                       "fp8", "fp8", head_dim, page_size, gqa,
                                       arch_id, config_map);
    auto it = config_map->find(kernel_key);
    AITER_CHECK(it != config_map->end(),
                "pa_decode_bf16_asm: kernel not found in CFG: ", kernel_key);

    const char* name    = it->second.knl_name.c_str();
    const char* co_name = it->second.co_name.c_str();
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        name, [&]() { return AiterAsmKernel(name, co_name); });

    // ---- launch (persistent) ----------------------------------------------
    // grid.x MUST equal the available_tgs the work metadata was built for: each
    // TG t processes work_indptr[t]..work_indptr[t+1].  The metadata is generated
    // (Python) for a specific TG count, so derive grid.x from work_indptr's length
    // (= available_tgs + 1) rather than get_num_cu_func() — those can differ on
    // gfx1250 (HIP CU count vs the count used to build the metadata), and a grid
    // larger than the metadata makes extra TGs read work_indptr OOB -> they pick up
    // garbage work and overwrite O with zeros.  Falls back to CU count if no metadata.
    const int gdx = (work_indptr != nullptr)
                        ? (int)(work_indptr->size(0) - 1)
                        : (int)get_num_cu_func();
    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx,      // gdx
                             1,        // gdy
                             1,        // gdz
                             PA_BDX,   // bdx: 4 wv32
                             1,        // bdy
                             1,        // bdz
                             stream});
}
