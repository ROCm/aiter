// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ASM FMHA forward (MXFP8, gfx1250).
//
// This is a **dedicated** integration path, intentionally kept separate from
// the bf16 `asm_fmha_fwd_with_sink` path and from the shared `fmha_v3` path.
// The MXFP8 kernel uses its own tightly-packed kernarg ABI which carries the
// q/k/v micro-scaling (e8m0) descale pointers and which is expected to diverge
// further from the MI350 / bf16 layouts.  Keeping a separate translation unit
// and KernelArgs struct here means future MXFP8 kernarg changes never disturb
// the other paths.
//
// Layout: q/k/v are passed in **bshd shape** ([batch, seq, head, dim]).  The
// kernel reads per-dim strides directly from the input tensors, so callers may
// pass non-contiguous bshd-shaped views backed by bhsd memory (the MXFP8
// kernel is validated for bhsd memory order: stride_head > stride_seq).  Only
// `tensor.stride(-1) == 1` (last-dim contiguous) is required.
//
// Memory-allocation policy: every tensor (q, k, v, out, lse, q/k/v_scale) is
// allocated by the Python caller.  This entry point performs only pointer +
// stride bookkeeping and kernel launch -- no GPU allocation, no torch dep.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "aiter_hip_common.h"   // HipDeviceGuard, AiterAsmKernel, ...
#include "asm_fmha_fwd_mxfp8_configs.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <memory>

// Packed kernarg ABI (148 B = 0x94) -- matches the poc host
// (poc_kl/mi400/fmha_fwd_mxfp8/fmha_fwd_mxfp8.cpp :: FmhaFwdKernelArgsBase,
// branch features/yj/prefill).  This is the *new* tightly-packed layout: 8-byte
// pointers and 4-byte ints with no slot padding.  Byte offsets (in comments)
// are part of the kernel ABI emitted in the .co metadata -- do NOT reorder or
// repack.  Notable changes vs the old 560-B slot-padded layout: scale pointers
// moved to the middle (0x60-0x77), q_ts moved to 0x78, varlen QSeq/KSeq
// pointers removed.
#pragma pack(push, 1)
struct KernelArgs
{
    void*       ptr_O;          // 0x00  s_D_addr
    const void* ptr_Q;          // 0x08  s_Q_addr
    const void* ptr_K;          // 0x10  s_K_addr
    const void* ptr_V;          // 0x18  s_V_addr
    void*       ptr_LSE;        // 0x20  s_LSE_addr
    float       scalar;         // 0x28  s_scalar
    int         q_seq_len;      // 0x2C  s_Q_seq_len
    int         stride_q_seq;   // 0x30  s_Q_Seqs
    int         stride_q_head;  // 0x34  s_Q_Hs
    int         stride_q_batch; // 0x38  s_Q_BAs
    int         gqa;            // 0x3C  s_gqa
    int         stride_k_seq;   // 0x40  s_K_Seqs
    int         stride_k_head;  // 0x44  s_K_Hs
    int         stride_k_batch; // 0x48  s_K_BAs
    int         kv_seq_len;     // 0x4C  s_KV_seq_len
    int         q_head_num;     // 0x50  s_Q_head_num
    int         stride_v_seq;   // 0x54  s_V_Seqs
    int         stride_v_head;  // 0x58  s_V_Hs
    int         stride_v_batch; // 0x5C  s_V_BAs
    const void* ptr_QScale;     // 0x60  s_Q_scale_addr
    const void* ptr_KScale;     // 0x68  s_K_scale_addr
    const void* ptr_VScale;     // 0x70  s_V_scale_addr
    int         stride_q_tg;    // 0x78  s_Q_Ts
    int         opt;            // 0x7C  s_opt
    int         lse;            // 0x80  s_LSE
    int         stride_o_seq;   // 0x84  s_D_Seqs
    int         stride_o_head;  // 0x88  s_D_Hs
    int         stride_o_batch; // 0x8C  s_D_BAs
    int         stride_lse_head;// 0x90  s_LSE_Hs
};
#pragma pack(pop)
static_assert(sizeof(KernelArgs) == 0x94,
              "fmha_fwd_mxfp8_asm: KernelArgs must be 148B packed, "
              "matching the poc FmhaFwdKernelArgsBase ABI (features/yj/prefill)");

// ---- helpers ---------------------------------------------------------------

static std::string get_heuristic_kernel_fmha_fwd_mxfp8(const std::string& dtype,
                                                       int hdim_q,
                                                       int hdim_v,
                                                       int mask_flag,
                                                       const std::string& arch_id,
                                                       CFG* cfgs)
{
    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(cfg.dtype != dtype)
            continue;
        if(cfg.hdim_q != hdim_q)
            continue;
        if(cfg.hdim_v != hdim_v)
            continue;
        if(cfg.mask != mask_flag)
            continue;
        return el.first;
    }
    AITER_CHECK(false,
                "fmha_fwd_mxfp8_asm: no kernel for dtype=", dtype,
                " hdim_q=", hdim_q, " hdim_v=", hdim_v,
                " mask=", mask_flag, " arch=", arch_id);
    return "";
}

// ---- main entry ------------------------------------------------------------

AITER_CTYPES_ERROR_DEF

// C ABI: every tensor is caller-allocated.  No GPU memory is allocated here.
//
// q/k/v  : bshd shape ([batch, seq, head, dim]) fp8 (e4m3), last dim contiguous,
//          bhsd memory order (stride_head > stride_seq).
// out    : [batch, q_seq_len, q_head_num, v_head_dim] bf16, last dim contiguous.
// lse    : [batch, q_head_num, q_seq_len] fp32.  Always touched by the kernel
//          ABI, so a valid buffer must be provided even when return_lse=0.
// q/k/v_scale : 1-D micro-scaling (e8m0) descale buffers, laid out exactly as
//          the kernel expects (block_size=32 along the head_dim).  Passed
//          through verbatim as raw pointers.
AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    fmha_fwd_mxfp8_asm,
    (aiter_tensor_t* q,
     aiter_tensor_t* k,
     aiter_tensor_t* v,
     aiter_tensor_t* q_scale,
     aiter_tensor_t* k_scale,
     aiter_tensor_t* v_scale,
     aiter_tensor_t* out,
     aiter_tensor_t* lse,
     float           softmax_scale,
     int             is_causal,
     int             return_lse,
     hipStream_t     stream),
    (q, k, v, q_scale, k_scale, v_scale, out, lse,
     softmax_scale, is_causal, return_lse, stream))
{
    // ---- null safety (validate before touching the device) ----------------
    AITER_CHECK(q && k && v && out && lse,
                "fmha_fwd_mxfp8_asm: q/k/v/out/lse must all be non-null");
    AITER_CHECK(q_scale && k_scale && v_scale,
                "fmha_fwd_mxfp8_asm: q_scale/k_scale/v_scale must all be non-null");

    HipDeviceGuard device_guard{q->device_id};

    // ---- arch + dtype validation ------------------------------------------
    const std::string arch_id = get_gpu_arch();
    AITER_CHECK(arch_id == "gfx1250",
                "fmha_fwd_mxfp8_asm: only supported on gfx1250, got ", arch_id);

    AITER_CHECK(q->dtype() == AITER_DTYPE_fp8 &&
                k->dtype() == AITER_DTYPE_fp8 &&
                v->dtype() == AITER_DTYPE_fp8,
                "fmha_fwd_mxfp8_asm: q/k/v must be fp8 (e4m3)");
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16,
                "fmha_fwd_mxfp8_asm: out must be bf16");
    AITER_CHECK(lse->dtype() == AITER_DTYPE_fp32,
                "fmha_fwd_mxfp8_asm: lse must be fp32");
    AITER_CHECK(q_scale->dtype() == AITER_DTYPE_fp8_e8m0 &&
                k_scale->dtype() == AITER_DTYPE_fp8_e8m0 &&
                v_scale->dtype() == AITER_DTYPE_fp8_e8m0,
                "fmha_fwd_mxfp8_asm: q/k/v_scale must be fp8_e8m0 (float8_e8m0fnu)");

    AITER_CHECK(is_causal == 0,
                "fmha_fwd_mxfp8_asm: causal masking is not supported yet");

    AITER_CHECK(q->dim() == 4 && k->dim() == 4 && v->dim() == 4,
                "fmha_fwd_mxfp8_asm: q/k/v must be 4-D tensors (bshd shape)");
    AITER_CHECK(q->stride(-1) == 1 && k->stride(-1) == 1 && v->stride(-1) == 1,
                "fmha_fwd_mxfp8_asm: q/k/v must have contiguous last dim");

    // ---- dimension extraction (bshd) --------------------------------------
    const int batch       = (int)q->size(0);
    const int q_seq_len   = (int)q->size(1);
    const int q_head_num  = (int)q->size(2);
    const int qk_head_dim = (int)q->size(3);

    const int kv_seq_len  = (int)k->size(1);
    const int kv_head_num = (int)k->size(2);
    const int v_head_dim  = (int)v->size(3);

    AITER_CHECK((int)k->size(0) == batch,       "fmha_fwd_mxfp8_asm: k batch mismatch");
    AITER_CHECK((int)v->size(0) == batch,       "fmha_fwd_mxfp8_asm: v batch mismatch");
    AITER_CHECK((int)k->size(3) == qk_head_dim, "fmha_fwd_mxfp8_asm: k head_dim mismatch");
    AITER_CHECK((int)v->size(1) == kv_seq_len,  "fmha_fwd_mxfp8_asm: v seq_len mismatch with k");
    AITER_CHECK((int)v->size(2) == kv_head_num, "fmha_fwd_mxfp8_asm: v head_num mismatch with k");
    AITER_CHECK(q_head_num % kv_head_num == 0,
                "fmha_fwd_mxfp8_asm: q_head_num must be a multiple of kv_head_num");
    AITER_CHECK(qk_head_dim == 64 || qk_head_dim == 128,
                "fmha_fwd_mxfp8_asm: only head_dim 64 or 128 supported, got ", qk_head_dim);
    AITER_CHECK(v_head_dim == qk_head_dim,
                "fmha_fwd_mxfp8_asm: v_head_dim must equal qk_head_dim");
    AITER_CHECK(kv_seq_len % 128 == 0,
                "fmha_fwd_mxfp8_asm: kv_seq_len must be a multiple of 128, got ", kv_seq_len);

    AITER_CHECK(out->dim() == 4 &&
                (int)out->size(0) == batch      && (int)out->size(1) == q_seq_len &&
                (int)out->size(2) == q_head_num && (int)out->size(3) == v_head_dim,
                "fmha_fwd_mxfp8_asm: out shape must be [batch, q_seq_len, q_head_num, v_head_dim]");
    AITER_CHECK(out->stride(-1) == 1,
                "fmha_fwd_mxfp8_asm: out must have contiguous last dim");

    AITER_CHECK(lse->dim() == 3 &&
                (int)lse->size(0) == batch &&
                (int)lse->size(1) == q_head_num &&
                (int)lse->size(2) == q_seq_len,
                "fmha_fwd_mxfp8_asm: lse shape must be [batch, q_head_num, q_seq_len]");

    const int gqa       = q_head_num / kv_head_num;
    const int mask_flag = is_causal ? 1 : 0;

    // ---- stride extraction (bytes), bshd dim layout -----------------------
    // bshd: dim0=b, dim1=s, dim2=h, dim3=d
    const int elem_size   = (int)q->element_size();    // 1 for fp8
    const int elem_size_o = (int)out->element_size();  // 2 for bf16

    const int stride_q_seq   = (int)q->stride(1) * elem_size;
    const int stride_q_head  = (int)q->stride(2) * elem_size;
    const int stride_q_batch = (int)q->stride(0) * elem_size;

    const int stride_k_seq   = (int)k->stride(1) * elem_size;
    const int stride_k_head  = (int)k->stride(2) * elem_size;
    const int stride_k_batch = (int)k->stride(0) * elem_size;

    const int stride_v_seq   = (int)v->stride(1) * elem_size;
    const int stride_v_head  = (int)v->stride(2) * elem_size;
    const int stride_v_batch = (int)v->stride(0) * elem_size;

    const int stride_o_seq   = (int)out->stride(1) * elem_size_o;
    const int stride_o_head  = (int)out->stride(2) * elem_size_o;
    const int stride_o_batch = (int)out->stride(0) * elem_size_o;

    // ts_qo (Q-tile size) for the MXFP8 kernels is 256 (poc sub_Q=256).
    const int sub_Q           = 256;
    const int stride_q_tg     = sub_Q * stride_q_seq;
    const int stride_lse_head = q_seq_len * (int)sizeof(float);

    // ---- kernel args ------------------------------------------------------
    KernelArgs args;
    memset(&args, 0, sizeof(args));
    args.ptr_O          = out->data_ptr();
    args.ptr_Q          = q->data_ptr();
    args.ptr_K          = k->data_ptr();
    args.ptr_V          = v->data_ptr();
    args.ptr_LSE        = lse->data_ptr();
    args.scalar         = softmax_scale;
    args.q_seq_len      = q_seq_len;
    args.stride_q_seq   = stride_q_seq;
    args.stride_q_head  = stride_q_head;
    args.stride_q_batch = stride_q_batch;
    args.gqa            = gqa;
    args.stride_k_seq   = stride_k_seq;
    args.stride_k_head  = stride_k_head;
    args.stride_k_batch = stride_k_batch;
    args.kv_seq_len     = kv_seq_len;
    args.q_head_num     = q_head_num;
    args.stride_v_seq   = stride_v_seq;
    args.stride_v_head  = stride_v_head;
    args.stride_v_batch = stride_v_batch;
    args.ptr_QScale     = q_scale->data_ptr();
    args.ptr_KScale     = k_scale->data_ptr();
    args.ptr_VScale     = v_scale->data_ptr();
    args.stride_q_tg    = stride_q_tg;
    args.opt            = 0;
    args.lse            = return_lse ? 1 : 0;
    args.stride_o_seq   = stride_o_seq;
    args.stride_o_head  = stride_o_head;
    args.stride_o_batch = stride_o_batch;
    args.stride_lse_head = stride_lse_head;

    size_t arg_size = sizeof(args);

    // ---- grid dims (mirror poc; no gdx/gdy swap) --------------------------
    const int num_wv        = 4;
    const int bdx           = (num_wv == 4) ? 128 : 256;
    const int ts_qo         = sub_Q;
    const int tg_div        = (mask_flag != 0) ? 2 : 1;
    const int q_tile_count  = (q_seq_len + ts_qo - 1) / ts_qo;
    const int gdx           = (q_tile_count + tg_div - 1) / tg_div;
    const int gdy           = q_head_num;
    const int gdz           = batch;

    // ---- kernel selection -------------------------------------------------
    const std::string dtype = "mxfp8bf16";
    CFG* cfg_map            = &cfg_fmha_fwd_mxfp8;
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;

    const std::string kernel_key = get_heuristic_kernel_fmha_fwd_mxfp8(
        dtype, qk_head_dim, v_head_dim, mask_flag, arch_id, cfg_map);
    auto it = cfg_map->find(kernel_key);
    AITER_CHECK(it != cfg_map->end(),
                "fmha_fwd_mxfp8_asm: kernel not found in CFG: ", kernel_key);

    const char* name    = it->second.knl_name.c_str();
    const char* co_name = it->second.co_name.c_str();

    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        name, [&]() { return AiterAsmKernel(name, co_name); });

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx,
                             gdy,
                             gdz,
                             bdx,
                             1,
                             1,
                             stream});
}
