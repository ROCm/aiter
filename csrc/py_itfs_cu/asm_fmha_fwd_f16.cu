// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ASM FMHA forward (BF16, gfx1250) — ported from poc_kl fmha_fwd_f16.
//
// Layout: q/k/v expected in **bshd shape** ([batch, seq, head, dim]).  The
// kernel reads per-dim strides directly from the input tensor, so callers may
// pass a non-contiguous bshd-shaped view backed by sbhd / bhsd memory and the
// kernel will follow the strides correctly.  Only `tensor.stride(-1) == 1`
// (last-dim contiguous) is required, matching flash_attn_func semantics.
//
// Memory-allocation policy:
//   All tensors (q, k, v, out, lse, sink) are allocated by the Python caller.
//   This C++ entry point performs **only pointer + stride bookkeeping and
//   kernel launch** — no GPU memory allocation, no temporary tensors, no torch
//   dependency.  In particular, the AITER post-scale → pre-scale conversion
//   for `sink` (multiply by sqrt(qk_head_dim)) is the caller's responsibility:
//   pass `sink` already in the kernel's pre-scale raw-logit domain.
//
// sink slot semantics (still enforced here):
//   D64 `_rxy_sink` kernels compile ENABLE_SINK=1 → `sink` MUST be non-null.
//   D128 `_rxy`     kernels compile ENABLE_SINK=0 → `sink` slot must still be
//                   a valid non-null pointer (kernarg layout requires it), but
//                   the kernel never reads its contents.  Pass a zero buffer.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_fmha_fwd_f16_configs.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <memory>

// Kernel argument block (ABI = FmhaFwdKernelArgsBase in fmha_fwd_f16.cpp).
// kernarg_size = 528 B (33 slots × 16 B, including ptr_SINK at the end).
struct __attribute__((packed)) KernelArgs
{
    void*        ptr_O;          p2 _padO;
    void*        ptr_Q;          p2 _padQ;
    void*        ptr_K;          p2 _padK;
    void*        ptr_V;          p2 _padV;
    void*        ptr_LSE;        p2 _padLSE;
    float        scalar_f;       p3 _padSc;
    int          q_seq_len;      p3 _p0;
    int          stride_q_seq;   p3 _p1;
    int          stride_q_tg;    p3 _p2;
    int          stride_q_head;  p3 _p3;
    int          stride_q_batch; p3 _p4;
    int          gqa;            p3 _p5;
    int          stride_k_seq;   p3 _p6;
    int          stride_k_head;  p3 _p7;
    int          stride_k_batch; p3 _p8;
    int          opt;            p3 _p9;
    int          lse;            p3 _p10;
    int          kv_seq_len;     p3 _p11;
    int          qk_head_dim;    p3 _p12;
    int          v_head_dim;     p3 _p13;
    int          q_head_num;     p3 _p14;
    int          stride_v_seq;   p3 _p15;
    int          stride_v_head;  p3 _p16;
    int          stride_v_batch; p3 _p17;
    int          stride_o_seq;   p3 _p18;
    int          stride_o_head;  p3 _p19;
    int          stride_o_batch; p3 _p20;
    void*        ptr_QSeq;       p2 _padQSeq;
    void*        ptr_KSeq;       p2 _padKSeq;
    int          stride_lse_head;p3 _p21;
    void*        ptr_QSeqPad;    p2 _padQSeqPad;
    void*        ptr_KSeqPad;    p2 _padKSeqPad;
    // per-Q-head f32 sink logits (pre-scale raw domain).
    // D64 `_rxy_sink` kernels: ENABLE_SINK reads this at UCONST offset 0x200.
    // D128 `_rxy` kernels: slot must exist for kernarg_size=528 but is unused.
    void*        ptr_SINK;       p2 _padSINK;
};

// ---- helpers ---------------------------------------------------------------

// Kernel selection: only (dtype, hdim_q, hdim_v, mask) — we always use the
// _brd (border) kernel variants which are a strict superset (handle aligned
// + unaligned q_seq_len/kv_seq_len uniformly).  The csv schema therefore has
// no `border` column.
static std::string get_heuristic_kernel_fmha_fwd_f16(const std::string& dtype,
                                                     int hdim_q,
                                                     int hdim_v,
                                                     int mask_flag,
                                                     const std::string& arch_id,
                                                     CFG* cfgs)
{
    for (const auto& el : *cfgs)
    {
        if (el.first.find(arch_id) != 0) continue;
        const auto& cfg = el.second;
        if (cfg.dtype   != dtype)       continue;
        if (cfg.hdim_q  != hdim_q)      continue;
        if (cfg.hdim_v  != hdim_v)      continue;
        if (cfg.mask    != mask_flag)   continue;
        return el.first;
    }
    AITER_CHECK(false,
                "fmha_fwd_f16_asm: no kernel for dtype=", dtype,
                " hdim_q=", hdim_q, " hdim_v=", hdim_v,
                " mask=", mask_flag,
                " arch=", arch_id);
    return "";
}

// ---- main entry ------------------------------------------------------------

AITER_CTYPES_ERROR_DEF

// C ABI: every tensor is caller-allocated.  No GPU memory is allocated here;
// no torch dependency.
//
// q/k/v have **bshd shape**, i.e. q.shape = [batch, seq_q, hq, d], k/v.shape =
// [batch, seq_k, hk, d].  Kernel reads strides directly from the tensor, so
// non-contiguous bshd-shaped views backed by sbhd / bhsd memory work — only
// `stride(-1) == 1` is required.
//
// out  : [batch, q_seq_len, q_head_num, v_head_dim] bf16, last dim contiguous.
// lse  : [batch, q_head_num, q_seq_len] fp32.  Always required by kernel ABI
//        (kernel may touch ptr_LSE even when return_lse=0); pass a buffer of
//        the right size regardless of whether you read it.
// sink : [q_head_num] fp32 in the kernel's pre-scale raw-logit domain.
//        Required for D64 (ENABLE_SINK=1).  For D128 (ENABLE_SINK=0) the
//        slot must still be a valid non-null pointer of the right size, but
//        contents are ignored — pass a zero buffer.
AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    fmha_fwd_f16_asm,
    (aiter_tensor_t* q,
     aiter_tensor_t* k,
     aiter_tensor_t* v,
     aiter_tensor_t* out,
     aiter_tensor_t* lse,
     aiter_tensor_t* sink,
     float           softmax_scale,
     int             is_causal,
     int             return_lse,
     hipStream_t     stream),
    (q, k, v, out, lse, sink, softmax_scale, is_causal, return_lse, stream))
{
    // ---- arch + dtype validation ------------------------------------------
    const std::string arch_id = get_gpu_arch();
    AITER_CHECK(arch_id == "gfx1250",
                "fmha_fwd_f16_asm: only supported on gfx1250, got ", arch_id);

    AITER_CHECK(q && k && v && out && lse && sink,
                "fmha_fwd_f16_asm: q/k/v/out/lse/sink must all be non-null");
    AITER_CHECK(q->dtype() == AITER_DTYPE_bf16 &&
                k->dtype() == AITER_DTYPE_bf16 &&
                v->dtype() == AITER_DTYPE_bf16,
                "fmha_fwd_f16_asm: q/k/v must be bf16");
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16,
                "fmha_fwd_f16_asm: out must be bf16");
    AITER_CHECK(lse->dtype() == AITER_DTYPE_fp32,
                "fmha_fwd_f16_asm: lse must be fp32");
    AITER_CHECK(sink->dtype() == AITER_DTYPE_fp32,
                "fmha_fwd_f16_asm: sink must be fp32");

    AITER_CHECK(q->dim() == 4 && k->dim() == 4 && v->dim() == 4,
                "fmha_fwd_f16_asm: q/k/v must be 4-D tensors (bshd shape)");
    AITER_CHECK(q->stride(-1) == 1 && k->stride(-1) == 1 && v->stride(-1) == 1,
                "fmha_fwd_f16_asm: q/k/v must have contiguous last dim");

    // ---- dimension extraction (bshd) ---------------------------------------
    const int batch        = (int)q->size(0);
    const int q_seq_len    = (int)q->size(1);
    const int q_head_num   = (int)q->size(2);
    const int qk_head_dim  = (int)q->size(3);

    const int kv_seq_len   = (int)k->size(1);
    const int kv_head_num  = (int)k->size(2);
    const int v_head_dim   = (int)v->size(3);

    AITER_CHECK((int)k->size(0) == batch,        "fmha_fwd_f16_asm: k batch mismatch");
    AITER_CHECK((int)v->size(0) == batch,        "fmha_fwd_f16_asm: v batch mismatch");
    AITER_CHECK((int)k->size(3) == qk_head_dim,  "fmha_fwd_f16_asm: k head_dim mismatch");
    AITER_CHECK((int)v->size(1) == kv_seq_len,   "fmha_fwd_f16_asm: v seq_len mismatch with k");
    AITER_CHECK((int)v->size(2) == kv_head_num,  "fmha_fwd_f16_asm: v head_num mismatch with k");
    AITER_CHECK(q_head_num % kv_head_num == 0,   "fmha_fwd_f16_asm: q_head_num must be a multiple of kv_head_num");
    AITER_CHECK(qk_head_dim == 64 || qk_head_dim == 128,
                "fmha_fwd_f16_asm: only head_dim 64 or 128 supported, got ", qk_head_dim);
    AITER_CHECK(v_head_dim == qk_head_dim,
                "fmha_fwd_f16_asm: v_head_dim must equal qk_head_dim");

    AITER_CHECK(out->dim() == 4 &&
                (int)out->size(0) == batch    && (int)out->size(1) == q_seq_len &&
                (int)out->size(2) == q_head_num && (int)out->size(3) == v_head_dim,
                "fmha_fwd_f16_asm: out shape must be [batch, q_seq_len, q_head_num, v_head_dim]");
    AITER_CHECK(out->stride(-1) == 1,
                "fmha_fwd_f16_asm: out must have contiguous last dim");

    AITER_CHECK(lse->dim() == 3 &&
                (int)lse->size(0) == batch &&
                (int)lse->size(1) == q_head_num &&
                (int)lse->size(2) == q_seq_len,
                "fmha_fwd_f16_asm: lse shape must be [batch, q_head_num, q_seq_len]");

    AITER_CHECK(sink->dim() == 1 && (int)sink->size(0) == q_head_num,
                "fmha_fwd_f16_asm: sink must be 1-D with size q_head_num (", q_head_num, ")");

    const int gqa       = q_head_num / kv_head_num;
    const int mask_flag = is_causal ? 1 : 0;

    // ---- stride extraction (in bytes), bshd dim layout --------------------
    // bshd: dim0=b, dim1=s, dim2=h, dim3=d
    const int elem_size = (int)q->element_size();  // 2 for bf16

    const int stride_q_batch = (int)q->stride(0) * elem_size;
    const int stride_q_seq   = (int)q->stride(1) * elem_size;
    const int stride_q_head  = (int)q->stride(2) * elem_size;

    const int stride_k_batch = (int)k->stride(0) * elem_size;
    const int stride_k_seq   = (int)k->stride(1) * elem_size;
    const int stride_k_head  = (int)k->stride(2) * elem_size;

    const int stride_v_batch = (int)v->stride(0) * elem_size;
    const int stride_v_seq   = (int)v->stride(1) * elem_size;
    const int stride_v_head  = (int)v->stride(2) * elem_size;

    const int stride_o_batch = (int)out->stride(0) * elem_size;
    const int stride_o_seq   = (int)out->stride(1) * elem_size;
    const int stride_o_head  = (int)out->stride(2) * elem_size;

    const int sub_Q           = 128;  // ts_qo: Q-tile size used by all kernels
    const int stride_q_tg     = sub_Q * stride_q_seq;
    const int stride_lse_head = q_seq_len * (int)sizeof(float);  // fixed layout

    // ---- kernel args -------------------------------------------------------
    KernelArgs args = {};
    args.ptr_O           = out->data_ptr();
    args.ptr_Q           = q->data_ptr();
    args.ptr_K           = k->data_ptr();
    args.ptr_V           = v->data_ptr();
    args.ptr_LSE         = lse->data_ptr();
    args.scalar_f        = softmax_scale;
    args.q_seq_len       = q_seq_len;
    args.stride_q_seq    = stride_q_seq;
    args.stride_q_tg     = stride_q_tg;
    args.stride_q_head   = stride_q_head;
    args.stride_q_batch  = stride_q_batch;
    args.gqa             = gqa;
    args.stride_k_seq    = stride_k_seq;
    args.stride_k_head   = stride_k_head;
    args.stride_k_batch  = stride_k_batch;
    // s_opt SGPR (kernarg dword @ offset 0xF0): packs three host-side switches.
    // Bit layout must stay in lockstep with poc_kl/.../fmha_fwd_f16.cpp::opt_packed
    // and the S_OPT_BIT_* defines in BF16_FMHA_FWD_*.sp3:
    //   bit0: reverse_kv   (compile-time gated by CAS_MASK build; ignored by mask=0 kernels)
    //   bit1: double_q     (compile-time gated by DOUBLE_Q   build; ignored by non-_dq kernels)
    //   bit2: remap_xy     (must be 1 — we swap gdx/gdy at launch below)
    // 7 = 0b111 enables all three.  Safe for the four shipped _brd_rxy /
    // _cas_brd_rxy [_sink] .co binaries because bits 0/1 are compile-time
    // gated off in those builds; bit2 matches the gdx/gdy swap on launch.
    args.opt             = 7;
    args.lse             = return_lse ? 1 : 0;
    args.kv_seq_len      = kv_seq_len;
    args.qk_head_dim     = qk_head_dim;
    args.v_head_dim      = v_head_dim;
    args.q_head_num      = q_head_num;
    args.stride_v_seq    = stride_v_seq;
    args.stride_v_head   = stride_v_head;
    args.stride_v_batch  = stride_v_batch;
    args.stride_o_seq    = stride_o_seq;
    args.stride_o_head   = stride_o_head;
    args.stride_o_batch  = stride_o_batch;
    args.ptr_QSeq        = nullptr;
    args.ptr_KSeq        = nullptr;
    args.stride_lse_head = stride_lse_head;
    args.ptr_QSeqPad     = nullptr;
    args.ptr_KSeqPad     = nullptr;
    args.ptr_SINK        = sink->data_ptr();

    size_t arg_size = sizeof(args);

    // ---- kernel selection --------------------------------------------------
    // Always use the _brd (border) kernel variant: it handles both aligned
    // and unaligned q_seq_len/kv_seq_len uniformly (border path is a no-op
    // when sequences are aligned), so there's no runtime branch on alignment.
    const std::string dtype = "bf16";
    CFG* cfg_map            = &cfg_fmha_fwd_f16;
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;

    const std::string kernel_key = get_heuristic_kernel_fmha_fwd_f16(
        dtype, qk_head_dim, v_head_dim, mask_flag, arch_id, cfg_map);
    auto it = cfg_map->find(kernel_key);
    AITER_CHECK(it != cfg_map->end(),
                "fmha_fwd_f16_asm: kernel not found in CFG: ", kernel_key);

    const char* name    = it->second.knl_name.c_str();
    const char* co_name = it->second.co_name.c_str();
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        name, [&]() { return AiterAsmKernel(name, co_name); });

    // ---- launch ------------------------------------------------------------
    const int wv_tg = 4;
    const int bdx   = (wv_tg == 4) ? 128 : 256;
    const int gdx   = (q_seq_len + sub_Q - 1) / sub_Q;  // Q-tile count
    const int gdy   = q_head_num;
    const int gdz   = batch;

    // All _rxy kernels use remap_xy=1: swap gdx↔gdy at launch so that
    // bid.x indexes heads and bid.y indexes Q-tiles.
    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdy,   // launch_gdx = head count  (swapped)
                             gdx,   // launch_gdy = Q-tile count (swapped)
                             gdz,
                             bdx,
                             1,
                             1,
                             stream});
}
