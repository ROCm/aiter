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

// Kernel argument block — packed ABI (132 B = 0x84), matches
// FmhaFwdKernelArgsBase in poc_kl/mi400/fmha_fwd_f16/fmha_fwd_f16.cpp
// and the .args YAML emitted into the v8 .s patched HSA metadata.
//
// Field names mirror poc_kl: short names (d_addr / q_seqs / k_hs / ...)
// rather than the older 528-B slot-padded layout we used pre-v8.
//
//   d   = output O
//   q/k/v_seqs = stride along seq dim (bytes)
//   q/k/v_hs   = stride along head dim (bytes)
//   q/k/v_bas  = stride along batch dim (bytes)
//   q_ts       = stride between Q-tiles (sub_Q * q_seqs)
//   lse_hs     = stride per Q head for LSE (q_seq_len * 4)
//   opt        = packed switches: bit0 reverse_kv, bit1 double_q,
//                bit2 remap_xy.  We swap gdx/gdy at launch, so bit2=1.
//   sink_addr  = per-Q-head f32 sink logits (pre-scale).  Read only by
//                D64 `_rxy_sink_*` kernels (ENABLE_SINK=1).  For D128
//                the slot must still be valid (kernarg layout) but is
//                ignored; we pass a zero buffer.
#pragma pack(push, 1)
struct KernelArgs
{
    void*        d_addr;           // off 0x00  s_D_addr
    const void*  q_addr;           // off 0x08  s_Q_addr
    const void*  k_addr;           // off 0x10  s_K_addr
    const void*  v_addr;           // off 0x18  s_V_addr
    void*        lse_addr;         // off 0x20  s_LSE_addr
    float        scalar;           // off 0x28  s_scalar
    int          q_seq_len;        // off 0x2C  s_Q_seq_len
    int          q_seqs;           // off 0x30  s_Q_Seqs
    int          q_ts;             // off 0x34  s_Q_Ts
    int          q_hs;             // off 0x38  s_Q_Hs
    int          q_bas;            // off 0x3C  s_Q_BAs
    int          gqa;              // off 0x40  s_gqa
    int          k_seqs;           // off 0x44  s_K_Seqs
    int          k_hs;             // off 0x48  s_K_Hs
    int          k_bas;            // off 0x4C  s_K_BAs
    int          opt;              // off 0x50  s_opt  (bits 0..2)
    int          lse;              // off 0x54  s_LSE  (1 = write LSE)
    int          kv_seq_len;       // off 0x58  s_KV_seq_len
    int          q_head_num;       // off 0x5C  s_Q_head_num
    int          v_seqs;           // off 0x60  s_V_Seqs
    int          v_hs;             // off 0x64  s_V_Hs
    int          v_bas;            // off 0x68  s_V_BAs
    int          d_seqs;           // off 0x6C  s_D_Seqs (== O stride along seq)
    int          d_hs;             // off 0x70  s_D_Hs
    int          d_bas;            // off 0x74  s_D_BAs
    int          lse_hs;           // off 0x78  s_LSE_Hs
    void*        sink_addr;        // off 0x7C  s_SINK_buf_addr
};
#pragma pack(pop)
static_assert(sizeof(KernelArgs) == 0x84,
              "fmha_fwd_f16_asm: KernelArgs must be 132B packed (matches v8 .args)");

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
    // ABI = FmhaFwdKernelArgsBase from poc_kl/mi400/fmha_fwd_f16/fmha_fwd_f16.cpp
    // (132 B packed).  Field naming follows the poc_kl source-of-truth.
    KernelArgs args;
    memset(&args, 0, sizeof(args));
    args.d_addr     = out->data_ptr();
    args.q_addr     = q->data_ptr();
    args.k_addr     = k->data_ptr();
    args.v_addr     = v->data_ptr();
    args.lse_addr   = lse->data_ptr();
    args.scalar     = softmax_scale;
    args.q_seq_len  = q_seq_len;
    args.q_seqs     = stride_q_seq;
    args.q_ts       = stride_q_tg;
    args.q_hs       = stride_q_head;
    args.q_bas      = stride_q_batch;
    args.gqa        = gqa;
    args.k_seqs     = stride_k_seq;
    args.k_hs       = stride_k_head;
    args.k_bas      = stride_k_batch;
    // s_opt SGPR: packs three host-side switches.  Bit layout matches
    // poc_kl/.../fmha_fwd_f16.cpp::opt_packed and the S_OPT_BIT_* defines:
    //   bit0: reverse_kv   (compile-time gated by CAS_MASK build; ignored by mask=0 kernels)
    //   bit1: double_q     (compile-time gated by DOUBLE_Q   build; ignored by non-_dq kernels)
    //   bit2: remap_xy     (must be 1 — we swap gdx/gdy at launch below)
    // 7 = 0b111 enables all three.  Safe for the four shipped _rxy_brd /
    // _rxy_cas_brd [_sink] .co binaries because bits 0/1 are compile-time
    // gated off in those builds; bit2 matches the gdx/gdy swap on launch.
    args.opt        = 7;
    args.lse        = return_lse ? 1 : 0;
    args.kv_seq_len = kv_seq_len;
    args.q_head_num = q_head_num;
    args.v_seqs     = stride_v_seq;
    args.v_hs       = stride_v_head;
    args.v_bas      = stride_v_batch;
    args.d_seqs     = stride_o_seq;
    args.d_hs       = stride_o_head;
    args.d_bas      = stride_o_batch;
    args.lse_hs     = stride_lse_head;
    args.sink_addr  = sink->data_ptr();

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
