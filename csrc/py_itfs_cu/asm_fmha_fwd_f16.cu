// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ASM FMHA forward (BF16, gfx1250 / MI4xx) — ported from poc_kl/mi400/fmha_fwd_f16.
//
// Layout: q/k/v expected in **bshd shape** ([batch, seq, head, dim]).  The
// kernel reads per-dim strides directly from the input tensor, so callers may
// pass a non-contiguous bshd-shaped view backed by sbhd / bhsd memory and the
// kernel will follow the strides correctly.  Only `tensor.stride(-1) == 1`
// (last-dim contiguous) is required, matching flash_attn_func semantics.
//
// sink convention (AITER / CK-Tile post-scale):
//   The user passes sink in the same domain as Q*K^T * softmax_scale (post-scale).
//   The kernel expects pre-scale raw logits.  This file converts:
//       sink_raw = sink_user * sqrt(qk_head_dim)
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <memory>
#include <unordered_map>

#include "aiter_hip_common.h"
#include "asm_fmha_fwd_f16_configs.hpp"

namespace aiter {
namespace torch_itfs {

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
    TORCH_CHECK(false,
                "fmha_fwd_f16_asm: no kernel for dtype=", dtype,
                " hdim_q=", hdim_q, " hdim_v=", hdim_v,
                " mask=", mask_flag,
                " arch=", arch_id);
    return "";
}

// ---- main entry ------------------------------------------------------------

// API contract: q/k/v have **bshd shape**, i.e. q.shape = [batch, seq_q, hq, d],
// k/v.shape = [batch, seq_k, hk, d].  The kernel reads strides directly from
// `tensor.stride(...)`, so the underlying memory layout is whatever the user
// arranged — they may pass a non-contiguous bshd-shaped view of an sbhd / bhsd
// allocation, and the kernel will follow strides correctly.  Only `stride(-1)
// == 1` (last dim contiguous) is required, matching flash_attn_func.
//
// sink: optional [q_head_num] fp32 tensor in AITER post-scale convention.
//       Internally converted to pre-scale: sink_raw = sink_user * sqrt(qk_head_dim).
std::vector<at::Tensor> fmha_fwd_f16(at::Tensor&                      q,
                                     const at::Tensor&                k,
                                     const at::Tensor&                v,
                                     float                            softmax_scale,
                                     bool                             is_causal,
                                     bool                             return_lse,
                                     std::optional<at::Tensor>        sink_,
                                     std::optional<at::Tensor>        out_)
{
    // ---- basic validation --------------------------------------------------
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "fmha_fwd_f16_asm: q/k/v must be 4-D tensors (bshd shape)");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1,
                "fmha_fwd_f16_asm: q/k/v must have contiguous last dim");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16,
                "fmha_fwd_f16_asm: only bf16 is supported");
    TORCH_CHECK(k.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16,
                "fmha_fwd_f16_asm: k/v must also be bf16");

    // ---- dimension extraction (bshd) ---------------------------------------
    const int batch        = (int)q.size(0);
    const int q_seq_len    = (int)q.size(1);
    const int q_head_num   = (int)q.size(2);
    const int qk_head_dim  = (int)q.size(3);

    const int kv_seq_len   = (int)k.size(1);
    const int kv_head_num  = (int)k.size(2);
    const int v_head_dim   = (int)v.size(3);

    TORCH_CHECK((int)k.size(0) == batch,        "k batch mismatch");
    TORCH_CHECK((int)v.size(0) == batch,        "v batch mismatch");
    TORCH_CHECK((int)k.size(3) == qk_head_dim,  "k head_dim mismatch");
    TORCH_CHECK((int)v.size(1) == kv_seq_len,   "v seq_len mismatch with k");
    TORCH_CHECK((int)v.size(2) == kv_head_num,  "v head_num mismatch with k");
    TORCH_CHECK(q_head_num % kv_head_num == 0,  "q_head_num must be a multiple of kv_head_num");
    TORCH_CHECK(qk_head_dim == 64 || qk_head_dim == 128,
                "fmha_fwd_f16_asm: only head_dim 64 or 128 supported, got ", qk_head_dim);
    TORCH_CHECK(v_head_dim == qk_head_dim,
                "fmha_fwd_f16_asm: v_head_dim must equal qk_head_dim");

    const int gqa       = q_head_num / kv_head_num;
    const int mask_flag = is_causal ? 1 : 0;

    // ---- stride extraction (in bytes), bshd dim layout --------------------
    // bshd: dim0=b, dim1=s, dim2=h, dim3=d
    const int elem_size = q.element_size();  // 2 for bf16

    const int stride_q_batch = (int)q.stride(0) * elem_size;
    const int stride_q_seq   = (int)q.stride(1) * elem_size;
    const int stride_q_head  = (int)q.stride(2) * elem_size;

    const int stride_k_batch = (int)k.stride(0) * elem_size;
    const int stride_k_seq   = (int)k.stride(1) * elem_size;
    const int stride_k_head  = (int)k.stride(2) * elem_size;

    const int stride_v_batch = (int)v.stride(0) * elem_size;
    const int stride_v_seq   = (int)v.stride(1) * elem_size;
    const int stride_v_head  = (int)v.stride(2) * elem_size;

    const int sub_Q        = 128;  // ts_qo: Q-tile size used by all kernels
    const int stride_q_tg  = sub_Q * stride_q_seq;
    const int stride_lse_head = q_seq_len * (int)sizeof(float);  // fixed layout

    // ---- output allocation (bshd) -----------------------------------------
    at::Tensor out;
    if (out_.has_value())
    {
        out = out_.value();
        TORCH_CHECK(out.dim() == 4 &&
                    (int)out.size(0) == batch    && (int)out.size(1) == q_seq_len &&
                    (int)out.size(2) == q_head_num && (int)out.size(3) == v_head_dim,
                    "fmha_fwd_f16_asm: pre-allocated out shape must be "
                    "[batch, q_seq_len, q_head_num, v_head_dim]");
        TORCH_CHECK(out.stride(-1) == 1 && out.scalar_type() == q.scalar_type(),
                    "fmha_fwd_f16_asm: out must have contiguous last dim and same dtype as q");
    }
    else
    {
        out = at::empty({batch, q_seq_len, q_head_num, v_head_dim}, q.options());
    }

    const int stride_o_batch = (int)out.stride(0) * elem_size;
    const int stride_o_seq   = (int)out.stride(1) * elem_size;
    const int stride_o_head  = (int)out.stride(2) * elem_size;

    // ---- LSE allocation (fixed layout [batch, q_head_num, q_seq_len] fp32) -
    // Always allocate even when not returned: the kernel may access ptr_LSE.
    at::Tensor lse = at::empty({batch, q_head_num, q_seq_len},
                               q.options().dtype(at::kFloat));

    // ---- sink buffer -------------------------------------------------------
    // D64 `_rxy_sink` kernels (ENABLE_SINK=1): ptr_SINK is actively read.
    //   Sink must be provided for D64; passing a zero buffer silently passes
    //   logit=0 through the sink path (which still exercises the code path but
    //   is numerically equivalent to a very negative logit after max-subtraction).
    //   We therefore REQUIRE an explicit sink for D64 so callers are aware.
    //
    // D128 `_rxy` kernels (ENABLE_SINK=0): ptr_SINK is compiled out; the slot
    //   must still be a valid non-null pointer, but values are irrelevant.
    //   Zeros are used when no sink is supplied for D128.
    //
    // sink_ is in AITER post-scale convention (same domain as Q·K^T * scale).
    // Convert to pre-scale for kernel: sink_raw = sink_user * sqrt(qk_head_dim).
    at::Tensor sink;
    if (sink_.has_value())
    {
        TORCH_CHECK(sink_.value().dim() == 1 && sink_.value().size(0) == q_head_num,
                    "fmha_fwd_f16_asm: sink must be 1-D with size q_head_num (", q_head_num, ")");
        TORCH_CHECK(sink_.value().scalar_type() == at::kFloat,
                    "fmha_fwd_f16_asm: sink must be fp32");
        // AITER post-scale → pre-scale: multiply by sqrt(qk_head_dim)
        float pre_scale = std::sqrt(static_cast<float>(qk_head_dim));
        sink = (sink_.value() * pre_scale).contiguous();
    }
    else if (qk_head_dim == 64)
    {
        // D64 _rxy_sink kernels always compute the sink path (ENABLE_SINK=1).
        // Require an explicit sink so callers know it is active.
        TORCH_CHECK(false,
                    "fmha_fwd_f16_asm: D64 (_rxy_sink) kernels require an explicit `sink` "
                    "tensor of shape [q_head_num]=", q_head_num, " fp32 (AITER post-scale "
                    "convention).  Pass `sink=torch.zeros(q_head_num, dtype=torch.float32)` "
                    "if you want a zero-logit sink.");
    }
    else
    {
        // D128 _rxy kernels: ENABLE_SINK=0, ptr_SINK is ignored by the kernel.
        sink = at::zeros({q_head_num}, q.options().dtype(at::kFloat));
    }

    // ---- kernel args -------------------------------------------------------
    KernelArgs args = {};
    args.ptr_O           = out.data_ptr();
    args.ptr_Q           = q.data_ptr();
    args.ptr_K           = k.data_ptr();
    args.ptr_V           = v.data_ptr();
    args.ptr_LSE         = lse.data_ptr();
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
    args.opt             = 0;
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
    args.ptr_SINK        = sink.data_ptr();

    size_t arg_size = sizeof(args);

    // ---- kernel selection --------------------------------------------------
    // Always use the _brd (border) kernel variant: it handles both aligned
    // and unaligned q_seq_len/kv_seq_len uniformly (border path is a no-op
    // when sequences are aligned), so there's no runtime branch on alignment.
    const std::string dtype   = "bf16";
    const std::string arch_id = get_gpu_arch();
    CFG* cfg_map              = &cfg_fmha_fwd_f16;
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;

    const std::string kernel_key = get_heuristic_kernel_fmha_fwd_f16(
        dtype, qk_head_dim, v_head_dim, mask_flag, arch_id, cfg_map);
    auto it = cfg_map->find(kernel_key);
    TORCH_CHECK(it != cfg_map->end(),
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
    auto stream = at::hip::getCurrentHIPStream().stream();

    // ---- DEBUG DUMP -------------------------------------------------------
    fprintf(stderr,
        "\n[fmha_fwd_f16 DEBUG] kernel_key=%s  co=%s  arg_size=%zu\n"
        "  KernelArgs:\n"
        "    ptr_O=%p  ptr_Q=%p  ptr_K=%p  ptr_V=%p  ptr_LSE=%p\n"
        "    scalar_f=%g\n"
        "    q_seq_len=%d  kv_seq_len=%d  q_head_num=%d  gqa=%d\n"
        "    qk_head_dim=%d  v_head_dim=%d  opt=%d  lse=%d\n"
        "    stride_q_seq=%d  stride_q_tg=%d  stride_q_head=%d  stride_q_batch=%d\n"
        "    stride_k_seq=%d  stride_k_head=%d  stride_k_batch=%d\n"
        "    stride_v_seq=%d  stride_v_head=%d  stride_v_batch=%d\n"
        "    stride_o_seq=%d  stride_o_head=%d  stride_o_batch=%d\n"
        "    ptr_QSeq=%p  ptr_KSeq=%p  stride_lse_head=%d\n"
        "    ptr_QSeqPad=%p  ptr_KSeqPad=%p  ptr_SINK=%p\n"
        "  Launch dims (after rxy swap):  gdx(head)=%d  gdy(Qtile)=%d  gdz(batch)=%d\n"
        "                                 bdx=%d  bdy=1  bdz=1\n"
        "  Pre-swap:                      gdx(Qtile)=%d  gdy(head)=%d  gdz(batch)=%d\n",
        kernel_key.c_str(), co_name, arg_size,
        args.ptr_O, args.ptr_Q, args.ptr_K, args.ptr_V, args.ptr_LSE,
        args.scalar_f,
        args.q_seq_len, args.kv_seq_len, args.q_head_num, args.gqa,
        args.qk_head_dim, args.v_head_dim, args.opt, args.lse,
        args.stride_q_seq, args.stride_q_tg, args.stride_q_head, args.stride_q_batch,
        args.stride_k_seq, args.stride_k_head, args.stride_k_batch,
        args.stride_v_seq, args.stride_v_head, args.stride_v_batch,
        args.stride_o_seq, args.stride_o_head, args.stride_o_batch,
        args.ptr_QSeq, args.ptr_KSeq, args.stride_lse_head,
        args.ptr_QSeqPad, args.ptr_KSeqPad, args.ptr_SINK,
        gdy, gdx, gdz, bdx,
        gdx, gdy, gdz);
    fflush(stderr);

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdy,   // launch_gdx = head count  (swapped)
                             gdx,   // launch_gdy = Q-tile count (swapped)
                             gdz,
                             bdx,
                             1,
                             1,
                             stream});

    std::vector<at::Tensor> ret;
    ret.push_back(out);
    if (return_lse) ret.push_back(lse);
    return ret;
}

} // namespace torch_itfs
} // namespace aiter
