// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ASM FMHA forward (BF16, gfx1250 / MI4xx) — ported from poc_kl/mi400/fmha_fwd_f16.
//
// Layout convention (i_perm / o_perm):
//   0 = bshd  [batch, seq,  head, dim]
//   1 = bhsd  [batch, head, seq,  dim]sm_
//   2 = sbhd  [seq,   batch,head, dim]   ← default input  (i_perm=2)
//                                         ← default output (o_perm=0 → bshd)
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

static std::string get_heuristic_kernel_fmha_fwd_f16(const std::string& dtype,
                                                     int hdim_q,
                                                     int hdim_v,
                                                     int mask_flag,
                                                     int border_flag,
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
        if (cfg.border  != border_flag) continue;
        return el.first;
    }
    TORCH_CHECK(false,
                "fmha_fwd_f16_asm: no kernel for dtype=", dtype,
                " hdim_q=", hdim_q, " hdim_v=", hdim_v,
                " mask=", mask_flag, " border=", border_flag,
                " arch=", arch_id);
    return "";
}

// Extract logical dimensions from tensor shape given the perm code.
// perm: 0=bshd [b,s,h,d], 1=bhsd [b,h,s,d], 2=sbhd [s,b,h,d]
static void dims_from_perm(const at::Tensor& t, int perm,
                           int& batch, int& heads, int& seqlen, int& dim)
{
    switch (perm) {
    case 0:  // bshd
        batch = t.size(0); seqlen = t.size(1); heads = t.size(2); dim = t.size(3);
        break;
    case 1:  // bhsd
        batch = t.size(0); heads  = t.size(1); seqlen = t.size(2); dim = t.size(3);
        break;
    default: // sbhd
        seqlen = t.size(0); batch = t.size(1); heads = t.size(2); dim = t.size(3);
        break;
    }
}

// Stride (in bytes) of tensor t along its [batch, head, seq] logical dimensions
// given perm (the physical dimension ordering stored in t.shape).
static void strides_from_perm(const at::Tensor& t, int perm, int elem_size,
                               int& s_batch, int& s_head, int& s_seq)
{
    switch (perm) {
    case 0:  // bshd: dim0=b, dim1=s, dim2=h, dim3=d
        s_batch = (int)t.stride(0) * elem_size;
        s_seq   = (int)t.stride(1) * elem_size;
        s_head  = (int)t.stride(2) * elem_size;
        break;
    case 1:  // bhsd: dim0=b, dim1=h, dim2=s, dim3=d
        s_batch = (int)t.stride(0) * elem_size;
        s_head  = (int)t.stride(1) * elem_size;
        s_seq   = (int)t.stride(2) * elem_size;
        break;
    default: // sbhd: dim0=s, dim1=b, dim2=h, dim3=d
        s_seq   = (int)t.stride(0) * elem_size;
        s_batch = (int)t.stride(1) * elem_size;
        s_head  = (int)t.stride(2) * elem_size;
        break;
    }
}

// Build the expected shape vector for a tensor given logical dims and perm.
static std::vector<int64_t> shape_from_perm(int perm,
                                            int batch, int heads,
                                            int seqlen, int dim)
{
    switch (perm) {
    case 0: return {batch, seqlen, heads, dim};   // bshd
    case 1: return {batch, heads,  seqlen, dim};  // bhsd
    default:return {seqlen, batch, heads, dim};   // sbhd
    }
}

// ---- main entry ------------------------------------------------------------

// q/k/v layouts are determined by i_perm (default sbhd=2).
// Output layout is determined by o_perm (default bshd=0).
// sink: optional [q_head_num] fp32 tensor in AITER post-scale convention.
//       Internally converted to pre-scale: sink_raw = sink_user * sqrt(qk_head_dim).
std::vector<at::Tensor> fmha_fwd_f16(at::Tensor&                      q,
                                     const at::Tensor&                k,
                                     const at::Tensor&                v,
                                     float                            softmax_scale,
                                     bool                             is_causal,
                                     bool                             return_lse,
                                     int                              i_perm,
                                     int                              o_perm,
                                     std::optional<at::Tensor>        sink_,
                                     std::optional<at::Tensor>        out_)
{
    // ---- basic validation --------------------------------------------------
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "fmha_fwd_f16_asm: q/k/v must be 4-D tensors");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "fmha_fwd_f16_asm: q/k/v must be contiguous "
                "(physical layout must match i_perm=", i_perm, ")");
    TORCH_CHECK(i_perm >= 0 && i_perm <= 2, "i_perm must be 0, 1, or 2");
    TORCH_CHECK(o_perm >= 0 && o_perm <= 2, "o_perm must be 0, 1, or 2");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16,
                "fmha_fwd_f16_asm: only bf16 is supported");
    TORCH_CHECK(k.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16,
                "fmha_fwd_f16_asm: k/v must also be bf16");

    // ---- dimension extraction ----------------------------------------------
    int batch, q_head_num, q_seq_len, qk_head_dim;
    dims_from_perm(q, i_perm, batch, q_head_num, q_seq_len, qk_head_dim);

    int kv_batch, kv_head_num, kv_seq_len, kv_head_dim_check;
    dims_from_perm(k, i_perm, kv_batch, kv_head_num, kv_seq_len, kv_head_dim_check);

    int v_batch, v_heads_check, v_seq_check, v_head_dim;
    dims_from_perm(v, i_perm, v_batch, v_heads_check, v_seq_check, v_head_dim);

    TORCH_CHECK(kv_batch == batch,           "k batch mismatch");
    TORCH_CHECK(v_batch  == batch,           "v batch mismatch");
    TORCH_CHECK(kv_head_dim_check == qk_head_dim, "k head_dim mismatch");
    TORCH_CHECK(v_heads_check == kv_head_num,     "v head_num mismatch with k");
    TORCH_CHECK(v_seq_check   == kv_seq_len,      "v seq_len mismatch with k");
    TORCH_CHECK(q_head_num % kv_head_num == 0,    "q_head_num must be a multiple of kv_head_num");
    TORCH_CHECK(qk_head_dim == 64 || qk_head_dim == 128,
                "fmha_fwd_f16_asm: only head_dim 64 or 128 supported, got ", qk_head_dim);
    TORCH_CHECK(v_head_dim == qk_head_dim,
                "fmha_fwd_f16_asm: v_head_dim must equal qk_head_dim");

    const int gqa       = q_head_num / kv_head_num;
    const int mask_flag = is_causal ? 1 : 0;

    // ---- stride extraction (in bytes) from tensor's actual strides --------
    const int elem_size = q.element_size();  // 2 for bf16

    int stride_q_batch, stride_q_head, stride_q_seq;
    strides_from_perm(q, i_perm, elem_size, stride_q_batch, stride_q_head, stride_q_seq);

    int stride_k_batch, stride_k_head, stride_k_seq;
    strides_from_perm(k, i_perm, elem_size, stride_k_batch, stride_k_head, stride_k_seq);

    int stride_v_batch, stride_v_head, stride_v_seq;
    strides_from_perm(v, i_perm, elem_size, stride_v_batch, stride_v_head, stride_v_seq);

    const int sub_Q        = 128;  // ts_qo: Q-tile size used by all kernels
    const int stride_q_tg  = sub_Q * stride_q_seq;
    const int stride_lse_head = q_seq_len * (int)sizeof(float);  // fixed layout

    // ---- output allocation -------------------------------------------------
    at::Tensor out;
    if (out_.has_value())
    {
        out = out_.value();
        auto expected = shape_from_perm(o_perm, batch, q_head_num, q_seq_len, v_head_dim);
        TORCH_CHECK(out.sizes() == at::IntArrayRef(expected),
                    "fmha_fwd_f16_asm: pre-allocated out shape mismatch");
        TORCH_CHECK(out.is_contiguous() && out.scalar_type() == q.scalar_type(),
                    "fmha_fwd_f16_asm: out must be contiguous bf16");
    }
    else
    {
        auto shape = shape_from_perm(o_perm, batch, q_head_num, q_seq_len, v_head_dim);
        out = at::empty(at::IntArrayRef(shape), q.options());
    }

    int stride_o_batch, stride_o_head, stride_o_seq;
    strides_from_perm(out, o_perm, elem_size, stride_o_batch, stride_o_head, stride_o_seq);

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
    // border_flag: automatically detected from seq-len alignment.
    //   q_seq_len must be a multiple of sub_Q (128) and
    //   kv_seq_len a multiple of 256 for the non-border variants.
    const int border_flag = ((q_seq_len % 128) != 0 || (kv_seq_len % 256) != 0) ? 1 : 0;

    const std::string dtype   = "bf16";
    const std::string arch_id = get_gpu_arch();
    CFG* cfg_map              = &cfg_fmha_fwd_f16;
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;

    const std::string kernel_key = get_heuristic_kernel_fmha_fwd_f16(
        dtype, qk_head_dim, v_head_dim, mask_flag, border_flag, arch_id, cfg_map);
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
