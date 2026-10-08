// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"

// Must follow aiter_tensor.h, which provides the HIP error-bridge dependencies.
#include "aiter_ctypes_error.h"
#include "aiter_hip_common.h"
#include "asm_fmha_v3_fwd_qnorm_rope_configs.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>
#include <string>

// fmha v3 forward (hd128, bf16, non-causal, no dropout) that applies RMSNorm and interleaved RoPE to Q on load.
//
// q is the RAW query (bshd view, any strides with stride(-1) == 1, e.g. the q columns of a fused QKV projection).
// Per Q row (token s, batch b, head h), in fp32 with one final bf16 rounding:
//   rstd = rsqrt(mean(x^2) + eps),  n = (x * rstd) * w
//   out[2i] = n[2i] * cos_i - n[2i+1] * sin_i,  out[2i+1] = n[2i+1] * cos_i + n[2i] * sin_i
// The sum of squares and the per-element operation order are those of a Triton row kernel with 32 threads per row
// (4 columns each, butterfly over xor 16, 8, 4, 2, 1), so results match it bitwise.
// The rows of the sequence belong to two streams with their own norm weight and RoPE table: 256-row Q tiles
// [0, ntile_a) use (w_a, tab_a), the rest (w_b, tab_b), with the token index taken relative to the stream's start.
// Tables are compact bf16 [S_stream * B, 128], row s_local * B + b = [cos_0 .. cos_63 | sin_0 .. sin_63].
// Outputs: out / lse as the plain v3 forward; q_rstd fp32 [S * B * H] at (s * B + b) * H + h; optionally q_n, the
// normalized and rotated q, into an sbhd-contiguous [S, B, H, 128] buffer (what the v3 backward then takes as q).
namespace {

struct __attribute__((packed)) Slot64
{
    const void* p;
    uint64_t pad;
};
struct __attribute__((packed)) Slot32
{
    uint32_t v;
    uint32_t pad[3];
};

// the v3 forward's kernel arguments (every field in a 16-byte slot) followed by the Q-transform fields
struct __attribute__((packed)) FwdQnormRopeArgs
{
    Slot64 o, q, k, v, lse;
    Slot32 scalar, seq_len, q_seq_stride, q_tile_stride, q_head_stride, q_batch_stride, gqa;
    Slot32 k_seq_stride, k_head_stride, k_batch_stride, opt, has_lse, kv_seq_len, qk_head_dim, v_head_dim,
        q_head_num;
    Slot32 v_seq_stride, v_head_stride, v_batch_stride, o_seq_stride, o_head_stride, o_batch_stride;
    Slot64 qseq, kseq; // group mode only (unused here)
    Slot32 lse_head_stride;
    Slot64 qseq_padding, kseq_padding;
    // Q transform (byte offset 512)
    const void* w_a;
    const void* tab_a;
    const void* w_b;
    const void* tab_b;
    void* q_rstd;
    void* q_n;
    float eps;
    uint32_t ntile_a;
    uint32_t tab_token_stride;  // bytes per token in a table: B * 256
    uint32_t rstd_token_stride; // bytes per token in q_rstd: B * H * 4
    uint32_t qn_token_stride;   // bytes per token in q_n: B * H * 256
    uint32_t pad[3];
};
static_assert(offsetof(FwdQnormRopeArgs, scalar) == 80, "kernel argument layout");
static_assert(offsetof(FwdQnormRopeArgs, lse_head_stride) == 464, "kernel argument layout");
static_assert(offsetof(FwdQnormRopeArgs, w_a) == 512, "kernel argument layout");
static_assert(offsetof(FwdQnormRopeArgs, eps) == 560, "kernel argument layout");
static_assert(sizeof(FwdQnormRopeArgs) == 592, "kernel argument layout");

constexpr int kTile = 256;
constexpr int kHeadDim = 128;

const fmha_v3_fwd_qnorm_ropeConfig* find_row(const std::string& arch, int qn)
{
    for(const auto& kv : cfg_fmha_fwd_qnorm_rope)
        if(kv.second.arch == arch && kv.second.qn == qn)
            return &kv.second;
    return nullptr;
}
} // namespace

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    fmha_v3_fwd_qnorm_rope_asm,
    (aiter_tensor_t* q,      // raw q, bshd view [B, S, H, 128]
     aiter_tensor_t* k,      // bshd view [B, S, H, 128]
     aiter_tensor_t* v,      // bshd view [B, S, H, 128]
     aiter_tensor_t* out,    // bshd view [B, S, H, 128] bf16
     aiter_tensor_t* lse,    // [B, H, S] fp32 contiguous
     aiter_tensor_t* w_a,    // [128] bf16
     aiter_tensor_t* tab_a,  // [ntile_a * 256 * B, 128] bf16 contiguous
     aiter_tensor_t* w_b,    // [128] bf16
     aiter_tensor_t* tab_b,  // [(S - ntile_a * 256) * B, 128] bf16 contiguous
     aiter_tensor_t* q_rstd, // [S * B * H] fp32
     aiter_tensor_t* q_n,    // optional: bshd view of an sbhd-contiguous [S, B, H, 128] bf16 buffer
     float softmax_scale,
     float eps,
     int64_t ntile_a,
     hipStream_t stream),
    (q, k, v, out, lse, w_a, tab_a, w_b, tab_b, q_rstd, q_n, softmax_scale, eps, ntile_a, stream))
{
    AITER_CHECK(q && k && v && out && lse && w_a && tab_a && w_b && tab_b && q_rstd,
                __func__,
                " only q_n may be null");
    const HipDeviceGuard device_guard(q->device_id);
    AITER_CHECK(q->dim() == 4 && k->dim() == 4 && v->dim() == 4 && out->dim() == 4, __func__, " q/k/v/out are bshd");
    const int64_t B = q->size(0), S = q->size(1), H = q->size(2), D = q->size(3);
    for(const aiter_tensor_t* t : {k, v, out})
        AITER_CHECK(t->size(0) == B && t->size(1) == S && t->size(2) == H && t->size(3) == D,
                    __func__,
                    " q, k, v and out must have the same [B, S, H, D] shape");
    AITER_CHECK(D == kHeadDim, __func__, " head dim must be 128");
    AITER_CHECK(S > 0 && S % kTile == 0, __func__, " seq len must be a positive multiple of 256");
    AITER_CHECK(ntile_a >= 0 && ntile_a <= S / kTile, __func__, " ntile_a must be in [0, S / 256]");
    for(const aiter_tensor_t* t : {q, k, v, out})
        AITER_CHECK(t->dtype() == AITER_DTYPE_bf16 && t->stride(3) == 1,
                    __func__,
                    " q/k/v/out must be bf16 with a contiguous last dim");
    for(const aiter_tensor_t* t : {w_a, w_b})
        AITER_CHECK(t->dtype() == AITER_DTYPE_bf16 && t->numel() == kHeadDim && t->is_contiguous(),
                    __func__,
                    " norm weights must be contiguous bf16 [128]");
    AITER_CHECK(tab_a->dtype() == AITER_DTYPE_bf16 && tab_b->dtype() == AITER_DTYPE_bf16 && tab_a->dim() == 2 &&
                    tab_b->dim() == 2 && tab_a->is_contiguous() && tab_b->is_contiguous() &&
                    tab_a->size(1) == kHeadDim && tab_b->size(1) == kHeadDim,
                __func__,
                " RoPE tables must be contiguous bf16 [rows, 128]");
    const int64_t s_a = ntile_a * kTile;
    if(s_a > 0)
        AITER_CHECK(tab_a->size(0) >= s_a * B, __func__, " tab_a has fewer than ntile_a * 256 * B rows");
    if(S - s_a > 0)
        AITER_CHECK(tab_b->size(0) >= (S - s_a) * B, __func__, " tab_b has fewer than (S - ntile_a * 256) * B rows");
    AITER_CHECK(lse->dtype() == AITER_DTYPE_fp32 && lse->dim() == 3 && lse->size(0) == B && lse->size(1) == H &&
                    lse->size(2) == S && lse->is_contiguous(),
                __func__,
                " lse must be contiguous fp32 [B, H, S]");
    AITER_CHECK(q_rstd->dtype() == AITER_DTYPE_fp32 && q_rstd->numel() == S * B * H && q_rstd->is_contiguous(),
                __func__,
                " q_rstd must be contiguous fp32 [S * B * H]");
    if(q_n)
        AITER_CHECK(q_n->dtype() == AITER_DTYPE_bf16 && q_n->dim() == 4 && q_n->size(0) == B && q_n->size(1) == S &&
                        q_n->size(2) == H && q_n->size(3) == D && q_n->stride(3) == 1 && q_n->stride(2) == D &&
                        q_n->stride(0) == H * D && q_n->stride(1) == B * H * D,
                    __func__,
                    " q_n must be a bshd view of an sbhd-contiguous [S, B, H, 128] bf16 buffer");

    const std::string arch = get_gpu_arch();
    const fmha_v3_fwd_qnorm_ropeConfig* cfg = find_row(arch, q_n ? 1 : 0);
    AITER_CHECK(cfg != nullptr, __func__, " no kernel for arch ", arch);

    FwdQnormRopeArgs a;
    std::memset(&a, 0, sizeof(a));
    const int64_t bpe = 2;
    a.o.p = out->ptr, a.q.p = q->ptr, a.k.p = k->ptr, a.v.p = v->ptr, a.lse.p = lse->ptr;
    std::memcpy(&a.scalar.v, &softmax_scale, 4);
    a.seq_len.v        = static_cast<uint32_t>(S);
    a.q_seq_stride.v   = static_cast<uint32_t>(q->stride(1) * bpe);
    a.q_tile_stride.v  = static_cast<uint32_t>(cfg->ts_qo * q->stride(1) * bpe);
    a.q_head_stride.v  = static_cast<uint32_t>(q->stride(2) * bpe);
    a.q_batch_stride.v = static_cast<uint32_t>(q->stride(0) * bpe);
    a.gqa.v            = 1;
    a.k_seq_stride.v   = static_cast<uint32_t>(k->stride(1) * bpe);
    a.k_head_stride.v  = static_cast<uint32_t>(k->stride(2) * bpe);
    a.k_batch_stride.v = static_cast<uint32_t>(k->stride(0) * bpe);
    a.opt.v            = 5;
    a.has_lse.v        = 1;
    a.kv_seq_len.v     = static_cast<uint32_t>(S);
    a.qk_head_dim.v    = static_cast<uint32_t>(D);
    a.v_head_dim.v     = static_cast<uint32_t>(D);
    a.q_head_num.v     = static_cast<uint32_t>(H);
    a.v_seq_stride.v   = static_cast<uint32_t>(v->stride(1) * bpe);
    a.v_head_stride.v  = static_cast<uint32_t>(v->stride(2) * bpe);
    a.v_batch_stride.v = static_cast<uint32_t>(v->stride(0) * bpe);
    a.o_seq_stride.v   = static_cast<uint32_t>(out->stride(1) * bpe);
    a.o_head_stride.v  = static_cast<uint32_t>(out->stride(2) * bpe);
    a.o_batch_stride.v = static_cast<uint32_t>(out->stride(0) * bpe);
    a.lse_head_stride.v = static_cast<uint32_t>(S * 4);
    a.w_a = w_a->ptr, a.tab_a = tab_a->ptr, a.w_b = w_b->ptr, a.tab_b = tab_b->ptr;
    a.q_rstd = q_rstd->ptr;
    a.q_n    = q_n ? q_n->ptr : nullptr;
    a.eps    = eps;
    a.ntile_a           = static_cast<uint32_t>(ntile_a);
    a.tab_token_stride  = static_cast<uint32_t>(B * kHeadDim * 2);
    a.rstd_token_stride = static_cast<uint32_t>(B * H * 4);
    a.qn_token_stride   = static_cast<uint32_t>(B * H * kHeadDim * 2);

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* kernel = &impl_ptr_map.get_or_create(
        cfg->knl_name, [&]() { return AiterAsmKernel(cfg->knl_name.c_str(), cfg->co_name.c_str()); });
    size_t arg_size = sizeof(a);
    kernel->launch_kernel({&a,
                           &arg_size,
                           static_cast<int>(S / cfg->ts_qo),
                           static_cast<int>(H),
                           static_cast<int>(B),
                           512,
                           1,
                           1,
                           stream});
}
