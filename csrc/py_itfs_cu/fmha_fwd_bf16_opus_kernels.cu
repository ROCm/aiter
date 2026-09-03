// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch entry point for the OPUS gfx950 bf16 flash-attention forward kernels.
// `fmha_fwd_bf16_opus_fwd` validates the tensors and dispatches by head dim:
//   * (D_QK,D_V) = (128,128) -> gqa_d128_kernel        (batch mode only)
//   * (D_QK,D_V) = (192,128) -> gqa_d192_v128_kernel   (batch + group / varlen)
//
// This file owns the torch-facing validation and the tensor -> stride extraction only. The
// grid shape, the causal head/tail merge and the large-KV descriptor choice live in the
// torch-free `fmha_fwd_bf16_opus_launch.h`, so the standalone C++ benchmark reaches the same
// kernels through the same decisions. Defining the IMPL macro makes this translation unit
// the owner of the device kernel instantiation.

#define FMHA_FWD_BF16_OPUS_LAUNCH_IMPL
#include "fmha_fwd_bf16_opus_launch.h"

#include "torch/fmha_fwd_bf16_opus.h"
#include "aiter_hip_common.h"

#include <ATen/hip/HIPContext.h>

namespace {

// ─── D_QK=128 / D_V=128 (symmetric) launch — logic unchanged from the original
//     fmha_fwd_hd128_bf16_opus_fwd, only moved under the shared entry point. ───
void launch_d128(at::Tensor& q,
                 at::Tensor& k,
                 at::Tensor& v,
                 at::Tensor& out,
                 bool causal,
                 float softmax_scale,
                 std::optional<at::Tensor>& lse)
{
    TORCH_CHECK(q.dim() == 4, "q must be 4-D [B, N, H, D], got ndim=", q.dim());
    TORCH_CHECK(k.dim() == 4, "k must be 4-D [B, N, H_KV, D], got ndim=", k.dim());
    TORCH_CHECK(v.dim() == 4, "v must be 4-D [B, N, H_KV, D], got ndim=", v.dim());
    TORCH_CHECK(out.dim() == 4, "out must be 4-D [B, N, H, D], got ndim=", out.dim());

    const int B    = static_cast<int>(q.size(0));
    const int N    = static_cast<int>(q.size(1));      // seqlen_q
    const int H    = static_cast<int>(q.size(2));
    const int D    = static_cast<int>(q.size(3));
    const int H_KV = static_cast<int>(k.size(2));
    const int N_KV = static_cast<int>(k.size(1));      // seqlen_kv (cross-attn: may != N)

    TORCH_CHECK(D == 128, "launch_d128 only compiles D=128, got D=", D);
    TORCH_CHECK(k.size(0) == B && v.size(0) == B, "k/v batch must equal q batch B");
    TORCH_CHECK(v.size(1) == N_KV, "k/v seqlen must match (v seqlen != k seqlen)");
    TORCH_CHECK(v.size(2) == H_KV, "k/v must share H_KV");
    TORCH_CHECK(k.size(3) == D && v.size(3) == D, "k/v head dim must equal D=128");
    TORCH_CHECK(H_KV > 0 && (H % H_KV) == 0, "H must be divisible by H_KV (GQA group)");
    TORCH_CHECK(out.size(0) == B && out.size(1) == N && out.size(2) == H && out.size(3) == D,
                "out shape must match q [B, N, H, D]");

    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1 && out.stride(3) == 1,
                "q/k/v/out must be contiguous along the head dim D");

    // 32-bit KV buffer-offset guard: extent >= 2^32 wraps the async-load soffset (silent
    // wrong output), reject instead.
    const long long kv_slice_bytes =
        (long long)N_KV * std::max(k.stride(1), v.stride(1)) * 2LL;  // bf16
    TORCH_CHECK(kv_slice_bytes < (1LL << 32),
                "OPUS D=128: KV byte extent ", kv_slice_bytes,
                " reaches the 32-bit buffer-offset limit (2^32); reduce seqlen_kv or use another backend");

    if (B == 0 || N == 0 || H == 0) return;

    fmha_fwd_bf16_opus_args args{};
    args.q_ptr    = q.data_ptr();
    args.k_ptr    = k.data_ptr();
    args.v_ptr    = v.data_ptr();
    args.o_ptr    = out.data_ptr();
    args.batch    = B;
    args.nhead    = H;
    args.nhead_k  = H_KV;
    args.seqlen_q = N;
    args.seqlen_k = N_KV;
    args.hdim_q   = D;
    args.hdim_v   = D;
    args.stride_q_b  = static_cast<int>(q.stride(0));
    args.stride_q_n  = static_cast<int>(q.stride(1));
    args.stride_q_h  = static_cast<int>(q.stride(2));
    args.stride_o_b  = static_cast<int>(out.stride(0));
    args.stride_o_n  = static_cast<int>(out.stride(1));
    args.stride_o_h  = static_cast<int>(out.stride(2));
    args.stride_k_b  = static_cast<int>(k.stride(0));
    args.stride_k_n  = static_cast<int>(k.stride(1));
    args.stride_k_h  = static_cast<int>(k.stride(2));
    args.stride_v_b  = static_cast<int>(v.stride(0));
    args.stride_v_n  = static_cast<int>(v.stride(1));
    args.stride_v_h  = static_cast<int>(v.stride(2));
    args.softmax_scale = softmax_scale;  // <= 0 picks the launcher's 1/sqrt(D) default
    args.causal        = causal;

    // Optional LSE (fp32, natural log; one value per (head, query row)). Left as nullptr
    // when absent, which the kernel reads as "skip the store".
    if (lse.has_value()) {
        const at::Tensor& l = *lse;
        TORCH_CHECK(l.device() == q.device(), "lse must be on the same device as q");
        TORCH_CHECK(l.scalar_type() == at::kFloat, "lse must be float32");
        TORCH_CHECK(l.stride(-1) == 1, "lse must be contiguous along the query dim");
        TORCH_CHECK(l.dim() == 3 && static_cast<int>(l.size(0)) == B &&
                        static_cast<int>(l.size(1)) == H && static_cast<int>(l.size(2)) == N,
                    "lse must be [B, H, N]");
        args.lse_ptr      = l.data_ptr();
        args.stride_lse_b = static_cast<int>(l.stride(0));
        args.stride_lse_h = static_cast<int>(l.stride(1));
    }

    HipDeviceGuard guard(q.device().index());
    TORCH_CHECK(fmha_fwd_bf16_opus_launch(args, at::hip::getCurrentHIPStream()),
                "OPUS D=128: the launcher rejected this shape");
}

// ─── D_QK=192 / D_V=128 (asymmetric) launch — batch + group (varlen). ───
void launch_d192_v128(at::Tensor& q,
                      at::Tensor& k,
                      at::Tensor& v,
                      at::Tensor& out,
                      bool causal,
                      float softmax_scale,
                      std::optional<at::Tensor>& lse,
                      std::optional<at::Tensor>& seqstart_q,
                      std::optional<at::Tensor>& seqstart_k,
                      std::optional<at::Tensor>& seqstart_q_pad,
                      std::optional<at::Tensor>& seqstart_k_pad,
                      int max_seqlen_q,
                      int max_seqlen_k)
{
    constexpr int D_QK = 192;
    constexpr int D_V  = 128;
    constexpr int Q_TILE_SIZE = 32, NUM_WARPS = 8;
    constexpr int Q_BLOCK = Q_TILE_SIZE * NUM_WARPS;  // 256

    const bool is_group = seqstart_q.has_value() && seqstart_q->numel() > 0;

    fmha_fwd_bf16_opus_args args{};
    args.q_ptr  = q.data_ptr();
    args.k_ptr  = k.data_ptr();
    args.v_ptr  = v.data_ptr();
    args.o_ptr  = out.data_ptr();
    args.hdim_q = D_QK;
    args.hdim_v = D_V;
    args.softmax_scale = softmax_scale;  // <= 0 picks the launcher's 1/sqrt(D_QK) default
    args.causal        = causal;

    HipDeviceGuard guard(q.device().index());

    int B, N, N_KV, H, H_KV;
    int num_q_blocks;   // only used for the empty-launch short circuit below

    if (is_group) {
        // Packed / varlen: q [total_q, H, D_QK], k [total_k, H_KV, D_QK],
        // v [total_k, H_KV, D_V], out [total_q, H, D_V]. group = num sequences.
        TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && out.dim() == 3,
                    "group mode expects packed 3-D q/k/v/out [total, H, D]");
        TORCH_CHECK(static_cast<int>(q.size(2)) == D_QK && static_cast<int>(k.size(2)) == D_QK,
                    "group mode q/k head dim must be 192");
        TORCH_CHECK(static_cast<int>(v.size(2)) == D_V && static_cast<int>(out.size(2)) == D_V,
                    "group mode v/out head dim must be 128");
        TORCH_CHECK(seqstart_q.has_value() && seqstart_k.has_value(),
                    "group mode requires seqstart_q and seqstart_k");
        H    = static_cast<int>(q.size(1));
        H_KV = static_cast<int>(k.size(1));
        TORCH_CHECK(static_cast<int>(v.size(1)) == H_KV, "group mode k/v must share H_KV");
        TORCH_CHECK(static_cast<int>(out.size(0)) == static_cast<int>(q.size(0)) &&
                    static_cast<int>(out.size(1)) == H,
                    "group mode out must be [total_q, H, D_V]");
        B    = static_cast<int>(seqstart_q->numel()) - 1;   // num groups
        TORCH_CHECK(B > 0, "group mode requires seqstart_q length >= 2");
        TORCH_CHECK(max_seqlen_q > 0 && max_seqlen_k > 0,
                    "group mode requires max_seqlen_q / max_seqlen_k > 0");
        N    = max_seqlen_q;
        N_KV = max_seqlen_k;

        // Validate the cumulative-length arrays before reinterpreting their storage as
        // int32: a wrong dtype (e.g. int64), non-contiguous layout, wrong length or a
        // host/foreign-device buffer would otherwise silently corrupt the per-group
        // offsets, or hand the kernel a pointer it cannot dereference.
        auto check_seqstart = [&](const at::Tensor& s, const char* name) {
            TORCH_CHECK(s.device() == q.device(), name, " must be on the same device as q");
            TORCH_CHECK(s.scalar_type() == at::kInt, name, " must be int32");
            TORCH_CHECK(s.dim() == 1, name, " must be 1-D");
            TORCH_CHECK(s.is_contiguous(), name, " must be contiguous");
            TORCH_CHECK(static_cast<int>(s.numel()) == B + 1, name, " length must be num_groups+1");
        };
        check_seqstart(*seqstart_q, "seqstart_q");
        check_seqstart(*seqstart_k, "seqstart_k");
        if (seqstart_q_pad.has_value()) check_seqstart(*seqstart_q_pad, "seqstart_q_pad");
        if (seqstart_k_pad.has_value()) check_seqstart(*seqstart_k_pad, "seqstart_k_pad");

        // Packed single-sequence strides (no batch stride).
        args.stride_q_b = 0; args.stride_q_n = static_cast<int>(q.stride(0));   args.stride_q_h = static_cast<int>(q.stride(1));
        args.stride_o_b = 0; args.stride_o_n = static_cast<int>(out.stride(0)); args.stride_o_h = static_cast<int>(out.stride(1));
        args.stride_k_b = 0; args.stride_k_n = static_cast<int>(k.stride(0));   args.stride_k_h = static_cast<int>(k.stride(1));
        args.stride_v_b = 0; args.stride_v_n = static_cast<int>(v.stride(0));   args.stride_v_h = static_cast<int>(v.stride(1));

        args.seqstart_q_ptr     = reinterpret_cast<const int*>(seqstart_q->data_ptr());
        args.seqstart_k_ptr     = reinterpret_cast<const int*>(seqstart_k->data_ptr());
        args.seqstart_q_pad_ptr = reinterpret_cast<const int*>(
            (seqstart_q_pad.has_value() ? *seqstart_q_pad : *seqstart_q).data_ptr());
        args.seqstart_k_pad_ptr = reinterpret_cast<const int*>(
            (seqstart_k_pad.has_value() ? *seqstart_k_pad : *seqstart_k).data_ptr());

        num_q_blocks = ceil_div(N, Q_BLOCK);          // nqb_cap from max_seqlen_q
    } else {
        // Dense batch: q/k/v/out 4-D [B, N, H, D]. Cross-attention allowed (N != N_KV).
        TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && out.dim() == 4,
                    "batch mode expects 4-D q/k/v/out [B, N, H, D]");
        B    = static_cast<int>(q.size(0));
        N    = static_cast<int>(q.size(1));
        H    = static_cast<int>(q.size(2));
        H_KV = static_cast<int>(k.size(2));
        N_KV = static_cast<int>(k.size(1));
        TORCH_CHECK(v.size(0) == B && v.size(1) == N_KV && v.size(2) == H_KV,
                    "k/v must share [B, N_KV, H_KV]");
        TORCH_CHECK(static_cast<int>(q.size(3)) == D_QK && static_cast<int>(k.size(3)) == D_QK,
                    "q/k head dim must be 192");
        TORCH_CHECK(static_cast<int>(v.size(3)) == D_V && static_cast<int>(out.size(3)) == D_V,
                    "v/out head dim must be 128");
        TORCH_CHECK(out.size(0) == B && out.size(1) == N && out.size(2) == H,
                    "out shape must match q [B, N, H, D_V]");

        args.stride_q_b = static_cast<int>(q.stride(0));   args.stride_q_n = static_cast<int>(q.stride(1));   args.stride_q_h = static_cast<int>(q.stride(2));
        args.stride_o_b = static_cast<int>(out.stride(0)); args.stride_o_n = static_cast<int>(out.stride(1)); args.stride_o_h = static_cast<int>(out.stride(2));
        args.stride_k_b = static_cast<int>(k.stride(0));   args.stride_k_n = static_cast<int>(k.stride(1));   args.stride_k_h = static_cast<int>(k.stride(2));
        args.stride_v_b = static_cast<int>(v.stride(0));   args.stride_v_n = static_cast<int>(v.stride(1));   args.stride_v_h = static_cast<int>(v.stride(2));

        num_q_blocks = ceil_div(N, Q_BLOCK);
    }

    TORCH_CHECK(H_KV > 0 && (H % H_KV) == 0, "H must be divisible by H_KV (GQA group)");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1 && out.stride(-1) == 1,
                "q/k/v/out must be contiguous along the head dim");
    if (B == 0 || H == 0 || num_q_blocks == 0) return;

    args.batch = B; args.seqlen_q = N; args.seqlen_k = N_KV; args.nhead = H; args.nhead_k = H_KV;

    // Optional LSE (fp32, natural log; one value per (head, query row)). Left as
    // nullptr when absent, which the kernel reads as "skip the store".
    if (lse.has_value()) {
        const at::Tensor& l = *lse;
        TORCH_CHECK(l.device() == q.device(), "lse must be on the same device as q");
        TORCH_CHECK(l.scalar_type() == at::kFloat, "lse must be float32");
        TORCH_CHECK(l.stride(-1) == 1, "lse must be contiguous along the query dim");
        if (is_group) {
            TORCH_CHECK(l.dim() == 2 && static_cast<int>(l.size(0)) == H &&
                            l.size(1) == q.size(0),
                        "group mode lse must be [H, total_q]");
            args.stride_lse_b = 0;
            args.stride_lse_h = static_cast<int>(l.stride(0));
        } else {
            TORCH_CHECK(l.dim() == 3 && static_cast<int>(l.size(0)) == B &&
                            static_cast<int>(l.size(1)) == H && static_cast<int>(l.size(2)) == N,
                        "batch mode lse must be [B, H, N]");
            args.stride_lse_b = static_cast<int>(l.stride(0));
            args.stride_lse_h = static_cast<int>(l.stride(1));
        }
        args.lse_ptr = l.data_ptr();
    }

    TORCH_CHECK(fmha_fwd_bf16_opus_launch(args, at::hip::getCurrentHIPStream()),
                "OPUS D_QK=192/D_V=128: the launcher rejected this shape");
}

} // namespace

void fmha_fwd_bf16_opus_fwd(at::Tensor& q,
                            at::Tensor& k,
                            at::Tensor& v,
                            at::Tensor& out,
                            bool causal,
                            float softmax_scale,
                            std::optional<at::Tensor> lse,
                            std::optional<at::Tensor> seqstart_q,
                            std::optional<at::Tensor> seqstart_k,
                            std::optional<at::Tensor> seqstart_q_pad,
                            std::optional<at::Tensor> seqstart_k_pad,
                            int max_seqlen_q,
                            int max_seqlen_k)
{
    TORCH_CHECK(q.is_cuda(), "q must be a GPU tensor");
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device() &&
                    out.device() == q.device(),
                "q/k/v/out must be on the same device");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type() &&
                    q.scalar_type() == out.scalar_type(),
                "q/k/v/out must share dtype");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "fmha_fwd_bf16_opus_fwd only supports bf16");

    const int D_QK = static_cast<int>(q.size(-1));
    const int D_V  = static_cast<int>(v.size(-1));
    const bool is_group = seqstart_q.has_value() && seqstart_q->numel() > 0;

    if (D_QK == 128 && D_V == 128) {
        TORCH_CHECK(!is_group, "OPUS D=128 kernel supports batch mode only (no varlen)");
        launch_d128(q, k, v, out, causal, softmax_scale, lse);
    } else if (D_QK == 192 && D_V == 128) {
        launch_d192_v128(q, k, v, out, causal, softmax_scale, lse,
                         seqstart_q, seqstart_k, seqstart_q_pad, seqstart_k_pad,
                         max_seqlen_q, max_seqlen_k);
    } else {
        TORCH_CHECK(false, "OPUS fwd supports (D_QK,D_V) in {(128,128),(192,128)}, got (",
                    D_QK, ",", D_V, ")");
    }
}
