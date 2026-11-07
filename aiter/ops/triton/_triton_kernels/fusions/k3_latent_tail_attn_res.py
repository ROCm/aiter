# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Kernel body for the fused Kimi-K3 latent-tail up-proj GEMM + AttnRes read.
# See aiter/ops/triton/fusions/k3_latent_tail_attn_res.py for the contract and
# OSS attribution (sgl-project/sglang #44985/#44988, ROCm/aiter #4610).

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_repr = make_kernel_repr(
    "_latent_tail_attn_res_kernel",
    ["BLOCK_N", "BLOCK_K", "BD", "NVB", "L_CAND"],
)


@triton.jit(repr=_repr)
def _latent_tail_attn_res_kernel(
    latent_ptr,  # [M, L] bf16
    norm_w_ptr,  # [L]
    w_ptr,  # [H, L] bf16 up-proj weight
    shared_ptr,  # [M, H] bf16
    res_ptr,  # [M, H] bf16 residual prefix (add3), or aliased shared when HAS_RES off
    bank_ptr,  # [M, NB, H] bf16
    qk_w_ptr,  # [H] scoring query
    onorm_w_ptr,  # [H] output RMSNorm weight
    prefix_out_ptr,  # [M, H] bf16 produced prefix
    out_ptr,  # [M, H] bf16 AttnRes output
    M,
    eps,
    score_eps,
    out_eps,
    scale,
    stride_lm,
    stride_wn,
    stride_sm,
    stride_rm,
    stride_bm,
    stride_bl,
    stride_pm,
    stride_om,
    L: tl.constexpr,  # latent dim (3584)
    H: tl.constexpr,  # hidden dim (7168)
    NB: tl.constexpr,  # bank rows allocated
    NVB: tl.constexpr,  # bank rows scored
    L_CAND: tl.constexpr,  # NVB + 1 candidates
    BLOCK_N: tl.constexpr,  # unused in the per-token path (kept for the config key)
    BLOCK_K: tl.constexpr,  # latent K tile for the GEMV accumulation
    BD: tl.constexpr,  # next_pow2(H), the resident row width
    HAS_RES: tl.constexpr,
    WRITE_SNAPSHOT: tl.constexpr,
):
    """One program per token (grid = M), mirroring attn_res_gate's per-token grid.

    The program produces its token's full prefix row (the latent-tail up-proj as
    a row GEMV, Kevin #44985), keeps it resident in a [BD] register tile, and
    runs the AttnRes read on it directly -- the read's score / softmax / raw-value
    mix / output RMSNorm / snapshot all reuse the prefix already in registers, so
    the read's own launch and its HBM round trip on the prefix are eliminated.

    The weight streams through L2 once across the M concurrent programs; at the
    decode M this path serves (M <= ~256), the up-proj is weight-streaming-bound
    and underfills the chip, so the read rides in the spare cycles for ~free."""
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    offs_h = tl.arange(0, BD)
    mask_h = offs_h < H

    # ---- up-proj row GEMV (Kevin #44985): acc[h] = sum_k W[h,k]*bf16(x[k]*nw[k]) ----
    # acc over the full resident H width; sq folded in the same K pass.
    acc = tl.zeros([BD], dtype=tl.float32)
    sq = tl.zeros([1], dtype=tl.float32)
    for k0 in tl.range(0, L, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < L
        x = tl.load(
            latent_ptr + pid_m * stride_lm + offs_k, mask=mask_k, other=0.0
        ).to(tl.float32)  # [BLOCK_K]
        nw = tl.load(norm_w_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        sq += tl.sum(x * x)
        xb = (x * nw).to(tl.bfloat16)  # [BLOCK_K]
        # W tile [BD, BLOCK_K]; acc[h] += sum_k xb[k] * W[h,k]
        w = tl.load(
            w_ptr + offs_h[:, None] * stride_wn + offs_k[None, :],
            mask=mask_h[:, None] & mask_k[None, :],
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)  # [BD, BLOCK_K]
        acc += tl.sum(w * xb[None, :].to(tl.float32), axis=1)

    inv_rms = 1.0 / tl.sqrt(sq / L + eps)
    proj = (acc * inv_rms).to(tl.bfloat16).to(tl.float32)
    proj += tl.load(
        shared_ptr + pid_m * stride_sm + offs_h, mask=mask_h, other=0.0
    ).to(tl.float32)
    if HAS_RES:
        proj += tl.load(
            res_ptr + pid_m * stride_rm + offs_h, mask=mask_h, other=0.0
        ).to(tl.float32)
    prefix = proj.to(tl.bfloat16)  # [BD] the new residual candidate
    tl.store(prefix_out_ptr + pid_m * stride_pm + offs_h, prefix, mask=mask_h)
    if WRITE_SNAPSHOT:
        tl.store(
            bank_ptr + pid_m * stride_bm + NVB * stride_bl + offs_h, prefix, mask=mask_h
        )
    prefix_v = prefix.to(tl.float32)

    # ---- AttnRes read over L_CAND = NVB + 1 candidates (bank rows + prefix) ----
    qk = tl.load(qk_w_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)

    m_run = -float("inf")
    denom = 0.0
    mix = tl.zeros([BD], dtype=tl.float32)
    for l in tl.range(0, NVB):
        v = tl.load(
            bank_ptr + pid_m * stride_bm + l * stride_bl + offs_h, mask=mask_h, other=0.0
        ).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(v * v) / H + score_eps)
        logit = scale * rstd * tl.sum(v * qk)
        m_new = tl.maximum(m_run, logit)
        corr = tl.exp(m_run - m_new)
        p = tl.exp(logit - m_new)
        denom = denom * corr + p
        mix = mix * corr + p * v
        m_run = m_new

    rstd_p = 1.0 / tl.sqrt(tl.sum(prefix_v * prefix_v) / H + score_eps)
    logit_p = scale * rstd_p * tl.sum(prefix_v * qk)
    m_new = tl.maximum(m_run, logit_p)
    corr = tl.exp(m_run - m_new)
    p = tl.exp(logit_p - m_new)
    denom = denom * corr + p
    mix = mix * corr + p * prefix_v

    o = mix / denom
    o_rstd = 1.0 / tl.sqrt(tl.sum(o * o) / H + out_eps)
    ow = tl.load(onorm_w_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    y = (o * o_rstd) * ow
    tl.store(out_ptr + pid_m * stride_om + offs_h, y.to(out_ptr.dtype.element_ty), mask=mask_h)
