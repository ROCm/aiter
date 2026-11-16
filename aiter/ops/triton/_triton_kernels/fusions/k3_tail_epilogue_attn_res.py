# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Realizable GEMM-epilogue graft: hipBLASLt computes the up-proj matmul (acc);
# this ONE Triton kernel fuses the latent-tail epilogue (row 1/rms + shared +
# residual add3, Kevin #44985) AND the next-layer AttnRes read (score / softmax /
# raw-value mix / output RMSNorm / bank snapshot, aiter attn_res_gate / #4610)
# into a single launch on the freshly-produced prefix row.
#
# OSS attribution: sgl-project/sglang #44985 (latent-tail epilogue), #44988
# (bank producer), ROCm/aiter #4610 (snapshot-cap read). One program per token,
# mirroring attn_res_gate's per-token grid.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_repr = make_kernel_repr(
    "_tail_epilogue_attn_res_kernel", ["BD", "NVB", "L_CAND"]
)


@triton.jit(repr=_repr)
def _tail_epilogue_attn_res_kernel(
    acc_ptr,  # [M, H] fp32 raw up-proj matmul (hipBLASLt output, pre-rms)
    inv_rms_ptr,  # [M] fp32 latent row 1/rms (precomputed)
    shared_ptr,  # [M, H] bf16
    res_ptr,  # [M, H] bf16 residual prefix (add3), aliased shared if HAS_RES off
    bank_ptr,  # [M, NB, H] bf16
    qk_w_ptr,  # [H] scoring query
    onorm_w_ptr,  # [H] output RMSNorm weight
    prefix_out_ptr,  # [M, H] bf16 produced prefix
    out_ptr,  # [M, H] bf16 AttnRes output
    M,
    score_eps,
    out_eps,
    scale,
    stride_am,
    stride_sm,
    stride_rm,
    stride_bm,
    stride_bl,
    stride_pm,
    stride_om,
    H: tl.constexpr,
    NB: tl.constexpr,
    NVB: tl.constexpr,
    L_CAND: tl.constexpr,
    BD: tl.constexpr,
    HAS_RES: tl.constexpr,
    WRITE_SNAPSHOT: tl.constexpr,
):
    """One program per token. Epilogue + AttnRes read on the resident prefix row."""
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    offs_h = tl.arange(0, BD)
    mask_h = offs_h < H

    inv_rms = tl.load(inv_rms_ptr + pid_m)
    acc = tl.load(acc_ptr + pid_m * stride_am + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    proj = (acc * inv_rms).to(tl.bfloat16).to(tl.float32)
    proj += tl.load(shared_ptr + pid_m * stride_sm + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    if HAS_RES:
        proj += tl.load(res_ptr + pid_m * stride_rm + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    prefix = proj.to(tl.bfloat16)
    tl.store(prefix_out_ptr + pid_m * stride_pm + offs_h, prefix, mask=mask_h)
    if WRITE_SNAPSHOT:
        tl.store(bank_ptr + pid_m * stride_bm + NVB * stride_bl + offs_h, prefix, mask=mask_h)
    prefix_v = prefix.to(tl.float32)

    qk = tl.load(qk_w_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    m_run = -float("inf")
    denom = 0.0
    mix = tl.zeros([BD], dtype=tl.float32)
    for l in tl.range(0, NVB):
        v = tl.load(bank_ptr + pid_m * stride_bm + l * stride_bl + offs_h,
                    mask=mask_h, other=0.0).to(tl.float32)
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
