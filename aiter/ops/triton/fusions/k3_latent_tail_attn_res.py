# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# GEMM-epilogue graft: the decode MoE latent-tail up-projection GEMM whose
# epilogue computes the next layer's Attention-Residual (AttnRes) read on the
# freshly-produced prefix row, in one launch.
#
# Ideas drawn from, and attributed to, the competing sgl-project/sglang PRs:
#   * Kevin Zhao's fused Kimi-K3 latent-MoE tail (sgl-project/sglang #44985):
#     the column/row-parallel up-proj that folds the row 1/rms into the GEMM
#     epilogue (sq accumulated in the same K pass) and the add3 of shared +
#     prefix -- the GEMM shell this kernel grafts the read onto.
#   * Kevin Zhao's bank-partials producer (sgl-project/sglang #44988): the
#     acc-cache idea for scoring bank candidates without re-reading them.
# The AttnRes read math mirrors aiter's in-tree attn_res_gate / flydsl_attn_res
# (ROCm/aiter #4610 snapshot-cap fix), kept bit-faithful here.

"""Fused Kimi-K3 latent-tail up-proj GEMM + AttnRes read for small decode M.

At decode (M <= ~1024) the latent-tail up-projection GEMM underfills the chip
(it is weight-streaming-bound, ~28 CTAs on ~256 CUs), so the adjacent AttnRes
read -- otherwise a separate launch the next layer runs -- can be hidden in the
GEMM's epilogue for ~free: once the GEMM has produced a token's full prefix row
in registers/LDS, the read's score / softmax / raw-value mix / output RMSNorm /
snapshot all run on data already resident, eliminating the read's launch and its
HBM round trip on the prefix.

Pipeline, per token (one program per token, prefix row resident):

    1. up-proj GEMM (Kevin #44985):
         proj[h]  = inv_rms(latent) * sum_k W[h,k] * bf16(latent[k] * norm_w[k])
         prefix[h] = bf16(proj[h] + shared[h] [+ residual_prefix[h]])   # fp32 add
    2. AttnRes read (fla / attn_res_gate math) over L = nvb + 1 candidates
       (the nvb bank rows plus prefix as the last candidate):
         rstd_l  = 1 / sqrt(mean_h v[l,h]^2 + score_eps)
         logit_l = rstd_l * sum_h v[l,h] * qk_weight[h]
         gate    = softmax(scale * logit)
         o[h]    = sum_l gate_l * v[l,h]                 # raw values
         out[h]  = out_rmsnorm(o, output_norm_weight, out_eps)
    3. snapshot: bank[:, nvb, :] = prefix       (block-close / write layer only)

Scoring, mixing and both RMSNorms accumulate in fp32; only the prefix row is
rounded to bf16 before it is scored and snapshotted, matching attn_res_hip.

Falls back to the unfused chain outside its support window (see
``supports_latent_tail_attn_res``); the caller keeps the standalone up-proj GEMM
and attn_res_gate for prefill-M, where the GEMM already fills the chip and the
graft is a no-op.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.fusions.k3_latent_tail_attn_res import (
    _latent_tail_attn_res_kernel,
)

# Largest decode batch the fused path is built for. Above this M the up-proj GEMM
# fills the chip, so the read has no spare CUs/BW to hide in and the unfused chain
# (standalone GEMM + attn_res_gate) is the right call. Mirrors Kevin #44985's
# MAX_TOKENS gate, raised to the decode band the graft still wins in.
MAX_TOKENS: int = 256

# Up to eight bank rows are scored; the snapshot (write layer) lands in slot nvb.
# Mirrors the in-tree attn_res_hip / flydsl bank layout ([T, NB=9, H]).
_MAX_NVB: int = 8


def supports_latent_tail_attn_res(
    latent: torch.Tensor,
    up_weight: torch.Tensor,
    shared: torch.Tensor,
    prefix: Optional[torch.Tensor],
    bank: torch.Tensor,
    nvb: int,
) -> bool:
    """Whether the fused path can serve these inputs (else keep the unfused chain)."""
    M, L = latent.shape
    H = up_weight.shape[0]
    return (
        0 < M <= MAX_TOKENS
        and 0 < nvb <= _MAX_NVB
        and latent.dtype == up_weight.dtype == shared.dtype == torch.bfloat16
        and bank.dtype == torch.bfloat16
        and latent.stride(-1) == 1
        and up_weight.is_contiguous()
        and shared.stride(-1) == 1
        and bank.stride(-1) == 1
        and bank.shape[0] == M
        and bank.shape[-1] == H
        and bank.shape[-2] >= nvb + 1  # snapshot append slot must exist
        and (prefix is None or (prefix.dtype == torch.bfloat16 and prefix.stride(-1) == 1))
    )


def k3_latent_tail_attn_res(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    up_weight: torch.Tensor,
    shared: torch.Tensor,
    residual_prefix: Optional[torch.Tensor],
    bank: torch.Tensor,
    nvb: int,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
    score_eps: float,
    out_eps: float,
    *,
    scale: float = 1.0,
    write_snapshot: bool = True,
    out: Optional[torch.Tensor] = None,
    block_n: int = 256,
    block_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
    waves_per_eu: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused latent-tail up-proj GEMM + AttnRes read.

    The up-proj produces ``prefix = bf16(rmsnorm(latent) @ up_weight.T + shared
    [+ residual_prefix])``; that prefix is then scored with the ``nvb`` bank rows
    as the AttnRes candidate set and mixed, output-RMSNorm'd, and (if
    ``write_snapshot``) written into ``bank[:, nvb, :]`` -- all in one launch.

    Args:
      latent: ``[M, L]`` bf16 latent activation (the up-proj input).
      norm_weight: ``[L]`` latent RMSNorm weight (folded into the GEMM input).
      eps: latent RMSNorm epsilon.
      up_weight: ``[H, L]`` bf16 up-projection weight (contiguous).
      shared: ``[M, H]`` bf16 shared-expert output (add3 addend).
      residual_prefix: optional ``[M, H]`` bf16 running residual (add3 addend).
      bank: ``[M, NB, H]`` bf16 bank; rows ``0..nvb-1`` are scored, row ``nvb``
        is the snapshot append slot. ``NB >= nvb + 1``.
      nvb: number of scored bank rows (<= 8).
      qk_weight: ``[H]`` scoring query (``rms_weight * query``, pre-folded).
      output_norm_weight: ``[H]`` output RMSNorm weight.
      score_eps: per-candidate RMSNorm epsilon.
      out_eps: output RMSNorm epsilon.
      scale: multiplies the logits before the softmax.
      write_snapshot: when True, store the produced prefix into ``bank[:, nvb, :]``
        (the block-close / write layer); otherwise bank is read-only.
      out: optional ``[M, H]`` bf16 output buffer.
      block_m: token tile; defaults to the next pow2 of M, >= 16.

    Returns ``(out, prefix)``: the AttnRes output and the produced bf16 prefix.
    """
    M, L = latent.shape
    H = up_weight.shape[0]
    NB = bank.shape[-2]
    Lc = nvb + 1  # candidates: nvb bank rows plus prefix

    if out is None:
        out = torch.empty((M, H), dtype=torch.bfloat16, device=latent.device)
    prefix = torch.empty((M, H), dtype=torch.bfloat16, device=latent.device)

    bd = triton.next_power_of_2(H)

    has_res = residual_prefix is not None
    res = residual_prefix if has_res else shared

    # grid = M: one program per token (mirrors attn_res_gate), the prefix row
    # stays resident so the read reuses the GEMV output with no HBM round trip.
    _latent_tail_attn_res_kernel[(M,)](
        latent,
        norm_weight,
        up_weight,
        shared,
        res,
        bank,
        qk_weight,
        output_norm_weight,
        prefix,
        out,
        M,
        eps,
        score_eps,
        out_eps,
        scale,
        latent.stride(0),
        up_weight.stride(0),
        shared.stride(0),
        res.stride(0),
        bank.stride(0),
        bank.stride(1),
        prefix.stride(0),
        out.stride(0),
        L=L,
        H=H,
        NB=NB,
        NVB=nvb,
        L_CAND=Lc,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BD=bd,
        HAS_RES=has_res,
        WRITE_SNAPSHOT=write_snapshot,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
    )
    return out, prefix
