# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Realizable Kimi-K3 decode GEMM-epilogue graft: the up-proj matmul stays on the
# production GEMM (hipBLASLt), and this ONE Triton kernel fuses the latent-tail
# epilogue AND the next-layer AttnRes read into a single launch, eliminating the
# separate epilogue (rms + add3) and read launches and their HBM round trips.
#
# OSS attribution (bring the ideas of these PRs into aiter so everyone can use
# them, per Sami): the fused latent-tail epilogue -- row 1/rms folded into the
# epilogue, add3 of shared + residual prefix -- is Kevin Zhao's
# sgl-project/sglang #44985; the bank-producer idea is #44988; the AttnRes read
# math and the nvb=9 snapshot mirror aiter's attn_res_gate / flydsl_attn_res
# (ROCm/aiter #4610).

"""Fused Kimi-K3 latent-tail epilogue + AttnRes read for small decode M.

At decode (M <= ~1024) the latent-tail up-projection underfills the chip, so the
tail that follows it -- the row-RMSNorm epilogue (``bf16(acc * inv_rms) + shared
[+ residual]``) and the next layer's AttnRes read (score the ``nvb`` bank rows
plus the produced prefix, depth-softmax, raw-value mix, output RMSNorm, bank
snapshot) -- runs as two or three small separate launches on a nearly idle chip.
This kernel folds that whole tail into ONE launch that consumes the raw GEMM
accumulator ``acc`` (hipBLASLt output) directly: the prefix row is produced and
kept resident, so the read reuses it with no HBM round trip, and the epilogue's
``acc * inv_rms + add3`` and the read's add3 fold are both eliminated as separate
ops.

The production up-proj GEMM itself stays on hipBLASLt (it is far faster than any
Triton GEMM at this tall-skinny shape); this kernel is its fused epilogue+read
graft. One program per token, mirroring ``attn_res_gate``'s per-token grid.
Gate to decode M (``supports_tail_epilogue_attn_res``); prefill keeps the
unfused chain, where the GEMM already fills the chip and the graft is a no-op.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.k3_tail_epilogue_attn_res import (
    _tail_epilogue_attn_res_kernel,
)

MAX_TOKENS: int = 1024
_MAX_NVB: int = 8

# Tuned on MI355X/gfx950 (sweep.py): num_warps=4 wins across the decode band;
# num_stages=1 and a small waves_per_eu are marginal. The read is memory-bound
# (9 bank rows of M*H bf16), so these are near-flat.
_NUM_WARPS: int = 4
_NUM_STAGES: int = 2
_WAVES_PER_EU: int = 2


def supports_tail_epilogue_attn_res(
    acc: torch.Tensor,
    shared: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
) -> bool:
    """Whether the fused path can serve these inputs (else keep the unfused chain)."""
    M, H = acc.shape
    return (
        0 < M <= MAX_TOKENS
        and 0 < nvb <= _MAX_NVB
        and acc.dtype == torch.float32
        and shared.dtype == torch.bfloat16
        and bank.dtype == torch.bfloat16
        and acc.stride(-1) == 1
        and shared.stride(-1) == 1
        and bank.stride(-1) == 1
        and bank.shape[0] == M
        and bank.shape[-1] == H
        and bank.shape[-2] >= nvb + 1
    )


def k3_tail_epilogue_attn_res(
    acc: torch.Tensor,
    latent_inv_rms: torch.Tensor,
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
    prefix_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused latent-tail epilogue + AttnRes read on a raw up-proj accumulator.

    Args:
      acc: ``[M, H]`` fp32 raw up-proj matmul (hipBLASLt output, pre row-rms).
      latent_inv_rms: ``[M]`` fp32 latent row ``1/rms`` (commutes with the GEMM,
        applied in the epilogue; mirrors Kevin #44985).
      shared: ``[M, H]`` bf16 shared-expert output (add3 addend).
      residual_prefix: optional ``[M, H]`` bf16 running residual (add3 addend).
      bank: ``[M, NB, H]`` bf16; rows ``0..nvb-1`` scored, row ``nvb`` is the
        snapshot append slot (``NB >= nvb + 1``).
      nvb: number of scored bank rows (<= 8).
      qk_weight: ``[H]`` scoring query (``rms_weight * query``).
      output_norm_weight: ``[H]`` output RMSNorm weight.
      score_eps / out_eps: per-candidate / output RMSNorm epsilons.
      scale: logit scale before the softmax.
      write_snapshot: store the produced prefix into ``bank[:, nvb, :]``.

    Returns ``(out, prefix)``.
    """
    M, H = acc.shape
    NB = bank.shape[-2]
    if out is None:
        out = torch.empty((M, H), dtype=torch.bfloat16, device=acc.device)
    if prefix_out is None:
        prefix_out = torch.empty((M, H), dtype=torch.bfloat16, device=acc.device)
    has_res = residual_prefix is not None
    res = residual_prefix if has_res else shared
    bd = triton.next_power_of_2(H)

    _tail_epilogue_attn_res_kernel[(M,)](
        acc,
        latent_inv_rms,
        shared,
        res,
        bank,
        qk_weight,
        output_norm_weight,
        prefix_out,
        out,
        M,
        score_eps,
        out_eps,
        scale,
        acc.stride(0),
        shared.stride(0),
        res.stride(0),
        bank.stride(0),
        bank.stride(1),
        prefix_out.stride(0),
        out.stride(0),
        H=H,
        NB=NB,
        NVB=nvb,
        L_CAND=nvb + 1,
        BD=bd,
        HAS_RES=has_res,
        WRITE_SNAPSHOT=write_snapshot,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
        waves_per_eu=_WAVES_PER_EU,
    )
    return out, prefix_out
