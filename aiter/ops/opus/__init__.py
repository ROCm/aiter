# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public OPUS interfaces: GEMM/BMM backed by shared exact-kid launchers, and the gated
MHA backward."""

from __future__ import annotations

import torch
from torch import Tensor

from ._gated_mha_bwd import gated_mha_bwd
from .dispatch import _opus_dispatch


def opus_gemm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    *,
    kid: int,
    family: str,
    layout: str = "plain",
    x_scale: Tensor | None = None,
    w_scale: Tensor | None = None,
    bias: Tensor | None = None,
    split_k: int = 0,
    workspace: Tensor | None = None,
) -> Tensor:
    """Launch logical 2D ``[M,K] x [N,K] -> [M,N]`` by exact ``kid``.

    ``Y`` is caller-owned and returned. ``layout='bpreshuffle'`` declares a
    transformed WQ content layout that Tensor metadata cannot prove.
    """
    return _opus_dispatch(
        "opus_gemm",
        2,
        XQ,
        WQ,
        Y,
        kid=kid,
        family=family,
        layout=layout,
        x_scale=x_scale,
        w_scale=w_scale,
        bias=bias,
        split_k=split_k,
        workspace=workspace,
    )


def opus_bmm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    *,
    kid: int,
    family: str,
    layout: str = "plain",
    x_scale: Tensor | None = None,
    w_scale: Tensor | None = None,
    bias: Tensor | None = None,
    split_k: int = 0,
    workspace: Tensor | None = None,
) -> Tensor:
    """Launch batch-first ``[B,M,K] x [B,N,K] -> [B,M,N]`` by exact kid."""
    return _opus_dispatch(
        "opus_bmm",
        3,
        XQ,
        WQ,
        Y,
        kid=kid,
        family=family,
        layout=layout,
        x_scale=x_scale,
        w_scale=w_scale,
        bias=bias,
        split_k=split_k,
        workspace=workspace,
    )


def gemm_a16w16_opus(
    A: Tensor,
    B: Tensor,
    bias: Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
    *,
    kernelId: int | None = None,
    splitK: int | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """Run the legacy shape-driven A16W16 OPUS selection path."""
    from .gemm_op_a16w16 import gemm_a16w16_opus as _impl

    return _impl(
        A,
        B,
        bias,
        dtype,
        kernelId=kernelId,
        splitK=splitK,
        out=out,
    )


__all__ = [
    "gated_mha_bwd",
    "gemm_a16w16_opus",
    "opus_bmm",
    "opus_gemm",
]
