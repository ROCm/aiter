# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
DeepGEMM front-end (CK backend).

Hosts the CK-backed `deepgemm_ck` binding plus a thin `deepgemm()`
wrapper. Opus entries have been extracted under `aiter.ops.opus.*`;
see `aiter.ops.opus.gemm_a16w16_opus` for BF16 matmul and
`aiter.ops.opus.opus_gemm_a16w16_tune` for id-based kernel selection.

`opus_gemm_a16w16_tune` is kept here as a deprecation shim for one
release to ease migration from the old aggregate entry.
"""

import warnings
from typing import Optional

import torch
from torch import Tensor

from ..jit.core import compile_ops
from .opus.gemm_op_a16w16 import opus_gemm_a16w16_tune as _opus_tune


@compile_ops("module_deepgemm", fc_name="deepgemm")
def deepgemm_ck(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    group_layout: Tensor,
    x_scale: Optional[Tensor] = None,
    w_scale: Optional[Tensor] = None,
) -> Tensor: ...


def deepgemm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    group_layout: Tensor,
    x_scale: Optional[Tensor] = None,
    w_scale: Optional[Tensor] = None,
):
    return deepgemm_ck(XQ, WQ, Y, group_layout, x_scale, w_scale)


def opus_gemm_a16w16_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor:
    warnings.warn(
        "aiter.ops.deepgemm.opus_gemm_a16w16_tune has moved to "
        "aiter.ops.opus.gemm_op_a16w16.opus_gemm_a16w16_tune; this "
        "shim will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _opus_tune(XQ, WQ, Y, kernelId, splitK)


__all__ = [
    "deepgemm_ck",
    "deepgemm",
    "opus_gemm_a16w16_tune",
]
