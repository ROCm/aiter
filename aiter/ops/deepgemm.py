# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
DeepGEMM front-end (CK backend).

import os
from torch import Tensor
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


@compile_ops("module_deepgemm_opus", fc_name="opus_gemm")
def deepgemm_opus(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    group_layout: Optional[Tensor] = None,
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
    backend = os.environ.get("AITER_DEEPGEMM_BACKEND", "ck")
    if backend == "opus":
        return deepgemm_opus(XQ, WQ, Y, group_layout, x_scale, w_scale)
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
