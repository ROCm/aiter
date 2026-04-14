# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import torch
from torch import Tensor
from typing import Optional
from ..jit.core import (
    compile_ops,
)


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


def gen_opus_gemm_a16w16_tune_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a16w16_tune",
    gen_fake=gen_opus_gemm_a16w16_tune_fake_tensors,
)
def opus_gemm_a16w16_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor: ...
