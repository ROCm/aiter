# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepGEMM CK binding."""

from torch import Tensor

from ..jit.core import compile_ops


@compile_ops("module_deepgemm", fc_name="deepgemm")
def deepgemm_ck(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    group_layout: Tensor,
    x_scale: Tensor | None = None,
    w_scale: Tensor | None = None,
) -> Tensor: ...


def deepgemm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    group_layout: Tensor,
    x_scale: Tensor | None = None,
    w_scale: Tensor | None = None,
):
    return deepgemm_ck(XQ, WQ, Y, group_layout, x_scale, w_scale)


__all__ = [
    "deepgemm",
    "deepgemm_ck",
]
