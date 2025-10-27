# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from typing import Optional
from ..jit.core import (
    compile_ops,
)


@compile_ops("module_m_grouped_gemm", fc_name="m_grouped_gemm")
def m_grouped_gemm_ck(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    group_layout: Tensor,
    x_scale: Optional[Tensor] = None,
    w_scale: Optional[Tensor] = None,
) -> Tensor: ...


def m_grouped_gemm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    group_layout: Tensor,
    x_scale: Optional[Tensor] = None,
    w_scale: Optional[Tensor] = None,
):
    return m_grouped_gemm_ck(XQ, WQ, Y, group_layout, x_scale, w_scale)
