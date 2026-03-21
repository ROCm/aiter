# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""二维 fp32 向量加法（汇编 vadd_kernel），用于演示 HSACO 接入流程。

约束与参考 host（``vadd.cpp``）一致：``A``、``B``、``C`` 须为同形状的 **二维**、**连续**
``float32`` 张量，按行主序视作 ``height x width`` 网格；块大小为 16×16。

运行时由 JIT 设置 ``AITER_ASM_DIR``；若自行加载 ``.co``，请保证能解析到
``{AITER_ASM_DIR}/{arch}/vadd/k_vadd.co``。
"""

from __future__ import annotations

import torch

from .ops.vadd_asm import vadd_asm


def fused_vadd(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算 ``out = a + b``（与 PyTorch 逐元素相加语义一致，形状须为 2-D contiguous fp32）。"""
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("fused_vadd: a and b must be 2-D tensors")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("fused_vadd: only float32 is supported")
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("fused_vadd: a and b must be contiguous")
    if a.shape != b.shape:
        raise ValueError("fused_vadd: shape mismatch")
    if out is None:
        out = torch.empty_like(a)
    else:
        if out.shape != a.shape or out.dtype != a.dtype:
            raise RuntimeError("fused_vadd: out must match a shape and dtype")
        if not out.is_contiguous():
            raise ValueError("fused_vadd: out must be contiguous")
    vadd_asm(a, b, out)
    return out
