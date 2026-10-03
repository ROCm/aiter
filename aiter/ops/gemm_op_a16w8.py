# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..utility import dtypes


@compile_ops(
    "module_gemm_a16w8_asm",
    fc_name="gemm_a16w8_asm",
    ffi_type="ctypes",
)
def _gemm_a16w8_asm(
    A: Tensor,
    B: Tensor,
    B_scale: Tensor,
    out: Tensor,
) -> None: ...


def gemm_a16w8_asm(
    A: Tensor,
    B: Tensor,
    B_scale: Tensor,
    out: Tensor | None = None,
) -> Tensor:
    """out = (A @ B.T) * B_scale with BF16 activations and FP8 weights, for decode on gfx950.

    The activations stay in BF16, so there is no activation quantization step
    and no activation quantization error. The weights are read as FP8, half
    the bytes of BF16 weights.

    A: [M, K] bf16, contiguous, with M from 1 to 8.
    B: [N, K] torch.float8_e4m3fn, contiguous.
    B_scale: [N] or [N, 1] fp32, one scale per row of B.
    out: [M, N] bf16, contiguous. A new tensor is returned when out is None.

    The kernels cover (N, K) = (4608, 8192) and (8192, 2048). Other shapes,
    other GPUs and M above 8 raise a RuntimeError.
    """
    if B.dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            f"gemm_a16w8_asm: B must be torch.float8_e4m3fn, got {B.dtype}"
        )
    if out is None:
        out = torch.empty(A.shape[0], B.shape[0], dtype=dtypes.bf16, device=A.device)
    _gemm_a16w8_asm(A, B, B_scale, out)
    return out
