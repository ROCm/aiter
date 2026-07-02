# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_rmsnorm_opus"

_DTYPE_CODE = {torch.float16: 0, torch.bfloat16: 1}


# Raw C ABI (ctypes): pointers/dims/stream travel as int64, so the C++ side needs
# no torch / HIP-runtime / aiter_tensor.h and compiles in ~0.2s. Validation and
# pointer/stream extraction happen in the Python wrappers below.
@compile_ops("module_rmsnorm_opus", fc_name="rms_norm_opus", ffi_type="ctypes")
def _rms_norm_opus_raw(
    out: int,
    input: int,
    weight: int,
    epsilon: float,
    rows: int,
    hidden: int,
    is_bf16: int,
    stream: int,
) -> None: ...


@compile_ops(
    "module_rmsnorm_opus", fc_name="fused_add_rms_norm_opus", ffi_type="ctypes"
)
def _fused_add_rms_norm_opus_raw(
    input: int,
    residual: int,
    weight: int,
    epsilon: float,
    rows: int,
    hidden: int,
    is_bf16: int,
    stream: int,
) -> None: ...


def _check(input: Tensor, weight: Tensor):
    assert (
        input.dtype in _DTYPE_CODE
    ), f"rms_norm_opus: bf16/fp16 only, got {input.dtype}"
    assert weight.dtype == input.dtype, "rms_norm_opus: weight dtype must match input"
    assert (
        input.is_contiguous() and weight.is_contiguous()
    ), "rms_norm_opus: contiguous only"
    assert weight.shape[-1] == input.shape[-1], "rms_norm_opus: weight length != hidden"


def rms_norm_opus(out: Tensor, input: Tensor, weight: Tensor, epsilon: float) -> None:
    """out = rmsnorm(input) * weight (bf16/fp16, fp32 accumulate)."""
    _check(input, weight)
    assert out.dtype == input.dtype and out.is_contiguous(), "rms_norm_opus: bad out"
    hidden = input.shape[-1]
    rows = input.numel() // hidden
    _rms_norm_opus_raw(
        out.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        float(epsilon),
        rows,
        hidden,
        _DTYPE_CODE[input.dtype],
        torch.cuda.current_stream().cuda_stream,
    )


def fused_add_rms_norm_opus(
    input: Tensor, residual: Tensor, weight: Tensor, epsilon: float
) -> None:
    """In place: x = input + residual; residual = x; input = rmsnorm(x) * weight."""
    _check(input, weight)
    assert residual.dtype == input.dtype and residual.is_contiguous(), "bad residual"
    assert residual.numel() == input.numel(), "residual shape != input"
    hidden = input.shape[-1]
    rows = input.numel() // hidden
    _fused_add_rms_norm_opus_raw(
        input.data_ptr(),
        residual.data_ptr(),
        weight.data_ptr(),
        float(epsilon),
        rows,
        hidden,
        _DTYPE_CODE[input.dtype],
        torch.cuda.current_stream().cuda_stream,
    )


def rms_norm(input: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    """Functional wrapper mirroring aiter.ops.rmsnorm.rms_norm (allocates out)."""
    out = torch.empty_like(input)
    rms_norm_opus(out, input, weight, epsilon)
    return out
