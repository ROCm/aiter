# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_rmsnorm_opus"


# ffi_type="ctypes": torch-free / pybind-free C ABI. torch.Tensor args are
# converted to the POD aiter_tensor_t and the current HIP stream is appended
# automatically by aiter's _ctypes_call. The C++ side never sees torch, so the
# module is a single fast TU (no ~4s pybind11 / ~21s torch-extension TU).
@compile_ops("module_rmsnorm_opus", fc_name="rms_norm_opus", ffi_type="ctypes")
def rms_norm_opus(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    epsilon: float,
) -> None:
    """
    Opus RMSNorm: out = rmsnorm(input) * weight (bf16/fp16, fp32 accumulate).

    Self-contained opus kernel; no CK blob-generate, so the JIT cold build is a
    single torch-free TU instead of the ~1360 TUs of module_rmsnorm.
    """
    ...


@compile_ops(
    "module_rmsnorm_opus", fc_name="fused_add_rms_norm_opus", ffi_type="ctypes"
)
def fused_add_rms_norm_opus(
    input: Tensor,  # in-place: becomes rmsnorm(input + residual) * weight
    residual: Tensor,  # in-place: becomes input + residual (pre-norm sum)
    weight: Tensor,
    epsilon: float,
) -> None:
    """
    Opus fused residual-add + RMSNorm, in place:
        x = input + residual;  residual = x;  input = rmsnorm(x) * weight
    """
    ...


def rms_norm(input: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    """Functional wrapper mirroring aiter.ops.rmsnorm.rms_norm (allocates out)."""
    out = torch.empty_like(input)
    rms_norm_opus(out, input, weight, epsilon)
    return out
