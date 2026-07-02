# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_rmsnorm_opus"


# ffi_type="ctypes": torch-free / pybind-free C ABI (torch.Tensor -> aiter_tensor_t,
# stream appended by _ctypes_call), so the module is a single fast TU.
@compile_ops("module_rmsnorm_opus", fc_name="rms_norm_opus", ffi_type="ctypes")
def rms_norm_opus(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    epsilon: float,
) -> None:
    """out = rmsnorm(input) * weight (bf16/fp16, fp32 accumulate)."""
    ...


@compile_ops(
    "module_rmsnorm_opus", fc_name="fused_add_rms_norm_opus", ffi_type="ctypes"
)
def fused_add_rms_norm_opus(
    input: Tensor,  # in-place -> rmsnorm(input + residual) * weight
    residual: Tensor,  # in-place -> input + residual (pre-norm sum)
    weight: Tensor,
    epsilon: float,
) -> None:
    """In place: x = input + residual; residual = x; input = rmsnorm(x) * weight."""
    ...


def rms_norm(input: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    """Functional wrapper mirroring aiter.ops.rmsnorm.rms_norm (allocates out)."""
    out = torch.empty_like(input)
    rms_norm_opus(out, input, weight, epsilon)
    return out
