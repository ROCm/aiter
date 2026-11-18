# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton.utils._triton.arch_info import get_arch


def require_wave32():
    """Require the target used by the standalone mono stages."""
    if get_arch() not in ("gfx1250", "gfx1250-strict"):
        raise ValueError("Persistent mono stages require gfx1250 (four wave32 waves)")


def check_output(output, shape, dtype, device, name, *, contiguous=False):
    if output is None:
        return
    if output.shape != shape or output.dtype != dtype or output.device != device:
        raise ValueError(
            f"{name} must have shape {shape}, dtype {dtype}, device {device}"
        )
    if contiguous and not output.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if output.numel() and (
        output.stride(-1) != 1
        or (output.ndim == 2 and output.stride(0) < output.shape[1])
    ):
        raise ValueError(f"{name} must have non-overlapping, contiguous rows")


def validate_dual_norm(q, k, q_weight, k_weight, yq, sq, yk):
    require_wave32()
    if q.shape[1] <= 0 or k.shape[1] <= 0:
        raise ValueError("Q and K widths must be positive")
    for x in (q, k, q_weight, k_weight):
        if not x.is_cuda or x.device != q.device or x.dtype != torch.bfloat16:
            raise ValueError("Persistent dual norm requires BF16 tensors on one GPU")
    if not q_weight.is_contiguous() or not k_weight.is_contiguous():
        raise ValueError("Norm weights must be contiguous")
    check_output(yq, q.shape, torch.float8_e4m3fn, q.device, "yq")
    check_output(sq, (q.shape[0], q.shape[1] // 32), torch.uint8, q.device, "sq")
    check_output(yk, k.shape, k.dtype, q.device, "yk")
