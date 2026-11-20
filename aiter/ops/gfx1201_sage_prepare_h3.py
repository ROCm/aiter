# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import torch
from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_gfx1201_sage_prepare_h3"


@compile_ops("module_gfx1201_sage_prepare_h3")
def gfx1201_sage_prepare_h3_key(key: Tensor, mean: Tensor) -> list[Tensor]: ...


@compile_ops("module_gfx1201_sage_prepare_h3")
def gfx1201_sage_prepare_h3_value(key: Tensor, value: Tensor) -> list[Tensor]: ...


def gfx1201_sage_prepare_h3(key: Tensor, value: Tensor):
    """Quantize K/V for the H3 attention core.

    ``key``/``value``: contiguous BF16 ``[B, S, H, 128]``. Returns
    ``(key_int8, key_scale, value_fp8, value_scale, value_residual)``.

    ``key_int8`` is K with the per-head channel mean removed. ``key_scale`` is
    ``[B, H, ceil(S/64)]``, one INT8 scale per 64 key rows, derived from a sampled maximum
    times a guard margin rather than an exact maximum. ``value_fp8`` is e4m3, transposed to
    ``[B, H, 128, S rounded up to 16]``; ``value_scale`` is per channel and
    ``value_residual`` carries the mean quantization error the core folds back in.
    """
    mean, value_fp8, value_scale = gfx1201_sage_prepare_h3_value(key, value)
    key_int8, key_scale = gfx1201_sage_prepare_h3_key(key, mean)
    # The value kernel allocates 2 * B*H*128 floats and returns the first half as
    # value_scale; the residual reduction writes the second half in place.
    count = value_scale.numel()
    flat = torch.empty(0, dtype=torch.float32, device=value_scale.device)
    flat.set_(value_scale.untyped_storage(), 0, (2 * count,))
    return (
        key_int8,
        key_scale,
        value_fp8,
        value_scale,
        flat[count:].view_as(value_scale),
    )
