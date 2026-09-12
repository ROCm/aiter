# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Low-level Opus gfx942 A8W8 blockscale bpreshuffle entry points."""

import torch
from torch import Tensor

from ...jit.core import compile_ops


def _gen_opus_a8w8_blockscale_bpreshuffle_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor,
    kernelId: int,
) -> Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a8w8_blockscale_bpreshuffle_tune",
    gen_fake=_gen_opus_a8w8_blockscale_bpreshuffle_fake_tensors,
    develop=True,
)
def _opus_gemm_a8w8_blockscale_bpreshuffle_tune_raw(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor,
    kernelId: int,
) -> Tensor: ...


def opus_gemm_a8w8_blockscale_bpreshuffle_tune(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor | None = None,
    kernelId: int = 11000,
) -> Tensor:
    """Run one gfx942 Opus A8W8 blockscale bpreshuffle kernel by explicit id."""
    if Y is None:
        Y = torch.empty(
            (XQ.shape[-2], WQ.shape[-2]), device=XQ.device, dtype=torch.bfloat16
        )
    _opus_gemm_a8w8_blockscale_bpreshuffle_tune_raw(
        XQ, WQ, x_scale, w_scale, Y, kernelId
    )
    return Y


__all__ = ["opus_gemm_a8w8_blockscale_bpreshuffle_tune"]
