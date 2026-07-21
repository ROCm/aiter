# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opus batched-BMM Python bindings.

This module is intentionally separate from `gemm_op_a16w16.py`: BMM callers use
batch-in-the-middle or grouped layouts (for example DSV4 `wo_a`) while the
underlying kernels still live in the shared opus GEMM backend.
"""

import torch

from ...jit.core import compile_ops


def _gen_bmm_a8w8_scale_mmajor_fake_tensors(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    return Y


# mmajor fp8 block-scale BMM: x/Y are [M, batch, *] (dim0=M, dim1=batch),
# x_scale [M, batch, K/GROUP_K] (per-token M); wo_a + w_scale stay batch-major.
# Zero-copy DSV4 wo_a fp8 (no caller-side transpose). Y is fp32 today.
@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_bmm_a8w8_scale_mmajor",
    gen_fake=_gen_bmm_a8w8_scale_mmajor_fake_tensors,
    develop=True,
)
def _opus_bmm_a8w8_scale_mmajor_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor: ...


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_bmm_a8w8_mxscale_mmajor",
    gen_fake=_gen_bmm_a8w8_scale_mmajor_fake_tensors,
    develop=True,
)
def _opus_bmm_a8w8_mxscale_mmajor_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    kernelId: int = 710,
) -> torch.Tensor: ...


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_bmm_a8w8_mxscale_splitk_mmajor",
    gen_fake=_gen_bmm_a8w8_scale_mmajor_fake_tensors,
    develop=True,
)
def _opus_bmm_a8w8_mxscale_splitk_mmajor_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    splitK: int = 8,
) -> torch.Tensor: ...


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor",
    gen_fake=_gen_bmm_a8w8_scale_mmajor_fake_tensors,
    develop=True,
)
def _opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    splitK: int = 2,
    kernelId: int = 0,
) -> torch.Tensor: ...


def _gen_bmm_uniform_scale_fake_tensors(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    kernelId: int = 700,
) -> torch.Tensor:
    return Y


# batch-major fp8 block-scale uniform BMM: x/wo_a/Y are
# [batch,M,K]/[batch,N,K]/[batch,M,N]. Y may be fp32 or bf16.
@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_bmm_a8w8_uniform_scale",
    gen_fake=_gen_bmm_uniform_scale_fake_tensors,
    develop=True,
)
def _opus_bmm_a8w8_uniform_scale_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    kernelId: int = 700,
) -> torch.Tensor: ...


def _gen_bmm_uniform_scale_mmajor_fake_tensors(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    kernelId: int = 700,
) -> torch.Tensor:
    return Y


# mmajor fp8 block-scale uniform BMM. Same layout contract as
# _opus_bmm_a8w8_scale_mmajor_raw; kernelId selects tile 700/701. Y may be
# fp32 or bf16.
@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_bmm_a8w8_uniform_scale_mmajor",
    gen_fake=_gen_bmm_uniform_scale_mmajor_fake_tensors,
    develop=True,
)
def _opus_bmm_a8w8_uniform_scale_mmajor_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    kernelId: int = 700,
) -> torch.Tensor: ...


__all__ = [
    "_opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_raw",
    "_opus_bmm_a8w8_mxscale_mmajor_raw",
    "_opus_bmm_a8w8_mxscale_splitk_mmajor_raw",
    "_opus_bmm_a8w8_scale_mmajor_raw",
    "_opus_bmm_a8w8_uniform_scale_raw",
    "_opus_bmm_a8w8_uniform_scale_mmajor_raw",
]
