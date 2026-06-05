# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Tuple

import torch
import triton
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton._gluon_kernels.gfx1250.norm.fused_rmsnorm_add import (
    _gluon_fused_rms_kernel,
)
from aiter.ops.triton._triton_kernels.normalization.fused_rmsnorm_add import (
    _triton_fused_rms_kernel,
)


def _fused_rmsnorm_add_core(x, weight, epsilon, res1):
    """Shared RMSNorm (+ optional residual add) launcher.

    Returns ``(out, out_res1)`` where ``out_res1`` is ``None`` when ``res1`` is
    ``None``. The two guarded entry points below wrap this with a *fixed* return
    type each, so they can register as torch custom ops (a single op cannot have
    a Tensor-or-tuple return schema). Dispatches to the gfx1250 Gluon kernel on
    gfx1250, otherwise a portable Triton kernel.
    """
    assert x.dim() == 2, "fused_rmsnorm_add expects a 2D tensor"
    M, N = x.shape
    assert (
        weight.dim() == 1 and weight.numel() == N
    ), f"weight must be 1-D with {N} elements, got shape {tuple(weight.shape)}"
    assert (
        weight.dtype == x.dtype
    ), f"weight dtype {weight.dtype} must match x dtype {x.dtype}"
    if res1 is not None:
        assert (
            res1.shape == x.shape
        ), f"res1 shape {tuple(res1.shape)} must match x shape {tuple(x.shape)}"
        assert (
            res1.dtype == x.dtype
        ), f"res1 dtype {res1.dtype} must match x dtype {x.dtype}"

    BLOCK_SIZE_N = max(triton.next_power_of_2(N), 32)
    out1 = torch.empty((M, N), dtype=x.dtype, device=x.device)
    out_res1 = None
    res1_stride_m = 0
    out_res1_stride_m = 0
    if res1 is not None:
        out_res1 = torch.empty((M, N), dtype=x.dtype, device=x.device)
        res1_stride_m = res1.stride(0)
        out_res1_stride_m = out_res1.stride(0)

    if get_arch() == "gfx1250":
        BLOCK_SIZE_M = 1
        grid = (triton.cdiv(M, BLOCK_SIZE_M),)
        _gluon_fused_rms_kernel[grid](
            x,
            weight,
            res1,
            out1,
            out_res1,
            epsilon,
            M,
            N,
            x.stride(0),
            res1_stride_m,
            out1.stride(0),
            out_res1_stride_m,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            FIRST_INPUT_RES=(res1 is not None),
        )
    else:
        grid = (M,)
        _triton_fused_rms_kernel[grid](
            x,
            weight,
            res1,
            out1,
            out_res1,
            epsilon,
            M,
            N,
            x.stride(0),
            res1_stride_m,
            out1.stride(0),
            out_res1_stride_m,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            FIRST_INPUT_RES=(res1 is not None),
        )

    return out1, out_res1


def _fused_rmsnorm_fake(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    return torch.empty_like(x)


@torch_compile_guard(gen_fake=_fused_rmsnorm_fake)
def _fused_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    out1, _ = _fused_rmsnorm_add_core(x, weight, epsilon, None)
    return out1


def _fused_rmsnorm_add_fake(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float, res1: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(x)


@torch_compile_guard(gen_fake=_fused_rmsnorm_add_fake)
def _fused_rmsnorm_add(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float, res1: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    out1, out_res1 = _fused_rmsnorm_add_core(x, weight, epsilon, res1)
    return out1, out_res1


def fused_rmsnorm_add(x, weight, epsilon, res1=None):
    """RMSNorm over the last dim of a 2D tensor, with an optional residual add.

    x:      (M, N) tensor (bf16/fp16). Made contiguous if it isn't already.
    weight: (N,) tensor.
    res1:   optional (M, N) residual; when given, computes x += res1 first and
            returns (out, out_res1) where out_res1 is the pre-norm sum.
    Returns out (M, N) if res1 is None, else (out, out_res1).

    Dispatches to a gfx1250 Gluon kernel when running on gfx1250, otherwise
    falls back to a portable Triton kernel. The actual compute is registered as
    two torch custom ops (``_fused_rmsnorm`` / ``_fused_rmsnorm_add``) so the
    path is safe under torch.compile / CUDAGraph capture; this thin dispatcher
    selects the right one based on whether a residual was supplied (a static
    branch at trace time).
    """
    if res1 is None:
        return _fused_rmsnorm(x, weight, epsilon)
    return _fused_rmsnorm_add(x, weight, epsilon, res1)
