# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Block Hadamard rotation for activations whose consumer was quantized in the
rotated basis. ``(x H)(w H)^T = x w^T`` for an orthonormal Sylvester ``H``.

Quark online R1 checkpoints store both MoE GEMM weights already rotated along
K. The first activation is rotated by the caller. The second is the SwiGLU
output, which only exists between the two stages, so ``fused_moe`` applies
this rotation when ``inter_hadamard`` is set.
"""

import functools

import torch


@functools.cache
def hadamard_matrix(size: int, device: torch.device) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard, ``H / sqrt(size)``, in fp32.

    ``size`` must be a power of two. The matrix is the same one
    ``scipy.linalg.hadamard`` builds, scaled to be orthonormal.
    """
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {size}")
    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    while h.shape[0] < size:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / size**0.5


@functools.cache
def _hadamard_matrix_as(
    size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return hadamard_matrix(size, device).to(dtype)


def block_hadamard_rotate(x: torch.Tensor, size: int) -> torch.Tensor:
    """Rotate each contiguous ``size``-wide block of the last dimension.

    For bf16/fp16 ``x`` this is one GEMM in ``x.dtype`` with fp32
    accumulation. Every entry of the matrix is ``+-1/sqrt(size)``, so rounding
    it to ``x.dtype`` only scales the rotation by a factor within ``2**-8`` of
    1. Other dtypes are rotated in fp32. ``x.shape[-1]`` must be a multiple of
    ``size``.
    """
    if x.shape[-1] % size != 0:
        raise ValueError(
            f"Dimension {x.shape[-1]} is not divisible by the Hadamard size {size}."
        )
    if x.dtype in (torch.bfloat16, torch.float16):
        hadamard = _hadamard_matrix_as(size, x.device, x.dtype)
        return (x.unflatten(-1, (-1, size)) @ hadamard).flatten(-2)
    hadamard = hadamard_matrix(size, x.device)
    rotated = x.float().unflatten(-1, (-1, size)) @ hadamard
    return rotated.flatten(-2).to(dtype=x.dtype)
