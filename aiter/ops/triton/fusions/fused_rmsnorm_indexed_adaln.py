# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_rmsnorm_indexed_adaln import (
    _fused_rmsnorm_indexed_adaln_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_rmsnorm_indexed_adaln(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float = 1e-5,
    out: torch.Tensor | None = None,
    round_intermediate: bool = False,
) -> torch.Tensor:
    """RMSNorm followed by a table-indexed affine modulation, in one pass.

        out[m] = rmsnorm(x[m], weight) * (1 + scale[indices[m]]) + shift[indices[m]]

    This is the adaptive-layernorm (AdaLN) step of a diffusion transformer.
    Every token carries an index into a small table of modulation vectors --
    one row per (modality, timestep) combination -- so the affine parameters
    vary per token while the table stays tiny.

    Key parameters:
    - x: [M, N] activations, contiguous
    - weight: [N] RMSNorm gain
    - shift, scale: [G, N] modulation table, one row per index value
    - indices: [M] int32/int64, one table row per token
    - eps: RMSNorm epsilon
    - out: optional [M, N] destination; allocated when omitted
    - round_intermediate: round to the output dtype at the points an unfused
      implementation would store, trading accuracy for comparability with it

    Doing the norm and the modulation together is what makes this worth a
    kernel: separately, the activation is written and re-read between them, and
    ``scale``/``shift`` are materialised at [M, N] by the gather. Fused, x is
    read once and out written once, and the table is read straight from cache.

    Returns:
    - out: [M, N], same dtype as x
    """
    _LOGGER.info(
        f"FUSED_RMSNORM_INDEXED_ADALN: x={tuple(x.shape)} "
        f"table={tuple(scale.shape)} idx={tuple(indices.shape)}"
    )

    assert x.ndim == 2, f"x must be 2-D, got {tuple(x.shape)}"
    M, N = x.shape
    assert weight.shape == (N,), f"weight must be [{N}], got {tuple(weight.shape)}"
    assert shift.shape == scale.shape, "shift and scale must have the same shape"
    assert (
        shift.ndim == 2 and shift.shape[1] == N
    ), f"modulation table must be [G, {N}], got {tuple(shift.shape)}"
    assert indices.shape == (M,), f"indices must be [{M}], got {tuple(indices.shape)}"
    assert x.is_contiguous(), "x must be contiguous"
    assert indices.is_contiguous(), "indices must be contiguous"

    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.shape == x.shape and out.dtype == x.dtype
        assert out.is_contiguous(), "out must be contiguous"
    if M == 0:
        return out

    # Wide rows are walked in tiles rather than padded to the next power of two:
    # a 5376-wide row would mask off a third of every access at 8192.
    BLOCK_N = 1024 if N >= 1024 else triton.next_power_of_2(N)
    # Several rows per program so the modulation table is fetched once for a run
    # of tokens that share an index, which is the common case.
    BLOCK_M = 8 if M >= 8 else 1

    _fused_rmsnorm_indexed_adaln_kernel[(triton.cdiv(M, BLOCK_M),)](
        out,
        x,
        weight,
        shift,
        scale,
        indices,
        M,
        N,
        x.stride(0),
        out.stride(0),
        shift.stride(0),
        scale.stride(0),
        eps,
        ROUND_INTERMEDIATE=round_intermediate,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out
