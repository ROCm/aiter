# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor

from ..jit.core import compile_ops


@compile_ops("module_fused_indexed_gated_residual", develop=True)
def fused_indexed_gated_residual_hip(
    residual: Tensor,
    x: Tensor,
    gate: Tensor,
    indices: Tensor,
    out: Tensor,
    round_intermediate: bool,
) -> None: ...


def fused_indexed_gated_residual(
    residual: Tensor,
    x: Tensor,
    gate: Tensor,
    indices: Tensor,
    out: Tensor | None = None,
    round_intermediate: bool = False,
) -> Tensor:
    """Gated residual add with a table-indexed gate, in one pass (HIP).

        out[m] = residual[m] + gate[indices[m]] * x[m]

    The residual step that closes an attention or MLP branch of a diffusion
    transformer block: every token carries an index into a small table of gate
    vectors, one row per (modality, timestep) combination. Unfused,
    ``gate[indices]`` is materialised at [M, N] and the product is written and
    re-read before the add.

    - residual, x: [M, N] fp32/fp16/bf16, unit stride along N
    - gate: [G, N] gate table, same dtype
    - indices: [M] int32/int64, contiguous
    - out: optional [M, N] destination; may be ``residual`` (in-place update)
    - round_intermediate: round ``gate * x`` to the output dtype before the add,
      as the eager expression does; the result then matches it bit for bit

    Returns ``out``.
    """
    # Checked here so bad inputs raise; the kernel's own checks abort the process.
    if residual.ndim != 2:
        raise ValueError(f"residual must be 2-D, got {tuple(residual.shape)}")
    M, N = residual.shape
    if out is None:
        out = torch.empty_like(residual)
    if x.shape != residual.shape or out.shape != residual.shape:
        raise ValueError(f"x and out must be [{M}, {N}]")
    if gate.ndim != 2 or gate.shape[1] != N:
        raise ValueError(f"gate table must be [G, {N}], got {tuple(gate.shape)}")
    if indices.shape != (M,) or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"indices must be int32/int64 [{M}]")
    if M > 1 and indices.stride(0) != 1:
        raise ValueError("indices must be contiguous")
    if residual.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {residual.dtype}")
    for name, t in (("residual", residual), ("x", x), ("gate", gate), ("out", out)):
        if t.dtype != residual.dtype:
            raise ValueError(f"{name} must be {residual.dtype}, got {t.dtype}")
        if N > 1 and t.stride(1) != 1:
            raise ValueError(f"{name} must have unit stride along N")
    fused_indexed_gated_residual_hip(
        residual, x, gate, indices, out, round_intermediate
    )
    return out
