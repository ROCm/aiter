# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_indexed_gated_residual import (
    _fused_indexed_gated_residual_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_indexed_gated_residual(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor | None = None,
    round_intermediate: bool = False,
) -> torch.Tensor:
    """Gated residual add with a table-indexed gate, in one pass.

        out[m] = residual[m] + gate[indices[m]] * x[m]

    This is the residual step that closes an attention or MLP branch of a
    diffusion transformer block, the counterpart of
    ``fused_rmsnorm_indexed_adaln`` that opens it: every token carries an index
    into a small table of gate vectors -- one row per (modality, timestep)
    combination.

    Key parameters:
    - residual: [M, N] residual stream
    - x: [M, N] branch output, same dtype as residual
    - gate: [G, N] gate table, one row per index value
    - indices: [M] int32/int64, one table row per token
    - out: optional [M, N] destination; may be ``residual`` (in-place update)
    - round_intermediate: round ``gate * x`` to the output dtype before the add,
      like the unfused expression does, trading accuracy for bit-comparability

    Unfused, ``gate[indices]`` is materialised at [M, N] and the product is
    written and re-read before the add. Fused, residual and x are read once, the
    table is read from cache, and out is written once.

    All operands need unit stride along N; rows may be strided.

    Returns:
    - out: [M, N], same dtype as residual
    """
    _LOGGER.info(
        "FUSED_INDEXED_GATED_RESIDUAL: residual=%s table=%s idx=%s",
        tuple(residual.shape),
        tuple(gate.shape),
        tuple(indices.shape),
    )
    assert residual.ndim == 2, f"residual must be 2-D, got {tuple(residual.shape)}"
    M, N = residual.shape
    assert x.shape == residual.shape, f"x must be [{M}, {N}], got {tuple(x.shape)}"
    assert x.dtype == residual.dtype, "x and residual must have the same dtype"
    assert (
        gate.ndim == 2 and gate.shape[1] == N
    ), f"gate table must be [G, {N}], got {tuple(gate.shape)}"
    assert indices.shape == (M,), f"indices must be [{M}], got {tuple(indices.shape)}"
    assert indices.dtype in (torch.int32, torch.int64), "indices must be int32/int64"
    assert indices.is_contiguous(), "indices must be contiguous"
    if out is None:
        out = torch.empty_like(residual)
    else:
        assert out.shape == residual.shape, "out must match residual's shape"
        assert out.dtype == residual.dtype, "out must match residual's dtype"
    for name, t in (("residual", residual), ("x", x), ("gate", gate), ("out", out)):
        assert N <= 1 or t.stride(1) == 1, f"{name} must have unit stride along N"
    if M == 0 or N == 0:
        return out

    # 5376 (MiniMax-H3) = 21 x 256: no masked tail tile.
    BLOCK_N = 256 if N >= 256 else triton.next_power_of_2(N)
    BLOCK_M = 16 if M >= 16 else triton.next_power_of_2(M)

    _fused_indexed_gated_residual_kernel[
        (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    ](
        out,
        residual,
        x,
        gate,
        indices,
        M,
        N,
        residual.stride(0),
        x.stride(0),
        out.stride(0),
        gate.stride(0),
        ROUND_INTERMEDIATE=round_intermediate,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        # With fusion the compiler may contract the rounded product back into an
        # fma, which silently drops round_intermediate.
        enable_fp_fusion=False,
    )
    return out
