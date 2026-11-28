# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_indexed_gated_residual_layernorm_adaln import (
    _fused_indexed_gated_residual_layernorm_adaln_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_indexed_gated_residual_layernorm_adaln(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor | None,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float = 1e-6,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    h_out: torch.Tensor | None = None,
    round_intermediate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated residual add, LayerNorm and a table-indexed adaLN modulation, in one pass.

        h[m]   = residual[m] + gate[indices[m]] * x[m]
        out[m] = layernorm(h[m], weight, bias) * (1 + scale[indices[m]]) + shift[indices[m]]

    This is the step between two branches of a diffusion transformer block: it
    closes one branch into the residual stream (``fused_indexed_gated_residual``)
    and opens the next one (``fused_layernorm_indexed_adaln``), for example
    SGLang's ``ScaleResidualLayerNormScaleShift``. Run as two kernels, h is
    written and read back; here it is written once and normalised from
    registers or cache.

    For a per-sample modulation of shape [B, N], pass the table as is and set
    ``indices[m]`` to the sample of token m; for a per-frame one [B, F, 1, N],
    flatten it to [B * F, N] and index the frame of token m.

    Key parameters:
    - residual: [M, N] residual stream, bf16 or fp16
    - x: [M, N] branch output, same dtype as residual
    - gate: [G, N] gate table, one row per index value, or None for a gate of 1
    - shift, scale: [G, N] modulation tables, same dtype as residual
    - indices: [M] int32/int64, one table row per token
    - eps: LayerNorm epsilon
    - weight, bias: optional [N] LayerNorm affine parameters
    - out: optional [M, N] destination of the modulated output
    - h_out: optional [M, N] destination of the new residual stream; may be
      ``residual`` (in-place update)
    - round_intermediate: round ``gate * x`` to the output dtype before the add,
      like the unfused expression does, trading accuracy for bit-comparability

    h is rounded to the output dtype, as the unfused residual stream is, and the
    LayerNorm statistics are of the rounded values, in fp32.

    All operands are on one GPU and need unit stride along N; rows may be
    strided.

    Returns:
    - (out, h): both [M, N], same dtype as residual
    """
    _LOGGER.info(
        "FUSED_INDEXED_GATED_RESIDUAL_LAYERNORM_ADALN: residual=%s table=%s idx=%s",
        tuple(residual.shape),
        tuple(scale.shape),
        tuple(indices.shape),
    )
    assert residual.ndim == 2, f"residual must be 2-D, got {tuple(residual.shape)}"
    assert residual.dtype in (
        torch.bfloat16,
        torch.float16,
    ), f"unsupported dtype: {residual.dtype}"
    M, N = residual.shape
    assert x.shape == residual.shape, f"x must be [{M}, {N}], got {tuple(x.shape)}"
    tables = [("shift", shift), ("scale", scale)]
    if gate is not None:
        tables.append(("gate", gate))
    for name, t in tables:
        assert (
            t.ndim == 2 and t.shape[1] == N
        ), f"{name} table must be [G, {N}], got {tuple(t.shape)}"
    assert shift.shape == scale.shape, "shift and scale must have the same shape"
    assert indices.shape == (M,), f"indices must be [{M}], got {tuple(indices.shape)}"
    assert indices.dtype in (torch.int32, torch.int64), "indices must be int32/int64"
    assert indices.is_contiguous(), "indices must be contiguous"
    for name, p in (("weight", weight), ("bias", bias)):
        assert p is None or p.shape == (
            N,
        ), f"{name} must be [{N}], got {tuple(p.shape)}"
        # The kernel indexes these with the column offset alone.
        assert p is None or p.is_contiguous(), f"{name} must be contiguous"
    if out is None:
        out = torch.empty_like(residual)
    if h_out is None:
        h_out = torch.empty_like(residual)
    operands = [("x", x), ("out", out), ("h_out", h_out)] + tables
    for name, t in operands:
        assert t.dtype == residual.dtype, f"{name} must be {residual.dtype}"
    for name, t in (("out", out), ("h_out", h_out)):
        assert t.shape == residual.shape, f"{name} must be [{M}, {N}]"
    assert residual.is_cuda, "residual must be a CUDA tensor"
    for name, t in operands + [("indices", indices)]:
        assert t.device == residual.device, f"{name} must be on {residual.device}"
    for name, t in [("residual", residual)] + operands:
        assert N <= 1 or t.stride(1) == 1, f"{name} must have unit stride along N"
    if M == 0 or N == 0:
        return out, h_out

    config = _get_config(N)
    # A row narrower than the tile shrinks it rather than mask most lanes off.
    BLOCK_N = min(config.pop("BLOCK_N"), triton.next_power_of_2(N))
    BLOCK_M = min(config.pop("BLOCK_M"), triton.next_power_of_2(M))

    _fused_indexed_gated_residual_layernorm_adaln_kernel[(triton.cdiv(M, BLOCK_M),)](
        out,
        h_out,
        residual,
        x,
        gate,
        shift,
        scale,
        weight,
        bias,
        indices,
        M,
        N,
        residual.stride(0),
        x.stride(0),
        h_out.stride(0),
        out.stride(0),
        0 if gate is None else gate.stride(0),
        shift.stride(0),
        scale.stride(0),
        eps,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        SINGLE_TILE=BLOCK_N >= N,
        HAS_GATE=gate is not None,
        ROUND_INTERMEDIATE=round_intermediate and gate is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        **config,
        # With fusion the compiler may contract the rounded product back into an
        # fma, which silently drops round_intermediate.
        enable_fp_fusion=not round_intermediate,
    )
    return out, h_out
