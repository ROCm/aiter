# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_rmsnorm_indexed_adaln import (
    _fused_gated_residual_rmsnorm_indexed_adaln_fp8_per_token_quant_kernel,
    _fused_rmsnorm_indexed_adaln_fp8_per_token_quant_kernel,
    _fused_rmsnorm_indexed_adaln_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

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
        "FUSED_RMSNORM_INDEXED_ADALN: x=%s table=%s idx=%s",
        tuple(x.shape),
        tuple(scale.shape),
        tuple(indices.shape),
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
        assert (
            out.shape == x.shape and out.dtype == x.dtype
        ), f"out must be [{M}, {N}] {x.dtype}, got {tuple(out.shape)} {out.dtype}"
        assert out.is_contiguous(), "out must be contiguous"

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


_FP8_ROW_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
# The whole row is held in registers to find its amax before quantizing.
_FP8_MAX_ROW = 16384


def _fp8_per_token_bounds(dtype_quant):
    fp8_max = float(torch.finfo(dtype_quant).max)
    # Floor that keeps an all-zero row finite, as in dynamic per-token quant.
    return fp8_max, 1.0 / (fp8_max * 512.0)


def _check_modulation(weight, shift, scale, indices, M, N, device):
    assert weight.shape == (N,), f"weight must be [{N}], got {tuple(weight.shape)}"
    assert shift.shape == scale.shape, "shift and scale must have the same shape"
    assert (
        shift.ndim == 2 and shift.shape[1] == N
    ), f"modulation table must be [G, {N}], got {tuple(shift.shape)}"
    assert indices.shape == (M,), f"indices must be [{M}], got {tuple(indices.shape)}"
    assert indices.dtype in (torch.int32, torch.int64), "indices must be int32/int64"
    assert weight.is_contiguous() and indices.is_contiguous()
    assert shift.stride(1) == 1 and scale.stride(1) == 1
    for name, t in (("weight", weight), ("shift", shift), ("scale", scale)):
        assert t.device == device, f"{name} must be on {device}"
    assert indices.device == device, f"indices must be on {device}"


def fused_rmsnorm_indexed_adaln_fp8_per_token_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float = 1e-5,
    dtype_quant: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``fused_rmsnorm_indexed_adaln`` with a dynamic per-token FP8 output.

        y[m] = rmsnorm(x[m], weight) * (1 + scale[indices[m]]) + shift[indices[m]]
        out[m] * out_scale[m] ~= y[m],  out_scale[m] = max(amax(y[m]) / FP8_MAX, 1 / (FP8_MAX * 512))

    This is the activation format of per-token FP8 GEMMs (e.g.
    ``gemm_a8w8_bpreshuffle``): when the AdaLN output feeds such a linear, the
    quantization happens here instead of in a separate pass that reads y back.
    y is rounded to ``x.dtype`` before it is quantized, so the result matches the
    unfused op followed by a dynamic per-token quantization of its output, up to
    fp32 summation order.

    Key parameters:
    - x: [M, N] activations (bf16, fp16 or fp32), unit stride along N, N <= 16384
    - weight: [N] RMSNorm gain
    - shift, scale: [G, N] modulation table, one row per index value
    - indices: [M] int32/int64, one table row per token
    - eps: RMSNorm epsilon
    - dtype_quant: FP8 dtype of ``out``; the platform's e4m3 type by default

    Returns:
    - out: [M, N] ``dtype_quant``
    - out_scale: [M, 1] fp32
    """
    _LOGGER.info(
        "FUSED_RMSNORM_INDEXED_ADALN_FP8_PER_TOKEN_QUANT: x=%s table=%s idx=%s",
        tuple(x.shape),
        tuple(scale.shape),
        tuple(indices.shape),
    )
    assert x.ndim == 2, f"x must be 2-D, got {tuple(x.shape)}"
    assert x.dtype in _FP8_ROW_DTYPES, f"unsupported dtype: {x.dtype}"
    M, N = x.shape
    assert N <= _FP8_MAX_ROW, f"rows wider than {_FP8_MAX_ROW} are not supported"
    assert x.stride(1) == 1, "x must have unit stride along N"
    _check_modulation(weight, shift, scale, indices, M, N, x.device)

    dtype_quant = dtype_quant or get_fp8_e4m3_dtype()
    fp8_max, min_scale = _fp8_per_token_bounds(dtype_quant)
    out = torch.empty((M, N), dtype=dtype_quant, device=x.device)
    out_scale = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    if M == 0:
        return out, out_scale
    _fused_rmsnorm_indexed_adaln_fp8_per_token_quant_kernel[(M,)](
        out,
        out_scale,
        x,
        weight,
        shift,
        scale,
        indices,
        x.stride(0),
        shift.stride(0),
        scale.stride(0),
        N=N,
        eps=eps,
        BLOCK_N=triton.next_power_of_2(N),
        FP8_MAX=fp8_max,
        MIN_SCALE=min_scale,
        num_warps=4,
    )
    return out, out_scale


def fused_gated_residual_rmsnorm_indexed_adaln_fp8_per_token_quant(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float = 1e-5,
    dtype_quant: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Close a block branch with a gated residual, then AdaLN to per-token FP8.

        residual_out[m] = residual[m] + gate[indices[m]] * branch[m]
        y[m] = rmsnorm(residual_out[m], weight) * (1 + scale[indices[m]]) + shift[indices[m]]
        out[m] * out_scale[m] ~= y[m]

    The step between the attention and the MLP of a diffusion transformer block:
    ``fused_indexed_gated_residual`` and
    ``fused_rmsnorm_indexed_adaln_fp8_per_token_quant`` in one pass, with the
    residual read once. ``residual_out`` is rounded to ``residual.dtype`` before
    the norm reads it, as when it is stored in between.

    Key parameters:
    - residual: [M, N] residual stream (bf16, fp16 or fp32), N <= 16384
    - gate: [G, N] gate table, indexed like shift/scale
    - branch: [M, N] branch output
    - weight, shift, scale, indices, eps, dtype_quant: as in
      ``fused_rmsnorm_indexed_adaln_fp8_per_token_quant``

    Returns:
    - residual_out: [M, N], same dtype as residual
    - out: [M, N] ``dtype_quant``
    - out_scale: [M, 1] fp32
    """
    _LOGGER.info(
        "FUSED_GATED_RESIDUAL_RMSNORM_INDEXED_ADALN_FP8_PER_TOKEN_QUANT: residual=%s table=%s idx=%s",
        tuple(residual.shape),
        tuple(scale.shape),
        tuple(indices.shape),
    )
    assert residual.ndim == 2, f"residual must be 2-D, got {tuple(residual.shape)}"
    assert residual.dtype in _FP8_ROW_DTYPES, f"unsupported dtype: {residual.dtype}"
    M, N = residual.shape
    assert N <= _FP8_MAX_ROW, f"rows wider than {_FP8_MAX_ROW} are not supported"
    assert branch.shape == (
        M,
        N,
    ), f"branch must be [{M}, {N}], got {tuple(branch.shape)}"
    # indices are bounded by the modulation table, so the gate table must match it.
    assert (
        gate.shape == shift.shape
    ), f"gate table must match shift/scale {tuple(shift.shape)}, got {tuple(gate.shape)}"
    assert residual.stride(1) == 1 and branch.stride(1) == 1 and gate.stride(1) == 1
    assert branch.device == residual.device and gate.device == residual.device
    _check_modulation(weight, shift, scale, indices, M, N, residual.device)

    dtype_quant = dtype_quant or get_fp8_e4m3_dtype()
    fp8_max, min_scale = _fp8_per_token_bounds(dtype_quant)
    residual_out = torch.empty((M, N), dtype=residual.dtype, device=residual.device)
    out = torch.empty((M, N), dtype=dtype_quant, device=residual.device)
    out_scale = torch.empty((M, 1), dtype=torch.float32, device=residual.device)
    if M == 0:
        return residual_out, out, out_scale
    _fused_gated_residual_rmsnorm_indexed_adaln_fp8_per_token_quant_kernel[(M,)](
        residual_out,
        out,
        out_scale,
        residual,
        gate,
        branch,
        weight,
        shift,
        scale,
        indices,
        residual.stride(0),
        gate.stride(0),
        branch.stride(0),
        shift.stride(0),
        scale.stride(0),
        N=N,
        eps=eps,
        BLOCK_N=triton.next_power_of_2(N),
        FP8_MAX=fp8_max,
        MIN_SCALE=min_scale,
        num_warps=16,
        # An fma would skip rounding the gate * branch product, unlike eager fp32.
        enable_fp_fusion=False,
    )
    return residual_out, out, out_scale
