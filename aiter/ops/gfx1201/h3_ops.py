# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-H3 gfx1201 HIP row/element-wise ops (``module_gfx1201_h3_ops``).

DiT block (BF16): ``rms_modulate`` (RMSNorm + adaLN modulation), ``swiglu`` and ``gated_residual``
(bitwise equal to the Triton ``aiter.ops.gfx1201.gated_residual.gated_residual``).
Video VAE ViT decoder block (FP32 residual stream, FP16 activations): ``vae_residual_rms``,
``vae_qk_norm_rope`` and ``vae_swiglu``. All ops run on the current stream and only support inference.
"""

import torch
from torch import Tensor

from aiter.jit.core import compile_ops

MODULATION_DIM = 5376
VAE_THREADS, VAE_MAX_CHUNKS = 256, 4


@compile_ops("module_gfx1201_h3_ops", fc_name="gfx1201_rms_modulate_hip", develop=True)
def gfx1201_rms_modulate_hip(
    x: Tensor,
    weight: Tensor,
    scale: Tensor,
    shift: Tensor,
    indices: Tensor,
    out: Tensor,
    indices_int64: bool,
    eps: float,
) -> None:
    """out = bf16(bf16(bf16(w * x * rstd) * bf16(1 + scale[g])) + shift[g]), g = indices[row]."""


@compile_ops("module_gfx1201_h3_ops", fc_name="gfx1201_swiglu_hip", develop=True)
def gfx1201_swiglu_hip(inputs: Tensor, out: Tensor) -> None:
    """out = value * silu(gate) for BF16 [rows, value | gate]."""


@compile_ops(
    "module_gfx1201_h3_ops", fc_name="gfx1201_gated_residual_hip", develop=True
)
def gfx1201_gated_residual_hip(
    residual: Tensor,
    projected: Tensor,
    gate: Tensor,
    indices: Tensor,
    out: Tensor,
    indices_int64: bool,
    vectorized: bool,
) -> None:
    """out = residual + bf16(gate[indices] * projected)."""


@compile_ops(
    "module_gfx1201_h3_ops", fc_name="gfx1201_vae_residual_rms_hip", develop=True
)
def gfx1201_vae_residual_rms_hip(
    hidden: Tensor, addend: int, scale: int, weight: int, out: int, dim: int, eps: float
) -> None:
    """hidden += addend * scale (pointers, 0 = skip); out = fp16(rms_norm(hidden) * weight) (0 = skip)."""


@compile_ops(
    "module_gfx1201_h3_ops", fc_name="gfx1201_vae_qk_norm_rope_hip", develop=True
)
def gfx1201_vae_qk_norm_rope_hip(
    query: Tensor, key: Tensor, cos16: Tensor, sin16: Tensor, heads: int, eps: float
) -> None:
    """In-place per-head RMSNorm + partial RoPE of FP16 query/key rows."""


@compile_ops("module_gfx1201_h3_ops", fc_name="gfx1201_vae_swiglu_hip", develop=True)
def gfx1201_vae_swiglu_hip(inputs: Tensor, out: Tensor) -> None:
    """out = value * silu(gate) for FP16 [rows, value | gate]."""


def _address(tensor):
    return 0 if tensor is None else tensor.data_ptr()


def _indices_int64(indices, rows, device):
    if (
        indices.shape != (rows,)
        or indices.device != device
        or not indices.is_contiguous()
        or indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "Expected contiguous per-token int32/int64 indices on the same GPU"
        )
    return indices.dtype == torch.int64


def rms_modulate(x, weight, scale, shift, indices, eps):
    """Fused ``modulate(RMSNorm(x), scale, shift, indices)`` for the MiniMax-H3 DiT (D = 5376).

    x: contiguous BF16 ``[S, D]``; weight: BF16 ``[D]``; scale/shift: BF16 ``[groups, D]`` with unit column
    stride and row stride % 8 == 0; indices: per-token int32/int64 group of ``[S]``. Eager rounding:
    ``bf16(bf16(bf16(weight * x * rstd) * bf16(1 + scale[g])) + shift[g])`` with ``g = indices[row]``.
    """
    if (
        x.ndim != 2
        or x.shape[1] != MODULATION_DIM
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
    ):
        raise ValueError(
            f"Expected contiguous BF16 GPU x [S,{MODULATION_DIM}], got {tuple(x.shape)}"
        )
    if (
        weight.dtype != torch.bfloat16
        or weight.numel() != MODULATION_DIM
        or not weight.is_contiguous()
        or weight.device != x.device
    ):
        raise ValueError(
            f"Expected contiguous BF16 weight [{MODULATION_DIM}] on the same GPU"
        )
    for tensor in (scale, shift):
        if (
            tensor.dtype != torch.bfloat16
            or tensor.ndim != 2
            or tensor.shape[1] != MODULATION_DIM
            or tensor.stride(1) != 1
            or tensor.stride(0) % 8 != 0
            or tensor.device != x.device
        ):
            raise ValueError(
                f"Expected BF16 [groups,{MODULATION_DIM}] scale/shift with unit column stride and row stride % 8 == 0"
            )
    if scale.shape[0] != shift.shape[0]:
        raise ValueError("Expected scale and shift with the same group count")
    indices_int64 = _indices_int64(indices, x.shape[0], x.device)
    out = torch.empty_like(x)
    if x.shape[0]:
        gfx1201_rms_modulate_hip(
            x, weight, scale, shift, indices, out, indices_int64, float(eps)
        )
    return out


def swiglu(x):
    """``value * silu(gate)`` for a contiguous BF16 ``[S, 2F]`` up-projection laid out ``[value | gate]``, F % 8 == 0.

    Bitwise equal to eager ``value * F.silu(gate)``; returns ``[S, F]``.
    """
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[1] % 16 != 0
    ):
        raise ValueError(
            f"Expected contiguous BF16 GPU [S,2F] with F % 8 == 0, got {tuple(x.shape)}"
        )
    out = torch.empty((x.shape[0], x.shape[1] // 2), device=x.device, dtype=x.dtype)
    if x.shape[0]:
        gfx1201_swiglu_hip(x, out)
    return out


def gated_residual(residual, projected, gate, indices):
    """``residual + bf16(gate[indices] * projected)`` with FP32 math: HIP twin of the Triton
    ``aiter.ops.gfx1201.gated_residual.gated_residual`` (bitwise equal), 8-wide when shapes and alignment allow.
    """
    if residual.ndim != 2 or residual.shape != projected.shape or residual.numel() == 0:
        raise ValueError(
            "Expected matching positive [S,D] residual and projected tensors"
        )
    if any(
        tensor.device != residual.device or tensor.dtype != torch.bfloat16
        for tensor in (residual, projected, gate)
    ):
        raise ValueError("Expected BF16 tensors on one GPU")
    if (
        not residual.is_cuda
        or not residual.is_contiguous()
        or not projected.is_contiguous()
    ):
        raise ValueError("Expected contiguous GPU residual and projected tensors")
    if (
        gate.ndim != 2
        or gate.shape[1] != residual.shape[1]
        or gate.shape[0] < 1
        or gate.stride(1) != 1
    ):
        raise ValueError("Expected [groups,D] gate with contiguous columns")
    indices_int64 = _indices_int64(indices, residual.shape[0], residual.device)
    if any(tensor.requires_grad for tensor in (residual, projected, gate)):
        raise ValueError("Inference only")
    vectorized = (
        residual.shape[1] % 8 == 0
        and gate.stride(0) % 8 == 0
        and all(tensor.data_ptr() % 16 == 0 for tensor in (gate, residual, projected))
    )
    out = torch.empty_like(residual)
    gfx1201_gated_residual_hip(
        residual, projected, gate, indices, out, indices_int64, vectorized
    )
    return out


def vae_residual_rms(hidden, addend=None, scale=None, weight=None, out=None, eps=0.0):
    """VAE decoder block residual + RMSNorm, in place on the FP32 residual stream.

    ``hidden += addend * scale`` when ``addend`` (FP16, same numel) and ``scale`` (FP32 ``[D]``) are given, then
    ``out = fp16(RMSNorm(hidden) * weight)`` when ``out`` (FP16, same numel) and ``weight`` (FP32 ``[D]``) are given.
    hidden: contiguous FP32 ``[..., D]`` with D % 2048 == 0 and D <= 8192.
    """
    if (
        not hidden.is_cuda
        or hidden.dtype != torch.float32
        or not hidden.is_contiguous()
        or hidden.ndim < 1
    ):
        raise ValueError("Expected contiguous FP32 GPU hidden states")
    dim = hidden.shape[-1]
    if dim % (VAE_THREADS * 8) != 0 or dim // (VAE_THREADS * 8) > VAE_MAX_CHUNKS:
        raise ValueError(f"Unsupported hidden size {dim}")
    if (addend is None) != (scale is None) or (weight is None) != (out is None):
        raise ValueError("Expected addend with scale and weight with out")
    for tensor, dtype, numel in (
        (addend, torch.float16, hidden.numel()),
        (scale, torch.float32, dim),
        (weight, torch.float32, dim),
        (out, torch.float16, hidden.numel()),
    ):
        if tensor is not None and (
            tensor.dtype != dtype
            or tensor.numel() != numel
            or not tensor.is_contiguous()
            or tensor.device != hidden.device
        ):
            raise ValueError(
                "Expected contiguous FP16 addend/out like hidden and FP32 [D] scale/weight on the same GPU"
            )
    gfx1201_vae_residual_rms_hip(
        hidden,
        _address(addend),
        _address(scale),
        _address(weight),
        _address(out),
        dim,
        float(eps),
    )


def vae_qk_norm_rope(query, key, cos16, sin16, heads, eps):
    """In-place per-head (64) RMSNorm (no affine) + RoPE on the first 48 dims of FP16 query/key ``[rows, heads*64]``
    views (unit column stride, row stride % 8 == 0); cos16/sin16: contiguous FP16 ``[rows, 48]``.
    """
    for tensor in (query, key):
        if (
            tensor.dtype != torch.float16
            or tensor.ndim != 2
            or tensor.stride(1) != 1
            or tensor.stride(0) % 8 != 0
            or tensor.shape != (query.shape[0], heads * 64)
            or tensor.device != query.device
        ):
            raise ValueError(
                f"Expected FP16 [rows,{heads * 64}] query/key with unit column stride and row stride % 8 == 0"
            )
    for tensor in (cos16, sin16):
        if (
            tensor.dtype != torch.float16
            or not tensor.is_contiguous()
            or tensor.shape != (query.shape[0], 48)
        ):
            raise ValueError("Expected contiguous FP16 cos16/sin16 [rows,48]")
    gfx1201_vae_qk_norm_rope_hip(query, key, cos16, sin16, int(heads), float(eps))


def vae_swiglu(x):
    """``value * silu(gate)`` for an FP16 ``[rows, 2F]`` view laid out ``[value | gate]`` (unit column stride, row
    stride % 8 == 0, F % 8 == 0); returns contiguous ``[rows, F]``."""
    if (
        x.dtype != torch.float16
        or x.ndim != 2
        or x.stride(1) != 1
        or x.shape[1] % 16 != 0
        or x.stride(0) % 8 != 0
    ):
        raise ValueError(
            f"Expected FP16 [rows,2F] with F % 8 == 0 and row stride % 8 == 0, got {tuple(x.shape)}"
        )
    out = torch.empty((x.shape[0], x.shape[1] // 2), device=x.device, dtype=x.dtype)
    gfx1201_vae_swiglu_hip(x, out)
    return out
