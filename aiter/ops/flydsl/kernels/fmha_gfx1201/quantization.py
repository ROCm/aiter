# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.

"""FP8 quantization selection and fallbacks for gfx1201 flash attention."""

from __future__ import annotations

import math
import threading

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

from .fp8_quant import flydsl_fp8_pertensor_quant
from .stream_readiness import register_ready, wait_ready

_FP8_MAX = 448.0
_ROT_ROWS = 64
_HADAMARD_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
_HADAMARD_CACHE_LOCK = threading.Lock()

if _HAS_TRITON:

    @triton.jit
    def _rot_amax3(
        q, k, v, matrix, aq, ak, av, rows, D: tl.constexpr, BLOCK_ROWS: tl.constexpr
    ):
        pid = tl.program_id(0)
        row_blocks = tl.cdiv(rows, BLOCK_ROWS)
        tensor_id = pid // row_blocks
        row = pid % row_blocks * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        mask = row < rows
        source = tl.where(tensor_id == 0, q, tl.where(tensor_id == 1, k, v))
        offsets = row[:, None] * D + tl.arange(0, D)[None, :]
        values = tl.load(source + offsets, mask=mask[:, None], other=0.0)
        values_fp32 = values.to(tl.float32)
        # tensor_id is uniform for the whole program. Branch before loading the
        # matrix and issuing tl.dot so V programs do neither operation.
        if tensor_id < 2:
            rotation = tl.load(
                matrix + tl.arange(0, D)[:, None] * D + tl.arange(0, D)[None, :]
            )
            values_fp32 = tl.dot(values, rotation)
        maximum = tl.max(tl.abs(values_fp32))
        output = tl.where(tensor_id == 0, aq, tl.where(tensor_id == 1, ak, av))
        tl.atomic_max(output, maximum)

    @triton.jit
    def _rot_scale3(
        q,
        k,
        v,
        matrix,
        q8,
        k8,
        v8,
        sq,
        sk,
        sv,
        rows,
        D: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_blocks = tl.cdiv(rows, BLOCK_ROWS)
        tensor_id = pid // row_blocks
        row = pid % row_blocks * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        mask = row < rows
        source = tl.where(tensor_id == 0, q, tl.where(tensor_id == 1, k, v))
        output = tl.where(tensor_id == 0, q8, tl.where(tensor_id == 1, k8, v8))
        scale = tl.where(tensor_id == 0, sq, tl.where(tensor_id == 1, sk, sv))
        offsets = row[:, None] * D + tl.arange(0, D)[None, :]
        values = tl.load(source + offsets, mask=mask[:, None], other=0.0)
        values_fp32 = values.to(tl.float32)
        # As above, V is a uniform program branch and bypasses rotation work.
        if tensor_id < 2:
            rotation = tl.load(
                matrix + tl.arange(0, D)[:, None] * D + tl.arange(0, D)[None, :]
            )
            values_fp32 = tl.dot(values, rotation)
        values_fp32 = values_fp32 / tl.load(scale)
        values_fp32 = tl.minimum(tl.maximum(values_fp32, -448.0), 448.0)
        tl.store(
            output + offsets,
            values_fp32.to(output.dtype.element_ty),
            mask=mask[:, None],
        )


def _hadamard_matrix(head_dim: int, device, dtype) -> torch.Tensor | None:
    if head_dim <= 0 or head_dim & (head_dim - 1):
        return None
    device = torch.device(device)
    stream = torch.cuda.current_stream(device)
    key = (head_dim, device, dtype)
    with _HADAMARD_CACHE_LOCK:
        matrix = _HADAMARD_CACHE.get(key)
        if matrix is None:
            with torch.cuda.device(device.index), torch.cuda.stream(stream):
                matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
                while matrix.shape[0] < head_dim:
                    matrix = torch.cat(
                        [
                            torch.cat([matrix, matrix], dim=1),
                            torch.cat([matrix, -matrix], dim=1),
                        ],
                        dim=0,
                    )
                matrix = (matrix / math.sqrt(head_dim)).to(dtype)
                (matrix,) = register_ready((matrix,), stream=stream)
                _HADAMARD_CACHE[key] = matrix
        else:
            wait_ready(stream, (matrix,))
            matrix.record_stream(stream)
    return matrix


def _torch_fp8_quant(x: torch.Tensor, matrix: torch.Tensor | None):
    if matrix is not None:
        x = torch.matmul(x, matrix)
    scale = (x.abs().max().float() / _FP8_MAX).clamp(min=1e-12).reshape(1)
    quantized = (x.float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return quantized, scale


def _live_gfx(device=None) -> str:
    try:
        arch = torch.cuda.get_device_properties(device).gcnArchName
    except Exception:  # noqa: BLE001
        arch = ""
    if arch:
        return arch.lower().split(":")[0]
    try:
        return get_gfx_runtime()
    except Exception:  # noqa: BLE001
        return ""


def flydsl_fp8_quant(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    rotation: bool = True,
    backend: str = "flydsl",
):
    """Quantize bf16 Q/K/V for the gfx1201 per-tensor FP8 attention path.

    ``backend`` is a preference: unsupported native or Triton configurations
    fall back to another available implementation with the same result contract.
    """
    if not isinstance(backend, str):
        raise TypeError(f"backend must be a string, got {type(backend).__name__}")
    backend = backend.lower()
    if backend not in {"flydsl", "triton", "torch"}:
        raise ValueError(f"unsupported fp8 quant backend {backend!r}")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q/k/v must be CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError("q/k/v must reside on the same device")
    if not (q.dtype == k.dtype == v.dtype) or q.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        raise ValueError("q/k/v dtype must match and be bf16 or fp16")
    if any(x.dim() == 0 for x in (q, k, v)) or not (
        q.shape[-1] == k.shape[-1] == v.shape[-1]
    ):
        raise ValueError("q/k/v must share head_dim")
    if any(x.numel() == 0 for x in (q, k, v)):
        raise ValueError("q/k/v must be non-empty")

    head_dim = q.shape[-1]
    operation_stream = torch.cuda.current_stream(q.device)
    wait_ready(operation_stream, (q, k, v))
    for tensor in (q, k, v):
        tensor.record_stream(operation_stream)
    if (
        backend == "flydsl"
        and q.dtype == torch.bfloat16
        and head_dim == 128
        and _live_gfx(q.device) == "gfx1201"
    ):
        q8, sq = flydsl_fp8_pertensor_quant(q, rotate=rotation)
        k8, sk = flydsl_fp8_pertensor_quant(k, rotate=rotation)
        v8, sv = flydsl_fp8_pertensor_quant(v, rotate=False)
        # Each low-level producer already registers its output and descale on
        # operation_stream; retaining those three events avoids a redundant
        # grouped event while preserving readiness for all six tensors.
        return q8, k8, v8, sq, sk, sv

    with torch.cuda.device(q.device.index), torch.cuda.stream(operation_stream):
        matrix = _hadamard_matrix(head_dim, q.device, q.dtype) if rotation else None
        same_size = q.numel() == k.numel() == v.numel()
        if backend != "torch" and _HAS_TRITON and matrix is not None and same_size:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            for tensor in (q, k, v):
                tensor.record_stream(operation_stream)
            rows = q.numel() // head_dim
            grid = (3 * triton.cdiv(rows, _ROT_ROWS),)
            aq = torch.zeros(1, dtype=torch.float32, device=q.device)
            ak = torch.zeros(1, dtype=torch.float32, device=q.device)
            av = torch.zeros(1, dtype=torch.float32, device=q.device)
            _rot_amax3[grid](
                q, k, v, matrix, aq, ak, av, rows, D=head_dim, BLOCK_ROWS=_ROT_ROWS
            )
            sq = (aq / _FP8_MAX).clamp(min=1e-12)
            sk = (ak / _FP8_MAX).clamp(min=1e-12)
            sv = (av / _FP8_MAX).clamp(min=1e-12)
            q8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
            k8 = torch.empty_like(k, dtype=torch.float8_e4m3fn)
            v8 = torch.empty_like(v, dtype=torch.float8_e4m3fn)
            _rot_scale3[grid](
                q,
                k,
                v,
                matrix,
                q8,
                k8,
                v8,
                sq,
                sk,
                sv,
                rows,
                D=head_dim,
                BLOCK_ROWS=_ROT_ROWS,
            )
            return register_ready((q8, k8, v8, sq, sk, sv), stream=operation_stream)

        q8, sq = _torch_fp8_quant(q, matrix)
        k8, sk = _torch_fp8_quant(k, matrix)
        v8, sv = _torch_fp8_quant(v, None)
        return register_ready((q8, k8, v8, sq, sk, sv), stream=operation_stream)
