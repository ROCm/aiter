# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Flash Attention APIs.

``flydsl_flash_attn_batch_func`` / ``flydsl_flash_attn_varlen_func`` dispatch by
arch and dtype: gfx950 fp8 to ``kernels/fmha_gfx950``, gfx1250 bf16/f16 to the
m32x8 prefill kernel, anything else ``None`` so the caller falls through to
CK/Triton.

``flydsl_flash_attn_func`` (gfx1201 / RDNA4) wraps the
`flash_attn_func_gfx1201` kernel with:
  - Build cache keyed by shape, dtype, masking, and tuning parameters.
  - Automatic Q and KV sequence padding to their respective tile sizes.
  - BSHD ([B, S, H, D]) input/output convention to match upstream
    flash-attention layout.
  - Exact masking of padded non-causal K/V columns.
  - Non-causal cross-attention with independent Q and KV sequence lengths.
"""

from __future__ import annotations

import math
import os
import threading
from functools import lru_cache

import torch
import torch.nn.functional as F

from aiter.jit.utils.chip_info import get_gfx_runtime

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

from .fmha_bwd_gfx942 import flash_attn_varlen_bwd_d192_gfx942
from .kernels.flash_attn_func_fp8_gfx1201 import (
    build_flash_attn_func_module as build_flash_attn_fp8_func_module,
)
from .kernels.flash_attn_func_fp8_gfx1201 import get_flash_attn_fp8_lds_bytes
from .kernels.flash_attn_func_gfx1201 import (
    build_flash_attn_func_module,
    get_flash_attn_lds_bytes,
)
from .kernels.fmha_gfx1250.fmha_fwd_prefill_a16w16_m32x8 import (
    flash_attn_batch_m32x8,
    flash_attn_varlen_m32x8,
)
from .kernels.fp8_quant_gfx1201 import flydsl_fp8_pertensor_quant
from .stream_readiness import register_ready, wait_ready

__all__ = [
    "flydsl_flash_attn_batch_func",
    "flydsl_flash_attn_func",
    "flydsl_flash_attn_varlen_bwd",
    "flydsl_flash_attn_varlen_func",
    "flydsl_fp8_quant",
]

_FP8_DTYPES = (torch.float8_e4m3fn,)
_FP8_MAX = 448.0
_GFX1201_LDS_CAPACITY_BYTES = 65536
_GFX1201_KERNEL_INT32_MAX = (1 << 31) - 1
_GFX1201_BUFFER_MAX_BYTES = 1 << 32
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


def _pick_gfx1201_tiles(seq_len: int, head_dim: int, causal: bool) -> tuple[int, int]:
    """Select the best known gfx1201 tile for the production shape envelope."""
    if causal:
        return 128, 32
    if head_dim <= 64:
        return 128, 64
    if seq_len < 2048:
        return 128, 32
    return 256, 64


def _gfx1201_fmha_lds_bytes(head_dim: int, block_n: int, *, fp8: bool) -> int:
    """Return the exact static LDS allocation for a selected gfx1201 kernel."""
    if fp8:
        return get_flash_attn_fp8_lds_bytes(head_dim, block_n)
    return get_flash_attn_lds_bytes(head_dim, block_n)


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "f16"
    raise ValueError(f"flydsl_flash_attn_func only supports bf16/f16, got {dtype!r}")


def _storage_byte_range(tensor: torch.Tensor) -> tuple[int, int]:
    """Return the half-open byte range touched by a non-empty tensor view."""
    storage_ptr = tensor.untyped_storage().data_ptr()
    first = tensor.data_ptr() - storage_ptr
    last = first
    element_size = tensor.element_size()
    for size, stride in zip(tensor.shape, tensor.stride()):
        delta = (size - 1) * stride * element_size
        first += min(0, delta)
        last += max(0, delta)
    return first, last + element_size


def _storage_overlaps(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if lhs.device != rhs.device:
        return False
    if lhs.untyped_storage().data_ptr() != rhs.untyped_storage().data_ptr():
        return False
    lhs_first, lhs_last = _storage_byte_range(lhs)
    rhs_first, rhs_last = _storage_byte_range(rhs)
    return lhs_first < rhs_last and rhs_first < lhs_last


def _has_unsupported_internal_overlap(tensor: torch.Tensor) -> bool:
    checker = getattr(torch, "_debug_has_internal_overlap", None)
    if checker is None:
        # Older PyTorch versions do not expose the overlap checker. Accept only
        # contiguous storage there rather than risk concurrent writes aliasing.
        return not tensor.is_contiguous()
    # 0 means no overlap. Treat both definite overlap and "too hard" as unsafe.
    return int(checker(tensor)) != 0


def _validate_gfx1201_launch_limits(
    *,
    seq_len: int,
    seq_len_kv_real: int,
    seq_len_kv: int,
    num_heads: int,
    head_dim: int,
    fp8: bool,
) -> None:
    """Validate integer kernel arguments and BF16/F16 V descriptor capacity."""
    for name, value in (
        ("padded query sequence length", seq_len),
        ("real KV sequence length", seq_len_kv_real),
        ("padded KV sequence length", seq_len_kv),
    ):
        if value > _GFX1201_KERNEL_INT32_MAX:
            raise ValueError(
                f"{name}={value} exceeds the gfx1201 kernel Int32 limit "
                f"({_GFX1201_KERNEL_INT32_MAX})"
            )
    if not fp8:
        v_batch_bytes = seq_len_kv * num_heads * head_dim * 2
        if v_batch_bytes >= _GFX1201_BUFFER_MAX_BYTES:
            raise ValueError(
                f"one BF16/F16 V batch requires {v_batch_bytes} bytes, but the "
                "gfx1201 buffer descriptor requires a byte count below 2^32"
            )


@lru_cache(maxsize=64)
def _get_kernel(
    device_index: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    waves_per_eu: int,
    daz: bool,
    block_m: int,
    block_n: int,
    softmax_scale: float | None,
    tail_mask: bool,
    cross_attn: bool,
    lds_vec_width: int,
):
    # device_index intentionally participates in the cache key; the builder
    # observes the active device selected by the caller's device context.
    # lds_vec_width participates because it changes the cooperative load layout.
    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        waves_per_eu=waves_per_eu,
        daz=daz,
        block_m=block_m,
        block_n=block_n,
        sm_scale=softmax_scale,
        tail_mask=tail_mask,
        cross_attn=cross_attn,
        lds_vec_width=lds_vec_width,
    )


@lru_cache(maxsize=64)
def _get_fp8_gfx1201_kernel(
    device_index: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
    waves_per_eu: int,
    daz: bool,
    block_m: int,
    block_n: int,
    softmax_scale: float | None,
    tail_mask: bool,
    cross_attn: bool,
):
    # device_index intentionally participates in the cache key.
    return build_flash_attn_fp8_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str="bf16",
        waves_per_eu=waves_per_eu,
        daz=daz,
        block_m=block_m,
        block_n=block_n,
        sm_scale=softmax_scale,
        tail_mask=tail_mask,
        cross_attn=cross_attn,
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


def flydsl_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: torch.cuda.Stream | None = None,
    softmax_scale: float | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run FlyDSL Flash Attention on RDNA4 (gfx1201).

    Args:
        q: tensor with shape ``[batch, seqlen_q, num_heads, head_dim]`` (BSHD).
        k, v: tensors with shape ``[batch, seqlen_kv, num_heads, head_dim]``.
            All three inputs must share dtype, batch, num_heads, and head_dim.
        causal: apply causal masking when ``True``. Causal cross-attention is
            not supported.
        waves_per_eu: kernel occupancy hint passed to the FlyDSL builder.
        daz: enable denormals-are-zero on the kernel.
        stream: optional CUDA/HIP stream to launch on. Defaults to the current
            stream for ``q.device``.
        softmax_scale: optional positive finite QK scale. Defaults to
            ``1 / sqrt(head_dim)``.
        q_descale, k_descale, v_descale: one-element float32 device tensors
            required for FP8 input.
        out: optional output tensor. It must not alias an input. FP8 input
            requires BF16 output.

    Returns:
        Output with Q's shape. Its dtype matches Q for BF16/F16 input and is
        BF16 for FP8 input.

    Raises:
        ValueError: if shapes/dtypes/devices are incompatible, dimensions are
            empty, the output aliases an input, or the kernel's ``head_dim``
            constraints are not met.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_func requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(
            "q/k/v must reside on the same device, got "
            f"q={q.device} k={k.device} v={v.device}"
        )
    try:
        arch = torch.cuda.get_device_properties(q.device.index).gcnArchName
    except Exception:  # noqa: BLE001
        arch = ""
    arch_base = arch.lower().split(":")[0] if arch else ""
    if not arch_base.startswith("gfx1201"):
        raise ValueError(f"flydsl_flash_attn_func requires gfx1201, got {arch!r}")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"expected 4D BSHD tensors, got {q.dim()}/{k.dim()}/{v.dim()} dimensions"
        )
    if k.shape != v.shape:
        raise ValueError(
            f"k/v shapes must match, got {tuple(k.shape)}/{tuple(v.shape)}"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"q/k/v dtype must match: {q.dtype}/{k.dtype}/{v.dtype}")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError(
            "q/k must share batch, num_heads, and head dimension, got "
            f"{tuple(q.shape)}/{tuple(k.shape)}"
        )

    batch, seq_len_real, num_heads, head_dim = q.shape
    seq_len_kv_real = k.shape[1]
    if batch == 0 or seq_len_real == 0 or seq_len_kv_real == 0 or num_heads == 0:
        raise ValueError(
            "batch, query sequence length, KV sequence length, and num_heads "
            "must all be non-zero"
        )
    is_cross = seq_len_real != seq_len_kv_real
    if causal and is_cross:
        raise ValueError("causal cross-attention is not supported")
    if head_dim < 64 or head_dim % 32 != 0:
        raise ValueError(
            f"kernel requires head_dim >= 64 and head_dim % 32 == 0, got {head_dim}"
        )

    if softmax_scale is not None:
        if torch.is_tensor(softmax_scale):
            if softmax_scale.numel() != 1:
                raise ValueError("softmax_scale must be a scalar")
            if softmax_scale.device.type != "cpu":
                raise ValueError("tensor softmax_scale must be on CPU")
            softmax_scale = softmax_scale.item()
        softmax_scale = float(softmax_scale)
        if not math.isfinite(softmax_scale) or softmax_scale <= 0:
            raise ValueError("softmax_scale must be finite and positive")

    is_fp8 = q.dtype in _FP8_DTYPES
    if is_fp8:
        for name, scale in (
            ("q_descale", q_descale),
            ("k_descale", k_descale),
            ("v_descale", v_descale),
        ):
            if (
                not torch.is_tensor(scale)
                or scale.dtype != torch.float32
                or scale.numel() != 1
                or scale.device != q.device
            ):
                raise ValueError(
                    f"{name} must be a one-element float32 tensor on {q.device}"
                )
        dtype_str = "bf16"
    else:
        dtype_str = _torch_dtype_to_str(q.dtype)

    output_dtype = torch.bfloat16 if is_fp8 else q.dtype
    if out is not None:
        if out.shape != q.shape:
            raise ValueError(f"out must have shape {tuple(q.shape)}")
        if out.dtype != output_dtype:
            raise ValueError(f"out must have dtype {output_dtype}")
        if out.device != q.device:
            raise ValueError(f"out must be on {q.device}")
        if _has_unsupported_internal_overlap(out):
            raise ValueError(
                "out must not have internal overlap or unsupported striding"
            )
        alias_inputs = [q, k, v]
        if is_fp8:
            alias_inputs.extend((q_descale, k_descale, v_descale))
        if any(_storage_overlaps(out, tensor) for tensor in alias_inputs):
            raise ValueError("out must not overlap q, k, v, or FP8 descale storage")

    block_m, block_n = _pick_gfx1201_tiles(seq_len_real, head_dim, causal)
    lds_bytes = _gfx1201_fmha_lds_bytes(head_dim, block_n, fp8=is_fp8)
    if lds_bytes > _GFX1201_LDS_CAPACITY_BYTES and block_n != 32:
        # Keep the selected query tile, but fall back to the narrower KV tile
        # before rejecting a shape that the kernel can safely represent.
        block_n = 32
        lds_bytes = _gfx1201_fmha_lds_bytes(head_dim, block_n, fp8=is_fp8)
    if lds_bytes > _GFX1201_LDS_CAPACITY_BYTES:
        kernel_kind = "FP8" if is_fp8 else dtype_str.upper()
        raise ValueError(
            f"gfx1201 {kernel_kind} attention with head_dim={head_dim} and "
            f"BLOCK_N={block_n} requires {lds_bytes} bytes of LDS, exceeding "
            f"the {_GFX1201_LDS_CAPACITY_BYTES}-byte hardware limit"
        )

    # Pad sequence lengths to their respective tile sizes. Non-causal padded
    # K/V columns are masked in the kernel; padded query rows are sliced off.
    seq_len_pad = ((seq_len_real + block_m - 1) // block_m) * block_m
    seq_len_kv_pad = (
        seq_len_pad
        if not is_cross
        else ((seq_len_kv_real + block_n - 1) // block_n) * block_n
    )
    tail_mask = not causal and seq_len_kv_real % block_n != 0
    _validate_gfx1201_launch_limits(
        seq_len=seq_len_pad,
        seq_len_kv_real=seq_len_kv_real,
        seq_len_kv=seq_len_kv_pad,
        num_heads=num_heads,
        head_dim=head_dim,
        fp8=is_fp8,
    )

    def _pad_seq(tensor: torch.Tensor, pad: int) -> torch.Tensor:
        tensor = tensor.contiguous()
        if pad == 0:
            return tensor
        if tensor.dtype in _FP8_DTYPES:
            # Some backends do not implement constant padding for float8.
            # Padding the byte representation is equivalent because FP8 is
            # one byte per element and the all-zero bit pattern represents 0.
            return F.pad(tensor.view(torch.uint8), (0, 0, 0, 0, 0, pad)).view(
                tensor.dtype
            )
        return F.pad(tensor, (0, 0, 0, 0, 0, pad))

    # Wrap kernel build + launch in q.device context so multi-GPU callers
    # whose current device differs from q.device get the kernel compiled
    # and launched on the right device/stream.
    with torch.cuda.device(q.device.index):
        if stream is not None and not isinstance(stream, torch.cuda.Stream):
            raise TypeError(
                "stream must be a torch.cuda.Stream or None, got "
                f"{type(stream).__name__}"
            )
        launch_stream = (
            torch.cuda.current_stream(q.device) if stream is None else stream
        )
        if launch_stream.device != q.device:
            raise ValueError(
                f"`stream` must be on {q.device}, got {launch_stream.device}"
            )
        producer_stream = torch.cuda.current_stream(q.device)
        if launch_stream != producer_stream:
            launch_stream.wait_stream(producer_stream)
        wait_ready(launch_stream, (q, k, v))
        if is_fp8:
            wait_ready(launch_stream, (q_descale, k_descale, v_descale))
        if out is not None:
            wait_ready(launch_stream, (out,))

        with torch.cuda.stream(launch_stream):
            q_p = _pad_seq(q, seq_len_pad - seq_len_real)
            k_p = _pad_seq(k, seq_len_kv_pad - seq_len_kv_real)
            v_p = _pad_seq(v, seq_len_kv_pad - seq_len_kv_real)
            write_in_place = (
                out is not None and seq_len_pad == seq_len_real and out.is_contiguous()
            )
            o_p = out if write_in_place else torch.empty_like(q_p, dtype=output_dtype)

            if is_fp8:
                exe = _get_fp8_gfx1201_kernel(
                    device_index=q.device.index,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    causal=causal,
                    waves_per_eu=waves_per_eu,
                    daz=daz,
                    block_m=block_m,
                    block_n=block_n,
                    softmax_scale=softmax_scale,
                    tail_mask=tail_mask,
                    cross_attn=is_cross,
                )
                exe(
                    q_p.reshape(-1),
                    k_p.reshape(-1),
                    v_p.reshape(-1),
                    o_p.reshape(-1),
                    batch,
                    seq_len_pad,
                    seq_len_kv_real,
                    seq_len_kv_pad,
                    q_descale,
                    k_descale,
                    v_descale,
                    stream=launch_stream,
                )
            else:
                exe = _get_kernel(
                    device_index=q.device.index,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    causal=causal,
                    dtype_str=dtype_str,
                    waves_per_eu=waves_per_eu,
                    daz=daz,
                    block_m=block_m,
                    block_n=block_n,
                    softmax_scale=softmax_scale,
                    tail_mask=tail_mask,
                    cross_attn=is_cross,
                    lds_vec_width=(
                        16
                        if os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1")
                        == "1"
                        else 8
                    ),
                )
                exe(
                    q_p.reshape(-1),
                    k_p.reshape(-1),
                    v_p.reshape(-1),
                    o_p.reshape(-1),
                    batch,
                    seq_len_pad,
                    seq_len_kv_real,
                    seq_len_kv_pad,
                    stream=launch_stream,
                )

            result = o_p[:, :seq_len_real, :, :] if seq_len_pad != seq_len_real else o_p
            if out is not None and not write_in_place:
                out.copy_(result)
                result = out
            elif out is None:
                result = result.contiguous()

        for tensor in (q, k, v, q_p, k_p, v_p, o_p, result):
            tensor.record_stream(launch_stream)
        if is_fp8:
            q_descale.record_stream(launch_stream)
            k_descale.record_stream(launch_stream)
            v_descale.record_stream(launch_stream)
        (result,) = register_ready((result,), stream=launch_stream)

    return result


def _fp8_gfx950_supported(
    q,
    k,
    v,
    *,
    softmax_scale,
    dropout_p,
    window_size,
    bias,
    alibi_slopes,
    sink,
    return_attn_probs,
    block_table,
    q_descale,
    k_descale,
    v_descale,
    out,
) -> bool:
    """Gate for the gfx950 fp8 kernel.

    It needs e4m3fn Q/K/V with per-tensor descales and a positive, finite
    softmax scale, and writes bf16. Reject anything else so it falls through rather than
    silently dropping the feature.
    """
    if q.dtype is not torch.float8_e4m3fn or q_descale is None:
        return False

    from .kernels.flash_attn_func_fp8_gfx950 import (
        _is_valid_softmax_scale,
        flydsl_flash_attn_fp8_supported,
    )

    if q.dim() not in (3, 4) or k.dim() != q.dim() or v.dim() != q.dim():
        return False
    if not (k.dtype == v.dtype == torch.float8_e4m3fn):
        return False
    if out is not None and (out.dtype != torch.bfloat16 or not out.is_contiguous()):
        return False
    if not (q.is_cuda and k.device == q.device and v.device == q.device):
        return False
    if any(
        s is None
        or not torch.is_tensor(s)
        or s.dtype != torch.float32
        or s.numel() != 1
        or s.device != q.device
        for s in (q_descale, k_descale, v_descale)
    ):
        return False
    qk_hdim = q.shape[-1]
    if not _is_valid_softmax_scale(softmax_scale):
        return False
    nq, nkv = q.shape[-2], k.shape[-2]
    if not flydsl_flash_attn_fp8_supported(
        q.device, nq, nkv, qk_hdim, v.shape[-1], dtype=q.dtype
    ):
        return False
    return (
        k.shape[-1] == qk_hdim
        and nkv > 0
        and nq % nkv == 0
        and dropout_p == 0.0
        and all(w < 0 for w in window_size[:2])
        and (len(window_size) < 3 or window_size[2] == 0)
        and bias is None
        and alibi_slopes is None
        and sink is None
        and block_table is None
        and not return_attn_probs
    )


def flydsl_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    causal: bool = False,
    return_lse: bool = False,
    dropout_p: float = 0.0,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    out=None,
    sink=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
):
    """FlyDSL MHA forward, varlen THD layout.

    Returns the result if FlyDSL can handle this configuration,
    otherwise returns None so the caller falls through to Triton/CK.
    """
    from ...jit.core import is_experimental_enabled
    from ...jit.utils.chip_info import get_gfx

    if (
        q.dtype is torch.float8_e4m3fn
        and q.dim() == 3
        and _fp8_gfx950_supported(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            window_size=window_size,
            bias=bias,
            alibi_slopes=alibi_slopes,
            sink=sink,
            return_attn_probs=return_attn_probs,
            block_table=block_table,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            out=out,
        )
    ):
        from .kernels.flash_attn_func_fp8_gfx950 import flydsl_flash_attn_fp8_func

        return flydsl_flash_attn_fp8_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_k,
            cross_seqlen=max_seqlen_q != max_seqlen_k,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            out=out,
            return_lse=return_lse,
        )

    # FlyDSL m32x8 serves plain MHA plus attention-sink and sliding-window; other
    # features (bias, alibi, dropout, paging, return_attn_probs) fall through to
    # CK/Triton instead of being silently dropped.
    #
    # Routing (D_v=128, bf16, gfx1250): our m32x8 kernel is the DEFAULT for qk_hdim 128 and 192.
    # qk_hdim==256 routes to us only under AITER_ENABLE_EXPERIMENTAL=1 (else CK).
    qk_hdim = q.shape[-1]
    exp = is_experimental_enabled()
    _use_fdsl_wave8_fmha = qk_hdim in (128, 192) or (qk_hdim == 256 and exp)
    # sink must be a valid [nheads_q] fp32 tensor; heads must divide;
    # window_size[2] (sink_size) is unsupported (reject so it is never silently dropped).
    _nq, _nkv = q.shape[-2], k.shape[-2]
    _sink_ok = sink is None or (
        torch.is_tensor(sink) and sink.dtype == torch.float32 and sink.shape == (_nq,)
    )
    supported = (
        get_gfx() == "gfx1250"
        and _use_fdsl_wave8_fmha
        and v.shape[-1] == 128
        and k.shape[-1] == qk_hdim
        and q.dtype in (torch.bfloat16, torch.float16)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and _nkv > 0
        and _nq % _nkv == 0
        and dropout_p == 0.0
        and (len(window_size) < 3 or window_size[2] == 0)
        and block_table is None
        and bias is None
        and alibi_slopes is None
        and _sink_ok
        and not return_attn_probs
    )
    if not supported:
        return None

    # gfx1250 — varlen THD, D_v=128, bf16
    if out is None:
        out = torch.empty_like(q[:, :, : v.shape[-1]])

    # Clean-DSL 8-wave prefill kernel (m32x8), D_qk in {128,192,256}, D_v=128.
    return flash_attn_varlen_m32x8(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        out=out,
        return_lse=return_lse,
        sink=sink,
    )


def flydsl_flash_attn_batch_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    return_lse: bool = False,
    dropout_p: float = 0.0,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    sink=None,
    out=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
):
    """FlyDSL MHA forward, batched BSHD ``[B, S, H, D]`` layout.

    Routes gfx950 fp8 to the dual-wave kernel and gfx1250 bf16/f16 to the
    dedicated BSHD m32x8 kernel (uniform ``seq_len``, no ``cu_seqlens`` —
    CUDA-graph safe). Returns the result if FlyDSL can handle this
    configuration, otherwise returns ``None`` so the caller falls through
    to Triton/CK.
    """
    from ...jit.core import is_experimental_enabled
    from ...jit.utils.chip_info import get_gfx

    if (
        q.dtype is torch.float8_e4m3fn
        and q.dim() == 4
        and _fp8_gfx950_supported(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            window_size=window_size,
            bias=bias,
            alibi_slopes=alibi_slopes,
            sink=sink,
            return_attn_probs=return_attn_probs,
            block_table=None,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            out=out,
        )
    ):
        from .kernels.flash_attn_func_fp8_gfx950 import flydsl_flash_attn_fp8_func

        return flydsl_flash_attn_fp8_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            out=out,
            return_lse=return_lse,
        )

    # BSHD routes to the m32x8 kernel. D_v=128. D_qk 128/192 are
    # the DEFAULT; D_qk==256 needs AITER_ENABLE_EXPERIMENTAL=1 (else CK).
    qk_hdim = q.shape[-1]
    # Head count (BSHD [B,S,H,D]) and sink must satisfy the kernel's asserts, else validate
    # up front so an unsupported request returns None instead of tripping a kernel assert.
    _nq, _nkv = q.shape[-2], k.shape[-2]
    _sink_ok = sink is None or (
        torch.is_tensor(sink) and sink.dtype == torch.float32 and sink.shape == (_nq,)
    )
    supported = (
        get_gfx() == "gfx1250"
        and q.dim() == 4
        and (qk_hdim in (128, 192) or (qk_hdim == 256 and is_experimental_enabled()))
        and v.shape[-1] == 128
        and k.shape[-1] == qk_hdim
        and q.dtype in (torch.bfloat16, torch.float16)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and _nkv > 0
        and _nq % _nkv == 0
        and _sink_ok
        and dropout_p == 0.0
        and bias is None
        and alibi_slopes is None
        and (len(window_size) < 3 or window_size[2] == 0)
        and not return_attn_probs
    )
    # No `not deterministic` gate: it is a backward-only flag (this forward is atomic-free
    # / deterministic), and flash_attn_func defaults it True — gating would reject all.
    if not supported:
        return None

    return flash_attn_batch_m32x8(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        out=out,
        return_lse=return_lse,
        sink=sink,
    )


def flydsl_flash_attn_varlen_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
):
    """FlyDSL MHA backward, varlen THD layout.

    Returns ``(dq, dk, dv, softmax_d)`` to match ``mha_varlen_bwd`` and
    ``fmha_v3_varlen_bwd``.  The gradients are the same tensors that were passed
    in -- the kernel fills them in place -- and ``softmax_d`` is the ``[H, T]``
    fp32 ``rowsum(dO*O)`` those two also return.

    PRECONDITION: the caller has established this configuration is supported --
    causal varlen THD self-attention, d_qk=192 / d_v=128, bf16, no GQA,
    contiguous, ``[H, T]`` fp32 LSE, no dropout / sliding window / alibi / sink /
    padded cu_seqlens, on gfx942.  The authoritative gate is
    ``can_impl_fmha_bwd_flydsl`` inside ``_flash_attn_varlen_backward`` in
    ``aiter/ops/mha.py``; the screened feature arguments are absent from this
    signature precisely because that gate has already established they are unset,
    leaving no configuration for this function to branch on.
    """
    dq, dk, dv, softmax_d = flash_attn_varlen_bwd_d192_gfx942(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        dq=dq,
        dk=dk,
        dv=dv,
    )
    return dq, dk, dv, softmax_d
