# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.

"""Native gfx1201 Q/K/V FP8 quantization for FlyDSL flash attention.

This producer's supported head dimensions are independent of the attention
consumer's per-shape LDS limit, which the consumer validates separately.
"""

import warnings
from functools import lru_cache

import torch
import torch.nn.functional as F

from .kernels.fp8_quant_gfx1201 import (
    flydsl_fp8_pertensor_quant,
    flydsl_fp8_qkv_d64_quant,
)

_INT32_MAX = (1 << 31) - 1


@lru_cache(maxsize=16)
def _gpu_arch_cached(index: int | None) -> str:
    try:
        return torch.cuda.get_device_properties(index).gcnArchName.split(":")[0]
    except Exception:  # noqa: BLE001
        return ""


def _gpu_arch(device: torch.device) -> str:
    return _gpu_arch_cached(device.index)


def _validate_quant_dword_offsets(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    head_dim: int,
    native_head_dim: int,
) -> None:
    """Reject native quantizer inputs whose BF16 dword offsets overflow Int32."""
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        rows = tensor.numel() // head_dim
        dwords = rows * native_head_dim // 2
        if dwords - 1 > _INT32_MAX:
            raise ValueError(
                f"flydsl_fp8_quant {name} requires dword offset {dwords - 1}, "
                f"exceeding the native quantizer Int32 limit ({_INT32_MAX})"
            )


def flydsl_fp8_quant(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    rotation: bool = True,
):
    """Quantize BF16 BSHD Q/K/V for native gfx1201 FP8 flash attention.

    This is a native FlyDSL-only path. Inputs must be non-empty BF16 tensors on
    one gfx1201 device with a common head dimension from 64 through 1024,
    divisible by 32. Rotation uses an in-register FWHT and is available only for
    power-of-two head dimensions; non-power-of-two dimensions are padded to the
    next supported power of two and require ``rotation=False``.

    The returned tensors and scales are enqueued on the current stream of
    ``q.device``. Consumers on another stream must establish the usual PyTorch
    event dependency themselves.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q/k/v must be CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError("q/k/v must reside on the same device")
    arch = _gpu_arch(q.device).lower()
    if arch != "gfx1201":
        raise ValueError(f"flydsl_fp8_quant requires gfx1201, got {arch!r}")
    if not (q.dtype == k.dtype == v.dtype == torch.bfloat16):
        raise ValueError("q/k/v dtype must match and be bfloat16")
    if any(x.dim() == 0 for x in (q, k, v)) or not (
        q.shape[-1] == k.shape[-1] == v.shape[-1]
    ):
        raise ValueError("q/k/v must share head_dim")
    if any(x.numel() == 0 for x in (q, k, v)):
        raise ValueError("q/k/v must be non-empty")

    head_dim = q.shape[-1]
    if head_dim < 64 or head_dim % 32:
        raise ValueError(
            "native FP8 quant requires head_dim >= 64 and divisible by 32, "
            f"got {head_dim}"
        )
    if rotation and head_dim & (head_dim - 1):
        raise ValueError(
            f"FWHT rotation requires a power-of-two head_dim, got {head_dim}; "
            "pass rotation=False to quantize without rotation"
        )
    padded_dim = 1 << (head_dim - 1).bit_length()
    if head_dim not in (64, 128, 256, 512, 1024) and padded_dim > 1024:
        raise ValueError(
            f"native FP8 quant has no packed specialization for head_dim={head_dim}"
        )
    _validate_quant_dword_offsets(
        q, k, v, head_dim=head_dim, native_head_dim=padded_dim
    )
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        warnings.warn(
            "flydsl_fp8_quant materializes non-contiguous Q/K/V inputs; "
            "provide contiguous tensors to avoid the copy",
            stacklevel=2,
        )
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    # Same-shape D64 BSHD inputs share one 96-thread workgroup per row: wave 0
    # quantizes Q, wave 1 K, and wave 2 V. Cross-attention and all other shapes
    # use the per-tensor implementation.
    if head_dim == 64 and q.shape == k.shape == v.shape:
        return flydsl_fp8_qkv_d64_quant(q, k, v, rotate=rotation)
    if head_dim not in (64, 128, 256, 512, 1024):
        warnings.warn(
            f"flydsl_fp8_quant pads head_dim={head_dim} to {padded_dim} for "
            "the native packed quantizer; use a supported power-of-two head "
            "dimension to avoid the materialization",
            stacklevel=2,
        )

    def quant_one(x: torch.Tensor, *, do_rotate: bool):
        if head_dim in (64, 128, 256, 512, 1024):
            return flydsl_fp8_pertensor_quant(x, rotate=do_rotate)

        # Non-power-of-two dimensions use the unrotated packed specializations.
        padded = F.pad(x, (0, padded_dim - head_dim))
        quantized, scale = flydsl_fp8_pertensor_quant(padded, rotate=False)
        return quantized[..., :head_dim].contiguous(), scale

    q8, sq = quant_one(q, do_rotate=rotation)
    k8, sk = quant_one(k, do_rotate=rotation)
    v8, sv = quant_one(v, do_rotate=False)
    return q8, k8, v8, sq, sk, sv
