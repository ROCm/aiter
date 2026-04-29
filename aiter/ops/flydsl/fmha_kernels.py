# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Flash Attention APIs (gfx1201 / RDNA4).

Wraps the FlyDSL `flash_attn_func_gfx1201` kernel with:
  - Build cache keyed by (num_heads, head_dim, causal, dtype).
  - Automatic seq_len padding to the kernel's tile size (multiple of 128).
  - BSHD ([B, S, H, D]) input/output convention to match upstream
    flash-attention layout.

The kernel implements self-attention only (Lq == Lk). Cross-attention
(Lq != Lk) is rejected; callers should fall back to PyTorch SDPA.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

from .kernels.flash_attn_func_gfx1201 import build_flash_attn_func_module

__all__ = [
    "flydsl_flash_attn_func",
]


# Tile size baked into the gfx1201 kernel. Seq_len must be a multiple of this.
# Picked to match BLOCK_M=128 in the kernel; padding is invisible to callers.
_KERNEL_BLOCK_M = 128


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "f16"
    raise ValueError(f"flydsl_flash_attn_func only supports bf16/f16, got {dtype!r}")


@lru_cache(maxsize=32)
def _get_kernel(
    num_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    waves_per_eu: int,
    daz: bool,
):
    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        waves_per_eu=waves_per_eu,
        daz=daz,
    )


def flydsl_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    waves_per_eu: int = 2,
    daz: bool = True,
) -> torch.Tensor:
    """Run FlyDSL Flash Attention on RDNA4 (gfx1201).

    Args:
        q, k, v: tensors with shape ``[batch, seq_len, num_heads, head_dim]``
            (BSHD). All three must share dtype, batch, num_heads, head_dim,
            and seq_len. Must reside on a CUDA/HIP device.
        causal: apply causal masking when ``True``.
        waves_per_eu: kernel occupancy hint passed to the FlyDSL builder.
        daz: enable denormals-are-zero on the kernel.

    Returns:
        Output tensor with the same shape and dtype as ``q``.

    Raises:
        ValueError: if shapes/dtypes/devices are incompatible or the kernel's
            ``head_dim`` constraints are not met.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_func requires CUDA/HIP tensors")
    if not (q.shape == k.shape == v.shape):
        raise ValueError(
            "flydsl_flash_attn_func is self-attention; q/k/v must share "
            f"shape, got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"q/k/v dtype must match: {q.dtype}/{k.dtype}/{v.dtype}")
    if q.dim() != 4:
        raise ValueError(
            f"expected 4D BSHD tensor, got rank {q.dim()} ({tuple(q.shape)})"
        )

    batch, seq_len_real, num_heads, head_dim = q.shape
    if head_dim < 64 or head_dim % 32 != 0:
        raise ValueError(
            f"kernel requires head_dim >= 64 and head_dim % 32 == 0, got {head_dim}"
        )

    dtype_str = _torch_dtype_to_str(q.dtype)
    exe = _get_kernel(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        waves_per_eu=waves_per_eu,
        daz=daz,
    )

    # Pad seq_len up to the kernel's tile size. Zero-pad on K/V is safe for
    # non-causal: padded keys produce QK^T = 0 → uniform softmax mass that
    # gets normalized away as long as real keys dominate. Padded queries
    # produce garbage rows that we slice off before returning.
    seq_len_pad = (
        (seq_len_real + _KERNEL_BLOCK_M - 1) // _KERNEL_BLOCK_M
    ) * _KERNEL_BLOCK_M
    if seq_len_pad != seq_len_real:
        pad = seq_len_pad - seq_len_real
        # F.pad pads from the last dim; for BSHD (last=head_dim) the seq dim
        # is dim 1, so we pad (D_left, D_right, H_left, H_right, S_left, S_right).
        q_p = F.pad(q.contiguous(), (0, 0, 0, 0, 0, pad))
        k_p = F.pad(k.contiguous(), (0, 0, 0, 0, 0, pad))
        v_p = F.pad(v.contiguous(), (0, 0, 0, 0, 0, pad))
    else:
        q_p = q.contiguous()
        k_p = k.contiguous()
        v_p = v.contiguous()

    o_p = torch.empty_like(q_p)

    exe(
        q_p.reshape(-1),
        k_p.reshape(-1),
        v_p.reshape(-1),
        o_p.reshape(-1),
        batch,
        seq_len_pad,
    )

    if seq_len_pad != seq_len_real:
        return o_p[:, :seq_len_real, :, :].contiguous()
    return o_p
