# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Flash Attention APIs (gfx1201 / RDNA4).

Wraps the FlyDSL `flash_attn_func_gfx1201` kernel with:
  - Build cache keyed by (num_heads, head_dim, causal, dtype, waves_per_eu, daz).
  - Automatic seq_len padding to the kernel's tile size (multiple of 128).
  - BSHD ([B, S, H, D]) input/output convention to match upstream
    flash-attention layout.
  - Non-causal padding-ratio safety guard: padded K/V tokens contribute to
    the softmax denominator and would scale outputs. Calls with
    ``n_pad / seq_len_pad > 0.005`` (0.5%) and ``causal=False`` are rejected
    with a ``ValueError``. The 0.5% threshold is the bf16 mantissa precision
    floor plus 1 bit of margin; production Wan2.1 (S_real=32760, S_pad=32768,
    ratio=0.024%) clears it by 20x. See option (d) in
    ``2969_padded_softmax_rca.md``.

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

# Maximum tolerated ratio of padded tokens for non-causal attention.
# Padded K/V keys produce QK^T = 0, but exp(0) = 1 leaks into the softmax
# denominator and silently scales the output. 0.5% is the bf16 mantissa
# precision floor (~0.4%) plus 1 bit of margin. Above this the relative
# error grows quickly (50% pad -> 37% rel_err per RCA in
# 2969_padded_softmax_rca.md). Causal mode masks future tokens including
# the padded ones, so it is unaffected.
_MAX_NONCAUSAL_PAD_RATIO = 0.005


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
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Run FlyDSL Flash Attention on RDNA4 (gfx1201).

    Args:
        q, k, v: tensors with shape ``[batch, seq_len, num_heads, head_dim]``
            (BSHD). All three must share dtype, batch, num_heads, head_dim,
            and seq_len. Must reside on a CUDA/HIP device.
        causal: apply causal masking when ``True``.
        waves_per_eu: kernel occupancy hint passed to the FlyDSL builder.
        daz: enable denormals-are-zero on the kernel.
        stream: optional CUDA/HIP stream to launch on. Defaults to the current
            stream for ``q.device``.

    Returns:
        Output tensor with the same shape and dtype as ``q``.

    Raises:
        ValueError: if shapes/dtypes/devices are incompatible, the kernel's
            ``head_dim`` constraints are not met, or the non-causal padding
            ratio ``n_pad / seq_len_pad`` exceeds 0.5% (see module docstring
            for rationale).
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
    except Exception:
        arch = ""
    arch_base = arch.lower().split(":")[0] if arch else ""
    if not arch_base.startswith("gfx1201"):
        raise ValueError(f"flydsl_flash_attn_func requires gfx1201, got {arch!r}")
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

    # Pad seq_len up to the kernel's tile size. Tight padding (<= 0.5% of
    # S_pad) is empirically below the bf16 noise floor on production shapes
    # (Wan2.1 cos_sim >= 0.999992). Higher ratios are rejected upstream:
    # padded K/V tokens produce QK^T = 0 but exp(0) = 1 still contributes
    # to the softmax denominator and would scale the output. Padded queries
    # produce garbage rows that we slice off before returning.
    seq_len_pad = (
        (seq_len_real + _KERNEL_BLOCK_M - 1) // _KERNEL_BLOCK_M
    ) * _KERNEL_BLOCK_M
    n_pad = seq_len_pad - seq_len_real
    if not causal and n_pad > 0 and n_pad / seq_len_pad > _MAX_NONCAUSAL_PAD_RATIO:
        raise ValueError(
            "flydsl_flash_attn_func: non-causal path with padding ratio "
            f"{n_pad}/{seq_len_pad}={n_pad / seq_len_pad:.4f} exceeds 0.5% "
            "safety threshold; padded K/V tokens contribute to softmax "
            "denominator and would scale outputs. Either set causal=True, "
            "pad seq_len to a multiple of 128 before calling, or use a "
            "self-attn kernel with explicit attention masking."
        )
    if seq_len_pad != seq_len_real:
        pad = n_pad
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

    # Wrap kernel build + launch in q.device context so multi-GPU callers
    # whose current device differs from q.device get the kernel compiled
    # and launched on the right device/stream.
    with torch.cuda.device(q.device.index):
        launch_stream = (
            torch.cuda.current_stream(q.device) if stream is None else stream
        )
        if launch_stream.device != q.device:
            raise ValueError(
                f"`stream` must be on {q.device}, got {launch_stream.device}"
            )
        exe = _get_kernel(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=causal,
            dtype_str=dtype_str,
            waves_per_eu=waves_per_eu,
            daz=daz,
        )
        exe(
            q_p.reshape(-1),
            k_p.reshape(-1),
            v_p.reshape(-1),
            o_p.reshape(-1),
            batch,
            seq_len_pad,
            stream=launch_stream,
        )

    if seq_len_pad != seq_len_real:
        return o_p[:, :seq_len_real, :, :].contiguous()
    return o_p
