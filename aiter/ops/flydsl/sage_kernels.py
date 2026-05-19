# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Sage Attention APIs (CDNA / gfx942 / gfx950).

Wraps the FlyDSL ``sage_attn_cdna`` kernel with:
  - Build cache keyed by (num_q_heads, num_kv_heads, head_dim, causal, waves_per_eu).
  - Automatic Q/K Int8 + V FP8 quantization via ``sage_quant``.
  - BSHD ([B, S, H, D]) input/output convention.
  - Seq_len padding to the kernel's tile size (multiple of BLOCK_M).

The kernel implements self-attention and GQA. Cross-attention (different Q
and KV seq_len) is also supported since Q and KV shapes may differ.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Optional

import torch

from aiter.utility.dtypes import fp8 as _fp8_dtype
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    sage_quant as _triton_sage_quant,
)
from .sage_quant import flydsl_sage_quant as _flydsl_sage_quant

from .kernels.sage_attn_cdna import build_sage_attn_cdna_module


def _sage_quant(*args, **kwargs):
    """Dispatch to FlyDSL or Triton sage_quant per FLYDSL_SAGE_QUANT_BACKEND env var.

    Default: ``"flydsl"`` — single fused Q INT8 + K INT8 + V FP8 launch
    (~1.18-1.20x Triton on small/medium shapes, parity on long-S).
    Set ``FLYDSL_SAGE_QUANT_BACKEND=triton`` to fall back to the per-op
    Triton wrappers.
    """
    backend = os.environ.get("FLYDSL_SAGE_QUANT_BACKEND", "flydsl").lower()
    if backend == "flydsl":
        return _flydsl_sage_quant(*args, **kwargs)
    return _triton_sage_quant(*args, **kwargs)


# keep the legacy name `sage_quant` working below
sage_quant = _sage_quant

__all__ = [
    "flydsl_sage_attn_func",
]

# Tile size baked into the CDNA kernel. Seq_len_q must be a multiple of this.
_KERNEL_BLOCK_M = 256


def _check_cdna_arch(device: torch.device) -> str:
    """Return arch string; raise if not gfx942/gfx950."""
    try:
        arch = torch.cuda.get_device_properties(device.index).gcnArchName
    except Exception:
        arch = ""
    arch_base = arch.lower().split(":")[0] if arch else ""
    if not (arch_base.startswith("gfx942") or arch_base.startswith("gfx950")):
        raise ValueError(
            f"flydsl_sage_attn_func requires gfx942 or gfx950, got {arch!r}"
        )
    return arch_base


@lru_cache(maxsize=64)
def _get_kernel(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    waves_per_eu: int,
    block_m: int,
    block_n: int,
):
    return build_sage_attn_cdna_module(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=causal,
        waves_per_eu=waves_per_eu,
        block_m=block_m,
        block_n=block_n,
    )


def flydsl_sage_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    layout: str = "bshd",
    waves_per_eu: int = 2,
    block_m: Optional[int] = None,
    block_n: int = 128,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Run FlyDSL Sage Attention on CDNA (gfx942/MI300X, gfx950/MI350).

    Internally quantizes Q/K to Int8 and V to FP8, then runs the FlyDSL
    flash-attention kernel with Int8 MFMA (GEMM1) and BF16 MFMA (GEMM2).

    Args:
        q, k, v: tensors with shape ``[batch, seq_len, num_heads, head_dim]``
            (BSHD layout, default) or ``[batch, num_heads, seq_len, head_dim]``
            (BHSD layout, set ``layout="bhsd"``).
            Supported dtypes: bf16, fp16, fp32.
            Must reside on a CUDA/HIP device.
        softmax_scale: scaling factor for QK^T (default: 1/sqrt(head_dim)).
        causal: apply causal masking when ``True``.
        layout: ``"bshd"`` (default) or ``"bhsd"``.
        waves_per_eu: kernel occupancy hint.
        block_m: Q tile size. If ``None`` (default), auto-selected per shape:
            BLOCK_M=128 when grid_at_BM256 < CU count (small-S/low-occupancy
            shapes win ~17% from the doubled grid), else BLOCK_M=256.
        block_n: KV tile size (default 128).
        stream: optional CUDA/HIP stream. Defaults to current stream for q.device.

    Returns:
        Output tensor with the same BSHD shape as ``q``, dtype BF16.

    Raises:
        ValueError: if shapes/dtypes/devices are incompatible or the GPU is
            not gfx942/gfx950.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_sage_attn_func requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(
            f"q/k/v must be on the same device: q={q.device} k={k.device} v={v.device}"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"q/k/v dtype must match: {q.dtype}/{k.dtype}/{v.dtype}")
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if q.dtype not in supported_dtypes:
        raise ValueError(
            f"flydsl_sage_attn_func supports fp16/bf16/fp32 inputs, got {q.dtype}"
        )
    if q.dim() != 4:
        raise ValueError(f"expected 4D tensor, got rank {q.dim()} ({tuple(q.shape)})")
    if layout not in ("bshd", "bhsd"):
        raise ValueError(f"layout must be 'bshd' or 'bhsd', got {layout!r}")

    _check_cdna_arch(q.device)

    # Dimension mapping: bshd → [0,1,2,3], bhsd → [0,2,1,3]
    if layout == "bshd":
        batch, seq_q, num_q_heads, head_dim = q.shape
        _, seq_k, num_kv_heads, _ = k.shape
    else:
        batch, num_q_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    if head_dim < 64 or head_dim % 32 != 0:
        raise ValueError(
            f"kernel requires head_dim >= 64 and head_dim % 32 == 0, got {head_dim}"
        )

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # Auto-pick BLOCK_M for occupancy. BM=256 only saturates the GPU once
    # batch * num_q_heads * ceil(seq_q/256) >= CU count. For small-S/H1
    # shapes the BM=256 grid is well below CU count and BM=128 (which doubles
    # the grid) wins +17%; for shapes that already saturate, BM=256 wins
    # (larger MFMA chains, single coop-load batch).
    if block_m is None:
        try:
            cu_count = torch.cuda.get_device_properties(
                q.device.index
            ).multi_processor_count
        except Exception:
            cu_count = 256
        grid_at_bm256 = batch * num_q_heads * ((seq_q + 255) // 256)
        block_m = 128 if grid_at_bm256 < cu_count else 256

    fp8_dtype = _fp8_dtype
    fp8_max = torch.finfo(fp8_dtype).max

    # Quantize Q/K to Int8, V to FP8
    # sage_quant returns: q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale
    q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale = sage_quant(
        q,
        k,
        v,
        FP8_TYPE=fp8_dtype,
        FP8_MAX=fp8_max,
        BLKQ=block_m,
        BLKK=block_n,
        sm_scale=softmax_scale,
        layout=layout,
    )

    # Pad seq_q to multiple of block_m so the kernel tile loop is clean
    seq_q_pad = ((seq_q + block_m - 1) // block_m) * block_m
    n_pad_q = seq_q_pad - seq_q

    # Contiguous BSHD views for kernel (convert BHSD if needed)
    if layout == "bhsd":
        q_int8 = q_int8.permute(0, 2, 1, 3).contiguous()
        k_int8 = k_int8.permute(0, 2, 1, 3).contiguous()
        v_fp8 = v_fp8.permute(0, 2, 1, 3).contiguous()
        # q_scale / k_scale are [batch, heads, blocks] — no permute needed
        # v_scale is [batch, heads, ...] — no permute needed
    else:
        q_int8 = q_int8.contiguous()
        k_int8 = k_int8.contiguous()
        v_fp8 = v_fp8.contiguous()

    if n_pad_q > 0:
        # Pad Q seq dim (dim=1 in BSHD)
        q_int8 = torch.nn.functional.pad(q_int8, (0, 0, 0, 0, 0, n_pad_q))

    # Allocate BF16 output in BSHD (padded)
    o_shape = (batch, seq_q_pad, num_q_heads, head_dim)
    o = torch.empty(o_shape, dtype=torch.bfloat16, device=q.device)

    num_q_blocks = q_scale.shape[2]  # [batch, num_q_heads, num_q_blocks]

    with torch.cuda.device(q.device.index):
        launch_stream = (
            torch.cuda.current_stream(q.device) if stream is None else stream
        )
        if launch_stream.device != q.device:
            raise ValueError(
                f"`stream` must be on {q.device}, got {launch_stream.device}"
            )

        exe = _get_kernel(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            causal=causal,
            waves_per_eu=waves_per_eu,
            block_m=block_m,
            block_n=block_n,
        )
        exe(
            q_int8.reshape(-1),
            k_int8.reshape(-1),
            v_fp8.reshape(-1),
            o.reshape(-1),
            q_scale,
            k_scale,
            v_scale,
            batch,
            seq_q_pad,
            seq_k,
            num_q_blocks,
            stream=launch_stream,
        )

    if n_pad_q > 0:
        o = o[:, :seq_q, :, :].contiguous()

    # Convert back to BHSD if caller requested it
    if layout == "bhsd":
        o = o.permute(0, 2, 1, 3).contiguous()

    return o
