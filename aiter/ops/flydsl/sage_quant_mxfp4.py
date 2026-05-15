# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MXFP4 sage quantization wrapper.

Drop-in for ``aiter.ops.triton.quant.sage_attention_quant_wrappers.sage_quant_mxfp4``
when ``q_smoothing=False`` and ``hadamard_rotation=True`` (the bench path).

Replaces ~9 Triton/torch kernel launches (rotate-Q, rotate-K, K-mean
torch reduce, K_rot - K_mean, V abs.amax torch op, V FP8 quant, Q
downcast_to_mxfp, K downcast_to_mxfp) with **one** FlyDSL kernel launch
plus 2 small torch ops (K-mean + V abs.amax). The fused kernel handles
Q rotate+MXFP4, K rotate+MXFP4, and V FP8 quant in a single launch.

Output layout matches Triton exactly:
    q_fp4    : uint8 [B, S_q, Hq, D//2]
    q_d      : uint8 [B, S_q, Hq, D//32]   e8m0
    k_fp4    : uint8 [B, S_k, Hk, D//2]
    k_d      : uint8 [B, S_k, Hk, D//32]
    v_fp8    : fp8e4m3 [B, S_k, Hk, D]
    v_scale  : f32   [B, Hk, D]
    delta_s  : None  (q_smoothing=False)
"""

from __future__ import annotations

import math
import struct
from functools import lru_cache
from typing import Optional, Tuple

import torch

from .kernels.sage_quant_mxfp4_cdna import build_sage_quant_mxfp4_module


__all__ = ["flydsl_sage_quant_mxfp4"]


@lru_cache(maxsize=64)
def _get_kernel(head_dim: int, blk_q: int, blk_k: int,
                num_q_heads: int, num_kv_heads: int):
    return build_sage_quant_mxfp4_module(
        head_dim=head_dim,
        blk_q=blk_q,
        blk_k=blk_k,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )


def _f32_bits(x: float) -> int:
    """Pack an f32 into a signed int32 holding its bit pattern."""
    b = struct.pack("<f", float(x))
    u32 = struct.unpack("<I", b)[0]
    # Wrap to signed int32
    if u32 >= 0x80000000:
        return u32 - 0x100000000
    return u32


def flydsl_sage_quant_mxfp4(
    q: torch.Tensor,        # bf16 [B, S_q, Hq, D] (BSHD)
    k: torch.Tensor,        # bf16 [B, S_k, Hk, D]
    v: torch.Tensor,        # bf16 [B, S_k, Hk, D]
    fp8_type: torch.dtype,
    fp8_max: float,
    BLKQ: int = 256,
    BLKK: int = 64,
    sm_scale: Optional[float] = None,
    layout: str = "bshd",
    R: Optional[torch.Tensor] = None,
    BLOCK_R: int = 128,
    skip_k_mean: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, None]:
    """Returns (q_fp4, q_d, k_fp4, k_d, v_fp8, v_scale, None).

    BSHD layout only. q_smoothing must be False.
    R is ignored when present — the kernel uses a fixed normalized
    Walsh-Hadamard of size BLOCK_R=128 (matches Triton's default for
    the bench path).
    """
    assert layout in ("bshd", "bhsd"), f"layout must be bshd|bhsd, got {layout}"
    # bhsd: permute to bshd, run, permute outputs back. The kernel only
    # supports bshd-contiguous inputs.
    bhsd_in = layout == "bhsd"
    if bhsd_in:
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        layout = "bshd"
    assert q.dtype == torch.bfloat16, f"q dtype must be bf16, got {q.dtype}"
    assert k.dtype == torch.bfloat16, f"k dtype must be bf16, got {k.dtype}"
    assert v.dtype == torch.bfloat16, f"v dtype must be bf16, got {v.dtype}"
    if q.shape[-1] != 128:
        raise ValueError(
            f"flydsl MXFP4 quant only supports head_dim=128, got {q.shape[-1]}"
        )
    if BLOCK_R != 128:
        raise ValueError(f"flydsl MXFP4 quant only supports BLOCK_R=128, got {BLOCK_R}")

    B, S_q, Hq, D = q.shape
    _, S_k, Hk, _ = k.shape
    assert v.shape == k.shape

    if sm_scale is None:
        sm_scale = D ** -0.5
    sm_scale_log2e = sm_scale * 1.4426950408889634

    NUM_BLKS_Q = (S_q + BLKQ - 1) // BLKQ
    NUM_BLKS_K = (S_k + BLKK - 1) // BLKK

    q_task_count = B * Hq * NUM_BLKS_Q
    k_task_count = B * Hk * NUM_BLKS_K
    v_task_count = B * Hk * NUM_BLKS_K
    grid_size = q_task_count + k_task_count + v_task_count

    # Outputs (torch.empty: every element is written by the kernel)
    q_fp4 = torch.empty((B, S_q, Hq, D // 2), dtype=torch.uint8, device=q.device)
    q_d   = torch.empty((B, S_q, Hq, D // 32), dtype=torch.uint8, device=q.device)
    k_fp4 = torch.empty((B, S_k, Hk, D // 2), dtype=torch.uint8, device=k.device)
    k_d   = torch.empty((B, S_k, Hk, D // 32), dtype=torch.uint8, device=k.device)
    v_fp8 = torch.empty((B, S_k, Hk, D), dtype=fp8_type, device=v.device)

    # K-mean (matches Triton rotation_smooth_qk:419, applied to RAW K
    # not rotated K — sageattention smoothing). For q_smooth=False we may
    # skip this; controlled by skip_k_mean.
    if skip_k_mean:
        k_input = k
    else:
        k_input = k - k.mean(dim=1, keepdim=True)

    # Per-channel V scale: max along seq dim / FP8_MAX.
    v_scale = (v.abs().amax(dim=1).to(torch.float32) / fp8_max).contiguous()

    # Materialize input contiguity (kernel assumes BSHD-contiguous layout).
    q_c = q.contiguous()
    k_c = k_input.contiguous()
    v_c = v.contiguous()

    launcher = _get_kernel(
        head_dim=D, blk_q=BLKQ, blk_k=BLKK,
        num_q_heads=Hq, num_kv_heads=Hk,
    )

    sm_bits  = _f32_bits(sm_scale_log2e)
    norm_bits = _f32_bits(1.0 / math.sqrt(D))

    stream = torch.cuda.current_stream(q.device)
    launcher(
        q_c.reshape(-1),
        q_fp4.reshape(-1),
        q_d.reshape(-1),
        k_c.reshape(-1),
        k_fp4.reshape(-1),
        k_d.reshape(-1),
        v_c.reshape(-1),
        v_fp8.reshape(-1),
        v_scale.reshape(-1),
        S_q, S_k,
        NUM_BLKS_Q, NUM_BLKS_K,
        q_task_count, k_task_count,
        sm_bits, norm_bits,
        grid_size,
        stream=stream,
    )

    if bhsd_in:
        q_fp4 = q_fp4.permute(0, 2, 1, 3).contiguous()
        q_d   = q_d.permute(0, 2, 1, 3).contiguous()
        k_fp4 = k_fp4.permute(0, 2, 1, 3).contiguous()
        k_d   = k_d.permute(0, 2, 1, 3).contiguous()
        v_fp8 = v_fp8.permute(0, 2, 1, 3).contiguous()
        # v_scale is [B, Hk, D] regardless of layout (per-channel)

    return q_fp4, q_d, k_fp4, k_d, v_fp8, v_scale, None
