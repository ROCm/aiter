# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL replacement for Triton's sage_quant (Q/K Int8 + V FP8).

Single-launch fused kernel for the Q INT8 + K INT8 + V FP8 quant phase
of Sage Attention V1. Replaces Triton's ``sage_quant_kernel`` (3 kernels'
worth of work fused into one block-indexed dispatch).

Same return signature as
``aiter.ops.triton.quant.sage_attention_quant_wrappers.sage_quant``:
    q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale
"""

from __future__ import annotations

from functools import lru_cache

import torch

from .kernels.sage_quant_cdna import (
    build_sage_quant_v_module,
    build_sage_quant_fused_module,
)


__all__ = ["flydsl_sage_quant"]


@lru_cache(maxsize=64)
def _get_v_kernel(head_dim: int, blk_k: int, num_kv_heads: int):
    return build_sage_quant_v_module(
        head_dim=head_dim, blk_k=blk_k, num_kv_heads=num_kv_heads
    )


@lru_cache(maxsize=64)
def _get_fused_kernel(
    head_dim: int,
    blk_q: int,
    blk_k: int,
    num_q_heads: int,
    num_kv_heads: int,
):
    return build_sage_quant_fused_module(
        head_dim=head_dim,
        blk_q=blk_q,
        blk_k=blk_k,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )


def flydsl_sage_quant(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    FP8_TYPE,
    FP8_MAX,
    BLKQ: int = 128,
    BLKK: int = 64,
    sm_scale=None,
    layout: str = "bshd",
    smooth_k: bool = True,
):
    """FlyDSL fused sage_quant. BSHD layout uses the single-launch kernel;
    BHSD permutes to BSHD before/after for the FlyDSL path.
    """
    q_int8 = torch.empty_like(q, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty_like(k, dtype=torch.int8, device=k.device)
    v_fp8 = torch.empty_like(v, dtype=FP8_TYPE, device=v.device)

    if layout == "bhsd":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        v_seq_dim = 2
    elif layout == "bshd":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        v_seq_dim = 1
    else:
        raise ValueError(f"Unknown tensor layout: {layout}")

    Q_NUM_BLKS = (qo_len + BLKQ - 1) // BLKQ
    K_NUM_BLKS = (kv_len + BLKK - 1) // BLKK

    if smooth_k:
        k = k - k.mean(dim=v_seq_dim, keepdim=True)

    q_scale = torch.empty((b, h_qo, Q_NUM_BLKS), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, K_NUM_BLKS), device=q.device, dtype=torch.float32)

    v_scale = (
        v.abs().amax(dim=v_seq_dim).to(torch.float32) / FP8_MAX
    )

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    # The fused kernel only handles BSHD natively. For BHSD, permute to
    # BSHD-contiguous, run, permute back.
    if layout == "bshd":
        q_for_kernel = q.contiguous() if not q.is_contiguous() else q
        k_for_kernel = k.contiguous() if not k.is_contiguous() else k
        v_for_kernel = v.contiguous() if not v.is_contiguous() else v
        q_int8_for_kernel = q_int8
        k_int8_for_kernel = k_int8
        v_fp8_for_kernel = v_fp8
        v_scale_for_kernel = v_scale.contiguous() if not v_scale.is_contiguous() else v_scale
    else:
        q_for_kernel = q.permute(0, 2, 1, 3).contiguous()
        k_for_kernel = k.permute(0, 2, 1, 3).contiguous()
        v_for_kernel = v.permute(0, 2, 1, 3).contiguous()
        q_int8_for_kernel = torch.empty_like(q_for_kernel, dtype=torch.int8)
        k_int8_for_kernel = torch.empty_like(k_for_kernel, dtype=torch.int8)
        v_fp8_for_kernel = torch.empty_like(v_for_kernel, dtype=FP8_TYPE)
        v_scale_for_kernel = v_scale.contiguous()

    q_task_count = b * h_qo * Q_NUM_BLKS
    k_task_count = b * h_kv * K_NUM_BLKS
    v_task_count = k_task_count
    grid_size = q_task_count + k_task_count + v_task_count

    sm_scale_log2e_f = float(sm_scale) * 1.4426950408889634
    # Bit-pattern reinterpret to int32 — Float32 lacks a CallState fast-path
    # slot spec, which forced the slow DLPack dispatch (~150us per call).
    # The kernel reinterprets back to f32 via arith.bitcast.
    import struct
    sm_scale_log2e_bits = struct.unpack("<i", struct.pack("<f", sm_scale_log2e_f))[0]

    fused = _get_fused_kernel(
        head_dim=int(head_dim),
        blk_q=int(BLKQ),
        blk_k=int(BLKK),
        num_q_heads=int(h_qo),
        num_kv_heads=int(h_kv),
    )

    stream = torch.cuda.current_stream(q.device)
    fused(
        q_for_kernel,
        q_int8_for_kernel,
        q_scale,
        k_for_kernel,
        k_int8_for_kernel,
        k_scale,
        v_for_kernel,
        v_fp8_for_kernel,
        v_scale_for_kernel,
        qo_len,
        kv_len,
        Q_NUM_BLKS,
        K_NUM_BLKS,
        q_task_count,
        k_task_count,
        sm_scale_log2e_bits,
        grid_size,
        stream=stream,
    )

    if layout == "bhsd":
        q_int8.copy_(q_int8_for_kernel.permute(0, 2, 1, 3))
        k_int8.copy_(k_int8_for_kernel.permute(0, 2, 1, 3))
        v_fp8.copy_(v_fp8_for_kernel.permute(0, 2, 1, 3))

    return q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale
