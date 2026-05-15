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

from .kernels.sage_quant_mxfp4_cdna import (
    build_sage_quant_mxfp4_module,
    build_compute_delta_s_module,
)
from .kernels.sage_quant_cdna import build_sage_preprocess_module


__all__ = ["flydsl_sage_quant_mxfp4"]


@lru_cache(maxsize=64)
def _get_kernel(head_dim: int, blk_q: int, blk_k: int,
                num_q_heads: int, num_kv_heads: int,
                subtract_k_mean: bool, q_smoothing: bool):
    return build_sage_quant_mxfp4_module(
        head_dim=head_dim,
        blk_q=blk_q,
        blk_k=blk_k,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        subtract_k_mean=subtract_k_mean,
        q_smoothing=q_smoothing,
    )


@lru_cache(maxsize=64)
def _get_delta_s_kernel(head_dim: int, num_q_heads: int, num_kv_heads: int,
                         block_n: int, subtract_k_mean_in_kernel: bool):
    return build_compute_delta_s_module(
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        block_n=block_n,
        subtract_k_mean_in_kernel=subtract_k_mean_in_kernel,
    )


@lru_cache(maxsize=64)
def _get_preprocess(head_dim: int, num_kv_heads: int):
    return build_sage_preprocess_module(
        head_dim=head_dim,
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
    q_smoothing: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Returns (q_fp4, q_d, k_fp4, k_d, v_fp8, v_scale, delta_s).

    ``delta_s`` is None when ``q_smoothing=False``; otherwise an f32 tensor
    of shape ``[B, Hq, Q_NUM_BLKS, S_k]`` to be passed as ``bias`` to the
    attention kernel.

    BSHD layout (BHSD permuted in/out). R is ignored — the kernel uses a
    fixed normalized Walsh-Hadamard of size BLOCK_R=128.
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
    q_fp4   = torch.empty((B, S_q, Hq, D // 2),  dtype=torch.uint8, device=q.device)
    q_d     = torch.empty((B, S_q, Hq, D // 32), dtype=torch.uint8, device=q.device)
    k_fp4   = torch.empty((B, S_k, Hk, D // 2),  dtype=torch.uint8, device=k.device)
    k_d     = torch.empty((B, S_k, Hk, D // 32), dtype=torch.uint8, device=k.device)
    v_fp8   = torch.empty((B, S_k, Hk, D),        dtype=fp8_type,    device=v.device)
    k_mean  = torch.empty((B, Hk, D),             dtype=torch.float32, device=k.device)
    v_scale = torch.empty((B, Hk, D),             dtype=torch.float32, device=v.device)

    # Q_mean is per-Q-block; only allocated/written when q_smoothing=True.
    if q_smoothing:
        q_mean = torch.empty(
            (B, Hq, NUM_BLKS_Q, D), dtype=torch.float32, device=q.device,
        )
    else:
        q_mean = torch.empty(1, dtype=torch.float32, device=q.device)  # dummy

    # Materialize input contiguity (kernel assumes BSHD-contiguous layout).
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()

    stream = torch.cuda.current_stream(q.device)

    # In-kernel K-mean subtract is a big win on short S (saves 2 torch launches)
    # but a regression on long S (extra register pressure on the K branch hurts
    # the heavy bandwidth-bound work). Threshold empirically picked from bench.
    use_in_kernel_k_mean = (not skip_k_mean) and (S_k <= 8192)

    # ---- Stage 1: K_mean + V_scale.
    # When in-kernel subtract is on, use the FlyDSL preprocessor (1 launch) to
    # produce K_mean and V_scale together. Otherwise compute via torch (2-3
    # launches but faster on long S due to torch.mean's optimization).
    if skip_k_mean:
        k_mean.zero_()
        v_scale_torch = (v.abs().amax(dim=1).to(torch.float32) / fp8_max).contiguous()
        v_scale.copy_(v_scale_torch)
        k_subtract_torch = None
    elif use_in_kernel_k_mean:
        preprocess = _get_preprocess(head_dim=D, num_kv_heads=Hk)
        preprocess(
            k_c.reshape(-1),
            v_c.reshape(-1),
            k_mean.reshape(-1),
            v_scale.reshape(-1),
            B,
            S_k,
            _f32_bits(1.0 / fp8_max),
            _f32_bits(1.0 / S_k),
            stream=stream,
        )
        k_subtract_torch = None
    else:
        # Long-S path: torch K-mean (well-optimized) + torch broadcast subtract.
        # Pre-subtract here so the kernel doesn't pay the inline-subtract
        # register-pressure cost.
        k_mean_torch = k.mean(dim=1, keepdim=True)
        k_subtract_torch = (k - k_mean_torch).contiguous()
        v_scale_torch = (v.abs().amax(dim=1).to(torch.float32) / fp8_max).contiguous()
        v_scale.copy_(v_scale_torch)
        # k_mean tensor is unused by the kernel in this branch (subtract_k_mean=False)
        k_c = k_subtract_torch  # use pre-subtracted K as kernel input

    # ---- Stage 2: fused MXFP4 quant kernel (Q rotate + K rotate-(maybe-mean)-quant + V FP8)
    launcher = _get_kernel(
        head_dim=D, blk_q=BLKQ, blk_k=BLKK,
        num_q_heads=Hq, num_kv_heads=Hk,
        subtract_k_mean=use_in_kernel_k_mean,
        q_smoothing=q_smoothing,
    )
    sm_bits   = _f32_bits(sm_scale_log2e)
    norm_bits = _f32_bits(1.0 / math.sqrt(D))
    launcher(
        q_c.reshape(-1),
        q_fp4.reshape(-1),
        q_d.reshape(-1),
        q_mean.reshape(-1),
        k_c.reshape(-1),
        k_fp4.reshape(-1),
        k_d.reshape(-1),
        k_mean.reshape(-1),
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

    # ---- Stage 3 (q_smoothing only): compute delta_s = Q_mean @ (K - K_mean).T
    delta_s = None
    if q_smoothing:
        # Empirically tuned: 128 beats 64 on S>=8192 (21% faster) and is a
        # tie or near-tie on shorter S. 256 is slightly better on a few
        # shapes but bigger register/LDS footprint risks long-S regressions
        # in pathological cases. 128 is the safe global default.
        DS_BLOCK_N = 128
        ds_num_blocks_n = (S_k + DS_BLOCK_N - 1) // DS_BLOCK_N
        delta_s = torch.empty(
            (B, Hq, NUM_BLKS_Q, S_k), dtype=torch.float32, device=q.device,
        )
        ds_grid = B * Hq * NUM_BLKS_Q * ds_num_blocks_n
        # When K-mean was pre-subtracted by torch (long-S branch), K_mean
        # is unused and we pass K_mean tensor (zeros not required since
        # the kernel branch on subtract_k_mean_in_kernel guards the load).
        ds_launcher = _get_delta_s_kernel(
            head_dim=D, num_q_heads=Hq, num_kv_heads=Hk,
            block_n=DS_BLOCK_N,
            subtract_k_mean_in_kernel=use_in_kernel_k_mean,
        )
        # K_in must be the same K seen by the main kernel:
        #  - in-kernel K-mean: k_c is raw, K_mean is real → kernel computes (K - K_mean)
        #  - torch K-mean   : k_c is already K - K_mean, K_mean tensor is unused
        ds_launcher(
            q_mean.reshape(-1),
            k_c.reshape(-1),
            k_mean.reshape(-1),
            delta_s.reshape(-1),
            S_k, NUM_BLKS_Q, ds_num_blocks_n,
            ds_grid,
            stream=stream,
        )

    if bhsd_in:
        q_fp4 = q_fp4.permute(0, 2, 1, 3).contiguous()
        q_d   = q_d.permute(0, 2, 1, 3).contiguous()
        k_fp4 = k_fp4.permute(0, 2, 1, 3).contiguous()
        k_d   = k_d.permute(0, 2, 1, 3).contiguous()
        v_fp8 = v_fp8.permute(0, 2, 1, 3).contiguous()
        # delta_s is [B, Hq, Q_NUM_BLKS, S_k] — already layout-agnostic

    return q_fp4, q_d, k_fp4, k_d, v_fp8, v_scale, delta_s
