# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Flash Attention Backward API.

Three-pass execution:
  1. Preprocess  — delta = rowsum(O * dO), shape [B, H, Sq]
  2. Main kernel — compute dQ (atomic-add fp32), dK, dV
  3. Cast        — fp32 accumulators → input dtype via Python .to()

Limitations (v1 — end-to-end correctness target, CK-algorithm rewrite next):
  - Non-causal only
  - MHA only (num_q_heads == num_k_heads)
  - BSHD layout tensors (matches mha.py convention)
  - seqlen_q must be a multiple of BLOCK_M (=16)
  - head_dim must be a multiple of warp size (64 on gfx950)
  - bf16 / fp16 inputs

Returns None when any constraint is unmet so callers can fall back to Triton.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from .kernels.fmha_bwd_preprocess import build_fmha_bwd_preprocess_module
from .kernels.fmha_bwd_kernel import build_fmha_bwd_kernel_module

__all__ = ["flydsl_flash_attn_backward"]

_BLOCK_M = 16  # inner Q-block size baked into the v1 main kernel


@lru_cache(maxsize=32)
def _get_preprocess(head_dim: int, dtype_str: str):
    return build_fmha_bwd_preprocess_module(head_dim=head_dim, dtype=dtype_str)


@lru_cache(maxsize=32)
def _get_main_kernel(head_dim: int, block_m: int, dtype_str: str):
    return build_fmha_bwd_kernel_module(
        head_dim=head_dim, block_m=block_m, dtype=dtype_str
    )


def flydsl_flash_attn_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    do: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    sm_scale: float,
    causal: bool = False,
    stream: torch.cuda.Stream | None = None,
) -> bool:
    """Run FlyDSL backward attention.

    All tensors are in BSHD layout ([B, S, H, D]).
    softmax_lse is [B, H, Sq] fp32 (produced by the forward pass).
    dq/dk/dv are pre-allocated output tensors (same dtype as q/k/v).

    Returns True on success, False if the configuration is not supported
    (callers should fall back to Triton/CK in that case).
    """
    if causal:
        return False

    B, Sq, Hq, D = q.shape
    Sk = k.shape[1]
    Hk = k.shape[2]

    if Hq != Hk:
        return False  # GQA not supported in v1

    if Sq % _BLOCK_M != 0:
        return False  # partial last Q-block not yet handled

    if q.dtype not in (torch.bfloat16, torch.float16):
        return False

    dtype_str = "bf16" if q.dtype == torch.bfloat16 else "fp16"

    if stream is None:
        stream = torch.cuda.current_stream(q.device)

    # FlyDSL JIT requires at least one stride-1 axis. PyTorch may pack saved
    # tensors in non-standard layouts (e.g. strides all 0 for a broadcast
    # placeholder). Make all inputs contiguous to guarantee valid strides.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = out.contiguous()
    do = do.contiguous()
    softmax_lse = softmax_lse.contiguous()

    # Allocate fp32 accumulation buffers (same layout as inputs: BSHD)
    dq_f32 = torch.zeros(B, Sq, Hq, D, dtype=torch.float32, device=q.device)
    dk_f32 = torch.zeros(B, Sk, Hk, D, dtype=torch.float32, device=q.device)
    dv_f32 = torch.zeros(B, Sk, Hk, D, dtype=torch.float32, device=q.device)
    delta = torch.zeros(B, Hq, Sq, dtype=torch.float32, device=q.device)

    # --- Pass 1: preprocess (delta = rowsum(O * dO)) ---
    # Kernel grid: (seqlen_q, batch, num_heads)
    # Strides passed as: stride_ob, stride_oh, stride_om, stride_ok
    # For BSHD layout [B, S, H, D]:  stride_ob=S*H*D, stride_oh=D, stride_om=H*D, stride_ok=1
    prep = _get_preprocess(D, dtype_str)
    prep(
        out,
        do,
        delta,
        # O strides: batch, head, seq-row, head-dim
        int(out.stride(0)),
        int(out.stride(2)),
        int(out.stride(1)),
        int(out.stride(3)),
        # dO strides
        int(do.stride(0)),
        int(do.stride(2)),
        int(do.stride(1)),
        int(do.stride(3)),
        # delta strides: [B, H, Sq] contiguous
        int(delta.stride(0)),
        int(delta.stride(1)),
        int(delta.stride(2)),
        Sq,
        B,
        Hq,
        stream=stream,
    )

    # --- Pass 2: main backward (dQ atomic-add fp32, dK/dV register-accumulate) ---
    # Kernel indexes tensors as: base = batch*sb + head*sh + row*sm + col*tid
    # For BSHD tensors, map kernel params:
    #   stride_qb  = q.stride(0)   (batch)
    #   stride_qh  = q.stride(2)   (head — dim 2 in BSHD)
    #   stride_qm  = q.stride(1)   (seq row — dim 1 in BSHD)
    bwd = _get_main_kernel(D, _BLOCK_M, dtype_str)
    bwd(
        q,
        k,
        v,
        do,
        dq_f32,
        dk_f32,
        dv_f32,
        softmax_lse,
        delta,
        sm_scale,
        # Q strides (batch, head, seq)
        int(q.stride(0)),
        int(q.stride(2)),
        int(q.stride(1)),
        # K strides
        int(k.stride(0)),
        int(k.stride(2)),
        int(k.stride(1)),
        # V strides
        int(v.stride(0)),
        int(v.stride(2)),
        int(v.stride(1)),
        # dO strides
        int(do.stride(0)),
        int(do.stride(2)),
        int(do.stride(1)),
        # dQ strides (BSHD fp32 buffer)
        int(dq_f32.stride(0)),
        int(dq_f32.stride(2)),
        int(dq_f32.stride(1)),
        # dK strides
        int(dk_f32.stride(0)),
        int(dk_f32.stride(2)),
        int(dk_f32.stride(1)),
        # dV strides
        int(dv_f32.stride(0)),
        int(dv_f32.stride(2)),
        int(dv_f32.stride(1)),
        # softmax_lse strides: [B, H, Sq]
        int(softmax_lse.stride(0)),
        int(softmax_lse.stride(1)),
        int(softmax_lse.stride(2)),
        # delta strides: [B, H, Sq]
        int(delta.stride(0)),
        int(delta.stride(1)),
        int(delta.stride(2)),
        Sq,
        Sk,
        Hq,
        B,
        stream=stream,
    )

    # --- Pass 3: cast fp32 accumulators → input dtype ---
    dq.copy_(dq_f32.to(q.dtype))
    dk.copy_(dk_f32.to(k.dtype))
    dv.copy_(dv_f32.to(v.dtype))

    return True
