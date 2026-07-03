# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Flash Attention Backward API.

Three-pass execution (CK-style fused dQ+dK+dV):
  1. Preprocess     — delta = rowsum(O * dO)
  2. dQ/dK/dV kernel — outer-K, inner-Q MFMA (K/V LDS-resident). dK/dV in VGPR;
                       dQ = dS^T·K fused in-kernel via fp32 atomic-add.
  3. Cast           — fp32 → input dtype

Constraints: non-causal, MHA (Hq==Hk), BSHD layout, Sq multiple of 64,
             head_dim=128, bf16/fp16.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from .kernels.fmha_bwd_preprocess import build_fmha_bwd_preprocess_module
from .kernels.fmha_bwd_kernel import (
    build_fmha_bwd_kernel_module,
    BLOCK_M as _BWD_BLOCK_M,
    BLOCK_N as _BWD_BLOCK_N,
)

__all__ = ["flydsl_flash_attn_backward"]

_BLOCK_M = 16  # inner Q-block size baked into the v1 main kernel


@lru_cache(maxsize=32)
def _get_preprocess(head_dim: int, dtype_str: str):
    return build_fmha_bwd_preprocess_module(head_dim=head_dim, dtype=dtype_str)


@lru_cache(maxsize=32)
def _get_dkdv_kernel(head_dim: int, block_m: int, dtype_str: str):
    return build_fmha_bwd_kernel_module(
        head_dim=head_dim, block_m=block_m, dtype=dtype_str
    )


# Sq/Sk must be a multiple of the fused kernel's K-tile (BLOCK_N=64).
_REQ_MULTIPLE = _BWD_BLOCK_N  # 64


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
    """FlyDSL backward: 3-kernel pipeline (preprocess → dK/dV → dQ).

    BSHD layout, fp32 internal accumulators, output cast to q.dtype.
    Returns False when configuration is unsupported → caller falls back to Triton.
    """
    if causal:
        return False

    B, Sq, Hq, D = q.shape
    Sk = k.shape[1]
    Hk = k.shape[2]

    if Hq != Hk:
        return False
    if Sq % _REQ_MULTIPLE != 0:
        return False
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False

    dtype_str = "bf16" if q.dtype == torch.bfloat16 else "fp16"
    if stream is None:
        stream = torch.cuda.current_stream(q.device)

    # Contiguous inputs (FlyDSL JIT requires stride-1 axis)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = out.contiguous()
    do = do.contiguous()
    softmax_lse = softmax_lse.contiguous()

    # fp32 accumulation buffers
    dq_f32 = torch.zeros(B, Sq, Hq, D, dtype=torch.float32, device=q.device)
    dk_f32 = torch.zeros(B, Sk, Hk, D, dtype=torch.float32, device=q.device)
    dv_f32 = torch.zeros(B, Sk, Hk, D, dtype=torch.float32, device=q.device)
    delta = torch.zeros(B, Hq, Sq, dtype=torch.float32, device=q.device)

    def _strides(t, dims):  # BSHD → (batch, head, seq) reordered strides
        return [int(t.stride(d)) for d in dims]

    # ── Pass 1: preprocess (delta = rowsum(O * dO)) ──────────────────────
    prep = _get_preprocess(D, dtype_str)
    prep(
        out,
        do,
        delta,
        *_strides(out, [0, 2, 1, 3]),  # stride_ob, oh, om, ok
        *_strides(do, [0, 2, 1, 3]),
        int(delta.stride(0)),
        int(delta.stride(1)),
        int(delta.stride(2)),
        Sq,
        B,
        Hq,
        stream=stream,
    )

    # Helper: pass BSHD strides as (batch, head, seq) to kernels
    def _bhs(t):
        return int(t.stride(0)), int(t.stride(2)), int(t.stride(1))

    # ── Pass 2: fused dQ/dK/dV kernel (outer-K, inner-Q, MFMA) ──────────
    # dK/dV via VGPR accumulate; dQ = dS^T·K fused, fp32 atomic-add into dq_f32.
    dkdv = _get_dkdv_kernel(D, _BWD_BLOCK_M, dtype_str)
    dkdv(
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
        *_bhs(q),
        *_bhs(k),
        *_bhs(v),
        *_bhs(do),
        *_bhs(dq_f32),
        *_bhs(dk_f32),
        *_bhs(dv_f32),
        int(softmax_lse.stride(0)),
        int(softmax_lse.stride(1)),
        int(softmax_lse.stride(2)),
        int(delta.stride(0)),
        int(delta.stride(1)),
        int(delta.stride(2)),
        Sq,
        Sk,
        Hq,
        B,
        stream=stream,
    )

    # NOTE: dQ is now produced by the dK/dV kernel above (fused, CK-style, fp32
    # atomic-add into dq_f32) — the separate outer-Q dQ kernel is retired.

    # ── Pass 3: cast fp32 → input dtype ──────────────────────────────────
    dq.copy_(dq_f32.to(q.dtype))
    dk.copy_(dk_f32.to(k.dtype))
    dv.copy_(dv_f32.to(v.dtype))

    return True
