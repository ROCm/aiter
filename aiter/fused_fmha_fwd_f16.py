# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
fused_fmha_fwd_f16
==================

Customer-facing API for the ASM-based FMHA forward kernel (BF16, gfx1250).

Layout convention
-----------------
Tensor shapes and their physical memory ordering are controlled by ``i_perm``
(input) and ``o_perm`` (output):

    0 = bshd  — [batch, seq, head, dim]
    1 = bhsd  — [batch, head, seq, dim]
    2 = sbhd  — [seq, batch, head, dim]   ← default input  (i_perm=2)

Default output is ``o_perm=0`` (bshd → [batch, seq_q, head_q, dim_v]).

Each tensor **must be contiguous** and its physical layout must match the
declared perm (e.g. for ``i_perm=2`` the tensor shape must be ``[s,b,h,d]``
with natural strides ``[b*h*d, h*d, d, 1]``).

Sink convention
---------------
``sink`` is an optional per-Q-head f32 tensor of shape ``[q_head_num]``.
Values are in the **AITER / CK-Tile post-scale domain** (same domain as the
softmax logit ``Q·Kᵀ / sqrt(d)``).  The kernel uses pre-scale internally;
this module performs the conversion: ``sink_raw = sink_user * sqrt(d)``.

Supported shapes
----------------
- ``q.shape`` determined by ``i_perm`` and ``(batch, q_head_num, q_seq_len, d)``
- ``d ∈ {64, 128}``
- dtype: bf16
- GQA: ``q_head_num % kv_head_num == 0``

The border variant (_brd) is selected automatically when ``q_seq_len`` is not
a multiple of 128 or ``kv_seq_len`` is not a multiple of 256.

Environment
-----------
Set ``AITER_ASM_DIR`` to ``{AITER_ROOT}/hsa`` and ``AITER_GPU_ARCHS=gfx1250``
so the compiled kernel objects (``*.co``) can be located at runtime.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch

from .ops.fmha_fwd_f16_asm import fmha_fwd_f16_asm


def fmha_fwd_f16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    return_lse: bool = False,
    i_perm: int = 2,
    o_perm: int = 0,
    sink: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """BF16 fused multi-head attention forward (ASM path, gfx1250).

    Parameters
    ----------
    q, k, v : torch.Tensor
        BF16 tensors.  Physical shape determined by ``i_perm``
        (default 2 = sbhd → ``[seq, batch, head, dim]``).
        All must be **contiguous**.
    softmax_scale : float, optional
        Defaults to ``1 / sqrt(head_dim)``.
    is_causal : bool
        Apply causal (lower-triangular) masking.
    return_lse : bool
        If True, also return LSE with shape ``[batch, q_head_num, q_seq_len]``
        in fp32.
    i_perm : int
        Input layout code: 0=bshd, 1=bhsd, 2=sbhd (default).
    o_perm : int
        Output layout code: 0=bshd (default), 1=bhsd, 2=sbhd.
    sink : torch.Tensor, optional
        Per-Q-head sink logits, shape ``[q_head_num]``, fp32, **post-scale**
        (AITER convention).  Converted to pre-scale internally.
        **Required for D64 (head_dim=64)** — D64 `_rxy_sink` kernels always
        run the sink code path.  Pass ``torch.zeros(q_head_num)`` for a
        neutral zero-logit sink.
        Optional for D128 (head_dim=128) — D128 kernels ignore this field.
    out : torch.Tensor, optional
        Pre-allocated output buffer matching ``o_perm`` shape.

    Returns
    -------
    torch.Tensor or (torch.Tensor, torch.Tensor)
        ``out`` alone if ``return_lse=False``, otherwise ``(out, lse)``.
    """
    if softmax_scale is None:
        # head_dim is always the last dimension regardless of perm
        softmax_scale = 1.0 / math.sqrt(q.size(-1))

    results = fmha_fwd_f16_asm(
        q,
        k,
        v,
        float(softmax_scale),
        bool(is_causal),
        bool(return_lse),
        int(i_perm),
        int(o_perm),
        sink,
        out,
    )

    if return_lse:
        assert len(results) == 2
        return results[0], results[1]
    return results[0]
