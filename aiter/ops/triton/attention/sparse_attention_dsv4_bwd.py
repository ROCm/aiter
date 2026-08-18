# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 sparse-MLA training BACKWARD (gfx950 / CDNA4).

Counterpart to the DSv4 sparse prefill forward. The op is the official V4 form: shared-KV GQA
where ``K == V == kv`` is a single dense 512-wide tensor, RoPE already applied in place
caller-side, scale ``1/sqrt(512)``, ``attn_sink`` folded into the softmax denominator only, and
``topk_indices == -1`` masked out.

    P     = exp(Q@kv^T * scale - lse)
    dP    = dO@kv^T
    delta = rowsum(O * dO)
    dS    = P * (dP - delta) * scale
    dQ    = dS @ kv
    dKV   = scatter_add over top-k of  sum_h ( dS*Q + P*dO )

Pipeline (five kernels + one torch reduction), per rank chunk:

    delta            triton    rowsum(O*dO)
    dQ               gluon     also emits this chunk's dS / P
    dKV-interm       gluon     interm[t, slot, d] = sum_h (dS*Q + P*dO)
    CSR build        torch     inverted top-k index (sort + searchsorted)
    dKV gather       triton    reduce interm over the top-k mapping, atomic-free
    d_sink           torch     26 us, not worth a kernel

``lse`` and ``o`` come from the forward. The merged Gluon prefill kernel produces both::

    from aiter.ops.triton.gluon.mla_gluon import mla_gluon
    o, lse = mla_gluon(..., has_pe=False, attn_sink=sink, return_lse=True)

Its ``lse`` is sink-inclusive (the sink is folded into ``e_max``/``e_sum`` before
``lse = e_max + log(e_sum)``), which is the convention this backward expects.

Measured on MI355X at ``T=4096 H=128 topk=512`` with a realistic SWA(128)+pool top-k:
delta 0.178 / dQ 1.391 / interm 1.152 / CSR build 0.130 / gather 0.503 / d_sink 0.026 ms,
3.380 ms total = 407 TFLOPS.
"""

import torch

from aiter.ops.triton.gluon.sparse_attention_dsv4_bwd import (
    build_inverted_topk,
    delta_v4,
    dkv_gather_acc,
    sparse_mla_bwd_dkv_interm_v4 as _dkv_interm_gluon,
    sparse_mla_bwd_dq as _dq_gluon,
)
from aiter.ops.triton.utils._triton import arch_info

_BLOCK_H_DQ = 64
_TILE_K_DQ = 32
_BD_DKV = 256
_TILE_K_DKV = 128


def sparse_mla_bwd_dsv4(
    q,
    kv,
    do,
    o,
    lse,
    topk_indices,
    attn_sink=None,
    scale=None,
    R_CHUNK=None,
):
    """Backward for the DSv4 sparse-MLA prefill attention. gfx950 (CDNA4) only.

    Args:
        q:            [T, H, 512]      bf16
        kv:           [num_kv, 512]    bf16, K == V. ``num_kv >= T``; rows ``T..num_kv-1`` are
                                       the compressed pool, which ``topk_indices`` may reference.
        do:           [T, H, 512]      bf16, gradient of the attention output
        o:            [T, H, 512]      bf16, the forward output
        lse:          [T, H]           fp32, sink-inclusive log-sum-exp from the forward
        topk_indices: [T, TOPK]        int32, -1 marks an invalid slot
        attn_sink:    [H]              fp32 per-head sink bias, or None
        scale:        softmax scale, defaults to ``1/sqrt(512)``
        R_CHUNK:      split the rank dimension into chunks of this width. ``None`` (default)
                      runs unchunked, which is what you want. Chunking exists only to bound the
                      ``interm`` intermediate, which is ``T*TOPK*512`` bf16 (2.0 GiB at
                      T=4096, TOPK=512); it costs a dQ read-modify-write between chunks and one
                      CSR build per chunk. Any multiple of 32 is accepted.

    Returns:
        dq [T, H, 512] bf16, dkv [num_kv, 512] bf16, d_sink [H] fp32 (None if no ``attn_sink``)
    """
    assert (
        arch_info.get_arch() == "gfx950"
    ), f"sparse_mla_bwd_dsv4 requires gfx950 (CDNA4), got {arch_info.get_arch()}"

    T, H, D = q.shape
    TOPK = topk_indices.shape[1]
    num_kv = kv.shape[0]
    assert D == 512, f"DSv4 sparse-MLA backward is fixed to head_dim 512, got {D}"
    assert kv.shape[-1] == D and do.shape == q.shape and o.shape == q.shape
    assert num_kv >= T, f"num_kv ({num_kv}) must be >= T ({T})"
    assert q.is_contiguous() and kv.is_contiguous() and do.is_contiguous()
    assert o.is_contiguous() and topk_indices.is_contiguous()

    if scale is None:
        scale = 1.0 / (D**0.5)
    if R_CHUNK is None:
        R_CHUNK = TOPK
    lse = lse.float().contiguous()

    # Both mfma tiles must divide the chunk width; step down rather than making it the
    # caller's problem, so a small R_CHUNK still works.
    tk_dkv = next(
        (t for t in (_TILE_K_DKV, 64, 32) if t <= R_CHUNK and R_CHUNK % t == 0), None
    )
    tk_dq = next(
        (t for t in (_TILE_K_DQ, 32) if t <= R_CHUNK and R_CHUNK % t == 0), None
    )
    assert (
        tk_dkv is not None and tk_dq is not None
    ), f"R_CHUNK={R_CHUNK} must be a multiple of 32 (it is the mfma tile width)"

    delta = delta_v4(o, do)

    dq = torch.empty_like(q)
    dkv_acc = torch.zeros(num_kv, D, dtype=torch.float32, device=q.device)
    chunk_dS = torch.empty(T, H, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    chunk_P = torch.empty(T, H, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    interm = torch.empty(T, R_CHUNK, D, dtype=torch.bfloat16, device=q.device)

    for r in range(0, TOPK, R_CHUNK):
        _dq_gluon(
            q,
            kv,
            do,
            topk_indices,
            lse,
            delta,
            dq,
            chunk_dS,
            chunk_P,
            scale,
            r,
            R_CHUNK,
            BLOCK_H=_BLOCK_H_DQ,
            TILE_K=tk_dq,
            is_first_chunk=(r == 0),
        )
        _dkv_interm_gluon(
            q, do, chunk_dS, chunk_P, R_CHUNK, BD=_BD_DKV, TILE_K=tk_dkv, interm=interm
        )
        inv_ptr, inv_data = build_inverted_topk(
            topk_indices[:, r : r + R_CHUNK], num_kv
        )
        dkv_gather_acc(interm, inv_ptr, inv_data, dkv_acc)

    dkv = dkv_acc.to(kv.dtype)

    d_sink = None
    if attn_sink is not None:
        # d_sink[h] = -sum_t exp(sink[h] - lse[t,h]) * delta[t,h]
        d_sink = -(torch.exp(attn_sink[None, :].float() - lse) * delta).sum(dim=0)

    return dq, dkv, d_sink


__all__ = ["sparse_mla_bwd_dsv4"]
