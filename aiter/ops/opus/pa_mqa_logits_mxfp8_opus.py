# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP8 paged MQA logits for DeepSeek-style sparse attention on gfx950.

Thin launcher around the hand-written OPUS HIP kernel. Per query row ``r`` over a
window ``[s, e)``: ``out[r, s:e] = sum_H( relu(Q[r] . Kᵀ) * weight[r] ) * weight_scale``.

Q/KV must already be MXFP8-quantized and preshuffled into the kernel ABI (E4M3
data + E8M0 block scales, block=32). Both entry points are schedule-free (per-CTA
assignment derived in-kernel from ``blockIdx``) and cudagraph-safe:

* :func:`pa_mqa_logits_mxfp8_prefill` — ragged-window prefill (1D grid, one CTA/row).
* :func:`pa_mqa_logits_mxfp8_decode` — MTP decode (3D grid), fixed-``next_n`` or varqlen.
"""

import torch

from ...jit.core import compile_ops
from ...jit.utils.chip_info import get_gfx_runtime

MD_NAME = "module_pa_mqa_logits_mxfp8_opus"

DEFAULT_HEADS = 64
DEFAULT_HEAD_DIM = 128


# JIT stubs: signatures must match the pybind macro PA_MQA_LOGITS_MXFP8_PYBIND exactly.
@compile_ops("module_pa_mqa_logits_mxfp8_opus", develop=True)
def pa_mqa_logits_mxfp8_fwd_prefill(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    out: torch.Tensor,
    num_rows: int,
    weight_scale: float,
    block_k: int,
    kv_block_size: int,
    max_seq_len: int,
) -> None: ...


@compile_ops("module_pa_mqa_logits_mxfp8_opus", develop=True)
def pa_mqa_logits_mxfp8_fwd_decode(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    cu_seq_q: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor,
    batch: int,
    next_n_max: int,
    split_kv: int,
    weight_scale: float,
    block_k: int,
    kv_block_size: int,
    max_seq_len: int,
) -> None: ...


@compile_ops("module_pa_mqa_logits_mxfp8_opus", develop=True)
def pa_mqa_logits_mxfp8_prefill_windows(
    cu_seq_q: torch.Tensor,
    context_lens: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    total_q: int,
) -> None: ...


def compute_prefill_windows(
    cu_seq_q: torch.Tensor,
    context_lens: torch.Tensor,
    total_q: int,
    out: tuple | None = None,
):
    """Build the per-row ``[local_start, local_end)`` window arrays that
    :func:`pa_mqa_logits_mxfp8_prefill` consumes, from ``cu_seq_q`` + ``context_lens``
    (MTP tail-causal; plain causal when ``qlen == ctx``). Device-side, cudagraph-safe.
    """
    dev = cu_seq_q.device
    cu = cu_seq_q.to(torch.int32).contiguous()
    ctx = context_lens.to(torch.int32).contiguous()
    if out is None:
        row_to_batch = torch.empty(total_q, dtype=torch.int32, device=dev)
        local_starts = torch.empty(total_q, dtype=torch.int32, device=dev)
        local_ends = torch.empty(total_q, dtype=torch.int32, device=dev)
    else:
        row_to_batch, local_starts, local_ends = out
    pa_mqa_logits_mxfp8_prefill_windows(
        cu, ctx, row_to_batch, local_starts, local_ends, int(total_q)
    )
    return row_to_batch, local_starts, local_ends


# Public wrappers (mirror flydsl_pa_mqa_logits_fp4_* signatures).
def pa_mqa_logits_mxfp8_prefill(
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    max_seq_len: int,
    *,
    weight_scale: float = 1.0,
    block_k: int = 64,
    kv_block_size: int = 64,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ragged-prefill MXFP8 paged MQA logits (gfx950), schedule-free: one CTA per
    query row (1D grid), each covering its whole ``[local_start, local_end)`` window.

    ``block_k`` selects the compiled variant (64 -> 1-wave default, 256 -> 4-wave).
    A reused ``out`` must be pre-filled with -inf (the kernel only writes in-window cells).
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"pa_mqa_logits_mxfp8 requires gfx950, got {gfx}")

    total_tokens = int(q_fp8.shape[0])
    if out is None:
        out = torch.full(
            (total_tokens, max_seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp8.device,
        )
    pa_mqa_logits_mxfp8_fwd_prefill(
        q_fp8,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch.to(torch.int32),
        local_starts.to(torch.int32),
        local_ends.to(torch.int32),
        out,
        total_tokens,
        float(weight_scale),
        int(block_k),
        int(kv_block_size),
        int(max_seq_len),
    )
    return out


def pa_mqa_logits_mxfp8_decode(
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    max_seq_len: int,
    next_n_max: int,
    *,
    split_ctx_len: int,
    cu_seq_q: torch.Tensor | None = None,
    weight_scale: float = 1.0,
    kv_block_size: int = 64,
    cta_target: int = 1024,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode MQA logits (MTP), 1-wave, schedule-free + cudagraph-safe. One path for
    fixed-MTP (``cu_seq_q=None``) and varqlen (``cu_seq_q`` given). ``q_fp8`` /
    ``weights`` / ``out`` are PACKED [total_q, ...]. 3D grid (batch, next_n_max,
    split_kv); the MTP tail-causal window is derived inline from ``cu_seq_q`` +
    ``context_lens`` (no window arrays / no window-build kernel).

    ``next_n_max`` (REQUIRED): MTP width = grid y-dim. Fixed MTP -> every batch has
    exactly ``next_n_max`` tokens (uniform ``cu_seq_q`` built here). Varqlen -> padded
    upper bound (>= max per-batch qlen; rows with ``n >= qlen`` idle). Always
    caller-supplied (no host sync).

    ``split_ctx_len`` (REQUIRED): max KV length any row actually attends to this
    launch, bounding split_kv = ``ceil(split_ctx_len / block_k)``. Distinct from
    ``max_seq_len`` (output width): when a row processes only a slice of a longer
    sequence, sizing the split off ``max_seq_len`` would launch idle CTAs. Pass
    ``split_ctx_len == max_seq_len`` when each row spans the full sequence.
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"pa_mqa_logits_mxfp8 requires gfx950, got {gfx}")

    block_k = 64  # decode is 1-wave
    total_q = int(q_fp8.shape[0])
    batch = int(context_lens.shape[0])
    next_n_max = int(next_n_max)
    if cu_seq_q is None:  # fixed-MTP: uniform per-batch qlen == next_n_max
        cu_seq_q = torch.arange(
            0,
            (batch + 1) * next_n_max,
            next_n_max,
            dtype=torch.int32,
            device=q_fp8.device,
        )
    else:
        cu_seq_q = cu_seq_q.to(torch.int32)

    # split context across CTAs only when query rows alone under-fill the GPU.
    max_chunks = max(1, (int(split_ctx_len) + block_k - 1) // block_k)
    if total_q >= cta_target:
        split_kv = 1
    else:
        split_kv = min(max_chunks, (cta_target + total_q - 1) // total_q)

    if out is None:
        out = torch.full(
            (total_q, max_seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp8.device,
        )
    pa_mqa_logits_mxfp8_fwd_decode(
        q_fp8,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        cu_seq_q,
        context_lens.to(torch.int32),
        out,
        int(batch),
        int(next_n_max),
        int(split_kv),
        float(weight_scale),
        block_k,
        int(kv_block_size),
        int(max_seq_len),
    )
    return out


__all__ = [
    "compute_prefill_windows",
    "pa_mqa_logits_mxfp8_decode",
    "pa_mqa_logits_mxfp8_fwd_decode",
    "pa_mqa_logits_mxfp8_fwd_prefill",
    "pa_mqa_logits_mxfp8_prefill",
    "pa_mqa_logits_mxfp8_prefill_windows",
]
