# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP8 paged MQA logits for DeepSeek-style sparse attention on gfx950.

Thin launcher around the hand-written OPUS HIP kernel
(``csrc/include/pa_mqa_logits_mxfp8_kernel.hpp``). The kernel computes, per query
row ``r`` over a window ``[s, e)``::

    out[r, s:e] = sum_H( relu(Q[r] . Kᵀ) * weight[r] ) * weight_scale

One compiled configuration (4-wave MXFP8 variant): ``H=64``, ``D=128``,
``block_k=256``, ``kv_block_size(PAGE)=64``.

Q/KV must already be MXFP8-quantized and preshuffled into the kernel ABI
(E4M3 data + E8M0 block scales, block=32); quant/preshuffle is the caller's
responsibility. The persistent-grid schedule (``cta_info[n_ctas, 6]``) is built
device-side and is cudagraph-safe:

* :func:`pa_mqa_logits_mxfp8_prefill` — ragged-window prefill.
* :func:`pa_mqa_logits_mxfp8_varqlen` — per-batch variable qlen (MTP decode),
  implemented as varqlen-over-prefill (same kernel + a window adapter).

The public wrappers mirror the FlyDSL ``flydsl_pa_mqa_logits_fp4_*`` signatures
so they can be A/B compared as drop-in replacements.
"""

import torch

from ...jit.core import compile_ops
from ...jit.utils.chip_info import get_gfx_runtime

MD_NAME = "module_pa_mqa_logits_mxfp8_opus"

CTA_INFO_WIDTH = 6
DEFAULT_HEADS = 64
DEFAULT_HEAD_DIM = 128


# ---------------------------------------------------------------------------
# JIT-compiled entry points (torch.Tensor -> aiter_tensor_t via develop=True).
# Signatures must match the pybind macro PA_MQA_LOGITS_MXFP8_PYBIND exactly.
# ---------------------------------------------------------------------------
@compile_ops("module_pa_mqa_logits_mxfp8_opus", develop=True)
def pa_mqa_logits_mxfp8_fwd(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    cta_info: torch.Tensor,
    out: torch.Tensor,
    n_ctas: int,
    weight_scale: float,
    block_k: int,
    kv_block_size: int,
    max_seq_len: int,
) -> None: ...


@compile_ops("module_pa_mqa_logits_mxfp8_opus", develop=True)
def pa_mqa_logits_mxfp8_fwd_direct(
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
    split_kv: int,
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
    axis: int,
    weight_scale: float,
    block_k: int,
    kv_block_size: int,
    max_seq_len: int,
) -> None: ...


@compile_ops("module_pa_mqa_logits_mxfp8_opus", develop=True)
def pa_mqa_logits_mxfp8_prefill_schedule(
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    scratch: torch.Tensor,
    cta_info: torch.Tensor,
    total_tokens: int,
    parallel_unit_num: int,
    block_k: int,
    max_seq_len: int,
) -> None: ...


@compile_ops("module_pa_mqa_logits_mxfp8_opus", develop=True)
def pa_mqa_logits_mxfp8_varqlen_windows(
    cu_seq_q: torch.Tensor,
    context_lens: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    total_q: int,
) -> None: ...


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------
def compute_prefill_schedule(
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    block_k: int,
    parallel_unit_num: int,
    max_seq_len: int,
    cta_info_out: torch.Tensor | None = None,
    scratch_out: torch.Tensor | None = None,
):
    """Build the persistent-grid schedule ``cta_info[P, 6]`` (device, graph-safe).

    Pass ``cta_info_out`` / ``scratch_out`` (fixed buffers) to write into stable
    addresses for CUDAGraph replay. Returns ``(cta_info, n_ctas)``.
    """
    device = local_ends.device
    T = int(local_ends.shape[0])
    P = int(parallel_unit_num)
    assert P >= T, (
        f"parallel_unit_num={P} < rows={T} would silently drop rows past slot "
        f"{P} (their logits stay at the caller's pre-fill -> wrong top-k)."
    )
    rb = row_to_batch.to(torch.int32)
    ls = local_starts.to(torch.int32)
    le = local_ends.to(torch.int32)

    cta_info = (
        cta_info_out
        if cta_info_out is not None
        else torch.empty(P, CTA_INFO_WIDTH, dtype=torch.int32, device=device)
    )
    scratch = (
        scratch_out
        if scratch_out is not None
        else torch.empty(T + 2, dtype=torch.int32, device=device)
    )
    pa_mqa_logits_mxfp8_prefill_schedule(
        rb, ls, le, scratch, cta_info, T, P, int(block_k), int(max_seq_len)
    )
    return cta_info, P


def compute_varqlen_windows(
    cu_seq_q: torch.Tensor,
    context_lens: torch.Tensor,
    total_q: int,
    out: tuple | None = None,
):
    """Build ragged-row window metadata for per-batch variable qlen (MTP)."""
    dev = cu_seq_q.device
    cu = cu_seq_q.to(torch.int32).contiguous()
    ctx = context_lens.to(torch.int32).contiguous()
    if out is None:
        row_to_batch = torch.empty(total_q, dtype=torch.int32, device=dev)
        local_starts = torch.empty(total_q, dtype=torch.int32, device=dev)
        local_ends = torch.empty(total_q, dtype=torch.int32, device=dev)
    else:
        row_to_batch, local_starts, local_ends = out
    pa_mqa_logits_mxfp8_varqlen_windows(
        cu, ctx, row_to_batch, local_starts, local_ends, int(total_q)
    )
    return row_to_batch, local_starts, local_ends


# ---------------------------------------------------------------------------
# Public wrappers (mirror flydsl_pa_mqa_logits_fp4_* signatures)
# ---------------------------------------------------------------------------
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
    parallel_unit_num: int = 512,
    out: torch.Tensor | None = None,
    cta_info: torch.Tensor | None = None,
    n_ctas: int | None = None,
) -> torch.Tensor:
    """Ragged-prefill MXFP8 paged MQA logits (gfx950).

    Defaults to the 1-wave variant (``block_k=64``, one warp/CTA, finer kv split), which mirrors
    the triton single-warp prefill kernel. Pass ``block_k=256`` to use the 4-wave variant instead.
    The launcher selects the compiled kernel by ``block_k`` (64 -> 1-wave, 256 -> 4-wave).
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"pa_mqa_logits_mxfp8 requires gfx950, got {gfx}")

    total_tokens = int(q_fp8.shape[0])

    if (cta_info is None) != (n_ctas is None):
        raise ValueError("Pass both cta_info and n_ctas, or neither.")
    schedule_internal = cta_info is None
    if schedule_internal:
        parallel_unit_num = max(int(parallel_unit_num), total_tokens)
        cta_info, n_ctas = compute_prefill_schedule(
            row_to_batch,
            local_starts,
            local_ends,
            block_k,
            parallel_unit_num,
            max_seq_len,
        )

    if out is None:
        out = torch.full(
            (total_tokens, max_seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp8.device,
        )
    elif schedule_internal:
        out.fill_(float("-inf"))

    pa_mqa_logits_mxfp8_fwd(
        q_fp8,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        cta_info,
        out,
        int(n_ctas),
        float(weight_scale),
        int(block_k),
        int(kv_block_size),
        int(max_seq_len),
    )
    return out


def pa_mqa_logits_mxfp8_varqlen(
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    max_seq_len: int,
    *,
    cu_seq_q: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
    windows: tuple | None = None,
    weight_scale: float = 1.0,
    block_k: int = 256,
    kv_block_size: int = 64,
    parallel_unit_num: int | None = None,
    out: torch.Tensor | None = None,
    cta_info: torch.Tensor | None = None,
    n_ctas: int | None = None,
) -> torch.Tensor:
    """Variable-qlen (per-batch MTP) MXFP8 paged MQA logits (gfx950).

    Defaults to the 4-wave variant (``block_k=256``) for decode. Implemented as
    varqlen-over-prefill: builds explicit ragged windows from ``cu_seq_q`` +
    ``context_lens`` (MTP tail-causal), then reuses the same kernel + schedule.
    Each batch's query length (``cu_seq_q[b+1]-cu_seq_q[b]``) may differ, so the
    per-batch MTP width is fully variable — there is no fixed ``next_n`` (the
    6-field prefill schedule maps rows via prefix-sum searchsorted, with no next_n
    modulo constraint). ``parallel_unit_num`` is auto-derived from static shapes
    (cudagraph-safe) when not given. The launcher selects the kernel by ``block_k``
    (256 -> 4-wave, 64 -> 1-wave).
    """
    total_q = int(q_fp8.shape[0])
    if windows is None:
        if cu_seq_q is None or context_lens is None:
            raise ValueError(
                "pa_mqa_logits_mxfp8_varqlen: pass windows=(row_to_batch, "
                "local_starts, local_ends) built via compute_varqlen_windows, "
                "or both cu_seq_q and context_lens to build them here."
            )
        windows = compute_varqlen_windows(cu_seq_q, context_lens, total_q)
    row_to_batch, local_starts, local_ends = windows

    if parallel_unit_num is None:
        chunks_per_seq = max(1, (max_seq_len + block_k - 1) // block_k)
        parallel_unit_num = total_q * chunks_per_seq

    return pa_mqa_logits_mxfp8_prefill(
        q_fp8,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        local_starts,
        local_ends,
        max_seq_len,
        weight_scale=weight_scale,
        block_k=block_k,
        kv_block_size=kv_block_size,
        parallel_unit_num=parallel_unit_num,
        out=out,
        cta_info=cta_info,
        n_ctas=n_ctas,
    )


# ---------------------------------------------------------------------------
# Schedule-free (direct) wrappers: no cta_info table / no compute_prefill_schedule.
# The per-CTA assignment is derived in-kernel from blockIdx + per-row windows +
# split_kv (triton/deepgemm style). grid = num_rows * split_kv.
# ---------------------------------------------------------------------------
def pa_mqa_logits_mxfp8_prefill_direct(
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
    """Ragged-prefill MQA logits without a schedule: one CTA per query row.

    grid = total_tokens (split_kv=1). Each CTA derives its window from the
    per-row arrays (row_to_batch / local_starts / local_ends) and covers its
    whole window (no context split). No ``compute_prefill_schedule`` / cta_info.
    ``out`` (if reused across calls) must be pre-filled with -inf: the kernel
    only writes in-window cells.
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"pa_mqa_logits_mxfp8 requires gfx950, got {gfx}")

    total_tokens = int(q_fp8.shape[0])
    if out is None:
        out = torch.full(
            (total_tokens, max_seq_len), float("-inf"), dtype=torch.float32, device=q_fp8.device
        )
    pa_mqa_logits_mxfp8_fwd_direct(
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
        1,
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
    *,
    next_n: int | None = None,
    cu_seq_q: torch.Tensor | None = None,
    next_n_max: int | None = None,
    weight_scale: float = 1.0,
    block_k: int = 64,
    kv_block_size: int = 64,
    cta_target: int | None = None,
    split_kv: int | None = None,
    axis: int = 3,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode MQA logits (MTP), schedule-free + cudagraph-safe. One path for both
    fixed-MTP (``next_n``) and varqlen (``cu_seq_q``) decode.

    ``q_fp8`` / ``weights`` / ``out`` are PACKED ([total_q, ...], total_q = sum of
    per-batch qlen; fixed MTP => batch*next_n). The kernel uses a 3D grid
    (split_kv, next_n_max, batch): batch/MTP-position/context-split come straight
    from blockIdx and the MTP tail-causal window is derived inline from the
    per-batch ``cu_seq_q`` + ``context_lens`` -- NO per-row window arrays and NO
    window-build kernel (decode schedule collapses to ~0).

    Context is split across ``split_kv`` only when the query rows alone can't fill
    the GPU (``total_q < cta_target``). All grid dims are static shapes
    (cudagraph-safe); ``context_lens`` values are only read in-kernel. Default
    ``cta_target`` is 1024 for the 1-wave variant (block_k=64) / 256 for 4-wave.
    For varqlen under CUDAGraph pass a static ``next_n_max`` (else it is derived
    from ``cu_seq_q`` with a host sync).
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"pa_mqa_logits_mxfp8 requires gfx950, got {gfx}")

    total_q = int(q_fp8.shape[0])
    batch = int(context_lens.shape[0])
    if cu_seq_q is None:
        if next_n is None:
            raise ValueError(
                "pa_mqa_logits_mxfp8_decode: pass next_n (fixed MTP) or cu_seq_q (varqlen)."
            )
        cu_seq_q = torch.arange(
            0, (batch + 1) * next_n, next_n, dtype=torch.int32, device=q_fp8.device
        )
        if next_n_max is None:
            next_n_max = int(next_n)
    else:
        cu_seq_q = cu_seq_q.to(torch.int32)
        if next_n_max is None:  # host sync; pass next_n_max explicitly for cudagraph capture
            next_n_max = int((cu_seq_q[1:] - cu_seq_q[:-1]).max().item())

    if split_kv is None:
        if cta_target is None:
            cta_target = 1024 if int(block_k) <= 64 else 256
        max_chunks = max(1, (int(max_seq_len) + int(block_k) - 1) // int(block_k))
        if total_q >= cta_target:
            split_kv = 1
        else:
            split_kv = min(max_chunks, (cta_target + total_q - 1) // total_q)

    if out is None:
        out = torch.full(
            (total_q, max_seq_len), float("-inf"), dtype=torch.float32, device=q_fp8.device
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
        int(axis),
        float(weight_scale),
        int(block_k),
        int(kv_block_size),
        int(max_seq_len),
    )
    return out


__all__ = [
    "pa_mqa_logits_mxfp8_fwd",
    "pa_mqa_logits_mxfp8_fwd_direct",
    "pa_mqa_logits_mxfp8_fwd_decode",
    "pa_mqa_logits_mxfp8_prefill",
    "pa_mqa_logits_mxfp8_prefill_direct",
    "pa_mqa_logits_mxfp8_prefill_schedule",
    "pa_mqa_logits_mxfp8_varqlen",
    "pa_mqa_logits_mxfp8_decode",
    "pa_mqa_logits_mxfp8_varqlen_windows",
    "compute_prefill_schedule",
    "compute_varqlen_windows",
]
