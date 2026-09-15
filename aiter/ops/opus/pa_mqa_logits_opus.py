# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged MQA logits for DeepSeek-style sparse attention on gfx950 (OPUS kernels).

Thin launcher around the hand-written OPUS HIP kernels. Per query row ``r`` over a
window ``[s, e)``: ``out[r, s:e] = sum_H( relu(Q[r] . Kᵀ) * weight[r] ) * weight_scale``.

Two element types are supported, MXFP8 (E4M3) and MXFP4 (E2M1). They are separate
kernels on the C++ side -- the fp8 one sits at the VGPR=218 / occupancy-2 ceiling
while fp4 runs at VGPR=162 / occupancy-3 with zero LDS, so templating the element
type would risk the fp8 tuning for no benefit -- but everything above the launch is
identical, which is why they share this module. What differs per dtype:

* ``q`` is ``[T, H, D]`` for MXFP8 and ``[T, H, D/2]`` for MXFP4 (2 elements/byte,
  low nibble = even element).
* ``kv_cache`` is ``[num_blocks, K_TILES, 8, PAGE, 16]`` for MXFP8 (16-K chunks) and
  ``[num_blocks, K_TILES, 4, PAGE, 16]`` for MXFP4 (32-K chunks).

Everything else -- the E8M0 block-scale ABI (block=32), the ``q_scale`` preshuffle,
the window arrays, the grids -- is byte-identical between them. Q/KV must already be
quantized and preshuffled by the caller.

Both entry points are schedule-free (per-CTA assignment derived in-kernel from
``blockIdx``) and cudagraph-safe:

* ``pa_mqa_logits_{mxfp8,mxfp4}_prefill`` — ragged-window prefill (1D grid, one CTA/row).
* ``pa_mqa_logits_{mxfp8,mxfp4}_decode`` — MTP decode (3D grid), fixed-``next_n`` or varqlen.

``block_k`` picks between two compiled variants that produce identical results and
differ only in how much KV one CTA covers: 256 -> 4 waves/CTA, 64 -> 1 wave/CTA. It
is a pure performance knob; see :func:`pa_mqa_logits_mxfp4_prefill` for how to choose.
"""

import torch

from ...jit.core import compile_ops
from ...jit.utils.chip_info import get_gfx_runtime

MD_NAME_MXFP8 = "module_pa_mqa_logits_mxfp8_opus"
MD_NAME_MXFP4 = "module_pa_mqa_logits_mxfp4_opus"

DEFAULT_HEADS = 64
DEFAULT_HEAD_DIM = 128

# The two compiled variants, common to both dtypes. 64 (1 wave/CTA) is the better
# single default on every path; 256 (4 waves/CTA) only pays off once one CTA's
# window is long enough to amortize the wider tiles.
BLOCK_K_1WAVE = 64
BLOCK_K_4WAVE = 256


# ── JIT stubs: signatures must match the pybind macros exactly ────────────────
# PA_MQA_LOGITS_MXFP8_PYBIND / PA_MQA_LOGITS_MXFP4_PYBIND. The two dtypes are
# separate JIT modules, so importing this module builds neither until called.
@compile_ops(MD_NAME_MXFP8, develop=True)
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


@compile_ops(MD_NAME_MXFP4, develop=True)
def pa_mqa_logits_mxfp4_fwd_prefill(
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


@compile_ops(MD_NAME_MXFP8, develop=True)
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


@compile_ops(MD_NAME_MXFP4, develop=True)
def pa_mqa_logits_mxfp4_fwd_decode(
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


@compile_ops(MD_NAME_MXFP8, develop=True)
def pa_mqa_logits_mxfp8_prefill_windows(
    cu_seq_q: torch.Tensor,
    context_lens: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    total_q: int,
) -> None: ...


@compile_ops(MD_NAME_MXFP4, develop=True)
def pa_mqa_logits_mxfp4_prefill_windows(
    cu_seq_q: torch.Tensor,
    context_lens: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    total_q: int,
) -> None: ...


_WINDOW_BUILDERS = {
    "mxfp8": pa_mqa_logits_mxfp8_prefill_windows,
    "mxfp4": pa_mqa_logits_mxfp4_prefill_windows,
}


def compute_prefill_windows(
    cu_seq_q: torch.Tensor,
    context_lens: torch.Tensor,
    total_q: int,
    out: tuple | None = None,
    dtype: str = "mxfp8",
):
    """Build the per-row ``[local_start, local_end)`` window arrays the prefill launch
    consumes, from ``cu_seq_q`` + ``context_lens`` (MTP tail-causal; plain causal when
    ``qlen == ctx``). Device-side, cudagraph-safe.

    The window arrays carry no element type, so the two builders are the same kernel
    and either output feeds either dtype. ``dtype`` only picks which JIT module does
    the work — pass the one you are already using so the other never gets built.
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
    _WINDOW_BUILDERS[dtype](
        cu, ctx, row_to_batch, local_starts, local_ends, int(total_q)
    )
    return row_to_batch, local_starts, local_ends


# ── Shared launch bodies ──────────────────────────────────────────────────────
# Everything above the kernel call is dtype-independent: the ABI differs only in
# how many bytes a Q row and a KV chunk take, which the C++ side checks.


def _require_gfx950(name):
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"{name} requires gfx950, got {gfx}")


def _prefill(
    fwd,
    q,
    q_scale,
    kv_cache,
    kv_scale,
    block_tables,
    weights,
    row_to_batch,
    local_starts,
    local_ends,
    max_seq_len,
    weight_scale,
    block_k,
    kv_block_size,
    out,
):
    total_tokens = int(q.shape[0])
    if out is None:
        out = torch.full(
            (total_tokens, max_seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
    fwd(
        q,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch.to(torch.int32).contiguous(),
        local_starts.to(torch.int32).contiguous(),
        local_ends.to(torch.int32).contiguous(),
        out,
        total_tokens,
        float(weight_scale),
        int(block_k),
        int(kv_block_size),
        int(max_seq_len),
    )
    return out


def _decode(
    fwd,
    q,
    q_scale,
    kv_cache,
    kv_scale,
    block_tables,
    weights,
    context_lens,
    max_seq_len,
    next_n_max,
    split_ctx_len,
    cu_seq_q,
    weight_scale,
    block_k,
    kv_block_size,
    cta_target,
    out,
):
    block_k = int(block_k)
    total_q = int(q.shape[0])
    batch = int(context_lens.shape[0])
    next_n_max = int(next_n_max)
    if cu_seq_q is None:  # fixed-MTP: uniform per-batch qlen == next_n_max
        cu_seq_q = torch.arange(
            0,
            (batch + 1) * next_n_max,
            next_n_max,
            dtype=torch.int32,
            device=q.device,
        )
    else:
        cu_seq_q = cu_seq_q.to(torch.int32).contiguous()

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
            device=q.device,
        )
    fwd(
        q,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        cu_seq_q,
        context_lens.to(torch.int32).contiguous(),
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


_PREFILL_DOC = """Ragged-prefill paged MQA logits (gfx950), schedule-free: one CTA per
    query row (1D grid), each covering its whole ``[local_start, local_end)`` window.

    ``block_tables`` must be sized for the ``block_k`` in use. A reused ``out`` must be
    pre-filled with -inf (the kernel only writes in-window cells).

    Choosing ``block_k``: the quantity that decides it is the longest **window**
    ``max(local_end - local_start)``, not the query count. ``block_k`` quantizes a CTA's
    work into tiles, so a wide tile wastes MFMAs on short windows and has too few tiles
    to amortize the prologue over; on long windows it instead saves block-table lookups
    and loop iterations. Measured on a plain causal sweep (20 shapes, 16384 total_q), 64
    wins everywhere except the single-batch 16384-token case, where 256 is a couple of
    percent ahead -- so 64 is the default and 256 is worth passing only when every row's
    window is many thousands of tokens. Do NOT key this off ``q_len``: that only tracks
    the window length when ``kv_len == q_len``, and with a pooled KV cache it mispicks at
    both ends."""

_DECODE_DOC = """Decode MQA logits (MTP), schedule-free + cudagraph-safe. One path for
    fixed-MTP (``cu_seq_q=None``) and varqlen (``cu_seq_q`` given). ``q`` / ``weights`` /
    ``out`` are PACKED [total_q, ...]. 3D grid (batch, next_n_max, split_kv); the MTP
    tail-causal window is derived inline from ``cu_seq_q`` + ``context_lens`` (no window
    arrays / no window-build kernel).

    ``next_n_max`` (REQUIRED): MTP width = grid y-dim. Fixed MTP -> every batch has
    exactly ``next_n_max`` tokens (uniform ``cu_seq_q`` built here). Varqlen -> padded
    upper bound (>= max per-batch qlen; rows with ``n >= qlen`` idle). Always
    caller-supplied (no host sync).

    ``split_ctx_len`` (REQUIRED): max KV length any row actually attends to this launch,
    bounding split_kv = ``ceil(split_ctx_len / block_k)``. Distinct from ``max_seq_len``
    (output width): when a row processes only a slice of a longer sequence, sizing the
    split off ``max_seq_len`` would launch idle CTAs. Pass ``split_ctx_len ==
    max_seq_len`` when each row spans the full sequence.

    Choosing ``block_k``: decode windows are whole contexts, hence long, which favours
    the wide tile -- but decode also launches far fewer rows than prefill, and the narrow
    CTA's extra parallelism outweighs its extra loop iterations until the context work
    dominates. Over a 48-shape sweep 64 wins 42 of 48; 256 takes the lead only from about
    ``total_q * ctx >= 2M``, by 5-9%, and the margin narrows again at the very top. 64 is
    the default; measure before overriding."""


# ── Public wrappers, one pair per dtype ───────────────────────────────────────


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
    block_k: int = BLOCK_K_1WAVE,
    kv_block_size: int = 64,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _require_gfx950("pa_mqa_logits_mxfp8")
    return _prefill(
        pa_mqa_logits_mxfp8_fwd_prefill, q_fp8, q_scale, kv_cache, kv_scale,
        block_tables, weights, row_to_batch, local_starts, local_ends,
        max_seq_len, weight_scale, block_k, kv_block_size, out,
    )  # fmt: skip


def pa_mqa_logits_mxfp4_prefill(
    q_fp4: torch.Tensor,
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
    block_k: int = BLOCK_K_1WAVE,
    kv_block_size: int = 64,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _require_gfx950("pa_mqa_logits_mxfp4")
    return _prefill(
        pa_mqa_logits_mxfp4_fwd_prefill, q_fp4, q_scale, kv_cache, kv_scale,
        block_tables, weights, row_to_batch, local_starts, local_ends,
        max_seq_len, weight_scale, block_k, kv_block_size, out,
    )  # fmt: skip


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
    block_k: int = BLOCK_K_1WAVE,
    kv_block_size: int = 64,
    cta_target: int = 1024,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _require_gfx950("pa_mqa_logits_mxfp8")
    return _decode(
        pa_mqa_logits_mxfp8_fwd_decode, q_fp8, q_scale, kv_cache, kv_scale,
        block_tables, weights, context_lens, max_seq_len, next_n_max,
        split_ctx_len, cu_seq_q, weight_scale, block_k, kv_block_size,
        cta_target, out,
    )  # fmt: skip


def pa_mqa_logits_mxfp4_decode(
    q_fp4: torch.Tensor,
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
    block_k: int = BLOCK_K_1WAVE,
    kv_block_size: int = 64,
    cta_target: int = 1024,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _require_gfx950("pa_mqa_logits_mxfp4")
    return _decode(
        pa_mqa_logits_mxfp4_fwd_decode, q_fp4, q_scale, kv_cache, kv_scale,
        block_tables, weights, context_lens, max_seq_len, next_n_max,
        split_ctx_len, cu_seq_q, weight_scale, block_k, kv_block_size,
        cta_target, out,
    )  # fmt: skip


pa_mqa_logits_mxfp8_prefill.__doc__ = _PREFILL_DOC
pa_mqa_logits_mxfp4_prefill.__doc__ = _PREFILL_DOC
pa_mqa_logits_mxfp8_decode.__doc__ = _DECODE_DOC
pa_mqa_logits_mxfp4_decode.__doc__ = _DECODE_DOC


__all__ = [
    "compute_prefill_windows",
    "pa_mqa_logits_mxfp4_decode",
    "pa_mqa_logits_mxfp4_fwd_decode",
    "pa_mqa_logits_mxfp4_fwd_prefill",
    "pa_mqa_logits_mxfp4_prefill",
    "pa_mqa_logits_mxfp4_prefill_windows",
    "pa_mqa_logits_mxfp8_decode",
    "pa_mqa_logits_mxfp8_fwd_decode",
    "pa_mqa_logits_mxfp8_fwd_prefill",
    "pa_mqa_logits_mxfp8_prefill",
    "pa_mqa_logits_mxfp8_prefill_windows",
]
