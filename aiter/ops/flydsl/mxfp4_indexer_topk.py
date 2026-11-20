# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MXFP4 sparse-attention indexer: the top-512 keys of decode rows
without materialising the ``[rows, context]`` logits (gfx950).

The logit of key ``t`` for decode row ``r`` is the one
``aiter.ops.triton.attention.pa_mqa_logits_mxfp4.paged_mxfp4_mqa_logits``
computes, ``sum_h w[r, h] * relu(q[r, h] . k[t])`` over 32 heads of 128 dims
(e2m1 codes, e8m0 scales per 32 dims, exact products, fp32 sums), read from the
same preshuffled MXFP4 paged cache (``cache_format(32, 128, page)``: 32-token
groups, values as ``[chunk][token][16 B]``, scales as
``[scale % 2][token][scale // 2]``). The ops return each row's 512 best keys:
positions ascending, -1 padded, ties to the lower position, the same bits run to
run.

Entry points:

- ``flydsl_mxfp4_indexer_topk``: a dense indexer layer. Optionally also writes
  the row's 2048 candidate 8-token blocks (``candidates_out``: the block of the
  row's newest key and the 2047 best others by block maximum, the two-level
  indexer's candidate source) and the sparse attention's ragged slot lists
  (``ragged``).
- ``flydsl_mxfp4_indexer_candidate_topk``: the top-512 among a row's candidate
  blocks (the two-level indexer's consumer layers), optionally the ragged lists.
- ``flydsl_mxfp4_indexer_block_scores``: the 8-token block maxima alone.
- ``mxfp4_indexer_unsupported_reason``: why a layer cannot be served (None when
  it can), so a caller falls back to ``paged_mxfp4_mqa_logits`` + top-k.
- ``mxfp4_indexer_dense_preferred``: whether a dense or candidate-source call
  of a given ``max_ctx`` and row count is faster than
  ``paged_mxfp4_mqa_logits`` + top-k on MI355X. The candidate consumer was
  faster at every measured size from 16384 entries up (at 4096 entries it
  lost with 18 and 66 rows).
- ``mxfp4_indexer_workspace_bytes``: the scratch a call needs, for a caller
  that passes its own ``workspace``.

Supported: 32 heads x 128 dims, 64- or 128-entry pages, 1-8 rows per request
(``next_n``), up to ``MXFP4_INDEXER_MAX_ENTRIES`` (1048576) entries a row, int32
block tables; a call of zero rows returns at once. Every launch reads the
lengths on the device: CUDA-graph safe once a call of the same (next_n, page
size, ``max_ctx`` tier, options) ran eagerly (it compiles the kernels) and,
without ``workspace``, once a call at least as large ran eagerly (it sizes the
shared scratch).

NaN logits (a key or query with e8m0 scale 0xFF, NaN in the MX format) are
never selected, like positions past the row's end, where a top-k over
``paged_mxfp4_mqa_logits``' output ranks them first. A row with fewer than
min(512, row_end) other keys then has fewer positions (and its ragged ``lens``
entry still counts min(512, row_end)).

Dispatch (``max_ctx``: the launch's static bound on a row's entries):

- ``max_ctx <= 16384``: token path. The walk writes every token's logit and a
  256- or 512-thread workgroup a row selects the 512 best by four 8-bit radix
  passes.
- ``16384 < max_ctx <= 262144``: block path. The walk keeps the maximum logit
  of every 8-token block (one key read serves all rows of a request); a
  workgroup a row keeps a bucketed superset of the 512 best blocks, scores
  their tokens again with the same arithmetic and selects exactly.
- ``max_ctx > 262144``: rows up to 262144 entries take the block path, longer
  rows a segmented block select + merge, then the candidate scorer and token
  select below, on a second stream beside the block select.
- Candidate consumer: ``max_ctx > 262144`` always, else up to 160 rows when
  ``max_ctx`` is at least 3000 + 110 x rows entries: the 16384 candidate tokens
  are scored by (row, 16-tile segment) workgroups and a token select picks the
  512 best. Otherwise one workgroup a row scores and selects.

Every path makes the same arithmetic on every token, so a key scores to the same
fp32 bits on all of them and the paths agree bit for bit.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import torch

from .kernels.mxfp4_indexer.blocks import (
    PLAIN_SLOTS,
    SEGB,
    SOURCE_SLOTS,
    compile_block_select,
)
from .kernels.mxfp4_indexer.cand_logits import compile_candidate_logits
from .kernels.mxfp4_indexer.common import BLOCK, CANDIDATES, ENTRY_BYTES, TOPK, D, H
from .kernels.mxfp4_indexer.scores import compile_block_scores
from .kernels.mxfp4_indexer.select import compile_select
from .kernels.mxfp4_indexer.tokens import compile_tokens_select
from .kernels.tensor_shim import _run_compiled

MXFP4_INDEXER_TOPK = TOPK
MXFP4_INDEXER_CANDIDATES = CANDIDATES
MXFP4_INDEXER_CANDIDATE_BLOCK = BLOCK
MXFP4_INDEXER_HEADS = H
MXFP4_INDEXER_HEAD_DIM = D
MXFP4_INDEXER_MAX_ROWS = 2048  # rows of a step the ragged lists cover
MXFP4_INDEXER_MAX_ENTRIES = 1 << 20  # entries per row
PAGE_ENTRIES = (64, 128)

# The one-workgroup block select keeps a row's block maxima in registers and its
# page table in LDS up to this many entries; longer rows take the long path.
BLOCK_CTX_MAX = 262144
# The gathered token select's slot table covers positions up to this by default.
SHORT_ENTRIES = 262144
TIERS = (4096, 8192, 16384, 32768, 65536, 131072)  # blocks of a row's context
# Walk workgroups a step stays within: 4 a CU is one resident wave of the walk
# (218 VGPRs: 2 waves a SIMD, 2-wave workgroups).
SEG_GRID = 1024
LONG_TILES = 8192  # tiles of 32 entries: 262144 entries
SELECT_THREADS = 512  # the dense select's workgroup (two rows a CU)
GATHER_SELECT_THREADS = 1024  # the candidate consumer's gather select
# Token path: a tier is the logits row width and its select's threads.
TOKEN_TIERS = {4096: 256, 8192: 512, 16384: 512}
# Candidate consumer: split scoring over (row, CAND_TPS-tile segment)
# workgroups up to CAND_SPLIT_ROWS rows when the context is above
# CAND_SPLIT_CTX0 + CAND_SPLIT_CTX_ROW * rows (the one-workgroup-a-row gather
# select grows with the context, the split does not).
CAND_SPLIT_ROWS = 160
CAND_TPS = 16
CAND_SPLIT_CTX0 = 3000
CAND_SPLIT_CTX_ROW = 110


class Mxfp4IndexerRagged(NamedTuple):
    """The sparse attention's input lists, written by the select.

    ``slots`` [rows * 512] int32: row r's slots at ``indptr[r]``, the position
    ``p`` of each selected key in order as
    ``slot_block_table[r // next_n][p // B] * B + p % B`` (``B`` =
    ``slot_block_size`` entries per page of the attention cache: 64, 128 or
    256); ``indptr`` [rows + 1] int32; ``lens`` [rows] int32 (``min(512,
    row_end)`` where ``is_valid`` [rows] int32 is set, else 0);
    ``slot_block_table`` [requests, pages] int32.
    """

    slots: torch.Tensor
    indptr: torch.Tensor
    lens: torch.Tensor
    is_valid: torch.Tensor
    slot_block_table: torch.Tensor
    slot_block_size: int


@functools.cache
def _arch_ok() -> bool:
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() == "gfx950"


def mxfp4_indexer_unsupported_reason(
    page_entries: int,
    max_ctx: int,
    next_n: int,
    num_heads: int = H,
    head_dim: int = D,
    topk: int = TOPK,
) -> str | None:
    """Why the FlyDSL indexer cannot serve a layer, or None when it can. Cheap:
    host values only, no device reads."""
    if not _arch_ok():
        return "needs gfx950"
    if (num_heads, head_dim) != (H, D):
        return f"{num_heads} heads x {head_dim} dims (needs {H} x {D})"
    if topk != TOPK:
        return f"top-{topk} (needs {TOPK})"
    if page_entries not in PAGE_ENTRIES:
        return f"{page_entries} entries per page (64 or 128)"
    if not 1 <= next_n <= 8:
        return f"{next_n} rows per request (1-8)"
    if not 0 < max_ctx <= MXFP4_INDEXER_MAX_ENTRIES:
        return f"context {max_ctx} entries (1 to {MXFP4_INDEXER_MAX_ENTRIES})"
    return None


# Where a dense / candidate-source call beat paged_mxfp4_mqa_logits + top-k on
# MI355X (graph replay, 6 rows a request, rows of 0.75-1x max_ctx entries,
# 18-768 rows measured): (largest max_ctx, fewest rows of a dense call, of a
# source call); None: mixed or slower at the measured row counts. Dense calls
# up to 32768 entries lose or tie (the token select's fixed cost); the source
# layer wins there except at 16384, as its baseline adds a top-2048. Long
# bounds need enough rows to fill the GPU. Dense at 262144 won at 6 of 7
# measured row counts (0.96x at 126 rows).
DENSE_PREFERRED = (
    (8192, None, 1),
    (16384, None, None),
    (32768, None, 1),
    (131072, 1, 1),
    (262144, 1, 192),
    (524288, 126, 96),
    (MXFP4_INDEXER_MAX_ENTRIES, 1, 1),
)


def mxfp4_indexer_dense_preferred(
    max_ctx: int, rows: int, source: bool = False
) -> bool:
    """Whether ``flydsl_mxfp4_indexer_topk`` of ``rows`` decode rows (with
    ``candidates_out`` when ``source``) is expected to beat
    ``paged_mxfp4_mqa_logits`` + top-k for a launch bounded by ``max_ctx``
    entries a row. ``max_ctx`` should be the step's own longest row (rounded
    up), not a model-wide bound: a static bound far above the rows' lengths
    makes the walk and select size for it."""
    for ctx, dense_rows, source_rows in DENSE_PREFERRED:
        if max_ctx <= ctx:
            least = source_rows if source else dense_rows
            return least is not None and rows >= least
    return False


def _ragged_args(ragged: Mxfp4IndexerRagged | None, rows: int, device):
    if ragged is None:
        return (0, 0, 0, 0, 0, 0), 0
    r = ragged
    if r.slot_block_size not in (64, 128, 256):
        raise ValueError(f"slot_block_size {r.slot_block_size} (64, 128 or 256)")
    for name in ("slots", "indptr", "lens", "is_valid", "slot_block_table"):
        t = getattr(r, name)
        if t.dtype != torch.int32 or t.device != device:
            raise ValueError(f"ragged.{name} must be int32 on {device}")
    if r.slots.numel() < rows * TOPK or not r.slots.is_contiguous():
        raise ValueError("ragged.slots must be contiguous with rows * 512 entries")
    if r.indptr.shape != (rows + 1,) or r.lens.shape != (rows,):
        raise ValueError("ragged.indptr must be [rows + 1], ragged.lens [rows]")
    if r.is_valid.shape != (rows,):
        raise ValueError("ragged.is_valid must be [rows]")
    if r.slot_block_table.dim() != 2 or r.slot_block_table.stride(1) != 1:
        raise ValueError("ragged.slot_block_table must be 2D with unit inner stride")
    if rows > MXFP4_INDEXER_MAX_ROWS:
        raise ValueError(f"{rows} rows with ragged lists (at most 2048)")
    return (
        r.slots.data_ptr(),
        r.indptr.data_ptr(),
        r.lens.data_ptr(),
        r.is_valid.data_ptr(),
        r.slot_block_table.data_ptr(),
        r.slot_block_table.stride(0),
    ), r.slot_block_size.bit_length() - 1


def _slot_rows(rows: int) -> int:
    return 1024 * -(-rows // 1024)  # a multiple of every select's threads


def _groups(reqs: int, next_n: int) -> int:
    """Workgroups of the select: a request's rows on one XCD (workgroup w runs
    on XCD w % 8), requests round robin."""
    return 8 * next_n * (-(-reqs // 8))


def _tokens_tier(max_ctx: int) -> int | None:
    """The token-logit tier a dense call of ``max_ctx`` entries uses (None:
    the block path)."""
    return next((t for t in sorted(TOKEN_TIERS) if t >= max_ctx), None)


def _blocks_tier(max_ctx: int) -> int:
    need = -(-max_ctx // BLOCK)
    return next(t for t in TIERS if t >= need)


def _run(exe, device: torch.device, *args):
    # One compiled handle per device: a handle does not launch on another one.
    _run_compiled(exe, *args, specialization_key=device.index)


@functools.cache
def _scores_exe(next_n: int, page_entries: int, tokens: bool = False):
    return compile_block_scores(next_n, page_entries, tokens)


@functools.cache
def _tokens_exe(
    next_n: int,
    tokens: int,
    slot_log2: int,
    slot_rows: int,
    emit_candidates: bool = False,
    gathered: bool = False,
    max_position: int = SHORT_ENTRIES,
    min_end: int = 0,
):
    return compile_tokens_select(
        next_n,
        tokens,
        TOKEN_TIERS[tokens],
        slot_log2,
        slot_rows,
        emit_candidates,
        gathered,
        max_position,
        min_end,
    )


@functools.cache
def _blocks_exe(mode: str, k: int, newest: bool, min_end: int, merged: int = 4096):
    return compile_block_select(mode, k, newest, min_end, merged)


@functools.cache
def _cand_logits_exe(
    next_n: int, page_entries: int, min_end: int = 0, cand_blocks: int = CANDIDATES
):
    return compile_candidate_logits(
        next_n, page_entries, CAND_TPS, min_end, cand_blocks
    )


@functools.cache
def _select_exe(
    next_n: int,
    page_entries: int,
    blocks: int,
    gather: bool,
    emit_candidates: bool = False,
    slot_log2: int = 0,
    slot_rows: int = SELECT_THREADS,
    skip_long: bool = False,
):
    return compile_select(
        next_n,
        page_entries,
        blocks,
        gather,
        GATHER_SELECT_THREADS if gather else SELECT_THREADS,
        emit_candidates=emit_candidates,
        slot_log2=slot_log2,
        slot_rows=slot_rows,
        skip_long=skip_long,
    )


# Scratch. A call carves its buffers (``_plan``) from ``workspace`` when given,
# else from one shared buffer per device that grows to the largest call seen:
# every call on a device uses the same memory, so calls that may overlap (on
# two streams at the same time) must each pass their own ``workspace``. A
# buffer a CUDA graph captured is kept alive when a larger call replaces it.
_WS_ALIGN = 256
_SCRATCH: dict[int, torch.Tensor] = {}
_SCRATCH_CAPTURED: set[int] = set()  # devices whose scratch a graph captured
_RETIRED: list[torch.Tensor] = []


def _plan(rows: int, max_ctx: int, layer: str) -> list[tuple[int, int, torch.dtype]]:
    """The scratch buffers ([rows, width] of dtype, in order) of a ``layer``
    call: ``dense``, ``source`` (dense with ``candidates_out``) or
    ``consumer``."""
    f32, i32 = torch.float32, torch.int32
    if layer == "consumer":
        if not _candidate_split(rows, max_ctx):
            return []
        return [(rows, CANDIDATES * BLOCK + 64, f32)]
    ntok = _tokens_tier(max_ctx)
    if ntok is not None:
        return [(rows, ntok + 64, f32)]
    if max_ctx <= BLOCK_CTX_MAX:
        return [(rows, _blocks_tier(max_ctx) + 64, f32)]
    newest = layer == "source"
    cand_blocks = CANDIDATES if newest else TOPK
    plan = [
        (rows, _blocks_tier(max_ctx) + 64, f32),  # block maxima
        (rows, _merged(max_ctx, newest), i32),  # compact ids
        (rows, _merged(max_ctx, newest), f32),  # compact maxima
        (rows, cand_blocks * BLOCK + 64, f32),  # candidate token logits
    ]
    if not newest:
        plan.append((rows, CANDIDATES, i32))  # the best blocks
    return plan


def _nbytes(plan) -> int:
    total = 0
    for rows, width, dtype in plan:
        n = rows * width * dtype.itemsize
        total += -(-n // _WS_ALIGN) * _WS_ALIGN
    return total


def mxfp4_indexer_workspace_bytes(rows: int, max_ctx: int, layer: str = "dense") -> int:
    """Bytes of ``workspace`` a call of ``rows`` decode rows bounded by
    ``max_ctx`` entries needs: ``layer`` ``dense`` (``flydsl_mxfp4_indexer_topk``),
    ``source`` (the same with ``candidates_out``) or ``consumer``
    (``flydsl_mxfp4_indexer_candidate_topk``). 0 when it needs none."""
    if layer not in ("dense", "source", "consumer"):
        raise ValueError(f"layer {layer!r} (dense, source or consumer)")
    return _nbytes(_plan(rows, max_ctx, layer))


def _scratch(device: torch.device, nbytes: int) -> torch.Tensor:
    """The device's shared scratch, at least ``nbytes`` bytes."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    capturing = torch.cuda.is_current_stream_capturing()
    buf = _SCRATCH.get(idx)
    if buf is None or buf.numel() < nbytes:
        if capturing:
            raise RuntimeError(
                "mxfp4 indexer: run a call of this size eagerly before graph "
                "capture (or pass workspace=)"
            )
        if buf is not None:
            # Rare (the scratch only grows) and never inside a capture: let
            # every earlier call on any stream finish before the memory goes.
            torch.cuda.synchronize(device)
            if idx in _SCRATCH_CAPTURED:
                _RETIRED.append(buf)  # a captured graph still addresses it
        _SCRATCH_CAPTURED.discard(idx)
        buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
        _SCRATCH[idx] = buf
    if capturing:
        _SCRATCH_CAPTURED.add(idx)
    return buf


def _carve(plan, workspace: torch.Tensor | None, device) -> list[torch.Tensor]:
    """The plan's buffers from ``workspace`` (or the shared scratch)."""
    nbytes = _nbytes(plan)
    if nbytes == 0:
        return []
    if workspace is None:
        ws = _scratch(device, nbytes)
    else:
        if (
            workspace.device != device
            or not workspace.is_contiguous()
            or workspace.numel() * workspace.element_size() < nbytes
            or workspace.data_ptr() % 16
        ):
            raise ValueError(
                f"workspace must be a contiguous, 16-byte aligned tensor of >= "
                f"{nbytes} bytes on {device} (mxfp4_indexer_workspace_bytes)"
            )
        ws = workspace.view(-1).view(torch.uint8)
    out, off = [], 0
    for rows, width, dtype in plan:
        n = rows * width * dtype.itemsize
        out.append(ws[off : off + n].view(dtype).view(rows, width))
        off += -(-n // _WS_ALIGN) * _WS_ALIGN
    return out


@functools.cache
def _aux_stream(device: torch.device) -> torch.cuda.Stream:
    return torch.cuda.Stream(device)


def _merged(max_ctx: int, newest: bool) -> int:
    """Entries of the long path's compact lists a row."""
    segments = max(1, _blocks_tier(max_ctx) // SEGB)
    return max(4096, segments * (SOURCE_SLOTS if newest else PLAIN_SLOTS))


def _kv_pages(kv_cache: torch.Tensor) -> torch.Tensor:
    """The cache as [pages, page_entries, 68]; ``paged_mxfp4_mqa_logits``'s
    [pages, page_entries, 1, 68] view is accepted too."""
    if kv_cache.dim() == 4:
        if kv_cache.shape[2] != 1:
            raise ValueError("kv_cache must be [pages, P, 68] or [pages, P, 1, 68]")
        kv_cache = kv_cache.squeeze(2)
    if kv_cache.dim() != 3 or kv_cache.dtype != torch.uint8:
        raise ValueError("kv_cache must be uint8 [pages, P, 68]")
    if kv_cache.shape[-1] != ENTRY_BYTES:
        raise ValueError(f"kv_cache entries must be {ENTRY_BYTES} bytes")
    if kv_cache.stride(-1) != 1 or kv_cache.stride(-2) != ENTRY_BYTES:
        raise ValueError("kv_cache pages must be contiguous")
    if kv_cache.stride(0) % 16:
        raise ValueError("kv_cache page stride must be a multiple of 16 bytes")
    return kv_cache


def _check(q, q_scale, weights, kv_cache, block_table, row_ends, out, next_n, max_ctx):
    rows = q.shape[0]
    reason = mxfp4_indexer_unsupported_reason(kv_cache.shape[1], max_ctx, next_n)
    if reason is not None:
        raise ValueError(f"mxfp4 indexer: {reason}")
    dev = q.device
    if dev.type != "cuda":
        raise ValueError("q must be a GPU tensor")
    for name, t in (
        ("q_scale", q_scale),
        ("weights", weights),
        ("kv_cache", kv_cache),
        ("block_table", block_table),
        ("row_ends", row_ends),
        ("out", out),
    ):
        if t is not None and t.device != dev:
            raise ValueError(f"{name} is on {t.device}, q on {dev}")
    if q.shape != (rows, H, D // 2) or q.dtype != torch.uint8 or not q.is_contiguous():
        raise ValueError("q must be contiguous uint8 [rows, 32, 64]")
    if not q_scale.is_contiguous() or (
        (q_scale.dtype, tuple(q_scale.shape))
        not in ((torch.uint8, (rows, H, D // 32)), (torch.int32, (rows, H)))
    ):
        raise ValueError(
            "q_scale must be contiguous uint8 [rows, 32, 4] or int32 [rows, 32]"
        )
    if (
        weights.shape != (rows, H)
        or weights.dtype != torch.float32
        or not weights.is_contiguous()
    ):
        raise ValueError("weights must be contiguous fp32 [rows, 32]")
    if block_table.dtype != torch.int32 or block_table.stride(1) != 1:
        raise ValueError("block_table must be int32 with unit inner stride")
    if rows != block_table.shape[0] * next_n:
        raise ValueError(f"rows {rows} != block_table rows x next_n")
    if (
        row_ends.shape != (rows,)
        or row_ends.dtype != torch.int32
        or not row_ends.is_contiguous()
    ):
        raise ValueError("row_ends must be contiguous int32 [rows]")
    if out is not None and (
        out.shape != (rows, TOPK) or out.dtype != torch.int32 or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous int32 [rows, 512]")
    return rows


def _check_candidates(candidates: torch.Tensor, rows: int, device) -> None:
    if (
        candidates.shape != (rows, CANDIDATES)
        or candidates.dtype != torch.int32
        or not candidates.is_contiguous()
        or candidates.device != device
    ):
        raise ValueError("candidates must be contiguous int32 [rows, 2048]")


def flydsl_mxfp4_indexer_block_scores(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    next_n: int,
    max_ctx: int,
    scores: torch.Tensor | None = None,
    tokens: bool = False,
) -> torch.Tensor:
    """The maximum logit of each 8-token block of every decode row (``tokens``:
    every token's logit, ``max_ctx`` <= 16384).

    Returns ``scores`` [rows, tier + 64] fp32: block j of a row in column j, a
    block past the row's end -inf, columns past the row's blocks unspecified.
    Allocated when not given.
    """
    kv_cache = _kv_pages(kv_cache)
    rows = _check(
        q, q_scale, weights, kv_cache, block_table, row_ends, None, next_n, max_ctx
    )
    if scores is None:
        nb = _tokens_tier(max_ctx) if tokens else _blocks_tier(max_ctx)
        if nb is None:
            raise ValueError("tokens=True serves max_ctx <= 16384")
        scores = torch.empty(rows, nb + 64, dtype=torch.float32, device=q.device)
    if rows == 0:
        return scores
    return _block_scores(
        q,
        q_scale,
        weights,
        kv_cache,
        block_table,
        row_ends,
        next_n,
        max_ctx,
        scores,
        tokens,
    )


def _block_scores(
    q,
    q_scale,
    weights,
    kv_cache,
    block_table,
    row_ends,
    next_n,
    max_ctx,
    scores,
    tokens=False,
):
    rows = q.shape[0]
    P = kv_cache.shape[1]
    nb = _tokens_tier(max_ctx) if tokens else _blocks_tier(max_ctx)
    if (
        scores.shape[0] < rows
        or scores.shape[1] < nb + 64
        or scores.dtype != torch.float32
        or scores.stride(1) != 1
        or scores.device != q.device
    ):
        raise ValueError(f"scores must be fp32 [>= {rows}, >= {nb + 64}]")
    tiles = -(-max_ctx // 32)
    reqs = rows // next_n
    nseg = max(-(-tiles // 128), min(tiles // 8, SEG_GRID // max(reqs, 1))) or 1
    if tiles > LONG_TILES:
        # A long request among short ones: finer segments balance the walk.
        nseg = max(nseg, -(-tiles // 64))
    _run(
        _scores_exe(next_n, P, tokens),
        q.device,
        q.data_ptr(),
        q_scale.data_ptr(),
        weights.data_ptr(),
        kv_cache.data_ptr(),
        block_table.data_ptr(),
        row_ends.data_ptr(),
        scores.data_ptr(),
        kv_cache.stride(0),
        block_table.stride(0),
        scores.stride(0),
        nseg,
        reqs,
        torch.cuda.current_stream(q.device),
    )
    return scores


def flydsl_mxfp4_indexer_topk(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    row_ends: torch.Tensor,
    out: torch.Tensor,
    *,
    next_n: int,
    max_ctx: int,
    candidates_out: torch.Tensor | None = None,
    ragged: Mxfp4IndexerRagged | None = None,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """The 512 best keys of every decode row of a dense indexer layer.

    Args:
        q: [rows, 32, 64] uint8, e2m1 pairs (low nibble first); row ``r`` is
            row ``r % next_n`` of request ``r // next_n``.
        q_scale: [rows, 32, 4] uint8 e8m0 (or [rows, 32] int32 of the same
            bytes).
        weights: [rows, 32] fp32, the softmax and head scales folded in.
        kv_cache: [pages, P, 68] (or [pages, P, 1, 68]) uint8, P = 64 or 128,
            the preshuffled MXFP4 cache ``paged_mxfp4_mqa_logits`` reads.
            Pages need not be adjacent (stride(0) a multiple of 16).
        block_table: [requests, max_pages] int32.
        row_ends: [rows] int32, the exclusive key bound of each row.
        out: [rows, 512] int32, written: positions ascending, -1 padded.
        next_n: rows per request (1-8).
        max_ctx: static upper bound of a row's entries (at most 1048576); a
            row longer than its tier is truncated.
        candidates_out: [rows, 2048] int32, optional: also the row's candidate
            8-token blocks (the newest key's block and the 2047 best others by
            block maximum), ids ascending, -1 padded.
        ragged: optional, also the sparse attention's ragged slot lists.
        workspace: optional scratch, a contiguous tensor of at least
            ``mxfp4_indexer_workspace_bytes(rows, max_ctx, layer)`` bytes
            (``layer`` "source" with ``candidates_out``, else "dense") on the
            device; default: the device's shared scratch.

    Returns:
        ``out``.
    """
    kv_cache = _kv_pages(kv_cache)
    rows = _check(
        q, q_scale, weights, kv_cache, block_table, row_ends, out, next_n, max_ctx
    )
    dev = q.device
    if candidates_out is not None:
        _check_candidates(candidates_out, rows, dev)
    rargs, slot_log2 = _ragged_args(ragged, rows, dev)
    if rows == 0:
        return out
    bufs = _carve(
        _plan(rows, max_ctx, "dense" if candidates_out is None else "source"),
        workspace,
        dev,
    )
    P = kv_cache.shape[1]
    reqs = rows // next_n
    stream = torch.cuda.current_stream(dev)
    ntok = _tokens_tier(max_ctx)
    if ntok is not None:
        # Short rows: the walk writes the token logits, a workgroup a row
        # selects.
        logits = _block_scores(
            q,
            q_scale,
            weights,
            kv_cache,
            block_table,
            row_ends,
            next_n,
            max_ctx,
            bufs[0],
            tokens=True,
        )
        _run(
            _tokens_exe(
                next_n, ntok, slot_log2, _slot_rows(rows), candidates_out is not None
            ),
            dev,
            logits.data_ptr(),
            row_ends.data_ptr(),
            out.data_ptr(),
            logits.stride(0),
            reqs,
            *rargs,
            0 if candidates_out is None else candidates_out.data_ptr(),
            rows,
            stream,
        )
        return out
    long = max_ctx > BLOCK_CTX_MAX
    scores = _block_scores(
        q, q_scale, weights, kv_cache, block_table, row_ends, next_n, max_ctx, bufs[0]
    )
    nb = _blocks_tier(max_ctx)
    if long:
        nb = _blocks_tier(BLOCK_CTX_MAX)
        # The long rows' chain runs beside the block select of the short rows.
        aux = _aux_stream(dev)
        aux.wait_stream(stream)
        with torch.cuda.stream(aux):
            _topk_long(
                q,
                q_scale,
                weights,
                kv_cache,
                block_table,
                row_ends,
                out,
                next_n,
                max_ctx,
                candidates_out,
                rargs,
                slot_log2,
                rows,
                scores,
                bufs[1:],
            )
    _run(
        _select_exe(
            next_n,
            P,
            nb,
            False,
            candidates_out is not None,
            slot_log2,
            _slot_rows(rows),
            long,
        ),
        dev,
        scores.data_ptr(),
        0,
        0 if candidates_out is None else candidates_out.data_ptr(),
        q.data_ptr(),
        q_scale.data_ptr(),
        weights.data_ptr(),
        kv_cache.data_ptr(),
        block_table.data_ptr(),
        row_ends.data_ptr(),
        out.data_ptr(),
        kv_cache.stride(0),
        block_table.stride(0),
        scores.stride(0),
        reqs,
        *rargs,
        _groups(reqs, next_n),
        stream,
    )
    if long:
        stream.wait_stream(aux)
    return out


def _topk_long(
    q,
    q_scale,
    weights,
    kv_cache,
    block_table,
    row_ends,
    out,
    next_n,
    max_ctx,
    candidates_out,
    rargs,
    slot_log2,
    rows,
    scores,
    bufs,
):
    """Rows above the block select's entries (it leaves them alone): the best
    blocks from the walk's block maxima (512, or the source layer's candidate
    list), then the candidate scorer and token select for those rows only.
    ``bufs``: the compact ids and maxima (every slot a merge reads is written
    by a segment), the candidate token logits and (plain) the best blocks."""
    dev = q.device
    newest = candidates_out is not None
    ids2, vals2, logits = bufs[:3]
    cand = candidates_out if newest else bufs[3]
    split_end = _blocks_tier(BLOCK_CTX_MAX) * BLOCK
    segments = max(1, _blocks_tier(max_ctx) // SEGB)
    slots = SOURCE_SLOTS if newest else PLAIN_SLOTS
    merged = _merged(max_ctx, newest)
    stream = torch.cuda.current_stream(dev)
    _run(
        _blocks_exe("seg", CANDIDATES - 1 if newest else TOPK, newest, split_end),
        dev,
        scores.data_ptr(),
        ids2.data_ptr(),
        vals2.data_ptr(),
        row_ends.data_ptr(),
        0,
        scores.stride(0),
        merged,
        0,
        rows,
        segments,
        stream,
    )
    _run(
        _blocks_exe("merge", CANDIDATES if newest else TOPK, newest, split_end, merged),
        dev,
        vals2.data_ptr(),
        ids2.data_ptr(),
        0,
        row_ends.data_ptr(),
        cand.data_ptr(),
        merged,
        merged,
        segments * slots,
        rows,
        1,
        stream,
    )
    _select_from_candidates(
        q,
        q_scale,
        weights,
        kv_cache,
        block_table,
        row_ends,
        cand,
        out,
        next_n,
        rows,
        rargs,
        slot_log2,
        max_ctx,
        logits,
        split_end,
        CANDIDATES if newest else TOPK,
    )


def _select_from_candidates(
    q,
    q_scale,
    weights,
    kv_cache,
    block_table,
    row_ends,
    candidates,
    out,
    next_n,
    rows,
    rargs,
    slot_log2,
    max_ctx,
    logits,
    min_end=0,
    cand_blocks=CANDIDATES,
):
    """The 512 best tokens among the candidate blocks' tokens: scored on (row,
    segment) workgroups into ``logits`` [rows, cand_blocks * 8 + 64], then the
    gathered token select; rows of at most ``min_end`` entries are left
    alone."""
    dev = q.device
    stream = torch.cuda.current_stream(dev)
    _run(
        _cand_logits_exe(next_n, kv_cache.shape[1], min_end, cand_blocks),
        dev,
        q.data_ptr(),
        q_scale.data_ptr(),
        weights.data_ptr(),
        kv_cache.data_ptr(),
        block_table.data_ptr(),
        row_ends.data_ptr(),
        candidates.data_ptr(),
        logits.data_ptr(),
        kv_cache.stride(0),
        block_table.stride(0),
        logits.stride(0),
        rows,
        stream,
    )
    _run(
        _tokens_exe(
            next_n,
            cand_blocks * BLOCK,
            slot_log2,
            _slot_rows(rows),
            False,
            True,
            MXFP4_INDEXER_MAX_ENTRIES if max_ctx > SHORT_ENTRIES else SHORT_ENTRIES,
            min_end,
        ),
        dev,
        logits.data_ptr(),
        row_ends.data_ptr(),
        out.data_ptr(),
        logits.stride(0),
        rows // next_n,
        *rargs,
        candidates.data_ptr(),
        rows,
        stream,
    )


def _candidate_split(rows: int, max_ctx: int) -> bool:
    """Score the candidates on (row, segment) workgroups and select from the
    token logits, rather than one workgroup a row."""
    if max_ctx > BLOCK_CTX_MAX:  # the gather select's tables do not fit
        return True
    return rows <= CAND_SPLIT_ROWS and max_ctx >= (
        CAND_SPLIT_CTX0 + CAND_SPLIT_CTX_ROW * rows
    )


def flydsl_mxfp4_indexer_candidate_topk(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    row_ends: torch.Tensor,
    candidates: torch.Tensor,
    out: torch.Tensor,
    *,
    next_n: int,
    max_ctx: int,
    ragged: Mxfp4IndexerRagged | None = None,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """The 512 best keys among the tokens of each row's candidate blocks.

    ``candidates`` [rows, 2048] int32: 8-token block ids ascending, -1 padded
    (what ``flydsl_mxfp4_indexer_topk(candidates_out=...)`` writes). Other
    arguments and ``out`` as ``flydsl_mxfp4_indexer_topk`` (``workspace``: at
    least ``mxfp4_indexer_workspace_bytes(rows, max_ctx, "consumer")`` bytes).
    """
    kv_cache = _kv_pages(kv_cache)
    rows = _check(
        q, q_scale, weights, kv_cache, block_table, row_ends, out, next_n, max_ctx
    )
    dev = q.device
    _check_candidates(candidates, rows, dev)
    rargs, slot_log2 = _ragged_args(ragged, rows, dev)
    if rows == 0:
        return out
    if _candidate_split(rows, max_ctx):
        (logits,) = _carve(_plan(rows, max_ctx, "consumer"), workspace, dev)
        _select_from_candidates(
            q,
            q_scale,
            weights,
            kv_cache,
            block_table,
            row_ends,
            candidates,
            out,
            next_n,
            rows,
            rargs,
            slot_log2,
            max_ctx,
            logits,
        )
        return out
    _run(
        _select_exe(
            next_n,
            kv_cache.shape[1],
            _blocks_tier(max_ctx),
            True,
            False,
            slot_log2,
            _slot_rows(rows),
        ),
        dev,
        0,
        candidates.data_ptr(),
        0,
        q.data_ptr(),
        q_scale.data_ptr(),
        weights.data_ptr(),
        kv_cache.data_ptr(),
        block_table.data_ptr(),
        row_ends.data_ptr(),
        out.data_ptr(),
        kv_cache.stride(0),
        block_table.stride(0),
        0,
        rows // next_n,
        *rargs,
        _groups(rows // next_n, next_n),
        torch.cuda.current_stream(dev),
    )
    return out
