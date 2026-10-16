# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL QSA public surface (SILOTIGER-1047).

``qsa_indexer_block_ids`` writes indexer ``block_ids [M, 512]`` from paged
compressed K. Rows up to 512 blocks use the fused emit kernel; longer rows
use tiled BLOCK_N=32 BF16 MFMA scoring plus the stable decode radix below
32768 columns and streaming radix at or above that width. Single-request
prefill scores 16 query rows per workgroup. The indexer head count is 4 or
8, each a separate compile.

``qsa_sparse_gqa`` writes sparse GQA ``o [M, Hq, D]`` from paged K/V at the
selected token ids. Decode runs a BLOCK_N=16 two-wave tile with split-K and
an LSE merge; prefill runs BLOCK_N=32 two waves and writes output directly
once the grid alone fills the machine. Softmax is online in log2 space and the
split
partials are FP32. Expand+tail and the sigmoid gate stay unfused.

The fp32 reference, the paged-cache fixtures, and the concrete shapes
live in ``op_tests/qsa_ref.py``.

``qsa_layer`` is the ``qwen4_exp`` opt-in. It runs the indexer, the vendored
expand+tail, and sparse GQA. A rank-3 query pair missing from the measured
table logs once. A query that is not rank 3 is not a table miss; the indexer
or sparse GQA reports it. Sigmoid and partial RoPE stay outside the layer.
"""

import functools

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.kernels_common import LOG2E, get_warp_size
from .kernels.qsa.indexer_select import (
    _D,
    _DECODE_MAX_ROWS,
    _H,
    _K,
    _KV_HEADS,
    _R,
    _SCORE_HEADS,
    _SCORE_SCALE,
    _STREAM_SELECT_MIN_COLUMNS,
    build_qsa_indexer_active_columns_module,
    build_qsa_indexer_decode_scores_module,
    build_qsa_indexer_emit_module,
    build_qsa_indexer_prefill_scores_module,
)
from .kernels.qsa.sparse_gqa import (
    _k2_coverage_reason,
    _launch_config,
    build_qsa_sparse_gqa_module,
)
from .kernels.tensor_shim import _run_compiled
from .kernels.topk.topk_per_row_decode_persistent import (
    build_topk_per_row_decode_one_workgroup_module,
)
from .topk.topk_per_row import (
    _ONE_WORKGROUP_MAX_ROW_WIDTH,
    flydsl_top_k_per_row_decode,
)

_SUPPORTED_ARCHS = ("gfx942", "gfx950")


def _qsa_arch() -> str:
    """Live GPU arch, or raise when it is not gfx942 or gfx950.

    ``get_gfx_runtime`` ignores ``GPU_ARCHS`` and returns the bare token.
    """
    arch = get_gfx_runtime()
    if arch not in _SUPPORTED_ARCHS:
        raise ValueError(f"QSA supports {', '.join(_SUPPORTED_ARCHS)}, got {arch}")
    return arch


# GQA query ``(Hq, D)`` -> indexer head counts swept with it.
# An unswept rank-3 pair still runs; the table only feeds the warning.
# Sparse GQA does not read this table. Its grid is ``(rows, kv_heads, splits)``
# from ``_launch_config``, so a different KV-head count is a different grid.
# M past 512 reuses the M=512 single-split config.
_FLYDSL_DISPATCH_TABLE = {
    (24, 256): (4, 8),  # Flash-Next / qwen4_exp, TP1
    (12, 256): (4, 8),  # Flash-Next TP2: 12 query heads, 1 KV head
    (6, 256): (4, 8),  # Flash-Next TP4: 6 query heads, 1 KV head
    (3, 256): (4, 8),  # Flash-Next TP8: 3 query heads, 1 KV head
    (10, 128): (4, 8),
}

__all__ = [
    "qsa_indexer_block_ids",
    "qsa_layer",
    "qsa_sparse_gqa",
    "span_bytes",
]


def span_bytes(t: torch.Tensor) -> int:
    """Bytes from ``t``'s first element to one past its last, for any strides.

    An empty tensor spans nothing. A view with gaps counts those gaps, which
    is how a cache past 4 GiB selects the wide gather.
    """
    if t.numel() == 0:
        return 0
    last = sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
    return (last + 1) * t.element_size()


def _rank3_query_pair(
    q_indexer: torch.Tensor, q_gqa: torch.Tensor
) -> tuple[tuple[int, int], int] | None:
    """``((Hq, D), indexer heads)`` when both queries are rank 3.

    A scalar or any other rank is not a table key. The indexer and sparse GQA report that.
    """
    if q_indexer.dim() != 3 or q_gqa.dim() != 3:
        return None
    gqa_key = (int(q_gqa.shape[1]), int(q_gqa.shape[2]))
    return gqa_key, int(q_indexer.shape[1])


def _warn_unmeasured_query(q_indexer: torch.Tensor, q_gqa: torch.Tensor) -> None:
    """Log an unswept rank-3 pair. A bad rank stays quiet."""
    pair = _rank3_query_pair(q_indexer, q_gqa)
    if pair is None:
        return
    gqa_key, index_heads = pair
    allowed = _FLYDSL_DISPATCH_TABLE.get(gqa_key)
    if allowed is None or index_heads not in allowed:
        _log_unmeasured_gqa_query(gqa_key, index_heads)


@functools.cache
def _log_unmeasured_gqa_query(gqa_key: tuple[int, int], index_heads: int) -> None:
    """Log one table miss. The same pair does not log again."""
    from aiter import logger

    logger.warning(
        "QSA: query pair (GQA %s, indexer heads %s) is not in the measured table",
        gqa_key,
        index_heads,
    )


def _expand_block_ids(
    block_ids: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    indices: torch.Tensor | None,
) -> torch.Tensor:
    # Expand is the vendored Triton kernel. It is not fused into the indexer
    # or sparse GQA.
    from aiter.ops.triton.attention.qsa_expand import expand_qsa_block_indices

    return expand_qsa_block_indices(
        block_ids,
        query_positions,
        context_lens,
        token_to_req,
        compress_ratio,
        token_topk,
        out=indices,
    )


@functools.lru_cache(maxsize=8)
def _emit_plan(page_size: int):
    return build_qsa_indexer_emit_module(page_size)


@functools.lru_cache(maxsize=16)
def _scores_plan(
    page_size: int,
    use_k32: bool,
    block_n: int,
    n_heads: int = _H,
    wide_cache: bool = False,
    k_strides: tuple[int, int, int] | None = None,
):
    return build_qsa_indexer_decode_scores_module(
        page_size, use_k32, block_n, n_heads, wide_cache, k_strides
    )


@functools.lru_cache(maxsize=8)
def _prefill_scores_plan(
    page_size: int,
    use_k32: bool,
    n_heads: int = _H,
    wide_cache: bool = False,
    k_strides: tuple[int, int, int] | None = None,
):
    return build_qsa_indexer_prefill_scores_module(
        page_size, use_k32, n_heads, wide_cache, k_strides
    )


def _k1_prefill_lds_bytes(n_heads: int) -> int:
    """Bytes in the 16-row scorer's LDS tile.

    Q is ``16.H.128`` bf16, K is ``32.128`` bf16, the visibility vector is
    16 int32s, and C is ``H.2.2.64.4`` fp32. Each field is already a
    multiple of its 16-byte alignment, so nothing is inserted between them.
    """
    block_m = 16
    block_n = 32
    num_waves = 2
    n_subtiles = block_n // 16
    q_bytes = block_m * n_heads * _D * 2
    k_bytes = block_n * _D * 2
    visible_bytes = block_m * 4
    c_bytes = n_heads * n_subtiles * num_waves * 64 * 4 * 4
    return q_bytes + k_bytes + visible_bytes + c_bytes


def _k1_uses_prefill_scorer(
    n_requests: int, rows: int, n_heads: int, arch: str
) -> bool:
    """Whether this launch uses the 16-row scorer.

    gfx942 H=8 stays on the one-row scorer. That tile is 73920 bytes and
    gfx942 has 65536. gfx950 H=8 keeps the 16-row tile.
    """
    if n_requests != 1 or rows < 16:
        return False
    return not (arch.startswith("gfx942") and n_heads == 8)


@functools.lru_cache(maxsize=1)
def _active_columns_plan():
    return build_qsa_indexer_active_columns_module()


def _k1_max_visible_columns(
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    n_columns: int,
) -> int:
    """Widest visible column count, using the scorer's per-row formula.

    One device reduction and one 4-byte readback. Callers only pay it for
    a decode-sized batch whose allocation is wider than the one-workgroup
    selector, which is the padded ``max_model_len`` table. A packed table
    at or under that width already dispatches on its real column count,
    and a prefill keeps the allocation width.
    """
    rows = int(token_to_req.shape[0])
    n_req = int(context_lens.shape[0])
    if rows == 0 or n_req == 0 or n_columns <= 0:
        return 0
    scratch = torch.empty(1, dtype=torch.int32, device=token_to_req.device)
    _run_compiled(
        _active_columns_plan(),
        token_to_req,
        query_positions,
        context_lens,
        scratch,
        int(n_columns),
        n_req,
        rows,
        torch.cuda.current_stream(token_to_req.device),
    )
    return int(scratch.item())


def _check_k1_score_workspace(scores: torch.Tensor, rows: int, n_columns: int) -> None:
    """Reject a score buffer that is not fp32 ``[rows, n_columns]``.

    The column stride stays 1 so a narrowed prefix is contiguous in each
    row. The row stride is the caller's.
    """
    if scores.dtype != torch.float32 or tuple(scores.shape) != (rows, n_columns):
        raise ValueError(
            f"scores must be float32 [{rows}, {n_columns}], "
            f"got {scores.dtype} {tuple(scores.shape)}"
        )
    if scores.stride(-1) != 1:
        raise ValueError(
            f"scores needs a unit column stride, got {tuple(scores.stride())}"
        )


def qsa_indexer_score_and_select(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor,
    n_columns: int,
    score_scale: float,
    n_heads: int,
    live_columns: int | None = None,
    row_bounded: bool = False,
    scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score long rows into ``[M, n_columns]`` and write top-512 ids into ``out``.

    ``n_heads`` is 4 or 8, each a separate compile. One request with
    ``M >= 16`` uses the 16-row scorer, except gfx942 H=8, which stays
    on the one-row scorer. ``live_columns`` is
    the widest visible row. The score buffer stays the width of the
    table, and the selector sees that prefix. A padded table with at
    most 64 rows and at most 20000 live columns uses the one-workgroup
    decode radix. More rows keep the streaming selector. A packed table
    uses the stable decode radix below 32768 columns and streaming radix
    (``tie='low'``) at or above that. ``row_bounded``
    runs the one-workgroup decode radix on the full width instead: it
    stops at each row's ``row_lens``, so it needs no readback of the
    widest row.

    ``scores`` is an optional fp32 ``[M, n_columns]`` workspace with a
    unit column stride. ``None`` allocates it. The buffer is overwritten
    and is not returned.
    """
    if n_heads not in _SCORE_HEADS:
        raise ValueError(f"score heads must be {_SCORE_HEADS}, got {n_heads}")
    if q.shape[1] != n_heads:
        raise ValueError(f"q must have {n_heads} heads, got {tuple(q.shape)}")
    m = int(q.shape[0])
    n_columns = int(n_columns)
    if scores is not None:
        _check_k1_score_workspace(scores, m, n_columns)
        if scores.device != q.device:
            raise ValueError("scores must be on the same device as q")
    arch = _qsa_arch()
    use_k32 = arch == "gfx950"
    page_size = k_cache.shape[1]
    score_block_n = 32
    if scores is None:
        scores = torch.empty(m, n_columns, dtype=torch.float32, device=q.device)
    row_lens = torch.empty(m, dtype=torch.int32, device=q.device)
    score_tiles = (n_columns + score_block_n - 1) // score_block_n
    # numel() is the packed page. A vLLM layer view pads the page stride,
    # so the bytes the descriptor has to cover are the span.
    wide_cache = span_bytes(k_cache) > (1 << 32)
    k_strides = tuple(int(s) for s in k_cache.stride()[:3]) if wide_cache else None
    if _k1_uses_prefill_scorer(int(context_lens.shape[0]), m, n_heads, arch):
        _run_compiled(
            _prefill_scores_plan(page_size, use_k32, n_heads, wide_cache, k_strides),
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            context_lens,
            scores,
            row_lens,
            int(n_columns),
            m,
            float(score_scale),
            int(k_cache.shape[0]),
            (m + 15) // 16,
            score_tiles,
            torch.cuda.current_stream(q.device),
        )
    else:
        _run_compiled(
            _scores_plan(
                page_size, use_k32, score_block_n, n_heads, wide_cache, k_strides
            ),
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            context_lens,
            scores,
            row_lens,
            int(n_columns),
            int(context_lens.shape[0]),
            float(score_scale),
            int(k_cache.shape[0]),
            m,
            score_tiles,
            torch.cuda.current_stream(q.device),
        )
    if row_bounded:
        # row_starts is unread with packed_rows off; row_lens keeps the signature.
        _run_compiled(
            build_topk_per_row_decode_one_workgroup_module(
                _K,
                wave_size=get_warp_size(arch),
                write_values=False,
                compact=True,
            ),
            scores,
            row_lens,
            row_lens,
            out,
            scores,
            int(n_columns),
            1,
            scores.stride(0),
            m,
            torch.cuda.current_stream(q.device),
        )
        return out
    select_columns = n_columns if live_columns is None else int(live_columns)
    if select_columns < 1 or select_columns > n_columns:
        raise ValueError(
            f"live_columns must be in 1..{n_columns}, got {select_columns}"
        )
    # The prefix is contiguous in each row (stride 1). Narrowing it is what
    # lets a padded allocation take the short-row selector: that dispatch
    # reads the tensor width, not row_lens. One workgroup per row is the
    # decode shape (at most 64 rows). A prefill has too many rows for that
    # kernel, so it keeps the streaming selector, on the live prefix.
    scored = (
        scores if select_columns == n_columns else scores.narrow(1, 0, select_columns)
    )
    narrowed = select_columns < n_columns
    one_workgroup = (
        narrowed
        and select_columns <= _ONE_WORKGROUP_MAX_ROW_WIDTH
        and m <= _DECODE_MAX_ROWS
    )
    if one_workgroup or (not narrowed and select_columns < _STREAM_SELECT_MIN_COLUMNS):
        flydsl_top_k_per_row_decode(
            scored,
            1,
            row_lens,
            out,
            m,
            scored.stride(0),
            scored.stride(1),
            k=_K,
            stable=True,
        )
    else:
        # Import here so loading this module does not pull aiter.ops.enum.
        # That import calls get_module("module_aiter_core") and fails the
        # wheel's forked FlyDSL AOT workers.
        from aiter.ops.topk_select import topk_select

        topk_select(
            scored,
            _K,
            end=row_lens,
            output_idx=out,
            tie="low",
        )
    return out


def qsa_indexer_supported(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    heads: tuple[int, ...] = _SCORE_HEADS,
) -> str | None:
    """Why the indexer kernel does not support these tensors, or None if it does.

    ``heads`` narrows the accepted indexer head count; pass ``(4,)`` to
    reject the 8-head variant. Every other check is the same either way.
    """
    if not heads or any(h not in _SCORE_HEADS for h in heads):
        raise ValueError(f"heads must be a non-empty subset of {_SCORE_HEADS}")
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
        return f"q and k_cache must be bfloat16, got {q.dtype} and {k_cache.dtype}"
    if q.dim() != 3 or q.shape[1] not in heads or q.shape[2] != _D:
        allowed = "|".join(str(h) for h in heads)
        return f"q must be [M, {allowed}, {_D}], got {tuple(q.shape)}"
    if k_cache.dim() != 4:
        return f"k_cache must be [pages, page_size, H, D], got {tuple(k_cache.shape)}"
    if k_cache.shape[2] != _KV_HEADS or k_cache.shape[3] != _D:
        return f"k_cache KV/D must be ({_KV_HEADS}, {_D}), got {k_cache.shape[2:]}"
    if k_cache.shape[1] < 1:
        return f"page_size must be positive, got {int(k_cache.shape[1])}"
    # Read in place. vLLM's per-layer view keeps D contiguous and the token
    # stride at D; the page stride is padded. The gather is a 16-byte load,
    # so that stride has to be a multiple of 8 elements. A copy is not.
    if k_cache.stride(3) != 1:
        return f"k_cache needs a unit D stride, got {k_cache.stride()}"
    if any(s % 8 for s in k_cache.stride()[:3]):
        return (
            f"k_cache strides must be multiples of 8 elements, got {k_cache.stride()}"
        )
    if page_table.dim() != 2 or page_table.dtype != torch.int32:
        return (
            f"page_table must be int32 [n_req, n_pages], got {tuple(page_table.shape)}"
        )
    return None


def qsa_indexer_selection_supported(token_topk: int, compress_ratio: int) -> str | None:
    """Why this token budget is outside the baked K1 contract, or None.

    K1 writes ``[_K]`` block ids and divides positions by ``_R``. Expand
    then expects ``token_topk // compress_ratio`` columns, so a budget of
    1024 tokens at ratio 4 is rejected there, while 4096 tokens at ratio 8
    still has 512 columns and would be attended at the wrong positions.
    """
    if compress_ratio != _R or token_topk != _K * _R:
        return (
            f"FlyDSL K1 selects {_K} blocks at compress ratio {_R} "
            f"(token_topk={_K * _R}), got token_topk={token_topk} "
            f"compress_ratio={compress_ratio}"
        )
    return None


def qsa_indexer_block_ids(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor | None = None,
    score_scale: float = _SCORE_SCALE,
    heads: tuple[int, ...] = _SCORE_HEADS,
    scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Write indexer ``block_ids [M, 512]`` from paged compressed K.

    Rows no wider than 512 use the fused emit path. Longer rows materialize
    scores with a BLOCK_N=32 MFMA writer. ``scores`` is an optional fp32
    ``[M, n_columns]`` workspace for that writer, with a unit column
    stride. ``n_columns`` is ``page_table.shape[1] * page_size``. ``None``
    allocates it. Emit does not read it. The buffer is overwritten and is
    not returned. Single-request prefill batches 16 query rows, except
    gfx942 H=8; other inputs use one row per workgroup. Selection is the
    stable decode radix below 32768 columns
    and streaming radix (``tie='low'``) at or above that width.
    Expand+tail is still separate. An empty context, cache, or page table
    writes ``-1`` and does not launch. A page table with fewer request
    rows than ``context_lens`` is rejected.

    ``heads`` is the accepted head count. Pass ``(4,)`` to keep the contract
    narrow; the ``(4, 8)`` default also admits the 8-head indexer.
    """
    reason = qsa_indexer_supported(q, k_cache, page_table, heads)
    if reason is not None:
        raise ValueError(f"[FlyDSL qsa_indexer_block_ids] {reason}")
    m = q.shape[0]
    if token_to_req.shape != (m,) or token_to_req.dtype != torch.int32:
        raise ValueError(f"token_to_req must be int32 [{m}]")
    if query_positions.shape != (m,) or query_positions.dtype != torch.int32:
        raise ValueError(f"query_positions must be int32 [{m}]")
    if context_lens.dim() != 1 or context_lens.dtype != torch.int32:
        raise ValueError("context_lens must be 1-D int32")
    if context_lens.shape[0] > page_table.shape[0]:
        raise ValueError(
            f"page_table has {int(page_table.shape[0])} request rows for "
            f"{int(context_lens.shape[0])} context lengths"
        )
    if out is None:
        out = torch.empty(m, _K, dtype=torch.int32, device=q.device)
    elif out.shape != (m, _K) or out.dtype != torch.int32:
        raise ValueError(f"out must be int32 [{m}, {_K}], got {tuple(out.shape)}")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if scores is not None:
        _check_k1_score_workspace(
            scores, m, int(page_table.shape[1]) * int(k_cache.shape[1])
        )
    tensors = (q, k_cache, page_table, token_to_req, query_positions, context_lens, out)
    if scores is not None:
        tensors = tensors + (scores,)
    if any(not t.is_cuda for t in tensors):
        raise ValueError("every tensor must be on the GPU")
    if any(t.device != q.device for t in tensors[1:]):
        raise ValueError("every tensor must be on the same GPU")
    _qsa_arch()
    # The scorers clamp a missing request or page to 0 and still load it.
    # An empty axis has no such element, and every selection is -1.
    if (
        m == 0
        or context_lens.shape[0] == 0
        or k_cache.shape[0] == 0
        or page_table.shape[0] == 0
        or page_table.shape[1] == 0
    ):
        return out.fill_(-1)
    q = q.contiguous()
    # Leave k_cache strided. contiguous() would copy a vLLM layer view into
    # every captured decode graph.
    page_table = page_table.contiguous()
    token_to_req = token_to_req.contiguous()
    query_positions = query_positions.contiguous()
    context_lens = context_lens.contiguous()
    page_size = k_cache.shape[1]
    n_columns = page_table.shape[1] * page_size
    # A packed table at or under the one-workgroup cutoff already names its
    # real width, so emit and the selector see it without a readback. A
    # wider allocation is the padded max_model_len table. Decode (at most
    # 64 rows) reads the widest visible row and dispatches on that. A
    # prefill has too many rows for that readback to pay, and its selector
    # is already the streaming one, so it keeps the allocation width. So
    # does a graph capture: the readback is not allowed there, and the
    # graph would replay the width of the capture batch. Decode under
    # capture selects with the one-workgroup radix on the full width,
    # which stops at each row's visible length. A row longer than 20000
    # columns finishes the later radix passes from the first-pass
    # candidates when those fit in LDS.
    live_columns = n_columns
    padded_decode = n_columns > _ONE_WORKGROUP_MAX_ROW_WIDTH and m <= _DECODE_MAX_ROWS
    row_bounded = padded_decode and torch.cuda.is_current_stream_capturing()
    if padded_decode and not row_bounded:
        live_columns = _k1_max_visible_columns(
            token_to_req, query_positions, context_lens, n_columns
        )
    if live_columns <= _K:
        _run_compiled(
            _emit_plan(page_size),
            token_to_req,
            query_positions,
            context_lens,
            out,
            int(n_columns),
            int(context_lens.shape[0]),
            m,
            torch.cuda.current_stream(q.device),
        )
    else:
        qsa_indexer_score_and_select(
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            context_lens,
            out,
            int(n_columns),
            float(score_scale),
            int(q.shape[1]),
            live_columns=live_columns,
            row_bounded=row_bounded,
            scores=scores,
        )
    return out


@functools.lru_cache(maxsize=32)
def _plan(
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    page_size: int,
    use_k32: bool,
    block_n: int,
    block_threads: int,
    n_splits: int,
    wide_cache: bool,
    kv_strides: tuple[int, int, int] | None = None,
):
    return build_qsa_sparse_gqa_module(
        n_q_heads,
        n_kv_heads,
        head_dim,
        page_size,
        use_k32,
        block_n,
        block_threads,
        n_splits,
        wide_cache,
        kv_strides,
    )


def qsa_sparse_gqa_supported(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    page_table: torch.Tensor,
) -> str | None:
    """Why the sparse GQA kernel does not support these tensors, or None if it does."""
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
        return f"q and caches must be bfloat16, got {q.dtype} and {k_cache.dtype}"
    if v_cache.dtype != k_cache.dtype or v_cache.shape != k_cache.shape:
        return "v_cache must match k_cache dtype and shape"
    if q.dim() != 3:
        return f"q must be [M, Hq, D], got {tuple(q.shape)}"
    if k_cache.dim() != 4:
        return f"k_cache must be [pages, page_size, H, D], got {tuple(k_cache.shape)}"
    if k_cache.shape[3] != q.shape[2]:
        return f"k_cache D must be q's {q.shape[2]}, got {k_cache.shape[3]}"
    # The same geometries build_qsa_sparse_gqa_module rejects for the launch
    # _launch_config would compile from these tensors. A reason here keeps
    # the caller from launching a kernel that raises or drops a tail of D.
    # BLOCK_N and the workgroup still come from that policy.
    n_q_heads = int(q.shape[1])
    n_kv_heads = int(k_cache.shape[2])
    head_dim = int(q.shape[2])
    page_size = int(k_cache.shape[1])
    if n_kv_heads < 1 or n_q_heads % n_kv_heads:
        return f"{n_q_heads} query heads do not group over {n_kv_heads} KV heads"
    if n_q_heads // n_kv_heads > 16:
        return f"group must fit one MFMA M dim, got {n_q_heads // n_kv_heads}"
    if head_dim % 32:
        return f"head_dim must be a multiple of 32, got {head_dim}"
    if page_size < 1:
        return f"page_size must be positive, got {page_size}"
    # Paged caches are read in place. A K|V-interleaved view (vLLM's
    # [pages, page_size, H, 2 * D] split in two) is fine; a copy is not.
    if k_cache.stride(3) != 1 or v_cache.stride(3) != 1:
        return f"k_cache and v_cache need a unit D stride, got {k_cache.stride()}"
    if k_cache.stride() != v_cache.stride():
        return (
            f"k_cache and v_cache strides differ: {k_cache.stride()} vs "
            f"{v_cache.stride()}"
        )
    if any(s % 8 for s in k_cache.stride()[:3]):
        return (
            f"k_cache strides must be multiples of 8 elements, got {k_cache.stride()}"
        )
    if indices.dim() != 2 or indices.shape[0] != q.shape[0]:
        return "indices must be [M, W]"
    if indices.dtype != torch.int32:
        return f"indices must be int32, got {indices.dtype}"
    if page_table.dim() != 2 or page_table.dtype != torch.int32:
        return (
            f"page_table must be int32 [n_req, n_pages], got {tuple(page_table.shape)}"
        )
    block_n, block_threads, _ = _launch_config(
        int(q.shape[0]), int(indices.shape[1]), n_kv_heads, head_dim
    )
    return _k2_coverage_reason(head_dim, block_n, block_threads)


def _take_k2_workspace(workspace, n_splits, rows, n_q_heads, head_dim, device):
    """Return the caller pair, or raise if it cannot hold this split."""
    if not isinstance(workspace, tuple) or len(workspace) != 2:
        raise ValueError("workspace must be (partial_out, partial_lse)")
    partial_out, partial_lse = workspace
    out_shape = (n_splits, rows, n_q_heads, head_dim)
    lse_shape = (n_splits, rows, n_q_heads)
    if (
        not isinstance(partial_out, torch.Tensor)
        or partial_out.dtype != torch.float32
        or tuple(partial_out.shape) != out_shape
        or not partial_out.is_contiguous()
        or partial_out.device != device
    ):
        raise ValueError(
            f"partial_out must be contiguous float32 {out_shape} on {device}"
        )
    if (
        not isinstance(partial_lse, torch.Tensor)
        or partial_lse.dtype != torch.float32
        or tuple(partial_lse.shape) != lse_shape
        or not partial_lse.is_contiguous()
        or partial_lse.device != device
    ):
        raise ValueError(
            f"partial_lse must be contiguous float32 {lse_shape} on {device}"
        )
    return partial_out, partial_lse


def qsa_sparse_gqa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Write sparse GQA ``o [M, Hq, D]`` from paged K/V.

    ``workspace`` is ``(partial_out, partial_lse)`` for a split launch.
    ``None`` allocates that pair. A one-split launch writes ``out``
    directly and does not read ``workspace``.
    """
    reason = qsa_sparse_gqa_supported(q, k_cache, v_cache, indices, page_table)
    if reason is not None:
        raise ValueError(f"[FlyDSL qsa_sparse_gqa] {reason}")
    rows = q.shape[0]
    n_q_heads = int(q.shape[1])
    head_dim = int(q.shape[2])
    n_kv_heads = int(k_cache.shape[2])
    if token_to_req.shape != (rows,) or token_to_req.dtype != torch.int32:
        raise ValueError(f"token_to_req must be int32 [{rows}]")
    if out is None:
        # The plan cache keeps the first compile, and that compile bakes
        # the output stride. empty_like would copy a non-contiguous q.
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    elif out.shape != q.shape or out.dtype != q.dtype:
        raise ValueError(f"out must match q, got {tuple(out.shape)} {out.dtype}")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    tensors = (q, k_cache, v_cache, indices, page_table, token_to_req, out)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("every tensor must be on the GPU")
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("every tensor must be on the same GPU")
    arch = _qsa_arch()
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    if not rows or not indices.shape[1]:
        return out.zero_()
    # Nothing to gather. Clamping the index to 0 would still load, including
    # a table that has columns but no request row for safe_req.
    if k_cache.shape[0] == 0 or page_table.shape[0] == 0 or page_table.shape[1] == 0:
        return out.zero_()

    q = q.contiguous()
    indices = indices.contiguous()
    page_table = page_table.contiguous()
    token_to_req = token_to_req.contiguous()
    # Unsigned 32-bit V# span. Exactly 4 GiB still fits: the last byte
    # offset is 2^32 - 1. Python ints do not wrap. A strided view spans
    # more than its numel.
    wide_cache = span_bytes(k_cache) > (1 << 32)
    kv_strides = tuple(int(s) for s in k_cache.stride()[:3]) if wide_cache else None
    page_size = k_cache.shape[1]
    use_k32 = arch == "gfx950"
    n_sel = int(indices.shape[1])
    block_n, block_threads, n_splits = _launch_config(rows, n_sel, n_kv_heads, head_dim)
    if n_splits == 1:
        # The split kernel stores straight into out. A caller workspace
        # is not a second buffer on this path.
        partial_out = out
        partial_lse = out
    elif workspace is None:
        partial_out = torch.empty(
            (n_splits, rows, n_q_heads, head_dim), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (n_splits, rows, n_q_heads), dtype=torch.float32, device=q.device
        )
    else:
        partial_out, partial_lse = _take_k2_workspace(
            workspace, n_splits, rows, n_q_heads, head_dim, q.device
        )

    _run_compiled(
        _plan(
            n_q_heads,
            n_kv_heads,
            head_dim,
            page_size,
            use_k32,
            block_n,
            block_threads,
            n_splits,
            wide_cache,
            kv_strides,
        ),
        q,
        k_cache,
        v_cache,
        indices,
        page_table,
        token_to_req,
        partial_out,
        partial_lse,
        out,
        n_sel,
        int(page_table.shape[0]),
        int(page_table.shape[1]),
        int(k_cache.shape[0]),
        float(softmax_scale * LOG2E),
        int(rows),
        int(n_kv_heads),
        int(n_q_heads),
        torch.cuda.current_stream(q.device),
    )
    return out


def qsa_layer(
    q_indexer: torch.Tensor,
    index_k_cache: torch.Tensor,
    index_page_table: torch.Tensor,
    q_gqa: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    indices: torch.Tensor | None = None,
    block_ids: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    token_topk: int = 2048,
    compress_ratio: int = 4,
    score_scale: float | None = None,
    softmax_scale: float | None = None,
    scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one QSA layer: indexer select, expand+tail, sparse GQA.

    Returns the bf16 GQA tensor ``[M, Hq, D]``. A caller-provided ``out``
    is that tensor.

    The layer is K1, the vendored expand, and K2. It raises on a token
    budget other than 512 blocks at compress ratio 4. The live GPU must
    be gfx942 or gfx950. ``get_gfx_runtime`` decides that, and any other
    architecture raises. A rank-3 pair missing from the measured table
    logs once and still runs. A query that is not rank 3 does not log;
    K1 or K2 reports it. Indexer scores are multiplied by ``score_scale``
    and QK by ``softmax_scale``. ``None`` is ``head_dim**-0.5`` for that
    tensor.

    Args:
        q_indexer: bf16 ``[M, H, 128]``. FlyDSL accepts ``H`` of 4 or 8.
        index_k_cache: bf16 ``[pages, page_size, 1, 128]``. The D stride
            is 1. The page, token, and head strides are multiples of 8
            elements. The cache is read in place.
        index_page_table: int32 ``[n_req, n_pages]``.
        q_gqa: bf16 ``[M, Hq, D]``, on the same GPU as the other tensors.
        k_cache, v_cache: bf16 ``[pages, page_size, Hkv, D]``, matching
            each other in shape and stride, with a unit D stride and the
            same 8-element stride rule. ``Hq`` divides across ``Hkv`` in
            a group of at most 16. ``D`` is a multiple of 32 and fills
            the K2 gather, K LDS map, and PV tile.
        kv_page_table: int32 ``[n_req, n_pages]``.
        token_to_req, query_positions: int32 ``[M]``.
        context_lens: 1-D int32, with no more rows than ``index_page_table``.
        indices: optional int32 ``[M, token_topk + compress_ratio - 1]``
            expand buffer. ``None`` allocates it.
        block_ids: optional caller-owned contiguous int32 ``[M, 512]``.
            Each call overwrites it. A row is a packed set of unique
            compressed block ids, then ``-1`` through column 512. The
            order is not a score ranking, and an exact tie keeps the
            smaller index. ``-1`` is padding, including a row with no
            visible block. The incomplete tail is not stored here; expand
            writes those token ids into ``indices``.
        scores: optional fp32 ``[M, n_columns]`` workspace for the long-row
            scorer. ``n_columns`` is ``index_page_table.shape[1] *
            page_size``. The column stride is 1. ``None`` allocates it.
            The layer overwrites it and does not return it. Emit ignores
            a buffer of that shape.
        out: optional contiguous bf16 ``[M, Hq, D]`` matching ``q_gqa``.
            ``None`` allocates it.
        token_topk: selected tokens. The default is 2048, which is the
            FlyDSL budget of 512 blocks.
        compress_ratio: block size. The default, and the FlyDSL value, is 4.
        score_scale, softmax_scale: see above.

    A span past 4 GiB on a cache selects the wide gather compile. An empty
    query, cache, or page table writes ``-1`` block ids and a zero ``out``
    and does not launch. K2 gathers the token ids it is given and does not
    check that they are causal. It allocates its own split partials on a
    multi-split launch. The block ids are the selection for this call.
    Sigmoid and partial RoPE stay outside this call.

    Returns:
        Sparse GQA output, bf16 ``[M, Hq, D]``.
    """
    reason = qsa_indexer_selection_supported(token_topk, compress_ratio)
    if reason is not None:
        raise ValueError(reason)
    _warn_unmeasured_query(q_indexer, q_gqa)
    # A bad rank is not a table key and has no head or D to read here.
    extra = {}
    if q_indexer.dim() == 3:
        if score_scale is None:
            score_scale = float(q_indexer.shape[2]) ** -0.5
        extra["heads"] = (int(q_indexer.shape[1]),)
    if score_scale is not None:
        extra["score_scale"] = score_scale
    block_ids = qsa_indexer_block_ids(
        q_indexer,
        index_k_cache,
        index_page_table,
        token_to_req,
        query_positions,
        context_lens,
        out=block_ids,
        scores=scores,
        **extra,
    )
    indices = _expand_block_ids(
        block_ids,
        query_positions,
        context_lens,
        token_to_req,
        compress_ratio,
        token_topk,
        indices,
    )
    return qsa_sparse_gqa(
        q_gqa,
        k_cache,
        v_cache,
        indices,
        kv_page_table,
        token_to_req,
        out=out,
        softmax_scale=softmax_scale,
    )
