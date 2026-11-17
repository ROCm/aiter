# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Derived from tingqli/pyhip QSA (MIT), revision 8027faad29415747707670db7776528c6b72db81.
"""Workspace planning and selection for the gfx942 QSA indexer."""

import math
import threading
from collections import OrderedDict
from numbers import Integral

import numpy as np
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl import qsa_indexer_launch as launch
from aiter.ops.flydsl.kernels.qsa.sparse_prefill_prepare import upload

MAX_COMPRESSED_KEYS = 262144
_LOGITS_BUDGET_BYTES = 512 * 1024 * 1024
_CHUNK_ROWS = 1024
_BM, _KC, _PITCH = 128, 512, 544
_TOPK, _RATIO, _WIDTH, _PAD = 512, 4, 2051, 512
_SCALE = float(np.float32(1.0) / np.float32(math.sqrt(128)))


def _host_lengths(values, name):
    if isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a host sequence, not a tensor")
    values = tuple(values)
    if any(not isinstance(v, Integral) or isinstance(v, bool) or v < 0 for v in values):
        raise ValueError(f"{name} must contain nonnegative host integers")
    return tuple(int(v) for v in values)


class _Chunk:
    def __init__(self, row0, rows, width, items):
        self.row0, self.rows, self.width, self.items = row0, rows, width, items


class _Layout:
    """Host-derived per-forward plan; device tensors are read-only and shared."""

    def __init__(
        self, seq_lens, extend_lens, device, row_range=None, logits_workspace_bytes=None
    ):
        seq, extend = np.asarray(seq_lens, dtype=np.int64), np.asarray(
            extend_lens, dtype=np.int64
        )
        prefix, compressed = seq - extend, seq // _RATIO
        budget = (
            _LOGITS_BUDGET_BYTES
            if logits_workspace_bytes is None
            else max(1, logits_workspace_bytes - _PAD * 4)
        )
        min_rows = _CHUNK_ROWS if logits_workspace_bytes is None else 0
        tile_rows = _BM
        if logits_workspace_bytes is not None:
            row_bytes = max(1, -(-int(compressed.max()) // _KC)) * _PITCH * 4
            tile_rows = max(1, min(_BM, budget // row_bytes))
        row_base = np.cumsum(extend) - extend
        self.rows = int(extend.sum())
        sequence = np.repeat(np.arange(len(seq)), extend)
        info = np.stack(
            (np.arange(self.rows) - (row_base - prefix)[sequence], seq[sequence]),
            axis=1,
        )
        counts = -(-extend // tile_rows)
        owner = np.repeat(np.arange(len(seq)), counts)
        start = (
            np.arange(len(owner)) - np.repeat(np.cumsum(counts) - counts, counts)
        ) * tile_rows
        sizes = np.minimum(tile_rows, extend[owner] - start)
        firsts = row_base[owner] + start
        if row_range is not None:
            # Only the tiles' rows inside the range get logits and top-k; bounds follow each clipped tile.
            low, high = np.maximum(firsts, row_range[0]), np.minimum(
                firsts + sizes, row_range[1]
            )
            kept = low < high
            owner, start = owner[kept], (start + low - firsts)[kept]
            firsts, sizes = low[kept], (high - low)[kept]
        bounds = np.minimum(
            (prefix[owner] + start + sizes) // _RATIO, compressed[owner]
        )
        parts = [info.reshape(-1)]
        chunks, self.logits_elements = [], 0
        begin = 0
        while begin < len(owner):
            end, width = begin, 4
            while end < len(owner):
                grown = max(width, -(-int(bounds[end]) // 4) * 4)
                rows = int(firsts[end] + sizes[end] - firsts[begin])
                size = rows * -(-grown // _KC) * _PITCH * 4
                if end > begin and (
                    size >= 2**31
                    or size > budget
                    and firsts[end] - firsts[begin] >= min_rows
                ):
                    break
                width, end = grown, end + 1
            row0 = int(firsts[begin])
            rows = int(firsts[end - 1] + sizes[end - 1]) - row0
            keep = np.arange(begin, end)[bounds[begin:end] > _TOPK]
            steps = -(-bounds[keep] // _KC)
            tile = np.repeat(keep, steps)
            key = (
                np.arange(len(tile)) - np.repeat(np.cumsum(steps) - steps, steps)
            ) * _KC
            # Consecutive CTAs fill one key block's consecutive tiles: a sequential write stream.
            order = np.lexsort((tile, key))
            tile, key = tile[order], key[order]
            parts.append(
                np.stack(
                    (
                        firsts[tile],
                        firsts[tile] - row0,
                        sizes[tile],
                        owner[tile],
                        key,
                        np.minimum(key + _KC, bounds[tile]),
                    ),
                    axis=1,
                ).reshape(-1)
            )
            chunks.append((row0, rows, width, len(tile)))
            self.logits_elements = max(
                self.logits_elements, rows * -(-width // _KC) * _PITCH
            )
            begin = end
        # One pinned asynchronous upload: a new layout never drains the stream.
        data = upload(np.concatenate(parts), device)
        offsets = np.cumsum([0] + [len(p) for p in parts])
        views = [data[offsets[i] : offsets[i + 1]] for i in range(len(parts))]
        self.row_info = views[0].view(-1, 2)
        self.chunks = [
            _Chunk(row0, rows, width, view.view(count, 6))
            for (row0, rows, width, count), view in zip(chunks, views[1:])
        ]


_layouts = OrderedDict()
_captured_layouts = {}
_lock = threading.Lock()


def _layout(seq_lens, extend_lens, device, row_range=None, logits_workspace_bytes=None):
    key = (
        device,
        torch.cuda.current_stream(device).cuda_stream,
        seq_lens,
        extend_lens,
        row_range,
        logits_workspace_bytes,
    )
    with _lock:
        capturing = torch.cuda.is_current_stream_capturing()
        layout = _layouts.get(key, _captured_layouts.get(key))
        if layout is None:
            if capturing:
                raise RuntimeError(
                    "Warm the QSA prefill layout on this stream before capture"
                )
            layout = _Layout(
                seq_lens, extend_lens, device, row_range, logits_workspace_bytes
            )
        if capturing:
            # HIP graphs retain addresses, not the tensors behind a host plan.
            _captured_layouts[key] = layout
        _layouts[key] = layout
        _layouts.move_to_end(key)
        while len(_layouts) > 8:
            _layouts.popitem(last=False)
        return layout


def select_prefill(
    q,
    compressed,
    token_slot_table,
    logical_positions,
    layout,
    stats,
    out,
    *,
    paged=False,
):
    """Select prepared queries using the caller's immutable packed-row plan."""
    logits = torch.empty(
        max(layout.logits_elements, 1) + _PAD, dtype=torch.float32, device=q.device
    )
    for chunk in layout.chunks:
        view = logits[: chunk.rows * -(-chunk.width // _KC) * _PITCH]
        launch.prefill_logits(
            q,
            compressed,
            token_slot_table,
            view,
            chunk.items,
            chunk.rows,
            _PITCH,
            _SCALE,
            stats,
            paged=paged,
        )
        launch.prefill_topk(
            view,
            chunk.width,
            chunk.row0,
            chunk.rows,
            _PITCH,
            logical_positions,
            layout.row_info,
            out,
            stats,
        )
    return out


def select_decode(
    q,
    compressed,
    page_table,
    lengths,
    positions,
    seq_lens,
    *,
    stats=None,
    logits_out=None,
):
    """Select prepared queries with a page-table row per query token."""
    rows, width = q.shape[0], page_table.shape[1] * 16
    logits = logits_out
    if logits is None:
        logits = torch.empty(rows * width + _PAD, dtype=torch.float32, device=q.device)
        logits = logits[: rows * width].view(rows, width)
    out = torch.empty((rows, _WIDTH), dtype=torch.int32, device=q.device)
    launch.decode_logits(q, compressed, page_table, lengths, logits, _SCALE, stats)
    launch.decode_topk(
        logits, lengths, positions.contiguous(), seq_lens.contiguous(), out, stats
    )
    return out


def try_select_paged_tokens(
    q,
    compressed_k_cache,
    page_table,
    token_to_request,
    query_positions,
    context_lens,
    columns,
    out,
    *,
    query_lens=None,
    prefix_lens=None,
    logits_workspace_bytes=128 * 1024 * 1024,
):
    """Select prepared queries without changing raw/compressed model state.

    Packed prefill may supply host query/prefix lengths. Small query batches
    use device metadata and may be captured after warmup. Return None for
    the existing QSA implementation to handle unsupported layouts.
    """
    rows = q.shape[0]
    meta = (token_to_request, query_positions, context_lens)
    if (
        get_gfx_runtime() != "gfx942"
        or torch.compiler.is_compiling()
        or q.ndim != 3
        or q.shape[1:] != (4, 128)
        or q.dtype != torch.bfloat16
        or not q.is_contiguous()
        or not q.is_cuda
        or rows == 0
        or compressed_k_cache.ndim != 4
        or compressed_k_cache.shape[1:] != (16, 1, 128)
        or compressed_k_cache.dtype != torch.bfloat16
        or not compressed_k_cache.is_contiguous()
        or compressed_k_cache.numel() * 2 >= 2**31
        or page_table.ndim != 2
        or page_table.shape[0] == 0
        or page_table.dtype != torch.int32
        or page_table.stride(1) != 1
        or columns % 16
        or not 512 <= columns <= MAX_COMPRESSED_KEYS
        or page_table.shape[1] * 16 < columns
        or any(
            t.ndim != 1
            or t.dtype not in (torch.int32, torch.int64)
            or not t.is_contiguous()
            for t in meta
        )
        or token_to_request.shape != (rows,)
        or query_positions.shape != (rows,)
        or context_lens.shape != (page_table.shape[0],)
        or out.shape != (rows, _WIDTH)
        or out.dtype != torch.int32
        or not out.is_contiguous()
        or any(
            t.device != q.device for t in (compressed_k_cache, page_table, *meta, out)
        )
    ):
        return None
    lo, hi = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
    for tensor in (q, compressed_k_cache, page_table, *meta):
        size = 1 + sum(
            (n - 1) * stride for n, stride in zip(tensor.shape, tensor.stride())
        )
        first = tensor.data_ptr()
        if lo < first + size * tensor.element_size() and first < hi:
            raise ValueError("QSA selection out must not overlap its inputs")
    table = page_table[:, : columns // 16]
    compressed = compressed_k_cache.view(-1, 1, 128)
    if query_lens is not None and prefix_lens is not None:
        queries = _host_lengths(query_lens, "query_lens")
        prefixes = _host_lengths(prefix_lens, "prefix_lens")
        if (
            len(queries) != page_table.shape[0]
            or len(prefixes) != len(queries)
            or sum(queries) != rows
        ):
            return None
        sequences = tuple(p + n for p, n in zip(prefixes, queries))
        if max(sequences) // 4 > columns:
            return None
        layout = _layout(
            sequences, queries, q.device, logits_workspace_bytes=logits_workspace_bytes
        )
        if (max(layout.logits_elements, 1) + _PAD) * 4 > logits_workspace_bytes:
            return None
        stats = torch.full((rows, 2), -1, dtype=torch.int32, device=q.device)
        return select_prefill(
            q, compressed, table, query_positions, layout, stats, out, paged=True
        )
    if rows > 256 or (rows * columns + _PAD) * 4 > logits_workspace_bytes:
        return None
    lengths, sequences, stats = launch.row_metadata(
        token_to_request, query_positions, context_lens, columns
    )
    logits = torch.empty(rows * columns + _PAD, dtype=torch.float32, device=q.device)
    logits = logits[: rows * columns].view(rows, columns)
    launch.decode_logits(
        q,
        compressed,
        table,
        lengths,
        logits,
        _SCALE,
        stats,
        token_to_request=token_to_request,
    )
    launch.decode_topk(logits, lengths, query_positions, sequences, out, stats)
    return out
