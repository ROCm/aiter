# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Launch the gfx942 QSA indexer scorers and exact top-k kernels."""

import threading

import torch

from aiter.jit.utils.chip_info import get_cu_num
from aiter.ops.flydsl.kernels.qsa import (
    indexer_decode,
    indexer_logits,
    indexer_topk,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.flydsl.kernels.topk.topk_launch import _launch_key, _launcher_on

_compile_lock = threading.RLock()


def _run_variants(exe, args, count_index, variants, device):
    """Warm all constexpr variants with zero work, then execute this call once."""
    with torch.cuda.device(device):
        exe = _launcher_on(exe, device.index)
        tensor_key = _launch_key(args)
        count = len(variants[0])
        variant = tuple(args[count_index + 1 : count_index + 1 + count])
        key = (tensor_key, variant)
        cache = getattr(exe, "_cf_by_specialization", {})
        if key not in cache:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm the QSA indexer before graph capture")
            with _compile_lock:
                for choice in variants:
                    warm_args = (
                        *args[:count_index],
                        0,
                        *choice,
                        *args[count_index + 1 + count :],
                    )
                    _run_compiled(
                        exe, *warm_args, specialization_key=(tensor_key, choice)
                    )
        _run_compiled(exe, *args, specialization_key=key)


def prefill_logits(q, compressed, table, logits, items, rows, pitch, scale, stats):
    """Write key-block-major logits for the host-planned query/key tiles."""
    flat_table = table.as_strided(
        ((table.shape[0] - 1) * table.stride(0) + table.shape[1],), (1,)
    )
    args = (
        q.view(-1),
        compressed.view(-1),
        flat_table,
        logits.view(-1),
        items.view(-1),
        stats.view(-1),
        table.stride(0),
        compressed.numel() * compressed.element_size(),
        rows * pitch,
        pitch,
        scale,
        items.shape[0],
        torch.cuda.current_stream(q.device),
    )
    _run_variants(indexer_logits._launch, args, 11, [()], q.device)


def prefill_topk(logits, width, row0, rows, pitch, positions, row_info, out, stats):
    """Select and expand the chunk into its original rows of ``out``."""
    wide = width > indexer_topk.NARROW_KEYS
    split = width >= indexer_topk.SPLIT_KEYS
    args = (
        logits.view(-1),
        rows * pitch,
        pitch,
        row0,
        rows,
        positions,
        row_info.view(-1),
        out,
        stats.view(-1),
        rows if split else -(-rows // indexer_topk.WAVES),
        wide,
        split,
        torch.cuda.current_stream(logits.device),
    )
    variants = [(w, s) for w in (False, True) for s in (False, True)]
    _run_variants(indexer_topk._launch, args, 9, variants, logits.device)


def decode_logits(
    q,
    compressed,
    page_table,
    lengths,
    logits,
    scale,
    stats=None,
):
    """Score each query against its row of 16-key compressed pages."""
    rows, width = logits.shape
    waves = indexer_decode.WAVES
    splits = max(
        1,
        min(
            -(-page_table.shape[1] // waves),
            -(-get_cu_num() * indexer_decode.CTAS_PER_CU // rows),
        ),
    )
    fold = stats is not None
    flat_table = page_table.as_strided(
        ((page_table.shape[0] - 1) * page_table.stride(0) + page_table.shape[1],), (1,)
    )
    args = (
        q.view(-1),
        compressed.view(-1),
        flat_table,
        lengths,
        logits.view(-1),
        (stats if fold else lengths).view(-1),
        q.stride(0),
        page_table.stride(0),
        width,
        splits,
        compressed.numel() * compressed.element_size(),
        scale,
        rows,
        fold,
        torch.cuda.current_stream(q.device),
    )
    _run_variants(indexer_decode._launch, args, 12, [(False,), (True,)], q.device)


def decode_topk(logits, lengths, positions, sequence_lengths, out, stats=None):
    """Select each row, splitting long rows over multiple CTAs."""
    rows, width = logits.shape
    parts = 0
    if width > indexer_topk.LONG_KEYS:
        parts = max(
            1,
            min(
                indexer_topk.MAX_PARTS,
                -(-width // indexer_topk.GROUP_KEYS) // 8,
                -(-2 * get_cu_num() // rows),
            ),
        )
    if parts and stats is None:
        raise ValueError("Long QSA decode rows require logits statistics")
    device = logits.device
    stream = torch.cuda.current_stream(device)
    stats = stats.view(-1) if parts else lengths
    scratch = (lengths,) * 3
    if parts:
        scratch = tuple(
            torch.empty(rows * n, dtype=torch.int32, device=device)
            for n in (
                parts * indexer_topk.BINS,
                indexer_topk.TOPK,
                2 * indexer_topk.CANDIDATES,
            )
        )
    args = (
        logits.view(-1),
        width,
        lengths,
        stats,
        *scratch,
        max(parts, 1),
        rows if parts else 0,
        stream,
    )
    _run_variants(indexer_topk._launch_parts, args, 8, [()], device)
    args = (
        logits.view(-1),
        width,
        rows,
        lengths,
        positions,
        sequence_lengths,
        out,
        stats,
        *scratch,
        parts,
        rows,
        width > indexer_topk.NARROW_KEYS,
        parts > 0,
        stream,
    )
    variants = [(w, long) for w in (False, True) for long in (False, True)]
    _run_variants(indexer_topk._launch_decode, args, 12, variants, device)
