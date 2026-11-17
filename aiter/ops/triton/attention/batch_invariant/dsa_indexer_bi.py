# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Batch-invariant DSA indexer (FP8 MQA logits + exact top-k), Gluon kernels (gfx950)."""

import functools
import inspect

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx950.attention.batch_invariant.dsa_indexer_bi import (
    _dsa_indexer_bi_logits_kernel,
    _dsa_indexer_bi_topk_hist_kernel,
    _dsa_indexer_bi_topk_kernel,
    _dsa_indexer_bi_topk_select_kernel,
    _dsa_indexer_bi_topk_write_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir

__all__ = [
    "BI_CONTRACT_VERSION",
    "dsa_indexer_bi",
    "dsa_mqa_logits_bi",
    "dsa_topk_bi",
]

# Bump when a change can change the output bits; callers compare it at startup.
BI_CONTRACT_VERSION = 1

_HEAD_SIZE = 128
_NUM_HEADS = (32, 64)
_TILE = 64  # keys per K tile; cache blocks hold whole tiles
_MAX_COLS = (1 << 31) // 4  # logits row bytes stay below 2 GiB (buffer offsets)


@functools.cache
def _config(name: str) -> dict:
    """Pinned launch values of one kernel."""
    cfg_dir = resolve_config_dir("attention", "DSA-INDEXER-BI", backend="gluon")
    return dict(load_config_json(f"{cfg_dir}/DEFAULT.json")[name])


@functools.cache
def _padded_lds() -> bool:
    # Newer Triton takes a distributed layout in the async copy (padded LDS).
    from triton.experimental.gluon.language.amd.cdna4 import async_copy

    try:
        return "DistributedLayout" in inspect.getsource(
            async_copy.global_load_to_shared
        )
    except (OSError, TypeError):
        return False


@functools.cache
def _check_toolchain() -> None:
    if get_arch() != "gfx950":
        raise NotImplementedError("dsa_indexer_bi: gfx950 only")
    from triton.language.core import _unwrap_iterable, constexpr

    # The head fold permutes with a constexpr tuple (Triton PR #9751).
    if isinstance(_unwrap_iterable((constexpr((0, 1)),)), constexpr):
        raise NotImplementedError("dsa_indexer_bi: this Triton is too old")


def _check_device(*tensors):
    dev = tensors[0].device
    for t in tensors:
        if t.device.type != "cuda" or t.device != dev:
            raise ValueError("dsa_indexer_bi: all tensors must be on one GPU")


def _check_rows(T, row_req, row_len):
    for name, t in (("row_req", row_req), ("row_len", row_len)):
        if t.dtype != torch.int32 or t.dim() != 1 or t.shape[0] != T:
            raise ValueError(f"dsa_indexer_bi: {name} must be int32 [{T}]")
    if T and (row_req.stride(0) != 1 or row_len.stride(0) != 1):
        raise ValueError("dsa_indexer_bi: row_req and row_len must be contiguous")


def _check_logits_args(q, weights, index_cache, block_table, row_req, row_len, max_len):
    _check_toolchain()
    _check_device(q, weights, index_cache, block_table, row_req, row_len)
    if q.dim() != 3 or q.dtype != torch.float8_e4m3fn:
        raise TypeError("dsa_indexer_bi: q must be float8_e4m3fn [T, H, D]")
    T, H, D = q.shape
    if D != _HEAD_SIZE or H not in _NUM_HEADS:
        raise ValueError(
            f"dsa_indexer_bi: needs D={_HEAD_SIZE} and H in {_NUM_HEADS}, got H={H} D={D}"
        )
    if q.stride(2) != 1:
        raise ValueError("dsa_indexer_bi: q must be contiguous in D")
    if weights.dtype != torch.float32 or tuple(weights.shape) != (T, H):
        raise TypeError(f"dsa_indexer_bi: weights must be float32 [{T}, {H}]")
    if index_cache.dtype != torch.uint8 or index_cache.dim() != 3:
        raise TypeError("dsa_indexer_bi: index_cache must be uint8 [blocks, BS, D + 4]")
    _, bs, row = index_cache.shape
    if row != D + 4 or bs % _TILE:
        raise ValueError(
            f"dsa_indexer_bi: index_cache must be [blocks, BS, {D + 4}] with BS % {_TILE} == 0"
        )
    if not index_cache[0].is_contiguous() or index_cache.stride(0) % 4:
        raise ValueError("dsa_indexer_bi: each index_cache block must be contiguous")
    if block_table.dtype != torch.int32 or block_table.dim() != 2:
        raise TypeError("dsa_indexer_bi: block_table must be int32 [R, max_blocks]")
    if block_table.stride(1) != 1:
        raise ValueError("dsa_indexer_bi: block_table must be contiguous in its rows")
    _check_rows(T, row_req, row_len)
    if not 0 < max_len <= _MAX_COLS:
        raise ValueError(f"dsa_indexer_bi: max_len must be in (0, {_MAX_COLS}]")


def _flat_cache_views(index_cache):
    """(fp8 view, fp32 view) of the block-flat cache bytes, block strides in elements."""
    nb, bs, row = index_cache.shape
    flat = torch.as_strided(index_cache, (nb, bs * row), (index_cache.stride(0), 1))
    return flat.view(torch.float8_e4m3fn), flat.view(torch.float32)


def _launch_logits(q, weights, k8, s32, block_table, row_req, row_len, max_len, out):
    T, H, D = q.shape
    if T == 0:
        return
    cfg = _config("logits")
    max_tiles = triton.cdiv(max_len, _TILE)
    # The split only partitions columns, so it may follow the launch shape.
    rows = cfg["rows_by_heads"][str(H)]
    # Splits count the row pairs: in prefill one program of a pair does the work.
    splits = max(
        1,
        min(
            cfg["target_programs"] // triton.cdiv(T, rows),
            triton.cdiv(max_tiles, cfg["min_split_tiles"]),
        ),
    )
    tiles_per_split = triton.cdiv(max_tiles, splits)
    splits = triton.cdiv(max_tiles, tiles_per_split)
    _dsa_indexer_bi_logits_kernel[(triton.cdiv(T, rows), splits, rows)](
        q,
        weights,
        k8,
        s32,
        block_table,
        row_req,
        row_len,
        out,
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        weights.stride(1),
        k8.stride(0),
        s32.stride(0),
        block_table.stride(0),
        out.stride(0),
        T,
        tiles_per_split,
        max_len,
        NUM_HEADS=H,
        HEAD_SIZE=D,
        CACHE_BS=k8.shape[1] // (D + 4),
        BLOCK_KV=_TILE,
        ROWS=rows,
        NUM_WARPS=cfg["num_warps"],
        PADDED_LDS=_padded_lds(),
        num_warps=cfg["num_warps"],
        waves_per_eu=cfg["waves_per_eu"],
    )


def _launch_topk(logits, row_len, topk, max_len, out):
    T = logits.shape[0]
    if T == 0:
        return
    cfg = _config("topk")
    res = cfg["res"]
    _dsa_indexer_bi_topk_kernel[(T,)](
        logits,
        logits.stride(0),
        row_len,
        out,
        out.stride(0),
        max_len,
        TOPK=topk,
        RES=res,
        NUM_WARPS=cfg["num_warps"],
        num_warps=cfg["num_warps"],
    )
    if max_len <= res:
        return
    # Longer rows: chunk histograms, one select per round, then the write.
    # Counts are exact, so the chunk size only changes the launch.
    want = max(
        triton.cdiv(T * max_len, cfg["target_programs"]),
        triton.cdiv(max_len, cfg["max_chunks"]),
    )
    chunk = min(cfg["chunk_max"], max(cfg["chunk_min"], triton.next_power_of_2(want)))
    chunks = triton.cdiv(max_len, chunk)
    dev = logits.device
    state = torch.empty((T, 4), dtype=torch.int32, device=dev)
    hist = torch.empty((T, 4 * chunks * 256), dtype=torch.int32, device=dev)
    aux = torch.empty((T, 2 * chunks), dtype=torch.int32, device=dev)
    nw = cfg["chunk_warps"]
    for rnd in range(4):
        _dsa_indexer_bi_topk_hist_kernel[(T, chunks)](
            logits,
            logits.stride(0),
            row_len,
            state,
            hist,
            hist.stride(0),
            chunks * 256,
            max_len,
            ROUND=rnd,
            RES=res,
            CHUNK=chunk,
            NUM_WARPS=nw,
            num_warps=nw,
        )
        _dsa_indexer_bi_topk_select_kernel[(T,)](
            row_len,
            state,
            hist,
            hist.stride(0),
            chunks * 256,
            aux,
            aux.stride(0),
            max_len,
            ROUND=rnd,
            TOPK=topk,
            RES=res,
            CHUNK=chunk,
            num_warps=4,
        )
    _dsa_indexer_bi_topk_write_kernel[(T, chunks)](
        logits,
        logits.stride(0),
        row_len,
        state,
        aux,
        aux.stride(0),
        out,
        out.stride(0),
        max_len,
        RES=res,
        CHUNK=chunk,
        BLOCK=min(chunk, cfg["write_block"]),
        NUM_WARPS=nw,
        num_warps=nw,
    )


def _check_out(out, shape, dtype, name):
    if out is None:
        return
    if out.dtype != dtype or out.dim() != 2 or out.shape[0] != shape[0]:
        raise ValueError(
            f"dsa_indexer_bi: {name} must be {dtype} [{shape[0]}, >= {shape[1]}]"
        )
    if out.shape[1] < shape[1] or out.stride(1) != 1:
        raise ValueError(
            f"dsa_indexer_bi: {name} must be [{shape[0]}, >= {shape[1]}], contiguous rows"
        )


def dsa_mqa_logits_bi(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    row_req: torch.Tensor,
    row_len: torch.Tensor,
    max_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batch-invariant FP8 MQA logits of the DSA indexer over a paged index cache.

    ``out[t, j] = k_scale[j] * sum_h weights[t, h] * relu(q[t, h] . k[j])`` for
    ``j < row_len[t]``, where key ``j`` of row ``t`` is in block
    ``block_table[row_req[t], j // BS]``. One fixed MFMA shape and head-fold order,
    so a logit depends only on its row and key: not on the batch, the row order,
    ``max_len`` or the block placement. Prefill and decode rows are the same.

    Args:
        q: float8_e4m3fn [T, H, 128], H in (32, 64), contiguous in D.
        weights: float32 [T, H] head weights.
        index_cache: uint8 [blocks, BS, 132], block-flat layout
            (BS * 128 fp8 key bytes, then BS fp32 scales), BS a multiple of 64.
        block_table: int32 [R, max_blocks].
        row_req: int32 [T], the block-table row of each query row.
        row_len: int32 [T], keys [0, row_len[t]) of each row (clamped to max_len).
            The block-table row must name a valid block for each of these keys.
        max_len: column capacity; fixes the launch (CUDA-graph safe).
        out: optional float32 [T, >= max_len].

    Returns:
        float32 [T, max_len] (or ``out``); columns at or past row_len are not written.
    """
    _check_logits_args(q, weights, index_cache, block_table, row_req, row_len, max_len)
    T = q.shape[0]
    _check_out(out, (T, max_len), torch.float32, "out")
    if out is None:
        out = torch.empty((T, max_len), dtype=torch.float32, device=q.device)
    k8, s32 = _flat_cache_views(index_cache)
    _launch_logits(q, weights, k8, s32, block_table, row_req, row_len, max_len, out)
    return out


def dsa_topk_bi(
    logits: torch.Tensor,
    row_len: torch.Tensor,
    topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact, batch-invariant top-k of ``logits[t, 0:row_len[t]]``.

    Order: the fp32 total order of the bit patterns (-0.0 < +0.0), ties to the
    larger column. Writes the ``min(row_len[t], topk)`` selected columns ascending,
    then -1 to width ``topk``. The result depends only on the row.

    Args:
        logits: float32 [T, W], contiguous rows.
        row_len: int32 [T], clamped to [0, W].
        topk: number of columns to select, a power of 2 from 64 to 4096.
        out: optional int32 [T, >= topk]; only the first ``topk`` columns are written.

    Returns:
        int32 [T, topk] (or ``out``).
    """
    _check_toolchain()
    _check_device(logits, row_len)
    if logits.dtype != torch.float32 or logits.dim() != 2 or logits.stride(1) != 1:
        raise TypeError(
            "dsa_topk_bi: logits must be float32 [T, W] with contiguous rows"
        )
    if topk < 64 or topk > 4096 or topk & (topk - 1):
        raise ValueError("dsa_topk_bi: topk must be a power of 2 in [64, 4096]")
    T, W = logits.shape
    if W > _MAX_COLS:
        raise ValueError(f"dsa_topk_bi: at most {_MAX_COLS} columns")
    _check_rows(T, row_len, row_len)
    _check_out(out, (T, topk), torch.int32, "out")
    if out is None:
        out = torch.empty((T, topk), dtype=torch.int32, device=logits.device)
    _launch_topk(logits, row_len, topk, W, out)
    return out


def dsa_indexer_bi(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    row_req: torch.Tensor,
    row_len: torch.Tensor,
    topk: int,
    max_len: int,
    out: torch.Tensor | None = None,
    logits_ws: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batch-invariant DSA indexer: ``dsa_topk_bi(dsa_mqa_logits_bi(...))``.

    Rows run in groups that fit ``logits_ws`` (or the config byte budget); the
    groups do not change the result. See ``dsa_mqa_logits_bi`` and ``dsa_topk_bi``
    for the arguments and the contract.

    Args:
        logits_ws: optional float32 [G, >= max_len] workspace, G >= 1.

    Returns:
        int32 [T, topk] (or ``out``): row-local key indices, ascending, -1 padded.
    """
    _check_logits_args(q, weights, index_cache, block_table, row_req, row_len, max_len)
    if topk < 64 or topk > 4096 or topk & (topk - 1):
        raise ValueError("dsa_indexer_bi: topk must be a power of 2 in [64, 4096]")
    T = q.shape[0]
    _check_out(out, (T, topk), torch.int32, "out")
    if out is None:
        out = torch.empty((T, topk), dtype=torch.int32, device=q.device)
    if T == 0:
        return out
    if logits_ws is None:
        rows = _config("indexer")["workspace_bytes"] // (4 * max_len)
        group = max(1, min(T, rows))
        logits_ws = torch.empty((group, max_len), dtype=torch.float32, device=q.device)
    else:
        _check_device(q, logits_ws)
        if logits_ws.dim() != 2 or logits_ws.shape[0] < 1:
            raise ValueError(
                "dsa_indexer_bi: logits_ws must be float32 [G, >= max_len]"
            )
        _check_out(logits_ws, (logits_ws.shape[0], max_len), torch.float32, "logits_ws")
        group = logits_ws.shape[0]
    k8, s32 = _flat_cache_views(index_cache)
    for lo in range(0, T, group):
        hi = min(T, lo + group)
        ws = logits_ws[: hi - lo]
        _launch_logits(
            q[lo:hi],
            weights[lo:hi],
            k8,
            s32,
            block_table,
            row_req[lo:hi],
            row_len[lo:hi],
            max_len,
            ws,
        )
        _launch_topk(ws, row_len[lo:hi], topk, max_len, out[lo:hi])
    return out
