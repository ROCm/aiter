# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-side API for the atomicAdd-based FlyDSL MoE sorting (lazy index)."""

import os
import torch
import flydsl.compiler as flyc

from aiter import dtypes
from .kernels.moe_sorting_atomic import (
    compile_moe_sorting_atomic,
    BLOCK_SIZE,
    UNIT_SIZE,
)

_CF_CACHE = {}
_ZERO_STREAM = {}


def _zero_stream(device):
    s = _ZERO_STREAM.get(device.index)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _ZERO_STREAM[device.index] = s
    return s


def _launch_cached(key, launch_fn, args, stream):
    """First call: flyc.compile() compiles AND executes once; later calls reuse it."""
    import flydsl.expr as fx

    s = fx.Stream(stream)
    cf = _CF_CACHE.get(key)
    if cf is None:
        cf = flyc.compile(launch_fn, *args, s)
        _CF_CACHE[key] = cf
    else:
        cf(*args, s)


def moe_sorting_atomic_fwd(
    topk_ids,
    topk_weights,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
    num_experts,
    unit_size=UNIT_SIZE,
    skip_moe_buf_zero=False,
):
    """atomicAdd-based MoE sorting. Outputs must be pre-allocated.

    Drop-in shape contract matches aiter.moe_sorting_fwd (subset: no EP mask,
    no local tokens).
    """
    assert topk_ids.dtype == torch.int32, "topk_ids must be int32"
    M, topk = topk_ids.shape
    E = int(num_experts)
    device = topk_ids.device
    N = M * topk

    # workspace holds expert offsets [E+1]; partial holds per-block histograms
    # (then rewritten in-place to per-block base slots by cumsum).
    ws = torch.empty(E + 1, dtype=torch.int32, device=device)

    sorted_len = sorted_ids.shape[0]
    sentinel = (topk << 24) | M

    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    occ = 2
    n_fill = min((sorted_len + BLOCK_SIZE - 1) // BLOCK_SIZE, num_cu * occ)
    moe_buf_i32 = moe_buf.view(torch.int32).view(-1)
    v4_total = moe_buf_i32.numel() // 4
    _zocc = int(os.environ.get("AITER_ZERO_OCC", "8"))
    n_zero = min((v4_total + BLOCK_SIZE - 1) // BLOCK_SIZE, num_cu * _zocc)
    _cs_blocks_env = int(os.environ.get("AITER_SORT_BLOCKS", "0"))
    _cs_cap = _cs_blocks_env if _cs_blocks_env > 0 else 32
    n_cs = max(1, min((N + BLOCK_SIZE - 1) // BLOCK_SIZE, _cs_cap))

    partial = torch.empty(n_cs * E, dtype=torch.int32, device=device)

    # AITER_SORT_CONTIG=1: contiguous (not grid-stride) block partitioning in
    # count/scatter -> coarsely token-ordered output -> better stage2 locality,
    # fully on-GPU (no host argsort). Default off.
    # AITER_SORT_ORDERED=1: exact token-ascending order within each expert
    # (CK-quality), fully on-GPU; scatter uses a single-thread in-LDS rank pass
    # (slower sort, optimal stage2). Implies contiguous partitioning.
    # Default ON: exact token-ascending order within each expert. The unordered
    # grid-stride atomic mode sorts marginally faster but scrambles intra-expert
    # token order, which wrecks the downstream MoE GEMM's gather locality (stage1
    # gateup ~+1.2ms at 32k tokens) -> net e2e regression. Set AITER_SORT_ORDERED=0
    # to measure raw sort peak.
    _ordered = os.environ.get("AITER_SORT_ORDERED", "1") == "1"
    _contig = _ordered or os.environ.get("AITER_SORT_CONTIG", "0") == "1"
    # AITER_SORT_FUSE_CUMSUM=1 (default): drop cumsum's single-block phase C and
    # let each scatter block compute its base from raw counts + ws_offset. Pure
    # optimization (identical output); set 0 to use the legacy phase-C path.
    _fuse_cumsum = os.environ.get("AITER_SORT_FUSE_CUMSUM", "1") == "1"
    # AITER_SORT_FUSE_LAUNCH=1 (default): cut 5 launches to 3 by folding fill into
    # count (stage0) and write_eids into scatter (stage2). Identical output.
    _fuse_launch = os.environ.get("AITER_SORT_FUSE_LAUNCH", "1") == "1"

    (launch_fill, launch_count, launch_cumsum, launch_write_eids,
     launch_scatter) = compile_moe_sorting_atomic(
        num_experts=E, topk=topk, nblocks=n_cs, unit_size=unit_size,
        contig=_contig, ordered=_ordered, fuse_cumsum=_fuse_cumsum,
        fuse_launch=_fuse_launch,
    )

    stream = torch.cuda.current_stream()
    base = (E, topk, unit_size, device.index, n_cs, _contig, _ordered, _fuse_cumsum,
            _fuse_launch)

    # count is fused with moe_buf zeroing: blocks [n_cs, n_cs+n_zero) clear
    # moe_buf concurrently with the n_cs counting blocks (the clear is HBM-BW
    # bound; counting is cheap, so the clear is largely hidden).
    # AITER_SORT_SKIP_ZERO=1 skips the moe_buf clear (sort-only measurement /
    # or when the caller zeroes moe_buf elsewhere, e.g. fused into stage1 init).
    # Skip the moe_buf clear when the caller's stage2 doesn't atomic-accumulate
    # into it (reduce/split-reduce stage2), or when explicitly forced via env.
    if skip_moe_buf_zero or os.environ.get("AITER_SORT_SKIP_ZERO", "0") == "1":
        v4_total = 0
    _n_zero = n_zero if v4_total > 0 else 0

    if _fuse_launch:
        # stage0: count[0,n_cs) + fill[n_cs,n_cs+n_fill) + zero[..]; one launch.
        count_grid = n_cs + n_fill + _n_zero
        _launch_cached(base + ("count",), launch_count,
                       (topk_ids, partial, moe_buf_i32, sorted_ids, sorted_weights,
                        N, v4_total, sorted_len, sentinel, n_fill, count_grid), stream)
        _launch_cached(base + ("cumsum",), launch_cumsum,
                       (partial, ws, num_valid_ids, M), stream)
        # stage2: scatter[0,n_cs) + write_eids[n_cs,n_cs+E); one launch.
        _launch_cached(base + ("scatter",), launch_scatter,
                       (topk_ids, topk_weights, partial, ws, sorted_ids, sorted_weights,
                        sorted_expert_ids, N, n_cs + E),
                       stream)
    else:
        count_grid = n_cs + _n_zero
        _launch_cached(base + ("fill",), launch_fill,
                       (sorted_ids, sorted_weights, sorted_len, sentinel, n_fill), stream)
        _launch_cached(base + ("count",), launch_count,
                       (topk_ids, partial, moe_buf_i32, sorted_ids, sorted_weights,
                        N, v4_total, sorted_len, sentinel, 0, count_grid), stream)
        _launch_cached(base + ("cumsum",), launch_cumsum,
                       (partial, ws, num_valid_ids, M), stream)
        _launch_cached(base + ("weids",), launch_write_eids,
                       (ws, sorted_expert_ids, E), stream)
        _launch_cached(base + ("scatter",), launch_scatter,
                       (topk_ids, topk_weights, partial, ws, sorted_ids, sorted_weights,
                        sorted_expert_ids, N, n_cs),
                       stream)

    # DIAGNOSTIC: re-sort each expert segment so token ids are ascending (like
    # CK), keeping num_valid / padding / segments identical. Used to isolate the
    # effect of intra-expert ordering on the downstream GEMM. NOT for production.
    if os.environ.get("AITER_SORT_REORDER", "0") == "1":
        n = int(num_valid_ids[0].item())
        if n > 0:
            blk = int(unit_size)
            nb = n // blk
            se = sorted_expert_ids[:nb].to(torch.int64)
            expert_of_slot = se.repeat_interleave(blk)  # [n]
            ids = sorted_ids[:n]
            tok = (ids & 0x00FFFFFF).to(torch.int64)
            is_real = ids != sentinel
            big = 1 << 25
            key = expert_of_slot * big + torch.where(
                is_real, tok, torch.full_like(tok, M)
            )
            perm = torch.argsort(key, stable=True)
            sorted_ids[:n] = ids[perm]
            sorted_weights[:n] = sorted_weights[:n][perm]

    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


def moe_sorting_atomic(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size=UNIT_SIZE,
    expert_mask=None,
    num_local_tokens=None,
    dispatch_policy=0,
):
    """Allocate outputs (matching aiter.fused_moe._moe_sorting_impl) and run.

    Returns (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf).
    """
    device = topk_ids.device
    M, topk = topk_ids.shape
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.to(torch.float32)

    max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_size - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device)
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)

    return moe_sorting_atomic_fwd(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts,
        int(block_size),
    )
