# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf coverage for the FlyDSL MLA reduce kernel (aiter op_test
standard: a pytest-free script run as ``python3 test_flydsl_mla_reduce.py``,
gated on process exit code by aiter CI).

Irregular-first: most correctness cases use production-shaped metadata
(variable per-tile ``n_splits``, gapped ``reduce_partial_map``, MLDS tier
boundary, empty tiles). Uniform/dense layouts are kept only as a small smoke
layer. This mirrors real split-KV decode, where every tile can need a
different split count and the partial buffer is a sparsely-indexed pool.

Running this file (``main()``) does two things in order:
  1. ``run_checks()`` -- every invariant/correctness check below (guards,
     cudagraph capture/replay, split-K planning, dispatch-seam introspection,
     empty-tile/OOB regressions). Any failure aborts with a non-zero exit
     before the perf sweep runs.
  2. ``run_bench()`` -- the GLM-5.2 serving / uniform / irregular perf
     scoreboard (``wrapper`` vs ``hip``), one markdown table each.
"""

import argparse
import itertools
import os
import sys
from contextlib import contextmanager
from unittest import mock

import aiter
import pandas as pd
import torch

from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_mla_reduce_v1

from aiter.ops.flydsl.kernels.mla_reduce import (
    LDS_MAX_SPLITS,
    Tier,
    compile_mla_reduce,
    compile_mla_reduce_splitk,
    plan_splitk,
    select_tier,
    should_use_persistent_launch,
    splitk_enabled,
    waves_per_eu_from_env,
    _get_splitk_scratch,
)

SERVING_NUM_REDUCE_TILE = 16384
SERVING_PARTIAL_POOL = 606
# gfx950 not yet a target for this kernel; keep tests scoped to gfx942 until it is.
MLA_REDUCE_SUPPORTED_GFX = ["gfx942"]


def mla_reduce_out_dtype(dt: str) -> torch.dtype:
    return torch.bfloat16 if dt == "bf16" else torch.float16


def mla_reduce_out_atol(dt: str | torch.dtype) -> float:
    return 6.3e-2 if dt in ("bf16", torch.bfloat16) else 2e-3


_out_dtype = mla_reduce_out_dtype
_out_atol = mla_reduce_out_atol


def build_irregular_inputs(
    splits_per_tile,
    H,
    Dv,
    out_dtype,
    M=1,
    gap_stride=1,
    pool_slack=0,
    device="cuda",
    seed=0,
):
    """Build reduce inputs mirroring real decode metadata (variable ``n_splits``
    per tile, non-dense ``reduce_partial_map``, over-sized pool). ``build_inputs``
    is the dense special case (``[S]*num_tiles``, ``gap_stride=1``, ``pool_slack=0``).

    Args:
        splits_per_tile: per-tile ``n_splits``; ``0`` (or ``1``) marks an empty tile
            whose ``reduce_final_map`` q-range is garbage (exercises the empty-tile
            guard, which must never deref it).
        gap_stride: spacing between partial-pool base rows (``1`` dense, ``>1`` holes).
        pool_slack: extra unused rows appended to the pool (over-allocated buffer).

    For ``M > 1`` each split owns ``M`` contiguous partial rows and each tile's final
    q-range spans ``[tile*M, tile*M + M)`` (``get_mla_metadata_v1`` layout).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    num_tiles = len(splits_per_tile)
    total_splits = int(sum(int(s) for s in splits_per_tile))

    indptr_host = [0]
    for s in splits_per_tile:
        indptr_host.append(indptr_host[-1] + int(s))
    reduce_indptr = torch.tensor(indptr_host, dtype=torch.int32, device=device)

    if total_splits > 0:
        slot = torch.arange(total_splits, dtype=torch.int32, device=device)
        reduce_partial_map = slot * (gap_stride * M)
        max_base = int(reduce_partial_map.max().item())
    else:
        reduce_partial_map = torch.zeros(1, dtype=torch.int32, device=device)
        max_base = 0
    num_partial_rows = max_base + M + pool_slack * M

    partial_output = torch.randn(
        num_partial_rows, H, Dv, dtype=torch.float32, device=device, generator=g
    )
    partial_lse = (
        torch.randn(
            num_partial_rows, H, dtype=torch.float32, device=device, generator=g
        )
        * 2.0
    )

    q_start = torch.arange(num_tiles, dtype=torch.int32, device=device) * M
    reduce_final_map = torch.stack([q_start, q_start + M], dim=1).contiguous()
    for t, s in enumerate(splits_per_tile):
        if int(s) <= 1:
            reduce_final_map[t, 0] = 1 << 24
            reduce_final_map[t, 1] = (1 << 24) + M

    final_output = torch.empty(num_tiles * M, H, Dv, dtype=out_dtype, device=device)
    final_lse = torch.empty(num_tiles * M, H, dtype=torch.float32, device=device)
    return (
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        final_output,
        final_lse,
    )


def build_inputs(num_tiles, num_splits, H, Dv, out_dtype, M=1, device="cuda", seed=0):
    """Dense/uniform reduce inputs: every tile has ``num_splits`` splits and the
    gather map is contiguous."""
    return build_irregular_inputs(
        [num_splits] * num_tiles,
        H,
        Dv,
        out_dtype,
        M=M,
        gap_stride=1,
        device=device,
        seed=seed,
    )


def build_degenerate_inputs(num_tiles, H, Dv, out_dtype, device="cuda", seed=0):
    """All-empty (``n_splits=0``) metadata for the empty-tile guard regression."""
    return build_irregular_inputs(
        [0] * num_tiles, H, Dv, out_dtype, gap_stride=1, device=device, seed=seed
    )


def build_serving_decode_inputs(
    active_tiles,
    splits,
    out_dtype,
    H=16,
    Dv=512,
    num_reduce_tile=SERVING_NUM_REDUCE_TILE,
    partial_pool=SERVING_PARTIAL_POOL,
    device="cuda",
    seed=0,
):
    """Sparse serving decode grid with active-sized outputs."""
    active_splits = active_tiles * splits
    pool_slack = max(0, partial_pool - active_splits)
    splits_per_tile = [splits] * active_tiles + [0] * (num_reduce_tile - active_tiles)
    po, pl, indptr, fmap, pmap, _, _ = build_irregular_inputs(
        splits_per_tile,
        H,
        Dv,
        out_dtype,
        M=1,
        gap_stride=1,
        pool_slack=pool_slack,
        device=device,
        seed=seed,
    )
    return (
        po,
        pl,
        indptr,
        fmap,
        pmap,
        torch.empty(active_tiles, H, Dv, dtype=out_dtype, device=device),
        torch.empty(active_tiles, H, dtype=torch.float32, device=device),
    )


def torch_ref(
    partial_output, partial_lse, num_tiles, num_splits, H, Dv, out_dtype, M=1
):
    """Vectorized online-softmax reduce reference (any max_seqlen_q M)."""
    po = partial_output.view(num_tiles, num_splits, M, H, Dv).double()
    pl = partial_lse.view(num_tiles, num_splits, M, H).double()
    max_lse = pl.max(dim=1, keepdim=True).values
    w = torch.exp(pl - max_lse)
    denom = w.sum(dim=1)
    num = (w.unsqueeze(-1) * po).sum(dim=1)
    out = (num / denom.unsqueeze(-1)).to(out_dtype)
    lse = (max_lse.squeeze(1) + torch.log(denom)).float()
    return out.reshape(num_tiles * M, H, Dv), lse.reshape(num_tiles * M, H)


def torch_ref_gather(
    po,
    pl,
    indptr,
    fmap,
    pmap,
    H,
    Dv,
    out_dtype,
    M=1,
    num_partial_rows=None,
    num_final_rows=None,
):
    """Gather-based online-softmax reference for irregular metadata.

    Follows the kernel's CSR + gather contract: per tile it gathers partial rows via
    ``pmap[indptr[t]:indptr[t+1]]`` and merges them; tiles with ``n_splits <= 1`` are
    skipped (output rows stay zero). When ``num_partial_rows`` / ``num_final_rows``
    are set, out-of-range pmap rows / q_start are skipped (mirrors guards-ON).
    """
    num_tiles = fmap.shape[0]
    ref_out = torch.zeros(num_tiles * M, H, Dv, dtype=out_dtype, device=po.device)
    ref_lse = torch.zeros(num_tiles * M, H, dtype=torch.float32, device=po.device)
    indptr_h = indptr.tolist()
    pmap_h = pmap.tolist()
    fmap_h = fmap.tolist()
    pod = po.double()
    pld = pl.double()
    for t in range(num_tiles):
        s0, s1 = indptr_h[t], indptr_h[t + 1]
        if s1 - s0 <= 1:
            continue
        q_start = fmap_h[t][0]
        if num_final_rows is not None and (q_start < 0 or q_start >= num_final_rows):
            continue
        bases = pmap_h[s0:s1]
        for local in range(M):
            rows = []
            for b in bases:
                row = b + local
                if num_partial_rows is not None and (
                    row < 0 or row >= num_partial_rows
                ):
                    continue
                rows.append(row)
            if not rows:
                continue
            o = pod[rows]
            lg = pld[rows]
            max_lse = lg.max(dim=0, keepdim=True).values
            w = torch.exp(lg - max_lse)
            denom = w.sum(dim=0)
            num = (w.unsqueeze(-1) * o).sum(dim=0)
            ref_out[q_start + local] = (num / denom.unsqueeze(-1)).to(out_dtype)
            ref_lse[q_start + local] = (max_lse.squeeze(0) + torch.log(denom)).float()
    return ref_out, ref_lse


def hip_ref(po, pl, indptr, fmap, pmap, num_tiles, H, Dv, out_dtype, M=1):
    """Reference output from HIP kn_mla_reduce_v1. Outputs are zero-initialized so
    skipped (empty) tiles match a zero-initialized final buffer under test.
    """
    ref_out = torch.zeros(num_tiles * M, H, Dv, dtype=out_dtype, device=po.device)
    ref_lse = torch.zeros(num_tiles * M, H, dtype=torch.float32, device=po.device)
    aiter.mla_reduce_v1(po, pl, indptr, fmap, pmap, M, LDS_MAX_SPLITS, ref_out, ref_lse)
    torch.cuda.synchronize()
    return ref_out, ref_lse


def hip_ref_like_fout(po, pl, indptr, fmap, pmap, fout, flse, M=1):
    """HIP reference sized to ``fout`` / ``flse`` (serving grids with sparse tiles)."""
    ref_out = torch.zeros_like(fout)
    ref_lse = torch.zeros_like(flse)
    aiter.mla_reduce_v1(po, pl, indptr, fmap, pmap, M, LDS_MAX_SPLITS, ref_out, ref_lse)
    torch.cuda.synchronize()
    return ref_out, ref_lse


def build_serving_sparse_grid_inputs(
    H=16,
    Dv=512,
    out_dtype=torch.bfloat16,
    device="cuda",
    seed=0,
):
    """batch=8 steady-state: 16384-tile grid, 8 active tiles × 32 splits.

    Mirrors 131K-context serving metadata: partial pool 606 rows, CSR sentinel
    at 256, garbage ``reduce_final_map`` / ``reduce_partial_map`` tail slots.

    Tail tiles are flat at sentinel (``n_splits == 0``), already skipped by the
    ``n_splits > 1`` clamp, so this fixture does **not** discriminate the
    gather/store guards.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    num_reduce_tile = SERVING_NUM_REDUCE_TILE
    active_tiles = 8
    splits_per_active = 32
    total_splits = active_tiles * splits_per_active
    sentinel = total_splits
    num_partial_rows = SERVING_PARTIAL_POOL

    indptr_host = list(range(0, sentinel + 1, splits_per_active))
    while len(indptr_host) <= num_reduce_tile:
        indptr_host.append(sentinel)
    reduce_indptr = torch.tensor(indptr_host, dtype=torch.int32, device=device)

    partial_output = torch.randn(
        num_partial_rows, H, Dv, dtype=torch.float32, device=device, generator=g
    )
    partial_lse = (
        torch.randn(
            num_partial_rows, H, dtype=torch.float32, device=device, generator=g
        )
        * 2.0
    )

    reduce_partial_map = torch.empty(num_partial_rows, dtype=torch.int32, device=device)
    reduce_partial_map[:total_splits] = torch.arange(
        total_splits, dtype=torch.int32, device=device
    )
    garbage_pmap = torch.randint(
        -(1 << 30),
        1 << 30,
        (num_partial_rows - total_splits,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    reduce_partial_map[total_splits:] = garbage_pmap

    reduce_final_map = torch.empty(num_reduce_tile, 2, dtype=torch.int32, device=device)
    q = torch.arange(active_tiles, dtype=torch.int32, device=device)
    reduce_final_map[:active_tiles, 0] = q
    reduce_final_map[:active_tiles, 1] = q + 1
    reduce_final_map[active_tiles:, 0] = 1 << 24
    reduce_final_map[active_tiles:, 1] = (1 << 24) + 1

    final_output = torch.zeros(active_tiles, H, Dv, dtype=out_dtype, device=device)
    final_lse = torch.zeros(active_tiles, H, dtype=torch.float32, device=device)
    return (
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        final_output,
        final_lse,
    )


def build_serving_mapped_slack_inputs(
    H=16,
    Dv=512,
    out_dtype=torch.bfloat16,
    device="cuda",
    seed=0,
    num_tiles=64,
    splits_per_active=32,
    slack_p=4,
    slack_f=2,
):
    """Small serving grid with allocation slack for in-process guard differentials.

    Allocates ``partial_output`` / ``final_output`` with extra rows but returns
    smaller *logical* ``num_partial_rows`` / ``num_final_rows`` for the kernel.
    The fake-active tail (non-sentinel ``n_splits``) exercises the gather/store
    guards adversarially.

    Returns tensors plus a ``meta`` dict with logical bounds and discriminator
    tile indices for tests.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    active_tiles = 3
    gather_tile = 0
    store_tile = 10
    store_splits = 4
    total_active_splits = active_tiles * splits_per_active
    total_splits = total_active_splits + store_splits

    indptr_host = [0]
    for _ in range(active_tiles):
        indptr_host.append(indptr_host[-1] + splits_per_active)
    base_flat = indptr_host[-1]
    while len(indptr_host) <= store_tile:
        indptr_host.append(base_flat)
    indptr_host.append(base_flat + store_splits)
    while len(indptr_host) <= num_tiles:
        indptr_host.append(indptr_host[-1])
    reduce_indptr = torch.tensor(indptr_host, dtype=torch.int32, device=device)

    logical_partial_rows = 256
    alloc_partial_rows = logical_partial_rows + slack_p
    logical_final_rows = active_tiles
    alloc_final_rows = logical_final_rows + slack_f

    partial_output = torch.randn(
        alloc_partial_rows, H, Dv, dtype=torch.float32, device=device, generator=g
    )
    partial_lse = (
        torch.randn(
            alloc_partial_rows, H, dtype=torch.float32, device=device, generator=g
        )
        * 2.0
    )

    reduce_partial_map = torch.arange(total_splits, dtype=torch.int32, device=device)
    # Gather discriminator: one split on tile 0 points into mapped slack.
    slack_p_row = logical_partial_rows
    bad_split = splits_per_active // 2
    reduce_partial_map[bad_split] = slack_p_row
    partial_output[slack_p_row].fill_(1000.0)
    partial_lse[slack_p_row].fill_(50.0)
    tile0_lse_max = partial_lse[:splits_per_active].max().item()
    partial_lse[slack_p_row].fill_(tile0_lse_max + 5.0)

    reduce_final_map = torch.empty(num_tiles, 2, dtype=torch.int32, device=device)
    q = torch.arange(active_tiles, dtype=torch.int32, device=device)
    reduce_final_map[:active_tiles, 0] = q
    reduce_final_map[:active_tiles, 1] = q + 1
    reduce_final_map[active_tiles:store_tile, 0] = 1 << 24
    reduce_final_map[active_tiles:store_tile, 1] = (1 << 24) + 1

    store_slack_q = logical_final_rows
    reduce_final_map[store_tile, 0] = store_slack_q
    reduce_final_map[store_tile, 1] = store_slack_q + 1
    reduce_final_map[store_tile + 1 :, 0] = 1 << 24
    reduce_final_map[store_tile + 1 :, 1] = (1 << 24) + 1

    last = int(reduce_indptr[num_tiles].item())
    t0_store = int(reduce_indptr[store_tile].item())
    n_splits_store = int(
        reduce_indptr[store_tile + 1].item() - reduce_indptr[store_tile].item()
    )
    assert n_splits_store >= 2, "store discriminator must be fake-active"
    assert t0_store != last, "store discriminator must not hit sentinel skip"

    final_output = torch.zeros(alloc_final_rows, H, Dv, dtype=out_dtype, device=device)
    final_lse = torch.zeros(alloc_final_rows, H, dtype=torch.float32, device=device)
    fout_slack_seed = 42.0
    final_output[store_slack_q:].fill_(fout_slack_seed)

    meta = {
        "logical_partial_rows": logical_partial_rows,
        "logical_final_rows": logical_final_rows,
        "gather_tile": gather_tile,
        "gather_q_row": gather_tile,
        "store_tile": store_tile,
        "store_slack_q": store_slack_q,
        "fout_slack_seed": fout_slack_seed,
    }
    return (
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        final_output,
        final_lse,
        meta,
    )


def build_serving_true_oob_inputs(
    H=16,
    Dv=512,
    out_dtype=torch.bfloat16,
    device="cuda",
    seed=0,
    num_tiles=64,
):
    """Fake-active tail with genuine OOB indices (guards-ON no-fault regression).

    Cannot be run with ``disable_guards=True`` in-process (would GPU-fault).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    active_tiles = 2
    splits_per_active = 32
    tail_splits = 4
    total_active = active_tiles * splits_per_active
    sentinel = total_active + tail_splits

    indptr_host = list(range(0, total_active + 1, splits_per_active))
    while len(indptr_host) < num_tiles:
        indptr_host.append(total_active)
    tail_tile = num_tiles - 2
    indptr_host[tail_tile + 1] = sentinel
    while len(indptr_host) <= num_tiles:
        indptr_host.append(sentinel)
    reduce_indptr = torch.tensor(indptr_host, dtype=torch.int32, device=device)

    num_partial_rows = 128
    partial_output = torch.randn(
        num_partial_rows, H, Dv, dtype=torch.float32, device=device, generator=g
    )
    partial_lse = (
        torch.randn(
            num_partial_rows, H, dtype=torch.float32, device=device, generator=g
        )
        * 2.0
    )

    reduce_partial_map = torch.arange(sentinel, dtype=torch.int32, device=device)
    tail_t0 = int(reduce_indptr[tail_tile].item())
    for i in range(tail_splits):
        reduce_partial_map[tail_t0 + i] = num_partial_rows + i

    reduce_final_map = torch.empty(num_tiles, 2, dtype=torch.int32, device=device)
    reduce_final_map[:active_tiles, 0] = torch.arange(
        active_tiles, dtype=torch.int32, device=device
    )
    reduce_final_map[:active_tiles, 1] = torch.arange(
        1, active_tiles + 1, dtype=torch.int32, device=device
    )
    reduce_final_map[active_tiles:tail_tile, 0] = 1 << 24
    reduce_final_map[active_tiles:tail_tile, 1] = (1 << 24) + 1
    reduce_final_map[tail_tile, 0] = 1 << 24
    reduce_final_map[tail_tile, 1] = (1 << 24) + 1
    reduce_final_map[tail_tile + 1 :, 0] = 1 << 24
    reduce_final_map[tail_tile + 1 :, 1] = (1 << 24) + 1

    final_output = torch.zeros(active_tiles, H, Dv, dtype=out_dtype, device=device)
    final_lse = torch.zeros(active_tiles, H, dtype=torch.float32, device=device)
    return (
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        final_output,
        final_lse,
    )


def build_serving_stale_indptr_inputs(
    H=16,
    Dv=512,
    out_dtype=torch.bfloat16,
    device="cuda",
    seed=0,
):
    """Batch-transition metadata: batch=8 sparse grid patched to batch=1 layout.

    Tile 0 becomes ``n_splits=128`` for batch=1; tiles 1..7 keep stale batch=8 CSR
    (``n_splits=32``, stale pmap rows beyond the logical bound, stale fmap q 1..7).
    Uses the allocation-slack trick so logical bounds are smaller than the buffers.

    Returns tensors plus ``meta`` with logical bounds for guard differentials.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    num_reduce_tile = SERVING_NUM_REDUCE_TILE
    active_tiles_batch8 = 8
    splits_per_active = 32
    batch1_splits = 128
    sentinel = batch1_splits + (active_tiles_batch8 - 1) * splits_per_active

    indptr_host = [0, batch1_splits]
    for t in range(1, active_tiles_batch8):
        indptr_host.append(indptr_host[-1] + splits_per_active)
    while len(indptr_host) <= num_reduce_tile:
        indptr_host.append(sentinel)
    reduce_indptr = torch.tensor(indptr_host, dtype=torch.int32, device=device)

    logical_partial_rows = batch1_splits
    slack_p = 256
    alloc_partial_rows = logical_partial_rows + slack_p
    logical_final_rows = 1
    slack_f = 8
    alloc_final_rows = logical_final_rows + slack_f

    partial_output = torch.randn(
        alloc_partial_rows, H, Dv, dtype=torch.float32, device=device, generator=g
    )
    partial_lse = (
        torch.randn(
            alloc_partial_rows, H, dtype=torch.float32, device=device, generator=g
        )
        * 2.0
    )

    total_pmap = sentinel
    reduce_partial_map = torch.arange(total_pmap, dtype=torch.int32, device=device)
    # Stale tiles 1..7: pmap rows from batch=8 now >= logical_partial_rows.
    stale_base = batch1_splits
    for t in range(1, active_tiles_batch8):
        t0 = indptr_host[t]
        for s in range(splits_per_active):
            reduce_partial_map[t0 + s] = stale_base + s
    partial_output[stale_base : stale_base + splits_per_active].fill_(500.0)
    tile1_lse_max = partial_lse[stale_base : stale_base + splits_per_active].max()
    partial_lse[stale_base].fill_(tile1_lse_max + 5.0)

    reduce_final_map = torch.empty(num_reduce_tile, 2, dtype=torch.int32, device=device)
    reduce_final_map[0, 0] = 0
    reduce_final_map[0, 1] = 1
    for t in range(1, active_tiles_batch8):
        reduce_final_map[t, 0] = t
        reduce_final_map[t, 1] = t + 1
    reduce_final_map[active_tiles_batch8:, 0] = 1 << 24
    reduce_final_map[active_tiles_batch8:, 1] = (1 << 24) + 1

    final_output = torch.zeros(alloc_final_rows, H, Dv, dtype=out_dtype, device=device)
    final_lse = torch.zeros(alloc_final_rows, H, dtype=torch.float32, device=device)
    fout_slack_seed = 42.0
    final_output[logical_final_rows:].fill_(fout_slack_seed)

    meta = {
        "logical_partial_rows": logical_partial_rows,
        "logical_final_rows": logical_final_rows,
        "gather_tile": 1,
        "gather_q_row": 1,
        "store_tile": 2,
        "store_slack_q": 2,
        "fout_slack_seed": fout_slack_seed,
    }
    return (
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        final_output,
        final_lse,
        meta,
    )


def make_runner(
    po,
    pl,
    indptr,
    pmap,
    fmap,
    fout,
    flse,
    H,
    Dv,
    out_dtype_str,
    output_lse,
    M=1,
    *,
    tier=None,
    disable_guards=False,
    num_partial_rows=None,
    num_final_rows=None,
):
    """Precompile + bind args; return a zero-overhead closure for the timed loop.

    tier: compile-time tier override for isolated tests. None (default) uses
    Tier.ALL (production path with device-side runtime tier selection).
    """
    num_tiles = fmap.shape[0]
    num_cu = torch.cuda.get_device_properties(0).multi_processor_count
    if tier is None:
        compile_tier = Tier.ALL
    else:
        compile_tier = tier
    if num_partial_rows is None:
        num_partial_rows = int(po.size(0))
    if num_final_rows is None:
        num_final_rows = int(fout.size(0))
    # Split-K (opt-in): cooperative multi-block reduction for the low-tile /
    # high-split decode case. Metadata is inspected at setup (outside any
    # CUDA-graph capture); the scratch is pre-allocated + reused, so the
    # captured run() only launches the two kernels (capture-safe).
    if splitk_enabled() and tier is None and not disable_guards:
        diffs = indptr[1:] - indptr[:-1]
        active_tiles = int((diffs > 1).sum().item())
        max_splits_val = int(diffs.max().item()) if diffs.numel() else 0
        engage, K, num_slots = plan_splitk(
            active_tiles=active_tiles,
            H=H,
            max_seqlen_q=M,
            max_splits=max_splits_val,
            num_cu=num_cu,
        )
        if engage:
            lp, lc = compile_mla_reduce_splitk(
                H=H,
                Dv=Dv,
                out_dtype=out_dtype_str,
                K=K,
                output_lse=output_lse,
                waves_per_eu=waves_per_eu_from_env(),
            )
            sk_acc, sk_ml = _get_splitk_scratch(num_slots, K, Dv, fout.device.index)
            _npr = int(num_partial_rows)
            _nfr = int(num_final_rows)
            _ss = int(fout.stride(0))
            _sh = int(fout.stride(1))
            _grid_p = int(num_slots * K)
            _grid_c = int(num_slots)

            def run():
                st = torch.cuda.current_stream()
                lp(po, pl, indptr, pmap, sk_acc, sk_ml, _npr, _grid_p, st)
                lc(fmap, sk_acc, sk_ml, fout, flse, _ss, _sh, _nfr, _grid_c, st)

            return run

    use_persistent = should_use_persistent_launch(
        H=H,
        max_seqlen_q=M,
        num_reduce_tile=num_tiles,
        num_cu=num_cu,
    )
    kernel = compile_mla_reduce(
        H=H,
        Dv=Dv,
        out_dtype=out_dtype_str,
        tier=compile_tier,
        persistent=use_persistent,
        output_lse=output_lse,
        use_reduce_final_map=True,
        disable_guards=disable_guards,
        waves_per_eu=waves_per_eu_from_env(),
    )
    head = (
        po,
        pl,
        indptr,
        pmap,
        fmap,
        fout,
        flse,
        int(fout.stride(0)),
        int(fout.stride(1)),
        int(num_cu),
        int(num_tiles),
        int(M),
        int(num_partial_rows),
        int(num_final_rows),
    )

    def run():
        kernel(*head, torch.cuda.current_stream())

    return run


def bench_cudagraph(fn, num_warmup=25, num_iters=100):
    """CUDA-graph replay timing; returns ms/iter."""
    for _ in range(max(1, num_warmup)):
        fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        fn()
        side.synchronize()
        with torch.cuda.graph(graph, stream=side):
            for _ in range(num_iters):
                fn()
    torch.cuda.current_stream().wait_stream(side)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / num_iters


def run_cudagraph_replay(fn, num_warmup=3, num_replays=3):
    """Capture ``fn`` into a CUDA graph and replay it, to surface replay-only
    faults (the failure mode reported under real serving). ``fn`` writes into its
    bound output tensors; the caller inspects those after this returns."""
    for _ in range(max(1, num_warmup)):
        fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        fn()
        side.synchronize()
        with torch.cuda.graph(graph, stream=side):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    for _ in range(max(1, num_replays)):
        graph.replay()
    torch.cuda.synchronize()


@contextmanager
def _env(**kwargs):
    """Set/unset environment variables for the duration of the block,
    restoring the prior state on exit (a pytest-free ``monkeypatch.setenv``/
    ``delenv``). Pass ``None`` for a var that should be unset."""
    sentinel = object()
    old = {k: os.environ.get(k, sentinel) for k in kwargs}
    try:
        for k, v in kwargs.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is sentinel:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# DeepSeek shape: HIP MLA_REDUCE_ROUTER has a Dv=512 template, so these compare
# against the HIP kernel directly.
_HIP_SHAPE = (128, 512)
# GLM-5.2 production shape (tp=8). HIP has no Dv=256 template, so these compare
# against the torch online-softmax reference.
_GLM_SHAPE = (8, 256)

# Irregular scenarios: (id, splits_per_tile, gap_stride, M).
_IRREGULAR_SCENARIOS = [
    ("tier_mismatch", [8, 304], 1, 1),  # tile 0 small, tile 1 forces MLDS tier
    ("variable_splits", [4, 32, 8, 64], 1, 1),  # mixed per-tile counts
    ("gapped_pmap", [8, 8, 8, 8], 4, 1),  # non-dense gather rows
    ("empty_middle", [8, 0, 16, 8], 1, 1),  # empty tile + garbage final map
    ("mlds_boundary", [300], 1, 1),  # MLDS tier just under cap
    ("mlds_max", [304], 1, 1),  # LDS_MAX_SPLITS
    ("mtp_irregular", [8, 32, 16], 2, 4),  # MTP (M>1) + gaps
    ("pool_oversize", [8, 304], 8, 1),  # large slack in partial pool
]

# fp16 (in addition to the bf16 default) only on the most layout-sensitive cases,
# to keep the matrix small.
_HIP_FP16_IDS = {"tier_mismatch", "gapped_pmap"}
_TORCH_FP16_IDS = {"tier_mismatch", "mlds_max"}


def _expand(fp16_ids):
    cases = []
    for name, spt, gap, M in _IRREGULAR_SCENARIOS:
        cases.append((name, spt, gap, M, "bf16"))
        if name in fp16_ids:
            cases.append((name, spt, gap, M, "fp16"))
    return cases


_HIP_CASES = _expand(_HIP_FP16_IDS)
_TORCH_CASES = _expand(_TORCH_FP16_IDS)
_HIP_IDS = [f"{n}_{dt}" for n, _, _, _, dt in _HIP_CASES]
_TORCH_IDS = [f"{n}_{dt}" for n, _, _, _, dt in _TORCH_CASES]

# Uniform/dense smoke: one tile count, M=1, bf16 only. Just enough to cover each
# compile tier on both reference paths.
_SMOKE_TILES = 4
_SMOKE_CASES = [
    (_HIP_SHAPE, "hip", 2),  # simple
    (_HIP_SHAPE, "hip", 8),  # m64
    (_HIP_SHAPE, "hip", 64),  # m64 upper
    (_HIP_SHAPE, "hip", 256),  # m256
    (_GLM_SHAPE, "torch", 2),  # simple (GLM)
    (_GLM_SHAPE, "torch", 8),  # m64 (GLM)
    (_GLM_SHAPE, "torch", 32),  # production split cap
    (_GLM_SHAPE, "torch", 256),  # stress
]
_SMOKE_IDS = [f"H{H}_Dv{Dv}_{ref}_s{S}" for (H, Dv), ref, S in _SMOKE_CASES]

# CUDA-graph replay: highest-risk irregular fixtures, to surface replay-only
# faults. (id, shape, ref, splits_per_tile, gap_stride, M).
_GRAPH_CASES = [
    ("tier_mismatch", _GLM_SHAPE, "torch", [8, 304], 1, 1),
    ("gapped_pmap", _HIP_SHAPE, "hip", [8, 8, 8, 8], 4, 1),
    ("empty_middle", _GLM_SHAPE, "torch", [8, 0, 16, 8], 1, 1),
    ("mlds_max", _GLM_SHAPE, "torch", [304], 1, 1),
    ("mtp_irregular", _GLM_SHAPE, "torch", [8, 32, 16], 2, 4),
]
_GRAPH_IDS = [c[0] for c in _GRAPH_CASES]

_DEGEN_TILES = [2, 4]


def _require_cuda():
    """``main()`` already gates on this before calling `run_checks()`; this is
    a defensive re-check for anyone importing/calling a check function
    directly (rule 7: functions stay independently importable)."""
    if not torch.cuda.is_available():
        raise RuntimeError("mla_reduce check requires CUDA")
    if get_gfx() not in MLA_REDUCE_SUPPORTED_GFX:
        raise RuntimeError(f"mla_reduce unsupported on {get_gfx()}")


def _assert_close(fout, flse, ref_out, ref_lse, dt):
    atol = _out_atol(dt)
    out_err = checkAllclose(
        ref_out.float(),
        fout.float(),
        rtol=0,
        atol=atol,
        msg=f"mla_reduce out ({dt})",
        printLog=False,
    )
    lse_err = checkAllclose(
        ref_lse.float(),
        flse.float(),
        rtol=0,
        atol=1e-3,
        msg=f"mla_reduce lse ({dt})",
        printLog=False,
    )
    assert out_err == 0, f"out mismatch ratio={out_err}"
    assert lse_err == 0, f"lse mismatch ratio={lse_err}"


def _masking_ref(po, pl, indptr, fmap, pmap, H, Dv, out_dtype, meta, M=1):
    return torch_ref_gather(
        po,
        pl,
        indptr,
        fmap,
        pmap,
        H,
        Dv,
        out_dtype,
        M,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )


def _run_guarded(
    po,
    pl,
    indptr,
    pmap,
    fmap,
    fout,
    flse,
    H,
    Dv,
    dt,
    meta,
    *,
    disable_guards=False,
    M=1,
):
    fout.zero_()
    flse.zero_()
    if meta.get("fout_slack_seed") is not None:
        sq = meta["store_slack_q"]
        fout[sq:].fill_(meta["fout_slack_seed"])
    run = make_runner(
        po,
        pl,
        indptr,
        pmap,
        fmap,
        fout,
        flse,
        H,
        Dv,
        dt,
        True,
        M,
        disable_guards=disable_guards,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )
    run()
    torch.cuda.synchronize()
    return fout.clone(), flse.clone()


def _assert_gather_differential(
    po, pl, indptr, fmap, pmap, fout, flse, H, Dv, dt, meta
):
    ref_out, ref_lse = _masking_ref(
        po, pl, indptr, fmap, pmap, H, Dv, _out_dtype(dt), meta
    )
    on_out, on_lse = _run_guarded(
        po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, meta, disable_guards=False
    )
    off_out, off_lse = _run_guarded(
        po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, meta, disable_guards=True
    )
    _assert_close(
        on_out[: meta["logical_final_rows"]],
        on_lse[: meta["logical_final_rows"]],
        ref_out[: meta["logical_final_rows"]],
        ref_lse[: meta["logical_final_rows"]],
        dt,
    )
    atol = _out_atol(dt)
    q_row = meta["gather_q_row"]
    gather_err = (off_out[q_row].float() - ref_out[q_row].float()).abs().max().item()
    assert gather_err > 5 * atol, (
        f"gather guard differential failed: guards-OFF row {q_row} "
        f"max_abs_err={gather_err:.3e} <= {5 * atol}"
    )


def _assert_store_differential(po, pl, indptr, fmap, pmap, fout, flse, H, Dv, dt, meta):
    ref_out, ref_lse = _masking_ref(
        po, pl, indptr, fmap, pmap, H, Dv, _out_dtype(dt), meta
    )
    on_out, on_lse = _run_guarded(
        po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, meta, disable_guards=False
    )
    off_out, _ = _run_guarded(
        po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, meta, disable_guards=True
    )
    _assert_close(
        on_out[: meta["logical_final_rows"]],
        on_lse[: meta["logical_final_rows"]],
        ref_out[: meta["logical_final_rows"]],
        ref_lse[: meta["logical_final_rows"]],
        dt,
    )
    atol = _out_atol(dt)
    sq = meta["store_slack_q"]
    seed = meta["fout_slack_seed"]
    on_slack_err = (on_out[sq:].float() - seed).abs().max().item()
    assert on_slack_err <= atol, f"guards-ON mutated slack: err={on_slack_err:.3e}"
    off_slack_err = (off_out[sq:].float() - seed).abs().max().item()
    assert off_slack_err > 5 * atol, (
        f"store guard differential failed: guards-OFF slack "
        f"max_abs_err={off_slack_err:.3e} <= {5 * atol}"
    )


def _run_irregular(spt, gap, M, H, Dv, dt):
    """Build irregular inputs, run the kernel, return (po, pl, indptr, fmap, pmap,
    fout, flse)."""
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = build_irregular_inputs(
        spt, H, Dv, out_dtype, M=M, gap_stride=gap
    )
    fout.zero_()
    flse.zero_()
    run = make_runner(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, True, M)
    run()
    torch.cuda.synchronize()
    return po, pl, indptr, fmap, pmap, fout, flse


def test_flydsl_mla_reduce_irregular_vs_hip(case):
    """Irregular metadata matches HIP kn_mla_reduce_v1 (DeepSeek shape, Dv=512)."""
    _require_cuda()
    name, spt, gap, M, dt = case
    H, Dv = _HIP_SHAPE
    po, pl, indptr, fmap, pmap, fout, flse = _run_irregular(spt, gap, M, H, Dv, dt)
    ref_out, ref_lse = hip_ref(
        po, pl, indptr, fmap, pmap, len(spt), H, Dv, _out_dtype(dt), M
    )
    _assert_close(fout, flse, ref_out, ref_lse, dt)


def test_flydsl_mla_reduce_irregular_vs_torch_ref(case):
    """Irregular metadata matches the gather-based torch ref (GLM-5.2, Dv=256)."""
    _require_cuda()
    name, spt, gap, M, dt = case
    H, Dv = _GLM_SHAPE
    po, pl, indptr, fmap, pmap, fout, flse = _run_irregular(spt, gap, M, H, Dv, dt)
    ref_out, ref_lse = torch_ref_gather(
        po, pl, indptr, fmap, pmap, H, Dv, _out_dtype(dt), M
    )
    _assert_close(fout, flse, ref_out, ref_lse, dt)


def test_flydsl_mla_reduce_uniform_smoke(case):
    """Dense/uniform smoke: each compile tier on both reference paths."""
    _require_cuda()
    (H, Dv), ref, S = case
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = build_inputs(
        _SMOKE_TILES, S, H, Dv, out_dtype
    )
    fout.zero_()
    flse.zero_()
    run = make_runner(
        po,
        pl,
        indptr,
        pmap,
        fmap,
        fout,
        flse,
        H,
        Dv,
        dt,
        True,
        tier=select_tier(S),
    )
    run()
    torch.cuda.synchronize()
    if ref == "hip":
        ref_out, ref_lse = hip_ref(
            po, pl, indptr, fmap, pmap, _SMOKE_TILES, H, Dv, out_dtype
        )
    else:
        ref_out, ref_lse = torch_ref(po, pl, _SMOKE_TILES, S, H, Dv, out_dtype)
    _assert_close(fout, flse, ref_out, ref_lse, dt)


def test_flydsl_mla_reduce_cudagraph_replay(case):
    """Irregular metadata stays correct under CUDA-graph capture + replay (the
    serving failure mode); no GPU fault and output matches the reference."""
    _require_cuda()
    name, (H, Dv), ref, spt, gap, M = case
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = build_irregular_inputs(
        spt, H, Dv, out_dtype, M=M, gap_stride=gap
    )
    fout.zero_()
    flse.zero_()
    run = make_runner(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, True, M)
    run_cudagraph_replay(run)
    if ref == "hip":
        ref_out, ref_lse = hip_ref(
            po, pl, indptr, fmap, pmap, len(spt), H, Dv, out_dtype, M
        )
    else:
        ref_out, ref_lse = torch_ref_gather(
            po, pl, indptr, fmap, pmap, H, Dv, out_dtype, M
        )
    _assert_close(fout, flse, ref_out, ref_lse, dt)


def test_flydsl_mla_reduce_small_split_cudagraph_replay():
    """Uniform 32-split tiles under cudagraph replay (runtime NLSE=1 path)."""
    _require_cuda()
    H, Dv = _GLM_SHAPE
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    spt = [32] * 4
    po, pl, indptr, fmap, pmap, fout, flse = build_irregular_inputs(
        spt, H, Dv, out_dtype, M=1, gap_stride=1
    )
    fout.zero_()
    flse.zero_()
    run = make_runner(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, True, M=1)
    run_cudagraph_replay(run)
    ref_out, ref_lse = torch_ref_gather(
        po, pl, indptr, fmap, pmap, H, Dv, out_dtype, M=1
    )
    _assert_close(fout, flse, ref_out, ref_lse, dt)


_SERVING_SHAPE = (16, 512)


def test_serving_sparse_grid_vs_hip():
    """batch=8 layout: 16384-tile grid, 8 active tiles, garbage tail."""
    _require_cuda()
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = build_serving_sparse_grid_inputs(
        *_SERVING_SHAPE, out_dtype=out_dtype
    )
    fout.zero_()
    flse.zero_()
    run = make_runner(po, pl, indptr, pmap, fmap, fout, flse, *_SERVING_SHAPE, dt, True)
    run()
    torch.cuda.synchronize()
    ref_out, ref_lse = hip_ref_like_fout(po, pl, indptr, fmap, pmap, fout, flse)
    _assert_close(fout, flse, ref_out, ref_lse, dt)


def test_serving_sparse_grid_cudagraph_replay():
    """16384-tile serving grid under CUDA-graph replay (prod failure mode)."""
    _require_cuda()
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = build_serving_sparse_grid_inputs(
        *_SERVING_SHAPE, out_dtype=out_dtype
    )
    fout.zero_()
    flse.zero_()
    run = make_runner(po, pl, indptr, pmap, fmap, fout, flse, *_SERVING_SHAPE, dt, True)
    run_cudagraph_replay(run)
    ref_out, ref_lse = hip_ref_like_fout(po, pl, indptr, fmap, pmap, fout, flse)
    _assert_close(fout, flse, ref_out, ref_lse, dt)


# ---------------------------------------------------------------------------
# Split-K cooperative reduction (opt-in): low-tile / high-split decode case.
# b1_s128 = 1 active tile x H=16 heads = 16 active blocks / 304 CUs, each
# serially reducing 128 splits. Split-K fans each head's 128 splits across K
# blocks (partial online-softmax) + a cheap combine.
# ---------------------------------------------------------------------------
_SPLITK_H, _SPLITK_DV = 16, 512
_SPLITK_GRID = 16384


def _build_splitk_b1_s128(out_dtype):
    """1 active tile x 128 splits in a 16384-tile serving grid (b1_s128)."""
    spt = [128] + [0] * (_SPLITK_GRID - 1)
    return build_irregular_inputs(
        spt, _SPLITK_H, _SPLITK_DV, out_dtype, M=1, gap_stride=1, pool_slack=0
    )


def _assert_splitk_engages(indptr):
    from aiter.ops.flydsl.kernels.mla_reduce import plan_splitk

    diffs = indptr[1:] - indptr[:-1]
    engage, K, num_slots = plan_splitk(
        active_tiles=int((diffs > 1).sum().item()),
        H=_SPLITK_H,
        max_seqlen_q=1,
        max_splits=int(diffs.max().item()),
        num_cu=304,
    )
    assert engage, "split-K did not engage for b1_s128 (test is meaningless)"
    return K, num_slots


def test_splitk_b1_s128_vs_torch_ref(K):
    """Split-K partial+combine matches the gather-based torch reference for the
    low-tile/high-split decode case, across split factors K."""
    _require_cuda()
    with _env(AITER_MLA_REDUCE_SPLITK="1", MLA_SPLITK_FACTOR=str(K)):
        dt = "bf16"
        out_dtype = _out_dtype(dt)
        po, pl, indptr, fmap, pmap, fout, flse = _build_splitk_b1_s128(out_dtype)
        _assert_splitk_engages(indptr)
        fout.zero_()
        flse.zero_()
        run = make_runner(
            po,
            pl,
            indptr,
            pmap,
            fmap,
            fout,
            flse,
            _SPLITK_H,
            _SPLITK_DV,
            dt,
            True,
            tier=None,
        )
        run()
        torch.cuda.synchronize()
        ref_out, ref_lse = torch_ref_gather(
            po, pl, indptr, fmap, pmap, _SPLITK_H, _SPLITK_DV, out_dtype, 1
        )
        _assert_close(fout, flse, ref_out, ref_lse, dt)


def test_splitk_b1_s128_vs_hip():
    """Split-K matches the production HIP kn_mla_reduce_v1 (Dv=512 template)."""
    _require_cuda()
    with _env(AITER_MLA_REDUCE_SPLITK="1"):
        dt = "bf16"
        out_dtype = _out_dtype(dt)
        po, pl, indptr, fmap, pmap, fout, flse = _build_splitk_b1_s128(out_dtype)
        _assert_splitk_engages(indptr)
        fout.zero_()
        flse.zero_()
        run = make_runner(
            po,
            pl,
            indptr,
            pmap,
            fmap,
            fout,
            flse,
            _SPLITK_H,
            _SPLITK_DV,
            dt,
            True,
            tier=None,
        )
        run()
        torch.cuda.synchronize()
        ref_out, ref_lse = hip_ref_like_fout(po, pl, indptr, fmap, pmap, fout, flse)
        _assert_close(fout, flse, ref_out, ref_lse, dt)


def test_splitk_b1_s128_cudagraph_replay():
    """Split-K (2-kernel) stays correct under CUDA-graph capture + replay, with
    a pre-allocated scratch buffer."""
    _require_cuda()
    with _env(AITER_MLA_REDUCE_SPLITK="1"):
        dt = "bf16"
        out_dtype = _out_dtype(dt)
        po, pl, indptr, fmap, pmap, fout, flse = _build_splitk_b1_s128(out_dtype)
        _assert_splitk_engages(indptr)
        fout.zero_()
        flse.zero_()
        run = make_runner(
            po,
            pl,
            indptr,
            pmap,
            fmap,
            fout,
            flse,
            _SPLITK_H,
            _SPLITK_DV,
            dt,
            True,
            tier=None,
        )
        run_cudagraph_replay(run)
        ref_out, ref_lse = torch_ref_gather(
            po, pl, indptr, fmap, pmap, _SPLITK_H, _SPLITK_DV, out_dtype, 1
        )
        _assert_close(fout, flse, ref_out, ref_lse, dt)


def test_splitk_disabled_by_default():
    """Without the env flag, plan_splitk never engages (default path untouched)."""
    from aiter.ops.flydsl.kernels.mla_reduce import plan_splitk, splitk_enabled

    with _env(AITER_MLA_REDUCE_SPLITK=None):
        assert not splitk_enabled()
        engage, _, _ = plan_splitk(
            active_tiles=1, H=16, max_seqlen_q=1, max_splits=128, num_cu=304
        )
        assert not engage


# ---------------------------------------------------------------------------
# Device-adaptive, capture-safe, default-able split-K (through the production
# wrapper flydsl_mla_reduce_v1). Unlike the opt-in plan_splitk above, the plan is
# taken from HOST-only values (final_output.size(0), num_kv_splits, num_cu) with
# no device read, so it can engage under CUDA-graph capture and is on by default.
# ---------------------------------------------------------------------------


def _build_da_single_tile(out_dtype, splits, pool=304):
    """One active tile (bs=1: final_output has a single row) with `splits`
    splits, partial pool sized to `pool` so the split count can be MUTATED up to
    `pool` across CUDA-graph replays."""
    po, pl, indptr, fmap, pmap, fout, flse = build_irregular_inputs(
        [splits],
        _SPLITK_H,
        _SPLITK_DV,
        out_dtype,
        M=1,
        gap_stride=1,
        pool_slack=pool - splits,
    )
    pmap = torch.arange(pool, dtype=torch.int32, device=pmap.device)
    return po, pl, indptr, fmap, pmap, fout, flse


def test_da_splitk_no_engage_large_batch():
    """Default-safe: a saturated grid (base_slots >= num_cu) never engages, so
    large-batch decode keeps the unchanged single-kernel path."""
    from aiter.ops.flydsl.kernels.mla_reduce import plan_splitk_capture_safe

    engage, _, _ = plan_splitk_capture_safe(
        num_final_rows=32, H=16, max_seqlen_q=1, num_kv_splits=128, num_cu=304
    )
    assert not engage


def test_da_splitk_no_engage_low_splits():
    """Default-safe: too few splits (num_kv_splits < min) never engages, so the
    combine kernel is not paid on short-context / low-split decode."""
    from aiter.ops.flydsl.kernels.mla_reduce import plan_splitk_capture_safe

    engage, _, _ = plan_splitk_capture_safe(
        num_final_rows=1, H=16, max_seqlen_q=1, num_kv_splits=32, num_cu=304
    )
    assert not engage


def test_da_splitk_wrapper_vs_hip():
    """The default-able wrapper path (DA split-K on) matches HIP for b1_s128."""
    _require_cuda()
    from aiter.ops.flydsl import flydsl_mla_reduce_v1

    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = _build_da_single_tile(
        out_dtype, 128, pool=128
    )
    fout.zero_()
    flse.zero_()

    def run():
        flydsl_mla_reduce_v1(
            po, pl, indptr, fmap, pmap, 1, fout, flse, num_kv_splits=128
        )

    run_cudagraph_replay(run)
    ref_out, ref_lse = hip_ref_like_fout(po, pl, indptr, fmap, pmap, fout, flse)
    _assert_close(fout, flse, ref_out, ref_lse, dt)


def test_da_splitk_capture_safe_varying_splits():
    """One CUDA-graph capture (bs=1, grid/K/scratch baked from host num_kv_splits)
    stays correct across replays whose per-tile split count changes on-device."""
    _require_cuda()
    from aiter.ops.flydsl import flydsl_mla_reduce_v1
    from aiter.ops.flydsl.kernels.mla_reduce import plan_splitk_capture_safe

    dt = "bf16"
    out_dtype = _out_dtype(dt)
    pool = 304
    nkv = 128
    po, pl, indptr, fmap, pmap, fout, flse = _build_da_single_tile(
        out_dtype, pool, pool=pool
    )
    engage, K, slots = plan_splitk_capture_safe(
        num_final_rows=1,
        H=_SPLITK_H,
        max_seqlen_q=1,
        num_kv_splits=nkv,
        num_cu=304,
    )
    assert engage and slots == _SPLITK_H, "DA split-K must engage for bs=1"

    def run():
        flydsl_mla_reduce_v1(
            po, pl, indptr, fmap, pmap, 1, fout, flse, num_kv_splits=nkv
        )

    # Warm up, then capture ONCE (grid/K/scratch fixed for this bs=1 bucket).
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        run()
        side.synchronize()
        with torch.cuda.graph(graph, stream=side):
            run()
    torch.cuda.current_stream().wait_stream(side)

    # Replay the SAME graph with DIFFERENT per-tile split counts (bs fixed): the
    # device reads the mutated CSR each replay and adapts its K allocation.
    for s_k in [128, 304, 64, 200, 8, 96]:
        indptr[1] = s_k  # mutate the captured CSR in place (host->device copy)
        fout.zero_()
        flse.zero_()
        graph.replay()
        torch.cuda.synchronize()
        ref_out, ref_lse = torch_ref_gather(
            po, pl, indptr, fmap, pmap, _SPLITK_H, _SPLITK_DV, out_dtype, 1
        )
        _assert_close(fout, flse, ref_out, ref_lse, dt)


# ---------------------------------------------------------------------------
# actual_max_splits gate
# ---------------------------------------------------------------------------


def test_actual_max_splits_gate_loose_budget():
    """Loose num_kv_splits budget (304) with small actual_max_splits does not
    engage; true high actual_max_splits does."""
    from aiter.ops.flydsl.kernels.mla_reduce import plan_splitk_capture_safe

    engage_loose, _, _ = plan_splitk_capture_safe(
        num_final_rows=1,
        H=16,
        max_seqlen_q=1,
        num_kv_splits=304,
        num_cu=304,
        actual_max_splits=8,
    )
    assert not engage_loose

    engage_hi, K, slots = plan_splitk_capture_safe(
        num_final_rows=1,
        H=16,
        max_seqlen_q=1,
        num_kv_splits=304,
        num_cu=304,
        actual_max_splits=128,
    )
    assert engage_hi and K == 16 and slots == 16


def test_derive_actual_max_splits():
    """Helper matches CSR max tile width."""
    from aiter.ops.flydsl.kernels.mla_reduce import derive_actual_max_splits

    indptr = torch.tensor([0, 8, 12, 12], dtype=torch.int32, device="cuda")
    assert derive_actual_max_splits(indptr) == 8


def test_actual_max_splits_wrapper_loose_budget_correct():
    """Loose budget (304) + small actual splits stays on single-kernel path and
    matches the torch reference."""
    _require_cuda()
    from aiter.ops.flydsl import flydsl_mla_reduce_v1
    from aiter.ops.flydsl.kernels.mla_reduce import (
        derive_actual_max_splits,
        plan_splitk_capture_safe,
    )

    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = _build_da_single_tile(out_dtype, 8, pool=8)
    fout = fout[:1].contiguous()
    flse = flse[:1].contiguous()
    actual = derive_actual_max_splits(indptr)
    assert actual == 8
    engage, _, _ = plan_splitk_capture_safe(
        num_final_rows=1,
        H=_SPLITK_H,
        max_seqlen_q=1,
        num_kv_splits=304,
        num_cu=304,
        actual_max_splits=actual,
    )
    assert not engage

    fout.zero_()
    flse.zero_()

    def run():
        flydsl_mla_reduce_v1(
            po,
            pl,
            indptr,
            fmap,
            pmap,
            1,
            fout,
            flse,
            num_kv_splits=304,
            actual_max_splits=actual,
        )

    run()
    torch.cuda.synchronize()
    ref_out, ref_lse = torch_ref_gather(
        po, pl, indptr, fmap, pmap, _SPLITK_H, _SPLITK_DV, out_dtype, 1
    )
    _assert_close(fout, flse, ref_out, ref_lse, dt)


def test_actual_max_splits_wrapper_cudagraph_replay():
    """Loose budget + actual_max_splits gate stays correct under graph replay."""
    _require_cuda()
    from aiter.ops.flydsl import flydsl_mla_reduce_v1
    from aiter.ops.flydsl.kernels.mla_reduce import derive_actual_max_splits

    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = _build_da_single_tile(
        out_dtype, 128, pool=128
    )
    fout = fout[:1].contiguous()
    flse = flse[:1].contiguous()
    actual = derive_actual_max_splits(indptr)
    assert actual == 128
    fout.zero_()
    flse.zero_()

    def run():
        flydsl_mla_reduce_v1(
            po,
            pl,
            indptr,
            fmap,
            pmap,
            1,
            fout,
            flse,
            num_kv_splits=304,
            actual_max_splits=actual,
        )

    run_cudagraph_replay(run)
    ref_out, ref_lse = torch_ref_gather(
        po, pl, indptr, fmap, pmap, _SPLITK_H, _SPLITK_DV, out_dtype, 1
    )
    _assert_close(fout, flse, ref_out, ref_lse, dt)


def test_serving_stale_indptr_cudagraph_replay():
    """Cudagraph replay after batch-8→batch-1 layout (guards-ON/OFF differential)."""
    _require_cuda()
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse, meta = build_serving_stale_indptr_inputs(
        *_SERVING_SHAPE, out_dtype=out_dtype
    )
    ref_out, ref_lse = _masking_ref(
        po, pl, indptr, fmap, pmap, *_SERVING_SHAPE, out_dtype, meta
    )
    on_out = fout.clone()
    on_lse = flse.clone()
    run_on = make_runner(
        po,
        pl,
        indptr,
        pmap,
        fmap,
        on_out,
        on_lse,
        *_SERVING_SHAPE,
        dt,
        True,
        disable_guards=False,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )
    on_out.zero_()
    on_lse.zero_()
    on_out[meta["logical_final_rows"] :].fill_(meta["fout_slack_seed"])
    run_cudagraph_replay(run_on)
    off_out = fout.clone()
    off_lse = flse.clone()
    run_off = make_runner(
        po,
        pl,
        indptr,
        pmap,
        fmap,
        off_out,
        off_lse,
        *_SERVING_SHAPE,
        dt,
        True,
        disable_guards=True,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )
    off_out.zero_()
    off_lse.zero_()
    off_out[meta["logical_final_rows"] :].fill_(meta["fout_slack_seed"])
    run_cudagraph_replay(run_off)
    _assert_close(
        on_out[: meta["logical_final_rows"]],
        on_lse[: meta["logical_final_rows"]],
        ref_out[: meta["logical_final_rows"]],
        ref_lse[: meta["logical_final_rows"]],
        dt,
    )
    atol = _out_atol(dt)
    q_row = meta["gather_q_row"]
    gather_err = (off_out[q_row].float() - ref_out[q_row].float()).abs().max().item()
    assert gather_err > 5 * atol
    sq = meta["store_slack_q"]
    off_slack_err = (off_out[sq:].float() - meta["fout_slack_seed"]).abs().max().item()
    assert off_slack_err > 5 * atol


def test_serving_gather_guard_differential():
    """Mapped-slack gather: guards-OFF miscompares, guards-ON matches masking ref."""
    _require_cuda()
    dt = "bf16"
    H, Dv = _SERVING_SHAPE
    po, pl, indptr, fmap, pmap, fout, flse, meta = build_serving_mapped_slack_inputs(
        H, Dv, _out_dtype(dt)
    )
    _assert_gather_differential(po, pl, indptr, fmap, pmap, fout, flse, H, Dv, dt, meta)


def test_serving_store_guard_differential():
    """Mapped-slack store: guards-OFF writes slack, guards-ON preserves seed."""
    _require_cuda()
    dt = "bf16"
    H, Dv = _SERVING_SHAPE
    po, pl, indptr, fmap, pmap, fout, flse, meta = build_serving_mapped_slack_inputs(
        H, Dv, _out_dtype(dt)
    )
    _assert_store_differential(po, pl, indptr, fmap, pmap, fout, flse, H, Dv, dt, meta)


def test_serving_gather_guard_differential_cudagraph_replay():
    """Gather guard differential holds under CUDA-graph capture/replay."""
    _require_cuda()
    dt = "bf16"
    H, Dv = _SERVING_SHAPE
    po, pl, indptr, fmap, pmap, fout, flse, meta = build_serving_mapped_slack_inputs(
        H, Dv, _out_dtype(dt)
    )
    ref_out, ref_lse = _masking_ref(
        po, pl, indptr, fmap, pmap, H, Dv, _out_dtype(dt), meta
    )
    on_out = fout.clone()
    on_lse = flse.clone()
    run_on = make_runner(
        po,
        pl,
        indptr,
        pmap,
        fmap,
        on_out,
        on_lse,
        H,
        Dv,
        dt,
        True,
        disable_guards=False,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )
    on_out.zero_()
    on_lse.zero_()
    on_out[meta["store_slack_q"] :].fill_(meta["fout_slack_seed"])
    run_cudagraph_replay(run_on)
    off_out = fout.clone()
    off_lse = flse.clone()
    run_off = make_runner(
        po,
        pl,
        indptr,
        pmap,
        fmap,
        off_out,
        off_lse,
        H,
        Dv,
        dt,
        True,
        disable_guards=True,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )
    off_out.zero_()
    off_lse.zero_()
    off_out[meta["store_slack_q"] :].fill_(meta["fout_slack_seed"])
    run_cudagraph_replay(run_off)
    _assert_close(
        on_out[: meta["logical_final_rows"]],
        on_lse[: meta["logical_final_rows"]],
        ref_out[: meta["logical_final_rows"]],
        ref_lse[: meta["logical_final_rows"]],
        dt,
    )
    atol = _out_atol(dt)
    q_row = meta["gather_q_row"]
    gather_err = (off_out[q_row].float() - ref_out[q_row].float()).abs().max().item()
    assert gather_err > 5 * atol


def test_serving_store_guard_differential_cudagraph_replay():
    """Store guard differential holds under CUDA-graph capture/replay."""
    _require_cuda()
    dt = "bf16"
    H, Dv = _SERVING_SHAPE
    po, pl, indptr, fmap, pmap, fout, flse, meta = build_serving_mapped_slack_inputs(
        H, Dv, _out_dtype(dt)
    )
    ref_out, ref_lse = _masking_ref(
        po, pl, indptr, fmap, pmap, H, Dv, _out_dtype(dt), meta
    )
    on_out = fout.clone()
    on_lse = flse.clone()
    run_on = make_runner(
        po,
        pl,
        indptr,
        pmap,
        fmap,
        on_out,
        on_lse,
        H,
        Dv,
        dt,
        True,
        disable_guards=False,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )
    on_out.zero_()
    on_lse.zero_()
    on_out[meta["store_slack_q"] :].fill_(meta["fout_slack_seed"])
    run_cudagraph_replay(run_on)
    off_out = fout.clone()
    run_off = make_runner(
        po,
        pl,
        indptr,
        pmap,
        fmap,
        off_out,
        flse.clone(),
        H,
        Dv,
        dt,
        True,
        disable_guards=True,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )
    off_out.zero_()
    off_out[meta["store_slack_q"] :].fill_(meta["fout_slack_seed"])
    run_cudagraph_replay(run_off)
    _assert_close(
        on_out[: meta["logical_final_rows"]],
        on_lse[: meta["logical_final_rows"]],
        ref_out[: meta["logical_final_rows"]],
        ref_lse[: meta["logical_final_rows"]],
        dt,
    )
    atol = _out_atol(dt)
    sq = meta["store_slack_q"]
    off_slack_err = (off_out[sq:].float() - meta["fout_slack_seed"]).abs().max().item()
    assert off_slack_err > 5 * atol


def test_serving_true_oob_no_fault():
    """Genuine OOB indices: guards-ON only (guards-OFF would abort the process)."""
    _require_cuda()
    dt = "bf16"
    H, Dv = _SERVING_SHAPE
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = build_serving_true_oob_inputs(
        H, Dv, out_dtype
    )
    fout.zero_()
    flse.zero_()
    run = make_runner(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, True)
    run()
    torch.cuda.synchronize()
    ref_out, ref_lse = torch_ref_gather(
        po,
        pl,
        indptr,
        fmap,
        pmap,
        H,
        Dv,
        out_dtype,
        num_partial_rows=po.size(0),
        num_final_rows=fout.size(0),
    )
    _assert_close(fout, flse, ref_out[: fout.size(0)], ref_lse[: fout.size(0)], dt)


def test_flydsl_mla_reduce_degenerate_empty_tile(num_tiles):
    """Empty-tile guard: all-empty (n_splits=0) metadata never stores through the
    garbage q-ranges, leaving the output untouched."""
    _require_cuda()
    H, Dv = _GLM_SHAPE
    out_dtype = _out_dtype("bf16")
    po, pl, indptr, fmap, pmap, fout, flse = build_degenerate_inputs(
        num_tiles, H, Dv, out_dtype
    )
    fout.fill_(12345.0)
    flse.fill_(12345.0)
    expected_out = fout.clone()
    expected_lse = flse.clone()
    run = make_runner(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, "bf16", True)
    run()
    torch.cuda.synchronize()
    assert torch.equal(fout, expected_out)
    assert torch.equal(flse, expected_lse)


def test_dispatch_does_not_thread_actual_max_splits():
    """mla_decode_fwd and _mla_reduce_v1_dispatch do not accept or forward
    actual_max_splits; the FlyDSL wrapper auto-resolves it from reduce_indptr
    (capture-safe warmup cache)."""
    import inspect

    import aiter.mla as mla
    import aiter.ops.flydsl as flydsl

    assert (
        "actual_max_splits"
        not in inspect.signature(mla._mla_reduce_v1_dispatch).parameters
    )
    assert "actual_max_splits" not in inspect.signature(mla.mla_decode_fwd).parameters

    # _flydsl_mla_reduce_enabled() re-reads the env var on every call (no
    # lru_cache on the gate itself), so _env() takes effect immediately.
    with _env(AITER_MLA_REDUCE_FLYDSL="1"):
        captured = {}

        def _capture(*args, **kwargs):
            captured["kwargs"] = kwargs

        with mock.patch.object(flydsl, "flydsl_mla_reduce_v1", _capture):
            mla._mla_reduce_v1_dispatch(None, None, None, None, None, 1, 0, None, None)
        assert "actual_max_splits" not in captured["kwargs"]
        assert captured["kwargs"].get("num_kv_splits") == 0


def test_resolve_actual_max_splits_eager_and_capture():
    """The warmup cache resolves the true max split eager, then serves the same
    value under CUDA-graph capture (no device sync), and misses -> None."""
    _require_cuda()
    import torch

    from aiter.ops.flydsl.mla_reduce_kernels import (
        _ACTUAL_MAX_SPLITS_CACHE,
        _resolve_actual_max_splits,
    )
    from aiter.ops.flydsl.kernels.mla_reduce import derive_actual_max_splits

    # CSR with per-tile widths {5, 8, 3} -> max 8.
    indptr = torch.tensor([0, 5, 13, 16], dtype=torch.int32, device="cuda")
    _ACTUAL_MAX_SPLITS_CACHE.clear()

    eager = _resolve_actual_max_splits(indptr)
    assert eager == derive_actual_max_splits(indptr) == 8

    # Simulate capture: value served from cache (buffer identity), no sync.
    key = (indptr.data_ptr(), int(indptr.numel()))
    assert key in _ACTUAL_MAX_SPLITS_CACHE

    # A never-seen buffer under capture -> miss -> None (safe degrade).
    other = torch.tensor([0, 4, 4], dtype=torch.int32, device="cuda")
    okey = (other.data_ptr(), int(other.numel()))
    _ACTUAL_MAX_SPLITS_CACHE.pop(okey, None)
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=side):
        miss = _resolve_actual_max_splits(other)
    assert miss is None


# ---------------------------------------------------------------------------
# Adaptive launch: one block per active (tile, head) instead of the persistent
# grid-stride kernel.
# ---------------------------------------------------------------------------
_ADAPTIVE_SCENARIOS = [
    ("b8_s32", 8, 32),
    ("b8_s13", 8, 13),
    ("b8_s6", 8, 6),
    ("b8_s2", 8, 2),
    ("b1_s32", 1, 32),
]


def test_adaptive_launch_wrapper_vs_hip(label, active, splits):
    """Adaptive launch (split-K off) matches HIP on the serving decode shapes."""
    _require_cuda()
    from aiter.ops.flydsl import flydsl_mla_reduce_v1

    dt = "bf16"
    po, pl, indptr, fmap, pmap, fout, flse = build_serving_decode_inputs(
        active, splits, _out_dtype(dt)
    )
    fout.zero_()
    flse.zero_()

    def run():
        flydsl_mla_reduce_v1(
            po, pl, indptr, fmap, pmap, 1, fout, flse, num_kv_splits=splits
        )

    run()
    torch.cuda.synchronize()
    ref_out, ref_lse = hip_ref_like_fout(po, pl, indptr, fmap, pmap, fout, flse)
    _assert_close(fout, flse, ref_out, ref_lse, dt)


def test_adaptive_launch_cudagraph_replay():
    """Adaptive launch stays correct under CUDA-graph capture/replay (b8_s32)."""
    _require_cuda()
    from aiter.ops.flydsl import flydsl_mla_reduce_v1

    dt = "bf16"
    po, pl, indptr, fmap, pmap, fout, flse = build_serving_decode_inputs(
        8, 32, _out_dtype(dt)
    )
    fout.zero_()
    flse.zero_()

    def run():
        flydsl_mla_reduce_v1(
            po, pl, indptr, fmap, pmap, 1, fout, flse, num_kv_splits=32
        )

    run_cudagraph_replay(run)
    ref_out, ref_lse = hip_ref_like_fout(po, pl, indptr, fmap, pmap, fout, flse)
    _assert_close(fout, flse, ref_out, ref_lse, dt)


def test_adaptive_launch_single_tile_uses_persistent():
    """bs=1 (num_final_rows==1) must not engage adaptive; still matches HIP."""
    _require_cuda()
    from aiter.ops.flydsl import flydsl_mla_reduce_v1

    dt = "bf16"
    po, pl, indptr, fmap, pmap, fout, flse = build_serving_decode_inputs(
        1, 32, _out_dtype(dt)
    )
    fout.zero_()
    flse.zero_()

    def run():
        flydsl_mla_reduce_v1(
            po, pl, indptr, fmap, pmap, 1, fout, flse, num_kv_splits=32
        )

    run()
    torch.cuda.synchronize()
    ref_out, ref_lse = hip_ref_like_fout(po, pl, indptr, fmap, pmap, fout, flse)
    _assert_close(fout, flse, ref_out, ref_lse, dt)


def run_checks():
    """Run every invariant/correctness check. Returns a list of ``(name, exc)``
    for any that failed; an empty list means everything passed. Aggregates
    failures instead of stopping at the first one, mirroring a full pytest
    report without depending on pytest."""
    failures = []

    def _run(name, fn, *args):
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001 - collect every failure, keep going
            failures.append((name, exc))

    for case in _HIP_CASES:
        _run(
            f"irregular_vs_hip[{case[0]}_{case[4]}]",
            test_flydsl_mla_reduce_irregular_vs_hip,
            case,
        )
    for case in _TORCH_CASES:
        _run(
            f"irregular_vs_torch_ref[{case[0]}_{case[4]}]",
            test_flydsl_mla_reduce_irregular_vs_torch_ref,
            case,
        )
    for case in _SMOKE_CASES:
        (H, Dv), ref, S = case
        _run(
            f"uniform_smoke[H{H}_Dv{Dv}_{ref}_s{S}]",
            test_flydsl_mla_reduce_uniform_smoke,
            case,
        )
    for case in _GRAPH_CASES:
        _run(
            f"cudagraph_replay[{case[0]}]",
            test_flydsl_mla_reduce_cudagraph_replay,
            case,
        )
    _run(
        "small_split_cudagraph_replay",
        test_flydsl_mla_reduce_small_split_cudagraph_replay,
    )
    _run("serving_sparse_grid_vs_hip", test_serving_sparse_grid_vs_hip)
    _run(
        "serving_sparse_grid_cudagraph_replay",
        test_serving_sparse_grid_cudagraph_replay,
    )
    for K in [4, 8, 16]:
        _run(f"splitk_b1_s128_vs_torch_ref[K={K}]", test_splitk_b1_s128_vs_torch_ref, K)
    _run("splitk_b1_s128_vs_hip", test_splitk_b1_s128_vs_hip)
    _run("splitk_b1_s128_cudagraph_replay", test_splitk_b1_s128_cudagraph_replay)
    _run("splitk_disabled_by_default", test_splitk_disabled_by_default)
    _run("da_splitk_no_engage_large_batch", test_da_splitk_no_engage_large_batch)
    _run("da_splitk_no_engage_low_splits", test_da_splitk_no_engage_low_splits)
    _run("da_splitk_wrapper_vs_hip", test_da_splitk_wrapper_vs_hip)
    _run(
        "da_splitk_capture_safe_varying_splits",
        test_da_splitk_capture_safe_varying_splits,
    )
    _run(
        "actual_max_splits_gate_loose_budget", test_actual_max_splits_gate_loose_budget
    )
    _run("derive_actual_max_splits", test_derive_actual_max_splits)
    _run(
        "actual_max_splits_wrapper_loose_budget_correct",
        test_actual_max_splits_wrapper_loose_budget_correct,
    )
    _run(
        "actual_max_splits_wrapper_cudagraph_replay",
        test_actual_max_splits_wrapper_cudagraph_replay,
    )
    _run(
        "serving_stale_indptr_cudagraph_replay",
        test_serving_stale_indptr_cudagraph_replay,
    )
    _run("serving_gather_guard_differential", test_serving_gather_guard_differential)
    _run("serving_store_guard_differential", test_serving_store_guard_differential)
    _run(
        "serving_gather_guard_differential_cudagraph_replay",
        test_serving_gather_guard_differential_cudagraph_replay,
    )
    _run(
        "serving_store_guard_differential_cudagraph_replay",
        test_serving_store_guard_differential_cudagraph_replay,
    )
    _run("serving_true_oob_no_fault", test_serving_true_oob_no_fault)
    for num_tiles in _DEGEN_TILES:
        _run(
            f"degenerate_empty_tile[tiles{num_tiles}]",
            test_flydsl_mla_reduce_degenerate_empty_tile,
            num_tiles,
        )
    _run(
        "dispatch_does_not_thread_actual_max_splits",
        test_dispatch_does_not_thread_actual_max_splits,
    )
    _run(
        "resolve_actual_max_splits_eager_and_capture",
        test_resolve_actual_max_splits_eager_and_capture,
    )
    for label, active, splits in _ADAPTIVE_SCENARIOS:
        _run(
            f"adaptive_launch_wrapper_vs_hip[{label}]",
            test_adaptive_launch_wrapper_vs_hip,
            label,
            active,
            splits,
        )
    _run("adaptive_launch_cudagraph_replay", test_adaptive_launch_cudagraph_replay)
    _run(
        "adaptive_launch_single_tile_uses_persistent",
        test_adaptive_launch_single_tile_uses_persistent,
    )

    return failures


# ---------------------------------------------------------------------------
# Perf scoreboard: GLM-5.2 serving decode grid, dense/uniform occupancy
# control, and irregular per-tile cost factors. Each candidate (``wrapper`` =
# FlyDSL, ``hip`` = production baseline where a Dv=512 HIP template exists) is
# timed with `run_perftest`/`bench_cudagraph` and checked with `checkAllclose`
# against a torch online-softmax reference (untimed, not tabled).
# ---------------------------------------------------------------------------

# (active_tiles, splits) serving decode buckets: 1 tile x 128 splits exercises
# the split-K path; 8 tiles x N splits exercises the sparse adaptive launch.
_SERVING_SCENARIOS = [
    (1, 128),
    (8, 32),
    (8, 26),
    (8, 13),
    (8, 6),
    (8, 5),
    (8, 3),
    (8, 2),
]


def _roofline(active, splits, H, Dv, out_dtype):
    """FLOPs (online-softmax weighted-sum FMA) and byte traffic for the reduce."""
    total_splits = active * splits
    out_bytes = torch.finfo(out_dtype).bits // 8
    flops = 2 * total_splits * H * Dv
    nbytes = (
        total_splits * H * Dv * 4  # partial_output fp32 read
        + total_splits * H * 4  # partial_lse   fp32 read
        + active * H * Dv * out_bytes  # final_output  write
        + active * H * 4  # final_lse     fp32 write
    )
    return flops, nbytes


@benchmark()
def test_mla_reduce(active, splits, H, Dv, dtype):
    po, pl, indptr, fmap, pmap, fout, flse = build_serving_decode_inputs(
        active, splits, dtype, H=H, Dv=Dv
    )

    # torch online-softmax reference over the active prefix only (tail tiles are
    # empty and skipped by both the kernels and the ref).
    ref_out, ref_lse = torch_ref_gather(
        po, pl, indptr[: active + 1], fmap[:active], pmap, H, Dv, dtype, M=1
    )

    candidates = {
        "wrapper": lambda: flydsl_mla_reduce_v1(
            po, pl, indptr, fmap, pmap, 1, fout, flse, num_kv_splits=splits
        )
    }
    if Dv == 512:
        # mla_reduce_v1 signature: (..., max_seqlen_q, num_kv_splits, out, lse)
        # HIP MLA_REDUCE_ROUTER only has a Dv=512 template; skip it elsewhere.
        candidates["hip"] = lambda: aiter.mla_reduce_v1(
            po, pl, indptr, fmap, pmap, 1, 0, fout, flse
        )

    flops, nbytes = _roofline(active, splits, H, Dv, dtype)

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        fout.zero_()
        flse.zero_()
        _, us = run_perftest(fn, num_warmup=25, num_iters=100)
        out = fout.clone()
        lse = flse.clone()
        err = checkAllclose(
            ref_out.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=_out_atol(dtype),
            msg=f"{name}: mla_reduce out",
            printLog=False,
        )
        checkAllclose(
            ref_lse.to(dtypes.fp32),
            lse.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-3,
            msg=f"{name}: mla_reduce lse",
            printLog=False,
        )
        # CUDA-graph replay µs (serving path): host dispatch captured once and
        # amortized away. TFLOPS/TB/s are derived from this, not eager us.
        graph_us = bench_cudagraph(fn) * 1e3
        ret[f"{name} us"] = us
        ret[f"{name} graph us"] = graph_us
        ret[f"{name} TFLOPS"] = flops / graph_us / 1e6
        ret[f"{name} TB/s"] = nbytes / graph_us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_mla_reduce_uniform(tiles, splits, H, Dv, M, dtype):
    """Dense/uniform occupancy control: every tile has ``splits`` splits."""
    po, pl, indptr, fmap, pmap, fout, flse = build_inputs(
        tiles, splits, H, Dv, dtype, M=M
    )
    ref_out, ref_lse = torch_ref(po, pl, tiles, splits, H, Dv, dtype, M=M)

    candidates = {
        "wrapper": lambda: flydsl_mla_reduce_v1(
            po, pl, indptr, fmap, pmap, M, fout, flse, num_kv_splits=splits
        )
    }
    if Dv == 512:
        candidates["hip"] = lambda: aiter.mla_reduce_v1(
            po, pl, indptr, fmap, pmap, M, 0, fout, flse
        )

    out_bytes = torch.finfo(dtype).bits // 8
    flops = 2 * tiles * splits * H * Dv
    nbytes = (
        tiles * splits * H * Dv * 4
        + tiles * splits * H * 4
        + tiles * M * H * Dv * out_bytes
        + tiles * M * H * 4
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        fout.zero_()
        flse.zero_()
        _, us = run_perftest(fn, num_warmup=25, num_iters=100)
        err = checkAllclose(
            ref_out.to(dtypes.fp32),
            fout.clone().to(dtypes.fp32),
            rtol=1e-2,
            atol=_out_atol(dtype),
            msg=f"{name}: mla_reduce_uniform out",
            printLog=False,
        )
        graph_us = bench_cudagraph(fn) * 1e3
        ret[f"{name} us"] = us
        ret[f"{name} graph us"] = graph_us
        ret[f"{name} TFLOPS"] = flops / graph_us / 1e6
        ret[f"{name} TB/s"] = nbytes / graph_us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_mla_reduce_irregular(splits_per_tile, gap_stride, pool_slack, H, Dv, dtype):
    """Irregular per-tile split cost factors: tier mismatch, gaps, pool slack."""
    po, pl, indptr, fmap, pmap, fout, flse = build_irregular_inputs(
        list(splits_per_tile),
        H,
        Dv,
        dtype,
        gap_stride=gap_stride,
        pool_slack=pool_slack,
    )
    ref_out, ref_lse = torch_ref_gather(po, pl, indptr, fmap, pmap, H, Dv, dtype)

    candidates = {
        "wrapper": lambda: flydsl_mla_reduce_v1(
            po,
            pl,
            indptr,
            fmap,
            pmap,
            1,
            fout,
            flse,
            num_kv_splits=max(splits_per_tile),
        )
    }
    if Dv == 512:
        candidates["hip"] = lambda: aiter.mla_reduce_v1(
            po, pl, indptr, fmap, pmap, 1, 0, fout, flse
        )

    total_splits = sum(splits_per_tile)
    active = sum(1 for s in splits_per_tile if s > 1)
    out_bytes = torch.finfo(dtype).bits // 8
    flops = 2 * total_splits * H * Dv
    nbytes = (
        total_splits * H * Dv * 4
        + total_splits * H * 4
        + active * H * Dv * out_bytes
        + active * H * 4
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        fout.zero_()
        flse.zero_()
        _, us = run_perftest(fn, num_warmup=25, num_iters=100)
        err = checkAllclose(
            ref_out.to(dtypes.fp32),
            fout.clone().to(dtypes.fp32),
            rtol=1e-2,
            atol=_out_atol(dtype),
            msg=f"{name}: mla_reduce_irregular out",
            printLog=False,
        )
        graph_us = bench_cudagraph(fn) * 1e3
        ret[f"{name} us"] = us
        ret[f"{name} graph us"] = graph_us
        ret[f"{name} TFLOPS"] = flops / graph_us / 1e6
        ret[f"{name} TB/s"] = nbytes / graph_us / 1e6
        ret[f"{name} err"] = err
    return ret


def run_bench(args):
    """Run the three perf sweeps and print one markdown table each."""
    for dtype in args.dtype:
        df = []
        for (H, Dv), (active, splits) in itertools.product(args.hdv, args.scenario):
            df.append(test_mla_reduce(active, splits, H, Dv, dtype))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "mla_reduce GLM-5.2 serving summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

        df = []
        for (H, Dv), tiles, splits in itertools.product(
            args.hdv, args.tiles, args.uniform_splits
        ):
            df.append(test_mla_reduce_uniform(tiles, splits, H, Dv, 1, dtype))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "mla_reduce uniform (occupancy) summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

        df = []
        for (H, Dv), spt, gap_stride, pool_slack in itertools.product(
            args.hdv, args.splits_per_tile, args.gap_stride, args.pool_slack
        ):
            df.append(
                test_mla_reduce_irregular(spt, gap_stride, pool_slack, H, Dv, dtype)
            )
        df = pd.DataFrame(df)
        aiter.logger.info(
            "mla_reduce irregular (cost-factor) summary (markdown):\n%s",
            df.to_markdown(index=False),
        )


def main():
    if not torch.cuda.is_available() or get_gfx() not in MLA_REDUCE_SUPPORTED_GFX:
        aiter.logger.warning("mla_reduce unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default="bf16,",
        metavar="{bf16,fp16}",
        help="Output data type, e.g. -d bf16",
    )
    parser.add_argument(
        "--hdv",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(16, 512)],
        help="(H, Dv) shape, e.g. --hdv 16,512 128,512",
    )
    parser.add_argument(
        "-s",
        "--scenario",
        type=dtypes.str2tuple,
        nargs="*",
        default=_SERVING_SCENARIOS,
        help="(active_tiles, splits) decode buckets, e.g. -s 1,128 8,32",
    )
    parser.add_argument(
        "--tiles",
        type=int,
        nargs="*",
        default=[256],
        help="uniform sweep: dense reduce-tile counts, e.g. --tiles 128 256",
    )
    parser.add_argument(
        "--uniform-splits",
        type=int,
        nargs="*",
        default=[8],
        help="uniform sweep: splits per tile (dense), e.g. --uniform-splits 8 128",
    )
    parser.add_argument(
        "--splits-per-tile",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(8, 304), (4, 32, 8, 64)],
        help='irregular sweep: per-tile n_splits, e.g. --splits-per-tile "8,304" "4,32,8,64"',
    )
    parser.add_argument(
        "--gap-stride",
        type=int,
        nargs="*",
        default=[1],
        help="irregular sweep: partial-pool row stride, e.g. --gap-stride 1 4",
    )
    parser.add_argument(
        "--pool-slack",
        type=int,
        nargs="*",
        default=[0],
        help="irregular sweep: extra unused partial-pool rows",
    )
    args = parser.parse_args()

    aiter.logger.info("mla_reduce: running invariant/correctness checks...")
    failures = run_checks()
    if failures:
        for name, exc in failures:
            aiter.logger.error("FAILED %s: %r", name, exc)
        aiter.logger.error(
            "mla_reduce: %d invariant check(s) failed; skipping perf sweep",
            len(failures),
        )
        sys.exit(1)
    aiter.logger.info("mla_reduce: all invariant checks passed")

    run_bench(args)


if __name__ == "__main__":
    main()
