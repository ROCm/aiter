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
import warnings
from contextlib import contextmanager
from typing import NamedTuple
from unittest import mock

import flydsl.expr as fx
import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_mla_reduce_v1
from aiter.ops.flydsl.kernels.mla_reduce import (
    LDS_MAX_SPLITS,
    Tier,
    _get_splitk_scratch,
    compile_mla_reduce,
    compile_mla_reduce_splitk,
    derive_actual_max_splits,
    plan_splitk,
    plan_splitk_capture_safe,
    select_tier,
    should_use_persistent_launch,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.flydsl.mla_reduce_kernels import _pointer_arg
from aiter.test_common import benchmark, checkAllclose, run_perftest

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


class Inputs(NamedTuple):
    """One reduce fixture: the five metadata tensors plus the two output buffers.

    Field order matches the ``flydsl_mla_reduce_v1`` / HIP ``mla_reduce_v1``
    argument order (``reduce_final_map`` before ``reduce_partial_map``). The raw
    pointer launch takes the two maps the other way round, so ``make_runner``
    names ``pmap``/``fmap`` explicitly rather than splatting a bundle.
    """

    po: torch.Tensor  # fp32 [num_partial_rows, H, Dv]
    pl: torch.Tensor  # fp32 [num_partial_rows, H]
    indptr: torch.Tensor  # i32 [num_reduce_tile + 1], CSR over splits
    fmap: torch.Tensor  # i32 [num_reduce_tile, 2], {q_start, q_end}
    pmap: torch.Tensor  # i32 [indptr[-1]], partial-pool gather rows
    fout: torch.Tensor  # out_dtype [num_final_rows, H, Dv]
    flse: torch.Tensor  # fp32 [num_final_rows, H]

    @property
    def maps(self):
        """The five metadata tensors, in ``mla_reduce_v1`` positional order."""
        return self[:5]


def _rand_partials(rows, H, Dv, device, g):
    """Random fp32 partial pool; ``partial_lse`` is scaled to widen the online-softmax range."""
    po = torch.randn(rows, H, Dv, dtype=torch.float32, device=device, generator=g)
    pl = torch.randn(rows, H, dtype=torch.float32, device=device, generator=g) * 2.0
    return po, pl


def _garbage_final_map(num_tiles, active_tiles, device):
    """``reduce_final_map`` with an identity q-range over the active prefix and an unmapped ``1 << 24`` tail, mirroring metadata that never writes the inactive slots."""
    fmap = torch.empty(num_tiles, 2, dtype=torch.int32, device=device)
    q = torch.arange(active_tiles, dtype=torch.int32, device=device)
    fmap[:active_tiles, 0] = q
    fmap[:active_tiles, 1] = q + 1
    fmap[active_tiles:, 0] = 1 << 24
    fmap[active_tiles:, 1] = (1 << 24) + 1
    return fmap


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

    partial_output, partial_lse = _rand_partials(num_partial_rows, H, Dv, device, g)

    q_start = torch.arange(num_tiles, dtype=torch.int32, device=device) * M
    reduce_final_map = torch.stack([q_start, q_start + M], dim=1).contiguous()
    for t, s in enumerate(splits_per_tile):
        if int(s) <= 1:
            reduce_final_map[t, 0] = 1 << 24
            reduce_final_map[t, 1] = (1 << 24) + M

    return Inputs(
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        torch.empty(num_tiles * M, H, Dv, dtype=out_dtype, device=device),
        torch.empty(num_tiles * M, H, dtype=torch.float32, device=device),
    )


def build_inputs(num_tiles, num_splits, H, Dv, out_dtype, M=1, device="cuda", seed=0):
    """Dense/uniform reduce inputs: every tile has ``num_splits`` splits and the gather map is contiguous."""
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
    x = build_irregular_inputs(
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
    return x._replace(
        fout=torch.empty(active_tiles, H, Dv, dtype=out_dtype, device=device),
        flse=torch.empty(active_tiles, H, dtype=torch.float32, device=device),
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
    x, H, Dv, out_dtype, M=1, num_partial_rows=None, num_final_rows=None
):
    """Gather-based online-softmax reference for irregular metadata. Follows the kernel's CSR + gather contract: per tile it gathers partial rows via ``pmap[indptr[t]:indptr[t+1]]`` and merges them; tiles with ``n_splits <= 1`` are skipped (output rows stay zero). When ``num_partial_rows`` / ``num_final_rows`` are set, out-of-range pmap rows / q_start are skipped (mirrors guards-ON)."""
    po, pl, indptr, fmap, pmap = x.maps
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


def hip_ref(x, num_tiles, H, Dv, out_dtype, M=1):
    """Reference output from HIP kn_mla_reduce_v1. Outputs are zero-initialized so skipped (empty) tiles match a zero-initialized final buffer under test."""
    ref_out = torch.zeros(num_tiles * M, H, Dv, dtype=out_dtype, device=x.po.device)
    ref_lse = torch.zeros(num_tiles * M, H, dtype=torch.float32, device=x.po.device)
    aiter.mla_reduce_v1(*x.maps, M, LDS_MAX_SPLITS, ref_out, ref_lse)
    torch.cuda.synchronize()
    return ref_out, ref_lse


def hip_ref_like_fout(x, M=1):
    """HIP reference sized to ``x.fout`` / ``x.flse`` (serving grids with sparse tiles)."""
    ref_out = torch.zeros_like(x.fout)
    ref_lse = torch.zeros_like(x.flse)
    aiter.mla_reduce_v1(*x.maps, M, LDS_MAX_SPLITS, ref_out, ref_lse)
    torch.cuda.synchronize()
    return ref_out, ref_lse


def build_serving_sparse_grid_inputs(
    H=16,
    Dv=512,
    out_dtype=torch.bfloat16,
    device="cuda",
    seed=0,
):
    """batch=8 steady-state: 16384-tile grid, 8 active tiles × 32 splits. Mirrors 131K-context serving metadata: partial pool 606 rows, CSR sentinel at 256, garbage ``reduce_final_map`` / ``reduce_partial_map`` tail slots. Tail tiles are flat at sentinel (``n_splits == 0``), already skipped by the ``n_splits > 1`` clamp, so this fixture does **not** discriminate the gather/store guards."""
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

    partial_output, partial_lse = _rand_partials(num_partial_rows, H, Dv, device, g)

    reduce_partial_map = torch.empty(num_partial_rows, dtype=torch.int32, device=device)
    reduce_partial_map[:total_splits] = torch.arange(
        total_splits, dtype=torch.int32, device=device
    )
    reduce_partial_map[total_splits:] = torch.randint(
        -(1 << 30),
        1 << 30,
        (num_partial_rows - total_splits,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )

    return Inputs(
        partial_output,
        partial_lse,
        reduce_indptr,
        _garbage_final_map(num_reduce_tile, active_tiles, device),
        reduce_partial_map,
        torch.zeros(active_tiles, H, Dv, dtype=out_dtype, device=device),
        torch.zeros(active_tiles, H, dtype=torch.float32, device=device),
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
    """Small serving grid with allocation slack for in-process guard differentials. Allocates ``partial_output`` / ``final_output`` with extra rows but returns smaller *logical* ``num_partial_rows`` / ``num_final_rows`` for the kernel. The fake-active tail (non-sentinel ``n_splits``) exercises the gather/store guards adversarially. Returns tensors plus a ``meta`` dict with logical bounds and discriminator tile indices for tests."""
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

    partial_output, partial_lse = _rand_partials(alloc_partial_rows, H, Dv, device, g)

    reduce_partial_map = torch.arange(total_splits, dtype=torch.int32, device=device)
    # Gather discriminator: one split on tile 0 points into mapped slack, holding
    # the tile's max LSE so guards-OFF must visibly pull the result toward it.
    slack_p_row = logical_partial_rows
    reduce_partial_map[splits_per_active // 2] = slack_p_row
    partial_output[slack_p_row].fill_(1000.0)
    tile0_lse_max = partial_lse[:splits_per_active].max().item()
    partial_lse[slack_p_row].fill_(tile0_lse_max + 5.0)

    # Store discriminator: one fake-active tail tile targets final-output slack.
    reduce_final_map = _garbage_final_map(num_tiles, active_tiles, device)
    store_slack_q = logical_final_rows
    reduce_final_map[store_tile, 0] = store_slack_q
    reduce_final_map[store_tile, 1] = store_slack_q + 1

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
        Inputs(
            partial_output,
            partial_lse,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            final_output,
            final_lse,
        ),
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
    """Fake-active tail with genuine OOB indices (guards-ON no-fault regression). Cannot be run with ``disable_guards=True`` in-process (would GPU-fault)."""
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
    partial_output, partial_lse = _rand_partials(num_partial_rows, H, Dv, device, g)

    # The tail tile's gather rows land past the end of the partial pool.
    reduce_partial_map = torch.arange(sentinel, dtype=torch.int32, device=device)
    tail_t0 = int(reduce_indptr[tail_tile].item())
    for i in range(tail_splits):
        reduce_partial_map[tail_t0 + i] = num_partial_rows + i

    return Inputs(
        partial_output,
        partial_lse,
        reduce_indptr,
        _garbage_final_map(num_tiles, active_tiles, device),
        reduce_partial_map,
        torch.zeros(active_tiles, H, Dv, dtype=out_dtype, device=device),
        torch.zeros(active_tiles, H, dtype=torch.float32, device=device),
    )


def build_serving_stale_indptr_inputs(
    H=16,
    Dv=512,
    out_dtype=torch.bfloat16,
    device="cuda",
    seed=0,
):
    """Batch-transition metadata: batch=8 sparse grid patched to batch=1 layout. Tile 0 becomes ``n_splits=128`` for batch=1; tiles 1..7 keep stale batch=8 CSR (``n_splits=32``, stale pmap rows beyond the logical bound, stale fmap q 1..7). Uses the allocation-slack trick so logical bounds are smaller than the buffers. Returns tensors plus ``meta`` with logical bounds for guard differentials."""
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

    partial_output, partial_lse = _rand_partials(alloc_partial_rows, H, Dv, device, g)

    reduce_partial_map = torch.arange(sentinel, dtype=torch.int32, device=device)
    # Stale tiles 1..7: pmap rows from batch=8 now >= logical_partial_rows.
    stale_base = batch1_splits
    for t in range(1, active_tiles_batch8):
        t0 = indptr_host[t]
        for s in range(splits_per_active):
            reduce_partial_map[t0 + s] = stale_base + s
    partial_output[stale_base : stale_base + splits_per_active].fill_(500.0)
    tile1_lse_max = partial_lse[stale_base : stale_base + splits_per_active].max()
    partial_lse[stale_base].fill_(tile1_lse_max + 5.0)

    # Stale fmap: tiles 1..7 keep their batch=8 q rows, past the batch=1 bound.
    reduce_final_map = _garbage_final_map(num_reduce_tile, active_tiles_batch8, device)

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
        Inputs(
            partial_output,
            partial_lse,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            final_output,
            final_lse,
        ),
        meta,
    )


def make_runner(
    x,
    H,
    Dv,
    out_dtype_str,
    M=1,
    *,
    output_lse=True,
    tier=None,
    disable_guards=False,
    num_partial_rows=None,
    num_final_rows=None,
    waves_per_eu=4,
    use_splitk=False,
    splitk_factor=None,
):
    """Precompile + bind args; return a zero-overhead closure for the timed loop. tier: compile-time tier override for isolated tests. None (default) uses Tier.ALL (production path with device-side runtime tier selection)."""
    po, pl, indptr, fmap, pmap, fout, flse = x
    num_tiles = fmap.shape[0]
    num_cu = torch.cuda.get_device_properties(0).multi_processor_count
    compile_tier = Tier.ALL if tier is None else tier
    if num_partial_rows is None:
        num_partial_rows = int(po.size(0))
    if num_final_rows is None:
        num_final_rows = int(fout.size(0))
    # Split-K (opt-in via use_splitk): cooperative multi-block reduction for
    # the low-tile / high-split decode case. Metadata is inspected at setup
    # (outside any CUDA-graph capture); the scratch is pre-allocated +
    # reused, so the captured run() only launches the two kernels
    # (capture-safe).
    if use_splitk and tier is None and not disable_guards:
        diffs = indptr[1:] - indptr[:-1]
        active_tiles = int((diffs > 1).sum().item())
        max_splits_val = int(diffs.max().item()) if diffs.numel() else 0
        splitk_kwargs = {} if splitk_factor is None else {"factor": splitk_factor}
        engage, K, num_slots = plan_splitk(
            active_tiles=active_tiles,
            H=H,
            max_seqlen_q=M,
            max_splits=max_splits_val,
            num_cu=num_cu,
            **splitk_kwargs,
        )
        if engage:
            lp, lc = compile_mla_reduce_splitk(
                H=H,
                Dv=Dv,
                out_dtype=out_dtype_str,
                K=K,
                output_lse=output_lse,
                waves_per_eu=waves_per_eu,
            )
            sk_acc, sk_ml = _get_splitk_scratch(num_slots, K, Dv, fout.device.index)
            partial_head = (
                _pointer_arg(po, torch.float32),
                _pointer_arg(pl, torch.float32),
                _pointer_arg(indptr, torch.int32),
                _pointer_arg(pmap, torch.int32),
                _pointer_arg(sk_acc, torch.float32),
                _pointer_arg(sk_ml, torch.float32),
                int(num_partial_rows),
                int(num_slots * K),
            )
            combine_head = (
                _pointer_arg(indptr, torch.int32),
                _pointer_arg(fmap, torch.int32),
                _pointer_arg(sk_acc, torch.float32),
                _pointer_arg(sk_ml, torch.float32),
                _pointer_arg(fout, fout.dtype),
                _pointer_arg(flse, torch.float32),
                int(fout.stride(0)),
                int(fout.stride(1)),
                int(num_final_rows),
                int(num_slots),
            )

            def run():
                st = fx.Stream(torch.cuda.current_stream())
                _run_compiled(lp, *partial_head, st)
                _run_compiled(lc, *combine_head, st)

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
        waves_per_eu=waves_per_eu,
    )
    head = (
        _pointer_arg(po, torch.float32),
        _pointer_arg(pl, torch.float32),
        _pointer_arg(indptr, torch.int32),
        _pointer_arg(pmap, torch.int32),
        _pointer_arg(fmap, torch.int32),
        _pointer_arg(fout, fout.dtype),
        _pointer_arg(flse, torch.float32),
        int(fout.stride(0)),
        int(fout.stride(1)),
        int(num_cu),
        int(num_tiles),
        int(M),
        int(num_partial_rows),
        int(num_final_rows),
    )

    def run():
        _run_compiled(kernel, *head, fx.Stream(torch.cuda.current_stream()))

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
    """Capture ``fn`` into a CUDA graph and replay it, to surface replay-only faults (the failure mode reported under real serving). ``fn`` writes into its bound output tensors; the caller inspects those after this returns."""
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


def _run_kernel(x, H, Dv, dt, M=1, *, replay=False, **runner_kwargs):
    """Zero ``x``'s outputs, bind the pointer-ABI runner, and run it either eagerly or through a CUDA-graph capture + replay. Results land in ``x.fout`` / ``x.flse``."""
    x.fout.zero_()
    x.flse.zero_()
    run = make_runner(x, H, Dv, dt, M, **runner_kwargs)
    if replay:
        run_cudagraph_replay(run)
    else:
        run()
        torch.cuda.synchronize()


def _run_wrapper(x, M=1, **kwargs):
    """Call the production torch wrapper on ``x``; results land in ``x.fout`` / ``x.flse``."""
    flydsl_mla_reduce_v1(*x.maps, M, x.fout, x.flse, **kwargs)


def _single_final_row(x):
    """Narrow the fixture's output buffers to one final row (bs=1 decode)."""
    return x._replace(fout=x.fout[:1].contiguous(), flse=x.flse[:1].contiguous())


@contextmanager
def _env(**kwargs):
    """Set/unset environment variables for the duration of the block, restoring the prior state on exit (a pytest-free ``monkeypatch.setenv``/ ``delenv``). Pass ``None`` for a var that should be unset."""
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

# CUDA-graph replay: highest-risk irregular fixtures, to surface replay-only
# faults. (id, shape, ref, splits_per_tile, gap_stride, M).
_GRAPH_CASES = [
    ("tier_mismatch", _GLM_SHAPE, "torch", [8, 304], 1, 1),
    ("gapped_pmap", _HIP_SHAPE, "hip", [8, 8, 8, 8], 4, 1),
    ("empty_middle", _GLM_SHAPE, "torch", [8, 0, 16, 8], 1, 1),
    ("mlds_max", _GLM_SHAPE, "torch", [304], 1, 1),
    ("mtp_irregular", _GLM_SHAPE, "torch", [8, 32, 16], 2, 4),
    # Uniform 32-split tiles (runtime NLSE=1 path).
    ("small_split", _GLM_SHAPE, "torch", [32] * 4, 1, 1),
]

_DEGEN_TILES = [2, 4]


def _require_cuda():
    """``main()`` already gates on this before calling `run_checks()`; this is a defensive re-check for anyone importing/calling a check function directly (rule 7: functions stay independently importable)."""
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


def _masking_ref(x, H, Dv, out_dtype, meta, M=1):
    return torch_ref_gather(
        x,
        H,
        Dv,
        out_dtype,
        M,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )


def _logical_rows(out, lse, meta):
    """``out``/``lse`` clipped to the fixture's logical final-row bound."""
    n = meta["logical_final_rows"]
    return out[:n], lse[:n]


def _run_guarded(x, H, Dv, dt, meta, *, disable_guards=False, M=1):
    x.fout.zero_()
    x.flse.zero_()
    if meta.get("fout_slack_seed") is not None:
        x.fout[meta["store_slack_q"] :].fill_(meta["fout_slack_seed"])
    run = make_runner(
        x,
        H,
        Dv,
        dt,
        M,
        disable_guards=disable_guards,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )
    run()
    torch.cuda.synchronize()
    return x.fout.clone(), x.flse.clone()


def _run_guarded_cudagraph(x, H, Dv, dt, meta, *, disable_guards):
    """CUDA-graph-replay counterpart of ``_run_guarded``, for the mapped-slack differential tests only (``store_slack_q`` bounds the slack region for both the gather and store checks there)."""
    out = torch.zeros_like(x.fout)
    lse = torch.zeros_like(x.flse)
    out[meta["store_slack_q"] :].fill_(meta["fout_slack_seed"])
    run = make_runner(
        x._replace(fout=out, flse=lse),
        H,
        Dv,
        dt,
        disable_guards=disable_guards,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )
    run_cudagraph_replay(run)
    return out, lse


def _guards_on_off(x, H, Dv, dt, meta, runner=_run_guarded):
    """Run one mapped-slack fixture with guards ON then OFF, asserting that the guards-ON result matches the masking reference over the logical rows. Returns ``(on_out, off_out, ref_out)`` so each caller applies its own differential."""
    ref_out, ref_lse = _masking_ref(x, H, Dv, _out_dtype(dt), meta)
    on_out, on_lse = runner(x, H, Dv, dt, meta, disable_guards=False)
    off_out, _off_lse = runner(x, H, Dv, dt, meta, disable_guards=True)
    _assert_close(
        *_logical_rows(on_out, on_lse, meta),
        *_logical_rows(ref_out, ref_lse, meta),
        dt,
    )
    return on_out, off_out, ref_out


def _assert_gather_differential(x, H, Dv, dt, meta):
    _on_out, off_out, ref_out = _guards_on_off(x, H, Dv, dt, meta)
    atol = _out_atol(dt)
    q_row = meta["gather_q_row"]
    gather_err = (off_out[q_row].float() - ref_out[q_row].float()).abs().max().item()
    assert gather_err > 5 * atol, (
        f"gather guard differential failed: guards-OFF row {q_row} "
        f"max_abs_err={gather_err:.3e} <= {5 * atol}"
    )


def _assert_store_differential(x, H, Dv, dt, meta):
    on_out, off_out, _ref_out = _guards_on_off(x, H, Dv, dt, meta)
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
    """Build irregular inputs and run the kernel; returns the fixture, outputs filled."""
    x = build_irregular_inputs(spt, H, Dv, _out_dtype(dt), M=M, gap_stride=gap)
    _run_kernel(x, H, Dv, dt, M)
    return x


def test_flydsl_mla_reduce_irregular_vs_hip(case):
    """Irregular metadata matches HIP kn_mla_reduce_v1 (DeepSeek shape, Dv=512)."""
    _require_cuda()
    _name, spt, gap, M, dt = case
    H, Dv = _HIP_SHAPE
    x = _run_irregular(spt, gap, M, H, Dv, dt)
    ref_out, ref_lse = hip_ref(x, len(spt), H, Dv, _out_dtype(dt), M)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


def test_flydsl_mla_reduce_irregular_vs_torch_ref(case):
    """Irregular metadata matches the gather-based torch ref (GLM-5.2, Dv=256)."""
    _require_cuda()
    _name, spt, gap, M, dt = case
    H, Dv = _GLM_SHAPE
    x = _run_irregular(spt, gap, M, H, Dv, dt)
    ref_out, ref_lse = torch_ref_gather(x, H, Dv, _out_dtype(dt), M)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


def test_flydsl_mla_reduce_uniform_smoke(case):
    """Dense/uniform smoke: each compile tier on both reference paths."""
    _require_cuda()
    (H, Dv), ref, S = case
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    x = build_inputs(_SMOKE_TILES, S, H, Dv, out_dtype)
    _run_kernel(x, H, Dv, dt, tier=select_tier(S))
    if ref == "hip":
        ref_out, ref_lse = hip_ref(x, _SMOKE_TILES, H, Dv, out_dtype)
    else:
        ref_out, ref_lse = torch_ref(x.po, x.pl, _SMOKE_TILES, S, H, Dv, out_dtype)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


def test_flydsl_mla_reduce_cudagraph_replay(case):
    """Irregular metadata stays correct under CUDA-graph capture + replay (the serving failure mode); no GPU fault and output matches the reference."""
    _require_cuda()
    _name, (H, Dv), ref, spt, gap, M = case
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    x = build_irregular_inputs(spt, H, Dv, out_dtype, M=M, gap_stride=gap)
    _run_kernel(x, H, Dv, dt, M, replay=True)
    if ref == "hip":
        ref_out, ref_lse = hip_ref(x, len(spt), H, Dv, out_dtype, M)
    else:
        ref_out, ref_lse = torch_ref_gather(x, H, Dv, out_dtype, M)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


_SERVING_SHAPE = (16, 512)


def test_serving_sparse_grid(replay=False):
    """batch=8 layout: 16384-tile grid, 8 active tiles, garbage tail; eager (vs HIP) or under CUDA-graph replay (prod failure mode)."""
    _require_cuda()
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    x = build_serving_sparse_grid_inputs(*_SERVING_SHAPE, out_dtype=out_dtype)
    _run_kernel(x, *_SERVING_SHAPE, dt, replay=replay)
    ref_out, ref_lse = hip_ref_like_fout(x)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


# Split-K cooperative reduction for low-tile/high-split decode.
_SPLITK_H, _SPLITK_DV = 16, 512
_SPLITK_GRID = SERVING_NUM_REDUCE_TILE


def _build_splitk_b1_s128(out_dtype):
    """1 active tile x 128 splits in a 16384-tile serving grid (b1_s128)."""
    spt = [128] + [0] * (_SPLITK_GRID - 1)
    return build_irregular_inputs(
        spt, _SPLITK_H, _SPLITK_DV, out_dtype, M=1, gap_stride=1, pool_slack=0
    )


def _assert_splitk_engages(indptr):
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


def _run_splitk_b1_s128(K=None, replay=False):
    """Build b1_s128 split-K inputs, assert engagement, and run eager or under CUDA-graph replay; caller applies its own reference comparison."""
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    x = _build_splitk_b1_s128(out_dtype)
    _assert_splitk_engages(x.indptr)
    _run_kernel(
        x,
        _SPLITK_H,
        _SPLITK_DV,
        dt,
        replay=replay,
        use_splitk=True,
        splitk_factor=K,
    )
    return x, dt, out_dtype


def test_splitk_b1_s128_vs_torch_ref(K, replay=False):
    """Split-K partial+combine matches the gather-based torch reference for the low-tile/high-split decode case, across split factors K (``K=None`` uses the planner's own default), eager or under CUDA-graph capture + replay."""
    _require_cuda()
    x, dt, out_dtype = _run_splitk_b1_s128(K, replay)
    ref_out, ref_lse = torch_ref_gather(x, _SPLITK_H, _SPLITK_DV, out_dtype)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


def test_splitk_b1_s128_vs_hip():
    """Split-K matches the production HIP kn_mla_reduce_v1 (Dv=512 template)."""
    _require_cuda()
    x, dt, _out_dtype_ = _run_splitk_b1_s128()
    ref_out, ref_lse = hip_ref_like_fout(x)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


# Split-K planner boundaries: (kwargs, expected (engage, K, num_slots)).
# All cases share H=16, max_seqlen_q=1, num_cu=304 unless a case overrides it.
# plan_splitk reads CSR-derived values at planning time; the capture-safe variant
# decides from host values alone, so it is legal under graph capture.
_PLAN_SPLITK_CASES = [
    (dict(active_tiles=1, max_splits=128), (True, 16, 16)),
    (dict(active_tiles=1, max_splits=8), (False, 1, 0)),  # below min_splits
    (dict(active_tiles=20, max_splits=128), (False, 1, 0)),  # saturated grid
    (
        dict(active_tiles=1, max_splits=128, max_seqlen_q=2),
        (False, 1, 0),
    ),  # prefill
]
_PLAN_CAPTURE_SAFE_CASES = [
    (dict(num_final_rows=32, num_kv_splits=128), (False, 1, 0)),  # saturated grid
    (dict(num_final_rows=1, num_kv_splits=32), (False, 1, 0)),  # below min_splits
    (
        dict(num_final_rows=1, num_kv_splits=304, actual_max_splits=8),
        (False, 1, 0),
    ),
    (
        dict(num_final_rows=1, num_kv_splits=304, actual_max_splits=128),
        (True, 16, 16),
    ),
]


def test_splitk_planner_boundaries():
    """Both planners require decode, enough splits, and an unsaturated grid.

    The capture-safe planner uses actual_max_splits when supplied, so a loose
    304 budget with eight real splits must stay on the single-kernel path.
    """
    base = {"H": 16, "max_seqlen_q": 1, "num_cu": 304}
    for planner, cases in (
        (plan_splitk, _PLAN_SPLITK_CASES),
        (plan_splitk_capture_safe, _PLAN_CAPTURE_SAFE_CASES),
    ):
        for kwargs, expected in cases:
            got = planner(**{**base, **kwargs})
            assert got == expected, (planner.__name__, kwargs, got, expected)


# Device-adaptive, capture-safe split-K through the production wrapper.


def _build_da_single_tile(out_dtype, splits, pool=304):
    """One active tile (bs=1: final_output has a single row) with `splits` splits, partial pool sized to `pool` so the split count can be MUTATED up to `pool` across CUDA-graph replays."""
    x = build_irregular_inputs(
        [splits],
        _SPLITK_H,
        _SPLITK_DV,
        out_dtype,
        M=1,
        gap_stride=1,
        pool_slack=pool - splits,
    )
    return x._replace(pmap=torch.arange(pool, dtype=torch.int32, device=x.pmap.device))


def test_da_splitk_wrapper_vs_hip():
    """The default-able wrapper path (DA split-K on) matches HIP for b1_s128."""
    _require_cuda()
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    x = _build_da_single_tile(out_dtype, 128, pool=128)
    x.fout.zero_()
    x.flse.zero_()
    run_cudagraph_replay(lambda: _run_wrapper(x, num_kv_splits=128))
    ref_out, ref_lse = hip_ref_like_fout(x)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


def _build_da_mixed_split_inputs(out_dtype, skipped_splits):
    """Build a high-split tile beside a stage-1-finalized stale-map tile."""
    x = build_irregular_inputs(
        [skipped_splits, 64], _SPLITK_H, _SPLITK_DV, out_dtype, M=1, gap_stride=1
    )
    # V1.2 metadata does not write this entry for n_splits <= 1. Model a
    # prior decode/capture replay leaving a valid but stale q-range behind.
    x.fmap[0] = torch.tensor([0, 1], dtype=torch.int32, device=x.fmap.device)
    x.fout.fill_(7.0)
    x.flse.fill_(9.0)
    return x


def test_da_splitk_preserves_stage1_rows(skipped_splits, replay=False):
    """Split-K must not overwrite stale-map rows finalized directly by stage 1, eager or through CUDA-graph replay."""
    _require_cuda()
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    x = _build_da_mixed_split_inputs(out_dtype, skipped_splits)
    expected_out = x.fout[0].clone()
    expected_lse = x.flse[0].clone()

    def run():
        _run_wrapper(x, num_kv_splits=64, actual_max_splits=64)

    if replay:
        run_cudagraph_replay(run)
    else:
        run()
        torch.cuda.synchronize()
    assert torch.equal(x.fout[0], expected_out)
    assert torch.equal(x.flse[0], expected_lse)
    ref_out, ref_lse = torch_ref_gather(x, _SPLITK_H, _SPLITK_DV, out_dtype)
    _assert_close(x.fout[1:], x.flse[1:], ref_out[1:], ref_lse[1:], out_dtype)


def test_da_splitk_capture_safe_varying_splits():
    """One CUDA-graph capture (bs=1, grid/K/scratch baked from host num_kv_splits) stays correct across replays whose per-tile split count changes on-device."""
    _require_cuda()
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    pool = 304
    nkv = 128
    x = _build_da_single_tile(out_dtype, pool, pool=pool)
    engage, _K, slots = plan_splitk_capture_safe(
        num_final_rows=1,
        H=_SPLITK_H,
        max_seqlen_q=1,
        num_kv_splits=nkv,
        num_cu=304,
    )
    assert engage and slots == _SPLITK_H, "DA split-K must engage for bs=1"

    def run():
        _run_wrapper(x, num_kv_splits=nkv)

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
        x.indptr[1] = s_k  # mutate the captured CSR in place (host->device copy)
        x.fout.zero_()
        x.flse.zero_()
        graph.replay()
        torch.cuda.synchronize()
        ref_out, ref_lse = torch_ref_gather(x, _SPLITK_H, _SPLITK_DV, out_dtype)
        _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


# actual_max_splits gate.


def test_derive_actual_max_splits():
    """Helper matches CSR max tile width."""
    indptr = torch.tensor([0, 8, 12, 12], dtype=torch.int32, device="cuda")
    assert derive_actual_max_splits(indptr) == 8


def test_actual_max_splits_wrapper_loose_budget_correct():
    """Loose budget (304) + small actual splits stays on single-kernel path and matches the torch reference."""
    _require_cuda()
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    x = _single_final_row(_build_da_single_tile(out_dtype, 8, pool=8))
    actual = derive_actual_max_splits(x.indptr)
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

    x.fout.zero_()
    x.flse.zero_()
    _run_wrapper(x, num_kv_splits=304, actual_max_splits=actual)
    torch.cuda.synchronize()
    ref_out, ref_lse = torch_ref_gather(x, _SPLITK_H, _SPLITK_DV, out_dtype)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


def test_actual_max_splits_wrapper_cudagraph_replay():
    """Loose budget + actual_max_splits gate stays correct under graph replay."""
    _require_cuda()
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    x = _single_final_row(_build_da_single_tile(out_dtype, 128, pool=128))
    actual = derive_actual_max_splits(x.indptr)
    assert actual == 128
    x.fout.zero_()
    x.flse.zero_()
    run_cudagraph_replay(
        lambda: _run_wrapper(x, num_kv_splits=304, actual_max_splits=actual)
    )
    ref_out, ref_lse = torch_ref_gather(x, _SPLITK_H, _SPLITK_DV, out_dtype)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


def test_serving_stale_indptr_cudagraph_replay():
    """Cudagraph replay after batch-8→batch-1 layout (guards-ON/OFF differential)."""
    _require_cuda()
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    x, meta = build_serving_stale_indptr_inputs(*_SERVING_SHAPE, out_dtype=out_dtype)
    ref_out, ref_lse = _masking_ref(x, *_SERVING_SHAPE, out_dtype, meta)

    def replay(disable_guards):
        """Fresh output buffers (slack pre-seeded), captured and replayed."""
        out = torch.zeros_like(x.fout)
        lse = torch.zeros_like(x.flse)
        out[meta["logical_final_rows"] :].fill_(meta["fout_slack_seed"])
        run_cudagraph_replay(
            make_runner(
                x._replace(fout=out, flse=lse),
                *_SERVING_SHAPE,
                dt,
                disable_guards=disable_guards,
                num_partial_rows=meta["logical_partial_rows"],
                num_final_rows=meta["logical_final_rows"],
            )
        )
        return out, lse

    on_out, on_lse = replay(False)
    off_out, _off_lse = replay(True)
    _assert_close(
        *_logical_rows(on_out, on_lse, meta),
        *_logical_rows(ref_out, ref_lse, meta),
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
    x, meta = build_serving_mapped_slack_inputs(H, Dv, _out_dtype(dt))
    _assert_gather_differential(x, H, Dv, dt, meta)


def test_serving_store_guard_differential():
    """Mapped-slack store: guards-OFF writes slack, guards-ON preserves seed."""
    _require_cuda()
    dt = "bf16"
    H, Dv = _SERVING_SHAPE
    x, meta = build_serving_mapped_slack_inputs(H, Dv, _out_dtype(dt))
    _assert_store_differential(x, H, Dv, dt, meta)


def test_serving_gather_guard_differential_cudagraph_replay():
    """Gather guard differential holds under CUDA-graph capture/replay."""
    _require_cuda()
    dt = "bf16"
    H, Dv = _SERVING_SHAPE
    x, meta = build_serving_mapped_slack_inputs(H, Dv, _out_dtype(dt))
    _, off_out, ref_out = _guards_on_off(
        x, H, Dv, dt, meta, runner=_run_guarded_cudagraph
    )
    q_row = meta["gather_q_row"]
    gather_err = (off_out[q_row].float() - ref_out[q_row].float()).abs().max().item()
    assert gather_err > 5 * _out_atol(dt)


def test_serving_store_guard_differential_cudagraph_replay():
    """Store guard differential holds under CUDA-graph capture/replay."""
    _require_cuda()
    dt = "bf16"
    H, Dv = _SERVING_SHAPE
    x, meta = build_serving_mapped_slack_inputs(H, Dv, _out_dtype(dt))
    _, off_out, _ref_out = _guards_on_off(
        x, H, Dv, dt, meta, runner=_run_guarded_cudagraph
    )
    sq = meta["store_slack_q"]
    off_slack_err = (off_out[sq:].float() - meta["fout_slack_seed"]).abs().max().item()
    assert off_slack_err > 5 * _out_atol(dt)


def test_serving_true_oob_no_fault():
    """Genuine OOB indices: guards-ON only (guards-OFF would abort the process)."""
    _require_cuda()
    dt = "bf16"
    H, Dv = _SERVING_SHAPE
    out_dtype = _out_dtype(dt)
    x = build_serving_true_oob_inputs(H, Dv, out_dtype)
    _run_kernel(x, H, Dv, dt)
    rows = x.fout.size(0)
    ref_out, ref_lse = torch_ref_gather(
        x,
        H,
        Dv,
        out_dtype,
        num_partial_rows=x.po.size(0),
        num_final_rows=rows,
    )
    _assert_close(x.fout, x.flse, ref_out[:rows], ref_lse[:rows], dt)


def test_flydsl_mla_reduce_degenerate_empty_tile(num_tiles):
    """Empty-tile guard: all-empty (n_splits=0) metadata never stores through the garbage q-ranges, leaving the output untouched."""
    _require_cuda()
    H, Dv = _GLM_SHAPE
    out_dtype = _out_dtype("bf16")
    x = build_degenerate_inputs(num_tiles, H, Dv, out_dtype)
    x.fout.fill_(12345.0)
    x.flse.fill_(12345.0)
    expected_out = x.fout.clone()
    expected_lse = x.flse.clone()
    make_runner(x, H, Dv, "bf16")()
    torch.cuda.synchronize()
    assert torch.equal(x.fout, expected_out)
    assert torch.equal(x.flse, expected_lse)


def test_dispatch_does_not_thread_actual_max_splits():
    """mla_decode_fwd and _mla_reduce_v1_dispatch do not accept or forward actual_max_splits; the FlyDSL wrapper auto-resolves it from reduce_indptr (capture-safe warmup cache)."""
    import inspect

    from aiter import mla
    from aiter.ops import flydsl

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

        with (
            mock.patch.object(flydsl, "flydsl_mla_reduce_v1", _capture),
            mock.patch.object(mla, "_flydsl_mla_reduce_supported", return_value=True),
        ):
            mla._mla_reduce_v1_dispatch(None, None, None, None, None, 1, 0, None, None)
        assert "actual_max_splits" not in captured["kwargs"]
        assert captured["kwargs"].get("num_kv_splits") == 0


def _dispatch_test_tensors(Dv=512):
    """Host-side placeholders: the dispatch seam is inspected, never launched."""
    return Inputs(
        torch.empty(2, 16, Dv, dtype=torch.float32),
        torch.empty(2, 16, dtype=torch.float32),
        torch.empty(2, dtype=torch.int32),
        torch.empty(1, 2, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
        torch.empty(1, 16, Dv, dtype=torch.bfloat16),
        torch.empty(1, 16, dtype=torch.float32),
    )


def test_dispatch_falls_back_outside_validated_target():
    """Unsupported architecture, prefill, and Dv=128 retain the HIP path."""
    from aiter import mla

    cases = [
        ("gfx950", 512, 1),
        ("gfx942", 128, 1),
        ("gfx942", 512, 2),
    ]
    for gfx, Dv, max_seqlen_q in cases:
        x = _dispatch_test_tensors(Dv)
        args = (*x.maps, max_seqlen_q, 0, x.fout, x.flse)
        with (
            mock.patch.object(mla, "_flydsl_mla_reduce_enabled", return_value=True),
            mock.patch.object(mla, "get_gfx", return_value=gfx),
            mock.patch.object(mla.aiter, "mla_reduce_v1") as hip_reduce,
        ):
            mla._mla_reduce_v1_dispatch(*args)
        hip_reduce.assert_called_once_with(*args)


def _assert_wrapper_rejects(expected_error, text, call):
    from aiter.ops.flydsl import mla_reduce_kernels

    with mock.patch.object(mla_reduce_kernels, "compile_mla_reduce") as compile_kernel:
        try:
            call()
        except expected_error as exc:
            assert text in str(exc), f"expected {text!r} in {exc!s}"
        else:
            raise AssertionError(f"expected {expected_error.__name__}: {text}")
        compile_kernel.assert_not_called()


def test_mla_reduce_wrapper_rejects_invalid_pointer_abi():
    """Invalid direct calls fail before JIT compilation or GPU launch."""
    _require_cuda()
    H, Dv = 16, 512
    out_dtype = _out_dtype("bf16")
    x = build_irregular_inputs([2], H, Dv, out_dtype, M=1)
    unpacked = torch.empty(1, H, Dv * 2, dtype=out_dtype, device=x.fout.device)[
        ..., ::2
    ]

    def call(*, num_kv_splits=2, actual_max_splits=2, **fields):
        """One otherwise-valid wrapper call, with ``fields`` overriding the fixture."""
        _run_wrapper(
            x._replace(**fields),
            num_kv_splits=num_kv_splits,
            actual_max_splits=actual_max_splits,
        )

    # (expected error, required substring, the single field that makes it invalid)
    cases = [
        (TypeError, "partial_output", dict(po=x.po.bfloat16())),
        (TypeError, "partial_lse", dict(pl=x.pl.bfloat16())),
        (TypeError, "reduce_indptr", dict(indptr=x.indptr.float())),
        (ValueError, "partial_lse", dict(pl=x.pl[:-1].contiguous())),
        (ValueError, "packed last dimension", dict(fout=unpacked)),
        (ValueError, "num_kv_splits", dict(num_kv_splits=LDS_MAX_SPLITS + 1)),
        (ValueError, "actual_max_splits", dict(actual_max_splits=LDS_MAX_SPLITS + 1)),
        (ValueError, "partial_output", dict(po=x.po.cpu())),
    ]
    for expected_error, text, invalid in cases:
        _assert_wrapper_rejects(expected_error, text, lambda kw=invalid: call(**kw))

    # actual_max_splits=None cannot be validated when the warmup cache misses.
    from aiter.ops.flydsl import mla_reduce_kernels

    with mock.patch.object(
        mla_reduce_kernels, "_resolve_actual_max_splits", return_value=None
    ):
        _assert_wrapper_rejects(
            RuntimeError,
            "Cannot validate `actual_max_splits`",
            lambda: call(actual_max_splits=None),
        )


def test_resolve_actual_max_splits_eager_and_capture():
    """The warmup cache resolves the true max split eager, then serves the same value under CUDA-graph capture (no device sync), and misses -> None."""
    _require_cuda()
    from aiter.ops.flydsl.mla_reduce_kernels import (
        _ACTUAL_MAX_SPLITS_CACHE,
        _resolve_actual_max_splits,
    )

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


# Adaptive launch: one block per active (tile, head).
_ADAPTIVE_SCENARIOS = [
    ("b8_s32", 8, 32),
    ("b8_s13", 8, 13),
    ("b8_s6", 8, 6),
    ("b8_s2", 8, 2),
    ("b1_s32", 1, 32),
]


def test_adaptive_launch_wrapper_vs_hip(label, active, splits, replay=False):
    """Adaptive launch (split-K off) matches HIP on the serving decode shapes, eager or under CUDA-graph capture/replay."""
    _require_cuda()
    dt = "bf16"
    x = build_serving_decode_inputs(active, splits, _out_dtype(dt))
    x.fout.zero_()
    x.flse.zero_()

    def run():
        _run_wrapper(x, num_kv_splits=splits)

    if replay:
        run_cudagraph_replay(run)
    else:
        run()
        torch.cuda.synchronize()
    ref_out, ref_lse = hip_ref_like_fout(x)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


def test_adaptive_launch_single_tile_uses_persistent():
    """bs=1 (num_final_rows==1) must not engage adaptive; still matches HIP."""
    _require_cuda()
    dt = "bf16"
    x = build_serving_decode_inputs(1, 32, _out_dtype(dt))
    x.fout.zero_()
    x.flse.zero_()
    _run_wrapper(x, num_kv_splits=32)
    torch.cuda.synchronize()
    ref_out, ref_lse = hip_ref_like_fout(x)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


def test_explicit_waves_per_eu_compile_hints():
    """WPE variants carry distinct primitive FlyDSL cache hints."""
    _require_cuda()

    compile_mla_reduce.cache_clear()
    compile_mla_reduce_splitk.cache_clear()
    try:
        common = {
            "H": 16,
            "Dv": 512,
            "out_dtype": "bf16",
            "tier": Tier.ALL,
            "output_lse": True,
            "use_reduce_final_map": True,
        }
        normal_wpe1 = compile_mla_reduce(
            **common,
            persistent=False,
            adaptive=True,
            waves_per_eu=1,
        )
        normal_wpe4 = compile_mla_reduce(
            **common,
            persistent=False,
            adaptive=True,
            waves_per_eu=4,
        )
        assert normal_wpe1 is not normal_wpe4
        assert normal_wpe1.compile_hints == {"waves_per_eu": 1}
        assert normal_wpe4.compile_hints == {"waves_per_eu": 4}

        splitk_common = {
            "H": 16,
            "Dv": 512,
            "out_dtype": "bf16",
            "K": 16,
            "output_lse": True,
        }
        partial_wpe1, combine_wpe1 = compile_mla_reduce_splitk(
            **splitk_common, waves_per_eu=1
        )
        partial_wpe4, combine_wpe4 = compile_mla_reduce_splitk(
            **splitk_common, waves_per_eu=4
        )
        assert partial_wpe1 is not partial_wpe4
        assert combine_wpe1 is not combine_wpe4
        for launcher in (partial_wpe1, combine_wpe1):
            assert launcher.compile_hints == {"waves_per_eu": 1}
        for launcher in (partial_wpe4, combine_wpe4):
            assert launcher.compile_hints == {"waves_per_eu": 4}
    finally:
        compile_mla_reduce.cache_clear()
        compile_mla_reduce_splitk.cache_clear()


def test_explicit_waves_per_eu_equivalence():
    """Explicit occupancy hints preserve normal/split-K graph-replay results."""
    _require_cuda()

    dt = "bf16"
    for active_tiles, splits in ((8, 32), (1, 128)):
        x = build_serving_decode_inputs(active_tiles, splits, _out_dtype(dt))
        ref_out, ref_lse = hip_ref_like_fout(x)

        for waves_per_eu in (1, 4):
            x.fout.zero_()
            x.flse.zero_()
            run_cudagraph_replay(
                lambda x=x, s=splits, w=waves_per_eu: _run_wrapper(
                    x, num_kv_splits=s, waves_per_eu=w
                )
            )
            _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


def test_pointer_launch_abi_has_no_tensor_annotation_warnings():
    """Normal and split-K pointer launches must not silently resolve tensors."""
    _require_cuda()

    compile_mla_reduce.cache_clear()
    compile_mla_reduce_splitk.cache_clear()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for active_tiles, splits in ((8, 32), (1, 128)):
                x = build_serving_decode_inputs(
                    active_tiles, splits, _out_dtype("bf16")
                )
                ref_out, ref_lse = hip_ref_like_fout(x)
                x.fout.zero_()
                x.flse.zero_()
                _run_wrapper(x, num_kv_splits=splits, waves_per_eu=3)
                torch.cuda.synchronize()
                _assert_close(x.fout, x.flse, ref_out, ref_lse, "bf16")

        annotation_warnings = [
            str(warning.message)
            for warning in caught
            if "annotated as 'Pointer'" in str(warning.message)
            and "resolves to 'Tensor'" in str(warning.message)
        ]
        assert not annotation_warnings, annotation_warnings
    finally:
        compile_mla_reduce.cache_clear()
        compile_mla_reduce_splitk.cache_clear()


def _check_name(fn):
    """Default registry label: the check function's name minus its ``test_`` prefix."""
    return fn.__name__.removeprefix("test_flydsl_mla_reduce_").removeprefix("test_")


def run_checks():
    """Run every invariant/correctness check. Returns a list of ``(name, exc)`` for any that failed; an empty list means everything passed. The registry below is data-driven: ``add`` labels a check after its function unless given an explicit ``name``, and ``add_each`` packs one case per positional arg."""
    checks = []

    def add(fn, *args, name=None):
        checks.append((name or _check_name(fn), fn, args))

    def add_each(cases, name_fn, fn):
        for case in cases:
            checks.append((name_fn(case), fn, (case,)))

    add_each(
        _HIP_CASES,
        lambda c: f"irregular_vs_hip[{c[0]}_{c[4]}]",
        test_flydsl_mla_reduce_irregular_vs_hip,
    )
    add_each(
        _TORCH_CASES,
        lambda c: f"irregular_vs_torch_ref[{c[0]}_{c[4]}]",
        test_flydsl_mla_reduce_irregular_vs_torch_ref,
    )
    add_each(
        _SMOKE_CASES,
        lambda c: f"uniform_smoke[H{c[0][0]}_Dv{c[0][1]}_{c[1]}_s{c[2]}]",
        test_flydsl_mla_reduce_uniform_smoke,
    )
    add_each(
        _GRAPH_CASES,
        lambda c: f"cudagraph_replay[{c[0]}]",
        test_flydsl_mla_reduce_cudagraph_replay,
    )
    add(test_serving_sparse_grid, name="serving_sparse_grid_vs_hip")
    add(test_serving_sparse_grid, True, name="serving_sparse_grid_cudagraph_replay")
    add_each(
        [4, 8, 16],
        lambda K: f"splitk_b1_s128_vs_torch_ref[K={K}]",
        test_splitk_b1_s128_vs_torch_ref,
    )
    add(test_splitk_b1_s128_vs_hip)
    add(
        test_splitk_b1_s128_vs_torch_ref,
        None,
        True,
        name="splitk_b1_s128_cudagraph_replay",
    )
    add(test_splitk_planner_boundaries)
    add(test_da_splitk_wrapper_vs_hip)
    for skipped_splits in (0, 1):
        add(
            test_da_splitk_preserves_stage1_rows,
            skipped_splits,
            name=f"da_splitk_preserves_stage1_rows[splits={skipped_splits}]",
        )
        add(
            test_da_splitk_preserves_stage1_rows,
            skipped_splits,
            True,
            name=f"da_splitk_preserves_stage1_rows_cudagraph[splits={skipped_splits}]",
        )
    add(test_da_splitk_capture_safe_varying_splits)
    add(test_derive_actual_max_splits)
    add(test_actual_max_splits_wrapper_loose_budget_correct)
    add(test_actual_max_splits_wrapper_cudagraph_replay)
    add(test_serving_stale_indptr_cudagraph_replay)
    add(test_serving_gather_guard_differential)
    add(test_serving_store_guard_differential)
    add(test_serving_gather_guard_differential_cudagraph_replay)
    add(test_serving_store_guard_differential_cudagraph_replay)
    add(test_serving_true_oob_no_fault)
    add_each(
        _DEGEN_TILES,
        lambda n: f"degenerate_empty_tile[tiles{n}]",
        test_flydsl_mla_reduce_degenerate_empty_tile,
    )
    add(test_dispatch_does_not_thread_actual_max_splits)
    add(test_dispatch_falls_back_outside_validated_target)
    add(test_mla_reduce_wrapper_rejects_invalid_pointer_abi)
    add(test_resolve_actual_max_splits_eager_and_capture)
    for label, active, splits in _ADAPTIVE_SCENARIOS:
        add(
            test_adaptive_launch_wrapper_vs_hip,
            label,
            active,
            splits,
            name=f"adaptive_launch_wrapper_vs_hip[{label}]",
        )
    add(
        test_adaptive_launch_wrapper_vs_hip,
        "b8_s32",
        8,
        32,
        True,
        name="adaptive_launch_cudagraph_replay",
    )
    add(test_adaptive_launch_single_tile_uses_persistent)
    add(test_explicit_waves_per_eu_compile_hints)
    add(test_explicit_waves_per_eu_equivalence)
    add(test_pointer_launch_abi_has_no_tensor_annotation_warnings)

    failures = []
    for name, fn, args in checks:
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001 - collect every failure, keep going
            failures.append((name, exc))
    return failures


# Perf scoreboard: serving, uniform, and irregular decode scenarios.

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


def _reduce_roofline(total_splits, final_rows, H, Dv, out_dtype):
    """FLOPs (online-softmax weighted-sum FMA) and byte traffic for the reduce, given the total partial-row reads and final-row writes for the scenario."""
    out_bytes = torch.finfo(out_dtype).bits // 8
    flops = 2 * total_splits * H * Dv
    nbytes = (
        total_splits * H * Dv * 4  # partial_output fp32 read
        + total_splits * H * 4  # partial_lse   fp32 read
        + final_rows * H * Dv * out_bytes  # final_output  write
        + final_rows * H * 4  # final_lse     fp32 write
    )
    return flops, nbytes


def _reduce_candidates(x, Dv, M, kv_splits):
    """``wrapper`` (+ ``hip`` when the Dv=512 template applies) candidates shared by the three reduce perf sweeps."""
    candidates = {"wrapper": lambda: _run_wrapper(x, M, num_kv_splits=kv_splits)}
    if Dv == 512:
        # mla_reduce_v1 signature: (..., max_seqlen_q, num_kv_splits, out, lse)
        # HIP MLA_REDUCE_ROUTER only has a Dv=512 template; skip it elsewhere.
        candidates["hip"] = lambda: aiter.mla_reduce_v1(*x.maps, M, 0, x.fout, x.flse)
    return candidates


def _bench_reduce_candidates(
    candidates, x, ref_out, dtype, flops, nbytes, op_name, ref_lse=None
):
    """Time + validate each candidate; the us/graph-us/TFLOPS/TB-s/err columns shared by the mla_reduce perf sweeps. ``ref_lse`` is only checked (not tabulated) when provided, matching the serving sweep's extra invariant."""
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        x.fout.zero_()
        x.flse.zero_()
        _, us = run_perftest(fn, num_warmup=25, num_iters=100)
        err = checkAllclose(
            ref_out.to(dtypes.fp32),
            x.fout.clone().to(dtypes.fp32),
            rtol=1e-2,
            atol=_out_atol(dtype),
            msg=f"{name}: {op_name} out",
            printLog=False,
        )
        if ref_lse is not None:
            checkAllclose(
                ref_lse.to(dtypes.fp32),
                x.flse.clone().to(dtypes.fp32),
                rtol=1e-2,
                atol=1e-3,
                msg=f"{name}: {op_name} lse",
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
def test_mla_reduce(active, splits, H, Dv, dtype):
    x = build_serving_decode_inputs(active, splits, dtype, H=H, Dv=Dv)
    # torch online-softmax reference over the active prefix only (tail tiles are
    # empty and skipped by both the kernels and the ref).
    active_prefix = x._replace(indptr=x.indptr[: active + 1], fmap=x.fmap[:active])
    ref_out, ref_lse = torch_ref_gather(active_prefix, H, Dv, dtype)
    candidates = _reduce_candidates(x, Dv, 1, splits)
    flops, nbytes = _reduce_roofline(active * splits, active, H, Dv, dtype)
    return _bench_reduce_candidates(
        candidates, x, ref_out, dtype, flops, nbytes, "mla_reduce", ref_lse
    )


@benchmark()
def test_mla_reduce_uniform(tiles, splits, H, Dv, M, dtype):
    """Dense/uniform occupancy control: every tile has ``splits`` splits."""
    x = build_inputs(tiles, splits, H, Dv, dtype, M=M)
    ref_out, _ref_lse = torch_ref(x.po, x.pl, tiles, splits, H, Dv, dtype, M=M)
    candidates = _reduce_candidates(x, Dv, M, splits)
    flops, nbytes = _reduce_roofline(tiles * splits, tiles * M, H, Dv, dtype)
    return _bench_reduce_candidates(
        candidates, x, ref_out, dtype, flops, nbytes, "mla_reduce_uniform"
    )


@benchmark()
def test_mla_reduce_irregular(splits_per_tile, gap_stride, pool_slack, H, Dv, dtype):
    """Irregular per-tile split cost factors: tier mismatch, gaps, pool slack."""
    x = build_irregular_inputs(
        list(splits_per_tile),
        H,
        Dv,
        dtype,
        gap_stride=gap_stride,
        pool_slack=pool_slack,
    )
    ref_out, _ref_lse = torch_ref_gather(x, H, Dv, dtype)
    candidates = _reduce_candidates(x, Dv, 1, max(splits_per_tile))
    total_splits = sum(splits_per_tile)
    active = sum(1 for s in splits_per_tile if s > 1)
    flops, nbytes = _reduce_roofline(total_splits, active, H, Dv, dtype)
    return _bench_reduce_candidates(
        candidates, x, ref_out, dtype, flops, nbytes, "mla_reduce_irregular"
    )


def _log_sweep_table(rows, title):
    """Render one perf sweep's rows as a markdown table, same log format shared by all three ``run_bench`` sweeps."""
    df = pd.DataFrame(rows)
    aiter.logger.info("%s (markdown):\n%s", title, df.to_markdown(index=False))


def run_bench(args):
    """Run the three perf sweeps and print one markdown table each."""
    for dtype in args.dtype:
        rows = [
            test_mla_reduce(active, splits, H, Dv, dtype)
            for (H, Dv), (active, splits) in itertools.product(args.hdv, args.scenario)
        ]
        _log_sweep_table(rows, "mla_reduce GLM-5.2 serving summary")

        rows = [
            test_mla_reduce_uniform(tiles, splits, H, Dv, 1, dtype)
            for (H, Dv), tiles, splits in itertools.product(
                args.hdv, args.tiles, args.uniform_splits
            )
        ]
        _log_sweep_table(rows, "mla_reduce uniform (occupancy) summary")

        rows = [
            test_mla_reduce_irregular(spt, gap_stride, pool_slack, H, Dv, dtype)
            for (H, Dv), spt, gap_stride, pool_slack in itertools.product(
                args.hdv, args.splits_per_tile, args.gap_stride, args.pool_slack
            )
        ]
        _log_sweep_table(rows, "mla_reduce irregular (cost-factor) summary")


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
