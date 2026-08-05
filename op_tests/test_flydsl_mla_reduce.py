# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance coverage for FlyDSL MLA reduce."""

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
# Validated on gfx942; gfx950 is permitted but unvalidated.
MLA_REDUCE_SUPPORTED_GFX = ["gfx942"]


def mla_reduce_out_dtype(dt: str) -> torch.dtype:
    return torch.bfloat16 if dt == "bf16" else torch.float16


def mla_reduce_out_atol(dt: str | torch.dtype) -> float:
    return 6.3e-2 if dt in ("bf16", torch.bfloat16) else 2e-3


_out_dtype = mla_reduce_out_dtype
_out_atol = mla_reduce_out_atol


class Inputs(NamedTuple):
    """Reduce fixture in wrapper ABI order; raw launches unpack maps by name."""

    po: torch.Tensor  # fp32 [rows, H, Dv] or decode [rows, 1, H, Dv]
    pl: torch.Tensor  # fp32 [rows, H] or decode [rows, 1, H, 1]
    indptr: torch.Tensor  # i32 [num_reduce_tile + 1], CSR over splits
    fmap: torch.Tensor  # i32 [num_reduce_tile, 2], {q_start, q_end}
    pmap: torch.Tensor  # i32 [indptr[-1]], partial-pool gather rows
    fout: torch.Tensor  # out_dtype [num_final_rows, H, Dv]
    flse: torch.Tensor  # fp32 [num_final_rows, H]

    @property
    def maps(self):
        """Return metadata in wrapper ABI order."""
        return self[:5]


def _ptrs(*tensors, dtype=torch.float32):
    """Convert tensors to validated launcher pointers."""
    return tuple(_pointer_arg(tensor, dtype) for tensor in tensors)


def _rand_partials(rows, H, Dv, device, g):
    """Build fp32 partial buffers with a widened LSE range."""
    po = torch.randn(rows, H, Dv, dtype=torch.float32, device=device, generator=g)
    pl = torch.randn(rows, H, dtype=torch.float32, device=device, generator=g) * 2.0
    return po, pl


def _garbage_final_map(num_tiles, active_tiles, device):
    """Build active identity ranges with an unmapped tail."""
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
    """Build irregular decode metadata with gapped partial-map support."""
    g = torch.Generator(device=device).manual_seed(seed)
    num_tiles = len(splits_per_tile)
    total_splits = int(sum(int(s) for s in splits_per_tile))

    indptr_host = [0]
    for s in splits_per_tile:
        indptr_host.append(indptr_host[-1] + int(s))
    indptr = torch.tensor(indptr_host, dtype=torch.int32, device=device)

    if total_splits > 0:
        slot = torch.arange(total_splits, dtype=torch.int32, device=device)
        pmap = slot * (gap_stride * M)
        max_base = int(pmap.max().item())
    else:
        pmap = torch.zeros(1, dtype=torch.int32, device=device)
        max_base = 0
    num_partial_rows = max_base + M + pool_slack * M

    po, pl = _rand_partials(num_partial_rows, H, Dv, device, g)

    q_start = torch.arange(num_tiles, dtype=torch.int32, device=device) * M
    fmap = torch.stack([q_start, q_start + M], dim=1).contiguous()
    for t, s in enumerate(splits_per_tile):
        if int(s) <= 1:
            fmap[t, 0] = 1 << 24
            fmap[t, 1] = (1 << 24) + M

    fout = torch.empty(num_tiles * M, H, Dv, dtype=out_dtype, device=device)
    flse = torch.empty(num_tiles * M, H, dtype=torch.float32, device=device)
    return Inputs(po, pl, indptr, fmap, pmap, fout, flse)


def build_inputs(num_tiles, num_splits, H, Dv, out_dtype, M=1, device="cuda", seed=0):
    """Build dense decode metadata."""
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
    """Build all-empty metadata."""
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
    """Build sparse serving-decode metadata."""
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
    """Gather-based reference; optional bounds model guards-on behavior."""
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
    """Run the HIP reference into zeroed output buffers."""
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
    """Build batch-8 serving metadata with a sentinel tail."""
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
    indptr = torch.tensor(indptr_host, dtype=torch.int32, device=device)

    po, pl = _rand_partials(num_partial_rows, H, Dv, device, g)

    pmap = torch.empty(num_partial_rows, dtype=torch.int32, device=device)
    pmap[:total_splits] = torch.arange(total_splits, dtype=torch.int32, device=device)
    pmap[total_splits:] = torch.randint(
        -(1 << 30),
        1 << 30,
        (num_partial_rows - total_splits,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )

    fmap = _garbage_final_map(num_reduce_tile, active_tiles, device)
    fout = torch.zeros(active_tiles, H, Dv, dtype=out_dtype, device=device)
    flse = torch.zeros(active_tiles, H, dtype=torch.float32, device=device)
    return Inputs(po, pl, indptr, fmap, pmap, fout, flse)


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
    """Build mapped gather/store slack for guard-differential tests."""
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
    indptr = torch.tensor(indptr_host, dtype=torch.int32, device=device)

    logical_partial_rows = 256
    alloc_partial_rows = logical_partial_rows + slack_p
    logical_final_rows = active_tiles
    alloc_final_rows = logical_final_rows + slack_f

    po, pl = _rand_partials(alloc_partial_rows, H, Dv, device, g)

    pmap = torch.arange(total_splits, dtype=torch.int32, device=device)
    # Mapped slack makes guards-off gathering visibly wrong.
    slack_p_row = logical_partial_rows
    pmap[splits_per_active // 2] = slack_p_row
    po[slack_p_row].fill_(1000.0)
    tile0_lse_max = pl[:splits_per_active].max().item()
    pl[slack_p_row].fill_(tile0_lse_max + 5.0)

    # A fake-active tail targets output slack without store guards.
    fmap = _garbage_final_map(num_tiles, active_tiles, device)
    store_slack_q = logical_final_rows
    fmap[store_tile, 0] = store_slack_q
    fmap[store_tile, 1] = store_slack_q + 1

    last = int(indptr[num_tiles].item())
    t0_store = int(indptr[store_tile].item())
    n_splits_store = int(indptr[store_tile + 1].item() - indptr[store_tile].item())
    assert n_splits_store >= 2, "store discriminator must be fake-active"
    assert t0_store != last, "store discriminator must not hit sentinel skip"

    fout = torch.zeros(alloc_final_rows, H, Dv, dtype=out_dtype, device=device)
    flse = torch.zeros(alloc_final_rows, H, dtype=torch.float32, device=device)
    fout_slack_seed = 42.0
    fout[store_slack_q:].fill_(fout_slack_seed)

    meta = {
        "logical_partial_rows": logical_partial_rows,
        "logical_final_rows": logical_final_rows,
        "gather_tile": gather_tile,
        "gather_q_row": gather_tile,
        "store_tile": store_tile,
        "store_slack_q": store_slack_q,
        "fout_slack_seed": fout_slack_seed,
    }
    return Inputs(po, pl, indptr, fmap, pmap, fout, flse), meta


def build_serving_true_oob_inputs(
    H=16,
    Dv=512,
    out_dtype=torch.bfloat16,
    device="cuda",
    seed=0,
    num_tiles=64,
):
    """Build true-OOB metadata for guards-on testing."""
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
    indptr = torch.tensor(indptr_host, dtype=torch.int32, device=device)

    num_partial_rows = 128
    po, pl = _rand_partials(num_partial_rows, H, Dv, device, g)

    # Tail gather rows are outside the partial pool.
    pmap = torch.arange(sentinel, dtype=torch.int32, device=device)
    tail_t0 = int(indptr[tail_tile].item())
    for i in range(tail_splits):
        pmap[tail_t0 + i] = num_partial_rows + i

    fmap = _garbage_final_map(num_tiles, active_tiles, device)
    fout = torch.zeros(active_tiles, H, Dv, dtype=out_dtype, device=device)
    flse = torch.zeros(active_tiles, H, dtype=torch.float32, device=device)
    return Inputs(po, pl, indptr, fmap, pmap, fout, flse)


def build_serving_stale_indptr_inputs(
    H=16,
    Dv=512,
    out_dtype=torch.bfloat16,
    device="cuda",
    seed=0,
):
    """Build stale batch-8 metadata for a batch-1 guard test."""
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
    indptr = torch.tensor(indptr_host, dtype=torch.int32, device=device)

    logical_partial_rows = batch1_splits
    slack_p = 256
    alloc_partial_rows = logical_partial_rows + slack_p
    logical_final_rows = 1
    slack_f = 8
    alloc_final_rows = logical_final_rows + slack_f

    po, pl = _rand_partials(alloc_partial_rows, H, Dv, device, g)

    pmap = torch.arange(sentinel, dtype=torch.int32, device=device)
    # Stale pmap rows are outside the batch-1 partial bound.
    stale_base = batch1_splits
    for t in range(1, active_tiles_batch8):
        t0 = indptr_host[t]
        for s in range(splits_per_active):
            pmap[t0 + s] = stale_base + s
    po[stale_base : stale_base + splits_per_active].fill_(500.0)
    tile1_lse_max = pl[stale_base : stale_base + splits_per_active].max()
    pl[stale_base].fill_(tile1_lse_max + 5.0)

    # Stale fmap rows are outside the batch-1 output bound.
    fmap = _garbage_final_map(num_reduce_tile, active_tiles_batch8, device)

    fout = torch.zeros(alloc_final_rows, H, Dv, dtype=out_dtype, device=device)
    flse = torch.zeros(alloc_final_rows, H, dtype=torch.float32, device=device)
    fout_slack_seed = 42.0
    fout[logical_final_rows:].fill_(fout_slack_seed)

    meta = {
        "logical_partial_rows": logical_partial_rows,
        "logical_final_rows": logical_final_rows,
        "gather_tile": 1,
        "gather_q_row": 1,
        "store_tile": 2,
        "store_slack_q": 2,
        "fout_slack_seed": fout_slack_seed,
    }
    return Inputs(po, pl, indptr, fmap, pmap, fout, flse), meta


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
    """Compile a raw-pointer runner; ``tier=None`` uses ``Tier.ALL``."""
    po, pl, indptr, fmap, pmap, fout, flse = x
    num_tiles = fmap.shape[0]
    num_cu = torch.cuda.get_device_properties(0).multi_processor_count
    compile_tier = Tier.ALL if tier is None else tier
    if num_partial_rows is None:
        num_partial_rows = int(po.size(0))
    if num_final_rows is None:
        num_final_rows = int(fout.size(0))
    # Inspect split-K metadata and allocate scratch before capture.
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
                _ptrs(po, pl)
                + _ptrs(indptr, pmap, dtype=torch.int32)
                + _ptrs(sk_acc, sk_ml)
                + (int(num_partial_rows), int(num_slots * K))
            )
            combine_head = (
                _ptrs(indptr, fmap, dtype=torch.int32)
                + _ptrs(sk_acc, sk_ml)
                + _ptrs(fout, dtype=fout.dtype)
                + _ptrs(flse)
                + (int(fout.stride(0)), int(fout.stride(1)))
                + (int(num_final_rows), int(num_slots))
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
        _ptrs(po, pl)
        + _ptrs(indptr, pmap, fmap, dtype=torch.int32)
        + _ptrs(fout, dtype=fout.dtype)
        + _ptrs(flse)
        + (int(fout.stride(0)), int(fout.stride(1)))
        + (int(num_cu), int(num_tiles), int(M))
        + (int(num_partial_rows), int(num_final_rows))
    )

    def run():
        _run_compiled(kernel, *head, fx.Stream(torch.cuda.current_stream()))

    return run


def capture_graph(fn, num_warmup=3, num_iters=1):
    """Warm up and capture ``fn`` on a side stream."""
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
    return graph


def bench_cudagraph(fn, num_warmup=25, num_iters=100):
    """Time CUDA-graph replay in ms per iteration."""
    graph = capture_graph(fn, num_warmup, num_iters)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / num_iters


def run_cudagraph_replay(fn, num_warmup=3, num_replays=3):
    """Capture and replay ``fn``."""
    graph = capture_graph(fn, num_warmup)
    for _ in range(max(1, num_replays)):
        graph.replay()
    torch.cuda.synchronize()


def _run_kernel(x, H, Dv, dt, M=1, *, replay=False, **runner_kwargs):
    """Run the raw-pointer kernel eagerly or under graph replay."""
    x.fout.zero_()
    x.flse.zero_()
    run = make_runner(x, H, Dv, dt, M, **runner_kwargs)
    if replay:
        run_cudagraph_replay(run)
    else:
        run()
        torch.cuda.synchronize()


def _run_wrapper(x, M=1, **kwargs):
    """Run the production wrapper in place."""
    flydsl_mla_reduce_v1(*x.maps, M, x.fout, x.flse, **kwargs)


def _single_final_row(x):
    """Restrict outputs to one decode row."""
    return x._replace(fout=x.fout[:1].contiguous(), flse=x.flse[:1].contiguous())


@contextmanager
def _env(**kwargs):
    """Temporarily set environment variables; ``None`` unsets a variable."""
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


# HIP has a Dv=512 template; Dv=256 uses the torch reference.
_HIP_SHAPE = (128, 512)
_TORCH_REF_SHAPE = (8, 256)

# (name, splits_per_tile, gap_stride, M)
_IRREGULAR_SCENARIOS = [
    ("tier_mismatch", [8, 304], 1, 1),
    ("variable_splits", [4, 32, 8, 64], 1, 1),
    ("gapped_pmap", [8, 8, 8, 8], 4, 1),
    ("empty_middle", [8, 0, 16, 8], 1, 1),
    ("mlds_boundary", [300], 1, 1),
    ("mlds_max", [304], 1, 1),
    ("pool_oversize", [8, 304], 8, 1),
]

# Limit fp16 coverage to layout-sensitive cases.
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

# Dense tier-smoke cases.
_SMOKE_TILES = 4
_SMOKE_CASES = [
    (_HIP_SHAPE, "hip", 2),  # simple
    (_HIP_SHAPE, "hip", 8),  # m64
    (_HIP_SHAPE, "hip", 64),  # m64 upper
    (_HIP_SHAPE, "hip", 256),  # m256
    (_TORCH_REF_SHAPE, "torch", 2),  # simple
    (_TORCH_REF_SHAPE, "torch", 8),  # m64
    (_TORCH_REF_SHAPE, "torch", 32),  # serving split cap
    (_TORCH_REF_SHAPE, "torch", 256),  # stress
]

# CUDA-graph replay cases: (name, shape, ref, splits, gap, M).
_GRAPH_CASES = [
    ("tier_mismatch", _TORCH_REF_SHAPE, "torch", [8, 304], 1, 1),
    ("gapped_pmap", _HIP_SHAPE, "hip", [8, 8, 8, 8], 4, 1),
    ("empty_middle", _TORCH_REF_SHAPE, "torch", [8, 0, 16, 8], 1, 1),
    ("mlds_max", _TORCH_REF_SHAPE, "torch", [304], 1, 1),
    ("small_split", _TORCH_REF_SHAPE, "torch", [32] * 4, 1, 1),
]

_DEGEN_TILES = [2, 4]


def _require_cuda():
    """Require a validated CUDA target."""
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
        x.fout[meta["logical_final_rows"] :].fill_(meta["fout_slack_seed"])
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
    """Run guarded reduction under CUDA-graph replay."""
    out = torch.zeros_like(x.fout)
    lse = torch.zeros_like(x.flse)
    out[meta["logical_final_rows"] :].fill_(meta["fout_slack_seed"])
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
    """Run guards on and off, returning outputs plus the masked reference."""
    ref_out, ref_lse = _masking_ref(x, H, Dv, _out_dtype(dt), meta)
    on_out, on_lse = runner(x, H, Dv, dt, meta, disable_guards=False)
    off_out, _off_lse = runner(x, H, Dv, dt, meta, disable_guards=True)
    _assert_close(
        *_logical_rows(on_out, on_lse, meta),
        *_logical_rows(ref_out, ref_lse, meta),
        dt,
    )
    return on_out, off_out, ref_out


def _assert_guard_differentials(x, H, Dv, dt, meta, runner=_run_guarded):
    """Assert guards block mapped gather and store corruption."""
    on_out, off_out, ref_out = _guards_on_off(x, H, Dv, dt, meta, runner=runner)
    atol = _out_atol(dt)
    q_row = meta["gather_q_row"]
    gather_err = (off_out[q_row].float() - ref_out[q_row].float()).abs().max().item()
    assert gather_err > 5 * atol, (
        f"gather guard differential failed: guards-OFF row {q_row} "
        f"max_abs_err={gather_err:.3e} <= {5 * atol}"
    )
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
    """Build and run an irregular case."""
    x = build_irregular_inputs(spt, H, Dv, _out_dtype(dt), M=M, gap_stride=gap)
    _run_kernel(x, H, Dv, dt, M)
    return x


def test_irregular_vs_hip(case):
    """Irregular metadata matches HIP kn_mla_reduce_v1 (DeepSeek shape, Dv=512)."""
    _require_cuda()
    _name, spt, gap, M, dt = case
    H, Dv = _HIP_SHAPE
    x = _run_irregular(spt, gap, M, H, Dv, dt)
    ref_out, ref_lse = hip_ref(x, len(spt), H, Dv, _out_dtype(dt), M)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


def test_irregular_vs_torch_ref(case):
    """Irregular metadata matches the gather-based torch reference at narrow Dv."""
    _require_cuda()
    _name, spt, gap, M, dt = case
    H, Dv = _TORCH_REF_SHAPE
    x = _run_irregular(spt, gap, M, H, Dv, dt)
    ref_out, ref_lse = torch_ref_gather(x, H, Dv, _out_dtype(dt), M)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


def test_uniform_smoke(case):
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


def test_irregular_cudagraph_replay(case):
    """Validate irregular metadata under CUDA-graph replay."""
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
    """Validate the sparse serving grid eagerly or under replay."""
    _require_cuda()
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    x = build_serving_sparse_grid_inputs(*_SERVING_SHAPE, out_dtype=out_dtype)
    _run_kernel(x, *_SERVING_SHAPE, dt, replay=replay)
    ref_out, ref_lse = hip_ref_like_fout(x)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


# Direct split-K coverage.
_SPLITK_H, _SPLITK_DV = 16, 512
_SPLITK_GRID = SERVING_NUM_REDUCE_TILE


def _build_splitk_b1_s128(out_dtype):
    """Build the b1_s128 split-K fixture."""
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
    """Run b1_s128 through direct split-K."""
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
    """Compare b1_s128 split-K against the torch reference."""
    _require_cuda()
    x, dt, out_dtype = _run_splitk_b1_s128(K, replay)
    ref_out, ref_lse = torch_ref_gather(x, _SPLITK_H, _SPLITK_DV, out_dtype)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


def test_splitk_b1_s128_vs_hip():
    """Compare b1_s128 split-K against HIP."""
    _require_cuda()
    x, dt, _ = _run_splitk_b1_s128()
    ref_out, ref_lse = hip_ref_like_fout(x)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


# Split-K planner cases: (kwargs, expected result).
_PLAN_SPLITK_CASES = [
    ({"active_tiles": 1, "max_splits": 128}, (True, 16, 16)),
    ({"active_tiles": 1, "max_splits": 8}, (False, 1, 0)),  # below min_splits
    ({"active_tiles": 20, "max_splits": 128}, (False, 1, 0)),  # saturated grid
    (
        {"active_tiles": 1, "max_splits": 128, "max_seqlen_q": 2},
        (False, 1, 0),
    ),  # non-single-token query width is outside the split-K scope
]
_PLAN_CAPTURE_SAFE_CASES = [
    ({"num_final_rows": 32, "num_kv_splits": 128}, (False, 1, 0)),  # saturated grid
    ({"num_final_rows": 1, "num_kv_splits": 32}, (False, 1, 0)),  # below min_splits
    (
        {"num_final_rows": 1, "num_kv_splits": 304, "actual_max_splits": 8},
        (False, 1, 0),
    ),
    (
        {"num_final_rows": 1, "num_kv_splits": 304, "actual_max_splits": 128},
        (True, 16, 16),
    ),
]


def test_splitk_planner_boundaries():
    """Validate direct and capture-safe split-K planner boundaries."""
    base = {"H": 16, "max_seqlen_q": 1, "num_cu": 304}
    for planner, cases in (
        (plan_splitk, _PLAN_SPLITK_CASES),
        (plan_splitk_capture_safe, _PLAN_CAPTURE_SAFE_CASES),
    ):
        for kwargs, expected in cases:
            got = planner(**{**base, **kwargs})
            assert got == expected, (planner.__name__, kwargs, got, expected)


def _build_da_single_tile(out_dtype, splits, pool=304):
    """Build a single-tile fixture with a mutable CSR width."""
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
    """Compare device-adaptive split-K with HIP."""
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
    # Give the skipped tile a stale q-range.
    x.fmap[0] = torch.tensor([0, 1], dtype=torch.int32, device=x.fmap.device)
    x.fout.fill_(7.0)
    x.flse.fill_(9.0)
    return x


def test_da_splitk_preserves_stage1_rows(skipped_splits, replay=False):
    """Ensure split-K preserves stage-1 output rows."""
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
    """Reuse a split-K capture while changing the CSR width."""
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

    graph = capture_graph(run)

    # Reuse the capture with changed CSR widths.
    for s_k in [128, 304, 64, 200, 8, 96]:
        x.indptr[1] = s_k  # mutate the captured CSR in place (host->device copy)
        x.fout.zero_()
        x.flse.zero_()
        graph.replay()
        torch.cuda.synchronize()
        ref_out, ref_lse = torch_ref_gather(x, _SPLITK_H, _SPLITK_DV, out_dtype)
        _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


def test_derive_actual_max_splits():
    """Derive the maximum CSR tile width."""
    indptr = torch.tensor([0, 8, 12, 12], dtype=torch.int32, device="cuda")
    assert derive_actual_max_splits(indptr) == 8


def test_actual_max_splits_wrapper_loose_budget_correct():
    """Validate a loose split-K budget with a small actual width."""
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
    """Validate actual_max_splits under graph replay."""
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
    """Validate stale metadata under CUDA-graph replay."""
    _require_cuda()
    dt = "bf16"
    x, meta = build_serving_stale_indptr_inputs(
        *_SERVING_SHAPE, out_dtype=_out_dtype(dt)
    )
    _assert_guard_differentials(
        x, *_SERVING_SHAPE, dt, meta, runner=_run_guarded_cudagraph
    )


def test_serving_guard_diffs(replay=False):
    """Validate mapped gather/store guards eagerly or under replay."""
    _require_cuda()
    dt = "bf16"
    H, Dv = _SERVING_SHAPE
    x, meta = build_serving_mapped_slack_inputs(H, Dv, _out_dtype(dt))
    runner = _run_guarded_cudagraph if replay else _run_guarded
    _assert_guard_differentials(x, H, Dv, dt, meta, runner=runner)


def test_serving_true_oob_no_fault():
    """Validate guards-on behavior for genuine OOB metadata."""
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


def test_degenerate_empty_tile(num_tiles):
    """Ensure empty tiles leave outputs unchanged."""
    _require_cuda()
    H, Dv = _TORCH_REF_SHAPE
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


def _decode_fwd_reduce_tensors(Dv=512, split_len=1, flat=False):
    """Build host placeholders for decode or the old flat prefill layout."""
    rows, H = SERVING_PARTIAL_POOL, _SERVING_SHAPE[0]
    final_rows = 8
    if flat:
        po = torch.empty(rows, H, Dv, dtype=torch.float32)
        pl = torch.empty(rows, H, dtype=torch.float32)
    else:
        po = torch.empty(rows, split_len, H, Dv, dtype=torch.float32)
        pl = torch.empty(rows, split_len, H, 1, dtype=torch.float32)
    return Inputs(
        po,
        pl,
        torch.empty(SERVING_NUM_REDUCE_TILE + 1, dtype=torch.int32),
        torch.empty(SERVING_NUM_REDUCE_TILE, 2, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
        torch.empty(final_rows, H, Dv, dtype=torch.bfloat16),
        torch.empty(final_rows, H, dtype=torch.float32),
    )


def test_dispatch_scope_and_decode_fwd_layout():
    """Validate dispatch scope and decode layout adaptation."""
    import inspect

    from aiter import mla
    from aiter.ops import flydsl

    for fn in (mla._mla_decode_reduce_v1_dispatch, mla.mla_decode_fwd):
        assert "actual_max_splits" not in inspect.signature(fn).parameters

    x = _decode_fwd_reduce_tensors()
    assert tuple(x.po.shape) == (606, 1, 16, 512)
    assert tuple(x.po.stride()) == (8192, 8192, 512, 1)
    assert tuple(x.pl.shape) == (606, 1, 16, 1)
    assert tuple(x.pl.stride()) == (16, 16, 1, 1)
    noncontiguous = torch.empty(606, 1, 16, 1024, dtype=torch.float32)[..., ::2]
    cases = [
        ("gfx942", "gfx942", x, 1, True),
        ("gfx950", "gfx950", x, 1, True),
        ("unsupported architecture", "gfx90a", _decode_fwd_reduce_tensors(), 1, False),
        ("Dv=128", "gfx942", _decode_fwd_reduce_tensors(128), 1, False),
        ("multi-token decode", "gfx942", _decode_fwd_reduce_tensors(), 2, False),
        (
            "flat partial layout",
            "gfx942",
            _decode_fwd_reduce_tensors(flat=True),
            1,
            False,
        ),
        (
            "noncontiguous partial output",
            "gfx942",
            _decode_fwd_reduce_tensors()._replace(po=noncontiguous),
            1,
            False,
        ),
        (
            "non-singleton split axis",
            "gfx942",
            _decode_fwd_reduce_tensors(split_len=2),
            1,
            False,
        ),
    ]
    with _env(AITER_MLA_REDUCE_FLYDSL="1"):
        assert mla._flydsl_mla_reduce_enabled()
        for name, gfx, x, max_seqlen_q, expect_flydsl in cases:
            args = (*x.maps, max_seqlen_q, 0, x.fout, x.flse)
            with (
                mock.patch.object(mla, "get_gfx", return_value=gfx),
                mock.patch.object(flydsl, "flydsl_mla_reduce_v1") as flydsl_reduce,
                mock.patch.object(mla.aiter, "mla_reduce_v1") as hip_reduce,
            ):
                mla._mla_decode_reduce_v1_dispatch(*args)

            if expect_flydsl:
                assert flydsl_reduce.call_count == 1, name
                assert hip_reduce.call_count == 0, name
                assert "actual_max_splits" not in flydsl_reduce.call_args.kwargs, name
                assert flydsl_reduce.call_args.kwargs.get("num_kv_splits") == 0, name
                partial_output, partial_lse = flydsl_reduce.call_args.args[:2]
                assert tuple(partial_output.shape) == (606, 16, 512), name
                assert tuple(partial_output.stride()) == (8192, 512, 1), name
                assert partial_output.data_ptr() == x.po.data_ptr(), name
                assert tuple(partial_lse.shape) == (606, 16), name
                assert tuple(partial_lse.stride()) == (16, 1), name
                assert partial_lse.data_ptr() == x.pl.data_ptr(), name
            else:
                assert flydsl_reduce.call_count == 0, name
                assert hip_reduce.call_count == 1, name
                assert hip_reduce.call_args == mock.call(*args), name


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
    """Reject invalid wrapper ABI inputs before compilation."""
    _require_cuda()
    H, Dv = 16, 512
    out_dtype = _out_dtype("bf16")
    x = build_irregular_inputs([2], H, Dv, out_dtype, M=1)
    unpacked = torch.empty(1, H, Dv * 2, dtype=out_dtype, device=x.fout.device)[
        ..., ::2
    ]

    def call(*, num_kv_splits=2, actual_max_splits=2, **fields):
        """Call the wrapper with fixture overrides."""
        _run_wrapper(
            x._replace(**fields),
            num_kv_splits=num_kv_splits,
            actual_max_splits=actual_max_splits,
        )

    # (exception, message fragment, override)
    cases = [
        (TypeError, "partial_output", {"po": x.po.bfloat16()}),
        (TypeError, "partial_lse", {"pl": x.pl.bfloat16()}),
        (TypeError, "reduce_indptr", {"indptr": x.indptr.float()}),
        (ValueError, "partial_lse", {"pl": x.pl[:-1].contiguous()}),
        (ValueError, "packed last dimension", {"fout": unpacked}),
        (ValueError, "num_kv_splits", {"num_kv_splits": LDS_MAX_SPLITS + 1}),
        (
            ValueError,
            "actual_max_splits",
            {"actual_max_splits": LDS_MAX_SPLITS + 1},
        ),
        (ValueError, "partial_output", {"po": x.po.cpu()}),
    ]
    for expected_error, text, invalid in cases:
        _assert_wrapper_rejects(expected_error, text, lambda kw=invalid: call(**kw))

    # A cache miss cannot validate actual_max_splits=None.
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
    """Resolve actual_max_splits eagerly and under capture."""
    _require_cuda()
    from aiter.ops.flydsl.mla_reduce_kernels import (
        _ACTUAL_MAX_SPLITS_CACHE,
        _resolve_actual_max_splits,
    )

    # Widths {5, 8, 3} yield 8.
    indptr = torch.tensor([0, 5, 13, 16], dtype=torch.int32, device="cuda")
    _ACTUAL_MAX_SPLITS_CACHE.clear()

    eager = _resolve_actual_max_splits(indptr)
    assert eager == derive_actual_max_splits(indptr) == 8

    # Cached values avoid synchronization during capture.
    key = (indptr.data_ptr(), int(indptr.numel()))
    assert key in _ACTUAL_MAX_SPLITS_CACHE

    # An unseen buffer safely returns None during capture.
    other = torch.tensor([0, 4, 4], dtype=torch.int32, device="cuda")
    okey = (other.data_ptr(), int(other.numel()))
    _ACTUAL_MAX_SPLITS_CACHE.pop(okey, None)
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=side):
        miss = _resolve_actual_max_splits(other)
    assert miss is None


# Adaptive launch scenarios.
_ADAPTIVE_SCENARIOS = [
    ("b8_s32", 8, 32),
    ("b8_s13", 8, 13),
    ("b8_s6", 8, 6),
    ("b8_s2", 8, 2),
    ("b1_s32", 1, 32),
]


def test_adaptive_launch_wrapper_vs_hip(label, active, splits, replay=False):
    """Compare adaptive launch against HIP, eagerly or under replay."""
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
    """Ensure a single output row uses the persistent launch."""
    _require_cuda()
    from aiter.ops.flydsl import mla_reduce_kernels

    dt = "bf16"
    x = build_serving_decode_inputs(1, 32, _out_dtype(dt))
    x.fout.zero_()
    x.flse.zero_()
    with mock.patch.object(
        mla_reduce_kernels,
        "compile_mla_reduce",
        wraps=mla_reduce_kernels.compile_mla_reduce,
    ) as compile_kernel:
        _run_wrapper(x, num_kv_splits=32)
    torch.cuda.synchronize()
    assert compile_kernel.call_args.kwargs["adaptive"] is False
    assert compile_kernel.call_args.kwargs["persistent"] is True
    ref_out, ref_lse = hip_ref_like_fout(x)
    _assert_close(x.fout, x.flse, ref_out, ref_lse, dt)


def test_explicit_waves_per_eu_compile_hints():
    """Ensure WPE variants receive distinct compile hints."""
    _require_cuda()

    common = {"H": 16, "Dv": 512, "out_dtype": "bf16", "output_lse": True}
    entries = (
        lambda wpe: (
            compile_mla_reduce(
                **common,
                tier=Tier.ALL,
                use_reduce_final_map=True,
                persistent=False,
                adaptive=True,
                waves_per_eu=wpe,
            ),
        ),
        lambda wpe: compile_mla_reduce_splitk(**common, K=16, waves_per_eu=wpe),
    )
    compile_mla_reduce.cache_clear()
    compile_mla_reduce_splitk.cache_clear()
    try:
        for entry in entries:
            variants = {wpe: entry(wpe) for wpe in (1, 4)}
            for wpe, launchers in variants.items():
                for launcher in launchers:
                    assert launcher.compile_hints == {"waves_per_eu": wpe}
            for wpe1, wpe4 in zip(variants[1], variants[4]):
                assert wpe1 is not wpe4
    finally:
        compile_mla_reduce.cache_clear()
        compile_mla_reduce_splitk.cache_clear()


def test_explicit_waves_per_eu_equivalence():
    """Validate WPE variants under graph replay."""
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
    """Ensure pointer launches emit no tensor-resolution warnings."""
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
    """Return the registry label for ``fn``."""
    return fn.__name__.removeprefix("test_")


def run_checks():
    """Run registered checks and return failures."""
    checks = []

    def add(fn, *args, name=None):
        checks.append((name or _check_name(fn), fn, args))

    def add_each(cases, fn, suffix):
        """Register one entry per case."""
        for case in cases:
            checks.append((f"{_check_name(fn)}[{suffix.format(case)}]", fn, (case,)))

    add_each(_HIP_CASES, test_irregular_vs_hip, "{0[0]}_{0[4]}")
    add_each(_TORCH_CASES, test_irregular_vs_torch_ref, "{0[0]}_{0[4]}")
    add_each(_SMOKE_CASES, test_uniform_smoke, "H{0[0][0]}_Dv{0[0][1]}_{0[1]}_s{0[2]}")
    add_each(_GRAPH_CASES, test_irregular_cudagraph_replay, "{0[0]}")
    add(test_serving_sparse_grid, name="serving_sparse_grid_vs_hip")
    add(test_serving_sparse_grid, True, name="serving_sparse_grid_cudagraph_replay")
    splitk_ref = test_splitk_b1_s128_vs_torch_ref
    add_each([4, 8, 16], splitk_ref, "K={0}")
    add(test_splitk_b1_s128_vs_hip)
    add(splitk_ref, None, True, name="splitk_b1_s128_cudagraph_replay")
    add(test_splitk_planner_boundaries)
    add(test_da_splitk_wrapper_vs_hip)
    stage1 = test_da_splitk_preserves_stage1_rows
    for s in (0, 1):
        add(stage1, s, name=f"da_splitk_preserves_stage1_rows[splits={s}]")
        add(stage1, s, True, name=f"da_splitk_stage1_rows_cudagraph[splits={s}]")
    add(test_da_splitk_capture_safe_varying_splits)
    add(test_derive_actual_max_splits)
    add(test_actual_max_splits_wrapper_loose_budget_correct)
    add(test_actual_max_splits_wrapper_cudagraph_replay)
    add(test_serving_stale_indptr_cudagraph_replay)
    add(test_serving_guard_diffs)
    add(test_serving_guard_diffs, True, name="serving_guard_diffs_cudagraph_replay")
    add(test_serving_true_oob_no_fault)
    add_each(_DEGEN_TILES, test_degenerate_empty_tile, "tiles{0}")
    add(test_dispatch_scope_and_decode_fwd_layout)
    add(test_mla_reduce_wrapper_rejects_invalid_pointer_abi)
    add(test_resolve_actual_max_splits_eager_and_capture)
    adaptive = test_adaptive_launch_wrapper_vs_hip
    for label, active, splits in _ADAPTIVE_SCENARIOS:
        add(adaptive, label, active, splits, name=f"adaptive_launch_vs_hip[{label}]")
    add(adaptive, "b8_s32", 8, 32, True, name="adaptive_launch_cudagraph_replay")
    add(test_adaptive_launch_single_tile_uses_persistent)
    add(test_explicit_waves_per_eu_compile_hints)
    add(test_explicit_waves_per_eu_equivalence)
    add(test_pointer_launch_abi_has_no_tensor_annotation_warnings)

    failures = []
    for name, fn, args in checks:
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
    return failures


# Performance sweeps.

# (active_tiles, splits); b1_s128 exercises split-K.
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
    """Return reduce FLOPs and byte traffic."""
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
    """Build wrapper and HIP benchmark candidates."""
    candidates = {"wrapper": lambda: _run_wrapper(x, M, num_kv_splits=kv_splits)}
    if Dv == 512:
        # HIP only supports Dv=512.
        candidates["hip"] = lambda: aiter.mla_reduce_v1(*x.maps, M, 0, x.fout, x.flse)
    return candidates


def _bench_reduce_candidates(
    candidates, x, ref_out, dtype, flops, nbytes, op_name, ref_lse=None
):
    """Time and validate benchmark candidates."""
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        x.fout.zero_()
        x.flse.zero_()
        # ROCTracer can crash after repeated HIP graph captures in CI.
        _, us = run_perftest(fn, num_warmup=25, num_iters=100, use_cuda_event=True)
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
        # Use graph replay throughput for roofline metrics.
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
    # Tail tiles are empty; reference active metadata only.
    active_prefix = x._replace(indptr=x.indptr[: active + 1], fmap=x.fmap[:active])
    ref_out, ref_lse = torch_ref_gather(active_prefix, H, Dv, dtype)
    candidates = _reduce_candidates(x, Dv, 1, splits)
    flops, nbytes = _reduce_roofline(active * splits, active, H, Dv, dtype)
    return _bench_reduce_candidates(
        candidates, x, ref_out, dtype, flops, nbytes, "mla_reduce", ref_lse
    )


@benchmark()
def test_mla_reduce_uniform(tiles, splits, H, Dv, M, dtype):
    """Benchmark dense occupancy cases."""
    x = build_inputs(tiles, splits, H, Dv, dtype, M=M)
    ref_out, _ref_lse = torch_ref(x.po, x.pl, tiles, splits, H, Dv, dtype, M=M)
    candidates = _reduce_candidates(x, Dv, M, splits)
    flops, nbytes = _reduce_roofline(tiles * splits, tiles * M, H, Dv, dtype)
    return _bench_reduce_candidates(
        candidates, x, ref_out, dtype, flops, nbytes, "mla_reduce_uniform"
    )


@benchmark()
def test_mla_reduce_irregular(splits_per_tile, gap_stride, pool_slack, H, Dv, dtype):
    """Benchmark irregular split layouts."""
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
    """Log one benchmark table."""
    df = pd.DataFrame(rows)
    aiter.logger.info("%s (markdown):\n%s", title, df.to_markdown(index=False))


def run_bench(args):
    """Run all benchmark sweeps."""
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
