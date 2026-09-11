# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for the FlyDSL MoE auxiliary (non-GEMM) kernels.

These are the index/scatter/reduce kernels the grouped-MoE path launches around
its GEMMs: the EP global->local LUT, the tile-aligned prefix sum, the route ->
grouped-row map, the route-gather row copy, and the weighted gather-reduce
epilogue.

They are dispatched only on gfx1250 by ``grouped_moe_gfx1250.py``, but the
kernels themselves contain no gfx1250-only instruction, so they build and run on
every supported card -- which is what lets this test cover them here.

Only gather_reduce carries a TFLOPS column. The rest are index / scatter /
prefix-sum kernels with no float arithmetic at all, so their FLOP count is zero
and a TFLOPS column would be seven columns of 0.0 rather than a roofline. TB/s
is the meaningful metric for them, and it is reported everywhere.

Each op is run both eagerly and replayed from a CUDA graph.  The graph candidate
is not just a perf number: these kernels sit on the decode hot path where the
whole MoE layer is captured, so "is it capture-safe" (no host sync, no
capture-time allocation, stable pointers) is a correctness property.
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.grouped_moe_gfx1250 import _grouped_a8w4_preshuffle_e8m0_scale
from aiter.ops.flydsl.kernels.moe_contiguous_psum import (
    build_moe_contiguous_psum_module,
    build_moe_contiguous_psum_remap_ep_module,
    build_moe_contiguous_psum_remap_module,
    build_moe_route_psum_fused_module,
)
from aiter.ops.flydsl.kernels.moe_g2l_lut import build_moe_g2l_lut_module
from aiter.ops.flydsl.kernels.moe_gather_reduce import build_moe_gather_reduce_module
from aiter.ops.flydsl.kernels.moe_route_maps import (
    DROPPED_ROUTE_ROW,
    build_moe_route_g2l_fused_module,
    build_moe_route_g2l_lds_module,
    build_moe_route_maps_module,
    build_moe_topids_to_rows_g2l_module,
    build_moe_topids_to_rows_module,
)
from aiter.ops.flydsl.kernels.moe_scatter_copy_preshuffle_scale import (
    build_moe_scatter_copy_preshuffle_scale_module,
)
from aiter.ops.flydsl.kernels.moe_scatter_copy_token import (
    build_moe_scatter_copy_token_module,
)
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950", "gfx1250"]
I32 = torch.int32


def _graph_runner(fn):
    """Capture ``fn`` into a CUDA graph and return a replay callable.

    Warm up on a side stream first: the flydsl launcher does its JIT dispatch and
    module load on the first call, which is illegal during capture.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


# --------------------------------------------------------------- references
def run_torch_g2l_lut(mask, E, nvt, topk):
    incl = torch.cumsum(mask.to(torch.int64), 0).to(I32)
    lut = torch.where(mask != 0, incl - 1, torch.full_like(incl, E))
    return lut, torch.zeros(E, dtype=I32, device=mask.device), nvt * topk


def run_torch_psum(masked_m, tile_m):
    m = masked_m.to(torch.int64)
    aligned = ((m + tile_m - 1) // tile_m) * tile_m
    starts = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=m.device),
            torch.cumsum(aligned, 0)[:-1],
        ]
    )
    total = int(aligned.sum())
    return starts.to(I32), (starts + m).to(I32), max(total, tile_m)


def run_torch_scatter_copy(src, dst, dst_src):
    out = dst.clone()
    keep = dst_src >= 0
    out[keep] = src[dst_src[keep].long()]
    return out


def run_torch_gather_reduce(grouped, rmap, w, dtype):
    g, r, wt = grouped.float(), rmap, w.float()
    out = torch.zeros(rmap.shape[0], grouped.shape[1], device=g.device)
    for k in range(rmap.shape[1]):
        row = r[:, k]
        keep = row >= 0
        out[keep] += wt[keep, k, None] * g[row[keep].long()]
    return out.to(dtype)


# ------------------------------------------------------------------- tests
@benchmark()
def test_g2l_lut(n, E, topk):
    """EP global->local expert LUT build (single-block Hillis-Steele scan)."""
    launch = build_moe_g2l_lut_module()
    nvt = max(1, n // 4)
    mask = (torch.rand(n) < 0.6).to(I32)
    nvt_t = torch.tensor([nvt], dtype=I32)
    ref_lut, ref_cnt, ref_nvr = run_torch_g2l_lut(mask, E, nvt, topk)

    lut = torch.full((n,), -99, dtype=I32)
    cnt = torch.full((E,), -99, dtype=I32)
    nvr = torch.full((1,), -99, dtype=I32)

    def fn():
        launch(
            ptr_arg(mask),
            ptr_arg(lut),
            ptr_arg(cnt),
            ptr_arg(nvt_t),
            ptr_arg(nvr),
            n,
            E,
            topk,
            stream=torch.cuda.current_stream().cuda_stream,
        )

    # mask in + lut out + counter out + 2 scalars
    nbytes = (n * 2 + E) * 4
    ret = {"gfx": get_gfx()}
    # eager vs graph replay are the two candidates: on the decode path these
    # kernels run inside a captured MoE layer, so replay is the real call.
    candidates = {"eager": fn, "cudagraph": _graph_runner(fn)}
    for name, run in candidates.items():
        lut.fill_(-99)
        cnt.fill_(-99)
        nvr.fill_(-99)
        _, us = run_perftest(run)
        err = checkAllclose(
            ref_lut.float(), lut.float(), rtol=0, atol=0, msg=f"{name}: g2l lut"
        )
        checkAllclose(
            ref_cnt.float(), cnt.float(), rtol=0, atol=0, msg=f"{name}: g2l counter"
        )
        checkAllclose(
            torch.tensor([float(ref_nvr)]),
            nvr.float(),
            rtol=0,
            atol=0,
            msg=f"{name}: g2l nvr",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_contiguous_psum(E, tile_m):
    """Tile-aligned exclusive prefix sum over per-expert counts."""
    launch = build_moe_contiguous_psum_module()
    masked = torch.randint(0, 400, (E,), dtype=I32)
    ref_starts, ref_psum, ref_cm = run_torch_psum(masked, tile_m)

    starts = torch.full((E,), -99, dtype=I32)
    psum = torch.full((E,), -99, dtype=I32)
    cm = torch.full((1,), -99, dtype=I32)

    def fn():
        launch(
            ptr_arg(masked),
            ptr_arg(starts),
            ptr_arg(psum),
            ptr_arg(cm),
            E,
            tile_m,
            stream=torch.cuda.current_stream().cuda_stream,
        )

    nbytes = E * 3 * 4
    ret = {"gfx": get_gfx()}
    # eager vs graph replay are the two candidates: on the decode path these
    # kernels run inside a captured MoE layer, so replay is the real call.
    candidates = {"eager": fn, "cudagraph": _graph_runner(fn)}
    for name, run in candidates.items():
        starts.fill_(-99)
        psum.fill_(-99)
        cm.fill_(-99)
        _, us = run_perftest(run)
        err = checkAllclose(
            ref_starts.float(),
            starts.float(),
            rtol=0,
            atol=0,
            msg=f"{name}: psum starts",
        )
        checkAllclose(
            ref_psum.float(), psum.float(), rtol=0, atol=0, msg=f"{name}: psum psum"
        )
        checkAllclose(
            torch.tensor([float(ref_cm)]),
            cm.float(),
            rtol=0,
            atol=0,
            msg=f"{name}: contiguous_m",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_psum_remap(numel, E, tile_m):
    """Prefix sum fused with the in-place masked->contiguous row remap.

    This is the O(numel) sibling of test_contiguous_psum: the remap loop walks
    every route, so it is where a per-access addressing change actually shows up.
    """
    launch = build_moe_contiguous_psum_remap_module()
    route_max_m = max(512, 4 * numel // E)
    experts_of = torch.randint(0, E, (numel,), dtype=I32)
    counts = torch.bincount(experts_of.long(), minlength=E).to(I32)
    assert int(counts.max()) < route_max_m, "route_max_m too small for this shape"
    # grouped row = e*route_max_m + slot, where slot is the route's rank within
    # its expert. Built on CPU with a stable argsort (the per-expert rank is
    # position-within-group), then moved -- a Python loop here is O(numel) host work.
    order = experts_of.cpu().long()
    perm = torch.argsort(order, stable=True)
    sorted_e = order[perm]
    # explicit device="cpu": the module sets cuda as the default device.
    group_base = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device="cpu"),
            torch.cumsum(torch.bincount(sorted_e, minlength=E), 0)[:-1],
        ]
    )
    slot_sorted = (
        torch.arange(numel, dtype=torch.int64, device="cpu") - group_base[sorted_e]
    )
    slot = torch.empty(numel, dtype=torch.int64, device="cpu")
    slot[perm] = slot_sorted
    rows0 = (order * route_max_m + slot).to(I32).cuda()
    rows0[::101] = -1  # DROPPED_ROUTE_ROW must be left untouched
    masked = counts.clone()
    nvr = torch.tensor([numel], dtype=I32)

    ref_starts = run_torch_psum(masked, tile_m)[0]
    keep = rows0.cpu() >= 0
    ref_rows = rows0.cpu().clone()
    r = ref_rows[keep].long()
    ref_rows[keep] = (ref_starts.cpu()[r // route_max_m] + (r % route_max_m)).to(I32)

    rows = rows0.clone()
    starts = torch.full((E,), -99, dtype=I32)
    psum = torch.full((E,), -99, dtype=I32)
    cm = torch.full((1,), -99, dtype=I32)

    def fn():
        launch(
            ptr_arg(masked),
            ptr_arg(rows),
            ptr_arg(starts),
            ptr_arg(psum),
            ptr_arg(cm),
            numel,
            E,
            route_max_m,
            tile_m,
            ptr_arg(nvr),
            stream=torch.cuda.current_stream().cuda_stream,
        )

    nbytes = (numel * 2 + E * 3) * 4
    ret = {"gfx": get_gfx()}
    # eager vs graph replay are the two candidates: on the decode path these
    # kernels run inside a captured MoE layer, so replay is the real call.
    candidates = {"eager": fn, "cudagraph": _graph_runner(fn)}
    for name, run in candidates.items():
        rows.copy_(rows0)
        starts.fill_(-99)
        psum.fill_(-99)
        cm.fill_(-99)
        run()
        torch.cuda.synchronize()
        err = checkAllclose(
            ref_rows.float().cuda(),
            rows.float(),
            rtol=0,
            atol=0,
            msg=f"{name}: psum_remap rows",
        )
        checkAllclose(
            ref_starts.float(),
            starts.float(),
            rtol=0,
            atol=0,
            msg=f"{name}: psum_remap starts",
        )
        # remap is idempotent-unsafe (in place), so time on a throwaway copy
        _, us = run_perftest(run)
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_psum_remap_ep(numel, E, tile_m, topk):
    """psum + remap fused with the gemm2 EP ep_rowmap scatter.

    Reachable from fused_moe only when stage2_scatter is set (the EP
    scatter-combine / MegaMoE path), so it is the rarest of the reachable
    kernels -- and the only one that writes the packed (dest, weight-bits) pair.
    """
    launch = build_moe_contiguous_psum_remap_ep_module()
    route_max_m = max(512, 4 * numel // E)
    max_tok, slot_stride = 1024, 8192
    tokens = (numel + topk - 1) // topk

    experts_of = torch.randint(0, E, (numel,), dtype=I32)
    counts = torch.bincount(experts_of.long(), minlength=E).to(I32)
    assert int(counts.max()) < route_max_m
    order = experts_of.cpu().long()
    perm = torch.argsort(order, stable=True)
    sorted_e = order[perm]
    base = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device="cpu"),
            torch.cumsum(torch.bincount(sorted_e, minlength=E), 0)[:-1],
        ]
    )
    slot = torch.empty(numel, dtype=torch.int64, device="cpu")
    slot[perm] = torch.arange(numel, dtype=torch.int64, device="cpu") - base[sorted_e]
    rows0 = (order * route_max_m + slot).to(I32).cuda()
    rows0[::101] = -1  # dropped routes are skipped entirely

    masked = counts.clone()
    nvr = torch.tensor([numel], dtype=I32)
    gather_w = (torch.rand(numel) * 2 - 1).to(torch.bfloat16)
    tis = torch.randint(0, max_tok * 4, (tokens,), dtype=I32)
    cap_rows = E * route_max_m
    ep0 = torch.full((cap_rows + 1, 2), -1, dtype=I32)
    ep0[:, 1] = 0

    # reference
    ref_starts = run_torch_psum(masked, tile_m)[0]
    ref_rows = rows0.cpu().clone()
    ref_ep = ep0.cpu().clone()
    r0 = rows0.cpu()
    st = ref_starts.cpu()
    tisc, gwc = tis.cpu(), gather_w.cpu().float()
    for i in range(numel):
        if r0[i] < 0:
            continue
        e, sl = int(r0[i]) // route_max_m, int(r0[i]) % route_max_m
        final = int(st[e]) + sl
        ref_rows[i] = final
        t, k = i // topk, i % topk
        enc = int(tisc[t])
        pe, lid = enc // max_tok, enc % max_tok
        ref_ep[final, 0] = pe * slot_stride + lid * topk + k
        ref_ep[final, 1] = gwc[i].view(torch.int32)

    rows = rows0.clone()
    starts = torch.full((E,), -99, dtype=I32)
    psum = torch.full((E,), -99, dtype=I32)
    cm = torch.full((1,), -99, dtype=I32)
    ep = ep0.clone()

    def fn():
        launch(
            ptr_arg(masked),
            ptr_arg(rows),
            ptr_arg(starts),
            ptr_arg(psum),
            ptr_arg(cm),
            E,
            route_max_m,
            tile_m,
            ptr_arg(nvr),
            ptr_arg(gather_w),
            ptr_arg(tis),
            ptr_arg(ep),
            topk,
            max_tok,
            slot_stride,
            stream=torch.cuda.current_stream().cuda_stream,
        )

    nbytes = (numel * 3 + cap_rows * 2) * 4
    ret = {"gfx": get_gfx()}
    # eager vs graph replay are the two candidates: on the decode path these
    # kernels run inside a captured MoE layer, so replay is the real call.
    candidates = {"eager": fn, "cudagraph": _graph_runner(fn)}
    for name, run in candidates.items():
        rows.copy_(rows0)
        ep.copy_(ep0)
        starts.fill_(-99)
        run()
        torch.cuda.synchronize()
        err = checkAllclose(
            ref_rows.float().cuda(),
            rows.float(),
            rtol=0,
            atol=0,
            msg=f"{name}: psum_remap_ep rows",
        )
        checkAllclose(
            ref_ep.float().cuda(),
            ep.float(),
            rtol=0,
            atol=0,
            msg=f"{name}: psum_remap_ep ep_rowmap",
        )
        checkAllclose(
            ref_starts.float().cuda(),
            starts.float(),
            rtol=0,
            atol=0,
            msg=f"{name}: psum_remap_ep starts",
        )
        rows.copy_(rows0)
        ep.copy_(ep0)
        _, us = run_perftest(run)
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_route_g2l(numel, E_global, n_buckets, variant, w_dtype):
    """EP route: global->local remap + atomic row claim + fused weight cast.

    Both variants reachable from fused_moe share this ABI -- "lds" is the default
    (two-level LDS->global atomic reduction), "plain" the fallback taken when the
    LDS path is disabled or E exceeds MAX_ROUTE_BUCKETS.

    Row assignment is an atomic argsort, so rows are checked on invariants. The
    weight cast is NOT: kept -> cast(f32 weight), dropped -> 0 is exact, and it is
    the one output that would silently corrupt if a pointer carried the wrong
    element type.
    """
    build = (
        build_moe_route_g2l_lds_module
        if variant == "lds"
        else build_moe_topids_to_rows_g2l_module
    )
    wdt = torch.bfloat16 if w_dtype == "bf16" else torch.float16
    launch = build(w_dtype)
    max_m = max(512, 4 * numel // max(1, n_buckets))

    # g2l_lut: the first n_buckets global experts are local, the rest dropped.
    g2l = torch.full((E_global,), n_buckets, dtype=I32)
    local = torch.randperm(E_global, device="cpu")[:n_buckets]
    g2l[local.to(g2l.device)] = torch.arange(n_buckets, dtype=I32)
    topk_ids = torch.randint(0, E_global, (numel,), dtype=I32)
    weight_in = torch.rand(numel, dtype=torch.float32)
    nvr = torch.tensor([numel], dtype=I32)

    le = g2l.cpu()[topk_ids.cpu().long()]
    is_drop = le == n_buckets
    ref_w = torch.where(is_drop, torch.zeros_like(weight_in.cpu()), weight_in.cpu())
    ref_w = ref_w.to(wdt).cuda()
    ref_cnt = torch.bincount(le[~is_drop].long(), minlength=n_buckets).to(I32).cuda()

    atomic = torch.zeros(n_buckets, dtype=I32)
    rows = torch.full((numel,), -99, dtype=I32)
    gw = torch.full((numel,), 7.0, dtype=wdt)
    blocks = (numel + 255) // 256

    def fn():
        launch(
            ptr_arg(topk_ids),
            ptr_arg(g2l),
            ptr_arg(atomic),
            ptr_arg(rows),
            ptr_arg(weight_in),
            ptr_arg(gw),
            ptr_arg(nvr),
            numel,
            max_m,
            n_buckets,
            blocks,
            stream=torch.cuda.current_stream().cuda_stream,
        )

    nbytes = numel * (4 + 4 + 4 + gw.element_size())
    ret = {"gfx": get_gfx()}
    # eager vs graph replay are the two candidates: on the decode path these
    # kernels run inside a captured MoE layer, so replay is the real call.
    candidates = {"eager": fn, "cudagraph": _graph_runner(fn)}
    for name, run in candidates.items():
        atomic.zero_()
        rows.fill_(-99)
        gw.fill_(7.0)
        _, us = run_perftest(run)
        atomic.zero_()
        rows.fill_(-99)
        gw.fill_(7.0)
        run()
        torch.cuda.synchronize()
        err = checkAllclose(
            ref_w.float(), gw.float(), rtol=0, atol=0, msg=f"{name}: g2l gather_w cast"
        )
        checkAllclose(
            ref_cnt.float(),
            atomic.float(),
            rtol=0,
            atol=0,
            msg=f"{name}: g2l bucket counts",
        )
        r = rows.cpu()
        assert bool(
            (r[is_drop] == DROPPED_ROUTE_ROW).all()
        ), f"{name}: dropped route did not get the sentinel"
        kept = r[~is_drop].long()
        assert bool(
            (kept // max_m == le[~is_drop].long()).all()
        ), f"{name}: kept route outside its local bucket band"
        assert len(set(kept.tolist())) == int((~is_drop).sum()), f"{name}: rows collide"
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_scatter_copy_token(num_dst, row_bytes):
    """Route-gather row copy. row_bytes picks the dwordx4/x2/dword/byte ladder."""
    launch = build_moe_scatter_copy_token_module(row_bytes)
    n_src = max(1, num_dst // 2)
    src = torch.randint(0, 256, (n_src, row_bytes), dtype=torch.uint8)
    dst = torch.full((num_dst, row_bytes), 0xEE, dtype=torch.uint8)
    dst_src = torch.randint(-1, n_src, (num_dst,), dtype=I32)
    ref = run_torch_scatter_copy(src, dst, dst_src)

    def fn():
        launch(
            ptr_arg(src),
            ptr_arg(dst),
            ptr_arg(dst_src),
            num_dst,
            torch.cuda.current_stream().cuda_stream,
        )

    # every kept dst row is read once from src and written once
    kept = int((dst_src >= 0).sum())
    nbytes = kept * row_bytes * 2
    ret = {"gfx": get_gfx()}
    # eager vs graph replay are the two candidates: on the decode path these
    # kernels run inside a captured MoE layer, so replay is the real call.
    candidates = {"eager": fn, "cudagraph": _graph_runner(fn)}
    for name, run in candidates.items():
        dst.fill_(0xEE)
        _, us = run_perftest(run)
        err = checkAllclose(
            ref.float(), dst.float(), rtol=0, atol=0, msg=f"{name}: scatter_copy"
        )
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_gather_reduce(tokens, model_dim, topk, vec_dwords, dtype):
    """Weighted gather-reduce epilogue: out[t] = sum_k w(t,k) * grouped[row(t,k)]."""
    out_dtype = "bf16" if dtype == torch.bfloat16 else "f16"
    launch = build_moe_gather_reduce_module(
        model_dim, topk, out_dtype, 1, vec_dwords, "f32"
    )
    rows_total = max(64, tokens * topk)
    grouped = (torch.randn(rows_total, model_dim) * 0.5).to(dtype)
    rmap = torch.randint(0, rows_total, (tokens, topk), dtype=I32)
    rmap[0, 0] = -1  # DROPPED_ROUTE_ROW must contribute nothing
    w = torch.rand(tokens, topk, dtype=torch.float32)
    nvt = torch.tensor([tokens], dtype=I32)
    ref = run_torch_gather_reduce(grouped, rmap, w, dtype)
    out = torch.full((tokens, model_dim), 7.0, dtype=dtype)

    def fn():
        launch(
            ptr_arg(grouped),
            ptr_arg(rmap),
            ptr_arg(w),
            ptr_arg(out),
            tokens,
            rows_total * (model_dim // 2),
            ptr_arg(nvt),
            stream=torch.cuda.current_stream().cuda_stream,
        )

    esz = grouped.element_size()
    # topk source rows read + one row written per token, plus the (t,k) maps
    nbytes = tokens * (topk * model_dim + model_dim) * esz + tokens * topk * 8
    flops = tokens * topk * model_dim * 2  # one multiply-add per contribution
    ret = {"gfx": get_gfx()}
    # eager vs graph replay are the two candidates: on the decode path these
    # kernels run inside a captured MoE layer, so replay is the real call.
    candidates = {"eager": fn, "cudagraph": _graph_runner(fn)}
    for name, run in candidates.items():
        out.fill_(7.0)
        _, us = run_perftest(run)
        err = checkAllclose(
            ref.float(), out.float(), rtol=3e-2, atol=3e-2, msg=f"{name}: gather_reduce"
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_topids_to_rows(numel, E, max_m):
    """Atomic-scatter argsort. Intra-bucket order is unspecified by design, so the
    check is on the invariants (counts, own-expert band, bijection), not values."""
    launch = build_moe_topids_to_rows_module()
    topk_ids = torch.randint(0, E, (numel,), dtype=I32)
    counts = torch.bincount(topk_ids.long(), minlength=E).to(I32)
    # max_m is the per-expert grouped capacity: row = e*max_m + slot. If a bucket
    # can hold more routes than max_m its rows spill into the next expert's band,
    # which is a caller contract violation, not a kernel bug -- so the sweep must
    # not generate it.
    assert max_m > int(counts.max()), (
        f"max_m={max_m} too small for numel={numel}, E={E} "
        f"(largest bucket {int(counts.max())})"
    )
    atomic = torch.zeros(E, dtype=I32)
    rows = torch.full((numel,), -99, dtype=I32)
    blocks = (numel + 255) // 256

    def fn():
        launch(
            ptr_arg(topk_ids),
            ptr_arg(atomic),
            ptr_arg(rows),
            numel,
            max_m,
            blocks,
            stream=torch.cuda.current_stream().cuda_stream,
        )

    nbytes = numel * 2 * 4
    ret = {"gfx": get_gfx()}
    # eager vs graph replay are the two candidates: on the decode path these
    # kernels run inside a captured MoE layer, so replay is the real call.
    candidates = {"eager": fn, "cudagraph": _graph_runner(fn)}
    for name, run in candidates.items():
        # Time first: the counter keeps accumulating across iterations, which only
        # inflates the row *values* (writes stay indexed by route, so in-bounds)
        # and leaves the atomic contention pattern -- the thing being measured --
        # unchanged. Then reset and take one clean pass for the correctness check.
        atomic.zero_()
        rows.fill_(-99)
        _, us = run_perftest(run)
        atomic.zero_()
        rows.fill_(-99)
        run()
        torch.cuda.synchronize()
        err = checkAllclose(
            counts.float(), atomic.float(), rtol=0, atol=0, msg=f"{name}: route counts"
        )
        r = rows.cpu()
        assert bool(
            (r.long() // max_m == topk_ids.cpu().long()).all()
        ), f"{name}: row outside own expert band"
        assert bool(
            (r.long() % max_m < counts.cpu()[topk_ids.cpu().long()]).all()
        ), f"{name}: row past bucket count"
        assert len(set(r.tolist())) == numel, f"{name}: rows not a bijection"
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_route_maps(numel, E, topk, max_m):
    """Atomic-scatter argsort plus the inverse ``rows_to_tokens`` map.

    Same row invariants as test_topids_to_rows (intra-bucket order is
    unspecified), with the extra check that the inverse map inverts: every row
    the kernel claimed for route r must name r's token.
    """
    launch = build_moe_route_maps_module()
    topk_ids = torch.randint(0, E, (numel,), dtype=I32)
    counts = torch.bincount(topk_ids.long(), minlength=E).to(I32)
    assert max_m > int(counts.max()), (
        f"max_m={max_m} too small for numel={numel}, E={E} "
        f"(largest bucket {int(counts.max())})"
    )
    atomic = torch.zeros(E, dtype=I32)
    rows = torch.full((numel,), -99, dtype=I32)
    r2t = torch.full((E * max_m,), -99, dtype=I32)
    blocks = (numel + 255) // 256

    def fn():
        # The counter must be re-zeroed per call, not just per timing pass: this
        # kernel writes rows_to_tokens *indexed by the claimed row*, so a counter
        # carried over from a previous iteration would push that store past
        # E*max_m and scribble on whatever follows. Production zeroes it the same
        # way, so the memset belongs inside the timed region.
        atomic.zero_()
        launch(
            ptr_arg(topk_ids),
            ptr_arg(atomic),
            ptr_arg(rows),
            ptr_arg(r2t),
            numel,
            topk,
            max_m,
            blocks,
            stream=torch.cuda.current_stream().cuda_stream,
        )

    # topk_ids in, rows out, one rows_to_tokens slot out per route
    nbytes = numel * 3 * 4
    ret = {"gfx": get_gfx()}
    # eager vs graph replay are the two candidates: on the decode path these
    # kernels run inside a captured MoE layer, so replay is the real call.
    candidates = {"eager": fn, "cudagraph": _graph_runner(fn)}
    for name, run in candidates.items():
        rows.fill_(-99)
        _, us = run_perftest(run)
        rows.fill_(-99)
        r2t.fill_(-99)
        run()
        torch.cuda.synchronize()
        err = checkAllclose(
            counts.float(), atomic.float(), rtol=0, atol=0, msg=f"{name}: route counts"
        )
        r = rows.cpu()
        ids = topk_ids.cpu().long()
        assert bool(
            (r.long() // max_m == ids).all()
        ), f"{name}: row outside own expert band"
        assert len(set(r.tolist())) == numel, f"{name}: rows not a bijection"
        inv = r2t.cpu()[r.long()]
        tokens = torch.arange(numel, device="cpu") // topk
        assert bool(
            (inv == tokens).all()
        ), f"{name}: rows_to_tokens does not invert topids_to_rows"
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_route_psum_fused(numel, E, tile_m):
    """Single-workgroup route + LDS atomic + psum + remap, all in one launch.

    Collapses what test_topids_to_rows and test_contiguous_psum cover separately,
    so the checks are the union: exact masked_m/starts/psum against the torch
    prefix sum, and the row invariants against the *contiguous* bands it emits.
    """
    launch = build_moe_route_psum_fused_module()
    max_m = max(512, 4 * numel // E)
    topk_ids = torch.randint(0, E, (numel,), dtype=I32)
    counts = torch.bincount(topk_ids.long(), minlength=E).to(I32)
    assert max_m > int(counts.max()), "max_m too small for this shape"
    ref_starts, ref_psum, _ = run_torch_psum(counts, tile_m)

    rows = torch.full((numel,), -99, dtype=I32)
    masked = torch.full((E,), -99, dtype=I32)
    starts = torch.full((E,), -99, dtype=I32)
    psum = torch.full((E,), -99, dtype=I32)

    def fn():
        launch(
            ptr_arg(topk_ids),
            ptr_arg(rows),
            ptr_arg(masked),
            ptr_arg(starts),
            ptr_arg(psum),
            numel,
            E,
            max_m,
            tile_m,
            stream=torch.cuda.current_stream().cuda_stream,
        )

    nbytes = numel * 2 * 4 + E * 3 * 4
    ret = {"gfx": get_gfx()}
    # eager vs graph replay are the two candidates: on the decode path these
    # kernels run inside a captured MoE layer, so replay is the real call.
    candidates = {"eager": fn, "cudagraph": _graph_runner(fn)}
    for name, run in candidates.items():
        # The LDS counter is zeroed by the kernel itself, so unlike the split
        # route kernels this one is idempotent and needs no reset between runs.
        rows.fill_(-99)
        _, us = run_perftest(run)
        err = checkAllclose(
            counts.float(), masked.float(), rtol=0, atol=0, msg=f"{name}: masked_m"
        )
        checkAllclose(
            ref_starts.float(), starts.float(), rtol=0, atol=0, msg=f"{name}: starts"
        )
        checkAllclose(
            ref_psum.float(), psum.float(), rtol=0, atol=0, msg=f"{name}: psum"
        )
        r = rows.cpu().long()
        ids = topk_ids.cpu().long()
        lo = ref_starts.cpu().long()[ids]
        assert bool(
            ((r >= lo) & (r < lo + counts.cpu().long()[ids])).all()
        ), f"{name}: row outside its expert's contiguous band"
        assert len(set(r.tolist())) == numel, f"{name}: rows not a bijection"
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_route_g2l_fused(numel, E_global, n_buckets, w_dtype):
    """Single-block g2l-LUT build fused into the EP route pass.

    The LUT never reaches global memory here, so it is checked indirectly: the
    bucket a route lands in must match the LUT the standalone moe_g2l_lut kernel
    would have produced from the same mask.
    """
    launch = build_moe_route_g2l_fused_module(w_dtype)
    wdt = torch.bfloat16 if w_dtype == "bf16" else torch.float16
    max_m = max(512, 4 * numel // max(1, n_buckets))

    # Exactly n_buckets enabled global experts; the LUT is their rank order.
    mask = torch.zeros(E_global, dtype=I32)
    mask[torch.randperm(E_global, device="cpu")[:n_buckets].to(mask.device)] = 1
    g2l, _, _ = run_torch_g2l_lut(mask, n_buckets, 1, 1)
    topk_ids = torch.randint(0, E_global, (numel,), dtype=I32)
    weight_in = torch.rand(numel, dtype=torch.float32)
    nvr = torch.tensor([numel], dtype=I32)

    le = g2l.cpu()[topk_ids.cpu().long()]
    is_drop = le == n_buckets
    ref_w = torch.where(is_drop, torch.zeros_like(weight_in.cpu()), weight_in.cpu())
    ref_w = ref_w.to(wdt).cuda()
    ref_cnt = torch.bincount(le[~is_drop].long(), minlength=n_buckets).to(I32).cuda()

    counter = torch.full((n_buckets,), -99, dtype=I32)
    rows = torch.full((numel,), -99, dtype=I32)
    gw = torch.full((numel,), 7.0, dtype=wdt)

    def fn():
        launch(
            ptr_arg(mask),
            ptr_arg(topk_ids),
            ptr_arg(weight_in),
            ptr_arg(counter),
            ptr_arg(rows),
            ptr_arg(gw),
            ptr_arg(nvr),
            E_global,
            numel,
            max_m,
            n_buckets,
            stream=torch.cuda.current_stream().cuda_stream,
        )

    nbytes = numel * (4 + 4 + 4 + gw.element_size()) + E_global * 4
    ret = {"gfx": get_gfx()}
    # eager vs graph replay are the two candidates: on the decode path these
    # kernels run inside a captured MoE layer, so replay is the real call.
    candidates = {"eager": fn, "cudagraph": _graph_runner(fn)}
    for name, run in candidates.items():
        # The kernel zeroes its own (E,) counter, so repeated runs are idempotent.
        rows.fill_(-99)
        gw.fill_(7.0)
        _, us = run_perftest(run)
        err = checkAllclose(
            ref_w.float(), gw.float(), rtol=0, atol=0, msg=f"{name}: fused gather_w"
        )
        checkAllclose(
            ref_cnt.float(),
            counter.float(),
            rtol=0,
            atol=0,
            msg=f"{name}: fused bucket counts",
        )
        r = rows.cpu()
        assert bool(
            (r[is_drop] == DROPPED_ROUTE_ROW).all()
        ), f"{name}: dropped route did not get the sentinel"
        kept = r[~is_drop].long()
        assert bool(
            (kept // max_m == le[~is_drop].long()).all()
        ), f"{name}: kept route outside its local bucket band"
        assert len(set(kept.tolist())) == int((~is_drop).sum()), f"{name}: rows collide"
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_preshuffle_scale(E, max_m, row_bytes, wmma_rep, scale_k_per_tile, gather):
    """e8m0 scale WMMA preshuffle, with (stage1) and without (stage2) route-gather.

    The oracle is the torch permute this kernel replaced
    (``_grouped_a8w4_preshuffle_e8m0_scale``), applied to a separately built
    row-major grouped scale -- so the reference shares no index math with the
    kernel. Padding rows (-1 in rows_to_tokens) must come out as 0.
    """
    launch = build_moe_scatter_copy_preshuffle_scale_module(
        row_bytes, wmma_rep, scale_k_per_tile, gather=gather
    )
    rows_per_tile = wmma_rep * 16
    assert max_m % rows_per_tile == 0
    tiles_per_expert = max_m // rows_per_tile
    n_grouped = E * max_m

    if gather:
        n_src = max(1, n_grouped // 2)
        # -1 is the padding sentinel: those grouped rows must be zero-filled.
        r2t = torch.randint(-1, n_src, (n_grouped,), dtype=I32)
        src = torch.randint(0, 256, (n_src, row_bytes), dtype=torch.uint8)
        raw = torch.zeros(n_grouped, row_bytes, dtype=torch.uint8)
        keep = r2t >= 0
        raw[keep] = src[r2t[keep].long()]
    else:
        r2t = None
        src = torch.randint(0, 256, (n_grouped, row_bytes), dtype=torch.uint8)
        raw = src
    ref = _grouped_a8w4_preshuffle_e8m0_scale(
        raw.view(E, max_m, row_bytes), rows_per_tile, scale_k_per_tile
    ).reshape(-1)

    dst = torch.full((n_grouped * row_bytes,), 0xEE, dtype=torch.uint8)
    args = (ptr_arg(src), ptr_arg(dst))
    if gather:
        args += (ptr_arg(r2t),)

    def fn():
        launch(
            *args,
            max_m,
            E,
            tiles_per_expert,
            stream=torch.cuda.current_stream().cuda_stream,
        )

    # source rows read once (gather skips padding), whole output written once
    n_read = int((r2t >= 0).sum()) if gather else n_grouped
    nbytes = (n_read + n_grouped) * row_bytes
    ret = {"gfx": get_gfx(), "gather": gather}
    # eager vs graph replay are the two candidates: on the decode path these
    # kernels run inside a captured MoE layer, so replay is the real call.
    candidates = {"eager": fn, "cudagraph": _graph_runner(fn)}
    for name, run in candidates.items():
        dst.fill_(0xEE)
        _, us = run_perftest(run)
        err = checkAllclose(
            ref.float(), dst.float(), rtol=0, atol=0, msg=f"{name}: preshuffle scale"
        )
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def summarize(name, rows):
    aiter.logger.info(
        "%s summary (markdown):\n%s", name, pd.DataFrame(rows).to_markdown(index=False)
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl MoE aux kernels unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument("-d", "--dtype", type=str, nargs="*", default=["bf16", "f16"])
    parser.add_argument(
        "-e", "--experts", type=int, nargs="*", default=[8, 40, 300, 512]
    )
    parser.add_argument("-t", "--tokens", type=int, nargs="*", default=[1, 24, 512])
    parser.add_argument("--model-dim", type=int, nargs="*", default=[512, 2050])
    parser.add_argument("--topk", type=int, nargs="*", default=[1, 4, 8])
    parser.add_argument("--vec", type=int, nargs="*", default=[2, 4])
    parser.add_argument("--tile-m", type=int, nargs="*", default=[32, 128])
    parser.add_argument("--row-bytes", type=int, nargs="*", default=[4096, 90, 40, 12])
    parser.add_argument("--numel", type=int, nargs="*", default=[256, 4096, 65536])
    args = parser.parse_args()
    dmap = {"bf16": torch.bfloat16, "f16": torch.float16}

    # n <= 512: the LUT scan is single-workgroup (MAX_G2L_EXPERTS).
    summarize(
        "moe_g2l_lut",
        [
            test_g2l_lut(n, E, topk)
            for n, E, topk in itertools.product(
                [64, 512], [e for e in args.experts if e <= 512], [2, 8]
            )
            if E <= n
        ],
    )
    summarize(
        "moe_contiguous_psum",
        [
            test_contiguous_psum(E, tile_m)
            for E, tile_m in itertools.product(args.experts + [896], args.tile_m)
        ],
    )
    summarize(
        "moe_psum_remap",
        [
            test_psum_remap(n, E, tm)
            for n, E, tm in itertools.product(
                args.numel, [e for e in args.experts if e <= 64], args.tile_m
            )
        ],
    )
    summarize(
        "moe_scatter_copy_token",
        [
            test_scatter_copy_token(nd, rb)
            for nd, rb in itertools.product([51, 1024], args.row_bytes)
        ],
    )
    summarize(
        "moe_gather_reduce",
        [
            test_gather_reduce(t, md, tk, v, dmap[d])
            for d, t, md, tk, v in itertools.product(
                args.dtype, args.tokens, args.model_dim, args.topk, args.vec
            )
        ],
    )
    # Real decode shape (model_dim=7168) at both vector widths -- swept apart from
    # the cross-product above so the big rows are actually visible in the table.
    summarize(
        "moe_gather_reduce (model_dim=7168)",
        [
            test_gather_reduce(t, 7168, tk, v, dmap[args.dtype[0]])
            for t, tk, v in itertools.product(args.tokens, args.topk, args.vec)
        ],
    )
    summarize(
        "moe_psum_remap_ep",
        [
            test_psum_remap_ep(n, E, tm, 4)
            for n, E, tm in itertools.product(
                args.numel, [e for e in args.experts if e <= 64], args.tile_m
            )
        ],
    )
    summarize(
        "moe_route_g2l",
        [
            test_route_g2l(n, eg, nb, v, wd)
            for n, (eg, nb), v, wd in itertools.product(
                args.numel,
                [(64, 8), (256, 40)],  # global experts -> local buckets (EP shard)
                ("lds", "plain"),
                args.dtype,
            )
        ],
    )
    summarize(
        "moe_topids_to_rows",
        [
            # Size the per-expert capacity to the sweep: uniform routing gives
            # ~numel/E per bucket, so 4x that leaves ample headroom for the
            # random imbalance without inventing an over-capacity config.
            test_topids_to_rows(n, E, max(512, 4 * n // E))
            for n, E in itertools.product(
                args.numel, [e for e in args.experts if e <= 64]
            )
        ],
    )
    summarize(
        "moe_route_maps",
        [
            test_route_maps(n, E, tk, max(512, 4 * n // E))
            for n, E, tk in itertools.product(
                args.numel, [e for e in args.experts if e <= 64], args.topk
            )
            if n % tk == 0
        ],
    )
    # Single-workgroup fused variants: the scan is one block, so E (and E_global
    # for the g2l fusion) is capped at the 512-thread block size.
    summarize(
        "moe_route_psum_fused",
        [
            test_route_psum_fused(n, E, tm)
            for n, E, tm in itertools.product(
                args.numel, [e for e in args.experts if e <= 512], args.tile_m
            )
        ],
    )
    summarize(
        "moe_route_g2l_fused",
        [
            test_route_g2l_fused(n, eg, nb, wd)
            for n, (eg, nb), wd in itertools.product(
                args.numel,
                [(64, 8), (256, 40), (512, 300)],
                args.dtype,
            )
        ],
    )
    # gather=True is the stage1 fused route-gather; gather=False the stage2 pure
    # preshuffle. row_bytes = K//32, so 224 is the K=7168 decode shape.
    summarize(
        "moe_preshuffle_scale",
        [
            test_preshuffle_scale(E, mm, rb, wr, 4, g)
            for E, mm, rb, wr, g in itertools.product(
                [2, 8], [32, 256], [16, 224], [1, 2], (True, False)
            )
            if mm % (wr * 16) == 0
        ],
    )


if __name__ == "__main__":
    main()
