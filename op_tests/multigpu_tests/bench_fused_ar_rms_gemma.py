# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP4 timing and numerics of every backend that can serve the fused Gemma
all-reduce + residual add + RMSNorm (MiniMax-M3.1: hidden 6144, bf16), through
the real interface (``tensor_model_parallel_fused_allreduce_rmsnorm`` inside
aiter's ``graph_capture``).

Candidates (rows x 6144 bf16 per rank):

* ``iface``          the interface as configured (pull first when it accepts)
* ``iface_nopull``   the interface with the pull engine removed: what aiter
                     did before the pull schedule (custom fused AR+RMS 1-/2-stage)
* ``custom_1stage`` / ``custom_2stage``   ``CustomAllreduce.custom_fused_ar_rms``
* ``unfused``        ``all_reduce`` + ``rmsnorm2d_fwd_with_add`` (gemma)
* ``pf_*``           one pull kernel forced on every row (ladder fit): one-shot
                     with acks, one-row pull (block / wave hand-offs), streamed
                     pull at 2 / 4 rows per workgroup, with and without the
                     weight hoist. Outside the shipped window these are
                     experiments: the numerics columns say whether a kernel is
                     exact there.

The FlyDSL one-shot / quick-reduce fused ops decline ``gemma_norm`` (they take a
plain weight), so they cannot serve this call and are not candidates.

Timing: ``--ncall`` calls per graph on distinct inputs, ``--rounds`` ABAB
rounds (candidate order flips every round), median of the rounds after two
warm-up rounds, maximum over the four ranks. ``--eager`` times eager calls
(prefill without a token graph) with events instead of graphs.

Numerics (every candidate, last call of the graph): ``res`` is the number of
``residual_out`` elements that differ bit for bit from the fp32 two-stage
reference (chunk owner first, one bf16 rounding, as AITER's two-stage
all-reduce), ``ulp`` the largest bf16 distance of ``out``.

    python3 op_tests/multigpu_tests/bench_fused_ar_rms_gemma.py --rows 8 64 512
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import statistics
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
H = 6144
EPS = 1e-6
DEFAULT_ROWS = (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 640)
PREFILL_ROWS = (1024, 2048, 4096)
# rows the one-shot with read acks is forced on: the shipped ladder uses it to 11 rows
ONESHOT_MAX_ROWS = 16
# a payload whose kernel is exact when the reference says so, in bf16 ulps
EXACT_ULP = 1


def _data(rows, seed, rank, scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed * 7919 + rank * 131 + rows)
    return (torch.randn(rows, H, generator=g) * scale).to(torch.bfloat16)


def _reference(xs, residual, weight):
    """fp32 two-stage all-reduce + residual add + Gemma RMSNorm of the ranks'
    partials *xs* (list of [rows, H] on the device)."""
    rows = residual.shape[0]
    q = -(-rows // WORLD)
    reduced = torch.empty(rows, H, dtype=torch.float32, device=residual.device)
    for c in range(WORLD):
        lo, hi = c * q, min(rows, (c + 1) * q)
        if lo >= hi:
            continue
        acc = xs[c][lo:hi].float()
        for k in range(1, WORLD):
            acc = acc + xs[(c + k) % WORLD][lo:hi].float()
        reduced[lo:hi] = acc
    s = reduced.to(torch.bfloat16).float() + residual.float()
    res_out = s.to(torch.bfloat16)
    rstd = torch.rsqrt((s * s).sum(-1, keepdim=True) / H + EPS)
    out = (s * rstd * (weight.float() + 1.0)).to(torch.bfloat16)
    return out, res_out


def _forced_rungs(rows):
    """The pull kernels worth timing at *rows*, as ``(name, FusedPullRung)``;
    the one-shot only where its flag table has room (``ACK_ROWS``)."""
    from aiter.ops.flydsl.allreduce_policy import FusedPullRung as R

    out = []
    if rows <= ONESHOT_MAX_ROWS:
        out.append(("pf_oneshot", R(0, "oneshot")))
    out.append(("pf_pull_block", R(0, "pull")))
    out.append(("pf_pull_wave", R(0, "pull", wave=True)))
    if rows >= 16:
        for rpw in (2, 4):
            out.append((f"pf_stream_r{rpw}", R(0, "stream", rows_per_wg=rpw)))
            out.append(
                (
                    f"pf_stream_r{rpw}_nohoist",
                    R(0, "stream", rows_per_wg=rpw, hoist=False),
                )
            )
    return out


def _worker(rank, args, port, result_q):
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    from aiter import rmsnorm2d_fwd_with_add
    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm as fused,
    )
    from aiter.dist.parallel_state import (
        ensure_model_parallel_initialized,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        set_custom_all_reduce,
    )
    from aiter.ops.flydsl import allreduce_policy as policy
    from aiter.ops.flydsl.pull_allreduce_rmsnorm import PullAllReduceRMSNorm

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=WORLD,
        rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    ensure_model_parallel_initialized(WORLD, 1)
    tp = get_tp_group()
    dc = tp.device_communicator
    ca = dc.ca_comm
    group = tp.cpu_group
    pull_on = ca is not None and getattr(ca, "_fly_pull", None) is not None
    if rank == 0:
        print(f"pull engine built: {pull_on}; ca_comm: {ca is not None}", flush=True)
    weight = (
        (torch.randn(H, generator=torch.Generator().manual_seed(7)) * 0.3)
        .to(torch.bfloat16)
        .to(dev)
    )

    # forced pull engines, one per kernel configuration (collective, same order)
    forced = {}
    if "pf" in args.groups:
        names = {}
        for rows in args.rows:
            for name, rung in _forced_rungs(rows):
                names[name] = rung
        for name in sorted(n for n in names if not args.cands or n in args.cands):
            rung = names[name]
            # An unbounded window makes _check_ladder() measure every single-rung ladder
            # against the full MAX_ROWS=1024: a 16-row oneshot or 640-row pull rung then
            # raises at construction (its one rung would also have to serve rows above its
            # known cap). Bound the window to the rung's own kernel cap instead, so cells
            # outside it decline through accepts() during capture rather than aborting setup.
            cap = policy.PULL_KERNEL_MAX_ROWS[rung.kernel]
            forced[name] = PullAllReduceRMSNorm(
                group=group,
                device=dev,
                rank=rank,
                world_size=WORLD,
                window=policy.Window(0, (cap + 1) * 2 * H),
                ladders={H: (rung,)},
            )
            forced[name].preload(H)
    dist.barrier(group)

    def impl(name, x, r, w):
        if name == "iface":
            return fused(x, r, w, EPS, gemma_norm=True)
        if name == "iface_nopull":
            saved = ca._fly_pull
            ca._fly_pull = None
            try:
                return fused(x, r, w, EPS, gemma_norm=True)
            finally:
                ca._fly_pull = saved
        if name in ("custom_1stage", "custom_2stage"):
            return ca.custom_fused_ar_rms(
                x, r, w, EPS, name == "custom_1stage", out_hidden_dim=H, gemma_norm=True
            )
        if name == "unfused":
            ar = dc.all_reduce(x)
            out = torch.empty_like(ar)
            res = torch.empty_like(ar)
            rmsnorm2d_fwd_with_add(out, ar, r, res, w, EPS, True, 0)
            return out, res
        eng = forced[name]
        return eng.allreduce_rmsnorm(x, r, w, EPS)

    groups = set(args.groups)
    base = ["iface", "iface_nopull", "custom_1stage", "custom_2stage", "unfused"]
    rows_out = []
    for rows in args.rows:
        names = list(base) if "base" in groups else []
        if "pf" in groups:
            names += [n for n, _ in _forced_rungs(rows)]
        if args.cands:
            names = [n for n in names if n in args.cands]
        if rows < 1:
            continue
        # distinct inputs per call so the replay is not all cache hits
        nx = max(2, min(args.ncall, (2 << 30) // (3 * rows * H * 2)))
        xs = [_data(rows, 100 + i, rank).to(dev) for i in range(nx)]
        rs = [_data(rows, 500 + i, rank, 0.5).to(dev) for i in range(nx)]
        ncall = args.ncall if rows <= 1024 else min(args.ncall, 16)
        graphs, last = {}, {}
        ok = {}
        with contextlib.ExitStack() as st:
            for e in forced.values():
                st.enter_context(e.capture())
            gc_ = st.enter_context(graph_capture())
            for name in names:
                try:
                    gr = torch.cuda.CUDAGraph()
                    outs = []
                    with torch.cuda.graph(gr, stream=gc_.stream):
                        for i in range(ncall):
                            outs.append(impl(name, xs[i % nx], rs[i % nx], weight))
                    graphs[name] = gr
                    last[name] = outs[-1]
                    ok[name] = True
                except Exception as e:  # noqa: BLE001
                    if rank == 0:
                        print(f"rows={rows} {name}: capture declined: {e!r}"[:200])
                    ok[name] = False
        # numerics of the last call, against the reference built from the ranks' inputs
        li = (ncall - 1) % nx
        ref_out, ref_res = _reference(
            [_data(rows, 100 + li, r).to(dev) for r in range(WORLD)], rs[li], weight
        )
        num = {}
        for name, graph in graphs.items():
            if rank == 0:
                print(f"rows={rows} {name}: numerics replay", flush=True)
            torch.cuda.synchronize()
            dist.barrier(group)
            graph.replay()
            torch.cuda.synchronize()
            o, rr = last[name]
            res = (rr.view(torch.int16) != ref_res.view(torch.int16)).sum().item()
            ulp = (o.view(torch.int16).int() - ref_out.view(torch.int16).int()).abs()
            num[name] = (res, int(ulp.max()))
        # timing: ABAB
        order = list(graphs)
        times = {n: [] for n in order}
        for rd in range(args.rounds + 2):
            if rank == 0 and rd == 0:
                print(f"rows={rows}: timing rounds", flush=True)
            for name in order if rd % 2 == 0 else order[::-1]:
                torch.cuda.synchronize()
                dist.barrier(group)
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                graphs[name].replay()
                e.record()
                torch.cuda.synchronize()
                times[name].append(s.elapsed_time(e) * 1000.0 / ncall)
        med = torch.tensor(
            [statistics.median(times[n][2:]) for n in order], dtype=torch.float64
        )
        dist.all_reduce(med, op=dist.ReduceOp.MAX, group=group)
        nm = torch.tensor([v for n in order for v in num[n]], dtype=torch.int64)
        dist.all_reduce(nm, op=dist.ReduceOp.MAX, group=group)
        for i, name in enumerate(order):
            rows_out.append(
                {
                    "mode": "graph",
                    "rows": rows,
                    "bytes": rows * H * 2,
                    "cand": name,
                    "us": med[i].item(),
                    "res_mismatch": int(nm[2 * i]),
                    "out_max_ulp": int(nm[2 * i + 1]),
                }
            )
        if rank == 0:
            line = ", ".join(f"{n} {med[i].item():.2f}" for i, n in enumerate(order))
            print(f"graph rows={rows}: {line}", flush=True)
        del graphs, last, xs, rs
        torch.cuda.empty_cache()

    if args.eager:
        for rows in args.eager_rows:
            x = _data(rows, 100, rank).to(dev)
            r = _data(rows, 500, rank, 0.5).to(dev)
            names = ["iface", "custom_2stage", "unfused"]
            res = {n: [] for n in names}
            for rd in range(args.rounds + 2):
                for name in names if rd % 2 == 0 else names[::-1]:
                    torch.cuda.synchronize()
                    dist.barrier(group)
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(args.eager_calls):
                        impl(name, x, r, weight)
                    e.record()
                    torch.cuda.synchronize()
                    res[name].append(s.elapsed_time(e) * 1000.0 / args.eager_calls)
            med = torch.tensor(
                [statistics.median(res[n][2:]) for n in names], dtype=torch.float64
            )
            dist.all_reduce(med, op=dist.ReduceOp.MAX, group=group)
            for i, name in enumerate(names):
                rows_out.append(
                    {
                        "mode": "eager",
                        "rows": rows,
                        "bytes": rows * H * 2,
                        "cand": name,
                        "us": med[i].item(),
                        "res_mismatch": -1,
                        "out_max_ulp": -1,
                    }
                )
            if rank == 0:
                print(
                    f"eager rows={rows}: "
                    + ", ".join(
                        f"{n} {med[i].item():.2f}" for i, n in enumerate(names)
                    ),
                    flush=True,
                )
    dist.barrier(group)
    if rank == 0:
        result_q.put(rows_out)
    for e in forced.values():
        e.close()
    dist.barrier(group)
    dist.destroy_process_group()


def _summary(rows_out):
    """Markdown: per mode and row count, every candidate's microseconds and the
    fastest candidate that is exact (residual bit-exact, out within 1 ulp)."""
    lines = []
    for mode in ("graph", "eager"):
        sel = [r for r in rows_out if r["mode"] == mode]
        if not sel:
            continue
        cands = sorted({r["cand"] for r in sel})
        lines.append(f"\n### {mode}\n")
        lines.append("| rows | " + " | ".join(cands) + " | winner (exact) |")
        lines.append("|---|" + "---|" * (len(cands) + 1))
        for rows in sorted({r["rows"] for r in sel}):
            at = {r["cand"]: r for r in sel if r["rows"] == rows}
            cells, best = [], None
            for c in cands:
                r = at.get(c)
                if r is None:
                    cells.append("-")
                    continue
                exact = mode == "eager" or (
                    r["res_mismatch"] == 0 and r["out_max_ulp"] <= EXACT_ULP
                )
                tag = (
                    ""
                    if exact
                    else f" (res {r['res_mismatch']}, ulp {r['out_max_ulp']})"
                )
                cells.append(f"{r['us']:.2f}{tag}")
                if exact and (best is None or r["us"] < best[1]):
                    best = (c, r["us"])
            lines.append(
                f"| {rows} | " + " | ".join(cells) + f" | {best[0] if best else '-'} |"
            )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="*", default=list(DEFAULT_ROWS))
    ap.add_argument("--groups", nargs="*", default=["base", "pf"], help="base, pf")
    ap.add_argument("--cands", nargs="*", default=None, help="only these candidates")
    ap.add_argument("--ncall", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=21)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--eager-rows", type=int, nargs="*", default=list(PREFILL_ROWS))
    ap.add_argument("--eager-calls", type=int, default=8)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if torch.cuda.device_count() < WORLD:
        print(
            f"fewer than {WORLD} GPUs visible ({torch.cuda.device_count()}): "
            "skipping the pull/custom-fused AR+RMS benchmark"
        )
        return
    if get_gfx_runtime() != "gfx950":
        print(
            f"the pull schedule is gfx950-only, not {get_gfx_runtime()}: "
            "skipping the benchmark"
        )
        return
    port = 29800 + os.getpid() % 150
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    mp.spawn(_worker, args=(args, port, q), nprocs=WORLD, join=True)
    rows_out = q.get()
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
    print(_summary(rows_out))


if __name__ == "__main__":
    sys.exit(main())
