# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Did PR #6369's low-latency fused-norm build ever help, in its original form?

Times ``QuickAllReduceInt4.allreduce_rmsnorm`` exactly as #6369 shipped it
(commit ``3c10d34a4``), with its low-latency build on and off, at the same shape.

The PR's host takes the low-latency build by itself whenever the payload is at
most one tile per workgroup at two workgroups per CU (``st == 1 and num_tiles <=
min(grid, latency_max_tiles)``; 512 tiles = 16 MiB on a 256-CU part). Setting
the instance's ``_latency_max_tiles`` to 0 makes the same call take the
occupancy-pinned build of the same ST=1 engine instead, so the two runs differ
in that one choice. Where the low-latency build does not engage (``ll`` is
False), both runs are the same kernel and the ratio is noise.

The PR's files are extracted from the commit into ``--work`` (they no longer
exist in the tree) and imported from there as the package ``pr6369_qr``; the
helpers they import from ``aiter`` by absolute path are the current ones.

Per shape: ``ll_us`` (as shipped), ``pinned_us`` (low-latency off),
``pinned/ll`` (> 1: the low-latency build is faster) and whether both builds
wrote bitwise-equal ``out``/``residual_out``. Each time is the slowest rank's
median over ``--rounds`` rounds that alternate the two builds, each round a
``run_perftest`` of ``--iters`` iterations with CUDA events, as the PR's test
timed.

Default shapes per TP: the PR's benchmark shapes (TP8, TP4) and a sweep of
small payloads, where the low-latency build engages.

usage: pr6369_low_latency.py [--tp 8 4 2] [--shapes T,H ...] [--work DIR]
                             [--csv FILE] [--rounds N] [--iters N]
"""

from __future__ import annotations

import argparse
import functools
import os
import statistics
import subprocess
import sys
from multiprocessing import Pool, freeze_support, set_start_method

# A second copy of libamdhip64 breaks CustomAllreduce's IPC checks (tune.sh).
os.environ.pop("TRITON_LIBHIP_PATH", None)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_MULTIGPU = os.path.join(_REPO, "op_tests", "multigpu_tests")
for _p in (_REPO, _MULTIGPU):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import torch

set_start_method("spawn", force=True)

COMMIT = "3c10d34a4"
PKG = "pr6369_qr"
# Tree path at COMMIT -> path under the vendored package.
_FILES = {
    "aiter/ops/flydsl/quick_allreduce_int4.py": "quick_allreduce_int4.py",
    "aiter/ops/flydsl/quick_allreduce_int4_ipc.py": "quick_allreduce_int4_ipc.py",
    "aiter/ops/flydsl/kernels/quick_allreduce_int4.py": "kernels/quick_allreduce_int4.py",
    "aiter/ops/flydsl/kernels/buffer_ops.py": "kernels/buffer_ops.py",
    "aiter/ops/flydsl/kernels/quant_utils.py": "kernels/quant_utils.py",
    "aiter/ops/flydsl/kernels/tensor_shim.py": "kernels/tensor_shim.py",
}
EPS = 1e-6
TILE_BYTES = 32 * 1024
# The PR's benchmark tables (MI355X).
PR_SHAPES = {
    8: [(2048, 8192), (8192, 8192), (16384, 8192), (16384, 4096)],
    4: [(256, 8192), (1024, 8192), (2048, 8192), (8192, 8192), (16384, 8192)],
    2: [],
}
# Small payloads, where the low-latency build engages (<= 16 MiB on 256 CUs).
SWEEP = [(t, h) for h in (2048, 4096, 8192) for t in (1, 4, 16, 64, 256, 1024)]


def vendor(work: str) -> str:
    """Extract the PR's files into ``work/src/pr6369_qr``; returns ``work/src``."""
    src = os.path.join(work, "src")
    for tree_path, rel in _FILES.items():
        dst = os.path.join(src, PKG, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            text = subprocess.run(
                ["git", "-C", _REPO, "show", f"{COMMIT}:{tree_path}"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except subprocess.CalledProcessError as e:
            raise SystemExit(
                f"cannot read {tree_path} at {COMMIT} (#6369 on main; "
                f"try: git fetch origin main): {e.stderr.strip()}"
            ) from e
        with open(dst, "w") as fh:
            fh.write(text)
    for pkg in (PKG, f"{PKG}/kernels"):
        open(os.path.join(src, pkg, "__init__.py"), "a").close()
    return src


def _time(fn, iters):
    from aiter.test_common import run_perftest

    _, us = run_perftest(fn, num_iters=iters, use_cuda_event=True)
    return us


def _run_rank(rank, tp, init_method, src, shapes, rounds, iters, grid_cap):
    import torch.distributed as dist

    sys.path.insert(0, src)
    from pr6369_qr.quick_allreduce_int4 import QuickAllReduceInt4

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=tp,
        rank=rank,
        device_id=device,
    )
    gloo = dist.new_group(backend="gloo")
    kw = {} if grid_cap is None else {"grid_cap": grid_cap}
    fly = QuickAllReduceInt4(group=gloo, device=device, rank=rank, world_size=tp, **kw)
    ll_max = fly._latency_max_tiles
    rows = []
    try:
        for hidden in sorted({h for _t, h in shapes}):
            fly.compile_rmsnorm(hidden, EPS)
        for ntok, hidden in shapes:
            gen = torch.Generator().manual_seed(1234 + rank)
            inp = (torch.randn((ntok, hidden), generator=gen) * 0.1).to(
                device=device, dtype=torch.bfloat16
            )
            shared = torch.Generator().manual_seed(99)
            res = torch.randn((ntok, hidden), generator=shared).to(
                device=device, dtype=torch.bfloat16
            )
            weight = (torch.randn((hidden,), generator=shared) * 0.2 + 1.0).to(
                device=device, dtype=torch.bfloat16
            )
            num_tiles = max(1, -(-ntok * hidden * 2 // TILE_BYTES))
            st = fly._pick_st(num_tiles)
            ll = st == 1 and num_tiles <= min(fly._by_st[1].grid, ll_max)

            outs = {}
            for mode in ("ll", "pinned"):
                fly._latency_max_tiles = ll_max if mode == "ll" else 0
                out, res_out = torch.empty_like(inp), torch.empty_like(inp)
                fly.allreduce_rmsnorm(inp, res, weight, EPS, out, res_out)
                torch.cuda.synchronize()
                outs[mode] = (out, res_out)
            equal = all(
                torch.equal(a, b) for a, b in zip(outs["ll"], outs["pinned"])
            )

            times = {"ll": [], "pinned": []}
            for rnd in range(rounds):
                order = ("ll", "pinned") if rnd % 2 == 0 else ("pinned", "ll")
                for mode in order:
                    fly._latency_max_tiles = ll_max if mode == "ll" else 0
                    out, res_out = outs[mode]
                    dist.barrier(group=gloo)
                    torch.cuda.synchronize()
                    # Bound, not passed: run_perftest deep-copies its args.
                    times[mode].append(
                        _time(
                            functools.partial(
                                fly.allreduce_rmsnorm,
                                inp,
                                res,
                                weight,
                                EPS,
                                out,
                                res_out,
                            ),
                            iters,
                        )
                    )
            fly._latency_max_tiles = ll_max
            rows.append(
                {
                    "tokens": ntok,
                    "hidden": hidden,
                    "tiles": num_tiles,
                    "st": st,
                    "ll": ll,
                    "ll_us": statistics.median(times["ll"]),
                    "pinned_us": statistics.median(times["pinned"]),
                    "equal": equal,
                    "grid": fly._by_st[st].grid,
                    "ll_max_tiles": ll_max,
                }
            )
            dist.barrier(group=gloo)
    finally:
        fly.close()
        dist.destroy_process_group()
    return rows


def _spawn(tp, src, shapes, rounds, iters, grid_cap):
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    init_method = get_distributed_init_method(get_ip(), get_open_port())
    timeout = float(os.environ.get("FLYDSL_QR_TIMEOUT", "3600"))
    pool = Pool(processes=tp)
    try:
        futs = [
            pool.apply_async(
                _run_rank,
                args=(rank, tp, init_method, src, shapes, rounds, iters, grid_cap),
            )
            for rank in range(tp)
        ]
        ranks = [f.get(timeout=timeout) for f in futs]
    except Exception:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()
    return ranks


def _merge(tp, ranks) -> pd.DataFrame:
    """One row per shape: the slowest rank's time per build."""
    out = []
    for i, r0 in enumerate(ranks[0]):
        row = {k: r0[k] for k in ("tokens", "hidden", "tiles", "st", "ll", "grid")}
        row["tp"] = tp
        row["KiB"] = r0["tokens"] * r0["hidden"] * 2 // 1024
        row["ll_us"] = max(r[i]["ll_us"] for r in ranks)
        row["pinned_us"] = max(r[i]["pinned_us"] for r in ranks)
        row["equal"] = all(r[i]["equal"] for r in ranks)
        out.append(row)
    df = pd.DataFrame(out)
    df["pinned/ll"] = df["pinned_us"] / df["ll_us"]
    return df


def main():
    import flydsl_allreduce_bench as fab
    from aiter.jit.utils.chip_info import get_gfx_runtime

    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    ap.add_argument("--tp", type=int, nargs="*", default=[8, 4, 2])
    ap.add_argument(
        "--shapes",
        nargs="*",
        default=None,
        help="tokens,hidden pairs at every TP (default: the PR's shapes + a sweep)",
    )
    ap.add_argument(
        "--work",
        default=os.path.join(_HERE, "out", "pr6369"),
        help="where the PR's files are extracted to (default: out/pr6369)",
    )
    ap.add_argument("--csv", default=None, help="also write the per-shape table here")
    ap.add_argument("--link", default="?", help="fabric label for the report")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=101)
    ap.add_argument("--grid-cap", type=int, default=None, help="default: the PR's")
    ap.add_argument(
        "--min-gain",
        type=float,
        default=0.03,
        help="pinned/ll above 1 + this counts as a low-latency win (default 0.03)",
    )
    args = ap.parse_args()

    n_gpu = fab.gpu_count()
    arch = get_gfx_runtime()
    src = vendor(args.work)
    # The FlyDSL disk cache does not hash imported kernel modules (tune.sh).
    os.environ["FLYDSL_EXTRA_SOURCE_DIRS"] = os.pathsep.join(
        p
        for p in (
            os.path.join(src, PKG, "kernels"),
            os.environ.get("FLYDSL_EXTRA_SOURCE_DIRS"),
        )
        if p
    )
    if args.shapes is not None:
        fixed = [tuple(int(x) for x in s.split(",")) for s in args.shapes]
    frames = []
    for tp in args.tp:
        if tp > n_gpu:
            print(f"skip tp{tp}: only {n_gpu} GPU(s) visible")
            continue
        shapes = (
            fixed
            if args.shapes is not None
            else sorted(set(PR_SHAPES.get(tp, [])) | set(SWEEP), key=lambda s: (s[1], s[0]))
        )
        print(f"tp{tp}: {len(shapes)} shape(s)", flush=True)
        ranks = _spawn(tp, src, shapes, args.rounds, args.iters, args.grid_cap)
        frames.append(_merge(tp, ranks))
    if not frames:
        raise SystemExit("no world size ran")
    df = pd.concat(frames, ignore_index=True)
    df["pr_shape"] = [
        (t, h) in PR_SHAPES.get(tp, []) for tp, t, h in zip(df.tp, df.tokens, df.hidden)
    ]
    if args.csv:
        df.assign(arch=arch, link=args.link).to_csv(args.csv, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(f"# PR #6369 ({COMMIT}) low-latency build on vs off ({arch}, {args.link})")
    print("pinned/ll > 1: the low-latency build is faster; times in us")
    cols = ["tokens", "hidden", "KiB", "tiles", "st", "grid", "ll", "pr_shape",
            "ll_us", "pinned_us", "pinned/ll", "equal"]
    for tp, g in df.groupby("tp", sort=False):
        print(f"\n## TP{tp}")
        print(g[cols].round(3).to_string(index=False))
    eng = df[df["ll"]]
    wins = eng[eng["pinned/ll"] > 1 + args.min_gain]
    losses = eng[eng["pinned/ll"] < 1 / (1 + args.min_gain)]
    geo = (
        float(eng["pinned/ll"].prod() ** (1 / len(eng))) if len(eng) else float("nan")
    )
    print(
        f"\nverdict ({arch}, {args.link}): the low-latency build engaged at "
        f"{len(eng)} of {len(df)} shape(s); faster by > {args.min_gain:.0%} at "
        f"{len(wins)}, slower at {len(losses)}; geomean pinned/ll {geo:.3f}"
    )
    if not df["equal"].all():
        print("note: the two builds' outputs differ bitwise at some shape(s) (column equal)")


if __name__ == "__main__":
    freeze_support()
    main()
