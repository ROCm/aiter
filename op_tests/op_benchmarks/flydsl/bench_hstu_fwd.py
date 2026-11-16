# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the FlyDSL HSTU forward on gfx942 at production-like shapes.

Backends, timed on the same bf16 inputs and seeded jagged lengths:

  ``fallback``  ``pingpong=False``: the general kernel, and the baseline.
  ``pingpong``  ``pingpong=True`` at the dispatcher's tile (--pp-tile pins one).
  ``default``   no override: shows which kernel the dispatch picks.
  ``ck``        the CK-tile binary, dropped when it is missing.

Each cell first checks that the FlyDSL backends' outputs agree. Masks: ``hstu``
(causal + targets) and ``windowed`` (max_attn_len=512, or --window sizes).
N=16384, sparsity 0.95, d=64 unless --head-dim; Training is B=1024 with 10
targets, Inference B=120 with 300. A pass is 3 warmup + 11 timed launches
(median; CK reports one mean per process); a cell is the median over --reps
passes, interleaved across backends. CK gets lengths - targets as -seqlens.

Usage, from the repo root:

    python -m op_tests.op_benchmarks.flydsl.bench_hstu_fwd \
        --reps 3 --head-dim 64 128 --backend fallback pingpong default
    python -m op_tests.op_benchmarks.flydsl.bench_hstu_fwd --mask windowed \
        --window 64 256 512 1024 --backend fallback pingpong default -o fwd.csv

CK needs tile_example_hstu_attention_fwd built with the HSTU_WARMUP/HSTU_ITERS
patch (path overridable with CK_HSTU_FWD_BIN):

    cd 3rdparty/composable_kernel && mkdir -p build && cd build
    cmake -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
          -G Ninja -D CMAKE_BUILD_TYPE=Release -D GPU_TARGETS="gfx942" ..
    ninja -j 50 tile_example_hstu_attention_fwd
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.hstu_attention import flydsl_hstu_attention_fwd

REPO = Path(__file__).resolve().parents[3]
CK_BIN = Path(
    os.environ.get(
        "CK_HSTU_FWD_BIN",
        REPO / "3rdparty/composable_kernel/build/bin/tile_example_hstu_attention_fwd",
    )
)
CK_SRC = REPO / (
    "3rdparty/composable_kernel/example/ck_tile/18_hstu_attention/"
    "example_hstu_attention_fwd.cpp"
)

# Fixed across every cell.
SEQ_LEN = 16384
DTYPE = torch.bfloat16
# Length-distribution knob, NOT model sparsity: `length = uniform[1, N] * 0.95`.
SPARSITY = 0.95
SEED = 1001
WARMUP = 3
ITERS = 11

# mask -> max_attn_len (0 = no window); --window overrides the windowed size.
MASKS = {"hstu": 0, "windowed": 512}

# (regime, batch, heads, target_size), run once per selected mask.
CELLS = [
    ("Training", 1024, 4, 10),
    ("Training", 1024, 8, 10),
    ("Inference", 120, 4, 300),
    ("Inference", 120, 8, 300),
]

BACKENDS = ("fallback", "pingpong", "default", "ck")
LABELS = {
    "fallback": "FlyDSL fallback",
    "pingpong": "FlyDSL ping-pong",
    "default": "FlyDSL default",
    "ck": "CK",
}
BASELINE = "fallback"
CK_TIMING = re.compile(r"execution time .*? is ([0-9.]+) milli-seconds")

# Each backend is within atol=1e-3 of the torch reference (outputs scaled by
# N) in the tests, so two correct backends differ by at most twice that.
AGREE_ATOL = 2e-3


def alpha_for(head_dim: int) -> float:
    return 1.0 / head_dim * 10000


def make_inputs(b, h, head_dim, target_size):
    """Jagged q/k/v, offsets, targets and lengths. One seed fixes all of them."""
    g = torch.Generator(device="cuda").manual_seed(SEED)
    lengths = torch.randint(1, SEQ_LEN + 1, (b,), device="cuda", generator=g)
    lengths = (lengths.float() * SPARSITY).clamp(min=1.0).to(torch.int64)
    offsets = torch.zeros(b + 1, dtype=torch.int64, device="cuda")
    offsets[1:] = torch.cumsum(lengths, dim=0)
    total = int(offsets[-1].item())
    x = torch.empty((total, h, 3 * head_dim), dtype=DTYPE, device="cuda").uniform_(
        -0.01, 0.01, generator=g
    )
    q, k, v = (t.contiguous() for t in torch.split(x, head_dim, dim=-1))
    num_targets = torch.minimum(
        torch.full((b,), target_size, device="cuda"), lengths
    ).to(torch.int32)
    return q, k, v, offsets, num_targets, lengths


def time_pass(fn) -> np.ndarray:
    """Per-launch latencies in ms for one pass of WARMUP + ITERS launches.

    Each launch has its own event pair and the pass syncs once at the end, so
    launches stay pipelined. The warmup absorbs JIT compile, which is never timed.
    """
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for s, e in zip(starts, ends):
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])


def check_outputs_agree(outputs, rows=1 << 16):
    """Fail the cell if any FlyDSL backend's output disagrees with the fallback.

    The dense torch reference does not fit at these shapes, so the comparison
    runs on a random sample of rows, scaled by max_seq_len as in the tests.
    """
    ref = outputs[BASELINE]
    g = torch.Generator(device=ref.device).manual_seed(0)
    idx = torch.randint(0, ref.shape[0], (min(rows, ref.shape[0]),), generator=g)
    ref_s = ref[idx].float() * SEQ_LEN
    for name, out in outputs.items():
        err = float((out[idx].float() * SEQ_LEN - ref_s).abs().max())
        if not err <= AGREE_ATOL:
            raise AssertionError(
                f"{name} vs {BASELINE}: max|diff| {err:.3e} (scaled by N) "
                f"exceeds {AGREE_ATOL:.0e}"
            )


def run_ck(b, h, head_dim, lengths, num_targets, max_attn_len) -> float:
    """One CK process = one pass. Returns its reported mean over the timed launches."""
    uih = (lengths - num_targets.to(lengths.dtype)).tolist()
    cmd = [
        str(CK_BIN), "-prec=bf16", f"-b={b}", f"-nhead={h}",
        f"-hdim_qk={head_dim}", f"-hdim_v={head_dim}", "-jagged=1",
        "-v=0", "-perf=1", "-causal=1",
        f"-local_len={max_attn_len}", "-context_len=0", "-minfull_len=0",
        f"-alpha={alpha_for(head_dim)}",
        "-seqlens=" + ",".join(map(str, uih)),
        "-targets=" + ",".join(map(str, num_targets.tolist())),
    ]  # fmt: skip
    env = dict(os.environ, HSTU_WARMUP=str(WARMUP), HSTU_ITERS=str(ITERS))
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=3600, env=env, check=False
    )
    match = CK_TIMING.search(proc.stdout)
    if not match:
        raise RuntimeError(
            f"CK produced no timing (rc={proc.returncode})\n"
            f"stdout: {proc.stdout.strip()[-500:]}\n"
            f"stderr: {proc.stderr.strip()[-500:]}"
        )
    return float(match.group(1))


def bench_cell(
    mask, max_attn_len, regime, b, h, target_size, head_dim, backends, reps, pp_tile
):
    """Time the selected backends on one cell. Returns one row per backend."""
    q, k, v, offsets, num_targets, lengths = make_inputs(b, h, head_dim, target_size)
    total = int(offsets[-1].item())
    fwd_args = (
        SEQ_LEN, alpha_for(head_dim), q, k, v, offsets, True, num_targets,
        max_attn_len, 0,
    )  # fmt: skip
    tile = {"block_m": pp_tile[0], "block_n": pp_tile[1]} if pp_tile else {}
    flydsl = {
        "fallback": lambda: flydsl_hstu_attention_fwd(*fwd_args, pingpong=False),
        "pingpong": lambda: flydsl_hstu_attention_fwd(*fwd_args, pingpong=True, **tile),
        "default": lambda: flydsl_hstu_attention_fwd(*fwd_args),
    }
    launch = {name: fn for name, fn in flydsl.items() if name in backends}

    if BASELINE in launch and len(launch) > 1:
        check_outputs_agree({name: fn() for name, fn in launch.items()})

    # Each pass is (median, min) over its launches. CK's pass is one mean.
    passes = {p: [] for p in backends}
    for rep in range(reps):
        order = backends if rep % 2 == 0 else backends[::-1]
        for name in order:
            if name == "ck":
                value = run_ck(b, h, head_dim, lengths, num_targets, max_attn_len)
                passes[name].append((value, value))
            else:
                times = time_pass(launch[name])
                passes[name].append((float(np.median(times)), float(times.min())))

    medians = {p: float(np.median([m for m, _ in passes[p]])) for p in backends}
    mins = {p: float(min(x for _, x in passes[p])) for p in backends}
    base = medians.get(BASELINE, float("nan"))
    ck = medians.get("ck", float("nan"))

    rows = []
    for name in backends:
        rows.append({
            "mask": mask, "regime": regime, "B": b, "H": h, "d": head_dim,
            "targets": target_size, "backend": LABELS[name], "L": total,
            "median_ms": medians[name], "min_ms": mins[name],
            "vs fallback": base / medians[name],
            "vs CK": ck / medians[name],
        })  # fmt: skip

    return rows


def format_table(df):
    """One row per shape, one ms column per backend, plus speedups over the fallback.

    Speedups are time ratios fallback / backend: above 1.00 means the backend
    is faster. The fastest backend in each row is bold.
    """
    short = {
        "fallback": "fallback",
        "pingpong": "ping-pong",
        "default": "default",
        "ck": "CK",
    }
    present = [p for p in BACKENDS if LABELS[p] in set(df["backend"])]
    ratios = [p for p in ("pingpong", "default") if p in present]
    if BASELINE not in present:
        ratios = []
    cols = (
        ["Shape"]
        + [f"{short[p]} (ms)" for p in present]
        + [f"{short[p]} speedup" for p in ratios]
    )
    keys = ["mask", "regime", "B", "H", "d", "targets"]
    out = []
    for (mask, regime, b, h, d, _t), g in df.groupby(keys, sort=False):
        ms = {
            p: float(g[g["backend"] == LABELS[p]]["median_ms"].iloc[0]) for p in present
        }
        best = min(ms.values())
        kind = "train" if regime == "Training" else "infer"
        shape = f"{kind} B={b} H={h} d={d}" + (f" {mask}" if mask != "hstu" else "")
        row = [shape]
        for p in present:
            cell = f"{ms[p]:.2f}"
            row.append(f"**{cell}**" if ms[p] == best else cell)
        row += [f"{ms[BASELINE] / ms[p]:.2f}x" for p in ratios]
        out.append(row)
    return pd.DataFrame(out, columns=cols).to_markdown(
        index=False, disable_numparse=True
    )


def parse_args():
    parser = argparse.ArgumentParser(
        prog="bench_hstu_fwd",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Benchmark the FlyDSL HSTU forward at production-like shapes "
        "(N=16384, bf16, hstu mask): fallback vs ping-pong vs default dispatch vs CK.",
    )
    parser.add_argument(
        "--backend", nargs="*", default=list(BACKENDS), choices=list(BACKENDS),
        help="""Backends to time.
        e.g.: --backend fallback pingpong default""",
    )  # fmt: skip
    parser.add_argument(
        "--mask", nargs="*", default=list(MASKS), choices=list(MASKS),
        help="""Masks to run: hstu (max_attn_len=0) and/or windowed (max_attn_len=512,
        or the sizes given by --window).
        e.g.: --mask windowed""",
    )  # fmt: skip
    parser.add_argument(
        "--window", type=int, nargs="+", default=[MASKS["windowed"]],
        help="""max_attn_len values for the windowed mask; each one is a full set
        of cells. Default 512.
        e.g.: --window 32 64 128 256 512 1024 2048 4096""",
    )  # fmt: skip
    parser.add_argument(
        "--regime", nargs="*", default=["Training", "Inference"],
        choices=["Training", "Inference"],
        help="""Regimes to run. These name B=1024 (targets 10) and B=120 (targets 300).
        e.g.: --regime Inference""",
    )  # fmt: skip
    parser.add_argument(
        "-H", "--heads", type=int, nargs="*", default=[4, 8],
        help="""Head counts to keep.
        e.g.: -H 4""",
    )  # fmt: skip
    parser.add_argument(
        "--head-dim", type=int, nargs="*", default=[64],
        help="""Head dims (d_qk == d_v) to sweep.
        e.g.: --head-dim 64 128""",
    )  # fmt: skip
    parser.add_argument(
        "--reps", type=int, default=1,
        help="""Passes of (3 warmup + 11 timed) per backend. The cell reports the
        median over passes. Default 1.""",
    )  # fmt: skip
    parser.add_argument(
        "--pp-tile", default=None,
        help="""Ping-pong BLOCK_M,BLOCK_N for every cell. Default: the dispatcher's
        geometry for the shape.""",
    )  # fmt: skip
    parser.add_argument(
        "-o", type=str, default=None,
        help="""Also write the full results to this CSV (incl. L and min_ms).""",
    )  # fmt: skip
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_default_device("cuda")
    if get_gfx_runtime() != "gfx942":
        aiter.logger.warning(
            "the hstu fwd ping-pong is gfx942-only; skipping on %s", get_gfx_runtime()
        )
        return
    backends = [p for p in BACKENDS if p in args.backend]

    if "ck" in backends:
        if not CK_BIN.exists():
            aiter.logger.warning("CK forward binary missing at %s; dropping ck", CK_BIN)
            backends.remove("ck")
        elif not CK_SRC.exists() or "HSTU_WARMUP" not in CK_SRC.read_text():
            aiter.logger.warning(
                "CK source lacks the HSTU_WARMUP/HSTU_ITERS patch; CK will run its "
                "hardcoded iteration count with no warmup"
            )
    if not backends:
        return

    pp_tile = tuple(int(x) for x in args.pp_tile.split(",")) if args.pp_tile else None

    # (label, max_attn_len) per mask instance; a swept window gets one per size.
    masks = []
    if "hstu" in args.mask:
        masks.append(("hstu", MASKS["hstu"]))
    if "windowed" in args.mask:
        sweep = args.window != [MASKS["windowed"]]
        masks += [(f"windowed W={w}" if sweep else "windowed", w) for w in args.window]
    cells = [
        (m, w, *c, d)
        for d in args.head_dim
        for m, w in masks
        for c in CELLS
        if c[0] in args.regime and c[2] in args.heads
    ]
    rows = []
    for index, cell in enumerate(cells, 1):
        mask, _, regime, b, h, target_size, d = cell
        tag = f"{mask} {regime} B={b} H={h} d={d} targets={target_size}"
        # Printed before the cell runs so a long compile is not silent.
        print(f"[{index}/{len(cells)}] {tag} ... ", end="", flush=True)
        cell_rows = bench_cell(*cell, backends, args.reps, pp_tile)
        rows += cell_rows
        # The cell's tensors are unreachable once bench_cell returns.
        gc.collect()
        torch.cuda.empty_cache()
        print(
            "  ".join(f"{r['backend']} {r['median_ms']:.2f}" for r in cell_rows),
            flush=True,
        )

    df = pd.DataFrame(rows)
    aiter.logger.info("hstu fwd, all cells (markdown):\n%s", format_table(df))
    if args.o:
        df.to_csv(args.o, index=False)
        aiter.logger.info("results saved to %s", args.o)


if __name__ == "__main__":
    main()
