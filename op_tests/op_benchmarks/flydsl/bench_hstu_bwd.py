# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the FlyDSL HSTU backward on gfx942, at the SILOTIGER-1197 shapes.

Times two backends on identical bf16 inputs and reports the speedup of one
over the other:

  ``split``   today's pair -- the KV-owned dV/dK kernel, then a second pass for
              dQ. The only path that covers fp16, head_dim != 64 and gfx950, so
              it is the baseline and the fallback.
  ``fused``   the single-pass dV/dK/dQ kernel, which emits dQ as a split-K
              partial per KV tile and sums it with a reduction kernel. Gated to
              gfx942 / bf16 / head_dim == hidden_dim == 64.

The fused arm is reported as ``fused`` + ``dq_reduce`` as well as a total: the
reduction is fully EXPOSED -- reduce(g) must finish before fused(g+1) overwrites
the scratch -- so it is a standing cost to tune independently, not something
hiding under the next pass.

Each arm runs ONE pass of per-iteration event-timed launches on one GPU, and
the median of those iterations is reported. A single pass is enough here: the
spread between an arm's median and its minimum is 0.1-1.5% on a locked idle
GPU, well under any difference this table is read for. Run it on an idle card
(tmp/pick_idle_gpu.sh) -- a neighbour on the package moves these far more than
the sampling does.

``--meta`` additionally prints, per cell, the compiled-kernel resources (VGPR /
LDS / spills, and the occupancy they imply) and the device footprint -- the fp32
split-K scratch dominates the latter and is the reason kv-chunking exists at
all. The resources come from a forked worker, because the ISA dump needs
``FLYDSL_DUMP_IR`` set before FlyDSL is imported and would otherwise dump on
every compile in the timed run.

A note on the regime labels: ``Training`` and ``Inference`` are the ticket's
names for **B=1024 and B=120**. Both are BACKWARD measurements -- neither
denotes a forward path. The B=120 cells are cheap and are the only ones where
the scratch fits in a single pass (G=1), which is a distinct code path.

Usage -- run from the repo root, which has to be on ``sys.path`` for the
``op_tests`` import below to resolve (``-m`` does that; a plain script path
would put this file's own directory there instead):

    # Every cell, both backends
    python -m op_tests.op_benchmarks.flydsl.bench_hstu_bwd

    # The two B=120 cells only, with kernel resources and footprint
    python -m op_tests.op_benchmarks.flydsl.bench_hstu_bwd --regime Inference --meta

    # Fused only, written to CSV
    python -m op_tests.op_benchmarks.flydsl.bench_hstu_bwd --backend fused -o bwd.csv
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.hstu_attention import (
    _make_bwd_fused_runners,
    _make_bwd_kernel_runners,
    flydsl_hstu_attention_bwd,
)
from op_tests.test_flydsl_hstu_attention_bwd import (
    SUPPORTED_GFX,
    build_bwd_inputs,
)

# Fixed across every cell.
SEQ_LEN = 16384
HEAD_DIM = 64
DTYPE = torch.bfloat16
# Length-distribution knob, NOT model sparsity: `length = uniform[1, N] * 0.95`,
# which is what reproduces the ticket's token counts (L = 7,885,787 at B=1024
# and 925,531 at B=120). 0.949 or 0.951 miss by ~8,000 tokens.
SPARSITY = 0.95

# (regime, batch, heads, mask, target_size, iters). One pass of `iters` timed
# iterations per arm; the median is reported.
#
# Only the SHAPES come from the ticket -- no reference latency is recorded
# here. A number measured on another machine, at another time, by another
# build is not something this table can honestly compare against; `split` is
# measured right beside `fused` and is the baseline.
#
# Training hstu appears at BOTH target counts because the ticket's two tables
# disagree -- one says targets=10, the other targets=300.
CELLS = [
    ("Training", 1024, 4, "causal", 0, 11),
    ("Training", 1024, 4, "hstu", 300, 11),
    ("Training", 1024, 4, "hstu", 10, 11),
    ("Training", 1024, 8, "causal", 0, 11),
    ("Training", 1024, 8, "hstu", 300, 11),
    ("Training", 1024, 8, "hstu", 10, 11),
    ("Inference", 120, 4, "causal", 0, 21),
    ("Inference", 120, 4, "hstu", 300, 21),
    ("Inference", 120, 8, "causal", 0, 21),
    ("Inference", 120, 8, "hstu", 300, 21),
]

BACKENDS = ("split", "fused")
BASELINE = "split total"

DISPLAY_COLS = [
    "regime",
    "B",
    "H",
    "mask",
    "targets",
    "backend",
    "median_ms",
    "min_ms",
    "TFLOPS",
    "vs split",
    "G",
]

# gfx942 budgets, per SIMD lane-slot and per workgroup.
VGPR_BUDGET = 512
LDS_BUDGET = 65536
ISA_FIELDS = [
    (".vgpr_count", "VGPRs"),
    (".agpr_count", "AGPRs"),
    (".vgpr_spill_count", "VGPR spills"),
    (".sgpr_count", "SGPRs"),
    (".private_segment_fixed_size", "scratch B"),
    (".group_segment_fixed_size", "LDS B"),
]


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #


def time_arm(fn, iters, warmup=2) -> np.ndarray:
    """Per-iteration latencies in ms for one arm, as an array of `iters`.

    Every iteration is bracketed by its own event pair and the whole pass is
    synchronized once at the end, so the launches stay pipelined: a sync per
    iteration would drain the queue each time and bill the drain to a kernel
    that in practice never pays it.

    One pass, not several: on a locked idle GPU the spread between the median
    and the minimum of these iterations is 0.1-1.5%, so repeating the pass
    buys resolution the comparison does not need. The warmup absorbs the
    first-call JIT compile, which is therefore never timed.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])


def bench_cell(regime, b, h, mask, target_size, iters, names):
    """Time the selected backends on one cell. Returns (rows, meta)."""
    alpha = 1.0 / HEAD_DIM * 10000
    q, k, v, seq_offsets, num_targets = build_bwd_inputs(
        b, h, HEAD_DIM, SEQ_LEN, target_size, DTYPE,
        sparsity=SPARSITY, fixed_targets=True,
    )  # fmt: skip
    dout = torch.randn_like(v)
    total_tokens = int(seq_offsets[-1].item())

    args = (
        SEQ_LEN, alpha, q, k, v, dout, seq_offsets, True, num_targets, 0, 0,
    )  # fmt: skip

    # Both arms are built ONCE, outside any timed window, and reuse their
    # buffers across repeats. That is what makes them comparable: the fused
    # arm's fp32 scratch is tens of GB, so re-allocating per iteration would
    # time the caching allocator, and hoisting only one arm's setup (the
    # balance permutation included) would hand that arm the difference.
    arms, meta = {}, {"L": total_tokens}
    if "split" in names:
        split_runners = _make_bwd_kernel_runners(*args)
        arms["dvdk"] = split_runners["dvdk"][0]
        arms["dq"] = split_runners["dq"][0]
        arms["split e2e"] = lambda: flydsl_hstu_attention_bwd(*args, fused=False)
    if "fused" in names:
        runners = _make_bwd_fused_runners(*args)
        arms["fused"] = runners["fused"]
        arms["dq_reduce"] = runners["dq_reduce"]
        meta["G"] = runners["groups"]
        meta["spans"] = runners["spans"]
        meta["reduce_GB"] = runners["reduce_bytes"] / 1e9
        meta["scratch_GB"] = runners["scratch_bytes"] / 1e9
        meta["carry_GB"] = runners["carry_bytes"] / 1e9
        # The PUBLIC entry point, planner and allocation included -- what
        # autograd actually calls. The arms above time launch closures built
        # once outside the loop, which is right for comparing kernels but is
        # not what a caller pays. Keeping both makes the per-call planning cost
        # visible; it was 6.8% and went unnoticed because nothing measured it.
        arms["fused e2e"] = lambda: flydsl_hstu_attention_bwd(*args, fused=True)

    # One pass per arm. `time_arm`'s warmup absorbs the JIT compile, so no
    # separate compile pass is needed to keep it out of the timed window.
    e2e_arms = {k: arms.pop(k) for k in ("split e2e", "fused e2e") if k in arms}
    results = {name: time_arm(fn, iters) for name, fn in arms.items()}

    if e2e_arms:
        # Time the public entry point only AFTER releasing the phase runners.
        # They hold the fp32 scratch alive, so otherwise each public call
        # allocates a SECOND one beside it -- tens of GB of pure measurement
        # artifact, which showed up as 18 ms of phantom overhead against a real
        # 6.8 ms. The warmup inside `time_arm` absorbs the one-time allocation,
        # which is correct: a training loop pays it once, not per step.
        # Drop every reference the phase arms hold. The fused runner owns the
        # fp32 scratch, so otherwise each public call allocates a SECOND one
        # beside it -- tens of GB of pure measurement artifact, which showed up
        # as 18 ms of phantom overhead against a real 6.8 ms.
        arms.clear()
        runners = split_runners = None
        gc.collect()
        torch.cuda.empty_cache()
        for name, fn in e2e_arms.items():
            results[name] = time_arm(fn, iters)

    # Causal pairs per sequence = L*(L+1)/2; bwd = 3*f1 + 2*f2 with
    # f1 = 2*attn_dim (S recompute / dK / dQ share the attn-dim contraction),
    # f2 = 2*hidden_dim (dA = dO*V^T and dV share the hidden-dim contraction).
    lengths = (seq_offsets[1:] - seq_offsets[:-1]).to(torch.float64)
    pairs = float((lengths * (lengths + 1) / 2).sum().item()) * h
    flops = (3 * 2 * HEAD_DIM + 2 * 2 * HEAD_DIM) * pairs

    rows = []
    for name, times in results.items():
        median = float(np.median(times))
        rows.append({
            "regime": regime, "B": b, "H": h, "mask": mask,
            "targets": target_size, "backend": name,
            "median_ms": median, "min_ms": float(times.min()),
            # `flops` counts the WHOLE backward, so dividing it by one phase's
            # time would overstate that phase. Phases get no roofline; the e2e
            # arms are complete backwards, so they do.
            "TFLOPS": (
                flops / median / 1e9 if name.endswith(" e2e") else float("nan")
            ),
            "L": total_tokens,
            "G": meta.get("G", float("nan")),
        })  # fmt: skip

    # Each arm's two phases run back to back and are both exposed, so the whole
    # backward is their sum. Reported as its own row so the table carries the
    # phases and the result a caller actually sees.
    for label, phases in (("split", ("dvdk", "dq")), ("fused", ("fused", "dq_reduce"))):
        if not all(p in results for p in phases):
            continue
        total = sum(float(np.median(results[p])) for p in phases)
        rows.append({
            "regime": regime, "B": b, "H": h, "mask": mask,
            "targets": target_size, "backend": f"{label} total",
            "median_ms": total,
            "min_ms": sum(float(results[p].min()) for p in phases),
            "TFLOPS": flops / total / 1e9,
            "L": total_tokens, "G": meta.get("G", float("nan")),
        })  # fmt: skip
    if "dq_reduce" in results:
        meta["reduce_TBs"] = meta["reduce_GB"] / float(np.median(results["dq_reduce"]))

    # Compare like with like: the `* total` rows are launch closures with setup
    # hoisted out, the `* e2e` rows are the public entry point with setup
    # included. Dividing one family by the other charges the fused path for
    # overhead the baseline never paid.
    base_by_family = {
        " total": next(
            (r["median_ms"] for r in rows if r["backend"] == BASELINE), float("nan")
        ),
        " e2e": next(
            (r["median_ms"] for r in rows if r["backend"] == "split e2e"), float("nan")
        ),
    }
    for r in rows:
        # `dvdk`/`dq` and `fused`/`dq_reduce` are PHASES of one backward, not
        # backwards of their own, so a speedup column on them would invite
        # reading a phase as a complete result. Only the two whole backwards
        # carry it.
        family = next((f for f in base_by_family if r["backend"].endswith(f)), None)
        base = base_by_family.get(family, float("nan"))
        r["vs split"] = base / r["median_ms"] if family else float("nan")

    # q k v dO dq dv dk, all bf16. The fp32 scratch and carry come from the
    # runner, which allocated them.
    meta["tensors_GB"] = 7 * total_tokens * h * HEAD_DIM * 2 / 1e9

    # Each cell holds a full set of activations plus the fp32 scratch; drop them
    # before the next cell builds its own.
    del q, k, v, dout, seq_offsets, num_targets
    gc.collect()
    torch.cuda.empty_cache()
    return rows, meta


def format_footprint(meta):
    """The fused path's device footprint for one cell, as markdown.

    The fp32 partial scratch dominates and is allocated ONCE and reused by every
    pass -- that is what kv-chunking buys. At G=1 it would be G times larger,
    which is what does not fit at the Training shapes.
    """
    _free, total = torch.cuda.mem_get_info()
    parts = {
        "bf16 q/k/v/dO/dQ/dV/dK": meta["tensors_GB"],
        "fp32 partial scratch (per pass, reused)": meta.get("scratch_GB", 0.0),
        "fp32 dQ carry (G > 1 only)": meta.get("carry_GB", 0.0),
    }
    rows = [{"component": name, "GB": f"{gb:.1f}"} for name, gb in parts.items()]
    rows.append({
        "component": f"total (card has {total / 1e9:.0f} GB)",
        "GB": f"{sum(parts.values()):.1f}",
    })  # fmt: skip
    return pd.DataFrame(rows).to_markdown(index=False)


# --------------------------------------------------------------------------- #
# Compiled-kernel resources
#
# The ISA dump needs FLYDSL_DUMP_IR set before FlyDSL is imported, and a warm
# kernel cache emits nothing unless the kernel is actually LAUNCHED. Both are
# reasons to do this in a forked worker rather than in the timed process: the
# env var would otherwise dump IR on every compile of the real run.
# --------------------------------------------------------------------------- #


def _parse_isa(path):
    text = Path(path).read_text()
    out = {}
    for key, _label in ISA_FIELDS:
        m = re.search(
            rf"^\s*(?:-\s*)?{re.escape(key)}:\s*(\d+)\s*$", text, re.MULTILINE
        )
        if m:
            out[key] = int(m.group(1))
    return out


def _isa_worker(num_heads, target_size):
    """Compile and launch the fused path at a TINY batch, then dump the ISA.

    Batch never reaches the compiler, so this costs seconds; `max_seq_len` DOES
    reach it, so it stays at the real value -- compiling at a convenient small
    one would report a different binary from the benchmarked kernel.
    """
    dump_dir = tempfile.mkdtemp(prefix="hstu_isa_")
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = dump_dir

    import torch as _torch  # must follow the env vars above

    _torch.set_default_device("cuda")
    from aiter.ops.flydsl.hstu_attention import (
        flydsl_hstu_attention_bwd as _bwd,
    )
    from op_tests.test_flydsl_hstu_attention_bwd import (
        build_bwd_inputs as _inputs,
    )

    q, k, v, so, nt = _inputs(
        4, num_heads, HEAD_DIM, SEQ_LEN, target_size, DTYPE,
        sparsity=0.02, fixed_targets=True,
    )  # fmt: skip
    _bwd(
        SEQ_LEN, 1.0 / HEAD_DIM * 10000, q, k, v, _torch.randn_like(v), so,
        True, nt, 0, 0, fused=True,
    )  # fmt: skip
    _torch.cuda.synchronize()

    out = {}
    for label, pattern in (("fused", "*fused_pp*"), ("dq_reduce", "*dq_reduce*")):
        hits = sorted(glob.glob(os.path.join(dump_dir, pattern, "22_final_isa.s")))
        out[label] = _parse_isa(hits[-1]) if hits else {}
    print("ISA_JSON " + json.dumps(out), flush=True)
    # A full IR dump is tens of MB per kernel and nothing reads it after this.
    shutil.rmtree(dump_dir, ignore_errors=True)


def kernel_resources(num_heads, target_size):
    """Run `_isa_worker` out-of-process and return its parsed stats."""
    proc = subprocess.run(
        [sys.executable, "-m", __spec__.name, "--isa-worker",
         str(num_heads), str(target_size)],
        cwd=str(Path(__file__).resolve().parents[3]),
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    for line in proc.stdout.splitlines():
        if line.startswith("ISA_JSON "):
            return json.loads(line[len("ISA_JSON ") :])
    aiter.logger.warning("could not collect kernel resources:\n%s", proc.stderr[-2000:])
    return {}


def format_resources(stats, num_waves=8):
    """The resource block for one compile, as markdown."""
    rows = []
    for label, st in stats.items():
        if not st:
            continue
        row = {"kernel": label}
        row.update({name: st.get(key) for key, name in ISA_FIELDS})
        vgpr = st.get(".vgpr_count", 0) + st.get(".agpr_count", 0)
        lds = st.get(".group_segment_fixed_size", 0)
        if label == "fused":
            # 8 waves across 4 SIMDs is 2 per SIMD, so each lane-slot holds two
            # waves' registers out of the 512-register pool.
            per_simd = vgpr * (num_waves // 4)
            row["VGPR+AGPR/SIMD"] = f"{per_simd} of {VGPR_BUDGET}"
            row["LDS/workgroup"] = f"{lds} of {LDS_BUDGET}"
            row["WGs per CU"] = str(
                min(
                    VGPR_BUDGET // max(per_simd, 1),
                    LDS_BUDGET // lds if lds else VGPR_BUDGET,
                )
            )
        rows.append(row)
    if not rows:
        return "(unavailable)"
    # The occupancy columns only apply to the fused kernel's 8-wave workgroup;
    # blank rather than NaN for the reduction, which is a plain 1-D kernel.
    return pd.DataFrame(rows).fillna("").to_markdown(index=False, disable_numparse=True)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _fmt(value, spec):
    """Format, leaving a blank cell where the metric does not apply."""
    return "" if value is None or np.isnan(value) else format(value, spec)


def _suffix(value, spec, suffix):
    formatted = _fmt(value, spec)
    return formatted + suffix if formatted else ""


def format_table(df):
    out = df[DISPLAY_COLS].copy()
    for col in ("median_ms", "min_ms"):
        out[col] = out[col].map(lambda v: _fmt(v, ".1f"))
    out["TFLOPS"] = out["TFLOPS"].map(lambda v: _fmt(v, ".1f"))
    out["vs split"] = out["vs split"].map(lambda v: _suffix(v, ".2f", "x"))
    out["G"] = out["G"].map(lambda v: _fmt(v, ".0f"))
    # `disable_numparse` keeps tabulate from re-parsing these formatted strings
    # and dropping trailing zeros, which otherwise prints 45.0 ms as "45".
    return out.to_markdown(index=False, disable_numparse=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="bench_hstu_bwd",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Benchmark the FlyDSL HSTU backward at the SILOTIGER-1197 "
        "shapes (N=16384, d=64, bf16, causal).",
    )
    parser.add_argument(
        "--backend", nargs="*", default=list(BACKENDS), choices=list(BACKENDS),
        help="""Backends to time.
        e.g.: --backend fused""",
    )  # fmt: skip
    parser.add_argument(
        "--regime", nargs="*", default=["Training", "Inference"],
        choices=["Training", "Inference"],
        help="""Ticket regimes to run. These name B=1024 and B=120; both are
        backward measurements.
        e.g.: --regime Inference""",
    )  # fmt: skip
    parser.add_argument(
        "-H", "--heads", type=int, nargs="*", default=[4, 8],
        help="""Head counts to keep.
        e.g.: -H 4""",
    )  # fmt: skip
    parser.add_argument(
        "--mask", nargs="*", default=["causal", "hstu"],
        choices=["causal", "hstu"],
        help="""Mask presets to keep.""",
    )  # fmt: skip
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="""Also print a table and device footprint per cell. The summary
        table at the end carries the same latencies and G.""",
    )  # fmt: skip
    parser.add_argument(
        "--meta", action="store_true",
        help="""Also print compiled-kernel resources and the device footprint.""",
    )  # fmt: skip
    parser.add_argument(
        "-o", type=str, default=None,
        help="""Also write the full results (incl. L, G, TFLOPS) to this CSV.""",
    )  # fmt: skip
    parser.add_argument("--isa-worker", nargs=2, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.isa_worker:
        _isa_worker(int(args.isa_worker[0]), int(args.isa_worker[1]))
        return

    torch.set_default_device("cuda")
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl hstu attention bwd unsupported on %s; skipping", get_gfx()
        )
        return
    names = [b for b in BACKENDS if b in args.backend]
    if "fused" in names and get_gfx() != "gfx942":
        aiter.logger.warning(
            "hstu bwd fused is gfx942-only; dropping it on %s", get_gfx()
        )
        names = [n for n in names if n != "fused"]
    if not names:
        return

    cells = [
        c for c in CELLS
        if c[0] in args.regime and c[2] in args.heads and c[3] in args.mask
    ]  # fmt: skip

    rows = []
    for index, cell in enumerate(cells, 1):
        regime, b, h, mask, target_size = cell[:5]
        tag = f"{regime} B={b} H={h} {mask} targets={target_size}"
        # Printed BEFORE the cell runs and completed after, so a sweep that
        # spends minutes compiling is not silent. One line per cell, and it
        # reads the same redirected to a file as it does on a terminal.
        print(f"[{index}/{len(cells)}] {tag} ... ", end="", flush=True)

        cell_rows, meta = bench_cell(*cell, names)
        rows += cell_rows

        done = {r["backend"]: r["median_ms"] for r in cell_rows}
        parts = [
            f"{label} {done[key]:.1f}"
            for key, label in (("split total", "split"), ("fused total", "fused"))
            if key in done
        ]
        if "split total" in done and "fused total" in done:
            parts.append(f"{done['split total'] / done['fused total']:.2f}x")
        if "G" in meta:
            parts.append(f"G={meta['G']}")
        print("  ".join(parts), flush=True)
        extra = f"L={meta['L']:,}"
        if "G" in meta:
            extra += f"  G={meta['G']}  spans={meta['spans']}"
        if "reduce_TBs" in meta:
            extra += (
                f"  reduction {meta['reduce_GB']:.0f} GB "
                f"@ {meta['reduce_TBs']:.2f} TB/s"
            )
        if args.verbose:
            aiter.logger.info(
                "hstu bwd %s (%s) (markdown):\n%s",
                tag,
                extra,
                format_table(pd.DataFrame(cell_rows)),
            )
            if args.meta and "scratch_GB" in meta:
                aiter.logger.info(
                    "hstu bwd %s device footprint (markdown):\n%s",
                    tag,
                    format_footprint(meta),
                )

    df = pd.DataFrame(rows)
    aiter.logger.info("hstu bwd, all cells (markdown):\n%s", format_table(df))

    if args.meta:
        heads = min(args.heads)
        target_size = 300 if "hstu" in args.mask else 0
        aiter.logger.info(
            "hstu bwd compiled-kernel resources, H=%d (markdown):\n%s",
            heads,
            format_resources(kernel_resources(heads, target_size)),
        )

    if args.o:
        df.to_csv(args.o, index=False)
        aiter.logger.info("results saved to %s", args.o)


if __name__ == "__main__":
    main()
