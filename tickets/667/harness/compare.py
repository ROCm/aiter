#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""G9 FlyDSL-vs-CK warp-decode COLD benchmark comparison (SILOTIGER-667).

Drives both harnesses over the same shapes/dtypes/batches and emits a joined
FlyDSL/CK ratio table.  Design (see SILOTIGER-667-plan-bench.md):

  * Compare on TIME (us) -- the one method-independent quantity.  The headline
    metric is the unit-free ratio ``flydsl_us / ck_us``.
  * Derived metrics (TB/s, TFLOPS, %peak) are recomputed from the SHARED
    ``compute_metrics`` helper in the op_test module, applied identically to both
    sides' raw times (``--method weight_stream`` default, or ``total_traffic``).
  * Both sides COLD: CK via ``CK_WD_ROTATE`` disjoint-expert rotation (A1);
    FlyDSL via its cold-HBM benches.  Default-vs-default config (D3).
  * D5 variance: clocks can't be pinned on this gfx950 (see plan D1), so each sweep
    is repeated ``--repeats`` times; the headline us is the per-cell median, with a
    spread%% column per side and a "noisy" flag when spread exceeds ``--noise-pct``.
    The effective SCLK is sampled during the run and recorded in the header.

Env / prerequisites:
  * Run under flydsl_venv on the isolated GPU, e.g.
      HIP_VISIBLE_DEVICES=6 ./flydsl_venv/bin/python tickets/667/harness/compare.py
  * CK binary path via ``CK_BENCH`` (exported by build_ck_bench.sh) or ``--ck-bench``.

Examples:
  # full cold sweep, of-record iters (D1) + variance (D5), markdown + csv artifacts
  HIP_VISIBLE_DEVICES=6 ./flydsl_venv/bin/python tickets/667/harness/compare.py \
      --iters 1000 --cold 20 --repeats 3 --md-out tickets/667/g9_compare.md \
      --csv-out tickets/667/g9_compare.csv
  # quick smoke (one shape, tiny iters, single pass)
  HIP_VISIBLE_DEVICES=6 ./flydsl_venv/bin/python tickets/667/harness/compare.py \
      --shapes qwen3next --batches 1 --iters 30 --cold 5 --repeats 1
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import statistics
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # /workspaces/aiter
OPTEST = REPO / "op_tests" / "flydsl_tests" / "test_flydsl_warp_decode_moe.py"
CK_BENCH_DEFAULT = "/workspaces/rocm-libraries-wdec/bench_ck_warp_decode"

# Shape set (dims are the join key; CK shape names match these keys).
#   H=hidden, I=inter, E=num_experts, K=top_k
SHAPES = {
    "deepseek-v3": dict(H=7168, I=2048, E=256, K=8),
    "minimax": dict(H=3072, I=1536, E=256, K=8),
    "qwen3next": dict(H=2048, I=512, E=512, K=10),
}

# Canonical CK kernel -> (op, w_dtype, act_dtype, recommended).  ``recommended``
# marks the maintainer-intended default variant used for the headline join (D3);
# the non-recommended CK rows (e.g. non-dot2) are kept only as info.
CK_MAP = {
    "gate_up_bf16": ("gate_up", "fp8", "bf16", False),  # non-dot2 default
    "gate_bf16_d2": ("gate_up", "fp8", "bf16", True),  # <- gate_up FP8 (bf16-act) peer
    "gate_up_fp8": ("gate_up", "fp8", "fp8", False),
    "gate_fp8_d2": ("gate_up", "fp8", "fp8", True),  # <- FP8-act peer (needs FlyDSL B4)
    "down_h2_d2": ("down", "fp8", None, True),
    "down_fp4_h2": ("down", "fp4", None, True),
    "gate_up_fp4": (
        "gate_up",
        "fp4",
        "bf16",
        True,
    ),  # <- gate_up FP4 peer (needs CK A4)
}

# FlyDSL cold benches produce these (op, w_dtype, act_dtype) cells today.
FLYDSL_CELLS = [
    ("down", "fp4", None),
    ("down", "fp8", None),
    ("gate_up", "fp4", "bf16"),
    ("gate_up", "fp8", "bf16"),
]


def _key(H, I, E, K, B, op, wd, act):  # noqa: E741
    return (int(H), int(I), int(E), int(K), int(B), op, wd, act)


# --------------------------------------------------------------------------
# D5: run-to-run variance + effective-clock sampling
# --------------------------------------------------------------------------
# Clocks can't be pinned on this gfx950 (only discrete {500,2400} SCLK levels,
# MCLK fixed 2000; see plan D1), so we control drift empirically instead: repeat
# each sweep N times, report per-cell spread, flag noisy cells, and sample the
# effective SCLK during the run into provenance.
_SCLK_RE = re.compile(r"sclk clock level:\s*\S+\s*\(?(\d+)\s*Mhz", re.IGNORECASE)


class ClockSampler:
    """Background poller of `rocm-smi -d <gpu> --showgpuclocks` SCLK (MHz).

    Best-effort: if rocm-smi is missing/unparseable it just collects nothing.
    """

    def __init__(self, gpu, period_s=0.25):
        self.gpu = gpu
        self.period_s = period_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll_once(self):
        try:
            out = subprocess.run(
                ["rocm-smi", "-d", str(self.gpu), "--showgpuclocks"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        except Exception:
            return
        m = _SCLK_RE.search(out)
        if m:
            self.samples.append(int(m.group(1)))

    def _loop(self):
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self.period_s)

    def __enter__(self):
        if self.gpu is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self):
        s = self.samples
        if not s:
            return "sclk: (not sampled)"
        # Drop obvious-idle samples (< 400 MHz) so the summary reflects the clock
        # while kernels run, not startup/gap idle; fall back to raw if all idle.
        loaded = [x for x in s if x >= 400] or s
        tag = "loaded sclk" if loaded is not s else "sclk"
        return (
            f"{tag} MHz min/median/max = "
            f"{min(loaded)}/{int(statistics.median(loaded))}/{max(loaded)} "
            f"(n={len(loaded)}/{len(s)})"
        )


def _agg(vals):
    """(median_us, spread_pct) from a list of per-repeat times; NaN-safe.

    spread_pct = 100*(max-min)/median -- a robust run-to-run drift indicator.
    """
    xs = [
        v for v in vals if isinstance(v, (int, float)) and not math.isnan(v) and v > 0
    ]
    if not xs:
        return float("nan"), float("nan")
    med = statistics.median(xs)
    spr = 100.0 * (max(xs) - min(xs)) / med if len(xs) > 1 and med > 0 else 0.0
    return med, spr


# --------------------------------------------------------------------------
# CK side
# --------------------------------------------------------------------------
def run_ck(shapes, batches, iters, cold, ck_bench):
    """Run the CK bench in CSV mode; return (records, provenance_str).

    records: {key -> us} for the *recommended* CK kernel of each cell.
    """
    env = os.environ.copy()
    env["CK_WD_FORMAT"] = "csv"
    env["CK_WD_SHAPES"] = ",".join(shapes)
    env["CK_WD_BATCHES"] = ",".join(str(b) for b in batches)
    env["CK_WD_ITERS"] = str(iters)
    env["CK_WD_COLD"] = str(cold)
    proc = subprocess.run(
        [ck_bench], env=env, capture_output=True, text=True, check=True
    )
    records = {}
    info = {}  # non-recommended peers, for reference
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("shape,"):
            continue
        parts = line.split(",")
        if len(parts) != 8:
            continue
        _name, H, I, K, E, B, kernel, us = parts  # noqa: E741
        if kernel not in CK_MAP:
            continue
        op, wd, act, recommended = CK_MAP[kernel]
        key = _key(H, I, E, K, B, op, wd, act)
        (records if recommended else info)[key] = float(us)
    return records, proc.stderr.strip()


# --------------------------------------------------------------------------
# FlyDSL side
# --------------------------------------------------------------------------
def load_flydsl_module():
    spec = importlib.util.spec_from_file_location("wd_optest", OPTEST)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_flydsl(mod, shapes, batches, iters, warmup, timing):
    """Call the cold benches and melt merged rows into per-cell {key -> (us, cos)}."""
    records = {}
    for name in shapes:
        s = SHAPES[name]
        H, I, E, K = s["H"], s["I"], s["E"], s["K"]  # noqa: E741
        for B in batches:
            dn = mod.bench_down_cold(
                B, I, H, E, K, timing=timing, num_iters=iters, num_warmup=warmup
            )
            records[_key(H, I, E, K, B, "down", "fp4", None)] = (
                dn.get("fp4_us"),
                dn.get("fp4_cos"),
            )
            records[_key(H, I, E, K, B, "down", "fp8", None)] = (
                dn.get("fp8_us"),
                dn.get("fp8_cos"),
            )
            gu = mod.bench_gate_up_cold(
                B, H, I, E, K, timing=timing, num_iters=iters, num_warmup=warmup
            )
            records[_key(H, I, E, K, B, "gate_up", "fp4", "bf16")] = (
                gu.get("fp4_us"),
                gu.get("fp4_cos"),
            )
            records[_key(H, I, E, K, B, "gate_up", "fp8", "bf16")] = (
                gu.get("fp8_us"),
                gu.get("fp8_cos"),
            )
    return records


def run_ck_repeats(shapes, batches, iters, cold, ck_bench, repeats):
    """Run the CK sweep `repeats` times; return ({key -> [us,...]}, provenance)."""
    agg: dict = {}
    prov = "(CK skipped)"
    for r in range(repeats):
        records, prov = run_ck(shapes, batches, iters, cold, ck_bench)
        for k, us in records.items():
            agg.setdefault(k, []).append(us)
        print(f"[compare] CK repeat {r + 1}/{repeats} done", file=sys.stderr)
    return agg, prov


def run_flydsl_repeats(mod, shapes, batches, iters, warmup, timing, repeats):
    """Run the FlyDSL sweep `repeats` times; return {key -> ([us,...], cos)}."""
    agg: dict = {}
    for r in range(repeats):
        records = run_flydsl(mod, shapes, batches, iters, warmup, timing)
        for k, (us, cos) in records.items():
            lst, _ = agg.setdefault(k, ([], cos))
            lst.append(us)
        print(f"[compare] FlyDSL repeat {r + 1}/{repeats} done", file=sys.stderr)
    return agg


# --------------------------------------------------------------------------
# Join + emit
# --------------------------------------------------------------------------
def _isnum(x):
    return isinstance(x, (int, float)) and not (x is None or math.isnan(x))


def _fmt(x, prec=2):
    return f"{x:.{prec}f}" if _isnum(x) else "n/a"


def build_rows(mod, fly, ck, shapes, batches, method, noise_pct):
    """Produce the joined comparison rows (list of dicts).

    `fly` is {key -> ([us,...], cos)} and `ck` is {key -> [us,...]} (D5 repeats);
    the headline us is the per-cell median, with spread% and a noisy flag.
    """
    rows = []
    for name in shapes:
        s = SHAPES[name]
        H, I, E, K = s["H"], s["I"], s["E"], s["K"]  # noqa: E741
        for B in batches:
            for op, wd, act in FLYDSL_CELLS:
                key = _key(H, I, E, K, B, op, wd, act)
                fus_list, fcos = fly.get(key, ([], None))
                fus, fspr = _agg(fus_list)
                cus, cspr = _agg(ck.get(key, []))
                fm = mod.compute_metrics(
                    op,
                    B,
                    H,
                    I,
                    K,
                    wd,
                    fus,
                    method=method,
                    act_dtype=(act or "bf16"),
                )
                cm = mod.compute_metrics(
                    op,
                    B,
                    H,
                    I,
                    K,
                    wd,
                    cus,
                    method=method,
                    act_dtype=(act or "bf16"),
                )
                ratio = (
                    fus / cus
                    if _isnum(fus) and _isnum(cus) and cus > 0
                    else float("nan")
                )
                notes = []
                if not _isnum(fus):
                    notes.append("FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5)")
                elif not _isnum(cus):
                    if op == "gate_up" and wd == "fp4":
                        notes.append("CK n/a (gate_up FP4 -> A4)")
                    else:
                        notes.append("CK n/a")
                noisy = (_isnum(fspr) and fspr > noise_pct) or (
                    _isnum(cspr) and cspr > noise_pct
                )
                if noisy:
                    notes.append(f"noisy (>{noise_pct:.0f}% spread)")
                rows.append(
                    dict(
                        shape=name,
                        B=B,
                        op=op,
                        wdtype=wd,
                        act=(act or "-"),
                        flydsl_us=fus,
                        ck_us=cus,
                        ratio=ratio,
                        fly_tbs=fm["TB/s"],
                        ck_tbs=cm["TB/s"],
                        fly_peak=fm["%peak"],
                        fly_spr=fspr,
                        ck_spr=cspr,
                        cos=fcos,
                        note="; ".join(notes),
                    )
                )
    return rows


def to_markdown(rows, method, header_lines):
    cols = [
        ("shape", "shape", 0),
        ("B", "B", 0),
        ("op", "op", 0),
        ("dtype", "wdtype", 0),
        ("act", "act", 0),
        ("flydsl_us", "flydsl_us", 4),
        ("ck_us", "ck_us", 4),
        ("ratio(f/c)", "ratio", 3),
        ("fly_TB/s", "fly_tbs", 1),
        ("ck_TB/s", "ck_tbs", 1),
        ("fly_%peak", "fly_peak", 1),
        ("fly_spr%", "fly_spr", 1),
        ("ck_spr%", "ck_spr", 1),
        ("cos", "cos", 4),
        ("note", "note", None),
    ]
    out = [f"<!-- {ln} -->" for ln in header_lines]
    out.append(
        f"\n**metric method:** `{method}` &nbsp; (ratio = flydsl_us / ck_us; "
        "CK is perf-only / uninitialized weights)\n"
    )
    out.append("| " + " | ".join(h for h, _, _ in cols) + " |")
    out.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        cells = []
        for _h, k, prec in cols:
            v = r[k]
            if prec is None:
                cells.append(str(v))
            elif isinstance(v, str):
                cells.append(v)
            else:
                cells.append(_fmt(v, prec))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def to_csv(rows):
    fields = [
        "shape",
        "B",
        "op",
        "wdtype",
        "act",
        "flydsl_us",
        "ck_us",
        "ratio",
        "fly_tbs",
        "ck_tbs",
        "fly_peak",
        "fly_spr",
        "ck_spr",
        "cos",
        "note",
    ]
    lines = [",".join(fields)]
    for r in rows:
        vals = []
        for f in fields:
            v = r[f]
            vals.append(
                ""
                if v is None
                else (
                    f"{v}"
                    if not isinstance(v, float)
                    else ("" if math.isnan(v) else f"{v:.6g}")
                )
            )
        lines.append(",".join(vals))
    return "\n".join(lines) + "\n"


def _git_commit(path):
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--shapes",
        default=",".join(SHAPES),
        help="comma list from %s" % ",".join(SHAPES),
    )
    ap.add_argument("--batches", default="1,2,4,8,32")
    ap.add_argument(
        "--iters", type=int, default=1000, help="timed iters (D1: 1000 of-record)"
    )
    ap.add_argument("--cold", type=int, default=20, help="warmup iters (D1: >=15)")
    ap.add_argument("--warmup", type=int, default=20, help="FlyDSL num_warmup")
    ap.add_argument(
        "--timing", default="device", choices=["device", "cuda_event", "graph"]
    )
    ap.add_argument(
        "--method", default="weight_stream", choices=["weight_stream", "total_traffic"]
    )
    ap.add_argument("--ck-bench", default=os.environ.get("CK_BENCH", CK_BENCH_DEFAULT))
    ap.add_argument("--no-ck", action="store_true", help="skip CK (FlyDSL-only)")
    ap.add_argument("--no-flydsl", action="store_true", help="skip FlyDSL (CK-only)")
    ap.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="D5: repeat each sweep N times; report per-cell median + spread",
    )
    ap.add_argument(
        "--noise-pct",
        type=float,
        default=5.0,
        help="D5: flag a cell noisy if either side's spread%% exceeds this",
    )
    ap.add_argument("--md-out", default=None)
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()

    if args.repeats < 1:
        ap.error("--repeats must be >= 1")

    shapes = [s for s in args.shapes.split(",") if s]
    unknown = [s for s in shapes if s not in SHAPES]
    if unknown:
        ap.error(f"unknown shapes {unknown}; choose from {list(SHAPES)}")
    batches = [int(b) for b in args.batches.split(",") if b]

    gpu = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    gpu = gpu.split(",")[0].strip() if gpu else None

    ck_records, ck_prov = {}, "(CK skipped)"
    mod = load_flydsl_module()
    fly_records = {}
    with ClockSampler(gpu) as clk:
        if not args.no_ck:
            if not Path(args.ck_bench).exists():
                ap.error(
                    f"CK binary not found: {args.ck_bench} (build_ck_bench.sh / --ck-bench)"
                )
            ck_records, ck_prov = run_ck_repeats(
                shapes, batches, args.iters, args.cold, args.ck_bench, args.repeats
            )
        if not args.no_flydsl:
            fly_records = run_flydsl_repeats(
                mod, shapes, batches, args.iters, args.warmup, args.timing, args.repeats
            )
    clk_summary = clk.summary()

    rows = build_rows(
        mod, fly_records, ck_records, shapes, batches, args.method, args.noise_pct
    )

    header_lines = [
        "SILOTIGER-667 G9 FlyDSL-vs-CK cold warp-decode comparison",
        f"gfx={mod.get_gfx()}  aiter={_git_commit(REPO)}  "
        f"ck_worktree={_git_commit(Path(args.ck_bench).parent)}",
        f"iters={args.iters} cold={args.cold} timing={args.timing} method={args.method} "
        f"repeats={args.repeats}",
        f"CK provenance: {ck_prov}",
        f"clocks: auto (unpinnable on this gfx950; D1) -- effective {clk_summary} on GPU "
        f"{gpu}; per-cell spread%% + noisy flag (>{args.noise_pct:.0f}%) capture drift (D5).",
        "config policy: default-vs-default (D3); "
        "treat under-converged fast cells as noisy (D1).",
    ]
    md = to_markdown(rows, args.method, header_lines)
    print(md)

    if args.md_out:
        Path(args.md_out).write_text(md)
        print(f"[compare] wrote {args.md_out}", file=sys.stderr)
    if args.csv_out:
        Path(args.csv_out).write_text(to_csv(rows))
        print(f"[compare] wrote {args.csv_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
