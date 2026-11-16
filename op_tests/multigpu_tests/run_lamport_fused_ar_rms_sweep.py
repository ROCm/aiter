# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Full sweep of test_lamport_fused_ar_rms.py, several runs, one summary to
paste back for the dispatch (POC, temporary).

    python3 op_tests/multigpu_tests/run_lamport_fused_ar_rms_sweep.py
    python3 op_tests/multigpu_tests/run_lamport_fused_ar_rms_sweep.py --runs 1 -- -t 8
    python3 op_tests/multigpu_tests/run_lamport_fused_ar_rms_sweep.py --logs a.log b.log

Run from the aiter repo root on an idle 8-GPU node. It removes the
module_custom_all_reduce JIT build first (the .cuh changes are not picked up
otherwise), runs the test --runs times (default 3, ~15 min each), and prints:
  1. environment (GPU arch, ROCm, commit)
  2. correctness: every lamport err column, worst over the runs (0 expected)
  3. every candidate's time, median over the runs, per TP
  4. per shape: fastest fused variant vs lamport_ar+norm and car_fused
The same text goes to <out>/summary.md, the raw logs to <out>/run<i>.log.
Arguments after "--" go to the test as they are (e.g. -t 8 -n 7168).
"""

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import time

TEST = "op_tests/multigpu_tests/test_lamport_fused_ar_rms.py"
JIT_SO = "aiter/jit/module_custom_all_reduce.so"
JIT_BUILD = "aiter/jit/build/module_custom_all_reduce"
# baselines in the test, everything else ending in " us" is a fused lamport variant
BASELINES = ["car_fused", "car_1stage", "lamport_ar+norm"]
ONE_SHOT = ["flat_row", "flat_bs64", "flat_bs128", "flat_bs256", "ws_g1", "ws_g2", "ws_g4", "ws_g8"]
TWO_SHOT = ["ts_s1", "ts_s2", "ts_s4"]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # environment info only
        return f"? ({e})"


def environment():
    lines = [
        "gfx: " + sh("rocminfo | grep -m1 -o 'gfx[0-9a-f]*'"),
        "gpu: " + sh("rocm-smi --showproductname | grep -m1 'Card SKU\\|Card series' | sed 's/.*: //'"),
        "rocm: " + sh("cat /opt/rocm/.info/version"),
        "hipcc: " + sh("hipcc --version | grep -m1 -i 'clang version'"),
        "commit: " + sh("git rev-parse --short HEAD") + " (" + sh("git rev-parse --abbrev-ref HEAD") + ")",
        "dirty files: " + (sh("git status --porcelain --untracked-files=no | wc -l") or "?"),
        "date: " + time.strftime("%Y-%m-%d %H:%M:%S"),
    ]
    return lines


def readTable(path):
    """The markdown summary the test logs, as {(tp, m, n): {column: str}}."""
    text = open(path, errors="replace").read()
    if "summary (markdown):" not in text:
        return None
    block = text.split("summary (markdown):", 1)[1].splitlines()
    rows, header = {}, None
    for line in block:
        if not line.startswith("|"):
            if header is not None:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if set(line.replace("|", "").strip()) <= set("-: "):
            continue
        d = dict(zip(header, cells))
        rows[(int(float(d["tp"])), int(float(d["m"])), int(float(d["n"])))] = d
    return rows


def num(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def summarize(tables):
    out = []
    keys = sorted({k for t in tables for k in t}, key=lambda k: (k[0], k[2], k[1]))
    cols = []
    for t in tables:
        for d in t.values():
            for c in d:
                if c not in cols:
                    cols.append(c)
    us_cols = [c for c in cols if c.endswith(" us")]
    err_cols = [c for c in cols if c.endswith(" err")]

    def med(k, c):
        xs = [num(t[k].get(c)) for t in tables if k in t]
        xs = [x for x in xs if x is not None]
        return statistics.median(xs) if xs else None

    def worst(k, c):
        xs = [num(t[k].get(c)) for t in tables if k in t]
        xs = [x for x in xs if x is not None]
        return max(xs) if xs else None

    def kb(k):
        return k[1] * k[2] * 2 // 1024

    # 2. correctness
    out.append("## 2. correctness (fraction of mismatching elements, worst over runs; lamport variants must be 0)\n")
    bad = []
    for k in keys:
        for c in err_cols:
            name = c[: -len(" err")]
            if name in BASELINES:
                continue
            v = worst(k, c)
            if v:
                bad.append(f"- tp{k[0]} m={k[1]} n={k[2]} {name}: {v:g}")
    out += bad if bad else ["all lamport variants bit exact in every run"]
    out.append("")

    # 3. every candidate
    out.append("## 3. every candidate, us (median over runs; — = shape not supported by the variant)\n")
    for tp in sorted({k[0] for k in keys}):
        names = [c[: -len(" us")] for c in us_cols]
        out.append(f"### TP{tp}\n")
        out.append("| M | hidden | KB | " + " | ".join(names) + " |")
        out.append("|" + "---|" * (3 + len(names)))
        for k in keys:
            if k[0] != tp:
                continue
            vals = [med(k, c) for c in us_cols]
            out.append(f"| {k[1]} | {k[2]} | {kb(k)} | " +
                       " | ".join("—" if v is None else f"{v:.2f}" for v in vals) + " |")
        out.append("")

    # 4. fastest fused vs the baselines
    out.append("## 4. fastest fused variant vs baselines (ratio = baseline us / fused us, > 1 fused faster)\n")
    for tp in sorted({k[0] for k in keys}):
        out.append(f"### TP{tp}\n")
        out.append("| M | hidden | KB | lamport_ar+norm | car_fused | best one-shot fused | best two-shot fused "
                   "| best fused vs ar+norm | vs car_fused |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for k in keys:
            if k[0] != tp:
                continue
            arn, car = med(k, "lamport_ar+norm us"), med(k, "car_fused us")
            o = [(med(k, c + " us"), c) for c in ONE_SHOT]
            t = [(med(k, c + " us"), c) for c in TWO_SHOT]
            o = min([x for x in o if x[0] is not None], default=None)
            t = min([x for x in t if x[0] is not None], default=None)
            fb = min([x for x in (o, t) if x is not None], default=None)
            f = lambda x: f"{x[0]:.2f} ({x[1]})" if x else "—"
            r = lambda base: f"{base / fb[0]:.2f}" if base and fb else "—"
            fmt = lambda v: "—" if v is None else f"{v:.2f}"
            out.append(f"| {k[1]} | {k[2]} | {kb(k)} | {fmt(arn)} | {fmt(car)} | {f(o)} | {f(t)} | {r(arn)} | {r(car)} |")
        out.append("")
    return out


def main():
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=3, help="full test runs, median over them (default 3)")
    ap.add_argument("--out", default="lamport_ar_rms_sweep", help="directory for logs and summary.md")
    ap.add_argument("--no-rebuild", action="store_true", help="keep the existing JIT build")
    ap.add_argument("--logs", nargs="+", help="only summarize these existing logs, run nothing")
    args = ap.parse_args(argv)

    if not os.path.exists(TEST):
        sys.exit(f"run from the aiter repo root ({TEST} not found)")
    os.makedirs(args.out, exist_ok=True)
    report = ["# lamport fused AR + RMSNorm sweep\n", "## 1. environment\n"]
    report += [f"- {l}" for l in environment()]

    logs = args.logs
    if not logs:
        if not args.no_rebuild:
            if os.path.exists(JIT_SO):
                os.remove(JIT_SO)
            shutil.rmtree(JIT_BUILD, ignore_errors=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.getcwd() + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        logs = []
        for r in range(1, args.runs + 1):
            log = os.path.join(args.out, f"run{r}.log")
            t0 = time.time()
            print(f"run {r}/{args.runs}: python3 {TEST} {' '.join(passthrough)} > {log}", flush=True)
            with open(log, "w") as f:
                rc = subprocess.run([sys.executable, TEST] + passthrough, stdout=f,
                                    stderr=subprocess.STDOUT, env=env).returncode
            msg = f"run {r}: rc={rc}, {time.time() - t0:.0f}s"
            if rc != 0:
                msg += " (rc=1 with a table = some lamport variant not bit exact, see section 2; no table = crash, see the log)"
            print(msg, flush=True)
            report.append(f"- {msg}")
            logs.append(log)
    report.append(f"- test args: {' '.join(passthrough) or '(defaults: TP 2/4/8, hidden 4096/7168/8192)'}")
    report.append("")

    tables = []
    for log in logs:
        t = readTable(log)
        if t is None:
            report.append(f"**{log}: no summary table (crash or hang?), last lines:**\n```")
            report += open(log, errors="replace").read().splitlines()[-15:]
            report.append("```\n")
        else:
            tables.append(t)
    if tables:
        report += summarize(tables)
    text = "\n".join(report)
    print("\n" + text)
    path = os.path.join(args.out, "summary.md")
    open(path, "w").write(text + "\n")
    print(f"\nsummary written to {path}: paste that file back")


if __name__ == "__main__":
    main()
