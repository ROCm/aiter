# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Full sweep of test_lamport_all_reduce_add.py, several runs, one summary to
paste back for the dispatch (POC, temporary).

    python3 op_tests/multigpu_tests/run_lamport_all_reduce_add_sweep.py
    python3 op_tests/multigpu_tests/run_lamport_all_reduce_add_sweep.py --runs 1 -- -t 8
    python3 op_tests/multigpu_tests/run_lamport_all_reduce_add_sweep.py --logs a.log b.log

Run from the aiter repo root on an idle 8-GPU node. It removes the
module_custom_all_reduce JIT build first (the .cuh changes are not picked up
otherwise), runs the test --runs times (default 3), and prints:
  1. environment (GPU arch, ROCm, commit)
  2. correctness: every lamport err column and the mixed check, worst over
     the runs (0 expected)
  3. every candidate's time, median over the runs, per TP
  4. per shape: fastest fused kernel vs prod_route (x + y, then all_reduce)
     and vs x + y then the same lamport kernel
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

TEST = "op_tests/multigpu_tests/test_lamport_all_reduce_add.py"
JIT_SO = "aiter/jit/module_custom_all_reduce.so"
JIT_BUILD = "aiter/jit/build/module_custom_all_reduce"
FUSED = ["fused_1shot", "fused_2shot"]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # environment info only
        return f"? ({e})"


def environment():
    return [
        "gfx: " + sh("rocminfo | grep -m1 -o 'gfx[0-9a-f]*'"),
        "gpu: " + sh("rocm-smi --showproductname | grep -m1 'Card SKU\\|Card series' | sed 's/.*: //'"),
        "rocm: " + sh("cat /opt/rocm/.info/version"),
        "hipcc: " + sh("hipcc --version | grep -m1 -i 'clang version'"),
        "commit: " + sh("git rev-parse --short HEAD") + " (" + sh("git rev-parse --abbrev-ref HEAD") + ")",
        "dirty files: " + (sh("git status --porcelain --untracked-files=no | wc -l") or "?"),
        "date: " + time.strftime("%Y-%m-%d %H:%M:%S"),
    ]


def readTable(path):
    """The markdown summary the test logs, as {(tp, m, n): {column: str}}."""
    text = open(path, errors="replace").read()
    if "summary (markdown):" not in text:
        return None
    rows, header = {}, None
    for line in text.split("summary (markdown):", 1)[1].splitlines():
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

    def vals(k, c):
        xs = [num(t[k].get(c)) for t in tables if k in t]
        return [x for x in xs if x is not None]

    def med(k, c):
        xs = vals(k, c)
        return statistics.median(xs) if xs else None

    def kb(k):
        return k[1] * k[2] * 2 // 1024

    fmt = lambda v: "—" if v is None else f"{v:.2f}"

    out.append("## 2. correctness (fraction of mismatching elements, worst over runs; lamport candidates and mixed must be 0)\n")
    bad = []
    for k in keys:
        for c in err_cols:
            xs = vals(k, c)
            if xs and max(xs) and c != "prod_route err":
                bad.append(f"- tp{k[0]} m={k[1]} n={k[2]} {c}: {max(xs):g}")
        xs = vals(k, "prod_route err")
        if xs and max(xs):
            bad.append(f"- (checkAllclose, informational) tp{k[0]} m={k[1]} n={k[2]} prod_route err: {max(xs):g}")
    out += bad if bad else ["all lamport candidates bit exact in every run, mixed check clean"]
    out.append("")

    out.append("## 3. every candidate, us (median over runs; — = size outside the candidate's range)\n")
    names = [c[: -len(" us")] for c in us_cols]
    for tp in sorted({k[0] for k in keys}):
        out.append(f"### TP{tp}\n")
        out.append("| M | hidden | KB | " + " | ".join(names) + " |")
        out.append("|" + "---|" * (3 + len(names)))
        for k in keys:
            if k[0] == tp:
                out.append(f"| {k[1]} | {k[2]} | {kb(k)} | " + " | ".join(fmt(med(k, c)) for c in us_cols) + " |")
        out.append("")

    out.append("## 4. fastest fused kernel vs prod_route and vs add + the same lamport kernel "
               "(ratio = other us / fused us, > 1 fused faster)\n")
    for tp in sorted({k[0] for k in keys}):
        out.append(f"### TP{tp}\n")
        out.append("| M | hidden | KB | prod_route | fused_auto | best fused | vs prod_route "
                   "| add+same kernel | vs add+same kernel |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for k in keys:
            if k[0] != tp:
                continue
            prod = med(k, "prod_route us")
            fs = [(med(k, f + " us"), f) for f in FUSED]
            fs = [x for x in fs if x[0] is not None]
            best = min(fs) if fs else None
            same = med(k, "add+lamport_" + best[1].split("_", 1)[1] + " us") if best else None
            r = lambda base: f"{base / best[0]:.2f}" if base and best else "—"
            out.append(
                f"| {k[1]} | {k[2]} | {kb(k)} | {fmt(prod)} | {fmt(med(k, 'fused_auto us'))} | "
                f"{(f'{best[0]:.2f} ({best[1]})' if best else '—')} | {r(prod)} | {fmt(same)} | {r(same)} |"
            )
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
    ap.add_argument("--out", default="lamport_ar_add_sweep", help="directory for logs and summary.md")
    ap.add_argument("--no-rebuild", action="store_true", help="keep the existing JIT build")
    ap.add_argument("--logs", nargs="+", help="only summarize these existing logs, run nothing")
    args = ap.parse_args(argv)

    if not os.path.exists(TEST):
        sys.exit(f"run from the aiter repo root ({TEST} not found)")
    os.makedirs(args.out, exist_ok=True)
    report = ["# lamport all-reduce + add sweep\n", "## 1. environment\n"]
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
                msg += " (rc=1 with a table = a lamport candidate not bit exact, see section 2; no table = crash, see the log)"
            print(msg, flush=True)
            report.append(f"- {msg}")
            logs.append(log)
    report.append(f"- test args: {' '.join(passthrough) or '(defaults: TP 2/4/8, hidden 4096/5120/7168, up to 2MB)'}")
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
