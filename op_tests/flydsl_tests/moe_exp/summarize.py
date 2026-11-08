# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Summarize a bench.sh output directory.

Per variant and timed kernel, over the timed processes p1..pN (p0 only compiles):
  kernel_us  rocprofv3 kernel time. A round's value is the mean over its dispatches: those that start inside the
             round's host window (<OUT>/out/<variant>_p<k>.pt.rounds.json; the driver's timestamps share rocprofv3's
             clock). Per process: the median over the rounds, for all rounds and for the second half of the rounds.
             Per variant: the median over the processes.
  event_us   the driver's CUDA-event time per dispatch (median over the rounds), median over the processes
  TB/s       the byte model below / kernel_us over all rounds (event_us when there is no kernel trace)
Also each variant's accuracy line, the kernel names in the trace, and a bit-exact comparison of every saved output
with the first variant's p1 output. Per-round values, medians over the processes, go to
<OUT>/rounds_<variant>_<kernel>.csv.

Byte model, every byte counted once (routed rows R = tokens * topk, all experts active, MXFP4 = 0.5 B per element
plus one E8M0 byte per 32 elements):
  gemm1  A (R x model_dim) + scales, W1 (E x 2*inter x model_dim) + scales,
         output R x inter as MXFP4 + scales (the kernel fuses SiLU and quant)
  gemm2  A2 (R x inter) + scales, W2 (E x model_dim x inter) + scales,
         output R x model_dim bf16 (one row per route, before gather-reduce)
With rotation (meta.txt ``rotation``) every one of these bytes comes from HBM.

usage: python summarize.py <bench.sh OUT directory>
"""

import bisect
import csv
import glob
import json
import os
import re
import statistics
import sys

med = statistics.median
csv.field_size_limit(10**9)


def read(path):
    with open(path) as f:
        return f.read()


def arg(args, name, default=None):
    m = re.search(rf"--{name}\s+(\S+)", args)
    return m.group(1) if m else default


def byte_model(args):
    E = int(arg(args, "experts"))
    T = int(arg(args, "tokens"))
    topk = int(arg(args, "topk"))
    D = int(arg(args, "model-dim"))
    inter = int(arg(args, "inter-dim"))
    R = T * topk

    def fp4(n):  # MXFP4 payload + E8M0 scales for n elements
        return n // 2 + n // 32

    g1 = fp4(R * D) + fp4(E * 2 * inter * D) + fp4(R * inter)
    g2 = fp4(R * inter) + fp4(E * D * inter) + R * D * 2
    return {"gemm1": g1, "gemm2": g2}


def round_kernel_us(trace_csv, rounds):
    """Mean kernel time of the dispatches that start inside each round's host window, and their kernel names."""
    rows = sorted(
        (int(x["Start_Timestamp"]), int(x["End_Timestamp"]), x["Kernel_Name"]) for x in csv.DictReader(open(trace_csv))
    )
    starts = [r[0] for r in rows]
    means, names, counts = [], set(), set()
    for rd in rounds:
        t = rd["t"]
        sel = rows[bisect.bisect_left(starts, t[0]) : bisect.bisect_right(starts, t[4])]
        counts.add(len(sel))
        names.update(r[2] for r in sel)
        means.append(sum((e - b) / 1e3 for b, e, _ in sel) / len(sel) if sel else float("nan"))
    return means, names, counts


def main(d):
    meta = read(os.path.join(d, "meta.txt"))
    args = re.search(r"^args\s+(.*)$", meta, re.MULTILINE).group(1)
    names = [n.split(":")[0] for n in re.search(r"^variants\s+(.*)$", meta, re.MULTILINE).group(1).split()]
    km = re.search(r"^kernels\s+(\S+)", meta, re.MULTILINE).group(1)
    kernels = ("quant_a1", "gemm1", "gemm2") if km == "all" else tuple(km.split(","))
    nbytes = byte_model(args)
    print(meta.rstrip())
    print()
    print("byte model (MB): " + ", ".join(f"{k} {v / 1e6:.1f}" for k, v in nbytes.items() if k in kernels))
    print()
    hdr = (
        f"{'variant':12} {'kernel':8} {'procs':>5} {'kernel_us':>10} {'2nd half':>9} {'event_us':>9} "
        f"{'TB/s':>6} {'vs 1st':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    first = {}
    for n in names:
        procs = sorted(
            (int(m.group(1)), p)
            for p in glob.glob(os.path.join(d, f"{n}_p*.log"))
            if (m := re.search(r"_p(\d+)\.log$", p)) and m.group(1) != "0"
        )
        for k in kernels:
            ev, k_all, k_half, per_round_k, per_round_e, knames, bad = [], [], [], [], [], set(), []
            for i, log in procs:
                m = re.findall(rf"\[moe-exp\] {k}: event_us = ([0-9.]+)", read(log))
                if m:
                    ev.append(float(m[-1]))
                rj = os.path.join(d, "out", f"{n}_p{i}.pt.rounds.json")
                tr = glob.glob(os.path.join(d, f"trace-{n}_p{i}", "**", "*kernel_trace.csv"), recursive=True)
                if not (os.path.exists(rj) and tr):
                    continue
                rk = json.load(open(rj))["kernels"].get(k)
                if not rk:
                    continue
                means, kn, counts = round_kernel_us(tr[0], rk["rounds"])
                knames |= kn
                if counts != {rk["dispatches"]}:
                    bad.append(f"p{i}: dispatches per round {sorted(counts)}, expected {rk['dispatches']}")
                k_all.append(med(means))
                k_half.append(med(means[len(means) // 2 :]))
                per_round_k.append(means)
                per_round_e.append([r["event_us"] for r in rk["rounds"]])
            if not ev:
                print(f"{n:12} {k:8} n/a: no event_us line in any timed process")
                continue
            ku = med(k_all) if k_all else None
            base = ku if ku is not None else med(ev)
            tbs = f"{nbytes[k] / (base * 1e-6) / 1e12:6.2f}" if k in nbytes else "     -"
            first.setdefault(k, base)
            rel = (base / first[k] - 1) * 100
            fmt = lambda x: f"{x:.2f}" if x is not None else "-"
            print(
                f"{n:12} {k:8} {len(procs):5} {fmt(ku):>10} {fmt(med(k_half) if k_half else None):>9} "
                f"{med(ev):9.2f} {tbs} {rel:+6.1f}%"
            )
            for b in bad:
                print(f"    warning {b}")
            if per_round_k:
                with open(os.path.join(d, f"rounds_{n}_{k}.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["round", "kernel_us", "event_us"])
                    for j in range(min(len(x) for x in per_round_k)):
                        w.writerow([j + 1, f"{med(x[j] for x in per_round_k):.3f}", f"{med(x[j] for x in per_round_e):.3f}"])
            if knames:
                for kn in sorted(knames):
                    print(f"    {kn}")
    print()
    for n in names:
        acc = set()
        for p in glob.glob(os.path.join(d, f"{n}_p*.log")):
            acc.update(re.findall(r"kernel: (logits_diff=\S+ rel_l2=\S+)", read(p)))
        print(f"{n}: accuracy {' | '.join(sorted(acc)) or 'n/a'}")
    outs = sorted(glob.glob(os.path.join(d, "out", "*.pt")))
    ref_path = os.path.join(d, "out", f"{names[0]}_p1.pt")
    if outs and os.path.exists(ref_path):
        import torch

        ref = torch.load(ref_path)
        print(f"\noutputs vs {os.path.basename(ref_path)}:")
        for p in outs:
            o = torch.load(p)
            same = torch.equal(o, ref)
            nan = int(torch.isnan(o.float()).sum())
            print(f"    {os.path.basename(p):24} bit-identical={same} NaN={nan}")


if __name__ == "__main__":
    main(sys.argv[1])
