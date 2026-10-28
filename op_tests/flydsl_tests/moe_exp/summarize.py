# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Summarize a bench.sh output directory.

Per variant: median gemm1/gemm2 time over the timed rounds (round 0 is warm-up),
for both timers (``profiler_us`` = the test's kernel-bench, ``graph_us`` =
bench_driver.py's CUDA-graph timer), TB/s from the byte model below, the test's
accuracy line, the kernel names, and a bit-exact comparison of every saved output
with the first variant's round-1 output.

Byte model, every byte counted once (routed rows R = tokens * topk, all experts
active, MXFP4 = 0.5 B per element plus one E8M0 byte per 32 elements):
  gemm1  A (R x model_dim) + scales, W1 (E x 2*inter x model_dim) + scales,
         output R x inter as MXFP4 + scales (the kernel fuses SiLU and quant)
  gemm2  A2 (R x inter) + scales, W2 (E x model_dim x inter) + scales,
         output R x model_dim bf16 (one row per route, before gather-reduce)
The test's own "Bandwidth" counts gemm1's output as bf16 instead ("test GB/s").

usage: python summarize.py <bench.sh OUT directory>
"""

import glob
import os
import re
import statistics
import sys


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


def main(d):
    meta = read(os.path.join(d, "meta.txt"))
    args = re.search(r"^args\s+(.*)$", meta, re.MULTILINE).group(1)
    names = re.search(r"^variants\s+(.*)$", meta, re.MULTILINE).group(1).split()
    names = [n.split(":")[0] for n in names]
    nbytes = byte_model(args)
    print(meta.rstrip())
    print()
    print(
        "byte model (MB): "
        + ", ".join(f"{k} {v / 1e6:.1f}" for k, v in nbytes.items())
        + "  (see summarize.py)"
    )
    print()
    hdr = f"{'variant':10} {'kernel':6} {'timer':9} {'per-round us':34} {'median':>8} {'TB/s':>6} {'vs 1st':>7}"
    print(hdr)
    print("-" * len(hdr))
    first = {}
    for n in names:
        logs = sorted(
            glob.glob(os.path.join(d, f"{n}_r*.log")),
            key=lambda p: int(re.search(r"_r(\d+)\.log$", p).group(1)),
        )
        timed = [p for p in logs if not p.endswith("_r0.log")]
        for k in ("gemm1", "gemm2"):
            for timer, pat in (
                ("profiler", rf"\] {k}: us = ([0-9.]+)"),
                ("graph", rf"\[moe-exp\] {k}: graph_us = ([0-9.]+)"),
            ):
                vals = []
                for p in timed:
                    m = re.findall(pat, read(p))
                    if m:
                        vals.append(float(m[-1]))
                if not vals:
                    continue
                med = statistics.median(vals)
                tbs = nbytes[k] / (med * 1e-6) / 1e12
                key = (k, timer)
                first.setdefault(key, med)
                rel = (med / first[key] - 1) * 100
                cells = " ".join(f"{v:.2f}" for v in vals)
                print(
                    f"{n:10} {k:6} {timer:9} {cells:34} {med:8.2f} {tbs:6.2f} {rel:+6.1f}%"
                )
    print()
    for n in names:
        acc = set()
        for p in glob.glob(os.path.join(d, f"{n}_r*.log")):
            acc.update(re.findall(r"kernel: (logits_diff=\S+ rel_l2=\S+)", read(p)))
        kern = sorted(
            os.path.basename(p)
            for p in glob.glob(os.path.join(d, f"dump-{n}", "*"))
            if "a8w4" in os.path.basename(p)
        )
        print(f"{n}: accuracy {' | '.join(sorted(acc)) or 'n/a'}")
        for kname in kern:
            print(f"    {kname}")
    outs = sorted(glob.glob(os.path.join(d, "out", "*.pt")))
    ref_path = os.path.join(d, "out", f"{names[0]}_r1.pt")
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
