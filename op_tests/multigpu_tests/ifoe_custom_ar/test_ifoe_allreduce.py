# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""IFOE cross-node custom all-reduce (gfx1250).

Builds and runs ``ualoe_allreduce`` -- a standalone multi-rank all-reduce that
reuses aiter's 2-stage kernel (``cross_device_reduce_2stage`` + ``start_sync`` /
``end_sync``) but shares peer buffers via HIP **fabric handles**
(``hipMemExportToShareableHandle`` / ``hipMemImportFromShareableHandle``) instead
of IPC.  Fabric handles are node-independent, so the exact same kernel runs
cross-node over IFOE -- there is no inter/intra-node difference inside the kernel.

TP4 (single node, 4 GPUs) is exercised here.  TP8 (two nodes x 4 GPUs) is a
two-host launch, documented in ``README.md``.
"""

import argparse
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ualoe_allreduce.cpp")
BIN = os.path.join(HERE, "ualoe_allreduce")


def build(arch: str = "gfx1250") -> None:
    cmd = [
        "/opt/rocm/bin/hipcc",
        "-std=c++17",
        "-O3",
        f"--offload-arch={arch}",
        SRC,
        "-o",
        BIN,
    ]
    subprocess.run(cmd, check=True)


def run_tp4(mb: int = 64, tdm: bool = False, port: int = 55570) -> str:
    """Launch 4 ranks on the local node (GPU i <-> GPU i); return rank 0 stdout."""
    procs = []
    for rank in range(4):
        cmd = [
            BIN,
            "--rank",
            str(rank),
            "--world",
            "4",
            "--gpu",
            str(rank),
            "--coord",
            "127.0.0.1",
            "--port",
            str(port),
            "--mb",
            str(mb),
        ]
        if tdm:
            cmd.append("--tdm")
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True))
    out0 = procs[0].communicate()[0]
    for p in procs[1:]:
        p.wait()
    return out0


def main() -> None:
    ap = argparse.ArgumentParser(description="IFOE custom all-reduce TP4 test")
    ap.add_argument("--mb", type=int, default=64, help="tensor size in MiB")
    ap.add_argument("--tdm", action="store_true", help="use the TDM reduce path")
    ap.add_argument("--arch", default="gfx1250")
    args = ap.parse_args()

    build(args.arch)
    out = run_tp4(mb=args.mb, tdm=args.tdm)
    print(out, end="")

    assert "PASS" in out and "FAIL" not in out, "all-reduce correctness failed"
    m = re.search(r"busbw ([\d.]+) GB/s", out)
    assert m is not None, "no benchmark line found"
    print(f"TP4 OK: busbw = {m.group(1)} GB/s")


if __name__ == "__main__":
    main()
