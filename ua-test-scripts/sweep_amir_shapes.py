#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""Sweep the unified-attention test script over a representative subset of
the production decode + prefill shapes captured during a vLLM model run
(/root/AmirMM_extrapolated 1.jsonl).

The full JSONL contains 640 records. The varying axes there are only
batch (`num_seqs`) and (max_seqlen_q, max_seqlen_k); everything else
is fixed:
    head_size   = 128
    block_size  = 64
    Hq, Hkv     = 12, 2          (GQA-6)
    q/k/v dtype = float8_e4m3fn  (FP8)
    out dtype   = bfloat16
    no alibi / sliding-window / sinks / softcap

448 records are decode (Sq=1) at 7 distinct Sk ∈
{1, 1000, 5000, 10000, 50000, 131072, 196608}; the other 192 are square
prefill (Sq=Sk) at 3 distinct lengths ∈ {1000, 5000, 10000}. Batches
range over 4..64 + {128, 256, 512}.

Sweeping all 640 would take ~hour and the variance across adjacent
batches at the same (Sq, Sk) is tiny. We instead pick a representative
ladder of batches that covers the device's CTA-saturation regimes
(few-CTA → split-KV, batch-saturated → no split, between).

For each shape we run `test_unified_attention_ck.py` in single-shape
mode with `--triton` for the head-to-head, capture CK + Triton times
and the heuristic's chosen num_splits, and stream rows to a CSV. The
reference torch path is disabled (`--no-reference`) for sweep speed —
we've already verified end-to-end correctness on representative
shapes; this run is for perf bookkeeping.

Usage:
    ./sweep_amir_shapes.py             # full default sweep, ~10 min
    ./sweep_amir_shapes.py --quick     # smaller subset for iteration
    ./sweep_amir_shapes.py --gpu 3     # device override
    ./sweep_amir_shapes.py --csv out.csv

The output CSV is the deliverable: it's joined into the analysis
write-up in ua-test-scripts/STATUS.md.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
AITER_ROOT = HERE.parent
TEST_SCRIPT = AITER_ROOT / "op_tests" / "test_unified_attention_ck.py"


# --- Sweep matrix ----------------------------------------------------------

# Decode sweep: hit every Sk band the JSONL has and a ladder of batches
# that crosses the heuristic's CTA-saturation tiers
# (batch=4..64 → splits=128, batch=128 → 8, 256 → 4, 512 → 2).
DECODE_BATCHES = [4, 8, 16, 32, 64, 128, 256, 512]
DECODE_SKS     = [1, 1000, 5000, 10000, 50000, 131072, 196608]

# Prefill: only Sq==Sk in the JSONL. Total_q gets large fast (batch × Sq),
# so cap batches at 32 to keep the sweep under ~15 min total. Production
# prefill batches at Sq=10k are typically ≤16 anyway (TTFT-bound).
PREFILL_BATCHES = [4, 8, 16, 32]
PREFILL_SQ_SK   = [(1000, 1000), (5000, 5000), (10000, 10000)]

QUICK_DECODE_BATCHES = [4, 64, 512]
QUICK_DECODE_SKS     = [1, 5000, 131072]
QUICK_PREFILL_BATCHES = [4, 16]
QUICK_PREFILL_SQ_SK   = [(1000, 1000), (10000, 10000)]


# --- Parsing the test-script output ---------------------------------------

_RE_SPLITS  = re.compile(r"num_splits=(\d+)")
_RE_CK_T    = re.compile(r"CK time\s*=\s*([\d.]+)\s*ms")
_RE_TRI_T   = re.compile(r"Triton time\s*=\s*([\d.]+)\s*ms")
_RE_CK_BW   = re.compile(r"CK Bandwidth\s*=\s*([\d.]+)\s*GB/s")
_RE_TRI_BW  = re.compile(r"Triton Bandwidth\s*=\s*([\d.]+)\s*GB/s")
_RE_CK_TF   = re.compile(r"CK TFLOPs\s*=\s*([\d.]+)")
_RE_SPEEDUP = re.compile(r"Speedup\s*=\s*([\d.]+)x")


def _parse_run(stdout: str) -> dict:
    def _f(rx, default=None):
        m = rx.search(stdout)
        return float(m.group(1)) if m else default

    def _i(rx, default=None):
        m = rx.search(stdout)
        return int(m.group(1)) if m else default

    return {
        "num_splits":   _i(_RE_SPLITS, 1),
        "ck_ms":        _f(_RE_CK_T),
        "triton_ms":    _f(_RE_TRI_T),
        "ck_bw":        _f(_RE_CK_BW),
        "triton_bw":    _f(_RE_TRI_BW),
        "ck_tflops":    _f(_RE_CK_TF),
        "speedup":      _f(_RE_SPEEDUP),
    }


# --- Driver ---------------------------------------------------------------

def _run_one(batch: int, sq: int, sk: int, gpu: str, timeout_s: int) -> dict:
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = gpu

    cmd = [
        sys.executable,
        str(TEST_SCRIPT),
        "-b", str(batch),
        "-sq", str(sq),
        "-sk", str(sk),
        "--num-heads", "12,2",
        "--head-size", "128",
        "--block-size", "64",
        "--dtype", "fp8",
        "--num-blocks", "auto",
        "--triton",
        "--no-reference",  # speed: corr is already validated on representative shapes
        "--seed", "42",
    ]

    t0 = time.time()
    try:
        p = subprocess.run(
            cmd, cwd=AITER_ROOT, env=env,
            capture_output=True, text=True, timeout=timeout_s,
        )
        stdout = (p.stdout or "") + "\n" + (p.stderr or "")
        rc = p.returncode
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or "") + "\n" + (e.stderr or "")
        rc = -1
    dt = time.time() - t0

    parsed = _parse_run(stdout)
    parsed.update({
        "batch": batch,
        "sq":    sq,
        "sk":    sk,
        "phase": "decode" if sq == 1 else "prefill",
        "rc":    rc,
        "wall_s": round(dt, 1),
    })
    if parsed["ck_ms"] is None or parsed["triton_ms"] is None:
        parsed["error_tail"] = "\n".join(stdout.splitlines()[-10:])
    return parsed


def _build_matrix(args) -> list[tuple[int, int, int]]:
    if args.quick:
        db, dsk, pb, psk = (QUICK_DECODE_BATCHES, QUICK_DECODE_SKS,
                            QUICK_PREFILL_BATCHES, QUICK_PREFILL_SQ_SK)
    else:
        db, dsk, pb, psk = (DECODE_BATCHES, DECODE_SKS,
                            PREFILL_BATCHES, PREFILL_SQ_SK)
    cells: list[tuple[int, int, int]] = []
    for b in db:
        for sk in dsk:
            cells.append((b, 1, sk))
    for b in pb:
        for (sq, sk) in psk:
            cells.append((b, sq, sk))
    return cells


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--gpu", default="2",
                    help="HIP_VISIBLE_DEVICES (default: 2)")
    ap.add_argument("--csv", default=str(HERE / "sweep_amir_shapes.csv"),
                    help="Output CSV path")
    ap.add_argument("--quick", action="store_true",
                    help="Run a smaller subset (~10 cells) for iteration")
    ap.add_argument("--timeout", type=int, default=300,
                    help="Per-cell timeout in seconds (default: 300)")
    args = ap.parse_args()

    matrix = _build_matrix(args)
    print(f"[sweep] device={args.gpu}  cells={len(matrix)}  out={args.csv}")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "phase", "batch", "sq", "sk",
        "num_splits",
        "ck_ms", "triton_ms", "speedup",
        "ck_bw", "triton_bw", "ck_tflops",
        "rc", "wall_s", "error_tail",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for i, (b, sq, sk) in enumerate(matrix, 1):
            print(f"[{i:>3}/{len(matrix)}] b={b:<3} sq={sq:<5} sk={sk:<7}",
                  end="", flush=True)
            row = _run_one(b, sq, sk, args.gpu, args.timeout)
            ck   = row["ck_ms"]
            tri  = row["triton_ms"]
            sp   = row["speedup"]
            ns   = row["num_splits"]
            if ck is not None and tri is not None:
                tag = "CK wins " if sp and sp >= 1.0 else "Triton  "
                print(f"  splits={ns:<3} CK={ck:6.3f}ms  Tri={tri:6.3f}ms  "
                      f"{sp:.2f}x [{tag}]  ({row['wall_s']}s)")
            else:
                print(f"  FAILED (rc={row['rc']}, {row['wall_s']}s)")
            for k in fieldnames:
                row.setdefault(k, None)
            w.writerow({k: row.get(k) for k in fieldnames})
            f.flush()

    print(f"\n[sweep] wrote {csv_path}")


if __name__ == "__main__":
    main()
