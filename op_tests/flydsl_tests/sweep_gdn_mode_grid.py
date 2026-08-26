# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sweep GDN prefill shapes and record which opus mode wins each one.

Rows are (H, B) pairs -- H fixed by the TP degree, B the number of packed
sequences -- since the winner tracks their product, the chain count B*H, rather
than either one alone.  Columns are the segment length.

Reuses the input builder, the callable factory and the wall-clock timer from
``bench_gdn_block_ws_vs_flydsl`` so a cell here is the same measurement that
script reports, minus the per-kernel profiler breakdown -- the grid only needs
to know which path is fastest, and dropping the profiler is what makes the
sweep affordable.

Output is a JSON consumed by ``render_gdn_mode_grid.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_gdn_block_ws_vs_flydsl as B

DEFAULT_JSON = Path(__file__).with_name("gdn_prefill_mode_grid.json")

BACKENDS = ("ws", "wf", "cf", "cs")
# Per-rank head count is not free to choose -- it is the model's value head
# count divided by the TP degree.  Qwen3.5 has Hv=64, so TP 1/2/4/8 gives
# H 64/32/16/8, with TP=8 the production配置.
HV_MODEL = 64
TPS = (1, 2, 4, 8)
# The serving case: the scheduler cuts its token budget into equal segments of
# the model's prompt length, so B sequences of `seqlen` are packed into one
# call and T = B x seqlen.  Since only B*H decides the winner, (H, B) is one
# axis of the grid rather than two, and seqlen gets the other.
N_SEQS = (1, 2, 4, 8)
SEQLENS = (1024, 2048, 4096, 8192)
# A GQA ratio r needs H % r == 0 to split into Hg = H/r key heads.
VARIANTS = {"gqa4": 4, "gqa2": 2}


def measure(variant_ratio: int, h: int, seqlen: int, n_seqs: int) -> dict:
    """One grid cell: every backend's wall time at this shape."""
    if h % variant_ratio:
        return {"skipped": f"H={h} is not a multiple of the GQA ratio {variant_ratio}"}
    hg = h // variant_ratio

    # The bench module reads the shape off its own globals.
    B.HK, B.HV, B.TP = hg, h, 1
    B.HG, B.H = hg, h
    B.FULL_PROMPT_LEN = seqlen

    cell: dict = {
        "H": h,
        "Hg": hg,
        "seqlen": seqlen,
        "n_seqs": n_seqs,
        "total_tokens": n_seqs * seqlen,
        "walls": {},
        "errors": {},
    }
    try:
        t = B.build_inputs(n_seqs)
    except Exception as exc:  # noqa: BLE001
        cell["skipped"] = f"inputs: {exc}"
        return cell

    for backend in BACKENDS:
        reason = B.unsupported_reason(backend)
        if reason:
            cell["errors"][backend] = reason
            continue
        try:
            run = B.make_callable(backend, t)
            run()
            if backend == "prepare":
                # Existence check only, so a handful of iterations is enough.
                saved, B.PROF_ITERS = B.PROF_ITERS, 5
                try:
                    names = B.profile_kernels(run)
                finally:
                    B.PROF_ITERS = saved
                if not any("gdn_prepare" in n for n in names):
                    cell["errors"][backend] = "fell back to the Triton prepare pair"
                    continue
            cell["walls"][backend] = B.bench_wall_us(run)
        except Exception as exc:  # noqa: BLE001
            cell["errors"][backend] = f"{type(exc).__name__}: {exc}"
        finally:
            torch.cuda.synchronize()

    del t
    torch.cuda.empty_cache()
    return cell


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seqlens",
        type=int,
        nargs="+",
        default=list(SEQLENS),
        help="segment lengths; the column axis",
    )
    ap.add_argument(
        "--tps", type=int, nargs="+", default=list(TPS), help="TP degrees; H = Hv/TP"
    )
    ap.add_argument(
        "--n-seqs",
        type=int,
        nargs="+",
        default=list(N_SEQS),
        help="packed sequence counts B; rows are the (H, B) product",
    )
    ap.add_argument("--hv", type=int, default=HV_MODEL, help="model value head count")
    ap.add_argument(
        "--out", default=str(DEFAULT_JSON), help="where to write the grid JSON"
    )
    args = ap.parse_args()

    import aiter

    props = torch.cuda.get_device_properties(0)
    out = {
        "gfx": props.gcnArchName,
        "cus": props.multi_processor_count,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "aiter_path": aiter.__file__,
        "seqlens": list(args.seqlens),
        "hv_model": args.hv,
        "rows": [
            {"tp": tp, "H": args.hv // tp, "n_seqs": n, "bh": args.hv // tp * n}
            for tp in sorted(args.tps)
            for n in sorted(args.n_seqs)
        ],
        "backends": list(BACKENDS),
        "labels": {b: B.LABEL[b] for b in BACKENDS},
        "short": {b: B.SHORT[b] for b in BACKENDS},
        "num_iters": B.NUM_ITERS,
        "tables": [],
    }

    total = len(VARIANTS) * len(out["rows"]) * len(args.seqlens)
    done = 0
    t0 = time.time()
    for variant, ratio in VARIANTS.items():
        cells = []
        for row in out["rows"]:
            for seqlen in args.seqlens:
                cell = measure(ratio, row["H"], seqlen, row["n_seqs"])
                cell["tp"] = row["tp"]
                cells.append(cell)
                done += 1
                walls = cell.get("walls", {})
                best = min(walls, key=walls.get) if walls else None
                print(
                    f"[{done:3d}/{total}] {variant:5s} TP={row['tp']} H={row['H']:3d} "
                    f"B={row['n_seqs']:2d} B·H={row['bh']:3d} s={seqlen:5d} "
                    f"T={row['n_seqs'] * seqlen:6d}  "
                    + (
                        f"best={B.SHORT[best]:4s} {walls[best]:9.1f}us  "
                        + " ".join(
                            f"{B.SHORT[b]}={walls[b]:.1f}"
                            for b in BACKENDS
                            if b in walls
                        )
                        if best
                        else f"no result ({cell.get('skipped', cell.get('errors'))})"
                    ),
                    flush=True,
                )
        out["tables"].append({"variant": variant, "gqa_ratio": ratio, "cells": cells})

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
