# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune FlyDSL MX GEMM tiles and regenerate aiter/configs/a6w4_flydsl_tuned_gemm.csv.

M is a runtime argument of ``launch_gemm``, so a tile config compiles once per
(N, K, tiles, a_dtype) and can then be timed at every batch size -- sweeping ten
batch sizes costs barely more than sweeping one.

Only configs that beat :func:`pick_mx_tiles` by more than ``--keep-pct`` are
written out. A tuned table that merely restates the heuristic is dead weight at
lookup time and noise for reviewers.

Usage:
    python op_tests/op_benchmarks/hip/tune_gemm_mx_a6w4.py            # rewrite CSV
    python op_tests/op_benchmarks/hip/tune_gemm_mx_a6w4.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time

import torch

# (label, N, K) for a [M,K] x [N,K].T linear.
#
# DeepSeek-R1 at TP=8: hidden 7168, MLA q_lora 1536 / kv_lora 512, 128 heads of
# 128+64 (qk) and 128 (v), moe_intermediate 2048, dense intermediate 18432.
# Qwen3.8-27B at TP=1: hidden 5120, intermediate 17408, 24 heads x head_dim 256,
# 4 KV heads.
SHAPES = [
    ("r1.q_b_proj", 3072, 1536),
    ("r1.kv_b_proj", 4096, 512),
    ("r1.o_proj", 7168, 2048),
    ("r1.moe_gate_up", 4096, 7168),
    ("r1.dense_gate_up", 4608, 7168),
    ("r1.dense_down", 7168, 2304),
    ("q38.qkv", 8192, 5120),
    ("q38.o_proj", 5120, 6144),
    ("q38.gate_up", 34816, 5120),
    ("q38.down", 5120, 17408),
]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
TILES = list(itertools.product((32, 64, 128, 256), (64, 128, 256), (128, 256)))
ITERS, WARMUP = 20, 5


def _bench(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / ITERS * 1e6)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtypes", default="fp6,fp4,fp8")
    ap.add_argument("--keep-pct", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 1
    props = torch.cuda.get_device_properties(0)
    gfx = props.gcnArchName.split(":")[0]
    if gfx != "gfx950":
        print(f"gfx950-only tuner, got {gfx}", file=sys.stderr)
        return 1
    cu = props.multi_processor_count

    import aiter.ops.flydsl.batched_gemm_mxfp4 as bg

    kept, dropped = [], 0
    for label, N, K in SHAPES:
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.5
        w_codes, w_scales = bg.preshuffle_mx_weight(w)
        x_max = torch.randn(max(BATCH_SIZES), K, device="cuda",
                            dtype=torch.bfloat16) * 0.5
        for a_dtype in args.dtypes.split(","):
            quant = {M: bg.quant_mx_act(x_max[:M].contiguous(), a_dtype)
                     for M in BATCH_SIZES}

            def run(M, tiles):
                a_codes, a_scales = quant[M]
                return bg.flydsl_gemm_mxfp4(
                    a_codes, w_codes, a_scales, w_scales, N, torch.bfloat16,
                    a_dtype=a_dtype, tile_m=tiles[0], tile_n=tiles[1],
                    tile_k=tiles[2])

            best = {M: (float("inf"), None) for M in BATCH_SIZES}
            for tiles in TILES:
                if not bg.tiles_are_valid(*tiles, N, K, a_dtype):
                    continue
                try:  # one compile per (N, K, tiles, a_dtype)
                    run(BATCH_SIZES[-1], tiles)
                    torch.cuda.synchronize()
                except Exception:  # noqa: BLE001
                    continue
                for M in BATCH_SIZES:
                    us = _bench(lambda M=M, tiles=tiles: run(M, tiles))
                    if us < best[M][0]:
                        best[M] = (us, tiles)

            for M in BATCH_SIZES:
                us, tiles = best[M]
                if tiles is None:
                    continue
                heur = bg.pick_mx_tiles(M, N, K, a_dtype)
                if tiles == heur:
                    dropped += 1
                    continue
                h_us = _bench(lambda: run(M, heur))
                gain = (h_us / us - 1) * 100
                if gain <= args.keep_pct:
                    dropped += 1
                    continue
                kept.append([gfx, cu, M, N, K, a_dtype, *tiles, f"{us:.3f}",
                             f"{2 * M * N * K / (us / 1e6) / 1e12:.2f}",
                             f"{gain:.1f}"])
            print(f"  {label:<20} {a_dtype}  kept so far {len(kept)}", flush=True)

    kept.sort(key=lambda r: (r[5], r[3], r[4], r[2]))
    print(f"\nkept {len(kept)} rows (>{args.keep_pct}% over heuristic), "
          f"dropped {dropped}")
    if args.dry_run:
        return 0
    # The canonical table, i.e. AITER_CONFIG_GEMM_A6W4 -- deliberately not
    # AITER_CONFIGS.AITER_CONFIG_GEMM_A6W4_FILE, which may resolve to a merged
    # copy under /tmp when model_configs/ tables are present.
    from aiter.jit.core import AITER_CONFIG_GEMM_A6W4

    out = os.path.normpath(AITER_CONFIG_GEMM_A6W4)
    with open(out, "w", newline="") as fh:
        # lineterminator: csv.writer defaults to CRLF, and every other table in
        # aiter/configs/ is LF.
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["gfx", "cu_num", "M", "N", "K", "a_dtype", "tile_m",
                     "tile_n", "tile_k", "us", "tflops",
                     "gain_vs_heuristic_pct"])
        wr.writerows(kept)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
