# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Microbench and config sweep for dense gfx950 FlyDSL BF16 x MXFP4 GEMM.

Not collected by pytest (no test_ prefix). Examples:

    python op_tests/flydsl_tests/bench_flydsl_gemm_a16wfp4.py --mode default
    python op_tests/flydsl_tests/bench_flydsl_gemm_a16wfp4.py --mode sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

import torch

from aiter.ops.flydsl.gemm_a16wfp4 import (
    DenseGemmConfig,
    _select_gemm_config,
    a16wfp4_config_legal,
    flydsl_gemm_a16wfp4,
    prepare_gemm_a16wfp4_weight,
)
from aiter.test_common import run_perftest

_K3_SHAPES = ((8448, 7168), (1536, 7168), (7168, 768))
_TP2_SHAPES = ((1024, 3584), (3584, 512))
_AB_SHAPES = _K3_SHAPES + _TP2_SHAPES
_AB_MS = (1, 4096)
_SWEEP_MS = (1, 128, 4096)


def _make_inputs(m: int, n: int, k: int, seed: int = 0):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    packed = torch.randint(
        0,
        256,
        (n, k // 2),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    scale = torch.full((n, k // 32), 0x7F, device="cuda", dtype=torch.uint8)
    return a, packed, scale


def _metrics(m: int, n: int, k: int, us: float):
    flops = 2.0 * m * n * k
    tflops = flops / (us * 1e-6) / 1e12 if us > 0 else 0.0
    bytes_moved = m * k * 2 + n * (k / 2) + n * (k / 32) + m * n * 2
    tb_s = bytes_moved / (us * 1e-6) / 1e12 if us > 0 else 0.0
    return tflops, tb_s


def _bench(m: int, n: int, k: int, config: DenseGemmConfig | None = None):
    a, packed, scale = _make_inputs(m, n, k, seed=m + n + k)
    prepared = prepare_gemm_a16wfp4_weight(packed, scale)
    kwargs = {} if config is None else {"config": config}
    flydsl_gemm_a16wfp4(a, prepared, **kwargs)
    torch.cuda.synchronize()
    _, us = run_perftest(
        flydsl_gemm_a16wfp4,
        a,
        prepared,
        num_iters=21,
        num_warmup=2,
        num_rotate_args=1,
        use_cuda_event=True,
        **kwargs,
    )
    tflops, tb_s = _metrics(m, n, k, us)
    cfg = config if config is not None else _select_gemm_config(m, n, k)
    row = {
        "m": m,
        "n": n,
        "k": k,
        "block_m": cfg.block_m,
        "tile_n": cfg.tile_n,
        "tile_k": cfg.tile_k,
        "k_wave": cfg.k_wave,
        "waves_per_eu": cfg.waves_per_eu,
        "us": float(us),
        "tflops": tflops,
        "tb_s": tb_s,
        "status": "ok",
    }
    return row


def _print_row(row: dict):
    if row.get("status") != "ok":
        print(
            f"SKIP M={row['m']} N={row['n']} K={row['k']} "
            f"BM={row.get('block_m')} TN={row.get('tile_n')} TK={row.get('tile_k')} "
            f"kw={row.get('k_wave')} wpe={row.get('waves_per_eu')}: {row.get('error')}"
        )
        return
    print(
        f"a16wfp4 M={row['m']} N={row['n']} K={row['k']} "
        f"BM={row['block_m']} TN={row['tile_n']} TK={row['tile_k']} "
        f"k_wave={row['k_wave']} wpe={row['waves_per_eu']}: "
        f"{row['us']:.2f} us, {row['tflops']:.2f} TFLOPS, {row['tb_s']:.2f} TB/s"
    )


def _iter_sweep_configs(m: int, n: int, k: int):
    block_ms = (16,) if m == 1 else (16, 32)
    tile_ns = tuple(tn for tn in (128, 256) if n % tn == 0)
    tile_ks = (256, 512) if k % 512 == 0 else (256,)
    for block_m in block_ms:
        for tile_n in tile_ns:
            for tile_k in tile_ks:
                for k_wave in (1, 2, 4):
                    for waves_per_eu in (1, 2, 4):
                        cfg = DenseGemmConfig(
                            block_m, tile_n, tile_k, k_wave, waves_per_eu
                        )
                        if a16wfp4_config_legal(n, k, cfg):
                            yield cfg


def _write(path: str | None, row: dict):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _run_default(out: str | None):
    rows = []
    for n, k in _AB_SHAPES:
        for m in _AB_MS:
            row = _bench(m, n, k)
            _print_row(row)
            _write(out, row)
            rows.append(row)
    return rows


def _run_sweep(out: str | None):
    rows = []
    for n, k in _K3_SHAPES:
        for m in _SWEEP_MS:
            for cfg in _iter_sweep_configs(m, n, k):
                try:
                    row = _bench(m, n, k, config=cfg)
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "m": m,
                        "n": n,
                        "k": k,
                        "block_m": cfg.block_m,
                        "tile_n": cfg.tile_n,
                        "tile_k": cfg.tile_k,
                        "k_wave": cfg.k_wave,
                        "waves_per_eu": cfg.waves_per_eu,
                        "status": "skip",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    traceback.print_exc()
                _print_row(row)
                _write(out, row)
                rows.append(row)
    winners = _pick_winners(rows)
    print("\nWinners (lowest us; tie -> smaller k_wave, tile_k, tile_n, block_m):")
    for key, row in sorted(winners.items()):
        print(f"  {key}: {row}")
    if out:
        with open(out + ".winners.json", "w", encoding="utf-8") as handle:
            json.dump({str(k): v for k, v in winners.items()}, handle, indent=2)
    return rows, winners


def _pick_winners(rows: list[dict]) -> dict[tuple[int, int, int], dict]:
    winners: dict[tuple[int, int, int], dict] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row["m"], row["n"], row["k"])
        cur = winners.get(key)
        cand = (
            row["us"],
            row["k_wave"],
            row["tile_k"],
            row["tile_n"],
            row["block_m"],
        )
        if cur is None:
            winners[key] = row
            continue
        best = (
            cur["us"],
            cur["k_wave"],
            cur["tile_k"],
            cur["tile_n"],
            cur["block_m"],
        )
        if cand < best:
            winners[key] = row
    return winners


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("default", "sweep"), default="default")
    parser.add_argument("--out", default=None, help="JSONL path for per-config rows")
    args = parser.parse_args(argv)
    if args.mode == "default":
        _run_default(args.out)
    else:
        _run_sweep(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
