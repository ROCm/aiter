# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT precompile for the FlyDSL MXFP4/MXFP6/MXFP8 preshuffle GEMM (gfx950).

Walks a tuned CSV, parses each `flydsl_mxpsh_*` kernelName into its launch config,
and compiles the kernel into the JIT cache under compile_only_env() so runtime hits
the cache. Mirrors aiter/aot/flydsl/gemm.py.

Usage:
    COMPILE_ONLY=1 python -m aiter.aot.flydsl.mxscale_preshuffle \
        --csv aiter/configs/mxscale_preshuffle_tuned_gemm.csv
"""

from __future__ import annotations

import argparse
import csv as _csv
import os

from aiter.aot.flydsl.common import compile_only_env
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
    parse_kernel_name,
)


def _default_csvs() -> list[str]:
    return [AITER_CONFIGS.AITER_CONFIG_GEMM_MXSCALE_PRESHUFFLE_FILE]


def _compile_to_cache(
    M,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    a_dtype,
    b_dtype,
    out_dtype,
    waves_per_eu,
    xcd_swizzle=0,
    split_k=1,
    blockscale=False,
):
    import torch

    from aiter.ops.flydsl.mxscale_preshuffle_kernels import (
        flydsl_mxscale_preshuffle_gemm,
    )

    a_bytes = K // 2 if a_dtype == "fp4" else K
    b_bytes = K // 2 if b_dtype == "fp4" else K
    out_dt = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with compile_only_env():
        A = torch.zeros((M, a_bytes), dtype=torch.uint8, device=dev)
        Bt = torch.zeros((N, b_bytes), dtype=torch.uint8, device=dev)
        if blockscale:
            # compact coarse scales: A 1x128 (rows, K//128), B 128x128 (N//128, K//128)
            a_scale = torch.zeros(
                ((M + 31) // 32 * 32, K // 128), dtype=torch.uint8, device=dev
            )
            b_scale = torch.zeros((N // 128, K // 128), dtype=torch.uint8, device=dev)
        else:
            a_scale = torch.zeros(
                ((M + 31) // 32 * 32, K // 32), dtype=torch.uint8, device=dev
            )
            b_scale = torch.zeros(
                ((N + 31) // 32 * 32, K // 32), dtype=torch.uint8, device=dev
            )
        Out = torch.zeros((M, N), dtype=out_dt, device=dev)
        flydsl_mxscale_preshuffle_gemm(
            A,
            Bt,
            a_scale,
            b_scale,
            Out,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            waves_per_eu=waves_per_eu,
            xcd_swizzle=xcd_swizzle,
            split_k=split_k,
            blockscale=blockscale,
        )


def parse_csv(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(_csv.DictReader(f))


def compile_one_config(row: dict, blockscale: bool = False) -> bool:
    name = (row.get("kernelName") or "").strip()
    parsed = parse_kernel_name(name)
    if parsed is None:
        return False
    M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
    cfg = {
        "M": M,
        "N": N,
        "K": K,
        "tile_m": parsed["tile_m"],
        "tile_n": parsed["tile_n"],
        "tile_k": parsed["tile_k"],
        "a_dtype": parsed["a_dtype"],
        "b_dtype": parsed["b_dtype"],
        "out_dtype": parsed["out_dtype"],
        "waves_per_eu": parsed["waves_per_eu"],
        "xcd_swizzle": parsed["xcd_swizzle"],
        "split_k": parsed["split_k"],
    }
    # MX per-1x32 path (default, always).
    _compile_to_cache(**cfg)
    # blockscale path (a8w8 only, N%128==0) is a distinct Constexpr -> separate
    # binary; precompile it too so runtime doesn't JIT on first call.
    if (
        blockscale
        and parsed["a_dtype"] == "fp8"
        and parsed["b_dtype"] == "fp8"
        and N % 128 == 0
    ):
        _compile_to_cache(**cfg, blockscale=True)
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--csv", nargs="+", default=_default_csvs(), help="tuned CSV(s) to precompile"
    )
    ap.add_argument(
        "--blockscale",
        action="store_true",
        help="also precompile the a8w8 blockscale variant (A 1x128, B 128x128) "
        "for each fp8/fp8, N%%128==0 config",
    )
    args = ap.parse_args()

    total = 0
    done = 0
    for csv_path in args.csv:
        if not os.path.exists(csv_path):
            print(f"[aot.mxscale_preshuffle] skip missing csv: {csv_path}", flush=True)
            continue
        for row in parse_csv(csv_path):
            total += 1
            try:
                if compile_one_config(row, blockscale=args.blockscale):
                    done += 1
                    print(
                        f"[aot.mxscale_preshuffle] compiled {row.get('kernelName')} "
                        f"(M={row.get('M')} N={row.get('N')} K={row.get('K')})",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[aot.mxscale_preshuffle] FAILED {row.get('kernelName')}: "
                    f"{str(exc).splitlines()[0][:120]}",
                    flush=True,
                )
    print(f"[aot.mxscale_preshuffle] compiled {done}/{total} configs", flush=True)


if __name__ == "__main__":
    main()
