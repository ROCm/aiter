#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for FlyDSL GEMM kernels from aiter tuned CSV configs.

Reads tuned GEMM CSV config files, extracts all unique FlyDSL kernel entries,
and pre-compiles them into the FlyDSL cache. The default CSV set is resolved
through ``AITER_CONFIGS`` so model-specific tuned CSVs can be merged the same
way as runtime JIT config lookup.

Supported kernel families:
  - ``flydsl_gemm2_*``           split-K HGEMM kernels
  - ``flydsl_bpreshuflle_*``     a8w8 preshuffle GEMM kernels

Usage:
    # Compile all unique FlyDSL GEMM kernels from default CSVs
    python -m aiter.aot.flydsl.gemm

    # Custom CSV file(s)
    python -m aiter.aot.flydsl.gemm --csv /path/to/config1.csv /path/to/config2.csv

Environment variables:
    FLYDSL_RUNTIME_CACHE_DIR  Cache directory (default: ~/.flydsl/cache)
    GPU_ARCHS / ARCH          Target GPU architecture information for logging.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from typing import Dict, Optional

import flydsl.expr as fx

from aiter.aot.flydsl.common import (
    collect_aot_jobs,
    compile_only_env,
    cu_num_to_arch,
    job_identity,
    override_env,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.gemm_kernels import get_flydsl_splitk_hgemm_kernel_params
from aiter.ops.flydsl.kernels.hgemm_dispatch import compile_flydsl_hgemm_kernel
from aiter.ops.flydsl.kernels.preshuffle_gemm import compile_preshuffle_gemm_a8

# Keep the default AOT coverage aligned with runtime config resolution.
DEFAULT_CSVS = [
    AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE,
    AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_FILE,
    AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE,
    AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE,
    AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE,
    AITER_CONFIGS.AITER_CONFIG_A8W8_BATCHED_GEMM_FILE,
    AITER_CONFIGS.AITER_CONFIG_BF16_BATCHED_GEMM_FILE,
    AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE,
]
GEMM_AOT_ARCH_DEFAULT = "gfx950"

_PRESHUFFLE_RE = re.compile(
    r"^flydsl_bpreshuflle_"
    r"(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"(?P<qa>[A-Z0-9]+)_(?P<qw>[A-Z0-9]+)_(?P<out>[A-Z0-9]+)_"
    r"(?P<lds_stage>\d+)x(?P<cshuffle>\d+)x(?P<async_copy>\d+)x(?P<waves_per_eu>\d+)_"
    r"(?P<scheduler>[A-Za-z0-9_]+)$"
)
_SHORT_DTYPE = {
    "F8": "fp8",
    "I8": "int8",
    "B16": "bf16",
    "F16": "fp16",
}


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized == "":
        return False
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"Expected True/False, got {value!r}")


def _parse_preshuffle_kernel_name(name: str) -> Optional[Dict]:
    m = _PRESHUFFLE_RE.fullmatch(name)
    if m is None:
        return None

    qa = _SHORT_DTYPE.get(m.group("qa"))
    qw = _SHORT_DTYPE.get(m.group("qw"))
    out = _SHORT_DTYPE.get(m.group("out"))
    if qa is None or qw is None or out is None:
        return None
    if qa != qw:
        raise ValueError(
            f"Unsupported mixed preshuffle input dtypes in {name!r}: {qa} vs {qw}"
        )

    return {
        "kind": "preshuffle",
        "tile_m": int(m.group("tile_m")),
        "tile_n": int(m.group("tile_n")),
        "tile_k": int(m.group("tile_k")),
        "in_dtype": qa,
        "out_dtype": out,
        "lds_stage": int(m.group("lds_stage")),
        "use_cshuffle_epilog": int(m.group("cshuffle")),
        "use_async_copy": int(m.group("async_copy")),
        "waves_per_eu": int(m.group("waves_per_eu")),
        "scheduler": m.group("scheduler"),
    }


def parse_csv(csv_path: str):
    """Parse a GEMM tuned CSV and return a list of unique FlyDSL compile jobs."""
    jobs = []
    seen = set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel_name = row.get("kernelName", "").strip()
            libtype = row.get("libtype", "").strip()
            if libtype != "flydsl" or not kernel_name.startswith("flydsl_"):
                continue

            m = int(row["M"])
            n = int(row["N"])
            k = int(row["K"])
            cu_num = int(row.get("cu_num", "0"))

            if kernel_name.startswith("flydsl_bpreshuflle_"):
                params = _parse_preshuffle_kernel_name(kernel_name)
            elif kernel_name.startswith("flydsl_gemm"):
                params = get_flydsl_splitk_hgemm_kernel_params(kernel_name)
                if params is not None:
                    params = dict(params)
                    params["kind"] = "hgemm"
            else:
                params = None

            if params is None:
                print(
                    f"  [WARN] Unknown FlyDSL GEMM kernel name: {kernel_name}, skipping"
                )
                continue

            job = {
                "kernel_name": kernel_name,
                "m": m,
                "n": n,
                "k": k,
                "cu_num": cu_num,
                "has_bias": _parse_bool(row.get("bias")),
                **params,
            }
            key = job_identity(job)
            if key in seen:
                continue
            seen.add(key)

            jobs.append(job)

    return jobs


def _torch_dtype_for_kernel(dtype_name: str):
    import torch

    mapping = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "fp16": torch.float16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype name for GEMM AOT: {dtype_name!r}")
    return mapping[dtype_name]


def _compile_executable_to_cache(exe, *args) -> None:
    with compile_only_env():
        exe(*args)


def _compile_hgemm_to_cache(
    *,
    m: int,
    n: int,
    k: int,
    dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int,
    block_m_warps: int,
    block_n_warps: int,
    n_tile_repeat: int = 1,
    persistent_n_tiles: int = 1,
    waves_per_eu: int = 0,
    b_to_lds_unroll: int = 0,
    async_copy: bool,
    b_to_lds: bool,
    b_preshuffle: bool,
    c_to_lds: bool,
    target_gfx: str,
    kernel_family: str = "hgemm",
    has_bias: bool = False,
    **kwargs,
):
    del kwargs, out_dtype

    import torch

    has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    dev = torch.device("cuda") if has_cuda else torch.device("cpu")
    torch_dtype = _torch_dtype_for_kernel(dtype)

    out = torch.empty((m, n), device=dev, dtype=torch_dtype)
    a = torch.empty((m, k), device=dev, dtype=torch_dtype)
    b = torch.empty((n, k), device=dev, dtype=torch_dtype)
    bias = torch.empty((n,), device=dev, dtype=torch_dtype)
    counter = torch.zeros(
        (128 * 3,),
        device=dev,
        dtype=torch.int32,
    )
    stream = fx.Stream(torch.cuda.current_stream(device=dev) if has_cuda else 0)

    exe = compile_flydsl_hgemm_kernel(
        dtype,
        n,
        k,
        kernel_family=kernel_family,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        split_k=split_k,
        block_m_warps=block_m_warps,
        block_n_warps=block_n_warps,
        n_tile_repeat=n_tile_repeat,
        persistent_n_tiles=persistent_n_tiles,
        waves_per_eu=waves_per_eu,
        b_to_lds_unroll=b_to_lds_unroll,
        async_copy=async_copy,
        b_to_lds=b_to_lds,
        b_preshuffle=b_preshuffle,
        c_to_lds=c_to_lds,
        has_bias=has_bias,
    )
    # FlyDSL JIT does not accept None for tensor slots; pass a real buffer when
    # bias fusion is disabled (matches runtime launcher dummy tensor behavior).
    _compile_executable_to_cache(exe, out, a, b, bias, m, counter, 0, stream)


def _compile_preshuffle_to_cache(
    *,
    m: int,
    n: int,
    k: int,
    in_dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    lds_stage: int,
    use_cshuffle_epilog: int,
    use_async_copy: int,
    waves_per_eu: int,
    **kwargs,
):
    del kwargs

    import torch

    has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    dev = torch.device("cuda") if has_cuda else torch.device("cpu")
    out_torch_dtype = _torch_dtype_for_kernel(out_dtype)

    # FlyDSL preshuffle kernels consume raw quantized bytes for fp8/int8 paths.
    a = torch.empty((m * k,), device=dev, dtype=torch.int8)
    b = torch.empty((n * k,), device=dev, dtype=torch.int8)
    out = torch.empty((m * n,), device=dev, dtype=out_torch_dtype)
    scale_a = torch.empty((max(m, 1),), device=dev, dtype=torch.float32)
    scale_b = torch.empty((max(n, 1),), device=dev, dtype=torch.float32)
    stream = fx.Stream(torch.cuda.current_stream(device=dev) if has_cuda else 0)

    exe = compile_preshuffle_gemm_a8(
        N=n,
        K=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype="bf16" if out_torch_dtype == torch.bfloat16 else "fp16",
        lds_stage=lds_stage,
        use_cshuffle_epilog=bool(use_cshuffle_epilog),
        use_async_copy=bool(use_async_copy),
        waves_per_eu=None if waves_per_eu <= 0 else waves_per_eu,
    )
    _compile_executable_to_cache(exe, out, a, b, scale_a, scale_b, m, n, stream)


def compile_one_config(
    kernel_name: str, kind: str, m: int, n: int, k: int, cu_num: int = 0, **kwargs
) -> dict:
    """Compile one GEMM kernel configuration and save it to cache."""
    aot_arch = cu_num_to_arch(cu_num, default=GEMM_AOT_ARCH_DEFAULT)
    shape_str = f"{kernel_name}  M={m} N={n} K={k}"
    result = {
        "kernel_name": kernel_name,
        "kind": kind,
        "shape": shape_str,
        "compile_time": None,
        "compile_arch": aot_arch,
    }

    t0 = time.time()
    try:
        with override_env("ARCH", aot_arch), override_env("FLYDSL_GPU_ARCH", aot_arch):
            if kind == "hgemm":
                _compile_hgemm_to_cache(m=m, n=n, k=k, target_gfx=aot_arch, **kwargs)
            elif kind == "preshuffle":
                _compile_preshuffle_to_cache(m=m, n=n, k=k, **kwargs)
            else:
                raise ValueError(f"Unknown GEMM AOT kind: {kind}")

        elapsed = time.time() - t0
        result["compile_time"] = elapsed
        print(f"  [OK] compile  {elapsed:6.1f}s  {shape_str}  arch={aot_arch}")
    except Exception as e:
        print(f"  [FAIL] compile  {shape_str}  arch={aot_arch}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="AOT pre-compile FlyDSL GEMM kernels from aiter CSV config",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        default=DEFAULT_CSVS,
        help="Path(s) to tuned CSV config file(s); defaults come from AITER_CONFIGS",
    )
    args = parser.parse_args()

    csv_paths = [os.path.abspath(p) for p in args.csv]
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)

    cache_dir = os.path.expanduser(
        os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "~/.flydsl/cache")
    )
    arch = os.environ.get("ARCH") or os.environ.get("GPU_ARCHS") or "(auto-detect)"

    all_jobs = collect_aot_jobs(csv_paths, parse_csv)

    hgemm_jobs = [j for j in all_jobs if j["kind"] == "hgemm"]
    preshuffle_jobs = [j for j in all_jobs if j["kind"] == "preshuffle"]

    print("=" * 72)
    print("FlyDSL GEMM AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:              {csv_path}")
    print(f"  HGEMM jobs:       {len(hgemm_jobs)}")
    print(f"  Preshuffle jobs:  {len(preshuffle_jobs)}")
    print(f"  Total jobs:       {len(all_jobs)}")
    print("  Compile arch:     (from cu_num)")
    print(f"  Cache dir:        {cache_dir}")
    print(f"  Target arch:      {arch}")
    print("=" * 72)

    total_t0 = time.time()
    results = []

    if hgemm_jobs:
        print(f"\n--- HGEMM ({len(hgemm_jobs)} kernels) ---")
        for i, job in enumerate(hgemm_jobs, 1):
            print(f"\n[{i}/{len(hgemm_jobs)}] ", end="")
            results.append(compile_one_config(**job))

    if preshuffle_jobs:
        print(f"\n--- Preshuffle GEMM ({len(preshuffle_jobs)} kernels) ---")
        for i, job in enumerate(preshuffle_jobs, 1):
            print(f"\n[{i}/{len(preshuffle_jobs)}] ", end="")
            results.append(compile_one_config(**job))

    total_elapsed = time.time() - total_t0

    ok = sum(1 for r in results if r["compile_time"] is not None)
    fail = sum(1 for r in results if r["compile_time"] is None)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Total time:   {total_elapsed:.1f}s")
    print(f"  Compiled:     {ok} ok, {fail} failed")
    print(f"  Cache dir:    {cache_dir}")
    print()

    exit_code = 0
    if fail > 0:
        print("Some compilations failed. Check output above for details.")
        exit_code = 1
    else:
        print("All compilations succeeded. Cache is ready.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
