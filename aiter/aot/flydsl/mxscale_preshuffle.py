#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT precompile FlyDSL MX-scale preshuffle GEMMs from tuned CSV rows."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import flydsl.expr as fx

from aiter.aot.flydsl.common import (
    collect_aot_jobs,
    compile_only_env,
    cu_num_to_arch,
    job_identity,
    override_env,
    run_jobs_parallel,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
    parse_kernel_name,
)

DEFAULT_CSVS = [AITER_CONFIGS.AITER_CONFIG_GEMM_MXSCALE_PRESHUFFLE_FILE]
MXSCALE_AOT_ARCH_DEFAULT = "gfx950"


def parse_csv(csv_path: str):
    """Return unique exact-signature compile jobs from one tuned CSV."""
    jobs = []
    seen = set()
    with open(csv_path, newline="") as file:
        for row in csv.DictReader(file):
            kernel_name = (row.get("kernelName") or "").strip()
            if row.get("libtype", "").strip() != "flydsl" or not kernel_name:
                continue
            parsed = parse_kernel_name(kernel_name)
            if parsed is None:
                print(
                    f"  [WARN] Unknown MX-scale kernel name: {kernel_name}, skipping"
                )
                continue
            row_a_dtype = row["a_dtype"].strip()
            row_b_dtype = row["b_dtype"].strip()
            if (
                parsed["a_dtype"] != row_a_dtype
                or parsed["b_dtype"] != row_b_dtype
            ):
                raise ValueError(
                    f"{kernel_name}: encoded dtypes "
                    f"{parsed['a_dtype']}/{parsed['b_dtype']} do not match CSV "
                    f"{row_a_dtype}/{row_b_dtype}"
                )
            job = {
                "kernel_name": kernel_name,
                "m": int(row["M"]),
                "n": int(row["N"]),
                "k": int(row["K"]),
                "cu_num": int(row.get("cu_num", "0")),
                "gfx": row.get("gfx", "").strip(),
                **parsed,
            }
            identity = job_identity(job)
            if identity not in seen:
                seen.add(identity)
                jobs.append(job)
    return jobs


def job_arch(cu_num: int = 0, gfx: str = "") -> str:
    return gfx or cu_num_to_arch(cu_num, default=MXSCALE_AOT_ARCH_DEFAULT)


def _compile_to_cache(
    *,
    m,
    n,
    k,
    tile_m,
    tile_n,
    tile_k,
    a_dtype,
    b_dtype,
    out_dtype,
    waves_per_eu,
    xcd_swizzle,
    split_k,
):
    import torch

    from aiter.ops.flydsl.mxscale_preshuffle_kernels import (
        flydsl_mxscale_preshuffle_gemm,
    )

    device = torch.device("cpu")
    a_bytes = k // 2 if a_dtype == "fp4" else k
    b_bytes = k // 2 if b_dtype == "fp4" else k
    output_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    A = torch.empty((m, a_bytes), dtype=torch.uint8, device=device)
    B = torch.empty((n, b_bytes), dtype=torch.uint8, device=device)
    a_scale = torch.empty(
        ((m + 31) // 32 * 32, k // 32), dtype=torch.uint8, device=device
    )
    b_scale = torch.empty(
        ((n + 31) // 32 * 32, k // 32), dtype=torch.uint8, device=device
    )
    Out = torch.empty((m, n), dtype=output_dtype, device=device)
    workspace = (
        torch.empty((split_k, m, n), dtype=torch.float32, device=device)
        if split_k > 1
        else None
    )
    with compile_only_env():
        flydsl_mxscale_preshuffle_gemm(
            A,
            B,
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
            splitk_workspace=workspace,
            stream=fx.Stream(0),
        )


def compile_one_config(
    kernel_name: str,
    m: int,
    n: int,
    k: int,
    cu_num: int = 0,
    gfx: str = "",
    **kwargs,
) -> dict:
    """Compile one exact CSV signature into the FlyDSL cache."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    architecture = job_arch(cu_num, gfx)
    shape = f"{kernel_name} M={m} N={n} K={k}"
    result = {
        "kernel_name": kernel_name,
        "shape": shape,
        "compile_time": None,
        "compile_arch": architecture,
    }
    start = time.time()
    try:
        with (
            override_env("FLYDSL_GPU_ARCH", architecture),
            FakeTensorMode(),
        ):
            _compile_to_cache(m=m, n=n, k=k, **kwargs)
        result["compile_time"] = time.time() - start
        print(
            f"  [OK] compile {result['compile_time']:6.1f}s "
            f"{shape} arch={architecture}"
        )
    except Exception as error:  # noqa: BLE001 - AOT reports per-job failures
        print(f"  [FAIL] compile {shape} arch={architecture}: {error}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", default=DEFAULT_CSVS)
    args = parser.parse_args()
    csv_paths = [os.path.abspath(path) for path in args.csv]
    for path in csv_paths:
        if not os.path.isfile(path):
            print(f"Error: CSV file not found: {path}")
            return 1

    jobs = collect_aot_jobs(csv_paths, parse_csv)
    results = run_jobs_parallel(compile_one_config, jobs)
    passed = sum(result["compile_time"] is not None for result in results)
    failed = len(results) - passed
    print(
        f"[aiter] FlyDSL MX-scale AOT: compiled {passed} ok, {failed} failed "
        f"from {len(jobs)} exact signatures"
    )
    return int(failed != 0)


if __name__ == "__main__":
    sys.exit(main())
