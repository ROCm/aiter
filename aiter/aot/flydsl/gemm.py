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

from aiter.aot.flydsl.common import (
    AotJob,
    CompileBundle,
    collect_aot_jobs,
    compile_bundle_to_cache,
    job_identity,
    make_compile_one_config,
    make_aot_job,
    print_summary,
    resolve_csv_paths,
    run_job_sections,
    torch_dtype_for_kernel,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
from aiter.ops.flydsl.kernels.splitk_hgemm import compile_hgemm_kernel

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

_PRESHUFFLE_RE = re.compile(
    r"^flydsl_bpreshuflle_"
    r"(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"(?P<qa>[A-Z0-9]+)_(?P<qw>[A-Z0-9]+)_(?P<out>[A-Z0-9]+)_"
    r"(?P<lds_stage>\d+)x(?P<cshuffle>\d+)x(?P<async_copy>\d+)x(?P<waves_per_eu>\d+)_"
    r"(?P<scheduler>[A-Za-z0-9_]+)$"
)
_HGEMM_RE = re.compile(
    r"^flydsl_gemm(?P<stage>\d+)_"
    r"a(?P<a_dtype>[a-z0-9]+)_w(?P<w_dtype>[a-z0-9]+)_(?P<out_dtype>[a-z0-9]+)_"
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"split_k(?P<split_k>\d+)_"
    r"block_m_warp(?P<block_m_warps>\d+)_"
    r"block_n_warp(?P<block_n_warps>\d+)_"
    r"async_copy(?P<async_copy>True|False)_"
    r"b_to_lds(?P<b_to_lds>True|False)_"
    r"b_preshuffle(?P<b_preshuffle>True|False)_"
    r"c_to_lds(?P<c_to_lds>True|False)_"
    r"(?P<target_gfx>gfx[0-9a-z]+)$"
)
_SHORT_DTYPE = {
    "F8": "fp8",
    "I8": "int8",
    "B16": "bf16",
    "F16": "fp16",
}


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
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


def _parse_hgemm_kernel_name(name: str) -> Optional[Dict]:
    m = _HGEMM_RE.fullmatch(name)
    if m is None:
        return None

    a_dtype = m.group("a_dtype")
    w_dtype = m.group("w_dtype")
    if a_dtype != w_dtype:
        raise ValueError(
            f"Unsupported mixed HGEMM input dtypes in {name!r}: {a_dtype} vs {w_dtype}"
        )

    return {
        "kind": "hgemm",
        "stage": int(m.group("stage")),
        "dtype": a_dtype,
        "out_dtype": m.group("out_dtype"),
        "tile_m": int(m.group("tile_m")),
        "tile_n": int(m.group("tile_n")),
        "tile_k": int(m.group("tile_k")),
        "split_k": int(m.group("split_k")),
        "block_m_warps": int(m.group("block_m_warps")),
        "block_n_warps": int(m.group("block_n_warps")),
        "async_copy": _parse_bool(m.group("async_copy")),
        "b_to_lds": _parse_bool(m.group("b_to_lds")),
        "b_preshuffle": _parse_bool(m.group("b_preshuffle")),
        "c_to_lds": _parse_bool(m.group("c_to_lds")),
        "target_gfx": m.group("target_gfx"),
    }


def _row_int(row: dict[str, str], *keys: str) -> int:
    for key in keys:
        value = row.get(key, "").strip()
        if value:
            return int(value)
    raise KeyError(f"Missing integer field in CSV row; tried {keys!r}")


def _normalize_gemm_row(row: dict[str, str]) -> list[AotJob]:
    kernel_name = row.get("kernel_name", row.get("kernelName", "")).strip()
    libtype = row.get("libtype", "").strip()
    op_family = row.get("op_family", "").strip()
    if op_family and op_family != "gemm":
        return []
    if libtype and libtype != "flydsl":
        return []
    if not kernel_name.startswith("flydsl_"):
        return []

    if kernel_name.startswith("flydsl_bpreshuflle_"):
        spec = _parse_preshuffle_kernel_name(kernel_name)
    elif kernel_name.startswith("flydsl_gemm"):
        spec = _parse_hgemm_kernel_name(kernel_name)
    else:
        spec = None

    if spec is None:
        print(f"  [WARN] Unknown FlyDSL GEMM kernel name: {kernel_name}, skipping")
        return []

    problem = {
        "m": _row_int(row, "m", "M"),
        "n": _row_int(row, "n", "N"),
        "k": _row_int(row, "k", "K"),
    }
    return [make_aot_job(kernel_name=kernel_name, problem=problem, spec=spec)]


def parse_csv(csv_path: str) -> list[AotJob]:
    """Parse a GEMM tuned CSV and return unique normalized FlyDSL compile jobs."""
    jobs: list[AotJob] = []
    seen = set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for job in _normalize_gemm_row(row):
                key = job_identity(job)
                if key in seen:
                    continue
                seen.add(key)
                jobs.append(job)

    return jobs


def _build_hgemm_bundle(
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
    async_copy: bool,
    b_to_lds: bool,
    b_preshuffle: bool,
    c_to_lds: bool,
    target_gfx: str,
    **kwargs,
) -> CompileBundle:
    del kwargs, out_dtype

    import torch

    dev = torch.device("cuda")
    torch_dtype = torch_dtype_for_kernel(dtype)

    current_gfx = get_gfx()
    if target_gfx != current_gfx:
        print(
            f"  [WARN] Kernel targets {target_gfx} but current target is {current_gfx}; "
            "compiling with current target parameters"
        )

    out = torch.empty((m, n), device=dev, dtype=torch_dtype)
    a = torch.empty((m, k), device=dev, dtype=torch_dtype)
    b = torch.empty((n, k), device=dev, dtype=torch_dtype)
    counter = torch.zeros(
        (128 * 3,),
        device=dev,
        dtype=torch.int32,
    )
    stream = torch.cuda.current_stream(device=dev)

    exe = compile_hgemm_kernel(
        dtype,
        n,
        k,
        TILE_M=tile_m,
        TILE_N=tile_n,
        TILE_K=tile_k,
        SPLIT_K=split_k,
        BLOCK_M_WARPS=block_m_warps,
        BLOCK_N_WARPS=block_n_warps,
        B_PRE_SHUFFLE=b_preshuffle,
        B_TO_LDS=b_to_lds,
    )
    return exe, (out, a, b, m, counter, 0, stream)


def _build_preshuffle_bundle(
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
) -> CompileBundle:
    del kwargs

    import torch

    dev = torch.device("cuda")
    out_torch_dtype = torch_dtype_for_kernel(out_dtype)

    # FlyDSL preshuffle kernels consume raw quantized bytes for fp8/int8 paths.
    a = torch.empty((m * k,), device=dev, dtype=torch.int8)
    b = torch.empty((n * k,), device=dev, dtype=torch.int8)
    out = torch.empty((m * n,), device=dev, dtype=out_torch_dtype)
    scale_a = torch.empty((max(m, 1),), device=dev, dtype=torch.float32)
    scale_b = torch.empty((max(n, 1),), device=dev, dtype=torch.float32)
    stream = torch.cuda.current_stream(device=dev)

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
    return exe, (out, a, b, scale_a, scale_b, m, n, stream)


_BUNDLE_BUILDERS = {
    "hgemm": _build_hgemm_bundle,
    "preshuffle": _build_preshuffle_bundle,
}


def build_gemm_bundle(problem: dict[str, object], spec: dict[str, object]) -> CompileBundle:
    kind = spec.get("kind")
    bundle_builder = _BUNDLE_BUILDERS.get(kind)
    if bundle_builder is None:
        raise ValueError(f"Unknown GEMM AOT kind: {kind}")
    return bundle_builder(**problem, **spec)


def _compile_one_gemm_job(job: AotJob) -> None:
    compile_bundle_to_cache(build_gemm_bundle(job["problem"], job["spec"]))


def _gemm_shape_str(job: AotJob) -> str:
    problem = job["problem"]
    return (
        f'{job["kernel_name"]}  '
        f'M={problem["m"]} N={problem["n"]} K={problem["k"]}'
    )


def _gemm_result_fields(job: AotJob) -> dict:
    return {"kernel_name": job["kernel_name"], "kind": job["spec"]["kind"]}


compile_one_config = make_compile_one_config(
    shape_builder=_gemm_shape_str,
    compile_action=_compile_one_gemm_job,
    result_builder=_gemm_result_fields,
)


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

    csv_paths = resolve_csv_paths(args.csv)

    cache_dir = os.path.expanduser(
        os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "~/.flydsl/cache")
    )
    arch = os.environ.get("ARCH") or os.environ.get("GPU_ARCHS") or get_gfx()

    all_jobs = collect_aot_jobs(csv_paths, parse_csv)

    hgemm_jobs = [j for j in all_jobs if j["spec"]["kind"] == "hgemm"]
    preshuffle_jobs = [j for j in all_jobs if j["spec"]["kind"] == "preshuffle"]

    print("=" * 72)
    print("FlyDSL GEMM AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:              {csv_path}")
    print(f"  HGEMM jobs:       {len(hgemm_jobs)}")
    print(f"  Preshuffle jobs:  {len(preshuffle_jobs)}")
    print(f"  Total jobs:       {len(all_jobs)}")
    print(f"  Cache dir:        {cache_dir}")
    print(f"  Target arch:      {arch}")
    print("=" * 72)

    total_t0 = time.time()
    results = run_job_sections(
        [
            ("HGEMM", hgemm_jobs),
            ("Preshuffle GEMM", preshuffle_jobs),
        ],
        compile_one_config,
    )

    total_elapsed = time.time() - total_t0
    sys.exit(print_summary(total_elapsed=total_elapsed, results=results, cache_dir=cache_dir))


if __name__ == "__main__":
    main()
