#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for MoE / Mixed-MoE FlyDSL kernels from aiter CSV configs.

Reads tuned CSV config files (e.g. dsv3_fp4_tuned_fmoe.csv), extracts all
unique FlyDSL kernel names, and pre-compiles them into the cache. The default
CSV set is resolved through ``AITER_CONFIGS`` so model-specific tuned CSVs can
be merged the same way as runtime JIT config lookup.

Usage:
    # Compile all unique FlyDSL kernels from default CSVs
    python -m aiter.aot.flydsl.moe

    # Custom CSV file(s)
    python -m aiter.aot.flydsl.moe --csv /path/to/config1.csv /path/to/config2.csv

Environment variables:
    FLYDSL_RUNTIME_CACHE_DIR  Cache directory (default: ~/.flydsl/cache)
    ARCH                      Target GPU architecture (e.g. gfx942, gfx950).
"""

import argparse
import csv
import os
import sys
import time

from aiter.aot.flydsl.common import (
    AotJob,
    CompileBundle,
    collect_aot_jobs,
    compile_bundle_to_cache,
    execute_bundle,
    job_identity,
    make_compile_one_config,
    make_aot_job,
    print_summary,
    resolve_csv_paths,
    run_job_sections,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.moe_kernels import (
    compile_flydsl_moe_stage1,
    compile_flydsl_moe_stage2,
    get_flydsl_kernel_params,
    _s1_args_fp4,
    _s1_args_std,
    _s2_args_fp4,
    _s2_args_std,
)

# Keep the default AOT coverage aligned with runtime config resolution.
DEFAULT_CSVS = [
    AITER_CONFIGS.AITER_CONFIG_FMOE_FILE,
]


def _row_int(row: dict[str, str], *keys: str) -> int:
    for key in keys:
        value = row.get(key, "").strip()
        if value:
            return int(value)
    raise KeyError(f"Missing integer field in CSV row; tried {keys!r}")


def _iter_moe_kernel_names(row: dict[str, str]) -> list[str]:
    kernel_name = row.get("kernel_name", "").strip()
    if kernel_name:
        return [kernel_name]
    return [
        name
        for name in (
            row.get("kernelName1", "").strip(),
            row.get("kernelName2", "").strip(),
        )
        if name
    ]


def _normalize_moe_row(row: dict[str, str]) -> list[AotJob]:
    op_family = row.get("op_family", "").strip()
    if op_family and op_family != "moe":
        return []

    problem = {
        "model_dim": _row_int(row, "model_dim"),
        "inter_dim": _row_int(row, "inter_dim"),
        "experts": _row_int(row, "experts", "expert"),
        "topk": _row_int(row, "topk"),
        "doweight_stage1": bool(int(row.get("doweight_stage1", "0"))),
    }
    jobs: list[AotJob] = []
    for kernel_name in _iter_moe_kernel_names(row):
        if not kernel_name.startswith("flydsl_"):
            continue

        spec = get_flydsl_kernel_params(kernel_name)
        if spec is None:
            print(f"  [WARN] Unknown kernel name: {kernel_name}, skipping")
            continue

        jobs.append(make_aot_job(kernel_name=kernel_name, problem=problem, spec=spec))
    return jobs


def parse_csv(csv_path: str) -> list[AotJob]:
    """Parse the CSV and return unique normalized MoE compile jobs."""
    jobs: list[AotJob] = []
    seen = set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for job in _normalize_moe_row(row):
                key = job_identity(job)
                if key in seen:
                    continue
                seen.add(key)
                jobs.append(job)

    return jobs


def _build_stage1_bundle(
    stage: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_dtype: str = "fp4",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    doweight_stage1: bool = False,
    waves_per_eu: int = 3,
    k_batch: int = 1,
    b_nt: int = 2,
    gate_only: bool = False,
    fuse_fp4_quant: bool = False,
    mode: str = "atomic",
    persist: bool = False,
    sort_block_m: int = 0,
    **kwargs,
) -> CompileBundle:
    """Build the stage1 executable and dummy args bundle for AOT compilation.

    Constructs minimal zero-filled tensors matching the kernel's expected
    signature.
    """
    del stage, mode, persist, sort_block_m, kwargs

    import torch

    dev = torch.device("cuda")
    is_fp4 = b_dtype == "fp4"
    tokens = tile_m
    E = experts
    _grid_y = 1
    _is_splitk = k_batch > 1
    n_in = inter_dim * 2 if is_fp4 else inter_dim
    k_in = model_dim

    # Dummy routing tensors (shape matters, data doesn't)
    sorted_ids = torch.zeros(tokens * topk, device=dev, dtype=torch.int32)
    sorted_expert_ids = torch.zeros(_grid_y, device=dev, dtype=torch.int32)
    num_valid_ids = torch.zeros(1, device=dev, dtype=torch.int32)
    sw = torch.zeros(tokens * topk, device=dev, dtype=torch.float32)

    if is_fp4:
        out = torch.zeros(
            tokens * topk * inter_dim // 2, device=dev, dtype=torch.uint8
        )
        a = torch.zeros(tokens * model_dim // 2, device=dev, dtype=torch.uint8)
        w = torch.zeros(
            E * 2 * inter_dim * model_dim // 2, device=dev, dtype=torch.uint8
        )
        a_scale = torch.zeros(1, device=dev, dtype=torch.uint8)
        w_scale = torch.zeros(1, device=dev, dtype=torch.uint8)
        out_scale = torch.zeros(1, device=dev, dtype=torch.uint8)
        args = _s1_args_fp4(
            out,
            a,
            w,
            a_scale,
            w_scale,
            sorted_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            out_scale,
            tokens,
            n_in,
            k_in,
            _grid_y,
            dev,
        )
    else:
        out = torch.zeros(tokens * topk * inter_dim, device=dev, dtype=torch.bfloat16)
        a = torch.zeros(tokens * model_dim, device=dev, dtype=torch.int8)
        w = torch.zeros(E * 2 * inter_dim * model_dim, device=dev, dtype=torch.int8)
        a_scale = torch.zeros(1, device=dev, dtype=torch.float32)
        w_scale = torch.zeros(1, device=dev, dtype=torch.float32)
        args = _s1_args_std(
            out,
            a,
            w,
            a_scale,
            w_scale,
            sorted_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            tokens,
            n_in,
            k_in,
            _grid_y,
        )

    exe = compile_flydsl_moe_stage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=doweight_stage1,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        waves_per_eu=waves_per_eu,
        k_batch=k_batch,
        b_nt=b_nt,
        gate_only=gate_only,
        fuse_fp4_quant=fuse_fp4_quant and not _is_splitk,
        fuse_sort_scale=fuse_fp4_quant and not _is_splitk,
    )
    return exe, args


def _build_stage2_bundle(
    stage: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_dtype: str = "fp4",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    mode: str = "atomic",
    persist: bool = False,
    sort_block_m: int = 0,
    **kwargs,
) -> CompileBundle:
    """Build the stage2 executable and dummy args bundle for AOT compilation."""
    del stage, kwargs

    import torch

    dev = torch.device("cuda")
    is_fp4 = b_dtype == "fp4"
    tokens = tile_m
    E = experts
    _grid_y = 1
    accumulate = mode != "reduce"
    _persist_m = -1 if persist else 4
    n_in = model_dim
    k_in = inter_dim

    # Dummy routing tensors (shape matters, data doesn't)
    sorted_ids = torch.zeros(tokens * topk, device=dev, dtype=torch.int32)
    sorted_expert_ids = torch.zeros(_grid_y, device=dev, dtype=torch.int32)
    num_valid_ids = torch.zeros(1, device=dev, dtype=torch.int32)
    sw = torch.zeros(tokens * topk, device=dev, dtype=torch.float32)

    if is_fp4:
        out = torch.zeros(tokens * model_dim, device=dev, dtype=torch.bfloat16)
        a = torch.zeros(tokens * topk * inter_dim // 2, device=dev, dtype=torch.uint8)
        w = torch.zeros(E * model_dim * inter_dim // 2, device=dev, dtype=torch.uint8)
        a_scale = torch.zeros(1, device=dev, dtype=torch.uint8)
        w_scale = torch.zeros(1, device=dev, dtype=torch.uint8)
        args = _s2_args_fp4(
            out,
            a,
            w,
            a_scale,
            w_scale,
            sorted_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            tokens,
            n_in,
            k_in,
            _grid_y,
            dev,
        )
    else:
        out = torch.zeros(tokens * model_dim, device=dev, dtype=torch.bfloat16)
        a = torch.zeros(tokens * topk * inter_dim, device=dev, dtype=torch.int8)
        w = torch.zeros(E * model_dim * inter_dim, device=dev, dtype=torch.int8)
        a_scale = torch.zeros(1, device=dev, dtype=torch.float32)
        w_scale = torch.zeros(1, device=dev, dtype=torch.float32)
        args = _s2_args_std(
            out,
            a,
            w,
            a_scale,
            w_scale,
            sorted_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            tokens,
            n_in,
            k_in,
            _grid_y,
        )

    exe = compile_flydsl_moe_stage2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=False,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        accumulate=accumulate,
        persist_m=_persist_m,
        sort_block_m=sort_block_m,
    )
    return exe, args


_BUNDLE_BUILDERS = {
    1: _build_stage1_bundle,
    2: _build_stage2_bundle,
}


def build_moe_bundle(problem: dict[str, object], spec: dict[str, object]) -> CompileBundle:
    stage = spec.get("stage")
    bundle_builder = _BUNDLE_BUILDERS.get(stage)
    if bundle_builder is None:
        raise ValueError(f"Unsupported MoE AOT stage: {stage}")
    return bundle_builder(**problem, **spec)


def _compile_one_moe_job(job: AotJob) -> None:
    compile_bundle_to_cache(build_moe_bundle(job["problem"], job["spec"]), runner=execute_bundle)


def _moe_shape_str(job: AotJob) -> str:
    problem = job["problem"]
    return (
        f'{job["kernel_name"]}  '
        f'model_dim={problem["model_dim"]} inter_dim={problem["inter_dim"]} '
        f'E={problem["experts"]} topk={problem["topk"]}'
    )


def _moe_result_fields(job: AotJob) -> dict:
    return {"kernel_name": job["kernel_name"]}


compile_one_config = make_compile_one_config(
    shape_builder=_moe_shape_str,
    compile_action=_compile_one_moe_job,
    result_builder=_moe_result_fields,
)


def main():
    parser = argparse.ArgumentParser(
        description="AOT pre-compile MoE / Mixed-MoE FlyDSL kernels from aiter CSV config",
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
    arch = os.environ.get("ARCH", "(auto-detect)")

    all_jobs = collect_aot_jobs(csv_paths, parse_csv)

    stage1_jobs = [j for j in all_jobs if j["spec"]["stage"] == 1]
    stage2_jobs = [j for j in all_jobs if j["spec"]["stage"] == 2]

    print("=" * 72)
    print("FlyDSL MoE AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:          {csv_path}")
    print(f"  Stage1 jobs:  {len(stage1_jobs)}")
    print(f"  Stage2 jobs:  {len(stage2_jobs)}")
    print(f"  Total jobs:   {len(all_jobs)}")
    print(f"  Cache dir:    {cache_dir}")
    print(f"  Target arch:  {arch}")
    print("=" * 72)

    total_t0 = time.time()
    results = run_job_sections(
        [
            ("Stage 1", stage1_jobs),
            ("Stage 2", stage2_jobs),
        ],
        compile_one_config,
    )

    total_elapsed = time.time() - total_t0
    sys.exit(print_summary(total_elapsed=total_elapsed, results=results, cache_dir=cache_dir))


if __name__ == "__main__":
    main()
