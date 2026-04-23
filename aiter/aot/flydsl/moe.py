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
    collect_aot_jobs,
    compile_only_env,
    job_identity,
    override_env,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.moe_kernels import (
    compile_flydsl_moe_stage1,
    compile_flydsl_moe_stage2,
    get_flydsl_kernel_params,
    _run_compiled,
    _s1_args_fp4,
    _s1_args_std,
    _s2_args_fp4,
    _s2_args_std,
)

# Keep the default AOT coverage aligned with runtime config resolution.
DEFAULT_CSVS = [
    AITER_CONFIGS.AITER_CONFIG_FMOE_FILE,
]
MOE_AOT_ARCH = "gfx950"


def parse_csv(csv_path: str):
    """Parse the CSV and return a list of unique compile jobs.

    Each job is a dict with keys:
        kernel_name, stage, model_dim, inter_dim, experts, topk,
        doweight_stage1 (for stage1), and all params from get_flydsl_kernel_params.

    Deduplicates by
    (kernel_name, model_dim, inter_dim, experts, topk, doweight_stage1).
    """
    jobs = []
    seen = set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_dim = int(row["model_dim"])
            inter_dim = int(row["inter_dim"])
            experts = int(row["expert"])
            topk = int(row["topk"])
            doweight_stage1 = bool(int(row.get("doweight_stage1", "0")))
            cu_num = int(row.get("cu_num", "0"))

            for col in ("kernelName1", "kernelName2"):
                name = row.get(col, "").strip()
                if not name or not name.startswith("flydsl_"):
                    continue

                job = {
                    "kernel_name": name,
                    "model_dim": model_dim,
                    "inter_dim": inter_dim,
                    "experts": experts,
                    "topk": topk,
                    "doweight_stage1": doweight_stage1,
                    "cu_num": cu_num,
                }
                key = job_identity(job)
                if key in seen:
                    continue
                seen.add(key)

                params = get_flydsl_kernel_params(name)
                if params is None:
                    print(f"  [WARN] Unknown kernel name: {name}, skipping")
                    continue

                jobs.append({**job, **params})

    return jobs


def _precompile_to_cache(
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
    gate_mode: str = "separated",
    mode: str = "atomic",
    persist: bool = False,
    sort_block_m: int = 0,
    cu_num: int = 0,
    **kwargs,
):
    """Trigger MLIR compilation with dummy tensors and COMPILE_ONLY=1.

    Constructs minimal zero-filled tensors matching the kernel's expected
    signature, then calls the JitFunction.  With COMPILE_ONLY=1 the compiled
    artifact is saved to the pkl cache without executing on GPU.
    No dependency on HIP ops (moe_sorting, shuffle_weight, etc.).
    """
    import torch

    dev = torch.device("cpu")
    _stream = 0
    is_fp4 = b_dtype == "fp4"
    tokens = tile_m
    E = experts
    _grid_y = 1

    # Dummy routing tensors (shape matters, data doesn't)
    sorted_ids = torch.zeros(tokens * topk, device=dev, dtype=torch.int32)
    sorted_expert_ids = torch.zeros(_grid_y, device=dev, dtype=torch.int32)
    num_valid_ids = torch.zeros(1, device=dev, dtype=torch.int32)
    sw = torch.zeros(tokens * topk, device=dev, dtype=torch.float32)

    _cu_num_str = str(cu_num) if cu_num > 0 else None
    with compile_only_env(), override_env("CU_NUM", _cu_num_str):
        # Clear cached CU count so get_cu_num() re-reads the env var.
        from aiter.jit.utils.chip_info import get_cu_num
        get_cu_num.cache_clear()

        if stage == 1:
            _is_splitk = k_batch > 1
            n_in = inter_dim * 2 if is_fp4 else inter_dim
            k_in = model_dim

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
                    stream=_stream,
                )
            else:
                out = torch.zeros(
                    tokens * topk * inter_dim, device=dev, dtype=torch.bfloat16
                )
                a = torch.zeros(tokens * model_dim, device=dev, dtype=torch.int8)
                w = torch.zeros(
                    E * 2 * inter_dim * model_dim, device=dev, dtype=torch.int8
                )
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
                    stream=_stream,
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
                gate_mode=gate_mode,
            )
            _run_compiled(exe, args)

        elif stage == 2:
            accumulate = mode != "reduce"
            _persist_m = -1 if persist else 4
            n_in = model_dim
            k_in = inter_dim

            if is_fp4:
                out = torch.zeros(tokens * model_dim, device=dev, dtype=torch.bfloat16)
                a = torch.zeros(
                    tokens * topk * inter_dim // 2, device=dev, dtype=torch.uint8
                )
                w = torch.zeros(
                    E * model_dim * inter_dim // 2, device=dev, dtype=torch.uint8
                )
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
                    stream=_stream,
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
                    stream=_stream,
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
            _run_compiled(exe, args)


def compile_one_config(
    kernel_name: str, model_dim: int, inter_dim: int, experts: int, topk: int, **kwargs
) -> dict:
    """Compile one MoE kernel configuration and save to cache.

    Uses COMPILE_ONLY=1 with dummy tensors to trigger MLIR compilation and
    pkl cache write without depending on HIP ops or executing on GPU.

    Returns a dict with timing info.
    """
    shape_str = (
        f"{kernel_name}  "
        f"model_dim={model_dim} inter_dim={inter_dim} "
        f"E={experts} topk={topk}"
    )
    result = {
        "kernel_name": kernel_name,
        "shape": shape_str,
        "compile_time": None,
        "compile_arch": MOE_AOT_ARCH,
    }

    t0 = time.time()
    try:
        with override_env("ARCH", MOE_AOT_ARCH), override_env(
            "FLYDSL_GPU_ARCH", MOE_AOT_ARCH
        ):
            _precompile_to_cache(
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                **kwargs,
            )
        elapsed = time.time() - t0
        result["compile_time"] = elapsed
        print(f"  [OK] compile  {elapsed:6.1f}s  {shape_str}  arch={MOE_AOT_ARCH}")
    except Exception as e:
        print(f"  [FAIL] compile  {shape_str}  arch={MOE_AOT_ARCH}: {e}")

    return result


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

    stage1_jobs = [j for j in all_jobs if j["stage"] == 1]
    stage2_jobs = [j for j in all_jobs if j["stage"] == 2]
    print("=" * 72)
    print("FlyDSL MoE AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:          {csv_path}")
    print(f"  Stage1 jobs:  {len(stage1_jobs)}")
    print(f"  Stage2 jobs:  {len(stage2_jobs)}")
    print(f"  Total jobs:   {len(all_jobs)}")
    print(f"  Compile arch: {MOE_AOT_ARCH}")
    print(f"  Cache dir:    {cache_dir}")
    print(f"  Target arch:  {arch}")
    print("=" * 72)

    total_t0 = time.time()
    results = []

    if stage1_jobs:
        print(f"\n--- Stage 1 ({len(stage1_jobs)} kernels) ---")
        for i, job in enumerate(stage1_jobs, 1):
            print(f"\n[{i}/{len(stage1_jobs)}] ", end="")
            r = compile_one_config(**job)
            results.append(r)

    if stage2_jobs:
        print(f"\n--- Stage 2 ({len(stage2_jobs)} kernels) ---")
        for i, job in enumerate(stage2_jobs, 1):
            print(f"\n[{i}/{len(stage2_jobs)}] ", end="")
            r = compile_one_config(**job)
            results.append(r)

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
