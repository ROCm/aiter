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
    cu_num_to_arch,
    job_identity,
    override_env,
    run_jobs_parallel,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.aot_backend import compile_aot, create_compile_context
from aiter.ops.flydsl.compile_plan import RocmTarget
from aiter.ops.flydsl.launch_context import LaunchContext
from aiter.ops.flydsl.moe_kernels import (
    build_stage2_compile_inputs,
    flydsl_moe_stage2,
    get_flydsl_kernel_params,
)

# Keep the default AOT coverage aligned with runtime config resolution.
DEFAULT_CSVS = [
    AITER_CONFIGS.AITER_CONFIG_FMOE_FILE,
]
MOE_AOT_ARCH_DEFAULT = "gfx950"


def _parse_optional_float(value, source: str) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError as e:
        raise ValueError(f"{source} must be a float, got {value!r}") from e


def _row_swiglu_limit(row: dict[str, str]) -> float:
    return _parse_optional_float(row.get("swiglu_limit"), "swiglu_limit") or 0.0


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
            token = int(row["token"])
            model_dim = int(row["model_dim"])
            inter_dim = int(row["inter_dim"])
            experts = int(row["expert"])
            topk = int(row["topk"])
            doweight_stage1 = bool(int(row.get("doweight_stage1", "0")))
            cu_num = int(row.get("cu_num", "0"))
            block_m = int(row.get("block_m", "0") or "0")
            act_type = row.get("act_type", "")
            act = (
                "swiglu"
                if act_type.strip().split(".")[-1].lower() == "swiglu"
                else "silu"
            )
            q_type = row.get("q_type", "")
            dtype = row.get("dtype", "")
            q_dtype_w = row.get("q_dtype_w", "")
            swiglu_limit = _row_swiglu_limit(row)
            # Cover both runtime bias choices for fp4-weight MoE. Model configs
            # share kernel families, and runtime bias selection can vary by
            # activation dtype/model semantics.
            bias_supported = (
                q_type.strip().split(".")[-1] == "per_1x32"
                and dtype in ("torch.bfloat16", "torch.float16")
                and "float4_e2m1fn_x2" in q_dtype_w
            )
            enable_bias_options = [False, True] if bias_supported else [False]

            # Detect stage1's fuse_quant from kernel suffix to align stage2's
            # a2_scale shape with what runtime actually passes.
            stage1_name = row.get("kernelName1", "").strip()
            stage1_params = (
                get_flydsl_kernel_params(stage1_name)
                if stage1_name.startswith("flydsl_")
                else None
            )
            stage1_out_dtype = stage1_params.get("out_dtype") if stage1_params else None

            # cktile_ stage1 runs a FlyDSL post-activation epilogue (silu ->
            # silu_and_mul_fq, swiglu -> swiglu_and_mul) that the flydsl_-only loop
            # below skips, so emit its job here. The cache key needs only
            # (inter_dim, topk)/(inter_dim), which the CSV shape covers regardless
            # of runtime split_k.
            if stage1_name.startswith("cktile_"):
                epi_job = {
                    "kernel_name": f"cktile_epilogue_{act}",
                    "stage": "epilogue",
                    "act": act,
                    "inter_dim": inter_dim,
                    "topk": topk,
                    "cu_num": cu_num,
                    "split_k": 2,
                    "post_activation_layout": "interleaved",
                    "enable_bias": False,
                    # Not used by the epilogue compile; zeroed so dedup keys on
                    # (act, inter_dim, topk, cu_num) only.
                    "model_dim": 0,
                    "experts": 0,
                }
                key = job_identity(epi_job)
                if key not in seen:
                    seen.add(key)
                    jobs.append(epi_job)

            for col in ("kernelName1", "kernelName2"):
                name = row.get(col, "").strip()
                if not name or not name.startswith("flydsl_"):
                    continue

                params = get_flydsl_kernel_params(name)
                if params is None:
                    print(f"  [WARN] Unknown kernel name: {name}, skipping")
                    continue

                for enable_bias in enable_bias_options:
                    job = {
                        "kernel_name": name,
                        "model_dim": model_dim,
                        "inter_dim": inter_dim,
                        "experts": experts,
                        "topk": topk,
                        "doweight_stage1": doweight_stage1,
                        "cu_num": cu_num,
                        "act": act,
                        "enable_bias": enable_bias,
                        "token_num": token,
                        "block_m": block_m,
                        "swiglu_limit": swiglu_limit,
                    }
                    # Stage2 needs to know whether stage1 fuses fp4/fp8 quant —
                    # this changes the shape of a2_scale (sorted scale buffer
                    # vs separate quant call output).
                    if params["stage"] == 2:
                        job["stage1_fuse_quant"] = (
                            stage1_out_dtype
                            if stage1_out_dtype in ("fp4", "fp8")
                            else None
                        )
                    else:
                        # This is the production Stage1 wrapper setting.  Keep
                        # it in job metadata so direct plan resolution sees the
                        # same real compiler argument without entering the host.
                        job["use_async_copy"] = True

                    full_job = {**job, **params}
                    key = job_identity(full_job)
                    if key in seen:
                        continue
                    seen.add(key)

                    jobs.append(full_job)

    return jobs


def _job_target(aot_arch: str, cu_num: int) -> RocmTarget:
    """Resolve an explicit CSV/build target without querying a live device."""

    resolved_cu = int(cu_num)
    if resolved_cu <= 0:
        configured = os.environ.get("CU_NUM")
        resolved_cu = int(configured) if configured else 0
    if resolved_cu <= 0 and aot_arch == MOE_AOT_ARCH_DEFAULT:
        resolved_cu = 256
    if resolved_cu <= 0:
        raise ValueError(
            f"cu_num must be explicit for AOT target {aot_arch!r}; "
            "set it in the CSV or CU_NUM"
        )
    return RocmTarget(aot_arch, resolved_cu)


def _compile_stage1_plan(cfg: dict, context) -> int:
    from aiter.ops.flydsl.moe_compile_plan import (
        resolve_moe_stage1_compile_plan,
    )

    plan = resolve_moe_stage1_compile_plan(context=context, **cfg)
    for unit in plan.units:
        compile_aot(unit, context=context)
    return len(plan.units)


def _compile_cktile_epilogue_plan(cfg: dict, context) -> int:
    from aiter.ops.flydsl.moe_compile_plan import (
        resolve_cktile_stage1_compile_plan,
    )

    plan = resolve_cktile_stage1_compile_plan(
        context=context,
        inter_dim=cfg["inter_dim"],
        topk=cfg["topk"],
        split_k=cfg["split_k"],
        act=cfg["act"],
        post_activation_layout=cfg["post_activation_layout"],
        enable_bias=cfg["enable_bias"],
    )
    for unit in plan.units:
        compile_aot(unit, context=context)
    return len(plan.units)


def _precompile_epilogue_to_cache(
    act: str,
    inter_dim: int,
    topk: int,
    *,
    context,
) -> int:
    """Resolve and directly compile one CK-Tile Stage1 FlyDSL epilogue."""

    return _compile_cktile_epilogue_plan(
        {
            "act": act,
            "inter_dim": inter_dim,
            "topk": topk,
            "split_k": 2,
            "post_activation_layout": "interleaved",
            "enable_bias": False,
        },
        context,
    )


def compile_one_config(
    kernel_name: str,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    cu_num: int = 0,
    **kwargs,
) -> dict:
    """Compile one MoE kernel configuration and save to cache.

    Stage1 and its CK-Tile FlyDSL epilogues compile directly from resolved
    ``CompileUnit.signature`` metadata. Stage2 intentionally remains on the
    transitional FakeTensor/full-host path until its own CompilePlan migration.

    Returns a dict with timing info.
    """
    stage = kwargs.pop("stage")
    aot_arch = cu_num_to_arch(cu_num, default=MOE_AOT_ARCH_DEFAULT)
    is_epilogue = stage == "epilogue"
    shape_str = (
        f"{kernel_name}  inter_dim={inter_dim} topk={topk}"
        if is_epilogue
        else (
            f"{kernel_name}  "
            f"model_dim={model_dim} inter_dim={inter_dim} "
            f"E={experts} topk={topk}"
        )
    )
    result = {
        "kernel_name": kernel_name,
        "shape": shape_str,
        "compile_time": None,
        "compile_arch": aot_arch,
    }

    cfg = dict(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        **kwargs,
    )

    t0 = time.time()
    try:
        target = _job_target(aot_arch, cu_num)
        context = create_compile_context(target)
        if stage == 1:
            unit_count = _compile_stage1_plan(cfg, context)
            result["compile_units"] = unit_count
            result["direct_stage1_aot"] = True
        elif is_epilogue:
            unit_count = _compile_cktile_epilogue_plan(cfg, context)
            result["compile_units"] = unit_count
            result["direct_stage1_aot"] = True
        elif stage == 2:
            # Stage2 is deliberately transitional: it still executes its full
            # host under FakeTensorMode, but receives an explicit null launch
            # context instead of reading COMPILE_ONLY inside kernel code.
            from torch._subclasses.fake_tensor import FakeTensorMode

            with (
                override_env("ARCH", target.arch),
                override_env("FLYDSL_GPU_ARCH", target.arch),
                compile_only_env(),
                override_env("CU_NUM", str(target.cu_count)),
                FakeTensorMode(),
            ):
                from aiter.jit.utils.chip_info import get_cu_num

                get_cu_num.cache_clear()
                flydsl_moe_stage2(
                    **build_stage2_compile_inputs(**cfg),
                    launch_context=LaunchContext(None),
                )
        else:
            raise ValueError(f"unsupported MoE AOT stage: {stage!r}")
        elapsed = time.time() - t0
        result["compile_time"] = elapsed
        direct_label = (
            f"  direct_units={result['compile_units']}"
            if result.get("direct_stage1_aot")
            else "  stage2_transitional=True"
        )
        print(
            f"  [OK] compile  {elapsed:6.1f}s  {shape_str}  "
            f"arch={aot_arch}{direct_label}"
        )
    except Exception as e:
        print(f"  [FAIL] compile  {shape_str}  arch={aot_arch}: {e}")

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
    epilogue_jobs = [j for j in all_jobs if j["stage"] == "epilogue"]
    print("=" * 72)
    print("FlyDSL MoE AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:          {csv_path}")
    print(f"  Stage1 jobs:    {len(stage1_jobs)}")
    print(f"  Stage2 jobs:    {len(stage2_jobs)}")
    print(f"  Epilogue jobs:  {len(epilogue_jobs)}")
    print(f"  Total jobs:     {len(all_jobs)}")
    print("  Compile arch: (from cu_num)")
    print(f"  Cache dir:    {cache_dir}")
    print(f"  Target arch:  {arch}")
    print("=" * 72)

    total_t0 = time.time()

    # Stage1, stage2 and CK-Tile epilogue kernels are independent compiles
    # (each writes its own artifact to cache; none reads another's output), so
    # they share a single pool for maximum fan-out instead of serial passes.
    print(f"\n--- Compiling {len(all_jobs)} kernels (stage1 + stage2 + epilogue) ---")
    results = run_jobs_parallel(
        compile_one_config, stage1_jobs + stage2_jobs + epilogue_jobs
    )

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
