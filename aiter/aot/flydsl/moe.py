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
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg as _ptr_view_safe
from aiter.ops.flydsl.moe_kernels import (
    build_stage1_compile_inputs,
    build_stage2_compile_inputs,
    flydsl_moe_stage1,
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

                    full_job = {**job, **params}
                    key = job_identity(full_job)
                    if key in seen:
                        continue
                    seen.add(key)

                    jobs.append(full_job)

    return jobs


def _precompile_epilogue_to_cache(act: str, inter_dim: int, topk: int):
    """Precompile the CK-Tile split-K post-activation epilogue kernel.

    cktile_moe_stage1 with split_k>1 emits a workspace of interleaved gate/up
    output and then runs a FlyDSL epilogue to apply the activation:
      silu  -> flydsl_silu_and_mul_interleaved (silu_and_mul_fq with
               quant_mode="none", gui_layout=True)
      swiglu -> flydsl_swiglu_and_mul_interleaved (swiglu_and_mul)
    Dispatch through the same runtime builders so the cache key matches; the
    key depends only on inter_dim (swiglu) / (inter_dim, topk) (silu), and the
    row/token dims are dynamic, so dummy buffers suffice.
    """
    import torch

    from aiter.ops.flydsl.moe_kernels import (
        _explicit_stage1_target,
        _get_compiled_silu_fused,
        _get_compiled_swiglu,
        _run_compiled,
    )

    dev = torch.device("cpu")
    rows = 256
    plan_launcher = None
    if os.environ.get("AITER_FLYDSL_USE_STAGE1_COMPILE_PLAN", "0") == "1":
        from aiter.ops.flydsl.compile_plan import DEFAULT_COMPILE_OP_REGISTRY
        from aiter.ops.flydsl.moe_compile_plan import (
            resolve_cktile_stage1_compile_plan,
        )

        plan = resolve_cktile_stage1_compile_plan(
            target=_explicit_stage1_target(),
            inter_dim=inter_dim,
            topk=topk,
            split_k=2,
            act=act,
            post_activation_layout="interleaved",
        )
        launchers = DEFAULT_COMPILE_OP_REGISTRY.compile_plan(plan)
        if len(launchers) != 1:
            raise RuntimeError("CK-Tile epilogue plan must contain exactly one unit")
        plan_launcher = launchers[0]

    # COMPILE_ONLY=1 makes the executor compile + persist the artifact without
    # launching the kernel; without it _run_compiled would dispatch on the
    # (absent) GPU under fake tensors and fault.
    with compile_only_env():
        if act == "swiglu":
            exe = (
                plan_launcher
                if plan_launcher is not None
                else _get_compiled_swiglu(inter_dim)
            )
            x = torch.zeros((rows, inter_dim * 2), dtype=torch.bfloat16, device=dev)
            out = torch.zeros((rows, inter_dim), dtype=torch.bfloat16, device=dev)
            # trailing 0 = stream: null/default (compile-only, never launched)
            _run_compiled(exe, (x, out, rows, 0))
            return

        exe = (
            plan_launcher
            if plan_launcher is not None
            else _get_compiled_silu_fused(
                inter_dim, topk, quant_mode="none", gui_layout=True, act="silu"
            )
        )
        x = torch.zeros((rows, inter_dim * 2), dtype=torch.bfloat16, device=dev)
        out = torch.zeros((rows, inter_dim), dtype=torch.bfloat16, device=dev)
        empty_scale = torch.empty(0, dtype=torch.uint8, device=dev)
        empty_i32 = torch.empty(0, dtype=torch.int32, device=dev)
        empty_f32 = torch.empty(0, dtype=torch.float32, device=dev)
        sorted_token_ids = torch.zeros(rows, dtype=torch.int32, device=dev)
        num_valid_ids = torch.zeros(2, dtype=torch.int32, device=dev)
        _run_compiled(
            exe,
            (
                _ptr_view_safe(x),
                _ptr_view_safe(out),
                _ptr_view_safe(empty_scale),
                _ptr_view_safe(sorted_token_ids),
                _ptr_view_safe(num_valid_ids),
                _ptr_view_safe(empty_i32),
                _ptr_view_safe(empty_f32),
                rows,
                sorted_token_ids.shape[0],
                float("inf"),  # swiglu_limit (unused for silu)
                0,  # stream: null/default (compile-only, kernel is never launched)
            ),
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

    Drives the *runtime* ``flydsl_moe_stage1`` / ``flydsl_moe_stage2`` with fake
    top-level inputs under ``COMPILE_ONLY=1`` + ``FakeTensorMode``. Because the
    same runtime host code derives the internal buffers, packs the kernel args
    and issues the compile, the JIT cache key written here is identical to the
    one inference looks up -- there is no hand-mirrored shape/arg logic to drift.

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

    from torch._subclasses.fake_tensor import FakeTensorMode

    cfg = dict(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        **kwargs,
    )
    cu_str = str(cu_num) if cu_num > 0 else None
    stage1_plan_opt_in = (
        stage == 1
        and os.environ.get("AITER_FLYDSL_USE_STAGE1_COMPILE_PLAN", "0") == "1"
    )
    if stage1_plan_opt_in:
        from aiter.ops.flydsl.moe_kernels import (
            get_stage1_compile_plan_resolution_count,
        )

        plan_count_before = get_stage1_compile_plan_resolution_count()

    t0 = time.time()
    try:
        with (
            override_env("FLYDSL_GPU_ARCH", aot_arch),
            compile_only_env(),
            override_env("CU_NUM", cu_str),
            FakeTensorMode(),
        ):
            if is_epilogue:
                _precompile_epilogue_to_cache(
                    act=kwargs.get("act", "silu"),
                    inter_dim=inter_dim,
                    topk=topk,
                )
            else:
                from aiter.jit.utils.chip_info import get_cu_num

                get_cu_num.cache_clear()
                if stage == 1:
                    flydsl_moe_stage1(**build_stage1_compile_inputs(**cfg))
                    if stage1_plan_opt_in:
                        plan_count_after = get_stage1_compile_plan_resolution_count()
                        if plan_count_after <= plan_count_before:
                            raise RuntimeError(
                                "Stage1 AOT opt-in did not reach CompilePlan"
                            )
                        result["stage1_compile_plan"] = True
                else:
                    flydsl_moe_stage2(**build_stage2_compile_inputs(**cfg))
        elapsed = time.time() - t0
        result["compile_time"] = elapsed
        plan_label = "  stage1_plan=True" if stage1_plan_opt_in else ""
        print(
            f"  [OK] compile  {elapsed:6.1f}s  {shape_str}  arch={aot_arch}{plan_label}"
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
