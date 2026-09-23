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
from collections.abc import Mapping

from aiter.aot.flydsl.common import (
    collect_aot_jobs,
    cu_num_to_arch,
    job_identity,
    run_jobs_parallel,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.aot_backend import compile_aot, create_compile_context
from aiter.ops.flydsl.compile_request import RocmTarget
from aiter.ops.flydsl.moe_kernels import (
    _S2_LEGACY_FP8_PITCH_ALIGN,
    _S2_LEGACY_FP8_SCALE_BLK,
    get_flydsl_kernel_params,
    requires_flydsl_stage2_reduce,
    resolve_flydsl_grid_y_persist_m,
    resolve_flydsl_stage1_tile_n,
    resolve_flydsl_stage2_tile_k,
)
from aiter.ops.flydsl.mxfp4_kname import parse_flydsl_v2_gemm2_kernel

# Keep the default AOT coverage aligned with runtime config resolution.
DEFAULT_CSVS = [
    AITER_CONFIGS.AITER_CONFIG_FMOE_FILE,
    AITER_CONFIGS.AITER_CONFIG_FHMOE_FILE,
]
MOE_AOT_ARCH_DEFAULT = "gfx950"


def parse_csv(csv_path: str):
    """Parse the CSV and return a list of unique compile jobs.

    Each job is a dict with keys:
        kernel_name, stage, model_dim, inter_dim, experts, topk,
        doweight_stage1 (for stage1), and all params from get_flydsl_kernel_params.

    Deduplicates with ``job_identity``, including token bucket, block size, and
    the shared-expert ID when present.
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
            shared_expert_id = int(row.get("shared_expert_id", "-1") or "-1")
            act_type = row.get("act_type", "")
            act_name = act_type.strip().split(".")[-1].lower()
            act = act_name if act_name in ("swiglu", "situv2") else "silu"
            q_type = row.get("q_type", "")
            dtype = row.get("dtype", "")
            q_dtype_w = row.get("q_dtype_w", "")
            # Cover both runtime bias choices for fp4-weight MoE. Model configs
            # share kernel families, and runtime bias selection can vary by
            # activation dtype/model semantics.
            bias_supported = (
                q_type.strip().split(".")[-1] == "per_1x32"
                and dtype in ("torch.bfloat16", "torch.float16")
                and "float4_e2m1fn_x2" in q_dtype_w
            )
            enable_bias_options = (
                [False]
                if shared_expert_id >= 0
                else ([False, True] if bias_supported else [False])
            )

            # Detect stage1's fuse_quant from kernel suffix to align stage2's
            # a2_scale shape with what runtime actually passes.
            stage1_name = row.get("kernelName1", "").strip()
            stage2_name = row.get("kernelName2", "").strip()
            stage1_params = (
                get_flydsl_kernel_params(stage1_name)
                if stage1_name.startswith("flydsl_")
                else None
            )
            stage1_out_dtype = stage1_params.get("out_dtype") if stage1_params else None
            stage1_v2_output_layout = "_moe2_layout_" in stage2_name
            stage2_v2_params = (
                parse_flydsl_v2_gemm2_kernel(stage2_name)
                if stage1_v2_output_layout
                else None
            )

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
                if name.startswith("flydsl_moe2_layout_"):
                    continue
                # a4w4 mxmoe-port kernels are precompiled by mxfp4_moe.py; they
                # share the flydsl_ prefix but are absent from this module's
                # registry, so without this guard every CSV row naming one is
                # reported as an unknown kernel.
                if name.startswith("flydsl_mxmoe_"):
                    continue

                params = get_flydsl_kernel_params(name)
                if params is None:
                    print(f"  [WARN] Unknown kernel name: {name}, skipping")
                    continue

                family_bias_options = (
                    [False]
                    if params["a_dtype"] == "bf16"
                    and params["b_dtype"] in ("fp4", "int4")
                    else enable_bias_options
                )
                for enable_bias in family_bias_options:
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
                    }
                    if shared_expert_id >= 0:
                        job["shared_expert_id"] = shared_expert_id
                    # Stage2 needs to know whether stage1 fuses fp4/fp8 quant --
                    # this changes the shape of a2_scale (sorted scale buffer
                    # vs separate quant call output).
                    if params["stage"] == 2:
                        job["stage1_fuse_quant"] = (
                            stage1_out_dtype
                            if stage1_out_dtype in ("fp4", "fp8")
                            else None
                        )
                    full_job = {**job, **params}
                    if params["stage"] == 1 and stage1_v2_output_layout:
                        full_job["v2_output_layout"] = True
                        if stage2_v2_params is not None:
                            full_job["out_dtype"] = stage2_v2_params["a_dtype"]
                    key = job_identity(full_job)
                    if key in seen:
                        continue
                    seen.add(key)

                    jobs.append(full_job)

    return jobs


def get_aot_jobs():
    """Return the default jobs registered with the unified AOT driver."""

    return collect_aot_jobs(DEFAULT_CSVS, parse_csv)


def _job_target(aot_arch: str, cu_num: int) -> RocmTarget:
    """Resolve one explicit AOT target without querying a live device."""

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


def _positive_job_int(cfg: dict, name: str) -> int:
    value = cfg[name]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _runtime_token_num(cfg: dict) -> int:
    value = cfg.get("token_num", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"token_num must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"token_num must be non-negative, got {value!r}")
    return value if value > 0 else _positive_job_int(cfg, "tile_m")


def _routing_capacity(cfg: dict) -> tuple[int, int]:
    """Mirror the runtime sorter's padded row and block capacities."""

    tokens = _runtime_token_num(cfg)
    topk = _positive_job_int(cfg, "topk")
    experts = _positive_job_int(cfg, "experts")
    tile_m = _positive_job_int(cfg, "tile_m")
    block_m = cfg.get("block_m", 0)
    if isinstance(block_m, bool) or not isinstance(block_m, int) or block_m < 0:
        raise ValueError(f"block_m must be a non-negative integer, got {block_m!r}")
    sort_block_m = block_m
    if sort_block_m <= 0:
        configured = cfg.get("sort_block_m", 0)
        if isinstance(configured, bool) or not isinstance(configured, int):
            raise TypeError(f"sort_block_m must be an integer, got {configured!r}")
        sort_block_m = configured or tile_m
    if sort_block_m <= 0:
        raise ValueError(
            f"sort_block_m must resolve to a positive integer, got {sort_block_m!r}"
        )
    padded_rows = tokens * topk + experts * sort_block_m - topk
    routing_blocks = (padded_rows + sort_block_m - 1) // sort_block_m
    return padded_rows, routing_blocks


def _stage1_requests(cfg: dict, context):
    """Create the same Stage1 requests the runtime path resolves."""

    from aiter.ops.flydsl.moe_compile_requests import stage1_compile_requests

    cfg = dict(cfg)
    token_num = _runtime_token_num(cfg)
    cfg["token_num"] = token_num
    a_dtype = cfg["a_dtype"]
    b_dtype = cfg["b_dtype"]
    if a_dtype == "fp8" and b_dtype == "fp4":
        cfg["tile_n"] = resolve_flydsl_stage1_tile_n(cfg["inter_dim"], cfg["tile_n"])
    padded_rows, routing_blocks = _routing_capacity(cfg)
    dense_blocks = min(
        token_num * int(cfg["topk"]) * int(cfg["tile_m"]),
        padded_rows,
    ) // int(cfg["tile_m"])
    grid_y = min(dense_blocks, routing_blocks)
    cfg["persist_m"] = resolve_flydsl_grid_y_persist_m(
        grid_y, int(cfg.get("persist_m", 0) or 0)
    )
    return stage1_compile_requests(
        cfg,
        context.target,
        registry=context.registry,
    )


def _stage2_requests(cfg: dict, context):
    """Create the same Stage2 GEMM/reduction requests as runtime."""

    from aiter.ops.flydsl.moe_compile_requests import (
        Stage2RuntimeMetadata,
        stage2_compile_requests,
    )

    cfg = dict(cfg)
    token_num = _runtime_token_num(cfg)
    cfg["token_num"] = token_num
    a_dtype = cfg["a_dtype"]
    b_dtype = cfg["b_dtype"]
    is_a16w_mix = a_dtype == "bf16" and b_dtype in ("fp4", "int4")
    if not is_a16w_mix:
        cfg["tile_k"] = resolve_flydsl_stage2_tile_k(cfg["inter_dim"], cfg["tile_k"])
    cfg["doweight_stage2"] = not bool(cfg["doweight_stage1"])
    requested_mode = cfg.get("mode", "atomic")
    mode = (
        "reduce"
        if requested_mode == "reduce"
        or requires_flydsl_stage2_reduce(token_num, cfg["model_dim"], 2)
        else "atomic"
    )
    if is_a16w_mix and b_dtype != "int4":
        # The current a16w4 port only has an atomic epilogue; this is also what
        # the runtime wrapper selects for bf16 x fp4.
        mode = "atomic"
    accumulate = mode != "reduce"
    _, routing_blocks = _routing_capacity(cfg)
    storage_bytes = {
        "fp4": token_num * int(cfg["topk"]) * int(cfg["inter_dim"]) // 2,
        "fp8": token_num * int(cfg["topk"]) * int(cfg["inter_dim"]),
        "bf16": token_num * int(cfg["topk"]) * int(cfg["inter_dim"]) * 2,
    }[a_dtype]
    cfg["use_global_a"] = storage_bytes >= (1 << 32)
    fp8_intermediate = (
        not accumulate
        and not is_a16w_mix
        and b_dtype in ("fp4", "fp8")
        and os.environ.get("AITER_FLYDSL_STAGE2_FP8", "0") == "1"
    )
    reduction_input_dtype = (
        "fp8"
        if fp8_intermediate
        else {"fp16": "f16", "half": "f16"}.get(cfg["out_dtype"], cfg["out_dtype"])
    )
    runtime_metadata = Stage2RuntimeMetadata(
        mode=mode,
        accumulate=accumulate,
        return_per_slot=False,
        persist=cfg.get("persist"),
        token_num=token_num,
        routing_block_count=routing_blocks,
        dtype_str=reduction_input_dtype,
        use_mask=False,
        topk_ids_available=False,
        num_experts=0,
        fp8_intermediate=fp8_intermediate,
        out_dtype_str={"fp16": "f16", "half": "f16"}.get(
            cfg["out_dtype"], cfg["out_dtype"]
        ),
        use_weight=False,
        scale_blk=_S2_LEGACY_FP8_SCALE_BLK if fp8_intermediate else None,
        pitch_align=_S2_LEGACY_FP8_PITCH_ALIGN if fp8_intermediate else None,
    )
    return stage2_compile_requests(
        cfg,
        runtime_metadata,
        context.target,
        registry=context.registry,
    )


def _epilogue_requests(cfg: dict, context):
    from aiter.ops.flydsl.moe_compile_requests import (
        cktile_epilogue_compile_requests,
    )

    return cktile_epilogue_compile_requests(
        {
            **cfg,
            "split_k": 2,
            "post_activation_layout": "interleaved",
            "enable_bias": False,
        },
        context.target,
        registry=context.registry,
    )


def build_moe_compile_requests(cfg: Mapping[str, object], context):
    """Build every compile request for one normalized MoE AOT job."""

    if not isinstance(cfg, Mapping):
        raise TypeError(f"cfg must be a mapping, got {type(cfg).__name__}")
    stage = cfg["stage"]
    if isinstance(stage, bool) or stage not in (1, 2, "epilogue"):
        raise ValueError(f"unsupported MoE AOT stage: {stage!r}")
    if stage == 1:
        return _stage1_requests(cfg, context)
    if stage == 2:
        return _stage2_requests(cfg, context)
    return _epilogue_requests(cfg, context)


def compile_moe_job(job: Mapping[str, object]):
    """Compile one normalized MoE job and return its generated artifacts."""

    if not isinstance(job, Mapping):
        raise TypeError(f"job must be a mapping, got {type(job).__name__}")
    cfg = dict(job)
    stage = cfg.get("stage")
    if stage in (1, 2):
        cfg["token_num"] = _runtime_token_num(cfg)
    cu_num = cfg.get("cu_num", 0)
    if isinstance(cu_num, bool) or not isinstance(cu_num, int):
        raise TypeError(f"cu_num must be an integer, got {cu_num!r}")
    aot_arch = cu_num_to_arch(cu_num, default=MOE_AOT_ARCH_DEFAULT)
    context = create_compile_context(_job_target(aot_arch, cu_num))
    requests = build_moe_compile_requests(cfg, context)
    return tuple(compile_aot(request, context=context) for request in requests)


def compile_one_config(
    kernel_name: str,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    cu_num: int = 0,
    **kwargs,
) -> dict:
    """Compile one MoE configuration through the shared runtime request path."""
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

    cfg = {
        "kernel_name": kernel_name,
        "stage": stage,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "experts": experts,
        "topk": topk,
        "cu_num": cu_num,
        **kwargs,
    }

    t0 = time.time()
    try:
        artifacts = compile_moe_job(cfg)
        elapsed = time.time() - t0
        result["compile_time"] = elapsed
        result["compile_requests"] = len(artifacts)
        print(
            f"  [OK] compile  {elapsed:6.1f}s  {shape_str}  "
            f"arch={aot_arch} requests={len(artifacts)}"
        )
    except Exception as e:  # noqa: BLE001
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
