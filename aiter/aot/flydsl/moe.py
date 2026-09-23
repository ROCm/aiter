#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for MoE / Mixed-MoE FlyDSL kernels from aiter CSV configs.

Reads tuned CSV config files (e.g. dsv3_fp4_tuned_fmoe.csv), extracts all
unique FlyDSL kernel names, and pre-compiles them into the cache.

Each job is compiled by calling the real runtime entry (the ``fused_moe`` /
FHMoE stage wrappers) with shape-only meta tensors under ``COMPILE_ONLY=1``.
AOT therefore runs exactly the runtime launch logic and cannot drift from it;
only the wrapper inputs (the shapes ``fused_moe`` allocates) are modelled here.

The default CSV set is resolved through ``AITER_CONFIGS`` so model-specific
tuned CSVs can be merged the same way as runtime JIT config lookup.

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
from contextlib import contextmanager
from unittest import mock

import torch

from aiter.aot.flydsl.common import (
    collect_aot_jobs,
    compile_only_env,
    cu_num_to_arch,
    job_identity,
    override_env,
    run_jobs_parallel,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params
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

            stage1_name = row.get("kernelName1", "").strip()
            stage2_name = row.get("kernelName2", "").strip()
            stage1_v2_output_layout = "_moe2_layout_" in stage2_name
            stage2_v2_params = (
                parse_flydsl_v2_gemm2_kernel(stage2_name)
                if stage1_v2_output_layout
                else None
            )

            # cktile_ stage1 runs a FlyDSL split-K post-activation epilogue that
            # the flydsl_-only loop below skips. Runtime has one only for
            # silu/swiglu (gelu is torch, anything else raises).
            if stage1_name.startswith("cktile_") and act_name in ("silu", "swiglu"):
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
                    }
                    if shared_expert_id >= 0:
                        job["shared_expert_id"] = shared_expert_id
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


def _meta(*shape, dtype=torch.uint8):
    # Shape/dtype only: the kernel cache key never depends on buffer contents.
    return torch.empty(shape, dtype=dtype, device="meta")


def _bucket_tokens(token: int) -> list[int]:
    """Smallest and largest token counts fused_moe maps onto a CSV token row.

    Token-dependent launch choices (persist_m, reduce promotion) are resolved
    at both ends so the whole bucket hits the AOT cache.
    """
    from aiter.fused_moe import _PADDED_M_TIERS

    if token < _PADDED_M_TIERS[0]:
        return sorted({token // 2 + 1, token})
    upper = [tier - 1 for tier in _PADDED_M_TIERS if tier > token]
    return [token, *upper[:1]]


def _stage_call(job: dict, tokens: int):
    """Return the runtime stage wrapper and the arguments fused_moe passes it."""
    from aiter import ActivationType, dtypes, fused_moe
    from aiter import fhmoe as fhmoe_ops

    stage, topk, experts = job["stage"], job["topk"], job["experts"]
    model_dim, inter_dim, tile_m = job["model_dim"], job["inter_dim"], job["tile_m"]
    a_dtype, b_dtype = job["a_dtype"], job["b_dtype"]
    act_dtype = {"fp4": torch.uint8, "fp8": dtypes.fp8, "bf16": torch.bfloat16}
    # moe_sorting's padded routing buffers (see fused_moe.moe_sorting).
    block_m = job.get("block_m", 0) or job.get("sort_block_m", 0) or tile_m
    blocks = (tokens * topk + experts * block_m - topk + block_m - 1) // block_m
    rows = blocks * block_m
    shared_expert_id = job.get("shared_expert_id", -1)
    kwargs = {
        "sorted_token_ids": _meta(rows, dtype=torch.int32),
        "sorted_expert_ids": _meta(blocks, dtype=torch.int32),
        "num_valid_ids": _meta(2, dtype=torch.int32),
        "topk": topk,
        "kernelName": job["kernel_name"],
        "topk_ids": _meta(tokens, topk, dtype=torch.int32),
    }
    if shared_expert_id >= 0:
        kwargs["shared_expert_id"] = shared_expert_id
    # fused_moe hands the sorted route weights to exactly one stage.
    weights = _meta(rows, dtype=torch.float32)
    if stage == 1:
        a_cols = model_dim // 2 if a_dtype == "fp4" else model_dim
        kwargs.update(
            hidden_states=_meta(tokens, a_cols, dtype=act_dtype[a_dtype]),
            w1=_meta(experts, 2 * inter_dim, 1),
            w2=None,
            out=None,
            activation={
                "silu": ActivationType.Silu,
                "swiglu": ActivationType.Swiglu,
                "situv2": ActivationType.Situv2,
            }[job["act"]],
            sorted_weights=weights if job["doweight_stage1"] else None,
            v2_output_layout=job.get("v2_output_layout", False),
        )
        if job.get("enable_bias"):
            kwargs["bias1"] = _meta(experts, 2 * inter_dim, dtype=torch.float32)
        if job.get("v2_output_layout"):
            kwargs["out_dtype"] = job["out_dtype"]
        if shared_expert_id >= 0:
            kwargs.update(shared_w1=_meta(1), shared_w1_scale=_meta(1))
            return fhmoe_ops._flydsl_fhmoe_stage1_wrapper, kwargs
        return fused_moe._flydsl_stage1_wrapper, kwargs

    if a_dtype == "bf16" and b_dtype in ("fp4", "int4"):
        # The a16w stage1 port emits a sorted [blocks * tile_m, inter_dim] layout.
        inter = _meta(blocks * tile_m, inter_dim, dtype=torch.bfloat16)
    else:
        cols = inter_dim // 2 if a_dtype == "fp4" else inter_dim
        inter = _meta(tokens, topk, cols, dtype=act_dtype[a_dtype])
    kwargs.update(
        inter_states=inter,
        w1=None,
        w2=_meta(experts, model_dim, 1),
        out=_meta(tokens, model_dim, dtype=torch.bfloat16),
        sorted_weights=None if job["doweight_stage1"] else weights,
    )
    if job.get("block_m"):
        kwargs["block_m"] = job["block_m"]
    if job.get("enable_bias"):
        kwargs["bias2"] = _meta(experts, model_dim, dtype=torch.float32)
    if shared_expert_id >= 0:
        kwargs.update(shared_w2=_meta(1), shared_w2_scale=_meta(1))
        return fhmoe_ops._flydsl_fhmoe_stage2_wrapper, kwargs
    return fused_moe._flydsl_stage2_wrapper, kwargs


def _compile_epilogue(job: dict):
    """Compile the CK-Tile split-K post-activation via its runtime entry."""
    from aiter.ops.flydsl.moe_kernels import (
        flydsl_silu_and_mul_interleaved,
        flydsl_swiglu_and_mul_interleaved,
    )

    inter_dim, rows = job["inter_dim"], 256
    x = _meta(rows, inter_dim * 2, dtype=torch.bfloat16)
    out = _meta(rows, inter_dim, dtype=torch.bfloat16)
    if job["act"] == "swiglu":
        flydsl_swiglu_and_mul_interleaved(x, out)
    else:
        flydsl_silu_and_mul_interleaved(
            x,
            out,
            _meta(rows, dtype=torch.int32),
            _meta(2, dtype=torch.int32),
            rows,
            job["topk"],
        )


@contextmanager
def _null_stream():
    # AOT workers are forked after CUDA init and cannot touch the device.
    # COMPILE_ONLY never launches, so the runtime's stream is a placeholder.
    with mock.patch.object(torch.cuda, "current_stream", lambda *_a, **_k: 0):
        yield


def compile_one_config(**job) -> dict:
    """Compile one MoE job into the FlyDSL cache through the runtime entry."""
    aot_arch = cu_num_to_arch(job.get("cu_num", 0), default=MOE_AOT_ARCH_DEFAULT)
    shape_str = (
        f"{job['kernel_name']}  model_dim={job['model_dim']} "
        f"inter_dim={job['inter_dim']} E={job['experts']} topk={job['topk']}"
    )
    result = {
        "kernel_name": job["kernel_name"],
        "shape": shape_str,
        "compile_time": None,
        "compile_arch": aot_arch,
    }
    t0 = time.time()
    try:
        with (
            override_env("FLYDSL_GPU_ARCH", aot_arch),
            compile_only_env(),
            _null_stream(),
        ):
            if job["stage"] == "epilogue":
                _compile_epilogue(job)
            else:
                token = job.get("token_num", 0) or job["tile_m"]
                for tokens in _bucket_tokens(token):
                    func, kwargs = _stage_call(job, tokens)
                    func(**kwargs)
        result["compile_time"] = time.time() - t0
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
