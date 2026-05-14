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
)
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.moe_kernels import (
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
            q_dtype_a = row.get("q_dtype_a", "")
            swiglu_limit = _row_swiglu_limit(row)
            # Match the RT condition in fused_moe.py / test_moe_2stage.py:
            #   _needs_swiglu_bias_support(dtype, q_type) and q_dtype_w == fp4x2
            #   AND the caller actually passes a bias tensor (bias1 is not None).
            # Bias is passed only when q_dtype_a is fp8 or bf16 (NOT fp4x2).
            # For pure a4w4 models (q_dtype_a == fp4x2), bias1=None regardless of
            # activation type (Silu or Swiglu), so enable_bias=False.
            enable_bias = (
                q_type.strip().split(".")[-1] == "per_1x32"
                and dtype in ("torch.bfloat16", "torch.float16")
                and "float4_e2m1fn_x2" in q_dtype_w
                and "float4_e2m1fn_x2" not in q_dtype_a  # fp8/bf16 activation only
            )

            # Detect stage1's fuse_quant from kernel suffix to align stage2's
            # a2_scale shape with what runtime actually passes.
            stage1_name = row.get("kernelName1", "").strip()
            stage1_params = (
                get_flydsl_kernel_params(stage1_name)
                if stage1_name.startswith("flydsl_")
                else None
            )
            stage1_out_dtype = stage1_params.get("out_dtype") if stage1_params else None

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
                    "act": act,
                    "enable_bias": enable_bias,
                }
                params = get_flydsl_kernel_params(name)
                if params is None:
                    print(f"  [WARN] Unknown kernel name: {name}, skipping")
                    continue

                job["token_num"] = token
                job["block_m"] = block_m
                job["swiglu_limit"] = swiglu_limit
                # Stage2 needs to know whether stage1 fuses fp4/fp8 quant —
                # this changes the shape of a2_scale (sorted scale buffer
                # vs separate quant call output).
                if params["stage"] == 2:
                    job["stage1_fuse_quant"] = (
                        stage1_out_dtype if stage1_out_dtype in ("fp4", "fp8") else None
                    )

                full_job = {**job, **params}
                key = job_identity(full_job)
                if key in seen:
                    continue
                seen.add(key)

                jobs.append(full_job)

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
    act: str = "silu",
    doweight_stage1: bool = False,
    waves_per_eu: int = 3,
    k_batch: int = 1,
    b_nt: int = 2,
    gate_mode: str = "separated",
    mode: str = "atomic",
    persist=None,
    sort_block_m: int = 0,
    cu_num: int = 0,
    token_num: int = 0,
    block_m: int = 0,
    a_scale_one: bool = False,
    xcd_swizzle: int = 0,
    enable_bias: bool = False,
    stage1_fuse_quant=None,
    swiglu_limit: float = 0.0,
    **kwargs,
):
    """Trigger MLIR compilation by calling the runtime stage1/stage2 entry points
    with dummy GPU tensors and ``COMPILE_ONLY=1``.

    Builds dummy inputs that exactly mirror the tensor shapes that
    ``fused_moe_2stages`` would pass into ``flydsl_moe_stage1`` /
    ``flydsl_moe_stage2`` for a given ``(token_num, model_dim, inter_dim, E,
    topk, a_dtype, b_dtype, ...)`` combination, then dispatches into the same
    runtime entry points used by the fused-MoE op.  ``COMPILE_ONLY=1`` causes
    the executor to compile and persist the artifact without launching a
    kernel.  This guarantees that the cache key written here equals the cache
    key the runtime will look up at inference time.
    """
    import torch

    dev = torch.device("cpu")
    is_fp4_weight = b_dtype == "fp4"
    is_int4_weight = b_dtype == "int4"
    tokens = token_num if token_num > 0 else tile_m
    E = experts
    _sort_block_m = sort_block_m if sort_block_m > 0 else tile_m
    _block_m_for_sort = block_m if block_m > 0 else _sort_block_m

    max_num_tokens_padded = tokens * topk + E * _block_m_for_sort - topk
    max_num_m_blocks = (
        max_num_tokens_padded + _block_m_for_sort - 1
    ) // _block_m_for_sort

    def _storage_dtype(dtype: str):
        if dtype in ("fp4", "fp8"):
            return torch.uint8
        if dtype in ("fp16", "f16"):
            return torch.float16
        if dtype == "bf16":
            return torch.bfloat16
        if dtype == "int4":
            return torch.int4 if hasattr(torch, "int4") else torch.uint8
        return torch.int8

    def _alloc(shape, dtype):
        # torch.zeros doesn't support sub-byte dtypes (int4); use empty for those.
        # Cache key only depends on shape+dtype+strides — values don't matter.
        if dtype == getattr(torch, "int4", None):
            return torch.empty(shape, device=dev, dtype=dtype)
        return torch.zeros(shape, device=dev, dtype=dtype)

    def _user_a_shape():
        # User-level activation shape: (token_num, model_dim) in storage dtype.
        if a_dtype == "fp4":
            return (tokens, model_dim // 2)
        return (tokens, model_dim)

    def _user_w1_shape():
        # User-level w1 shape: (E, 2*inter_dim, model_dim) in storage dtype.
        if b_dtype == "fp4":
            return (E, 2 * inter_dim, model_dim // 2)
        if b_dtype == "int4":
            # int4 packed: 2 elements per byte
            return (E, 2 * inter_dim, model_dim // 2)
        return (E, 2 * inter_dim, model_dim)

    def _user_w2_shape():
        # User-level w2 shape: (E, model_dim, inter_dim) in storage dtype.
        if b_dtype == "fp4":
            return (E, model_dim, inter_dim // 2)
        if b_dtype == "int4":
            return (E, model_dim, inter_dim // 2)
        return (E, model_dim, inter_dim)

    def _make_routing():
        sorted_token_ids = torch.zeros(
            max_num_tokens_padded, device=dev, dtype=torch.int32
        )
        sorted_expert_ids = torch.zeros(max_num_m_blocks, device=dev, dtype=torch.int32)
        num_valid_ids = torch.zeros(2, device=dev, dtype=torch.int32)
        return sorted_token_ids, sorted_expert_ids, num_valid_ids

    def _make_sorted_weights(doweight: bool):
        if doweight:
            return torch.zeros(max_num_tokens_padded, device=dev, dtype=torch.float32)
        return None

    def _make_a1_scale():
        """Mirror fused_moe_2stages a1_scale construction (per_1x32 + fp4-weight path)."""
        if not is_fp4_weight:
            if is_int4_weight:
                # a16wi4: bf16 activations, int4 weights — no activation scale.
                return None
            return None
        if a_dtype == "fp8":
            if a_scale_one:
                # fused_moe_2stages: metadata.fuse_quant == "fp8"
                return torch.empty(0, dtype=torch.uint8, device=dev)
            # fused_moe_2stages line 1501
            return torch.ones(
                [max_num_tokens_padded, model_dim // 32],
                dtype=torch.uint8,
                device=dev,
            )
        if a_dtype == "bf16":
            return torch.ones(
                [max_num_tokens_padded, model_dim // 32],
                dtype=torch.uint8,
                device=dev,
            )
        if a_dtype == "fp4":
            # fused_dynamic_mxfp4_quant_moe_sort or mxfp4_moe_sort_fwd:
            # output shape is ((sorted_ids+31)//32*32, (cols+31)//32) in fp8_e8m0.
            rows = (max_num_tokens_padded + 31) // 32 * 32
            cols = (model_dim + 31) // 32
            return torch.zeros(rows * cols, dtype=torch.uint8, device=dev)
        return None

    def _make_a2_scale_for_stage2():
        """Stage2 a2_scale construction per fused_moe_2stages.

        When upstream stage1 fuses fp4/fp8 quant (``stage1_fuse_quant`` set),
        stage2 receives stage1's ``out_scale_sorted`` buffer directly — that
        buffer is padded to 256 rows and 8 cols.  Otherwise stage2 quantizes
        its own input and the resulting sorted scale uses 32-row alignment.
        """
        if not is_fp4_weight:
            return None
        if stage1_fuse_quant in ("fp4", "fp8"):
            # mirror flydsl_moe_stage1's out_scale_sorted_flat allocation:
            #   sorted_size = max(sorted_token_ids.shape[0],
            #                     sorted_expert_ids.shape[0] * sort_block_m)
            #   padded_rows = (sorted_size + 255) // 256 * 256
            #   padded_cols = (inter_dim // 32 + 7) // 8 * 8
            _sorted_size = max(
                max_num_tokens_padded,
                max_num_m_blocks * tile_m,
            )
            _padded_rows = (_sorted_size + 255) // 256 * 256
            _padded_cols = ((inter_dim // 32) + 7) // 8 * 8
            return torch.zeros(
                _padded_rows * _padded_cols, dtype=torch.uint8, device=dev
            )
        if a_dtype == "fp8":
            if act == "silu" and swiglu_limit == 0.0:
                # fused_moe_2stages uses fused_quant_fp8_sort for this path.
                rows = (max_num_tokens_padded + 31) // 32 * 32
                cols = (inter_dim + 31) // 32
                return torch.zeros(rows * cols, dtype=torch.uint8, device=dev)

            # Otherwise fused_moe_2stages reuses a1_scale for stage2.
            return torch.ones(
                [max_num_tokens_padded, model_dim // 32],
                dtype=torch.uint8,
                device=dev,
            )
        if a_dtype == "fp4":
            # fused_dynamic_mxfp4_quant_moe_sort / mxfp4_moe_sort_fwd path:
            # 32-row alignment.
            rows = (max_num_tokens_padded + 31) // 32 * 32
            cols = (inter_dim + 31) // 32
            return torch.zeros(rows * cols, dtype=torch.uint8, device=dev)
        if a_dtype == "bf16":
            return None
        return None

    def _make_w_scale(scale_storage_numel: int):
        # mxfp4 e8m0 scale — viewed as uint8 by _view_safe before kernel launch.
        return torch.zeros(scale_storage_numel, dtype=torch.uint8, device=dev)

    def _make_a_user(a_dtype_user_shape):
        return _alloc(a_dtype_user_shape, _storage_dtype(a_dtype))

    _cu_num_str = str(cu_num) if cu_num > 0 else None
    with compile_only_env(), override_env("CU_NUM", _cu_num_str):
        from aiter.jit.utils.chip_info import get_cu_num

        get_cu_num.cache_clear()

        sorted_token_ids, sorted_expert_ids, num_valid_ids = _make_routing()

        if stage == 1:
            a = _make_a_user(_user_a_shape())
            w1_shape = _user_w1_shape()
            w1 = _alloc(w1_shape, _storage_dtype(b_dtype))

            a1_scale = _make_a1_scale()
            # w1_scale: per-32 group along K dimension. Storage size in bytes.
            if is_fp4_weight:
                w1_scale = _make_w_scale(E * 2 * inter_dim * (model_dim // 32))
            elif is_int4_weight:
                # a16wi4: bf16 groupwise scale over (E, K//32, N).
                w1_scale = torch.zeros(
                    E * (model_dim // 32) * (2 * inter_dim),
                    device=dev,
                    dtype=torch.bfloat16,
                )
            else:
                w1_scale = torch.zeros(1, device=dev, dtype=torch.float32)

            sw = _make_sorted_weights(doweight_stage1)
            bias = (
                torch.zeros(E * inter_dim * 2, device=dev, dtype=torch.float32)
                if enable_bias
                else None
            )

            # Topk_ids is only used by silu_fused for the bias-broadcast path; an
            # int32 (token_num*topk,) tensor is enough for cache-key purposes.
            topk_ids = torch.zeros(tokens * topk, device=dev, dtype=torch.int32)

            flydsl_moe_stage1(
                a=a,
                w1=w1,
                sorted_token_ids=sorted_token_ids,
                sorted_expert_ids=sorted_expert_ids,
                num_valid_ids=num_valid_ids,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                out_dtype=out_dtype,
                act=act,
                w1_scale=w1_scale,
                a1_scale=a1_scale,
                sorted_weights=sw,
                use_async_copy=True,
                k_batch=k_batch,
                waves_per_eu=waves_per_eu,
                b_nt=b_nt,
                gate_mode=gate_mode,
                bias=bias,
                topk_ids=topk_ids if enable_bias else None,
                a_scale_one=a_scale_one,
                xcd_swizzle=xcd_swizzle,
                swiglu_limit=swiglu_limit,
            )

        elif stage == 2:
            # Stage2 input is (token_num, topk, inter_dim) in a_dtype storage.
            if a_dtype == "fp4":
                a_shape = (tokens, topk, inter_dim // 2)
            else:
                a_shape = (tokens, topk, inter_dim)
            a = _alloc(a_shape, _storage_dtype(a_dtype))
            w2_shape = _user_w2_shape()
            w2 = _alloc(w2_shape, _storage_dtype(b_dtype))

            a2_scale = _make_a2_scale_for_stage2()
            if is_fp4_weight:
                w2_scale = _make_w_scale(E * model_dim * (inter_dim // 32))
            elif is_int4_weight:
                w2_scale = torch.zeros(
                    E * (inter_dim // 32) * model_dim,
                    device=dev,
                    dtype=torch.bfloat16,
                )
            else:
                w2_scale = torch.zeros(1, device=dev, dtype=torch.float32)

            sw = _make_sorted_weights(not doweight_stage1)
            bias = (
                torch.zeros(E * model_dim, device=dev, dtype=torch.float32)
                if enable_bias
                else None
            )

            flydsl_moe_stage2(
                inter_states=a,
                w2=w2,
                sorted_token_ids=sorted_token_ids,
                sorted_expert_ids=sorted_expert_ids,
                num_valid_ids=num_valid_ids,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                out_dtype=out_dtype,
                mode=mode,
                w2_scale=w2_scale,
                a2_scale=a2_scale,
                sorted_weights=sw,
                sort_block_m=sort_block_m,
                persist=persist,
                b_nt=b_nt,
                xcd_swizzle=xcd_swizzle,
                bias=bias,
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

    Uses COMPILE_ONLY=1 with dummy tensors to trigger MLIR compilation and
    pkl cache write without depending on HIP ops or executing on GPU.

    Returns a dict with timing info.
    """
    aot_arch = cu_num_to_arch(cu_num, default=MOE_AOT_ARCH_DEFAULT)
    shape_str = (
        f"{kernel_name}  "
        f"model_dim={model_dim} inter_dim={inter_dim} "
        f"E={experts} topk={topk}"
    )
    result = {
        "kernel_name": kernel_name,
        "shape": shape_str,
        "compile_time": None,
        "compile_arch": aot_arch,
    }

    from torch._subclasses.fake_tensor import FakeTensorMode

    t0 = time.time()
    try:
        with override_env("ARCH", aot_arch), override_env(
            "FLYDSL_GPU_ARCH", aot_arch
        ), FakeTensorMode():
            _precompile_to_cache(
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                cu_num=cu_num,
                **kwargs,
            )
        elapsed = time.time() - t0
        result["compile_time"] = elapsed
        print(f"  [OK] compile  {elapsed:6.1f}s  {shape_str}  arch={aot_arch}")
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
    print("=" * 72)
    print("FlyDSL MoE AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:          {csv_path}")
    print(f"  Stage1 jobs:  {len(stage1_jobs)}")
    print(f"  Stage2 jobs:  {len(stage2_jobs)}")
    print(f"  Total jobs:   {len(all_jobs)}")
    print("  Compile arch: (from cu_num)")
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
