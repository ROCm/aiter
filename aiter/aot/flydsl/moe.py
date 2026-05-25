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
    _get_compiled_silu_fused,
    _run_compiled,
    _s1_args_fp4,
    _s1_args_std,
    _s2_args_fp4,
    _s2_args_std,
    compile_flydsl_moe_stage1,
    compile_flydsl_moe_stage2,
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


def _row_optional_int(row: dict[str, str], name: str) -> int:
    value = row.get(name)
    if value is None:
        return 0
    value = str(value).strip()
    if value == "":
        return 0
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"{name} must be an int, got {value!r}") from e


def _row_optional_bool(row: dict[str, str], name: str) -> bool | None:
    value = row.get(name)
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    value_lower = value.lower()
    if value_lower in ("1", "true", "t", "yes", "y"):
        return True
    if value_lower in ("0", "false", "f", "no", "n"):
        return False
    raise ValueError(f"{name} must be a bool, got {value!r}")


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
            hidden_pad = _row_optional_int(row, "hidden_pad")
            intermediate_pad = _row_optional_int(row, "intermediate_pad")
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
            bias_supported = (
                q_type.strip().split(".")[-1] == "per_1x32"
                and dtype in ("torch.bfloat16", "torch.float16")
                and "float4_e2m1fn_x2" in q_dtype_w
            )
            bias = _row_optional_bool(row, "bias")
            if bias is not None:
                enable_bias_options = [bias]
            elif bias_supported:
                enable_bias_options = [False, True]
            else:
                enable_bias_options = [False]

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
                        "hidden_pad": hidden_pad,
                        "intermediate_pad": intermediate_pad,
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
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
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

            _need_fp4 = out_dtype == "fp4"
            _need_fp8 = out_dtype == "fp8"
            _fuse_any_quant = _need_fp4 or _need_fp8
            _base_out_dtype = "bf16" if _fuse_any_quant else out_dtype
            _is_splitk = k_batch > 1
            _splitk_fp4 = _is_splitk and _need_fp4
            _gui_sk = gate_mode == "interleave" and _is_splitk
            _gui_sk_fused = _gui_sk and _fuse_any_quant
            _gemm_out_dtype = _base_out_dtype if _is_splitk else out_dtype
            _gemm_out_torch_dtype = (
                torch.bfloat16 if _gemm_out_dtype == "bf16" else torch.float16
            )

            if _is_splitk:
                tmp_out = torch.zeros(
                    (tokens, topk, inter_dim * 2),
                    dtype=_gemm_out_torch_dtype,
                    device=dev,
                )
                out = (
                    torch.zeros(
                        (tokens, topk, inter_dim // 2), device=dev, dtype=torch.uint8
                    )
                    if _need_fp4
                    else (
                        torch.zeros(
                            (tokens, topk, inter_dim), device=dev, dtype=torch.uint8
                        )
                        if _need_fp8
                        else torch.empty(
                            (tokens, topk, inter_dim),
                            device=dev,
                            dtype=_gemm_out_torch_dtype,
                        )
                    )
                )
            else:
                tmp_out = None
                out = (
                    torch.empty(
                        (tokens, topk, inter_dim // 2), device=dev, dtype=torch.uint8
                    )
                    if _need_fp4
                    else (
                        torch.empty(
                            (tokens, topk, inter_dim), device=dev, dtype=torch.uint8
                        )
                        if _need_fp8
                        else torch.empty(
                            (tokens, topk, inter_dim),
                            device=dev,
                            dtype=_gemm_out_torch_dtype,
                        )
                    )
                )

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

            flat_a_scale = (
                a1_scale.view(-1)
                if a1_scale is not None
                else torch.empty(0, device=dev)
            )
            flat_w_scale = (
                w1_scale.view(-1)
                if w1_scale is not None
                else torch.empty(0, device=dev)
            )
            sw_arg = (
                sw
                if sw is not None
                else torch.empty(0, device=dev, dtype=torch.float32)
            )
            _grid_y = min(max_num_m_blocks, tokens * topk)
            _kernel_out = tmp_out if _is_splitk else out
            kernel_bias = None if _is_splitk else bias
            _n_in = inter_dim * 2 if is_fp4_weight else inter_dim
            _k_in = model_dim

            scale_cols = inter_dim // 32
            sorted_size = max(max_num_tokens_padded, max_num_m_blocks * tile_m)
            padded_rows = (sorted_size + 255) // 256 * 256
            padded_cols = (scale_cols + 7) // 8 * 8
            out_scale_sorted_flat = (
                torch.empty(padded_rows * padded_cols, dtype=torch.uint8, device=dev)
                if (_fuse_any_quant or _splitk_fp4 or _gui_sk_fused)
                else torch.empty(0, dtype=torch.uint8, device=dev)
            )

            if is_fp4_weight:
                args = _s1_args_fp4(
                    _kernel_out.view(-1),
                    a.view(-1),
                    w1.view(-1),
                    flat_a_scale,
                    flat_w_scale,
                    sorted_token_ids,
                    sorted_expert_ids,
                    sw_arg,
                    num_valid_ids,
                    out_scale_sorted_flat.view(-1),
                    tokens,
                    _n_in,
                    _k_in,
                    _grid_y,
                    dev,
                    bias=(
                        kernel_bias.view(-1)
                        if kernel_bias is not None
                        else torch.empty(0, device=dev)
                    ),
                    stream=0,
                )
            else:
                args = _s1_args_std(
                    _kernel_out.view(-1),
                    a.view(-1),
                    w1.view(-1),
                    flat_a_scale,
                    flat_w_scale,
                    sorted_token_ids,
                    sorted_expert_ids,
                    sw_arg,
                    num_valid_ids,
                    tokens,
                    _n_in,
                    _k_in,
                    _grid_y,
                    stream=0,
                )

            exe = compile_flydsl_moe_stage1(
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=E,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                doweight_stage1=(sw is not None),
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                out_dtype=_gemm_out_dtype,
                act=act,
                use_async_copy=True,
                k_batch=k_batch,
                waves_per_eu=waves_per_eu,
                b_nt=b_nt,
                gate_mode=gate_mode,
                enable_bias=(kernel_bias is not None),
                a_scale_one=a_scale_one,
                xcd_swizzle=xcd_swizzle,
                swiglu_limit=swiglu_limit,
                model_dim_pad=hidden_pad,
                inter_dim_pad=intermediate_pad,
            )
            _run_compiled(exe, args)

            if _gui_sk_fused or _gui_sk or _splitk_fp4:
                if _gui_sk_fused:
                    quant_mode = "fp4" if _need_fp4 else "fp8"
                    gui_layout = True
                elif _gui_sk:
                    quant_mode = "none"
                    gui_layout = True
                else:
                    quant_mode = "fp4"
                    gui_layout = False
                silu_fused = _get_compiled_silu_fused(
                    inter_dim,
                    topk,
                    quant_mode,
                    gui_layout=gui_layout,
                    act=act,
                    enable_bias=False,
                    swiglu_limit=swiglu_limit,
                )
                _run_compiled(
                    silu_fused,
                    (
                        tmp_out.view(-1, inter_dim * 2),
                        out.view(-1).view(torch.uint8),
                        out_scale_sorted_flat,
                        sorted_token_ids,
                        num_valid_ids,
                        sorted_token_ids.view(-1),
                        torch.empty(0, device=dev, dtype=torch.float32),
                        tokens,
                        sorted_token_ids.shape[0],
                        0,
                    ),
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

            torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
            accumulate = mode != "reduce"
            out = torch.zeros((tokens, model_dim), dtype=torch_out_dtype, device=dev)
            target = out
            if not accumulate:
                target = torch.empty(
                    (tokens * topk * model_dim,),
                    device=dev,
                    dtype=torch_out_dtype,
                )

            flat_a_scale = (
                a2_scale.view(-1)
                if a2_scale is not None
                else torch.empty(0, device=dev)
            )
            flat_w_scale = (
                w2_scale.view(-1)
                if w2_scale is not None
                else torch.empty(0, device=dev)
            )
            sw_arg = (
                sw
                if sw is not None
                else torch.empty(
                    sorted_token_ids.shape, dtype=torch.float32, device=dev
                )
            )

            _sbm = sort_block_m if sort_block_m > 0 else tile_m
            if _sbm == tile_m:
                m_blocks = min(sorted_expert_ids.shape[0], tokens * topk)
            else:
                total_sorted = sorted_expert_ids.shape[0] * _sbm
                m_blocks = (total_sorted + tile_m - 1) // tile_m
            if persist is True:
                _persist_m = -1
            elif persist is False:
                _persist_m = 4 if m_blocks > 256 else 1
            else:
                _persist_m = -1 if m_blocks > 256 else 1
            if a_dtype == "fp8":
                _persist_m = 1

            _n_in = model_dim
            _k_in = inter_dim

            if is_fp4_weight:
                args = _s2_args_fp4(
                    target,
                    a,
                    w2,
                    flat_a_scale,
                    flat_w_scale,
                    sorted_token_ids,
                    sorted_expert_ids,
                    sw_arg,
                    num_valid_ids,
                    tokens,
                    _n_in,
                    _k_in,
                    m_blocks,
                    dev,
                    bias=bias,
                    stream=0,
                )
            else:
                args = _s2_args_std(
                    target,
                    a,
                    w2,
                    flat_a_scale,
                    flat_w_scale,
                    sorted_token_ids,
                    sorted_expert_ids,
                    sw_arg,
                    num_valid_ids,
                    tokens,
                    _n_in,
                    _k_in,
                    m_blocks,
                    stream=0,
                )

            exe = compile_flydsl_moe_stage2(
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=E,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                doweight_stage2=(sw is not None),
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                out_dtype=out_dtype,
                accumulate=accumulate,
                persist_m=_persist_m,
                sort_block_m=sort_block_m,
                b_nt=b_nt,
                xcd_swizzle=xcd_swizzle,
                enable_bias=enable_bias,
                model_dim_pad=hidden_pad,
                inter_dim_pad=intermediate_pad,
            )
            _run_compiled(exe, args)


def compile_one_config(
    kernel_name: str,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    cu_num: int = 0,
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
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
        f"hidden_pad={hidden_pad} "
        f"intermediate_pad={intermediate_pad} "
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
                hidden_pad=hidden_pad,
                intermediate_pad=intermediate_pad,
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
