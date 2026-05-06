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
    compile_flydsl_moe_stage1,
    compile_flydsl_moe_stage2,
    get_flydsl_kernel_params,
    _get_compiled_silu_fused,
    _get_compiled_swiglu,
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
MOE_AOT_ARCH_DEFAULT = "gfx950"
BF16_FP8_MOE_BOUND_ENV = "AITER_BF16_FP8_MOE_BOUND"


def _csv_leaf(value: str) -> str:
    return value.strip().split(".")[-1]


def _is_per1x32_fp4_weight(q_type_leaf: str, q_dtype_w_leaf: str) -> bool:
    return q_type_leaf == "per_1x32" and q_dtype_w_leaf == "float4_e2m1fn_x2"


def _default_ksplit(
    token: int,
    topk: int,
    experts: int,
    inter_dim: int,
    model_dim: int,
    cu_num: int,
) -> int:
    env_ksplit = int(os.environ.get("AITER_KSPLIT", "0"))
    if env_ksplit != 0:
        return env_ksplit
    if token * topk > experts:
        return 0
    if cu_num <= 0:
        return 0

    tile_n = 128
    tg_m = token * topk
    tg_n = (inter_dim + tile_n - 1) // tile_n
    tg_num = tg_n * tg_m
    if tg_num >= cu_num:
        return 0

    tile_k = 256
    split_max = (cu_num + tg_num - 1) // tg_num
    for i in reversed(range(2, split_max + 1)):
        if (model_dim % i == 0) and ((model_dim // i) % tile_k == 0):
            return i
    return 0


def _runtime_switched_to_bf16_a_for_swiglu(
    token: int,
    cu_num: int,
    q_type_leaf: str,
    q_dtype_w_leaf: str,
    act: str,
) -> bool:
    if act != "swiglu":
        return False
    if not _is_per1x32_fp4_weight(q_type_leaf, q_dtype_w_leaf):
        return False

    arch = cu_num_to_arch(cu_num, default=MOE_AOT_ARCH_DEFAULT)
    bf16_fp8_bound = int(os.environ.get(BF16_FP8_MOE_BOUND_ENV, "256"))
    return arch != "gfx950" or token < bf16_fp8_bound


def _needs_cktile_swiglu_post_job(
    *,
    has_flydsl_kernel: bool,
    token: int,
    topk: int,
    experts: int,
    inter_dim: int,
    model_dim: int,
    cu_num: int,
    q_type_leaf: str,
    q_dtype_w_leaf: str,
    act: str,
) -> bool:
    """Mirror the default CKTile dispatch path that calls FlyDSL swiglu post-op.

    Some CSV cases contain FlyDSL kernel names, but runtime switches small-token
    Swiglu per_1x32 FP4-weight cases to BF16 activations before config lookup.
    That can miss the tuned FlyDSL row and fall back to cktile_moe_stage1, which
    invokes launch_swiglu_and_mul when split-K is enabled.
    """
    if not has_flydsl_kernel:
        return False
    if not _runtime_switched_to_bf16_a_for_swiglu(
        token,
        cu_num,
        q_type_leaf,
        q_dtype_w_leaf,
        act,
    ):
        return False
    return _default_ksplit(token, topk, experts, inter_dim, model_dim, cu_num) > 1


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
            q_type = row.get("q_type", "")
            q_dtype_a = row.get("q_dtype_a", "")
            q_dtype_w = row.get("q_dtype_w", "")
            act = (
                "swiglu"
                if _csv_leaf(act_type).lower() == "swiglu"
                else "silu"
            )
            dtype = row.get("dtype", "")
            q_type_leaf = _csv_leaf(q_type)
            q_dtype_a_leaf = _csv_leaf(q_dtype_a)
            q_dtype_w_leaf = _csv_leaf(q_dtype_w)
            if (
                _is_per1x32_fp4_weight(q_type_leaf, q_dtype_w_leaf)
                and q_dtype_a_leaf
                in ("bfloat16", "float16", "float8_e4m3fn", "float8_e4m3fnuz")
            ):
                # Match test/runtime dispatch: per_1x32 BF16/FP8 + FP4 weight
                # rows use Swiglu even when the CSV act_type column says Silu.
                act = "swiglu"
            # TODO: Replace this GPT-OSS-specific bias inference with an
            # explicit CSV/config field once the model config can express it.
            enable_bias = (
                act == "swiglu"
                and q_type_leaf == "per_1x32"
                and dtype in ("torch.bfloat16", "torch.float16")
            )

            has_flydsl_kernel = any(
                row.get(col, "").strip().startswith("flydsl_")
                for col in ("kernelName1", "kernelName2")
            )

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

                if params["stage"] == 2:
                    job["token_num"] = token
                    job["block_m"] = block_m
                elif (
                    params["stage"] == 1
                    and params.get("k_batch", 1) > 1
                ):
                    # Split-K stage1 materializes token-dependent temporary
                    # tensors, so the AOT cache key must match runtime token.
                    job["token_num"] = token

                full_job = {**job, **params}
                key = job_identity(full_job)
                if key in seen:
                    continue
                seen.add(key)

                jobs.append(full_job)

            if _needs_cktile_swiglu_post_job(
                has_flydsl_kernel=has_flydsl_kernel,
                token=token,
                topk=topk,
                experts=experts,
                inter_dim=inter_dim,
                model_dim=model_dim,
                cu_num=cu_num,
                q_type_leaf=q_type_leaf,
                q_dtype_w_leaf=q_dtype_w_leaf,
                act=act,
            ):
                post_job = {
                    "kernel_name": "launch_swiglu_and_mul",
                    "stage": 0,
                    "model_dim": model_dim,
                    "inter_dim": inter_dim,
                    "experts": experts,
                    "topk": topk,
                    "doweight_stage1": doweight_stage1,
                    "cu_num": cu_num,
                    "act": act,
                    "enable_bias": enable_bias,
                    "token_num": token,
                    "tile_m": 1,
                    "tile_n": 0,
                    "tile_k": 0,
                    "k_batch": 1,
                    "b_dtype": "fp4",
                    "out_dtype": "bf16",
                }
                key = job_identity(post_job)
                if key not in seen:
                    seen.add(key)
                    jobs.append(post_job)

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
    is_fp4_weight = b_dtype == "fp4"
    tokens = tile_m
    E = experts

    def _storage_numel(element_count: int, dtype: str) -> int:
        return element_count // 2 if dtype == "fp4" else element_count

    def _storage_dtype(dtype: str):
        if dtype in ("fp4", "fp8"):
            return torch.uint8
        if dtype in ("fp16", "f16"):
            return torch.float16
        if dtype == "bf16":
            return torch.bfloat16
        return torch.int8

    def _aot_token_count() -> int:
        return token_num if token_num > 0 else tokens

    def _aot_num_rows() -> int:
        return _aot_token_count() * topk

    def _sorting_block_m() -> int:
        return block_m if block_m > 0 else tile_m

    def _aot_sort_blocks() -> int:
        sorting_block_m = _sorting_block_m()
        return (_aot_sorted_len() + sorting_block_m - 1) // sorting_block_m

    def _aot_sorted_size() -> int:
        return max(_aot_sorted_len(), _aot_sort_blocks() * tile_m)

    def _aot_sorted_len() -> int:
        return _aot_num_rows() + E * _sorting_block_m() - topk

    def _aot_stage2_m_blocks() -> int:
        _sbm = sort_block_m if sort_block_m > 0 else tile_m
        sort_blocks = _aot_sort_blocks()
        if _sbm == tile_m:
            return min(sort_blocks, _aot_num_rows())
        total_sorted = sort_blocks * _sbm
        return (total_sorted + tile_m - 1) // tile_m

    def _aot_stage1_grid_y() -> int:
        dense_blocks = min(_aot_num_rows() * tile_m, _aot_sorted_len()) // tile_m
        return min(dense_blocks, _aot_sort_blocks())

    def _aot_stage2_persist_m(m_blocks: int) -> int:
        if persist is True:
            persist_m = -1
        elif persist is False:
            persist_m = 4 if m_blocks > 256 else 1
        else:
            persist_m = -1 if m_blocks > 256 else 1

        if a_dtype == "fp8":
            persist_m = 1

        return persist_m

    def _precompile_silu_fused():
        is_splitk = k_batch > 1
        need_fp4 = out_dtype == "fp4"
        need_fp8 = out_dtype == "fp8"
        fuse_any_quant = need_fp4 or need_fp8
        gate_up_interleave = gate_mode == "interleave"
        splitk_fp4 = is_splitk and need_fp4
        gui_sk = gate_up_interleave and is_splitk
        gui_sk_fused = gui_sk and fuse_any_quant

        if gui_sk_fused:
            quant_mode = "fp4" if need_fp4 else "fp8"
            gui_layout = True
        elif gui_sk:
            quant_mode = "none"
            gui_layout = True
        elif splitk_fp4:
            quant_mode = "fp4"
            gui_layout = False
        else:
            return

        silu_fused = _get_compiled_silu_fused(
            inter_dim,
            topk,
            quant_mode,
            gui_layout,
        )
        num_rows = _aot_num_rows()
        sorted_len = _aot_sorted_len()
        sorted_size = _aot_sorted_size()
        padded_cols = ((inter_dim // 32) + 7) // 8 * 8
        scale_rows = (sorted_size + 255) // 256 * 256
        tmp_out = torch.zeros(
            num_rows * inter_dim * 2, device=dev, dtype=torch.bfloat16
        )
        out_buf = torch.zeros(
            (
                num_rows * inter_dim * 2
                if quant_mode == "none"
                else _storage_numel(num_rows * inter_dim, quant_mode)
            ),
            device=dev,
            dtype=torch.uint8,
        )
        out_scale_sorted = torch.zeros(
            scale_rows * padded_cols,
            device=dev,
            dtype=torch.uint8,
        )
        sorted_token_ids = torch.zeros(sorted_len, device=dev, dtype=torch.int32)
        num_valid = torch.zeros(1, device=dev, dtype=torch.int32)
        _run_compiled(
            silu_fused,
            (
                tmp_out.view(-1, inter_dim * 2),
                out_buf.view(-1),
                out_scale_sorted,
                sorted_token_ids,
                num_valid,
                _aot_token_count(),
                sorted_len,
                _stream,
            ),
        )

    def _precompile_swiglu():
        swiglu = _get_compiled_swiglu(inter_dim)
        num_rows = _aot_num_rows()
        tmp_out = torch.zeros(num_rows * inter_dim * 2, device=dev, dtype=torch.bfloat16)
        out = torch.zeros(num_rows * inter_dim, device=dev, dtype=torch.bfloat16)
        _run_compiled(
            swiglu,
            (
                tmp_out.view(-1, inter_dim * 2),
                out.view(-1, inter_dim),
                num_rows,
                _stream,
            ),
        )

    def _precompile_stage1_post_op_deps():
        is_splitk = k_batch > 1
        need_fp4 = out_dtype == "fp4"
        need_fp8 = out_dtype == "fp8"
        fuse_any_quant = need_fp4 or need_fp8
        gate_up_interleave = gate_mode == "interleave"
        splitk_fp4 = is_splitk and need_fp4
        gui_sk = gate_up_interleave and is_splitk
        gui_sk_fused = gui_sk and fuse_any_quant

        if gui_sk_fused:
            _precompile_silu_fused()
        elif gui_sk:
            _precompile_silu_fused()
        elif splitk_fp4:
            _precompile_silu_fused()
        elif is_splitk and act == "swiglu":
            _precompile_swiglu()

    # Dummy routing tensors (shape matters, data doesn't)
    sorted_ids = torch.zeros(_aot_sorted_len(), device=dev, dtype=torch.int32)
    sorted_expert_ids = torch.zeros(_aot_sort_blocks(), device=dev, dtype=torch.int32)
    num_valid_ids = torch.zeros(2, device=dev, dtype=torch.int32)
    sw = torch.zeros(_aot_sorted_len(), device=dev, dtype=torch.float32)

    _cu_num_str = str(cu_num) if cu_num > 0 else None
    with compile_only_env(), override_env("CU_NUM", _cu_num_str):
        # Clear cached CU count so get_cu_num() re-reads the env var.
        from aiter.jit.utils.chip_info import get_cu_num

        get_cu_num.cache_clear()

        if stage == 0:
            _precompile_swiglu()

        elif stage == 1:

            _is_splitk = k_batch > 1
            n_in = inter_dim * 2 if is_fp4_weight else inter_dim
            k_in = model_dim
            stage1_grid_y = _aot_stage1_grid_y()
            sorted_weights = (
                sw
                if doweight_stage1
                else torch.empty(0, device=dev, dtype=torch.float32)
            )

            if a_dtype == "bf16" and b_dtype == "int4":
                from aiter import dtypes

                out_elems = _aot_num_rows() * inter_dim * (2 if _is_splitk else 1)
                out = torch.zeros(out_elems, device=dev, dtype=torch.bfloat16)
                a = torch.zeros(
                    _aot_token_count() * model_dim,
                    device=dev,
                    dtype=torch.bfloat16,
                )
                w = torch.empty(E * 2 * inter_dim * model_dim, device=dev, dtype=dtypes.i4x2)
                a_scale = torch.empty(0, device=dev, dtype=torch.float32)
                w_scale = torch.zeros(E * 2 * inter_dim * (model_dim // 32), device=dev, dtype=torch.bfloat16)
                args = _s1_args_std(
                    out,
                    a,
                    w,
                    a_scale,
                    w_scale,
                    sorted_ids,
                    sorted_expert_ids,
                    sorted_weights,
                    num_valid_ids,
                    _aot_token_count(),
                    n_in,
                    k_in,
                    stage1_grid_y,
                    stream=_stream,
                )
            elif is_fp4_weight:
                gemm_out_dtype = (
                    "bf16" if _is_splitk and out_dtype in ("fp4", "fp8") else out_dtype
                )
                gemm_out_elems = _aot_num_rows() * inter_dim
                if _is_splitk:
                    gemm_out_elems *= 2
                out = torch.zeros(
                    _storage_numel(gemm_out_elems, gemm_out_dtype),
                    device=dev,
                    dtype=_storage_dtype(gemm_out_dtype),
                )
                a = torch.zeros(
                    _storage_numel(_aot_token_count() * model_dim, a_dtype),
                    device=dev,
                    dtype=_storage_dtype(a_dtype),
                )
                w = torch.zeros(
                    _storage_numel(E * 2 * inter_dim * model_dim, b_dtype),
                    device=dev,
                    dtype=_storage_dtype(b_dtype),
                )
                a_scale = (
                    torch.zeros(1, device=dev, dtype=torch.uint8)
                    if a_dtype in ("fp4", "fp8")
                    else torch.empty(0, device=dev, dtype=torch.float32)
                )
                w_scale = torch.zeros(1, device=dev, dtype=torch.uint8)
                out_scale = torch.zeros(1, device=dev, dtype=torch.uint8)
                bias = torch.zeros(1, device=dev, dtype=torch.float32)
                args = _s1_args_fp4(
                    out,
                    a,
                    w,
                    a_scale,
                    w_scale,
                    sorted_ids,
                    sorted_expert_ids,
                    sorted_weights,
                    num_valid_ids,
                    out_scale,
                    _aot_token_count(),
                    n_in,
                    k_in,
                    stage1_grid_y,
                    dev,
                    bias=bias if enable_bias else None,
                    stream=_stream,
                )
            else:
                out = torch.zeros(
                    _aot_num_rows() * inter_dim, device=dev, dtype=torch.bfloat16
                )
                a = torch.zeros(_aot_token_count() * model_dim, device=dev, dtype=torch.int8)
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
                    sorted_weights,
                    num_valid_ids,
                    _aot_token_count(),
                    n_in,
                    k_in,
                    stage1_grid_y,
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
                out_dtype=gemm_out_dtype if is_fp4_weight else out_dtype,
                act=act,
                use_async_copy=True,
                waves_per_eu=waves_per_eu,
                k_batch=k_batch,
                b_nt=b_nt,
                gate_mode=gate_mode,
                a_scale_one=a_scale_one,
                xcd_swizzle=xcd_swizzle,
                enable_bias=enable_bias,
            )
            _run_compiled(exe, args)
            _precompile_stage1_post_op_deps()

        elif stage == 2:

            accumulate = mode != "reduce"
            _m_blocks = _aot_stage2_m_blocks()
            _persist_m = _aot_stage2_persist_m(_m_blocks)
            n_in = model_dim
            k_in = inter_dim

            if a_dtype == "bf16" and b_dtype == "int4":
                from aiter import dtypes

                out = torch.zeros(
                    _aot_token_count() * model_dim,
                    device=dev,
                    dtype=torch.bfloat16,
                )
                a = torch.zeros(
                    _aot_num_rows() * inter_dim,
                    device=dev,
                    dtype=torch.bfloat16,
                )
                w = torch.empty(E * model_dim * inter_dim, device=dev, dtype=dtypes.i4x2)
                a_scale = torch.empty(0, device=dev, dtype=torch.float32)
                w_scale = torch.zeros(E * model_dim * (inter_dim // 32), device=dev, dtype=torch.bfloat16)
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
                    _aot_token_count(),
                    n_in,
                    k_in,
                    _m_blocks,
                    stream=_stream,
                )
            elif is_fp4_weight:
                out = torch.zeros(
                    _aot_token_count() * model_dim,
                    device=dev,
                    dtype=torch.bfloat16,
                )
                a = torch.zeros(
                    _storage_numel(_aot_num_rows() * inter_dim, a_dtype),
                    device=dev,
                    dtype=_storage_dtype(a_dtype),
                )
                w = torch.zeros(
                    _storage_numel(E * model_dim * inter_dim, b_dtype),
                    device=dev,
                    dtype=_storage_dtype(b_dtype),
                )
                a_scale = (
                    torch.zeros(1, device=dev, dtype=torch.uint8)
                    if a_dtype in ("fp4", "fp8")
                    else torch.empty(0, device=dev, dtype=torch.float32)
                )
                w_scale = torch.zeros(1, device=dev, dtype=torch.uint8)
                bias = torch.zeros(1, device=dev, dtype=torch.float32)
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
                    _aot_token_count(),
                    n_in,
                    k_in,
                    _m_blocks,
                    dev,
                    bias=bias if enable_bias else None,
                    stream=_stream,
                )
            else:
                out = torch.zeros(
                    _aot_token_count() * model_dim,
                    device=dev,
                    dtype=torch.bfloat16,
                )
                a = torch.zeros(_aot_num_rows() * inter_dim, device=dev, dtype=torch.int8)
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
                    _aot_token_count(),
                    n_in,
                    k_in,
                    _m_blocks,
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
                doweight_stage2=not doweight_stage1,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                out_dtype=out_dtype,
                accumulate=accumulate,
                persist_m=_persist_m,
                sort_block_m=sort_block_m,
                b_nt=b_nt,
                xcd_swizzle=xcd_swizzle,
                enable_bias=enable_bias,
            )
            _run_compiled(exe, args)


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

    t0 = time.time()
    try:
        with override_env("ARCH", aot_arch), override_env("FLYDSL_GPU_ARCH", aot_arch):
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
    post_jobs = [j for j in all_jobs if j["stage"] == 0]
    print("=" * 72)
    print("FlyDSL MoE AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:          {csv_path}")
    print(f"  Stage1 jobs:  {len(stage1_jobs)}")
    print(f"  Stage2 jobs:  {len(stage2_jobs)}")
    print(f"  Post-op jobs: {len(post_jobs)}")
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

    if post_jobs:
        print(f"\n--- Post-op ({len(post_jobs)} kernels) ---")
        for i, job in enumerate(post_jobs, 1):
            print(f"\n[{i}/{len(post_jobs)}] ", end="")
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
