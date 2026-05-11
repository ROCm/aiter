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

    # GUI-only deployment (replaces every SEP fp4 stage1 row with its GUI twin).
    python -m aiter.aot.flydsl.moe --gui-policy gugu

    # Compat deployment (covers both SEP and GUI for every fp4 stage1 row).
    python -m aiter.aot.flydsl.moe --gui-policy gguu

Environment variables:
    FLYDSL_RUNTIME_CACHE_DIR  Cache directory (default: ~/.flydsl/cache)
    ARCH                      Target GPU architecture (e.g. gfx942, gfx950).
"""

import argparse
import csv
import os
import sys
import time
from typing import Optional

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
    csv_caller_gate_modes,
    get_flydsl_kernel_params,
    is_csv_fallback_row,
    swap_flydsl_stage1_kernel_for_gate_mode,
    _get_compiled_silu_fused,
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


def _row_to_common_meta(row):
    """Extract the per-row, gate_mode-independent compile metadata used by
    every job synthesized from this CSV row."""
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
    return {
        "token": token,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "experts": experts,
        "topk": topk,
        "doweight_stage1": doweight_stage1,
        "cu_num": cu_num,
        "block_m": block_m,
        "act": act,
        "enable_bias": enable_bias,
    }


def _stage1_kernel_for_gate_mode(literal_name: str, gate_mode: str) -> Optional[str]:
    """Resolve the stage1 kernel name that ``fused_moe`` will ACTUALLY
    dispatch when the caller requests ``gate_mode``, given the literal
    name pulled from the CSV row.

    - flydsl rows: route through ``swap_flydsl_stage1_kernel_for_gate_mode``
      so SEP<->GUI siblings are picked transparently.  Returns None if the
      requested gate_mode has no registered sibling for this row (e.g.
      ``_kb`` / ``_go`` SEP kernels have no GUI twin -- such rows are
      simply skipped under that gate_mode, matching the runtime which would
      keep using the SEP literal).
    - non-flydsl (ck / cktile / ...): not subject to flydsl SEP/GUI swap;
      AOT does not compile these (they live in a different backend), so
      return None to signal "skip".
    """
    if not literal_name.startswith("flydsl_"):
        return None
    if not literal_name.startswith("flydsl_moe1_"):
        # stage2 or other -- caller handles separately, not via swap
        return literal_name
    swapped = swap_flydsl_stage1_kernel_for_gate_mode(literal_name, gate_mode)
    sib = get_flydsl_kernel_params(swapped)
    if sib is None:
        return None
    # Kernels without an explicit ``gate_mode`` in their registry params
    # (e.g. wint4 stage1 from ``get_flydsl_stage1_kernels_int4_bf16`` --
    # not subject to SEP/GUI dispatch) are SEP-only by convention; honor
    # them when caller asks for ``separated`` and skip otherwise.
    sib_gm = sib.get("gate_mode", "separated")
    if sib_gm != gate_mode:
        return None  # no sibling for this gate_mode (e.g. SEP-only _kb under gugu)
    return swapped


def parse_csv(csv_path: str, *, caller_gate_modes_override: Optional[list] = None):
    """Parse the CSV and return a list of unique compile jobs.

    The job set produced for a CSV is the SAME as what
    ``op_tests/test_moe_2stage.py`` would iterate, by construction:

      1. Compute per-CSV ``caller_gate_modes`` via the shared
         ``csv_caller_gate_modes`` (ck / cktile / a4w4 pinning rules).
      2. For every row × every caller_gate_mode in the sweep:
           - stage1 (``kernelName1``): swap to the SEP/GUI sibling that
             would actually be dispatched at runtime for that gate_mode.
             Skip the (row, gate_mode) pair when no sibling exists.
           - stage2 (``kernelName2``): emitted as-is (stage2 is not
             SEP/GUI sensitive -- it shares the same kernel for both).
      3. Dedup by ``job_identity`` so siblings shared across rows compile
         once.

    ``caller_gate_modes_override`` lets ``--gui-policy gugu/gguu`` short-
    circuit the per-CSV decision (gugu -> [INTERLEAVE], gguu -> [SEP, INTL]).
    """
    jobs = []
    seen = set()

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if caller_gate_modes_override is not None:
        gate_modes = list(caller_gate_modes_override)
    else:
        gate_modes = csv_caller_gate_modes(rows)

    for row in rows:
        try:
            meta = _row_to_common_meta(row)
        except (KeyError, ValueError):
            continue

        kn1 = (row.get("kernelName1") or "").strip()
        kn2 = (row.get("kernelName2") or "").strip()
        is_fallback = is_csv_fallback_row(row)

        # ---- stage1: per-gate_mode sibling resolution ----------------
        if kn1.startswith("flydsl_moe1_"):
            for gm in gate_modes:
                effective = _stage1_kernel_for_gate_mode(kn1, gm)
                if effective is None:
                    if is_fallback:
                        # Fallback rows tolerate missing siblings silently
                        # (they're backups, not the deployment kernel).
                        continue
                    print(
                        f"  [skip] {kn1}: no '{gm}' sibling registered "
                        f"(row dropped under this gate_mode)"
                    )
                    continue
                params = get_flydsl_kernel_params(effective)
                if params is None:
                    print(f"  [WARN] Unknown kernel name: {effective}, skipping")
                    continue
                job = {
                    "kernel_name": effective,
                    "model_dim": meta["model_dim"],
                    "inter_dim": meta["inter_dim"],
                    "experts": meta["experts"],
                    "topk": meta["topk"],
                    "doweight_stage1": meta["doweight_stage1"],
                    "cu_num": meta["cu_num"],
                    "act": meta["act"],
                    "enable_bias": meta["enable_bias"],
                    **params,
                }
                key = job_identity(job)
                if key in seen:
                    continue
                seen.add(key)
                jobs.append(job)

        # ---- stage2: literal name, gate_mode-independent -------------
        if kn2.startswith("flydsl_moe2_"):
            params = get_flydsl_kernel_params(kn2)
            if params is None:
                print(f"  [WARN] Unknown kernel name: {kn2}, skipping")
            else:
                job = {
                    "kernel_name": kn2,
                    "model_dim": meta["model_dim"],
                    "inter_dim": meta["inter_dim"],
                    "experts": meta["experts"],
                    "topk": meta["topk"],
                    "doweight_stage1": meta["doweight_stage1"],
                    "cu_num": meta["cu_num"],
                    "act": meta["act"],
                    "enable_bias": meta["enable_bias"],
                    "token_num": meta["token"],
                    "block_m": meta["block_m"],
                    **params,
                }
                key = job_identity(job)
                if key not in seen:
                    seen.add(key)
                    jobs.append(job)

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
    _grid_y = 1

    def _storage_numel(element_count: int, dtype: str) -> int:
        return element_count // 2 if dtype == "fp4" else element_count

    def _storage_dtype(dtype: str):
        if dtype in ("fp4", "fp8"):
            return torch.uint8
        if dtype in ("fp16", "f16"):
            return torch.float16
        if dtype == "bf16":
            return torch.bfloat16
        if dtype == "int4":
            return torch.int4
        return torch.int8

    def _aot_sort_blocks() -> int:
        token_count = token_num if token_num > 0 else tokens
        sorting_block_m = block_m if block_m > 0 else tile_m
        max_tokens_padded = token_count * topk + E * sorting_block_m - topk
        return (max_tokens_padded + sorting_block_m - 1) // sorting_block_m

    def _aot_stage2_m_blocks() -> int:
        token_count = token_num if token_num > 0 else tokens
        _sbm = sort_block_m if sort_block_m > 0 else tile_m
        sort_blocks = _aot_sort_blocks()
        if _sbm == tile_m:
            return min(sort_blocks, token_count * topk)
        total_sorted = sort_blocks * _sbm
        return (total_sorted + tile_m - 1) // tile_m

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
            act=act,
            enable_bias=enable_bias,
        )
        sorted_len = max(tokens * topk, _aot_sort_blocks() * tile_m)
        padded_cols = ((inter_dim // 32) + 7) // 8 * 8
        scale_rows = (sorted_len + 255) // 256 * 256
        tmp_out = torch.zeros(
            tokens * topk * inter_dim * 2, device=dev, dtype=torch.bfloat16
        )
        out_buf = torch.zeros(
            (
                tokens * topk * inter_dim * 2
                if quant_mode == "none"
                else _storage_numel(tokens * topk * inter_dim, quant_mode)
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
        topk_ids = torch.zeros(tokens * topk, device=dev, dtype=torch.int32)
        bias = torch.zeros(E * inter_dim * 2, device=dev, dtype=torch.float32)
        _run_compiled(
            silu_fused,
            (
                tmp_out.view(-1, inter_dim * 2),
                out_buf.view(-1),
                out_scale_sorted,
                sorted_token_ids,
                num_valid,
                topk_ids,
                bias,
                tokens,
                sorted_len,
                _stream,
            ),
        )

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
            n_in = inter_dim * 2 if is_fp4_weight else inter_dim
            k_in = model_dim

            if is_fp4_weight:
                gemm_out_dtype = (
                    "bf16" if _is_splitk and out_dtype in ("fp4", "fp8") else out_dtype
                )
                gemm_out_elems = tokens * topk * inter_dim
                if _is_splitk:
                    gemm_out_elems *= 2
                out = torch.zeros(
                    _storage_numel(gemm_out_elems, gemm_out_dtype),
                    device=dev,
                    dtype=_storage_dtype(gemm_out_dtype),
                )
                a = torch.zeros(
                    _storage_numel(tokens * model_dim, a_dtype),
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
                    sw,
                    num_valid_ids,
                    out_scale,
                    tokens,
                    n_in,
                    k_in,
                    _grid_y,
                    dev,
                    bias=bias if enable_bias else None,
                    stream=_stream,
                )
            else:
                out = torch.zeros(
                    tokens * topk * inter_dim, device=dev, dtype=torch.bfloat16
                )
                # torch.zeros doesn't support int4 on CPU; use torch.empty for sub-byte types
                _a_dtype_torch = _storage_dtype(a_dtype)
                _b_dtype_torch = _storage_dtype(b_dtype)
                a = torch.empty(1, device=dev, dtype=_a_dtype_torch)
                w = torch.empty(1, device=dev, dtype=_b_dtype_torch)
                a_scale = torch.zeros(1, device=dev, dtype=torch.float32)
                # W4A16 groupwise scales are bf16 (scale_is_bf16=True in compile_moe_gemm1)
                w_scale = torch.zeros(1, device=dev, dtype=torch.bfloat16)
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
            if is_fp4_weight:
                _precompile_silu_fused()

        elif stage == 2:

            accumulate = mode != "reduce"
            _m_blocks = _aot_stage2_m_blocks()
            _persist_m = _aot_stage2_persist_m(_m_blocks)
            n_in = model_dim
            k_in = inter_dim

            if is_fp4_weight:
                out = torch.zeros(tokens * model_dim, device=dev, dtype=torch.bfloat16)
                a = torch.zeros(
                    _storage_numel(tokens * topk * inter_dim, a_dtype),
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
                    tokens,
                    n_in,
                    k_in,
                    _m_blocks,
                    dev,
                    bias=bias if enable_bias else None,
                    stream=_stream,
                )
            else:
                out = torch.zeros(tokens * model_dim, device=dev, dtype=torch.bfloat16)
                # torch.zeros doesn't support int4 on CPU; use torch.empty for sub-byte types
                _a_dtype_torch = _storage_dtype(a_dtype)
                _b_dtype_torch = _storage_dtype(b_dtype)
                a = torch.empty(1, device=dev, dtype=_a_dtype_torch)
                w = torch.empty(1, device=dev, dtype=_b_dtype_torch)
                a_scale = torch.zeros(1, device=dev, dtype=torch.float32)
                # W4A16 groupwise scales are bf16 (scale_is_bf16=True in compile_moe_gemm1)
                w_scale = torch.zeros(1, device=dev, dtype=torch.bfloat16)
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


def _gui_policy_to_override(policy: str):
    """Translate the CLI gui-policy into a ``caller_gate_modes_override``
    consumed by ``parse_csv``.

      - ``auto``: None -> ``parse_csv`` uses the per-CSV decision shared
        with ``op_tests/test_moe_2stage.py`` (ck/cktile/a4w4 pinning).
      - ``gugu``: force every fp4 stage1 row to compile its INTL sibling
        only (mimics ``AITER_MOE_GUI=gugu`` runtime).
      - ``gguu``: force every fp4 stage1 row to compile BOTH SEP and INTL
        (mimics ``AITER_MOE_GUI=gguu`` runtime; cache serves either
        caller_gate_mode).
    """
    if policy == "gugu":
        return ["interleave"]
    if policy == "gguu":
        return ["separated", "interleave"]
    return None  # auto


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
    parser.add_argument(
        "--gui-policy",
        choices=("auto", "gugu", "gguu"),
        default="auto",
        help=(
            "Stage1 GUI/SEP coverage policy for fp4-weight kernels:\n"
            "  auto (default) - per-CSV decision IDENTICAL to\n"
            "                   op_tests/test_moe_2stage.py:\n"
            "                     ck_*    -> SEP only,\n"
            "                     cktile_* -> INTL only,\n"
            "                     pure flydsl + a4w4 -> [SEP, INTL] sweep,\n"
            "                     pure flydsl, non-a4w4 -> majority _gui vote.\n"
            "                   AOT cache covers EXACTLY the kernel set the\n"
            "                   UT exercises.\n"
            "  gugu           - Override: GUI-only.  Compile every fp4\n"
            "                   stage1 row's INTL sibling.  Skip rows\n"
            "                   without a GUI sibling (e.g. _kb/_go).\n"
            "  gguu           - Override: compile BOTH SEP and INTL\n"
            "                   siblings of every fp4 stage1 row."
        ),
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

    override = _gui_policy_to_override(args.gui_policy)
    if override is not None:
        from functools import partial
        parser_fn = partial(parse_csv, caller_gate_modes_override=override)
    else:
        parser_fn = parse_csv
    all_jobs = collect_aot_jobs(csv_paths, parser_fn)

    stage1_jobs = [j for j in all_jobs if j["stage"] == 1]
    stage2_jobs = [j for j in all_jobs if j["stage"] == 2]
    print("=" * 72)
    print("FlyDSL MoE AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:          {csv_path}")
    print(f"  GUI policy:   {args.gui_policy}")
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
