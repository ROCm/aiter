# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Grouped MXFP4 GEMM launchers."""

from __future__ import annotations

import os

import torch

from .kernels.mega_moe_gfx1250.types import Stage2ScatterContext
from .kernels.tensor_shim import ptr_arg

_SUPPORTED_CLUSTER_N = (4, 3, 2)


def _select_bool_env(name: str, csv_value: int) -> int:
    """Select a strict 0/1 environment override or the CSV setting."""
    value = os.environ.get(name)
    if value is None:
        return int(bool(csv_value))
    value = value.strip()
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return int(value)


def _select_tdm_b_th(csv_tdm_b_th: int) -> int:
    """Selects the B-only TDM temporal hint."""
    value = os.environ.get("AITER_TDM_B_TH")
    temporal_hint = int(csv_tdm_b_th) if value is None else int(value.strip())
    if not 0 <= temporal_hint <= 6:
        raise ValueError("AITER_TDM_B_TH must be between 0 and 6")
    return temporal_hint


def _select_cluster_n(n_tiles: int, csv_cluster_n: int) -> int:
    """Selects the environment override or CSV cluster degree."""
    env_cluster_n = os.environ.get("AITER_FLYDSL_MXFP4_CLUSTER_N")
    try:
        if env_cluster_n is not None:
            requested_cluster_n = int(env_cluster_n)
        else:
            requested_cluster_n = int(csv_cluster_n)
    except (TypeError, ValueError) as exc:
        raise ValueError("AITER_FLYDSL_MXFP4_CLUSTER_N must be an integer") from exc
    if requested_cluster_n <= 1:
        return 1
    if requested_cluster_n not in _SUPPORTED_CLUSTER_N:
        return 1
    return requested_cluster_n if n_tiles % requested_cluster_n == 0 else 1


def _select_num_waves_per_tensor_tdm(csv_num_waves: int) -> int:
    """Selects the CSV value or falls back to the environment setting."""
    if csv_num_waves in (1, 2, 4):
        return csv_num_waves

    try:
        num_waves = int(os.environ.get("AITER_FLYDSL_NUM_WAVES_PER_TENSOR_TDM", "2"))
    except ValueError as exc:
        raise ValueError(
            "AITER_FLYDSL_NUM_WAVES_PER_TENSOR_TDM must be 1, 2, or 4"
        ) from exc
    if num_waves not in (1, 2, 4):
        raise ValueError(
            "AITER_FLYDSL_NUM_WAVES_PER_TENSOR_TDM must be 1, 2, or 4, got "
            f"{num_waves}"
        )
    return num_waves


def _select_gemm1_num_waves_per_tensor_tdm(default: int) -> int:
    """Select a GEMM1-only TDM wave override without changing GEMM2."""
    value = os.environ.get("AITER_FLYDSL_GEMM1_WAVES_PER_TENSOR_TDM")
    if value is None:
        return default
    try:
        num_waves = int(value)
    except ValueError as exc:
        raise ValueError(
            "AITER_FLYDSL_GEMM1_WAVES_PER_TENSOR_TDM must be 1, 2, or 4"
        ) from exc
    if num_waves not in (1, 2, 4):
        raise ValueError(
            "AITER_FLYDSL_GEMM1_WAVES_PER_TENSOR_TDM must be 1, 2, or 4"
        )
    return num_waves

def _select_gemm2_num_waves_per_tensor_tdm(default: int) -> int:
    """Select a GEMM2-only TDM owner count without changing GEMM1."""
    value = os.environ.get("AITER_FLYDSL_GEMM2_WAVES_PER_TENSOR_TDM")
    if value is None:
        return default
    try:
        num_waves = int(value)
    except ValueError as exc:
        raise ValueError(
            "AITER_FLYDSL_GEMM2_WAVES_PER_TENSOR_TDM must be 1, 2, or 4"
        ) from exc
    if num_waves not in (1, 2, 4):
        raise ValueError(
            "AITER_FLYDSL_GEMM2_WAVES_PER_TENSOR_TDM must be 1, 2, or 4"
        )
    return num_waves

def _select_gemm2_output_split_wm(default: int = 3) -> int:
    """Select the first GEMM2 output-TDM slice in logical WM rows."""
    try:
        split_wm = int(
            os.environ.get("AITER_FLYDSL_GEMM2_OUTPUT_SPLIT_WM", str(default))
        )
    except ValueError as exc:
        raise ValueError(
            "AITER_FLYDSL_GEMM2_OUTPUT_SPLIT_WM must be an integer from 1 to 7"
        ) from exc
    if split_wm not in range(1, 8):
        raise ValueError(
            "AITER_FLYDSL_GEMM2_OUTPUT_SPLIT_WM must be an integer from 1 to 7"
        )
    return split_wm

def _select_epilogue_batch_wn(default: int) -> int:
    """Selects the target GEMM1 SiLU epilogue batch width."""
    try:
        batch_wn = int(
            os.environ.get("AITER_FLYDSL_GEMM1_EPILOGUE_BATCH_WN", str(default))
        )
    except ValueError as exc:
        raise ValueError(
            "AITER_FLYDSL_GEMM1_EPILOGUE_BATCH_WN must be 1, 2, 4, or 8"
        ) from exc
    if batch_wn not in (1, 2, 4, 8):
        raise ValueError("AITER_FLYDSL_GEMM1_EPILOGUE_BATCH_WN must be 1, 2, 4, or 8")
    return batch_wn

def _select_schedule_hints(default: int) -> int:
    value = os.environ.get("AITER_FLYDSL_GEMM1_SCHEDULE_HINTS", str(default)).strip()
    if value not in ("0", "1"):
        raise ValueError("AITER_FLYDSL_GEMM1_SCHEDULE_HINTS must be 0 or 1")
    return int(value)

def _select_relax_cluster_wrap_dscnt(default: int) -> int:
    value = os.environ.get(
        "AITER_FLYDSL_GEMM1_RELAX_CLUSTER_WRAP_DSCNT", str(default)
    ).strip()
    if value not in ("0", "1"):
        raise ValueError("AITER_FLYDSL_GEMM1_RELAX_CLUSTER_WRAP_DSCNT must be 0 or 1")
    return int(value)

def _select_binary_int(name: str, default: int) -> int:
    value = os.environ.get(name, str(default)).strip()
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return int(value)

def _select_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value

def _select_tristate(name: str, default: int = -1) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be -1, 0, or 1") from exc
    if value not in (-1, 0, 1):
        raise ValueError(f"{name} must be -1, 0, or 1")
    return value

def _select_wmma_reuse(default: int = 0) -> int:
    try:
        value = int(os.environ.get("AITER_FLYDSL_GEMM1_WMMA_REUSE", str(default)))
    except ValueError as exc:
        raise ValueError("AITER_FLYDSL_GEMM1_WMMA_REUSE must be 0, 1, 2, or 3") from exc
    if value not in (0, 1, 2, 3):
        raise ValueError("AITER_FLYDSL_GEMM1_WMMA_REUSE must be 0, 1, 2, or 3")
    return value


def supports_gfx1250_a_preshuffle(
    *,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    num_buffers: int,
    out_is_f16: int,
    a_is_fp4: int,
    stage1_act: int,
    stage1_quant_out: int,
    has_bias: int,
    cluster_n: int,
    next_stage_prefetch: int,
    waves_per_tensor_tdm: int,
    n_experts: int,
) -> bool:
    """Return whether this stage exactly matches an A-preshuffle tuned tile."""
    num_buffers = min(num_buffers, max(1, K // tile_k))
    n_tiles = (N + tile_n - 1) // tile_n
    cluster_n = _select_cluster_n(n_tiles, cluster_n)
    waves_per_tensor_tdm = _select_num_waves_per_tensor_tdm(waves_per_tensor_tdm)
    next_stage_prefetch = _select_bool_env(
        "AITER_TDM_NEXT_STAGE_PREFETCH", next_stage_prefetch
    )
    common = all(
        (
            a_is_fp4,
            stage1_quant_out == 0,
            out_is_f16 == 0,
            has_bias == 0,
            n_experts > 0,
            (tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)
            == (256, 256, 256, 2, 2, 4),
            cluster_n == 4,
            next_stage_prefetch == 1,
        )
    )
    if not common:
        return False
    if stage1_act == 1:
        waves_per_tensor_tdm = _select_gemm1_num_waves_per_tensor_tdm(
            waves_per_tensor_tdm
        )
        return (
            K == 7168
            and N in (4096, 6144)
            and waves_per_tensor_tdm in (1, 2, 4)
        )
    if stage1_act == 0 and N == 7168 and K in (2048, 3072):
        waves_per_tensor_tdm = _select_gemm2_num_waves_per_tensor_tdm(
            waves_per_tensor_tdm
        )
        return waves_per_tensor_tdm in (1, 2)
    return False


def flydsl_grouped_gemm_a8w4_masked(
    out,
    a,
    w,
    a_scales,
    w_scales,
    m_tile_map,
    *,
    n_experts,
    contiguous_m,
    N,
    K,
    tile_m=64,
    tile_n=256,
    tile_k=256,
    m_warp=1,
    n_warp=4,
    num_buffers=3,
    out_is_f16=0,
    a_is_fp4=0,
    stage1_act=0,
    bias=None,
    swiglu_limit=7.0,
    stream=None,
    stage1_quant_out=0,
    quant_scale=None,
    quant_wmma_rep=1,
    cluster_n=-1,
    waves_per_tensor_tdm=-1,
    next_stage_prefetch=0,
    tdm_as_in_prologue=0,
    tdm_b_th=0,
    stage2_scatter: Stage2ScatterContext | None = None,
    ep_destination_stride=0,
    ep_row_map=None,
    situ_beta=1.0,
    situ_linear_beta=1.0,
    row_major_ascale=0,
    a_row_stride_bytes=0,
    a_scale_row_stride_bytes=0,
    a_preshuffle=0,
):
    """Launches a contiguous-M grouped a8w4 GEMM on the TDM kernel."""
    from .kernels.mxfp4_preshuffle_gfx1250_tdm import (
        launch_gemm_a8w4_tdm,
        launch_gemm_a8w4_tdm_optimized,
    )

    if stream is None:
        stream = torch.cuda.current_stream()
    if stage1_act == 3:
        if float(situ_beta) <= 0.0:
            raise ValueError(f"situ_beta must be > 0, got {situ_beta!r}")
        if float(situ_linear_beta) <= 0.0:
            raise ValueError(f"situ_linear_beta must be > 0, got {situ_linear_beta!r}")
    num_buffers = min(num_buffers, max(1, K // tile_k))
    has_bias = 1 if bias is not None else 0
    bias_ptr = ptr_arg(bias) if bias is not None else ptr_arg(a)
    quant_scale_tensor = out if quant_scale is None else quant_scale.view(torch.uint8)
    n_tiles = (N + tile_n - 1) // tile_n
    cluster_n = _select_cluster_n(n_tiles, cluster_n)
    waves_per_tensor_tdm = _select_num_waves_per_tensor_tdm(waves_per_tensor_tdm)
    next_stage_prefetch = _select_bool_env(
        "AITER_TDM_NEXT_STAGE_PREFETCH", next_stage_prefetch
    )
    target_gemm1_apre = all(
        (
            bool(a_preshuffle),
            a_is_fp4,
            K == 7168,
            tile_m == 256,
            tile_n == 256,
            tile_k == 256,
            m_warp == 2,
            n_warp == 2,
            num_buffers == 4,
            stage1_act == 1,
            stage1_quant_out == 0,
            out_is_f16 == 0,
            has_bias == 0,
            cluster_n == 4,
            next_stage_prefetch == 1,
            n_experts > 0,
        )
    )
    if target_gemm1_apre:
        waves_per_tensor_tdm = _select_gemm1_num_waves_per_tensor_tdm(
            waves_per_tensor_tdm
        )
    if cluster_n > 1 and n_tiles % cluster_n:
        raise ValueError(
            f"[grouped-moe tdm] cluster_n={cluster_n} needs n_tiles={n_tiles} "
            f"(N={N}, tile_n={tile_n}) to be an exact multiple"
        )
    target_fp4_prefill_common = all(
        (
            a_is_fp4,
            K == 7168,
            tile_m in (128, 256),
            tile_n in (128, 256),
            (m_warp, n_warp) in ((2, 2), (4, 2), (4, 4), (8, 2)),
            stage1_act == 1,
            stage1_quant_out == 0,
            out_is_f16 == 0,
            has_bias == 0,
            cluster_n == 4,
            next_stage_prefetch == 1,
            n_experts > 0,
        )
    )
    target_fp4_prefill = target_fp4_prefill_common and (
        (tile_k, num_buffers, waves_per_tensor_tdm)
        in (
            (128, 4, 1),
            (128, 4, 2),
            (128, 4, 4),
            (256, 4, 1),
            (256, 4, 2),
            (256, 4, 4),
            (256, 3, 1),
            (256, 3, 2),
            (256, 3, 4),
            (256, 2, 1),
            (512, 2, 1),
        )
    )
    target_gemm2 = all(
        (
            a_is_fp4,
            N == 7168,
            K in (2048, 3072),
            tile_n == 256,
            tile_k == 256,
            num_buffers == 4,
            tile_m == 256,
            m_warp == 2,
            n_warp == 2,
            stage1_act == 0,
            stage1_quant_out == 0,
            out_is_f16 == 0,
            has_bias == 0,
            cluster_n == 4,
            next_stage_prefetch == 1,
            n_experts > 0,
        )
    )
    if target_gemm2:
        waves_per_tensor_tdm = _select_gemm2_num_waves_per_tensor_tdm(
            waves_per_tensor_tdm
        )

    enable_ep_scatter = stage2_scatter is not None
    use_optimized = (target_fp4_prefill or target_gemm2) and not any(
        (
            enable_ep_scatter,
            bool(tdm_as_in_prologue),
            bool(tdm_b_th),
            bool(row_major_ascale),
            bool(a_row_stride_bytes),
            bool(a_scale_row_stride_bytes),
        )
    )
    if a_preshuffle and not use_optimized:
        raise ValueError(
            "A-preshuffled input is only supported by the retained gfx1250 "
            "GEMM1/GEMM2 optimized shapes"
        )
    if use_optimized:
        launch_gemm_a8w4_tdm_optimized(
            out,
            ptr_arg(a),
            ptr_arg(w),
            a_scales.view(torch.int32),
            w_scales.view(torch.int32),
            contiguous_m,
            stream,
            N,
            K,
            tile_m,
            tile_n,
            tile_k,
            m_warp,
            n_warp,
            out_is_f16,
            num_buffers,
            a_is_fp4,
            ptr_arg(m_tile_map),
            n_experts,
            stage1_act,
            has_bias,
            bias_ptr,
            float(swiglu_limit),
            stage1_quant_out,
            quant_wmma_rep,
            quant_scale_tensor,
            cluster_n,
            next_stage_prefetch,
            waves_per_tensor_tdm,
            float(situ_beta),
            float(situ_linear_beta),
            _select_epilogue_batch_wn(8 if target_fp4_prefill else 1),
            int(bool(a_preshuffle)),
            (
                _select_schedule_hints(1)
                if target_fp4_prefill
                else (
                    _select_binary_int("AITER_FLYDSL_GEMM2_SCHEDULE_HINTS", 0)
                    if target_gemm2
                    else 0
                )
            ),
            _select_relax_cluster_wrap_dscnt(1 if target_fp4_prefill else 0),
            _select_binary_int("AITER_FLYDSL_GEMM1_DIRECT_SCALES", 0)
            if target_fp4_prefill
            else 0,
            _select_binary_int("AITER_FLYDSL_GEMM1_TRANSITIVE_CLUSTER_SYNC", 0)
            if target_fp4_prefill
            else 0,
            _select_binary_int("AITER_FLYDSL_GEMM1_TDM_EARLY_TIMEOUT", 1),
            _select_binary_int("AITER_FLYDSL_GEMM1_M_MAJOR_SWIZZLE", 0)
            if target_fp4_prefill
            else 0,
            (
                _select_positive_int("AITER_FLYDSL_GEMM1_MMA_GROUP", 4)
                if target_fp4_prefill
                else _select_positive_int("AITER_FLYDSL_GEMM2_MMA_GROUP", 4)
            ),
            (
                _select_positive_int("AITER_FLYDSL_GEMM1_FENCE_COVER_MMA", 8)
                if target_fp4_prefill
                else _select_positive_int("AITER_FLYDSL_GEMM2_FENCE_COVER_MMA", 8)
            ),
            (
                _select_tristate("AITER_FLYDSL_GEMM1_DISABLE_XDL_ARB_STALL")
                if target_fp4_prefill
                else -1
            ),
            _select_binary_int("AITER_FLYDSL_GEMM1_SILU_POLY9", 0)
            if stage1_act == 1
            else 0,
            _select_binary_int("AITER_FLYDSL_GEMM1_SILU_HARD", 0)
            if stage1_act == 1
            else 0,
            _select_binary_int("AITER_FLYDSL_GEMM1_SILU_RELU", 0)
            if stage1_act == 1
            else 0,
            _select_wmma_reuse() if target_fp4_prefill else 0,
            _select_binary_int("AITER_FLYDSL_GEMM1_DELAY_ACC_ZERO", 0),
            (
                _select_binary_int("AITER_FLYDSL_GEMM1_OVERLAP_OUTPUT_STORE", 0)
                if target_fp4_prefill
                else _select_binary_int("AITER_FLYDSL_GEMM2_OVERLAP_OUTPUT_STORE", 0)
            ),
            (
                _select_gemm2_output_split_wm()
                if target_gemm2
                and _select_binary_int("AITER_FLYDSL_GEMM2_OVERLAP_OUTPUT_STORE", 0)
                else 0
            ),
            (
                _select_binary_int("AITER_FLYDSL_GEMM2_OUTPUT_WAVE_SPLIT", 0)
                if target_gemm2
                else 0
            ),
        )
        return out

    ep_row_map_tensor = ep_row_map if ep_row_map is not None else out
    launch_gemm_a8w4_tdm(
        out,
        ptr_arg(a),
        ptr_arg(w),
        a_scales.view(torch.int32),
        w_scales.view(torch.int32),
        contiguous_m,
        stream,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        out_is_f16,
        num_buffers,
        a_is_fp4,
        ptr_arg(m_tile_map),
        n_experts,
        stage1_act,
        has_bias,
        bias_ptr,
        float(swiglu_limit),
        stage1_quant_out,
        quant_wmma_rep,
        quant_scale_tensor,
        cluster_n,
        next_stage_prefetch,
        waves_per_tensor_tdm,
        _select_bool_env("AITER_GROUPED_GEMM_AS_PROLOGUE", tdm_as_in_prologue),
        _select_tdm_b_th(tdm_b_th),
        enable_ep_scatter=int(enable_ep_scatter),
        ep_arena_handle=(int(stage2_scatter.arena_handle) if enable_ep_scatter else 0),
        ep_combine_input_offset=(
            int(stage2_scatter.combine_input_offset) if enable_ep_scatter else 0
        ),
        ep_slot_stride_bytes=(
            int(stage2_scatter.slot_stride_bytes) if enable_ep_scatter else 0
        ),
        ep_destination_stride=int(ep_destination_stride),
        ep_world_size=int(stage2_scatter.world_size) if enable_ep_scatter else 0,
        arg_ep_row_map=ep_row_map_tensor,
        f32_situ_beta=float(situ_beta),
        f32_situ_linear_beta=float(situ_linear_beta),
        row_major_ascale=int(row_major_ascale),
        a_row_stride_bytes=int(a_row_stride_bytes),
        a_scale_row_stride_bytes=int(a_scale_row_stride_bytes),
    )
    return out
