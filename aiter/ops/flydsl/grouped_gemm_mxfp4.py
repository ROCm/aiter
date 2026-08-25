# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Grouped MXFP4 GEMM launchers."""

from __future__ import annotations

import os

import torch

from .kernels.mega_moe_gfx1250.types import Stage2ScatterContext
from .kernels.tensor_shim import ptr_arg

_SUPPORTED_CLUSTER_N = (4, 3, 2)


def _select_next_stage_prefetch(csv_next_stage_prefetch: int) -> int:
    """Selects the environment override or the CSV setting."""
    value = os.environ.get("AITER_TDM_NEXT_STAGE_PREFETCH")
    if value is None:
        return int(bool(csv_next_stage_prefetch))
    value = value.strip()
    if value not in ("0", "1"):
        raise ValueError("AITER_TDM_NEXT_STAGE_PREFETCH must be 0 or 1")
    return int(value)


def _select_as_in_prologue(csv_as_in_prologue: int) -> int:
    """Selects the environment override or the CSV setting."""
    value = os.environ.get("AITER_GROUPED_GEMM_AS_PROLOGUE")
    if value is None:
        return int(bool(csv_as_in_prologue))
    value = value.strip()
    if value not in ("0", "1"):
        raise ValueError("AITER_GROUPED_GEMM_AS_PROLOGUE must be 0 or 1")
    return int(value)


def _select_b_prefetch(csv_b_prefetch: int) -> int:
    """Selects and validates the B page-table prefetch switch."""
    value = os.environ.get("AITER_TDM_B_PREFETCH")
    if value is None:
        return int(bool(csv_b_prefetch))
    value = value.strip()
    if value not in ("0", "1"):
        raise ValueError("AITER_TDM_B_PREFETCH must be 0 or 1")
    return int(value)


def _select_b_prefetch_scope() -> int:
    """Selects the B page-table prefetch cache policy."""
    scope = os.environ.get("AITER_TDM_B_PREFETCH_SCOPE", "se").strip().lower()
    if scope not in ("cu", "se"):
        raise ValueError("AITER_TDM_B_PREFETCH_SCOPE must be 'cu' or 'se'")
    default_th = "0" if scope == "cu" else "1"
    th = int(os.environ.get("AITER_TDM_B_PREFETCH_TH", default_th).strip())
    if not 0 <= th <= 6:
        raise ValueError("AITER_TDM_B_PREFETCH_TH must be between 0 and 6")
    if scope == "cu" and th:
        raise ValueError("AITER_TDM_B_PREFETCH_TH must be 0 for CU scope")
    return (0 if scope == "cu" else 8) | th


def _select_tdm_b_th(csv_tdm_b_th: int) -> int:
    """Selects the B-only TDM temporal hint."""
    value = os.environ.get("AITER_TDM_B_TH")
    th = int(csv_tdm_b_th) if value is None else int(value.strip())
    if not 0 <= th <= 6:
        raise ValueError("AITER_TDM_B_TH must be between 0 and 6")
    return th


def _select_max_tasks_per_worker(csv_max_tasks: int) -> int:
    """Selects the persistent-N task cap from the environment or CSV."""
    value = os.environ.get("AITER_FLYDSL_MAX_TASKS_PER_WORKER")
    try:
        max_tasks = int(value) if value is not None else int(csv_max_tasks)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "AITER_FLYDSL_MAX_TASKS_PER_WORKER must be a nonnegative integer"
        ) from exc
    if max_tasks < 0:
        raise ValueError("max_tasks_per_worker must be nonnegative")
    return max_tasks


def _select_cluster_n(n_tiles: int, csv_cluster_n: int) -> int:
    """Selects the environment override or CSV cluster degree."""
    env_cluster_n = os.environ.get("AITER_FLYDSL_MXFP4_CLUSTER_N")
    try:
        requested_cluster_n = (
            int(env_cluster_n) if env_cluster_n is not None else int(csv_cluster_n)
        )
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
    b_prefetch=0,
    tdm_b_th=1,
    max_tasks_per_worker=0,
    stage2_scatter: Stage2ScatterContext | None = None,
    ep_destination_stride=0,
    ep_row_map=None,
    situ_beta=1.0,
    situ_linear_beta=1.0,
):
    """Launches a contiguous-M grouped a8w4 GEMM on the TDM kernel."""
    from .kernels.mxfp4_preshuffle_gfx1250_tdm import launch_gemm_a8w4_tdm

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
    b_prefetch = _select_b_prefetch(b_prefetch)
    b_prefetch_scope = _select_b_prefetch_scope() if b_prefetch else 0
    tdm_b_th = _select_tdm_b_th(tdm_b_th)
    max_tasks_per_worker = _select_max_tasks_per_worker(max_tasks_per_worker)
    enable_ep_scatter = stage2_scatter is not None
    if max_tasks_per_worker and cluster_n > 1:
        raise ValueError("persistent-N requires cluster_n=1")
    if max_tasks_per_worker and enable_ep_scatter:
        raise ValueError("persistent-N does not yet support EP scatter")
    if b_prefetch and tile_n > N:
        raise ValueError(
            "B CU-prefetch requires tile_n <= N, got "
            f"tile_n={tile_n}, N={N}"
        )
    if cluster_n > 1 and n_tiles % cluster_n:
        raise ValueError(
            f"[grouped-moe tdm] cluster_n={cluster_n} needs n_tiles={n_tiles} "
            f"(N={N}, tile_n={tile_n}) to be an exact multiple"
        )
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
        _select_next_stage_prefetch(next_stage_prefetch),
        waves_per_tensor_tdm,
        _select_as_in_prologue(tdm_as_in_prologue),
        b_prefetch,
        b_prefetch_scope,
        tdm_b_th,
        max_tasks_per_worker,
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
    )
    return out
