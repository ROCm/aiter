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
    n_experts: int,
) -> bool:
    """Return whether this stage exactly matches an A-preshuffle tuned tile."""
    num_buffers = min(num_buffers, max(1, K // tile_k))
    n_tiles = (N + tile_n - 1) // tile_n
    cluster_n = _select_cluster_n(n_tiles, cluster_n)
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
            tile_m in (192, 256),
            (tile_n, tile_k, m_warp, n_warp, num_buffers) == (256, 256, 2, 2, 4),
            cluster_n == 4,
            next_stage_prefetch == 1,
        )
    )
    if not common:
        return False
    if stage1_act == 1:
        return K == 7168 and N in (4096, 6144)
    return stage1_act == 0 and N == 7168 and K in (2048, 3072)


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
    cluster_m=-1,
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
    if cluster_n > 1 and n_tiles % cluster_n:
        raise ValueError(
            f"[grouped-moe tdm] cluster_n={cluster_n} needs n_tiles={n_tiles} "
            f"(N={N}, tile_n={tile_n}) to be an exact multiple"
        )
    target_optimized = bool(a_preshuffle) and supports_gfx1250_a_preshuffle(
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        out_is_f16=out_is_f16,
        a_is_fp4=a_is_fp4,
        stage1_act=stage1_act,
        stage1_quant_out=stage1_quant_out,
        has_bias=has_bias,
        cluster_n=cluster_n,
        next_stage_prefetch=next_stage_prefetch,
        n_experts=n_experts,
    )
    if target_optimized:
        waves_per_tensor_tdm = 2

    enable_ep_scatter = stage2_scatter is not None
    if enable_ep_scatter and stage2_scatter.combine_quant_bits:
        # N is the payload plane's length and so the scale plane's base, which
        # has to land on a cache line; fp4 packs two elements to the byte and
        # needs twice the element alignment fp8 does. Checked here because N is
        # traced inside @flyc.jit, same as the cluster_n constraint above.
        n_align = 128 * (8 // stage2_scatter.combine_quant_bits)
        if N % n_align:
            raise ValueError(
                f"[grouped-moe tdm] combine_quant_bits="
                f"{stage2_scatter.combine_quant_bits} needs N % {n_align} == 0 to "
                f"keep the scale plane cache-line aligned, got N={N}"
            )

    use_optimized = target_optimized and not any(
        (
            enable_ep_scatter,
            bool(tdm_as_in_prologue),
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
            cluster_m,
            next_stage_prefetch,
            waves_per_tensor_tdm,
            _select_tdm_b_th(tdm_b_th),
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
        ep_quant_bits=(
            int(stage2_scatter.combine_quant_bits) if enable_ep_scatter else 0
        ),
        arg_ep_row_map=ep_row_map_tensor,
        f32_situ_beta=float(situ_beta),
        f32_situ_linear_beta=float(situ_linear_beta),
        row_major_ascale=int(row_major_ascale),
        a_row_stride_bytes=int(a_row_stride_bytes),
        a_scale_row_stride_bytes=int(a_scale_row_stride_bytes),
    )
    return out
