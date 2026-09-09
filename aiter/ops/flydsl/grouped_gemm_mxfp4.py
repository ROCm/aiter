# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Grouped MXFP4 GEMM launchers."""

from __future__ import annotations

import math
import os

import torch

from .kernels.mega_moe_gfx1250.types import Stage2ScatterContext
from .kernels.tensor_shim import ptr_arg

_SUPPORTED_CLUSTER_N = (4, 3, 2)


def _read_quad_cluster() -> tuple[int, int]:
    """2-D cluster (cluster_m x cluster_n) of the quadrant-pipeline a4w4 kernel.

    Defaults to the tuned 4x4. ``AITER_A4W4_QUAD_CLUSTER_M`` / ``_N`` override it
    for sweeps. The mcast masks are 32-bit, so ``cluster_m * cluster_n <= 32`` is
    the representability limit; a larger or non-positive request raises here
    rather than deadlocking a cluster the hardware cannot form. Whether the
    hardware can actually co-schedule the requested cluster is a separate,
    ungated question -- some valid-on-paper shapes still hang.
    """
    m = int(os.environ.get("AITER_A4W4_QUAD_CLUSTER_M", "4"))
    n = int(os.environ.get("AITER_A4W4_QUAD_CLUSTER_N", "4"))
    if m < 1 or n < 1:
        raise ValueError(f"AITER_A4W4_QUAD_CLUSTER_{{M,N}} must be >= 1, got {m}x{n}")
    if m * n > 32:
        raise ValueError(
            f"AITER_A4W4_QUAD_CLUSTER {m}x{n} exceeds the 32-workgroup mcast-mask "
            "limit (cluster_m*cluster_n must be <= 32)"
        )
    return (m, n)


# 2-D cluster of the quadrant-pipeline a4w4 kernel (cluster_m x cluster_n).
A4W4_QUAD_CLUSTER = _read_quad_cluster()

# XCD-aware tile order for the quad kernel: 0 = off (raw row-major map, the
# shipped behaviour), >0 = on, and the value is the group width in M-clusters.
# Off by default because it changes the tile->workgroup map for every launch;
# turn it on per run to A/B it. 16 mirrors the TILES_PER_GROUP that
# mxfp4_preshuffle_gfx1250_tdm uses.
A4W4_QUAD_XCD_SWIZZLE = int(os.environ.get("AITER_A4W4_XCD_SWIZZLE", "0"))


def a4w4_quad_pipeline_ok(
    *,
    a_is_fp4,
    tile_m,
    tile_n,
    tile_k,
    m_warp,
    n_warp,
    num_buffers,
    N,
    enable_ep_scatter=False,
) -> bool:
    """Whether the quadrant-pipeline a4w4 kernel can serve this launch.

    Says nothing about the contiguous-M alignment its B multicast needs -- that
    is the MoE driver's to arrange, and it is why the caller passes
    ``quad_pipeline`` explicitly instead of this being re-derived down here.
    """
    if os.environ.get("AITER_A4W4_QUAD_PIPELINE", "1") == "0":
        return False
    if not a_is_fp4 or enable_ep_scatter:
        return False
    from .kernels.gemm_a4w4_moe_gfx1250 import supports

    if not supports(tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers):
        return False
    cluster_m, cluster_n = A4W4_QUAD_CLUSTER
    n_tiles = (int(N) + int(tile_n) - 1) // int(tile_n)
    # grid.y must fill the cluster exactly, or a cluster never forms.
    return n_tiles % cluster_n == 0 and cluster_m > 0


def _select_next_stage_prefetch(csv_next_stage_prefetch: int) -> int:
    """Selects the environment override or the CSV setting."""
    value = os.environ.get("AITER_TDM_NEXT_STAGE_PREFETCH")
    if value is None:
        return int(bool(csv_next_stage_prefetch))
    value = value.strip()
    if value not in ("0", "1"):
        raise ValueError("AITER_TDM_NEXT_STAGE_PREFETCH must be 0 or 1")
    return int(value)


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
    stage2_scatter: Stage2ScatterContext | None = None,
    ep_destination_stride=0,
    ep_row_map=None,
    situ_beta=1.0,
    situ_linear_beta=1.0,
    quad_pipeline=False,
):
    """Launches a contiguous-M grouped a8w4 GEMM on the TDM kernel.

    ``quad_pipeline`` routes an a4w4 launch to the quadrant-pipeline kernel with
    its 2-D cluster. Its B multicast fans one weight load across ``cluster_m``
    M-tiles, so it is only correct when every expert's contiguous-M block is
    aligned to ``tile_m * cluster_m`` rows. The caller owns that alignment and
    therefore owns this flag; it is never inferred here.
    """
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
    if cluster_n > 1 and n_tiles % cluster_n:
        raise ValueError(
            f"[grouped-moe tdm] cluster_n={cluster_n} needs n_tiles={n_tiles} "
            f"(N={N}, tile_n={tile_n}) to be an exact multiple"
        )
    enable_ep_scatter = stage2_scatter is not None
    ep_row_map_tensor = ep_row_map if ep_row_map is not None else out

    if quad_pipeline:
        if not a4w4_quad_pipeline_ok(
            a_is_fp4=a_is_fp4,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            m_warp=m_warp,
            n_warp=n_warp,
            num_buffers=num_buffers,
            N=N,
            enable_ep_scatter=enable_ep_scatter,
        ):
            raise ValueError(
                "quad_pipeline requested but this launch does not qualify "
                f"(a_is_fp4={a_is_fp4} tile={tile_m}x{tile_n}x{tile_k} "
                f"warps={m_warp}x{n_warp} buffers={num_buffers} N={N})"
            )
        from .kernels.gemm_a4w4_moe_gfx1250 import launch_gemm_a4w4_moe

        cluster_m, cluster_n_2d = A4W4_QUAD_CLUSTER
        if os.environ.get("AITER_A4W4_LOG_GRID"):
            # cluster_m sets the per-expert contiguous-M alignment, so it also
            # sets how many padding M tiles the grid carries. Comparing two
            # cluster shapes on time alone is only fair alongside these.
            print(
                f"[a4w4-grid] cluster={cluster_m}x{cluster_n_2d} K={K} N={N} "
                f"contiguous_m={int(contiguous_m)} "
                f"mtiles={-(-int(contiguous_m) // tile_m)} "
                f"ntiles={-(-int(N) // tile_n)}",
                flush=True,
            )
        launch_gemm_a4w4_moe(
            out,
            ptr_arg(a),
            ptr_arg(w),
            a_scales.view(torch.int32),
            w_scales.view(torch.int32),
            ptr_arg(m_tile_map),
            bias_ptr,
            quant_scale_tensor,
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
            n_experts,
            stage1_act,
            has_bias,
            float(swiglu_limit),
            stage1_quant_out,
            quant_wmma_rep,
            cluster_m,
            cluster_n_2d,
            f32_situ_beta=float(situ_beta),
            f32_situ_linear_beta=float(situ_linear_beta),
            # An infinite limit makes the gpt-oss clamp a no-op, but it arrives
            # as a runtime kernel argument, so only the host can fold it away.
            act_has_limit=int(math.isfinite(float(swiglu_limit))),
            xcd_swizzle=A4W4_QUAD_XCD_SWIZZLE,
        )
        return out

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
