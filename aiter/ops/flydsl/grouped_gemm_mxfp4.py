# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Grouped MXFP4 GEMM launchers."""

from __future__ import annotations

import os

import torch

from .kernels.mega_moe_gfx1250.types import Stage2ScatterContext
from .kernels.tensor_shim import _run_compiled, ptr_arg

_SUPPORTED_CLUSTER_M = (1, 2, 4)
_SUPPORTED_CLUSTER_N = (1, 2, 3, 4)


def _launch_address(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Rank-1 tensor at ``tensor``'s base address.

    The launcher cache must not specialize on buffer rank. ``view(-1)`` does
    that for a contiguous buffer. Compact FP4 scales are a strided window on
    the dispatch-wire pitch, so that view is rejected. The kernel steps rows
    with ``a_scale_row_stride_bytes`` and only needs the base pointer, which a
    unit-stride span of the pitched storage still provides.
    """
    if tensor.numel() == 0:
        empty = tensor if tensor.dtype == dtype else tensor.view(dtype)
        return empty.view(0)
    if not tensor.is_contiguous():
        last = 0
        for size, stride in zip(tensor.shape, tensor.stride()):
            if size > 1:
                last += (size - 1) * stride
        span = last + 1
        # view(dtype) needs the pitched span to be a whole number of elements.
        # The bytes in between rows are the wire padding, so rounding up to
        # the dtype stays inside the row pitch.
        byte_span = span * tensor.element_size()
        unit = dtype.itemsize
        if byte_span % unit:
            byte_span = (byte_span + unit - 1) // unit * unit
            span = byte_span // tensor.element_size()
        tensor = torch.as_strided(tensor, (span,), (1,))
    if tensor.dtype != dtype:
        tensor = tensor.view(dtype)
    return tensor.view(-1)


def _select_bool_env(name: str, csv_value: int) -> int:
    """Select a strict 0/1 environment override or the CSV setting."""
    value = os.environ.get(name)
    if value is None:
        return int(bool(csv_value))
    value = value.strip()
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return int(value)


def _a_gather_mode(a_row_index, a_index_bound) -> int:
    """Kernel ``a_gather`` bits: 1/2 = 32/16-bit indices, 4 = resident A-scale."""
    if a_row_index is None:
        return 0
    # 16-bit indices halve the gather op count and need every valid row id to
    # fit. At a bound of exactly 0x10000 (4 ranks x 16k tokens, the e2e arena)
    # a negative padding id wraps to the in-bounds row 0xFFFF instead of OOB;
    # padding rows are dropped by GEMM2's scatter, so their contents are moot.
    idx16 = 0 < int(a_index_bound) <= 0x10000
    return (2 if idx16 else 1) | 4


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


def _select_cluster_m(csv_cluster_m: int, cluster_n: int, stage1_act: int) -> int:
    """Selects the M cluster degree, preserving the legacy automatic policy."""
    env_cluster_m = os.environ.get("AITER_FLYDSL_MXFP4_CLUSTER_M")
    try:
        requested_cluster_m = (
            int(env_cluster_m) if env_cluster_m is not None else int(csv_cluster_m)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("AITER_FLYDSL_MXFP4_CLUSTER_M must be an integer") from exc

    if requested_cluster_m < 1:
        requested_cluster_m = 4 if stage1_act and cluster_n == 4 else 1
    if requested_cluster_m not in _SUPPORTED_CLUSTER_M:
        raise ValueError(
            f"cluster_m must be one of {_SUPPORTED_CLUSTER_M}, got "
            f"{requested_cluster_m}"
        )
    if requested_cluster_m * cluster_n > 16:
        raise ValueError(
            "cluster_m * cluster_n must not exceed 16 workgroups, got "
            f"{requested_cluster_m}x{cluster_n}"
        )
    return requested_cluster_m


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


_BIAS_MODE = {torch.bfloat16: 1, torch.float32: 2, torch.float16: 3}


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
    a_preshuffled=0,
    stage1_act=0,
    bias=None,
    swiglu_limit=7.0,
    stream=None,
    stage1_quant_out=0,
    quant_scale=None,
    quant_wmma_rep=1,
    cluster_m=-1,
    cluster_n=-1,
    waves_per_tensor_tdm=-1,
    next_stage_prefetch=0,
    tdm_as_in_prologue=0,
    tdm_b_th=0,
    lds_soa_load_interleave=0,
    stage2_scatter: Stage2ScatterContext | None = None,
    ep_destination_stride=0,
    ep_row_map=None,
    situ_beta=1.0,
    situ_linear_beta=1.0,
    row_major_ascale=0,
    a_row_stride_bytes=0,
    a_scale_row_stride_bytes=0,
    persistent_workers=0,
    rolled_drain=0,
    m_align=0,
    a_row_index=None,
    a_index_bound=0,
    a_index_stride=1,
    a_index_divisor=1,
):
    """Launch a contiguous-M grouped GEMM, optionally with persistent workers.

    A positive persistent_workers bounds the grid independently of the routing
    buffer capacity. Workers traverse the live device-side tile range.
    Preshuffled MXFP4 A uses logical (M/16, K/128, 4, 16, 32) tiles, with the
    innermost 32 FP4 values packed into 16 bytes.
    """
    if a_preshuffled and row_major_ascale:
        raise ValueError("A-preshuffled GEMM requires M16K4 activation scales")
    if a_preshuffled and not a_is_fp4:
        raise ValueError("preshuffled A requires MXFP4 activations")
    if a_preshuffled and (
        K % 128
        or contiguous_m % 16
        or tile_m % 16
        or tile_k % 128
        or a_row_stride_bytes
    ):
        raise ValueError(
            "preshuffled A requires M and tile_m divisible by 16, K and "
            "tile_k divisible by 128, and no padded A row stride"
        )
    if a_preshuffled and stage1_quant_out and (N % 256 or tile_n % 64):
        raise ValueError(
            "preshuffled MXFP4 quant output requires N divisible by 256 "
            "and tile_n divisible by 64"
        )
    from .kernels.mxfp4_preshuffle_gfx1250_tdm import launch_gemm_a8w4_tdm

    if stream is None:
        stream = torch.cuda.current_stream()
    if stage1_act == 3:
        if float(situ_beta) <= 0.0:
            raise ValueError(f"situ_beta must be > 0, got {situ_beta!r}")
        if float(situ_linear_beta) <= 0.0:
            raise ValueError(f"situ_linear_beta must be > 0, got {situ_linear_beta!r}")
    k_tiles = (K + tile_k - 1) // tile_k
    num_buffers = min(num_buffers, max(1, k_tiles))
    if bias is None:
        has_bias = 0
    elif bias.dtype in _BIAS_MODE:
        has_bias = _BIAS_MODE[bias.dtype]
    else:
        raise TypeError(f"bias must be bf16, fp32 or fp16, got {bias.dtype}")
    bias_ptr = ptr_arg(bias) if bias is not None else ptr_arg(a)
    # These tensor arguments only supply base addresses, so normalize their
    # ranks without specializing the compiled callable on buffer shapes.
    quant_scale_tensor = (
        out if quant_scale is None else quant_scale.view(torch.uint8)
    ).view(-1)
    n_tiles = (N + tile_n - 1) // tile_n
    cluster_n = _select_cluster_n(n_tiles, cluster_n)
    cluster_m = _select_cluster_m(cluster_m, cluster_n, stage1_act)
    if m_align:
        # A tile skipped for starting in an expert's padding never reaches the
        # M-cluster barrier its peers wait in; cluster_m only buys B multicast.
        cluster_m = 1
    stage = 1 if stage1_act else 2
    persistent_workers = int(
        os.environ.get(
            f"AITER_FLYDSL_PERSISTENT_WORKERS{stage}",
            os.environ.get("AITER_FLYDSL_PERSISTENT_WORKERS", persistent_workers),
        )
    )
    if persistent_workers < 0:
        raise ValueError("persistent_workers must be nonnegative")
    if persistent_workers:
        if cluster_m != 1:
            raise ValueError("persistent workers require cluster_m=1")
        if stage2_scatter is not None and stage2_scatter.combine_quant_bits:
            raise ValueError(
                "persistent EP scatter requires an unquantized combine wire"
            )
        persistent_workers = (
            (persistent_workers + cluster_n - 1) // cluster_n * cluster_n
        )
    waves_per_tensor_tdm = _select_num_waves_per_tensor_tdm(waves_per_tensor_tdm)
    rolled_drain = int(rolled_drain)
    if rolled_drain not in (0, 1):
        raise ValueError("rolled_drain must be 0 or 1")
    if cluster_n > 1 and n_tiles % cluster_n:
        raise ValueError(
            f"[grouped-moe tdm] cluster_n={cluster_n} needs n_tiles={n_tiles} "
            f"(N={N}, tile_n={tile_n}) to be an exact multiple"
        )
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
    ep_row_map_tensor = (ep_row_map if ep_row_map is not None else out).view(-1)
    a_gather = _a_gather_mode(a_row_index, a_index_bound)
    as_prologue = _select_bool_env("AITER_GROUPED_GEMM_AS_PROLOGUE", tdm_as_in_prologue)
    if a_gather:
        # The gather kernel has no A-scale prologue load.
        as_prologue = 0
        if m_warp * n_warp != 2 * waves_per_tensor_tdm or tile_m % (
            8 * waves_per_tensor_tdm
        ):
            raise ValueError(
                f"indexed A gather needs num_waves == 2 * waves_per_tensor_tdm "
                f"and tile_m % (8 * waves_per_tensor_tdm) == 0, got "
                f"{m_warp}x{n_warp} waves, waves_per_tensor_tdm="
                f"{waves_per_tensor_tdm}, tile_m={tile_m}"
            )
    if a_gather & 4:
        # A resident full-K A-scale beats one more stage of the per-K-tile
        # scale row gather (t256 decode, 1536 tok: 562 -> 513 us/layer at b3).
        # Approximate stage bytes; the kernel still drops residency if short.
        as_resident_b = tile_m * (K // 32) * 9 // 8
        stage_b = (tile_m + tile_n) * (tile_k // 2 + tile_k // 32)
        while num_buffers > 2 and num_buffers * stage_b + as_resident_b > 320 * 1024:
            num_buffers -= 1
    gemm_spec = (
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        out_is_f16,
        num_buffers,
        a_is_fp4,
    )
    route_spec = (n_experts, stage1_act, has_bias)
    quant_spec = (stage1_quant_out, quant_wmma_rep)
    tdm_spec = (
        cluster_n,
        cluster_m,
        _select_bool_env("AITER_TDM_NEXT_STAGE_PREFETCH", next_stage_prefetch),
        waves_per_tensor_tdm,
        as_prologue,
        _select_tdm_b_th(tdm_b_th),
    )
    ep_spec = (
        int(enable_ep_scatter),
        int(stage2_scatter.arena_handle) if enable_ep_scatter else 0,
        int(stage2_scatter.combine_input_offset) if enable_ep_scatter else 0,
        int(stage2_scatter.slot_stride_bytes) if enable_ep_scatter else 0,
        int(ep_destination_stride),
        int(stage2_scatter.world_size) if enable_ep_scatter else 0,
        int(stage2_scatter.combine_quant_bits) if enable_ep_scatter else 0,
    )
    layout_spec = (
        int(row_major_ascale),
        int(a_row_stride_bytes),
        int(a_scale_row_stride_bytes),
        int(lds_soa_load_interleave),
        persistent_workers,
        rolled_drain,
        int(a_preshuffled),
        int(m_align),
    )
    gather_spec = (
        int(a_index_stride),
        int(a_index_divisor),
        int(a_gather),
    )
    _run_compiled(
        launch_gemm_a8w4_tdm,
        out.view(-1),
        ptr_arg(a),
        ptr_arg(w),
        _launch_address(a_scales, torch.int32),
        _launch_address(w_scales, torch.int32),
        contiguous_m,
        stream,
        N,
        *gemm_spec,
        ptr_arg(m_tile_map),
        *route_spec,
        bias_ptr,
        float(swiglu_limit),
        *quant_spec,
        quant_scale_tensor,
        *tdm_spec,
        *ep_spec,
        ep_row_map_tensor,
        float(situ_beta),
        float(situ_linear_beta),
        *layout_spec,
        ptr_arg(a_row_index if a_row_index is not None else a),
        int(a_index_bound),
        *gather_spec,
        specialization_key=(
            gemm_spec,
            route_spec,
            quant_spec,
            tdm_spec,
            ep_spec,
            layout_spec,
            gather_spec,
        ),
    )
    return out
