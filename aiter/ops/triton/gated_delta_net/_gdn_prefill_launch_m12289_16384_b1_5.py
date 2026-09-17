# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launcher for the m12289_16384_b1_5 fused Qwen3-Next GDN prefill Gluon tile.

Holds the torch/triton host orchestration (buffer allocation + kernel
launches) for the ``m12289_16384_b1_5`` M-tile so the kernel module
``aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_prefill_qkvz._tile_m12289_16384_b1_5``
stays torch-free (Gluon only). Selected at runtime by the public wrapper
``fused_gdn_prefill_qkvz`` via its (tokens, batch) dispatch.
"""

from dataclasses import dataclass

import torch

from aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_prefill_qkvz._tile_m12289_16384_b1_5 import (
    _build_segments,
    _build_segments_reverse,
    _build_segments_stacked,
    _chunk_output_quant,
    _chunk_state_rows,
    _chunk_transform,
    _normalize_quantize,
    _prefix_segments,
    _prepare_gates,
    _prepare_qkv_window,
    _state_and_core,
)


@dataclass(frozen=True)
class FP8PrecisionConfig:
    dtype: torch.dtype
    max_finite: float
    group_quant_max: float


FP8_E4M3_FN = FP8PrecisionConfig(torch.float8_e4m3fn, 448.0, 448.0)


@dataclass(frozen=True)
class _Schedule:
    """Shape-only tuning; sequence boundaries and cache ownership stay on-device."""

    segment_tokens: int
    build_rows: int
    reverse_columns: int
    prefix_rows: int
    core_rows: int
    core_transposed: bool
    prep_warps: int
    prep_tokens: int
    norm_rows: int
    norm_warps: int
    fused_gates: bool
    fused_core: bool
    time_major: bool
    unit_major: bool
    preweight_key: bool
    prepare_bounds: bool


def _schedule(m: int, batch: int) -> _Schedule:
    wide = batch == 2 and m >= 16384
    if 2 < batch < 8 and m <= 4096:
        segment = 0
    elif batch <= 4 and m <= 4096:
        segment = 128
    elif m <= 2048 * batch:
        segment = 0
    elif batch == 1:
        if m >= 32768:
            segment = 1024
        elif m <= 12288:
            segment = ((m + 511) // 512) * 32
        else:
            segment = 512
    elif 3 <= batch <= 4 and m >= 24576:
        # Fewer prefix matrices, balanced by wider column-wise summaries.
        segment = 1024
    elif 3 <= batch <= 5 and 12288 <= m <= 24576:
        segment = 704
    elif wide or (batch > 4 and m > 4096 * batch):
        segment = 512
    else:
        segment = 256
    core_rows = (
        (64 if wide or m >= 32768 else 32) if segment else (32 if batch < 32 else 64)
    )
    fused_core = segment != 0 or batch >= 8
    build_rows = (
        128 if wide or (batch == 1 and m >= 32768) else (64 if m >= 16384 else 32)
    )
    reverse_columns = 0
    if segment and m // segment + batch >= 16:
        reverse_columns = (
            64 if build_rows == 128 or (3 <= batch <= 4 and m >= 24576) else 32
        )
    return _Schedule(
        segment_tokens=segment,
        build_rows=build_rows,
        reverse_columns=reverse_columns,
        prefix_rows=4 if batch == 1 else (8 if batch == 2 else 16),
        core_rows=core_rows,
        # Keep the established ownership of each recurrent path.
        core_transposed=(segment != 0 or batch >= 32)
        and not (batch == 1 and m <= 1024),
        prep_warps=4 if m <= 2048 else 1,
        prep_tokens=4 if m <= 12288 else 8,
        norm_rows=16 if m > 8192 else 32,
        norm_warps=1 if m > 8192 else 4,
        fused_gates=m <= 8192,
        fused_core=fused_core,
        time_major=batch >= 32,
        unit_major=segment != 0 and (m < 32768 or batch >= 3),
        preweight_key=(
            not fused_core or (batch == 1 and m <= 1024) or (batch == 8 and m <= 8192)
        ),
        prepare_bounds=batch >= 8 or (2 < batch < 8 and m <= 4096),
    )


def gdn_prefill_group_fp8_quant(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_state: torch.Tensor,
    delta_state: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    scale: float,
    eps: float = 1.0e-6,
    precision_config: FP8PrecisionConfig = FP8_E4M3_FN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m = projected_qkvz.shape[0]
    batch = cache_indices.numel()
    device = projected_qkvz.device
    schedule = _schedule(m, batch)
    bt = 32
    # A single sequence begins at zero, so no inter-sequence reserve is needed.
    # Multi-sequence bounds remain entirely on-device and use the safe capacity.
    chunks = (m + bt - 1) // bt if batch == 1 else m // bt + batch
    # Separate direct, preweighted head planes by eight unused rows. Consumers
    # address only initialized chunk rows; the gap changes physical head pitch.
    qkv_gap = (
        8
        if schedule.fused_core
        and not schedule.segment_tokens
        and schedule.preweight_key
        else 0
    )
    qkv_rows = chunks * bt + qkv_gap
    qkv = torch.empty((16, qkv_rows, 128), device=device, dtype=torch.bfloat16)
    bounds = (
        torch.empty((chunks, 2), device=device, dtype=torch.int32)
        if schedule.prepare_bounds
        else None
    )
    fused_gates = schedule.fused_gates
    gates = (
        projected_ba
        if fused_gates
        else torch.empty((8, 2, qkv_rows), device=device, dtype=torch.float32)
    )
    normalized = torch.empty((m, 8, 128), device=device, dtype=torch.bfloat16)
    quantized = torch.empty((m, 1024), device=device, dtype=precision_config.dtype)
    scales = torch.empty((m, 8), device=device, dtype=torch.float32)
    prep_warps = schedule.prep_warps
    prep_tokens = schedule.prep_tokens
    _prepare_qkv_window[(chunks * 32 // (prep_tokens * prep_warps), 16)](
        projected_qkvz,
        conv_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        conv_weight,
        conv_bias,
        qkv,
        qkv_rows,
        batch,
        prep_tokens,
        prep_warps,
        Bounds=bounds,
        num_warps=prep_warps,
        enable_fp_fusion=False,
    )
    if not fused_gates:
        _prepare_gates[(chunks,)](
            projected_ba,
            cu_seqlens,
            a_log,
            dt_bias,
            gates,
            qkv_rows,
            batch,
            bt,
            Bounds=bounds,
            num_warps=4,
            enable_fp_fusion=False,
        )
    seg = schedule.segment_tokens
    fused_core = schedule.fused_core
    time_major = schedule.time_major
    u = torch.empty((chunks, 8, 128, bt), device=device, dtype=torch.float32)
    w = torch.empty_like(u)
    scores = torch.empty((chunks, 8, bt, bt), device=device, dtype=torch.float32)
    coeff = torch.empty((chunks, 8, 2, bt), device=device, dtype=torch.float32)
    tail_key = (
        torch.empty((chunks, 8, bt, 128), device=device, dtype=torch.float32)
        if schedule.preweight_key
        else None
    )
    if fused_core:
        core = normalized
    else:
        chunk_state = torch.empty(
            (chunks, 8, 128, 128), device=device, dtype=torch.float32
        )
    # Adjacent heads share the tiny-input factorization traversal.
    head_major = m <= 1024 and batch == 1
    factor_grid = (8, chunks) if head_major else (chunks, 8)
    _chunk_transform[factor_grid](
        qkv,
        gates,
        u,
        w,
        scores,
        coeff,
        qkv_rows,
        bt,
        num_warps=4,
        enable_fp_fusion=False,
        TIME_MAJOR=time_major,
        BA=projected_ba,
        Starts=cu_seqlens,
        ALog=a_log,
        DTBias=dt_bias,
        BATCH=batch,
        FUSED_GATES=fused_gates,
        TailKey=tail_key,
        Bounds=bounds,
        HEAD_MAJOR=head_major,
    )
    if seg:
        segments = (m + seg - 1) // seg if batch == 1 else m // seg + batch
        affine = torch.empty(
            (segments, 8, 2, 128, 128), device=device, dtype=torch.float32
        )
        segment_state = affine
        build_bv = schedule.build_rows
        build_wm = 4 if build_bv == 128 else 2
        if schedule.reverse_columns:
            columns = schedule.reverse_columns
            build_kernel = (
                _build_segments_stacked if columns == 64 else _build_segments_reverse
            )
            build_kernel[(8, 128 // columns, segments)](
                qkv,
                u,
                w,
                coeff,
                cu_seqlens,
                affine,
                qkv_rows,
                batch,
                bt,
                seg,
                columns,
                4,
                MMA_SIZE=32 if columns == 32 else 16,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            _build_segments[(8, 2 * 128 // build_bv, segments)](
                qkv,
                u,
                w,
                coeff,
                cu_seqlens,
                affine,
                qkv_rows,
                batch,
                bt,
                seg,
                build_bv,
                4,
                build_wm,
                TailKey=tail_key,
                num_warps=4,
                enable_fp_fusion=False,
            )
        prefix_bv = schedule.prefix_rows
        _prefix_segments[(8, 128 // prefix_bv, batch)](
            affine,
            delta_state,
            cache_indices,
            cu_seqlens,
            seg,
            prefix_bv,
            num_warps=2 if prefix_bv == 4 else 4,
            enable_fp_fusion=False,
        )
    if fused_core:
        units = segments if seg else batch
        bv = schedule.core_rows
        core_grid = (
            (8, 128 // bv, units) if schedule.unit_major else (units, 8, 128 // bv)
        )
        _state_and_core[core_grid](
            qkv,
            u,
            w,
            scores,
            coeff,
            delta_state,
            cache_indices,
            cu_seqlens,
            segment_state if seg else delta_state,
            core,
            scale,
            qkv_rows,
            batch,
            bt,
            seg,
            bv,
            4,
            2,
            num_warps=4,
            enable_fp_fusion=False,
            TIME_MAJOR=time_major,
            UNIT_MAJOR=schedule.unit_major,
            TailKey=tail_key,
            TRANSPOSED=schedule.core_transposed,
        )
    else:
        bv, wm = 16, 1
        _chunk_state_rows[(batch, 8, 128 // bv)](
            qkv,
            u,
            w,
            coeff,
            delta_state,
            cache_indices,
            cu_seqlens,
            chunk_state,
            qkv_rows,
            bt,
            bv,
            4,
            wm,
            num_warps=4,
            enable_fp_fusion=False,
            TailKey=tail_key,
        )
    maximum = precision_config.group_quant_max
    inverse_maximum = 1.0 / maximum
    if fused_core:
        norm_rows = schedule.norm_rows
        _normalize_quantize[((m * 8 + norm_rows - 1) // norm_rows,)](
            core,
            projected_qkvz,
            norm_weight,
            normalized,
            quantized,
            scales,
            eps,
            maximum,
            inverse_maximum,
            m,
            norm_rows,
            schedule.norm_warps,
            num_warps=schedule.norm_warps,
            enable_fp_fusion=False,
        )
    else:
        _chunk_output_quant[(chunks, 8)](
            qkv,
            u,
            scores,
            coeff,
            cu_seqlens,
            chunk_state,
            projected_qkvz,
            norm_weight,
            normalized,
            quantized,
            scales,
            scale,
            eps,
            maximum,
            inverse_maximum,
            qkv_rows,
            batch,
            bt,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return normalized, conv_state, delta_state, quantized, scales
