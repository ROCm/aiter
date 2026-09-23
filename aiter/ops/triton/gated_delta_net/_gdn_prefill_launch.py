# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launchers for the fused Qwen3-Next GDN prefill Gluon tiles.

One launcher per M-tile schedule (``gdn_prefill_group_fp8_quant_<key>``), all in
one place next to the public wrapper. Each holds only the torch/triton host
orchestration (buffer allocation + kernel launches) for its schedule; the
``@gluon.jit`` kernels stay torch-free in
``aiter/ops/triton/_gluon_kernels/gfx950/gated_delta_net/fused_gdn_prefill_qkvz``.
The public wrapper ``fused_gdn_prefill_qkvz`` picks one via its (tokens, batch)
dispatch (``_select_tile_key`` -> ``gdn_prefill_group_fp8_quant_<key>``).
"""

from dataclasses import dataclass

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_prefill_qkvz import (
    _block_transforms,
    _build_group_maps,
    _build_segments,
    _build_segments_reverse,
    _build_segments_stacked,
    _chunk_delta_recurrence,
    _chunk_offsets,
    _chunk_output_quant,
    _chunk_state_rows,
    _chunk_transform,
    _convolve_sliding,
    _emit_chunk_outputs,
    _evaluate_groups,
    _finish_blocks,
    _gated_rms_quant,
    _norm_and_quantize,
    _normalize_quantize,
    _output_norm_quant,
    _prefix_segments,
    _prepare_chunk_factors,
    _prepare_chunk_matrices,
    _prepare_chunks_compact,
    _prepare_chunks_full,
    _prepare_gates,
    _prepare_inputs_tiled,
    _prepare_qkv_window,
    _prepare_segment_maps,
    _prepare_tokens,
    _propagate_block_states,
    _propagate_chunks,
    _propagate_group_maps,
    _propagate_segments,
    _recurrence_compact,
    _recurrence_full,
    _render_segment_chunks,
    _scan_chunk_states,
    _state_and_core,
    _update_conv_state,
    _update_conv_state_b6_15,
)


@dataclass(frozen=True)
class FP8PrecisionConfig:
    dtype: torch.dtype
    max_finite: float
    group_quant_max: float


# The m1024_3071 launcher names this dataclass ``_PrecisionConfig``; keep the
# alias so its body stays verbatim.
_PrecisionConfig = FP8PrecisionConfig

FP8_E4M3_FN = FP8PrecisionConfig(torch.float8_e4m3fn, 448.0, 448.0)

##############################################################################
# schedule m1024_3071
##############################################################################


def _run_chunked_delta(
    prepared,
    projected,
    ba,
    gates,
    states,
    indices,
    starts,
    a_log,
    dt_bias,
    norm_weight,
    normalized,
    quantized,
    scales,
    scale,
    eps,
    C=32,
    BV=16,
    S=0,
    WM=4,
):
    m = prepared.shape[0]
    batch = indices.numel()
    chunks = triton.cdiv(m, C) + batch
    device = prepared.device
    offsets = torch.empty((batch + 1,), device=device, dtype=torch.int32)
    # Short, low-batch sequences benefit from parallel query/state products.
    # Larger workloads retain the smaller output-base buffer.
    history = S == 0 and batch <= 2 and C >= 32
    # Packing pays off only when the additional operands stay small.
    planes = (2 if history else 3) if S == 0 and m <= 4096 and batch <= 4 else 0
    w = torch.empty(
        (chunks, 8, max(1, planes), C, 128), device=device, dtype=torch.float32
    )
    u = torch.empty((chunks, 8, C, 128), device=device, dtype=torch.float32)
    updates = torch.empty_like(u)
    carry_buffer = torch.empty(
        (chunks, 8, 128 if history else C, 128), device=device, dtype=torch.float32
    )
    g = torch.empty((chunks, 8, C * C + C), device=device, dtype=torch.float32)
    _chunk_offsets[(batch + 1,)](
        starts, offsets, batch, triton.next_power_of_2(batch), C, num_warps=1
    )
    _prepare_chunk_factors[(chunks, 8)](
        prepared,
        ba,
        gates,
        a_log,
        dt_bias,
        starts,
        offsets,
        w,
        u,
        g,
        batch,
        C,
        planes,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if S:
        segments = triton.cdiv(m, C * S) + batch
        segment_offsets = torch.empty_like(offsets)
        transitions = torch.empty(
            (segments, 8, 128, 128), device=device, dtype=torch.float32
        )
        contributions = torch.empty_like(transitions)
        initial_states = torch.empty_like(transitions)
        _chunk_offsets[(batch + 1,)](
            starts,
            segment_offsets,
            batch,
            triton.next_power_of_2(batch),
            C * S,
            num_warps=1,
        )
        _prepare_segment_maps[(segments, 8, 4)](
            prepared,
            starts,
            offsets,
            segment_offsets,
            w,
            u,
            g,
            transitions,
            contributions,
            batch,
            C,
            S,
            32,
            2,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _propagate_segments[(batch, 8, 8)](
            states,
            indices,
            segment_offsets,
            transitions,
            contributions,
            initial_states,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _render_segment_chunks[(segments, 8, 4)](
            prepared,
            starts,
            offsets,
            segment_offsets,
            w,
            u,
            g,
            initial_states,
            updates,
            carry_buffer,
            batch,
            C,
            S,
            32,
            2,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        _propagate_chunks[(batch, 8, 128 // BV)](
            prepared,
            states,
            indices,
            starts,
            offsets,
            w,
            u,
            g,
            updates,
            carry_buffer,
            C,
            BV,
            WM,
            history,
            planes,
            num_warps=4,
            enable_fp_fusion=False,
        )
    _output_norm_quant[(chunks, 8)](
        prepared,
        projected,
        norm_weight,
        starts,
        offsets,
        g,
        updates,
        carry_buffer,
        normalized,
        quantized,
        scales,
        scale,
        eps,
        batch,
        C,
        history,
        num_warps=4,
        enable_fp_fusion=False,
    )


def gdn_prefill_group_fp8_quant_m1024_3071(
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
    precision_config=FP8_E4M3_FN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m = projected_qkvz.shape[0]
    batch = cache_indices.numel()
    device = projected_qkvz.device
    prepared = torch.empty((m, 2048), device=device, dtype=torch.bfloat16)
    gates = torch.empty((m, 8), device=device, dtype=torch.float32)
    normalized = torch.empty((m, 8, 128), device=device, dtype=torch.bfloat16)
    quantized = torch.empty((m, 1024), device=device, dtype=precision_config.dtype)
    scales = torch.empty((m, 8), device=device, dtype=torch.float32)
    _prepare_inputs_tiled[(triton.cdiv(m, 16), 16)](
        projected_qkvz,
        projected_ba,
        conv_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        conv_weight,
        conv_bias,
        prepared,
        gates,
        m,
        batch,
        16,
        8,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _update_conv_state[(batch, 8)](
        projected_qkvz,
        conv_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        num_warps=4,
    )
    # Target about 16 or 32 independent temporal segments, rounded to an even
    # number of chunks. This balances parallel maps against the segment carry.
    target_segments = 16 if m <= 6144 else 32
    segment_chunks = (
        2 * triton.cdiv(m, 128 * target_segments) if batch <= 2 and m >= 4096 else 0
    )
    _run_chunked_delta(
        prepared,
        projected_qkvz,
        projected_ba,
        gates,
        delta_state,
        cache_indices,
        cu_seqlens,
        a_log,
        dt_bias,
        norm_weight,
        normalized,
        quantized,
        scales,
        scale,
        eps,
        C=32 if batch >= 16 or 5 <= batch <= 7 else 64,
        BV=64 if batch >= 16 else (8 if batch <= 2 else 16),
        WM=1 if batch >= 16 else 4,
        S=segment_chunks,
    )
    return normalized, conv_state, delta_state, quantized, scales


##############################################################################
# schedule m12289_16384_b1_5
##############################################################################


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


def gdn_prefill_group_fp8_quant_m12289_16384_b1_5(
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


##############################################################################
# schedule m12289_16384_b6_15
##############################################################################


def gdn_prefill_group_fp8_quant_m12289_16384_b6_15(
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
    """Return BF16 norm, updated state pools, FP8 values, and FP32 scales."""
    m = projected_qkvz.shape[0]
    batch = cache_indices.numel()
    slots, heads, dim, key_dim = delta_state.shape
    assert dim == key_dim == 128 and heads == 8
    assert projected_qkvz.shape == (m, 3072) and projected_ba.shape == (m, 16)
    assert conv_state.shape == (slots, 2048, 3)
    assert conv_weight.shape == (2048, 4) and conv_bias.shape == (2048,)
    assert cu_seqlens.shape == (batch + 1,) and has_initial_state.shape == (batch,)
    assert cache_indices.dtype is cu_seqlens.dtype is torch.int32
    assert has_initial_state.dtype is torch.bool and delta_state.dtype is torch.float32
    assert a_log.shape == dt_bias.shape == (8,) and a_log.dtype is torch.float32
    assert norm_weight.shape == (128,) and m > 0
    bf16 = (
        projected_qkvz,
        projected_ba,
        conv_state,
        conv_weight,
        conv_bias,
        dt_bias,
        norm_weight,
    )
    assert all(t.dtype is torch.bfloat16 and t.is_contiguous() for t in bf16)

    device = projected_qkvz.device
    prepared = torch.empty((16, m, 128), device=device, dtype=torch.bfloat16)
    gates = torch.empty((8, m, 2), device=device, dtype=torch.float32)
    prep_rows, prep_warps = (16, 4) if m <= 4096 else (4, 1)
    _prepare_tokens[((m + prep_rows - 1) // prep_rows, 16)](
        projected_qkvz,
        projected_ba,
        conv_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        prepared,
        gates,
        M=m,
        LOG_BATCH=(batch - 1).bit_length(),
        BATCH=batch,
        ROWS=prep_rows,
        LANES=16,
        PACK=4,
        num_warps=prep_warps,
        enable_fp_fusion=False,
    )
    # All readers of the old convolution history complete before this launch.
    _update_conv_state_b6_15[(batch, 16)](
        projected_qkvz,
        conv_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        num_warps=1,
    )
    core = torch.empty((m, 8, 128), device=device, dtype=torch.bfloat16)
    use_groups = batch == 1 or (batch <= 4 and m // batch >= 4096)
    if use_groups:
        token_block, prep_warps = 32, 1
        value_block, recur_warps = (16, 2) if m <= 2048 else (32, 4)
    elif batch <= 4:
        token_block, prep_warps, value_block, recur_warps = 32, 1, 16, 4
    elif batch <= 8:
        token_block, prep_warps, value_block, recur_warps = 32, 1, 16, 2
    elif batch <= 16:
        token_block, prep_warps, value_block, recur_warps = 32, 1, 32, 2
    else:
        token_block, prep_warps, value_block, recur_warps = 16, 1, 32, 2
    num_chunks = m // token_block + batch
    chunk_c = torch.empty(
        (num_chunks, 8, token_block, token_block),
        device=device,
        dtype=torch.float32,
    )
    if batch <= 4:
        chunk_w = torch.empty(
            (num_chunks, 8, token_block, 128), device=device, dtype=torch.float32
        )
        chunk_u = torch.empty_like(chunk_w)
        chunk_q = torch.empty_like(chunk_w)
        chunk_k = torch.empty_like(chunk_w)
        chunk_g = torch.empty((num_chunks, 8), device=device, dtype=torch.float32)
        _prepare_chunks_full[(num_chunks, 8)](
            prepared,
            gates,
            cu_seqlens,
            chunk_w,
            chunk_u,
            chunk_q,
            chunk_k,
            chunk_c,
            chunk_g,
            M=m,
            BATCH=batch,
            BT=token_block,
            num_warps=prep_warps,
        )
        if use_groups:
            group_tokens = (
                128 if m <= 2048 else 512 if m <= 8192 else 1024 if m <= 16384 else 2048
            )
            group_row_warps = 1 if value_block == 16 else 2
            num_groups = m // group_tokens + batch
            maps = torch.empty(
                (num_groups, 8, 256, 128), device=device, dtype=torch.float32
            )
            boundaries = torch.empty(
                (num_groups, 8, 128, 128), device=device, dtype=torch.float32
            )
            _build_group_maps[(256 // value_block, 8, num_groups)](
                chunk_w,
                chunk_u,
                chunk_k,
                chunk_g,
                cu_seqlens,
                maps,
                BATCH=batch,
                BT=token_block,
                BV=value_block,
                GROUP=group_tokens,
                ROW_WARPS=group_row_warps,
                num_warps=recur_warps,
            )
            _propagate_group_maps[(batch, 8, 8)](
                maps,
                boundaries,
                delta_state,
                cache_indices,
                cu_seqlens,
                GROUP=group_tokens,
                BV=16,
                num_warps=4,
            )
            _evaluate_groups[(128 // value_block, 8, num_groups)](
                chunk_w,
                chunk_u,
                chunk_q,
                chunk_k,
                chunk_c,
                chunk_g,
                boundaries,
                cu_seqlens,
                core,
                scale,
                BATCH=batch,
                BT=token_block,
                BV=value_block,
                GROUP=group_tokens,
                ROW_WARPS=group_row_warps,
                num_warps=recur_warps,
            )
        else:
            _recurrence_full[(batch, 8, 128 // value_block)](
                chunk_w,
                chunk_u,
                chunk_q,
                chunk_k,
                chunk_c,
                chunk_g,
                delta_state,
                cache_indices,
                cu_seqlens,
                core,
                scale,
                BT=token_block,
                BV=value_block,
                num_warps=recur_warps,
            )
    else:
        chunk_inverse = torch.empty_like(chunk_c)
        chunk_decay = torch.empty(
            (num_chunks, 8, 2, token_block),
            device=device,
            dtype=torch.float32,
        )
        _prepare_chunks_compact[(num_chunks, 8)](
            prepared,
            gates,
            cu_seqlens,
            chunk_inverse,
            chunk_c,
            chunk_decay,
            M=m,
            BATCH=batch,
            BT=token_block,
            num_warps=prep_warps,
        )
        _recurrence_compact[(batch, 8, 128 // value_block)](
            prepared,
            gates,
            chunk_inverse,
            chunk_c,
            chunk_decay,
            delta_state,
            cache_indices,
            cu_seqlens,
            core,
            scale,
            M=m,
            BT=token_block,
            BV=value_block,
            num_warps=recur_warps,
            ROW_WARPS=2 if batch > 8 else 1,
            TRANSPOSED=batch <= 8,
        )
    normalized = torch.empty_like(core)
    values = torch.empty((m, 1024), device=device, dtype=precision_config.dtype)
    scales = torch.empty((m, 8), device=device, dtype=torch.float32)
    epi_rows, epi_lanes, epi_pack = (4, 16, 4) if m < 4096 else (8, 8, 8)
    _gated_rms_quant[((m * 8 + epi_rows - 1) // epi_rows,)](
        core,
        projected_qkvz,
        norm_weight,
        normalized,
        values,
        scales,
        eps,
        precision_config.group_quant_max,
        1.0 / precision_config.group_quant_max,
        TOTAL=m * 8,
        ROWS=epi_rows,
        LANES=epi_lanes,
        PACK=epi_pack,
        num_warps=1,
    )
    return normalized, conv_state, delta_state, values, scales


##############################################################################
# schedule m3072_16384
##############################################################################


def gdn_prefill_group_fp8_quant_m3072_16384(
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
    """Return BF16 output, the mutated pools, FP8 output, and group scales."""
    m = projected_qkvz.shape[0]
    batch = cache_indices.numel()
    device = projected_qkvz.device
    prepared = torch.empty((m, 2048), dtype=torch.bfloat16, device=device)
    core = (
        prepared.view(-1)[: m * 1024].view(m, 8, 128)
        if m >= 8192
        else torch.empty((m, 8, 128), dtype=torch.bfloat16, device=device)
    )
    normalized = core
    quantized = torch.empty((m, 1024), dtype=precision_config.dtype, device=device)
    scales = torch.empty((m, 8), dtype=torch.float32, device=device)
    bt = 32
    snapshot_outputs = 3 <= batch <= 4 and m <= 4096
    native_u = snapshot_outputs or (12288 < m <= 16384 and batch == 2)
    hierarchical = batch <= 5 and (batch < 4 or m > 4096) and not snapshot_outputs
    compact_qk = batch >= 16
    packed_rhs = hierarchical or snapshot_outputs or compact_qk
    chunks = triton.cdiv(m, bt) + batch
    chunk_offsets = torch.empty((batch + 1,), dtype=torch.int32, device=device)
    group = (
        4
        if m <= 2048
        else (
            12
            if 4096 < m <= 6144
            else 8 if m <= 8192 else 24 if m <= 12288 else 16 if m <= 16384 else 32
        )
    )
    wide_summary = 12288 < m <= 16384 and 3 <= batch <= 5
    if wide_summary:
        group = 20
    if hierarchical:
        block_offsets = torch.empty((batch + 1,), dtype=torch.int32, device=device)
    else:
        block_offsets = chunk_offsets
    matrices = torch.empty(
        (chunks, 8, 2 if compact_qk else 4, bt, 128), dtype=torch.float32, device=device
    )
    output_weights = torch.empty(
        (chunks, 8, bt, bt), dtype=torch.float32, device=device
    )
    chunk_decay = torch.empty((chunks, 8), dtype=torch.float32, device=device)
    chunk_begin = torch.empty((chunks,), dtype=torch.int32, device=device)
    raw_qk = (
        torch.empty((chunks, 4, 2, bt, 128), dtype=torch.bfloat16, device=device)
        if compact_qk
        else matrices
    )
    qk_factors = (
        torch.empty((chunks, 8, 2, bt), dtype=torch.float32, device=device)
        if compact_qk
        else chunk_decay
    )

    convolution_tile = 8 if m >= 16384 else 4
    _convolve_sliding[(triton.cdiv(m, convolution_tile), 4, 2)](
        projected_qkvz,
        conv_state,
        cu_seqlens,
        cache_indices,
        has_initial_state,
        conv_weight,
        conv_bias,
        prepared,
        chunk_offsets,
        block_offsets,
        m,
        batch,
        batch.bit_length(),
        triton.next_power_of_2(batch),
        bt,
        group,
        hierarchical,
        convolution_tile,
        num_warps=2,
        enable_fp_fusion=False,
    )
    _prepare_chunk_matrices[(8, chunks)](
        prepared,
        projected_ba,
        a_log,
        dt_bias,
        cu_seqlens,
        chunk_offsets,
        matrices,
        output_weights,
        chunk_decay,
        chunk_begin,
        batch,
        batch.bit_length(),
        bt,
        packed_rhs,
        num_warps=4,
        enable_fp_fusion=False,
        COMPACT_QK=compact_qk,
        raw_qk=raw_qk,
        qk_factors=qk_factors,
        NATIVE_U=native_u,
    )
    if snapshot_outputs:
        snapshots = torch.empty(
            (chunks, 8, 128, 128), dtype=torch.float32, device=device
        )
        _scan_chunk_states[(batch, 8, 8)](
            matrices,
            chunk_decay,
            chunk_offsets,
            delta_state,
            cache_indices,
            snapshots,
            bt,
            16,
            2,
            num_warps=2,
            enable_fp_fusion=False,
        )
        _emit_chunk_outputs[(chunks, 8)](
            matrices,
            output_weights,
            snapshots,
            chunk_begin,
            chunk_offsets,
            projected_qkvz,
            norm_weight,
            normalized,
            quantized,
            scales,
            conv_state,
            cu_seqlens,
            cache_indices,
            has_initial_state,
            scale,
            eps,
            precision_config.group_quant_max,
            1.0 / precision_config.group_quant_max,
            batch,
            batch.bit_length(),
            bt,
            num_warps=4,
        )
    elif hierarchical:
        direct_first = m < 32768
        value_tile, warps, value_warps = (
            (16, 2, 1)
            if m < 4096
            else (
                (128, 4, 2)
                if m >= 32768
                else (64, 4, 2) if wide_summary else (32, 4, 2)
            )
        )
        finish_warps, finish_value_warps = warps, value_warps
        if 12288 < m <= 16384 and batch == 2:
            value_tile, warps, value_warps = 32, 2, 1
        blocks = triton.cdiv(m, bt * group) + batch
        transforms = torch.empty(
            (blocks, 8, 2, 128, 128), dtype=torch.float32, device=device
        )
        block_info = torch.empty((blocks, 3), dtype=torch.int32, device=device)
        _block_transforms[(blocks, 8, 2 * 128 // value_tile)](
            matrices,
            chunk_decay,
            chunk_offsets,
            block_offsets,
            transforms,
            block_info,
            delta_state,
            cache_indices,
            batch,
            batch.bit_length(),
            bt,
            group,
            value_tile,
            warps,
            value_warps,
            direct_first,
            num_warps=warps,
            enable_fp_fusion=False,
            NATIVE_U=native_u,
        )
        _propagate_block_states[(batch, 8, 8)](
            transforms,
            block_offsets,
            delta_state,
            cache_indices,
            16,
            4,
            direct_first,
            num_warps=4,
            enable_fp_fusion=False,
        )
        finish_tile = 16 if 8192 < m <= 12288 else 64 if m >= 8192 else value_tile
        if 8192 < m <= 12288:
            # More independent replay CTAs benefit this intermediate-length regime.
            finish_warps, finish_value_warps = 2, 1
        finish_unroll = 1 if (m <= 2048 and batch > 1) or 6144 < m <= 8192 else 4
        _finish_blocks[(blocks, 8, 128 // finish_tile)](
            matrices,
            output_weights,
            chunk_decay,
            chunk_begin,
            chunk_offsets,
            block_offsets,
            block_info,
            transforms,
            delta_state,
            cache_indices,
            cu_seqlens,
            core,
            scale,
            batch,
            bt,
            finish_tile,
            finish_warps,
            finish_value_warps,
            num_warps=finish_warps,
            enable_fp_fusion=False,
            UNROLL=finish_unroll,
            NATIVE_U=native_u,
        )
    else:
        value_tile, warps, value_warps = (64, 4, 2) if batch >= 32 else (32, 4, 2)
        transposed = not (8 <= batch < 16)
        preload = 1 if 8 <= batch < 16 else 3 if batch >= 16 else 0
        _chunk_delta_recurrence[(batch, 8, 128 // value_tile)](
            matrices,
            output_weights,
            chunk_decay,
            chunk_begin,
            chunk_offsets,
            delta_state,
            cache_indices,
            cu_seqlens,
            core,
            scale,
            bt,
            value_tile,
            warps,
            value_warps,
            packed_rhs,
            transposed,
            preload,
            num_warps=warps,
            enable_fp_fusion=False,
            COMPACT_QK=compact_qk,
            raw_qk=raw_qk,
            qk_factors=qk_factors,
        )
    if not snapshot_outputs:
        _norm_and_quantize[(max(triton.cdiv(m * 8, 16), batch * 8),)](
            core,
            projected_qkvz,
            norm_weight,
            normalized,
            quantized,
            scales,
            conv_state,
            cu_seqlens,
            cache_indices,
            has_initial_state,
            eps,
            precision_config.group_quant_max,
            1.0 / precision_config.group_quant_max,
            m,
            16,
            batch,
            num_warps=4,
        )
    return normalized, conv_state, delta_state, quantized, scales
