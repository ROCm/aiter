# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launcher for the m3072_16384 fused Qwen3-Next GDN prefill Gluon tile.

Holds the torch/triton host orchestration (buffer allocation + kernel
launches) for the ``m3072_16384`` M-tile so the kernel module
``aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_prefill_qkvz._tile_m3072_16384``
stays torch-free (Gluon only). Selected at runtime by the public wrapper
``fused_gdn_prefill_qkvz`` via its (tokens, batch) dispatch.
"""

from dataclasses import dataclass

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_prefill_qkvz._tile_m3072_16384 import (
    _block_transforms,
    _chunk_delta_recurrence,
    _convolve_sliding,
    _emit_chunk_outputs,
    _finish_blocks,
    _norm_and_quantize,
    _prepare_chunk_matrices,
    _propagate_block_states,
    _scan_chunk_states,
)


@dataclass(frozen=True)
class FP8PrecisionConfig:
    dtype: torch.dtype
    max_finite: float
    group_quant_max: float


FP8_E4M3_FN = FP8PrecisionConfig(torch.float8_e4m3fn, 448.0, 448.0)


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
