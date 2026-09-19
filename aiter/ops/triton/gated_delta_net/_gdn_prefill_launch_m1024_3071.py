# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launcher for the m1024_3071 fused Qwen3-Next GDN prefill Gluon tile.

Holds the torch/triton host orchestration (buffer allocation + kernel
launches) for the ``m1024_3071`` M-tile so the kernel module
``aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_prefill_qkvz._tile_m1024_3071``
stays torch-free (Gluon only). Selected at runtime by the public wrapper
``fused_gdn_prefill_qkvz`` via its (tokens, batch) dispatch.
"""

from dataclasses import dataclass

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_prefill_qkvz._tile_m1024_3071 import (
    _chunk_offsets,
    _output_norm_quant,
    _prepare_chunk_factors,
    _prepare_inputs_tiled,
    _prepare_segment_maps,
    _propagate_chunks,
    _propagate_segments,
    _render_segment_chunks,
    _update_conv_state,
)


@dataclass(frozen=True)
class _PrecisionConfig:
    dtype: torch.dtype
    max_finite: float
    group_quant_max: float


FP8_E4M3_FN = _PrecisionConfig(torch.float8_e4m3fn, 448.0, 448.0)


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
