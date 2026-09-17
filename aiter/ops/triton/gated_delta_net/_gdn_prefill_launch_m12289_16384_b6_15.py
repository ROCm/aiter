# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launcher for the m12289_16384_b6_15 fused Qwen3-Next GDN prefill Gluon tile.

Holds the torch/triton host orchestration (buffer allocation + kernel
launches) for the ``m12289_16384_b6_15`` M-tile so the kernel module
``aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_prefill_qkvz._tile_m12289_16384_b6_15``
stays torch-free (Gluon only). Selected at runtime by the public wrapper
``fused_gdn_prefill_qkvz`` via its (tokens, batch) dispatch.
"""

from dataclasses import dataclass

import torch

from aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_prefill_qkvz._tile_m12289_16384_b6_15 import (
    _build_group_maps,
    _evaluate_groups,
    _gated_rms_quant,
    _prepare_chunks_compact,
    _prepare_chunks_full,
    _prepare_tokens,
    _propagate_group_maps,
    _recurrence_compact,
    _recurrence_full,
    _update_conv_state,
)


@dataclass(frozen=True, slots=True)
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
    _update_conv_state[(batch, 16)](
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
