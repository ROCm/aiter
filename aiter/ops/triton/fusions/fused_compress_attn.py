# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton._gluon_kernels.gfx1250.fusions.fused_compress_attn import (
    HEAD_DIM,
    RATIO,
    ROPE_DIM,
    fused_compress_attn_fp4_gfx1250,
    get_fused_compress_attn_fp4_layouts,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_CONFIG_NAME = "FUSED_COMPRESS_ATTN_FP4"


def _get_config(num_plan_rows: int) -> dict:
    config_dir = resolve_config_dir("fusions", _CONFIG_NAME, backend="gluon")
    configs = load_config_json(f"{config_dir}/DEFAULT.json")
    return select_leq_config(configs, num_plan_rows, prefix="P_LEQ_")


def fused_compress_attn_fp4(
    kv_in: torch.Tensor,
    score_in: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    state_slots: torch.Tensor,
    plan: torch.Tensor,
    ape: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_scale: torch.Tensor,
    block_tables: torch.Tensor,
) -> None:
    """Fused compress attention: one MXFP4 cache row for each plan row.

    A plan row merges its last 8 tokens with a softmax pool. Then it applies RMSNorm
    to the 128 values and interleaved RoPE (adjacent pairs rotate together) to the
    last 64 values. It writes e2m1 values with one e8m0 scale for each 32 values.
    The bytes are the same as ``aiter.ops.torch_ref.fused_compress_attn`` with
    overlap=True, quant_mode="fp4" and preshuffle=False.

    Args:
        kv_in, score_in: [T, 256] bf16 input rows with the same row stride.
        kv_state, score_state: [S, state_size, 256] f32 state rings. The slots
            can be strided.
        state_slots: [B] int32 state slot of each sequence.
        plan: [P, 4] int32 rows of (last_token_row, seq, last_pos, num_from_state).
            num_from_state is the number of window tokens from the state ring. A row
            with a negative last_pos writes nothing.
        ape: [4, 256] f32 position bias of the input scores.
        rms_weight: [128] RMSNorm weight.
        rms_eps: RMSNorm epsilon.
        cos, sin: [max_pos, 32] RoPE tables.
        kv_cache: [NB, k_per_block, 64] uint8, two e2m1 values in each byte.
        cache_scale: [NB, k_per_block, 4] uint8 e8m0 scales.
        block_tables: [B, max_blocks] int32 physical block of each cache block.

    Returns:
        None. It writes kv_cache and cache_scale.

    Limits: gfx1250 only.
    """
    assert arch_info.get_arch() == "gfx1250", "this kernel is for gfx1250 only"
    assert kv_in.shape[1] == 2 * HEAD_DIM and cos.shape[1] == ROPE_DIM // 2
    assert ape.shape[0] == RATIO
    # The kernel reads both inputs with one stride and 16-byte loads.
    assert kv_in.stride(0) == score_in.stride(0) and kv_in.stride(0) % 8 == 0
    assert kv_in.stride(1) == 1 and score_in.stride(1) == 1
    # The kernel takes the other strides as constants.
    for state in (kv_state, score_state):
        assert (
            state.stride(1) == 2 * HEAD_DIM
            and state.stride(2) == 1
            and state.stride(0) % 8 == 0
        )
    for tensor in (plan, ape, cos, sin, kv_cache, cache_scale):
        assert tensor.is_contiguous()
    assert block_tables.stride(1) == 1

    _LOGGER.info("FUSED_COMPRESS_ATTN_FP4: plan rows=%d", plan.shape[0])
    config = _get_config(plan.shape[0])
    fused_compress_attn_fp4_gfx1250[(plan.shape[0],)](
        plan,
        state_slots,
        block_tables,
        kv_state,
        score_state,
        kv_in,
        score_in,
        ape,
        rms_weight,
        cos,
        sin,
        kv_cache,
        cache_scale,
        state_slots.numel(),
        block_tables.stride(0),
        kv_in.stride(0),
        RMS_EPS=rms_eps,
        STATE_SIZE=kv_state.shape[1],
        K_PER_BLOCK=kv_cache.shape[1],
        KV_STATE_SLOT_STRIDE=kv_state.stride(0),
        SCORE_STATE_SLOT_STRIDE=score_state.stride(0),
        **config,
        **get_fused_compress_attn_fp4_layouts(config["num_warps"]),
    )
