# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton._gluon_kernels.gfx1250.fusions.fused_qk_norm_rope_group_quant import (
    HEAD_DIM,
    ROPE_DIM,
    fused_qk_norm_rope_group_quant_gfx1250,
    get_fused_qk_norm_rope_group_quant_layouts,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.types import e4m3_dtype

_LOGGER = AiterTritonLogger()


def fused_qk_norm_rope_group_quant(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    swa_nope: torch.Tensor,
    swa_rope: torch.Tensor,
    swa_dest_rows: torch.Tensor,
    batch_ids: torch.Tensor,
):
    """Q/K RMSNorm, RoPE, fp8 group quant and SWA cache write.

    Each row of 512 values gets an RMSNorm. K also uses ``k_weight``. Values
    [0, 448) go to fp8 e4m3 with one e8m0 scale for each 64 values. Values
    [448, 512) get RoPE on adjacent value pairs with ``cos/sin[positions]``.
    The K row of token t also goes to row ``swa_dest_rows[t]`` of the SWA cache.

    Args:
        q: [T, H, 512] contiguous bf16. H must be a multiple of 8.
        kv: [T, 512] bf16.
        k_weight: [512] RMSNorm weight of K.
        positions: [T] int.
        cos, sin: [max_pos, 32] contiguous bf16.
        eps: RMSNorm epsilon.
        swa_nope, swa_rope: contiguous SWA cache, [rows, 512] fp8 and [rows, 64] bf16.
        swa_dest_rows: [T] int32 SWA row of each token. A negative row skips the write.
        batch_ids: [T] int32. A negative batch id skips the SWA write.

    Returns:
        q_nope [T, H, 512] fp8, q_rope [T, H, 64] bf16, k_nope [T, 512] fp8 and
        k_rope [T, 64] bf16. Bytes [448, 462) of a row hold the 7 scales, each
        written twice. The kernel does not write bytes [462, 512).
    """
    assert arch_info.get_arch() == "gfx1250"
    num_tokens, num_heads, head_dim = q.shape
    assert head_dim == HEAD_DIM and num_heads % 8 == 0
    assert kv.shape == (num_tokens, head_dim)
    assert cos.shape[1] == ROPE_DIM // 2 and cos.is_contiguous() and sin.is_contiguous()
    # The kernel uses fixed row strides for q and the SWA cache.
    assert q.is_contiguous() and swa_nope.is_contiguous() and swa_rope.is_contiguous()
    # The TDM loads need contiguous rows on 16-byte boundaries.
    assert kv.stride(1) == 1 and kv.stride(0) * kv.element_size() % 16 == 0
    assert q.data_ptr() % 16 == 0 and kv.data_ptr() % 16 == 0

    q_nope = torch.empty(
        (num_tokens, num_heads, head_dim), dtype=e4m3_dtype, device=q.device
    )
    q_rope = torch.empty(
        (num_tokens, num_heads, ROPE_DIM), dtype=q.dtype, device=q.device
    )
    k_nope = torch.empty((num_tokens, head_dim), dtype=e4m3_dtype, device=q.device)
    k_rope = torch.empty((num_tokens, ROPE_DIM), dtype=q.dtype, device=q.device)
    _LOGGER.info("FUSED_QK_NORM_ROPE_GROUP_QUANT: T=%d H=%d", num_tokens, num_heads)

    # Measured on gfx1250. waves_per_eu is the number of waves on each SIMD that the
    # register use must allow. Up to T = 512, steps of 4 rows fit 9 waves. Above
    # T = 512, steps of 8 rows run fewer instructions and fit 6 waves. T <= 256
    # needs 4 heads per program to make enough programs.
    if num_tokens <= 256:
        heads_per_program, rows_per_step, waves_per_eu = 4, 4, 9
    elif num_tokens <= 512:
        heads_per_program, rows_per_step, waves_per_eu = 8, 4, 9
    else:
        heads_per_program, rows_per_step, waves_per_eu = 8, 8, 6
    # One program for the K row of each token, then the Q programs.
    num_q_programs = num_tokens * num_heads // heads_per_program
    grid = (num_tokens + num_q_programs,)
    fused_qk_norm_rope_group_quant_gfx1250[grid](
        q,
        positions,
        cos,
        sin,
        q_nope,
        q_rope,
        num_tokens,
        cos.shape[0],
        kv,
        k_weight,
        k_nope,
        k_rope,
        kv.stride(0),
        swa_dest_rows,
        batch_ids,
        swa_nope,
        swa_rope,
        EPS=eps,
        NUM_HEADS=num_heads,
        HEADS_PER_PROGRAM=heads_per_program,
        ROWS_PER_STEP=rows_per_step,
        num_warps=1,
        waves_per_eu=waves_per_eu,
        **get_fused_qk_norm_rope_group_quant_layouts(),
    )
    return q_nope, q_rope, k_nope, k_rope
