# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""fused_qknorm_idxrqknorm's FP8-cache insert, byte for byte, for MiniMax-M3.1 at TP1
(64 q / 4 kv / 4 index heads of 128, rotary 64) on gfx950; faster from ~640 tokens."""

from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime

__all__ = ["qknorm_v3"]


def is_supported() -> bool:
    """Whether qknorm_v3 runs on this device (gfx950 only; False without a GPU)."""
    try:
        return get_gfx_runtime() == "gfx950"
    except RuntimeError:
        return False


@compile_ops("module_m3_tp1_qknorm", fc_name="m3_tp1_qknorm_v3", develop=True)
def _qknorm_v3(
    qkv: Tensor,
    q_out: Tensor,
    index_q_out: Tensor,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    index_q_norm_weight: Tensor,
    index_k_norm_weight: Tensor,
    cos_sin_cache: Tensor,
    positions: Tensor,
    slot_mapping: Tensor,
    index_slot_mapping: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    index_cache: Tensor,
    k_scale: Tensor,
    v_scale: Tensor,
    eps: float,
    block_size: int,
) -> None: ...


def qknorm_v3(
    qkv: Tensor,  # [T, 9856] bf16: q 64 | k 4 | v 4 | index q 4 | index k 1 heads of 128
    q_out: Tensor,  # [T, 64 * 128] bf16
    index_q_out: Tensor,  # [T, 4 * 128] bf16
    q_norm_weight: Tensor,  # [128] bf16 each, applied as x * (1 + w)
    k_norm_weight: Tensor,
    index_q_norm_weight: Tensor,
    index_k_norm_weight: Tensor,
    cos_sin_cache: Tensor,  # [max_pos, 64] bf16: cos | sin
    positions: Tensor,  # [T] int64
    slot_mapping: Tensor,  # [T] int64, -1 skips the token's K / V
    index_slot_mapping: Tensor,  # [T] int64, -1 skips the token's index K
    k_cache: Tensor,  # [num_blocks, 4, 8, 16, 16] fp8, page-16 SHUFFLE
    v_cache: Tensor,  # [num_blocks, 4, 1, 128, 16] fp8, page-16 SHUFFLE
    index_cache: Tensor,  # [rows, 128] fp8
    k_scale: Tensor,  # [num_blocks, 4, 16] fp32, per-token scales written
    v_scale: Tensor,  # [num_blocks, 4, 16] fp32
    eps: float,
    block_size: int,  # 16
) -> None:
    if not is_supported():
        raise RuntimeError(f"qknorm_v3 requires gfx950, got {get_gfx_runtime()}")
    _qknorm_v3(
        qkv,
        q_out,
        index_q_out,
        q_norm_weight,
        k_norm_weight,
        index_q_norm_weight,
        index_k_norm_weight,
        cos_sin_cache,
        positions,
        slot_mapping,
        index_slot_mapping,
        k_cache,
        v_cache,
        index_cache,
        k_scale,
        v_scale,
        eps,
        block_size,
    )
