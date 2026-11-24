# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Token-tiled FP8 sparse prefill attention for MiniMax-M3 at TP1 (gfx950): a workgroup
takes TT_TQ tokens of one request x the 16 query heads of one KV head."""

from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime

MD_NAME = "module_m3_tt_sparse_prefill"
TT_TQ = 12  # query tokens per workgroup: the longest tile of a tile map
# a workgroup's bitmap of selected blocks holds 4096 blocks of 128 tokens
TT_MAX_SEQ_LEN = 4096 * 128


def is_supported() -> bool:
    """True on gfx950, the only arch it runs on (False without a GPU)."""
    try:
        return get_gfx_runtime() == "gfx950"
    except RuntimeError:
        return False


@compile_ops(MD_NAME, fc_name="m3_tt_sparse_prefill", develop=True)
def _m3_tt_sparse_prefill(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    k_scale: Tensor,
    v_scale: Tensor,
    topk_idx: Tensor,
    block_table: Tensor,
    row_req: Tensor,
    row_kv_len: Tensor,
    tile_t0: Tensor,
    tile_n: Tensor,
    ntiles: int,
    sm_scale: float,
    out: Tensor,
) -> None: ...


def tt_sparse_prefill(
    q: Tensor,  # [T, 16 * hkv, 128] bf16
    k_cache: Tensor,  # [pages, hkv, 8, 16, 16] fp8 page-16 SHUFFLE, 8 pages a block
    v_cache: Tensor,  # [pages, hkv, 1, 128, 16] fp8
    k_scale: Tensor,  # [pages, hkv, 16] fp32, one per key
    v_scale: Tensor,  # [pages, hkv, 16] fp32
    topk_idx: Tensor,  # [hkv, T, 16] int32: distinct 128-token blocks per row, -1 pads
    block_table: Tensor,  # [batch, >= blocks] int32: request block -> cache block
    row_req: Tensor,  # [T] int32: each row's block_table row
    row_kv_len: Tensor,  # [T] int32: each row's causal key count
    tile_t0: Tensor,  # [ntiles] int32: first row of each tile
    tile_n: Tensor,  # [ntiles] int32: <= TT_TQ rows of one request, 0 pads
    ntiles: int,
    sm_scale: float,
    max_seq_len: int,  # the step's longest sequence
    out: Tensor,  # [T, 16 * hkv, 128] bf16
) -> None:
    """Writes every row's attention to bf16 `out`. Every row is in exactly one tile."""
    if max_seq_len > TT_MAX_SEQ_LEN:
        raise ValueError(
            f"tt_sparse_prefill: max_seq_len {max_seq_len} > {TT_MAX_SEQ_LEN}, "
            "block ids past the kernel's bitmap would be dropped"
        )
    if not is_supported():
        raise RuntimeError(
            f"tt_sparse_prefill supports gfx950 only, got {get_gfx_runtime()}"
        )
    _m3_tt_sparse_prefill(
        q,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        topk_idx,
        block_table,
        row_req,
        row_kv_len,
        tile_t0,
        tile_n,
        ntiles,
        sm_scale,
        out,
    )
