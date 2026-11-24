# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3 lightning-indexer FP8 prefill block scorer fused with a per-chunk
top-16 (gfx950).

For index head ``h``, query token ``t`` at global position ``pos`` and 128-token
block ``b`` of its request, the indexer scores

    score[h, t, b] = max over the block's keys of dot(e4m3(q[t, h]), K[b, key]) * scale

with ``e4m3(q)`` the RNE rounding of the bf16 query, one fp32 k=128 MFMA dot per
key, and the max taken before the fp32 multiply. Instead of writing that
``[4, total_q, blocks]`` fp32 plane, the kernel keeps, per (head, token), the 16
best keys of each chunk of ``chunk_blocks`` blocks in registers and writes only
those: ``partial[h, t, 16 * c : 16 * c + 16]`` holds chunk ``c``'s keys, descending.

A key is ``1 << 48 | ordered(score) << 16 | (b + 1)`` for a block ``b <= pos // 128``,
where ``ordered`` maps the fp32 bits to an unsigned order (-0 as +0, NaN as 0)
and the query's own block ``pos // 128`` scores 1e29. A chunk with fewer than 16
such blocks is padded with 15, 14, ... (bit 48 clear). Keys order blocks by
score, then by the higher block id, so the top 16 of a row's ``nchunks * 16``
keys is its top-16 selection.

Fixed shape: 4 index heads, head_dim 128, an e4m3 index cache of 128-token
pages, top-16. ``nchunks * chunk_blocks`` must cover every request's blocks, and
block ids stay below 65535.
"""

from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime

__all__ = ["fused_index_topk_partial"]

MD_NAME = "module_m3_fused_index_topk"


def is_supported() -> bool:
    """True on gfx950, the only arch it runs on (False without a GPU)."""
    try:
        return get_gfx_runtime() == "gfx950"
    except RuntimeError:
        return False


@compile_ops(MD_NAME, fc_name="m3_fused_index_topk_partial", develop=True)
def _fused_index_topk_partial(
    idx_q: Tensor,
    index_cache: Tensor,
    partial: Tensor,
    block_table: Tensor,
    cu_seqlens_q: Tensor,
    prefix_lens: Tensor,
    max_query_len: int,
    nchunks: int,
    chunk_blocks: int,
    scale: float,
) -> None: ...


def fused_index_topk_partial(
    idx_q: Tensor,  # [total_q, 4, 128] bf16, head stride 128
    index_cache: Tensor,  # [pages, 128, 128] e4m3
    partial: Tensor,  # [4, total_q, nchunks * 16] int64, written
    block_table: Tensor,  # [batch, >= blocks] int32
    cu_seqlens_q: Tensor,  # [batch + 1] int32
    prefix_lens: Tensor,  # [batch] int32: cached tokens before each request's queries
    max_query_len: int,
    nchunks: int,
    chunk_blocks: int,
    scale: float,  # > 0; MiniMax-M3: fp32(sm_scale) * fp32(log2 e)
) -> None:
    """Write each block chunk's 16 best keys of every (index head, query token)
    into ``partial`` (see the module docstring)."""
    if not is_supported():
        raise RuntimeError(
            f"fused_index_topk_partial supports gfx950 only, got {get_gfx_runtime()}"
        )
    _fused_index_topk_partial(
        idx_q,
        index_cache,
        partial,
        block_table,
        cu_seqlens_q,
        prefix_lens,
        int(max_query_len),
        int(nchunks),
        int(chunk_blocks),
        float(scale),
    )
