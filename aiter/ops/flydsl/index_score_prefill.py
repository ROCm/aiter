# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Causal prefill block scores of index queries over an NVFP4 key cache (gfx1250).

See ``kernels.index_score_prefill_nvfp4_gfx1250``.
"""

import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.index_score_prefill_nvfp4_gfx1250 import (
    CHUNK_MAX,
    compile_index_score_prefill_nvfp4,
    sm_scale_log2e,
)
from .kernels.minimax_m3_index_score import (
    DATA_BYTES,
    HEAD_DIM,
    NVFP4_SCALE_BYTES,
    PAGE,
)
from .kernels.tensor_shim import _run_compiled, ptr_arg


def _byte_pages(t, row_bytes):
    """``[pages, PAGE, row_bytes]`` bytes, each page contiguous and 16 B aligned:
    a page is one tensor DMA of ``PAGE * row_bytes`` bytes."""
    return (
        t.element_size() == 1
        and t.shape[1:] == (PAGE, row_bytes)
        and t.stride()[1:] == (row_bytes, 1)
        and t.stride(0) % 16 == 0
        and t.data_ptr() % 16 == 0
    )


def index_score_prefill_nvfp4_supported(
    idx_q,
    payload,
    scale,
    score,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    sm_scale,
) -> bool:
    """Whether ``index_score_prefill_nvfp4`` takes these, its first nine arguments.
    The kernel gets bare pointers and int32 strides, so these are what it assumes
    of them; the op raises ``ValueError`` when this is False."""
    meta = (block_table, cu_seqlens_q, seq_lens, prefix_lens)
    batch = cu_seqlens_q.shape[0] - 1
    return (
        get_gfx_runtime() == "gfx1250"
        and idx_q.dtype == torch.bfloat16
        and idx_q.shape[2:] == (HEAD_DIM,)
        and idx_q.stride(2) == 1
        and _byte_pages(payload, DATA_BYTES)
        and _byte_pages(scale, NVFP4_SCALE_BYTES)
        and all(t.device == idx_q.device for t in (payload, scale, score) + meta)
        and all(t.dtype == torch.int32 and t.stride(-1) == 1 for t in meta)
        and min(block_table.shape[0], seq_lens.shape[0], prefix_lens.shape[0]) >= batch
        and score.dtype == torch.float32
        and score.dim() == 3
        and score.shape[0] == idx_q.shape[1]
        and score.shape[1] >= idx_q.shape[0]
        and score.stride(2) == 1
        and max(score.stride()) < 2**31
        # The kernel takes the block max before the scale.
        and 0.0 < sm_scale_log2e(sm_scale) < float("inf")
    )


def index_score_prefill_nvfp4(
    idx_q: torch.Tensor,  # [total_q, heads, 128] bf16
    payload: torch.Tensor,  # [pages, 128, 64] uint8: e2m1 pairs
    scale: torch.Tensor,  # [pages, 128, 8] uint8 / float8_e4m3fn: one per 16 values
    score: torch.Tensor,  # [heads, >= total_q, max_block] fp32
    block_table: torch.Tensor,  # [reqs, max_blocks] int32
    cu_seqlens_q: torch.Tensor,  # [reqs + 1] int32
    seq_lens: torch.Tensor,  # [reqs] int32
    prefix_lens: torch.Tensor,  # [reqs] int32
    sm_scale: float,
    max_query_len: int,
    chunk_blocks: int,
    *,
    block_q: int = 256,
    waves: int = 4,
    ring: int = 4,
) -> None:
    """For query row ``r`` of request ``b`` (at position ``prefix_lens[b] + r -
    cu_seqlens_q[b]``), index head ``h`` and 128-key block ``blk`` of ``b``'s pages
    (page ``block_table[b, blk]``)::

        score[h, r, blk] = max over the block's keys k at or before r's position of
                           dot(e4m3(clamp(idx_q[r, h], +-448)), dequant(K[k]))
                           * (sm_scale * log2(e))

    ``-inf`` for a block past the row's own. Written for every block up to the
    block of the last row of r's ``block_q``-row tile; later entries of ``score``
    are left as they are. A CTA scores at most ``CHUNK_MAX`` blocks
    (``chunk_blocks`` is clamped to ``[1, CHUNK_MAX]``; the scores do not depend on
    it).

    What only the device can see is the caller's to guarantee: ``cu_seqlens_q``
    ends at ``total_q``, ``max_query_len`` covers every request, and
    ``block_table`` and ``score`` hold ``ceil(seq_len / 128)`` blocks of each
    request (pages inside the cache)."""
    if not index_score_prefill_nvfp4_supported(
        idx_q,
        payload,
        scale,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        sm_scale,
    ):
        raise ValueError(
            "index_score_prefill_nvfp4: unsupported device, dtype, layout or scale"
        )
    total_q, heads, _ = idx_q.shape
    batch = cu_seqlens_q.shape[0] - 1
    max_block = score.shape[2]
    if total_q == 0 or batch == 0:
        return
    dev = idx_q.device
    chunk_blocks = max(1, min(chunk_blocks, CHUNK_MAX))
    q_tiles = (max_query_len + block_q - 1) // block_q
    chunks = (max_block + chunk_blocks - 1) // chunk_blocks
    compiled = compile_index_score_prefill_nvfp4(
        block_q=block_q,
        waves=waves,
        ring=ring,
    )
    _run_compiled(
        compiled["launch"],
        ptr_arg(idx_q, fx.BFloat16),
        ptr_arg(payload, fx.Uint8),
        ptr_arg(scale, fx.Uint8),
        ptr_arg(score, fx.Float32),
        ptr_arg(block_table, fx.Int32),
        ptr_arg(cu_seqlens_q, fx.Int32),
        ptr_arg(seq_lens, fx.Int32),
        ptr_arg(prefix_lens, fx.Int32),
        heads,
        sm_scale_log2e(sm_scale),
        chunk_blocks,
        idx_q.stride(0),
        idx_q.stride(1),
        payload.stride(0),
        scale.stride(0),
        score.stride(0),
        score.stride(1),
        block_table.stride(0),
        q_tiles,
        batch,
        chunks,
        torch.cuda.current_stream(dev),
        specialization_key=(dev.index,),
    )
