# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""MiniMax-M3 prefill index block scores over the NVFP4 index cache (gfx1250).

Bitwise drop-in for ATOM's Triton ``_index_block_score_kernel`` on the FP4 index
cache; see ``kernels.index_score_prefill_nvfp4_gfx1250``.
"""

import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.index_score_prefill_nvfp4_gfx1250 import (
    CHUNK_MAX,
    compile_index_score_prefill_nvfp4,
    sm_scale_log2e,
)
from .kernels.minimax_m3_index_score import HEAD_DIM, PAGE
from .kernels.tensor_shim import _run_compiled, ptr_arg


def index_score_prefill_nvfp4_supported(idx_q, payload, scale) -> bool:
    return (
        get_gfx_runtime() == "gfx1250"
        and idx_q.dim() == 3
        and idx_q.shape[2] == HEAD_DIM
        and idx_q.dtype == torch.bfloat16
        and idx_q.stride(2) == 1
        and payload.dtype == torch.uint8
        and payload.shape[1:] == (PAGE, HEAD_DIM // 2)
        and payload.stride()[1:] == (HEAD_DIM // 2, 1)
        and scale.shape[1:] == (PAGE, HEAD_DIM // 16)
        and scale.stride()[1:] == (HEAD_DIM // 16, 1)
        and payload.stride(0) % 16 == 0
        and scale.stride(0) % 16 == 0
        and payload.data_ptr() % 16 == 0
        and scale.data_ptr() % 16 == 0
    )


def index_score_prefill_nvfp4(
    idx_q: torch.Tensor,  # [total_q, heads, 128] bf16
    payload: torch.Tensor,  # [blocks, 128, 64] uint8
    scale: torch.Tensor,  # [blocks, 128, 8] uint8 (E4M3 bits)
    score: torch.Tensor,  # [heads, total_q, max_block] fp32, written like Triton's
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
    """``score[h, r, blk]`` for every block up to the last row of r's 256-row
    tile, exactly as ATOM's Triton prefill scorer (same bits). A CTA scores at
    most ``CHUNK_MAX`` blocks (``chunk_blocks`` is capped; the scores do not
    depend on the chunking) and stores each row's chunk with one coalesced store."""
    total_q, heads, _ = idx_q.shape
    batch = cu_seqlens_q.shape[0] - 1
    max_block = score.shape[2]
    if total_q == 0 or batch == 0:
        return
    # Everything reaches the kernel as a bare pointer: check what it assumes.
    dev, meta = idx_q.device, (block_table, cu_seqlens_q, seq_lens, prefix_lens)
    if not (
        index_score_prefill_nvfp4_supported(idx_q, payload, scale)
        and all(t.device == dev for t in (payload, scale, score) + meta)
        and all(t.dtype == torch.int32 for t in meta)
        and score.dtype == torch.float32
        and score.shape[0] == heads
        and score.shape[1] >= total_q
        and score.stride(2) == 1
        and score.stride(0) <= 0x7FFFFFFF  # the kernel takes int32 strides
        and block_table.stride(1) == 1
    ):
        raise ValueError(
            "index_score_prefill_nvfp4: unsupported device, dtype or layout"
        )
    chunk_blocks = min(chunk_blocks, CHUNK_MAX)
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
