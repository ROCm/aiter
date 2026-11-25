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
    HEAD_DIM,
    PAGE,
    compile_index_score_prefill_nvfp4,
    sm_scale_log2e,
)
from .kernels.tensor_shim import _run_compiled, ptr_arg


def nvfp4_index_inputs_supported(idx_q, payload, scale) -> bool:
    """What the gfx1250 index scorers (prefill and decode) share: a bf16 query
    ``[rows, heads, 128]`` and the NVFP4 index cache as two byte views with
    whole contiguous rows (the scales uint8 or an FP8 dtype), the payload pages
    16 B apart."""
    return (
        get_gfx_runtime() == "gfx1250"
        and idx_q.dim() == 3
        and idx_q.shape[2] == HEAD_DIM
        and idx_q.dtype == torch.bfloat16
        and idx_q.stride(2) == 1
        and payload.dtype == torch.uint8
        and payload.shape[1:] == (PAGE, HEAD_DIM // 2)
        and payload.stride()[1:] == (HEAD_DIM // 2, 1)
        and scale.element_size() == 1
        and scale.shape[1:] == (PAGE, HEAD_DIM // 16)
        and scale.stride()[1:] == (HEAD_DIM // 16, 1)
        and payload.stride(0) % 16 == 0
    )


def index_score_prefill_nvfp4_supported(idx_q, payload, scale) -> bool:
    return (
        nvfp4_index_inputs_supported(idx_q, payload, scale)
        and scale.stride(0) % 16 == 0
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
    lds_scores: bool = True,
) -> None:
    """``score[h, r, blk]`` for every block up to the last row of r's 256-row
    tile, exactly as ATOM's Triton prefill scorer (same bits). ``lds_scores``:
    a CTA scores at most ``CHUNK_MAX`` blocks (``chunk_blocks`` is capped; the
    scores do not depend on the chunking) and stores each row's chunk with one
    coalesced store instead of a 4-byte store a block."""
    total_q, heads, _ = idx_q.shape
    batch = cu_seqlens_q.shape[0] - 1
    max_block = score.shape[2]
    assert score.stride(2) == 1 and block_table.stride(1) == 1
    assert score.shape[0] == heads and score.dtype == torch.float32
    if total_q == 0 or batch == 0:
        return
    if waves < 2:
        # Wave 0 copies a page's payload and wave 1 its scales: with one wave
        # the scales never arrive.
        raise ValueError(f"index_score_prefill_nvfp4 needs waves >= 2, got {waves}")
    if lds_scores:
        chunk_blocks = min(chunk_blocks, CHUNK_MAX)
    q_tiles = (max_query_len + block_q - 1) // block_q
    chunks = (max_block + chunk_blocks - 1) // chunk_blocks
    compiled = compile_index_score_prefill_nvfp4(
        block_q=block_q,
        waves=waves,
        ring=ring,
        chunk_max=CHUNK_MAX if lds_scores else 0,
    )
    dev = idx_q.device
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
