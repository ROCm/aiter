# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""MiniMax-M3 prefill index block scores over the NVFP4 index cache (gfx1250).

Bitwise drop-in for ATOM's Triton ``_index_block_score_kernel`` on the FP4 index
cache, and the same scorer fused with the top-16 block selection; see
``kernels.index_score_prefill_nvfp4_gfx1250``.
"""

import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

from .kernels.index_score_prefill_nvfp4_gfx1250 import (
    CHUNK_MAX,
    HEAD_DIM,
    INT32_MIN,
    PAGE,
    TOPK,
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
    fold_max: bool = False,
    pad_lds: bool = False,
    _probe: str = "",
) -> None:
    """``score[h, r, blk]`` for every block up to the last row of r's 256-row
    tile, exactly as ATOM's Triton prefill scorer (same bits). ``lds_scores``:
    a CTA scores at most ``CHUNK_MAX`` blocks (``chunk_blocks`` is capped; the
    scores do not depend on the chunking) and stores each row's chunk with one
    coalesced store instead of a 4-byte store a block. ``fold_max`` (the same
    scores, fewer VALU instructions), ``pad_lds``
    (the same scores, no LDS bank conflicts on the key rows) and
    ``_probe`` (debug timing builds, wrong
    scores): see ``compile_index_score_prefill_nvfp4``."""
    total_q, heads, _ = idx_q.shape
    batch = cu_seqlens_q.shape[0] - 1
    max_block = score.shape[2]
    assert score.stride(2) == 1 and block_table.stride(1) == 1
    assert score.shape[0] == heads and score.dtype == torch.float32
    if total_q == 0 or batch == 0:
        return
    if lds_scores:
        chunk_blocks = min(chunk_blocks, CHUNK_MAX)
    q_tiles = (max_query_len + block_q - 1) // block_q
    chunks = (max_block + chunk_blocks - 1) // chunk_blocks
    compiled = compile_index_score_prefill_nvfp4(
        block_q=block_q,
        waves=waves,
        ring=ring,
        chunk_max=CHUNK_MAX if lds_scores else 0,
        fold_max=bool(fold_max),
        pad_lds=bool(pad_lds),
        _probe=_probe,
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


# Blocks a workgroup of the fused scorer + top-16 owns: a row hands on ``TOPK``
# keys a chunk. A longer chunk loads its query tile and fills its page ring
# fewer times, tries fewer blocks (a list restarts with its chunk) and leaves
# the merge fewer keys; a shorter one gives a short prefill more workgroups.
# The default is the longest power of two in [TOPK_CHUNK_MIN, TOPK_CHUNK_MAX]
# that still gives each CU TOPK_WORKGROUPS_PER_CU workgroups (a guess: 3-4 are
# resident, the rest balance the tail), and never so short that a row hands the
# merge more than TOPK_MERGE_MAX keys. Not measured on a GPU yet.
TOPK_CHUNK_MIN, TOPK_CHUNK_MAX = 32, 1024
TOPK_WORKGROUPS_PER_CU = 16
TOPK_MERGE_MAX = 1024


def index_topk_prefill_chunk_blocks(
    idx_q, max_query_len, batch, max_block, *, block_q=256, **_launch
) -> int:
    """The default ``chunk_blocks`` of :func:`index_topk_prefill_nvfp4` for this
    grid (the other launch keywords are accepted and do not matter). A power of
    two, so a caller compiling per chunk sees few values."""
    tiles = batch * idx_q.shape[1] * -(-max_query_len // block_q)
    floor = TOPK_CHUNK_MIN
    while floor < TOPK_CHUNK_MAX and -(-max_block // floor) * TOPK > TOPK_MERGE_MAX:
        floor *= 2
    chunk = TOPK_CHUNK_MAX
    while (
        chunk > floor
        and tiles * -(-max_block // chunk) < TOPK_WORKGROUPS_PER_CU * get_cu_num()
    ):
        chunk //= 2
    return chunk


def index_topk_prefill_nvfp4_supported(idx_q, payload, scale, topk, max_block) -> bool:
    """Whether :func:`index_topk_prefill_nvfp4` serves this call: what the scorer
    serves, 16 blocks a row and block ids that fit the key's 16 bits."""
    return (
        index_score_prefill_nvfp4_supported(idx_q, payload, scale)
        and topk == TOPK
        and max_block <= 0xFFFF
    )


def index_topk_prefill_nvfp4(
    idx_q: torch.Tensor,  # [total_q, heads, 128] bf16
    payload: torch.Tensor,  # [blocks, 128, 64] uint8
    scale: torch.Tensor,  # [blocks, 128, 8] uint8 (E4M3 bits)
    block_table: torch.Tensor,  # [reqs, max_blocks] int32
    cu_seqlens_q: torch.Tensor,  # [reqs + 1] int32
    seq_lens: torch.Tensor,  # [reqs] int32
    prefix_lens: torch.Tensor,  # [reqs] int32
    sm_scale: float,
    max_query_len: int,
    max_block: int,
    init_blocks: int,
    local_blocks: int,
    *,
    chunk_blocks: int | None = None,
    block_q: int = 256,
    waves: int = 4,
    ring: int = 4,
    fold_max: bool = False,
    pad_lds: bool = False,
    _probe: str = "",
) -> torch.Tensor:
    """The prefill scorer and ATOM's top-16 block selection in one launch: the
    scores of :func:`index_score_prefill_nvfp4` never reach memory.

    Returns ``[heads, total_q, chunks * 16]`` int64 (not contiguous), chunks =
    ceil(max_block / chunk_blocks): at ``z * 16`` the 16 best blocks of row r
    among the blocks ``[z, z + 1) * chunk_blocks`` it may attend, as the keys of
    ATOM's ``_pack_score_key`` (score, then the larger block; a NaN last; the
    first ``init_blocks`` and the row's last ``local_blocks`` blocks pinned),
    in no order, 0 for no block. The 16 largest keys of a row are its selection,
    in order: every one of its 16 best blocks is in its chunk's 16, though a
    chunk may hold fewer than its own 16 best (blocks another chunk's rule out).
    Chunks from ``ceil(blocks of r / chunk_blocks)`` on hold no key or are not
    written; rows past a request's ``cu_seqlens_q`` are not written."""
    total_q, heads, _ = idx_q.shape
    batch = cu_seqlens_q.shape[0] - 1
    assert block_table.stride(1) == 1
    if chunk_blocks is None:
        chunk_blocks = index_topk_prefill_chunk_blocks(
            idx_q, max_query_len, batch, max_block, block_q=block_q
        )
    if chunk_blocks < 1:
        raise ValueError(f"chunk_blocks must be positive, got {chunk_blocks}")
    chunks = (max_block + chunk_blocks - 1) // chunk_blocks
    # Column 0: the row's floor, shared by its workgroups.
    keys = torch.empty(
        (heads, total_q, 1 + chunks * TOPK), dtype=torch.int64, device=idx_q.device
    )
    if total_q == 0 or batch == 0 or chunks == 0:
        return keys[:, :, 1:]
    keys[:, :, 0] = INT32_MIN
    compiled = compile_index_score_prefill_nvfp4(
        block_q=block_q,
        waves=waves,
        ring=ring,
        chunk_max=CHUNK_MAX,
        select=(init_blocks, local_blocks),
        fold_max=bool(fold_max),
        pad_lds=bool(pad_lds),
        _probe=_probe,
    )
    dev = idx_q.device
    _run_compiled(
        compiled["launch"],
        ptr_arg(idx_q, fx.BFloat16),
        ptr_arg(payload, fx.Uint8),
        ptr_arg(scale, fx.Uint8),
        ptr_arg(keys, fx.Int64),
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
        keys.stride(0),
        keys.stride(1),
        block_table.stride(0),
        (max_query_len + block_q - 1) // block_q,
        batch,
        chunks,
        torch.cuda.current_stream(dev),
        specialization_key=(dev.index,),
    )
    return keys[:, :, 1:]
