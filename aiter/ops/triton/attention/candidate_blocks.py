# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Level-one candidate selection for the DeepSeek-V4.1-Flash hierarchical indexer.

The first indexing layer scores every visible compressed position, reduces the
scores to blocks of ``block_size`` (block score = max over its positions), keeps
the ``topk_blocks`` best blocks per query, and hands the later indexing layers a
position mask so they only search inside that pool.

Split into three ops so the top-k in the middle stays a reusable AITER kernel:

    candidate_block_scores     -> [T, num_blocks] fp32 block scores
    <top-k over the blocks>    -> [T, k] block ids
    candidate_mask_from_blocks -> [T, width] bool position mask

``select_candidate_blocks`` chains them; ``apply_candidate_mask`` applies the
result to the next layer's logits.
"""

import functools

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.candidate_blocks import (
    _apply_candidate_mask_kernel,
    _candidate_block_scores_kernel,
    _candidate_mask_scatter_kernel,
)
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_CONFIG_NAME = "CANDIDATE-BLOCKS"


def _get_config(kernel_name: str) -> dict:
    cfg_dir = resolve_config_dir("attention", _CONFIG_NAME, backend="triton")
    return dict(load_config_json(f"{cfg_dir}/DEFAULT.json")[kernel_name])


@functools.lru_cache(maxsize=32)
def _row_bounds(num_rows: int, num_blocks: int, device_index: int):
    """Cached full-row start/end bounds for the per-row top-k."""
    device = torch.device("cuda", device_index)
    starts = torch.zeros(num_rows, dtype=torch.int32, device=device)
    ends = torch.full((num_rows,), num_blocks, dtype=torch.int32, device=device)
    return starts, ends


def candidate_block_scores(
    logits: torch.Tensor,
    compress_lens,
    block_size: int = 8,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce position logits to per-block scores.

    Pads the last block out with -inf, takes the max over each block, and pins
    the query's own (partially filled) last block to +inf so it is always kept.

    Args:
        logits: [T, L] fp32. Positions the query cannot see are already -inf.
        compress_lens: int, or int32 tensor of shape [T] / [T, 1] / scalar,
            giving the visible compressed length of each query.
        block_size: positions per candidate block (8 in DSV4.1-Flash).
        out: optional [T, num_blocks] fp32 destination.

    Returns:
        [T, num_blocks] fp32 block scores, num_blocks = ceil(L / block_size).

    Config (attention/candidate_blocks/DEFAULT.json):
        BLOCKS_PER_TILE - blocks reduced per program; num_warps / num_stages.
    """
    assert logits.ndim == 2, "logits must be [T, L]; flatten leading dims first"
    assert logits.dtype == torch.float32, "logits must be fp32"
    assert block_size > 0
    T, L = logits.shape
    num_blocks = triton.cdiv(L, block_size)
    _LOGGER.info(
        f"CANDIDATE_BLOCK_SCORES: logits={tuple(logits.shape)} block_size={block_size}"
    )

    if out is None:
        out = torch.empty((T, num_blocks), dtype=torch.float32, device=logits.device)
    else:
        assert out.shape == (T, num_blocks) and out.dtype == torch.float32

    cl_is_tensor = isinstance(compress_lens, torch.Tensor)
    if cl_is_tensor:
        cl = compress_lens.reshape(-1)
        assert cl.numel() in (1, T), "compress_lens must hold 1 or T entries"
        stride_cl = 0 if cl.numel() == 1 else cl.stride(0)
        cl_scalar = 0
    else:
        cl = out  # unused pointer argument
        stride_cl = 0
        cl_scalar = int(compress_lens)

    config = _get_config("_candidate_block_scores_kernel")
    blocks_per_tile = config.pop("BLOCKS_PER_TILE")
    grid = (T, triton.cdiv(num_blocks, blocks_per_tile))
    _candidate_block_scores_kernel[grid](
        logits,
        cl,
        out,
        cl_scalar,
        L,
        num_blocks,
        logits.stride(0),
        logits.stride(1),
        stride_cl,
        out.stride(0),
        BLOCK_SIZE=block_size,
        PAD_BLOCK_SIZE=triton.next_power_of_2(block_size),
        BLOCKS_PER_TILE=blocks_per_tile,
        CL_IS_TENSOR=cl_is_tensor,
        **config,
    )
    return out


def candidate_mask_from_blocks(
    block_idx: torch.Tensor,
    block_scores: torch.Tensor,
    width: int,
    block_size: int = 8,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand selected block ids into a bool mask over positions.

    A pick whose block score is exactly -inf is dropped, so a row with fewer
    reachable blocks than k does not keep its filler picks.

    Args:
        block_idx: [T, k] int32/int64 block ids from the top-k.
        block_scores: [T, num_blocks] fp32 scores the ids index into.
        width: number of positions in the output row (L).
        block_size: positions per candidate block.
        out: optional [T, width] bool destination; it is zeroed first.

    Returns:
        [T, width] bool mask.

    Config (attention/candidate_blocks/DEFAULT.json):
        TOPK_TILE - block ids scattered per program; num_warps / num_stages.
    """
    assert block_idx.ndim == 2 and block_scores.ndim == 2
    assert block_idx.shape[0] == block_scores.shape[0]
    assert block_scores.dtype == torch.float32
    T, k = block_idx.shape
    num_blocks = block_scores.shape[1]
    _LOGGER.info(
        f"CANDIDATE_MASK_FROM_BLOCKS: T={T} k={k} width={width} bs={block_size}"
    )

    if out is None:
        out = torch.zeros((T, width), dtype=torch.bool, device=block_idx.device)
    else:
        assert out.shape == (T, width) and out.dtype == torch.bool
        out.zero_()

    if k == 0 or T == 0:
        return out

    config = _get_config("_candidate_mask_scatter_kernel")
    topk_tile = config.pop("TOPK_TILE")
    grid = (T, triton.cdiv(k, topk_tile))
    _candidate_mask_scatter_kernel[grid](
        block_idx,
        block_scores,
        out,
        k,
        num_blocks,
        width,
        block_idx.stride(0),
        block_scores.stride(0),
        out.stride(0),
        BLOCK_SIZE=block_size,
        PAD_BLOCK_SIZE=triton.next_power_of_2(block_size),
        TOPK_TILE=topk_tile,
        **config,
    )
    return out


def apply_candidate_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """In-place ``logits[~mask] = -inf``.

    Fused because the torch spelling materializes ``~mask`` into a temporary and
    then rewrites every logit; this kernel reads the mask once and writes only
    the lanes it clears.

    Args:
        logits: [T, width] fp32, modified in place.
        mask: [T, width] bool candidate mask.

    Returns:
        ``logits``.

    Config (attention/candidate_blocks/DEFAULT.json):
        TILE_W - positions per program; num_warps / num_stages.
    """
    assert logits.shape == mask.shape and logits.ndim == 2
    assert logits.dtype == torch.float32 and mask.dtype == torch.bool
    T, width = logits.shape
    _LOGGER.info(f"APPLY_CANDIDATE_MASK: logits={tuple(logits.shape)}")

    config = _get_config("_apply_candidate_mask_kernel")
    tile_w = config.pop("TILE_W")
    grid = (T, triton.cdiv(width, tile_w))
    _apply_candidate_mask_kernel[grid](
        logits,
        mask,
        width,
        logits.stride(0),
        mask.stride(0),
        TILE_W=tile_w,
        **config,
    )
    return logits


def _topk_block_ids(block_scores: torch.Tensor, k: int) -> torch.Tensor:
    """Per-row top-k over block scores, returning [T, k] block ids."""
    T, num_blocks = block_scores.shape
    try:
        from aiter.ops.topk import top_k_per_row_prefill
    except ImportError:  # pure-Triton environment, no HIP extension
        from aiter.ops.triton.topk import topk as triton_topk

        return triton_topk(block_scores, k)[1]

    starts, ends = _row_bounds(T, num_blocks, block_scores.device.index)
    idx = torch.empty((T, k), dtype=torch.int32, device=block_scores.device)
    top_k_per_row_prefill(
        block_scores,
        starts,
        ends,
        idx,
        None,
        T,
        block_scores.stride(0),
        block_scores.stride(1),
        k,
        False,
    )
    return idx


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens,
    block_size: int = 8,
    topk_blocks: int = 2048,
) -> torch.Tensor:
    """Keep the ``topk_blocks`` highest-scoring blocks per query, as a position mask.

    Chains :func:`candidate_block_scores` -> per-row top-k ->
    :func:`candidate_mask_from_blocks`.

    Top-k choice: ``aiter.ops.topk.top_k_per_row_prefill`` with full-row
    rowStarts/rowEnds serves decode and prefill alike -- it takes the explicit
    per-row range this op can supply (the decode entry point instead wants
    raw-KV seqLens/next_n) and already routes small T to its one-block path --
    with ``aiter.ops.triton.topk`` as the fallback when the HIP extension is
    unavailable.

    Args:
        logits: [T, L] fp32, unreachable positions already -inf.
        compress_lens: int, or int32 tensor of shape [T] / [T, 1] / scalar.
        block_size: positions per candidate block.
        topk_blocks: blocks kept per query; clamped to num_blocks.

    Returns:
        [T, L] bool candidate mask.
    """
    _LOGGER.info(
        f"SELECT_CANDIDATE_BLOCKS: logits={tuple(logits.shape)} "
        f"block_size={block_size} topk_blocks={topk_blocks}"
    )
    width = logits.shape[-1]
    scores = candidate_block_scores(logits, compress_lens, block_size)
    k = min(topk_blocks, scores.shape[-1])
    idx = _topk_block_ids(scores, k)
    return candidate_mask_from_blocks(idx, scores, width, block_size)
