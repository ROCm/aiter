# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.step5_paged_mqa_logits import (
    _step5_paged_mqa_logits_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.tuned_config_utils import get_tuned_kernel_config

_LOGGER = AiterTritonLogger()

_HEAD_DIM = 256
_NO_CONFIG = triton.Config({})


@functools.cache
def _launch_config() -> triton.Config:
    config = get_tuned_kernel_config(
        "attention",
        "STEP5_PAGED_MQA_LOGITS",
        "_step5_paged_mqa_logits_kernel",
        _NO_CONFIG,
    )
    if config is _NO_CONFIG:
        raise RuntimeError(
            "step5_paged_mqa_logits has no launch config for "
            f"{arch_info.get_arch()}; add configs/<arch>/triton/attention/"
            "step5_paged_mqa_logits/DEFAULT.json"
        )
    return config


def step5_paged_mqa_logits(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Compute the Step-5 CSA indexer logits over packed FP8 summary pages.

    The contract is the same as ``paged_mqa_logits_sm90`` in the vLLM Step-5
    model.

    Args:
        q: FP8 e4m3fn queries, ``[batch, next_n, heads, 256]``. ``heads`` is 4,
            8 or 16. The last dimension must be contiguous.
        kv_fused: Contiguous uint8 pages, ``[pages, page_size, 1, 260]``. Each
            page holds ``page_size * 256`` FP8 e4m3fn keys, then ``page_size``
            FP32 key scales.
        weights: FP32 head weights, ``[batch * next_n, heads]``.
        block_tables: int32 physical page of each logical page,
            ``[batch, pages_per_request]``.
        seq_lens: int32 number of visible keys of each request, ``[batch]``.
            All ``next_n`` rows of a request see the same keys.
        max_seq_len: Number of output columns.

    Returns:
        FP32 logits, ``[batch * next_n, max_seq_len]``. Entry ``[row, k]`` is
        ``scale[k] * sum_h(weights[row, h] * relu(q[row, h] . key[k]))`` for
        ``k < seq_lens[batch]``, and ``-inf`` for the other columns.

    Rows that map their visible keys to the same pages share one key tile.
    Rows that map to different pages use a slower path that loads the keys of
    each row. On gfx942 the kernel reads the e4m3fn bits as e4m3fnuz for the
    native FP8 MFMA, so NaN inputs are not kept.
    """
    batch, next_n, heads, head_dim = q.shape
    assert q.dtype == torch.float8_e4m3fn and head_dim == _HEAD_DIM
    assert heads in (4, 8, 16) and q.stride(-1) == 1
    assert kv_fused.dtype == torch.uint8 and kv_fused.is_contiguous()
    assert kv_fused.shape[2:] == (1, _HEAD_DIM + 4)
    assert weights.shape == (batch * next_n, heads)
    _LOGGER.info(
        "STEP5_PAGED_MQA_LOGITS: q=%s kv_fused=%s max_seq_len=%d",
        tuple(q.shape),
        tuple(kv_fused.shape),
        max_seq_len,
    )

    rows = batch * next_n
    logits = torch.empty((rows, max_seq_len), dtype=torch.float32, device=q.device)
    if rows == 0 or max_seq_len == 0:
        return logits

    config = _launch_config()
    block_r = config.kwargs["BLOCK_M"] // heads
    tile = config.kwargs["TILE"]
    tiles = triton.cdiv(max_seq_len, tile)
    tiles_per_program = min(config.kwargs["TILES_PER_PROGRAM"], tiles)
    grid = (triton.cdiv(rows, block_r), triton.cdiv(tiles, tiles_per_program))
    _step5_paged_mqa_logits_kernel[grid](
        q.view(torch.uint8),
        kv_fused,
        weights,
        seq_lens,
        block_tables,
        logits,
        rows,
        *q.stride()[:3],
        *weights.stride(),
        kv_fused.stride(0),
        *block_tables.stride(),
        seq_lens.stride(0),
        max_seq_len,
        NEXT_N=next_n,
        HEADS=heads,
        HEAD_DIM=_HEAD_DIM,
        PAGE_SIZE=kv_fused.shape[1],
        BLOCK_R=block_r,
        TILE=tile,
        TILES_PER_PROGRAM=tiles_per_program,
        FNUZ=arch_info.get_arch() == "gfx942",
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return logits
