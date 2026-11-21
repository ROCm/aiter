# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.pa_mqa_logits_shared_pages import (
    _pa_mqa_logits_shared_pages_kernel,
    _pa_mqa_logits_shared_pages_rows_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.tuned_config_utils import get_tuned_kernel_config

_LOGGER = AiterTritonLogger()

_NO_CONFIG = triton.Config({})
SUPPORTED_HEADS = (4, 8, 16)
SUPPORTED_HEAD_DIMS = (128, 256)


@functools.cache
def _launch_config() -> triton.Config | None:
    config = get_tuned_kernel_config(
        "attention",
        "PA_MQA_LOGITS_SHARED_PAGES",
        "_pa_mqa_logits_shared_pages_kernel",
        _NO_CONFIG,
        warn_on_miss=False,
    )
    return None if config is _NO_CONFIG else config


def has_launch_config() -> bool:
    """Whether this device has a published launch config."""
    return _launch_config() is not None


def pa_mqa_logits_shared_pages(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    out_logits: torch.Tensor,
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """FP8 paged MQA logits for few heads, with keys shared across query rows.

    Computes, for each query row, the DSA indexer score of every visible key:
    ``scale[k] * sum_h(weights[row, h] * relu(q[row, h] . key[k]))``.

    Rows next to each other that map their keys to the same pages, such as the
    rows of one prefill chunk, share each key tile. A second kernel rescores
    per row the tiles where rows of a block map different pages.

    Args:
        q_fp8: ``[B, next_n, H, D]`` queries, ``float8_e4m3fn`` or, on gfx942,
            ``float8_e4m3fnuz``, with ``H`` in 4, 8 or 16 and ``D`` in 128 or
            256. The last dimension must be contiguous. On gfx942,
            ``float8_e4m3fn`` uses the native FP8 MFMA with a bit-exact
            rescale, so NaN inputs are not kept.
        kv_cache: Contiguous uint8 ``[num_blocks, page_size, 1, D + 4]``. Each
            block holds ``page_size * D`` FP8 keys of the dtype of ``q_fp8``,
            then ``page_size`` FP32 key scales.
        weights: FP32 head weights, ``[B * next_n, H]``.
        out_logits: FP32 ``[B * next_n, >= max_model_len]``, written in place.
            Columns outside a row's causal bound are not written.
        context_lens: Contiguous int32 ``[B]`` or ``(B, next_n)``. Only the
            last column is read: query row ``j`` sees ``L - next_n + 1 + j``
            keys of a request of length ``L``.
        kv_indices: int32 block table, ``[B, max_blocks]``.
        max_model_len: Logical row width.

    Returns:
        ``out_logits``.
    """
    batch, next_n, heads, head_dim = q_fp8.shape
    is_gfx942 = arch_info.get_arch() == "gfx942"
    assert q_fp8.dtype == torch.float8_e4m3fn or (
        is_gfx942 and q_fp8.dtype == torch.float8_e4m3fnuz
    )
    assert q_fp8.stride(-1) == 1
    assert heads in SUPPORTED_HEADS and head_dim in SUPPORTED_HEAD_DIMS
    assert kv_cache.dtype == torch.uint8 and kv_cache.is_contiguous()
    assert kv_cache.shape[2:] == (1, head_dim + 4)
    assert weights.shape == (batch * next_n, heads)
    assert out_logits.dtype == torch.float32 and out_logits.stride(-1) == 1
    assert out_logits.shape[0] == batch * next_n
    assert out_logits.shape[1] >= max_model_len
    assert context_lens.dtype == torch.int32 and context_lens.is_contiguous()
    length_stride = context_lens.shape[1] if context_lens.dim() == 2 else 1
    assert context_lens.shape[0] == batch and length_stride in (1, next_n)
    _LOGGER.info(
        "PA_MQA_LOGITS_SHARED_PAGES: q=%s kv_cache=%s max_model_len=%d",
        tuple(q_fp8.shape),
        tuple(kv_cache.shape),
        max_model_len,
    )

    rows = batch * next_n
    if rows == 0 or max_model_len == 0:
        return out_logits

    config = _launch_config()
    if config is None:
        raise RuntimeError(
            "pa_mqa_logits_shared_pages has no launch config for "
            f"{arch_info.get_arch()}; add configs/<arch>/triton/attention/"
            "pa_mqa_logits_shared_pages/DEFAULT.json"
        )
    block_r = config.kwargs["BLOCK_M"] // heads
    tile = config.kwargs["TILE"]
    tiles_per_program = config.kwargs["TILES_PER_PROGRAM"]
    tiles = triton.cdiv(max_model_len, tile)
    grid = (triton.cdiv(rows, block_r), triton.cdiv(tiles, tiles_per_program))
    args = (
        q_fp8.view(torch.uint8),
        kv_cache,
        weights,
        context_lens,
        kv_indices,
        out_logits,
        rows,
        *q_fp8.stride()[:3],
        *weights.stride(),
        kv_cache.stride(0),
        *kv_indices.stride(),
        length_stride,
        out_logits.stride(0),
        max_model_len,
    )
    meta = {
        "NEXT_N": next_n,
        "HEADS": heads,
        "HEAD_DIM": head_dim,
        "PAGE_SIZE": kv_cache.shape[1],
        "BLOCK_R": block_r,
        "TILE": tile,
        "TILES_PER_PROGRAM": tiles_per_program,
        "READ_FNUZ": q_fp8.dtype == torch.float8_e4m3fnuz,
        "FN_AS_FNUZ": is_gfx942 and q_fp8.dtype == torch.float8_e4m3fn,
    }
    _pa_mqa_logits_shared_pages_kernel[grid](
        *args, **meta, num_warps=config.num_warps, num_stages=config.num_stages
    )
    _pa_mqa_logits_shared_pages_rows_kernel[grid](
        *args, **meta, DOT_M=max(heads, 16), num_warps=4
    )
    return out_logits
