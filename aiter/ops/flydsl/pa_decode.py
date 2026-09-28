# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL BF16 paged-attention decode (``pa_decode_tile``) for gfx942/gfx950.

Public entry for the vendored kernel in ``kernels/pa/``. The kernel flattens
``query_length * query_group_size`` onto the MFMA M axis, so a decode row fills
the 16-row tile with its GQA group instead of occupying one row of a
prefill-shaped 128-row tile.

Layouts (``kVectorSize = 16 // element_size``; 8 for BF16)::

    query        [num_seqs * query_length, num_q_heads, head_dim]
    key_cache    [num_blocks, num_kv_heads, head_dim // kV, block_size, kV]
    value_cache  [num_blocks, num_kv_heads, block_size // kV, v_head_dim, kV]
    block_tables [num_seqs, max_blocks_per_seq]            int32
    seq_lens     [num_seqs]                                int32
    output       [num_seqs * query_length, num_q_heads, v_head_dim]

``seq_lens`` bounds each sequence; slots past it are masked, so a partially
filled last block only needs to be free of NaN/Inf bit patterns.
"""

from __future__ import annotations

from functools import lru_cache

import torch

__all__ = [
    "flydsl_pa_decode_supported",
    "flydsl_pa_decode",
    "flydsl_pa_decode_partials",
    "flydsl_pa_decode_recommended_splits",
]

_SUPPORTED_ARCHS = ("gfx942", "gfx950")
_SUPPORTED_BLOCK_SIZES = (16, 64)


@lru_cache(maxsize=8)
def _arch(device_index: int) -> str:
    try:
        return torch.cuda.get_device_properties(device_index).gcnArchName.split(":")[0]
    except Exception:  # noqa: BLE001 -- no device / driver mismatch
        return ""


def _unsupported_reason(
    *,
    arch: str,
    dtype: torch.dtype,
    head_dim: int,
    v_head_dim: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> str | None:
    """Return why this shape is unserved, or None. Mirrors the kernel's own
    asserts so a caller can gate without catching an AssertionError."""
    if arch not in _SUPPORTED_ARCHS:
        return f"BF16 paged decode needs {' or '.join(_SUPPORTED_ARCHS)}, got {arch or 'unknown'}"
    if dtype is not torch.bfloat16:
        return f"BF16 KV MFMA requires a bf16 query/cache, got {dtype}"
    if head_dim % 64 or v_head_dim % 64:
        return f"head_dim and v_head_dim must be multiples of 64, got {head_dim}/{v_head_dim}"
    if block_size not in _SUPPORTED_BLOCK_SIZES:
        return f"block_size must be one of {_SUPPORTED_BLOCK_SIZES}, got {block_size}"
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        return f"num_q_heads ({num_q_heads}) must be a positive multiple of num_kv_heads ({num_kv_heads})"
    return None


def flydsl_pa_decode_supported(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    *,
    v_head_dim: int | None = None,
    block_size: int | None = None,
) -> bool:
    """Whether ``flydsl_pa_decode`` serves this shape. Never raises."""
    try:
        head_dim = query.shape[-1]
        num_q_heads = query.shape[-2]
        num_kv_heads = key_cache.shape[1]
        if block_size is None:
            block_size = key_cache.shape[3]
        return (
            _unsupported_reason(
                arch=_arch(query.device.index),
                dtype=query.dtype,
                head_dim=head_dim,
                v_head_dim=head_dim if v_head_dim is None else v_head_dim,
                block_size=int(block_size),
                num_q_heads=int(num_q_heads),
                num_kv_heads=int(num_kv_heads),
            )
            is None
        )
    except Exception:  # noqa: BLE001 -- a malformed call is simply unsupported
        return False


def flydsl_pa_decode_recommended_splits(
    num_seqs: int, num_kv_heads: int, block_size: int
) -> int:
    """Split-KV count for this shape.

    ``num_partitions`` is a compile-time constant in the kernel, so a CUDA-graph
    caller must pin one value across every captured batch rather than letting it
    vary. Upstream clamps to ``[4, 8]``, which was tuned for batched serving; at
    concurrency 1 the grid is only ``num_seqs * num_kv_heads * splits`` CTAs, so
    a larger value can be worth forcing.
    """
    from aiter.ops.flydsl.kernels.pa.pa_support import (
        KV_COMPUTE_BLOCK,
        get_recommended_splits,
    )

    return int(
        get_recommended_splits(
            int(num_seqs),
            int(num_kv_heads),
            split_kv_blocks=max(1, KV_COMPUTE_BLOCK // int(block_size)),
        )
    )


def flydsl_pa_decode_partials(
    num_seqs: int,
    num_kv_heads: int,
    num_partitions: int,
    query_length: int,
    query_group_size: int,
    v_head_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate the ``(pmax, psum, pout)`` split-KV partials.

    The kernel refuses to allocate these mid-capture, so a CUDA-graph caller
    must build them once, outside the graph, at the captured maximum batch.
    """
    shape = (
        int(num_seqs),
        int(num_kv_heads),
        int(num_partitions),
        int(query_length) * int(query_group_size),
    )
    pmax = torch.empty(shape, dtype=torch.float32, device=device)
    psum = torch.empty(shape, dtype=torch.float32, device=device)
    pout = torch.empty((*shape, int(v_head_dim)), dtype=dtype, device=device)
    return pmax, psum, pout


def flydsl_pa_decode(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    *,
    num_partitions: int,
    pmax: torch.Tensor,
    psum: torch.Tensor,
    pout: torch.Tensor,
    stream=None,
) -> torch.Tensor:
    """Run the kernel into ``output`` and return it.

    ``num_partitions`` and the partial buffers are required rather than
    optional: every caller here runs under CUDA-graph capture at least some of
    the time, and the kernel raises if it has to allocate or pick a split count
    itself while capturing. Use ``flydsl_pa_decode_recommended_splits`` and
    ``flydsl_pa_decode_partials`` to build them ahead of capture.
    """
    from aiter.ops.flydsl.kernels.pa.pa_decode_tile import pa_decode_tile

    pa_decode_tile(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        context_lengths=seq_lens,
        key_scale=None,
        value_scale=None,
        softmax_scale=float(softmax_scale),
        stream=stream,
        num_partitions=int(num_partitions),
        pmax=pmax,
        psum=psum,
        pout=pout,
    )
    return output
