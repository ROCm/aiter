# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""OPUS-based paged-attention decode for gfx950.

Follows the sp3 kernel ``PA_A16W16_*_1TG_4W_16mx1_64nx4``: one thread group of
4 waves per (sequence, kv-head), 16 query rows, waves split along the KV axis
for Q*K and along the head axis for P*V.

The user-facing entry is :func:`pa_decode_opus`; it forwards to the
JIT-compiled HIP kernel via :func:`pa_decode_opus_fwd`.

The kernel currently only compiles a single configuration:

* Head dim ``128``, page size ``16``, GQA ratio ``<= 16``.
* dtype ``bf16`` for Q/K/V/O (A16W16 -- no KV quantization).
* K cache packed as ``[num_blocks, num_kv_heads, 128/8, 16, 8]`` and V cache as
  ``[num_blocks, num_kv_heads, 128, 16]`` (the standard vLLM layout for bf16,
  where the pack factor ``x`` is ``16 bytes / itemsize == 8``).

See ``aiter/csrc/include/pa_decode_opus.h`` for the C++ API.
"""

import torch

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime
from ..jit.utils.torch_guard import torch_compile_guard

MD_NAME = "module_pa_decode_opus"

_HEAD_DIM = 128
_PAGE_SIZE = 16
_MAX_GQA = 16


@compile_ops("module_pa_decode_opus", develop=True)
def pa_decode_opus_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> None: ...


def _pa_decode_opus_fake(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return out if out is not None else torch.empty_like(q)


@torch_compile_guard(mutates_args=["out"], gen_fake=_pa_decode_opus_fake)
def pa_decode_opus(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged-attention decode over a block table, backed by the OPUS gfx950 kernel.

    The trailing ``out`` keyword is an aiter-only convenience for callers that
    want to reuse a pre-allocated output buffer; pass ``None`` (the default) to
    have one allocated for you.

    Args:
      q:            ``[batch, num_heads, 128]`` bf16 query. Decode only, so there
                    is no token dim: one query token per batch row.
      k_cache:      ``[num_blocks, num_kv_heads, D/K_PACK, PAGE, K_PACK]`` bf16 key cache.
      v_cache:      ``[num_blocks, num_kv_heads, 128, 16]`` bf16 value cache.
      block_tables: ``[batch, max_blocks_per_batch_row]`` int32 page indices.
      context_lens: ``[batch]`` int32 KV length per batch row.
      softmax_scale: float scalar applied to the QK^T scores.
      out:          Optional ``[batch, num_heads, 128]`` output buffer.

    Returns:
      ``out`` (``[batch, num_heads, 128]`` bf16).
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"pa_decode_opus requires gfx950, got {gfx}")

    if q.dtype != torch.bfloat16:
        raise RuntimeError(f"pa_decode_opus expects bf16 q, got {q.dtype}")
    if k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
        raise RuntimeError(
            f"KV cache dtype mismatch: k_cache={k_cache.dtype}, "
            f"v_cache={v_cache.dtype}, q={q.dtype}"
        )
    if q.size(-1) != _HEAD_DIM:
        raise RuntimeError(
            f"pa_decode_opus only supports head_dim={_HEAD_DIM}, got {q.size(-1)}"
        )
    if v_cache.size(-1) != _PAGE_SIZE:
        raise RuntimeError(
            f"pa_decode_opus only supports page size={_PAGE_SIZE}, got {v_cache.size(-1)}"
        )

    num_heads, num_kv_heads = q.size(1), k_cache.size(1)
    if num_heads % num_kv_heads != 0:
        raise RuntimeError(
            f"num_heads={num_heads} not divisible by num_kv_heads={num_kv_heads}"
        )
    if num_heads // num_kv_heads > _MAX_GQA:
        raise RuntimeError(
            f"GQA ratio must be <= {_MAX_GQA}, got {num_heads // num_kv_heads}"
        )

    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={tuple(q.shape)} dtype={q.dtype}"
        )

    pa_decode_opus_fwd(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        out,
        float(softmax_scale),
    )
    return out


__all__ = [
    "pa_decode_opus",
    "pa_decode_opus_fwd",
]
