# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse paged-prefill attention over two KV sources (prefix + extend) with
per-head sink bias — gfx1250 (gluon) only.

Exposes ``pa_prefill_sparse`` — grid ``(T, cdiv(H, BLOCK_H))``, one token
and BLOCK_H heads per CTA. Same grid as the decode kernel. No split-K:
prefill fills the GPU via the token dimension.
"""

import torch
import triton

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._gluon_kernels.gfx1250.attention.pa_prefill_sparse import (
    _pa_prefill_sparse as gluon_pa_prefill_sparse,
)

DEVICE_ARCH = arch_info.get_arch()

_LOGGER = AiterTritonLogger()


def pa_prefill_sparse(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse prefill attention over two KV sources with sink.

    Args:
        q:                 [T, H, D] BF16/FP16 — queries.
        unified_kv:        [total_pages, D] — prefix KV source (paged).
        kv_indices_prefix: [total_prefix] int32 — flat per-token slot lists
            into unified_kv. ``-1`` sentinels skipped.
        kv_indptr_prefix:  [T+1] int32 — true prefix sum.
        kv:                [total_tokens, D] — extend KV source (this fwd's
            input K, not yet in paged buffer).
        kv_indices_extend: [total_extend] int32 — flat per-token row idx lists
            into kv. ``-1`` sentinels skipped.
        kv_indptr_extend:  [T+1] int32 — true prefix sum.
        attn_sink:         [H] fp32 — per-head softmax-denom bias.
        softmax_scale:     float.

    Returns:
        [T, H, D] attention output, same dtype as q.
    """
    assert (
        DEVICE_ARCH == "gfx1250"
    ), f"pa_prefill_sparse requires gfx1250, got {DEVICE_ARCH}"
    if not q.is_cuda:
        raise RuntimeError("pa_prefill_sparse requires CUDA/HIP tensors")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"pa_prefill_sparse expects fp16/bf16 q, got {q.dtype}")
    if unified_kv.dtype != q.dtype:
        raise RuntimeError(
            f"unified_kv dtype mismatch: kv={unified_kv.dtype}, q={q.dtype}"
        )
    if kv.dtype != q.dtype:
        raise RuntimeError(f"kv dtype mismatch: kv={kv.dtype}, q={q.dtype}")

    T, H, D = q.shape
    _LOGGER.info(
        f"PA_PREFILL_SPARSE T={T} H={H} D={D} "
        f"prefix_indices={kv_indices_prefix.shape[0]} "
        f"extend_indices={kv_indices_extend.shape[0]}"
    )

    out = torch.empty_like(q)
    assert kv_indices_prefix.dtype == torch.int32 and kv_indices_prefix.is_contiguous()
    assert kv_indptr_prefix.dtype == torch.int32 and kv_indptr_prefix.is_contiguous()
    assert kv_indices_extend.dtype == torch.int32 and kv_indices_extend.is_contiguous()
    assert kv_indptr_extend.dtype == torch.int32 and kv_indptr_extend.is_contiguous()

    block_h = max(triton.next_power_of_2(min(H, 16)), 16)
    block_d = triton.next_power_of_2(D)
    assert block_d == D
    block_k = 16

    total_prefix_pages = unified_kv.shape[0]
    total_extend_tokens = kv.shape[0]

    USE_EXP2 = True

    grid = (T, triton.cdiv(H, block_h))

    gluon_pa_prefill_sparse[grid](
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        total_prefix_pages,
        total_extend_tokens,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        unified_kv.stride(0),
        unified_kv.stride(1),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H,
        D,
        float(softmax_scale),
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        USE_EXP2=USE_EXP2,
        num_warps=1,
    )
    return out
