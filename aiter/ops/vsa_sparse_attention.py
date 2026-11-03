# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.torch_guard import torch_compile_guard


def _validate_vsa_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_lut: Tensor,
    block_counts: Tensor,
) -> None:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, Tensor) or not tensor.is_cuda:
            raise RuntimeError(f"{name} must be a GPU tensor")
        if tensor.dim() != 4:
            raise RuntimeError(f"{name} must have shape [B, H, S, D]")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous BHSD")
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(f"{name} must have dtype float16 or bfloat16")

    if q.device != k.device or q.device != v.device:
        raise RuntimeError("q, k, and v must be on the same GPU")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise RuntimeError("q, k, and v must have the same dtype")
    if q.size(0) <= 0 or q.size(1) <= 0 or k.size(1) <= 0:
        raise RuntimeError("batch size and head counts must be positive")
    if q.size(0) != k.size(0) or q.size(0) != v.size(0):
        raise RuntimeError("q, k, and v must have the same batch size")
    if k.shape != v.shape:
        raise RuntimeError("k and v must have the same shape")
    if q.size(1) % k.size(1) != 0:
        raise RuntimeError(
            "the number of query heads must be divisible by the number of KV heads"
        )
    if q.size(3) != 128 or k.size(3) != 128:
        raise RuntimeError(
            "VSA sparse attention currently supports head dimension 128 only"
        )
    if q.size(2) <= 0 or k.size(2) <= 128:
        raise RuntimeError(
            "query length must be positive and key length must exceed 128"
        )

    for name, tensor in (("block_lut", block_lut), ("block_counts", block_counts)):
        if not isinstance(tensor, Tensor) or not tensor.is_cuda:
            raise RuntimeError(f"{name} must be a GPU tensor")
        if tensor.device != q.device:
            raise RuntimeError(f"{name} must be on the same GPU as q")
        if tensor.dtype != torch.int32:
            raise RuntimeError(f"{name} must have dtype int32")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")

    batch, query_heads, seqlen_q, _ = q.shape
    query_blocks = (seqlen_q + 127) // 128
    kv_blocks = (k.size(2) + 127) // 128
    if block_lut.shape != (batch, query_heads, query_blocks, kv_blocks):
        raise RuntimeError(
            "block_lut must have shape [B, Hq, ceil(Sq/128), ceil(Sk/128)]"
        )
    if block_counts.shape != (batch, query_heads, query_blocks):
        raise RuntimeError("block_counts must have shape [B, Hq, ceil(Sq/128)]")

    min_count, max_count = torch.aminmax(block_counts)
    if min_count.item() < 1:
        raise RuntimeError("every query block must select at least one KV block")
    if max_count.item() >= kv_blocks:
        raise RuntimeError(
            "block_counts must be smaller than the LUT row capacity; "
            "the final slot is reserved for CK's lookahead"
        )


def _vsa_sparse_attention_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_lut: Tensor,
    block_counts: Tensor,
) -> Tensor:
    del k, v, block_lut, block_counts
    return torch.empty_like(q)


@compile_ops(
    "module_vsa_sparse_attention",
    fc_name="vsa_sparse_attention_fwd",
    develop=True,
)
def _vsa_sparse_attention_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_lut: Tensor,
    block_counts: Tensor,
    out: Tensor,
) -> None: ...


@torch_compile_guard(gen_fake=_vsa_sparse_attention_fake)
def vsa_sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_lut: Tensor,
    block_counts: Tensor,
) -> Tensor:
    """Run VSA block-sparse attention on contiguous BHSD tensors.

    ``block_lut`` stores one absolute K-block index followed by delta-encoded
    indices for each 128-token Q block. ``block_counts`` gives the number of
    active entries in each LUT row. The final LUT slot is reserved as a
    lookahead sentinel by the current CK pipeline.
    """
    _validate_vsa_inputs(q, k, v, block_lut, block_counts)
    out = torch.empty_like(q)
    _vsa_sparse_attention_fwd(q, k, v, block_lut, block_counts, out)
    return out
