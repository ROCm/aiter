# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.block_lut import (
    block_attn_mask_to_lut_kernel,
)
from aiter.ops.triton._triton_kernels.attention.lse_merge import _lse_merge_kernel
from aiter.ops.triton.utils.tuned_config_utils import get_tuned_kernel_config

# An arch without a measured entry gets a tile small enough to launch anywhere. On MI355X a sweep
# of BLOCK_M 8-128 x num_warps 1-8 over the kv_splits shapes of mha_v4 found 16 x 4 within 2% of
# each shape's best on average, and this fallback 8% behind it.
_LSE_MERGE_FALLBACK = triton.Config({"BLOCK_M": 32}, num_warps=4)


@functools.cache
def _lse_merge_config() -> triton.Config:
    """Resolved once per process: an arch without an entry warns on every lookup."""
    return get_tuned_kernel_config(
        "attention", "LSE_MERGE", "_lse_merge_kernel", _LSE_MERGE_FALLBACK
    )


def block_attn_mask_to_ragged_lut(
    block_attn_mask: torch.Tensor,
    num_heads: int | None = None,
    return_none_if_dense: bool = False,
    BLOCK_KB: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Convert a dense block attention mask to a ragged look-up table of KV block
    indices per (batch, head, q_block). Used for block-sparse attention with no
    per-iteration branching in the kernel.

    block_attn_mask: Either (batch, num_q_blocks, num_kv_blocks) boolean for
        same mask for all heads, or (batch, num_heads, num_q_blocks, num_kv_blocks)
        for per-head masks. True = may attend, False = must not attend.
    num_heads: Required when block_attn_mask is 3D (number of Q heads). Ignored when 4D.
    return_none_if_dense: If True and the mask is all True (dense), return None so the
        caller can pass block_lut=None to fav3_sage_wrapper_func and use the dense path.
        Avoids building a very large LUT that can trigger munmap_chunk on MI300X/ROCm.
    Returns:
        kv_block_indices: 1D int32, concatenation of all KV block index lists.
        lut_start: 1D int32, length batch * num_heads * num_q_blocks. Index
            idx = batch_idx * (num_heads * num_q_blocks) + head_idx * num_q_blocks + q_block_idx.
        lut_count: 1D int32, same length as lut_start.
        When return_none_if_dense is True and the mask is all True, returns None instead.
    """
    device = block_attn_mask.device

    # 3D -> 4D: expand and fall through to 4D path
    if block_attn_mask.dim() == 3:
        if num_heads is None:
            raise ValueError("num_heads must be provided when block_attn_mask is 3D")
        batch, num_q_blocks, num_kv_blocks = block_attn_mask.shape
        if return_none_if_dense and block_attn_mask.all():
            return None
        block_attn_mask = block_attn_mask.unsqueeze(1).expand(
            batch, num_heads, num_q_blocks, num_kv_blocks
        )

    # 4D: (batch, num_heads, num_q_blocks, num_kv_blocks) — GPU vectorized path
    batch, num_heads, num_q_blocks, num_kv_blocks = block_attn_mask.shape
    if return_none_if_dense and block_attn_mask.all():
        return None

    counts = block_attn_mask.sum(dim=-1, dtype=torch.int32)
    lut_count = counts.reshape(-1)
    lut_start = torch.cumsum(lut_count, dim=0, dtype=torch.int32) - lut_count

    # NOTE: Overallocating the LUT is a waste of memory, but the
    # alternative lut_count.sum(), will cause graph break with torch compile.
    # TODO(sparse refresh): entries past the used range stay uninitialized, and an empty row's
    # lut_start points at them. Kernels that speculatively read LUT[lut_start] before testing
    # lut_count must clamp the derived offsets, as the FP8 prologue does, or this must be zeroed.
    max_count = batch * num_heads * num_q_blocks * num_kv_blocks
    kv_block_indices = torch.empty(max_count, dtype=torch.int32, device=device)
    block_attn_mask_to_lut_kernel(
        block_attn_mask,
        lut_start,
        lut_count,
        kv_block_indices,
        BLOCK_KB=BLOCK_KB,
    )

    return kv_block_indices, lut_start, lut_count


def lse_merge(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor | None = None,
) -> None:
    """
    Merge attention partials over disjoint key ranges into `out` (and `lse`) by log-sum-exp.

    partial_out: BSHD [batch * splits, Sq, H, D]; partial batch b * splits + i holds key
        range i of batch b. Any strides with a contiguous last dimension.
    partial_lse: FP32 [batch * splits, H, Sq], natural-log log-sum-exp of each range,
        with a contiguous last dimension.
    out: BSHD [batch, Sq, H, D], written as weighted by exp(partial_lse - lse).
    lse: optional FP32 [batch, H, Sq] receiving the merged log-sum-exp.
    """
    batch, sequence, heads, head_dim = out.shape
    if partial_out.shape[0] % batch or partial_out.shape[1:] != out.shape[1:]:
        raise ValueError(
            f"partial_out {tuple(partial_out.shape)} does not fold whole key ranges of "
            f"out {tuple(out.shape)} into its batch"
        )
    splits = partial_out.shape[0] // batch
    if partial_lse.shape != (partial_out.shape[0], heads, sequence):
        raise ValueError(
            f"partial_lse must be [batch * splits, heads, Sq] = "
            f"{(partial_out.shape[0], heads, sequence)}, got {tuple(partial_lse.shape)}"
        )
    if lse is not None and lse.shape != (batch, heads, sequence):
        raise ValueError(
            f"lse must be [batch, heads, Sq] = {(batch, heads, sequence)}, "
            f"got {tuple(lse.shape)}"
        )
    for name, tensor in (
        ("partial_out", partial_out),
        ("partial_lse", partial_lse),
        ("out", out),
        ("lse", lse),
    ):
        if tensor is not None and tensor.stride(-1) != 1:
            raise ValueError(f"{name} must have a contiguous last dimension")
    if head_dim & (head_dim - 1):
        raise ValueError(f"head dimension must be a power of two, got {head_dim}")

    config = _lse_merge_config()
    block_m = config.kwargs["BLOCK_M"]
    lse_strides = (lse.stride(0), lse.stride(1)) if lse is not None else (0, 0)
    _lse_merge_kernel[(triton.cdiv(sequence, block_m), heads, batch)](
        partial_out,
        partial_lse,
        out,
        lse if lse is not None else partial_lse,
        sequence,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        *lse_strides,
        SPLITS=splits,
        BLOCK_M=block_m,
        HEAD_DIM=head_dim,
        WRITE_LSE=lse is not None,
        num_warps=config.num_warps,
    )
