# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

# The kernel in this file is adapted from FlagGems' topk:
# https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/topk.py

#  Top-K on GPU:  1-stage (tiny rows) + 2-stage (large rows) Triton kernels,
from __future__ import annotations
from typing import Tuple
import math
import torch
import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import _log2, zeros_like
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


# 1-STAGE KERNEL (tiny rows)
@triton.jit
def _topk_kernel(
    X,
    OUT_V,
    OUT_I,
    stride_xm,
    stride_ovm,
    stride_oim,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_ptr = X + pid * stride_xm
    offs = tl.arange(0, BLOCK)
    mask = offs < M
    FILL_VALUE = tl.constexpr(torch.finfo(torch.float32).min)
    vals = tl.load(row_ptr + offs, mask=mask, other=FILL_VALUE).to(tl.float32)
    idxs = offs.to(tl.int64)

    out_v_ptr = OUT_V + pid * stride_ovm
    out_i_ptr = OUT_I + pid * stride_oim

    # unrolled exactly K iterations -- no break/continue needed
    for j in core.static_range(0, K):
        vmax = tl.max(vals, axis=0)
        eq = vals == vmax
        big = tl.where(
            eq, tl.zeros_like(idxs), tl.zeros_like(idxs) + BLOCK
        )  # BLOCK as int64
        arg = tl.min(idxs + big, axis=0)

        tl.store(out_v_ptr + j, vmax)
        tl.store(out_i_ptr + j, arg)

        vals = tl.where(idxs == arg, FILL_VALUE, vals)


def _pick_block(m: int, k: int) -> int:
    blk = max(128, k)
    while blk < m and blk < 1024:
        blk <<= 1
    return blk


def one_stage_topk(
    x: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, M = x.shape
    BLOCK = _pick_block(M, k)
    if M > BLOCK or BLOCK > 1024:
        raise ValueError("row length too large for this kernel (<=1024)")

    out_v = torch.empty((B, k), device=x.device, dtype=x.dtype)
    out_i = torch.empty((B, k), device=x.device, dtype=torch.int64)

    _topk_kernel[(B,)](
        x.contiguous(),
        out_v,
        out_i,
        x.stride(0),
        out_v.stride(0),
        out_i.stride(0),
        M=M,
        K=k,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=2,
    )
    return out_v, out_i


# 2-STAGE KERNEL (large rows)
@triton.jit
def topk_stage1_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    k,
    N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_chunk_idx = tl.program_id(1)
    chunk_num = tl.num_programs(1)

    y_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k
    index_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k

    chunk_offset = cur_chunk_idx * CHUNK_SIZE
    x_ptr += cur_batch * N + chunk_offset

    cols = tl.arange(0, CHUNK_SIZE)
    mask = (chunk_offset + cols) < N

    FILL_VALUE = tl.constexpr(
        torch.finfo(torch.float32).min if DESCENDING else torch.finfo(torch.float32).max
    )
    x_val = tl.load(x_ptr + cols, mask=mask, other=FILL_VALUE).to(tl.float32)
    for k_idx in range(k):
        if DESCENDING:
            chunk_select_val, chunk_select_idx = tl.max(
                x_val, axis=0, return_indices=True
            )
        else:
            chunk_select_val, chunk_select_idx = tl.min(
                x_val, axis=0, return_indices=True
            )

        tl.store(y_ptr + k_idx, chunk_select_val)
        tl.store(index_ptr + k_idx, chunk_select_idx + chunk_offset)

        if DESCENDING:
            x_val = tl.where(
                cols == chunk_select_idx,
                tl.constexpr(torch.finfo(torch.float32).min),
                x_val,
            )
        else:
            x_val = tl.where(
                cols == chunk_select_idx,
                tl.constexpr(torch.finfo(torch.float32).max),
                x_val,
            )


@triton.jit
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]

    y = core.reshape(x, shape)
    y_idx = core.reshape(ids, shape)

    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = core.arange(0, 2)[None, :, None]
    left = core.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(x.dtype)
    right = core.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(x.dtype)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)

    left_idx = core.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape).to(
        ids.dtype
    )
    right_idx = core.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape).to(
        ids.dtype
    )
    left_idx = core.reshape(left_idx, ids.shape)
    right_idx = core.reshape(right_idx, ids.shape)

    # actual compare-and-swap
    if core.constexpr(x.dtype.primitive_bitwidth) == 8:
        idtype = core.int8
    elif core.constexpr(x.dtype.primitive_bitwidth) == 16:
        idtype = core.int16
    elif core.constexpr(x.dtype.primitive_bitwidth) == 32:
        idtype = core.int32
    elif core.constexpr(x.dtype.primitive_bitwidth) == 64:
        idtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) ^ flip
    ret = ix ^ core.where(cond, ileft ^ iright, zeros_like(ix))

    if core.constexpr(ids.dtype.primitive_bitwidth) == 8:
        idx_dtype = core.int8
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 16:
        idx_dtype = core.int16
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 32:
        idx_dtype = core.int32
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 64:
        idx_dtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft_idx = left_idx.to(idx_dtype, bitcast=True)
    iright_idx = right_idx.to(idx_dtype, bitcast=True)
    ix_idx = ids.to(idx_dtype, bitcast=True)
    ret_idx = ix_idx ^ core.where(cond, ileft_idx ^ iright_idx, zeros_like(ix_idx))

    return ret.to(x.dtype, bitcast=True), ret_idx.to(ids.dtype, bitcast=True)


@triton.jit
def _bitonic_merge(
    x, ids, stage: core.constexpr, order: core.constexpr, n_dims: core.constexpr
):
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: core.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = core.reshape(
            core.broadcast_to(core.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x, ids, dim: tl.constexpr, descending: core.constexpr):
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = dim
    n_dims: core.constexpr = _log2(x.shape[_dim])
    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids


@triton.jit
def topk_stage2_kernel(
    y_ptr,
    index_ptr,
    chunk_x,
    chunk_index,
    k: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    chunk_x += cur_batch * N
    chunk_index += cur_batch * N
    y_ptr += cur_batch * k
    index_ptr += cur_batch * k

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    FILL_VALUE = tl.constexpr(
        torch.finfo(torch.float32).min if DESCENDING else torch.finfo(torch.float32).max
    )
    mask_index_val = (
        tl.constexpr(torch.iinfo(torch.int32).min)
        if DESCENDING
        else tl.constexpr(torch.iinfo(torch.int32).max)
    )

    chunk_x_val = tl.load(chunk_x + cols, mask=mask, other=FILL_VALUE).to(tl.float32)
    chunk_index_val = tl.load(chunk_index + cols, mask=mask, other=mask_index_val).to(
        tl.int32
    )

    sorted_chunk_x, sorted_chunk_index = argsort(
        chunk_x_val, chunk_index_val, 0, descending=DESCENDING
    )
    tl.store(y_ptr + cols, sorted_chunk_x, mask=cols < k)
    tl.store(index_ptr + cols, sorted_chunk_index, mask=cols < k)


def two_stage_topk(x, k, dim=-1, largest=True):
    descending = True
    if not largest:
        descending = False

    topk_elem_cnt = x.shape[dim]
    batch_size = math.prod(x.shape) // topk_elem_cnt

    if topk_elem_cnt < 1024:
        chunk_size = 256
    else:
        chunk_size = 1024

    if chunk_size < k:
        chunk_size = triton.next_power_of_2(k)

    chunk_num = triton.cdiv(topk_elem_cnt, chunk_size)

    stage1_out = torch.empty(batch_size * chunk_num * k, device=x.device, dtype=x.dtype)
    stage1_out_idx = torch.empty(
        batch_size * chunk_num * k, device=x.device, dtype=torch.int64
    )

    out_shape = x.shape[:-1] + (k,)
    stage2_out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    stage2_out_idx = torch.empty(out_shape, device=x.device, dtype=torch.int64)

    topk_stage1_kernel[
        batch_size,
        chunk_num,
    ](
        stage1_out,  # pointer to the output
        stage1_out_idx,  # pointer to the output
        x,  # pointer to the input
        k,
        topk_elem_cnt,
        chunk_size,
        descending,
    )
    stage2_elem_cnt = chunk_num * k
    BLOCK_SIZE = triton.next_power_of_2(stage2_elem_cnt)

    topk_stage2_kernel[batch_size,](
        stage2_out,
        stage2_out_idx,
        stage1_out,
        stage1_out_idx,
        k,
        stage2_elem_cnt,
        BLOCK_SIZE,
        descending,
    )

    return (stage2_out, stage2_out_idx)


# For dispatcher
MAX_TINY_ROW = 1024

"""
Triton Top-K operator
=========================================

Selects the "k" largest elements (and their indices) along the "last"
dimension of a 2-D input tensor.  A fast path and a hierarchical path are
chosen automatically based on the row length "M".

Algorithm selection
-------------------
- 1-stage kernel - used when M <= 1024 ("tiny" rows).
  Each row is processed by one Triton launch.
- 2-stage kernel - used when M > 1024 ("large" rows).
  The row is first tiled, each tile computes a local Top-K, and the partial
  results are merged in a second stage.

Interface & constraints
-----------------------
1. Only the last dimension can be reduced.
2. Input must be a 2-D tensor of shape (B, M).
3. Exactly k largest elements are returned.
4. Returned values are **sorted in descending order.

Returns
-------
(values, indices) - both tensors have shape (B, k) and reside on the
same device as the input.

"""


def topk(
    x: torch.Tensor,
    k: int,
    *,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    tiny_row_thresh: int = MAX_TINY_ROW,
):
    _LOGGER.info(f"TOPK: x={tuple(x.shape)}, k={k}, largest={largest}, sorted={sorted}")
    if dim < 0:
        dim += x.ndim
    if dim != x.ndim - 1:
        raise ValueError("only last-dim Top-K is implemented")
    if x.ndim != 2:
        raise ValueError("input tensor must be 2-D (batch, M)")
    if not largest:
        raise ValueError("only largest=True supported")
    if not sorted:
        raise ValueError("sorted=False not supported")

    if not x.is_contiguous():
        x = x.contiguous()

    row_len = x.shape[-1]
    if row_len <= tiny_row_thresh:
        # if (row_len <= tiny_row_thresh) and (k <= 8):
        return one_stage_topk(x.view(-1, row_len), k)
    else:
        return two_stage_topk(x, k, dim=dim, largest=True)
