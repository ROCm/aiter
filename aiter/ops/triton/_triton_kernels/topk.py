# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

# The kernel in this file is adapted from FlagGems' topk:
# https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/topk.py

#  Top-K on GPU:  1-stage (tiny rows) + 2-stage (large rows) Triton kernels,
import math
import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import _log2, zeros_like


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
    FILL_VALUE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_ptr = X + pid * stride_xm
    offs = tl.arange(0, BLOCK)
    mask = offs < M
    # FILL_VALUE = tl.constexpr(torch.finfo(torch.float32).min)
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
    FILL_VALUE: tl.constexpr,
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

    # FILL_VALUE = tl.constexpr(
    #    torch.finfo(torch.float32).min if DESCENDING else torch.finfo(torch.float32).max
    # )
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
                FILL_VALUE,
                # tl.constexpr(torch.finfo(torch.float32).min),
                x_val,
            )
        else:
            x_val = tl.where(
                cols == chunk_select_idx,
                FILL_VALUE,
                # tl.constexpr(torch.finfo(torch.float32).max),
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
    FILL_VALUE: tl.constexpr,
    MASK_INDEX_VAL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    chunk_x += cur_batch * N
    chunk_index += cur_batch * N
    y_ptr += cur_batch * k
    index_ptr += cur_batch * k

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # FILL_VALUE = tl.constexpr(
    #    torch.finfo(torch.float32).min if DESCENDING else torch.finfo(torch.float32).max
    # )
    # mask_index_val = (
    #    tl.constexpr(torch.iinfo(torch.int32).min)
    #    if DESCENDING
    #    else tl.constexpr(torch.iinfo(torch.int32).max)
    # )

    chunk_x_val = tl.load(chunk_x + cols, mask=mask, other=FILL_VALUE).to(tl.float32)
    chunk_index_val = tl.load(chunk_index + cols, mask=mask, other=MASK_INDEX_VAL).to(
        tl.int32
    )

    sorted_chunk_x, sorted_chunk_index = argsort(
        chunk_x_val, chunk_index_val, 0, descending=DESCENDING
    )
    tl.store(y_ptr + cols, sorted_chunk_x, mask=cols < k)
    tl.store(index_ptr + cols, sorted_chunk_index, mask=cols < k)
