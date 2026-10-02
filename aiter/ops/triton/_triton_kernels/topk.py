# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

# The kernel in this file is adapted from FlagGems' topk:
# https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/topk.py

#  Top-K on GPU:  1-stage (tiny rows) + 2-stage (large rows) Triton kernels,
import triton
import triton.language as tl
from triton.language import core
from triton.language.standard import _log2

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_topk_kernel_repr = make_kernel_repr(
    "_topk_kernel",
    [
        "M",
        "K",
        "BLOCK",
    ],
)

_topk_stage1_kernel_repr = make_kernel_repr(
    "topk_stage1_kernel",
    [
        "N",
        "CHUNK_SIZE",
        "DESCENDING",
    ],
)

_topk_stage2_kernel_repr = make_kernel_repr(
    "topk_stage2_kernel",
    [
        "k",
        "N",
        "BLOCK_SIZE",
        "DESCENDING",
    ],
)


# 1-STAGE KERNEL (tiny rows)
@triton.jit(repr=_topk_kernel_repr)
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
    USE_TDM: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    idxs = offs.to(tl.int64)
    out_v_ptr = OUT_V + pid * stride_ovm
    out_i_ptr = OUT_I + pid * stride_oim

    if USE_TDM:
        row_desc = tl.make_tensor_descriptor(
            base=X + pid * stride_xm,
            shape=(1, M),
            strides=(M, 1),
            block_shape=(1, BLOCK),
        )
        vals = tl.reshape(row_desc.load([0, 0]), (BLOCK,)).to(tl.float32)
    else:
        row_ptr = X + pid * stride_xm
        mask = offs < M
        # FILL_VALUE = tl.constexpr(torch.finfo(torch.float32).min)
        vals = tl.load(row_ptr + offs, mask=mask, other=FILL_VALUE).to(tl.float32)

    # Track candidates by index, not by value: a value sentinel cannot be told
    # apart from a real input equal to it (e.g. -inf in masked logits).
    valid = offs < M
    vals = tl.where(valid, vals, FILL_VALUE)

    # unrolled exactly K iterations -- no break/continue needed
    for j in core.static_range(0, K):
        vmax = tl.max(vals, axis=0)
        # lowest index among the remaining candidates equal to the max
        arg = tl.min(tl.where(valid & (vals == vmax), idxs, BLOCK), axis=0)

        tl.store(out_v_ptr + j, vmax)
        tl.store(out_i_ptr + j, arg)

        valid = valid & (idxs != arg)
        vals = tl.where(valid, vals, FILL_VALUE)


# 2-STAGE KERNEL (large rows)
@triton.jit(repr=_topk_stage1_kernel_repr)
def topk_stage1_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    k,
    N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    FILL_VALUE: tl.constexpr,
    USE_TDM: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_chunk_idx = tl.program_id(1)
    chunk_num = tl.num_programs(1)

    y_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k
    index_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k

    chunk_offset = cur_chunk_idx * CHUNK_SIZE
    cols = tl.arange(0, CHUNK_SIZE)

    if USE_TDM:
        x_desc = tl.make_tensor_descriptor(
            base=x_ptr + cur_batch * N,
            shape=(1, N),
            strides=(N, 1),
            block_shape=(1, CHUNK_SIZE),
        )
        x_val = tl.reshape(x_desc.load([0, chunk_offset]), (CHUNK_SIZE,)).to(tl.float32)
    else:
        x_ptr += cur_batch * N + chunk_offset
        mask = (chunk_offset + cols) < N
        # FILL_VALUE = tl.constexpr(
        #    torch.finfo(torch.float32).min if DESCENDING else torch.finfo(torch.float32).max
        # )
        x_val = tl.load(x_ptr + cols, mask=mask, other=FILL_VALUE).to(tl.float32)

    # Track candidates by index, not by value: a value sentinel cannot be told
    # apart from a real input equal to it (e.g. -inf in masked logits).
    valid = (chunk_offset + cols) < N
    x_val = tl.where(valid, x_val, FILL_VALUE)
    for k_idx in range(k):
        if DESCENDING:
            chunk_select_val = tl.max(x_val, axis=0)
        else:
            chunk_select_val = tl.min(x_val, axis=0)
        # lowest index among the remaining candidates equal to the extremum.
        # Only the last chunk can run out of candidates (CHUNK_SIZE >= k); it
        # then yields CHUNK_SIZE, i.e. a global index >= N, which stage 2
        # ranks below every in-range index of equal value.
        chunk_select_idx = tl.min(
            tl.where(valid & (x_val == chunk_select_val), cols, CHUNK_SIZE), axis=0
        )

        tl.store(y_ptr + k_idx, chunk_select_val)
        tl.store(index_ptr + k_idx, chunk_select_idx + chunk_offset)

        valid = valid & (cols != chunk_select_idx)
        x_val = tl.where(valid, x_val, FILL_VALUE)


@triton.jit
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]

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

    # Fetch the partner at 'stride' 2**(n_dims - i - 1) by xor-reducing the bit
    # patterns of each pair: exact for every value, unlike slicing the floats
    # with a 0/1 multiply, which turns -inf into NaN and -0.0 into +0.0.
    ix = x.to(idtype, bitcast=True)
    ix_pair = core.reshape(ix, shape)
    iy = ix_pair ^ tl.xor_sum(ix_pair, 1, keep_dims=True)
    y = core.reshape(iy, x.shape).to(x.dtype, bitcast=True)

    ix_idx = ids.to(idx_dtype, bitcast=True)
    ix_idx_pair = core.reshape(ix_idx, shape)
    iy_idx = ix_idx_pair ^ tl.xor_sum(ix_idx_pair, 1, keep_dims=True)
    y_ids = core.reshape(iy_idx, ids.shape).to(ids.dtype, bitcast=True)

    # Rank the pair as (left, right) in both slots, so that both agree on the
    # swap even when the values are unordered (NaN).
    is_right = core.reshape(
        core.broadcast_to(core.arange(0, 2)[None, :, None] == 1, shape), x.shape
    )
    left = core.where(is_right, y, x)
    right = core.where(is_right, x, y)
    left_ids = core.where(is_right, y_ids, ids)
    right_ids = core.where(is_right, ids, y_ids)

    # Order by value, ties by lower index. Padding and exhausted-chunk entries
    # carry indices >= N, so they rank below in-range entries of equal value.
    left_first = (left > right) | ((left == right) & (left_ids < right_ids))
    swap = left_first != (flip == 1)

    ret = core.where(swap, y, x)
    ret_idx = core.where(swap, y_ids, ids)
    return ret, ret_idx


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


@triton.jit(repr=_topk_stage2_kernel_repr)
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
    USE_TDM: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    y_ptr += cur_batch * k
    index_ptr += cur_batch * k
    cols = tl.arange(0, BLOCK_SIZE)

    if USE_TDM:
        cx_desc = tl.make_tensor_descriptor(
            base=chunk_x + cur_batch * N,
            shape=(1, N),
            strides=(N, 1),
            block_shape=(1, BLOCK_SIZE),
        )
        ci_desc = tl.make_tensor_descriptor(
            base=chunk_index + cur_batch * N,
            shape=(1, N),
            strides=(N, 1),
            block_shape=(1, BLOCK_SIZE),
        )
        chunk_x_val = tl.reshape(cx_desc.load([0, 0]), (BLOCK_SIZE,)).to(tl.float32)
        chunk_index_val = tl.reshape(ci_desc.load([0, 0]), (BLOCK_SIZE,)).to(tl.int32)
        # out-of-range lanes of the descriptor load are zero-filled; mark them as
        # padding like the masked load below does
        chunk_x_val = tl.where(cols < N, chunk_x_val, FILL_VALUE)
        chunk_index_val = tl.where(cols < N, chunk_index_val, MASK_INDEX_VAL)
    else:
        chunk_x += cur_batch * N
        chunk_index += cur_batch * N
        mask = cols < N
        chunk_x_val = tl.load(chunk_x + cols, mask=mask, other=FILL_VALUE).to(
            tl.float32
        )
        chunk_index_val = tl.load(
            chunk_index + cols, mask=mask, other=MASK_INDEX_VAL
        ).to(tl.int32)

    sorted_chunk_x, sorted_chunk_index = argsort(
        chunk_x_val, chunk_index_val, 0, descending=DESCENDING
    )
    tl.store(y_ptr + cols, sorted_chunk_x, mask=cols < k)
    tl.store(index_ptr + cols, sorted_chunk_index, mask=cols < k)
