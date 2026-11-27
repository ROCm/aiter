# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# Split rebalance partition for the bh16 regimes of aiter.ops.triton.gluon.mla_gluon.
# The batch's KV is cut into segments of one length, so a long sequence gets more
# work items than a short one; mla_gluon_split_metadata launches this kernel.

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.common.utils import sum_combine
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.jit
def _cdiv_f32(a, b):
    """cdiv(a, b) for int32 a in [0, 2**22) and int32 b >= 1, through fp32.

    The fp32 quotient is off by at most 1, and one integer step corrects it; an
    int32 division takes several times the instructions.
    """
    n = a + b - 1
    q = (n.to(gl.float32) * (1.0 / b.to(gl.float32))).to(gl.int32)
    r = n - q * b
    return q + (r >= b).to(gl.int32) - (r < 0).to(gl.int32)


@gluon.jit
def _split_rebalance(
    B_seq_len,
    Split_meta,
    batch_size,
    work_id,
    write_meta,
    USE_2D_VIEW: gl.constexpr,
    NUM_WORK: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_B: gl.constexpr,
):
    """Map work item work_id to (batch, seq_len, kv_start, kv_end, indptr).

    Cut the batch's KV into segments of s BLOCK_N tiles: the smallest s that needs
    at most NUM_WORK (> batch_size) of them. indptr is the batch's kv_indptr entry
    (1-D view), else 0. write_meta stores each batch's first work item, and the
    total at index batch_size, for the reduce.
    """
    # BLOCK_B elements on the lanes of one warp; the other warps hold copies, so
    # the reductions and the scan stay in the warp and use no LDS.
    gl.static_assert(BLOCK_B >= 64)
    layout_b: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[BLOCK_B // 64],
        threads_per_warp=[64],
        warps_per_cta=[4],
        order=[0],
    )
    offs_b = gl.arange(0, BLOCK_B, layout=layout_b)
    mask_b = offs_b < batch_size
    if USE_2D_VIEW:
        lens = gl.load(B_seq_len + offs_b, mask=mask_b, other=0).to(gl.int32)
    else:
        starts = gl.load(B_seq_len + offs_b, mask=mask_b, other=0).to(gl.int32)
        lens = (
            gl.load(B_seq_len + offs_b + 1, mask=mask_b, other=0).to(gl.int32) - starts
        )
    tiles = (lens + BLOCK_N - 1) // BLOCK_N

    # sum_i cdiv(tiles_i, s) <= NUM_WORK holds for s >= cdiv(total, NUM_WORK - batch_size)
    # and fails for s < cdiv(total, NUM_WORK). hi always holds, so the result is
    # valid for any step count.
    total = gl.sum(tiles, axis=0)
    lo = gl.maximum((total + NUM_WORK - 1) // NUM_WORK, 1)
    hi = gl.maximum((total + NUM_WORK - batch_size - 1) // (NUM_WORK - batch_size), lo)
    # an even batch fits at the lower bound: one test, no search
    if gl.sum(_cdiv_f32(tiles, lo), axis=0) <= NUM_WORK:
        hi = lo
    else:
        lo = lo + 1
        # ceil(log2(candidates)) binary-search steps
        steps = (gl.log((hi - lo + 1).to(gl.float32)) * 1.4426950408889634 + 0.9999).to(
            gl.int32
        )
        for _ in range(steps):
            mid = (lo + hi) // 2
            if gl.sum(_cdiv_f32(tiles, mid), axis=0) <= NUM_WORK:
                hi = mid
            else:
                lo = mid + 1

    num = _cdiv_f32(tiles, hi)
    incl = gl.associative_scan(num, 0, sum_combine)
    excl = incl - num
    if write_meta:
        gl.store(Split_meta + offs_b, excl, mask=mask_b)
        gl.store(
            Split_meta + batch_size + offs_b * 0,
            excl * 0 + gl.sum(num, axis=0),
            mask=offs_b == 0,
        )

    # sequences that end at or before work_id come first; all of them past the
    # batch's last segment
    batch_raw = gl.sum((incl <= work_id).to(gl.int32), axis=0)
    batch = gl.minimum(batch_raw, batch_size - 1)
    sel = offs_b == batch
    seg_len = hi * BLOCK_N
    seq_len = gl.sum(gl.where(sel, lens, 0), axis=0)
    kv_start = (work_id - gl.sum(gl.where(sel, excl, 0), axis=0)) * seg_len
    kv_end = gl.minimum(kv_start + seg_len, seq_len) * (batch_raw < batch_size).to(
        gl.int32
    )
    if USE_2D_VIEW:
        indptr = batch * 0
    else:
        indptr = gl.sum(gl.where(sel, starts, 0), axis=0)
    return batch, seq_len, kv_start, kv_end, indptr


_mla_gluon_split_metadata_repr = make_kernel_repr(
    "_mla_gluon_split_metadata_kernel",
    ["USE_2D_VIEW", "NUM_WORK", "BLOCK_N", "BLOCK_B"],
)


@gluon.jit(repr=_mla_gluon_split_metadata_repr)
def _mla_gluon_split_metadata_kernel(
    B_seq_len,  # cache_seqlens [batch] (USE_2D_VIEW) or kv_indptr [batch + 1]
    Split_meta,  # out: int32 [batch + 1], each batch's first work item
    Split_work,  # out: int32 [NUM_WORK, 8], (batch, kv_start, kv_end, seq_len, indptr)
    batch_size,
    USE_2D_VIEW: gl.constexpr,
    NUM_WORK: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_B: gl.constexpr,
):
    """One program per work item: write its row of the partition."""
    work_id = gl.program_id(0)
    batch, seq_len, kv_start, kv_end, indptr = _split_rebalance(
        B_seq_len,
        Split_meta,
        batch_size,
        work_id,
        work_id == 0,
        USE_2D_VIEW,
        NUM_WORK,
        BLOCK_N,
        BLOCK_B,
    )
    layout_f: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1], threads_per_warp=[64], warps_per_cta=[4], order=[0]
    )
    offs_f = gl.arange(0, 8, layout=layout_f)
    row = gl.where(
        offs_f == 0,
        batch,
        gl.where(
            offs_f == 1,
            kv_start,
            gl.where(offs_f == 2, kv_end, gl.where(offs_f == 3, seq_len, indptr)),
        ),
    )
    gl.store(Split_work + work_id * 8 + offs_f, row, mask=offs_f < 5)
