# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton kernels of the pull collectives. The wrapper and the protocol are in
``aiter/ops/triton/comms/pull_collective.py``."""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_wait_repr = make_kernel_repr("_pull_wait_kernel", ["FLAG", "DELTA"])
_publish_wait_repr = make_kernel_repr("_pull_publish_wait_kernel", [])
_copy_repr = make_kernel_repr("_pull_copy_kernel", ["W", "BLOCK", "ALL_TO_ALL"])
_rs_repr = make_kernel_repr("_pull_rs_kernel", ["W", "BLOCK"])
_rs_persistent_repr = make_kernel_repr("_pull_rs_persistent_kernel", ["W", "BLOCK"])
_a2a_mxfp8_repr = make_kernel_repr(
    "_pull_a2a_mxfp8_kernel", ["W", "D64", "S64", "RC", "RS"]
)


@triton.jit
def _spin_ge(ptr, target):
    v = tl.atomic_add(ptr, 0, sem="acquire", scope="sys")
    while v < target:
        v = tl.atomic_add(ptr, 0, sem="acquire", scope="sys")


@triton.jit(repr=_wait_repr)
def _pull_wait_kernel(base_ptrs, epoch_ptr, FLAG: tl.constexpr, DELTA: tl.constexpr):
    # program r polls peer r's flag: one poller per peer
    r = tl.program_id(0)
    e = tl.load(epoch_ptr)
    fp = tl.load(base_ptrs + r).to(tl.pointer_type(tl.int64))
    _spin_ge(fp + FLAG, e + DELTA)


@triton.jit(repr=_publish_wait_repr)
def _pull_publish_wait_kernel(base_ptrs, epoch_ptr, rank):
    # publish my ready flag (after the producer's writes, by stream order), then
    # wait for every peer's
    r = tl.program_id(0)
    e = tl.load(epoch_ptr)
    fp = tl.load(base_ptrs + r).to(tl.pointer_type(tl.int64))
    if r == rank:
        tl.atomic_xchg(fp, e + 1, sem="release", scope="sys")
    _spin_ge(fp, e + 1)


@triton.jit
def _finish(base_ptrs, epoch_ptr, cnt_ptr, e, rank, total):
    # the last block of the op publishes done and advances the epoch
    old = tl.atomic_add(cnt_ptr, 1, sem="acq_rel", scope="gpu")
    if old == total - 1:
        tl.store(cnt_ptr, 0)
        mine = tl.load(base_ptrs + rank).to(tl.pointer_type(tl.int64))
        tl.atomic_xchg(mine + 1, e + 1, sem="release", scope="sys")
        tl.store(epoch_ptr, e + 1)


@triton.jit(repr=_copy_repr)
def _pull_copy_kernel(
    out,
    base_ptrs,
    dptrs,
    epoch_ptr,
    cnt_ptr,
    src_off64,
    n64,
    rank,
    total,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
    ALL_TO_ALL: tl.constexpr,
):
    # all-gather: out[r] = peer r's n64 int64s at src_off64
    # all-to-all: out[r] = peer r's block `rank` (its n64-element block number rank)
    pid = tl.program_id(0)
    r = tl.program_id(1)
    e = tl.load(epoch_ptr)
    src = tl.load(dptrs + r).to(tl.pointer_type(tl.int64)) + src_off64
    if ALL_TO_ALL:
        src += rank * n64.to(tl.int64)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n64
    v = tl.load(src + offs, mask=m)
    tl.store(out.to(tl.pointer_type(tl.int64)) + r.to(tl.int64) * n64 + offs, v, mask=m)
    _finish(base_ptrs, epoch_ptr, cnt_ptr, e, rank, total)


@triton.jit(repr=_rs_repr)
def _pull_rs_kernel(
    out,
    base_ptrs,
    dptrs,
    epoch_ptr,
    cnt_ptr,
    shard,
    rank,
    total,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    e = tl.load(epoch_ptr)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < shard
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for r in tl.static_range(W):  # ascending rank order, fp32
        p = tl.load(dptrs + r).to(tl.pointer_type(tl.bfloat16)) + rank * shard.to(
            tl.int64
        )
        acc += tl.load(p + offs, mask=m).to(tl.float32)
    tl.store(out.to(tl.pointer_type(tl.bfloat16)) + offs, acc.to(tl.bfloat16), mask=m)
    _finish(base_ptrs, epoch_ptr, cnt_ptr, e, rank, total)


@triton.jit(repr=_rs_persistent_repr)
def _pull_rs_persistent_kernel(
    out,
    base_ptrs,
    dptrs,
    epoch_ptr,
    cnt_ptr,
    shard,
    rank,
    nprog,
    ntiles,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Persistent: nprog programs loop over the tiles of my shard. Each tile sums
    # the W ranks' rows in fp32, ascending rank order, and rounds once.
    pid = tl.program_id(0)
    e = tl.load(epoch_ptr)
    base = rank * shard.to(tl.int64)
    for t in range(pid, ntiles, nprog):
        offs = t.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < shard
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for r in tl.static_range(W):
            p = tl.load(dptrs + r).to(tl.pointer_type(tl.bfloat16)) + base
            acc += tl.load(p + offs, mask=m).to(tl.float32)
        tl.store(
            out.to(tl.pointer_type(tl.bfloat16)) + offs, acc.to(tl.bfloat16), mask=m
        )
    _finish(base_ptrs, epoch_ptr, cnt_ptr, e, rank, nprog)


@triton.jit(repr=_a2a_mxfp8_repr)
def _pull_a2a_mxfp8_kernel(
    out_c,
    out_s,
    base_ptrs,
    dptrs,
    epoch_ptr,
    cnt_ptr,
    rank,
    rows,
    soff64,
    nc,
    total,
    W: tl.constexpr,
    D64: tl.constexpr,
    S64: tl.constexpr,
    RC: tl.constexpr,
    RS: tl.constexpr,
):
    # Peer p's buffer holds codes [W*rows, d] then scales [W*rows, d/32]; row block j
    # is for rank j. Pull row block `rank` of both fields from every peer and store
    # it at column block p of out_c [rows, W*d] / out_s [rows, W*d/32], the layout
    # a GEMM over the concatenated heads reads, so no transpose copy follows.
    pid = tl.program_id(0)
    r = tl.program_id(1)
    e = tl.load(epoch_ptr)
    srcp = tl.load(dptrs + r).to(tl.pointer_type(tl.int64))
    if pid < nc:
        rb = pid * RC + tl.arange(0, RC)
        c = tl.arange(0, D64)
        m = (rb < rows)[:, None]
        src = srcp + ((rank * rows + rb)[:, None]).to(tl.int64) * D64 + c[None, :]
        v = tl.load(src, mask=m)
        dst = (
            out_c.to(tl.pointer_type(tl.int64))
            + rb[:, None].to(tl.int64) * (W * D64)
            + r * D64
            + c[None, :]
        )
        tl.store(dst, v, mask=m)
    else:
        rb2 = (pid - nc) * RS + tl.arange(0, RS)
        c2 = tl.arange(0, S64)
        m2 = (rb2 < rows)[:, None]
        src2 = (
            srcp
            + soff64
            + ((rank * rows + rb2)[:, None]).to(tl.int64) * S64
            + c2[None, :]
        )
        v2 = tl.load(src2, mask=m2)
        dst2 = (
            out_s.to(tl.pointer_type(tl.int64))
            + rb2[:, None].to(tl.int64) * (W * S64)
            + r * S64
            + c2[None, :]
        )
        tl.store(dst2, v2, mask=m2)
    _finish(base_ptrs, epoch_ptr, cnt_ptr, e, rank, total)
