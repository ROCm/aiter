# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon grouped row-reduce for gfx1250 MoE scatter-combine; the gluon path for
reduce_grouped's plain (no-swiglu) sum. One workgroup per group gathers its K*B
rows with TDM and sums them in-register, with N split across waves. N need not be
a power of 2: the TDM block is padded to the next pow2 and the out-of-bounds
columns are zero-filled, then masked off on store.
"""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def reduce_grouped_gluon(
    X,            # [B*M, N] view of x[B, M, N]
    Out,          # [num_groups, N]
    InIndx,       # [num_groups, K] gather indices (flattened)
    stride_xm,
    stride_om,
    stride_on,
    M,
    N: gl.constexpr,
    N_PAD: gl.constexpr,   # next_power_of_2(N)
    B: gl.constexpr,
    K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    group = gl.program_id(0)
    SIZE_N: gl.constexpr = N_PAD // (NUM_WARPS * 32)
    BLKN: gl.constexpr = gl.BlockedLayout([1, SIZE_N], [1, 32], [1, NUM_WARPS], [1, 0])
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    # block is pow2 N_PAD over a tensor of true width N; OOB cols are zero-filled
    x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        X, [B * M, N], [stride_xm, 1], [1, N_PAD], SH)
    smem = gl.allocate_shared_memory(X.dtype.element_ty, [K * B, 1, N_PAD], SH)

    buf = 0
    for i in gl.static_range(K):
        idx_i = gl.load(InIndx + group * K + i)
        for b in gl.static_range(B):
            gl.amd.gfx1250.tdm.async_load(x_desc, [b * M + idx_i, 0], smem.index(buf))
            buf += 1
    gl.amd.gfx1250.tdm.async_wait(0)

    acc = gl.zeros([1, N_PAD], dtype=gl.float32, layout=BLKN)
    buf = 0
    for i in gl.static_range(K):
        for b in gl.static_range(B):
            acc += smem.index(buf).load(BLKN).to(gl.float32)
            buf += 1

    offs_n = gl.arange(0, N_PAD, layout=gl.SliceLayout(0, BLKN))
    o_offs = group * stride_om + offs_n[None, :] * stride_on
    gl.amd.gfx1250.buffer_store(
        acc.to(Out.dtype.element_ty), Out, o_offs, mask=offs_n[None, :] < N)
