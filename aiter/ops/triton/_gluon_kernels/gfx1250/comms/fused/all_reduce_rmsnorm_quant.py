# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""All-reduce + residual add + RMSNorm (+ fp8 quant) over 4 GPUs in one kernel.
A row is cut into chunks of 384 values: unit u of chunk k has the values 384k + u,
384k + 128 + u and 384k + 256 + u of the row, as in the plain all-reduce. A program
sends or receives one row, so a receiver holds the whole row: it adds, normalizes and
quantizes it before it stores, as a GEMM epilogue does."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.runtime.jit import constexpr_function

from aiter.ops.triton._gluon_kernels.gfx1250.comms.all_reduce import (
    _lines,
    _post,
    _receive,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@constexpr_function
def _slots(chunks):
    """The chunks of a row, up to a power of 2: a tensor dimension."""
    return 1 << (chunks - 1).bit_length()


@gluon.jit
def _columns(v):
    """The values [2N, 2] of the lines as [N, 4]: line h * N + i has columns 2h and
    2h + 1 of lane i."""
    ROW: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    N: gl.constexpr = v.shape[0] // 2
    v = gl.permute(gl.reshape(v, [2, N, 2]), (1, 0, 2))
    return gl.convert_layout(gl.reshape(v, [N, 4]), ROW, assert_trivial=True)


@gluon.jit
def _add_max(sum_a, max_a, sum_b, max_b):
    return sum_a + sum_b, gl.maximum(max_a, max_b)


@gluon.jit
def _group_quant(y, mask, q, at, scale, scale_at, lead):
    """Stores y as fp8 at at, with one scale for each 128 values: a plane of a chunk.
    The lanes in lead store the scales at scale_at."""
    ROW: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    a = gl.reshape(gl.abs(y), [y.shape[0] // 32, 32, 4])
    big = gl.maximum(gl.max(gl.max(a, axis=2), axis=1), 1e-10)
    big = gl.expand_dims(gl.expand_dims(big, 1), 2)
    # One division for each group. 448 is the fp8 maximum.
    inv = gl.reshape(gl.broadcast(448 / big, a)[0], y.shape)
    inv = gl.convert_layout(inv, ROW, assert_trivial=True)
    sc = gl.reshape(gl.broadcast(big / 448, a)[0], y.shape)
    sc = gl.convert_layout(sc, ROW, assert_trivial=True)
    gl.amd.cdna5.buffer_store((y * inv).to(gl.float8e4nv), q, at, mask=mask)
    gl.amd.cdna5.buffer_store(sc, scale, scale_at, mask=mask & lead)


_all_reduce_rmsnorm_repr = make_kernel_repr(
    "_all_reduce_rmsnorm_kernel",
    ["RANK", "HIDDEN", "GEMMA", "QUANT", "GROUP", "STORE_OUT"],
)


@gluon.jit(repr=_all_reduce_rmsnorm_repr, do_not_specialize=["tokens", "programs"])
def _all_reduce_rmsnorm_kernel(
    x,
    residual,
    weight,
    out,
    residual_out,
    q,
    scale,
    inboxes,
    calls,
    err,
    tokens,
    programs,
    eps,
    RANK: gl.constexpr,
    HIDDEN: gl.constexpr,
    SLOT: gl.constexpr,
    MAXIT: gl.constexpr,
    GEMMA: gl.constexpr,
    QUANT: gl.constexpr,
    GROUP: gl.constexpr,
    STORE_OUT: gl.constexpr,
):
    """Program r < T sends row r of x (T = tokens). Programs T to T + P - 1 receive
    rows p, p + P, ... (P = programs). An fp8 group has GROUP values: 128, or HIDDEN
    for the per-token quant. Without STORE_OUT, out is not stored."""
    # Lane i of chunk k has the columns col = 384k + 4i to 384k + 4i + 3, and col + 128
    # and col + 256 (planes 1 and 2). A plane of a chunk is one fp8 group of 128.
    ROW: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    CHUNKS: gl.constexpr = (HIDDEN + 383) // 384
    N: gl.constexpr = 32 * _slots(CHUNKS)
    WORDS: gl.constexpr = 256 * CHUNKS  # int32 words of inbox for a row
    GROUPS: gl.constexpr = HIDDEN // GROUP
    pid = gl.program_id(0)
    lane = gl.arange(0, N, layout=gl.SliceLayout(1, ROW))[:, None]
    col = 384 * (lane // 32) + 4 * (lane % 32)
    col += gl.arange(0, 4, layout=gl.SliceLayout(0, ROW))[None, :]
    ok0 = col < HIDDEN
    ok1 = col + 128 < HIDDEN
    ok2 = col + 256 < HIDDEN
    # A chunk is 64 lines: columns 4i and 4i + 1 in line i, 4i + 2 and 4i + 3 in line
    # 32 + i. Then a store of a wave writes 512 contiguous bytes.
    line = gl.arange(0, 2 * N, layout=gl.SliceLayout(1, ROW))[:, None]
    i = line % N
    real = i // 32 < CHUNKS  # the lanes past the last chunk have no lines
    pos = 64 * (i // 32) + 32 * (line // N) + i % 32
    pos = 4 * pos + gl.arange(0, 4, layout=gl.SliceLayout(0, ROW))[None, :]
    if pid < tokens:
        call, lines = _lines(x + pid * HIDDEN, col, ok0, ok1, ok2, 128, calls + 2 * pid)
        # All lines of the row on each call: no old tag can stay.
        _post(lines, inboxes, WORDS * pid + pos, real, call, RANK, SLOT)
    else:
        # The weight once for all rows.
        w0 = gl.amd.cdna5.buffer_load(weight, col, mask=ok0, other=0)
        w1 = gl.amd.cdna5.buffer_load(weight, col + 128, mask=ok1, other=0)
        w2 = gl.amd.cdna5.buffer_load(weight, col + 256, mask=ok2, other=0)
        w0 = w0.to(gl.float32)
        w1 = w1.to(gl.float32)
        w2 = w2.to(gl.float32)
        if GEMMA:
            w0 += 1.0
            w1 += 1.0
            w2 += 1.0
        lead = col % GROUP == 0  # the first lane of each fp8 group
        for row in range(pid - tokens, tokens, programs):
            call, lines = _lines(
                x + row * HIDDEN, col, ok0, ok1, ok2, 128, calls + 2 * row + 1
            )
            # The residual is local: load it before the wait.
            res = residual + row * HIDDEN
            r0 = gl.amd.cdna5.buffer_load(res, col, mask=ok0, other=0)
            r1 = gl.amd.cdna5.buffer_load(res, col + 128, mask=ok1, other=0)
            r2 = gl.amd.cdna5.buffer_load(res, col + 256, mask=ok2, other=0)
            s0, s1, s2 = _receive(
                inboxes,
                WORDS * row + pos,
                real,
                call,
                lines,
                err,
                RANK,
                SLOT,
                MAXIT,
            )
            # The residual last, in fp32: all ranks get the same bits.
            s0 = _columns(s0) + r0.to(gl.float32)
            s1 = _columns(s1) + r1.to(gl.float32)
            s2 = _columns(s2) + r2.to(gl.float32)
            at = row * HIDDEN + col
            gl.amd.cdna5.buffer_store(s0.to(gl.bfloat16), residual_out, at, mask=ok0)
            gl.amd.cdna5.buffer_store(
                s1.to(gl.bfloat16), residual_out, at + 128, mask=ok1
            )
            gl.amd.cdna5.buffer_store(
                s2.to(gl.bfloat16), residual_out, at + 256, mask=ok2
            )
            # out = (s * w) * rstd, so the largest |out| is rstd * the largest |s * w|.
            p0 = s0 * w0
            p1 = s1 * w1
            p2 = s2 * w2
            square = s0 * s0 + s1 * s1 + s2 * s2
            if QUANT and GROUP == HIDDEN:
                # One reduction over the waves for the square sum and the max.
                big = gl.maximum(gl.maximum(gl.abs(p0), gl.abs(p1)), gl.abs(p2))
                square, big = gl.reduce((square, big), 1, _add_max)
                square_sum, big = gl.reduce((square, big), 0, _add_max)
            else:
                square_sum = gl.sum(gl.sum(square, axis=1), axis=0)
            rstd = gl.rsqrt(square_sum / HIDDEN + eps)
            y0 = p0 * rstd
            y1 = p1 * rstd
            y2 = p2 * rstd
            if STORE_OUT:
                gl.amd.cdna5.buffer_store(y0.to(gl.bfloat16), out, at, mask=ok0)
                gl.amd.cdna5.buffer_store(y1.to(gl.bfloat16), out, at + 128, mask=ok1)
                gl.amd.cdna5.buffer_store(y2.to(gl.bfloat16), out, at + 256, mask=ok2)

            if QUANT and GROUP == HIDDEN:
                big = gl.maximum(big * rstd, 1e-10)  # the largest |out| of the row
                inv = 448 / big  # one division for the row. 448 is the fp8 maximum.
                gl.amd.cdna5.buffer_store((y0 * inv).to(gl.float8e4nv), q, at, mask=ok0)
                gl.amd.cdna5.buffer_store(
                    (y1 * inv).to(gl.float8e4nv), q, at + 128, mask=ok1
                )
                gl.amd.cdna5.buffer_store(
                    (y2 * inv).to(gl.float8e4nv), q, at + 256, mask=ok2
                )
                gl.store(scale + row, big / 448)
            elif QUANT:
                first = row * GROUPS + col // GROUP
                _group_quant(y0, ok0, q, at, scale, first, lead)
                _group_quant(y1, ok1, q, at + 128, scale, first + 1, lead)
                _group_quant(y2, ok2, q, at + 256, scale, first + 2, lead)
