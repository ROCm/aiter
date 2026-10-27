# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""All-reduce + residual add + RMSNorm (+ fp8 group quant). The receiver of a row
holds the whole row, so it adds, normalizes and quantizes the row before it stores
anything, as a GEMM epilogue does.

A group of 128 values of a row moves as 24 lines of 2 units [value, value, value,
tag]: unit u of the group has the values u, 48 + u and 96 + u of the group. 24
lines are 3 whole 128-byte cache lines: a store of a part of a cache line into a
peer is slower. In the tile, a group is 32 lines, so one wave holds the group and
finds its fp8 scale."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx1250.comms.all_reduce import (
    _hi,
    _lines,
    _lo,
    _poll,
    _post,
    _unit_words,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_all_reduce_rmsnorm_repr = make_kernel_repr(
    "_all_reduce_rmsnorm_kernel", ["RANK", "HIDDEN", "GEMMA", "QUANT"]
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
):
    """Program r sends row r of x. Programs tokens to tokens + P - 1 receive and
    store rows p, p + P, ... (P = programs), so at most P receivers poll."""
    WORDS: gl.constexpr = gl.BlockedLayout([1, 2], [32, 1], [gl.num_warps(), 1], [1, 0])
    LINE: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    GROUPS: gl.constexpr = HIDDEN // 128
    pid = gl.program_id(0)
    line = gl.arange(0, 32 * GROUPS, layout=gl.SliceLayout(1, WORDS))[:, None]
    ok = line % 32 < 24  # a group has 24 lines, 32 in the tile
    ok2 = line % 32 < 16  # value 2 of lines 16 to 23 is past the group
    first = 128 * (line // 32) + 2 * (line % 32)  # in the row
    col = first + gl.arange(0, 2, layout=gl.SliceLayout(0, WORDS))[None, :]
    line = gl.arange(0, 32 * GROUPS, layout=gl.SliceLayout(1, LINE))[:, None]
    sent = line % 32 < 24
    pos = 24 * (line // 32) + line % 32  # in the row
    pos = 4 * pos + gl.arange(0, 4, layout=gl.SliceLayout(0, LINE))[None, :]
    if pid < tokens:
        call, lines = _lines(x + pid * HIDDEN, col, ok, ok, ok2, 48, calls + 2 * pid)
        # All 24 lines of each group on each call: no old tag can stay.
        _post(lines, inboxes, 96 * GROUPS * pid + pos, sent, call, RANK, SLOT)
    else:
        # The weight is the same for all rows: load it once.
        w0 = gl.amd.cdna5.buffer_load(weight, col, mask=ok, other=0).to(gl.float32)
        w1 = gl.amd.cdna5.buffer_load(weight, col + 48, mask=ok, other=0)
        w2 = gl.amd.cdna5.buffer_load(weight, col + 96, mask=ok2, other=0)
        w1 = w1.to(gl.float32)
        w2 = w2.to(gl.float32)
        if GEMMA:
            w0 += 1.0
            w1 += 1.0
            w2 += 1.0
        for row in range(pid - tokens, tokens, programs):
            call, lines = _lines(
                x + row * HIDDEN, col, ok, ok, ok2, 48, calls + 2 * row + 1
            )
            # The residual is local: load it before the wait.
            res = residual + row * HIDDEN
            r0 = gl.amd.cdna5.buffer_load(res, col, mask=ok, other=0)
            r1 = gl.amd.cdna5.buffer_load(res, col + 48, mask=ok, other=0)
            r2 = gl.amd.cdna5.buffer_load(res, col + 96, mask=ok2, other=0)
            row_pos = 96 * GROUPS * row + pos
            l0, l1, l2, l3 = _poll(
                inboxes, row_pos, sent, call, lines, err, RANK, SLOT, MAXIT
            )
            lo0, hi0 = _unit_words(l0)
            lo1, hi1 = _unit_words(l1)
            lo2, hi2 = _unit_words(l2)
            lo3, hi3 = _unit_words(l3)
            # Rank order in fp32, then the residual, so that all ranks get the same bits.
            s0 = _lo(lo0) + _lo(lo1) + _lo(lo2) + _lo(lo3) + r0.to(gl.float32)
            s1 = _hi(lo0) + _hi(lo1) + _hi(lo2) + _hi(lo3) + r1.to(gl.float32)
            s2 = _lo(hi0) + _lo(hi1) + _lo(hi2) + _lo(hi3) + r2.to(gl.float32)
            ro = residual_out + row * HIDDEN
            gl.amd.cdna5.buffer_store(s0.to(gl.bfloat16), ro, col, mask=ok)
            gl.amd.cdna5.buffer_store(s1.to(gl.bfloat16), ro, col + 48, mask=ok)
            gl.amd.cdna5.buffer_store(s2.to(gl.bfloat16), ro, col + 96, mask=ok2)

            # The values past the groups are 0 here.
            square_sum = gl.sum(gl.sum(s0 * s0 + s1 * s1 + s2 * s2, axis=1), axis=0)
            rstd = gl.rsqrt(square_sum / HIDDEN + eps)
            y0 = s0 * rstd * w0
            y1 = s1 * rstd * w1
            y2 = s2 * rstd * w2
            o = out + row * HIDDEN
            gl.amd.cdna5.buffer_store(y0.to(gl.bfloat16), o, col, mask=ok)
            gl.amd.cdna5.buffer_store(y1.to(gl.bfloat16), o, col + 48, mask=ok)
            gl.amd.cdna5.buffer_store(y2.to(gl.bfloat16), o, col + 96, mask=ok2)

            if QUANT:
                # A group is 32 lines of the tile. 448 is the fp8 maximum.
                a = gl.maximum(gl.maximum(gl.abs(y0), gl.abs(y1)), gl.abs(y2))
                a = gl.reshape(a, [GROUPS, 32, 2])
                group_max = gl.max(gl.max(a, axis=2), axis=1)
                group_scale = gl.maximum(group_max, 1e-10) / 448
                s = gl.expand_dims(gl.expand_dims(group_scale, 1), 2)
                s = gl.reshape(gl.broadcast(s, a)[0], [32 * GROUPS, 2])
                s = gl.convert_layout(s, WORDS, assert_trivial=True)
                q0 = (y0 / s).to(gl.float8e4nv)
                q1 = (y1 / s).to(gl.float8e4nv)
                q2 = (y2 / s).to(gl.float8e4nv)
                qo = q + row * HIDDEN
                gl.amd.cdna5.buffer_store(q0, qo, col, mask=ok)
                gl.amd.cdna5.buffer_store(q1, qo, col + 48, mask=ok)
                gl.amd.cdna5.buffer_store(q2, qo, col + 96, mask=ok2)
                lead = col % 128 == 0  # one scale for each group
                so = scale + row * GROUPS
                gl.amd.cdna5.buffer_store(s, so, col // 128, mask=lead)
