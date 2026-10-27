# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""All-reduce + residual add + RMSNorm (+ fp8 group quant). The receiver of a row
holds the whole row, so it adds, normalizes and quantizes the row before it stores
anything, as a GEMM epilogue does.

A program sends or receives 3 rows (a triple). Unit c of a triple has value c of
each of the 3 rows and the tag, so a triple of hidden values is hidden / 2 lines
of 16 bytes. A lane holds 4 columns of each row, and a wave a group of 128 columns:
the fp8 scale of a group is in one wave."""

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


@gluon.jit
def _columns(words):
    """Words [2n, 2] of the lines as [n, 4]: line h * n + i has columns 2h and 2h + 1
    of row i."""
    ROW: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    N: gl.constexpr = words.shape[0] // 2
    words = gl.permute(gl.reshape(words, [2, N, 2]), (1, 0, 2))
    return gl.convert_layout(gl.reshape(words, [N, 4]), ROW, assert_trivial=True)


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
    """Program t sends rows 3t to 3t + 2 of x. Programs T to T + P - 1 receive and
    store triples p, p + P, ... (T triples, P = programs), so at most P receivers
    poll. The rows from tokens on are 0 and are not stored."""
    ROW: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    N: gl.constexpr = HIDDEN // 4  # 4 columns of a row for each lane
    GROUPS: gl.constexpr = HIDDEN // 128
    triples = gl.cdiv(tokens, 3)
    pid = gl.program_id(0)
    lane = gl.arange(0, N, layout=gl.SliceLayout(1, ROW))[:, None]
    col = 4 * lane + gl.arange(0, 4, layout=gl.SliceLayout(0, ROW))[None, :]
    # In the inbox, the 128 columns of a group are 64 lines: columns 4i and 4i + 1
    # in line i, 4i + 2 and 4i + 3 in line 32 + i. Then a store of a wave writes
    # 512 contiguous bytes.
    line = gl.arange(0, 2 * N, layout=gl.SliceLayout(1, ROW))[:, None]
    i = line % N
    pos = 64 * (i // 32) + 32 * (line // N) + i % 32
    pos = 4 * pos + gl.arange(0, 4, layout=gl.SliceLayout(0, ROW))[None, :]
    if pid < triples:
        row = 3 * pid
        call, lines = _lines(
            x + row * HIDDEN,
            col,
            row < tokens,
            row + 1 < tokens,
            row + 2 < tokens,
            HIDDEN,
            calls + 2 * pid,
        )
        # All lines of the triple on each call: no old tag can stay.
        _post(lines, inboxes, 2 * HIDDEN * pid + pos, line < 2 * N, call, RANK, SLOT)
    else:
        # The weight is the same for all rows: load it once.
        w = gl.amd.cdna5.buffer_load(weight, col).to(gl.float32)
        if GEMMA:
            w += 1.0
        for t in range(pid - triples, triples, programs):
            row = 3 * t
            call, lines = _lines(
                x + row * HIDDEN,
                col,
                row < tokens,
                row + 1 < tokens,
                row + 2 < tokens,
                HIDDEN,
                calls + 2 * t + 1,
            )
            # The residual is local: load it before the wait.
            res = residual + row * HIDDEN
            r0 = gl.amd.cdna5.buffer_load(res, col, mask=row < tokens, other=0)
            r1 = gl.amd.cdna5.buffer_load(
                res, col + HIDDEN, mask=row + 1 < tokens, other=0
            )
            r2 = gl.amd.cdna5.buffer_load(
                res, col + 2 * HIDDEN, mask=row + 2 < tokens, other=0
            )
            l0, l1, l2, l3 = _poll(
                inboxes,
                2 * HIDDEN * t + pos,
                line < 2 * N,
                call,
                lines,
                err,
                RANK,
                SLOT,
                MAXIT,
            )
            lo0, hi0 = _unit_words(l0)
            lo1, hi1 = _unit_words(l1)
            lo2, hi2 = _unit_words(l2)
            lo3, hi3 = _unit_words(l3)
            lo0 = _columns(lo0)
            hi0 = _columns(hi0)
            lo1 = _columns(lo1)
            hi1 = _columns(hi1)
            lo2 = _columns(lo2)
            hi2 = _columns(hi2)
            lo3 = _columns(lo3)
            hi3 = _columns(hi3)
            # Rank order in fp32, then the residual, so that all ranks get the same
            # bits.
            s0 = _lo(lo0) + _lo(lo1) + _lo(lo2) + _lo(lo3) + r0.to(gl.float32)
            s1 = _hi(lo0) + _hi(lo1) + _hi(lo2) + _hi(lo3) + r1.to(gl.float32)
            s2 = _lo(hi0) + _lo(hi1) + _lo(hi2) + _lo(hi3) + r2.to(gl.float32)
            # One reduction over the waves for the 3 rows, before the stores. After
            # the reshape, join(join(a, b), join(c, d)) is [a, c, b, d].
            sq = gl.join(gl.join(s0 * s0, s2 * s2), gl.join(s1 * s1, s2 * 0))
            sq = gl.sum(gl.sum(gl.reshape(sq, [N, 4, 4]), axis=1), axis=0)
            idx = gl.arange(0, 4, layout=sq.type.layout)
            for r in gl.static_range(3):
                if row + r < tokens:  # the last triple can have empty rows
                    at = (row + r) * HIDDEN + col
                    s = (s0, s1, s2)[r]
                    gl.amd.cdna5.buffer_store(s.to(gl.bfloat16), residual_out, at)
                    square_sum = gl.sum(gl.where(idx == r, sq, 0.0), axis=0)
                    y = s * gl.rsqrt(square_sum / HIDDEN + eps) * w
                    gl.amd.cdna5.buffer_store(y.to(gl.bfloat16), out, at)

                    if QUANT:
                        # A group is 32 lanes of one wave. 448 is the fp8 maximum.
                        a = gl.reshape(gl.abs(y), [GROUPS, 32, 4])
                        group_max = gl.max(gl.max(a, axis=2), axis=1)
                        group_scale = gl.maximum(group_max, 1e-10) / 448
                        sc = gl.expand_dims(gl.expand_dims(group_scale, 1), 2)
                        sc = gl.reshape(gl.broadcast(sc, a)[0], [N, 4])
                        sc = gl.convert_layout(sc, ROW, assert_trivial=True)
                        yq = (y / sc).to(gl.float8e4nv)
                        gl.amd.cdna5.buffer_store(yq, q, at)
                        lead = col % 128 == 0  # one scale for each group
                        at = (row + r) * GROUPS + col // 128
                        gl.amd.cdna5.buffer_store(sc, scale, at, mask=lead)
