# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""All-reduce + residual add + RMSNorm (+ fp8 group quant). The receiver of a row
holds the whole row, so it adds, normalizes and quantizes the row before it stores
anything, as a GEMM epilogue does."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx1250.comms.all_reduce import (
    _hi,
    _lo,
    _pack,
    _receive,
    _send,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_all_reduce_rmsnorm_repr = make_kernel_repr(
    "_all_reduce_rmsnorm_kernel", ["RANK", "LINES", "GEMMA", "QUANT"]
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
    LINES: gl.constexpr,
    SLOT: gl.constexpr,
    MAXIT: gl.constexpr,
    GEMMA: gl.constexpr,
    QUANT: gl.constexpr,
):
    """Program r sends row r of x. Programs tokens to tokens + P - 1 receive and
    store rows p, p + P, ... (P = programs), so at most P receivers poll. The bf16
    tensors are words of two bf16 values."""
    n_words = tokens * 2 * LINES
    pid = gl.program_id(0)
    if pid < tokens:
        _send(x, inboxes, calls, pid, n_words, RANK, LINES, SLOT)
    else:
        WORDS: gl.constexpr = gl.BlockedLayout(
            [1, 2], [32, 1], [gl.num_warps(), 1], [1, 0]
        )
        line = gl.arange(0, LINES, layout=gl.SliceLayout(1, WORDS))[:, None]
        col = 2 * line + gl.arange(0, 2, layout=gl.SliceLayout(0, WORDS))[None, :]
        w = gl.load(weight + col)
        w_lo = _lo(w)
        w_hi = _hi(w)
        if GEMMA:
            w_lo += 1.0
            w_hi += 1.0
        for row in range(pid - tokens, tokens, programs):
            word = row * 2 * LINES + col
            res = gl.load(residual + word)  # local: load it before the wait
            w0, w1, w2, w3, _, _ = _receive(
                x, inboxes, calls, err, row, n_words, RANK, LINES, SLOT, MAXIT
            )
            s_lo = _lo(w0) + _lo(w1) + _lo(w2) + _lo(w3) + _lo(res)
            s_hi = _hi(w0) + _hi(w1) + _hi(w2) + _hi(w3) + _hi(res)
            gl.store(residual_out + word, _pack(s_lo, s_hi))

            square_sum = gl.sum(gl.sum(s_lo * s_lo + s_hi * s_hi, axis=1), axis=0)
            rstd = gl.rsqrt(square_sum / (4 * LINES) + eps)
            y_lo = s_lo * rstd * w_lo
            y_hi = s_hi * rstd * w_hi
            gl.store(out + word, _pack(y_lo, y_hi))

            if QUANT:
                # A group of 128 values is 32 lines. 448 is the fp8 maximum.
                a = gl.maximum(gl.abs(y_lo), gl.abs(y_hi))
                a = gl.reshape(a, [LINES // 32, 32, 2])
                group_max = gl.max(gl.max(a, axis=2), axis=1)
                group_scale = gl.maximum(group_max, 1e-10) / 448
                s = gl.expand_dims(gl.expand_dims(group_scale, 1), 2)
                s = gl.reshape(gl.broadcast(s, a)[0], [LINES, 2])
                s = gl.convert_layout(s, WORDS, assert_trivial=True)
                gl.store(q + 2 * word, (y_lo / s).to(gl.float8e4nv))
                gl.store(q + 2 * word + 1, (y_hi / s).to(gl.float8e4nv))
                first = col % 64 == 0  # one scale for each group
                gl.store(scale + row * (LINES // 32) + col // 64, s, mask=first)
