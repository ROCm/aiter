# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reduce-scatter + residual add + RMSNorm + all-gather over 4 GPUs in one kernel,
with a row-sharded residual. The rows are cut into G slices, one a program, and
rank q owns quarter q of each slice. A program moves its slice with TDM copies:
it sends each peer that peer's quarter of the partial sums, sums its own quarter
with the residual, normalizes it, and sends the result rows to every peer."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

HW_WORDS = 1024  # int32 words of a bf16 row of 2048
_HW = gl.constexpr(HW_WORDS)


@gluon.jit
def _unpack(w):
    """The two bf16 values of each int32 word, as fp32."""
    return (w << 16).to(gl.float32, bitcast=True), (w & -65536).to(
        gl.float32, bitcast=True
    )


@gluon.jit
def _pack(lo, hi):
    """Two fp32 values as the bf16 pair of an int32 word."""
    a = lo.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 65535
    b = hi.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) << 16
    return a | b


@gluon.jit
def _poll(ptr, offs, mask, call, err, MAXIT: gl.constexpr):
    """Waits until the flags at offs hold call. Counts a timeout in err."""
    late = 1
    it = 0
    while (late > 0) & (it < MAXIT):
        v = gl.amd.cdna5.buffer_load(ptr, offs, mask=mask, other=call, cache=".cv")
        gl.inline_asm("", "~{memory}", [], [], is_pure=False)
        late = gl.sum((v != call).to(gl.int32), axis=0)
        it += 1
    if late > 0:
        gl.atomic_add(err, 1)


@gluon.jit
def _signal(ptr, off, call, NW: gl.constexpr):
    """Writes the flag after the program's TDM stores are complete."""
    L: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    gl.amd.cdna5.buffer_store(
        gl.full([1], 0, gl.int32, L) + call,
        ptr,
        off + gl.arange(0, 1, layout=L),
        cache=".wt",
    )


_reduce_scatter_rmsnorm_all_gather_repr = make_kernel_repr(
    "_reduce_scatter_rmsnorm_all_gather_kernel", ["G", "QRM", "NW", "GEMMA"]
)


@gluon.jit(repr=_reduce_scatter_rmsnorm_all_gather_repr)
def _reduce_scatter_rmsnorm_all_gather_kernel(
    P,
    R,
    RO,
    O,
    Wt,
    nv,
    ctrl,
    err,
    b0,
    b1,
    b2,
    b3,
    RANK: gl.constexpr,
    G: gl.constexpr,
    QRM: gl.constexpr,
    NW: gl.constexpr,
    RG: gl.constexpr,
    MAXIT: gl.constexpr,
    EPS: gl.constexpr,
    GEMMA: gl.constexpr,
    RSF: gl.constexpr,
    ZF: gl.constexpr,
    INB: gl.constexpr,
    ZO: gl.constexpr,
):
    HW: gl.constexpr = _HW
    ROWL: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [1, NW], [1, 0])
    FL: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    S: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    b = gl.program_id(0)
    n = gl.load(nv)
    call = gl.load(ctrl) + 1
    bank = call % 2
    SR = gl.cdiv(gl.cdiv(n, G), 4) * 4
    QR = SR // 4
    row0 = b * SR
    boxes = (b0, b1, b2, b3)
    mine = boxes[RANK].to(gl.pointer_type(gl.int32))
    if row0 < n:
        # A TDM copy takes a whole bufs.index(q): the lowering drops .slice() offsets.
        bufs = gl.allocate_shared_memory(gl.int32, [4, QRM, HW], S)
        own = gl.minimum(gl.maximum(n - (row0 + RANK * QR), 0), QR)
        # 1 The slice of this rank's partial sums.
        for q in gl.static_range(4):
            cnt = gl.minimum(gl.maximum(n - (row0 + q * QR), 0), QR)
            if cnt > 0:
                d = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                    base=P + (row0 + q * QR) * HW,
                    shape=[cnt, HW],
                    strides=[HW, 1],
                    block_shape=[QRM, HW],
                    layout=S,
                )
                gl.amd.gfx1250.tdm.async_load(d, [0, 0], bufs.index(q))
        gl.amd.gfx1250.tdm.async_wait(0)
        gl.barrier()
        # 2 Each peer's quarter into the inbox of that peer.
        for q in gl.static_range(4):
            cnt = gl.minimum(gl.maximum(n - (row0 + q * QR), 0), QR)
            dst = boxes[q].to(gl.pointer_type(gl.int32))
            if q != RANK:  # noqa: SIM102 (a compile-time test, then a runtime one)
                if cnt > 0:
                    d = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                        base=dst + INB + ((bank * 4 + RANK) * G + b) * QRM * HW,
                        shape=[cnt, HW],
                        strides=[HW, 1],
                        block_shape=[QRM, HW],
                        layout=S,
                    )
                    gl.amd.gfx1250.tdm.async_store(d, [0, 0], bufs.index(q))
        gl.amd.gfx1250.tdm.async_wait(0)
        gl.barrier()
        for q in gl.static_range(4):
            if q != RANK:
                dst = boxes[q].to(gl.pointer_type(gl.int32))
                _signal(dst, RSF + (bank * 4 + RANK) * G + b, call, NW)
        # 3 The peers' partial sums of the own quarter. The weight and the first
        # residual rows load during the wait.
        rows = gl.arange(0, RG, layout=gl.SliceLayout(1, ROWL))[:, None]
        col = gl.arange(0, HW, layout=gl.SliceLayout(0, ROWL))[None, :]
        w_lo, w_hi = _unpack(gl.amd.cdna5.buffer_load(Wt, col))
        if GEMMA:
            w_lo = w_lo + 1.0
            w_hi = w_hi + 1.0
        gr0 = row0 + RANK * QR
        r_nx = gl.amd.cdna5.buffer_load(
            R, (gr0 + rows) * HW + col, mask=(rows < own) & (col < HW)
        )
        src4 = gl.arange(0, 4, layout=FL)
        _poll(mine, RSF + (bank * 4 + src4) * G + b, src4 != RANK, call, err, MAXIT)
        for s in gl.static_range(4):
            if s != RANK:  # noqa: SIM102 (a compile-time test, then a runtime one)
                if own > 0:
                    d = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                        base=mine + INB + ((bank * 4 + s) * G + b) * QRM * HW,
                        shape=[own, HW],
                        strides=[HW, 1],
                        block_shape=[QRM, HW],
                        layout=S,
                    )
                    gl.amd.gfx1250.tdm.async_load(
                        d, [0, 0], bufs.index(s), cache_modifier=".cv"
                    )
        gl.amd.gfx1250.tdm.async_wait(0)
        gl.barrier()
        # 4 Sum in rank order + residual, then RMSNorm, RG rows at a time.
        for g in gl.static_range(QRM // RG):
            i = g * RG + rows
            m = (i < own) & (col < HW)
            gr = gr0 + i
            r_w = r_nx
            if g + 1 < QRM // RG:
                r_nx = gl.amd.cdna5.buffer_load(
                    R, (gr + RG) * HW + col, mask=(i + RG < own) & (col < HW)
                )
            if g * RG < own:
                for s in gl.static_range(4):
                    lo, hi = _unpack(bufs.index(s).slice(g * RG, RG, dim=0).load(ROWL))
                    if s == 0:
                        a_lo = lo
                        a_hi = hi
                    else:
                        a_lo = a_lo + lo
                        a_hi = a_hi + hi
                r_lo, r_hi = _unpack(r_w)
                # The new residual of the own rows, rounded to bf16.
                y_w = _pack(r_lo + a_lo, r_hi + a_hi)
                gl.amd.cdna5.buffer_store(y_w, RO, gr * HW + col, mask=m)
                y_lo, y_hi = _unpack(y_w)
                ss = gl.sum(y_lo * y_lo + y_hi * y_hi, axis=1)
                rs = gl.rsqrt(ss / 2048.0 + EPS)
                z_w = _pack((y_lo * rs[:, None]) * w_lo, (y_hi * rs[:, None]) * w_hi)
                bufs.index(RANK).slice(g * RG, RG, dim=0).store(z_w)
        gl.barrier()
        # 5 The own result rows into the 3 peers' boxes and into the local output.
        if own > 0:
            for q in gl.static_range(4):
                if q != RANK:
                    dst = boxes[q].to(gl.pointer_type(gl.int32))
                    d = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                        base=dst + ZO + gr0 * HW,
                        shape=[own, HW],
                        strides=[HW, 1],
                        block_shape=[QRM, HW],
                        layout=S,
                    )
                    gl.amd.gfx1250.tdm.async_store(d, [0, 0], bufs.index(RANK))
            d = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=O + gr0 * HW,
                shape=[own, HW],
                strides=[HW, 1],
                block_shape=[QRM, HW],
                layout=S,
            )
            gl.amd.gfx1250.tdm.async_store(d, [0, 0], bufs.index(RANK))
        gl.amd.gfx1250.tdm.async_wait(0)
        gl.barrier()
        for q in gl.static_range(4):
            if q != RANK:
                dst = boxes[q].to(gl.pointer_type(gl.int32))
                _signal(dst, ZF + (bank * 4 + RANK) * G + b, call, NW)
        # 6 The peers' result rows of this slice: from this rank's box into the
        # local output.
        _poll(mine, ZF + (bank * 4 + src4) * G + b, src4 != RANK, call, err, MAXIT)
        for s in gl.static_range(4):
            if s != RANK:
                cnt = gl.minimum(gl.maximum(n - (row0 + s * QR), 0), QR)
                if cnt > 0:
                    d = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                        base=mine + ZO + (row0 + s * QR) * HW,
                        shape=[cnt, HW],
                        strides=[HW, 1],
                        block_shape=[QRM, HW],
                        layout=S,
                    )
                    gl.amd.gfx1250.tdm.async_load(
                        d, [0, 0], bufs.index(s), cache_modifier=".cv"
                    )
        gl.amd.gfx1250.tdm.async_wait(0)
        gl.barrier()
        for s in gl.static_range(4):
            if s != RANK:
                cnt = gl.minimum(gl.maximum(n - (row0 + s * QR), 0), QR)
                if cnt > 0:
                    d = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                        base=O + (row0 + s * QR) * HW,
                        shape=[cnt, HW],
                        strides=[HW, 1],
                        block_shape=[QRM, HW],
                        layout=S,
                    )
                    gl.amd.gfx1250.tdm.async_store(d, [0, 0], bufs.index(s))
        gl.amd.gfx1250.tdm.async_wait(0)
    # The last program to finish advances the call counter.
    gl.barrier()
    old = gl.atomic_add(ctrl + 1, 1, sem="acq_rel", scope="gpu")
    if old == G - 1:
        gl.atomic_xchg(ctrl + 1, 0, sem="relaxed", scope="gpu")
        gl.atomic_xchg(ctrl, call, sem="release", scope="gpu")
