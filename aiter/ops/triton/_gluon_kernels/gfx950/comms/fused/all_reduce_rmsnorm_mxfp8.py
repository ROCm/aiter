# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""All-reduce + residual add + RMSNorm + MXFP8 quant over 2, 4 or 8 GPUs in one kernel.

The kernel is a bf16 reduce-scatter, then an all-gather of the bf16 sum. Rank g owns
the rows g * G to g * G + G - 1, with G = ceil(rows / WORLD). CTA b < G owns a row:
it reads the rows of the peers from its inbox, adds the ranks in rank order in fp32,
rounds once to bf16 and pushes the sum to the peers. The other CTAs push the rows of
this rank to their owners, then receive one sum each. The CTA of each row adds the
residual, normalizes the row and quantizes it.

The inboxes are Lamport slots. 0xFFFF marks an empty 16-bit word: in bf16 it is a NaN,
and a sender writes the NaN 0x7FFF in its place. A slot is ready when none of its words
is 0xFFFF. The reader writes 0xFFFF back. Each launch uses the inbox half of its parity: each CTA
keeps a launch colour. Each wait has a time limit. A wait that times out writes an
error record into the buffer of each rank.

The numbers do not depend on the timing. Thread t of A = HIDDEN / 16 threads holds the
columns 8t to 8t + 7 and HIDDEN / 2 + 8t to HIDDEN / 2 + 8t + 7. It adds the squares
of its 16 values in column order (no fma). Then a butterfly adds the 64 lanes of a
wave (lane offsets 32, 16, 8, 4, 2, 1), and the waves are added in order."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import PropagateNan as _PropagateNan
from triton.language.extra.hip import libdevice

from aiter.ops.triton._gluon_kernels.common.comms_gfx950 import (
    BFLY_ADD,
    bf_hi,
    bf_lo,
    bfly_op,
    canon,
    dword_idx,
    fail_record,
    fp8_pack4,
    ld4,
    pack_bf16,
    peer,
    poll_slots,
    quad_xor,
    st2,
    st4,
    tids,
)
from aiter.ops.triton._gluon_kernels.common.utils import mx_e8m0_scale
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_NAN_ALL: gl.constexpr = gl.constexpr(_PropagateNan.ALL)
_SITE_RS: gl.constexpr = gl.constexpr(1)
_SITE_AG: gl.constexpr = gl.constexpr(2)


_all_reduce_rmsnorm_mxfp8_repr = make_kernel_repr(
    "_all_reduce_rmsnorm_mxfp8_kernel",
    ["HIDDEN", "WORLD", "GEMMA", "BLOCKS", "num_warps"],
)


@gluon.jit(
    repr=_all_reduce_rmsnorm_mxfp8_repr,
    do_not_specialize=["rows", "rank", "wait_us"],
)
def _all_reduce_rmsnorm_mxfp8_kernel(
    inp,
    res,
    norm_w,
    xq,
    xs,
    res_out,
    colors,
    peers,
    rows,
    rank,
    wait_us,
    eps,
    HIDDEN: gl.constexpr,
    WORLD: gl.constexpr,
    GEMMA: gl.constexpr,
    GMAX: gl.constexpr,
    RS0: gl.constexpr,
    AG0: gl.constexpr,
    BLOCKS: gl.constexpr,
    PMAX: gl.constexpr,
):
    """inp, res: [rows, HIDDEN] bf16 as int32 (the partial of this rank, the residual).
    norm_w: [HIDDEN] bf16 as int32. xq: [rows, HIDDEN] e4m3 as int32. xs: [rows,
    HIDDEN / 32] uint8. res_out: [rows, HIDDEN] bf16 as int32. colors: [BLOCKS] int32.
    peers: the WORLD buffer addresses of ranks rank, rank + 1, .... RS0 and AG0: byte
    offsets of the inboxes, each [2 parities][WORLD ranks][GMAX rows][HIDDEN bf16].
    PMAX: the most push units of a CTA."""
    W: gl.constexpr = gl.num_warps()
    V: gl.constexpr = 2  # 16-byte vectors of a thread
    NV: gl.constexpr = HIDDEN // 8
    A: gl.constexpr = NV // V
    NWA: gl.constexpr = (A + 63) // 64
    NP: gl.constexpr = WORLD - 1
    RSW: gl.constexpr = RS0 // 4
    AGW: gl.constexpr = AG0 // 4
    gl.static_assert(HIDDEN % 64 == 0, "HIDDEN must be a multiple of 64")
    gl.static_assert(A <= 64 * W, "too few warps for HIDDEN")
    gl.static_assert(len(peers) == WORLD, "one buffer for each rank")
    bufs = ()
    for d in gl.static_range(WORLD):
        bufs = bufs + (gl.multiple_of(peers[d].to(gl.pointer_type(gl.int32)), 16),)
    own = bufs[0]
    bid = gl.program_id(0)
    tid = tids(W)
    lane = tid % 64
    wave = tid // 64
    act = tid < A
    k4 = dword_idx(W)
    colour = gl.load(colors + bid)
    par = colour & 1
    ticks = wait_us.to(gl.int64) * 100  # s_memrealtime is 100 MHz
    G = (rows + WORLD - 1) // WORLD

    # CTA b < G owns row rank * G + b. CTA G + k receives row k // NP of owner
    # rank + 1 + k % NP.
    is_owner = bid < G
    kr = bid - G
    is_recv = (kr >= 0) & (kr < NP * G)
    kc = gl.where(is_recv, kr, 0)
    g_src = (rank + 1 + kc % NP) % WORLD
    r_recv = kc // NP
    t_n = gl.where(is_owner, rank * G + gl.minimum(bid, G - 1), g_src * G + r_recv)
    has_row = (is_owner | is_recv) & (t_n < rows)
    tn_c = gl.where(has_row, t_n, 0)

    # Loads of the row data before the pushes and the waits.
    own_w = ()
    res_w = ()
    w_w = ()
    for j in gl.static_range(V):
        vec = j * A + tid
        own_w = own_w + (
            ld4(inp, tn_c * (HIDDEN // 2) + vec * 4, act & is_owner & has_row, W),
        )
        res_w = res_w + (ld4(res, tn_c * (HIDDEN // 2) + vec * 4, act & has_row, W),)
        w_w = w_w + (ld4(norm_w, vec * 4, act & has_row, W),)

    # Reduce-scatter push. Unit u is peer d = u % NP + 1, row and vector block
    # u // NP. All loads go out before the first store.
    units = NP * G * V
    n_push = gl.minimum(BLOCKS - G, units)
    pb = bid - G
    plan = ()
    for p in gl.static_range(PMAX):
        p_u = pb + p * n_push
        p_ok = (pb >= 0) & (pb < n_push) & (p_u < units)
        p_uu = gl.where(p_ok, p_u, 0)
        p_d = p_uu % NP + 1
        p_k = p_uu // NP
        p_r = p_k // V
        p_row = ((rank + p_d) % WORLD) * G + p_r
        p_ok = p_ok & (p_row < rows)
        p_vec = (p_k % V) * A + tid
        p_x = gl.amd.cdna4.buffer_load(
            inp,
            (gl.where(p_ok, p_row, 0) * (HIDDEN // 2) + p_vec * 4)[:, None] + k4,
            mask=(act & p_ok)[:, None] & (k4 < 4),
        )
        plan = plan + ((p_ok, p_d, p_r, p_vec, p_x),)
    for p in gl.static_range(PMAX):
        p_ok, p_d, p_r, p_vec, p_x = plan[p]
        p_slot = RSW + ((par * WORLD + rank) * GMAX + p_r) * (NV * 4)
        gl.amd.cdna4.buffer_store(
            canon(p_x),
            peer(bufs, p_d, WORLD),
            (p_slot + p_vec * 4)[:, None] + k4,
            mask=(act & p_ok)[:, None] & (k4 < 4),
        )

    # The bf16 sum h: owners add the ranks and push h, receivers wait for it.
    if is_owner:
        o_addrs = ()
        for o_d in gl.static_range(1, WORLD):
            o_base = RSW + ((par * WORLD + (rank + o_d) % WORLD) * GMAX + bid) * (
                NV * 4
            )
            for o_j in gl.static_range(V):
                o_addrs = o_addrs + (own + o_base + (o_j * A + tid) * 4,)
        o_failed, o_got = poll_slots(own, o_addrs, act & has_row, ticks, NP * V)
        fail_record(bufs, rank, colour, _SITE_RS, (lane == 0) & (o_failed == 1))
        hw = ()
        for o_j in gl.static_range(V):
            for o_e in gl.static_range(4):
                acc_lo = gl.zeros_like(tid).to(gl.float32)
                acc_hi = gl.zeros_like(tid).to(gl.float32)
                for src in gl.static_range(WORLD):
                    o_x = own_w[o_j][o_e]
                    for o_d in gl.static_range(1, WORLD):
                        o_x = gl.where(
                            (rank + o_d) % WORLD == src,
                            o_got[(o_d - 1) * V + o_j][o_e],
                            o_x,
                        )
                    if src == 0:
                        acc_lo = bf_lo(o_x)
                        acc_hi = bf_hi(o_x)
                    else:
                        acc_lo = acc_lo + bf_lo(o_x)
                        acc_hi = acc_hi + bf_hi(o_x)
                hw = hw + (pack_bf16(acc_lo, acc_hi),)
        o_ag = AGW + ((par * WORLD + rank) * GMAX + bid) * (NV * 4)
        for o_j in gl.static_range(V):
            cw = (
                canon(hw[4 * o_j]),
                canon(hw[4 * o_j + 1]),
                canon(hw[4 * o_j + 2]),
                canon(hw[4 * o_j + 3]),
            )
            for o_d in gl.static_range(1, WORLD):
                st4(bufs[o_d], o_ag + (o_j * A + tid) * 4, cw, act & has_row, W)
    else:
        r_ag = AGW + ((par * WORLD + g_src) * GMAX + r_recv) * (NV * 4)
        r_addrs = ()
        for r_j in gl.static_range(V):
            r_addrs = r_addrs + (own + r_ag + (r_j * A + tid) * 4,)
        r_failed, r_got = poll_slots(own, r_addrs, act & has_row, ticks, V)
        fail_record(bufs, rank, colour, _SITE_AG, (lane == 0) & (r_failed == 1))
        hw = ()
        for r_j in gl.static_range(V):
            for r_e in gl.static_range(4):
                hw = hw + (r_got[r_j][r_e],)

    # Residual add, RMSNorm and MXFP8 quant of the row of this CTA.
    s = ()
    for j in gl.static_range(V):
        for e in gl.static_range(4):
            s = s + (
                bf_lo(hw[4 * j + e]) + bf_lo(res_w[j][e]),
                bf_hi(hw[4 * j + e]) + bf_hi(res_w[j][e]),
            )
    sq = s[0] * s[0]
    for i in gl.static_range(1, 8 * V):
        sq = sq + s[i] * s[i]
    sq = bfly_op(gl.where(act, sq, 0.0), BFLY_ADD)
    red = gl.allocate_shared_memory(
        gl.float32, [64], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    red.scatter(sq, gl.where(lane == 0, wave, 63), 0)
    gl.barrier()
    zi = gl.zeros_like(tid)
    tot = red.gather(zi, 0)
    for wv in gl.static_range(1, NWA):
        tot = tot + red.gather(zi + wv, 0)
    var = gl.div_rn(tot, gl.full(tot.shape, HIDDEN, gl.float32, tot.type.layout))
    one = gl.full(tot.shape, 1.0, gl.float32, tot.type.layout)
    rstd = gl.div_rn(one, gl.sqrt_rn(var + eps))
    xrow = tn_c * (HIDDEN // 4)
    for j in gl.static_range(V):
        vec = j * A + tid
        y = ()
        for e in gl.static_range(4):
            wl = bf_lo(w_w[j][e])
            wh = bf_hi(w_w[j][e])
            if GEMMA:
                wl = 1.0 + wl
                wh = 1.0 + wh
            yp = pack_bf16(
                (s[8 * j + 2 * e] * rstd) * wl, (s[8 * j + 2 * e + 1] * rstd) * wh
            )
            y = y + (bf_lo(yp), bf_hi(yp))
        am = gl.abs(y[0])
        for e in gl.static_range(1, 8):
            am = gl.maximum(am, gl.abs(y[e]), propagate_nan=_NAN_ALL)
        am = gl.maximum(am, quad_xor(am, 1), propagate_nan=_NAN_ALL)
        am = gl.maximum(am, quad_xor(am, 2), propagate_nan=_NAN_ALL)
        sb = mx_e8m0_scale(am, 8)
        # y / 2^(sb - 127) with ldexp: 2^-127 (sb 254) is not a normal fp32 value.
        n = 127 - sb.to(gl.int32)
        q = ()
        for e in gl.static_range(8):
            q = q + (libdevice.ldexp(y[e], n),)
        c0 = fp8_pack4(q[0], q[1], q[2], q[3])
        c1 = fp8_pack4(q[4], q[5], q[6], q[7])
        st2(xq, xrow + 2 * vec, c0, c1, act & has_row, W)
        gl.store(
            xs + tn_c * (HIDDEN // 32) + vec // 4,
            sb,
            mask=act & has_row & (lane % 4 == 0),
        )
    for j in gl.static_range(V):
        vec = j * A + tid
        rw = ()
        for e in gl.static_range(4):
            rw = rw + (pack_bf16(s[8 * j + 2 * e], s[8 * j + 2 * e + 1]),)
        st4(res_out, tn_c * (HIDDEN // 2) + vec * 4, rw, act & has_row, W)

    # Write the empty mark back into the slots of this CTA, then advance the colour.
    sent = gl.full(
        [64 * W, 4], -1, gl.int32, gl.BlockedLayout([1, 4], [64, 1], [W, 1], [1, 0])
    )
    if is_owner:
        for q_d in gl.static_range(1, WORLD):
            q_base = RSW + ((par * WORLD + (rank + q_d) % WORLD) * GMAX + bid) * (
                NV * 4
            )
            for q_j in gl.static_range(V):
                gl.amd.cdna4.buffer_store(
                    sent,
                    own,
                    (q_base + (q_j * A + tid) * 4)[:, None] + k4,
                    mask=(act & has_row)[:, None] & (k4 < 4),
                )
    else:
        q_ag = AGW + ((par * WORLD + g_src) * GMAX + r_recv) * (NV * 4)
        for q_j in gl.static_range(V):
            gl.amd.cdna4.buffer_store(
                sent,
                own,
                (q_ag + (q_j * A + tid) * 4)[:, None] + k4,
                mask=(act & has_row)[:, None] & (k4 < 4),
            )
    gl.barrier()
    gl.store(
        colors + bid + gl.zeros_like(tid),
        colour + 1 + gl.zeros_like(tid),
        mask=tid == 0,
    )
