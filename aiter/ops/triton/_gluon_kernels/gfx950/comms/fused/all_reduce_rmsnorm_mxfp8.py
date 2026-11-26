# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""All-reduce + residual add + RMSNorm + MXFP8 quant over 2, 4 or 8 gfx950 GPUs.

A bf16 reduce-scatter, then an all-gather of the bf16 sum. Rank g owns the rows g G to
g G + G - 1 (G = ceil(rows / WORLD)): CTA b < G adds the ranks of one row and pushes
the sum. The other CTAs push the rows of this rank to their owners and receive one sum.
The inboxes are Lamport slots (0xFFFF is empty), with one half for each launch parity.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.common.comms_gfx950 import (
    canon,
    dword_idx,
    fail_record,
    ld4,
    peer,
    poll_slots,
    rmsnorm_row,
    row_sum,
    st4,
    store_rmsnorm_row,
    tids,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

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
    """inp, res, res_out: [rows, HIDDEN] bf16 as int32. xq: e4m3 as int32. xs: [rows,
    HIDDEN / 32] uint8. peers: the buffers of ranks rank, rank + 1, .... RS0, AG0: the
    byte offsets of the inboxes. PMAX: the most push units of a CTA."""
    W: gl.constexpr = gl.num_warps()
    V: gl.constexpr = 2  # 16-byte vectors of a thread
    NV: gl.constexpr = HIDDEN // 8
    A: gl.constexpr = NV // V
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
        hw = row_sum(own_w, o_got, rank, WORLD, V, tid)
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
    s, y = rmsnorm_row(hw, res_w, w_w, act, eps, HIDDEN, V, GEMMA, True, W)
    store_rmsnorm_row(s, y, xq, xs, res_out, tn_c, act & has_row, A, HIDDEN, V, True, W)

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
