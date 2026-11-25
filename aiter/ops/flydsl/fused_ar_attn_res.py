# SPDX-License-Identifier: MIT
# Kimi-K3 decode: the TP all-reduce and the attention-residual read, fused for small batches.
#
#   S       = A1 + ... + A8                         (Lamport one-shot over the ranks, per 16 B atom, fp32 in
#                                                    rank order, one bf16 rounding: the all-reduce's bits)
#   prefix  = bf16(residual + S)                    (the new running residual)
#   B[m,k]  = blocks bank[m, 0..nvb) and prefix     (candidates)
#   score_k = dot(B_k, cw) * rsqrt(mean(B_k^2) + score_eps)
#   w       = softmax(score)
#   mix     = sum_k w_k * B_k
#   out     = bf16(mix * rsqrt(mean(mix^2) + out_eps) * ow)
#
# Each token row is split over NWG workgroups. The epilogue comes from attlas runs on a one-GPU proxy of this op:
# the candidates' Gram matrix on the matrix core in the first pass (mean(mix^2) = w^T G w, no second pass), the
# partials of a row exchanged as epoch-tagged words instead of through a grid barrier, workgroups placed by XCD.
# AllReduceAttnRes runs it up to FUSED_MAX_TOKENS tokens and the all-reduce + attn_res_gate above that.
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import arith
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import gpu as fgpu
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, ReductionOp, T
from flydsl.extension.coop.warp.rocdl import warp_reduce

from aiter.ops.flydsl.allreduce_policy import resolve_link
from aiter.ops.flydsl.comm_collectives_host_shared import _StEngine, resolve_inbox_flags
from aiter.ops.flydsl.kernels import communication_ops_utils as C
from aiter.ops.flydsl.kernels.comm.one_shot_allreduce import (
    ATOM_BYTES,
    INBOX_BUFFERS,
    SENTINEL_FILL_BYTE,
    _canonicalize,
    _has_sentinel,
)
from aiter.ops.flydsl.kernels.comm.transport import (
    _INBOX_POLICY,
    _RECV_POLICY,
    _buffer_load,
    _buffer_ptr,
    _buffer_store,
    _color_io,
    _i32_to_bytes,
    _load_peers,
    _poll_v4i32,
    _release_inbox,
    _store_v4i32_peer,
    _to_sgpr_i64,
    atom_bf16_to_f32,
    atom_f32_to_bf16,
)
from aiter.ops.flydsl.kernels.tensor_shim import buf_base_i64
from aiter.ops.triton.fusions.attn_res import attn_res_gate

ATOM_ELEMS = 8
ATOM_I32 = ATOM_ELEMS // 2
WAVE = 64
KARG_PRELOAD = 7
POLL_L2_MAX = 2000
NREC = 0x7FFF0000  # descriptor bound: an offset past it loads 0
BIG_ATOM = 0x07FFF800  # x16 B = 0x7FFF8000 > NREC; x32 B still < 2^32


def geometry(num_tokens):
    """(workgroups per row, waves per workgroup): the best of a full 2D sweep of the TP8 kernel per token count."""
    if num_tokens < 8:
        return 7, 2
    return (
        8,
        2,
    )  # one column slice per XCD (block % 8): cw / ow slices stay in that XCD's L2


def _dpp_sum(v):
    return warp_reduce(v, ReductionOp.ADD, width=WAVE)


def _tag(f, tag):
    bits = fx.Vector.from_elements([f], fx.Float32).bitcast(fx.Int32)[0]
    return fx.Vector.from_elements([bits, tag], fx.Int32).bitcast(fx.Int64)[0]


def _store_tagged(ptr, v):
    fx.generic_store(
        ptr,
        v,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    )


def _load_tagged(ptr):
    return fx.Int64(
        fx.generic_load(
            ptr,
            volatile=True,
            memory_order=fx.AtomicOrdering.Monotonic,
            syncscope=fx.rocdl.SyncScope.AgentOneAs,
        )
    )


def _load_all(ptrs):
    return fx.Vector.from_elements([_load_tagged(p) for p in ptrs], fx.Int64)


def _load_tagged_l2(ptr):
    return fx.Int64(
        fx.generic_load(
            ptr,
            volatile=False,
            memory_order=fx.AtomicOrdering.Monotonic,
            syncscope=fx.rocdl.SyncScope.WorkgroupOneAs,
        )
    )


def _load_all_l2(ptrs):
    return fx.Vector.from_elements([_load_tagged(p) for p in ptrs], fx.Int64)


def _count_missing(v, n, epoch):
    vi = v.bitcast(fx.Int32)
    miss = fx.Int32(0)
    for k in range(n):
        miss = miss + (vi[2 * k + 1] == epoch).select(fx.Int32(0), fx.Int32(1))
    return miss


def _any_missing(v, n, epoch):
    """wave-uniform: some lane still sees a stale tag. XOR/OR on the vector unit, one ballot, a scalar branch."""
    vi = v.bitcast(fx.Int32)
    x = vi[1] ^ epoch
    for k in range(1, n):
        x = x | (vi[2 * k + 1] ^ epoch)
    b = fx.Int64(rocdl.ballot(T.i64, (x != fx.Int32(0)).ir_value()))
    return b != fx.Int64(0)


def _values(v, n):
    vf = v.bitcast(fx.Float32)
    return [vf[2 * k] for k in range(n)]


def _pin_vec(v, n):
    """v out of an empty side-effecting asm: computed here, not sunk past the barriers into the epilogue."""
    from flydsl._mlir.dialects import llvm

    return fx.Vector(
        llvm.inline_asm(
            T.vec(n, T.f32),
            [fx.Vector(v).ir_value()],
            "",
            "=v,0",
            has_side_effects=True,
        )
    )


def _from_row_above(x):
    """lanes of rows 0 and 2 get x from 16 lanes up (v_permlane16_swap: one VALU op, no LDS round trip)."""
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm

    xi = fx.Vector.from_elements([fx.Float32(x)], fx.Float32).bitcast(fx.Int32)[0]
    st = ir.Type.parse("!llvm.struct<(i32, i32)>")
    r = rocdl.permlane16_swap(st, xi.ir_value(), xi.ir_value(), False, False)
    v = fx.Int32(llvm.extractvalue(T.i32, r, [1]))
    return fx.Vector.from_elements([v], fx.Int32).bitcast(fx.Float32)[0]


def _lds_barrier():
    rocdl.s_waitcnt(lgkmcnt=0)
    rocdl.s_barrier()


def _pin_i32(v):
    from flydsl._mlir.dialects import llvm

    return fx.Int32(
        llvm.inline_asm(
            T.i32, [fx.Int32(v).ir_value()], "", "=v,0", has_side_effects=True
        )
    )


def _finish_idx(lane, N_CAND, NG, HALFD, PERM):
    """_finish's lane-only indices, computed (and pinned) before the poll instead of after it."""
    glive = lane < fx.Int32(NG)
    # slot layout: dots in lanes 0..N-1, G_ii in lanes 16 + i, the other entries in lanes N..15 and 16 + N..
    if PERM:
        gk = (lane < fx.Int32(N_CAND)).select(
            lane + fx.Int32(16),
            (lane < fx.Int32(16)).select(lane, lane + fx.Int32(N_CAND)),
        )
    else:
        gk = lane + fx.Int32(N_CAND)
    gk = glive.select(gk, fx.Int32(0))
    # lane k holds G[i, (i + d) % N] with i = k % N, d = k // N
    gi_l = lane % fx.Int32(N_CAND)
    gd_l = lane // fx.Int32(N_CAND)
    gj_l = gi_l + gd_l
    gj_l = (gj_l >= fx.Int32(N_CAND)).select(gj_l - fx.Int32(N_CAND), gj_l)
    gi_l = glive.select(gi_l, fx.Int32(0))
    gj_l = glive.select(gj_l, fx.Int32(0))
    cf = ((gd_l == fx.Int32(0)) | (gd_l == fx.Int32(HALFD))).select(
        fx.Int32(1), fx.Int32(2)
    )
    cf = glive.select(cf, fx.Int32(0))  # 0 drops the lanes past NG
    # ds_bpermute byte addresses, ready before the poll (no bounds select after it)
    return [
        _pin_i32(v)
        for v in (gk * fx.Int32(4), gi_l * fx.Int32(4), gj_l * fx.Int32(4), cf)
    ]


def _code_prefetch(lane, nlines):
    """one vector load per lane from this kernel's own code, 128 B apart: the code lines come into L2 now, so the
    instruction fetch misses later in the straight-line code hit L2 instead of HBM after the cache flush.
    """
    from flydsl._mlir.dialects import llvm

    pc = llvm.inline_asm(T.i64, [], "s_getpc_b64 $0", "=s", has_side_effects=True)
    off = (lane < fx.Int32(nlines)).select(lane * fx.Int32(128), fx.Int32(0))
    return fx.Int32(
        llvm.inline_asm(
            T.i32,
            [fx.Int32(off).ir_value(), pc],
            "global_load_dword $0, $1, $2",
            "=&v,v,s",
            has_side_effects=True,
        )
    )


def _consume(v):
    from flydsl._mlir.dialects import llvm

    llvm.inline_asm(
        None, [fx.Int32(v).ir_value()], "; keep $0", "v", has_side_effects=True
    )


def _pin_i4(v):
    """a packed bf16 atom out of an empty asm: LLVM cannot fold fpext(fptrunc(x)) into per-element cvt + shift"""
    from flydsl._mlir.dialects import llvm

    return fx.Vector(
        llvm.inline_asm(
            T.vec(4, T.i32),
            [fx.Vector(v).ir_value()],
            "",
            "=v,0",
            has_side_effects=True,
        )
    )


def _endpgm():
    from flydsl._mlir.dialects import llvm

    llvm.inline_asm(None, [], "s_endpgm", "", has_side_effects=True)


def _pin_s64(v):
    from flydsl._mlir.dialects import llvm

    return fx.Int64(
        llvm.inline_asm(
            T.i64, [fx.Int64(v).ir_value()], "", "=s,0", has_side_effects=True
        )
    )


def _pin_v64(v):
    from flydsl._mlir.dialects import llvm

    return fx.Int64(
        llvm.inline_asm(
            T.i64, [fx.Int64(v).ir_value()], "", "=v,0", has_side_effects=True
        )
    )


def _pub_idx(lane, N_CAND, NV, PERM):
    """the publisher's LDS part indices (lane-only), computed and pinned before the first barrier."""
    live = lane < fx.Int32(NV)
    isdot = lane < fx.Int32(N_CAND)
    # position -> logical index of [dots, G in circulant order]
    lg = lane
    if PERM:
        lg = (lane < fx.Int32(16)).select(
            lane + fx.Int32(N_CAND),
            (lane < fx.Int32(16 + N_CAND)).select(lane - fx.Int32(16 - N_CAND), lane),
        )
    kk = lg - fx.Int32(N_CAND)
    ci = kk % fx.Int32(N_CAND)
    cd = kk // fx.Int32(N_CAND)
    cj = ci + cd
    cj = (cj >= fx.Int32(N_CAND)).select(cj - fx.Int32(N_CAND), cj)
    i1 = isdot.select(lane * fx.Int32(16) + fx.Int32(N_CAND), ci * fx.Int32(16) + cj)
    i1 = live.select(i1, fx.Int32(0))
    i2 = isdot.select(lane * fx.Int32(16) + fx.Int32(N_CAND + 1), fx.Int32(0))
    return [_pin_i32(i1), _pin_i32(i2)]


def _bperm(v, addr):
    bits = fx.Vector.from_elements([fx.Float32(v)], fx.Float32).bitcast(fx.Int32)[0]
    r = fx.Int32(rocdl.ds_bpermute(T.i32, addr.ir_value(), bits.ir_value()))
    return fx.Vector.from_elements([r], fx.Int32).bitcast(fx.Float32)[0]


def _tree_sum(vals):
    vals = list(vals)
    while len(vals) > 1:
        nxt = [vals[i] + vals[i + 1] for i in range(0, len(vals) - 1, 2)]
        if len(vals) % 2:
            nxt.append(vals[-1])
        vals = nxt
    return vals[0]


L2E = 1.4426950408889634


def _row_sum(x):
    """sum over each row of 16 lanes into every lane of it: 4 v_add_f32_dpp"""
    from flydsl._mlir.dialects import llvm

    m = fx.Float32(x)
    for ctrl in (0xB1, 0x4E, 0x141, 0x140):
        d = rocdl.update_dpp(
            T.f32, llvm.mlir_undef(T.f32), m.ir_value(), ctrl, 0xF, 0xF, True
        )
        m = m + fx.Float32(d)
    return m


def _swap_sum(x, swap):
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm

    xi = fx.Vector.from_elements([fx.Float32(x)], fx.Float32).bitcast(fx.Int32)[0]
    st = ir.Type.parse("!llvm.struct<(i32, i32)>")
    r = swap(st, xi.ir_value(), xi.ir_value(), False, False)
    a = fx.Vector.from_elements(
        [fx.Int32(llvm.extractvalue(T.i32, r, [0]))], fx.Int32
    ).bitcast(fx.Float32)[0]
    b = fx.Vector.from_elements(
        [fx.Int32(llvm.extractvalue(T.i32, r, [1]))], fx.Int32
    ).bitcast(fx.Float32)[0]
    return a + b


def _wave_sum(x):
    """sum over the 64 lanes into every lane: row sums, then v_permlane16_swap and v_permlane32_swap"""
    return _swap_sum(
        _swap_sum(_row_sum(x), rocdl.permlane16_swap), rocdl.permlane32_swap
    )


def _row_max(x):
    """max over each row of 16 lanes: 4 DPP steps, v_max_f32 with the lane move folded in (maxnum: the scores are
    finite or -inf)"""
    m = fx.Float32(x)
    for ctrl in (
        0xB1,
        0x4E,
        0x141,
        0x140,
    ):  # quad_perm [1,0,3,2], [2,3,0,1], row_half_mirror, row_mirror
        from flydsl._mlir.dialects import llvm

        d = rocdl.update_dpp(
            T.f32, llvm.mlir_undef(T.f32), m.ir_value(), ctrl, 0xF, 0xF, True
        )
        m = fx.Float32(
            arith.maxnumf(
                m.ir_value(),
                d,
                fastmath=arith.FastMathFlags.nnan | arith.FastMathFlags.ninf,
            )
        )
    return m


def _finish(
    F, FD, lane, wbuf, scbuf, N_CAND, NG, HALFD, INV_H, PERM, idxs, SCORE_EPS, OUT_EPS
):
    """wave 0: lane l holds value l of [dot_0..dot_{N-1}, G in circulant order] in F, and lanes l < N hold G_ll in
    FD; the softmax weights times the output scale rsqrt(w^T G w / H + eps) go to wbuf. No branches.
    w^T G w = (e^T G e) / denom^2. Scores are kept in log2 units so every exp is one v_exp_f32.
    """
    gk, gi_l, gj_l, cf = idxs
    live = lane < fx.Int32(N_CAND)
    # log2(e) folded into the rsq argument: rsq(x / L2E^2) = L2E * rsq(x)
    score = F * fx.Float32(
        rocdl.rsq(
            T.f32, (FD * (INV_H / (L2E * L2E)) + SCORE_EPS / (L2E * L2E)).ir_value()
        )
    )
    score = live.select(score, fx.Float32(float("-inf")))
    fg = _bperm(F, gk)
    # the exchanges take score (ready before the max), so they overlap the max reduction instead of following it
    si = _bperm(score, gi_l)
    sj = _bperm(score, gj_l)
    m = _row_max(score)
    e = live.select(
        fx.Float32(rocdl.exp2(T.f32, (score - m).ir_value())), fx.Float32(0.0)
    )
    m0 = fx.Float32(rocdl.readlane(T.f32, m, 0))  # lanes past 16 reduced only -inf
    ei = fx.Float32(rocdl.exp2(T.f32, (si - m0).ir_value()))
    ej = fx.Float32(rocdl.exp2(T.f32, (sj - m0).ir_value()))
    denom = _row_sum(e)
    term = fx.Float32(cf) * ei * ej * fg
    qe = _wave_sum(term)
    with fx.fastmath("afn"):
        rd = fx.Float32(1.0) / denom
    sc = fx.Float32(rocdl.rsq(T.f32, (qe * rd * rd * INV_H + OUT_EPS).ir_value()))
    idx = live.select(lane, fx.Int32(N_CAND))
    fx.ptr_store(e * (rd * sc), wbuf + fx.Int64(idx))


def build(
    hidden,
    nvb,
    nwg,
    nwaves,
    num_tokens,
    world_size,
    rank,
    inbox_memory,
    score_eps,
    out_eps,
):
    SCORE_EPS = float(score_eps)
    OUT_EPS = float(out_eps)  # compile-time: part of each build
    hidden = int(hidden)
    nvb = int(nvb)
    NWG = int(nwg)
    NWAVES = int(nwaves)
    M = int(num_tokens)
    WS = int(world_size)
    self_rank = int(rank)
    push_peers = [p for p in range(WS) if p != self_rank]
    policy = _INBOX_POLICY[inbox_memory]
    payload_policy = policy["payload"]
    release_scope = policy["release"]
    A_TOTAL = hidden // ATOM_ELEMS
    BLOCK = NWAVES * WAVE
    base = A_TOTAL // NWG
    rem = A_TOTAL % NWG
    MAX_ATOMS = base + (1 if rem else 0)
    MAXIT = (MAX_ATOMS + BLOCK - 1) // BLOCK
    N_CAND = nvb + 1
    INV_H = 1.0 / hidden
    # Gram entries in circulant order: G[i, (i + d) % N] for d = 0 .. N // 2 (pairs at d = N / 2 appear twice)
    GPAIRS = [
        (k % N_CAND, (k % N_CAND + k // N_CAND) % N_CAND)
        for k in range(N_CAND * (N_CAND // 2 + 1))
    ]
    NG = len(GPAIRS)
    HALFD = N_CAND // 2 if N_CAND % 2 == 0 else -1
    NV = N_CAND + NG
    assert NV <= WAVE and N_CAND <= 16
    PERM = (
        NV >= 16 + N_CAND
    )  # G_ii in lane 16 + i: one permlane16_swap brings it beside dot_i
    NPC = NV
    TOTWG = M * NWG
    REMAP = M < 8
    GRID = (
        (((M - 1) // 8) * NWG + NWG - 1) * 8 + (M - 1) % 8 + 1
        if (REMAP and M % 8)
        else TOTWG
    )
    TMAGIC = -(-65536 // NWG)
    assert TOTWG * (TMAGIC * NWG - 65536) < 65536 and TOTWG * TMAGIC < 2**31
    NCOL = N_CAND + 2
    assert NCOL <= 16
    XROW = NCOL * ATOM_I32
    N_PART = NWAVES * 256
    ROW_I32 = hidden // 2
    RANK_I32 = M * ROW_I32
    COLOR_STRIDE = WS * (M * A_TOTAL) * ATOM_I32
    SRC_STRIDE = (M * A_TOTAL) * ATOM_I32
    data_bytes = INBOX_BUFFERS * WS * (M * A_TOTAL) * ATOM_BYTES

    def _wrap3(c):
        over = c >= fx.Int32(INBOX_BUFFERS)
        return over.select(c - fx.Int32(INBOX_BUFFERS), c)

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, N_PART, 16]
        xpose: fx.Array[fx.Int32, NWAVES * WAVE * XROW, 16]
        weight: fx.Array[fx.Float32, N_CAND + 1, 16]
        scale: fx.Array[fx.Float32, 1, 16]

    @flyc.kernel(name=f"attnres_fuse_kernel_m{M}_v{nvb}_h{hidden}")
    def attnres_fuse_kernel(
        bank_stride_t: Int32,
        bank_stride_b: Int32,
        inp_ptr: fx.Int64,
        res_ptr: fx.Int64,
        bank_ptr: fx.Int64,
        cw_ptr: fx.Int64,
        counter_ptr: fx.Int64,
        out_ptr: fx.Int64,
        ow_ptr: fx.Int64,
        gpart_ptr: fx.Int64,
        gsq_ptr: fx.Int64,
        peer_ptrs: fx.Int64,
        colors_ptr: fx.Int64,
        rank: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bidx = fx.block_idx.x
        # a row's NWG workgroups on one XCD (xcd = block % 8): block = (tgrp * NWG + g) * 8 + t % 8
        if const_expr(REMAP):
            xcd = fx.Int32(bidx) & fx.Int32(7)
            rr = fx.Int32(bidx) >> fx.Int32(3)
            # rr >= 0 and small: an exact multiply-shift instead of a signed division's fix-ups
            tgrp = (rr * fx.Int32(TMAGIC)) >> fx.Int32(16)
            g = rr - tgrp * fx.Int32(NWG)
            t = tgrp * fx.Int32(8) + xcd
            if const_expr(M % 8 != 0):  # noqa: SIM102
                if t >= fx.Int32(M):
                    _endpgm()  # a dead workgroup (no row): exits before any load
        elif const_expr(NWG == 8):
            g = fx.Int32(bidx) & fx.Int32(7)
            t = fx.Int32(bidx) >> fx.Int32(3)
        else:
            t = (fx.Int32(bidx) * fx.Int32(TMAGIC)) >> fx.Int32(16)
            g = fx.Int32(bidx) - t * fx.Int32(NWG)
        wave = tid // fx.Int32(WAVE)
        lane = tid % fx.Int32(WAVE)
        # pf = _code_prefetch(lane, 32 if nvb >= 8 else (29 if nvb >= 5 else 24))   # inside each build's code

        extra = (g < fx.Int32(rem)).select(fx.Int32(1), fx.Int32(0))
        cnt = fx.Int32(base) + extra
        gmin = (g < fx.Int32(rem)).select(g, fx.Int32(rem))
        start = g * fx.Int32(base) + gmin

        sh = fx.SharedAllocator().allocate(Shared).peek()
        part = sh.part.ptr
        xp = sh.xpose.ptr
        wbuf = sh.weight.ptr
        scbuf = sh.scale.ptr

        inp_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(inp_ptr)), T.i32, 16, NREC)
        peers = _load_peers(peer_ptrs, WS)
        peer_vec = fx.Vector.from_elements(peers, dtype=fx.Int64)
        my_inbox = _buffer_ptr(_to_sgpr_i64(peer_vec[rank]), T.i32, 16)
        _load_color, _store_color = _color_io(colors_ptr, bidx)
        color = _load_color()
        rearm_c = _wrap3(color + fx.Int32(2))

        def arslot(c, src, a):
            return (
                c * fx.Int32(COLOR_STRIDE)
                + src * fx.Int32(SRC_STRIDE)
                + a * fx.Int32(ATOM_I32)
            )

        res_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(res_ptr)), T.i32, 16, NREC)
        bank_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(bank_ptr)), T.i32, 16, NREC)
        cw_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(cw_ptr)), T.f32, 16, NREC)
        # counter_ptr carries gpart's address: the epoch is the tag (hi dword) of this workgroup's own slot, lane 0
        counter_addr = (
            buf_base_i64(counter_ptr)
            + fx.Int64(t * fx.Int32(2 * NWG) + g) * fx.Int64(NPC * 8)
            + fx.Int64(4)
        )
        if const_expr(NWG > 1):
            epoch_raw = fx.Int32(C.load_i32_global_agent(counter_addr))

        # bf16 strides are even: halve them once (a signed // 2 per row cost 12 scalar ops before each bank load)
        hstride_b = fx.Int32(bank_stride_b) >> fx.Int32(1)
        bank_half = t * (fx.Int32(bank_stride_t) >> fx.Int32(1))
        gacc = fx.Vector.from_elements([fx.Float32(0.0)] * 4, fx.Float32)
        pv_keep = []
        bank_keep = []
        col_keep = []
        valid_keep = []

        for it in range_constexpr(MAXIT):
            la = tid + fx.Int32(it * BLOCK)
            valid = la < cnt
            atom = start + la
            safe_atom = valid.select(
                atom, fx.Int32(BIG_ATOM)
            )  # invalid lanes read past num_records -> 0
            col_i32 = safe_atom * fx.Int32(ATOM_I32)
            col_f32 = safe_atom * fx.Int32(ATOM_ELEMS)
            col_keep.append(col_i32)
            valid_keep.append(valid)
            row_i32 = t * fx.Int32(ROW_I32) + col_i32

            # loads in the order their data is consumed: the parts (the all-reduce chain) first, the bank rows last,
            # since a bank row goes straight to LDS and so ends the tail after the last load lands
            # cw first: its hi/lo split then runs while the parts are still in flight, off the all-reduce's path
            cw_lo = _buffer_load(cw_buf, col_f32, 4, fx.Float32)
            cw_hi = _buffer_load(cw_buf, col_f32 + fx.Int32(4), 4, fx.Float32)
            rocdl.sched_barrier(0)
            # the all-reduce over the GPUs (Lamport one-shot): push this rank's atom to every peer now
            aglob = t * fx.Int32(A_TOTAL) + valid.select(
                atom, fx.Int32(0)
            )  # in-range for inactive lanes too
            my_inp = _canonicalize(_buffer_load(inp_buf, row_i32, ATOM_I32, fx.Int32))
            if valid:
                for peer in push_peers:
                    _store_v4i32_peer(
                        peer_vec[peer]
                        + _i32_to_bytes(arslot(color, fx.Int32(self_rank), aglob)),
                        my_inp,
                        payload_policy,
                    )
                if const_expr(release_scope is not None):
                    _release_inbox(release_scope)
            rocdl.sched_barrier(
                0
            )  # keep the issue order: the parts' chain first, the bank rows last
            res_raw = _buffer_load(res_buf, row_i32, ATOM_I32, fx.Int32)
            rocdl.sched_barrier(0)
            raw_banks = []
            for r in range_constexpr(nvb):
                raw_banks.append(
                    _buffer_load(
                        bank_buf,
                        bank_half + fx.Int32(r) * hstride_b + col_i32,
                        ATOM_I32,
                        fx.Int32,
                    )
                )
            if const_expr(it == 0):
                # the arguments past the preloaded ones are first needed here: their kernarg wait follows the loads
                rocdl.sched_barrier(0)
                ow_buf = _buffer_ptr(
                    _to_sgpr_i64(buf_base_i64(ow_ptr)), T.f32, 16, NREC
                )
                out_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(out_ptr)), T.i32, 16)
                # pinned here so its kernarg s_load (a new 64 B line) is issued now, not sunk behind the 1st barrier
                gaddr = _pin_s64(buf_base_i64(gpart_ptr))
                # lane-only index math of the reduction and _finish, done while the loads are in flight
                fidx = _finish_idx(lane, N_CAND, NG, HALFD, PERM)
                pub_idx = _pub_idx(lane, N_CAND, NV, PERM)
                # this wave's LDS staging addresses (lane-only), pinned here in the loads' shadow, not computed
                # between the bank rows' arrival and the Gram reads
                # pinned in 16 B units, so the * 4 after the pin keeps the 16 B alignment the b128 accesses need
                xbase_p = _pin_i32(
                    wave * fx.Int32(WAVE * NCOL) + lane * fx.Int32(NCOL)
                ) * fx.Int32(ATOM_I32)
                mcol_ = (lane % fx.Int32(16)) % fx.Int32(NCOL)
                rbase_p = _pin_i32(
                    wave * fx.Int32(WAVE * NCOL)
                    + (lane // fx.Int32(16)) * fx.Int32(NCOL)
                    + mcol_
                ) * fx.Int32(ATOM_I32)
                if const_expr(NWG > 1):
                    # slot element offsets of the publish and of the poll (the other NWG - 1 slots), before barrier 1
                    # a doubled ring per token row: each workgroup publishes to slot g and slot g + NWG, so the
                    # NWG - 1 peers polled from g + 1 on lie at constant offsets from one address (no wrap, no
                    # per-slot address registers: the offsets go into the loads' immediate field)
                    lane8 = _pin_i32(
                        ((lane < fx.Int32(NV)).select(lane, fx.Int32(0))) * fx.Int32(8)
                    )
                    lane8z = fx.Int64(arith.extui(T.i64, lane8.ir_value()))
                    pub_base = _pin_v64(
                        gaddr
                        + fx.Int64(
                            (t * fx.Int32(2 * NWG * NPC) + g * fx.Int32(NPC))
                            * fx.Int32(8)
                        )
                        + lane8z
                    )
            xbase = xbase_p
            rocdl.sched_barrier(0)
            cw = fx.Vector.from_elements(
                [cw_lo[e] for e in range_constexpr(4)]
                + [cw_hi[e] for e in range_constexpr(4)],
                fx.Float32,
            )
            cwh = _pin_i4(atom_f32_to_bf16(cw))
            cwl = atom_f32_to_bf16(cw - atom_bf16_to_f32(cwh))
            fx.ptr_store(cwh, xp + fx.Int64(xbase + fx.Int32((nvb + 1) * ATOM_I32)))
            fx.ptr_store(cwl, xp + fx.Int64(xbase + fx.Int32((nvb + 2) * ATOM_I32)))
            rocdl.sched_barrier(0)
            # the all-reduce: S = A1 + ... + A8 in rank order, one rounding to bf16
            addrs = [
                peer_vec[rank] + _i32_to_bytes(arslot(color, fx.Int32(src_), aglob))
                for src_ in push_peers
            ]

            def _pack(vecs):
                return fx.Vector.from_elements(
                    [v[e] for v in vecs for e in range_constexpr(ATOM_I32)], fx.Int32
                )

            got = _pack(
                [
                    _buffer_load(
                        my_inbox,
                        arslot(color, fx.Int32(src_), aglob),
                        ATOM_I32,
                        fx.Int32,
                        _RECV_POLICY,
                    )
                    for src_ in push_peers
                ]
            )
            while valid & _has_sentinel(got):
                got = _pack([_poll_v4i32(a) for a in addrs])
            recv = {
                src_: fx.Vector.from_elements(
                    [got[i * ATOM_I32 + e] for e in range_constexpr(ATOM_I32)], fx.Int32
                )
                for i, src_ in enumerate(push_peers)
            }
            s = None
            for src_ in range_constexpr(WS):
                v_ = atom_bf16_to_f32(my_inp if src_ == self_rank else recv[src_])
                s = v_ if s is None else s + v_
            # inactive lanes polled no slot of their own: zero their sum, as the out-of-range loads did, so no
            # sentinel NaN reaches the Gram the whole wave builds
            s = fx.Vector.from_elements(
                [
                    valid.select(s[e], fx.Float32(0.0))
                    for e in range_constexpr(ATOM_ELEMS)
                ],
                fx.Float32,
            )
            if valid:
                sentinel = fx.Vector.filled((ATOM_I32,), -1, fx.Int32)
                for src_ in push_peers:
                    _store_v4i32_peer(
                        peer_vec[rank]
                        + _i32_to_bytes(arslot(rearm_c, fx.Int32(src_), aglob)),
                        sentinel,
                        payload_policy,
                    )
            s_bf = atom_bf16_to_f32(_pin_i4(atom_f32_to_bf16(s)))
            res = atom_bf16_to_f32(res_raw)
            prefix_i32 = _pin_i4(atom_f32_to_bf16(res + s_bf))
            # Gram on the matrix core: X = [B_0 .. B_{N-1}, cw_hi, cw_lo] (bf16 columns, cw split in two bf16),
            # staged through this wave's LDS so lane l feeds column l % 16 of 4 atoms; C += X^T X
            fx.ptr_store(prefix_i32, xp + fx.Int64(xbase + fx.Int32(nvb * ATOM_I32)))
            rocdl.sched_barrier(0)
            for c_ in range_constexpr(nvb):
                fx.ptr_store(
                    raw_banks[c_], xp + fx.Int64(xbase + fx.Int32(c_ * ATOM_I32))
                )
            rbase = rbase_p  # columns past NCOL repeat real ones; their C is unused
            NT = WAVE // 4
            assert NT > nvb
            row_banks = []
            LA = 4
            xq = [
                fx.ptr_load(
                    xp + fx.Int64(rbase + fx.Int32(t_ * 4 * XROW)), T.vec(4, T.i32)
                )
                for t_ in range_constexpr(LA)
            ]
            rocdl.sched_barrier(0)
            if const_expr(it == 0):
                pass  # after the bank rows' vmcnt(0): the prefetch's register is free again
            for t_ in range_constexpr(NT):
                xa = fx.Vector(xq[t_]).bitcast(fx.BFloat16)
                if const_expr(t_ + LA < NT):
                    xq.append(
                        fx.ptr_load(
                            xp + fx.Int64(rbase + fx.Int32((t_ + LA) * 4 * XROW)),
                            T.vec(4, T.i32),
                        )
                    )
                gacc = rocdl.mfma_f32_16x16x32_bf16(
                    T.vec(4, T.f32), [xa, xa, gacc, 0, 0, 0]
                )
                # the epilogue's bf16 -> f32 unpacking fills the matrix chain's VALU gaps (not after the 2nd barrier)
                if const_expr(t_ < nvb):
                    row_banks.append(
                        _pin_vec(atom_bf16_to_f32(raw_banks[t_]), ATOM_ELEMS)
                    )
                if const_expr(t_ == nvb):
                    pv_keep.append(_pin_vec(atom_bf16_to_f32(prefix_i32), ATOM_ELEMS))
                rocdl.sched_barrier(0)
            bank_keep.append(row_banks)
            if valid:
                _buffer_store(out_buf, fx.Int32(RANK_I32) + row_i32, prefix_i32, 17)

        # C[row][col], row = 4 * (lane // 16) + r, col = lane % 16 (symmetric)
        gvec = fx.Vector(gacc)
        for r_ in range_constexpr(4):
            gi_ = (fx.Int32(4) * (lane // fx.Int32(16)) + fx.Int32(r_)) * fx.Int32(
                16
            ) + lane % fx.Int32(16)
            fx.ptr_store(
                fx.Float32(gvec[r_]), part + fx.Int64(wave * fx.Int32(256) + gi_)
            )
        _lds_barrier()

        # the epilogue's ow loads fly during the grid barrier
        ow_keep = []
        for it in range_constexpr(MAXIT):
            col_f32 = col_keep[it] * fx.Int32(2)
            ow_lo = _buffer_load(ow_buf, col_f32, 4, fx.Float32)
            ow_hi = _buffer_load(ow_buf, col_f32 + fx.Int32(4), 4, fx.Float32)
            ow_keep.append(
                fx.Vector.from_elements(
                    [ow_lo[e] for e in range_constexpr(4)]
                    + [ow_hi[e] for e in range_constexpr(4)],
                    fx.Float32,
                )
            )
        if wave == fx.Int32(0):
            live = lane < fx.Int32(NV)
            isdot = lane < fx.Int32(N_CAND)
            i1, i2 = pub_idx
            tot_d = fx.Float32(0.0)
            i3 = isdot.select(lane * fx.Int32(17), fx.Int32(0))
            # the waves' partials summed as a tree (dot lanes add the cw-lo column's entry)
            tot_c = _tree_sum(
                [
                    fx.Float32(part[i1 + fx.Int32(w * 256)])
                    + isdot.select(
                        fx.Float32(part[i2 + fx.Int32(w * 256)]), fx.Float32(0.0)
                    )
                    for w in range_constexpr(NWAVES)
                ]
            )
            for w in range_constexpr(NWAVES):
                if const_expr(NWG == 1):
                    tot_d = tot_d + fx.Float32(part[i3 + fx.Int32(w * 256)])
            if const_expr(NWG == 1):
                _finish(
                    tot_c,
                    tot_d,
                    lane,
                    wbuf,
                    scbuf,
                    N_CAND,
                    NG,
                    HALFD,
                    INV_H,
                    PERM,
                    fidx,
                    SCORE_EPS,
                    OUT_EPS,
                )
            else:
                epoch = epoch_raw + fx.Int32(1)
                if live:
                    pptr = C._to_ptr_global(pub_base, fx.Int64, 8)
                    tv = _tag(tot_c, epoch)
                    _store_tagged(pptr, tv)
                    _store_tagged(pptr + fx.Int64(NWG * NPC), tv)
                fsum = fx.Float32(0.0)
                dsum = fx.Float32(0.0)
                dlive = lane < fx.Int32(N_CAND)
                ptrs = []
                # the other NWG - 1 slots only: this workgroup's own partial is tot_c, already in registers
                for k_ in range_constexpr(NWG - 1):
                    ptrs.append(
                        C._to_ptr_global(pub_base, fx.Int64, 8)
                        + fx.Int64((k_ + 1) * NPC)
                    )
                pv_all = _load_all(ptrs)
                while _any_missing(pv_all, NWG - 1, epoch):
                    rocdl.s_sleep(1)
                    pv_all = _load_all(ptrs)
                fsum = _tree_sum([tot_c] + _values(pv_all, NWG - 1))
                # G_ii sits in lane N_CAND + i of the same sum
                if const_expr(PERM):
                    dsum = _from_row_above(fsum)
                else:
                    dsrc = dlive.select(lane + fx.Int32(N_CAND), fx.Int32(0))
                    dsum = fx.Float32(fgpu.shuffle_idx(fsum, dsrc, WAVE))
                _finish(
                    fsum,
                    dsum,
                    lane,
                    wbuf,
                    scbuf,
                    N_CAND,
                    NG,
                    HALFD,
                    INV_H,
                    PERM,
                    fidx,
                    SCORE_EPS,
                    OUT_EPS,
                )

        _lds_barrier()

        wp = fx.Float32(wbuf[fx.Int32(nvb)])
        mix_keep = []
        wv = [
            fx.Vector.from_elements(
                [fx.Float32(wbuf[fx.Int32(r)])] * ATOM_ELEMS, fx.Float32
            )
            for r in range(nvb)
        ]
        for it in range_constexpr(MAXIT):
            mix = pv_keep[it] * wp
            for r in range_constexpr(nvb):
                mix = fmath.fma(bank_keep[it][r], wv[r], mix)
            mix_keep.append(mix)

        for it in range_constexpr(MAXIT):
            col_f32 = col_keep[it] * fx.Int32(2)
            out_f = mix_keep[it] * ow_keep[it]
            if valid_keep[it]:
                _buffer_store(
                    out_buf,
                    t * fx.Int32(ROW_I32) + col_keep[it],
                    atom_f32_to_bf16(out_f),
                    17,
                )
        # next call uses the next inbox colour (the one re-armed two calls ago)
        if tid == fx.Int32(0):
            _store_color(_wrap3(color + fx.Int32(1)))

    @flyc.jit
    def launch(
        bank_stride_t: Int32,
        bank_stride_b: Int32,
        inp_ptr: fx.Int64,
        res_ptr: fx.Int64,
        bank_ptr: fx.Int64,
        cw_ptr: fx.Int64,
        counter_ptr: fx.Int64,
        out_ptr: fx.Int64,
        ow_ptr: fx.Int64,
        gpart_ptr: fx.Int64,
        gsq_ptr: fx.Int64,
        peer_ptrs: fx.Int64,
        colors_ptr: fx.Int64,
        rank: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        # plain addresses, the prologue's first: the dispatcher preloads them into SGPRs (no kernarg s_load wait)
        attnres_fuse_kernel(
            bank_stride_t,
            bank_stride_b,
            inp_ptr,
            res_ptr,
            bank_ptr,
            cw_ptr,
            counter_ptr,
            out_ptr,
            ow_ptr,
            gpart_ptr,
            gsq_ptr,
            peer_ptrs,
            colors_ptr,
            rank,
        ).launch(grid=(fx.Int64(GRID), 1, 1), block=(BLOCK, 1, 1), stream=stream)

    launch.compile_hints = {
        **launch.compile_hints,
        "llvm_options": {"amdgpu-kernarg-preload-count": KARG_PRELOAD},
    }
    return {
        "launch": launch,
        "flags_bytes": 0,
        "data_bytes": data_bytes,
        "lds_bytes": 0,
        "tile_bytes": ATOM_BYTES,
        "wire_tile_bytes": ATOM_BYTES,
        "super_tile": 1,
        "rank_tile_bytes": ATOM_BYTES,
        "tile_fp16": ATOM_BYTES // 2,
        "atoms": 1,
        "world_size": WS,
        "inbox_memory": inbox_memory,
        "fanout": "peer",
        "rank": self_rank,
        "inbox_fill": SENTINEL_FILL_BYTE,
        "grid": GRID,
        "block": BLOCK,
    }


# Largest batch the fused kernel serves. Decode mini-e2e on TP8 MI355X (GEMM, op, GEMM in a CUDA graph, H 7168,
# nvb 8): the fused kernel beats the all-reduce + read of aiter main and of SGLang at every M up to 28 (by 0.6 to 9.7
# us) and loses from M = 32 on (by 0.5 us at 32, 26 us at 128), where its one-shot all-reduce falls behind the lamport
# two-shot.
FUSED_MAX_TOKENS = 28
MAX_NVB = 8


class _FusedRows:
    """The fused kernel for one batch size M: one IPC inbox (a collective to create), a compiled launch and an
    epoch-tagged scratch per nvb."""

    def __init__(
        self,
        *,
        group,
        device,
        rank,
        world_size,
        hidden,
        num_tokens,
        inbox_memory,
        link,
        eps,
    ):
        self.rank, self.world_size, self.hidden, self.M = (
            int(rank),
            int(world_size),
            int(hidden),
            int(num_tokens),
        )
        self.device = device
        self.nwg, self.nwaves = geometry(self.M)
        self.inbox_flags, self.resolved = resolve_inbox_flags(
            inbox_memory, world_size, resolve_link(link)
        )
        spec = self._build(MAX_NVB, eps)
        self._eng = _StEngine(
            spec=spec,
            group=group,
            rank=self.rank,
            world_size=self.world_size,
            inbox_flags=self.inbox_flags,
        )
        self._by_key = {(MAX_NVB, eps): self._prepare(spec, MAX_NVB)}

    def _build(self, nvb, eps):
        return build(
            self.hidden,
            nvb,
            self.nwg,
            self.nwaves,
            self.M,
            self.world_size,
            self.rank,
            self.resolved,
            *eps,
        )

    def _prepare(self, spec, nvb):
        n = nvb + 1
        gpart = torch.zeros(
            2 * self.M * self.nwg * (n * (n // 2 + 2)),
            dtype=torch.int64,
            device=self.device,
        )
        gsq = torch.zeros(self.M * self.nwg, dtype=torch.float32, device=self.device)
        return spec["launch"], gpart, gsq

    def run(self, inp, residual, bank, nvb, cw, ow, out2, eps):
        if (entry := self._by_key.get((nvb, eps))) is None:
            self._by_key[(nvb, eps)] = entry = self._prepare(self._build(nvb, eps), nvb)
        launch, gpart, gsq = entry
        launch(
            bank.stride(0),
            bank.stride(1),
            inp.data_ptr(),
            residual.data_ptr(),
            bank.data_ptr(),
            cw.data_ptr(),
            gpart.data_ptr(),
            out2.data_ptr(),
            ow.data_ptr(),
            gpart.data_ptr(),
            gsq.data_ptr(),
            int(self._eng._gpu_peer_ptrs),
            int(self._eng._colors),
            self.rank,
            fx.Stream(torch.cuda.current_stream().cuda_stream),
        )

    def close(self):
        self._eng.close()


class AllReduceAttnRes:
    """The decode all-reduce of a TP group followed by the attention-residual read, as one call.

    ``__call__(inp, residual, bank, nvb, cw, ow)`` takes this rank's partial ``inp`` [M, H] bf16, the running
    residual [M, H] bf16, the blocks ``bank`` [M, NB, H] bf16 (rows ``0..nvb`` valid; pass exactly the nvb
    valid rows, packed, to keep the unfused path fast), and the fp32 [H] score
    weight ``cw`` (score projection times score-norm weight) and output-norm weight ``ow``. It returns
    ``(out, prefix)``: the normed read and the new residual ``residual + all_reduce(inp)``.

    Up to ``FUSED_MAX_TOKENS`` tokens (and H % 512 == 0, 1 <= nvb <= 8) it runs the fused kernel, which keeps the
    sum in registers; above that, ``all_reduce(inp)`` and ``attn_res_gate``. The first fused call at a new M sets up
    that M's IPC inbox, a collective: every rank must make it, outside graph capture (a warm-up pass does).
    """

    def __init__(
        self,
        *,
        group,
        device,
        rank,
        world_size,
        hidden,
        all_reduce,
        inbox_memory="uncached",
        link=None,
        max_fused_tokens=FUSED_MAX_TOKENS,
    ):
        self.group, self.device, self.rank, self.world_size = (
            group,
            device,
            int(rank),
            int(world_size),
        )
        self.hidden, self.all_reduce = int(hidden), all_reduce
        self.inbox_memory, self.link = inbox_memory, link
        self.max_fused_tokens = int(max_fused_tokens)
        self._rows = {}

    def fuses(self, num_tokens, nvb):
        """Whether a call with these shapes runs the fused kernel (the same answer on every rank)."""
        return (
            1 <= num_tokens <= self.max_fused_tokens
            and 1 <= nvb <= MAX_NVB
            and self.hidden % 512 == 0
        )

    def __call__(self, inp, residual, bank, nvb, cw, ow, score_eps=1e-6, out_eps=1e-6):
        m = inp.shape[0]
        if not self.fuses(m, nvb):
            # attn_res_gate reads packed candidate rows: a strided [:, :nvb] view of a larger buffer costs it about
            # 5 us at M = 32 (H 7168), so the caller's nvb valid blocks are passed as they are when they fit
            blocks = bank if bank.shape[1] == nvb else bank[:, :nvb]
            return attn_res_gate(
                residual,
                blocks,
                cw,
                score_eps,
                add_hidden=self.all_reduce(inp),
                output_rms_weight=ow,
                output_rms_eps=out_eps,
            )
        if (rows := self._rows.get(m)) is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"AllReduceAttnRes: M={m} was not warmed up before graph capture"
                )
            rows = self._rows[m] = _FusedRows(
                group=self.group,
                device=self.device,
                rank=self.rank,
                world_size=self.world_size,
                hidden=self.hidden,
                num_tokens=m,
                inbox_memory=self.inbox_memory,
                link=self.link,
                eps=(float(score_eps), float(out_eps)),
            )
        out2 = torch.empty((2, m, self.hidden), dtype=inp.dtype, device=inp.device)
        rows.run(
            inp, residual, bank, nvb, cw, ow, out2, (float(score_eps), float(out_eps))
        )
        return out2[0], out2[1]

    def close(self):
        for rows in self._rows.values():
            rows.close()
        self._rows.clear()
