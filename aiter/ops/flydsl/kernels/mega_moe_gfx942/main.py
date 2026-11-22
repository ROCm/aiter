# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The main path's mega kernel (gfx942): an EP8 a8w4 MoE layer as one persistent kernel
that dispatches the tokens (P2P), runs gemm1 and gemm2 interleaved, and adds every route
into its source rank's output with BF16 atomics (the combine).

The GEMM tiles are ``kloop._make_core``'s (fd2 mxfp4 W streamed into VGPRs, fp8 A
through LDS). ``comm.py`` (quant, prep, plan, finish) runs around it.

Rows: every rank receives its routes sorted by local expert, each expert padded to
``BLOCK_M`` (128) rows = one m-block, the sources' routes interleaved inside an expert.
Producers write a route's fp8 x row + its 1x128 scales (QX 0: its row scale), (source rank,
token, slot) and weight straight into the destination's row (symmetric MORI heap), then add
1 to the destination's ``arrived[block]``.

Workgroups (one per CU, 512 threads, 64 KB LDS) take tickets from 8 per-XCD queues (a
contiguous share of the blocks each; an empty queue sends a workgroup to the next ones)
until all are empty. Each queue holds:
  [0, producers / 8)  producers (send this rank's routes in the order the destinations'
                      queues need them, as the plan kernel sorted them);
  then                groups of 2 blocks (both read each W tile at about the same time):
                      the gemm1 tiles of group u and the gemm2 tiles of group u - lag
                      (the combine traffic spreads over the whole kernel).
gemm1(b, n): waits until arrived[b] = the routes block b receives; 128 rows x 128 inter
cols (gate + up = 256 W rows); A = the received fp8 rows, rescaled per 128-K group (XG;
QX 0: one scale per row, applied in the epilogue); SiLU / SiTUv2; fp8 a2 + one f32 scale
per (row, 128 cols), written through; done[b] += 1.
gemm2(b, n): waits until done[b] = I / 128; 128 rows x 256 out cols; a2 rescaled per
128-K group; times the route weight; BF16 atomics into out[token] of the source rank.
A task only waits for producers (which never wait) or for lower tickets of its queue
(held by running workgroups whose tasks wait for nothing more), so the grid needs no
co-residency.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..act import gate_up_act, situ_params
from .atomic import _atomic_add_at, _atomic_load_at, _inv_l1, _xcc_id
from .cvt import _channel_factor, _fp8x4_pack
from .hw import (
    CM_NT,
    CM_SC1,
    CM_SYS,
    LDS_BYTES_MAX,
    _opaque_tx,
    _readlane,
    _sleep,
    _u,
)
from .kloop import _dispatch_rows, _make_core
from .mem import (
    _buffer_i32_scalar_read,
    _global_base_ptr1,
    _global_i32_at,
    _global_i32_buffer_tiles,
    _ld,
    _lds_f32_store_masked,
    _raw,
    _rsrc,
    _store,
    buffer_store,
    get_element_ptr,
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
)
from .protocol import (
    ARRIVED,
    BLOCK_M,
    E_LOCAL,
    NPES,
    QSTRIDE,
    QUEUES,
    ROW_BITS,
    SCHED_EXPECTED,
)

CHUNK = 8  # routes per producer wave between two signals
P2P_AUX = (
    CM_SYS | CM_NT
)  # stores written through to the destination (vmcnt covers visibility)
WT_AUX = CM_SC1  # a2 written through (read by gemm2 tasks on any XCD)
ARB = 128  # A bytes per row per K tile (one 128-K group)


def ep_mega_kernel(
    *,
    H,
    I,
    TOPK,
    token_shift,
    producers=16,
    lag=2,
    ROWV=(32, 64, 128),
    NI=128,
    NO=256,
    NW=8,
    PF=2,
    QX=1,
    ACT="silu",
    BCM1=2,
    BCM2=2,
    WQ="fp4",
):
    """The mega kernel for (H, I, top-k); ``token_shift``: bits of the source
    token in a row's down id (max tokens per rank <= 1 << token_shift). ``QX``: x scales
    (``mega_quant_kernel``), 1 one per (row, 128 cols), 0 one per row. ``BCM1`` /
    ``BCM2``: gemm1 / gemm2 W load cache modifier (2 = nt). ``WQ``: "fp4" fd2 mxfp4 W
    (a8w4), "fp8" fp8 W in aiter's (16, 16) preshuffle with f32 128x128 block scales
    (a8w8)."""
    assert ACT in ("silu", "silu_clamped", "situv2") and QX in (0, 1), (ACT, QX)
    assert WQ in ("fp4", "fp8"), WQ
    FP8W = WQ == "fp8"
    BM = BLOCK_M
    NT = 64 * NW
    N2 = 2 * I
    NG, ND = I // NI, H // NO  # gemm1 / gemm2 tiles per block
    UNIT = NG + ND
    PAIR = 2
    QPROD = producers // QUEUES
    G1N, G2N = H // 128, I // 128  # 128-K scale groups of gemm1 / gemm2
    RATIO = (
        2 * BM * ARB
    )  # bytes: f32 ratio tables [G][BM] + fin [BM] after the A buffers
    # fp8 W: per 128-col W block of a task, the ratios of its 128-K group scales and its
    # last scale (``build_b_ratios``), after the A ratio tables: gemm1 has the gate and
    # the up block of each 128 inter cols, gemm2 one block per 128 out cols.
    NB1, NB2 = 2 * (NI // 128), NO // 128
    BT1, BT2 = RATIO + (G1N + 1) * BM * 4, RATIO + (G2N + 1) * BM * 4
    W1S_BYTES = E_LOCAL * (N2 // 128) * G1N * 4  # f32 [E_LOCAL, 2I / 128, H / 128]
    W2S_BYTES = E_LOCAL * (H // 128) * G2N * 4  # f32 [E_LOCAL, H / 128, I / 128]
    LDS_BYTES = max(
        BT1 + (NB1 * (G1N + 1) * 4 if FP8W else 0),
        BT2 + (NB2 * (G2N + 1) * 4 if FP8W else 0),
        BM * NO * 2,
    )
    NIW, NOW = NI // NW, NO // NW
    NC1, NC2 = NIW // 16, NOW // 16
    RPS = NT // 32  # gemm2 scatter rows per pass (32 lanes x 2 cols per row)
    TOPK_SHIFT = TOPK.bit_length() - 1
    SCALE_LANES = H // 128 * 4 // 16  # 16 B lanes of a row's x scales
    TOKEN_MASK = (1 << token_shift) - 1
    ROW_MASK = (1 << ROW_BITS) - 1
    assert 1 << TOPK_SHIFT == TOPK and TOPK <= 8 and token_shift + 3 <= 24
    assert H % 1024 == 0 and I % NI == 0 and H % NO == 0 and NO % 64 == 0
    assert NI % 128 == 0 and 128 % NIW == 0 and NIW % 16 == 0 and NOW % 16 == 0
    assert (
        producers % QUEUES == 0 and ROWV[-1] == BM and LDS_BYTES <= LDS_BYTES_MAX
    ), LDS_BYTES
    # build_b_ratios: a thread per (W block, 128-K group) of a task; a gemm2 wave's
    # columns lie in one 128-col W block
    assert not FP8W or (
        max(NB1 * G1N, NB2 * G2N) <= NT and 128 % NOW == 0 and 128 % NIW == 0
    )
    GPW = NI // 128  # gemm1 epilogue: quant groups per WG row, waves per group
    WPG = NW // GPW
    kname = (
        f"a8w{8 if FP8W else 4}_megamoe_ep_h{H}_i{I}_k{TOPK}_t{token_shift}"
        f"_p{producers}_l{lag}_r{'x'.join(map(str, ROWV))}_ni{NI}_no{NO}_pf{PF}"
        + ("" if (BCM1, BCM2) == (2, 2) else f"_bcm{BCM1}{BCM2}")
        + ("" if QX else "_xrow")
        + ("" if ACT == "silu" else f"_{ACT}")
    )

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    N64 = 27  # i64 kernel args (in kernel() order), then 4 i32 and 5 f32
    NAMES64 = (
        "x8",
        "xs",
        "tw",
        "route",
        "info",
        "tab_a",
        "tab_s",
        "tab_id",
        "tab_w",
        "tab_ws",
        "tab_out",
        "ra",
        "rs",
        "rid",
        "rw",
        "ws",
        "w1",
        "w1s",
        "w1e",
        "w2",
        "w2s",
        "w2e",
        "a2",
        "a2s",
        "eids",
        "nv",
        "sched",
    )
    NAMES32 = ("routes", "tokens", "me", "nblk")
    NAMESF = ("sb", "sbr", "lb", "lbr", "lim")
    assert len(NAMES64) == N64

    @flyc.kernel(name=kname, known_block_size=[NT, 1, 1])
    def kernel(
        x8: fx.Int64,
        xs: fx.Int64,
        tw: fx.Int64,
        route: fx.Int64,
        info: fx.Int64,
        tab_a: fx.Int64,
        tab_s: fx.Int64,
        tab_id: fx.Int64,
        tab_w: fx.Int64,
        tab_ws: fx.Int64,
        tab_out: fx.Int64,
        ra: fx.Int64,
        rs: fx.Int64,
        rid: fx.Int64,
        rw: fx.Int64,
        ws: fx.Int64,
        w1: fx.Int64,
        w1s: fx.Int64,
        w1e: fx.Int64,
        w2: fx.Int64,
        w2s: fx.Int64,
        w2e: fx.Int64,
        a2: fx.Int64,
        a2s: fx.Int64,
        eids: fx.Int64,
        nv: fx.Int64,
        sched: fx.Int64,
        routes: fx.Int32,
        tokens: fx.Int32,
        me: fx.Int32,
        nblk: fx.Int32,
        sb: fx.Float32,
        sbr: fx.Float32,
        lb: fx.Float32,
        lbr: fx.Float32,
        lim: fx.Float32,
    ):
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        lds_i32 = fx.Int32(fx.ptrtoint(lds_ptr))
        lds = lds_typed_ptr(lds_i32, T.i32)
        lds_f32 = lds_typed_ptr(lds_i32, T.f32)
        tx0 = fx.Int32(gpu.thread_id("x"))
        lane0 = tx0 % fx.Int32(64)
        wave0 = _u(tx0 // fx.Int32(64))
        sched0 = fx.Int64(sched)
        nblk0 = nblk
        blocks = fx.min(
            (_u(_global_i32_at(nv, fx.Int32(0))) + fx.Int32(BM - 1)) // fx.Int32(BM),
            nblk0,
        )

        def queue_layout(q):
            """(first block, blocks) of queue q: a contiguous eighth of the blocks."""
            first = blocks * q // fx.Int32(QUEUES)
            return first, blocks * (q + fx.Int32(1)) // fx.Int32(QUEUES) - first

        def queue_tasks(q):
            own = queue_layout(q)[1]
            return fx.Int32(QPROD) + fx.Int32(UNIT * PAIR) * (
                (own + fx.Int32(PAIR - 1)) // fx.Int32(PAIR)
            )

        def lane0_add(addr, value):
            """Atomic add by lane 0 only (old value in lane 0). The address goes through an
            opaque VGPR zero: at a uniform address LLVM's atomic optimizer would sum the
            lanes in a 64-step loop."""
            taken = fx.Int32(0)
            if lane0 == fx.Int32(0):
                # Raw asm: FlyDSL 0.5.0 has no stable opaque value. Without it the
                # atomic gains an mbcnt / bcnt / readfirstlane lane reduction (+10 of 24
                # instructions in a gfx942 compile-only probe).
                zero = fx.Int32(
                    llvm.inline_asm(
                        T.i32, [], "v_mov_b32 $0, 0", "=v", has_side_effects=True
                    )
                )
                taken = _atomic_add_at(
                    fx.Int64(addr) + fx.Int64(zero), value, "agent-one-as"
                )
            return taken

        def take_ticket(after_task):
            """(queue, ticket) of this workgroup's next task; ticket -1 when all queues are
            empty. Plain loads find the queues that may still hold tasks first (an atomic
            per leftover workgroup slows the tasks still running)."""
            if const_expr(after_task):
                gpu.barrier()  # the previous task's LDS reads are done
            if wave0 == fx.Int32(0):
                home = _xcc_id()
                open_mask = fx.Int32(0)
                for q in range_constexpr(QUEUES):
                    seen = _u(
                        _atomic_load_at(
                            sched0 + fx.Int64(q * QSTRIDE * 4), "agent-one-as"
                        )
                    )
                    open_mask = open_mask | (seen < queue_tasks(fx.Int32(q))).select(
                        fx.Int32(1 << q), fx.Int32(0)
                    )
                got = fx.Int32(-1)
                cand = home
                tries = fx.Int32(0)
                while (got < fx.Int32(0)) & (tries < fx.Int32(QUEUES)):
                    cand = (home + tries) & fx.Int32(QUEUES - 1)
                    if ((open_mask >> cand) & fx.Int32(1)) != fx.Int32(0):
                        taken = _u(
                            lane0_add(
                                sched0 + fx.Int64(cand * fx.Int32(QSTRIDE * 4)), 1
                            )
                        )
                        got = (taken < queue_tasks(cand)).select(taken, fx.Int32(-1))
                    tries = tries + fx.Int32(1)
                _store(lds, fx.Int32(0), cand)
                _store(lds, fx.Int32(1), got)
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            queue = _u(lds[fx.Int32(0)])
            ticket = _u(lds[fx.Int32(1)])
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            return queue, ticket

        def reload():
            """Kernel args read again from the kernarg segment through an opaque base: with
            loop-invariant args LLVM hoists every task kind's address math out of the ticket
            loop and runs out of registers."""
            # Raw intrinsic, asm and loads: FlyDSL 0.5.0 has no stable way to re-read a
            # kernel argument. Passing the args themselves raises the SGPR spills from 9 to
            # 151 and the VGPRs from 239 to 250 (gfx942 compile-only, every config).
            ptr4 = ir.Type.parse("!llvm.ptr<4>")
            seg = llvm.call_intrinsic(
                ptr4, "llvm.amdgcn.kernarg.segment.ptr", [], [], []
            )
            base = fx.Int64(
                llvm.inline_asm(
                    T.i64,
                    [llvm.ptrtoint(T.i64, seg)],
                    "s_mov_b64 $0, $1",
                    "=s,s",
                    has_side_effects=True,
                )
            )

            def load(off, ty, align):
                return llvm.LoadOp(
                    ty,
                    llvm.inttoptr(ptr4, (base + fx.Int64(off)).ir_value()),
                    alignment=align,
                ).result

            a = {n: fx.Int64(load(8 * i, T.i64, 8)) for i, n in enumerate(NAMES64)}
            a.update(
                {
                    n: fx.Int32(load(8 * N64 + 4 * j, T.i32, 4))
                    for j, n in enumerate(NAMES32)
                }
            )
            a.update(
                {
                    n: fx.Float32(load(8 * N64 + 4 * (len(NAMES32) + j), T.f32, 4))
                    for j, n in enumerate(NAMESF)
                }
            )
            return a

        def spin_ge(addr, target, scope):
            """wave 0: poll until the i32 at addr >= target, then invalidate the L1."""
            seen = _atomic_load_at(addr, scope)
            while seen < target:
                _sleep(2)
                seen = _atomic_load_at(addr, scope)
            _inv_l1()

        def build_b_ratios(scales, nbytes, rows, G, table, tx):
            """fp8 W: LDS ``table`` (bytes) [len(rows)][G + 1] f32 of a task's 128-col W
            blocks: s[g - 1] / s[g] for each 128-K group g (1 for g = 0), then s[G - 1],
            from the f32 block scales at ``scales`` (``nbytes`` of them; ``rows``: each
            block's scale row of G). The K loop scales acc by them at every group start
            (with x's ratios), so acc ends in units of s[G - 1]. A zero scale acts as 1
            (its W block is all zero, see weights.prepare_weights_a8w8). Called after
            pre_a's barrier (before it, the previous task's scatter may still read this
            LDS); visible after the K loop's first barrier."""
            ok = tx < fx.Int32(len(rows) * G)
            idx = ok.select(tx, fx.Int32(0))
            t = idx // fx.Int32(G)
            g = idx % fx.Int32(G)
            row = rows[0]
            for j in range_constexpr(1, len(rows)):
                row = (t == fx.Int32(j)).select(rows[j], row)
            # bounded: a masked lane's offset is 0x7FFFFFFF
            rs = _rsrc(scales, nbytes)
            one = fx.Float32(1.0)
            cur = fx.Int32(_ld(rs, row * fx.Int32(G) + g, mask=ok)).bitcast(fx.Float32)
            prv = fx.Int32(
                _ld(
                    rs,
                    row * fx.Int32(G) + fx.max(g - fx.Int32(1), fx.Int32(0)),
                    mask=ok,
                )
            ).bitcast(fx.Float32)
            cur = (cur == fx.Float32(0.0)).select(one, cur)
            prv = (prv == fx.Float32(0.0)).select(one, prv)
            base = fx.Int32(table // 4) + t * fx.Int32(G + 1)
            _lds_f32_store_masked(
                lds_f32, base + g, (g == fx.Int32(0)).select(one, prv / cur), ok
            )
            _lds_f32_store_masked(
                lds_f32, base + fx.Int32(G), cur, ok & (g == fx.Int32(G - 1))
            )

        # ------------------------------------------------------------------ producer
        def produce(a, index, lane, wave):
            """Wave j of producer ``index`` sends the routes the plan kernel sorted into slice
            8 index + j, in order, CHUNK at a time: all loads of a chunk, then the P2P
            stores, a vmcnt wait, and one arrived[block] increment per route."""
            rows_cap = a["nblk"] * fx.Int32(BM)
            n_routes = a["routes"]
            x_rs = _rsrc(a["x8"], fx.Int64(a["tokens"]) * fx.Int64(H))
            xs_rs = _rsrc(
                a["xs"], fx.Int64(a["tokens"]) * fx.Int64(SCALE_LANES * 16 if QX else 4)
            )
            w_rs = _rsrc(a["tw"], fx.Int64(n_routes) * fx.Int64(4))
            route_rs = _rsrc(a["route"], fx.Int64(n_routes) * fx.Int64(4))
            info_rs = _rsrc(a["info"], fx.Int64(n_routes) * fx.Int64(4))
            scale_lane = lane < fx.Int32(SCALE_LANES if QX else 1)
            # lane r < NPES (and every lane, mod NPES): (lo, hi) of rank r's buffer bases
            peer = (lane & fx.Int32(NPES - 1)) * fx.Int32(2)
            bases = []
            for t in ("tab_a", "tab_s", "tab_id", "tab_w", "tab_ws"):
                t_rs = _rsrc(a[t], NPES * 8)
                bases.append(
                    (fx.Int32(_ld(t_rs, peer)), fx.Int32(_ld(t_rs, peer + fx.Int32(1))))
                )

            def lane_value(values, j):
                return fx.Int32(fx.gpu.shuffle_idx(fx.Int32(values), fx.Int32(j), 64))

            def peer_rsrc(which, rank, nbytes):
                lo, hi = [fx.Uint32(_readlane(h, rank)) for h in bases[which]]
                return _rsrc(
                    fx.Int64((fx.Uint64(hi) << fx.Uint64(32)) | fx.Uint64(lo)), nbytes
                )

            per_slice = (n_routes + fx.Int32(producers * 8 - 1)) // fx.Int32(
                producers * 8
            )
            start = (index * fx.Int32(8) + wave) * per_slice
            stop = fx.min(start + per_slice, n_routes)
            nxt = start
            while nxt < stop:
                pos = nxt + lane
                mine = (lane < fx.Int32(CHUNK)) & (pos < stop)
                rts = fx.Int32(_ld(route_rs, pos, mask=mine))
                infos = fx.Int32(_ld(info_rs, pos, mask=mine))
                send = mine & (infos >= fx.Int32(0))
                sends = send.select(fx.Int32(1), fx.Int32(0))
                # all loads of the chunk before any store (a store may alias a later load
                # for all LLVM knows)
                pending = []
                for j in range_constexpr(CHUNK):
                    rt = lane_value(rts, j)
                    ok = lane_value(sends, j) != fx.Int32(0)
                    token = rt >> fx.Int32(TOPK_SHIFT)
                    data = [
                        _ld(
                            x_rs,
                            lane * fx.Int32(4)
                            + fx.Int32(part * 256)
                            + token * fx.Int32(H // 4),
                            4,
                            ok,
                        )
                        for part in range_constexpr(H // 1024)
                    ]
                    if const_expr(QX):
                        scales = _ld(
                            xs_rs,
                            lane * fx.Int32(4) + token * fx.Int32(SCALE_LANES * 4),
                            4,
                            ok & scale_lane,
                        )
                    else:
                        scales = _ld(xs_rs, token, 1, ok & scale_lane)
                    weight = _ld(w_rs, rt, 1, ok)
                    pending.append(
                        (rt, lane_value(infos, j), ok, token, data, scales, weight)
                    )
                for rt, inf, ok, token, data, scales, weight in pending:
                    row = inf & fx.Int32(ROW_MASK)
                    dest = ok.select(inf >> fx.Int32(ROW_BITS + 5), fx.Int32(0))
                    a_dst = peer_rsrc(0, dest, fx.Int64(rows_cap) * fx.Int64(H))
                    for part in range_constexpr(H // 1024):
                        buffer_store(
                            data[part],
                            a_dst,
                            lane * fx.Int32(4)
                            + fx.Int32(part * 256)
                            + row * fx.Int32(H // 4),
                            mask=ok,
                            cache_modifier=P2P_AUX,
                        )
                    buffer_store(
                        scales,
                        peer_rsrc(
                            1, dest, fx.Int64(rows_cap) * fx.Int64(SCALE_LANES * 16)
                        ),
                        (lane * fx.Int32(4) if QX else fx.Int32(0))
                        + row * fx.Int32(SCALE_LANES * 4),
                        mask=ok & scale_lane,
                        cache_modifier=P2P_AUX,
                    )
                    first = ok & (lane == fx.Int32(0))
                    down_id = ((a["me"] << fx.Int32(token_shift)) | token) | (
                        (rt & fx.Int32(TOPK - 1)) << fx.Int32(24)
                    )
                    buffer_store(
                        _raw(down_id),
                        peer_rsrc(2, dest, fx.Int64(rows_cap) * fx.Int64(4)),
                        row,
                        mask=first,
                        cache_modifier=P2P_AUX,
                    )
                    buffer_store(
                        weight,
                        peer_rsrc(3, dest, fx.Int64(rows_cap) * fx.Int64(4)),
                        row,
                        mask=first,
                        cache_modifier=P2P_AUX,
                    )
                rocdl.s_waitcnt(vmcnt=0)
                src = (infos >> fx.Int32(ROW_BITS + 5)) & fx.Int32(NPES - 1)
                ws_lo, ws_hi = [
                    fx.Uint32(fx.gpu.shuffle_idx(h, src, 64)) for h in bases[4]
                ]
                blk = (infos & fx.Int32(ROW_MASK)) // fx.Int32(BM)
                arrived = fx.Int64(
                    (fx.Uint64(ws_hi) << fx.Uint64(32)) | fx.Uint64(ws_lo)
                ) + fx.Int64((fx.Int32(ARRIVED) + blk) * fx.Int32(4))
                if send:
                    _atomic_add_at(arrived, 1, "one-as")
                nxt = nxt + fx.Int32(CHUNK)

        # ------------------------------------------------------------------ gemm1
        def gemm1_task(a, block, nb, tx, lane, wave):
            lane_div_16 = lane // fx.Int32(16)
            lane_mod_16 = lane % fx.Int32(16)
            e = _u(_global_i32_at(a["eids"], block))
            n_valid = _u(_global_i32_at(a["sched"], fx.Int32(SCHED_EXPECTED) + block))
            rows_cap = a["nblk"] * fx.Int32(BM)
            bx_m = block * fx.Int32(BM)
            col0 = nb * fx.Int32(NI) + wave * fx.Int32(NIW)
            cols = []
            for gu in range_constexpr(2):
                for c in range_constexpr(NC1):
                    n0 = e * fx.Int32(N2) + fx.Int32(gu * I) + col0 + fx.Int32(c * 16)
                    cols.append(((gu, c), n0, (gu, c // 2)))
            # fp8 W: the task's B table rows (gate blocks, then up blocks) and each
            # key's entry: its 128-col block of gate (gu 0) or up (gu 1)
            b_rows = (
                [
                    e * fx.Int32(N2 // 128)
                    + fx.Int32(gu * (I // 128) + j)
                    + nb * fx.Int32(NI // 128)
                    for gu in range(2)
                    for j in range(NI // 128)
                ]
                if FP8W
                else []
            )
            b_tab = (
                {
                    (gu, c): fx.Int32(BT1 // 4)
                    + (
                        fx.Int32(gu * (NI // 128))
                        + (wave * fx.Int32(NIW) + fx.Int32(c * 16)) // fx.Int32(128)
                    )
                    * fx.Int32(G1N + 1)
                    for gu in range(2)
                    for c in range(NC1)
                }
                if FP8W
                else {}
            )
            exps = (
                {}
                if FP8W
                else {
                    (gu, c, v): fx.Int32(
                        _global_i32_at(
                            a["w1e"],
                            e * fx.Int32(N2)
                            + fx.Int32(gu * I)
                            + col0
                            + fx.Int32(c * 16)
                            + lane_div_16 * fx.Int32(4)
                            + fx.Int32(v),
                        )
                    )
                    for gu in range(2)
                    for c in range(NC1)
                    for v in range(4)
                }
            )
            core = _make_core(
                K=H,
                W_ROWS=E_LOCAL * N2,
                BM=BM,
                PF=PF,
                bcm=BCM1,
                cols=cols,
                lds_ptr=lds_ptr,
                LDS_BYTES=LDS_BYTES,
                lane=lane,
                tx=tx,
                NT=NT,
                arg_w=a["w1"],
                arg_ws=a["w1s"],
                arg_a=a["ra"],
                a_num_bytes=fx.Int64(rows_cap) * fx.Int64(H),
                ARB=ARB,
                acm=2,
                TR=True,
                WQ=WQ,
            )

            def row_base16(
                row,
            ):  # sorted rows of the block; padding -> past the end (zeros)
                return (row < n_valid).select(
                    (bx_m + row) * fx.Int32(H // 16), rows_cap * fx.Int32(H // 16)
                )

            def build_ratios(R):
                """LDS: ratio[g][r] = s[r][g-1] / s[r][g] (ratio[0] = 1), fin[r] = s[r][G-1];
                a zero scale (padding row) acts as 1."""
                xs_tiles = _global_i32_buffer_tiles(
                    a["rs"], fx.Int64(rows_cap) * fx.Int64(G1N * 4), 1
                )
                xs_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
                one = fx.Float32(1.0)
                for p in range_constexpr(-(-(R * G1N) // NT)):
                    idx = tx + fx.Int32(p * NT)
                    r = idx // fx.Int32(G1N)
                    g = idx % fx.Int32(G1N)
                    ok = r < fx.Int32(R)
                    rr = ok.select(r, fx.Int32(0))
                    srow = (rr < n_valid).select(
                        (bx_m + rr) * fx.Int32(G1N), rows_cap * fx.Int32(G1N)
                    )
                    cur = _buffer_i32_scalar_read(xs_tiles, srow + g, xs_atom).bitcast(
                        fx.Float32
                    )
                    prv = _buffer_i32_scalar_read(
                        xs_tiles, srow + fx.max(g - fx.Int32(1), fx.Int32(0)), xs_atom
                    ).bitcast(fx.Float32)
                    cur = (cur == fx.Float32(0.0)).select(one, cur)
                    prv = (prv == fx.Float32(0.0)).select(one, prv)
                    ratio = (g == fx.Int32(0)).select(one, prv / cur)
                    _lds_f32_store_masked(
                        lds_f32, fx.Int32(RATIO // 4) + g * fx.Int32(BM) + rr, ratio, ok
                    )
                    _lds_f32_store_masked(
                        lds_f32,
                        fx.Int32(RATIO // 4 + G1N * BM) + rr,
                        cur,
                        ok & (g == fx.Int32(G1N - 1)),
                    )

            def hook(acc, kt, s):
                kk = s * 32  # K offset of MFMA step s in the 128-K tile
                if const_expr(kk % 128 != 0):
                    return
                for key in acc:
                    rb = None
                    if const_expr(FP8W):  # W's ratio: one per key
                        rb = fx.Float32(lds_f32[b_tab[key] + kt])
                    for mi in range_constexpr(
                        len(acc[key])
                    ):  # TR: lane = row mi*16 + L%16
                        if const_expr(QX):
                            rt = lds_f32[
                                fx.Int32(RATIO // 4 + mi * 16)
                                + kt * fx.Int32(BM)
                                + lane_mod_16
                            ]
                            if const_expr(FP8W):
                                rt = fx.Float32(rt) * rb
                        else:
                            rt = rb
                        a4 = Vec(acc[key][mi].load())
                        acc[key][mi].store(
                            Vec.from_elements(
                                [a4[v] * rt for v in range(4)], fx.Float32
                            )
                        )

            def epilogue(R, acc):
                """dequant -> act(gate)*up in registers -> row amax over the lane's columns,
                the 4 lane groups and the NW waves (LDS [row][NW]) -> fp8 a2 + f32 scale per
                (row, 128 cols), written through."""
                MR = R // 16
                rows = [fx.Int32(mi * 16) + lane_mod_16 for mi in range(MR)]
                valid = [r < n_valid for r in rows]
                if const_expr(FP8W):  # acc is in units of the last group's scale
                    fac = {
                        (gu, c, v): fx.Float32(lds_f32[b_tab[gu, c] + fx.Int32(G1N)])
                        for gu in range(2)
                        for c in range(NC1)
                        for v in range(4)
                    }
                else:
                    fac = {k: _channel_factor(v) for k, v in exps.items()}
                rsc = [
                    fx.Float32(lds_f32[fx.Int32(RATIO // 4 + G1N * BM) + r])
                    for r in rows
                ]
                situ = None
                if const_expr(ACT != "silu"):
                    situ = situ_params(*[fx.Float32(a[k]) for k in NAMESF])
                ys = {}
                for mi in range_constexpr(MR):
                    for c in range_constexpr(NC1):
                        g4 = Vec(fx.memref_load_vec(acc[0, c][mi]))
                        u4 = Vec(fx.memref_load_vec(acc[1, c][mi]))
                        gv = [
                            fx.Float32(g4[v]) * (rsc[mi] * fac[0, c, v])
                            for v in range(4)
                        ]
                        uv = [
                            fx.Float32(u4[v]) * (rsc[mi] * fac[1, c, v])
                            for v in range(4)
                        ]
                        ys[mi, c] = gate_up_act(ACT, gv, uv, situ)
                col = (
                    nb * fx.Int32(NI) + wave * fx.Int32(NIW) + lane_div_16 * fx.Int32(4)
                )
                for mi in range_constexpr(MR):
                    am = fx.Float32(0.0)
                    for c in range_constexpr(NC1):
                        for v in range_constexpr(4):
                            am = fx.maxnumf(am, fx.absf(ys[mi, c][v]))
                    am = fx.maxnumf(am, am.shuffle_xor(16, 64))
                    am = fx.maxnumf(am, am.shuffle_xor(32, 64))
                    lds_f32[rows[mi] * fx.Int32(NW) + wave] = am  # 4 lanes, same value
                gpu.barrier()
                out_rs = _rsrc(a["a2"], fx.Int64(bx_m + fx.Int32(BM)) * fx.Int64(I))
                os_rs = _rsrc(
                    a["a2s"], fx.Int64(bx_m + fx.Int32(BM)) * fx.Int64(G2N * 4)
                )
                grp = wave // fx.Int32(WPG)
                for mi in range_constexpr(MR):
                    am = fx.Float32(0.0)
                    pv = Vec(
                        lds_vec_load(
                            lds_i32,
                            (rows[mi] * fx.Int32(NW) + grp * fx.Int32(WPG))
                            * fx.Int32(4),
                            Vec.make_type(WPG, fx.Float32),
                            fx.Float32,
                            align=4 * WPG,
                        )
                    )
                    for j in range_constexpr(WPG):
                        am = fx.maxnumf(am, fx.Float32(pv[j]))
                    s = (am == 0.0).select(
                        fx.Float32(1.0), am * fx.Float32(1.0 / 240.0)
                    )
                    with fx.fastmath("afn"):  # v_rcp_f32, not an IEEE division
                        inv = fx.Float32(1.0) / s
                    srow = bx_m + rows[mi]
                    for c in range_constexpr(NC1):
                        q = _fp8x4_pack(*[ys[mi, c][v] * inv for v in range(4)])
                        buffer_store(
                            q,
                            out_rs,
                            (srow * fx.Int32(I) + col + fx.Int32(c * 16))
                            // fx.Int32(4),
                            mask=valid[mi],
                            cache_modifier=WT_AUX,
                        )
                    buffer_store(
                        s,
                        os_rs,
                        srow * fx.Int32(G2N) + nb * fx.Int32(GPW) + grp,
                        mask=valid[mi]
                        & (lane_div_16 == fx.Int32(0))
                        & (wave % fx.Int32(WPG) == fx.Int32(0)),
                        cache_modifier=WT_AUX,
                    )

            def run_rows(R):
                def pre_a():  # the W prefetch is in flight
                    if wave == fx.Int32(0):
                        spin_ge(
                            a["ws"]
                            + fx.Int64((fx.Int32(ARRIVED) + block) * fx.Int32(4)),
                            n_valid,
                            "one-as",
                        )
                    gpu.barrier()
                    if const_expr(FP8W):
                        build_b_ratios(a["w1s"], W1S_BYTES, b_rows, G1N, BT1, tx)
                    if const_expr(QX):
                        build_ratios(R)  # visible after the K loop's first barrier
                    else:  # LDS fin[r]: row r's x scale (the first of its G1N); padding: 1
                        rs_rs = _rsrc(a["rs"], fx.Int64(rows_cap) * fx.Int64(G1N * 4))
                        ok = tx < n_valid
                        sc = fx.Int32(_ld(rs_rs, (bx_m + tx) * fx.Int32(G1N), mask=ok))
                        _lds_f32_store_masked(
                            lds_f32,
                            fx.Int32(RATIO // 4 + G1N * BM) + tx,
                            ok.select(sc.bitcast(fx.Float32), fx.Float32(1.0)),
                            tx < fx.Int32(R),
                        )

                acc = core(R, row_base16, hook if (QX or FP8W) else None, pre_a=pre_a)
                epilogue(R, acc)
                rocdl.s_waitcnt(vmcnt=0)  # a2 written through
                gpu.barrier()
                if wave == fx.Int32(0):
                    lane0_add(
                        a["sched"]
                        + fx.Int64(
                            (fx.Int32(SCHED_EXPECTED) + a["nblk"] + block) * fx.Int32(4)
                        ),
                        1,
                    )

            _dispatch_rows(ROWV, [n_valid <= fx.Int32(r) for r in ROWV[:-1]], run_rows)

        # ------------------------------------------------------------------ gemm2
        def gemm2_task(a, block, nb, tx, lane, wave):
            lane_div_16 = lane // fx.Int32(16)
            lane_mod_16 = lane % fx.Int32(16)
            e = _u(_global_i32_at(a["eids"], block))
            n_valid = _u(_global_i32_at(a["sched"], fx.Int32(SCHED_EXPECTED) + block))
            rows_cap = a["nblk"] * fx.Int32(BM)
            bx_m = block * fx.Int32(BM)
            col0 = nb * fx.Int32(NO) + wave * fx.Int32(NOW)
            cols = [
                ((0, c), e * fx.Int32(H) + col0 + fx.Int32(c * 16), (0, c // 2))
                for c in range(NC2)
            ]
            # fp8 W: the task's B table rows and this wave's entry (its 128-col block)
            b_rows = (
                [
                    e * fx.Int32(H // 128) + nb * fx.Int32(NO // 128) + fx.Int32(j)
                    for j in range(NB2)
                ]
                if FP8W
                else []
            )
            b_tab = (
                fx.Int32(BT2 // 4)
                + (wave * fx.Int32(NOW)) // fx.Int32(128) * fx.Int32(G2N + 1)
                if FP8W
                else None
            )
            exps = (
                []
                if FP8W
                else [
                    fx.Int32(
                        _global_i32_at(
                            a["w2e"],
                            e * fx.Int32(H) + col0 + fx.Int32(c * 16) + lane_mod_16,
                        )
                    )
                    for c in range(NC2)
                ]
            )
            core = _make_core(
                K=I,
                W_ROWS=E_LOCAL * H,
                BM=BM,
                PF=PF,
                bcm=BCM2,
                cols=cols,
                lds_ptr=lds_ptr,
                LDS_BYTES=LDS_BYTES,
                lane=lane,
                tx=tx,
                NT=NT,
                arg_w=a["w2"],
                arg_ws=a["w2s"],
                arg_a=a["a2"],
                a_num_bytes=fx.Int64(rows_cap) * fx.Int64(I),
                ARB=ARB,
                acm=2,
                TR=False,
                WQ=WQ,
            )

            def row_base16(row):
                return (bx_m + row) * fx.Int32(I // 16)

            def build_ratios(R):
                as_tiles = _global_i32_buffer_tiles(
                    a["a2s"], fx.Int64(rows_cap) * fx.Int64(G2N * 4), 1
                )
                as_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
                for p in range_constexpr(-(-(R * G2N) // NT)):
                    idx = tx + fx.Int32(p * NT)
                    r = idx // fx.Int32(G2N)
                    g = idx % fx.Int32(G2N)
                    ok = r < fx.Int32(R)
                    rr = ok.select(r, fx.Int32(0))
                    srow = (bx_m + rr) * fx.Int32(G2N)
                    cur = _buffer_i32_scalar_read(as_tiles, srow + g, as_atom).bitcast(
                        fx.Float32
                    )
                    prv = _buffer_i32_scalar_read(
                        as_tiles, srow + fx.max(g - fx.Int32(1), fx.Int32(0)), as_atom
                    ).bitcast(fx.Float32)
                    cur_ok = (cur == fx.Float32(0.0)).select(fx.Float32(1.0), cur)
                    ratio = (g == fx.Int32(0)).select(fx.Float32(1.0), prv / cur_ok)
                    _lds_f32_store_masked(
                        lds_f32, fx.Int32(RATIO // 4) + g * fx.Int32(BM) + rr, ratio, ok
                    )
                    _lds_f32_store_masked(
                        lds_f32,
                        fx.Int32(RATIO // 4 + G2N * BM) + rr,
                        cur,
                        ok & (g == fx.Int32(G2N - 1)),
                    )

            def ratio4(base_f32, mi):
                """this lane's 4 rows (L//16)*4 + v of 16-row block mi from an f32 [BM] row."""
                off = (
                    base_f32 + fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)
                ) * fx.Int32(4)
                return Vec(
                    lds_vec_load(
                        lds_i32, off, Vec.make_type(4, fx.Float32), fx.Float32, align=16
                    )
                )

            def hook(acc, kt, s):
                kk = s * 32
                if const_expr(kk % 128 != 0):
                    return
                rb = None
                if const_expr(FP8W):  # W's ratio: one per wave
                    rb = fx.Float32(lds_f32[b_tab + kt])
                for mi in range_constexpr(len(acc[0, 0])):
                    rt = ratio4(fx.Int32(RATIO // 4) + kt * fx.Int32(BM), mi)
                    if const_expr(FP8W):
                        rt = [fx.Float32(rt[v]) * rb for v in range(4)]
                    for c in range_constexpr(NC2):
                        a4 = Vec(acc[0, c][mi].load())
                        acc[0, c][mi].store(
                            Vec.from_elements(
                                [a4[v] * rt[v] for v in range(4)], fx.Float32
                            )
                        )

            def scatter(R):
                """rows of the bf16 LDS tile [R, NO] -> BF16x2 atomics into out[token] of the
                route's source rank (128 B per row per pass)."""
                tab = global_typed_ptr(a["tab_out"], T.i64, align=8)
                rid = global_typed_ptr(a["rid"], T.i32)
                cs = (tx % fx.Int32(32)) * fx.Int32(2)
                rows = []
                for p in range_constexpr(-(-R // RPS)):
                    rib = fx.Int32(p * RPS) + tx // fx.Int32(32)
                    sid = fx.Int32(rid[bx_m + fx.min(rib, fx.Int32(R - 1))])
                    ok = (rib < fx.Int32(R)) & (rib < n_valid) & (sid >= fx.Int32(0))
                    rank = ok.select(
                        (sid >> fx.Int32(token_shift)) & fx.Int32(NPES - 1), fx.Int32(0)
                    )
                    row_addr = (
                        fx.Int64(tab[rank])
                        + fx.Int64(sid & fx.Int32(TOKEN_MASK)) * fx.Int64(H * 2)
                        + fx.Int64((nb * fx.Int32(NO) + cs) * fx.Int32(2))
                    )
                    rows.append((rib, row_addr, ok))
                for rib, row_addr, ok in rows:
                    if ok:
                        for s in range_constexpr(NO // 64):
                            pk = lds_vec_load(
                                lds_i32,
                                (rib * fx.Int32(NO) + cs + fx.Int32(s * 64))
                                * fx.Int32(2),
                                Vec.make_type(2, fx.BFloat16),
                                fx.BFloat16,
                                align=4,
                            )
                            fx.atomic_add(
                                get_element_ptr(
                                    _global_base_ptr1(row_addr), fx.Int32(s * 128)
                                ),
                                pk,
                                syncscope="agent",
                            )

            def epilogue(R, acc):
                """acc * fin * channel factor * route weight -> bf16 LDS tile [R, NO] -> scatter."""
                MR = R // 16
                if const_expr(FP8W):  # acc is in units of the last group's scale
                    fac = [fx.Float32(lds_f32[b_tab + fx.Int32(G2N)])] * NC2
                else:
                    fac = [_channel_factor(x) for x in exps]
                sw_tiles = _global_i32_buffer_tiles(
                    a["rw"], fx.Int64(rows_cap) * fx.Int64(4), 1
                )
                sw_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
                lds_bf16 = lds_typed_ptr(lds_i32, T.bf16)
                vals = {}
                for mi in range_constexpr(MR):
                    rows4 = fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)
                    wts = [
                        _buffer_i32_scalar_read(
                            sw_tiles, bx_m + rows4 + fx.Int32(v), sw_atom
                        ).bitcast(fx.Float32)
                        for v in range(4)
                    ]
                    fin = ratio4(fx.Int32(RATIO // 4 + G2N * BM), mi)
                    wts = [wts[v] * fin[v] for v in range(4)]
                    for c in range_constexpr(NC2):
                        a4 = Vec(acc[0, c][mi].load())
                        vals[mi, c] = [a4[v] * (wts[v] * fac[c]) for v in range(4)]
                gpu.barrier()  # ratio table / A region free for the staging tile
                for mi in range_constexpr(MR):
                    rows4 = fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)
                    for c in range_constexpr(NC2):
                        col = wave * fx.Int32(NOW) + fx.Int32(c * 16) + lane_mod_16
                        bf4 = Vec.from_elements(vals[mi, c], fx.Float32).to(fx.BFloat16)
                        for v in range_constexpr(4):
                            lds_bf16[(rows4 + fx.Int32(v)) * fx.Int32(NO) + col] = bf4[
                                v
                            ]
                gpu.barrier()
                scatter(R)

            def run_rows(R):
                def pre_a():  # the W prefetch is in flight
                    if wave == fx.Int32(0):
                        spin_ge(
                            a["sched"]
                            + fx.Int64(
                                (fx.Int32(SCHED_EXPECTED) + a["nblk"] + block)
                                * fx.Int32(4)
                            ),
                            fx.Int32(NG),
                            "agent-one-as",
                        )
                    gpu.barrier()
                    if const_expr(FP8W):
                        build_b_ratios(a["w2s"], W2S_BYTES, b_rows, G2N, BT2, tx)
                    build_ratios(R)

                acc = core(R, row_base16, hook, pre_a=pre_a)
                epilogue(R, acc)

            _dispatch_rows(ROWV, [n_valid <= fx.Int32(r) for r in ROWV[:-1]], run_rows)

        # ------------------------------------------------------------------ tasks
        def run_task(a, queue, ticket):
            tx, lane, wave = _opaque_tx(tx0)
            if ticket < fx.Int32(QPROD):
                produce(a, queue * fx.Int32(QPROD) + ticket, lane, wave)
            else:
                index = ticket - fx.Int32(QPROD)
                first, own = queue_layout(queue)
                # groups of PAIR blocks; their tasks alternate between the blocks tile by
                # tile, so both blocks of a group read each W tile at about the same time
                groups = (own + fx.Int32(PAIR - 1)) // fx.Int32(PAIR)
                gate_tasks, down_tasks = PAIR * NG, PAIR * ND
                group_tasks = gate_tasks + down_tasks
                lead = fx.min(groups, fx.Int32(lag))
                # gemm1 of the first `lead` groups; then units of gemm1(group u) and
                # gemm2(group u - lead); then the gemm2 of the last `lead` groups
                in_head = index < fx.Int32(gate_tasks) * lead
                mid = index - fx.Int32(gate_tasks) * lead
                unit_of = mid // fx.Int32(group_tasks)
                slot = mid - unit_of * fx.Int32(group_tasks)
                in_mid = mid < fx.Int32(group_tasks) * (groups - lead)
                tail = mid - fx.Int32(group_tasks) * (groups - lead)
                is_gate = in_head | (in_mid & (slot < fx.Int32(gate_tasks)))
                group = in_head.select(
                    index // fx.Int32(gate_tasks),
                    in_mid.select(
                        lead
                        + unit_of
                        - (slot >= fx.Int32(gate_tasks)).select(lead, fx.Int32(0)),
                        groups - lead + tail // fx.Int32(down_tasks),
                    ),
                )
                sub = in_head.select(
                    index % fx.Int32(gate_tasks),
                    in_mid.select(
                        (slot < fx.Int32(gate_tasks)).select(
                            slot, slot - fx.Int32(gate_tasks)
                        ),
                        tail % fx.Int32(down_tasks),
                    ),
                )
                local = group * fx.Int32(PAIR) + sub % fx.Int32(PAIR)
                tile = sub // fx.Int32(PAIR)
                block = first + local
                if (
                    local < own
                ):  # an odd last block leaves its pair partner's tasks empty
                    if is_gate:
                        gemm1_task(a, block, tile, tx, lane, wave)
                    else:
                        gemm2_task(a, block, tile, tx, lane, wave)

        queue, ticket = take_ticket(False)
        while ticket >= fx.Int32(0):
            run_task(reload(), queue, ticket)
            queue, ticket = take_ticket(True)

    @flyc.jit
    def launch(
        x8: fx.Int64,
        xs: fx.Int64,
        tw: fx.Int64,
        route: fx.Int64,
        info: fx.Int64,
        tab_a: fx.Int64,
        tab_s: fx.Int64,
        tab_id: fx.Int64,
        tab_w: fx.Int64,
        tab_ws: fx.Int64,
        tab_out: fx.Int64,
        ra: fx.Int64,
        rs: fx.Int64,
        rid: fx.Int64,
        rw: fx.Int64,
        ws: fx.Int64,
        w1: fx.Int64,
        w1s: fx.Int64,
        w1e: fx.Int64,
        w2: fx.Int64,
        w2s: fx.Int64,
        w2e: fx.Int64,
        a2: fx.Int64,
        a2s: fx.Int64,
        eids: fx.Int64,
        nv: fx.Int64,
        sched: fx.Int64,
        routes: fx.Int32,
        tokens: fx.Int32,
        me: fx.Int32,
        nblk: fx.Int32,
        sb: fx.Float32,
        sbr: fx.Float32,
        lb: fx.Float32,
        lbr: fx.Float32,
        lim: fx.Float32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            x8,
            xs,
            tw,
            route,
            info,
            tab_a,
            tab_s,
            tab_id,
            tab_w,
            tab_ws,
            tab_out,
            ra,
            rs,
            rid,
            rw,
            ws,
            w1,
            w1s,
            w1e,
            w2,
            w2s,
            w2e,
            a2,
            a2s,
            eids,
            nv,
            sched,
            routes,
            tokens,
            me,
            nblk,
            sb,
            sbr,
            lb,
            lbr,
            lim,
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(NT, 1, 1), stream=stream)

    return launch


def max_tasks(nblk, H, I, producers=16, NI=128, NO=256):
    """Upper bound of the tickets of all queues (each may add one empty pair partner)."""
    return producers + (I // NI + H // NO) * (nblk + 2 * QUEUES)
