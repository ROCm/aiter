# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Large-batch (big-M) path of MegaMoeTP.forward: one persistent launch runs the MoE of T
tokens (m = T / tp rows per rank, ag_rs: every rank's routing of its own rows), in C chunks:

  RB  this rank's rows -> MXFP4 rows, + their routing        -> SDMA gather of rows + routing (C)
  S1  wait C; routes per expert       S2  route-list offsets, per-XCD unit queues
  S3  route lists (route index, weight) per expert
  U   fused units (bigm_gemm): GEMM1 -> act + MXFP4 in LDS -> GEMM2 -> route rows
  RQ  topk sum: own rows bf16, peer rows int8                -> SDMA MoE-output RS (E)
  RF  wait E; y = the sum (+ tail: res_out = y + res, next-layer norm, per-token FP8
      -> SDMA gather (G); or the y gather of the ar mode)
  GW  wait every G; advance the persistent signal base

All communication is MORI SDMA (puts with remote signals) issued and awaited inside the
kernel; the GEMM1 -> GEMM2 intermediate stays in LDS. CTAs (one per CU, 4 waves) claim
items with a per-XCD atomic ticket in a fixed phase order; an item only waits on items
before it (and on peers' puts of earlier items), so the schedule cannot deadlock.
Data another XCD reads in the same launch is stored write-through (sc0 | sc1).
"""

import functools
import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ..communication_ops_utils import atomic_add_agent, load_i32_global_agent
from ..mxfp4_gemm_common import _e8m0_from_amax
from .bigm_config import (
    C_CNT,
    C_ERR,
    C_FILL,
    C_OFF,
    C_PH,
    C_SB,
    C_SBH,
    C_TK,
    C_TURN,
    C_UN,
    C_XW,
    NKIND,
    NQ,
    P_RB,
    P_RF,
    P_RQ,
    P_S1,
    P_S2,
    P_S3,
    P_U,
    P_UB,
    PUTS_PER_CHUNK,
    QE,
    QID,
    QQ,
    QS,
    QUEUES,
    QX,
    QXS,
    SRT,
    UNIT_ROWS,
    UQ,
    ZQ,
)
from .bigm_gemm import NW, build_bigm_unit
from .common import (
    AUX_SYS,
    alive,
    amax,
    before,
    bf16x8_to_f32,
    bld,
    bst,
    f32_bits_add,
    f32_bits_max,
    fp4_pack,
    fp8x4_pack,
    fp8x8_decode,
    g_ld_sys,
    g_st_sys,
    i4x8_pack,
    i4x8_unpack,
    i8x4_pack,
    i8x4_unpack,
    i32,
    l2_writeback,
    lds_atomic_add,
    lds_ld_i32,
    lds_st,
    now,
    opaque,
    pack_bf16x8,
    rsrc,
    sum_live,
    uni,
    wait_vm,
    wave_red,
)

NTH = NW * 64
const_expr = fx.const_expr


@functools.cache
def compile_bigm_layer(
    H: int,
    I: int,
    E: int,
    K: int,
    tp: int,
    C: int,
    eps: float,
    gemma: bool,
    swiglu_limit: float,
    NAB: int = 4,
    BST: int = 3,
    BS2: int = 3,
    order: int = 3,
    opts: tuple = (),
    tailq: int = 1,
    act: str = "swiglu",
):
    """act: the units' activation ("swiglu", "silu"); E < 512 routed experts (+ the shared one).
    tailq: 0 no tail, 1 the tail with the q gather, 2 the local tail (this rank's q rows),
    3 no tail, y of every rank's rows (bf16, the "q" region: the y gather of the ar mode).
    The MoE interface (MegaMoeTP.forward, ag_rs): the caller's normed own rows x [m, H] and their
    routing (ids / tw [m, K1], the shared expert's column included) are gathered; the tail sums the
    MoE output (y), and with tailq res_out = y + res_in, the next-layer norm + FP8 rows of every
    rank (q gather).
    """
    spec = repr(
        sorted(locals().items())
    )  # every compile argument (the disk cache key, _bigm_cached)
    RG = UNIT_ROWS  # (one CTA of NW waves per CU)
    K1, NE = K + 1, E + 1
    HV = H // 8  # 16 B (8 bf16) columns of a row
    PIT = -(
        -HV // NTH
    )  # row iterations per thread (the last one partial unless NTH | HV)
    # per-expert ctrl / LDS entries (one thread per expert, NEB rounds of NTH)
    NEB = -(-NE // NTH)
    NES = NEB * NTH
    XST = 256 if NE <= 256 else 512
    if NE <= 256:
        CC_CNT, CC_OFF, CC_FILL = C_CNT, C_OFF, C_FILL
    else:
        CC_CNT, CC_OFF, CC_FILL = C_XW, C_XW + 8 * 512, C_XW + 16 * 512
    assert NE <= 512 and act in ("swiglu", "silu")

    od = dict(opts)  # tuned AITER_BIGM_* options (configs/mega_moe_tp_bigm_tuned.csv)

    def _opt(name, default):
        return od.get(name, default)

    # cache policy of the data one CTA writes and another (on another XCD: separate L2s) reads in
    # this launch -- route rows, own MoE rows, received E rows: stored cached and this XCD's L2
    # written back (buffer_wbl2 sc1) before a unit counts as done (release), loaded sc1
    CAUX = 16
    BS2 = int(
        _opt("AITER_BIGM_BS2", str(BS2))
    )  # GEMM2 B ring depth (6 = a whole group: the
    # route stores of a group never sit between a GEMM2 load and its wait)
    # the B rings' depths divide the unit's GEMM1 chunks (H / 256) / GEMM2 k-steps (I / 128): the
    # largest divisor up to the requested depth (H 4096: BST 3 -> 2; I 512: BS2 3 -> 2, 6 -> 4)
    BST = max(d for d in range(1, BST + 1) if (H // 256) % d == 0)
    BS2 = max(d for d in range(1, BS2 + 1) if (I // 128) % d == 0)
    # item order (see ORDER below); the host picks 3 (one unit round per chunk) or 4 (several)
    ORD = int(_opt("AITER_BIGM_ORD", str(order)))
    # tail sub-blocks: each chunk's topk sums / drains post their E / q puts per row sub-block
    NSB = int(_opt("AITER_BIGM_NSB", "4"))
    # head sub-blocks: each chunk's gather of the input rows + routing (C) per row sub-block
    NSBA = int(_opt("AITER_BIGM_NSBA", "1"))
    assert NSBA in (1, 2, 4)
    UA = int(_opt("AITER_BIGM_UA", "32"))  # ORD 4: first-round unit slots per XCD
    RQD = int(
        _opt("AITER_BIGM_RQD", "0")
    )  # topk sum: token pipeline depth (0: ZQ, a batch: every load first)
    # R8: route rows as MXFP8 (half the bytes of bf16 GEMM2 stores / topk-sum loads)
    R8 = int(_opt("AITER_BIGM_R8", "0"))
    E4 = int(
        _opt("AITER_BIGM_E4", "0")
    )  # MoE-output RS rows as int4 (+ bf16 scale per 32)
    # SXW 0: the routing plan waits only for the gathered routing; units wait for the gathered rows
    SXW = int(_opt("AITER_BIGM_SXW", "1"))
    if R8:
        RQD = max(RQD, 1)
    elif K1 * PIT > 16 and "AITER_BIGM_RQD" not in od:
        # bf16 route rows of many routes (GLM-5: 9 x 6144): a batch's topk-sum loads alone
        # take the register file (512 VGPRs + scratch spills): pipelined instead
        RQD = 1
    assert NSB in (1, 2, 4) and 16 % NSB == 0
    PPC = dict(PUTS_PER_CHUNK)
    PPC[QE] = PPC[QQ] = PPC[QS] = NSB
    PPC[QX] = PPC[QXS] = NSBA
    PPC[QID] = 2 * NSBA
    if tailq not in (1, 3):
        PPC[QQ] = PPC[QS] = (
            0  # no q gather (no tail, or the local tail: own rows' q only)
        )
    if tailq == 3:
        PPC[QS] = 0  # the y gather: bf16 rows on QQ, no scales
    ZX = 16  # io: rows per input item (quant + routing copy)
    assert H % 256 == 0 and 0 < E < 512 and K <= 64
    EPL = -(-E // 64)  # experts per lane of the plan's wave
    # int8 rows of the attention / MoE-output RS: one fp32 scale per 32 values
    RB = H + (H // 32) * 4
    SCO = H
    lay, unit_body = build_bigm_unit(
        H,
        I,
        K1,
        RG,
        NAB,
        BST,
        BS2,
        act,
        swiglu_limit,
        R8=R8,
    )
    LBASE = lay[
        "BYTES"
    ]  # row-phase / plan scratch after the unit's LDS (also reused ranges)
    LSC = LBASE  # 16 B scratch for block reductions
    LT = LBASE + 64  # ticket broadcast
    LH = LBASE + 128  # [NE] histogram / ranks
    LB2 = LH + 4 * NES  # [NE] bases
    LP = LB2 + 4 * NES  # [SRT * K1] route positions of a sort item ([NE] plan scratch)
    LDS_BYTES = LP + max(4 * SRT * K1, 64 + 4 * max(NE, 64 * EPL)) + 64
    assert LDS_BYTES <= 160 * 1024, LDS_BYTES

    def cpart(kk):  # row iteration kk is the partial last one
        return HV % NTH != 0 and kk == PIT - 1

    def col(tid, kk):
        """this thread's column of row iteration kk; past the row (a partial last iteration) the
        same lane of the last full 4-column group: those threads redo its loads, math and stores
        (identical values, its per-32 scales included; the fused tail: see own_col)"""
        p = tid + i32(kk * NTH)
        if not cpart(kk):
            return p
        return (p < i32(HV)).select(p, i32(HV - 4) + (p & i32(3)))

    def own_col(tid, kk):
        """None, else (a partial iteration) whether this thread's column is its own, not redone.
        The fused tail updates res in place (res_in == res_out): a redone column's res load
        can follow its owner's store, so there the redone columns store nothing and stay out
        of the row's sums."""
        if not cpart(kk):
            return None
        return tid + i32(kk * NTH) < i32(HV)

    def at_col(
        base, tid, kk
    ):  # base + this thread's first element (bf16) of iteration kk
        if not cpart(kk):
            return base + tid * i32(8) + i32(kk * NTH * 8)
        return base + col(tid, kk) * i32(8)

    def fb(v):
        return f"{struct.unpack('<I', struct.pack('<f', v))[0]:x}"

    # item order: chunk k+1's prep (input rows RB, plan) ahead of chunk k's units and drain
    # (a CTA spinning on an item cannot take later ones: items that wait for a transfer come
    # after the work that overlaps it) chunk k's units, then chunk k+1's prep (claimed as
    # chunk k's units drain), chunk k's topk sum, then chunk k-1's drain after k's units
    def prep(k):
        return [(k, P_RB), (k, P_S1), (k, P_S2), (k, P_S3)]

    assert ORD in (3, 4), ORD
    if ORD == 4:
        # U(k) = the first unit slot of every CTA, UB(k) the rest: chunk k+1's input rows run
        # between them and its routing plan right after, so chunk k+1's units fill the CTAs
        # chunk k's last unit round leaves idle
        ORDER = prep(0)
        for k in range(C):
            ORDER += [(k, P_U)]
            if k >= 1:
                ORDER += [(k - 1, P_RQ)]
            if k + 1 < C:
                ORDER += [(k + 1, P_RB)]
            ORDER += [(k, P_UB)]
            if k >= 1:
                ORDER += [(k - 1, P_RF)]
            if k + 1 < C:
                ORDER += [(k + 1, P_S1), (k + 1, P_S2), (k + 1, P_S3)]
        ORDER += [(C - 1, P_RQ), (C - 1, P_RF)]
    elif ORD == 3:
        # chunk k's topk sum right after chunk k+1's prep (it fills the gap while chunk k+1's
        # routing lands, on the whole machine) so E(k), RF(k), G(k) overlap chunk k+1's units
        ORDER = prep(0)
        for k in range(C):
            ORDER += [(k, P_U)]
            if k >= 1:  # on the CTAs chunk k's units leave idle (E(k-1) is on its way)
                ORDER += [(k - 1, P_RF)]
            if k + 1 < C:
                ORDER += prep(k + 1)
            ORDER += [(k, P_RQ)]
        ORDER += [(C - 1, P_RF)]
    if not any(x == P_UB for _, x in ORDER):
        ORDER += [(k, P_UB) for k in range(C)]  # empty: U(k) takes every unit slot
    KINDS = (P_RB, P_S1, P_S2, P_S3, P_U, P_RQ, P_RF, P_UB)
    assert sorted(ORDER) == sorted((k, x) for k in range(C) for x in KINDS), ORDER
    USPLIT = ORD == 4

    name = (
        f"bigm_layer_h{H}_i{I}_e{E}k{K}_tp{tp}_c{C}_rg{RG}_{fb(eps)}{'g' if gemma else 'r'}"
        + f"_v23_t{tailq}_c1h{NSBA}_o{ORD}s{NSB}a{UA}d{RQD}r{R8}w{SXW}e{E4}b{BS2}"
        + ("" if act == "swiglu" else f"_{act}{fb(swiglu_limit)}")
    )

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.jit
    def blk_red_vec(L, tid, vals, op):
        """block reduction of len(vals) <= 4 values with one barrier pair"""
        n = len(vals)
        rs_ = [wave_red(v, tid % i32(64), op) for v in vals]
        if (tid % i32(64)) == i32(0):
            for i in range_constexpr(n):
                lds_st(
                    L, i32(LSC) + ((tid // i32(64)) * i32(n) + i32(i)) * i32(4), rs_[i]
                )
        gpu.barrier()
        out = []
        for i in range_constexpr(n):
            t = lds_ld_i32(L, i32(LSC + i * 4))
            for k in range_constexpr(1, NW):
                t = op(t, lds_ld_i32(L, i32(LSC + (k * n + i) * 4)))
            out.append(t)
        gpu.barrier()
        return out

    def i8_scale_ld(rx, row, p):
        """the fp32 bits of the scale of values p*8.. of an int8 row"""
        g = p // i32(4)  # scale group (32 values)
        return fx.Int32(bld(rx, row * i32(RB) + i32(SCO) + g * i32(4), 0, T.i32, CAUX))

    def i8_rows_ld(rx, row, p):
        return (
            fx.Vector(bld(rx, row * i32(RB) + p * i32(8), 0, T.vec(2, T.i32), CAUX)),
            i8_scale_ld(rx, row, p),
        )

    def i8_rows_val(ld):
        d, sc = ld
        return i8x4_unpack(d[0], sc.bitcast(fx.Float32)) + i8x4_unpack(
            d[1], sc.bitcast(fx.Float32)
        )

    # E (MoE-output RS) rows: int8 + f32 scale per 32 (RB bytes), or E4: int4 + bf16 scale per 32
    # (QuickReduce INT4's code: ATOM's MoE-output reduce-scatter)
    RBE = (H // 2 + H // 16) if E4 else RB

    def e_ld(rx, row, p):
        if const_expr(E4):
            d = fx.Int32(bld(rx, row * i32(RBE) + p * i32(4), 0, T.i32, CAUX))
            sb = fx.Int32(
                bld(
                    rx,
                    row * i32(RBE) + i32(H // 2) + (p // i32(4)) * i32(2),
                    0,
                    T.i16,
                    CAUX,
                )
            ) & i32(0xFFFF)
            return (d, sb)
        return i8_rows_ld(rx, row, p)

    def e_val(ld):
        if const_expr(E4):
            d, sb = ld
            return i4x8_unpack(d, (sb << i32(16)).bitcast(fx.Float32))
        return i8_rows_val(ld)

    @flyc.jit
    def e_store(f, rd, drow, p, lane):
        if const_expr(E4):
            am = amax(f)
            am = am.maximumf(am.shuffle_xor(i32(1), i32(64)))
            am = am.maximumf(am.shuffle_xor(i32(2), i32(64)))
            # QuickReduce CodecQ4's code (ATOM's INT4 MoE-output reduce-scatter): scale -amax / 8,
            # all 16 levels [-8, 7] (amax -> -8, -amax -> 8 clamped to 7); MoE output rel_l2 vs
            # torch 0.0856 (symmetric [-7, 7]: 0.0882)
            sc = fx.Float32(
                fx.Float32(am.maximumf(fx.Float32(1e-30)) / fx.Float32(-8.0))
                .to(fx.BFloat16)
                .to(fx.Float32)
            )
            bst(
                i4x8_pack(f, fx.Float32(1.0) / sc),
                rd,
                drow * i32(RBE) + p * i32(4),
                0,
                AUX_SYS,
            )
            if (lane & i32(3)) == i32(0):
                bst(
                    fx.Int16(sc.bitcast(fx.Int32) >> i32(16)),
                    rd,
                    drow * i32(RBE) + i32(H // 2) + (p // i32(4)) * i32(2),
                    0,
                    AUX_SYS,
                )
        else:
            i8_row_store(f, rd, drow, p, lane)

    @flyc.jit
    def i8_row_store(f, rd, drow, p, lane):
        am = amax(f)
        am = am.maximumf(am.shuffle_xor(i32(1), i32(64)))
        am = am.maximumf(am.shuffle_xor(i32(2), i32(64)))
        g = p // i32(4)
        sc = am.maximumf(fx.Float32(1e-30)) / fx.Float32(127.0)
        inv = fx.Float32(1.0) / sc
        dv = fx.Vector.from_elements(
            [i8x4_pack(f[0:4], inv), i8x4_pack(f[4:8], inv)], fx.Int32
        )
        bst(dv, rd, drow * i32(RB) + p * i32(8), 0, AUX_SYS)
        if (lane & i32(3)) == i32(0):
            bst(
                sc.bitcast(fx.Int32),
                rd,
                drow * i32(RB) + i32(SCO) + g * i32(4),
                0,
                AUX_SYS,
            )

    @flyc.kernel(name=name, known_block_size=[NTH, 1, 1])
    def kernel(
        wb: fx.Int64,  # this rank's data arena (SdmaComm: hipIpc, not a MORI window)
        pbases: fx.Int64,  # [tp] int64 every rank's arena as mapped here (the put destinations)
        dc: fx.Int64,  # MORI device communicator
        consumed: fx.Int64,  # [tp][NQ] int32 puts awaited before this launch (persistent;
        # the next launch's at consumed ^ (tp * NQ * 4): two buffers in turn)
        offs: fx.Int64,  # [8] int64 arena region offsets
        part: fx.Int64,  # x [m, H] bf16 own rows
        resid: fx.Int64,  # ids [m, K1] int32
        wpost: fx.Int64,  # tw [m, K1] f32
        wnext: fx.Int64,  # [H] the next layer's norm weight (tail)
        res: fx.Int64,  # res_in [m, H] bf16 (tail)
        normed: fx.Int64,  # y [m, H] bf16 out (the MoE output)
        resout: fx.Int64,  # [m, H] bf16 out
        ybuf: fx.Int64,  # [m, H] bf16 own MoE rows
        routes: fx.Int64,  # [T * K1, H] bf16 route rows (chunk k's from route tp * o_k * K1)
        lrix: fx.Int64,  # [C][T * K1] route lists
        lwt: fx.Int64,
        utab: fx.Int64,  # [C][8][ucap][4] units
        w1: fx.Int64,
        w1s: fx.Int64,
        w2: fx.Int64,
        w2s: fx.Int64,
        ctrl: fx.Int64,
        chunks: fx.Int64,  # [C][2] int32: (rows offset o, rows s) per rank
        rank: fx.Int32,
        m: fx.Int32,
        ucap: fx.Int32,
    ):
        sa = fx.SharedAllocator()
        st = sa.allocate(Shared).peek()
        L = fx.Int32(fx.ptrtoint(st.raw.ptr))
        tid0 = fx.Int32(gpu.thread_id("x"))
        tid = tid0
        lane = tid % i32(64)
        bid = i32(gpu.block_id("x"))
        xc = bid % i32(8)
        roffs = rsrc(offs)

        def off_of(i):
            return fx.Int64(bld(roffs, i32(i * 8), 0, T.i64))

        a = {
            "wb": wb,
            "off_xq": off_of(0),
            "off_xs": off_of(1),
            "off_ids": off_of(2),
            "off_tw": off_of(3),
            "off_eq": off_of(4),
            "off_erecv": off_of(5),
            "off_q": off_of(6),
            "off_qs": off_of(7),
        }

        def ca(i):
            return ctrl + fx.Int64(i * i32(4))

        def wp(name):  # local window address of a region
            return wb + a["off_" + name]

        rch = rsrc(chunks)

        def chunk(k):  # (rows offset o, rows s) of chunk k (traced)
            return fx.Int32(bld(rch, k * i32(8), 0, T.i32)), fx.Int32(
                bld(rch, k * i32(8) + i32(4), 0, T.i32)
            )

        def lbase(k):  # chunk k's first route (route lists, route rows: chunks packed)
            return i32(tp) * chunk(k)[0] * i32(K1)

        def rbase(k):  # chunk k's route rows (bytes)
            return fx.Int64(lbase(k)) * fx.Int64(H * 2)

        # this launch's signal base per (peer, queue) (GW writes the next launch's into the
        # other buffer)
        rcons = rsrc(consumed)

        def sig_base(p, q):
            return fx.Int32(
                bld(rcons, (p * i32(NQ) + i32(q)) * i32(4), 0, T.i32, AUX_SYS)
            )

        # ---- SDMA (MORI device functions) ----
        from flydsl.compiler.extern_link import link_extern
        from flydsl.expr.extern import ffi

        from .sdma import sdma_bitcode

        bc = sdma_bitcode(6)
        put_ptr = link_extern(
            ffi(
                "cco_sdma_put_ptr_rsig__thread",
                ["uint64", "int32", "uint64", "uint64", "uint64", "int32"],
                "int32",
            ),
            bitcode_path=bc,
        )
        sig_slot = link_extern(
            ffi("cco_sdma_signal_slot", ["uint64", "int32", "int32"], "uint64"),
            bitcode_path=bc,
        )

        @flyc.jit
        def take_turn(q, nth):
            # posts on one queue in a fixed order, one poster at a time (MORI loses remote
            # signals of concurrent puts on the same queue)
            cur = fx.Int32(load_i32_global_agent(ca(i32(C_TURN + 32 * q))))
            while cur < nth:
                rocdl.s_sleep(1)
                cur = fx.Int32(load_i32_global_agent(ca(i32(C_TURN + 32 * q))))

        def give_turn(q, nth):
            g_st_sys(ca(i32(C_TURN + 32 * q)), nth + i32(1))

        rpb = rsrc(pbases)

        def put(peer, dst_region, dst_off, src_region, src_off, nbytes, q):
            # region + offset of the peer's arena <- the same of ours (remote-signalled)
            dst = (
                fx.Int64(bld(rpb, peer * i32(8), 0, T.i64))
                + a["off_" + dst_region]
                + fx.Int64(dst_off)
            )
            src = wb + a["off_" + src_region] + fx.Int64(src_off)
            put_ptr(dc, peer, dst, src, fx.Int64(nbytes), i32(q))

        def wait_puts(q, n_per, k):
            # every peer's first (k + 1) * n_per puts on queue q of this launch landed
            wait_puts_n(q, (k + i32(1)) * i32(n_per))

        @flyc.jit
        def wait_puts_n(q, n):
            if lane < i32(tp):
                p = lane
                if p != rank:
                    slot = fx.Int64(sig_slot(dc, p, i32(q)))
                    tgt = sig_base(p, q) + n
                    cur = g_ld_sys(slot)
                    t0 = now()
                    while before(cur, tgt) & alive(t0):
                        rocdl.s_sleep(2)
                        cur = g_ld_sys(slot)
                    if before(
                        cur, tgt
                    ):  # gave up on a peer: flag it (the results are not valid)
                        g_st_sys(ca(i32(C_ERR)), i32(1))

        def ph(k, kind):
            return ca(i32(C_PH) + (k * i32(NKIND) + i32(kind)) * i32(32))

        @flyc.jit
        def done_last(k, kind, n):
            """this CTA finished an item of (k, kind): True (uniform) for the last of n"""
            wait_vm(0)
            gpu.barrier()
            if tid == i32(0):
                v = fx.Int32(atomic_add_agent(ph(k, kind), i32(1)))
                lds_st(L, i32(LT) + i32(4), (v == n - i32(1)).select(i32(1), i32(0)))
            gpu.barrier()
            return lds_ld_i32(L, i32(LT) + i32(4)) == i32(1)

        @flyc.jit
        def wait_ph(k, kind, n):
            wait_at(ph(k, kind), n)

        @flyc.jit
        def wait_at(addr, n):
            if tid == i32(0):
                cur = fx.Int32(load_i32_global_agent(addr))
                while cur < n:
                    rocdl.s_sleep(1)
                    cur = fx.Int32(load_i32_global_agent(addr))
            gpu.barrier()

        def sbc(k, w, b):  # sub-block done counter (w 0: RQ items, 1: RF items)
            return ca(i32(C_SB) + ((k * i32(2) + i32(w)) * i32(16) + b) * i32(8))

        @flyc.jit
        def done_at(addr, n):
            """like done_last, on a counter address (callers ran done_last first: stores drained)"""
            if tid == i32(0):
                v = fx.Int32(atomic_add_agent(addr, i32(1)))
                lds_st(L, i32(LT) + i32(4), (v == n - i32(1)).select(i32(1), i32(0)))
            gpu.barrier()
            return lds_ld_i32(L, i32(LT) + i32(4)) == i32(1)

        def il(j, n):
            # XCD-interleaved item order (XCD x's contiguous slice -> every 8th item) so the
            # sub-blocks complete in order; identity when n is not a multiple of 8
            q = n // i32(8)
            qq = fx.max(q, i32(1))
            x = j // qq
            jn = (j - x * qq) * i32(8) + x
            return ((n % i32(8) == i32(0)) & (q > i32(0))).select(jn, j)

        def claim():
            if tid == i32(0):
                lds_st(
                    L,
                    i32(LT),
                    fx.Int32(atomic_add_agent(ca(i32(C_TK) + xc * i32(32)), i32(1))),
                )
            gpu.barrier()
            v = lds_ld_i32(L, i32(LT))
            gpu.barrier()
            return v

        def part8(n):
            q = n // i32(8)
            r = n - q * i32(8)
            return xc * q + (xc < r).select(xc, r), q + (xc < r).select(i32(1), i32(0))

        # ---- item counts per (chunk, kind) (rows s per rank of the chunk) ----

        def cnt(kind, s_, k_=0):
            if const_expr(kind == P_RB):  # the input rows (item_rx)
                return s_ // i32(ZX)
            if const_expr(kind == P_RF):
                return s_ // i32(ZQ)
            if const_expr(kind in (P_S1, P_S3)):
                return (i32(tp) * s_ + i32(SRT - 1)) // i32(SRT)
            if const_expr(kind == P_S2):
                return i32(1)
            if const_expr(kind == P_U):
                return ucap * i32(8)
            return (i32(tp) * s_) // i32(ZQ)

        def chunk_v(k):  # per-claim (not hoisted: system-scope load)
            return fx.Int32(bld(rch, k * i32(8) + i32(4), 0, T.i32, AUX_SYS))

        def slices_of(sk):
            order = ORDER
            out = []
            tot_ = i32(0)
            for oi in range_constexpr(len(order)):
                k, kind = order[oi]
                n = cnt(kind, sk[k], k)
                if const_expr(kind == P_S2):
                    lo_, c_ = i32(0), (xc == i32(0)).select(i32(1), i32(0))
                elif const_expr(kind == P_U):
                    lo_, c_ = i32(0), (fx.min(ucap, i32(UA)) if USPLIT else ucap)
                elif const_expr(kind == P_UB):
                    lo_, c_ = i32(UA), (
                        fx.max(ucap - i32(UA), i32(0)) if USPLIT else i32(0)
                    )
                else:
                    lo_, c_ = part8(n)
                out.append((k, kind, lo_, tot_, tot_ + c_))
                tot_ = tot_ + c_
            return out, tot_

        _, gw_lo = slices_of([chunk(i32(k))[1] for k in range(C)])
        nall = gw_lo + (xc == i32(0)).select(i32(1), i32(0))

        r_part, r_res, r_resid = rsrc(part), rsrc(res), rsrc(resid)
        r_wpost, r_wnext, r_y = rsrc(wpost), rsrc(wnext), rsrc(ybuf)
        r_norm, r_out = rsrc(normed), rsrc(resout)

        def norm_w(rw, p):
            return bf16x8_to_f32(bld(rw, p * i32(16), 0, T.vec(4, T.i32)))

        def gate_w(x, wk):
            return x * ((wk + fx.Float32(1.0)) if gemma else wk)

        # ======================= items =======================

        def sbh(k, w, b):  # head sub-block done counter (w 1: input row items)
            return ca(i32(C_SBH) + ((k * i32(2) + i32(w)) * i32(16) + b) * i32(8))

        @flyc.jit
        def post_c(k, bh):
            # thread 0: this head sub-block's quantized rows + routing -> every peer
            o, s = chunk(k)
            B = i32(tp) * o
            sa = s // i32(NSBA)
            r0b = B + rank * s + bh * sa
            t1 = k * i32(NSBA) + bh
            CPUTS = (
                (QX, "xq", H // 2),
                (QXS, "xs", H // 32),
                (QID, "ids", K1 * 4),
                (QID, "tw", K1 * 4),
            )
            for ci in range_constexpr(4):
                q_, reg, rowb = CPUTS[ci]
                tq = t1 * i32(2) if const_expr(q_ == QID) else t1
                if const_expr(reg != "tw"):
                    take_turn(q_, tq)
                for p_ in range_constexpr(tp):
                    if i32(p_) != rank:
                        put(
                            i32(p_),
                            reg,
                            r0b * i32(rowb),
                            reg,
                            r0b * i32(rowb),
                            sa * i32(rowb),
                            q_,
                        )
                if const_expr(reg in ("xq", "xs")):
                    give_turn(q_, tq)
                if const_expr(reg == "tw"):
                    give_turn(q_, tq + i32(1))

        @flyc.jit
        def item_rx(k, j):
            # io: ZX of this rank's (normed) rows -> MXFP4 rows + routing in the window; the last
            # item of a head sub-block posts it to every peer (gather C)
            tid = opaque(tid0)
            lane = tid % i32(64)
            o, s = chunk(k)
            B = i32(tp) * o
            jx = il(j, cnt(P_RB, s))
            rq, rqs = rsrc(wp("xq")), rsrc(wp("xs"))
            i0 = jx * i32(ZX)
            ld = [
                [
                    bld(
                        r_part,
                        ((o + i0 + i32(ii)) * i32(H) + col(tid, kk) * i32(8)) * i32(2),
                        0,
                        T.vec(4, T.i32),
                    )
                    for kk in range(PIT)
                ]
                for ii in range(ZX)
            ]
            if tid < i32(ZX * K1):
                rr_ = tid // i32(K1)
                src = (o + i0 + rr_) * i32(K1) + (tid - rr_ * i32(K1))
                dst = (B + rank * s + i0) * i32(K1) + tid
                bst(
                    fx.Int32(bld(r_resid, src * i32(4), 0, T.i32)),
                    rsrc(wp("ids")),
                    dst * i32(4),
                    0,
                    AUX_SYS,
                )
                bst(
                    fx.Int32(bld(r_wpost, src * i32(4), 0, T.i32)),
                    rsrc(wp("tw")),
                    dst * i32(4),
                    0,
                    AUX_SYS,
                )
            for ii in range_constexpr(ZX):
                row0 = B + rank * s + i0 + i32(ii)
                for kk in range_constexpr(PIT):
                    p = col(tid, kk)
                    y = bf16x8_to_f32(ld[ii][kk])
                    am = amax(y)
                    am = am.maximumf(am.shuffle_xor(i32(1), i32(64)))
                    am = am.maximumf(am.shuffle_xor(i32(2), i32(64)))
                    e8, qs = _e8m0_from_amax(am, max_norm=6.0)
                    bst(
                        fp4_pack(y, qs), rq, row0 * i32(H // 2) + p * i32(4), 0, AUX_SYS
                    )
                    if (lane & i32(3)) == i32(0):
                        bst(
                            fx.Int8(e8),
                            rqs,
                            row0 * i32(H // 32) + p // i32(4),
                            0,
                            AUX_SYS,
                        )
            done_last(k, P_RB, cnt(P_RB, s))
            sa = s // i32(NSBA)
            bh = i0 // sa
            if done_at(  # noqa: SIM102 - preserve FlyDSL branches
                sbh(k, 1, bh), sa // i32(ZX)
            ):
                if tid == i32(0):
                    post_c(k, bh)

        @flyc.jit
        def item_s1(k, j):
            tid = opaque(tid0)
            o, s = chunk(k)
            B = i32(tp) * o
            if const_expr(SXW):
                wait_puts(QX, NSBA, k)
                wait_puts(QXS, NSBA, k)
            wait_puts(QID, 2 * NSBA, k)
            wait_ph(k, P_RB, cnt(P_RB, s))  # own rows' routing copied into the window
            for b_ in range_constexpr(NEB):
                if tid + i32(b_ * NTH) < i32(NE):
                    lds_st(L, i32(LH) + (tid + i32(b_ * NTH)) * i32(4), i32(0))
            gpu.barrier()
            rids = rsrc(wp("ids"))
            r0 = (B + j * i32(SRT)) * i32(K1)
            nr = fx.min((B + i32(tp) * s) * i32(K1) - r0, i32(SRT * K1))
            for kk in range_constexpr((SRT * K1 + NTH - 1) // NTH):
                r = tid + i32(kk * NTH)
                if r < nr:
                    e = fx.Int32(bld(rids, (r0 + r) * i32(4), 0, T.i32, AUX_SYS))
                    if (e >= i32(0)) & (e < i32(NE)):  # masked / invalid ids: no route
                        lds_atomic_add(L, i32(LH) + e * i32(4), i32(1))
            gpu.barrier()
            for b_ in range_constexpr(NEB):
                e = tid + i32(b_ * NTH)
                if e < i32(NE):
                    c = lds_ld_i32(L, i32(LH) + e * i32(4))
                    if c > i32(0):
                        atomic_add_agent(ca(i32(CC_CNT) + k * i32(XST) + e), c)
            done_last(k, P_S1, cnt(P_S1, s))

        @flyc.jit
        def item_s2(k):
            tid = opaque(tid0)
            _o, s = chunk(k)
            wait_ph(k, P_S1, cnt(P_S1, s))
            # route-list offsets (prefix of the counts, each thread its own expert)
            for b_ in range_constexpr(NEB):
                e = tid + i32(b_ * NTH)
                if e < i32(NE):
                    lds_st(
                        L,
                        i32(LH) + e * i32(4),
                        fx.Int32(
                            load_i32_global_agent(ca(i32(CC_CNT) + k * i32(XST) + e))
                        ),
                    )
            gpu.barrier()
            accs = [i32(0)] * NEB
            for e_ in range_constexpr(NE - 1):
                v = lds_ld_i32(L, i32(LH + 4 * e_))
                for b_ in range_constexpr(NEB):
                    accs[b_] = accs[b_] + (i32(e_) < tid + i32(b_ * NTH)).select(
                        v, i32(0)
                    )
            for b_ in range_constexpr(NEB):
                e = tid + i32(b_ * NTH)
                if e < i32(NE):
                    g_st_sys(ca(i32(CC_OFF) + k * i32(XST) + e), accs[b_])
                    lds_st(L, i32(LB2) + e * i32(4), accs[b_])
            gpu.barrier()
            s2_greedy(k)
            done_last(k, P_S2, i32(1))

        @flyc.jit
        def s2_greedy(k):
            tid = opaque(tid0)
            # units -> per-XCD queues: unit g of one round-robin sequence (the shared expert's
            # units, then those of the routed experts with more than UQ units) to queue g % 8,
            # slot g / 8; every other routed expert's units whole to the least-loaded queue (its
            # weights shared in that XCD's L2)
            rut = rsrc(utab)
            ns = (lds_ld_i32(L, i32(LH + 4 * E)) + i32(RG - 1)) // i32(RG)
            soff = lds_ld_i32(L, i32(LB2 + 4 * E))
            cs_ = lds_ld_i32(L, i32(LH + 4 * E))
            for u_ in range(tid, ns, i32(NTH)):
                u = i32(u_)
                ent = ((k * i32(8) + u % i32(8)) * ucap + u // i32(8)) * i32(16)
                bst(i32(E), rut, ent, 0, AUX_SYS)
                bst(soff + u * i32(RG), rut, ent + i32(4), 0, AUX_SYS)
                bst(fx.min(cs_ - u * i32(RG), i32(RG)), rut, ent + i32(8), 0, AUX_SYS)
            # the routed experts' queue / first slot to LDS (wave 0), then every expert's units
            LEQ = LP + 64
            # wave 0: the queue loads and unit counts wave-uniform; key = load * 8 + queue is
            # unique per queue, so the least-loaded (lowest on ties) queue is the one whose key is
            # the minimum; each expert's (queue, first slot) is kept by lane e % 64
            # (a spread expert's entry: bit 31 | its first g)
            if tid < i32(64):
                nuv = [
                    (
                        lds_ld_i32(L, i32(LH) + (tid + i32(64 * j)) * i32(4))
                        + i32(RG - 1)
                    )
                    // i32(RG)
                    for j in range(EPL)
                ]
                g_ = uni(ns)
                qb = [i32(0)] * EPL
                for e_ in range_constexpr(E):
                    nu = i32(rocdl.readlane(T.i32, nuv[e_ // 64], i32(e_ % 64)))
                    sp = nu > i32(UQ)
                    qb[e_ // 64] = ((tid == i32(e_ % 64)) & sp).select(
                        g_ | i32(-(1 << 31)), qb[e_ // 64]
                    )
                    g_ = g_ + sp.select(nu, i32(0))
                ky = [
                    ((g_ - i32(x) + i32(7)) // i32(8)) * i32(8) + i32(x)
                    for x in range(8)
                ]
                for e_ in range_constexpr(E):
                    nu = i32(rocdl.readlane(T.i32, nuv[e_ // 64], i32(e_ % 64)))
                    sp = nu > i32(UQ)
                    nu8 = sp.select(i32(0), nu * i32(8))
                    m_ = fx.min(
                        fx.min(fx.min(ky[0], ky[1]), fx.min(ky[2], ky[3])),
                        fx.min(fx.min(ky[4], ky[5]), fx.min(ky[6], ky[7])),
                    )
                    qb[e_ // 64] = (tid == i32(e_ % 64)).select(
                        sp.select(
                            qb[e_ // 64], (m_ & i32(7)) | ((m_ >> i32(3)) << i32(8))
                        ),
                        qb[e_ // 64],
                    )
                    for x in range_constexpr(8):
                        ky[x] = ky[x] + (ky[x] == m_).select(nu8, i32(0))
                for j in range_constexpr(EPL):
                    lds_st(L, i32(LEQ) + (tid + i32(64 * j)) * i32(4), qb[j])
                if tid == i32(0):
                    for x in range_constexpr(8):
                        lds_st(L, i32(LP + 4 * x), ky[x] >> i32(3))
            gpu.barrier()
            for b_ in range_constexpr(
                -(-E // NTH)
            ):  # (one round per NTH routed experts)
                e = tid if b_ == 0 else tid + i32(b_ * NTH)
                if e < i32(E):
                    c = lds_ld_i32(L, i32(LH) + e * i32(4))
                    off = lds_ld_i32(L, i32(LB2) + e * i32(4))
                    nu = (c + i32(RG - 1)) // i32(RG)
                    qb = lds_ld_i32(L, i32(LEQ) + e * i32(4))
                    spr = qb < i32(0)
                    g0 = qb & i32(0x7FFFFFFF)
                    x_ = qb & i32(0xFF)
                    b0 = (qb >> i32(8)) & i32(0xFFFFFF)
                    for u_ in range(i32(0), nu, i32(1)):
                        u = i32(u_)
                        xs = spr.select((g0 + u) % i32(8), x_)
                        slot = fx.min(
                            spr.select((g0 + u) // i32(8), b0 + u), ucap - i32(1)
                        )
                        ent = ((k * i32(8) + xs) * ucap + slot) * i32(16)
                        bst(e, rut, ent, 0, AUX_SYS)
                        bst(off + u * i32(RG), rut, ent + i32(4), 0, AUX_SYS)
                        bst(
                            fx.min(c - u * i32(RG), i32(RG)),
                            rut,
                            ent + i32(8),
                            0,
                            AUX_SYS,
                        )
            gpu.barrier()
            if tid < i32(8):
                g_st_sys(
                    ca(i32(C_UN) + tid * i32(32) + k * i32(256)),
                    fx.min(lds_ld_i32(L, i32(LP) + tid * i32(4)), ucap),
                )

        @flyc.jit
        def item_s3(k, j):
            tid = opaque(tid0)
            o, s = chunk(k)
            B = i32(tp) * o
            wait_ph(k, P_S2, i32(1))  # the route-list offsets
            for b_ in range_constexpr(NEB):
                if tid + i32(b_ * NTH) < i32(NE):
                    lds_st(L, i32(LH) + (tid + i32(b_ * NTH)) * i32(4), i32(0))
            gpu.barrier()
            rids, rtw = rsrc(wp("ids")), rsrc(wp("tw"))
            r0 = (B + j * i32(SRT)) * i32(K1)
            nr = fx.min((B + i32(tp) * s) * i32(K1) - r0, i32(SRT * K1))
            NRI = (SRT * K1 + NTH - 1) // NTH
            es, rk, oks = [], [], []
            for kk in range_constexpr(NRI):
                r = tid + i32(kk * NTH)
                e = fx.Int32(
                    bld(rids, (fx.min(r, nr - i32(1)) + r0) * i32(4), 0, T.i32, AUX_SYS)
                )
                ok = (
                    (r < nr) & (e >= i32(0)) & (e < i32(NE))
                )  # masked / invalid ids: no route
                e = ok.select(e, i32(0))
                es.append(e)
                oks.append(ok)
                rk.append(
                    lds_atomic_add(L, i32(LH) + e * i32(4), ok.select(i32(1), i32(0)))
                )
            gpu.barrier()
            for b_ in range_constexpr(NEB):
                e = tid + i32(b_ * NTH)
                if e < i32(NE):
                    c = lds_ld_i32(L, i32(LH) + e * i32(4))
                    base = fx.Int32(
                        atomic_add_agent(ca(i32(CC_FILL) + k * i32(XST) + e), c)
                    )
                    lds_st(
                        L,
                        i32(LB2) + e * i32(4),
                        fx.Int32(
                            load_i32_global_agent(ca(i32(CC_OFF) + k * i32(XST) + e))
                        )
                        + base,
                    )
            gpu.barrier()
            rl, rlw = rsrc(lrix), rsrc(lwt)
            for kk in range_constexpr(NRI):
                if oks[kk]:
                    r = tid + i32(kk * NTH)
                    pos = lds_ld_i32(L, i32(LB2) + es[kk] * i32(4)) + rk[kk]
                    lo_ = lbase(k) + pos
                    bst(
                        (r0 + r) - B * i32(K1), rl, lo_ * i32(4), 0, AUX_SYS
                    )  # chunk-local route index
                    bst(
                        fx.Int32(bld(rtw, (r0 + r) * i32(4), 0, T.i32, AUX_SYS)),
                        rlw,
                        lo_ * i32(4),
                        0,
                        AUX_SYS,
                    )
            done_last(k, P_S3, cnt(P_S3, s))

        @flyc.jit
        def item_u_own(k, j0):
            tid = opaque(tid0)
            o, s = chunk(k)
            B = i32(tp) * o
            wait_ph(k, P_S3, cnt(P_S3, s))
            if const_expr(
                not SXW
            ):  # the plan only waited for the routing: the rows now
                wait_puts(QX, NSBA, k)
                wait_puts(QXS, NSBA, k)
                gpu.barrier()
            nu = fx.Int32(
                load_i32_global_agent(ca(i32(C_UN) + k * i32(256) + xc * i32(32)))
            )
            j = j0
            if j < nu:
                rut = rsrc(utab)
                ent = ((k * i32(8) + xc) * ucap + j) * i32(16)
                e = uni(fx.Int32(bld(rut, ent, 0, T.i32, AUX_SYS)))
                r0 = uni(fx.Int32(bld(rut, ent + i32(4), 0, T.i32, AUX_SYS)))
                rows = uni(fx.Int32(bld(rut, ent + i32(8), 0, T.i32, AUX_SYS)))
                rl = rsrc(lrix + fx.Int64(lbase(k) * i32(4)))
                rlw = rsrc(lwt + fx.Int64(lbase(k) * i32(4)))
                rx = rsrc(wp("xq") + fx.Int64(B * i32(H // 2)))
                rxs = rsrc(wp("xs") + fx.Int64(B * i32(H // 32)))
                nrt = i32(tp) * s * i32(K1)
                r_rt = rsrc(routes + rbase(k), nrt * i32(H * 2))
                unit_body(
                    L,
                    tid,
                    e,
                    r0,
                    rows,
                    rl,
                    rlw,
                    rx,
                    rxs,
                    w1,
                    w2,
                    rsrc(w1s),
                    rsrc(w2s),
                    r_rt,
                    nrt * i32(H * 2),
                )
                # release: the unit's route rows out of this XCD's L2
                wait_vm(0)
                gpu.barrier()
                if tid == i32(0):
                    l2_writeback()
                done_last(k, P_U, i32(-1))  # counts units done (no single last)

        @flyc.jit
        def item_rq(k, j):
            tid = opaque(tid0)
            lane = tid % i32(64)
            o, s = chunk(k)
            B = i32(tp) * o
            wait_ph(k, P_S2, i32(1))  # the unit plan (C_UN)
            un = i32(0)
            for x in range_constexpr(8):
                un = un + fx.Int32(
                    load_i32_global_agent(ca(i32(C_UN + 32 * x) + k * i32(256)))
                )
            wait_ph(k, P_U, un)
            rx = rsrc(routes + rbase(k))
            rd = rsrc(wp("eq"))
            rids_ = rsrc(wp("ids"))

            def rvalid(
                wrow, t
            ):  # (token, slot) routes to an expert: else its route row is not written
                e_ = fx.Int32(
                    bld(
                        rids_,
                        ((B + wrow) * i32(K1) + i32(t)) * i32(4),
                        0,
                        T.i32,
                        AUX_SYS,
                    )
                )
                return (e_ >= i32(0)) & (e_ < i32(NE))

            sb = s // i32(NSB)
            nb = (i32(tp) * sb) // i32(ZQ)
            jx = il(j, cnt(P_RQ, s))
            b = jx // nb
            rem = (jx - b * nb) * i32(ZQ)
            pd = rem // sb
            w0 = pd * s + b * sb + (rem - pd * sb)  # first (dest rank, row) of the item
            own = pd == rank

            nrt_ = i32(tp) * s * i32(K1)

            def rq_load(
                ii,
            ):  # token ii of the item: its K1 route rows (this thread's columns)
                if const_expr(R8):
                    return [
                        [
                            (
                                bld(
                                    rx,
                                    ((w0 + i32(ii)) * i32(K1) + i32(t)) * i32(H)
                                    + col(tid, kk) * i32(8),
                                    0,
                                    T.vec(2, T.i32),
                                    CAUX,
                                ),
                                bld(
                                    rx,
                                    nrt_ * i32(H)
                                    + ((w0 + i32(ii)) * i32(K1) + i32(t)) * i32(H // 32)
                                    + col(tid, kk) // i32(4),
                                    0,
                                    T.i8,
                                    CAUX,
                                ),
                            )
                            for t in range(K1)
                        ]
                        for kk in range(PIT)
                    ]
                return [
                    [
                        bld(
                            rx,
                            (
                                at_col(
                                    ((w0 + i32(ii)) * i32(K1) + i32(t)) * i32(H),
                                    tid,
                                    kk,
                                )
                            )
                            * i32(2),
                            0,
                            T.vec(4, T.i32),
                            CAUX,
                        )
                        for t in range(K1)
                    ]
                    for kk in range(PIT)
                ]

            def rq_store(ii, ld):
                wrow = w0 + i32(ii)
                vt = [rvalid(wrow, t) for t in range(K1)]
                for kk in range_constexpr(PIT):
                    p = col(tid, kk)
                    acc = [fx.Float32(0.0)] * 8
                    for t in range_constexpr(K1):
                        v = (
                            fp8x8_decode(ld[kk][t])
                            if const_expr(R8)
                            else bf16x8_to_f32(ld[kk][t])
                        )
                        acc = [vt[t].select(a_ + b_, a_) for a_, b_ in zip(acc, v)]
                    yb = pack_bf16x8(acc)
                    if own:
                        bst(
                            yb,
                            r_y,
                            ((o + wrow - rank * s) * i32(H) + p * i32(8)) * i32(2),
                            0,
                            AUX_SYS,
                        )
                    else:
                        e_store(bf16x8_to_f32(yb), rd, B + wrow, p, lane)

            # token pipeline: the next D tokens' loads in flight while one is summed (D = ZQ:
            # every load of the item first)
            D = RQD or ZQ
            rqp = [rq_load(ii) for ii in range(min(D, ZQ))]
            for ii in range_constexpr(ZQ):
                if const_expr(ii + D < ZQ):
                    rqp.append(rq_load(ii + D))
                rq_store(ii, rqp[ii])
            done_last(k, P_RQ, cnt(P_RQ, s))
            if done_at(sbc(k, 0, b), nb):  # noqa: SIM102 - preserve FlyDSL branches
                if tid == i32(0):
                    tn = k * i32(NSB) + b
                    take_turn(QE, tn)
                    for p_ in range_constexpr(tp):
                        if i32(p_) != rank:
                            put(
                                i32(p_),
                                "erecv",
                                (B + rank * s + b * sb) * i32(RBE),
                                "eq",
                                (B + i32(p_) * s + b * sb) * i32(RBE),
                                sb * i32(RBE),
                                QE,
                            )
                    give_turn(QE, tn)

        def rf_head(k, j):
            # an RF item (ZQ rows of chunk k, from i0, in sub-block b), once every peer's E
            # rows of its sub-block and this rank's topk sums of it are in:
            # (tid, o, s, B, sb, i0, b, the received E rows)
            tid = opaque(tid0)
            o, s = chunk(k)
            B = i32(tp) * o
            sb = s // i32(NSB)
            jx = il(j, cnt(P_RF, s))
            i0 = jx * i32(ZQ)
            b = i0 // sb
            wait_puts_n(QE, k * i32(NSB) + b + i32(1))
            wait_at(sbc(k, 0, b), (i32(tp) * sb) // i32(ZQ))
            return tid, o, s, B, sb, i0, b, rsrc(wp("erecv"))

        def y_sum(ly, lsrc):
            # y of a row's 8 columns: this rank's partial (ly) + every peer's (lsrc(src): its
            # E value, loaded here or earlier)
            f = bf16x8_to_f32(ly)
            for src_ in range_constexpr(tp):
                f = sum_live(f, e_val(lsrc(src_)), i32(src_) != rank)
            return f

        @flyc.jit
        def item_rf(k, j):
            tid, o, s, B, sb, i0, b, rv = rf_head(k, j)
            rq, rqs = rsrc(wp("q")), rsrc(wp("qs"))
            one = fx.Float32(1.0)
            # every load of the item first (one memory round trip), then the math
            ld = []
            for ii in range_constexpr(ZQ):
                i = i0 + i32(ii)
                lo = ((o + i) * i32(H)) * i32(2)
                row = []
                for kk in range_constexpr(PIT):
                    off = lo + col(tid, kk) * i32(16)
                    row.append(
                        (
                            bld(r_y, off, 0, T.vec(4, T.i32), CAUX),
                            [
                                e_ld(rv, B + i32(src_) * s + i, col(tid, kk))
                                for src_ in range(tp)
                            ],
                            bld(r_res, off, 0, T.vec(4, T.i32), 0),
                        )
                    )
                ld.append(row)
            fss, sss = [], []
            for ii in range_constexpr(ZQ):
                i = i0 + i32(ii)
                lo = ((o + i) * i32(H)) * i32(2)
                fs = []
                ss = fx.Float32(0.0)
                for kk in range_constexpr(PIT):
                    off = lo + col(tid, kk) * i32(16)
                    ly, lr, lres = ld[ii][kk]
                    f = y_sum(ly, lambda src_, lr=lr: lr[src_])
                    # the MoE output y (this rank's rows)
                    f = [fx.Float32(x.to(fx.BFloat16).to(fx.Float32)) for x in f]
                    live = own_col(tid, kk)
                    if const_expr(live is None):
                        bst(pack_bf16x8(f), r_norm, off, 0)
                    else:
                        yv = pack_bf16x8(f)
                    f = [x + yy for x, yy in zip(f, bf16x8_to_f32(lres))]
                    f = [fx.Float32(x.to(fx.BFloat16).to(fx.Float32)) for x in f]
                    if const_expr(live is None):
                        bst(pack_bf16x8(f), r_out, off, 0)
                        for x in f:
                            ss = ss + x * x
                    else:
                        if live:
                            bst(yv, r_norm, off, 0)
                            bst(pack_bf16x8(f), r_out, off, 0)
                        for x in f:
                            ss = ss + live.select(x * x, fx.Float32(0.0))
                    fs.append(f)
                fss.append(fs)
                sss.append(ss.bitcast(fx.Int32))
            tots = blk_red_vec(L, tid, sss, f32_bits_add)
            xss, ams = [], []
            for ii in range_constexpr(ZQ):
                rcp = fx.Float32(
                    fmath.rsqrt(
                        tots[ii].bitcast(fx.Float32) / fx.Float32(float(H))
                        + fx.Float32(float(eps))
                    )
                )
                xs_, am = [], fx.Float32(0.0)
                for kk in range_constexpr(PIT):
                    wv = norm_w(r_wnext, col(tid, kk))
                    xk = [
                        fx.Float32(
                            fx.Float32(gate_w(x * rcp, wk))
                            .to(fx.BFloat16)
                            .to(fx.Float32)
                        )
                        for x, wk in zip(fss[ii][kk], wv)
                    ]
                    live = own_col(tid, kk)
                    if const_expr(live is None):
                        am = am.maximumf(amax(xk))
                    else:
                        am = am.maximumf(live.select(amax(xk), fx.Float32(0.0)))
                    xs_.append(xk)
                xss.append(xs_)
                ams.append(am.bitcast(fx.Int32))
            amxs = blk_red_vec(L, tid, ams, f32_bits_max)
            for ii in range_constexpr(ZQ):
                i = i0 + i32(ii)
                scl = amxs[ii].bitcast(fx.Float32).maximumf(
                    fx.Float32(1e-10)
                ) / fx.Float32(448.0)
                inv = one / scl
                g = rank * m + o + i
                for kk in range_constexpr(PIT):
                    p = col(tid, kk)
                    qv = [x * inv for x in xss[ii][kk]]
                    d = fx.Vector.from_elements(
                        [fp8x4_pack(qv[0:4], one), fp8x4_pack(qv[4:8], one)],
                        fx.Int32,
                    )
                    live = own_col(tid, kk)
                    if const_expr(live is None):
                        bst(d, rq, g * i32(H) + p * i32(8), 0, AUX_SYS)
                    else:
                        if live:
                            bst(d, rq, g * i32(H) + p * i32(8), 0, AUX_SYS)
                if tid == i32(0):
                    bst(scl.bitcast(fx.Int32), rqs, g * i32(4), 0, AUX_SYS)
            done_last(k, P_RF, cnt(P_RF, s))
            # (nested ifs: preserve FlyDSL branches)
            if done_at(sbc(k, 1, b), sb // i32(ZQ)) & (  # noqa: SIM102
                i32(int(tailq == 1)) == i32(1)
            ):
                if tid == i32(0):
                    g0 = rank * m + o + b * sb
                    tn = k * i32(NSB) + b
                    take_turn(QQ, tn)
                    take_turn(QS, tn)
                    for p_ in range_constexpr(tp):
                        if i32(p_) != rank:
                            put(
                                i32(p_),
                                "q",
                                g0 * i32(H),
                                "q",
                                g0 * i32(H),
                                sb * i32(H),
                                QQ,
                            )
                            put(
                                i32(p_),
                                "qs",
                                g0 * i32(4),
                                "qs",
                                g0 * i32(4),
                                sb * i32(4),
                                QS,
                            )
                    give_turn(QQ, tn)
                    give_turn(QS, tn)

        @flyc.jit
        def item_rf_y(k, j):
            # io without tail: y = own partial + the peers' MoE-output rows (this rank's rows);
            # tailq 3: y rows to the "q" region (global row rank * m + ..), each sub-block then
            # put to every peer
            tid, o, s, B, sb, i0, b, rv = rf_head(k, j)
            ryg = rsrc(wp("q"))
            for ii in range_constexpr(ZQ):
                i = i0 + i32(ii)
                lo = ((o + i) * i32(H)) * i32(2)
                for kk in range_constexpr(PIT):
                    off = lo + col(tid, kk) * i32(16)
                    f = y_sum(
                        bld(r_y, off, 0, T.vec(4, T.i32), CAUX),
                        lambda src_, i=i, kk=kk: e_ld(
                            rv, B + i32(src_) * s + i, col(tid, kk)
                        ),
                    )
                    if const_expr(tailq == 3):
                        bst(
                            pack_bf16x8(f),
                            ryg,
                            (rank * m + o + i) * i32(H * 2) + col(tid, kk) * i32(16),
                            0,
                            AUX_SYS,
                        )
                    else:
                        bst(pack_bf16x8(f), r_norm, off, 0)
            done_last(k, P_RF, cnt(P_RF, s))
            if const_expr(tailq == 3):  # noqa: SIM102 - preserve FlyDSL branches
                if done_at(sbc(k, 1, b), sb // i32(ZQ)):  # noqa: SIM102
                    if tid == i32(0):
                        g0 = rank * m + o + b * sb
                        tn = k * i32(NSB) + b
                        take_turn(QQ, tn)
                        for p_ in range_constexpr(tp):
                            if i32(p_) != rank:
                                put(
                                    i32(p_),
                                    "q",
                                    g0 * i32(H * 2),
                                    "q",
                                    g0 * i32(H * 2),
                                    sb * i32(H * 2),
                                    QQ,
                                )
                        give_turn(QQ, tn)

        @flyc.jit
        def item_gw():
            for k in range_constexpr(C):
                wait_ph(i32(k), P_RF, cnt(P_RF, chunk(i32(k))[1]))
            if const_expr(tailq in (1, 3)):
                wait_puts(QQ, NSB, i32(C - 1))
            if const_expr(tailq == 1):
                wait_puts(QS, NSB, i32(C - 1))
            gpu.barrier()
            # every put of this launch landed: the next launch's per-(peer, queue) base, into
            # the other buffer (a late wait of this launch still reads this launch's base).
            # One thread per peer (tid, not lane: every wave would store it)
            if tid < i32(tp):  # noqa: SIM102 - preserve FlyDSL branches
                if tid != rank:
                    for qi in range_constexpr(len(QUEUES)):
                        q_ = QUEUES[qi]
                        addr = (consumed ^ fx.Int64(tp * NQ * 4)) + fx.Int64(
                            (tid * i32(NQ) + i32(q_)) * i32(4)
                        )
                        g_st_sys(addr, sig_base(tid, q_) + i32(C * PPC[q_]))

        # ======================= the work loop =======================
        def decode(slices, t):
            kind_r = i32(-1)
            k_r = i32(0)
            jj = i32(0)
            for si in range_constexpr(len(slices)):
                k, kind, lo_, b0, b1 = slices[si]
                hit = (t >= b0) & (t < b1)
                kind_r = hit.select(i32(kind), kind_r)
                k_r = hit.select(i32(k), k_r)
                jj = hit.select(lo_ + (t - b0), jj)
            return kind_r, k_r, jj

        rf_item = item_rf_y if tailq in (0, 3) else item_rf

        t = claim()
        alive_ = t < nall
        while alive_:
            kind_r, k_r, jj = decode(
                slices_of([chunk_v(i32(k)) for k in range(C)])[0], t
            )
            kind_r = uni(kind_r)
            k_r = uni(k_r)
            jj = uni(jj)
            if kind_r == i32(P_RB):
                item_rx(k_r, jj)
            elif kind_r == i32(P_S1):
                item_s1(k_r, jj)
            elif kind_r == i32(P_S2):
                item_s2(k_r)
            elif kind_r == i32(P_S3):
                item_s3(k_r, jj)
            elif (kind_r == i32(P_U)) | (kind_r == i32(P_UB)):
                item_u_own(k_r, jj)
            elif kind_r == i32(P_RQ):
                item_rq(k_r, jj)
            elif kind_r == i32(P_RF):
                rf_item(k_r, jj)
            if t == gw_lo:
                item_gw()
            gpu.barrier()
            t = claim()
            alive_ = t < nall

    @flyc.jit
    def launch(
        wb: fx.Int64,
        pbases: fx.Int64,
        dc: fx.Int64,
        consumed: fx.Int64,
        offs: fx.Int64,
        part: fx.Int64,
        resid: fx.Int64,
        wpost: fx.Int64,
        wnext: fx.Int64,
        res: fx.Int64,
        normed: fx.Int64,
        resout: fx.Int64,
        ybuf: fx.Int64,
        routes: fx.Int64,
        lrix: fx.Int64,
        lwt: fx.Int64,
        utab: fx.Int64,
        w1: fx.Int64,
        w1s: fx.Int64,
        w2: fx.Int64,
        w2s: fx.Int64,
        ctrl: fx.Int64,
        chunks: fx.Int64,
        rank: fx.Int32,
        m: fx.Int32,
        ucap: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            wb,
            pbases,
            dc,
            consumed,
            offs,
            part,
            resid,
            wpost,
            wnext,
            res,
            normed,
            resout,
            ybuf,
            routes,
            lrix,
            lwt,
            utab,
            w1,
            w1s,
            w2,
            w2s,
            ctrl,
            chunks,
            rank,
            m,
            ucap,
        ).launch(grid=(fx.Int64(grid), 1, 1), block=(NTH, 1, 1), stream=stream)

    launch._bigm_spec = spec
    return launch
