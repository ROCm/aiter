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
import os
import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ..communication_ops_utils import atomic_add_agent, load_i32_global_agent
from ..mxfp4_gemm_common import _e8m0_from_amax
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
    i8x4_pack,
    i8x4_unpack,
    i32,
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

# SDMA queues per peer (MORI spreads a peer's queues over its two engines by parity; queues 0
# and 7 carry no puts)
QE, QX, QQ, QXS, QS, QID = 1, 2, 3, 4, 5, 6
NQ = 8
QUEUES = (QE, QX, QQ, QXS, QS, QID)
PUTS_PER_CHUNK = {QE: 1, QX: 1, QQ: 1, QXS: 1, QS: 1, QID: 2}

# phase kinds (per chunk), in item order within a chunk (their done-counter slots; 0 and 2
# unused)
P_RB, P_S1, P_S2, P_S3, P_U, P_RQ, P_RF, P_UB = 1, 3, 4, 5, 6, 7, 8, 9
NKIND = 16  # done-counter slots per chunk
ZQ = 4  # rows per RQ / RF item
SRT = 32  # tokens per sort item
# experts with more units than this (in a chunk) go round-robin over the XCD queues, right after
# the shared expert's units; the others whole to the least-loaded queue. A queue then holds at
# most ceil(units / 8) + UQ (see unit_cap) for any routing (hot experts included)
UQ = 16

# ctrl ints (zeroed by the host before every launch)
C_TK = 0  # per-XCD tickets, 32 apart
C_TURN = 256  # per-queue post turns, 32 apart
C_PH = 512  # [chunk][NKIND] done counters, 32 apart
C_CNT = 4096  # [chunk][NE] routes per expert (256 apart)
C_OFF = 6144  # [chunk][NE] route-list offset of each expert
C_FILL = 8192  # [chunk][NE] routes listed so far
C_UN = 10240  # [chunk][8] units of each XCD (32 apart)
C_SB = 13312  # [chunk][RQ, RF][sub-block] done items, 8 apart
C_SBH = 18432  # [chunk][-, RB][head sub-block] done items, 8 apart
C_XW = 28928  # NE > 256: C_CNT / C_OFF / C_FILL here instead, [chunk][NE] 512 apart
CTRL_INTS = C_XW + 3 * 8 * 512
C_ERR = CTRL_INTS  # sticky watchdog word after the per-launch ctrl block (not zeroed per launch)


def _l2_writeback():
    """this XCD's dirty L2 lines to memory (device-scope release of plain stores)"""
    from flydsl._mlir.dialects import llvm as _llvm

    _llvm.InlineAsmOp(
        None, [], "buffer_wbl2 sc1\ns_waitcnt vmcnt(0)", "", has_side_effects=True
    )


def i8_row_bytes(H):
    return H + H // 8


def _i4x8_pack(f, inv):
    w = i32(0)
    for k in range_constexpr(8):
        q = fx.Int32(fmath.roundeven(f[k] * inv))
        q = fx.max(fx.min(q, i32(7)), i32(-8))
        w = w | ((q & i32(0xF)) << i32(4 * k))
    return w


def _i4x8_unpack(d, sc):
    out = []
    for k in range_constexpr(8):
        b = (fx.Int32(d) << i32(28 - 4 * k)) >> i32(28)
        out.append(b.to(fx.Float32) * sc)
    return out


def unit_cap(tp, s, K1, NE, RG):
    """Upper bound of the units one XCD gets for a chunk of s rows per rank."""
    ubound = -(-tp * s * K1 // RG) + NE
    return -(-ubound // 8) + UQ + 1


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
            return _i4x8_unpack(d, (sb << i32(16)).bitcast(fx.Float32))
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
                _i4x8_pack(f, fx.Float32(1.0) / sc),
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
                    _l2_writeback()
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


# ---- disk cache of the compiled big-M kernels ----
# FlyDSL keeps kernels that link external bitcode (link_extern: the SDMA device functions here)
# out of its disk cache -- its key does not cover the bitcode -- so every process would compile
# every big-M variant again (~86 s each, all ranks waiting). The artifact itself pickles fine
# (no post-load processors), so this keeps it on disk keyed by the compile arguments plus the
# digest of everything else it is built from: these kernel sources, the SDMA bitcode, the
# FlyDSL compiler (its own cache key) and the GPU arch.
_DIGEST = None


def _bigm_digest():
    global _DIGEST
    if _DIGEST is None:
        import glob
        import hashlib

        from aiter.jit.utils.chip_info import get_gfx

        from .sdma import sdma_bitcode

        here = os.path.dirname(os.path.abspath(__file__))
        h = hashlib.sha256(get_gfx().encode())
        srcs = sorted(glob.glob(os.path.join(here, "*.py"))) + sorted(
            glob.glob(os.path.join(here, "..", "*.py"))
        )
        for f in srcs + [sdma_bitcode(6)]:
            with open(f, "rb") as fh:
                h.update(fh.read())
        try:
            from flydsl.compiler.jit_function import _flydsl_key

            h.update(_flydsl_key().encode())
        except Exception:  # noqa: BLE001 (older / newer FlyDSL: its version instead)
            import flydsl

            h.update(str(getattr(flydsl, "__version__", "")).encode())
        _DIGEST = h.hexdigest()
    return _DIGEST


def _bigm_cache_dir():
    """FlyDSL's cache root (FLYDSL_RUNTIME_CACHE_DIR) / aiter_bigm; None when FlyDSL caching is off."""
    try:
        from flydsl.compiler.jit_function import env as fenv

        root = fenv.runtime.cache_dir if fenv.runtime.enable_cache else None
    except Exception:  # noqa: BLE001
        root = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR") or os.path.expanduser(
            "~/.flydsl/cache"
        )
    return os.path.join(root, "aiter_bigm") if root else None


_WARNED = set()


def _warn_once(key, msg, *args):
    if key not in _WARNED:
        import logging

        _WARNED.add(key)
        logging.getLogger("aiter").warning(msg, *args)


def _private(path):
    """path (the cache directory or an entry) is this user's and only writable by them: a
    pickled entry runs code when loaded, so a shared / writable-by-others cache is skipped
    """
    import stat

    try:
        st = os.stat(path)
    except OSError:
        return False
    return st.st_uid == os.getuid() and not st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


def _bigm_cached(launch, args):
    """Collective-free: make launch(*args)'s compiled artifact resident -- from this process's
    memory, else the disk cache, else compiled (no kernel dispatch) and written to the disk
    cache -- and its code object loaded. Returns how it was obtained."""
    import hashlib
    import pickle
    import re
    import tempfile

    from ..tensor_shim import _preload_compiled

    try:  # FlyDSL's in-process key of these arguments (private API: on any change, just compile)
        launch._ensure_sig()
        launch._ensure_cache_manager()
        bound = launch._sig.bind(*args)
        bound.apply_defaults()
        ck = launch._build_full_cache_key(
            bound.arguments, effective_hints=launch._effective_compile_hints()
        )
        mem = launch._mem_cache
    except Exception:  # noqa: BLE001
        _warn_once(
            "api",
            "MegaMoeTP big-M: FlyDSL's cache API changed (private names): no disk cache, "
            "every process compiles its big-M kernels (minutes each)",
        )
        _preload_compiled(launch, *args)
        return "compiled (no disk cache)"
    if ck in mem:
        mem[ck]._get_func_exe()
        return "memory"
    d = _bigm_cache_dir()
    path = None
    if d:
        sig = re.sub(r" at 0x[0-9a-f]+", "", str(ck))
        name = hashlib.sha256(
            f"{launch._bigm_spec}|{_bigm_digest()}|{sig}".encode()
        ).hexdigest()[:32]
        path = os.path.join(d, f"bigm_{name}.pkl")
        if os.path.isdir(d) and not _private(d):
            _warn_once(
                d,
                "MegaMoeTP big-M: disk cache %s is not private to this user "
                "(group / others writable or another owner): not used",
                d,
            )
            path = None
        elif os.path.isfile(path) and not _private(path):
            path = None
        if path and os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    art = pickle.load(f)
                art._get_func_exe()  # code object loaded here, not in the first launch
                mem[ck] = art
                return "disk"
            except (
                Exception  # noqa: BLE001 (stale / corrupt entry: compile again)
            ) as e:
                import logging

                logging.getLogger("aiter").warning(
                    "MegaMoeTP big-M: disk cache %s unusable (%s)", path, e
                )
    _preload_compiled(launch, *args)
    art = mem.get(ck)
    if (
        art is None
        and launch._last_compiled is not None
        and launch._last_compiled[0] == ck
    ):
        art = launch._last_compiled[1]
    if art is None:
        return "compiled (artifact not found)"
    art._get_func_exe()
    # scratch spills: the register file is exhausted (seen with the waits timing out)
    m_ = re.search(r"private_segment_fixed_size = (\d+)", getattr(art, "ir", "") or "")
    if m_ and int(m_.group(1)):
        import logging

        logging.getLogger("aiter").warning(
            "MegaMoeTP big-M: kernel %s spills to scratch (%s B per lane): lower its "
            "register use (e.g. AITER_BIGM_RQD=1 / AITER_BIGM_R8=1)",
            launch._bigm_spec[:120],
            m_.group(1),
        )
    if path:
        try:
            os.makedirs(d, mode=0o700, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                pickle.dump(art, f)
            os.replace(tmp, path)  # atomic: concurrent ranks write the same bytes
        except Exception as e:  # noqa: BLE001
            import logging

            logging.getLogger("aiter").warning(
                "MegaMoeTP big-M: disk cache write %s failed (%s)", path, e
            )
    return "compiled"


def bigm_tuned(*, model_dim, inter_dim, experts, topk, tp, tokens, device=None):
    """The tuned plan of the nearest tuned batch (<= tokens, else the smallest;
    configs/mega_moe_tp_bigm_tuned.csv, see tuned.py): {"chunks": [..], "opts":
    {"AITER_BIGM_*": ..}, "tokens": n} or None."""
    from aiter.jit.core import AITER_CONFIGS

    from .tuned import device_key, nearest, tuned_rows

    key = {
        **device_key(device),
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "experts": experts,
        "topk": topk,
        "tp": tp,
    }
    r = nearest(
        tuned_rows(AITER_CONFIGS.AITER_CONFIG_MEGA_MOE_TP_BIGM_FILE, key, "tokens"),
        tokens,
        clamp=True,
    )
    if r is None:
        return None
    opts = dict(kv.split("=", 1) for kv in r.get("opts", "").split(";") if kv)
    return {
        "chunks": [int(x) for x in r["chunks"].split(",")],
        "opts": opts,
        "tokens": int(r["tokens"]),
    }


CMAX = 7  # chunks per launch (ctrl layout)
PAD = 64  # the kernel's local rows (and chunk rows) are multiples of this (moe() pads)
# rows per unit (AITER_BIGM_RG), one CTA (NW waves) per CU: the validated geometry (64-row
# units: 16% slower at 16384 tokens and failing with two CTAs per CU / three chunks)
UNIT_ROWS = 128


_HIP = None


def _hip():
    """the HIP runtime torch uses (ctypes): stream write / wait value"""
    global _HIP
    if _HIP is None:
        import ctypes

        import torch

        torch.cuda.init()
        with open("/proc/self/maps") as maps:
            path = next(l.split()[-1] for l in maps if "libamdhip64" in l)
        lib = ctypes.CDLL(path)
        lib.hipStreamWriteValue32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint,
        ]
        lib.hipStreamWaitValue32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint,
            ctypes.c_uint32,
        ]
        lib.hipStreamWriteValue32.restype = lib.hipStreamWaitValue32.restype = (
            ctypes.c_int
        )
        _HIP = lib
    return _HIP


def _hip_check(err):
    if err != 0:
        raise RuntimeError(f"MegaMoeTP big-M: HIP stream value op failed ({err})")


def _kcopy(dst, src):
    """dst <- src by a shader kernel (dst: a peer's memory). Not copy_: a contiguous copy_ is a
    hipMemcpyAsync, which ROCm runs on the SDMA engine of that xGMI link -- the engines the
    big-M kernel's MORI queues use; ROCr's own queue there slows every later big-M put.
    """
    import torch

    torch.mul(src, 1, out=dst)


@functools.cache
def _compile_signal_read(tp):
    """launch(dc, out, stream): out [tp][NQ] int32 <- the received signal count per (sender,
    queue) (MORI's signal slots, low 32 bits)"""
    from flydsl.compiler.extern_link import link_extern
    from flydsl.expr.extern import ffi

    from .sdma import sdma_bitcode

    sig_slot = link_extern(
        ffi("cco_sdma_signal_slot", ["uint64", "int32", "int32"], "uint64"),
        bitcode_path=sdma_bitcode(6),
    )

    @flyc.kernel(name=f"bigm_signal_read_tp{tp}", known_block_size=[64, 1, 1])
    def kernel(dc: fx.Int64, out: fx.Int64):
        lane = fx.Int32(gpu.thread_id("x"))
        if lane < i32(tp * NQ):
            slot = fx.Int64(sig_slot(dc, lane // i32(NQ), lane % i32(NQ)))
            g_st_sys(out + fx.Int64(lane * i32(4)), g_ld_sys(slot))

    @flyc.jit
    def launch(dc: fx.Int64, out: fx.Int64, stream: fx.Stream):
        kernel(dc, out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


class _CudaBuf:
    """__cuda_array_interface__ of raw device bytes (a peer's arena as mapped here)."""

    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class MegaMoeTPBigM:
    """Host side of the big-M dispatch: the SDMA window (same regions on every rank),
    scratch buffers and the launch.
    moe(x [m, H], topk_weights / topk_ids [m, K1], out, tail) (MegaMoeTP.forward, ag_rs)
    -> y [m, H], with tail = (res_in, res_out, norm_w) (y, q [T, H] fp8, scales [T]).
    Per batch the chunk plan and options come from the tuned csv (bigm_tuned), else defaults.
    """

    REGIONS = ("xq", "xs", "ids", "tw", "eq", "erecv", "q", "qs")

    def __init__(
        self,
        *,
        hidden,
        inter,
        experts,
        topk,
        max_tokens,
        eps,
        gemma,
        swiglu_limit,
        w1,
        w1_scale,
        w2,
        w2_scale,
        group=None,
        device=None,
        opts=None,
        act="swiglu",
        ygather=False,
    ):
        import torch
        import torch.distributed as dist

        from .sdma import SdmaComm

        self.H, self.I, self.E, self.K = hidden, inter, experts, topk
        self.K1, self.NE = topk + 1, experts + 1
        self.tp = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.Tmax = max_tokens
        self.act = act  # the units' activation ("swiglu", "silu")
        self.ygather = bool(ygather)  # moe(gather=True): the "q" region holds bf16 rows
        self._plans = {}
        self._pads = None
        self.cfg = (float(eps), bool(gemma), float(swiglu_limit))
        self.opts = tuple(sorted((opts or {}).items()))  # tuned AITER_BIGM_* options
        self._check_geom(dict(self.opts))
        self.w = (
            w1.view(torch.uint8),
            w1_scale.view(torch.uint8),
            w2.view(torch.uint8),
            w2_scale.view(torch.uint8),
        )
        H, T, K1, tp = hidden, max_tokens, self.K1, self.tp
        RB = i8_row_bytes(H)
        # the kernel's 32-bit offsets into these regions (the route rows: per chunk, _chunks)
        for what, n in (("E rows", T * RB), ("y gather rows", T * H * 2)):
            if n >= 1 << 31:
                raise ValueError(
                    f"MegaMoeTP big-M: {what} of {T} tokens exceed 32-bit offsets"
                )
        sizes = {
            "xq": T * H // 2,
            "xs": T * H // 32,
            "ids": T * K1 * 4,
            "tw": T * K1 * 4,
            "eq": T * RB,
            "erecv": T * RB,
            "q": T * H * (2 if self.ygather else 1),
            "qs": T * 4,
        }
        sd = self.sd = SdmaComm(group=group, num_queues=NQ, device=device)
        for r_ in self.REGIONS:
            sd.reserve(r_, sizes[r_])
        # gather_rows: every rank's bf16 tail rows (not a kernel region)
        sd.reserve("brows", 2 * T * H * 2)  # two buffers in turn
        sd.reserve("bflag", tp * 64)  # per-rank epoch flags (gather_rows)
        sd.commit()
        # the signal-base buffer each launch reads (the other: the next launch's, from GW)
        self._cpar = 0
        self._cons = tuple(int(c.data_ptr()) for c in sd.consumed)
        if self._cons[0] % (2 * tp * NQ * 4) or self._cons[1] != self._cons[0] ^ (
            tp * NQ * 4
        ):
            raise RuntimeError(f"MegaMoeTP big-M: misaligned signal bases {self._cons}")
        dev = self.device = sd.device
        self.offs = torch.tensor(
            [sd.offset(r_) for r_ in self.REGIONS], dtype=torch.int64, device=dev
        )
        m = T // tp
        bf = torch.bfloat16
        self.normed = torch.empty((m, H), dtype=bf, device=dev)
        self.resout = torch.empty((m, H), dtype=bf, device=dev)
        self.ybuf = torch.empty((m, H), dtype=bf, device=dev)
        # route rows / lists of every chunk packed (chunk k from route tp * o_k * K1)
        self.routes = torch.empty(T * K1 * H, dtype=bf, device=dev)
        self.lrix = torch.empty(T * K1, dtype=torch.int32, device=dev)
        self.lwt = torch.empty(T * K1, dtype=torch.float32, device=dev)
        # unit slots of every chunk ([C][8][ucap of the batch]): sum over chunks <= the bound
        self.uslots = 2 * (-(-tp * m * K1 // UNIT_ROWS) // 8) + CMAX * (
            -(-self.NE // 8) + UQ + 2
        )
        self.utab = torch.zeros(8 * self.uslots * 4, dtype=torch.int32, device=dev)
        self.ctrl = torch.zeros(CTRL_INTS + 32, dtype=torch.int32, device=dev)
        self.chunks_t = {}
        # moe()'s padding buffers (m not a multiple of PAD), the largest padded size
        mpx = T // tp
        self._pads = (
            torch.zeros((mpx, H), dtype=bf, device=dev),
            torch.zeros((mpx, K1), dtype=torch.float32, device=dev),
            torch.full((mpx, K1), -1, dtype=torch.int32, device=dev),
            torch.zeros((mpx, H), dtype=bf, device=dev),
            torch.empty((tp * mpx, H), dtype=torch.uint8, device=dev),
            torch.empty((tp * mpx,), dtype=torch.float32, device=dev),
        )
        self._pad_dirty = 0  # rows below may hold an earlier call's tokens / routes
        self._ready = set()
        self.cu = torch.cuda.get_device_properties(dev).multi_processor_count
        self.nab = self._nab()

    def _nab(self):
        """the deepest A ring (dividing the GEMM1 chunk count) whose LDS fits a CU"""
        from .bigm_gemm import bigm_lds

        kch = self.H // 256
        return next(
            n
            for n in (4, 3, 2)
            if kch % n == 0
            and bigm_lds(UNIT_ROWS, n, self.I)["BYTES"] + 4096 <= 160 * 1024
        )

    @staticmethod
    def _check_geom(o):
        rg, cpu = int(o.get("AITER_BIGM_RG", UNIT_ROWS)), int(
            o.get("AITER_BIGM_CPU", 1)
        )
        if (rg, cpu) != (UNIT_ROWS, 1):
            raise ValueError(
                f"MegaMoeTP big-M: AITER_BIGM_RG={rg} / AITER_BIGM_CPU={cpu} unsupported "
                f"(only {UNIT_ROWS} / 1)"
            )

    def gather_rows(self, rows, local=False):
        """Collective: every rank's rows [m, H] bf16 -> [tp * m, H] (rank r's at r * m), a
        view of the arena valid until the next call. local: this rank's rows only (placed
        at rank * m; no peer writes). Each rank writes its rows into every
        peer's arena (hipIpc mapping; one side stream per peer), then stream-ordered flags:
        hipStreamWriteValue32 of this call's epoch into every peer's flag slot of this rank,
        hipStreamWaitValue32 on every peer's slot here. No host sync (the host keeps running
        ahead of the GPU). Two buffers in turn: a peer's writes of call k + 2 follow its wait
        for this rank's flag of call k + 1, which this stream writes after whatever it ran on
        call k's rows."""
        import torch
        import torch.distributed as dist

        assert (
            not torch.cuda.is_current_stream_capturing()
        ), "gather_rows: not graph-capturable"
        m, H = rows.shape
        T = m * self.tp
        assert T <= self.Tmax and rows.dtype == torch.bfloat16
        stream = torch.cuda.current_stream()
        if getattr(self, "_brows_peer", None) is None:
            off, half = self.sd.offset("brows"), self.Tmax * H * 2
            bases = [int(b_) for b_ in self.sd.arena.base_ptrs]
            self._brows_peer = [
                [
                    torch.as_tensor(
                        _CudaBuf(b_ + off + i * half, half), device=self.device
                    )
                    .view(torch.bfloat16)
                    .view(self.Tmax, H)
                    for b_ in bases
                ]
                for i in range(2)
            ]
            foff = self.sd.offset("bflag")
            # flag slot of rank r (64 B apart) in rank p's arena, as mapped here
            self._bflag = [[b_ + foff + r * 64 for r in range(self.tp)] for b_ in bases]
            self.sd.view("bflag", (self.tp * 16,), torch.int32).zero_()
            self._brows_streams = [
                torch.cuda.Stream(self.device) for _ in range(self.tp)
            ]
            self._bepoch = 0
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.sd.group)
        self._bepoch += 1
        ep = self._bepoch & 0x7FFFFFFF
        dst = self._brows_peer[self._bepoch & 1]
        src = rows.contiguous()
        lo, hi = self.rank * m, (self.rank + 1) * m
        if local:
            _kcopy(dst[self.rank][lo:hi], src)
            return dst[self.rank][:T]
        ev = torch.cuda.Event()
        ev.record(stream)
        for p in range(self.tp):
            if p == self.rank:
                continue
            s_ = self._brows_streams[p]
            s_.wait_event(ev)
            with torch.cuda.stream(s_):
                _kcopy(dst[p][lo:hi], src)
            src.record_stream(s_)
        _kcopy(dst[self.rank][lo:hi], src)
        hip, sh = _hip(), stream.cuda_stream
        for p in range(self.tp):
            if p != self.rank:
                stream.wait_stream(self._brows_streams[p])
        for p in range(self.tp):  # this rank's rows landed in p's arena
            if p != self.rank:
                _hip_check(
                    hip.hipStreamWriteValue32(sh, self._bflag[p][self.rank], ep, 0)
                )
        for p in range(self.tp):  # every peer's rows landed here
            if p != self.rank:
                _hip_check(
                    hip.hipStreamWaitValue32(
                        sh, self._bflag[self.rank][p], ep, 0, 0x7FFFFFFF
                    )
                )
        return dst[self.rank][:T]

    def poll_errors(self) -> int:
        """Nonzero if a wait on a peer inside the kernel gave up since the last clear."""
        return int(self.ctrl[C_ERR].item())

    def clear_errors(self) -> None:
        self.ctrl[C_ERR:].zero_()

    def error_flag(self):
        """The sticky watchdog word as a 1-element device tensor (no sync): 1 if a wait gave
        up since the last clear."""
        return self.ctrl[C_ERR : C_ERR + 1]

    def reset(self) -> None:
        """Collective: re-base the per-(peer, queue) signal bases on the received signals and
        clear the watchdog word -- after a wait gave up the bases may be out of step with the
        peers' puts (every later wait on that queue would time out)."""
        import time

        import torch
        import torch.distributed as dist

        from ..tensor_shim import _run_compiled

        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.sd.group)
        time.sleep(0.2)  # the puts of the peers' finished launches land
        got = torch.zeros(self.tp * NQ, dtype=torch.int32, device=self.device)
        _run_compiled(
            _compile_signal_read(self.tp),
            self.sd.dev_comm.ptr,
            got.data_ptr(),
            torch.cuda.current_stream(),
        )
        self.sd.consumed[self._cpar].copy_(got)
        self.ctrl.zero_()
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.sd.group)

    def close(self) -> None:
        """Collective: free the SDMA queues and the arena (unusable afterwards)."""
        self.sd.close()

    def _plan(self, T):
        """(chunk weights, AITER_BIGM_* options) of a batch of T tokens (T / tp a multiple
        of PAD): the tuned csv's, else the defaults; at most one chunk per PAD local rows
        """
        p = self._plans.get(T)
        if p is None:
            o = dict(self.opts)
            tu = bigm_tuned(
                model_dim=self.H,
                inter_dim=self.I,
                experts=self.NE,
                topk=self.K1,
                tp=self.tp,
                tokens=T,
                device=self.device,
            )
            if tu is not None:
                w = tu["chunks"]
                o = {**tu["opts"], **o}
            else:
                w = [1, 1] if T <= 16384 else [1, 2, 2, 1]
            self._check_geom(o)
            if not 0 < len(w) <= CMAX or min(w) <= 0:
                raise ValueError(f"MegaMoeTP big-M: bad chunk weights {w} (T={T})")
            n = max(1, T // self.tp // PAD)
            if len(w) > n:  # few rows: fewer chunks (the largest weights)
                w = sorted(w, reverse=True)[:n]
            p = self._plans[T] = (tuple(w), tuple(sorted(o.items())))
        return p

    def _split(self, m):
        w = self._plan(m * self.tp)[0]
        tot = sum(w)
        # multiples of 64 rows (moe() pads m to PAD): whole input row items (ZX) in every head
        # sub-block (AITER_BIGM_NSBA <= 4)
        g = PAD
        sizes = [max(g, (m * x // tot) // g * g) for x in w]
        sizes[w.index(max(w))] += m - sum(sizes)
        if m % g or sum(sizes) != m or min(sizes) <= 0 or any(x % g for x in sizes):
            raise RuntimeError(f"MegaMoeTP big-M: cannot split {m} rows by {w}")
        return sizes

    def _chunks(self, T):
        import torch

        m = T // self.tp
        sizes = tuple(self._split(m))
        c = self.chunks_t.get(
            (T, sizes)
        )  # (the plan of T may be replaced: keyed by its sizes)
        if c is None:
            # the kernel's 32-bit offsets of a chunk's route rows (rows * H bf16)
            if self.tp * max(sizes) * self.K1 * self.H * 2 >= 1 << 31:
                raise ValueError(
                    f"MegaMoeTP big-M: chunks {sizes} of {T} tokens: a chunk's route rows "
                    "exceed 32-bit offsets (more chunks: the tuned csv)"
                )
            need = len(sizes) * self._ucap(T)
            if need > self.uslots:
                self.uslots = need
                self.utab = torch.zeros(
                    8 * need * 4, dtype=torch.int32, device=self.device
                )
            os_ = [sum(sizes[:i]) for i in range(len(sizes))]
            c = self.chunks_t[(T, sizes)] = torch.tensor(
                [[o, s] for o, s in zip(os_, sizes)],
                dtype=torch.int32,
                device=self.device,
            )
        return c

    def set_weights(self, w1, w1_scale, w2, w2_scale):
        """the next launches use another layer's weights of the same shape"""
        import torch

        self.w = (
            w1.view(torch.uint8),
            w1_scale.view(torch.uint8),
            w2.view(torch.uint8),
            w2_scale.view(torch.uint8),
        )

    def _ucap(self, T):
        return unit_cap(
            self.tp,
            max(self._split(T // self.tp)),
            self.K1,
            self.NE,
            UNIT_ROWS,
        )

    def _run(self, launch, args):
        import torch

        from ..tensor_shim import _run_compiled

        if launch not in self._ready:
            # compile (or load) on every rank first: the kernel's waits on peers have a 2 s
            # watchdog, a launch must not start while a peer is still compiling
            import torch.distributed as dist

            _bigm_cached(launch, args)
            torch.cuda.synchronize()
            dist.barrier(group=self.sd.group)
            self._ready.add(launch)
        _run_compiled(launch, *args)

    def moe(
        self, x, topk_weights, topk_ids, out=None, tail=None, local=False, gather=False
    ):
        """io: the MoE of every rank's tokens (ag_rs): x [m, H] bf16 (this rank's rows, global
        rows rank * m + i), topk_weights f32 / topk_ids int32 [m, K1] (the shared expert E in
        any column; ids outside [0, experts] are no route) -> y [m, H] bf16 = this rank's rows of
        sum_r MoE_r(every row); tail = (res_in, res_out, norm_w): also res_out = y + res_in and
        the (Gemma)RMSNorm of res_out as FP8 rows (+ f32 scales) of every rank -> (y, q [T, H]
        fp8 (uint8), scales [T]); local: only this rank's q rows (rank * m ..) are written.
        gather (ygather, no tail): y of every rank's rows -> [tp * m, H] (rank r's at r * m; out,
        else a view of the window valid until the next launch), gathered inside the kernel;
        copies out of the window by shader kernels (_kcopy).
        Any m: the kernel runs on m rounded up to a multiple of 64 (PAD; the extra rows are zero,
        with no routes), its results cut back to m rows."""
        import torch

        if gather and not (self.ygather and tail is None):
            raise ValueError("MegaMoeTP big-M: gather needs ygather and no tail")
        m = x.shape[0]
        if (
            x.dtype != torch.bfloat16
            or tuple(x.shape) != (m, self.H)
            or not x.is_contiguous()
            or tuple(topk_ids.shape) != (m, self.K1)
            or tuple(topk_weights.shape) != (m, self.K1)
            or topk_ids.dtype != torch.int32
            or topk_weights.dtype != torch.float32
            or not (topk_ids.is_contiguous() and topk_weights.is_contiguous())
        ):
            raise ValueError(
                f"MegaMoeTP big-M: need contiguous bf16 x [m, {self.H}], int32 topk_ids / "
                f"f32 topk_weights [m, {self.K1}]; got {x.dtype} {tuple(x.shape)}, "
                f"{topk_ids.dtype} {tuple(topk_ids.shape)}, {topk_weights.dtype} "
                f"{tuple(topk_weights.shape)}"
            )
        mp = -(-m // PAD) * PAD
        if mp * self.tp > self.Tmax:
            raise ValueError(
                f"MegaMoeTP big-M: {m} local tokens (padded {mp}) exceed "
                f"{self.Tmax // self.tp}"
            )
        if mp == m:
            if gather:
                ya = self._moe(x, topk_weights, topk_ids, None, None, local, gather)
                if out is not None:
                    _kcopy(out, ya)
                return ya if out is None else out
            return self._moe(x, topk_weights, topk_ids, out, tail, local)
        # one set of padded buffers (allocated at init, the largest padded size), sliced to mp:
        # rows m.. of x / tw / ids / res must be zero / no route; rows below _pad_dirty may hold
        # an earlier call's tokens
        xp, twp, idp, rp, qc, qsc = self._pads
        lo_, hi_ = m, min(self._pad_dirty, mp)
        if hi_ > lo_:
            xp[lo_:hi_].zero_()
            twp[lo_:hi_].zero_()
            idp[lo_:hi_].fill_(-1)
            rp[lo_:hi_].zero_()
        self._pad_dirty = max(m, self._pad_dirty) if self._pad_dirty > mp else m
        xp, twp, idp, rp = xp[:mp], twp[:mp], idp[:mp], rp[:mp]
        qc, qsc = qc[: self.tp * m], qsc[: self.tp * m]
        xp[:m].copy_(x)
        twp[:m].copy_(topk_weights)
        idp[:m].copy_(topk_ids)
        if tail is not None:
            res_in, res_out, norm_w = tail
            rp[:m].copy_(res_in)
            y, q, qs = self._moe(xp, twp, idp, None, (rp, rp, norm_w), local)
            res_out.copy_(rp[:m])
            # q / scales as [rank][m] rows (the kernel's are [rank][mp])
            src = [self.rank] if local else range(self.tp)
            for r in src:
                qc[r * m : (r + 1) * m].copy_(q[r * mp : r * mp + m])
                qsc[r * m : (r + 1) * m].copy_(qs[r * mp : r * mp + m])
            q, qs = qc, qsc
        elif gather:
            ya = self._moe(xp, twp, idp, None, None, local, gather)
            if out is None:
                out = torch.empty(
                    (self.tp * m, self.H), dtype=ya.dtype, device=ya.device
                )
            for r in range(self.tp):
                _kcopy(out[r * m : (r + 1) * m], ya[r * mp : r * mp + m])
            return out
        else:
            y = self._moe(xp, twp, idp, None, None, local)
        y = y[:m]
        if out is not None:
            y = out.copy_(y)
        return y if tail is None else (y, q, qs)

    def _io_launch(self, m, tailq):
        """the io kernel of m (a multiple of PAD) local rows"""
        T = m * self.tp
        eps, gemma, lim = self.cfg
        w_, opts = self._plan(T)
        order = 4 if self.tp * max(self._split(m)) >= 8192 else 3
        return compile_bigm_layer(
            self.H,
            self.I,
            self.E,
            self.K,
            self.tp,
            len(w_),
            eps,
            gemma,
            lim,
            NAB=self.nab,
            order=order,
            opts=opts,
            tailq=tailq,
            act=self.act,
        )

    def io_rows_per_kernel(self, lo, hi):
        """One local row count per distinct io kernel (whatever the tail) among the batches of
        lo .. hi local rows (padded to PAD as moe() does): warming these up compiles every big-M
        kernel such batches can launch."""
        reps = {}
        for m in range(-(-max(lo, 1) // PAD) * PAD, -(-hi // PAD) * PAD + 1, PAD):
            reps.setdefault(self._io_launch(m, 0), m)
        return sorted(reps.values())

    def precompile(self, rows, tailq):
        """Collective: the io kernels of these local row counts (tailq: 0 - 3 as moe()) into
        the disk cache, split over the ranks -- rank r compiles every tp-th -- so a cold cache
        costs about one compile, not one per kernel; the moe() warm-ups then load them. No-op
        without a disk cache."""
        import inspect

        import torch
        import torch.distributed as dist

        if not _bigm_cache_dir():
            return
        mine = [m for i, m in enumerate(rows) if i % self.tp == self.rank]
        for m in mine:
            launch = self._io_launch(m, tailq)
            if launch in self._ready:
                continue
            # (compile only: the argument values do not matter, their types are the launch's)
            n = len(inspect.signature(launch.func).parameters)
            _bigm_cached(launch, (0,) * (n - 1) + (torch.cuda.current_stream(),))
        dist.barrier(group=self.sd.group)

    def _moe(self, x, topk_weights, topk_ids, out, tail, local, gather=False):
        """moe() of m (a multiple of PAD) rows"""
        import torch

        m = x.shape[0]
        T = m * self.tp
        w1, w1s, w2, w2s = self.w
        y = self.normed[:m] if out is None else out
        if tail is not None:
            res_in, res_out, norm_w = tail
        else:
            res_in, res_out, norm_w = x, self.resout, x
        tq = 3 if gather else 0 if tail is None else 2 if local else 1
        launch = self._io_launch(m, tq)
        chunks = self._chunks(T)  # (first: it may grow the unit table)
        self.ctrl[:CTRL_INTS].zero_()
        self._run(
            launch,
            (
                self.sd.base,
                self.sd.bases.data_ptr(),
                self.sd.dev_comm.ptr,
                self._cons[self._cpar],
                self.offs.data_ptr(),
                x.data_ptr(),
                topk_ids.data_ptr(),
                topk_weights.data_ptr(),
                norm_w.data_ptr(),
                res_in.data_ptr(),
                y.data_ptr(),
                res_out.data_ptr(),
                self.ybuf.data_ptr(),
                self.routes.data_ptr(),
                self.lrix.data_ptr(),
                self.lwt.data_ptr(),
                self.utab.data_ptr(),
                w1.data_ptr(),
                w1s.data_ptr(),
                w2.data_ptr(),
                w2s.data_ptr(),
                self.ctrl.data_ptr(),
                chunks.data_ptr(),
                self.rank,
                m,
                self._ucap(T),
                self.cu,
                torch.cuda.current_stream(),
            ),
        )
        self._cpar ^= 1  # (the next launch reads what this one's GW wrote)
        if gather:
            return self.sd.view("q", (T, self.H), torch.bfloat16)
        if tail is None:
            return y
        q = self.sd.view("q", (T, self.H), torch.uint8)
        qs = self.sd.view("qs", (T,), torch.float32)
        return y, q, qs
