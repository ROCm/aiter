# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Large-batch (big-M) dispatch of the MegaMoE TP kernel: one persistent launch runs the
whole MoE region of a layer for T tokens (m = T / tp rows per rank), in C chunks:

  QA  int8 + fp32 / 32 quant of this rank's partial rows owed to each peer   -> SDMA RS (A)
  RB  wait A; sum + residual -> res, (Gemma)RMSNorm, MXFP4 rows; router GEMM + topk
                                                     -> SDMA gather of rows + routing (C)
  S1  wait C; routes per expert       S2  route-list offsets, per-XCD unit queues
  S3  route lists (route index, weight) per expert
  U   fused units (bigm_gemm): GEMM1 -> act + MXFP4 in LDS -> GEMM2 -> route rows
  RQ  topk sum: own rows bf16, peer rows int8                -> SDMA MoE-output RS (E)
  RF  wait E; sum + res -> res_out, next-layer norm, per-token FP8          -> SDMA gather (G)
  GW  wait every G; advance the persistent signal base

All communication is MORI SDMA (puts with remote signals) issued and awaited inside the
kernel; the GEMM1 -> GEMM2 intermediate stays in LDS. CTAs (one per CU, 4 waves) claim
items with a per-XCD atomic ticket in a fixed phase order; an item only waits on items
before it (and on peers' puts of earlier items), so the schedule cannot deadlock.
Data another XCD reads in the same launch is stored write-through (sc0 | sc1).
"""

import os
import functools
import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ..communication_ops_utils import atomic_add_agent, load_i32_global_agent
from ..kernels_common import LOG2E
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
    fp4_pack,
    fp8x4_pack,
    fp8x8_decode,
    g_ld_sys,
    g_st_sys,
    i32,
    lds_atomic_add,
    lds_atomic_or,
    lds_ld_i32,
    lds_st,
    now,
    pack_bf16x8,
    rsrc,
    sum_live,
    uni,
    wait_vm,
    wave_red,
)

NTH = NW * 64
const_expr = fx.const_expr

# SDMA queues per peer (MORI spreads a peer's queues over its two engines by parity)
# A's queue is chosen per chunk count (compile_bigm_layer): with <= 2 chunks an odd queue (the
# other engine than the C gather on the even queues, so A(k+1) does not queue behind C(k));
# with more chunks queue 0 (A(k+2) would queue behind the drains E / G on the odd queues)
QA0, QA1 = 0, 7
QE, QX, QQ, QXS, QS, QID = 1, 2, 3, 4, 5, 6
NQ = 8
QUEUES = (QA0, QE, QX, QQ, QXS, QS, QID, QA1)
PUTS_PER_CHUNK = {QA0: 1, QE: 1, QX: 1, QQ: 1, QXS: 1, QS: 1, QID: 2, QA1: 1}

# phase kinds (per chunk), in item order within a chunk
P_QA, P_RB, P_RT, P_S1, P_S2, P_S3, P_U, P_RQ, P_RF, P_UB, P_U2 = range(11)
NKIND = 16  # done-counter slots per chunk
ZT = 16  # rows per row-phase item (QA, RQ, RF)
ZR = 4  # rows per RB item (reduce + norm + router: more items, the router is latency bound)
ZQ = 4  # rows per QA / RQ / RF item
ZRT = 16  # rows per router item (each wave: E / NW experts over the full K)
SRT = 32  # tokens per sort item
UQ = 4  # experts with more units than this are spread over the XCDs

# ctrl ints (zeroed by the host before every launch)
C_TK = 0  # per-XCD tickets, 32 apart
C_TURN = 256  # per-queue post turns, 32 apart
C_PH = 512  # [chunk][NKIND] done counters, 32 apart
C_CNT = 4096  # [chunk][NE] routes per expert (256 apart)
C_OFF = 6144  # [chunk][NE] route-list offset of each expert
C_FILL = 8192  # [chunk][NE] routes listed so far
C_UN = 10240  # [chunk][8] units of each XCD (32 apart)
C_RTG = 11264  # [chunk][256] reduced rows per router group
C_AGATE = 12224  # [chunk] A(k) posted by the second of: QA(k) done, C(k-1) posted
C_SB = 13312  # [chunk][RQ, RF][sub-block] done items, 8 apart
C_TLS = 14848  # [dbg] [chunk][NKIND] (sum of item times, items)
C_TL = 12288  # [dbg] [chunk][NKIND] (~first start, last end) realtime (AITER_BIGM_TL=1)
C_RTX = 16384  # [chunk][256] router items done per group (expert-split router)
C_SBH = 18432  # [chunk][QA, RT][head sub-block] done items, 8 apart
C_CPOST = 20224  # [chunk] head sub-blocks whose gather C is posted, 8 apart
C_UG1 = 20480  # GSPLIT: [chunk][XCD][unit slot] GEMM1 passes done
C_TKU = 28672  # ROLE: per-XCD unit-queue tickets, 32 apart
CTRL_INTS = 28928


def _opaque(v):
    """v through a side-effecting copy: derived values are not hoisted out of the work loop
    (each item's address math stays inside the item, not live across every item)"""
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.expr import arith

    return fx.Int32(_llvm.inline_asm(T.i32, [arith.unwrap(v)], "v_mov_b32 $0, $1", "=v,v", has_side_effects=True))


def _g_umax(addr, v):
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.expr import arith

    from ..communication_ops_utils import _to_ptr_global

    _llvm.AtomicRMWOp(
        _llvm.AtomicBinOp.umax, _to_ptr_global(addr), arith.unwrap(fx.Int32(v)),
        _llvm.AtomicOrdering.monotonic, syncscope=fx.rocdl.SyncScope.Agent,
    )


def _g_cas(addr, cmp, val):
    """agent-scope compare-and-swap on a global i32: returns the old value"""
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.expr import arith

    from ..communication_ops_utils import _to_ptr_global

    r = _llvm.cmpxchg(
        _to_ptr_global(addr), arith.unwrap(fx.Int32(cmp)), arith.unwrap(fx.Int32(val)),
        _llvm.AtomicOrdering.monotonic, _llvm.AtomicOrdering.monotonic, syncscope=fx.rocdl.SyncScope.Agent,
    )
    return fx.Int32(_llvm.extractvalue(T.i32, r, [0]))


def i8_row_bytes(H):
    return H + H // 8


def _fadd(x, y):
    return (x.bitcast(fx.Float32) + y.bitcast(fx.Float32)).bitcast(fx.Int32)


def _fmax(x, y):
    return x.bitcast(fx.Float32).maximumf(y.bitcast(fx.Float32)).bitcast(fx.Int32)


def _i4x8_pack(f, inv):
    w = i32(0)
    for k in range_constexpr(8):
        q = fx.Int32(fmath.roundeven(f[k] * inv))
        q = fx.max(fx.min(q, i32(7)), i32(-7))
        w = w | ((q & i32(0xF)) << i32(4 * k))
    return w


def _i4x8_unpack(d, sc):
    out = []
    for k in range_constexpr(8):
        b = (fx.Int32(d) << i32(28 - 4 * k)) >> i32(28)
        out.append(b.to(fx.Float32) * sc)
    return out


def _i8x4_pack(f, inv):
    w = i32(0)
    for k in range_constexpr(4):
        q = fx.Int32(fmath.roundeven(f[k] * inv))
        q = fx.max(fx.min(q, i32(127)), i32(-127))
        w = w | ((q & i32(0xFF)) << i32(8 * k))
    return w


def _i8x4_unpack(d, sc):
    out = []
    for k in range_constexpr(4):
        b = (fx.Int32(d) << i32(24 - 8 * k)) >> i32(24)
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
    scale: float,
    shared_w: float,
    swiglu_limit: float,
    RG: int = 128,
    NAB: int = 4,
    BST: int = 3,
    BS2: int = 3,
    order: int = 3,
    ext: int = 0,
    opts: tuple = (),
    CPU: int = 1,
    io: int = 0,
    tailq: int = 1,
):
    """tailq (io): 0 no tail, 1 the tail with the q gather, 2 the local tail (this rank's q rows).
    io=1: the MoE interface (MegaMoeTP.forward, ag_rs): the caller's normed own rows x [m, H] and
    their routing (ids / tw [m, K1], the shared expert's column included) replace the attention RS +
    norm + router; the tail sums the MoE output (y), and with tailq res_out = y + res_in, the
    next-layer norm + FP8 rows of every rank (q gather).
    ext=0: the whole layer in one launch (own routing plan).
    ext=2 (pre kernel): every chunk's attention RS (SDMA), reduce + norm + router and the gather
    of the quantized rows + routing (SDMA); exits once the routing landed (moe_sorting of each
    chunk runs next, outside the kernels).
    ext=1 (MoE kernel): each chunk's units (one per sorted block, after its gathered rows
    landed), the topk sums, the MoE-output RS, the final reduce + norm + FP8 and the q gather."""
    K1, NE = K + 1, E + 1
    PIT = H // 8 // NTH
    import os as _os

    od = dict(opts)  # tuned AITER_BIGM_* options (the csv); the environment overrides them

    def _opt(name, default):
        return _os.environ.get(name, od.get(name, default))

    TLD = _opt("AITER_BIGM_TL", "0") == "1"
    BS2 = int(_opt("AITER_BIGM_BS2", str(BS2)))  # GEMM2 B ring depth (6 = a whole group: the
    # route stores of a group never sit between a GEMM2 load and its wait)
    # item order (see ORDER below); the host picks 3 (one unit round per chunk) or 4 (several)
    ORD = int(_opt("AITER_BIGM_ORD", str(order)))
    AGATE = int(_opt("AITER_BIGM_AGATE", "0" if (ext == 2 or (ORD == 7 and not ext)) else "1"))  # A(k) after C(k-1)
    EORD = int(_opt("AITER_BIGM_EORD", "1"))  # ext=1 item order
    # ext=1: eager topk sums: the unit that completes a token's K1 route rows sums them
    RQE = ext == 1 and int(_opt("AITER_BIGM_RQE", "0")) == 1
    # tail sub-blocks: each chunk's topk sums / drains post their E / q puts per row sub-block
    NSB = int(_opt("AITER_BIGM_NSB", "4"))
    # head sub-blocks: each chunk's attention RS (A) and gather (C) per row sub-block, the
    # reduce / router of a sub-block start once its A landed
    NSBA = int(_opt("AITER_BIGM_NSBA", "1"))
    assert NSBA in (1, 2, 4)
    RBDBG = int(_opt("AITER_BIGM_RBDBG", "0"))  # [dbg] reduce + norm: 1 no res / normed stores, 2 no peer rows
    RDBG = int(_opt("AITER_BIGM_RDBG", "0"))  # [dbg] router: 1 no GEMM, 2 top-1 only
    RNS = int(_opt("AITER_BIGM_RNS", "4"))  # router items per 16-row group (expert slices), chunk 0
    # later chunks' router runs beside the units on the CTAs they leave idle: there the total
    # CTA time counts, fewer slices read the normed rows fewer times
    RNS1 = int(_opt("AITER_BIGM_RNS1", str(RNS)))
    assert E % (16 * RNS) == 0 and E % (16 * RNS1) == 0
    RUNR = int(_opt("AITER_BIGM_RUNR", "8"))
    UA = int(_opt("AITER_BIGM_UA", "32"))  # ORD 4: first-round unit slots per XCD
    # reduce + norm items: ZR rows per batch, NSUBR batches per item, the next batch's loads in
    # flight while one batch is reduced / normed / stored
    ZR = int(_opt("AITER_BIGM_ZR", "4"))
    NSUBR = int(_opt("AITER_BIGM_NSUBR", "1"))
    # row sub-batches per attention-quant / topk-sum / final-reduce item
    NSUBQ = int(_opt("AITER_BIGM_NSUBQ", "1"))
    RQD = int(_opt("AITER_BIGM_RQD", "0"))  # topk sum: token pipeline depth (0: batch of ZQ rows)
    # R8: route rows as MXFP8 (half the bytes of bf16 GEMM2 stores / topk-sum loads)
    R8 = int(_opt("AITER_BIGM_R8", "0"))
    E4 = int(_opt("AITER_BIGM_E4", "0"))  # MoE-output RS rows as int4 (+ bf16 scale per 32)
    assert not (E4 and ext)
    # SXW 0: the routing plan waits only for the gathered routing; units wait for the gathered rows
    SXW = int(_opt("AITER_BIGM_SXW", "1"))
    # S2P: the routing plan's unit -> XCD queue assignment in parallel (balanced contiguous
    # ranges in expert order) instead of thread 0's greedy least-loaded loop
    S2P = int(_opt("AITER_BIGM_S2P", "0"))
    # S2E: the routing plan signals its route-list offsets before its unit -> queue plan (the sort
    # items stop spinning on the serial plan)
    S2E = int(_opt("AITER_BIGM_S2E", "0"))
    S2N = 2 if S2E else 1
    if R8:
        assert not ext
        RQD = max(RQD, 1)
    assert NSUBQ in (1, 2, 4) and 16 % (ZQ * NSUBQ) == 0
    assert ZR in (1, 2, 4) and NSUBR in (1, 2, 4, 8, 16) and ZR * NSUBR <= ZRT
    assert (H // NW // 32) % RUNR == 0
    assert NSB in (1, 2, 4) and 16 % NSB == 0
    QA = int(_opt("AITER_BIGM_QA", str(QA1 if (C <= 2 or ext) else QA0)))
    assert QA in (QA0, QA1)
    PPC = dict(PUTS_PER_CHUNK)
    PPC[QA1 if QA == QA0 else QA0] = 0
    assert not io or not ext
    PPC[QE] = PPC[QQ] = PPC[QS] = NSB
    PPC[QA] = PPC[QX] = PPC[QXS] = NSBA
    PPC[QID] = 2 * NSBA
    if io:
        PPC[QA0] = PPC[QA1] = 0  # no attention RS
        if tailq != 1:
            PPC[QQ] = PPC[QS] = 0  # no q gather (no tail, or the local tail: own rows' q only)
    ZX = 16  # io: rows per input item (quant + routing copy)
    assert H % (8 * NTH) == 0 and E % 64 == 0 and E // 64 in (2, 4) and K <= 64
    EPL = E // 64
    # int8 rows of the attention / MoE-output RS: one scale per 32 values, I8S bytes each
    # (4: fp32, 2: bf16 -- lowers the route match 0.973 -> 0.970 at 8192, not used); the host's
    # regions are sized for fp32
    I8S = int(_opt("AITER_BIGM_I8S", "4"))
    assert I8S in (2, 4)
    RB = H + (H // 32) * I8S
    SCO = H
    lay, unit_body = build_bigm_unit(
        H, I, K1, RG, NAB, BST, BS2, "swiglu", swiglu_limit, 0, AUX_SYS, PACKED=1 if ext else 0, R8=R8
    )
    # GSPLIT: a unit's work as items -- one GEMM1 pass each (G1: -> the unit's intermediate block in
    # global) and G2N GEMM2 column groups each (G2, once the unit's G1 passes are done): short
    # items, so other phases' items run between them on the same CTAs
    GSPLIT = int(_opt("AITER_BIGM_GSPLIT", "0"))
    # ROLE: two item queues per XCD -- P (exchange / prep / topk sum / drain) and U (units). NP CTAs
    # of each XCD only take P items; the others take units, and while the next unit's chunk is not
    # planned yet, that chunk's (or earlier) prep items; once every unit is taken, any P item.
    # Chunk k+1's prep and chunk k-1's tail then run beside chunk k's units.
    NP = int(_opt("AITER_BIGM_NP", "0"))
    RQU = int(_opt("AITER_BIGM_RQU", "1"))  # ROLE: unit CTAs also take topk sums of fully taken chunks
    G2N = int(_opt("AITER_BIGM_G2N", "4"))
    PASSES_U = I // (NW * 32)
    NSUB2 = (H // (NW * 64)) // G2N
    UBYTES = PASSES_U * RG * (NW * 16 + NW)  # a unit's intermediate block (bigm_gemm's layout)
    unit_g1 = unit_g2 = None
    if GSPLIT:
        assert not ext and (H // (NW * 64)) % G2N == 0
        BST1 = int(_opt("AITER_BIGM_BST1", "2" if CPU > 1 and RG > 64 else str(BST)))  # G1 B ring (regs)
        lay3, unit_g1 = build_bigm_unit(H, I, K1, RG, NAB, BST1, BS2, "swiglu", swiglu_limit, 0, AUX_SYS, MODE=3)
        lay4, unit_g2 = build_bigm_unit(H, I, K1, RG, NAB, BST, BS2, "swiglu", swiglu_limit, 0, AUX_SYS, MODE=4, G2N=G2N, R8=R8)
        lay = dict(lay, BYTES=max(lay3["BYTES"], lay4["BYTES"]))  # the fused unit is not used
    LBASE = lay["BYTES"]  # row-phase / plan scratch after the unit's LDS (also reused ranges)
    # router partials [wave][E/16][64 lanes][4] f32 reuse the unit's A ring
    # router: per-wave partial logits [NW][16][E / RNS] f32 (and the topk's [16][E]) at the CTA's LDS start
    assert NW * 16 * (E // min(RNS, RNS1)) * 4 <= LBASE and 16 * E * 4 <= LBASE
    LSC = LBASE  # 16 B scratch for block reductions
    LT = LBASE + 64  # ticket broadcast
    LT_R = LT + 8  # router: last item of the group
    LH = LBASE + 128  # [NE] histogram / ranks
    LB2 = LH + 4 * 256  # [NE] bases
    LP = LB2 + 4 * 256  # [SRT * K1] route positions of a sort item
    LDS_BYTES = LP + 4 * SRT * K1 + 64
    assert LDS_BYTES <= 160 * 1024 // CPU, LDS_BYTES  # CPU CTAs per CU

    def fb(v):
        return f"{struct.unpack('<I', struct.pack('<f', v))[0]:x}"

    # item order: chunk k+1's prep (QA, RB, plan) ahead of chunk k's units and drain
    # (a CTA spinning on an item cannot take later ones: items that wait for a transfer come
    # after the work that overlaps it) chunk k's units, then chunk k+1's prep (claimed as
    # chunk k's units drain), chunk k's topk sum, then chunk k-1's drain after k's units
    def prep(k):
        return [(k, P_RB), (k, P_RT), (k, P_S1), (k, P_S2), (k, P_S3)]

    # ORD 0: chunk k's units, then chunk k+1's prep (claimed as chunk k's units drain), chunk
    # k-1's topk sum, chunk k-2's drain; chunk k+2's partials after chunk k's units.
    # ORD 1: every chunk's partials first and chunk k+1's prep ahead of chunk k's units (its
    # items wait for A(k+1) and block the units: slower at 2 chunks)
    if ORD == 1:
        ORDER = [(k, P_QA) for k in range(C)] + prep(0) + (prep(1) if C > 1 else [])
        for k in range(C):
            ORDER += [(k, P_U)]
            if k + 2 < C:
                ORDER += prep(k + 2)
            if k >= 1:
                ORDER += [(k - 1, P_RQ)]
            if k >= 2:
                ORDER += [(k - 2, P_RF)]
        ORDER += [(C - 1, P_RQ)] + ([(C - 2, P_RF)] if C > 1 else []) + [(C - 1, P_RF)]
    elif ORD == 14:
        # io: every chunk's input rows first (their gathers stream back to back), then per chunk
        # k: its units, chunk k+1's routing plan (gather long landed), chunk k-1's topk sum (its
        # units done by the time chunk k's are all taken), chunk k-2's drain
        def sp(k):
            return [(k, P_S1), (k, P_S2), (k, P_S3)]

        ORDER = [(k, P_QA) for k in range(C)] + [(k, P_RT) for k in range(C)]
        ORDER += [(k, P_RB) for k in range(C)] + sp(0)
        for k in range(C):
            ORDER += [(k, P_U)]
            if k + 1 < C:
                ORDER += sp(k + 1)
            if k >= 1:
                ORDER += [(k - 1, P_RQ)]
            if k >= 2:
                ORDER += [(k - 2, P_RF)]
        ORDER += [(C - 1, P_RQ)] + ([(C - 2, P_RF)] if C > 1 else []) + [(C - 1, P_RF)]
    elif ORD == 13:
        # ROLE queues: P = head, then per chunk k: prep(k+1), QA(k+2), RF(k-1), RQ(k); U = units
        ORDER = [(0, P_QA)] + prep(0) + ([(1, P_QA)] if C > 1 else [])
        for k in range(C):
            if k + 1 < C:
                ORDER += prep(k + 1)
            if k + 2 < C:
                ORDER += [(k + 2, P_QA)]
            if k >= 1:
                ORDER += [(k - 1, P_RF)]
            ORDER += [(k, P_RQ)]
        ORDER += [(C - 1, P_RF)]
        ORDER += [y for k in range(C) for y in ([(k, P_U)] + ([(k, P_U2)] if GSPLIT else []))]
    elif ORD == 12:
        # GSPLIT: chunk k+1's reduce / router between chunk k's G1 (GEMM1 pass) items and its G2 items,
        # its routing plan after them; chunk k's topk sum before chunk k+1's units
        ORDER = [(0, P_QA)] + prep(0) + ([(1, P_QA)] if C > 1 else [])
        for k in range(C):
            ORDER += [(k, P_U)]
            if k + 1 < C:
                ORDER += [(k + 1, P_RB), (k + 1, P_RT)]
            ORDER += [(k, P_U2)]
            if k >= 1:
                ORDER += [(k - 1, P_RF)]
            if k + 1 < C:
                ORDER += [(k + 1, P_S1), (k + 1, P_S2), (k + 1, P_S3)]
            if k + 2 < C:
                ORDER += [(k + 2, P_QA)]
            ORDER += [(k, P_RQ)]
        ORDER += [(C - 1, P_RF)]
    elif ORD == 9:
        # chunk k+1's reduce / router ahead of chunk k's units (the whole machine once A(k+1),
        # posted after C(k) landed, is in), its routing plan beside chunk k's units, chunk k's
        # topk sum between chunk k's and chunk k+1's units, chunk k's drain after chunk k+1's
        def sp(k):
            return [(k, P_S1), (k, P_S2), (k, P_S3)]

        ORDER = [(0, P_QA), (0, P_RB), (0, P_RT)] + ([(1, P_QA)] if C > 1 else []) + sp(0)
        for k in range(C):
            if k + 1 < C:
                ORDER += [(k + 1, P_RB), (k + 1, P_RT)]
            ORDER += [(k, P_U)]
            if k >= 1:
                ORDER += [(k - 1, P_RF)]
            if k + 1 < C:
                ORDER += sp(k + 1)
            if k + 2 < C:
                ORDER += [(k + 2, P_QA)]
            ORDER += [(k, P_RQ)]
        ORDER += [(C - 1, P_RF)]
    elif ORD == 8:
        # chunk k+1's reduce / router ahead of chunk k's units (they get the whole machine once
        # A(k+1) lands), its routing plan right after chunk k's units, then chunk k+1's units:
        # the units of consecutive chunks pack into one pool; chunk k-1's topk sum / drain run
        # on the CTAs the unit rounds leave idle
        def sp(k):
            return [(k, P_S1), (k, P_S2), (k, P_S3)]

        ORDER = [(0, P_QA), (0, P_RB), (0, P_RT)] + ([(1, P_QA)] if C > 1 else []) + sp(0)
        for k in range(C):
            if k + 1 < C:
                ORDER += [(k + 1, P_RB), (k + 1, P_RT)]
            ORDER += [(k, P_U)]
            if k + 1 < C:
                ORDER += sp(k + 1)
            if k + 2 < C:
                ORDER += [(k + 2, P_QA)]
            if k >= 1:
                ORDER += [(k - 1, P_RQ)]
            if k >= 2:
                ORDER += [(k - 2, P_RF)]
        ORDER += [(C - 1, P_RQ)] + ([(C - 2, P_RF)] if C > 1 else []) + [(C - 1, P_RF)]
    elif ORD == 7:
        # every chunk's prep up front (no unit traffic yet: A(k) not gated), then every chunk's
        # units back to back (they pack across chunks), chunk k's topk sum / drain on the CTAs
        # the unit rounds leave idle
        ORDER = [(k, P_QA) for k in range(C)]
        for k in range(C):  # every chunk's reduce / router before any routing plan waits on a gather
            ORDER += [(k, P_RB), (k, P_RT)]
        for k in range(C):
            ORDER += [(k, P_S1), (k, P_S2), (k, P_S3)]
        ORDER += [(k, P_U) for k in range(C)]
        for k in range(C):
            ORDER += [(k, P_RQ)]
            if k >= 1:
                ORDER += [(k - 1, P_RF)]
        ORDER += [(C - 1, P_RF)]
    elif ORD == 6:
        # every chunk's prep up front (their transfers pipeline back to back), then the unit
        # rounds with chunk k's topk sum between chunk k's and k+1's units and chunk k-1's drain
        # after chunk k's units
        ORDER = [(k, P_QA) for k in range(C)]
        for k in range(C):
            ORDER += prep(k)
        for k in range(C):
            ORDER += [(k, P_U)]
            if k >= 1:
                ORDER += [(k - 1, P_RF)]
            ORDER += [(k, P_RQ)]
        ORDER += [(C - 1, P_RF)]
    elif ORD == 5:
        # chunk k+1's reduce / router ahead of chunk k's units (A(k+1) lands as chunk k's routing
        # does), its routing plan right after them (on the CTAs the unit round leaves idle), then
        # chunk k's topk sum and chunk k+1's units
        def sp(k):
            return [(k, P_S1), (k, P_S2), (k, P_S3)]

        ORDER = [(0, P_QA), (0, P_RB), (0, P_RT)] + ([(1, P_QA)] if C > 1 else []) + sp(0)
        for k in range(C):
            if k + 1 < C:
                ORDER += [(k + 1, P_RB), (k + 1, P_RT)]
            ORDER += [(k, P_U)]
            if k + 1 < C:
                ORDER += sp(k + 1)
            if k + 2 < C:
                ORDER += [(k + 2, P_QA)]
            if k >= 1:
                ORDER += [(k - 1, P_RF)]
            ORDER += [(k, P_RQ)]
        ORDER += [(C - 1, P_RF)]
    elif ORD == 4:
        # U(k) = the first unit slot of every CTA, UB(k) the rest: chunk k+1's reduce / router run
        # between them (A(k+1) has landed) and its routing plan right after, so chunk k+1's units
        # fill the CTAs chunk k's last unit round leaves idle
        ORDER = [(0, P_QA)] + prep(0) + ([(1, P_QA)] if C > 1 else [])
        for k in range(C):
            ORDER += [(k, P_U)]
            if k >= 1:
                ORDER += [(k - 1, P_RQ)]
            if k + 1 < C:
                ORDER += [(k + 1, P_RB), (k + 1, P_RT)]
            ORDER += [(k, P_UB)]
            if k >= 1:
                ORDER += [(k - 1, P_RF)]
            if k + 1 < C:
                ORDER += [(k + 1, P_S1), (k + 1, P_S2), (k + 1, P_S3)]
            if k + 2 < C:
                ORDER += [(k + 2, P_QA)]
        ORDER += [(C - 1, P_RQ), (C - 1, P_RF)]
    elif ORD == 3:
        # chunk k's topk sum right after chunk k+1's prep (it fills the gap while chunk k+1's
        # routing lands, on the whole machine) so E(k), RF(k), G(k) overlap chunk k+1's units
        ORDER = [(0, P_QA)] + prep(0) + ([(1, P_QA)] if C > 1 else [])
        for k in range(C):
            ORDER += [(k, P_U)]
            if k >= 1:  # on the CTAs chunk k's units leave idle (E(k-1) is on its way)
                ORDER += [(k - 1, P_RF)]
            if k + 1 < C:
                ORDER += prep(k + 1)
            if k + 2 < C:
                ORDER += [(k + 2, P_QA)]
            ORDER += [(k, P_RQ)]
        ORDER += [(C - 1, P_RF)]
    elif ORD == 2:
        # every chunk's partials first (no dependencies; A(k) is gated on C(k-1)), chunk k+1's
        # prep after chunk k's units, the last topk sum ahead of the previous chunk's drain
        ORDER = [(k, P_QA) for k in range(C)] + prep(0)
        for k in range(C):
            ORDER += [(k, P_U)]
            if k + 1 < C:
                ORDER += prep(k + 1)
            if k >= 1:
                ORDER += [(k - 1, P_RQ)]
            if k >= 2:
                ORDER += [(k - 2, P_RF)]
        ORDER += [(C - 1, P_RQ)] + ([(C - 2, P_RF)] if C > 1 else []) + [(C - 1, P_RF)]
    else:
        ORDER = [(0, P_QA)] + prep(0) + ([(1, P_QA)] if C > 1 else [])
        for k in range(C):
            ORDER += [(k, P_U)]
            if k + 1 < C:
                ORDER += prep(k + 1)
            if k + 2 < C:
                ORDER += [(k + 2, P_QA)]
            if k >= 1:
                ORDER += [(k - 1, P_RQ)]
            if k >= 2:
                ORDER += [(k - 2, P_RF)]
        ORDER += ([(C - 2, P_RF)] if C > 1 else []) + [(C - 1, P_RQ), (C - 1, P_RF)]
    if ext == 2:
        ORDER = [(k, P_QA) for k in range(C)]
        for k in range(C):
            ORDER += [(k, P_RB), (k, P_RT)]
    elif ext and EORD == 1:
        # every chunk's units back to back; chunk k's topk sum on the CTAs the units leave idle
        ORDER = [(k, P_U) for k in range(C)]
        for k in range(C):
            ORDER += [] if RQE else [(k, P_RQ)]
            if k >= 1:
                ORDER += [(k - 1, P_RF)]
        ORDER += [(C - 1, P_RF)]
    elif ext:
        ORDER = []
        for k in range(C):
            ORDER += [(k, P_U)]
            if k >= 1:
                ORDER += [(k - 1, P_RF)]
            ORDER += [(k, P_RQ)]
        ORDER += [(C - 1, P_RF)]
    if GSPLIT and not any(x == P_U2 for _, x in ORDER):  # G2 items right after the unit's G1 items
        ORDER = [y for kx in ORDER for y in ([kx, (kx[0], P_U2)] if kx[1] == P_U else [kx])]
    if not any(x == P_UB for _, x in ORDER):
        ORDER += [(k, P_UB) for k in range(C)]  # empty: U(k) takes every unit slot
    if not ext:
        assert sorted(ORDER) == sorted((k, x) for k in range(C) for x in range(11 if GSPLIT else 10)), ORDER
    USPLIT = ORD == 4
    assert not (USPLIT and GSPLIT)
    assert (NP > 0) == (ORD == 13)
    assert ORD != 14 or io  # (attention RS / router items are empty only in io mode)
    assert not NP or (not ext and not USPLIT)
    UKINDS = (P_U, P_UB, P_U2)
    ORDER_P = [x for x in ORDER if x[1] not in UKINDS] if NP else ORDER
    ORDER_U = [x for x in ORDER if x[1] in UKINDS] if NP else []

    name = f"bigm_layer_h{H}_i{I}_e{E}k{K}_tp{tp}_c{C}_rg{RG}_{fb(eps)}{'g' if gemma else 'r'}_{fb(scale)}_{fb(shared_w)}" + ("_tl" if TLD else "") + f"_v14_i{io}{tailq}_c{CPU}g{GSPLIT}{G2N}p{NP}{RQU}_x{ext}h{NSBA}s{I8S}e{EORD}{'r' if RQE else ''}_q{QA}_o{ORD}g{AGATE}s{NSB}u{RUNR}r{RNS}{RNS1}a{UA}n{ZR}{NSUBR}{NSUBQ}d{RQD}r{R8}w{SXW}p{S2P}e{E4}b{BS2}x{S2E}" + (f"_rd{RDBG}" if RDBG else "") + (f"_rb{RBDBG}" if RBDBG else "")

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.jit
    def blk_red(L, tid, v, op):
        r = wave_red(v, tid % i32(64), op)
        if (tid % i32(64)) == i32(0):
            lds_st(L, i32(LSC) + (tid // i32(64)) * i32(4), r)
        gpu.barrier()
        t = lds_ld_i32(L, i32(LSC))
        for k in range_constexpr(1, NW):
            t = op(t, lds_ld_i32(L, i32(LSC + k * 4)))
        gpu.barrier()
        return t

    @flyc.jit
    def blk_red_vec(L, tid, vals, op):
        """block reduction of len(vals) <= 4 values with one barrier pair"""
        n = len(vals)
        rs_ = [wave_red(v, tid % i32(64), op) for v in vals]
        if (tid % i32(64)) == i32(0):
            for i in range_constexpr(n):
                lds_st(L, i32(LSC) + ((tid // i32(64)) * i32(n) + i32(i)) * i32(4), rs_[i])
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
        if const_expr(I8S == 4):
            return fx.Int32(bld(rx, row * i32(RB) + i32(SCO) + g * i32(4), 0, T.i32))
        w_ = fx.Int32(bld(rx, row * i32(RB) + i32(SCO) + (g // i32(4 // I8S)) * i32(4), 0, T.i32))
        return ((w_ >> ((g & i32(1)) * i32(16))) & i32(0xFFFF)) << i32(16)

    def i8_rows_ld(rx, row, p):
        return (
            fx.Vector(bld(rx, row * i32(RB) + p * i32(8), 0, T.vec(2, T.i32))),
            i8_scale_ld(rx, row, p),
        )

    def i8_rows_val(ld):
        d, sc = ld
        return _i8x4_unpack(d[0], sc.bitcast(fx.Float32)) + _i8x4_unpack(d[1], sc.bitcast(fx.Float32))

    def i8_row_add(rx, row, p):
        d = fx.Vector(bld(rx, row * i32(RB) + p * i32(8), 0, T.vec(2, T.i32)))
        sc = i8_scale_ld(rx, row, p)
        return _i8x4_unpack(d[0], sc.bitcast(fx.Float32)) + _i8x4_unpack(d[1], sc.bitcast(fx.Float32))

    # E (MoE-output RS) rows: int8 + f32 scale per 32 (RB bytes), or E4: int4 + bf16 scale per 32
    RBE = (H // 2 + H // 16) if E4 else RB

    def e_ld(rx, row, p):
        if const_expr(E4):
            d = fx.Int32(bld(rx, row * i32(RBE) + p * i32(4), 0, T.i32))
            sb = fx.Int32(bld(rx, row * i32(RBE) + i32(H // 2) + (p // i32(4)) * i32(2), 0, T.i16)) & i32(0xFFFF)
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
            sc = fx.Float32(fx.Float32(am.maximumf(fx.Float32(1e-30)) / fx.Float32(7.0)).to(fx.BFloat16).to(fx.Float32))
            bst(_i4x8_pack(f, fx.Float32(1.0) / sc), rd, drow * i32(RBE) + p * i32(4), 0, AUX_SYS)
            if (lane & i32(3)) == i32(0):
                bst(fx.Int16(sc.bitcast(fx.Int32) >> i32(16)), rd, drow * i32(RBE) + i32(H // 2) + (p // i32(4)) * i32(2), 0, AUX_SYS)
        else:
            i8_row_store(f, rd, drow, p, lane)

    @flyc.jit
    def i8_row_store(f, rd, drow, p, lane):
        am = amax(f)
        am = am.maximumf(am.shuffle_xor(i32(1), i32(64)))
        am = am.maximumf(am.shuffle_xor(i32(2), i32(64)))
        g = p // i32(4)
        sc = am.maximumf(fx.Float32(1e-30)) / fx.Float32(127.0)
        if const_expr(I8S == 2):  # quantize with the stored (bf16) scale
            sc = fx.Float32(sc.to(fx.BFloat16).to(fx.Float32))
        inv = fx.Float32(1.0) / sc
        dv = fx.Vector.from_elements([_i8x4_pack(f[0:4], inv), _i8x4_pack(f[4:8], inv)], fx.Int32)
        bst(dv, rd, drow * i32(RB) + p * i32(8), 0, AUX_SYS)
        if (lane & i32(3)) == i32(0):
            if const_expr(I8S == 4):
                bst(sc.bitcast(fx.Int32), rd, drow * i32(RB) + i32(SCO) + g * i32(4), 0, AUX_SYS)
            else:
                bst(fx.Int16(sc.bitcast(fx.Int32) >> i32(16)), rd, drow * i32(RB) + i32(SCO) + g * i32(2), 0, AUX_SYS)

    @flyc.jit
    def router_group(L, tid, a, n0, rows, rl0, es=None, lgt=None, ctr=None, rns=RNS):
        """router GEMM + topk of `rows` (<= ZRT) normed rows n0.. (local) -> ids / tw rows rl0..
        wave w: experts w*E/NW.. over the full K (no cross-wave sum); logits -> LDS -> topk"""
        lane = tid % i32(64)
        w = tid // i32(64)
        NT = E // 16 // (4 if RDBG == 3 else 1) // rns  # expert tiles per wave (the waves split K)
        EW = NT * 16  # experts of this item
        e0 = i32(0) if es is None else es * i32(EW)
        rn = rsrc(a["normed"])
        rg = rsrc(a["wg"])
        arow = (n0 + fx.min(lane % i32(16), rows - i32(1))) * i32(H)
        brow = [(e0 + i32(n * 16) + lane % i32(16)) * i32(H) for n in range(NT)]
        k0 = w * i32(H // NW) + (lane // i32(16)) * i32(8)
        UNR = RUNR
        zero = fx.Vector.filled(4, 0.0, fx.Float32)
        for k_, st in range(i32(0), i32(0 if RDBG == 1 else H // NW // 32 // UNR), i32(1), init=[zero] * NT):
            acc = list(st)
            # every load of the UNR k-steps first, then their MFMAs
            ab = []
            for u in range_constexpr(UNR):
                kc = k0 + (i32(k_) * i32(UNR) + i32(u)) * i32(32)
                ab.append(
                    (
                        bld(rn, (arow + kc) * i32(2), 0, T.vec(4, T.i32)),
                        [bld(rg, (brow[n] + kc) * i32(2), 0, T.vec(4, T.i32)) for n in range(NT)],
                    )
                )
            for u in range_constexpr(UNR):
                av = fx.Vector(ab[u][0]).bitcast(fx.BFloat16)
                for n in range_constexpr(NT):
                    bv = fx.Vector(ab[u][1][n]).bitcast(fx.BFloat16)
                    acc[n] = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [av, bv, acc[n], 0, 0, 0]))
            res = yield acc
        # partial logits [wave][16 rows][EW] f32 into LDS (the unit's A ring)
        for n in range_constexpr(NT):
            v = fx.Vector(res[n])
            for i in range_constexpr(4):
                r_ = (lane // i32(16)) * i32(4) + i32(i)
                e_ = i32(n * 16) + lane % i32(16)
                lds_st(L, ((w * i32(16) + r_) * i32(EW) + e_) * i32(4), fx.Float32(v[i]).bitcast(fx.Int32))
        gpu.barrier()
        NPW = NW  # partial logits per row to sum in the topk
        if const_expr(rns > 1):
            # sum the waves' partials -> this slice's logits [16][EW] f32 (global, system scope);
            # the last of the group's rns items reads every slice back and does the topk
            rlg = rsrc(lgt)
            for ii in range_constexpr((16 * EW + NTH - 1) // NTH):
                idx = tid + i32(ii * NTH)
                if idx < i32(16 * EW):
                    r_ = idx // i32(EW)
                    e_ = idx - r_ * i32(EW)
                    x = fx.Float32(0.0)
                    for ww in range_constexpr(NW):
                        x = x + lds_ld_i32(L, ((i32(ww * 16) + r_) * i32(EW) + e_) * i32(4)).bitcast(fx.Float32)
                    bst(x.bitcast(fx.Int32), rlg, ((n0 + r_) * i32(E) + e0 + e_) * i32(4), 0, AUX_SYS)
            wait_vm(0)
            gpu.barrier()
            if tid == i32(0):
                v_ = fx.Int32(atomic_add_agent(ctr, i32(1)))
                lds_st(L, i32(LT_R), (v_ == i32(rns - 1)).select(i32(1), i32(0)))
            gpu.barrier()
            last = lds_ld_i32(L, i32(LT_R)) == i32(1)
            gpu.barrier()
            if last:
                for ii in range_constexpr(16 * E // NTH):
                    idx = tid + i32(ii * NTH)
                    lds_st(L, idx * i32(4), fx.Int32(bld(rlg, (n0 * i32(E) + idx) * i32(4), 0, T.i32, AUX_SYS)))
            gpu.barrier()
            NPW = 1
            EW = E
        else:
            last = None
        rb_ = rsrc(a["bias"])
        ri = rsrc(a["wb"] + fx.Int64(a["off_ids"]))
        rw = rsrc(a["wb"] + fx.Int64(a["off_tw"]))
        for rr_ in range_constexpr(ZRT // NW):
            rl = w * i32(ZRT // NW) + i32(rr_)
            vals, orig, idxs = [], [], []
            for i in range_constexpr(EPL):
                e = lane + i32(i * 64)
                x = fx.Float32(0.0)
                for ww in range_constexpr(NPW):
                    x = x + lds_ld_i32(L, ((i32(ww * 16) + fx.min(rl, rows - i32(1))) * i32(E) + e) * i32(4)).bitcast(fx.Float32)
                x = fx.Float32(x).to(fx.BFloat16).to(fx.Float32)
                sc = fx.Float32(1.0) / (fx.Float32(1.0) + fx.Float32(fmath.exp2(x * fx.Float32(-LOG2E))))
                orig.append(sc)
                b = fx.Int32(bld(rb_, e * i32(4), 0, T.i32)).bitcast(fx.Float32)
                vals.append(sc + b)
                idxs.append(e)
            tot = fx.Float32(0.0)
            my_id = i32(0)
            my_w = fx.Float32(0.0)
            ninf = fx.Float32(float("-inf"))
            for k in range_constexpr(1 if RDBG == 2 else K):
                bv, bo, bi, bj = vals[0], orig[0], idxs[0], i32(0)
                for j in range_constexpr(1, EPL):
                    t_ = vals[j] > bv
                    bv = t_.select(vals[j], bv)
                    bo = t_.select(orig[j], bo)
                    bi = t_.select(idxs[j], bi)
                    bj = t_.select(i32(j), bj)
                mx = wave_red(
                    bv.bitcast(fx.Int32),
                    lane,
                    lambda p_, q_: p_.bitcast(fx.Float32).maximumf(q_.bitcast(fx.Float32)).bitcast(fx.Int32),
                ).bitcast(fx.Float32)
                bal = fx.Int64(rocdl.ballot(T.i64, bv == mx))
                win = i32(fx.ctpop(fx.Int64((bal & (fx.Int64(0) - bal)) - fx.Int64(1))))
                win = (bal == fx.Int64(0)).select(i32(0), win)
                wid = i32(rocdl.readlane(T.i32, bi, win))
                wgt = fx.Int32(rocdl.readlane(T.i32, bo.bitcast(fx.Int32), win)).bitcast(fx.Float32)
                for j in range_constexpr(EPL):
                    vals[j] = ((lane == win) & (bj == i32(j))).select(ninf, vals[j])
                tot = tot + wgt
                my_id = (lane == i32(k)).select(wid, my_id)
                my_w = (lane == i32(k)).select(wgt, my_w)
            f = fx.Float32(float(scale)) / tot.maximumf(fx.Float32(1e-20))
            ok = (rl < rows) if last is None else ((rl < rows) & last)
            if ok:
                r = rl0 + rl
                if lane < i32(K):
                    bst(my_id, ri, (r * i32(K1) + lane) * i32(4), 0, AUX_SYS)
                    bst((my_w * f).bitcast(fx.Int32), rw, (r * i32(K1) + lane) * i32(4), 0, AUX_SYS)
                if lane == i32(K):
                    bst(i32(E), ri, (r * i32(K1) + lane) * i32(4), 0, AUX_SYS)
                    bst(fx.Float32(float(shared_w)).bitcast(fx.Int32), rw, (r * i32(K1) + lane) * i32(4), 0, AUX_SYS)
        gpu.barrier()

    @flyc.kernel(name=name, known_block_size=[NTH, 1, 1])
    def kernel(
        wb: fx.Int64,  # this rank's SDMA window (local address)
        win: fx.Int64,  # window handle
        dc: fx.Int64,  # MORI device communicator
        consumed: fx.Int64,  # [tp][NQ] int32 puts awaited before this launch (persistent)
        offs: fx.Int64,  # [10] int64 window region offsets
        part: fx.Int64,  # [T, H] bf16 attention partial (every row); io: x [m, H] bf16 own rows
        resid: fx.Int64,  # [T, H] bf16 residual; io: ids [m, K1] int32
        wpost: fx.Int64,  # io: tw [m, K1] f32
        wg: fx.Int64,  # [E, H] bf16 router
        bias: fx.Int64,  # [E] f32
        wnext: fx.Int64,
        res: fx.Int64,  # [m, H] bf16 scratch; io: res_in
        normed: fx.Int64,  # [m, H] bf16 scratch; io: y [m, H] bf16 out (the MoE output)
        resout: fx.Int64,  # [m, H] bf16 out
        ybuf: fx.Int64,  # [m, H] bf16 own MoE rows
        routes: fx.Int64,  # [C][T * K1, H] bf16 route rows of a chunk (chunk stride below)
        lrix: fx.Int64,  # [C][T * K1] route lists
        lwt: fx.Int64,
        utab: fx.Int64,  # [C][8][ucap][4] units
        w1: fx.Int64,
        w1s: fx.Int64,
        w2: fx.Int64,
        w2s: fx.Int64,
        ctrl: fx.Int64,
        chunks: fx.Int64,  # [C][2] int32: (rows offset o, rows s) per rank
        logit: fx.Int64,  # [m, E] f32 router logits (expert-split router)
        stab: fx.Int64,  # ext: [C][5] int64 (sorted ids, weights, block experts, num valid, token counts)
        inter: fx.Int64,  # GSPLIT: [C][8][ucap] unit intermediate blocks
        rank: fx.Int32,
        m: fx.Int32,
        ucap: fx.Int32,
        rstride: fx.Int64,  # (unused: chunk k's routes / route lists start at its first route tp * o * K1)
        lstride: fx.Int32,
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
            "off_aq": off_of(0),
            "off_arecv": off_of(1),
            "off_xq": off_of(2),
            "off_xs": off_of(3),
            "off_ids": off_of(4),
            "off_tw": off_of(5),
            "off_eq": off_of(6),
            "off_erecv": off_of(7),
            "off_q": off_of(8),
            "off_qs": off_of(9),
            "normed": normed,
            "wg": wg,
            "bias": bias,
        }

        def ca(i):
            return ctrl + fx.Int64(i * i32(4))

        def wp(name):  # local window address of a region
            return wb + a["off_" + name]

        rch = rsrc(chunks)

        def chunk(k):  # (rows offset o, rows s) of chunk k (traced)
            return fx.Int32(bld(rch, k * i32(8), 0, T.i32)), fx.Int32(bld(rch, k * i32(8) + i32(4), 0, T.i32))

        def lbase(k):  # chunk k's first route (route lists, route rows: chunks packed)
            return i32(tp) * chunk(k)[0] * i32(K1)

        def rbase(k):  # chunk k's route rows (bytes)
            return fx.Int64(lbase(k)) * fx.Int64(H * 2)

        # this launch's signal base per (peer, queue): read once, before any item
        rcons = rsrc(consumed)

        def sig_base(p, q):
            return fx.Int32(bld(rcons, (p * i32(NQ) + i32(q)) * i32(4), 0, T.i32, AUX_SYS))

        # ---- SDMA (MORI device functions) ----
        from flydsl.compiler.extern_link import link_extern
        from flydsl.expr.extern import ffi

        from .sdma import _mori

        bc = _mori()[3](6)
        put_rsig = link_extern(
            ffi(
                "cco_sdma_put_rsig__thread",
                ["uint64", "int32", "uint64", "uint64", "uint64", "uint64", "uint64", "int32", "int32"],
                "int32",
            ),
            bitcode_path=bc,
        )
        sig_slot = link_extern(ffi("cco_sdma_signal_slot", ["uint64", "int32", "int32"], "uint64"), bitcode_path=bc)

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

        def put(peer, dst_region, dst_off, src_region, src_off, nbytes, q):
            put_rsig(
                dc,
                peer,
                win,
                a["off_" + dst_region] + fx.Int64(dst_off),
                win,
                a["off_" + src_region] + fx.Int64(src_off),
                fx.Int64(nbytes),
                i32(q),
                0,
            )

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

        rst = rsrc(stab)

        def sptr(k, i):  # ext: chunk k's sort output i (0 ids, 1 weights, 2 block experts, 3 count)
            # (4: per-token count of stored route rows, zeroed by the host)
            return fx.Int64(bld(rst, (k * i32(5) + i32(i)) * i32(8), 0, T.i64, AUX_SYS))

        def nblk(k):  # ext: chunk k's sorted blocks (after its flag)
            return fx.Int32(bld(rsrc(sptr(k, 3)), i32(0), 0, T.i32, AUX_SYS)) // i32(RG)

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
                lds_st(L, i32(LT), fx.Int32(atomic_add_agent(ca(i32(C_TK) + xc * i32(32)), i32(1))))
            gpu.barrier()
            v = lds_ld_i32(L, i32(LT))
            gpu.barrier()
            return v

        def part8(n):
            q = n // i32(8)
            r = n - q * i32(8)
            return xc * q + (xc < r).select(xc, r), q + (xc < r).select(i32(1), i32(0))

        # ---- item counts per (chunk, kind) (rows s per rank of the chunk) ----
        def rns_of(k_):
            if const_expr(RNS1 == RNS):
                return i32(RNS)
            return (i32(k_) == i32(0)).select(i32(RNS), i32(RNS1))

        def cnt(kind, s_, k_=0):
            if const_expr(io and kind in (P_QA, P_RT)):
                return i32(0)
            if const_expr(io and kind == P_RB):
                return s_ // i32(ZX)
            if const_expr(kind == P_QA):
                return (i32(tp - 1) * s_) // i32(ZQ * NSUBQ)
            if const_expr(kind == P_RB):
                return s_ // i32(ZR * NSUBR)
            if const_expr(kind == P_RT):
                return (s_ + i32(ZRT - 1)) // i32(ZRT) * rns_of(k_)
            if const_expr(kind == P_RF):
                return s_ // i32(ZQ * NSUBQ)
            if const_expr(kind in (P_S1, P_S3)):
                return (i32(tp) * s_ + i32(SRT - 1)) // i32(SRT)
            if const_expr(kind == P_S2):
                return i32(1)
            if const_expr(kind == P_U):
                return ucap * i32(8)
            return (i32(tp) * s_) // i32(ZQ * NSUBQ)

        def chunk_v(k):  # per-claim (not hoisted: system-scope load)
            return fx.Int32(bld(rch, k * i32(8) + i32(4), 0, T.i32, AUX_SYS))

        def slices_of(sk, order=None):
            order = ORDER_P if order is None else order
            out = []
            tot_ = i32(0)
            for oi in range_constexpr(len(order)):
                k, kind = order[oi]
                n = cnt(kind, sk[k], k)
                if const_expr(kind == P_S2):
                    lo_, c_ = i32(0), (xc == i32(0)).select(i32(1), i32(0))
                elif const_expr(kind == P_U):
                    lo_, c_ = i32(0), (fx.min(ucap, i32(UA)) if USPLIT else ucap * i32(PASSES_U) if GSPLIT else ucap)
                elif const_expr(kind == P_U2):
                    lo_, c_ = i32(0), ucap * i32(NSUB2)
                elif const_expr(kind == P_UB):
                    lo_, c_ = i32(UA), (fx.max(ucap - i32(UA), i32(0)) if USPLIT else i32(0))
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
        @flyc.jit
        def post_a(k, b):
            o, s = chunk(k)
            B = i32(tp) * o
            sa = s // i32(NSBA)
            tn = k * i32(NSBA) + b
            take_turn(QA, tn)
            for p_ in range_constexpr(tp):
                if i32(p_) != rank:
                    put(
                        i32(p_), "arecv", (B + rank * s + b * sa) * i32(RB), "aq", (B + i32(p_) * s + b * sa) * i32(RB),
                        sa * i32(RB), QA,
                    )
            give_turn(QA, tn)

        @flyc.jit
        def gate_ab(k, b):
            # thread 0: the second of (QA(k) sub-block b done, C(k - 1) posted) posts A(k, b);
            # chunk 0 needs no C
            v = fx.Int32(atomic_add_agent(ca(i32(C_AGATE) + k * i32(NSBA) + b), i32(1)))
            if (k == i32(0)) | (v == i32(1)) | (i32(AGATE) == i32(0)):
                post_a(k, b)

        def gate_a(k):  # C(k - 1) posted / landed: every head sub-block of chunk k
            for bq in range_constexpr(NSBA):
                gate_ab(k, i32(bq))

        def sbh(k, w, b):  # head sub-block done counter (w 0: QA items, 1: RT items)
            return ca(i32(C_SBH) + ((k * i32(2) + i32(w)) * i32(16) + b) * i32(8))

        @flyc.jit
        def item_qa(k, j):
            tid = _opaque(tid0)
            lane = tid % i32(64)
            o, s = chunk(k)
            B = i32(tp) * o
            rd = rsrc(wp("aq"))
            # head sub-block major (XCD-interleaved): sub-block b's rows of every peer first
            sa = s // i32(NSBA)
            nbq = (i32(tp - 1) * sa) // i32(ZQ * NSUBQ)
            jx = il(j, cnt(P_QA, s))
            bh = jx // nbq
            rem = (jx - bh * nbq) * i32(ZQ * NSUBQ)
            qb = rem // sa
            wrow0 = qb * s + bh * sa + (rem - qb * sa)
            for sbi in range_constexpr(NSUBQ):
                rows_ = []
                for ii in range_constexpr(ZQ):
                    wrow = wrow0 + i32(sbi * ZQ + ii)
                    blk = wrow // s
                    i = wrow - blk * s
                    q = blk + (blk >= rank).select(i32(1), i32(0))
                    srow = q * m + o + i
                    rows_.append(
                        (B + q * s + i, [bld(r_part, (srow * i32(H) + (tid + i32(kk * NTH)) * i32(8)) * i32(2), 0, T.vec(4, T.i32)) for kk in range(PIT)])
                    )
                for ii in range_constexpr(ZQ):
                    drow, lds_ = rows_[ii]
                    for kk in range_constexpr(PIT):
                        i8_row_store(bf16x8_to_f32(lds_[kk]), rd, drow, tid + i32(kk * NTH), lane)
            done_last(k, P_QA, cnt(P_QA, s))
            if done_at(sbh(k, 0, bh), nbq):
                if tid == i32(0):
                    gate_ab(k, bh)

        @flyc.jit
        def item_rb(k, j):
            tid = _opaque(tid0)
            lane = tid % i32(64)
            o, s = chunk(k)
            B = i32(tp) * o
            jx = il(j, cnt(P_RB, s))
            wait_puts_n(QA, k * i32(NSBA) + (jx * i32(ZR * NSUBR)) // (s // i32(NSBA)) + i32(1))
            gpu.barrier()
            rv = rsrc(wp("arecv"))
            rq, rqs = rsrc(wp("xq")), rsrc(wp("xs"))
            def load_rows(sbi_):
                ld = []
                for ii in range_constexpr(ZR):
                    i = (jx * i32(NSUBR) + i32(sbi_)) * i32(ZR) + i32(ii)
                    g = rank * m + o + i
                    row = []
                    for kk in range_constexpr(PIT):
                        p = tid + i32(kk * NTH)
                        goff = (g * i32(H) + p * i32(8)) * i32(2)
                        row.append(
                            (
                                bld(r_part, goff, 0, T.vec(4, T.i32)),
                                [i8_rows_ld(rv, B + i32(src_) * s + i, p) for src_ in range(1 if RBDBG == 2 else tp)],
                                bld(r_resid, goff, 0, T.vec(4, T.i32)),
                            )
                        )
                    ld.append(row)
                return ld

            ld_next = load_rows(0)
            for sbi in range_constexpr(NSUBR):
                ld = ld_next
                ld_next = load_rows(sbi + 1) if sbi + 1 < NSUBR else None
                fss, sss = [], []
                for ii in range_constexpr(ZR):
                    i = (jx * i32(NSUBR) + i32(sbi)) * i32(ZR) + i32(ii)
                    fs = []
                    ss = fx.Float32(0.0)
                    for kk in range_constexpr(PIT):
                        p = tid + i32(kk * NTH)
                        lp_, lr, lres = ld[ii][kk]
                        f = bf16x8_to_f32(lp_)
                        for src_ in range_constexpr(1 if RBDBG == 2 else tp):
                            f = sum_live(f, i8_rows_val(lr[src_]), i32(src_) != rank)
                        f = [x + y for x, y in zip(f, bf16x8_to_f32(lres))]
                        if const_expr(RBDBG != 1):
                            bst(pack_bf16x8(f), r_res, ((o + i) * i32(H) + p * i32(8)) * i32(2), 0, AUX_SYS)
                        for x in f:
                            ss = ss + x * x
                        fs.append(f)
                    fss.append(fs)
                    sss.append(ss.bitcast(fx.Int32))
                tots = blk_red_vec(L, tid, sss, _fadd)
                for ii in range_constexpr(ZR):
                    i = (jx * i32(NSUBR) + i32(sbi)) * i32(ZR) + i32(ii)
                    rcp = fx.Float32(fmath.rsqrt(tots[ii].bitcast(fx.Float32) / fx.Float32(float(H)) + fx.Float32(float(eps))))
                    row0 = B + rank * s + i
                    for kk in range_constexpr(PIT):
                        p = tid + i32(kk * NTH)
                        wv = norm_w(r_wpost, p)
                        y = [fx.Float32(fx.Float32(gate_w(x * rcp, wk)).to(fx.BFloat16).to(fx.Float32)) for x, wk in zip(fss[ii][kk], wv)]
                        if const_expr(RBDBG != 1):
                            bst(pack_bf16x8(y), r_norm, ((o + i) * i32(H) + p * i32(8)) * i32(2), 0, AUX_SYS)
                        am = amax(y)
                        am = am.maximumf(am.shuffle_xor(i32(1), i32(64)))
                        am = am.maximumf(am.shuffle_xor(i32(2), i32(64)))
                        e8, qs = _e8m0_from_amax(am, max_norm=6.0)
                        bst(fp4_pack(y, qs), rq, row0 * i32(H // 2) + p * i32(4), 0, AUX_SYS)
                        if (lane & i32(3)) == i32(0):
                            bst(fx.Int8(e8), rqs, row0 * i32(H // 32) + p // i32(4), 0, AUX_SYS)
            wait_vm(0)
            gpu.barrier()
            if tid == i32(0):
                atomic_add_agent(ca(i32(C_RTG) + k * i32(256) + (jx * i32(ZR * NSUBR)) // i32(ZRT)), i32(NSUBR))
            done_last(k, P_RB, cnt(P_RB, s))

        @flyc.jit
        def item_rt(k, jr0):
            if const_expr(RNS1 == RNS):
                rt_body(k, jr0, RNS)
            elif k == i32(0):
                rt_body(k, jr0, RNS)
            else:
                rt_body(k, jr0, RNS1)

        @flyc.jit
        def rt_body(k, jr0, rns):
            tid = _opaque(tid0)
            o0, s0 = chunk(k)
            jr = il(jr0, cnt(P_RT, s0, k))
            g = jr // i32(rns)
            lane = tid % i32(64)
            o, s = chunk(k)
            B = i32(tp) * o
            rows = fx.min(s - g * i32(ZRT), i32(ZRT))
            if tid == i32(0):
                cur = fx.Int32(load_i32_global_agent(ca(i32(C_RTG) + k * i32(256) + g)))
                while cur < rows // i32(ZR):
                    rocdl.s_sleep(1)
                    cur = fx.Int32(load_i32_global_agent(ca(i32(C_RTG) + k * i32(256) + g)))
            gpu.barrier()
            router_group(
                L, tid, a, o + g * i32(ZRT), rows, B + rank * s + g * i32(ZRT),
                jr - g * i32(rns), logit, ca(i32(C_RTX) + k * i32(256) + g), rns,
            )
            done_last(k, P_RT, cnt(P_RT, s, k))
            sa = s // i32(NSBA)
            bh = (g * i32(ZRT)) // sa
            if done_at(sbh(k, 1, bh), (sa // i32(ZRT)) * i32(rns)):
                if tid == i32(0):
                    post_c(k, bh)

        @flyc.jit
        def post_c(k, bh):
            # thread 0: this head sub-block's quantized rows + routing -> every peer
            o, s = chunk(k)
            B = i32(tp) * o
            sa = s // i32(NSBA)
            r0b = B + rank * s + bh * sa
            t1 = k * i32(NSBA) + bh
            CPUTS = ((QX, "xq", H // 2), (QXS, "xs", H // 32), (QID, "ids", K1 * 4), (QID, "tw", K1 * 4))
            for ci in range_constexpr(4):
                q_, reg, rowb = CPUTS[ci]
                tq = t1 * i32(2) if const_expr(q_ == QID) else t1
                if const_expr(reg != "tw"):
                    take_turn(q_, tq)
                for p_ in range_constexpr(tp):
                    if i32(p_) != rank:
                        put(i32(p_), reg, r0b * i32(rowb), reg, r0b * i32(rowb), sa * i32(rowb), q_)
                if const_expr(reg in ("xq", "xs")):
                    give_turn(q_, tq)
                if const_expr(reg == "tw"):
                    give_turn(q_, tq + i32(1))
            if const_expr(not io):
                cp = fx.Int32(atomic_add_agent(ca(i32(C_CPOST) + k * i32(8)), i32(1)))
                if (cp == i32(NSBA - 1)) & (k + i32(1) < i32(C)) & (i32(AGATE) == i32(1)):
                    gate_a(k + i32(1))

        @flyc.jit
        def item_rx(k, j):
            # io: ZX of this rank's (normed) rows -> MXFP4 rows + routing in the window; the last
            # item of a head sub-block posts it to every peer (gather C)
            tid = _opaque(tid0)
            lane = tid % i32(64)
            o, s = chunk(k)
            B = i32(tp) * o
            jx = il(j, cnt(P_RB, s))
            rq, rqs = rsrc(wp("xq")), rsrc(wp("xs"))
            i0 = jx * i32(ZX)
            ld = [
                [bld(r_part, ((o + i0 + i32(ii)) * i32(H) + (tid + i32(kk * NTH)) * i32(8)) * i32(2), 0, T.vec(4, T.i32)) for kk in range(PIT)]
                for ii in range(ZX)
            ]
            if tid < i32(ZX * K1):
                rr_ = tid // i32(K1)
                src = (o + i0 + rr_) * i32(K1) + (tid - rr_ * i32(K1))
                dst = (B + rank * s + i0) * i32(K1) + tid
                bst(fx.Int32(bld(r_resid, src * i32(4), 0, T.i32)), rsrc(wp("ids")), dst * i32(4), 0, AUX_SYS)
                bst(fx.Int32(bld(r_wpost, src * i32(4), 0, T.i32)), rsrc(wp("tw")), dst * i32(4), 0, AUX_SYS)
            for ii in range_constexpr(ZX):
                row0 = B + rank * s + i0 + i32(ii)
                for kk in range_constexpr(PIT):
                    p = tid + i32(kk * NTH)
                    y = bf16x8_to_f32(ld[ii][kk])
                    am = amax(y)
                    am = am.maximumf(am.shuffle_xor(i32(1), i32(64)))
                    am = am.maximumf(am.shuffle_xor(i32(2), i32(64)))
                    e8, qs = _e8m0_from_amax(am, max_norm=6.0)
                    bst(fp4_pack(y, qs), rq, row0 * i32(H // 2) + p * i32(4), 0, AUX_SYS)
                    if (lane & i32(3)) == i32(0):
                        bst(fx.Int8(e8), rqs, row0 * i32(H // 32) + p // i32(4), 0, AUX_SYS)
            done_last(k, P_RB, cnt(P_RB, s))
            sa = s // i32(NSBA)
            bh = i0 // sa
            if done_at(sbh(k, 1, bh), sa // i32(ZX)):
                if tid == i32(0):
                    post_c(k, bh)

        @flyc.jit
        def item_s1(k, j):
            tid = _opaque(tid0)
            lane = tid % i32(64)
            o, s = chunk(k)
            B = i32(tp) * o
            if const_expr(SXW):
                wait_puts(QX, NSBA, k)
                wait_puts(QXS, NSBA, k)
            wait_puts(QID, 2 * NSBA, k)
            if const_expr(io):
                wait_ph(k, P_RB, cnt(P_RB, s))  # own rows' routing copied into the window
            else:
                wait_ph(k, P_RT, cnt(P_RT, s, k))  # own rows' routing written (and on the peers)
            if tid < i32(NE):
                lds_st(L, i32(LH) + tid * i32(4), i32(0))
            gpu.barrier()
            rids = rsrc(wp("ids"))
            r0 = (B + j * i32(SRT)) * i32(K1)
            nr = fx.min((B + i32(tp) * s) * i32(K1) - r0, i32(SRT * K1))
            for kk in range_constexpr((SRT * K1 + NTH - 1) // NTH):
                r = tid + i32(kk * NTH)
                if r < nr:
                    e = fx.Int32(bld(rids, (r0 + r) * i32(4), 0, T.i32, AUX_SYS))
                    lds_atomic_add(L, i32(LH) + e * i32(4), i32(1))
            gpu.barrier()
            if tid < i32(NE):
                c = lds_ld_i32(L, i32(LH) + tid * i32(4))
                if c > i32(0):
                    atomic_add_agent(ca(i32(C_CNT) + k * i32(256) + tid), c)
            if done_last(k, P_S1, cnt(P_S1, s)):
                # AGATE 2: A(k+1) once C(k) has arrived (the gathers do not share the links)
                if (tid == i32(0)) & (k + i32(1) < i32(C)) & (i32(AGATE) == i32(2)):
                    gate_a(k + i32(1))

        @flyc.jit
        def item_s2(k):
            tid = _opaque(tid0)
            lane = tid % i32(64)
            o, s = chunk(k)
            wait_ph(k, P_S1, cnt(P_S1, s))
            # route-list offsets (prefix of the counts, each thread its own expert)
            if tid < i32(NE):
                lds_st(L, i32(LH) + tid * i32(4), fx.Int32(load_i32_global_agent(ca(i32(C_CNT) + k * i32(256) + tid))))
            gpu.barrier()
            if tid < i32(NE):
                acc = i32(0)
                for e_ in range_constexpr(NE - 1):
                    acc = acc + (i32(e_) < tid).select(lds_ld_i32(L, i32(LH + 4 * e_)), i32(0))
                g_st_sys(ca(i32(C_OFF) + k * i32(256) + tid), acc)
                lds_st(L, i32(LB2) + tid * i32(4), acc)
            gpu.barrier()
            if const_expr(S2E):
                # the route-list offsets are out: the sort items (S3) start (count 1); the unit plan
                # below completes the count (2: units, topk sums)
                done_last(k, P_S2, i32(2))
            if const_expr(S2P):
                s2_parallel(k)
            else:
                s2_greedy(k)
            done_last(k, P_S2, i32(2) if S2E else i32(1))

        @flyc.jit
        def s2_parallel(k):
            # units in expert order (the prefix of every expert's unit count), XCD x the equal
            # contiguous range [ceil(x tot / 8), ceil((x + 1) tot / 8)): balanced unit counts,
            # an expert's units on one XCD (but at a range boundary); every expert in parallel
            tid = _opaque(tid0)
            rut = rsrc(utab)
            if tid < i32(NE):
                lds_st(L, i32(LP) + tid * i32(4), (lds_ld_i32(L, i32(LH) + tid * i32(4)) + i32(RG - 1)) // i32(RG))
            gpu.barrier()
            if tid < i32(NE):
                c = lds_ld_i32(L, i32(LH) + tid * i32(4))
                off = lds_ld_i32(L, i32(LB2) + tid * i32(4))
                nu = (c + i32(RG - 1)) // i32(RG)
                ue = i32(0)
                tot = i32(0)
                for e_ in range_constexpr(NE):
                    v = lds_ld_i32(L, i32(LP + 4 * e_))
                    ue = ue + (i32(e_) < tid).select(v, i32(0))
                    tot = tot + v
                totc = fx.max(tot, i32(1))
                for u_ in range(i32(0), nu, i32(1)):
                    u = i32(u_)
                    g = ue + u
                    x_ = (g * i32(8)) // totc
                    slot = fx.min(g - (x_ * totc + i32(7)) // i32(8), ucap - i32(1))
                    ent = ((k * i32(8) + x_) * ucap + slot) * i32(16)
                    bst(tid, rut, ent, 0, AUX_SYS)
                    bst(off + u * i32(RG), rut, ent + i32(4), 0, AUX_SYS)
                    bst(fx.min(c - u * i32(RG), i32(RG)), rut, ent + i32(8), 0, AUX_SYS)
                if tid < i32(8):
                    nx = ((tid + i32(1)) * tot + i32(7)) // i32(8) - (tid * tot + i32(7)) // i32(8)
                    g_st_sys(ca(i32(C_UN) + tid * i32(32) + k * i32(256)), fx.min(nx, ucap))

        @flyc.jit
        def s2_greedy(k):
            tid = _opaque(tid0)
            # units -> per-XCD queues: the shared expert's units round-robin (fixed slots), a
            # routed expert's units appended to queue e % 8 (its weights shared in that XCD's
            # L2), the next queue when one is full
            rut = rsrc(utab)
            ns = (lds_ld_i32(L, i32(LH + 4 * E)) + i32(RG - 1)) // i32(RG)
            if tid < i32(8):
                lds_st(L, i32(LP) + tid * i32(4), (ns - tid + i32(7)) // i32(8))
            gpu.barrier()
            soff = lds_ld_i32(L, i32(LB2 + 4 * E))
            cs_ = lds_ld_i32(L, i32(LH + 4 * E))
            for u_ in range(tid, ns, i32(NTH)):
                u = i32(u_)
                ent = ((k * i32(8) + u % i32(8)) * ucap + u // i32(8)) * i32(16)
                bst(i32(E), rut, ent, 0, AUX_SYS)
                bst(soff + u * i32(RG), rut, ent + i32(4), 0, AUX_SYS)
                bst(fx.min(cs_ - u * i32(RG), i32(RG)), rut, ent + i32(8), 0, AUX_SYS)
            # routed experts: greedy least-loaded queue (thread 0, loads in registers), each
            # expert's queue / first slot to LDS; then every expert's units in parallel
            LEQ = LP + 64
            if tid == i32(0):
                ld = [lds_ld_i32(L, i32(LP + 4 * x)) for x in range(8)]
                for e_ in range_constexpr(E):
                    nu = (lds_ld_i32(L, i32(LH + 4 * e_)) + i32(RG - 1)) // i32(RG)
                    best = i32(0)
                    bl = ld[0]
                    for x in range_constexpr(1, 8):
                        t_ = ld[x] < bl
                        best = t_.select(i32(x), best)
                        bl = t_.select(ld[x], bl)
                    lds_st(L, i32(LEQ + 4 * e_), best | (bl << i32(8)))
                    for x in range_constexpr(8):
                        ld[x] = ld[x] + (best == i32(x)).select(nu, i32(0))
                for x in range_constexpr(8):
                    lds_st(L, i32(LP + 4 * x), ld[x])
            gpu.barrier()
            if tid < i32(E):
                c = lds_ld_i32(L, i32(LH) + tid * i32(4))
                off = lds_ld_i32(L, i32(LB2) + tid * i32(4))
                nu = (c + i32(RG - 1)) // i32(RG)
                qb = lds_ld_i32(L, i32(LEQ) + tid * i32(4))
                x_ = qb & i32(0xFF)
                b0 = qb >> i32(8)
                for u_ in range(i32(0), nu, i32(1)):
                    u = i32(u_)
                    slot = fx.min(b0 + u, ucap - i32(1))
                    ent = ((k * i32(8) + x_) * ucap + slot) * i32(16)
                    bst(tid, rut, ent, 0, AUX_SYS)
                    bst(off + u * i32(RG), rut, ent + i32(4), 0, AUX_SYS)
                    bst(fx.min(c - u * i32(RG), i32(RG)), rut, ent + i32(8), 0, AUX_SYS)
            gpu.barrier()
            if tid < i32(8):
                g_st_sys(ca(i32(C_UN) + tid * i32(32) + k * i32(256)), fx.min(lds_ld_i32(L, i32(LP) + tid * i32(4)), ucap))

        @flyc.jit
        def item_s3(k, j):
            tid = _opaque(tid0)
            lane = tid % i32(64)
            o, s = chunk(k)
            B = i32(tp) * o
            wait_ph(k, P_S2, i32(1))  # the route-list offsets (S2E: before the unit plan)
            if tid < i32(NE):
                lds_st(L, i32(LH) + tid * i32(4), i32(0))
            gpu.barrier()
            rids, rtw = rsrc(wp("ids")), rsrc(wp("tw"))
            r0 = (B + j * i32(SRT)) * i32(K1)
            nr = fx.min((B + i32(tp) * s) * i32(K1) - r0, i32(SRT * K1))
            NRI = (SRT * K1 + NTH - 1) // NTH
            es, rk = [], []
            for kk in range_constexpr(NRI):
                r = tid + i32(kk * NTH)
                e = fx.Int32(bld(rids, (fx.min(r, nr - i32(1)) + r0) * i32(4), 0, T.i32, AUX_SYS))
                es.append(e)
                rk.append(lds_atomic_add(L, i32(LH) + e * i32(4), (r < nr).select(i32(1), i32(0))))
            gpu.barrier()
            if tid < i32(NE):
                c = lds_ld_i32(L, i32(LH) + tid * i32(4))
                base = fx.Int32(atomic_add_agent(ca(i32(C_FILL) + k * i32(256) + tid), c))
                lds_st(L, i32(LB2) + tid * i32(4), fx.Int32(load_i32_global_agent(ca(i32(C_OFF) + k * i32(256) + tid))) + base)
            gpu.barrier()
            rl, rlw = rsrc(lrix), rsrc(lwt)
            for kk in range_constexpr(NRI):
                r = tid + i32(kk * NTH)
                if r < nr:
                    pos = lds_ld_i32(L, i32(LB2) + es[kk] * i32(4)) + rk[kk]
                    lo_ = lbase(k) + pos
                    bst((r0 + r) - B * i32(K1), rl, lo_ * i32(4), 0, AUX_SYS)  # chunk-local route index
                    bst(fx.Int32(bld(rtw, (r0 + r) * i32(4), 0, T.i32, AUX_SYS)), rlw, lo_ * i32(4), 0, AUX_SYS)
            done_last(k, P_S3, cnt(P_S3, s))

        @flyc.jit
        def item_u(k, j):
            if const_expr(ext):
                item_u_ext(k, j)
            else:
                item_u_own(k, j)

        @flyc.jit
        def item_u_ext(k, j):
            # chunk k's sorted blocks: XCD x runs the contiguous range [x nb / 8, (x+1) nb / 8)
            # (a routed expert's blocks are adjacent: its weights stay in one XCD's L2)
            tid = _opaque(tid0)
            lane = tid % i32(64)
            o, s = chunk(k)
            B = i32(tp) * o
            wait_puts(QX, NSBA, k)  # this chunk's quantized rows from every peer
            wait_puts(QXS, NSBA, k)
            gpu.barrier()
            nb = nblk(k)
            b = (xc * nb) // i32(8) + j
            if b < ((xc + i32(1)) * nb) // i32(8):
                rid = rsrc(sptr(k, 0))
                if tid == i32(0):
                    lds_st(L, i32(LT) + i32(12), i32(0))
                gpu.barrier()
                if tid < i32(RG):
                    v_ = fx.Int32(bld(rid, (b * i32(RG) + tid) * i32(4), 0, T.i32, AUX_SYS))
                    if (v_ & i32(0xFFFFFF)) < i32(tp) * s:
                        lds_atomic_add(L, i32(LT) + i32(12), i32(1))
                gpu.barrier()
                rows = uni(lds_ld_i32(L, i32(LT) + i32(12)))
                e = uni(fx.Int32(bld(rsrc(sptr(k, 2)), b * i32(4), 0, T.i32, AUX_SYS)))
                rx = rsrc(wp("xq") + fx.Int64(B * i32(H // 2)))
                rxs = rsrc(wp("xs") + fx.Int64(B * i32(H // 32)))
                nrt = i32(tp) * s * i32(K1)
                r_rt = rsrc(routes + rbase(k), nrt * i32(H * 2))
                unit_body(
                    L, tid, e, b * i32(RG), rows, rid, rsrc(sptr(k, 1)), rx, rxs, w1, w2, rsrc(w1s), rsrc(w2s),
                    r_rt, nrt * i32(H * 2),
                )
                if const_expr(RQE):
                    rq_eager(k, b, rows, rid)
                done_last(k, P_U, i32(-1))

        @flyc.jit
        def rq_eager(k, b, rows, rid):
            # this unit's rows are stored: count them per token; the tokens this unit completes
            # (K1 route rows stored) are summed here (own rows -> y, peer rows -> int8 E rows),
            # then counted per sub-block; the unit completing a sub-block posts its E puts
            tid = _opaque(tid0)
            lane = tid % i32(64)
            o, s = chunk(k)
            B = i32(tp) * o
            sb = s // i32(NSB)
            wait_vm(0)
            gpu.barrier()
            if tid == i32(0):
                lds_st(L, i32(LT) + i32(16), i32(0))
                lds_st(L, i32(LT) + i32(20), i32(0))
            gpu.barrier()
            tcnt = sptr(k, 4)
            if tid < rows:
                v_ = fx.Int32(bld(rid, (b * i32(RG) + tid) * i32(4), 0, T.i32, AUX_SYS))
                tok = v_ & i32(0xFFFFFF)
                c_ = fx.Int32(atomic_add_agent(tcnt + fx.Int64(tok * i32(4)), i32(1)))
                if c_ == i32(K1 - 1):
                    ix = lds_atomic_add(L, i32(LT) + i32(16), i32(1))
                    lds_st(L, i32(LP) + ix * i32(4), tok)
            gpu.barrier()
            nt = lds_ld_i32(L, i32(LT) + i32(16))
            rx = rsrc(routes + rbase(k))
            rd = rsrc(wp("eq"))
            for i_ in range(i32(0), nt, i32(1)):
                wrow = lds_ld_i32(L, i32(LP) + i32(i_) * i32(4))
                own = (wrow // s) == rank
                ld = [
                    [
                        bld(rx, ((wrow * i32(K1) + i32(t)) * i32(H) + tid * i32(8) + i32(kk * NTH * 8)) * i32(2), 0, T.vec(4, T.i32), AUX_SYS)
                        for t in range(K1)
                    ]
                    for kk in range(PIT)
                ]
                for kk in range_constexpr(PIT):
                    p = tid + i32(kk * NTH)
                    acc = [fx.Float32(0.0)] * 8
                    for t in range_constexpr(K1):
                        acc = [a_ + b_ for a_, b_ in zip(acc, bf16x8_to_f32(ld[kk][t]))]
                    yb = pack_bf16x8(acc)
                    if own:
                        bst(yb, r_y, ((o + wrow - rank * s) * i32(H) + p * i32(8)) * i32(2), 0, AUX_SYS)
                    else:
                        e_store(bf16x8_to_f32(yb), rd, B + wrow, p, lane)
            wait_vm(0)
            gpu.barrier()
            # sub-block counts (tokens of rows [b sb, (b+1) sb) of every rank): completed ones -> LDS
            if tid < nt:
                wrow = lds_ld_i32(L, i32(LP) + tid * i32(4))
                bb = (wrow % s) // sb
                c2 = fx.Int32(atomic_add_agent(sbc(k, 0, bb), i32(1)))
                if c2 == i32(tp) * sb - i32(1):
                    lds_atomic_or(L, i32(LT) + i32(20), i32(1) << bb)
            gpu.barrier()
            if tid == i32(0):
                msk = lds_ld_i32(L, i32(LT) + i32(20))
                for bq in range_constexpr(NSB):
                    if ((msk >> i32(bq)) & i32(1)) == i32(1):
                        tn = k * i32(NSB) + i32(bq)
                        take_turn(QE, tn)
                        for p_ in range_constexpr(tp):
                            if i32(p_) != rank:
                                put(
                                    i32(p_), "erecv", (B + rank * s + i32(bq) * sb) * i32(RB), "eq",
                                    (B + i32(p_) * s + i32(bq) * sb) * i32(RB), sb * i32(RB), QE,
                                )
                        give_turn(QE, tn)

        @flyc.jit
        def item_u_own(k, j0):
            tid = _opaque(tid0)
            lane = tid % i32(64)
            o, s = chunk(k)
            B = i32(tp) * o
            wait_ph(k, P_S3, cnt(P_S3, s))
            if const_expr(S2E):
                wait_ph(k, P_S2, i32(2))  # the unit plan
            if const_expr(not SXW):  # the plan only waited for the routing: the rows now
                wait_puts(QX, NSBA, k)
                wait_puts(QXS, NSBA, k)
                gpu.barrier()
            nu = fx.Int32(load_i32_global_agent(ca(i32(C_UN) + k * i32(256) + xc * i32(32))))
            if const_expr(GSPLIT):
                j = j0 // i32(PASSES_U)  # unit slot; pass j0 % PASSES_U
            else:
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
                if const_expr(GSPLIT):
                    uid = (k * i32(8) + xc) * ucap + j
                    unit_g1(
                        L, tid, e, r0, rows, rl, rlw, rx, rxs, w1, w2, rsrc(w1s), rsrc(w2s), r_rt, nrt * i32(H * 2),
                        # passes of a unit in reverse order (forward order corrupts rows: open issue,
                        # see opt_summ_1012_2_v15; reverse verified bitwise against the whole-unit G1)
                        i32(0), rsrc(inter), i32(PASSES_U - 1) - (j0 - j * i32(PASSES_U)), uid,
                    )
                    wait_vm(0)
                    gpu.barrier()
                    if tid == i32(0):
                        atomic_add_agent(ca(i32(C_UG1)) + fx.Int64(uid * i32(4)), i32(1))
                else:
                    unit_body(
                        L, tid, e, r0, rows, rl, rlw, rx, rxs, w1, w2, rsrc(w1s), rsrc(w2s), r_rt, nrt * i32(H * 2)
                    )
                    done_last(k, P_U, i32(-1))  # counts units done (no single last)

        @flyc.jit
        def item_u2(k, j0):
            # GEMM2 column groups [(j0 % NSUB2) G2N, ..) of unit slot j0 / NSUB2, once its passes are in
            tid = _opaque(tid0)
            o, s = chunk(k)
            B = i32(tp) * o
            wait_ph(k, P_S3, cnt(P_S3, s))
            if const_expr(S2E):
                wait_ph(k, P_S2, i32(2))
            nu = fx.Int32(load_i32_global_agent(ca(i32(C_UN) + k * i32(256) + xc * i32(32))))
            j = j0 // i32(NSUB2)
            if j < nu:
                uid = (k * i32(8) + xc) * ucap + j
                wait_at(ca(i32(C_UG1)) + fx.Int64(uid * i32(4)), i32(PASSES_U))
                rut = rsrc(utab)
                ent = ((k * i32(8) + xc) * ucap + j) * i32(16)
                e = uni(fx.Int32(bld(rut, ent, 0, T.i32, AUX_SYS)))
                r0 = uni(fx.Int32(bld(rut, ent + i32(4), 0, T.i32, AUX_SYS)))
                rows = uni(fx.Int32(bld(rut, ent + i32(8), 0, T.i32, AUX_SYS)))
                rl = rsrc(lrix + fx.Int64(lbase(k) * i32(4)))
                rlw = rsrc(lwt + fx.Int64(lbase(k) * i32(4)))
                nrt = i32(tp) * s * i32(K1)
                r_rt = rsrc(routes + rbase(k), nrt * i32(H * 2))
                unit_g2(
                    L, tid, e, r0, rows, rl, rlw, rl, rl, w1, w2, rsrc(w1s), rsrc(w2s), r_rt, nrt * i32(H * 2),
                    i32(0), rsrc(inter), j0 - j * i32(NSUB2), uid,
                )
                wait_vm(0)  # the route rows are stored before they are counted
                gpu.barrier()
                if tid == i32(0):
                    atomic_add_agent(ph(k, P_U2), i32(1))

        @flyc.jit
        def item_rq(k, j):
            tid = _opaque(tid0)
            lane = tid % i32(64)
            o, s = chunk(k)
            B = i32(tp) * o
            if const_expr(ext):
                un = nblk(k)
            else:
                wait_ph(k, P_S2, i32(S2N))  # the unit plan (C_UN)
                un = i32(0)
                for x in range_constexpr(8):
                    un = un + fx.Int32(load_i32_global_agent(ca(i32(C_UN + 32 * x) + k * i32(256))))
            if const_expr(GSPLIT):
                wait_ph(k, P_U2, un * i32(NSUB2))
            else:
                wait_ph(k, P_U, un)
            rx = rsrc(routes + rbase(k))
            rd = rsrc(wp("eq"))
            sb = s // i32(NSB)
            nb = (i32(tp) * sb) // i32(ZQ * NSUBQ)
            jx = il(j, cnt(P_RQ, s))
            b = jx // nb
            rem = (jx - b * nb) * i32(ZQ * NSUBQ)
            pd = rem // sb
            w0 = pd * s + b * sb + (rem - pd * sb)  # first (dest rank, row) of the item
            own = pd == rank

            nrt_ = i32(tp) * s * i32(K1)

            def rq_load(ii):  # token ii of the item: its K1 route rows (this thread's columns)
                if const_expr(R8):
                    return [
                        [
                            (
                                bld(rx, ((w0 + i32(ii)) * i32(K1) + i32(t)) * i32(H) + (tid + i32(kk * NTH)) * i32(8), 0, T.vec(2, T.i32)),
                                bld(rx, nrt_ * i32(H) + ((w0 + i32(ii)) * i32(K1) + i32(t)) * i32(H // 32) + (tid + i32(kk * NTH)) // i32(4), 0, T.i8),
                            )
                            for t in range(K1)
                        ]
                        for kk in range(PIT)
                    ]
                return [
                    [
                        bld(rx, (((w0 + i32(ii)) * i32(K1) + i32(t)) * i32(H) + tid * i32(8) + i32(kk * NTH * 8)) * i32(2), 0, T.vec(4, T.i32))
                        for t in range(K1)
                    ]
                    for kk in range(PIT)
                ]

            def rq_store(ii, ld):
                wrow = w0 + i32(ii)
                for kk in range_constexpr(PIT):
                    p = tid + i32(kk * NTH)
                    acc = [fx.Float32(0.0)] * 8
                    for t in range_constexpr(K1):
                        v = fp8x8_decode(ld[kk][t]) if const_expr(R8) else bf16x8_to_f32(ld[kk][t])
                        acc = [a_ + b_ for a_, b_ in zip(acc, v)]
                    yb = pack_bf16x8(acc)
                    if own:
                        bst(yb, r_y, ((o + wrow - rank * s) * i32(H) + p * i32(8)) * i32(2), 0, AUX_SYS)
                    else:
                        e_store(bf16x8_to_f32(yb), rd, B + wrow, p, lane)

            NTQ = ZQ * NSUBQ
            # RQD > 0: token-level pipeline (the next RQD tokens' loads in flight while one is summed)
            rqp = [rq_load(ii) for ii in range(min(RQD, NTQ))]
            for ii in range_constexpr(NTQ if RQD else 0):
                if const_expr(ii + RQD < NTQ):
                    rqp.append(rq_load(ii + RQD))
                rq_store(ii, rqp[ii])
            for sbi in range_constexpr(0 if RQD else NSUBQ):
                lds_ = [
                    [
                        [
                            bld(rx, (((w0 + i32(sbi * ZQ + ii)) * i32(K1) + i32(t)) * i32(H) + tid * i32(8) + i32(kk * NTH * 8)) * i32(2), 0, T.vec(4, T.i32))
                            for t in range(K1)
                        ]
                        for kk in range(PIT)
                    ]
                    for ii in range(ZQ)
                ]
                for ii in range_constexpr(ZQ):
                    wrow = w0 + i32(sbi * ZQ + ii)
                    own = pd == rank
                    for kk in range_constexpr(PIT):
                        p = tid + i32(kk * NTH)
                        acc = [fx.Float32(0.0)] * 8
                        for t in range_constexpr(K1):
                            v = bf16x8_to_f32(lds_[ii][kk][t])
                            acc = [a_ + b_ for a_, b_ in zip(acc, v)]
                        yb = pack_bf16x8(acc)
                        if own:
                            bst(yb, r_y, ((o + wrow - rank * s) * i32(H) + p * i32(8)) * i32(2), 0, AUX_SYS)
                        else:
                            e_store(bf16x8_to_f32(yb), rd, B + wrow, p, lane)
            done_last(k, P_RQ, cnt(P_RQ, s))
            if done_at(sbc(k, 0, b), nb):
                if tid == i32(0):
                    tn = k * i32(NSB) + b
                    take_turn(QE, tn)
                    for p_ in range_constexpr(tp):
                        if i32(p_) != rank:
                            put(i32(p_), "erecv", (B + rank * s + b * sb) * i32(RBE), "eq", (B + i32(p_) * s + b * sb) * i32(RBE), sb * i32(RBE), QE)
                    give_turn(QE, tn)

        @flyc.jit
        def item_rf(k, j):
            tid = _opaque(tid0)
            lane = tid % i32(64)
            o, s = chunk(k)
            B = i32(tp) * o
            sb = s // i32(NSB)
            jx = il(j, cnt(P_RF, s))
            i0 = jx * i32(ZQ * NSUBQ)
            b = i0 // sb
            wait_puts_n(QE, k * i32(NSB) + b + i32(1))
            wait_at(sbc(k, 0, b), (i32(tp) * sb) if RQE else (i32(tp) * sb) // i32(ZQ * NSUBQ))
            rv = rsrc(wp("erecv"))
            rq, rqs = rsrc(wp("q")), rsrc(wp("qs"))
            one = fx.Float32(1.0)
            for sbi in range_constexpr(NSUBQ):
                # every load of the item first (one memory round trip), then the math
                ld = []
                for ii in range_constexpr(ZQ):
                    i = i0 + i32(sbi * ZQ + ii)
                    lo = ((o + i) * i32(H)) * i32(2)
                    row = []
                    for kk in range_constexpr(PIT):
                        off = lo + (tid + i32(kk * NTH)) * i32(16)
                        row.append(
                            (
                                bld(r_y, off, 0, T.vec(4, T.i32)),
                                [e_ld(rv, B + i32(src_) * s + i, tid + i32(kk * NTH)) for src_ in range(tp)],
                                bld(r_res, off, 0, T.vec(4, T.i32), AUX_SYS if ext else 0),
                            )
                        )
                    ld.append(row)
                fss, sss = [], []
                for ii in range_constexpr(ZQ):
                    i = i0 + i32(sbi * ZQ + ii)
                    lo = ((o + i) * i32(H)) * i32(2)
                    fs = []
                    ss = fx.Float32(0.0)
                    for kk in range_constexpr(PIT):
                        off = lo + (tid + i32(kk * NTH)) * i32(16)
                        ly, lr, lres = ld[ii][kk]
                        f = bf16x8_to_f32(ly)
                        for src_ in range_constexpr(tp):
                            f = sum_live(f, e_val(lr[src_]), i32(src_) != rank)
                        if const_expr(io):  # the MoE output y (this rank's rows)
                            f = [fx.Float32(x.to(fx.BFloat16).to(fx.Float32)) for x in f]
                            bst(pack_bf16x8(f), r_norm, off, 0)
                        f = [x + yy for x, yy in zip(f, bf16x8_to_f32(lres))]
                        f = [fx.Float32(x.to(fx.BFloat16).to(fx.Float32)) for x in f]
                        bst(pack_bf16x8(f), r_out, off, 0)
                        for x in f:
                            ss = ss + x * x
                        fs.append(f)
                    fss.append(fs)
                    sss.append(ss.bitcast(fx.Int32))
                tots = blk_red_vec(L, tid, sss, _fadd)
                xss, ams = [], []
                for ii in range_constexpr(ZQ):
                    rcp = fx.Float32(fmath.rsqrt(tots[ii].bitcast(fx.Float32) / fx.Float32(float(H)) + fx.Float32(float(eps))))
                    xs_, am = [], fx.Float32(0.0)
                    for kk in range_constexpr(PIT):
                        wv = norm_w(r_wnext, tid + i32(kk * NTH))
                        xk = [fx.Float32(fx.Float32(gate_w(x * rcp, wk)).to(fx.BFloat16).to(fx.Float32)) for x, wk in zip(fss[ii][kk], wv)]
                        am = am.maximumf(amax(xk))
                        xs_.append(xk)
                    xss.append(xs_)
                    ams.append(am.bitcast(fx.Int32))
                amxs = blk_red_vec(L, tid, ams, _fmax)
                for ii in range_constexpr(ZQ):
                    i = i0 + i32(sbi * ZQ + ii)
                    scl = amxs[ii].bitcast(fx.Float32).maximumf(fx.Float32(1e-10)) / fx.Float32(448.0)
                    inv = one / scl
                    g = rank * m + o + i
                    for kk in range_constexpr(PIT):
                        p = tid + i32(kk * NTH)
                        qv = [x * inv for x in xss[ii][kk]]
                        d = fx.Vector.from_elements([fp8x4_pack(qv[0:4], one), fp8x4_pack(qv[4:8], one)], fx.Int32)
                        bst(d, rq, g * i32(H) + p * i32(8), 0, AUX_SYS)
                    if tid == i32(0):
                        bst(scl.bitcast(fx.Int32), rqs, g * i32(4), 0, AUX_SYS)
            done_last(k, P_RF, cnt(P_RF, s))
            if done_at(sbc(k, 1, b), sb // i32(ZQ * NSUBQ)) & (i32(int(not io or tailq == 1)) == i32(1)):
                if tid == i32(0):
                    g0 = rank * m + o + b * sb
                    tn = k * i32(NSB) + b
                    take_turn(QQ, tn)
                    take_turn(QS, tn)
                    for p_ in range_constexpr(tp):
                        if i32(p_) != rank:
                            put(i32(p_), "q", g0 * i32(H), "q", g0 * i32(H), sb * i32(H), QQ)
                            put(i32(p_), "qs", g0 * i32(4), "qs", g0 * i32(4), sb * i32(4), QS)
                    give_turn(QQ, tn)
                    give_turn(QS, tn)

        @flyc.jit
        def item_rf_y(k, j):
            # io without tail: y = own partial + the peers' MoE-output rows (this rank's rows)
            tid = _opaque(tid0)
            o, s = chunk(k)
            B = i32(tp) * o
            sb = s // i32(NSB)
            jx = il(j, cnt(P_RF, s))
            i0 = jx * i32(ZQ * NSUBQ)
            b = i0 // sb
            wait_puts_n(QE, k * i32(NSB) + b + i32(1))
            wait_at(sbc(k, 0, b), (i32(tp) * sb) // i32(ZQ * NSUBQ))
            rv = rsrc(wp("erecv"))
            for ii in range_constexpr(ZQ * NSUBQ):
                i = i0 + i32(ii)
                lo = ((o + i) * i32(H)) * i32(2)
                for kk in range_constexpr(PIT):
                    off = lo + (tid + i32(kk * NTH)) * i32(16)
                    f = bf16x8_to_f32(bld(r_y, off, 0, T.vec(4, T.i32)))
                    for src_ in range_constexpr(tp):
                        f = sum_live(f, e_val(e_ld(rv, B + i32(src_) * s + i, tid + i32(kk * NTH))), i32(src_) != rank)
                    bst(pack_bf16x8(f), r_norm, off, 0)
            done_last(k, P_RF, cnt(P_RF, s))

        @flyc.jit
        def item_gw():
            if const_expr(ext == 2):
                for k in range_constexpr(C):
                    wait_ph(i32(k), P_RT, cnt(P_RT, chunk(i32(k))[1], k))
                wait_puts(QID, 2 * NSBA, i32(C - 1))  # every peer's routing landed (the sorts read it)
            else:
                for k in range_constexpr(C):
                    wait_ph(i32(k), P_RF, cnt(P_RF, chunk(i32(k))[1]))
                if const_expr(tailq == 1 or not io):
                    wait_puts(QQ, NSB, i32(C - 1))
                    wait_puts(QS, NSB, i32(C - 1))
            gpu.barrier()
            # every put of this launch landed: advance the persistent per-(peer, queue) base
            if lane < i32(tp):
                if lane != rank:
                    # every queue whose puts this launch waited for (pre: A, routing; MoE: rows, E, q)
                    OWNQ = (QA, QID) if ext == 2 else (QX, QXS, QE, QQ, QS) if ext else QUEUES
                    for qi in range_constexpr(len(OWNQ)):
                        q_ = OWNQ[qi]
                        addr = consumed + fx.Int64((lane * i32(NQ) + i32(q_)) * i32(4))
                        g_st_sys(addr, sig_base(lane, q_) + i32(C * PPC[q_]))

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

        def pick():
            # ROLE: (queue 1 P / 2 U / 3 pk_none left, ticket), uniform
            if tid0 == i32(0):
                pk_sk = [chunk_v(i32(k)) for k in range(C)]
                pk_slp, pk_np_ = slices_of(pk_sk, ORDER_P)
                pk_slu, pk_nu_ = slices_of(pk_sk, ORDER_U)
                pk_np_ = pk_np_ + (xc == i32(0)).select(i32(1), i32(0))  # + GW
                pk_tkp = ca(i32(C_TK) + xc * i32(32))
                pk_tku = ca(i32(C_TKU) + xc * i32(32))
                pk_st = i32(0)
                pk_tg = i32(0)
                if bid // i32(8) < i32(NP):
                    pk_v = fx.Int32(atomic_add_agent(pk_tkp, i32(1)))
                    pk_st = (pk_v < pk_np_).select(i32(1), i32(3))
                    pk_tg = pk_v
                else:
                    while pk_st == i32(0):
                        pk_tu = fx.Int32(load_i32_global_agent(pk_tku))
                        pk_tq = fx.Int32(load_i32_global_agent(pk_tkp))
                        _, pk_ku, _ = decode(pk_slu, pk_tu)
                        pk_kq_kind, pk_kq, _ = decode(pk_slp, pk_tq)
                        pk_u_left = pk_tu < pk_nu_
                        pk_u_none = pk_tu >= pk_nu_
                        pk_p_left = pk_tq < pk_np_
                        pk_ku = fx.min(pk_ku, i32(C - 1))
                        pk_sku = i32(0)
                        for k in range_constexpr(C):
                            pk_sku = (pk_ku == i32(k)).select(cnt(P_S3, pk_sk[k]), pk_sku)
                        pk_ready_u = pk_u_left & (fx.Int32(load_i32_global_agent(ph(pk_ku, P_S3))) >= pk_sku)
                        # a unit CTA takes: any attention quant (no waits), the prep of chunks up to
                        # the next unit's, the topk sum of chunks whose units are all taken
                        pk_prep_ok = (
                            (pk_kq_kind == i32(P_QA))
                            | ((pk_kq_kind >= i32(0)) & (pk_kq_kind <= i32(P_S3)) & (pk_kq <= pk_ku))
                            | ((pk_kq_kind == i32(P_RQ)) & (pk_kq < pk_ku) & (i32(RQU) != i32(0)))
                        )
                        pk_p_ok = pk_p_left & (pk_u_none | pk_prep_ok)
                        pk_cu = pk_ready_u.select(i32(1), i32(0))
                        pk_cp = pk_ready_u.select(i32(0), pk_p_ok.select(i32(1), i32(0)))
                        # claim exactly the ticket checked (compare-and-swap; a lost race re-checks)
                        pk_vu = _g_cas(pk_tku, pk_tu, pk_tu + pk_cu)
                        pk_vp = _g_cas(pk_tkp, pk_tq, pk_tq + pk_cp)
                        pk_got_u = (pk_cu == i32(1)) & (pk_vu == pk_tu)
                        pk_got_p = (pk_cp == i32(1)) & (pk_vp == pk_tq)
                        pk_none = pk_u_none & (pk_tq >= pk_np_)
                        pk_st = pk_got_u.select(i32(2), pk_got_p.select(i32(1), pk_none.select(i32(3), i32(0))))
                        pk_tg = pk_got_u.select(pk_vu, pk_vp)
                        if pk_st == i32(0):
                            rocdl.s_sleep(2)
                lds_st(L, i32(LT), pk_st)
                lds_st(L, i32(LT) + i32(8), pk_tg)
            gpu.barrier()
            st_ = lds_ld_i32(L, i32(LT))
            tg_ = lds_ld_i32(L, i32(LT) + i32(8))
            gpu.barrier()
            return st_, tg_

        rb_item = item_rx if io else item_rb
        rf_item = item_rf_y if (io and not tailq) else item_rf

        def next_role():
            a_, b_ = pick()
            return a_, b_, a_ != i32(3)

        def next_plain():
            b_ = claim()
            return i32(1), b_, b_ < nall

        def decode_role(q, t_):
            sk_ = [chunk_v(i32(k)) for k in range(C)]
            a0, a1, a2 = decode(slices_of(sk_, ORDER_P)[0], t_)
            b0_, b1_, b2_ = decode(slices_of(sk_, ORDER_U)[0], t_)
            isu = q == i32(2)
            return isu.select(b0_, a0), isu.select(b1_, a1), isu.select(b2_, a2)

        def decode_plain(q, t_):
            return decode(slices_of([chunk_v(i32(k)) for k in range(C)], ORDER_P)[0], t_)

        next_item = next_role if NP else next_plain
        decode_item = decode_role if NP else decode_plain
        q_, t, alive_ = next_item()
        while alive_:
            kind_r, k_r, jj = decode_item(q_, t)
            kind_r = uni(kind_r)
            k_r = uni(k_r)
            jj = uni(jj)
            t0 = fx.Int32(now()) if const_expr(TLD) else i32(0)
            if kind_r == i32(P_QA):
                item_qa(k_r, jj)
            elif kind_r == i32(P_RB):
                rb_item(k_r, jj)
            elif kind_r == i32(P_RT):
                item_rt(k_r, jj)
            elif kind_r == i32(P_S1):
                item_s1(k_r, jj)
            elif kind_r == i32(P_S2):
                item_s2(k_r)
            elif kind_r == i32(P_S3):
                item_s3(k_r, jj)
            elif (kind_r == i32(P_U)) | (kind_r == i32(P_UB)):
                item_u(k_r, jj)
            elif (kind_r == i32(P_U2)) & (i32(GSPLIT) != i32(0)):
                if const_expr(GSPLIT):
                    item_u2(k_r, jj)
            elif kind_r == i32(P_RQ):
                item_rq(k_r, jj)
            elif kind_r == i32(P_RF):
                rf_item(k_r, jj)
            if const_expr(TLD):
                if (tid == i32(0)) & (kind_r >= i32(0)):
                    slot = i32(C_TL) + (k_r * i32(NKIND) + kind_r) * i32(2)
                    _g_umax(ca(slot), i32(-1) - t0)
                    t1 = fx.Int32(now())
                    _g_umax(ca(slot + i32(1)), t1)
                    slot2 = i32(C_TLS) + (k_r * i32(NKIND) + kind_r) * i32(2)
                    atomic_add_agent(ca(slot2), t1 - t0)
                    atomic_add_agent(ca(slot2 + i32(1)), i32(1))
            if (t == gw_lo) & (q_ == i32(1)):
                item_gw()
            gpu.barrier()
            q_, t, alive_ = next_item()

    @flyc.jit
    def launch(
        wb: fx.Int64, win: fx.Int64, dc: fx.Int64, consumed: fx.Int64, offs: fx.Int64, part: fx.Int64,
        resid: fx.Int64, wpost: fx.Int64, wg: fx.Int64, bias: fx.Int64, wnext: fx.Int64, res: fx.Int64,
        normed: fx.Int64, resout: fx.Int64, ybuf: fx.Int64, routes: fx.Int64, lrix: fx.Int64,
        lwt: fx.Int64, utab: fx.Int64, w1: fx.Int64, w1s: fx.Int64, w2: fx.Int64, w2s: fx.Int64,
        ctrl: fx.Int64, chunks: fx.Int64, logit: fx.Int64, stab: fx.Int64, inter: fx.Int64, rank: fx.Int32, m: fx.Int32, ucap: fx.Int32,
        rstride: fx.Int64, lstride: fx.Int32, grid: fx.Int32, stream: fx.Stream,
    ):
        kernel(
            wb, win, dc, consumed, offs, part, resid, wpost, wg, bias, wnext, res, normed, resout,
            ybuf, routes, lrix, lwt, utab, w1, w1s, w2, w2s, ctrl, chunks, logit, stab, inter, rank, m, ucap, rstride,
            lstride,
        ).launch(grid=(fx.Int64(grid), 1, 1), block=(NTH, 1, 1), stream=stream)

    if CPU > 1:
        launch.compile_hints = {"waves_per_eu": CPU}  # CPU CTAs (of NW waves) per CU: <= 512 / CPU registers
    return launch


# per-batch tuned big-M configs (op_tests/op_benchmarks/flydsl/tune_mega_moe_tp_bigm.py):
# chunks + AITER_BIGM_* options; AITER_MEGAMOE_TP_BIGM_TUNED: csv path, "0" disables
_BIGM_KEY = ("gfx", "model_dim", "inter_dim", "experts", "topk", "tp")
_BIGM_TUNED_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "configs", "mega_moe_tp_bigm_tuned.csv"
)


def bigm_tuned(*, model_dim, inter_dim, experts, topk, tp, tokens):
    """The csv row of the nearest tuned batch (<= tokens, else the smallest):
    {"chunks": [..], "opts": {"AITER_BIGM_*": ..}, "tokens": n} or None."""
    import csv

    from aiter.jit.utils.chip_info import get_gfx

    path = os.environ.get("AITER_MEGAMOE_TP_BIGM_TUNED", _BIGM_TUNED_DEFAULT)
    if path == "0" or not os.path.isfile(path):
        return None
    key = tuple(str(x) for x in (get_gfx(), model_dim, inter_dim, experts, topk, tp))
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if tuple(r[k] for k in _BIGM_KEY) == key:
                rows[int(r["tokens"])] = r
    if not rows:
        return None
    t = max((x for x in rows if x <= tokens), default=min(rows))
    r = rows[t]
    opts = dict(kv.split("=", 1) for kv in r.get("opts", "").split(";") if kv)
    return {"chunks": [int(x) for x in r["chunks"].split(",")], "opts": opts, "tokens": t}


CMAX = 7  # chunks per launch (ctrl layout)


class MegaMoeTPBigM:
    """Host side of the big-M dispatch: the SDMA window (same regions on every rank),
    scratch buffers and the launch.
    io=False: __call__(part [T, H], residual [T, H], w_post, w_next, wg [E, H], bias [E])
    -> (q [T, H] fp8, q scales [T] f32, res_out [m, H] bf16) (the whole layer).
    io=True: moe(x [m, H], topk_weights / topk_ids [m, K1], out, tail) (MegaMoeTP.forward,
    ag_rs) -> y [m, H], with tail = (res_in, res_out, norm_w) (y, q [T, H] fp8, scales [T]).
    chunks=None: per batch from the tuned csv (bigm_tuned: chunks + AITER_BIGM_* options)."""

    REGIONS = ("aq", "arecv", "xq", "xs", "ids", "tw", "eq", "erecv", "q", "qs")

    def __init__(
        self, *, hidden, inter, experts, topk, max_tokens, chunks, eps, gemma, scale, shared_w,
        swiglu_limit, w1, w1_scale, w2, w2_scale, group=None, device=None, cu=None, sep=None, opts=None,
        io=False,
    ):
        import torch
        import torch.distributed as dist

        from .sdma import SdmaComm

        self.H, self.I, self.E, self.K = hidden, inter, experts, topk
        self.K1, self.NE = topk + 1, experts + 1
        self.tp = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.Tmax = max_tokens
        self.io = bool(io)
        if chunks is None:
            self.weights = None  # per batch (_plan)
        else:
            self.weights = [1] * chunks if isinstance(chunks, int) else [int(x) for x in chunks]
        self.C = CMAX if self.weights is None else len(self.weights)
        assert self.C <= CMAX
        self._plans = {}
        self.cfg = (float(eps), bool(gemma), float(scale), float(shared_w), float(swiglu_limit))
        self.opts = tuple(sorted((opts or {}).items()))  # tuned AITER_BIGM_* options
        od = dict(self.opts)
        # unit rows and CTAs per CU (RG=64 units fit two CTAs in a CU: units beside row phases)
        self.rg = int(os.environ.get("AITER_BIGM_RG", od.get("AITER_BIGM_RG", "128")))
        self.cpu = int(os.environ.get("AITER_BIGM_CPU", od.get("AITER_BIGM_CPU", "1")))
        self.w = (w1.view(torch.uint8), w1_scale.view(torch.uint8), w2.view(torch.uint8), w2_scale.view(torch.uint8))
        H, T, K1, tp = hidden, max_tokens, self.K1, self.tp
        RB = i8_row_bytes(H)
        ra = 16 if self.io else T * RB  # io: no attention RS
        sizes = {
            "aq": ra, "arecv": ra, "xq": T * H // 2, "xs": T * H // 32, "ids": T * K1 * 4,
            "tw": T * K1 * 4, "eq": T * RB, "erecv": T * RB, "q": T * H, "qs": T * 4,
        }
        sd = self.sd = SdmaComm(group=group, num_queues=NQ, device=device)
        for r_ in self.REGIONS:
            sd.reserve(r_, sizes[r_])
        sd.commit()
        dev = self.device = sd.device
        self.offs = torch.tensor([sd.offset(r_) for r_ in self.REGIONS], dtype=torch.int64, device=dev)
        m = T // tp
        bf = torch.bfloat16
        self.res = torch.empty((m, H), dtype=bf, device=dev)
        self.normed = torch.empty((m, H), dtype=bf, device=dev)
        self.resout = torch.empty((m, H), dtype=bf, device=dev)
        self.ybuf = torch.empty((m, H), dtype=bf, device=dev)
        smax = m if self.weights is None else max(self._split(m))
        # route rows / lists of every chunk packed (chunk k from route tp * o_k * K1)
        self.lstride = tp * smax * K1
        self.rstride = self.lstride * H * 2
        self.routes = torch.empty(T * K1 * H, dtype=bf, device=dev)
        self.lrix = torch.empty(T * K1, dtype=torch.int32, device=dev)
        self.lwt = torch.empty(T * K1, dtype=torch.float32, device=dev)
        self.ucap = unit_cap(tp, smax, K1, self.NE, self.rg)
        # unit slots of every chunk ([C][8][ucap of the batch]): sum over chunks <= the bound
        self.uslots = 2 * (-(-tp * m * K1 // self.rg) // 8) + self.C * (-(-self.NE // 8) + UQ + 2)
        self.utab = torch.zeros(8 * self.uslots * 4, dtype=torch.int32, device=dev)
        self.ctrl = torch.zeros(CTRL_INTS, dtype=torch.int32, device=dev)
        self.logit = torch.empty((m, experts), dtype=torch.float32, device=dev)
        # GSPLIT unit intermediate blocks: [C][8][ucap] x passes x RG rows x (64 B codes + 4 B scales)
        ub = (inter // 128) * self.rg * 68
        self.inter = torch.empty(
            (8 * self.uslots * ub if int(os.environ.get("AITER_BIGM_GSPLIT", od.get("AITER_BIGM_GSPLIT", "0"))) else 64,),
            dtype=torch.uint8, device=dev,
        )
        # sep: the pre kernel (RS + norm + router + gathers), moe_sorting of each chunk outside
        # the kernels, then the MoE kernel (units of the sorted blocks + the tail)
        self.sep = os.environ.get("BIGM_SEP", "0" if sep is None else str(int(sep))) == "1"
        if self.sep:
            npad = tp * smax * K1 + self.NE * 128 - K1
            nblk = -(-npad // 128)
            npad = nblk * 128
            self.s_ids = torch.empty((self.C, npad), dtype=torch.int32, device=dev)
            self.s_w = torch.empty((self.C, npad), dtype=torch.float32, device=dev)
            self.s_e = torch.empty((self.C, nblk), dtype=torch.int32, device=dev)
            self.s_n = torch.zeros((self.C, 2), dtype=torch.int32, device=dev)
            self.s_buf = torch.empty((0, 0), dtype=bf, device=dev)
            self.tokc = torch.zeros((self.C, tp * smax), dtype=torch.int32, device=dev)
            self.stab = torch.tensor(
                [[self.s_ids[k].data_ptr(), self.s_w[k].data_ptr(), self.s_e[k].data_ptr(), self.s_n[k].data_ptr(),
                  self.tokc[k].data_ptr()] for k in range(self.C)],
                dtype=torch.int64, device=dev,
            )
            self.ucap_moe = -(-nblk // 8)
            self.ctrl2 = torch.zeros(CTRL_INTS, dtype=torch.int32, device=dev)
            self._ws = {}
        self.chunks_t = {}
        self._ready = set()
        self.cu = cu or torch.cuda.get_device_properties(dev).multi_processor_count
        # deepest A ring (dividing the GEMM1 chunk count) whose LDS fits a CU
        from .bigm_gemm import bigm_lds

        kch = hidden // 256
        self.nab = next(
            n for n in (4, 3, 2)
            if kch % n == 0 and self._unit_lds(n, inter) + 4096 <= 160 * 1024 // self.cpu
        )

    def _unit_lds(self, n, inter):
        from .bigm_gemm import bigm_lds

        if int(os.environ.get("AITER_BIGM_GSPLIT", dict(self.opts).get("AITER_BIGM_GSPLIT", "0"))):
            return max(bigm_lds(self.rg, n, inter, MODE=3)["BYTES"], bigm_lds(self.rg, n, inter, MODE=4)["BYTES"])
        return bigm_lds(self.rg, n, inter)["BYTES"]

    def _plan(self, T):
        """(chunk weights, AITER_BIGM_* options) of a batch of T tokens"""
        p = self._plans.get(T)
        if p is None:
            w, o = self.weights, dict(self.opts)
            if w is None:
                tu = bigm_tuned(
                    model_dim=self.H, inter_dim=self.I, experts=self.NE, topk=self.K1, tp=self.tp, tokens=T
                )
                ch = os.environ.get("BIGM_CHUNKS")
                if ch:
                    w = [int(x) for x in ch.split(",")]
                elif tu is not None:
                    w = tu["chunks"]
                else:
                    w = [1, 1] if T <= 16384 else [1, 2, 2, 1]
                if tu is not None:
                    o = {**tu["opts"], **o}
            assert len(w) <= self.C
            p = self._plans[T] = (tuple(w), tuple(sorted(o.items())))
        return p

    def _split(self, m, w=None):
        w = w or self.weights or self._plan(m * self.tp)[0]
        tot = sum(w)
        # multiples of 64 rows: whole router groups in every head sub-block (AITER_BIGM_NSBA <= 4)
        g = 64 if m % 64 == 0 else 16
        sizes = [max(g, (m * x // tot) // g * g) for x in w]
        sizes[w.index(max(w))] += m - sum(sizes)
        assert all(x > 0 and x % g == 0 for x in sizes) and sum(sizes) == m, (m, sizes)
        return sizes

    def _chunks(self, T):
        import torch

        c = self.chunks_t.get(T)
        if c is None:
            m = T // self.tp
            sizes = self._split(m)
            assert self.uslots >= len(sizes) * self._ucap(T), "unit table"
            os_ = [sum(sizes[:i]) for i in range(len(sizes))]
            c = self.chunks_t[T] = torch.tensor([[o, s] for o, s in zip(os_, sizes)], dtype=torch.int32, device=self.device)
        return c

    def set_weights(self, w1, w1_scale, w2, w2_scale):
        """the next launches use another layer's weights of the same shape"""
        import torch

        self.w = (w1.view(torch.uint8), w1_scale.view(torch.uint8), w2.view(torch.uint8), w2_scale.view(torch.uint8))

    def _ucap(self, T):
        return unit_cap(self.tp, max(self._split(T // self.tp)), self.K1, self.NE, self.rg)

    def own_ids(self, T):
        import torch

        m = T // self.tp
        r = self.rank
        ids = self.sd.view("ids", (T, self.K1), torch.int32)
        out, o = [], 0
        for s in self._split(m):
            B = self.tp * o
            out.append(ids[B + r * s : B + r * s + s])
            o += s
        return torch.cat(out)

    def _sort(self, k, ids, tw):
        import aiter
        import torch

        M = int(ids.shape[0])
        ws = self._ws.get(M, False)
        if ws is False:
            n = aiter.moe_sorting_opus_get_workspace_size(M, self.NE, self.K1, 0)
            ws = self._ws[M] = torch.empty(n, dtype=torch.uint8, device=self.device) if n > 0 else None
        aiter.moe_sorting_opus_fwd(
            ids, tw, self.s_ids[k], self.s_w[k], self.s_e[k], self.s_n[k], self.s_buf, self.NE, 128,
            None, None, ws, 0, None, None, None,
        )

    def _run(self, launch, args):
        import torch

        from ..tensor_shim import _run_compiled

        if launch not in self._ready:
            # compile on every rank first: the kernel's waits on peers have a 2 s watchdog,
            # a launch must not start while a peer is still compiling
            import torch.distributed as dist

            from ..tensor_shim import _preload_compiled

            _preload_compiled(launch, *args)
            torch.cuda.synchronize()
            dist.barrier(group=self.sd.group)
            self._ready.add(launch)
        _run_compiled(launch, *args)

    def __call__(self, part, residual, w_post, w_next, wg, bias):
        import torch

        T = part.shape[0]
        m = T // self.tp
        assert T <= self.Tmax and T % self.tp == 0
        eps, gemma, scale, shared_w, lim = self.cfg
        w1, w1s, w2, w2s = self.w
        stream = torch.cuda.current_stream()

        def args(ctrl, stab, ucap):
            return (
                self.sd.base, self.sd.handle, self.sd.dev_comm.ptr, self.sd.consumed.data_ptr(),
                self.offs.data_ptr(), part.data_ptr(), residual.data_ptr(), w_post.data_ptr(),
                wg.data_ptr(), bias.data_ptr(), w_next.data_ptr(), self.res.data_ptr(),
                self.normed.data_ptr(), self.resout.data_ptr(), self.ybuf.data_ptr(),
                self.routes.data_ptr(), self.lrix.data_ptr(), self.lwt.data_ptr(), self.utab.data_ptr(),
                w1.data_ptr(), w1s.data_ptr(), w2.data_ptr(), w2s.data_ptr(), ctrl.data_ptr(),
                self._chunks(T).data_ptr(), self.logit.data_ptr(), stab, self.inter.data_ptr(), self.rank, m, ucap, self.rstride,
                self.lstride, self.cu * self.cpu, stream,
            )

        w_, opts = self._plan(T)
        cl = (self.H, self.I, self.E, self.K, self.tp, len(w_), eps, gemma, scale, shared_w, lim)
        if not self.sep:
            # chunks of >= 8192 tokens take two or more unit rounds: chunk k+1's reduce / router
            # go between chunk k's first round and the rest (ORD 4); else after its units (ORD 3)
            order = 4 if self.tp * max(self._split(m)) >= 8192 else 3
            self.ctrl.zero_()
            self._run(compile_bigm_layer(*cl, NAB=self.nab, order=order, opts=opts, RG=self.rg, CPU=self.cpu), args(self.ctrl, 0, self._ucap(T)))
        else:
            self.ctrl.zero_()
            self.ctrl2.zero_()
            self.tokc.zero_()
            self._run(compile_bigm_layer(*cl, NAB=self.nab, ext=2, opts=self.opts, RG=self.rg, CPU=self.cpu), args(self.ctrl, 0, self.ucap))
            ids = self.sd.view("ids", (T, self.K1), torch.int32)
            tw = self.sd.view("tw", (T, self.K1), torch.float32)
            o = 0
            for k, s_ in enumerate(self._split(m)):
                B = self.tp * o
                self._sort(k, ids[B : B + self.tp * s_], tw[B : B + self.tp * s_])
                o += s_
            self._run(
                compile_bigm_layer(*cl, NAB=self.nab, ext=1, opts=self.opts, RG=self.rg, CPU=self.cpu), args(self.ctrl2, self.stab.data_ptr(), self.ucap_moe)
            )
        q = self.sd.view("q", (T, self.H), torch.uint8)
        qs = self.sd.view("qs", (T,), torch.float32)
        return q, qs, self.resout[:m]

    def moe(self, x, topk_weights, topk_ids, out=None, tail=None, local=False):
        """io: the MoE of every rank's tokens (ag_rs): x [m, H] bf16 (this rank's rows, global
        rows rank * m + i), topk_weights f32 / topk_ids int32 [m, K1] (the shared expert E in
        any column) -> y [m, H] bf16 = this rank's rows of sum_r MoE_r(every row); tail =
        (res_in, res_out, norm_w): also res_out = y + res_in and the (Gemma)RMSNorm of res_out
        as FP8 rows (+ f32 scales) of every rank -> (y, q [T, H] fp8 (uint8), scales [T]);
        local: only this rank's q rows (rank * m ..) are written."""
        import torch

        assert self.io
        m = x.shape[0]
        T = m * self.tp
        assert T <= self.Tmax and x.is_contiguous() and topk_ids.shape == (m, self.K1)
        assert topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
        eps, gemma, scale, shared_w, lim = self.cfg
        w1, w1s, w2, w2s = self.w
        y = self.normed[:m] if out is None else out
        if tail is not None:
            res_in, res_out, norm_w = tail
        else:
            res_in, res_out, norm_w = x, self.resout, x
        w_, opts = self._plan(T)
        order = 4 if self.tp * max(self._split(m)) >= 8192 else 3
        launch = compile_bigm_layer(
            self.H, self.I, self.E, self.K, self.tp, len(w_), eps, gemma, scale, shared_w, lim,
            NAB=self.nab, order=order, opts=opts, RG=self.rg, CPU=self.cpu, io=1,
            tailq=0 if tail is None else 2 if local else 1,
        )
        self.ctrl.zero_()
        self._run(
            launch,
            (
                self.sd.base, self.sd.handle, self.sd.dev_comm.ptr, self.sd.consumed.data_ptr(),
                self.offs.data_ptr(), x.data_ptr(), topk_ids.data_ptr(), topk_weights.data_ptr(),
                x.data_ptr(), x.data_ptr(), norm_w.data_ptr(), res_in.data_ptr(),
                y.data_ptr(), res_out.data_ptr(), self.ybuf.data_ptr(),
                self.routes.data_ptr(), self.lrix.data_ptr(), self.lwt.data_ptr(), self.utab.data_ptr(),
                w1.data_ptr(), w1s.data_ptr(), w2.data_ptr(), w2s.data_ptr(), self.ctrl.data_ptr(),
                self._chunks(T).data_ptr(), self.logit.data_ptr(), 0, self.inter.data_ptr(), self.rank, m,
                self._ucap(T), self.rstride, self.lstride, self.cu * self.cpu, torch.cuda.current_stream(),
            ),
        )
        if tail is None:
            return y
        q = self.sd.view("q", (T, self.H), torch.uint8)
        qs = self.sd.view("qs", (T,), torch.float32)
        return y, q, qs
