# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations

import functools
import os  # [tl]
import struct
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from .. import buffer_ops
from ..mxfp4_gemm_common import _activation_mul_batch, _e8m0_from_amax, _fabs_f32

__all__ = [
    "CTRL_ERR",
    "CTRL_INTS",
    "DYN_MAX",
    "FLAG_INTS",
    "MAX_TP",
    "NCTA_MAX",
    "UNIT_G1X",
    "UNIT_G2COL",
    "UNIT_REC",
    "UNIT_SIGNAL",
    "XQ_MAX",
    "XQ_P",
    "XQ_SHIFT",
    "compile_mega_moe_tp",
    "gemm2_chunk_groups",
    "gemm2_group_step",
    "mega_moe_tp_consts",
    "mega_moe_tp_shape_supported",
]

MAX_TP = 8
# tokens per launch (all ranks) up to which the host may pick the dynamic schedule
DYN_MAX = int(os.environ.get("AITER_MEGAMOE_TP_DYN_MAX", "256"))  # [exp]
NCK_MAX = 64
NCHA_MAX = 64
NCTA_MAX = 256
# Flag arena (ints, written by peers, one writer each, values = launch epoch):
#   FLAG_RDY + c*MAX_TP + r              rank r pushed RS chunk c
#   FLAG_AGM + r*32 + b                  rank r's routing, CTA b's part
#   FLAG_PRE + (g*MAX_TP + r)*NCTA + b   AR: rank r sent input group g of CTA b's rows
#   FLAG_YAG + (c*MAX_TP + r)*NCTA + b   AR: rank r sent output chunk c of CTA b's rows
#   FLAG_AGQ + q*MAX_TP + r              every CTA of rank r landed K-chunk q
#                                        (per rank, not per row: per-row polls by
#                                        every CTA flood the lines peers write)
FLAG_RDY = 0
FLAG_AGM = FLAG_RDY + MAX_TP * NCK_MAX
FLAG_PRE = FLAG_AGM + MAX_TP * 32
PRE_CH = 4
NPRE_MAX = 16
FLAG_YAG = FLAG_PRE + NPRE_MAX * MAX_TP * NCTA_MAX
FLAG_AGQ = FLAG_YAG + NCK_MAX * MAX_TP * NCTA_MAX
#   FLAG_TN + r*TN_MAX + i                     tail (tn): rank r sent its output row i
TN_MAX = 2048
FLAG_TN = FLAG_AGQ + NCHA_MAX * MAX_TP
FLAG_INTS = FLAG_TN + MAX_TP * TN_MAX

# Local control ints. Polled/atomic counters sit on their own 128 B lines:
# same-line device atomics from every CTA serialize and stall weight streams.
CTRL_ERR = 20
CTRL_XF = 32
CTRL_CNT = 64
ERR_FLAG, ERR_META, ERR_CHUNK, ERR_COMM, ERR_YAG = 1, 2, 4, 8, 16
DEADLINE = 200_000_000
N_XCD = 8
LRDY_STRIDE = 32
CTRL_LRDY = CTRL_CNT + 2 * NCK_MAX
CTRL_XQ = CTRL_LRDY + NCK_MAX * LRDY_STRIDE
XQ_P = 8
XQ_MAX = 264
CTRL_SC = CTRL_XQ + XQ_MAX * XQ_P
SC_ALL, SC_PUSH, SC_SHARD = N_XCD, N_XCD + 1, N_XCD + 2
SC_E = SC_SHARD + N_XCD
SC_LATE = SC_E + 1
SC_LINES = SC_LATE + 1
CTRL_EPB = CTRL_SC + NCK_MAX * SC_LINES * LRDY_STRIDE
CTRL_XES = CTRL_EPB + NCTA_MAX
CTRL_AGX = CTRL_XES + N_XCD * LRDY_STRIDE
CTRL_AGG = CTRL_AGX + NCHA_MAX * N_XCD * LRDY_STRIDE
CTRL_CLM = CTRL_AGG + NCHA_MAX * LRDY_STRIDE
# LB schedule: launches counted by CTRL_LBSEQ (bumped by CTA 0 once every CTA
# read it); its parity picks the bank of the counters below, and every LB
# launch zeroes the other bank for the next one.
#   CTRL_LBR + (bank*N_XCD + x)*LRDY_STRIDE      CTAs of XCD x whose route list
#                                                slice landed (one line per XCD:
#                                                256 atomics on one serialize)
#   CTRL_COLC + (bank*NCK_MAX + c)*LRDY_STRIDE   GEMM2 units done of column group c
#   CTRL_G1C + (bank*NCHLB_MAX + j)*G1C_STRIDE   GEMM1 pieces done of row chunk j
NCHLB_MAX = 2048
G1C_STRIDE = 16
CTRL_LBSEQ = CTRL_CLM + 2 * LRDY_STRIDE
CTRL_LBR = CTRL_LBSEQ + LRDY_STRIDE
CTRL_COLC = CTRL_LBR + 2 * N_XCD * LRDY_STRIDE
CTRL_G1C = CTRL_COLC + 2 * NCK_MAX * LRDY_STRIDE
#   CTRL_CLMX + (bank*N_XCD + x)*LRDY_STRIDE     claim counter of XCD x's queue
CTRL_CLMX = CTRL_G1C + 2 * NCHLB_MAX * G1C_STRIDE
#   CTRL_TNC + i                                 tail (tn): CTAs done with row i
CTRL_TNC = CTRL_CLMX + 2 * N_XCD * LRDY_STRIDE
CTRL_INTS = CTRL_TNC + TN_MAX
UNIT_REC = 8
# units[u] = (expert, i0, icnt, kind); kind = type | groups << 8 | SIGNAL | slot << XQ_SHIFT
#   UNIT_G1X: GEMM1 inter slice exporting its intermediate
#   UNIT_G2COL: GEMM2 column slice (i0 = first group) importing it
#   SIGNAL: the unit after which the CTA signals every chunk
UNIT_FULL, UNIT_G2COL, UNIT_G1X = 0, 2, 3
UNIT_SIGNAL = 1 << 16
XQ_SHIFT = 20

NW = 4
NT = NW * 64
NTT = NT + 4 * 64
NSK = 4
SLOT = 4 * 1024 + 2 * 256
OPS = 6
KCS = 2
ALOAD_DEPTH = 3
RED_INFLIGHT = 32
POLL_SLEEP = 16
AUX_SC1 = 16
AUX_SYS = 1 | 16


def mega_moe_tp_shape_supported(model_dim: int, inter_dim: int, tp: int) -> bool:
    return (
        model_dim % 512 == 0
        and (model_dim // 64) % NW == 0
        and model_dim // 256 <= NCHA_MAX
        and model_dim // 256 // gemm2_chunk_groups(model_dim, inter_dim) <= 31
        and inter_dim % 128 == 0
        and inter_dim > 0
        and 1 <= tp <= MAX_TP
    )


def _nsk2_for(nks: int, g2: int) -> int:
    if g2 % (NSK * nks // math.gcd(NSK, nks) // nks) == 0:
        return NSK
    return 3 if nks % 3 == 0 else 2


def gemm2_group_step(H: int, I: int) -> int:
    ks2, g2 = I // 128, H // 64 // NW
    nsk2 = _nsk2_for(ks2, g2)
    return nsk2 * ks2 // math.gcd(nsk2, ks2) // ks2


def gemm2_chunk_groups(H: int, I: int) -> int:
    return 2 if (H // 64 // NW) % 2 == 0 else 1


def _attr(v):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), int(v))


def _u(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


traced = ASTRewriter.transform


@functools.cache
def mega_moe_tp_consts(
    H: int,
    I: int,
    MT: int,
    TMAX: int,
    agr: int = 1,
    dyn_e: int = 0,
    nab: int = 4,
    nsk: int = NSK,
    nch: int = 0,
    a8: bool = False,
    lb: bool = False,
) -> dict:
    RG = MT * 16
    KS1 = H // 128
    KS2 = I // 128
    G2 = H // 64 // NW
    NSC_BLK = (RG + 63) // 64
    # activation bytes per element: MXFP4 (2 per byte) or MXFP8 (a8)
    acb = KCS * (128 if a8 else 64)
    NA_ROWOPS = RG * acb // 1024
    GPC = gemm2_chunk_groups(H, I)
    c = {
        "RG": RG,
        "KS1": KS1,
        "NCH": KS1 // KCS,
        "KS2": KS2,
        "G2": G2,
        "CH1": (H // 32 + 7) // 8,
        "CH2": (I // 32 + 7) // 8,
        "ACB": acb,
        "XB": H if a8 else H // 2,
        # (LB: 8 B pad, so a second intermediate fits the A ring, L_B1)
        "SI_STRIDE": (I if a8 else I // 2) + (8 if lb and not a8 else 16),
        "NA_ROWOPS": NA_ROWOPS,
        "NSC_BLK": NSC_BLK,
        "NA_L": NA_ROWOPS + NSC_BLK * KCS,
        "GPC": GPC,
        "NCK": G2 // GPC,
        "CW": GPC * NW * 64,
        "NAB": nab,
        "NBW": (dyn_e + 31) // 32,
    }
    off = 0

    def take(n):
        nonlocal off
        off = (off + 15) // 16 * 16
        start, off = off, off + n
        return start

    c["L_RING"] = take(NW * nsk * SLOT)
    c["L_A"] = take(nab * RG * acb)
    c["L_AS"] = take(nab * KCS * NSC_BLK * 64 * 4)
    # LB GEMM2 prefetch: the second intermediate buffer (rows, then scales)
    # reuses the A ring (idle once the CTA's GEMM1 units are done)
    c["L_B1"] = c["L_A"]
    c["L_B1S"] = c["L_A"] + RG * c["SI_STRIDE"]
    c["B1_FITS"] = c["L_B1S"] + RG * (I // 32) <= c["L_AS"] + nab * KCS * NSC_BLK * 64 * 4
    c["L_INTER"] = take(
        max(RG * c["SI_STRIDE"], agr * (c["XB"] + H // 32) - RG * (I // 32))
    )
    c["L_INTERS"] = take(RG * (I // 32))
    # LB: one row chunk's routes at a time; the dynamic schedule: an expert's
    # routes of at most DYN_MAX tokens; else up to every token's
    nrix = 256 if lb else min(TMAX, DYN_MAX) if dyn_e else TMAX
    c["L_RIX"] = take(nrix * 4)
    c["L_WT"] = take(nrix * 4)
    c["L_CTL"] = take(128 * 4)
    c["L_DYN"] = take((2 * c["NBW"] + dyn_e) * 4)
    # row chunks of the dynamic schedule: routes per expert, then one packed
    # (expert, chunk, chunks) entry per chunk
    c["L_DCNT"] = take(dyn_e * 4) if nch else 0
    c["L_DCH"] = take((nch + 2 * NW) * 4) if nch else 0
    # LB: route list offset of every expert, then (per CTA) its routes ahead
    # of the CTA's slice of the routing
    c["L_EOFF"] = take(dyn_e * 4) if lb else 0
    c["L_DPRE"] = take(dyn_e * 4) if lb else 0
    c["L_CLS"] = take((TMAX + 31) // 32 * 4)
    c["LDS_BYTES"] = (off + 127) // 128 * 128
    assert H // 256 <= NCHA_MAX
    assert KS1 % nsk == 0 and nsk % KCS == 0 and nsk >= NSK
    assert (nsk - 1) * OPS <= 63 and c["NA_L"] * (max(ALOAD_DEPTH, nab) - 1) <= 63
    assert not lb or RG <= 256
    assert c["NCK"] <= NCK_MAX and c["NCK"] <= 31
    return c


C_CNT, C_BARCNT, C_BARGEN, C_LRED, C_PULL, C_EPOCH = 0, 1, 2, 3, 4, 5
C_USEQ, C_UROWS, C_NSIG, C_LQ, C_LPUB = 6, 7, 8, 9, 10
# routes key (expert + 1, plus the row chunk) that L_RIX / L_WT / C_CNT hold
# (0: none; zeroed per launch). Not next to C_CLAIM: col_claim writes NW slots.
C_GEXP = 11
C_DONE = 12
C_ASEQ = 16
C_AFREE = 20
C_QBASE = 24
C_MBOX = 28
C_FBITS = 36
C_AGK = 37
C_CLS = 38
C_YAGM = 39
C_NCH = 40
C_PRDY, C_FRDY, C_PBITS = 60, 61, 62
C_RLAND = 63
C_AGFREE = 41
C_PLAN, C_NACT = 42, 43
C_ARDY = 44
C_DYNP = 45
C_CDONE = 46
C_COLJ = 47
C_UNIT = 48
C_DXON = 54
C_VBON = 55
C_CLAIM = 56
C_XC, C_XC_N = 64, 32
# LB (no column-split units, so the C_XC counters are free): bank of this
# launch's counters; set once the CTA saw every route list slice
C_LBB, C_LBN = C_XC, C_XC + 1
C_UL = 96
UL_MAX = (128 - C_UL) // 4
# LB queue: GEMM1 units this CTA posted to its A loader; it claimed past them
C_G1N, C_G1FIN = 66, 67
# LB queue: the XCD queue (this CTA's + C_QK) it claims from
C_QK = 68
# tail (tn): this CTA takes the row
C_TNL = 69
# LB GEMM2 prefetch: units ready, buffers released, no more units; unit n's
# (row chunk entry, column group, rows, chunk) at C_G2U + (n % 2) * 4
C_G2RDY, C_G2FREE, C_G2END, C_G2V, C_G2U = 76, 77, 78, 79, 80
C_G2LAST = 88  # LB prefetch: compute in the last groups of unit C_G2LAST - 1
# LB GEMM2: fetch index + 1 of the claim in C_G2V; per compute wave: its ring
# already holds the first slots of its next unit
C_G2VN, C_PFW = 89, 90
# LB mixed GEMM1 split: row chunks >= C_LBJ2 run as KS2 single-block pieces
C_LBJ2 = 94
# LB (ZMA): this CTA's masked route rows are zeroed
C_ZDONE = 95
C_TLN, C_TLI, C_TLA, C_TLB, C_TLC, C_TLD = 70, 71, 72, 73, 74, 75  # [tl] (LB: free C_XC slots) events written; import end; wave 0 A / B wait
TLU_EV = 32  # [tl] events per CTA
TLU_BASE = CTRL_INTS + NCTA_MAX * 16 + 128  # [tl]
TLU_INTS = NCTA_MAX * TLU_EV * 8  # [tl]


@functools.cache
def compile_mega_moe_tp(
    *,
    H: int,
    I: int,
    TOPK: int,
    MT: int,
    TMAX: int,
    E: int,
    act: str = "silu",
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
    swiglu_limit: float | None = None,
    npieces: int = 0,
    route_fp8: bool = False,
    agr: int = 1,
    tp: int = MAX_TP,
    ar: bool = False,
    xsplit: int = 0,
    xrem: int = 0,
    xw: int = 0,
    dyn_e: int = 0,
    nab: int = 4,
    nsk: int = NSK,
    npp: int = 1,
    ll_rs: bool = False,
    ll_route: bool = False,
    xl: bool = False,
    xl_s0: int = 0,
    comm_bf16: bool = False,
    xrep: bool = False,
    ag8: bool = False,
    dx: bool = False,
    rch: int = 0,
    nch: int = 0,
    vb: bool = True,
    a8: bool = False,
    lb: bool = False,
    lbq: int = 1,
    tn: int = 0,
    tn_eps: float = 1e-6,
    tn_gemma: bool = True,
    lbpf: bool = False,
    lbp: int = 0,
):
    """Build the launcher for one (shape, variant) instance.

    ar: AR/AR layer (input is every rank's bf16 partial of all tokens; the
        input is reduce-scattered, then all-gathered as MXFP4; the output is
        all-reduced into ``yall``).
    dyn_e: dynamic schedule over the active experts (E = dyn_e), P equal inter
        pieces each; piece k writes its GEMM2 partial into route region k.
    npieces: static pieces per leftover expert (partials summed by the push).
    xsplit/xrem/xw: static column split of leftover experts (GEMM1 slices per
        expert, column-split experts, column-slice width in groups).
    ll_rs: LL ReduceScatter packets; ll_route: LL route rows too.
    comm_bf16: bf16 (not MXFP8) ReduceScatter / all-reduce partials (no LL).
    xrep: every rank holds the same full input and routing of all tokens
        (each rank quantizes and all-gathers its 1/tp of the rows); with ar
        this is the standard TP layer (only the output is all-reduced), without
        it the output is reduce-scattered.
    ag8: all-reduce whose second hop (owner -> every rank) carries MXFP8 rows
        (else bf16); every rank decodes them into ``y``.
    dx: (dynamic schedule, LL route rows) when the units overflow the CTAs by
        at most half a round, every unit only runs GEMM1 and exports its
        quantized intermediate; the GEMM2 column halves of all units are then
        claimed by whichever CTAs are free.
    rch / nch: (dynamic schedule) an expert routed by R tokens runs as
        ceil(R / rch) row chunks, chunk c taking its tokens t with
        t % chunks == c, so hot experts spread over CTAs instead of looping
        row tiles on one; nch entries bound the chunk table.
    vb: (dynamic schedule) weights of single-block units into VGPRs when the
        units are few (C_VBON); else always through the LDS ring.
    a8: activations (the input and the GEMM2 intermediate) as MXFP8 (E4M3 +
        E8M0 per 32) instead of MXFP4: the GEMMs run E2M1 weights x E4M3
        activations. Dynamic schedule without DX only.
    lb: (dynamic schedule, row chunks of rch = 16 * MT rows) large batches in
        two phases without partial sums: GEMM1 units (row chunk x inter piece)
        export their quantized intermediate; then GEMM2 units (output column
        chunk x row chunk, column chunk major) take the full inter dim and are
        claimed by whichever CTA is free; a column chunk is pushed once all its
        GEMM2 units are done.
    lbq: (lb) output column chunks per GEMM2 unit (divides the chunk count).
    tn: (ag_rs) the next layer's input fused in: each output row r of this
        rank, once final, becomes res_out[r] = y[r] + res_in[r] and the
        per-token FP8 quant of GemmaRMSNorm(res_out[r]; nw, tn_eps) (RMSNorm
        unless tn_gemma), whose rows (and fp32 scales) every rank gathers into
        its qall (sall) arena buffer; tn 2: the bf16 normed rows too (qall
        after the FP8 rows of mmax * tp tokens).
    """
    DYN = dyn_e > 0
    CHUNK = DYN and rch > 0 and nch > 0
    if swiglu_limit is None:
        swiglu_limit = 7.0 if act == "swiglu" else float("inf")
    A8 = bool(a8)
    LB = bool(lb)
    TN = int(tn) in (1, 2)
    # tn 2: also the bf16 rows of the norm (before the quant), gathered after
    # the FP8 rows in qall (for a co-consumer of the next layer's input in bf16)
    TNB = int(tn) == 2
    assert not TN or (not ar and not xrep)
    c = mega_moe_tp_consts(
        H, I, MT, TMAX, agr, dyn_e, nab, nsk, nch if CHUNK else 0, A8, LB
    )
    RG, KS1, NCH, KS2, G2 = c["RG"], c["KS1"], c["NCH"], c["KS2"], c["G2"]
    CH1, CH2, SI_STRIDE = c["CH1"], c["CH2"], c["SI_STRIDE"]
    # A-ring bytes of a row's KCS K steps; activation bytes of an input row
    ACB, XB = c["ACB"], c["XB"]
    NA_ROWOPS, NSC_BLK, NA_L = c["NA_ROWOPS"], c["NSC_BLK"], c["NA_L"]
    GPC, NCK, CW, NAB, NBW = c["GPC"], c["NCK"], c["CW"], c["NAB"], c["NBW"]
    L_RING, L_A, L_AS, L_DYN = c["L_RING"], c["L_A"], c["L_AS"], c["L_DYN"]
    L_INTER, L_INTERS, L_RIX, L_WT = c["L_INTER"], c["L_INTERS"], c["L_RIX"], c["L_WT"]
    L_CTL, LDS_BYTES = c["L_CTL"], c["LDS_BYTES"]
    L_B1, L_B1S = c["L_B1"], c["L_B1S"]
    L_CLS = c["L_CLS"]
    L_DCNT, L_DCH = c["L_DCNT"], c["L_DCH"]
    L_EOFF, L_DPRE = c["L_EOFF"], c["L_DPRE"]
    RCH = int(rch)
    NCH_MAX = int(nch)
    L_WSUM = L_DCH + NCH_MAX * 4
    assert npp >= 1 and nsk % (npp * KCS) == 0 and (KS1 * npp) % nsk == 0
    assert 3 <= NAB <= 4
    NE = dyn_e
    VPL = (CW // 8 + 63) // 64
    # a dynamic schedule only runs batches of at most DYN_MAX tokens
    SCAN_TOK = min(TMAX, DYN_MAX) if DYN else TMAX
    SCAN_IT = (SCAN_TOK * TOPK // 4 + NT - 1) // NT
    SCAN_G = 16
    NPC = 1 if LB else (KS2 if DYN else max(npieces, 1))
    assert not DYN or xsplit == 0
    FP8R = bool(route_fp8)
    LL = bool(ll_rs)
    CB16 = bool(comm_bf16)
    assert not (CB16 and LL)
    TB = 1 if DYN else max(1, (RED_INFLIGHT // 2 if NPC > 2 else RED_INFLIGHT) // TOPK)
    DYN_PS = [p for p in range(1, KS2 + 1) if KS2 % p == 0]
    # [exp] force the dynamic schedule's inter pieces per unit
    DYN_P = int(os.environ.get("AITER_MEGAMOE_TP_DYNP", "0"))  # [exp]
    if DYN_P in DYN_PS:  # [exp]
        DYN_PS = [DYN_P]  # [exp]
    DLL = LL and bool(ll_route) and (DYN or NPC == 1)
    assert not DLL or FP8R
    XREP = bool(xrep)
    # routing of every token on every rank (no routing all-gather)
    RREP = bool(ar) or XREP
    # LB (ag_rs): an aux wave zeroes the masked route rows (off the compute
    # waves' path to their first unit); the waits on it need that wave
    ZMA = LB and not RREP and not DLL
    # input is each rank's partial of every token: reduce it before quantizing
    AIN = bool(ar) and not XREP
    MLL = DLL and not RREP
    ARLL = DLL and bool(ar)
    AG8 = bool(ag8) and bool(ar) and not ARLL
    TLX = os.environ.get("AITER_MEGAMOE_TP_TL", "0") == "1"  # [tl]
    # dynamic schedule (vb): when the units take at most ~1.5 rounds of the
    # CTAs (C_VBON), the weights of single-block units go straight into VGPRs
    # (an LDS-DMA ring caps a wave at ~9 GB/s; with few units that, not HBM,
    # bounds the GEMMs, with more of them the VGPR path only costs). On model
    # routing the units are rarely that few: the GEMM2 VGPR ring drains at
    # every group's route stores and costs ~12% (m3 tp4, 64-128 tokens).
    VBS = os.environ.get("AITER_MEGAMOE_TP_VBS") == "1"  # [exp]
    VB = (DYN and bool(vb)) or VBS
    AUX_B = int(os.environ.get("AITER_MEGAMOE_TP_VBAUX", "0"))  # [exp]

    def _g2_step(n):
        nsk2 = _nsk2_for(n, G2)
        return nsk2 * n // math.gcd(nsk2, n) // n

    # GEMM2 column halves of a DX unit, in groups of NW * 64 columns: a
    # multiple of every piece width's group step (the second may be shorter)
    DX_STEP = math.lcm(*[_g2_step(KS2 // q) for q in DYN_PS]) if DYN else G2
    DX_CG = ((G2 + 1) // 2 + DX_STEP - 1) // DX_STEP * DX_STEP
    DX_NG = (G2 + DX_CG - 1) // DX_CG
    DX = bool(dx) and DYN and DLL and KS2 <= XQ_P and G2 % DX_STEP == 0
    ROW_B = 2 * H if DLL else (H + H // 32 if FP8R else 2 * H)
    XSPLIT = xsplit > 0
    assert not XSPLIT or (xw % gemm2_group_step(H, I) == 0 and xw % GPC == 0)
    assert xsplit <= XQ_P
    XCOL_PER_CHUNK = xrem if XSPLIT else 0
    XL = bool(xl)
    assert not XL or (XSPLIT and not DYN and not LL and NPC == 1)
    NV = 2 * NCK if XL else NCK
    assert NV <= 31
    AR = bool(ar)
    TPC = int(tp)
    AGR = int(agr)
    # the MXFP4 intermediate layout of DX / column-split units stays MXFP4-only
    assert not A8 or (DYN and not DX and not XSPLIT)
    # (routing of every token: replicated, or all-gathered as plain ids)
    assert not LB or (CHUNK and not LL and not XSPLIT and not A8 and RCH == RG)
    assert not LB or NCH_MAX <= NCHLB_MAX
    LBQ = int(lbq) if LB else 1
    assert NCK % LBQ == 0
    # (a GEMM2 unit's column groups: whole laps of its weight ring)
    assert not LB or LBQ * GPC % gemm2_group_step(H, I) == 0
    NCG = NCK // LBQ
    # LB column groups (GEMM2 unit width), in output column chunks: LBQ each,
    # except 12 chunks in groups of 3: 3,3,3,2,1 (a short last group shortens
    # the push + final work left once the last GEMM2 unit is done)
    CGW = [LBQ] * NCG
    _cg = LB and NCK == 12 and LBQ == 3
    if _cg:
        CGW = [3, 3, 3, 2, 1]
        NCG = len(CGW)
    CGB = [sum(CGW[:k]) for k in range(NCG + 1)]  # group k: chunks [CGB[k], CGB[k+1])
    # LB: the expert-sorted route list (route index, then routing weight) after
    # the exported intermediate (MXFP4 rows, then their scales) in xg
    XG_LIST = TMAX * TOPK * (I // 2 + I // 32)
    RBITS = TMAX.bit_length()
    # LB plan: routing loads (4 routes each) per compute thread
    LB_IT = (TMAX * TOPK // 4 + NT - 1) // NT
    LB_B = min(LB_IT, 10)
    # LB GEMM1 inter pieces per row chunk: picked at run time (the candidates,
    # the first kept on ties), or forced (lbp)
    LB_PS = [p for p in range(1, KS2 + 1) if KS2 % p == 0]
    if lbp and int(lbp) in LB_PS:
        LB_PS = [int(lbp)]
    # LB: one round of 2-piece GEMM1 units plus single-block pieces for the
    # few row chunks past it (run-time P choice only)
    LBMIX = LB and MT <= 3 and len(LB_PS) > 1 and 2 in LB_PS and KS2 % 2 == 0
    LBMIX_DIV = 4  # (at most C / LBMIX_DIV extra units)
    # lanes per row of a 256-element activation chunk (16 B each)
    XLPR = (256 if A8 else 128) // 16
    assert AGR <= 64 // XLPR
    # LB: route lists, GEMM1 intermediates and GEMM2 route rows go out system
    # scope (written through), so the LB counters after them need no release
    # fence (L2 write-back)
    AUX_RT = AUX_SYS if LB else AUX_SC1
    # A chunks the loader keeps in flight (LB: every A buffer)
    # (chunk p is published once p + ADEPTH - 1 is issued, i.e. once the
    # consumer freed chunk p + ADEPTH - 1 - NAB: below p - 1 only if
    # ADEPTH < NAB, else the consumer waits a loader round trip per chunk)
    ADEPTH = min(ALOAD_DEPTH, NAB - 1) if LB and NAB > 3 else ALOAD_DEPTH
    # LB: the A loader wave prefetches the GEMM2 units (routes + intermediate
    # into the other of two buffers) while the compute waves run the last
    LBPF = (
        LB
        and c["B1_FITS"]
        and bool(lbpf)
    )
    # LB at MT >= 4: units of at most 48 rows run 3 row tiles
    MTSKIP = LB and MT >= 4 and not VB
    name = (
        f"mega_moe_tp_fused_h{H}_i{I}_e{E}_k{TOPK}_mt{MT}_t{TMAX}_{act}"
        + (f"_lim{swiglu_limit:g}" if math.isfinite(swiglu_limit) else "")
        + (f"_p{NPC}" if NPC > 1 else "")
        + ("_r8" if route_fp8 else "")
        + f"_ag{agr}_tp{tp}"
        + ("_ar" if ar else "")
        + (f"_x{xsplit}r{xrem}w{xw}" if xsplit else "")
        + (f"_dyn{NE}" if DYN else "")
        + (f"_dm{DYN_MAX}" if DYN and DYN_MAX != 256 else "")  # [exp]
        + (f"_ch{RCH}n{NCH_MAX}" if CHUNK else "")
        + (f"_fp{DYN_P}" if DYN and DYN_PS == [DYN_P] else "")  # [exp]
        + (f"_nab{NAB}" if NAB != 4 else "")
        + (f"_nsk{nsk}" if nsk != NSK else "")
        + (f"_npp{npp}" if npp > 1 else "")
        + ("_ll" if LL else "")
        + ("_llr" if DLL else "")
        + (f"_xl{xl_s0}" if XL else "")
        + ("_cb16" if CB16 else "")
        + ("_xrep" if XREP else "")
        + ("_ag8" if AG8 else "")
        + ("_tlx" if TLX else "")  # [tl]
        + (f"_dx{DX_NG}" if DX else "")
        + (f"_vbs{AUX_B}" if VBS else f"_a{AUX_B}")  # [exp]
        + ("_novb" if DYN and not VB else "")
        + ("_a8" if A8 else "")
        + ("_lb" if LB else "")
        + (f"_q{LBQ}" if LBQ > 1 else "")
        + (("_cg" + "x".join(str(w) for w in CGW)) if _cg else "")
        + (f"_lp{LB_PS[0]}" if LB and len(LB_PS) == 1 else "")
        + (f"_tn{int(tn)}e{struct.unpack('<I', struct.pack('<f', tn_eps))[0]:x}" if tn else "")
        + ("_rms" if tn and not tn_gemma else "")
        + ("_zma" if ZMA else "")
        + (f"_mix{LBMIX_DIV}" if LBMIX else "")
        + ("_mtskip" if MTSKIP else "")
        + (f"_ad{ADEPTH}" if ADEPTH != ALOAD_DEPTH else "")
        + ("_pf" if LBPF else "")
    )
    const_expr = fx.const_expr
    # flydsl's cache key ignores module constants: the kernel references this tag
    layout_tag = f"{CTRL_INTS}/{FLAG_INTS}/{DEADLINE}/{UNIT_REC}/{NCTA_MAX}"

    def i32(v):
        raw = v if isinstance(v, ir.Value) else getattr(v, "ir_value", lambda: None)()
        if raw is not None and isinstance(raw.type, ir.IndexType):
            return fx.Int32(fx.index_cast(T.i32, raw))
        return fx.Int32(v)

    class _LazyTy:
        def __init__(self, fn):
            self.fn = fn

    V4I = _LazyTy(lambda: T.vec(4, T.i32))
    V4F = _LazyTy(lambda: T.vec(4, T.f32))
    V2I = _LazyTy(lambda: T.vec(2, T.i32))

    def _ty(t):
        return t.fn() if isinstance(t, _LazyTy) else t

    def lds_ptr(base, off, ty, align=4):
        ty = _ty(ty)
        if isinstance(ty, ir.VectorType):
            ty = ty.element_type
        return fx.inttoptr(
            fx.PointerType.get(ty, fx.AddressSpace.Shared, align), base + i32(off)
        )

    def lds_ld(base, off, ty=None, align=4):
        ty = _ty(ty) if ty is not None else T.i32
        return fx.ptr_load(lds_ptr(base, off, ty, align), result_type=ty)

    def lds_ld_i32(base, off):
        return i32(lds_ld(base, off))

    def lds_st(base, off, val, align=4):
        v = _u(val)
        fx.ptr_store(v, lds_ptr(base, off, v.type, align))

    MONO, ACQ, REL, ACQ_REL = (
        fx.AtomicOrdering.Monotonic,
        fx.AtomicOrdering.Acquire,
        fx.AtomicOrdering.Release,
        fx.AtomicOrdering.AcqRel,
    )

    def lds_p(base, off):
        return fx.inttoptr(
            fx.PointerType.get(T.i32, fx.AddressSpace.Shared, 4),
            fx.Int32(base + i32(off)),
        )

    def gptr(addr):
        return fx.inttoptr(
            fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), fx.Int64(addr)
        )

    def lds_atomic_add(base, off, v, ordering=MONO):
        return i32(
            fx.atomic_add(
                lds_p(base, off), i32(v), syncscope="workgroup", ordering=ordering
            )
        )

    def lds_atomic_or(base, off, v):
        return i32(fx.atomic_or(lds_p(base, off), i32(v), syncscope="workgroup"))

    def lds_ld_acq(base, off):
        return i32(
            fx.generic_load(
                lds_p(base, off),
                dtype=fx.Int32,
                memory_order=MONO,
                syncscope="workgroup",
                volatile=True,
            )
        )

    def lds_st_rel(base, off, v):
        fx.generic_store(
            lds_p(base, off), i32(v), memory_order=REL, syncscope="workgroup"
        )

    def lds_cas(base, off, cmp, new):
        return i32(
            fx.atomic_cas(lds_p(base, off), i32(cmp), i32(new), syncscope="workgroup")[
                0
            ]
        )

    def _ctpop(v):
        return i32(fx.ctpop(i32(v)))

    def g_ld_rel(addr, scope):
        return i32(
            fx.generic_load(
                gptr(addr),
                dtype=fx.Int32,
                memory_order=MONO,
                syncscope=scope,
                volatile=True,
            )
        )

    def g_ld_sys(addr):
        return g_ld_rel(addr, "one-as")

    def g_st_sys(addr, v):
        fx.generic_store(gptr(addr), i32(v), memory_order=MONO, syncscope="one-as")

    def g_add_agent(addr, v):
        return i32(fx.atomic_add(gptr(addr), i32(v), syncscope="agent"))

    def g_or_agent(addr, v):
        fx.atomic_or(gptr(addr), i32(v), syncscope="agent")

    def g_ld_i32(addr):
        return i32(fx.generic_load(gptr(addr), dtype=fx.Int32))

    def uni(v):
        return i32(rocdl.readfirstlane(T.i32, _u(i32(v))))

    def uni64(v):
        v = fx.Int64(v)
        lo = uni(fx.Int32(v & fx.Int64(0xFFFFFFFF)))
        hi = uni(fx.Int32(v >> fx.Int64(32)))
        return (fx.Int64(hi) << fx.Int64(32)) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))

    # Buffer resources from raw arena addresses (base + per-lane byte offset):
    # no layout form, so these stay on buffer_ops / raw buffer load/store with
    # explicit cache bits (sc0/sc1/nt).
    def rsrc(addr, nbytes=None):
        if nbytes is None:
            return buffer_ops.create_buffer_resource_from_addr(_u(uni64(addr)))
        return buffer_ops.create_buffer_resource_from_addr(
            _u(uni64(addr)), num_records_bytes=_u(fx.Int64(uni(nbytes)))
        )

    def bld(rs, voff, soff, ty, aux):
        return rocdl.raw_ptr_buffer_load(
            _ty(ty), rs, _u(i32(voff)), _u(i32(soff)), aux=_attr(aux)
        )

    def bst(val, rs, voff, soff, aux):
        rocdl.raw_ptr_buffer_store(
            _u(val), rs, _u(i32(voff)), _u(i32(soff)), aux=_attr(aux)
        )

    def asm(text, cons="", args=()):
        _llvm.inline_asm(None, [_u(a) for a in args], text, cons, has_side_effects=True)

    # LDS-DMA (buffer_load ... lds) has no wrapper: inline asm, so the compiler
    # neither sees nor waits on it; waits are explicit vmcnt counts.
    def dma16(lds_addr, rs, voff, soff, nt=False):
        asm(
            "s_mov_b32 m0, $0\n\tbuffer_load_dwordx4 $1, $2, $3 offen"
            + (" nt" if nt else "")
            + " lds",
            "s,v,s,s",
            (uni(lds_addr), i32(voff), rs, uni(soff)),
        )

    def dma4(lds_addr, rs, voff, soff, sys=False):
        asm(
            "s_mov_b32 m0, $0\n\tbuffer_load_dword $1, $2, $3 offen"
            + (" sc0 sc1" if sys else "")
            + " lds",
            "s,v,s,s",
            (uni(lds_addr), i32(voff), rs, uni(soff)),
        )

    # Waits stay inline asm: they must count the LDS-DMA loads above, which the
    # compiler cannot see (its s_waitcnt pass would merge or drop the waits).
    def wait_vm(n):
        # VMEM completes in order: extra ops only make a wait stricter
        asm(f"s_waitcnt vmcnt({min(int(n), 63)})")

    def wait_lgkm0():
        asm("s_waitcnt lgkmcnt(0)")

    def ctrl_at(a, idx):
        return fx.Int64(a["ctrl"]) + fx.Int64(i32(idx) * i32(4))

    def peer_sel(a, p):
        v = fx.Int64(a["peer"][0])
        for j in range_constexpr(1, MAX_TP):
            v = (p == i32(j)).select(fx.Int64(a["peer"][j]), v)
        return v

    def peer_rs(a, p, key, nbytes=None):
        return rsrc(fx.Int64(a["peer"][p]) + fx.Int64(a[key]), nbytes)

    def own_rs(a, key, nbytes=None):
        return rsrc(a["mine"] + fx.Int64(a[key]), nbytes)

    def route_region_bytes(ttot):
        return ttot * i32(TOPK * ROW_B)

    def amax(vals):
        am = _fabs_f32(vals[0])
        for v in vals[1:]:
            am = am.maximumf(_fabs_f32(v))
        return am

    def fp4_pack(vals, qs):
        pk = _u(i32(0))
        for k in range(len(vals) // 2):
            pk = rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, pk, _u(vals[2 * k]), _u(vals[2 * k + 1]), _u(qs), k
            )
        return fx.Int32(pk)

    def fp8x4_pack(f, qs):
        zero = fx.Vector.filled(2, 0, fx.Int16)
        w = rocdl.cvt_scalef32_pk_fp8_f32(
            T.vec(2, T.i16), _u(zero), _u(f[0]), _u(f[1]), _u(qs), False
        )
        w = rocdl.cvt_scalef32_pk_fp8_f32(
            T.vec(2, T.i16), w, _u(f[2]), _u(f[3]), _u(qs), True
        )
        return fx.Int32(fx.Vector(w).bitcast(fx.Int32)[0])

    def fp8x4_unpack(d, sc):
        out = []
        for sel in (False, True):
            v = fx.Vector(
                rocdl.cvt_scalef32_pk_f32_fp8(T.vec(2, T.f32), _u(i32(d)), _u(sc), sel)
            )
            out += [fx.Float32(v[0]), fx.Float32(v[1])]
        return out

    def e8_scale(e8):
        return ((fx.Int32(e8) & i32(0xFF)) << i32(23)).bitcast(fx.Float32)

    def fp8x8_decode(ld):
        dv = fx.Vector(ld[0])
        sc = e8_scale(ld[1])
        return fp8x4_unpack(fx.Int32(dv[0]), sc) + fp8x4_unpack(fx.Int32(dv[1]), sc)

    def bf16_bits(f):
        return fx.Int32(
            fx.Vector.from_elements(
                [fx.Float32(f).to(fx.BFloat16)], fx.BFloat16
            ).bitcast(fx.Int16)[0]
        ) & i32(0xFFFF)

    def pack_bf16x2(a, b):
        return bf16_bits(a) | (bf16_bits(b) << i32(16))

    def bf16x8_to_f32(d):
        dv = fx.Vector(d)
        out = []
        for q in range_constexpr(4):
            w = fx.Int32(dv[q])
            out.append((w << i32(16)).bitcast(fx.Float32))
            out.append((w & i32(-65536)).bitcast(fx.Float32))
        return out

    def pack_bf16x8(acc):
        return fx.Vector.from_elements(
            [pack_bf16x2(acc[2 * q], acc[2 * q + 1]) for q in range(4)], fx.Int32
        )

    def sum_live(acc, vals, live):
        return [x + live.select(y, fx.Float32(0.0)) for x, y in zip(acc, vals)]

    # LL packets [d0, epoch, d1, tag]: tag = epoch << 8 | E8M0 of the 8 E4M3 in d0, d1
    def ll_pkt(a, d0, d1, e8):
        tag = ((a["epoch"] & i32(0xFFFFFF)) << i32(8)) | e8
        return fx.Vector.from_elements([d0, a["epoch"], d1, tag], fx.Int32)

    def ll_ok(a, q):
        return (fx.Int32(q[1]) == a["epoch"]) & (
            fx.Int32(q[3]).shrui(i32(8)) == (a["epoch"] & i32(0xFFFFFF))
        )

    def ll_vals(q):
        sc = e8_scale(q[3])
        return fp8x4_unpack(fx.Int32(q[0]), sc) + fp8x4_unpack(fx.Int32(q[2]), sc)

    def ll_pending(a, lane, pkts):
        bad = i32(0)
        for q, live in pkts:
            bad = fx.max(
                bad, (live & (ll_ok(a, q) == fx.Boolean(False))).select(i32(1), i32(0))
            )
        return wave_red(bad, lane, fx.max)

    def wave_red(v, lane, op):
        for k in (1, 2, 4, 8, 16, 32):
            v = op(
                v, i32(rocdl.ds_bpermute(T.i32, _u((lane ^ i32(k)) * i32(4)), _u(v)))
            )
        return uni(v)

    def route_load(rs, region, ttot, ridx, col, take=None):
        if FP8R:
            doff = region + ridx * i32(H) + col
            soff = region + ttot * i32(TOPK * H) + ridx * i32(H // 32) + col // i32(32)
            if take is not None:
                end = region + route_region_bytes(ttot) * i32(NPC)
                doff, soff = take.select(doff, end), take.select(soff, end)
            return (bld(rs, doff, 0, V2I, AUX_SC1), bld(rs, soff, 0, T.i8, AUX_SC1))
        off = region + (ridx * i32(H) + col) * i32(2)
        if take is not None:
            off = take.select(off, region + route_region_bytes(ttot) * i32(NPC))
        return bld(rs, off, 0, V4I, AUX_SC1)

    def route_decode(ld):
        return fp8x8_decode(ld) if FP8R else bf16x8_to_f32(ld)

    def _swap(fn, x, y):
        # the permlane swaps return an {i32, i32} struct (no wrapper)
        st = _llvm.StructType.get_literal([T.i32, T.i32])
        r = fn(st, _u(i32(x)), _u(i32(y)), False, False)
        return (
            i32(_llvm.ExtractValueOp(T.i32, r, [0]).res),
            i32(_llvm.ExtractValueOp(T.i32, r, [1]).res),
        )

    def swap16(x, y):
        return _swap(rocdl.permlane16_swap, x, y)

    def swap32(x, y):
        return _swap(rocdl.permlane32_swap, x, y)

    def widen(v4):
        z = fx.Vector.filled(4, 0, fx.Int32)
        return _u(fx.Vector(v4).shuffle(z, list(range(8))))

    def mfma(a4, b4, cacc, sa, sb, b8=False):
        # weights (E2M1, 4 dwords) x activations (E2M1 4 dwords; b8: E4M3 8)
        bx = _u(b4) if b8 else widen(b4)
        return fx.Vector(
            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                _ty(V4F),
                [widen(a4), bx, _u(cacc), 4, 0 if b8 else 4, 0, _u(i32(sa)), 0, _u(i32(sb))],
            )
        )

    def cat8(lo, hi):
        return _u(fx.Vector(lo).shuffle(fx.Vector(hi), list(range(8))))

    def a_frag(abuf, row, k, q4):
        # this lane's operand for step k of a row's A chunk, whose 16 B pieces
        # sit at piece ^ (row & 7): MXFP4 piece k * 4 + q4; E4M3 (f8f6f4 ABI)
        # pieces k * 8 + q4 and k * 8 + 4 + q4, 64 B apart
        if const_expr(A8):
            lo = (i32(k * 8) + q4) ^ (row & i32(7))
            hi = (i32(k * 8 + 4) + q4) ^ (row & i32(7))
            return cat8(
                lds_ld(abuf, row * i32(ACB) + lo * i32(16), V4I, 16),
                lds_ld(abuf, row * i32(ACB) + hi * i32(16), V4I, 16),
            )
        col = (i32(k * 4) + q4) ^ (row & i32(7))
        return lds_ld(abuf, row * i32(ACB) + col * i32(16), V4I, 16)

    def act_batch(gs, us):
        return _activation_mul_batch(
            gs,
            us,
            act=act,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            swiglu_limit=swiglu_limit,
        )

    @traced
    def cbar(L, tid):
        wait_lgkm0()
        if (tid % i32(64)) == i32(0):
            g = lds_ld_acq(L, L_CTL + C_BARGEN * 4)
            old = lds_atomic_add(L, L_CTL + C_BARCNT * 4, 1, ACQ_REL)
            if old == i32(NW - 1):
                lds_st(L, L_CTL + C_BARCNT * 4, i32(0))
                lds_st_rel(L, L_CTL + C_BARGEN * 4, g + i32(1))
            else:
                cur = lds_ld_acq(L, L_CTL + C_BARGEN * 4)
                while cur == g:
                    rocdl.s_sleep(0)
                    cur = lds_ld_acq(L, L_CTL + C_BARGEN * 4)
        rocdl.sched_barrier(0)

    @traced
    def spin_lds_ge(L, off, target):
        cur = lds_ld_acq(L, off)
        while cur < target:
            rocdl.s_sleep(0)
            cur = lds_ld_acq(L, off)

    def report(a, code):
        g_or_agent(ctrl_at(a, CTRL_ERR), code)

    @traced
    def _report_if(a, bad, code):
        if bad:
            report(a, code)

    def _now():
        # s_memrealtime (100 MHz) has no wrapper
        return fx.Int64(
            _llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], [])
        )

    _TLA = [None]  # [tl] the kernel's arg dict, for helpers without it
  # [tl]
    @traced  # [tl]
    def tlx(a, k):  # [tl]
        if const_expr(TLX):  # [tl]
            off = i32(CTRL_INTS) + i32(gpu.block_id("x")) * i32(16) + i32(k)  # [tl]
            g_st_sys(ctrl_at(a, off), fx.Int32(_now()))  # [tl]
  # [tl]
    @traced  # [tl]
    def tlc(a, slot):  # [tl] per-chunk stamps after the per-CTA ones
        if const_expr(TLX):  # [tl]
            g_st_sys(ctrl_at(a, i32(CTRL_INTS + NCTA_MAX * 16) + slot), fx.Int32(_now()))  # [tl]
  # [tl]
    @traced  # [tl]
    def tlx0(a, tid, k):  # [tl]
        if tid == i32(0):  # [tl]
            tlx(a, k)  # [tl]

    # [tl] per-unit events: TLU_EV records of 8 ints per CTA after the stamps
    @traced  # [tl]
    def tle(L, a, tid, w):  # [tl]
        if const_expr(TLX):  # [tl]
            if tid == i32(0):  # [tl]
                n = lds_ld_i32(L, L_CTL + C_TLN * 4)  # [tl]
                if n < i32(TLU_EV):  # [tl]
                    base = i32(TLU_BASE) + i32(gpu.block_id("x")) * i32(TLU_EV * 8)  # [tl]
                    for k in range_constexpr(len(w)):  # [tl]
                        g_st_sys(ctrl_at(a, base + n * i32(8) + i32(k)), w[k])  # [tl]
                    lds_st(L, L_CTL + C_TLN * 4, n + i32(1))  # [tl]

    def tnow():  # [tl]
        return fx.Int32(_now())  # [tl]

    @traced  # [tl]
    def tl_acc(L, tid, slot, t0):  # [tl] wave 0 adds the time since t0
        if const_expr(TLX):  # [tl]
            rocdl.sched_barrier(0)  # [tl]
            dt = tnow() - t0  # [tl]
            if tid == i32(0):  # [tl]
                lds_st(L, L_CTL + slot * 4, lds_ld_i32(L, L_CTL + slot * 4) + dt)  # [tl]
            rocdl.sched_barrier(0)  # [tl]
  # [tl]
    def _alive(t0):
        return (_now() - t0) < fx.Int64(DEADLINE)

    @traced
    def spin_sys_ge(addr, target, a=None):
        cur = g_ld_sys(addr)
        t0 = _now()
        while (cur < target) & _alive(t0):
            rocdl.s_sleep(2)
            cur = g_ld_sys(addr)
        if const_expr(a is not None):
            _report_if(a, cur < target, ERR_FLAG)

    def masked(e):
        return fx.Int32(e).bitcast(fx.Uint32) >= fx.Uint32(E)

    @traced
    def gather_routes(L, tid, ids_addr, tw_addr, ttot, expert):
        # consecutive units of one expert (pair / split schedules: its GEMM1
        # slice, then its GEMM2 columns) reuse the routes already in LDS
        key = expert + i32(1)
        if lds_ld_i32(L, L_CTL + C_GEXP * 4) != key:
            _gather_scan(L, tid, ids_addr, tw_addr, ttot, expert, key)
        return fx.min(lds_ld_i32(L, L_CTL + C_CNT * 4), i32(TMAX))

    @traced
    def gather_routes_chunk(L, tid, ids_addr, tw_addr, ttot, expert, cc, ce):
        # routes of row chunk cc of ce: the tokens t with t % ce == cc
        key = expert + i32(1) + (cc << i32(12)) + (ce << i32(20))
        if lds_ld_i32(L, L_CTL + C_GEXP * 4) != key:
            if ce == i32(1):
                _gather_scan(L, tid, ids_addr, tw_addr, ttot, expert, key)
            else:
                _gather_scan_chunk(L, tid, ids_addr, tw_addr, ttot, expert, key, cc, ce)
        return fx.min(lds_ld_i32(L, L_CTL + C_CNT * 4), i32(TMAX))

    @traced
    def _gather_scan_chunk(L, tid, ids_addr, tw_addr, ttot, expert, key, cc, ce):
        inv = fx.Float32(1.0) / ce.to(fx.Float32)
        _gather_scan(L, tid, ids_addr, tw_addr, ttot, expert, key, (cc, ce, inv))

    def _in_chunk(idx, chunk):
        # token t of route idx is in chunk cc of ce iff t % ce == cc; the
        # quotient in fp32 is exact for t <= 8192 and ce <= 255
        if chunk is None:
            return fx.Boolean(True)
        cc, ce, inv = chunk
        t = idx // i32(TOPK)
        q = ((t.to(fx.Float32) + fx.Float32(0.5)) * inv).to(fx.Int32)
        return (t - q * ce) == cc

    @traced
    def _gather_scan(L, tid, ids_addr, tw_addr, ttot, expert, key, chunk=None):
        # every wave read the previous count before it is reset
        tlx0(_TLA[0], tid, 12)  # [tl]
        cbar(L, tid)
        if tid == i32(0):
            lds_st(L, L_CTL + C_CNT * 4, i32(0))
        cbar(L, tid)
        tlx0(_TLA[0], tid, 14)  # [tl]
        n = ttot * i32(TOPK)
        rid = rsrc(ids_addr, n * i32(16 if MLL else 4))
        if const_expr(MLL):
            for i_ in range(tid, n, i32(NT)):
                i = i32(i_)
                pk = fx.Vector(bld(rid, i * i32(16), 0, V4I, 0))
                hit = (fx.Int32(pk[0]) == expert) & _in_chunk(i, chunk)
                _gather_one(L, i, hit, fx.Int32(pk[2]))
        else:
            n4 = n // i32(4)
            rtw = rsrc(tw_addr, n * i32(4))
            for g0 in range_constexpr(0, SCAN_IT, SCAN_G):
                its = list(range(g0, min(g0 + SCAN_G, SCAN_IT)))
                vs, ws = [], []
                for it in its:
                    q = fx.min(tid + i32(it * NT), n4 - i32(1))
                    vs.append(fx.Vector(bld(rid, q * i32(16), 0, V4I, 0)))
                    ws.append(fx.Vector(bld(rtw, q * i32(16), 0, V4I, 0)))
                # wave-compacted: slots from ballot / mbcnt, one LDS atomic
                # per wave and group (per-hit atomics serialize: ~20 us at
                # 2k tokens)
                hits, run = [], i32(0)
                for x, it in enumerate(its):
                    q = tid + i32(it * NT)
                    for j in range_constexpr(4):
                        idx = q * i32(4) + i32(j)
                        hit = (
                            (q < n4)
                            & (fx.Int32(vs[x][j]) == expert)
                            & _in_chunk(idx, chunk)
                        )
                        pos, n = _wave_rank(hit)
                        hits.append((hit, run + pos, idx, ws[x][j]))
                        run = run + n
                base = _wave_claim(L, tid, run)
                for hit, pos, idx, wv in hits:
                    _gather_put(L, hit, base + pos, idx, fx.Int32(wv))
            for idx_ in range(n4 * i32(4) + tid, n, i32(NT)):
                idx = i32(idx_)
                e = g_ld_i32(fx.Int64(ids_addr) + fx.Int64(idx) * fx.Int64(4))
                wv = g_ld_i32(fx.Int64(tw_addr) + fx.Int64(idx) * fx.Int64(4))
                _gather_one(L, idx, (e == expert) & _in_chunk(idx, chunk), wv)
        cbar(L, tid)
        tlx0(_TLA[0], tid, 11)  # [tl]
        if tid == i32(0):
            lds_st(L, L_CTL + C_GEXP * 4, key)
        cbar(L, tid)

    @traced
    def zero_masked(L, lane, a, w0, nw):
        # no expert writes the route rows of ids outside [0, E) (masked
        # routes): waves w0.. of this CTA zero those of its share of the ids,
        # after they landed and before any of its chunk signals (LL rows: any
        # time before the push, which checks their epoch tags)
        n = a["ttot"] * i32(TOPK)
        nblk = i32(gpu.grid_dim.x)
        per = (n + nblk - i32(1)) // nblk
        lo = i32(gpu.block_id("x")) * per
        hi = fx.min(lo + per, n)
        rid = rsrc(a["ids"], n * i32(16 if MLL else 4))
        if const_expr(MLL):
            # the compute waves saw every routing packet land
            if lane == i32(0):
                spin_lds_ge(L, L_CTL + C_RLAND * 4, i32(1))
            rocdl.sched_barrier(0)
        for b_ in range(lo + i32(w0 * 64), hi, i32(nw * 64)):
            i = i32(b_) + lane
            off = fx.min(i, n - i32(1)) * i32(16 if MLL else 4)
            e = fx.Int32(bld(rid, off, 0, T.i32, 0))
            bal = fx.Int64(rocdl.ballot(T.i64, _u((i < hi) & masked(e))))
            if bal != fx.Int64(0):
                for j_ in range(i32(0), i32(64), i32(1)):
                    j = i32(j_)
                    if ((bal >> fx.Int64(j)) & fx.Int64(1)) != fx.Int64(0):
                        _zero_route(a, lane, i32(b_) + j)
                wait_vm(0)

    @traced
    def zma_run(L, lane, a):
        # (an aux wave) once every rank's routing landed: this CTA's share
        # of the masked route rows, zeroed
        pend = _meta_pending(a, lane, a["epoch"])
        t0 = _now()
        while (pend != i32(0)) & _alive(t0):
            rocdl.s_sleep(1)
            pend = _meta_pending(a, lane, a["epoch"])
        zero_masked(L, lane, a, 0, 1)
        wait_vm(0)
        if lane == i32(0):
            lds_st_rel(L, L_CTL + C_ZDONE * 4, i32(1))

    @traced
    def _zero_route(a, lane, ridx):
        rb = route_region_bytes(a["ttot"])
        z = fx.Vector.from_elements([i32(0)] * 4, fx.Int32)
        for p in range_constexpr(NPC):
            rs = rsrc(a["routes"], rb) if p == 0 else rsrc(a["proutes"], i32(p) * rb)
            base = i32(0) if p == 0 else i32(p - 1) * rb
            for q_ in range(lane, i32(H // 8), i32(64)):
                q = i32(q_)
                if const_expr(DLL):
                    v = ll_pkt(a, i32(0), i32(0), i32(0))
                    bst(v, rs, base + (ridx * i32(H) + q * i32(8)) * i32(2), 0, AUX_SC1)
                elif const_expr(FP8R):
                    if q < i32(H // 16):
                        bst(z, rs, base + ridx * i32(H) + q * i32(16), 0, AUX_SC1)
                    if q < i32(H // 512):
                        soff = a["ttot"] * i32(TOPK * H) + ridx * i32(H // 32)
                        bst(z, rs, base + soff + q * i32(16), 0, AUX_SC1)
                else:
                    bst(z, rs, base + (ridx * i32(H) + q * i32(8)) * i32(2), 0, AUX_SC1)

    def _wave_rank(hit):
        b = fx.Int64(rocdl.ballot(T.i64, _u(hit)))
        lo = i32(b & fx.Int64(0xFFFFFFFF))
        hi = i32(b >> fx.Int64(32))
        below = rocdl.mbcnt_lo(T.i32, _u(lo), _u(i32(0)))
        below = i32(rocdl.mbcnt_hi(T.i32, _u(hi), below))
        return below, _ctpop(lo) + _ctpop(hi)

    @traced
    def _wave_claim(L, tid, n):
        got = i32(0)
        if ((tid % i32(64)) == i32(0)) & (n > i32(0)):
            got = lds_atomic_add(L, L_CTL + C_CNT * 4, n)
        return uni(got)

    @traced
    def _gather_put(L, hit, slot, idx, wv):
        if hit & (slot < i32(TMAX)):
            lds_st(L, L_RIX + slot * i32(4), idx)
            lds_st(L, L_WT + slot * i32(4), wv)

    @traced
    def _gather_one(L, idx, hit, wv):
        if hit:
            slot = lds_atomic_add(L, L_CTL + C_CNT * 4, 1)
            if slot < i32(TMAX):
                lds_st(L, L_RIX + slot * i32(4), idx)
                lds_st(L, L_WT + slot * i32(4), wv)

    def act_quant_store(L, lane, w, accg, accu, nb, mte=None):
        q4 = lane // i32(16)
        for m in range_constexpr(MT if mte is None else mte):
            row = i32(m * 16) + lane % i32(16)
            gs, us = [], []
            for t in range_constexpr(2):
                ga, ua = fx.Vector(accg[m * 2 + t]), fx.Vector(accu[m * 2 + t])
                for v in range_constexpr(4):
                    gs.append(fx.Float32(ga[v]))
                    us.append(fx.Float32(ua[v]))
            xs = [
                fx.Float32(x).to(fx.BFloat16).to(fx.Float32) for x in act_batch(gs, us)
            ]
            am = amax(xs)
            am = am.maximumf(am.shuffle_xor(i32(16), i32(64)))
            am = am.maximumf(am.shuffle_xor(i32(32), i32(64)))
            e8, qs = _e8m0_from_amax(am, max_norm=448.0 if A8 else 6.0)
            for t in range_constexpr(2):
                col = nb * i32(128) + w * i32(32) + i32(t * 16) + q4 * i32(4)
                if const_expr(A8):
                    pk = fp8x4_pack(xs[t * 4 : t * 4 + 4], qs)
                    lds_st(L, L_INTER + row * i32(SI_STRIDE) + col, pk, align=4)
                else:
                    pk = fp4_pack(xs[t * 4 : t * 4 + 4], qs)
                    lds_st(
                        L,
                        L_INTER + row * i32(SI_STRIDE) + col // i32(2),
                        fx.Int16(pk & i32(0xFFFF)),
                        align=2,
                    )
            _q0_store(L, q4, row, nb, w, e8)

    @traced
    def _q0_store(L, q4, row, nb, w, e8):
        if q4 == i32(0):
            lds_st(
                L, L_INTERS + row * i32(I // 32) + nb * i32(4) + w, fx.Int8(e8), align=1
            )

    @traced
    def wait_a_chunk(L, lane, buf, q):
        if const_expr(VB):
            # a spin loop the compiler sees makes it drain the weight loads in
            # flight (vmcnt(0)) before every A chunk
            _llvm.inline_asm(
                T.i32,
                [_u(L + i32(L_CTL + C_ASEQ * 4) + buf * i32(4)), _u(q + i32(1))],
                "1:\n\tds_read_b32 $0, $1\n\ts_waitcnt lgkmcnt(0)\n\t"
                "v_cmp_lt_i32 vcc, $0, $2\n\ts_cbranch_vccz 2f\n\t"
                "s_sleep 0\n\ts_branch 1b\n2:",
                "=&v,v,v,~{vcc},~{memory}",
                has_side_effects=True,
            )
        else:
            if lane == i32(0):
                spin_lds_ge(L, L_CTL + (C_ASEQ * 4) + buf * i32(4), q + i32(1))
        rocdl.sched_barrier(0)

    @traced
    def release_a_chunk(L, lane, buf):
        if const_expr(VB):
            # branch-free (a branch splits the GEMM1 step loop and the merge
            # drains the weight loads too); after this wave's reads of the
            # chunk, as LDS executes a wave's ops in order
            asm(
                "ds_add_u32 $0, $1",
                "v,v,~{memory}",
                (
                    L + i32(L_CTL + C_AFREE * 4) + buf * i32(4),
                    (lane == i32(0)).select(i32(1), i32(0)),
                ),
            )
        else:
            if lane == i32(0):
                lds_atomic_add(L, L_CTL + C_AFREE * 4 + buf * i32(4), 1, REL)

    def g1_group(nnb):
        # column blocks per pass over A (gemm1's choice)
        if const_expr(npp > 1):
            return ((nnb % i32(npp)) == i32(0)).select(i32(npp), i32(1))
        return i32(1)

    @traced
    def _gemm1_disp(L, tid, a, expert, i0, nnb, mte):
        if const_expr(npp > 1):
            if (nnb % i32(npp)) == i32(0):
                _gemm1(L, tid, a, expert, i0, nnb, npp, mte)
            else:
                _gemm1(L, tid, a, expert, i0, nnb, 1, mte)
        else:
            _gemm1(L, tid, a, expert, i0, nnb, 1, mte)

    @traced
    def gemm1(L, tid, a, expert, i0, nnb, rows=None):
        nnb = uni(nnb)
        if const_expr(MTSKIP and rows is not None):
            # (units of at most 48 rows: 3 row tiles)
            if uni(rows) <= i32(3 * 16):
                _gemm1_disp(L, tid, a, expert, i0, nnb, 3)
            else:
                _gemm1_disp(L, tid, a, expert, i0, nnb, MT)
            wait_vm(0)
            cbar(L, tid)
            return
        if const_expr(VB):
            if ((nnb == i32(1)) & (lds_ld_i32(L, L_CTL + C_VBON * 4) != i32(0))) | VBS:
                _gemm1_vb(L, tid, a, expert, i0, nnb, 1)
            else:
                _gemm1(L, tid, a, expert, i0, nnb, 1)
        elif const_expr(npp > 1):
            if (nnb % i32(npp)) == i32(0):
                _gemm1(L, tid, a, expert, i0, nnb, npp)
            else:
                _gemm1(L, tid, a, expert, i0, nnb, 1)
        else:
            _gemm1(L, tid, a, expert, i0, nnb, 1)
        wait_vm(0)
        cbar(L, tid)

    @traced
    def _gemm1(L, tid, a, expert, i0, nnb, P, MTE=None):
        MTE = MT if MTE is None else MTE
        lane = tid % i32(64)
        w = uni(tid // i32(64))
        e = uni(expert)
        rw = rsrc(fx.Int64(a["w1"]) + fx.Int64(e) * fx.Int64(2 * I * (H // 2)))
        rws = rsrc(a["w1s"])
        icol0 = uni(i0) + w * i32(32)
        vg0 = icol0 * i32(H // 2) + lane * i32(16)
        vg1 = vg0 + i32(16 * (H // 2))
        vu0 = vg0 + i32(I * (H // 2))
        vu1 = vu0 + i32(16 * (H // 2))
        vsl = lane * i32(4)
        sg0 = uni((e * i32(2 * I) + icol0) // i32(32)) * i32(CH1 * 256)
        su0 = uni((e * i32(2 * I) + i32(I) + icol0) // i32(32)) * i32(CH1 * 256)
        ring = uni(L + i32(L_RING) + w * i32(nsk * SLOT))
        total = nnb * i32(KS1)
        qbase = lds_ld_i32(L, L_CTL + C_QBASE * 4 + w * i32(4))
        GS = KS1 * P
        # slot uses per loop trip: whole A chunks (P * KCS uses each) and
        # whole laps of the ring (use g in slot g % nsk)
        IL = nsk * (P * KCS) // math.gcd(nsk, P * KCS)
        assert GS % IL == 0

        def issue_b(g, slot_idx):
            gc = fx.min(g, total - i32(1))
            grp = gc // i32(GS)
            r = gc - grp * i32(GS)
            kk = r // i32(P)
            pas = grp * i32(P) + (r - kk * i32(P))
            slot = ring + i32(slot_idx * SLOT)
            so = pas * i32(128 * (H // 2)) + kk * i32(1024)
            dma16(slot + i32(0), rw, vg0, so, nt=True)
            dma16(slot + i32(1024), rw, vg1, so, nt=True)
            dma16(slot + i32(2048), rw, vu0, so, nt=True)
            dma16(slot + i32(3072), rw, vu1, so, nt=True)
            sso = pas * i32((128 // 32) * CH1 * 256) + (kk // i32(2)) * i32(256)
            dma4(slot + i32(4096), rws, vsl, sg0 + sso)
            dma4(slot + i32(4096 + 256), rws, vsl, su0 + sso)

        for kk in range_constexpr(nsk):
            issue_b(i32(kk), kk)

        zero = fx.Vector.filled(4, 0.0, fx.Float32)
        NA = 4 * MTE
        for grp_ in range(i32(0), nnb // i32(P), i32(1)):
            grp = i32(grp_)
            init = [zero] * (NA * P)
            for g0_, st in range(
                grp * i32(GS), (grp + i32(1)) * i32(GS), i32(IL), init=init
            ):
                g0 = i32(g0_)
                acc = list(st)
                qb = qbase + grp * i32(NCH) + (g0 - grp * i32(GS)) // i32(P * KCS)
                for ks in range_constexpr(IL // P):
                    k = ks % KCS
                    q = qb + i32(ks // KCS)
                    buf = q % i32(NAB)
                    rocdl.sched_barrier(0)
                    if k == 0:
                        ta = tnow()  # [tl]
                        wait_a_chunk(L, lane, buf, q)
                        tl_acc(L, tid, C_TLA, ta)  # [tl]
                    abuf = L + i32(L_A) + buf * i32(RG * ACB)
                    asbuf = (
                        L
                        + i32(L_AS)
                        + buf * i32(KCS * NSC_BLK * 64 * 4)
                        + i32(k * NSC_BLK * 64 * 4)
                    )
                    kh = ks & 1
                    for pp in range_constexpr(P):
                        s_ = ks * P + pp
                        g = g0 + i32(s_)
                        tb = tnow()  # [tl]
                        wait_vm((nsk - 1) * OPS)
                        tl_acc(L, tid, C_TLB, tb)  # [tl]
                        slot = ring + i32((s_ % nsk) * SLOT)
                        b = [
                            lds_ld(slot, i32(t * 1024) + lane * i32(16), V4I, 16)
                            for t in range(4)
                        ]
                        sgw = lds_ld_i32(slot, i32(4096) + lane * i32(4))
                        suw = lds_ld_i32(slot, i32(4096 + 256) + lane * i32(4))
                        g0s = (sgw >> i32(8 * (kh * 2))) & i32(0xFF)
                        g1s = (sgw >> i32(8 * (kh * 2 + 1))) & i32(0xFF)
                        u0s = (suw >> i32(8 * (kh * 2))) & i32(0xFF)
                        u1s = (suw >> i32(8 * (kh * 2 + 1))) & i32(0xFF)
                        o = pp * NA
                        for m in range_constexpr(MTE):
                            row = i32(m * 16) + lane % i32(16)
                            af = a_frag(abuf, row, k, lane // i32(16))
                            sa = fx.Int32(
                                lds_ld(asbuf, row * i32(4) + lane // i32(16), T.i8, 1)
                            ) & i32(0xFF)
                            acc[o + m * 2 + 0] = mfma(
                                b[0], af, acc[o + m * 2 + 0], g0s, sa, A8
                            )
                            acc[o + m * 2 + 1] = mfma(
                                b[1], af, acc[o + m * 2 + 1], g1s, sa, A8
                            )
                            acc[o + 2 * MTE + m * 2 + 0] = mfma(
                                b[2], af, acc[o + 2 * MTE + m * 2 + 0], u0s, sa, A8
                            )
                            acc[o + 2 * MTE + m * 2 + 1] = mfma(
                                b[3], af, acc[o + 2 * MTE + m * 2 + 1], u1s, sa, A8
                            )
                        wait_lgkm0()
                        rocdl.sched_barrier(0)
                        issue_b(g + i32(nsk), s_ % nsk)
                        rocdl.sched_barrier(0)
                    if k == KCS - 1:
                        release_a_chunk(L, lane, buf)
                res = yield acc
            ag_stage_free(L, lane, a)
            for pp in range_constexpr(P):
                o = pp * NA
                act_quant_store(
                    L,
                    lane,
                    w,
                    res[o : o + 2 * MTE],
                    res[o + 2 * MTE : o + NA],
                    grp * i32(P) + i32(pp),
                    MTE,
                )
        _store_qbase(L, lane, w, qbase + (nnb // i32(P)) * i32(NCH))

    @traced
    def _gemm1_vb(L, tid, a, expert, i0, nnb, P):
        # _gemm1 with the weight ring in VGPRs, carried by the loops; the K
        # steps of a column block are unrolled (across a loop back-edge the
        # compiler's vmcnt bookkeeping gives up and drains the ring)
        lane = tid % i32(64)
        w = uni(tid // i32(64))
        e = uni(expert)
        rw = rsrc(fx.Int64(a["w1"]) + fx.Int64(e) * fx.Int64(2 * I * (H // 2)))
        rws = rsrc(a["w1s"])
        icol0 = uni(i0) + w * i32(32)
        vg0 = icol0 * i32(H // 2) + lane * i32(16)
        vg1 = vg0 + i32(16 * (H // 2))
        vu0 = vg0 + i32(I * (H // 2))
        vu1 = vu0 + i32(16 * (H // 2))
        vsl = lane * i32(4)
        sg0 = uni((e * i32(2 * I) + icol0) // i32(32)) * i32(CH1 * 256)
        su0 = uni((e * i32(2 * I) + i32(I) + icol0) // i32(32)) * i32(CH1 * 256)
        total = nnb * i32(KS1)
        qbase = lds_ld_i32(L, L_CTL + C_QBASE * 4 + w * i32(4))
        GS = KS1 * P

        def load_b(g):
            gc = fx.min(g, total - i32(1))
            grp = gc // i32(GS)
            r = gc - grp * i32(GS)
            kk = r // i32(P)
            pas = grp * i32(P) + (r - kk * i32(P))
            so = pas * i32(128 * (H // 2)) + kk * i32(1024)
            sso = pas * i32((128 // 32) * CH1 * 256) + (kk // i32(2)) * i32(256)
            return [bld(rw, v, so, V4I, AUX_B) for v in (vg0, vg1, vu0, vu1)] + [
                bld(rws, vsl, sg0 + sso, T.i32, 0),
                bld(rws, vsl, su0 + sso, T.i32, 0),
            ]

        ring0 = []
        for kk in range_constexpr(nsk):
            ring0 += load_b(i32(kk))
        zero = fx.Vector.filled(4, 0.0, fx.Float32)
        NA = 4 * MT
        for grp_, ost in range(i32(0), nnb // i32(P), i32(1), init=ring0):
            grp = i32(grp_)
            acc = [zero] * (NA * P)
            ring = list(ost)
            for g0c in range_constexpr(0, GS, nsk):
                g0 = grp * i32(GS) + i32(g0c)
                qb = qbase + grp * i32(NCH) + i32(g0c // (P * KCS))
                for ks in range_constexpr(nsk // P):
                    k = ks % KCS
                    q = qb + i32(ks // KCS)
                    buf = q % i32(NAB)
                    rocdl.sched_barrier(0)
                    if k == 0:
                        wait_a_chunk(L, lane, buf, q)
                    abuf = L + i32(L_A) + buf * i32(RG * ACB)
                    asbuf = (
                        L
                        + i32(L_AS)
                        + buf * i32(KCS * NSC_BLK * 64 * 4)
                        + i32(k * NSC_BLK * 64 * 4)
                    )
                    kh = ks & 1
                    for pp in range_constexpr(P):
                        s_ = ks * P + pp
                        g = g0 + i32(s_)
                        b = ring[6 * s_ : 6 * s_ + 4]
                        sgw = fx.Int32(ring[6 * s_ + 4])
                        suw = fx.Int32(ring[6 * s_ + 5])
                        g0s = (sgw >> i32(8 * (kh * 2))) & i32(0xFF)
                        g1s = (sgw >> i32(8 * (kh * 2 + 1))) & i32(0xFF)
                        u0s = (suw >> i32(8 * (kh * 2))) & i32(0xFF)
                        u1s = (suw >> i32(8 * (kh * 2 + 1))) & i32(0xFF)
                        o = pp * NA
                        for m in range_constexpr(MT):
                            row = i32(m * 16) + lane % i32(16)
                            af = a_frag(abuf, row, k, lane // i32(16))
                            sa = fx.Int32(
                                lds_ld(asbuf, row * i32(4) + lane // i32(16), T.i8, 1)
                            ) & i32(0xFF)
                            acc[o + m * 2 + 0] = mfma(
                                b[0], af, acc[o + m * 2 + 0], g0s, sa, A8
                            )
                            acc[o + m * 2 + 1] = mfma(
                                b[1], af, acc[o + m * 2 + 1], g1s, sa, A8
                            )
                            acc[o + 2 * MT + m * 2 + 0] = mfma(
                                b[2], af, acc[o + 2 * MT + m * 2 + 0], u0s, sa, A8
                            )
                            acc[o + 2 * MT + m * 2 + 1] = mfma(
                                b[3], af, acc[o + 2 * MT + m * 2 + 1], u1s, sa, A8
                            )
                        ring[6 * s_ : 6 * s_ + 6] = load_b(g + i32(nsk))
                        rocdl.sched_barrier(0)
                    if k == KCS - 1:
                        release_a_chunk(L, lane, buf)
            ag_stage_free(L, lane, a)
            for pp in range_constexpr(P):
                o = pp * NA
                act_quant_store(
                    L,
                    lane,
                    w,
                    acc[o : o + 2 * MT],
                    acc[o + 2 * MT : o + NA],
                    grp * i32(P) + i32(pp),
                )
            _ = yield ring
        _store_qbase(L, lane, w, qbase + (nnb // i32(P)) * i32(NCH))

    @traced
    def _store_qbase(L, lane, w, v):
        if lane == i32(0):
            lds_st(L, L_CTL + C_QBASE * 4 + w * i32(4), v)

    def unit_fields(L, a, u, ub):
        if const_expr(DYN):
            f = [
                lds_ld_i32(L, L_CTL + (i32(C_UL) + u * i32(4) + i32(k)) * i32(4))
                for k in range(4)
            ]
            return f[0], f[1], f[2], f[3]
        return _unit_fields(L, a, u, ub)

    @traced
    def _unit_fields(L, a, u, ub):
        f = [lds_ld_i32(L, L_CTL + i32((C_UNIT + 2 + k) * 4)) for k in range(4)]
        if u != ub:
            ubase = fx.Int64(a["units"]) + fx.Int64(u) * fx.Int64(16)
            f = [g_ld_i32(ubase + fx.Int64(4 * k)) for k in range(4)]
        return f[0], f[1], f[2], f[3]

    @traced
    def a_loader(L, tid, a):
        lane = tid % i32(64)
        rx = rsrc(a["ax"])
        rxs = rsrc(a["axs"])
        if const_expr(DYN):
            if lane == i32(0):
                spin_lds_ge(L, L_CTL + C_PLAN * 4, i32(1))
            rocdl.sched_barrier(0)
        ub = uni(lds_ld_acq(L, L_CTL + C_UNIT * 4))
        ue = uni(lds_ld_acq(L, L_CTL + (C_UNIT + 1) * 4))
        for u_ in range(ub, ue, i32(1)):
            u = i32(u_)
            if lane == i32(0):
                spin_lds_ge(L, L_CTL + C_USEQ * 4, u - ub + i32(1))
            _a_unit(L, lane, a, rx, rxs, u, ub)

    @traced
    def _a_unit(L, lane, a, rx, rxs, u, ub):
        if const_expr(True):
            R = uni(lds_ld_acq(L, L_CTL + C_UROWS * 4))
            nnb = unit_fields(L, a, u, ub)[2] // i32(128)
            for r0_ in range(i32(0), R, i32(RG)):
                r0 = i32(r0_)
                rows = fx.min(R - r0, i32(RG))
                arow = []
                for j in range_constexpr(NA_ROWOPS):
                    # a 1 KB op fills 1024 / ACB rows; the 16 B piece a lane
                    # writes holds the column a_frag reads there
                    if const_expr(A8):
                        row = i32(j * 4) + lane // i32(16)
                        col = (lane % i32(16)) ^ (row & i32(7))
                    else:
                        row = i32(j * 8) + lane // i32(8)
                        col = (lane % i32(8)) ^ (row & i32(7))
                    rr = (row < rows).select(row, i32(0))
                    arow.append(
                        (lds_ld_i32(L, L_RIX + (r0 + rr) * i32(4)) // i32(TOPK))
                        * i32(XB)
                        + col * i32(16)
                    )
                ascl = []
                for j in range_constexpr(NSC_BLK):
                    row = i32(j * 64) + lane
                    rr = (row < rows).select(row, i32(0))
                    ascl.append(
                        (lds_ld_i32(L, L_RIX + (r0 + rr) * i32(4)) // i32(TOPK))
                        * i32(H // 32)
                    )
                nq = (nnb // g1_group(nnb)) * i32(NCH)
                if lane == i32(0):
                    lds_st(L, L_CTL + C_ARDY * 4, i32(0))
                for cidx_ in range(i32(0), nq, i32(1)):
                    cidx = i32(cidx_)
                    q = lds_ld_i32(L, L_CTL + C_LQ * 4)
                    b = q % i32(NAB)
                    tf = tnow()  # [tl]
                    if (lane == i32(0)) & (q >= i32(NAB)):
                        spin_lds_ge(
                            L,
                            L_CTL + C_AFREE * 4 + b * i32(4),
                            i32(NW) * (q // i32(NAB)),
                        )
                    rocdl.sched_barrier(0)
                    tl_acc(L, lane, C_TLC, tf)  # [tl]
                    cc = cidx % i32(NCH)
                    # (the ARDY slot is read only for the first NCH chunks)
                    if cidx < i32(NCH):  # noqa: SIM102
                        if cidx >= uni(lds_ld_i32(L, L_CTL + C_ARDY * 4)):
                            ag_wait_chunk(L, lane, a, a["epoch"], cc)
                    abase = L + i32(L_A) + b * i32(RG * ACB)
                    for j in range_constexpr(NA_ROWOPS):
                        dma16(abase + i32(j * 1024), rx, arow[j], cc * i32(ACB))
                    for k in range_constexpr(KCS):
                        for j in range_constexpr(NSC_BLK):
                            dst = (
                                L
                                + i32(L_AS)
                                + b * i32(KCS * NSC_BLK * 64 * 4)
                                + i32((k * NSC_BLK + j) * 64 * 4)
                            )
                            # a scale line holds 16 K-chunks of a row, some maybe
                            # still in flight: bypass the caches
                            dma4(dst, rxs, ascl[j], cc * i32(8) + i32(k * 4), sys=True)
                    _loader_advance(L, lane, q)
                wait_vm(0)
                _loader_flush(L, lane)

    @traced
    def _loader_advance(L, lane, q):
        pub = lds_ld_i32(L, L_CTL + C_LPUB * 4)
        if q + i32(1) - pub >= i32(ADEPTH):
            tv = tnow()  # [tl]
            wait_vm(NA_L * (ADEPTH - 1))
            tl_acc(L, lane, C_TLD, tv)  # [tl]
            if lane == i32(0):
                lds_st_rel(
                    L, L_CTL + C_ASEQ * 4 + (pub % i32(NAB)) * i32(4), pub + i32(1)
                )
                lds_st(L, L_CTL + C_LPUB * 4, pub + i32(1))
        if lane == i32(0):
            lds_st(L, L_CTL + C_LQ * 4, q + i32(1))
        rocdl.sched_barrier(0)

    @traced
    def _loader_flush(L, lane):
        if lane == i32(0):
            pub = lds_ld_i32(L, L_CTL + C_LPUB * 4)
            q = lds_ld_i32(L, L_CTL + C_LQ * 4)
            for p_ in range(pub, q, i32(1)):
                p = i32(p_)
                lds_st_rel(L, L_CTL + C_ASEQ * 4 + (p % i32(NAB)) * i32(4), p + i32(1))
            lds_st(L, L_CTL + C_LPUB * 4, q)
        rocdl.sched_barrier(0)

    @traced
    def report_chunk(L, lane, w, cidx):
        if lane == i32(0):
            lds_st_rel(L, L_CTL + C_DONE * 4 + w * i32(4), cidx)

    @traced
    def maybe_report(L, lane, w, gi, signal, rlag_wait, lag=1):
        if (
            signal
            & (((gi + i32(1)) % i32(GPC)) == i32(0))
            & (gi + i32(1) > i32(lag * GPC))
        ):
            wait_vm(rlag_wait)
            report_chunk(L, lane, w, (gi // i32(GPC)) - i32(lag))

    def store_route_fp8(rs, accs, wt, rix, n0, q4, ok, oob, a):
        # accs: the 4 column tiles of a row; lane row q4 holds columns
        # 16 t + 4 q4 + i of tile t. Quantize each 32-column half (its rows'
        # amax across the 4 lane rows), then transpose the 4 x 4 dwords across
        # the lane rows: row q4 ends with the 16 columns of tile q4
        d, e8s = [], []
        for half in range_constexpr(2):
            f = [
                fx.Float32(fx.Vector(accs[2 * half + h])[i]) * wt
                for h in range(2)
                for i in range(4)
            ]
            am = amax(f)
            am = am.maximumf(am.shuffle_xor(i32(16), i32(64)))
            am = am.maximumf(am.shuffle_xor(i32(32), i32(64)))
            e8, qs = _e8m0_from_amax(am, max_norm=448.0)
            d += [fp8x4_pack(f[0:4], qs), fp8x4_pack(f[4:8], qs)]
            e8s.append(fx.Int32(e8) & i32(0xFF))
        d0, d1 = swap16(d[0], d[1])
        d2, d3 = swap16(d[2], d[3])
        d0, d2 = swap32(d0, d2)
        d1, d3 = swap32(d1, d3)
        lo, hi = [d0, d1], [d2, d3]
        col = n0 + q4 * i32(16)
        if const_expr(DLL):
            e8 = ((q4 >> i32(1)) == i32(0)).select(e8s[0], e8s[1])
            off = (rix * i32(H) + col) * i32(2)
            for h in range_constexpr(2):
                d = lo if h == 0 else hi
                pkt = ll_pkt(a, d[0], d[1], e8)
                bst(pkt, rs, ok.select(off + i32(16 * h), oob), 0, AUX_SC1)
            return
        ov = fx.Vector.from_elements(lo + hi, fx.Int32)
        bst(ov, rs, ok.select(rix * i32(H) + col, oob), 0, AUX_RT)
        sc = fx.Int16(e8s[0] | (e8s[1] << i32(8)))
        soff = a["ttot"] * i32(TOPK * H) + rix * i32(H // 32) + n0 // i32(32)
        bst(sc, rs, (ok & (q4 == i32(0))).select(soff, oob), 0, AUX_RT)

    def g2_slot_dma(a, lane, w, slot, e, gi, k):
        # one GEMM2 weight ring slot (K step k of group gi of expert e)
        n0 = (w + i32(NW) * gi) * i32(64)
        for t in range_constexpr(4):
            dma16(
                slot + i32(t * 1024),
                rsrc(a["w2"]),
                lane * i32(16),
                e * i32(H * (I // 2)) + (n0 + i32(t * 16)) * i32(I // 2) + k * i32(1024),
                nt=True,
            )
        for p in range_constexpr(2):
            rb = (e * i32(H) + n0 + i32(p * 32)) // i32(32)
            dma4(
                slot + i32(4096 + p * 256),
                rsrc(a["w2s"]),
                lane * i32(4),
                (rb * i32(CH2) + k // i32(2)) * i32(256),
            )

    @traced
    def gemm2(
        L,
        tid,
        a,
        expert,
        ks0,
        r0,
        rows,
        signal,
        NKS,
        pidx,
        gi_lo=None,
        gi_hi=None,
        colsig=None,
        pre=None,
        span=None,
        buf=None,
        progress=None,
        nxt=None,
    ):
        # groups this call covers when known at trace time (unrolled loop)
        span = G2 if gi_lo is None else span
        args = (L, tid, a, expert, ks0, r0, rows, signal, NKS, pidx)
        kw = {
            "gi_lo": gi_lo,
            "gi_hi": gi_hi,
            "colsig": colsig,
            "pre": pre,
            "span": span,
            "buf": buf,
            "progress": progress,
            "nxt": nxt,
        }
        if const_expr(VB and span is not None):
            if (lds_ld_i32(L, L_CTL + C_VBON * 4) != i32(0)) | VBS:
                _gemm2(*args, vb2=True, **kw)
            else:
                _gemm2(*args, **kw)
        elif const_expr(MTSKIP):
            # (row tiles past the unit's rows skipped)
            if uni(rows) <= i32(3 * 16):
                _gemm2(*args, mte=3, **kw)
            else:
                _gemm2(*args, **kw)
        else:
            _gemm2(*args, **kw)

    @traced
    def _gemm2(
        L,
        tid,
        a,
        expert,
        ks0,
        r0,
        rows,
        signal,
        NKS,
        pidx,
        gi_lo=None,
        gi_hi=None,
        colsig=None,
        pre=None,
        span=None,
        vb2=False,
        buf=None,
        progress=None,
        mte=None,
        nxt=None,
    ):
        VB2 = vb2
        MTE = MT if mte is None else mte
        assert not VB2 or MTE == MT
        if const_expr(buf is None):
            ib0, is0 = i32(L_INTER), i32(L_INTERS)
            rix0, wt0 = i32(L_RIX), i32(L_WT)
        else:
            # LB prefetch: buffer 0 (L_INTER, routes [0, 128)) or 1 (L_B1, [128, 256))
            b0 = buf == i32(0)
            ib0 = b0.select(i32(L_INTER), i32(L_B1))
            is0 = b0.select(i32(L_INTERS), i32(L_B1S))
            rix0 = i32(L_RIX) + buf * i32(128 * 4)
            wt0 = i32(L_WT) + buf * i32(128 * 4)
        NSK2 = _nsk2_for(NKS, G2)
        GPI = (NSK2 * NKS // math.gcd(NSK2, NKS)) // NKS
        assert G2 % GPI == 0
        WAIT_B2 = (NSK2 - 1) * OPS
        LAG = max(1, -(-NSK2 // (GPC * NKS)))
        RLAG = GPC * NKS * OPS
        lane = tid % i32(64)
        w = uni(tid // i32(64))
        e = uni(expert)
        ks0 = uni(ks0)
        rw = rsrc(fx.Int64(a["w2"]) + fx.Int64(e) * fx.Int64(H * (I // 2)))
        rws = rsrc(a["w2s"])
        routes_bytes = route_region_bytes(a["ttot"])
        pidx = uni(pidx)
        dst = (pidx == i32(0)).select(
            fx.Int64(a["routes"]),
            fx.Int64(a["proutes"]) + fx.Int64(pidx - i32(1)) * fx.Int64(routes_bytes),
        )
        r_routes = rsrc(dst, routes_bytes)
        oob = routes_bytes
        ring = uni(L + i32(L_RING) + w * i32(nsk * SLOT))
        q4 = lane // i32(16)
        g_lo = i32(0) if gi_lo is None else uni(gi_lo)
        g_hi = i32(G2) if gi_hi is None else uni(gi_hi)

        # (LB prefetch: a unit's ring tail loads the next unit's first slots)
        XPF = nxt is not None and not VB2

        def issue_x(q, slot_idx):
            # past this unit's slots: the next unit's first ones when its
            # claim is in (decided once per wave, at the first such slot)
            tq = g_hi * i32(NKS)
            if q == tq:
                # (the claim's sequence first: C_G2V is published before it)
                vn = lds_ld_acq(L, L_CTL + C_G2VN * 4)
                ok = (vn == nxt + i32(2)) & (
                    lds_ld_i32(L, L_CTL + C_G2V * 4) < lds_ld_i32(L, L_CTL + C_NCH * 4) * i32(NCG)
                )
                if lane == i32(0):
                    lds_st(L, L_CTL + (C_PFW + w) * i32(4), ok.select(i32(1), i32(0)))
            pf = lds_ld_i32(L, L_CTL + (C_PFW + w) * i32(4)) == i32(1)
            nun_ = fx.max(lds_ld_i32(L, L_CTL + C_NCH * 4), i32(1))
            v = lds_ld_i32(L, L_CTL + C_G2V * 4)
            cc_ = v // nun_
            ent_ = lds_ld_i32(L, L_DCH + fx.min(v - cc_ * nun_, nun_ - i32(1)) * i32(4))
            ee = uni(pf.select(ent_ & i32(0xFFFF), e))
            qc = uni(pf.select(q - tq + cg_lo(cc_) * i32(GPC * NKS), tq - i32(1)))
            gi = qc // i32(NKS)
            g2_slot_dma(a, lane, w, ring + i32(slot_idx * SLOT), ee, gi, ks0 + qc - gi * i32(NKS))

        def issue(q, slot_idx):
            if const_expr(XPF):
                # (a runtime branch: no early return)
                if q >= g_hi * i32(NKS):
                    issue_x(q, slot_idx)
                else:
                    issue_n(q, slot_idx)
                return None
            return issue_n(q, slot_idx)

        def issue_n(q, slot_idx):
            qc = fx.min(q, g_hi * i32(NKS) - i32(1))
            gi = qc // i32(NKS)
            k = ks0 + qc - gi * i32(NKS)
            n0 = (w + i32(NW) * gi) * i32(64)
            if const_expr(VB2):
                return [
                    bld(
                        rw,
                        lane * i32(16),
                        (n0 + i32(t * 16)) * i32(I // 2) + k * i32(1024),
                        V4I,
                        AUX_B,
                    )
                    for t in range(4)
                ] + [
                    bld(
                        rws,
                        lane * i32(4),
                        (
                            ((e * i32(H) + n0 + i32(p * 32)) // i32(32)) * i32(CH2)
                            + k // i32(2)
                        )
                        * i32(256),
                        T.i32,
                        0,
                    )
                    for p in range(2)
                ]
            slot = ring + i32(slot_idx * SLOT)
            for t in range_constexpr(4):
                dma16(
                    slot + i32(t * 1024),
                    rw,
                    lane * i32(16),
                    (n0 + i32(t * 16)) * i32(I // 2) + k * i32(1024),
                    nt=True,
                )
            for p in range_constexpr(2):
                rb = (e * i32(H) + n0 + i32(p * 32)) // i32(32)
                dma4(
                    slot + i32(4096 + p * 256),
                    rws,
                    lane * i32(4),
                    (rb * i32(CH2) + k // i32(2)) * i32(256),
                )

        rr = []
        if const_expr(XPF):
            # (the last unit's tail may have loaded this one's first slots)
            if lds_ld_i32(L, L_CTL + (C_PFW + w) * i32(4)) == i32(0):
                for qq in range_constexpr(NSK2):
                    issue(g_lo * i32(NKS) + i32(qq), qq)
            if lane == i32(0):
                lds_st(L, L_CTL + (C_PFW + w) * i32(4), i32(0))
        else:
            for qq in range_constexpr(NSK2):
                rr += issue(g_lo * i32(NKS) + i32(qq), qq) or []
        if const_expr(pre is not None):
            pre()
        zero = fx.Vector.filled(4, 0.0, fx.Float32)

        # each lane's row (weight, route index): the same for every group
        row_meta = []
        for m in range_constexpr(MTE):
            R = i32(m * 16) + lane % i32(16)
            ok = R < rows
            Rc = ok.select(R, i32(0))
            row_meta.append(
                (
                    ok,
                    fx.Float32(lds_ld(L, wt0 + (r0 + Rc) * i32(4), T.f32)),
                    lds_ld_i32(L, rix0 + (r0 + Rc) * i32(4)),
                )
            )

        def groups(gi0, rr):
            rr = list(rr)
            for j in range_constexpr(GPI):
                gi = gi0 + i32(j)
                acc = [zero] * (4 * MTE)
                for k in range_constexpr(NKS):
                    sidx = (j * NKS + k) % NSK2
                    if const_expr(VB2):
                        b = rr[6 * sidx : 6 * sidx + 4]
                        s0 = fx.Int32(rr[6 * sidx + 4])
                        s1 = fx.Int32(rr[6 * sidx + 5])
                    else:
                        if const_expr(k < NSK2 and NKS >= NSK2):
                            # the slot went out before the last group's route
                            # row stores (2 per row tile; vmcnt counts them in
                            # order): those may stay in flight
                            if const_expr(j > 0):
                                wait_vm(WAIT_B2 + 2 * MTE)
                            elif gi > g_lo:
                                wait_vm(WAIT_B2 + 2 * MTE)
                            else:
                                wait_vm(WAIT_B2)
                        else:
                            wait_vm(WAIT_B2)
                        slot = ring + i32(sidx * SLOT)
                        b = [
                            lds_ld(slot, i32(t * 1024) + lane * i32(16), V4I, 16)
                            for t in range(4)
                        ]
                        s0 = lds_ld_i32(slot, i32(4096) + lane * i32(4))
                        s1 = lds_ld_i32(slot, i32(4096 + 256) + lane * i32(4))
                    sh0 = ((ks0 + i32(k)) & i32(1)) * i32(16)
                    sb = [
                        (s0 >> sh0) & i32(0xFF),
                        (s0 >> (sh0 + i32(8))) & i32(0xFF),
                        (s1 >> sh0) & i32(0xFF),
                        (s1 >> (sh0 + i32(8))) & i32(0xFF),
                    ]
                    af_l, sa_l = [], []
                    for m in range_constexpr(MTE):
                        row = i32(m * 16) + lane % i32(16)
                        ib = ib0 + row * i32(SI_STRIDE)
                        if const_expr(A8):
                            # E4M3 (f8f6f4 ABI): 16 B at q4 and 64 B further
                            kb = ib + i32(k * 128) + q4 * i32(16)
                            af_l.append(
                                cat8(
                                    lds_ld(L, kb, V4I, 16),
                                    lds_ld(L, kb + i32(64), V4I, 16),
                                )
                            )
                        else:
                            af_l.append(
                                lds_ld(L, ib + i32(k * 64) + q4 * i32(16), V4I, 16)
                            )
                        sa_l.append(
                            fx.Int32(
                                lds_ld(
                                    L,
                                    is0
                                    + row * i32(I // 32)
                                    + i32(k * 4)
                                    + q4,
                                    T.i8,
                                    1,
                                )
                            )
                            & i32(0xFF)
                        )
                    rocdl.sched_barrier(0)
                    rocdl.s_setprio(1)
                    for m in range_constexpr(MTE):
                        for t in range_constexpr(4):
                            acc[m * 4 + t] = mfma(
                                b[t], af_l[m], acc[m * 4 + t], sb[t], sa_l[m], A8
                            )
                    rocdl.s_setprio(0)
                    wait_lgkm0()
                    rocdl.sched_barrier(0)
                    nxt = issue(gi * i32(NKS) + i32(k + NSK2), sidx)
                    if const_expr(VB2):
                        rr[6 * sidx : 6 * sidx + 6] = nxt
                    rocdl.sched_barrier(0)
                n0 = (w + i32(NW) * gi) * i32(64)
                ccol = (q4 & i32(1)) * i32(16) + (q4 >> i32(1)) * i32(8)
                for m in range_constexpr(MTE):
                    ok, wt, rix = row_meta[m]
                    if const_expr(FP8R):
                        store_route_fp8(
                            r_routes,
                            acc[m * 4 : m * 4 + 4],
                            wt,
                            rix,
                            n0,
                            q4,
                            ok,
                            oob,
                            a,
                        )
                    else:
                        pk = []
                        for t in range_constexpr(4):
                            av = fx.Vector(acc[m * 4 + t])
                            pk.append(
                                [
                                    pack_bf16x2(
                                        fx.Float32(av[2 * h]) * wt,
                                        fx.Float32(av[2 * h + 1]) * wt,
                                    )
                                    for h in range(2)
                                ]
                            )
                        for half in range_constexpr(2):
                            ta, tb = 2 * half, 2 * half + 1
                            lo, hi = [], []
                            for h in range_constexpr(2):
                                x, y = swap16(pk[ta][h], pk[tb][h])
                                lo.append(x)
                                hi.append(y)
                            ov = fx.Vector.from_elements(lo + hi, fx.Int32)
                            col = n0 + i32(half * 32) + ccol
                            voff = (rix * i32(H) + col) * i32(2)
                            bst(ov, r_routes, ok.select(voff, oob), 0, AUX_RT)
                if const_expr(not DLL):
                    maybe_report(L, lane, w, gi, signal, LAG * RLAG, LAG)
                    if const_expr(colsig is not None):
                        col_report(L, a, lane, gi, g_lo, colsig, RLAG)
            return rr

        if const_expr(VB2):
            # unrolled: a loop back-edge makes the compiler drain the ring
            for gi0c in range_constexpr(0, span, GPI):
                rr = groups(g_lo + i32(gi0c), rr)
        else:
            for gi0_ in range(g_lo, g_hi, i32(GPI)):
                if const_expr(progress is not None):
                    # (the last iteration starts: e.g. the next unit may load)
                    if i32(gi0_) + i32(GPI) >= g_hi:
                        progress()
                groups(i32(gi0_), rr)
        wait_vm(0)
        if const_expr(not DLL):
            _final_report(L, lane, w, signal)
            if const_expr(colsig is not None):  # noqa: SIM102 (compile-time guard)
                if colsig:
                    col_done(L, a, lane, (g_hi - i32(1)) // i32(GPC))

    @traced
    def col_report(L, a, lane, gi, g_lo, colsig, rlag_wait):
        if (
            colsig
            & (((gi + i32(1)) % i32(GPC)) == i32(0))
            & (gi + i32(1) > g_lo + i32(GPC))
        ):
            wait_vm(rlag_wait)
            col_done(L, a, lane, (gi // i32(GPC)) - i32(1))
        rocdl.sched_barrier(0)

    @traced
    def col_done(L, a, lane, cidx):
        if lane == i32(0):
            off = L_CTL + (C_XC * 4) + (cidx % i32(C_XC_N)) * i32(4)
            if lds_atomic_add(L, off, i32(1)) == i32(NW - 1):
                lds_st(L, off, i32(0))
                _signal_col(a, a["epoch"], cidx, lds_ld_i32(L, L_CTL + C_COLJ * 4))

    @traced
    def _final_report(L, lane, w, signal):
        if signal:
            report_chunk(L, lane, w, i32(NCK - 1))

    @traced
    def gemm2_dispatch(L, tid, a, expert, ks0, icnt, r0, rows, signal):
        if const_expr(DYN):
            for q in DYN_PS:
                if icnt == i32(I // q):
                    n = KS2 // q
                    gemm2(L, tid, a, expert, ks0, r0, rows, signal, n, ks0 // i32(n))
        else:
            if icnt == i32(I):
                gemm2(L, tid, a, expert, ks0, r0, rows, signal, KS2, i32(0))
            else:
                if const_expr(NPC > 1):
                    pidx = ks0 * i32(NPC) // i32(KS2)
                    gemm2(L, tid, a, expert, ks0, r0, rows, signal, KS2 // NPC, pidx)

    @traced
    def _drop_stale(tid, a):
        if tid < i32(64):
            asm("buffer_inv sc0")
        bid = i32(gpu.block_id("x"))
        if (tid < i32(64)) & (bid < i32(N_XCD)) & (a["epoch"] == i32(1)):
            fx.memory_fence(syncscope="one-as", ordering=ACQ)
            if tid == i32(0):
                g_st_sys(ctrl_at(a, i32(CTRL_XF) + bid), a["epoch"])

    @traced
    def _stale_dropped(tid, a):
        if (tid == i32(0)) & (a["epoch"] == i32(1)):
            x = i32(gpu.block_id("x")) % i32(N_XCD)
            spin_sys_ge(ctrl_at(a, i32(CTRL_XF) + x), a["epoch"])

    def unit_tile(L, tid, a, expert, i0, icnt, r0, rows, sig):
        gemm1(L, tid, a, expert, i0, icnt // i32(128))
        tlx0(a, tid, 4)  # [tl]
        gemm2_dispatch(L, tid, a, expert, i0 // i32(128), icnt, r0, rows, sig)

    @traced
    def compute_units(L, tid, a):
        lane = tid % i32(64)
        w = tid // i32(64)
        if const_expr(MLL):
            meta_ll_wait(tid, a)
        elif const_expr(not RREP):
            ag_wait_meta(a, tid, a["epoch"])
        _stale_dropped(tid, a)
        # ag_rs: after the cbar, i.e. after every wave's routing wait
        if const_expr(RREP and not DLL):
            zero_masked(L, tid % i32(64), a, tid // i32(64), NW)
        cbar(L, tid)
        if const_expr(MLL):  # noqa: SIM102 (compile-time guard)
            if tid == i32(0):
                lds_st_rel(L, L_CTL + C_RLAND * 4, i32(1))
        if const_expr(not RREP and not DLL and not ZMA):
            zero_masked(L, tid % i32(64), a, tid // i32(64), NW)
        if const_expr(LB):
            lb_plan(L, tid, a)
        elif const_expr(DYN):
            dyn_plan(L, tid, a)
        ub = lds_ld_i32(L, L_CTL + C_UNIT * 4)
        ue = lds_ld_i32(L, L_CTL + (C_UNIT + 1) * 4)
        if ub == ue:
            report_chunk(L, lane, w, i32(NCK - 1))
        for u_ in range(ub, ue, i32(1)):
            u = i32(u_)
            expert, i0, icnt, kind = unit_fields(L, a, u, ub)
            _tl_unit(a, tid, u - ub, 6)  # [tl]
            t_c = tnow()  # [tl]
            if const_expr(LB):
                R = lb_routes(L, tid, a, expert)
            elif const_expr(CHUNK):
                cc = expert.shrui(i32(24))
                ce = expert.shrui(i32(16)) & i32(0xFF)
                expert = expert & i32(0xFFFF)
                R = gather_routes_chunk(
                    L, tid, a["ids"], a["tw"], a["ttot"], expert, cc, ce
                )
            else:
                R = gather_routes(L, tid, a["ids"], a["tw"], a["ttot"], expert)
            _tl_unit(a, tid, u - ub, 13)  # [tl2]
            t_r = tnow()  # [tl]
            if tid == i32(0):
                lds_st(L, L_CTL + C_UROWS * 4, R)
                lds_st_rel(L, L_CTL + C_USEQ * 4, u - ub + i32(1))
            if const_expr(XSPLIT):
                typ = kind & i32(0xFF)
                sig_unit = (kind & i32(UNIT_SIGNAL)) != i32(0)
                if sig_unit & ((typ != i32(UNIT_FULL)) | (R == i32(0))):
                    report_chunk(L, lane, w, i32(NCK - 1))
            else:
                if (u == ue - i32(1)) & (R == i32(0)):
                    report_chunk(L, lane, w, i32(NCK - 1))
            for r0_ in range(i32(0), R, i32(RG)):
                r0 = i32(r0_)
                rows = fx.min(R - r0, i32(RG))
                if const_expr(XSPLIT):
                    unit_tile_x(L, tid, a, expert, i0, icnt, kind, r0, rows, R)
                elif const_expr(LB):
                    lb_g1_tile(L, tid, a, expert, i0, icnt, rows)
                else:
                    sig = (u == ue - i32(1)) & (r0 + i32(RG) >= R)
                    if const_expr(DX):
                        unit_tile_dx(L, tid, a, expert, i0, icnt, kind, r0, rows, sig)
                    else:
                        unit_tile(L, tid, a, expert, i0, icnt, r0, rows, sig)
                cbar(L, tid)
            if const_expr(LB):
                lb_g1_done(L, tid, a, kind)
                tle(L, a, tid, [i32(1), kind, t_c, t_r, tnow(), R, lds_ld_i32(L, L_CTL + C_TLA * 4), lds_ld_i32(L, L_CTL + C_TLB * 4)])  # [tl]
                if const_expr(TLX):  # [tl]
                    if tid == i32(0):  # [tl]
                        lds_st(L, L_CTL + C_TLA * 4, i32(0))  # [tl]
                        lds_st(L, L_CTL + C_TLB * 4, i32(0))  # [tl]
            if const_expr(DX):
                _dx_flag(a, tid, kind)
            _tl_unit(a, tid, u - ub, 7)  # [tl]
            if const_expr(XSPLIT):
                _xq_flag(a, tid, i0, icnt, kind)
                if const_expr(not DLL):
                    _xcol_empty(a, tid, kind, i0, R)
        if const_expr(XSPLIT):
            # column slices units[col0 : col0 + ncol] are claimed at run time
            # by the CTAs that run out of work first. Bounded loop (a dynamic
            # while around barriers + atomics miscompiles): at most twice the
            # mean per CTA, so the CTAs together always take all of them; each
            # unit claims the next one, except in the last iteration (that
            # claim would never run).
            nblk = i32(gpu.grid_dim.x)
            cap = (a["ncol"] + nblk - i32(1)) // nblk * i32(2) + i32(2)
            col_claim(L, tid, a)
            for it_ in range(i32(0), cap, i32(1)):
                _col_unit(L, tid, a, i32(it_) < cap - i32(1))
        tlx0(a, tid, 1)  # [tl]
        if const_expr(LBPF):
            lb_g2_consume(L, tid, a)
        elif const_expr(LB):
            lb_g2_phase(L, tid, a)
        if const_expr(DX):  # noqa: SIM102 (compile-time guard)
            if lds_ld_i32(L, L_CTL + C_DXON * 4) != i32(0):
                dx_cols(L, tid, a)
        tlx0(a, tid, 5)  # [tl]
        if const_expr(DLL):
            _cdone(tid, L)

    @traced  # [tl]
    def _tl_unit(a, tid, du, k):  # [tl]
        if (tid == i32(0)) & (du < i32(2)):  # [tl]
            tlx(a, i32(k) + du * i32(2))  # [tl]
  # [tl]
    @traced
    def unit_tile_dx(L, tid, a, expert, i0, icnt, kind, r0, rows, sig):
        if (kind & i32(0xFF)) == i32(UNIT_G1X):
            gemm1(L, tid, a, expert, i0, icnt // i32(128))
            tlx0(a, tid, 4)  # [tl]
            xq_export(L, tid, a, i0, icnt, r0, rows)
        else:
            unit_tile(L, tid, a, expert, i0, icnt, r0, rows, sig)

    def _dx_flag_addr(a, j, s):
        return ctrl_at(a, i32(CTRL_XQ) + j * i32(XQ_P) + s)

    @traced
    def _dx_flag(a, tid, kind):
        if ((kind & i32(0xFF)) == i32(UNIT_G1X)) & (tid == i32(0)):
            j = kind.shrui(i32(XQ_SHIFT))
            g_st_sys(_dx_flag_addr(a, j, kind.shrui(i32(8)) & i32(0xFF)), a["epoch"])

    @traced
    def dx_cols(L, tid, a):
        # bounded like the static column claims: at most twice the mean per CTA
        P = lds_ld_i32(L, L_CTL + C_DYNP * 4)
        ncol = lds_ld_i32(L, L_CTL + C_NACT * 4) * P * i32(DX_NG)
        nblk = i32(gpu.grid_dim.x)
        cap = (ncol + nblk - i32(1)) // nblk * i32(2) + i32(2)
        col_claim(L, tid, a)
        for it_ in range(i32(0), cap, i32(1)):
            _dx_col(L, tid, a, ncol, P, i32(it_) < cap - i32(1))

    @traced
    def _dx_col(L, tid, a, ncol, P, more):
        c = lds_ld_i32(L, L_CTL + C_CLAIM * 4)
        if c < ncol:
            u = c // i32(DX_NG)
            g = c - u * i32(DX_NG)
            j = u // P
            s = u - j * P
            expert = lds_ld_i32(L, L_DYN + (i32(2 * NBW) + j) * i32(4))
            if tid == i32(0):
                spin_sys_ge(_dx_flag_addr(a, j, s), a["epoch"], a)
            cbar(L, tid)
            R = gather_routes(L, tid, a["ids"], a["tw"], a["ttot"], expert)
            for r0_ in range(i32(0), R, i32(RG)):
                r0 = i32(r0_)
                _dx_gemm2(L, tid, a, expert, P, s, g, r0, fx.min(R - r0, i32(RG)))
                cbar(L, tid)
            if more:
                col_claim(L, tid, a)

    @traced
    def _dx_gemm2(L, tid, a, expert, P, s, g, r0, rows):
        for q in DYN_PS:
            if P == i32(q):
                icnt = I // q
                i0 = s * i32(icnt)
                gemm2(
                    L,
                    tid,
                    a,
                    expert,
                    s * i32(KS2 // q),
                    r0,
                    rows,
                    fx.Boolean(False),
                    KS2 // q,
                    s,
                    g * i32(DX_CG),
                    fx.min((g + i32(1)) * i32(DX_CG), i32(G2)),
                    pre=functools.partial(xq_import_dx, L, tid, a, i0, icnt, r0, rows),
                    span=DX_CG if DX_CG * DX_NG == G2 else None,
                )

    @traced
    def xq_import_dx(L, tid, a, i0, icnt, r0, rows):
        ag_stage_free(L, tid % i32(64), a)
        cbar(L, tid)
        u16 = icnt // 32
        rx, rxs = xg_rs(a)
        for q_ in range(tid, rows * i32(u16), i32(NT)):
            q = i32(q_)
            row = q // i32(u16)
            c = q - row * i32(u16)
            rix = lds_ld_i32(L, L_RIX + (r0 + row) * i32(4))
            v = bld(rx, rix * i32(I // 2) + i0 // i32(2) + c * i32(16), 0, V4I, AUX_SYS)
            lds_st(L, i32(L_INTER) + row * i32(SI_STRIDE) + c * i32(16), v, 16)
        nsd = icnt // 128
        for q_ in range(tid, rows * i32(nsd), i32(NT)):
            q = i32(q_)
            row = q // i32(nsd)
            c = q - row * i32(nsd)
            rix = lds_ld_i32(L, L_RIX + (r0 + row) * i32(4))
            sv = fx.Int32(
                bld(
                    rxs,
                    rix * i32(I // 32) + i0 // i32(32) + c * i32(4),
                    0,
                    T.i32,
                    AUX_SYS,
                )
            )
            lds_st(L, i32(L_INTERS) + row * i32(I // 32) + c * i32(4), sv)
        cbar(L, tid)

    @traced
    def _cdone(tid, L):
        if tid == i32(0):
            lds_st_rel(L, L_CTL + C_CDONE * 4, i32(1))

    @traced
    def dyn_plan(L, tid, a):
        n = a["ttot"] * i32(TOPK)
        rid = rsrc(a["ids"], n * i32(16 if MLL else 4))
        for idx_ in range(tid, n, i32(NT)):
            idx = i32(idx_)
            e = fx.Int32(bld(rid, idx * i32(16 if MLL else 4), 0, T.i32, 0))
            if masked(e) == fx.Boolean(False):
                lds_atomic_or(
                    L, L_DYN + (e >> i32(5)) * i32(4), i32(1) << (e & i32(31))
                )
                if const_expr(CHUNK):
                    lds_atomic_add(L, L_DCNT + e * i32(4), 1)
        cbar(L, tid)
        tlx0(a, tid, 5)  # [tl]
        _dyn_prefix(L, tid)
        cbar(L, tid)
        for it in range_constexpr((NE + NT - 1) // NT):
            _dyn_list(L, tid + i32(it * NT))
        cbar(L, tid)
        tlx0(a, tid, 8)  # [tl]
        if const_expr(CHUNK):
            _dyn_chunks(L, tid)
            cbar(L, tid)
        tlx0(a, tid, 9)  # [tl]
        _dyn_units(L, tid)
        cbar(L, tid)
        tlx0(a, tid, 15)  # [tl]

    @traced
    def _dyn_chunks(L, tid):
        # chunk table in active-expert order, identical in every CTA (units are
        # taken from it by index); an entry is the unit's expert field
        # expert | chunks << 16 | chunk << 24. One thread per active expert;
        # the table offsets are a wave prefix sum (ballots over the bits of the
        # chunk count) plus the totals of the lower waves. NCH_MAX covers the
        # sum of ceil(rows / RCH) over the experts of a dynamic batch.
        nact = lds_ld_i32(L, L_CTL + C_NACT * 4)
        w = tid // i32(64)
        lane = tid % i32(64)
        base = i32(0)
        for rnd in range_constexpr((NE + NT - 1) // NT):
            j = tid + i32(rnd * NT)
            live = j < nact
            e = live.select(
                lds_ld_i32(L, L_DYN + (i32(2 * NBW) + fx.min(j, i32(NE - 1))) * i32(4)),
                i32(0),
            )
            r = lds_ld_i32(L, L_DCNT + e * i32(4))
            ce = fx.min(fx.max((r + i32(RCH - 1)) // i32(RCH), i32(1)), i32(255))
            ce = live.select(ce, i32(0))
            pre, tot = i32(0), i32(0)
            for b in range_constexpr(8):
                below, cnt = _wave_rank(((ce >> i32(b)) & i32(1)) == i32(1))
                pre = pre + (below << i32(b))
                tot = tot + (cnt << i32(b))
            if lane == i32(0):
                lds_st(L, L_WSUM + w * i32(4), tot)
            cbar(L, tid)
            off = base + pre
            total = i32(0)
            for v in range_constexpr(NW):
                s = lds_ld_i32(L, L_WSUM + i32(v * 4))
                off = off + (i32(v) < w).select(s, i32(0))
                total = total + s
            for c_ in range(i32(0), ce, i32(1)):
                c = i32(c_)
                lds_st(
                    L,
                    L_DCH + (off + c) * i32(4),
                    e | (ce << i32(16)) | (c << i32(24)),
                )
            base = base + total
            cbar(L, tid)
        if tid == i32(0):
            lds_st(L, L_CTL + C_NCH * 4, base)

    @traced
    def _dyn_prefix(L, tid):
        if tid == i32(0):
            acc = i32(0)
            for w in range_constexpr(NBW):
                lds_st(L, L_DYN + i32((NBW + w) * 4), acc)
                acc = acc + _ctpop(lds_ld_i32(L, L_DYN + i32(w * 4)))
            lds_st(L, L_CTL + C_NACT * 4, acc)

    @traced
    def _dyn_list(L, e):
        if e < i32(NE):
            wv = lds_ld_i32(L, L_DYN + (e >> i32(5)) * i32(4))
            if ((wv >> (e & i32(31))) & i32(1)) != i32(0):
                low = (i32(1) << (e & i32(31))) - i32(1)
                rank = lds_ld_i32(
                    L, L_DYN + (i32(NBW) + (e >> i32(5))) * i32(4)
                ) + _ctpop(wv & low)
                lds_st(L, L_DYN + (i32(2 * NBW) + rank) * i32(4), e)

    @traced
    def _dyn_units(L, tid):
        if tid == i32(0):
            nact = lds_ld_i32(L, L_CTL + C_NACT * 4)
            # units are (expert or row chunk) x inter piece
            nun = nact
            if const_expr(CHUNK):
                nun = lds_ld_i32(L, L_CTL + C_NCH * 4)
            C = i32(gpu.grid_dim.x)
            P = i32(DYN_PS[0])
            best = ((nun * i32(DYN_PS[0]) + C - i32(1)) // C) * i32(KS2 // DYN_PS[0])
            for q in DYN_PS[1:]:
                cost = ((nun * i32(q) + C - i32(1)) // C) * i32(KS2 // q)
                # a CTA's unit list holds at most UL_MAX units
                better = (cost < best) & (nun * i32(q) <= C * i32(UL_MAX))
                P = better.select(i32(q), P)
                best = better.select(cost, best)
            U = nun * P
            bid = i32(gpu.block_id("x"))
            # CTAs ranked XCD by XCD (bid % N_XCD is the XCD), so that when the
            # units do not fill the CTAs every XCD (its share of the memory
            # bandwidth) gets the same number of them
            bid = (C % i32(N_XCD) == i32(0)).select(
                (bid % i32(N_XCD)) * (C // i32(N_XCD)) + bid // i32(N_XCD), bid
            )
            u_lo = bid * U // C
            u_hi = (bid + i32(1)) * U // C
            icnt = i32(I) // P
            if const_expr(VB):
                lds_st(
                    L,
                    L_CTL + C_VBON * 4,
                    (U * i32(2) <= C * i32(3)).select(i32(1), i32(0)),
                )
            use_x = fx.Boolean(False)
            if const_expr(DX):
                # one full round and a tail of at most half a round: GEMM2 of
                # the units moves off their CTAs to the ones the tail leaves idle
                # (its intermediate slots are per expert: no row chunks)
                use_x = (
                    (U > C)
                    & ((U - C) * i32(2) <= C)
                    & (nact <= i32(XQ_MAX))
                    & (nun == nact)
                )
                lds_st(L, L_CTL + C_DXON * 4, use_x.select(i32(1), i32(0)))
            for u_ in range(u_lo, u_hi, i32(1)):
                u = i32(u_)
                j = u // P
                ul = L_CTL + (i32(C_UL) + (u - u_lo) * i32(4)) * i32(4)
                if const_expr(CHUNK):
                    lds_st(L, ul, lds_ld_i32(L, L_DCH + j * i32(4)))
                else:
                    lds_st(
                        L, ul, lds_ld_i32(L, L_DYN + (i32(2 * NBW) + j) * i32(4))
                    )
                lds_st(L, ul + i32(4), (u - j * P) * icnt)
                lds_st(L, ul + i32(8), icnt)
                kind = i32(UNIT_G1X) | ((u - j * P) << i32(8)) | (j << i32(XQ_SHIFT))
                lds_st(L, ul + i32(12), use_x.select(kind, i32(0)))
            lds_st(L, L_CTL + C_DYNP * 4, P)
            lds_st(L, L_CTL + C_UNIT * 4, i32(0))
            lds_st(L, L_CTL + (C_UNIT + 1) * 4, u_hi - u_lo)
            lds_st_rel(L, L_CTL + C_PLAN * 4, i32(1))

    def lb_g1c(a, j, bank):
        return ctrl_at(
            a, i32(CTRL_G1C) + (bank * i32(NCHLB_MAX) + j) * i32(G1C_STRIDE)
        )

    def lb_colc(a, c, bank):
        return ctrl_at(
            a, i32(CTRL_COLC) + (bank * i32(NCK_MAX) + c) * i32(LRDY_STRIDE)
        )

    def lb_lbr(a, bank, x):
        return ctrl_at(a, i32(CTRL_LBR) + (bank * i32(N_XCD) + x) * i32(LRDY_STRIDE))

    @traced
    def lb_lists_wait(a, bank):
        # every CTA's route list slice landed: each XCD's counter is complete
        nblk = i32(gpu.grid_dim.x)
        for x in range_constexpr(N_XCD):
            spin_sys_ge(
                lb_lbr(a, bank, i32(x)),
                (nblk - i32(x) + i32(N_XCD - 1)) // i32(N_XCD),
                a,
            )

    def lb_lists(a):
        base = fx.Int64(a["xg"]) + fx.Int64(XG_LIST)
        return rsrc(base), rsrc(base + fx.Int64(TMAX * TOPK * 4))

    def lb_bank(L):
        return lds_ld_i32(L, L_CTL + C_LBB * 4)

    @traced
    def lb_plan(L, tid, a):
        # every CTA counts all routes (and those ahead of its slice), builds
        # the same chunk table, then writes its slice of the expert-sorted
        # route list
        n = a["ttot"] * i32(TOPK)
        nblk = i32(gpu.grid_dim.x)
        per = (n + nblk - i32(1)) // nblk
        lo = fx.min(i32(gpu.block_id("x")) * per, n)
        hi = fx.min(lo + per, n)
        if tid == i32(0):
            lds_st(
                L, L_CTL + C_LBB * 4, g_ld_sys(ctrl_at(a, i32(CTRL_LBSEQ))) & i32(1)
            )
        rid = rsrc(a["ids"], n * i32(4))
        n4 = n // i32(4)
        # LB_B loads of the thread in flight at once (one at a time is ~1 us
        # each), as many rounds as the batch's routes need
        for r0_ in range(i32(0), n4, i32(NT * LB_B)):
            r0 = i32(r0_)
            vs = [
                fx.Vector(bld(rid, fx.min(r0 + tid + i32(it * NT), n4 - i32(1)) * i32(16), 0, V4I, 0))
                for it in range(LB_B)
            ]
            for it in range_constexpr(LB_B):
                q = r0 + tid + i32(it * NT)
                for j in range_constexpr(4):
                    e = fx.Int32(vs[it][j])
                    _lb_count(L, e, q < n4, q * i32(4) + i32(j) < lo)
        for idx_ in range(n4 * i32(4) + tid, n, i32(NT)):
            idx = i32(idx_)
            e = fx.Int32(bld(rid, idx * i32(4), 0, T.i32, 0))
            _lb_count(L, e, fx.Boolean(True), idx < lo)
        cbar(L, tid)
        tlx0(a, tid, 12)  # [tl]
        _lb_bitmap(L, tid)
        cbar(L, tid)
        _dyn_prefix(L, tid)
        cbar(L, tid)
        for it in range_constexpr((NE + NT - 1) // NT):
            _dyn_list(L, tid + i32(it * NT))
        cbar(L, tid)
        _lb_chunks(L, tid)
        tlx0(a, tid, 14)  # [tl]
        lb_scatter(L, tid, a, lo, hi)
        tlx0(a, tid, 11)  # [tl]
        _lb_units(L, tid)
        cbar(L, tid)

    @traced
    def _lb_count(L, e, live, ahead):
        if live & (masked(e) == fx.Boolean(False)):
            lds_atomic_add(L, L_DCNT + e * i32(4), 1)
            if ahead:
                lds_atomic_add(L, L_DPRE + e * i32(4), 1)

    @traced
    def _lb_bitmap(L, tid):
        # the active-expert bitmap from the counts (one ballot per 64 experts)
        lane = tid % i32(64)
        w = tid // i32(64)
        for rnd in range_constexpr((NE + NT - 1) // NT):
            e = tid + i32(rnd * NT)
            c = (e < i32(NE)).select(
                lds_ld_i32(L, L_DCNT + fx.min(e, i32(NE - 1)) * i32(4)), i32(0)
            )
            b = fx.Int64(rocdl.ballot(T.i64, _u(c > i32(0))))
            for h in range_constexpr(2):
                wi = i32(rnd * NT // 32 + h) + w * i32(2)
                if (lane == i32(0)) & (wi < i32(NBW)):
                    lds_st(
                        L,
                        L_DYN + wi * i32(4),
                        i32((b >> fx.Int64(32 * h)) & fx.Int64(0xFFFFFFFF)),
                    )

    @traced
    def lb_zero_next(tid, a, bank):
        # the counters of the next LB launch (the other bank: no one uses it
        # in this launch, and the last LB launch, which did, is over), spread
        # over the CTAs' tails (system-scope stores are slow one by one)
        nb = bank ^ i32(1)
        bid = i32(gpu.block_id("x"))
        nblk = i32(gpu.grid_dim.x)
        if tid < i32(64):
            for j_ in range(bid + tid * nblk, i32(NCHLB_MAX), nblk * i32(64)):
                g_st_sys(lb_g1c(a, i32(j_), nb), i32(0))
            if bid == i32(0):
                if tid < i32(NCK):
                    g_st_sys(lb_colc(a, tid, nb), i32(0))
                if (tid >= i32(NCK)) & (tid < i32(NCK + N_XCD)):
                    g_st_sys(lb_lbr(a, nb, tid - i32(NCK)), i32(0))

    @traced
    def _lb_chunks(L, tid):
        # _dyn_chunks, plus each active expert's offset in the route list
        nact = lds_ld_i32(L, L_CTL + C_NACT * 4)
        w = tid // i32(64)
        lane = tid % i32(64)
        base = i32(0)
        rbase = i32(0)
        for rnd in range_constexpr((NE + NT - 1) // NT):
            j = tid + i32(rnd * NT)
            live = j < nact
            e = live.select(
                lds_ld_i32(L, L_DYN + (i32(2 * NBW) + fx.min(j, i32(NE - 1))) * i32(4)),
                i32(0),
            )
            r = live.select(lds_ld_i32(L, L_DCNT + e * i32(4)), i32(0))
            ce = fx.min(fx.max((r + i32(RCH - 1)) // i32(RCH), i32(1)), i32(255))
            ce = live.select(ce, i32(0))
            pre, tot = i32(0), i32(0)
            for b in range_constexpr(8):
                below, cnt = _wave_rank(((ce >> i32(b)) & i32(1)) == i32(1))
                pre = pre + (below << i32(b))
                tot = tot + (cnt << i32(b))
            rpre, rtot = i32(0), i32(0)
            for b in range_constexpr(RBITS):
                below, cnt = _wave_rank(((r >> i32(b)) & i32(1)) == i32(1))
                rpre = rpre + (below << i32(b))
                rtot = rtot + (cnt << i32(b))
            if lane == i32(0):
                lds_st(L, L_WSUM + w * i32(4), tot)
                lds_st(L, L_WSUM + (w + i32(NW)) * i32(4), rtot)
            cbar(L, tid)
            off = base + pre
            roff = rbase + rpre
            total = i32(0)
            rtotal = i32(0)
            for v in range_constexpr(NW):
                s = lds_ld_i32(L, L_WSUM + i32(v * 4))
                rs_ = lds_ld_i32(L, L_WSUM + i32((NW + v) * 4))
                off = off + (i32(v) < w).select(s, i32(0))
                roff = roff + (i32(v) < w).select(rs_, i32(0))
                total = total + s
                rtotal = rtotal + rs_
            if live:
                lds_st(L, L_EOFF + e * i32(4), roff)
            for c_ in range(i32(0), ce, i32(1)):
                c = i32(c_)
                lds_st(
                    L,
                    L_DCH + (off + c) * i32(4),
                    e | (ce << i32(16)) | (c << i32(24)),
                )
            base = base + total
            rbase = rbase + rtotal
            cbar(L, tid)
        if tid == i32(0):
            lds_st(L, L_CTL + C_NCH * 4, base)
        cbar(L, tid)

    @traced
    def lb_scatter(L, tid, a, lo, hi):
        # a route's slot: its expert's list offset + that expert's routes ahead
        # of this slice + its rank inside the slice (order within an expert is
        # free: rows are independent)
        n = a["ttot"] * i32(TOPK)
        rid = rsrc(a["ids"], n * i32(4))
        rtw = rsrc(a["tw"], n * i32(4))
        rl, rw = lb_lists(a)
        for idx_ in range(lo + tid, hi, i32(NT)):
            idx = i32(idx_)
            e = fx.Int32(bld(rid, idx * i32(4), 0, T.i32, 0))
            if masked(e) == fx.Boolean(False):
                pos = lds_atomic_add(L, L_DPRE + e * i32(4), 1)
                slot = lds_ld_i32(L, L_EOFF + e * i32(4)) + pos
                wv = fx.Int32(bld(rtw, idx * i32(4), 0, T.i32, 0))
                bst(idx, rl, slot * i32(4), 0, AUX_SYS)
                bst(wv, rw, slot * i32(4), 0, AUX_SYS)
        # (also covers this CTA's zero_masked route rows)
        wait_vm(0)
        cbar(L, tid)
        if tid == i32(0):
            if const_expr(ZMA):
                # (every CTA's masked route rows are zeroed before any unit
                # runs: units wait for every CTA's slice)
                spin_lds_ge(L, L_CTL + C_ZDONE * 4, i32(1))
            g_add_agent(
                lb_lbr(a, lb_bank(L), i32(gpu.block_id("x")) % i32(N_XCD)), 1
            )

    @traced
    def _lb_units(L, tid):
        # GEMM1 units (row chunk j, inter piece s) = u = j * P + s, round robin
        # over the CTAs: the chunks complete in order, so GEMM2 can follow
        if tid == i32(0):
            nun = lds_ld_i32(L, L_CTL + C_NCH * 4)
            C = i32(gpu.grid_dim.x)
            P = i32(LB_PS[0])
            best = ((nun * i32(LB_PS[0]) + C - i32(1)) // C) * i32(KS2 // LB_PS[0])
            for q in LB_PS[1:]:
                cost = ((nun * i32(q) + C - i32(1)) // C) * i32(KS2 // q)
                better = (cost < best) & (nun * i32(q) <= C * i32(UL_MAX))
                P = better.select(i32(q), P)
                best = better.select(cost, best)
            U = nun * P
            bid = i32(gpu.block_id("x"))
            # CTAs ranked XCD by XCD (bid % N_XCD is the XCD): consecutive
            # units (the row chunks of one expert) share an XCD and its L2
            bid = (C % i32(N_XCD) == i32(0)).select(
                (bid % i32(N_XCD)) * (C // i32(N_XCD)) + bid // i32(N_XCD), bid
            )
            icnt = i32(I) // P
            mix = fx.Boolean(False)
            J2 = nun
            if const_expr(LBMIX):
                # a few row chunks past one round of 2-piece units: those
                # chunks in single-block pieces, 3 per CTA without a unit,
                # the rest one more on as many CTAs (a third longer there)
                X = i32(2) * nun - C
                K = (X + i32(1)) // i32(2)
                M = i32(2) * (nun - K)
                mix = (X > i32(0)) & (X <= C // i32(LBMIX_DIV)) & (i32(3) * X <= M)
                P = mix.select(i32(2), P)
                U = mix.select(i32(0), U)
                J2 = mix.select(nun - K, nun)
                if mix:
                    _lb_units_mix(L, bid, C, nun, K, M, X)
            lds_st(L, L_CTL + C_LBJ2 * 4, J2)
            for u_ in range(bid, U, C):
                u = i32(u_)
                j = u // P
                s = u - j * P
                ul = L_CTL + (i32(C_UL) + ((u - bid) // C) * i32(4)) * i32(4)
                lds_st(L, ul, lds_ld_i32(L, L_DCH + j * i32(4)))
                lds_st(L, ul + i32(4), s * icnt)
                lds_st(L, ul + i32(8), icnt)
                lds_st(
                    L,
                    ul + i32(12),
                    i32(UNIT_G1X) | (s << i32(8)) | (j << i32(XQ_SHIFT)),
                )
            mine = (bid < U).select((U - bid + C - i32(1)) // C, i32(0))
            if const_expr(LBMIX):
                mine = mix.select(lds_ld_i32(L, L_CTL + (C_UNIT + 1) * 4), mine)
            lds_st(L, L_CTL + C_DYNP * 4, P)
            lds_st(L, L_CTL + C_UNIT * 4, i32(0))
            lds_st(L, L_CTL + (C_UNIT + 1) * 4, mine)
            lds_st_rel(L, L_CTL + C_PLAN * 4, i32(1))

    def lb_pieces(L, j):
        # GEMM1 units of row chunk j (LBMIX: KS2 for the chunks past C_LBJ2)
        P = lds_ld_i32(L, L_CTL + C_DYNP * 4)
        if const_expr(LBMIX):
            P = (j >= lds_ld_i32(L, L_CTL + C_LBJ2 * 4)).select(i32(KS2), P)
        return P

    def _lb_ul(L, n, ent, i0, icnt, kind):
        ul = L_CTL + (i32(C_UL) + n * i32(4)) * i32(4)
        lds_st(L, ul, ent)
        lds_st(L, ul + i32(4), i0)
        lds_st(L, ul + i32(8), icnt)
        lds_st(L, ul + i32(12), kind)

    @traced
    def _lb_units_mix(L, r, C, nun, K, M, X):
        # (thread 0) rank r: 2-piece unit r of the first nun - K chunks, or
        # (r >= M) 3 single-block pieces; plus (r < 3X) one more piece
        J2 = nun - K
        if r < M:
            j = r // i32(2)
            s_ = r - j * i32(2)
            _lb_ul(L, i32(0), lds_ld_i32(L, L_DCH + j * i32(4)), s_ * i32(I // 2), i32(I // 2),
                   i32(UNIT_G1X) | (s_ << i32(8)) | (j << i32(XQ_SHIFT)))
            lds_st(L, L_CTL + (C_UNIT + 1) * 4, i32(1))
        else:
            for k in range_constexpr(3):
                f = (r - M) * i32(3) + i32(k)
                j = J2 + f // i32(KS2)
                s_ = f - (j - J2) * i32(KS2)
                _lb_ul(L, i32(k), lds_ld_i32(L, L_DCH + j * i32(4)), s_ * i32(128), i32(128),
                       i32(UNIT_G1X) | (s_ << i32(8)) | (j << i32(XQ_SHIFT)))
            lds_st(L, L_CTL + (C_UNIT + 1) * 4, i32(3))
        if r < i32(3) * X:
            f = (C - M) * i32(3) + r
            j = J2 + f // i32(KS2)
            s_ = f - (j - J2) * i32(KS2)
            nn = lds_ld_i32(L, L_CTL + (C_UNIT + 1) * 4)
            _lb_ul(L, nn, lds_ld_i32(L, L_DCH + j * i32(4)), s_ * i32(128), i32(128),
                   i32(UNIT_G1X) | (s_ << i32(8)) | (j << i32(XQ_SHIFT)))
            lds_st(L, L_CTL + (C_UNIT + 1) * 4, nn + i32(1))

    @traced
    def lb_routes(L, tid, a, ent):
        # the row chunk's routes (a slice of its expert's list) into L_RIX/L_WT
        e = ent & i32(0xFFFF)
        r0 = ent.shrui(i32(24)) * i32(RCH)
        R = fx.min(lds_ld_i32(L, L_DCNT + e * i32(4)) - r0, i32(RCH))
        base = lds_ld_i32(L, L_EOFF + e * i32(4)) + r0
        if (tid < i32(N_XCD)) & (lds_ld_i32(L, L_CTL + C_LBN * 4) == i32(0)):
            # one lane per XCD counter, in parallel (each poll is ~1 us)
            nblk = i32(gpu.grid_dim.x)
            spin_sys_ge(
                lb_lbr(a, lb_bank(L), tid),
                (nblk - tid + i32(N_XCD - 1)) // i32(N_XCD),
                a,
            )
        cbar(L, tid)
        if (tid == i32(0)) & (lds_ld_i32(L, L_CTL + C_LBN * 4) == i32(0)):
            lds_st(L, L_CTL + C_LBN * 4, i32(1))
            tlx(a, i32(4))  # [tl]
        cbar(L, tid)
        rl, rw = lb_lists(a)
        if tid < R:
            lds_st(
                L,
                L_RIX + tid * i32(4),
                fx.Int32(bld(rl, (base + tid) * i32(4), 0, T.i32, AUX_SYS)),
            )
            lds_st(
                L,
                L_WT + tid * i32(4),
                fx.Int32(bld(rw, (base + tid) * i32(4), 0, T.i32, AUX_SYS)),
            )
        if tid == i32(0):
            lds_st(L, L_CTL + C_GEXP * 4, i32(0))
        cbar(L, tid)
        return R

    def lb_g1_tile(L, tid, a, ent, i0, icnt, rows):
        gemm1(L, tid, a, ent & i32(0xFFFF), i0, icnt // i32(128), rows)
        xq_export(L, tid, a, i0, icnt, i32(0), rows)

    @traced
    def lb_g1_done(L, tid, a, kind):
        if tid == i32(0):
            # (the intermediate went out system scope, every store drained)
            g_add_agent(lb_g1c(a, kind.shrui(i32(XQ_SHIFT)), lb_bank(L)), 1)

    @traced
    def _lb_g1_fin(L, tid):
        # no more GEMM1 units for this CTA: its A loader may stop
        if tid == i32(0):
            lds_st_rel(L, L_CTL + C_G1FIN * 4, i32(1))

    def cg_lo(cc):
        # first column chunk of group cc (CGB lookup)
        v = i32(0)
        for k in range_constexpr(1, NCG + 1):
            v = (cc >= i32(k)).select(i32(CGB[k]), v)
        return v

    def lb_g2_cap(U2, nblk):
        return (U2 + nblk - i32(1)) // nblk * i32(4) + i32(4)

    @traced
    def lb_g2_consume(L, tid, a):
        # LBPF: GEMM2 units as the A loader wave prefetches them (lb_g2_prefetch)
        nun = lds_ld_i32(L, L_CTL + C_NCH * 4)
        nblk = i32(gpu.grid_dim.x)
        if (nun == i32(0)) & (tid == i32(0)) & (i32(gpu.block_id("x")) == i32(0)):
            # nothing routed: every route row is a zeroed masked one
            lb_lists_wait(a, lb_bank(L))
            for c in range_constexpr(NCK):
                g_st_sys(ctrl_at(a, i32(CTRL_LRDY) + i32(c * LRDY_STRIDE)), a["epoch"])
        # the A ring and L_INTER are free: the prefetcher may fill them
        _lb_g1_fin(L, tid)
        cap = lb_g2_cap(nun * i32(NCG), nblk)
        for it_ in range(i32(0), cap, i32(1)):
            _lb_g2_take(L, tid, a, nun, i32(it_))

    @traced
    def spin_g2(L, target):
        cur = lds_ld_acq(L, L_CTL + C_G2RDY * 4)
        end = lds_ld_acq(L, L_CTL + C_G2END * 4)
        while (cur < target) & (end == i32(0)):
            rocdl.s_sleep(0)
            cur = lds_ld_acq(L, L_CTL + C_G2RDY * 4)
            end = lds_ld_acq(L, L_CTL + C_G2END * 4)

    @traced
    def _lb_g2_take(L, tid, a, nun, n):
        if tid == i32(0):
            spin_g2(L, n + i32(1))
        cbar(L, tid)
        if lds_ld_acq(L, L_CTL + C_G2RDY * 4) > n:
            t_c = tnow()  # [tl]
            b = n % i32(2)
            u = L_CTL + (i32(C_G2U) + b * i32(4)) * i32(4)
            ent = lds_ld_i32(L, u)
            cc = lds_ld_i32(L, u + i32(4))
            R = lds_ld_i32(L, u + i32(8))
            gemm2(
                L,
                tid,
                a,
                ent & i32(0xFFFF),
                i32(0),
                i32(0),
                R,
                fx.Boolean(False),
                KS2,
                i32(0),
                cg_lo(cc) * i32(GPC),
                cg_lo(cc + i32(1)) * i32(GPC),
                span=LBQ * GPC,
                buf=b,
                progress=functools.partial(_g2_last, L, tid, n),
                nxt=n,
            )
            cbar(L, tid)
            tle(L, a, tid, [i32(2), n, t_c, t_c, t_c, t_c, tnow(), R])  # [tl]
            if tid == i32(0):
                lds_st_rel(L, L_CTL + C_G2FREE * 4, n + i32(1))
                if const_expr(ZMA):
                    # (this CTA's masked route rows are zeroed)
                    spin_lds_ge(L, L_CTL + C_ZDONE * 4, i32(1))
                if g_add_agent(lb_colc(a, cc, lb_bank(L)), 1) == nun - i32(1):
                    tlc(a, cc)  # [tl]
                    for q_ in range(cg_lo(cc), cg_lo(cc + i32(1)), i32(1)):
                        g_st_sys(
                            ctrl_at(a, i32(CTRL_LRDY) + i32(q_) * i32(LRDY_STRIDE)),
                            a["epoch"],
                        )

    @traced
    def _g2_last(L, tid, n):
        if tid == i32(0):
            lds_st_rel(L, L_CTL + C_G2LAST * 4, n + i32(1))

    @traced
    def lb_g2_prefetch(L, lane, a):
        # LBPF (the A loader wave, once the CTA's GEMM1 units are done): claim
        # GEMM2 units and load each one's routes and intermediate into the
        # buffer the compute waves are not using, one unit ahead of them
        if lane == i32(0):
            spin_lds_ge(L, L_CTL + C_G1FIN * 4, i32(1))
        rocdl.sched_barrier(0)
        nun = lds_ld_i32(L, L_CTL + C_NCH * 4)
        U2 = nun * i32(NCG)
        bank = lb_bank(L)
        if (lane < i32(N_XCD)) & (lds_ld_i32(L, L_CTL + C_LBN * 4) == i32(0)):
            # (a CTA without GEMM1 units never waited for the route lists)
            nblk = i32(gpu.grid_dim.x)
            spin_sys_ge(
                lb_lbr(a, bank, lane), (nblk - lane + i32(N_XCD - 1)) // i32(N_XCD), a
            )
        rocdl.sched_barrier(0)
        cap = lb_g2_cap(U2, i32(gpu.grid_dim.x))
        for it_ in range(i32(0), cap, i32(1)):
            _lb_g2_fetch(L, lane, a, nun, U2, bank, i32(it_))
        if lane == i32(0):
            lds_st_rel(L, L_CTL + C_G2END * 4, i32(1))

    @traced
    def _lb_g2_fetch(L, lane, a, nun, U2, bank, n):
        if lds_ld_acq(L, L_CTL + C_G2END * 4) == i32(0):
            if (lane == i32(0)) & (n > i32(0)):
                # claim unit n once unit n - 1 runs its last groups: late
                # enough not to hold work another CTA could take, early enough
                # to hide the route and intermediate loads
                spin_lds_ge(L, L_CTL + C_G2LAST * 4, n)
            rocdl.sched_barrier(0)
            if lane == i32(0):
                lds_st(L, L_CTL + C_G2V * 4, g_add_agent(claim_addr(a, 0), 1))
                lds_st_rel(L, L_CTL + C_G2VN * 4, n + i32(1))
            rocdl.sched_barrier(0)
            v = uni(lds_ld_acq(L, L_CTL + C_G2V * 4))
            if v >= U2:
                if lane == i32(0):
                    lds_st_rel(L, L_CTL + C_G2END * 4, i32(1))
            else:
                _lb_g2_load(L, lane, a, nun, bank, n, v)

    @traced
    def _lb_g2_load(L, lane, a, nun, bank, n, v):
        b = n % i32(2)
        if lane == i32(0):
            # buffer b is free once the compute waves are done with unit n - 2
            spin_lds_ge(L, L_CTL + C_G2FREE * 4, n - i32(1))
        rocdl.sched_barrier(0)
        cc = v // nun
        j = v - cc * nun
        ent = lds_ld_i32(L, L_DCH + j * i32(4))
        e = ent & i32(0xFFFF)
        r0 = ent.shrui(i32(24)) * i32(RCH)
        R = fx.min(lds_ld_i32(L, L_DCNT + e * i32(4)) - r0, i32(RCH))
        base = lds_ld_i32(L, L_EOFF + e * i32(4)) + r0
        if lane == i32(0):
            spin_sys_ge(lb_g1c(a, j, bank), lb_pieces(L, j), a)
        rocdl.sched_barrier(0)
        rix0 = i32(L_RIX) + b * i32(128 * 4)
        wt0 = i32(L_WT) + b * i32(128 * 4)
        ib0 = (b == i32(0)).select(i32(L_INTER), i32(L_B1))
        is0 = (b == i32(0)).select(i32(L_INTERS), i32(L_B1S))
        rl, rw = lb_lists(a)
        # routes (lanes past R repeat the last: slots >= R are never read)
        rv = []
        for k in range_constexpr((RG + 63) // 64):
            r = lane + i32(k * 64)
            rc = fx.min(r, R - i32(1))
            rv.append(
                (
                    r,
                    bld(rl, (base + rc) * i32(4), 0, T.i32, AUX_SYS),
                    bld(rw, (base + rc) * i32(4), 0, T.i32, AUX_SYS),
                )
            )
        for r, iv, wv in rv:
            if r < i32(RG):
                lds_st(L, rix0 + r * i32(4), fx.Int32(iv))
                lds_st(L, wt0 + r * i32(4), fx.Int32(wv))
        wait_lgkm0()
        rx, rxs = xg_rs(a)
        u16 = I // 2 // 16
        nq = fx.max(R * i32(u16), i32(1))
        NQ = (RG * u16 + 63) // 64
        for b0 in range_constexpr(0, NQ, 8):
            dq = []
            for k in range_constexpr(b0, min(b0 + 8, NQ)):
                q = fx.min(lane + i32(k * 64), nq - i32(1))
                row = q // i32(u16)
                c = q - row * i32(u16)
                rix = lds_ld_i32(L, rix0 + row * i32(4))
                dq.append((row, c, bld(rx, rix * i32(I // 2) + c * i32(16), 0, V4I, AUX_SYS)))
            for row, c, dv in dq:
                lds_st(L, ib0 + row * i32(SI_STRIDE) + c * i32(16), dv, 16)
        ns = fx.max(R * i32(I // 128), i32(1))
        ds = []
        for k in range_constexpr((RG * (I // 128) + 63) // 64):
            q = fx.min(lane + i32(k * 64), ns - i32(1))
            row = q // i32(I // 128)
            c = q - row * i32(I // 128)
            rix = lds_ld_i32(L, rix0 + row * i32(4))
            ds.append((row, c, bld(rxs, rix * i32(I // 32) + c * i32(4), 0, T.i32, AUX_SYS)))
        for row, c, sv in ds:
            lds_st(L, is0 + row * i32(I // 32) + c * i32(4), fx.Int32(sv))
        wait_lgkm0()
        if lane == i32(0):
            u = L_CTL + (i32(C_G2U) + b * i32(4)) * i32(4)
            lds_st(L, u, ent)
            lds_st(L, u + i32(4), cc)
            lds_st(L, u + i32(8), R)
            lds_st(L, u + i32(12), j)
            lds_st_rel(L, L_CTL + C_G2RDY * 4, n + i32(1))
        rocdl.sched_barrier(0)

    @traced
    def lb_g2_phase(L, tid, a):
        # GEMM2 units v = g * nun + j (column group g = LBQ column chunks, of
        # row chunk j), claimed in order; bounded like the static column claims
        nun = lds_ld_i32(L, L_CTL + C_NCH * 4)
        U2 = nun * i32(NCG)
        nblk = i32(gpu.grid_dim.x)
        if (nun == i32(0)) & (tid == i32(0)) & (i32(gpu.block_id("x")) == i32(0)):
            # nothing routed: every route row is a zeroed masked one
            lb_lists_wait(a, lb_bank(L))
            for c in range_constexpr(NCK):
                g_st_sys(ctrl_at(a, i32(CTRL_LRDY) + i32(c * LRDY_STRIDE)), a["epoch"])
        cap = (U2 + nblk - i32(1)) // nblk * i32(2) + i32(2)
        col_claim(L, tid, a)
        for it_ in range(i32(0), cap, i32(1)):
            _lb_g2_unit(L, tid, a, nun, U2, i32(it_) < cap - i32(1))

    @traced
    def _lb_g2_unit(L, tid, a, nun, U2, more):
        v = lds_ld_i32(L, L_CTL + C_CLAIM * 4)
        if v < U2:
            cc = v // nun
            _lb_g2_run(L, tid, a, nun, cc, v - cc * nun, v)
        if more:
            col_claim(L, tid, a)

    @traced
    def _lb_g2_run(L, tid, a, nun, cc, j, v):
        if const_expr(True):
            t_c = tnow()  # [tl]
            ent = lds_ld_i32(L, L_DCH + j * i32(4))
            bank = lb_bank(L)
            if tid == i32(0):
                spin_sys_ge(lb_g1c(a, j, bank), lb_pieces(L, j), a)
            t_w = tnow()  # [tl]
            R = lb_routes(L, tid, a, ent)
            t_r = tnow()  # [tl]
            gemm2(
                L,
                tid,
                a,
                ent & i32(0xFFFF),
                i32(0),
                i32(0),
                R,
                fx.Boolean(False),
                KS2,
                i32(0),
                cg_lo(cc) * i32(GPC),
                cg_lo(cc + i32(1)) * i32(GPC),
                pre=functools.partial(lb_import, L, tid, a, R),
                span=LBQ * GPC,
            )
            cbar(L, tid)
            t_e = tnow()  # [tl]
            tle(L, a, tid, [i32(2), v, t_c, t_w, t_r, lds_ld_i32(L, L_CTL + C_TLI * 4), t_e, R])  # [tl]
            if tid == i32(0):
                if const_expr(ZMA):
                    spin_lds_ge(L, L_CTL + C_ZDONE * 4, i32(1))
                if g_add_agent(lb_colc(a, cc, bank), 1) == nun - i32(1):
                    tlc(a, cc)  # [tl]
                    for q_ in range(cg_lo(cc), cg_lo(cc + i32(1)), i32(1)):
                        g_st_sys(
                            ctrl_at(a, i32(CTRL_LRDY) + i32(q_) * i32(LRDY_STRIDE)),
                            a["epoch"],
                        )

    @traced
    def lb_import(L, tid, a, rows):
        # the full intermediate of the chunk's rows (every inter piece landed)
        ag_stage_free(L, tid % i32(64), a)
        cbar(L, tid)
        u16 = I // 2 // 16
        rx, rxs = xg_rs(a)
        # every load of the thread in flight at once (one at a time, each a
        # system-scope miss, is ~1 us apiece); lanes past the rows redo the
        # last piece (same value, same slot)
        nq = fx.max(rows * i32(u16), i32(1))
        ns = fx.max(rows * i32(I // 128), i32(1))
        dq, ds = [], []
        for k in range_constexpr((RG * u16 + NT - 1) // NT):
            q = fx.min(tid + i32(k * NT), nq - i32(1))
            row = q // i32(u16)
            c = q - row * i32(u16)
            rix = lds_ld_i32(L, L_RIX + row * i32(4))
            dq.append((row, c, bld(rx, rix * i32(I // 2) + c * i32(16), 0, V4I, AUX_SYS)))
        for k in range_constexpr((RG * (I // 128) + NT - 1) // NT):
            q = fx.min(tid + i32(k * NT), ns - i32(1))
            row = q // i32(I // 128)
            c = q - row * i32(I // 128)
            rix = lds_ld_i32(L, L_RIX + row * i32(4))
            ds.append(
                (row, c, bld(rxs, rix * i32(I // 32) + c * i32(4), 0, T.i32, AUX_SYS))
            )
        for row, c, v in dq:
            lds_st(L, i32(L_INTER) + row * i32(SI_STRIDE) + c * i32(16), v, 16)
        for row, c, sv in ds:
            lds_st(L, i32(L_INTERS) + row * i32(I // 32) + c * i32(4), fx.Int32(sv))
        cbar(L, tid)
        if const_expr(TLX):  # [tl]
            if tid == i32(0):  # [tl]
                lds_st(L, L_CTL + C_TLI * 4, tnow())  # [tl]

    @traced
    def lb_finish(tid, a, L):
        lb_zero_next(tid, a, lb_bank(L))
        # once every CTA read CTRL_LBSEQ (they all reported their list slice)
        if (tid == i32(0)) & (i32(gpu.block_id("x")) == i32(0)):
            lb_lists_wait(a, lb_bank(L))
            g_add_agent(ctrl_at(a, i32(CTRL_LBSEQ)), 1)

    @traced
    def unit_tile_x(L, tid, a, expert, i0, icnt, kind, r0, rows, R):
        k = kind & i32(0xFF)
        if k == i32(UNIT_G2COL):
            if tid == i32(0):
                lds_st(L, L_CTL + C_COLJ * 4, kind.shrui(i32(XQ_SHIFT)))
            gemm2(
                L,
                tid,
                a,
                expert,
                i32(0),
                r0,
                rows,
                fx.Boolean(False),
                KS2,
                i32(0),
                i0,
                i0 + (kind.shrui(i32(8)) & i32(0xFF)),
                r0 + i32(RG) >= R,
                pre=lambda: xq_import(L, tid, a, kind, r0, rows),
            )
        else:
            if k == i32(UNIT_G1X):
                gemm1(L, tid, a, expert, i0, icnt // i32(128))
                xq_export(L, tid, a, i0, icnt, r0, rows)
            else:
                sig = ((kind & i32(UNIT_SIGNAL)) != i32(0)) & (r0 + i32(RG) >= R)
                unit_tile(L, tid, a, expert, i0, icnt, r0, rows, sig)

    def xg_rs(a):
        rx = rsrc(a["xg"])
        rxs = rsrc(fx.Int64(a["xg"]) + fx.Int64(a["ttot"] * i32(TOPK * (I // 2))))
        return rx, rxs

    @traced
    def xq_export(L, tid, a, i0, icnt, r0, rows):
        u16 = icnt // i32(32)
        rx, rxs = xg_rs(a)
        for q_ in range(tid, rows * u16, i32(NT)):
            q = i32(q_)
            row = q // u16
            c = q - row * u16
            rix = lds_ld_i32(L, L_RIX + (r0 + row) * i32(4))
            v = lds_ld(L, i32(L_INTER) + row * i32(SI_STRIDE) + c * i32(16), V4I, 16)
            bst(v, rx, rix * i32(I // 2) + i0 // i32(2) + c * i32(16), 0, AUX_SYS)
        nsd = icnt // i32(128)
        for q_ in range(tid, rows * nsd, i32(NT)):
            q = i32(q_)
            row = q // nsd
            c = q - row * nsd
            rix = lds_ld_i32(L, L_RIX + (r0 + row) * i32(4))
            sv = lds_ld_i32(L, i32(L_INTERS) + row * i32(I // 32) + c * i32(4))
            bst(sv, rxs, rix * i32(I // 32) + i0 // i32(32) + c * i32(4), 0, AUX_SYS)
        wait_vm(0)

    @traced
    def _xq_flag(a, tid, i0, icnt, kind):
        s = tid - i0 // i32(128)
        if (
            ((kind & i32(0xFF)) == i32(UNIT_G1X))
            & (s >= i32(0))
            & (s < icnt // i32(128))
        ):
            j = kind.shrui(i32(XQ_SHIFT))
            g_st_sys(ctrl_at(a, i32(CTRL_XQ) + j * i32(XQ_P) + tid), a["epoch"])

    @traced
    def xq_import(L, tid, a, kind, r0, rows):
        if tid < i32(xsplit):
            j = kind.shrui(i32(XQ_SHIFT))
            spin_sys_ge(ctrl_at(a, i32(CTRL_XQ) + j * i32(XQ_P) + tid), a["epoch"], a)
        ag_stage_free(L, tid % i32(64), a)
        cbar(L, tid)
        u16 = I // 2 // 16
        rx, rxs = xg_rs(a)
        for q_ in range(tid, rows * i32(u16), i32(NT)):
            q = i32(q_)
            row = q // i32(u16)
            c = q - row * i32(u16)
            rix = lds_ld_i32(L, L_RIX + (r0 + row) * i32(4))
            v = bld(rx, rix * i32(I // 2) + c * i32(16), 0, V4I, AUX_SYS)
            lds_st(L, i32(L_INTER) + row * i32(SI_STRIDE) + c * i32(16), v, 16)
        for q_ in range(tid, rows * i32(I // 128), i32(NT)):
            q = i32(q_)
            row = q // i32(I // 128)
            c = q - row * i32(I // 128)
            rix = lds_ld_i32(L, L_RIX + (r0 + row) * i32(4))
            sv = fx.Int32(bld(rxs, rix * i32(I // 32) + c * i32(4), 0, T.i32, AUX_SYS))
            lds_st(L, i32(L_INTERS) + row * i32(I // 32) + c * i32(4), sv)
        cbar(L, tid)

    @traced
    def _xcol_empty(a, tid, kind, gi, R):
        if ((kind & i32(0xFF)) == i32(UNIT_G2COL)) & (R == i32(0)) & (tid == i32(0)):
            c0 = gi // i32(GPC)
            for c_ in range(
                c0, c0 + (kind.shrui(i32(8)) & i32(0xFF)) // i32(GPC), i32(1)
            ):
                _signal_col(a, a["epoch"], i32(c_), kind.shrui(i32(XQ_SHIFT)))

    @traced
    def _col_unit(L, tid, a, more):
        c = lds_ld_i32(L, L_CTL + C_CLAIM * 4)
        if c < a["ncol"]:
            u = a["col0"] + c
            expert, i0, icnt, kind = _unit_fields(L, a, u, i32(-1))
            R = gather_routes(L, tid, a["ids"], a["tw"], a["ttot"], expert)
            for r0_ in range(i32(0), R, i32(RG)):
                r0 = i32(r0_)
                rows = fx.min(R - r0, i32(RG))
                unit_tile_x(L, tid, a, expert, i0, icnt, kind, r0, rows, R)
                cbar(L, tid)
            if const_expr(not DLL):
                _xcol_empty(a, tid, kind, i0, R)
            if more:
                col_claim(L, tid, a)

    def claim_addr(a, bank_off):
        bank = (a["epoch"] + i32(bank_off)) & i32(1)
        return ctrl_at(a, i32(CTRL_CLM) + bank * i32(LRDY_STRIDE))

    @traced
    def col_claim(L, tid, a):
        cbar(L, tid)
        if tid == i32(0):
            v = g_add_agent(claim_addr(a, 0), 1)
            for w in range_constexpr(NW):
                lds_st(L, L_CTL + (C_CLAIM * 4) + i32(w * 4), v)
        cbar(L, tid)

    @traced
    def claim_reset(tid, a):
        if (tid == i32(0)) & (i32(gpu.block_id("x")) == i32(0)):
            g_st_sys(claim_addr(a, 1), i32(0))

    def vchunk(v):
        if const_expr(XL):
            late = v >= i32(NCK)
            return late.select(v - i32(NCK), v), late
        return v, fx.Boolean(False)

    def tok_late(L, t):
        w = lds_ld_i32(L, L_CLS + (t >> i32(5)) * i32(4))
        return ((w >> (t & i32(31))) & i32(1)) == i32(1)

    CLS_G = 8

    @traced
    def classify_tokens(L, lane, a):
        if const_expr(XL):
            if const_expr(not RREP):
                pend = _meta_pending(a, lane, a["epoch"])
                t0 = _now()
                while (pend != i32(0)) & _alive(t0):
                    rocdl.s_sleep(1)
                    pend = _meta_pending(a, lane, a["epoch"])
            for i_ in range(lane, (a["ttot"] + i32(31)) // i32(32), i32(64)):
                lds_st(L, L_CLS + i32(i_) * i32(4), i32(0))
            wait_lgkm0()
            n4 = a["ttot"] * i32(TOPK // 4) + (
                a["ttot"] * i32(TOPK % 4) + i32(3)
            ) // i32(4)
            rid = rsrc(a["ids"], a["ttot"] * i32(TOPK * 4))
            for q0_ in range(lane, n4, i32(64 * CLS_G)):
                q0 = i32(q0_)
                vs = [
                    fx.Vector(bld(rid, (q0 + i32(g * 64)) * i32(16), 0, V4I, 0))
                    for g in range(CLS_G)
                ]
                for g in range_constexpr(CLS_G):
                    q = q0 + i32(g * 64)
                    for j in range_constexpr(4):
                        i = q * i32(4) + i32(j)
                        hit = (q < n4) & (fx.Int32(vs[g][j]) >= a["piece_e0"])
                        _cls_set(L, i // i32(TOPK), hit)
            wait_lgkm0()
            if lane == i32(0):
                lds_st_rel(L, L_CTL + C_CLS * 4, i32(1))
            rocdl.sched_barrier(0)

    @traced
    def cls_wait(L, lane):
        if const_expr(XL):
            if lane == i32(0):
                spin_lds_ge(L, L_CTL + C_CLS * 4, i32(1))
            rocdl.sched_barrier(0)

    @traced
    def _cls_set(L, t, late):
        if late:
            lds_atomic_or(L, L_CLS + (t >> i32(5)) * i32(4), i32(1) << (t & i32(31)))

    def push_units(ttot):
        nblk = i32(gpu.grid_dim.x)
        return fx.min(i32(2) * nblk, (ttot + i32(TB - 1)) // i32(TB))

    def push_first_unit(ns, cidx):
        nblk = i32(gpu.grid_dim.x)
        rot = (cidx * ns) % nblk
        return (i32(gpu.block_id("x")) - rot + nblk) % nblk

    def token_dst(a, t):
        owner = t // a["m"]
        r_dst = rsrc(peer_sel(a, owner) + fx.Int64(a["off_part"]))
        return r_dst, a["rank"] * a["mmax"] + (t - owner * a["m"])

    @traced
    def push_chunk(L, tid, a, epoch, cidx):
        lane = tid % i32(64)
        ttot = a["ttot"]
        nblk = gpu.grid_dim.x
        ns = push_units(ttot)
        u0 = push_first_unit(ns, cidx)
        pc_, late = vchunk(cidx)
        c0 = pc_ * i32(CW)
        routes_bytes = route_region_bytes(ttot)
        r_routes = rsrc(a["routes"], routes_bytes)
        pr_bytes = i32(NPC - 1) * routes_bytes
        r_pr = rsrc(a["proutes"], pr_bytes)
        ids_base = fx.Int64(a["ids"])
        n = fx.max((ns - u0 + i32(nblk) - i32(1)) // i32(nblk), i32(0))
        for u_ in range(u0, ns, i32(nblk)):
            u = i32(u_)
            for t0_ in range(u, ttot, ns * i32(TB)):
                t0 = i32(t0_)
                d = []
                for b in range_constexpr(TB):
                    t = fx.min(t0 + i32(b) * ns, ttot - i32(1))
                    take = None
                    if const_expr(XL):
                        take = tok_late(L, t) == late
                    row = []
                    for k in range_constexpr(TOPK):
                        kv = []
                        for j in range_constexpr(VPL):
                            v = fx.min(lane + i32(j * 64), i32(CW // 8 - 1))
                            ridx = t * i32(TOPK) + i32(k)
                            kv.append(
                                route_load(
                                    r_routes, i32(0), ttot, ridx, c0 + v * i32(8), take
                                )
                            )
                        row.append(kv)
                    d.append(row)
                for b in range_constexpr(TB):
                    t_raw = t0 + i32(b) * ns
                    t = fx.min(t_raw, ttot - i32(1))
                    r_dst, prow = token_dst(a, t)
                    for j in range_constexpr(VPL):
                        v = lane + i32(j * 64)
                        acc = [fx.Float32(0.0)] * 8
                        for k in range_constexpr(TOPK):
                            acc = [x + y for x, y in zip(acc, route_decode(d[b][k][j]))]
                        live = (t_raw < ttot) & (v < i32(CW // 8))
                        if const_expr(XL):
                            live = live & (tok_late(L, t) == late)
                        if const_expr(NPC > 1):
                            eids = [
                                g_ld_i32(
                                    ids_base
                                    + fx.Int64((t * i32(TOPK) + i32(k)) * i32(4))
                                )
                                for k in range(TOPK)
                            ]
                            hit = eids[0] >= a["piece_e0"]
                            for k in range_constexpr(1, TOPK):
                                hit = hit | (eids[k] >= a["piece_e0"])
                            _push_split_token(
                                a,
                                r_dst,
                                prow,
                                c0,
                                r_pr,
                                v,
                                acc,
                                live,
                                hit,
                                eids,
                                t,
                                routes_bytes,
                            )
                        else:
                            _push_store(a, r_dst, prow, c0, v, acc, live)
        if const_expr(not LL):
            wait_vm(0)
            _push_done(a, lane, epoch, cidx, ns, n)

    @traced
    def _push_split_token(a, r_dst, prow, c0, r_pr, v, acc, live, hit, eids, t, rbytes):
        if hit:
            vc = fx.min(v, i32(CW // 8 - 1))
            tot = acc
            for k in range_constexpr(TOPK):
                split = eids[k] >= a["piece_e0"]
                for p in range_constexpr(1, NPC):
                    ld = route_load(
                        r_pr,
                        i32(p - 1) * rbytes,
                        a["ttot"],
                        t * i32(TOPK) + i32(k),
                        c0 + vc * i32(8),
                        split,
                    )
                    tot = [x + y for x, y in zip(tot, route_decode(ld))]
            _push_store(a, r_dst, prow, c0, v, tot, live)
        else:
            _push_store(a, r_dst, prow, c0, v, acc, live)

    def part_offs(a, prow, col):
        soff = a["tp"] * a["mmax"] * i32(H) + prow * i32(H // 32) + col // i32(32)
        return prow * i32(H) + col, soff

    @traced
    def _push_store(a, r_dst, prow, c0, v, acc, ok):
        col = c0 + v * i32(8)
        if const_expr(CB16):
            if ok:
                bst(pack_bf16x8(acc), r_dst, (prow * i32(H) + col) * i32(2), 0, AUX_SYS)
        else:
            _push_store_fp8(a, r_dst, prow, col, v, acc, ok)

    @traced
    def _push_store_fp8(a, r_dst, prow, col, v, acc, ok):
        am = amax(acc)
        am = am.maximumf(am.shuffle_xor(i32(1), i32(64)))
        am = am.maximumf(am.shuffle_xor(i32(2), i32(64)))
        e8, qs = _e8m0_from_amax(am, max_norm=448.0)
        d0, d1 = fp8x4_pack(acc[0:4], qs), fp8x4_pack(acc[4:8], qs)
        if const_expr(LL):
            pkt = ll_pkt(a, d0, d1, fx.Int32(e8) & i32(0xFF))
            if ok:
                bst(pkt, r_dst, (prow * i32(H) + col) * i32(2), 0, AUX_SYS)
        else:
            doff, soff = part_offs(a, prow, col)
            e = (fx.Int32(e8) & i32(0xFF)).bitcast(fx.Float32)
            sc = fx.Int32(e8) & i32(0xFF)
            for k in range_constexpr(1, 4):
                sc = sc | (
                    e.shuffle_xor(i32(4 * k), i32(64)).bitcast(fx.Int32) << i32(8 * k)
                )
            if ok:
                bst(
                    fx.Vector.from_elements([d0, d1], fx.Int32), r_dst, doff, 0, AUX_SYS
                )
                if (v & i32(15)) == i32(0):
                    bst(sc, r_dst, soff, 0, AUX_SYS)

    def sc_addr(a, cidx, slot):
        return ctrl_at(
            a, i32(CTRL_SC) + (cidx * i32(SC_LINES) + slot) * i32(LRDY_STRIDE)
        )

    @traced
    def _push_done(a, lane, epoch, cidx, ns, n):
        if (lane == i32(0)) & (n > i32(0)):
            _push_done_all(a, epoch, cidx, ns, n)

    @traced
    def _push_done_all(a, epoch, cidx, target, n):
        cnt_addr = sc_addr(a, cidx, i32(SC_PUSH))
        if g_add_agent(cnt_addr, n) + n == target:
            g_st_sys(cnt_addr, i32(0))
            tlc(a, i32(32) + cidx)  # [tl]
            fo = fx.Int64((i32(FLAG_RDY) + cidx * i32(MAX_TP) + a["rank"]) * i32(4))
            for p in range_constexpr(MAX_TP):
                if i32(p) < a["tp"]:
                    g_st_sys(
                        fx.Int64(a["peer"][p]) + fx.Int64(a["off_flag"]) + fo, epoch
                    )

    def fin_split(a):
        m = fx.max(a["m"], i32(1))
        nblk = i32(gpu.grid_dim.x)
        return (m < nblk).select(fx.min(i32(NV), nblk // m), i32(1))

    def fin_key(a):
        bid = i32(gpu.block_id("x"))
        return (fin_split(a) > i32(1)).select(bid % fx.max(a["m"], i32(1)), bid)

    def fin_owned(a):
        bid = i32(gpu.block_id("x"))
        m = fx.max(a["m"], i32(1))
        S = fin_split(a)
        sidx = bid // m
        own = i32(0)
        for c in range_constexpr(NV):
            own = own | ((i32(c) % S) == sidx).select(i32(1 << c), i32(0))
        return (bid < m * S).select(own, i32(0))

    def recv_pkts(a, r_recv, row, rstride, c0, vc):
        out = []
        for p in range_constexpr(MAX_TP):
            pc = fx.min(i32(p), a["tp"] - i32(1))
            off = ((pc * rstride + row) * i32(H) + c0 + vc * i32(8)) * i32(2)
            out.append(fx.Vector(bld(r_recv, off, 0, V4I, AUX_SYS)))
        return out

    @traced
    def final_chunk(L, tid, a, cidx):
        lane = tid % i32(64)
        nblk = gpu.grid_dim.x
        pc_, late = vchunk(cidx)
        c0 = pc_ * i32(CW)
        r_recv = own_rs(a, "off_part")
        step = (fin_split(a) > i32(1)).select(a["m"], i32(nblk))
        for row_ in range(fin_key(a), a["m"], step):
            row = i32(row_)
            _final_row(L, tid, a, r_recv, row, c0, late)
        if const_expr(AR):
            _yag_note(L, lane, cidx)

    @traced
    def _final_row(L, tid, a, r_recv, row, c0, late):
        lane = tid % i32(64)
        mine = fx.Boolean(True)
        if const_expr(XL):
            mine = tok_late(L, a["rank"] * a["m"] + row) == late
        if mine:
            for j in range_constexpr(VPL):
                v = lane + i32(j * 64)
                vc = fx.min(v, i32(CW // 8 - 1))
                if const_expr(LL):
                    _final_ll(tid, a, r_recv, row, c0, v, vc)
                    continue
                acc = [fx.Float32(0.0)] * 8
                if const_expr(CB16):
                    pk = recv_pkts(a, r_recv, row, a["mmax"], c0, vc)
                    for p in range_constexpr(MAX_TP):
                        acc = sum_live(acc, bf16x8_to_f32(pk[p]), i32(p) < a["tp"])
                else:
                    lds_ = []
                    for p in range_constexpr(MAX_TP):
                        pc = fx.min(i32(p), a["tp"] - i32(1))
                        doff, soff = part_offs(
                            a, pc * a["mmax"] + row, c0 + vc * i32(8)
                        )
                        lds_.append(
                            (
                                bld(r_recv, doff, 0, V2I, AUX_SYS),
                                bld(r_recv, soff, 0, T.i8, AUX_SYS),
                            )
                        )
                    for p in range_constexpr(MAX_TP):
                        acc = sum_live(acc, fp8x8_decode(lds_[p]), i32(p) < a["tp"])
                _y_store(a, row, c0, v, acc)

    @traced
    def _final_ll(tid, a, r_recv, row, c0, v, vc):
        lane = tid % i32(64)
        pk = recv_pkts(a, r_recv, row, a["mmax"], c0, vc)
        live = [(i32(p) < a["tp"]) & (v < i32(CW // 8)) for p in range(MAX_TP)]
        pend = ll_pending(a, lane, zip(pk, live))
        t0 = _now()
        while (pend != i32(0)) & _alive(t0):
            rocdl.s_sleep(1)
            pk = recv_pkts(a, r_recv, row, a["mmax"], c0, vc)
            pend = ll_pending(a, lane, zip(pk, live))
        _report_if(a, (pend != i32(0)) & (lane == i32(0)), ERR_COMM)
        acc = [fx.Float32(0.0)] * 8
        for p in range_constexpr(MAX_TP):
            acc = sum_live(acc, ll_vals(pk[p]), i32(p) < a["tp"])
        _y_store(a, row, c0, v, acc)

    @traced
    def _y_store(a, row, c0, v, acc):
        if const_expr(AG8):
            _y_store_ag8(a, row, c0, v, acc)
        else:
            _y_store_bf16(a, row, c0, v, pack_bf16x8(acc))

    @traced
    def _y_store_bf16(a, row, c0, v, o):
        if v < i32(CW // 8):
            if const_expr(AR):
                off = ((a["rank"] * a["m"] + row) * i32(H) + c0 + v * i32(8)) * i32(2)
                for p in range_constexpr(TPC):
                    bst(o, peer_rs(a, p, "off_yall"), off, 0, AUX_SYS)
            else:
                # (tn: written through: the tail reads it system scope)
                bst(
                    o,
                    rsrc(a["y"]),
                    (row * i32(H) + c0 + v * i32(8)) * i32(2),
                    0,
                    AUX_SYS if TN else 0,
                )

    def ag8_offs(a, grow, col):
        soff = a["tp"] * a["mmax"] * i32(H) + grow * i32(H // 32) + col // i32(32)
        return grow * i32(H) + col, soff

    @traced
    def _y_store_ag8(a, row, c0, v, acc):
        # the owner's final row: MXFP8 to the peers (decoded there by
        # ag8_convert) and decoded here, so every rank holds the same bf16
        col = c0 + v * i32(8)
        grow = a["rank"] * a["m"] + row
        am = amax(acc)
        am = am.maximumf(am.shuffle_xor(i32(1), i32(64)))
        am = am.maximumf(am.shuffle_xor(i32(2), i32(64)))
        e8, qs = _e8m0_from_amax(am, max_norm=448.0)
        d0, d1 = fp8x4_pack(acc[0:4], qs), fp8x4_pack(acc[4:8], qs)
        e8i = fx.Int32(e8) & i32(0xFF)
        e = e8i.bitcast(fx.Float32)
        sc = e8i
        for k in range_constexpr(1, 4):
            sc = sc | (
                e.shuffle_xor(i32(4 * k), i32(64)).bitcast(fx.Int32) << i32(8 * k)
            )
        f = fp8x4_unpack(d0, e8_scale(e8i)) + fp8x4_unpack(d1, e8_scale(e8i))
        if v < i32(CW // 8):
            bst(pack_bf16x8(f), rsrc(a["y"]), (grow * i32(H) + col) * i32(2), 0, 0)
            doff, soff = ag8_offs(a, grow, col)
            dv = fx.Vector.from_elements([d0, d1], fx.Int32)
            for p in range_constexpr(TPC):
                if i32(p) != a["rank"]:
                    rd = peer_rs(a, p, "off_yall")
                    bst(dv, rd, doff, 0, AUX_SYS)
                    if (v & i32(15)) == i32(0):
                        bst(sc, rd, soff, 0, AUX_SYS)

    @traced
    def ag8_convert(tid, a):
        # this CTA's output rows of every other owner (their YAG flags are in)
        bid = i32(gpu.block_id("x"))
        if bid < a["m"]:
            step = (fin_split(a) > i32(1)).select(a["m"], i32(gpu.grid_dim.x))
            ry = own_rs(a, "off_yall")
            yo = rsrc(a["y"])
            nq = i32(H // 8)
            for row_ in range(bid, a["m"], step):
                row = i32(row_)
                for q_ in range(tid, i32(TPC * (H // 8)), i32(NTT)):
                    q = i32(q_)
                    p = q // nq
                    col = (q - p * nq) * i32(8)
                    if p != a["rank"]:
                        grow = p * a["m"] + row
                        doff, soff = ag8_offs(a, grow, col)
                        ld = (
                            bld(ry, doff, 0, V2I, AUX_SYS),
                            bld(ry, soff, 0, T.i8, AUX_SYS),
                        )
                        bst(
                            pack_bf16x8(fp8x8_decode(ld)),
                            yo,
                            (grow * i32(H) + col) * i32(2),
                            0,
                            0,
                        )

    @traced
    def _yag_note(L, lane, cidx):
        if lane == i32(0):
            lds_atomic_or(L, L_CTL + C_YAGM * 4, i32(1) << cidx)

    @traced
    def yag_flush(L, tid, a):
        if tid < i32(64):
            mask = lds_ld_i32(L, L_CTL + C_YAGM * 4)
            for it in range_constexpr((NV * TPC + 63) // 64):
                e = i32(it * 64) + tid
                v = e // i32(TPC)
                p = e - v * i32(TPC)
                vc = fx.min(v, i32(NV - 1))
                idx = (
                    i32(FLAG_YAG)
                    + (vc * i32(MAX_TP) + a["rank"]) * i32(NCTA_MAX)
                    + fin_key(a)
                )
                ok = (v < i32(NV)) & (((mask >> vc) & i32(1)) == i32(1))
                _yag_post(a, p, idx, ok)

    @traced
    def _yag_post(a, p, idx, ok):
        if ok:
            fo = fx.Int64(a["off_flag"]) + fx.Int64(idx * i32(4))
            g_st_sys(peer_sel(a, p) + fo, a["epoch"])

    def _yag_pending(a, lane, epoch):
        rf = own_rs(a, "off_flag")
        bid = i32(gpu.block_id("x"))
        pend = i32(0)
        for it in range_constexpr((NV * MAX_TP + 63) // 64):
            e = i32(it * 64) + lane
            cidx = e // i32(MAX_TP)
            src = e - cidx * i32(MAX_TP)
            ok = (cidx < i32(NV)) & (src < a["tp"])
            idx = (
                i32(FLAG_YAG)
                + (fx.min(cidx, i32(NV - 1)) * i32(MAX_TP) + src) * i32(NCTA_MAX)
                + bid
            )
            f = fx.Int32(bld(rf, idx * i32(4), 0, T.i32, AUX_SYS))
            pend = fx.max(pend, (ok & (f < epoch)).select(i32(1), i32(0)))
        return wave_red(pend, lane, fx.max)

    @traced
    def yag_wait(tid, a, epoch):
        if (tid < i32(64)) & (i32(gpu.block_id("x")) < a["m"]):
            lane = tid % i32(64)
            pend = _yag_pending(a, lane, epoch)
            t0 = _now()
            while (pend != i32(0)) & _alive(t0):
                rocdl.s_sleep(1)
                pend = _yag_pending(a, lane, epoch)
            _report_if(a, (pend != i32(0)) & (lane == i32(0)), ERR_YAG)

    def comm_idle(a):
        bid = i32(gpu.block_id("x"))
        ns = push_units(a["ttot"])
        some = (ns * i32(NV) >= i32(gpu.grid_dim.x)) | (bid < ns * i32(NV))
        return (some.select(i32(0), i32(1)) == i32(1)) & (fin_owned(a) == i32(0))

    VMASK = (1 << NV) - 1

    def _lowest(v):
        return _ctpop((v & (i32(0) - v)) - i32(1))

    @traced
    def _claim_bit(L, rdy_off, bits_off, cnt_off, w):
        res = i32(-1)
        n = lds_ld_acq(L, L_CTL + cnt_off * 4)
        avail = (
            lds_ld_acq(L, L_CTL + rdy_off * 4)
            & (lds_ld_acq(L, L_CTL + bits_off * 4) ^ i32(-1))
            & i32(VMASK)
        )
        if (n < i32(NV)) & (avail != i32(0)):
            c = _lowest(avail)
            old = lds_atomic_or(L, L_CTL + bits_off * 4, i32(1) << c)
            if ((old >> c) & i32(1)) == i32(0):
                lds_atomic_add(L, L_CTL + cnt_off * 4, i32(1))
                res = c
        lds_st(L, L_CTL + (C_MBOX * 4) + w * i32(4), res)

    @traced
    def claim_push(L, lane, w):
        if lane == i32(0):
            _claim_bit(L, C_PRDY, C_PBITS, C_LRED, w)
        rocdl.sched_barrier(0)
        return uni(lds_ld_acq(L, L_CTL + (C_MBOX * 4) + w * i32(4)))

    @traced
    def claim_final(L, lane, w):
        if lane == i32(0):
            _claim_bit(L, C_FRDY, C_FBITS, C_PULL, w)
        rocdl.sched_barrier(0)
        return uni(lds_ld_acq(L, L_CTL + (C_MBOX * 4) + w * i32(4)))

    @traced
    def poll_ready(L, lane, a, epoch):
        c = fx.min(lane, i32(NV - 1))
        live = lane < i32(NV)
        v = g_ld_sys(ctrl_at(a, i32(CTRL_LRDY) + c * i32(LRDY_STRIDE)))
        pm = i32(
            fx.Int64(rocdl.ballot(T.i64, _u(live & (v >= epoch)))) & fx.Int64(VMASK)
        )
        fm = i32(
            fx.Int64(rocdl.ballot(T.i64, _u(live & _final_ready(a, c, epoch))))
            & fx.Int64(VMASK)
        )
        pm = uni(pm)
        fm = uni(fm)
        old_p = lds_ld_acq(L, L_CTL + C_PRDY * 4)
        old_f = lds_ld_acq(L, L_CTL + C_FRDY * 4)
        _poll_publish(L, lane, pm, fm)
        return (((pm & (old_p ^ i32(-1))) | (fm & (old_f ^ i32(-1)))) != i32(0)).select(
            i32(1), i32(0)
        )

    @traced
    def _poll_publish(L, lane, pm, fm):
        if lane == i32(0):
            lds_atomic_or(L, L_CTL + C_PRDY * 4, pm)
            lds_atomic_or(L, L_CTL + C_FRDY * 4, fm)

    def _comm_pending(L):
        full = i32(VMASK)
        push = (lds_ld_acq(L, L_CTL + C_LRED * 4) < i32(NV)) & (
            (lds_ld_acq(L, L_CTL + C_PRDY * 4) & full) != full
        )
        fin = (lds_ld_acq(L, L_CTL + C_PULL * 4) < i32(NV)) & (
            (
                (lds_ld_acq(L, L_CTL + C_FRDY * 4) | lds_ld_acq(L, L_CTL + C_FBITS * 4))
                & full
            )
            != full
        )
        return (lds_ld_acq(L, L_CTL + C_NSIG * 4) < i32(NCK)) | push | fin

    @traced
    def poll_loop(L, tid, a, epoch):
        t0 = _now()
        while _comm_pending(L) & _alive(t0):
            p1 = comm_signal(L, tid, a, epoch)
            p2 = poll_ready(L, tid % i32(64), a, epoch)
            if (p1 | p2) == i32(0):
                rocdl.s_sleep(POLL_SLEEP)
        _report_if(
            a,
            (_now() - t0 >= fx.Int64(DEADLINE)) & ((tid % i32(64)) == i32(0)),
            ERR_COMM,
        )

    def _final_ready(a, c, epoch):
        if const_expr(LL):
            r_recv = own_rs(a, "off_part")
            r1 = fx.Boolean(True)
            for p in range_constexpr(MAX_TP):
                pc = fx.min(i32(p), a["tp"] - i32(1))
                off = (
                    (pc * a["mmax"] + fin_key(a)) * i32(H) + c * i32(CW) + i32(CW - 8)
                ) * i32(2)
                t = fx.Int32(bld(r_recv, off + i32(4), 0, T.i32, AUX_SYS))
                r1 = r1 & ((i32(p) >= a["tp"]) | (t == epoch))
            return r1
        rf = own_rs(a, "off_flag")
        fo = (i32(FLAG_RDY) + c * i32(MAX_TP)) * i32(4)
        fl = fx.Vector(bld(rf, fo, 0, V4I, AUX_SYS))
        fh = fx.Vector(bld(rf, fo + i32(16), 0, V4I, AUX_SYS))
        r1 = fx.Boolean(True)
        for p in range_constexpr(MAX_TP):
            f = fx.Int32((fl if p < 4 else fh)[p % 4])
            r1 = r1 & ((i32(p) >= a["tp"]) | (f >= epoch))
        return r1

    @traced
    def push_chunk_dyn(L, tid, a, epoch, cidx):
        lane = tid % i32(64)
        ttot = a["ttot"]
        nblk = gpu.grid_dim.x
        ns = push_units(ttot)
        u0 = push_first_unit(ns, cidx)
        c0 = cidx * i32(CW)
        routes_bytes = route_region_bytes(ttot)
        r_routes = rsrc(a["routes"], routes_bytes)
        r_pr = rsrc(a["proutes"], i32(NPC - 1) * routes_bytes)
        P = uni(lds_ld_i32(L, L_CTL + C_DYNP * 4))
        n = fx.max((ns - u0 + i32(nblk) - i32(1)) // i32(nblk), i32(0))
        for u_ in range(u0, ns, i32(nblk)):
            u = i32(u_)
            for t_ in range(u, ttot, ns):
                t = i32(t_)
                for j in range_constexpr(VPL):
                    v = lane + i32(j * 64)
                    vc = fx.min(v, i32(CW // 8 - 1))
                    lds_ = []
                    for k in range_constexpr(TOPK):
                        ridx = t * i32(TOPK) + i32(k)
                        lds_.append(
                            route_load(r_routes, i32(0), ttot, ridx, c0 + vc * i32(8))
                        )
                        for p in range_constexpr(1, NPC):
                            lds_.append(
                                route_load(
                                    r_pr,
                                    i32(p - 1) * routes_bytes,
                                    ttot,
                                    ridx,
                                    c0 + vc * i32(8),
                                    i32(p) < P,
                                )
                            )
                    acc = [fx.Float32(0.0)] * 8
                    for ld in lds_:
                        acc = [x + y for x, y in zip(acc, route_decode(ld))]
                    r_dst, prow = token_dst(a, t)
                    _push_store(a, r_dst, prow, c0, v, acc, v < i32(CW // 8))
        if const_expr(not LL):
            wait_vm(0)
            _push_done(a, lane, epoch, cidx, ns, n)

    @traced
    def comm_work(L, tid, a, epoch):
        lane = tid % i32(64)
        w = tid // i32(64)
        cp = claim_final(L, lane, w)
        if cp >= i32(0):
            cls_wait(L, lane)
            final_chunk(L, tid, a, cp)
        cr = claim_push(L, lane, w)
        if cr >= i32(0):
            cls_wait(L, lane)
            if const_expr(DYN):
                push_chunk_dyn(L, tid, a, epoch, cr)
            else:
                push_chunk(L, tid, a, epoch, cr)
        return ((cr >= i32(0)) | (cp >= i32(0))).select(i32(1), i32(0))

    @traced
    def comm_signal(L, tid, a, epoch):
        lane = tid % i32(64)
        w = tid // i32(64)
        mn = i32(NCK)
        for wv in range_constexpr(NW):
            mn = fx.min(mn, lds_ld_acq(L, L_CTL + C_DONE * 4 + wv * 4))
        start = uni(lds_ld_acq(L, L_CTL + C_NSIG * 4))
        end = uni(fx.min(mn + i32(1), i32(NCK)))
        _signal_claim(L, lane, w, start, end)
        rocdl.sched_barrier(0)
        won = uni(lds_ld_acq(L, L_CTL + (C_MBOX * 4) + w * i32(4)))
        if won == i32(1):  # noqa: SIM102 (a runtime lane guard)
            if start + lane < end:
                _signal_one(a, epoch, start + lane)
        rocdl.sched_barrier(0)
        return (end > start).select(i32(1), i32(0))

    @traced
    def _signal_claim(L, lane, w, start, end):
        if lane == i32(0):
            won = i32(0)
            if end > start:
                got = lds_cas(L, L_CTL + C_NSIG * 4, start, end)
                won = (got == start).select(i32(1), i32(0))
            lds_st(L, L_CTL + (C_MBOX * 4) + w * i32(4), won)

    def _signal_one(a, epoch, cidx):
        nblk = i32(gpu.grid_dim.x)
        x = i32(gpu.block_id("x")) % i32(N_XCD)
        _signal_part(a, epoch, cidx, x, (nblk - x + i32(N_XCD - 1)) // i32(N_XCD))

    def _signal_col(a, epoch, cidx, j):
        if const_expr(XL):
            _signal_col_xl(a, epoch, cidx, j)
        else:
            sh = j % i32(N_XCD)
            _signal_part(
                a,
                epoch,
                cidx,
                i32(SC_SHARD) + sh,
                (i32(XCOL_PER_CHUNK) - sh + i32(N_XCD - 1)) // i32(N_XCD),
            )

    @traced
    def _signal_col_xl(a, epoch, cidx, j):
        if j < i32(xl_s0):
            sh = j % i32(N_XCD)
            _signal_part(
                a,
                epoch,
                cidx,
                i32(SC_SHARD) + sh,
                (i32(xl_s0) - sh + i32(N_XCD - 1)) // i32(N_XCD),
            )
        else:
            _signal_part(a, epoch, cidx, i32(SC_LATE), i32(XCOL_PER_CHUNK - xl_s0))

    @traced
    def _signal_part(a, epoch, cidx, slot, target):
        xa = sc_addr(a, cidx, slot)
        if g_add_agent(xa, 1) + i32(1) == target:
            g_st_sys(xa, i32(0))
            if const_expr(XL):
                _signal_early(a, epoch, cidx, slot)
            _signal_count(a, epoch, cidx)

    @traced
    def _signal_early(a, epoch, cidx, slot):
        if slot != i32(SC_LATE):
            ea = sc_addr(a, cidx, i32(SC_E))
            if g_add_agent(ea, 1) + i32(1) == fx.min(
                i32(gpu.grid_dim.x), i32(N_XCD)
            ) + i32(min(xl_s0, N_XCD)):
                g_st_sys(ea, i32(0))
                g_st_sys(ctrl_at(a, i32(CTRL_LRDY) + cidx * i32(LRDY_STRIDE)), epoch)
                tlc(a, cidx)  # [tl]

    @traced
    def _signal_count(a, epoch, cidx):
        cnt_addr = sc_addr(a, cidx, i32(SC_ALL))
        ncol_parts = (
            min(xl_s0, N_XCD) + (1 if XCOL_PER_CHUNK > xl_s0 else 0)
            if XL
            else min(XCOL_PER_CHUNK, N_XCD)
        )
        parts = fx.min(i32(gpu.grid_dim.x), i32(N_XCD)) + i32(ncol_parts)
        if g_add_agent(cnt_addr, 1) + i32(1) == parts:
            g_st_sys(cnt_addr, i32(0))
            vc = (cidx + i32(NCK)) if XL else cidx
            g_st_sys(ctrl_at(a, i32(CTRL_LRDY) + vc * i32(LRDY_STRIDE)), epoch)
            tlc(a, vc)  # [tl]

    @traced
    def signal_loop(L, tid, a, epoch):
        t0 = _now()
        while (lds_ld_acq(L, L_CTL + C_NSIG * 4) < i32(NCK)) & _alive(t0):
            if comm_signal(L, tid, a, epoch) == i32(0):
                rocdl.s_sleep(1)

    @traced
    def comm_wave(L, tid, a, epoch):
        t0 = _now()
        while (
            (lds_ld_acq(L, L_CTL + C_NSIG * 4) < i32(NCK))
            | (lds_ld_acq(L, L_CTL + C_LRED * 4) < i32(NV))
            | (lds_ld_acq(L, L_CTL + C_PULL * 4) < i32(NV))
        ) & _alive(t0):
            p1 = comm_signal(L, tid, a, epoch)
            if const_expr(not ARLL):
                p1 = p1 | poll_ready(L, tid % i32(64), a, epoch)
            p2 = comm_work(L, tid, a, epoch)
            if (p1 | p2) == i32(0):
                rocdl.s_sleep(POLL_SLEEP)
        _report_if(
            a,
            (_now() - t0 >= fx.Int64(DEADLINE)) & ((tid % i32(64)) == i32(0)),
            ERR_COMM,
        )

    @traced
    def comm_help(L, tid, a, epoch):
        t0 = _now()
        while (
            (lds_ld_acq(L, L_CTL + C_LRED * 4) < i32(NV))
            | (lds_ld_acq(L, L_CTL + C_PULL * 4) < i32(NV))
        ) & _alive(t0):
            if comm_work(L, tid, a, epoch) == i32(0):
                rocdl.s_sleep(POLL_SLEEP)

    AG_SROW = XB
    AG_SCB = AGR * AG_SROW
    assert AG_SCB + AGR * (H // 32) <= L_INTERS - L_INTER + RG * (I // 32)
    NCHA = H // 256
    NPRE = (NCHA + PRE_CH - 1) // PRE_CH
    assert NPRE <= NPRE_MAX
    NMETA_MAX = 32
    AG_WIN = max(1, 64 // TPC)

    def ag_split(a):
        m = fx.max(a["m"], i32(1))
        return fx.min(fx.max(i32(gpu.grid_dim.x) // m, i32(1)), i32(NCHA))

    def ag_rows(a):
        bid = i32(gpu.block_id("x"))
        nblk = i32(gpu.grid_dim.x)
        S = ag_split(a)
        nr = fx.min(fx.max((a["m"] - bid + nblk - i32(1)) // nblk, i32(0)), i32(AGR))
        m = fx.max(a["m"], i32(1))
        s = bid // m
        split = S > i32(1)
        live = s < S
        row0 = split.select(bid - s * m, bid)
        nrows = split.select(live.select(i32(1), i32(0)), nr)
        qb = split.select(s, i32(0))
        qs = split.select(S, i32(1))
        cnt = split.select(
            live.select((i32(NCHA) - s + S - i32(1)) // S, i32(0)), i32(NCHA)
        )
        return row0, nblk, nrows, qb, qs, cnt

    def ag_group(a):
        return (ag_split(a) > i32(1)).select(i32(1), i32(PRE_CH))

    def ag_npar(a):
        return ((ag_split(a) > i32(1)) & fx.Boolean(AIN)).select(i32(2), i32(1))

    def _row32(rs, row, g, aux):
        f = []
        for v in range_constexpr(4):
            f += bf16x8_to_f32(
                bld(
                    rs,
                    ((row * i32(H)) + g * i32(32) + i32(v * 8)) * i32(2),
                    0,
                    V4I,
                    aux,
                )
            )
        return f

    def ag_row_vals(a, rxl, i, g):
        if const_expr(not AIN):
            return _row32(rxl, (a["rank"] * a["m"] if RREP else i32(0)) + i, g, 0)
        f = _row32(rxl, a["rank"] * a["m"] + i, g, 0)
        rpre = own_rs(a, "off_pre")
        for src in range_constexpr(TPC):
            vals = _row32(rpre, i32(src) * a["mmax"] + i, g, AUX_SYS)
            f = sum_live(f, vals, i32(src) != a["rank"])
        return [fx.Float32(x).to(fx.BFloat16).to(fx.Float32) for x in f]

    @traced
    def pre_send(L, lane, a):
        row0, stride, nrows, qb, qs, cnt = ag_rows(a)
        rank, m = a["rank"], a["m"]
        x_bytes = a["tp"] * m * i32(H * 2)
        rxl = rsrc(a["x"], x_bytes)
        pc = ag_group(a)
        pre_bytes = a["tp"] * a["mmax"] * i32(H * 2)
        for g_ in range(i32(0), (cnt + pc - i32(1)) // pc, i32(1)):
            g = i32(g_)
            _pre_throttle(L, lane, a, g, pc)
            lo = (qb + g * pc * qs) * i32(256)
            upg = fx.min(pc, cnt - g * pc) * i32(32)
            for q_ in range(lane, nrows * upg, i32(64)):
                q = i32(q_)
                r = q // upg
                c = lo + (q - r * upg) * i32(8)
                i = row0 + r * stride
                vs = [
                    bld(
                        rxl,
                        (i32(p) != rank).select(
                            ((i32(p) * m + i) * i32(H) + c) * i32(2), x_bytes
                        ),
                        0,
                        V4I,
                        0,
                    )
                    for p in range(TPC)
                ]
                off = ((rank * a["mmax"] + i) * i32(H) + c) * i32(2)
                for p in range_constexpr(TPC):
                    rd = peer_rs(a, p, "off_pre", pre_bytes)
                    bst(vs[p], rd, (i32(p) != rank).select(off, pre_bytes), 0, AUX_SYS)
            wait_vm(0)
            _pre_flag(a, lane, g)

    @traced
    def _pre_throttle(L, lane, a, g, pc):
        lag = (ag_split(a) == i32(1)).select(i32(1), i32(2))
        if (g >= lag) & (lane == i32(0)):
            spin_lds_ge(L, L_CTL + C_AGK * 4, (g - lag + i32(1)) * pc)
        rocdl.sched_barrier(0)

    @traced
    def _pre_flag(a, lane, g):
        if lane == i32(0):
            fo = (
                i32(FLAG_PRE)
                + (g * i32(MAX_TP) + a["rank"]) * i32(NCTA_MAX)
                + i32(gpu.block_id("x"))
            ) * i32(4)
            for p in range_constexpr(TPC):
                if i32(p) != a["rank"]:
                    g_st_sys(
                        fx.Int64(a["peer"][p]) + fx.Int64(a["off_flag"]) + fx.Int64(fo),
                        a["epoch"],
                    )

    @traced
    def pre_wait(lane, a, g):
        src = fx.min(lane, i32(TPC - 1))
        addr = (
            a["mine"]
            + fx.Int64(a["off_flag"])
            + fx.Int64(
                (
                    i32(FLAG_PRE)
                    + (g * i32(MAX_TP) + src) * i32(NCTA_MAX)
                    + i32(gpu.block_id("x"))
                )
                * i32(4)
            )
        )
        live = (lane < i32(TPC)) & (lane != a["rank"])
        pend = wave_red(
            (live & (g_ld_sys(addr) < a["epoch"])).select(i32(1), i32(0)), lane, fx.max
        )
        t0 = _now()
        while (pend != i32(0)) & _alive(t0):
            rocdl.s_sleep(1)
            pend = wave_red(
                (live & (g_ld_sys(addr) < a["epoch"])).select(i32(1), i32(0)),
                lane,
                fx.max,
            )
        _report_if(a, (pend != i32(0)) & (lane == i32(0)), ERR_FLAG)

    @traced
    def quant_chunks(L, t0, stride, a, k_lo, k_hi):
        row0, stride_r, nrows, qb, qstep, cnt = ag_rows(a)
        k_hi = fx.min(k_hi, cnt)
        nk = fx.max(k_hi - k_lo, i32(0))
        rxl = rsrc(a["x"], (a["ttot"] if RREP else a["m"]) * i32(H * 2))
        for q_ in range(t0, nrows * nk * i32(8), i32(stride)):
            q = i32(q_)
            r = q // (nk * i32(8))
            e = q - r * nk * i32(8)
            g = (qb + (k_lo + e // i32(8)) * qstep) * i32(8) + e % i32(8)
            f = ag_row_vals(a, rxl, row0 + r * stride_r, g)
            e8, qs = _e8m0_from_amax(amax(f), max_norm=448.0 if A8 else 6.0)
            if const_expr(A8):
                words = [fp8x4_pack(f[dw * 4 : dw * 4 + 4], qs) for dw in range(8)]
            else:
                words = [fp4_pack(f[dw * 8 : dw * 8 + 8], qs) for dw in range(4)]
            lds_st(
                L,
                L_INTER + r * i32(AG_SROW) + g * i32(4 * len(words)),
                fx.Vector.from_elements(words, fx.Int32),
                16,
            )
            lds_st(
                L, L_INTER + i32(AG_SCB) + r * i32(H // 32) + g, fx.Int8(e8), align=1
            )

    @traced
    def ag_send(L, lane, a, par=0):
        row0, stride, nrows, qb, qs, cnt = ag_rows(a)
        rank = a["rank"]
        if const_expr(not RREP and par == 0 and not MLL):
            mp = (ag_split(a) > i32(1)).select(
                i32(0), _meta_pending(a, lane, a["epoch"], own=True)
            )
            t0 = _now()
            while (mp != i32(0)) & _alive(t0):
                rocdl.s_sleep(1)
                mp = _meta_pending(a, lane, a["epoch"], own=True)
        r = lane // i32(XLPR)
        j = lane - r * i32(XLPR)
        rok = r < nrows
        i = row0 + fx.min(r, fx.max(nrows - i32(1), i32(0))) * stride
        gslot = rank * a["m"] + i
        x_bytes = a["ttot"] * i32(XB)
        xs_bytes = a["ttot"] * i32(H // 32)
        rx = [peer_rs(a, p, "off_x", x_bytes) for p in range(TPC)]
        rs = [peer_rs(a, p, "off_xs", xs_bytes) for p in range(TPC)]
        pc = ag_group(a)
        split = ag_split(a) > i32(1)
        pipe = (split & fx.Boolean(AIN)) == fx.Boolean(False)
        npar = ag_npar(a)
        part = i32(par) < npar
        for k in range_constexpr(NCHA):
            mine = (i32(k) % npar) == i32(par)
            kl = (i32(k) < cnt) & mine
            q = fx.min(qb + i32(k) * qs, i32(NCHA - 1))
            if const_expr(AIN and k >= 1):  # noqa: SIM102 (compile-time guard)
                if split & mine & (i32(k) >= npar) & (i32(k) - npar < cnt):
                    wait_vm(0)
                    _ag_bump(a, lane, qb + (i32(k) - npar) * qs)
            if kl:
                ag_chunk(
                    L, lane, a, k, q, pc, r, j, rok, gslot, rx, rs, x_bytes, xs_bytes
                )
            if const_expr(AIN):
                _agk_note(L, lane, kl)
            if const_expr(k == NCHA - 1):
                wait_lgkm0()
                if part:
                    _ag_free(L, lane)
            if const_expr(k >= 1):  # noqa: SIM102 (compile-time guard)
                if pipe & part & (i32(k - 1) < cnt):
                    if kl:
                        wait_vm(2 * TPC)
                    else:
                        wait_vm(0)
                    _ag_bump(a, lane, qb + i32(k - 1) * qs)
            rocdl.sched_barrier(0)
        wait_vm(0)
        if pipe & part & (i32(NCHA - 1) < cnt):
            _ag_bump(a, lane, qb + i32(NCHA - 1) * qs)

    @traced
    def ag_chunk(L, lane, a, k, q, pc, r, j, rok, gslot, rx, rs, x_bytes, xs_bytes):
        if const_expr(AIN):
            if (i32(k) % pc) == i32(0):
                pre_wait(lane, a, i32(k) // pc)
                quant_chunks(L, lane, 64, a, i32(k), i32(k) + pc)
            wait_lgkm0()
            rocdl.sched_barrier(0)
        rr = fx.min(r, i32(AGR - 1))
        cb = q * i32(XLPR * 16) + j * i32(16)
        dv = lds_ld(L, L_INTER + rr * i32(AG_SROW) + cb, V4I, 16)
        sv = lds_ld(L, L_INTER + i32(AG_SCB) + rr * i32(H // 32) + q * i32(8), V2I, 8)
        doff = rok.select(gslot * i32(XB) + cb, x_bytes)
        soff = (rok & (j == i32(0))).select(gslot * i32(H // 32) + q * i32(8), xs_bytes)
        for p in range_constexpr(TPC):
            bst(dv, rx[p], doff, 0, AUX_SYS)
            bst(sv, rs[p], soff, 0, AUX_SYS)

    @traced
    def _agk_note(L, lane, sent):
        if (lane == i32(0)) & sent:
            lds_atomic_add(L, L_CTL + C_AGK * 4, 1, REL)

    @traced
    def _ag_free(L, lane):
        if lane == i32(0):
            lds_atomic_add(L, L_CTL + C_AGFREE * 4, 1, REL)

    @traced
    def ag_stage_free(L, lane, a):
        if lane == i32(0):
            spin_lds_ge(L, L_CTL + C_AGFREE * 4, ag_npar(a))
        rocdl.sched_barrier(0)

    def _ag_senders(a, q, x):
        m = fx.max(a["m"], i32(1))
        S = ag_split(a)
        lo = (S > i32(1)).select((q % S) * m, i32(0))
        hi = (S > i32(1)).select(lo + m, i32(gpu.grid_dim.x))
        return (hi - x + i32(N_XCD - 1)) // i32(N_XCD) - (
            lo - x + i32(N_XCD - 1)
        ) // i32(N_XCD)

    @traced
    def _ag_bump(a, lane, q):
        if lane == i32(0):
            x = i32(gpu.block_id("x")) % i32(N_XCD)
            xa = ctrl_at(a, i32(CTRL_AGX) + (q * i32(N_XCD) + x) * i32(LRDY_STRIDE))
            if g_add_agent(xa, 1) + i32(1) == _ag_senders(a, q, x):
                g_st_sys(xa, i32(0))
                _ag_bump_rank(a, q)

    @traced
    def _ag_bump_rank(a, q):
        ga = ctrl_at(a, i32(CTRL_AGG) + q * i32(LRDY_STRIDE))
        m = fx.max(a["m"], i32(1))
        nx = fx.min((ag_split(a) > i32(1)).select(m, i32(gpu.grid_dim.x)), i32(N_XCD))
        if g_add_agent(ga, 1) + i32(1) == nx:
            g_st_sys(ga, i32(0))
            idx = i32(FLAG_AGQ) + q * i32(MAX_TP) + a["rank"]
            for p in range_constexpr(TPC):
                bst(a["epoch"], peer_rs(a, p, "off_flag"), idx * i32(4), 0, AUX_SYS)

    def _ag_nmeta(a):
        return fx.max((a["m"] * i32(TOPK) // i32(4) + i32(63)) // i32(64), i32(1))

    @traced
    def _ag_send_meta(lane, a):
        bid = i32(gpu.block_id("x"))
        if bid < _ag_nmeta(a):
            n = a["m"] * i32(TOPK)
            n4 = n // i32(4)
            # lanes past the input read 0 (their stores are dropped)
            rid = rsrc(a["ids_in"], n * i32(4))
            rtw = rsrc(a["tw_in"], n * i32(4))
            dst0 = a["rank"] * n * i32(4)
            v = bid * i32(64) + lane
            idv = bld(rid, v * i32(16), 0, V4I, 0)
            wvv = bld(rtw, v * i32(16), 0, V4I, 0)
            e = n4 * i32(4) + lane
            ide = bld(rid, e * i32(4), 0, T.i32, 0)
            wte = bld(rtw, e * i32(4), 0, T.i32, 0)
            big = i32(1 << 30)
            vo = (v < n4).select(dst0 + v * i32(16), big)
            eo = ((bid == i32(0)) & (e < n)).select(dst0 + e * i32(4), big)
            for p in range_constexpr(TPC):
                ri = peer_rs(a, p, "off_ids", big)
                rw = peer_rs(a, p, "off_w", big)
                bst(idv, ri, vo, 0, AUX_SYS)
                bst(wvv, rw, vo, 0, AUX_SYS)
                bst(ide, ri, eo, 0, AUX_SYS)
                bst(wte, rw, eo, 0, AUX_SYS)
            wait_vm(0)
            if lane == i32(0):
                fo = (i32(FLAG_AGM) + a["rank"] * i32(32) + bid) * i32(4)
                # every copy landed: this rank's flag needs no wait for the
                # remote flags
                bst(a["epoch"], own_rs(a, "off_flag"), fo, 0, AUX_SYS)
                for p in range_constexpr(TPC):
                    if i32(p) != a["rank"]:
                        bst(a["epoch"], peer_rs(a, p, "off_flag"), fo, 0, AUX_SYS)

    @traced
    def _ag_send_meta_ll(lane, a):
        n = a["m"] * i32(TOPK)
        rid = rsrc(a["ids_in"], n * i32(4))
        rtw = rsrc(a["tw_in"], n * i32(4))
        big = i32(1 << 30)
        for i_ in range(
            i32(gpu.block_id("x")) * i32(64) + lane, n, i32(gpu.grid_dim.x) * i32(64)
        ):
            i = i32(i_)
            e = fx.Int32(bld(rid, i * i32(4), 0, T.i32, 0))
            wv = fx.Int32(bld(rtw, i * i32(4), 0, T.i32, 0))
            pkt = fx.Vector.from_elements([e, a["epoch"], wv, a["epoch"]], fx.Int32)
            off = (a["rank"] * n + i) * i32(16)
            for p in range_constexpr(TPC):
                bst(pkt, peer_rs(a, p, "off_ids", big), off, 0, AUX_SYS)

    @traced
    def meta_ll_wait(tid, a):
        n = a["ttot"] * i32(TOPK)
        rid = rsrc(a["ids"])
        for i_ in range(tid, n, i32(NT)):
            i = i32(i_)
            t = fx.Int32(fx.Vector(bld(rid, i * i32(16), 0, V4I, AUX_SYS))[1])
            t0 = _now()
            while (t != a["epoch"]) & _alive(t0):
                rocdl.s_sleep(1)
                t = fx.Int32(fx.Vector(bld(rid, i * i32(16), 0, V4I, AUX_SYS))[1])
            _report_if(a, t != a["epoch"], ERR_META)

    def _meta_pending(a, lane, epoch, own=False):
        rf = own_rs(a, "off_flag")
        nmeta = _ag_nmeta(a)
        pend = i32(0)
        for it in range_constexpr(MAX_TP * NMETA_MAX // 64):
            e = i32(it * 64) + lane
            p = e // i32(NMETA_MAX)
            j = e - p * i32(NMETA_MAX)
            ok = ((p == a["rank"]) if own else (p < a["tp"])) & (j < nmeta)
            f = fx.Int32(
                bld(rf, (i32(FLAG_AGM) + p * i32(32) + j) * i32(4), 0, T.i32, AUX_SYS)
            )
            pend = fx.max(pend, (ok & (f < epoch)).select(i32(1), i32(0)))
        return wave_red(pend, lane, fx.max)

    @traced
    def ag_wait_meta(a, tid, epoch):
        if tid < i32(64):
            lane = tid % i32(64)
            pend = _meta_pending(a, lane, epoch)
            t0 = _now()
            while (pend != i32(0)) & _alive(t0):
                rocdl.s_sleep(1)
                pend = _meta_pending(a, lane, epoch)
            _report_if(a, (pend != i32(0)) & (lane == i32(0)), ERR_META)

    def _ag_first_pending(lane, a, epoch, c_lo, nwin):
        rf = own_rs(a, "off_flag")
        end = fx.min(c_lo + i32(nwin), i32(NCH))
        first = end
        for it in range_constexpr((TPC * nwin + 63) // 64):
            e = i32(it * 64) + lane
            cc = c_lo + e // i32(TPC)
            src = e % i32(TPC)
            idx = i32(FLAG_AGQ) + fx.min(cc, i32(NCH - 1)) * i32(MAX_TP) + src
            f = fx.Int32(bld(rf, idx * i32(4), 0, T.i32, AUX_SYS))
            first = fx.min(first, ((cc < end) & (f < epoch)).select(cc, end))
        return wave_red(first, lane, fx.min)

    @traced
    def ag_wait_chunk(L, lane, a, epoch, cc):
        rdy = _ag_first_pending(lane, a, epoch, cc, AG_WIN)
        if cc == i32(0):
            rdy = _ag_first_pending(lane, a, epoch, i32(0), NCH)
        t0 = _now()
        while (rdy <= cc) & _alive(t0):
            rocdl.s_sleep(1)
            rdy = _ag_first_pending(lane, a, epoch, cc, AG_WIN)
        _report_if(a, (rdy <= cc) & (lane == i32(0)), ERR_CHUNK)
        if lane == i32(0):
            lds_st(L, L_CTL + C_ARDY * 4, rdy)
        rocdl.sched_barrier(0)

    @traced
    def finish(tid, a, epoch):
        if tid == i32(0):
            bid = i32(gpu.block_id("x"))
            g_st_sys(ctrl_at(a, i32(CTRL_EPB) + bid), epoch)
            x = bid % i32(N_XCD)
            nx = (i32(gpu.grid_dim.x) - x + i32(N_XCD - 1)) // i32(N_XCD)
            xc = ctrl_at(a, i32(CTRL_XES) + x * i32(LRDY_STRIDE))
            if g_add_agent(xc, 1) == nx - i32(1):
                g_st_sys(xc, i32(0))
                fx.memory_fence(syncscope="one-as", ordering=ACQ)

    @traced
    def _dyn_zero(L, tid):
        if (tid >= i32(NT + 64)) & (tid < i32(NT + 64 + NBW)):
            lds_st(L, L_DYN + (tid - i32(NT + 64)) * i32(4), i32(0))
        if const_expr(CHUNK):
            for e_ in range(tid, i32(NE), i32(NTT)):
                lds_st(L, L_DCNT + i32(e_) * i32(4), i32(0))
        if const_expr(LB):
            for e_ in range(tid, i32(NE), i32(NTT)):
                lds_st(L, L_DPRE + i32(e_) * i32(4), i32(0))

    @traced
    def init_lds(L, tid, a):
        bid = i32(gpu.block_id("x"))
        if (tid >= i32(64)) & (tid < i32(64 + 6)):
            k = tid - i32(64)
            v = g_ld_i32(
                fx.Int64(a["cta_units"])
                + fx.Int64(bid * i32(UNIT_REC) + k) * fx.Int64(4)
            )
            lds_st(L, L_CTL + (i32(C_UNIT) + k) * i32(4), v)
        if (tid < i32(C_UNIT)) | (
            (tid >= i32(C_UNIT + 6)) & (tid < i32(C_XC + C_XC_N))
        ):
            is_done = (tid >= i32(C_DONE)) & (tid < i32(C_DONE + NW))
            ep = g_ld_rel(ctrl_at(a, i32(CTRL_EPB) + bid), "agent") + i32(1)
            v = is_done.select(i32(-1), (tid == i32(C_EPOCH)).select(ep, i32(0)))
            stage = (tid == i32(C_LRED)) | (tid == i32(C_PULL))
            v = (stage & comm_idle(a)).select(i32(NV), v)
            if const_expr(DLL):
                v = ((tid == i32(C_LRED)) | (tid == i32(C_NSIG))).select(i32(NCK), v)
            if const_expr(LB):
                # column chunks are signaled by their last GEMM2 unit
                v = (tid == i32(C_NSIG)).select(i32(NCK), v)
            own = fin_owned(a)
            v = (tid == i32(C_PULL)).select(i32(NV) - _ctpop(own), v)
            v = (tid == i32(C_FBITS)).select(i32((1 << NV) - 1) & (own ^ i32(-1)), v)
            if const_expr(ARLL):
                v = (tid == i32(C_PULL)).select(i32(NCK), v)
            lds_st(L, L_CTL + tid * i32(4), v)

    DLL_NWS = 4
    DLL_CARRY = 48

    def _dll_rows(a, t, c0, vc, P):
        rb = route_region_bytes(a["ttot"])
        r_routes = rsrc(a["routes"], rb)
        r_pr = rsrc(a["proutes"], i32(NPC - 1) * rb)
        out = []
        for k in range_constexpr(TOPK):
            off = ((t * i32(TOPK) + i32(k)) * i32(H) + c0 + vc * i32(8)) * i32(2)
            out.append(fx.Vector(bld(r_routes, off, 0, V4I, AUX_SC1)))
            for p in range_constexpr(1, NPC):
                o2 = (i32(p) < P).select(i32(p - 1) * rb + off, i32(NPC) * rb)
                out.append(fx.Vector(bld(r_pr, o2, 0, V4I, AUX_SC1)))
        return out

    def _dll_items(ws):
        nblk = i32(gpu.grid_dim.x)
        return i32(gpu.block_id("x")) + i32(ws) * nblk, nblk * i32(DLL_NWS)

    @traced
    def push_dll(L, tid, a, ws=0):
        lane = tid % i32(64)
        # a chunk's rows are complete about when this CTA's GEMM2 is done: no
        # polling (memory traffic) before that
        if lane == i32(0):
            spin_lds_ge(L, L_CTL + C_CDONE * 4, i32(1))
        rocdl.sched_barrier(0)
        P = uni(lds_ld_acq(L, L_CTL + C_DYNP * 4)) if DYN else i32(1)
        ttot = a["ttot"]
        it0, istep = _dll_items(ws)
        for it_ in range(it0, ttot * i32(NCK), istep):
            it = i32(it_)
            c = it // ttot
            t = it - c * ttot
            c0 = c * i32(CW)
            r_dst, prow = token_dst(a, t)
            if const_expr(ARLL):
                prow = a["rank"] * a["mmax"] * a["tp"] + t
            for j in range_constexpr(VPL):
                v = lane + i32(j * 64)
                vc = fx.min(v, i32(CW // 8 - 1))
                live = [(v < i32(CW // 8)) & (i32(p) < P) for p in range(NPC)] * TOPK
                if const_expr(TOPK * NPC <= DLL_CARRY):
                    pk = _dll_rows(a, t, c0, vc, P)
                    pend = ll_pending(a, lane, zip(pk, live))
                    t0 = _now()
                    while (pend != i32(0)) & _alive(t0):
                        rocdl.s_sleep(8)
                        pk = _dll_rows(a, t, c0, vc, P)
                        pend = ll_pending(a, lane, zip(pk, live))
                else:
                    # many packets: carrying them through the poll loop spills
                    pend = ll_pending(a, lane, zip(_dll_rows(a, t, c0, vc, P), live))
                    t0 = _now()
                    while (pend != i32(0)) & _alive(t0):
                        rocdl.s_sleep(8)
                        pend = ll_pending(
                            a, lane, zip(_dll_rows(a, t, c0, vc, P), live)
                        )
                    pk = _dll_rows(a, t, c0, vc, P)
                _report_if(a, (pend != i32(0)) & (lane == i32(0)), ERR_COMM)
                acc = [fx.Float32(0.0)] * 8
                for k in range_constexpr(TOPK):
                    for p in range_constexpr(NPC):
                        acc = sum_live(acc, ll_vals(pk[k * NPC + p]), i32(p) < P)
                if const_expr(ARLL):
                    for pr in range_constexpr(TPC):
                        _push_store(
                            a,
                            peer_rs(a, pr, "off_part"),
                            prow,
                            c0,
                            v,
                            acc,
                            v < i32(CW // 8),
                        )
                else:
                    _push_store(a, r_dst, prow, c0, v, acc, v < i32(CW // 8))
        if const_expr(ARLL):
            final_all_ll(L, tid, a, ws)

    @traced
    def final_all_ll(L, tid, a, ws=0):
        lane = tid % i32(64)
        ttot = a["ttot"]
        trows = a["mmax"] * a["tp"]
        r_recv = own_rs(a, "off_part")
        ry = rsrc(a["y"], ttot * i32(H * 2))
        it0, istep = _dll_items(ws)
        for it_ in range(it0, ttot * i32(NCK), istep):
            it = i32(it_)
            c = it // ttot
            t = it - c * ttot
            c0 = c * i32(CW)
            for j in range_constexpr(VPL):
                v = lane + i32(j * 64)
                vc = fx.min(v, i32(CW // 8 - 1))
                live = [(i32(p) < a["tp"]) & (v < i32(CW // 8)) for p in range(MAX_TP)]
                pk = recv_pkts(a, r_recv, t, trows, c0, vc)
                pend = ll_pending(a, lane, zip(pk, live))
                t0 = _now()
                while (pend != i32(0)) & _alive(t0):
                    rocdl.s_sleep(1)
                    pk = recv_pkts(a, r_recv, t, trows, c0, vc)
                    pend = ll_pending(a, lane, zip(pk, live))
                _report_if(a, (pend != i32(0)) & (lane == i32(0)), ERR_COMM)
                acc = [fx.Float32(0.0)] * 8
                for p in range_constexpr(MAX_TP):
                    acc = sum_live(acc, ll_vals(pk[p]), i32(p) < a["tp"])
                if v < i32(CW // 8):
                    bst(
                        pack_bf16x8(acc),
                        ry,
                        (t * i32(H) + c0 + v * i32(8)) * i32(2),
                        0,
                        0,
                    )

    TN_V = H // 8  # 8-column pieces of a row
    TN_IT = (TN_V + NTT - 1) // NTT

    @traced
    def blk_red(L, tid, v, op, slot):
        # every thread of the CTA: op over the CTA (waves via LDS at slot)
        r = wave_red(v, tid % i32(64), op)
        if (tid % i32(64)) == i32(0):
            lds_st(L, i32(L_RING) + (i32(slot * (NTT // 64)) + tid // i32(64)) * i32(4), r)
        gpu.barrier()
        t = lds_ld_i32(L, i32(L_RING) + i32(slot * (NTT // 64) * 4))
        for k in range_constexpr(1, NTT // 64):
            t = op(t, lds_ld_i32(L, i32(L_RING) + i32((slot * (NTT // 64) + k) * 4)))
        return t

    @traced
    def tn_rows(L, tid, a):
        # this rank's output rows, each once every CTA that finals some of its
        # column chunks is done (the last of them takes it)
        m = a["m"]
        S = fin_split(a)
        step = (S > i32(1)).select(m, i32(gpu.grid_dim.x))
        mine = fin_owned(a) != i32(0)
        # (S > 1: only the m * S CTAs that final column chunks count a row's
        # arrivals; the rest, when m * S < the grid, must not touch the counters)
        start = ((S > i32(1)) & (i32(gpu.block_id("x")) >= m * S)).select(m, fin_key(a))
        for r_ in range(start, m, step):
            r = i32(r_)
            if tid == i32(0):
                last = i32(1)
                if S > i32(1):
                    # (y went out system scope: every CTA's chunks are in
                    # memory once its stores drained)
                    ca = ctrl_at(a, i32(CTRL_TNC) + r)
                    old = g_add_agent(ca, 1)
                    last = (old == S - i32(1)).select(i32(1), i32(0))
                    if last == i32(1):
                        g_st_sys(ca, i32(0))
                lds_st(L, L_CTL + C_TNL * 4, mine.select(last, i32(0)))
            gpu.barrier()
            if lds_ld_i32(L, L_CTL + C_TNL * 4) == i32(1):
                # one CTA finals every chunk of its rows (S = 1): its own
                # stores, plain loads
                tn_row(L, tid, a, r, True)
            gpu.barrier()

    @traced
    def tn_row(L, tid, a, r, sysy):
        H8 = i32(TN_V)
        ry = rsrc(a["y"])
        rres = rsrc(a["tn_res"])
        rout = rsrc(a["tn_out"])
        rw = rsrc(a["tn_w"])
        f, ss = [], fx.Float32(0.0)
        for k in range_constexpr(TN_IT):
            q = tid + i32(k * NTT)
            qc = fx.min(q, H8 - i32(1))
            off = (r * i32(H) + qc * i32(8)) * i32(2)
            yv = bf16x8_to_f32(bld(ry, off, 0, V4I, AUX_SYS if sysy else 0))
            rv = bf16x8_to_f32(bld(rres, off, 0, V4I, 0))
            fk = [x + y for x, y in zip(yv, rv)]
            live = q < H8
            if live:
                bst(pack_bf16x8(fk), rout, off, 0, 0)
            fk = [live.select(x, fx.Float32(0.0)) for x in fk]
            for x in fk:
                ss = ss + x * x
            f.append(fk)
        tot = blk_red(L, tid, ss.bitcast(fx.Int32), lambda x, y: (x.bitcast(fx.Float32) + y.bitcast(fx.Float32)).bitcast(fx.Int32), 0)
        rcp = fx.Float32(
            fmath.rsqrt(
                _u(tot.bitcast(fx.Float32) / fx.Float32(float(H)) + fx.Float32(float(tn_eps)))
            )
        )
        xs, am = [], fx.Float32(0.0)
        for k in range_constexpr(TN_IT):
            q = fx.min(tid + i32(k * NTT), H8 - i32(1))
            wv = bf16x8_to_f32(bld(rw, q * i32(16), 0, V4I, 0))
            # GemmaRMSNorm scales by (1 + w), RMSNorm by w
            xk = [
                fx.Float32(
                    fx.Float32(x * rcp * ((w + fx.Float32(1.0)) if tn_gemma else w))
                    .to(fx.BFloat16)
                    .to(fx.Float32)
                )
                for x, w in zip(f[k], wv)
            ]
            am = am.maximumf(amax(xk))
            xs.append(xk)
        amx = blk_red(L, tid, am.bitcast(fx.Int32), lambda x, y: x.bitcast(fx.Float32).maximumf(y.bitcast(fx.Float32)).bitcast(fx.Int32), 1)
        scale = amx.bitcast(fx.Float32).maximumf(fx.Float32(1e-10)) / fx.Float32(448.0)
        inv = fx.Float32(1.0) / scale
        one = fx.Float32(1.0)
        tlx0(a, tid, 9)  # [tl]
        grow = a["rank"] * a["m"] + r
        for k in range_constexpr(TN_IT):
            q = tid + i32(k * NTT)
            qv = [x * inv for x in xs[k]]
            d = fx.Vector.from_elements([fp8x4_pack(qv[0:4], one), fp8x4_pack(qv[4:8], one)], fx.Int32)
            if q < H8:
                for p in range_constexpr(TPC):
                    bst(d, peer_rs(a, p, "off_qall"), grow * i32(H) + q * i32(8), 0, AUX_SYS)
        # tn 2: the bf16 rows too (qall bytes: the FP8 rows of every rank's
        # tokens, then their bf16 rows)
        b0 = a["mmax"] * i32(TPC * H)
        for k in range_constexpr(TN_IT if TNB else 0):
            q = tid + i32(k * NTT)
            d = pack_bf16x8(xs[k])
            if q < H8:
                for p in range_constexpr(TPC):
                    bst(d, peer_rs(a, p, "off_qall"), b0 + (grow * i32(H) + q * i32(8)) * i32(2), 0, AUX_SYS)
        if tid == i32(0):
            for p in range_constexpr(TPC):
                bst(scale.bitcast(fx.Int32), peer_rs(a, p, "off_sall"), grow * i32(4), 0, AUX_SYS)
        wait_vm(0)
        tlx0(a, tid, 15)  # [tl]
        gpu.barrier()
        if tid < i32(TPC):
            fo = fx.Int64((i32(FLAG_TN) + a["rank"] * i32(TN_MAX) + r) * i32(4))
            g_st_sys(peer_sel(a, tid) + fx.Int64(a["off_flag"]) + fo, a["epoch"])

    @traced
    def tn_wait(tid, a):
        # every rank's rows landed in qall / sall: rows bid, bid + nblk, ...
        if tid < i32(64):
            nblk = i32(gpu.grid_dim.x)
            t = i32(gpu.block_id("x")) + tid * nblk
            live = t < a["ttot"]
            tc = fx.min(t, a["ttot"] - i32(1))
            src = tc // a["m"]
            row = tc - src * a["m"]
            addr = a["mine"] + fx.Int64(a["off_flag"]) + fx.Int64((i32(FLAG_TN) + src * i32(TN_MAX) + row) * i32(4))
            pend = wave_red((live & (g_ld_sys(addr) < a["epoch"])).select(i32(1), i32(0)), tid, fx.max)
            t0 = _now()
            while (pend != i32(0)) & _alive(t0):
                rocdl.s_sleep(1)
                pend = wave_red((live & (g_ld_sys(addr) < a["epoch"])).select(i32(1), i32(0)), tid, fx.max)
            _report_if(a, (pend != i32(0)) & (tid == i32(0)), ERR_YAG)

    @traced
    def roles(L, tid, a, epoch):
        if tid < i32(NT):
            compute_units(L, tid, a)
            comm_help(L, tid, a, epoch)
        else:
            if tid < i32(NT + 64):
                ag_send(L, tid % i32(64), a)
                if const_expr(DLL):
                    push_dll(L, tid, a)
                    comm_wave(L, tid, a, epoch)
                else:
                    poll_loop(L, tid, a, epoch)
            elif tid < i32(NT + 128):
                a_loader(L, tid, a)
                if const_expr(LBPF):
                    lb_g2_prefetch(L, tid % i32(64), a)
                if const_expr(DLL):
                    push_dll(L, tid, a, 1)
                    signal_loop(L, tid, a, epoch)
                comm_help(L, tid, a, epoch)
            else:
                if tid >= i32(NT + 192):
                    classify_tokens(L, tid % i32(64), a)
                if const_expr(AIN):
                    if tid < i32(NT + 192):
                        pre_send(L, tid % i32(64), a)
                    else:
                        ag_send(L, tid % i32(64), a, 1)
                if const_expr(DLL):
                    if tid < i32(NT + 192):
                        zero_masked(L, tid % i32(64), a, 0, 1)
                        push_dll(L, tid, a, 2)
                    else:
                        push_dll(L, tid, a, 3)
                if const_expr(ZMA):
                    if tid < i32(NT + 192):
                        zma_run(L, tid % i32(64), a)
                comm_help(L, tid, a, epoch)

    Shared = fx.struct(
        type(
            "Shared", (), {"__annotations__": {"buf": fx.Array[fx.Int8, LDS_BYTES, 16]}}
        )
    )

    @flyc.kernel(name=name, known_block_size=[NTT, 1, 1])
    def mega_moe_tp_kernel(
        w1: fx.Int64,
        w1s: fx.Int64,
        w2: fx.Int64,
        w2s: fx.Int64,
        x: fx.Int64,
        ids_in: fx.Int64,
        tw_in: fx.Int64,
        y: fx.Int64,
        routes: fx.Int64,
        proutes: fx.Int64,
        ctrl: fx.Int64,
        units: fx.Int64,
        cta_units: fx.Int64,
        p0: fx.Int64,
        p1: fx.Int64,
        p2: fx.Int64,
        p3: fx.Int64,
        p4: fx.Int64,
        p5: fx.Int64,
        p6: fx.Int64,
        p7: fx.Int64,
        off_x: fx.Int64,
        off_xs: fx.Int64,
        off_ids: fx.Int64,
        off_w: fx.Int64,
        off_part: fx.Int64,
        off_flag: fx.Int64,
        off_pre: fx.Int64,
        off_yall: fx.Int64,
        rank: fx.Int32,
        tp: fx.Int32,
        m: fx.Int32,
        mmax: fx.Int32,
        piece_e0: fx.Int32,
        xg: fx.Int64,
        col0: fx.Int32,
        ncol: fx.Int32,
        tn_res: fx.Int64,
        tn_out: fx.Int64,
        tn_w: fx.Int64,
        off_qall: fx.Int64,
        off_sall: fx.Int64,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        assert layout_tag and name
        lds = fx.SharedAllocator().allocate(Shared).peek()
        L = uni(fx.Int32(fx.ptrtoint(lds.buf.ptr)))
        peers = [p0, p1, p2, p3, p4, p5, p6, p7]
        mine = peers[0]
        for j in range_constexpr(1, MAX_TP):
            mine = (rank == i32(j)).select(peers[j], mine)
        a = {
            "w1": w1,
            "w1s": w1s,
            "w2": w2,
            "w2s": w2s,
            "x": x,
            "ids_in": ids_in,
            "tw_in": tw_in,
            "y": y,
            "routes": routes,
            "proutes": proutes,
            "ctrl": ctrl,
            "units": units,
            "cta_units": cta_units,
            "peer": peers,
            "mine": fx.Int64(mine),
            "off_x": off_x,
            "off_xs": off_xs,
            "off_ids": off_ids,
            "off_w": off_w,
            "off_part": off_part,
            "off_flag": off_flag,
            "off_pre": off_pre,
            "off_yall": off_yall,
            "rank": rank,
            "tp": tp,
            "m": m,
            "mmax": mmax,
            "ttot": tp * m,
            "piece_e0": piece_e0,
            "xg": xg,
            "col0": col0,
            "ncol": ncol,
            "tn_res": tn_res,
            "tn_out": tn_out,
            "tn_w": tn_w,
            "off_qall": off_qall,
            "off_sall": off_sall,
        }
        _TLA[0] = a  # [tl]
        tlx0(a, tid, 10)  # [tl2]
        if const_expr(DYN):
            _dyn_zero(L, tid)
        init_lds(L, tid, a)
        gpu.barrier()
        epoch = lds_ld_i32(L, L_CTL + C_EPOCH * 4)
        a["epoch"] = epoch
        # every later arena read follows its data's flag: dropping stale lines
        # at the start suffices
        _drop_stale(tid, a)
        claim_reset(tid, a)
        if const_expr(RREP):
            a["ids"] = fx.Int64(ids_in)
            a["tw"] = fx.Int64(tw_in)
        else:
            par = fx.Int64(epoch & i32(1)) * fx.Int64(a["tp"] * a["mmax"] * i32(TOPK))
            a["off_ids"] = off_ids + par * fx.Int64(16)
            a["off_w"] = off_w + par * fx.Int64(4)
            a["ids"] = fx.Int64(mine) + a["off_ids"]
            a["tw"] = fx.Int64(mine) + a["off_w"]
            if (tid >= i32(NT)) & (tid < i32(NT + 64)):
                if const_expr(MLL):
                    _ag_send_meta_ll(tid % i32(64), a)
                else:
                    _ag_send_meta(tid % i32(64), a)
        if const_expr(not AIN):
            quant_chunks(L, tid, NTT, a, i32(0), i32(NCHA))
        gpu.barrier()
        a["ax"] = fx.Int64(mine) + off_x
        a["axs"] = fx.Int64(mine) + off_xs
        tlx0(a, tid, 0)  # [tl]
        roles(L, tid, a, epoch)
        if const_expr(AR and not ARLL):
            wait_vm(0)
        if const_expr(TN):
            wait_vm(0)
        gpu.barrier()
        if const_expr(AR and not ARLL):
            yag_flush(L, tid, a)
            yag_wait(tid, a, epoch)
        if const_expr(TN):
            tlx0(a, tid, 6)  # [tl]
            tn_rows(L, tid, a)
            tlx0(a, tid, 7)  # [tl]
            tn_wait(tid, a)
            tlx0(a, tid, 8)  # [tl]
        tlx0(a, tid, 2)  # [tl]
        if const_expr(AG8):
            gpu.barrier()
            ag8_convert(tid, a)
        tlx0(a, tid, 3)  # [tl]
        if const_expr(LB):
            lb_finish(tid, a, L)
        finish(tid, a, epoch)

    @flyc.jit
    def launch(
        w1: fx.Int64,
        w1s: fx.Int64,
        w2: fx.Int64,
        w2s: fx.Int64,
        x: fx.Int64,
        ids_in: fx.Int64,
        tw_in: fx.Int64,
        y: fx.Int64,
        routes: fx.Int64,
        proutes: fx.Int64,
        ctrl: fx.Int64,
        units: fx.Int64,
        cta_units: fx.Int64,
        p0: fx.Int64,
        p1: fx.Int64,
        p2: fx.Int64,
        p3: fx.Int64,
        p4: fx.Int64,
        p5: fx.Int64,
        p6: fx.Int64,
        p7: fx.Int64,
        off_x: fx.Int64,
        off_xs: fx.Int64,
        off_ids: fx.Int64,
        off_w: fx.Int64,
        off_part: fx.Int64,
        off_flag: fx.Int64,
        off_pre: fx.Int64,
        off_yall: fx.Int64,
        rank: fx.Int32,
        tp: fx.Int32,
        m: fx.Int32,
        mmax: fx.Int32,
        piece_e0: fx.Int32,
        xg: fx.Int64,
        col0: fx.Int32,
        ncol: fx.Int32,
        tn_res: fx.Int64,
        tn_out: fx.Int64,
        tn_w: fx.Int64,
        off_qall: fx.Int64,
        off_sall: fx.Int64,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        mega_moe_tp_kernel(
            w1,
            w1s,
            w2,
            w2s,
            x,
            ids_in,
            tw_in,
            y,
            routes,
            proutes,
            ctrl,
            units,
            cta_units,
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            off_x,
            off_xs,
            off_ids,
            off_w,
            off_part,
            off_flag,
            off_pre,
            off_yall,
            rank,
            tp,
            m,
            mmax,
            piece_e0,
            xg,
            col0,
            ncol,
            tn_res,
            tn_out,
            tn_w,
            off_qall,
            off_sall,
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(NTT, 1, 1), stream=stream)

    return launch
