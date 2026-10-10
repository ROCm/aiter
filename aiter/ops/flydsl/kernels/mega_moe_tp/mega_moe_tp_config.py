# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compile-time layout (flag / ctrl / LDS slots), shape limits and the per-instance
KernelCtx shared by the parts of the fused TP MegaMoE kernel."""

from __future__ import annotations

import functools
import math
import struct
import types

import flydsl.expr as fx

from ..kernels_common import ceildiv
from .common import (
    AUX_SC1,
    AUX_SYS,
    DEADLINE,
    MAX_TP,
)

# tokens per launch (all ranks) up to which the host may pick the dynamic schedule
DYN_MAX, NCTA_MAX, NCK_MAX, NCHA_MAX = 256, 256, 64, 64
# routing-meta flags per rank (one per sending CTA: m * topk <= 64 * NMETA_CAP)
NMETA_CAP = 256
# Flag arena (ints, written by peers, one writer each, values = launch epoch, compared
# wrap-safe: see common.before):
#   FLAG_RDY + c*MAX_TP + r              rank r pushed RS chunk c
#   FLAG_AGM + r*NMETA + b               rank r's routing, CTA b's part
#   FLAG_YAG + (c*MAX_TP + r)*NCTA + b   AR: rank r sent output chunk c of CTA b's rows
#   FLAG_AGQ + q*MAX_TP + r              every CTA of rank r landed K-chunk q
#                                        (per rank, not per row: per-row polls by
#                                        every CTA flood the lines peers write)
#   FLAG_TN + r*TN_MAX + i               tail (tn): rank r sent its output row i
FLAG_RDY = 0
FLAG_AGM = FLAG_RDY + MAX_TP * NCK_MAX
FLAG_YAG = FLAG_AGM + MAX_TP * NMETA_CAP
FLAG_AGQ = FLAG_YAG + NCK_MAX * MAX_TP * NCTA_MAX
TN_MAX = 2048
FLAG_TN = FLAG_AGQ + NCHA_MAX * MAX_TP
FLAG_INTS = FLAG_TN + MAX_TP * TN_MAX

# Local control ints. Polled/atomic counters sit on their own 128 B lines (stride
# LRDY_STRIDE): same-line device atomics from every CTA serialize. Counters are reset
# by their last arriver (a timed-out launch leaves residue: reset()).
CTRL_ERR = 20
CTRL_XF = 32
ERR_FLAG, ERR_META, ERR_CHUNK, ERR_COMM, ERR_YAG = 1, 2, 4, 8, 16
N_XCD = 8
LRDY_STRIDE = 32
CTRL_LRDY = 64 + 2 * NCK_MAX
CTRL_SC = CTRL_LRDY + NCK_MAX * LRDY_STRIDE
# per RS chunk: one line per XCD (CTAs done), all XCDs done, push units done
SC_ALL, SC_PUSH = N_XCD, N_XCD + 1
SC_LINES = SC_PUSH + 1
CTRL_EPB = CTRL_SC + NCK_MAX * SC_LINES * LRDY_STRIDE
CTRL_XES = CTRL_EPB + NCTA_MAX
CTRL_AGX = CTRL_XES + N_XCD * LRDY_STRIDE
CTRL_AGG = CTRL_AGX + NCHA_MAX * N_XCD * LRDY_STRIDE
CTRL_CLM = CTRL_AGG + NCHA_MAX * LRDY_STRIDE
# LB schedule: launches counted by CTRL_LBSEQ (bumped by CTA 0 once every CTA read it);
# its parity picks the bank of the counters below, and every LB launch zeroes the other
# bank for the next one.
#   CTRL_LBR + (bank*N_XCD + x)*LRDY_STRIDE      CTAs of XCD x whose route list slice landed
#   CTRL_COLC + (bank*NCK_MAX + c)*LRDY_STRIDE   GEMM2 units done of column group c
#   CTRL_G1C + (bank*NCHLB_MAX + j)*G1C_STRIDE   GEMM1 pieces done of row chunk j
#   CTRL_TNC + i                                 tail (tn): CTAs done with row i
NCHLB_MAX = 2048
G1C_STRIDE = 16
CTRL_LBSEQ = CTRL_CLM + 2 * LRDY_STRIDE
CTRL_LBR = CTRL_LBSEQ + LRDY_STRIDE
CTRL_COLC = CTRL_LBR + 2 * N_XCD * LRDY_STRIDE
CTRL_G1C = CTRL_COLC + 2 * NCK_MAX * LRDY_STRIDE
CTRL_TNC = CTRL_G1C + 2 * NCHLB_MAX * G1C_STRIDE
CTRL_INTS = CTRL_TNC + TN_MAX
# unit records (expert, i0, icnt, kind); LB: kind = UNIT_G1X | piece << 8 | chunk << XQ_SHIFT
# (a GEMM1 piece exporting its intermediate)
UNIT_G1X = 3
XQ_SHIFT = 20

NW = 4
NT = NW * 64
NTT = NT + 4 * 64
NSK = 4
SLOT = 4 * 1024 + 2 * 256
OPS = 6
KCS = 2
ALOAD_DEPTH = 3
POLL_SLEEP = 16


def mega_moe_tp_shape_supported(model_dim: int, inter_dim: int, tp: int) -> bool:
    return (
        model_dim % 512 == 0
        and (model_dim // 64) % NW == 0
        and model_dim // 256 <= NCHA_MAX
        and model_dim // 256 // gemm2_chunk_groups(model_dim, inter_dim) <= 31
        and inter_dim % 128 == 0
        and inter_dim >= 256
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


@functools.cache
def mega_moe_tp_consts(
    H: int,
    I: int,
    MT: int,
    TMAX: int,
    agr: int = 1,
    dyn_e: int = 0,
    nab: int = 4,
    nch: int = 0,
    a8: bool = False,
    lb: bool = False,
) -> dict:
    RG = MT * 16
    KS1 = H // 128
    KS2 = I // 128
    G2 = H // 64 // NW
    NSC_BLK = ceildiv(RG, 64)
    acb = KCS * (128 if a8 else 64)
    NA_ROWOPS = RG * acb // 1024
    GPC = gemm2_chunk_groups(H, I)
    c = {
        "RG": RG,
        "KS1": KS1,
        "NCH": KS1 // KCS,
        "KS2": KS2,
        "G2": G2,
        "CH1": ceildiv(H // 32, 8),
        "CH2": ceildiv(I // 32, 8),
        "ACB": acb,
        "XB": H if a8 else H // 2,
        "SI_STRIDE": (I if a8 else I // 2) + (8 if lb and not a8 else 16),
        "NA_ROWOPS": NA_ROWOPS,
        "NSC_BLK": NSC_BLK,
        "NA_L": NA_ROWOPS + NSC_BLK * KCS,
        "GPC": GPC,
        "NCK": G2 // GPC,
        "CW": GPC * NW * 64,
        "NAB": nab,
        "NBW": ceildiv(dyn_e, 32),
    }
    off = 0

    def take(n):
        nonlocal off
        off = ceildiv(off, 16) * 16
        start, off = off, off + n
        return start

    c["L_RING"] = take(NW * NSK * SLOT)
    c["L_A"] = take(nab * RG * acb)
    c["L_AS"] = take(nab * KCS * NSC_BLK * 64 * 4)
    c["L_B1"] = c["L_A"]
    c["L_B1S"] = c["L_A"] + RG * c["SI_STRIDE"]
    c["B1_FITS"] = (
        c["L_B1S"] + RG * (I // 32) <= c["L_AS"] + nab * KCS * NSC_BLK * 64 * 4
    )
    c["L_INTER"] = take(
        max(RG * c["SI_STRIDE"], agr * (c["XB"] + H // 32) - RG * (I // 32))
    )
    c["L_INTERS"] = take(RG * (I // 32))
    nrix = 256 if lb else min(TMAX, DYN_MAX)
    c["L_RIX"] = take(nrix * 4)
    c["L_WT"] = take(nrix * 4)
    c["L_CTL"] = take(128 * 4)
    c["L_DYN"] = take((2 * c["NBW"] + dyn_e) * 4)
    c["L_DCNT"] = take(dyn_e * 4) if nch else 0
    c["L_DCH"] = take((nch + 2 * NW) * 4) if nch else 0
    c["L_EOFF"] = take(dyn_e * 4) if lb else 0
    c["L_DPRE"] = take(dyn_e * 4) if lb else 0
    c["LDS_BYTES"] = ceildiv(off, 128) * 128
    assert H // 256 <= NCHA_MAX
    assert KS1 % NSK == 0 and NSK % KCS == 0
    assert (NSK - 1) * OPS <= 63 and c["NA_L"] * (max(ALOAD_DEPTH, nab) - 1) <= 63
    assert not lb or RG <= 256
    assert c["NCK"] <= NCK_MAX and c["NCK"] <= 31
    return c


# LDS control slots (ints at L_CTL); NW-wide ones: C_DONE, C_ASEQ, C_AFREE, C_QBASE (one
# per compute wave), C_MBOX (one per wave), C_UNIT (6: unit range + first record), C_CLAIM.
# C_GEXP: routes key (expert + 1, plus the row chunk) held by L_RIX / L_WT / C_CNT.
C_CNT, C_BARCNT, C_BARGEN, C_LRED, C_PULL, C_EPOCH, C_USEQ, C_UROWS = range(8)
C_NSIG, C_LQ, C_LPUB, C_GEXP, C_DONE = range(8, 13)
C_ASEQ, C_AFREE, C_QBASE, C_MBOX = 16, 20, 24, 28
C_FBITS, C_YAGM, C_NCH, C_AGFREE, C_PLAN, C_NACT = 36, 39, 40, 41, 42, 43
C_ARDY, C_DYNP, C_CDONE, C_UNIT = 44, 45, 46, 48
C_CLAIM, C_PRDY, C_FRDY, C_PBITS, C_RLAND = 56, 60, 61, 62, 63
# LB: C_LBB bank of this launch's LB counters, C_LBN route lists seen, C_G1FIN no more
# GEMM1 units, C_TNL tail row taken, C_G2*: GEMM2 prefetch ring (C_G2U: 2 x 4 ints),
# C_PFW: per compute wave, the ring holds the next unit's first slots, C_LBJ2: first row
# chunk run as single-block pieces (LB mix).
C_LBB, C_LBN, C_G1FIN, C_TNL = 64, 65, 67, 69
C_G2RDY, C_G2FREE, C_G2END, C_G2V, C_G2U = 76, 77, 78, 79, 80
C_G2LAST, C_G2VN, C_PFW, C_LBJ2 = 88, 89, 90, 94
# per-CTA unit list (dynamic / LB): UL_MAX records of 4 ints
C_UL = 96
UL_MAX = (128 - C_UL) // 4


class KernelCtx(types.SimpleNamespace):
    """Compile-time names of one kernel instance: the compile_mega_moe_tp arguments,
    the values derived from them and the device functions of the parts built so far."""

    def get(self, names: str) -> tuple:
        return tuple(getattr(self, n) for n in names.split())

    def add(self, env: dict) -> None:
        vars(self).update((n, v) for n, v in env.items() if n != "kc")


def kernel_config(
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
    route_fp8: bool = False,
    agr: int = 1,
    tp: int = MAX_TP,
    ar: bool = False,
    nab: int = 4,
    npp: int = 1,
    ll_rs: bool = False,
    ll_route: bool = False,
    comm_bf16: bool = False,
    ag8: bool = False,
    rch: int = 16,
    nch: int = 0,
    a8: bool = False,
    lb: bool = False,
    lbq: int = 1,
    tn: int = 0,
    tn_eps: float = 1e-6,
    tn_gemma: bool = True,
    lbpf: bool = False,
    lbp: int = 0,
):
    """The derived compile-time values of one compile_mega_moe_tp instance."""
    if swiglu_limit is None:
        swiglu_limit = 7.0 if act == "swiglu" else float("inf")
    TN = tn in (1, 2)
    assert not TN or not ar
    nsk = NSK
    c = mega_moe_tp_consts(H, I, MT, TMAX, agr, E, nab, nch, a8, lb)
    RG, KS1, NCH, KS2, G2 = c["RG"], c["KS1"], c["NCH"], c["KS2"], c["G2"]
    CH1, CH2, SI_STRIDE = c["CH1"], c["CH2"], c["SI_STRIDE"]
    ACB, XB = c["ACB"], c["XB"]
    NA_ROWOPS, NSC_BLK, NA_L = c["NA_ROWOPS"], c["NSC_BLK"], c["NA_L"]
    GPC, NCK, CW, NAB, NBW = c["GPC"], c["NCK"], c["CW"], c["NAB"], c["NBW"]
    L_RING, L_A, L_AS, L_DYN = c["L_RING"], c["L_A"], c["L_AS"], c["L_DYN"]
    L_INTER, L_INTERS, L_RIX, L_WT = c["L_INTER"], c["L_INTERS"], c["L_RIX"], c["L_WT"]
    L_CTL, LDS_BYTES = c["L_CTL"], c["LDS_BYTES"]
    L_B1, L_B1S = c["L_B1"], c["L_B1S"]
    L_DCNT, L_DCH = c["L_DCNT"], c["L_DCH"]
    L_EOFF, L_DPRE = c["L_EOFF"], c["L_DPRE"]
    L_WSUM = L_DCH + nch * 4
    assert npp >= 1 and nsk % (npp * KCS) == 0 and (KS1 * npp) % nsk == 0
    assert 3 <= NAB <= 4
    VPL = ceildiv(CW // 8, 64)
    SCAN_IT = ceildiv(min(TMAX, DYN_MAX) * TOPK // 4, NT)
    SCAN_G = 16
    NPC = 1 if lb else KS2
    assert not (comm_bf16 and ll_rs)
    DYN_PS = [p for p in range(1, KS2 + 1) if KS2 % p == 0]
    DLL = bool(ll_rs and ll_route)
    assert not DLL or route_fp8
    MLL = DLL and not ar
    ARLL = DLL and ar
    AG8 = bool(ag8 and ar and not ARLL)
    NV = NCK
    TPC = tp
    # routing-meta flag slots per rank: one per sending CTA (64 lanes x 4 routes each)
    NMETA = max(32, 1 << (ceildiv(TMAX // tp * TOPK, 256) - 1).bit_length())
    assert NMETA <= NMETA_CAP, "m * topk > 64 * NMETA_CAP"
    assert not lb or (not ll_rs and not a8 and rch == RG)
    assert not lb or nch <= NCHLB_MAX
    LBQ = lbq if lb else 1
    assert NCK % LBQ == 0
    assert not lb or LBQ * GPC % gemm2_group_step(H, I) == 0
    NCG = NCK // LBQ
    CGW = [LBQ] * NCG
    _cg = lb and NCK == 12 and LBQ == 3
    if _cg:
        CGW = [3, 3, 3, 2, 1]
        NCG = len(CGW)
    LB_B = min(ceildiv(TMAX * TOPK // 4, NT), 10)
    LB_PS = [p for p in range(1, KS2 + 1) if KS2 % p == 0]
    if lbp and lbp in LB_PS:
        LB_PS = [lbp]
    # LB mix: one round of 2-piece GEMM1 units; the chunks past it run as KS2 single-block
    # pieces, H2 = KS2 // 2 (one 2-piece unit's work) on each CTA without a unit, one more
    # on the first H2 * X CTAs
    H2 = KS2 // 2
    LBMIX = (
        lb
        and MT <= 3
        and len(LB_PS) > 1
        and 2 in LB_PS
        and KS2 % 2 == 0
        and H2 < UL_MAX
    )
    LBMIX_DIV = 4
    XLPR = (256 if a8 else 128) // 16
    assert agr <= 64 // XLPR
    AUX_RT = AUX_SYS if lb else AUX_SC1
    ADEPTH = min(ALOAD_DEPTH, NAB - 1) if lb and NAB > 3 else ALOAD_DEPTH
    LBPF = bool(lb and c["B1_FITS"] and lbpf)
    MTSKIP = lb and MT >= 4
    name = (
        f"mega_moe_tp_fused_h{H}_i{I}_e{E}_k{TOPK}_mt{MT}_t{TMAX}_{act}"
        + (f"_lim{swiglu_limit:g}" if math.isfinite(swiglu_limit) else "")
        + (f"_p{NPC}" if NPC > 1 else "")
        + ("_r8" if route_fp8 else "")
        + f"_ag{agr}_tp{tp}"
        + ("_ar" if ar else "")
        + (f"_nab{NAB}" if NAB != 4 else "")
        + (f"_npp{npp}" if npp > 1 else "")
        + ("_ll" if ll_rs else "")
        + ("_llr" if DLL else "")
        + ("_cb16" if comm_bf16 else "")
        + ("_ag8" if AG8 else "")
        + ("_a8" if a8 else "")
        + ("_lb" if lb else "")
        + (f"_q{LBQ}" if LBQ > 1 else "")
        + (("_cg" + "x".join(str(w) for w in CGW)) if _cg else "")
        + (f"_lp{LB_PS[0]}" if lb and len(LB_PS) == 1 else "")
        + (
            f"_tn{tn}e{struct.unpack('<I', struct.pack('<f', tn_eps))[0]:x}"
            if tn
            else ""
        )
        + ("_rms" if tn and not tn_gemma else "")
        + (f"_mix{LBMIX_DIV}" if LBMIX else "")
        + ("_mtskip" if MTSKIP else "")
        + (f"_ad{ADEPTH}" if ADEPTH != ALOAD_DEPTH else "")
        + ("_pf" if LBPF else "")
    )
    const_expr = fx.const_expr
    layout_tag = f"{CTRL_INTS}/{FLAG_INTS}/{DEADLINE}/{NCTA_MAX}"
    return KernelCtx(**locals())
