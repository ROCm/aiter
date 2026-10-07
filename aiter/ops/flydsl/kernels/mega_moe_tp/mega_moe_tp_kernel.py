# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations

import functools
import math
import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ..kernels_common import ceildiv
from ..mxfp4_gemm_common import _activation_mul_batch, _e8m0_from_amax
from .common import (
    ACQ,
    ACQ_REL,
    AUX_SC1,
    AUX_SYS,
    DEADLINE,
    MAX_TP,
    REL,
    V2I,
    V4I,
    _ctpop,
    _u,
    alive,
    amax,
    asm,
    bf16x8_to_f32,
    bld,
    bst,
    cat8,
    dma4,
    dma16,
    e8_scale,
    fp4_pack,
    fp8x4_pack,
    fp8x4_unpack,
    fp8x8_decode,
    g_add_agent,
    g_ld_i32,
    g_ld_rel,
    g_ld_sys,
    g_or_agent,
    g_st_sys,
    i32,
    lds_atomic_add,
    lds_atomic_or,
    lds_cas,
    lds_ld,
    lds_ld_acq,
    lds_ld_i32,
    lds_st,
    lds_st_rel,
    mfma,
    mxfp8x8,
    now,
    pack_bf16x2,
    pack_bf16x8,
    poll_sys_ge,
    rsrc,
    scales4,
    spin0,
    spin_lds_ge,
    sum_live,
    swap16,
    swap32,
    traced,
    uni,
    wait_lgkm0,
    wait_vm,
    wave_rank,
    wave_red,
)

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

DYN_MAX = 256
NCK_MAX = 64
NCHA_MAX = 64
NCTA_MAX = 256
FLAG_RDY = 0
FLAG_AGM = FLAG_RDY + MAX_TP * NCK_MAX
FLAG_PRE = FLAG_AGM + MAX_TP * 32
PRE_CH = 4
NPRE_MAX = 16
FLAG_YAG = FLAG_PRE + NPRE_MAX * MAX_TP * NCTA_MAX
FLAG_AGQ = FLAG_YAG + NCK_MAX * MAX_TP * NCTA_MAX
TN_MAX = 2048
FLAG_TN = FLAG_AGQ + NCHA_MAX * MAX_TP
FLAG_INTS = FLAG_TN + MAX_TP * TN_MAX

CTRL_ERR = 20
CTRL_XF = 32
CTRL_CNT = 64
ERR_FLAG, ERR_META, ERR_CHUNK, ERR_COMM, ERR_YAG = 1, 2, 4, 8, 16
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
NCHLB_MAX = 2048
G1C_STRIDE = 16
CTRL_LBSEQ = CTRL_CLM + 2 * LRDY_STRIDE
CTRL_LBR = CTRL_LBSEQ + LRDY_STRIDE
CTRL_COLC = CTRL_LBR + 2 * N_XCD * LRDY_STRIDE
CTRL_G1C = CTRL_COLC + 2 * NCK_MAX * LRDY_STRIDE
CTRL_TNC = CTRL_G1C + 2 * NCHLB_MAX * G1C_STRIDE
CTRL_INTS = CTRL_TNC + TN_MAX
UNIT_REC = 8
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

    c["L_RING"] = take(NW * nsk * SLOT)
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
    nrix = 256 if lb else min(TMAX, DYN_MAX) if dyn_e else TMAX
    c["L_RIX"] = take(nrix * 4)
    c["L_WT"] = take(nrix * 4)
    c["L_CTL"] = take(128 * 4)
    c["L_DYN"] = take((2 * c["NBW"] + dyn_e) * 4)
    c["L_DCNT"] = take(dyn_e * 4) if nch else 0
    c["L_DCH"] = take((nch + 2 * NW) * 4) if nch else 0
    c["L_EOFF"] = take(dyn_e * 4) if lb else 0
    c["L_DPRE"] = take(dyn_e * 4) if lb else 0
    c["L_CLS"] = take(ceildiv(TMAX, 32) * 4)
    c["LDS_BYTES"] = ceildiv(off, 128) * 128
    assert H // 256 <= NCHA_MAX
    assert KS1 % nsk == 0 and nsk % KCS == 0 and nsk >= NSK
    assert (nsk - 1) * OPS <= 63 and c["NA_L"] * (max(ALOAD_DEPTH, nab) - 1) <= 63
    assert not lb or RG <= 256
    assert c["NCK"] <= NCK_MAX and c["NCK"] <= 31
    return c


C_CNT, C_BARCNT, C_BARGEN, C_LRED, C_PULL, C_EPOCH = 0, 1, 2, 3, 4, 5
C_USEQ, C_UROWS, C_NSIG, C_LQ, C_LPUB = 6, 7, 8, 9, 10
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
C_LBB, C_LBN = C_XC, C_XC + 1
C_UL = 96
UL_MAX = (128 - C_UL) // 4
C_G1FIN = 67
C_TNL = 69
C_G2RDY, C_G2FREE, C_G2END, C_G2V, C_G2U = 76, 77, 78, 79, 80
C_G2LAST = 88
C_G2VN, C_PFW = 89, 90
C_LBJ2 = 94
C_ZDONE = 95


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
    DYN = dyn_e > 0
    CHUNK = DYN and rch > 0 and nch > 0
    if swiglu_limit is None:
        swiglu_limit = 7.0 if act == "swiglu" else float("inf")
    A8 = bool(a8)
    LB = bool(lb)
    TN = int(tn) in (1, 2)
    TNB = int(tn) == 2
    assert not TN or (not ar and not xrep)
    c = mega_moe_tp_consts(
        H, I, MT, TMAX, agr, dyn_e, nab, nsk, nch if CHUNK else 0, A8, LB
    )
    RG, KS1, NCH, KS2, G2 = c["RG"], c["KS1"], c["NCH"], c["KS2"], c["G2"]
    CH1, CH2, SI_STRIDE = c["CH1"], c["CH2"], c["SI_STRIDE"]
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
    VPL = ceildiv(CW // 8, 64)
    SCAN_TOK = min(TMAX, DYN_MAX) if DYN else TMAX
    SCAN_IT = ceildiv(SCAN_TOK * TOPK // 4, NT)
    SCAN_G = 16
    NPC = 1 if LB else (KS2 if DYN else max(npieces, 1))
    assert not DYN or xsplit == 0
    FP8R = bool(route_fp8)
    LL = bool(ll_rs)
    CB16 = bool(comm_bf16)
    assert not (CB16 and LL)
    TB = 1 if DYN else max(1, (RED_INFLIGHT // 2 if NPC > 2 else RED_INFLIGHT) // TOPK)
    DYN_PS = [p for p in range(1, KS2 + 1) if KS2 % p == 0]
    DLL = LL and bool(ll_route) and (DYN or NPC == 1)
    assert not DLL or FP8R
    XREP = bool(xrep)
    RREP = bool(ar) or XREP
    ZMA = LB and not RREP and not DLL
    AIN = bool(ar) and not XREP
    MLL = DLL and not RREP
    ARLL = DLL and bool(ar)
    AG8 = bool(ag8) and bool(ar) and not ARLL
    VB = DYN and bool(vb)

    def _g2_step(n):
        nsk2 = _nsk2_for(n, G2)
        return nsk2 * n // math.gcd(nsk2, n) // n

    DX_STEP = math.lcm(*[_g2_step(KS2 // q) for q in DYN_PS]) if DYN else G2
    DX_CG = ceildiv((G2 + 1) // 2, DX_STEP) * DX_STEP
    DX_NG = ceildiv(G2, DX_CG)
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
    assert not A8 or (DYN and not DX and not XSPLIT)
    assert not LB or (CHUNK and not LL and not XSPLIT and not A8 and RCH == RG)
    assert not LB or NCH_MAX <= NCHLB_MAX
    LBQ = int(lbq) if LB else 1
    assert NCK % LBQ == 0
    assert not LB or LBQ * GPC % gemm2_group_step(H, I) == 0
    NCG = NCK // LBQ
    CGW = [LBQ] * NCG
    _cg = LB and NCK == 12 and LBQ == 3
    if _cg:
        CGW = [3, 3, 3, 2, 1]
        NCG = len(CGW)
    CGB = [sum(CGW[:k]) for k in range(NCG + 1)]
    XG_LIST = TMAX * TOPK * (I // 2 + I // 32)
    RBITS = TMAX.bit_length()
    LB_IT = ceildiv(TMAX * TOPK // 4, NT)
    LB_B = min(LB_IT, 10)
    LB_PS = [p for p in range(1, KS2 + 1) if KS2 % p == 0]
    if lbp and int(lbp) in LB_PS:
        LB_PS = [int(lbp)]
    LBMIX = LB and MT <= 3 and len(LB_PS) > 1 and 2 in LB_PS and KS2 % 2 == 0
    LBMIX_DIV = 4
    XLPR = (256 if A8 else 128) // 16
    assert AGR <= 64 // XLPR
    AUX_RT = AUX_SYS if LB else AUX_SC1
    ADEPTH = min(ALOAD_DEPTH, NAB - 1) if LB and NAB > 3 else ALOAD_DEPTH
    LBPF = LB and c["B1_FITS"] and bool(lbpf)
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
        + (f"_ch{RCH}n{NCH_MAX}" if CHUNK else "")
        + (f"_nab{NAB}" if NAB != 4 else "")
        + (f"_nsk{nsk}" if nsk != NSK else "")
        + (f"_npp{npp}" if npp > 1 else "")
        + ("_ll" if LL else "")
        + ("_llr" if DLL else "")
        + (f"_xl{xl_s0}" if XL else "")
        + ("_cb16" if CB16 else "")
        + ("_xrep" if XREP else "")
        + ("_ag8" if AG8 else "")
        + (f"_dx{DX_NG}" if DX else "")
        + ("_novb" if DYN and not VB else "")
        + ("_a8" if A8 else "")
        + ("_lb" if LB else "")
        + (f"_q{LBQ}" if LBQ > 1 else "")
        + (("_cg" + "x".join(str(w) for w in CGW)) if _cg else "")
        + (f"_lp{LB_PS[0]}" if LB and len(LB_PS) == 1 else "")
        + (
            f"_tn{int(tn)}e{struct.unpack('<I', struct.pack('<f', tn_eps))[0]:x}"
            if tn
            else ""
        )
        + ("_rms" if tn and not tn_gemma else "")
        + ("_zma" if ZMA else "")
        + (f"_mix{LBMIX_DIV}" if LBMIX else "")
        + ("_mtskip" if MTSKIP else "")
        + (f"_ad{ADEPTH}" if ADEPTH != ALOAD_DEPTH else "")
        + ("_pf" if LBPF else "")
    )
    const_expr = fx.const_expr
    layout_tag = f"{CTRL_INTS}/{FLAG_INTS}/{DEADLINE}/{UNIT_REC}/{NCTA_MAX}"

    def ctrl_at(a, idx):
        return fx.Int64(a["ctrl"]) + fx.Int64(i32(idx) * i32(4))

    def lrdy_at(a, c):
        return ctrl_at(a, i32(CTRL_LRDY) + c * i32(LRDY_STRIDE))

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

    def a_frag(abuf, row, k, q4):
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

    def report(a, code):
        g_or_agent(ctrl_at(a, CTRL_ERR), code)

    @traced
    def _report_if(a, bad, code):
        if bad:
            report(a, code)

    @traced
    def poll_zero(fn, a=None, lane=None, err=0):
        pend = fn()
        t0 = now()
        while (pend != i32(0)) & alive(t0):
            rocdl.s_sleep(1)
            pend = fn()
        if const_expr(err != 0):
            _report_if(a, (pend != i32(0)) & (lane == i32(0)), err)

    @traced
    def spin_sys_ge(addr, target, a=None):
        cur = poll_sys_ge(addr, target)
        if const_expr(a is not None):
            _report_if(a, cur < target, ERR_FLAG)

    def masked(e):
        return fx.Int32(e).bitcast(fx.Uint32) >= fx.Uint32(E)

    @traced
    def gather_routes(L, tid, ids_addr, tw_addr, ttot, expert):
        key = expert + i32(1)
        if lds_ld_i32(L, L_CTL + C_GEXP * 4) != key:
            _gather_scan(L, tid, ids_addr, tw_addr, ttot, expert, key)
        return fx.min(lds_ld_i32(L, L_CTL + C_CNT * 4), i32(TMAX))

    @traced
    def gather_routes_chunk(L, tid, ids_addr, tw_addr, ttot, expert, cc, ce):
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
        if chunk is None:
            return fx.Boolean(True)
        cc, ce, inv = chunk
        t = idx // i32(TOPK)
        q = ((t.to(fx.Float32) + fx.Float32(0.5)) * inv).to(fx.Int32)
        return (t - q * ce) == cc

    @traced
    def _gather_scan(L, tid, ids_addr, tw_addr, ttot, expert, key, chunk=None):
        cbar(L, tid)
        if tid == i32(0):
            lds_st(L, L_CTL + C_CNT * 4, i32(0))
        cbar(L, tid)
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
                        pos, n = wave_rank(hit)
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
        if tid == i32(0):
            lds_st(L, L_CTL + C_GEXP * 4, key)
        cbar(L, tid)

    @traced
    def zero_masked(L, lane, a, w0, nw):
        n = a["ttot"] * i32(TOPK)
        nblk = i32(gpu.grid_dim.x)
        per = ceildiv(n, nblk)
        lo = i32(gpu.block_id("x")) * per
        hi = fx.min(lo + per, n)
        rid = rsrc(a["ids"], n * i32(16 if MLL else 4))
        if const_expr(MLL):
            spin0(L, lane, L_CTL + C_RLAND * 4, i32(1))
        for b_ in range(lo + i32(w0 * 64), hi, i32(nw * 64)):
            i = i32(b_) + lane
            off = fx.min(i, n - i32(1)) * i32(16 if MLL else 4)
            e = fx.Int32(bld(rid, off, 0, T.i32, 0))
            bal = fx.Int64(rocdl.ballot(T.i64, (i < hi) & masked(e)))
            if bal != fx.Int64(0):
                for j_ in range(i32(0), i32(64), i32(1)):
                    j = i32(j_)
                    if ((bal >> fx.Int64(j)) & fx.Int64(1)) != fx.Int64(0):
                        _zero_route(a, lane, i32(b_) + j)
                wait_vm(0)

    @traced
    def zma_run(L, lane, a):
        poll_zero(lambda: _meta_pending(a, lane, a["epoch"]))
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
            # a spin loop the compiler cannot see: one it sees makes it drain
            # the in-flight weight loads (vmcnt(0)) before every A chunk
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
            # branch-free: a branch splits the GEMM1 step loop, which also
            # drains the weight loads
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
        if const_expr(npp > 1):
            return ((nnb % i32(npp)) == i32(0)).select(i32(npp), i32(1))
        return i32(1)

    @traced
    def _gemm1_disp(L, tid, a, expert, i0, nnb, mte=None):
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
            if uni(rows) <= i32(3 * 16):
                _gemm1_disp(L, tid, a, expert, i0, nnb, 3)
            else:
                _gemm1_disp(L, tid, a, expert, i0, nnb)
        elif const_expr(VB):
            if (nnb == i32(1)) & (lds_ld_i32(L, L_CTL + C_VBON * 4) != i32(0)):
                _gemm1_vb(L, tid, a, expert, i0, nnb)
            else:
                _gemm1(L, tid, a, expert, i0, nnb, 1)
        else:
            _gemm1_disp(L, tid, a, expert, i0, nnb)
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
                        wait_vm((nsk - 1) * OPS)
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
    def _gemm1_vb(L, tid, a, expert, i0, nnb, P=1):
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
            return [bld(rw, v, so, V4I, 0) for v in (vg0, vg1, vu0, vu1)] + [
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
            spin0(L, lane, L_CTL + C_PLAN * 4, i32(1))
        ub = uni(lds_ld_acq(L, L_CTL + C_UNIT * 4))
        ue = uni(lds_ld_acq(L, L_CTL + (C_UNIT + 1) * 4))
        for u_ in range(ub, ue, i32(1)):
            u = i32(u_)
            if lane == i32(0):
                spin_lds_ge(L, L_CTL + C_USEQ * 4, u - ub + i32(1))
            _a_unit(L, lane, a, rx, rxs, u, ub)

    @traced
    def _a_unit(L, lane, a, rx, rxs, u, ub):
        R = uni(lds_ld_acq(L, L_CTL + C_UROWS * 4))
        nnb = unit_fields(L, a, u, ub)[2] // i32(128)
        for r0_ in range(i32(0), R, i32(RG)):
            r0 = i32(r0_)
            rows = fx.min(R - r0, i32(RG))
            arow = []
            for j in range_constexpr(NA_ROWOPS):
                if const_expr(A8):
                    row = i32(j * 4) + lane // i32(16)
                    col = (lane % i32(16)) ^ (row & i32(7))
                else:
                    row = i32(j * 8) + lane // i32(8)
                    col = (lane % i32(8)) ^ (row & i32(7))
                rr = (row < rows).select(row, i32(0))
                arow.append(
                    (lds_ld_i32(L, L_RIX + (r0 + rr) * i32(4)) // i32(TOPK)) * i32(XB)
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
                if (lane == i32(0)) & (q >= i32(NAB)):
                    spin_lds_ge(
                        L,
                        L_CTL + C_AFREE * 4 + b * i32(4),
                        i32(NW) * (q // i32(NAB)),
                    )
                rocdl.sched_barrier(0)
                cc = cidx % i32(NCH)
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
                        dma4(dst, rxs, ascl[j], cc * i32(8) + i32(k * 4), sys=True)
                _loader_advance(L, lane, q)
            wait_vm(0)
            _loader_flush(L, lane)

    @traced
    def _loader_advance(L, lane, q):
        pub = lds_ld_i32(L, L_CTL + C_LPUB * 4)
        if q + i32(1) - pub >= i32(ADEPTH):
            wait_vm(NA_L * (ADEPTH - 1))
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
        n0 = (w + i32(NW) * gi) * i32(64)
        for t in range_constexpr(4):
            dma16(
                slot + i32(t * 1024),
                rsrc(a["w2"]),
                lane * i32(16),
                e * i32(H * (I // 2))
                + (n0 + i32(t * 16)) * i32(I // 2)
                + k * i32(1024),
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
            if lds_ld_i32(L, L_CTL + C_VBON * 4) != i32(0):
                _gemm2(*args, vb2=True, **kw)
            else:
                _gemm2(*args, **kw)
        elif const_expr(MTSKIP):
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
            b0 = buf == i32(0)
            ib0 = b0.select(i32(L_INTER), i32(L_B1))
            is0 = b0.select(i32(L_INTERS), i32(L_B1S))
            rix0 = i32(L_RIX) + buf * i32(128 * 4)
            wt0 = i32(L_WT) + buf * i32(128 * 4)
        NSK2 = _nsk2_for(NKS, G2)
        GPI = (NSK2 * NKS // math.gcd(NSK2, NKS)) // NKS
        assert G2 % GPI == 0
        WAIT_B2 = (NSK2 - 1) * OPS
        LAG = max(1, ceildiv(NSK2, GPC * NKS))
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

        XPF = nxt is not None and not VB2

        def issue_x(q, slot_idx):
            tq = g_hi * i32(NKS)
            if q == tq:
                vn = lds_ld_acq(L, L_CTL + C_G2VN * 4)
                ok = (vn == nxt + i32(2)) & (
                    lds_ld_i32(L, L_CTL + C_G2V * 4)
                    < lds_ld_i32(L, L_CTL + C_NCH * 4) * i32(NCG)
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
            g2_slot_dma(
                a,
                lane,
                w,
                ring + i32(slot_idx * SLOT),
                ee,
                gi,
                ks0 + qc - gi * i32(NKS),
            )

        def issue(q, slot_idx):
            if const_expr(XPF):
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
                        0,
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
                            if const_expr(j > 0):  # noqa: SIM114
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
                                    is0 + row * i32(I // 32) + i32(k * 4) + q4,
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
            for gi0c in range_constexpr(0, span, GPI):
                rr = groups(g_lo + i32(gi0c), rr)
        else:
            for gi0_ in range(g_lo, g_hi, i32(GPI)):
                if const_expr(progress is not None):  # noqa: SIM102
                    if i32(gi0_) + i32(GPI) >= g_hi:
                        progress()
                groups(i32(gi0_), rr)
        wait_vm(0)
        if const_expr(not DLL):
            _final_report(L, lane, w, signal)
            if const_expr(colsig is not None):  # noqa: SIM102
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
            asm("buffer_inv sc0")  # no wrapper
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
        if const_expr(RREP and not DLL):
            zero_masked(L, tid % i32(64), a, tid // i32(64), NW)
        cbar(L, tid)
        if const_expr(MLL):  # noqa: SIM102
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
            if const_expr(DX):
                _dx_flag(a, tid, kind)
            if const_expr(XSPLIT):
                _xq_flag(a, tid, i0, icnt, kind)
                if const_expr(not DLL):
                    _xcol_empty(a, tid, kind, i0, R)
        if const_expr(XSPLIT):
            nblk = i32(gpu.grid_dim.x)
            cap = ceildiv(a["ncol"], nblk) * i32(2) + i32(2)
            col_claim(L, tid, a)
            for it_ in range(i32(0), cap, i32(1)):
                _col_unit(L, tid, a, i32(it_) < cap - i32(1))
        if const_expr(LBPF):
            lb_g2_consume(L, tid, a)
        elif const_expr(LB):
            lb_g2_phase(L, tid, a)
        if const_expr(DX):  # noqa: SIM102
            if lds_ld_i32(L, L_CTL + C_DXON * 4) != i32(0):
                dx_cols(L, tid, a)
        if const_expr(DLL):
            _cdone(tid, L)

    @traced
    def unit_tile_dx(L, tid, a, expert, i0, icnt, kind, r0, rows, sig):
        if (kind & i32(0xFF)) == i32(UNIT_G1X):
            gemm1(L, tid, a, expert, i0, icnt // i32(128))
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
        P = lds_ld_i32(L, L_CTL + C_DYNP * 4)
        ncol = lds_ld_i32(L, L_CTL + C_NACT * 4) * P * i32(DX_NG)
        nblk = i32(gpu.grid_dim.x)
        cap = ceildiv(ncol, nblk) * i32(2) + i32(2)
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
                    pre=functools.partial(xq_load, L, tid, a, r0, rows, i0, icnt),
                    span=DX_CG if DX_CG * DX_NG == G2 else None,
                )

    @traced
    def xq_load(L, tid, a, r0, rows, i0=None, icnt=None):
        icnt = I if icnt is None else icnt
        ag_stage_free(L, tid % i32(64), a)
        cbar(L, tid)
        u16 = icnt // 32
        rx, rxs = xg_rs(a)
        for q_ in range(tid, rows * i32(u16), i32(NT)):
            q = i32(q_)
            row = q // i32(u16)
            c = q - row * i32(u16)
            rix = lds_ld_i32(L, L_RIX + (r0 + row) * i32(4))
            off = rix * i32(I // 2) + c * i32(16)
            off = off if i0 is None else off + i0 // i32(2)
            v = bld(rx, off, 0, V4I, AUX_SYS)
            lds_st(L, i32(L_INTER) + row * i32(SI_STRIDE) + c * i32(16), v, 16)
        nsd = icnt // 128
        for q_ in range(tid, rows * i32(nsd), i32(NT)):
            q = i32(q_)
            row = q // i32(nsd)
            c = q - row * i32(nsd)
            rix = lds_ld_i32(L, L_RIX + (r0 + row) * i32(4))
            off = rix * i32(I // 32) + c * i32(4)
            off = off if i0 is None else off + i0 // i32(32)
            sv = fx.Int32(bld(rxs, off, 0, T.i32, AUX_SYS))
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
        _dyn_prefix(L, tid)
        cbar(L, tid)
        for it in range_constexpr(ceildiv(NE, NT)):
            _dyn_list(L, tid + i32(it * NT))
        cbar(L, tid)
        if const_expr(CHUNK):
            _chunk_table(L, tid)
            cbar(L, tid)
        _dyn_units(L, tid)
        cbar(L, tid)

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

    def _pick_p(nun, C, ps):
        P = i32(ps[0])
        best = ceildiv(nun * i32(ps[0]), C) * i32(KS2 // ps[0])
        for q in ps[1:]:
            cost = ceildiv(nun * i32(q), C) * i32(KS2 // q)
            better = (cost < best) & (nun * i32(q) <= C * i32(UL_MAX))
            P = better.select(i32(q), P)
            best = better.select(cost, best)
        return P

    def _xcd_rank(C):
        bid = i32(gpu.block_id("x"))
        return (C % i32(N_XCD) == i32(0)).select(
            (bid % i32(N_XCD)) * (C // i32(N_XCD)) + bid // i32(N_XCD), bid
        )

    @traced
    def _dyn_units(L, tid):
        if tid == i32(0):
            nact = lds_ld_i32(L, L_CTL + C_NACT * 4)
            nun = nact
            if const_expr(CHUNK):
                nun = lds_ld_i32(L, L_CTL + C_NCH * 4)
            C = i32(gpu.grid_dim.x)
            P = _pick_p(nun, C, DYN_PS)
            U = nun * P
            bid = _xcd_rank(C)
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
                    lds_st(L, ul, lds_ld_i32(L, L_DYN + (i32(2 * NBW) + j) * i32(4)))
                lds_st(L, ul + i32(4), (u - j * P) * icnt)
                lds_st(L, ul + i32(8), icnt)
                kind = i32(UNIT_G1X) | ((u - j * P) << i32(8)) | (j << i32(XQ_SHIFT))
                lds_st(L, ul + i32(12), use_x.select(kind, i32(0)))
            lds_st(L, L_CTL + C_DYNP * 4, P)
            lds_st(L, L_CTL + C_UNIT * 4, i32(0))
            lds_st(L, L_CTL + (C_UNIT + 1) * 4, u_hi - u_lo)
            lds_st_rel(L, L_CTL + C_PLAN * 4, i32(1))

    def lb_g1c(a, j, bank):
        return ctrl_at(a, i32(CTRL_G1C) + (bank * i32(NCHLB_MAX) + j) * i32(G1C_STRIDE))

    def lb_colc(a, c, bank):
        return ctrl_at(a, i32(CTRL_COLC) + (bank * i32(NCK_MAX) + c) * i32(LRDY_STRIDE))

    def lb_lbr(a, bank, x):
        return ctrl_at(a, i32(CTRL_LBR) + (bank * i32(N_XCD) + x) * i32(LRDY_STRIDE))

    def lb_lbr_wait(a, bank, x):
        nblk = i32(gpu.grid_dim.x)
        spin_sys_ge(lb_lbr(a, bank, x), ceildiv(nblk - x, i32(N_XCD)), a)

    @traced
    def lb_none(L, tid, a, nun):
        if (nun == i32(0)) & (tid == i32(0)) & (i32(gpu.block_id("x")) == i32(0)):
            lb_lists_wait(a, lb_bank(L))
            for c in range_constexpr(NCK):
                g_st_sys(lrdy_at(a, i32(c)), a["epoch"])

    @traced
    def lb_col_done(L, a, cc, nun, bank):
        if const_expr(ZMA):
            spin_lds_ge(L, L_CTL + C_ZDONE * 4, i32(1))
        if g_add_agent(lb_colc(a, cc, bank), 1) == nun - i32(1):
            for q_ in range(cg_lo(cc), cg_lo(cc + i32(1)), i32(1)):
                g_st_sys(lrdy_at(a, i32(q_)), a["epoch"])

    @traced
    def lb_lists_wait(a, bank):
        for x in range_constexpr(N_XCD):
            lb_lbr_wait(a, bank, i32(x))

    def lb_lists(a):
        base = fx.Int64(a["xg"]) + fx.Int64(XG_LIST)
        return rsrc(base), rsrc(base + fx.Int64(TMAX * TOPK * 4))

    def lb_bank(L):
        return lds_ld_i32(L, L_CTL + C_LBB * 4)

    @traced
    def lb_plan(L, tid, a):
        n = a["ttot"] * i32(TOPK)
        nblk = i32(gpu.grid_dim.x)
        per = ceildiv(n, nblk)
        lo = fx.min(i32(gpu.block_id("x")) * per, n)
        hi = fx.min(lo + per, n)
        if tid == i32(0):
            lds_st(L, L_CTL + C_LBB * 4, g_ld_sys(ctrl_at(a, i32(CTRL_LBSEQ))) & i32(1))
        rid = rsrc(a["ids"], n * i32(4))
        n4 = n // i32(4)
        for r0_ in range(i32(0), n4, i32(NT * LB_B)):
            r0 = i32(r0_)
            vs = [
                fx.Vector(
                    bld(
                        rid,
                        fx.min(r0 + tid + i32(it * NT), n4 - i32(1)) * i32(16),
                        0,
                        V4I,
                        0,
                    )
                )
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
        _lb_bitmap(L, tid)
        cbar(L, tid)
        _dyn_prefix(L, tid)
        cbar(L, tid)
        for it in range_constexpr(ceildiv(NE, NT)):
            _dyn_list(L, tid + i32(it * NT))
        cbar(L, tid)
        _chunk_table(L, tid, True)
        lb_scatter(L, tid, a, lo, hi)
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
        lane = tid % i32(64)
        w = tid // i32(64)
        for rnd in range_constexpr(ceildiv(NE, NT)):
            e = tid + i32(rnd * NT)
            c = (e < i32(NE)).select(
                lds_ld_i32(L, L_DCNT + fx.min(e, i32(NE - 1)) * i32(4)), i32(0)
            )
            b = fx.Int64(rocdl.ballot(T.i64, c > i32(0)))
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

    def _wave_prefix(v, bits):
        pre, tot = i32(0), i32(0)
        for b in range_constexpr(bits):
            below, cnt = wave_rank(((v >> i32(b)) & i32(1)) == i32(1))
            pre = pre + (below << i32(b))
            tot = tot + (cnt << i32(b))
        return pre, tot

    @traced
    def _chunk_table(L, tid, lb=False):
        nact = lds_ld_i32(L, L_CTL + C_NACT * 4)
        w = tid // i32(64)
        lane = tid % i32(64)
        base = [i32(0), i32(0)]
        for rnd in range_constexpr(ceildiv(NE, NT)):
            j = tid + i32(rnd * NT)
            live = j < nact
            e = live.select(
                lds_ld_i32(L, L_DYN + (i32(2 * NBW) + fx.min(j, i32(NE - 1))) * i32(4)),
                i32(0),
            )
            r = lds_ld_i32(L, L_DCNT + e * i32(4))
            r = live.select(r, i32(0)) if lb else r
            ce = fx.min(fx.max(ceildiv(r, i32(RCH)), i32(1)), i32(255))
            ce = live.select(ce, i32(0))
            sums = [_wave_prefix(ce, 8)] + ([_wave_prefix(r, RBITS)] if lb else [])
            if lane == i32(0):
                for q in range_constexpr(len(sums)):
                    lds_st(L, L_WSUM + (w + i32(q * NW)) * i32(4), sums[q][1])
            cbar(L, tid)
            off = []
            for q in range_constexpr(len(sums)):
                o, total = base[q] + sums[q][0], i32(0)
                for v in range_constexpr(NW):
                    sv = lds_ld_i32(L, L_WSUM + i32((q * NW + v) * 4))
                    o = o + (i32(v) < w).select(sv, i32(0))
                    total = total + sv
                off.append(o)
                base[q] = base[q] + total
            if const_expr(lb):  # noqa: SIM102
                if live:
                    lds_st(L, L_EOFF + e * i32(4), off[1])
            for c_ in range(i32(0), ce, i32(1)):
                c = i32(c_)
                lds_st(
                    L,
                    L_DCH + (off[0] + c) * i32(4),
                    e | (ce << i32(16)) | (c << i32(24)),
                )
            cbar(L, tid)
        if tid == i32(0):
            lds_st(L, L_CTL + C_NCH * 4, base[0])
        if const_expr(lb):
            cbar(L, tid)

    @traced
    def lb_scatter(L, tid, a, lo, hi):
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
        wait_vm(0)
        cbar(L, tid)
        if tid == i32(0):
            if const_expr(ZMA):
                spin_lds_ge(L, L_CTL + C_ZDONE * 4, i32(1))
            g_add_agent(lb_lbr(a, lb_bank(L), i32(gpu.block_id("x")) % i32(N_XCD)), 1)

    @traced
    def _lb_units(L, tid):
        if tid == i32(0):
            nun = lds_ld_i32(L, L_CTL + C_NCH * 4)
            C = i32(gpu.grid_dim.x)
            P = _pick_p(nun, C, LB_PS)
            U = nun * P
            bid = _xcd_rank(C)
            icnt = i32(I) // P
            mix = fx.Boolean(False)
            J2 = nun
            if const_expr(LBMIX):
                X = i32(2) * nun - C
                K = ceildiv(X, i32(2))
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
            mine = (bid < U).select(ceildiv(U - bid, C), i32(0))
            if const_expr(LBMIX):
                mine = mix.select(lds_ld_i32(L, L_CTL + (C_UNIT + 1) * 4), mine)
            lds_st(L, L_CTL + C_DYNP * 4, P)
            lds_st(L, L_CTL + C_UNIT * 4, i32(0))
            lds_st(L, L_CTL + (C_UNIT + 1) * 4, mine)
            lds_st_rel(L, L_CTL + C_PLAN * 4, i32(1))

    def lb_pieces(L, j):
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
        J2 = nun - K
        if r < M:
            j = r // i32(2)
            s_ = r - j * i32(2)
            _lb_ul(
                L,
                i32(0),
                lds_ld_i32(L, L_DCH + j * i32(4)),
                s_ * i32(I // 2),
                i32(I // 2),
                i32(UNIT_G1X) | (s_ << i32(8)) | (j << i32(XQ_SHIFT)),
            )
            lds_st(L, L_CTL + (C_UNIT + 1) * 4, i32(1))
        else:
            for k in range_constexpr(3):
                f = (r - M) * i32(3) + i32(k)
                j = J2 + f // i32(KS2)
                s_ = f - (j - J2) * i32(KS2)
                _lb_ul(
                    L,
                    i32(k),
                    lds_ld_i32(L, L_DCH + j * i32(4)),
                    s_ * i32(128),
                    i32(128),
                    i32(UNIT_G1X) | (s_ << i32(8)) | (j << i32(XQ_SHIFT)),
                )
            lds_st(L, L_CTL + (C_UNIT + 1) * 4, i32(3))
        if r < i32(3) * X:
            f = (C - M) * i32(3) + r
            j = J2 + f // i32(KS2)
            s_ = f - (j - J2) * i32(KS2)
            nn = lds_ld_i32(L, L_CTL + (C_UNIT + 1) * 4)
            _lb_ul(
                L,
                nn,
                lds_ld_i32(L, L_DCH + j * i32(4)),
                s_ * i32(128),
                i32(128),
                i32(UNIT_G1X) | (s_ << i32(8)) | (j << i32(XQ_SHIFT)),
            )
            lds_st(L, L_CTL + (C_UNIT + 1) * 4, nn + i32(1))

    @traced
    def lb_routes(L, tid, a, ent):
        e = ent & i32(0xFFFF)
        r0 = ent.shrui(i32(24)) * i32(RCH)
        R = fx.min(lds_ld_i32(L, L_DCNT + e * i32(4)) - r0, i32(RCH))
        base = lds_ld_i32(L, L_EOFF + e * i32(4)) + r0
        if (tid < i32(N_XCD)) & (lds_ld_i32(L, L_CTL + C_LBN * 4) == i32(0)):
            lb_lbr_wait(a, lb_bank(L), tid)
        cbar(L, tid)
        if (tid == i32(0)) & (lds_ld_i32(L, L_CTL + C_LBN * 4) == i32(0)):
            lds_st(L, L_CTL + C_LBN * 4, i32(1))
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
            g_add_agent(lb_g1c(a, kind.shrui(i32(XQ_SHIFT)), lb_bank(L)), 1)

    @traced
    def _lb_g1_fin(L, tid):
        if tid == i32(0):
            lds_st_rel(L, L_CTL + C_G1FIN * 4, i32(1))

    def cg_lo(cc):
        v = i32(0)
        for k in range_constexpr(1, NCG + 1):
            v = (cc >= i32(k)).select(i32(CGB[k]), v)
        return v

    def lb_g2_cap(U2, nblk):
        return ceildiv(U2, nblk) * i32(4) + i32(4)

    @traced
    def lb_g2_consume(L, tid, a):
        nun = lds_ld_i32(L, L_CTL + C_NCH * 4)
        nblk = i32(gpu.grid_dim.x)
        lb_none(L, tid, a, nun)
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
            if tid == i32(0):
                lds_st_rel(L, L_CTL + C_G2FREE * 4, n + i32(1))
                lb_col_done(L, a, cc, nun, lb_bank(L))

    @traced
    def _g2_last(L, tid, n):
        if tid == i32(0):
            lds_st_rel(L, L_CTL + C_G2LAST * 4, n + i32(1))

    @traced
    def lb_g2_prefetch(L, lane, a):
        spin0(L, lane, L_CTL + C_G1FIN * 4, i32(1))
        nun = lds_ld_i32(L, L_CTL + C_NCH * 4)
        U2 = nun * i32(NCG)
        bank = lb_bank(L)
        if (lane < i32(N_XCD)) & (lds_ld_i32(L, L_CTL + C_LBN * 4) == i32(0)):
            lb_lbr_wait(a, bank, lane)
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
        rv = []
        for k in range_constexpr(ceildiv(RG, 64)):
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
        NQ = ceildiv(RG * u16, 64)
        for b0 in range_constexpr(0, NQ, 8):
            dq = []
            for k in range_constexpr(b0, min(b0 + 8, NQ)):
                q = fx.min(lane + i32(k * 64), nq - i32(1))
                row = q // i32(u16)
                c = q - row * i32(u16)
                rix = lds_ld_i32(L, rix0 + row * i32(4))
                dq.append(
                    (row, c, bld(rx, rix * i32(I // 2) + c * i32(16), 0, V4I, AUX_SYS))
                )
            for row, c, dv in dq:
                lds_st(L, ib0 + row * i32(SI_STRIDE) + c * i32(16), dv, 16)
        ns = fx.max(R * i32(I // 128), i32(1))
        ds = []
        for k in range_constexpr(ceildiv(RG * (I // 128), 64)):
            q = fx.min(lane + i32(k * 64), ns - i32(1))
            row = q // i32(I // 128)
            c = q - row * i32(I // 128)
            rix = lds_ld_i32(L, rix0 + row * i32(4))
            ds.append(
                (row, c, bld(rxs, rix * i32(I // 32) + c * i32(4), 0, T.i32, AUX_SYS))
            )
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
        nun = lds_ld_i32(L, L_CTL + C_NCH * 4)
        U2 = nun * i32(NCG)
        nblk = i32(gpu.grid_dim.x)
        lb_none(L, tid, a, nun)
        cap = ceildiv(U2, nblk) * i32(2) + i32(2)
        col_claim(L, tid, a)
        for it_ in range(i32(0), cap, i32(1)):
            _lb_g2_unit(L, tid, a, nun, U2, i32(it_) < cap - i32(1))

    @traced
    def _lb_g2_unit(L, tid, a, nun, U2, more):
        v = lds_ld_i32(L, L_CTL + C_CLAIM * 4)
        if v < U2:
            cc = v // nun
            _lb_g2_run(L, tid, a, nun, cc, v - cc * nun)
        if more:
            col_claim(L, tid, a)

    @traced
    def _lb_g2_run(L, tid, a, nun, cc, j):
        ent = lds_ld_i32(L, L_DCH + j * i32(4))
        bank = lb_bank(L)
        if tid == i32(0):
            spin_sys_ge(lb_g1c(a, j, bank), lb_pieces(L, j), a)
        R = lb_routes(L, tid, a, ent)
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
        if tid == i32(0):
            lb_col_done(L, a, cc, nun, bank)

    @traced
    def lb_import(L, tid, a, rows):
        ag_stage_free(L, tid % i32(64), a)
        cbar(L, tid)
        u16 = I // 2 // 16
        rx, rxs = xg_rs(a)
        nq = fx.max(rows * i32(u16), i32(1))
        ns = fx.max(rows * i32(I // 128), i32(1))
        dq, ds = [], []
        for k in range_constexpr(ceildiv(RG * u16, NT)):
            q = fx.min(tid + i32(k * NT), nq - i32(1))
            row = q // i32(u16)
            c = q - row * i32(u16)
            rix = lds_ld_i32(L, L_RIX + row * i32(4))
            dq.append(
                (row, c, bld(rx, rix * i32(I // 2) + c * i32(16), 0, V4I, AUX_SYS))
            )
        for k in range_constexpr(ceildiv(RG * (I // 128), NT)):
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

    @traced
    def lb_finish(tid, a, L):
        lb_zero_next(tid, a, lb_bank(L))
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
        xq_load(L, tid, a, r0, rows)

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
                poll_zero(lambda: _meta_pending(a, lane, a["epoch"]))
            for i_ in range(lane, ceildiv(a["ttot"], i32(32)), i32(64)):
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
            spin0(L, lane, L_CTL + C_CLS * 4, i32(1))

    @traced
    def _cls_set(L, t, late):
        if late:
            lds_atomic_or(L, L_CLS + (t >> i32(5)) * i32(4), i32(1) << (t & i32(31)))

    def push_units(ttot):
        nblk = i32(gpu.grid_dim.x)
        return fx.min(i32(2) * nblk, ceildiv(ttot, i32(TB)))

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
        n = fx.max(ceildiv(ns - u0, i32(nblk)), i32(0))
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
        d0, d1, e8i = mxfp8x8(acc)
        if const_expr(LL):
            pkt = ll_pkt(a, d0, d1, e8i)
            if ok:
                bst(pkt, r_dst, (prow * i32(H) + col) * i32(2), 0, AUX_SYS)
        else:
            doff, soff = part_offs(a, prow, col)
            sc = scales4(e8i)
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
        t0 = now()
        while (pend != i32(0)) & alive(t0):
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
                bst(
                    o,
                    rsrc(a["y"]),
                    (row * i32(H) + c0 + v * i32(8)) * i32(2),
                    0,
                    AUX_SYS if TN else 0,
                )

    @traced
    def _y_store_ag8(a, row, c0, v, acc):
        col = c0 + v * i32(8)
        grow = a["rank"] * a["m"] + row
        d0, d1, e8i = mxfp8x8(acc)
        sc = scales4(e8i)
        f = fp8x4_unpack(d0, e8_scale(e8i)) + fp8x4_unpack(d1, e8_scale(e8i))
        if v < i32(CW // 8):
            bst(pack_bf16x8(f), rsrc(a["y"]), (grow * i32(H) + col) * i32(2), 0, 0)
            doff, soff = part_offs(a, grow, col)
            dv = fx.Vector.from_elements([d0, d1], fx.Int32)
            for p in range_constexpr(TPC):
                if i32(p) != a["rank"]:
                    rd = peer_rs(a, p, "off_yall")
                    bst(dv, rd, doff, 0, AUX_SYS)
                    if (v & i32(15)) == i32(0):
                        bst(sc, rd, soff, 0, AUX_SYS)

    @traced
    def ag8_convert(tid, a):
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
                        doff, soff = part_offs(a, grow, col)
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
            for it in range_constexpr(ceildiv(NV * TPC, 64)):
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
        for it in range_constexpr(ceildiv(NV * MAX_TP, 64)):
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
            poll_zero(lambda: _yag_pending(a, lane, epoch), a, lane, ERR_YAG)

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
        v = g_ld_sys(lrdy_at(a, c))
        pm = i32(fx.Int64(rocdl.ballot(T.i64, live & (v >= epoch))) & fx.Int64(VMASK))
        fm = i32(
            fx.Int64(rocdl.ballot(T.i64, live & _final_ready(a, c, epoch)))
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
        t0 = now()
        while _comm_pending(L) & alive(t0):
            p1 = comm_signal(L, tid, a, epoch)
            p2 = poll_ready(L, tid % i32(64), a, epoch)
            if (p1 | p2) == i32(0):
                rocdl.s_sleep(POLL_SLEEP)
        _report_if(
            a,
            (now() - t0 >= fx.Int64(DEADLINE)) & ((tid % i32(64)) == i32(0)),
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
        n = fx.max(ceildiv(ns - u0, i32(nblk)), i32(0))
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
        if won == i32(1):  # noqa: SIM102
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
        _signal_part(a, epoch, cidx, x, ceildiv(nblk - x, i32(N_XCD)))

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
                ceildiv(i32(XCOL_PER_CHUNK) - sh, i32(N_XCD)),
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
                ceildiv(i32(xl_s0) - sh, i32(N_XCD)),
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
                g_st_sys(lrdy_at(a, cidx), epoch)

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
            g_st_sys(lrdy_at(a, vc), epoch)

    @traced
    def signal_loop(L, tid, a, epoch):
        t0 = now()
        while (lds_ld_acq(L, L_CTL + C_NSIG * 4) < i32(NCK)) & alive(t0):
            if comm_signal(L, tid, a, epoch) == i32(0):
                rocdl.s_sleep(1)

    @traced
    def comm_wave(L, tid, a, epoch):
        t0 = now()
        while (
            (lds_ld_acq(L, L_CTL + C_NSIG * 4) < i32(NCK))
            | (lds_ld_acq(L, L_CTL + C_LRED * 4) < i32(NV))
            | (lds_ld_acq(L, L_CTL + C_PULL * 4) < i32(NV))
        ) & alive(t0):
            p1 = comm_signal(L, tid, a, epoch)
            if const_expr(not ARLL):
                p1 = p1 | poll_ready(L, tid % i32(64), a, epoch)
            p2 = comm_work(L, tid, a, epoch)
            if (p1 | p2) == i32(0):
                rocdl.s_sleep(POLL_SLEEP)
        _report_if(
            a,
            (now() - t0 >= fx.Int64(DEADLINE)) & ((tid % i32(64)) == i32(0)),
            ERR_COMM,
        )

    @traced
    def comm_help(L, tid, a, epoch):
        t0 = now()
        while (
            (lds_ld_acq(L, L_CTL + C_LRED * 4) < i32(NV))
            | (lds_ld_acq(L, L_CTL + C_PULL * 4) < i32(NV))
        ) & alive(t0):
            if comm_work(L, tid, a, epoch) == i32(0):
                rocdl.s_sleep(POLL_SLEEP)

    AG_SROW = XB
    AG_SCB = AGR * AG_SROW
    assert AG_SCB + AGR * (H // 32) <= L_INTERS - L_INTER + RG * (I // 32)
    NCHA = H // 256
    NPRE = ceildiv(NCHA, PRE_CH)
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
        nr = fx.min(fx.max(ceildiv(a["m"] - bid, nblk), i32(0)), i32(AGR))
        m = fx.max(a["m"], i32(1))
        s = bid // m
        split = S > i32(1)
        live = s < S
        row0 = split.select(bid - s * m, bid)
        nrows = split.select(live.select(i32(1), i32(0)), nr)
        qb = split.select(s, i32(0))
        qs = split.select(S, i32(1))
        cnt = split.select(live.select(ceildiv(i32(NCHA) - s, S), i32(0)), i32(NCHA))
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
        for g_ in range(i32(0), ceildiv(cnt, pc), i32(1)):
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
        poll_zero(
            lambda: wave_red(
                (live & (g_ld_sys(addr) < a["epoch"])).select(i32(1), i32(0)),
                lane,
                fx.max,
            ),
            a,
            lane,
            ERR_FLAG,
        )

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
            t0 = now()
            while (mp != i32(0)) & alive(t0):
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
            if const_expr(AIN and k >= 1):  # noqa: SIM102
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
            if const_expr(k >= 1):  # noqa: SIM102
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
        spin0(L, lane, L_CTL + C_AGFREE * 4, ag_npar(a))

    def _ag_senders(a, q, x):
        m = fx.max(a["m"], i32(1))
        S = ag_split(a)
        lo = (S > i32(1)).select((q % S) * m, i32(0))
        hi = (S > i32(1)).select(lo + m, i32(gpu.grid_dim.x))
        return ceildiv(hi - x, i32(N_XCD)) - ceildiv(lo - x, i32(N_XCD))

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
        return fx.max(ceildiv(a["m"] * i32(TOPK) // i32(4), i32(64)), i32(1))

    @traced
    def _ag_send_meta(lane, a):
        bid = i32(gpu.block_id("x"))
        if bid < _ag_nmeta(a):
            n = a["m"] * i32(TOPK)
            n4 = n // i32(4)
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
            t0 = now()
            while (t != a["epoch"]) & alive(t0):
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
            poll_zero(lambda: _meta_pending(a, lane, epoch), a, lane, ERR_META)

    def _ag_first_pending(lane, a, epoch, c_lo, nwin):
        rf = own_rs(a, "off_flag")
        end = fx.min(c_lo + i32(nwin), i32(NCH))
        first = end
        for it in range_constexpr(ceildiv(TPC * nwin, 64)):
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
        t0 = now()
        while (rdy <= cc) & alive(t0):
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
            nx = ceildiv(i32(gpu.grid_dim.x) - x, i32(N_XCD))
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
        spin0(L, lane, L_CTL + C_CDONE * 4, i32(1))
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
                    t0 = now()
                    while (pend != i32(0)) & alive(t0):
                        rocdl.s_sleep(8)
                        pk = _dll_rows(a, t, c0, vc, P)
                        pend = ll_pending(a, lane, zip(pk, live))
                else:
                    pend = ll_pending(a, lane, zip(_dll_rows(a, t, c0, vc, P), live))
                    t0 = now()
                    while (pend != i32(0)) & alive(t0):
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
                t0 = now()
                while (pend != i32(0)) & alive(t0):
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

    TN_V = H // 8
    TN_IT = ceildiv(TN_V, NTT)

    @traced
    def blk_red(L, tid, v, op, slot):
        r = wave_red(v, tid % i32(64), op)
        if (tid % i32(64)) == i32(0):
            lds_st(
                L, i32(L_RING) + (i32(slot * (NTT // 64)) + tid // i32(64)) * i32(4), r
            )
        gpu.barrier()
        t = lds_ld_i32(L, i32(L_RING) + i32(slot * (NTT // 64) * 4))
        for k in range_constexpr(1, NTT // 64):
            t = op(t, lds_ld_i32(L, i32(L_RING) + i32((slot * (NTT // 64) + k) * 4)))
        return t

    @traced
    def tn_rows(L, tid, a):
        m = a["m"]
        S = fin_split(a)
        step = (S > i32(1)).select(m, i32(gpu.grid_dim.x))
        mine = fin_owned(a) != i32(0)
        start = ((S > i32(1)) & (i32(gpu.block_id("x")) >= m * S)).select(m, fin_key(a))
        for r_ in range(start, m, step):
            r = i32(r_)
            if tid == i32(0):
                last = i32(1)
                if S > i32(1):
                    ca = ctrl_at(a, i32(CTRL_TNC) + r)
                    old = g_add_agent(ca, 1)
                    last = (old == S - i32(1)).select(i32(1), i32(0))
                    if last == i32(1):
                        g_st_sys(ca, i32(0))
                lds_st(L, L_CTL + C_TNL * 4, mine.select(last, i32(0)))
            gpu.barrier()
            if lds_ld_i32(L, L_CTL + C_TNL * 4) == i32(1):
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
        tot = blk_red(
            L,
            tid,
            ss.bitcast(fx.Int32),
            lambda x, y: (x.bitcast(fx.Float32) + y.bitcast(fx.Float32)).bitcast(
                fx.Int32
            ),
            0,
        )
        rcp = fx.Float32(
            fmath.rsqrt(
                tot.bitcast(fx.Float32) / fx.Float32(float(H))
                + fx.Float32(float(tn_eps))
            )
        )
        xs, am = [], fx.Float32(0.0)
        for k in range_constexpr(TN_IT):
            q = fx.min(tid + i32(k * NTT), H8 - i32(1))
            wv = bf16x8_to_f32(bld(rw, q * i32(16), 0, V4I, 0))
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
        amx = blk_red(
            L,
            tid,
            am.bitcast(fx.Int32),
            lambda x, y: x.bitcast(fx.Float32)
            .maximumf(y.bitcast(fx.Float32))
            .bitcast(fx.Int32),
            1,
        )
        scale = amx.bitcast(fx.Float32).maximumf(fx.Float32(1e-10)) / fx.Float32(448.0)
        inv = fx.Float32(1.0) / scale
        one = fx.Float32(1.0)
        grow = a["rank"] * a["m"] + r
        for k in range_constexpr(TN_IT):
            q = tid + i32(k * NTT)
            qv = [x * inv for x in xs[k]]
            d = fx.Vector.from_elements(
                [fp8x4_pack(qv[0:4], one), fp8x4_pack(qv[4:8], one)], fx.Int32
            )
            if q < H8:
                for p in range_constexpr(TPC):
                    bst(
                        d,
                        peer_rs(a, p, "off_qall"),
                        grow * i32(H) + q * i32(8),
                        0,
                        AUX_SYS,
                    )
        b0 = a["mmax"] * i32(TPC * H)
        for k in range_constexpr(TN_IT if TNB else 0):
            q = tid + i32(k * NTT)
            d = pack_bf16x8(xs[k])
            if q < H8:
                for p in range_constexpr(TPC):
                    bst(
                        d,
                        peer_rs(a, p, "off_qall"),
                        b0 + (grow * i32(H) + q * i32(8)) * i32(2),
                        0,
                        AUX_SYS,
                    )
        if tid == i32(0):
            for p in range_constexpr(TPC):
                bst(
                    scale.bitcast(fx.Int32),
                    peer_rs(a, p, "off_sall"),
                    grow * i32(4),
                    0,
                    AUX_SYS,
                )
        wait_vm(0)
        gpu.barrier()
        if tid < i32(TPC):
            fo = fx.Int64((i32(FLAG_TN) + a["rank"] * i32(TN_MAX) + r) * i32(4))
            g_st_sys(peer_sel(a, tid) + fx.Int64(a["off_flag"]) + fo, a["epoch"])

    @traced
    def tn_wait(tid, a):
        if tid < i32(64):
            nblk = i32(gpu.grid_dim.x)
            t = i32(gpu.block_id("x")) + tid * nblk
            live = t < a["ttot"]
            tc = fx.min(t, a["ttot"] - i32(1))
            src = tc // a["m"]
            row = tc - src * a["m"]
            addr = (
                a["mine"]
                + fx.Int64(a["off_flag"])
                + fx.Int64((i32(FLAG_TN) + src * i32(TN_MAX) + row) * i32(4))
            )
            poll_zero(
                lambda: wave_red(
                    (live & (g_ld_sys(addr) < a["epoch"])).select(i32(1), i32(0)),
                    tid,
                    fx.max,
                ),
                a,
                tid,
                ERR_YAG,
            )

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
                if const_expr(ZMA):  # noqa: SIM102
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
        if const_expr(DYN):
            _dyn_zero(L, tid)
        init_lds(L, tid, a)
        gpu.barrier()
        epoch = lds_ld_i32(L, L_CTL + C_EPOCH * 4)
        a["epoch"] = epoch
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
            tn_rows(L, tid, a)
            tn_wait(tid, a)
        if const_expr(AG8):
            gpu.barrier()
            ag8_convert(tid, a)
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
