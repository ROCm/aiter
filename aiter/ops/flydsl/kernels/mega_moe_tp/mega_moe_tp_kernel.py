# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Single-kernel tensor-parallel MegaMoE.

One persistent launch runs the token AllGather (or input all-reduce), route
scan, GEMM1, activation, MXFP4 requant, GEMM2, top-k reduce and ReduceScatter
(or output all-reduce). The GEMM1 -> GEMM2 intermediate stays in LDS.

Waves per CTA: 4 compute (weights stream global -> LDS through a private ring
of ``buffer_load ... lds``), 1 comm, 1 A-loader, 2 push. Cross-rank memory is
an IPC-mapped symmetric arena. Flags carry a monotonic launch epoch, so
nothing is reset between launches (CUDA-graph safe). Polls use relaxed
cache-bypassing accesses: on gfx950 an acquire/release fence would flush the
whole XCD L2 under the compute waves.
"""

from __future__ import annotations

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .. import buffer_ops
from ..mxfp4_gemm_common import _activation_mul_batch, _e8m0_from_amax, _fabs_f32

__all__ = [
    "CTRL_ERR",
    "CTRL_INTS",
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
    "mega_moe_tp_consts",
    "mega_moe_tp_shape_supported",
    "gemm2_chunk_groups",
    "gemm2_group_step",
]

MAX_TP = 8
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
FLAG_INTS = FLAG_AGQ + NCHA_MAX * MAX_TP

# Local control ints. Polled/atomic counters sit on their own 128 B lines:
# same-line device atomics from every CTA serialize and stall weight streams.
CTRL_ERR = 20  # OR of ERR_* of every wait that gave up
CTRL_XF = 32  # [N_XCD] first launch: XCD x dropped its L2
CTRL_CNT = 64
ERR_FLAG, ERR_META, ERR_CHUNK, ERR_COMM, ERR_YAG = 1, 2, 4, 8, 16
# a wait without progress for 2 s (100 MHz clock) gives up and reports
DEADLINE = 200_000_000
N_XCD = 8  # CTAs are dealt round-robin over the XCDs
LRDY_STRIDE = 32
CTRL_LRDY = CTRL_CNT + 2 * NCK_MAX  # [NCK] chunk c ready for the RS push
# column split: CTRL_XQ + j*XQ_P + p is up once slice p of expert slot j exported
CTRL_XQ = CTRL_LRDY + NCK_MAX * LRDY_STRIDE
XQ_P = 8
XQ_MAX = 264
# per chunk: N_XCD per-XCD readiness counts, the XCD count, the push count,
# and N_XCD per-shard column-slice counts
CTRL_SC = CTRL_XQ + XQ_MAX * XQ_P
SC_ALL, SC_PUSH, SC_SHARD = N_XCD, N_XCD + 1, N_XCD + 2
SC_LINES = SC_SHARD + N_XCD
CTRL_EPB = CTRL_SC + NCK_MAX * SC_LINES * LRDY_STRIDE  # [NCTA] launches per CTA
CTRL_XES = CTRL_EPB + NCTA_MAX  # [N_XCD] finished CTAs per XCD
CTRL_AGX = CTRL_XES + N_XCD * LRDY_STRIDE  # [NCHA][N_XCD] AllGather senders
CTRL_AGG = CTRL_AGX + NCHA_MAX * N_XCD * LRDY_STRIDE  # [NCHA] XCDs done
CTRL_CLM = CTRL_AGG + NCHA_MAX * LRDY_STRIDE  # column-slice claims, 2 banks
CTRL_INTS = CTRL_CLM + 2 * LRDY_STRIDE
UNIT_REC = 8  # ints per CTA record: ub, ue, first unit (4), pad
# units[u] = (expert, i0, icnt, kind); kind = type | groups << 8 | SIGNAL | slot << XQ_SHIFT
#   UNIT_G1X: GEMM1 inter slice exporting its intermediate
#   UNIT_G2COL: GEMM2 column slice (i0 = first group) importing it
#   SIGNAL: the unit after which the CTA signals every chunk
UNIT_FULL, UNIT_G2COL, UNIT_G1X = 0, 2, 3
UNIT_SIGNAL = 1 << 16
XQ_SHIFT = 20

NW = 4  # compute waves
NT = NW * 64
NTT = NT + 4 * 64  # + comm, loader and 2 push waves
NSK = 4  # weight ring depth (k-steps) per compute wave
SLOT = 4 * 1024 + 2 * 256  # 4 x 1 KB fragments + 2 scale dwords (256 B each)
OPS = 6  # DMA ops per k-step
KCS = 2  # k-steps per A chunk (256 K)
ACB = KCS * 64  # A bytes per row per chunk
ALOAD_DEPTH = 3
RED_INFLIGHT = 32
POLL_SLEEP = 16
AUX_SC1 = 16
AUX_SYS = 1 | 16


def mega_moe_tp_shape_supported(model_dim: int, inter_dim: int, tp: int) -> bool:
    return (
        model_dim % 512 == 0
        and (model_dim // 64) % NW == 0
        and inter_dim % 128 == 0
        and 1 <= tp <= MAX_TP
    )


def _nsk2_for(nks: int, g2: int) -> int:
    """GEMM2 ring depth for NKS k-steps per column group: the full ring
    whenever its period (lcm(NSK, NKS) / NKS groups) tiles G2."""
    if g2 % (NSK * nks // math.gcd(NSK, nks) // nks) == 0:
        return NSK
    return 3 if nks % 3 == 0 else 2


def gemm2_group_step(H: int, I: int) -> int:
    """Column groups per full-K GEMM2 step (GPI); column ranges must be multiples."""
    ks2, g2 = I // 128, H // 64 // NW
    nsk2 = _nsk2_for(ks2, g2)
    return nsk2 * ks2 // math.gcd(nsk2, ks2) // ks2


def gemm2_chunk_groups(H: int, I: int) -> int:
    """Column groups per ReduceScatter chunk (GPC)."""
    ks2, g2 = I // 128, H // 64 // NW
    return 2 if (2 * ks2 * OPS <= 63 and g2 % 2 == 0) else 1


def _attr(v):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), int(v))


def _u(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


traced = ASTRewriter.transform


@functools.cache
def mega_moe_tp_consts(
    H: int, I: int, MT: int, TMAX: int, agr: int = 1, dyn_e: int = 0, nab: int = 4
) -> dict:
    """Compile-time geometry and LDS layout shared by the kernel and the host."""
    RG = MT * 16
    KS1 = H // 128
    KS2 = I // 128
    G2 = H // 64 // NW
    NSC_BLK = (RG + 63) // 64
    NA_ROWOPS = RG * ACB // 1024
    GPC = gemm2_chunk_groups(H, I)
    c = dict(
        RG=RG,
        KS1=KS1,
        NCH=KS1 // KCS,
        KS2=KS2,
        G2=G2,
        CH1=(H // 32 + 7) // 8,
        CH2=(I // 32 + 7) // 8,
        SI_STRIDE=I // 2 + 16,
        NA_ROWOPS=NA_ROWOPS,
        NSC_BLK=NSC_BLK,
        NA_L=NA_ROWOPS + NSC_BLK * KCS,
        GPC=GPC,
        NCK=G2 // GPC,
        CW=GPC * NW * 64,
        NAB=nab,
        NBW=(dyn_e + 31) // 32,
    )
    off = 0

    def take(n):
        nonlocal off
        off = (off + 15) // 16 * 16
        start, off = off, off + n
        return start

    c["L_RING"] = take(NW * NSK * SLOT)
    c["L_A"] = take(nab * RG * ACB)
    c["L_AS"] = take(nab * KCS * NSC_BLK * 64 * 4)
    # the AllGather staging (agr MXFP4 rows + scales) shares the GEMM2 operand area
    c["L_INTER"] = take(max(RG * c["SI_STRIDE"], agr * (H // 2 + H // 32) - RG * (I // 32)))
    c["L_INTERS"] = take(RG * (I // 32))
    c["L_RIX"] = take(TMAX * 4)
    c["L_WT"] = take(TMAX * 4)
    c["L_CTL"] = take(128 * 4)
    # dyn: active-expert bitmap, its prefix popcounts, the active list
    c["L_DYN"] = take((2 * c["NBW"] + dyn_e) * 4)
    c["LDS_BYTES"] = (off + 127) // 128 * 128
    assert H // 256 <= NCHA_MAX
    assert KS1 % NSK == 0 and NSK % KCS == 0
    assert (NSK - 1) * OPS <= 63 and c["NA_L"] * (ALOAD_DEPTH - 1) <= 63
    assert c["NCK"] <= NCK_MAX and c["NCK"] <= 31  # C_FBITS
    return c


# control-int indices inside L_CTL
C_CNT, C_BARCNT, C_BARGEN, C_LRED, C_PULL, C_EPOCH = 0, 1, 2, 3, 4, 5
C_USEQ, C_UROWS, C_NSIG, C_LQ, C_LPUB = 6, 7, 8, 9, 10
C_DONE = 12  # [NW]
C_ASEQ = 16  # [NAB]
C_AFREE = 20  # [NAB]
C_QBASE = 24  # [NW]
C_MBOX = 28  # [8] per-wave mailbox
C_FBITS = 36  # final stage: claimed chunks
CLAIM_W = 3  # unclaimed chunks a final claim looks at
C_AGFREE = 41  # AllGather staging read out
C_PLAN, C_NACT = 42, 43  # dyn: units planned; active experts
C_ARDY = 44  # loader: K-chunks known to have arrived
C_DYNP = 45  # dyn: pieces per active expert
C_CDONE = 46  # LL routes: compute waves done
C_COLJ = 47  # running column slice's expert slot
C_UNIT = 48  # [6] ub, ue, first unit (expert, i0, icnt, kind)
C_CLAIM = 56  # [NW] last claimed column slice, per compute wave
C_XC, C_XC_N = 64, 32  # [C_XC_N] compute waves done with a column slice's chunk
C_UL = 96  # dyn: this CTA's units (4 ints each)


@functools.cache
def compile_mega_moe_tp(
    *,
    H: int,
    I: int,
    TOPK: int,
    MT: int,
    TMAX: int,
    act: str = "silu",
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
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
    ll_rs: bool = False,
    ll_route: bool = False,
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
    """
    DYN = dyn_e > 0
    c = mega_moe_tp_consts(H, I, MT, TMAX, agr, dyn_e, nab)
    RG, KS1, NCH, KS2, G2 = c["RG"], c["KS1"], c["NCH"], c["KS2"], c["G2"]
    CH1, CH2, SI_STRIDE = c["CH1"], c["CH2"], c["SI_STRIDE"]
    NA_ROWOPS, NSC_BLK, NA_L = c["NA_ROWOPS"], c["NSC_BLK"], c["NA_L"]
    GPC, NCK, CW, NAB, NBW = c["GPC"], c["NCK"], c["CW"], c["NAB"], c["NBW"]
    L_RING, L_A, L_AS, L_DYN = c["L_RING"], c["L_A"], c["L_AS"], c["L_DYN"]
    L_INTER, L_INTERS, L_RIX, L_WT = c["L_INTER"], c["L_INTERS"], c["L_RIX"], c["L_WT"]
    L_CTL, LDS_BYTES = c["L_CTL"], c["LDS_BYTES"]
    assert 3 <= NAB <= 4
    NE = dyn_e
    VPL = (CW // 8 + 63) // 64
    SCAN_IT = (TMAX * TOPK // 4 + NT - 1) // NT  # route-scan vector loads per thread
    SCAN_G = 8
    NPC = KS2 if DYN else max(npieces, 1)  # route regions (pieces) per route
    assert not DYN or xsplit == 0
    # route rows: bf16, or E4M3 + one E8M0 per 32 columns (all rows' data,
    # then all rows' scales per region)
    FP8R = bool(route_fp8)
    # LL: every 16 B is a packet [data, tag, data, tag], tag = epoch << 8 | E8M0;
    # the reader polls the data itself (no completion wait / flag round trips)
    LL = bool(ll_rs)
    # tokens per push batch (more pieces keep more offsets live: VGPR spills)
    TB = 1 if DYN else max(1, (RED_INFLIGHT // 2 if NPC > 2 else RED_INFLIGHT) // TOPK)
    DYN_PS = [p for p in range(1, KS2 + 1) if KS2 % p == 0]
    # LL route rows ([E4M3 x4, tag, E4M3 x4, tag] per 8 columns): each
    # (token, chunk) push polls them -- no chunk counts, flags or claims
    DLL = LL and bool(ll_route) and (DYN or NPC == 1)
    assert not DLL or FP8R
    MLL = DLL and not ar  # + LL routing AllGather (16 B packet per route)
    ARLL = DLL and bool(ar)  # + LL output all-reduce
    ROW_B = 2 * H if DLL else (H + H // 32 if FP8R else 2 * H)
    XSPLIT = xsplit > 0
    assert not XSPLIT or (xw % gemm2_group_step(H, I) == 0 and xw % GPC == 0)
    assert xsplit <= XQ_P
    XCOL_PER_CHUNK = xrem if XSPLIT else 0
    AR = bool(ar)
    TPC = int(tp)
    AGR = int(agr)
    name = (
        f"mega_moe_tp_fused_h{H}_i{I}_k{TOPK}_mt{MT}_t{TMAX}_{act}"
        + (f"_p{NPC}" if NPC > 1 else "")
        + ("_r8" if route_fp8 else "")
        + f"_ag{agr}_tp{tp}"
        + ("_ar" if ar else "")
        + (f"_x{xsplit}r{xrem}w{xw}" if xsplit else "")
        + (f"_dyn{NE}" if DYN else "")
        + (f"_nab{NAB}" if NAB != 4 else "")
        + ("_ll" if LL else "")
        + ("_llr" if DLL else "")
    )
    const_expr = fx.const_expr
    # flydsl's cache key ignores module constants: the kernel references this tag
    layout_tag = f"{CTRL_INTS}/{FLAG_INTS}/{DEADLINE}/{UNIT_REC}/{NCTA_MAX}"

    # ---------------------------- low-level helpers ----------------------------
    def i32(v):
        raw = v if isinstance(v, ir.Value) else getattr(v, "ir_value", lambda: None)()
        if raw is not None and isinstance(raw.type, ir.IndexType):
            return fx.Int32(fx.index_cast(T.i32, raw))
        return fx.Int32(v)

    class _LazyTy:
        # MLIR types need the trace-time context
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

    # Scoped atomics and ordered accesses on typed pointers from raw addresses.
    MONO, ACQ, REL, ACQ_REL = (
        fx.AtomicOrdering.Monotonic,
        fx.AtomicOrdering.Acquire,
        fx.AtomicOrdering.Release,
        fx.AtomicOrdering.AcqRel,
    )

    def lds_p(base, off):
        return fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Shared, 4), fx.Int32(base + i32(off)))

    def gptr(addr):
        return fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), fx.Int64(addr))

    def lds_atomic_add(base, off, v, ordering=MONO):
        return i32(fx.atomic_add(lds_p(base, off), i32(v), syncscope="workgroup", ordering=ordering))

    def lds_atomic_or(base, off, v):
        return i32(fx.atomic_or(lds_p(base, off), i32(v), syncscope="workgroup"))

    def lds_ld_acq(base, off):
        return i32(fx.generic_load(lds_p(base, off), dtype=fx.Int32, memory_order=MONO,
                                   syncscope="workgroup", volatile=True))

    def lds_st_rel(base, off, v):
        fx.generic_store(lds_p(base, off), i32(v), memory_order=REL, syncscope="workgroup")

    def lds_cas(base, off, cmp, new):
        return i32(fx.atomic_cas(lds_p(base, off), i32(cmp), i32(new), syncscope="workgroup")[0])

    def _ctpop(v):
        return i32(fx.ctpop(i32(v)))

    def g_ld_rel(addr, scope):
        return i32(fx.generic_load(gptr(addr), dtype=fx.Int32, memory_order=MONO, syncscope=scope, volatile=True))

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
        # sys: read memory, not a cached copy (sc0 sc1)
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
        """a['peer'][p] for a runtime (wave-uniform) rank p."""
        v = fx.Int64(a["peer"][0])
        for j in range_constexpr(1, MAX_TP):
            v = (p == i32(j)).select(fx.Int64(a["peer"][j]), v)
        return v

    def peer_rs(a, p, key, nbytes=None):
        """Resource over region `key` of peer p's arena (p: python int)."""
        return rsrc(fx.Int64(a["peer"][p]) + fx.Int64(a[key]), nbytes)

    def own_rs(a, key, nbytes=None):
        return rsrc(a["mine"] + fx.Int64(a[key]), nbytes)

    def route_region_bytes(ttot):
        return ttot * i32(TOPK * ROW_B)

    # ---------------------------- numerics ----------------------------
    def amax(vals):
        am = _fabs_f32(vals[0])
        for v in vals[1:]:
            am = am.maximumf(_fabs_f32(v))
        return am

    def fp4_pack(vals, qs):
        """2 * n floats -> n packed E2M1 pairs (one dword for n <= 4)."""
        pk = _u(i32(0))
        for k in range(len(vals) // 2):
            pk = rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, pk, _u(vals[2 * k]), _u(vals[2 * k + 1]), _u(qs), k
            )
        return fx.Int32(pk)

    def fp8x4_pack(f, qs):
        """4 floats -> one dword of E4M3 (scaled by 1/qs)."""
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
        """(8 E4M3 as v2i32, their E8M0 byte) -> 8 floats."""
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
        """v4i of 8 packed bf16 -> 8 floats."""
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

    # LL packets [d0, tag, d1, tag]: tag = epoch << 8 | E8M0 of the 8 E4M3 in d0, d1
    def ll_tag(a, e8):
        return ((a["epoch"] & i32(0xFFFFFF)) << i32(8)) | e8

    def ll_ok(a, q):
        want = a["epoch"] & i32(0xFFFFFF)
        return (fx.Int32(q[1]).shrui(i32(8)) == want) & (
            fx.Int32(q[3]).shrui(i32(8)) == want
        )

    def ll_vals(q):
        sc = e8_scale(q[1])
        return fp8x4_unpack(fx.Int32(q[0]), sc) + fp8x4_unpack(fx.Int32(q[2]), sc)

    def ll_pending(a, lane, pkts):
        """1 on every lane while some live packet (q, live) lacks this launch's tag."""
        bad = i32(0)
        for q, live in pkts:
            bad = fx.max(bad, (live & (ll_ok(a, q) == fx.Boolean(False))).select(i32(1), i32(0)))
        return wave_red(bad, lane, fx.max)

    def wave_red(v, lane, op):
        for k in (1, 2, 4, 8, 16, 32):
            v = op(v, i32(rocdl.ds_bpermute(T.i32, _u((lane ^ i32(k)) * i32(4)), _u(v))))
        return uni(v)

    def route_load(rs, region, ttot, ridx, col, take=None):
        """Loads of 8 route-row columns [col, col+8) of route ridx in a region;
        take false reads past the end (zeros)."""
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
        """v_permlane16_swap: odd 16-lane rows of x trade with even rows of y."""
        return _swap(rocdl.permlane16_swap, x, y)

    def swap32(x, y):
        """v_permlane32_swap: the upper 32 lanes of x trade with the lower of y."""
        return _swap(rocdl.permlane32_swap, x, y)

    def widen(v4):
        z = fx.Vector.filled(4, 0, fx.Int32)
        return _u(fx.Vector(v4).shuffle(z, list(range(8))))

    def mfma(a4, b4, cacc, sa, sb):
        return fx.Vector(
            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                _ty(V4F),
                [widen(a4), widen(b4), _u(cacc), 4, 4, 0, _u(i32(sa)), 0, _u(i32(sb))],
            )
        )

    def act_batch(gs, us):
        return _activation_mul_batch(
            gs,
            us,
            act=act,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            swiglu_limit=7.0,
        )

    # ---------------------------- traced helpers ----------------------------
    @traced
    def cbar(L, tid):
        """Barrier over the NW compute waves (s_barrier would wait for all waves)."""
        wait_lgkm0()
        if (tid % i32(64)) == i32(0):
            g = lds_ld_acq(L, L_CTL + C_BARGEN * 4)
            old = lds_atomic_add(
                L, L_CTL + C_BARCNT * 4, 1, ACQ_REL
            )
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
        return fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))

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

    # ---------------------------- route scan ----------------------------
    @traced
    def gather_routes(L, tid, ids_addr, tw_addr, ttot, expert):
        """Collect every route to `expert` into LDS (rix, wt); return the count."""
        if tid == i32(0):
            lds_st(L, L_CTL + C_CNT * 4, i32(0))
        cbar(L, tid)
        n = ttot * i32(TOPK)
        rid = rsrc(ids_addr)
        if const_expr(MLL):
            # LL routing: a route's id and weight are words 0 and 2 of its packet
            for i_ in range(tid, n, i32(NT)):
                i = i32(i_)
                pk = fx.Vector(bld(rid, i * i32(16), 0, V4I, 0))
                _gather_one(L, i, fx.Int32(pk[0]) == expert, fx.Int32(pk[2]))
        else:
            n4 = n // i32(4)
            rtw = rsrc(tw_addr)
            # every load of a batch goes out before any compare (one HBM
            # latency per batch); batches of SCAN_G bound the registers
            for g0 in range_constexpr(0, SCAN_IT, SCAN_G):
                its = list(range(g0, min(g0 + SCAN_G, SCAN_IT)))
                vs, ws = [], []
                for it in its:
                    q = fx.min(tid + i32(it * NT), n4 - i32(1))
                    vs.append(fx.Vector(bld(rid, q * i32(16), 0, V4I, 0)))
                    ws.append(fx.Vector(bld(rtw, q * i32(16), 0, V4I, 0)))
                for x, it in enumerate(its):
                    q = tid + i32(it * NT)
                    for j in range_constexpr(4):
                        hit = (q < n4) & (fx.Int32(vs[x][j]) == expert)
                        _gather_one(L, q * i32(4) + i32(j), hit, fx.Int32(ws[x][j]))
            for idx_ in range(n4 * i32(4) + tid, n, i32(NT)):
                idx = i32(idx_)
                e = g_ld_i32(fx.Int64(ids_addr) + fx.Int64(idx) * fx.Int64(4))
                wv = g_ld_i32(fx.Int64(tw_addr) + fx.Int64(idx) * fx.Int64(4))
                _gather_one(L, idx, e == expert, wv)
        cbar(L, tid)
        cnt = lds_ld_i32(L, L_CTL + C_CNT * 4)
        cbar(L, tid)
        return fx.min(cnt, i32(TMAX))

    @traced
    def _gather_one(L, idx, hit, wv):
        if hit:
            slot = lds_atomic_add(L, L_CTL + C_CNT * 4, 1)
            if slot < i32(TMAX):
                lds_st(L, L_RIX + slot * i32(4), idx)
                lds_st(L, L_WT + slot * i32(4), wv)

    # ---------------------------- GEMM1 ----------------------------
    def act_quant_store(L, lane, w, accg, accu, nb):
        """Lane holds row m*16 + lane%16, inter columns t*16 + 4*(lane/16) + v of
        gate / up tile t; a 32-column MXFP4 group is 8 values in-lane x the 4
        lanes sharing lane%16."""
        q4 = lane // i32(16)
        for m in range_constexpr(MT):
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
            e8, qs = _e8m0_from_amax(am, max_norm=6.0)
            for t in range_constexpr(2):
                pk = fp4_pack(xs[t * 4 : t * 4 + 4], qs)
                cb = (nb * i32(128) + w * i32(32) + i32(t * 16) + q4 * i32(4)) // i32(2)
                lds_st(
                    L,
                    L_INTER + row * i32(SI_STRIDE) + cb,
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
        if lane == i32(0):
            spin_lds_ge(L, L_CTL + (C_ASEQ * 4) + buf * i32(4), q + i32(1))
        rocdl.sched_barrier(0)

    @traced
    def release_a_chunk(L, lane, buf):
        if lane == i32(0):
            lds_atomic_add(
                L, L_CTL + C_AFREE * 4 + buf * i32(4), 1, REL
            )

    @traced
    def gemm1(L, tid, a, expert, i0, nnb):
        lane = tid % i32(64)
        w = uni(tid // i32(64))
        e = uni(expert)
        rw = rsrc(fx.Int64(a["w1"]) + fx.Int64(e) * fx.Int64(2 * I * (H // 2)))
        rws = rsrc(a["w1s"])
        icol0 = uni(i0) + w * i32(32)
        nnb = uni(nnb)
        vg0 = icol0 * i32(H // 2) + lane * i32(16)
        vg1 = vg0 + i32(16 * (H // 2))
        vu0 = vg0 + i32(I * (H // 2))
        vu1 = vu0 + i32(16 * (H // 2))
        vsl = lane * i32(4)
        sg0 = uni((e * i32(2 * I) + icol0) // i32(32)) * i32(CH1 * 256)
        su0 = uni((e * i32(2 * I) + i32(I) + icol0) // i32(32)) * i32(CH1 * 256)
        ring = uni(L + i32(L_RING) + w * i32(NSK * SLOT))
        total = nnb * i32(KS1)
        qbase = lds_ld_i32(L, L_CTL + C_QBASE * 4 + w * i32(4))

        def issue_b(g, slot_idx):
            gc = fx.min(g, total - i32(1))
            pas = gc // i32(KS1)
            kk = gc - pas * i32(KS1)
            slot = ring + i32(slot_idx * SLOT)
            so = pas * i32(128 * (H // 2)) + kk * i32(1024)
            dma16(slot + i32(0), rw, vg0, so, nt=True)
            dma16(slot + i32(1024), rw, vg1, so, nt=True)
            dma16(slot + i32(2048), rw, vu0, so, nt=True)
            dma16(slot + i32(3072), rw, vu1, so, nt=True)
            sso = pas * i32((128 // 32) * CH1 * 256) + (kk // i32(2)) * i32(256)
            dma4(slot + i32(4096), rws, vsl, sg0 + sso)
            dma4(slot + i32(4096 + 256), rws, vsl, su0 + sso)

        for kk in range_constexpr(NSK):
            issue_b(i32(kk), kk)

        zero = fx.Vector.filled(4, 0.0, fx.Float32)
        for pas_ in range(i32(0), nnb, i32(1)):
            pas = i32(pas_)
            init = [zero] * (4 * MT)
            for g0_, st in range(
                pas * i32(KS1), (pas + i32(1)) * i32(KS1), i32(NSK), init=init
            ):
                g0 = i32(g0_)
                acc = list(st)
                for kk in range_constexpr(NSK):
                    g = g0 + i32(kk)
                    q = qbase + g // i32(KCS)
                    k = kk % KCS
                    buf = q % i32(NAB)
                    rocdl.sched_barrier(0)
                    if k == 0:
                        wait_a_chunk(L, lane, buf, q)
                    wait_vm((NSK - 1) * OPS)
                    slot = ring + i32(kk * SLOT)
                    b = [
                        lds_ld(slot, i32(t * 1024) + lane * i32(16), V4I, 16)
                        for t in range(4)
                    ]
                    sgw = lds_ld_i32(slot, i32(4096) + lane * i32(4))
                    suw = lds_ld_i32(slot, i32(4096 + 256) + lane * i32(4))
                    kh = kk & 1
                    g0s = (sgw >> i32(8 * (kh * 2))) & i32(0xFF)
                    g1s = (sgw >> i32(8 * (kh * 2 + 1))) & i32(0xFF)
                    u0s = (suw >> i32(8 * (kh * 2))) & i32(0xFF)
                    u1s = (suw >> i32(8 * (kh * 2 + 1))) & i32(0xFF)
                    abuf = L + i32(L_A) + buf * i32(RG * ACB)
                    asbuf = (
                        L
                        + i32(L_AS)
                        + buf * i32(KCS * NSC_BLK * 64 * 4)
                        + i32(k * NSC_BLK * 64 * 4)
                    )
                    for m in range_constexpr(MT):
                        row = i32(m * 16) + lane % i32(16)
                        col = (i32(k * 4) + lane // i32(16)) ^ (row & i32(7))
                        af = lds_ld(abuf, row * i32(ACB) + col * i32(16), V4I, 16)
                        sa = fx.Int32(
                            lds_ld(asbuf, row * i32(4) + lane // i32(16), T.i8, 1)
                        ) & i32(0xFF)
                        acc[m * 2 + 0] = mfma(b[0], af, acc[m * 2 + 0], g0s, sa)
                        acc[m * 2 + 1] = mfma(b[1], af, acc[m * 2 + 1], g1s, sa)
                        acc[2 * MT + m * 2 + 0] = mfma(
                            b[2], af, acc[2 * MT + m * 2 + 0], u0s, sa
                        )
                        acc[2 * MT + m * 2 + 1] = mfma(
                            b[3], af, acc[2 * MT + m * 2 + 1], u1s, sa
                        )
                    wait_lgkm0()
                    rocdl.sched_barrier(0)
                    issue_b(g + i32(NSK), kk)
                    rocdl.sched_barrier(0)
                    if k == KCS - 1:
                        release_a_chunk(L, lane, buf)
                res = yield acc
            ag_stage_free(L, lane, a)
            act_quant_store(L, lane, w, res[: 2 * MT], res[2 * MT :], pas)
        _store_qbase(L, lane, w, qbase + nnb * i32(NCH))
        wait_vm(0)
        cbar(L, tid)

    @traced
    def _store_qbase(L, lane, w, v):
        if lane == i32(0):
            lds_st(L, L_CTL + C_QBASE * 4 + w * i32(4), v)

    # ---------------------------- A loader ----------------------------
    def unit_fields(L, a, u, ub):
        """(expert, i0, icnt, kind) of unit u."""
        if const_expr(DYN):
            f = [lds_ld_i32(L, L_CTL + (i32(C_UL) + u * i32(4) + i32(k)) * i32(4)) for k in range(4)]
            return f[0], f[1], f[2], f[3]
        return _unit_fields(L, a, u, ub)

    @traced
    def _unit_fields(L, a, u, ub):
        """Static: the first unit from LDS (read at entry), the rest from the table."""
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
            R = uni(lds_ld_acq(L, L_CTL + C_UROWS * 4))
            nnb = unit_fields(L, a, u, ub)[2] // i32(128)  # 0: GEMM2 column slice
            for r0_ in range(i32(0), R, i32(RG)):
                r0 = i32(r0_)
                rows = fx.min(R - r0, i32(RG))
                arow = []
                for j in range_constexpr(NA_ROWOPS):
                    row = i32(j * 8) + lane // i32(8)
                    col = (lane % i32(8)) ^ (row & i32(7))
                    rr = (row < rows).select(row, i32(0))
                    arow.append(
                        (lds_ld_i32(L, L_RIX + (r0 + rr) * i32(4)) // i32(TOPK))
                        * i32(H // 2)
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
                nq = nnb * i32(NCH)
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
                    if cidx < i32(NCH):
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
        """Chunk q issued; publish the oldest once ALOAD_DEPTH are in flight."""
        pub = lds_ld_i32(L, L_CTL + C_LPUB * 4)
        if q + i32(1) - pub >= i32(ALOAD_DEPTH):
            wait_vm(NA_L * (ALOAD_DEPTH - 1))
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

    # ---------------------------- GEMM2 ----------------------------
    @traced
    def report_chunk(L, lane, w, cidx):
        if lane == i32(0):
            lds_st_rel(L, L_CTL + C_DONE * 4 + w * i32(4), cidx)

    @traced
    def maybe_report(L, lane, w, gi, signal, rlag_wait, lag=1):
        """Report the chunk `lag` chunks back; rlag_wait ring loads follow its
        stores, so the wait does not drain the ring."""
        if signal & (((gi + i32(1)) % i32(GPC)) == i32(0)) & (gi + i32(1) > i32(lag * GPC)):
            wait_vm(rlag_wait)
            report_chunk(L, lane, w, (gi // i32(GPC)) - i32(lag))

    def store_route_fp8(rs, halves, rix, n0, q4, ok, oob, a):
        """One row's 64 columns as E4M3 + two E8M0. Each 32-column MX block
        lives in the row's four lanes (q4); after permlane32_swap lane q4 holds
        16 columns of half q4 >> 1 at 16 * (q4 & 1). Dead rows store past the
        end (dropped by the hardware)."""
        packs, e8s = [], []
        for half in range_constexpr(2):
            f = bf16x8_to_f32(fx.Vector.from_elements(halves[half], fx.Int32))
            am = amax(f)
            am = am.maximumf(am.shuffle_xor(i32(16), i32(64)))
            am = am.maximumf(am.shuffle_xor(i32(32), i32(64)))
            e8, qs = _e8m0_from_amax(am, max_norm=448.0)
            packs.append([fp8x4_pack(f[0:4], qs), fp8x4_pack(f[4:8], qs)])
            e8s.append(fx.Int32(e8) & i32(0xFF))
        lo, hi = [], []
        for dw in range_constexpr(2):
            x, y = swap32(packs[0][dw], packs[1][dw])
            lo.append(x)
            hi.append(y)
        col = n0 + (q4 >> i32(1)) * i32(32) + (q4 & i32(1)) * i32(16)
        if const_expr(DLL):
            tag = ll_tag(a, ((q4 >> i32(1)) == i32(0)).select(e8s[0], e8s[1]))
            off = (rix * i32(H) + col) * i32(2)
            for h in range_constexpr(2):
                d = lo if h == 0 else hi
                pkt = fx.Vector.from_elements([d[0], tag, d[1], tag], fx.Int32)
                bst(pkt, rs, ok.select(off + i32(16 * h), oob), 0, AUX_SC1)
            return
        ov = fx.Vector.from_elements(lo + hi, fx.Int32)
        bst(ov, rs, ok.select(rix * i32(H) + col, oob), 0, AUX_SC1)
        # the row's two scale bytes, from its q4 == 0 lane
        sc = fx.Int16(e8s[0] | (e8s[1] << i32(8)))
        soff = a["ttot"] * i32(TOPK * H) + rix * i32(H // 32) + n0 // i32(32)
        bst(sc, rs, (ok & (q4 == i32(0))).select(soff, oob), 0, AUX_SC1)

    @traced
    def gemm2(L, tid, a, expert, ks0, r0, rows, signal, NKS, pidx, gi_lo=None, gi_hi=None, colsig=None, pre=None):
        """Column groups [gi_lo, gi_hi) (default all) over NKS k-steps from ks0.
        colsig: a column slice's last row tile (count every chunk once stored);
        pre: runs once the ring's first loads are issued (column-slice import)."""
        NSK2 = _nsk2_for(NKS, G2)
        GPI = (NSK2 * NKS // math.gcd(NSK2, NKS)) // NKS
        assert G2 % GPI == 0
        WAIT_B2 = (NSK2 - 1) * OPS
        # a report lags at least the ring's depth of loads behind its stores
        # (only the ring's loads may be counted: stores and loads complete out
        # of order on gfx950)
        LAG = max(1, -(-NSK2 // (GPC * NKS)))
        RLAG = GPC * NKS * OPS
        lane = tid % i32(64)
        w = uni(tid // i32(64))
        e = uni(expert)
        ks0 = uni(ks0)
        rw = rsrc(fx.Int64(a["w2"]) + fx.Int64(e) * fx.Int64(H * (I // 2)))
        rws = rsrc(a["w2s"])
        # piece p writes its K-slice partial: p = 0 to the route's own row,
        # p > 0 to proutes[p - 1]; the push adds them up
        routes_bytes = route_region_bytes(a["ttot"])
        pidx = uni(pidx)
        dst = (pidx == i32(0)).select(
            fx.Int64(a["routes"]),
            fx.Int64(a["proutes"]) + fx.Int64(pidx - i32(1)) * fx.Int64(routes_bytes),
        )
        r_routes = rsrc(dst, routes_bytes)
        oob = routes_bytes
        ring = uni(L + i32(L_RING) + w * i32(NSK * SLOT))
        q4 = lane // i32(16)
        g_lo = i32(0) if gi_lo is None else uni(gi_lo)
        g_hi = i32(G2) if gi_hi is None else uni(gi_hi)

        def issue(q, slot_idx):
            qc = fx.min(q, g_hi * i32(NKS) - i32(1))
            gi = qc // i32(NKS)
            k = ks0 + qc - gi * i32(NKS)
            n0 = (w + i32(NW) * gi) * i32(64)
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

        for qq in range_constexpr(NSK2):
            issue(g_lo * i32(NKS) + i32(qq), qq)
        if const_expr(pre is not None):
            pre()
        zero = fx.Vector.filled(4, 0.0, fx.Float32)
        for gi0_ in range(g_lo, g_hi, i32(GPI)):
            gi0 = i32(gi0_)
            for j in range_constexpr(GPI):
                gi = gi0 + i32(j)
                acc = [zero] * (4 * MT)
                for k in range_constexpr(NKS):
                    sidx = (j * NKS + k) % NSK2
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
                    for m in range_constexpr(MT):
                        row = i32(m * 16) + lane % i32(16)
                        af_l.append(
                            lds_ld(
                                L,
                                i32(L_INTER) + row * i32(SI_STRIDE) + i32(k * 64) + q4 * i32(16),
                                V4I,
                                16,
                            )
                        )
                        sa_l.append(
                            fx.Int32(
                                lds_ld(
                                    L,
                                    i32(L_INTERS) + row * i32(I // 32) + i32(k * 4) + q4,
                                    T.i8,
                                    1,
                                )
                            )
                            & i32(0xFF)
                        )
                    rocdl.sched_barrier(0)
                    rocdl.s_setprio(1)
                    for m in range_constexpr(MT):
                        for t in range_constexpr(4):
                            acc[m * 4 + t] = mfma(
                                b[t], af_l[m], acc[m * 4 + t], sb[t], sa_l[m]
                            )
                    rocdl.s_setprio(0)
                    wait_lgkm0()
                    rocdl.sched_barrier(0)
                    issue(gi * i32(NKS) + i32(k + NSK2), (j * NKS + k + NSK2) % NSK2)
                    rocdl.sched_barrier(0)
                # epilogue: lane (row lane%16, q4) holds cols n0 + t*16 + 4*q4 + v;
                # permlane16_swap pairs q4 with q4^1 so each lane stores 8
                # contiguous columns of tile ta + (q4 & 1) at 8 * (q4 >> 1)
                n0 = (w + i32(NW) * gi) * i32(64)
                ccol = (q4 & i32(1)) * i32(16) + (q4 >> i32(1)) * i32(8)
                for m in range_constexpr(MT):
                    R = i32(m * 16) + lane % i32(16)
                    ok = R < rows
                    Rc = ok.select(R, i32(0))
                    wt = fx.Float32(lds_ld(L, i32(L_WT) + (r0 + Rc) * i32(4), T.f32))
                    rix = lds_ld_i32(L, L_RIX + (r0 + Rc) * i32(4))
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
                    halves = []
                    for half in range_constexpr(2):
                        ta, tb = 2 * half, 2 * half + 1
                        lo, hi = [], []
                        for h in range_constexpr(2):
                            x, y = swap16(pk[ta][h], pk[tb][h])
                            lo.append(x)
                            hi.append(y)
                        if const_expr(FP8R):
                            halves.append(lo + hi)
                        else:
                            ov = fx.Vector.from_elements(lo + hi, fx.Int32)
                            col = n0 + i32(half * 32) + ccol
                            voff = (rix * i32(H) + col) * i32(2)
                            bst(ov, r_routes, ok.select(voff, oob), 0, AUX_SC1)
                    if const_expr(FP8R):
                        store_route_fp8(r_routes, halves, rix, n0, q4, ok, oob, a)
                if const_expr(not DLL):
                    maybe_report(L, lane, w, gi, signal, LAG * RLAG, LAG)
                    if const_expr(colsig is not None):
                        col_report(L, a, lane, gi, g_lo, colsig, RLAG)
        wait_vm(0)
        if const_expr(not DLL):
            _final_report(L, lane, w, signal)
            if const_expr(colsig is not None):
                if colsig:
                    col_done(L, a, lane, (g_hi - i32(1)) // i32(GPC))

    @traced
    def col_report(L, a, lane, gi, g_lo, colsig, rlag_wait):
        """The previous chunk's rows are stored (report lag as maybe_report)."""
        if colsig & (((gi + i32(1)) % i32(GPC)) == i32(0)) & (gi + i32(1) > g_lo + i32(GPC)):
            wait_vm(rlag_wait)
            col_done(L, a, lane, (gi // i32(GPC)) - i32(1))
        rocdl.sched_barrier(0)

    @traced
    def col_done(L, a, lane, cidx):
        """The last compute wave done with a column slice's chunk counts the slice in."""
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
        """GEMM2 of a unit's inter range: whole expert, a static piece (NPC
        equal pieces) or a dyn piece (P | KS2 pieces, region ks0 / NKS)."""
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

    # ---------------------------- compute units ----------------------------
    @traced
    def _drop_stale(tid, a):
        """Each CU drops its L1; on the first launch one CTA per XCD drops its
        L2 (later launches: finish() does it at the end, off the critical path)."""
        if tid < i32(64):
            asm("buffer_inv sc0")
        bid = i32(gpu.block_id("x"))
        if (tid < i32(64)) & (bid < i32(N_XCD)) & (a["epoch"] == i32(1)):
            fx.memory_fence(syncscope="one-as", ordering=ACQ)
            if tid == i32(0):
                g_st_sys(ctrl_at(a, i32(CTRL_XF) + bid), a["epoch"])

    @traced
    def _stale_dropped(tid, a):
        """This XCD's L2 no longer holds the previous launch's arena lines."""
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
        elif const_expr(not AR):
            ag_wait_meta(a, tid, a["epoch"])
        _stale_dropped(tid, a)
        cbar(L, tid)
        if const_expr(DYN):
            dyn_plan(L, tid, a)
        ub = lds_ld_i32(L, L_CTL + C_UNIT * 4)
        ue = lds_ld_i32(L, L_CTL + (C_UNIT + 1) * 4)
        if ub == ue:
            report_chunk(L, lane, w, i32(NCK - 1))
        for u_ in range(ub, ue, i32(1)):
            u = i32(u_)
            expert, i0, icnt, kind = unit_fields(L, a, u, ub)
            R = gather_routes(L, tid, a["ids"], a["tw"], a["ttot"], expert)
            if tid == i32(0):
                lds_st(L, L_CTL + C_UROWS * 4, R)
                lds_st_rel(L, L_CTL + C_USEQ * 4, u - ub + i32(1))
            if const_expr(XSPLIT):
                typ = kind & i32(0xFF)
                sig_unit = (kind & i32(UNIT_SIGNAL)) != i32(0)
                # a SIGNAL unit without route rows of its own signals right away
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
                else:
                    sig = (u == ue - i32(1)) & (r0 + i32(RG) >= R)
                    unit_tile(L, tid, a, expert, i0, icnt, r0, rows, sig)
                cbar(L, tid)
            if const_expr(XSPLIT):
                _xq_flag(a, tid, i0, icnt, kind)
                if const_expr(not DLL):
                    _xcol_empty(a, tid, kind, i0, R)
        if const_expr(XSPLIT):
            # column slices units[col0 : col0 + ncol] are claimed at run time
            # by the CTAs that run out of work first. Bounded loop (a dynamic
            # while around barriers + atomics miscompiles): at most twice the
            # mean per CTA, so the CTAs together always take all of them.
            nblk = i32(gpu.grid_dim.x)
            cap = (a["ncol"] + nblk - i32(1)) // nblk * i32(2) + i32(2)
            col_claim(L, tid, a)
            for it_ in range(i32(0), cap, i32(1)):
                _col_unit(L, tid, a)
        if const_expr(DLL):
            _cdone(tid, L)

    @traced
    def _cdone(tid, L):
        if tid == i32(0):
            lds_st_rel(L, L_CTL + C_CDONE * 4, i32(1))

    # ---------------------------- dynamic schedule ----------------------------
    @traced
    def dyn_plan(L, tid, a):
        """Active-expert bitmap, its prefix popcounts, the active list, and this
        CTA's units: slices [bid * U / C, (bid + 1) * U / C) of the n_act * P
        pieces of the active experts (bitmap + popcount: the same list on
        every CTA, unlike LDS atomics)."""
        n = a["ttot"] * i32(TOPK)
        rid = rsrc(a["ids"])
        for idx_ in range(tid, n, i32(NT)):
            idx = i32(idx_)
            e = fx.Int32(bld(rid, idx * i32(16 if MLL else 4), 0, T.i32, 0))
            lds_atomic_or(L, L_DYN + (e >> i32(5)) * i32(4), i32(1) << (e & i32(31)))
        cbar(L, tid)
        _dyn_prefix(L, tid)
        cbar(L, tid)
        for it in range_constexpr((NE + NT - 1) // NT):
            _dyn_list(L, tid + i32(it * NT))
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
                rank = lds_ld_i32(L, L_DYN + (i32(NBW) + (e >> i32(5))) * i32(4)) + _ctpop(wv & low)
                lds_st(L, L_DYN + (i32(2 * NBW) + rank) * i32(4), e)

    @traced
    def _dyn_units(L, tid):
        if tid == i32(0):
            nact = lds_ld_i32(L, L_CTL + C_NACT * 4)
            C = i32(gpu.grid_dim.x)
            # P: least max per-CTA slices ceil(nact * P / C) * KS2 / P (ties: fewer pieces)
            P = i32(DYN_PS[0])
            best = ((nact * i32(DYN_PS[0]) + C - i32(1)) // C) * i32(KS2 // DYN_PS[0])
            for q in DYN_PS[1:]:
                cost = ((nact * i32(q) + C - i32(1)) // C) * i32(KS2 // q)
                better = cost < best
                P = better.select(i32(q), P)
                best = better.select(cost, best)
            U = nact * P
            bid = i32(gpu.block_id("x"))
            u_lo = bid * U // C
            u_hi = (bid + i32(1)) * U // C
            icnt = i32(I) // P
            for u_ in range(u_lo, u_hi, i32(1)):
                u = i32(u_)
                j = u // P
                ul = L_CTL + (i32(C_UL) + (u - u_lo) * i32(4)) * i32(4)
                lds_st(L, ul, lds_ld_i32(L, L_DYN + (i32(2 * NBW) + j) * i32(4)))
                lds_st(L, ul + i32(4), (u - j * P) * icnt)
                lds_st(L, ul + i32(8), icnt)
                lds_st(L, ul + i32(12), i32(0))
            lds_st(L, L_CTL + C_DYNP * 4, P)
            lds_st(L, L_CTL + C_UNIT * 4, i32(0))
            lds_st(L, L_CTL + (C_UNIT + 1) * 4, u_hi - u_lo)
            lds_st_rel(L, L_CTL + C_PLAN * 4, i32(1))

    # ---------------------------- column split ----------------------------
    @traced
    def unit_tile_x(L, tid, a, expert, i0, icnt, kind, r0, rows, R):
        """UNIT_G1X: GEMM1 inter slice exporting its MXFP4 intermediate;
        UNIT_G2COL: GEMM2 column groups [i0, i0 + groups) over the whole inter
        dim, on the imported intermediate; else a normal unit."""
        k = kind & i32(0xFF)
        if k == i32(UNIT_G2COL):
            if tid == i32(0):
                lds_st(L, L_CTL + C_COLJ * 4, kind.shrui(i32(XQ_SHIFT)))
            gemm2(L, tid, a, expert, i32(0), r0, rows, fx.Boolean(False), KS2, i32(0),
                  i0, i0 + (kind.shrui(i32(8)) & i32(0xFF)), r0 + i32(RG) >= R,
                  pre=lambda: xq_import(L, tid, a, kind, r0, rows))
        else:
            if k == i32(UNIT_G1X):
                gemm1(L, tid, a, expert, i0, icnt // i32(128))
                xq_export(L, tid, a, i0, icnt, r0, rows)
            else:
                sig = ((kind & i32(UNIT_SIGNAL)) != i32(0)) & (r0 + i32(RG) >= R)
                unit_tile(L, tid, a, expert, i0, icnt, r0, rows, sig)

    def xg_rs(a):
        """The route-indexed intermediate scratch: MXFP4 rows, then E8M0 rows."""
        rx = rsrc(a["xg"])
        rxs = rsrc(fx.Int64(a["xg"]) + fx.Int64(a["ttot"] * i32(TOPK * (I // 2))))
        return rx, rxs

    @traced
    def xq_export(L, tid, a, i0, icnt, r0, rows):
        """Copy this tile's intermediate slice [i0, i0 + icnt) to the scratch."""
        u16 = icnt // i32(32)  # 16 B units per row
        rx, rxs = xg_rs(a)
        for q_ in range(tid, rows * u16, i32(NT)):
            q = i32(q_)
            row = q // u16
            c = q - row * u16
            rix = lds_ld_i32(L, L_RIX + (r0 + row) * i32(4))
            v = lds_ld(L, i32(L_INTER) + row * i32(SI_STRIDE) + c * i32(16), V4I, 16)
            bst(v, rx, rix * i32(I // 2) + i0 // i32(2) + c * i32(16), 0, AUX_SYS)
        nsd = icnt // i32(128)  # scale dwords per row
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
        """Every inter slice this unit exported is in the scratch."""
        s = tid - i0 // i32(128)
        if ((kind & i32(0xFF)) == i32(UNIT_G1X)) & (s >= i32(0)) & (s < icnt // i32(128)):
            j = kind.shrui(i32(XQ_SHIFT))
            g_st_sys(ctrl_at(a, i32(CTRL_XQ) + j * i32(XQ_P) + tid), a["epoch"])

    @traced
    def xq_import(L, tid, a, kind, r0, rows):
        """Wait for every slice of the expert's intermediate, then load this
        tile's rows of it into the GEMM2 operand."""
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
        """A column slice without rows counts itself into its chunks here."""
        if ((kind & i32(0xFF)) == i32(UNIT_G2COL)) & (R == i32(0)) & (tid == i32(0)):
            c0 = gi // i32(GPC)
            for c_ in range(c0, c0 + (kind.shrui(i32(8)) & i32(0xFF)) // i32(GPC), i32(1)):
                _signal_col(a, a["epoch"], i32(c_), kind.shrui(i32(XQ_SHIFT)))

    @traced
    def _col_unit(L, tid, a):
        """Run the claimed column slice (if any is left), claim the next."""
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
            col_claim(L, tid, a)

    def claim_addr(a, bank_off):
        bank = (a["epoch"] + i32(bank_off)) & i32(1)
        return ctrl_at(a, i32(CTRL_CLM) + bank * i32(LRDY_STRIDE))

    @traced
    def col_claim(L, tid, a):
        """Claim the next column slice into every compute wave's C_CLAIM slot
        (one lane's atomic: per-lane atomics to one line queue up)."""
        cbar(L, tid)  # every wave read the previous claim
        if tid == i32(0):
            v = g_add_agent(claim_addr(a, 0), 1)
            for w in range_constexpr(NW):
                lds_st(L, L_CTL + (C_CLAIM * 4) + i32(w * 4), v)
        cbar(L, tid)

    @traced
    def claim_reset(tid, a):
        """CTA 0 zeroes the idle claim bank for the next launch (every variant:
        an engine may alternate variants)."""
        if (tid == i32(0)) & (i32(gpu.block_id("x")) == i32(0)):
            g_st_sys(claim_addr(a, 1), i32(0))

    # ---------------------------- ReduceScatter ----------------------------
    def push_units(ttot):
        """Push units per chunk (TB tokens each)."""
        nblk = i32(gpu.grid_dim.x)
        return fx.min(i32(2) * nblk, (ttot + i32(TB - 1)) // i32(TB))

    def push_first_unit(ns, cidx):
        """CTA b pushes units b - rot, + nblk, ... of chunk cidx (rot = cidx * ns
        mod nblk): the chunks rotate over all CTAs."""
        nblk = i32(gpu.grid_dim.x)
        rot = (cidx * ns) % nblk
        return (i32(gpu.block_id("x")) - rot + nblk) % nblk

    def token_dst(a, t):
        """Receive-slot resource and row of token t's partial at its owner."""
        owner = t // a["m"]
        r_dst = rsrc(peer_sel(a, owner) + fx.Int64(a["off_part"]))
        return r_dst, a["rank"] * a["mmax"] + (t - owner * a["m"])

    @traced
    def push_chunk(L, tid, a, epoch, cidx):
        """Sum this CTA's tokens' top-k route rows of the chunk (unit u: tokens
        u, u + ns, ...) and push them to their owners."""
        lane = tid % i32(64)
        ttot = a["ttot"]
        nblk = gpu.grid_dim.x
        ns = push_units(ttot)
        u0 = push_first_unit(ns, cidx)
        c0 = cidx * i32(CW)
        routes_bytes = route_region_bytes(ttot)
        r_routes = rsrc(a["routes"], routes_bytes)
        # pieces 1.. of split experts' partial rows; others read past the end (0)
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
                    row = []
                    for k in range_constexpr(TOPK):
                        kv = []
                        for j in range_constexpr(VPL):
                            v = fx.min(lane + i32(j * 64), i32(CW // 8 - 1))
                            ridx = t * i32(TOPK) + i32(k)
                            kv.append(route_load(r_routes, i32(0), ttot, ridx, c0 + v * i32(8)))
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
                        if const_expr(NPC > 1):
                            # tokens routed to a split expert add pieces 1..
                            eids = [
                                g_ld_i32(ids_base + fx.Int64((t * i32(TOPK) + i32(k)) * i32(4)))
                                for k in range(TOPK)
                            ]
                            hit = eids[0] >= a["piece_e0"]
                            for k in range_constexpr(1, TOPK):
                                hit = hit | (eids[k] >= a["piece_e0"])
                            _push_split_token(
                                a, r_dst, prow, c0, r_pr, v, acc, live, hit, eids, t, routes_bytes
                            )
                        else:
                            _push_store(a, r_dst, prow, c0, v, acc, live)
        if const_expr(not LL):
            wait_vm(0)
            _push_done(a, lane, epoch, cidx, ns, n)

    @traced
    def _push_split_token(a, r_dst, prow, c0, r_pr, v, acc, live, hit, eids, t, rbytes):
        """Add split pieces' partials only for a token routed to a split expert."""
        if hit:
            vc = fx.min(v, i32(CW // 8 - 1))
            tot = acc
            for k in range_constexpr(TOPK):
                split = eids[k] >= a["piece_e0"]
                for p in range_constexpr(1, NPC):
                    ld = route_load(
                        r_pr, i32(p - 1) * rbytes, a["ttot"],
                        t * i32(TOPK) + i32(k), c0 + vc * i32(8), split,
                    )
                    tot = [x + y for x, y in zip(tot, route_decode(ld))]
            _push_store(a, r_dst, prow, c0, v, tot, live)
        else:
            _push_store(a, r_dst, prow, c0, v, acc, live)

    def part_offs(a, prow, col):
        """Receive-area byte offsets of columns [col, col+8) of partial row prow:
        E4M3 rows, then one E8M0 per 32 columns."""
        soff = a["tp"] * a["mmax"] * i32(H) + prow * i32(H // 32) + col // i32(32)
        return prow * i32(H) + col, soff

    @traced
    def _push_store(a, r_dst, prow, c0, v, acc, ok):
        """Push 8 columns [c0 + 8 v, +8) of a partial row as E4M3 (a 32-column
        MX block spans 4 adjacent lanes), or as an LL packet."""
        col = c0 + v * i32(8)
        am = amax(acc)
        am = am.maximumf(am.shuffle_xor(i32(1), i32(64)))
        am = am.maximumf(am.shuffle_xor(i32(2), i32(64)))
        e8, qs = _e8m0_from_amax(am, max_norm=448.0)
        d0, d1 = fp8x4_pack(acc[0:4], qs), fp8x4_pack(acc[4:8], qs)
        if const_expr(LL):
            tag = ll_tag(a, fx.Int32(e8) & i32(0xFF))
            pkt = fx.Vector.from_elements([d0, tag, d1, tag], fx.Int32)
            if ok:
                bst(pkt, r_dst, (prow * i32(H) + col) * i32(2), 0, AUX_SYS)
        else:
            doff, soff = part_offs(a, prow, col)
            # four blocks' scales in one dword, from the first lane of 16
            e = (fx.Int32(e8) & i32(0xFF)).bitcast(fx.Float32)
            sc = fx.Int32(e8) & i32(0xFF)
            for k in range_constexpr(1, 4):
                sc = sc | (e.shuffle_xor(i32(4 * k), i32(64)).bitcast(fx.Int32) << i32(8 * k))
            if ok:
                bst(fx.Vector.from_elements([d0, d1], fx.Int32), r_dst, doff, 0, AUX_SYS)
                if (v & i32(15)) == i32(0):
                    bst(sc, r_dst, soff, 0, AUX_SYS)

    def sc_addr(a, cidx, slot):
        return ctrl_at(a, i32(CTRL_SC) + (cidx * i32(SC_LINES) + slot) * i32(LRDY_STRIDE))

    @traced
    def _push_done(a, lane, epoch, cidx, ns, n):
        """Count this CTA's n pushed units; the last raises the chunk's flag
        at every peer."""
        if (lane == i32(0)) & (n > i32(0)):
            cnt_addr = sc_addr(a, cidx, i32(SC_PUSH))
            if g_add_agent(cnt_addr, n) + n == ns:
                g_st_sys(cnt_addr, i32(0))
                fo = fx.Int64((i32(FLAG_RDY) + cidx * i32(MAX_TP) + a["rank"]) * i32(4))
                for p in range_constexpr(MAX_TP):
                    if i32(p) < a["tp"]:
                        g_st_sys(fx.Int64(a["peer"][p]) + fx.Int64(a["off_flag"]) + fo, epoch)

    def fin_split(a):
        """Final CTAs per output row: with fewer rows than CTAs, row r's chunks
        are dealt over S CTAs (CTA r + s * m takes chunks c = s mod S)."""
        m = fx.max(a["m"], i32(1))
        nblk = i32(gpu.grid_dim.x)
        return (m < nblk).select(fx.min(i32(NCK), nblk // m), i32(1))

    def fin_key(a):
        """This CTA's first output row (its rows: key, key + nblk, ...)."""
        bid = i32(gpu.block_id("x"))
        return (fin_split(a) > i32(1)).select(bid % fx.max(a["m"], i32(1)), bid)

    def fin_owned(a):
        """Bit mask of the chunks this CTA finalizes."""
        bid = i32(gpu.block_id("x"))
        m = fx.max(a["m"], i32(1))
        S = fin_split(a)
        sidx = bid // m
        own = i32(0)
        for c in range_constexpr(NCK):
            own = own | ((i32(c) % S) == sidx).select(i32(1 << c), i32(0))
        return (bid < m * S).select(own, i32(0))

    def recv_pkts(a, r_recv, row, rstride, c0, vc):
        """Every rank's 16 B at columns [c0 + 8 vc, +8) of receive row row."""
        out = []
        for p in range_constexpr(MAX_TP):
            pc = fx.min(i32(p), a["tp"] - i32(1))
            off = ((pc * rstride + row) * i32(H) + c0 + vc * i32(8)) * i32(2)
            out.append(fx.Vector(bld(r_recv, off, 0, V4I, AUX_SYS)))
        return out

    @traced
    def final_chunk(tid, a, cidx):
        """Sum every rank's partial of this CTA's output rows of the chunk."""
        lane = tid % i32(64)
        nblk = gpu.grid_dim.x
        c0 = cidx * i32(CW)
        r_recv = own_rs(a, "off_part")
        step = (fin_split(a) > i32(1)).select(a["m"], i32(nblk))
        for row_ in range(fin_key(a), a["m"], step):
            row = i32(row_)
            for j in range_constexpr(VPL):
                v = lane + i32(j * 64)
                vc = fx.min(v, i32(CW // 8 - 1))
                if const_expr(LL):
                    _final_ll(tid, a, r_recv, row, c0, v, vc)
                    continue
                # every rank's partial in flight at once, then the sum
                lds_ = []
                for p in range_constexpr(MAX_TP):
                    pc = fx.min(i32(p), a["tp"] - i32(1))
                    doff, soff = part_offs(a, pc * a["mmax"] + row, c0 + vc * i32(8))
                    lds_.append(
                        (bld(r_recv, doff, 0, V2I, AUX_SYS), bld(r_recv, soff, 0, T.i8, AUX_SYS))
                    )
                acc = [fx.Float32(0.0)] * 8
                for p in range_constexpr(MAX_TP):
                    acc = sum_live(acc, fp8x8_decode(lds_[p]), i32(p) < a["tp"])
                _y_store(a, row, c0, v, pack_bf16x8(acc))
        if const_expr(AR):
            _yag_flag(a, lane, cidx)

    @traced
    def _final_ll(tid, a, r_recv, row, c0, v, vc):
        """LL final: poll every rank's packets until tagged, then sum them."""
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
        _y_store(a, row, c0, v, pack_bf16x8(acc))

    @traced
    def _y_store(a, row, c0, v, o):
        if v < i32(CW // 8):
            if const_expr(AR):
                # the output AllGather: every rank's yall gets the row
                off = ((a["rank"] * a["m"] + row) * i32(H) + c0 + v * i32(8)) * i32(2)
                for p in range_constexpr(TPC):
                    bst(o, peer_rs(a, p, "off_yall"), off, 0, AUX_SYS)
            else:
                bst(o, rsrc(a["y"]), (row * i32(H) + c0 + v * i32(8)) * i32(2), 0, 0)

    @traced
    def _yag_flag(a, lane, cidx):
        """Output chunk cidx of this CTA's rows landed everywhere."""
        wait_vm(0)
        if lane == i32(0):
            idx = i32(FLAG_YAG) + (cidx * i32(MAX_TP) + a["rank"]) * i32(NCTA_MAX) + fin_key(a)
            for p in range_constexpr(TPC):
                bst(a["epoch"], peer_rs(a, p, "off_flag"), idx * i32(4), 0, AUX_SYS)

    def _yag_pending(a, lane, epoch):
        """1 while some (chunk, rank) of this CTA's output rows has not arrived."""
        rf = own_rs(a, "off_flag")
        bid = i32(gpu.block_id("x"))
        pend = i32(0)
        for it in range_constexpr((NCK * MAX_TP + 63) // 64):
            e = i32(it * 64) + lane
            cidx = e // i32(MAX_TP)
            src = e - cidx * i32(MAX_TP)
            ok = (cidx < i32(NCK)) & (src < a["tp"])
            idx = i32(FLAG_YAG) + (fx.min(cidx, i32(NCK - 1)) * i32(MAX_TP) + src) * i32(NCTA_MAX) + bid
            f = fx.Int32(bld(rf, idx * i32(4), 0, T.i32, AUX_SYS))
            pend = fx.max(pend, (ok & (f < epoch)).select(i32(1), i32(0)))
        return wave_red(pend, lane, fx.max)

    @traced
    def yag_wait(tid, a, epoch):
        """AR: every rank's output rows are in yall (before the end-of-launch L2 drop)."""
        if (tid < i32(64)) & (i32(gpu.block_id("x")) < a["m"]):
            lane = tid % i32(64)
            pend = _yag_pending(a, lane, epoch)
            t0 = _now()
            while (pend != i32(0)) & _alive(t0):
                rocdl.s_sleep(1)
                pend = _yag_pending(a, lane, epoch)
            _report_if(a, (pend != i32(0)) & (lane == i32(0)), ERR_YAG)

    def comm_idle(a):
        """No push unit and no output row in this CTA (same on every rank)."""
        bid = i32(gpu.block_id("x"))
        ns = push_units(a["ttot"])
        # chunk c's units sit on CTAs [c * ns, (c + 1) * ns) (push_first_unit)
        some = (ns * i32(NCK) >= i32(gpu.grid_dim.x)) | (bid < ns * i32(NCK))
        return (some.select(i32(0), i32(1)) == i32(1)) & (fin_owned(a) == i32(0))

    @traced
    def claim_push(L, lane, w, a, epoch):
        """Claim the next push chunk iff it is ready (CAS on the checked value).
        Result through this wave's mailbox: chunk index or -1."""
        if lane == i32(0):
            cc = lds_ld_acq(L, L_CTL + C_LRED * 4)
            rdy = i32(0)
            # a finished stage stops polling (uncached loads on shared lines)
            if cc < i32(NCK):
                v = g_ld_sys(ctrl_at(a, i32(CTRL_LRDY) + fx.min(cc, i32(NCK - 1)) * i32(LRDY_STRIDE)))
                rdy = (v >= epoch).select(i32(1), i32(0))
            want = (cc < i32(NCK)) & (rdy == i32(1))
            got = lds_cas(L, L_CTL + C_LRED * 4, cc, want.select(cc + i32(1), cc))
            res = (want & (got == cc)).select(cc, i32(-1))
            lds_st(L, L_CTL + (C_MBOX * 4) + w * i32(4), res)
        rocdl.sched_barrier(0)
        return uni(lds_ld_acq(L, L_CTL + (C_MBOX * 4) + w * i32(4)))

    @traced
    def claim_final(L, lane, w, a, epoch):
        """Claim a chunk whose partials all arrived and no wave took yet: the
        next CLAIM_W unclaimed chunks are checked together (the last chunks
        land about at once). C_PULL counts the claims."""
        if lane == i32(0):
            boff = L_CTL + C_FBITS * 4
            bits = lds_ld_acq(L, boff)
            ndone = lds_ld_acq(L, L_CTL + C_PULL * 4)
            cc = _ctpop(bits ^ (bits + i32(1))) - i32(1)  # lowest unclaimed chunk
            res = i32(-1)
            if (ndone < i32(NCK)) & (cc < i32(NCK)):
                rdys = [_final_ready(a, fx.min(cc + i32(d), i32(NCK - 1)), epoch)
                        for d in range(CLAIM_W)]
                r2 = i32(-1)
                for d in range_constexpr(CLAIM_W):
                    c = fx.min(cc + i32(d), i32(NCK - 1))
                    free = ((bits >> c) & i32(1)) == i32(0)
                    try_ = (r2 < i32(0)) & (cc + i32(d) < i32(NCK)) & rdys[d] & free
                    old = lds_atomic_or(L, boff, try_.select(i32(1) << c, i32(0)))
                    won = try_ & (((old >> c) & i32(1)) == i32(0))
                    r2 = won.select(c, r2)
                lds_atomic_add(L, L_CTL + C_PULL * 4, (r2 >= i32(0)).select(i32(1), i32(0)))
                res = r2
            lds_st(L, L_CTL + (C_MBOX * 4) + w * i32(4), res)
        rocdl.sched_barrier(0)
        return uni(lds_ld_acq(L, L_CTL + (C_MBOX * 4) + w * i32(4)))

    def _final_ready(a, c, epoch):
        if const_expr(LL):
            # peek at the last packet of this CTA's first row from every rank,
            # so a wave never blocks in a final while it could push
            r_recv = own_rs(a, "off_part")
            r1 = fx.Boolean(True)
            for p in range_constexpr(MAX_TP):
                pc = fx.min(i32(p), a["tp"] - i32(1))
                off = ((pc * a["mmax"] + fin_key(a)) * i32(H) + c * i32(CW) + i32(CW - 8)) * i32(2)
                t = fx.Int32(bld(r_recv, off + i32(4), 0, T.i32, AUX_SYS))
                r1 = r1 & ((i32(p) >= a["tp"]) | (t.shrui(i32(8)) == (epoch & i32(0xFFFFFF))))
            return r1
        # the chunk's MAX_TP push flags share one 32 B line: two loads
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
        """Dyn: a push unit is one token; its K routes' P piece partials are
        loaded in one round trip, summed and pushed."""
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
                        lds_.append(route_load(r_routes, i32(0), ttot, ridx, c0 + vc * i32(8)))
                        for p in range_constexpr(1, NPC):
                            lds_.append(
                                route_load(r_pr, i32(p - 1) * routes_bytes, ttot, ridx,
                                           c0 + vc * i32(8), i32(p) < P)
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
        cr = claim_push(L, lane, w, a, epoch)
        if cr >= i32(0):
            if const_expr(DYN):
                push_chunk_dyn(L, tid, a, epoch, cr)
            else:
                push_chunk(L, tid, a, epoch, cr)
        cp = claim_final(L, lane, w, a, epoch)
        if cp >= i32(0):
            final_chunk(tid, a, cp)
        return ((cr >= i32(0)) | (cp >= i32(0))).select(i32(1), i32(0))

    @traced
    def comm_signal(L, tid, a, epoch):
        """Count this CTA into every chunk its compute waves finished since the
        last call (range taken with a CAS on C_NSIG; one lane per chunk); the
        rank's last CTA publishes the chunk ready."""
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
        if won == i32(1):
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
        """Count this CTA into chunk cidx: per XCD, then the XCD into the chunk."""
        nblk = i32(gpu.grid_dim.x)
        x = i32(gpu.block_id("x")) % i32(N_XCD)
        _signal_part(a, epoch, cidx, x, (nblk - x + i32(N_XCD - 1)) // i32(N_XCD))

    def _signal_col(a, epoch, cidx, j):
        """Count a column slice of expert slot j into chunk cidx (per shard j mod N_XCD)."""
        sh = j % i32(N_XCD)
        _signal_part(a, epoch, cidx, i32(SC_SHARD) + sh,
                     (i32(XCOL_PER_CHUNK) - sh + i32(N_XCD - 1)) // i32(N_XCD))

    @traced
    def _signal_part(a, epoch, cidx, slot, target):
        xa = sc_addr(a, cidx, slot)
        if g_add_agent(xa, 1) + i32(1) == target:
            g_st_sys(xa, i32(0))
            _signal_count(a, epoch, cidx)

    @traced
    def _signal_count(a, epoch, cidx):
        """Chunk parts: one per XCD with CTAs, plus one per column-slice shard."""
        cnt_addr = sc_addr(a, cidx, i32(SC_ALL))
        parts = fx.min(i32(gpu.grid_dim.x), i32(N_XCD)) + i32(min(XCOL_PER_CHUNK, N_XCD))
        if g_add_agent(cnt_addr, 1) + i32(1) == parts:
            g_st_sys(cnt_addr, i32(0))
            g_st_sys(ctrl_at(a, i32(CTRL_LRDY) + cidx * i32(LRDY_STRIDE)), epoch)

    @traced
    def signal_loop(L, tid, a, epoch):
        """Loader wave after its A chunks: only signals finished chunks."""
        t0 = _now()
        while (lds_ld_acq(L, L_CTL + C_NSIG * 4) < i32(NCK)) & _alive(t0):
            if comm_signal(L, tid, a, epoch) == i32(0):
                rocdl.s_sleep(1)

    @traced
    def comm_wave(L, tid, a, epoch):
        t0 = _now()
        while (
            (lds_ld_acq(L, L_CTL + C_NSIG * 4) < i32(NCK))
            | (lds_ld_acq(L, L_CTL + C_LRED * 4) < i32(NCK))
            | (lds_ld_acq(L, L_CTL + C_PULL * 4) < i32(NCK))
        ) & _alive(t0):
            p1 = comm_signal(L, tid, a, epoch)
            p2 = comm_work(L, tid, a, epoch)
            if (p1 | p2) == i32(0):
                rocdl.s_sleep(POLL_SLEEP)
        _report_if(a, (_now() - t0 >= fx.Int64(DEADLINE)) & ((tid % i32(64)) == i32(0)), ERR_COMM)

    @traced
    def comm_help(L, tid, a, epoch):
        t0 = _now()
        while (
            (lds_ld_acq(L, L_CTL + C_LRED * 4) < i32(NCK))
            | (lds_ld_acq(L, L_CTL + C_PULL * 4) < i32(NCK))
        ) & _alive(t0):
            if comm_work(L, tid, a, epoch) == i32(0):
                rocdl.s_sleep(POLL_SLEEP)

    # ---------------------------- AllGather ----------------------------
    # Every CTA owns token rows bid, bid + nblk, ... (at most AGR). They are
    # quantized into the GEMM2 operand area (idle until the first GEMM1 pass
    # ends); the comm wave sends them in K-chunk order, so peers start GEMM1
    # on chunk 0 while the rest is on the wire. One chunk in flight per wave:
    # deeper remote-store queues back up every CTA's local loads.
    AG_SROW = H // 2
    AG_SCB = AGR * AG_SROW
    assert AG_SCB + AGR * (H // 32) <= L_INTERS - L_INTER + RG * (I // 32)
    NCHA = H // 256
    NPRE = (NCHA + PRE_CH - 1) // PRE_CH  # input ReduceScatter column groups
    assert NPRE <= NPRE_MAX
    NMETA_MAX = 32  # CTAs sending routing metadata, 1 KB each (FLAG_AGM stride)
    # loader polls: the first looks at every chunk (small batches: one round
    # trip clears the tile), later ones at AG_WIN chunks (hundreds of CTAs
    # polling every flag slow the AllGather down)
    AG_WIN = max(1, 64 // TPC)

    def ag_split(a):
        """S > 1 with fewer rows than CTAs: a row's chunks are dealt round-robin
        over S CTAs, so every CTA's first chunk is among the first S."""
        m = fx.max(a["m"], i32(1))
        return fx.min(fx.max(i32(gpu.grid_dim.x) // m, i32(1)), i32(NCHA))

    def ag_rows(a):
        """This CTA's rows row0 + r * stride (r < nrows), chunks qb + k * qs (k < cnt)."""
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
        """Chunks per AR input group: PRE_CH, or one with split rows."""
        return (ag_split(a) > i32(1)).select(i32(1), i32(PRE_CH))

    def ag_npar(a):
        """Waves sending a CTA's chunks: 2 for AR with split rows (the push wave
        takes every other chunk of the per-chunk reduce/quantize/send chain)."""
        return ((ag_split(a) > i32(1)) & fx.Boolean(AR)).select(i32(2), i32(1))

    def _row32(rs, row, g, aux):
        """The 32 bf16 values of group g of a row, as floats."""
        f = []
        for v in range_constexpr(4):
            f += bf16x8_to_f32(
                bld(rs, ((row * i32(H)) + g * i32(32) + i32(v * 8)) * i32(2), 0, V4I, aux)
            )
        return f

    def ag_row_vals(a, rxl, i, g):
        """Row i of this rank's shard, group g; AR: the bf16-rounded sum of every
        rank's partial (own from the input, peers' from the pre slot)."""
        if const_expr(not AR):
            return _row32(rxl, i, g, 0)
        f = _row32(rxl, a["rank"] * a["m"] + i, g, 0)
        rpre = own_rs(a, "off_pre")
        for src in range_constexpr(TPC):
            vals = _row32(rpre, i32(src) * a["mmax"] + i, g, AUX_SYS)
            f = sum_live(f, vals, i32(src) != a["rank"])
        return [fx.Float32(x).to(fx.BFloat16).to(fx.Float32) for x in f]

    @traced
    def pre_send(lane, a):
        """AR, push wave: send this CTA's rows of every peer's input shard (bf16
        partials) to their owners group by group, each group's flag raised
        once its stores landed."""
        row0, stride, nrows, qb, qs, cnt = ag_rows(a)
        rank, m = a["rank"], a["m"]
        x_bytes = a["tp"] * m * i32(H * 2)
        rxl = rsrc(a["x"], x_bytes)
        pc = ag_group(a)
        pre_bytes = a["tp"] * a["mmax"] * i32(H * 2)
        for g_ in range(i32(0), (cnt + pc - i32(1)) // pc, i32(1)):
            g = i32(g_)
            lo = (qb + g * pc * qs) * i32(256)
            upg = fx.min(pc, cnt - g * pc) * i32(32)  # 16 B units per row
            for q_ in range(lane, nrows * upg, i32(64)):
                q = i32(q_)
                r = q // upg
                c = lo + (q - r * upg) * i32(8)
                i = row0 + r * stride
                # every peer's loads before any store (own slot reads past the end)
                vs = [
                    bld(
                        rxl,
                        (i32(p) != rank).select(((i32(p) * m + i) * i32(H) + c) * i32(2), x_bytes),
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
    def _pre_flag(a, lane, g):
        if lane == i32(0):
            fo = (i32(FLAG_PRE) + (g * i32(MAX_TP) + a["rank"]) * i32(NCTA_MAX) + i32(gpu.block_id("x"))) * i32(4)
            for p in range_constexpr(TPC):
                if i32(p) != a["rank"]:
                    g_st_sys(fx.Int64(a["peer"][p]) + fx.Int64(a["off_flag"]) + fx.Int64(fo), a["epoch"])

    @traced
    def pre_wait(lane, a, g):
        """AR, comm wave: every peer's group g of this CTA's rows arrived."""
        src = fx.min(lane, i32(TPC - 1))
        addr = (
            a["mine"]
            + fx.Int64(a["off_flag"])
            + fx.Int64((i32(FLAG_PRE) + (g * i32(MAX_TP) + src) * i32(NCTA_MAX) + i32(gpu.block_id("x"))) * i32(4))
        )
        live = (lane < i32(TPC)) & (lane != a["rank"])
        pend = wave_red((live & (g_ld_sys(addr) < a["epoch"])).select(i32(1), i32(0)), lane, fx.max)
        t0 = _now()
        while (pend != i32(0)) & _alive(t0):
            rocdl.s_sleep(1)
            pend = wave_red((live & (g_ld_sys(addr) < a["epoch"])).select(i32(1), i32(0)), lane, fx.max)
        _report_if(a, (pend != i32(0)) & (lane == i32(0)), ERR_FLAG)

    @traced
    def quant_chunks(L, t0, stride, a, k_lo, k_hi):
        """MXFP4-quantize chunks k in [k_lo, k_hi) (chunk qb + k * qs, eight
        32-column groups each) of this CTA's rows into the LDS staging."""
        row0, stride_r, nrows, qb, qstep, cnt = ag_rows(a)
        k_hi = fx.min(k_hi, cnt)
        nk = fx.max(k_hi - k_lo, i32(0))
        rxl = rsrc(a["x"])
        for q_ in range(t0, nrows * nk * i32(8), i32(stride)):
            q = i32(q_)
            r = q // (nk * i32(8))
            e = q - r * nk * i32(8)
            g = (qb + (k_lo + e // i32(8)) * qstep) * i32(8) + e % i32(8)
            f = ag_row_vals(a, rxl, row0 + r * stride_r, g)
            e8, qs = _e8m0_from_amax(amax(f), max_norm=6.0)
            words = [fp4_pack(f[dw * 8 : dw * 8 + 8], qs) for dw in range(4)]
            lds_st(
                L,
                L_INTER + r * i32(AG_SROW) + g * i32(16),
                fx.Vector.from_elements(words, fx.Int32),
                16,
            )
            lds_st(L, L_INTER + i32(AG_SCB) + r * i32(H // 32) + g, fx.Int8(e8), align=1)

    @traced
    def ag_send(L, lane, a, par=0):
        """Comm wave (and push wave 1 with par=1, see ag_npar): the payload chunk
        by chunk. Every chunk issues exactly 2 * TPC stores (dead rows store
        past the end), so wait_vm(2 * TPC) after chunk q means chunk q - 1
        landed and may be counted."""
        row0, stride, nrows, qb, qs, cnt = ag_rows(a)
        rank = a["rank"]
        if const_expr(not AR and par == 0 and not MLL):
            # the routing metadata first (with whole rows the payload would
            # queue it behind a megabyte per peer)
            mp = (ag_split(a) > i32(1)).select(i32(0), _meta_pending(a, lane, a["epoch"], own=True))
            t0 = _now()
            while (mp != i32(0)) & _alive(t0):
                rocdl.s_sleep(1)
                mp = _meta_pending(a, lane, a["epoch"], own=True)
        r = lane // i32(8)
        j = lane - r * i32(8)
        rok = r < nrows
        i = row0 + fx.min(r, fx.max(nrows - i32(1), i32(0))) * stride
        gslot = rank * a["m"] + i
        x_bytes = a["ttot"] * i32(H // 2)
        xs_bytes = a["ttot"] * i32(H // 32)
        rx = [peer_rs(a, p, "off_x", x_bytes) for p in range(TPC)]
        rs = [peer_rs(a, p, "off_xs", xs_bytes) for p in range(TPC)]
        pc = ag_group(a)
        # AR with split rows counts each chunk once it landed, before waiting
        # for the next chunk's inputs; otherwise one chunk stays in flight
        split = ag_split(a) > i32(1)
        pipe = (split & fx.Boolean(AR)) == fx.Boolean(False)
        npar = ag_npar(a)
        part = i32(par) < npar
        for k in range_constexpr(NCHA):
            mine = (i32(k) % npar) == i32(par)
            kl = (i32(k) < cnt) & mine
            q = fx.min(qb + i32(k) * qs, i32(NCHA - 1))
            if const_expr(AR and k >= 1):
                if split & mine & (i32(k) >= npar) & (i32(k) - npar < cnt):
                    wait_vm(0)
                    _ag_bump(a, lane, qb + (i32(k) - npar) * qs)
            if kl:
                ag_chunk(L, lane, a, k, q, pc, r, j, rok, gslot, rx, rs, x_bytes, xs_bytes)
            if const_expr(k == NCHA - 1):
                # every staged byte is in registers: GEMM1 may reuse the area
                wait_lgkm0()
                if part:
                    _ag_free(L, lane)
            if const_expr(k >= 1):
                if pipe & part & (i32(k - 1) < cnt):
                    if kl:
                        wait_vm(2 * TPC)
                    else:
                        wait_vm(0)
                    _ag_bump(a, lane, qb + i32(k - 1) * qs)
            # keep the scheduler from hoisting every chunk's LDS reads
            rocdl.sched_barrier(0)
        wait_vm(0)
        if pipe & part & (i32(NCHA - 1) < cnt):
            _ag_bump(a, lane, qb + i32(NCHA - 1) * qs)

    @traced
    def ag_chunk(L, lane, a, k, q, pc, r, j, rok, gslot, rx, rs, x_bytes, xs_bytes):
        """Send chunk q of this CTA's rows to every peer (2 * TPC stores)."""
        if const_expr(AR):
            if (i32(k) % pc) == i32(0):
                pre_wait(lane, a, i32(k) // pc)
                quant_chunks(L, lane, 64, a, i32(k), i32(k) + pc)
            wait_lgkm0()
            rocdl.sched_barrier(0)
        rr = fx.min(r, i32(AGR - 1))
        dv = lds_ld(L, L_INTER + rr * i32(AG_SROW) + q * i32(128) + j * i32(16), V4I, 16)
        sv = lds_ld(L, L_INTER + i32(AG_SCB) + rr * i32(H // 32) + q * i32(8), V2I, 8)
        doff = rok.select(gslot * i32(H // 2) + q * i32(128) + j * i32(16), x_bytes)
        soff = (rok & (j == i32(0))).select(gslot * i32(H // 32) + q * i32(8), xs_bytes)
        for p in range_constexpr(TPC):
            bst(dv, rx[p], doff, 0, AUX_SYS)
            bst(sv, rs[p], soff, 0, AUX_SYS)

    @traced
    def _ag_free(L, lane):
        if lane == i32(0):
            lds_atomic_add(L, L_CTL + C_AGFREE * 4, 1, REL)

    @traced
    def ag_stage_free(L, lane, a):
        """GEMM1 is about to write the GEMM2 operand: the staging sharing it is read out."""
        if lane == i32(0):
            spin_lds_ge(L, L_CTL + C_AGFREE * 4, ag_npar(a))
        rocdl.sched_barrier(0)

    def _ag_senders(a, q, x):
        """CTAs of XCD x that send K-chunk q: all with whole rows, rows' CTA set
        q mod S with split rows."""
        m = fx.max(a["m"], i32(1))
        S = ag_split(a)
        lo = (S > i32(1)).select((q % S) * m, i32(0))
        hi = (S > i32(1)).select(lo + m, i32(gpu.grid_dim.x))
        return (hi - x + i32(N_XCD - 1)) // i32(N_XCD) - (lo - x + i32(N_XCD - 1)) // i32(N_XCD)

    @traced
    def _ag_bump(a, lane, q):
        """Chunk q of this CTA's rows landed everywhere: count it per XCD, then
        the XCDs; the rank's last sender raises the chunk's flag at every peer."""
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
        """The first nmeta CTAs each copy 64 x 16 B of this rank's routing (ids,
        weights) to every peer, then raise their flag: peers' first, the own
        one once they are acknowledged (the payload waits on the own flag, so
        it cannot overtake the peers' flags on the links)."""
        bid = i32(gpu.block_id("x"))
        if bid < _ag_nmeta(a):
            n = a["m"] * i32(TOPK)
            n4 = n // i32(4)
            rid = rsrc(a["ids_in"])
            rtw = rsrc(a["tw_in"])
            dst0 = a["rank"] * n * i32(4)
            v = bid * i32(64) + lane
            idv = bld(rid, v * i32(16), 0, V4I, 0)
            wvv = bld(rtw, v * i32(16), 0, V4I, 0)
            e = n4 * i32(4) + lane  # the < 4 tail ints, by CTA 0
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
                for p in range_constexpr(TPC):
                    if i32(p) != a["rank"]:
                        bst(a["epoch"], peer_rs(a, p, "off_flag"), fo, 0, AUX_SYS)
                wait_vm(0)
                bst(a["epoch"], own_rs(a, "off_flag"), fo, 0, AUX_SYS)

    @traced
    def _ag_send_meta_ll(lane, a):
        """LL routing AllGather: each route as a packet [id, epoch, weight, epoch]
        to every peer (no completion wait, no flags)."""
        n = a["m"] * i32(TOPK)
        rid = rsrc(a["ids_in"])
        rtw = rsrc(a["tw_in"])
        big = i32(1 << 30)
        for i_ in range(i32(gpu.block_id("x")) * i32(64) + lane, n, i32(gpu.grid_dim.x) * i32(64)):
            i = i32(i_)
            e = fx.Int32(bld(rid, i * i32(4), 0, T.i32, 0))
            wv = fx.Int32(bld(rtw, i * i32(4), 0, T.i32, 0))
            pkt = fx.Vector.from_elements([e, a["epoch"], wv, a["epoch"]], fx.Int32)
            off = (a["rank"] * n + i) * i32(16)
            for p in range_constexpr(TPC):
                bst(pkt, peer_rs(a, p, "off_ids", big), off, 0, AUX_SYS)

    @traced
    def meta_ll_wait(tid, a):
        """Compute waves: every route packet of this launch landed."""
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
            f = fx.Int32(bld(rf, (i32(FLAG_AGM) + p * i32(32) + j) * i32(4), 0, T.i32, AUX_SYS))
            pend = fx.max(pend, (ok & (f < epoch)).select(i32(1), i32(0)))
        return wave_red(pend, lane, fx.max)

    @traced
    def ag_wait_meta(a, tid, epoch):
        """Every rank's routing metadata landed (checked by compute wave 0)."""
        if tid < i32(64):
            lane = tid % i32(64)
            pend = _meta_pending(a, lane, epoch)
            t0 = _now()
            while (pend != i32(0)) & _alive(t0):
                rocdl.s_sleep(1)
                pend = _meta_pending(a, lane, epoch)
            _report_if(a, (pend != i32(0)) & (lane == i32(0)), ERR_META)

    def _ag_first_pending(lane, a, epoch, c_lo, nwin):
        """The first K-chunk in [c_lo, c_lo + nwin) some rank has not landed
        (c_lo + nwin: none); the window's flags in one round trip."""
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
        """Loader: K-chunk cc of every row is in the arena; the chunks known to
        have arrived are kept in C_ARDY."""
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

    # ---------------------------- launch begin / end ----------------------------
    @traced
    def finish(tid, a, epoch):
        if tid == i32(0):
            bid = i32(gpu.block_id("x"))
            # this CTA's launch count (the next launch's epoch - 1)
            g_st_sys(ctrl_at(a, i32(CTRL_EPB) + bid), epoch)
            # the XCD's last CTA drops its L2 (arena lines peers rewrite next launch)
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

    @traced
    def init_lds(L, tid, a):
        """One writer per control int: done[] = -1, the epoch slot, the unit
        record (static schedule), stage counters preset for idle CTAs, else 0."""
        bid = i32(gpu.block_id("x"))
        if (tid >= i32(64)) & (tid < i32(64 + 6)):
            k = tid - i32(64)
            v = g_ld_i32(fx.Int64(a["cta_units"]) + fx.Int64(bid * i32(UNIT_REC) + k) * fx.Int64(4))
            lds_st(L, L_CTL + (i32(C_UNIT) + k) * i32(4), v)
        if (tid < i32(C_UNIT)) | ((tid >= i32(C_UNIT + 6)) & (tid < i32(C_XC + C_XC_N))):
            is_done = (tid >= i32(C_DONE)) & (tid < i32(C_DONE + NW))
            ep = g_ld_rel(ctrl_at(a, i32(CTRL_EPB) + bid), "agent") + i32(1)
            v = is_done.select(i32(-1), (tid == i32(C_EPOCH)).select(ep, i32(0)))
            # no push unit / no output row: nothing to claim
            stage = (tid == i32(C_LRED)) | (tid == i32(C_PULL))
            v = (stage & comm_idle(a)).select(i32(NCK), v)
            if const_expr(DLL):
                # push_dll polls the route packets: nothing to signal or claim
                v = ((tid == i32(C_LRED)) | (tid == i32(C_NSIG))).select(i32(NCK), v)
            # the final stage claims only its own chunks
            own = fin_owned(a)
            v = (tid == i32(C_PULL)).select(i32(NCK) - _ctpop(own), v)
            v = (tid == i32(C_FBITS)).select(i32((1 << NCK) - 1) & (own ^ i32(-1)), v)
            if const_expr(ARLL):
                v = (tid == i32(C_PULL)).select(i32(NCK), v)  # final_all_ll instead
            lds_st(L, L_CTL + tid * i32(4), v)

    # ---------------------------- LL route rows ----------------------------
    DLL_NWS = 4  # waves per CTA running push_dll items (the 4 non-compute waves)
    DLL_CARRY = 48  # route packets per lane a push poll may keep live (16 B each)

    def _dll_rows(a, t, c0, vc, P):
        """Token t's K x NPC route packets at columns [c0 + 8 vc, +8); pieces
        p >= P read past the end."""
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
        """This wave's (token, chunk) items: it = bid + ws * nblk + k * 4 * nblk."""
        nblk = i32(gpu.grid_dim.x)
        return i32(gpu.block_id("x")) + i32(ws) * nblk, nblk * i32(DLL_NWS)

    @traced
    def push_dll(L, tid, a, ws=0):
        """LL route rows: per (token, chunk) poll the K x P route packets until
        tagged, sum them, LL-push the sum to the owner (ARLL: to every rank)."""
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
                    # the polled packets are the data (no second read)
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
                        pend = ll_pending(a, lane, zip(_dll_rows(a, t, c0, vc, P), live))
                    pk = _dll_rows(a, t, c0, vc, P)
                _report_if(a, (pend != i32(0)) & (lane == i32(0)), ERR_COMM)
                acc = [fx.Float32(0.0)] * 8
                for k in range_constexpr(TOPK):
                    for p in range_constexpr(NPC):
                        acc = sum_live(acc, ll_vals(pk[k * NPC + p]), i32(p) < P)
                if const_expr(ARLL):
                    for pr in range_constexpr(TPC):
                        _push_store(a, peer_rs(a, pr, "off_part"), prow, c0, v, acc, v < i32(CW // 8))
                else:
                    _push_store(a, r_dst, prow, c0, v, acc, v < i32(CW // 8))
        if const_expr(ARLL):
            final_all_ll(L, tid, a, ws)

    @traced
    def final_all_ll(L, tid, a, ws=0):
        """ARLL: per (token, chunk) poll every rank's packets, sum them in rank
        order, write the local yall row."""
        lane = tid % i32(64)
        ttot = a["ttot"]
        trows = a["mmax"] * a["tp"]
        r_recv = own_rs(a, "off_part")
        ry = own_rs(a, "off_yall")
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
                    bst(pack_bf16x8(acc), ry, (t * i32(H) + c0 + v * i32(8)) * i32(2), 0, 0)

    # ---------------------------- roles / kernel ----------------------------
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
            elif tid < i32(NT + 128):
                a_loader(L, tid, a)
                if const_expr(DLL):
                    push_dll(L, tid, a, 1)
                signal_loop(L, tid, a, epoch)
                comm_help(L, tid, a, epoch)
            else:
                if const_expr(AR):
                    if tid < i32(NT + 192):
                        pre_send(tid % i32(64), a)
                    else:
                        ag_send(L, tid % i32(64), a, 1)
                if const_expr(DLL):
                    if tid < i32(NT + 192):
                        push_dll(L, tid, a, 2)
                    else:
                        push_dll(L, tid, a, 3)
                comm_help(L, tid, a, epoch)

    # explicit annotation: the module's postponed annotations cannot see LDS_BYTES
    Shared = fx.struct(
        type("Shared", (), {"__annotations__": {"buf": fx.Array[fx.Int8, LDS_BYTES, 16]}})
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
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        assert layout_tag and name  # (cache key)
        lds = fx.SharedAllocator().allocate(Shared).peek()
        L = uni(fx.Int32(fx.ptrtoint(lds.buf.ptr)))
        peers = [p0, p1, p2, p3, p4, p5, p6, p7]
        mine = peers[0]
        for j in range_constexpr(1, MAX_TP):
            mine = (rank == i32(j)).select(peers[j], mine)
        a = dict(
            w1=w1, w1s=w1s, w2=w2, w2s=w2s, x=x, ids_in=ids_in, tw_in=tw_in, y=y,
            routes=routes, proutes=proutes, ctrl=ctrl, units=units, cta_units=cta_units,
            peer=peers, mine=fx.Int64(mine), off_x=off_x, off_xs=off_xs, off_ids=off_ids,
            off_w=off_w, off_part=off_part, off_flag=off_flag, off_pre=off_pre,
            off_yall=off_yall, rank=rank, tp=tp, m=m, mmax=mmax, ttot=tp * m,
            piece_e0=piece_e0, xg=xg, col0=col0, ncol=ncol,
        )
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
        if const_expr(AR):
            # every rank has the routing of all tokens: no metadata AllGather
            a["ids"] = fx.Int64(ids_in)
            a["tw"] = fx.Int64(tw_in)
        else:
            a["ids"] = fx.Int64(mine) + off_ids
            a["tw"] = fx.Int64(mine) + off_w
            if (tid >= i32(NT)) & (tid < i32(NT + 64)):
                if const_expr(MLL):
                    _ag_send_meta_ll(tid % i32(64), a)
                else:
                    _ag_send_meta(tid % i32(64), a)
            quant_chunks(L, tid, NTT, a, i32(0), i32(NCHA))
        gpu.barrier()
        a["ax"] = fx.Int64(mine) + off_x
        a["axs"] = fx.Int64(mine) + off_xs
        roles(L, tid, a, epoch)
        gpu.barrier()
        if const_expr(AR and not ARLL):
            yag_wait(tid, a, epoch)
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
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        mega_moe_tp_kernel(
            w1, w1s, w2, w2s, x, ids_in, tw_in, y, routes, proutes, ctrl, units,
            cta_units, p0, p1, p2, p3, p4, p5, p6, p7, off_x, off_xs, off_ids, off_w,
            off_part, off_flag, off_pre, off_yall, rank, tp, m, mmax, piece_e0, xg,
            col0, ncol,
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(NTT, 1, 1), stream=stream)

    return launch
