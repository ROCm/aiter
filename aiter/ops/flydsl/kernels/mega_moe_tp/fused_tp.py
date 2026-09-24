# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Single-kernel tensor-parallel MegaMoE: AllGather + route scan + GEMM1 + act +
MXFP4 requant + GEMM2 + top-k reduce + ReduceScatter in one launch.

The GEMM1 -> GEMM2 intermediate never leaves LDS: a CTA owns one expert unit
(all of its routed rows, a range of its inter columns), runs GEMM1 one
128-column inter block per pass over K, quantizes the activation straight into
an LDS-resident MXFP4 operand, and GEMM2 reads it from there.

Block = 4 compute waves + 1 comm wave + 1 loader wave:

* compute waves stream their own weight tiles global -> LDS through a private
  ring filled by ``buffer_load ... lds`` (inline asm, so the compiler neither
  sees nor waits on it; ``s_waitcnt vmcnt(N)`` with N fixed by the issue
  pattern -- VMEM completes in order, so any extra VMEM op only makes a wait
  stricter);
* the loader wave stages the gathered A rows of every GEMM1 chunk into an LDS
  ring (per-buffer published / released counters, no barrier);
* the comm wave turns finished GEMM2 column chunks into the ReduceScatter while
  GEMM2 keeps running: once every CTA of the rank stored a chunk's route rows,
  CTAs sum their tokens' top-k rows and push the result into the owner rank's
  receive slot; the owner sums its slots into ``y``.

Cross-rank memory is an IPC-mapped symmetric arena; flags carry a monotonic
epoch so nothing is reset between launches (CUDA-graph safe). Polling uses
relaxed cache-bypassing accesses: on gfx950 an acquire/release fence
invalidates / writes back the whole XCD L2 under the compute waves.
"""

from __future__ import annotations

import functools
import math
import os as _os

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
    "FLAG_INTS",
    "MAX_TP",
    "NCK_MAX",
    "NCTA_MAX",
    "compile_fused_tp",
    "fused_tp_consts",
    "fused_tp_supported",
]

MAX_TP = 8
NCK_MAX = 64
# Flag arena (ints), all written by peers:
#   FLAG_RDY + c*MAX_TP + r   ReduceScatter: rank r pushed chunk c
#   FLAG_AGM + r*32           AllGather: rank r's routing (ids, weights) landed
#   FLAG_AGC + (q*MAX_TP + r)*NCTA_MAX + b
#                             AllGather: rank r's CTA b landed its rows' K-chunk q
# Flags hold the launch epoch (monotonic), so nothing is reset; each has one
# writer, so nothing contends.
FLAG_RDY = 0
FLAG_AGM = FLAG_RDY + MAX_TP * NCK_MAX
FLAG_AGC = FLAG_AGM + MAX_TP * 32
NCHA_MAX = 64
NCTA_MAX = 256
FLAG_INTS = FLAG_AGC + NCHA_MAX * MAX_TP * NCTA_MAX
CTRL_CNT = (
    64  # ctrl ints: [0] epoch [2] fin [32 + x] XCD x's L2 dropped; counters at 64
)
CTRL_XF = 32
N_XCD = 8  # workgroups are dealt round-robin over the XCDs
CTRL_LRDY = CTRL_CNT + 2 * NCK_MAX
LRDY_STRIDE = 32  # polled flags live on their own 128 B lines
# ReduceScatter push slices are taken from a per-chunk counter; launches
# alternate between two banks and a chunk's publisher zeroes the idle bank.
CTRL_GRAB = CTRL_LRDY + NCK_MAX * LRDY_STRIDE
CTRL_INTS = CTRL_GRAB + 2 * NCK_MAX * LRDY_STRIDE

NW = 4  # compute waves
NCOMM = 2  # comm wave + loader wave
NT = NW * 64
NTT = NT + 64 * NCOMM
NSK = 4  # weight ring depth (k-steps) per compute wave
SLOT = 4 * 1024 + 2 * 256  # 4 x 1 KB fragments + 2 scale dwords (256 B each)
OPS = 6  # DMA ops per k-step
KCS = 2  # k-steps per A chunk (256 K)
ACB = KCS * 64  # A bytes per row per chunk
NAB = 4  # A ring buffers
ALOAD_DEPTH = 3
RED_INFLIGHT = 32
POLL_SLEEP = 16
AUX_SC1 = 16
AUX_SYS = 1 | 16


def fused_tp_supported(model_dim: int, inter_dim: int, tp: int) -> bool:
    return (
        model_dim % 512 == 0
        and (model_dim // 64) % NW == 0
        and inter_dim % 128 == 0
        and 1 <= tp <= MAX_TP
    )


def _attr(v):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), int(v))


def _u(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


traced = ASTRewriter.transform


@functools.cache
def fused_tp_consts(H: int, I: int, TOPK: int, MT: int, TMAX: int) -> dict:
    """Compile-time geometry shared by the kernel and the host."""
    RG = MT * 16
    KS1 = H // 128
    NCH = KS1 // KCS
    KS2 = I // 128
    G2 = H // 64 // NW
    CH1 = (H // 32 + 7) // 8
    CH2 = (I // 32 + 7) // 8
    SI_STRIDE = I // 2 + 16
    NA_ROWOPS = RG * ACB // 1024
    NSC_BLK = (RG + 63) // 64
    NA_L = NA_ROWOPS + NSC_BLK * KCS
    GPC = 2 if (2 * KS2 * OPS <= 63 and G2 % 2 == 0) else 1
    NCK = G2 // GPC
    CW = GPC * NW * 64
    c = {
        "RG": RG,
        "KS1": KS1,
        "NCH": NCH,
        "KS2": KS2,
        "G2": G2,
        "CH1": CH1,
        "CH2": CH2,
        "SI_STRIDE": SI_STRIDE,
        "NA_ROWOPS": NA_ROWOPS,
        "NSC_BLK": NSC_BLK,
        "NA_L": NA_L,
        "GPC": GPC,
        "NCK": NCK,
        "CW": CW,
    }
    # LDS layout (bytes).
    off = 0

    def take(n, align=16):
        nonlocal off
        off = (off + align - 1) // align * align
        start = off
        off += n
        return start

    c["L_RING"] = take(NW * NSK * SLOT)
    c["L_A"] = take(NAB * RG * ACB)
    c["L_AS"] = take(NAB * KCS * NSC_BLK * 64 * 4)
    c["L_INTER"] = take(RG * SI_STRIDE)
    c["L_INTERS"] = take(RG * (I // 32))
    c["L_RIX"] = take(TMAX * 4)
    c["L_WT"] = take(TMAX * 4)
    # control ints
    c["L_CTL"] = take(64 * 4)
    c["LDS_BYTES"] = (off + 127) // 128 * 128
    assert H // 256 <= NCHA_MAX
    assert KS1 % NSK == 0 and NSK % KCS == 0
    assert (NSK - 1) * OPS <= 63 and NA_L * (ALOAD_DEPTH - 1) <= 63
    assert NCK <= NCK_MAX
    return c


# control-int indices inside L_CTL
C_CNT, C_BARCNT, C_BARGEN, C_LRED, C_PULL, C_EPOCH = 0, 1, 2, 3, 4, 5
C_USEQ, C_UROWS, C_NSIG, C_LQ, C_LPUB = 6, 7, 8, 9, 10
C_DONE = 12  # [NW]
C_ASEQ = 16  # [NAB]
C_AFREE = 20  # [NAB]
C_QBASE = 24  # [NW]
C_MBOX = 28  # [NW + NCOMM] per-wave mailbox
C_AGFREE = 41  # the AllGather staging (in the GEMM2 operand area) is read out


@functools.cache
def compile_fused_tp(
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
    rs_fp8: bool = False,
    agr: int = 1,
    tp: int = MAX_TP,
):
    """Build the launcher for one (shape, MT) instance."""
    c = fused_tp_consts(H, I, TOPK, MT, TMAX)
    RG, KS1, NCH, KS2, G2 = c["RG"], c["KS1"], c["NCH"], c["KS2"], c["G2"]
    CH1, CH2, SI_STRIDE = c["CH1"], c["CH2"], c["SI_STRIDE"]
    NA_ROWOPS, NSC_BLK, NA_L = c["NA_ROWOPS"], c["NSC_BLK"], c["NA_L"]
    GPC, NCK, CW = c["GPC"], c["NCK"], c["CW"]
    L_RING, L_A, L_AS = c["L_RING"], c["L_A"], c["L_AS"]
    L_INTER, L_INTERS, L_RIX, L_WT = c["L_INTER"], c["L_INTERS"], c["L_RIX"], c["L_WT"]
    L_CTL, LDS_BYTES = c["L_CTL"], c["LDS_BYTES"]
    WAIT_B1 = (NSK - 1) * OPS
    VPL = (CW // 8 + 63) // 64
    HALF = KS2 // 2 if (KS2 % 2 == 0 and KS2 > 2) else 1
    SCAN_IT = (TMAX * TOPK // 4 + NT - 1) // NT  # route-scan vector loads per thread
    SCAN_G = 8
    NPC = max(int(npieces), 1)  # pieces per split expert (1: no split experts)
    # Route rows: bf16, or (route_fp8) E4M3 with one E8M0 scale per 32 columns
    # -- the numerics of the split path's FP8 stage-2 route output -- which
    # halves the GEMM2 -> ReduceScatter traffic. fp8 layout per region: all
    # rows' H data bytes, then all rows' H/32 scale bytes.
    FP8R = bool(route_fp8)
    # ReduceScatter partial rows on the wire: bf16, or (rs_fp8) E4M3 with one
    # E8M0 per 32 columns -- half the xGMI bytes, where large batches are
    # bandwidth bound.
    RSF8 = bool(rs_fp8)
    # Tokens per push batch. With more than two pieces per split expert the
    # partial-row offsets the push keeps live push it past 256 VGPRs (spills),
    # so it takes half the batch.
    TB = max(1, (RED_INFLIGHT // 2 if NPC > 2 else RED_INFLIGHT) // TOPK)
    ROW_B = H + H // 32 if FP8R else 2 * H
    name = (
        "dev" + "".join(_os.environ.get(k, "0") for k in ("XP_NOCW", "XP_V1", "XP_V2", "XP_V3")) +
        f"mega_moe_tp_fused_h{H}_i{I}_k{TOPK}_mt{MT}_t{TMAX}_{act}"
        + (f"_p{NPC}" if NPC > 1 else "")
        + ("_r8" if route_fp8 else "")
        + ("_s8" if rs_fp8 else "")
        + f"_ag{agr}_tp{tp}"
    )

    const_expr = fx.const_expr

    # ------------------------------------------------------------------
    # low-level helpers (emit IR; no runtime control flow)
    # ------------------------------------------------------------------
    def i32(v):
        raw = v if isinstance(v, ir.Value) else getattr(v, "ir_value", lambda: None)()
        if raw is not None and isinstance(raw.type, ir.IndexType):
            return fx.Int32(fx.index_cast(T.i32, raw))
        return fx.Int32(v)

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

    def lds_ptr3(base, off):
        return _llvm.IntToPtrOp(
            _llvm.PointerType.get(address_space=3), _u(fx.Int32(base + i32(off)))
        ).result

    def lds_atomic_add(base, off, v, ordering=_llvm.AtomicOrdering.monotonic):
        return i32(
            _llvm.AtomicRMWOp(
                _llvm.AtomicBinOp.add,
                lds_ptr3(base, off),
                _u(i32(v)),
                ordering,
                syncscope="workgroup",
            ).res
        )

    def lds_ld_acq(base, off):
        return i32(
            _llvm.LoadOp(
                T.i32,
                lds_ptr3(base, off),
                alignment=4,
                volatile_=True,
                ordering=_llvm.AtomicOrdering.monotonic,
                syncscope="workgroup",
            ).res
        )

    def lds_st_rel(base, off, v):
        _llvm.StoreOp(
            _u(i32(v)),
            lds_ptr3(base, off),
            alignment=4,
            ordering=_llvm.AtomicOrdering.release,
            syncscope="workgroup",
        )

    def lds_cas(base, off, cmp, new):
        r = _llvm.AtomicCmpXchgOp(
            lds_ptr3(base, off),
            _u(i32(cmp)),
            _u(i32(new)),
            _llvm.AtomicOrdering.monotonic,
            _llvm.AtomicOrdering.monotonic,
            syncscope="workgroup",
        ).res
        return fx.Int32(_llvm.ExtractValueOp(T.i32, r, [0]).res)

    def gptr(addr):
        return _llvm.IntToPtrOp(
            _llvm.PointerType.get(address_space=1), _u(fx.Int64(addr))
        ).result

    def g_ld_rel(addr, scope):
        return i32(
            _llvm.LoadOp(
                T.i32,
                gptr(addr),
                alignment=4,
                volatile_=True,
                ordering=_llvm.AtomicOrdering.monotonic,
                syncscope=scope,
            ).res
        )

    def g_ld_sys(addr):
        return g_ld_rel(addr, "one-as")

    def g_st_sys(addr, v):
        _llvm.StoreOp(
            _u(i32(v)),
            gptr(addr),
            alignment=4,
            ordering=_llvm.AtomicOrdering.monotonic,
            syncscope="one-as",
        )

    def g_add_agent(addr, v):
        return i32(
            _llvm.AtomicRMWOp(
                _llvm.AtomicBinOp.add,
                gptr(addr),
                _u(i32(v)),
                _llvm.AtomicOrdering.monotonic,
                syncscope="agent",
            ).res
        )

    def g_ld_i32(addr):
        return i32(_llvm.LoadOp(T.i32, gptr(addr), alignment=4).res)

    def _now():
        return fx.Int64(_llvm.inline_asm(T.i64, [], "s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", has_side_effects=True))

    def _paddr(a, slot):
        return fx.Int64(a["ctrl"]) + fx.Int64(CTRL_INTS * 4 + 64) + fx.Int64(i32(gpu.block_id("x"))) * fx.Int64(4096) + fx.Int64(i32(slot)) * fx.Int64(8)

    def fence(ordering, scope):
        _llvm.FenceOp(ordering, syncscope=scope)

    def uni(v):
        return i32(rocdl.readfirstlane(T.i32, _u(i32(v))))

    def uni64(v):
        v = fx.Int64(v)
        lo = uni(fx.Int32(v & fx.Int64(0xFFFFFFFF)))
        hi = uni(fx.Int32(v >> fx.Int64(32)))
        return (fx.Int64(hi) << fx.Int64(32)) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))

    def rsrc(addr, nbytes=None):
        if nbytes is None:
            return buffer_ops.create_buffer_resource_from_addr(_u(uni64(addr)))
        return buffer_ops.create_buffer_resource_from_addr(
            _u(uni64(addr)), num_records_bytes=_u(fx.Int64(uni(nbytes)))
        )

    def asm(text, cons="", args=()):
        _llvm.inline_asm(None, [_u(a) for a in args], text, cons, has_side_effects=True)

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

    def wait_vm(n):
        # vmcnt is 6 bits; a smaller count only makes the wait stricter.
        asm(f"s_waitcnt vmcnt({min(int(n), 63)})")

    def wait_lgkm0():
        asm("s_waitcnt lgkmcnt(0)")

    def bld(rs, voff, soff, ty, aux):
        return rocdl.raw_ptr_buffer_load(
            _ty(ty), rs, _u(i32(voff)), _u(i32(soff)), aux=_attr(aux)
        )

    def bst(val, rs, voff, soff, aux):
        rocdl.raw_ptr_buffer_store(
            _u(val), rs, _u(i32(voff)), _u(i32(soff)), aux=_attr(aux)
        )

    class _LazyTy:
        # MLIR types need the trace-time context; build them on use.
        def __init__(self, fn):
            self.fn = fn

    V4I = _LazyTy(lambda: T.vec(4, T.i32))
    V4F = _LazyTy(lambda: T.vec(4, T.f32))
    V2I = _LazyTy(lambda: T.vec(2, T.i32))

    def _ty(t):
        return t.fn() if isinstance(t, _LazyTy) else t

    def route_region_bytes(ttot):
        return ttot * i32(TOPK * ROW_B)

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

    def route_load(rs, region, ttot, ridx, col, take=None):
        """Issue the loads of 8 route-row columns [col, col+8) of route row ridx
        in a region; `take` false makes them read past the end (zeros)."""
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
        if FP8R:
            dv = fx.Vector(ld[0])
            sc = ((fx.Int32(ld[1]) & i32(0xFF)) << i32(23)).bitcast(fx.Float32)
            return fp8x4_unpack(fx.Int32(dv[0]), sc) + fp8x4_unpack(fx.Int32(dv[1]), sc)
        return bf16x8_to_f32(ld)

    def swap16(x, y):
        """v_permlane16_swap: odd 16-lane rows of x trade with even rows of y."""
        st = _llvm.StructType.get_literal([T.i32, T.i32])
        r = rocdl.permlane16_swap(st, _u(i32(x)), _u(i32(y)), False, False)
        return i32(_llvm.ExtractValueOp(T.i32, r, [0]).res), i32(
            _llvm.ExtractValueOp(T.i32, r, [1]).res
        )

    def swap32(x, y):
        """v_permlane32_swap: the upper 32 lanes of x trade with the lower of y."""
        st = _llvm.StructType.get_literal([T.i32, T.i32])
        r = rocdl.permlane32_swap(st, _u(i32(x)), _u(i32(y)), False, False)
        return i32(_llvm.ExtractValueOp(T.i32, r, [0]).res), i32(
            _llvm.ExtractValueOp(T.i32, r, [1]).res
        )

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

    def act_batch(gs, us):
        return _activation_mul_batch(
            gs,
            us,
            act=act,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            swiglu_limit=7.0,
        )

    # ------------------------------------------------------------------
    # runtime-control-flow helpers (AST rewritten)
    # ------------------------------------------------------------------
    @traced
    def cbar(L, tid):
        """Barrier over the NW compute waves (s_barrier would wait for all waves)."""
        wait_lgkm0()
        if (tid % i32(64)) == i32(0):
            g = lds_ld_acq(L, L_CTL + C_BARGEN * 4)
            old = lds_atomic_add(
                L, L_CTL + C_BARCNT * 4, 1, _llvm.AtomicOrdering.acq_rel
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

    @traced
    def spin_sys_ge(addr, target):
        cur = g_ld_sys(addr)
        while cur < target:
            rocdl.s_sleep(2)
            cur = g_ld_sys(addr)

    @traced
    def gather_routes(L, tid, ids_addr, tw_addr, ttot, expert):
        """Collect every route to `expert` into LDS (rix, wt); return the count."""
        if tid == i32(0):
            lds_st(L, L_CTL + C_CNT * 4, i32(0))
        cbar(L, tid)
        n = ttot * i32(TOPK)
        n4 = n // i32(4)
        rid = rsrc(ids_addr)
        # All of this thread's loads go out before any compare: the scan runs
        # while other CTAs stream weights, and a load-compare chain would pay
        # the loaded-HBM latency once per iteration.
        rtw = rsrc(tw_addr)
        # (in groups of SCAN_G so the batch never pushes the kernel's registers)
        for g0 in range_constexpr(0, SCAN_IT, SCAN_G):
            its = list(range(g0, min(g0 + SCAN_G, SCAN_IT)))
            vs = []
            for it in its:
                q = fx.min(tid + i32(it * NT), n4 - i32(1))
                vs.append(fx.Vector(bld(rid, q * i32(16), 0, V4I, 0)))
            for x, it in enumerate(its):
                q = tid + i32(it * NT)
                for j in range_constexpr(4):
                    if (q < n4) & (fx.Int32(vs[x][j]) == expert):
                        slot = lds_atomic_add(L, L_CTL + C_CNT * 4, 1)
                        idx = q * i32(4) + i32(j)
                        if slot < i32(TMAX):
                            lds_st(L, L_RIX + slot * i32(4), idx)
                            lds_st(
                                L,
                                L_WT + slot * i32(4),
                                bld(rtw, idx * i32(4), 0, T.i32, 0),
                            )
        for idx_ in range(n4 * i32(4) + tid, n, i32(NT)):
            idx = i32(idx_)
            if g_ld_i32(fx.Int64(ids_addr) + fx.Int64(idx) * fx.Int64(4)) == expert:
                slot = lds_atomic_add(L, L_CTL + C_CNT * 4, 1)
                if slot < i32(TMAX):
                    lds_st(L, L_RIX + slot * i32(4), idx)
                    lds_st(
                        L,
                        L_WT + slot * i32(4),
                        g_ld_i32(fx.Int64(tw_addr) + fx.Int64(idx) * fx.Int64(4)),
                    )
        cbar(L, tid)
        cnt = lds_ld_i32(L, L_CTL + C_CNT * 4)
        cbar(L, tid)
        return fx.min(cnt, i32(TMAX))

    # ---------------------------- GEMM1 -------------------------------
    def act_quant_store(L, lane, w, accg, accu, nb):
        """Lane holds row m*16 + lane%16, inter columns ... + t*16 + 4*(lane/16) + v of
        gate tile t and the same columns of up tile t; the 32-column quant group is
        both tiles: 8 values in-lane, then the 4 lanes sharing lane%16."""
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
            am = _fabs_f32(xs[0])
            for j in range_constexpr(1, 8):
                am = am.maximumf(_fabs_f32(xs[j]))
            am = am.maximumf(am.shuffle_xor(i32(16), i32(64)))
            am = am.maximumf(am.shuffle_xor(i32(32), i32(64)))
            e8, qs = _e8m0_from_amax(am, max_norm=6.0)
            for t in range_constexpr(2):
                pk = rocdl.cvt_scalef32_pk_fp4_f32(
                    T.i32, _u(i32(0)), _u(xs[t * 4 + 0]), _u(xs[t * 4 + 1]), _u(qs), 0
                )
                pk = rocdl.cvt_scalef32_pk_fp4_f32(
                    T.i32, pk, _u(xs[t * 4 + 2]), _u(xs[t * 4 + 3]), _u(qs), 1
                )
                cb = (nb * i32(128) + w * i32(32) + i32(t * 16) + q4 * i32(4)) // i32(2)
                lds_st(
                    L,
                    L_INTER + row * i32(SI_STRIDE) + cb,
                    fx.Int16(fx.Int32(pk) & i32(0xFFFF)),
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
                L, L_CTL + C_AFREE * 4 + buf * i32(4), 1, _llvm.AtomicOrdering.release
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
                    wait_vm(WAIT_B1)
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
            ag_stage_free(L, lane)
            act_quant_store(L, lane, w, res[: 2 * MT], res[2 * MT :], pas)
        _store_qbase(L, lane, w, qbase + nnb * i32(NCH))
        wait_vm(0)
        cbar(L, tid)

    @traced
    def _store_qbase(L, lane, w, v):
        if lane == i32(0):
            lds_st(L, L_CTL + C_QBASE * 4 + w * i32(4), v)

    # ---------------------------- loader -------------------------------
    @traced
    def a_loader(L, tid, a):
        lane = tid % i32(64)
        rx = rsrc(a["ax"])
        rxs = rsrc(a["axs"])
        ub = g_ld_i32(
            fx.Int64(a["cta_units"]) + fx.Int64(gpu.block_id("x")) * fx.Int64(4)
        )
        ue = g_ld_i32(
            fx.Int64(a["cta_units"]) + fx.Int64(gpu.block_id("x") + 1) * fx.Int64(4)
        )
        for u_ in range(ub, ue, i32(1)):
            u = i32(u_)
            if lane == i32(0):
                spin_lds_ge(L, L_CTL + C_USEQ * 4, u - ub + i32(1))
            R = uni(lds_ld_acq(L, L_CTL + C_UROWS * 4))
            nnb = g_ld_i32(
                fx.Int64(a["units"]) + fx.Int64(u) * fx.Int64(16) + fx.Int64(8)
            ) // i32(128)
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
                    if const_expr(_os.environ.get("XP_NOCW", "0") != "1"):
                        if cidx < i32(NCH):
                            ag_wait_chunk(L, lane, a, a["epoch"], cc, r0, rows)
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
                            # A-scale lines hold 16 K-chunks of a row, some of
                            # them possibly still on the wire when an earlier
                            # chunk is read: never let a cached copy serve them
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

    # ---------------------------- GEMM2 -------------------------------
    def nsk2_for(nks):
        if 4 % nks == 0 and G2 % (4 // nks) == 0:
            return 4
        if nks % 3 == 0:
            return 3
        return 2

    @traced
    def report_chunk(L, lane, w, cidx):
        if lane == i32(0):
            lds_st_rel(L, L_CTL + C_DONE * 4 + w * i32(4), cidx)

    @traced
    def maybe_report(L, lane, w, gi, signal, rlag_wait):
        if signal & (((gi + i32(1)) % i32(GPC)) == i32(0)) & (gi + i32(1) > i32(GPC)):
            wait_vm(rlag_wait)
            report_chunk(L, lane, w, (gi // i32(GPC)) - i32(1))

    # Dead rows are not branched around: their offset points past the
    # resource's num_records and the hardware drops the write.
    def store_route(rs, voff, o, ok, oob):
        bst(o, rs, ok.select(voff, oob), 0, AUX_SC1)

    def store_route_fp8(rs, halves, rix, n0, q4, lane, ok, oob, a):
        """One row's 64 columns as E4M3 + two E8M0 scales. Each half's 32-column
        MX block lives in the row's four lanes (q4); after permlane32_swap lane q4
        holds 16 contiguous columns of half q4 >> 1 at 16 * (q4 & 1)."""
        packs, e8s = [], []
        for half in range_constexpr(2):
            f = bf16x8_to_f32(fx.Vector.from_elements(halves[half], fx.Int32))
            am = _fabs_f32(f[0])
            for j in range_constexpr(1, 8):
                am = am.maximumf(_fabs_f32(f[j]))
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
        ov = fx.Vector.from_elements(lo + hi, fx.Int32)
        col = n0 + (q4 >> i32(1)) * i32(32) + (q4 & i32(1)) * i32(16)
        bst(ov, rs, ok.select(rix * i32(H) + col, oob), 0, AUX_SC1)
        # the row's two scale bytes, from its q4 == 0 lane
        sc = fx.Int16(e8s[0] | (e8s[1] << i32(8)))
        soff = a["ttot"] * i32(TOPK * H) + rix * i32(H // 32) + n0 // i32(32)
        bst(sc, rs, (ok & (q4 == i32(0))).select(soff, oob), 0, AUX_SC1)

    @traced
    def gemm2(L, tid, a, expert, ks0, r0, rows, signal, NKS, pidx):
        NSK2 = nsk2_for(NKS)
        GPI = (NSK2 * NKS // math.gcd(NSK2, NKS)) // NKS
        assert G2 % GPI == 0
        WAIT_B2 = (NSK2 - 1) * OPS
        # Only the ring's own loads may be counted: on gfx950 VMEM stores and
        # loads complete out of order, so the epilogue writes cannot widen a wait.
        RLAG = GPC * NKS * OPS
        lane = tid % i32(64)
        w = uni(tid // i32(64))
        e = uni(expert)
        ks0 = uni(ks0)
        rw = rsrc(fx.Int64(a["w2"]) + fx.Int64(e) * fx.Int64(H * (I // 2)))
        rws = rsrc(a["w2s"])
        # Piece p of a split expert writes its K-slice partial of each route
        # row: p = 0 into the route's own row, p > 0 into proutes[p - 1]; the
        # push adds them up. Plain stores, no atomics, no zeroing.
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

        def issue(q, slot_idx):
            qc = fx.min(q, i32(G2 * NKS - 1))
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
            issue(i32(qq), qq)
        zero = fx.Vector.filled(4, 0.0, fx.Float32)
        for gi0_ in range(i32(0), i32(G2), i32(GPI)):
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
                    kh = (ks0 + i32(k)) & i32(1)
                    sh0 = kh * i32(16)
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
                                i32(L_INTER)
                                + row * i32(SI_STRIDE)
                                + i32(k * 64)
                                + q4 * i32(16),
                                V4I,
                                16,
                            )
                        )
                        sa_l.append(
                            fx.Int32(
                                lds_ld(
                                    L,
                                    i32(L_INTERS)
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
                # epilogue: lane (row lane%16, q4) holds cols n0 + t*16 + 4*q4 + v.
                # permlane16_swap pairs q4 with q4^1: for tiles (ta, tb) the even lane
                # keeps its ta quad and takes the odd lane's, the odd lane keeps its tb
                # quad and takes the even lane's, so each lane stores 8 contiguous
                # columns of tile ta + (q4 & 1) at 8 * (q4 >> 1).
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
                            store_route(
                                r_routes, (rix * i32(H) + col) * i32(2), ov, ok, oob
                            )
                    if const_expr(FP8R):
                        store_route_fp8(r_routes, halves, rix, n0, q4, lane, ok, oob, a)
                maybe_report(L, lane, w, gi, signal, RLAG)
        wait_vm(0)
        _final_report(L, lane, w, signal)

    @traced
    def _final_report(L, lane, w, signal):
        if signal:
            report_chunk(L, lane, w, i32(NCK - 1))

    @traced
    def gemm2_dispatch(L, tid, a, expert, ks0, icnt, r0, rows, signal):
        pidx = ks0 * i32(128) // icnt
        if icnt == i32(I):
            gemm2(L, tid, a, expert, ks0, r0, rows, signal, KS2, i32(0))
        else:
            if const_expr(HALF > 1):
                if icnt == i32(HALF * 128):
                    gemm2(L, tid, a, expert, ks0, r0, rows, signal, HALF, pidx)
                else:
                    gemm2(L, tid, a, expert, ks0, r0, rows, signal, 1, pidx)
            else:
                gemm2(L, tid, a, expert, ks0, r0, rows, signal, 1, pidx)

    @traced
    def _drop_stale(tid, a):
        if tid < i32(64):
            asm("buffer_inv sc0")
        bid = i32(gpu.block_id("x"))
        if (tid < i32(64)) & (bid < i32(N_XCD)):
            fence(_llvm.AtomicOrdering.acquire, "one-as")
            if tid == i32(0):
                g_st_sys(
                    fx.Int64(a["ctrl"]) + fx.Int64((CTRL_XF + bid) * 4), a["epoch"]
                )

    @traced
    def _stale_dropped(tid, a):
        """This XCD's L2 no longer holds the previous launch's arena lines."""
        if tid == i32(0):
            x = i32(gpu.block_id("x")) % i32(N_XCD)
            spin_sys_ge(
                fx.Int64(a["ctrl"]) + fx.Int64((i32(CTRL_XF) + x) * i32(4)), a["epoch"]
            )

    @traced
    def _pacc(a, tid, slot, dt):
        if tid == i32(0):
            _llvm.AtomicRMWOp(_llvm.AtomicBinOp.add, gptr(_paddr(a, slot)), _u(dt), _llvm.AtomicOrdering.monotonic, syncscope="agent")

    @traced
    def _pacc_lane(a, lane, slot, dt, on):
        if (lane == i32(0)) & on:
            _llvm.AtomicRMWOp(_llvm.AtomicBinOp.add, gptr(_paddr(a, slot)), _u(dt), _llvm.AtomicOrdering.monotonic, syncscope="agent")

    @traced
    def _pst_lane(a, lane, slot, t):
        if lane == i32(0):
            _llvm.StoreOp(_u(t), gptr(_paddr(a, slot)), alignment=8)

    def pchunk(a, lane, cidx, k):
        _pst_lane(a, lane, i32(16 + k) + cidx * i32(6), _now())

    @traced
    def compute_units(L, tid, a):
        lane = tid % i32(64)
        ag_wait_meta(a, tid, a["epoch"])
        _stale_dropped(tid, a)
        cbar(L, tid)
        bid = gpu.block_id("x")
        ub = g_ld_i32(fx.Int64(a["cta_units"]) + fx.Int64(bid) * fx.Int64(4))
        ue = g_ld_i32(fx.Int64(a["cta_units"]) + fx.Int64(bid + 1) * fx.Int64(4))
        if ub == ue:
            report_chunk(L, lane, tid // i32(64), i32(NCK - 1))
        for u_ in range(ub, ue, i32(1)):
            u = i32(u_)
            ubase = fx.Int64(a["units"]) + fx.Int64(u) * fx.Int64(16)
            expert = g_ld_i32(ubase)
            i0 = g_ld_i32(ubase + fx.Int64(4))
            icnt = g_ld_i32(ubase + fx.Int64(8))
            tg = _now()
            R = gather_routes(L, tid, a["ids"], a["tw"], a["ttot"], expert)
            _pacc(a, tid, 9, _now() - tg)
            if tid == i32(0):
                lds_st(L, L_CTL + C_UROWS * 4, R)
                lds_st_rel(L, L_CTL + C_USEQ * 4, u - ub + i32(1))
            if (u == ue - i32(1)) & (R == i32(0)):
                report_chunk(L, lane, tid // i32(64), i32(NCK - 1))
            for r0_ in range(i32(0), R, i32(RG)):
                r0 = i32(r0_)
                rows = fx.min(R - r0, i32(RG))
                gemm1(L, tid, a, expert, i0, icnt // i32(128))
                sig = (u == ue - i32(1)) & (r0 + i32(RG) >= R)
                gemm2_dispatch(L, tid, a, expert, i0 // i32(128), icnt, r0, rows, sig)
                cbar(L, tid)

    # ---------------------------- comm ---------------------------------
    def peer_sel(a, p):
        """a['peer'][p] for a runtime rank index p (wave-uniform)."""
        v = fx.Int64(a["peer"][0])
        for j in range_constexpr(1, MAX_TP):
            v = (p == i32(j)).select(fx.Int64(a["peer"][j]), v)
        return v

    def rdy_flag_addr(a, rank_p, dst_rank_base, cidx):
        return (
            dst_rank_base
            + fx.Int64(a["off_flag"])
            + fx.Int64((i32(FLAG_RDY) + cidx * i32(MAX_TP) + rank_p) * i32(4))
        )

    def grab_slot(a, epoch, cidx, bank_off):
        bank = (epoch + i32(bank_off)) & i32(1)
        return fx.Int64(a["ctrl"]) + fx.Int64(
            (i32(CTRL_GRAB) + (bank * i32(NCK_MAX) + cidx) * i32(LRDY_STRIDE)) * i32(4)
        )

    @traced
    def grab(lane, addr):
        # every lane issues the atomic (only lane 0 adds): a lane-0 branch
        # here miscompiles inside push_chunk's dynamic while loop
        return uni(g_add_agent(addr, (lane == i32(0)).select(i32(1), i32(0))))

    @traced
    def push_chunk(L, tid, a, epoch, cidx):
        """Push units of the chunk's token rows (unit u: tokens u, u + ns, ...)
        until none is left: CTAs whose waves are free take more, so slow CTAs
        do not hold the chunk."""
        lane = tid % i32(64)
        ttot = a["ttot"]
        nblk = gpu.grid_dim.x
        ns = fx.min(i32(2) * i32(nblk), (ttot + i32(TB - 1)) // i32(TB))
        gslot = grab_slot(a, epoch, cidx, 0)
        c0 = cidx * i32(CW)
        routes_bytes = route_region_bytes(ttot)
        r_routes = rsrc(a["routes"], routes_bytes)
        # partial rows of pieces 1.. of split experts; others read past the end (0)
        pr_bytes = i32(NPC - 1) * routes_bytes
        r_pr = rsrc(a["proutes"], pr_bytes)
        ids_base = peer_sel(a, a["rank"]) + fx.Int64(a["off_ids"])
        on = cidx < i32(4)
        tq = _now()
        u = grab(lane, gslot)
        tq2 = _now()
        _pacc_lane(a, lane, 100, tq2 - tq, on)
        n = i32(0)
        while u < ns:
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
                            kv.append(
                                route_load(
                                    r_routes,
                                    i32(0),
                                    ttot,
                                    t * i32(TOPK) + i32(k),
                                    c0 + v * i32(8),
                                )
                            )
                        row.append(kv)
                    d.append(row)
                for b in range_constexpr(TB):
                    t_raw = t0 + i32(b) * ns
                    t = fx.min(t_raw, ttot - i32(1))
                    owner = t // a["m"]
                    orow = t - owner * a["m"]
                    r_dst = rsrc(peer_sel(a, owner) + fx.Int64(a["off_part"]))
                    pd = (a["rank"] * a["mmax"] + orow, c0)
                    for j in range_constexpr(VPL):
                        v = lane + i32(j * 64)
                        acc = [fx.Float32(0.0)] * 8
                        for k in range_constexpr(TOPK):
                            vals = route_decode(d[b][k][j])
                            acc = [x + y for x, y in zip(acc, vals)]
                        live = (t_raw < ttot) & (v < i32(CW // 8))
                        if const_expr(NPC > 1):
                            # routes to a split expert also carry pieces 1.. partials
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
                                pd,
                                r_pr,
                                v,
                                acc,
                                live,
                                hit,
                                eids,
                                t,
                                c0,
                                routes_bytes,
                                pr_bytes,
                            )
                        else:
                            _push_store(a, r_dst, pd, v, acc, live)
            n = n + i32(1)
            tq3 = _now()
            _pacc_lane(a, lane, 101, tq3 - tq2, on)
            u = grab(lane, gslot)
            tq2 = _now()
            _pacc_lane(a, lane, 100, tq2 - tq3, on)
        tq4 = _now()
        wait_vm(0)
        tq5 = _now()
        _pacc_lane(a, lane, 102, tq5 - tq4, on)
        _push_done(a, lane, epoch, cidx, ns, n)
        _pacc_lane(a, lane, 103, _now() - tq5, on)
        _pacc_lane(a, lane, 104, fx.Int64(n), on)
        _pacc_lane(a, lane, 105, fx.Int64(1), on)

    @traced
    def _push_split_token(
        a, r_dst, pd, r_pr, v, acc, live, hit, eids, t, c0, rbytes, prbytes
    ):
        """Push one token's row, adding split pieces' partials only when the
        token routes to a split expert (uniform per token, and rare)."""
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
                    vals = route_decode(ld)
                    tot = [x + y for x, y in zip(tot, vals)]
            _push_store(a, r_dst, pd, v, tot, live)
        else:
            _push_store(a, r_dst, pd, v, acc, live)

    def part_offs(a, prow, col):
        """Byte offsets of columns [col, col+8) of partial row prow in the
        receive area: E4M3 rows, then one E8M0 per 32 columns (RSF8), or bf16."""
        if RSF8:
            soff = a["tp"] * a["mmax"] * i32(H) + prow * i32(H // 32) + col // i32(32)
            return prow * i32(H) + col, soff
        return (prow * i32(H) + col) * i32(2), None

    @traced
    def _push_store(a, r_dst, pd, v, acc, ok):
        doff, soff = part_offs(a, pd[0], pd[1] + v * i32(8))
        if const_expr(RSF8):
            # a 32-column MX block spans 4 adjacent lanes
            am = _fabs_f32(acc[0])
            for j in range_constexpr(1, 8):
                am = am.maximumf(_fabs_f32(acc[j]))
            am = am.maximumf(am.shuffle_xor(i32(1), i32(64)))
            am = am.maximumf(am.shuffle_xor(i32(2), i32(64)))
            e8, qs = _e8m0_from_amax(am, max_norm=448.0)
            o = fx.Vector.from_elements(
                [fp8x4_pack(acc[0:4], qs), fp8x4_pack(acc[4:8], qs)], fx.Int32
            )
            # four blocks' scales in one dword, from the first lane of 16
            e = (fx.Int32(e8) & i32(0xFF)).bitcast(fx.Float32)
            sc = fx.Int32(e8) & i32(0xFF)
            for k in range_constexpr(1, 4):
                sc = sc | (
                    e.shuffle_xor(i32(4 * k), i32(64)).bitcast(fx.Int32) << i32(8 * k)
                )
            if ok:
                bst(o, r_dst, doff, 0, AUX_SYS)
                if (v & i32(15)) == i32(0):
                    bst(sc, r_dst, soff, 0, AUX_SYS)
        else:
            if ok:
                bst(pack_bf16x8(acc), r_dst, doff, 0, AUX_SYS)

    @traced
    def _push_done(a, lane, epoch, cidx, ns, n):
        if (lane == i32(0)) & (n > i32(0)):
            cnt_addr = fx.Int64(a["ctrl"]) + fx.Int64(
                (i32(CTRL_CNT + NCK_MAX) + cidx) * i32(4)
            )
            old = g_add_agent(cnt_addr, n)
            if old + n == ns:
                g_st_sys(cnt_addr, i32(0))
                g_st_sys(grab_slot(a, epoch, cidx, 1), i32(0))
                for p in range_constexpr(MAX_TP):
                    if i32(p) < a["tp"]:
                        g_st_sys(
                            rdy_flag_addr(a, a["rank"], fx.Int64(a["peer"][p]), cidx),
                            epoch,
                        )

    @traced
    def final_chunk(tid, a, cidx):
        lane = tid % i32(64)
        nblk = gpu.grid_dim.x
        c0 = cidx * i32(CW)
        r_recv = rsrc(peer_sel(a, a["rank"]) + fx.Int64(a["off_part"]))
        for row_ in range(i32(gpu.block_id("x")), a["m"], i32(nblk)):
            row = i32(row_)
            for j in range_constexpr(VPL):
                v = lane + i32(j * 64)
                vc = fx.min(v, i32(CW // 8 - 1))
                acc = [fx.Float32(0.0)] * 8
                for p in range_constexpr(MAX_TP):
                    pc = fx.min(i32(p), a["tp"] - i32(1))
                    doff, soff = part_offs(a, pc * a["mmax"] + row, c0 + vc * i32(8))
                    if const_expr(RSF8):
                        vals = route_decode(
                            (
                                bld(r_recv, doff, 0, V2I, AUX_SYS),
                                bld(r_recv, soff, 0, T.i8, AUX_SYS),
                            )
                        )
                    else:
                        vals = bf16x8_to_f32(bld(r_recv, doff, 0, V4I, AUX_SYS))
                    live = i32(p) < a["tp"]
                    acc = [
                        x + live.select(y, fx.Float32(0.0)) for x, y in zip(acc, vals)
                    ]
                _y_store(a, row, c0, v, pack_bf16x8(acc))

    @traced
    def _y_store(a, row, c0, v, o):
        if v < i32(CW // 8):
            bst(o, rsrc(a["y"]), (row * i32(H) + c0 + v * i32(8)) * i32(2), 0, 0)

    @traced
    def claim(L, lane, w, ctr_idx, stage, a, epoch):
        """Claim the stage's next chunk iff it is ready (CAS on the checked value).
        Result through this wave's LDS mailbox: chunk index or -1."""
        if lane == i32(0):
            cc = lds_ld_acq(L, L_CTL + ctr_idx * 4)
            ccl = fx.min(cc, i32(NCK - 1))
            rdy = i32(1)
            if const_expr(stage == 0):
                v = g_ld_sys(
                    fx.Int64(a["ctrl"])
                    + fx.Int64((i32(CTRL_LRDY) + ccl * i32(LRDY_STRIDE)) * i32(4))
                )
                rdy = (v >= epoch).select(i32(1), i32(0))
            else:
                # the chunk's MAX_TP push flags share one 32 B line: two loads
                rf = rsrc(peer_sel(a, a["rank"]) + fx.Int64(a["off_flag"]))
                fo = (i32(FLAG_RDY) + ccl * i32(MAX_TP)) * i32(4)
                fl = fx.Vector(bld(rf, fo, 0, V4I, AUX_SYS))
                fh = fx.Vector(bld(rf, fo + i32(16), 0, V4I, AUX_SYS))
                for p in range_constexpr(MAX_TP):
                    f = fx.Int32((fl if p < 4 else fh)[p % 4])
                    ok_p = (i32(p) >= a["tp"]) | (f >= epoch)
                    rdy = ok_p.select(rdy, i32(0))
            want = (cc < i32(NCK)) & (rdy == i32(1))
            newv = want.select(cc + i32(1), cc)
            got = lds_cas(L, L_CTL + ctr_idx * 4, cc, newv)
            res = (want & (got == cc)).select(cc, i32(-1))
            lds_st(L, L_CTL + (C_MBOX * 4) + w * i32(4), res)
        rocdl.sched_barrier(0)
        return uni(lds_ld_acq(L, L_CTL + (C_MBOX * 4) + w * i32(4)))

    @traced
    def comm_work(L, tid, a, epoch):
        lane = tid % i32(64)
        w = tid // i32(64)
        cr = claim(L, lane, w, C_LRED, 0, a, epoch)
        if cr >= i32(0):
            pchunk(a, lane, cr, 2)
            push_chunk(L, tid, a, epoch, cr)
            pchunk(a, lane, cr, 3)
        cp = claim(L, lane, w, C_PULL, 1, a, epoch)
        if cp >= i32(0):
            pchunk(a, lane, cp, 4)
            final_chunk(tid, a, cp)
            pchunk(a, lane, cp, 5)
        return ((cr >= i32(0)) | (cp >= i32(0))).select(i32(1), i32(0))

    @traced
    def comm_signal(L, tid, a, epoch):
        """Count this CTA into every chunk its compute waves finished since the
        last call; the rank's last CTA publishes the chunk ready. Any wave may
        call it: the range is taken with a CAS on C_NSIG."""
        lane = tid % i32(64)
        nblk = gpu.grid_dim.x
        mn = i32(NCK)
        for w in range_constexpr(NW):
            mn = fx.min(mn, lds_ld_acq(L, L_CTL + C_DONE * 4 + w * 4))
        start = lds_ld_acq(L, L_CTL + C_NSIG * 4)
        end = fx.min(mn + i32(1), i32(NCK))
        if (lane == i32(0)) & (end > start):
            got = lds_cas(L, L_CTL + C_NSIG * 4, start, end)
            if got == start:
                for cidx_ in range(start, end, i32(1)):
                    cidx = i32(cidx_)
                    _signal_one(a, lane, epoch, nblk, cidx)
        rocdl.sched_barrier(0)
        return (end > start).select(i32(1), i32(0))

    @traced
    def _signal_one(a, lane, epoch, nblk, cidx):
        cnt_addr = fx.Int64(a["ctrl"]) + fx.Int64((i32(CTRL_CNT) + cidx) * i32(4))
        pchunk(a, lane, cidx, 0)
        old = g_add_agent(cnt_addr, 1)
        if old == i32(nblk) - i32(1):
            g_st_sys(cnt_addr, i32(0))
            g_st_sys(
                fx.Int64(a["ctrl"])
                + fx.Int64((i32(CTRL_LRDY) + cidx * i32(LRDY_STRIDE)) * i32(4)),
                epoch,
            )

    @traced
    def signal_loop(L, tid, a, epoch):
        """The loader wave, once its A chunks are staged, only watches for
        finished chunks, so a chunk is signalled without waiting behind a push."""
        while lds_ld_acq(L, L_CTL + C_NSIG * 4) < i32(NCK):
            if comm_signal(L, tid, a, epoch) == i32(0):
                rocdl.s_sleep(1)

    @traced
    def comm_wave(L, tid, a, epoch):
        while (lds_ld_acq(L, L_CTL + C_NSIG * 4) < i32(NCK)) | (
            lds_ld_acq(L, L_CTL + C_PULL * 4) < i32(NCK)
        ):
            p1 = comm_signal(L, tid, a, epoch)
            p2 = comm_work(L, tid, a, epoch)
            if (p1 | p2) == i32(0):
                rocdl.s_sleep(POLL_SLEEP)

    @traced
    def comm_help(L, tid, a, epoch):
        while lds_ld_acq(L, L_CTL + C_PULL * 4) < i32(NCK):
            p = comm_work(L, tid, a, epoch)
            if p == i32(0):
                rocdl.s_sleep(POLL_SLEEP)

    # ---------------------------- AllGather ------------------------------
    # AllGather. Every CTA owns token rows bid, bid + nblk, ... (at most AGR).
    # All threads quantize them into the GEMM2 operand area (idle until the
    # first GEMM1 pass ends); the comm wave then sends ids/weights and the
    # MXFP4 payload in K-chunk order, so peers start GEMM1 on chunk 0 while
    # the rest is on the wire, and bumps the receivers' arrival counters.
    AGR = int(agr)
    TPC = int(tp)  # peers the AllGather sends to (the engine's world size)
    # chunks in flight per comm wave before the oldest one is counted
    AG_D = max(1, min(8, 63 // (2 * TPC)))
    AG_SROW = H // 2
    AG_SCB = AGR * AG_SROW
    assert AG_SCB + AGR * (H // 32) <= L_INTERS - L_INTER + RG * (I // 32)

    def ag_rows(a):
        bid = i32(gpu.block_id("x"))
        nblk = i32(gpu.grid_dim.x)
        nrows = fx.min(fx.max((a["m"] - bid + nblk - i32(1)) // nblk, i32(0)), i32(AGR))
        return bid, nblk, nrows

    @traced
    def ag_quant(L, tid, a):
        bid, nblk, nrows = ag_rows(a)
        rxl = rsrc(a["x"])
        for q_ in range(tid, nrows * i32(H // 32), i32(NTT)):
            q = i32(q_)
            r = q // i32(H // 32)
            g = q - r * i32(H // 32)
            i = bid + r * nblk
            f = []
            for v in range_constexpr(4):
                raw = fx.Vector(
                    bld(
                        rxl,
                        ((i * i32(H)) + g * i32(32) + i32(v * 8)) * i32(2),
                        0,
                        V4I,
                        0,
                    )
                )
                for j in range_constexpr(4):
                    wv = fx.Int32(raw[j])
                    f.append((wv << i32(16)).bitcast(fx.Float32))
                    f.append((wv & i32(-65536)).bitcast(fx.Float32))
            am = _fabs_f32(f[0])
            for j in range_constexpr(1, 32):
                am = am.maximumf(_fabs_f32(f[j]))
            e8, qs = _e8m0_from_amax(am, max_norm=6.0)
            words = []
            for dw in range_constexpr(4):
                pk = rocdl.cvt_scalef32_pk_fp4_f32(
                    T.i32, _u(i32(0)), _u(f[dw * 8 + 0]), _u(f[dw * 8 + 1]), _u(qs), 0
                )
                pk = rocdl.cvt_scalef32_pk_fp4_f32(
                    T.i32, pk, _u(f[dw * 8 + 2]), _u(f[dw * 8 + 3]), _u(qs), 1
                )
                pk = rocdl.cvt_scalef32_pk_fp4_f32(
                    T.i32, pk, _u(f[dw * 8 + 4]), _u(f[dw * 8 + 5]), _u(qs), 2
                )
                pk = rocdl.cvt_scalef32_pk_fp4_f32(
                    T.i32, pk, _u(f[dw * 8 + 6]), _u(f[dw * 8 + 7]), _u(qs), 3
                )
                words.append(fx.Int32(pk))
            lds_st(
                L,
                L_INTER + r * i32(AG_SROW) + g * i32(16),
                fx.Vector.from_elements(words, fx.Int32),
                16,
            )
            lds_st(
                L, L_INTER + i32(AG_SCB) + r * i32(H // 32) + g, fx.Int8(e8), align=1
            )

    @traced
    def ag_send(L, lane, a):
        """Comm wave: ids/weights, then the payload chunk by chunk. Every chunk
        issues exactly 2 * TPC stores (dead rows read past num_records), so
        wait_vm(2 * TPC * AG_D) after chunk q means chunk q - AG_D landed and
        its counters may be bumped; AG_D chunks stay in flight."""
        bid, nblk, nrows = ag_rows(a)
        m = a["m"]
        rank = a["rank"]
        # let the (small) routing metadata out first: the payload would queue
        # it behind a megabyte per peer
        mp = _meta_pending(a, lane, a["epoch"], own=True)
        while mp != i32(0):
            rocdl.s_sleep(1)
            mp = _meta_pending(a, lane, a["epoch"], own=True)
        NCHA = H // 256
        r = lane // i32(8)
        j = lane - r * i32(8)
        rok = r < nrows
        i = bid + fx.min(r, fx.max(nrows - i32(1), i32(0))) * nblk
        gslot = rank * m + i
        x_bytes = a["ttot"] * i32(H // 2)
        xs_bytes = a["ttot"] * i32(H // 32)
        rx = [
            rsrc(fx.Int64(a["peer"][p]) + fx.Int64(a["off_x"]), x_bytes)
            for p in range(TPC)
        ]
        rs = [
            rsrc(fx.Int64(a["peer"][p]) + fx.Int64(a["off_xs"]), xs_bytes)
            for p in range(TPC)
        ]
        for q in range_constexpr(NCHA):
            rr = fx.min(r, i32(AGR - 1))
            dv = lds_ld(
                L, L_INTER + rr * i32(AG_SROW) + i32(q * 128) + j * i32(16), V4I, 16
            )
            sv = lds_ld(
                L, L_INTER + i32(AG_SCB) + rr * i32(H // 32) + i32(q * 8), V2I, 8
            )
            for p in range_constexpr(TPC):
                live = rok
                bst(
                    dv,
                    rx[p],
                    live.select(
                        gslot * i32(H // 2) + i32(q * 128) + j * i32(16), x_bytes
                    ),
                    0,
                    AUX_SYS,
                )
                bst(
                    sv,
                    rs[p],
                    (live & (j == i32(0))).select(
                        gslot * i32(H // 32) + i32(q * 8), xs_bytes
                    ),
                    0,
                    AUX_SYS,
                )
            if const_expr(q == NCHA - 1):
                # every staged byte is in registers: GEMM2 may reuse its area
                wait_lgkm0()
                _ag_free(L, lane)
            if const_expr(q >= AG_D):
                wait_vm(2 * TPC * AG_D)
                _ag_bump(a, lane, rank, i32(q - AG_D))
            # keep the scheduler from hoisting every chunk's LDS reads (and
            # their registers) to the top of the fully unrolled loop
            rocdl.sched_barrier(0)
        wait_vm(0)
        for q in range_constexpr(max(0, NCHA - AG_D), NCHA):
            _ag_bump(a, lane, rank, i32(q))

    @traced
    def _ag_free(L, lane):
        if lane == i32(0):
            lds_st_rel(L, L_CTL + C_AGFREE * 4, i32(1))

    NMETA_MAX = 32  # CTAs that send routing metadata, 1 KB each (FLAG_AGM stride)

    def _ag_nmeta(a):
        return fx.max((a["m"] * i32(TOPK) // i32(4) + i32(63)) // i32(64), i32(1))

    @traced
    def _ag_send_meta(lane, a):
        """The first nmeta CTAs each copy 64 x 16 B of this rank's routing (ids,
        weights) to every peer -- one load and one store round, no serial chain
        -- then raise their own flag per peer."""
        bid = i32(gpu.block_id("x"))
        if bid < _ag_nmeta(a):
            m = a["m"]
            rank = a["rank"]
            n = m * i32(TOPK)
            n4 = n // i32(4)
            rid = rsrc(a["ids_in"])
            rtw = rsrc(a["tw_in"])
            dst0 = rank * n * i32(4)
            v = bid * i32(64) + lane
            vok = v < n4
            idv = bld(rid, v * i32(16), 0, V4I, 0)
            wvv = bld(rtw, v * i32(16), 0, V4I, 0)
            e = n4 * i32(4) + lane  # the < 4 tail ints, by CTA 0
            eok = (bid == i32(0)) & (e < n)
            ide = bld(rid, e * i32(4), 0, T.i32, 0)
            wte = bld(rtw, e * i32(4), 0, T.i32, 0)
            big = i32(1 << 30)
            for p in range_constexpr(TPC):
                base = fx.Int64(a["peer"][p])
                ri = rsrc(base + fx.Int64(a["off_ids"]), big)
                rw = rsrc(base + fx.Int64(a["off_w"]), big)
                vo = vok.select(dst0 + v * i32(16), big)
                eo = eok.select(dst0 + e * i32(4), big)
                bst(idv, ri, vo, 0, AUX_SYS)
                bst(wvv, rw, vo, 0, AUX_SYS)
                bst(ide, ri, eo, 0, AUX_SYS)
                bst(wte, rw, eo, 0, AUX_SYS)
            wait_vm(0)
            if lane == i32(0):
                for p in range_constexpr(TPC):
                    rf = rsrc(fx.Int64(a["peer"][p]) + fx.Int64(a["off_flag"]))
                    bst(
                        a["epoch"],
                        rf,
                        (i32(FLAG_AGM) + rank * i32(32) + bid) * i32(4),
                        0,
                        AUX_SYS,
                    )

    def _wave_any(v, lane):
        for k in (1, 2, 4, 8, 16, 32):
            v = fx.max(
                v, i32(rocdl.ds_bpermute(T.i32, _u((lane ^ i32(k)) * i32(4)), _u(v)))
            )
        return uni(v)

    def _meta_pending(a, lane, epoch, own=False):
        rf = rsrc(peer_sel(a, a["rank"]) + fx.Int64(a["off_flag"]))
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
        return _wave_any(pend, lane)

    @traced
    def ag_stage_free(L, lane):
        """GEMM1 is about to write the GEMM2 operand: the AllGather staging
        that shares the area must have been read out."""
        if lane == i32(0):
            spin_lds_ge(L, L_CTL + C_AGFREE * 4, i32(1))
        rocdl.sched_barrier(0)

    def _ag_ctr(a, p, idx):
        return fx.Int64(a["peer"][p]) + fx.Int64(a["off_flag"]) + fx.Int64(idx * i32(4))

    @traced
    def _ag_bump(a, lane, rank, q):
        """Chunk q of this CTA's rows landed everywhere: raise its flag."""
        if lane == i32(0):
            idx = (
                i32(FLAG_AGC)
                + (q * i32(MAX_TP) + rank) * i32(NCTA_MAX)
                + i32(gpu.block_id("x"))
            )
            for p in range_constexpr(TPC):
                rf = rsrc(fx.Int64(a["peer"][p]) + fx.Int64(a["off_flag"]))
                bst(a["epoch"], rf, idx * i32(4), 0, AUX_SYS)

    @traced
    def ag_wait_meta(a, tid, epoch):
        """Every rank's routing metadata landed (checked by compute wave 0)."""
        if tid < i32(64):
            lane = tid % i32(64)
            pend = _meta_pending(a, lane, epoch)
            while pend != i32(0):
                rocdl.s_sleep(1)
                pend = _meta_pending(a, lane, epoch)

    def _ag_pending(L, lane, a, epoch, cc, r0, rows):
        """1 if some row of this tile still misses K-chunk cc (wave-uniform)."""
        rf = rsrc(peer_sel(a, a["rank"]) + fx.Int64(a["off_flag"]))
        nblk = i32(gpu.grid_dim.x)
        pend = i32(0)
        for jr in range_constexpr((RG + 63) // 64):
            row = i32(jr * 64) + lane
            ok = row < rows
            tok = lds_ld_i32(L, L_RIX + (r0 + ok.select(row, i32(0))) * i32(4)) // i32(
                TOPK
            )
            srank = tok // a["m"]
            scta = (tok - srank * a["m"]) % nblk
            idx = i32(FLAG_AGC) + (cc * i32(MAX_TP) + srank) * i32(NCTA_MAX) + scta
            f = fx.Int32(bld(rf, idx * i32(4), 0, T.i32, AUX_SYS))
            pend = fx.max(pend, (ok & (f < epoch)).select(i32(1), i32(0)))
        return _wave_any(pend, lane)

    @traced
    def ag_wait_chunk(L, lane, a, epoch, cc, r0, rows):
        """Loader: K-chunk cc of every row of this tile is in the arena."""
        pend = _ag_pending(L, lane, a, epoch, cc, r0, rows)
        while pend != i32(0):
            rocdl.s_sleep(1)
            pend = _ag_pending(L, lane, a, epoch, cc, r0, rows)
        rocdl.sched_barrier(0)

    @traced
    def finish(tid, a, epoch):
        if tid == i32(0):
            nblk = gpu.grid_dim.x
            fin = fx.Int64(a["ctrl"]) + fx.Int64(8)
            old = g_add_agent(fin, 1)
            if old == i32(nblk) - i32(1):
                g_st_sys(fin, i32(0))
                _llvm.StoreOp(
                    _u(i32(epoch)),
                    gptr(fx.Int64(a["ctrl"])),
                    alignment=4,
                    ordering=_llvm.AtomicOrdering.release,
                    syncscope="agent",
                )

    @traced
    def init_lds(L, tid, a):
        # One writer per control int: done[] start at -1, the epoch slot takes
        # this launch's epoch, everything else 0.
        if tid < i32(64):
            is_done = (tid >= i32(C_DONE)) & (tid < i32(C_DONE + NW))
            ep = g_ld_rel(fx.Int64(a["ctrl"]), "agent") + i32(1)
            v = is_done.select(i32(-1), (tid == i32(C_EPOCH)).select(ep, i32(0)))
            lds_st(L, L_CTL + tid * i32(4), v)

    @traced
    def roles(L, tid, a, epoch):
        if tid < i32(NT):
            compute_units(L, tid, a)
            if tid == i32(0):
                _llvm.StoreOp(_u(_now()), gptr(_paddr(a, 3)), alignment=8)
            comm_help(L, tid, a, epoch)
        else:
            if tid < i32(NT + 64):
                ag_send(L, tid % i32(64), a)
                comm_wave(L, tid, a, epoch)
            else:
                a_loader(L, tid, a)
                signal_loop(L, tid, a, epoch)
                comm_help(L, tid, a, epoch)

    # Explicit annotation: the module's postponed annotations cannot see LDS_BYTES.
    Shared = fx.struct(
        type(
            "Shared", (), {"__annotations__": {"buf": fx.Array[fx.Int8, LDS_BYTES, 16]}}
        )
    )

    @flyc.kernel(name=name, known_block_size=[NTT, 1, 1])
    def fused_tp_kernel(
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
        rank: fx.Int32,
        tp: fx.Int32,
        m: fx.Int32,
        mmax: fx.Int32,
        pieces: fx.Int32,
        piece_e0: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        lds = fx.SharedAllocator().allocate(Shared).peek()
        L = uni(fx.Int32(fx.ptrtoint(lds.buf.ptr)))
        peers = [p0, p1, p2, p3, p4, p5, p6, p7]
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
            "off_x": off_x,
            "off_xs": off_xs,
            "off_ids": off_ids,
            "off_w": off_w,
            "off_part": off_part,
            "off_flag": off_flag,
            "rank": rank,
            "tp": tp,
            "m": m,
            "mmax": mmax,
            "pieces": pieces,
            "piece_e0": piece_e0,
            "ttot": tp * m,
        }
        if tid == i32(0):
            _llvm.StoreOp(_u(_now()), gptr(_paddr(a, 0)), alignment=8)
        init_lds(L, tid, a)
        gpu.barrier()
        epoch = lds_ld_i32(L, L_CTL + C_EPOCH * 4)
        a["epoch"] = epoch
        # Drop stale cached copies of the arena (previous launch). Every later
        # arena read happens only after its data's flag is up, so doing it at
        # the start suffices. Each CU drops its own L1; each XCD's L2 is dropped
        # once, by its first workgroup -- 256 L2 invalidations back to back
        # stall every XCD's memory traffic for ~20 us.
        _drop_stale(tid, a)
        if (tid >= i32(NT)) & (tid < i32(NT + 64)):
            _ag_send_meta(tid % i32(64), a)
        ag_quant(L, tid, a)
        gpu.barrier()
        # the gathered operand of this rank lives in its own arena
        mine = peers[0]
        for j in range_constexpr(1, MAX_TP):
            mine = (rank == i32(j)).select(peers[j], mine)
        a["ax"] = fx.Int64(mine) + off_x
        a["axs"] = fx.Int64(mine) + off_xs
        a["ids"] = fx.Int64(mine) + off_ids
        a["tw"] = fx.Int64(mine) + off_w
        roles(L, tid, a, epoch)
        gpu.barrier()
        finish(tid, a, epoch)
        if tid == i32(0):
            _llvm.StoreOp(_u(_now()), gptr(_paddr(a, 5)), alignment=8)

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
        rank: fx.Int32,
        tp: fx.Int32,
        m: fx.Int32,
        mmax: fx.Int32,
        pieces: fx.Int32,
        piece_e0: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        fused_tp_kernel(
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
            rank,
            tp,
            m,
            mmax,
            pieces,
            piece_e0,
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(NTT, 1, 1), stream=stream)

    launch.block = NTT
    launch.consts = c
    return launch
