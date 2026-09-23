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
    "MAX_TP",
    "NCK_MAX",
    "compile_fused_tp",
    "fused_tp_consts",
    "fused_tp_supported",
]

MAX_TP = 8
NCK_MAX = 64
FLAG_STRIDE = 16  # ints between AllGather flags (one 64 B line each)
FLAG_RDY = MAX_TP * FLAG_STRIDE
CTRL_CNT = 64  # ctrl ints: [0] epoch [1] ag_arrive [2] fin; counters at 64
CTRL_LRDY = CTRL_CNT + 2 * NCK_MAX
LRDY_STRIDE = 32  # polled flags live on their own 128 B lines
CTRL_INTS = CTRL_LRDY + NCK_MAX * LRDY_STRIDE

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
    NPC = max(int(npieces), 1)  # pieces per split expert (1: no split experts)
    # Route rows: bf16, or (route_fp8) E4M3 with one E8M0 scale per 32 columns
    # -- the numerics of the split path's FP8 stage-2 route output -- which
    # halves the GEMM2 -> ReduceScatter traffic. fp8 layout per region: all
    # rows' H data bytes, then all rows' H/32 scale bytes.
    FP8R = bool(route_fp8)
    # Tokens per push batch. With more than two pieces per split expert the
    # partial-row offsets the push keeps live push it past 256 VGPRs (spills),
    # so it takes half the batch.
    TB = max(1, (RED_INFLIGHT // 2 if NPC > 2 else RED_INFLIGHT) // TOPK)
    ROW_B = H + H // 32 if FP8R else 2 * H
    name = (
        f"mega_moe_tp_fused_h{H}_i{I}_k{TOPK}_mt{MT}_t{TMAX}_{act}"
        + (f"_p{NPC}" if NPC > 1 else "")
        + ("_r8" if route_fp8 else "")
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

    def dma4(lds_addr, rs, voff, soff):
        asm(
            "s_mov_b32 m0, $0\n\tbuffer_load_dword $1, $2, $3 offen lds",
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
        for q_ in range(tid, n4, i32(NT)):
            q = i32(q_)
            v = fx.Vector(bld(rid, q * i32(16), 0, V4I, 0))
            for j in range_constexpr(4):
                if fx.Int32(v[j]) == expert:
                    slot = lds_atomic_add(L, L_CTL + C_CNT * 4, 1)
                    idx = q * i32(4) + i32(j)
                    if slot < i32(TMAX):
                        lds_st(L, L_RIX + slot * i32(4), idx)
                        lds_st(
                            L,
                            L_WT + slot * i32(4),
                            g_ld_i32(fx.Int64(tw_addr) + fx.Int64(idx) * fx.Int64(4)),
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
                            dma4(dst, rxs, ascl[j], cc * i32(8) + i32(k * 4))
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
    def compute_units(L, tid, a):
        lane = tid % i32(64)
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
            R = gather_routes(L, tid, a["ids"], a["tw"], a["ttot"], expert)
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

    @traced
    def push_chunk(tid, a, epoch, cidx):
        lane = tid % i32(64)
        nblk = gpu.grid_dim.x
        ttot = a["ttot"]
        c0 = cidx * i32(CW)
        routes_bytes = route_region_bytes(ttot)
        r_routes = rsrc(a["routes"], routes_bytes)
        # partial rows of pieces 1.. of split experts; others read past the end (0)
        pr_bytes = i32(NPC - 1) * routes_bytes
        r_pr = rsrc(a["proutes"], pr_bytes)
        ids_base = peer_sel(a, a["rank"]) + fx.Int64(a["off_ids"])
        for t0_ in range(i32(gpu.block_id("x")), ttot, i32(nblk) * i32(TB)):
            t0 = i32(t0_)
            d = []
            for b in range_constexpr(TB):
                t = fx.min(t0 + i32(b) * i32(nblk), ttot - i32(1))
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
                t_raw = t0 + i32(b) * i32(nblk)
                t = fx.min(t_raw, ttot - i32(1))
                owner = t // a["m"]
                orow = t - owner * a["m"]
                dst = (
                    peer_sel(a, owner)
                    + fx.Int64(a["off_part"])
                    + fx.Int64(((a["rank"] * a["mmax"] + orow) * i32(H) + c0) * i32(2))
                )
                r_dst = rsrc(dst)
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
                                ids_base + fx.Int64((t * i32(TOPK) + i32(k)) * i32(4))
                            )
                            for k in range(TOPK)
                        ]
                        hit = eids[0] >= a["piece_e0"]
                        for k in range_constexpr(1, TOPK):
                            hit = hit | (eids[k] >= a["piece_e0"])
                        _push_split_token(
                            a,
                            r_dst,
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
                        _push_store(r_dst, v, pack_bf16x8(acc), live)
        wait_vm(0)
        _push_done(a, lane, epoch, cidx, nblk)

    @traced
    def _push_split_token(
        a, r_dst, r_pr, v, acc, live, hit, eids, t, c0, rbytes, prbytes
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
            _push_store(r_dst, v, pack_bf16x8(tot), live)
        else:
            _push_store(r_dst, v, pack_bf16x8(acc), live)

    @traced
    def _push_store(r_dst, v, o, ok):
        if ok:
            bst(o, r_dst, v * i32(16), 0, AUX_SYS)

    @traced
    def _push_done(a, lane, epoch, cidx, nblk):
        if lane == i32(0):
            cnt_addr = fx.Int64(a["ctrl"]) + fx.Int64(
                (i32(CTRL_CNT + NCK_MAX) + cidx) * i32(4)
            )
            old = g_add_agent(cnt_addr, 1)
            if old == i32(nblk) - i32(1):
                g_st_sys(cnt_addr, i32(0))
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
                    vals = bf16x8_to_f32(
                        bld(
                            r_recv,
                            ((pc * a["mmax"] + row) * i32(H) + c0 + vc * i32(8))
                            * i32(2),
                            0,
                            V4I,
                            AUX_SYS,
                        )
                    )
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
            push_chunk(tid, a, epoch, cr)
        cp = claim(L, lane, w, C_PULL, 1, a, epoch)
        if cp >= i32(0):
            final_chunk(tid, a, cp)
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
    @traced
    def quant_push_row(L, tid, a, i):
        gslot = a["rank"] * a["m"] + i
        rxl = rsrc(a["x"])
        for g_ in range(tid, i32(H // 32), i32(NTT)):
            g = i32(g_)
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
            # 16 consecutive groups' E8M0 bytes -> one 16 B word held by the
            # group-of-16 leader (H/32 is a multiple of 16, so a 16-group span
            # is always fully active); no LDS round trip, no barrier.
            lane = tid % i32(64)
            sw = (fx.Int32(e8) & i32(0xFF)) << ((g & i32(3)) * i32(8))
            sw = sw | i32(
                rocdl.ds_bpermute(T.i32, _u((lane ^ i32(1)) * i32(4)), _u(sw))
            )
            sw = sw | i32(
                rocdl.ds_bpermute(T.i32, _u((lane ^ i32(2)) * i32(4)), _u(sw))
            )
            sv = fx.Vector.from_elements(
                [sw]
                + [
                    i32(
                        rocdl.ds_bpermute(
                            T.i32, _u((lane + i32(4 * k)) * i32(4)), _u(sw)
                        )
                    )
                    for k in range(1, 4)
                ],
                fx.Int32,
            )
            pkv = fx.Vector.from_elements(words, fx.Int32)
            for p in range_constexpr(MAX_TP):
                if i32(p) < a["tp"]:
                    bst(
                        pkv,
                        rsrc(
                            fx.Int64(a["peer"][p])
                            + fx.Int64(a["off_x"])
                            + fx.Int64(gslot) * fx.Int64(H // 2)
                        ),
                        g * i32(16),
                        0,
                        AUX_SYS,
                    )
            if (g & i32(15)) == i32(0):
                for p in range_constexpr(MAX_TP):
                    if i32(p) < a["tp"]:
                        bst(
                            sv,
                            rsrc(
                                fx.Int64(a["peer"][p])
                                + fx.Int64(a["off_xs"])
                                + fx.Int64(gslot) * fx.Int64(H // 32)
                            ),
                            g,
                            0,
                            AUX_SYS,
                        )
        kk = tid - i32(NTT - 64)
        if (kk >= i32(0)) & (kk < i32(TOPK)):
            idv = g_ld_i32(
                fx.Int64(a["ids_in"]) + fx.Int64(i * i32(TOPK) + kk) * fx.Int64(4)
            )
            wvv = g_ld_i32(
                fx.Int64(a["tw_in"]) + fx.Int64(i * i32(TOPK) + kk) * fx.Int64(4)
            )
            for p in range_constexpr(MAX_TP):
                if i32(p) < a["tp"]:
                    base = fx.Int64(a["peer"][p])
                    g_st_sys(
                        base
                        + fx.Int64(a["off_ids"])
                        + fx.Int64(gslot * i32(TOPK) + kk) * fx.Int64(4),
                        idv,
                    )
                    g_st_sys(
                        base
                        + fx.Int64(a["off_w"])
                        + fx.Int64(gslot * i32(TOPK) + kk) * fx.Int64(4),
                        wvv,
                    )

    @traced
    def ag_handshake(L, tid, a, epoch):
        if tid == i32(0):
            nblk = gpu.grid_dim.x
            arr = fx.Int64(a["ctrl"]) + fx.Int64(4)
            old = g_add_agent(arr, 1)
            if old == i32(nblk) - i32(1):
                g_st_sys(arr, i32(0))
                for p in range_constexpr(MAX_TP):
                    if i32(p) < a["tp"]:
                        g_st_sys(
                            fx.Int64(a["peer"][p])
                            + fx.Int64(a["off_flag"])
                            + fx.Int64(a["rank"] * i32(FLAG_STRIDE * 4)),
                            epoch,
                        )
            mine = peer_sel(a, a["rank"]) + fx.Int64(a["off_flag"])
            for p in range_constexpr(MAX_TP):
                if i32(p) < a["tp"]:
                    spin_sys_ge(mine + fx.Int64(p * FLAG_STRIDE * 4), epoch)

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
            comm_help(L, tid, a, epoch)
        else:
            if tid < i32(NT + 64):
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
        init_lds(L, tid, a)
        gpu.barrier()
        epoch = lds_ld_i32(L, L_CTL + C_EPOCH * 4)
        # AllGather: quantize + push this rank's tokens.
        for i_ in range(i32(gpu.block_id("x")), m, i32(gpu.grid_dim.x)):
            i = i32(i_)
            quant_push_row(L, tid, a, i)
        wait_vm(0)
        gpu.barrier()
        ag_handshake(L, tid, a, epoch)
        gpu.barrier()
        fence(_llvm.AtomicOrdering.acquire, "one-as")
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
