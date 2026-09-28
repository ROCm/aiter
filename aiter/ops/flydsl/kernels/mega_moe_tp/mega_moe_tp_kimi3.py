# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# TileRT shared/reuse MonoKernel reference (fusion-boundary comparison):
# https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9
# Kimi-K3 routed latent-MoE shapes and SiTUv2 activation follow
# https://github.com/ROCm/FlyDSL/pull/1204 (kernels/common/fused_layer_config.py).

"""Kimi-K3 decode MoE + TP all-reduce in one launch (A4W4, gfx950).

Per TP8 rank and launch of S = 1, 2 or 4 tokens:
  logits = x @ Wr^T (x [S, 7168] bf16, Wr [896, 7168] bf16), scores = sigmoid
  top-16 of scores + bias (f32, ties to the lower id), probs = score / sum
  per slot: ug = W_ug[e] @ mxfp4(latent) (W_ug [768, 3584] MXFP4, latent [S, 3584])
            mid = bf16(SiTUv2(gate, up)), beta 4, linear beta 25
            part += prob * W_dn[e] @ mxfp4(mid) (W_dn [3584, 384] MXFP4)
  out = sum over ranks of bf16(part) (+ residual)
256 workgroups x 512 threads, all resident; every workgroup belongs to one
token (BPS = 256 / S per token). Cross-workgroup data (scores, mids) goes
through tagged 8-byte pairs in global memory, the all-reduce through flag
packets in a symmetric peer buffer.
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .. import buffer_ops

__all__ = [
    "DN_BLOCKS",
    "DN_EXPERT_BYTES",
    "DN_SC_EXPERT",
    "ERR_AR",
    "ERR_MIDS",
    "ERR_SCORES",
    "GRID",
    "HIDDEN",
    "INTER",
    "MAX_PES",
    "MX_BLOCK",
    "NUM_EXPERTS",
    "ROUTER_HIDDEN",
    "RT_CH",
    "RT_WAVES",
    "SITU_BETA",
    "SITU_LINEAR_BETA",
    "SLOTS",
    "SUPPORTED_SAMPLES",
    "SYM_BYTES",
    "TIMELINE_SLOTS",
    "TOP_K",
    "UG_EXPERT_BYTES",
    "UG_SC_EXPERT",
    "UG_TILES",
    "compile_mega_moe_tp_kimi3",
]

ROUTER_HIDDEN = 7168  # router input (model hidden)
HIDDEN = 3584  # routed latent: expert K and output rows
NUM_EXPERTS = 896
TOP_K = 16
SLOTS = TOP_K
INTER = 384  # per TP8 rank
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
ROUTE_SCALE = 1.0
GRID = 256
NT = 512
NWAVE = NT // 64
MAX_PES = 8
SUPPORTED_SAMPLES = (1, 2, 4)
MX_BLOCK = 32

KBLK = HIDDEN // 128  # 28 k-blocks of the up/gate GEMV
KB_BYTES = 64 * 16
RT_WAVES = NWAVE - 1
RT_CH = KBLK // RT_WAVES  # 4 k-blocks per wave
UG_TILES = 2 * INTER // 16  # 48 tiles of 8 gate + 8 up rows
UG_TILE_BYTES = KBLK * KB_BYTES
UG_EXPERT_BYTES = UG_TILES * UG_TILE_BYTES
UG_SC_TILE = RT_WAVES * 64 * 4
UG_SC_EXPERT = UG_TILES * UG_SC_TILE
JOBS = SLOTS * UG_TILES  # up/gate tiles per token
DKB = INTER // 128  # 3 k-blocks of the down GEMV
DN_BLOCKS = HIDDEN // 16  # 224 blocks of 16 output rows
DN_BLK_BYTES = DKB * KB_BYTES
DN_EXPERT_BYTES = DN_BLOCKS * DN_BLK_BYTES
DN_SC_BLK = 64 * 4
DN_SC_EXPERT = DN_BLOCKS * DN_SC_BLK
ROUTER_CTAS = NUM_EXPERTS // 4  # 4 experts per workgroup
RK_CHUNKS = ROUTER_HIDDEN // 2 // (64 * 8)  # 16 B chunks per lane per half row
SC_PER_LANE = NUM_EXPERTS // 64
MID_LANES = INTER // 8  # lanes holding 8 mids of a slot
ASC_STRIDE = 32  # bytes per lane group in the activation-scale table
WIRE_PKT = 16
SYM_BYTES = (MAX_PES - 1) * 2 * GRID * WIRE_PKT * 4 * 4

TIMELINE_SLOTS = 16
POLL_LIMIT = 1 << 22
ERR_SCORES, ERR_MIDS, ERR_AR = 1, 2, 4
LDS_MAX = 160 * 1024

LOG2E = 1.4426950408889634
INV_FP4_MAX_BITS = 0x3E2AAAAB
E8M0_ONE = 127

AUX_NT = 2
AUX_SC1 = 16
AUX_SYS = 1 | 16

ROW_SHR = (0x111, 0x112, 0x114, 0x118)
ROW_BCAST15, ROW_BCAST31 = 0x142, 0x143
QUAD_XOR1, QUAD_XOR2 = 0xB1, 0x4E

traced = ASTRewriter.transform


def _u(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _attr(v):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), int(v))


def _maxnb(samples: int) -> int:
    bps = GRID // samples
    return (DN_BLOCKS + bps - 1) // bps


@functools.cache
def mega_moe_tp_kimi3_consts(samples: int) -> dict:
    S = samples
    c = {"S": S, "BPS": GRID // S, "NJ": JOBS * S // GRID, "MAXNB": _maxnb(S)}
    c["ROWS"] = 16 * c["MAXNB"]
    c["PKTS"] = c["ROWS"] // 4
    c["WIRE"] = c["PKTS"] * WIRE_PKT
    off = 0

    def take(n, align=16):
        nonlocal off
        off = (off + align - 1) // align * align
        start = off
        off += n
        return start

    c["L_ACT"] = take(HIDDEN // 2)
    c["L_ASC"] = take(4 * ASC_STRIDE)
    c["L_REDR"] = take(NWAVE * S * 4)
    c["L_RED"] = take(c["NJ"] * NWAVE * 16 * 4)
    c["L_MIDT"] = take(c["NJ"] * 16)
    c["L_TIDX"] = take(TOP_K * 4)
    c["L_TPRB"] = take(TOP_K * 4)
    c["L_SIG"] = take(NUM_EXPERTS * 4)
    c["L_MID4"] = take(SLOTS * INTER // 2)
    c["L_MSC"] = take(SLOTS * INTER // MX_BLOCK)
    c["L_P"] = take(c["MAXNB"] * SLOTS * 16 * 4)
    c["L_PART"] = take(c["ROWS"] * 2)
    c["L_RECV"] = take((MAX_PES - 1) * c["ROWS"] * 2)
    c["LDS_BYTES"] = (off + 127) // 128 * 128
    assert JOBS * S % GRID == 0 and c["PKTS"] <= 64
    assert c["LDS_BYTES"] <= LDS_MAX
    return c


@functools.cache
def compile_mega_moe_tp_kimi3(samples: int, timeline: bool = False, device: int = 0):
    if samples not in SUPPORTED_SAMPLES:
        raise ValueError(f"samples must be one of {SUPPORTED_SAMPLES}, got {samples}")
    c = mega_moe_tp_kimi3_consts(samples)
    S, BPS, NJ, MAXNB = c["S"], c["BPS"], c["NJ"], c["MAXNB"]
    ROWS, PKTS, WIRE = c["ROWS"], c["PKTS"], c["WIRE"]
    L_ACT, L_ASC, L_REDR, L_RED, L_MIDT = c["L_ACT"], c["L_ASC"], c["L_REDR"], c["L_RED"], c["L_MIDT"]
    L_TIDX, L_TPRB, L_SIG, L_MID4, L_MSC = c["L_TIDX"], c["L_TPRB"], c["L_SIG"], c["L_MID4"], c["L_MSC"]
    L_P, L_PART, L_RECV, LDS_BYTES = c["L_P"], c["L_PART"], c["L_RECV"], c["LDS_BYTES"]
    name = f"mega_moe_tp_kimi3_s{S}" + ("_tl" if timeline else "")
    const_expr = fx.const_expr

    def i32(v):
        return fx.Int32(v)

    def i64(v):
        return fx.Int64(v)

    def f32(v):
        return fx.Float32(v)

    def uni(v):
        return i32(rocdl.readfirstlane(T.i32, _u(i32(v))))

    def uni64(v):
        v = i64(v)
        lo = uni(i32(v & i64(0xFFFFFFFF)))
        hi = uni(i32(v >> i64(32)))
        return (i64(hi) << i64(32)) | (i64(lo) & i64(0xFFFFFFFF))

    def rsrc(addr, nbytes=None):
        if nbytes is None:
            return buffer_ops.create_buffer_resource_from_addr(_u(uni64(addr)))
        return buffer_ops.create_buffer_resource_from_addr(
            _u(uni64(addr)), num_records_bytes=_u(i64(uni(nbytes)))
        )

    def bld(rs, voff, ty, aux=0, soff=0):
        return rocdl.raw_ptr_buffer_load(ty, rs, _u(i32(voff)), _u(i32(soff)), aux=_attr(aux))

    def bst(val, rs, voff, aux=0, soff=0):
        rocdl.raw_ptr_buffer_store(_u(val), rs, _u(i32(voff)), _u(i32(soff)), aux=_attr(aux))

    def ld_v4i(rs, voff, aux=0):
        return fx.Vector(bld(rs, voff, T.vec(4, T.i32), aux))

    def ld_f(rs, voff):
        return f32(bld(rs, voff, T.f32))

    def lds_ptr(base, off, ty, align):
        if isinstance(ty, ir.VectorType):
            ty = ty.element_type
        return fx.inttoptr(fx.PointerType.get(ty, fx.AddressSpace.Shared, align), base + i32(off))

    def lds_ld(base, off, ty, align=4):
        return fx.ptr_load(lds_ptr(base, off, ty, align), result_type=ty)

    def lds_st(base, off, val, align=4):
        v = _u(val)
        fx.ptr_store(v, lds_ptr(base, off, v.type, align))

    def lds_i_vol(base, off):
        p3 = _llvm.IntToPtrOp(_llvm.PointerType.get(address_space=3), _u(i32(base) + i32(off))).result
        return i32(
            _llvm.LoadOp(
                T.i32, p3, alignment=4, ordering=_llvm.AtomicOrdering.monotonic, syncscope="workgroup"
            ).res
        )

    def lds_f(base, off):
        return f32(lds_ld(base, off, T.f32))

    def lds_i(base, off):
        return i32(lds_ld(base, off, T.i32))

    def lds_u8(base, off):
        return i32(lds_ld(base, off, T.i8, 1)) & i32(0xFF)

    def gptr(addr):
        return _llvm.IntToPtrOp(_llvm.PointerType.get(address_space=1), _u(i64(addr))).result

    def ld_agent_i64(addr):
        return i64(
            _llvm.LoadOp(
                T.i64, gptr(addr), alignment=8, ordering=_llvm.AtomicOrdering.monotonic, syncscope="agent"
            ).res
        )

    def ld_agent_i32(addr):
        return i32(
            _llvm.LoadOp(
                T.i32, gptr(addr), alignment=4, ordering=_llvm.AtomicOrdering.monotonic, syncscope="agent"
            ).res
        )

    def ld_sys_v4i(addr):
        return fx.Vector(_llvm.LoadOp(T.vec(4, T.i32), gptr(addr), alignment=16, volatile_=True).res)

    def st_agent_i32(addr, v):
        _llvm.StoreOp(
            _u(i32(v)), gptr(addr), alignment=4, ordering=_llvm.AtomicOrdering.monotonic, syncscope="agent"
        )

    def atomic_or_agent(addr, v):
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp._or, gptr(addr), _u(i32(v)), _llvm.AtomicOrdering.monotonic, syncscope="agent"
        )

    def ballot(pred):
        return i64(rocdl.ballot(T.i64, _u(pred)))

    def readlane_i(v, lane):
        return i32(rocdl.readlane(T.i32, _u(i32(v)), _u(i32(lane))))

    def dpp(src, ctrl):
        return i32(
            _llvm.call_intrinsic(
                T.i32,
                "llvm.amdgcn.update.dpp.i32",
                [_u(i32(0)), _u(i32(src)), _u(i32(ctrl)), _u(i32(0xF)), _u(i32(0xF)), _u(fx.Boolean(True))],
                [],
                [],
            )
        )

    def dpp_f(x, ctrl):
        return dpp(f32(x).bitcast(fx.Int32), ctrl).bitcast(fx.Float32)

    def wave_sum_f32(x):
        x = f32(x)
        for ctl in ROW_SHR + (ROW_BCAST15, ROW_BCAST31):
            x = x + dpp_f(x, ctl)
        return f32(rocdl.readlane(T.f32, _u(x), _u(i32(63))))

    def wave_max_i32(x):
        x = i32(x)
        for ctl in ROW_SHR + (ROW_BCAST15, ROW_BCAST31):
            x = fx.max(x, dpp(x, ctl))
        return readlane_i(x, 63)

    def wave_min_i32(x):
        x = i32(x)
        for ctl in ROW_SHR + (ROW_BCAST15, ROW_BCAST31):
            x = fx.min(x, dpp(x, ctl))
        return readlane_i(x, 63)

    def quad_max_pos(x):
        v = f32(x).bitcast(fx.Int32)
        for ctl in (QUAD_XOR1, QUAD_XOR2):
            v = fx.max(v, dpp(v, ctl))
        return v.bitcast(fx.Float32)

    def exp2_raw(x):
        return f32(_llvm.call_intrinsic(T.f32, "llvm.amdgcn.exp2.f32", [_u(f32(x))], [], []))

    def sigmoid(x):
        return f32(1.0) / (f32(1.0) + exp2_raw(f32(x) * f32(-LOG2E)))

    def tanh(x):
        return f32(2.0) * sigmoid(f32(2.0) * f32(x)) - f32(1.0)

    def bf16_lo(w):
        return (i32(w) << i32(16)).bitcast(fx.Float32)

    def bf16_hi(w):
        return (i32(w) & i32(-65536)).bitcast(fx.Float32)

    def bf16_bits(f):
        return i32(
            fx.Vector.from_elements([f32(f).to(fx.BFloat16)], fx.BFloat16).bitcast(fx.Int16)[0]
        ) & i32(0xFFFF)

    def _absf(x):
        return (f32(x).bitcast(fx.Int32) & i32(0x7FFFFFFF)).bitcast(fx.Float32)

    def mx_scale(amax):
        w = (f32(amax) * i32(INV_FP4_MAX_BITS).bitcast(fx.Float32)).bitcast(fx.Int32)
        e = (w.shrui(i32(23)) & i32(0xFF)) + ((w & i32(0x7FFFFF)) != i32(0)).select(i32(1), i32(0))
        e = fx.min(e, i32(254))
        inv = ((i32(254) - e) << i32(23)).bitcast(fx.Float32)
        return e, inv

    def fp4x4(v, inv):
        one = _u(f32(1.0))
        w = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, _u(i32(0)), _u(v[0] * inv), _u(v[1] * inv), one, 0)
        w = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, w, _u(v[2] * inv), _u(v[3] * inv), one, 1)
        return i32(w) & i32(0xFFFF)

    def fp4x8(v, inv):
        return fp4x4(v[0:4], inv) | (fp4x4(v[4:8], inv) << i32(16))

    def mfma_fp4(a4, b4, acc, sa, opa, sb, opb):
        return fx.Vector(
            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.vec(4, T.f32), [_u(a4), _u(b4), _u(acc), 4, 4, opa, _u(i32(sa)), opb, _u(i32(sb))]
            )
        )

    def fdot2(a, b, acc):
        av = fx.Vector.from_elements([i32(a)], fx.Int32).bitcast(fx.BFloat16)
        bv = fx.Vector.from_elements([i32(b)], fx.Int32).bitcast(fx.BFloat16)
        return f32(rocdl.fdot2_f32_bf16_(T.f32, _u(av), _u(bv), _u(f32(acc)), clamp=False))

    def _order_key(x):
        b = f32(x).bitcast(fx.Int32)
        return (b < i32(0)).select(b ^ i32(0x7FFFFFFF), b)

    def _tag_bad(v, tag):
        return i32(v >> i64(32)) != tag

    # ------------------------------------------------------------ router
    def router_dot(a, wave, lane, bid):
        """Workgroup bid < ROUTER_CTAS: logits of experts 4*bid + (wave & 3),
        K half wave >> 2, for every token (one partial per wave)."""
        e = fx.min(bid, i32(ROUTER_CTAS - 1)) * i32(4) + (wave & i32(3))
        k0 = (wave >> i32(2)) * i32(ROUTER_HIDDEN // 2)
        # workgroups past the last expert block read nothing (0-byte resource)
        wrs = rsrc(
            i64(a["router_w"]) + i64(e) * i64(ROUTER_HIDDEN * 2),
            (bid < i32(ROUTER_CTAS)).select(i32(ROUTER_HIDDEN * 2), i32(0)),
        )
        xrs = rsrc(a["x"])
        ws = [ld_v4i(wrs, (k0 + (i32(ch * 64) + lane) * i32(8)) * i32(2), AUX_NT) for ch in range(RK_CHUNKS)]
        accs = []
        for s in range_constexpr(S):
            acc = f32(0.0)
            for ch in range_constexpr(RK_CHUNKS):
                xv = ld_v4i(xrs, (i32(s * ROUTER_HIDDEN) + k0 + (i32(ch * 64) + lane) * i32(8)) * i32(2))
                for q in range_constexpr(4):
                    acc = fdot2(ws[ch][q], xv[q], acc)
            accs.append(wave_sum_f32(acc))
        return accs

    @traced
    def router_publish(L, a, tid, wave, lane, bid, accs, tag):
        if lane == i32(0):
            for s in range_constexpr(S):
                lds_st(L, L_REDR + (wave * i32(S) + i32(s)) * i32(4), accs[s])
        gpu.barrier()
        if (bid < i32(ROUTER_CTAS)) & (tid < i32(4 * S)):
            r = tid & i32(3)
            s = tid >> i32(2)
            logit = lds_f(L, L_REDR + (r * i32(S) + s) * i32(4)) + lds_f(L, L_REDR + ((r + i32(4)) * i32(S) + s) * i32(4))
            e = bid * i32(4) + r
            bst(logit, rsrc(a["scores"]), (s * i32(NUM_EXPERTS) + e) * i32(4))
            pair = fx.Vector.from_elements([sigmoid(logit).bitcast(fx.Int32), tag], fx.Int32)
            bst(pair, rsrc(a["score_pairs"]), (s * i32(NUM_EXPERTS) + e) * i32(8), AUX_SC1)

    # ----------------------------------------------------- activation quant
    @traced
    def act_quant(L, a, wave, lane, s):
        """Waves 1..7: MXFP4 of the token's latent K range (wave - 1) * 512 into
        the up/gate B operand (per k-block 64 B, lane group g at 16 * g) and its
        E8M0 scales (byte g * ASC_STRIDE + k-block)."""
        if wave != i32(0):
            q = wave - i32(1)
            k = q * i32(512) + lane * i32(8)
            xv = ld_v4i(rsrc(i64(a["latent"]) + i64(s * i32(HIDDEN * 2))), k * i32(2))
            vals = []
            for j in range_constexpr(4):
                w = i32(xv[j])
                vals += [bf16_lo(w), bf16_hi(w)]
            am = _absf(vals[0])
            for j in range_constexpr(1, 8):
                am = am.maximumf(_absf(vals[j]))
            am = quad_max_pos(am)
            e, inv = mx_scale(am)
            lds_st(L, L_ACT + k // i32(2), fp4x8(vals, inv))
            if (lane & i32(3)) == i32(0):
                kb = k >> i32(7)
                g = (k >> i32(5)) & i32(3)
                lds_st(L, L_ASC + g * i32(ASC_STRIDE) + kb, fx.Int8(e), align=1)

    # --------------------------------------------------------------- top-k
    @traced
    def topk(L, a, lane, wave, s, lb, tag, bid):
        if wave == i32(0):
            base = i64(a["score_pairs"]) + i64(s * i32(NUM_EXPERTS * 8)) + i64(lane * i32(8))
            vs = [ld_agent_i64(base + i64(512 * j)) for j in range(SC_PER_LANE)]
            bad = _any_bad(vs, tag)
            n = i32(0)
            while (bad != i64(0)) & (n < i32(POLL_LIMIT)):
                vs = [ld_agent_i64(base + i64(512 * j)) for j in range(SC_PER_LANE)]
                bad = _any_bad(vs, tag)
                n = n + i32(1)
            _poll_report(a, lane, bid, bad != i64(0), ERR_SCORES)
            brs = rsrc(a["bias"])
            sig = [i32(v & i64(0xFFFFFFFF)).bitcast(fx.Float32) for v in vs]
            keys = []
            for j in range_constexpr(SC_PER_LANE):
                lds_st(L, L_SIG + (lane + i32(64 * j)) * i32(4), sig[j])
                keys.append(_order_key(sig[j] + ld_f(brs, (lane + i32(64 * j)) * i32(4))))
            ids = []
            for it in range_constexpr(TOP_K):
                keys, win = topk_round(keys, lane)
                _lane0_st(L, lane, L_TIDX + i32(it * 4), win)
                ids.append(win)
            scs = [lds_f(L, L_SIG + ids[k] * i32(4)) for k in range(TOP_K)]
            tot = scs[0]
            for k in range_constexpr(1, TOP_K):
                tot = tot + scs[k]
            fac = f32(ROUTE_SCALE) / tot
            my = scs[0] * fac
            my_id = ids[0]
            for k in range_constexpr(1, TOP_K):
                my = (lane == i32(k)).select(scs[k] * fac, my)
                my_id = (lane == i32(k)).select(ids[k], my_id)
            _topk_store(L, a, lane, s, lb, my_id, my)

    def _any_bad(vs, tag):
        bad = _tag_bad(vs[0], tag)
        for v in vs[1:]:
            bad = bad | _tag_bad(v, tag)
        return ballot(bad)

    def topk_round(keys, lane):
        """Take the largest key (ties to the lower expert id) out of the wave."""
        lm = keys[0]
        lj = i32(0)
        for j in range(1, SC_PER_LANE):
            gt = keys[j] > lm
            lm = gt.select(keys[j], lm)
            lj = gt.select(i32(j), lj)
        m = wave_max_i32(lm)
        cand = (lm == m).select(lj * i32(64) + lane, i32(1 << 20))
        win = wave_min_i32(cand)
        wl = win & i32(63)
        wj = win >> i32(6)
        hit = lane == wl
        keys = [(hit & (wj == i32(j))).select(i32(-(2**31)), keys[j]) for j in range(SC_PER_LANE)]
        return keys, win

    @traced
    def _lane0_st(L, lane, off, v):
        if lane == i32(0):
            lds_st(L, off, v)

    @traced
    def _topk_store(L, a, lane, s, lb, my_id, my_prb):
        if lane < i32(TOP_K):
            lds_st(L, L_TPRB + lane * i32(4), my_prb)
            if lb == i32(0):
                o = (s * i32(TOP_K) + lane) * i32(4)
                bst(my_prb, rsrc(a["probs_out"]), o)
                bst(my_id, rsrc(a["indices_out"]), o)

    @traced
    def _spin_slot(L, slot):
        e = lds_i_vol(L, L_TIDX + slot * i32(4))
        while e < i32(0):
            rocdl.s_sleep(1)
            e = lds_i_vol(L, L_TIDX + slot * i32(4))
        return e

    @traced
    def _topk_clear(L, wave, lane):
        if (wave == i32(0)) & (lane < i32(TOP_K)):
            lds_st(L, L_TIDX + lane * i32(4), i32(-1))

    # ------------------------------------------------------------ up/gate
    def job_of(lb, k):
        j = lb + i32(BPS * k)
        return j // i32(UG_TILES), j % i32(UG_TILES)

    def ug_issue(L, a, wave, lane, lb, k):
        slot, tile = job_of(lb, k)
        e = _spin_slot(L, slot)
        base = i64(a["ug_w"]) + i64(e) * i64(UG_EXPERT_BYTES) + i64(tile) * i64(UG_TILE_BYTES)
        rs = rsrc(base)
        cb = (wave - i32(1)) * i32(RT_CH)
        ws = [ld_v4i(rs, (cb + i32(ch)) * i32(KB_BYTES) + lane * i32(16), AUX_NT) for ch in range(RT_CH)]
        sbase = i64(a["ug_scales"]) + i64(e) * i64(UG_SC_EXPERT) + i64(tile) * i64(UG_SC_TILE)
        sv = i32(bld(rsrc(sbase), ((wave - i32(1)) * i32(64) + lane) * i32(4), T.i32))
        rocdl.sched_barrier(0)
        return ws, sv

    def ug_mfma(L, wave, lane, ws, sv, k):
        cb = (wave - i32(1)) * i32(RT_CH)
        g = lane >> i32(4)
        asc = lds_i(L, L_ASC + g * i32(ASC_STRIDE) + cb)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for ch in range_constexpr(RT_CH):
            b4 = lds_ld(L, L_ACT + (cb + i32(ch)) * i32(64) + g * i32(16), T.vec(4, T.i32), 16)
            acc = mfma_fp4(ws[ch], b4, acc, sv, ch, asc, ch)
        _store_red(L, lane, L_RED + ((i32(k * NWAVE) + wave) * i32(16) + g * i32(4)) * i32(4), acc)

    @traced
    def _store_red(L, lane, off, acc):
        if (lane & i32(15)) == i32(0):
            lds_st(L, off, acc, align=16)

    @traced
    def up_jobs(L, a, wave, lane, lb):
        if wave != i32(0):
            pend = ug_issue(L, a, wave, lane, lb, 0)
            for k in range_constexpr(NJ):
                cur = pend
                if k + 1 < NJ:
                    pend = ug_issue(L, a, wave, lane, lb, k + 1)
                ug_mfma(L, wave, lane, cur[0], cur[1], k)

    def situ(g, u):
        gg = f32(SITU_BETA) * tanh(g * f32(1.0 / SITU_BETA)) * sigmoid(g)
        uu = f32(SITU_LINEAR_BETA) * tanh(u * f32(1.0 / SITU_LINEAR_BETA))
        return gg * uu

    @traced
    def up_publish(L, a, wave, lane, s, lb, tag):
        """Job k's 16 rows (8 gate, 8 up) summed over the seven waves -> 8 mids,
        published as tagged pairs by the wave k % NWAVE."""
        for k in range_constexpr(NJ):
            if wave == i32(k % NWAVE):
                slot, tile = job_of(lb, k)
                if lane < i32(8):
                    gsum = f32(0.0)
                    usum = f32(0.0)
                    for w in range_constexpr(1, NWAVE):
                        gsum = gsum + lds_f(L, L_RED + ((i32(k * NWAVE + w)) * i32(16) + lane) * i32(4))
                        usum = usum + lds_f(L, L_RED + ((i32(k * NWAVE + w)) * i32(16) + i32(8) + lane) * i32(4))
                    lds_st(L, L_MIDT + i32(k * 16) + lane * i32(2), fx.Int16(bf16_bits(situ(gsum, usum))), align=2)
                if lane < i32(4):
                    d = lds_i(L, L_MIDT + i32(k * 16) + lane * i32(4))
                    widx = (s * i32(SLOTS) + slot) * i32(INTER) + tile * i32(8) + lane * i32(2)
                    bst(fx.Vector.from_elements([d, tag], fx.Int32), rsrc(a["mid_pairs"]), widx * i32(4), AUX_SC1)
                    if i64(a["hidden_mid"]) != i64(0):
                        bst(d, rsrc(a["hidden_mid"]), ((s * i32(SLOTS) + slot) * i32(INTER) + tile * i32(8) + lane * i32(2)) * i32(2))

    # --------------------------------------------------------------- down
    def dn_block(lb, b):
        return lb + i32(BPS * b)

    def dn_issue(L, a, wave, lane, lb):
        """Down weights of this wave's two slots for every block of the
        workgroup (issued before the mids arrive)."""
        items = []
        for b in range_constexpr(MAXNB):
            blk = fx.min(dn_block(lb, b), i32(DN_BLOCKS - 1))
            for h in range_constexpr(2):
                j = wave * i32(2) + i32(h)
                e = lds_i(L, L_TIDX + j * i32(4))
                rs = rsrc(i64(a["down_w"]) + i64(e) * i64(DN_EXPERT_BYTES) + i64(blk) * i64(DN_BLK_BYTES))
                ws = [ld_v4i(rs, i32(kb * KB_BYTES) + lane * i32(16), AUX_NT) for kb in range(DKB)]
                srs = rsrc(i64(a["down_scales"]) + i64(e) * i64(DN_SC_EXPERT) + i64(blk) * i64(DN_SC_BLK))
                sd = i32(bld(srs, lane * i32(4), T.i32))
                items.append((b, j, ws, sd))
        return items

    @traced
    def mid_slot(L, a, lane, s, j, tag, bid):
        live = lane < i32(MID_LANES)
        lc = fx.min(lane, i32(MID_LANES - 1))
        addr = i64(a["mid_pairs"]) + i64(((s * i32(SLOTS) + j) * i32(INTER) + lc * i32(8)) * i32(4))
        vs = [ld_agent_i64(addr + i64(8 * q)) for q in range(4)]
        bad = ballot(live & (_tag_bad(vs[0], tag) | _tag_bad(vs[1], tag) | _tag_bad(vs[2], tag) | _tag_bad(vs[3], tag)))
        n = i32(0)
        while (bad != i64(0)) & (n < i32(POLL_LIMIT)):
            vs = [ld_agent_i64(addr + i64(8 * q)) for q in range(4)]
            bad = ballot(live & (_tag_bad(vs[0], tag) | _tag_bad(vs[1], tag) | _tag_bad(vs[2], tag) | _tag_bad(vs[3], tag)))
            n = n + i32(1)
        _poll_report(a, lane, bid, bad != i64(0), ERR_MIDS)
        vals = []
        for q in range_constexpr(4):
            w = i32(vs[q] & i64(0xFFFFFFFF))
            vals += [bf16_lo(w), bf16_hi(w)]
        am = _absf(vals[0])
        for q in range_constexpr(1, 8):
            am = am.maximumf(_absf(vals[q]))
        am = quad_max_pos(am)
        e, inv = mx_scale(am)
        if live:
            lds_st(L, L_MID4 + j * i32(INTER // 2) + lane * i32(4), fp4x8(vals, inv))
            if (lane & i32(3)) == i32(0):
                lds_st(L, L_MSC + j * i32(INTER // MX_BLOCK) + (lane >> i32(2)), fx.Int8(e), align=1)

    @traced
    def mids_all(L, a, lane, wave, s, tag, bid):
        mid_slot(L, a, lane, s, wave * i32(2), tag, bid)
        mid_slot(L, a, lane, s, wave * i32(2) + i32(1), tag, bid)
        gpu.barrier()

    def dn_mfma(L, lane, items):
        g = lane >> i32(4)
        zero = fx.Vector.filled(4, 0.0, fx.Float32)
        for b, j, ws, sd in items:
            acc = zero
            for kb in range_constexpr(DKB):
                b4 = lds_ld(L, L_MID4 + j * i32(INTER // 2) + i32(kb * 64) + g * i32(16), T.vec(4, T.i32), 16)
                sb = lds_u8(L, L_MSC + j * i32(INTER // MX_BLOCK) + i32(kb * 4) + g)
                acc = mfma_fp4(ws[kb], b4, acc, sd, kb, sb, 0)
            _store_red(L, lane, L_P + ((i32(b * SLOTS) + j) * i32(16) + g * i32(4)) * i32(4), acc)

    @traced
    def dn_finish(L, tid):
        if tid < i32(ROWS):
            b = tid >> i32(4)
            r = tid & i32(15)
            acc = f32(0.0)
            for j in range_constexpr(SLOTS):
                acc = acc + lds_f(L, L_P + ((b * i32(SLOTS) + i32(j)) * i32(16) + r) * i32(4)) * lds_f(L, L_TPRB + i32(j * 4))
            lds_st(L, L_PART + tid * i32(2), fx.Int16(bf16_bits(acc)), align=2)
        gpu.barrier()

    # ---------------------------------------------------------- all-reduce
    def peer_base(a, p):
        return i64(bld(rsrc(a["sym"]), p * i32(8), T.i64))

    @traced
    def allreduce(L, a, lane, wave, bid, mype, npes, arflag):
        par = arflag & i32(1)
        if wave < npes - i32(1):
            p = wave + (wave >= mype).select(i32(1), i32(0))
            ms = mype - (mype > p).select(i32(1), i32(0))
            dst = peer_base(a, p) + i64(((ms * i32(2) + par) * i32(GRID) + bid) * i32(WIRE))
            if lane < i32(PKTS):
                d = fx.Vector(lds_ld(L, L_PART + lane * i32(8), T.vec(2, T.i32), 8))
                pkt = fx.Vector.from_elements([i32(d[0]), arflag, i32(d[1]), arflag], fx.Int32)
                bst(pkt, rsrc(dst), lane * i32(16), AUX_SYS)
            src = (
                peer_base(a, mype)
                + i64(((wave * i32(2) + par) * i32(GRID) + bid) * i32(WIRE))
                + i64(fx.min(lane, i32(PKTS - 1)) * i32(16))
            )
            active = lane < i32(PKTS)
            v = ld_sys_v4i(src)
            bad = ballot(active & ((i32(v[1]) != arflag) | (i32(v[3]) != arflag)))
            n = i32(0)
            while (bad != i64(0)) & (n < i32(POLL_LIMIT)):
                v = ld_sys_v4i(src)
                bad = ballot(active & ((i32(v[1]) != arflag) | (i32(v[3]) != arflag)))
                n = n + i32(1)
            _poll_report(a, lane, bid, bad != i64(0), ERR_AR)
            if active:
                lds_st(
                    L,
                    L_RECV + (wave * i32(ROWS) + lane * i32(4)) * i32(2),
                    fx.Vector.from_elements([i32(v[0]), i32(v[2])], fx.Int32),
                    align=8,
                )
        gpu.barrier()

    @traced
    def ar_finish(L, a, tid, s, lb, mype, npes):
        b = tid >> i32(4)
        blk = dn_block(lb, b)
        if (tid < i32(ROWS)) & (blk < i32(DN_BLOCKS)):
            acc = f32(0.0)
            own = bf16_lo(i32(lds_ld(L, L_PART + tid * i32(2), T.i16, 2)) & i32(0xFFFF))
            for p in range_constexpr(MAX_PES):
                slot = i32(p) - (i32(p) > mype).select(i32(1), i32(0))
                sl = fx.min(fx.max(slot, i32(0)), i32(MAX_PES - 2))
                rv = bf16_lo(i32(lds_ld(L, L_RECV + (sl * i32(ROWS) + tid) * i32(2), T.i16, 2)) & i32(0xFFFF))
                v = (i32(p) == mype).select(own, rv)
                acc = (i32(p) < npes).select(acc + v, acc)
            row = s * i32(HIDDEN) + blk * i32(16) + (tid & i32(15))
            if i64(a["residual"]) != i64(0):
                acc = acc + bf16_lo(i32(bld(rsrc(a["residual"]), row * i32(2), T.i16)) & i32(0xFFFF))
            bst(fx.Int16(bf16_bits(acc)), rsrc(a["out"]), row * i32(2))

    @traced
    def _poll_report(a, lane, bid, timed_out, code):
        if timed_out & (lane == i32(0)):
            atomic_or_agent(i64(a["flags"]) + i64((i32(GRID) + bid) * i32(4)), code)

    @traced
    def _mark(tl, tid, bid, k):
        if tid == i32(0):
            t = i64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))
            bst(t, rsrc(tl), (bid * i32(TIMELINE_SLOTS) + i32(k)) * i32(8))

    def mark(tl, tid, bid, k):
        if const_expr(timeline):
            _mark(tl, tid, bid, k)

    @traced
    def epoch_finish(a, tid, bid, use_epoch, ep):
        if use_epoch & (tid == i32(0)):
            st_agent_i32(i64(a["flags"]) + i64(bid * i32(4)), ep + i32(1))

    Shared = fx.struct(type("Shared", (), {"__annotations__": {"buf": fx.Array[fx.Int8, LDS_BYTES, 16]}}))

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def mega_moe_tp_kimi3_kernel(
        x: fx.Int64,
        latent: fx.Int64,
        router_w: fx.Int64,
        bias: fx.Int64,
        ug_w: fx.Int64,
        ug_scales: fx.Int64,
        down_w: fx.Int64,
        down_scales: fx.Int64,
        residual: fx.Int64,
        sym: fx.Int64,
        mype: fx.Int32,
        npes: fx.Int32,
        flag: fx.Int32,
        scores: fx.Int64,
        score_pairs: fx.Int64,
        flags: fx.Int64,
        probs_out: fx.Int64,
        indices_out: fx.Int64,
        hidden_mid: fx.Int64,
        out: fx.Int64,
        mid_pairs: fx.Int64,
        sen_tag: fx.Int32,
        timeline_ptr: fx.Int64,
    ):
        a = {
            "x": x,
            "latent": latent,
            "router_w": router_w,
            "bias": bias,
            "ug_w": ug_w,
            "ug_scales": ug_scales,
            "down_w": down_w,
            "down_scales": down_scales,
            "residual": residual,
            "sym": sym,
            "scores": scores,
            "score_pairs": score_pairs,
            "flags": flags,
            "probs_out": probs_out,
            "indices_out": indices_out,
            "hidden_mid": hidden_mid,
            "out": out,
            "mid_pairs": mid_pairs,
        }
        tid = i32(gpu.thread_id("x"))
        bid = i32(gpu.block_id("x"))
        lane = tid % i32(64)
        wave = uni(tid // i32(64))
        s = bid // i32(BPS)
        lb = bid % i32(BPS)
        lds = fx.SharedAllocator().allocate(Shared).peek()
        L = uni(i32(fx.ptrtoint(lds.buf.ptr)))

        use_epoch = sen_tag == i32(0)
        ep = ld_agent_i32(i64(flags) + i64(bid * i32(4)))
        tag = use_epoch.select(ep + i32(1), sen_tag)
        arflag = use_epoch.select(ep + i32(1), flag)
        tl = timeline_ptr
        mark(tl, tid, bid, 0)

        _topk_clear(L, wave, lane)
        accs = router_dot(a, wave, lane, bid)
        router_publish(L, a, tid, wave, lane, bid, accs, tag)
        mark(tl, tid, bid, 1)
        act_quant(L, a, wave, lane, s)
        gpu.barrier()
        mark(tl, tid, bid, 2)

        topk(L, a, lane, wave, s, lb, tag, bid)
        up_jobs(L, a, wave, lane, lb)
        gpu.barrier()
        mark(tl, tid, bid, 3)
        up_publish(L, a, wave, lane, s, lb, tag)
        dn_items = dn_issue(L, a, wave, lane, lb)
        rocdl.sched_barrier(0)
        mark(tl, tid, bid, 4)

        mids_all(L, a, lane, wave, s, tag, bid)
        mark(tl, tid, bid, 5)
        dn_mfma(L, lane, dn_items)
        gpu.barrier()
        dn_finish(L, tid)
        mark(tl, tid, bid, 6)

        allreduce(L, a, lane, wave, bid, mype, npes, arflag)
        mark(tl, tid, bid, 7)
        ar_finish(L, a, tid, s, lb, mype, npes)
        epoch_finish(a, tid, bid, use_epoch, ep)
        mark(tl, tid, bid, 8)

    @flyc.jit
    def launch(
        x: fx.Int64,
        latent: fx.Int64,
        router_w: fx.Int64,
        bias: fx.Int64,
        ug_w: fx.Int64,
        ug_scales: fx.Int64,
        down_w: fx.Int64,
        down_scales: fx.Int64,
        residual: fx.Int64,
        sym: fx.Int64,
        mype: fx.Int32,
        npes: fx.Int32,
        flag: fx.Int32,
        scores: fx.Int64,
        score_pairs: fx.Int64,
        flags: fx.Int64,
        probs_out: fx.Int64,
        indices_out: fx.Int64,
        hidden_mid: fx.Int64,
        out: fx.Int64,
        mid_pairs: fx.Int64,
        sen_tag: fx.Int32,
        timeline_ptr: fx.Int64,
        stream: fx.Stream,
    ):
        mega_moe_tp_kimi3_kernel(
            x,
            latent,
            router_w,
            bias,
            ug_w,
            ug_scales,
            down_w,
            down_scales,
            residual,
            sym,
            mype,
            npes,
            flag,
            scores,
            score_pairs,
            flags,
            probs_out,
            indices_out,
            hidden_mid,
            out,
            mid_pairs,
            sen_tag,
            timeline_ptr,
        ).launch(grid=(GRID, 1, 1), block=(NT, 1, 1), stream=stream)

    launch.consts = c
    return launch
