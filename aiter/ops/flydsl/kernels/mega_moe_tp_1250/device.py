# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device helpers of the gfx1250 MegaMoE TP kernel: raw-address global access, LDS
words, grid / cross-rank synchronization and the MXFP4 row quantizer.

Memory ordering uses the LLVM AMDGPU memory model (scoped fences, atomics): agent
scope between the CTAs of one GPU (its XCDs do not share an L2), system scope
("one-as") for the peers' symmetric memory reached over xGMI."""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ..quant_utils import emit_mx_e8m0_scale
from .config import WAVE as WAVE_W
from .config import DEADLINE, LINE, SYNC_FLAG, SYNC_GS, SYNC_TOP

MONO = fx.AtomicOrdering.Monotonic
AGENT = fx.rocdl.SyncScope.Agent
SYS = fx.rocdl.SyncScope.OneAs


# Device-scope release (global_wb SCOPE_DEV: a whole-L2 writeback) at the grid syncs
# and the per-m-tile signals is off by default: the stale-data stress (scripts/
# stress_mm.py: alternating inputs, back-to-back launches) passes without it on
# gfx1250 and it costs ~10 us at small M. MEGA_DEV_WB=1 restores it. Acquires
# (global_inv) and system-scope releases stay.
# perf experiments (MEGA_EXP="a+b"): nosys: no system-scope writeback; noacq: no
# device-scope invalidate (both currently hit an assembler error)
_EXP = set(__import__("os").environ.get("MEGA_EXP", "").split("+"))
if __import__("os").environ.get("MEGA_DEV_WB", "0") != "1":
    _EXP.add("nofence")
_EXP_NOFENCE = "nofence" in _EXP


def i32(v):
    return fx.Int32(v)


def i64(v):
    return fx.Int64(v)


def uni(v):
    """Wave-uniform copy of an i32 (readfirstlane)."""
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def rdlane(v, j):
    """Lane j's copy of an i32 (readlane; j an int or a wave-uniform i32)."""
    return fx.Int32(rocdl.readlane(T.i32, fx.Int32(v), j))


def uni64(v):
    v = fx.Int64(v)
    lo = uni(fx.Int32(v & fx.Int64(0xFFFFFFFF)))
    hi = uni(fx.Int32(v >> 32))
    return (fx.Int64(hi) << 32) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))


def gptr(addr, dtype=fx.Int32, align=4):
    ty = fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, align)
    return fx.inttoptr(ty, fx.Int64(addr))


def ld_i32(addr):
    return fx.Int32(fx.generic_load(gptr(addr)))


def ld_f32(addr):
    return fx.Float32(fx.generic_load(gptr(addr, fx.Float32)))


def st_i32(addr, v):
    fx.generic_store(gptr(addr), fx.Int32(v))


def st_i8(addr, v):
    fx.generic_store(gptr(addr, fx.Int8, 1), fx.Int8(v))


def ld_v2(addr):
    """8 B global load (2 x i32)."""
    return fx.Vector(fx.generic_load(gptr(addr, fx.Int32, 8), count=2))


def st_v2(addr, v):
    fx.generic_store(gptr(addr, fx.Int32, 8), fx.Vector(v))


def ld_u8(addr):
    return fx.Int32(fx.generic_load(gptr(addr, fx.Int8, 1))) & i32(0xFF)


def ld_v4(addr):
    """16 B global load (4 x i32)."""
    return fx.Vector(fx.generic_load(gptr(addr, fx.Int32, 16), count=4))


def st_v4(addr, v):
    fx.generic_store(gptr(addr, fx.Int32, 16), fx.Vector(v))


def ld_agent(addr):
    return fx.Int32(
        fx.generic_load(gptr(addr), memory_order=MONO, syncscope=AGENT, volatile=True)
    )


def ld_sys(addr):
    return fx.Int32(
        fx.generic_load(gptr(addr), memory_order=MONO, syncscope=SYS, volatile=True)
    )


def st_sys_rel(addr, v):
    fx.generic_store(
        gptr(addr), fx.Int32(v), memory_order=fx.AtomicOrdering.Release, syncscope=SYS
    )


def st_agent(addr, v):
    """Agent-scope (write-through) store: no L2-resident copy outlives the launch."""
    fx.generic_store(gptr(addr), fx.Int32(v), memory_order=MONO, syncscope=AGENT)


def st_agent_rel(addr, v):
    fx.generic_store(
        gptr(addr), fx.Int32(v), memory_order=fx.AtomicOrdering.Release, syncscope=AGENT
    )


def add_sys(addr, v):
    """System-scope atomic add (a peer's memory over xGMI)."""
    return fx.Int32(fx.atomic_add(gptr(addr), fx.Int32(v), syncscope=SYS))


def st_sys(addr, v):
    fx.generic_store(gptr(addr), fx.Int32(v), memory_order=MONO, syncscope=SYS)


def add_agent(addr, v):
    return fx.Int32(fx.atomic_add(gptr(addr), fx.Int32(v), syncscope=AGENT))


def fence(order, scope):
    fx.memory_fence(ordering=order, syncscope=scope)


def rel_agent():
    if "nofence" not in _EXP:
        fence(fx.AtomicOrdering.Release, AGENT)


def acq_agent():
    if "noacq" not in _EXP:
        fence(fx.AtomicOrdering.Acquire, AGENT)


def rel_sys():
    if "nosys" not in _EXP:
        fence(fx.AtomicOrdering.Release, SYS)


def acq_sys():
    fence(fx.AtomicOrdering.Acquire, SYS)


# Device / system scope fences write back / invalidate the L2 (global_wb / global_inv),
# which serializes across the ~1000 CTAs: a CTA meets at workgroup scope (all waves'
# accesses complete, s_barrier) and only thread 0 runs the wide fence (cumulative).
WG = fx.rocdl.SyncScope.Workgroup


def rel_wg():
    fence(fx.AtomicOrdering.Release, WG)


def acq_wg():
    fence(fx.AtomicOrdering.Acquire, WG)


# ---- LDS words (one byte arena, addressed by i32 byte offsets) ----
def _lp(addr, dtype=fx.Int32, align=4):
    ty = fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Shared, align)
    return fx.inttoptr(ty, fx.Int32(addr))


def lds_ld(addr):
    return fx.Int32(fx.ptr_load(_lp(addr)))


def lds_st(addr, v):
    fx.ptr_store(fx.Int32(v), _lp(addr))


def lds_add(addr, v):
    return fx.Int32(fx.atomic_add(_lp(addr), fx.Int32(v), syncscope="workgroup"))


# ---- time / waits ----
def now():
    from flydsl._mlir.dialects import llvm as _llvm

    return fx.Int64(
        _llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], [])
    )


def before(cur, target):
    """cur has not reached target (serial-number order, wrap-safe for int32)."""
    return (fx.Int32(cur) - fx.Int32(target)) < i32(0)


@flyc.jit
def spin_agent(addr, target, err_addr, err_bit):
    """Wait until the agent-visible counter at addr reaches target (wrap-safe); on a
    timeout set err_bit at err_addr and go on. The clock (s_sendmsg_rtn) is read
    only every 1024 polls: ~1000 spinning CTAs reading it each poll starve a CTA
    that reads it while working (stalls until the spinners time out)."""
    cur = ld_agent(addr)
    t0 = now()
    n = i32(0)
    alive = fx.Boolean(True)
    while before(cur, target) & alive:
        rocdl.s_sleep(1)
        cur = ld_agent(addr)
        n = n + i32(1)
        if (n & i32(1023)) == i32(0):
            alive = (now() - t0) < fx.Int64(DEADLINE)
    if before(cur, target):
        fx.atomic_or(gptr(err_addr), fx.Int32(err_bit), syncscope=AGENT)


@flyc.jit
def spin_sys(addr, target, err_addr, err_bit):
    """spin_agent on a flag a peer writes over xGMI (system scope)."""
    cur = ld_sys(addr)
    t0 = now()
    n = i32(0)
    alive = fx.Boolean(True)
    while before(cur, target) & alive:
        rocdl.s_sleep(1)
        cur = ld_sys(addr)
        n = n + i32(1)
        if (n & i32(1023)) == i32(0):
            alive = (now() - t0) < fx.Int64(DEADLINE)
    if before(cur, target):
        fx.atomic_or(gptr(err_addr), fx.Int32(err_bit), syncscope=AGENT)


@flyc.jit
def sync_arrive(sync, bid, ep, n_cta):
    """Thread 0 of a CTA arrives at a grid sync point (config SYNC_*) once in launch
    ep; returns 1 in the grid's last arriver (after an agent-scope acquire of every
    arrival), else 0. The caller releases its CTA's writes first (a global_wb: they
    are in memory before its arrival), so a group's last arriver forwards to the top
    counter without fences of its own."""
    gi = bid // i32(SYNC_GS)
    ng = (n_cta + SYNC_GS - 1) // SYNC_GS
    gs = fx.min(i32(n_cta) - gi * i32(SYNC_GS), i32(SYNC_GS))  # the last group may be short
    last = i32(0)
    old = add_agent(sync + i64(gi) * (LINE * 4), 1)
    if old + i32(1) == gs:
        top = add_agent(sync + SYNC_TOP * 4, 1)
        last = (top + i32(1) == i32(ng)).select(i32(1), i32(0))
    if last == i32(1):
        acq_agent()
    return last


@flyc.jit
def grid_barrier(tid, bid, sync, ep, n_cta, err_addr, err_bit):
    """All CTAs of the grid (co-resident) meet at a sync point: groups' last arrivers
    count at the top counter, every CTA polls it. Agent-scope release / acquire."""
    rel_wg()
    gpu.barrier()
    if tid == i32(0):
        if fx.const_expr(not _EXP_NOFENCE):
            rel_agent()
        ng = (n_cta + SYNC_GS - 1) // SYNC_GS
        old = add_agent(sync + i64(bid // i32(SYNC_GS)) * (LINE * 4), 1)
        gs = fx.min(i32(n_cta) - (bid // i32(SYNC_GS)) * i32(SYNC_GS), i32(SYNC_GS))
        if old + i32(1) == gs:
            add_agent(sync + SYNC_TOP * 4, 1)
        spin_agent(sync + SYNC_TOP * 4, i32(ng), err_addr, err_bit)
        acq_agent()
    gpu.barrier()
    acq_wg()


@flyc.jit
def stamp(tid, addr):
    """Debug: thread 0 stores the steady counter (int64) at addr."""
    if tid == i32(0):
        t = now()
        # write-through: a plain store can still sit in an XCD's L2 when the host
        # copies the stamps (rows of an older launch read back)
        st_sys(addr, fx.Int32(t & fx.Int64(0xFFFFFFFF)))
        st_sys(addr + 4, fx.Int32(t >> 32))


# ---- int8 + e8m0 (32-element blocks over 4 lanes x 8 values) ----
def q8_block(vals):
    """A lane's f32s of a 32-element block shared by 32 / len(vals) aligned lanes ->
    (len(vals) / 4 x i32 of int8, the block's e8m0 as i32). Scale: the power of 2
    >= amax / 127."""
    am = abs(fx.Float32(vals[0]))
    for v in vals[1:]:
        am = fx.maxnumf(am, abs(fx.Float32(v)))
    lanes = 32 // len(vals)
    d = 1
    while d < lanes:
        am = fx.maxnumf(am, am.shuffle_xor(i32(d), i32(WAVE_W)))
        d *= 2
    bits = (am * fx.Float32(1.0 / 127.0)).bitcast(fx.Int32)
    e8 = ((bits + i32(0x7FFFFF)) >> i32(23)) & i32(0xFF)
    inv = ((i32(254) - e8) << i32(23)).bitcast(fx.Float32)
    words = []
    for h in range_constexpr(len(vals) // 4):
        w = i32(0)
        for k in range_constexpr(4):
            q = fx.Int32(fmath.roundeven(fx.Float32(vals[4 * h + k]) * inv))
            q = fx.max(fx.min(q, i32(127)), i32(-127))
            w = w | ((q & i32(0xFF)) << i32(8 * k))
        words.append(w)
    return words, e8


def q8_unpack(vw, e8, n):
    """n x i32 of int8 and their e8m0 (i32) -> 4 n f32."""
    sc = ((e8 & i32(0xFF)) << i32(23)).bitcast(fx.Float32)
    out = []
    for h in range_constexpr(n):
        d = fx.Int32(vw[h])
        for k in range_constexpr(4):
            b = (d << i32(24 - 8 * k)) >> i32(24)
            out.append(b.to(fx.Float32) * sc)
    return out


# ---- MXFP4 quantization ----
def bf16x8_f32(v4):
    """A 16 B load (8 bf16 in 4 x i32) -> 8 f32."""
    out = []
    for q in range_constexpr(4):
        w = fx.Int32(v4[q])
        out.append((w << i32(16)).bitcast(fx.Float32))
        out.append((w & i32(-65536)).bitcast(fx.Float32))
    return out


def quant_fp4_block32(v4s):
    """One 32-element MX block, held as four 16 B bf16 loads -> (16 B of packed FP4
    as 4 x i32, the e8m0 byte as i32). Scale RoundUp (torch per_1x32_f4_quant)."""
    vals = []
    for v in v4s:
        vals += bf16x8_f32(v)
    level = [abs(fx.Float32(v)) for v in vals]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 3):
            acc = level[i]
            for v in level[i + 1 : i + 3]:
                acc = fx.maxnumf(acc, v)
            nxt.append(acc)
        level = nxt
    amax = fx.minnumf(level[0], fx.Float32(3.4028234663852886e38))
    e8 = fx.Int32(emit_mx_e8m0_scale(amax))
    scale = (e8 << i32(23)).bitcast(fx.Float32)
    words = []
    for v in v4s:
        bf = fx.Vector(v).bitcast(fx.BFloat16)
        words.append(
            fx.Int32(
                rocdl.cvt_scalef32_pk8_fp4_bf16(
                    T.i32, bf.ir_value(), scale.ir_value()
                )
            )
        )
    return fx.Vector.from_elements(words, fx.Int32), e8
