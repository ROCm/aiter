# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decode-shape (<=128 rows, E128 top4) in-kernel routing: one workgroup's canonical 16-row
expert block straight from ``topk_ids``, with no sort launch and no global atomics.

Ported from amd-kernels' ``moe/mxfp4_h6144_i768_e128_top4/routing_ids.py`` (Apache-2.0,
"Copyright contributors to the amd-kernels project") and the ``wave_incl_scan_i32`` DPP wave
scan from its sibling ``common/utils.py``, reimplemented here against aiter's own
``aiter.ops.flydsl.kernels.buffer_ops`` instead of amd-kernels' private copy of the same
buffer-op wrappers. The algorithm (LDS-atomic histogram of the M*4 routing pairs, canonical
block order by ``ceil(count / 16)`` per expert, row assignment by ballot in pair order) is
unchanged from the amd-kernels source; this is a relocation of the technique into aiter's own
kernel-helper namespace, not a redesign.

This is a technique landing spot for aiter's mega_moe / mega_moe_tp family
(``aiter/ops/flydsl/kernels/mega_moe_tp/``): those kernels already carry the epoch-flag /
XCD-affinity / loader-wave LDS-DMA primitives a fused decode GEMM1->GEMM2 kernel needs, but
only at 128/256-row prefill tiles; this module supplies the small-M (8-128 row) in-kernel
routing primitive they are missing, generalized over the pair/row count only (no
MiniMax-M3.1-specific constant besides the E128/top4/16-row-block shape the call site fixes).
"""
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

E, TOPK, MAX_PAIRS = 128, 4, 256
HIST = 256  # 128 experts, 64 per-lane dummy slots; 64 lanes x 16 B zeroed in one store
TAB = 80  # 16 rows, 64 per-lane dummy slots
WAVE_BYTES = (HIST + TAB) * 4  # one wave's LDS slice

_WAVE_SCAN_DPP = (
    (0x111, 0xF, True),
    (0x112, 0xF, True),
    (0x114, 0xF, True),
    (0x118, 0xF, True),
    (0x142, 0xA, False),
    (0x143, 0xC, False),
)


def wave_incl_scan_i32(x):
    """Inclusive prefix sum of an i32 over the 64 lanes of a wave (DPP, no LDS).

    Ported from amd-kernels' ``common/utils.py``.
    """
    zero = _raw(fx.Int32(0))
    for ctrl, row_mask, bound in _WAVE_SCAN_DPP:
        t = _rocdl_ops.update_dpp(T.i32, zero, _raw(x), ctrl, row_mask, 0xF, bound)
        x = x + fx.Int32(t)
    return x


def _raw(x):
    return x.ir_value() if hasattr(x, "ir_value") else x


def _ptr(addr):
    """``!llvm.ptr<3>`` at LDS address ``addr``: the raw LDS atomic adds, slot stores and loads
    below take it."""
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(addr)))


def wave_expert(ids, m, block, lane, hb, pairs=MAX_PAIRS):
    """(expert of canonical 16-row block ``block``, its first rank, number of blocks, the
    lane's pair ids).

    Args:
        hb: the wave's ``WAVE_BYTES`` LDS slice.
        pairs: the pair slots scanned, a compile-time multiple of 64, >= M * 4.
    """
    n_pairs = m * TOPK
    ir_ = buffer_ops.create_buffer_resource(ids, max_size=False, num_records_bytes=n_pairs * 4)
    vals = []
    for c in range_constexpr(pairs // 64):
        q = lane + c * 64
        v = fx.Int32(buffer_ops.buffer_load(ir_, q, vec_width=1, dtype=fx.Int32))
        vals.append((q < n_pairs).select(v, fx.Int32(-1)))
    llvm.StoreOp(_raw(Vec.from_elements([fx.Int32(0)] * 4, fx.Int32)), _ptr(hb + lane * 16), alignment=16)
    llvm.StoreOp(_raw(fx.Int32(m)), _ptr(hb + HIST * 4 + lane * 4), alignment=4)
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    for c in range_constexpr(len(vals)):
        slot = (vals[c] >= 0).select(vals[c], E + lane)
        llvm.AtomicRMWOp(llvm.AtomicBinOp.add, _ptr(hb + slot * 4), _raw(fx.Int32(1)),
                         llvm.AtomicOrdering.monotonic, syncscope="workgroup", alignment=4)
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    cc = Vec(llvm.load(T.vec(2, T.i32), _ptr(hb + lane * 8), alignment=8))
    c0, c1 = fx.Int32(cc[0]), fx.Int32(cc[1])
    # the canonical block order on the counts
    b0, b1 = (c0 + 15) >> 4, (c1 + 15) >> 4
    end = wave_incl_scan_i32(b0 + b1)
    start = end - b0 - b1
    total = fx.Int32(fx.rocdl.readlane(T.i32, end, fx.Int32(63)))
    has = (block >= start) & (block < end)
    mask = fx.Int64(fx.rocdl.ballot(T.i64, has))
    winner = (mask != fx.Int64(0)).select(fx.math.cttz(mask).to(fx.Int32), fx.Int32(0))
    second = block >= start + b0
    local_e = lane * 2 + second.select(fx.Int32(1), fx.Int32(0))
    local_s = second.select(start + b0, start)
    expert = fx.Int32(fx.rocdl.readlane(T.i32, local_e, winner))
    origin = fx.Int32(fx.rocdl.readlane(T.i32, local_s, winner))
    return expert, (block - origin) * 16, total, vals


def wave_block_rows(vals, expert, first, lane, hb):
    """This lane's row ``lane % 16`` of the block (token | slot << 24, token M = padding): the
    expert's pairs of rank ``first`` .. ``first + 15`` in pair order, by ballot."""
    run = fx.Int32(0)
    for c in range_constexpr(len(vals)):
        q = lane + c * 64
        match = vals[c] == expert
        bal = rocdl.ballot(T.i64, _raw(match))
        lo = arith.trunci(T.i32, bal)
        hi = arith.trunci(T.i32, arith.shrui(bal, arith.constant(32, type=T.i64)))
        below = fx.Int32(rocdl.mbcnt_hi(T.i32, hi, rocdl.mbcnt_lo(T.i32, lo, _raw(fx.Int32(0)))))
        r = run + below - first
        slot = (match & (r >= 0) & (r < 16)).select(r, 16 + lane)
        llvm.StoreOp(_raw((q >> 2) | ((q & 3) << 24)), _ptr(hb + HIST * 4 + slot * 4), alignment=4)
        tot = run + below + match.select(fx.Int32(1), fx.Int32(0))
        run = fx.Int32(fx.rocdl.readlane(T.i32, tot, fx.Int32(63)))
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    return fx.Int32(llvm.load(T.i32, _ptr(hb + HIST * 4 + (lane % 16) * 4), alignment=4))
