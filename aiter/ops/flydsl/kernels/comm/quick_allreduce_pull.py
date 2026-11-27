# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project

"""The ``pull`` all-reduce schedule: TP4 all-reduce + residual add + Gemma RMSNorm on registered inputs.

Unlike the quantized ``mesh`` and ``ring`` schedules, nothing is packed or pushed. The wire is bf16 rows read in
place: the inputs of the ranks are registered once per CUDA graph (see ``PullAllReduceRMSNorm`` in
``aiter.ops.flydsl.quick_allreduce_pull``), a reducer workgroup reads the peers' rows of its chunk directly, sums the
four partials in fp32 in the fixed order of AITER's two-stage all-reduce (chunk owner first, denormals flushed, one
bf16 rounding), publishes the reduced row, and the other three ranks pull it. The norm then runs in the lane map and
reduction order of ``gemma_add_rmsnorm`` (Triton's ``tl.sum`` order): ``y = s * rstd * (w + 1)``.

Three kernels cover the decode row counts, each with one workgroup of 12 waves per row (H = 6144):

* ``make_oneshot_ack_kernel``: rows 1-11, every rank reads all four inputs and acknowledges the reads.
* ``make_pull_kernel``: rows 12-287, one row per workgroup, the two-stage reduce / pull exchange.
* ``make_stream_pull_kernel``: rows 288-640, R rows per workgroup, batched handshakes, optional weight hoist.

The algorithm, the flag protocols, the thresholds and the arithmetic come from amd-kernels
(``amd_kernels/norm/gemma_add_rmsnorm_h6144/allreduce_add_rmsnorm``: ``ar_add_rmsnorm_oneshot_pull_ack``,
``ar_add_rmsnorm_pull``, ``ar_add_rmsnorm_stream_pull``, ``common`` and ``add_rmsnorm/common``), re-expressed on
aiter's own ``buffer_ops``. The flag buffer layout here is aiter's own: one uncached buffer per rank holds the
three domains side by side.

Raw ops that stay, and why (no stable ``fx`` API covers them): the system-scope ``atomicrmw`` flag stores and polls
(``_flag_store``/``_flag_read``: write-through peer flags, a poll that is never hoisted), the inline asm that scopes the
hardware FP32 flush-denormals mode to the rank sum (``_enter_fp32_ftz``/``_restore_fp32_denorm_mode``) and ``update_dpp``
for Triton's ``tl.sum`` wave reduction order, which bit-exactness with the unfused norm needs. Loads and stores use the
legacy ``buffer_ops`` of the sibling mesh and ring kernels (peer rows are addressed by raw IPC pointers).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .. import buffer_ops

WORLD = 4
H = 6144
NW = H // 8 // 64  # 12 waves of 8 columns per lane
NT = 64 * NW  # 768 threads: thread t holds columns 8 t .. 8 t + 7
ROWB = H * 2
SYS = 0x11  # sc0 | sc1: system-coherent loads, write-through stores
MAX_ROWS = 1024  # rows of the flag tables (the call-count arrays are int32[MAX_ROWS])
MAX_Q = MAX_ROWS // WORLD

# kernel arguments preloaded into SGPRs (as Triton does), not s_load'ed first
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}

# --- per-rank buffer layout (bytes): three flag domains, each with its own call counts ---------------------------
# one-shot with acks: a 128-byte line per (row, wave), word 4 + s written by rank s
ACK_ROWS = 32
ACK_OFF = 0
ACK_BYTES = ACK_ROWS * NW * 128
# two-stage pull: ARR / RED flag words (LINE words per row), then the reduced rows of this rank's chunk
PULL_LINE = 64  # 12 waves x 4 ranks (+ pad)
PULL_RED = MAX_ROWS * PULL_LINE  # flag word PULL_RED + b * PULL_LINE + wave
PULL_RED_OFF = 2 * MAX_ROWS * PULL_LINE * 4
PULL_BYTES = PULL_RED_OFF + MAX_Q * ROWB
PULL_OFF = (ACK_OFF + ACK_BYTES + 4095) // 4096 * 4096
# streamed pull: a 128-byte flag line per row
STREAM_LINE = 32
STREAM_RED = MAX_ROWS * STREAM_LINE
STREAM_RED_OFF = 2 * MAX_ROWS * STREAM_LINE * 4
STREAM_BYTES = STREAM_RED_OFF + MAX_Q * ROWB
STREAM_OFF = (PULL_OFF + PULL_BYTES + 4095) // 4096 * 4096
BUF_BYTES = STREAM_OFF + STREAM_BYTES


# --- peer flag primitives ----------------------------------------------------------------------------------------
def _gptr(addr):
    return fx.inttoptr(
        fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4),
        fx.Int64(addr),
    )


def _flag_store(addr, v):
    """An atomic exchange at system scope (write-through)."""
    _llvm.AtomicRMWOp(
        _llvm.AtomicBinOp.xchg,
        _gptr(addr).llvm_ptr,
        fx.Int32(v).ir_value(),
        _llvm.AtomicOrdering.monotonic,
        syncscope="",
        alignment=4,
    )


def _flag_read(addr):
    """An atomic add of 0 at system scope (a spin on it is never hoisted)."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _gptr(addr).llvm_ptr,
            fx.Int32(0).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="",
            alignment=4,
        ).result
    )


@flyc.jit
def _spin(a, e):
    f = _flag_read(a)
    while (f - e) < 0:  # signed difference: the counts may wrap
        f = _flag_read(a)


def sel4(p, a0, a1, a2, a3):
    return (p == 0).select(a0, (p == 1).select(a1, (p == 2).select(a2, a3)))


# streamed domain: ARR (b, s) on the reducer = the puller on rank s has started; RED (b) on a puller = published
@flyc.jit
def _arrive(b0, b1, b2, b3, c, me, b, e, puller):
    """A puller's lane 0 stores e into ARR (b, me) on the reducer c."""
    if puller & (fx.thread_idx.x == 0):
        _flag_store(sel4(c, b0, b1, b2, b3) + fx.Int64((b * STREAM_LINE + me) * 4), e)


@flyc.jit
def _wait_pullers(mine, me, b, e, reducer):
    """The reducer's lanes 0-2 spin until ARR (b, s) of the 3 pullers reach e, then the barrier. A poll waits for the
    wave's outstanding vector loads (vmcnt): issue none before it."""
    lane = fx.thread_idx.x
    if reducer & (lane < WORLD - 1):
        _spin(mine + fx.Int64((b * STREAM_LINE + (me + 1 + lane) % WORLD) * 4), e)
    fx.gpu.barrier()


@flyc.jit
def _publish(b0, b1, b2, b3, me, b, e, reducer):
    """The reducer's lanes 0-2 store e into RED (b) on the 3 pullers."""
    lane = fx.thread_idx.x
    if reducer & (lane < WORLD - 1):
        p = (me + 1 + lane) % WORLD
        _flag_store(
            sel4(p, b0, b1, b2, b3) + fx.Int64((STREAM_RED + b * STREAM_LINE) * 4), e
        )


@flyc.jit
def _wait_reduced(mine, b, e):
    """A puller's lane 0 spins until RED (b) reaches e; then the barrier."""
    if fx.thread_idx.x == 0:
        _spin(mine + fx.Int64((STREAM_RED + b * STREAM_LINE) * 4), e)
    fx.gpu.barrier()


# pull domain: per-wave flags (line of PULL_LINE words per row; word wave * 4 + rank)
@flyc.jit
def _pull_arrive(b0, b1, b2, b3, c, me, b, v, e, puller):
    if puller & ((fx.thread_idx.x % 64) == 0):
        _flag_store(
            sel4(c, b0, b1, b2, b3) + fx.Int64((b * PULL_LINE + v * WORLD + me) * 4), e
        )


@flyc.jit
def _pull_wait_pullers(mine, me, b, v, e, reducer):
    lane = fx.thread_idx.x % 64
    if reducer & (lane < WORLD - 1):
        _spin(
            mine + fx.Int64((b * PULL_LINE + v * WORLD + (me + 1 + lane) % WORLD) * 4),
            e,
        )


@flyc.jit
def _pull_publish(b0, b1, b2, b3, me, b, v, e, reducer):
    lane = fx.thread_idx.x % 64
    if reducer & (lane < WORLD - 1):
        _flag_store(
            sel4((me + 1 + lane) % WORLD, b0, b1, b2, b3)
            + fx.Int64((PULL_RED + b * PULL_LINE + v) * 4),
            e,
        )


@flyc.jit
def _pull_wait_reduced(mine, b, v, e):
    if fx.thread_idx.x % 64 == 0:
        _spin(mine + fx.Int64((PULL_RED + b * PULL_LINE + v) * 4), e)


# --- registered inputs and the rank sum --------------------------------------------------------------------------
def _load_ptrs(arg_xs):
    """The 4 ranks' input addresses from the int64[4] at arg_xs (scalar loads)."""
    tr = buffer_ops.create_buffer_resource_from_addr(arg_xs, num_records_bytes=32)
    out = []
    for h in range_constexpr(2):
        v = Vec(buffer_ops.buffer_load(tr, h * 4, vec_width=4, is_scalar=True))
        for i in range_constexpr(2):
            lo = fx.Int64(fx.Int32(v[2 * i])) & fx.Int64(0xFFFFFFFF)
            out.append(lo | (fx.Int64(fx.Int32(v[2 * i + 1])) << 32))
    return out


def _unrotate(rot, vs):
    """The 3 packs loaded in rotated order (slot t = peer 1 + (t + rot) % 3) back in peer order 1, 2, 3."""
    out = []
    for i in range_constexpr(3):
        a, b, c = vs[i], vs[(i + 2) % 3], vs[(i + 1) % 3]
        out.append(
            Vec.from_elements(
                [
                    (rot == 0).select(
                        fx.Int32(a[d]),
                        (rot == 1).select(fx.Int32(b[d]), fx.Int32(c[d])),
                    )
                    for d in range_constexpr(4)
                ],
                fx.Int32,
            )
        )
    return out


def _as_f32(i):
    return fx.Int32(i).bitcast(fx.Float32)


def _ftz(u):
    """fp32 bits with a denormal flushed to a zero of its sign."""
    return ((u & 0x7F800000) == 0).select(u & fx.Int32(-0x80000000), u)


def _add_ftz(a, b):
    return _ftz((_as_f32(a) + _as_f32(_ftz(b))).bitcast(fx.Int32))


def _sum4(packs):
    """bf16(((p0 + p1) + p2) + p3) per element of four 16 B packs in fp32, with AITER's denormal flush
    (-fgpu-flush-denormals-to-zero: operands and sums), done with integer masks."""
    out = []
    for i in range_constexpr(4):
        ws = [fx.Int32(p[i]) for p in packs]
        lo = _ftz(ws[0] << 16)
        hi = _ftz(ws[0] & fx.Int32(-65536))
        for r in range_constexpr(1, WORLD):
            lo = _add_ftz(lo, ws[r] << 16)
            hi = _add_ftz(hi, ws[r] & fx.Int32(-65536))
        out.append(_pk_bf16(_as_f32(lo), _as_f32(hi)))
    return Vec.from_elements(out, fx.Int32)


def _enter_fp32_ftz(packs):
    """Save the wave's FP32 denormal mode and flush denormals (operands and sums) until the restore.

    Every source pack crosses this opaque identity asm, so LLVM cannot move a rank-sum op above the mode change.
    """
    n = len(packs)
    ty = ir.Type.parse(
        "!llvm.struct<(" + ", ".join(["vector<4xi32>"] * n + ["i32"]) + ")>"
    )
    asm = (
        f"s_getreg_b32 ${n}, hwreg(HW_REG_MODE,4,2)\n"
        "s_setreg_imm32_b32 hwreg(HW_REG_MODE,4,2), 0"
    )
    constraints = ",".join(["=v"] * n + ["=s"] + [str(i) for i in range(n)])
    result = _llvm.inline_asm(
        ty, [v.ir_value() for v in packs], asm, constraints, has_side_effects=True
    )
    values = [Vec(_llvm.extractvalue(T.vec(4, T.i32), result, [i])) for i in range(n)]
    return values, fx.Int32(_llvm.extractvalue(T.i32, result, [n]))


def _restore_fp32_denorm_mode(packs, mode):
    """Restore the saved mode after all rank sums: the sums cross this identity asm, so the residual add and the norm
    run in the restored mode."""
    n = len(packs)
    ty = ir.Type.parse("!llvm.struct<(" + ", ".join(["vector<4xi32>"] * n) + ")>")
    constraints = ",".join(["=v"] * n + ["s"] + [str(i) for i in range(n)])
    result = _llvm.inline_asm(
        ty,
        [mode.ir_value()] + [v.ir_value() for v in packs],
        f"s_setreg_b32 hwreg(HW_REG_MODE,4,2), ${n}",
        constraints,
        has_side_effects=True,
    )
    return [Vec(_llvm.extractvalue(T.vec(4, T.i32), result, [i])) for i in range(n)]


def _sum4_hardware_ftz(packs):
    """The fixed-order BF16 rank sum; hardware FP32 FTZ is active only here."""
    out = []
    for i in range_constexpr(4):
        ws = [fx.Int32(p[i]) for p in packs]
        lo = (ws[0] << 16).bitcast(fx.Float32)
        hi = (ws[0] & fx.Int32(-65536)).bitcast(fx.Float32)
        for r in range_constexpr(1, WORLD):
            lo = lo + (ws[r] << 16).bitcast(fx.Float32)
            hi = hi + (ws[r] & fx.Int32(-65536)).bitcast(fx.Float32)
        out.append(_pk_bf16(lo, hi))
    return Vec.from_elements(out, fx.Int32)


# --- the Gemma RMSNorm arithmetic, in Triton's tl.sum reduction order (gfx950, wave64) ---------------------------
def _bf16x8(v):
    """8 fp32 of 4 dwords of bf16 (low half first)."""
    out = []
    for i in range(4):
        w = fx.Int32(v[i])
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def _pk_bf16(a, b):
    """Two fp32 -> two bf16 in a dword (v_cvt_pk_bf16_f32, RNE), ``a`` low."""
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(
            fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))
        )
    )


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def _dpp_f(v, ctrl, row_mask=0xF, old=None):
    """v_mov_b32_dpp of an fp32; source lanes outside the row read 0 (bound_ctrl), rows ``row_mask`` leaves out keep
    ``old``."""
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(_bits(old))
    r = _rocdl_ops.update_dpp(
        T.i32, o, fx.as_ir_value(_bits(v)), ctrl, row_mask, 0xF, True
    )
    return fx.Int32(r).bitcast(fx.Float32)


def _seq_sum_sq(vals):
    """((v0^2 + v1^2) + v2^2) + ...: each square rounded, no FMA (Triton's per-thread accumulation)."""
    sq = [v * v for v in vals]
    acc = sq[0] + sq[1]
    for q in sq[2:]:
        acc = acc + q
    return acc


def _wave_sum_lane63(acc):
    """Triton's wave reduction: row_shr 8/4/2/1 adds (lane 15 of each row holds the row sum), row_bcast 15 into rows 1
    and 3, row_bcast 31; the result is valid in lane 63: (R0 + R1) + (R2 + R3)."""
    for sh in (8, 4, 2, 1):
        acc = acc + _dpp_f(acc, 0x110 + sh)
    acc = _dpp_f(acc, 0x142, row_mask=0xA, old=acc) + acc
    return acc + _dpp_f(acc, 0x143)


def _block_sum(w):
    """Triton's fold of 16 wave sums (lane 0 of xor 8, 4, 2, 1), waves 12-15 = +0."""
    return (((w[0] + w[8]) + w[4]) + ((w[2] + w[10]) + w[6])) + (
        ((w[1] + w[9]) + w[5]) + ((w[3] + w[11]) + w[7])
    )


def _rsqrt_rn(ss, eps):
    """1 / v_sqrt_f32(ss / 6144 + eps), both divisions correctly rounded (Triton's ``1.0 / tl.sqrt(var + eps)`` with
    ``var = sum / n_cols``)."""
    var = ss / fx.Float32(float(H))
    sq = fx.Float32(_rocdl_ops.sqrt(T.f32, fx.as_ir_value(var + eps)))
    return fx.Float32(1.0) / sq


def _xcd_row(bid, m):
    """(row, live) of workgroup ``bid``: workgroup b runs on XCD b % 8, and XCD x takes the contiguous rows
    [x * (m // 8) + min(x, m % 8), ...); grid ``_xcd_grid(m)``."""
    x, k = bid % 8, bid // 8
    q8, r8 = m // 8, m % 8
    extra = x < r8
    row = x * q8 + extra.select(x, r8) + k
    return row, k < q8 + extra.select(fx.Int32(1), fx.Int32(0))


def _xcd_grid(m):
    return (m + 7) // 8 * 8


def _row_loader(off):
    def row(base, cm, m):
        rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=ROWB)
        return Vec(
            buffer_ops.buffer_load(
                rs, off, vec_width=4, dtype=fx.Int32, mask=m, cache_modifier=cm
            )
        )

    return row


@fx.struct
class _Shared:
    part: fx.Array[fx.Float32, 16]


def _make_launch(kernel, name, grid):
    """The ``@flyc.jit`` launcher shared by the three kernels: one workgroup of ``NT`` threads per grid slot,
    ``grid(rows, q)`` slots, every kernel taking the same 15 operands."""

    @flyc.jit
    def launch(
        i32_rows: fx.Int32,
        i32_q: fx.Int32,
        i32_me: fx.Int32,
        i32_sr: fx.Int32,
        arg_x: fx.Int64,
        arg_b0: fx.Int64,
        arg_b1: fx.Int64,
        arg_b2: fx.Int64,
        arg_b3: fx.Int64,
        arg_r: fx.Int64,
        arg_w: fx.Int64,
        arg_out: fx.Int64,
        arg_res: fx.Int64,
        arg_cnt: fx.Int64,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_rows, i32_q, i32_me, i32_sr, arg_x, arg_b0, arg_b1, arg_b2, arg_b3,
            arg_r, arg_w, arg_out, arg_res, arg_cnt, i32_eps,
        ).launch(
            grid=(fx.Int64(grid(i32_rows, i32_q)), 1, 1), block=(NT, 1, 1), stream=stream
        )  # fmt: skip

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch


def make_oneshot_ack_kernel():
    """Rows 1-11 on registered inputs: every rank reads the four inputs in the row owner's order, with read acks.

    Each wave announces its input ready, loads the 4 registered inputs in the row owner's order and acknowledges the
    reads before the norm; it waits for all readers before it ends, so the producer can reuse its input. Both phases
    share one generation word: 2e = ready, 2e + 1 = reads done. Hardware FP32 FTZ around the rank sum only.
    """
    name = "allreduce_oneshot_pull_ack_ranks4_gemma_add_rmsnorm_h6144"

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_rows: fx.Int32,  # grid = rows
        i32_q: fx.Int32,  # ceil(rows / 4): row b's 2-stage chunk owner is b / q
        i32_me: fx.Int32,
        i32_sr: fx.Int32,  # residual row stride (elements, a multiple of 8)
        arg_x: fx.Int64,  # int64[4] registered contiguous input pointers
        arg_b0: fx.Int64,  # the ranks' buffers (+ ACK_OFF)
        arg_b1: fx.Int64,
        arg_b2: fx.Int64,
        arg_b3: fx.Int64,
        arg_r: fx.Int64,  # residual [rows, H] bf16
        arg_w: fx.Int64,  # norm weight [H] bf16
        arg_out: fx.Int64,  # [rows, H] bf16
        arg_res: fx.Int64,  # residual out [rows, H] bf16
        arg_cnt: fx.Int64,  # int32[MAX_ROWS] call counts
        i32_eps: fx.Int32,  # fp32 bits
    ):
        tx, rw = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        me = i32_me
        smem = fx.SharedAllocator().allocate(_Shared).peek()
        part = smem.part.ptr
        bs = (arg_b0, arg_b1, arg_b2, arg_b3)
        mine = sel4(me, *bs)
        cr = buffer_ops.create_buffer_resource_from_addr(
            arg_cnt, num_records_bytes=MAX_ROWS * 4
        )
        e = (
            fx.Int32(
                buffer_ops.buffer_load(
                    cr, rw, vec_width=1, dtype=fx.Int32, is_scalar=True
                )
            )
            + 1
        )
        off = tx * 4
        rowoff = fx.Int64(rw) * ROWB
        row = _row_loader(off)
        xs = _load_ptrs(arg_x)
        rv = row(arg_r + fx.Int64(rw) * (fx.Int64(i32_sr) * fx.Int64(2)), 0, None)
        wv = row(arg_w, 0, None)
        pl = (me + 1 + lane) % WORLD
        wflag = fx.Int64((rw * NW + wave) * 128 + 16)
        if lane < WORLD - 1:
            _flag_store(sel4(pl, *bs) + wflag + fx.Int64(me * 4), e * 2)
            _spin(mine + wflag + fx.Int64(pl * 4), e * 2)
        owner = rw // i32_q
        ordered = [
            row(sel4((owner + t) % WORLD, *xs) + rowoff, SYS, None)
            for t in range_constexpr(WORLD)
        ]
        ordered, mode = _enter_fp32_ftz(ordered)
        reduced, rv = _restore_fp32_denorm_mode([_sum4_hardware_ftz(ordered), rv], mode)
        # the rank sum consumed every source pack: acknowledge the reads now, their delivery overlaps the norm
        if lane < WORLD - 1:
            _flag_store(sel4(pl, *bs) + wflag + fx.Int64(me * 4), e * 2 + 1)
        xv = _bf16x8(reduced)
        rvf = _bf16x8(rv)
        wvf = _bf16x8(wv)
        s = [xv[n] + rvf[n] for n in range_constexpr(8)]
        res = [_pk_bf16(s[2 * n], s[2 * n + 1]) for n in range_constexpr(4)]
        acc = _wave_sum_lane63(_seq_sum_sq(s))
        if lane == 63:
            part[wave] = acc
        resr = buffer_ops.create_buffer_resource_from_addr(
            arg_res + rowoff, num_records_bytes=ROWB
        )
        buffer_ops.buffer_store(Vec.from_elements(res, fx.Int32), resr, off)
        fx.gpu.barrier()
        rstd = _rsqrt_rn(
            _block_sum([part[n] for n in range_constexpr(NW)]),
            i32_eps.bitcast(fx.Float32),
        )
        y = [(s[n] * rstd) * (wvf[n] + fx.Float32(1.0)) for n in range_constexpr(8)]
        out = [_pk_bf16(y[2 * n], y[2 * n + 1]) for n in range_constexpr(4)]
        outr = buffer_ops.create_buffer_resource_from_addr(
            arg_out + rowoff, num_records_bytes=ROWB
        )
        buffer_ops.buffer_store(Vec.from_elements(out, fx.Int32), outr, off)
        # the producer may reuse its input only after all readers acknowledged
        if lane < WORLD - 1:
            _spin(mine + wflag + fx.Int64(pl * 4), e * 2 + 1)
        buffer_ops.buffer_store(e, cr, rw, mask=tx == 0)

    return _make_launch(kernel, name, lambda i32_rows, i32_q: i32_rows)


def make_pull_kernel(wave_handoffs=False):
    """Rows 12-287: two-stage on registered inputs, one row per workgroup.

    Workgroups 0 .. q - 1 (dispatched first) reduce chunk ``me`` (the reducer reads the peers' registered inputs
    after the pullers announced them); the others pull the reduced rows of chunks me + 1 .. me + 3.
    ``wave_handoffs``: per-wave flags instead of workgroup barriers; hardware FP32 FTZ around the rank sum.
    """
    name = (
        "allreduce_pull_ranks4_gemma_add_rmsnorm_h6144"
        + ("_wave" if wave_handoffs else "_block")
        + "_ftz"
    )

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_rows: fx.Int32,
        i32_q: fx.Int32,  # rows per chunk ceil(rows / 4); grid = 4 q
        i32_me: fx.Int32,
        i32_sr: fx.Int32,  # residual row stride (elements, a multiple of 8)
        arg_x: fx.Int64,  # int64[4]: the ranks' inputs [rows, H] bf16
        arg_b0: fx.Int64,  # the ranks' flag + reduced-row buffers (+ PULL_OFF)
        arg_b1: fx.Int64,
        arg_b2: fx.Int64,
        arg_b3: fx.Int64,
        arg_r: fx.Int64,  # residual [rows, H] bf16
        arg_w: fx.Int64,  # norm weight [H] bf16
        arg_out: fx.Int64,  # [rows, H] bf16
        arg_res: fx.Int64,  # residual out [rows, H] bf16
        arg_cnt: fx.Int64,  # int32[MAX_ROWS] call counts
        i32_eps: fx.Int32,  # fp32 bits
    ):
        tx, b = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        me, q = i32_me, i32_q
        # row = c q + j
        reducer = b < q
        puller = b >= q
        k = b - q
        one, zero = fx.Int32(1), fx.Int32(0)
        i = (k >= q).select(one, zero) + (k >= 2 * q).select(one, zero)
        c = reducer.select(me, (me + 1 + i) % WORLD)
        j = reducer.select(b, k - i * q)  # the row's slot in its reducer's buffer
        rw = c * q + j
        if rw < i32_rows:
            smem = fx.SharedAllocator().allocate(_Shared).peek()
            part = smem.part.ptr
            bs = (arg_b0, arg_b1, arg_b2, arg_b3)
            mine = sel4(me, *bs)
            cr = buffer_ops.create_buffer_resource_from_addr(
                arg_cnt, num_records_bytes=MAX_ROWS * 4
            )
            e = (
                fx.Int32(
                    buffer_ops.buffer_load(
                        cr, rw, vec_width=1, dtype=fx.Int32, is_scalar=True
                    )
                )
                + 1
            )
            off = tx * 4
            rowoff = fx.Int64(rw) * ROWB
            row = _row_loader(off)
            xs = _load_ptrs(arg_x)
            local = row(sel4(me, *xs) + rowoff, 0, reducer)
            if const_expr(wave_handoffs):
                _pull_arrive(*bs, c, me, rw, wave, e, puller)
                _pull_wait_pullers(mine, me, rw, wave, e, reducer)
            else:
                _pull_arrive(*bs, c, me, rw, fx.Int32(0), e, puller & (wave == 0))
                _pull_wait_pullers(mine, me, rw, fx.Int32(0), e, reducer & (wave == 0))
                fx.gpu.barrier()
            # wave v issues its peer loads in the order me + 1 + (t + v) % 3: each CU's requests spread over the 3 links
            # (the sum keeps the order me, me + 1, ..)
            rot = wave % (WORLD - 1)
            peer = [
                row(
                    sel4((me + 1 + (t + rot) % (WORLD - 1)) % WORLD, *xs) + rowoff,
                    SYS,
                    reducer,
                )
                for t in range_constexpr(WORLD - 1)
            ]
            rr = buffer_ops.create_buffer_resource_from_addr(
                arg_r + fx.Int64(rw) * (fx.Int64(i32_sr) * fx.Int64(2)),
                num_records_bytes=ROWB,
            )
            rvraw = Vec(buffer_ops.buffer_load(rr, off, vec_width=4, dtype=fx.Int32))
            wv = _bf16x8(row(arg_w, 0, None))
            packs, mode = _enter_fp32_ftz([local] + _unrotate(rot, peer))
            reduced, rvpack = _restore_fp32_denorm_mode(
                [_sum4_hardware_ftz(packs), rvraw], mode
            )
            slot = buffer_ops.create_buffer_resource_from_addr(
                mine + PULL_RED_OFF + fx.Int64(j) * ROWB, num_records_bytes=ROWB
            )
            buffer_ops.buffer_store(
                reduced, slot, off, mask=reducer, cache_modifier=SYS
            )
            fx.rocdl.s_waitcnt(vmcnt=0)
            if const_expr(wave_handoffs):
                _pull_publish(*bs, me, rw, wave, e, reducer)
            else:
                fx.gpu.barrier()
                _pull_publish(*bs, me, rw, fx.Int32(0), e, reducer & (wave == 0))
            # no vmcnt wait after publishing: the reducer's peer flag stores are acknowledged behind the link traffic
            x0 = fx.Int32(reduced[0])
            x1 = fx.Int32(reduced[1])
            x2 = fx.Int32(reduced[2])
            x3 = fx.Int32(reduced[3])
            if puller:
                if const_expr(wave_handoffs):
                    _pull_wait_reduced(mine, rw, wave, e)
                else:
                    _pull_wait_reduced(mine, rw, fx.Int32(0), e)
                    fx.gpu.barrier()
                pulled = row(
                    sel4(c, *bs) + PULL_RED_OFF + fx.Int64(j) * ROWB, SYS, puller
                )
                x0 = fx.Int32(pulled[0])
                x1 = fx.Int32(pulled[1])
                x2 = fx.Int32(pulled[2])
                x3 = fx.Int32(pulled[3])
            x = _bf16x8(Vec.from_elements([x0, x1, x2, x3], fx.Int32))
            rv = _bf16x8(rvpack)
            s = [x[i] + rv[i] for i in range_constexpr(8)]
            res = [_pk_bf16(s[2 * n], s[2 * n + 1]) for n in range_constexpr(4)]
            acc = _wave_sum_lane63(_seq_sum_sq(s))
            if lane == 63:
                part[wave] = acc
            resr = buffer_ops.create_buffer_resource_from_addr(
                arg_res + rowoff, num_records_bytes=ROWB
            )
            buffer_ops.buffer_store(Vec.from_elements(res, fx.Int32), resr, off)
            fx.gpu.barrier()
            rstd = _rsqrt_rn(
                _block_sum([part[n] for n in range_constexpr(NW)]),
                i32_eps.bitcast(fx.Float32),
            )
            y = [(s[n] * rstd) * (wv[n] + fx.Float32(1.0)) for n in range_constexpr(8)]
            out = [_pk_bf16(y[2 * n], y[2 * n + 1]) for n in range_constexpr(4)]
            outr = buffer_ops.create_buffer_resource_from_addr(
                arg_out + rowoff, num_records_bytes=ROWB
            )
            buffer_ops.buffer_store(Vec.from_elements(out, fx.Int32), outr, off)
            buffer_ops.buffer_store(e, cr, rw, mask=tx == 0)

    return _make_launch(kernel, name, lambda i32_rows, i32_q: i32_q * WORLD)


def _sub_rows(rpw):
    """Rows per handshake: 2 up to R16, 4 at R32."""
    return 2 if rpw <= 16 else 4


def make_stream_pull_kernel(rpw=2, hoist_weight=False):
    """Rows 288-640: R rows per workgroup of a chunk, batched handshakes, registered inputs read in place.

    Grid: 4 groups of workgroups of R consecutive rows of a chunk (chunk c = rows [c q, (c + 1) q)). Group 0
    (dispatched first) reduces chunk ``me``, groups 1-3 pull chunks me + 1 .. me + 3. A puller signals every
    sub-batch at once (its input is complete at kernel start); a puller waits for the publication of all its rows
    before it ends: when a rank's launch completes, the owners have consumed its input.
    ``hoist_weight``: load the Gemma weight once after the exchange and reuse it for every norm row.
    """
    R = rpw
    S = _sub_rows(R)
    name = (
        f"allreduce_stream_pull{'_hoist_weight' if hoist_weight else ''}"
        f"_ranks4_gemma_add_rmsnorm_h6144_r{R}"
    )

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_rows: fx.Int32,
        i32_q: fx.Int32,  # rows per chunk ceil(rows / 4); grid = 4 xcd_grid(ceil(q / R))
        i32_me: fx.Int32,
        i32_sr: fx.Int32,  # residual row stride (elements, a multiple of 8)
        arg_x: fx.Int64,  # int64[4]: the ranks' registered inputs [rows, H] bf16 (contiguous)
        arg_b0: fx.Int64,  # the ranks' flag + reduced-row buffers (+ STREAM_OFF)
        arg_b1: fx.Int64,
        arg_b2: fx.Int64,
        arg_b3: fx.Int64,
        arg_r: fx.Int64,  # residual [rows, H] bf16
        arg_w: fx.Int64,  # norm weight [H] bf16
        arg_out: fx.Int64,  # [rows, H] bf16
        arg_res: fx.Int64,  # residual out [rows, H] bf16
        arg_cnt: fx.Int64,  # int32[MAX_ROWS] call counts
        i32_eps: fx.Int32,  # fp32 bits
    ):
        tx, b = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        me, q = i32_me, i32_q
        # the workgroup's rows: c q + j0 + r, r < R
        qg = (q + (R - 1)) // R
        qp = (qg + 7) // 8 * 8
        one, zero = fx.Int32(1), fx.Int32(0)
        grp = (
            (b >= qp).select(one, zero)
            + (b >= 2 * qp).select(one, zero)
            + (b >= 3 * qp).select(one, zero)
        )
        jg, inq = _xcd_row(b - grp * qp, qg)
        c = (me + grp) % WORLD
        j0 = jg * R  # the first row's slot in its reducer's buffer
        rw0 = c * q + j0
        live = inq & (rw0 < i32_rows)
        reducer = (grp == 0) & live
        puller = (grp != 0) & live
        if live:
            smem = fx.SharedAllocator().allocate(_Shared).peek()
            part = smem.part.ptr
            bs = (arg_b0, arg_b1, arg_b2, arg_b3)
            mine = sel4(me, *bs)
            cr = buffer_ops.create_buffer_resource_from_addr(
                arg_cnt, num_records_bytes=MAX_ROWS * 4
            )
            off = tx * 4
            always = fx.Boolean(True)
            row = _row_loader(off)

            def store_row(v, base, m, cm):
                rs = buffer_ops.create_buffer_resource_from_addr(
                    base, num_records_bytes=ROWB
                )
                buffer_ops.buffer_store(v, rs, off, mask=m, cache_modifier=cm)

            xs = _load_ptrs(arg_x)

            def xrow(rw, p=None):
                return sel4(me if p is None else p, *xs) + fx.Int64(rw) * ROWB

            def count(j):
                """Sub-batch j's call count + 1 (its handshake's first row)."""
                return (
                    fx.Int32(
                        buffer_ops.buffer_load(
                            cr, rw0 + j * S, vec_width=1, dtype=fx.Int32, is_scalar=True
                        )
                    )
                    + 1
                )

            def sub_ok(j):
                return (j0 + j * S < q) & (rw0 + j * S < i32_rows)

            rot = wave % (WORLD - 1)
            n_sub = fx.Int32(R // S)
            if puller:
                # The input is complete at kernel start: signal every sub-batch now.
                for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                    _arrive(*bs, c, me, rw0 + j * S, count(j), sub_ok(j))
            if reducer:
                for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                    _wait_pullers(mine, me, rw0 + j * S, count(j), sub_ok(j))
                    for r in range(fx.Int32(0), fx.Int32(S), fx.Int32(1)):
                        ri = j * S + r
                        rw = rw0 + ri
                        ok = (j0 + ri < q) & (rw < i32_rows)
                        loc = row(xrow(rw), 0, ok)
                        peer = [
                            row(
                                xrow(rw, (me + 1 + (t + rot) % (WORLD - 1)) % WORLD),
                                SYS,
                                ok,
                            )
                            for t in range_constexpr(WORLD - 1)
                        ]
                        reduced = _sum4([loc] + _unrotate(rot, peer))
                        store_row(
                            reduced,
                            mine + STREAM_RED_OFF + fx.Int64(j0 + ri) * ROWB,
                            ok,
                            SYS,
                        )
                # one publish after the batch: a publish per sub-batch put its remote flag stores before the next poll
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                    _publish(*bs, me, rw0 + j * S, count(j), sub_ok(j))
            # every rank reads the owner's rows (local for the reducer), a puller each sub-batch once it is published
            if puller:
                for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                    if sub_ok(j):
                        _wait_reduced(mine, rw0 + j * S, count(j))
            wv_once = _bf16x8(row(arg_w, 0, always)) if hoist_weight else None
            for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                for r in range(fx.Int32(0), fx.Int32(S), fx.Int32(1)):
                    ri = j * S + r
                    rw = rw0 + ri
                    ok = (j0 + ri < q) & (rw < i32_rows)
                    xv = _bf16x8(
                        row(
                            sel4(c, *bs) + STREAM_RED_OFF + fx.Int64(j0 + ri) * ROWB,
                            SYS,
                            ok,
                        )
                    )
                    rv = _bf16x8(
                        row(
                            arg_r + fx.Int64(rw) * (fx.Int64(i32_sr) * fx.Int64(2)),
                            0,
                            ok,
                        )
                    )
                    wv = wv_once if hoist_weight else _bf16x8(row(arg_w, 0, always))
                    s = [xv[n] + rv[n] for n in range_constexpr(8)]
                    acc = _wave_sum_lane63(_seq_sum_sq(s))
                    if lane == 63:
                        part[wave] = acc
                    res = [_pk_bf16(s[2 * n], s[2 * n + 1]) for n in range_constexpr(4)]
                    store_row(
                        Vec.from_elements(res, fx.Int32),
                        arg_res + fx.Int64(rw) * ROWB,
                        ok,
                        0,
                    )
                    fx.gpu.barrier()
                    rstd = _rsqrt_rn(
                        _block_sum([part[n] for n in range_constexpr(NW)]),
                        i32_eps.bitcast(fx.Float32),
                    )
                    y = [
                        (s[n] * rstd) * (wv[n] + fx.Float32(1.0))
                        for n in range_constexpr(8)
                    ]
                    out = [_pk_bf16(y[2 * n], y[2 * n + 1]) for n in range_constexpr(4)]
                    store_row(
                        Vec.from_elements(out, fx.Int32),
                        arg_out + fx.Int64(rw) * ROWB,
                        ok,
                        0,
                    )
                    fx.gpu.barrier()
            for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                buffer_ops.buffer_store(
                    count(j), cr, rw0 + j * S, mask=(tx == 0) & sub_ok(j)
                )

    return _make_launch(
        kernel,
        name,
        lambda i32_rows, i32_q: _xcd_grid((i32_q + (R - 1)) // R) * WORLD,
    )
