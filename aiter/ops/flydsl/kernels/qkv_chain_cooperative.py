# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Technique source: amd-kernels attention/mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128
# (kernel_cooperative.py) and the decode MXFP8 GEMM primitives of gemm/mxfp8_common/mxfp8_gemm_decode.py.
"""One-launch MiniMax-M3.1 QKV front chain at 1-64 rows, TP4 (hidden 6144 -> 2560 columns).

The deferred split-3 MXFP8 GEMM (``mxpsh``-layout preshuffled weight, (16, 16)-tile activation codes with
AITER-shuffled E8M0 scales), q / k / index-q / index-k RMSNorm, partial RoPE and the NVFP4 K / V / index-K
page inserts, all in ONE launch. Each of 240 five-wave workgroups computes a 32-column tile of one K third in
wave 0 while the other waves preload the epilogue inputs. A head's 12 workgroups publish sc1 slabs, drain their
stores and bump the head's epoch counter; the last roles wait for all 12 and each processes eight rows. Epoch
counters: four split-0 arrivals add two, the other eight add one; owners wait for the epoch of 16, nobody tops
up, counts completions or resets (counters belong to a device and a stream or capture, see the host wrapper).
The result is bitwise the unfused chain (split-3 GEMM, bf16 slab rounding, norm / RoPE, quantization).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import BFloat16, Float32, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.qkv_front_common import (
    BLK,
    F448,
    F448N,
    RCP6,
    D,
    _bf16x4,
    _bits,
    _e2m1x2,
    _e4m3_byte,
    _e4m3_value,
    _f32,
    _gptr,
    _iq_fp8,
    _k_scale_off,
    _maxnum,
    _minnum,
    _pk_bf16,
    _rsq,
    _xmax,
    _xor_lane,
    _xsum,
)

# ---- decode MXFP8 GEMM primitives (amd-kernels gemm/mxfp8_common/mxfp8_gemm_decode.py) ----
W_CACHE_MOD = 2  # non-temporal W loads: a decode GEMM reads every weight byte once
# VGPRs the kernel may hold at once. An MFMA operand is 8 contiguous VGPRs, so the register
# allocator runs out well before all 256 are live (measured: clean at 199 live, parking
# operands in AGPRs at 202). A parked operand is copied back with v_accvgpr_read right
# before its MFMA and races the AGPR write: wrong sums, some of them only now and then.
VGPR_BUDGET = 192
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def _addr(t):
    """A tensor argument's device address (in a launch body)."""
    return fx.Int64(fx.ptrtoint(fx.get_iter(t)))


def _lookahead(SW, RT, NI, RP, CP, vgpr_budget=None, a_vgprs=8, reserve=0, fold=1):
    """(W steps, A blocks) issued ahead of the MFMAs: the longest common horizon whose
    registers fit. Held at once: the operands in flight plus the one in use of each
    queue (8 VGPRs per 128-K operand), a scale dword per 32-row / 32-column group for
    every other W step, and the per-lane offset vector of every row / column tile.
    An A block is one (step, row tile) pair, or ``fold`` steps of the folded bf16 mode.
    """
    budget = (vgpr_budget or VGPR_BUDGET) - (RT + NI) - reserve
    best = None
    for la_a in range(1, SW * RT // fold + 1):
        la_w = min(SW, -(-la_a * fold // RT))
        regs = a_vgprs * (la_a + 1) + 8 * NI * (la_w + 1) + (CP + RP) * (la_w // 2 + 2)
        if regs > budget:
            break
        best = (la_w, la_a)
    assert best is not None, f"RT {RT} x NI {NI} leaves no VGPRs for the operands"
    return best


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _pack8(h0, h1):
    """two 4 x i32 halves -> one 8 x i32 MFMA operand (VGPRs 0-3 = K [16*klane, +16),
    VGPRs 4-7 = K [64 + 16*klane, +16))."""
    a, b = Vec(h0), Vec(h1)
    return Vec.from_elements(
        [a[k] for k in range(4)] + [b[k] for k in range(4)], fx.Int32
    )


def _mfma(w_op, a_op, acc, sw, sa, jw, ia, k2):
    """One scaled 16x16x128 fp8 MFMA, accumulator pinned in AGPR (``=a`` / tied ``0``).
    Operand order (W, A): lane (q, r) accumulates out[row r, columns 4q .. 4q+3].
    Scale byte of a dword = op_sel + 2 * op_sel_hi (16-row / 16-col half, 128-K half).
    """
    src2 = "$0" if acc is not None else "0"
    asm = (
        f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {src2}, $3, $4 "
        f"op_sel:[{jw},{ia},0] op_sel_hi:[{k2},{k2},0] cbsz:0 blgp:0"
    )
    ops = [
        fx.as_ir_value(w_op),
        fx.as_ir_value(a_op),
        fx.as_ir_value(sw),
        fx.as_ir_value(sa),
    ]
    cons = "=a,v,v,v,v"
    if acc is not None:
        ops.append(fx.as_ir_value(acc))
        cons += ",0"
    return llvm.inline_asm(
        Vec.make_type(4, fx.Float32), ops, asm, cons, has_side_effects=True
    )


def _pin_accumulators(values):
    """Route the accumulators through an inline asm that outputs them again (tied
    ``=a`` / ``0``) with the XDL wait states inside: the reads that follow depend on
    this fence and cannot be hoisted right behind the last inline-asm MFMA (the
    compiler sees no hazard through the asm; without it the last MFMA of a wave is
    missing from half of each accumulator)."""
    one = len(values) == 1  # an inline asm with one output returns it bare
    ty = ir.Type.parse(
        "vector<4xf32>"
        if one
        else "!llvm.struct<(" + ", ".join(["vector<4xf32>"] * len(values)) + ")>"
    )
    constraints = ",".join(["=a"] * len(values) + [str(i) for i in range(len(values))])
    r = llvm.inline_asm(
        ty,
        [_raw(v) for v in values],
        "s_nop 15\ns_nop 15",
        constraints,
        has_side_effects=True,
    )
    if one:
        return [r]
    return [llvm.extractvalue(T.vec(4, T.f32), r, [i]) for i in range(len(values))]


# ---- exact NVFP4 quantization with the scale reciprocals staged in LDS (amd-kernels quant.py) ----
def _quant4_table(xs, table):
    """``_quant4``'s four codes and group scale, using the current call's exact reciprocal table."""
    a = [fx.math.absf(x) for x in xs]
    m = _maxnum(_maxnum(a[0], a[1]), _maxnum(a[2], a[3]))
    m = _xmax(_xmax(m, 1), 2)
    inv_gs = fx.Int32(table[256]).bitcast(fx.Float32)
    sf = _minnum((m * _f32(RCP6)) * inv_gs, _f32(F448))
    byte = _e4m3_byte(sf)
    quant = fx.Int32(table[byte]).bitcast(fx.Float32)
    y = [x * quant for x in xs]
    return _e2m1x2(y[2], y[3], _e2m1x2(y[0], y[1], fx.Int32(0), 0), 1), byte


# ---- V page words from per-row nibbles (amd-kernels packing.py) ----
def _v_perm(hi, lo, select):
    """Select four bytes from (hi:lo); the native byte-permute instruction implements the exact integer mapping."""
    return fx.Int32(fx.rocdl.perm_b32(fx.Int32(hi), fx.Int32(lo), fx.Int32(select)))


def _v_expand(nibbles):
    """Four low-16-bit nibbles -> the low nibble of each byte, before the LDS join."""
    value = (nibbles | (nibbles << 8)) & 0x00FF00FF
    return (value | (value << 4)) & 0x0F0F0F0F


def _v_expanded_words(nibbles):
    """Four/eight expanded rows -> page words; each producer has already expanded its own row."""
    pairs = [nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, len(nibbles), 2)]
    x = _v_perm(pairs[1], pairs[0], 0x05010400)
    y = _v_perm(pairs[1], pairs[0], 0x07030602)
    if len(nibbles) == 4:
        return [x & 0xFFFF, (x >> 16) & 0xFFFF, y & 0xFFFF, (y >> 16) & 0xFFFF]
    u = _v_perm(pairs[3], pairs[2], 0x05010400)
    v = _v_perm(pairs[3], pairs[2], 0x07030602)
    return [
        _v_perm(u, x, 0x05040100),
        _v_perm(u, x, 0x07060302),
        _v_perm(v, y, 0x05040100),
        _v_perm(v, y, 0x07060302),
    ]


# ---- the kernel ----
N, K = 2560, 6144
NQ, NKV, NIQ, ROT = 16, 1, 1, 64
NI, KW, KS = (
    2,
    1,
    3,
)  # 32-column tiles and three sequential K thirds, as in the deferred GEMM
NB = N // (16 * NI)  # 80 column tiles
HEADS = N // D  # 20
TILES_PER_HEAD = D // (16 * NI)  # 4
WG_PER_HEAD = TILES_PER_HEAD * KS  # 12
SC1 = 0x10  # agent-scope (cross-XCD) stores / loads of the slabs
CNT_STRIDE = 32  # one counter per head, each on its own 128-byte line
EPOCH = 16  # what one call adds to each counter it uses (a power of 2 >= its producer count)
NR = NQ + NKV + 1 + NIQ  # norm / RoPE slots: 19
S = NR + NKV  # reduce-insert wave slots per token pair (one pair per wave): 20
ROW = N
HALF = ROT // 2
SIDE = NKV * BLK * 72


def _atomic_add(addr, v):
    """Returning scalar global RMW, called only by the epoch publisher/poller (one active lane).

    FlyDSL has no SMEM atomic wrapper. Native assembly fixes the SGPR instruction and return acknowledgement.
    The caller drains coherent slab stores before publication; the memory clobber prevents compiler reordering.
    """
    low = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int64(addr).to(fx.Int32)))
    high = fx.Int32(fx.rocdl.readfirstlane(T.i32, (fx.Int64(addr) >> 32).to(fx.Int32)))
    pointer = (
        low.to(fx.Uint32).to(fx.Uint64) | (high.to(fx.Uint32).to(fx.Uint64) << 32)
    ).bitcast(fx.Int64)
    inc = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(v)))
    return fx.Int32(
        _llvm.inline_asm(
            T.i32,
            [fx.as_ir_value(inc), fx.as_ir_value(pointer)],
            "s_atomic_add $0, $2, 0 glc\n\ts_waitcnt lgkmcnt(0)",
            "=s,0,s,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
        )
    )


def _lds_ptr(addr):
    return _llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.as_ir_value(fx.Int64(addr)))


def _lds_rmw(addr, v, order):
    """An LDS atomic add (workgroup scope) returning the old value."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _lds_ptr(addr),
            fx.Int32(v).ir_value(),
            order,
            syncscope="workgroup",
            alignment=4,
        ).result
    )


@flyc.jit
def _lds_wait(addr, target):
    """Spin until the LDS counter at ``addr`` reaches ``target`` (acquire: later LDS reads stay after it)."""
    f = _lds_rmw(addr, 0, _llvm.AtomicOrdering.acquire)
    while f < fx.Int32(target):
        f = _lds_rmw(addr, 0, _llvm.AtomicOrdering.acquire)


@functools.cache
def compile_qkv_cooperative(n_tokens: int, lookahead: tuple[int, int] | None = None):
    """The measured 32 / 48 / 64-row GEMM tiles; the tail assigns eight rows per owner.

    The activation is the public quantizer's row-major operand: e4m3 codes ``[m, K]`` and row-major E8M0 scales
    ``[m, K / 32]``, no padding and no relayout (a lane's 16 B of a 64-K block come from its own row, the scale bytes of a
    row pair are gathered and packed in registers, rows past m read as zero through the buffer bound; scheme of
    ``mxfp8_gemm_decode_splitk``). ``lookahead`` = (W steps, A blocks) in flight, default the measured table.
    """
    assert 1 <= n_tokens <= 64
    RT = (n_tokens + 15) // 16
    OWNERS = 2 * RT
    ROWS_PER_OWNER = 8
    NWG, PW, pairs = 5, 1, 4
    PRE_WORDS = 12
    # Same-process cold graph sweeps. Per-step scheduling barriers keep all variants under 192 VGPRs.
    law, laa = {1: (4, 4), 2: (4, 6), 3: (4, 8), 4: (3, 8)}[RT]
    KSTEPS = K // 128
    SW = KSTEPS // (KS * KW)  # 16 K steps per workgroup
    SCB = K // 256
    LA_W, LA_A = _lookahead(SW, RT, NI, 1, 1)
    if lookahead is not None:
        law, laa = lookahead
    if law:
        LA_W, LA_A = law, laa
    NA = SW * RT
    name = f"mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128_ks3_tp{pairs}_cooperative_rt{RT}_own{OWNERS}_la{LA_W}x{LA_A}_ep16"

    # every tile stages the head scale reciprocals in LDS; K / V / index-K share the exact helper
    name += "_vtable"
    name += "_vexpand_weighted_halo"

    name += "_satomic"
    name += "_arow"

    @fx.struct
    class Shared:
        ticket: fx.Array[fx.Int32, 4]
        pre: fx.Array[fx.Int32, (NWG - 1) * PW * PRE_WORDS * 64]
        vrow: fx.Array[
            fx.Int32, 16 * 32 * 2
        ]  # the V head's rows quantized: (codes, scale byte) per token and lane
        vcnt: fx.Array[fx.Int32, 4]  # V tail waves whose rows are in vrow
        reciprocal: fx.Array[fx.Int32, 260, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NWG, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        a_scale_bytes: fx.Int32,
        arg_w: fx.Int64,
        arg_ws: fx.Int64,
        arg_as: fx.Int64,
        arg_a: fx.Int64,
        arg_slab: fx.Int64,  # [3, m, 2560] fp32 partials (workspace)
        arg_cnt: fx.Int64,  # [20 * CNT_STRIDE] i32 head counters, a multiple of EPOCH between calls
        i32_cs_rows: fx.Int32,
        arg_pos: fx.Int64,
        arg_slot: fx.Int64,
        arg_islot: fx.Int64,
        arg_cs: fx.Int64,
        arg_qw: fx.Int64,
        arg_kw: fx.Int64,
        arg_iqw: fx.Int64,
        arg_ikw: fx.Int64,
        arg_q: fx.Int64,
        arg_iq: fx.Int64,
        arg_kv: fx.Int64,
        arg_idx: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        i64_kv_st: fx.Int64,
        i64_idx_st: fx.Int64,
        i32_iq_st: fx.Int32,
        i32_eps: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        kz, nb = bx // NB, bx % NB
        head = nb // TILES_PER_HEAD
        role = kz * TILES_PER_HEAD + nb % TILES_PER_HEAD
        owner = role >= WG_PER_HEAD - OWNERS
        row_base = (role - (WG_PER_HEAD - OWNERS)) * ROWS_PER_OWNER
        lane_off = q16 * 256 + l16 * 16
        k_base = kz * (SW * 2048)
        slab = buffer_ops.create_buffer_resource_from_addr(
            arg_slab, num_records_bytes=i32_m * (N * 4 * KS)
        )

        def gemm():
            a_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_a, num_records_bytes=i32_m * K
            )
            w_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_w, num_records_bytes=N * K
            )
            as_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_as, num_records_bytes=a_scale_bytes
            )
            ws_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_ws, num_records_bytes=(N // 32) * SCB * 256
            )
            n16 = [nb * NI + ni for ni in range_constexpr(NI)]
            w_dw = [
                (n16[ni] * (16 * K) + k_base + lane_off) // 4
                for ni in range_constexpr(NI)
            ]
            # row-major codes: lane (q16, l16) -> row 16 * tile + l16, K bytes [16 * q16, +16) of a 64-K block
            a_dw = [
                ((mi * 16 + l16) * K + kz * (SW * 128) + q16 * 16) // 4
                for mi in range_constexpr(RT)
            ]
            # raw row-major scales of 32-row pair rp (tiles 2 rp, 2 rp + 1): the rows' 8 K-group bytes of a 256-K
            # block (an 8-byte read: step 0's four groups, then step 1's)
            as_row_dw = [
                ((32 * rp + l16) * (K // 32) + kz * (SW * 4)) // 4
                for rp in range_constexpr((RT + 1) // 2)
            ]
            sc_lane = q16 * 16 + l16
            sc_k = kz * (SW // 2 * 64)
            ws_dw = ((n16[0] // 2) * SCB) * 64 + sc_k + sc_lane

            # A rows past m are the producer's padding: an output row reads only its own A row and rows past m are
            # never stored, so their loads are masked off (half of A's bytes at 8 rows)

            def load_pair(rsrc, dw, s, nt, mi=0, step=2048, half=1024):
                # a step's two 64-K blocks: 1 KB apart in the (16, 16) layout, 64 B apart in a row-major row
                kw = {"vec_width": 4, "dtype": fx.Int32}
                if const_expr(nt):
                    kw["cache_modifier"] = W_CACHE_MOD
                else:
                    kw["mask"] = l16 + mi * 16 < i32_m
                return _pack8(
                    buffer_ops.buffer_load(rsrc, dw, soffset_bytes=s * step, **kw),
                    buffer_ops.buffer_load(
                        rsrc, dw, soffset_bytes=s * step + half, **kw
                    ),
                )

            def load_scale(rsrc, dw, s):
                return fx.Int32(
                    buffer_ops.buffer_load(
                        rsrc,
                        dw,
                        vec_width=1,
                        dtype=fx.Int32,
                        soffset_bytes=(s // 2) * 256,
                    )
                )

            def load_a(p):
                mi, s = p % RT, p // RT
                return load_pair(a_rsrc, a_dw[mi], s, False, mi, 128, 64)

            def load_scales_row(s):
                out = []
                for dw in as_row_dw:
                    pair = []
                    for hi in (0, 1):
                        v = Vec(
                            buffer_ops.buffer_load(
                                as_rsrc,
                                dw + hi * (16 * (K // 32) // 4),
                                vec_width=2,
                                dtype=fx.Int32,
                                soffset_bytes=(s // 2) * 8,
                            )
                        )
                        pair.append((fx.Int32(v[0]), fx.Int32(v[1])))
                    out.append(pair)
                return out

            def pack_a_scale_row(raw, p):
                # bytes (step 0, row l16), (step 0, row l16 + 16), (step 1, row l16), (step 1, row l16 + 16), each this
                # lane's K quad (q16): the layout the MFMA op_sel immediates address (op_sel = row half, op_sel_hi = step)
                (lo0, lo1), (hi0, hi1) = raw[p]
                sh = q16 * 8
                m8 = fx.Int32(0xFF)
                return (
                    (lo0.shrui(sh) & m8)
                    | ((hi0.shrui(sh) & m8) << 8)
                    | ((lo1.shrui(sh) & m8) << 16)
                    | (hi1.shrui(sh) << 24)
                )

            def load_w_step(s):
                bb = [
                    load_pair(w_rsrc, w_dw[ni], s, True) for ni in range_constexpr(NI)
                ]
                if const_expr(s % 2 == 0):
                    return bb, (
                        load_scale(ws_rsrc, ws_dw, s),
                        load_scales_row(s),
                    )
                return bb, None

            acc = [None] * (NI * RT)
            w_q = [None] * SW
            a_q = [None] * NA
            for s in range_constexpr(LA_W):
                w_q[s] = load_w_step(s)
            for t in range_constexpr(LA_A):
                a_q[t] = load_a(t)
            fx.rocdl.sched_barrier(0)
            sw = sa = None
            for s in range_constexpr(SW):
                if const_expr(s + LA_W < SW):
                    w_q[s + LA_W] = load_w_step(s + LA_W)
                bb, sc = w_q[s]
                if const_expr(s % 2 == 0):
                    sw, sa = sc
                    sa_raw, sa = sa, [None] * ((RT + 1) // 2)  # packed at first use
                for mi in range_constexpr(RT):
                    t = s * RT + mi
                    if const_expr(t + LA_A < NA):
                        p = t + LA_A
                        a_q[p] = load_a(p)
                    if const_expr(sa[mi // 2] is None):
                        sa[mi // 2] = pack_a_scale_row(sa_raw, mi // 2)
                    a_op = a_q[t]
                    a_q[t] = None
                    for ni in range_constexpr(NI):
                        ai = mi * NI + ni
                        acc[ai] = _mfma(
                            bb[ni],
                            a_op,
                            acc[ai],
                            sw,
                            sa[mi // 2],
                            ni % 2,
                            mi % 2,
                            s % 2,
                        )
                fx.rocdl.sched_barrier(0)
                w_q[s] = None
            flat = _pin_accumulators(acc)
            for mi in range_constexpr(RT):
                row = l16 + mi * 16
                for ni in range_constexpr(NI):
                    col = (nb * NI + ni) * 16 + q16 * 4
                    buffer_ops.buffer_store(
                        Vec(flat[mi * NI + ni]).bitcast(fx.Int32),
                        slab,
                        ((kz * i32_m + row) * N + col) * 4,
                        offset_is_bytes=True,
                        mask=row < i32_m,
                        cache_modifier=SC1,
                    )

        def tail_pre(s, pair):
            """Every load of the reduce-insert wave (slot ``s``, one token pair) that does not read the slabs, issued
            while wave 0 runs the GEMM (the same addresses and masks as the reduce-insert's).
            """
            L_i32_m = i32_m
            L_i32_cs_rows = i32_cs_rows
            L_arg_pos = arg_pos
            L_arg_slot = arg_slot
            L_arg_islot = arg_islot
            L_arg_cs = arg_cs
            L_arg_qw = arg_qw
            L_arg_kw = arg_kw
            L_arg_ks = arg_ks
            L_arg_vs = arg_vs
            L_arg_iqw = arg_iqw
            L_arg_ikw = arg_ikw
            L_lane = lane
            hl = L_lane % 32
            dim0 = hl * 4
            tok = pair * 2 + L_lane // 32
            live = tok < L_i32_m
            is_n = s < NR
            is_v = s >= NR
            is_q = s < NQ
            is_k = (s >= NQ) & (s < NQ + NKV)
            is_ik = s == NQ + NKV
            sl_r = buffer_ops.create_buffer_resource_from_addr(
                L_arg_slot, num_records_bytes=fx.Int64(L_i32_m) * 8
            )
            pos_r = buffer_ops.create_buffer_resource_from_addr(
                L_arg_pos, num_records_bytes=fx.Int64(L_i32_m) * 8
            )
            isl_r = buffer_ops.create_buffer_resource_from_addr(
                L_arg_islot, num_records_bytes=fx.Int64(L_i32_m) * 8
            )
            cs_r = buffer_ops.create_buffer_resource_from_addr(
                L_arg_cs, num_records_bytes=fx.Int64(L_i32_cs_rows) * (ROT * 2)
            )
            w_addr = is_q.select(
                L_arg_qw, is_k.select(L_arg_kw, is_ik.select(L_arg_ikw, L_arg_iqw))
            )
            wr = buffer_ops.create_buffer_resource_from_addr(
                w_addr, num_records_bytes=D * 2
            )
            pos = fx.Int32(
                buffer_ops.buffer_load(
                    pos_r, tok * 2, vec_width=1, dtype=fx.Int32, mask=live & is_n
                )
            )
            mslot = fx.Int32(
                buffer_ops.buffer_load(
                    sl_r,
                    tok * 2,
                    vec_width=1,
                    dtype=fx.Int32,
                    mask=live & (is_k | is_v),
                )
            )
            islot = fx.Int32(
                buffer_ops.buffer_load(
                    isl_r, tok * 2, vec_width=1, dtype=fx.Int32, mask=live & is_ik
                )
            )
            wv = Vec(buffer_ops.buffer_load(wr, hl * 2, vec_width=2, dtype=fx.Int32))
            first = dim0 < HALF
            in_rope = dim0 < ROT
            ib = first.select(dim0, dim0 - HALF)
            cs_off = (pos * ROT + ib) // 2
            cosv = Vec(
                buffer_ops.buffer_load(
                    cs_r,
                    cs_off,
                    vec_width=2,
                    dtype=fx.Int32,
                    mask=live & in_rope & is_n,
                )
            )
            sinv = Vec(
                buffer_ops.buffer_load(
                    cs_r,
                    cs_off + HALF // 2,
                    vec_width=2,
                    dtype=fx.Int32,
                    mask=live & in_rope & is_n,
                )
            )
            near_t = tok - 7 + hl
            has_near = is_v & (hl < 16) & (near_t >= 0) & (near_t < L_i32_m)
            near = has_near.select(
                fx.Int32(
                    buffer_ops.buffer_load(
                        sl_r, near_t * 2, vec_width=1, dtype=fx.Int32, mask=has_near
                    )
                ),
                fx.Int32(-1),
            )
            has_prev = is_v & live & (tok > 0)
            prev = has_prev.select(
                fx.Int32(
                    buffer_ops.buffer_load(
                        sl_r, tok * 2 - 2, vec_width=1, dtype=fx.Int32, mask=has_prev
                    )
                ),
                fx.Int32(-1),
            )
            # the global K / V scales, so the tail's quantizers do not wait for them
            ksb = fx.Float32(_gptr(L_arg_ks, T.f32, 4)[0]).bitcast(fx.Int32)
            vsb = fx.Float32(_gptr(L_arg_vs, T.f32, 4)[0]).bitcast(fx.Int32)
            return [mslot, islot, wv, cosv, sinv, near, prev, ksb, vsb]

        def tail_post(s, pairs, pres):
            """The reduce-insert wave's slab reads and arithmetic (slot ``s``) for this wave's token pairs (loads of
            every pair first, then the arithmetic, then the stores), slabs at agent scope.
            """
            L_i32_m = i32_m
            L_i32_eps = i32_eps
            L_i32_iq_st = i32_iq_st
            L_i64_kv_st = i64_kv_st
            L_i64_idx_st = i64_idx_st
            L_arg_q = arg_q
            L_arg_iq = arg_iq
            L_arg_kv = arg_kv
            L_arg_idx = arg_idx
            L_slab = slab
            L_lane = lane
            L_wave = wave
            L_row_base = row_base
            L_vrow = fx.Int32(fx.ptrtoint(smem.vrow.ptr))
            L_vcnt = fx.Int32(fx.ptrtoint(smem.vcnt.ptr))
            hl = L_lane % 32
            hi16 = (hl & 16) != 0
            dim0 = hl * 4
            toks = [pair * 2 + L_lane // 32 for pair in pairs]
            lives = [tk < L_i32_m for tk in toks]
            eps = L_i32_eps.bitcast(fx.Float32)
            TP = len(pairs)

            def load4(tok, col_elem, live):
                return [
                    Vec(
                        buffer_ops.buffer_load(
                            L_slab,
                            (k * L_i32_m + tok) * ROW + col_elem + dim0,
                            vec_width=4,
                            dtype=Float32,
                            mask=live,
                            cache_modifier=SC1,
                        )
                    )
                    for k in range_constexpr(KS)
                ]

            def sum4(frags):
                acc = Vec.filled(4, 0.0, Float32)
                for k in range_constexpr(KS):
                    acc = acc + frags[k]
                e = acc.to(BFloat16).to(Float32)
                return [e[i] for i in range_constexpr(4)]

            if s < NR:
                is_q = s < NQ
                is_k = (s >= NQ) & (s < NQ + NKV)
                is_ik = s == NQ + NKV
                is_kq = is_k | is_ik
                col = (s <= NQ + NKV).select(
                    is_ik.select(fx.Int32(NQ + 2 * NKV + NIQ), s),
                    s - (NQ + NKV + 1) + (NQ + 2 * NKV),
                )
                frags = [load4(toks[t], col * D, lives[t]) for t in range_constexpr(TP)]
                first = dim0 < HALF
                in_rope = dim0 < ROT
                qvs, slots = [], []
                for t in range_constexpr(TP):
                    mslot, islot, wv, cosv, sinv, near, prev = pres[t][:7]
                    e = sum4(frags[t])
                    slots.append(is_k.select(mslot, islot))
                    w = _bf16x4(wv)
                    cos, sin = _bf16x4(cosv), _bf16x4(sinv)
                    ss = e[0] * e[0] + e[1] * e[1]
                    ss = ss + e[2] * e[2]
                    ss = ss + e[3] * e[3]
                    for m in (16, 8, 4, 2, 1):
                        ss = _xsum(ss, m)
                    r = _rsq(fx.math.fma(ss, fx.Float32(0.0078125), eps))
                    nrm = [
                        (e[i] * r) * (fx.Float32(1.0) + w[i])
                        for i in range_constexpr(4)
                    ]
                    out = []
                    for i in range_constexpr(4):
                        pr = _xor_lane(_bits(nrm[i]), HALF // 4, hi16).bitcast(
                            fx.Float32
                        )
                        ps = pr * sin[i]
                        roped = fx.math.fma(nrm[i], cos[i], first.select(-ps, ps))
                        out.append(in_rope.select(roped, nrm[i]))
                    lo, hi = _pk_bf16(out[0], out[1]), _pk_bf16(out[2], out[3])
                    qvs.append(Vec.from_elements([lo, hi], fx.Int32))
                if is_q:
                    q_r = buffer_ops.create_buffer_resource_from_addr(
                        L_arg_q, num_records_bytes=fx.Int64(L_i32_m) * (NQ * D * 2)
                    )
                    for t in range_constexpr(TP):
                        buffer_ops.buffer_store(
                            qvs[t],
                            q_r,
                            toks[t] * (NQ * D // 2) + s * (D // 2) + hl * 2,
                            mask=lives[t],
                        )
                if s > NQ + NKV:
                    iq_r = buffer_ops.create_buffer_resource_from_addr(
                        L_arg_iq,
                        num_records_bytes=fx.Int64(L_i32_m) * fx.Int64(L_i32_iq_st),
                    )
                    iq_col = (s - (NQ + NKV + 1)) * D
                    for t in range_constexpr(TP):
                        f = [
                            _minnum(_maxnum(x, _f32(F448N)), _f32(F448))
                            for x in _bf16x4(qvs[t])
                        ]
                        word = _iq_fp8(
                            f[2], f[3], _iq_fp8(f[0], f[1], fx.Int32(0), False), True
                        )
                        buffer_ops.buffer_store(
                            word,
                            iq_r,
                            toks[t] * L_i32_iq_st + iq_col + dim0,
                            mask=lives[t],
                            offset_is_bytes=True,
                        )
                if is_kq:
                    quants = [
                        _quant4_table(_bf16x4(qvs[t]), smem.reciprocal.ptr)
                        for t in range_constexpr(TP)
                    ]
                    h = s - NQ
                    grp = hl // 4
                    for t in range_constexpr(TP):
                        slot = slots[t]
                        tp_ = slot & (BLK - 1)
                        page = is_k.select(
                            L_arg_kv + fx.Int64(slot >> 7) * L_i64_kv_st,
                            L_arg_idx + fx.Int64(slot >> 7) * L_i64_idx_st,
                        )
                        d_off = (
                            is_k.select(h * (BLK * 64), fx.Int32(0)) + tp_ * 64 + hl * 2
                        )
                        s_off = is_k.select(
                            NKV * BLK * 64 + h * (BLK * 8) + _k_scale_off(tp_, grp),
                            BLK * 64 + tp_ * 8 + grp,
                        )
                        d_ptr = _gptr(page + fx.Int64(d_off), T.i16, 2)
                        s_ptr = _gptr(page + fx.Int64(s_off), T.i8, 1)
                        packed = quants[t][0].to(fx.Int16)
                        sf8 = quants[t][1].to(fx.Int8)
                        if lives[t] & (slot >= 0):
                            d_ptr[0] = packed
                            if hl % 4 == 0:
                                s_ptr[0] = sf8
            else:
                vh = 0
                vcol = (NQ + NKV + vh) * D
                # each V wave sums, rounds and quantizes its own token pairs' rows (3 slab loads and one ``_quant4``
                # a lane and pair, as a q wave) into LDS; once all tail waves have, the first token of a run packs
                # the run's codes and scales from LDS (the reduce-insert's V wave loads the 8 rows a run can cover,
                # 24 slab vectors a lane, and quantizes them one after the other: the tail's long pole)
                v_toks = toks + [toks[0] + ROWS_PER_OWNER]
                frags = [load4(tk, vcol, tk < L_i32_m) for tk in v_toks]
                # Keep owned slab loads before optional halo work. This compiler fence avoids the measured
                # long-run regression of the unfenced variant.
                _llvm.FenceOp(_llvm.AtomicOrdering.acq_rel, syncscope="singlethread")
                # the runs (their slots were preloaded) and the octet dwords they keep, read before the wait so the
                # page round trip overlaps it; a lane that writes nothing reads page 0 (only this V tail writes these
                # dwords, so the values are the ones the reduce-insert reads right before its stores)
                runs = []
                for t in range_constexpr(TP):
                    mslot, islot, wv, cosv, sinv, near, prev = pres[t][:7]
                    slot = mslot
                    base = slot - slot % 8
                    in_oct = (near >= base) & (near < base + 8)
                    votes = fx.Int64(fx.rocdl.ballot(T.i64, in_oct))
                    mine = (votes >> fx.Int64((L_lane // 32) * 32)).to(
                        fx.Int32
                    ) & 0xFFFF
                    n_run = fx.Int32(fx.math.ctpop(mine))
                    run_first = (prev < 0) | ((prev >> 3) != (slot >> 3))
                    go = lives[t] & (slot >= 0) & run_first
                    wslot = go.select(slot, fx.Int32(0))
                    tp_ = wslot & (BLK - 1)
                    v_side = L_arg_kv + fx.Int64(wslot >> 7) * L_i64_kv_st + SIDE
                    v_data = v_side + fx.Int64(vh * (BLK * 64) + (tp_ // 8) * (D * 4))
                    v_sf = v_side + fx.Int64(
                        NKV * BLK * 64 + vh * (BLK * 8) + tp_ - tp_ % 8
                    )
                    dp = _gptr(v_data + fx.Int64(dim0 * 4), T.i32, 4)
                    sp = _gptr(v_sf + fx.Int64((hl // 4) * BLK), T.i32, 4)
                    old = [fx.Int32(dp[i]) for i in range_constexpr(4)] + [
                        fx.Int32(sp[i]) for i in range_constexpr(2)
                    ]
                    runs.append((n_run, go, tp_, dp, sp, old))
                # The existing neighbor vector contains both sides of the owner's 8-row boundary.
                # Every tail wave reads the same two slots. Different octets (or no next token) mean no
                # owned contiguous V run can consume a halo row. Keep the fixed per-wave readiness weight.
                boundary_lane = fx.Int32(14) - 2 * (L_wave - 1)
                last_slot = fx.Int32(
                    fx.rocdl.readlane(T.i32, pres[0][5], boundary_lane)
                )
                next_slot = fx.Int32(
                    fx.rocdl.readlane(T.i32, pres[0][5], boundary_lane + 1)
                )
                use_halo = (
                    (last_slot >= 0)
                    & (next_slot >= 0)
                    & ((last_slot >> 3) == (next_slot >> 3))
                )
                for t in range_constexpr(TP + 1):
                    if const_expr(t < TP):
                        nib_o, sf_o = _quant4_table(sum4(frags[t]), smem.reciprocal.ptr)
                        _llvm.StoreOp(
                            fx.as_ir_value(
                                Vec.from_elements(
                                    [_v_expand(fx.Int32(nib_o)), fx.Int32(sf_o)],
                                    fx.Int32,
                                )
                            ),
                            _lds_ptr(L_vrow + ((v_toks[t] - L_row_base) * 32 + hl) * 8),
                            alignment=8,
                        )
                    else:
                        if use_halo:
                            nib_o, sf_o = _quant4_table(
                                sum4(frags[t]), smem.reciprocal.ptr
                            )
                            _llvm.StoreOp(
                                fx.as_ir_value(
                                    Vec.from_elements(
                                        [_v_expand(fx.Int32(nib_o)), fx.Int32(sf_o)],
                                        fx.Int32,
                                    )
                                ),
                                _lds_ptr(
                                    L_vrow + ((v_toks[t] - L_row_base) * 32 + hl) * 8
                                ),
                                alignment=8,
                            )
                if L_lane == 0:
                    _lds_rmw(L_vcnt, TP + 1, _llvm.AtomicOrdering.release)
                _lds_wait(L_vcnt, (NWG - 1) * (TP + 1))
                for t in range_constexpr(TP):
                    tok = toks[t]
                    n_run, go, tp_, dp, sp, oldv = runs[t]
                    for run_rows, take in (
                        (4, go & (n_run <= 4)),
                        (8, go & (n_run > 4)),
                    ):
                        if take:
                            old = oldv[:4]
                            old_sc = oldv[4:6]
                            nibbles = []
                            scales = fx.Uint64(0)
                            for r in range_constexpr(run_rows):
                                got = Vec(
                                    _llvm.load(
                                        T.vec(2, T.i32),
                                        _lds_ptr(
                                            L_vrow
                                            + ((tok + r - L_row_base) * 32 + hl) * 8
                                        ),
                                        alignment=8,
                                    )
                                )
                                ok = (tok + r < L_i32_m) & (r < n_run)
                                nib = ok.select(got[0], fx.Int32(0))
                                sf = ok.select(got[1], fx.Int32(0))
                                nibbles.append(nib)
                                scales = scales | (sf.to(fx.Uint64) << (8 * r))
                            words = _v_expanded_words(nibbles)
                            p0 = tp_ % 8
                            mask = (fx.Uint32(0xFFFFFFFF) >> ((8 - n_run) * 4)) << (
                                p0 * 4
                            )
                            for i in range_constexpr(4):
                                dp[i] = (words[i] << (p0 * 4)) | (
                                    old[i] & ~mask.to(fx.Int32)
                                )
                            smask = (
                                fx.Uint64(0xFFFFFFFFFFFFFFFF) >> ((8 - n_run) * 8)
                            ) << (p0 * 8)
                            scales = scales << (p0 * 8)
                            sc0 = scales.to(fx.Int32) | (
                                old_sc[0] & ~smask.to(fx.Int32)
                            )
                            sc1 = (scales >> 32).to(fx.Int32) | (
                                old_sc[1] & ~(smask >> 32).to(fx.Int32)
                            )
                            if hl % 4 == 0:
                                sp[0] = sc0
                                sp[1] = sc1

        # head -> the reduce-insert's slot: q h, k 16, v 19, index-q 18, index-k 17
        slot_s = (head <= NQ).select(
            head,
            (head == NQ + 1).select(
                fx.Int32(NR),
                (head == NQ + 2).select(fx.Int32(NQ + NKV + 1), fx.Int32(NQ + NKV)),
            ),
        )
        cnt_addr = arg_cnt + fx.Int64(head) * (CNT_STRIDE * 4)
        if tx == 64:
            smem.vcnt.ptr[0] = fx.Int32(0)
        if wave == 0:
            gemm()
            # every slab store is acknowledged before the ticket is taken
            fx.rocdl.s_waitcnt(vmcnt=0)
        else:
            # the tail's slab-independent loads while wave 0 runs the GEMM, parked in this wave's LDS slot
            for p in range_constexpr(PW):
                pre0 = tail_pre(slot_s, row_base // 2 + (wave - 1) + p * (NWG - 1))
                words0 = [
                    pre0[0],
                    pre0[1],
                    *[Vec(pre0[k])[i] for k in (2, 3, 4) for i in range(2)],
                    *pre0[5:],
                ]
                for k in range_constexpr(PRE_WORDS):
                    smem.pre.ptr[
                        (((wave - 1) * PW + p) * PRE_WORDS + k) * 64 + lane
                    ] = fx.Int32(words0[k])
            # Wave 1 builds the head's e4m3 reciprocals while wave 0 runs the GEMM. The existing CTA barrier
            # publishes this table before any elected/owning tail can read it. Each call rebuilds it.
            if (wave == 1) & ((head == NQ) | (head == NQ + 1) | (head == NQ + 3)):
                gs = (head == NQ).select(
                    fx.Int32(pre0[7]).bitcast(fx.Float32),
                    (head == NQ + 1).select(
                        fx.Int32(pre0[8]).bitcast(fx.Float32), fx.Float32(1.0)
                    ),
                )
                for i in range_constexpr(4):
                    byte = lane + i * 64
                    sv = _e4m3_value(byte)
                    quant = (sv != 0.0).select(
                        fx.Float32(1.0) / (gs * sv), fx.Float32(0.0)
                    )
                    smem.reciprocal.ptr[byte] = quant.bitcast(fx.Int32)
                if lane == 0:
                    smem.reciprocal.ptr[256] = (fx.Float32(1.0) / gs).bitcast(fx.Int32)
        fx.gpu.barrier()
        # Positive weighted arrivals sum to EPOCH (16), so there is no top-up or reset. Owners wait
        # for the entire epoch, not merely the producer count; subtract the base modulo i32 at wrap.
        if tx == 0:
            inc = (kz == 0).select(fx.Int32(2), fx.Int32(1))
            old0 = _atomic_add(cnt_addr, inc)
            ebase = old0 - (old0 & (EPOCH - 1))
            if owner:
                arrived = old0 + inc - ebase
                while arrived < EPOCH:
                    arrived = _atomic_add(cnt_addr, 0) - ebase
        fx.gpu.barrier()
        if owner & (wave > 0):
            pres1 = []
            for p in range_constexpr(PW):
                got = [
                    fx.Int32(
                        smem.pre.ptr[
                            (((wave - 1) * PW + p) * PRE_WORDS + k) * 64 + lane
                        ]
                    )
                    for k in range_constexpr(PRE_WORDS)
                ]
                pres1.append(
                    [
                        got[0],
                        got[1],
                        Vec.from_elements(got[2:4], fx.Int32),
                        Vec.from_elements(got[4:6], fx.Int32),
                        Vec.from_elements(got[6:8], fx.Int32),
                        *got[8:],
                    ]
                )
            tail_post(
                slot_s,
                [row_base // 2 + (wave - 1) + p * (NWG - 1) for p in range(PW)],
                pres1,
            )

    @flyc.jit
    def launch_qkv_cooperative_satomic(
        arg_a: fx.Int64,
        arg_w: fx.Int64,
        arg_as: fx.Int64,
        arg_ws: fx.Int64,
        arg_slab: fx.Int64,
        arg_cnt: fx.Int64,
        arg_q: fx.Int64,
        arg_iq: fx.Int64,
        arg_qw: fx.Int64,
        arg_kw: fx.Int64,
        arg_iqw: fx.Int64,
        arg_ikw: fx.Int64,
        arg_cs: fx.Int64,
        arg_pos: fx.Int64,
        arg_slot: fx.Int64,
        arg_islot: fx.Int64,
        arg_kv: fx.Int64,
        arg_idx: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        i32_m: fx.Int32,
        a_scale_bytes: fx.Int32,
        i32_iq_st: fx.Int32,
        i64_kv_st: fx.Int64,
        i64_idx_st: fx.Int64,
        i32_cs_rows: fx.Int32,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_m,
            a_scale_bytes,
            arg_w,
            arg_ws,
            arg_as,
            arg_a,
            arg_slab,
            arg_cnt,
            i32_cs_rows,
            arg_pos,
            arg_slot,
            arg_islot,
            arg_cs,
            arg_qw,
            arg_kw,
            arg_iqw,
            arg_ikw,
            arg_q,
            arg_iq,
            arg_kv,
            arg_idx,
            arg_ks,
            arg_vs,
            i64_kv_st,
            i64_idx_st,
            i32_iq_st,
            i32_eps,
        ).launch(grid=(KS * NB, 1, 1), block=(64 * NWG, 1, 1), stream=stream)

    launch_qkv_cooperative_satomic.kernel_name = name
    launch_qkv_cooperative_satomic.compile_hints = PRELOAD_HINTS
    return launch_qkv_cooperative_satomic
