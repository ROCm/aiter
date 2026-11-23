# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SiLU-mul + group-32 MXFP8 quant of a ``[gate | up]`` bf16 GEMM output, as
the tail of the GEMM launch that wrote it (gfx950).

A GEMM whose N columns are ``[gate (N / 2) | up (N / 2)]`` keeps its kernel:
tiles, K loop, accumulators and bf16 stores are the tuned kernel's. Activation
column ``j`` needs gate column ``j`` and up column ``N / 2 + j``, which sit in
two different N tiles, i.e. two workgroups. Each of the two publishes its bf16
tile (the output buffer itself, stored at device scope), then counts its
arrival on the pair's counter; the second to arrive runs the activation and the
quant over the pair's rows and columns and re-arms the counter. Nobody waits.

The arithmetic is ``silu_mul_mxfp8_kernel<HAS_LIMIT = true>`` of
csrc/kernels/dsv41_seg_fuse.cu, operation for operation as that kernel is
compiled (a lane owns eight consecutive values, four adjacent lanes one group
of 32):

  clamp    gate = min(limit, gate) rounded to bf16; up = med3(-limit, up, limit)
  silu     exp(-gate) as the device library expands it -- p = gate * -log2(e),
           its error fused (fma) with the constant's low part, the integer
           part n = rint(p) removed by an fma, 2**(fraction) through the exp2
           instruction, ldexp by n, zero past the underflow bound -- then
           gate * rcp(1 + exp(-gate))
  product  with up, rounded to bf16
  quantize group amax (floored at 1e-10) by a quad DPP exchange; e8m0 code =
           ceil_pow2(amax / 448) as (bits + 0x7fffff) >> 23 clamped to
           [1, 254]; e4m3 through the scaled bf16 convert at 2**(code - 127).

That kernel runs with FP32 denormals flushed (inputs and results), a FlyDSL
kernel with them kept, and the two modes do not always give the same bytes
(a huge negative gate, whose exp2 argument is rounding noise, is one case).
The tail therefore first puts the wave's MODE register in the reference's
denormal mode; the GEMM and its bf16 rounding ran before, in the kernel's own.

The arithmetic helpers are trace-time only; the three with runtime control flow
(pair_arrival, silu_mul_quant_tile, pair_counter_rearm) are rewritten the way
a kernel body is, so a kernel calls them like its own nested functions.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_ops
from flydsl._mlir.dialects import rocdl as rocdl_ops
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import BFloat16, Float32, Int16, Int32, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.communication_ops_utils import atomic_add_agent

GROUP = 32
LANE_VALUES = 8  # values a lane owns: four adjacent lanes hold a group
# Buffer cache policy SC1 (device scope): the two tiles of a pair may run on
# different XCDs, whose L2s are not coherent with each other, so the bf16 tile
# is written through to, and read back from, the level they share.
SC1 = 16

_NEG_LOG2E_HI = 0xBFB8AA3B  # -log2(e), rounded
_NEG_LOG2E_LO = 0xB2A5705F  # -(log2(e) - its rounding)
_EXP_UNDERFLOW = 0x42CE8ED0  # exp(-x) is 0 past this x
_AMAX_FLOOR = 1e-10
_INV_448 = 0x3B124925  # 1 / 448
_DPP_SWAP_1 = 0xB1  # quad_perm [1, 0, 3, 2]
_DPP_SWAP_2 = 0x4E  # quad_perm [2, 3, 0, 1]
# hwreg(HW_REG_MODE = 1, offset 4, size 2): MODE's FP32 denormal field; 0
# flushes denormal inputs and results.
_HWREG_MODE_FP32_DENORM = 1 | (4 << 6) | ((2 - 1) << 11)


def silu_mul_quant_runs(N, tile_n, limit):
    """Whether an [*, N] = [gate | up] output in tile_n-wide tiles can take
    the epilogue: both halves split into whole tiles of whole 32-column
    groups, and a positive clamp (the reference's HAS_LIMIT form)."""
    return (
        N % 2 == 0
        and tile_n % GROUP == 0
        and (N // 2) % tile_n == 0
        and float(limit) > 0.0
    )


def _f32(bits):
    return fx.Int32(bits).bitcast(Float32)


def _raw(value):
    return value.ir_value() if hasattr(value, "ir_value") else value


def _exp_neg(x):
    """exp(-x), the device library's expansion with the negation folded into
    its constants."""
    c_hi = _f32(_NEG_LOG2E_HI)
    p = x * c_hi
    err = fmath.fma(x, c_hi, -p)
    n = fmath.roundeven(p)
    err = fmath.fma(_f32(_NEG_LOG2E_LO), x, err)
    frac = fmath.fma(x, c_hi, -n)
    power = fx.Float32(rocdl_ops.exp2(T.f32, _raw(frac + err)))
    scaled = fx.Float32(
        llvm_ops.intr_ldexp(T.f32, _raw(power), _raw(fx.Int32(n.to(Int32))))
    )
    return (x > _f32(_EXP_UNDERFLOW)).select(fx.Float32(0.0), scaled)


def flush_fp32_denormals():
    """From here on this wave runs FP32 with denormals flushed, the float
    mode silu_mul_mxfp8_kernel is launched in."""
    llvm_ops.call_intrinsic(
        None,
        "llvm.amdgcn.s.setreg",
        [fx.Int32(_HWREG_MODE_FP32_DENORM).ir_value(), fx.Int32(0).ir_value()],
        [],
        [],
    )


def silu_mul_bf16(gate_words, up_words, limit):
    """``gate_words`` / ``up_words``: a lane's eight bf16 gate / up values as
    four i32 (memory order). Returns the eight activation values rounded to
    bf16, as four i32 in the same order."""
    lim = fx.Float32(float(limit))
    neg = fx.Float32(-float(limit))
    out = []
    for w in range_constexpr(LANE_VALUES // 2):
        g = Vec.from_elements([gate_words[w]], Int32).bitcast(BFloat16).to(Float32)
        u = Vec.from_elements([up_words[w]], Int32).bitcast(BFloat16).to(Float32)
        # The clamped gate is rounded to bf16 before the SiLU reads it.
        x = (
            Vec.from_elements([fx.minnumf(lim, g[0]), fx.minnumf(lim, g[1])], Float32)
            .to(BFloat16)
            .to(Float32)
        )
        y = [
            fx.Float32(rocdl_ops.fmed3(T.f32, _raw(neg), _raw(u[e]), _raw(lim)))
            for e in range_constexpr(2)
        ]
        act = []
        for e in range_constexpr(2):
            den = fx.Float32(1.0) + _exp_neg(x[e])
            act.append(fx.Float32(rocdl_ops.rcp(T.f32, _raw(den))) * x[e])
        prod = Vec.from_elements(act, Float32) * Vec.from_elements(y, Float32)
        out.append(prod.to(BFloat16).bitcast(Int32)[0])
    return out


def group_code(act_words):
    """The biased e8m0 exponent (i32) of the group this lane's quad holds:
    its amax floored at 1e-10, over 448, rounded up to a power of two."""
    amax = fx.Float32(_AMAX_FLOOR)
    for w in range_constexpr(LANE_VALUES // 2):
        amax = fx.maxnumf(amax, ((act_words[w] & 0x7FFF) << 16).bitcast(Float32))
        amax = fx.maxnumf(amax, (act_words[w] & 0x7FFF0000).bitcast(Float32))
    for ctrl in (_DPP_SWAP_1, _DPP_SWAP_2):
        bits = amax.bitcast(Int32)
        peer = fx.Int32(
            rocdl_ops.update_dpp(T.i32, _raw(bits), _raw(bits), ctrl, 0xF, 0xF, False)
        )
        amax = fx.maxnumf(amax, peer.bitcast(Float32))
    bits = (amax * _f32(_INV_448)).bitcast(Int32)
    return fx.min(fx.max((bits + 0x7FFFFF).shrui(fx.Int32(23)), 1), 254)


def quant_words(act_words, code):
    """The lane's eight e4m3 bytes (two i32, memory order) at the group's
    exponent ``code``."""
    scale = (fx.Int32(code) << 23).bitcast(Float32)
    pair_ty = T.vec(2, T.i16)
    words = []
    for h in range_constexpr(2):
        pair = fx.full(2, 0, Int16).ir_value()
        for sel in range_constexpr(2):
            src = Vec.from_elements([act_words[2 * h + sel]], Int32).bitcast(BFloat16)
            pair = rocdl_ops.cvt_scalef32_pk_fp8_bf16(
                pair_ty, pair, _raw(src), _raw(scale), sel == 1
            )
        words.append(fx.Vector(pair).bitcast(Int32)[0])
    return words


def pair_arrival(tid, counter_addr, lds_flag):
    """Count this workgroup's arrival on its pair's counter (int32 at
    ``counter_addr``) once every wave's bf16 stores have drained, and return
    the count before it, the same in every lane (``lds_flag``: an i32 LDS slot
    free at this point)."""
    rocdl.s_waitcnt(0)
    gpu.barrier()
    if tid == 0:
        fx.ptr_store(fx.Int32(atomic_add_agent(counter_addr, fx.Int32(1))), lds_flag)
    gpu.barrier()
    return fx.Int32(fx.ptr_load(lds_flag))


def silu_mul_quant_tile(
    tid,
    threads,
    c_addr,
    out_addr,
    scale_addr,
    i32_m,
    row0,
    col0,
    N,
    tile_m,
    tile_n,
    limit,
):
    """The activation and quant of rows [row0, row0 + tile_m) (those below
    ``i32_m``) and activation columns [col0, col0 + tile_n), read from the
    bf16 ``[M, N]`` at ``c_addr``: e4m3 into ``out_addr`` ([M, N / 2]) and the
    e8m0 exponents into ``scale_addr`` ([M, N / 64], row-major). A lane takes
    eight consecutive columns of a row; the four lanes of a group are a DPP
    quad, live or idle together. Leaves the wave in the reference's float
    mode (flush_fp32_denormals): call it last."""
    half = N // 2
    chunks = tile_n // LANE_VALUES
    assert chunks % 4 == 0 and threads % 64 == 0
    rows = i32_m - row0
    rows = (rows < tile_m).select(rows, fx.Int32(tile_m))
    c_ptr_ty = fx.PointerType.get(T.i32, fx.AddressSpace.Global, 16)
    c_view = fx.make_view(
        fx.inttoptr(c_ptr_ty, c_addr + fx.Int64(row0) * (N * 2)),
        fx.make_layout(1 << 28, 1),
    )
    c_src = fx.logical_divide(
        fx.rocdl.make_buffer_tensor(
            fx.Tensor(c_view),
            max_size=False,
            num_records_bytes=fx.Int64(rows) * (N * 2),
        ),
        fx.make_layout(1, 1),
    )
    load16 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(SC1), 32)
    i32_ty = fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4)
    i8_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Global, 1)

    def _load(dword):
        frag = fx.make_rmem_tensor(4, Int32)
        fx.copy(load16, fx.slice(c_src, (None, dword)), frag)
        vec = frag.load()
        return [vec[j] for j in range_constexpr(4)]

    flush_fp32_denormals()
    for i in range_constexpr(-(-(tile_m * chunks) // threads)):
        idx = i * threads + tid
        row = idx // chunks
        col = col0 + (idx % chunks) * LANE_VALUES
        if row < rows:
            base = row * (N // 2) + col // 2  # dwords from the tile's first row
            gate = _load(base)
            up = _load(base + half // 2)
            act = silu_mul_bf16(gate, up, limit)
            code = group_code(act)
            words = quant_words(act, code)
            grow = fx.Int64(row0 + row)
            out_ptr = fx.inttoptr(i32_ty, out_addr + grow * half + fx.Int64(col))
            fx.ptr_store(words[0], out_ptr)
            fx.ptr_store(words[1], fx.add_offset(out_ptr, 1))
            # The quad's first lane stores the group's exponent.
            if idx % 4 == 0:
                fx.ptr_store(
                    fx.Int8(code),
                    fx.inttoptr(
                        i8_ty,
                        scale_addr + grow * (half // GROUP) + fx.Int64(col // GROUP),
                    ),
                )


def pair_counter_rearm(tid, counter_addr):
    """The pair's second arrival leaves its counter at zero for the next
    launch."""
    if tid == 0:
        fx.ptr_store(
            fx.Int32(0),
            fx.inttoptr(
                fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), counter_addr
            ),
        )


pair_arrival = ASTRewriter.transform(pair_arrival)
silu_mul_quant_tile = ASTRewriter.transform(silu_mul_quant_tile)
pair_counter_rearm = ASTRewriter.transform(pair_counter_rearm)
