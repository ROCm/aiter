# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# gfx950-only BF16 warp-per-scalar GEMM for autoregressive decode.
#
# C[M, N] = A[M, K] @ B[N, K]^T   (B row-major weight matrix)
#
# One wavefront computes a compile-time MxN register tile:
#   - 64 lanes split K, each owns KVEC adjacent BF16 elements per full iteration
#   - Accumulate via v_dot2_f32_bf16 (2 BF16 MACs -> FP32)
#   - A and B vectors are reused across the register tile
#   - A separate predicated scalar tail handles K outside the vectorized loop
#   - Partial output-column tiles use safe B loads and conditional stores
#   - 6-stage butterfly XOR reduce per accumulator (no LDS)
#   - Lane 0 converts FP32 to BF16 with configurable compile-time rounding

from dataclasses import dataclass
from enum import Enum

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _arith_dialect
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops, gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec


class OutputRounding(str, Enum):
    """Compile-time FP32-to-BF16 conversion choices."""

    RNE = "rne"
    RTZ = "rtz"
    TRUNCATE = "truncate"
    STOCHASTIC = "stochastic"


@dataclass(frozen=True)
class GemmDecodeConfig:
    """Named compile-time defaults for the direct kernel."""

    kvec: int = 8
    n_per_wave: int = 2
    n_per_wave_m1: int = 1
    m_per_wave: int = 4
    waves_per_eu: int = 2
    b_cache_modifier: int = 0x2000
    output_rounding: OutputRounding = OutputRounding.RNE

    def validate(self) -> None:
        if self.kvec <= 0 or self.kvec % 4 != 0:
            raise ValueError("kvec must be a positive multiple of 4")
        if self.n_per_wave != 2 or self.n_per_wave_m1 != 1:
            raise ValueError("this kernel mapping requires N tiles of 2 (1 for M=1)")
        if self.m_per_wave != 4:
            raise ValueError("this kernel mapping requires an M tile of 4")
        if self.waves_per_eu <= 0:
            raise ValueError("waves_per_eu must be positive")


DEFAULT_CONFIG = GemmDecodeConfig()
DEFAULT_CONFIG.validate()
KVEC = DEFAULT_CONFIG.kvec
NP = DEFAULT_CONFIG.n_per_wave
MP = DEFAULT_CONFIG.m_per_wave
WAVES_PER_EU = DEFAULT_CONFIG.waves_per_eu
B_CACHE_MODIFIER = DEFAULT_CONFIG.b_cache_modifier
OUTPUT_ROUNDING = DEFAULT_CONFIG.output_rounding


# -- helpers -------------------------------------------------------------------


def _to_ir(v):
    """Unwrap FlyDSL value to raw MLIR ir.Value."""
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _const_f32(val: float) -> ir.Value:
    """Create a constant FP32 MLIR value."""
    return _arith_dialect.ConstantOp(
        ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), val)
    ).result


def _const_bf16(val: float) -> ir.Value:
    """Create a constant BF16 MLIR value."""
    return _arith_dialect.ConstantOp(
        ir.BF16Type.get(), ir.FloatAttr.get(ir.BF16Type.get(), val)
    ).result


def _add_f32(a: ir.Value, b) -> ir.Value:
    """FP32 addition: a + b."""
    return _arith_dialect.AddFOp(lhs=a, rhs=_to_ir(b)).result


def dot2_f32_bf16(acc: ir.Value, a_i32: ir.Value, b_i32: ir.Value) -> ir.Value:
    """acc += a.lo*b.lo + a.hi*b.hi  (2 BF16 MACs -> FP32 accumulator)."""
    return _llvm.inline_asm(
        ir.F32Type.get(),
        [acc, _to_ir(a_i32), _to_ir(b_i32)],
        "v_dot2_f32_bf16 $0, $2, $3, $1",
        "=v,0,v,v",
        has_side_effects=False,
    )


def pack_bf16x2(lo, hi) -> ir.Value:
    """Pack two BF16 scalars into one i32 (lo in bits[15:0], hi in bits[31:16])."""
    lo_i16 = ArithValue(_to_ir(lo)).bitcast(T.i16)
    hi_i16 = ArithValue(_to_ir(hi)).bitcast(T.i16)
    lo_i32 = ArithValue(lo_i16).extui(T.i32)
    hi_i32 = ArithValue(hi_i16).extui(T.i32)
    return ArithValue(lo_i32) | (ArithValue(hi_i32) << fx.Int32(16))


def _stochastic_seed(c_elem):
    """Deterministic per-output seed for repeatable stochastic conversion."""
    x = ArithValue(_to_ir(c_elem))
    return (x * fx.Int32(0x45D9F3B)) ^ (x << fx.Int32(16)) ^ fx.Int32(0x27D4EB2D)


def _convert_bf16(acc: ir.Value, c_elem, rounding: OutputRounding) -> ir.Value:
    """Convert one FP32 value to BF16 with a compile-time rounding mode."""
    if rounding == OutputRounding.RNE:
        return _arith_dialect.TruncFOp(ir.BF16Type.get(), acc).result
    if rounding == OutputRounding.RTZ:
        acc_i32 = ArithValue(acc).bitcast(T.i32)
        bf16_i32 = ArithValue(acc_i32).shrui(fx.Int32(16))
        abs_i32 = ArithValue(acc_i32) & fx.Int32(0x7FFFFFFF)
        is_nan = abs_i32 > fx.Int32(0x7F800000)
        quiet_nan = ArithValue(bf16_i32) | fx.Int32(0x40)
        bf16_i32 = ArithValue(_to_ir(is_nan)).select(quiet_nan, bf16_i32)
    elif rounding == OutputRounding.STOCHASTIC:
        bf16_i32 = _llvm.inline_asm(
            ir.IntegerType.get_signless(32),
            [acc, _to_ir(_stochastic_seed(c_elem))],
            "v_cvt_sr_bf16_f32 $0, $1, $2",
            "=v,v,v",
            has_side_effects=False,
        )
    elif rounding == OutputRounding.TRUNCATE:
        acc_i32 = ArithValue(acc).bitcast(T.i32)
        bf16_i32 = ArithValue(acc_i32).shrui(fx.Int32(16))
    else:
        raise ValueError(f"unsupported output rounding: {rounding}")

    bf16_i16 = ArithValue(_to_ir(bf16_i32)).trunci(T.i16)
    return ArithValue(bf16_i16).bitcast(T.bf16)


def store_bf16(
    acc: ir.Value,
    rsrc_c,
    c_elem,
    rounding: OutputRounding,
) -> None:
    """Convert FP32 accumulator to BF16 and store to C."""
    bf16_val = _convert_bf16(acc, c_elem, rounding)
    buffer_ops.buffer_store(bf16_val, rsrc_c, c_elem)


def wavefront_reduce_sum_f32(val: ir.Value, lane) -> ir.Value:
    """6-stage butterfly XOR reduce. Returns the full sum in all 64 lanes."""
    for stage in range_constexpr(6):
        partner = lane ^ fx.Int32(1 << stage)
        val_i32 = ArithValue(val).bitcast(T.i32)
        peer_i32 = fx.rocdl.ds_bpermute(T.i32, partner * fx.Int32(4), val_i32)
        val = _add_f32(val, ArithValue(peer_i32).bitcast(T.f32))
    return val


def load_kvec_a(rsrc, base_elem):
    """Load the fixed default K vector through L2."""
    return tuple(
        Vec(
            buffer_ops.buffer_load(
                rsrc,
                base_elem + fx.Int32(i * 4),
                vec_width=4,
                dtype=T.bf16,
            )
        )
        for i in range(KVEC // 4)
    )


def load_kvec_b(rsrc, base_elem):
    """Load the fixed default K vector with the configured B cache policy."""
    return tuple(
        Vec(
            buffer_ops.buffer_load(
                rsrc,
                base_elem + fx.Int32(i * 4),
                vec_width=4,
                dtype=T.bf16,
                cache_modifier=B_CACHE_MODIFIER,
            )
        )
        for i in range(KVEC // 4)
    )


def _load_tail_pair(
    rsrc,
    row,
    K,
    tail_lane_base,
    pair,
    cache_modifier=0,
    row_valid=None,
):
    """Load one masked BF16 pair from the scalar K tail."""
    k0 = tail_lane_base + fx.Int32(pair * 2)
    k1 = k0 + fx.Int32(1)
    valid0 = k0 < fx.Int32(K)
    valid1 = k1 < fx.Int32(K)
    safe_row = row
    if row_valid is not None:
        valid0 = valid0 & row_valid
        valid1 = valid1 & row_valid
        safe_row = ArithValue(_to_ir(row_valid)).select(row, fx.Int32(0))
    safe_k0 = ArithValue(_to_ir(valid0)).select(k0, fx.Int32(0))
    safe_k1 = ArithValue(_to_ir(valid1)).select(k1, fx.Int32(0))
    row_base = ArithValue(_to_ir(safe_row)) * fx.Int32(K)
    lo = buffer_ops.buffer_load(
        rsrc,
        row_base + safe_k0,
        vec_width=1,
        dtype=T.bf16,
        cache_modifier=cache_modifier,
    )
    hi = buffer_ops.buffer_load(
        rsrc,
        row_base + safe_k1,
        vec_width=1,
        dtype=T.bf16,
        cache_modifier=cache_modifier,
    )
    zero = _const_bf16(0.0)
    lo = ArithValue(_to_ir(valid0)).select(lo, zero)
    hi = ArithValue(_to_ir(valid1)).select(hi, zero)
    return pack_bf16x2(lo, hi)


def _dots_multi(M, lane, m_base, n_base, N, K, rsrc_a, rsrc_b):
    """Original M=2..4 hot loop plus a separately compiled scalar K tail."""
    k_tile = 64 * KVEC
    nv = KVEC // 4
    acc00 = _const_f32(0.0)
    acc01 = _const_f32(0.0)
    acc10 = _const_f32(0.0)
    acc11 = _const_f32(0.0)
    acc20 = _const_f32(0.0)
    acc21 = _const_f32(0.0)
    acc30 = _const_f32(0.0)
    acc31 = _const_f32(0.0)
    m0 = m_base
    m1 = m_base + fx.Int32(1)
    m2 = m_base + fx.Int32(2)
    m3 = m_base + fx.Int32(3)
    n0 = n_base
    n1 = n_base + fx.Int32(1)
    n1_valid = None
    n1_load = n1
    if N % 2 != 0:
        n1_valid = n1 < fx.Int32(N)
        n1_load = ArithValue(_to_ir(n1_valid)).select(n1, fx.Int32(0))

    for i in range_constexpr(K // k_tile):
        k_elem = fx.Int32(i * k_tile) + lane * fx.Int32(KVEC)
        av0 = load_kvec_a(rsrc_a, m0 * fx.Int32(K) + k_elem)
        if M > 1:
            av1 = load_kvec_a(rsrc_a, m1 * fx.Int32(K) + k_elem)
        if M > 2:
            av2 = load_kvec_a(rsrc_a, m2 * fx.Int32(K) + k_elem)
        if M > 3:
            av3 = load_kvec_a(rsrc_a, m3 * fx.Int32(K) + k_elem)
        bv0 = load_kvec_b(rsrc_b, n0 * fx.Int32(K) + k_elem)
        bv1 = load_kvec_b(rsrc_b, n1_load * fx.Int32(K) + k_elem)
        for v in range_constexpr(nv):
            p0a = pack_bf16x2(av0[v][0], av0[v][1])
            p0b = pack_bf16x2(av0[v][2], av0[v][3])
            q0a = pack_bf16x2(bv0[v][0], bv0[v][1])
            q0b = pack_bf16x2(bv0[v][2], bv0[v][3])
            q1a = pack_bf16x2(bv1[v][0], bv1[v][1])
            q1b = pack_bf16x2(bv1[v][2], bv1[v][3])
            acc00 = dot2_f32_bf16(acc00, p0a, q0a)
            acc01 = dot2_f32_bf16(acc01, p0a, q1a)
            acc00 = dot2_f32_bf16(acc00, p0b, q0b)
            acc01 = dot2_f32_bf16(acc01, p0b, q1b)
            if M > 1:
                p1a = pack_bf16x2(av1[v][0], av1[v][1])
                p1b = pack_bf16x2(av1[v][2], av1[v][3])
                acc10 = dot2_f32_bf16(acc10, p1a, q0a)
                acc11 = dot2_f32_bf16(acc11, p1a, q1a)
                acc10 = dot2_f32_bf16(acc10, p1b, q0b)
                acc11 = dot2_f32_bf16(acc11, p1b, q1b)
            if M > 2:
                p2a = pack_bf16x2(av2[v][0], av2[v][1])
                p2b = pack_bf16x2(av2[v][2], av2[v][3])
                acc20 = dot2_f32_bf16(acc20, p2a, q0a)
                acc21 = dot2_f32_bf16(acc21, p2a, q1a)
                acc20 = dot2_f32_bf16(acc20, p2b, q0b)
                acc21 = dot2_f32_bf16(acc21, p2b, q1b)
            if M > 3:
                p3a = pack_bf16x2(av3[v][0], av3[v][1])
                p3b = pack_bf16x2(av3[v][2], av3[v][3])
                acc30 = dot2_f32_bf16(acc30, p3a, q0a)
                acc31 = dot2_f32_bf16(acc31, p3a, q1a)
                acc30 = dot2_f32_bf16(acc30, p3b, q0b)
                acc31 = dot2_f32_bf16(acc31, p3b, q1b)

    if K % k_tile != 0:
        tail_lane_base = fx.Int32((K // k_tile) * k_tile) + lane * fx.Int32(KVEC)
        for pair in range_constexpr(KVEC // 2):
            p0 = _load_tail_pair(rsrc_a, m0, K, tail_lane_base, pair)
            if M > 1:
                p1 = _load_tail_pair(rsrc_a, m1, K, tail_lane_base, pair)
            if M > 2:
                p2 = _load_tail_pair(rsrc_a, m2, K, tail_lane_base, pair)
            if M > 3:
                p3 = _load_tail_pair(rsrc_a, m3, K, tail_lane_base, pair)
            q0 = _load_tail_pair(
                rsrc_b,
                n0,
                K,
                tail_lane_base,
                pair,
                cache_modifier=B_CACHE_MODIFIER,
            )
            q1 = _load_tail_pair(
                rsrc_b,
                n1,
                K,
                tail_lane_base,
                pair,
                cache_modifier=B_CACHE_MODIFIER,
                row_valid=n1_valid,
            )
            acc00 = dot2_f32_bf16(acc00, p0, q0)
            acc01 = dot2_f32_bf16(acc01, p0, q1)
            if M > 1:
                acc10 = dot2_f32_bf16(acc10, p1, q0)
                acc11 = dot2_f32_bf16(acc11, p1, q1)
            if M > 2:
                acc20 = dot2_f32_bf16(acc20, p2, q0)
                acc21 = dot2_f32_bf16(acc21, p2, q1)
            if M > 3:
                acc30 = dot2_f32_bf16(acc30, p3, q0)
                acc31 = dot2_f32_bf16(acc31, p3, q1)

    return (
        acc00,
        acc01,
        acc10,
        acc11,
        acc20,
        acc21,
        acc30,
        acc31,
        m0,
        m1,
        m2,
        m3,
        n0,
        n1,
        n1_valid,
    )


def _make_kernel_multi(M):
    @flyc.kernel
    def kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        K: fx.Constexpr[int],
        N: fx.Constexpr[int],
    ):
        lane = gpu.thread_idx.x
        m_base = gpu.block_idx.x * fx.Int32(MP)
        n_base = gpu.block_idx.y * fx.Int32(2)
        rsrc_a = buffer_ops.create_buffer_resource(A)
        rsrc_b = buffer_ops.create_buffer_resource(B)
        rsrc_c = buffer_ops.create_buffer_resource(C)
        (
            acc00,
            acc01,
            acc10,
            acc11,
            acc20,
            acc21,
            acc30,
            acc31,
            m0,
            m1,
            m2,
            m3,
            n0,
            n1,
            n1_valid,
        ) = _dots_multi(
            M,
            lane,
            m_base,
            n_base,
            N,
            K,
            rsrc_a,
            rsrc_b,
        )
        acc00 = wavefront_reduce_sum_f32(acc00, lane)
        acc01 = wavefront_reduce_sum_f32(acc01, lane)
        if M > 1:
            acc10 = wavefront_reduce_sum_f32(acc10, lane)
            acc11 = wavefront_reduce_sum_f32(acc11, lane)
        if M > 2:
            acc20 = wavefront_reduce_sum_f32(acc20, lane)
            acc21 = wavefront_reduce_sum_f32(acc21, lane)
        if M > 3:
            acc30 = wavefront_reduce_sum_f32(acc30, lane)
            acc31 = wavefront_reduce_sum_f32(acc31, lane)
        if lane == fx.Int32(0):
            store_bf16(acc00, rsrc_c, m0 * fx.Int32(N) + n0, OUTPUT_ROUNDING)
            if n1_valid is None:
                store_bf16(acc01, rsrc_c, m0 * fx.Int32(N) + n1, OUTPUT_ROUNDING)
            elif n1_valid:
                store_bf16(acc01, rsrc_c, m0 * fx.Int32(N) + n1, OUTPUT_ROUNDING)
            if M > 1:
                store_bf16(acc10, rsrc_c, m1 * fx.Int32(N) + n0, OUTPUT_ROUNDING)
                if n1_valid is None:
                    store_bf16(acc11, rsrc_c, m1 * fx.Int32(N) + n1, OUTPUT_ROUNDING)
                elif n1_valid:
                    store_bf16(acc11, rsrc_c, m1 * fx.Int32(N) + n1, OUTPUT_ROUNDING)
            if M > 2:
                store_bf16(acc20, rsrc_c, m2 * fx.Int32(N) + n0, OUTPUT_ROUNDING)
                if n1_valid is None:
                    store_bf16(acc21, rsrc_c, m2 * fx.Int32(N) + n1, OUTPUT_ROUNDING)
                elif n1_valid:
                    store_bf16(acc21, rsrc_c, m2 * fx.Int32(N) + n1, OUTPUT_ROUNDING)
            if M > 3:
                store_bf16(acc30, rsrc_c, m3 * fx.Int32(N) + n0, OUTPUT_ROUNDING)
                if n1_valid is None:
                    store_bf16(acc31, rsrc_c, m3 * fx.Int32(N) + n1, OUTPUT_ROUNDING)
                elif n1_valid:
                    store_bf16(acc31, rsrc_c, m3 * fx.Int32(N) + n1, OUTPUT_ROUNDING)

    return kernel


def _make_kernel_m1():
    @flyc.kernel
    def kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        K: fx.Constexpr[int],
        N: fx.Constexpr[int],
    ):
        lane = gpu.thread_idx.x
        m = gpu.block_idx.x
        n = gpu.block_idx.y
        rsrc_a = buffer_ops.create_buffer_resource(A)
        rsrc_b = buffer_ops.create_buffer_resource(B)
        rsrc_c = buffer_ops.create_buffer_resource(C)
        acc = _const_f32(0.0)
        k_tile = 64 * KVEC
        nv = KVEC // 4
        for i in range_constexpr(K // k_tile):
            k_elem = fx.Int32(i * k_tile) + lane * fx.Int32(KVEC)
            av = load_kvec_a(rsrc_a, m * fx.Int32(K) + k_elem)
            bv = load_kvec_b(rsrc_b, n * fx.Int32(K) + k_elem)
            for v in range_constexpr(nv):
                acc = dot2_f32_bf16(
                    acc,
                    pack_bf16x2(av[v][0], av[v][1]),
                    pack_bf16x2(bv[v][0], bv[v][1]),
                )
                acc = dot2_f32_bf16(
                    acc,
                    pack_bf16x2(av[v][2], av[v][3]),
                    pack_bf16x2(bv[v][2], bv[v][3]),
                )
        if K % k_tile != 0:
            tail_lane_base = fx.Int32((K // k_tile) * k_tile) + lane * fx.Int32(KVEC)
            for pair in range_constexpr(KVEC // 2):
                av = _load_tail_pair(rsrc_a, m, K, tail_lane_base, pair)
                bv = _load_tail_pair(
                    rsrc_b,
                    n,
                    K,
                    tail_lane_base,
                    pair,
                    cache_modifier=B_CACHE_MODIFIER,
                )
                acc = dot2_f32_bf16(acc, av, bv)
        acc = wavefront_reduce_sum_f32(acc, lane)
        if lane == fx.Int32(0):
            c_elem = m * fx.Int32(N) + n
            store_bf16(acc, rsrc_c, c_elem, OUTPUT_ROUNDING)

    return kernel


gemm_decode_bf16_kernel_m1 = _make_kernel_m1()
gemm_decode_bf16_kernel_m2 = _make_kernel_multi(2)
gemm_decode_bf16_kernel_m3 = _make_kernel_multi(3)
gemm_decode_bf16_kernel_m4 = _make_kernel_multi(4)


@flyc.jit
def gemm_decode_bf16(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    M: fx.Constexpr[int],
    N: fx.Constexpr[int],
    K: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    if M < 1 or M > 4:
        raise ValueError("gemm_decode_bf16 supports M in [1, 4]")
    if N <= 0 or K <= 0:
        raise ValueError("gemm_decode_bf16 requires positive N and K")
    attrs = {"rocdl.waves_per_eu": WAVES_PER_EU}
    if M == 1:
        gemm_decode_bf16_kernel_m1(A, B, C, K, N).launch(
            grid=(1, N, 1),
            block=(64, 1, 1),
            stream=stream,
            value_attrs=attrs,
        )
    elif M == 2:
        gemm_decode_bf16_kernel_m2(A, B, C, K, N).launch(
            grid=(1, (N + NP - 1) // NP, 1),
            block=(64, 1, 1),
            stream=stream,
            value_attrs=attrs,
        )
    elif M == 3:
        gemm_decode_bf16_kernel_m3(A, B, C, K, N).launch(
            grid=(1, (N + NP - 1) // NP, 1),
            block=(64, 1, 1),
            stream=stream,
            value_attrs=attrs,
        )
    else:
        gemm_decode_bf16_kernel_m4(A, B, C, K, N).launch(
            grid=(1, (N + NP - 1) // NP, 1),
            block=(64, 1, 1),
            stream=stream,
            value_attrs=attrs,
        )
