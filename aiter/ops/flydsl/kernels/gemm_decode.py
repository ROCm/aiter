# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# ===========================================================================
#
# WHAT THIS VERSION IS:
#   BF16 warp-per-scalar GEMM with all non-pipeline optimizations applied.
#   This is the current best working version before software pipelining.
#
# OPTIMIZATIONS APPLIED (vs Phase 1 baseline KVEC=8, NP=1):
#   KVEC=8        : 64-bit loads per lane per K-iter
#   NP=2          : A-reuse — one wavefront computes 2 adjacent output columns
#                   A[m,:] loaded once, reused for B[n,:] AND B[n+1,:]
#                   -> grid shrinks by 2x in N direction
#   Constexpr M   : separate kernel per M=1,2,3,4 — no runtime clamping/branches
#   B bypass L2   : 0x2000 cache modifier — B (234MB) too large for L2
#                   A (<=57KB) goes through L2 and benefits from caching
# ===========================================================================
#
# gemm_decode: warp-per-scalar small-M GEMM for LLM autoregressive decode.
#
# C[M, N] = A[M, K] @ B[N, K]^T   (B row-major weight matrix)
#
# One wavefront (64 threads) computes MPxNP output scalars C[m..m+MP-1, n..n+NP-1]:
#   - 64 lanes split K, each owns KVEC=8 BF16 elements per K-iteration
#   - Accumulate via v_dot2_f32_bf16 (2 BF16 MACs -> FP32)
#   - A rows loaded once, reused across NP=2 B columns (A-reuse)
#   - B loaded with L2 bypass (0x2000) — 234MB too large for L2
#   - 6-stage butterfly XOR reduce per accumulator (no LDS)
#   - Lane 0 stores MPxNP FP32->BF16 results to C
#   - Separate kernel per M=1,2,3,4 (Constexpr M — no clamping)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _arith_dialect
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops, gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

KVEC = 8  # BF16 elements per lane per K-iteration (K must be divisible by 64*KVEC)
NP = 2  # output columns per wavefront (N must be divisible by NP)
MP = 4  # output rows per wavefront; one wavefront computes MPxNP outputs
WAVES_PER_EU = 2  # rocdl hint: 2 waves per SIMD — standard for memory-bound kernels


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


def store_bf16(acc: ir.Value, rsrc_c, c_elem) -> None:
    """Convert FP32 accumulator to BF16 and store to C."""
    acc_i32 = ArithValue(acc).bitcast(T.i32)
    bf16_i32 = ArithValue(acc_i32).shrui(fx.Int32(16))
    bf16_i16 = ArithValue(_to_ir(bf16_i32)).trunci(T.i16)
    bf16_val = ArithValue(bf16_i16).bitcast(T.bf16)
    buffer_ops.buffer_store(bf16_val, rsrc_c, c_elem)


def wavefront_reduce_sum_f32(val: ir.Value, lane) -> ir.Value:
    """6-stage butterfly XOR reduce. Returns full sum in all 64 lanes."""
    for stage in range_constexpr(6):
        partner = lane ^ fx.Int32(1 << stage)
        val_i32 = ArithValue(val).bitcast(T.i32)
        peer_i32 = fx.rocdl.ds_bpermute(T.i32, partner * fx.Int32(4), val_i32)
        val = _add_f32(val, ArithValue(peer_i32).bitcast(T.f32))
    return val


def load_kvec_a(rsrc, base_elem):
    """Load KVEC BF16 values from A — through L2 (small matrix, reused)."""
    return tuple(
        Vec(
            buffer_ops.buffer_load(
                rsrc, base_elem + fx.Int32(i * 4), vec_width=4, dtype=T.bf16
            )
        )
        for i in range(KVEC // 4)
    )


def load_kvec_b(rsrc, base_elem):
    """Load KVEC BF16 values from B — bypass L2 (0x2000), large matrix."""
    return tuple(
        Vec(
            buffer_ops.buffer_load(
                rsrc,
                base_elem + fx.Int32(i * 4),
                vec_width=4,
                dtype=T.bf16,
                cache_modifier=0x2000,
            )
        )
        for i in range(KVEC // 4)
    )


def dot2_kvec(acc: ir.Value, av0, av1, av2, av3, bv0, bv1, bv2, bv3) -> ir.Value:
    """4 x dot2 covering KVEC=8 BF16 elements. acc += dot(a, b)."""
    acc = dot2_f32_bf16(acc, pack_bf16x2(av0[0], av0[1]), pack_bf16x2(bv0[0], bv0[1]))
    acc = dot2_f32_bf16(acc, pack_bf16x2(av0[2], av0[3]), pack_bf16x2(bv0[2], bv0[3]))
    acc = dot2_f32_bf16(acc, pack_bf16x2(av1[0], av1[1]), pack_bf16x2(bv1[0], bv1[1]))
    acc = dot2_f32_bf16(acc, pack_bf16x2(av1[2], av1[3]), pack_bf16x2(bv1[2], bv1[3]))
    acc = dot2_f32_bf16(acc, pack_bf16x2(av2[0], av2[1]), pack_bf16x2(bv2[0], bv2[1]))
    acc = dot2_f32_bf16(acc, pack_bf16x2(av2[2], av2[3]), pack_bf16x2(bv2[2], bv2[3]))
    acc = dot2_f32_bf16(acc, pack_bf16x2(av3[0], av3[1]), pack_bf16x2(bv3[0], bv3[1]))
    acc = dot2_f32_bf16(acc, pack_bf16x2(av3[2], av3[3]), pack_bf16x2(bv3[2], bv3[3]))
    return acc


# -- GPU kernels: one specialization per M value -------------------------------
# M known at compile time -> no clamping, no branches, no dead loads/stores.


def _dots(M, lane, m_base, n_base, K, nv, kTileN, rsrc_a, rsrc_b):
    """Pure dot-product loop, returns accumulators. M is a Python int (Constexpr)."""
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
    for i in range_constexpr(K // kTileN):
        k_elem = fx.Int32(i * kTileN) + lane * fx.Int32(KVEC)
        av0 = load_kvec_a(rsrc_a, m0 * fx.Int32(K) + k_elem)
        if M > 1:
            av1 = load_kvec_a(rsrc_a, m1 * fx.Int32(K) + k_elem)
        if M > 2:
            av2 = load_kvec_a(rsrc_a, m2 * fx.Int32(K) + k_elem)
        if M > 3:
            av3 = load_kvec_a(rsrc_a, m3 * fx.Int32(K) + k_elem)
        bv0 = load_kvec_b(rsrc_b, n_base * fx.Int32(K) + k_elem)
        bv1 = load_kvec_b(rsrc_b, (n_base + fx.Int32(1)) * fx.Int32(K) + k_elem)
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
    return acc00, acc01, acc10, acc11, acc20, acc21, acc30, acc31, m0, m1, m2, m3


def _make_kern(M_val):
    @flyc.kernel
    def kern(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        K: fx.Constexpr[int],
        N: fx.Constexpr[int],
    ):
        lane = gpu.thread_idx.x
        m_base = gpu.block_idx.x * fx.Int32(MP)
        n_base = gpu.block_idx.y * fx.Int32(NP)
        rsrc_a = buffer_ops.create_buffer_resource(A)
        rsrc_b = buffer_ops.create_buffer_resource(B)
        rsrc_c = buffer_ops.create_buffer_resource(C)
        kTileN = 64 * KVEC
        nv = KVEC // 4
        acc00, acc01, acc10, acc11, acc20, acc21, acc30, acc31, m0, m1, m2, m3 = _dots(
            M_val, lane, m_base, n_base, K, nv, kTileN, rsrc_a, rsrc_b
        )
        acc00 = wavefront_reduce_sum_f32(acc00, lane)
        acc01 = wavefront_reduce_sum_f32(acc01, lane)
        if M_val > 1:
            acc10 = wavefront_reduce_sum_f32(acc10, lane)
            acc11 = wavefront_reduce_sum_f32(acc11, lane)
        if M_val > 2:
            acc20 = wavefront_reduce_sum_f32(acc20, lane)
            acc21 = wavefront_reduce_sum_f32(acc21, lane)
        if M_val > 3:
            acc30 = wavefront_reduce_sum_f32(acc30, lane)
            acc31 = wavefront_reduce_sum_f32(acc31, lane)
        if lane == fx.Int32(0):
            store_bf16(acc00, rsrc_c, m0 * fx.Int32(N) + n_base)
            store_bf16(acc01, rsrc_c, m0 * fx.Int32(N) + n_base + fx.Int32(1))
            if M_val > 1:
                store_bf16(acc10, rsrc_c, m1 * fx.Int32(N) + n_base)
                store_bf16(acc11, rsrc_c, m1 * fx.Int32(N) + n_base + fx.Int32(1))
            if M_val > 2:
                store_bf16(acc20, rsrc_c, m2 * fx.Int32(N) + n_base)
                store_bf16(acc21, rsrc_c, m2 * fx.Int32(N) + n_base + fx.Int32(1))
            if M_val > 3:
                store_bf16(acc30, rsrc_c, m3 * fx.Int32(N) + n_base)
                store_bf16(acc31, rsrc_c, m3 * fx.Int32(N) + n_base + fx.Int32(1))

    return kern


gemm_decode_bf16_kernel_m2 = _make_kern(2)
gemm_decode_bf16_kernel_m3 = _make_kern(3)
gemm_decode_bf16_kernel_m4 = _make_kern(4)


# -- M=1: NP=1, MP=1, grid=(1,N,1) -------------------------------------------
# Single output per wavefront — no A-reuse needed, simpler than MP=4 kernel.


@flyc.kernel
def gemm_decode_bf16_kernel_m1(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    K: fx.Constexpr[int],
    N: fx.Constexpr[int],
):
    lane = gpu.thread_idx.x
    m = gpu.block_idx.x  # always 0 for M=1
    n = gpu.block_idx.y
    rsrc_a = buffer_ops.create_buffer_resource(A)
    rsrc_b = buffer_ops.create_buffer_resource(B)
    rsrc_c = buffer_ops.create_buffer_resource(C)
    acc = _const_f32(0.0)
    kTileN = 64 * KVEC
    num_iter = K // kTileN
    nv = KVEC // 4
    for i in range_constexpr(num_iter):
        k_elem = fx.Int32(i * kTileN) + lane * fx.Int32(KVEC)
        av = tuple(
            Vec(
                buffer_ops.buffer_load(
                    rsrc_a,
                    m * fx.Int32(K) + k_elem + fx.Int32(j * 4),
                    vec_width=4,
                    dtype=T.bf16,
                )
            )
            for j in range(nv)
        )
        bv = tuple(
            Vec(
                buffer_ops.buffer_load(
                    rsrc_b,
                    n * fx.Int32(K) + k_elem + fx.Int32(j * 4),
                    vec_width=4,
                    dtype=T.bf16,
                    cache_modifier=0x2000,
                )
            )
            for j in range(nv)
        )
        for v in range_constexpr(nv):
            acc = dot2_f32_bf16(
                acc, pack_bf16x2(av[v][0], av[v][1]), pack_bf16x2(bv[v][0], bv[v][1])
            )
            acc = dot2_f32_bf16(
                acc, pack_bf16x2(av[v][2], av[v][3]), pack_bf16x2(bv[v][2], bv[v][3])
            )
    acc = wavefront_reduce_sum_f32(acc, lane)
    if lane == fx.Int32(0):
        store_bf16(acc, rsrc_c, m * fx.Int32(N) + n)


# -- JIT launcher --------------------------------------------------------------


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
    if M == 1:
        gemm_decode_bf16_kernel_m1(A, B, C, K, N).launch(
            grid=(1, N, 1),
            block=(64, 1, 1),
            stream=stream,
            value_attrs={"rocdl.waves_per_eu": WAVES_PER_EU},
        )
    elif M == 2:
        gemm_decode_bf16_kernel_m2(A, B, C, K, N).launch(
            grid=(1, N // NP, 1),
            block=(64, 1, 1),
            stream=stream,
            value_attrs={"rocdl.waves_per_eu": WAVES_PER_EU},
        )
    elif M == 3:
        gemm_decode_bf16_kernel_m3(A, B, C, K, N).launch(
            grid=(1, N // NP, 1),
            block=(64, 1, 1),
            stream=stream,
            value_attrs={"rocdl.waves_per_eu": WAVES_PER_EU},
        )
    elif M == 4:
        gemm_decode_bf16_kernel_m4(A, B, C, K, N).launch(
            grid=(1, N // NP, 1),
            block=(64, 1, 1),
            stream=stream,
            value_attrs={"rocdl.waves_per_eu": WAVES_PER_EU},
        )
