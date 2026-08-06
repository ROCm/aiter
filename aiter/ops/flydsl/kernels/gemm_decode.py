# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# gfx942/gfx950 BF16 warp-per-scalar GEMM for autoregressive decode.
#
# C[M, N] = A[M, K] @ B[N, K]^T   (B row-major weight matrix)
#
# One wavefront computes a compile-time MxN register tile:
#   - 64 lanes split K, each owns KVEC adjacent BF16 elements per full iteration
#   - gfx950 accumulates with v_dot2_f32_bf16 (2 BF16 MACs -> FP32)
#   - gfx942 expands BF16 exactly and accumulates with scalar FP32 FMA
#   - A and B vectors are reused across the register tile
#   - K tails use wide loads plus an in-kernel masked BF16-pair remainder
#   - Partial output-column tiles select an exact NP=1 instance
#   - Configurable 6-stage wave64 DPP or bpermute reduction (no LDS)
#   - Lane 63 converts FP32 to BF16 with configurable compile-time rounding

from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _arith_dialect
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import scf as _scf
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from .gemm_decode_common import (
    CACHE_POLICY_DEFAULT,
    CACHE_POLICY_NON_TEMPORAL,
    validate_cache_policy,
    validate_gemm_decode_tensors,
)


class OutputRounding(str, Enum):
    """Compile-time FP32-to-BF16 conversion choices."""

    RNE = "rne"
    RTZ = "rtz"
    TRUNCATE = "truncate"
    STOCHASTIC = "stochastic"


class ReductionMode(str, Enum):
    """Compile-time wave reduction choices."""

    DPP = "dpp"
    BPERMUTE_REFERENCE = "bpermute_reference"


class ContractionMode(str, Enum):
    """Compile-time architecture-specific contraction choices."""

    AUTO = "auto"
    DOT2_BF16 = "dot2_bf16"
    SCALAR_F32 = "scalar_f32"
    PACKED_F32 = "packed_f32"


@dataclass(frozen=True)
class GemmDecodeConfig:
    """Compile-time axes for one direct-kernel instance."""

    kvec: int = 8
    n_per_wave: int = 2
    m_per_wave: int = 4
    prefetch_depth: int = 0
    waves_per_eu: int = 2
    b_cache_modifier: int = CACHE_POLICY_DEFAULT
    reduction: ReductionMode = ReductionMode.DPP
    output_rounding: OutputRounding = OutputRounding.RNE
    contraction: ContractionMode = ContractionMode.AUTO

    def validate(self) -> None:
        if self.kvec not in (2, 4, 8):
            raise ValueError("kvec must be one of 2, 4, or 8")
        if self.n_per_wave not in (1, 2, 4):
            raise ValueError("n_per_wave must be one of 1, 2, or 4")
        if self.m_per_wave not in (1, 2, 3, 4):
            raise ValueError("m_per_wave must be one of 1, 2, 3, or 4")
        if self.prefetch_depth not in (0, 1, 2, 4):
            raise ValueError("prefetch_depth must be one of 0, 1, 2, or 4")
        if self.waves_per_eu <= 0:
            raise ValueError("waves_per_eu must be positive")
        validate_cache_policy(self.b_cache_modifier)
        if not isinstance(self.contraction, ContractionMode):
            raise ValueError("contraction must be a ContractionMode")


DEFAULT_CONFIG = GemmDecodeConfig()
DEFAULT_CONFIG.validate()
_M1_LOW_K_CONFIG = GemmDecodeConfig(
    kvec=2,
    n_per_wave=4,
    m_per_wave=1,
)
_M1_HIGH_K_CONFIG = GemmDecodeConfig(
    kvec=8,
    n_per_wave=1,
    m_per_wave=1,
)
_M4_LOW_K_CONFIG = GemmDecodeConfig(
    kvec=2,
    n_per_wave=2,
    m_per_wave=4,
)
_M4_HIGH_K_CONFIG = GemmDecodeConfig(
    kvec=8,
    n_per_wave=4,
    m_per_wave=4,
    reduction=ReductionMode.BPERMUTE_REFERENCE,
)


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
        [_to_ir(acc), _to_ir(a_i32), _to_ir(b_i32)],
        "v_dot2_f32_bf16 $0, $2, $3, $1",
        "=v,0,v,v",
        has_side_effects=False,
    )


def unpack_bf16x2_f32(packed) -> tuple[ir.Value, ir.Value]:
    """Expand two packed BF16 values exactly into two FP32 values."""
    packed = ArithValue(_to_ir(packed))
    lo_bits = (packed & fx.Int32(0xFFFF)) << fx.Int32(16)
    hi_bits = packed & fx.Int32(0xFFFF0000)
    return (
        _to_ir(ArithValue(lo_bits).bitcast(T.f32)),
        _to_ir(ArithValue(hi_bits).bitcast(T.f32)),
    )


def scalar_fma_bf16x2(
    acc: ir.Value,
    a_pair: tuple[ir.Value, ir.Value],
    b_pair: tuple[ir.Value, ir.Value],
) -> ir.Value:
    """Accumulate one exactly expanded BF16 pair with scalar FP32 FMAs."""
    acc = _llvm.intr_fma(a_pair[0], b_pair[0], _to_ir(acc))
    return _llvm.intr_fma(a_pair[1], b_pair[1], acc)


def packed_fma_bf16x2(acc, a_pair, b_pair) -> ir.Value:
    """Accumulate two expanded BF16 products with one packed FP32 FMA."""
    packed_type = ir.VectorType.get([2], ir.F32Type.get())
    return _llvm.inline_asm(
        packed_type,
        [_to_ir(acc), _to_ir(a_pair), _to_ir(b_pair)],
        "v_pk_fma_f32 $0, $2, $3, $1",
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


def _dpp_add_f32(val: ir.Value, control: str) -> ir.Value:
    """Add a DPP-selected lane value to ``val``."""
    return _llvm.inline_asm(
        ir.F32Type.get(),
        [_to_ir(val), _to_ir(val), _to_ir(val)],
        f"s_nop 3\n\tv_add_f32 $0, $2, $3 {control} bound_ctrl:0",
        "=v,0,v,v",
        has_side_effects=False,
    )


def wavefront_reduce_sum_f32(val: ir.Value) -> ir.Value:
    """Reduce a wave64 sum with DPP; the complete sum is in lane 63."""
    for shift in (8, 4, 2, 1):
        val = _dpp_add_f32(val, f"row_shr:{shift}")
    val = _dpp_add_f32(val, "row_bcast:15")
    val = _dpp_add_f32(val, "row_bcast:31")
    return val


def _load_kvec(rsrc, base_elem, kvec, cache_modifier=0):
    """Load one compile-time BF16 vector."""
    return Vec(
        buffer_ops.buffer_load(
            rsrc,
            base_elem,
            vec_width=kvec,
            dtype=T.bf16,
            cache_modifier=cache_modifier,
        )
    )


def _const_index(value: int) -> ir.Value:
    """Create an index constant for an explicit SCF loop."""
    index_type = ir.IndexType.get()
    return _arith_dialect.ConstantOp(
        index_type,
        ir.IntegerAttr.get(index_type, value),
    ).result


def _index_to_i32(value: ir.Value) -> ir.Value:
    """Cast an SCF induction variable to the kernel's i32 arithmetic type."""
    return _arith_dialect.IndexCastOp(T.i32, value).result


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
    if isinstance(safe_row, int):
        safe_row = fx.Int32(safe_row)
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


def _bpermute_reduce_sum_f32(val, lane):
    """Reference butterfly reduction; the complete sum is in every lane."""
    val = _llvm.inline_asm(
        ir.F32Type.get(),
        [_to_ir(val)],
        "s_nop 3\n\tv_mov_b32 $0, $1",
        "=v,v",
        has_side_effects=False,
    )
    for stage in range_constexpr(6):
        partner = lane ^ fx.Int32(1 << stage)
        val_i32 = ArithValue(_to_ir(val)).bitcast(T.i32)
        peer_i32 = fx.rocdl.ds_bpermute(
            T.i32,
            partner * fx.Int32(4),
            val_i32,
        )
        val = _add_f32(_to_ir(val), ArithValue(peer_i32).bitcast(T.f32))
    return val


def _make_kernel(config: GemmDecodeConfig):
    kvec = config.kvec
    np = config.n_per_wave
    mp = config.m_per_wave
    prefetch_depth = config.prefetch_depth
    cache_modifier = config.b_cache_modifier
    reduction = config.reduction
    rounding = config.output_rounding
    gfx = get_rocm_arch()
    contraction = config.contraction
    if contraction == ContractionMode.AUTO:
        contraction = (
            ContractionMode.DOT2_BF16
            if gfx == "gfx950"
            else ContractionMode.SCALAR_F32
        )
    if contraction == ContractionMode.DOT2_BF16 and gfx != "gfx950":
        raise ValueError(f"dot2_bf16 contraction requires gfx950, got {gfx}")
    use_packed = contraction == ContractionMode.PACKED_F32
    acc_type = T.vec(2, T.f32) if use_packed else T.f32
    use_dpp = reduction == ReductionMode.DPP
    store_lane = 63 if use_dpp else 0

    def zero_accumulator():
        if use_packed:
            return arith.constant_vector(0.0, T.vec(2, T.f32))
        return _const_f32(0.0)

    def prepare_pair(packed):
        if contraction == ContractionMode.DOT2_BF16:
            return packed
        expanded = unpack_bf16x2_f32(packed)
        if contraction == ContractionMode.PACKED_F32:
            return vector.from_elements(T.vec(2, T.f32), list(expanded))
        return expanded

    def contract_pair(acc, a_pair, b_pair):
        if contraction == ContractionMode.DOT2_BF16:
            return dot2_f32_bf16(acc, a_pair, b_pair)
        if contraction == ContractionMode.PACKED_F32:
            return packed_fma_bf16x2(acc, a_pair, b_pair)
        return scalar_fma_bf16x2(acc, a_pair, b_pair)

    def _accumulate_loaded_kvecs(accs, avs, bvs, mp, np, npairs):
        """Accumulate one loaded K tile with the selected contraction."""
        for pair in range_constexpr(npairs):
            ap = [
                prepare_pair(pack_bf16x2(av[2 * pair], av[2 * pair + 1]))
                for av in avs
            ]
            bp = [
                prepare_pair(pack_bf16x2(bv[2 * pair], bv[2 * pair + 1]))
                for bv in bvs
            ]
            for row in range_constexpr(mp):
                for col in range_constexpr(np):
                    idx = row * np + col
                    accs[idx] = contract_pair(
                        accs[idx],
                        ap[row],
                        bp[col],
                    )
        return accs

    @flyc.kernel
    def kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        K: fx.Constexpr[int],
        N: fx.Constexpr[int],
    ):
        lane = gpu.thread_idx.x
        m_base = gpu.block_idx.x * fx.Int32(mp)
        n_base = gpu.block_idx.y * fx.Int32(np)
        rsrc_a = buffer_ops.create_buffer_resource(A)
        rsrc_b = buffer_ops.create_buffer_resource(B)
        rsrc_c = buffer_ops.create_buffer_resource(C)
        m_rows = []
        for row in range_constexpr(mp):
            m_rows.append(m_base + fx.Int32(row))
        n_cols = []
        for col in range_constexpr(np):
            n_col = n_base + fx.Int32(col)
            n_cols.append(n_col)
        n_load = n_cols

        accs = [
            zero_accumulator()
            for _ in range_constexpr(mp * np)
        ]
        k_tile = 64 * kvec
        npairs = kvec // 2
        full_k_tiles = K // k_tile
        if prefetch_depth == 0:
            for i in range_constexpr(full_k_tiles):
                k_elem = fx.Int32(i * k_tile) + lane * fx.Int32(kvec)
                avs = [
                    _load_kvec(rsrc_a, row * fx.Int32(K) + k_elem, kvec)
                    for row in m_rows
                ]
                bvs = [
                    _load_kvec(
                        rsrc_b,
                        col * fx.Int32(K) + k_elem,
                        kvec,
                        cache_modifier,
                    )
                    for col in n_load
                ]
                accs = _accumulate_loaded_kvecs(
                    accs,
                    avs,
                    bvs,
                    mp,
                    np,
                    npairs,
                )
        else:
            full_prefetch_batches = full_k_tiles // prefetch_depth
            prefetch_remainder = full_k_tiles % prefetch_depth
            prefetch_loop = _scf.ForOp(
                _const_index(0),
                _const_index(full_prefetch_batches),
                _const_index(1),
                accs,
            )
            with ir.InsertionPoint(prefetch_loop.body):
                batch = _index_to_i32(prefetch_loop.induction_variable)
                current_accs = list(prefetch_loop.inner_iter_args)
                prefetched_a = []
                prefetched_b = []
                for stage in range_constexpr(prefetch_depth):
                    tile = batch * fx.Int32(prefetch_depth) + fx.Int32(stage)
                    k_elem = tile * fx.Int32(k_tile) + lane * fx.Int32(kvec)
                    prefetched_a.append(
                        [
                            _load_kvec(
                                rsrc_a,
                                row * fx.Int32(K) + k_elem,
                                kvec,
                            )
                            for row in m_rows
                        ]
                    )
                    prefetched_b.append(
                        [
                            _load_kvec(
                                rsrc_b,
                                col * fx.Int32(K) + k_elem,
                                kvec,
                                cache_modifier,
                            )
                            for col in n_load
                        ]
                    )
                for stage in range_constexpr(prefetch_depth):
                    current_accs = _accumulate_loaded_kvecs(
                        current_accs,
                        prefetched_a[stage],
                        prefetched_b[stage],
                        mp,
                        np,
                        npairs,
                    )
                _scf.YieldOp([_to_ir(acc) for acc in current_accs])
            accs = list(prefetch_loop.results)

            for stage in range_constexpr(prefetch_remainder):
                tile = full_prefetch_batches * prefetch_depth + stage
                k_elem = fx.Int32(tile * k_tile) + lane * fx.Int32(kvec)
                avs = [
                    _load_kvec(rsrc_a, row * fx.Int32(K) + k_elem, kvec)
                    for row in m_rows
                ]
                bvs = [
                    _load_kvec(
                        rsrc_b,
                        col * fx.Int32(K) + k_elem,
                        kvec,
                        cache_modifier,
                    )
                    for col in n_load
                ]
                accs = _accumulate_loaded_kvecs(
                    accs,
                    avs,
                    bvs,
                    mp,
                    np,
                    npairs,
                )

        if K % k_tile != 0:
            tail_start = (K // k_tile) * k_tile
            tail_full_lanes = (K % k_tile) // kvec
            tail_scalar = (K % k_tile) % kvec
            if tail_full_lanes != 0:
                incoming_accs = tuple(accs)
                vector_tail = _scf.IfOp(
                    _to_ir(lane < fx.Int32(tail_full_lanes)),
                    results_=[acc_type] * len(accs),
                    has_else=True,
                )
                with ir.InsertionPoint(vector_tail.then_block):
                    k_elem = fx.Int32(tail_start) + lane * fx.Int32(kvec)
                    avs = [
                        _load_kvec(
                            rsrc_a,
                            row * fx.Int32(K) + k_elem,
                            kvec,
                        )
                        for row in m_rows
                    ]
                    bvs = [
                        _load_kvec(
                            rsrc_b,
                            col * fx.Int32(K) + k_elem,
                            kvec,
                            cache_modifier,
                        )
                        for col in n_load
                    ]
                    for pair in range_constexpr(npairs):
                        ap = [
                            prepare_pair(
                                pack_bf16x2(
                                    av[2 * pair],
                                    av[2 * pair + 1],
                                )
                            )
                            for av in avs
                        ]
                        bp = [
                            prepare_pair(
                                pack_bf16x2(
                                    bv[2 * pair],
                                    bv[2 * pair + 1],
                                )
                            )
                            for bv in bvs
                        ]
                        for row in range_constexpr(mp):
                            for col in range_constexpr(np):
                                idx = row * np + col
                                accs[idx] = contract_pair(
                                    accs[idx],
                                    ap[row],
                                    bp[col],
                                )
                    _scf.YieldOp([_to_ir(acc) for acc in accs])
                with ir.InsertionPoint(vector_tail.else_block):
                    _scf.YieldOp([_to_ir(acc) for acc in incoming_accs])
                accs = list(vector_tail.results)

            if tail_scalar != 0:
                tail_lane_base = (
                    fx.Int32(tail_start + tail_full_lanes * kvec)
                    + lane * fx.Int32(kvec)
                )
                for pair in range_constexpr(npairs):
                    ap = [
                        prepare_pair(
                            _load_tail_pair(
                                rsrc_a,
                                row,
                                K,
                                tail_lane_base,
                                pair,
                            )
                        )
                        for row in m_rows
                    ]
                    bp = [
                        prepare_pair(
                            _load_tail_pair(
                                rsrc_b,
                                col,
                                K,
                                tail_lane_base,
                                pair,
                                cache_modifier=cache_modifier,
                            )
                        )
                        for col in n_cols
                    ]
                    for row in range_constexpr(mp):
                        for col in range_constexpr(np):
                            idx = row * np + col
                            accs[idx] = contract_pair(
                                accs[idx],
                                ap[row],
                                bp[col],
                            )

        if contraction == ContractionMode.PACKED_F32:
            packed_accs = accs
            accs = []
            for packed_acc in packed_accs:
                lo = vector.extract(
                    packed_acc,
                    static_position=[0],
                    dynamic_position=[],
                )
                hi = vector.extract(
                    packed_acc,
                    static_position=[1],
                    dynamic_position=[],
                )
                if use_dpp and K % kvec == 0:
                    lo = wavefront_reduce_sum_f32(lo)
                    hi = wavefront_reduce_sum_f32(hi)
                else:
                    lo = _bpermute_reduce_sum_f32(lo, lane)
                    hi = _bpermute_reduce_sum_f32(hi, lane)
                accs.append(_add_f32(lo, hi))
        elif use_dpp and K % kvec == 0:
            accs = [wavefront_reduce_sum_f32(acc) for acc in accs]
        else:
            accs = [
                _bpermute_reduce_sum_f32(acc, lane)
                for acc in accs
            ]

        if lane == fx.Int32(store_lane):
            for row in range_constexpr(mp):
                for col in range_constexpr(np):
                    c_elem = m_rows[row] * fx.Int32(N) + n_cols[col]
                    store_bf16(
                        accs[row * np + col],
                        rsrc_c,
                        c_elem,
                        rounding,
                    )

    return kernel


@lru_cache(maxsize=None)
def get_gemm_decode_bf16(config: GemmDecodeConfig = DEFAULT_CONFIG):
    """Return a stable launch callable for ``config``."""
    config.validate()

    def launch(A, B, C, M, N, K, stream=fx.Stream(None)):
        return gemm_decode_bf16_configured(
            A,
            B,
            C,
            M,
            N,
            K,
            config,
            stream,
        )

    return launch


@flyc.jit
def _launch_gemm_decode_bf16(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    M: fx.Constexpr[int],
    N: fx.Constexpr[int],
    K: fx.Constexpr[int],
    kvec: fx.Constexpr[int],
    mp: fx.Constexpr[int],
    np: fx.Constexpr[int],
    prefetch_depth: fx.Constexpr[int],
    waves_per_eu: fx.Constexpr[int],
    cache_modifier: fx.Constexpr[int],
    reduction_code: fx.Constexpr[int],
    rounding_code: fx.Constexpr[int],
    contraction_code: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    reduction = (
        ReductionMode.DPP
        if reduction_code == 0
        else ReductionMode.BPERMUTE_REFERENCE
    )
    rounding = (
        OutputRounding.RNE
        if rounding_code == 0
        else OutputRounding.RTZ
        if rounding_code == 1
        else OutputRounding.TRUNCATE
        if rounding_code == 2
        else OutputRounding.STOCHASTIC
    )
    contraction = (
        ContractionMode.AUTO
        if contraction_code == 0
        else ContractionMode.DOT2_BF16
        if contraction_code == 1
        else ContractionMode.SCALAR_F32
        if contraction_code == 2
        else ContractionMode.PACKED_F32
    )
    config = GemmDecodeConfig(
        kvec=kvec,
        n_per_wave=np,
        m_per_wave=mp,
        prefetch_depth=prefetch_depth,
        waves_per_eu=waves_per_eu,
        b_cache_modifier=cache_modifier,
        reduction=reduction,
        output_rounding=rounding,
        contraction=contraction,
    )
    kernel = _make_kernel(config)
    kernel(
        A,
        B,
        C,
        K,
        N,
        value_attrs={"rocdl.waves_per_eu": waves_per_eu},
    ).launch(
        grid=(M // mp, N // np, 1),
        block=(64, 1, 1),
        stream=stream,
    )


def gemm_decode_bf16_configured(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    config: GemmDecodeConfig,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch one explicit compile-time configuration."""
    validate_gemm_decode_tensors(A, B, C, M, N, K)
    config.validate()
    if (
        config.output_rounding == OutputRounding.STOCHASTIC
        and get_rocm_arch() != "gfx950"
    ):
        raise ValueError("stochastic BF16 conversion requires gfx950")
    if N % config.n_per_wave != 0:
        raise ValueError("N must be divisible by config.n_per_wave")
    if M % config.m_per_wave != 0:
        raise ValueError("M must be divisible by config.m_per_wave")
    reduction_code = 0 if config.reduction == ReductionMode.DPP else 1
    rounding_code = {
        OutputRounding.RNE: 0,
        OutputRounding.RTZ: 1,
        OutputRounding.TRUNCATE: 2,
        OutputRounding.STOCHASTIC: 3,
    }[config.output_rounding]
    contraction_code = {
        ContractionMode.AUTO: 0,
        ContractionMode.DOT2_BF16: 1,
        ContractionMode.SCALAR_F32: 2,
        ContractionMode.PACKED_F32: 3,
    }[config.contraction]
    return _launch_gemm_decode_bf16(
        A,
        B,
        C,
        M,
        N,
        K,
        config.kvec,
        config.m_per_wave,
        config.n_per_wave,
        config.prefetch_depth,
        config.waves_per_eu,
        config.b_cache_modifier,
        reduction_code,
        rounding_code,
        contraction_code,
        stream=stream,
    )


@flyc.jit
def _launch_m1_low_k(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    M: fx.Constexpr[int],
    N: fx.Constexpr[int],
    K: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    kernel = _make_kernel(_M1_LOW_K_CONFIG)
    kernel(
        A,
        B,
        C,
        K,
        N,
        value_attrs={"rocdl.waves_per_eu": 2},
    ).launch(
        grid=(1, N // 4, 1),
        block=(64, 1, 1),
        stream=stream,
    )


@flyc.jit
def _launch_m1_high_k(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    M: fx.Constexpr[int],
    N: fx.Constexpr[int],
    K: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    kernel = _make_kernel(_M1_HIGH_K_CONFIG)
    kernel(
        A,
        B,
        C,
        K,
        N,
        value_attrs={"rocdl.waves_per_eu": 2},
    ).launch(
        grid=(1, N, 1),
        block=(64, 1, 1),
        stream=stream,
    )


@flyc.jit
def _launch_m4_low_k(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    M: fx.Constexpr[int],
    N: fx.Constexpr[int],
    K: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    kernel = _make_kernel(_M4_LOW_K_CONFIG)
    kernel(
        A,
        B,
        C,
        K,
        N,
        value_attrs={"rocdl.waves_per_eu": 2},
    ).launch(
        grid=(1, N // 2, 1),
        block=(64, 1, 1),
        stream=stream,
    )


@flyc.jit
def _launch_m4_high_k(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    M: fx.Constexpr[int],
    N: fx.Constexpr[int],
    K: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    kernel = _make_kernel(_M4_HIGH_K_CONFIG)
    kernel(
        A,
        B,
        C,
        K,
        N,
        value_attrs={"rocdl.waves_per_eu": 2},
    ).launch(
        grid=(1, N // 4, 1),
        block=(64, 1, 1),
        stream=stream,
    )


def select_gemm_decode_config(M: int, N: int, K: int) -> GemmDecodeConfig:
    """Select a target-specific starting config for the direct family."""
    if M < 1 or M > 4:
        raise ValueError("gemm_decode_bf16 supports M in [1, 4]")
    if get_rocm_arch() == "gfx942":
        if K <= 256:
            kvec = 2
        elif K == 768:
            kvec = 4
        else:
            kvec = 8
        use_m4_d2 = M == 4 and K > 1536 and N >= 16384 and N % 2 == 0
        cache_policy = (
            CACHE_POLICY_NON_TEMPORAL
            if M == 1 and K >= 4096
            else CACHE_POLICY_DEFAULT
        )
        return GemmDecodeConfig(
            kvec=kvec,
            n_per_wave=2 if use_m4_d2 else 1,
            m_per_wave=M if M in (1, 2, 4) else 1,
            waves_per_eu=4,
            b_cache_modifier=cache_policy,
            reduction=ReductionMode.DPP,
            contraction=(
                ContractionMode.PACKED_F32
                if use_m4_d2
                else ContractionMode.SCALAR_F32
            ),
        )
    if M == 1:
        config = _M1_LOW_K_CONFIG if K <= 1536 else _M1_HIGH_K_CONFIG
    elif M == 4:
        config = _M4_LOW_K_CONFIG if K <= 1536 else _M4_HIGH_K_CONFIG
    else:
        config = GemmDecodeConfig(
            kvec=2 if K <= 1536 else 8,
            n_per_wave=4 if K <= 1536 else 2,
            m_per_wave=M,
            prefetch_depth=2 if M == 3 and (N, K) == (16384, 7168) else 0,
        )
    np = config.n_per_wave
    if N % np != 0:
        return replace(config, n_per_wave=1)
    return config


def gemm_decode_bf16(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch the default direct-kernel configuration for ``M``."""
    validate_gemm_decode_tensors(A, B, C, M, N, K)
    if get_rocm_arch() == "gfx950":
        if M == 1 and K <= 1536 and N % 4 == 0:
            return _launch_m1_low_k(A, B, C, M, N, K, stream=stream)
        if M == 1 and K > 1536:
            return _launch_m1_high_k(A, B, C, M, N, K, stream=stream)
        if M == 4 and K <= 1536 and N % 2 == 0:
            return _launch_m4_low_k(A, B, C, M, N, K, stream=stream)
        if M == 4 and K > 1536 and N % 4 == 0:
            return _launch_m4_high_k(A, B, C, M, N, K, stream=stream)
    config = select_gemm_decode_config(M, N, K)
    return gemm_decode_bf16_configured(
        A,
        B,
        C,
        M,
        N,
        K,
        config,
        stream,
    )
