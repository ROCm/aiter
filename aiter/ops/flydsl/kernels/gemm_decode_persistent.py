# SPDX-License-Identifier: MIT

"""Persistent multi-wave BF16 decode GEMM for gfx950.

The kernel computes ``C[M, N] = A[M, K] @ B[N, K].T`` for ``M <= 4``.
One workgroup stages the complete activation matrix in right-sized LDS, then
its waves persistently traverse output columns. Each wave uses the gfx950
4x4x4 BF16 MFMA atom and a DPP-only reduction before deterministic BF16 stores.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _arith_dialect
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP, SmemAllocator, SmemPtr

from .gemm_decode_common import validate_gemm_decode_tensors
from .tensor_shim import GTensor, STensor

WAVE_SIZE = 64
MFMA_K = 4
STAGE_VECTOR = 8
DTYPE_BYTES = 2


@dataclass(frozen=True)
class PersistentDecodeConfig:
    """Compile-time launch geometry for the persistent family."""

    waves_per_workgroup: int = 16
    columns_per_wave: int = 1
    workgroups_per_cu: int = 1
    waves_per_eu: int = 0
    b_cache_modifier: int = 0x2000

    def validate(self, *, m: int, k: int) -> None:
        if self.waves_per_workgroup not in (4, 8, 16):
            raise ValueError("waves_per_workgroup must be one of 4, 8, or 16")
        if self.columns_per_wave not in (1, 2, 4):
            raise ValueError("columns_per_wave must be one of 1, 2, or 4")
        if self.workgroups_per_cu not in (1, 2, 4):
            raise ValueError("workgroups_per_cu must be one of 1, 2, or 4")
        if self.waves_per_eu < 0:
            raise ValueError("waves_per_eu must be non-negative")
        lds_bytes = _align_up(m * _staged_k(k) * DTYPE_BYTES, 128)
        if lds_bytes * self.workgroups_per_cu > SMEM_CAPACITY_MAP["gfx950"]:
            raise ValueError(
                "requested workgroups_per_cu exceeds the gfx950 LDS capacity"
            )


DEFAULT_CONFIG = PersistentDecodeConfig()


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _align_up(x: int, y: int) -> int:
    return _ceil_div(x, y) * y


def _staged_k(k: int) -> int:
    padding = MFMA_K if k % (WAVE_SIZE * MFMA_K) else 0
    return _align_up(k + padding, STAGE_VECTOR)


def select_persistent_decode_config(
    m: int,
    n: int,
    k: int,
) -> PersistentDecodeConfig:
    """Return the conservative winner from the bounded gfx950 sweep."""
    del n
    if k <= 1024:
        if m == 1:
            return PersistentDecodeConfig(waves_per_workgroup=8)
        return PersistentDecodeConfig(
            waves_per_workgroup=16,
            workgroups_per_cu=2,
        )
    return PersistentDecodeConfig(
        waves_per_workgroup=16,
        columns_per_wave=4,
    )


def _raw(value):
    if isinstance(value, ir.Value):
        return value
    if hasattr(value, "ir_value"):
        return _raw(value.ir_value())
    if hasattr(value, "value"):
        return _raw(value.value)
    return value


def _add_f32(lhs, rhs):
    return _arith_dialect.AddFOp(_raw(lhs), _raw(rhs)).result


def _dpp_move_f32(value, control: int):
    return fx.rocdl.update_dpp(
        T.f32,
        _raw(value),
        _raw(value),
        control,
        0xF,
        0xF,
        True,
    )


def _mfma_4x4x4_bf16(a_fragment, b_fragment, accumulator):
    a_i16 = vector.bitcast(T.vec(4, T.i16), a_fragment)
    b_i16 = vector.bitcast(T.vec(4, T.i16), b_fragment)
    # FlyDSL has no matching high-level MMA atom for this gfx950 instruction.
    return fx.rocdl.mfma_f32_4x4x4bf16_1k_(
        T.vec(4, T.f32),
        _raw(a_i16),
        _raw(b_i16),
        _raw(accumulator),
        0,
        0,
        0,
    )


def _reduce_mfma_scalar(accumulator):
    """Reduce one 4x4x4 MFMA accumulator to lane 63 using DPP only."""
    components = [
        vector.extract(accumulator, static_position=[i], dynamic_position=[])
        for i in range_constexpr(4)
    ]
    result = components[0]
    result = _add_f32(result, _dpp_move_f32(components[1], 0x101))
    result = _add_f32(result, _dpp_move_f32(components[2], 0x102))
    result = _add_f32(result, _dpp_move_f32(components[3], 0x103))
    result = _add_f32(result, _dpp_move_f32(result, 0x104))
    result = _add_f32(result, _dpp_move_f32(result, 0x108))
    result = _dpp_move_f32(result, 0x11F)
    result = _add_f32(result, _dpp_move_f32(result, 0x142))
    return _add_f32(result, _dpp_move_f32(result, 0x143))


def _masked_bf16x4(
    resource,
    row_base,
    column_base,
    row_size: int,
    cache_modifier: int,
):
    zero = arith.constant(0.0, type=T.bf16)
    values = []
    for i in range_constexpr(MFMA_K):
        column = column_base + fx.Int32(i)
        valid = column < fx.Int32(row_size)
        safe_column = ArithValue(_raw(valid)).select(column, fx.Int32(0))
        loaded = buffer_ops.buffer_load(
            resource,
            row_base + safe_column,
            vec_width=1,
            dtype=T.bf16,
            cache_modifier=cache_modifier,
        )
        values.append(ArithValue(_raw(valid)).select(loaded, zero))
    return vector.from_elements(T.vec(MFMA_K, T.bf16), values)


@functools.lru_cache(maxsize=512)
def compile_gemm_decode_persistent_bf16(
    m: int,
    n: int,
    k: int,
    num_cus: int,
    config: PersistentDecodeConfig = DEFAULT_CONFIG,
):
    """Compile one shape/configuration of the persistent decode kernel."""
    if not (1 <= m <= 4):
        raise ValueError("persistent BF16 decode supports m in [1, 4]")
    if n <= 0 or k <= 0 or num_cus <= 0:
        raise ValueError("n, k, and num_cus must be positive")
    config.validate(m=m, k=k)
    gpu_arch = get_rocm_arch()
    if gpu_arch != "gfx950":
        raise ValueError(f"persistent BF16 decode requires gfx950, got {gpu_arch}")

    waves = config.waves_per_workgroup
    columns = config.columns_per_wave
    block_threads = waves * WAVE_SIZE
    columns_per_workgroup = waves * columns
    logical_workgroups = _ceil_div(n, columns_per_workgroup)
    grid_workgroups = min(logical_workgroups, num_cus * config.workgroups_per_cu)
    persistent_turns = _ceil_div(logical_workgroups, grid_workgroups)
    has_k_tail = k % (WAVE_SIZE * MFMA_K) != 0
    full_k_tiles = k // (WAVE_SIZE * MFMA_K)
    staged_k = _staged_k(k)
    activation_elements = m * staged_k
    activation_vectors_per_row = k // STAGE_VECTOR
    activation_vectors = m * activation_vectors_per_row
    activation_tail_per_row = staged_k - activation_vectors_per_row * STAGE_VECTOR
    activation_tail_elements = m * activation_tail_per_row
    stage_iterations = _ceil_div(activation_vectors, block_threads)
    kernel_name = (
        f"gemm_decode_persistent_bf16_m{m}_n{n}_k{k}_w{waves}_c{columns}_"
        f"g{config.workgroups_per_cu}"
    )

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=(
            f"gemm_decode_persistent_smem_{m}_{n}_{k}_{waves}_{columns}_"
            f"{config.workgroups_per_cu}"
        ),
    )
    activation_smem_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = activation_smem_offset + activation_elements * DTYPE_BYTES

    @flyc.kernel(
        name=kernel_name,
        known_block_size=[block_threads, 1, 1],
    )
    def persistent_decode_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        wave_id = tid // fx.Int32(WAVE_SIZE)
        lane = tid % fx.Int32(WAVE_SIZE)

        a_global = GTensor(A, dtype=T.bf16, shape=(activation_elements,))
        b_global = GTensor(
            B,
            dtype=T.bf16,
            shape=(n, k),
            cache_modifier=config.b_cache_modifier,
        )
        c_global = GTensor(C, dtype=T.bf16, shape=(m, n))
        smem_base = allocator.get_base()
        a_smem_ptr = SmemPtr(
            smem_base,
            activation_smem_offset,
            T.bf16,
            shape=(activation_elements,),
        )
        a_smem = STensor(a_smem_ptr, T.bf16, shape=(activation_elements,))

        for stage_i in range_constexpr(stage_iterations):
            vector_slot = tid + fx.Int32(stage_i * block_threads)
            stage_if = scf.IfOp(
                _raw(vector_slot < fx.Int32(activation_vectors)),
                results_=[],
                has_else=False,
            )
            with ir.InsertionPoint(stage_if.then_block):
                row = vector_slot // fx.Int32(activation_vectors_per_row)
                row_vector = vector_slot % fx.Int32(activation_vectors_per_row)
                global_element = (
                    row * fx.Int32(k) + row_vector * fx.Int32(STAGE_VECTOR)
                )
                smem_element = (
                    row * fx.Int32(staged_k) + row_vector * fx.Int32(STAGE_VECTOR)
                )
                staged = a_global.vec_load((global_element,), STAGE_VECTOR)
                a_smem.vec_store((fx.Index(smem_element),), staged, STAGE_VECTOR)
                scf.YieldOp([])
        if activation_tail_elements:
            tail_if = scf.IfOp(
                _raw(tid < fx.Int32(activation_tail_elements)),
                results_=[],
                has_else=False,
            )
            with ir.InsertionPoint(tail_if.then_block):
                row = tid // fx.Int32(activation_tail_per_row)
                row_tail = tid % fx.Int32(activation_tail_per_row)
                column = fx.Int32(activation_vectors_per_row * STAGE_VECTOR) + row_tail
                valid = column < fx.Int32(k)
                safe_column = ArithValue(_raw(valid)).select(column, fx.Int32(0))
                global_element = row * fx.Int32(k) + safe_column
                loaded = a_global[global_element]
                zero = arith.constant(0.0, type=T.bf16)
                staged = ArithValue(_raw(valid)).select(loaded, zero)
                smem_element = row * fx.Int32(staged_k) + column
                a_smem_ptr.store(staged, [fx.Index(smem_element)])
                scf.YieldOp([])

        # This is the only synchronization required before persistent A reuse.
        gpu.barrier()

        first_column = (
            (gpu.block_idx.x * fx.Int32(waves) + wave_id) * fx.Int32(columns)
        )
        column_stride = fx.Int32(grid_workgroups * columns_per_workgroup)
        acc_zero = arith.constant_vector(0.0, T.vec(4, T.f32))

        for turn in range_constexpr(persistent_turns):
            column_base = first_column + fx.Int32(turn) * column_stride
            safe_columns = []
            column_valid = []
            for column_i in range_constexpr(columns):
                column = column_base + fx.Int32(column_i)
                valid = column < fx.Int32(n)
                safe = ArithValue(_raw(valid)).select(column, fx.Int32(0))
                safe_columns.append(safe)
                column_valid.append(valid)

            accumulators = [acc_zero] * (m * columns)
            for k_tile in range_constexpr(full_k_tiles):
                k_base = fx.Int32(k_tile * WAVE_SIZE * MFMA_K) + lane * fx.Int32(
                    MFMA_K
                )
                a_fragments = [
                    a_smem.vec_load(
                        (fx.Index(fx.Int32(row * staged_k) + k_base),),
                        MFMA_K,
                    )
                    for row in range_constexpr(m)
                ]
                for column_i in range_constexpr(columns):
                    b_fragment = b_global.vec_load(
                        (safe_columns[column_i], k_base), MFMA_K
                    )
                    for row in range_constexpr(m):
                        acc_index = row * columns + column_i
                        accumulators[acc_index] = _mfma_4x4x4_bf16(
                            a_fragments[row],
                            b_fragment,
                            accumulators[acc_index],
                        )

            if has_k_tail:
                k_base = fx.Int32(full_k_tiles * WAVE_SIZE * MFMA_K) + lane * fx.Int32(
                    MFMA_K
                )
                a_fragments = []
                for row in range_constexpr(m):
                    valid_base = k_base < fx.Int32(k)
                    safe_k_base = ArithValue(_raw(valid_base)).select(
                        k_base,
                        fx.Int32(k),
                    )
                    a_fragments.append(
                        a_smem.vec_load(
                            (
                                fx.Index(
                                    fx.Int32(row * staged_k) + safe_k_base
                                ),
                            ),
                            MFMA_K,
                        )
                    )
                for column_i in range_constexpr(columns):
                    b_row_base = safe_columns[column_i] * fx.Int32(k)
                    b_fragment = _masked_bf16x4(
                        b_global.rsrc,
                        b_row_base,
                        k_base,
                        k,
                        config.b_cache_modifier,
                    )
                    for row in range_constexpr(m):
                        acc_index = row * columns + column_i
                        accumulators[acc_index] = _mfma_4x4x4_bf16(
                            a_fragments[row],
                            b_fragment,
                            accumulators[acc_index],
                        )

            reduced = [
                _reduce_mfma_scalar(accumulator) for accumulator in accumulators
            ]
            for row in range_constexpr(m):
                for column_i in range_constexpr(columns):
                    store_valid = (lane == fx.Int32(WAVE_SIZE - 1)) & column_valid[
                        column_i
                    ]
                    store_if = scf.IfOp(
                        _raw(store_valid),
                        results_=[],
                        has_else=False,
                    )
                    with ir.InsertionPoint(store_if.then_block):
                        output = _arith_dialect.TruncFOp(
                            T.bf16,
                            reduced[row * columns + column_i],
                        ).result
                        c_global[row, column_base + fx.Int32(column_i)] = output
                        scf.YieldOp([])

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        context = CompilationContext.get_current()
        with ir.InsertionPoint(context.gpu_module_body):
            allocator.finalize()

        attrs = {}
        if config.waves_per_eu > 0:
            attrs["rocdl.waves_per_eu"] = config.waves_per_eu
        persistent_decode_kernel(
            A,
            B,
            C,
            value_attrs=attrs,
        ).launch(
            grid=(grid_workgroups, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch.kernel_name = kernel_name
    launch.lds_bytes = _align_up(activation_elements * DTYPE_BYTES, 128)
    launch.grid_workgroups = grid_workgroups
    return launch


def gemm_decode_persistent_bf16(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    m: int,
    n: int,
    k: int,
    num_cus: int,
    *,
    config: PersistentDecodeConfig | None = None,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch a cached persistent BF16 decode kernel."""
    validate_gemm_decode_tensors(A, B, C, m, n, k)
    if config is None:
        config = select_persistent_decode_config(m, n, k)
    launcher = compile_gemm_decode_persistent_bf16(m, n, k, num_cus, config)
    launcher(A, B, C, stream=stream)
    return C
