# SPDX-License-Identifier: MIT

"""Persistent multi-wave BF16 decode GEMM for gfx942 and gfx950.

The kernel computes ``C[M, N] = A[M, K] @ B[N, K].T`` for ``M <= 4``.
One workgroup stages the complete activation matrix in right-sized LDS, then
its waves persistently traverse output columns. Each wave uses the shared
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
from flydsl._mlir.dialects import llvm as _llvm_dialect
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T, Int32
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP, SmemAllocator, SmemPtr

from .gemm_decode_common import (
    CACHE_POLICY_DEFAULT,
    CACHE_POLICY_NON_TEMPORAL,
    validate_cache_policy,
    validate_gemm_decode_tensors,
)
from .tensor_shim import GTensor, STensor

WAVE_SIZE = 64
MFMA_K = 4
K_CHUNK = 8
K_UNROLL = 2
STAGE_VECTOR = 8
DTYPE_BYTES = 2


@dataclass(frozen=True)
class PersistentDecodeConfig:
    """Compile-time launch geometry for the persistent family."""

    waves_per_workgroup: int = 16
    columns_per_wave: int = 1
    workgroups_per_cu: int = 1
    waves_per_eu: int = 0
    b_cache_modifier: int = CACHE_POLICY_DEFAULT
    preload_b_tile: bool = False
    b_load_width: int = MFMA_K
    prefetch_stages: int = 1

    def validate(self, *, m: int, k: int, gpu_arch: str | None = None) -> None:
        if not 4 <= self.waves_per_workgroup <= 16:
            raise ValueError("waves_per_workgroup must be in [4, 16]")
        if self.columns_per_wave not in (1, 2, 4):
            raise ValueError("columns_per_wave must be one of 1, 2, or 4")
        if self.workgroups_per_cu not in (1, 2, 4):
            raise ValueError("workgroups_per_cu must be one of 1, 2, or 4")
        if self.waves_per_eu < 0:
            raise ValueError("waves_per_eu must be non-negative")
        validate_cache_policy(self.b_cache_modifier)
        if self.b_load_width not in (MFMA_K, 2 * MFMA_K):
            raise ValueError("b_load_width must be 4 or 8")
        if self.prefetch_stages not in (1, 2):
            raise ValueError("prefetch_stages must be 1 or 2")
        if self.prefetch_stages == 2 and m * self.columns_per_wave > 12:
            raise ValueError(
                "two-stage prefetch exceeds the validated register budget"
            )
        if gpu_arch is None:
            gpu_arch = get_rocm_arch()
        if gpu_arch not in ("gfx942", "gfx950"):
            raise ValueError(
                f"persistent BF16 decode requires gfx942 or gfx950, got {gpu_arch}"
            )
        if gpu_arch == "gfx942":
            if self.b_load_width != MFMA_K or self.prefetch_stages != 1:
                raise ValueError(
                    "b_load_width and prefetch_stages are gfx950-only"
                )
        elif self.preload_b_tile:
            raise ValueError("preload_b_tile is gfx942-only")
        lds_bytes = _align_up(m * _staged_k(k) * DTYPE_BYTES, 128)
        if lds_bytes * self.workgroups_per_cu > SMEM_CAPACITY_MAP[gpu_arch]:
            raise ValueError(
                f"requested workgroups_per_cu exceeds the {gpu_arch} LDS capacity"
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
    num_cus: int = 304,
) -> PersistentDecodeConfig:
    """Return a target-specific config from the bounded architecture sweeps."""
    if k <= 1024:
        if m == 1:
            return PersistentDecodeConfig(waves_per_workgroup=8)
        return PersistentDecodeConfig(
            waves_per_workgroup=16,
            workgroups_per_cu=2,
        )
    if get_rocm_arch() == "gfx942":
        if n < 4096:
            return PersistentDecodeConfig(
                waves_per_workgroup=16,
                columns_per_wave=1,
            )
        if (m, n, k) == (4, 6288, 7168):
            return PersistentDecodeConfig(
                waves_per_workgroup=4,
                columns_per_wave=1,
            )
        if (m, n, k) == (4, 8448, 7168):
            return PersistentDecodeConfig(
                waves_per_workgroup=12,
                columns_per_wave=4,
            )
        if m == 1:
            columns = 2 if n >= 16384 else 4
        elif m == 4 and n < 8192:
            columns = 2
        else:
            columns = 4
        if m == 1 and k >= 4096 and columns == 2:
            active_waves = 12
        else:
            active_waves = (
                min(16, max(4, _ceil_div(n, num_cus * columns)))
                if columns == 2
                else 16
            )
        return PersistentDecodeConfig(
            waves_per_workgroup=active_waves,
            columns_per_wave=columns,
            preload_b_tile=(
                m == 1
                and k == 7168
                and n in (6288, 20480)
            ),
        )
    return PersistentDecodeConfig(
        waves_per_workgroup=8 if m == 4 and n >= 32768 and k >= 4096 else 16,
        columns_per_wave=4,
        waves_per_eu=2 if m == 3 and k >= 4096 else 0,
        b_cache_modifier=(
            CACHE_POLICY_NON_TEMPORAL
            if n >= 32768 and k >= 4096
            else CACHE_POLICY_DEFAULT
        ),
        b_load_width=8 if m >= 2 else MFMA_K,
        prefetch_stages=2 if m == 3 and k >= 4096 else 1,
    )


def _raw(value):
    if isinstance(value, ir.Value):
        return value
    if hasattr(value, "ir_value"):
        return _raw(value.ir_value())
    if hasattr(value, "value"):
        return _raw(value.value)
    return value


def _compiler_memory_barrier() -> None:
    """Prevent loop-invariant LDS loads without synchronizing the workgroup."""
    _llvm_dialect.inline_asm(
        None,
        [],
        "",
        "~{memory}",
        has_side_effects=True,
    )


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
    # FlyDSL has no matching high-level MMA atom for this shared instruction.
    return fx.rocdl.mfma_f32_4x4x4bf16_1k_(
        T.vec(4, T.f32),
        _raw(a_i16),
        _raw(b_i16),
        _raw(accumulator),
        0,
        0,
        0,
    )


def _bf16x4_slice(fragment, fragment_index: int):
    return vector.extract_strided_slice(
        T.vec(MFMA_K, T.bf16),
        _raw(fragment),
        [fragment_index * MFMA_K],
        [MFMA_K],
        [1],
    )


def _accumulate_chunks(
    accumulators,
    a_chunks,
    b_chunks,
    *,
    column_start: int,
    column_end: int,
    m: int,
    columns: int,
    b_load_width: int,
):
    for column_i in range_constexpr(column_start, column_end):
        for fragment_i in range_constexpr(b_load_width // MFMA_K):
            b_fragment = _bf16x4_slice(
                b_chunks[column_i],
                fragment_i,
            )
            for row in range_constexpr(m):
                acc_index = row * columns + column_i
                a_fragment = _bf16x4_slice(
                    a_chunks[row],
                    fragment_i,
                )
                accumulators[acc_index] = _mfma_4x4x4_bf16(
                    a_fragment,
                    b_fragment,
                    accumulators[acc_index],
                )
    return accumulators


def _accumulate_k_chunks_runtime(
    accumulators,
    *,
    a_smem,
    b_global,
    safe_columns,
    lane,
    m: int,
    columns: int,
    b_load_width: int,
    full_k_chunks,
    staged_k: int,
    k_chunk: int,
):
    """Accumulate full K chunks in one loop body with loop-carried AGPRs."""
    iteration = _raw(fx.Int32(0))
    state = [iteration, *[_raw(accumulator) for accumulator in accumulators]]
    state_types = [value.type for value in state]
    loop = scf.WhileOp(state_types, state)
    loop.regions[0].blocks.append(*state_types)
    loop.regions[1].blocks.append(*state_types)

    with ir.InsertionPoint(loop.regions[0].blocks[0]):
        before_args = list(loop.regions[0].blocks[0].arguments)
        condition = fx.Int32(before_args[0]) < full_k_chunks
        scf.ConditionOp(_raw(condition), before_args)

    with ir.InsertionPoint(loop.regions[1].blocks[0]):
        after_args = list(loop.regions[1].blocks[0].arguments)
        k_chunk_i = fx.Int32(after_args[0])
        current_accumulators = list(after_args[1:])
        k_base = k_chunk_i * fx.Int32(k_chunk) + lane * fx.Int32(
            b_load_width
        )
        a_chunks = [
            a_smem.vec_load(
                (fx.Index(fx.Int32(row * staged_k) + k_base),),
                b_load_width,
            )
            for row in range_constexpr(m)
        ]
        b_chunks = [
            b_global.vec_load(
                (safe_columns[column_i], k_base),
                b_load_width,
            )
            for column_i in range_constexpr(columns)
        ]
        current_accumulators = _accumulate_chunks(
            current_accumulators,
            a_chunks,
            b_chunks,
            column_start=0,
            column_end=columns,
            m=m,
            columns=columns,
            b_load_width=b_load_width,
        )
        next_iteration = _raw(k_chunk_i + fx.Int32(1))
        scf.YieldOp([next_iteration, *current_accumulators])

    return list(loop.results[1:])


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


def _masked_bf16x8(
    resource,
    row_base,
    column_base,
    valid,
    row_size: int,
    cache_modifier: int,
):
    if row_size % K_CHUNK:
        zero = arith.constant(0.0, type=T.bf16)
        values = []
        for i in range_constexpr(K_CHUNK):
            column = column_base + fx.Int32(i)
            element_valid = valid & (column < fx.Int32(row_size))
            safe_column = ArithValue(_raw(element_valid)).select(
                column,
                fx.Int32(0),
            )
            loaded = buffer_ops.buffer_load(
                resource,
                row_base + safe_column,
                vec_width=1,
                dtype=T.bf16,
                cache_modifier=cache_modifier,
            )
            values.append(
                ArithValue(_raw(element_valid)).select(loaded, zero)
            )
        return vector.from_elements(T.vec(K_CHUNK, T.bf16), values)

    safe_column = ArithValue(_raw(valid)).select(column_base, fx.Int32(0))
    loaded = buffer_ops.buffer_load(
        resource,
        row_base + safe_column,
        vec_width=K_CHUNK,
        dtype=T.bf16,
        cache_modifier=cache_modifier,
    )
    zero = arith.constant_vector(0.0, T.vec(K_CHUNK, T.bf16))
    return ArithValue(_raw(valid)).select(loaded, zero)


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


def _bf16x4_fragment(chunk, offset: int):
    return vector.extract_strided_slice(
        T.vec(MFMA_K, T.bf16),
        chunk,
        offsets=[offset],
        sizes=[MFMA_K],
        strides=[1],
    )


def _compute_persistent_column_tile(
    *,
    b_global,
    c_global,
    a_smem,
    column_base,
    lane,
    m: int,
    n: int,
    k: int,
    columns: int,
    b_load_width: int,
    prefetch_stages: int,
    cache_modifier: int,
    full_k_chunks: int,
    tail_mfma_tiles: int,
    staged_k: int,
    k_chunk: int,
) -> None:
    """Emit one persistent column tile while keeping N-loop state minimal."""
    safe_columns = []
    column_valid = []
    for column_i in range_constexpr(columns):
        column = column_base + fx.Int32(column_i)
        valid = column < fx.Int32(n)
        safe = ArithValue(_raw(valid)).select(column, fx.Int32(0))
        safe_columns.append(safe)
        column_valid.append(valid)

    acc_zero = arith.constant_vector(0.0, T.vec(4, T.f32))
    accumulators = [acc_zero] * (m * columns)
    if prefetch_stages == 2 and full_k_chunks:
        current_k_base = lane * fx.Int32(b_load_width)
        current_b_chunks = [
            b_global.vec_load(
                (safe_columns[column_i], current_k_base),
                b_load_width,
            )
            for column_i in range_constexpr(columns)
        ]
        prefetch_split = _ceil_div(columns, 2)
        for k_chunk_i in range_constexpr(full_k_chunks - 1):
            next_k_base = (
                fx.Int32((k_chunk_i + 1) * k_chunk)
                + lane * fx.Int32(b_load_width)
            )
            next_b_chunks = [None] * columns
            for column_i in range_constexpr(prefetch_split):
                next_b_chunks[column_i] = b_global.vec_load(
                    (safe_columns[column_i], next_k_base),
                    b_load_width,
                )
            a_chunks = [
                a_smem.vec_load(
                    (
                        fx.Index(
                            fx.Int32(row * staged_k) + current_k_base
                        ),
                    ),
                    b_load_width,
                )
                for row in range_constexpr(m)
            ]
            accumulators = _accumulate_chunks(
                accumulators,
                a_chunks,
                current_b_chunks,
                column_start=0,
                column_end=prefetch_split,
                m=m,
                columns=columns,
                b_load_width=b_load_width,
            )
            for column_i in range_constexpr(prefetch_split, columns):
                next_b_chunks[column_i] = b_global.vec_load(
                    (safe_columns[column_i], next_k_base),
                    b_load_width,
                )
            accumulators = _accumulate_chunks(
                accumulators,
                a_chunks,
                current_b_chunks,
                column_start=prefetch_split,
                column_end=columns,
                m=m,
                columns=columns,
                b_load_width=b_load_width,
            )
            current_k_base = next_k_base
            current_b_chunks = next_b_chunks
        a_chunks = [
            a_smem.vec_load(
                (fx.Index(fx.Int32(row * staged_k) + current_k_base),),
                b_load_width,
            )
            for row in range_constexpr(m)
        ]
        accumulators = _accumulate_chunks(
            accumulators,
            a_chunks,
            current_b_chunks,
            column_start=0,
            column_end=columns,
            m=m,
            columns=columns,
            b_load_width=b_load_width,
        )
    else:
        accumulators = _accumulate_k_chunks_runtime(
            accumulators,
            a_smem=a_smem,
            b_global=b_global,
            safe_columns=safe_columns,
            lane=lane,
            m=m,
            columns=columns,
            b_load_width=b_load_width,
            full_k_chunks=full_k_chunks,
            staged_k=staged_k,
            k_chunk=k_chunk,
        )

    for tail_tile in range_constexpr(tail_mfma_tiles):
        k_base = (
            fx.Int32(
                full_k_chunks * k_chunk
                + tail_tile * WAVE_SIZE * MFMA_K
            )
            + lane * fx.Int32(MFMA_K)
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
                cache_modifier,
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


@functools.lru_cache(maxsize=512)
def _compile_gemm_decode_persistent_bf16(
    m: int,
    n: int,
    k: int,
    num_cus: int,
    config: PersistentDecodeConfig,
    gpu_arch: str,
):
    """Compile one shape/configuration of the persistent decode kernel."""
    if not (1 <= m <= 4):
        raise ValueError("persistent BF16 decode supports m in [1, 4]")
    if n <= 0 or k <= 0 or num_cus <= 0:
        raise ValueError("n, k, and num_cus must be positive")
    config.validate(m=m, k=k, gpu_arch=gpu_arch)

    waves = config.waves_per_workgroup
    columns = config.columns_per_wave
    block_threads = waves * WAVE_SIZE
    columns_per_workgroup = waves * columns
    logical_workgroups = _ceil_div(n, columns_per_workgroup)
    grid_workgroups = min(logical_workgroups, num_cus * config.workgroups_per_cu)
    persistent_turns = _ceil_div(logical_workgroups, grid_workgroups)
    has_k_tail = k % (WAVE_SIZE * K_CHUNK) != 0
    gfx942_full_k_chunks = k // (WAVE_SIZE * K_CHUNK)
    k_unroll = K_UNROLL if columns == 2 or m == 1 else 1
    full_k_groups = _ceil_div(gfx942_full_k_chunks, k_unroll)
    b_load_width = config.b_load_width
    k_chunk = WAVE_SIZE * b_load_width
    full_k_chunks = k // k_chunk
    tail_mfma_tiles = _ceil_div(k % k_chunk, WAVE_SIZE * MFMA_K)
    staged_k = _staged_k(k)
    activation_elements = m * staged_k
    activation_vectors_per_row = k // STAGE_VECTOR
    activation_vectors = m * activation_vectors_per_row
    activation_tail_per_row = staged_k - activation_vectors_per_row * STAGE_VECTOR
    activation_tail_elements = m * activation_tail_per_row
    stage_iterations = _ceil_div(activation_vectors, block_threads)
    kernel_name = (
        f"gemm_decode_persistent_bf16_m{m}_n{n}_k{k}_w{waves}_c{columns}_"
        f"g{config.workgroups_per_cu}_v{b_load_width}_p{config.prefetch_stages}"
    )

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=(
            f"gemm_decode_persistent_smem_{m}_{n}_{k}_{waves}_{columns}_"
            f"{config.workgroups_per_cu}_{b_load_width}_{config.prefetch_stages}"
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
        full_k_groups_runtime: Int32,
        runtime_full_k_chunks: fx.Int32,
        runtime_persistent_turns: fx.Int32,
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
                a_smem.vec_store(
                    (fx.Index(smem_element),),
                    staged,
                    STAGE_VECTOR,
                )
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
        if gpu_arch == "gfx942":
            acc_zero = arith.constant_vector(0.0, T.vec(4, T.f32))

            def compute_full_k_group(
                k_group_base,
                accumulators,
                safe_columns,
                group_chunks,
            ):
                k_bases = [
                    k_group_base
                    + fx.Int32(unroll_i * WAVE_SIZE * K_CHUNK)
                    + lane * fx.Int32(K_CHUNK)
                    for unroll_i in range_constexpr(group_chunks)
                ]
                a_chunks = [
                    [
                        a_smem.vec_load(
                            (
                                fx.Index(
                                    fx.Int32(row * staged_k)
                                    + k_bases[unroll_i]
                                ),
                            ),
                            K_CHUNK,
                        )
                        for row in range_constexpr(m)
                    ]
                    for unroll_i in range_constexpr(group_chunks)
                ]
                preload_columns = 2 if config.preload_b_tile else 1
                column_groups = _ceil_div(columns, preload_columns)
                for column_group in range_constexpr(column_groups):
                    first_group_column = column_group * preload_columns
                    group_columns = min(
                        preload_columns,
                        columns - first_group_column,
                    )
                    # The c2 long-K path mirrors the native UNRL-major schedule,
                    # which lowers to progressive vmcnt waits. Keep the proven
                    # column-major c4 schedule unchanged.
                    fragment_order = (
                        tuple(
                            (unroll_i, group_col)
                            for unroll_i in range_constexpr(group_chunks)
                            for group_col in range_constexpr(group_columns)
                        )
                        if config.preload_b_tile and columns == 2
                        else tuple(
                            (unroll_i, group_col)
                            for group_col in range_constexpr(group_columns)
                            for unroll_i in range_constexpr(group_chunks)
                        )
                    )
                    b_tile = {
                        (unroll_i, group_col): b_global.vec_load(
                            (
                                safe_columns[first_group_column + group_col],
                                k_bases[unroll_i],
                            ),
                            K_CHUNK,
                        )
                        for unroll_i, group_col in fragment_order
                    }
                    for unroll_i, group_col in fragment_order:
                        column_i = first_group_column + group_col
                        b_chunk = b_tile[(unroll_i, group_col)]
                        for fragment_i in range_constexpr(
                            K_CHUNK // MFMA_K
                        ):
                            fragment_offset = fragment_i * MFMA_K
                            b_fragment = _bf16x4_fragment(
                                b_chunk,
                                fragment_offset,
                            )
                            for row in range_constexpr(m):
                                acc_index = row * columns + column_i
                                a_fragment = _bf16x4_fragment(
                                    a_chunks[unroll_i][row],
                                    fragment_offset,
                                )
                                accumulators[acc_index] = (
                                    _mfma_4x4x4_bf16(
                                        a_fragment,
                                        b_fragment,
                                        accumulators[acc_index],
                                    )
                                )
                return accumulators

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

                logical_workgroup = (
                    gpu.block_idx.x + fx.Int32(turn * grid_workgroups)
                )
                active_turn = scf.IfOp(
                    _raw(logical_workgroup < fx.Int32(logical_workgroups)),
                    results_=[],
                    has_else=False,
                )
                with ir.InsertionPoint(active_turn.then_block):
                    accumulators = [acc_zero] * (m * columns)
                    if const_expr(
                        config.preload_b_tile
                        and columns == 2
                        and full_k_chunks % k_unroll == 0
                    ):
                        init_state = [fx.Int32(0)] + accumulators
                        for _, state in range(
                            0,
                            full_k_groups_runtime,
                            1,
                            init=init_state,
                        ):
                            k_group_base = state[0]
                            accumulators = compute_full_k_group(
                                k_group_base,
                                list(state[1:]),
                                safe_columns,
                                k_unroll,
                            )
                            next_k_group_base = (
                                k_group_base
                                + fx.Int32(
                                    k_unroll * WAVE_SIZE * K_CHUNK
                                )
                            )
                            loop_results = yield [
                                next_k_group_base,
                                *accumulators,
                            ]
                        accumulators = list(loop_results[1:])
                    else:
                        for k_group in range_constexpr(full_k_groups):
                            group_chunks = min(
                                k_unroll,
                                full_k_chunks - k_group * k_unroll,
                            )
                            accumulators = compute_full_k_group(
                                fx.Int32(
                                    k_group
                                    * k_unroll
                                    * WAVE_SIZE
                                    * K_CHUNK
                                ),
                                accumulators,
                                safe_columns,
                                group_chunks,
                            )

                    if has_k_tail:
                        k_base = fx.Int32(
                            full_k_chunks * WAVE_SIZE * K_CHUNK
                        ) + lane * fx.Int32(K_CHUNK)
                        valid_base = k_base < fx.Int32(k)
                        safe_k_base = ArithValue(_raw(valid_base)).select(
                            k_base,
                            fx.Int32(0),
                        )
                        a_chunks = []
                        for row in range_constexpr(m):
                            a_chunks.append(
                                a_smem.vec_load(
                                    (
                                        fx.Index(
                                            fx.Int32(row * staged_k) + safe_k_base
                                        ),
                                    ),
                                    K_CHUNK,
                                )
                            )
                        for column_i in range_constexpr(columns):
                            b_row_base = safe_columns[column_i] * fx.Int32(k)
                            b_chunk = _masked_bf16x8(
                                b_global.rsrc,
                                b_row_base,
                                k_base,
                                valid_base,
                                k,
                                config.b_cache_modifier,
                            )
                            for fragment_i in range_constexpr(K_CHUNK // MFMA_K):
                                fragment_offset = fragment_i * MFMA_K
                                b_fragment = _bf16x4_fragment(
                                    b_chunk, fragment_offset
                                )
                                for row in range_constexpr(m):
                                    acc_index = row * columns + column_i
                                    a_fragment = _bf16x4_fragment(
                                        a_chunks[row], fragment_offset
                                    )
                                    accumulators[acc_index] = _mfma_4x4x4_bf16(
                                        a_fragment,
                                        b_fragment,
                                        accumulators[acc_index],
                                    )

                    reduced = [
                        _reduce_mfma_scalar(accumulator)
                        for accumulator in accumulators
                    ]
                    for row in range_constexpr(m):
                        for column_i in range_constexpr(columns):
                            store_valid = (
                                lane == fx.Int32(WAVE_SIZE - 1)
                            ) & column_valid[column_i]
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
                                c_global[
                                    row,
                                    column_base + fx.Int32(column_i),
                                ] = output
                                scf.YieldOp([])
                    scf.YieldOp([])
        else:
            if const_expr(persistent_turns == 1):
                _compute_persistent_column_tile(
                    b_global=b_global,
                    c_global=c_global,
                    a_smem=a_smem,
                    column_base=first_column,
                    lane=lane,
                    m=m,
                    n=n,
                    k=k,
                    columns=columns,
                    b_load_width=b_load_width,
                    prefetch_stages=config.prefetch_stages,
                    cache_modifier=config.b_cache_modifier,
                    full_k_chunks=(
                        full_k_chunks
                        if config.prefetch_stages == 2
                        else runtime_full_k_chunks
                    ),
                    tail_mfma_tiles=tail_mfma_tiles,
                    staged_k=staged_k,
                    k_chunk=k_chunk,
                )
            else:
                turn = fx.Int32(0)
                while turn < runtime_persistent_turns:
                    column_base = first_column + turn * column_stride
                    # Keep LDS reads inside the persistent loop. Without a memory
                    # fence, LLVM hoists every A chunk and spills the cached matrix.
                    _compiler_memory_barrier()
                    if column_base < fx.Int32(n):
                        _compute_persistent_column_tile(
                            b_global=b_global,
                            c_global=c_global,
                            a_smem=a_smem,
                            column_base=column_base,
                            lane=lane,
                            m=m,
                            n=n,
                            k=k,
                            columns=columns,
                            b_load_width=b_load_width,
                            prefetch_stages=config.prefetch_stages,
                            cache_modifier=config.b_cache_modifier,
                            full_k_chunks=(
                                full_k_chunks
                                if config.prefetch_stages == 2
                                else runtime_full_k_chunks
                            ),
                            tail_mfma_tiles=tail_mfma_tiles,
                            staged_k=staged_k,
                            k_chunk=k_chunk,
                        )
                    turn = turn + fx.Int32(1)

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
            fx.Int32(full_k_groups),
            fx.Int32(full_k_chunks),
            fx.Int32(persistent_turns),
            value_attrs=attrs,
        ).launch(
            grid=(grid_workgroups, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch.kernel_name = kernel_name
    launch.lds_bytes = _align_up(activation_elements * DTYPE_BYTES, 128)
    launch.grid_workgroups = grid_workgroups
    launch.persistent_turns = persistent_turns
    return launch


def compile_gemm_decode_persistent_bf16(
    m: int,
    n: int,
    k: int,
    num_cus: int,
    config: PersistentDecodeConfig = DEFAULT_CONFIG,
):
    """Compile one shape/config with the target architecture in the cache key."""
    return _compile_gemm_decode_persistent_bf16(
        m,
        n,
        k,
        num_cus,
        config,
        get_rocm_arch(),
    )


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
        config = select_persistent_decode_config(m, n, k, num_cus)
    launcher = compile_gemm_decode_persistent_bf16(m, n, k, num_cus, config)
    launcher(A, B, C, stream=stream)
    return C
