# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL AttnRes prefill kernel for Kimi-K3, on the ``attn_res_gate`` contract.

``flydsl_attn_res`` takes the arguments of Triton ``attn_res_gate`` and returns
the same outputs. The candidates are the ``B`` rows of ``block_residual`` plus
the prefix; the prefix candidate is the FP32 sum ``prefix + add_hidden [+
add_hidden2]``, used unrounded. Scores use RMS-normalized candidates against the
pre-folded ``score_weight``, and an online softmax mixes the raw values.
"""

# Do not add ``from __future__ import annotations``: FlyDSL inspects annotations
# at trace time and PEP 563 can interfere with its runtime argument detection.

import math
from dataclasses import dataclass
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime

from . import dpp_utils
from .tensor_shim import _run_compiled

KERNEL_NAME = "flydsl_attn_res"

_MAX_BLOCKS = 8
_WARP_SIZE = 64
_VEC_WIDTH = 8  # 8 bf16 values = one 128-bit global-memory transaction.
_TAIL_VEC_WIDTH = 4  # 4 bf16 values = one 64-bit transaction for a row's tail tile.
_TILE_CANDIDATES = (2, 4, 8, 1)
_MAX_BLOCK_THREADS = 1024
# Buffer loads and stores take a 32-bit byte offset from the tensor's base.
_BUFFER_LIMIT_BYTES = 1 << 32
# 2 is the non-temporal cache policy. Outputs are write-once and never read back.
# Block sources are read once per launch, so they use it only where
# _ArchTuning.nontemporal_block_loads is set.
_NONTEMPORAL_CACHE_MODIFIER = 2
# Multi-wave rows on the wide grid.
_WIDE_ROW_THREADS = 256
_WIDE_SOURCE_PREFETCH_IN_FLIGHT = 1
# Two sources fill one batched wave-sum register; one source per barrier lost
# 3 to 7% at T=1024.
_WIDE_SOURCE_GROUP_SIZE = 2
# On the 256-thread grid the BF16 dot wins at T=320 to 2048 but loses 2.5 to 3%
# at T=8192, where the grid is bandwidth-bound. Above this limit it uses FP32
# math.
_WIDE_BF16_DOT_MAX_TOKENS = 4096
_LOG2E = math.log2(math.e)
_DPP_QUAD_XOR1 = 0xB1
_DPP_QUAD_XOR2 = 0x4E
_DPP_ROW_SHR1 = 0x111
_DPP_ROW_SHR2 = 0x112
_DPP_ROW_SHR4 = 0x114
_DPP_ROW_SHR8 = 0x118
_DPP_ROW_BCAST15 = 0x142
_DPP_ROW_BCAST31 = 0x143
_DPP_ROW_MIRROR = 0x140
_DPP_ROW_HALF_MIRROR = 0x141
_DPP_ROW_MASK = 0xF
_ROW_LANES = 16
# row_bcast:15 moves row 0 -> 1 and row 2 -> 3.
_DPP_ROW_BCAST15_MASK = 0xA
# row_bcast:31 moves the row-0/1 total into rows 2 and 3.
_DPP_ROW_BCAST31_MASK = 0xC


def _dpp_row_f32(
    value,
    dpp_ctrl: int,
    row_mask: int = _DPP_ROW_MASK,
    bank_mask: int = _DPP_ROW_MASK,
):
    """Return the DPP row value, with zero for masked or out-of-bounds lanes."""
    bits = value.bitcast(fx.Int32)
    peer = dpp_utils.update_dpp_i32(
        fx.Int32(0), bits, dpp_ctrl, row_mask, bank_mask, True
    )
    return fx.Int32(peer).bitcast(fx.Float32)


def _combine_f32(a, b, op: str, fastmath):
    """``max`` treats zero as identity, so inputs must be non-negative.

    Masked DPP lanes and idle LDS slots read zero.
    """
    if op == "max":
        return a.maximumf(b)
    return a.addf(b, fastmath=fastmath)


def _wave_reduce_to_one_lane_f32(value, fastmath, op: str = "add"):
    """Wave64 sum or max via a ``row_shr`` cascade into ``row_bcast``.

    Lane 63 holds the total; ``readlane`` moves it into an SGPR, so the
    result is wave-uniform and every lane can consume it without a
    broadcast.
    """
    reduced = value
    for dpp_ctrl in (
        _DPP_ROW_SHR1,
        _DPP_ROW_SHR2,
        _DPP_ROW_SHR4,
        _DPP_ROW_SHR8,
    ):
        reduced = _combine_f32(reduced, _dpp_row_f32(reduced, dpp_ctrl), op, fastmath)
    for dpp_ctrl, row_mask in (
        (_DPP_ROW_BCAST15, _DPP_ROW_BCAST15_MASK),
        (_DPP_ROW_BCAST31, _DPP_ROW_BCAST31_MASK),
    ):
        peer = _dpp_row_f32(reduced, dpp_ctrl, row_mask)
        reduced = _combine_f32(reduced, peer, op, fastmath)
    bits = reduced.bitcast(fx.Int32)
    return fx.Int32(fx.rocdl.readlane(T.i32, bits, 63)).bitcast(fx.Float32)


def _has_gfx950_ops() -> bool:
    """Whether the target has the permlane swaps and the packed BF16 dot product."""
    return get_gfx_runtime() == "gfx950"


def _bf16_dot2(a, b, accumulator):
    return fx.Float32(
        fx.rocdl.fdot2_f32_bf16_(T.f32, a, b, accumulator.ir_value(), clamp=False)
    )


def _exchange_halves(a, b, offset: int, lane, native: bool):
    """Return ``(x, y)`` with ``x + y`` equal to ``a[l] + a[l ^ offset]`` on lanes in
    even ``offset``-lane blocks and ``b[l] + b[l ^ offset]`` on lanes in odd blocks.
    """
    if native:
        swap = fx.rocdl.permlane32_swap if offset == 32 else fx.rocdl.permlane16_swap
        pair = swap(
            llvm.StructType.get_literal([T.i32, T.i32]),
            a.bitcast(fx.Int32).ir_value(),
            b.bitcast(fx.Int32).ir_value(),
            False,
            False,
        )
        return tuple(
            fx.Int32(llvm.extractvalue(T.i32, pair, [index])).bitcast(fx.Float32)
            for index in range(2)
        )
    keeps_a = (lane // offset) % 2 == 0
    received = keeps_a.select(b, a).shuffle_xor(offset, _WARP_SIZE)
    return keeps_a.select(a, received), keeps_a.select(received, b)


def _row_reduce_f32(value, span: int, fastmath, op: str = "add"):
    """Reduce each aligned group of ``span`` lanes (2, 4, 8, or 16). Every lane gets the result."""
    reduced = value
    for dpp_ctrl in (
        _DPP_QUAD_XOR1,
        _DPP_QUAD_XOR2,
        _DPP_ROW_HALF_MIRROR,
        _DPP_ROW_MIRROR,
    )[: int(math.log2(span))]:
        reduced = _combine_f32(reduced, _dpp_row_f32(reduced, dpp_ctrl), op, fastmath)
    return reduced


def _wave_sum_batched(values, lane, fastmath, native: bool):
    """Return ``ceil(len(values) / 4)`` registers; every lane of 16-lane row ``g`` of
    register ``k`` holds the wave total of ``values[4 * k + g]``.
    """
    zero = fx.Float32(0.0)
    totals = []
    for first in range_constexpr(0, len(values), 4):
        quad = [
            values[first + g] if first + g < len(values) else zero for g in range(4)
        ]
        x, y = _exchange_halves(quad[0], quad[2], 32, lane, native)
        low = x + y
        x, y = _exchange_halves(quad[1], quad[3], 32, lane, native)
        high = x + y
        x, y = _exchange_halves(low, high, 16, lane, native)
        totals.append(_row_reduce_f32(x + y, 16, fastmath))
    return totals


def _pack_fp8(values, width: int):
    """Bytes are the arch's e4m3 variant, matching ``dtypes.fp8``."""
    words = []
    for word in range_constexpr(width // 4):
        lo = fx.rocdl.cvt_pk_fp8_f32(
            T.i32, values[4 * word], values[4 * word + 1], fx.Int32(0), False
        )
        hi = fx.rocdl.cvt_pk_fp8_f32(
            T.i32, values[4 * word + 2], values[4 * word + 3], lo, True
        )
        words.append(fx.Int32(hi))
    return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.Int8)


def _wide_row_supported(hidden_size: int) -> bool:
    """Return whether 256 threads split the row into 128-bit tiles and at most one 64-bit tail."""
    if hidden_size % _WIDE_ROW_THREADS:
        return False
    elems_per_thread = hidden_size // _WIDE_ROW_THREADS
    return elems_per_thread >= _VEC_WIDTH and not elems_per_thread % _TAIL_VEC_WIDTH


@lru_cache(maxsize=64)
def _num_cus(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _block_threads_for(num_vec: int) -> int:
    """Return a wave-aligned thread count that prefers two vector tiles per thread."""
    if num_vec <= 0:
        raise ValueError(f"num_vec must be positive, got {num_vec}")

    for tiles_per_thread in _TILE_CANDIDATES:
        if num_vec % tiles_per_thread:
            continue
        row_threads = num_vec // tiles_per_thread
        if row_threads % _WARP_SIZE == 0 and row_threads <= _MAX_BLOCK_THREADS:
            return row_threads

    raise ValueError(f"no wave-aligned block size covers {num_vec} vectors per row")


def _resolve_rows_per_wg(row_threads: int, rows_per_wg: int | None) -> int:
    """Return a validated rows-per-workgroup; packing is one-wave rows only."""
    if rows_per_wg is None:
        return 1
    if rows_per_wg < 1:
        raise ValueError(f"rows_per_wg must be >= 1, got {rows_per_wg}")
    if rows_per_wg != 1 and row_threads != _WARP_SIZE:
        raise ValueError(
            "packing multiple rows per workgroup is only supported for "
            f"one-wave rows ({_WARP_SIZE} threads), got row_threads={row_threads}"
        )
    max_rows = _MAX_BLOCK_THREADS // row_threads
    if rows_per_wg > max_rows:
        raise ValueError(
            f"rows_per_wg={rows_per_wg} exceeds max {max_rows} "
            f"for row_threads={row_threads} (block limit {_MAX_BLOCK_THREADS})"
        )
    return rows_per_wg


@dataclass(frozen=True)
class _ArchTuning:
    """Launch and codegen choices for one GPU arch.

    The gfx942 values are measured on MI300X. gfx950 keeps its earlier values
    because these levers are unmeasured there.
    """

    nontemporal_block_loads: bool
    # Block-source loads in flight on the narrow grid; the prefix is already in
    # registers.
    narrow_prefetch_in_flight: int
    wide_grid_cu_divisor: int
    alias_wide_grid_cu_divisor: int


_ARCH_TUNING = {
    "gfx942": _ArchTuning(
        nontemporal_block_loads=True,
        narrow_prefetch_in_flight=8,
        wide_grid_cu_divisor=2,
        alias_wide_grid_cu_divisor=1,
    ),
    "gfx950": _ArchTuning(
        nontemporal_block_loads=False,
        narrow_prefetch_in_flight=3,
        wide_grid_cu_divisor=1,
        alias_wide_grid_cu_divisor=1,
    ),
}


@dataclass(frozen=True)
class _AttnResSpec:
    """Must hold every compile-time choice: it is the ``_build_attn_res`` cache key."""

    hidden_size: int
    num_blocks: int
    has_add: bool
    has_add2: bool
    close_block: bool
    output_norm: bool
    eps: float
    output_rms_eps: float
    scale: float
    rows_per_wg: int
    wide_grid: bool
    native_swap: bool
    nontemporal_block_loads: bool
    narrow_prefetch_in_flight: int
    fp8_dtype: torch.dtype | None = None
    block_alias: bool = False
    bf16_dot: bool = False

    @property
    def kernel_name(self) -> str:
        return (
            f"{KERNEL_NAME}_d{self.hidden_size}_b{self.num_blocks}"
            f"_r{self.rows_per_wg}_add{int(self.has_add) + int(self.has_add2)}"
            f"_close{int(self.close_block)}{'a' if self.block_alias else ''}"
            f"_onorm{int(self.output_norm)}"
            f"{'_fp8' if self.fp8_dtype is not None else ''}"
            f"{'_wide' if self.wide_grid else ''}"
            f"{'_bf16dot' if self.bf16_dot else ''}"
        )


_LAUNCH_PLANS: dict = {}
# One plan per distinct T. Past about 2000 live keys, cyclic GC makes a miss
# slower.
_MAX_LAUNCH_PLANS = 1024
_ALIGNMENT_MASK = _VEC_WIDTH * 2 - 1


def _clear_caches() -> None:
    _build_attn_res.cache_clear()
    _LAUNCH_PLANS.clear()


@lru_cache(maxsize=256)
def _build_attn_res(spec: _AttnResSpec):
    """Return one fixed-shape, compile-time-specialized AttnRes launcher."""
    hidden_size = spec.hidden_size
    num_blocks = spec.num_blocks
    has_add = spec.has_add
    has_add2 = spec.has_add2
    close_block = spec.close_block
    block_alias = spec.block_alias
    output_norm = spec.output_norm
    eps = spec.eps
    output_rms_eps = spec.output_rms_eps
    scale = spec.scale
    wide_grid = spec.wide_grid
    nontemporal_block_loads = spec.nontemporal_block_loads
    fp8 = spec.fp8_dtype is not None
    fp8_max = torch.finfo(spec.fp8_dtype).max if fp8 else 1.0
    if hidden_size <= 0 or hidden_size % _VEC_WIDTH:
        raise ValueError(f"D={hidden_size} must be a positive multiple of {_VEC_WIDTH}")
    if not 0 <= num_blocks <= _MAX_BLOCKS:
        raise ValueError(f"num_blocks must be in [0, {_MAX_BLOCKS}], got {num_blocks}")

    num_vec = hidden_size // _VEC_WIDTH
    row_threads = _block_threads_for(num_vec)
    if wide_grid:
        if row_threads == _WARP_SIZE or not _wide_row_supported(hidden_size):
            raise ValueError(
                f"wide_grid needs a multi-wave row that {_WIDE_ROW_THREADS} threads "
                f"split into 128-bit tiles plus a 64-bit tail, got D={hidden_size}"
            )
        row_threads = _WIDE_ROW_THREADS
    rows_per_wg = _resolve_rows_per_wg(row_threads, spec.rows_per_wg)
    block_threads = row_threads * rows_per_wg
    elems_per_thread = hidden_size // row_threads
    full_tiles = elems_per_thread // _VEC_WIDTH
    tile_widths = (_VEC_WIDTH,) * full_tiles
    tile_bases = tuple(tile * row_threads for tile in range(full_tiles))
    if elems_per_thread % _VEC_WIDTH:
        tile_widths += (_TAIL_VEC_WIDTH,)
        tile_bases += (full_tiles * 2 * row_threads,)
    tiles_per_thread = len(tile_widths)
    red_slots = row_threads // _WARP_SIZE
    num_sources = num_blocks + 1
    prefetch = (
        _WIDE_SOURCE_PREFETCH_IN_FLIGHT if wide_grid else spec.narrow_prefetch_in_flight
    )
    prefetch_in_flight = min(prefetch, max(num_blocks - 1, 0))
    prefix_slot = 0 if wide_grid else num_blocks
    first_block_slot = 1 if wide_grid else 0
    group_size = _WIDE_SOURCE_GROUP_SIZE if wide_grid else num_sources
    num_groups = (num_sources + group_size - 1) // group_size
    if red_slots > _ROW_LANES:
        raise ValueError(f"at most {_ROW_LANES} waves per row, got {red_slots}")

    native_swap = spec.native_swap
    bf16_dot = spec.bf16_dot
    # Alternate buffers per group so the next group's writes need no second
    # barrier. Waves pad to a power of two so a DPP row reduce over wave_stride
    # lanes sums one value.
    wave_stride = 1 << max(red_slots - 1, 0).bit_length()
    values_per_reg = _WARP_SIZE // wave_stride
    values_per_buf = -(-2 * group_size // values_per_reg) * values_per_reg

    def partial_index(buf, value):
        return (buf * values_per_buf + value) * wave_stride

    @fx.struct
    class SharedStorage:
        partials: fx.Array[fx.Float32, 2 * values_per_buf * wave_stride, 16]

    @flyc.kernel(name=spec.kernel_name, known_block_size=[block_threads, 1, 1])
    def attn_res_kernel(
        prefix: fx.Tensor,
        add_hidden: fx.Tensor,
        add_hidden2: fx.Tensor,
        block_residual: fx.Tensor,
        score_weight: fx.Tensor,
        output_rms_weight: fx.Tensor,
        y: fx.Tensor,
        y_scale: fx.Tensor,
        prefix_out: fx.Tensor,
        block_out: fx.Tensor,
        num_tokens: fx.Int32,
    ):
        tid = fx.thread_idx.x
        row_owner = fx.make_layout((rows_per_wg, row_threads), (row_threads, 1))
        row_coord = fx.idx2crd(fx.Int32(tid), row_owner)
        wg_row = fx.Int32(fx.get(row_coord, 0))
        local_tid = fx.Int32(fx.get(row_coord, 1))
        token = fx.block_idx.x * rows_per_wg + wg_row
        fm_fast = arith.FastMathFlags.fast

        def vector_index_for(tile):
            return local_tid + tile_bases[tile]

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_partials = lds.partials.view(
            fx.make_layout(2 * values_per_buf * wave_stride, 1)
        )
        lane = local_tid % _WARP_SIZE
        wave = local_tid // _WARP_SIZE
        wave_slot = lane % wave_stride

        zero = fx.Float32(0.0)
        neg_inf = fx.Float32(float("-inf"))
        inv_hidden_size = 1.0 / float(hidden_size)
        log2e = _LOG2E

        # Without the pin the compiler sinks each FP32 mix update to the final
        # store and keeps every source live.
        def pin_registers(values):
            llvm.inline_asm(
                None,
                [value.ir_value() for value in values],
                "",
                ",".join("v" for _ in values),
                has_side_effects=True,
            )

        def store_reduce(value, op="add"):
            return _wave_reduce_to_one_lane_f32(value, fm_fast, op)

        def store_group_partials(pairs, buf):
            totals = _wave_sum_batched(
                [value for pair in pairs for value in pair], lane, fm_fast, native_swap
            )
            if lane % _ROW_LANES == 0:
                row = lane // _ROW_LANES
                for register in range_constexpr(len(totals)):
                    fx.memref_store(
                        totals[register],
                        s_partials,
                        partial_index(buf, 4 * register) + row * wave_stride + wave,
                    )

        def uniform_value(registers, value, values_per_register, lanes_per_value):
            register, row = divmod(value, values_per_register)
            bits = registers[register].bitcast(fx.Int32)
            return fx.Int32(
                fx.rocdl.readlane(T.i32, bits, row * lanes_per_value)
            ).bitcast(fx.Float32)

        def finish_group_reduce(n_members, buf):
            is_wave = wave_slot < red_slots
            totals = []
            for register in range_constexpr(-(-2 * n_members // values_per_reg)):
                partial = fx.memref_load(
                    s_partials, partial_index(buf, 0) + register * _WARP_SIZE + lane
                )
                totals.append(
                    _row_reduce_f32(is_wave.select(partial, zero), wave_stride, fm_fast)
                )
            return [
                tuple(
                    uniform_value(totals, 2 * slot + kind, values_per_reg, wave_stride)
                    for kind in range(2)
                )
                for slot in range(n_members)
            ]

        def reduce_group(pairs, buf):
            if const_expr(red_slots == 1):
                totals = _wave_sum_batched(
                    [value for pair in pairs for value in pair],
                    lane,
                    fm_fast,
                    native_swap,
                )
                return [
                    tuple(
                        uniform_value(totals, 2 * slot + kind, 4, _ROW_LANES)
                        for kind in range(2)
                    )
                    for slot in range(len(pairs))
                ]
            store_group_partials(pairs, buf)
            fx.gpu.barrier()
            return finish_group_reduce(len(pairs), buf)

        def block_reduce(value, value_index, buf, op):
            if const_expr(red_slots == 1):
                return store_reduce(value, op)

            wave_total = store_reduce(value, op)
            base = partial_index(buf, value_index)

            if lane == 0:
                fx.memref_store(wave_total, s_partials, base + wave)
            fx.gpu.barrier()

            partial = fx.memref_load(s_partials, base + wave_slot)
            partial = (wave_slot < red_slots).select(partial, zero)
            reduced = _row_reduce_f32(partial, wave_stride, fm_fast, op)
            bits = reduced.bitcast(fx.Int32)
            return fx.Int32(fx.rocdl.readfirstlane(T.i32, bits)).bitcast(fx.Float32)

        copy_atoms = {
            width: fx.make_copy_atom(copy_op(), fx.BFloat16)
            for width, copy_op in (
                (_VEC_WIDTH, fx.rocdl.BufferCopy128b),
                (_TAIL_VEC_WIDTH, fx.rocdl.BufferCopy64b),
            )
            if width in tile_widths
        }
        nontemporal_atoms = {
            width: fx.make_copy_atom(copy_op(_NONTEMPORAL_CACHE_MODIFIER), fx.BFloat16)
            for width, copy_op in (
                (_VEC_WIDTH, fx.rocdl.BufferCopy128b),
                (_TAIL_VEC_WIDTH, fx.rocdl.BufferCopy64b),
            )
            if width in tile_widths
        }
        block_atoms = (
            nontemporal_atoms if const_expr(nontemporal_block_loads) else copy_atoms
        )
        vector_layouts = {width: fx.make_layout(width, 1) for width in copy_atoms}

        def divide_row(row):
            return {
                width: fx.logical_divide(row, layout)
                for width, layout in vector_layouts.items()
            }

        def issue_bf16_vec(divided, tile, atoms=copy_atoms):
            width = tile_widths[tile]
            register = fx.make_rmem_tensor(width, fx.BFloat16)
            fx.copy(
                atoms[width],
                fx.slice(divided[width], (None, vector_index_for(tile))),
                register,
            )
            return register

        def load_bf16_vec(divided, tile):
            return fx.memref_load_vec(issue_bf16_vec(divided, tile))

        def store_bf16_vec(value, divided, tile):
            width = tile_widths[tile]
            register = fx.make_rmem_tensor(width, fx.BFloat16)
            fx.memref_store_vec(value, register)
            fx.copy(
                nontemporal_atoms[width],
                register,
                fx.slice(divided[width], (None, vector_index_for(tile))),
            )

        fp8_atoms = {
            width: fx.make_copy_atom(copy_op(_NONTEMPORAL_CACHE_MODIFIER), fx.Int8)
            for width, copy_op in (
                (_VEC_WIDTH, fx.rocdl.BufferCopy64b),
                (_TAIL_VEC_WIDTH, fx.rocdl.BufferCopy32b),
            )
            if width in tile_widths
        }

        def store_fp8_vec(value, divided, tile):
            width = tile_widths[tile]
            register = fx.make_rmem_tensor(width, fx.Int8)
            fx.memref_store_vec(_pack_fp8(value, width), register)
            fx.copy(
                fp8_atoms[width],
                register,
                fx.slice(divided[width], (None, vector_index_for(tile))),
            )

        def store_scale(value):
            register = fx.make_rmem_tensor(1, fx.Float32)
            fx.memref_store_vec(fx.full(1, value, fx.Float32), register)
            scale_row = fx.slice(fx.rocdl.make_buffer_tensor(y_scale), (token, None))
            fx.copy(
                fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32),
                register,
                fx.slice(fx.logical_divide(scale_row, fx.make_layout(1, 1)), (None, 0)),
            )

        def row_of(buffer, *index):
            return divide_row(fx.slice(buffer, (*index, None)))

        prefix_buf = fx.rocdl.make_buffer_tensor(prefix)
        score_weight_buf = fx.rocdl.make_buffer_tensor(score_weight)
        y_buf = fx.rocdl.make_buffer_tensor(y)
        if const_expr(num_blocks):
            blocks_buf = fx.rocdl.make_buffer_tensor(block_residual)
        if const_expr(close_block):
            block_out_buf = fx.rocdl.make_buffer_tensor(block_out)

        def run_row():
            prefix_div = row_of(prefix_buf, token)
            score_weight_div = divide_row(score_weight_buf)
            y_div = row_of(y_buf, token)

            def issue_block_source(block):
                block_div = row_of(blocks_buf, token, block)
                handles = []
                for tile in range_constexpr(tiles_per_thread):
                    handles.append(issue_bf16_vec(block_div, tile, block_atoms))
                return handles

            def load_issued_source(handles):
                source_local = []
                for tile in range_constexpr(tiles_per_thread):
                    source_local.append(fx.memref_load_vec(handles[tile]))
                return source_local

            def take_block_source(block):
                source_local = load_issued_source(in_flight.pop(0))
                if const_expr(close_block and not block_alias):
                    block_out_div = row_of(block_out_buf, token, block)
                    for tile in range_constexpr(tiles_per_thread):
                        store_bf16_vec(source_local[tile], block_out_div, tile)
                return source_local

            # Issue the first block loads before the prefix prologue so their
            # latency overlaps it.
            in_flight = []
            for source in range_constexpr(min(prefetch_in_flight, num_blocks)):
                in_flight.append(issue_block_source(source))

            # Mix the unrounded FP32 sum, as attn_res_gate does; only the stores
            # round to BF16.
            q_local = []
            prefix_local = []
            prefix_rounded = []
            if const_expr(has_add):
                add_div = row_of(fx.rocdl.make_buffer_tensor(add_hidden), token)
                if const_expr(has_add2):
                    add2_div = row_of(fx.rocdl.make_buffer_tensor(add_hidden2), token)
            for tile in range_constexpr(tiles_per_thread):
                q_local.append(load_bf16_vec(score_weight_div, tile))
                prefix_value = load_bf16_vec(prefix_div, tile)
                if const_expr(has_add):
                    prefix_value = prefix_value.to(fx.Float32) + load_bf16_vec(
                        add_div, tile
                    ).to(fx.Float32)
                    if const_expr(has_add2):
                        prefix_value = prefix_value + load_bf16_vec(add2_div, tile).to(
                            fx.Float32
                        )
                    prefix_rounded.append(prefix_value.to(fx.BFloat16))
                else:
                    prefix_rounded.append(prefix_value)
                prefix_local.append(prefix_value)

            if const_expr(has_add):
                prefix_out_div = row_of(fx.rocdl.make_buffer_tensor(prefix_out), token)
                for tile in range_constexpr(tiles_per_thread):
                    store_bf16_vec(prefix_rounded[tile], prefix_out_div, tile)
            if const_expr(close_block):
                block_out_div = row_of(block_out_buf, token, num_blocks)
                for tile in range_constexpr(tiles_per_thread):
                    store_bf16_vec(prefix_rounded[tile], block_out_div, tile)

            mixed_local = [
                fx.Vector.filled(tile_widths[tile], 0.0, fx.Float32)
                for tile in range_constexpr(tiles_per_thread)
            ]
            max_logit = neg_inf
            denominator = zero

            def local_sumsq_dot(values_local, is_bf16):
                thread_sumsq = zero
                thread_dot = zero
                for tile in range_constexpr(tiles_per_thread):
                    if const_expr(bf16_dot and is_bf16):
                        for pair in range_constexpr(tile_widths[tile] // 2):
                            mask = [2 * pair, 2 * pair + 1]
                            value = values_local[tile].shuffle(values_local[tile], mask)
                            q = q_local[tile].shuffle(q_local[tile], mask)
                            thread_sumsq = _bf16_dot2(value, value, thread_sumsq)
                            thread_dot = _bf16_dot2(value, q, thread_dot)
                    else:
                        value = values_local[tile].to(fx.Float32)
                        thread_sumsq = thread_sumsq + (value * value).reduce(
                            ReductionOp.ADD, fastmath=fm_fast
                        )
                        thread_dot = thread_dot + (
                            value * q_local[tile].to(fx.Float32)
                        ).reduce(ReductionOp.ADD, fastmath=fm_fast)
                return thread_sumsq, thread_dot

            def update_softmax(
                group_values, group_sums, old_max, old_denominator, old_mixed
            ):
                logits = []
                for sumsq, dot in group_sums:
                    reciprocal_rms = fmath.rsqrt(
                        sumsq * inv_hidden_size + eps, fastmath=fm_fast
                    )
                    logit = dot * reciprocal_rms
                    if const_expr(scale != 1.0):
                        logit = logit * scale
                    logits.append(logit)

                new_max = old_max
                for logit in logits:
                    new_max = new_max.maximumf(logit)
                old_scale_active = fmath.exp2(
                    (old_max - new_max) * log2e, fastmath=fm_fast
                )
                old_scale = (old_max == neg_inf).select(zero, old_scale_active)
                new_scales = [
                    fmath.exp2((logit - new_max) * log2e, fastmath=fm_fast)
                    for logit in logits
                ]
                new_denominator = old_denominator * old_scale
                for new_scale in new_scales:
                    new_denominator = new_denominator + new_scale

                new_mixed = []
                for tile in range_constexpr(tiles_per_thread):
                    mixed = old_mixed[tile] * old_scale
                    for values_local, new_scale in zip(group_values, new_scales):
                        mixed = mixed + values_local[tile].to(fx.Float32) * new_scale
                    new_mixed.append(mixed)
                return new_max, new_denominator, new_mixed

            # Block sources stay BF16 across the barrier; widening them for the mix
            # is exact.
            for group in range_constexpr(num_groups):
                start = group * group_size
                n_members = min(group_size, num_sources - start)
                buf = group % 2
                group_local = []
                group_partials = []
                for slot in range_constexpr(n_members):
                    src = start + slot
                    if const_expr(src == prefix_slot):
                        source_local = prefix_local
                    else:
                        block = src - first_block_slot
                        if const_expr(block + prefetch_in_flight < num_blocks):
                            in_flight.append(
                                issue_block_source(block + prefetch_in_flight)
                            )
                        source_local = take_block_source(block)
                    group_partials.append(
                        local_sumsq_dot(
                            source_local, not (has_add and src == prefix_slot)
                        )
                    )
                    group_local.append(source_local)
                max_logit, denominator, mixed_local = update_softmax(
                    group_local,
                    reduce_group(group_partials, buf),
                    max_logit,
                    denominator,
                    mixed_local,
                )
                if const_expr(wide_grid):
                    pin_registers(mixed_local)

            # Use the LDS half the last group did not touch, so epilogue writes
            # cannot race waves still reading it.
            epilogue_buf = num_groups % 2
            if const_expr(output_norm):
                thread_sumsq = zero
                for tile in range_constexpr(tiles_per_thread):
                    thread_sumsq = thread_sumsq + (
                        mixed_local[tile] * mixed_local[tile]
                    ).reduce(ReductionOp.ADD, fastmath=fm_fast)
                sumsq = block_reduce(thread_sumsq, 0, epilogue_buf, "add")
                norm_scale = fmath.rsqrt(
                    sumsq * inv_hidden_size
                    + output_rms_eps * denominator * denominator,
                    fastmath=fm_fast,
                )

                output_rms_weight_div = divide_row(
                    fx.rocdl.make_buffer_tensor(output_rms_weight)
                )

                def y_tile(tile):
                    output_gamma = load_bf16_vec(output_rms_weight_div, tile).to(
                        fx.Float32
                    )
                    return mixed_local[tile] * norm_scale * output_gamma

            else:
                inverse_denominator = 1.0 / denominator

                def y_tile(tile):
                    return mixed_local[tile] * inverse_denominator

            if const_expr(fp8):
                y_local = [y_tile(tile) for tile in range_constexpr(tiles_per_thread)]
                thread_amax = zero
                for tile in range_constexpr(tiles_per_thread):
                    thread_amax = thread_amax.maximumf(
                        fmath.absf(y_local[tile]).reduce(ReductionOp.MAX)
                    )
                amax = block_reduce(thread_amax, 1, epilogue_buf, "max")
                # amax * (1 / fp8_max), not a divide, matches the HIP per-token quant.
                row_scale = amax * (1.0 / fp8_max)
                inverse_scale = (row_scale > zero).select(1.0 / row_scale, zero)
                for tile in range_constexpr(tiles_per_thread):
                    store_fp8_vec(y_local[tile] * inverse_scale, y_div, tile)
                if local_tid == 0:
                    store_scale(row_scale)
            else:
                for tile in range_constexpr(tiles_per_thread):
                    store_bf16_vec(y_tile(tile).to(fx.BFloat16), y_div, tile)

        if const_expr(rows_per_wg == 1):
            run_row()
        else:
            if token < num_tokens:
                run_row()

    @flyc.jit
    def launch_attn_res(
        prefix: fx.Tensor,
        add_hidden: fx.Tensor,
        add_hidden2: fx.Tensor,
        block_residual: fx.Tensor,
        score_weight: fx.Tensor,
        output_rms_weight: fx.Tensor,
        y: fx.Tensor,
        y_scale: fx.Tensor,
        prefix_out: fx.Tensor,
        block_out: fx.Tensor,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        attn_res_kernel(
            prefix,
            add_hidden,
            add_hidden2,
            block_residual,
            score_weight,
            output_rms_weight,
            y,
            y_scale,
            prefix_out,
            block_out,
            num_tokens,
        ).launch(
            grid=((num_tokens + rows_per_wg - 1) // rows_per_wg, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_attn_res


def _check_row_layout(tensor: torch.Tensor, name: str) -> None:
    """Validate a vector-copy-compatible layout for a multi-dimensional tensor."""
    if tensor.stride(-1) != 1:
        raise ValueError(
            f"{name} must have unit trailing stride, got strides {tensor.stride()}"
        )

    for dim in range(tensor.ndim - 1):
        stride = tensor.stride(dim)
        if stride <= 0 or stride % _VEC_WIDTH:
            raise ValueError(
                f"{name}.stride({dim})={stride} must be a positive multiple of "
                f"{_VEC_WIDTH} so every row start is "
                f"{_VEC_WIDTH * 2}-byte aligned for BufferCopy128b"
            )

    alignment_bytes = _VEC_WIDTH * tensor.element_size()
    if tensor.data_ptr() % alignment_bytes:
        raise ValueError(
            f"{name} base pointer must be {alignment_bytes}-byte aligned for "
            "BufferCopy128b"
        )

    if tensor.numel():
        last_offset = sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
        extent_bytes = (last_offset + 1) * tensor.element_size()
        if extent_bytes >= _BUFFER_LIMIT_BYTES:
            raise ValueError(
                f"{name} spans {extent_bytes} bytes; buffer offsets are 32-bit, "
                "so each tensor must span less than 4 GiB"
            )


def _geometry(tensor: torch.Tensor) -> tuple:
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tensor.stride(),
        tensor.element_size(),
    )


def _byte_extent(
    ptr: int, shape: tuple, strides: tuple, itemsize: int
) -> tuple[int, int]:
    if 0 in shape:
        return 0, 0
    last = sum((size - 1) * stride for size, stride in zip(shape, strides))
    return ptr, ptr + (last + 1) * itemsize


def _rows_self_disjoint(shape: tuple, strides: tuple) -> bool:
    """Sufficient, not necessary: some non-overlapping layouts return False."""
    span = 1
    for stride, size in sorted((st, sz) for sz, st in zip(shape, strides) if sz > 1):
        if stride < span:
            return False
        span = stride * (size - 1) + span
    return True


@lru_cache(maxsize=512)
def _block_out_aliases(out: tuple, blocks: tuple, inputs: tuple) -> bool:
    ptr, shape, strides, itemsize = out
    num_tokens, rows, hidden_size = shape
    num_blocks = rows - 1
    if not _rows_self_disjoint(shape, strides):
        raise ValueError(f"block_out rows overlap each other, got strides {strides}")
    block_shape = (num_tokens, num_blocks, hidden_size)
    alias = (
        num_blocks > 0
        and ptr == blocks[0]
        and blocks[1] == block_shape
        and all(
            out_stride == block_stride
            for size, out_stride, block_stride in zip(block_shape, strides, blocks[2])
            if size > 1
        )
    )
    if alias:
        written = _byte_extent(
            ptr + num_blocks * strides[1] * itemsize,
            (num_tokens, 1, hidden_size),
            strides,
            itemsize,
        )
    else:
        written = _byte_extent(ptr, shape, strides, itemsize)
    for name, geometry in inputs:
        if alias and name == "block_residual":
            continue
        lo, hi = _byte_extent(*geometry)
        if lo < written[1] and written[0] < hi:
            raise ValueError(
                f"block_out overlaps {name}; only block_out[..., :B, :] may be "
                "block_residual itself"
            )
    return alias


@dataclass(frozen=True)
class _LaunchPlan:
    launcher: object | None
    device: torch.device
    num_tokens: int
    rows_shape: tuple
    blocks_shape: tuple
    weight_shape: tuple
    block_out_shape: tuple | None
    block_out_rows_shape: tuple | None
    out_shape: tuple
    scale_shape: tuple
    y_dtype: torch.dtype
    quantized: bool

    def launch(
        self,
        prefix,
        block_residual,
        score_weight,
        add_hidden,
        add_hidden2,
        output_rms_weight,
        block_out,
        stream,
    ):
        device = self.device
        rows = _view_as(prefix, self.rows_shape)
        blocks = _view_as(block_residual, self.blocks_shape)
        score_weight = _view_as(score_weight, self.weight_shape)
        y = torch.empty(*self.rows_shape, device=device, dtype=self.y_dtype)
        y_scale = (
            torch.empty(self.num_tokens, 1, device=device, dtype=torch.float32)
            if self.quantized
            else None
        )
        prefix_out = (
            rows
            if add_hidden is None
            else torch.empty(*self.rows_shape, device=device, dtype=rows.dtype)
        )
        block_out_rows = None
        if self.block_out_shape is not None:
            if block_out is None:
                block_out = torch.empty(
                    *self.block_out_shape, device=device, dtype=rows.dtype
                )
            block_out_rows = _view_as(block_out, self.block_out_rows_shape)

        if self.launcher is not None:
            if stream is None:
                launch_stream = torch._C._cuda_getCurrentRawStream(device.index)
            elif stream.device != device:
                raise ValueError(f"stream must be on {device}, got {stream.device}")
            else:
                launch_stream = stream
            args = (
                self.launcher,
                rows,
                rows if add_hidden is None else _view_as(add_hidden, self.rows_shape),
                (
                    rows
                    if add_hidden2 is None
                    else _view_as(add_hidden2, self.rows_shape)
                ),
                blocks if blocks.shape[1] else rows,
                score_weight,
                (
                    rows
                    if output_rms_weight is None
                    else _view_as(output_rms_weight, self.weight_shape)
                ),
                y.view(torch.int8) if self.quantized else y,
                y_scale if self.quantized else rows,
                prefix_out,
                rows if block_out_rows is None else block_out_rows,
                self.num_tokens,
                launch_stream,
            )
            # Entering torch.cuda.device on every call adds host time that shows at
            # small T.
            if torch.cuda.current_device() == device.index:
                _run_compiled(*args)
            else:
                with torch.cuda.device(device):
                    _run_compiled(*args)
            if stream is not None:
                for output in (y, y_scale, prefix_out, block_out_rows):
                    if output is not None and output is not rows:
                        output.record_stream(stream)

        y_result = _view_as(y, self.out_shape)
        if self.quantized:
            y_result = (y_result, _view_as(y_scale, self.scale_shape))
        prefix_result = (
            prefix if add_hidden is None else _view_as(prefix_out, self.out_shape)
        )
        if block_out is None:
            return y_result, prefix_result
        return y_result, prefix_result, block_out


def _view_as(tensor: torch.Tensor, shape: tuple) -> torch.Tensor:
    return tensor if tensor.shape == shape else tensor.reshape(shape)


def _key_geometry(tensor: torch.Tensor | None, pointer_mask: int):
    """Must cover every tensor property ``_plan_launch`` validates, or a cached plan
    skips a check.
    """
    if tensor is None:
        return None
    return (
        tensor.shape,
        tensor.stride(),
        tensor.dtype,
        tensor.device,
        tensor.data_ptr() & pointer_mask,
    )


def _plan_launch(
    prefix,
    block_residual,
    score_weight,
    eps,
    add_hidden,
    add_hidden2,
    output_rms_weight,
    output_rms_eps,
    scale,
    close_block,
    out_quant_dtype,
    rows_per_wg,
    block_out,
) -> _LaunchPlan:
    if add_hidden2 is not None and add_hidden is None:
        raise ValueError("add_hidden2 requires add_hidden")
    if out_quant_dtype is not None and output_rms_weight is None:
        raise ValueError("out_quant_dtype requires output_rms_weight")
    if out_quant_dtype is not None and out_quant_dtype != dtypes.fp8:
        raise ValueError(
            f"out_quant_dtype must be the arch-native FP8 type {dtypes.fp8}, "
            f"got {out_quant_dtype}"
        )

    hidden_size = prefix.shape[-1]
    if block_residual.ndim < 2 or block_residual.shape[-1] != hidden_size:
        raise ValueError(
            f"block_residual must have shape [.., B, {hidden_size}], "
            f"got {tuple(block_residual.shape)}"
        )
    num_blocks = block_residual.shape[-2]
    if not 0 <= num_blocks <= _MAX_BLOCKS:
        raise ValueError(f"B must be in [0, {_MAX_BLOCKS}], got {num_blocks}")
    if hidden_size <= 0 or hidden_size % _VEC_WIDTH:
        raise ValueError(
            f"prefix hidden size must be a positive multiple of {_VEC_WIDTH}, "
            f"got {hidden_size}"
        )
    row_threads = _block_threads_for(hidden_size // _VEC_WIDTH)
    rows_per_wg = _resolve_rows_per_wg(row_threads, rows_per_wg)

    rows = prefix.reshape(math.prod(prefix.shape[:-1]), hidden_size)
    blocks = block_residual.reshape(
        math.prod(block_residual.shape[:-2]), num_blocks, hidden_size
    )
    if blocks.shape[0] != rows.shape[0]:
        raise ValueError(
            f"prefix has {rows.shape[0]} rows but block_residual has "
            f"{blocks.shape[0]}; the leading dimensions must match"
        )
    weights = {"score_weight": score_weight.reshape(-1)}
    if output_rms_weight is not None:
        weights["output_rms_weight"] = output_rms_weight.reshape(-1)
    for name, weight in weights.items():
        if weight.shape != (hidden_size,):
            raise ValueError(f"{name} must have {hidden_size} elements")
    adds = {}
    for name, addend in (("add_hidden", add_hidden), ("add_hidden2", add_hidden2)):
        if addend is not None:
            if addend.shape[-1] != hidden_size:
                raise ValueError(f"{name} must have shape [.., {hidden_size}]")
            adds[name] = addend.reshape(-1, hidden_size)
            if adds[name].shape != rows.shape:
                raise ValueError(f"{name} must have {rows.shape[0]} rows")

    tensors = {"prefix": rows, "block_residual": blocks, **weights, **adds}
    block_out_rows = None
    if block_out is not None:
        expected_shape = (*prefix.shape[:-1], num_blocks + 1, hidden_size)
        if block_out.shape != expected_shape:
            raise ValueError(
                f"block_out must have shape {expected_shape}, "
                f"got {tuple(block_out.shape)}"
            )
        block_out_rows = block_out.reshape(rows.shape[0], num_blocks + 1, hidden_size)
        if block_out.numel() and block_out_rows.data_ptr() != block_out.data_ptr():
            raise ValueError(
                "block_out leading dims must be viewable as flat rows, got "
                f"strides {block_out.stride()}"
            )
        tensors["block_out"] = block_out_rows
    if any(t.dtype != torch.bfloat16 for t in tensors.values()):
        raise TypeError("only BF16 inputs are supported")
    if any(t.device != prefix.device for t in tensors.values()):
        raise ValueError("all tensors must be on prefix.device")
    if not prefix.is_cuda:
        raise ValueError("all tensors must be CUDA tensors")

    gfx = get_gfx_runtime()
    if gfx not in _ARCH_TUNING:
        raise ValueError(
            f"flydsl_attn_res requires wave64 CDNA ({', '.join(_ARCH_TUNING)}); "
            f"got {gfx}. This kernel uses _WARP_SIZE={_WARP_SIZE} DPP "
            "reductions and readlane(..., 63)."
        )
    tuning = _ARCH_TUNING[gfx]

    for name, tensor in tensors.items():
        if tensor.numel():
            _check_row_layout(tensor, name)

    block_alias = False
    if block_out is not None:
        tensors.pop("block_out")
        block_alias = _block_out_aliases(
            _geometry(block_out_rows),
            _geometry(blocks),
            tuple((name, _geometry(tensor)) for name, tensor in tensors.items()),
        )

    num_tokens = rows.shape[0]
    leading = tuple(prefix.shape[:-1])
    block_out_shape = None
    block_out_rows_shape = None
    if block_out is not None or close_block:
        block_out_shape = (*leading, num_blocks + 1, hidden_size)
        block_out_rows_shape = (num_tokens, num_blocks + 1, hidden_size)
        if block_out is None:
            _check_row_layout(
                torch.empty(block_out_rows_shape, device="meta"), "block_out"
            )

    launcher = None
    if num_tokens:
        cu_divisor = (
            tuning.alias_wide_grid_cu_divisor
            if block_alias
            else tuning.wide_grid_cu_divisor
        )
        wide_grid = (
            row_threads > _WARP_SIZE
            and num_tokens > _num_cus(prefix.device) // cu_divisor
            and _wide_row_supported(hidden_size)
        )
        gfx950_ops = _has_gfx950_ops()
        launcher = _build_attn_res(
            _AttnResSpec(
                hidden_size=hidden_size,
                num_blocks=num_blocks,
                has_add=add_hidden is not None,
                has_add2=add_hidden2 is not None,
                close_block=block_out_shape is not None,
                output_norm=output_rms_weight is not None,
                eps=float(eps),
                output_rms_eps=(
                    float(output_rms_eps) if output_rms_weight is not None else 0.0
                ),
                scale=float(scale),
                rows_per_wg=rows_per_wg,
                wide_grid=wide_grid,
                fp8_dtype=out_quant_dtype,
                block_alias=block_alias,
                native_swap=gfx950_ops,
                nontemporal_block_loads=tuning.nontemporal_block_loads,
                narrow_prefetch_in_flight=tuning.narrow_prefetch_in_flight,
                bf16_dot=gfx950_ops
                and not (wide_grid and num_tokens > _WIDE_BF16_DOT_MAX_TOKENS),
            )
        )

    return _LaunchPlan(
        launcher=launcher,
        device=prefix.device,
        num_tokens=num_tokens,
        rows_shape=tuple(rows.shape),
        blocks_shape=tuple(blocks.shape),
        weight_shape=(hidden_size,),
        block_out_shape=block_out_shape,
        block_out_rows_shape=block_out_rows_shape,
        out_shape=tuple(prefix.shape),
        scale_shape=(*leading, 1),
        y_dtype=out_quant_dtype or prefix.dtype,
        quantized=out_quant_dtype is not None,
    )


def flydsl_attn_res(
    prefix: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float = 1e-6,
    add_hidden: torch.Tensor | None = None,
    add_hidden2: torch.Tensor | None = None,
    *,
    output_rms_weight: torch.Tensor | None = None,
    output_rms_eps: float = 1e-6,
    scale: float = 1.0,
    close_block: bool = False,
    out_quant_dtype: torch.dtype | None = None,
    rows_per_wg: int | None = None,
    stream: torch.cuda.Stream | None = None,
    block_out: torch.Tensor | None = None,
):
    """Drop-in for Triton ``attn_res_gate`` on BF16 inputs over ``B + 1`` candidates.

    ``prefix`` is ``[.., D]``, ``block_residual`` is ``[.., B, D]`` with ``B`` in
    0..8 (a strided view such as ``cap[:, :B]`` is fine), and ``score_weight``
    is the pre-folded ``[D]`` scoring vector. ``logit = scale * rstd *
    dot(v, score_weight)`` with ``rstd`` from ``eps``, softmaxed over the
    candidates; the raw values are mixed. ``add_hidden`` [and ``add_hidden2``]
    fold into the prefix candidate as an unrounded FP32 sum. ``output_rms_weight``
    applies an output RMSNorm with ``output_rms_eps``.

    Returns ``(y, prefix_out)``, or ``(y, prefix_out, block_out)`` with
    ``close_block`` or a caller-owned ``block_out``. ``prefix_out`` is a new BF16
    tensor holding the rounded sum when ``add_hidden`` is given, otherwise
    ``prefix`` itself. ``block_out`` is
    ``cat([block_residual, prefix_out[..., None, :]], -2)``. Inputs are never
    mutated.

    ``block_out`` is an optional caller-owned BF16 ``[.., B + 1, D]`` output and
    implies ``close_block``. When ``block_out[..., :B, :]`` is ``block_residual``
    itself (same shape, data pointer, and strides) the ``B`` copies are skipped and
    only row ``B`` is written. Otherwise the byte extent of ``block_out`` must not
    meet the extent of any input, so views that interleave in one buffer raise
    ``ValueError``.

    Every input must be a BF16 CUDA tensor on ``prefix.device`` with unit
    trailing stride, positive leading strides that are multiples of
    8, a 16-byte-aligned base pointer, and an extent under 4 GiB.
    ``stream`` must be on ``prefix.device``. Requires gfx942 or gfx950 (wave64).
    ``D`` must be a positive multiple of 8 with a wave-aligned row
    size. ``rows_per_wg`` packs one-wave rows (D=1024) per workgroup; multi-wave
    rows reject any value other than 1.

    With ``block_out``, cached plans key on exact data pointers, so each new
    buffer address revalidates.
    """
    # Without block_out only the base alignment matters. With it, overlap and
    # the alias flag depend on exact addresses, so the key holds them.
    pointer_mask = _ALIGNMENT_MASK if block_out is None else -1
    key = (
        get_gfx_runtime(),
        _key_geometry(prefix, pointer_mask),
        _key_geometry(block_residual, pointer_mask),
        _key_geometry(score_weight, pointer_mask),
        _key_geometry(add_hidden, pointer_mask),
        _key_geometry(add_hidden2, pointer_mask),
        _key_geometry(output_rms_weight, pointer_mask),
        _key_geometry(block_out, pointer_mask),
        eps,
        output_rms_eps,
        scale,
        close_block,
        out_quant_dtype,
        rows_per_wg,
    )
    plan = _LAUNCH_PLANS.get(key)
    if plan is None:
        plan = _plan_launch(
            prefix,
            block_residual,
            score_weight,
            eps,
            add_hidden,
            add_hidden2,
            output_rms_weight,
            output_rms_eps,
            scale,
            close_block,
            out_quant_dtype,
            rows_per_wg,
            block_out,
        )
        if len(_LAUNCH_PLANS) >= _MAX_LAUNCH_PLANS:
            _LAUNCH_PLANS.clear()
        _LAUNCH_PLANS[key] = plan
    return plan.launch(
        prefix,
        block_residual,
        score_weight,
        add_hidden,
        add_hidden2,
        output_rms_weight,
        block_out,
        stream,
    )
