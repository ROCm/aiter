# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL AttnRes prefill kernel for Kimi-K3.

Each workgroup's wave-aligned size is derived from the hidden size. D=7168
keeps one 448-thread row per workgroup. D=1024 defaults to one 64-thread row
per workgroup. ``flydsl_attn_res(..., rows_per_wg=4)`` packs four independent
one-wave rows into a 256-thread workgroup; idle tail waves skip when ``T`` is
not a multiple of four. Multi-wave rows always stay at one row per workgroup.
Inputs may be contiguous or have padded leading dimensions when their trailing
stride is one and their row starts are 16-byte aligned. The kernel reads up to
eight block sources plus the live prefix, scores sources with RMS-normalized
keys, and mixes raw values with an online softmax accumulator.

Delta update, canonical append snapshot, and output RMSNorm are independent
compile-time specializations. ``block_write_idx=-1`` disables the snapshot;
otherwise it must equal ``num_blocks`` in 0..7, so the current prefix is stored
in the first unused block slot.
"""

# Do not add ``from __future__ import annotations``: FlyDSL inspects annotations
# at trace time and PEP 563 can interfere with its runtime argument detection.

import math
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, Stream, T

from aiter.jit.utils.chip_info import get_gfx

from . import dpp_utils
from .tensor_shim import _run_compiled

KERNEL_NAME = "flydsl_attn_res"

_MAX_BLOCKS = 8
_WARP_SIZE = 64
_VEC_WIDTH = 8  # 8 bf16 values = one 128-bit global-memory transaction.
_TILE_CANDIDATES = (2, 4, 8, 1)
_MAX_BLOCK_THREADS = 1024
_ALLOWED_GFX = ("gfx942", "gfx950")
_SMALL_D_ROWS_PER_WG = 1
# Extra block sources kept outstanding during consume_source. Value 3 means
# sources i+1, i+2, i+3 are already issued when reducing source i. Prefix is
# already resident and is not prefetched. Changing this constant requires
# _build_attn_res.cache_clear().
_SOURCE_PREFETCH_IN_FLIGHT = 3
# Changing this constant requires _build_attn_res.cache_clear().
_SOURCE_GROUP_SIZE = 2
# "auto" uses VALU DPP for one-wave rows and shuffle_xor for multi-wave
# rows. "swizzle" / "dpp" force one path for A/B. Changing this constant
# requires _build_attn_res.cache_clear().
_WAVE_REDUCE_MODE = "auto"
_LOG2E = math.log2(math.e)
_DPP_QUAD_XOR1 = 0xB1
_DPP_QUAD_XOR2 = 0x4E
_DPP_ROW_SHR1 = 0x111
_DPP_ROW_SHR2 = 0x112
_DPP_ROW_SHL4 = 0x104
_DPP_ROW_SHR4 = 0x114
_DPP_ROW_SHL8 = 0x108
_DPP_ROW_SHR8 = 0x118
_DPP_ROW_BCAST15 = 0x142
_DPP_ROW_BCAST31 = 0x143
_DPP_ROW_MASK = 0xF
# row_bcast:15 moves row 0 -> 1 and row 2 -> 3.
_DPP_ROW_BCAST15_MASK = 0xA
# row_bcast:31 moves the row-0/1 total into rows 2 and 3.
_DPP_ROW_BCAST31_MASK = 0xC
# Banks 0+2 read n+4; banks 1+3 read n-4.
_DPP_BANK_XOR4_SHL = 0x5
_DPP_BANK_XOR4_SHR = 0xA
# Banks 0+1 read n+8; banks 2+3 read n-8.
_DPP_BANK_XOR8_SHL = 0x3
_DPP_BANK_XOR8_SHR = 0xC


def _dpp_xor_f32(value, offset: int):
    """Return the f32 held by ``lane ^ offset`` via VALU DPP. Offsets 1, 2, 4, 8 only."""
    bits = value.bitcast(fx.Int32)
    if offset == 1:
        peer = dpp_utils.update_dpp_i32(
            bits, bits, _DPP_QUAD_XOR1, _DPP_ROW_MASK, _DPP_ROW_MASK, True
        )
    elif offset == 2:
        peer = dpp_utils.update_dpp_i32(
            bits, bits, _DPP_QUAD_XOR2, _DPP_ROW_MASK, _DPP_ROW_MASK, True
        )
    elif offset == 4:
        shifted = dpp_utils.update_dpp_i32(
            bits, bits, _DPP_ROW_SHL4, _DPP_ROW_MASK, _DPP_BANK_XOR4_SHL, True
        )
        peer = dpp_utils.update_dpp_i32(
            shifted, bits, _DPP_ROW_SHR4, _DPP_ROW_MASK, _DPP_BANK_XOR4_SHR, True
        )
    elif offset == 8:
        shifted = dpp_utils.update_dpp_i32(
            bits, bits, _DPP_ROW_SHL8, _DPP_ROW_MASK, _DPP_BANK_XOR8_SHL, True
        )
        peer = dpp_utils.update_dpp_i32(
            shifted, bits, _DPP_ROW_SHR8, _DPP_ROW_MASK, _DPP_BANK_XOR8_SHR, True
        )
    else:
        raise ValueError(
            "DPP XOR is row-local; offset must be 1, 2, 4, or 8, " f"got {offset}"
        )
    return fx.Int32(peer).bitcast(fx.Float32)


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


def _wave_reduce_to_one_lane_f32(value, fastmath):
    """Wave64 sum for consumers where a single lane reads the total.

    Callers that only store from one lane do not need the broadcast,
    so this walks a ``row_shr`` cascade into ``row_bcast`` and reads
    lane 63 into an SGPR, which every lane can then use.
    """
    reduced = value
    for dpp_ctrl in (
        _DPP_ROW_SHR1,
        _DPP_ROW_SHR2,
        _DPP_ROW_SHR4,
        _DPP_ROW_SHR8,
    ):
        reduced = reduced.addf(
            _dpp_row_f32(reduced, dpp_ctrl),
            fastmath=fastmath,
        )
    reduced = reduced.addf(
        _dpp_row_f32(reduced, _DPP_ROW_BCAST15, _DPP_ROW_BCAST15_MASK),
        fastmath=fastmath,
    )
    reduced = reduced.addf(
        _dpp_row_f32(reduced, _DPP_ROW_BCAST31, _DPP_ROW_BCAST31_MASK),
        fastmath=fastmath,
    )
    bits = reduced.bitcast(fx.Int32)
    return fx.Int32(fx.rocdl.readlane(T.i32, bits, 63)).bitcast(fx.Float32)


def _wave_reduce_add_f32(value, mode: str, fastmath):
    """Wave64 sum. ``dpp`` uses VALU DPP for offsets <= 8; ``swizzle`` uses shuffle_xor."""
    reduced = value
    for shift_exp in range_constexpr(int(math.log2(_WARP_SIZE))):
        offset = _WARP_SIZE // (2 << shift_exp)
        if mode == "dpp" and offset <= 8:
            peer = _dpp_xor_f32(reduced, offset)
        else:
            peer = reduced.shuffle_xor(offset, _WARP_SIZE)
        reduced = reduced.addf(peer, fastmath=fastmath)
    return reduced


def _narrow_dpp_reduce_add_f32(value, red_slots: int, fastmath):
    """Sum the first ``red_slots`` lanes (rest already zero) and broadcast lane 0.

    Offsets 32/16/8 of a wave64 butterfly add zeros when ``red_slots <= 8``.
    The remaining tree (span/2 .. 1) fits DPP; ``v_readfirstlane`` replaces
    the zero-add copy that used to spread the total across the wave.
    """
    if red_slots <= 1:
        return value
    if red_slots > 8:
        raise ValueError(
            "DPP finish reduce only covers red_slots in 2..8, " f"got {red_slots}"
        )
    span = 1 << math.ceil(math.log2(red_slots))
    reduced = value
    for shift_exp in range_constexpr(int(math.log2(span))):
        offset = span // (2 << shift_exp)
        peer = _dpp_xor_f32(reduced, offset)
        reduced = reduced.addf(peer, fastmath=fastmath)
    bits = reduced.bitcast(fx.Int32)
    broadcast = fx.Int32(fx.rocdl.readfirstlane(T.i32, bits))
    return broadcast.bitcast(fx.Float32)


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
        return _SMALL_D_ROWS_PER_WG if row_threads == _WARP_SIZE else 1
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


@lru_cache(maxsize=256)
def _build_attn_res(
    hidden_size: int,
    num_blocks: int,
    eps: float,
    output_norm_eps: float,
    has_delta: bool,
    block_write_idx: int,
    apply_output_norm: bool,
    rows_per_wg: int,
):
    """Return one fixed-shape, compile-time-specialized AttnRes launcher."""
    if hidden_size <= 0 or hidden_size % _VEC_WIDTH:
        raise ValueError(f"D={hidden_size} must be a positive multiple of {_VEC_WIDTH}")
    if not 0 <= num_blocks <= _MAX_BLOCKS:
        raise ValueError(f"num_blocks must be in [0, {_MAX_BLOCKS}], got {num_blocks}")
    if not -1 <= block_write_idx < _MAX_BLOCKS:
        raise ValueError(
            f"block_write_idx must be -1 or in [0, {_MAX_BLOCKS - 1}], "
            f"got {block_write_idx}"
        )
    if block_write_idx >= 0 and block_write_idx != num_blocks:
        raise ValueError(
            "this build only emits the canonical append slot: "
            f"block_write_idx={block_write_idx}, num_blocks={num_blocks}"
        )

    num_vec = hidden_size // _VEC_WIDTH
    row_threads = _block_threads_for(num_vec)
    rows_per_wg = _resolve_rows_per_wg(row_threads, rows_per_wg)
    block_threads = row_threads * rows_per_wg
    tiles_per_thread = num_vec // row_threads
    red_slots = row_threads // _WARP_SIZE
    num_sources = num_blocks + 1
    write_block = block_write_idx >= 0
    prefetch_in_flight = min(_SOURCE_PREFETCH_IN_FLIGHT, max(num_blocks - 1, 0))
    # One-wave rows have no cross-wave barrier to collapse.
    group_size = _SOURCE_GROUP_SIZE if red_slots > 1 else 1
    num_groups = (num_sources + group_size - 1) // group_size
    if _WAVE_REDUCE_MODE == "auto":
        # One-wave rows have no real LDS traffic, so DPP replaces ds_swizzle.
        # Multi-wave rows still handshake through LDS; DPP lost wall time there.
        wave_reduce_mode = "dpp" if red_slots == 1 else "swizzle"
    elif _WAVE_REDUCE_MODE in ("swizzle", "dpp"):
        wave_reduce_mode = _WAVE_REDUCE_MODE
    else:
        raise ValueError(
            f"_WAVE_REDUCE_MODE must be 'auto', 'swizzle', or 'dpp', "
            f"got {_WAVE_REDUCE_MODE!r}"
        )
    # Multi-wave finish is 3-step DPP + readfirstlane. DPP XOR only reaches
    # offsets 1/2/4/8, so more than 8 waves still uses the wave64 butterfly.
    # Production is 7 (D=7168) or 1 (D=1024, which never enters finish).
    use_dpp_finish = 1 < red_slots <= 8
    kernel_name = (
        f"{KERNEL_NAME}_d{hidden_size}_k{num_sources}_bt{block_threads}"
        f"_r{rows_per_wg}_v{_VEC_WIDTH}_delta{int(has_delta)}_write{int(write_block)}"
        f"_onorm{int(apply_output_norm)}_pf{prefetch_in_flight}_g{group_size}"
        f"_wr{wave_reduce_mode}"
    )

    @fx.struct
    class SharedStorage:
        # Ping-pong (dim 0) so successive groups write disjoint LDS; slot
        # (dim 1) holds G independent reductions behind one barrier. Array is
        # 1D; the view is (2, group_size, red_slots).
        sumsq: fx.Array[fx.Float32, 2 * group_size * red_slots, 16]
        dot: fx.Array[fx.Float32, 2 * group_size * red_slots, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
    def attn_res_kernel(
        prefix: fx.Tensor,
        delta: fx.Tensor,
        blocks: fx.Tensor,
        norm_weight: fx.Tensor,
        qk_weight: fx.Tensor,
        output_norm_weight: fx.Tensor,
        out: fx.Tensor,
        num_tokens: fx.Int32,
    ):
        tid = fx.thread_idx.x
        row_owner = fx.make_layout((rows_per_wg, row_threads), (row_threads, 1))
        row_coord = fx.idx2crd(fx.Int32(tid), row_owner)
        wg_row = fx.Int32(fx.get(row_coord, 0))
        local_tid = fx.Int32(fx.get(row_coord, 1))
        token = fx.block_idx.x * rows_per_wg + wg_row
        fm_fast = arith.FastMathFlags.fast
        vec_layout = fx.make_layout((tiles_per_thread, row_threads), (row_threads, 1))

        def vector_index_for(tile):
            return fx.Int32(fx.get_scalar(fx.crd2idx((tile, local_tid), vec_layout)))

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        pingpong_layout = fx.make_layout(
            (2, group_size, red_slots),
            (group_size * red_slots, red_slots, 1),
        )
        s_sumsq_all = lds.sumsq.view(pingpong_layout)
        s_dot_all = lds.dot.view(pingpong_layout)

        zero = fx.Float32(0.0)
        neg_inf = fx.Float32(float("-inf"))
        inv_hidden_size = 1.0 / float(hidden_size)
        log2e = _LOG2E

        def wave_reduce_add(value):
            return _wave_reduce_add_f32(value, wave_reduce_mode, fm_fast)

        def store_reduce_add(value):
            return _wave_reduce_to_one_lane_f32(value, fm_fast)

        def finish_reduce(value):
            if const_expr(use_dpp_finish):
                return _narrow_dpp_reduce_add_f32(value, red_slots, fm_fast)
            return wave_reduce_add(value)

        def store_pair_partials(sumsq_local, dot_local, slot, buf):
            """Wave-reduce and write LDS partials. No barrier."""
            s_sumsq = fx.slice(s_sumsq_all, (buf, slot, None))
            s_dot = fx.slice(s_dot_all, (buf, slot, None))
            lane = local_tid % _WARP_SIZE
            wave = local_tid // _WARP_SIZE
            sumsq_wave = store_reduce_add(sumsq_local)
            dot_wave = store_reduce_add(dot_local)
            if lane == 0:
                fx.memref_store(sumsq_wave, s_sumsq, wave)
                fx.memref_store(dot_wave, s_dot, wave)

        def finish_pair_reduce(slot, buf):
            """Every wave reduces the LDS partials in registers."""
            s_sumsq = fx.slice(s_sumsq_all, (buf, slot, None))
            s_dot = fx.slice(s_dot_all, (buf, slot, None))
            lane = local_tid % _WARP_SIZE
            in_range = lane < red_slots
            lane_safe = in_range.select(lane, 0)
            sumsq_partial = fx.memref_load(s_sumsq, lane_safe)
            dot_partial = fx.memref_load(s_dot, lane_safe)
            sumsq_partial = in_range.select(sumsq_partial, zero)
            dot_partial = in_range.select(dot_partial, zero)
            return finish_reduce(sumsq_partial), finish_reduce(dot_partial)

        def block_reduce_add2(sumsq_local, dot_local, buf, slot):
            if const_expr(red_slots == 1):
                return wave_reduce_add(sumsq_local), wave_reduce_add(dot_local)
            store_pair_partials(sumsq_local, dot_local, slot, buf)
            fx.gpu.barrier()
            return finish_pair_reduce(slot, buf)

        def block_reduce_add(value, buf, slot):
            if const_expr(red_slots == 1):
                return wave_reduce_add(value)

            # Keep this structurally aligned with the pair reduce. The output
            # RMSNorm epilogue has one payload, so reducing a dummy zero through
            # s_dot would spend shuffles and LDS traffic for no result.
            s_sumsq = fx.slice(s_sumsq_all, (buf, slot, None))
            lane = local_tid % _WARP_SIZE
            wave = local_tid // _WARP_SIZE
            wave_total = store_reduce_add(value)

            if lane == 0:
                fx.memref_store(wave_total, s_sumsq, wave)
            fx.gpu.barrier()

            in_range = lane < red_slots
            lane_safe = in_range.select(lane, 0)
            partial = fx.memref_load(s_sumsq, lane_safe)
            partial = in_range.select(partial, zero)
            return finish_reduce(partial)

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        vector_layout = fx.make_layout(_VEC_WIDTH, 1)

        def issue_bf16_vec(divided_tensor, index):
            register = fx.make_rmem_tensor(_VEC_WIDTH, fx.BFloat16)
            fx.copy(copy_atom, fx.slice(divided_tensor, (None, index)), register)
            return register

        def load_bf16_vec(divided_tensor, index):
            return fx.memref_load_vec(issue_bf16_vec(divided_tensor, index))

        def store_bf16_vec(value, divided_tensor, index):
            register = fx.make_rmem_tensor(_VEC_WIDTH, fx.BFloat16)
            fx.memref_store_vec(value, register)
            fx.copy(copy_atom, register, fx.slice(divided_tensor, (None, index)))

        prefix_buf = fx.rocdl.make_buffer_tensor(prefix)
        blocks_buf = fx.rocdl.make_buffer_tensor(blocks)
        norm_weight_buf = fx.rocdl.make_buffer_tensor(norm_weight)
        qk_weight_buf = fx.rocdl.make_buffer_tensor(qk_weight)
        out_buf = fx.rocdl.make_buffer_tensor(out)

        def run_row():
            prefix_row = fx.slice(prefix_buf, (token, None))
            out_row = fx.slice(out_buf, (token, None))
            prefix_div = fx.logical_divide(prefix_row, vector_layout)
            norm_weight_div = fx.logical_divide(norm_weight_buf, vector_layout)
            qk_weight_div = fx.logical_divide(qk_weight_buf, vector_layout)
            out_div = fx.logical_divide(out_row, vector_layout)
            if const_expr(has_delta):
                delta_buf = fx.rocdl.make_buffer_tensor(delta)
                delta_row = fx.slice(delta_buf, (token, None))
                delta_div = fx.logical_divide(delta_row, vector_layout)

            def issue_block_source(source_idx):
                block_row = fx.slice(blocks_buf, (token, source_idx, None))
                block_div = fx.logical_divide(block_row, vector_layout)
                handles = []
                for tile in range_constexpr(tiles_per_thread):
                    handles.append(issue_bf16_vec(block_div, vector_index_for(tile)))
                return handles

            def load_issued_source(handles):
                source_local = []
                for tile in range_constexpr(tiles_per_thread):
                    source_local.append(fx.memref_load_vec(handles[tile]))
                return source_local

            # Prime `prefetch_in_flight` extra block sources above the
            # prefix/delta/snapshot prologue so their latency overlaps it.
            # Prefix is already resident. `block_write_idx == num_blocks` is a
            # build-time invariant, so these sources are disjoint from the
            # snapshot slot and the kernel never reads an invalid block slot.
            in_flight = []
            for source in range_constexpr(min(prefetch_in_flight, num_blocks)):
                in_flight.append(issue_block_source(source))

            # q = gamma * w is invariant over all depth sources.  Keep q and the
            # live prefix in registers so the final source requires no global load.
            q_local = []
            prefix_local = []
            for tile in range_constexpr(tiles_per_thread):
                vector_index = vector_index_for(tile)
                gamma = load_bf16_vec(norm_weight_div, vector_index).to(fx.Float32)
                weight = load_bf16_vec(qk_weight_div, vector_index).to(fx.Float32)
                q_local.append(gamma * weight)
                prefix_value = load_bf16_vec(prefix_div, vector_index)
                if const_expr(has_delta):
                    delta_value = load_bf16_vec(delta_div, vector_index).to(fx.Float32)
                    prefix_value = (prefix_value.to(fx.Float32) + delta_value).to(
                        fx.BFloat16
                    )
                    store_bf16_vec(prefix_value, prefix_div, vector_index)
                prefix_local.append(prefix_value)

            if const_expr(write_block):
                block_out_row = fx.slice(blocks_buf, (token, block_write_idx, None))
                block_out_div = fx.logical_divide(block_out_row, vector_layout)
                for tile in range_constexpr(tiles_per_thread):
                    vector_index = vector_index_for(tile)
                    store_bf16_vec(prefix_local[tile], block_out_div, vector_index)

            mixed_local = [
                fx.Vector.filled(_VEC_WIDTH, 0.0, fx.Float32)
                for _ in range_constexpr(tiles_per_thread)
            ]
            max_logit = neg_inf
            denominator = zero

            def local_sumsq_dot(values_local):
                thread_sumsq = zero
                thread_dot = zero
                for tile in range_constexpr(tiles_per_thread):
                    value = values_local[tile].to(fx.Float32)
                    thread_sumsq = thread_sumsq + (value * value).reduce(
                        ReductionOp.ADD, fastmath=fm_fast
                    )
                    thread_dot = thread_dot + (value * q_local[tile]).reduce(
                        ReductionOp.ADD, fastmath=fm_fast
                    )
                return thread_sumsq, thread_dot

            def store_source_partials(values_local, slot, buf):
                thread_sumsq, thread_dot = local_sumsq_dot(values_local)
                store_pair_partials(thread_sumsq, thread_dot, slot, buf)

            def finish_source_reduce(slot, buf):
                return finish_pair_reduce(slot, buf)

            def update_softmax(
                values_local, sumsq, dot, old_max, old_denominator, old_mixed
            ):
                reciprocal_rms = fmath.rsqrt(
                    sumsq * inv_hidden_size + eps, fastmath=fm_fast
                )
                logit = dot * reciprocal_rms

                new_max = old_max.maximumf(logit)
                old_scale_active = fmath.exp2(
                    (old_max - new_max) * log2e, fastmath=fm_fast
                )
                old_scale = (old_max == neg_inf).select(zero, old_scale_active)
                new_scale = fmath.exp2((logit - new_max) * log2e, fastmath=fm_fast)
                new_denominator = old_denominator * old_scale + new_scale

                new_mixed = []
                for tile in range_constexpr(tiles_per_thread):
                    value = values_local[tile].to(fx.Float32)
                    new_mixed.append(old_mixed[tile] * old_scale + value * new_scale)
                return new_max, new_denominator, new_mixed

            def consume_source(values_local, old_max, old_denominator, old_mixed, buf):
                """One-wave path: full reduce in registers, then softmax."""
                thread_sumsq, thread_dot = local_sumsq_dot(values_local)
                sumsq, dot = block_reduce_add2(thread_sumsq, thread_dot, buf, 0)
                return update_softmax(
                    values_local, sumsq, dot, old_max, old_denominator, old_mixed
                )

            if const_expr(red_slots == 1 or num_sources == 1):
                for source in range_constexpr(num_blocks):
                    if const_expr(source + prefetch_in_flight < num_blocks):
                        in_flight.append(
                            issue_block_source(source + prefetch_in_flight)
                        )
                    source_local = load_issued_source(in_flight.pop(0))
                    max_logit, denominator, mixed_local = consume_source(
                        source_local, max_logit, denominator, mixed_local, source % 2
                    )
                max_logit, denominator, mixed_local = consume_source(
                    prefix_local, max_logit, denominator, mixed_local, num_blocks % 2
                )
            else:
                # Group G independent reductions behind one barrier. Softmax
                # updates stay in source order. Sources stay in bf16 across the
                # barrier; the reconvert for the mix is lossless.
                for group in range_constexpr(num_groups):
                    start = group * group_size
                    n_members = min(group_size, num_sources - start)
                    buf = group % 2
                    group_local = []
                    for slot in range_constexpr(n_members):
                        src = start + slot
                        if const_expr(src < num_blocks):
                            if const_expr(src + prefetch_in_flight < num_blocks):
                                in_flight.append(
                                    issue_block_source(src + prefetch_in_flight)
                                )
                            source_local = load_issued_source(in_flight.pop(0))
                        else:
                            source_local = prefix_local
                        store_source_partials(source_local, slot, buf)
                        group_local.append(source_local)
                    fx.gpu.barrier()
                    for slot in range_constexpr(n_members):
                        sumsq, dot = finish_source_reduce(slot, buf)
                        max_logit, denominator, mixed_local = update_softmax(
                            group_local[slot],
                            sumsq,
                            dot,
                            max_logit,
                            denominator,
                            mixed_local,
                        )

            if const_expr(apply_output_norm):
                thread_sumsq = zero
                for tile in range_constexpr(tiles_per_thread):
                    thread_sumsq = thread_sumsq + (
                        mixed_local[tile] * mixed_local[tile]
                    ).reduce(ReductionOp.ADD, fastmath=fm_fast)
                # Last group used (num_groups-1)%2; epilogue takes the other half.
                sumsq = block_reduce_add(thread_sumsq, num_groups % 2, 0)
                scale = fmath.rsqrt(
                    sumsq * inv_hidden_size
                    + output_norm_eps * denominator * denominator,
                    fastmath=fm_fast,
                )

                output_norm_weight_buf = fx.rocdl.make_buffer_tensor(output_norm_weight)
                output_norm_weight_div = fx.logical_divide(
                    output_norm_weight_buf, vector_layout
                )
                for tile in range_constexpr(tiles_per_thread):
                    vector_index = vector_index_for(tile)
                    output_gamma = load_bf16_vec(
                        output_norm_weight_div, vector_index
                    ).to(fx.Float32)
                    result = (mixed_local[tile] * scale * output_gamma).to(fx.BFloat16)
                    store_bf16_vec(result, out_div, vector_index)
            else:
                inverse_denominator = 1.0 / denominator
                for tile in range_constexpr(tiles_per_thread):
                    vector_index = vector_index_for(tile)
                    result = (mixed_local[tile] * inverse_denominator).to(fx.BFloat16)
                    store_bf16_vec(result, out_div, vector_index)

        if const_expr(rows_per_wg == 1):
            run_row()
        else:
            if token < num_tokens:
                run_row()

    @flyc.jit
    def launch_attn_res(
        prefix: fx.Tensor,
        delta: fx.Tensor,
        blocks: fx.Tensor,
        norm_weight: fx.Tensor,
        qk_weight: fx.Tensor,
        output_norm_weight: fx.Tensor,
        out: fx.Tensor,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        attn_res_kernel(
            prefix,
            delta,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            out,
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


def flydsl_attn_res(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
    rows_per_wg: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Run a BF16 prefill AttnRes specialization on contiguous or padded rows.

    ``delta`` and ``output_norm_weight`` independently enable the in-place
    prefix update and output RMSNorm. ``block_write_idx=-1`` disables snapshots;
    otherwise it must equal ``num_blocks`` in 0..7 and stores the current prefix
    (post-delta when present) in ``blocks[:, block_write_idx]``. Non-canonical
    in-range write indices raise ``NotImplementedError``. Out-of-range
    ``block_write_idx`` or ``num_blocks`` values raise ``ValueError``.

    Multi-dimensional inputs must have unit trailing stride, positive leading
    strides that are multiples of ``_VEC_WIDTH``, and a 16-byte-aligned base
    pointer for ``BufferCopy128b``. Weight vectors must satisfy the same
    vector-copy layout rules. Requires gfx942 or gfx950 (wave64). ``D`` must
    be a positive multiple of ``_VEC_WIDTH`` with a wave-aligned row size;
    D=7168 uses 448 threads and one row per workgroup, while D=1024 defaults
    to one 64-thread row per workgroup. ``rows_per_wg=4`` packs four
    independent D=1024 rows into a 256-thread workgroup; packing is rejected
    when ``row_threads * rows_per_wg`` exceeds ``_MAX_BLOCK_THREADS``, and
    multi-wave rows reject any value other than 1.
    """
    has_delta = delta is not None
    apply_output_norm = output_norm_weight is not None
    if not 0 <= num_blocks <= _MAX_BLOCKS:
        raise ValueError(f"num_blocks must be in [0, {_MAX_BLOCKS}], got {num_blocks}")
    if not -1 <= block_write_idx < _MAX_BLOCKS:
        raise ValueError(
            f"block_write_idx must be -1 or in [0, {_MAX_BLOCKS - 1}], "
            f"got {block_write_idx}"
        )
    write_block = block_write_idx >= 0
    if write_block and block_write_idx != num_blocks:
        raise NotImplementedError(
            "snapshot writes are limited to the canonical append slot: "
            f"block_write_idx must equal num_blocks, got "
            f"block_write_idx={block_write_idx}, num_blocks={num_blocks}"
        )

    if prefix.ndim != 2:
        raise ValueError(f"prefix must have shape [T, D], got {tuple(prefix.shape)}")
    hidden_size = prefix.shape[1]
    if hidden_size <= 0 or hidden_size % _VEC_WIDTH:
        raise ValueError(
            f"prefix hidden size must be a positive multiple of {_VEC_WIDTH}, "
            f"got {hidden_size}"
        )
    row_threads = _block_threads_for(hidden_size // _VEC_WIDTH)
    rows_per_wg = _resolve_rows_per_wg(row_threads, rows_per_wg)

    if blocks.ndim != 3 or blocks.shape != (
        prefix.shape[0],
        _MAX_BLOCKS,
        hidden_size,
    ):
        raise ValueError(
            "blocks must have shape "
            f"[T, {_MAX_BLOCKS}, {hidden_size}], got {tuple(blocks.shape)}"
        )
    if norm_weight.shape != (hidden_size,) or qk_weight.shape != (hidden_size,):
        raise ValueError(
            f"norm_weight and qk_weight must each have shape ({hidden_size},)"
        )
    if has_delta and delta.shape != prefix.shape:
        raise ValueError(
            f"delta must have shape {tuple(prefix.shape)}, got {tuple(delta.shape)}"
        )
    if apply_output_norm and output_norm_weight.shape != (hidden_size,):
        raise ValueError(
            "output_norm_weight must have shape "
            f"({hidden_size},), got {tuple(output_norm_weight.shape)}"
        )

    tensors = (prefix, blocks, norm_weight, qk_weight)
    if has_delta:
        tensors += (delta,)
    if apply_output_norm:
        tensors += (output_norm_weight,)
    if any(t.device != prefix.device for t in tensors):
        raise ValueError("all tensors must be on prefix.device")
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise TypeError("only BF16 inputs are supported")
    if not all(t.is_cuda for t in tensors):
        raise ValueError("all tensors must be CUDA tensors")

    gfx = get_gfx()
    if gfx not in _ALLOWED_GFX:
        raise ValueError(
            f"flydsl_attn_res requires wave64 CDNA ({', '.join(_ALLOWED_GFX)}); "
            f"got {gfx}. This kernel uses _WARP_SIZE={_WARP_SIZE} DPP "
            "reductions and readlane(..., 63)."
        )

    _check_row_layout(prefix, "prefix")
    _check_row_layout(blocks, "blocks")
    if has_delta:
        _check_row_layout(delta, "delta")
    _check_row_layout(norm_weight, "norm_weight")
    _check_row_layout(qk_weight, "qk_weight")
    if apply_output_norm:
        _check_row_layout(output_norm_weight, "output_norm_weight")

    out = torch.empty(prefix.shape, device=prefix.device, dtype=prefix.dtype)
    if prefix.shape[0] == 0:
        return out

    if stream is None:
        stream = torch.cuda.current_stream(prefix.device)
    delta_arg = delta if has_delta else prefix
    output_norm_weight_arg = output_norm_weight if apply_output_norm else norm_weight
    launcher = _build_attn_res(
        hidden_size,
        num_blocks,
        float(eps),
        float(output_norm_eps) if apply_output_norm else 0.0,
        has_delta,
        block_write_idx,
        apply_output_norm,
        rows_per_wg,
    )
    _run_compiled(
        launcher,
        prefix,
        delta_arg,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight_arg,
        out,
        prefix.shape[0],
        Stream(stream),
    )
    return out
