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
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.arith import CmpFPredicate
from flydsl.expr.typing import ReductionOp, Stream

from .tensor_shim import _run_compiled

KERNEL_NAME = "flydsl_attn_res"

_MAX_BLOCKS = 8
_WARP_SIZE = 64
_VEC_WIDTH = 8  # 8 bf16 values = one 128-bit global-memory transaction.
_TILE_CANDIDATES = (2, 4, 8, 1)
_MAX_BLOCK_THREADS = 1024
_SMALL_D_ROWS_PER_WG = 1
# Extra block sources kept outstanding during consume_source. Value 3 means
# sources i+1, i+2, i+3 are already issued when reducing source i. Prefix is
# already resident and is not prefetched. Changing this constant requires
# _build_attn_res.cache_clear().
_SOURCE_PREFETCH_IN_FLIGHT = 3
_LOG2E = math.log2(math.e)


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
    kernel_name = (
        f"{KERNEL_NAME}_d{hidden_size}_k{num_sources}_bt{block_threads}"
        f"_r{rows_per_wg}_v{_VEC_WIDTH}_delta{int(has_delta)}_write{int(write_block)}"
        f"_onorm{int(apply_output_norm)}_pf{prefetch_in_flight}"
    )

    @fx.struct
    class SharedStorage:
        # Two buffers so successive reductions write disjoint LDS slots and
        # do not need a write-after-read barrier. Array is 1D; the view is (2, red_slots).
        sumsq: fx.Array[fx.Float32, 2 * red_slots, 16]
        dot: fx.Array[fx.Float32, 2 * red_slots, 16]

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
        local_tid = tid % row_threads
        token = fx.block_idx.x * rows_per_wg + tid // row_threads
        fm_fast = arith.FastMathFlags.fast

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        pingpong_layout = fx.make_layout((2, red_slots), (red_slots, 1))
        s_sumsq_all = lds.sumsq.view(pingpong_layout)
        s_dot_all = lds.dot.view(pingpong_layout)

        zero = fx.Float32(0.0)
        neg_inf = fx.Float32(float("-inf"))
        inv_hidden_size = 1.0 / float(hidden_size)
        log2e = _LOG2E

        def wave_reduce_add(value):
            reduced = value
            for shift_exp in range_constexpr(int(math.log2(_WARP_SIZE))):
                offset = _WARP_SIZE // (2 << shift_exp)
                peer = reduced.shuffle_xor(offset, _WARP_SIZE)
                reduced = reduced.addf(peer, fastmath=fm_fast)
            return reduced

        def block_reduce_add2(sumsq_local, dot_local, buf):
            if const_expr(red_slots == 1):
                return wave_reduce_add(sumsq_local), wave_reduce_add(dot_local)

            # buf is a compile-time 0/1 ping-pong index. Successive reductions
            # write disjoint LDS slots, so there is no write-after-read barrier.
            s_sumsq = fx.slice(s_sumsq_all, (buf, None))
            s_dot = fx.slice(s_dot_all, (buf, None))

            lane = local_tid % _WARP_SIZE
            wave = local_tid // _WARP_SIZE
            sumsq_wave = wave_reduce_add(sumsq_local)
            dot_wave = wave_reduce_add(dot_local)

            if lane == 0:
                fx.memref_store(sumsq_wave, s_sumsq, wave)
                fx.memref_store(dot_wave, s_dot, wave)
            gpu.barrier()

            # Every wave reduces the seven LDS partials in registers, so there
            # is no slot-zero write-back or trailing broadcast barrier.
            in_range = lane < red_slots
            lane_safe = in_range.select(lane, 0)
            sumsq_partial = fx.memref_load(s_sumsq, lane_safe)
            dot_partial = fx.memref_load(s_dot, lane_safe)
            sumsq_partial = in_range.select(sumsq_partial, zero)
            dot_partial = in_range.select(dot_partial, zero)
            return wave_reduce_add(sumsq_partial), wave_reduce_add(dot_partial)

        def block_reduce_add(value, buf):
            if const_expr(red_slots == 1):
                return wave_reduce_add(value)

            # Keep this structurally aligned with block_reduce_add2. The output
            # RMSNorm epilogue has one payload, so reducing a dummy zero through
            # s_dot would spend shuffles and LDS traffic for no result.
            s_sumsq = fx.slice(s_sumsq_all, (buf, None))

            lane = local_tid % _WARP_SIZE
            wave = local_tid // _WARP_SIZE
            wave_total = wave_reduce_add(value)

            if lane == 0:
                fx.memref_store(wave_total, s_sumsq, wave)
            gpu.barrier()

            in_range = lane < red_slots
            lane_safe = in_range.select(lane, 0)
            partial = fx.memref_load(s_sumsq, lane_safe)
            partial = in_range.select(partial, zero)
            return wave_reduce_add(partial)

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 16)
        vector_layout = fx.make_layout(_VEC_WIDTH, 1)

        def issue_bf16_vec(divided_tensor, index):
            register = fx.make_rmem_tensor(_VEC_WIDTH, fx.BFloat16)
            fx.copy_atom_call(
                copy_atom, fx.slice(divided_tensor, (None, index)), register
            )
            return register

        def load_bf16_vec(divided_tensor, index):
            return fx.memref_load_vec(issue_bf16_vec(divided_tensor, index))

        def store_bf16_vec(value, divided_tensor, index):
            register = fx.make_rmem_tensor(_VEC_WIDTH, fx.BFloat16)
            fx.memref_store_vec(value, register)
            fx.copy_atom_call(
                copy_atom, register, fx.slice(divided_tensor, (None, index))
            )

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

            # q = gamma * w is invariant over all depth sources.  Keep q and the
            # live prefix in registers so the final source requires no global load.
            q_local = []
            prefix_local = []
            for tile in range_constexpr(tiles_per_thread):
                vector_index = local_tid + tile * row_threads
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
                    vector_index = local_tid + tile * row_threads
                    store_bf16_vec(prefix_local[tile], block_out_div, vector_index)

            mixed_local = [
                fx.Vector.filled(_VEC_WIDTH, 0.0, fx.Float32)
                for _ in range_constexpr(tiles_per_thread)
            ]
            max_logit = neg_inf
            denominator = zero

            def consume_source(values_local, old_max, old_denominator, old_mixed, buf):
                """Update online-softmax state while ``values_local`` is resident."""
                thread_sumsq = zero
                thread_dot = zero
                values_f32 = []
                for tile in range_constexpr(tiles_per_thread):
                    value = values_local[tile].to(fx.Float32)
                    values_f32.append(value)
                    thread_sumsq = thread_sumsq + (value * value).reduce(
                        ReductionOp.ADD, fastmath=fm_fast
                    )
                    thread_dot = thread_dot + (value * q_local[tile]).reduce(
                        ReductionOp.ADD, fastmath=fm_fast
                    )

                sumsq, dot = block_reduce_add2(thread_sumsq, thread_dot, buf)
                reciprocal_rms = fmath.rsqrt(
                    sumsq * inv_hidden_size + eps, fastmath=fm_fast
                )
                logit = dot * reciprocal_rms

                new_max = old_max.maximumf(logit)
                is_first_source = arith.cmpf(CmpFPredicate.OEQ, old_max, neg_inf)
                old_scale_active = fmath.exp2(
                    (old_max - new_max) * log2e, fastmath=fm_fast
                )
                old_scale = arith.select(is_first_source, zero, old_scale_active)
                new_scale = fmath.exp2((logit - new_max) * log2e, fastmath=fm_fast)
                new_denominator = old_denominator * old_scale + new_scale

                new_mixed = []
                for tile in range_constexpr(tiles_per_thread):
                    new_mixed.append(
                        old_mixed[tile] * old_scale + values_f32[tile] * new_scale
                    )
                return new_max, new_denominator, new_mixed

            def issue_block_source(source_idx):
                block_row = fx.slice(blocks_buf, (token, source_idx, None))
                block_div = fx.logical_divide(block_row, vector_layout)
                handles = []
                for tile in range_constexpr(tiles_per_thread):
                    handles.append(
                        issue_bf16_vec(block_div, local_tid + tile * row_threads)
                    )
                return handles

            # The loop is specialized by num_blocks, then followed by the resident
            # prefix source so the kernel never reads an invalid block slot.
            # Prime `prefetch_in_flight` extra block sources, then issue the
            # replacement before each consume so that many stay outstanding
            # across the remaining reduction barrier.
            in_flight = []
            for source in range_constexpr(min(prefetch_in_flight, num_blocks)):
                in_flight.append(issue_block_source(source))

            for source in range_constexpr(num_blocks):
                if const_expr(source + prefetch_in_flight < num_blocks):
                    in_flight.append(issue_block_source(source + prefetch_in_flight))
                handles = in_flight.pop(0)
                source_local = []
                for tile in range_constexpr(tiles_per_thread):
                    source_local.append(fx.memref_load_vec(handles[tile]))
                max_logit, denominator, mixed_local = consume_source(
                    source_local, max_logit, denominator, mixed_local, source % 2
                )

            max_logit, denominator, mixed_local = consume_source(
                prefix_local, max_logit, denominator, mixed_local, num_blocks % 2
            )

            if const_expr(apply_output_norm):
                thread_sumsq = zero
                for tile in range_constexpr(tiles_per_thread):
                    thread_sumsq = thread_sumsq + (
                        mixed_local[tile] * mixed_local[tile]
                    ).reduce(ReductionOp.ADD, fastmath=fm_fast)
                sumsq = block_reduce_add(thread_sumsq, (num_blocks + 1) % 2)
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
                    vector_index = local_tid + tile * row_threads
                    output_gamma = load_bf16_vec(
                        output_norm_weight_div, vector_index
                    ).to(fx.Float32)
                    result = (mixed_local[tile] * scale * output_gamma).to(fx.BFloat16)
                    store_bf16_vec(result, out_div, vector_index)
            else:
                inverse_denominator = 1.0 / denominator
                for tile in range_constexpr(tiles_per_thread):
                    vector_index = local_tid + tile * row_threads
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
    pointer for ``BufferCopy128b``. Weight vectors must have ``stride(0) == 1``.
    ``D`` must be a positive multiple of ``_VEC_WIDTH`` with a wave-aligned
    row size; D=7168 uses 448 threads and one row per workgroup, while D=1024
    defaults to one 64-thread row per workgroup. ``rows_per_wg=4`` packs four
    independent D=1024 rows into a 256-thread workgroup; multi-wave rows reject
    any value other than 1.
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

    _check_row_layout(prefix, "prefix")
    _check_row_layout(blocks, "blocks")
    if has_delta:
        _check_row_layout(delta, "delta")
    for tensor, name in (
        (norm_weight, "norm_weight"),
        (qk_weight, "qk_weight"),
    ):
        if tensor.stride(0) != 1:
            raise ValueError(
                f"{name}.stride(0) must be 1, got strides {tensor.stride()}"
            )
    if apply_output_norm and output_norm_weight.stride(0) != 1:
        raise ValueError(
            "output_norm_weight.stride(0) must be 1, got strides "
            f"{output_norm_weight.stride()}"
        )

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
