# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL AttnRes mix kernel for Kimi-K3.

This first implementation deliberately covers only the bandwidth-critical
prefill mix path:

* BF16 contiguous ``prefix`` and ``blocks``;
* D=7168, eight block sources plus the live prefix;
* no delta update, block snapshot write, or output RMSNorm.

Each workgroup owns one token row.  It reads every source once, calculates the
source RMS score, and uses an online softmax accumulator to mix the raw source
values without a second HBM pass.
"""

# Do not add ``from __future__ import annotations``: FlyDSL inspects annotations
# at trace time and PEP 563 can interfere with its runtime argument detection.

import math
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.arith import CmpFPredicate
from flydsl.expr.typing import ReductionOp, Stream

from .tensor_shim import _run_compiled

KERNEL_NAME = "flydsl_attn_res_mix"

_HIDDEN_SIZE = 7168
_NUM_SOURCES = 9
_BLOCK_THREADS = 448
_WARP_SIZE = 64
_VEC_WIDTH = 8  # 8 bf16 values = one 128-bit global-memory transaction.
_LOG2E = math.log2(math.e)


@lru_cache(maxsize=16)
def _build_attn_res_mix(hidden_size: int, eps: float):
    """Return the cached D=7168 / nine-source mix-only launcher."""
    if hidden_size != _HIDDEN_SIZE:
        raise ValueError(
            f"only hidden_size={_HIDDEN_SIZE} is supported, got {hidden_size}"
        )

    num_vec = hidden_size // _VEC_WIDTH
    if hidden_size % _VEC_WIDTH != 0 or num_vec % _BLOCK_THREADS != 0:
        raise ValueError(
            f"D={hidden_size} must divide into {_BLOCK_THREADS}-thread vector tiles"
        )
    tiles_per_thread = num_vec // _BLOCK_THREADS
    red_slots = _BLOCK_THREADS // _WARP_SIZE
    kernel_name = (
        f"{KERNEL_NAME}_d{hidden_size}_k{_NUM_SOURCES}_bt{_BLOCK_THREADS}_v{_VEC_WIDTH}"
    )

    @fx.struct
    class SharedStorage:
        sumsq: fx.Array[fx.Float32, red_slots, 16]
        dot: fx.Array[fx.Float32, red_slots, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[_BLOCK_THREADS, 1, 1])
    def attn_res_mix_kernel(
        prefix: fx.Tensor,
        blocks: fx.Tensor,
        norm_weight: fx.Tensor,
        qk_weight: fx.Tensor,
        out: fx.Tensor,
    ):
        token = fx.block_idx.x
        tid = fx.thread_idx.x
        fm_fast = arith.FastMathFlags.fast

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_sumsq = lds.sumsq.view(fx.make_layout(red_slots, 1))
        s_dot = lds.dot.view(fx.make_layout(red_slots, 1))

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

        def block_reduce_add2(sumsq_local, dot_local):
            # The previous source's callers must all finish reading slot zero
            # before this source's wave leaders reuse the LDS arrays.
            gpu.barrier()

            lane = tid % _WARP_SIZE
            wave = tid // _WARP_SIZE
            sumsq_wave = wave_reduce_add(sumsq_local)
            dot_wave = wave_reduce_add(dot_local)

            if lane == 0:
                fx.memref_store(sumsq_wave, s_sumsq, wave)
                fx.memref_store(dot_wave, s_dot, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < red_slots
                lane_safe = in_range.select(lane, 0)
                sumsq_partial = fx.memref_load(s_sumsq, lane_safe)
                dot_partial = fx.memref_load(s_dot, lane_safe)
                sumsq_partial = in_range.select(sumsq_partial, zero)
                dot_partial = in_range.select(dot_partial, zero)
                sumsq_total = wave_reduce_add(sumsq_partial)
                dot_total = wave_reduce_add(dot_partial)

                if lane == 0:
                    fx.memref_store(sumsq_total, s_sumsq, 0)
                    fx.memref_store(dot_total, s_dot, 0)
            gpu.barrier()

            return fx.memref_load(s_sumsq, 0), fx.memref_load(s_dot, 0)

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 16)
        vector_layout = fx.make_layout(_VEC_WIDTH, 1)

        def load_bf16_vec(divided_tensor, index):
            register = fx.make_rmem_tensor(_VEC_WIDTH, fx.BFloat16)
            fx.copy_atom_call(
                copy_atom, fx.slice(divided_tensor, (None, index)), register
            )
            return fx.memref_load_vec(register)

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

        prefix_row = fx.slice(prefix_buf, (token, None))
        out_row = fx.slice(out_buf, (token, None))
        prefix_div = fx.logical_divide(prefix_row, vector_layout)
        norm_weight_div = fx.logical_divide(norm_weight_buf, vector_layout)
        qk_weight_div = fx.logical_divide(qk_weight_buf, vector_layout)
        out_div = fx.logical_divide(out_row, vector_layout)

        # q = gamma * w is invariant over all depth sources.  Keep q and the
        # live prefix in registers so source eight requires no global load.
        q_local = []
        prefix_local = []
        for tile in range_constexpr(tiles_per_thread):
            vector_index = tid + tile * _BLOCK_THREADS
            gamma = load_bf16_vec(norm_weight_div, vector_index).to(fx.Float32)
            weight = load_bf16_vec(qk_weight_div, vector_index).to(fx.Float32)
            q_local.append(gamma * weight)
            prefix_local.append(load_bf16_vec(prefix_div, vector_index))

        mixed_local = [
            fx.Vector.filled(_VEC_WIDTH, 0.0, fx.Float32)
            for _ in range_constexpr(tiles_per_thread)
        ]
        max_logit = neg_inf
        denominator = zero

        def consume_source(values_local, old_max, old_denominator, old_mixed):
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

            sumsq, dot = block_reduce_add2(thread_sumsq, thread_dot)
            reciprocal_rms = fmath.rsqrt(
                sumsq * inv_hidden_size + eps, fastmath=fm_fast
            )
            logit = dot * reciprocal_rms

            new_max = old_max.maximumf(logit)
            is_first_source = arith.cmpf(CmpFPredicate.OEQ, old_max, neg_inf)
            old_scale_active = fmath.exp2((old_max - new_max) * log2e, fastmath=fm_fast)
            old_scale = arith.select(is_first_source, zero, old_scale_active)
            new_scale = fmath.exp2((logit - new_max) * log2e, fastmath=fm_fast)
            new_denominator = old_denominator * old_scale + new_scale

            new_mixed = []
            for tile in range_constexpr(tiles_per_thread):
                new_mixed.append(
                    old_mixed[tile] * old_scale + values_f32[tile] * new_scale
                )
            return new_max, new_denominator, new_mixed

        # Sources 0..7 are block snapshots. Together with the resident-prefix
        # call below, the constexpr loop produces nine explicit source bodies.
        for source in range_constexpr(_NUM_SOURCES - 1):
            block_row = fx.slice(blocks_buf, (token, source, None))
            block_div = fx.logical_divide(block_row, vector_layout)
            source_local = []
            for tile in range_constexpr(tiles_per_thread):
                vector_index = tid + tile * _BLOCK_THREADS
                source_local.append(load_bf16_vec(block_div, vector_index))
            max_logit, denominator, mixed_local = consume_source(
                source_local, max_logit, denominator, mixed_local
            )

        max_logit, denominator, mixed_local = consume_source(
            prefix_local, max_logit, denominator, mixed_local
        )

        inverse_denominator = 1.0 / denominator
        for tile in range_constexpr(tiles_per_thread):
            vector_index = tid + tile * _BLOCK_THREADS
            result = (mixed_local[tile] * inverse_denominator).to(fx.BFloat16)
            store_bf16_vec(result, out_div, vector_index)

    @flyc.jit
    def launch_attn_res_mix(
        prefix: fx.Tensor,
        blocks: fx.Tensor,
        norm_weight: fx.Tensor,
        qk_weight: fx.Tensor,
        out: fx.Tensor,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        attn_res_mix_kernel(prefix, blocks, norm_weight, qk_weight, out).launch(
            grid=(num_tokens, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_attn_res_mix


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
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Run the mix-only FlyDSL AttnRes specialization.

    The public signature intentionally matches the eventual fully fused
    operator.  Unsupported paths fail explicitly rather than silently dropping
    a required side effect.
    """
    del output_norm_eps  # Reserved for the output-RMSNorm specialization.

    if delta is not None:
        raise NotImplementedError("delta fusion is not implemented yet")
    if block_write_idx >= 0:
        raise NotImplementedError("block snapshot writes are not implemented yet")
    if output_norm_weight is not None:
        raise NotImplementedError("output RMSNorm is not implemented yet")
    if num_blocks != _NUM_SOURCES - 1:
        raise NotImplementedError(
            f"Currently only supports num_blocks={_NUM_SOURCES - 1}, got {num_blocks}"
        )

    if prefix.ndim != 2 or prefix.shape[1] != _HIDDEN_SIZE:
        raise ValueError(
            f"prefix must have shape [T, {_HIDDEN_SIZE}], got {tuple(prefix.shape)}"
        )
    if blocks.ndim != 3 or blocks.shape != (
        prefix.shape[0],
        _NUM_SOURCES - 1,
        _HIDDEN_SIZE,
    ):
        raise ValueError(
            "blocks must have shape "
            f"[T, {_NUM_SOURCES - 1}, {_HIDDEN_SIZE}], got {tuple(blocks.shape)}"
        )
    if norm_weight.shape != (_HIDDEN_SIZE,) or qk_weight.shape != (_HIDDEN_SIZE,):
        raise ValueError(
            f"norm_weight and qk_weight must each have shape ({_HIDDEN_SIZE},)"
        )

    tensors = (prefix, blocks, norm_weight, qk_weight)
    if any(t.device != prefix.device for t in tensors):
        raise ValueError("all tensors must be on prefix.device")
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise TypeError("Currently supports BF16 prefix, blocks, and weights only")
    if not all(t.is_cuda for t in tensors):
        raise ValueError("all tensors must be CUDA tensors")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("Currently supports contiguous tensors only")

    out = torch.empty(prefix.shape, device=prefix.device, dtype=prefix.dtype)
    if prefix.shape[0] == 0:
        return out

    if stream is None:
        stream = torch.cuda.current_stream(prefix.device)
    launcher = _build_attn_res_mix(prefix.shape[1], float(eps))
    _run_compiled(
        launcher,
        prefix,
        blocks,
        norm_weight,
        qk_weight,
        out,
        prefix.shape[0],
        Stream(stream),
    )
    return out
