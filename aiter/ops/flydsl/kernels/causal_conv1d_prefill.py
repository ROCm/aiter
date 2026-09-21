# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Depthwise causal prefill convolution over a packed (varlen) batch.

Companion to ``../causal_conv1d_flydsl.py``. That module fuses the q/k/v split
into the convolution and returns three tensors; this one implements the plain
``causal_conv1d_fn`` contract -- one output tensor, ``conv_states`` updated in
place -- so it can stand in for a serving stack's existing prefill convolution
without touching the surrounding split/reshape code.

Channels map to lanes rather than going through an implicit GEMM, which suits
depthwise work. Two details carry the correctness:

* **Only chunk zero touches the cache.** Every other chunk takes its whole halo
  from immutable ``x``, so there is no inter-block state race.
* **The old window is snapshotted before any state store**, because a sequence
  shorter than ``width - 1`` must shift the previous history rather than read
  values it has just overwritten.

Tail token addresses are clamped instead of branched, so loads stay inside the
current (non-empty) sequence; those lanes never reach a valid output.
"""

import functools
from typing import Optional, Sequence

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr


@functools.lru_cache(maxsize=128)
def create_causal_conv1d_prefill_kernel(dim: int, width: int, xs: tuple, ws: tuple, ss: tuple,
           os: tuple, has_bias: bool, has_cache: bool, has_indices: bool,
           has_initial: bool, silu: bool, pad_slot: Optional[int],
           block: int, tokens: int, dtype: torch.dtype,
           prefetch: int, lanes: int):
    element = {torch.bfloat16: fx.BFloat16, torch.float16: fx.Float16,
               torch.float32: fx.Float32}[dtype]
    @flyc.jit
    def load_channels(tensor: fx.Tensor, offset: fx.Int64, stride: fx.Int64):
        return fx.Vector.from_elements([tensor[offset + lane * stride] for lane in range_constexpr(lanes)])

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(x: fx.Tensor, w: fx.Tensor, bias: fx.Tensor, state: fx.Tensor,
               starts: fx.Tensor, indices: fx.Tensor, initial: fx.Tensor,
               out: fx.Tensor, x_span: fx.Int64, w_span: fx.Int64,
               state_span: fx.Int64, out_span: fx.Int64):
        # Explicit physical offsets below already include the original strides.
        # Rebase to unit-stride views so Tensor indexing does not apply them twice.
        x = fx.Tensor(fx.make_view(fx.get_iter(x), fx.make_layout(x_span, 1)))
        w = fx.Tensor(fx.make_view(fx.get_iter(w), fx.make_layout(w_span, 1)))
        state = fx.Tensor(fx.make_view(fx.get_iter(state), fx.make_layout(state_span, 1)))
        out = fx.Tensor(fx.make_view(fx.get_iter(out), fx.make_layout(out_span, 1)))
        seq = fx.block_idx.y
        chunk = fx.block_idx.x
        c = (fx.block_idx.z * block + fx.thread_idx.x) * lanes
        start = fx.Int64(starts[seq])
        length = fx.Int64(starts[seq + 1]) - start
        offset = chunk * tokens
        slot = fx.Int64(seq)
        if const_expr(has_indices):
            slot = fx.Int64(indices[seq])
        valid_slot = fx.Int32(1)
        if const_expr(pad_slot is not None):
            valid_slot = fx.Int32(slot != pad_slot)
        if (c < dim) & (offset < length) & (valid_slot != 0):
            use_history = fx.Int32(0)
            if const_expr(has_initial):
                use_history = fx.Int32(initial[seq])
            # Start independent weight loads before consuming the input halo.
            raw_weights = [load_channels(w, c * ws[0] + j * ws[1], fx.Int64(ws[0])) for j in range_constexpr(width)]
            # Only chunk zero reads/writes the cache. Other chunks obtain their
            # entire halo from immutable x, avoiding inter-block state races.
            history = []
            for j in range_constexpr(width - 1):
                hvalue = fx.Vector.filled(lanes, 0.0, fx.Float32)
                if chunk == 0:
                    if const_expr(has_cache):
                        if use_history != 0:
                            hvalue = load_channels(state, slot * ss[0] + c * ss[1] + j * ss[2], fx.Int64(ss[1])).to(fx.Float32)
                else:
                    hvalue = load_channels(x, c * xs[0] + (start + offset - (width - 1) + j) * xs[1], fx.Int64(xs[0])).to(fx.Float32)
                history.append(hvalue)
            weights = [raw_weights[j].to(fx.Float32) for j in range_constexpr(width)]
            base = fx.Vector.filled(lanes, 0.0, fx.Float32)
            if const_expr(has_bias):
                base = load_channels(bias, fx.Int64(c), fx.Int64(1)).to(fx.Float32)
            # Snapshot the whole old window before any state stores (short
            # sequences must shift old history, not read overwritten values).
            if const_expr(has_cache):
                if chunk == 0:
                    for j in range_constexpr(width - 1):
                        tail = length - (width - 1) + j
                        value = fx.Vector.filled(lanes, 0.0, fx.Float32)
                        if tail >= 0:
                            value = load_channels(x, c * xs[0] + (start + tail) * xs[1], fx.Int64(xs[0])).to(fx.Float32)
                        else:
                            for h in range_constexpr(width - 1):
                                if tail + width - 1 == h:
                                    value = history[h]
                        for lane in range_constexpr(lanes):
                            state[slot * ss[0] + (c + lane) * ss[1] + j * ss[2]] = element(value[lane])
            # Issue a group of raw loads before conversions/arithmetic consume it.
            # Clamped tail addresses avoid load-side control flow and stay inside
            # this nonempty sequence. Tail values never reach a valid output.
            for t in range_constexpr(tokens):
                if const_expr(t % prefetch == 0):
                    prefetched = []
                    for p in range_constexpr(min(prefetch, tokens - t)):
                        position = offset + t + p
                        safe_position = (position < length).select(position, length - 1)
                        raw = load_channels(x, c * xs[0] + (start + safe_position) * xs[1], fx.Int64(xs[0]))
                        prefetched.append(raw)
                value = prefetched[t % prefetch].to(fx.Float32)
                acc = base
                for j in range_constexpr(width - 1):
                    acc = acc + history[j] * weights[j]
                acc = acc + value * weights[width - 1]
                if const_expr(silu):
                    acc = acc / (fx.Vector.filled(lanes, 1.0, fx.Float32) + fx.exp(-acc))
                if offset + t < length:
                    for lane in range_constexpr(lanes):
                        out[(c + lane) * os[0] + (start + offset + t) * os[1]] = element(acc[lane])
                for j in range_constexpr(width - 2):
                    history[j] = history[j + 1]
                history[width - 2] = value

    @flyc.jit
    def launch(x: fx.Tensor, w: fx.Tensor, bias: fx.Tensor, state: fx.Tensor,
               starts: fx.Tensor, indices: fx.Tensor, initial: fx.Tensor,
               out: fx.Tensor, batch: fx.Int64, max_len: fx.Int64,
               x_span: fx.Int64, w_span: fx.Int64, state_span: fx.Int64,
               out_span: fx.Int64, stream: fx.Stream = fx.Stream(None)):
        kernel(x, w, bias, state, starts, indices, initial, out,
               x_span, w_span, state_span, out_span).launch(
            grid=((max_len + tokens - 1) // tokens, batch, (dim + block * lanes - 1) // (block * lanes)),
            block=(block, 1, 1), stream=stream,
        )
    return launch
