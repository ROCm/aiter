# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Vectorized depthwise causal prefill convolution (16-bit inputs).

Same contract and per-element arithmetic as ``causal_conv1d_prefill.py``: fp32
taps summed from a zero (or bias) base in tap order, SiLU as
acc / (1 + exp(-acc)), one rounding to the input dtype per output, the conv
state written by the first token run of each sequence after its old window is
read. Outputs and states are bit for bit that kernel's. What differs:

* Each thread owns VEC = 8 adjacent channels, so a token is one 16-byte buffer
  load and one 16-byte store (the base kernel: 2-byte accesses, at most 2
  channels a thread). Needs a unit channel stride, 16-byte aligned token rows
  and a channel count divisible by 8.
* A run of ``tokens`` tokens reads its width - 1 halo rows once, and issues the
  loads of the next ``prefetch`` tokens before computing the current ones.
* For bf16 outputs the SiLU may take a cheaper form (``_silu_fast``): raw
  v_exp_f32 and v_rcp_f32. A lane keeps it only where the result is farther
  than FAST_TIE_MARGIN fp32 ulps from a bf16 rounding tie and |acc| is in
  (FAST_ABS_MIN, FAST_ABS_MAX); otherwise the thread's whole vector takes the
  exact expression. ``create_silu_check_kernel`` counts the inputs where
  bf16(fast-with-fallback) != bf16(exact) over all 2^32 fp32 bit patterns.

With 8 channels a thread the exact SiLU (about 31 VALU instructions an
element) bounds the kernel, hence the fast form.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

VEC = 8
# -log2(e) as fp32 head + tail: the head alone is 1.3e-8 off relative, which
# near |acc| = 64 moves exp(-acc) by 7-14 fp32 ulps.
NEG_LOG2E_HI = -1.4426950216293335
NEG_LOG2E_LO = -1.925963033500011e-08
LN2 = 0.6931471805599453
FAST_ABS_MAX = 64.0
FAST_ABS_MIN = 2.0**-60
FAST_TIE_MARGIN = 8


def _silu_exact(acc):
    """The base kernel's SiLU: acc / (1 + exp(-acc)), IEEE division."""
    return acc / (fx.Vector.filled(VEC, 1.0, fx.Float32) + fx.exp(-acc))


def _lanes(vec, fn):
    return fx.Vector.from_elements(
        [fx.Float32(fn(fx.Float32(vec[i]).ir_value())) for i in range_constexpr(VEC)],
        dtype=fx.Float32,
    )


def _silu_fast(acc, margin=FAST_TIE_MARGIN):
    """(fast SiLU, count of lanes that need the exact one).

    exp(-acc) = exp2(hi + lo): hi = fp32(acc * c_hi), lo = its rounding error
    plus acc * c_lo, folded in as e * (1 + lo * ln2); then v_rcp_f32 and one
    multiply. The caller's kernel body branches on the count: FlyDSL rewrites
    runtime ifs only there.
    """
    c = fx.Vector.filled(VEC, NEG_LOG2E_HI, fx.Float32)
    hi = acc * c
    lo = fx.math.fma(
        acc,
        fx.Vector.filled(VEC, NEG_LOG2E_LO, fx.Float32),
        fx.math.fma(acc, c, -hi),
    )
    e = _lanes(hi, lambda v: rocdl.exp2(T.f32, v))
    e = fx.math.fma(e, lo * fx.Vector.filled(VEC, LN2, fx.Float32), e)
    r = _lanes(
        fx.Vector.filled(VEC, 1.0, fx.Float32) + e, lambda v: rocdl.rcp(T.f32, v)
    )
    y = acc * r
    low = y.bitcast(fx.Int32) & fx.Vector.filled(VEC, 0xFFFF, fx.Int32)
    dist = abs(low - fx.Vector.filled(VEC, 0x8000, fx.Int32))
    mag = abs(acc)
    bad = (
        (dist <= fx.Vector.filled(VEC, margin, fx.Int32))
        | (mag >= fx.Vector.filled(VEC, FAST_ABS_MAX, fx.Float32))
        | (mag <= fx.Vector.filled(VEC, FAST_ABS_MIN, fx.Float32))
    )
    return y, bad.select(1, 0).reduce("add")


def _view(t, base):
    """Buffer-resource iterator at element ``base`` of flat tensor ``t``."""
    v = fx.make_view(fx.get_iter(t) + base, fx.make_layout(1, 1))
    return fx.get_iter(fx.rocdl.make_buffer_tensor(v))


def _ld(it, off, dt):
    view = fx.make_view(it + off, fx.make_layout(VEC, 1))
    frag = fx.make_rmem_tensor(VEC, dt)
    fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy(VEC * dt.width), dt), view, frag)
    return frag.load()


def _st(it, off, vec, dt):
    view = fx.make_view(it + off, fx.make_layout(VEC, 1))
    frag = fx.make_rmem_tensor(VEC, dt)
    frag.store(vec)
    fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy(VEC * dt.width), dt), frag, view)


@functools.lru_cache(maxsize=64)
def create_causal_conv1d_prefill_vec_kernel(
    dim: int,
    width: int,
    ws: tuple,
    ss: tuple,
    has_bias: bool,
    has_cache: bool,
    has_indices: bool,
    has_initial: bool,
    silu: bool,
    pad_slot: int | None,
    block: int,
    tokens: int,
    prefetch: int,
    dtype: torch.dtype,
    fast_silu: bool,
):
    """Launcher over a grid (runs, sequences, channel blocks). x and out are
    [dim, total] with unit channel stride, passed flat with their token strides;
    the weight, state and stride tuples are compiled in."""
    element = {torch.bfloat16: fx.BFloat16, torch.float16: fx.Float16}[dtype]
    groups = tokens // prefetch
    halo = width - 1

    @flyc.jit
    def load_channels(tensor: fx.Tensor, offset: fx.Int64, stride: fx.Int64):
        return fx.Vector.from_elements(
            [tensor[offset + lane * stride] for lane in range_constexpr(VEC)]
        )

    @flyc.kernel(known_block_size=[block, 1, 1])
    def causal_conv1d_prefill_vec_kernel(
        x: fx.Tensor,
        w: fx.Tensor,
        bias: fx.Tensor,
        state: fx.Tensor,
        starts: fx.Tensor,
        indices: fx.Tensor,
        initial: fx.Tensor,
        out: fx.Tensor,
        x_stride: fx.Int32,
        out_stride: fx.Int32,
    ):
        seq = fx.block_idx.y
        chunk = fx.block_idx.x
        c = (fx.block_idx.z * block + fx.thread_idx.x) * VEC
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
            raw_weights = [
                load_channels(w, c * ws[0] + j * ws[1], fx.Int64(ws[0]))
                for j in range_constexpr(width)
            ]
            # Runs after the first read their halo from x: the input
            # descriptor starts width - 1 rows before the run.
            hb = (chunk == 0).select(fx.Int64(0), fx.Int64(halo))
            xs = fx.Int64(x_stride)
            cc = fx.Int64(c)
            x_it = _view(x, (start + fx.Int64(offset) - hb) * xs)
            o_it = _view(out, (start + fx.Int64(offset)) * fx.Int64(out_stride))
            history = []
            for j in range_constexpr(halo):
                hvalue = fx.Vector.filled(VEC, 0.0, fx.Float32)
                if chunk == 0:
                    if const_expr(has_cache):  # noqa: SIM102
                        if use_history != 0:
                            hvalue = load_channels(
                                state,
                                slot * ss[0] + c * ss[1] + j * ss[2],
                                fx.Int64(ss[1]),
                            ).to(fx.Float32)
                else:
                    hvalue = _ld(x_it, fx.Int32(fx.Int64(j) * xs + cc), element).to(
                        fx.Float32
                    )
                history.append(hvalue)
            weights = [raw_weights[j].to(fx.Float32) for j in range_constexpr(width)]
            base = fx.Vector.filled(VEC, 0.0, fx.Float32)
            if const_expr(has_bias):
                base = load_channels(bias, fx.Int64(c), fx.Int64(1)).to(fx.Float32)
            if const_expr(has_cache):  # noqa: SIM102
                if chunk == 0:
                    for j in range_constexpr(halo):
                        tail = length - halo + j
                        value = fx.Vector.filled(VEC, 0.0, fx.Float32)
                        if tail >= 0:
                            # Own descriptor at the tail row: its distance from
                            # the run can exceed a 32-bit buffer offset.
                            t_it = _view(x, (start + tail) * xs)
                            value = _ld(t_it, fx.Int32(cc), element).to(fx.Float32)
                        else:
                            for h in range_constexpr(halo):
                                if tail + halo == h:
                                    value = history[h]
                        for lane in range_constexpr(VEC):
                            state[slot * ss[0] + (c + lane) * ss[1] + j * ss[2]] = (
                                element(value[lane])
                            )

            def issue(group):
                loads = []
                for p in range_constexpr(prefetch):
                    position = fx.Int64(offset + group * prefetch + p)
                    safe = (position < length).select(position, length - 1)
                    row = hb + safe - fx.Int64(offset)
                    loads.append(_ld(x_it, fx.Int32(row * xs + cc), element))
                return loads

            current = issue(0)
            for gi in range_constexpr(groups):
                upcoming = issue(gi + 1) if gi + 1 < groups else None
                for p in range_constexpr(prefetch):
                    t = gi * prefetch + p
                    value = current[p].to(fx.Float32)
                    acc = base
                    for j in range_constexpr(halo):
                        acc = acc + history[j] * weights[j]
                    acc = acc + value * weights[width - 1]
                    if const_expr(silu):
                        if const_expr(fast_silu):
                            y, nbad = _silu_fast(acc)
                            if nbad != 0:
                                y = _silu_exact(acc)
                            acc = y
                        else:
                            acc = _silu_exact(acc)
                    if offset + t < length:
                        _st(
                            o_it,
                            fx.Int32(fx.Int64(t) * fx.Int64(out_stride) + cc),
                            fx.Vector.from_elements(
                                [element(acc[lane]) for lane in range_constexpr(VEC)],
                                dtype=element,
                            ),
                            element,
                        )
                    for j in range_constexpr(halo - 1):
                        history[j] = history[j + 1]
                    history[halo - 1] = value
                current = upcoming

    @flyc.jit
    def launch(
        x: fx.Tensor,
        w: fx.Tensor,
        bias: fx.Tensor,
        state: fx.Tensor,
        starts: fx.Tensor,
        indices: fx.Tensor,
        initial: fx.Tensor,
        out: fx.Tensor,
        batch: fx.Int32,
        max_len: fx.Int32,
        x_stride: fx.Int32,
        out_stride: fx.Int32,
        stream: fx.Stream,
    ):
        causal_conv1d_prefill_vec_kernel(
            x, w, bias, state, starts, indices, initial, out, x_stride, out_stride
        ).launch(
            grid=(
                (max_len + tokens - 1) // tokens,
                batch,
                (dim + block * VEC - 1) // (block * VEC),
            ),
            block=(block, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=8)
def create_silu_check_kernel(margin=FAST_TIE_MARGIN, iters=8, block=256):
    """Launcher of a kernel that, per thread, counts the inputs among ``iters``
    x VEC consecutive fp32 bit patterns from ``base`` where
    bf16(fast-with-fallback) != bf16(exact) bitwise (NaN payloads included)."""

    @flyc.kernel(known_block_size=[block, 1, 1])
    def silu_check_kernel(counts: fx.Tensor, base: fx.Int32):
        tid = fx.block_idx.x * block + fx.thread_idx.x
        mism = fx.Int32(0)
        for it in range_constexpr(iters):
            b0 = base + (tid * iters + it) * VEC
            acc = fx.Vector.from_elements(
                [fx.Int32(b0 + i) for i in range_constexpr(VEC)], dtype=fx.Int32
            ).bitcast(fx.Float32)
            exact = _silu_exact(acc)
            y, nbad = _silu_fast(acc, margin=margin)
            if nbad != 0:
                y = _silu_exact(acc)
            # bf16 per lane, as the conv kernel rounds its outputs.
            ref = fx.Vector.from_elements(
                [fx.BFloat16(exact[i]) for i in range_constexpr(VEC)],
                dtype=fx.BFloat16,
            ).bitcast(fx.Int16)
            got = fx.Vector.from_elements(
                [fx.BFloat16(y[i]) for i in range_constexpr(VEC)], dtype=fx.BFloat16
            ).bitcast(fx.Int16)
            mism = mism + (ref != got).select(1, 0).reduce("add")
        counts[tid] = mism

    @flyc.jit
    def launch(counts: fx.Tensor, base: fx.Int32, nblocks: fx.Int32, stream: fx.Stream):
        silu_check_kernel(counts, base).launch(
            grid=(nblocks, 1, 1), block=(block, 1, 1), stream=stream
        )

    return launch
