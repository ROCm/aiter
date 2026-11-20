# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""NCDHW -> kH x kW channel-folded NDHWC input for small-C gfx1250 convs.

Output (N, D, Ho, Wo, CGP_ALIGN), Ho = H + 2 ph - kH + 1 (Wo likewise): channel
(r * kW + s) * C + c of pixel (d, oh, ow) is input (c, d, oh + r - ph,
ow + s - pw), zero outside the input and in channels kH * kW * C and up. It
replaces the transpose, zero pad, channel pad and window copy the host would
otherwise run, in one pass: each thread gathers one output pixel's window
(lanes read consecutive W) and writes its 32 channels as four 16-byte stores.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import gpu, range_constexpr

from .conv3d_gfx950_utils import (
    BF16_BYTES,
    CONV_COMPILE_HINTS,
    OOB_SENTINEL_ELEM,
    _as_stream,
)
from .conv3d_gfx1250_utils import CGP_ALIGN, rmem_vec

FOLD_THREADS = 256


def fold_extent_gfx1250(h, w, kh, kw, ph, pw):
    return h + 2 * ph - kh + 1, w + 2 * pw - kw + 1


def fold_fits_gfx1250(n, c, d, h, w, kh, kw, ph, pw):
    """True when the folded tensor and the input index within the i32 buffer range."""
    ho, wo = fold_extent_gfx1250(h, w, kh, kw, ph, pw)
    return (
        kh * kw * c <= CGP_ALIGN
        and ho >= 1
        and wo >= 1
        and n * d * ho * wo * CGP_ALIGN < OOB_SENTINEL_ELEM
        and n * c * d * h * w < OOB_SENTINEL_ELEM
    )


@functools.lru_cache(maxsize=64)
def compile_fold_hw_gfx1250(n, c, d, h, w, kh, kw, ph, pw):
    assert fold_fits_gfx1250(n, c, d, h, w, kh, kw, ph, pw)
    ho, wo = fold_extent_gfx1250(h, w, kh, kw, ph, pw)
    pixels = n * d * ho * wo
    in_elems = n * c * d * h * w
    out_elems = pixels * CGP_ALIGN
    used = kh * kw * c
    elem_ty = fx.BFloat16

    def _flat(tensor, elems):
        buf = fx.rocdl.make_buffer_tensor(
            tensor, max_size=False, num_records_bytes=elems * BF16_BYTES
        )
        return fx.logical_divide(
            fx.Tensor(fx.make_view(fx.get_iter(buf), fx.make_layout(elems, 1))),
            fx.make_layout(1, 1),
        )

    @flyc.kernel(known_block_size=[FOLD_THREADS, 1, 1])
    def fold_kernel(out: fx.Tensor, inp: fx.Tensor):
        in_div = _flat(inp, in_elems)
        out_div = _flat(out, out_elems)
        ld_atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), elem_ty)
        st_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_ty)

        p = fx.Int32(gpu.block_id("x")) * FOLD_THREADS + fx.Int32(gpu.thread_id("x"))
        live = p < pixels
        ow = p % wo
        t = p // wo
        oh = t % ho
        t = t // ho
        dd = t % d
        nn = t // d

        vals = []
        for r in range_constexpr(kh):
            ih = oh + (r - ph)
            for s in range_constexpr(kw):
                iw = ow + (s - pw)
                # Out-of-range offsets read as zero from the buffer descriptor.
                ok = live & (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)
                for cc in range_constexpr(c):
                    off = ((nn * c + cc) * d + dd) * (h * w) + ih * w + iw
                    src = ok.select(fx.Int32(off), fx.Int32(OOB_SENTINEL_ELEM))
                    reg = fx.make_rmem_tensor(1, elem_ty)
                    fx.copy(ld_atom, fx.slice(in_div, (None, src)), reg)
                    vals.append(fx.Vector(reg.load())[0])
        zero = fx.Vector.filled(1, fx.BFloat16(0.0), elem_ty)[0]
        vals += [zero] * (CGP_ALIGN - used)
        for q in range_constexpr(CGP_ALIGN // 8):
            v = fx.Vector.from_elements(vals[q * 8 : (q + 1) * 8], elem_ty)
            dst = live.select(
                fx.Int32(p * CGP_ALIGN + q * 8), fx.Int32(OOB_SENTINEL_ELEM)
            )
            fx.copy(st_atom, rmem_vec(v, 8, elem_ty), fx.slice(out_div, (None, dst)))

    @flyc.jit
    def launch_fold(
        out: fx.Tensor,
        inp: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        fold_kernel(out, inp).launch(
            grid=((pixels + FOLD_THREADS - 1) // FOLD_THREADS, 1, 1),
            block=(FOLD_THREADS, 1, 1),
            stream=stream,
        )

    def _launch(out, inp, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return launch_fold(out, inp, stream=_as_stream(stream))

    def _compile(out, inp, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return flyc.compile(launch_fold, out, inp, _as_stream(stream))

    _launch.compile = _compile
    _launch.extra_args = ()
    return _launch
