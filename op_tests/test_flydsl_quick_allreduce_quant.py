# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The quick-allreduce quantizer's rounding (``_quant_atom_fp16``) against torch.

Each thread scales 8 fp16 by the group's encode factor, clamps to
``[-bias, bias - 1]``, rounds half to even and adds ``bias``, packing the codes
into the codec's nibble (and, for INT5/INT6, extra-bit) plane. The reference is
the same arithmetic in torch (``torch.round`` rounds half to even); every
element's code must match exactly.

``all_fp16`` feeds every fp16 bit pattern (ties, both clamps, ±inf, NaN,
subnormals), ``randn`` a normal fill spread over the code range.

Candidates: ``flydsl_magic`` is the codec's quantizer (the fp16 magic number
``w + (1024 + bias)``); ``flydsl_roundeven`` is the ``roundeven`` + ``cvt_i16``
form it replaced, kept here to time against.
"""

import argparse
import functools
import itertools

import flydsl.compiler as flyc
import flydsl.expr as fx
import pandas as pd
import torch
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Float32, Int64, Stream, T

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.comm.quick_allreduce_codec import (
    CODECS,
    _clamp_fp16_overflow,
    _f16x2,
    _i32,
    _pack_fields,
    _quant_atom_fp16,
    _splat_f16x2,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
BLOCK = 256
ELEMS_PER_THREAD = 8
ALL_FP16 = 1 << 16


def _quant_atom_fp16_roundeven(codec, atom, enc_pk):
    """The rounding ``_quant_atom_fp16`` replaced: ``roundeven`` + ``cvt_i16`` + bias."""
    lo = _splat_f16x2(fx.Float16(codec.qmin))
    hi = _splat_f16x2(fx.Float16(codec.qmax))
    bias = fx.Vector.filled(2, fx.Int16(codec.bias), fx.Int16)
    q = []
    for i in range_constexpr(4):
        w = fx.min(fx.maxnumf(_f16x2(atom[i]) * enc_pk, lo), hi)
        q.append(_i32(fx.roundeven(w).to(fx.Int16) + bias))
    if codec.n_words_per_thread == 1:
        return (_pack_fields(q, 4),)
    hi_mask = (1 << codec.hi_bits) - 1
    m_hi = fx.Int32(hi_mask | (hi_mask << 16))
    lo4 = [qi & fx.Int32(0x000F000F) for qi in q]
    his = [qi.shrui(fx.Int32(4)) & m_hi for qi in q]
    return (_pack_fields(lo4, 4), _pack_fields(his, codec.hi_bits))


QUANTIZERS = {
    "flydsl_magic": _quant_atom_fp16,
    "flydsl_roundeven": _quant_atom_fp16_roundeven,
}


@functools.cache
def _launcher(codec_name: str, variant: str):
    codec = CODECS[codec_name]
    quant = QUANTIZERS[variant]
    n_words = codec.n_words_per_thread

    def _ptr(addr):
        ty = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=16
        )
        return fx.inttoptr(ty, addr)

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def quant_kernel(inp_ptr: Int64, out_ptr: Int64, enc: Float32):
        _clamp_fp16_overflow()
        t = fx.Int32(gpu.thread_id("x")) + fx.Int32(gpu.block_id("x")) * fx.Int32(BLOCK)
        atom = fx.Vector(
            fx.ptr_load(
                _ptr(inp_ptr) + t * fx.Int32(4),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        words = quant(codec, atom, _splat_f16x2(enc.to(fx.Float16)))
        for w in range_constexpr(n_words):
            fx.ptr_store(words[w], _ptr(out_ptr) + t * fx.Int32(n_words) + fx.Int32(w))

    @flyc.jit
    def launch(
        inp_ptr: Int64,
        out_ptr: Int64,
        enc: Float32,
        grid_x: fx.Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        quant_kernel(inp_ptr, out_ptr, enc).launch(
            grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    launch.func.__name__ = f"launch_qr_quant_{codec_name}_{variant}"
    return launch


def run_torch(x, enc, codec):
    """Per-element code ``roundeven(clamp(fp16(x * enc))) + bias``.

    The kernel's packed fp16 multiply saturates overflow (FP16_OVFL) where torch
    gives inf, and ``maxnumf`` maps NaN to ``qmin``; the clamp makes both agree.
    """
    w = x * torch.tensor(enc, dtype=torch.float16)
    w = torch.where(torch.isnan(w), torch.full_like(w, codec.qmin), w)
    w = w.float().clamp(codec.qmin, codec.qmax)
    return (torch.round(w) + codec.bias).to(torch.int32)


def unpack_codes(words, codec):
    """Kernel output words -> one code per input element, in input order.

    Atom word ``i`` of a thread holds elements ``2i`` (low fp16 half) and
    ``2i + 1`` (high half); a plane packs field ``i`` at ``i * width``.
    """
    w = words.view(-1, codec.n_words_per_thread).to(torch.int64) & 0xFFFFFFFF
    i = torch.arange(4, dtype=torch.int64)
    halves = torch.tensor([0, 16], dtype=torch.int64)

    def plane(word, width):
        shift = (i[:, None] * width + halves[None, :]).flatten()  # elements 0..7
        return (word[:, None] >> shift[None, :]) & ((1 << width) - 1)

    codes = plane(w[:, 0], 4)
    if codec.n_words_per_thread > 1:
        codes = codes | (plane(w[:, 1], codec.hi_bits) << 4)
    return codes.flatten().to(torch.int32)


def make_input(fill, numel, codec):
    if fill == "all_fp16":
        pats = torch.arange(ALL_FP16, dtype=torch.int32).to(torch.int16)
        return pats.repeat(numel // ALL_FP16).view(torch.float16)
    # Spread over the code range so both clamps and every code are hit.
    return (torch.randn(numel) * codec.bias).to(torch.float16)


@benchmark()
def test_quick_allreduce_quant(codec_name, fill, numel, enc):
    codec = CODECS[codec_name]
    x = make_input(fill, numel, codec)
    ref = run_torch(x, enc, codec)
    n_threads = numel // ELEMS_PER_THREAD
    stream = Stream(torch.cuda.current_stream())

    # 4 fp16 ops per element: scale, two clamps, round.
    flops = 4 * numel
    nbytes = numel * x.element_size() + n_threads * codec.n_words_per_thread * 4

    ret = {"gfx": get_gfx_runtime()}
    for name in QUANTIZERS:
        launch = _launcher(codec_name, name)
        # Sentinel: a word the kernel never writes decodes to the wrong codes.
        out = torch.full((n_threads * codec.n_words_per_thread,), -1, dtype=torch.int32)

        def fn(launch=launch, out=out):
            _run_compiled(
                launch,
                Int64(x.data_ptr()),
                Int64(out.data_ptr()),
                Float32(enc),
                fx.Int32(n_threads // BLOCK),
                stream,
            )
            return out

        out, us = run_perftest(fn)
        got = unpack_codes(out, codec)
        err = checkAllclose(
            ref.float(), got.float(), rtol=0, atol=0, msg=f"{name}: {codec_name} codes"
        )
        assert err == 0, f"{name}: {err:.2%} of {codec_name} codes differ from torch"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "quick-allreduce quantizer unsupported on %s; skipping", get_gfx_runtime()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-c",
        "--codec",
        nargs="*",
        default=["int4", "int5", "int6"],
        choices=["int4", "int5", "int6"],
        help="Wire codecs.\n    e.g.: -c int4",
    )
    parser.add_argument(
        "-f",
        "--fill",
        nargs="*",
        default=["all_fp16", "randn"],
        choices=["all_fp16", "randn"],
        help="all_fp16: every fp16 bit pattern, tiled. randn: normal fill.",
    )
    parser.add_argument(
        "-n",
        "--numel",
        type=int,
        nargs="*",
        default=[ALL_FP16, 1 << 24],
        help=f"fp16 elements, a multiple of {ALL_FP16}.\n    e.g.: -n 65536",
    )
    parser.add_argument(
        "-e",
        "--enc",
        type=float,
        nargs="*",
        default=[1.0, 0.37],
        help="Encode factor the group scale applies before rounding.",
    )
    args = parser.parse_args()
    for n in args.numel:
        if n % ALL_FP16:
            parser.error(f"--numel must be a multiple of {ALL_FP16}, got {n}")

    df = [
        test_quick_allreduce_quant(codec, fill, numel, enc)
        for codec, fill, numel, enc in itertools.product(
            args.codec, args.fill, args.numel, args.enc
        )
    ]
    df = pd.DataFrame(df)
    aiter.logger.info(
        "quick-allreduce quantizer summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
