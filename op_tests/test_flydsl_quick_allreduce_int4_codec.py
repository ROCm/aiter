# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exhaustive check of the QuickAllReduceInt4 codec shortcuts (one GPU).

The quantizer replaces two exact operations with cheaper ones; this test
runs both forms on the GPU over their whole input domain, under the
all-reduce's FP16_OVFL mode, and requires identical bits:

* encode scale: ``v_rcp_f32`` vs the IEEE divide of ``1 / (d + 1e-7)``,
  rounded to fp16, for all 256 E4M3 scale bytes;
* nibble rounding: the ``w + 1032`` fp16 magic number vs ``roundeven`` +
  ``cvt_i16`` + bias, for all 65536 fp16 inputs (the clamp to [-8, 7] runs
  first in both, so every value the rounding can see is covered).

The results depend on the GPU's ``v_rcp_f32``, so run this on every arch
QuickAllReduceInt4 supports.
"""

import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64, Stream, T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels import quick_allreduce_int4 as qr
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled


def _ieee_encode_scale(e):
    d = qr._e4m3_to_f32(e) * fx.Float32(-0.125)
    return qr._splat_f16x2(fx.Float32(1.0) / (d + fx.Float32(1e-7)))


def _roundeven_quant(atom, enc_pk):
    q = []
    lo = qr._splat_f16x2(fx.Float16(-8.0))
    hi = qr._splat_f16x2(fx.Float16(7.0))
    bias = fx.full(2, 8, fx.Int16)
    for i in range_constexpr(4):
        w = fx.min(fx.maxnumf(qr._f16x2(atom[i]) * enc_pk, lo), hi)
        q.append(qr._i32(fx.roundeven(w).to(fx.Int16) + bias))
    return q[0] | (q[1] << fx.Int32(4)) | (q[2] << fx.Int32(8)) | (q[3] << fx.Int32(12))


def _store(val, ptr, idx):
    buffer_ops.buffer_store(val, buffer_ops.create_buffer_resource_from_addr(ptr), idx)


@flyc.kernel(known_block_size=[256, 1, 1])
def _scale_kernel(fast: Int64, exact: Int64):
    qr._clamp_fp16_overflow()
    e = fx.Int32(gpu.thread_id("x"))
    _store(qr._i32(qr._encode_scale(e)), fast, e)
    _store(qr._i32(_ieee_encode_scale(e)), exact, e)


@flyc.kernel(known_block_size=[256, 1, 1])
def _nibble_kernel(inp: Int64, fast: Int64, exact: Int64):
    qr._clamp_fp16_overflow()
    tid = fx.Int32(gpu.thread_id("x")) + fx.Int32(gpu.block_id("x")) * fx.Int32(256)
    rsrc = buffer_ops.create_buffer_resource_from_addr(inp)
    atom = fx.Vector(
        buffer_ops.buffer_load(rsrc, tid * fx.Int32(4), vec_width=4, dtype=T.i32)
    )
    one = qr._splat_f16x2(fx.Float16(1.0))
    _store(qr._quant_atom_fp16(atom, one), fast, tid)
    _store(_roundeven_quant(atom, one), exact, tid)


@flyc.jit
def _launch_scale(
    fast: Int64,
    exact: Int64,
    stream: Stream = Stream(None),  # noqa: B008
):
    _scale_kernel(fast, exact).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)


@flyc.jit
def _launch_nibble(
    inp: Int64,
    fast: Int64,
    exact: Int64,
    stream: Stream = Stream(None),  # noqa: B008
):
    _nibble_kernel(inp, fast, exact).launch(
        grid=(8, 1, 1), block=(256, 1, 1), stream=stream
    )


def test_quick_allreduce_int4_codec():
    dev = torch.device("cuda", torch.cuda.current_device())
    stream = Stream(torch.cuda.current_stream())

    fast = torch.zeros(256, dtype=torch.int32, device=dev)
    exact = torch.zeros_like(fast)
    _run_compiled(
        _launch_scale, Int64(fast.data_ptr()), Int64(exact.data_ptr()), stream
    )
    torch.cuda.synchronize()
    bad = (fast != exact).nonzero().flatten().tolist()
    assert (
        not bad
    ), f"rcp encode scale differs from the IEEE divide for E4M3 bytes {bad}"

    # Every fp16 bit pattern once: 8 per thread, as 4 x i32 of f16x2.
    inp = (
        torch.arange(65536, dtype=torch.int32).to(torch.int16).to(dev).view(torch.int32)
    )
    fast = torch.zeros(inp.numel() // 4, dtype=torch.int32, device=dev)
    exact = torch.zeros_like(fast)
    _run_compiled(
        _launch_nibble,
        Int64(inp.data_ptr()),
        Int64(fast.data_ptr()),
        Int64(exact.data_ptr()),
        stream,
    )
    torch.cuda.synchronize()
    n_bad = int((fast != exact).sum().item())
    assert n_bad == 0, f"magic-number rounding differs from roundeven in {n_bad} words"


if __name__ == "__main__":
    test_quick_allreduce_int4_codec()
    print(
        "quick_allreduce_int4 codec: exact over all 256 scale bytes and 65536 fp16 inputs"
    )
    sys.exit(0)
