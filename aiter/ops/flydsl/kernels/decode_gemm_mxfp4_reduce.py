# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Stateless, fixed-order reduction of compact fp32 decode split-K planes."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr

from aiter.ops.flydsl.kernels.kernels_common import ceildiv


@lru_cache(maxsize=128)
def compile_decode_gemm_mxfp4_reduce(*, M, N, split_k):
    @flyc.kernel
    def kernel(workspace: fx.Tensor, out: fx.Tensor):
        bid = fx.Int32(fx.gpu.block_id("x"))
        tid = fx.Int32(fx.gpu.thread_id("x"))
        src = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(workspace, max_size=False)),
            fx.make_layout((M * N, split_k), (1, M * N)),
        )
        src = fx.logical_divide(src, fx.make_tile(fx.make_layout(1024, 1), None))
        src = fx.slice(src, ((None, bid), None))
        src = fx.logical_divide(src, fx.make_tile(fx.make_layout(4, 1), None))
        src = fx.slice(src, ((None, tid), None))
        dst = fx.rocdl.make_buffer_tensor(out, max_size=False)
        dst = fx.logical_divide(dst, fx.make_layout(1024, 1))
        dst = fx.logical_divide(fx.slice(dst, (None, bid)), fx.make_layout(4, 1))
        load = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        store = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
        if bid * 1024 + tid * 4 < M * N:
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for s in range_constexpr(split_k):
                r = fx.make_rmem_tensor(4, fx.Float32)
                fx.copy(load, fx.slice(src, (None, s)), r)
                acc = acc + r.load()
            r_out = fx.make_rmem_tensor(4, fx.BFloat16)
            r_out.store(acc.to(fx.BFloat16))
            fx.copy(store, r_out, fx.slice(dst, (None, tid)))

    @flyc.jit
    def launch(workspace: fx.Tensor, out: fx.Tensor, stream: fx.Stream):
        kernel(workspace, out).launch(
            grid=(ceildiv(M * N, 1024),), block=(256,), stream=stream
        )

    return launch
