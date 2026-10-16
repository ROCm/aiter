# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Deterministic top-k combine: out[t] = sum_s y_rows[inv[t*k + s]] (weights already applied).

y_rows: [R, H] bf16 in compact expert-sorted order. out: [T, H] bf16.
One CTA per token, 16 B (8 bf16) per thread per chunk, fp32 accumulation.
gated: the pass is a no-op when the int32 device flag at skip_ptr is non-zero.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from . import primitives


@functools.cache
def build_combine(H: int, k: int, threads: int = 256, gated: bool = False):
    assert H % (8 * threads) == 0 or H % 8 == 0
    chunks = (H // 8 + threads - 1) // threads
    exact = (H // 8) % threads == 0

    kname = (
        f"moe_a4w4_compact_combine_h{H}_k{k}{'_g' if gated else ''}"
        f"_{primitives.SRC_HASH}"
    )

    def _body(y_ptr, inv_ptr, o_ptr):
        tid = fx.Int32(gpu.thread_id("x"))
        t = fx.Int32(gpu.block_id("x"))
        r_inv = primitives.rsrc(inv_ptr)
        # 64-bit bases: R * H * 2 can exceed 2^31.
        r_o = primitives.rsrc(fx.Int64(o_ptr) + fx.Int64(t) * fx.Int64(H * 2), H * 2)
        r_rows = []
        for s in range_constexpr(k):
            row = fx.Int32(
                rocdl.readfirstlane(
                    T.i32, primitives.bload(r_inv, (t * k + s) * 4, T.i32)
                )
            )
            r_rows.append(
                primitives.rsrc(
                    fx.Int64(y_ptr) + fx.Int64(row) * fx.Int64(H * 2), H * 2
                )
            )
        for c in range_constexpr(chunks):
            col = (tid + c * threads) * 8
            ok = col < H if not exact else None
            vals = [
                fx.Vector(primitives.bload(r_rows[s], col * 2, T.vec(8, T.bf16))).to(
                    fx.Float32
                )
                for s in range_constexpr(k)
            ]
            acc = vals[0]
            for s in range_constexpr(1, k):
                acc = acc + vals[s]
            off = col * 2
            if ok is not None:
                off = ok.select(off, fx.Int32(0x7FFFFF00))
            primitives.bstore(acc.to(fx.BFloat16), r_o, off)

    _body = ASTRewriter.transform(_body)

    @flyc.kernel(name=kname, known_block_size=[threads, 1, 1])
    def kern(
        y_ptr: fx.Int64,
        inv_ptr: fx.Int64,
        o_ptr: fx.Int64,
        n_rows: fx.Int32,
        n_tok: fx.Int32,
        skip_ptr: fx.Int64,
    ):
        if const_expr(kname == ""):  # name (incl. source hash) in the JIT cache key
            pass
        if const_expr(gated):
            skip = fx.Int32(
                rocdl.readfirstlane(
                    T.i32, primitives.bload(primitives.rsrc(skip_ptr, 4), 0, T.i32)
                )
            )
            if skip == fx.Int32(0):
                _body(y_ptr, inv_ptr, o_ptr)
        else:
            _body(y_ptr, inv_ptr, o_ptr)

    @flyc.jit
    def launch(
        y_ptr: fx.Int64,
        inv_ptr: fx.Int64,
        o_ptr: fx.Int64,
        n_rows: fx.Int32,
        n_tok: fx.Int32,
        skip_ptr: fx.Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kern(y_ptr, inv_ptr, o_ptr, n_rows, n_tok, skip_ptr).launch(
            grid=(n_tok, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch


_cf = {}


def run_combine(y_rows, inv, out, k, skip=None, stream=None):
    """skip: optional int32 device flag (gated combine, see module docstring)."""
    import torch

    T_, H = out.shape
    key = (H, k, skip is not None)
    args = (
        y_rows.data_ptr(),
        inv.data_ptr(),
        out.data_ptr(),
        y_rows.shape[0],
        T_,
        0 if skip is None else skip.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    if key not in _cf:
        _cf[key] = flyc.compile(build_combine(H, k, gated=skip is not None), *args)
    else:
        _cf[key](*args)
