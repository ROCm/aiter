# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused swiglu_and_mul kernel for interleaved (N0, 2, NLANE) layout (FlyDSL).

Input (from cktile split_k with a16w4 interleave preshuffle): each row has
inter_dim*2 bf16 columns laid out as
    [gate_b0(NLANE), up_b0(NLANE), gate_b1(NLANE), up_b1(NLANE), ...]
i.e. shape (N0, 2, NLANE) with N0 = inter_dim // NLANE, NLANE = 16.

Per output element:
    out = clamp(gate, max=limit) * sigmoid(alpha * clamp(gate, max=limit))
          * (clamp(up, -limit, limit) + 1)

The interleave is just a layout: gate = x[:, 0, :] and up = x[:, 1, :] are both
(N0, NLANE) views with stride (2*NLANE, 1); the output is the contiguous
(N0, NLANE) with stride (NLANE, 1). A single TV-tiled copy partitions all three,
so each thread loads V contiguous gate + up elems, applies swiglu in f32, and
stores V bf16 -- vectorized 128b loads/stores instead of per-dword scalar work.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, ptrtoint, range_constexpr
from flydsl.expr.typing import T

NLANE = 16
ALPHA = 1.702
LIMIT = 7.0
V = 8  # bf16 per 128b buffer copy


def build_swiglu_and_mul_module(inter_dim: int):
    """Return a JIT launcher for fused swiglu_and_mul on interleaved input.

    inter_dim is the output column count; input has inter_dim*2 columns.
    """
    assert inter_dim % NLANE == 0
    N0 = inter_dim // NLANE
    row_in = inter_dim * 2  # input elems per row
    NT = NLANE // V  # threads per block-row
    # inter_dim is compile-time, so size the tile to N0: one M-tile when it fits,
    # else 128 block-rows per tile with the ragged remainder dropped by the V#
    # descriptor (out-of-row loads read 0, their stores are OOB-dropped).
    BM = min(N0, 128)
    THREADS = BM * NT
    GY = (N0 + BM - 1) // BM

    @flyc.kernel
    def swiglu_and_mul_kernel(x: fx.Pointer, out: fx.Pointer, num_rows: fx.Int32):
        row = fx.Int64(gpu.block_id("x"))
        tile = gpu.block_id("y")
        tid = gpu.thread_id("x")

        def _view(base_i64, strides, nbytes):  # (N0, NLANE) bf16 V# buffer tensor
            pt = fx.PointerType.get(
                fx.BFloat16.ir_type, address_space=fx.AddressSpace.Global, alignment=2
            )
            view = fx.make_view(
                fx.inttoptr(pt, base_i64), fx.make_layout((N0, NLANE), strides)
            )
            return fx.rocdl.make_buffer_tensor(view, num_records_bytes=fx.Int64(nbytes))

        # Fold the per-row byte offset into each base ptr (i32-safe voffsets).
        xrow = fx.Int64(ptrtoint(x)) + row * fx.Int64(row_in * 2)
        orow = fx.Int64(ptrtoint(out)) + row * fx.Int64(inter_dim * 2)
        gate = _view(xrow, (2 * NLANE, 1), row_in * 2)
        up = _view(xrow + fx.Int64(NLANE * 2), (2 * NLANE, 1), row_in * 2 - NLANE * 2)
        ybuf = _view(orow, (NLANE, 1), inter_dim * 2)

        copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        tile_mn, tv = fx.make_layout_tv(
            fx.make_layout((BM, NT), (NT, 1)), fx.make_layout((1, V), (V, 1))
        )
        thr = fx.make_tiled_copy(copy, tv, tile_mn).get_slice(tid)

        def _part(buf, fn):
            return fn(fx.slice(fx.zipped_divide(buf, tile_mn), (None, (tile, 0))))

        p_gate = _part(gate, thr.partition_S)
        p_up = _part(up, thr.partition_S)
        p_out = _part(ybuf, thr.partition_D)
        gf, uf = fx.make_fragment_like(p_gate), fx.make_fragment_like(p_up)
        fx.copy(copy, p_gate, gf)
        fx.copy(copy, p_up, uf)

        f32 = T.f32
        neg_limit, one = fx.Float32(-LIMIT), fx.Float32(1.0)
        neg_alpha_log2e = fx.Float32(-ALPHA * 1.4426950408889634)
        gv = fx.Vector(fx.memref_load_vec(gf)).extf(T.vec(V, f32))
        uv = fx.Vector(fx.memref_load_vec(uf)).extf(T.vec(V, f32))
        outs = []
        for i in range_constexpr(V):
            # clamp: min(x, limit) == -max(-x, -limit) (fx has maximumf, not minimumf)
            g = -((-gv[i]).maximumf(neg_limit))
            u = (-((-uv[i]).maximumf(neg_limit))).maximumf(neg_limit)
            # sigmoid(alpha*g) via exp2/rcp: 1/(1+exp2(g*-alpha*log2e)).
            emu = fx.Float32(fx.rocdl.exp2(f32, (g * neg_alpha_log2e).ir_value()))
            sig = fx.Float32(fx.rocdl.rcp(f32, (one + emu).ir_value()))
            outs.append((g * sig * (u + one)).to(fx.BFloat16))
        of = fx.make_fragment_like(p_out)
        fx.memref_store_vec(fx.Vector.from_elements(outs, fx.BFloat16), of)
        fx.copy(copy, of, p_out)

    @flyc.jit
    def launch_swiglu_and_mul(
        x: fx.Pointer, out: fx.Pointer, num_rows: fx.Int32, stream: fx.Stream
    ):
        swiglu_and_mul_kernel(x, out, num_rows).launch(
            grid=(fx.Int64(num_rows), GY, 1), block=(THREADS, 1, 1), stream=stream
        )

    return launch_swiglu_and_mul
