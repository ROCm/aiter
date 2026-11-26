# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gated RMSNorm backward FlyDSL kernel (Kimi Delta Attention output norm).

``out = rmsnorm(o) * w * sigmoid(gp)`` over the last dim ``dim``.

Layout: each lane owns ``EPL = 8`` contiguous elements of a row, so a row takes
``LPR = dim / 8`` lanes and a 64-lane wave covers ``64 / LPR`` rows at once; the
row reductions are xor shuffles inside the LPR-lane group. o, gp and dy are read
in their own dtype with 128-bit buffer loads; do and dgp are written as fp32 with
128-bit buffer stores. A block of ``block`` threads walks ``passes`` groups of
rows (``rows_per_block`` rows in total) and keeps the norm-weight gradient in
registers; the block then reduces it over its row slots through LDS and writes
one ``dim``-wide partial per block. The wrapper sums the partials (stage two).

Out-of-range rows clamp their address to row 0 and are masked out of the stores
and of the weight gradient. The device index is only a cache key.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

_DEFAULT_STREAM = fx.Stream(None)
WAVE = 64
EPL = 8
_DTYPES = {"bf16": fx.BFloat16, "f16": fx.Float16, "f32": fx.Float32}
_BITS = {"bf16": 16, "f16": 16, "f32": 32}


def supported_dim(dim: int) -> bool:
    lpr = dim // EPL
    return dim % EPL == 0 and 0 < lpr <= WAVE and lpr & (lpr - 1) == 0


def _sigmoid(x):
    return fx.Float32(1.0) / (fx.Float32(1.0) + fx.exp(-x))


def _group_sum(v, lanes):
    sh = lanes // 2
    while sh:
        v = v + v.shuffle_xor(fx.Int32(sh), fx.Int32(WAVE))
        sh //= 2
    return v


def _flat(t, span):
    return fx.Tensor(fx.make_view(fx.get_iter(t), fx.make_layout(span, 1)))


def _vec_tiles(t, span, width):
    flat = fx.make_view(
        fx.get_iter(fx.rocdl.make_buffer_tensor(t)), fx.make_layout(span, 1)
    )
    return fx.logical_divide(flat, fx.make_layout(width, 1))


@functools.lru_cache(maxsize=32)
def create_gated_rmsnorm_bwd_kernel(
    dim: int,
    eps: float,
    in_dtype: str = "bf16",
    block: int = 256,
    passes: int = 4,
    device: int | None = None,
):
    """o, gp, dy: [rows, dim] in ``in_dtype``; w: [dim] fp32;
    do, dgp: [rows, dim] fp32; dw_part: [n_blocks, dim] fp32.

    out = o * r * w * sigmoid(gp), r = rsqrt(mean(o^2) + eps).
    Grid: ``n_blocks = ceil(rows / (block // (dim // 8) * passes))``.
    """
    assert supported_dim(dim), dim
    assert block % WAVE == 0
    LPR = dim // EPL
    SLOTS = block // LPR
    ROWS_PER_BLOCK = SLOTS * passes
    in_ty = _DTYPES[in_dtype]
    IN_W = 128 // _BITS[in_dtype]
    IN_N = EPL // IN_W
    OUT_W = 4
    OUT_N = EPL // OUT_W

    @fx.struct
    class SharedStorage:
        dw: fx.Array[fx.Float32, block * EPL, 16]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def gated_rmsnorm_bwd_kernel(
        o: fx.Tensor,
        gp: fx.Tensor,
        w: fx.Tensor,
        dy: fx.Tensor,
        do: fx.Tensor,
        dgp: fx.Tensor,
        dw_part: fx.Tensor,
        rows: fx.Int64,
        span: fx.Int64,
        dw_span: fx.Int64,
    ):
        dw_part = _flat(dw_part, dw_span)
        tid = fx.Int64(fx.thread_idx.x)
        bid = fx.Int64(fx.block_idx.x)
        slot = tid // LPR
        sub = tid % LPR
        col = sub * EPL

        o_t = _vec_tiles(o, span, IN_W)
        g_t = _vec_tiles(gp, span, IN_W)
        dy_t = _vec_tiles(dy, span, IN_W)
        do_t = _vec_tiles(do, span, OUT_W)
        dg_t = _vec_tiles(dgp, span, OUT_W)
        w_t = _vec_tiles(w, fx.Int64(dim), OUT_W)
        in_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), in_ty)
        f32_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

        def load_in(tiles, elem):
            vals = []
            for j in range_constexpr(IN_N):
                frag = fx.make_rmem_tensor(IN_W, in_ty)
                idx = (elem + j * IN_W) // IN_W
                fx.copy(in_atom, fx.slice(tiles, (None, idx)), frag)
                v = fx.Vector(fx.memref_load_vec(frag))
                if in_dtype != "f32":
                    v = v.extf(T.vec(IN_W, T.f32))
                for i in range_constexpr(IN_W):
                    vals.append(fx.Float32(v[i]))
            return vals

        def store_f32(tiles, elem, vals):
            for j in range_constexpr(OUT_N):
                frag = fx.make_rmem_tensor(OUT_W, fx.Float32)
                fx.memref_store_vec(
                    fx.Vector.from_elements(
                        vals[j * OUT_W : (j + 1) * OUT_W], fx.Float32
                    ),
                    frag,
                )
                idx = (elem + j * OUT_W) // OUT_W
                fx.copy(f32_atom, frag, fx.slice(tiles, (None, idx)))

        wv = []
        for j in range_constexpr(OUT_N):
            frag = fx.make_rmem_tensor(OUT_W, fx.Float32)
            fx.copy(f32_atom, fx.slice(w_t, (None, sub * OUT_N + j)), frag)
            v = fx.Vector(fx.memref_load_vec(frag))
            for i in range_constexpr(OUT_W):
                wv.append(fx.Float32(v[i]))

        dwacc = [fx.Float32(0.0) for _ in range_constexpr(EPL)]
        for p in range_constexpr(passes):
            row = bid * ROWS_PER_BLOCK + p * SLOTS + slot
            valid = row < rows
            base = valid.select(row, fx.Int64(0)) * dim + col
            ov = load_in(o_t, base)
            gv = [_sigmoid(x) for x in load_in(g_t, base)]
            # rows past the end contribute nothing to the weight gradient
            dv = [valid.select(x, fx.Float32(0.0)) for x in load_in(dy_t, base)]
            ss = fx.Float32(0.0)
            for e in range_constexpr(EPL):
                ss = ss + ov[e] * ov[e]
            rs = fx.Float32(1.0) / fx.sqrt(
                _group_sum(ss, LPR) * fx.Float32(1.0 / dim) + fx.Float32(eps)
            )
            dh = [dv[e] * wv[e] * gv[e] for e in range_constexpr(EPL)]
            dot = fx.Float32(0.0)
            for e in range_constexpr(EPL):
                dot = dot + dh[e] * ov[e]
            coef = _group_sum(dot, LPR) * rs * rs * rs * fx.Float32(1.0 / dim)
            d_o = []
            d_g = []
            for e in range_constexpr(EPL):
                on = ov[e] * rs
                t = dv[e] * on * gv[e]
                dwacc[e] = dwacc[e] + t
                d_o.append(dh[e] * rs - ov[e] * coef)
                d_g.append(t * wv[e] * (fx.Float32(1.0) - gv[e]))
            if valid:
                store_f32(do_t, base, d_o)
                store_f32(dg_t, base, d_g)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek().dw
        for e in range_constexpr(EPL):
            lds[tid * EPL + e] = dwacc[e]
        fx.gpu.barrier()
        for c in range_constexpr((dim + block - 1) // block):
            cc = tid + c * block
            if cc < dim:
                s = fx.Float32(0.0)
                lane_of = cc // EPL
                e_of = cc % EPL
                for k in range_constexpr(SLOTS):
                    s = s + fx.Float32(lds[(k * LPR + lane_of) * EPL + e_of])
                dw_part[bid * dim + cc] = s

    @flyc.jit
    def launch(
        o: fx.Tensor,
        gp: fx.Tensor,
        w: fx.Tensor,
        dy: fx.Tensor,
        do: fx.Tensor,
        dgp: fx.Tensor,
        dw_part: fx.Tensor,
        rows: fx.Int64,
        n_blocks: fx.Int64,
        span: fx.Int64,
        stream: fx.Stream = _DEFAULT_STREAM,
    ):
        gated_rmsnorm_bwd_kernel(
            o, gp, w, dy, do, dgp, dw_part, rows, span, n_blocks * dim
        ).launch(grid=(n_blocks, 1, 1), block=(block, 1, 1), stream=stream)

    return launch
