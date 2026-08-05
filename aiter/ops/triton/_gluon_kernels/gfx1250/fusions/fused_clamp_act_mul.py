# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx1250) port of ``_fused_clamp_silu_mul_kernel`` — 2D [BLOCK_SIZE_M,
BLOCK_SIZE_N] tile (2D layouts modeled on ``norm/fused_rmsnorm_add.py``).

Semantics mirror the Triton reference in
``aiter/ops/triton/_triton_kernels/fusions/fused_clamp_act_mul.py``: per token row
of an ``[M, 2*N]`` input (gate = first ``N`` cols, up = second ``N``): optional
SwiGLU clamp -> ``act(gate) * up`` -> optional weights -> optional
per-``QUANT_BLOCK_SIZE`` FP8 group quant, with an optional shuffled scale store.

Each program owns ``BLOCK_SIZE_M`` rows and loops over ``N_CHUNKS`` column chunks
of ``BLOCK_SIZE_N`` (grid = cdiv(M, BLOCK_SIZE_M)). When ``BLOCK_SIZE_N >= n_half``
there is a single chunk and the loop is a no-op wrapper (identical to the
single-tile path); when ``BLOCK_SIZE_N < n_half`` the ``N_CHUNKS = cdiv(n_half,
BLOCK_SIZE_N)`` chunks tile the row. Because ``BLOCK_SIZE_N`` is a multiple of
``QUANT_BLOCK_SIZE`` (asserted host-side), each chunk owns whole quant groups, so
the per-group quant reduction is independent per chunk. Tensors carry explicit 2D
``BlockedLayout``s (``gLayout2D`` for the data tile, ``sLayout2D`` for the
``[BM, G]`` scale tile); gate/up (and weights) are ``buffer_load``ed into
registers with 2D offset tiles and the result ``buffer_store``d, both bounded by
the tile mask (row tail + column overhang).
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._triton_kernels.activation import _apply_activation_from_str
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


# Human-readable repr for the compiled kernel: lists the constexpr keys that
# identify a unique specialization (shown in traces / cache keys).
_fused_clamp_silu_mul_repr = make_kernel_repr(
    "_fused_clamp_silu_mul_gfx1250_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "N_CHUNKS",
        "QUANT_BLOCK_SIZE",
        "SCALE_FMT",
        "HAVE_WEIGHTS",
        "WEIGHT_BROADCAST",
        "HAVE_SWIGLU_CLAMP",
        "HAS_QUANT",
        "num_warps",
        "cache_modifier",
    ],
)


@gluon.jit(repr=_fused_clamp_silu_mul_repr)
def _fused_clamp_silu_mul_kernel(
    inp_ptr,             # *in*  [M, 2*n_half] gate|up input
    out_ptr,             # *out* [M, n_half] result (native dtype or FP8)
    scale_ptr,           # *out* [M, num_blocks] block scales (only if HAS_QUANT)
    weights_ptr,         # *in*  [M,1] or [M,n_half] weights (only if HAVE_WEIGHTS)
    M,                   # number of token rows
    n_half,              # N — half the input's inner dim (gate/up width)
    inp_stride_m,        # inp row stride
    inp_stride_n,        # inp col stride
    out_stride_m,        # out row stride
    out_stride_n,        # out col stride
    scale_stride_m,      # scale row stride
    scale_stride_n,      # scale col stride
    weights_stride_m,    # weights row stride
    weights_stride_n,    # weights col stride
    swiglu_limit,        # clamp bound (used only when HAVE_SWIGLU_CLAMP)
    BLOCK_SIZE_M: gl.constexpr,   # rows per tile
    BLOCK_SIZE_N: gl.constexpr,   # cols per chunk (multiple of QUANT_BLOCK_SIZE)
    N_CHUNKS: gl.constexpr,       # cdiv(n_half, BLOCK_SIZE_N) column chunks per row
    QUANT_BLOCK_SIZE: gl.constexpr,
    SCALE_FMT: gl.constexpr,
    DTYPE_MAX: gl.constexpr,      # +max of the quant dtype (for scale calc)
    DTYPE_MIN: gl.constexpr,      # -max of the quant dtype
    HAVE_WEIGHTS: gl.constexpr,
    WEIGHT_BROADCAST: gl.constexpr,
    HAVE_SWIGLU_CLAMP: gl.constexpr,
    HAS_QUANT: gl.constexpr,
    ACTIVATION: gl.constexpr,     # "silu" | "gelu" | "gelu_tanh"
    SHUFFLE: gl.constexpr,        # write scales in preshuffled tiled layout
    SCALE_N_PAD: gl.constexpr,    # padded scale-col count (shuffle addressing)
    num_warps: gl.constexpr,
    cache_modifier: gl.constexpr,
):
    # constants
    NUM_QB: gl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE  # quant groups per chunk

    gLayout2D: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, max(1, BLOCK_SIZE_N // (num_warps * 32))],
        threads_per_warp=[1, 32],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )
    sLayout2D: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, max(1, NUM_QB // (num_warps * 32))],
        threads_per_warp=[1, 32],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )
    rowN: gl.constexpr = gl.SliceLayout(1, gLayout2D)   # [BM] row-index vector (data tile)
    colN: gl.constexpr = gl.SliceLayout(0, gLayout2D)   # [BN] col-index vector
    rowS: gl.constexpr = gl.SliceLayout(1, sLayout2D)   # [BM] row-index vector (scale tile)
    colS: gl.constexpr = gl.SliceLayout(0, sLayout2D)   # [G]  group-index vector

    # setup (loop-invariant across the N chunks)
    pid = gl.program_id(0)
    m_start = pid * BLOCK_SIZE_M
    rows = m_start + gl.arange(0, BLOCK_SIZE_M, layout=rowN)   # [BM]
    row_ok = rows < M
    inp_row_base = rows[:, None] * inp_stride_m               # [BM, 1]

    # N-chunk loop
    for c in gl.static_range(N_CHUNKS):
        cols = c * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=colN)  # [BN] abs gate col
        col_ok = cols < n_half
        tile_mask = row_ok[:, None] & col_ok[None, :]         # [BM, BN]

        # buffer-load offsets for the gate / up halves
        gate_offs = (inp_row_base + cols[None, :] * inp_stride_n).to(gl.int32)
        up_offs = (inp_row_base + (cols + n_half)[None, :] * inp_stride_n).to(gl.int32)

        # load gate and up
        gate = gl.amd.gfx1250.buffer_load(
            ptr=inp_ptr, offsets=gate_offs, mask=tile_mask, other=0.0,
            cache=cache_modifier,
        ).to(gl.float32)                                      # [BM, BN]
        up = gl.amd.gfx1250.buffer_load(
            ptr=inp_ptr, offsets=up_offs, mask=tile_mask, other=0.0,
            cache=cache_modifier,
        ).to(gl.float32)                                      # [BM, BN]

        # clamp
        if HAVE_SWIGLU_CLAMP:
            up = gl.clamp(up, -swiglu_limit, swiglu_limit)    # clamp up to [-lim, lim]
            gate = gl.minimum(gate, swiglu_limit)             # clamp gate to <= lim

        # act(gate) * up
        out = _apply_activation_from_str(gate, ACTIVATION) * up  # [BM, BN]

        # weights
        if HAVE_WEIGHTS:
            if WEIGHT_BROADCAST:
                # [M,1] broadcast: every column reads the row's single weight
                # (col 0), so `out * w` is a same-layout [BM, BN] multiply.
                w_offs = (
                    rows[:, None] * weights_stride_m + (cols * 0)[None, :]
                ).to(gl.int32)
            else:
                w_offs = (
                    rows[:, None] * weights_stride_m
                    + cols[None, :] * weights_stride_n
                ).to(gl.int32)
            w = gl.amd.gfx1250.buffer_load(
                ptr=weights_ptr, offsets=w_offs, mask=tile_mask, other=0.0,
                cache=cache_modifier,
            ).to(gl.float32)                                  # [BM, BN]
            out = out * w

        # group quant and store
        if HAS_QUANT:
            if SCALE_FMT == "ue8m0":
                # mxfp8, reduce over inner QUANT_BLOCK_SIZE axis.
                out_3d = gl.reshape(out, [BLOCK_SIZE_M, NUM_QB, QUANT_BLOCK_SIZE])
                abs_3d = gl.maximum(out_3d, -out_3d)          # tl.abs
                max_val = gl.max(abs_3d, axis=2, keep_dims=True)  # [BM, NQB, 1]
                dequant_scale = max_val / DTYPE_MAX
                # ROUND_UP to a power of two via the fp32 exponent field.
                dequant_scale_exp = (
                    dequant_scale.to(gl.uint32, bitcast=True) + 0x007FFFFF
                ) & 0x7F800000
                dequant_scale_rounded = dequant_scale_exp.to(gl.float32, bitcast=True)
                quant_scale = gl.where(                       # reciprocal, guard 0
                    dequant_scale_rounded == 0, 0.0, 1.0 / dequant_scale_rounded
                )
                out_q = gl.convert_layout(
                    gl.reshape(out_3d * quant_scale, [BLOCK_SIZE_M, BLOCK_SIZE_N]),
                    gLayout2D,
                )
                # ue8m0 scale = the biased exponent byte (bits 23..30) as uint8.
                scale_exp = (dequant_scale_exp >> 23).to(gl.uint8)  # [BM, NQB, 1]
                block_scales = gl.convert_layout(
                    gl.reshape(scale_exp, [BLOCK_SIZE_M, NUM_QB]), sLayout2D
                )
            else:
                # fp8 quant, mimics triton function
                out_3d = gl.reshape(out, [BLOCK_SIZE_M, NUM_QB, QUANT_BLOCK_SIZE])
                abs_3d = gl.maximum(out_3d, -out_3d)
                max_val = gl.maximum(
                    gl.max(abs_3d, axis=2, keep_dims=True), 1e-10
                )                                             # [BM, NQB, 1]
                scale_out = max_val.to(gl.float32) / DTYPE_MAX  # [BM, NQB, 1]
                quant_3d = gl.clamp(out_3d * (1.0 / scale_out), DTYPE_MIN, DTYPE_MAX)
                out_q = gl.convert_layout(
                    gl.reshape(quant_3d, [BLOCK_SIZE_M, BLOCK_SIZE_N]), gLayout2D
                )
                block_scales = gl.convert_layout(
                    gl.reshape(scale_out, [BLOCK_SIZE_M, NUM_QB]), sLayout2D
                )

            # store
            out_offs = (
                rows[:, None] * out_stride_m + cols[None, :] * out_stride_n
            ).to(gl.int32)
            gl.amd.gfx1250.buffer_store(
                out_q.to(out_ptr.dtype.element_ty), out_ptr, out_offs, mask=tile_mask
            )
            num_bs = gl.cdiv(n_half, QUANT_BLOCK_SIZE)         # total valid scale groups
            srows = m_start + gl.arange(0, BLOCK_SIZE_M, layout=rowS)  # [BM]
            g = c * NUM_QB + gl.arange(0, NUM_QB, layout=colS)  # [G] absolute group
            scale_mask = (srows[:, None] < M) & (g[None, :] < num_bs)  # [BM, G]
            if SHUFFLE:
                r0 = srows[:, None]
                bs_offs_0 = r0 // 32          # row-tile of 32
                bs_offs_1 = r0 % 32           # position within the 32-row tile
                bs_offs_2 = bs_offs_1 % 16    # sub-position within 16
                bs_offs_1 = bs_offs_1 // 16   # which half of the 32 (0/1)
                gg = g[None, :]
                bs_offs_3 = gg // 8           # block-col tile of 8
                bs_offs_4 = gg % 8            # position within the 8-col tile
                bs_offs_5 = bs_offs_4 % 4     # sub-position within 4
                bs_offs_4 = bs_offs_4 // 4    # which half of the 8 (0/1)
                bs_offs = (                   # weave the sub-indices into the tiled offset
                    bs_offs_1
                    + bs_offs_4 * 2
                    + bs_offs_2 * 2 * 2
                    + bs_offs_5 * 2 * 2 * 16
                    + bs_offs_3 * 2 * 2 * 16 * 4
                    + bs_offs_0 * 2 * 16 * SCALE_N_PAD
                )
                gl.amd.gfx1250.buffer_store(
                    block_scales.to(scale_ptr.dtype.element_ty),
                    scale_ptr, bs_offs.to(gl.int32), mask=scale_mask,
                )
            else:
                sptr_offs = (
                    srows[:, None] * scale_stride_m + g[None, :] * scale_stride_n
                ).to(gl.int32)
                gl.amd.gfx1250.buffer_store(
                    block_scales.to(scale_ptr.dtype.element_ty),
                    scale_ptr, sptr_offs, mask=scale_mask,
                )
        else:
            # no quant
            out_offs = (
                rows[:, None] * out_stride_m + cols[None, :] * out_stride_n
            ).to(gl.int32)
            gl.amd.gfx1250.buffer_store(
                out.to(out_ptr.dtype.element_ty), out_ptr, out_offs, mask=tile_mask
            )
