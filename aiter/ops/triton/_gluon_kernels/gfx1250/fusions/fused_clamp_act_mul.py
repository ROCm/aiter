# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx1250) port of triton _fused_clamp_silu_mul_kernel for GFX1250
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._triton_kernels.activation import _apply_activation_from_str
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# Human-readable repr for the compiled kernel: lists the constexpr keys that
# identify a unique specialization (shown in traces / cache keys).
_GLUON_REPR_KEYS = [
    "ROWS_PER_PROG",
    "BLOCK_SIZE_N",
    "QUANT_BLOCK_SIZE",
    "SCALE_FMT",
    "HAVE_WEIGHTS",
    "WEIGHT_BROADCAST",
    "HAVE_SWIGLU_CLAMP",
    "HAS_QUANT",
    "num_warps",
    "cache_modifier",
]

_fused_clamp_silu_mul_repr = make_kernel_repr(
    "_fused_clamp_silu_mul_gfx1250_kernel", _GLUON_REPR_KEYS
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
    ROWS_PER_PROG: gl.constexpr,    # ring buffers = rows per program = loop trips
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
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
    NUM_N_Q_GROUPS: gl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE  # quant groups per row

    # one 2d shared layout for ROWS_PER_PROGx2*N
    shared_tdm_layout_2d: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    pid = gl.program_id(0)
    m_start = pid * ROWS_PER_PROG

    # gate + up
    gate_up_smem = gl.allocate_shared_memory(
        inp_ptr.dtype.element_ty, [BLOCK_SIZE_M, 1, 2 * BLOCK_SIZE_N], shared_tdm_layout_2d
    ) # rows per prog tied to LDS, loop should be decoupled TODO

    inp_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=inp_ptr,
        shape=[M, 2 * n_half], # could be M - m_start TODO
        strides=[inp_stride_m, inp_stride_n],
        block_shape=[1, 2 * BLOCK_SIZE_N],
        layout=shared_tdm_layout_2d,
    )

    # load both gate + up
    gl.amd.gfx1250.tdm.async_load(inp_desc, [m_start, 0], gate_up_smem)

    gLayout2D: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[1, 32],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )
    sLayout2D: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], # issue?
        threads_per_warp=[1, 32], # issue?
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )
    row_layout: gl.constexpr = gl.SliceLayout(0, gLayout2D)         # [BLOCK_SIZE_N] row slice
    row_scale_layout: gl.constexpr = gl.SliceLayout(0, sLayout2D)   # [NUM_N_Q_GROUPS] scale slice

    # setup + setup store
    offs = gl.arange(0, BLOCK_SIZE_N, layout=row_layout).to(gl.int64)
    mask = offs < n_half
    num_bs = gl.cdiv(n_half, QUANT_BLOCK_SIZE)
    g_offs = gl.arange(0, NUM_N_Q_GROUPS, layout=row_scale_layout)
    store_col_offs = (offs * out_stride_n).to(gl.int32)

    # main loop
    for i in range(ROWS_PER_PROG):
        row = m_start + i

        # prefetch
        if i + 1 < ROWS_PER_PROG: # turn this to epilogue TODO, remove dynamic
            nxt = i + 1
            gl.amd.gfx1250.tdm.async_load(inp_desc, [m_start + nxt, 0], gate_up_smem.index(nxt))

        # weights load
        if HAVE_WEIGHTS:
            if WEIGHT_BROADCAST:
                w = gl.load(weights_ptr + row * weights_stride_m)  # scalar applied to all out
            else:
                # buffer load weight, also gives slightly better perf
                w = gl.amd.gfx1250.buffer_load(
                    weights_ptr + row.to(gl.int64) * weights_stride_m,
                    (offs * weights_stride_n).to(gl.int32),
                    mask=mask, other=0.0, cache=cache_modifier,
                )

        if SHUFFLE:
            bs_offs_0 = row // 32          # row-tile of 32
            bs_offs_1 = row % 32           # position within the 32-row tile
            bs_offs_2 = bs_offs_1 % 16     # sub-position within 16
            bs_offs_1 = bs_offs_1 // 16    # which half of the 32 (0/1)
            bs_offs_3 = g_offs // 8        # block-col tile of 8
            bs_offs_4 = g_offs % 8         # position within the 8-col tile
            bs_offs_5 = bs_offs_4 % 4      # sub-position within 4
            bs_offs_4 = bs_offs_4 // 4     # which half of the 8 (0/1)
            bs_offs = (                    # weave the sub-indices into the tiled offset
                bs_offs_1
                + bs_offs_4 * 2
                + bs_offs_2 * 2 * 2
                + bs_offs_5 * 2 * 2 * 16
                + bs_offs_3 * 2 * 2 * 16 * 4
                + bs_offs_0 * 2 * 16 * SCALE_N_PAD
            )
        else:
            bs_offs = 0 # not needed

        if i + 1 < ROWS_PER_PROG:
            gl.amd.gfx1250.tdm.async_wait(1)
        else:
            gl.amd.gfx1250.tdm.async_wait(0)

        # reshape then slice
        gate_up = gate_up_smem.index(i).reshape([2 * BLOCK_SIZE_N]) # index(i) -- issue TODO
        gate = gate_up.slice(0, BLOCK_SIZE_N, dim=0).load(row_layout).to(gl.float32)
        up = gate_up.slice(BLOCK_SIZE_N, BLOCK_SIZE_N, dim=0).load(row_layout).to(gl.float32)

        # clamp
        if HAVE_SWIGLU_CLAMP:
            up = gl.clamp(up, -swiglu_limit, swiglu_limit)   # clamp up to [-lim, lim]
            gate = gl.minimum(gate, swiglu_limit)            # clamp gate to <= lim

        # act(gate) * up
        out = _apply_activation_from_str(gate, ACTIVATION) * up

        # apply weights
        if HAVE_WEIGHTS:
            out = out * w.to(gl.float32)

        # group quant and store
        if HAS_QUANT:
            if SCALE_FMT == "ue8m0":
                # mxfp8, reduce over inner QUANT_BLOCK_SIZE axis.
                out_2d = gl.reshape(out, [NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE])
                abs_2d = gl.maximum(out_2d, -out_2d)
                max_val = gl.max(abs_2d, axis=1, keep_dims=True)
                dequant_scale = max_val / DTYPE_MAX
                # ROUND_UP to a power of two via the fp32 exponent field.
                dequant_scale_exp = (
                    dequant_scale.to(gl.uint32, bitcast=True) + 0x007FFFFF
                ) & 0x7F800000
                dequant_scale_rounded = dequant_scale_exp.to(gl.float32, bitcast=True)
                quant_scale = gl.where(                       # reciprocal, guard 0
                    dequant_scale_rounded == 0, 0.0, 1.0 / dequant_scale_rounded
                )
                quant_tensor = out_2d * quant_scale  # scale into fp8 range
                out_q = gl.convert_layout(gl.reshape(quant_tensor, [BLOCK_SIZE_N]), row_layout)
                scale_exp = (dequant_scale_exp >> 23).to(gl.uint8)
                block_scales = gl.convert_layout(gl.reshape(scale_exp, [NUM_N_Q_GROUPS]), row_scale_layout)
            else:
                # fp8 quant
                out_2d = gl.reshape(out, [NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE])
                abs_2d = gl.maximum(out_2d, -out_2d)
                max_val = gl.maximum(
                    gl.max(abs_2d, axis=1, keep_dims=True), 1e-10
                )  # [NUM_N_Q_GROUPS, 1]
                scale_out = max_val / DTYPE_MAX  # dequant (block) scale
                quant_2d = gl.clamp(out_2d * (1.0 / scale_out), DTYPE_MIN, DTYPE_MAX)
                out_q = gl.convert_layout(gl.reshape(quant_2d, [BLOCK_SIZE_N]), row_layout)
                block_scales = gl.convert_layout(
                    gl.reshape(scale_out, [NUM_N_Q_GROUPS]), row_scale_layout
                )

            result = out_q

            if SHUFFLE:
                gl.store(
                    scale_ptr + bs_offs, # exists
                    block_scales.to(scale_ptr.dtype.element_ty),
                    mask=g_offs < num_bs,
                )
            else:
                gl.store(
                    scale_ptr + row * scale_stride_m + g_offs * scale_stride_n,
                    block_scales.to(scale_ptr.dtype.element_ty),
                    mask=g_offs < num_bs,
                )
        else:
            # no quant
            result = out

        # buffer store for a bit of perf uplift
        gl.amd.gfx1250.buffer_store(
            result.to(out_ptr.dtype.element_ty),
            out_ptr + row.to(gl.int64) * out_stride_m,
            store_col_offs,
            mask=mask & (row < M),
        )
