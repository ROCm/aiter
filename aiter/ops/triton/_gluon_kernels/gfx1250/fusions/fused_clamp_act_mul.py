# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx1250) port of ``_fused_clamp_silu_mul_kernel``.

Semantics mirror the Triton reference in
``aiter/ops/triton/_triton_kernels/fusions/fused_clamp_act_mul.py``: per token row
of an ``[M, 2*N]`` input (gate = first ``N`` cols, up = second ``N``):
optional SwiGLU clamp -> ``act(gate) * up`` -> optional weights -> optional
per-``QUANT_BLOCK_SIZE`` FP8 group quant, with an optional shuffled scale store.

Each program processes ``BLOCK_M`` consecutive rows in a ``gl.static_range``
loop, ONE row per iteration (grid = cdiv(M, BLOCK_M)). Processing one row at a
time keeps the per-iteration register/LDS footprint at a single row (unlike
widening the register-distributed tile, which scales VGPRs linearly with the row
count); BLOCK_M just amortizes more rows onto each workgroup. Requires
``M % BLOCK_M == 0`` (asserted host-side) so no iteration addresses a row >= M.
With ``BLOCK_M == 1`` this is exactly the per-row version.

Gluon differences from the Triton reference: tensors carry explicit
``BlockedLayout``s (``row_layout`` for the data vector, ``row_scale_layout`` for
the scale vector), ``tl.ravel`` becomes ``gl.reshape`` back to 1D +
``gl.convert_layout`` onto the store layout, ``tl.abs`` becomes
``gl.maximum(x, -x)``, and the ue8m0 group reduction uses a 2D
``[NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE]`` reshape reduced over ``axis=1``.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._triton_kernels.activation import _apply_activation_from_str
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# Human-readable repr for the compiled kernel: lists the constexpr keys that
# identify a unique specialization (shown in traces / cache keys).
_GLUON_REPR_KEYS = [
    "BLOCK_M",
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
    BLOCK_M: gl.constexpr,        # rows processed per program (loop trip count)
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
    # 1D layouts
    row_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[max(1, BLOCK_SIZE_N // (num_warps * 32))], # div N over lanes, floor 1
        threads_per_warp=[32],
        warps_per_cta=[num_warps], # warps per thread block
        order=[0],
    )
    row_scale_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[max(1, NUM_N_Q_GROUPS // (num_warps * 32))], # scale group over lanes, floor 1
        threads_per_warp=[32],
        warps_per_cta=[num_warps],
        order=[0],
    )
    shared_tdm_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])

    pid = gl.program_id(0)                                  # pid = row-block
    offs = gl.arange(0, BLOCK_SIZE_N, layout=row_layout).to(gl.int64)  # offsets
    mask = offs < n_half                                    # mask
    num_bs = gl.cdiv(n_half, QUANT_BLOCK_SIZE)              # valid scale groups
    g_offs = gl.arange(0, NUM_N_Q_GROUPS, layout=row_scale_layout)  # scale-group indices

    out_smem = gl.allocate_shared_memory(
        out_ptr.dtype.element_ty, [BLOCK_SIZE_N], shared_tdm_layout
    )
    gate_smem = gl.allocate_shared_memory(
        inp_ptr.dtype.element_ty, [BLOCK_SIZE_N], shared_tdm_layout
    )
    up_smem = gl.allocate_shared_memory(
        inp_ptr.dtype.element_ty, [BLOCK_SIZE_N], shared_tdm_layout
    )

    # loop through rows, 
    for r in gl.static_range(BLOCK_M):
        row = pid * BLOCK_M + r
        row_base = row * inp_stride_m

        if num_warps > 1:
            gl.barrier()

        gate_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=inp_ptr + row_base,
            shape=[n_half],
            strides=[inp_stride_n],
            block_shape=[BLOCK_SIZE_N],
            layout=shared_tdm_layout,
        )
        up_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=inp_ptr + row_base + n_half * inp_stride_n,
            shape=[n_half],
            strides=[inp_stride_n],
            block_shape=[BLOCK_SIZE_N],
            layout=shared_tdm_layout,
        )
        out_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=out_ptr + row * out_stride_m,
            shape=[n_half],
            strides=[out_stride_n],
            block_shape=[BLOCK_SIZE_N],
            layout=shared_tdm_layout,
        )
        gl.amd.gfx1250.tdm.async_load(gate_desc, [0], gate_smem)
        gl.amd.gfx1250.tdm.async_load(up_desc, [0], up_smem)

        # weights
        if HAVE_WEIGHTS:
            if WEIGHT_BROADCAST:
                w = gl.load(weights_ptr + row * weights_stride_m).to(gl.float32)  # scalar applied to all out
            else:
                w = gl.load(
                    weights_ptr + row * weights_stride_m + offs * weights_stride_n,
                    mask=mask, other=0.0, cache_modifier=cache_modifier,
                ).to(gl.float32)

        gl.amd.gfx1250.tdm.async_wait(1)
        gate = gate_smem.load(row_layout).to(gl.float32)
        gl.amd.gfx1250.tdm.async_wait(0)
        up = up_smem.load(row_layout).to(gl.float32)

        # clamp
        if HAVE_SWIGLU_CLAMP:
            up = gl.clamp(up, -swiglu_limit, swiglu_limit)   # clamp up to [-lim, lim]
            gate = gl.minimum(gate, swiglu_limit)            # clamp gate to <= lim

        # act(gate) * up
        out = _apply_activation_from_str(gate, ACTIVATION) * up
        
        # apply weights
        if HAVE_WEIGHTS:
            out = out * w

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

            out_smem.store(out_q.to(out_ptr.dtype.element_ty))  # stage for TDM store

            if SHUFFLE:
                # Preshuffled scale store: same tiled index math as the Triton
                # reference (rows padded to 256, block-cols to 8, tiled layout).
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
                gl.store(
                    scale_ptr + bs_offs,
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
            out_smem.store(out.to(out_ptr.dtype.element_ty))  # stage for TDM store

        # TDM store (barrier so all warps finished writing out_smem first).
        if num_warps > 1:
            gl.barrier()
        gl.amd.gfx1250.tdm.async_store(out_desc, [0], out_smem)
        gl.amd.gfx1250.tdm.async_wait(0)
