# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx1250) port of ``_fused_clamp_silu_mul_kernel``.

Semantics mirror the Triton reference in
``aiter/ops/triton/_triton_kernels/fusions/fused_clamp_act_mul.py``: one program
per token row of an ``[M, 2*N]`` input (gate = first ``N`` cols, up = second ``N``):
optional SwiGLU clamp -> ``act(gate) * up`` -> optional weights -> optional
per-``QUANT_BLOCK_SIZE`` FP8 group quant, with an optional shuffled scale store.

Gluon differences from the Triton reference (each noted inline below): tensors
carry explicit ``BlockedLayout``s (``L`` for the data vector, ``L_SCALE`` for the
scale vector), ``tl.ravel`` becomes ``gl.reshape`` back to 1D + ``gl.convert_layout``
onto the store layout, ``tl.abs`` becomes ``gl.maximum(x, -x)``, and the ue8m0
group reduction uses a 2D ``[NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE]`` reshape (``out`` is 1D
here) reduced over ``axis=1``.
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
        "BLOCK_SIZE_N",       
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
    LANES: gl.constexpr = num_warps * 32
    L: gl.constexpr = gl.BlockedLayout(                        
        size_per_thread=[max(1, BLOCK_SIZE_N // LANES)], # div N over lanes, floor 1
        threads_per_warp=[32],
        warps_per_cta=[num_warps], # warps per thread block
        order=[0],
    )
    L_SCALE: gl.constexpr = gl.BlockedLayout(              
        size_per_thread=[max(1, NUM_N_Q_GROUPS // LANES)], # scale group over lanes, floor 1
        threads_per_warp=[32],
        warps_per_cta=[num_warps],
        order=[0],
    )

    # setup
    pid = gl.program_id(0)                                  # pid
    offs = gl.arange(0, BLOCK_SIZE_N, layout=L).to(gl.int64)  # offsets
    mask = offs < n_half                                    # mask

    # load gate and up via the gfx1250 TDM engine
    # Stage each contiguous half-row through LDS with an async TDM load, then read
    # it into registers on layout L. The descriptor shape is the true [n_half]
    # extent, so any BLOCK_SIZE_N overhang (when n_half isn't a power of two) is
    # hardware OOB zero-filled in LDS — matching the reference masked load's
    # other=0.0.
    row_base = pid * inp_stride_m
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    gate_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=inp_ptr + row_base,
        shape=[n_half],
        strides=[inp_stride_n],
        block_shape=[BLOCK_SIZE_N],
        layout=SH,
    )
    up_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=inp_ptr + row_base + n_half * inp_stride_n,
        shape=[n_half],
        strides=[inp_stride_n],
        block_shape=[BLOCK_SIZE_N],
        layout=SH,
    )
    gate_smem = gl.allocate_shared_memory(gate_desc.dtype, gate_desc.block_shape, SH)
    up_smem = gl.allocate_shared_memory(up_desc.dtype, up_desc.block_shape, SH)
    gl.amd.gfx1250.tdm.async_load(gate_desc, [0], gate_smem)
    gl.amd.gfx1250.tdm.async_load(up_desc, [0], up_smem)
    gl.amd.gfx1250.tdm.async_wait(0)

    gate = gate_smem.load(L).to(gl.float32)
    up = up_smem.load(L).to(gl.float32)

    # clamp
    if HAVE_SWIGLU_CLAMP:
        up = gl.clamp(up, -swiglu_limit, swiglu_limit)       # clamp up to [-lim, lim]
        gate = gl.minimum(gate, swiglu_limit)                # clamp gate to <= lim

    # act(gate) * up
    out = _apply_activation_from_str(gate, ACTIVATION) * up  # use triton helper for now

    # weights
    if HAVE_WEIGHTS:
        if WEIGHT_BROADCAST:
            w = gl.load(weights_ptr + pid * weights_stride_m).to(gl.float32)  # scalar applied to all out
            out = out * w                                
        else:
            w = gl.load(
                weights_ptr + pid * weights_stride_m + offs * weights_stride_n,
                mask=mask, other=0.0, cache_modifier=cache_modifier,
            ).to(gl.float32)
            out = out * w

    # TDM output store: one descriptor + LDS staging buffer for this output row.
    # shape=[n_half] bounds the store so any BLOCK_SIZE_N overhang is skipped by
    # the hardware, matching the reference masked store's mask=offs<n_half.
    out_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=out_ptr + pid * out_stride_m,
        shape=[n_half],
        strides=[out_stride_n],
        block_shape=[BLOCK_SIZE_N],
        layout=SH,
    )
    out_smem = gl.allocate_shared_memory(
        out_ptr.dtype.element_ty, out_desc.block_shape, SH
    )

    # group quant and store
    if HAS_QUANT:
        if SCALE_FMT == "ue8m0":
            # mxfp8, reduce over inner QUANT_BLOCK_SIZE axis.
            # make 2d
            out_2d = gl.reshape(out, [NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE]) # quant blocks
            # abs
            # below similar to triton
            abs_2d = gl.maximum(out_2d, -out_2d)
            max_val = gl.max(abs_2d, axis=1, keep_dims=True)
            dequant_scale = max_val / DTYPE_MAX
            # ROUND_UP to a power of two via the fp32 exponent field:
            # add 0x007FFFFF (round mantissa up) then mask to the exponent bits.
            dequant_scale_exp = (
                dequant_scale.to(gl.uint32, bitcast=True) + 0x007FFFFF
            ) & 0x7F800000
            dequant_scale_rounded = dequant_scale_exp.to(gl.float32, bitcast=True)
            quant_scale = gl.where(                              # reciprocal, guard 0
                dequant_scale_rounded == 0, 0.0, 1.0 / dequant_scale_rounded
            )
            quant_tensor = out_2d * quant_scale # scale into fp8 range
            out_q = gl.convert_layout(gl.reshape(quant_tensor, [BLOCK_SIZE_N]), L)
            scale_exp = (dequant_scale_exp >> 23).to(gl.uint8)
            block_scales = gl.convert_layout(gl.reshape(scale_exp, [NUM_N_Q_GROUPS]), L_SCALE)
        else:
            # fp8 quant redraft
            out_2d = gl.reshape(out, [NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE])
            abs_2d = gl.maximum(out_2d, -out_2d)
            max_val = gl.maximum(
                gl.max(abs_2d, axis=1, keep_dims=True), 1e-10
            )  # [NUM_N_Q_GROUPS, 1]
            scale_out = max_val / DTYPE_MAX  # dequant (block) scale
            quant_2d = gl.clamp(out_2d * (1.0 / scale_out), DTYPE_MIN, DTYPE_MAX)
            # reshape to 1D + convert_layout onto L / L_SCALE so they line up
            # with the offs / g_offs store offset layouts.
            out_q = gl.convert_layout(gl.reshape(quant_2d, [BLOCK_SIZE_N]), L)
            block_scales = gl.convert_layout(
                gl.reshape(scale_out, [NUM_N_Q_GROUPS]), L_SCALE
            )

        out_smem.store(out_q.to(out_ptr.dtype.element_ty))  # stage for TDM store

        num_bs = gl.cdiv(n_half, QUANT_BLOCK_SIZE) # valid scale groups
        g_offs = gl.arange(0, NUM_N_Q_GROUPS, layout=L_SCALE) # scale-group indices
        if SHUFFLE:
            # Preshuffled scale store: identical index arithmetic to the Triton
            # reference (rows padded to 256, block-cols to 8, tiled layout).
            bs_offs_0 = pid // 32          # row-tile of 32
            bs_offs_1 = pid % 32           # position within the 32-row tile
            bs_offs_2 = bs_offs_1 % 16       # sub-position within 16
            bs_offs_1 = bs_offs_1 // 16      # which half of the 32 (0/1)
            bs_offs_3 = g_offs // 8          # block-col tile of 8
            bs_offs_4 = g_offs % 8           # position within the 8-col tile
            bs_offs_5 = bs_offs_4 % 4        # sub-position within 4
            bs_offs_4 = bs_offs_4 // 4       # which half of the 8 (0/1)
            bs_offs = (                      # weave the sub-indices into the tiled offset
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
            # no shuffle
            gl.store(         
                scale_ptr + pid * scale_stride_m + g_offs * scale_stride_n,
                block_scales.to(scale_ptr.dtype.element_ty),
                mask=g_offs < num_bs,
            )
    else:
        # no quant
        out_smem.store(out.to(out_ptr.dtype.element_ty))  # stage for TDM store

    # TDM write-back of the output row: LDS -> global async store. The barrier
    # ensures every lane finished writing out_smem before the DMA reads it;
    # async_wait(0) drains the store before the program exits.
    gl.barrier()
    gl.amd.gfx1250.tdm.async_store(out_desc, [0], out_smem)
    gl.amd.gfx1250.tdm.async_wait(0)
