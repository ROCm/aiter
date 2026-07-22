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
group reduction uses a 2D ``[NUM_QB, QUANT_BLOCK_SIZE]`` reshape (``out`` is 1D
here) reduced over ``axis=1``.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._triton_kernels.activation import _apply_activation_from_str
from aiter.ops.triton._triton_kernels.quant.fused_fp8_quant import _fp8_quant_op
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
    NUM_QB: gl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE  # quant groups per row
    # 1D layouts
    LANES: gl.constexpr = num_warps * 32
    L: gl.constexpr = gl.BlockedLayout(                        
        size_per_thread=[max(1, BLOCK_SIZE_N // LANES)], # div N over lanes, floor 1
        threads_per_warp=[32],
        warps_per_cta=[num_warps], # warps per thread block
        order=[0],
    )
    L_SCALE: gl.constexpr = gl.BlockedLayout(              
        size_per_thread=[max(1, NUM_QB // LANES)], # scale group over lanes, floor 1
        threads_per_warp=[32],
        warps_per_cta=[num_warps],
        order=[0],
    )

    # setup
    m_pid = gl.program_id(0)                                  # pid
    n_offs = gl.arange(0, BLOCK_SIZE_N, layout=L).to(gl.int64)  # offsets
    mask = n_offs < n_half                                    # mask

    # load gate and up
    row_base = m_pid * inp_stride_m                           
    offs_gate = row_base + (n_half + n_offs) * inp_stride_n   
    gate = gl.load(                                      # load for now  
        inp_ptr + row_base + n_offs * inp_stride_n,
        mask=mask, other=0.0, cache_modifier=cache_modifier,
    ).to(gl.float32)                                        
    up = gl.load(                                             
        inp_ptr + offs_gate,
        mask=mask, other=0.0, cache_modifier=cache_modifier,
    ).to(gl.float32)

    # clamp
    if HAVE_SWIGLU_CLAMP:
        up = gl.clamp(up, -swiglu_limit, swiglu_limit)       # clamp up to [-lim, lim]
        gate = gl.minimum(gate, swiglu_limit)                # clamp gate to <= lim

    # act(gate) * up
    out = _apply_activation_from_str(gate, ACTIVATION) * up  # use triton helper for now

    # weights
    if HAVE_WEIGHTS:
        if WEIGHT_BROADCAST:
            w = gl.load(weights_ptr + m_pid * weights_stride_m).to(gl.float32)  # scalar applied to all out
            out = out * w                                
        else:
            w = gl.load(
                weights_ptr + m_pid * weights_stride_m + n_offs * weights_stride_n,
                mask=mask, other=0.0, cache_modifier=cache_modifier,
            ).to(gl.float32)
            out = out * w

    # group quant and store
    if HAS_QUANT:
        if SCALE_FMT == "ue8m0":
            # mxfp8, reduce over inner QUANT_BLOCK_SIZE axis.
            # make 2d
            out_2d = gl.reshape(out, [NUM_QB, QUANT_BLOCK_SIZE]) # quant blocks
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
            # Reshape to 1D + convert_layout onto L so the flat result matches the n_offs store layout.
            out_q = gl.convert_layout(gl.reshape(quant_tensor, [BLOCK_SIZE_N]), L)
            # ue8m0 scale = the biased exponent byte (bits 23..30) as uint8.
            scale_exp = (dequant_scale_exp >> 23).to(gl.uint8)
            block_scales = gl.convert_layout(gl.reshape(scale_exp, [NUM_QB]), L_SCALE)
        else:
            # fp8 quant
            out_q, block_scales = _fp8_quant_op(
                out, 1, BLOCK_SIZE_N, QUANT_BLOCK_SIZE, DTYPE_MAX, DTYPE_MIN
            ) # -> [1, NUM_QB, QUANT_BLOCK_SIZE] / [1, NUM_QB] 
            
            # reshape to 1D and convert_layouts onto L / L_SCALE so
            # they line up with the n_offs / g_offs store offset layouts.
            out_q = gl.convert_layout(gl.reshape(out_q, [BLOCK_SIZE_N]), L)
            block_scales = gl.convert_layout(gl.reshape(block_scales, [NUM_QB]), L_SCALE)

        gl.store( # write quantized values
            out_ptr + m_pid * out_stride_m + n_offs * out_stride_n,
            out_q.to(out_ptr.dtype.element_ty), mask=mask,
        )

        num_bs = gl.cdiv(n_half, QUANT_BLOCK_SIZE) # valid scale groups
        g_offs = gl.arange(0, NUM_QB, layout=L_SCALE) # scale-group indices
        if SHUFFLE:
            # Preshuffled scale store: identical index arithmetic to the Triton
            # reference (rows padded to 256, block-cols to 8, tiled layout).
            bs_offs_0 = m_pid // 32          # row-tile of 32
            bs_offs_1 = m_pid % 32           # position within the 32-row tile
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
                scale_ptr + m_pid * scale_stride_m + g_offs * scale_stride_n,
                block_scales.to(scale_ptr.dtype.element_ty),
                mask=g_offs < num_bs,
            )
    else:
        # no quant
        gl.store(          
            out_ptr + m_pid * out_stride_m + n_offs * out_stride_n,
            out.to(out_ptr.dtype.element_ty), mask=mask,
        )
