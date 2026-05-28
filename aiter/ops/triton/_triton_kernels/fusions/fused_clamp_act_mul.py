# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused SwiGLU clamp + SiLU * up + optional token weights + optional per-row FP8 group quant (128).

Each program handles one row (token). ``inp`` is ``[M, 2 * N]`` with gate in the first
``N`` columns and up in the second ``N`` (same layout as ``torch.chunk(2, dim=-1)``).
Gate clamp matches DeepSeek-V4 reference: ``clamp(gate, max=limit)`` only; up uses
``clamp(up, min=-limit, max=limit)``. When ``HAS_QUANT`` is False the result is written
directly to ``out`` in its native dtype and no scales are produced.
"""

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.activation import _apply_activation_from_str

from aiter.ops.triton._triton_kernels.quant.fused_fp8_quant import _fp8_quant_op
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_fused_clamp_silu_mul_repr = make_kernel_repr(
    "_fused_clamp_silu_mul_kernel",
    [
        "BLOCK_SIZE_N",
        "QUANT_BLOCK_SIZE",
        "SCALE_FMT",
        "HAVE_WEIGHTS",
        "WEIGHT_BROADCAST",
        "HAVE_SWIGLU_CLAMP",
        "HAS_QUANT",
    ],
)


@triton.jit(repr=_fused_clamp_silu_mul_repr)
def _fused_clamp_silu_mul_kernel(
    inp_ptr,
    out_ptr,
    scale_ptr,
    weights_ptr,
    M,
    n_half,
    inp_stride_m,
    inp_stride_n,
    out_stride_m,
    out_stride_n,
    scale_stride_m,
    scale_stride_n,
    weights_stride_m,
    weights_stride_n,
    swiglu_limit,
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    SCALE_FMT: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
    HAVE_WEIGHTS: tl.constexpr,
    WEIGHT_BROADCAST: tl.constexpr,
    HAVE_SWIGLU_CLAMP: tl.constexpr,
    HAS_QUANT: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    m_pid = tl.program_id(0)
    n_offs = tl.arange(0, BLOCK_SIZE_N)
    mask = n_offs < n_half

    gate = tl.load(
        inp_ptr + m_pid * inp_stride_m + n_offs * inp_stride_n,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)
    up = tl.load(
        inp_ptr + m_pid * inp_stride_m + (n_half + n_offs) * inp_stride_n,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)

    if HAVE_SWIGLU_CLAMP:
        up = tl.clamp(up, -swiglu_limit, swiglu_limit)
        gate = tl.minimum(gate, swiglu_limit)

    out = _apply_activation_from_str(gate, ACTIVATION) * up

    if HAVE_WEIGHTS:
        if WEIGHT_BROADCAST:
            w = tl.load(weights_ptr + m_pid * weights_stride_m).to(tl.float32)
            out = out * w
        else:
            w = tl.load(
                weights_ptr + m_pid * weights_stride_m + n_offs * weights_stride_n,
                mask=mask,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            out = out * w

    if HAS_QUANT:
        if SCALE_FMT == "ue8m0":
            # Per-1×QUANT_BLOCK_SIZE MXFP8 emit: fp8 e4m3 values + uint8 ue8m0
            # biased-exponent scales. Mirrors the ue8m0 path used by moe_gemm_a8w4.
            NUM_QB: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE
            out_3d = tl.reshape(out, [1, NUM_QB, QUANT_BLOCK_SIZE])
            abs_3d = tl.abs(out_3d)
            max_val = tl.max(abs_3d, axis=2, keep_dims=True)
            dequant_scale = max_val / DTYPE_MAX
            # ROUND_UP via exponent: 2 ** ceil(log2(dequant_scale))
            dequant_scale_exp = (
                dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF
            ) & 0x7F800000
            dequant_scale_rounded = dequant_scale_exp.to(tl.float32, bitcast=True)
            quant_scale = tl.where(
                dequant_scale_rounded == 0, 0.0, 1.0 / dequant_scale_rounded
            )
            quant_tensor = out_3d * quant_scale
            quant_2d = tl.reshape(quant_tensor, [1, BLOCK_SIZE_N])
            out_q = tl.ravel(quant_2d)
            scale_exp = (dequant_scale_exp >> 23).to(tl.uint8)
            scale_exp_2d = tl.reshape(scale_exp, [1, NUM_QB])
            block_scales = tl.ravel(scale_exp_2d)
        else:
            out_q, block_scales = _fp8_quant_op(
                out, 1, BLOCK_SIZE_N, QUANT_BLOCK_SIZE, DTYPE_MAX, DTYPE_MIN
            )
            out_q = tl.ravel(out_q)
            block_scales = tl.ravel(block_scales)

        tl.store(
            out_ptr + m_pid * out_stride_m + n_offs * out_stride_n,
            out_q.to(out_ptr.dtype.element_ty),
            mask=mask,
        )

        num_bs = tl.cdiv(n_half, QUANT_BLOCK_SIZE)
        NUM_QB_S: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE
        g_offs = tl.arange(0, NUM_QB_S)
        tl.store(
            scale_ptr + m_pid * scale_stride_m + g_offs * scale_stride_n,
            block_scales.to(scale_ptr.dtype.element_ty),
            mask=g_offs < num_bs,
        )
    else:
        tl.store(
            out_ptr + m_pid * out_stride_m + n_offs * out_stride_n,
            out.to(out_ptr.dtype.element_ty),
            mask=mask,
        )
