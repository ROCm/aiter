# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from triton.language.extra.libdevice import fast_dividef

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_clip_repr = make_kernel_repr("moe_activation_clip", ["clip_lower"])
_swiglu_repr = make_kernel_repr("moe_swiglu", ["ADD_RESIDUAL"])
_silu_repr = make_kernel_repr("moe_silu", [])
_silu_grad_repr = make_kernel_repr("moe_silu_grad", [])
_gelu_tanh_repr = make_kernel_repr("moe_gelu_tanh", [])
_gelu_tanh_grad_repr = make_kernel_repr("moe_gelu_tanh_grad", [])
_relu_repr = make_kernel_repr("moe_relu", [])
_relu_grad_repr = make_kernel_repr("moe_relu_grad", [])
_relu_sq_repr = make_kernel_repr("moe_relu_sq", [])
_relu_sq_grad_repr = make_kernel_repr("moe_relu_sq_grad", [])


@triton.jit(repr=_clip_repr)
def clip(x, limit, clip_lower: tl.constexpr):
    # Keep the upper clamp scalar to avoid the register-pressure regression from
    # https://github.com/llvm/llvm-project/commit/86aaf7b55ef5bfe4f96c8d58ce6addfe5e85967b
    # because AMDGPU later scalarizes the packed minimum during lowering.
    res = tl.inline_asm_elementwise(
        "v_min_f32 $0, $1, $2",
        "=v,v,v",
        [x, limit],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


@triton.jit(repr=_swiglu_repr)
def _swiglu(input, alpha, limit, ADD_RESIDUAL: tl.constexpr):
    """
    SwiGLU activation

    s = silu(gelu), then returns s * (linear + 1) if ADD_RESIDUAL else s * linear.
    if alpha=1.0, then this is the same as the SiLU activation.
    """
    gelu, linear = tl.split(tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    gelu = gelu.to(tl.float32)
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(tl.float32)
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)
    s = fast_dividef(gelu, 1 + tl.exp2(-1.44269504089 * alpha * gelu))
    if ADD_RESIDUAL:
        return tl.fma(s, linear, s)  # s * (linear + 1)
    else:
        return s * linear


@triton.jit(repr=_silu_repr)
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit(repr=_silu_grad_repr)
def silu_grad(x):
    sigmoid = tl.sigmoid(x)
    return sigmoid * (1.0 + x * (1.0 - sigmoid))


@triton.jit(repr=_gelu_tanh_repr)
def gelu_tanh(x):
    sqrt_2_over_pi: tl.constexpr = 0.7978845608028654
    coeff: tl.constexpr = 0.044715
    inner = sqrt_2_over_pi * (x + coeff * x * x * x)
    return 0.5 * x * (1.0 + tl.extra.hip.libdevice.tanh(inner))


@triton.jit(repr=_gelu_tanh_grad_repr)
def gelu_tanh_grad(x):
    sqrt_2_over_pi: tl.constexpr = 0.7978845608028654
    coeff: tl.constexpr = 0.044715
    inner = sqrt_2_over_pi * (x + coeff * x * x * x)
    tanh_value = tl.extra.hip.libdevice.tanh(inner)
    derivative = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x)
    return (
        0.5 * (1.0 + tanh_value)
        + 0.5 * x * (1.0 - tanh_value * tanh_value) * derivative
    )


@triton.jit(repr=_relu_repr)
def relu(x):
    return tl.where(x > 0, x, 0.0)


@triton.jit(repr=_relu_grad_repr)
def relu_grad(x):
    return tl.where(x > 0, 1.0, 0.0)


@triton.jit(repr=_relu_sq_repr)
def relu_sq(x):
    value = relu(x)
    return value * value


@triton.jit(repr=_relu_sq_grad_repr)
def relu_sq_grad(x):
    return tl.where(x > 0, 2.0 * x, 0.0)


_glu_fwd_repr = make_kernel_repr(
    "sonicmoe_glu_fwd", ["I", "BLOCK_M", "BLOCK_I", "CONCAT_LAYOUT", "ACT_TYPE"]
)
_glu_bwd_repr = make_kernel_repr(
    "sonicmoe_glu_bwd", ["I", "BLOCK_M", "BLOCK_I", "CONCAT_LAYOUT", "ACT_TYPE"]
)
_pointwise_act_fwd_repr = make_kernel_repr(
    "sonicmoe_pointwise_act_fwd", ["I", "BLOCK_M", "BLOCK_I", "ACT_TYPE"]
)
_pointwise_act_bwd_repr = make_kernel_repr(
    "sonicmoe_pointwise_act_bwd", ["I", "BLOCK_M", "BLOCK_I", "ACT_TYPE"]
)


@triton.jit(repr=_glu_fwd_repr)
def _glu_fwd_kernel(
    h_ptr,
    a_ptr,
    TK,
    I: tl.constexpr,
    stride_h_m,
    stride_h_i,
    stride_a_m,
    stride_a_i,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    CONCAT_LAYOUT: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    m_mask = offs_m < TK
    i_mask = offs_i < I

    if CONCAT_LAYOUT:
        gate_offs = offs_i
        up_offs = offs_i + I
    else:
        gate_offs = offs_i * 2
        up_offs = offs_i * 2 + 1

    gate = tl.load(
        h_ptr
        + offs_m[:, None].to(tl.int64) * stride_h_m
        + gate_offs[None, :].to(tl.int64) * stride_h_i,
        mask=m_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        h_ptr
        + offs_m[:, None].to(tl.int64) * stride_h_m
        + up_offs[None, :].to(tl.int64) * stride_h_i,
        mask=m_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    if ACT_TYPE == 0:  # swiglu
        act_gate = silu(gate)
    elif ACT_TYPE == 1:  # geglu (tanh approx)
        act_gate = gelu_tanh(gate)
    elif ACT_TYPE == 2:  # reglu
        act_gate = relu(gate)

    out = act_gate * up

    tl.store(
        a_ptr
        + offs_m[:, None].to(tl.int64) * stride_a_m
        + offs_i[None, :].to(tl.int64) * stride_a_i,
        out.to(a_ptr.dtype.element_ty),
        mask=m_mask[:, None] & i_mask[None, :],
    )


@triton.jit(repr=_glu_bwd_repr)
def _glu_bwd_kernel(
    h_ptr,
    dh_ptr,
    da_ptr,
    TK,
    I: tl.constexpr,
    stride_h_m,
    stride_h_i,
    stride_dh_m,
    stride_dh_i,
    stride_da_m,
    stride_da_i,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    CONCAT_LAYOUT: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    m_mask = offs_m < TK
    i_mask = offs_i < I

    if CONCAT_LAYOUT:
        gate_offs = offs_i
        up_offs = offs_i + I
    else:
        gate_offs = offs_i * 2
        up_offs = offs_i * 2 + 1

    gate = tl.load(
        h_ptr
        + offs_m[:, None].to(tl.int64) * stride_h_m
        + gate_offs[None, :].to(tl.int64) * stride_h_i,
        mask=m_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        h_ptr
        + offs_m[:, None].to(tl.int64) * stride_h_m
        + up_offs[None, :].to(tl.int64) * stride_h_i,
        mask=m_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    da = tl.load(
        da_ptr
        + offs_m[:, None].to(tl.int64) * stride_da_m
        + offs_i[None, :].to(tl.int64) * stride_da_i,
        mask=m_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    if ACT_TYPE == 0:  # swiglu
        d_up = da * silu(gate)
        d_gate = da * up * silu_grad(gate)
    elif ACT_TYPE == 1:  # geglu (tanh approx)
        d_up = da * gelu_tanh(gate)
        d_gate = da * up * gelu_tanh_grad(gate)
    elif ACT_TYPE == 2:  # reglu
        d_up = da * relu(gate)
        d_gate = da * up * relu_grad(gate)

    tl.store(
        dh_ptr
        + offs_m[:, None].to(tl.int64) * stride_dh_m
        + gate_offs[None, :].to(tl.int64) * stride_dh_i,
        d_gate.to(dh_ptr.dtype.element_ty),
        mask=m_mask[:, None] & i_mask[None, :],
    )
    tl.store(
        dh_ptr
        + offs_m[:, None].to(tl.int64) * stride_dh_m
        + up_offs[None, :].to(tl.int64) * stride_dh_i,
        d_up.to(dh_ptr.dtype.element_ty),
        mask=m_mask[:, None] & i_mask[None, :],
    )


@triton.jit(repr=_pointwise_act_fwd_repr)
def _pointwise_act_fwd_kernel(
    h_ptr,
    a_ptr,
    TK,
    I: tl.constexpr,
    stride_h_m,
    stride_h_i,
    stride_a_m,
    stride_a_i,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    m_mask = offs_m < TK
    i_mask = offs_i < I

    x = tl.load(
        h_ptr
        + offs_m[:, None].to(tl.int64) * stride_h_m
        + offs_i[None, :].to(tl.int64) * stride_h_i,
        mask=m_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    if ACT_TYPE == 3:  # gelu (tanh approx)
        out = gelu_tanh(x)
    elif ACT_TYPE == 4:  # relu
        out = relu(x)
    elif ACT_TYPE == 5:  # silu
        out = silu(x)
    elif ACT_TYPE == 6:  # relu_sq
        out = relu_sq(x)

    tl.store(
        a_ptr
        + offs_m[:, None].to(tl.int64) * stride_a_m
        + offs_i[None, :].to(tl.int64) * stride_a_i,
        out.to(a_ptr.dtype.element_ty),
        mask=m_mask[:, None] & i_mask[None, :],
    )


@triton.jit(repr=_pointwise_act_bwd_repr)
def _pointwise_act_bwd_kernel(
    h_ptr,
    dh_ptr,
    da_ptr,
    TK,
    I: tl.constexpr,
    stride_h_m,
    stride_h_i,
    stride_dh_m,
    stride_dh_i,
    stride_da_m,
    stride_da_i,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    m_mask = offs_m < TK
    i_mask = offs_i < I

    x = tl.load(
        h_ptr
        + offs_m[:, None].to(tl.int64) * stride_h_m
        + offs_i[None, :].to(tl.int64) * stride_h_i,
        mask=m_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    da = tl.load(
        da_ptr
        + offs_m[:, None].to(tl.int64) * stride_da_m
        + offs_i[None, :].to(tl.int64) * stride_da_i,
        mask=m_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    if ACT_TYPE == 3:  # gelu (tanh approx)
        dx = da * gelu_tanh_grad(x)
    elif ACT_TYPE == 4:  # relu
        dx = da * relu_grad(x)
    elif ACT_TYPE == 5:  # silu
        dx = da * silu_grad(x)
    elif ACT_TYPE == 6:  # relu_sq
        dx = da * relu_sq_grad(x)

    tl.store(
        dh_ptr
        + offs_m[:, None].to(tl.int64) * stride_dh_m
        + offs_i[None, :].to(tl.int64) * stride_dh_i,
        dx.to(dh_ptr.dtype.element_ty),
        mask=m_mask[:, None] & i_mask[None, :],
    )
