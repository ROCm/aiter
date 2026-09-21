import torch
import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.moe.activations import (
    gelu_tanh,
    gelu_tanh_grad,
    relu,
    relu_grad,
    relu_sq,
    relu_sq_grad,
    silu,
    silu_grad,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.sonicmoe_config_utils import (
    get_sonicmoe_kernel_config,
    split_launch_config,
)

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


_GLU_ACT_MAP = {"swiglu": 0, "geglu": 1, "reglu": 2}
_POINTWISE_ACT_MAP = {"gelu_tanh_approx": 3, "relu": 4, "silu": 5, "relu_sq": 6}


def _launch_config(TK, I):
    config = get_sonicmoe_kernel_config("activation_kernel")
    block_i = min(triton.next_power_of_2(I), config.pop("BLOCK_I_MAX"))
    block_m = config["BLOCK_M"]
    grid = (triton.cdiv(TK, block_m), triton.cdiv(I, block_i))
    constexprs, launch = split_launch_config(config)
    constexprs["BLOCK_I"] = block_i
    return grid, constexprs, launch


def activation_fwd(
    h: torch.Tensor, I: int, activation_type: str, concat_layout: bool = False
) -> torch.Tensor:
    TK = h.shape[0]

    if activation_type in _GLU_ACT_MAP:
        a = torch.empty(TK, I, dtype=h.dtype, device=h.device)
        grid, constexprs, launch = _launch_config(TK, I)
        _glu_fwd_kernel[grid](
            h,
            a,
            TK,
            I,
            h.stride(0),
            h.stride(1),
            a.stride(0),
            a.stride(1),
            CONCAT_LAYOUT=concat_layout,
            ACT_TYPE=_GLU_ACT_MAP[activation_type],
            **constexprs,
            **launch,
        )
        return a
    elif activation_type in _POINTWISE_ACT_MAP:
        a = torch.empty(TK, I, dtype=h.dtype, device=h.device)
        grid, constexprs, launch = _launch_config(TK, I)
        _pointwise_act_fwd_kernel[grid](
            h,
            a,
            TK,
            I,
            h.stride(0),
            h.stride(1),
            a.stride(0),
            a.stride(1),
            ACT_TYPE=_POINTWISE_ACT_MAP[activation_type],
            **constexprs,
            **launch,
        )
        return a
    else:
        raise NotImplementedError(f"activation_type={activation_type}")


def activation_bwd(
    h: torch.Tensor,
    da: torch.Tensor,
    I: int,
    activation_type: str,
    concat_layout: bool = False,
) -> torch.Tensor:
    TK = h.shape[0]

    if activation_type in _GLU_ACT_MAP:
        dh = torch.empty_like(h)
        grid, constexprs, launch = _launch_config(TK, I)
        _glu_bwd_kernel[grid](
            h,
            dh,
            da,
            TK,
            I,
            h.stride(0),
            h.stride(1),
            dh.stride(0),
            dh.stride(1),
            da.stride(0),
            da.stride(1),
            CONCAT_LAYOUT=concat_layout,
            ACT_TYPE=_GLU_ACT_MAP[activation_type],
            **constexprs,
            **launch,
        )
        return dh
    elif activation_type in _POINTWISE_ACT_MAP:
        dh = torch.empty_like(h)
        grid, constexprs, launch = _launch_config(TK, I)
        _pointwise_act_bwd_kernel[grid](
            h,
            dh,
            da,
            TK,
            I,
            h.stride(0),
            h.stride(1),
            dh.stride(0),
            dh.stride(1),
            da.stride(0),
            da.stride(1),
            ACT_TYPE=_POINTWISE_ACT_MAP[activation_type],
            **constexprs,
            **launch,
        )
        return dh
    else:
        raise NotImplementedError(f"activation_type={activation_type}")
