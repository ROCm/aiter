import torch
import triton
import triton.language as tl

# ============================================================================
# GLU-family forward kernels (SwiGLU, GEGLU, ReGLU)
# ============================================================================


@triton.jit
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
        act_gate = gate * tl.sigmoid(gate)
    elif ACT_TYPE == 1:  # geglu (tanh approx)
        SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
        COEFF: tl.constexpr = 0.044715
        inner = SQRT_2_OVER_PI * (gate + COEFF * gate * gate * gate)
        act_gate = 0.5 * gate * (1.0 + tl.extra.hip.libdevice.tanh(inner))
    elif ACT_TYPE == 2:  # reglu
        act_gate = tl.where(gate > 0, gate, 0.0)

    out = act_gate * up

    tl.store(
        a_ptr
        + offs_m[:, None].to(tl.int64) * stride_a_m
        + offs_i[None, :].to(tl.int64) * stride_a_i,
        out.to(a_ptr.dtype.element_ty),
        mask=m_mask[:, None] & i_mask[None, :],
    )


# ============================================================================
# GLU-family backward kernels
# ============================================================================


@triton.jit
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
        sig = tl.sigmoid(gate)
        act_gate = gate * sig
        d_up = da * act_gate
        d_gate = da * up * sig * (1.0 + gate * (1.0 - sig))
    elif ACT_TYPE == 1:  # geglu (tanh approx)
        SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
        COEFF: tl.constexpr = 0.044715
        inner = SQRT_2_OVER_PI * (gate + COEFF * gate * gate * gate)
        tanh_val = tl.extra.hip.libdevice.tanh(inner)
        act_gate = 0.5 * gate * (1.0 + tanh_val)
        d_up = da * act_gate
        dtanh = 1.0 - tanh_val * tanh_val
        dinner = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * gate * gate)
        d_gate = da * up * (0.5 * (1.0 + tanh_val) + 0.5 * gate * dtanh * dinner)
    elif ACT_TYPE == 2:  # reglu
        relu_mask = gate > 0
        act_gate = tl.where(relu_mask, gate, 0.0)
        d_up = da * act_gate
        d_gate = da * up * tl.where(relu_mask, 1.0, 0.0)

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


# ============================================================================
# Non-GLU forward kernels (GELU, ReLU, SiLU, ReLU²)
# ============================================================================


@triton.jit
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
        SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
        COEFF: tl.constexpr = 0.044715
        inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x)
        out = 0.5 * x * (1.0 + tl.extra.hip.libdevice.tanh(inner))
    elif ACT_TYPE == 4:  # relu
        out = tl.where(x > 0, x, 0.0)
    elif ACT_TYPE == 5:  # silu
        out = x * tl.sigmoid(x)
    elif ACT_TYPE == 6:  # relu_sq
        relu_x = tl.where(x > 0, x, 0.0)
        out = relu_x * relu_x

    tl.store(
        a_ptr
        + offs_m[:, None].to(tl.int64) * stride_a_m
        + offs_i[None, :].to(tl.int64) * stride_a_i,
        out.to(a_ptr.dtype.element_ty),
        mask=m_mask[:, None] & i_mask[None, :],
    )


@triton.jit
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
        SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
        COEFF: tl.constexpr = 0.044715
        inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x)
        tanh_val = tl.extra.hip.libdevice.tanh(inner)
        dtanh = 1.0 - tanh_val * tanh_val
        dinner = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x * x)
        dx = da * (0.5 * (1.0 + tanh_val) + 0.5 * x * dtanh * dinner)
    elif ACT_TYPE == 4:  # relu
        dx = da * tl.where(x > 0, 1.0, 0.0)
    elif ACT_TYPE == 5:  # silu
        sig = tl.sigmoid(x)
        dx = da * sig * (1.0 + x * (1.0 - sig))
    elif ACT_TYPE == 6:  # relu_sq
        relu_mask = x > 0
        dx = da * tl.where(relu_mask, 2.0 * x, 0.0)

    tl.store(
        dh_ptr
        + offs_m[:, None].to(tl.int64) * stride_dh_m
        + offs_i[None, :].to(tl.int64) * stride_dh_i,
        dx.to(dh_ptr.dtype.element_ty),
        mask=m_mask[:, None] & i_mask[None, :],
    )


# ============================================================================
# Dispatcher functions
# ============================================================================

_GLU_ACT_MAP = {"swiglu": 0, "geglu": 1, "reglu": 2}
_POINTWISE_ACT_MAP = {"gelu_tanh_approx": 3, "relu": 4, "silu": 5, "relu_sq": 6}


def _launch_grid(TK, I, BLOCK_M=32, BLOCK_I=None):
    if BLOCK_I is None:
        BLOCK_I = min(triton.next_power_of_2(I), 1024)
    return (triton.cdiv(TK, BLOCK_M), triton.cdiv(I, BLOCK_I)), BLOCK_M, BLOCK_I


def activation_fwd(
    h: torch.Tensor, I: int, activation_type: str, concat_layout: bool = False
) -> torch.Tensor:
    TK = h.shape[0]

    if activation_type in _GLU_ACT_MAP:
        a = torch.empty(TK, I, dtype=h.dtype, device=h.device)
        grid, BLOCK_M, BLOCK_I = _launch_grid(TK, I)
        _glu_fwd_kernel[grid](
            h,
            a,
            TK,
            I,
            h.stride(0),
            h.stride(1),
            a.stride(0),
            a.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_I=BLOCK_I,
            CONCAT_LAYOUT=concat_layout,
            ACT_TYPE=_GLU_ACT_MAP[activation_type],
        )
        return a
    elif activation_type in _POINTWISE_ACT_MAP:
        a = torch.empty(TK, I, dtype=h.dtype, device=h.device)
        grid, BLOCK_M, BLOCK_I = _launch_grid(TK, I)
        _pointwise_act_fwd_kernel[grid](
            h,
            a,
            TK,
            I,
            h.stride(0),
            h.stride(1),
            a.stride(0),
            a.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_I=BLOCK_I,
            ACT_TYPE=_POINTWISE_ACT_MAP[activation_type],
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
        grid, BLOCK_M, BLOCK_I = _launch_grid(TK, I)
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
            BLOCK_M=BLOCK_M,
            BLOCK_I=BLOCK_I,
            CONCAT_LAYOUT=concat_layout,
            ACT_TYPE=_GLU_ACT_MAP[activation_type],
        )
        return dh
    elif activation_type in _POINTWISE_ACT_MAP:
        dh = torch.empty_like(h)
        grid, BLOCK_M, BLOCK_I = _launch_grid(TK, I)
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
            BLOCK_M=BLOCK_M,
            BLOCK_I=BLOCK_I,
            ACT_TYPE=_POINTWISE_ACT_MAP[activation_type],
        )
        return dh
    else:
        raise NotImplementedError(f"activation_type={activation_type}")
