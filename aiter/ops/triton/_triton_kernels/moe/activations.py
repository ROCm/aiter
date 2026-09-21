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
