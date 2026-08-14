import triton
import triton.language as tl


@triton.jit
def clip(x, limit, clip_lower: tl.constexpr):
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


_LOG2E = tl.constexpr(1.44269504089)


@triton.jit
def _swiglu_pair(gate, up, alpha, limit, ADD_RESIDUAL: tl.constexpr):
    """Clamped GPT-OSS / MiniMax swiglu on separated gate and up tensors."""
    gate = gate.to(tl.float32)
    up = up.to(tl.float32)
    if limit is not None:
        gate = clip(gate, limit, clip_lower=False)
        up = clip(up, limit, clip_lower=True)
    s = gate / (1 + tl.exp2(-_LOG2E * alpha * gate))
    if ADD_RESIDUAL:
        return tl.fma(s, up, s)  # s * (up + 1)
    return s * up


@triton.jit
def _swiglu_separated(input, alpha, limit, ADD_RESIDUAL: tl.constexpr):
    """
    SwiGLU on non-interleaved gate|up layout: ``[..., 2 * d]`` with gate in the
    first ``d`` columns and up in the second ``d`` columns.
    """
    half = input.shape[1] // 2
    gate = input[:, :half]
    up = input[:, half:]
    return _swiglu_pair(gate, up, alpha, limit, ADD_RESIDUAL=ADD_RESIDUAL)


@triton.jit
def _swiglu(input, alpha, limit, ADD_RESIDUAL: tl.constexpr):
    """
    SwiGLU activation on interleaved gate/up layout.

    s = gate * sigmoid(alpha * gate), then returns s * (up + 1) if ADD_RESIDUAL
    else s * up. When alpha=1.0 this matches plain SiLU on the gate half.
    """
    gate, up = tl.split(tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    return _swiglu_pair(gate, up, alpha, limit, ADD_RESIDUAL=ADD_RESIDUAL)
