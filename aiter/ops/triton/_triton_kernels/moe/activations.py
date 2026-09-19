import triton
import triton.language as tl
from triton.language.extra.libdevice import fast_dividef


@triton.jit
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


@triton.jit
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
