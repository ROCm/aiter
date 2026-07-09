import triton
import triton.language as tl


@triton.jit
def clip(x, limit, clip_lower: tl.constexpr):
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


@triton.jit
def _swiglu(
    input, alpha, limit, ADD_RESIDUAL: tl.constexpr, INTERLEAVED: tl.constexpr = True
):
    """
    SwiGLU activation

    s = silu(gelu), then returns s * (linear + 1) if ADD_RESIDUAL else s * linear.
    if alpha=1.0, then this is the same as the SiLU activation.

    INTERLEAVED selects how gate/up are packed along the last axis of ``input``:
      True  (GUGU): interleaved pairs [g0,u0,g1,u1,...]  -> gelu=even, linear=odd.
      False (GGUU): contiguous halves [g0..g_{h-1}, u0..u_{h-1}] -> gelu=first
                    half, linear=second half. Only valid when the activation
                    block spans the full output row (both halves in one block).
    """
    half: tl.constexpr = input.shape[1] // 2
    if INTERLEAVED:
        gelu, linear = tl.split(tl.reshape(input, (input.shape[0], half, 2)))
    else:
        # Move the [gate|up] axis to the last dim so tl.split separates halves.
        gelu, linear = tl.split(
            tl.permute(tl.reshape(input, (input.shape[0], 2, half)), (0, 2, 1))
        )
    gelu = gelu.to(tl.float32)
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(tl.float32)
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)
    s = gelu / (1 + tl.exp2(-1.44269504089 * alpha * gelu))
    if ADD_RESIDUAL:
        return tl.fma(s, linear, s)  # s * (linear + 1)
    else:
        return s * linear
