import triton
import triton.language as tl


@triton.jit
def _silu_exp2(x):
    return x / (1.0 + tl.exp2(-(x * 1.44269504089)))


@triton.jit
def _silu(x):
    return _silu_exp2(x)


@triton.jit
def _sigmoid_exp2(x):
    return 1.0 / (1.0 + tl.exp2(-(x * 1.44269504089)))


@triton.jit
def _sigmoid(x):
    return _sigmoid_exp2(x)


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _gelu(x):
    M_SQRT1_2 = 0.70710678118654752440
    ALPHA = M_SQRT1_2
    return 0.5 * x * (1.0 + tl.erf(x * ALPHA))


@triton.jit
def _gelu_tanh(x):
    M_SQRT2 = 1.41421356237309504880
    M_2_SQRTPI = 1.12837916709551257390
    BETA = M_SQRT2 * M_2_SQRTPI * 0.5
    KAPPA = 0.044715
    x_cube = x * x * x
    inner = BETA * (x + KAPPA * x_cube)
    return 0.5 * x * (1.0 + _tanh(inner))


@triton.jit
def _relu(x):
    return tl.maximum(0.0, x)


@triton.jit
def _relu6(x):
    return tl.minimum(tl.maximum(0.0, x), 6.0)


def _get_activation_from_str(activation: str):
    mapping = {
        "gelu": _gelu,
        "gelu_tanh": _gelu_tanh,
        "sigmoid": _sigmoid,
        "sigmoid_exp2": _sigmoid_exp2,
        "silu": _silu,
        "silu_exp2": _silu_exp2,
        "relu": _relu,
        "relu6": _relu6,
    }
    return mapping[activation]


@triton.jit
def _apply_activation_from_str(x, activation: tl.constexpr):
    if activation == "gelu":
        return _gelu(x)
    elif activation == "gelu_tanh":
        return _gelu_tanh(x)
    elif activation == "sigmoid":
        return _sigmoid(x)
    elif activation == "sigmoid_exp2":
        return _sigmoid_exp2(x)
    elif activation == "silu":
        return _silu(x)
    elif activation == "silu_exp2":
        return _silu_exp2(x)
    elif activation == "relu":
        return _relu(x)
    elif activation == "relu6":
        return _relu6(x)
    else:
        return x  # No activation if it is not recognized
