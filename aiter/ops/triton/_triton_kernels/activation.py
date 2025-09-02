from .quant import _mxfp4_quant_op
import triton
import triton.language as tl


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _silu_exp2(x):
    return x / (1.0 + tl.exp2(-(x * 1.44269504089)))


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


def _get_activation_from_str(activation: str):
    mapping = {
        "gelu": _gelu,
        "gelu_tanh": _gelu_tanh,
        "silu": _silu,
        "silu_exp2": _silu_exp2,
        "relu": _relu,
    }
    return mapping[activation]


@triton.jit
def _apply_activation_from_str(x, activation: tl.constexpr):
    if activation == "gelu":
        return _gelu(x)
    elif activation == "gelu_tanh":
        return _gelu_tanh(x)
    elif activation == "silu":
        return _silu(x)
    elif activation == "silu_exp2":
        return _silu_exp2(x)
    elif activation == "relu":
        return _relu(x)
    else:
        return x  # No activation if it is not recognized


@triton.heuristics(
    {
        "EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N"] % (args["BLOCK_SIZE_N"] * args["NUM_ITER"]) == 0,
    }
)
@triton.jit
def _act_mul_and_dynamic_mxfp4_quant_kernel(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp4_m_in,
    stride_x_fp4_n_in,
    stride_bs_m_in,
    stride_bs_n_in,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_ITER: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    EVEN_M_N: tl.constexpr,
    SCALING_MODE: tl.constexpr,
    ACTIVATION: tl.constexpr,
    scaleN: tl.constexpr,
    scaleM_pad: tl.constexpr,
    scaleN_pad: tl.constexpr,
    SHUFFLE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    start_n = tl.program_id(1) * NUM_ITER
    # cast strides to int64, in case M*N > max int32
    stride_x_m = tl.cast(stride_x_m_in, tl.int64)
    stride_x_n = tl.cast(stride_x_n_in, tl.int64)
    stride_x_fp4_m = tl.cast(stride_x_fp4_m_in, tl.int64)
    stride_x_fp4_n = tl.cast(stride_x_fp4_n_in, tl.int64)
    stride_bs_m = tl.cast(stride_bs_m_in, tl.int64)
    stride_bs_n = tl.cast(stride_bs_n_in, tl.int64)

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE

    for pid_n in tl.range(start_n, min(start_n + NUM_ITER, N), num_stages=NUM_STAGES):
        x_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        x_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n

        if EVEN_M_N:
            a = tl.load(x_ptr + x_offs, cache_modifier=".cg").to(tl.float32)
            b = tl.load(x_ptr + x_offs + stride_x_n * N, cache_modifier=".cg").to(
                tl.float32
            )
        else:
            x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
            a = tl.load(x_ptr + x_offs, mask=x_mask, cache_modifier=".cg").to(
                tl.float32
            )
            # a and b can share the same mask
            b = tl.load(
                x_ptr + x_offs + stride_x_n * N, mask=x_mask, cache_modifier=".cg"
            ).to(tl.float32)

        x = _apply_activation_from_str(a, ACTIVATION) * b

        out_tensor, bs_e8m0 = _mxfp4_quant_op(
            x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
        )

        out_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        out_offs_n = pid_n * BLOCK_SIZE_N // 2 + tl.arange(0, BLOCK_SIZE_N // 2)
        out_offs = (
            out_offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n
        )

        if EVEN_M_N:
            tl.store(x_fp4_ptr + out_offs, out_tensor)
        else:
            out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
            tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

        bs_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        bs_offs_n = pid_n * NUM_QUANT_BLOCKS + tl.arange(0, NUM_QUANT_BLOCKS)
        if SHUFFLE:
            bs_offs_0 = bs_offs_m[:, None] // 32
            bs_offs_1 = bs_offs_m[:, None] % 32
            bs_offs_2 = bs_offs_1 % 16
            bs_offs_1 = bs_offs_1 // 16
            bs_offs_3 = bs_offs_n[None, :] // 8
            bs_offs_4 = bs_offs_n[None, :] % 8
            bs_offs_5 = bs_offs_4 % 4
            bs_offs_4 = bs_offs_4 // 4
            bs_offs = (
                bs_offs_1
                + bs_offs_4 * 2
                + bs_offs_2 * 2 * 2
                + bs_offs_5 * 2 * 2 * 16
                + bs_offs_3 * 2 * 2 * 16 * 4
                + bs_offs_0 * 2 * 16 * scaleN
            )
            bs_mask1 = (bs_offs_m < M)[:, None] & (bs_offs_n < scaleN)[None, :]
            bs_mask = (bs_offs_m < scaleM_pad)[:, None] & (bs_offs_n < scaleN_pad)[
                None, :
            ]
            bs_e8m0 = tl.where(bs_mask1, bs_e8m0, 127)
        else:
            bs_offs = (
                bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
            )
            bs_mask = (bs_offs_m < M)[:, None] & (bs_offs_n < scaleN)[None, :]
        if EVEN_M_N:
            tl.store(bs_ptr + bs_offs, bs_e8m0)
        else:

            tl.store(
                bs_ptr + bs_offs,
                bs_e8m0,
                mask=bs_mask,
            )
