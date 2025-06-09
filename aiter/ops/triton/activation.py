from .quant import _mxfp4_quant_op
from typing import Literal
import triton
import triton.language as tl
import torch


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


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


@tl.constexpr_function
def _get_activation_from_str(activation: str):
    mapping = {
        "gelu": _gelu,
        "gelu_tanh": _gelu_tanh,
        "silu": _silu,
    }
    return mapping[activation]


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

        x = _get_activation_from_str(ACTIVATION)(a) * b

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


def act_mul_and_mxfp4_quant(
    x: torch.Tensor,
    activation: Literal["silu", "gelu", "gelu_tanh"],
    scaling_mode: str = "even",
    shuffle: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the activation function and quantize the result to MX FP4 format.

    Args:
        x: The input tensor, typically fp16 or bf16.
        activation: activation function to apply before quantization.
            - It splits the features into two parts and applies the activation to the first part.
            - Then, it adds the results together before quantization.
            - Supports the following activations:
                - "silu"
                - "gelu"
                - "gelu_tanh"

        scaling_mode: The method to calculate MX block scaling.
            - "even" (default): `even_round` in `quark.torch.quantization.utils`.
            - etc.
        shuffle: Indicates whether to enable preshuffling of scales.
            - When enabled, scale dimensions (X, Y) are adjusted to be multiples of 8 and 256, respectively.
    Returns:
        A tuple of (x_fp4, blockscale_e8m0).
    """
    # Assume x is 2D-Tensor for now
    M, N = x.shape
    # Activation (N/2) and storing results in uint8 (N/2) results in a feature dimension of N/4
    assert N % 4 == 0

    # This is fixed by spec for MXFP4. Do not tune this.
    MXFP4_QUANT_BLOCK_SIZE = 32
    N_half = N // 2
    x_fp4 = torch.empty((M, N_half // 2), dtype=torch.uint8, device=x.device)
    scaleN_valid = triton.cdiv(N_half, MXFP4_QUANT_BLOCK_SIZE)
    # Setting scale M to be multiple of 256 and scale N to be multiple of 8
    if shuffle:
        scaleM = triton.cdiv(M, 256) * 256
        scaleN = triton.cdiv(scaleN_valid, 8) * 8
    else:
        scaleM = M
        scaleN = scaleN_valid

    blockscale_e8m0 = torch.empty(
        (scaleM, scaleN),
        dtype=torch.uint8,
        device=x.device,
    )

    # for large N values
    if M <= 32:
        NUM_ITER = 1
        BLOCK_SIZE_M = min(8, triton.next_power_of_2(M))
        BLOCK_SIZE_N = 128
        NUM_WARPS = 1 if BLOCK_SIZE_M < 4 else 4
        NUM_STAGES = 1
    else:
        NUM_ITER = 1
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 256
        NUM_WARPS = 4
        NUM_STAGES = 1

    # for small N values
    if N_half <= 1024:
        NUM_ITER = 1
        NUM_STAGES = 1
        NUM_WARPS = 4
        BLOCK_SIZE_N = min(256, triton.next_power_of_2(N_half))
        # BLOCK_SIZE_N needs to be multiple of 32
        BLOCK_SIZE_N = max(32, BLOCK_SIZE_N)
        BLOCK_SIZE_M = min(8, triton.next_power_of_2(N_half))

    # shuffle requires block sizes to be multiple of 32
    if shuffle:
        BLOCK_SIZE_M = triton.cdiv(BLOCK_SIZE_M, 32) * 32
        BLOCK_SIZE_N = triton.cdiv(BLOCK_SIZE_N, 32) * 32

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N_half, BLOCK_SIZE_N * NUM_ITER),
    )
    _act_mul_and_dynamic_mxfp4_quant_kernel[grid](
        x,
        x_fp4,
        blockscale_e8m0,
        *x.stride(),
        *x_fp4.stride(),
        *blockscale_e8m0.stride(),
        M=M,
        N=N_half,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        SCALING_MODE=0,
        ACTIVATION=activation,
        scaleN=scaleN_valid,
        scaleM_pad=scaleM,
        scaleN_pad=scaleN,
        SHUFFLE=shuffle,
        NUM_ITER=NUM_ITER,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_STAGES=NUM_STAGES,
        num_warps=NUM_WARPS,
        waves_per_eu=0,
        num_stages=1,
    )

    return x_fp4, blockscale_e8m0
