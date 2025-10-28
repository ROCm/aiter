from typing import Literal
import triton
import triton.language as tl
import torch
import aiter

fp8_dtype = aiter.dtypes.fp8
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._triton_kernels.activation import (
    _act_mul_and_dynamic_mxfp4_quant_kernel,
    _act_mul_and_dynamic_fp8_group_quant_kernel,
)

_LOGGER = AiterTritonLogger()


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
    _LOGGER.info(f"ACT_MUL_MXFP4_QUANT: x={tuple(x.shape)} activation={activation}")
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
        blockscale_e8m0 = torch.empty(
            (scaleM, scaleN),
            dtype=torch.uint8,
            device=x.device,
        )
    else:
        scaleM = M
        scaleN = scaleN_valid
        blockscale_e8m0 = torch.empty(
            (scaleN, scaleM),
            dtype=torch.uint8,
            device=x.device,
        ).T

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


def act_mul_and_fp8_group_quant(
    x: torch.Tensor,
    activation: Literal["silu", "gelu", "gelu_tanh"],
    group_size,
    dtype_quant=fp8_dtype,
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
    _LOGGER.info(f"ACT_MUL_FP8_GROUP_QUANT: x={tuple(x.shape)} activation={activation}")
    # Assume x is 2D-Tensor for now
    M, N = x.shape
    assert N % 2 == 0

    N_half = N // 2
    scaleN = triton.cdiv(N, group_size)
    x_fp8 = torch.empty((M, N_half), dtype=dtype_quant, device=x.device)
    out_bs = torch.empty(
        (M, triton.cdiv(N_half, group_size)), dtype=torch.float32, device=x.device
    )

    DTYPE_MAX = (
        torch.finfo(x_fp8.dtype).max
        if torch.is_floating_point(x_fp8)
        else torch.iinfo(x_fp8.dtype).max
    )
    BLOCK_SIZE_N = group_size

    grid = (
        M,
        triton.cdiv(N_half, BLOCK_SIZE_N),
    )
    _act_mul_and_dynamic_fp8_group_quant_kernel[grid](
        x,
        x_fp8,
        out_bs,
        *x.stride(),
        *x_fp8.stride(),
        *out_bs.stride(),
        N=N_half,
        ACTIVATION=activation,
        scaleN=scaleN,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        QUANT_BLOCK_SIZE=group_size,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        # num_warps=NUM_WARPS,
        # waves_per_eu=0,
        # num_stages=1,
    )

    return x_fp8, out_bs
