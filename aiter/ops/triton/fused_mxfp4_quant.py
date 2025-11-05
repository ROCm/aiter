from typing import Literal
import torch
import triton
import triton.language as tl
from typing import Optional

from aiter.ops.triton._triton_kernels.fused_mxfp4_quant import (
    _rmsmorm_op,
    _fused_rms_mxfp4_quant_kernel,
    _fused_flatten_mxfp4_quant,
    _fused_reduce_act_mul_and_dynamic_mxfp4_quant_kernel,
)
from aiter.ops.triton._triton_kernels.activation import (
    _get_activation_from_str,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_rms_mxfp4_quant(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: float = 0.0,
    res1: Optional[torch.Tensor] = None,
    shuffle: Optional[bool] = False,
    scale_shuffle_padding: Optional[bool] = False,
):
    """
    This op contains several steps:
        1. if res1 is not None, x1 = x1 + res1, and store x1 to out_res1
        2. perform RMS norm along the last dimenion for x1
        3. if x2 is not None, perform RMS norm along the last dimenion for x2
        4. perform mxfp4 quantization for x1 only

    Key parameters:
    - x: Matrix X with shape (M, N1, N2).

    Returns:
    - out1_fp4: The output matrix with shape (M, N1 // 2).
    - out1_bs: The output matrix with shape (M, cdiv(N1, MXFP4_QUANT_BLOCK_SIZE)).
    - out2: The output matrix with shape (M, N2).
    - out_res1: The output matrix with shape (M, N1).

        if both x2 and res1 provided, return (out1_fp4, out1_bs), out2, out_res1
        if x2 provided, return (out1_fp4, out1_bs), out2
        if res1 provided, return (out1_fp4, out1_bs), out_res1
        if both x2 and res1 not provided, return (out1_fp4, out1_bs)
    """
    _LOGGER.info(f"FUSED_RMS_MXFP4_QUANT: inp1={tuple(x1.shape)}")

    MXFP4_QUANT_BLOCK_SIZE = 32
    M, N1 = x1.shape
    BLOCK_SIZE_N = max(triton.next_power_of_2(N1), MXFP4_QUANT_BLOCK_SIZE)
    BLOCK_SIZE_N2 = 1
    if x2 is not None:
        N2 = x2.shape[1]
        BLOCK_SIZE_N2 = triton.next_power_of_2(N2)
    else:
        N2 = 0
    # as we merge 2 fp4s to 1 uint8
    assert N1 % 2 == 0
    BLOCK_SIZE_M = 1
    # BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = max(BLOCK_SIZE_N, MXFP4_QUANT_BLOCK_SIZE)
    out1_fp4 = torch.empty((M, N1 // 2), dtype=torch.uint8, device=x1.device)
    SCALE_N_valid = triton.cdiv(N1, MXFP4_QUANT_BLOCK_SIZE)
    use_scale_shuffle_padding = shuffle or scale_shuffle_padding
    if use_scale_shuffle_padding:
        SCALE_M = triton.cdiv(M, 256) * 256
        SCALE_N = triton.cdiv(SCALE_N_valid, 8) * 8
        # BLOCK_SIZE_M = triton.cdiv(BLOCK_SIZE_M, 32) * 32
        BLOCK_SIZE_N = triton.cdiv(BLOCK_SIZE_N, 32) * 32
    else:
        SCALE_M = M
        SCALE_N = SCALE_N_valid
    out1_bs = torch.empty(
        (SCALE_M, SCALE_N),
        dtype=torch.uint8,
        device=x1.device,
    )

    out_res1 = None
    res1_stride_m = 0
    out_res1_stride_m = 0
    if res1 is not None:
        out_res1 = torch.empty((M, N1), dtype=x1.dtype, device=x1.device)
        res1_stride_m = res1.stride(0)
        out_res1_stride_m = out_res1.stride(0)

    out2 = None
    out2_stride_m = 0
    x2_stride_m = 0
    if x2 is not None:
        out2 = torch.empty((M, N2), dtype=x1.dtype, device=x1.device)
        x2_stride_m = x2.stride(0)
        out2_stride_m = out2.stride(0)

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * (2 if (x2 is not None) else 1),)
    _fused_rms_mxfp4_quant_kernel[grid](
        x1,
        x1_weight,
        x2,
        x2_weight,
        res1,
        out1_fp4,
        out1_bs,
        out2,
        out_res1,
        x1_epsilon,
        x2_epsilon,
        M,
        N1,
        N2,
        x1.stride(0),
        x2_stride_m,
        res1_stride_m,
        out1_fp4.stride(0),
        *out1_bs.stride(),
        out2_stride_m,
        out_res1_stride_m,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        HAS_SECOND_INPUT=(x2 is not None),
        FIRST_INPUT_RES=(res1 is not None),
        SCALE_N=SCALE_N_valid,
        SCALE_M_PAD=(SCALE_M if use_scale_shuffle_padding else 1),
        SCALE_N_PAD=SCALE_N,
        SHUFFLE=shuffle,
        SHUFFLE_PAD=use_scale_shuffle_padding,
    )

    return (out1_fp4, out1_bs), out2, out_res1


def fused_flatten_mxfp4_quant(
    x: torch.Tensor,
):
    """
    Flatten the last two dimension of x and perform mxfp4 quantization along the last dimension

    Key parameters:
    - x: Matrix X with shape (M, N1, N2).

    Returns:
    - out: The output matrix with shape (M, (N1 * N2) // 2).
    - out_block_scales: The output matrix with shape (M, cdiv(N1 * N2, MXFP4_QUANT_BLOCK_SIZE)).
    """
    _LOGGER.info(f"FUSED_FLATTEN_MXFP4_QUANT: x={tuple(x.shape)}")
    M, N1, N2 = x.shape

    MXFP4_QUANT_BLOCK_SIZE = 32
    BLOCK_SIZE_N2 = max(triton.next_power_of_2(N2), MXFP4_QUANT_BLOCK_SIZE)
    N = N1 * N2
    out = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    out_block_scales = torch.empty(
        (triton.cdiv(N, MXFP4_QUANT_BLOCK_SIZE), M),
        dtype=torch.uint8,
        device=x.device,
    ).T

    grid = (
        M,
        N1,
    )
    _fused_flatten_mxfp4_quant[grid](
        x,
        out,
        out_block_scales,
        *x.stride(),
        *out.stride(),
        *out_block_scales.stride(),
        N2,
        BLOCK_SIZE_N2,
        MXFP4_QUANT_BLOCK_SIZE,
    )

    return out, out_block_scales


def fused_reduce_act_mul_and_mxfp4_quant(
    x: torch.Tensor,
    activation: Literal["silu", "gelu", "gelu_tanh"],
    x2: Optional[torch.Tensor] = None,
    scaling_mode: str = "even",
    shuffle: bool = False,
    scale_shuffle_padding: bool = False,
    dtype: Optional[float] = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply reduction along the first dimension and apply the activation function + per-token group quantization to MX FP4 format.
    If x2 is provided, the only reduction along the first dimension is applied to x2

    Args:
        if x is 3-dim,
            x: (SPK, M, 2*N1), dtype = fp32.
            x2: (SPK, M, 2*N1), dtype = fp32.

        if x is 2-dim,
            x: (M, 2*N1), dtype = fp16 or bf16.
            x2 must be None
            the kernel is essentially identical to aiter.ops.triton.activation.act_mul_and_mxfp4_group_quant

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
        tuple: (y, y_scale), y2
            if shuffle or scale_shuffle_padding:
                y: (M_pad, N1_pad), dtype = uint8
                y_scale: (M_pad, N1_pad), dtype = uint8
                y2: (M, N2), dtype = dtype

                where M_pad = cdiv(M, 256) * 256
                      N1_pad = cdiv(cdiv(N1, MXFP4_QUANT_BLOCK_SIZE), 8) * 8
            else:
                y: (M, N1), dtype = uint8
                y_scale: (M, cdiv(N1, MXFP4_QUANT_BLOCK_SIZE)), dtype = uint8
                y2: (M, N2), dtype = dtype

        A tuple of (y, y_scale).
    """
    _LOGGER.info(
        f"ACT_MUL_MXFP4_QUANT: x={tuple(x.shape)} activation={activation} shuffle={shuffle}"
    )

    assert (
        x.dim() == 2 or x.dim() == 3
    ), "The number of dimentions for x should be 2 or 3"
    X_HAS_SPLITK = False
    x_num_splitk = 1
    N2 = 1
    y2 = None
    if x.dim() == 3:
        x_num_splitk, M, N1 = x.shape
        x_num_splitk, _, N2 = x2.shape
        assert (
            x.shape[0] == x2.shape[0] and x.shape[1] == x2.shape[1]
        ), "The first two dimensions should be identical between x and x2"
        assert (
            x_num_splitk > 1
        ), "x.shape[0] should be larger then 1 in x.dim() == 3 cases"
        X_HAS_SPLITK = True
        y2 = torch.empty((M, N2), dtype=dtype, device=x2.device)
    else:
        M, N1 = x.shape
    # Activation (N/2) and storing results in uint8 (N/2) results in a feature dimension of N/4
    assert (
        N1 % 4 == 0
    ), "The last dimension for x1 should be multiple of 4 for acitvation, multiplication and mxfp4 quantization"

    MXFP4_QUANT_BLOCK_SIZE = 32
    N_half = N1 // 2
    y = torch.empty((M, N_half // 2), dtype=torch.uint8, device=x.device)
    scaleN_valid = triton.cdiv(N_half, MXFP4_QUANT_BLOCK_SIZE)
    # Setting scale M to be multiple of 256 and scale N to be multiple of 8
    use_scale_shuffle_padding = shuffle or scale_shuffle_padding
    if use_scale_shuffle_padding:
        scaleM = triton.cdiv(M, 256) * 256
        scaleN = triton.cdiv(scaleN_valid, 8) * 8
    else:
        scaleM = M
        scaleN = scaleN_valid
    y_scale = torch.empty(
        (scaleM, scaleN),
        dtype=torch.uint8,
        device=x.device,
    )

    # for large N values
    if M <= 32:
        NUM_ITER = 1
        BLOCK_SIZE_M1 = min(8, triton.next_power_of_2(M))
        BLOCK_SIZE_N1 = 128
        NUM_WARPS = 1 if BLOCK_SIZE_M1 < 4 else 4
        NUM_STAGES = 1
    else:
        NUM_ITER = 1
        BLOCK_SIZE_M1 = 16
        BLOCK_SIZE_N1 = 256
        NUM_WARPS = 4
        NUM_STAGES = 1

    # for small N values
    if N_half <= 1024:
        NUM_ITER = 1
        NUM_STAGES = 1
        NUM_WARPS = 4
        BLOCK_SIZE_N1 = min(256, triton.next_power_of_2(N_half))
        # BLOCK_SIZE_N needs to be multiple of 32
        BLOCK_SIZE_N1 = max(32, BLOCK_SIZE_N1)
        BLOCK_SIZE_M1 = min(8, triton.next_power_of_2(N_half))

    # shuffle requires block sizes to be multiple of 32
    if shuffle:
        BLOCK_SIZE_M1 = triton.cdiv(BLOCK_SIZE_M1, 32) * 32
        BLOCK_SIZE_N1 = triton.cdiv(BLOCK_SIZE_N1, 32) * 32

    BLOCK_SIZE_M2 = 1 if M <= 128 else 4
    BLOCK_SIZE_N2 = 16

    num_pid = triton.cdiv(M, BLOCK_SIZE_M1) * triton.cdiv(
        N_half, BLOCK_SIZE_N1 * NUM_ITER
    )
    if X_HAS_SPLITK:
        num_pid += triton.cdiv(M, BLOCK_SIZE_M2) * triton.cdiv(N2, BLOCK_SIZE_N2)

    grid = (num_pid,)
    _fused_reduce_act_mul_and_dynamic_mxfp4_quant_kernel[grid](
        x,
        y,
        y_scale,
        x2,
        y2,
        0 if not X_HAS_SPLITK else x.stride(0),
        x.stride(0) if not X_HAS_SPLITK else x.stride(1),
        x.stride(1) if not X_HAS_SPLITK else x.stride(2),
        y.stride(0),
        y.stride(1),
        y_scale.stride(0),
        y_scale.stride(1),
        0 if not X_HAS_SPLITK else x2.stride(0),
        0 if not X_HAS_SPLITK else x2.stride(1),
        0 if not X_HAS_SPLITK else x2.stride(2),
        0 if not X_HAS_SPLITK else y2.stride(0),
        0 if not X_HAS_SPLITK else y2.stride(1),
        M=M,
        N1=N_half,
        N2=N2,
        BLOCK_SIZE_M1=BLOCK_SIZE_M1,
        BLOCK_SIZE_N1=BLOCK_SIZE_N1,
        BLOCK_SIZE_M2=BLOCK_SIZE_M2,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2,
        NUM_ITER=NUM_ITER,
        NUM_STAGES=NUM_STAGES,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        SCALING_MODE=0,
        ACTIVATION=_get_activation_from_str(activation) if activation else "",
        scaleN=scaleN_valid,
        scaleM_pad=(scaleM if use_scale_shuffle_padding else 1),
        scaleN_pad=scaleN,
        SHUFFLE=shuffle,
        X_HAS_SPLITK=X_HAS_SPLITK,
        X_NUM_KSPLIT=x_num_splitk,
        X_NUM_KSPLIT_POW2=triton.next_power_of_2(x_num_splitk),
        num_warps=NUM_WARPS,
        waves_per_eu=0,
        num_stages=1,
    )

    return (y, y_scale), y2
