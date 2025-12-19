from typing import Literal
import torch
import triton
import triton.language as tl
from typing import Optional
from aiter.utility import dtypes
from aiter.ops.triton._triton_kernels.fused_mxfp4_quant import (
    _fused_rms_mxfp4_quant_kernel,
    _fused_flatten_mxfp4_quant,
    _fused_reduce_act_mul_and_dynamic_mxfp4_quant_kernel,
    _fused_reduce_rms_mxfp4_quant_kernel,
    _fused_dynamic_mxfp4_quant_moe_sort_kernel,
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
    output_unquantized_inp1=False,
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

        always returns (out1_fp4, out1_bs), out1, out2, out_res1
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

    out1 = None
    out1_stride_m = 0
    if output_unquantized_inp1:
        out1 = torch.empty((M, N1), dtype=x1.dtype, device=x1.device)
        out1_stride_m = out1.stride(0)

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
        out1,
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
        out1_stride_m,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        HAS_SECOND_INPUT=(x2 is not None),
        FIRST_INPUT_RES=(res1 is not None),
        FIRST_INPUT_OUT=output_unquantized_inp1,
        SCALE_N=SCALE_N_valid,
        SCALE_M_PAD=(SCALE_M if use_scale_shuffle_padding else 1),
        SCALE_N_PAD=SCALE_N,
        SHUFFLE=shuffle,
        SHUFFLE_PAD=use_scale_shuffle_padding,
    )

    return (out1_fp4, out1_bs), out1, out2, out_res1


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

    NUM_ITER = 1
    NUM_WARPS = 4
    NUM_STAGES = 1

    BLOCK_SIZE_M1 = 1 if M <= 128 else 4
    BLOCK_SIZE_M2 = 1 if M <= 128 else 4

    # for small N values
    if N_half <= 1024:
        BLOCK_SIZE_N1 = 32
    else:
        BLOCK_SIZE_N1 = 128

    if N2 <= 256:
        BLOCK_SIZE_N2 = 8
    elif N2 <= 1024:
        BLOCK_SIZE_N2 = 32
    else:
        BLOCK_SIZE_N2 = 128

    # shuffle requires block sizes to be multiple of 32
    if shuffle:
        BLOCK_SIZE_M1 = triton.cdiv(BLOCK_SIZE_M1, 32) * 32
        BLOCK_SIZE_N1 = triton.cdiv(BLOCK_SIZE_N1, 32) * 32

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


def fused_reduce_rms_mxfp4_quant(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: float = 0.0,
    x3: Optional[torch.Tensor] = None,
    res1: Optional[torch.Tensor] = None,
    shuffle: Optional[bool] = False,
    scale_shuffle_padding: Optional[bool] = False,
    output_unquantized_inp1=False,
    dtype=None,
    out3=None,
):
    """
    This op contains several steps:
        1. if res1 is not None, x1 = x1 + res1, and store x1 to out_res1
        2. perform RMS norm along the last dimenion for x1
        3. if x2 is not None, perform RMS norm along the last dimenion for x2
        4. perform mxfp4 quantization for x1 only
        5. if inp3 is not None, perform sum reduction along the first dimension, in the meantime, the x1 and x2 has to have the identical first diemsion as x3

    Key parameters:
    - x: Matrix X with shape (M, N1, N2).

    Returns:
    - out1_fp4: The output matrix with shape (M, N1 // 2).
    - out1_bs: The output matrix with shape (M, cdiv(N1, MXFP4_QUANT_BLOCK_SIZE)).
    - out2: The output matrix with shape (M, N2).
    - out_res1: The output matrix with shape (M, N1).
    - out3: The output matrix with shape (M, N3).
    - out1: The output matrix with shape (M, N1).

        always returns (out1_fp4, out1_bs), out1, out2, out_res1, out3
    """
    _LOGGER.info(f"FUSED_RMS_MXFP4_QUANT: inp1={tuple(x1.shape)}")

    out_dtype = dtype if dtype is not None else x1.dtype
    MXFP4_QUANT_BLOCK_SIZE = 32
    SPK = 1
    HAS_SPLITK = False
    x1_stride_spk = 0
    x1_stride_m = 0
    if x1.dim() == 3:
        SPK, M, N1 = x1.shape
        assert SPK > 1, "Split-k dimension should have more than 1 element."
        HAS_SPLITK = True
        x1_stride_spk = x1.stride(0)
        x1_stride_m = x1.stride(1)
    else:
        M, N1 = x1.shape
        x1_stride_m = x1.stride(0)
    BLOCK_SIZE_N = max(triton.next_power_of_2(N1), MXFP4_QUANT_BLOCK_SIZE)

    BLOCK_SIZE_N2 = 1
    x2_stride_spk = 0
    x2_stride_m = 0
    if x2 is not None:
        if SPK > 1:
            _, _, N2 = x2.shape
            assert (
                x2.dim() == 3 and x1.shape[0] == SPK and x2.shape[1] == M
            ), f"Incompatible shapes {x1.shape=}, {x2.shape=}"
            x2_stride_spk = x2.stride(0)
            x2_stride_m = x2.stride(1)
        else:
            _, N2 = x2.shape
            x2_stride_m = x2.stride(0)
        BLOCK_SIZE_N2 = triton.next_power_of_2(N2)
    else:
        N2 = 0

    BLOCK_SIZE_N3 = 1
    x3_stride_spk = 0
    x3_stride_m = 0
    if x3 is not None:
        assert x3.dim() == 3 and x3.shape[0] == SPK and x3.shape[1] == M
        _, _, N3 = x3.shape
        BLOCK_SIZE_N3 = triton.next_power_of_2(N3)
        x3_stride_spk = x3.stride(0)
        x3_stride_m = x3.stride(1)
    else:
        N3 = 0

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

    out1 = None
    out1_stride_m = 0
    if output_unquantized_inp1:
        out1 = torch.empty((M, N1), dtype=out_dtype, device=x1.device)
        out1_stride_m = out1.stride(0)

    out_res1 = None
    res1_stride_m = 0
    out_res1_stride_m = 0
    if res1 is not None:
        out_res1 = torch.empty((M, N1), dtype=out_dtype, device=x1.device)
        res1_stride_m = res1.stride(0)
        out_res1_stride_m = out_res1.stride(0)

    out2 = None
    out2_stride_m = 0
    if x2 is not None:
        out2 = torch.empty((M, N2), dtype=out_dtype, device=x1.device)
        out2_stride_m = out2.stride(0)

    out3_stride_m = 0
    if x3 is not None:
        if out3 is None:
            out3 = torch.empty((M, N3), dtype=out_dtype, device=x1.device)
        out3_stride_m = out3.stride(0)

    r = 1
    if HAS_SPLITK:
        r = 3
    elif x2 is not None:
        r = 2
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * r,)
    _fused_reduce_rms_mxfp4_quant_kernel[grid](
        x1,
        x1_weight,
        x2,
        x2_weight,
        x3,
        res1,
        out1_fp4,
        out1_bs,
        out1,
        out2,
        out3,
        out_res1,
        x1_epsilon,
        x2_epsilon,
        M,
        N1,
        N2,
        N3,
        x1_stride_spk,
        x1_stride_m,
        x2_stride_spk,
        x2_stride_m,
        x3_stride_spk,
        x3_stride_m,
        res1_stride_m,
        out1_fp4.stride(0),
        *out1_bs.stride(),
        out1_stride_m,
        out2_stride_m,
        out3_stride_m,
        out_res1_stride_m,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2,
        BLOCK_SIZE_N3=BLOCK_SIZE_N3,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        HAS_SECOND_INPUT=(x2 is not None),
        FIRST_INPUT_RES=(res1 is not None),
        FIRST_INPUT_OUT=output_unquantized_inp1,
        HAS_SPLITK=HAS_SPLITK,
        NUM_SPLITK=SPK,
        NUM_SPLITK_POW2=triton.next_power_of_2(SPK),
        SCALE_N=SCALE_N_valid,
        SCALE_M_PAD=(SCALE_M if use_scale_shuffle_padding else 1),
        SCALE_N_PAD=SCALE_N,
        SHUFFLE=shuffle,
        SHUFFLE_PAD=use_scale_shuffle_padding,
    )

    return (out1_fp4, out1_bs), out1, out2, out_res1, out3


def fused_dynamic_mxfp4_quant_moe_sort(
    x: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    topk: int,
    block_size: int = 32,
    scaling_mode: str = "even",
):
    """
    Fusing dynamic_mxfp4_quant and moe_mxfp4_sort

    Args:
        x: The input tensor, typically fp16 or bf16.
        scaling_mode: The method to calculate MX block scaling.
            - "even" (default): `even_round` in `quark.torch.quantization.utils`.
            - etc.
        sorted_ids: The indices used for sorting.

    shuffle is not supported here

    Returns:
        A tuple of (x_fp4, blockscale_e8m0).
    """
    # Assume x is 2D-Tensor for now
    M, N = x.shape

    assert (N // 2) % 2 == 0

    # This is fixed by spec for MXFP4. Do not tune this.
    # For performance, perhaps, we should look at passing multiple of 32 column blocks
    # that a triton program can process
    MXFP4_QUANT_BLOCK_SIZE = 32

    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    # scaleM = triton.cdiv(M, 32) * 32
    scaleN_valid = triton.cdiv(N, MXFP4_QUANT_BLOCK_SIZE)
    # scaleN = triton.cdiv(scaleN_valid, 8) * 8
    scaleN = scaleN_valid

    BLOCK_SIZE_Mx = 128

    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 8
    BLOCK_SIZE_M_u32, BLOCK_SIZE_N_u32 = 16, 4

    M_i, N_i = M, scaleN
    M_o, N_o = sorted_ids.shape[0], N_i
    assert (N_i // 2) % 2 == 0
    assert block_size % BLOCK_SIZE_M == 0

    blockscale_e8m0_sorted = torch.empty(
        (
            triton.cdiv(M_o, BLOCK_SIZE_M),
            triton.cdiv(N_o, BLOCK_SIZE_N),
            BLOCK_SIZE_N_u32,
            BLOCK_SIZE_M_u32,
            4,
        ),
        dtype=torch.uint8,
        device=x.device,
    )  # .fill_(0)

    num_pid = triton.cdiv(M, BLOCK_SIZE_Mx) * scaleN + triton.cdiv(
        M_o, BLOCK_SIZE_M
    ) * triton.cdiv(N_i, BLOCK_SIZE_N)
    _fused_dynamic_mxfp4_quant_moe_sort_kernel[(num_pid,)](
        x,
        x_fp4,
        sorted_ids,
        num_valid_ids,
        blockscale_e8m0_sorted,
        M,
        N,
        scaleN,
        *x.stride(),
        *x_fp4.stride(),
        *blockscale_e8m0_sorted.stride(),
        token_num=token_num,
        M_i=M_i,
        N_i=N_i,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        BLOCK_SIZE_Mx=BLOCK_SIZE_Mx,
        BLOCK_SIZE_M=BLOCK_SIZE_M // 2,
        BLOCK_SIZE_N=BLOCK_SIZE_N // 2,
        TOPK=topk,
    )

    return (
        x_fp4.view(dtypes.fp4x2),
        blockscale_e8m0_sorted.view(dtypes.fp8_e8m0).view(-1, N_o),
    )


@triton.jit
def _fused_quant_fp8_sort_kernel(
    # Pointers
    input_ptr,
    sorted_ids_ptr,
    num_valid_ids_ptr,
    x_fp8_ptr,
    scale_sorted_ptr,
    # Input/Output strides
    stride_input_m: tl.constexpr,
    stride_input_n: tl.constexpr,
    stride_x_fp8_m: tl.constexpr,
    stride_x_fp8_n: tl.constexpr,
    stride_scale_o3: tl.constexpr,
    stride_scale_o2: tl.constexpr,
    stride_scale_o1: tl.constexpr,
    stride_scale_o0: tl.constexpr,
    # Problem size
    M_input: tl.constexpr,
    N_input: tl.constexpr,
    N_scale_cols: tl.constexpr,
    token_num: tl.constexpr,
    # Block configuration
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,  # quant_block_size / 2
    QUANT_BLOCK_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    # Quantization parameters
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
):
    pid_m = tl.program_id(0) * 2
    pid_n = tl.program_id(1) * 2

    num_valid_ids = tl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_SIZE_M >= num_valid_ids:
        return

    out = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.uint32)

    for i in range(4):
        m = i % 2 * BLOCK_SIZE_M  # 0 or BLOCK_SIZE_M
        n = i // 2 * BLOCK_SIZE_N  # 0 or BLOCK_SIZE_N

        sorted_ids_offs_m = pid_m * BLOCK_SIZE_M + m + tl.arange(0, BLOCK_SIZE_M)
        sorted_ids_mask = sorted_ids_offs_m < num_valid_ids
        sorted_ids = tl.load(
            sorted_ids_ptr + sorted_ids_offs_m,
            mask=sorted_ids_mask,
            other=0,
        )
        topk_ids = sorted_ids >> 24
        token_ids = sorted_ids & 0xFFFFFF

        if TOPK == 1:
            original_m_idx = token_ids
        else:
            original_m_idx = token_ids * TOPK + topk_ids

        input_offs_n = (pid_n * BLOCK_SIZE_N + n) * QUANT_BLOCK_SIZE + tl.arange(
            0, BLOCK_SIZE_N * QUANT_BLOCK_SIZE
        )
        input_offs = (
            original_m_idx[:, None] * stride_input_m
            + input_offs_n[None, :] * stride_input_n
        )
        input_mask = (original_m_idx < M_input)[:, None] & (input_offs_n < N_input)[
            None, :
        ]

        x = tl.load(input_ptr + input_offs, mask=input_mask, other=0.0).to(tl.float32)

        x_reshaped = x.reshape(BLOCK_SIZE_M * BLOCK_SIZE_N, QUANT_BLOCK_SIZE)

        amax = tl.max(tl.abs(x_reshaped), axis=-1, keep_dims=True)

        amax = amax.to(tl.int32, bitcast=True)
        amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
        amax = amax.to(tl.float32, bitcast=True)

        scale_e8m0_unbiased = tl.log2(amax).floor() - tl.log2(DTYPE_MAX).floor()
        scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)

        quant_scale = tl.exp2(-scale_e8m0_unbiased)
        x_fp8 = tl.clamp(x_reshaped * quant_scale, DTYPE_MIN, DTYPE_MAX)
        x_fp8 = x_fp8.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N * QUANT_BLOCK_SIZE)

        scale_e8m0 = (scale_e8m0_unbiased.to(tl.uint8) + 127).to(tl.uint8)
        scale_e8m0 = scale_e8m0.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)  # [BLOCK_SIZE_M]

        out_offs_n = (pid_n * BLOCK_SIZE_N + n) * QUANT_BLOCK_SIZE + tl.arange(
            0, BLOCK_SIZE_N * QUANT_BLOCK_SIZE
        )
        out_offs = (
            original_m_idx[:, None] * stride_x_fp8_m
            + out_offs_n[None, :] * stride_x_fp8_n
        )
        out_mask = (original_m_idx < M_input)[:, None] & (out_offs_n < N_input)[None, :]
        tl.store(
            x_fp8_ptr + out_offs, x_fp8.to(x_fp8_ptr.type.element_ty), mask=out_mask
        )

        out = out | (scale_e8m0.to(tl.uint32) << (i * 8))

    offs_0 = tl.arange(0, BLOCK_SIZE_M)
    offs_1 = tl.arange(0, BLOCK_SIZE_N)
    offs_2 = pid_n // 2
    offs_3 = pid_m // 2
    offs = (
        offs_0[:, None] * stride_scale_o0
        + offs_1[None, :] * stride_scale_o1
        + offs_2 * stride_scale_o2
        + offs_3 * stride_scale_o3
    )
    tl.store(scale_sorted_ptr + offs, out)


def fused_quant_fp8_sort(
    input: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    block_size: int = 32,
    quant_block_size: int = 8,
    quant_dtype: torch.dtype = dtypes.fp8,
) -> tuple[torch.Tensor, torch.Tensor]:
    BLOCK_SIZE_M = block_size
    BLOCK_SIZE_N = quant_block_size
    BLOCK_SIZE_M_u32 = BLOCK_SIZE_M // 2
    BLOCK_SIZE_N_u32 = BLOCK_SIZE_N // 2

    M, N = input.shape
    assert (
        N % quant_block_size == 0
    ), f"N ({N}) must be multiple of quant_block_size ({quant_block_size})"
    assert block_size % 32 == 0, "block_size must be multiple of 32"

    N_blocks = triton.cdiv(N, block_size)

    if quant_dtype == dtypes.fp8:
        DTYPE_MAX = 448.0
        DTYPE_MIN = -448.0
    elif quant_dtype == torch.float8_e4m3fn:
        DTYPE_MAX = 448.0
        DTYPE_MIN = -448.0
    else:
        DTYPE_MAX = 448.0
        DTYPE_MIN = -448.0

    x_fp8 = torch.empty_like(input, dtype=quant_dtype, device="cuda")
    M_o, N_o = sorted_ids.shape[0], N_blocks

    # [M_sorted_blocks/2, N_blocks/2, BLOCK_SIZE_N_u32, BLOCK_SIZE_M_u32]
    scale_e8m0_packed = torch.empty(
        (
            triton.cdiv(M_o, BLOCK_SIZE_M),
            triton.cdiv(N_o, BLOCK_SIZE_N),
            BLOCK_SIZE_N_u32,
            BLOCK_SIZE_M_u32,
        ),
        dtype=torch.uint32,
        device=input.device,
    )

    grid = (
        triton.cdiv(M_o, BLOCK_SIZE_M),  # 32
        triton.cdiv(N_o, BLOCK_SIZE_N),  # 8
    )

    _fused_quant_fp8_sort_kernel[grid](
        input,
        sorted_ids,
        num_valid_ids,
        x_fp8,
        scale_e8m0_packed,
        *input.stride(),
        *x_fp8.stride(),
        *scale_e8m0_packed.stride(),
        M_input=M,
        N_input=N,
        N_scale_cols=N_blocks,
        token_num=token_num,
        BLOCK_SIZE_M=BLOCK_SIZE_M // 2,
        BLOCK_SIZE_N=BLOCK_SIZE_N // 2,
        QUANT_BLOCK_SIZE=32,
        TOPK=M // token_num,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=DTYPE_MIN,
    )

    return x_fp8, scale_e8m0_packed.view(dtypes.fp8_e8m0).view(-1, N_o)
