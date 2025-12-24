from typing import Optional
import torch
import triton
import aiter
from aiter.ops.triton._triton_kernels.fused_fp8_quant import (
    _fused_rms_fp8_per_tensor_static_quant_kernel,
    _fused_rms_fp8_group_quant_kernel,
    _fused_flatten_fp8_group_quant_kernel,
    _fused_reduce_act_mul_fp8_group_quant,
    _fused_reduce_rms_fp8_group_quant_kernel,
    _fused_silu_mul_fp8_per_tensor_static_quant_kernel,
)
from aiter.ops.triton._triton_kernels.activation import (
    _get_activation_from_str,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


fp8_dtype = aiter.dtypes.fp8


def fused_rms_fp8_per_tensor_static_quant(
    inp1,
    inp1_weight,
    inp1_epsilon,
    inp1_scale,
    inp2=None,
    inp2_weight=None,
    inp2_epsilon=None,
    dtype_quant=fp8_dtype,
    res1=None,
    output_unquantized_inp1=False,
    rmsnorm_convert_to_inp1_type=False,
):
    """
    This op contains several steps:
        1. if res1 is not None, inp1 = inp1 + res1, and store inp1 to out_res1
        2. perform RMS norm along the last dimenion for inp1
        3. if inp2 is not None, perform RMS norm along the last dimenion for inp2
        4. perform fp8 quantization for inp1 only

    Key parameters:
    - x: Matrix X with shape (M, N1, N2).

    Returns:
    - out1_fp8: The output matrix with shape (M, N1).
    - out1_s: The output matrix with shape (1,).
    - out1: The output matrix with shape (M, N1).
    - out2: The output matrix with shape (M, N2).
    - out_res1: The output matrix with shape (M, N1).
    - out1: The output matrix with shape (M, N1).
    """
    M, N1 = inp1.shape
    BLOCK_SIZE_N = triton.next_power_of_2(N1)
    if inp2 is not None:
        M2, N2 = inp2.shape
        BLOCK_SIZE_N = triton.next_power_of_2(N2)
        assert (
            M == M2
        ), "The leading dimension should be identical between inp1 and inp2"
    else:
        N2 = 0
    out1_fp8 = torch.empty((M, N1), dtype=dtype_quant, device=inp1.device)

    out2 = None
    out2_row_stride = 0
    out2_col_stride = 0
    inp2_row_stride = 0
    inp2_col_stride = 0
    if inp2 is not None:
        out2 = torch.empty((M, N2), dtype=inp1.dtype, device=inp1.device)
        inp2_row_stride = inp2.stride(0)
        inp2_col_stride = inp2.stride(1)
        out2_row_stride = out2.stride(0)
        out2_col_stride = out2.stride(1)

    out1 = None
    out1_row_stride = 0
    out1_col_stride = 0
    if output_unquantized_inp1:
        out1 = torch.empty((M, N1), dtype=inp1.dtype, device=inp1.device)
        out1_row_stride = out1.stride(0)
        out1_col_stride = out1.stride(1)

    out_res1 = None
    res1_row_stride = 0
    res1_col_stride = 0
    out_res1_row_stride = 0
    out_res1_col_stride = 0
    if res1 is not None:
        Mr, Nr = res1.shape
        assert (
            M == Mr and N1 == Nr
        ), "The shape should be identical between inp1 and res1"
        out_res1 = torch.empty((M, N1), dtype=inp1.dtype, device=inp1.device)
        res1_row_stride = res1.stride(0)
        res1_col_stride = res1.stride(1)
        out_res1_row_stride = out_res1.stride(0)
        out_res1_col_stride = out_res1.stride(1)

    if BLOCK_SIZE_N <= 512:
        num_warps = 1
    elif BLOCK_SIZE_N <= 2048:
        num_warps = 4
    elif BLOCK_SIZE_N <= 4096:
        num_warps = 8
    else:
        num_warps = 16

    DTYPE_MAX = (
        torch.finfo(out1_fp8.dtype).max
        if torch.is_floating_point(out1_fp8)
        else torch.iinfo(out1_fp8.dtype).max
    )

    _fused_rms_fp8_per_tensor_static_quant_kernel[(M,)](
        inp1,
        inp1_weight,
        inp2,
        inp2_weight,
        res1,
        out1_fp8,
        out2,
        out_res1,
        out1,
        inp1_scale,
        inp1_epsilon,
        inp2_epsilon,
        M,
        N1,
        N2,
        inp1.stride(0),
        inp2_row_stride,
        inp1.stride(1),
        inp2_col_stride,
        res1_row_stride,
        res1_col_stride,
        out1_fp8.stride(0),
        out1_fp8.stride(1),
        out2_row_stride,
        out2_col_stride,
        out_res1_row_stride,
        out_res1_col_stride,
        out1_row_stride,
        out1_col_stride,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        HAVE_SECOND_INPUT=(inp2 is not None),
        FIRST_INPUT_RES=(res1 is not None),
        FIRST_INPUT_OUT=output_unquantized_inp1,
        RMSNORM_CONVERT_TO_INP1_TYPE=rmsnorm_convert_to_inp1_type,
        num_warps=num_warps,
    )

    return out1_fp8, out1, out2, out_res1


def fused_rms_fp8_group_quant(
    inp1,
    inp1_weight,
    inp1_epsilon,
    inp2=None,
    inp2_weight=None,
    inp2_epsilon=None,
    group_size=128,
    dtype_quant=fp8_dtype,
    res1=None,
    output_unquantized_inp1=False,
    transpose_scale=False,
):
    """
    This op contains several steps:
        1. if res1 is not None, inp1 = inp1 + res1, and store inp1 to out_res1
        2. perform RMS norm along the last dimenion for inp1
        3. if inp2 is not None, perform RMS norm along the last dimenion for inp2
        4. perform fp8 quantization for inp1 only

    Key parameters:
    - x: Matrix X with shape (M, N1, N2).
    - transpose_scale: If True, return scale with shape (M, cdiv(N1, group_size)) but stored in
                      column-major (transposed) memory layout. Equivalent to:
                      scale.transpose(0, 1).contiguous().view(*scale.shape)

    Returns:
    - out1_fp8: The output matrix with shape (M, N1).
    - out1_bs: The output matrix with shape (M, cdiv(N1, group_size)).
              When transpose_scale=True, has column-major memory layout (transposed storage).
    - out1: The output matrix with shape (M, N1).
    - out2: The output matrix with shape (M, N2).
    - out_res1: The output matrix with shape (M, N1).
    - out1: The output matrix with shape (M, N1).
    """

    M, N1 = inp1.shape
    BLOCK_SIZE_N = max(triton.next_power_of_2(N1), group_size)
    if inp2 is not None:
        M2, N2 = inp2.shape
        BLOCK_SIZE_N = max(triton.next_power_of_2(N2), BLOCK_SIZE_N)
        assert (
            M == M2
        ), "The leading dimension should be identical between inp1 and inp2"
    else:
        N2 = 0
    out1_fp8 = torch.empty((M, N1), dtype=dtype_quant, device=inp1.device)
    num_bs_cols = (N1 + group_size - 1) // group_size
    if transpose_scale:
        # Create with transposed shape for direct transposed storage
        out1_bs = torch.empty(
            (num_bs_cols, M),
            dtype=torch.float32,
            device=inp1.device,
        )
    else:
        out1_bs = torch.empty(
            (M, num_bs_cols),
            dtype=torch.float32,
            device=inp1.device,
        )

    out2 = None
    out2_row_stride = 0
    out2_col_stride = 0
    inp2_row_stride = 0
    inp2_col_stride = 0
    if inp2 is not None:
        out2 = torch.empty((M, N2), dtype=inp1.dtype, device=inp1.device)
        inp2_row_stride = inp2.stride(0)
        inp2_col_stride = inp2.stride(1)
        out2_row_stride = out2.stride(0)
        out2_col_stride = out2.stride(1)

    out1 = None
    out1_row_stride = 0
    out1_col_stride = 0
    if output_unquantized_inp1:
        out1 = torch.empty((M, N1), dtype=inp1.dtype, device=inp1.device)
        out1_row_stride = out1.stride(0)
        out1_col_stride = out1.stride(1)

    BLOCK_SIZE_N = max(BLOCK_SIZE_N, group_size)
    out_res1 = None
    res1_row_stride = 0
    res1_col_stride = 0
    out_res1_row_stride = 0
    out_res1_col_stride = 0
    if res1 is not None:
        Mr, Nr = res1.shape
        assert (
            M == Mr and N1 == Nr
        ), "The shape should be identical between inp1 and res1"
        out_res1 = torch.empty((M, N1), dtype=inp1.dtype, device=inp1.device)
        res1_row_stride = res1.stride(0)
        res1_col_stride = res1.stride(1)
        out_res1_row_stride = out_res1.stride(0)
        out_res1_col_stride = out_res1.stride(1)

    if BLOCK_SIZE_N <= 512:
        num_warps = 1
    elif BLOCK_SIZE_N <= 2048:
        num_warps = 4
    elif BLOCK_SIZE_N <= 4096:
        num_warps = 8
    else:
        num_warps = 16

    DTYPE_MAX = (
        torch.finfo(out1_fp8.dtype).max
        if torch.is_floating_point(out1_fp8)
        else torch.iinfo(out1_fp8.dtype).max
    )

    # When transpose_scale=True, swap the strides to write directly in transposed layout
    if transpose_scale:
        out1_bs_row_stride = out1_bs.stride(1)
        out1_bs_col_stride = out1_bs.stride(0)
    else:
        out1_bs_row_stride = out1_bs.stride(0)
        out1_bs_col_stride = out1_bs.stride(1)

    _fused_rms_fp8_group_quant_kernel[(M,)](
        inp1,
        inp1_weight,
        inp2,
        inp2_weight,
        res1,
        out1_fp8,
        out1_bs,
        out2,
        out_res1,
        out1,
        inp1_epsilon,
        inp2_epsilon,
        M,
        N1,
        N2,
        inp1.stride(0),
        inp2_row_stride,
        inp1.stride(1),
        inp2_col_stride,
        res1_row_stride,
        res1_col_stride,
        out1_fp8.stride(0),
        out1_fp8.stride(1),
        out1_bs_row_stride,
        out1_bs_col_stride,
        out2_row_stride,
        out2_col_stride,
        out_res1_row_stride,
        out_res1_col_stride,
        out1_row_stride,
        out1_col_stride,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        QUANT_BLOCK_SIZE=group_size,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        HAVE_SECOND_INPUT=(inp2 is not None),
        FIRST_INPUT_RES=(res1 is not None),
        FIRST_INPUT_OUT=output_unquantized_inp1,
        num_warps=num_warps,
    )
    # When transpose_scale=True, view the transposed buffer back to original shape
    # This keeps shape (M, num_bs_cols) but with column-major memory layout
    if transpose_scale:
        out1_bs = out1_bs.view(M, num_bs_cols)

    return (out1_fp8, out1_bs), out1, out2, out_res1


def fused_flatten_fp8_group_quant(
    x: torch.Tensor,
    group_size,
    dtype_quant=fp8_dtype,
):
    """
    Flatten the last two dimension of x and perform FP8 per-token group quantization along the last dimension

    Key parameters:
    - x: Matrix X with shape (M, N1, N2).

    Returns:
    - out: The output matrix with shape (M, N1 * N2).
    - out_block_scales: The output matrix with shape (M, cdiv((N1 * N2), group_size)).
    """
    M, N1, N2 = x.shape

    BLOCK_SIZE_N2 = max(triton.next_power_of_2(N2), group_size)
    N = N1 * N2
    out = torch.empty((M, N), dtype=dtype_quant, device=x.device)
    out_block_scales = torch.empty(
        (M, triton.cdiv(N, group_size)), dtype=torch.float32, device=x.device
    )

    DTYPE_MAX = (
        torch.finfo(out.dtype).max
        if torch.is_floating_point(out)
        else torch.iinfo(out.dtype).max
    )
    grid = (
        M,
        N1,
    )
    _fused_flatten_fp8_group_quant_kernel[grid](
        x,
        out,
        out_block_scales,
        *x.stride(),
        *out.stride(),
        *out_block_scales.stride(),
        N2,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2,
        QUANT_BLOCK_SIZE=group_size,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
    )

    return out, out_block_scales


def fused_reduce_act_mul_fp8_group_quant(
    x: torch.Tensor,
    activation: str = "silu",
    x2: Optional[torch.Tensor] = None,
    group_size=128,
    dtype_quant=fp8_dtype,
    dtype: Optional[float] = torch.bfloat16,
):
    """
    Apply reduction along the first dimension and apply the activation function + per-token group quantization.
    If x2 is provided, the only reduction along the first dimension is applied to x2

    Args:
        if x is 3-dim,
            x: (SPK, M, 2*N1), dtype = fp32.
            x2: (SPK, M, 2*N1), dtype = fp32.

        if x is 2-dim,
            x: (M, 2*N1), dtype = fp16 or bf16.
            x2 must be None
            the kernel is essentially identical to aiter.ops.triton.activation.act_mul_and_fp8_group_quant

        activation: activation function to apply before quantization.
            - It splits the features into two parts and applies the activation to the first part.
            - Then, it adds the results together before quantization.
            - Supports the following activations:
                - "silu"
                - "gelu"
                - "gelu_tanh"

    Returns:
        tuple: (y, y_scale), y2
            y: (M, N1), dtype = dtype_quant
            y_scale: (M, cdiv(N1, group_size)), dtype = fp32
            y2: (M, N2), dtype = dtype
    """
    _LOGGER.info(
        f"FUSED_REDUCTION_ACT_MUL_FP8_GROUP_QUANT: x={tuple(x.shape)} activation={activation}"
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
        assert x2 is None, "x2 should be None in x.dim() == 2 cases"

    assert (
        N1 % 2 == 0
    ), "The last dimension for x1 should be multiple of 2 for acitvation and multiplication"
    N1 = N1 // 2

    y = torch.empty((M, N1), dtype=dtype_quant, device=x.device)
    y_scale = torch.empty(
        (M, (N1 + group_size - 1) // group_size),
        dtype=torch.float32,
        device=x.device,
    )

    BLOCK_SIZE_N1 = max(triton.next_power_of_2(N1), group_size)
    BLOCK_SIZE_N2 = max(triton.next_power_of_2(N2), 32)
    BLOCK_SIZE_M2 = 1 if M <= 128 else 4
    X_MASK = N1 % BLOCK_SIZE_N1 != 0

    DTYPE_MAX = (
        torch.finfo(y.dtype).max
        if torch.is_floating_point(y)
        else torch.iinfo(y.dtype).max
    )
    num_pid = M
    if X_HAS_SPLITK:
        num_pid += triton.cdiv(M, BLOCK_SIZE_M2) * triton.cdiv(N2, BLOCK_SIZE_N2)
    grid = (num_pid,)
    _fused_reduce_act_mul_fp8_group_quant[grid](
        x,
        y,
        y_scale,
        x2,
        y2,
        M,
        N1,
        N2,
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
        ACTIVATION=_get_activation_from_str(activation) if activation else "",
        BLOCK_SIZE_M2=BLOCK_SIZE_M2,
        BLOCK_SIZE_N1=BLOCK_SIZE_N1,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2,
        QUANT_BLOCK_SIZE=group_size,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        X_HAS_SPLITK=X_HAS_SPLITK,
        X_NUM_KSPLIT=x_num_splitk,
        X_NUM_KSPLIT_POW2=triton.next_power_of_2(x_num_splitk),
        X_MASK=X_MASK,
        num_warps=1 if max(BLOCK_SIZE_N1, BLOCK_SIZE_N2) <= 512 else 4,
    )

    return (y, y_scale), y2


def fused_reduce_rms_fp8_group_quant(
    inp1,
    inp1_weight,
    inp1_epsilon,
    inp2=None,
    inp2_weight=None,
    inp2_epsilon=None,
    inp3=None,
    group_size=128,
    dtype_quant=fp8_dtype,
    dtype=None,
    res1=None,
    output_unquantized_inp1=False,
    out3=None,
):
    """
    This op contains several steps:
        1. if res1 is not None, inp1 = inp1 + res1, and store inp1 to out_res1
        2. perform RMS norm along the last dimenion for inp1
        3. if inp2 is not None, perform RMS norm along the last dimenion for inp2
        4. perform fp8 quantization for inp1 only
        5. if inp3 is not None, perform sum reduction along the first dimension, in the meantime, the inp1 and inp2 has to have the identical first diemsion as inp3

    Key parameters:
    - x: Matrix X with shape (M, N1, N2).

    Returns:
    - out1_fp8: The output matrix with shape (M, N1).
    - out1_bs: The output matrix with shape (M, cdiv(N1, group_size)).
    - out1: The output matrix with shape (M, N1).
    - out2: The output matrix with shape (M, N2).
    - out_res1: The output matrix with shape (M, N1).
    - out3: The output matrix with shape (M, N3).
    - out1: The output matrix with shape (M, N1).
    """

    out_dtype = dtype if dtype is not None else inp1.dtype
    SPK = 1
    HAS_SPLITK = False
    inp1_spk_stride = 0
    inp1_row_stride = 0
    inp1_col_stride = 0
    if inp1.dim() == 3:
        SPK, M, N1 = inp1.shape
        assert SPK > 1, "Split-k dimension should have more than 1 element."
        HAS_SPLITK = True
        inp1_spk_stride = inp1.stride(0)
        inp1_row_stride = inp1.stride(1)
        inp1_col_stride = inp1.stride(2)
    else:
        M, N1 = inp1.shape
        inp1_row_stride = inp1.stride(0)
        inp1_col_stride = inp1.stride(1)
    BLOCK_SIZE_N1 = max(triton.next_power_of_2(N1), group_size)
    if inp2 is not None:
        if SPK > 1:
            assert (
                inp2.dim() == 3 and inp2.shape[0] == SPK and inp2.shape[1] == M
            ), f"Incompatible shapes {inp1.shape=}, {inp2.shape=}"
            _, _, N2 = inp2.shape
        else:
            _, N2 = inp2.shape
        BLOCK_SIZE_N2 = triton.next_power_of_2(N2)
    else:
        N2 = 0
        BLOCK_SIZE_N2 = 1
    if inp3 is not None:
        assert (
            inp3.dim() == 3 and inp3.shape[0] == SPK and inp3.shape[1] == M
        ), f"Incompatible shapes {inp1.shape=}, {inp3.shape=}"
        _, _, N3 = inp3.shape
        BLOCK_SIZE_N3 = triton.next_power_of_2(N3)
    else:
        N3 = 0
        BLOCK_SIZE_N3 = 1

    out1_fp8 = torch.empty((M, N1), dtype=dtype_quant, device=inp1.device)
    out1_bs = torch.empty(
        (M, (N1 + group_size - 1) // group_size),
        dtype=torch.float32,
        device=inp1.device,
    )
    out1_fp8_row_stride = out1_fp8.stride(0)
    out1_fp8_col_stride = out1_fp8.stride(1)
    out1_bs_row_stride = out1_bs.stride(0)
    out1_bs_col_stride = out1_bs.stride(1)

    out2 = None
    inp2_spk_stride = 0
    out2_row_stride = 0
    out2_col_stride = 0
    inp2_row_stride = 0
    inp2_col_stride = 0
    if inp2 is not None:
        out2 = torch.empty((M, N2), dtype=out_dtype, device=inp1.device)
        if SPK > 1:
            inp2_spk_stride = inp2.stride(0)
            inp2_row_stride = inp2.stride(1)
            inp2_col_stride = inp2.stride(2)
        else:
            inp2_row_stride = inp2.stride(0)
            inp2_col_stride = inp2.stride(1)
        out2_row_stride = out2.stride(0)
        out2_col_stride = out2.stride(1)

    inp3_spk_stride = 0
    out3_row_stride = 0
    out3_col_stride = 0
    inp3_row_stride = 0
    inp3_col_stride = 0
    if inp3 is not None:
        if out3 is None:
            out3 = torch.empty((M, N3), dtype=out_dtype, device=inp1.device)
        inp3_spk_stride = inp3.stride(0)
        inp3_row_stride = inp3.stride(1)
        inp3_col_stride = inp3.stride(2)
        out3_row_stride = out3.stride(0)
        out3_col_stride = out3.stride(1)

    out1 = None
    out1_row_stride = 0
    out1_col_stride = 0
    if output_unquantized_inp1:
        out1 = torch.empty((M, N1), dtype=out_dtype, device=inp1.device)
        out1_row_stride = out1.stride(0)
        out1_col_stride = out1.stride(1)

    out_res1 = None
    res1_row_stride = 0
    res1_col_stride = 0
    out_res1_row_stride = 0
    out_res1_col_stride = 0
    if res1 is not None:
        Mr, Nr = res1.shape
        assert (
            M == Mr and N1 == Nr
        ), "The shape should be identical between inp1 and res1"
        out_res1 = torch.empty((M, N1), dtype=out_dtype, device=inp1.device)
        res1_row_stride = res1.stride(0)
        res1_col_stride = res1.stride(1)
        out_res1_row_stride = out_res1.stride(0)
        out_res1_col_stride = out_res1.stride(1)

    max_BN = max(BLOCK_SIZE_N1, BLOCK_SIZE_N2, BLOCK_SIZE_N3)
    if max_BN <= 512:
        num_warps = 1
    elif max_BN <= 2048:
        num_warps = 4
    elif max_BN <= 4096:
        num_warps = 8
    else:
        num_warps = 16

    DTYPE_MAX = (
        torch.finfo(out1_fp8.dtype).max
        if torch.is_floating_point(out1_fp8)
        else torch.iinfo(out1_fp8.dtype).max
    )
    _fused_reduce_rms_fp8_group_quant_kernel[(3 * M if HAS_SPLITK else 2 * M,)](
        inp1,
        inp1_weight,
        inp2,
        inp2_weight,
        inp3,
        res1,
        out1_fp8,
        out1_bs,
        out2,
        out_res1,
        out1,
        out3,
        inp1_epsilon,
        inp2_epsilon,
        M,
        N1,
        N2,
        N3,
        inp1_spk_stride,
        inp2_spk_stride,
        inp3_spk_stride,
        inp1_row_stride,
        inp2_row_stride,
        inp3_row_stride,
        inp1_col_stride,
        inp2_col_stride,
        inp3_col_stride,
        res1_row_stride,
        res1_col_stride,
        out1_fp8_row_stride,
        out1_fp8_col_stride,
        out1_bs_row_stride,
        out1_bs_col_stride,
        out2_row_stride,
        out2_col_stride,
        out_res1_row_stride,
        out_res1_col_stride,
        out1_row_stride,
        out1_col_stride,
        out3_row_stride,
        out3_col_stride,
        BLOCK_SIZE_N1=BLOCK_SIZE_N1,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2,
        BLOCK_SIZE_N3=BLOCK_SIZE_N3,
        N_MASK1=(BLOCK_SIZE_N1 != N1),
        N_MASK2=(BLOCK_SIZE_N2 != N2),
        N_MASK3=(BLOCK_SIZE_N3 != N3),
        QUANT_BLOCK_SIZE=group_size,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        HAVE_SECOND_INPUT=(inp2 is not None),
        FIRST_INPUT_RES=(res1 is not None),
        FIRST_INPUT_OUT=output_unquantized_inp1,
        HAS_SPLITK=HAS_SPLITK,
        NUM_SPLITK=SPK,
        NUM_SPLITK_POW2=triton.next_power_of_2(SPK),
        num_warps=num_warps,
    )

    return (out1_fp8, out1_bs), out1, out2, out_res1, out3


def fused_silu_mul_fp8_per_tensor_static_quant(
    inp,
    inp_scale,
    dtype_quant=fp8_dtype,
    silu_convert_to_inp_type=False,
):
    """
    This op contains two steps:
        1. compute the silu mul operations
        2. perform fp8 quantization for inp1 only

    Key parameters:
    - x: Matrix X with shape (M, 2 * N).

    Returns:
    - out_fp8: The output matrix with shape (M, N).
    """
    M, N2 = inp.shape
    assert N2 % 2 == 0
    N = N2 // 2
    BLOCK_SIZE_N = triton.next_power_of_2(N)

    out_fp8 = torch.empty((M, N), dtype=dtype_quant, device=inp.device)

    if BLOCK_SIZE_N <= 512:
        num_warps = 1
    elif BLOCK_SIZE_N <= 2048:
        num_warps = 4
    elif BLOCK_SIZE_N <= 4096:
        num_warps = 8
    else:
        num_warps = 16

    DTYPE_MAX = (
        torch.finfo(out_fp8.dtype).max
        if torch.is_floating_point(out_fp8)
        else torch.iinfo(out_fp8.dtype).max
    )

    _fused_silu_mul_fp8_per_tensor_static_quant_kernel[(M,)](
        inp,
        out_fp8,
        inp_scale,
        M,
        N,
        inp.stride(0),
        inp.stride(1),
        out_fp8.stride(0),
        out_fp8.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        SILU_CONVERT_TO_INP_TYPE=silu_convert_to_inp_type,
        num_warps=num_warps,
    )

    return out_fp8
