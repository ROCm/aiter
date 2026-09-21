from typing import Literal

import torch
import triton

import aiter
from aiter.ops.triton._triton_kernels.activation import (
    _act_mul_and_dynamic_fp8_group_quant_kernel,
    _act_mul_and_dynamic_mxfp4_quant_kernel,
    _silu_and_mul_backward_kernel,
    fused_silu_mul_kernel,
)
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

fp8_dtype = aiter.dtypes.fp8

_LOGGER = AiterTritonLogger()


def _get_silu_and_mul_backward_config(n_cols: int) -> dict:
    config_dir = resolve_config_dir("fusions", "SILU-AND-MUL-BACKWARD")
    configs = load_config_json(f"{config_dir}/DEFAULT.json", required=True)
    return select_leq_config(configs, n_cols)


def act_mul_and_mxfp4_quant(
    x: torch.Tensor,
    activation: Literal["silu", "gelu", "gelu_tanh"],
    scaling_mode: str = "even",
    shuffle: bool = False,
    scale_shuffle_padding: bool = False,
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
    use_scale_shuffle_padding = shuffle or scale_shuffle_padding
    if use_scale_shuffle_padding:
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
        scaleM_pad=(scaleM if use_scale_shuffle_padding else 1),
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


def fused_silu_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused SiLU-and-mul along the last dimension (same pattern as MoE silu-fused GEMM).

    ``x`` must be contiguous with even ``size(-1)``. For last size ``2 * d``, the first
    ``d`` lanes are passed through SiLU (``_silu_exp2``); the second ``d`` lanes are the
    multipliers. Output shape matches ``x`` except ``out.size(-1) == d``.

    Returns:
        ``out`` if provided, else a newly allocated tensor.
    """

    def _pick_block_n(d: int, n_rows: int) -> int:
        """Tile size along the reduced last dim (cap 1024); at least 32 for vectorization.

        Tuned on ROCm for MoE TP4 locals (GLM-4.7 ``d=384``, Kimi-K2.5 ``d=512``) and wide
        MoE activations: ``n_rows`` selects decode vs prefill N-tiling (see sweep in repo
        history / ``bench_moe.py -bench_silu_mul``).
        """
        n = max(d, 1)
        # Kimi-K2.5 TP4 (d=512): prefill favors one 512-wide N tile; decode keeps 256×2.
        if n == 512:
            return 512 if n_rows > 4096 else 256
        # GLM-4.7 TP4 (d=384): wider decode rows use 256×2; larger batches favor 128×3 N tiles.
        if n == 384:
            return 256 if n_rows <= 128 else 128
        upper = min(n, 1024)
        p = 1
        while p * 2 <= upper:
            p *= 2
        return max(32, p)

    def _pick_block_m(n_rows: int, block_n: int, d: int) -> int:
        """Row tile size: latency shapes use wide M tiles; prefill uses tuned (d, n_rows) pairs."""
        if n_rows <= 64:
            return min(32, max(4, triton.next_power_of_2(n_rows)))
        if d == 384 and n_rows > 128:
            return 32 if n_rows > 8192 else 8
        if d == 512 and n_rows > 4096:
            return 8
        if d == 512 and 128 < n_rows <= 4096:
            return 8
        if block_n >= 1024:
            return 8
        if block_n >= 512:
            return 8
        return 16

    def _pick_num_warps(n_rows: int, block_m: int, block_n: int) -> int:
        """ROCm: 8 warps for tiny full-wavefront decode tiles; 2 warps for larger tiles."""
        if n_rows <= 128 and block_m >= 16 and block_n >= 128:
            return 8
        return 2

    assert x.is_cuda, "fused_silu_mul requires a CUDA tensor"
    assert x.is_contiguous(), "x must be contiguous"
    last = x.size(-1)
    assert last % 2 == 0, "last dimension must be even (2 * d)"
    d = last // 2
    leading = x.shape[:-1]
    n_rows = x.numel() // (2 * d)
    if n_rows == 0:
        return (
            torch.empty(*leading, d, dtype=x.dtype, device=x.device)
            if out is None
            else out
        )

    _LOGGER.info(f"fused_silu_mul: x={tuple(x.shape)} last_half={d} rows={n_rows}")

    if out is None:
        out = torch.empty(*leading, d, dtype=x.dtype, device=x.device)
    else:
        assert out.is_contiguous(), "out must be contiguous"
        assert out.shape == (*leading, d), "out shape must match x with last dim halved"
        assert out.dtype == x.dtype and out.device == x.device

    row_stride_in = 2 * d
    col_stride_in = 1
    row_stride_out = d
    col_stride_out = 1

    block_n = _pick_block_n(d, n_rows)
    block_m = _pick_block_m(n_rows, block_n, d)
    grid_m = triton.cdiv(n_rows, block_m)
    grid_n = triton.cdiv(d, block_n)
    num_warps = _pick_num_warps(n_rows, block_m, block_n)

    grid = (grid_m, grid_n)
    fused_silu_mul_kernel[grid](
        x,
        out,
        n_rows,
        d,
        row_stride_in,
        col_stride_in,
        row_stride_out,
        col_stride_out,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        waves_per_eu=0,
    )
    return out


def silu_and_mul_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the input gradient of ``silu(gate) * up``.

    Args:
        grad_output: Gradient shaped like ``x`` with its last dimension halved.
            It must share ``x``'s dtype and device; strided tensors are supported.
        x: Contiguous FP16, BF16, or FP32 tensor with concatenated ``[gate, up]``.
        out: Optional contiguous destination matching ``x``.

    Returns:
        The gradient with respect to ``x``. Computation uses FP32 intermediates.

    Width thresholds select the launch configuration from the current
    architecture's ``DEFAULT.json``. Only gfx942 and gfx950 are supported.
    """
    assert x.is_contiguous(), "x must be contiguous"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"unsupported dtype: {x.dtype}"
    last = x.size(-1)
    assert last > 0, "x last dimension must be non-zero"
    assert last % 2 == 0, "x last dimension must be even"
    n_cols = last // 2
    expected_shape = (*x.shape[:-1], n_cols)
    assert (
        grad_output.shape == expected_shape
    ), f"grad_output shape must be {expected_shape}, got {tuple(grad_output.shape)}"
    assert grad_output.dtype == x.dtype, "grad_output dtype must match x dtype"
    assert grad_output.device == x.device, "grad_output device must match x device"

    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.shape == x.shape, "out shape must match x shape"
        assert out.dtype == x.dtype, "out dtype must match x dtype"
        assert out.device == x.device, "out device must match x device"
        assert out.is_contiguous(), "out must be contiguous"

    flat_x = x.reshape(-1, last)
    flat_grad_output = grad_output.reshape(-1, n_cols)
    flat_out = out.reshape(-1, last)
    n_rows = flat_x.size(0)
    if n_rows == 0:
        return out

    _LOGGER.info(
        "SILU_AND_MUL_BACKWARD: x=%s grad=%s",
        tuple(x.shape),
        tuple(grad_output.shape),
    )
    with torch.cuda.device(x.device):
        config = _get_silu_and_mul_backward_config(n_cols)
        block_m = min(config.pop("BLOCK_M"), triton.next_power_of_2(n_rows))
        block_n = config.pop("BLOCK_N")
        grid = (triton.cdiv(n_rows, block_m), triton.cdiv(n_cols, block_n))
        _silu_and_mul_backward_kernel[grid](
            flat_grad_output,
            flat_x,
            flat_out,
            n_rows,
            n_cols,
            flat_grad_output.stride(0),
            flat_grad_output.stride(1),
            flat_x.stride(0),
            flat_x.stride(1),
            flat_out.stride(0),
            flat_out.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            **config,
        )
    return out
