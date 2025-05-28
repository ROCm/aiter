# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
import torch


@triton.jit
def _static_per_tensor_fp8_quant_kernel(
    qx_ptr: torch.Tensor,
    x_in_ptr: torch.Tensor,
    scale_in_ptr: torch.Tensor,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    scale = tl.load(scale_in_ptr)
    scale_recip = 1 / scale

    qx = (x * scale_recip).to(qx_ptr.dtype.element_ty)

    tl.store(qx_ptr + offs, qx, mask=mask)


def static_per_tensor_fp8_quant(
    qx: torch.Tensor, x_in: torch.Tensor, scale_in: torch.Tensor
):
    """
    #TODO: Add Doc
    """
    assert scale_in.numel() == 1  # only single scale value
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)
    _static_per_tensor_fp8_quant_kernel[grid](
        qx, x_in, scale_in, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2
    )

    return qx


@triton.jit
def _dynamic_per_tensor_fp8_quant_kernel(
    x_in_ptr: torch.Tensor,
    scale_out_ptr: torch.Tensor,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    m = tl.max(tl.abs(x))
    tl.atomic_max(scale_out_ptr, m / FP8_MAX, sem="relaxed")


def dynamic_per_tensor_fp8_quant(
    qx: torch.Tensor, x_in: torch.Tensor, scale_out: torch.Tensor
):
    """
    #TODO: Add Doc
    """
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)
    _dynamic_per_tensor_fp8_quant_kernel[grid](
        x_in,
        scale_out,
        cols,
        x_in.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
        FP8_MAX=torch.finfo(qx.dtype).max,
    )

    _static_per_tensor_fp8_quant_kernel[grid](
        qx, x_in, scale_out, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2
    )

    return qx, scale_out


@triton.jit
def _dynamic_per_token_fp8_quant_kernel(
    qx_ptr: torch.Tensor,
    scale_out_ptr: torch.Tensor,
    x_in_ptr: torch.Tensor,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    m = tl.max(tl.abs(x), axis=-1)
    scale_out = m / FP8_MAX
    scale_recip = 1 / scale_out

    qx = x * scale_recip
    qx = qx.to(qx_ptr.dtype.element_ty)

    scale_offs = pid
    tl.store(scale_out_ptr + scale_offs, scale_out)

    tl.store(qx_ptr + offs, qx, mask=mask, cache_modifier=".cs")


def dynamic_per_token_fp8_quant(
    qx: torch.Tensor,
    x_in: torch.Tensor,
    scale_out: torch.Tensor,
    quant_dtype=torch.float8_e4m3fnuz,
    dtypeMax: torch.Tensor = torch.finfo(torch.float8_e4m3fnuz).max,
):
    """
    #TODO: Add doc
    """
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)
    _dynamic_per_token_fp8_quant_kernel[grid](
        qx,
        scale_out,
        x_in,
        cols,
        x_in.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
        FP8_MAX=dtypeMax,
    )

    return qx, scale_out


@triton.heuristics(
    {
        "EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N"] % (args["BLOCK_SIZE_N"] * args["NUM_ITER"]) == 0,
    }
)
@triton.jit
def _dynamic_mxfp4_quant_kernel(
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

    for pid_n in tl.range(start_n, min(start_n + NUM_ITER, N), num_stages=NUM_STAGES):
        x_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        x_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n

        if EVEN_M_N:
            x = tl.load(x_ptr + x_offs, cache_modifier=".cg").to(tl.float32)
        else:
            x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
            x = tl.load(x_ptr + x_offs, mask=x_mask, cache_modifier=".cg").to(
                tl.float32
            )

        NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
        x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)
        # Calculate scale
        amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
        amax = amax.to(tl.int32, bitcast=True)
        amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
        amax = amax.to(tl.float32, bitcast=True)
        scale_e8m0_unbiased = tl.log2(amax).floor() - 2
        scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)

        # blockscale_e8m0
        bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127  # in fp32, we have 2&(e - 127)

        quant_scale = tl.exp2(-scale_e8m0_unbiased)

        # Compute quantized x
        qx = x * quant_scale

        # Convert quantized fp32 tensor to uint32 before converting to mxfp4 format
        # Note: MXFP4  S:1-bit, E:2-bit, M:1-bit
        #   Zeros: S000 -> +/-0
        #   Denormal Numbers: S001 -> +/- 0.5
        #   Normal Numbers:
        #           S010 -> +/- 1.0
        #           S011 -> +/- 1.5
        #           S100 -> +/- 2.0
        #           S101 -> +/- 3.0
        #           S110 -> +/- 4.0
        #           S111 -> +/- 6.0
        qx = qx.to(tl.uint32, bitcast=True)

        # Extract sign, exponents and mantissa fields from FP32
        s = qx & 0x80000000
        e = (qx >> 23) & 0xFF
        m = qx & 0x7FFFFF
        E8_BIAS: tl.constexpr = 127
        E2_BIAS: tl.constexpr = 1

        # Denormal numbers
        # If exponent is less than 127, then it's a denormal number
        # See above, for denormal number mantissa is always 1 and we set bit 1 of mantissa
        adjusted_exponents = tl.core.sub(E8_BIAS, e + 1, sanitize_overflow=False)
        m = tl.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)
        # For normal numbers, bias is changed from 127 to 1, and for subnormals, we keep exponent as 0.
        # Note: E8_BIAS - E2_BIAS = 126, so for normals we subtract that.
        e = tl.maximum(e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        # Combine sign, exponent, and mantissa, while saturating
        # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
        e2m1_tmp = tl.minimum((((e << 2) | (m >> 21)) + 1) >> 1, 0x7)
        e2m1_value = ((s >> 28) | e2m1_tmp).to(tl.uint8)
        e2m1_value = tl.reshape(
            e2m1_value, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
        )
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)
        out_tensor = out_tensor.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

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
        bs_offs = bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
        if EVEN_M_N:
            tl.store(bs_ptr + bs_offs, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS))
        else:
            bs_mask = (bs_offs_m < M)[:, None] & (
                bs_offs_n < (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
            )[None, :]
            tl.store(
                bs_ptr + bs_offs,
                bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS),
                mask=bs_mask,
            )


def dynamic_mxfp4_quant(
    x: torch.Tensor, scaling_mode: str = "even"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to MX FP4 format.

    Args:
        x: The input tensor, typically fp16 or bf16.
        scaling_mode: The method to calculate MX block scaling.
            - "even" (default): `even_round` in `quark.torch.quantization.utils`.
            - etc.
    Returns:
        A tuple of (x_fp4, blockscale_e8m0).
    """
    # Assume x is 2D-Tensor for now
    M, N = x.shape

    assert (N // 2) % 2 == 0

    # This is fixed by spec for MXFP4. Do not tune this.
    MXFP4_QUANT_BLOCK_SIZE = 32
    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    blockscale_e8m0 = torch.empty(
        ((N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE, M),
        dtype=torch.uint8,
        device=x.device,
    ).T

    # for large N values
    if M <= 32:
        NUM_ITER = 1
        BLOCK_SIZE_M = triton.next_power_of_2(M)
        BLOCK_SIZE_N = 32
        NUM_WARPS = 1
        NUM_STAGES = 1
    else:
        NUM_ITER = 4
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        NUM_WARPS = 4
        NUM_STAGES = 2

        if N <= 16384:
            BLOCK_SIZE_M = 32
            BLOCK_SIZE_N = 128

    # for small N values
    if N <= 1024:
        NUM_ITER = 1
        NUM_STAGES = 1
        NUM_WARPS = 4
        BLOCK_SIZE_N = min(256, triton.next_power_of_2(N))
        # BLOCK_SIZE_N needs to be multiple of 32
        BLOCK_SIZE_N = max(32, BLOCK_SIZE_N)
        BLOCK_SIZE_M = min(8, triton.next_power_of_2(M))

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N * NUM_ITER),
    )

    _dynamic_mxfp4_quant_kernel[grid](
        x,
        x_fp4,
        blockscale_e8m0,
        *x.stride(),
        *x_fp4.stride(),
        *blockscale_e8m0.stride(),
        M=M,
        N=N,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        SCALING_MODE=0,
        NUM_ITER=NUM_ITER,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_STAGES=NUM_STAGES,
        num_warps=NUM_WARPS,
        waves_per_eu=0,
        num_stages=1,
    )

    return (x_fp4, blockscale_e8m0)
