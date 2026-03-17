# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton.language as tl
from aiter.ops.triton._triton_kernels.quant.fused_fp8_quant import _fp8_quant_op
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

import triton

_batched_gemm_a16w8_repr = make_kernel_repr(
    "_batched_gemm_a16w8_kernel",
    [
        "HAS_BIAS",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "EVEN_K",
        "GRID_MN",
        "PREQUANT",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_batched_gemm_a16w8_repr)
def _batched_gemm_a16w8_kernel(
    # Pointers to matrices
    a_ptr,  # (B, M, K) fp16 / bf16
    b_ptr,  # (B, K, N) fp8
    c_ptr,  # (B, M, N)
    b_scale_ptr,  # (B, scale_k, scale_n)
    bias_ptr,  # (B, N)
    # Matrix sizes
    M,
    N,
    K,
    # Strides - how data moves in memory (contiguous/non-contiguous)
    # more specifically the distance between consecutive elements accessed by threads within a wavefront
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    stride_bscaleb,
    stride_bscale_k,
    stride_bscale_n,
    stride_biasb,
    # Meta-parameters
    GROUP_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    PREQUANT: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call batched_gemm_a16w8 function
    below

    Batched GEMM:
        Computes the matmul C[i] = A[i] x B[i] where A is FP16BF/16 and B is INT8, applying blockscale weight dequantization
        for every i in a given batch. Optionally, adds a bias to each result.

    The dequantization uses blockscale quantization where weights are divided into blocks
    and each block has its own scale factor.

    Key parameters:
    - A: Batch tensor A with shape (B, M, K) in FP16/BF16.
    - B: Batch tensor B with shape (B, K, N) in FP8.
    - C: Batch tensor C with shape (B, M, N).
    - B_scale: Weight scale batch tensor with shape (B, scale_k, scale_n) for blockscale.
      where scale_k = (K + GROUP_K - 1) // GROUP_K, scale_n = (N + GROUP_N - 1) // GROUP_N
    - Bias: Bias batch tensor with shape (B, 1, N) or (B, N).
    - PREQUANT: If True, quantize A to FP8 and use both a_scale and b_scale.
                If False, keep A as FP16/BF16 and cast B up to match A's dtype
    """

    tl.assume(stride_ab > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bb > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cb > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_bscaleb > 0)
    tl.assume(stride_bscale_k > 0)
    tl.assume(stride_bscale_n > 0)
    tl.assume(stride_biasb > 0)

    # Get batch program ID
    batch_id = tl.program_id(axis=0)
    # Map program IDs 'pid' to the block of C it should compute
    # This is done in a grouped ordering to promote L2 data reuse
    pid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Cast to int64 for safe addressing (avoids int32 overflow during offset calculation)
    batch_id = tl.cast(batch_id, tl.int64)
    stride_ab = tl.cast(stride_ab, tl.int64)
    stride_bb = tl.cast(stride_bb, tl.int64)
    stride_cb = tl.cast(stride_cb, tl.int64)
    stride_bscaleb = tl.cast(stride_bscaleb, tl.int64)

    # Offsets
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Precompute batch base pointers
    a_batch_ptr = a_ptr + batch_id * stride_ab
    b_batch_ptr = b_ptr + batch_id * stride_bb
    b_scale_batch_ptr = b_scale_ptr + batch_id * stride_bscaleb

    # A & B pointers
    a_ptrs = a_batch_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_batch_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Create pointers for the blockscale tensor
    # B_scale has shape (B, scale_k, scale_n)
    offs_bsn = offs_bn // GROUP_N
    # Scale rows to advance per K block
    offs_ks_step = BLOCK_SIZE_K // GROUP_K
    b_scale_ptrs = b_scale_batch_ptr + offs_bsn * stride_bscale_n

    # Accumulator (FP32)
    # Why choose FP32 accumulation? FP8 x FP8 or FP16 x FP8 accumulation will overflow
    # Maintains model accuracy, prevents acc error and numerical stability
    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    # K loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking K dimension
        # If it is out of bounds, set it to 0
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=(offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )

        # Load scale
        b_scale = tl.load(b_scale_ptrs)

        # PREQUANT logic:
        # - If PREQUANT=True: Quantize A to FP8 on-the-fly, do FP8×FP8 matmul with both a_scale and b_scale
        # - If PREQUANT=False: Keep A as FP16/BF16, cast B up to A's dtype, do FP16×FP16 matmul with only b_scale
        if PREQUANT:
            # Quantize A to FP8 dynamically per block
            a, a_scale = _fp8_quant_op(
                a, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_K, DTYPE_MAX, DTYPE_MIN
            )
            a = a.to(b_ptr.type.element_ty).reshape(BLOCK_SIZE_M, BLOCK_SIZE_K)
            a_scale = a_scale.reshape(BLOCK_SIZE_M)

            # FP8 × FP8 → FP32, apply both scales
            fused_scale = a_scale[:, None] * b_scale[None, :]
            accumulator += tl.dot(a, b, input_precision="ieee") * fused_scale
        else:
            # Cast B up to match A's datatype (FP16/BF16)
            b = b.to(a_ptr.type.element_ty)
            # A in FP16/BF16, B dequantized from FP8 to FP16/BF16 -> FP32, apply only b_scale
            accumulator += tl.dot(a, b, input_precision="ieee") * b_scale[None, :]

        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        # Advance b_scale pointer by the number of scale rows that were consumed this K block
        b_scale_ptrs += offs_ks_step * stride_bscale_k

    # Bias
    if HAS_BIAS:
        offs_bias = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        bias = tl.load(bias_ptr + batch_id * stride_biasb + offs_bias)
        accumulator = accumulator.to(bias_ptr.type.element_ty) + bias[None, :]

    # Store
    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    c_ptrs = (
        c_ptr
        + batch_id * stride_cb
        + offs_cm[:, None] * stride_cm
        + offs_cn[None, :] * stride_cn
    )

    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _get_config(M: int, N: int, K: int):
    return get_gemm_config("BATCHED_GEMM-A16W8_BLOCKSCALE", M, N, K)
