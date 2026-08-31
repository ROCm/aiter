# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton kernel for fused MXFP8 grouped (MoE) GEMM.

Each expert's GEMM is dispatched as contiguous M-blocks in a single kernel
launch.  E8M0 microscales (one uint8 per ``QUANT_BLOCK_SIZE`` elements along K)
are converted to FP32 power-of-two values inside the kernel and applied per
K-iteration during the accumulation loop.

Grid layout:  ``(total_m_blocks * n_blocks,)``
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_repr = make_kernel_repr(
    "_moe_gemm_mxfp8_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "QUANT_BLOCK_SIZE",
        "EVEN_K",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
    }
)
@triton.jit(repr=_repr)
def _moe_gemm_mxfp8_kernel(
    # Data pointers
    lhs_ptr,  # [total_tokens, K]  FP8
    rhs_ptr,  # [E, N, K]          FP8
    out_ptr,  # [total_tokens, N]  output dtype
    x_scale_ptr,  # [total_tokens, K // QBS]  uint8 E8M0
    w_scale_ptr,  # [E, N, K // QBS]          uint8 E8M0
    bias_ptr,  # [E, N] or None
    # Block-to-expert mapping
    block_expert_ids_ptr,  # [total_m_blocks]  int32
    block_token_offsets_ptr,  # [total_m_blocks]  int32
    # Dimensions
    total_M,
    N,
    K,
    # Strides
    stride_lhs_m,
    stride_lhs_k,
    stride_rhs_e,
    stride_rhs_n,
    stride_rhs_k,
    stride_out_m,
    stride_out_n,
    stride_xs_m,  # x_scale row stride
    stride_xs_k,  # x_scale col stride
    stride_ws_e,  # w_scale expert stride
    stride_ws_n,  # w_scale N stride
    stride_ws_k,  # w_scale K stride
    # Flags
    HAS_BIAS: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """BLOCK_K must equal QUANT_BLOCK_SIZE for this kernel."""

    tl.assume(stride_lhs_m > 0)
    tl.assume(stride_lhs_k > 0)
    tl.assume(stride_rhs_e > 0)
    tl.assume(stride_rhs_n > 0)
    tl.assume(stride_rhs_k > 0)
    tl.assume(stride_out_m > 0)
    tl.assume(stride_out_n > 0)

    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_mb = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    expert_id = tl.load(block_expert_ids_ptr + pid_mb)
    token_offset = tl.load(block_token_offsets_ptr + pid_mb)

    tl.assume(expert_id >= 0)
    tl.assume(token_offset >= 0)

    offs_m = token_offset + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < total_M
    n_mask = offs_n < N

    # lhs pointers
    a_ptrs = lhs_ptr + offs_m[:, None] * stride_lhs_m + offs_k[None, :] * stride_lhs_k
    # rhs[expert_id] transposed view (K, N)
    b_ptrs = (
        rhs_ptr
        + expert_id * stride_rhs_e
        + offs_k[:, None] * stride_rhs_k
        + offs_n[None, :] * stride_rhs_n
    )

    # Scale pointers — advance by 1 per K-iteration since BLOCK_K == QUANT_BLOCK_SIZE
    xs_ptrs = x_scale_ptr + offs_m * stride_xs_m
    ws_ptrs = w_scale_ptr + expert_id * stride_ws_e + offs_n * stride_ws_n

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_scale_idx in range(tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
        else:
            k_mask = offs_k < K
            a = tl.load(
                a_ptrs,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

        # Load E8M0 scales and convert to FP32
        a_scale_e8m0 = tl.load(
            xs_ptrs + k_scale_idx * stride_xs_k,
            mask=m_mask,
            other=127,  # 2^0 = neutral
        )
        b_scale_e8m0 = tl.load(
            ws_ptrs + k_scale_idx * stride_ws_k,
            mask=n_mask,
            other=127,
        )
        a_scale = (a_scale_e8m0.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        b_scale = (b_scale_e8m0.to(tl.uint32) << 23).to(tl.float32, bitcast=True)

        accumulator += (
            tl.dot(a, b, input_precision="ieee") * a_scale[:, None] * b_scale[None, :]
        )

        a_ptrs += BLOCK_K * stride_lhs_k
        b_ptrs += BLOCK_K * stride_rhs_k
        offs_k += BLOCK_K

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + expert_id * N + offs_n,
            mask=n_mask,
            other=0.0,
        )
        accumulator += bias[None, :]

    c = accumulator.to(out_ptr.type.element_ty)
    c_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, c, mask=c_mask)
