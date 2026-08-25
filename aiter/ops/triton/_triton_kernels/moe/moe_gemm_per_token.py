# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton kernel for fused per-token-scaled grouped (MoE) GEMM.

Each expert's GEMM is dispatched as contiguous M-blocks in a single kernel
launch.  Per-token activation scales and per-expert weight scales are applied
after the FP8 dot product accumulation.

Grid layout:  ``(total_m_blocks * n_blocks,)``
Each program decodes which expert it belongs to via a pre-computed lookup
table (``block_expert_ids``, ``block_token_offsets``).
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_repr = make_kernel_repr(
    "_moe_gemm_per_token_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "EVEN_K",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
    }
)
@triton.jit(repr=_repr)
def _moe_gemm_per_token_kernel(
    # Data pointers
    lhs_ptr,  # [total_tokens, K]  FP8
    rhs_ptr,  # [E, N, K]          FP8
    out_ptr,  # [total_tokens, N]  output dtype
    x_scale_ptr,  # [total_tokens]     FP32 per-token scale
    w_scale_ptr,  # [E]                FP32 per-expert scale
    bias_ptr,  # [E, N] or None
    # Block-to-expert mapping (pre-computed on host)
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
    # Flags
    HAS_BIAS: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
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

    # lhs[offs_m, offs_k]
    a_ptrs = lhs_ptr + offs_m[:, None] * stride_lhs_m + offs_k[None, :] * stride_lhs_k
    # rhs[expert_id] transposed: (K, N) view
    b_ptrs = (
        rhs_ptr
        + expert_id * stride_rhs_e
        + offs_k[:, None] * stride_rhs_k
        + offs_n[None, :] * stride_rhs_n
    )

    m_mask = offs_m < total_M
    n_mask = offs_n < N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(tl.cdiv(K, BLOCK_K)):
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

        accumulator += tl.dot(a, b, input_precision="ieee")

        a_ptrs += BLOCK_K * stride_lhs_k
        b_ptrs += BLOCK_K * stride_rhs_k
        offs_k += BLOCK_K

    # Apply per-token x_scale and per-expert w_scale
    xs = tl.load(x_scale_ptr + offs_m, mask=m_mask, other=1.0)
    ws = tl.load(w_scale_ptr + expert_id)
    accumulator *= xs[:, None] * ws

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
