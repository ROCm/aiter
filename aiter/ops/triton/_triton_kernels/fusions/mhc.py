# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Fused Triton kernels for mHC (manifold-constrained Hyper Connection).
"""

import triton
import triton.language as tl


@triton.jit
def _mhc_fused_rmsnorm_matmul_kernel(
    # Pointers to matrices
    x_ptr,      # Input: (M, K) - flattened n-stream residual
    phi_ptr,    # Weight: (K, N) - projection matrix
    out_ptr,    # Output: (M, N)
    bias_ptr,   # Bias: (N,)
    # Scalars
    alpha,      # Scaling factor
    # Matrix dimensions
    M,          # Batch size
    N,          # Output dimension (n_streams)
    K,          # Input dimension (n_streams * hidden_dim)
    # Strides
    stride_xm,
    stride_xk,
    stride_phik,
    stride_phin,
    stride_outm,
    stride_outn,
    # Epsilon for numerical stability
    eps,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel: RMSNorm + MatMul + Scale + Bias with deferred division.

    Computes: out = α · (x · φ / ||x||_rms) + b

    Grid: (cdiv(M, BLOCK_SIZE_M), cdiv(N, BLOCK_SIZE_N))
    Each program handles one (BLOCK_SIZE_M, BLOCK_SIZE_N) tile of output.

    Key optimization: The RMS norm division is deferred to after the matmul,
    reducing the number of divisions from K to N elements per row.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Accumulators
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Pointers to first block of x and phi
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    phi_ptrs = phi_ptr + offs_k[:, None] * stride_phik + offs_n[None, :] * stride_phin

    # Iterate over K dimension
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_offs = k_start + offs_k
        mask_k = k_offs < K
        mask_m = offs_m < M
        mask_mk = mask_m[:, None] & mask_k[None, :]

        # Load x block: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        x_block = tl.load(x_ptrs, mask=mask_mk, other=0.0)
        x_block_f32 = x_block.to(tl.float32)

        # Accumulate sum of squares for RMS
        sum_sq += tl.sum(x_block_f32 * x_block_f32, axis=1)

        # Load phi block: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        mask_kn = mask_k[:, None] & (offs_n[None, :] < N)
        phi_block = tl.load(phi_ptrs, mask=mask_kn, other=0.0)
        phi_block_f32 = phi_block.to(tl.float32)

        # Accumulate matmul
        acc += tl.dot(x_block_f32, phi_block_f32)

        # Advance pointers
        x_ptrs += BLOCK_SIZE_K * stride_xk
        phi_ptrs += BLOCK_SIZE_K * stride_phik

    # Compute RMS normalization factor: rsigma = 1 / sqrt(mean(x^2) + eps)
    mean_sq = sum_sq / K
    rsigma = tl.rsqrt(mean_sq + eps)

    # Apply deferred normalization
    result = acc * rsigma[:, None]

    # Load bias and apply scaling
    bias_ptrs = bias_ptr + offs_n
    mask_n = offs_n < N
    bias = tl.load(bias_ptrs, mask=mask_n, other=0.0).to(tl.float32)

    out = alpha * result + bias[None, :]

    # Store output
    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=mask_out)
