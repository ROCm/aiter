# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Fused Triton kernels for mHC (manifold-constrained Hyper Connection).
"""

import triton
import triton.language as tl


@triton.jit
def _mhc_fused_rmsnorm_matmul_sigmoid_kernel(
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
    Fused kernel: RMSNorm + MatMul + Scale + Bias + Sigmoid with deferred division.

    Computes: out = sigmoid(α · (x · φ / ||x||_rms) + b)

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

    linear_out = alpha * result + bias[None, :]
    
    # Apply sigmoid activation: σ(x) = 1 / (1 + exp(-x))
    out = tl.sigmoid(linear_out)

    # Store output
    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=mask_out)


@triton.jit
def _sinkhorn_knopp_log_domain_kernel(
    # Pointers
    logits_ptr,     # Input: (M, N, N) raw logits
    out_ptr,        # Output: (M, N, N) doubly stochastic matrices
    # Dimensions
    M,              # Batch size (number of matrices)
    # Strides
    stride_batch,   # Stride for batch dimension
    stride_row,     # Stride for row dimension
    stride_col,     # Stride for column dimension
    # Meta-parameters
    N: tl.constexpr,            # Matrix size (must be power of 2, max 64)
    NUM_ITERS: tl.constexpr,    # Number of Sinkhorn iterations
):
    """
    Log-domain Sinkhorn-Knopp kernel for projecting raw logits onto doubly stochastic matrices.

    Computes doubly stochastic matrix P where all rows and columns sum to 1.

    Grid: (M,) - one program per batch element

    Reference algorithm (Exponential Domain)- Sinkhorn & Knopp (1967):
    ──────────────────────────────────────────────────────
        1. P = exp(A)                        # Ensure positivity
        2. For each iteration:
           - P = P / P.sum(axis=cols)        # Row normalize
           - P = P / P.sum(axis=rows)        # Col normalize
        3. Output: P

        Problem: exp(large) → Inf, exp(-large) → 0, causing overflow/underflow.

    Implementation algorithm (Log Domain):
    ───────────────────────────────────────────────────────
        1. log_u = 0, log_v = 0
        2. For each iteration:
           - log_u = -logsumexp(A + log_v, axis=cols)  # Row normalize
           - log_v = -logsumexp(A + log_u, axis=rows)  # Col normalize
        3. Output: P = exp(A + log_u + log_v)

        Key insight: Division becomes subtraction in log space.
        logsumexp uses stable formula: max(x) + log(Σ exp(x - max(x)))

    """
    batch_idx = tl.program_id(axis=0)

    if batch_idx >= M:
        return

    # Base offset for this batch
    batch_offset = batch_idx * stride_batch

    # Compute flat indices within this batch's matrix
    row_idx = tl.arange(0, N)[:, None]  # (N, 1)
    col_idx = tl.arange(0, N)[None, :]  # (1, N)
    flat_idx = row_idx * stride_row + col_idx * stride_col


    # Load the NxN matrix (raw logits) in log domain
    log_A = tl.load(logits_ptr + batch_offset + flat_idx).to(tl.float32)

    # STEP 2: Initialize log scaling factors
    # Initially u = v = 1 (no scaling), so log(1) = 0, 
    log_u = tl.zeros((N,), dtype=tl.float32)  # Row scalings
    log_v = tl.zeros((N,), dtype=tl.float32)  # Column scalings

    #  Iterate and alternate between row and column normalization.
    for _ in range(NUM_ITERS):
        # Add column scaling: scaled[i,j] = log_A[i,j] + log_v[j]
        scaled_row = log_A + log_v[None, :]  # (N, N)

        # Compute max per row for numerical stability (prevents overflow in exp)
        row_max = tl.max(scaled_row, axis=1)  # (N,)

        # Compute logsumexp per row
        exp_shifted = tl.exp(scaled_row - row_max[:, None])
        row_sum_exp = tl.sum(exp_shifted, axis=1)  # (N,)
        log_row_sums = row_max + tl.log(row_sum_exp)  # (N,)

        # Update row scaling: log_u = -log(row_sum) to normalize rows to 1
        log_u = -log_row_sums

        # Add row scaling: scaled[i,j] = log_A[i,j] + log_u[i]
        scaled_col = log_A + log_u[:, None]  # (N, N)

        # Compute max per column for numerical stability
        col_max = tl.max(scaled_col, axis=0)  # (N,)

        # Compute logsumexp per column
        exp_shifted = tl.exp(scaled_col - col_max[None, :])
        col_sum_exp = tl.sum(exp_shifted, axis=0)  # (N,)
        log_col_sums = col_max + tl.log(col_sum_exp)  # (N,)

        # Update column scaling: log_v = -log(col_sum) to normalize cols to 1
        log_v = -log_col_sums

    # Combine base logits with accumulated scaling factors:
    log_P = log_A + log_u[:, None] + log_v[None, :]
    P = tl.exp(log_P)

    tl.store(out_ptr + batch_offset + flat_idx, P.to(out_ptr.dtype.element_ty))
