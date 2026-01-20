# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernel for mHC (manifold-constrained Hyper Connection) operations.

Implements equations 14-18 from the mHC paper in a single fused kernel:
- Eq 14: H̃ = x̃φ (matrix multiplication)
- Eq 15: r = ||x̃||₂ / √(nC) (RMS normalization)
- Eq 16: [H^pre, H^post, H^res] = 1/r [α^pre·H̃^pre, α^post·H̃^post, α^res·H̃^res] + b
- Eq 17: H^pre = σ(H^pre)
- Eq 18: H^post = 2σ(H^post)
- H^res: identity (no activation, ready for equation 19: Sinkhorn-Knopp)

Single fused kernel minimizes memory traffic and kernel launch overhead.
"""

import triton
import triton.language as tl


@triton.jit
def _mhc_fused_kernel(
    x_ptr,
    phi_ptr,
    alpha_pre,
    alpha_post,
    alpha_res,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,   # rows: x.shape[0] - the batch/sequence dimension. Represents how many input vectors we're processing
    K: tl.constexpr,   # input features: nC = x.shape[1] - must match phi.shape[0]. Called nC in the paper (n × C where C is some latent dimension)
    N: tl.constexpr,   # output features: n² + 2n - total output dimension split into 3 streams (pre: n², post: n, res: n). Must match phi.shape[1] and bias length
    n: tl.constexpr,   # stream parameter: n - Hyperparameter from paper controlling manifold dimension. Determines stream sizes
    eps: tl.constexpr, # epsilon for numerical stability in RMSNorm
    stride_xm,
    stride_xk,
    stride_phik,
    stride_phin,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel for equations 14-18.
    
    Computes H = [H^pre, H^post, H^res] where:
    - H^pre: n elements with sigmoid activation (H^{pre} ∈ ℝ^{1×n})
    - H^post: n elements with 2*sigmoid activation (H^{post} ∈ ℝ^{1×n})
    - H^res: n² elements with identity (no activation) (H^{res} ∈ ℝ^{n×n})
    
    All operations fused in a single kernel pass for maximum efficiency.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column indices
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Eq 14: Compute matrix multiplication H̃ = x̃φ with deferred normalization
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)

        x_tile = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )

        phi_tile = tl.load(
            phi_ptr + rk[:, None] * stride_phik + rn[None, :] * stride_phin,
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(x_tile, phi_tile)
    
    # Eq 15: Compute RMS norm r = ||x̃||₂ / √(nC)
    # This computes r = sqrt(sum(x²) / K) where K = nC
    # The .to(tl.float32) is critical for numerical stability in accumulation
    acc_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )
        x_tile_f32 = x_tile.to(tl.float32)
        acc_sq += tl.sum(x_tile_f32 * x_tile_f32, axis=1)

    rms = tl.sqrt(acc_sq / K + eps)
    # Performance optimization: compute 1/r once instead of dividing N times
    # Division is ~10x slower than multiplication on GPUs
    rsigma = 1.0 / rms
    
    # Load bias
    bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    
    # Eq 16: Apply stream-specific scaling and bias
    # Output is split into 3 contiguous streams:
    #   Pre-stream:  indices [0, n) - n elements for manifold projection (H^{pre} ∈ ℝ^{1×n})
    #   Post-stream: indices [n, 2n) - n elements for post-processing (H^{post} ∈ ℝ^{1×n})
    #   Res-stream:  indices [2n, 2n+n²) - n² elements for residual connections (H^{res} ∈ ℝ^{n×n})
    n_pre_end = n                # End of pre-stream
    n_post_end = 2 * n           # End of post-stream
    
    # Create boolean masks to identify which stream each output column belongs to
    is_pre = rn < n_pre_end
    is_post = (rn >= n_pre_end) & (rn < n_post_end)
    
    # Select the appropriate scaling factor (alpha) for each stream
    # This creates a vector where each element has its stream-specific alpha
    alpha = tl.where(is_pre, alpha_pre, 
                     tl.where(is_post, alpha_post, alpha_res))
    
    # Apply Eq 16: H = (1/r) * α * H̃ + b
    # rsigma[:, None] broadcasts 1/r across columns (per-row normalization)
    # alpha is per-column (stream-specific scaling)
    # acc is the matrix product H̃ from Eq 14
    out = rsigma[:, None] * alpha * acc + bias[None, :]
    
    # Apply stream-specific activations
    # Pre-stream (Eq 17): H^pre = σ(H^pre) - sigmoid activation
    out = tl.where(is_pre[None, :], tl.sigmoid(out), out)
    
    # Post-stream (Eq 18): H^post = 2σ(H^post) - scaled sigmoid activation
    out = tl.where(is_post[None, :], 2.0 * tl.sigmoid(out), out)
    
    # Res-stream: H^res remains unchanged (identity activation)
    # This preserves the values for subsequent Sinkhorn-Knopp normalization (Eq 19)

    # Store result (cast back to input dtype)
    tl.store(
        out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on,
        out,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


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
