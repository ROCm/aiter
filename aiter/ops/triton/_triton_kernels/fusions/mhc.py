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
    phi_pre_ptr,
    phi_post_ptr,
    phi_res_ptr,
    alpha_pre,
    alpha_post,
    alpha_res,
    bias_ptr,
    out_pre_ptr,
    out_post_ptr,
    out_res_ptr,
    M: tl.constexpr,   # rows: x.shape[0] - the batch/sequence dimension
    K: tl.constexpr,   # input features: nC = x.shape[1]
    N: tl.constexpr,   # output features: n² + 2n - total output dimension
    n: tl.constexpr,   # stream parameter controlling manifold dimension
    eps: tl.constexpr, # epsilon for numerical stability in RMSNorm
    stride_xm,
    stride_xk,
    stride_phi_pre_k,
    stride_phi_pre_n,
    stride_phi_post_k,
    stride_phi_post_n,
    stride_phi_res_k,
    stride_phi_res_n,
    stride_pre_m,
    stride_pre_n,
    stride_post_m,
    stride_post_n,
    stride_res_m,
    stride_res_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel for equations 14-18 with stream-aware grid.
    
    Computes three separate outputs:
    - H^pre: (M, n) with sigmoid activation (Eq 17)
    - H^post: (M, n) with 2*sigmoid activation (Eq 18)
    - H^res: (M, n²) with identity (no activation, for Eq 19)
    
    Grid structure:
    - The grid is organized per-stream so each program processes exactly one stream
    - pid_n maps to: [0, n_blocks_pre) = pre, [n_blocks_pre, n_blocks_pre+post) = post, rest = res
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row indices
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Compute stream block counts
    n_squared = n * n
    n_blocks_pre = tl.cdiv(n, BLOCK_N)
    n_blocks_post = n_blocks_pre # post stream has the same number of blocks as pre stream
    
    # Determine stream type from pid_n, each program processes exactly one stream
    is_pre_program = pid_n < n_blocks_pre
    is_post_program = (pid_n >= n_blocks_pre) & (pid_n < n_blocks_pre + n_blocks_post)
    # is_res_program implied when neither pre nor post
    
    # Compute local block index within the stream
    local_pid_n = tl.where(is_pre_program, pid_n,
                           tl.where(is_post_program, pid_n - n_blocks_pre,
                                    pid_n - n_blocks_pre - n_blocks_post))
    
    # Local column indices within the stream
    rn_local = local_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Global column index (for bias lookup)
    col_offset = tl.where(is_pre_program, 0, tl.where(is_post_program, n, 2 * n))
    rn_global = rn_local + col_offset
    
    # Output dimension for this stream
    n_out = tl.where(is_pre_program, n, tl.where(is_post_program, n, n_squared))
    
    # MATMUL (Eq 14) + RMS ACCUMULATION (Eq 15)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_sq = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Select phi pointers and strides based on stream type
    phi_ptr = tl.where(is_pre_program, phi_pre_ptr,
                       tl.where(is_post_program, phi_post_ptr, phi_res_ptr))
    stride_phi_k = tl.where(is_pre_program, stride_phi_pre_k,
                       tl.where(is_post_program, stride_phi_post_k, stride_phi_res_k))
    stride_phi_n = tl.where(is_pre_program, stride_phi_pre_n,
                       tl.where(is_post_program, stride_phi_post_n, stride_phi_res_n))
    
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        
        # Load x tile
        x_tile = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )
        
        # SINGLE PHI LOAD         
        phi_tile = tl.load(
            phi_ptr + rk[:, None] * stride_phi_k + rn_local[None, :] * stride_phi_n,
            mask=(rk[:, None] < K) & (rn_local[None, :] >= 0) & (rn_local[None, :] < n_out),
            other=0.0,
        )
        
        acc += tl.dot(x_tile, phi_tile)
        x_tile_f32 = x_tile.to(tl.float32)
        acc_sq += tl.sum(x_tile_f32 * x_tile_f32, axis=1)
    
    # RMS NORMALIZATION (Eq 15)
    rms = tl.sqrt(acc_sq / K + eps)
    rsigma = 1.0 / rms
    
    # BIAS + ALPHA SCALING (Eq 16)
    bias = tl.load(bias_ptr + rn_global, mask=rn_global < N, other=0.0).to(tl.float32)
    alpha_val = tl.where(is_pre_program, alpha_pre,
                        tl.where(is_post_program, alpha_post, alpha_res))
    
    # Apply Eq 16: H = (1/r) * α * H̃ + b
    out = rsigma[:, None] * alpha_val * acc + bias[None, :]
    
    # Apply stream-specific activation
    out_activated = tl.where(
        is_pre_program, tl.sigmoid(out),
        tl.where(is_post_program, 2.0 * tl.sigmoid(out), out)
    )

    out_ptr = tl.where(is_pre_program, out_pre_ptr,
                       tl.where(is_post_program, out_post_ptr, out_res_ptr))
    stride_out_m = tl.where(is_pre_program, stride_pre_m,
                       tl.where(is_post_program, stride_post_m, stride_res_m))
    stride_out_n = tl.where(is_pre_program, stride_pre_n,
                       tl.where(is_post_program, stride_post_n, stride_res_n))
    
    tl.store(
        out_ptr + rm[:, None] * stride_out_m + rn_local[None, :] * stride_out_n,
        out_activated,
        mask=(rm[:, None] < M) & (rn_local[None, :] >= 0) & (rn_local[None, :] < n_out),
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

    # Initialize log scaling factors
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
