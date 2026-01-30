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
    NUM_KSPLIT: tl.constexpr,
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
    is_res_program = ~is_pre_program & ~is_post_program
    
    # Compute local block index within the stream using arithmetic
    # is_post gives 0 or 1, is_res gives 0 or 1
    stream_offset = is_post_program.to(tl.int32) * n_blocks_pre + is_res_program.to(tl.int32) * (n_blocks_pre + n_blocks_post)
    local_pid_n = pid_n - stream_offset
    
    # Local column indices within the stream
    rn_local = local_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Global column index (for bias lookup) - use arithmetic instead of nested tl.where
    col_offset = n * (is_post_program.to(tl.int32) + 2 * is_res_program.to(tl.int32))
    rn_global = rn_local + col_offset
    
    # Output dimension for this stream - use arithmetic
    n_out = n + n * (n - 1) * is_res_program.to(tl.int32)  # n for pre/post, n*n for res
    
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
            mask=(rk[:, None] < K) & (rn_local[None, :] < n_out),
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
    # Use arithmetic blending for alpha selection
    alpha_val = (is_pre_program.to(tl.float32) * alpha_pre +
                 is_post_program.to(tl.float32) * alpha_post +
                 is_res_program.to(tl.float32) * alpha_res)
    
    # Apply Eq 16: H = (1/r) * α * H̃ + b
    out = rsigma[:, None] * alpha_val * acc + bias[None, :]
    
    # Apply stream-specific activation
    # Compute sigmoid once and reuse
    sigmoid_out = tl.sigmoid(out)
    # pre: sigmoid (1x), post: 2*sigmoid, res: identity
    out_activated = tl.where(
        is_pre_program | is_post_program,
        sigmoid_out * (1.0 + is_post_program.to(tl.float32)),
        out
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
        mask=(rm[:, None] < M) & (rn_local[None, :] < n_out),
    )


@triton.jit
def _mhc_fused_split_kernel(
    x_ptr,
    phi_pre_ptr,
    phi_post_ptr,
    phi_res_ptr,
    # Intermediate output buffers (float32)
    acc_pre_ptr,      # (NUM_KSPLIT, M, n)
    acc_post_ptr,     # (NUM_KSPLIT, M, n)
    acc_res_ptr,      # (NUM_KSPLIT, M, n²)
    acc_sq_ptr,       # (NUM_KSPLIT, M)
    M: tl.constexpr,   # rows: x.shape[0] - the batch/sequence dimension
    K: tl.constexpr,   # input features: nC = x.shape[1]
    N: tl.constexpr,   # output features: n² + 2n - total output dimension
    n: tl.constexpr,   # stream parameter controlling manifold dimension
    stride_xm,
    stride_xk,
    stride_phi_pre_k,
    stride_phi_pre_n,
    stride_phi_post_k,
    stride_phi_post_n,
    stride_phi_res_k,
    stride_phi_res_n,
    # Strides for intermediate buffers
    stride_acc_pre_k,
    stride_acc_pre_m,
    stride_acc_pre_n,
    stride_acc_post_k,
    stride_acc_post_m,
    stride_acc_post_n,
    stride_acc_res_k,
    stride_acc_res_m,
    stride_acc_res_n,
    stride_acc_sq_k,
    stride_acc_sq_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
):
    """
    Split-K kernel for mHC - computes partial results for equations 14-15.
    
    Each program processes a portion of the K dimension and writes partial
    dot products and sum-of-squares to intermediate buffers.
    
    Grid structure: (M_blocks, N_blocks_total, NUM_KSPLIT)
    - pid_m: row block index
    - pid_n: stream-aware column block index  
    - pid_k: K-split index
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Row indices
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Compute stream block counts
    n_squared = n * n
    n_blocks_pre = tl.cdiv(n, BLOCK_N)
    n_blocks_post = n_blocks_pre
    
    # Determine stream type from pid_n
    is_pre_program = pid_n < n_blocks_pre
    is_post_program = (pid_n >= n_blocks_pre) & (pid_n < n_blocks_pre + n_blocks_post)
    is_res_program = ~is_pre_program & ~is_post_program
    
    # Compute local block index within the stream
    stream_offset = is_post_program.to(tl.int32) * n_blocks_pre + is_res_program.to(tl.int32) * (n_blocks_pre + n_blocks_post)
    local_pid_n = pid_n - stream_offset
    
    # Local column indices within the stream
    rn_local = local_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Output dimension for this stream
    n_out = n + n * (n - 1) * is_res_program.to(tl.int32)  # n for pre/post, n*n for res
    
    # Calculate K range for this split
    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
    
    # Early exit if this split has no work
    if split_k_start >= K:
        return
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_sq = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Select phi pointers and strides based on stream type
    phi_ptr = tl.where(is_pre_program, phi_pre_ptr,
                       tl.where(is_post_program, phi_post_ptr, phi_res_ptr))
    stride_phi_k = tl.where(is_pre_program, stride_phi_pre_k,
                       tl.where(is_post_program, stride_phi_post_k, stride_phi_res_k))
    stride_phi_n = tl.where(is_pre_program, stride_phi_pre_n,
                       tl.where(is_post_program, stride_phi_post_n, stride_phi_res_n))
    
    # Loop over this split's K range
    k_span = split_k_end - split_k_start
    num_k_iter = tl.cdiv(k_span, BLOCK_K)
    
    for k_idx in range(num_k_iter):
        k = split_k_start + k_idx * BLOCK_K
        rk = k + tl.arange(0, BLOCK_K)
        
        # Load x tile
        x_tile = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
            mask=(rm[:, None] < M) & (rk[None, :] < split_k_end),
            other=0.0,
        )
        
        # Load phi tile
        phi_tile = tl.load(
            phi_ptr + rk[:, None] * stride_phi_k + rn_local[None, :] * stride_phi_n,
            mask=(rk[:, None] < split_k_end) & (rn_local[None, :] < n_out),
            other=0.0,
        )
        
        acc += tl.dot(x_tile, phi_tile)
        x_tile_f32 = x_tile.to(tl.float32)
        acc_sq += tl.sum(x_tile_f32 * x_tile_f32, axis=1)
    
    # Store partial results to intermediate buffers
    # Select output buffer based on stream type
    acc_ptr = tl.where(is_pre_program, acc_pre_ptr,
                       tl.where(is_post_program, acc_post_ptr, acc_res_ptr))
    stride_acc_k = tl.where(is_pre_program, stride_acc_pre_k,
                       tl.where(is_post_program, stride_acc_post_k, stride_acc_res_k))
    stride_acc_m = tl.where(is_pre_program, stride_acc_pre_m,
                       tl.where(is_post_program, stride_acc_post_m, stride_acc_res_m))
    stride_acc_n = tl.where(is_pre_program, stride_acc_pre_n,
                       tl.where(is_post_program, stride_acc_post_n, stride_acc_res_n))
    
    # Store partial dot product
    tl.store(
        acc_ptr + pid_k * stride_acc_k + rm[:, None] * stride_acc_m + rn_local[None, :] * stride_acc_n,
        acc,
        mask=(rm[:, None] < M) & (rn_local[None, :] < n_out),
    )
    
    # Store partial acc_sq only from pre-stream programs (to avoid redundant writes)
    # All streams compute the same acc_sq value, so we only need to store once
    if is_pre_program:
        # Only the first N-block of pre-stream stores acc_sq
        if local_pid_n == 0:
            tl.store(
                acc_sq_ptr + pid_k * stride_acc_sq_k + rm * stride_acc_sq_m,
                acc_sq,
                mask=rm < M,
            )


@triton.jit
def _mhc_fused_reduce_kernel(
    # Intermediate buffers (input)
    acc_pre_ptr,      # (NUM_KSPLIT, M, n)
    acc_post_ptr,     # (NUM_KSPLIT, M, n)
    acc_res_ptr,      # (NUM_KSPLIT, M, n²)
    acc_sq_ptr,       # (NUM_KSPLIT, M)
    # Parameters for post-processing
    alpha_pre,
    alpha_post,
    alpha_res,
    bias_ptr,
    # Final outputs
    out_pre_ptr,
    out_post_ptr,
    out_res_ptr,
    # Dimensions
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    n: tl.constexpr,
    eps: tl.constexpr,
    # Strides for intermediate buffers
    stride_acc_pre_k,
    stride_acc_pre_m,
    stride_acc_pre_n,
    stride_acc_post_k,
    stride_acc_post_m,
    stride_acc_post_n,
    stride_acc_res_k,
    stride_acc_res_m,
    stride_acc_res_n,
    stride_acc_sq_k,
    stride_acc_sq_m,
    # Strides for final outputs
    stride_pre_m,
    stride_pre_n,
    stride_post_m,
    stride_post_n,
    stride_res_m,
    stride_res_n,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
):
    """
    Reduce kernel for mHC - combines partial results and applies post-processing.
    
    Sums partial dot products and sum-of-squares across K-splits, then applies:
    - RMS normalization (Eq 15)
    - Bias + alpha scaling (Eq 16)
    - Stream-specific activations (Eq 17-18)
    
    Grid structure: (M_blocks, N_blocks_total)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row indices
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Compute stream block counts
    n_squared = n * n
    n_blocks_pre = tl.cdiv(n, BLOCK_N)
    n_blocks_post = n_blocks_pre
    
    # Determine stream type from pid_n
    is_pre_program = pid_n < n_blocks_pre
    is_post_program = (pid_n >= n_blocks_pre) & (pid_n < n_blocks_pre + n_blocks_post)
    is_res_program = ~is_pre_program & ~is_post_program
    
    # Compute local block index within the stream
    stream_offset = is_post_program.to(tl.int32) * n_blocks_pre + is_res_program.to(tl.int32) * (n_blocks_pre + n_blocks_post)
    local_pid_n = pid_n - stream_offset
    
    # Local column indices within the stream
    rn_local = local_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Global column index (for bias lookup)
    col_offset = n * (is_post_program.to(tl.int32) + 2 * is_res_program.to(tl.int32))
    rn_global = rn_local + col_offset
    
    # Output dimension for this stream
    n_out = n + n * (n - 1) * is_res_program.to(tl.int32)  # n for pre/post, n*n for res
    
    # Select intermediate buffer based on stream type
    acc_ptr = tl.where(is_pre_program, acc_pre_ptr,
                       tl.where(is_post_program, acc_post_ptr, acc_res_ptr))
    stride_acc_k = tl.where(is_pre_program, stride_acc_pre_k,
                       tl.where(is_post_program, stride_acc_post_k, stride_acc_res_k))
    stride_acc_m = tl.where(is_pre_program, stride_acc_pre_m,
                       tl.where(is_post_program, stride_acc_post_m, stride_acc_res_m))
    stride_acc_n = tl.where(is_pre_program, stride_acc_pre_n,
                       tl.where(is_post_program, stride_acc_post_n, stride_acc_res_n))
    
    # Sum partial dot products across K-splits
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for ks in range(ACTUAL_KSPLIT):
        acc_partial = tl.load(
            acc_ptr + ks * stride_acc_k + rm[:, None] * stride_acc_m + rn_local[None, :] * stride_acc_n,
            mask=(rm[:, None] < M) & (rn_local[None, :] < n_out),
            other=0.0,
        )
        acc += acc_partial
    
    # Sum partial acc_sq across K-splits
    acc_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for ks in range(ACTUAL_KSPLIT):
        acc_sq_partial = tl.load(
            acc_sq_ptr + ks * stride_acc_sq_k + rm * stride_acc_sq_m,
            mask=rm < M,
            other=0.0,
        )
        acc_sq += acc_sq_partial
    
    # RMS NORMALIZATION (Eq 15)
    rms = tl.sqrt(acc_sq / K + eps)
    rsigma = 1.0 / rms
    
    # BIAS + ALPHA SCALING (Eq 16)
    bias = tl.load(bias_ptr + rn_global, mask=rn_global < N, other=0.0).to(tl.float32)

    alpha_val = (is_pre_program.to(tl.float32) * alpha_pre +
                 is_post_program.to(tl.float32) * alpha_post +
                 is_res_program.to(tl.float32) * alpha_res)
    
    # Apply Eq 16: H = (1/r) * α * H̃ + b
    out = rsigma[:, None] * alpha_val * acc + bias[None, :]
    
    # Apply stream-specific activation
    sigmoid_out = tl.sigmoid(out)
    out_activated = tl.where(
        is_pre_program | is_post_program,
        sigmoid_out * (1.0 + is_post_program.to(tl.float32)),
        out
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
        mask=(rm[:, None] < M) & (rn_local[None, :] < n_out),
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
    N: tl.constexpr,            # Matrix size (must be power of 2)
    NUM_ITERS: tl.constexpr,    # Number of Sinkhorn iterations
    BLOCK_M: tl.constexpr,      # Batch elements per program
):
    """
    Log-domain Sinkhorn-Knopp kernel for projecting raw logits onto doubly stochastic matrices.

    Computes doubly stochastic matrix P where all rows and columns sum to 1.

    Grid: (cdiv(M, BLOCK_M),) - one program per BLOCK_M batch elements

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
        logsumexp uses stable formula: max(x) + log2(Σ exp2((x - max(x)) * log2(e))) * ln(2)

    """
    pid_m = tl.program_id(axis=0)
    batch_start = pid_m * BLOCK_M

    # Constants for exp2/log2 conversion
    LOG2_E: tl.constexpr = 1.4426950408889634  # 1/ln(2), for exp(x) = exp2(x * LOG2_E)
    LN_2: tl.constexpr = 0.6931471805599453    # ln(2), for log(x) = log2(x) * LN_2

    # Compute flat indices within each batch's matrix (shared across all batches)
    row_idx = tl.arange(0, N)[:, None]  # (N, 1)
    col_idx = tl.arange(0, N)[None, :]  # (1, N)
    flat_idx = row_idx * stride_row + col_idx * stride_col

    # Loop over batch elements in this block
    for batch_local in range(BLOCK_M):
        batch_idx = batch_start + batch_local
        if batch_idx < M:
            # Base offset for this batch
            batch_offset = batch_idx * stride_batch

            # Load the NxN matrix (raw logits) in log domain
            log_A = tl.load(logits_ptr + batch_offset + flat_idx).to(tl.float32)

            # Initialize log scaling factors
            # Initially u = v = 1 (no scaling), so log(1) = 0,
            log_u = tl.zeros((N,), dtype=tl.float32)  # Row scalings
            log_v = tl.zeros((N,), dtype=tl.float32)  # Column scalings

            # Iterate and alternate between row and column normalization.
            for _ in range(NUM_ITERS):
                # Add column scaling: scaled[i,j] = log_A[i,j] + log_v[j]
                scaled_row = log_A + log_v[None, :]  # (N, N)

                row_max = tl.max(scaled_row, axis=1)  # (N,)

                # Compute logsumexp per row
                exp_shifted = tl.exp2((scaled_row - row_max[:, None]) * LOG2_E)
                row_sum_exp = tl.sum(exp_shifted, axis=1)  # (N,)
                log_row_sums = row_max + tl.log2(row_sum_exp) * LN_2  # (N,)

                # Update row scaling: log_u = -log(row_sum) to normalize rows to 1
                log_u = -log_row_sums

                # Add row scaling: scaled[i,j] = log_A[i,j] + log_u[i]
                scaled_col = log_A + log_u[:, None]  # (N, N)

                col_max = tl.max(scaled_col, axis=0)  # (N,)

                # Compute logsumexp per column
                exp_shifted = tl.exp2((scaled_col - col_max[None, :]) * LOG2_E)
                col_sum_exp = tl.sum(exp_shifted, axis=0)  # (N,)
                log_col_sums = col_max + tl.log2(col_sum_exp) * LN_2  # (N,)

                # Update column scaling: log_v = -log(col_sum) to normalize cols to 1
                log_v = -log_col_sums

            # Combine base logits with accumulated scaling factors:
            log_P = log_A + log_u[:, None] + log_v[None, :]
            P = tl.exp2(log_P * LOG2_E)

            tl.store(out_ptr + batch_offset + flat_idx, P.to(out_ptr.dtype.element_ty))


@triton.jit
def _mhc_lite_fused_split_kernel(
    x_ptr,
    phi_pre_ptr,
    phi_post_ptr,
    phi_res_ptr,  # Shape: (nC, n!) for mHC-Lite
    # Intermediate output buffers (float32)
    acc_pre_ptr,      # (NUM_KSPLIT, M, n)
    acc_post_ptr,     # (NUM_KSPLIT, M, n)
    acc_res_ptr,      # (NUM_KSPLIT, M, n!) - logits for permutation weights
    acc_sq_ptr,       # (NUM_KSPLIT, M)
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,            # Total output: n + n + n²
    n: tl.constexpr,
    N_FACTORIAL: tl.constexpr,  # n!
    stride_xm,
    stride_xk,
    stride_phi_pre_k,
    stride_phi_pre_n,
    stride_phi_post_k,
    stride_phi_post_n,
    stride_phi_res_k,
    stride_phi_res_n,  # n! dimension
    # Strides for intermediate buffers
    stride_acc_pre_k,
    stride_acc_pre_m,
    stride_acc_pre_n,
    stride_acc_post_k,
    stride_acc_post_m,
    stride_acc_post_n,
    stride_acc_res_k,
    stride_acc_res_m,
    stride_acc_res_n,  # n! dimension
    stride_acc_sq_k,
    stride_acc_sq_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
):
    """
    Split-K kernel for mHC-Lite - computes partial matmuls.
    
    Key difference from standard mHC: phi_res has shape (nC, n!) instead of (nC, n²).
    This produces logits for the n! permutation weights that will be softmaxed
    in the reduce kernel.
    
    Grid structure: (M_blocks, N_blocks_total, NUM_KSPLIT)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Compute stream block counts
    n_squared = n * n
    n_blocks_pre = tl.cdiv(n, BLOCK_N)
    n_blocks_post = n_blocks_pre
    n_blocks_res = tl.cdiv(N_FACTORIAL, BLOCK_N)  # n! dimension for lite
    
    # Stream determination using clearer logic
    threshold_post = n_blocks_pre + n_blocks_post
    is_pre_program = pid_n < n_blocks_pre
    is_post_program = (pid_n >= n_blocks_pre) & (pid_n < threshold_post)
    is_res_program = pid_n >= threshold_post
    
    # Compute local block index within stream
    stream_offset = is_post_program.to(tl.int32) * n_blocks_pre + is_res_program.to(tl.int32) * threshold_post
    local_pid_n = pid_n - stream_offset
    
    rn_local = local_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Output dimension for this stream (pre-compute for better performance)
    n_out = tl.where(is_res_program, N_FACTORIAL, n)  # n! for res, n for pre/post
    
    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
    
    if split_k_start >= K:
        return
    
    # Pre-compute phi pointer and strides based on stream type (once, not per iteration)
    phi_ptr = tl.where(is_pre_program, phi_pre_ptr,
                       tl.where(is_post_program, phi_post_ptr, phi_res_ptr))
    stride_phi_k = tl.where(is_pre_program, stride_phi_pre_k,
                       tl.where(is_post_program, stride_phi_post_k, stride_phi_res_k))
    stride_phi_n = tl.where(is_pre_program, stride_phi_pre_n,
                       tl.where(is_post_program, stride_phi_post_n, stride_phi_res_n))
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    k_span = split_k_end - split_k_start
    num_k_iter = tl.cdiv(k_span, BLOCK_K)
    
    # Pre-compute common masks and offsets
    rm_mask = rm < M
    rn_mask = rn_local < n_out
    
    for k_idx in range(num_k_iter):
        k = split_k_start + k_idx * BLOCK_K
        rk = k + tl.arange(0, BLOCK_K)
        rk_mask = rk < split_k_end
        
        # Load x tile with optimized masking
        x_tile = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
            mask=rm_mask[:, None] & rk_mask[None, :],
            other=0.0,
        )
        
        # Load phi tile with optimized masking
        phi_tile = tl.load(
            phi_ptr + rk[:, None] * stride_phi_k + rn_local[None, :] * stride_phi_n,
            mask=rk_mask[:, None] & rn_mask[None, :],
            other=0.0,
        )
        
        # Accumulate matmul result
        acc += tl.dot(x_tile, phi_tile)
        
        # Accumulate squared norm (convert once)
        x_tile_f32 = x_tile.to(tl.float32)
        acc_sq += tl.sum(x_tile_f32 * x_tile_f32, axis=1)
    
    # Select output accumulator buffer based on stream type
    acc_ptr = tl.where(is_pre_program, acc_pre_ptr,
                       tl.where(is_post_program, acc_post_ptr, acc_res_ptr))
    stride_acc_k = tl.where(is_pre_program, stride_acc_pre_k,
                       tl.where(is_post_program, stride_acc_post_k, stride_acc_res_k))
    stride_acc_m = tl.where(is_pre_program, stride_acc_pre_m,
                       tl.where(is_post_program, stride_acc_post_m, stride_acc_res_m))
    stride_acc_n = tl.where(is_pre_program, stride_acc_pre_n,
                       tl.where(is_post_program, stride_acc_post_n, stride_acc_res_n))
    
    # Store accumulated results with pre-computed mask
    tl.store(
        acc_ptr + pid_k * stride_acc_k + rm[:, None] * stride_acc_m + rn_local[None, :] * stride_acc_n,
        acc,
        mask=rm_mask[:, None] & rn_mask[None, :],
    )
    
    # Store acc_sq only once per M-block (from first pre program)
    if is_pre_program and local_pid_n == 0:
        tl.store(
            acc_sq_ptr + pid_k * stride_acc_sq_k + rm * stride_acc_sq_m,
            acc_sq,
            mask=rm_mask,
        )


@triton.jit
def _mhc_lite_fused_reduce_kernel(
    # Intermediate buffers
    acc_pre_ptr,      # (NUM_KSPLIT, M, n)
    acc_post_ptr,     # (NUM_KSPLIT, M, n)
    acc_res_ptr,      # (NUM_KSPLIT, M, n!) - logits
    acc_sq_ptr,       # (NUM_KSPLIT, M)
    # Permutation matrices
    perm_mats_ptr,    # (n!, n, n)
    # Parameters
    alpha_pre,
    alpha_post,
    alpha_res,
    bias_ptr,
    # Final outputs
    out_pre_ptr,
    out_post_ptr,
    out_res_ptr,      # (M, n²)
    # Dimensions
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    n: tl.constexpr,
    N_FACTORIAL: tl.constexpr,
    N_FACTORIAL_POW2: tl.constexpr,
    eps: tl.constexpr,
    # Intermediate buffer strides
    stride_acc_pre_k,
    stride_acc_pre_m,
    stride_acc_pre_n,
    stride_acc_post_k,
    stride_acc_post_m,
    stride_acc_post_n,
    stride_acc_res_k,
    stride_acc_res_m,
    stride_acc_res_n,
    stride_acc_sq_k,
    stride_acc_sq_m,
    # Permutation matrix strides
    stride_perm_idx,
    stride_perm_row,
    stride_perm_col,
    # Output strides
    stride_pre_m,
    stride_pre_n,
    stride_post_m,
    stride_post_n,
    stride_res_m,
    stride_res_n,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
):
    """
    Reduce kernel for mHC-Lite - reduces partial results and applies Sinkhorn-Knopp fusion.
    
    For pre/post streams: standard reduce + RMS + bias + activation
    For res stream: reduce → softmax over n! → weighted permutation sum → doubly stochastic
    
    This implements the core mHC-Lite innovation: H^res = Σ softmax(logits)_k * P_k
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    n_squared = n * n
    n_blocks_pre = tl.cdiv(n, BLOCK_N)
    n_blocks_post = n_blocks_pre
    
    is_pre_program = pid_n < n_blocks_pre
    is_post_program = (pid_n >= n_blocks_pre) & (pid_n < n_blocks_pre + n_blocks_post)
    is_res_program = ~is_pre_program & ~is_post_program
    
    stream_offset = is_post_program.to(tl.int32) * n_blocks_pre + is_res_program.to(tl.int32) * (n_blocks_pre + n_blocks_post)
    local_pid_n = pid_n - stream_offset
    
    rn_local = local_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    col_offset = n * (is_post_program.to(tl.int32) + 2 * is_res_program.to(tl.int32))
    rn_global = rn_local + col_offset
    
    # Reduce acc_sq
    acc_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for ks in range(ACTUAL_KSPLIT):
        acc_sq_partial = tl.load(
            acc_sq_ptr + ks * stride_acc_sq_k + rm * stride_acc_sq_m,
            mask=rm < M,
            other=0.0,
        )
        acc_sq += acc_sq_partial
    
    rms = tl.sqrt(acc_sq / K + eps)
    rsigma = 1.0 / rms
    
    if is_res_program:
        # mHC-Lite path: reduce logits → softmax → weighted permutation sum
        # STEP 1: Reduce logits across K-splits (M, n!)
        n_blocks_res = tl.cdiv(n_squared, BLOCK_N)
        if local_pid_n >= n_blocks_res:
            return
        
        # Allocate for n! logits per batch
        logits = tl.zeros([BLOCK_M, N_FACTORIAL_POW2], dtype=tl.float32) + float('-inf')
        
        # Reduce partial logits
        for perm_idx in tl.static_range(N_FACTORIAL):
            logit_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
            for ks in range(ACTUAL_KSPLIT):
                logit_partial = tl.load(
                    acc_res_ptr + ks * stride_acc_res_k + rm * stride_acc_res_m + perm_idx * stride_acc_res_n,
                    mask=rm < M,
                    other=0.0,
                )
                logit_acc += logit_partial
            
            # Apply RMSNorm + alpha + bias
            bias_val = tl.load(bias_ptr + (n + n + perm_idx), mask=(n + n + perm_idx) < N, other=0.0).to(tl.float32)
            logit_normalized = rsigma * alpha_res * logit_acc + bias_val
            
            perm_indices = tl.arange(0, N_FACTORIAL_POW2)
            logits = tl.where((perm_indices == perm_idx)[None, :], logit_normalized[:, None], logits)
        
        # STEP 2: Softmax over n! dimension
        LOG2_E: tl.constexpr = 1.4426950408889634
        logits_max = tl.max(logits, axis=1, keep_dims=True)
        exp_shifted = tl.exp2((logits - logits_max) * LOG2_E)
        sum_exp = tl.sum(exp_shifted, axis=1, keep_dims=True)
        weights = exp_shifted / sum_exp  # (BLOCK_M, N_FACTORIAL_POW2)
        
        # STEP 3: Weighted sum of permutation matrices
        # Compute which n² elements this program handles
        elem_start = local_pid_n * BLOCK_N
        elem_indices = elem_start + tl.arange(0, BLOCK_N)
        elem_mask = elem_indices < n_squared
        
        row_coords = elem_indices // n
        col_coords = elem_indices % n
        
        output = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for perm_idx in tl.static_range(N_FACTORIAL):
            perm_ptrs = perm_mats_ptr + perm_idx * stride_perm_idx + row_coords * stride_perm_row + col_coords * stride_perm_col
            perm_vals = tl.load(perm_ptrs, mask=elem_mask, other=0.0).to(tl.float32)
            
            perm_indices = tl.arange(0, N_FACTORIAL_POW2)
            weight_mask = perm_indices == perm_idx
            perm_weights = tl.sum(tl.where(weight_mask[None, :], weights, 0.0), axis=1)
            
            output = tl.fma(perm_weights[:, None], perm_vals[None, :], output)
        
        # Store H^res (already doubly stochastic)
        out_ptrs = out_res_ptr + rm[:, None] * stride_res_m + elem_indices[None, :] * stride_res_n
        store_mask = (rm < M)[:, None] & elem_mask[None, :]
        tl.store(out_ptrs, output.to(out_res_ptr.dtype.element_ty), mask=store_mask)
        
    else:
        # Standard mHC path for pre/post streams
        n_out = n
        acc_ptr = tl.where(is_pre_program, acc_pre_ptr, acc_post_ptr)
        stride_acc_k = tl.where(is_pre_program, stride_acc_pre_k, stride_acc_post_k)
        stride_acc_m = tl.where(is_pre_program, stride_acc_pre_m, stride_acc_post_m)
        stride_acc_n = tl.where(is_pre_program, stride_acc_pre_n, stride_acc_post_n)
        
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for ks in range(ACTUAL_KSPLIT):
            acc_partial = tl.load(
                acc_ptr + ks * stride_acc_k + rm[:, None] * stride_acc_m + rn_local[None, :] * stride_acc_n,
                mask=(rm[:, None] < M) & (rn_local[None, :] < n_out),
                other=0.0,
            )
            acc += acc_partial
        
        bias = tl.load(bias_ptr + rn_global, mask=rn_global < N, other=0.0).to(tl.float32)
        alpha_val = tl.where(is_pre_program, alpha_pre, alpha_post)
        
        out = rsigma[:, None] * alpha_val * acc + bias[None, :]
        
        sigmoid_out = tl.sigmoid(out)
        out_activated = tl.where(is_pre_program, sigmoid_out, sigmoid_out * 2.0)

        out_ptr = tl.where(is_pre_program, out_pre_ptr, out_post_ptr)
        stride_out_m = tl.where(is_pre_program, stride_pre_m, stride_post_m)
        stride_out_n = tl.where(is_pre_program, stride_pre_n, stride_post_n)
        
        tl.store(
            out_ptr + rm[:, None] * stride_out_m + rn_local[None, :] * stride_out_n,
            out_activated,
            mask=(rm[:, None] < M) & (rn_local[None, :] < n_out),
        )
