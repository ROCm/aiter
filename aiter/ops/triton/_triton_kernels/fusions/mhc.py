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
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
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
def _sinkhorn_knopp_lite(
    # Pointers
    logits_ptr,     # Input: (M, n_factorial) raw logits for permutation weights
    perm_mats_ptr,  # Input: (n_factorial, N, N) pre-computed permutation matrices
    out_ptr,        # Output: (M, N, N) doubly stochastic matrices
    # Dimensions
    M,              # Batch size (number of matrices)
    N: tl.constexpr,            # Matrix size
    N_FACTORIAL: tl.constexpr,  # Number of permutations (N!)
    N_FACTORIAL_POW2: tl.constexpr,  # Next power of 2 >= N_FACTORIAL
    # Strides
    stride_logits_m,      # Stride for batch dimension in logits
    stride_logits_perm,   # Stride for permutation dimension in logits
    stride_perm_idx,      # Stride for permutation index in perm_mats
    stride_perm_row,      # Stride for row in perm_mats
    stride_perm_col,      # Stride for col in perm_mats
    stride_out_batch,     # Stride for batch dimension in output
    stride_out_row,       # Stride for row dimension in output
    stride_out_col,       # Stride for column dimension in output
    # Meta-parameters
    BLOCK_M: tl.constexpr,      # Batch elements per program
    BLOCK_SIZE: tl.constexpr,   # Block size for matrix element processing
):
    """
    Zero-iteration Sinkhorn-Knopp kernel using permutation matrix basis.
    
    Instead of iteratively normalizing matrices, this directly parameterizes
    doubly stochastic matrices as convex combinations of permutation matrices,
    based on the Birkhoff-von Neumann theorem.
    
    Mathematical foundation:
        Any doubly stochastic matrix M can be written as:
        M = Σ_{k=1}^{n!} λ_k * P_k
        where P_k are permutation matrices and λ_k ≥ 0, Σλ_k = 1
    
    Algorithm:
        1. Apply softmax to logits to get weights λ: (M, n!)
        2. Compute weighted sum: M[b] = Σ_k λ[b,k] * P_k
        3. Result is automatically doubly stochastic (no iterations!)
    
    Advantages:
        - Zero iterations (one-shot computation)
        - Exact doubly stochastic property
        - Faster than iterative Sinkhorn-Knopp
    
    Limitations:
        - Requires storing n! permutation matrices
        - Only practical for small N (typically N=4, where 4!=24)
        - For N=5: 5!=120, N=6: 6!=720 (memory intensive)
    
    Grid: (cdiv(M, BLOCK_M),) - one program per BLOCK_M batch elements
    
    Reference: "mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations"
               https://arxiv.org/abs/2601.05732
    """
    pid_m = tl.program_id(axis=0)
    batch_start = pid_m * BLOCK_M
    
    # Constants for exp2/log2 conversion (for numerical stability in softmax)
    LOG2_E: tl.constexpr = 1.4426950408889634  # 1/ln(2)
    LN_2: tl.constexpr = 0.6931471805599453    # ln(2)
    
    # Compute flat indices for NxN matrices (shared across all batches)
    row_idx = tl.arange(0, N)[:, None]  # (N, 1)
    col_idx = tl.arange(0, N)[None, :]  # (1, N)
    flat_idx = row_idx * stride_out_row + col_idx * stride_out_col
    
    # Pre-compute permutation indices for weight extraction
    perm_indices = tl.arange(0, N_FACTORIAL_POW2)
    perm_mask = perm_indices < N_FACTORIAL
    
    # Loop over batch elements in this block
    for batch_local in range(BLOCK_M):
        batch_idx = batch_start + batch_local
        if batch_idx < M:
            # Load logits for this batch: (n_factorial,)
            # Note: tl.arange requires power of 2, so we use N_FACTORIAL_POW2 and mask
            logits_offset = batch_idx * stride_logits_m + perm_indices * stride_logits_perm
            logits = tl.load(
                logits_ptr + logits_offset,
                mask=perm_mask,
                other=float('-inf')  # Masked values won't contribute after softmax
            ).to(tl.float32)
            
            # Apply softmax to get weights λ (ensures Σλ = 1, λ ≥ 0)
            # Using log-domain for numerical stability
            logits_max = tl.max(logits)
            exp_shifted = tl.exp2((logits - logits_max) * LOG2_E)
            sum_exp = tl.sum(exp_shifted)
            weights = exp_shifted / sum_exp  # (n_factorial_pow2,) but only first N_FACTORIAL matter
            
            # Initialize output matrix with zeros
            out_matrix = tl.zeros((N, N), dtype=tl.float32)
            
            # Compute weighted sum: M = Σ_k λ_k * P_k
            # Unroll the loop over N_FACTORIAL permutations for better performance
            # Triton will optimize this at compile time since N_FACTORIAL is constexpr
            for perm_idx in tl.static_range(N_FACTORIAL):
                # Extract weight for this permutation using optimized indexing
                # Create a mask for exactly this permutation index
                weight_mask = perm_indices == perm_idx
                weight_val = tl.sum(tl.where(weight_mask, weights, 0.0))
                
                # Load permutation matrix P_k: (N, N)
                # Use contiguous memory access pattern for better cache utilization
                perm_base = perm_idx * stride_perm_idx
                perm_matrix = tl.load(
                    perm_mats_ptr + perm_base + flat_idx,
                ).to(tl.float32)
                
                # Fused multiply-add: accumulate weighted contribution
                # This is optimized to a single FMA instruction per element
                out_matrix = tl.fma(weight_val, perm_matrix, out_matrix)
            
            # Store result with coalesced memory access
            batch_offset = batch_idx * stride_out_batch
            tl.store(
                out_ptr + batch_offset + flat_idx, 
                out_matrix.to(out_ptr.dtype.element_ty)
            ) 
