# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton kernel for mHC (manifold-constrained Hyper Connection) operations."""

import triton
import triton.language as tl


@triton.jit
def _compute_hres_mhc_lite(
    out,  # (BLOCK_M, n_factorial) logits - padded to BLOCK_N for Triton
    perm_ptr,
    stride_perm_k,
    stride_perm_ij,
    n_squared: tl.constexpr,
    n_factorial: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Compute exact doubly stochastic H_res via Birkhoff-von Neumann theorem.

    Computes H_res = softmax(logits) @ P where:
    - logits: (BLOCK_M, n_factorial) raw weights for each permutation matrix
    - P: (n_factorial, n_squared) stacked permutation matrices
    - Result: (BLOCK_M, n_squared) convex combination = exact doubly stochastic

    """
    LOG2_E: tl.constexpr = 1.4426950408889634
    LN_2: tl.constexpr = 0.6931471805599453

    col_idx = tl.arange(0, BLOCK_N)
    n_factorial_mask = col_idx[None, :] < n_factorial
    out_masked = tl.where(n_factorial_mask, out, float("-inf"))

    # Log-domain softmax: logsumexp then subtract
    out_max = tl.max(out_masked, axis=1, keep_dims=True)
    out_shifted = out_masked - out_max
    exp_shifted = tl.exp2(out_shifted * LOG2_E)
    sum_exp = tl.sum(exp_shifted, axis=1, keep_dims=True)
    log_sum_exp = out_max + tl.log2(sum_exp) * LN_2

    log_alpha = out_masked - log_sum_exp
    alpha = tl.exp2(log_alpha * LOG2_E)

    # Load permutation matrices P: (n_factorial, n_squared) padded to (BLOCK_N, BLOCK_N)
    row_idx = tl.arange(0, BLOCK_N)[:, None]
    col_idx_2d = tl.arange(0, BLOCK_N)[None, :]
    perm_mask = (row_idx < n_factorial) & (col_idx_2d < n_squared)
    P = tl.load(
        perm_ptr + row_idx * stride_perm_k + col_idx_2d * stride_perm_ij,
        mask=perm_mask,
        other=0.0,
    )

    return tl.dot(alpha, P)


@triton.jit
def _mhc_fused_kernel(
    x_ptr,
    phi_ptr,  # Unified phi: (K, n + n + n_res)
   
    alpha_pre,
    alpha_post,
    alpha_res,
    bias_ptr,
    out_ptr,  # Unified output: (M, n + n + n_squared)
   
    perm_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    n: tl.constexpr,
    n_squared: tl.constexpr,
    n_factorial: tl.constexpr,
    eps: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_phi_k,  # Stride for K dimension
    stride_phi_n,  # Stride for N dimension (total_cols)
   
    stride_out_m,  # Stride for M dimension
    stride_out_n,  # Stride for N dimension (total_cols)
   
    stride_perm_k,
    stride_perm_ij,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    HRES_LITE_MODE: tl.constexpr,  # 0=sinkhorn (identity), 1=lite (softmax+perm)
):
    """
    Fused kernel for equations 14-18 with stream-aware grid.

    Computes three separate outputs:
    - H^pre: (M, n) with sigmoid activation (Eq 17)
    - H^post: (M, n) with 2*sigmoid activation (Eq 18)
    - H^res: (M, n²) - activation depends on HRES_LITE_MODE:
        - Sinkhorn mode (HRES_LITE_MODE=False):raw logits
        - Lite mode (HRES_LITE_MODE=True): softmax + permutation combination,

    Grid structure:
    - The grid is organized per-stream so each program processes exactly one stream
    - pid_n maps to: [0, n_blocks_pre) = pre, [n_blocks_pre, n_blocks_pre+post) = post, rest = res
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    n_blocks_pre = tl.cdiv(n, BLOCK_N)
    n_blocks_post = n_blocks_pre

    # Determine stream type from pid_n, each program processes exactly one stream
    is_pre_program = pid_n < n_blocks_pre
    is_post_program = (pid_n >= n_blocks_pre) & (pid_n < n_blocks_pre + n_blocks_post)
    is_res_program = ~is_pre_program & ~is_post_program
    is_post_f32 = is_post_program.to(tl.float32)
    is_post_i32 = is_post_program.to(tl.int32)
    is_res_i32 = is_res_program.to(tl.int32)

    stream_offset = is_post_i32 * n_blocks_pre + is_res_i32 * (
        n_blocks_pre + n_blocks_post
    )
    local_pid_n = pid_n - stream_offset

    rn_local = local_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    n_out_res = n_squared + (n_factorial - n_squared) * is_res_i32
    n_out = n + (n_out_res - n) * is_res_i32

    col_offset = n * (is_post_i32 + 2 * is_res_i32)
    rn_global = rn_local + col_offset

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_sq = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Compute phi column offset in unified tensor layout: [pre: 0..n-1, post: n..2n-1, res: 2n..2n+n_res-1]
    phi_col_start = tl.where(
        is_pre_program,
        0,
        tl.where(is_post_program, n, 2 * n)
    )
    # Unified phi tensor - strides are the same for all streams

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)

        x_tile = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )

        phi_col_offset = phi_col_start + rn_local
        phi_tile = tl.load(
            phi_ptr + rk[:, None] * stride_phi_k + phi_col_offset[None, :] * stride_phi_n,
            mask=(rk[:, None] < K) & (rn_local[None, :] < n_out),
            other=0.0,
        )

        acc += tl.dot(x_tile, phi_tile)
        x_tile_f32 = x_tile.to(tl.float32)
        acc_sq += tl.sum(x_tile_f32 * x_tile_f32, axis=1)

    rms = tl.sqrt(acc_sq / K + eps)
    rsigma = 1.0 / rms

    bias = tl.load(bias_ptr + rn_global, mask=rn_global < N, other=0.0).to(tl.float32)
    alpha_val = tl.where(
        is_pre_program, alpha_pre, tl.where(is_post_program, alpha_post, alpha_res)
    )

    out = rsigma[:, None] * alpha_val * acc + bias[None, :]

    # Apply stream-specific activation
    if HRES_LITE_MODE:
        out_res = _compute_hres_mhc_lite(
            out,
            perm_ptr,
            stride_perm_k,
            stride_perm_ij,
            n_squared,
            n_factorial,
            BLOCK_M,
            BLOCK_N,
        )
        # Output dimension for store: n for pre/post, n² for res
        n_out = n + (n_squared - n) * is_res_i32
    else:
        out_res = out

    # Pre: sigmoid(out), Post: 2*sigmoid(out), Res: out_res (identity or lite)
    out_activated = tl.where(
        is_pre_program | is_post_program, tl.sigmoid(out) * (1.0 + is_post_f32), out_res
    )

    # Compute output column offset in unified tensor layout: [pre: 0..n-1, post: n..2n-1, res: 2n..2n+n_squared-1]
    out_col_start = tl.where(
        is_pre_program,
        0,
        tl.where(is_post_program, n, 2 * n)
    )
    # Unified output tensor - strides are the same for all streams

    out_col_offset = out_col_start + rn_local
    tl.store(
        out_ptr + rm[:, None] * stride_out_m + out_col_offset[None, :] * stride_out_n,
        out_activated,
        mask=(rm[:, None] < M) & (rn_local[None, :] < n_out),
    )


@triton.jit
def _mhc_fused_split_kernel(
    x_ptr,
    phi_ptr,  # Unified phi: (K, n + n + n_res)
   
    acc_ptr,  # Single unified output: (NUM_KSPLIT, M, n + n + n_res)
    acc_sq_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    n: tl.constexpr,
    n_squared: tl.constexpr,
    n_factorial: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_phi_k,  # Stride for K dimension
    stride_phi_n,  # Stride for N dimension (total_cols)
   
    stride_acc_k,  # Stride for NUM_KSPLIT dimension
    stride_acc_m,  # Stride for M dimension
    stride_acc_n,  # Stride for N dimension (total_cols)
    stride_acc_sq_k,
    stride_acc_sq_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    HRES_LITE_MODE: tl.constexpr,
):
    """
    Split-K kernel for mHC - computes partial results for equations 14-15.

    Writes all streams to unified contiguous tensor: (NUM_KSPLIT, M, n + n + n_res)
    Memory layout: [pre_0...pre_{n-1}, post_0...post_{n-1}, res_0...res_{n_res-1}]

    Grid structure: (M_blocks, N_blocks_total, NUM_KSPLIT)
    - pid_m: row block index
    - pid_n: stream-aware column block index
    - pid_k: K-split index

    In lite mode, res stream outputs n! elements instead of n².
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    n_blocks_pre = tl.cdiv(n, BLOCK_N)
    n_blocks_post = n_blocks_pre

    is_pre_program = pid_n < n_blocks_pre
    is_post_program = (pid_n >= n_blocks_pre) & (pid_n < n_blocks_pre + n_blocks_post)
    is_res_program = ~is_pre_program & ~is_post_program

    is_post_i32 = is_post_program.to(tl.int32)
    is_res_i32 = is_res_program.to(tl.int32)

    stream_offset = is_post_i32 * n_blocks_pre + is_res_i32 * (
        n_blocks_pre + n_blocks_post
    )
    local_pid_n = pid_n - stream_offset

    rn_local = local_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    n_out_res = tl.where(HRES_LITE_MODE, n_factorial, n_squared)
    
    # Compute global column offset in unified tensor
    # Layout: [pre: 0..n-1, post: n..2n-1, res: 2n..2n+n_res-1]
    stream_start = tl.where(
        is_pre_program,
        0,
        tl.where(is_post_program, n, 2 * n)
    )
    
    n_out = tl.where(
        is_pre_program,
        n,
        tl.where(is_post_program, n, n_out_res)
    )

    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)

    if split_k_start >= K:
        return

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Only compute acc_sq for first column block (shared across all streams)
    compute_acc_sq = pid_n == 0
    acc_sq = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Compute phi column offset in unified tensor layout: [pre: 0..n-1, post: n..2n-1, res: 2n..2n+n_res-1]
    phi_col_start = tl.where(
        is_pre_program,
        0,
        tl.where(is_post_program, n, 2 * n)
    )
    # Unified phi tensor - strides are the same for all streams

    # Loop over this split's K range
    k_span = split_k_end - split_k_start
    num_k_iter = tl.cdiv(k_span, BLOCK_K)

    for k_idx in range(num_k_iter):
        k = split_k_start + k_idx * BLOCK_K
        rk = k + tl.arange(0, BLOCK_K)

        x_tile = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
            mask=(rm[:, None] < M) & (rk[None, :] < split_k_end),
            other=0.0,
        )

        phi_col_offset = phi_col_start + rn_local
        phi_tile = tl.load(
            phi_ptr + rk[:, None] * stride_phi_k + phi_col_offset[None, :] * stride_phi_n,
            mask=(rk[:, None] < split_k_end) & (rn_local[None, :] < n_out),
            other=0.0,
        )

        acc += tl.dot(x_tile, phi_tile)

        if compute_acc_sq:
            x_tile_f32 = x_tile.to(tl.float32)
            acc_sq += tl.sum(x_tile_f32 * x_tile_f32, axis=1)

    # Unified contiguous write
    col_offset = stream_start + rn_local

    tl.store(
        acc_ptr
        + pid_k * stride_acc_k
        + rm[:, None] * stride_acc_m
        + col_offset[None, :] * stride_acc_n,
        acc,
        mask=(rm[:, None] < M) & (rn_local[None, :] < n_out),
    )

    if compute_acc_sq:
        tl.store(
            acc_sq_ptr + pid_k * stride_acc_sq_k + rm * stride_acc_sq_m,
            acc_sq,
            mask=rm < M,
        )


@triton.jit
def _mhc_fused_reduce_kernel(
    acc_ptr,  # Unified input: (NUM_KSPLIT, M, n + n + n_res)
    acc_sq_ptr,
    alpha_pre,
    alpha_post,
    alpha_res,
    bias_ptr,
    out_ptr,  # Unified output: (M, n + n + n_squared)
   
    perm_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    n: tl.constexpr,
    n_squared: tl.constexpr,
    n_factorial: tl.constexpr,
    eps: tl.constexpr,
    stride_acc_k,  # Stride for NUM_KSPLIT dimension
    stride_acc_m,  # Stride for M dimension
    stride_acc_n,  # Stride for N dimension (total_cols)
    stride_acc_sq_k,
    stride_acc_sq_m,
    stride_out_m,  # Stride for M dimension
    stride_out_n,  # Stride for N dimension (total_cols)
   
    stride_perm_k,
    stride_perm_ij,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    HRES_LITE_MODE: tl.constexpr,
):
    """
    Reduce kernel for mHC - combines partial results and applies post-processing.

    Reads from unified contiguous tensor (NUM_KSPLIT, M, total_cols) where
    total_cols = n + n + n_res, laid out as: [pre, post, res].

    Sums partial dot products and sum-of-squares across K-splits, then applies:
    - RMS normalization (Eq 15)
    - Bias + alpha scaling (Eq 16)
    - Stream-specific activations (Eq 17-18)
    - For lite mode: softmax + permutation combination for exact doubly stochastic

    Grid structure: (M_blocks, N_blocks_total)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    n_blocks_pre = tl.cdiv(n, BLOCK_N)
    n_blocks_post = n_blocks_pre

    is_pre_program = pid_n < n_blocks_pre
    is_post_program = (pid_n >= n_blocks_pre) & (pid_n < n_blocks_pre + n_blocks_post)
    is_res_program = ~is_pre_program & ~is_post_program
    is_post_f32 = is_post_program.to(tl.float32)
    is_post_i32 = is_post_program.to(tl.int32)
    is_res_i32 = is_res_program.to(tl.int32)

    stream_offset = is_post_i32 * n_blocks_pre + is_res_i32 * (
        n_blocks_pre + n_blocks_post
    )
    local_pid_n = pid_n - stream_offset

    rn_local = local_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    n_out_res = tl.where(HRES_LITE_MODE, n_factorial, n_squared)
    n_out = tl.where(
        is_pre_program,
        n,
        tl.where(is_post_program, n, n_out_res)
    )

    # Global column offset in unified tensor layout: [pre: 0..n-1, post: n..2n-1, res: 2n..2n+n_res-1]
    stream_start = tl.where(
        is_pre_program,
        0,
        tl.where(is_post_program, n, 2 * n)
    )
    col_offset = stream_start + rn_local

    # For bias: use global column index
    rn_bias_global = stream_start + rn_local

    # Sum partial results across K-splits
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for ks in range(ACTUAL_KSPLIT):
        acc_partial = tl.load(
            acc_ptr
            + ks * stride_acc_k
            + rm[:, None] * stride_acc_m
            + col_offset[None, :] * stride_acc_n,
            mask=(rm[:, None] < M) & (rn_local[None, :] < n_out),
            other=0.0,
        )
        acc += acc_partial

        acc_sq_partial = tl.load(
            acc_sq_ptr + ks * stride_acc_sq_k + rm * stride_acc_sq_m,
            mask=rm < M,
            other=0.0,
        )
        acc_sq += acc_sq_partial

    rms = tl.sqrt(acc_sq / K + eps)
    rsigma = 1.0 / rms

    bias = tl.load(bias_ptr + rn_bias_global, mask=rn_bias_global < N, other=0.0).to(tl.float32)
    alpha_val = tl.where(
        is_pre_program, alpha_pre, tl.where(is_post_program, alpha_post, alpha_res)
    )

    out = rsigma[:, None] * alpha_val * acc + bias[None, :]

    if HRES_LITE_MODE:
        out_res = _compute_hres_mhc_lite(
            out,
            perm_ptr,
            stride_perm_k,
            stride_perm_ij,
            n_squared,
            n_factorial,
            BLOCK_M,
            BLOCK_N,
        )
        # Output dimension for store: n for pre/post, n² for res
        n_out = n + (n_squared - n) * is_res_i32
    else:
        out_res = out

    # Pre: sigmoid(out), Post: 2*sigmoid(out), Res: out_res (identity or lite)
    out_activated = tl.where(
        is_pre_program | is_post_program, tl.sigmoid(out) * (1.0 + is_post_f32), out_res
    )

    # Compute output column offset in unified tensor layout: [pre: 0..n-1, post: n..2n-1, res: 2n..2n+n_squared-1]
    out_col_start = tl.where(
        is_pre_program,
        0,
        tl.where(is_post_program, n, 2 * n)
    )
    # Unified output tensor - strides are the same for all streams

    out_col_offset = out_col_start + rn_local
    tl.store(
        out_ptr + rm[:, None] * stride_out_m + out_col_offset[None, :] * stride_out_n,
        out_activated,
        mask=(rm[:, None] < M) & (rn_local[None, :] < n_out),
    )


@triton.jit
def _sinkhorn_knopp_log_domain_kernel(
    logits_ptr,
    out_ptr,
    M,
    stride_batch,
    stride_row,
    stride_col,
    N: tl.constexpr,
    NUM_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Log-domain Sinkhorn-Knopp kernel for projecting raw logits onto doubly stochastic matrices."""
    pid_m = tl.program_id(axis=0)
    batch_start = pid_m * BLOCK_M

    LOG2_E: tl.constexpr = 1.4426950408889634
    LN_2: tl.constexpr = 0.6931471805599453

    row_idx = tl.arange(0, N)[:, None]
    col_idx = tl.arange(0, N)[None, :]
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
