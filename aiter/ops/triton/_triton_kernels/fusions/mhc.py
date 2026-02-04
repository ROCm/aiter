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
- H^res: Two modes available via HRES_LITE_MODE:
    - Sinkhorn mode: identity activation, ready for Sinkhorn-Knopp (Eq 19)
    - Lite mode : softmax + convex combination of permutation matrices

Single fused kernel minimizes memory traffic and kernel launch overhead.
"""

import triton
import triton.language as tl


@triton.jit
def _compute_hres_mhc_lite(
    out,           # (BLOCK_M, n_factorial) logits - padded to BLOCK_N for Triton
    perm_ptr,      # (n_factorial, n_squared) permutation matrices
    stride_perm_k,
    stride_perm_ij,
    n_squared: tl.constexpr,    # n*n: output dimension
    n_factorial: tl.constexpr,  # n!: number of permutation matrices / input weights
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HRES_OP: tl.constexpr,      # 0=fma loop, 1=dot product
):
    """
    Compute exact doubly stochastic H_res via Birkhoff-von Neumann theorem.
    
    Computes H_res = softmax(logits) @ P where:
    - logits: (BLOCK_M, n_factorial) raw weights for each permutation matrix  
    - P: (n_factorial, n_squared) stacked permutation matrices
    - Result: (BLOCK_M, n_squared) convex combination = exact doubly stochastic
    
    """
    LOG2_E: tl.constexpr = 1.4426950408889634  # 1/ln(2)
    
    col_idx = tl.arange(0, BLOCK_N)
    n_factorial_mask = col_idx[None, :] < n_factorial
    out_masked = tl.where(n_factorial_mask, out, float('-inf'))
    
    # Initialize output accumulator
    H_res = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    if HRES_OP == 0:
        # FMA loop: explicit iteration with optimized softmax (good for small n_factorial)
        # Optimized softmax: direct exp-normalize (faster than log-domain for small n_factorial)
        out_max = tl.max(out_masked, axis=1, keep_dims=True)
        exp_shifted = tl.exp2((out_masked - out_max) * LOG2_E)
        sum_exp = tl.sum(tl.where(n_factorial_mask, exp_shifted, 0.0), axis=1, keep_dims=True)
        alpha = exp_shifted / sum_exp  # (BLOCK_M, n_factorial) convex coefficients
        
        for perm_idx in tl.static_range(n_factorial):
            # Load permutation matrix P[perm_idx] - a vector of n_squared elements
            elem_idx = tl.arange(0, BLOCK_N)
            elem_mask = elem_idx < n_squared
            perm_vals = tl.load(
                perm_ptr + perm_idx * stride_perm_k + elem_idx * stride_perm_ij,
                mask=elem_mask,
                other=0.0
            ).to(tl.float32)
            
            # Extract weight for this permutation (alpha[:, perm_idx]). Use masking to extract the column
            weight_mask = (col_idx == perm_idx)
            weight_val = tl.sum(tl.where(weight_mask[None, :], alpha, 0.0), axis=1)
            
            # Fused multiply-add: H_res += weight_val * perm_vals
            H_res = tl.fma(weight_val[:, None], perm_vals[None, :], H_res)
    else:
        # Dot product: load entire permutation matrix and use tl.dot (may be better for larger problems)
        LN_2: tl.constexpr = 0.6931471805599453    # ln(2)
        
        # Log-domain softmax: logsumexp then subtract
        out_max = tl.max(out_masked, axis=1, keep_dims=True)
        out_shifted = out_masked - out_max
        exp_shifted = tl.exp2(out_shifted * LOG2_E)
        sum_exp = tl.sum(exp_shifted, axis=1, keep_dims=True)
        log_sum_exp = out_max + tl.log2(sum_exp) * LN_2
        
        log_alpha = out_masked - log_sum_exp
        alpha = tl.exp2(log_alpha * LOG2_E)  # (BLOCK_M, n_factorial) convex coefficients, padding ~0
        
        # Load permutation matrices P: (n_factorial, n_squared) padded to (BLOCK_N, BLOCK_N)
        row_idx = tl.arange(0, BLOCK_N)[:, None]
        col_idx_2d = tl.arange(0, BLOCK_N)[None, :]
        perm_mask = (row_idx < n_factorial) & (col_idx_2d < n_squared)
        P = tl.load(
            perm_ptr + row_idx * stride_perm_k + col_idx_2d * stride_perm_ij,
            mask=perm_mask,
            other=0.0
        )
        
        H_res = tl.dot(alpha, P)
    
    return H_res


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
    perm_ptr,          # (n!, n, n) permutation matrices for lite mode (flattened)
    M: tl.constexpr,   # rows: x.shape[0] - the batch/sequence dimension
    K: tl.constexpr,   # input features: nC = x.shape[1]
    N: tl.constexpr,   # output features: n² + 2n (sinkhorn) or n! + 2n (lite)
    n: tl.constexpr,   # stream parameter controlling manifold dimension
    n_squared: tl.constexpr,
    n_factorial: tl.constexpr,  # used in lite mode
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
    stride_perm_k,     # stride for perm batch dim (n! dimension)
    stride_perm_ij,    # stride for flattened (i,j) index within each perm matrix
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    HRES_LITE_MODE: tl.constexpr,  # 0=sinkhorn (identity), 1=lite (softmax+perm)
    HRES_OP: tl.constexpr,         # 0=fma loop, 1=dot product (only used if HRES_LITE_MODE=1)
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

    # Row indices
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Compute stream block counts
    n_blocks_pre = tl.cdiv(n, BLOCK_N)
    n_blocks_post = n_blocks_pre # post stream has the same number of blocks as pre stream
    
    # Determine stream type from pid_n, each program processes exactly one stream
    is_pre_program = pid_n < n_blocks_pre
    is_post_program = (pid_n >= n_blocks_pre) & (pid_n < n_blocks_pre + n_blocks_post)
    is_res_program = ~is_pre_program & ~is_post_program
    
    # Only convert types that are actually needed:
    # - is_post_f32: needed for activation scaling (2*sigmoid for post)
    # - is_post_i32, is_res_i32: needed for index arithmetic
    # Note: is_pre_f32 and is_res_f32 removed - use tl.where for alpha selection instead
    is_post_f32 = is_post_program.to(tl.float32)
    is_post_i32 = is_post_program.to(tl.int32)
    is_res_i32 = is_res_program.to(tl.int32)
    
    # Compute local block index within the stream
    stream_offset = is_post_i32 * n_blocks_pre + is_res_i32 * (n_blocks_pre + n_blocks_post)
    local_pid_n = pid_n - stream_offset
    
    # Local column indices within the stream
    rn_local = local_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Output dimension for res stream
    # - Sinkhorn mode: n² (raw logits for SK projection)
    # - Lite mode: n! (weights for permutation combination)
    n_out_res = n_squared + (n_factorial - n_squared) * is_res_i32
    
    # Output dimension for this stream
    # n for pre/post, n_out_res for res
    n_out = n + (n_out_res - n) * is_res_i32
    
    # Global column index (for bias lookup)
    # res bias starts at 2n and has n_factorial elements in lite mode and n² elements in sinkhorn mode
    col_offset = n * (is_post_i32 + 2 * is_res_i32)
    rn_global = rn_local + col_offset
    
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
    # Use tl.where for alpha selection (more efficient than arithmetic blending)
    alpha_val = tl.where(is_pre_program, alpha_pre,
                         tl.where(is_post_program, alpha_post, alpha_res))
    
    # Apply Eq 16: H = (1/r) * α * H̃ + b
    out = rsigma[:, None] * alpha_val * acc + bias[None, :]
    
    # Apply stream-specific activation
    # Compute res output first (before sigmoid to avoid unnecessary computation)
    if HRES_LITE_MODE:
        out_res = _compute_hres_mhc_lite(
            out, perm_ptr, stride_perm_k, stride_perm_ij,
            n_squared, n_factorial, BLOCK_M, BLOCK_N, HRES_OP
        )
        # Output dimension for store: n for pre/post, n² for res
        n_out = n + (n_squared - n) * is_res_i32
    else:
        out_res = out

    # Only compute sigmoid for pre/post programs, use out_res for res
    # Pre: sigmoid(out), Post: 2*sigmoid(out), Res: out_res (identity or lite)
    out_activated = tl.where(
        is_pre_program | is_post_program,
        tl.sigmoid(out) * (1.0 + is_post_f32),
        out_res
    )

    # Select output pointer and strides based on stream type
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
    acc_res_ptr,      # (NUM_KSPLIT, M, n² or n!)
    acc_sq_ptr,       # (NUM_KSPLIT, M)
    M: tl.constexpr,   # rows: x.shape[0] - the batch/sequence dimension
    K: tl.constexpr,   # input features: nC = x.shape[1]
    N: tl.constexpr,   # output features: n² + 2n (sinkhorn) or n! + 2n (lite)
    n: tl.constexpr,   # stream parameter controlling manifold dimension
    n_squared: tl.constexpr,  # n*n (precomputed for constexpr usage)
    n_factorial: tl.constexpr,  # n! = 24 for n=4, used in lite mode
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
    HRES_LITE_MODE: tl.constexpr,  # 0=sinkhorn (n² output), 1=lite (n! output)
):
    """
    Split-K kernel for mHC - computes partial results for equations 14-15.
    
    Each program processes a portion of the K dimension and writes partial
    dot products and sum-of-squares to intermediate buffers.
    
    Grid structure: (M_blocks, N_blocks_total, NUM_KSPLIT)
    - pid_m: row block index
    - pid_n: stream-aware column block index  
    - pid_k: K-split index
    
    In lite mode, res stream outputs n! elements instead of n².
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Row indices
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Compute stream block counts
    n_blocks_pre = tl.cdiv(n, BLOCK_N)
    n_blocks_post = n_blocks_pre
    
    # Determine stream type from pid_n
    is_pre_program = pid_n < n_blocks_pre
    is_post_program = (pid_n >= n_blocks_pre) & (pid_n < n_blocks_pre + n_blocks_post)
    is_res_program = ~is_pre_program & ~is_post_program
    
    # Precompute type conversions
    is_post_i32 = is_post_program.to(tl.int32)
    is_res_i32 = is_res_program.to(tl.int32)
    
    # Compute local block index within the stream
    stream_offset = is_post_i32 * n_blocks_pre + is_res_i32 * (n_blocks_pre + n_blocks_post)
    local_pid_n = pid_n - stream_offset
    
    # Local column indices within the stream
    rn_local = local_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Output dimension for res stream depends on mode
    n_out_res = tl.where(HRES_LITE_MODE, n_factorial, n_squared)
    # Output dimension for this stream: n for pre/post, n_out_res for res
    n_out = n + (n_out_res - n) * is_res_i32
    
    # Calculate K range for this split
    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
    
    # Early exit if this split has no work
    if split_k_start >= K:
        return
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Only compute acc_sq for pre-stream programs (first N-block)
    # to avoid redundant computation since acc_sq depends only on x.
    compute_acc_sq = is_pre_program & (local_pid_n == 0)
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
        
        # Only compute acc_sq for programs that need it (saves computation for post/res streams)
        if compute_acc_sq:
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
    
    # Store partial acc_sq only from the programs that computed it
    if compute_acc_sq:
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
    acc_res_ptr,      # (NUM_KSPLIT, M, n² or n!)
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
    perm_ptr,         # (n!, n, n) permutation matrices for lite mode
    # Dimensions
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    n: tl.constexpr,
    n_squared: tl.constexpr,
    n_factorial: tl.constexpr,  # n! = 24 for n=4
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
    stride_perm_k,    # stride for perm batch dim
    stride_perm_ij,   # stride for flattened (i,j) index
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    HRES_LITE_MODE: tl.constexpr,  # 0=sinkhorn, 1=lite
    HRES_OP: tl.constexpr,         # 0=fma loop, 1=dot product (only used if HRES_LITE_MODE=1)
):
    """
    Reduce kernel for mHC - combines partial results and applies post-processing.
    
    Sums partial dot products and sum-of-squares across K-splits, then applies:
    - RMS normalization (Eq 15)
    - Bias + alpha scaling (Eq 16)
    - Stream-specific activations (Eq 17-18)
    - For lite mode: softmax + permutation combination for exact doubly stochastic
    
    Grid structure: (M_blocks, N_blocks_total)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row indices
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Compute stream block counts
    n_blocks_pre = tl.cdiv(n, BLOCK_N)
    n_blocks_post = n_blocks_pre
    
    # Determine stream type from pid_n
    is_pre_program = pid_n < n_blocks_pre
    is_post_program = (pid_n >= n_blocks_pre) & (pid_n < n_blocks_pre + n_blocks_post)
    is_res_program = ~is_pre_program & ~is_post_program
    
    # Only convert types that are actually needed:
    # - is_post_f32: needed for activation scaling (2*sigmoid for post)
    # - is_post_i32, is_res_i32: needed for index arithmetic
    # Note: is_pre_f32 and is_res_f32 removed - use tl.where for alpha selection instead
    is_post_f32 = is_post_program.to(tl.float32)
    is_post_i32 = is_post_program.to(tl.int32)
    is_res_i32 = is_res_program.to(tl.int32)
    
    # Compute local block index within the stream
    stream_offset = is_post_i32 * n_blocks_pre + is_res_i32 * (n_blocks_pre + n_blocks_post)
    local_pid_n = pid_n - stream_offset
    
    # Local column indices within the stream
    rn_local = local_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Output dimension for res stream depends on mode
    n_out_res = tl.where(HRES_LITE_MODE, n_factorial, n_squared)
    # Output dimension for this stream: n for pre/post, n_out_res for res
    n_out = n + (n_out_res - n) * is_res_i32
    
    # Global column index (for bias lookup)
    col_offset = n * (is_post_i32 + 2 * is_res_i32)
    rn_global = rn_local + col_offset
    
    # Select intermediate buffer based on stream type
    acc_ptr = tl.where(is_pre_program, acc_pre_ptr,
                       tl.where(is_post_program, acc_post_ptr, acc_res_ptr))
    stride_acc_k = tl.where(is_pre_program, stride_acc_pre_k,
                       tl.where(is_post_program, stride_acc_post_k, stride_acc_res_k))
    stride_acc_m = tl.where(is_pre_program, stride_acc_pre_m,
                       tl.where(is_post_program, stride_acc_post_m, stride_acc_res_m))
    stride_acc_n = tl.where(is_pre_program, stride_acc_pre_n,
                       tl.where(is_post_program, stride_acc_post_n, stride_acc_res_n))
    
    # Sum partial results across K-splits (fused loop for better cache utilization)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for ks in range(ACTUAL_KSPLIT):
        # Load and accumulate partial dot products
        acc_partial = tl.load(
            acc_ptr + ks * stride_acc_k + rm[:, None] * stride_acc_m + rn_local[None, :] * stride_acc_n,
            mask=(rm[:, None] < M) & (rn_local[None, :] < n_out),
            other=0.0,
        )
        acc += acc_partial
        
        # Load and accumulate partial acc_sq
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
    # Use tl.where for alpha selection (more efficient than arithmetic blending)
    alpha_val = tl.where(is_pre_program, alpha_pre,
                         tl.where(is_post_program, alpha_post, alpha_res))
    
    # Apply Eq 16: H = (1/r) * α * H̃ + b
    out = rsigma[:, None] * alpha_val * acc + bias[None, :]
    
    # Apply stream-specific activation
    # Compute res output first (before sigmoid to avoid unnecessary computation)
    if HRES_LITE_MODE:
        out_res = _compute_hres_mhc_lite(
            out, perm_ptr, stride_perm_k, stride_perm_ij,
            n_squared, n_factorial, BLOCK_M, BLOCK_N, HRES_OP
        )
        # Output dimension for store: n for pre/post, n² for res
        n_out = n + (n_squared - n) * is_res_i32
    else:
        out_res = out

    # Only compute sigmoid for pre/post programs, use out_res for res
    # Pre: sigmoid(out), Post: 2*sigmoid(out), Res: out_res (identity or lite)
    out_activated = tl.where(
        is_pre_program | is_post_program,
        tl.sigmoid(out) * (1.0 + is_post_f32),
        out_res
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

