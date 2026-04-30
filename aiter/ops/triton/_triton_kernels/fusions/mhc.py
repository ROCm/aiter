# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton kernel for mHC (manifold-constrained Hyper Connection) operations."""

import triton
import triton.language as tl


@triton.jit
def _mhc_apply_pre_mix_tile(
    x_ptr,
    out_ptr,
    pre_mix_2d,  # (BLOCK_M, N_POW2) fp32, caller-supplied
    rm,
    rc,
    i_n,
    M,
    C: tl.constexpr,
    n: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_om,
    stride_oc,
):
    """Compute one (M-tile, C-tile) of the pre-stream apply step:

        out[rm, rc] = sum_{i in [0, n)} pre_mix_2d[rm, i] * x[rm, i*C + rc]

    `pre_mix_2d` must already be padded to width `N_POW2` along the n-axis
    (entries with `i_n >= n` masked to 0 by the caller).
    """
    x_tile = tl.load(
        x_ptr
        + rm[:, None, None] * stride_xm
        + (i_n[None, :, None] * C + rc[None, None, :]) * stride_xk,
        mask=(rm[:, None, None] < M)
        & (i_n[None, :, None] < n)
        & (rc[None, None, :] < C),
        other=0.0,
    ).to(tl.float32)
    li_acc = tl.sum(pre_mix_2d[:, :, None] * x_tile, axis=1)
    tl.store(
        out_ptr + rm[:, None] * stride_om + rc[None, :] * stride_oc,
        li_acc.to(out_ptr.dtype.element_ty),
        mask=(rm[:, None] < M) & (rc[None, :] < C),
    )


@triton.jit
def _mhc_fused_kernel(
    x_ptr,
    phi_ptr,  # Unified phi: (K, n + n + n_res), layout [pre | post | res]
    alpha_pre,
    alpha_post,
    alpha_res,
    bias_ptr,
    out_ptr,  # Shrunk output: (M, n + n_squared), layout [post | res]
    pre_mix_ptr,  # (M, n) fp32; consumed by `_mhc_reduce_splitc_kernel` when USE_REDUCE_SPLITC=True
    layer_input_ptr,  # (M, C); written directly when USE_REDUCE_SPLITC=False (inline apply)
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    n: tl.constexpr,
    n_squared: tl.constexpr,
    C: tl.constexpr,
    eps: tl.constexpr,
    hc_pre_eps,
    hc_post_mult_value,
    stride_xm,
    stride_xk,
    stride_phi_k,  # Stride for K dimension
    stride_phi_n,  # Stride for N dimension (total_cols)
    stride_out_m,  # Stride for M dimension
    stride_out_n,  # Stride for N dimension (post + res)
    stride_pm,  # Stride for M dimension of pre_mix
    stride_pn,  # Stride for n dimension of pre_mix
    stride_li_m,  # Stride for M dimension of layer_input (inline path)
    stride_li_c,  # Stride for C dimension of layer_input (inline path)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    N_POW2: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    NUM_SINKHORN_ITERS: tl.constexpr,
    USE_REDUCE_SPLITC: tl.constexpr,
):
    """
    Fused kernel for mHC equations 14-18 (pre/post/res streams).

    Computes three separate outputs:
    - H^pre: (M, n) - sigmoid activation (Eq 17) and dispatches on `USE_REDUCE_SPLITC`:
        * True  -> stores `pre_mix = sigmoid(H_pre) + hc_pre_eps` to `pre_mix_ptr`
                   for `_mhc_reduce_splitc_kernel` to consume (large-C path with C-axis parallelism).
        * False -> runs the inline 3D-broadcast apply directly to `layer_input_ptr`
                   (small-C path that avoids the extra apply-kernel launch overhead).
    - H^post: (M, n) with hc_post_mult_value * sigmoid activation (Eq 18)
    - H^res: (M, n, n) doubly-stochastic Sinkhorn-Knopp output when
      NUM_SINKHORN_ITERS > 0 (Eq 19), or raw logits when 0.

    Post and res streams write to a unified `(M, n + n_squared)` tensor following
    `[post | res]`. phi/bias indexing follows `[pre | post | res]` layout. When
    NUM_SINKHORN_ITERS > 0, the res branch reshapes its `(BLOCK_M, BLOCK_N)`
    tile to `(BLOCK_M, n, n)` and runs log-domain Sinkhorn-Knopp inline before
    the store; this requires `BLOCK_N == n_squared` (enforced by the wrapper).

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
    is_post_i32 = is_post_program.to(tl.int32)
    is_res_i32 = is_res_program.to(tl.int32)

    stream_offset = is_post_i32 * n_blocks_pre + is_res_i32 * (
        n_blocks_pre + n_blocks_post
    )
    local_pid_n = pid_n - stream_offset

    rn_local = local_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    n_out = n + (n_squared - n) * is_res_i32

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Compute phi column offset in unified tensor layout: [pre: 0..n-1, post: n..2n-1, res: 2n..2n+n_res-1]
    # phi/bias indexing keeps the original [pre | post | res] layout
    phi_col_start = tl.where(is_pre_program, 0, tl.where(is_post_program, n, 2 * n))
    rn_global = rn_local + phi_col_start

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
            phi_ptr
            + rk[:, None] * stride_phi_k
            + phi_col_offset[None, :] * stride_phi_n,
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

    if is_pre_program:
        pre_mix = tl.sigmoid(out) + hc_pre_eps  # (BLOCK_M, BLOCK_N)
        if USE_REDUCE_SPLITC:
            # Stash `pre_mix` for `_mhc_reduce_splitc_kernel` to consume
            tl.store(
                pre_mix_ptr + rm[:, None] * stride_pm + rn_local[None, :] * stride_pn,
                pre_mix,
                mask=(rm[:, None] < M) & (rn_local[None, :] < n),
            )
        else:
            # Run the apply step inline via a 3D-broadcast reduction
            pre_mix_n = tl.where(rn_local[None, :] < n, pre_mix, 0.0)
            i_n = tl.arange(0, N_POW2)
            pre_mix_2d = tl.sum(
                tl.where(rn_local[None, None, :] == i_n[None, :, None],
                         pre_mix_n[:, None, :], 0.0),
                axis=2,
            )  # (BLOCK_M, N_POW2)
            for c0 in range(0, C, BLOCK_C):
                rc = c0 + tl.arange(0, BLOCK_C)
                _mhc_apply_pre_mix_tile(
                    x_ptr,
                    layer_input_ptr,
                    pre_mix_2d,
                    rm,
                    rc,
                    i_n,
                    M,
                    C,
                    n,
                    stride_xm,
                    stride_xk,
                    stride_li_m,
                    stride_li_c,
                )
    else:
        # Post or Res branch.
        if is_post_program:
            out_activated = tl.sigmoid(out) * hc_post_mult_value
            out_col_start = 0
        else:
            # Res branch: log-domain Sinkhorn-Knopp on (BLOCK_M, n, n) sub-tile,
            # or raw logits when NUM_SINKHORN_ITERS == 0. Requires BLOCK_N == n_squared.
            if NUM_SINKHORN_ITERS > 0:
                LOG2_E: tl.constexpr = 1.4426950408889634
                LN_2: tl.constexpr = 0.6931471805599453

                log_A = tl.reshape(out, (BLOCK_M, n, n))

                log_u = tl.zeros((BLOCK_M, n), dtype=tl.float32)
                log_v = tl.zeros((BLOCK_M, n), dtype=tl.float32)

                for _ in range(NUM_SINKHORN_ITERS):
                    scaled_row = log_A + log_v[:, None, :]
                    row_max = tl.max(scaled_row, axis=2)
                    exp_shifted = tl.exp2(
                        (scaled_row - row_max[:, :, None]) * LOG2_E
                    )
                    row_sum_exp = tl.sum(exp_shifted, axis=2)
                    log_row_sums = row_max + tl.log2(row_sum_exp) * LN_2
                    log_u = -log_row_sums

                    scaled_col = log_A + log_u[:, :, None]
                    col_max = tl.max(scaled_col, axis=1)
                    exp_shifted = tl.exp2(
                        (scaled_col - col_max[:, None, :]) * LOG2_E
                    )
                    col_sum_exp = tl.sum(exp_shifted, axis=1)
                    log_col_sums = col_max + tl.log2(col_sum_exp) * LN_2
                    log_v = -log_col_sums

                log_P = log_A + log_u[:, :, None] + log_v[:, None, :]
                P = tl.exp2(log_P * LOG2_E)
                out_activated = tl.reshape(P, (BLOCK_M, n_squared))
            else:
                out_activated = out
            out_col_start = n
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
):
    """
    Split-K kernel for mHC - computes partial results for equations 14-15.

    Writes all streams to unified contiguous tensor: (NUM_KSPLIT, M, n + n + n_res)
    Memory layout: [pre_0...pre_{n-1}, post_0...post_{n-1}, res_0...res_{n_res-1}]

    Grid structure: (M_blocks, N_blocks_total, NUM_KSPLIT)
    - pid_m: row block index
    - pid_n: stream-aware column block index
    - pid_k: K-split index
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

    n_out_res = n_squared

    # Compute global column offset in unified tensor
    # Layout: [pre: 0..n-1, post: n..2n-1, res: 2n..2n+n_res-1]
    stream_start = tl.where(is_pre_program, 0, tl.where(is_post_program, n, 2 * n))

    n_out = tl.where(is_pre_program, n, tl.where(is_post_program, n, n_out_res))

    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)

    if split_k_start >= K:
        return

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Only compute acc_sq for first column block (shared across all streams)
    compute_acc_sq = pid_n == 0
    acc_sq = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Compute phi column offset in unified tensor layout: [pre: 0..n-1, post: n..2n-1, res: 2n..2n+n_res-1]
    phi_col_start = tl.where(is_pre_program, 0, tl.where(is_post_program, n, 2 * n))

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
            phi_ptr
            + rk[:, None] * stride_phi_k
            + phi_col_offset[None, :] * stride_phi_n,
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
    acc_ptr,  # Unified input: (NUM_KSPLIT, M, n + n + n_res), layout [pre | post | res]
    acc_sq_ptr,
    alpha_pre,
    alpha_post,
    alpha_res,
    bias_ptr,
    out_ptr,  # Unified output: (M, n + n_squared), layout [post | res]
    pre_mix_ptr,  # (M, n) fp32; consumed by `_mhc_reduce_splitc_kernel` when USE_REDUCE_SPLITC=True
    x_ptr,  # (M, n*C); needed for the inline apply path when USE_REDUCE_SPLITC=False
    layer_input_ptr,  # (M, C); written directly when USE_REDUCE_SPLITC=False (inline apply)
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    n: tl.constexpr,
    n_squared: tl.constexpr,
    C: tl.constexpr,
    eps: tl.constexpr,
    hc_pre_eps,
    hc_post_mult_value,
    stride_acc_k,  # Stride for NUM_KSPLIT dimension
    stride_acc_m,  # Stride for M dimension
    stride_acc_n,  # Stride for N dimension (total_cols)
    stride_acc_sq_k,
    stride_acc_sq_m,
    stride_out_m,  # Stride for M dimension
    stride_out_n,  # Stride for N dimension (post + res)
    stride_pm,  # Stride for M dimension of pre_mix
    stride_pn,  # Stride for n dimension of pre_mix
    stride_xm,
    stride_xk,
    stride_li_m,
    stride_li_c,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    N_POW2: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    NUM_SINKHORN_ITERS: tl.constexpr,
    USE_REDUCE_SPLITC: tl.constexpr,
):
    """
    Reduce kernel for mHC - combines split-K partials and produces stream outputs.

    Reads from unified `(NUM_KSPLIT, M, n + n + n_squared)` partials laid out as
    `[pre | post | res]`. Sums partial dot products and sum-of-squares across
    K-splits, then applies:
    - RMS normalization (Eq 15)
    - Bias + alpha scaling (Eq 16)
    - Stream-specific activations (Eq 17-18) using hc_post_mult_value for post
    - For pre: dispatches on `USE_REDUCE_SPLITC`:
        * True  -> stores `pre_mix = sigmoid(H_pre) + hc_pre_eps` to `pre_mix_ptr`
                   for `_mhc_reduce_splitc_kernel` to consume (large-C path with C-axis parallelism).
        * False -> runs the inline 3D-broadcast apply directly to `layer_input_ptr`
                   (small-C path that avoids the extra apply-kernel launch overhead).
    - For res: optional inline log-domain Sinkhorn-Knopp (Eq 19) when
      NUM_SINKHORN_ITERS > 0. Requires `BLOCK_N == n_squared`.

    The final `out` tensor is shrunk to `(M, n + n_squared)` with layout
    `[post | res]`.

    Grid structure: (M_blocks, N_blocks_total).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

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

    n_out = tl.where(is_pre_program, n, tl.where(is_post_program, n, n_squared))

    # Partials buffer keeps the original [pre | post | res] layout
    stream_start = tl.where(is_pre_program, 0, tl.where(is_post_program, n, 2 * n))
    col_offset = stream_start + rn_local
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

    bias = tl.load(bias_ptr + rn_bias_global, mask=rn_bias_global < N, other=0.0).to(
        tl.float32
    )
    alpha_val = tl.where(
        is_pre_program, alpha_pre, tl.where(is_post_program, alpha_post, alpha_res)
    )

    out = rsigma[:, None] * alpha_val * acc + bias[None, :]

    if is_pre_program:
        pre_mix = tl.sigmoid(out) + hc_pre_eps  # (BLOCK_M, BLOCK_N)
        if USE_REDUCE_SPLITC:
            # Stash `pre_mix` for `_mhc_reduce_splitc_kernel` to consume
            tl.store(
                pre_mix_ptr + rm[:, None] * stride_pm + rn_local[None, :] * stride_pn,
                pre_mix,
                mask=(rm[:, None] < M) & (rn_local[None, :] < n),
            )
        else:
            # Run the apply step inline via a 3D-broadcast reduction
            pre_mix_n = tl.where(rn_local[None, :] < n, pre_mix, 0.0)
            i_n = tl.arange(0, N_POW2)
            pre_mix_2d = tl.sum(
                tl.where(rn_local[None, None, :] == i_n[None, :, None],
                         pre_mix_n[:, None, :], 0.0),
                axis=2,
            )  # (BLOCK_M, N_POW2)
            for c0 in range(0, C, BLOCK_C):
                rc = c0 + tl.arange(0, BLOCK_C)
                _mhc_apply_pre_mix_tile(
                    x_ptr,
                    layer_input_ptr,
                    pre_mix_2d,
                    rm,
                    rc,
                    i_n,
                    M,
                    C,
                    n,
                    stride_xm,
                    stride_xk,
                    stride_li_m,
                    stride_li_c,
                )
    else:
        # Post or Res branch.
        if is_post_program:
            out_activated = tl.sigmoid(out) * hc_post_mult_value
            out_col_start = 0
        else:
            # Res branch: log-domain Sinkhorn-Knopp on (BLOCK_M, n, n) sub-tile,
            # or raw logits when NUM_SINKHORN_ITERS == 0. Requires BLOCK_N == n_squared.
            if NUM_SINKHORN_ITERS > 0:
                LOG2_E: tl.constexpr = 1.4426950408889634
                LN_2: tl.constexpr = 0.6931471805599453

                log_A = tl.reshape(out, (BLOCK_M, n, n))

                log_u = tl.zeros((BLOCK_M, n), dtype=tl.float32)
                log_v = tl.zeros((BLOCK_M, n), dtype=tl.float32)

                for _ in range(NUM_SINKHORN_ITERS):
                    scaled_row = log_A + log_v[:, None, :]
                    row_max = tl.max(scaled_row, axis=2)
                    exp_shifted = tl.exp2(
                        (scaled_row - row_max[:, :, None]) * LOG2_E
                    )
                    row_sum_exp = tl.sum(exp_shifted, axis=2)
                    log_row_sums = row_max + tl.log2(row_sum_exp) * LN_2
                    log_u = -log_row_sums

                    scaled_col = log_A + log_u[:, :, None]
                    col_max = tl.max(scaled_col, axis=1)
                    exp_shifted = tl.exp2(
                        (scaled_col - col_max[:, None, :]) * LOG2_E
                    )
                    col_sum_exp = tl.sum(exp_shifted, axis=1)
                    log_col_sums = col_max + tl.log2(col_sum_exp) * LN_2
                    log_v = -log_col_sums

                log_P = log_A + log_u[:, :, None] + log_v[:, None, :]
                P = tl.exp2(log_P * LOG2_E)
                out_activated = tl.reshape(P, (BLOCK_M, n_squared))
            else:
                out_activated = out
            out_col_start = n
        out_col_offset = out_col_start + rn_local
        tl.store(
            out_ptr + rm[:, None] * stride_out_m + out_col_offset[None, :] * stride_out_n,
            out_activated,
            mask=(rm[:, None] < M) & (rn_local[None, :] < n_out),
        )


@triton.jit
def _mhc_reduce_splitc_kernel(
    x_ptr,           # (M, n*C)
    pre_mix_ptr,     # (M, n) fp32
    out_ptr,         # (M, C) layer_input in x.dtype
    M,
    C: tl.constexpr,
    n: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_pm,
    stride_pn,
    stride_om,
    stride_oc,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    N_POW2: tl.constexpr,
):
    """
    Reduce-splitC step for the pre-stream of mHC: produces
        layer_input[m, c] = sum_i pre_mix[m, i] * x[m, i*C + c].

    The (M, C) tiling exposes C-axis parallelism preceded by the fused_mhc
    to more efficiently handle large C values.
    """
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    i_n = tl.arange(0, N_POW2)

    pre_mix_2d = tl.load(
        pre_mix_ptr + rm[:, None] * stride_pm + i_n[None, :] * stride_pn,
        mask=(rm[:, None] < M) & (i_n[None, :] < n),
        other=0.0,
    )  # (BLOCK_M, N_POW2) fp32

    _mhc_apply_pre_mix_tile(
        x_ptr,
        out_ptr,
        pre_mix_2d,
        rm,
        rc,
        i_n,
        M,
        C,
        n,
        stride_xm,
        stride_xk,
        stride_om,
        stride_oc,
    )
