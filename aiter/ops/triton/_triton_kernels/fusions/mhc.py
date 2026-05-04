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
    layer_input_ptr,  # (M, C); written directly via the inline apply step
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    n: tl.constexpr,
    n_squared: tl.constexpr,
    C: tl.constexpr,
    eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_phi_k,  # Stride for K dimension
    stride_phi_n,  # Stride for N dimension (total_cols)
    stride_out_m,  # Stride for M dimension
    stride_out_n,  # Stride for N dimension (post + res)
    stride_li_m,  # Stride for M dimension of layer_input
    stride_li_c,  # Stride for C dimension of layer_input
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    N_POW2: tl.constexpr,
    NUM_SINKHORN_ITERS: tl.constexpr,
):
    """
    Fused kernel for mHC equations 14-18 + the apply step (non-split-K path).

    Computes three separate outputs:
    - H^pre: (M, n) - sigmoid activation (Eq 17). The pre-stream program runs
      the inline 3D-broadcast apply directly to `layer_input_ptr`, producing
      ``layer_input[m, c] = sum_i (sigmoid(H_pre[m, i]) + hc_pre_eps) * x[m, i*C + c]``.
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
        # Run the apply step inline via a 3D-broadcast reduction
        i_n = tl.arange(0, N_POW2)
        pre_mix_2d = tl.sum(
            tl.where(
                rn_local[None, None, :] == i_n[None, :, None],
                pre_mix[:, None, :],
                0.0,
            ),
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

                log2_A = tl.reshape(out, (BLOCK_M, n, n)) * LOG2_E

                log2_u = tl.zeros((BLOCK_M, n), dtype=tl.float32)
                log2_v = tl.zeros((BLOCK_M, n), dtype=tl.float32)

                for _ in range(NUM_SINKHORN_ITERS):
                    scaled_row = log2_A + log2_v[:, None, :]
                    row_max = tl.max(scaled_row, axis=2)
                    exp_shifted = tl.exp2(scaled_row - row_max[:, :, None])
                    row_sum_exp = tl.sum(exp_shifted, axis=2)
                    log2_row_sums = row_max + tl.log2(row_sum_exp)
                    log2_u = -log2_row_sums

                    scaled_col = log2_A + log2_u[:, :, None]
                    col_max = tl.max(scaled_col, axis=1)
                    exp_shifted = tl.exp2(scaled_col - col_max[:, None, :])
                    col_sum_exp = tl.sum(exp_shifted, axis=1)
                    log2_col_sums = col_max + tl.log2(col_sum_exp)
                    log2_v = -log2_col_sums

                log2_P = log2_A + log2_u[:, :, None] + log2_v[:, None, :]
                P = tl.exp2(log2_P)
                out_activated = tl.reshape(P, (BLOCK_M, n_squared))
            else:
                out_activated = out
            out_col_start = n
        out_col_offset = out_col_start + rn_local
        tl.store(
            out_ptr
            + rm[:, None] * stride_out_m
            + out_col_offset[None, :] * stride_out_n,
            out_activated,
            mask=(rm[:, None] < M) & (rn_local[None, :] < n_out),
        )


@triton.jit
def _mhc_fused_split_kernel(
    x_ptr,
    phi_ptr,  # Unified phi: (K, n + n + n_squared)
    acc_ptr,  # Single unified output: (NUM_KSPLIT, M, n + n + n_squared)
    acc_sq_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,  # = 2*n + n_squared (logical width of unified phi)
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
    N_TOTAL_POW2: tl.constexpr,  # = next_pow2(N), full N-tile per program
    BLOCK_K: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
):
    """
    Split-K kernel for mHC - computes partial results for equations 14-15.

    Each program owns the *full* (BLOCK_M, N_TOTAL_POW2) tile for one
    `(pid_m, pid_k)` pair: load each x-tile once, dot it against the unified
    phi covering all 3 streams in a single MFMA, and write the entire output
    row in one store. Compared to the old per-stream layout this drops the 3x
    redundant x re-read and lifts MFMA utilization (the pre/post partial
    columns are now subsumed by the same dot as the res columns).

    Writes all streams to unified contiguous tensor: (NUM_KSPLIT, M, N_total)
    Memory layout: [pre_0..pre_{n-1}, post_0..post_{n-1}, res_0..res_{n_squared-1}]

    Grid structure: (M_blocks, NUM_KSPLIT).
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, N_TOTAL_POW2)

    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)

    if split_k_start >= K:
        return

    acc = tl.zeros([BLOCK_M, N_TOTAL_POW2], dtype=tl.float32)
    acc_sq = tl.zeros([BLOCK_M], dtype=tl.float32)

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
        phi_tile = tl.load(
            phi_ptr + rk[:, None] * stride_phi_k + rn[None, :] * stride_phi_n,
            mask=(rk[:, None] < split_k_end) & (rn[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(x_tile, phi_tile)
        x_tile_f32 = x_tile.to(tl.float32)
        acc_sq += tl.sum(x_tile_f32 * x_tile_f32, axis=1)

    tl.store(
        acc_ptr
        + pid_k * stride_acc_k
        + rm[:, None] * stride_acc_m
        + rn[None, :] * stride_acc_n,
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )
    tl.store(
        acc_sq_ptr + pid_k * stride_acc_sq_k + rm * stride_acc_sq_m,
        acc_sq,
        mask=rm < M,
    )


@triton.jit
def _mhc_reduce_apply_res_block(
    acc_res,  # (BLOCK_M, N_POW2_RES) fp32, already reduced over ks
    rsigma,  # (BLOCK_M,) fp32
    rm,
    rn_res_local,
    rn_res_global,
    alpha_res,
    bias_ptr,
    out_ptr,
    M,
    n: tl.constexpr,
    n_squared: tl.constexpr,
    N_POW2_RES: tl.constexpr,
    stride_out_m,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    NUM_SINKHORN_ITERS: tl.constexpr,
):
    """Compute h_res = rsigma * alpha_res * acc_res + bias_res, optionally run
    log-domain Sinkhorn-Knopp, and store to ``out[:, n:n+n_squared]``.

    Shared between the merged-CTA path (`RES_PID_C == 0`, fused with post on the
    same `for-ks` loop) and the split-CTA path (`RES_PID_C != 0`).
    """
    bias_res = tl.load(
        bias_ptr + rn_res_global,
        mask=rn_res_local < n_squared,
        other=0.0,
    ).to(tl.float32)
    h_res = rsigma[:, None] * alpha_res * acc_res + bias_res[None, :]

    if NUM_SINKHORN_ITERS > 0:
        LOG2_E: tl.constexpr = 1.4426950408889634

        log2_A = tl.reshape(h_res, (BLOCK_M, n, n)) * LOG2_E
        log2_u = tl.zeros((BLOCK_M, n), dtype=tl.float32)
        log2_v = tl.zeros((BLOCK_M, n), dtype=tl.float32)

        for _ in range(NUM_SINKHORN_ITERS):
            scaled_row = log2_A + log2_v[:, None, :]
            row_max = tl.max(scaled_row, axis=2)
            exp_shifted = tl.exp2(scaled_row - row_max[:, :, None])
            row_sum_exp = tl.sum(exp_shifted, axis=2)
            log2_row_sums = row_max + tl.log2(row_sum_exp)
            log2_u = -log2_row_sums

            scaled_col = log2_A + log2_u[:, :, None]
            col_max = tl.max(scaled_col, axis=1)
            exp_shifted = tl.exp2(scaled_col - col_max[:, None, :])
            col_sum_exp = tl.sum(exp_shifted, axis=1)
            log2_col_sums = col_max + tl.log2(col_sum_exp)
            log2_v = -log2_col_sums

        log2_P = log2_A + log2_u[:, :, None] + log2_v[:, None, :]
        P = tl.exp2(log2_P)
        out_res = tl.reshape(P, (BLOCK_M, n_squared))
    else:
        out_res = h_res

    tl.store(
        out_ptr
        + rm[:, None] * stride_out_m
        + (n + rn_res_local[None, :]) * stride_out_n,
        out_res,
        mask=(rm[:, None] < M) & (rn_res_local[None, :] < n_squared),
    )


@triton.jit
def _mhc_reduce_apply_kernel(
    acc_ptr,  # Unified split-K partials: (NUM_KSPLIT, M, n + n + n_squared), layout [pre | post | res]
    acc_sq_ptr,  # Sum-of-squares partials: (NUM_KSPLIT, M)
    alpha_pre,
    alpha_post,
    alpha_res,
    bias_ptr,  # (n + n + n_squared,) fp32
    x_ptr,  # (M, n*C)
    out_ptr,  # Unified output: (M, n + n_squared), layout [post | res]
    layer_input_ptr,  # (M, C) in x.dtype
    M,
    K: tl.constexpr,
    n: tl.constexpr,
    n_squared: tl.constexpr,
    C: tl.constexpr,
    eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    stride_acc_k,
    stride_acc_m,
    stride_acc_n,
    stride_acc_sq_k,
    stride_acc_sq_m,
    stride_xm,
    stride_xk,
    stride_out_m,
    stride_out_n,
    stride_li_m,
    stride_li_c,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    N_POW2: tl.constexpr,
    N_POW2_RES: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    NUM_SINKHORN_ITERS: tl.constexpr,
    RES_PID_C: tl.constexpr,
):
    """
    Reduce-and-apply kernel for the split-K mHC pipeline (Eq 15-19 + apply).

    Grid: ``(cdiv(M, BLOCK_M), cdiv(C, BLOCK_C))``.

    Each program reads its M-slice of split-K partials once and computes:

    - All pids: pre stream (RMS + bias + alpha + sigmoid + hc_pre_eps) and
      the apply step ``layer_input[m, c] = sum_i pre_mix[m, i] * x[m, i*C + c]``
      restricted to this pid's BLOCK_C slice of the hidden dimension.
    - ``pid_c == 0``: post stream (``hc_post_mult_value * sigmoid``), writes to
      ``out[:, :n]``.
    - ``pid_c == RES_PID_C``: res stream (in-kernel log-domain Sinkhorn-Knopp
      when ``NUM_SINKHORN_ITERS > 0``, else raw logits), writes to
      ``out[:, n:n+n_squared]``.

    Sinkhorn requires ``n_squared is`` a power of two; the wrapper enforces
     this when ``NUM_SINKHORN_ITERS > 0``.
    """
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    rn_pre = tl.arange(0, N_POW2)

    # --- 1) Reduce split-K partials: PRE columns + acc_sq ---
    acc_pre = tl.zeros([BLOCK_M, N_POW2], dtype=tl.float32)
    acc_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for ks in range(ACTUAL_KSPLIT):
        acc_pre += tl.load(
            acc_ptr
            + ks * stride_acc_k
            + rm[:, None] * stride_acc_m
            + rn_pre[None, :] * stride_acc_n,
            mask=(rm[:, None] < M) & (rn_pre[None, :] < n),
            other=0.0,
        )
        acc_sq += tl.load(
            acc_sq_ptr + ks * stride_acc_sq_k + rm * stride_acc_sq_m,
            mask=rm < M,
            other=0.0,
        )

    # --- 2) RMS normalization (Eq 15) ---
    rms = tl.sqrt(acc_sq / K + eps)
    rsigma = 1.0 / rms

    # --- 3) Pre stream: bias + alpha + sigmoid + hc_pre_eps (Eq 16-17) ---
    bias_pre = tl.load(bias_ptr + rn_pre, mask=rn_pre < n, other=0.0).to(tl.float32)
    h_pre = rsigma[:, None] * alpha_pre * acc_pre + bias_pre[None, :]
    pre_mix_2d = tl.sigmoid(h_pre) + hc_pre_eps

    # --- 4) Apply step for this pid's BLOCK_C slice ---
    _mhc_apply_pre_mix_tile(
        x_ptr,
        layer_input_ptr,
        pre_mix_2d,
        rm,
        rc,
        rn_pre,
        M,
        C,
        n,
        stride_xm,
        stride_xk,
        stride_li_m,
        stride_li_c,
    )

    # --- 5) Post stream on pid_c == 0; Res stream on pid_c == RES_PID_C ---
    # Two compile-time layouts:
    #   RES_PID_C == 0 (single C-tile, shared CTA): one for-ks loop loads both
    #     post and res partials, then post and res are computed back-to-back.
    #   RES_PID_C != 0 (multi C-tile, separate CTAs): each CTA runs its own
    #     for-ks loop. The res body is factored into _mhc_reduce_apply_res_block
    #     to avoid duplication with the shared-CTA branch.
    if RES_PID_C == 0:
        if pid_c == 0:
            rn_post_local = tl.arange(0, N_POW2)
            rn_post_global = n + rn_post_local
            rn_res_local = tl.arange(0, N_POW2_RES)
            rn_res_global = 2 * n + rn_res_local

            acc_post = tl.zeros([BLOCK_M, N_POW2], dtype=tl.float32)
            acc_res = tl.zeros([BLOCK_M, N_POW2_RES], dtype=tl.float32)
            for ks in range(ACTUAL_KSPLIT):
                acc_post += tl.load(
                    acc_ptr
                    + ks * stride_acc_k
                    + rm[:, None] * stride_acc_m
                    + rn_post_global[None, :] * stride_acc_n,
                    mask=(rm[:, None] < M) & (rn_post_local[None, :] < n),
                    other=0.0,
                )
                acc_res += tl.load(
                    acc_ptr
                    + ks * stride_acc_k
                    + rm[:, None] * stride_acc_m
                    + rn_res_global[None, :] * stride_acc_n,
                    mask=(rm[:, None] < M) & (rn_res_local[None, :] < n_squared),
                    other=0.0,
                )

            bias_post = tl.load(
                bias_ptr + rn_post_global,
                mask=rn_post_local < n,
                other=0.0,
            ).to(tl.float32)
            h_post = rsigma[:, None] * alpha_post * acc_post + bias_post[None, :]
            out_post = tl.sigmoid(h_post) * hc_post_mult_value
            tl.store(
                out_ptr
                + rm[:, None] * stride_out_m
                + rn_post_local[None, :] * stride_out_n,
                out_post,
                mask=(rm[:, None] < M) & (rn_post_local[None, :] < n),
            )

            _mhc_reduce_apply_res_block(
                acc_res,
                rsigma,
                rm,
                rn_res_local,
                rn_res_global,
                alpha_res,
                bias_ptr,
                out_ptr,
                M,
                n,
                n_squared,
                N_POW2_RES,
                stride_out_m,
                stride_out_n,
                BLOCK_M,
                NUM_SINKHORN_ITERS,
            )
    else:
        if pid_c == 0:
            rn_post_local = tl.arange(0, N_POW2)
            rn_post_global = n + rn_post_local
            acc_post = tl.zeros([BLOCK_M, N_POW2], dtype=tl.float32)
            for ks in range(ACTUAL_KSPLIT):
                acc_post += tl.load(
                    acc_ptr
                    + ks * stride_acc_k
                    + rm[:, None] * stride_acc_m
                    + rn_post_global[None, :] * stride_acc_n,
                    mask=(rm[:, None] < M) & (rn_post_local[None, :] < n),
                    other=0.0,
                )
            bias_post = tl.load(
                bias_ptr + rn_post_global,
                mask=rn_post_local < n,
                other=0.0,
            ).to(tl.float32)
            h_post = rsigma[:, None] * alpha_post * acc_post + bias_post[None, :]
            out_post = tl.sigmoid(h_post) * hc_post_mult_value
            tl.store(
                out_ptr
                + rm[:, None] * stride_out_m
                + rn_post_local[None, :] * stride_out_n,
                out_post,
                mask=(rm[:, None] < M) & (rn_post_local[None, :] < n),
            )

        if pid_c == RES_PID_C:
            rn_res_local = tl.arange(0, N_POW2_RES)
            rn_res_global = 2 * n + rn_res_local
            acc_res = tl.zeros([BLOCK_M, N_POW2_RES], dtype=tl.float32)
            for ks in range(ACTUAL_KSPLIT):
                acc_res += tl.load(
                    acc_ptr
                    + ks * stride_acc_k
                    + rm[:, None] * stride_acc_m
                    + rn_res_global[None, :] * stride_acc_n,
                    mask=(rm[:, None] < M) & (rn_res_local[None, :] < n_squared),
                    other=0.0,
                )
            _mhc_reduce_apply_res_block(
                acc_res,
                rsigma,
                rm,
                rn_res_local,
                rn_res_global,
                alpha_res,
                bias_ptr,
                out_ptr,
                M,
                n,
                n_squared,
                N_POW2_RES,
                stride_out_m,
                stride_out_n,
                BLOCK_M,
                NUM_SINKHORN_ITERS,
            )
