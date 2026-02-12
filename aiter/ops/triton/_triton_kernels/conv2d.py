# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton conv2d kernels using implicit-GEMM (im2col fused into matmul).

Two kernels are provided, differing only in K-decomposition order to
optimise memory coalescing for the respective data layout:

* ``_conv2d_implicit_gemm_kernel``  -- **NCHW** input/output
* ``_conv2d_nhwc_kernel``           -- **NHWC** input/output

Both support **split-K**: the K-reduction is distributed across multiple
workgroups (``SPLIT_K``), each accumulating a partial sum into a temporary
fp32 buffer that is reduced by the wrapper after the kernel completes.

Grid: ``(tiles_M * tiles_N, groups, SPLIT_K)``
"""

import triton
import triton.language as tl


# -------------------------------------------------------------------
# Block-size heuristic
# -------------------------------------------------------------------

_BLOCK_CANDIDATES = [
    # BM   BN   BK  warps
    (16,  16,  32,  4),
    (16,  32,  32,  4),
    (16,  64,  32,  4),
    (16, 128,  32,  4),
    (32,  16,  32,  4),
    (32,  32,  32,  4),
    (32,  64,  32,  4),
    (32,  64,  64,  4),
    (32, 128,  32,  4),
    (64,  64,  32,  4),
    (64,  64,  64,  4),
    (64, 128,  32,  8),
    (128, 64,  32,  8),
    (128,128,  32,  8),
]

# Target: 2 full waves on a 256 CU GPU.
_TARGET_TILES = 512


def pick_conv2d_block_config(M: int, N: int, K: int, groups: int = 1):
    """Return (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, split_k).

    Maximises the number of workgroups (tiles * groups * split_k) to
    fill the GPU, while keeping each tile and each K-split large enough
    for efficient compute.
    """
    best = (64, 64, 32, 4, 1)
    best_score = -1

    for bm, bn, bk, nw in _BLOCK_CANDIDATES:
        if bk > K:
            bk = max(16, triton.next_power_of_2(K))
        if bn > 4 * max(N, 1):
            continue

        tiles_m = triton.cdiv(M, bm)
        tiles_n = triton.cdiv(N, bn)
        base_tiles = tiles_m * tiles_n * groups

        # Determine split_k to reach _TARGET_TILES workgroups
        k_iters = triton.cdiv(K, bk)
        sk = 1
        while base_tiles * sk < _TARGET_TILES and sk < k_iters // 2:
            sk *= 2
        # Cap: each split must do at least 4 K-iterations for efficiency
        max_sk = max(1, k_iters // 4)
        sk = min(sk, max_sk, 32)

        total_wgs = base_tiles * sk

        # Penalise padding waste
        waste_m = (tiles_m * bm - M) / max(M, 1)
        waste_n = (tiles_n * bn - N) / max(N, 1)
        waste = waste_m + waste_n

        # Primary: workgroup count; secondary: lower waste
        score = total_wgs * 1000 - int(waste * 200)
        # Slight penalty if split_k > 1 (reduction overhead)
        score -= sk * 10

        if score > best_score:
            best_score = score
            best = (bm, bn, bk, nw, sk)

    return best


# -------------------------------------------------------------------
# NCHW kernel  (K decomposition: ci outermost, kw innermost)
# -------------------------------------------------------------------

@triton.jit
def _conv2d_implicit_gemm_kernel(
    # Pointers
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    # Input dims
    batch_size, in_channels, in_height, in_width,
    # Output dims
    out_channels, out_height, out_width,
    # Kernel dims
    kernel_h, kernel_w,
    # Conv params
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
    # Strides for input (N, C, H, W)
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    # Strides for weight (Co, Ci/g, kH, kW)
    stride_wt_co, stride_wt_ci, stride_wt_kh, stride_wt_kw,
    # Strides for output (N, Co, Ho, Wo) – or temp buffer when SPLIT_K>1
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    # GEMM dims
    M, N, K, ci_per_group,
    # Split-K
    k_per_split,       # ceil(K / SPLIT_K)
    split_k_stride,    # elements between SPLIT_K slices in output buffer
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    group_id = tl.program_id(1)
    pid_k = tl.program_id(2)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    hw = out_height * out_width
    batch_idx = offs_m // hw
    remainder = offs_m % hw
    oh = remainder // out_width
    ow = remainder % out_width

    co_offset = group_id * N
    ci_offset = group_id * ci_per_group

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    kHkW = kernel_h * kernel_w

    # Split-K: this program handles [k_begin, k_end) where
    # k_end = min(k_begin + k_per_split, K).
    k_begin = pid_k * k_per_split
    k_end = k_begin + k_per_split  # may exceed K; mask below handles it

    for k_off in range(0, k_per_split, BLOCK_K):
        offs_k = (k_begin + k_off) + tl.arange(0, BLOCK_K)
        mask_k = (offs_k < K) & (offs_k < k_end)

        ci_local = offs_k // kHkW
        k_rem = offs_k % kHkW
        kh = k_rem // kernel_w
        kw = k_rem % kernel_w

        ih = oh[:, None] * stride_h + kh[None, :] * dilation_h - pad_h
        iw = ow[:, None] * stride_w + kw[None, :] * dilation_w - pad_w

        valid_h = (ih >= 0) & (ih < in_height)
        valid_w = (iw >= 0) & (iw < in_width)
        valid = valid_h & valid_w & mask_m[:, None] & mask_k[None, :]

        ci_abs = ci_offset + ci_local
        inp_ptrs = (
            input_ptr
            + batch_idx[:, None] * stride_in_n
            + ci_abs[None, :] * stride_in_c
            + ih * stride_in_h
            + iw * stride_in_w
        )
        a = tl.load(inp_ptrs, mask=valid, other=0.0)

        co_abs = co_offset + offs_n
        wt_ptrs = (
            weight_ptr
            + co_abs[:, None] * stride_wt_co
            + ci_local[None, :] * stride_wt_ci
            + kh[None, :] * stride_wt_kh
            + kw[None, :] * stride_wt_kw
        )
        wt_mask = mask_n[:, None] & mask_k[None, :]
        b = tl.load(wt_ptrs, mask=wt_mask, other=0.0)

        acc += tl.dot(a, tl.trans(b), input_precision="ieee")

    # Bias: only when not splitting (when SPLIT_K>1, wrapper adds bias)
    if HAS_BIAS and SPLIT_K == 1:
        bias_vals = tl.load(bias_ptr + co_offset + offs_n, mask=mask_n, other=0.0)
        acc += bias_vals[None, :]

    # Store
    co_abs = co_offset + offs_n
    out_ptrs = (
        output_ptr
        + pid_k * split_k_stride
        + batch_idx[:, None] * stride_out_n
        + co_abs[None, :] * stride_out_c
        + oh[:, None] * stride_out_h
        + ow[:, None] * stride_out_w
    )
    out_mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=out_mask)
    else:
        tl.store(out_ptrs, acc, mask=out_mask)


# -------------------------------------------------------------------
# NHWC kernel  (K decomposition: kh outermost, ci innermost)
# -------------------------------------------------------------------

@triton.jit
def _conv2d_nhwc_kernel(
    # Pointers
    input_ptr,
    weight_ptr,      # layout: (Co, kH, kW, Ci/g)
    bias_ptr,
    output_ptr,
    # Input dims
    batch_size, in_channels, in_height, in_width,
    # Output dims
    out_channels, out_height, out_width,
    # Kernel dims
    kernel_h, kernel_w,
    # Conv params
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
    # Strides for input (N, H, W, C)
    stride_in_n, stride_in_h, stride_in_w, stride_in_c,
    # Strides for weight (Co, kH, kW, Ci/g)
    stride_wt_co, stride_wt_kh, stride_wt_kw, stride_wt_ci,
    # Strides for output (N, Ho, Wo, Co) – or temp buffer when SPLIT_K>1
    stride_out_n, stride_out_h, stride_out_w, stride_out_c,
    # GEMM dims
    M, N, K, ci_per_group,
    # Split-K
    k_per_split,
    split_k_stride,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    group_id = tl.program_id(1)
    pid_k = tl.program_id(2)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    hw = out_height * out_width
    batch_idx = offs_m // hw
    remainder = offs_m % hw
    oh = remainder // out_width
    ow = remainder % out_width

    co_offset = group_id * N
    ci_offset = group_id * ci_per_group

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    kw_ci = kernel_w * ci_per_group

    k_begin = pid_k * k_per_split
    k_end = k_begin + k_per_split

    for k_off in range(0, k_per_split, BLOCK_K):
        offs_k = (k_begin + k_off) + tl.arange(0, BLOCK_K)
        mask_k = (offs_k < K) & (offs_k < k_end)

        kh = offs_k // kw_ci
        kw = (offs_k % kw_ci) // ci_per_group
        ci_local = offs_k % ci_per_group

        ih = oh[:, None] * stride_h + kh[None, :] * dilation_h - pad_h
        iw = ow[:, None] * stride_w + kw[None, :] * dilation_w - pad_w

        valid_h = (ih >= 0) & (ih < in_height)
        valid_w = (iw >= 0) & (iw < in_width)
        valid = valid_h & valid_w & mask_m[:, None] & mask_k[None, :]

        ci_abs = ci_offset + ci_local
        inp_ptrs = (
            input_ptr
            + batch_idx[:, None] * stride_in_n
            + ih * stride_in_h
            + iw * stride_in_w
            + ci_abs[None, :] * stride_in_c
        )
        a = tl.load(inp_ptrs, mask=valid, other=0.0)

        co_abs = co_offset + offs_n
        wt_ptrs = (
            weight_ptr
            + co_abs[:, None] * stride_wt_co
            + kh[None, :] * stride_wt_kh
            + kw[None, :] * stride_wt_kw
            + ci_local[None, :] * stride_wt_ci
        )
        wt_mask = mask_n[:, None] & mask_k[None, :]
        b = tl.load(wt_ptrs, mask=wt_mask, other=0.0)

        acc += tl.dot(a, tl.trans(b), input_precision="ieee")

    if HAS_BIAS and SPLIT_K == 1:
        bias_vals = tl.load(bias_ptr + co_offset + offs_n, mask=mask_n, other=0.0)
        acc += bias_vals[None, :]

    co_abs = co_offset + offs_n
    out_ptrs = (
        output_ptr
        + pid_k * split_k_stride
        + batch_idx[:, None] * stride_out_n
        + oh[:, None] * stride_out_h
        + ow[:, None] * stride_out_w
        + co_abs[None, :] * stride_out_c
    )
    out_mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=out_mask)
    else:
        tl.store(out_ptrs, acc, mask=out_mask)
