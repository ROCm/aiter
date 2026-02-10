# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton conv2d kernels using implicit-GEMM (im2col fused into matmul).

Two kernels are provided, differing only in K-decomposition order to
optimise memory coalescing for the respective data layout:

* ``_conv2d_implicit_gemm_kernel``  -- **NCHW** input/output
    K index order: ``(ci, kh, kw)`` with ``kw`` varying fastest.
    Consecutive K loads touch adjacent W addresses (stride 1 in NCHW).

* ``_conv2d_nhwc_kernel``           -- **NHWC** input/output
    K index order: ``(kh, kw, ci)`` with ``ci`` varying fastest.
    Consecutive K loads touch adjacent channel addresses (stride 1 in NHWC).
    Weight is expected in ``(Co, kH, kW, Ci/g)`` layout so that weight
    loads along the K axis are also coalesced.

Both kernels map the convolution to a GEMM:
    M = N_batch * H_out * W_out   (output spatial positions)
    K = C_in_per_group * kH * kW  (filter volume)
    N = C_out_per_group            (output channels per group)
"""

import triton
import triton.language as tl


@triton.jit
def _conv2d_implicit_gemm_kernel(
    # Pointers
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    # Input dims
    batch_size,
    in_channels,
    in_height,
    in_width,
    # Output dims
    out_channels,
    out_height,
    out_width,
    # Kernel dims
    kernel_h,
    kernel_w,
    # Conv params
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    # Strides for input (N, C, H, W)
    stride_in_n,
    stride_in_c,
    stride_in_h,
    stride_in_w,
    # Strides for weight (Co, Ci/g, kH, kW)
    stride_wt_co,
    stride_wt_ci,
    stride_wt_kh,
    stride_wt_kw,
    # Strides for output (N, Co, Ho, Wo)
    stride_out_n,
    stride_out_c,
    stride_out_h,
    stride_out_w,
    # Derived
    M,  # batch_size * out_height * out_width
    N,  # out_channels_per_group
    K,  # in_channels_per_group * kernel_h * kernel_w
    ci_per_group,  # in_channels // groups
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Implicit-GEMM conv2d kernel.  One program per (group, tile_m, tile_n)."""
    pid = tl.program_id(0)
    group_id = tl.program_id(1)

    # Tile indices within the M x N output matrix
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Row / col offsets for the output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Decode spatial position from linear M index
    # M = batch_size * out_height * out_width
    hw = out_height * out_width
    batch_idx = offs_m // hw            # [BLOCK_M]
    remainder = offs_m % hw
    oh = remainder // out_width         # [BLOCK_M]
    ow = remainder % out_width          # [BLOCK_M]

    # Channel offset for this group
    co_offset = group_id * N  # output channel offset
    ci_offset = group_id * ci_per_group  # input channel offset

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # kH * kW for decomposition
    kHkW = kernel_h * kernel_w

    # Main loop over K dimension (ci_per_group * kH * kW)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = offs_k < K

        # Decode K index into (ci_local, kh, kw)
        ci_local = offs_k // kHkW             # [BLOCK_K]
        k_rem = offs_k % kHkW
        kh = k_rem // kernel_w                # [BLOCK_K]
        kw = k_rem % kernel_w                 # [BLOCK_K]

        # Compute input spatial coords for each (m, k) pair
        # ih = oh * stride_h + kh * dilation_h - pad_h
        # iw = ow * stride_w + kw * dilation_w - pad_w
        ih = oh[:, None] * stride_h + kh[None, :] * dilation_h - pad_h  # [BLOCK_M, BLOCK_K]
        iw = ow[:, None] * stride_w + kw[None, :] * dilation_w - pad_w  # [BLOCK_M, BLOCK_K]

        # Bounds check
        valid_h = (ih >= 0) & (ih < in_height)
        valid_w = (iw >= 0) & (iw < in_width)
        valid = valid_h & valid_w & mask_m[:, None] & mask_k[None, :]

        # Input pointer: input[batch_idx, ci_offset + ci_local, ih, iw]
        ci_abs = ci_offset + ci_local  # [BLOCK_K]
        inp_ptrs = (
            input_ptr
            + batch_idx[:, None] * stride_in_n
            + ci_abs[None, :] * stride_in_c
            + ih * stride_in_h
            + iw * stride_in_w
        )
        a = tl.load(inp_ptrs, mask=valid, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Weight pointer: weight[co_offset + offs_n, ci_local, kh, kw]
        co_abs = co_offset + offs_n  # [BLOCK_N]
        wt_ptrs = (
            weight_ptr
            + co_abs[:, None] * stride_wt_co
            + ci_local[None, :] * stride_wt_ci
            + kh[None, :] * stride_wt_kh
            + kw[None, :] * stride_wt_kw
        )
        wt_mask = mask_n[:, None] & mask_k[None, :]
        b = tl.load(wt_ptrs, mask=wt_mask, other=0.0)  # [BLOCK_N, BLOCK_K]

        # Accumulate: acc[m, n] += sum_k a[m, k] * b[n, k]
        acc += tl.dot(a, tl.trans(b), input_precision="ieee")

    # Add bias
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + co_offset + offs_n, mask=mask_n, other=0.0)
        acc += bias_vals[None, :]

    # Store output: output[batch_idx, co_offset + offs_n, oh, ow]
    co_abs = co_offset + offs_n
    out_ptrs = (
        output_ptr
        + batch_idx[:, None] * stride_out_n
        + co_abs[None, :] * stride_out_c
        + oh[:, None] * stride_out_h
        + ow[:, None] * stride_out_w
    )
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


# -----------------------------------------------------------------------
# NHWC-optimised kernel
# -----------------------------------------------------------------------

@triton.jit
def _conv2d_nhwc_kernel(
    # Pointers
    input_ptr,
    weight_ptr,      # expected layout: (Co, kH, kW, Ci/g)
    bias_ptr,
    output_ptr,
    # Input dims
    batch_size,
    in_channels,
    in_height,
    in_width,
    # Output dims
    out_channels,
    out_height,
    out_width,
    # Kernel dims
    kernel_h,
    kernel_w,
    # Conv params
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    # Strides for input – logical (N, H, W, C)
    stride_in_n,
    stride_in_h,
    stride_in_w,
    stride_in_c,
    # Strides for weight – logical (Co, kH, kW, Ci/g)
    stride_wt_co,
    stride_wt_kh,
    stride_wt_kw,
    stride_wt_ci,
    # Strides for output – logical (N, Ho, Wo, Co)
    stride_out_n,
    stride_out_h,
    stride_out_w,
    stride_out_c,
    # Derived
    M,  # batch_size * out_height * out_width
    N,  # out_channels_per_group
    K,  # ci_per_group * kernel_h * kernel_w
    ci_per_group,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Implicit-GEMM conv2d kernel optimised for NHWC layout.

    The K dimension is decomposed as ``(kh, kw, ci)`` so that consecutive
    K indices map to consecutive channel addresses (stride-1 in NHWC).
    Weight is expected in ``(Co, kH, kW, Ci/g)`` for the same reason.
    """
    pid = tl.program_id(0)
    group_id = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Decode M -> (batch, oh, ow)
    hw = out_height * out_width
    batch_idx = offs_m // hw
    remainder = offs_m % hw
    oh = remainder // out_width
    ow = remainder % out_width

    co_offset = group_id * N
    ci_offset = group_id * ci_per_group

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pre-compute ci_per_group * kernel_w for K decomposition
    kw_ci = kernel_w * ci_per_group

    # Main K loop – decomposition: kh outermost, ci innermost
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # NHWC decomposition: (kh, kw, ci) with ci fastest
        kh = offs_k // kw_ci
        kw = (offs_k % kw_ci) // ci_per_group
        ci_local = offs_k % ci_per_group

        # Input spatial coordinates
        ih = oh[:, None] * stride_h + kh[None, :] * dilation_h - pad_h
        iw = ow[:, None] * stride_w + kw[None, :] * dilation_w - pad_w

        valid_h = (ih >= 0) & (ih < in_height)
        valid_w = (iw >= 0) & (iw < in_width)
        valid = valid_h & valid_w & mask_m[:, None] & mask_k[None, :]

        # Input: layout (N, H, W, C)
        ci_abs = ci_offset + ci_local
        inp_ptrs = (
            input_ptr
            + batch_idx[:, None] * stride_in_n
            + ih * stride_in_h
            + iw * stride_in_w
            + ci_abs[None, :] * stride_in_c
        )
        a = tl.load(inp_ptrs, mask=valid, other=0.0)

        # Weight: layout (Co, kH, kW, Ci/g)
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

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + co_offset + offs_n, mask=mask_n, other=0.0)
        acc += bias_vals[None, :]

    # Store output: layout (N, Ho, Wo, Co)
    co_abs = co_offset + offs_n
    out_ptrs = (
        output_ptr
        + batch_idx[:, None] * stride_out_n
        + oh[:, None] * stride_out_h
        + ow[:, None] * stride_out_w
        + co_abs[None, :] * stride_out_c
    )
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=out_mask)
