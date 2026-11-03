# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import pid_grid
from aiter.ops.triton._triton_kernels.moe.activations import silu
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_moe_gemm_a16w16_repr = make_kernel_repr(
    "moe_gemm_a16w16",
    ["BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "XCD_SWIZZLE"],
)


@triton.jit
def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: tl.constexpr):
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE
    return group * pids_per_group + tl.minimum(group, extra_pid_groups) + local_pid


@triton.jit(repr=_moe_gemm_a16w16_repr)
def _moe_gemm_a16w16(
    Y,
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    W,
    stride_w_e,
    stride_w_n,
    stride_w_k,
    Gammas,
    N,
    K,
    GatherIndx,
    ExptHist,
    ExptOffs,
    ExptOffsSum,
    ExptData,
    grid_m,
    grid_n,
    APPLY_SEPARATED_SILU: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    XCD_SWIZZLE: tl.constexpr,
    EVEN_K: tl.constexpr,
    MASK_K_LIMIT: tl.constexpr,
    W_CACHE_MODIFIER: tl.constexpr,
    UPCAST_INDICES: tl.constexpr = False,
):
    tl.assume(stride_y_k >= 0)
    tl.assume(stride_y_m >= 0)
    tl.assume(stride_y_n >= 0)
    tl.assume(stride_x_m >= 0)
    tl.assume(stride_x_k >= 0)
    tl.assume(stride_w_e >= 0)
    tl.assume(stride_w_n >= 0)
    tl.assume(stride_w_k >= 0)
    tl.assume(grid_m >= 0)
    tl.assume(grid_n >= 0)

    yN: tl.constexpr = N // 2 if APPLY_SEPARATED_SILU else N

    pid = tl.program_id(0)
    if ExptOffsSum is not None and XCD_SWIZZLE > 1:
        padding_m = grid_m - tl.load(ExptOffsSum)
    else:
        padding_m: tl.constexpr = 0

    index_type: tl.constexpr = tl.int64 if UPCAST_INDICES else tl.int32
    unpadded_m = grid_m - padding_m
    tl.assume(unpadded_m >= 0)
    total_actual_tiles = unpadded_m * grid_n
    if padding_m > 0 and pid >= total_actual_tiles:
        return

    pid_emn = pid
    if XCD_SWIZZLE != 1:
        pid_emn = xcd_swizzle(pid_emn, total_actual_tiles, XCD_SWIZZLE)
    pid_m, pid_n = pid_grid(pid_emn, unpadded_m, grid_n, GROUP_M)

    expt_data = tl.load(ExptData + pid_m)
    if expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = tl.load(ExptHist + expt_id)
    start_m = tl.load(ExptOffs + expt_id)

    expt_id = expt_id.to(index_type)
    block_id = block_id.to(index_type)
    start_m = start_m.to(index_type)
    pid_n = pid_n.to(index_type)

    offs_x_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
    if GatherIndx is None:
        X += start_m * stride_x_m
    else:
        GatherIndx += start_m
        offs_x_m = tl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT

    offs_x_k = tl.arange(0, BLOCK_K)
    XPtrs = X + offs_x_m.to(index_type)[:, None] * stride_x_m + offs_x_k[
        None, :
    ].to(index_type) * stride_x_k

    offs_w_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_w_n = tl.max_contiguous(tl.multiple_of(offs_w_n % yN, BLOCK_N), BLOCK_N)
    offs_w_k = tl.arange(0, BLOCK_K)
    W += expt_id * stride_w_e
    WPtrs = W + offs_w_n.to(index_type)[None, :] * stride_w_n + offs_w_k[
        :, None
    ].to(index_type) * stride_w_k
    WPtrsUp = (
        W
        + (offs_w_n + yN).to(index_type)[None, :] * stride_w_n
        + offs_w_k[:, None].to(index_type) * stride_w_k
    )

    num_k_iter = tl.cdiv(K, BLOCK_K)
    if not EVEN_K:
        num_k_iter -= 1

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(num_k_iter):
        x = tl.load(XPtrs)
        w = tl.load(WPtrs, cache_modifier=W_CACHE_MODIFIER)
        acc = tl.dot(x, w, acc=acc)
        if APPLY_SEPARATED_SILU:
            w_up = tl.load(WPtrsUp, cache_modifier=W_CACHE_MODIFIER)
            acc_up = tl.dot(x, w_up, acc=acc_up)
        XPtrs += BLOCK_K * stride_x_k
        WPtrs += BLOCK_K * stride_w_k
        WPtrsUp += BLOCK_K * stride_w_k

    if not EVEN_K:
        mask_k = offs_x_k < MASK_K_LIMIT
        x = tl.load(XPtrs, mask=mask_k[None, :], other=0.0)
        w = tl.load(WPtrs, mask=mask_k[:, None], other=0.0, cache_modifier=W_CACHE_MODIFIER)
        acc = tl.dot(x, w, acc=acc)
        if APPLY_SEPARATED_SILU:
            w_up = tl.load(
                WPtrsUp, mask=mask_k[:, None], other=0.0, cache_modifier=W_CACHE_MODIFIER
            )
            acc_up = tl.dot(x, w_up, acc=acc_up)

    offs_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_y_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_y_n < N
    if APPLY_SEPARATED_SILU:
        mask_n = offs_y_n < yN
        out = silu(acc) * acc_up
    else:
        out = acc

    if Gammas is not None:
        gammas = tl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        out *= gammas[:, None]

    Y += start_m * stride_y_m
    offs_y_m = offs_m
    YPtrs = Y + offs_y_m.to(index_type)[:, None] * stride_y_m + offs_y_n[
        None, :
    ].to(index_type) * stride_y_n
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(YPtrs, out, mask=mask)
