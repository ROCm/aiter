# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import (
    pid_grid,
    remap_xcd,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

_gemm_a16w16x3_repr = make_kernel_repr(
    "_gemm_a16w16x3_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
        "cache_modifier",
        "EVEN_K",
        "GRID_MN",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"]) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % (args["SPLITK_BLOCK_SIZE"]) == 0),
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_gemm_a16w16x3_repr)
def _gemm_a16w16x3_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    ws_ptr,
    cnt_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bt,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    cache_modifier: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """C (fp32) = A @ (B[0] + B[1] + B[2])^T, with A (M, K) bf16 and B (3, N, K) bf16.

    Each K step loads A once and adds three MFMA dots into one fp32 accumulator. With
    NUM_KSPLIT > 1, the last split of a tile adds all partials in split order.
    """
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bt > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS=NUM_XCDS)
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    k_start = pid_k * SPLITK_BLOCK_SIZE
    if k_start < K:
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)
        k_end = tl.minimum(K, k_start + SPLITK_BLOCK_SIZE)

        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = k_start + offs_k
        offs_am = (pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )

        for k in range(num_k_iter):
            if EVEN_K:
                a = tl.load(a_ptrs)
                b0 = tl.load(b_ptrs, cache_modifier=cache_modifier)
                b1 = tl.load(b_ptrs + stride_bt, cache_modifier=cache_modifier)
                b2 = tl.load(b_ptrs + 2 * stride_bt, cache_modifier=cache_modifier)
            else:
                k_left = k_end - (k_start + k * BLOCK_SIZE_K)
                a = tl.load(a_ptrs, mask=offs_k[None, :] < k_left, other=0.0)
                b_mask = offs_k[:, None] < k_left
                b0 = tl.load(b_ptrs, mask=b_mask, other=0.0)
                b1 = tl.load(b_ptrs + stride_bt, mask=b_mask, other=0.0)
                b2 = tl.load(b_ptrs + 2 * stride_bt, mask=b_mask, other=0.0)

            accumulator = tl.dot(a, b0, acc=accumulator)
            accumulator = tl.dot(a, b1, acc=accumulator)
            accumulator = tl.dot(a, b2, acc=accumulator)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if NUM_KSPLIT == 1:
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        # Partials go write-through and are read volatile: the last split can run
        # on another XCD. The split order of the sum makes the result deterministic.
        ws_tile = offs_cm[:, None] * N + offs_cn[None, :]
        tl.store(
            ws_ptr + pid_k * M * N + ws_tile,
            accumulator,
            mask=c_mask,
            cache_modifier=".wt",
        )
        tile_id = pid_m * num_pid_n + pid_n
        arrived = tl.atomic_add(cnt_ptr + tile_id, 1, sem="acq_rel", scope="gpu")
        if arrived == NUM_KSPLIT - 1:
            total = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for s in range(NUM_KSPLIT):
                total += tl.load(
                    ws_ptr + s * M * N + ws_tile,
                    mask=c_mask,
                    other=0.0,
                    cache_modifier=".cv",
                )
            tl.store(c_ptrs, total, mask=c_mask)
            tl.atomic_xchg(cnt_ptr + tile_id, 0, sem="relaxed", scope="gpu")


def _get_config(M: int, N: int, K: int, backend: str = "triton"):
    return get_gemm_config("GEMM-A16W16X3", M, N, K, backend=backend)
