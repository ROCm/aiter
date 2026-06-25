# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_fused_gemm_a16w16_split_cat_repr = make_kernel_repr(
    "_fused_gemm_a16w16_split_cat",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "EVEN_K",
        "cache_modifier",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_fused_gemm_a16w16_split_cat_repr, do_not_specialize=["M", "N"])
def _fused_gemm_a16w16_split_cat(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    y_ptr,
    c1_ptr,
    c2_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    D,
    S1,
    S2,
    S3,
    # Strides
    stride_a_m,
    stride_a_k,
    stride_b_k,
    stride_b_n,
    stride_y_m,
    stride_y_d,
    stride_y_s,
    stride_c1_m,
    stride_c1_d,
    stride_c1_n,
    stride_c2_m,
    stride_c2_d,
    stride_c2_s,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_S3: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """BF16 GEMM C = X @ W^T fused with the split / k_pe-cat / fp8-cast epilogue.

    Computes ``C = a @ b`` (a=[M, K] bf16, b=[K, N] bf16), reshapes the N axis to
    ``[D, S1 + S2]`` and writes:
      * c1 = [M, D, S1 + S3]  -> K = [k_nope(S1) | k_pe(S3) from y]
      * c2 = [M, D, S2]       -> V = [v(S2)]
    Both outputs are written directly in ``c1_ptr``'s element type (fp8 for MLA
    prefill), giving a direct cast with scale 1.0 — matching ``.to(fp8)``.

    NOTE: N must be D * (S1 + S2).
    """
    tl.assume(stride_a_m > 0)
    tl.assume(stride_a_k > 0)
    tl.assume(stride_b_k > 0)
    tl.assume(stride_b_n > 0)
    tl.assume(stride_y_m > 0)
    tl.assume(stride_y_s > 0)
    tl.assume(stride_c1_m > 0)
    tl.assume(stride_c1_d > 0)
    tl.assume(stride_c1_n > 0)
    tl.assume(stride_c2_m > 0)
    tl.assume(stride_c2_d > 0)
    tl.assume(stride_c2_s > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # Grouped ordering promotes L2 data reuse.
    pid = tl.program_id(axis=0)
    # remap so that XCDs get continuous chunks of pids.
    pid = remap_xcd(pid, GRID_MN, NUM_XCDS=8)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Create pointers for the first block of A and B input matrices.
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_a_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_b_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (offs_a_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_b_k + offs_b_n[None, :] * stride_b_n)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_iter = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(num_k_iter):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs, cache_modifier=cache_modifier)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(
                b_ptrs,
                mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                other=0.0,
                cache_modifier=cache_modifier,
            )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_a_k
        b_ptrs += BLOCK_SIZE_K * stride_b_k

    c = accumulator.to(c1_ptr.type.element_ty)  # [BLOCK_SIZE_M, BLOCK_SIZE_N]

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_d = offs_n // (S1 + S2)
    offs_s = offs_n % (S1 + S2)

    # Write the k_nope part of K into c1 (offs_s < S1).
    c1_ptrs = (
        c1_ptr
        + stride_c1_m * offs_m[:, None]
        + stride_c1_d * offs_d[None, :]
        + stride_c1_n * offs_s[None, :]
    )
    c1_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D) & (offs_s[None, :] < S1)
    tl.store(c1_ptrs, c, mask=c1_mask)

    # Write V into c2 (offs_s >= S1).
    c2_ptrs = (
        c2_ptr
        + stride_c2_m * offs_m[:, None]
        + stride_c2_d * offs_d[None, :]
        + stride_c2_s * (offs_s[None, :] - S1)
    )
    c2_mask = (
        (offs_m[:, None] < M)
        & (offs_d[None, :] < D)
        & (offs_s[None, :] >= S1)
        & (offs_s[None, :] < S1 + S2)
    )
    tl.store(c2_ptrs, c, mask=c2_mask)

    # Concat y (k_pe) onto K in c1 at the [S1, S1 + S3) slot.
    offs_n = pid_n * BLOCK_SIZE_S3 + tl.arange(0, BLOCK_SIZE_S3).to(tl.int64)
    offs_d = offs_n // S3
    offs_s = offs_n % S3

    y_ptrs = (
        y_ptr
        + stride_y_m * offs_m[:, None]
        + stride_y_d * offs_d[None, :]
        + stride_y_s * offs_s[None, :]
    )
    y_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D) & (offs_s[None, :] < S3)
    y = tl.load(y_ptrs, mask=y_mask)

    c1_ptrs = (
        c1_ptr
        + stride_c1_m * offs_m[:, None]
        + stride_c1_d * offs_d[None, :]
        + stride_c1_n * (offs_s[None, :] + S1)
    )
    tl.store(c1_ptrs, y, mask=y_mask)
