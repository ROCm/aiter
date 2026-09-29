# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_transpose_2d_kernel_repr = make_kernel_repr(
    "_transpose_2d_kernel", ["BLOCK_M", "BLOCK_N"]
)

_transpose_packed_fp4_kernel_repr = make_kernel_repr(
    "_transpose_packed_fp4_kernel", ["BLOCK_M", "BLOCK_N_PACKED"]
)


@triton.jit(repr=_transpose_2d_kernel_repr)
def _transpose_2d_kernel(
    IN_ptr,
    OUT_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_n,
    stride_out_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Tiled 2D matrix transpose.

    Reads tiles from (M, N) input and writes transposed tiles to (N, M) output.
    Works with any element dtype including FP8 (e4m3, e5m2, fnuz variants).
    Whether the transposed write is staged through LDS is compiler-determined;
    the benchmark against ``t().contiguous()`` measures the net effect.
    """
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    offs_m = tl.cast(pid_m * BLOCK_M + tl.arange(0, BLOCK_M), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_N + tl.arange(0, BLOCK_N), tl.int64)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    in_ptrs = IN_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_n
    vals = tl.load(in_ptrs, mask=mask)

    out_ptrs = OUT_ptr + offs_n[None, :] * stride_out_n + offs_m[:, None] * stride_out_m
    tl.store(out_ptrs, vals, mask=mask)


@triton.jit(repr=_transpose_packed_fp4_kernel_repr)
def _transpose_packed_fp4_kernel(
    input_ptr,
    output_ptr,
    M,
    N_PACKED,
    stride_input_m,
    stride_input_n,
    stride_output_m,
    stride_output_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N_PACKED: tl.constexpr,
):
    """Transpose the logical matrix stored as two FP4 nibbles per byte."""
    tl.static_assert(BLOCK_M % 2 == 0)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    stride_input_m = tl.cast(stride_input_m, tl.int64)
    stride_input_n = tl.cast(stride_input_n, tl.int64)
    stride_output_m = tl.cast(stride_output_m, tl.int64)
    stride_output_n = tl.cast(stride_output_n, tl.int64)

    BLOCK_N: tl.constexpr = BLOCK_N_PACKED * 2
    BLOCK_M_PACKED: tl.constexpr = BLOCK_M // 2

    offsets_m = tl.cast(pid_m, tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n_packed = tl.cast(pid_n, tl.int64) * BLOCK_N_PACKED + tl.arange(
        0, BLOCK_N_PACKED
    )
    input_mask = (offsets_m[:, None] < M) & (offsets_n_packed[None, :] < N_PACKED)
    packed = tl.load(
        input_ptr
        + offsets_m[:, None] * stride_input_m
        + offsets_n_packed[None, :] * stride_input_n,
        mask=input_mask,
        other=0,
    ).to(tl.uint8)

    low_nibbles = packed & 0x0F
    high_nibbles = (packed >> 4) & 0x0F
    unpacked = tl.reshape(tl.join(low_nibbles, high_nibbles), (BLOCK_M, BLOCK_N))
    transposed = tl.trans(unpacked)

    transposed_pairs = tl.reshape(transposed, (BLOCK_N, BLOCK_M_PACKED, 2))
    low_nibbles, high_nibbles = tl.split(transposed_pairs)
    output = low_nibbles | (high_nibbles << 4)

    offsets_output_m = tl.cast(pid_n, tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_output_n = tl.cast(pid_m, tl.int64) * BLOCK_M_PACKED + tl.arange(
        0, BLOCK_M_PACKED
    )
    output_mask = (offsets_output_m[:, None] < tl.cast(N_PACKED, tl.int64) * 2) & (
        offsets_output_n[None, :] < tl.cast(M, tl.int64) // 2
    )
    tl.store(
        output_ptr
        + offsets_output_m[:, None] * stride_output_m
        + offsets_output_n[None, :] * stride_output_n,
        output,
        mask=output_mask,
    )
