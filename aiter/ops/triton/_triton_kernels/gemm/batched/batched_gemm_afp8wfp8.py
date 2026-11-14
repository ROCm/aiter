# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import (
    pid_grid,
    remap_xcd,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

_batched_gemm_afp8wfp8_repr = make_kernel_repr(
    "_batched_gemm_afp8wfp8_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "B_SCALE_N_GROUP",
        "EVEN_K",
        "cache_modifier",
    ],
)


@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0})
@triton.jit(repr=_batched_gemm_afp8wfp8_repr)
def _batched_gemm_afp8wfp8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    stride_asb,
    stride_asm,
    stride_ask,
    stride_bsb,
    stride_bsn,
    stride_bsk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    B_SCALE_N_GROUP: tl.constexpr,
    EVEN_K: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """C[b] = A[b] @ B[b] with E4M3 A (M, K), E4M3 B (K, N) and E8M0 scales.

    A scales are (M, K // 32). B scales are (ceil(N / B_SCALE_N_GROUP), K // 32):
    one scale for each 1x32 or 32x32 block of the (N, K) weight.
    """
    tl.assume(stride_ab > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bb > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cb > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_asb > 0)
    tl.assume(stride_asm > 0)
    tl.assume(stride_ask > 0)
    tl.assume(stride_bsb > 0)
    tl.assume(stride_bsn > 0)
    tl.assume(stride_bsk > 0)
    SCALE_GROUP_SIZE: tl.constexpr = 32
    tl.static_assert(BLOCK_SIZE_K % SCALE_GROUP_SIZE == 0)

    pid_batch = tl.program_id(axis=0).to(tl.int64)
    pid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid = remap_xcd(pid, num_pid_m * num_pid_n)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Move to this batch in int64, so that large batches do not overflow.
    a_ptr += pid_batch * stride_ab
    b_ptr += pid_batch * stride_bb
    c_ptr += pid_batch * stride_cb
    a_scales_ptr += pid_batch * stride_asb
    b_scales_ptr += pid_batch * stride_bsb

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_ks = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    m_mask = offs_am < M
    n_mask = offs_bn < N
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    a_scale_ptrs = (
        a_scales_ptr + offs_am[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    )
    # B scales are (N, K) although B is (K, N). A 32x32 block shares one row.
    b_scale_ptrs = (
        b_scales_ptr
        + (offs_bn // B_SCALE_N_GROUP)[:, None] * stride_bsn
        + offs_ks[None, :] * stride_bsk
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
            b = tl.load(
                b_ptrs, mask=n_mask[None, :], other=0.0, cache_modifier=cache_modifier
            )
            a_scales = tl.load(a_scale_ptrs, mask=m_mask[:, None], other=0)
            b_scales = tl.load(
                b_scale_ptrs,
                mask=n_mask[:, None],
                other=0,
                cache_modifier=cache_modifier,
            )
        else:
            k_left = K - k * BLOCK_SIZE_K
            ks_mask = offs_ks[None, :] < k_left // SCALE_GROUP_SIZE
            a = tl.load(
                a_ptrs, mask=m_mask[:, None] & (offs_k[None, :] < k_left), other=0.0
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_left) & n_mask[None, :],
                other=0.0,
                cache_modifier=cache_modifier,
            )
            a_scales = tl.load(a_scale_ptrs, mask=m_mask[:, None] & ks_mask, other=0)
            b_scales = tl.load(
                b_scale_ptrs,
                mask=n_mask[:, None] & ks_mask,
                other=0,
                cache_modifier=cache_modifier,
            )
        accumulator = tl.dot_scaled(
            a, a_scales, "e4m3", b, b_scales, "e4m3", accumulator
        )
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        a_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask
        b_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bsk

    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(
        c_ptrs,
        accumulator.to(c_ptr.type.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def _get_config(
    B: int,
    M: int,
    N: int,
    K: int,
    w_scale_group_size: tuple[int, int],
    backend: str = "triton",
):
    group_n, group_k = w_scale_group_size
    return get_gemm_config(
        f"BATCHED_GEMM-AFP8WFP8_A32_W{group_n}X{group_k}",
        M,
        N,
        K,
        B=B,
        backend=backend,
    )
