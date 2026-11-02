# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import (
    pid_grid,
    remap_xcd,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

_fused_gemm_a16w16_mul_add_repr = make_kernel_repr(
    "_fused_gemm_a16w16_mul_add_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "EVEN_K",
        "EVEN_MN",
        "cache_modifier",
        "ADD_BIAS",
        "IS_A_SCALAR",
        "IS_B_SCALAR",
        "NUM_XCDS",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "EVEN_MN": lambda args: (args["M"] % args["BLOCK_SIZE_M"] == 0)
        and (args["N"] % args["BLOCK_SIZE_N"] == 0),
    }
)
@triton.jit(
    repr=_fused_gemm_a16w16_mul_add_repr,
    do_not_specialize=["M", "N", "S"],
)
def _fused_gemm_a16w16_mul_add_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    c_a_ptr,
    c_b_ptr,
    M,
    N,
    K,
    S,
    stride_ab,
    stride_as,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_cab,
    stride_cas,
    stride_can,
    stride_cbb,
    stride_cbs,
    stride_cbn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    IS_A_SCALAR: tl.constexpr,
    IS_B_SCALAR: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """Kernel for C = c_a * (A x B + bias) + c_b.

    A has shape (M, K) with M = batch * S. Row m of A is at batch m // S and
    position m % S. B has shape (K, N) and C has shape (M, N). c_a and c_b are
    scalars or (batch, S, N) views that can have zero strides.
    """

    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid = remap_xcd(pid, num_pid_m * num_pid_n, NUM_XCDS=NUM_XCDS)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if EVEN_MN:
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    else:
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_ab = offs_am // S
    offs_as = offs_am - offs_ab * S

    a_ptrs = a_ptr + (
        (offs_ab * stride_ab + offs_as * stride_as)[:, None]
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Start from zero and add the bias after the loop. On gfx11, a bias
    # broadcast as the start value made the 64x128 tile up to 2x slower.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
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
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if ADD_BIAS:
        accumulator += tl.load(bias_ptr + offs_bn).to(tl.float32)[None, :]

    # Keep the epilogue offsets in int32, so that the loads can use buffer
    # instructions. The wrapper checks that the offsets fit in int32.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    offs_cb = offs_cm // S
    offs_cs = offs_cm - offs_cb * S

    if IS_A_SCALAR:
        c_a = c_a_ptr
    else:
        c_a_ptrs = c_a_ptr + (
            (offs_cb * stride_cab + offs_cs * stride_cas)[:, None]
            + offs_cn[None, :] * stride_can
        )
        if EVEN_MN:
            c_a = tl.load(c_a_ptrs).to(tl.float32)
        else:
            c_a = tl.load(c_a_ptrs, mask=c_mask, other=0.0).to(tl.float32)

    if IS_B_SCALAR:
        c_b = c_b_ptr
    else:
        c_b_ptrs = c_b_ptr + (
            (offs_cb * stride_cbb + offs_cs * stride_cbs)[:, None]
            + offs_cn[None, :] * stride_cbn
        )
        if EVEN_MN:
            c_b = tl.load(c_b_ptrs).to(tl.float32)
        else:
            c_b = tl.load(c_b_ptrs, mask=c_mask, other=0.0).to(tl.float32)

    c = (c_a * accumulator + c_b).to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    if EVEN_MN:
        tl.store(c_ptrs, c)
    else:
        tl.store(c_ptrs, c, mask=c_mask)


def _get_config(
    M: int,
    N: int,
    K: int,
):
    # On gfx1151 the mul_add epilogue needs a smaller tile than the plain GEMM.
    # Other arches use the base GEMM-A16W16 family.
    if arch_info.get_arch() == "gfx1151":
        return get_gemm_config("FUSED-GEMM-A16W16-MUL_ADD", M, N, K, backend="triton")
    return get_gemm_config("GEMM-A16W16", M, N, K, backend="triton")
