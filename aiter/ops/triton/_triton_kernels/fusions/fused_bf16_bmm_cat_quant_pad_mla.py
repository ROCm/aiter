# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import pid_grid
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_fused_bf16_bmm_cat_quant_pad_mla_repr = make_kernel_repr(
    "_fused_bf16_bmm_cat_quant_pad_mla_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "DUP",
        "EVEN_K",
        "cache_modifier",
    ],
)


@triton.jit
def _quant_pad_store(
    y,
    out_ptr,
    offs_m,
    offs_n,
    mask,
    pid_h,
    inv_scale,
    stride_om,
    stride_oh,
    stride_od,
    FP8_MAX: tl.constexpr,
    DUP: tl.constexpr,
):
    """Quantize the fp32 tile y and store it into every copy of head pid_h."""
    q = tl.clamp(y * inv_scale, -FP8_MAX, FP8_MAX).to(out_ptr.dtype.element_ty)
    for d in tl.static_range(DUP):
        tl.store(
            out_ptr
            + offs_m[:, None] * stride_om
            + (pid_h * DUP + d) * stride_oh
            + offs_n[None, :] * stride_od,
            q,
            mask=mask,
        )


@triton.jit
def _quant_store_pe(
    pe_ptr,
    out_ptr,
    row_m,
    row_h,
    row_mask,
    inv_scale,
    N,
    stride_rm,
    stride_rh,
    stride_rd,
    stride_om,
    stride_oh,
    stride_od,
    FP8_MAX: tl.constexpr,
    D_PE: tl.constexpr,
    BLOCK_D_PE: tl.constexpr,
    DUP: tl.constexpr,
):
    """Quantize q_pe[row_m[i], row_h[i], :] into the pe columns of every copy of
    head row_h[i]."""
    offs_d = tl.arange(0, BLOCK_D_PE)
    mask = row_mask[:, None] & (offs_d[None, :] < D_PE)
    pe = tl.load(
        pe_ptr
        + row_m[:, None] * stride_rm
        + row_h[:, None] * stride_rh
        + offs_d[None, :] * stride_rd,
        mask=mask,
        other=0.0,
    )
    y = tl.clamp(pe.to(tl.float32) * inv_scale, -FP8_MAX, FP8_MAX)
    y = y.to(out_ptr.dtype.element_ty)
    for d in tl.static_range(DUP):
        tl.store(
            out_ptr
            + row_m[:, None] * stride_om
            + (row_h[:, None] * DUP + d) * stride_oh
            + (N + offs_d[None, :]) * stride_od,
            y,
            mask=mask,
        )


@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0})
@triton.jit(repr=_fused_bf16_bmm_cat_quant_pad_mla_repr)
def _fused_bf16_bmm_cat_quant_pad_mla_kernel(
    a_ptr,
    b_ptr,
    pe_ptr,
    scale_ptr,
    out_ptr,
    M,
    N,
    K,
    QH,
    stride_ah,
    stride_am,
    stride_ak,
    stride_bh,
    stride_bk,
    stride_bn,
    stride_rm,
    stride_rh,
    stride_rd,
    stride_om,
    stride_oh,
    stride_od,
    FP8_MAX: tl.constexpr,
    D_PE: tl.constexpr,
    BLOCK_D_PE: tl.constexpr,
    DUP: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call
    fused_bf16_bmm_cat_quant_pad_mla function below.

    out[m, h * DUP + d, :N] = fp8(clamp(bf16(A[h] @ B[h])[m] / scale))
    out[m, h * DUP + d, N:] = fp8(clamp(pe[m, h] / scale))        for d < DUP

    - A: (QH, M, K) q_nope, B: (QH, K, N) W_UK^T, pe: (M, QH, D_PE) roped q_pe.
    - out: (M, QH * DUP, N + D_PE), each head stored DUP times in a row, which is
      what repeat_interleave(DUP, dim=1) of the concatenated query produces.

    Grid: the first QH * cdiv(M, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N) programs
    run the BMM tiles. The last QH * cdiv(M, BLOCK_SIZE_M) programs each write
    q_pe for BLOCK_SIZE_M rows of one head.
    """
    tl.assume(stride_ah > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bh > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_rm > 0)
    tl.assume(stride_rh > 0)
    tl.assume(stride_rd > 0)
    tl.assume(stride_om > 0)
    tl.assume(stride_oh > 0)
    tl.assume(stride_od > 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    grid_mn = num_pid_m * num_pid_n
    bmm_programs = QH * grid_mn

    inv_scale = 1.0 / tl.load(scale_ptr).to(tl.float32)

    if pid < bmm_programs:
        pid_h = pid // grid_mn
        pid_m, pid_n = pid_grid(pid % grid_mn, num_pid_m, num_pid_n, GROUP_SIZE_M)

        tl.assume(pid_h >= 0)
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        pid_h = tl.cast(pid_h, tl.int64)

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_am = tl.cast(offs_m % M, tl.int64)
        offs_bn = tl.cast(offs_n % N, tl.int64)
        a_ptrs = a_ptr + (
            pid_h * stride_ah
            + offs_am[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            pid_h * stride_bh
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
            if EVEN_K:
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
                a = tl.load(a_ptrs)
            else:
                k_left = K - k * BLOCK_SIZE_K
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < k_left,
                    other=0.0,
                    cache_modifier=cache_modifier,
                )
                a = tl.load(a_ptrs, mask=offs_k[None, :] < k_left, other=0.0)
            acc = tl.dot(a, b, acc=acc)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        # Round to the activation dtype before quantizing, as the unfused
        # bf16 bmm -> cat -> quant path does.
        y = acc.to(a_ptr.dtype.element_ty).to(tl.float32)
        _quant_pad_store(
            y,
            out_ptr,
            tl.cast(offs_m, tl.int64),
            tl.cast(offs_n, tl.int64),
            (offs_m[:, None] < M) & (offs_n[None, :] < N),
            pid_h,
            inv_scale,
            stride_om,
            stride_oh,
            stride_od,
            FP8_MAX,
            DUP,
        )
    else:
        pid_pe = pid - bmm_programs
        pid_h = tl.cast(pid_pe % QH, tl.int64)
        offs_m = (pid_pe // QH) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_om = tl.cast(offs_m, tl.int64)
        _quant_store_pe(
            pe_ptr,
            out_ptr,
            offs_om,
            offs_om * 0 + pid_h,
            offs_m < M,
            inv_scale,
            N,
            stride_rm,
            stride_rh,
            stride_rd,
            stride_om,
            stride_oh,
            stride_od,
            FP8_MAX,
            D_PE,
            BLOCK_D_PE,
            DUP,
        )
