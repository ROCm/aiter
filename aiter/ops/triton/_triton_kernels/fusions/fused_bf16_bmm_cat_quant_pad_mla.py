# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import pid_grid
from aiter.ops.triton._triton_kernels.rope.rope import (
    _get_gptj_rotated_x,
    _get_neox_rotated_x,
)
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
        "ROPE_AND_CACHE",
        "IS_NEOX",
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
def _rope_vllm(
    x,
    pos,
    cos_sin_ptr,
    stride_cs_pos,
    stride_cs_d,
    IS_NEOX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """RoPE of the rows of x (BLOCK_M, BLOCK_D) at positions pos, bit-exact to
    vLLM's concat_and_cache_mla_rope_fused HIP kernel. For a pair (a, b) it
    computes a' = a*cos - rnd(b*sin) and b' = a*sin + rnd(b*cos): hipcc fuses
    the products of a into FMAs, so only the products of b round to x.dtype."""
    BLOCK_D_HALF: tl.constexpr = BLOCK_D // 2
    offs_d = tl.arange(0, BLOCK_D)
    if IS_NEOX:
        is_a = offs_d < BLOCK_D_HALF
        offs_cs = offs_d % BLOCK_D_HALF
        x_rot = _get_neox_rotated_x(x, is_a[None, :], BLOCK_M, BLOCK_D, BLOCK_D_HALF)
    else:
        is_a = offs_d % 2 == 0
        offs_cs = offs_d // 2
        x_rot = _get_gptj_rotated_x(x, is_a[None, :], BLOCK_M, BLOCK_D, BLOCK_D_HALF)
    cs_ptrs = (
        cos_sin_ptr + pos[:, None] * stride_cs_pos + offs_cs[None, :] * stride_cs_d
    )
    cos = tl.load(cs_ptrs).to(tl.float32)
    sin = tl.load(cs_ptrs + BLOCK_D_HALF * stride_cs_d).to(tl.float32)
    # bf16 * bf16 is exact in fp32, so these match the FMAs bit for bit.
    xf = x.to(tl.float32)
    rf = x_rot.to(tl.float32)
    exact = tl.where(is_a[None, :], xf * cos, rf * sin)
    rounded = tl.where(is_a[None, :], rf * sin, xf * cos).to(x.dtype)
    return (exact + rounded.to(tl.float32)).to(x.dtype)


@triton.jit
def _cache_fp8(x, k_scale, CACHE_MAX: tl.constexpr, cache_dtype: tl.constexpr):
    """vLLM's fp8 KV-cache conversion: correctly rounded x / k_scale, then a
    saturating cast."""
    x = x.to(tl.float32)
    x = tl.math.div_rn(x, tl.zeros_like(x) + k_scale)
    return tl.clamp(x, -CACHE_MAX, CACHE_MAX).to(cache_dtype)


@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0})
@triton.jit(repr=_fused_bf16_bmm_cat_quant_pad_mla_repr)
def _fused_bf16_bmm_cat_quant_pad_mla_kernel(
    a_ptr,
    b_ptr,
    pe_ptr,
    scale_ptr,
    out_ptr,
    pos_ptr,
    cos_sin_ptr,
    slot_ptr,
    kv_c_ptr,
    k_pe_ptr,
    kv_cache_ptr,
    k_scale_ptr,
    M,
    N,
    K,
    QH,
    cache_block_size,
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
    stride_cs_pos,
    stride_cs_d,
    stride_kvc_m,
    stride_kvc_d,
    stride_kpe_m,
    stride_kpe_d,
    stride_cache_b,
    stride_cache_e,
    stride_cache_d,
    FP8_MAX: tl.constexpr,
    CACHE_MAX: tl.constexpr,
    D_PE: tl.constexpr,
    BLOCK_D_PE: tl.constexpr,
    BLOCK_N_KV: tl.constexpr,
    DUP: tl.constexpr,
    ROPE_AND_CACHE: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call
    fused_bf16_bmm_cat_quant_pad_mla or fused_bf16_bmm_rope_cat_quant_pad_cache_mla.

    out[m, h * DUP + d, :N] = fp8(clamp(bf16(A[h] @ B[h])[m] / scale))
    out[m, h * DUP + d, N:] = fp8(clamp(pe[m, h] / scale))        for d < DUP

    - A: (QH, M, K) q_nope, B: (QH, K, N) W_UK^T, pe: (M, QH, D_PE) q_pe.
    - out: (M, QH * DUP, N + D_PE), each head stored DUP times in a row, which is
      what repeat_interleave(DUP, dim=1) of the concatenated query produces.

    With ROPE_AND_CACHE, pe is not roped yet. The kernel ropes q_pe and k_pe and
    writes fp8([kv_c[m] | k_pe[m]] / k_scale) to the cache slot of token m, as
    vLLM's concat_and_cache_mla_rope_fused does. Tokens with slot < 0 keep their
    q_pe unroped and do not write the cache.

    Grid: the first QH * cdiv(M, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N) programs
    run the BMM tiles. The next QH * cdiv(M, BLOCK_SIZE_M) programs each write
    q_pe for BLOCK_SIZE_M rows of one head. With ROPE_AND_CACHE, the last M
    programs each write the cache entry of one token.
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
    elif pid < bmm_programs + QH * num_pid_m:
        pid_pe = pid - bmm_programs
        pid_h = tl.cast(pid_pe % QH, tl.int64)
        offs_m = (pid_pe // QH) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_om = tl.cast(offs_m, tl.int64)
        offs_d = tl.arange(0, BLOCK_D_PE)
        row_mask = offs_m < M
        mask = row_mask[:, None] & (offs_d[None, :] < D_PE)
        pe = tl.load(
            pe_ptr
            + offs_om[:, None] * stride_rm
            + pid_h * stride_rh
            + offs_d[None, :] * stride_rd,
            mask=mask,
            other=0.0,
        )
        if ROPE_AND_CACHE:
            pos = tl.load(pos_ptr + offs_om, mask=row_mask, other=0)
            slot = tl.load(slot_ptr + offs_om, mask=row_mask, other=-1)
            roped = _rope_vllm(
                pe,
                pos,
                cos_sin_ptr,
                stride_cs_pos,
                stride_cs_d,
                IS_NEOX,
                BLOCK_SIZE_M,
                BLOCK_D_PE,
            )
            pe = tl.where((slot >= 0)[:, None], roped, pe)
        _quant_pad_store(
            pe.to(tl.float32),
            out_ptr,
            offs_om,
            N + offs_d,
            mask,
            pid_h,
            inv_scale,
            stride_om,
            stride_oh,
            stride_od,
            FP8_MAX,
            DUP,
        )
    elif ROPE_AND_CACHE:
        tl.assume(stride_kvc_m > 0)
        tl.assume(stride_kpe_m > 0)
        tl.assume(stride_cache_b > 0)
        tl.assume(stride_cache_e > 0)
        tok = tl.cast(pid - bmm_programs - QH * num_pid_m, tl.int64)
        # Issue every load before the slot check so their latencies overlap.
        tok_slot = tl.load(slot_ptr + tok).to(tl.int64)
        k_scale = tl.load(k_scale_ptr).to(tl.float32)
        offs_c = tl.arange(0, BLOCK_N_KV)
        offs_d = tl.arange(0, BLOCK_D_PE)
        kv_c = tl.load(
            kv_c_ptr + tok * stride_kvc_m + offs_c * stride_kvc_d,
            mask=offs_c < N,
            other=0.0,
        )
        k_pe = tl.load(k_pe_ptr + tok * stride_kpe_m + offs_d * stride_kpe_d)
        k_pe = _rope_vllm(
            k_pe[None, :],
            tl.load(pos_ptr + tok + tl.arange(0, 1)),
            cos_sin_ptr,
            stride_cs_pos,
            stride_cs_d,
            IS_NEOX,
            1,
            BLOCK_D_PE,
        )
        valid = tok_slot >= 0
        cache_dtype = kv_cache_ptr.dtype.element_ty
        entry_ptr = (
            kv_cache_ptr
            + (tok_slot // cache_block_size) * stride_cache_b
            + (tok_slot % cache_block_size) * stride_cache_e
        )
        tl.store(
            entry_ptr + offs_c * stride_cache_d,
            _cache_fp8(kv_c, k_scale, CACHE_MAX, cache_dtype),
            mask=(offs_c < N) & valid,
        )
        tl.store(
            entry_ptr + (N + offs_d[None, :]) * stride_cache_d,
            _cache_fp8(k_pe, k_scale, CACHE_MAX, cache_dtype),
            mask=(offs_d[None, :] < BLOCK_D_PE) & valid,
        )
