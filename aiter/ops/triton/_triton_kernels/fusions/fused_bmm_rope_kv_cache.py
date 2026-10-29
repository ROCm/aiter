# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import pid_grid
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from aiter.ops.triton.rope.rope import _get_gptj_rotated_x_1D, _get_neox_rotated_x_1D
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

_fused_fp4_bmm_rope_cat_and_cache_mla_repr = make_kernel_repr(
    "_fused_fp4_bmm_rope_cat_and_cache_mla_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
        "QH_PER_KH",
        "REUSE_FREQS_FRONT_PART",
        "IS_NEOX",
        "BLOCK_D_nope",
        "BLOCK_DK_nope",
        "BLOCK_D_pe",
        "BLOCK_D_HALF_pe",
        "PRE_QUANT",
        "TRANSPOSE_BM",
        "OUTPUT_Q_NOPE_ZEROS",
        "HAVE_Y_SCALE",
        "HAVE_K_SCALE",
        "EVEN_K",
    ],
)


_fused_fp4_bmm_reduce_repr = make_kernel_repr(
    "_fused_fp4_bmm_reduce_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "ACTUAL_KSPLIT",
        "MAX_KSPLIT",
    ],
)


_fused_fp8_bmm_rope_cat_and_cache_mla_repr = make_kernel_repr(
    "_fused_fp8_bmm_rope_cat_and_cache_mla_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "QH_PER_KH",
        "REUSE_FREQS_FRONT_PART",
        "IS_NEOX",
        "BLOCK_D_nope",
        "BLOCK_DK_nope",
        "BLOCK_D_pe",
        "BLOCK_D_HALF_pe",
        "OUTPUT_Q_NOPE_ZEROS",
        "HAVE_K_SCALE",
        "EVEN_K",
        "cache_modifier",
        "ROPE_BLOCK_T",
        "BLOCK_QH",
        "N_INNER",
        "PERM_N",
        "IL_R",
        "IL_U",
        "K_TAIL",
        "XCD_COLOC",
        "B_PF",
        "ILB_NUM",
        "ILB_DEN",
        "M_PER_CTA",
        "AQH",
        "LPIN",
        "GRID_MN",
        "LDB",
        "DTYPE_MAX",
    ],
)


@triton.jit
def _unit_rope(
    x_ptrs,
    cos,
    sin,
    d_pe_offs,
    IS_NEOX: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
):
    """Apply RoPE to a vector. Copied from fused_kv_cache.py"""
    x_pe = tl.load(x_ptrs)

    if IS_NEOX:
        x_rotated_mask = d_pe_offs < BLOCK_D_HALF_pe
        x_pe_rotated = _get_neox_rotated_x_1D(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )
    else:
        x_rotated_mask = d_pe_offs % 2 == 0
        x_pe_rotated = _get_gptj_rotated_x_1D(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )

    x_pe = x_pe * cos + x_pe_rotated * sin
    return x_pe


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0),
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_fused_fp4_bmm_rope_cat_and_cache_mla_repr)
def _fused_fp4_bmm_rope_cat_and_cache_mla_kernel(
    a_ptr,
    b_ptr,
    b_scales_ptr,
    c_ptr,
    c_scale_ptr,
    q_pe_ptr,
    k_nope_ptr,
    k_pe_ptr,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    decode_q_pe_out_ptr,
    k_pe_out_ptr,
    q_nope_zeros_out_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    M,
    N,
    K,
    B,
    B_slot,
    num_decode_toks_for_zeros,
    QH,
    KH,
    bmm_programs,
    grid_mn,
    num_pid_m,
    num_pid_n,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bn,
    stride_bk,
    stride_bsb,
    stride_bsn,
    stride_bsk,
    stride_cb,
    stride_ck,
    stride_cm,
    stride_cn,
    q_pe_stride_b,
    q_pe_stride_h,
    q_pe_stride_d,
    k_nope_stride_b,
    k_nope_stride_h,
    k_nope_stride_d,
    k_pe_stride_b,
    k_pe_stride_h,
    k_pe_stride_d,
    pos_stride_b,
    cos_stride_b,
    cos_stride_d,
    q_out_stride_b,
    q_out_stride_h,
    q_out_stride_d,
    decode_q_pe_out_stride_b,
    decode_q_pe_out_stride_h,
    decode_q_pe_out_stride_d,
    k_pe_out_stride_b,
    k_pe_out_stride_h,
    k_pe_out_stride_d,
    q_nope_zeros_out_stride_b,
    q_nope_zeros_out_stride_h,
    q_nope_zeros_out_stride_d,
    kv_cache_stride_b,
    kv_cache_stride_h,
    kv_cache_stride_d,
    k_scale_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    QH_PER_KH: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_DK_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    PRE_QUANT: tl.constexpr,
    OUTPUT_Q_NOPE_ZEROS: tl.constexpr,
    HAVE_Y_SCALE: tl.constexpr,
    HAVE_K_SCALE: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr = ".ca",
):
    """
    Fused kernel for FP4 BMM + RoPE + KV cache write.

    These are INDEPENDENT operations fused to reduce kernel launch overhead:
    - BMM writes to c_ptr (either q_out[:, :, :kv_lora_rank] or y_pp for split-K)
    - RoPE writes to q_out[:, :, kv_lora_rank:] and handles KV cache

    Grid structure:
    - Phase 1: pid < bmm_programs → BMM (tiled over heads, K-splits, M, N)
    - Phase 2: pid in [bmm_programs, bmm_programs + B*QH) → RoPE + KV cache for decode
    - Phase 3: pid >= bmm_programs + B*QH → KV cache only for prefill tokens
    """

    pid = tl.program_id(0)
    tl.assume(pid >= 0)

    rope_start = bmm_programs
    prefill_start = bmm_programs + B * QH

    if pid < bmm_programs:
        tl.assume(stride_ab > 0)
        tl.assume(stride_am > 0)
        tl.assume(stride_ak > 0)
        tl.assume(stride_bb > 0)
        tl.assume(stride_bk > 0)
        tl.assume(stride_bn > 0)
        tl.assume(stride_cb > 0)
        tl.assume(stride_cm > 0)
        tl.assume(stride_cn > 0)
        tl.assume(stride_bsb > 0)
        tl.assume(stride_bsk > 0)
        tl.assume(stride_bsn > 0)

        stride_ab_i64 = tl.cast(stride_ab, tl.int64)
        stride_bb_i64 = tl.cast(stride_bb, tl.int64)
        tl.cast(stride_cb, tl.int64)
        stride_bsb_i64 = tl.cast(stride_bsb, tl.int64)

        SCALE_GROUP_SIZE: tl.constexpr = 32

        if HAVE_Y_SCALE:
            c_scale = tl.load(c_scale_ptr)
        else:
            c_scale = 1
        c_scale_rcprl = (1 / c_scale).to(tl.float32)

        pid_head = pid // (NUM_KSPLIT * GRID_MN)
        pid_unified = pid % (NUM_KSPLIT * GRID_MN)
        pid_k = pid_unified % NUM_KSPLIT
        pid_tile = pid_unified // NUM_KSPLIT

        pid_head_i64 = tl.cast(pid_head, tl.int64)

        if NUM_KSPLIT == 1:
            # remap_xcd(pid_tile, GRID_MN)
            pid_m, pid_n = pid_grid(
                pid_tile, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M
            )
        else:
            pid_m = pid_tile // num_pid_n
            pid_n = pid_tile % num_pid_n

        tl.assume(pid_head >= 0)
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:
            num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)

            offs_k_bf16 = tl.arange(0, BLOCK_SIZE_K)
            offs_k_split_bf16 = pid_k * SPLITK_BLOCK_SIZE + offs_k_bf16
            offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            a_ptrs = a_ptr + (
                pid_head_i64 * stride_ab_i64
                + offs_am[:, None] * stride_am
                + offs_k_split_bf16[None, :] * stride_ak
            )

            offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
            offs_k_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_k
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            b_ptrs = b_ptr + (
                pid_head_i64 * stride_bb_i64
                + offs_k_split[:, None] * stride_bk
                + offs_bn[None, :] * stride_bn
            )

            offs_ks = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)) + tl.arange(
                0, BLOCK_SIZE_K // SCALE_GROUP_SIZE
            )
            b_scale_ptrs = (
                b_scales_ptr
                + pid_head_i64 * stride_bsb_i64
                + offs_bn[:, None] * stride_bsn
                + offs_ks[None, :] * stride_bsk
            )

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for k_idx in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
                b_scales = tl.load(b_scale_ptrs)

                if EVEN_K:
                    a_bf16 = tl.load(a_ptrs)
                    b = tl.load(b_ptrs, cache_modifier=cache_modifier)
                else:
                    a_bf16 = tl.load(
                        a_ptrs,
                        mask=offs_k_bf16[None, :] < K - k_idx * BLOCK_SIZE_K,
                        other=0,
                    )
                    b = tl.load(
                        b_ptrs,
                        mask=offs_k[:, None] < K - k_idx * (BLOCK_SIZE_K // 2),
                        other=0,
                    )

                if PRE_QUANT:
                    a, a_scales = _mxfp4_quant_op(
                        a_bf16, BLOCK_SIZE_K, BLOCK_SIZE_M, SCALE_GROUP_SIZE
                    )

                accumulator = tl.dot_scaled(
                    a, a_scales, "e2m1", b, b_scales, "e2m1", acc=accumulator
                )

                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
                b_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bsk

            if HAVE_Y_SCALE:
                accumulator = accumulator * c_scale_rcprl

            c = accumulator.to(c_ptr.type.element_ty)

            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)

            if NUM_KSPLIT == 1:
                c_ptrs = (
                    c_ptr
                    + pid_head_i64 * stride_cb
                    + offs_cm[:, None] * stride_cm
                    + offs_cn[None, :] * stride_cn
                )
            else:
                c_ptrs = (
                    c_ptr
                    + pid_head_i64 * stride_cb
                    + pid_k.to(tl.int64) * stride_ck
                    + offs_cm[:, None] * stride_cm
                    + offs_cn[None, :] * stride_cn
                )

            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)

    elif pid < prefill_start:
        pid_adjusted = pid - rope_start
        pid_b = pid_adjusted // QH
        pid_hq = pid_adjusted % QH

        tl.assume(pid_b >= 0)
        tl.assume(pid_hq >= 0)

        dk_nope_offs = tl.arange(0, BLOCK_DK_nope).to(tl.int64)
        d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

        if REUSE_FREQS_FRONT_PART:
            if IS_NEOX:
                d_cos_offs = d_pe_offs
                d_cos_offs = tl.where(
                    (d_cos_offs >= BLOCK_D_HALF_pe) & (d_cos_offs < BLOCK_D_pe),
                    d_cos_offs - BLOCK_D_HALF_pe,
                    d_cos_offs,
                ).to(d_cos_offs.dtype)
            else:
                d_cos_offs = d_pe_offs // 2
        else:
            d_cos_offs = d_pe_offs

        pos = tl.load(pos_ptr + pid_b * pos_stride_b)
        cos_offs = pos * cos_stride_b + d_cos_offs * cos_stride_d
        cos = tl.load(cos_ptr + cos_offs)
        sin = tl.load(sin_ptr + cos_offs)

        q_pe_ptrs = (
            q_pe_ptr
            + pid_b * q_pe_stride_b
            + pid_hq * q_pe_stride_h
            + d_pe_offs * q_pe_stride_d
        )
        q_pe = _unit_rope(
            q_pe_ptrs,
            cos,
            sin,
            d_pe_offs,
            IS_NEOX,
            BLOCK_D_pe,
            BLOCK_D_HALF_pe,
        )

        q_out_base = q_out_ptr + pid_b * q_out_stride_b + pid_hq * q_out_stride_h
        tl.store(
            q_out_base + (d_pe_offs + BLOCK_D_nope) * q_out_stride_d,
            q_pe.to(q_out_ptr.dtype.element_ty),
        )

        if pid_adjusted < num_decode_toks_for_zeros * QH:
            decode_q_pe_out_ptrs = (
                decode_q_pe_out_ptr
                + pid_b * decode_q_pe_out_stride_b
                + pid_hq * decode_q_pe_out_stride_h
                + d_pe_offs * decode_q_pe_out_stride_d
            )
            tl.store(
                decode_q_pe_out_ptrs, q_pe.to(decode_q_pe_out_ptr.dtype.element_ty)
            )

        if OUTPUT_Q_NOPE_ZEROS and pid_adjusted < num_decode_toks_for_zeros * QH:
            z = tl.zeros((BLOCK_DK_nope,), dtype=q_nope_zeros_out_ptr.dtype.element_ty)
            tl.store(
                q_nope_zeros_out_ptr
                + pid_b * q_nope_zeros_out_stride_b
                + pid_hq * q_nope_zeros_out_stride_h
                + dk_nope_offs * q_nope_zeros_out_stride_d,
                z,
            )

        if pid_hq % QH_PER_KH == 0:
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)
            if pid_slot >= 0:
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1

                pid_hk = pid_hq // QH_PER_KH

                k_nope_ptrs = (
                    k_nope_ptr
                    + pid_b * k_nope_stride_b
                    + pid_hk * k_nope_stride_h
                    + dk_nope_offs * k_nope_stride_d
                )
                k_nope = tl.load(k_nope_ptrs)

                k_pe_load_ptrs = (
                    k_pe_ptr
                    + pid_b * k_pe_stride_b
                    + pid_hk * k_pe_stride_h
                    + d_pe_offs * k_pe_stride_d
                )
                k_pe = _unit_rope(
                    k_pe_load_ptrs,
                    cos,
                    sin,
                    d_pe_offs,
                    IS_NEOX,
                    BLOCK_D_pe,
                    BLOCK_D_HALF_pe,
                )

                k_pe_out_ptrs = (
                    k_pe_out_ptr
                    + pid_b * k_pe_out_stride_b
                    + pid_hk * k_pe_out_stride_h
                    + d_pe_offs * k_pe_out_stride_d
                )
                tl.store(k_pe_out_ptrs, k_pe.to(k_pe_out_ptr.dtype.element_ty))

                k_scale_rcprl = (1 / k_scale).to(tl.float32)
                k_nope_scaled = (k_nope.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )
                k_pe_scaled = (k_pe.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )

                kv_cache_ptrs = (
                    kv_cache_ptr
                    + pid_slot * kv_cache_stride_b
                    + pid_hk * kv_cache_stride_h
                )

                tl.store(
                    kv_cache_ptrs + dk_nope_offs * kv_cache_stride_d, k_nope_scaled
                )
                tl.store(
                    kv_cache_ptrs + (d_pe_offs + BLOCK_DK_nope) * kv_cache_stride_d,
                    k_pe_scaled,
                )

    else:
        pid_adjusted = pid - prefill_start
        pid_b = pid_adjusted // KH + B
        pid_hk = pid_adjusted % KH

        if pid_b < B_slot:
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)

            if pid_slot >= 0:
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1

                dk_nope_offs = tl.arange(0, BLOCK_DK_nope).to(tl.int64)
                d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

                k_nope_ptrs = (
                    k_nope_ptr
                    + pid_b * k_nope_stride_b
                    + pid_hk * k_nope_stride_h
                    + dk_nope_offs * k_nope_stride_d
                )
                k_nope = tl.load(k_nope_ptrs)

                k_pe_load_ptrs = (
                    k_pe_ptr
                    + pid_b * k_pe_stride_b
                    + pid_hk * k_pe_stride_h
                    + d_pe_offs * k_pe_stride_d
                )
                k_pe = tl.load(k_pe_load_ptrs)

                k_pe_out_ptrs = (
                    k_pe_out_ptr
                    + pid_b * k_pe_out_stride_b
                    + pid_hk * k_pe_out_stride_h
                    + d_pe_offs * k_pe_out_stride_d
                )
                tl.store(k_pe_out_ptrs, k_pe.to(k_pe_out_ptr.dtype.element_ty))

                k_scale_rcprl = (1 / k_scale).to(tl.float32)
                k_nope_scaled = (k_nope.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )
                k_pe_scaled = (k_pe.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )

                kv_cache_ptrs = (
                    kv_cache_ptr
                    + pid_slot * kv_cache_stride_b
                    + pid_hk * kv_cache_stride_h
                )

                tl.store(
                    kv_cache_ptrs + dk_nope_offs * kv_cache_stride_d, k_nope_scaled
                )
                tl.store(
                    kv_cache_ptrs + (d_pe_offs + BLOCK_DK_nope) * kv_cache_stride_d,
                    k_pe_scaled,
                )


@triton.jit(repr=_fused_fp4_bmm_reduce_repr)
def _fused_fp4_bmm_reduce_kernel(
    c_in_ptr,
    c_out_ptr,
    M,
    N,
    QH,
    stride_c_in_b,
    stride_c_in_k,
    stride_c_in_m,
    stride_c_in_n,
    stride_c_out_b,
    stride_c_out_h,
    stride_c_out_d,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
    TRANSPOSE_BM: tl.constexpr,
):
    """Reduce kernel for split-K BMM results."""
    pid_head = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, MAX_KSPLIT)

    c_in_ptrs = (
        c_in_ptr
        + pid_head * stride_c_in_b
        + (offs_k[:, None, None].to(tl.int64) * stride_c_in_k)
        + (offs_m[None, :, None] * stride_c_in_m)
        + (offs_n[None, None, :] * stride_c_in_n)
    )

    if ACTUAL_KSPLIT == MAX_KSPLIT:
        c = tl.load(c_in_ptrs)
    else:
        c = tl.load(c_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT)

    c = tl.sum(c, axis=0)
    c = c.to(c_out_ptr.type.element_ty)

    if TRANSPOSE_BM:
        c_out_ptrs = (
            c_out_ptr
            + (offs_m[:, None] * stride_c_out_b)
            + (pid_head * stride_c_out_h)
            + (offs_n[None, :] * stride_c_out_d)
        )
    else:
        c_out_ptrs = (
            c_out_ptr
            + (pid_head * stride_c_out_b)
            + (offs_m[:, None] * stride_c_out_h)
            + (offs_n[None, :] * stride_c_out_d)
        )

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_out_ptrs, c, mask=c_mask)


@triton.jit
def _fp8_load_pin(x):
    # Keep group quantization after the preceding loads.

    return tl.inline_asm_elementwise(
        "; fp8 load pin", "=v,0", [x], dtype=x.dtype, is_pure=False, pack=2
    )


@triton.jit
def _fp8_quant_group(x, DTYPE_MAX: tl.constexpr, b_ptr):
    one_over_DTYPE_MAX = 1.0 / DTYPE_MAX
    m = tl.maximum(tl.max(tl.abs(x), axis=-1), 1e-10)[:, None]
    s = m.to(tl.float32) * one_over_DTYPE_MAX
    r = 1.0 / s
    xq = tl.clamp(x * r, -DTYPE_MAX, DTYPE_MAX).to(b_ptr.dtype.element_ty)
    return xq, s


@triton.jit
def _fp8_mma_store(
    a0q,
    s0,
    a1q,
    s1,
    b0,
    b1,
    b_scale,
    c_ptrs,
    c_mask,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    PERM_N: tl.constexpr,
):
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc = tl.fma(tl.dot(a0q, b0), s0, acc)
    acc = tl.fma(tl.dot(a1q, b1), s1, acc)
    acc = acc * b_scale
    c_val = acc.to(c_ptrs.dtype.element_ty)
    if PERM_N:
        c_val = tl.reshape(
            tl.permute(
                tl.reshape(c_val, (BLOCK_SIZE_M, BLOCK_SIZE_N // 64, 4, 4, 4)),
                (0, 1, 3, 2, 4),
            ),
            (BLOCK_SIZE_M, BLOCK_SIZE_N),
        )
    tl.store(c_ptrs, c_val, mask=c_mask)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit(
    repr=_fused_fp8_bmm_rope_cat_and_cache_mla_repr,
    do_not_specialize=["gmn_magic", "gmn_shift"],
)
def _fused_fp8_bmm_rope_cat_and_cache_mla_kernel(
    a_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    q_pe_ptr,
    k_nope_ptr,
    k_pe_ptr,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    decode_q_pe_out_ptr,
    k_pe_out_ptr,
    q_nope_zeros_out_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    M,
    N,
    K,
    B,
    B_slot,
    num_decode_toks_for_zeros,
    QH,
    KH,
    bmm_programs,
    grid_mn,
    num_pid_m,
    num_pid_n,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    q_pe_stride_b,
    q_pe_stride_h,
    q_pe_stride_d,
    k_nope_stride_b,
    k_nope_stride_h,
    k_nope_stride_d,
    k_pe_stride_b,
    k_pe_stride_h,
    k_pe_stride_d,
    pos_stride_b,
    cos_stride_b,
    cos_stride_d,
    q_out_stride_b,
    q_out_stride_h,
    q_out_stride_d,
    decode_q_pe_out_stride_b,
    decode_q_pe_out_stride_h,
    decode_q_pe_out_stride_d,
    k_pe_out_stride_b,
    k_pe_out_stride_h,
    k_pe_out_stride_d,
    q_nope_zeros_out_stride_b,
    q_nope_zeros_out_stride_h,
    q_nope_zeros_out_stride_d,
    kv_cache_stride_b,
    kv_cache_stride_h,
    kv_cache_stride_d,
    k_scale_ptr,
    gmn_magic,
    gmn_shift,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    QH_PER_KH: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_DK_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    OUTPUT_Q_NOPE_ZEROS: tl.constexpr,
    HAVE_K_SCALE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    EVEN_K: tl.constexpr,
    cache_modifier: tl.constexpr = ".ca",
    ROPE_BLOCK_T: tl.constexpr = 0,
    BLOCK_QH: tl.constexpr = 0,
    N_INNER: tl.constexpr = 1,
    PERM_N: tl.constexpr = 0,
    IL_R: tl.constexpr = 0,
    IL_U: tl.constexpr = 1,
    K_TAIL: tl.constexpr = 0,
    XCD_COLOC: tl.constexpr = 0,
    B_PF: tl.constexpr = 0,
    ILB_NUM: tl.constexpr = 0,
    ILB_DEN: tl.constexpr = 1,
    M_PER_CTA: tl.constexpr = 1,
    AQH: tl.constexpr = 0,
    LPIN: tl.constexpr = 0,
    GRID_MN: tl.constexpr = 0,
    LDB: tl.constexpr = 0,
):
    """Compute the projection, rotary outputs and KV writes in one launch."""

    pid = tl.program_id(0)
    tl.assume(pid >= 0)

    rope_start = bmm_programs
    if ROPE_BLOCK_T > 0:
        prefill_start = bmm_programs + tl.cdiv(B, ROPE_BLOCK_T)
    else:
        prefill_start = bmm_programs + B * QH

    # Interleave projection and rotary work while preserving each group of IDs.
    if ILB_NUM > 0:
        nr = prefill_start - rope_start
        NBu = bmm_programs // IL_U
        NRi = tl.minimum(nr // IL_U, (NBu * ILB_NUM) // ILB_DEN)
        T = NBu + NRi
        if pid < T * IL_U:
            u = pid // IL_U
            gl = pid % IL_U
            a = (u * NRi) // T
            b = ((u + 1) * NRi) // T
            pid = tl.where(b > a, rope_start + a * IL_U + gl, (u - a) * IL_U + gl)
        elif pid < prefill_start:
            r = pid - T * IL_U
            nb_left = bmm_programs - NBu * IL_U
            pid = tl.where(
                r < nb_left, NBu * IL_U + r, rope_start + NRi * IL_U + (r - nb_left)
            )
    elif IL_R > 0:
        nr = prefill_start - rope_start
        ng = tl.minimum(bmm_programs // (IL_R * IL_U), nr // IL_U)
        GP: tl.constexpr = (IL_R + 1) * IL_U
        if pid < ng * GP:
            gi = pid // GP
            gr = pid % GP
            gj = gr // IL_U
            gl = gr % IL_U
            pid = tl.where(
                gj == IL_R, rope_start + gi * IL_U + gl, (gi * IL_R + gj) * IL_U + gl
            )
        elif pid < prefill_start:
            r = pid - ng * GP
            nb_left = bmm_programs - ng * IL_R * IL_U
            pid = tl.where(
                r < nb_left,
                ng * IL_R * IL_U + r,
                rope_start + ng * IL_U + (r - nb_left),
            )
    # Group the heads and rotary tiles that write the same query rows.
    if XCD_COLOC:
        if pid < bmm_programs:
            g8 = (grid_mn // IL_U) * IL_U
            if pid < QH * g8:
                xu = pid // IL_U
                pid = (xu % QH) * grid_mn + (xu // QH) * IL_U + pid % IL_U
            else:
                xv = pid - QH * g8
                xrem = grid_mn - g8
                pid = (xv // xrem) * grid_mn + g8 + xv % xrem
        elif pid < prefill_start:
            RPM: tl.constexpr = (BLOCK_SIZE_M * M_PER_CTA) // ROPE_BLOCK_T
            rho = pid - rope_start
            if rho < (nr // (IL_U * RPM)) * (IL_U * RPM):
                xs = rho // IL_U
                rho = ((xs // RPM) * IL_U + rho % IL_U) * RPM + xs % RPM
            pid = rope_start + rho

    if pid < bmm_programs:
        stride_ab_i64 = tl.cast(stride_ab, tl.int64)
        stride_am_i64 = tl.cast(stride_am, tl.int64)
        stride_ak_i64 = tl.cast(stride_ak, tl.int64)
        stride_bb_i64 = tl.cast(stride_bb, tl.int64)
        stride_bk_i64 = tl.cast(stride_bk, tl.int64)
        stride_bn_i64 = tl.cast(stride_bn, tl.int64)
        stride_cb_i64 = tl.cast(stride_cb, tl.int64)
        stride_cm_i64 = tl.cast(stride_cm, tl.int64)
        stride_cn_i64 = tl.cast(stride_cn, tl.int64)

        tl.assume(stride_ab_i64 > 0)
        tl.assume(stride_am_i64 > 0)
        tl.assume(stride_ak_i64 > 0)
        tl.assume(stride_bb_i64 > 0)
        tl.assume(stride_bk_i64 > 0)
        tl.assume(stride_bn_i64 > 0)
        tl.assume(stride_cb_i64 > 0)
        tl.assume(stride_cm_i64 > 0)
        tl.assume(stride_cn_i64 > 0)

        if GRID_MN > 0:

            pid_head = pid // GRID_MN
            pid_tile = pid % GRID_MN
        else:

            pid_head = (
                tl.umulhi(
                    pid.to(tl.uint32, bitcast=True) * 2,
                    gmn_magic.to(tl.uint32, bitcast=True),
                )
                >> gmn_shift
            ).to(tl.int32, bitcast=True)
            pid_tile = pid - pid_head * grid_mn

        pid_head_i64 = tl.cast(pid_head, tl.int64)

        if GROUP_SIZE_M == 1:
            pid_m = pid_tile // num_pid_n
            pid_n = pid_tile % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid_tile // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (pid_tile % group_size_m)
            pid_n = (pid_tile % num_pid_in_group) // group_size_m

        pid_m_i64 = tl.cast(pid_m, tl.int64)
        pid_n_i64 = tl.cast(pid_n, tl.int64)

        tl.assume(pid_m_i64 >= 0)
        tl.assume(pid_n_i64 >= 0)
        tl.assume(pid_head_i64 >= 0)

        # Reuse two quantized K groups across several output column chunks.
        if N_INNER > 1:
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            m_start = pid_m * (BLOCK_SIZE_M * M_PER_CTA)
            n_start = pid_n * (BLOCK_SIZE_N * N_INNER)
            offs_am_loc = tl.minimum(tl.arange(0, BLOCK_SIZE_M), M - 1 - m_start)
            if N_INNER > 1:

                offs_bn_loc = tl.arange(0, BLOCK_SIZE_N)
                if PERM_N:

                    ph = tl.arange(0, BLOCK_SIZE_N // 64)[:, None, None, None] * 64
                    pj = tl.arange(0, 4)[None, :, None, None] * 4
                    pg = tl.arange(0, 4)[None, None, :, None] * 16
                    pe = tl.arange(0, 4)[None, None, None, :]
                    offs_bn_loc = tl.reshape(ph + pj + pg + pe, (BLOCK_SIZE_N,))
            else:
                offs_bn_loc = tl.minimum(tl.arange(0, BLOCK_SIZE_N), N - 1 - n_start)

            a_ptrs = (
                a_ptr
                + pid_head_i64 * stride_ab_i64
                + tl.cast(m_start, tl.int64) * stride_am_i64
            ) + (offs_am_loc[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = (
                b_ptr
                + pid_head_i64 * stride_bb_i64
                + tl.cast(n_start, tl.int64) * stride_bn_i64
            ) + (offs_k[:, None] * stride_bk + offs_bn_loc[None, :] * stride_bn)

            # Use a larger row tile with a short accumulation chain.
            if LDB > 0 and N_INNER > 1 and K_TAIL > 0:

                ROWS: tl.constexpr = BLOCK_SIZE_M * M_PER_CTA
                offs_kt = tl.arange(0, K_TAIL)
                am = tl.minimum(tl.arange(0, ROWS), M - 1 - m_start)
                a_base = (
                    a_ptr
                    + pid_head_i64 * stride_ab_i64
                    + tl.cast(m_start, tl.int64) * stride_am_i64
                )
                x0 = tl.load(
                    a_base + (am[:, None] * stride_am + offs_k[None, :] * stride_ak),
                    cache_modifier=".cg",
                )
                x1 = tl.load(
                    a_base
                    + BLOCK_SIZE_K * stride_ak_i64
                    + (am[:, None] * stride_am + offs_kt[None, :] * stride_ak),
                    cache_modifier=".cg",
                )
                b_scale = tl.load(b_scale_ptr)
                q0, s0 = _fp8_quant_group(x0, DTYPE_MAX, b_ptr)
                q1, s1 = _fp8_quant_group(x1, DTYPE_MAX, b_ptr)
                b_head = (
                    b_ptr
                    + pid_head_i64 * stride_bb_i64
                    + tl.cast(n_start, tl.int64) * stride_bn_i64
                )
                b0_ptrs = b_head + (
                    offs_k[:, None] * stride_bk + offs_bn_loc[None, :] * stride_bn
                )
                b1_ptrs = (b_head + BLOCK_SIZE_K * stride_bk_i64) + (
                    offs_kt[:, None] * stride_bk + offs_bn_loc[None, :] * stride_bn
                )
                BSTEP = BLOCK_SIZE_N * stride_bn_i64
                offs_cm_loc = tl.arange(0, ROWS)
                offs_cn_loc = tl.arange(0, BLOCK_SIZE_N)
                c_ptrs = (
                    c_ptr
                    + pid_head_i64 * stride_cb_i64
                    + tl.cast(m_start, tl.int64) * stride_cm_i64
                    + tl.cast(n_start, tl.int64) * stride_cn_i64
                ) + (
                    offs_cm_loc[:, None] * stride_cm + offs_cn_loc[None, :] * stride_cn
                )
                cmask = offs_cm_loc[:, None] < M - m_start
                CNSTEP = BLOCK_SIZE_N * stride_cn_i64
                for n_i in range(N_INNER):
                    b0 = tl.load(b0_ptrs + n_i * BSTEP, cache_modifier=cache_modifier)
                    b1 = tl.load(b1_ptrs + n_i * BSTEP)
                    _fp8_mma_store(
                        q0,
                        s0,
                        q1,
                        s1,
                        b0,
                        b1,
                        b_scale,
                        c_ptrs + n_i * CNSTEP,
                        cmask,
                        ROWS,
                        BLOCK_SIZE_N,
                        PERM_N,
                    )
            elif N_INNER > 1 and B_PF > 0 and K_TAIL > 0:

                one_over_DTYPE_MAX = 1.0 / DTYPE_MAX
                offs_kt = tl.arange(0, K_TAIL)
                b1_ptrs = (
                    b_ptr
                    + pid_head_i64 * stride_bb_i64
                    + tl.cast(n_start, tl.int64) * stride_bn_i64
                    + BLOCK_SIZE_K * stride_bk_i64
                ) + (offs_kt[:, None] * stride_bk + offs_bn_loc[None, :] * stride_bn)
                BSTEP = BLOCK_SIZE_N * stride_bn_i64
                b0n = tl.load(b_ptrs, cache_modifier=cache_modifier)
                b1n = tl.load(b1_ptrs)
                if B_PF >= 3:

                    tl.static_assert(N_INNER <= 4)
                    if N_INNER > 1:
                        bq0_1 = tl.load(b_ptrs + BSTEP, cache_modifier=cache_modifier)
                        bq1_1 = tl.load(b1_ptrs + BSTEP)
                    if N_INNER > 2:
                        bq0_2 = tl.load(
                            b_ptrs + 2 * BSTEP, cache_modifier=cache_modifier
                        )
                        bq1_2 = tl.load(b1_ptrs + 2 * BSTEP)
                    if N_INNER > 3:
                        bq0_3 = tl.load(
                            b_ptrs + 3 * BSTEP, cache_modifier=cache_modifier
                        )
                        bq1_3 = tl.load(b1_ptrs + 3 * BSTEP)
                a0 = tl.load(a_ptrs, cache_modifier=".cg")
                a1 = tl.load(
                    a_ptr
                    + pid_head_i64 * stride_ab_i64
                    + tl.cast(m_start, tl.int64) * stride_am_i64
                    + BLOCK_SIZE_K * stride_ak_i64
                    + (offs_am_loc[:, None] * stride_am + offs_kt[None, :] * stride_ak),
                    cache_modifier=".cg",
                )
                b_scale = tl.load(b_scale_ptr)
                if LPIN > 0:

                    a0 = _fp8_load_pin(a0)
                    a1 = _fp8_load_pin(a1)
                if B_PF >= 2:

                    tl.debug_barrier()
                m0 = tl.maximum(tl.max(tl.abs(a0), axis=-1), 1e-10)[:, None]
                s0 = m0.to(tl.float32) * one_over_DTYPE_MAX
                r0 = 1.0 / s0
                a0q = tl.clamp(a0 * r0, -DTYPE_MAX, DTYPE_MAX).to(
                    b_ptr.dtype.element_ty
                )
                m1 = tl.maximum(tl.max(tl.abs(a1), axis=-1), 1e-10)[:, None]
                s1 = m1.to(tl.float32) * one_over_DTYPE_MAX
                r1 = 1.0 / s1
                a1q = tl.clamp(a1 * r1, -DTYPE_MAX, DTYPE_MAX).to(
                    b_ptr.dtype.element_ty
                )

                offs_cm_loc = tl.arange(0, BLOCK_SIZE_M)
                offs_cn_loc = tl.arange(0, BLOCK_SIZE_N)
                c_ptrs = (
                    c_ptr
                    + pid_head_i64 * stride_cb_i64
                    + tl.cast(m_start, tl.int64) * stride_cm_i64
                    + tl.cast(n_start, tl.int64) * stride_cn_i64
                ) + (
                    offs_cm_loc[:, None] * stride_cm + offs_cn_loc[None, :] * stride_cn
                )
                c_mask = offs_cm_loc[:, None] < M - m_start

                for n_i in tl.static_range(N_INNER):
                    if B_PF >= 3:
                        if n_i == 0:
                            b0 = b0n
                            b1 = b1n
                        elif n_i == 1:
                            b0 = bq0_1
                            b1 = bq1_1
                        elif n_i == 2:
                            b0 = bq0_2
                            b1 = bq1_2
                        else:
                            b0 = bq0_3
                            b1 = bq1_3
                    else:
                        b0 = b0n
                        b1 = b1n
                        if n_i + 1 < N_INNER:
                            b_ptrs += BSTEP
                            b1_ptrs += BSTEP
                            b0n = tl.load(b_ptrs, cache_modifier=cache_modifier)
                            b1n = tl.load(b1_ptrs)
                    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                    acc = tl.fma(tl.dot(a0q, b0), s0, acc)
                    acc = tl.fma(tl.dot(a1q, b1), s1, acc)
                    acc = acc * b_scale
                    c_val = acc.to(c_ptr.type.element_ty)
                    if PERM_N:
                        c_val = tl.reshape(
                            tl.permute(
                                tl.reshape(
                                    c_val, (BLOCK_SIZE_M, BLOCK_SIZE_N // 64, 4, 4, 4)
                                ),
                                (0, 1, 3, 2, 4),
                            ),
                            (BLOCK_SIZE_M, BLOCK_SIZE_N),
                        )
                    tl.store(c_ptrs, c_val, mask=c_mask)
                    c_ptrs += BLOCK_SIZE_N * stride_cn_i64
            elif N_INNER > 1:

                one_over_DTYPE_MAX = 1.0 / DTYPE_MAX
                b_scale = tl.load(b_scale_ptr)

                a0 = tl.load(a_ptrs, cache_modifier=".cg")
                if AQH > 0 and K_TAIL > 0:

                    a1 = tl.load(
                        a_ptr
                        + pid_head_i64 * stride_ab_i64
                        + tl.cast(m_start, tl.int64) * stride_am_i64
                        + BLOCK_SIZE_K * stride_ak_i64
                        + (
                            offs_am_loc[:, None] * stride_am
                            + tl.arange(0, K_TAIL)[None, :] * stride_ak
                        ),
                        cache_modifier=".cg",
                    )
                    a0 = _fp8_load_pin(a0)
                m0 = tl.maximum(tl.max(tl.abs(a0), axis=-1), 1e-10)[:, None]
                s0 = m0.to(tl.float32) * one_over_DTYPE_MAX
                r0 = 1.0 / s0
                a0q = tl.clamp(a0 * r0, -DTYPE_MAX, DTYPE_MAX).to(
                    b_ptr.dtype.element_ty
                )

                if K_TAIL > 0:

                    offs_kt = tl.arange(0, K_TAIL)
                    if AQH == 0:
                        a1 = tl.load(
                            a_ptr
                            + pid_head_i64 * stride_ab_i64
                            + tl.cast(m_start, tl.int64) * stride_am_i64
                            + BLOCK_SIZE_K * stride_ak_i64
                            + (
                                offs_am_loc[:, None] * stride_am
                                + offs_kt[None, :] * stride_ak
                            ),
                            cache_modifier=".cg",
                        )
                    b1_ptrs = (
                        b_ptr
                        + pid_head_i64 * stride_bb_i64
                        + tl.cast(n_start, tl.int64) * stride_bn_i64
                        + BLOCK_SIZE_K * stride_bk_i64
                    ) + (
                        offs_kt[:, None] * stride_bk + offs_bn_loc[None, :] * stride_bn
                    )
                else:
                    a1 = tl.load(
                        a_ptrs + BLOCK_SIZE_K * stride_ak_i64,
                        mask=offs_k[None, :] < K - BLOCK_SIZE_K,
                        other=0.0,
                    )
                m1 = tl.maximum(tl.max(tl.abs(a1), axis=-1), 1e-10)[:, None]
                s1 = m1.to(tl.float32) * one_over_DTYPE_MAX
                r1 = 1.0 / s1
                a1q = tl.clamp(a1 * r1, -DTYPE_MAX, DTYPE_MAX).to(
                    b_ptr.dtype.element_ty
                )

                offs_cm_loc = tl.arange(0, BLOCK_SIZE_M)
                offs_cn_loc = tl.arange(0, BLOCK_SIZE_N)
                c_ptrs = (
                    c_ptr
                    + pid_head_i64 * stride_cb_i64
                    + tl.cast(m_start, tl.int64) * stride_cm_i64
                    + tl.cast(n_start, tl.int64) * stride_cn_i64
                ) + (
                    offs_cm_loc[:, None] * stride_cm + offs_cn_loc[None, :] * stride_cn
                )
                c_mask = offs_cm_loc[:, None] < M - m_start

                for n_i in range(N_INNER):
                    b0 = tl.load(b_ptrs, cache_modifier=cache_modifier)
                    if K_TAIL > 0:
                        b1 = tl.load(b1_ptrs)
                        b1_ptrs += BLOCK_SIZE_N * stride_bn_i64
                    else:
                        b1 = tl.load(
                            b_ptrs + BLOCK_SIZE_K * stride_bk_i64,
                            mask=offs_k[:, None] < K - BLOCK_SIZE_K,
                            other=0.0,
                        )
                    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                    acc = tl.fma(tl.dot(a0q, b0), s0, acc)
                    acc = tl.fma(tl.dot(a1q, b1), s1, acc)
                    acc = acc * b_scale
                    c_val = acc.to(c_ptr.type.element_ty)
                    if PERM_N:

                        c_val = tl.reshape(
                            tl.permute(
                                tl.reshape(
                                    c_val, (BLOCK_SIZE_M, BLOCK_SIZE_N // 64, 4, 4, 4)
                                ),
                                (0, 1, 3, 2, 4),
                            ),
                            (BLOCK_SIZE_M, BLOCK_SIZE_N),
                        )
                    tl.store(c_ptrs, c_val, mask=c_mask)
                    b_ptrs += BLOCK_SIZE_N * stride_bn_i64
                    c_ptrs += BLOCK_SIZE_N * stride_cn_i64
        else:
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

            a_ptrs = a_ptr + (
                pid_head_i64 * stride_ab_i64
                + offs_am[:, None] * stride_am_i64
                + offs_k[None, :] * stride_ak_i64
            )
            b_ptrs = b_ptr + (
                pid_head_i64 * stride_bb_i64
                + offs_k[:, None] * stride_bk_i64
                + offs_bn[None, :] * stride_bn_i64
            )

            one_over_DTYPE_MAX = 1.0 / DTYPE_MAX
            b_scale = tl.load(b_scale_ptr)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for k_idx in range(tl.cdiv(K, BLOCK_SIZE_K)):
                if EVEN_K:
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs, cache_modifier=cache_modifier)
                else:
                    a = tl.load(
                        a_ptrs,
                        mask=offs_k[None, :] < K - k_idx * BLOCK_SIZE_K,
                        other=0.0,
                    )
                    b = tl.load(
                        b_ptrs,
                        mask=offs_k[:, None] < K - k_idx * BLOCK_SIZE_K,
                        other=0.0,
                    )

                m = tl.maximum(tl.max(tl.abs(a), axis=-1), 1e-10)[:, None]
                a_scale = m.to(tl.float32) * one_over_DTYPE_MAX
                a_scale_recip = 1.0 / a_scale
                a = tl.clamp(a * a_scale_recip, -DTYPE_MAX, DTYPE_MAX).to(
                    b_ptr.dtype.element_ty
                )

                accumulator += tl.dot(a, b) * a_scale

                a_ptrs += BLOCK_SIZE_K * stride_ak_i64
                b_ptrs += BLOCK_SIZE_K * stride_bk_i64

            accumulator *= b_scale

            c = accumulator.to(c_ptr.type.element_ty)

            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            c_ptrs = (
                c_ptr
                + pid_head_i64 * stride_cb_i64
                + offs_cm[:, None] * stride_cm_i64
                + offs_cn[None, :] * stride_cn_i64
            )

            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)

    # Process all query heads in a block of tokens.
    elif ROPE_BLOCK_T > 0 and pid < prefill_start:

        pid_adjusted = pid - rope_start
        offs_t = pid_adjusted * ROPE_BLOCK_T + tl.arange(0, ROPE_BLOCK_T)
        t_mask = offs_t < B
        offs_t64 = offs_t.to(tl.int64)
        offs_h = tl.arange(0, BLOCK_QH).to(tl.int64)
        d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)
        dk_nope_offs = tl.arange(0, BLOCK_DK_nope).to(tl.int64)

        if REUSE_FREQS_FRONT_PART:
            if IS_NEOX:
                d_cos_offs = tl.where(
                    (d_pe_offs >= BLOCK_D_HALF_pe) & (d_pe_offs < BLOCK_D_pe),
                    d_pe_offs - BLOCK_D_HALF_pe,
                    d_pe_offs,
                ).to(d_pe_offs.dtype)
            else:
                d_cos_offs = d_pe_offs // 2
        else:
            d_cos_offs = d_pe_offs

        if IS_NEOX:
            d_swap_offs = tl.where(
                d_pe_offs < BLOCK_D_HALF_pe,
                d_pe_offs + BLOCK_D_HALF_pe,
                d_pe_offs - BLOCK_D_HALF_pe,
            ).to(d_pe_offs.dtype)
            neg_mask = d_pe_offs < BLOCK_D_HALF_pe
        else:
            d_swap_offs = d_pe_offs ^ 1
            neg_mask = d_pe_offs % 2 == 0

        pos = tl.load(pos_ptr + offs_t64 * pos_stride_b, mask=t_mask, other=0)
        pid_slot = tl.load(slot_mapping_ptr + offs_t64, mask=t_mask, other=-1).to(
            tl.int64
        )
        q_base = (
            q_pe_ptr
            + offs_t64[:, None, None] * q_pe_stride_b
            + offs_h[None, :, None] * q_pe_stride_h
        )
        q_mask = t_mask[:, None, None]
        x = tl.load(
            q_base + d_pe_offs[None, None, :] * q_pe_stride_d,
            mask=q_mask,
            other=0.0,
            cache_modifier=".cg",
        )
        if BLOCK_QH == QH_PER_KH:
            kx0 = tl.load(
                k_pe_ptr
                + offs_t64[:, None] * k_pe_stride_b
                + d_pe_offs[None, :] * k_pe_stride_d,
                mask=t_mask[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
            k_nope0 = tl.load(
                k_nope_ptr
                + offs_t64[:, None] * k_nope_stride_b
                + dk_nope_offs[None, :] * k_nope_stride_d,
                mask=t_mask[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
        if HAVE_K_SCALE:
            k_scale = tl.load(k_scale_ptr)
        else:
            k_scale = 1
        cos_offs = pos[:, None] * cos_stride_b + d_cos_offs[None, :] * cos_stride_d
        cos = tl.load(
            cos_ptr + cos_offs, mask=t_mask[:, None], other=0.0, cache_modifier=".cg"
        )
        sin = tl.load(
            sin_ptr + cos_offs, mask=t_mask[:, None], other=0.0, cache_modifier=".cg"
        )

        if IS_NEOX:

            x_sw = tl.load(
                q_base + d_swap_offs[None, None, :] * q_pe_stride_d,
                mask=q_mask,
                other=0.0,
            )
            x_rot = tl.where(neg_mask[None, None, :], -x_sw, x_sw)
        else:

            xe, xo = tl.split(
                tl.reshape(x, (ROPE_BLOCK_T, BLOCK_QH, BLOCK_D_HALF_pe, 2))
            )
            x_rot = tl.reshape(tl.join(-xo, xe), (ROPE_BLOCK_T, BLOCK_QH, BLOCK_D_pe))
        q_pe = x * cos[:, None, :] + x_rot * sin[:, None, :]

        q_out_ptrs = (
            c_ptr
            + offs_t64[:, None, None] * q_out_stride_b
            + offs_h[None, :, None] * q_out_stride_h
            + (d_pe_offs[None, None, :] + BLOCK_D_nope) * q_out_stride_d
        )
        tl.store(q_out_ptrs, q_pe.to(c_ptr.dtype.element_ty), mask=q_mask)

        if pid_adjusted * ROPE_BLOCK_T < num_decode_toks_for_zeros:
            z_mask = (offs_t < num_decode_toks_for_zeros)[:, None, None]
            tl.store(
                decode_q_pe_out_ptr
                + offs_t64[:, None, None] * decode_q_pe_out_stride_b
                + offs_h[None, :, None] * decode_q_pe_out_stride_h
                + d_pe_offs[None, None, :] * decode_q_pe_out_stride_d,
                q_pe.to(decode_q_pe_out_ptr.dtype.element_ty),
                mask=z_mask,
            )
            if OUTPUT_Q_NOPE_ZEROS:
                z = tl.zeros(
                    (ROPE_BLOCK_T, BLOCK_QH, BLOCK_DK_nope),
                    dtype=q_nope_zeros_out_ptr.dtype.element_ty,
                )
                tl.store(
                    q_nope_zeros_out_ptr
                    + offs_t64[:, None, None] * q_nope_zeros_out_stride_b
                    + offs_h[None, :, None] * q_nope_zeros_out_stride_h
                    + dk_nope_offs[None, None, :] * q_nope_zeros_out_stride_d,
                    z,
                    mask=z_mask,
                )

        kv_mask = (t_mask & (pid_slot >= 0))[:, None]
        k_scale_rcprl = (1 / k_scale).to(tl.float32)
        for pid_hk in range(QH // QH_PER_KH):
            k_pe_base = (
                k_pe_ptr + offs_t64[:, None] * k_pe_stride_b + pid_hk * k_pe_stride_h
            )
            if BLOCK_QH == QH_PER_KH:
                kx = kx0
            else:
                kx = tl.load(
                    k_pe_base + d_pe_offs[None, :] * k_pe_stride_d,
                    mask=kv_mask,
                    other=0.0,
                )
            if IS_NEOX:
                kx_sw = tl.load(
                    k_pe_base + d_swap_offs[None, :] * k_pe_stride_d,
                    mask=kv_mask,
                    other=0.0,
                )
                kx_rot = tl.where(neg_mask[None, :], -kx_sw, kx_sw)
            else:
                ke, ko = tl.split(tl.reshape(kx, (ROPE_BLOCK_T, BLOCK_D_HALF_pe, 2)))
                kx_rot = tl.reshape(tl.join(-ko, ke), (ROPE_BLOCK_T, BLOCK_D_pe))
            k_pe = kx * cos + kx_rot * sin

            if BLOCK_QH == QH_PER_KH:
                k_nope = k_nope0
            else:
                k_nope = tl.load(
                    k_nope_ptr
                    + offs_t64[:, None] * k_nope_stride_b
                    + pid_hk * k_nope_stride_h
                    + dk_nope_offs[None, :] * k_nope_stride_d,
                    mask=kv_mask,
                    other=0.0,
                )

            tl.store(
                k_pe_out_ptr
                + offs_t64[:, None] * k_pe_out_stride_b
                + pid_hk * k_pe_out_stride_h
                + d_pe_offs[None, :] * k_pe_out_stride_d,
                k_pe.to(k_pe_out_ptr.dtype.element_ty),
                mask=kv_mask,
            )

            k_nope_scaled = (k_nope.to(tl.float32) * k_scale_rcprl).to(
                kv_cache_ptr.dtype.element_ty
            )
            k_pe_scaled = (k_pe.to(tl.float32) * k_scale_rcprl).to(
                kv_cache_ptr.dtype.element_ty
            )
            kv_base = (
                kv_cache_ptr
                + pid_slot[:, None] * kv_cache_stride_b
                + pid_hk * kv_cache_stride_h
            )
            tl.store(
                kv_base + dk_nope_offs[None, :] * kv_cache_stride_d,
                k_nope_scaled,
                mask=kv_mask,
                cache_modifier=".cs",
            )
            tl.store(
                kv_base + (d_pe_offs[None, :] + BLOCK_DK_nope) * kv_cache_stride_d,
                k_pe_scaled,
                mask=kv_mask,
                cache_modifier=".cs",
            )

    elif pid < prefill_start:
        pid_adjusted = pid - rope_start
        pid_b = pid_adjusted // QH
        pid_hq = pid_adjusted % QH

        tl.assume(pid_b >= 0)
        tl.assume(pid_hq >= 0)

        dk_nope_offs = tl.arange(0, BLOCK_DK_nope).to(tl.int64)
        d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

        if REUSE_FREQS_FRONT_PART:
            if IS_NEOX:
                d_cos_offs = d_pe_offs
                d_cos_offs = tl.where(
                    (d_cos_offs >= BLOCK_D_HALF_pe) & (d_cos_offs < BLOCK_D_pe),
                    d_cos_offs - BLOCK_D_HALF_pe,
                    d_cos_offs,
                ).to(d_cos_offs.dtype)
            else:
                d_cos_offs = d_pe_offs // 2
        else:
            d_cos_offs = d_pe_offs

        pos = tl.load(pos_ptr + pid_b * pos_stride_b)
        cos_offs = pos * cos_stride_b + d_cos_offs * cos_stride_d
        cos = tl.load(cos_ptr + cos_offs)
        sin = tl.load(sin_ptr + cos_offs)

        q_pe_ptrs = (
            q_pe_ptr
            + pid_b * q_pe_stride_b
            + pid_hq * q_pe_stride_h
            + d_pe_offs * q_pe_stride_d
        )
        q_pe = _unit_rope(
            q_pe_ptrs,
            cos,
            sin,
            d_pe_offs,
            IS_NEOX,
            BLOCK_D_pe,
            BLOCK_D_HALF_pe,
        )

        q_out_base = c_ptr + pid_b * q_out_stride_b + pid_hq * q_out_stride_h
        tl.store(
            q_out_base + (d_pe_offs + BLOCK_D_nope) * q_out_stride_d,
            q_pe.to(c_ptr.dtype.element_ty),
        )

        if pid_adjusted < num_decode_toks_for_zeros * QH:
            decode_q_pe_out_ptrs = (
                decode_q_pe_out_ptr
                + pid_b * decode_q_pe_out_stride_b
                + pid_hq * decode_q_pe_out_stride_h
                + d_pe_offs * decode_q_pe_out_stride_d
            )
            tl.store(
                decode_q_pe_out_ptrs, q_pe.to(decode_q_pe_out_ptr.dtype.element_ty)
            )

        if OUTPUT_Q_NOPE_ZEROS and pid_adjusted < num_decode_toks_for_zeros * QH:
            z = tl.zeros((BLOCK_DK_nope,), dtype=q_nope_zeros_out_ptr.dtype.element_ty)
            tl.store(
                q_nope_zeros_out_ptr
                + pid_b * q_nope_zeros_out_stride_b
                + pid_hq * q_nope_zeros_out_stride_h
                + dk_nope_offs * q_nope_zeros_out_stride_d,
                z,
            )

        if pid_hq % QH_PER_KH == 0:
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)
            if pid_slot >= 0:
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1

                pid_hk = pid_hq // QH_PER_KH

                k_nope_ptrs = (
                    k_nope_ptr
                    + pid_b * k_nope_stride_b
                    + pid_hk * k_nope_stride_h
                    + dk_nope_offs * k_nope_stride_d
                )
                k_nope = tl.load(k_nope_ptrs)

                k_pe_load_ptrs = (
                    k_pe_ptr
                    + pid_b * k_pe_stride_b
                    + pid_hk * k_pe_stride_h
                    + d_pe_offs * k_pe_stride_d
                )
                k_pe = _unit_rope(
                    k_pe_load_ptrs,
                    cos,
                    sin,
                    d_pe_offs,
                    IS_NEOX,
                    BLOCK_D_pe,
                    BLOCK_D_HALF_pe,
                )

                k_pe_out_ptrs = (
                    k_pe_out_ptr
                    + pid_b * k_pe_out_stride_b
                    + pid_hk * k_pe_out_stride_h
                    + d_pe_offs * k_pe_out_stride_d
                )
                tl.store(k_pe_out_ptrs, k_pe.to(k_pe_out_ptr.dtype.element_ty))

                k_scale_rcprl = (1 / k_scale).to(tl.float32)
                k_nope_scaled = (k_nope.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )
                k_pe_scaled = (k_pe.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )

                kv_cache_ptrs = (
                    kv_cache_ptr
                    + pid_slot * kv_cache_stride_b
                    + pid_hk * kv_cache_stride_h
                )

                tl.store(
                    kv_cache_ptrs + dk_nope_offs * kv_cache_stride_d, k_nope_scaled
                )
                tl.store(
                    kv_cache_ptrs + (d_pe_offs + BLOCK_DK_nope) * kv_cache_stride_d,
                    k_pe_scaled,
                )

    else:
        pid_adjusted = pid - prefill_start
        pid_b = pid_adjusted // KH + B
        pid_hk = pid_adjusted % KH

        if pid_b < B_slot:
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)

            if pid_slot >= 0:
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1

                dk_nope_offs = tl.arange(0, BLOCK_DK_nope).to(tl.int64)
                d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

                k_nope_ptrs = (
                    k_nope_ptr
                    + pid_b * k_nope_stride_b
                    + pid_hk * k_nope_stride_h
                    + dk_nope_offs * k_nope_stride_d
                )
                k_nope = tl.load(k_nope_ptrs)

                k_pe_load_ptrs = (
                    k_pe_ptr
                    + pid_b * k_pe_stride_b
                    + pid_hk * k_pe_stride_h
                    + d_pe_offs * k_pe_stride_d
                )
                k_pe = tl.load(k_pe_load_ptrs)

                k_pe_out_ptrs = (
                    k_pe_out_ptr
                    + pid_b * k_pe_out_stride_b
                    + pid_hk * k_pe_out_stride_h
                    + d_pe_offs * k_pe_out_stride_d
                )
                tl.store(k_pe_out_ptrs, k_pe.to(k_pe_out_ptr.dtype.element_ty))

                k_scale_rcprl = (1 / k_scale).to(tl.float32)
                k_nope_scaled = (k_nope.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )
                k_pe_scaled = (k_pe.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )

                kv_cache_ptrs = (
                    kv_cache_ptr
                    + pid_slot * kv_cache_stride_b
                    + pid_hk * kv_cache_stride_h
                )

                tl.store(
                    kv_cache_ptrs + dk_nope_offs * kv_cache_stride_d, k_nope_scaled
                )
                tl.store(
                    kv_cache_ptrs + (d_pe_offs + BLOCK_DK_nope) * kv_cache_stride_d,
                    k_pe_scaled,
                )


def _get_fp4_config(
    M: int,
    N: int,
    K: int,
):
    # Own config family: the fused BMM + RoPE grid prefers different tiles
    # than the standalone batched_gemm_a16wfp4 kernel.
    return get_gemm_config(
        "FUSED-FP4-BMM-ROPE-CAT-AND-CACHE-MLA",
        M,
        N,
        2 * K,
    )


def _get_fp8_config(
    M: int,
    N: int,
    K: int,
):
    # Own config family: the fused BMM + RoPE grid prefers different tiles
    # than the standalone batched_gemm_a8w8 prequant kernel.
    return get_gemm_config(
        "FUSED-FP8-BMM-ROPE-CAT-AND-CACHE-MLA",
        M,
        N,
        K,
    )


def _get_fp8_prefill_config(M: int, N: int, K: int):
    return get_gemm_config("FUSED-FP8-BMM-ROPE-PREFILL", M, N, K)
