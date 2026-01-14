# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant import _mxfp4_quant_op
from aiter.ops.triton.rope import _get_gptj_rotated_x_1D, _get_neox_rotated_x_1D
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd


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
        "TRANSPOSE_BM",
        "OUTPUT_Q_NOPE_ZEROS",
        "HAVE_K_SCALE",
        "EVEN_K",
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
    TRANSPOSE_BM: tl.constexpr,
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
        stride_cb_i64 = tl.cast(stride_cb, tl.int64)
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
            remap_xcd(pid_tile, GRID_MN)
            pid_m, pid_n = pid_grid(pid_tile, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
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
                        a_ptrs, mask=offs_k_bf16[None, :] < K - k_idx * BLOCK_SIZE_K, other=0
                    )
                    b = tl.load(
                        b_ptrs, mask=offs_k[:, None] < K - k_idx * (BLOCK_SIZE_K // 2), other=0
                    )

                if PRE_QUANT:
                    a, a_scales = _mxfp4_quant_op(
                        a_bf16, BLOCK_SIZE_K, BLOCK_SIZE_M, SCALE_GROUP_SIZE
                    )

                accumulator += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")

                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
                b_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bsk

            if HAVE_Y_SCALE:
                accumulator = accumulator * c_scale_rcprl

            c = accumulator.to(c_ptr.type.element_ty)

            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)

            if NUM_KSPLIT == 1:
                if TRANSPOSE_BM:
                    c_ptrs = (
                        c_ptr
                        + offs_cm[:, None] * stride_cb
                        + pid_head_i64 * stride_cm
                        + offs_cn[None, :] * stride_cn
                    )
                else:
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
                    + pid_k * stride_ck
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
            tl.store(decode_q_pe_out_ptrs, q_pe.to(decode_q_pe_out_ptr.dtype.element_ty))

        if OUTPUT_Q_NOPE_ZEROS:
            if pid_adjusted < num_decode_toks_for_zeros * QH:
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

                tl.store(kv_cache_ptrs + dk_nope_offs * kv_cache_stride_d, k_nope_scaled)
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

                tl.store(kv_cache_ptrs + dk_nope_offs * kv_cache_stride_d, k_nope_scaled)
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
        + (offs_k[:, None, None] * stride_c_in_k)
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



@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_fused_fp8_bmm_rope_cat_and_cache_mla_repr)
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
    TRANSPOSE_BM: tl.constexpr,
    OUTPUT_Q_NOPE_ZEROS: tl.constexpr,
    HAVE_K_SCALE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr = ".ca",
):
    """
    Fused kernel for FP8 BMM + RoPE + KV cache write.

    These are INDEPENDENT operations fused to reduce kernel launch overhead:
    - BMM writes to q_out[:, :, :kv_lora_rank]
    - RoPE writes to q_out[:, :, kv_lora_rank:] and handles KV cache

    Note: FP8 does not support split-K.

    Grid structure:
    - Phase 1: pid < bmm_programs → BMM (tiled over heads, M, N)
    - Phase 2: pid in [bmm_programs, bmm_programs + B*QH) → RoPE + KV cache for decode
    - Phase 3: pid >= bmm_programs + B*QH → KV cache only for prefill tokens
    """

    pid = tl.program_id(0)
    tl.assume(pid >= 0)

    rope_start = bmm_programs
    prefill_start = bmm_programs + B * QH

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

        pid_head = pid // GRID_MN
        pid_tile = pid % GRID_MN

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

        for k_idx in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k_idx * BLOCK_SIZE_K, other=0.0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_idx * BLOCK_SIZE_K, other=0.0)

            # Per-token group quantization
            m = tl.maximum(tl.max(tl.abs(a), axis=-1), 1e-10)[:, None]
            a_scale = m.to(tl.float32) * one_over_DTYPE_MAX
            a_scale_recip = 1.0 / a_scale
            a = tl.clamp(a * a_scale_recip, -DTYPE_MAX, DTYPE_MAX).to(b_ptr.dtype.element_ty)

            accumulator += tl.dot(a, b, input_precision="ieee") * a_scale

            a_ptrs += BLOCK_SIZE_K * stride_ak_i64
            b_ptrs += BLOCK_SIZE_K * stride_bk_i64

        accumulator *= b_scale

        c = accumulator.to(c_ptr.type.element_ty)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        if TRANSPOSE_BM:
            c_ptrs = (
                c_ptr
                + offs_cm[:, None] * stride_cb_i64
                + pid_head_i64 * stride_cm_i64
                + offs_cn[None, :] * stride_cn_i64
            )
        else:
            c_ptrs = (
                c_ptr
                + pid_head_i64 * stride_cb_i64
                + offs_cm[:, None] * stride_cm_i64
                + offs_cn[None, :] * stride_cn_i64
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
            tl.store(decode_q_pe_out_ptrs, q_pe.to(decode_q_pe_out_ptr.dtype.element_ty))

        if OUTPUT_Q_NOPE_ZEROS:
            if pid_adjusted < num_decode_toks_for_zeros * QH:
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

                tl.store(kv_cache_ptrs + dk_nope_offs * kv_cache_stride_d, k_nope_scaled)
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

                tl.store(kv_cache_ptrs + dk_nope_offs * kv_cache_stride_d, k_nope_scaled)
                tl.store(
                    kv_cache_ptrs + (d_pe_offs + BLOCK_DK_nope) * kv_cache_stride_d,
                    k_pe_scaled,
                )