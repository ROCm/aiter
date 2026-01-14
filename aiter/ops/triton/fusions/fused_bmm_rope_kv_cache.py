# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple

import torch
import triton

from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.common_utils import serialize_dict, deserialize_str
from aiter.ops.triton.gemm_a16wfp4 import get_splitk
from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a16wfp4 import _get_config as _get_fp4_config
from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    _get_config as _get_fp8_config,
)
from aiter.ops.triton._triton_kernels.fusions.fused_bmm_rope_kv_cache import (
    _fused_fp4_bmm_rope_cat_and_cache_mla_kernel,
    _fused_fp4_bmm_reduce_kernel,
    _fused_fp8_bmm_rope_cat_and_cache_mla_kernel,
)

_LOGGER = AiterTritonLogger()

global _USE_GEMM_SPLITK_BF16
_USE_GEMM_SPLITK_BF16 = False


def set_use_gemm_splitk_bf16(value: bool):
    global _USE_GEMM_SPLITK_BF16
    _USE_GEMM_SPLITK_BF16 = value



def fused_fp4_bmm_rope_cat_and_cache_mla(
    q_nope: torch.Tensor,          
    w_k: torch.Tensor,             
    w_k_scale: torch.Tensor,  
    q_pe: torch.Tensor,            
    k_nope: torch.Tensor,          
    k_rope: torch.Tensor,          
    kv_cache: torch.Tensor,        
    slot_mapping: torch.Tensor,    
    positions: torch.Tensor,       
    cos: torch.Tensor,          
    sin: torch.Tensor,          
    y: Optional[torch.Tensor] = None,
    transpose_bm: bool = True,
    prequant: bool = True,
    y_scale: Optional[torch.Tensor] = None,
    config: Optional[str] = None,
    k_scale: Optional[torch.Tensor] = None,
    is_neox: bool = False,
    q_out_dtype=None,
    num_decode_toks_for_zeros: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused FP4 BMM + RoPE + concat + KV-cache write for MLA.

    This function fuses two INDEPENDENT operations to reduce kernel launch overhead:
    1. Batched FP4 GEMM: ql_nope = q_nope @ w_k^T (with FP4 quantization)
       - Writes to q_out[:, :, :kv_lora_rank]
    2. RoPE + KV cache: Apply RoPE to q_pe and k_rope, write to cache
       - Writes to q_out[:, :, kv_lora_rank:] (the q_pe part after RoPE)
       - Writes k_nope and rotated k_pe to kv_cache

    Grid structure:
    - pid < QH * NUM_KSPLIT * grid_mn: BMM computation (tiled over heads, K-splits, M, N)
    - pid >= QH * NUM_KSPLIT * grid_mn and pid < ... + B * QH: RoPE + KV cache for decode
    - pid >= ... + B * QH: KV cache only for prefill tokens

    Args:
        q_nope: Query without positional encoding, shape (QH, B, P)
        w_k: FP4 weight matrix, shape (QH, kv_lora_rank, P//2) (packed)
        w_k_scale: E8M0 scales for w_k, shape (QH, kv_lora_rank, P//32)
        q_pe: Query positional encoding part, shape (B, QH, d_pe)
        k_nope: Key without rope, shape (B_slot, KH, kv_lora_rank)
        k_rope: Key rope part, shape (B_slot, KH, qk_rope_head_dim)
        kv_cache: KV cache, shape (B_cache, KH, kv_lora_rank + qk_rope_head_dim)
        slot_mapping: Mapping to cache slots, shape (B_slot,)
        positions: Position indices, shape (B_slot,)
        cos: Cosine cache, shape (max_pos, d_freq) or (max_pos, 1, 1, d_freq)
        sin: Sine cache, shape (max_pos, d_freq) or (max_pos, 1, 1, d_freq)
        y: Optional pre-allocated output for BMM (unused, for API compatibility)
        transpose_bm: Whether to transpose batch and M dimensions in BMM output
        prequant: Whether to use pre-quantization in BMM
        y_scale: Optional scale for BMM output
        config: Optional BMM kernel configuration (serialized string)
        k_scale: Optional scale for K values (can be float or tensor)
        is_neox: Whether to use NeoX-style RoPE
        q_out_dtype: Output dtype for q_out
        num_decode_toks_for_zeros: Number of decode tokens to output zeros for

    Returns:
        Tuple of:
        - q_out: Concatenated query output, shape (B, QH, kv_lora_rank + d_pe)
        - decode_q_pe_out: Decoded query PE output, shape (num_decode_toks_for_zeros, QH, d_pe)
        - k_pe_out: Key PE output, shape (B_slot, KH, qk_rope_head_dim)
        - kv_cache: Updated KV cache (modified in-place)
    """
    _LOGGER.info(
        "FUSED_FP4_BMM_ROPE_CAT_AND_CACHE_MLA: "
        f"q_nope={tuple(q_nope.shape)} w_k={tuple(w_k.shape)} w_k_scale={tuple(w_k_scale.shape)} "
        f"q_pe={tuple(q_pe.shape)} k_nope={tuple(k_nope.shape)} k_rope={tuple(k_rope.shape)} "
        f"positions={tuple(positions.shape)} cos={tuple(cos.shape)} sin={tuple(sin.shape)} "
        f"kv_cache={tuple(kv_cache.shape)} slot_mapping={tuple(slot_mapping.shape)} "
        f"transpose_bm={transpose_bm} prequant={prequant} is_neox={is_neox}"
    )

    assert prequant is True, "prequant=False is not yet supported in fused kernel"

    if cos.dim() == 4:
        cos = cos.squeeze(1).squeeze(1)
        sin = sin.squeeze(1).squeeze(1)

    if k_scale is not None and not isinstance(k_scale, torch.Tensor):
        k_scale = torch.tensor(k_scale, dtype=torch.float32, device=q_nope.device)

    qh, b, p = q_nope.shape
    qh_w, n, k_packed = w_k.shape
    k = k_packed * 2  

    b2, qh2, d_pe = q_pe.shape

    bk, kh, kv_lora_rank = k_nope.shape
    bk2, kh2, qk_rope_head_dim = k_rope.shape

    b_cache, h_cache, d_cache = kv_cache.shape
    (b_slot,) = slot_mapping.shape

    assert qh == qh_w == qh2, f"Query head dimensions must match: {qh} vs {qh_w} vs {qh2}"
    assert b == b2, f"Batch dimensions must match: {b} vs {b2}"
    assert bk == bk2, f"K batch dimensions must match: {bk} vs {bk2}"
    assert kh == kh2 == h_cache, f"KV head dimensions must match: {kh} vs {kh2} vs {h_cache}"
    assert b_slot <= bk, f"slot_mapping batch must not exceed k batch: {b_slot} > {bk}"
    assert b <= bk, f"q batch must not exceed k batch: {b} > {bk}"
    assert qh % kh == 0, f"Query heads must be multiple of KV heads: {qh} % {kh} != 0"
    assert p == k, f"BMM K dimension mismatch: q_nope has {p}, w_k (unpacked) has {k}"
    assert n == kv_lora_rank, f"BMM output dim must match kv_lora_rank: {n} vs {kv_lora_rank}"
    assert kv_lora_rank + qk_rope_head_dim == d_cache, \
        f"k_nope + k_rope dims must equal kv_cache dim: {kv_lora_rank} + {qk_rope_head_dim} != {d_cache}"

    d_freq = cos.shape[-1]
    assert (d_freq == d_pe // 2) or (d_freq == d_pe), \
        f"cos/sin last dim should be half or equal to d_pe: {d_freq} vs {d_pe}"
    reuse_freqs_front_part = d_freq == d_pe // 2

    M = b
    N = kv_lora_rank
    K = k_packed

    if config is None:
        config, _ = _get_fp4_config(M, N, K)
    else:
        config = deserialize_str(config)

    if config["NUM_KSPLIT"] > 1:
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
            K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
        )
        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K

    if config["BLOCK_SIZE_K"] >= 2 * K:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
        config["SPLITK_BLOCK_SIZE"] = 2 * K

    NUM_KSPLIT = config["NUM_KSPLIT"]

    num_pid_m = triton.cdiv(M, config["BLOCK_SIZE_M"])
    num_pid_n = triton.cdiv(N, config["BLOCK_SIZE_N"])
    grid_mn = num_pid_m * num_pid_n

    bmm_programs = qh * NUM_KSPLIT * grid_mn

    rope_programs = b * qh

    prefill_programs = (b_slot - b) * kh

    total_programs = bmm_programs + rope_programs + prefill_programs

    grid = (total_programs, 1, 1)

    if q_out_dtype is None:
        q_out_dtype = q_nope.dtype

    q_out = torch.empty(
        (b, qh, kv_lora_rank + d_pe),
        dtype=q_out_dtype,
        device=q_nope.device,
    )

    decode_q_pe_out = torch.empty(
        (num_decode_toks_for_zeros, qh, d_pe),
        dtype=q_nope.dtype,
        device=q_nope.device,
    )

    k_pe_out = torch.empty(
        (bk, kh, qk_rope_head_dim),
        dtype=k_rope.dtype,
        device=k_rope.device,
    )

    q_nope_zeros_out = None
    if num_decode_toks_for_zeros > 0:
        q_nope_zeros_out = torch.empty(
            (num_decode_toks_for_zeros, qh, kv_lora_rank),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )

    if NUM_KSPLIT > 1:
        if _USE_GEMM_SPLITK_BF16:
            y_pp = torch.empty(
                (qh, NUM_KSPLIT, M, N),
                dtype=q_out_dtype,
                device=q_nope.device,
            )
        else:
            y_pp = torch.empty(
                (qh, NUM_KSPLIT, M, N),
                dtype=torch.float32,
                device=q_nope.device,
            )
        c_ptr = y_pp
        stride_cb = y_pp.stride(0)
        stride_ck = y_pp.stride(1)
        stride_cm = y_pp.stride(2)
        stride_cn = y_pp.stride(3)
    else:
        y_pp = None
        c_ptr = q_out
        if transpose_bm:
            stride_cb = q_out.stride(0)  
            stride_ck = 0                
            stride_cm = q_out.stride(1)  
            stride_cn = q_out.stride(2)  
        else:
            stride_cb = q_out.stride(1)  
            stride_ck = 0
            stride_cm = q_out.stride(0)  
            stride_cn = q_out.stride(2)  


    _fused_fp4_bmm_rope_cat_and_cache_mla_kernel[grid](
        q_nope,
        w_k,
        w_k_scale,
        c_ptr,
        y_scale,
        q_pe,
        k_nope,
        k_rope,
        positions,
        cos,
        sin,
        q_out,
        decode_q_pe_out,
        k_pe_out,
        q_nope_zeros_out if q_nope_zeros_out is not None else q_out,
        kv_cache,
        slot_mapping,
        M,
        N,
        K,
        b,
        b_slot,
        num_decode_toks_for_zeros,
        qh,
        kh,
        bmm_programs,
        grid_mn,
        num_pid_m,
        num_pid_n,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        w_k.stride(0),
        w_k.stride(1),
        w_k.stride(2),
        w_k_scale.stride(0),
        w_k_scale.stride(1),
        w_k_scale.stride(2),
        stride_cb,
        stride_ck,
        stride_cm,
        stride_cn,
        q_pe.stride(0),
        q_pe.stride(1),
        q_pe.stride(2),
        k_nope.stride(0),
        k_nope.stride(1),
        k_nope.stride(2),
        k_rope.stride(0),
        k_rope.stride(1),
        k_rope.stride(2),
        positions.stride(0),
        cos.stride(0),
        cos.stride(-1),
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        decode_q_pe_out.stride(0) if num_decode_toks_for_zeros > 0 else 0,
        decode_q_pe_out.stride(1) if num_decode_toks_for_zeros > 0 else 0,
        decode_q_pe_out.stride(2) if num_decode_toks_for_zeros > 0 else 0,
        k_pe_out.stride(0),
        k_pe_out.stride(1),
        k_pe_out.stride(2),
        q_nope_zeros_out.stride(0) if q_nope_zeros_out is not None else 0,
        q_nope_zeros_out.stride(1) if q_nope_zeros_out is not None else 0,
        q_nope_zeros_out.stride(2) if q_nope_zeros_out is not None else 0,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        k_scale_ptr=k_scale,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        NUM_KSPLIT=NUM_KSPLIT,
        SPLITK_BLOCK_SIZE=config["SPLITK_BLOCK_SIZE"],
        QH_PER_KH=qh // kh,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        BLOCK_D_nope=kv_lora_rank,
        BLOCK_DK_nope=kv_lora_rank,
        BLOCK_D_pe=d_pe,
        BLOCK_D_HALF_pe=d_pe // 2,
        PRE_QUANT=prequant,
        TRANSPOSE_BM=transpose_bm,
        OUTPUT_Q_NOPE_ZEROS=(q_nope_zeros_out is not None),
        HAVE_Y_SCALE=(y_scale is not None),
        HAVE_K_SCALE=(k_scale is not None),
        num_warps=1,
    )

    if NUM_KSPLIT > 1:
        REDUCE_BLOCK_SIZE_M = 16
        REDUCE_BLOCK_SIZE_N = 128 if _USE_GEMM_SPLITK_BF16 else 64
        ACTUAL_KSPLIT = triton.cdiv(K, (config["SPLITK_BLOCK_SIZE"] // 2))

        grid_reduce = (
            qh,
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )

        _fused_fp4_bmm_reduce_kernel[grid_reduce](
            y_pp,
            q_out,
            M,
            N,
            qh,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y_pp.stride(3),
            q_out.stride(0),
            q_out.stride(1),
            q_out.stride(2),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            NUM_KSPLIT,
            transpose_bm,
        )

    if num_decode_toks_for_zeros > 0:
        return q_out, decode_q_pe_out, k_pe_out, kv_cache, q_nope_zeros_out
    return q_out, decode_q_pe_out, k_pe_out, kv_cache


def fused_fp8_bmm_rope_cat_and_cache_mla(

    q_nope: torch.Tensor,          
    w_k: torch.Tensor,            
    w_k_scale: torch.Tensor,      


    q_pe: torch.Tensor,           
    k_nope: torch.Tensor,         
    k_rope: torch.Tensor,          
    kv_cache: torch.Tensor,       
    slot_mapping: torch.Tensor,   
    positions: torch.Tensor,      
    cos: torch.Tensor,            
    sin: torch.Tensor,            


    group_size: int = 128,
    transpose_bm: bool = True,
    config: Optional[dict] = None,

    k_scale: Optional[torch.Tensor] = None,
    is_neox: bool = False,
    q_out_dtype: Optional[torch.dtype] = torch.bfloat16,
    num_decode_toks_for_zeros: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused FP8 BMM + RoPE + concat + KV-cache write for MLA.

    This function fuses two INDEPENDENT operations to reduce kernel launch overhead:
    1. Batched FP8 GEMM: ql_nope = q_nope @ w_k^T (with per-token group quantization)
       - Writes to q_out[:, :, :kv_lora_rank]
    2. RoPE + KV cache: Apply RoPE to q_pe and k_rope, write to cache
       - Writes to q_out[:, :, kv_lora_rank:] (the q_pe part after RoPE)
       - Writes k_nope and rotated k_pe to kv_cache

    Note: FP8 does not support split-K, so no reduce kernel is needed.

    Grid structure:
    - pid < QH * grid_mn: BMM computation (tiled over heads, M, N)
    - pid >= QH * grid_mn and pid < ... + B * QH: RoPE + KV cache for decode
    - pid >= ... + B * QH: KV cache only for prefill tokens

    Args:
        q_nope: Query without positional encoding, shape (QH, B, P)
        w_k: FP8 weight matrix, shape (QH, kv_lora_rank, P) - will be transposed internally
        w_k_scale: Per-batch scale for w_k, shape (1,)
        q_pe: Query positional encoding part, shape (B, QH, d_pe)
        k_nope: Key without rope, shape (B_slot, KH, kv_lora_rank)
        k_rope: Key rope part, shape (B_slot, KH, qk_rope_head_dim)
        kv_cache: KV cache, shape (B_cache, KH, kv_lora_rank + qk_rope_head_dim)
        slot_mapping: Mapping to cache slots, shape (B_slot,)
        positions: Position indices, shape (B_slot,)
        cos: Cosine cache, shape (max_pos, d_freq) or (max_pos, 1, 1, d_freq)
        sin: Sine cache, shape (max_pos, d_freq) or (max_pos, 1, 1, d_freq)
        group_size: Group size for per-token quantization (must be power of 2)
        transpose_bm: Whether to transpose batch and M dimensions in BMM output
        config: Optional BMM kernel configuration
        k_scale: Optional scale for K values (can be float or tensor)
        is_neox: Whether to use NeoX-style RoPE
        q_out_dtype: Output dtype for q_out
        num_decode_toks_for_zeros: Number of decode tokens to output zeros for

    Returns:
        Tuple of:
        - q_out: Concatenated query output, shape (B, QH, kv_lora_rank + d_pe)
        - decode_q_pe_out: Decoded query PE output, shape (num_decode_toks_for_zeros, QH, d_pe)
        - k_pe_out: Key PE output, shape (B_slot, KH, qk_rope_head_dim)
        - kv_cache: Updated KV cache (modified in-place)
    """
    _LOGGER.info(
        "FUSED_FP8_BMM_ROPE_CAT_AND_CACHE_MLA: "
        f"q_nope={tuple(q_nope.shape)} w_k={tuple(w_k.shape)} w_k_scale={tuple(w_k_scale.shape)} "
        f"q_pe={tuple(q_pe.shape)} k_nope={tuple(k_nope.shape)} k_rope={tuple(k_rope.shape)} "
        f"positions={tuple(positions.shape)} cos={tuple(cos.shape)} sin={tuple(sin.shape)} "
        f"kv_cache={tuple(kv_cache.shape)} slot_mapping={tuple(slot_mapping.shape)} "
        f"transpose_bm={transpose_bm} group_size={group_size} is_neox={is_neox}"
    )

    if cos.dim() == 4:
        cos = cos.squeeze(1).squeeze(1)
        sin = sin.squeeze(1).squeeze(1)

    if k_scale is not None and not isinstance(k_scale, torch.Tensor):
        k_scale = torch.tensor(k_scale, dtype=torch.float32, device=q_nope.device)


    qh, b, p = q_nope.shape
    qh_w, n, k = w_k.shape

    b2, qh2, d_pe = q_pe.shape

    bk, kh, kv_lora_rank = k_nope.shape
    bk2, kh2, qk_rope_head_dim = k_rope.shape

    b_cache, h_cache, d_cache = kv_cache.shape
    (b_slot,) = slot_mapping.shape

    assert qh == qh_w == qh2, f"Query head dimensions must match: {qh} vs {qh_w} vs {qh2}"
    assert b == b2, f"Batch dimensions must match: {b} vs {b2}"
    assert bk == bk2, f"K batch dimensions must match: {bk} vs {bk2}"
    assert kh == kh2 == h_cache, f"KV head dimensions must match: {kh} vs {kh2} vs {h_cache}"
    assert b_slot <= bk, f"slot_mapping batch must not exceed k batch: {b_slot} > {bk}"
    assert b <= bk, f"q batch must not exceed k batch: {b} > {bk}"
    assert qh % kh == 0, f"Query heads must be multiple of KV heads: {qh} % {kh} != 0"
    assert p == k, f"BMM K dimension mismatch: q_nope has {p}, w_k has {k}"
    assert n == kv_lora_rank, f"BMM output dim must match kv_lora_rank: {n} vs {kv_lora_rank}"
    assert kv_lora_rank + qk_rope_head_dim == d_cache, \
        f"k_nope + k_rope dims must equal kv_cache dim: {kv_lora_rank} + {qk_rope_head_dim} != {d_cache}"
    assert triton.next_power_of_2(group_size) == group_size, "group_size must be power of 2"

    d_freq = cos.shape[-1]
    assert (d_freq == d_pe // 2) or (d_freq == d_pe), \
        f"cos/sin last dim should be half or equal to d_pe: {d_freq} vs {d_pe}"
    reuse_freqs_front_part = d_freq == d_pe // 2

    w_k_t = w_k.transpose(1, 2)


    M = b
    N = kv_lora_rank
    K = k

    if config is None:
        config, _ = _get_fp8_config(M, N, K)
    
    config["BLOCK_SIZE_K"] = group_size


    num_pid_m = triton.cdiv(M, config["BLOCK_SIZE_M"])
    num_pid_n = triton.cdiv(N, config["BLOCK_SIZE_N"])
    grid_mn = num_pid_m * num_pid_n

    bmm_programs = qh * grid_mn

    rope_programs = b * qh

    prefill_programs = (b_slot - b) * kh

    total_programs = bmm_programs + rope_programs + prefill_programs

    grid = (total_programs, 1, 1)


    if q_out_dtype is None:
        q_out_dtype = torch.bfloat16

    q_out = torch.empty(
        (b, qh, kv_lora_rank + d_pe),
        dtype=q_out_dtype,
        device=q_nope.device,
    )

    decode_q_pe_out = torch.empty(
        (num_decode_toks_for_zeros, qh, d_pe),
        dtype=q_nope.dtype,
        device=q_nope.device,
    )

    k_pe_out = torch.empty(
        (bk, kh, qk_rope_head_dim),
        dtype=k_rope.dtype,
        device=k_rope.device,
    )

    q_nope_zeros_out = None
    if num_decode_toks_for_zeros > 0:
        q_nope_zeros_out = torch.empty(
            (num_decode_toks_for_zeros, qh, kv_lora_rank),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )

    if transpose_bm:
        stride_cb = q_out.stride(0)  
        stride_cm = q_out.stride(1)  
        stride_cn = q_out.stride(2)  
    else:
        stride_cb = q_out.stride(1)  
        stride_cm = q_out.stride(0)  
        stride_cn = q_out.stride(2)  


    DTYPE_MAX = (
        torch.finfo(w_k_t.dtype).max
        if torch.is_floating_point(w_k_t)
        else torch.iinfo(w_k_t.dtype).max
    )

    _fused_fp8_bmm_rope_cat_and_cache_mla_kernel[grid](
        q_nope,
        w_k_t,
        w_k_scale,
        q_out,
        q_pe,
        k_nope,
        k_rope,
        positions,
        cos,
        sin,
        decode_q_pe_out,
        k_pe_out,
        q_nope_zeros_out if q_nope_zeros_out is not None else q_out,
        kv_cache,
        slot_mapping,
        M,
        N,
        K,
        b,
        b_slot,
        num_decode_toks_for_zeros,
        qh,
        kh,
        bmm_programs,
        grid_mn,
        num_pid_m,
        num_pid_n,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        w_k_t.stride(0),
        w_k_t.stride(1),
        w_k_t.stride(2),
        stride_cb,
        stride_cm,
        stride_cn,
        q_pe.stride(0),
        q_pe.stride(1),
        q_pe.stride(2),
        k_nope.stride(0),
        k_nope.stride(1),
        k_nope.stride(2),
        k_rope.stride(0),
        k_rope.stride(1),
        k_rope.stride(2),
        positions.stride(0),
        cos.stride(0),
        cos.stride(-1),
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        decode_q_pe_out.stride(0) if num_decode_toks_for_zeros > 0 else 0,
        decode_q_pe_out.stride(1) if num_decode_toks_for_zeros > 0 else 0,
        decode_q_pe_out.stride(2) if num_decode_toks_for_zeros > 0 else 0,
        k_pe_out.stride(0),
        k_pe_out.stride(1),
        k_pe_out.stride(2),
        q_nope_zeros_out.stride(0) if q_nope_zeros_out is not None else 0,
        q_nope_zeros_out.stride(1) if q_nope_zeros_out is not None else 0,
        q_nope_zeros_out.stride(2) if q_nope_zeros_out is not None else 0,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        k_scale_ptr=k_scale,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        QH_PER_KH=qh // kh,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        BLOCK_D_nope=kv_lora_rank,
        BLOCK_DK_nope=kv_lora_rank,
        BLOCK_D_pe=d_pe,
        BLOCK_D_HALF_pe=d_pe // 2,
        TRANSPOSE_BM=transpose_bm,
        OUTPUT_Q_NOPE_ZEROS=(q_nope_zeros_out is not None),
        HAVE_K_SCALE=(k_scale is not None),
        DTYPE_MAX=DTYPE_MAX,
        num_warps=1,
    )

    if num_decode_toks_for_zeros > 0:
        return q_out, decode_q_pe_out, k_pe_out, kv_cache, q_nope_zeros_out
    return q_out, decode_q_pe_out, k_pe_out, kv_cache