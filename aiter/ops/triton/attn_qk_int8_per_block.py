# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import functools
import json
import os
import torch
import triton
import triton.language as tl
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton._triton_kernels.attn_qk_int8_per_block import (
    _attn_fwd_qk_int8_per_block,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# naive torch implementation of quantization for testing
import math
import torch
from typing import Tuple

# log2(e) = 1.44269504, used for exp2-based softmax in attention kernels
LOG2_E = 1.44269504

def int8_per_block_quantize_bshd(
    x: torch.Tensor,
    int8_dtype: torch.dtype,
    tensor_layout: str = "bshd",
    clamp_val: float = 1e-9,
    block_size: int = 128,
    include_sqrt_scale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a tensor to INT8 format, returning an INT8 tensor and a descale factor.
    Quantization is done per block on the seqlen dimension.
    x: [batch, seqlen, heads, dim] (bshd) or [batch, heads, seqlen, dim] (bhsd)
    include_sqrt_scale: if True, pre-multiply by sm_scale * log2(e) before quantization
                        (for attention's Q tensor, matching per_block_int8 behavior)
    
    Note: If seqlen is not divisible by block_size, the tensor is padded with zeros
    to the next multiple of block_size for quantization. The output x_int8 retains
    the original seqlen (unpadded).
    
    Returns:
        x_int8: same shape as x, stored as int8_dtype
        descale_factor: [batch, heads, num_blocks, 1] scale to dequantize:
                        x_fp32 ≈ x_int8 * descale_factor
    """
    if len(x.shape) != 4:
        raise ValueError(
            f"'bshd' tensor should have shape [batch, seqlen, heads, dim], got {x.shape}"
        )
    if tensor_layout == "bshd":
        batch, seqlen, num_heads, head_dim = x.shape
    elif tensor_layout == "bhsd":
        batch, num_heads, seqlen, head_dim = x.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}, supported layouts: bshd, bhsd")
    
    # Calculate number of blocks using ceiling division to handle non-divisible seqlen
    n_blocks = (seqlen + block_size - 1) // block_size
    padded_seqlen = n_blocks * block_size
    
    # Pad tensor if seqlen is not divisible by block_size
    if seqlen != padded_seqlen:
        pad_size = padded_seqlen - seqlen
        if tensor_layout == "bshd":
            # Pad along seq dimension (dim 1): [batch, seqlen, heads, dim]
            x_padded = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0)
        else:  # bhsd
            # Pad along seq dimension (dim 2): [batch, heads, seqlen, dim]
            x_padded = torch.nn.functional.pad(x, (0, 0, 0, pad_size), mode='constant', value=0)
    else:
        x_padded = x
    
    # For Q tensor (include_sqrt_scale=True): pre-multiply by sm_scale * log2(e)
    # This matches per_block_int8 kernel behavior which applies this scaling before quantization
    # The log2(e) factor is needed because attention kernels use exp2 instead of exp for softmax
    if include_sqrt_scale:
        sm_scale = head_dim ** -0.5
        pre_scale = sm_scale * LOG2_E
        x_padded = x_padded.float() * pre_scale
    
    # Reshape to expose blocks along seqlen: [b, n_blocks, block_size, h, d]
    if tensor_layout == "bshd":
        x_reshaped = x_padded.view(batch, n_blocks, block_size, num_heads, head_dim)
    else:  # bhsd
        # [batch, heads, seqlen, dim] -> [batch, heads, n_blocks, block_size, dim]
        x_reshaped = x_padded.view(batch, num_heads, n_blocks, block_size, head_dim)
        # Permute to [batch, n_blocks, block_size, heads, dim] for consistent processing
        x_reshaped = x_reshaped.permute(0, 2, 3, 1, 4)
    
    # Compute max absolute value per block (reduce over block_size and dim)
    # Shape: [b, n_blocks, num_heads, 1]
    max_abs = x_reshaped.abs().amax(dim=2)        # [b, n_blocks, h, d]
    max_abs = max_abs.amax(dim=-1, keepdim=True)  # [b, n_blocks, h, 1]
    # Avoid division by zero
    max_abs = torch.clamp(max_abs, min=clamp_val)
    # Symmetric INT8 range
    qmax = torch.iinfo(torch.int8).max  # 127
    # Scale used for quantization (fp32 -> int8)
    scale = qmax / max_abs  # [b, n_blocks, h, 1]
    # Apply scale per block
    # Broadcast scale over block_size and dim
    x_scaled = x_reshaped * scale.unsqueeze(2)  # [b, n_blocks, block_size, h, d]
    # Quantize and clamp to valid INT8 range
    x_int8 = torch.round(x_scaled).clamp(-qmax - 1, qmax).to(int8_dtype)
    
    # Reshape back to original layout and slice to original seqlen
    if tensor_layout == "bshd":
        x_int8 = x_int8.view(batch, padded_seqlen, num_heads, head_dim)
        x_int8 = x_int8[:, :seqlen, :, :].contiguous()  # Slice back to original seqlen
    else:  # bhsd
        # Permute back to [batch, heads, n_blocks, block_size, dim]
        x_int8 = x_int8.permute(0, 3, 1, 2, 4)
        x_int8 = x_int8.reshape(batch, num_heads, padded_seqlen, head_dim)
        x_int8 = x_int8[:, :, :seqlen, :].contiguous()  # Slice back to original seqlen
    
    # Descale factor for dequantization: x_fp32 ≈ x_int8 * descale_factor
    # descale = max_abs / qmax (inverse of quantization scale)
    descale_factor = max_abs / qmax  # [b, n_blocks, h, 1]
    
    # Kernel expects scale in [b, h, n_blocks, 1] format, so permute from [b, n_blocks, h, 1]
    # Must call contiguous() after permute so kernel's pointer arithmetic (+= 1) works correctly
    descale_factor = descale_factor.permute(0, 2, 1, 3).contiguous()  # [b, h, n_blocks, 1]
    return x_int8, descale_factor

@triton.jit
def quant_per_block_int8_kernel(Input, Output, Scale, L,
                                stride_iz, stride_ih, stride_in,
                                stride_oz, stride_oh, stride_on,
                                stride_sz, stride_sh,
                                sm_scale,
                                C: tl.constexpr, BLK: tl.constexpr):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    x *= sm_scale
    scale = tl.max(tl.abs(x)) / 127.
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)

def compute_k_smoothing_factors(k: torch.Tensor, tensor_layout: str = "NHD") -> torch.Tensor:
    """
    Compute per-channel smoothing factors for K tensor following SageAttention approach.
    
    This computes the mean across the sequence dimension for each channel (head_dim)
    to reduce outliers before quantization, improving INT8 accuracy.
    
    Args:
        k: Key tensor with shape (B, kv_len, H, head_dim) for NHD layout
           or (B, H, kv_len, head_dim) for HND layout
        tensor_layout: Either "NHD" or "HND"
    
    Returns:
        k_smooth: Smoothing factors with shape matching k, computed as per-channel mean
    """
    if tensor_layout == "NHD":
        # k shape: [B, kv_len, H, head_dim]
        # Compute mean across sequence dimension (dim=1), keep dims for broadcasting
        k_mean = k.mean(dim=1, keepdim=True)  # [B, 1, H, head_dim]
    elif tensor_layout == "HND":
        # k shape: [B, H, kv_len, head_dim]
        # Compute mean across sequence dimension (dim=2), keep dims for broadcasting
        k_mean = k.mean(dim=2, keepdim=True)  # [B, H, 1, head_dim]
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    
    return k_mean

def per_block_int8(q, k, km=None, BLKQ=128, BLKK=64, sm_scale=None, tensor_layout="NHD", smooth_k=True):
    """
    Quantize Q and K tensors to INT8 with per-block scaling.
    
    Args:
        q: Query tensor
        k: Key tensor
        km: Optional pre-computed K smoothing factors (if None and smooth_k=True, will be computed)
        BLKQ: Block size for Q quantization
        BLKK: Block size for K quantization
        sm_scale: Softmax scale factor (defaults to head_dim^-0.5)
        tensor_layout: Either "NHD" or "HND"
        smooth_k: Whether to apply SageAttention-style smoothing to K tensor (default: True)
    
    Returns:
        q_int8: Quantized Q tensor
        q_scale: Per-block scales for Q
        k_int8: Quantized K tensor  
        k_scale: Per-block scales for K
        k_smooth: K smoothing factors applied (or None if smooth_k=False)
    """
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    # Apply K tensor smoothing following SageAttention approach
    k_smooth = None
    if smooth_k:
        if km is None:
            km = compute_k_smoothing_factors(k, tensor_layout)
            k_smooth = km
        k = k - km

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(2), k_int8.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_per_block_int8_kernel[grid](
        q, q_int8, q_scale, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1),
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim, BLK=BLKQ
    )

    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_per_block_int8_kernel[grid](
        k, k_int8, k_scale, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1),
        sm_scale=1.0,
        C=head_dim, BLK=BLKK
    )

    return q_int8, q_scale, k_int8, k_scale, k_smooth

def _get_config():
    return {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 64,
        "num_warps": 4,
        "num_stages": 1,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 32,
        "cache_modifier": ".ca"
    }

def attn_qk_int8_per_block(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor = None,
    k_scale: torch.Tensor = None,
    tensor_layout: str = "HND",
    attn_mask: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = torch.float16,
    return_lse: bool = False,
    config: Optional[dict] = None,
):
    """
    Computes attention with INT8 quantized Q and K matrices using per-block scales.

    Args:
        q (torch.Tensor): INT8 query tensor with shape (B, H, qo_len, head_dim) for HND layout
            or (B, qo_len, H, head_dim) for NHD layout.
        k (torch.Tensor): INT8 key tensor with shape (B, H, kv_len, head_dim) for HND layout
            or (B, kv_len, H, head_dim) for NHD layout.
        v (torch.Tensor): FP16 value tensor with shape (B, H, kv_len, head_dim) for HND layout
            or (B, kv_len, H, head_dim) for NHD layout.
        q_scale (torch.Tensor): Per-block scale for Q with shape (B, H, qo_len // BLOCK_M, 1).
        k_scale (torch.Tensor): Per-block scale for K with shape (B, H, kv_len // BLOCK_N, 1).
        tensor_layout (str): Tensor layout, either "HND" or "NHD". Default: "HND".
        attn_mask (Optional[torch.Tensor]): Optional attention mask with shape (B, H, qo_len, kv_len).
        output_dtype (Optional[torch.dtype]): Output datatype (FP16 or BF16). Default: torch.float16.
        return_lse (bool): Whether to return log-sum-exp values. Default: False.
        config (Optional[dict]): Kernel tuning parameters (BLOCK_M, BLOCK_N, num_warps, etc.).

    Returns:
        tuple: (output tensor, lse tensor) if return_lse=True, else (output tensor, empty tensor).
    """
    
    assert (q_scale is None) == (k_scale is None), "Both q_scale and k_scale must be provided or both must be None"
    scales_provided = (q_scale is not None) and (k_scale is not None)
    if not scales_provided:
        if config is None:
            config = _get_config()
       
        sm_scale = q.shape[-1] ** -0.5
        q, q_scale, k, k_scale, k_smooth = per_block_int8(
            q, k, BLKQ=config["BLOCK_SIZE_M"], BLKK=config["BLOCK_SIZE_N"], sm_scale=sm_scale, tensor_layout=tensor_layout, smooth_k=True
        )
        # Note: k_smooth factors are applied during quantization to reduce outliers (SageAttention approach)
        # naive torch quantization
        # int8_dtype = torch.int8
        # q, q_scale = int8_per_block_quantize_bshd(q, int8_dtype, block_size=config["BLOCK_SIZE_M"], include_sqrt_scale=True)
        # k, k_scale = int8_per_block_quantize_bshd(k, int8_dtype, block_size=config["BLOCK_SIZE_N"])
    else:
        assert config is not None, "config must be provided because the scales are provided and they are assumed to have been calculated per-block."
    
    _LOGGER.info(
        f"ATTN_QK_INT8_PER_BLOCK: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} "
        f"q_scale={tuple(q_scale.shape)} k_scale={tuple(k_scale.shape)} layout={tensor_layout}"
    )

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    if attn_mask is not None:
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask = (
            attn_mask.stride(0),
            attn_mask.stride(1),
            attn_mask.stride(2),
            attn_mask.stride(3),
        )
    else:
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask = 0, 0, 0, 0

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    if return_lse:
        lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty([0], dtype=torch.float32, device="cpu")

    grid = (triton.cdiv(qo_len, config["BLOCK_SIZE_M"]) * h_qo * b, )
    _attn_fwd_qk_int8_per_block[grid](
        q,
        k,
        v,
        q_scale,
        k_scale,
        o,
        attn_mask,
        lse,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_v,
        stride_h_v,
        stride_seq_v,
        stride_bz_o,
        stride_h_o,
        stride_seq_o,
        stride_bz_mask,
        stride_h_mask,
        stride_m_mask,
        stride_n_mask,
        qo_len,
        kv_len,
        h_qo,
        num_kv_groups,
        HEAD_DIM=HEAD_DIM_K,
        RETURN_LSE=return_lse,
        BATCH=b,
        **config
    )

    if return_lse:
        return o, lse
    else:
        return o

