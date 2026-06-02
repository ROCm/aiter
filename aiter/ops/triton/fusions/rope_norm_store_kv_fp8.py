# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math
from typing import Optional, Tuple

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.rope_norm_store_kv_fp8 import (
    _rope_norm_store_kv_fp8_compute_pos_slot_kernel,
    _rope_norm_store_kv_fp8_kernel,
    _rope_norm_store_kv_fp8_zero_trailing_kernel,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# RMSNorm epsilon (matches BF16 op).
_RMS_NORM_EPS = 1e-5

_HADAMARD_CACHE = {}


def _get_hadamard(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (n, str(device), dtype)
    if key not in _HADAMARD_CACHE:
        H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
        while H.shape[0] < n:
            H = torch.cat(
                [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
            )
        H = H / math.sqrt(n)
        _HADAMARD_CACHE[key] = H.to(dtype).contiguous()
    return _HADAMARD_CACHE[key]


def _precompute_positions_slots(
    num_rows: int,
    num_req: int,
    block_size: int,
    q_index: torch.Tensor,
    num_seqlen_per_req: torch.Tensor,
    kvcache_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = q_index.device
    positions = torch.zeros(num_rows, dtype=torch.int32, device=device)
    slot_indices = torch.full((num_rows,), -1, dtype=torch.int64, device=device)
    req_ids = torch.zeros(num_rows, dtype=torch.int32, device=device)
    local_idx = torch.zeros(num_rows, dtype=torch.int32, device=device)

    BLOCK_R = 32
    _rope_norm_store_kv_fp8_compute_pos_slot_kernel[(num_req,)](
        q_index_ptr=q_index,
        num_seqlen_per_req_ptr=num_seqlen_per_req,
        kvcache_indices_ptr=kvcache_indices,
        positions_ptr=positions,
        slot_indices_ptr=slot_indices,
        req_ids_ptr=req_ids,
        local_idx_ptr=local_idx,
        stride_kvi_r=kvcache_indices.stride(0),
        stride_kvi_b=kvcache_indices.stride(1),
        BLOCK_R=BLOCK_R,
        BLOCK_SIZE=block_size,
    )
    return positions, slot_indices, req_ids, local_idx


def rope_norm_store_kv_fp8(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    qkv: torch.Tensor,
    cos_sin: torch.Tensor,
    num_seqlen_per_req: torch.Tensor,
    q_index: torch.Tensor,
    kvcache_indices: torch.Tensor,
    is_prefill: bool,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    quant_policy=0,
    max_seqlens: int = 0,
    upper_max: Optional[float] = None,
    q_scale_inv: Optional[torch.Tensor] = None,
    q_norm_weight: Optional[torch.Tensor] = None,
    k_norm_weight: Optional[torch.Tensor] = None,
    out_q: Optional[torch.Tensor] = None,
    out_k: Optional[torch.Tensor] = None,
    out_v: Optional[torch.Tensor] = None,
    qk_norm_policy: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Triton implementation of ``torch.ops.hpc.rope_norm_store_kv_fp8``.

    Returns ``(out_q_fp8, q_scale, split_k_flag)``.

    Cache layouts (vLLM-style):
      - ``key_cache``:   ``[num_blocks, num_kv_heads, qk_head_dim // X, block_size, X]``
        with ``X = key_cache.shape[-1]`` (typically ``16 // sizeof(dtype) = 16`` for fp8).
      - ``value_cache``: ``[num_blocks, num_kv_heads, v_head_dim, block_size]``.

    RMSNorm uses ``eps = 1e-5``.

    Note: ``block_size`` may be any positive integer for policies 1/2 (static K/V
    scales). For policies 0/3 (dynamic K scale), the API-prescribed ``k_scale``
    shape ``[num_blocks, R, num_kv_heads, L]`` with ``L = qk_head_dim // 4`` and
    ``R = block_size // L`` requires ``block_size`` to be a multiple of L (so
    ``block_size=16`` only works there for ``qk_head_dim <= 64``).
    """
    qp = int(getattr(quant_policy, "value", quant_policy))
    if qp not in (0, 1, 2, 3):
        raise ValueError(f"quant_policy must be 0/1/2/3 (got {qp})")
    if qk_norm_policy not in (0, 1, 2):
        raise ValueError(f"qk_norm_policy must be 0/1/2 (got {qk_norm_policy})")

    q_quant_dynamic = qp in (0, 1, 3)
    k_quant_dynamic = qp in (0, 3)
    v_quant_perhead = qp in (0, 3)
    apply_hadamard = (qp == 3)

    if qp == 2 and q_scale_inv is None:
        raise ValueError("q_scale_inv is required for quant_policy=2")
    if qk_norm_policy != 0 and (q_norm_weight is None or k_norm_weight is None):
        raise ValueError(
            "q_norm_weight and k_norm_weight are required when qk_norm_policy != 0"
        )

    fp8_dtype = get_fp8_e4m3_dtype()
    fp8_max = torch.finfo(fp8_dtype).max if upper_max is None else float(upper_max)
    if key_cache.dtype != fp8_dtype or value_cache.dtype != fp8_dtype:
        raise ValueError(
            f"key_cache/value_cache must be {fp8_dtype} for the FP8 path "
            f"(got {key_cache.dtype}, {value_cache.dtype})"
        )

    # ----- key_cache: 5-D [B, H, D/X, S, X] -----
    if key_cache.ndim != 5:
        raise ValueError(
            "key_cache must be 5-D [num_blocks, num_kv_heads, qk_head_dim/X, "
            f"block_size, X] (got shape {tuple(key_cache.shape)})"
        )
    num_blocks, num_kv_heads, qk_chunks, block_size, X = key_cache.shape
    qk_head_dim = qk_chunks * X
    if qk_head_dim % X != 0:
        raise ValueError(f"qk_head_dim ({qk_head_dim}) must be divisible by X ({X})")

    # ----- value_cache: 4-D [B, H, D, S] -----
    if value_cache.ndim != 4:
        raise ValueError(
            "value_cache must be 4-D [num_blocks, num_kv_heads, v_head_dim, "
            f"block_size] (got shape {tuple(value_cache.shape)})"
        )
    v_blocks, v_kv_heads, v_head_dim, v_block_size = value_cache.shape
    if (v_blocks, v_kv_heads, v_block_size) != (num_blocks, num_kv_heads, block_size):
        raise ValueError(
            "key_cache and value_cache must share num_blocks/num_kv_heads/block_size "
            f"(got K={tuple(key_cache.shape)} vs V={tuple(value_cache.shape)})"
        )

    if qk_head_dim % 2 != 0:
        raise ValueError(f"qk_head_dim must be even (got {qk_head_dim})")
    if triton.next_power_of_2(qk_head_dim) != qk_head_dim:
        raise ValueError(f"qk_head_dim must be a power of two (got {qk_head_dim})")
    if apply_hadamard and qk_head_dim != 128:
        raise ValueError(
            f"quant_policy=3 (Hadamard) only supports qk_head_dim == 128 (got {qk_head_dim})"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0 (got {block_size})")

    num_rows, hidden = qkv.shape
    q_dim_total = hidden - num_kv_heads * qk_head_dim - num_kv_heads * v_head_dim
    if q_dim_total <= 0 or q_dim_total % qk_head_dim != 0:
        raise ValueError(
            "qkv hidden does not decompose as num_q_heads*qk_head_dim + "
            "num_kv_heads*qk_head_dim + num_kv_heads*v_head_dim"
        )
    num_q_heads = q_dim_total // qk_head_dim
    if num_q_heads < num_kv_heads:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be >= num_kv_heads ({num_kv_heads})"
        )

    num_req = num_seqlen_per_req.shape[0]

    K_SCALE_L = qk_head_dim // 4
    if k_quant_dynamic:
        if block_size % K_SCALE_L != 0:
            raise ValueError(
                f"block_size ({block_size}) must be divisible by L=qk_head_dim/4 "
                f"({K_SCALE_L}) for dynamic K scaling (policy {qp}). Use a larger "
                f"block_size or a smaller qk_head_dim."
            )
        K_SCALE_R = block_size // K_SCALE_L
        expected_ks_shape = (num_blocks, K_SCALE_R, num_kv_heads, K_SCALE_L)
        if tuple(k_scale.shape) != expected_ks_shape:
            raise ValueError(
                f"k_scale shape must be {expected_ks_shape} for dynamic K "
                f"(got {tuple(k_scale.shape)})"
            )
    else:
        if k_scale.numel() != 1:
            raise ValueError(
                f"k_scale must be a single-element tensor for static K (got {tuple(k_scale.shape)})"
            )

    if v_quant_perhead:
        if v_scale.shape != (num_kv_heads,):
            raise ValueError(
                f"v_scale must be [num_kv_heads]=({num_kv_heads},) for dynamic V "
                f"(got {tuple(v_scale.shape)})"
            )
    else:
        if v_scale.numel() != 1:
            raise ValueError(
                f"v_scale must be a single-element tensor for static V (got {tuple(v_scale.shape)})"
            )

    _LOGGER.info(
        f"ROPE_NORM_STORE_KV_FP8: qkv={tuple(qkv.shape)} num_req={num_req} "
        f"qh={num_q_heads} kvh={num_kv_heads} qk_d={qk_head_dim} v_d={v_head_dim} "
        f"block_size={block_size} X={X} qk_policy={qk_norm_policy} qp={qp} "
        f"prefill={is_prefill} max_seqlens={max_seqlens}"
    )

    if out_q is None:
        out_q = torch.empty(
            (num_rows, num_q_heads, qk_head_dim),
            dtype=fp8_dtype, device=qkv.device,
        )

    if out_k is not None and out_k.dtype != fp8_dtype:
        raise ValueError(f"out_k must be {fp8_dtype}")
    if out_v is not None and out_v.dtype != fp8_dtype:
        raise ValueError(f"out_v must be {fp8_dtype}")
    if k_quant_dynamic and out_k is not None:
        raise ValueError(
            "Dynamic K scaling (policy 0/3) requires out_k=None so K is written "
            "to key_cache with paged-layout scales."
        )

    if q_quant_dynamic:
        if is_prefill:
            if max_seqlens <= 0:
                raise ValueError(
                    "max_seqlens > 0 is required in prefill for dynamic Q "
                    "quantization (cudagraph-safe path)"
                )
            pad128 = ((max_seqlens + 127) // 128) * 128
            q_scale_out = torch.empty(
                (num_req, num_q_heads, pad128),
                dtype=torch.float32, device=qkv.device,
            )
        else:
            q_scale_out = torch.empty(
                (num_rows, num_q_heads),
                dtype=torch.float32, device=qkv.device,
            )
    else:
        q_scale_out = None

    split_k_flag = torch.zeros(
        (num_req, num_kv_heads), dtype=torch.int32, device=qkv.device,
    )

    write_k_to_cache = out_k is None
    write_v_to_cache = out_v is None

    positions, slot_indices, req_ids, local_idx = _precompute_positions_slots(
        num_rows, num_req, block_size,
        q_index, num_seqlen_per_req, kvcache_indices,
    )

    if apply_hadamard:
        H = _get_hadamard(qk_head_dim, qkv.device, qkv.dtype)
    else:
        H = qkv

    BLOCK_T = 16
    grid = (triton.cdiv(num_rows, BLOCK_T), num_q_heads)
    v_head_dim_pad = triton.next_power_of_2(v_head_dim)

    qs_for_kernel = q_scale_out if q_scale_out is not None else qkv
    qs_strides = q_scale_out.stride() if q_scale_out is not None else (0, 0, 0)
    if q_scale_out is not None and len(qs_strides) == 2:
        qs_strides = (qs_strides[0], qs_strides[1], 0)

    ks_strides = k_scale.stride() if k_quant_dynamic else (0, 0, 0, 0)
    if k_quant_dynamic and len(ks_strides) != 4:
        raise ValueError("dynamic k_scale must be 4-D")

    _rope_norm_store_kv_fp8_kernel[grid](
        qkv_ptr=qkv,
        cos_sin_ptr=cos_sin,
        positions_ptr=positions,
        slot_indices_ptr=slot_indices,
        req_ids_ptr=req_ids,
        local_idx_ptr=local_idx,
        q_norm_weight_ptr=q_norm_weight if q_norm_weight is not None else qkv,
        k_norm_weight_ptr=k_norm_weight if k_norm_weight is not None else qkv,
        hadamard_ptr=H,
        q_scale_inv_ptr=q_scale_inv if q_scale_inv is not None else qkv,
        k_scale_ptr=k_scale,
        v_scale_ptr=v_scale,
        q_scale_out_ptr=qs_for_kernel,
        out_q_ptr=out_q,
        out_k_ptr=out_k if out_k is not None else key_cache,
        out_v_ptr=out_v if out_v is not None else value_cache,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        eps=_RMS_NORM_EPS,
        num_rows=num_rows,
        total_num_kv_cache_tokens=num_blocks * block_size,
        fp8_max=fp8_max,
        stride_qkv_t=qkv.stride(0),
        stride_qkv_d=qkv.stride(1),
        stride_cos_t=cos_sin.stride(0),
        stride_cos_d=cos_sin.stride(1),
        stride_out_q_t=out_q.stride(0),
        stride_out_q_h=out_q.stride(1),
        stride_out_q_d=out_q.stride(2),
        stride_out_k_t=out_k.stride(0) if out_k is not None else 0,
        stride_out_k_h=out_k.stride(1) if out_k is not None else 0,
        stride_out_k_d=out_k.stride(2) if out_k is not None else 0,
        stride_out_v_t=out_v.stride(0) if out_v is not None else 0,
        stride_out_v_h=out_v.stride(1) if out_v is not None else 0,
        stride_out_v_d=out_v.stride(2) if out_v is not None else 0,
        stride_kc_b=key_cache.stride(0),
        stride_kc_h=key_cache.stride(1),
        stride_kc_chunk=key_cache.stride(2),
        stride_kc_slot=key_cache.stride(3),
        stride_kc_x=key_cache.stride(4),
        stride_vc_b=value_cache.stride(0),
        stride_vc_h=value_cache.stride(1),
        stride_vc_d=value_cache.stride(2),
        stride_vc_slot=value_cache.stride(3),
        stride_ks_b=ks_strides[0],
        stride_ks_r=ks_strides[1],
        stride_ks_h=ks_strides[2],
        stride_ks_l=ks_strides[3],
        stride_qs_0=qs_strides[0],
        stride_qs_1=qs_strides[1],
        stride_qs_2=qs_strides[2],
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        QK_HEAD_DIM=qk_head_dim,
        QK_HEAD_DIM_HALF=qk_head_dim // 2,
        V_HEAD_DIM=v_head_dim,
        V_HEAD_DIM_PAD=v_head_dim_pad,
        BLOCK_SIZE=block_size,
        BLOCK_T=BLOCK_T,
        X=X,
        QK_NORM_POLICY=qk_norm_policy,
        APPLY_Q_NORM=(qk_norm_policy != 0 and q_norm_weight is not None),
        APPLY_K_NORM=(qk_norm_policy != 0 and k_norm_weight is not None),
        Q_QUANT_DYNAMIC=q_quant_dynamic,
        K_QUANT_DYNAMIC=k_quant_dynamic,
        V_QUANT_PERHEAD=v_quant_perhead,
        APPLY_HADAMARD=apply_hadamard,
        IS_PREFILL=is_prefill,
        WRITE_K_TO_CACHE=write_k_to_cache,
        WRITE_V_TO_CACHE=write_v_to_cache,
        K_SCALE_L=K_SCALE_L,
    )

    if is_prefill and (write_k_to_cache or write_v_to_cache):
        block_size_pad = triton.next_power_of_2(block_size)
        qk_head_dim_pad = triton.next_power_of_2(qk_head_dim)
        _rope_norm_store_kv_fp8_zero_trailing_kernel[(num_req, num_kv_heads)](
            num_seqlen_per_req_ptr=num_seqlen_per_req,
            kvcache_indices_ptr=kvcache_indices,
            key_cache_ptr=key_cache,
            value_cache_ptr=value_cache,
            stride_kvi_r=kvcache_indices.stride(0),
            stride_kvi_b=kvcache_indices.stride(1),
            stride_kc_b=key_cache.stride(0),
            stride_kc_h=key_cache.stride(1),
            stride_kc_chunk=key_cache.stride(2),
            stride_kc_slot=key_cache.stride(3),
            stride_kc_x=key_cache.stride(4),
            stride_vc_b=value_cache.stride(0),
            stride_vc_h=value_cache.stride(1),
            stride_vc_d=value_cache.stride(2),
            stride_vc_slot=value_cache.stride(3),
            BLOCK_SIZE=block_size,
            BLOCK_SIZE_PAD=block_size_pad,
            QK_HEAD_DIM=qk_head_dim,
            QK_HEAD_DIM_PAD=qk_head_dim_pad,
            V_HEAD_DIM=v_head_dim,
            V_HEAD_DIM_PAD=v_head_dim_pad,
            X=X,
        )

    return out_q, q_scale_out, split_k_flag
