# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.rope_norm_store_kv import (
    _rope_norm_store_kv_compute_pos_slot_kernel,
    _rope_norm_store_kv_kernel,
    _rope_norm_store_kv_zero_trailing_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def _pick_block_t(num_rows: int, device: torch.device) -> int:
    """Mirror `infer_rope_cache_triton_block_t`: scale tile with row count and CUs."""
    device_id = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    sm_count = max(
        int(torch.cuda.get_device_properties(device_id).multi_processor_count),
        1,
    )
    block_t = triton.next_power_of_2(triton.cdiv(max(num_rows, 1), 2 * sm_count))
    return max(1, min(int(block_t), 32))


def _precompute_positions_slots(
    num_rows: int,
    num_req: int,
    block_size: int,
    q_index: torch.Tensor,
    num_seqlen_per_req: torch.Tensor,
    kvcache_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = q_index.device
    positions = torch.empty(num_rows, dtype=torch.int32, device=device)
    slot_indices = torch.empty(num_rows, dtype=torch.int64, device=device)

    BLOCK_R = 32
    _rope_norm_store_kv_compute_pos_slot_kernel[(num_req,)](
        q_index_ptr=q_index,
        num_seqlen_per_req_ptr=num_seqlen_per_req,
        kvcache_indices_ptr=kvcache_indices,
        positions_ptr=positions,
        slot_indices_ptr=slot_indices,
        stride_kvi_r=kvcache_indices.stride(0),
        stride_kvi_b=kvcache_indices.stride(1),
        BLOCK_R=BLOCK_R,
        BLOCK_SIZE=block_size,
    )
    return positions, slot_indices


def rope_norm_store_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    qkv: torch.Tensor,
    cos_sin: torch.Tensor,
    num_seqlen_per_req: torch.Tensor,
    q_index: torch.Tensor,
    kvcache_indices: torch.Tensor,
    is_prefill: bool,
    q_norm_weight: Optional[torch.Tensor] = None,
    k_norm_weight: Optional[torch.Tensor] = None,
    out_q: Optional[torch.Tensor] = None,
    out_k: Optional[torch.Tensor] = None,
    out_v: Optional[torch.Tensor] = None,
    qk_norm_policy: int = 0,
) -> torch.Tensor:
    """Triton implementation of ``torch.ops.hpc.rope_norm_store_kv``.

    Fuses NeoX RoPE on Q/K, optional RMSNorm on Q/K (order controlled by
    ``qk_norm_policy``), and a paged BF16 KV-cache write into a single launch
    (plus tiny helpers for per-row position/slot computation and, in prefill,
    trailing-slot zeroing).

    See ``rope_norm_store_kv_api.py`` for argument shapes and dtypes.
    """
    if qk_norm_policy not in (0, 1, 2):
        raise ValueError(
            f"qk_norm_policy must be 0/1/2 (got {qk_norm_policy})."
        )
    if qk_norm_policy != 0:
        if q_norm_weight is None or k_norm_weight is None:
            raise ValueError(
                "q_norm_weight and k_norm_weight are required when "
                "qk_norm_policy != 0."
            )

    num_blocks, block_size, num_kv_heads, qk_head_dim = key_cache.shape
    v_blocks, v_block_size, v_kv_heads, v_head_dim = value_cache.shape
    if (num_blocks, block_size, num_kv_heads) != (v_blocks, v_block_size, v_kv_heads):
        raise ValueError(
            "key_cache and value_cache must share num_blocks/block_size/num_kv_heads "
            f"(got {tuple(key_cache.shape)} vs {tuple(value_cache.shape)})."
        )
    if qk_head_dim % 2 != 0:
        raise ValueError(f"qk_head_dim must be even (got {qk_head_dim}).")
    if cos_sin.shape[-1] != qk_head_dim:
        raise ValueError(
            "cos_sin last dim must equal qk_head_dim "
            f"(got {cos_sin.shape[-1]} vs {qk_head_dim})."
        )
    qk_head_dim_pow2 = triton.next_power_of_2(qk_head_dim)
    if qk_head_dim_pow2 != qk_head_dim:
        raise ValueError(
            f"qk_head_dim must be a power of two (got {qk_head_dim}). "
            "Non-pow2 qk_head_dim would require padding the NeoX rotation "
            "helper, which is unsupported."
        )

    num_rows, hidden = qkv.shape
    q_dim = hidden - num_kv_heads * qk_head_dim - num_kv_heads * v_head_dim
    if q_dim <= 0 or q_dim % qk_head_dim != 0:
        raise ValueError(
            "qkv hidden does not decompose as num_q_heads*qk_head_dim + "
            "num_kv_heads*qk_head_dim + num_kv_heads*v_head_dim "
            f"(hidden={hidden}, qk_head_dim={qk_head_dim}, v_head_dim={v_head_dim}, "
            f"num_kv_heads={num_kv_heads})."
        )
    num_q_heads = q_dim // qk_head_dim
    if num_q_heads < num_kv_heads:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be >= num_kv_heads ({num_kv_heads})."
        )

    num_req = num_seqlen_per_req.shape[0]
    if q_index.shape[0] != num_req + 1:
        raise ValueError(
            f"q_index length must be num_req+1 (got {q_index.shape[0]} vs "
            f"num_req+1={num_req + 1})."
        )

    _LOGGER.info(
        f"ROPE_NORM_STORE_KV: qkv={tuple(qkv.shape)} num_req={num_req} "
        f"qh={num_q_heads} kvh={num_kv_heads} qk_d={qk_head_dim} v_d={v_head_dim} "
        f"block_size={block_size} policy={qk_norm_policy} prefill={is_prefill}"
    )

    if out_q is None:
        out_q = torch.empty(
            (num_rows, num_q_heads, qk_head_dim),
            dtype=qkv.dtype,
            device=qkv.device,
        )

    write_k_to_cache = out_k is None
    write_v_to_cache = out_v is None

    # ---- Precompute per-row positions and absolute slot indices ----
    positions, slot_indices = _precompute_positions_slots(
        num_rows, num_req, block_size,
        q_index, num_seqlen_per_req, kvcache_indices,
    )

    # ---- Main kernel ----
    BLOCK_T = _pick_block_t(num_rows, qkv.device)
    grid = (triton.cdiv(num_rows, BLOCK_T), num_q_heads)

    out_k_for_kernel = out_k if out_k is not None else qkv  # any valid ptr; masked off
    out_v_for_kernel = out_v if out_v is not None else qkv

    v_head_dim_pad = triton.next_power_of_2(v_head_dim)

    _rope_norm_store_kv_kernel[grid](
        qkv_ptr=qkv,
        cos_sin_ptr=cos_sin,
        positions_ptr=positions,
        slot_indices_ptr=slot_indices,
        q_norm_weight_ptr=q_norm_weight if q_norm_weight is not None else qkv,
        k_norm_weight_ptr=k_norm_weight if k_norm_weight is not None else qkv,
        out_q_ptr=out_q,
        out_k_ptr=out_k_for_kernel,
        out_v_ptr=out_v_for_kernel,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        eps=1e-6,
        num_rows=num_rows,
        total_num_kv_cache_tokens=num_blocks * block_size,
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
        stride_kc_t=key_cache.stride(1),
        stride_kc_h=key_cache.stride(2),
        stride_kc_d=key_cache.stride(3),
        stride_vc_b=value_cache.stride(0),
        stride_vc_t=value_cache.stride(1),
        stride_vc_h=value_cache.stride(2),
        stride_vc_d=value_cache.stride(3),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        QK_HEAD_DIM=qk_head_dim,
        QK_HEAD_DIM_HALF=qk_head_dim // 2,
        V_HEAD_DIM=v_head_dim,
        V_HEAD_DIM_PAD=v_head_dim_pad,
        BLOCK_SIZE=block_size,
        BLOCK_T=BLOCK_T,
        QK_NORM_POLICY=qk_norm_policy,
        APPLY_Q_NORM=(qk_norm_policy != 0 and q_norm_weight is not None),
        APPLY_K_NORM=(qk_norm_policy != 0 and k_norm_weight is not None),
        WRITE_K_TO_CACHE=write_k_to_cache,
        WRITE_V_TO_CACHE=write_v_to_cache,
    )

    # ---- Trailing slots zeroing (prefill only) ----
    if is_prefill and (write_k_to_cache or write_v_to_cache):
        block_size_pad = triton.next_power_of_2(block_size)
        qk_head_dim_pad = triton.next_power_of_2(qk_head_dim)
        _rope_norm_store_kv_zero_trailing_kernel[(num_req, num_kv_heads)](
            num_seqlen_per_req_ptr=num_seqlen_per_req,
            kvcache_indices_ptr=kvcache_indices,
            key_cache_ptr=key_cache,
            value_cache_ptr=value_cache,
            stride_kvi_r=kvcache_indices.stride(0),
            stride_kvi_b=kvcache_indices.stride(1),
            stride_kc_b=key_cache.stride(0),
            stride_kc_t=key_cache.stride(1),
            stride_kc_h=key_cache.stride(2),
            stride_kc_d=key_cache.stride(3),
            stride_vc_b=value_cache.stride(0),
            stride_vc_t=value_cache.stride(1),
            stride_vc_h=value_cache.stride(2),
            stride_vc_d=value_cache.stride(3),
            BLOCK_SIZE=block_size,
            BLOCK_SIZE_PAD=block_size_pad,
            QK_HEAD_DIM=qk_head_dim,
            QK_HEAD_DIM_PAD=qk_head_dim_pad,
            V_HEAD_DIM=v_head_dim,
            V_HEAD_DIM_PAD=v_head_dim_pad,
        )

    return out_q
