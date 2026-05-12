# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Wrapper for ``_fused_rope_norm_store_kv_fp8_kernel``.

Fuses (RoPE + optional RMSNorm) on Q & K, dynamic per-token-per-head FP8
quant on Q & K, static per-head FP8 quant on V, paged-KV-cache scatter for
K and V, and the K-scale slab write. Mirrors the PyTorch reference at
``rope_norm_store_fp8.py``.
"""

from typing import Optional, Tuple

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_rope_norm_store_kv_fp8 import (
    _fused_rope_norm_store_kv_fp8_kernel,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def _build_token_meta(
    num_seqlen_per_req: torch.Tensor,
    q_index: torch.Tensor,
    num_rows: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (token_to_req[int32], token_kv_pos[int32], within_req_idx[int32]).

    For each new token, the kernel needs (a) which request it belongs to,
    (b) its global position in that request's sequence (= cache offset = cos/sin
    lookup row). The within_req index is returned for the prefill q_scale
    scatter in the wrapper.
    """
    q_index_i64 = q_index.to(torch.int64)
    n_new = q_index_i64[1:] - q_index_i64[:-1]                     # [num_req]
    num_req = n_new.numel()

    token_to_req = torch.repeat_interleave(
        torch.arange(num_req, device=device, dtype=torch.int32), n_new
    )                                                              # [num_rows]
    base = (num_seqlen_per_req.to(torch.int64) - n_new).to(torch.int32)
    base_per_token = torch.repeat_interleave(base, n_new)          # [num_rows]
    starts_per_token = torch.repeat_interleave(
        q_index_i64[:-1].to(torch.int32), n_new
    )
    arange_rows = torch.arange(num_rows, device=device, dtype=torch.int32)
    within_req = arange_rows - starts_per_token                    # [num_rows]
    token_kv_pos = base_per_token + within_req                     # [num_rows]
    return token_to_req, token_kv_pos, within_req


def rope_norm_store_kv_fp8(
    qkv: torch.Tensor,
    cos_sin: torch.Tensor,
    num_seqlen_per_req: torch.Tensor,
    q_index: torch.Tensor,
    kvcache_indices: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    is_prefill: bool,
    max_seqlens: int,
    q_norm_weight: Optional[torch.Tensor] = None,
    k_norm_weight: Optional[torch.Tensor] = None,
    qk_norm_policy: int = 1,
    is_neox: bool = True,
    rms_eps: float = 1e-6,
    out_q: Optional[torch.Tensor] = None,
    out_q_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused RoPE + Norm + FP8-quant + paged-KV-cache store.

    Args:
        qkv:                ``[num_rows, QH*D + 2*KH*D]`` bf16 (Q‖K‖V packed).
        cos_sin:            ``[max_seq_len, D]`` fp32, ``[:, :D/2]``=cos, ``[:, D/2:]``=sin.
        num_seqlen_per_req: ``[num_req]`` int (total seqlen incl. new tokens).
        q_index:            ``[num_req+1]`` int (CSR offsets of new tokens, qo_indptr).
        kvcache_indices:    ``[num_req, max_blocks]`` int (block table).
        key_cache:          ``[num_blocks, BLOCK_SIZE, KH, D]`` fp8 (written in place).
        value_cache:        ``[num_blocks, BLOCK_SIZE, KH, D]`` fp8 (written in place).
        k_scale:            ``[num_blocks, SCALE_ROWS, KH, SCALE_COLS]`` fp32 (written in place).
                            ``SCALE_ROWS * SCALE_COLS == BLOCK_SIZE``.
        v_scale:            ``[KH]`` fp32 (read).
        is_prefill:         If True, q_scale is reshaped to per-request layout.
        max_seqlens:        Maximum new-token count per request; controls prefill q_scale pad.
        q_norm_weight / k_norm_weight: ``[D]`` fp32, optional.
        qk_norm_policy:     0 = no norm, 1 = RoPE→Norm, 2 = Norm→RoPE.
        is_neox:            NeoX-style RoPE (default) vs GPT-J interleaved.

    Returns:
        out_q_fp8:    ``[num_rows, QH, D]`` fp8.
        q_scale:
            prefill: ``[num_req, QH, ceil(max_seqlens/128)*128]`` fp32 (ones-padded).
            decode:  ``[num_rows, QH]`` fp32.
        split_k_flag: ``[num_req, KH]`` int32 (zero).
    """
    assert qkv.is_cuda, "qkv must be on CUDA"
    assert qkv.is_contiguous(), "qkv must be contiguous"
    assert qkv.dim() == 2
    assert key_cache.dim() == 4 and value_cache.dim() == 4
    assert key_cache.shape == value_cache.shape

    num_rows, hidden = qkv.shape
    num_blocks, block_size, kh, head_dim = key_cache.shape

    qh = (hidden - 2 * kh * head_dim) // head_dim
    assert (
        qh * head_dim + 2 * kh * head_dim == hidden
    ), f"hidden={hidden} not consistent with QH={qh} KH={kh} D={head_dim}"
    assert head_dim == triton.next_power_of_2(head_dim), "head_dim must be power of 2"
    assert head_dim >= 2 and (head_dim % 2 == 0)
    assert qk_norm_policy in (0, 1, 2)

    # FP8 dtype + dtype max
    fp8_dtype = out_q_dtype if out_q_dtype is not None else key_cache.dtype
    if not fp8_dtype.is_floating_point or fp8_dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ):
        fp8_dtype = get_fp8_e4m3_dtype()
    dtype_max = float(torch.finfo(fp8_dtype).max)

    # Validate cache dtype
    assert key_cache.dtype == fp8_dtype, (
        f"key_cache dtype {key_cache.dtype} != expected fp8 dtype {fp8_dtype}"
    )
    assert value_cache.dtype == fp8_dtype, (
        f"value_cache dtype {value_cache.dtype} != expected fp8 dtype {fp8_dtype}"
    )

    # K-scale slab geometry
    assert k_scale.dim() == 4
    num_blocks_ks, scale_rows, kh_ks, scale_cols = k_scale.shape
    assert num_blocks_ks == num_blocks and kh_ks == kh, (
        f"k_scale {tuple(k_scale.shape)} inconsistent with key_cache "
        f"{tuple(key_cache.shape)}"
    )
    assert scale_rows * scale_cols == block_size, (
        f"k_scale rows*cols ({scale_rows}*{scale_cols}) must equal "
        f"BLOCK_SIZE ({block_size})"
    )
    assert k_scale.dtype == torch.float32 and v_scale.dtype == torch.float32
    assert v_scale.numel() == kh

    if q_norm_weight is not None:
        assert q_norm_weight.numel() == head_dim
        assert q_norm_weight.dtype == torch.float32
    if k_norm_weight is not None:
        assert k_norm_weight.numel() == head_dim
        assert k_norm_weight.dtype == torch.float32

    # cos_sin laid out as [..., :D/2]=cos, [..., D/2:]=sin
    assert cos_sin.dim() == 2 and cos_sin.shape[1] == head_dim, (
        f"cos_sin must be [max_seq_len, head_dim], got {tuple(cos_sin.shape)}"
    )
    assert cos_sin.dtype == torch.float32, "cos_sin must be float32"

    # kvcache_indices
    assert kvcache_indices.dim() == 2
    num_req = num_seqlen_per_req.numel()
    assert kvcache_indices.shape[0] == num_req
    assert q_index.numel() == num_req + 1

    # Outputs
    if out_q is None:
        out_q = torch.empty(
            (num_rows, qh, head_dim), dtype=fp8_dtype, device=qkv.device
        )
    else:
        assert out_q.shape == (num_rows, qh, head_dim) and out_q.dtype == fp8_dtype

    q_scale_flat = torch.empty(
        (num_rows, qh), dtype=torch.float32, device=qkv.device
    )

    # Token → (req, kv_pos, within_req) lookup tables
    token_to_req, token_kv_pos, within_req = _build_token_meta(
        num_seqlen_per_req, q_index, num_rows, qkv.device
    )

    # cos_sin / kvcache_indices must be contiguous on inner dim
    if not cos_sin.is_contiguous():
        cos_sin = cos_sin.contiguous()
    if not kvcache_indices.is_contiguous():
        kvcache_indices = kvcache_indices.contiguous()

    _LOGGER.info(
        "ROPE_NORM_STORE_KV_FP8: "
        f"rows={num_rows} QH={qh} KH={kh} D={head_dim} "
        f"BLOCK={block_size} norm_policy={qk_norm_policy} "
        f"is_neox={is_neox} is_prefill={is_prefill}"
    )

    num_q_pids = num_rows * qh
    num_k_pids = num_rows * kh
    n_pid = num_q_pids + 2 * num_k_pids
    grid = (n_pid,)

    _fused_rope_norm_store_kv_fp8_kernel[grid](
        qkv,
        out_q,
        q_scale_flat,
        key_cache,
        value_cache,
        k_scale,
        v_scale,
        q_norm_weight,
        k_norm_weight,
        cos_sin,
        token_to_req,
        token_kv_pos,
        kvcache_indices,
        qkv.stride(0),
        out_q.stride(0), out_q.stride(1), out_q.stride(2),
        q_scale_flat.stride(0), q_scale_flat.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), k_scale.stride(3),
        cos_sin.stride(0), cos_sin.stride(1),
        kvcache_indices.stride(0),
        QH=qh,
        KH=kh,
        BLOCK_D=head_dim,
        BLOCK_D_HALF=head_dim // 2,
        BLOCK_SIZE=block_size,
        SCALE_COLS=scale_cols,
        DTYPE_MAX=dtype_max,
        IS_NEOX=is_neox,
        QK_NORM_POLICY=qk_norm_policy,
        HAS_Q_NORM_WEIGHT=q_norm_weight is not None,
        HAS_K_NORM_WEIGHT=k_norm_weight is not None,
        RMS_EPS=rms_eps,
        NUM_Q_PIDS=num_q_pids,
        NUM_K_PIDS=num_k_pids,
        Q_HIDDEN_OFFSET=0,
        K_HIDDEN_OFFSET=qh * head_dim,
        V_HIDDEN_OFFSET=(qh + kh) * head_dim,
        num_warps=1,
    )

    # Reformat q_scale and zero split_k_flag in the wrapper (cheap, host-side
    # for the small prefix/scatter and a single zero_-style fill).
    if is_prefill:
        max_seqlens_pad128 = ((int(max_seqlens) + 127) // 128) * 128
        q_scale_out = torch.ones(
            (num_req, qh, max_seqlens_pad128),
            dtype=torch.float32,
            device=qkv.device,
        )
        # Vectorized scatter: q_scale_out[req[t], :, within_req[t]] = q_scale_flat[t, :]
        q_scale_out[
            token_to_req.long(), :, within_req.long()
        ] = q_scale_flat
    else:
        q_scale_out = q_scale_flat

    split_k_flag = torch.zeros(
        (num_req, kh), dtype=torch.int32, device=qkv.device
    )
    return out_q, q_scale_out, split_k_flag
