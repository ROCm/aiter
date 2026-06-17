# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""HIP drop-in for ``rope_norm_store_kv_fp8``.

``rope_norm_store_kv_fp8_hip`` has the exact same signature as the Triton
``aiter.ops.triton.fusions.rope_norm_store_kv_fp8.rope_norm_store_kv_fp8``, but
replaces the two Triton kernels (position/slot precompute +
``_rope_norm_store_kv_fp8_kernel``) with a single fused HIP kernel. The HIP
kernel computes position/slot on the fly, so no precompute pass is needed.

Supported config:
- quant_policy == 3 (dynamic Q/K + Hadamard, per-head static V)
- qk_norm_policy == 2 (RMSNorm before NeoX RoPE)
- is_prefill in {False, True}
- qk_head_dim == 128
- K/V written directly to cache (out_k/out_v are None)

The HIP fused kernel fully replaces the two Triton kernels
(``_precompute_positions_slots`` + ``_rope_norm_store_kv_fp8_kernel``). For
``is_prefill=True`` the kernel still produces the q_scale *values*; the wrapper
relays them out into the padded ``[num_req, num_q_heads, pad128]`` layout to
match the Triton main kernel, and the (unchanged) Triton trailing-zero kernel
clears each request's last-block tail.
"""

from functools import lru_cache
from typing import Optional, Tuple

import torch
import triton
from torch import Tensor

from ..jit.core import compile_ops
from aiter.ops.triton.fusions.rope_norm_store_kv_fp8 import _get_hadamard
from aiter.ops.triton._triton_kernels.fusions.rope_norm_store_kv_fp8 import (
    _rope_norm_store_kv_fp8_zero_trailing_kernel,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype


@compile_ops(
    "module_rope_norm_store_kv_fp8",
    fc_name="rope_norm_store_kv_fp8_fused_hip",
)
def rope_norm_store_kv_fp8_fused_hip(
    qkv: Tensor,
    cos_sin: Tensor,
    q_index: Tensor,
    num_seqlen_per_req: Tensor,
    kvcache_indices: Tensor,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    hadamard: Tensor,
    k_scale: Tensor,
    v_scale: Tensor,
    out_q: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    q_scale_out: Tensor,
    eps: float,
    fp8_max: float,
    assume_decode_one_token: bool,
    use_mfma: bool = False,
    tile_hpw: int = 1,
) -> None: ...


@lru_cache(maxsize=None)
def _num_cus(device_index: int) -> int:
    """Compute-unit (CU) count of the GPU; cached per device."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def select_tile_hpw(batch_size: int, num_q_heads: int) -> int:
    """Pick heads-per-warp from the batch size (the CUDA-graph bucket key).

    The crossover is derived from the GPU CU count instead of hardcoded batch
    thresholds: one warp covers (one row, hpw heads), so at hpw=1 the kernel
    launches ~batch_size*num_q_heads warps. While that already exceeds the GPU's
    CU count the device is saturated, so we raise hpw to add ILP per warp; small
    batches stay at hpw=1 for max occupancy. CUDA graphs bucket by batch size,
    so this stays constant within a bucket and never depends on context length.
    ``hpw`` must divide ``num_q_heads``; otherwise it is halved down (the C++
    launcher also clamps to 1).
    """
    cu = _num_cus(torch.cuda.current_device())
    # Warp groups normalized to a 32 q-head baseline, vs CU count.
    eff = batch_size * max(num_q_heads, 1) / 32.0
    if eff <= cu:
        hpw = 1
    elif eff <= cu * 2:
        hpw = 2
    elif eff <= cu * 4:
        hpw = 4
    else:
        hpw = 8
    while hpw > 1 and num_q_heads % hpw != 0:
        hpw //= 2
    return hpw


def _scatter_q_scale_prefill(
    q_scale_rows: torch.Tensor,
    q_index: torch.Tensor,
    num_req: int,
    num_q_heads: int,
    max_seqlens: int,
) -> torch.Tensor:
    """Relayout the HIP kernel's row-major q_scale ``[num_rows, num_q_heads]``
    into the Triton prefill layout ``[num_req, num_q_heads, pad128]``.

    This reproduces ``_rope_norm_store_kv_fp8_kernel``'s prefill store address
    ``q_scale_out[req_id, hq, local_idx]`` (q_scale *values* are identical to
    decode; only the placement differs). ``req_id``/``local_idx`` per row come
    from ``q_index`` (the cumulative row offsets), i.e. the same mapping
    ``_precompute_positions_slots`` computes.
    """
    if max_seqlens <= 0:
        raise ValueError(
            "max_seqlens > 0 is required in prefill for dynamic Q quantization"
        )
    device = q_scale_rows.device
    num_rows = q_scale_rows.shape[0]
    pad128 = ((int(max_seqlens) + 127) // 128) * 128
    qidx = q_index.to(torch.int64)
    counts = qidx[1:] - qidx[:-1]  # rows per request (0 for padding reqs)
    req_id_per_row = torch.repeat_interleave(
        torch.arange(num_req, device=device), counts
    )
    local_idx_per_row = torch.arange(num_rows, device=device) - qidx[:-1][req_id_per_row]
    q_scale_out = torch.empty(
        (num_req, num_q_heads, pad128), dtype=torch.float32, device=device
    )
    hh = torch.arange(num_q_heads, device=device)
    q_scale_out[req_id_per_row[:, None], hh[None, :], local_idx_per_row[:, None]] = (
        q_scale_rows
    )
    return q_scale_out


def _zero_trailing_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_seqlen_per_req: torch.Tensor,
    kvcache_indices: torch.Tensor,
    block_size: int,
    qk_head_dim: int,
    v_head_dim: int,
    k_cache_x: int,
) -> None:
    """Prefill trailing-zero, kept on the Triton kernel (NOT replaced by HIP):
    zeros the unused tail slots of each request's last KV block so attention
    reading a full block never sees stale data."""
    num_req = num_seqlen_per_req.shape[0]
    num_kv_heads = key_cache.shape[1]
    _rope_norm_store_kv_fp8_zero_trailing_kernel[(num_req, num_kv_heads)](
        num_seqlen_per_req_ptr=num_seqlen_per_req,
        kvcache_indices_ptr=kvcache_indices,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        stride_kvi_r=kvcache_indices.stride(0),
        stride_kvi_b=kvcache_indices.stride(1),
        stride_kc_b=key_cache.stride(0),
        stride_kc_h=key_cache.stride(1),
        stride_kc_g=key_cache.stride(2),
        stride_kc_t=key_cache.stride(3),
        stride_kc_x=key_cache.stride(4),
        stride_vc_b=value_cache.stride(0),
        stride_vc_h=value_cache.stride(1),
        stride_vc_d=value_cache.stride(2),
        stride_vc_t=value_cache.stride(3),
        BLOCK_SIZE=block_size,
        BLOCK_SIZE_PAD=triton.next_power_of_2(block_size),
        QK_HEAD_DIM=qk_head_dim,
        QK_HEAD_DIM_PAD=triton.next_power_of_2(qk_head_dim),
        V_HEAD_DIM=v_head_dim,
        V_HEAD_DIM_PAD=triton.next_power_of_2(v_head_dim),
        K_CACHE_X=k_cache_x,
    )


def rope_norm_store_kv_fp8_hip(
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
    """Same signature as the Triton ``rope_norm_store_kv_fp8``; one HIP kernel.

    Returns ``(out_q_fp8, q_scale, split_k_flag)``.
    """
    qp = int(getattr(quant_policy, "value", quant_policy))
    if qp != 3 or qk_norm_policy != 2:
        raise NotImplementedError(
            "rope_norm_store_kv_fp8_hip only supports quant_policy=3, qk_norm_policy=2"
        )
    if out_k is not None or out_v is not None:
        raise NotImplementedError("HIP path writes K/V directly to cache only")
    if q_scale_inv is not None:
        raise NotImplementedError("static-Q (quant_policy=2) is not supported")
    if q_norm_weight is None or k_norm_weight is None:
        raise ValueError("q_norm_weight and k_norm_weight are required")

    if key_cache.ndim != 5 or value_cache.ndim != 4:
        raise ValueError(
            "key_cache must be 5-D [N, H, D/x, B, x] and value_cache 4-D [N, H, D, B]"
        )
    num_blocks, num_kv_heads, qk_chunks, block_size, k_cache_x = key_cache.shape
    qk_head_dim = qk_chunks * k_cache_x
    if qk_head_dim != 128:
        raise NotImplementedError("HIP path supports qk_head_dim=128 only")
    v_head_dim = value_cache.shape[2]

    num_rows, hidden = qkv.shape
    q_dim_total = hidden - num_kv_heads * qk_head_dim - num_kv_heads * v_head_dim
    if q_dim_total <= 0 or q_dim_total % qk_head_dim != 0:
        raise ValueError("qkv hidden size is not compatible with cache/head dims")
    num_q_heads = q_dim_total // qk_head_dim
    num_req = num_seqlen_per_req.shape[0]

    fp8_dtype = get_fp8_e4m3_dtype()
    fp8_max = torch.finfo(fp8_dtype).max if upper_max is None else float(upper_max)

    # K/V dynamic-scale shape checks (paged [N, R, H, L] layout).
    K_SCALE_L = min(block_size, qk_head_dim // 4)
    expected_ks = (num_blocks, block_size // K_SCALE_L, num_kv_heads, K_SCALE_L)
    if tuple(k_scale.shape) != expected_ks:
        raise ValueError(f"k_scale must be {expected_ks}, got {tuple(k_scale.shape)}")
    if v_scale.shape != (num_kv_heads,):
        raise ValueError(f"v_scale must be ({num_kv_heads},), got {tuple(v_scale.shape)}")

    if out_q is None:
        out_q = torch.empty(
            (num_rows, num_q_heads, qk_head_dim), dtype=fp8_dtype, device=qkv.device
        )
    # The HIP kernel always writes row-major q_scale [num_rows, num_q_heads]
    # (the decode layout); prefill is relayed out to the padded [req, h, pos]
    # form below to match the Triton main kernel.
    q_scale_rows = torch.empty(
        (num_rows, num_q_heads), dtype=torch.float32, device=qkv.device
    )
    # split_k_flag: reserved/unused on the AMD path; empty (no memset).
    split_k_flag = torch.empty(
        (num_req, num_kv_heads), dtype=torch.int32, device=qkv.device
    )

    # Tile by actual token rows, not by q_scale layout. vLLM may pass
    # is_prefill=False for prefill/unified batches to keep row-major q_scale,
    # but HPW should follow the real row-head work size.
    decode_one_token = num_rows == num_req
    tile_hpw = select_tile_hpw(num_rows, num_q_heads)
    H = _get_hadamard(qk_head_dim, qkv.device, qkv.dtype)
    rope_norm_store_kv_fp8_fused_hip(
        qkv,
        cos_sin,
        q_index,
        num_seqlen_per_req,
        kvcache_indices,
        q_norm_weight,
        k_norm_weight,
        H,
        k_scale,
        v_scale,
        out_q,
        key_cache,
        value_cache,
        q_scale_rows,
        1e-5,
        float(fp8_max),
        bool(decode_one_token),
        False,
        tile_hpw,
    )

    if not is_prefill:
        return out_q, q_scale_rows, split_k_flag

    # ===== Prefill: match the Triton main-kernel q_scale layout, then run the
    # (Triton) trailing-zero kernel. =====
    q_scale_out = _scatter_q_scale_prefill(
        q_scale_rows, q_index, num_req, num_q_heads, max_seqlens
    )
    _zero_trailing_kv(
        key_cache,
        value_cache,
        num_seqlen_per_req,
        kvcache_indices,
        block_size,
        qk_head_dim,
        v_head_dim,
        k_cache_x,
    )
    return out_q, q_scale_out, split_k_flag
