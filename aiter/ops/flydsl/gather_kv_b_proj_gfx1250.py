# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 Kimi-K3 gather + FlyDSL ptpc projection."""

import torch
import triton
import triton.language as tl
from torch import Tensor

from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle_flydsl
from aiter.ops.triton.gather_kv_b_proj import gather_kv_b_proj

_HEADS = 96
_KV_C = 512
_KV_PE = 64
_NOPE = 128
_V_DIM = 128
_WEIGHT_N = _HEADS * (_NOPE + _V_DIM)
_FLYDSL_MIN_ROWS = 4096
_GEMM_CONFIG = {
    "kernelName": "flydsl_bpreshuffle_wmma_t256x128x128_" "mw4_nw1_nb3_sk1_cm1_cn1"
}


def unsupported_reason(
    k_buffer: Tensor,
    kv_proj_weight: Tensor,
    kv_proj_scale: Tensor | None,
    k_prefix: Tensor,
    v_prefix: Tensor,
    *,
    shuffled_kv_cache: bool = False,
    **_,
) -> str | None:
    if shuffled_kv_cache:
        return "shuffled_kv_cache is not supported"
    if k_buffer.dim() != 3 or tuple(k_buffer.shape[1:]) != (1, _KV_C + _KV_PE):
        return f"k_buffer must be [num_blocks, 1, {_KV_C + _KV_PE}]"
    if k_buffer.dtype != torch.float8_e4m3fn:
        return f"k_buffer must be torch.float8_e4m3fn, got {k_buffer.dtype}"
    if kv_proj_weight.dtype != torch.float8_e4m3fn:
        return f"kv_proj_weight must be torch.float8_e4m3fn, got {kv_proj_weight.dtype}"
    if tuple(kv_proj_weight.shape) != (_WEIGHT_N, _KV_C):
        return f"kv_proj_weight must be [{_WEIGHT_N}, {_KV_C}]"
    if kv_proj_scale is None or kv_proj_scale.numel() != _WEIGHT_N:
        return f"kv_proj_scale must have {_WEIGHT_N} per-output-row elements"
    if not getattr(kv_proj_weight, "is_shuffled", False):
        return "kv_proj_weight must use the 16x16 preshuffled layout"
    if (
        k_prefix.dim() != 3
        or tuple(k_prefix.shape[1:]) != (_HEADS, _NOPE + _KV_PE)
        or v_prefix.dim() != 3
        or tuple(v_prefix.shape[1:]) != (_HEADS, _V_DIM)
        or k_prefix.shape[0] != v_prefix.shape[0]
    ):
        return "outputs must be [M,96,192] and [M,96,128]"
    if k_prefix.dtype != torch.bfloat16 or v_prefix.dtype != torch.bfloat16:
        return "outputs must both be bfloat16"
    if not k_prefix.is_contiguous() or not v_prefix.is_contiguous():
        return "outputs must be contiguous"
    tensors = (kv_proj_weight, kv_proj_scale, k_prefix, v_prefix)
    if any(t.device != k_buffer.device for t in tensors):
        return "all tensors must be on the cache device"
    return None


@triton.jit
def _gather_latent(
    cache,
    indices,
    xq,
    x_scale,
    cache_scale,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_IS_PTR: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_K)
    row_mask = rows < M
    safe_rows = tl.where(row_mask, rows, 0)
    slots = tl.load(indices + safe_rows, mask=row_mask, other=0)
    values = tl.load(
        cache + slots[:, None] * 576 + cols[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    tl.store(
        xq + safe_rows[:, None] * 512 + cols[None, :],
        values,
        mask=row_mask[:, None],
    )
    scale = tl.load(cache_scale) if SCALE_IS_PTR else cache_scale
    tl.store(x_scale + safe_rows, scale, mask=row_mask)


@triton.jit
def _split_projected_and_rope(
    projected,
    cache,
    indices,
    cache_scale,
    k_out,
    v_out,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE_IS_PTR: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    head = tl.program_id(1)
    cols = tl.arange(0, BLOCK_D)
    row_mask = rows < M
    safe_rows = tl.where(row_mask, rows, 0)
    mask = row_mask[:, None] & (cols[None, :] < 128)
    proj_base = safe_rows[:, None] * 24576 + head * 256
    k = tl.load(projected + proj_base + cols[None, :], mask=mask, other=0.0)
    v = tl.load(projected + proj_base + 128 + cols[None, :], mask=mask, other=0.0)
    k_base = safe_rows[:, None] * 18432 + head * 192
    v_base = safe_rows[:, None] * 12288 + head * 128
    tl.store(k_out + k_base + cols[None, :], k, mask=mask)
    tl.store(v_out + v_base + cols[None, :], v, mask=mask)

    rope_mask = row_mask[:, None] & (cols[None, :] < 64)
    slots = tl.load(indices + safe_rows, mask=row_mask, other=0)
    rope = tl.load(
        cache + slots[:, None] * 576 + 512 + cols[None, :],
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(cache_scale) if SCALE_IS_PTR else cache_scale
    rope *= scale
    tl.store(k_out + k_base + 128 + cols[None, :], rope, mask=rope_mask)


def gather_kv_b_proj_flydsl_gfx1250(
    k_buffer: Tensor,
    k_scale: Tensor,
    kv_indptr: Tensor,
    kv_indices: Tensor,
    kv_prefix_sum_context_lens: Tensor,
    kv_proj_weight: Tensor,
    kv_proj_scale: Tensor,
    k_prefix: Tensor,
    v_prefix: Tensor,
    *,
    weight_preshuffle: bool = True,
    shuffled_kv_cache: bool = False,
    flydsl_min_rows: int = _FLYDSL_MIN_ROWS,
) -> None:
    """Use FlyDSL WMMA for long Kimi-K3 ptpc prefixes, Triton below crossover."""
    reason = unsupported_reason(
        k_buffer,
        kv_proj_weight,
        kv_proj_scale,
        k_prefix,
        v_prefix,
        shuffled_kv_cache=shuffled_kv_cache,
    )
    if reason is not None or not weight_preshuffle:
        raise ValueError(
            f"[gfx1250 FlyDSL gather_kv_b_proj] {reason or 'weight layout'}"
        )

    m = k_prefix.shape[0]
    if m < flydsl_min_rows:
        gather_kv_b_proj(
            k_buffer,
            k_scale,
            kv_indptr,
            kv_indices,
            kv_prefix_sum_context_lens,
            kv_proj_weight,
            kv_proj_scale,
            k_prefix,
            v_prefix,
            weight_preshuffle=True,
        )
        return
    if kv_indices.numel() < m:
        raise ValueError(f"kv_indices has {kv_indices.numel()} entries, need {m}")
    if k_scale.numel() != 1 or k_scale.device not in (
        k_buffer.device,
        torch.device("cpu"),
    ):
        raise ValueError("k_scale must be one element on the CPU or cache device")
    scale_is_ptr = k_scale.device.type != "cpu"
    scale_arg = k_scale if scale_is_ptr else float(k_scale)

    # Reuse k_prefix before its final write for the small gather/scale workspaces.
    raw = k_prefix.view(torch.uint8).view(-1)
    x_bytes = m * _KV_C
    scale_bytes = m * torch.float32.itemsize
    xq = raw[:x_bytes].view(torch.float8_e4m3fn).view(m, _KV_C)
    x_scale = raw[x_bytes : x_bytes + scale_bytes].view(torch.float32).view(m, 1)
    projected = torch.empty(
        (m, _WEIGHT_N), dtype=torch.bfloat16, device=k_buffer.device
    )

    _gather_latent[(triton.cdiv(m, 16),)](
        k_buffer,
        kv_indices,
        xq,
        x_scale,
        scale_arg,
        M=m,
        BLOCK_M=16,
        BLOCK_K=_KV_C,
        SCALE_IS_PTR=scale_is_ptr,
    )
    gemm_a8w8_bpreshuffle_flydsl(
        xq,
        kv_proj_weight,
        x_scale,
        kv_proj_scale,
        projected,
        _GEMM_CONFIG,
    )
    _split_projected_and_rope[(triton.cdiv(m, 16), _HEADS)](
        projected,
        k_buffer,
        kv_indices,
        scale_arg,
        k_prefix,
        v_prefix,
        M=m,
        BLOCK_M=16,
        BLOCK_D=128,
        SCALE_IS_PTR=scale_is_ptr,
    )
