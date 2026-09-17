# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged decode attention with a sigmoid output gate and an optional group-128
FP8 epilogue, for Qwen3-Next-style full-attention layers.

Replaces three launches -- paged decode attention, then `out * sigmoid(gate)`,
then the per-group activation quant the next GEMM needs -- with two, by folding
the gate and the quant into the split-K merge the attention already performs.

Why this is a new op rather than a flag on an existing one: no aiter paged-decode
op has this KV layout. `pa_decode` and `pa_decode_gluon` are block-table ops
(`kv_block_size` in {16, 64, 1024}); this takes a CSR index list at page_size 1.
`pa_decode_sparse` takes exactly that CSR layout but is MLA-style -- K and V are
the same tensor -- so it cannot express distinct K and V caches.

Coverage. There is no batch table: `_attention_partials` takes the row from
`gl.program_id(0)` and the context bound from `kv_indptr`, so one binary serves
any batch size and any context length. That matters because a CUDA graph capture
asks for non-power-of-two decode batches (12, 24, 40, 48, 56 ...), and dispatch
happens once at capture -- a miss is baked into the captured graph permanently,
not retried per step.

`max_context` is accepted for launch-geometry selection only; it is not a
compile-time constant and never clamps the live length.
"""

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx950.attention.paged_attention_output_gate import (
    _attention_partials,
    _merge_gate_quantize,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

HEAD_DIM = 256
QUANT_GROUP = 128
_FP8_MAX = {torch.float8_e4m3fn: 448.0, torch.float8_e4m3fnuz: 240.0}


def paged_attention_output_gate_supported(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    gate: torch.Tensor,
    quant_dtype: torch.dtype | None = None,
) -> tuple[bool, str]:
    """(ok, reason). Callers should fall back rather than assert on False."""
    if arch_info.get_arch() != "gfx950":
        return False, f"gluon body is gfx950 only, got {arch_info.get_arch()}"
    if query.ndim != 3 or query.shape[2] != HEAD_DIM:
        return False, f"query must be [T, H, {HEAD_DIM}], got {tuple(query.shape)}"
    heads = query.shape[1]
    if not 0 < heads <= 16:
        return False, f"HEAD_TILE is 16, so heads must be in 1..16, got {heads}"
    if query.dtype is not torch.bfloat16 or gate.dtype is not torch.bfloat16:
        return False, "query and gate must be bfloat16"
    if key_cache.shape != value_cache.shape:
        return False, "key_cache and value_cache must have the same shape"
    if key_cache.ndim != 3 or key_cache.shape[1:] != (1, HEAD_DIM):
        return False, (
            f"caches must be [pages, 1, {HEAD_DIM}] (page_size 1, one KV head), "
            f"got {tuple(key_cache.shape)}"
        )
    if key_cache.dtype is not value_cache.dtype:
        return False, "key_cache and value_cache must have the same dtype"
    if key_cache.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        return False, f"KV cache must be fp8 e4m3, got {key_cache.dtype}"
    if gate.shape != (query.shape[0], heads * HEAD_DIM):
        return False, (
            f"gate must be [T, H*{HEAD_DIM}] = "
            f"{(query.shape[0], heads * HEAD_DIM)}, got {tuple(gate.shape)}"
        )
    # The gather indexes the cache as [page, dim] with a unit dim stride; the
    # Artemis original assumes this without checking.
    for name, cache in (("key_cache", key_cache), ("value_cache", value_cache)):
        if cache.stride(2) != 1:
            return False, f"{name} must have stride(2) == 1"
    if gate.stride(1) != 1:
        return False, "gate must have stride(1) == 1"
    if quant_dtype is not None and quant_dtype not in _FP8_MAX:
        return False, f"quant_dtype must be an fp8 e4m3 type, got {quant_dtype}"
    return True, ""


def paged_attention_output_gate_group_fp8_quant(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    gate: torch.Tensor,
    *,
    scale: float,
    max_context: int | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    quant_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Gated paged-decode attention, optionally emitting group-128 FP8.

    Args:
        query: ``[T, H, 256]`` bf16 decode queries.
        key_cache, value_cache: ``[pages, 1, 256]`` fp8 e4m3, page_size 1.
        kv_indptr: ``[T+1]`` int32 true prefix sum into ``kv_indices``.
        kv_indices: ``[total]`` int32 per-token slot list.
        gate: ``[T, H*256]`` bf16; the output is multiplied by its sigmoid.
        scale: softmax scale (typically ``head_dim ** -0.5``).
        max_context: optional hint used only to pick launch geometry. The live
            context comes from ``kv_indptr``; this never clamps it.
        k_scale, v_scale: optional per-tensor fp32 dequant scales, both or
            neither.
        quant_dtype: when set, also emit ``[T, H*256]`` fp8 plus ``[T, H*2]``
            fp32 group-128 scales. ``None`` skips the epilogue entirely.

    Returns:
        ``(gated_bf16, quantized, scales)``. The last two are ``None`` unless
        ``quant_dtype`` is set.
    """
    ok, reason = paged_attention_output_gate_supported(
        query, key_cache, value_cache, gate, quant_dtype
    )
    if not ok:
        raise ValueError(f"paged_attention_output_gate: unsupported: {reason}")
    if (k_scale is None) != (v_scale is None):
        raise ValueError("k_scale and v_scale must both be given or both be None")
    if kv_indptr.dtype != torch.int32 or kv_indices.dtype != torch.int32:
        raise ValueError("kv_indptr and kv_indices must be int32")

    rows, heads, _ = query.shape
    _LOGGER.info(
        f"PAGED_ATTENTION_OUTPUT_GATE T={rows} H={heads} "
        f"quant={quant_dtype} max_context={max_context}"
    )

    # Launch geometry depends on shape only; every bound inside the kernel is
    # device data. More splits fill the GPU at low batch; past ~512 CTAs the
    # extra splits only add merge work.
    splits = (
        256 if rows == 1 else min(128, triton.next_power_of_2(triton.cdiv(512, rows)))
    )
    block_tokens = 128
    merge_warps = 2 if splits >= 128 else 4 if splits == 64 else 1
    merge_split_lanes = 2 if splits >= 64 else 4

    partials = torch.empty(
        (rows, heads, splits, HEAD_DIM), device=query.device, dtype=torch.float32
    )
    stats = torch.empty(
        (rows, heads, splits, 2), device=query.device, dtype=torch.float32
    )
    gated = torch.empty(
        (rows, heads * HEAD_DIM), device=query.device, dtype=torch.bfloat16
    )
    if quant_dtype is None:
        quantized = scales = None
        fp8_max = 0.0
        # The kernel still needs valid pointers for the unused stores.
        quantized_arg = scales_arg = gated
    else:
        quantized = torch.empty(
            (rows, heads * HEAD_DIM), device=query.device, dtype=quant_dtype
        )
        scales = torch.empty(
            (rows, heads * 2), device=query.device, dtype=torch.float32
        )
        fp8_max = _FP8_MAX[quant_dtype]
        quantized_arg, scales_arg = quantized, scales

    _attention_partials[(rows, splits)](
        query,
        key_cache,
        value_cache,
        kv_indptr,
        kv_indices,
        query if k_scale is None else k_scale,
        partials,
        stats,
        scale,
        HEADS=heads,
        SPLITS=splits,
        BLOCK=block_tokens,
        QUERY_ROW_STRIDE=query.stride(0),
        QUERY_HEAD_STRIDE=query.stride(1),
        KEY_STRIDE=key_cache.stride(0),
        VALUE_STRIDE=value_cache.stride(0),
        USE_SCALES=k_scale is not None,
        CYCLIC=rows >= 16,
        SWIZZLE_PHASES=16,
        CACHE_MODIFIER=".cg" if rows >= 8 else "",
        waves_per_eu=0,
        num_warps=4,
    )
    _merge_gate_quantize[(rows * heads, 2)](
        partials,
        stats,
        kv_indptr,
        gate,
        query if v_scale is None else v_scale,
        gated,
        quantized_arg,
        scales_arg,
        fp8_max,
        GATE_ROW_STRIDE=gate.stride(0),
        HEADS=heads,
        SPLITS=splits,
        BLOCK=block_tokens,
        USE_SCALES=v_scale is not None,
        QUANTIZE=quant_dtype is not None,
        SPLIT_LANES=merge_split_lanes,
        num_warps=merge_warps,
    )
    return gated, quantized, scales
