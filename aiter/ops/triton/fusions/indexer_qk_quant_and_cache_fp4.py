# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton._triton_kernels.fusions.indexer_qk_quant_and_cache_fp4 import (
    _publish,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir


def indexer_qk_quant_and_cache_fp4(
    q,
    q_out,
    weights,
    weights_out,
    k,
    kv_cache,
    slot_mapping,
    norm_weight,
    norm_bias,
    positions,
    cos_cache,
    sin_cache,
    epsilon,
    q_scale_out,
    kv_cache_scale,
    *,
    rope_dim=64,
    is_neox=True,
    compute_all_q_rope=False,
):
    """Write FP4 Q and normalized K with OPUS scales on gfx950.

    Q/K are BF16 with D=128 and H=32/64. RoPE is 0 or 64 dimensions.
    The caller owns all outputs. K uses 64-row pages. Q scales have shape
    [T,2,32,H/16]; K data/scales use [pages,4,64,16]/[pages,2,32,4].
    Copy head weights without scaling. Negative slots skip K; they skip Q
    unless compute_all_q_rope is True. Nonnegative slots must be unique and
    less than pages*64. Positions are clamped to the RoPE table. Return None.
    """
    if not q.is_cuda or get_arch() != "gfx950":
        raise ValueError("OPUS DSA FP4 writer requires a gfx950 GPU")
    if q.ndim != 3:
        raise ValueError("Q must have shape [T,H,128]")
    rows, heads, dim = q.shape
    if dim != 128 or heads not in (32, 64) or rope_dim not in (0, 64):
        raise ValueError("OPUS DSA FP4 requires H=32/64, D=128, RoPE=0/64")
    if k.shape != (rows, dim) or slot_mapping.ndim != 1:
        raise ValueError("Expected K [T,128] and one-dimensional slots")
    if slot_mapping.dtype != torch.int64 or not slot_mapping.is_contiguous():
        raise ValueError("Slots must be contiguous int64")
    if slot_mapping.numel() > rows:
        raise ValueError("Slots must not exceed Q rows")
    if weights.shape != (rows, heads) or weights_out.shape != weights.shape:
        raise ValueError("Head weights must have shape [T,H]")
    if rows > 1 and weights_out.stride(0) < heads:
        raise ValueError("Output head weight rows must not overlap")
    if norm_weight.shape != (dim,) or norm_bias.shape != (dim,):
        raise ValueError("LayerNorm parameters must have shape [128]")
    if norm_weight.dtype != torch.float32 or norm_bias.dtype != torch.float32:
        raise ValueError("LayerNorm parameters must be FP32")
    tensors = [q, k, weights, weights_out, slot_mapping, norm_weight, norm_bias]
    for tensor in (q, k, weights, weights_out):
        if tensor.dtype != torch.bfloat16 or tensor.stride(-1) != 1:
            raise ValueError(
                "Q/K and head weights must be BF16 with contiguous columns"
            )
    if not norm_weight.is_contiguous() or not norm_bias.is_contiguous():
        raise ValueError("LayerNorm parameters must be contiguous")
    if kv_cache.ndim != 4:
        raise ValueError("K cache must have shape [pages,4,64,16]")
    outputs = (
        (q_out, (rows, heads, 64)),
        (q_scale_out, (rows, 2, 32, heads // 16)),
        (kv_cache, (kv_cache.shape[0], 4, 64, 16)),
        (kv_cache_scale, (kv_cache.shape[0], 2, 32, 4)),
    )
    for tensor, shape in outputs:
        if (
            tensor is None
            or tensor.dtype != torch.uint8
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"Expected contiguous uint8 output with shape {shape}")
        tensors.append(tensor)
    if rope_dim:
        if positions is None or cos_cache is None or sin_cache is None:
            raise ValueError("RoPE requires positions, cosine and sine tables")
        if (
            positions.dtype != torch.int64
            or positions.shape != (rows,)
            or not positions.is_contiguous()
        ):
            raise ValueError("Positions must be contiguous int64 [T]")
        if (
            cos_cache.ndim != 2
            or cos_cache.shape[0] == 0
            or cos_cache.shape[1] != rope_dim // 2
        ):
            raise ValueError("RoPE tables must have shape [max_position,32]")
        if (
            cos_cache.shape != sin_cache.shape
            or cos_cache.stride() != sin_cache.stride()
        ):
            raise ValueError("RoPE table shapes and strides must match")
        if any(
            t.dtype != torch.bfloat16 or t.stride(-1) != 1
            for t in (cos_cache, sin_cache)
        ):
            raise ValueError("RoPE tables must be BF16 with contiguous columns")
        tensors.extend((positions, cos_cache, sin_cache))
    if any(t.device != q.device for t in tensors):
        raise ValueError("All writer tensors must be on the Q device")
    if rows == 0:
        return
    directory = resolve_config_dir("fusions", "INDEXER-QK-QUANT-AND-CACHE-FP4")
    config = load_config_json(f"{directory}/DEFAULT.json", required=True)
    _publish[(rows, heads + 1)](
        q,
        k,
        weights,
        weights_out,
        norm_weight,
        norm_bias,
        slot_mapping,
        kv_cache,
        kv_cache_scale,
        q_out,
        q_scale_out,
        positions,
        cos_cache,
        sin_cache,
        weights.stride(0),
        weights.stride(1),
        weights_out.stride(0),
        weights_out.stride(1),
        compute_all_q_rope,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        slot_mapping.numel(),
        heads,
        epsilon,
        rope_dim,
        is_neox,
        cos_cache.stride(0) if rope_dim else 0,
        cos_cache.shape[0] if rope_dim else 0,
        **config,
    )
