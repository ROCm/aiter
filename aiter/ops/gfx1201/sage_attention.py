# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1201 SageAttention: INT8 QK, FP8 PV, noncausal dense attention."""

import logging
import os

import torch

from csrc.cpp_itfs.torch_utils import direct_register_custom_op

logger = logging.getLogger("aiter")

_reported_routes = set()


def _validate(query, key, value):
    if query.ndim != 4 or query.shape[-1] != 128:
        raise ValueError("Expected Q/K/V layout [B,S,H,128]")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            "Q/K/V must have equal shapes; GQA and cross-attention are unsupported"
        )
    if any(size <= 0 for size in query.shape):
        raise ValueError("Expected positive dimensions")
    padded_sequence = ((query.shape[1] + 31) // 32) * 32
    if query.shape[0] * padded_sequence * query.shape[2] * 128 > 2**31 - 1:
        raise ValueError("Tensor exceeds the supported indexing range")
    for tensor in (query, key, value):
        if tensor.device != query.device or tensor.device.type != "cuda":
            raise ValueError("Q/K/V must be on the same ROCm device")
        if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError("Q/K/V must be contiguous BF16 tensors")
        if tensor.requires_grad:
            raise ValueError("gfx1201_sage_attention is inference-only")


def _check_device(query):
    if torch.version.hip is None:
        raise RuntimeError("gfx1201_sage_attention requires ROCm")
    architecture = torch.cuda.get_device_properties(query.device).gcnArchName.split(":")[0]
    if architecture != "gfx1201":
        raise RuntimeError(f"Unsupported architecture {architecture}; expected gfx1201")


def _launch_core(prepared, batch, sequence, heads, device, dtype):
    from .hip_attention import launch_hip_sage_core

    query_int8, query_scale, key_int8, key_scale, value_fp8, value_scale = prepared
    padded_sequence = ((sequence + 31) // 32) * 32
    output = torch.empty((batch, padded_sequence, heads, 128), device=device, dtype=dtype)
    route = "HIP_GENERAL"
    if os.environ.get("AITER_GFX1201_SAGE_FORCE_HIP", "0") != "1":
        from .asm_attention import launch_hip_sage_core

        route = "ASM_PV_EARLY"
    launch_hip_sage_core(
        query_int8,
        key_int8,
        value_fp8,
        query_scale,
        key_scale,
        value_scale,
        output,
        batch,
        padded_sequence,
        sequence,
        heads,
    )
    route_key = (str(device), route)
    if route_key not in _reported_routes:
        rank = os.environ.get("RANK", "0")
        logger.info(
            "gfx1201_sage_attention route=%s rank=%s device=%s shape=%s",
            route,
            rank,
            device,
            (batch, sequence, heads, 128),
        )
        _reported_routes.add(route_key)
    return output[:, :sequence].contiguous()


def gfx1201_sage_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Approximate dense noncausal attention, fixed scale D**-0.5, BF16 BSHD."""
    _validate(query, key, value)
    _check_device(query)
    from .prepare import prepare_sage

    with torch.cuda.device(query.device):
        batch, sequence, heads, _ = query.shape
        return _launch_core(
            prepare_sage(query, key, value),
            batch,
            sequence,
            heads,
            query.device,
            query.dtype,
        )


def _gfx1201_sage_attention_fake(query, key, value):
    _validate(query, key, value)
    return torch.empty_like(query)


direct_register_custom_op(
    op_name="gfx1201_sage_attention",
    op_func=gfx1201_sage_attention,
    mutates_args=[],
    fake_impl=_gfx1201_sage_attention_fake,
)


def _validate_norm_rope(query, key, value, query_weight, key_weight, cosine, sine):
    _validate(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0))
    if query.shape[1:] != (28, 128):
        raise ValueError(
            "Fused QK norm/RoPE attention requires local TP2 layout [S,28,128]"
        )
    for tensor in (query_weight, key_weight):
        if (
            tensor.shape != (128,)
            or tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous()
            or tensor.device != query.device
        ):
            raise ValueError("QK norm weights must be contiguous same-device BF16 [128]")
    for tensor in (cosine, sine):
        if (
            tensor.shape != (query.shape[0], 96)
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
            or tensor.device != query.device
        ):
            raise ValueError(
                "RoPE cosine/sine must be contiguous same-device FP32 [S,96]"
            )


def gfx1201_norm_rope_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """attention(rope(qk_norm(Q)), rope(qk_norm(K)), V) with norm eps 1e-5.

    Pre-norm BF16 ``[S, 28, 128]`` in, ``[S, 28, 128]`` out. Bitwise equal to the
    separate qk_norm, rope, and attention chain.
    """
    _validate_norm_rope(query, key, value, query_weight, key_weight, cosine, sine)
    _check_device(query)
    from .norm_rope_prepare import norm_rope_prepare_sage

    with torch.cuda.device(query.device):
        prepared = norm_rope_prepare_sage(
            query, key, value, query_weight, key_weight, cosine, sine
        )
        return _launch_core(
            prepared, 1, query.shape[0], query.shape[1], query.device, query.dtype
        )[0]


def _gfx1201_norm_rope_attention_fake(
    query, key, value, query_weight, key_weight, cosine, sine
):
    _validate_norm_rope(query, key, value, query_weight, key_weight, cosine, sine)
    return torch.empty_like(query)


direct_register_custom_op(
    op_name="gfx1201_norm_rope_attention",
    op_func=gfx1201_norm_rope_attention,
    mutates_args=[],
    fake_impl=_gfx1201_norm_rope_attention_fake,
)
