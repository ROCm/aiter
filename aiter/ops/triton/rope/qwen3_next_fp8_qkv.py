# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8 exact Q/K preparation with grouped FP8 Q/K/V outputs.

This operator is intentionally cache-independent. Sequence metadata and the
decode-prefix boundary are explicit inputs so framework integrations retain
ownership of scheduling, cache updates, and execution ordering.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

import aiter
from aiter.ops.triton._triton_kernels.rope.qwen3_next_fp8_qkv import (
    persistent_qk_norm_rope_gate_token_amax_kernel,
    quantize_qk_grouped_kernel,
    quantize_qk_grouped_offset_kernel,
    quantize_v_grouped_offset_kernel,
    segmented_qkv_partial_amax_kernel,
    segmented_qkv_scale_kernel,
    v_token_amax_kernel,
)

MAX_QUERY_TOKENS = 8192
MAX_SEQUENCES = 256
FP8_MAX = 448.0
SCALE_BLOCK_T = 256
SCALE_NUM_BLOCKS = MAX_QUERY_TOKENS // SCALE_BLOCK_T
QK_TOKENS_PER_PROGRAM = 4


class Qwen3NextFp8QKVPrepOutput(NamedTuple):
    query: torch.Tensor
    key: torch.Tensor
    gate: torch.Tensor
    query_fp8: torch.Tensor
    key_fp8: torch.Tensor
    value_fp8: torch.Tensor
    query_descale: torch.Tensor
    key_descale: torch.Tensor
    value_descale: torch.Tensor


def qwen3_next_fp8_qkv_prep(
    q_gate: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_norm_weight: torch.Tensor,
    key_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    num_actual_tokens: int,
    quant_token_start: int = 0,
    quant_sequence_start: int = 0,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float = 1.0e-6,
    query_out: torch.Tensor | None = None,
    key_out: torch.Tensor | None = None,
    gate_out: torch.Tensor | None = None,
    query_fp8_out: torch.Tensor | None = None,
    key_fp8_out: torch.Tensor | None = None,
    value_fp8_out: torch.Tensor | None = None,
    query_descale_out: torch.Tensor | None = None,
    key_descale_out: torch.Tensor | None = None,
    value_descale_out: torch.Tensor | None = None,
) -> Qwen3NextFp8QKVPrepOutput:
    """Prepare Qwen3.8 full-attention Q/K/V for dynamic FP8 FMHA.

    ``q_gate`` uses the Qwen3.8 interleaved per-head ``[Q, gate]`` layout.
    ``query_norm_weight`` and ``key_norm_weight`` are raw zero-centered Gemma
    RMSNorm parameters; the exact FP32 ``+1`` is applied in the kernel.

    Leading decode tokens are normalized and rotated to produce model-visible
    BF16 Q/K/gate outputs, but only the suffix beginning at
    ``quant_token_start`` is quantized. ``quant_sequence_start`` identifies the
    corresponding first suffix sequence in ``cu_seqlens``.

    The returned descales have fixed shape
    ``[MAX_SEQUENCES, num_kv_heads]`` for compile/CUDA-graph stability. Rows
    before ``quant_sequence_start`` and rows at or after ``num_sequences`` are
    unspecified and must not be consumed.
    """
    total_tokens = q_gate.shape[0]
    num_sequences = cu_seqlens.numel() - 1
    num_quant_tokens = num_actual_tokens - quant_token_start
    num_quant_sequences = num_sequences - quant_sequence_start

    if q_gate.device.type != "cuda":
        raise ValueError("qwen3_next_fp8_qkv_prep requires a CUDA/HIP device")
    if q_gate.ndim != 2 or key.ndim != 2 or value.ndim != 2:
        raise ValueError("q_gate, key, and value must be two-dimensional")
    if q_gate.shape[1] != 2 * num_query_heads * head_dim:
        raise ValueError(
            "q_gate has incompatible packed width: "
            f"expected {2 * num_query_heads * head_dim}, got {q_gate.shape[1]}"
        )
    expected_kv_width = num_kv_heads * head_dim
    if key.shape != (total_tokens, expected_kv_width):
        raise ValueError(
            "key has incompatible shape: "
            f"expected {(total_tokens, expected_kv_width)}, got {tuple(key.shape)}"
        )
    if value.shape != key.shape:
        raise ValueError(f"value shape must match key shape, got {tuple(value.shape)}")
    if query_norm_weight.shape != (head_dim,) or key_norm_weight.shape != (head_dim,):
        raise ValueError("Q/K norm weights must have shape [head_dim]")
    if positions.ndim != 1 or positions.shape[0] != total_tokens:
        raise ValueError("positions must be one-dimensional with one row per token")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != rotary_dim:
        raise ValueError("cos_sin_cache must have shape [max_position, rotary_dim]")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must be a one-dimensional int32 tensor")
    if not 0 < num_actual_tokens <= total_tokens:
        raise ValueError(
            f"num_actual_tokens must be in [1, {total_tokens}], got {num_actual_tokens}"
        )
    if not 0 <= quant_token_start <= num_actual_tokens:
        raise ValueError("quant_token_start must be within the actual-token range")
    if not 0 <= quant_sequence_start <= num_sequences:
        raise ValueError("quant_sequence_start must be within the sequence range")
    if (num_quant_tokens == 0) != (num_quant_sequences == 0):
        raise ValueError(
            "quantized token and sequence suffixes must both be empty or non-empty"
        )
    if num_quant_tokens == 0:
        raise ValueError(
            "qwen3_next_fp8_qkv_prep requires at least one prefill/extend token"
        )
    if total_tokens > MAX_QUERY_TOKENS:
        raise ValueError(
            f"at most {MAX_QUERY_TOKENS} padded query tokens are supported"
        )
    if num_sequences > MAX_SEQUENCES:
        raise ValueError(f"at most {MAX_SEQUENCES} sequences are supported")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be positive, even, and <= head_dim")
    if q_gate.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("q_gate must use bfloat16 or float16")
    if key.dtype != q_gate.dtype or value.dtype != q_gate.dtype:
        raise ValueError("q_gate, key, and value must have the same dtype")
    tensors = (
        key,
        value,
        query_norm_weight,
        key_norm_weight,
        cos_sin_cache,
        positions,
        cu_seqlens,
    )
    if any(tensor.device != q_gate.device for tensor in tensors):
        raise ValueError("all inputs must be on the same device")

    provided_outputs = (
        query_out,
        key_out,
        gate_out,
        query_fp8_out,
        key_fp8_out,
        value_fp8_out,
        query_descale_out,
        key_descale_out,
        value_descale_out,
    )
    if any(output is not None for output in provided_outputs):
        if not all(output is not None for output in provided_outputs):
            raise ValueError(
                "Qwen3 FP8 QKV output buffers must be all set or all unset"
            )
        query = query_out
        output_key = key_out
        gate = gate_out
        query_fp8 = query_fp8_out
        key_fp8 = key_fp8_out
        value_fp8 = value_fp8_out
        query_descale = query_descale_out
        key_descale = key_descale_out
        value_descale = value_descale_out
        assert query is not None
        assert output_key is not None
        assert gate is not None
        assert query_fp8 is not None
        assert key_fp8 is not None
        assert value_fp8 is not None
        assert query_descale is not None
        assert key_descale is not None
        assert value_descale is not None
    else:
        query = torch.empty(
            (total_tokens, num_query_heads * head_dim),
            dtype=q_gate.dtype,
            device=q_gate.device,
        )
        output_key = torch.empty(
            (total_tokens, num_kv_heads * head_dim),
            dtype=key.dtype,
            device=key.device,
        )
        gate = torch.empty_like(query)
        fp8_dtype = aiter.dtypes.fp8
        query_fp8 = torch.empty(
            (total_tokens, num_query_heads, head_dim),
            dtype=fp8_dtype,
            device=q_gate.device,
        )
        key_fp8 = torch.empty(
            (total_tokens, num_kv_heads, head_dim),
            dtype=fp8_dtype,
            device=key.device,
        )
        value_fp8 = torch.empty(
            (total_tokens, num_kv_heads, head_dim),
            dtype=fp8_dtype,
            device=value.device,
        )
        query_descale = torch.empty(
            (MAX_SEQUENCES, num_kv_heads),
            dtype=torch.float32,
            device=q_gate.device,
        )
        key_descale = torch.empty_like(query_descale)
        value_descale = torch.empty_like(query_descale)

    query_token_amax = torch.empty(
        (total_tokens, num_query_heads),
        dtype=torch.float32,
        device=q_gate.device,
    )
    key_token_amax = torch.empty(
        (total_tokens, num_kv_heads),
        dtype=torch.float32,
        device=q_gate.device,
    )
    value_token_amax = torch.empty(
        (total_tokens, num_kv_heads),
        dtype=torch.float32,
        device=q_gate.device,
    )
    partial_amax = torch.empty(
        (MAX_SEQUENCES, num_kv_heads, SCALE_NUM_BLOCKS, 3),
        dtype=torch.float32,
        device=q_gate.device,
    )

    half_rotary = rotary_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    rotary_half_block = triton.next_power_of_2(half_rotary)
    persistent_qk_norm_rope_gate_token_amax_kernel[
        (
            triton.cdiv(total_tokens, QK_TOKENS_PER_PROGRAM),
            num_query_heads + num_kv_heads,
        )
    ](
        q_gate,
        key,
        query,
        output_key,
        gate,
        query_token_amax,
        key_token_amax,
        query_norm_weight,
        key_norm_weight,
        cos_sin_cache,
        positions,
        total_tokens,
        q_gate.stride(0),
        key.stride(0),
        query.stride(0),
        output_key.stride(0),
        gate.stride(0),
        cos_sin_cache.stride(0),
        num_q_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        half_rotary=half_rotary,
        eps=eps,
        INPUT_DTYPE=(tl.bfloat16 if q_gate.dtype == torch.bfloat16 else tl.float16),
        HEAD_BLOCK=head_block,
        ROT_HALF_BLOCK=rotary_half_block,
        HAS_PASS=rotary_dim < head_dim,
        TOKENS_PER_PROGRAM=QK_TOKENS_PER_PROGRAM,
        ADD_GEMMA_OFFSET=True,
        REUSE_NORMALIZED_ROTARY=True,
        num_warps=max(1, head_block // 64),
        num_stages=2,
    )

    gqa_ratio = num_query_heads // num_kv_heads
    value_view = value.view(total_tokens, num_kv_heads, head_dim)
    v_token_amax_kernel[(num_quant_tokens, num_kv_heads)](
        value_view,
        value_token_amax,
        quant_token_start,
        num_quant_tokens,
        value_view.stride(0),
        value_view.stride(1),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )

    has_sequence_offset = quant_sequence_start != 0
    segmented_qkv_partial_amax_kernel[
        (num_quant_sequences, num_kv_heads, SCALE_NUM_BLOCKS)
    ](
        query_token_amax,
        key_token_amax,
        value_token_amax,
        cu_seqlens,
        partial_amax,
        quant_sequence_start,
        num_q_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        gqa_ratio=gqa_ratio,
        BLOCK_T=SCALE_BLOCK_T,
        NUM_BLOCKS=SCALE_NUM_BLOCKS,
        HAS_SEQUENCE_OFFSET=has_sequence_offset,
        num_warps=4,
    )
    segmented_qkv_scale_kernel[(num_quant_sequences, num_kv_heads)](
        partial_amax,
        query_descale,
        key_descale,
        value_descale,
        quant_sequence_start,
        num_kv_heads=num_kv_heads,
        FP8_MAX_VALUE=FP8_MAX,
        NUM_BLOCKS=SCALE_NUM_BLOCKS,
        BLOCK_B=triton.next_power_of_2(SCALE_NUM_BLOCKS),
        HAS_SEQUENCE_OFFSET=has_sequence_offset,
        num_warps=1,
    )

    query_view = query.view(total_tokens, num_query_heads, head_dim)
    key_view = output_key.view(total_tokens, num_kv_heads, head_dim)
    if quant_token_start == 0 and quant_sequence_start == 0:
        quantize_qk_grouped_kernel[(num_actual_tokens, num_query_heads + num_kv_heads)](
            query_view,
            key_view,
            query_fp8,
            key_fp8,
            query_descale,
            key_descale,
            cu_seqlens,
            num_quant_tokens,
            num_quant_sequences,
            query_view.stride(0),
            query_view.stride(1),
            key_view.stride(0),
            key_view.stride(1),
            query_fp8.stride(0),
            query_fp8.stride(1),
            key_fp8.stride(0),
            key_fp8.stride(1),
            num_q_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            gqa_ratio=gqa_ratio,
            head_dim=head_dim,
            FP8_MAX_VALUE=FP8_MAX,
            BLOCK_D=triton.next_power_of_2(head_dim),
            SEARCH_STEPS=8,
            num_warps=4,
        )
        search_steps = max(0, (num_quant_sequences - 1).bit_length())
    else:
        search_steps = max(0, (num_quant_sequences - 1).bit_length())
        quantize_qk_grouped_offset_kernel[
            (num_quant_tokens, num_query_heads + num_kv_heads)
        ](
            query_view,
            key_view,
            query_fp8,
            key_fp8,
            query_descale,
            key_descale,
            cu_seqlens,
            quant_token_start,
            num_quant_tokens,
            quant_sequence_start,
            num_quant_sequences,
            query_view.stride(0),
            query_view.stride(1),
            key_view.stride(0),
            key_view.stride(1),
            query_fp8.stride(0),
            query_fp8.stride(1),
            key_fp8.stride(0),
            key_fp8.stride(1),
            num_q_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            gqa_ratio=gqa_ratio,
            head_dim=head_dim,
            FP8_MAX_VALUE=FP8_MAX,
            BLOCK_D=triton.next_power_of_2(head_dim),
            SEARCH_STEPS=search_steps,
            SINGLE_SEQUENCE=num_quant_sequences == 1,
            num_warps=4,
        )

    quantize_v_grouped_offset_kernel[(num_quant_tokens, num_kv_heads)](
        value_view,
        value_fp8,
        value_descale,
        cu_seqlens,
        quant_token_start,
        num_quant_tokens,
        quant_sequence_start,
        num_quant_sequences,
        value_view.stride(0),
        value_view.stride(1),
        value_fp8.stride(0),
        value_fp8.stride(1),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        FP8_MAX_VALUE=FP8_MAX,
        BLOCK_D=triton.next_power_of_2(head_dim),
        SEARCH_STEPS=search_steps,
        SINGLE_SEQUENCE=num_quant_sequences == 1,
        num_warps=4,
    )

    return Qwen3NextFp8QKVPrepOutput(
        query=query,
        key=output_key,
        gate=gate,
        query_fp8=query_fp8,
        key_fp8=key_fp8,
        value_fp8=value_fp8,
        query_descale=query_descale,
        key_descale=key_descale,
        value_descale=value_descale,
    )
