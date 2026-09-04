# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exact-arithmetic Qwen3.8 Q/K RMSNorm, RoPE, gate, and amax kernel."""

import triton
import triton.language as tl


@triton.jit
def persistent_qk_norm_rope_gate_token_amax_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    q_token_amax_ptr,
    k_token_amax_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    num_tokens,
    q_gate_stride_t,
    k_stride_t,
    q_out_stride_t,
    k_out_stride_t,
    gate_out_stride_t,
    cache_stride_p,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    eps: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    HAS_PASS: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
    ADD_GEMMA_OFFSET: tl.constexpr,
    REUSE_NORMALIZED_ROTARY: tl.constexpr,
):
    token_block = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)
    if is_k:
        weight_ptr = k_weight_ptr
    else:
        weight_ptr = q_weight_ptr

    # Preserve the validated one-dimensional head reduction while reusing
    # immutable weights across several tokens in a workgroup.
    head_offsets = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offsets < head_dim
    weight = tl.load(
        weight_ptr + head_offsets,
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)
    if ADD_GEMMA_OFFSET:
        weight += 1.0

    rotary_offsets = tl.arange(0, ROT_HALF_BLOCK)
    rotary_mask = rotary_offsets < half_rotary
    if not REUSE_NORMALIZED_ROTARY:
        w1 = tl.load(
            weight_ptr + rotary_offsets,
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)
        w2 = tl.load(
            weight_ptr + half_rotary + rotary_offsets,
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)
        if ADD_GEMMA_OFFSET:
            w1 += 1.0
            w2 += 1.0

    for token_offset in tl.static_range(0, TOKENS_PER_PROGRAM):
        token = token_block * TOKENS_PER_PROGRAM + token_offset
        valid_token = token < num_tokens

        if is_k:
            in_base = k_ptr + token * k_stride_t + local_head * head_dim
            out_base = k_out_ptr + token * k_out_stride_t + local_head * head_dim
        else:
            in_base = q_gate_ptr + token * q_gate_stride_t + local_head * 2 * head_dim
            out_base = q_out_ptr + token * q_out_stride_t + local_head * head_dim

        full_mask = head_mask & valid_token
        values = tl.load(
            in_base + head_offsets,
            mask=full_mask,
            other=0.0,
        ).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / head_dim
        inv_rms = tl.rsqrt(variance + eps)
        normalized = (values * inv_rms * weight).to(INPUT_DTYPE).to(tl.float32)

        if REUSE_NORMALIZED_ROTARY:
            position = tl.load(
                positions_ptr + token,
                mask=valid_token,
                other=0,
            ).to(tl.int64)
            cache_base = position * cache_stride_p
            rotary_full_mask = (head_offsets < rotary_dim) & valid_token
            partner_offsets = tl.where(
                head_offsets < half_rotary,
                head_offsets + half_rotary,
                tl.where(
                    head_offsets < rotary_dim,
                    head_offsets - half_rotary,
                    head_offsets,
                ),
            )
            partner = tl.gather(normalized, partner_offsets, axis=0)
            frequency_offsets = head_offsets % half_rotary
            cos = tl.load(
                cos_sin_cache_ptr + cache_base + frequency_offsets,
                mask=rotary_full_mask,
                other=0.0,
            ).to(tl.float32)
            sin = tl.load(
                cos_sin_cache_ptr + cache_base + half_rotary + frequency_offsets,
                mask=rotary_full_mask,
                other=0.0,
            ).to(tl.float32)
            rotated = tl.where(
                head_offsets < half_rotary,
                normalized * cos - partner * sin,
                normalized * cos + partner * sin,
            )
            combined = (
                tl.where(
                    head_offsets < rotary_dim,
                    rotated,
                    normalized,
                )
                .to(INPUT_DTYPE)
                .to(tl.float32)
            )
            tl.store(
                out_base + head_offsets,
                combined,
                mask=full_mask,
            )
            token_amax = tl.max(
                tl.where(full_mask, tl.abs(combined), 0.0),
                axis=0,
            )
        else:
            if HAS_PASS:
                pass_mask = head_mask & (head_offsets >= rotary_dim) & valid_token
                tl.store(
                    out_base + head_offsets,
                    normalized,
                    mask=pass_mask,
                )
                pass_amax = tl.max(
                    tl.where(pass_mask, tl.abs(normalized), 0.0),
                    axis=0,
                )
            else:
                pass_amax = 0.0

            rotary_token_mask = rotary_mask & valid_token
            x1 = tl.load(
                in_base + rotary_offsets,
                mask=rotary_token_mask,
                other=0.0,
            ).to(tl.float32)
            x2 = tl.load(
                in_base + half_rotary + rotary_offsets,
                mask=rotary_token_mask,
                other=0.0,
            ).to(tl.float32)
            x1 = (x1 * inv_rms * w1).to(INPUT_DTYPE).to(tl.float32)
            x2 = (x2 * inv_rms * w2).to(INPUT_DTYPE).to(tl.float32)

            position = tl.load(
                positions_ptr + token,
                mask=valid_token,
                other=0,
            ).to(tl.int64)
            cache_base = position * cache_stride_p
            cos = tl.load(
                cos_sin_cache_ptr + cache_base + rotary_offsets,
                mask=rotary_token_mask,
                other=0.0,
            ).to(tl.float32)
            sin = tl.load(
                cos_sin_cache_ptr + cache_base + half_rotary + rotary_offsets,
                mask=rotary_token_mask,
                other=0.0,
            ).to(tl.float32)

            out1 = (x1 * cos - x2 * sin).to(INPUT_DTYPE).to(tl.float32)
            out2 = (x2 * cos + x1 * sin).to(INPUT_DTYPE).to(tl.float32)
            tl.store(
                out_base + rotary_offsets,
                out1,
                mask=rotary_token_mask,
            )
            tl.store(
                out_base + half_rotary + rotary_offsets,
                out2,
                mask=rotary_token_mask,
            )

            rotary_amax = tl.maximum(
                tl.max(
                    tl.where(
                        rotary_token_mask,
                        tl.abs(out1),
                        0.0,
                    ),
                    axis=0,
                ),
                tl.max(
                    tl.where(
                        rotary_token_mask,
                        tl.abs(out2),
                        0.0,
                    ),
                    axis=0,
                ),
            )
            token_amax = tl.maximum(pass_amax, rotary_amax)

        if is_k:
            tl.store(
                k_token_amax_ptr + token * num_kv_heads + local_head,
                token_amax,
                mask=valid_token,
            )
        else:
            tl.store(
                q_token_amax_ptr + token * num_q_heads + local_head,
                token_amax,
                mask=valid_token,
            )

        if not is_k:
            gate_in_base = in_base + head_dim
            gate_out_base = (
                gate_out_ptr + token * gate_out_stride_t + local_head * head_dim
            )
            gate = tl.load(
                gate_in_base + head_offsets,
                mask=full_mask,
                other=0.0,
            )
            tl.store(
                gate_out_base + head_offsets,
                gate,
                mask=full_mask,
            )


@triton.jit
def v_token_amax_kernel(
    v_ptr,
    v_token_amax_ptr,
    token_start,
    num_tokens,
    v_stride_t,
    v_stride_h,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    local_token = tl.program_id(0)
    token = token_start + local_token
    kv_head = tl.program_id(1)
    valid_token = local_token < num_tokens

    d_offsets = tl.arange(0, BLOCK_D)
    mask = valid_token & (d_offsets < head_dim)
    values = tl.load(
        v_ptr + token * v_stride_t + kv_head * v_stride_h + d_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.where(mask, tl.abs(values), 0.0), axis=0)
    tl.store(
        v_token_amax_ptr + token * num_kv_heads + kv_head,
        amax,
        mask=valid_token,
    )


@triton.jit
def segmented_qkv_partial_amax_kernel(
    q_token_amax_ptr,
    k_token_amax_ptr,
    v_token_amax_ptr,
    cu_seqlens_ptr,
    partial_amax_ptr,
    sequence_start,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    gqa_ratio: tl.constexpr,
    BLOCK_T: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    HAS_SEQUENCE_OFFSET: tl.constexpr,
):
    if HAS_SEQUENCE_OFFSET:
        sequence = sequence_start + tl.program_id(0)
    else:
        sequence = tl.program_id(0)
    kv_head = tl.program_id(1)
    block_idx = tl.program_id(2)
    start = tl.load(cu_seqlens_ptr + sequence)
    end = tl.load(cu_seqlens_ptr + sequence + 1)
    offsets = start + block_idx * BLOCK_T + tl.arange(0, BLOCK_T)
    token_mask = offsets < end
    k_values = tl.load(
        k_token_amax_ptr + offsets * num_kv_heads + kv_head,
        mask=token_mask,
        other=0.0,
    )
    k_max = tl.max(k_values, axis=0)
    v_values = tl.load(
        v_token_amax_ptr + offsets * num_kv_heads + kv_head,
        mask=token_mask,
        other=0.0,
    )
    v_max = tl.max(v_values, axis=0)
    q_max = 0.0
    for query_in_group in tl.static_range(0, gqa_ratio):
        q_head = kv_head * gqa_ratio + query_in_group
        q_values = tl.load(
            q_token_amax_ptr + offsets * num_q_heads + q_head,
            mask=token_mask,
            other=0.0,
        )
        q_max = tl.maximum(q_max, tl.max(q_values, axis=0))

    partial_base = ((sequence * num_kv_heads + kv_head) * NUM_BLOCKS + block_idx) * 3
    tl.store(partial_amax_ptr + partial_base, q_max)
    tl.store(partial_amax_ptr + partial_base + 1, k_max)
    tl.store(partial_amax_ptr + partial_base + 2, v_max)


@triton.jit
def segmented_qkv_scale_kernel(
    partial_amax_ptr,
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    sequence_start,
    num_kv_heads: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    HAS_SEQUENCE_OFFSET: tl.constexpr,
):
    if HAS_SEQUENCE_OFFSET:
        sequence = sequence_start + tl.program_id(0)
    else:
        sequence = tl.program_id(0)
    kv_head = tl.program_id(1)
    block_offsets = tl.arange(0, BLOCK_B)
    block_mask = block_offsets < NUM_BLOCKS
    partial_base = (
        (sequence * num_kv_heads + kv_head) * NUM_BLOCKS + block_offsets
    ) * 3
    q_max = tl.max(
        tl.load(
            partial_amax_ptr + partial_base,
            mask=block_mask,
            other=0.0,
        ),
        axis=0,
    )
    k_max = tl.max(
        tl.load(
            partial_amax_ptr + partial_base + 1,
            mask=block_mask,
            other=0.0,
        ),
        axis=0,
    )
    v_max = tl.max(
        tl.load(
            partial_amax_ptr + partial_base + 2,
            mask=block_mask,
            other=0.0,
        ),
        axis=0,
    )
    q_scale = tl.maximum(q_max / FP8_MAX_VALUE, 1.0e-12)
    k_scale = tl.maximum(k_max / FP8_MAX_VALUE, 1.0e-12)
    v_scale = tl.maximum(v_max / FP8_MAX_VALUE, 1.0e-12)
    tl.store(q_descale_ptr + sequence * num_kv_heads + kv_head, q_scale)
    tl.store(k_descale_ptr + sequence * num_kv_heads + kv_head, k_scale)
    tl.store(v_descale_ptr + sequence * num_kv_heads + kv_head, v_scale)


@triton.jit
def quantize_qk_grouped_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_descale_ptr,
    k_descale_ptr,
    cu_seqlens_ptr,
    num_actual_tokens,
    num_sequences,
    q_stride_t,
    q_stride_h,
    k_stride_t,
    k_stride_h,
    q_out_stride_t,
    q_out_stride_h,
    k_out_stride_t,
    k_out_stride_h,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    gqa_ratio: tl.constexpr,
    head_dim: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    valid_token = token < num_actual_tokens

    low = 0
    high = num_sequences
    for _ in tl.static_range(0, SEARCH_STEPS):
        middle = (low + high) // 2
        boundary = tl.load(
            cu_seqlens_ptr + middle + 1,
            mask=middle < num_sequences,
            other=num_actual_tokens,
        )
        move_right = token >= boundary
        low = tl.where(move_right, middle + 1, low)
        high = tl.where(move_right, high, middle)
    sequence = tl.minimum(low, num_sequences - 1)

    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)
    kv_head = tl.where(is_k, local_head, local_head // gqa_ratio)
    descale = tl.load(
        tl.where(is_k, k_descale_ptr, q_descale_ptr)
        + sequence * num_kv_heads
        + kv_head,
        mask=valid_token,
        other=1.0,
    )
    inv_scale = 1.0 / descale

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = valid_token & (d_offsets < head_dim)
    if is_k:
        in_base = k_ptr + token * k_stride_t + local_head * k_stride_h
        out_base = k_out_ptr + token * k_out_stride_t + local_head * k_out_stride_h
    else:
        in_base = q_ptr + token * q_stride_t + local_head * q_stride_h
        out_base = q_out_ptr + token * q_out_stride_t + local_head * q_out_stride_h

    values = tl.load(in_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    quantized = tl.clamp(
        values * inv_scale,
        -FP8_MAX_VALUE,
        FP8_MAX_VALUE,
    )
    tl.store(out_base + d_offsets, quantized.to(tl.float8e4nv), mask=d_mask)


@triton.jit
def quantize_qk_grouped_offset_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_descale_ptr,
    k_descale_ptr,
    cu_seqlens_ptr,
    quant_token_start,
    num_quant_tokens,
    sequence_start,
    num_sequences,
    q_stride_t,
    q_stride_h,
    k_stride_t,
    k_stride_h,
    q_out_stride_t,
    q_out_stride_h,
    k_out_stride_t,
    k_out_stride_h,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    gqa_ratio: tl.constexpr,
    head_dim: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    SINGLE_SEQUENCE: tl.constexpr,
):
    local_token = tl.program_id(0)
    token = quant_token_start + local_token
    head = tl.program_id(1)
    valid_token = local_token < num_quant_tokens

    if SINGLE_SEQUENCE:
        sequence = sequence_start
    else:
        low = sequence_start
        high = sequence_start + num_sequences - 1
        for _ in tl.static_range(0, SEARCH_STEPS):
            middle = (low + high) // 2
            boundary = tl.load(
                cu_seqlens_ptr + middle + 1,
            )
            move_right = token >= boundary
            low = tl.where(move_right, middle + 1, low)
            high = tl.where(move_right, high, middle)
        sequence = low

    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)
    kv_head = tl.where(is_k, local_head, local_head // gqa_ratio)
    descale = tl.load(
        tl.where(is_k, k_descale_ptr, q_descale_ptr)
        + sequence * num_kv_heads
        + kv_head,
        mask=valid_token,
        other=1.0,
    )
    inv_scale = 1.0 / descale

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = valid_token & (d_offsets < head_dim)
    if is_k:
        in_base = k_ptr + token * k_stride_t + local_head * k_stride_h
        out_base = k_out_ptr + token * k_out_stride_t + local_head * k_out_stride_h
    else:
        in_base = q_ptr + token * q_stride_t + local_head * q_stride_h
        out_base = q_out_ptr + token * q_out_stride_t + local_head * q_out_stride_h
    values = tl.load(in_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    quantized = tl.clamp(
        values * inv_scale,
        -FP8_MAX_VALUE,
        FP8_MAX_VALUE,
    )
    tl.store(out_base + d_offsets, quantized.to(tl.float8e4nv), mask=d_mask)


@triton.jit
def quantize_v_grouped_offset_kernel(
    v_ptr,
    v_out_ptr,
    v_descale_ptr,
    cu_seqlens_ptr,
    quant_token_start,
    num_quant_tokens,
    sequence_start,
    num_sequences,
    v_stride_t,
    v_stride_h,
    v_out_stride_t,
    v_out_stride_h,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    SINGLE_SEQUENCE: tl.constexpr,
):
    local_token = tl.program_id(0)
    token = quant_token_start + local_token
    kv_head = tl.program_id(1)
    valid_token = local_token < num_quant_tokens

    if SINGLE_SEQUENCE:
        sequence = sequence_start
    else:
        low = sequence_start
        high = sequence_start + num_sequences - 1
        for _ in tl.static_range(0, SEARCH_STEPS):
            middle = (low + high) // 2
            boundary = tl.load(cu_seqlens_ptr + middle + 1)
            move_right = token >= boundary
            low = tl.where(move_right, middle + 1, low)
            high = tl.where(move_right, high, middle)
        sequence = low

    descale = tl.load(
        v_descale_ptr + sequence * num_kv_heads + kv_head,
        mask=valid_token,
        other=1.0,
    )
    inv_scale = 1.0 / descale

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = valid_token & (d_offsets < head_dim)
    in_base = v_ptr + token * v_stride_t + kv_head * v_stride_h
    out_base = v_out_ptr + token * v_out_stride_t + kv_head * v_out_stride_h
    values = tl.load(in_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    quantized = tl.clamp(
        values * inv_scale,
        -FP8_MAX_VALUE,
        FP8_MAX_VALUE,
    )
    tl.store(out_base + d_offsets, quantized.to(tl.float8e4nv), mask=d_mask)
