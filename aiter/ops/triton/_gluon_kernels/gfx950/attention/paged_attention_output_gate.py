# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx950) paged decode attention with a sigmoid output gate and an
optional group-128 FP8 epilogue. Targets Qwen3-Next-style full-attention layers.

Split-K (flash-decoding) over the context: `_attention_partials` writes one
(numerator, denominator, maximum) record per split, then `_merge_gate_quantize`
reduces them, applies `out * sigmoid(gate)`, and optionally emits group-128 FP8.

Two properties make this a single body rather than a table:

* **The context bound is device data.** `begin`/`length` come from `kv_indptr`
  inside the kernel and there is no clamp against a compile-time maximum, so one
  binary serves any context length. The merge picks its reduction width
  (`PREFIX`) from the live length, so a short row does not pay for a wide split
  count.
* **Batch is not a constexpr.** The row index is `gl.program_id(0)`, so one
  binary serves any batch size, including the non-power-of-two sizes a CUDA
  graph capture asks for (12, 24, 40, 48, 56 ...).

`HEAD_TILE = 16` with masking makes the head count a runtime value too; the MFMA
is 16x16x32 regardless.

**Numerics are contract.** The epilogue rounds through BF16 at three points --
attention -> bf16 -> fp32, sigmoid -> bf16 -> fp32, and the product -> bf16 --
before the group absmax is taken. A reimplementation that keeps those
intermediates in fp32 produces different FP8 codes. The reference test asserts
them.

Launcher: aiter/ops/triton/attention/paged_attention_output_gate.py
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

HEAD_DIM: gl.constexpr = 256
QUANT_GROUP: gl.constexpr = 128


@gluon.jit
def _attention_tile(
    query,
    key_ptr,
    value_ptr,
    indices_ptr,
    key_smem,
    value_smem,
    key_copy_smem,
    value_copy_smem,
    begin,
    offset,
    end,
    tokens,
    cache_dims,
    score_tokens,
    key_scale,
    scale,
    maximum,
    denominator,
    numerator,
    BLOCK: gl.constexpr,
    KEY_STRIDE: gl.constexpr,
    VALUE_STRIDE: gl.constexpr,
    mma: gl.constexpr,
    FULL_TILE: gl.constexpr,
    CACHE_MODIFIER: gl.constexpr,
):
    """One BLOCK-token tile of online-softmax attention.

    K and V are gathered asynchronously into LDS and the QK MFMA is overlapped
    with the V transfer via `wait_group(1)`. `FULL_TILE` drops the predicates on
    interior tiles; only the last live tile needs them.
    """
    if FULL_TILE:
        slots = gl.load(indices_ptr + begin + offset + tokens).to(gl.int64)
        gl.amd.cdna4.async_copy.global_load_to_shared(
            key_copy_smem,
            key_ptr + slots[:, None] * KEY_STRIDE + cache_dims,
            cache_modifier=CACHE_MODIFIER,
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.global_load_to_shared(
            value_copy_smem,
            value_ptr + slots[:, None] * VALUE_STRIDE + cache_dims,
            cache_modifier=CACHE_MODIFIER,
        )
        gl.amd.cdna4.async_copy.commit_group()
    else:
        valid = offset + tokens < end
        slots = gl.load(indices_ptr + begin + offset + tokens, valid, other=0).to(
            gl.int64
        )
        gl.amd.cdna4.async_copy.global_load_to_shared(
            key_copy_smem,
            key_ptr + slots[:, None] * KEY_STRIDE + cache_dims,
            valid[:, None],
            other=0.0,
            cache_modifier=CACHE_MODIFIER,
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.global_load_to_shared(
            value_copy_smem,
            value_ptr + slots[:, None] * VALUE_STRIDE + cache_dims,
            valid[:, None],
            other=0.0,
            cache_modifier=CACHE_MODIFIER,
        )
        gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.wait_group(1)
    key = gl.amd.cdna4.async_copy.load_shared_relaxed(
        key_smem.permute((1, 0)), gl.DotOperandLayout(1, mma, 8)
    ).to(gl.bfloat16)
    scores = gl.amd.cdna4.mfma(
        query,
        key,
        gl.full((query.type.shape[0], BLOCK), 0.0, gl.float32, mma),
    )
    scores = scores * key_scale * scale
    if not FULL_TILE:
        scores = gl.where(offset + score_tokens[None, :] < end, scores, -float("inf"))
    next_maximum = gl.maximum(maximum, gl.max(scores, 1))
    correction = gl.exp(maximum - next_maximum)
    if FULL_TILE:
        probabilities = gl.exp(scores - next_maximum[:, None])
    else:
        probabilities = gl.where(
            offset + score_tokens[None, :] < end,
            gl.exp(scores - next_maximum[:, None]),
            0.0,
        )
    denominator = denominator * correction + gl.sum(probabilities, 1)
    numerator = numerator * correction[:, None]

    gl.amd.cdna4.async_copy.wait_group(0)
    value = gl.amd.cdna4.async_copy.load_shared_relaxed(
        value_smem, gl.DotOperandLayout(1, mma, 8)
    )
    # BF16 preserves the exponent range of small attention probabilities. Do not
    # "optimize" this to fp8: at these magnitudes e4m3 is subnormal.
    value = value.to(gl.bfloat16)
    probabilities = probabilities.to(gl.bfloat16)
    probabilities = gl.convert_layout(probabilities, gl.DotOperandLayout(0, mma, 8))
    numerator = gl.amd.cdna4.mfma(probabilities, value, numerator)
    # Finish all relaxed LDS reads before any wave can reuse the K/V storage.
    gl.barrier()
    maximum = next_maximum
    return maximum, denominator, numerator


_partials_repr = make_kernel_repr(
    "_paged_attention_output_gate_partials",
    ["HEADS", "SPLITS", "BLOCK", "CYCLIC", "USE_SCALES"],
)


@gluon.jit(repr=_partials_repr)
def _attention_partials(
    query_ptr,
    key_ptr,
    value_ptr,
    indptr_ptr,
    indices_ptr,
    key_scale_ptr,
    partial_ptr,
    stats_ptr,
    scale,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    BLOCK: gl.constexpr,
    QUERY_ROW_STRIDE: gl.constexpr,
    QUERY_HEAD_STRIDE: gl.constexpr,
    KEY_STRIDE: gl.constexpr,
    VALUE_STRIDE: gl.constexpr,
    USE_SCALES: gl.constexpr,
    CYCLIC: gl.constexpr,
    SWIZZLE_PHASES: gl.constexpr = 16,
    CACHE_MODIFIER: gl.constexpr = "",
):
    """grid = (tokens, SPLITS). One split's partial attention for one token."""
    row = gl.program_id(0)
    split = gl.program_id(1)
    begin = gl.load(indptr_ptr + row)
    length = gl.load(indptr_ptr + row + 1) - begin
    if CYCLIC:
        # Round-robin: split s owns tiles s, s+SPLITS, s+2*SPLITS ... so a short
        # row populates only the low splits and the merge can bound its
        # reduction. No per-split span, hence no dependence on a maximum.
        start = split * BLOCK
        end = length
    else:
        span = gl.cdiv(length, SPLITS * BLOCK) * BLOCK
        start = split * span
        end = gl.minimum(start + span, length)

    warps: gl.constexpr = gl.num_warps()
    HEAD_TILE: gl.constexpr = 16
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, warps],
    )
    query_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [warps, 1], [1, 0])
    copy_width: gl.constexpr = 128 // key_ptr.dtype.element_ty.primitive_bitwidth
    cache_layout: gl.constexpr = gl.BlockedLayout(
        [1, copy_width], [4, 16], [warps, 1], [1, 0]
    )
    shared: gl.constexpr = gl.SwizzledSharedLayout(
        copy_width, 1, SWIZZLE_PHASES, [1, 0]
    )
    key_smem = gl.allocate_shared_memory(
        key_ptr.dtype.element_ty, [BLOCK, HEAD_DIM], shared
    )
    value_smem = gl.allocate_shared_memory(
        value_ptr.dtype.element_ty, [BLOCK, HEAD_DIM], shared
    )
    heads = gl.arange(0, HEAD_TILE, gl.SliceLayout(1, query_layout))
    qdims = gl.arange(0, HEAD_DIM, gl.SliceLayout(0, query_layout))
    query = gl.load(
        query_ptr
        + row * QUERY_ROW_STRIDE
        + heads[:, None] * QUERY_HEAD_STRIDE
        + qdims[None, :],
        heads[:, None] < HEADS,
        other=0.0,
    )
    query = gl.convert_layout(query, gl.DotOperandLayout(0, mma, 8))
    tokens = gl.arange(0, BLOCK, gl.SliceLayout(1, cache_layout))
    dims = gl.arange(0, HEAD_DIM, gl.SliceLayout(0, cache_layout))
    score_tokens = gl.arange(0, BLOCK, gl.SliceLayout(0, mma))
    # Write pre-swizzled values through a linear alias. The MFMA consumers use
    # the original descriptor to recover the logical K/V tile. XOR preserves each
    # aligned copy_width-element contiguous group.
    linear_shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    key_copy_smem = key_smem._reinterpret(layout=linear_shared)
    value_copy_smem = value_smem._reinterpret(layout=linear_shared)
    cache_dims = gl.max_contiguous(
        gl.multiple_of(
            dims[None, :] ^ ((tokens[:, None] & (SWIZZLE_PHASES - 1)) * copy_width),
            [1, copy_width],
        ),
        [1, copy_width],
    )

    key_scale = 1.0
    if USE_SCALES:
        key_scale = gl.load(key_scale_ptr)
    maximum = gl.full((HEAD_TILE,), -float("inf"), gl.float32, gl.SliceLayout(1, mma))
    denominator = gl.full((HEAD_TILE,), 0.0, gl.float32, gl.SliceLayout(1, mma))
    numerator = gl.full((HEAD_TILE, HEAD_DIM), 0.0, gl.float32, mma)

    # Interior tiles are unmasked; only the last live tile needs predicates.
    full_end = end // BLOCK * BLOCK
    step: gl.constexpr = SPLITS * BLOCK if CYCLIC else BLOCK
    for offset in range(start, full_end, step):
        maximum, denominator, numerator = _attention_tile(
            query,
            key_ptr,
            value_ptr,
            indices_ptr,
            key_smem,
            value_smem,
            key_copy_smem,
            value_copy_smem,
            begin,
            offset,
            end,
            tokens,
            cache_dims,
            score_tokens,
            key_scale,
            scale,
            maximum,
            denominator,
            numerator,
            BLOCK,
            KEY_STRIDE,
            VALUE_STRIDE,
            mma,
            True,
            CACHE_MODIFIER,
        )
    if CYCLIC:
        owns_tail = split == (full_end // BLOCK) % SPLITS
    else:
        owns_tail = start < end
    if (full_end < end) & owns_tail:
        offset = full_end
        maximum, denominator, numerator = _attention_tile(
            query,
            key_ptr,
            value_ptr,
            indices_ptr,
            key_smem,
            value_smem,
            key_copy_smem,
            value_copy_smem,
            begin,
            offset,
            end,
            tokens,
            cache_dims,
            score_tokens,
            key_scale,
            scale,
            maximum,
            denominator,
            numerator,
            BLOCK,
            KEY_STRIDE,
            VALUE_STRIDE,
            mma,
            False,
            CACHE_MODIFIER,
        )

    out_heads = gl.arange(0, HEAD_TILE, gl.SliceLayout(1, mma))
    out_dims = gl.arange(0, HEAD_DIM, gl.SliceLayout(0, mma))
    record = (row * HEADS + out_heads) * SPLITS + split
    gl.store(
        partial_ptr + record[:, None] * HEAD_DIM + out_dims[None, :],
        numerator,
        out_heads[:, None] < HEADS,
    )
    gl.store(stats_ptr + record * 2, maximum, out_heads < HEADS)
    gl.store(stats_ptr + record * 2 + 1, denominator, out_heads < HEADS)


@gluon.jit
def _merge_prefix(
    partial_ptr,
    stats_ptr,
    head_row,
    group,
    SPLITS: gl.constexpr,
    PREFIX: gl.constexpr,
    SPLIT_LANES: gl.constexpr,
):
    """Reduce the first PREFIX split records of one (token, head, group)."""
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 4],
        [SPLIT_LANES, 64 // SPLIT_LANES],
        [gl.num_warps(), 1],
        [1, 0],
    )
    splits = gl.arange(0, PREFIX, gl.SliceLayout(1, layout))
    dims = gl.arange(0, QUANT_GROUP, gl.SliceLayout(0, layout))
    record = head_row * SPLITS + splits
    if gl.num_warps() > 1:
        stat_layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
        stat_splits = gl.arange(0, PREFIX, stat_layout)
    else:
        stat_splits = splits
    stat_record = head_row * SPLITS + stat_splits
    maxima = gl.load(stats_ptr + stat_record * 2)
    denominators = gl.load(stats_ptr + stat_record * 2 + 1)
    maximum = gl.max(maxima, 0)
    # Equal maxima need unit weight, including the neutral records of empty rows.
    # Do not replace a nonempty row's maximum based on an infinity sentinel.
    shifted = gl.where(maxima == maximum, 0.0, maxima - maximum)
    weights = gl.exp(shifted)
    denominator = gl.sum(denominators * weights, 0)
    weights = gl.convert_layout(weights, gl.SliceLayout(1, layout))
    partials = gl.load(
        partial_ptr + record[:, None] * HEAD_DIM + group * QUANT_GROUP + dims[None, :]
    )
    numerator = gl.sum(partials * weights[:, None], 0)
    out_layout: gl.constexpr = gl.BlockedLayout([2], [64], [gl.num_warps()], [0])
    return gl.convert_layout(numerator, out_layout), denominator


_merge_repr = make_kernel_repr(
    "_paged_attention_output_gate_merge",
    ["HEADS", "SPLITS", "BLOCK", "QUANTIZE", "USE_SCALES"],
)


@gluon.jit(repr=_merge_repr)
def _merge_gate_quantize(
    partial_ptr,
    stats_ptr,
    indptr_ptr,
    gate_ptr,
    value_scale_ptr,
    gated_ptr,
    quantized_ptr,
    scales_ptr,
    FP8_MAX: gl.constexpr,
    GATE_ROW_STRIDE: gl.constexpr,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    BLOCK: gl.constexpr,
    USE_SCALES: gl.constexpr,
    QUANTIZE: gl.constexpr,
    SPLIT_LANES: gl.constexpr = 8,
):
    """grid = (tokens * HEADS, 2). One program per 128-wide quant group.

    The group IS the CTA, which is what makes the group absmax a plain
    intra-CTA reduction.
    """
    head_row = gl.program_id(0)
    group = gl.program_id(1)
    row = head_row // HEADS
    head = head_row % HEADS
    # Bound the reduction by the live context: with the cyclic assignment above,
    # tile t lands on split t, so a short row genuinely only populates splits
    # 0..t_max and a narrow reduction is exact.
    if SPLITS >= 128:
        length = gl.load(indptr_ptr + row + 1) - gl.load(indptr_ptr + row)
        if length <= 32 * BLOCK:
            numerator, denominator = _merge_prefix(
                partial_ptr, stats_ptr, head_row, group, SPLITS, 32, SPLIT_LANES
            )
        elif length <= 64 * BLOCK:
            numerator, denominator = _merge_prefix(
                partial_ptr, stats_ptr, head_row, group, SPLITS, 64, SPLIT_LANES
            )
        elif SPLITS == 256 and length <= 128 * BLOCK:
            numerator, denominator = _merge_prefix(
                partial_ptr, stats_ptr, head_row, group, SPLITS, 128, SPLIT_LANES
            )
        else:
            numerator, denominator = _merge_prefix(
                partial_ptr, stats_ptr, head_row, group, SPLITS, SPLITS, SPLIT_LANES
            )
    else:
        numerator, denominator = _merge_prefix(
            partial_ptr, stats_ptr, head_row, group, SPLITS, SPLITS, SPLIT_LANES
        )

    out_layout: gl.constexpr = gl.BlockedLayout([2], [64], [gl.num_warps()], [0])
    out_dims = gl.arange(0, QUANT_GROUP, out_layout)
    value_scale = 1.0
    if USE_SCALES:
        value_scale = gl.load(value_scale_ptr)
    attention = gl.where(denominator > 0, numerator * value_scale / denominator, 0.0)

    # These three BF16 rounding boundaries are part of the public contract; the
    # op test asserts them. Keeping the intermediates in fp32 changes the FP8
    # codes this emits.
    attention = attention.to(gl.bfloat16).to(gl.float32)
    # The gate is [tokens, HEADS*HEAD_DIM] and need not be contiguous, so index
    # it through its row stride rather than assuming stride(0) == HEADS*HEAD_DIM.
    # (The Artemis original folds row and head together and silently requires
    # contiguity.) Outputs are always freshly allocated and contiguous.
    gate_offsets = (
        row * GATE_ROW_STRIDE + head * HEAD_DIM + group * QUANT_GROUP + out_dims
    )
    out_offsets = head_row * HEAD_DIM + group * QUANT_GROUP + out_dims
    gate = gl.load(gate_ptr + gate_offsets).to(gl.float32)
    sigmoid = (1.0 / (1.0 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
    gated = (attention * sigmoid).to(gl.bfloat16)
    gl.store(gated_ptr + out_offsets, gated)
    if QUANTIZE:
        values = gated.to(gl.float32)
        quant_scale = gl.maximum(gl.max(gl.abs(values), 0) / FP8_MAX, 1.0e-10)
        quantized = gl.clamp(values / quant_scale, -FP8_MAX, FP8_MAX)
        gl.store(quantized_ptr + out_offsets, quantized)
        gl.store(scales_ptr + head_row * 2 + group, quant_scale)
