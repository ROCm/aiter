# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA triton kernels for quantized cache writes, index scoring and sparse
paged GQA.

An FP8 row has one FP32 scale (row amax / FP8 max). An FP4 row is MXFP4: two
E2M1 values in each byte (dim 2j in the low nibble) and one E8M0 exponent for
each GROUP dims, rounded up so that no value clips.
"""

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_qsa_store_rows_fp8_repr = make_kernel_repr(
    "_qsa_store_rows_fp8",
    ["HEAD_DIM", "BLOCK_D", "HAS_SECOND"],
)

_qsa_store_rows_fp4_repr = make_kernel_repr(
    "_qsa_store_rows_fp4",
    ["HEAD_DIM", "GROUP"],
)

_qsa_paged_mqa_logits_fp4_repr = make_kernel_repr(
    "_qsa_paged_mqa_logits_fp4",
    ["NUM_HEADS", "HEAD_DIM", "COMPRESS_RATIO", "GROUP", "BLOCK_N"],
)

_qsa_paged_mqa_logits_repr = make_kernel_repr(
    "_qsa_paged_mqa_logits",
    ["NUM_HEADS", "HEAD_DIM", "COMPRESS_RATIO", "BLOCK_N", "BLOCK_D", "KEY_FP8"],
)

_qsa_paged_mqa_logits_mfma_repr = make_kernel_repr(
    "_qsa_paged_mqa_logits_mfma",
    ["NUM_HEADS", "HEAD_DIM", "COMPRESS_RATIO", "BMT", "BN", "KEY_FMT"],
)

_qsa_paged_mqa_logits_stacked_repr = make_kernel_repr(
    "_qsa_paged_mqa_logits_stacked",
    ["NUM_HEADS", "HEAD_DIM", "COMPRESS_RATIO", "BMT", "BN", "KEY_FMT"],
)

_expand_selected_groups_repr = make_kernel_repr(
    "_expand_selected_groups",
    ["RATIO", "BLOCK"],
)

_qsa_sparse_paged_gqa_repr = make_kernel_repr(
    "_qsa_sparse_paged_gqa",
    [
        "TOPK",
        "NUM_KV_HEADS",
        "GROUP_SIZE",
        "HEAD_DIM",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_D",
        "KV_SPLITS",
        "KV_FP8",
    ],
)


@triton.jit
def _e2m1_value_lo(code):
    """FP32 value of E2M1 `code` (int32, 0..15) times 2^-14, decoded with an
    FP16 bitcast. `_e8m0_value_hi` supplies the 2^14."""
    bits = ((code & 7) << 9) | ((code & 8) << 12)
    return bits.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _e8m0_value_hi(e):
    """2^(e - 127) times 2^14. It is exact for e <= 240, which the writer keeps."""
    return ((e.to(tl.int32) + 14) << 23).to(tl.float32, bitcast=True)


@triton.jit(repr=_qsa_store_rows_fp8_repr)
def _qsa_store_rows_fp8_kernel(
    cache_ptr,
    scale_ptr,
    slots_ptr,
    values_ptr,
    stride_cache_slot,
    stride_cache_head,
    stride_scale_slot,
    stride_scale_head,
    stride_value_row,
    stride_value_head,
    second_cache_ptr,
    second_scale_ptr,
    second_values_ptr,
    stride_second_value_row,
    stride_second_value_head,
    num_rows,
    num_slots,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP8_MAX_C: tl.constexpr,
    HAS_SECOND: tl.constexpr,
) -> None:
    """FP8 rows with one FP32 scale per (row, head). An optional second tensor
    (V with K) uses the same slots and strides."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_slots)
    if not valid:
        return
    slot = slot.to(tl.int64)
    cache_offset = slot * stride_cache_slot + head * stride_cache_head + dims
    scale_offset = slot * stride_scale_slot + head * stride_scale_head
    x = tl.load(
        values_ptr + row * stride_value_row + head * stride_value_head + dims,
        mask=dims < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1.0e-12)
    y = tl.clamp(x * (FP8_MAX_C / amax), -FP8_MAX_C, FP8_MAX_C)
    tl.store(
        cache_ptr + cache_offset,
        y.to(cache_ptr.dtype.element_ty),
        mask=dims < HEAD_DIM,
    )
    tl.store(scale_ptr + scale_offset, amax / FP8_MAX_C)
    if HAS_SECOND:
        x = tl.load(
            second_values_ptr
            + row * stride_second_value_row
            + head * stride_second_value_head
            + dims,
            mask=dims < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1.0e-12)
        y = tl.clamp(x * (FP8_MAX_C / amax), -FP8_MAX_C, FP8_MAX_C)
        tl.store(
            second_cache_ptr + cache_offset,
            y.to(second_cache_ptr.dtype.element_ty),
            mask=dims < HEAD_DIM,
        )
        tl.store(second_scale_ptr + scale_offset, amax / FP8_MAX_C)


@triton.jit(repr=_qsa_store_rows_fp4_repr)
def _qsa_store_rows_fp4_kernel(
    cache_ptr,
    scale_ptr,
    slots_ptr,
    values_ptr,
    stride_cache_slot,
    stride_cache_head,
    stride_scale_slot,
    stride_scale_head,
    stride_value_row,
    stride_value_head,
    num_rows,
    num_slots,
    HEAD_DIM: tl.constexpr,
    GROUP: tl.constexpr,
) -> None:
    """MXFP4 rows: packed E2M1 pairs plus one E8M0 exponent per GROUP dims."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_slots)
    if not valid:
        return
    slot = slot.to(tl.int64)
    HALF: tl.constexpr = HEAD_DIM // 2
    GROUPS: tl.constexpr = HEAD_DIM // GROUP
    dims = tl.arange(0, HEAD_DIM)
    x = tl.load(
        values_ptr + row * stride_value_row + head * stride_value_head + dims[None, :]
    ).to(tl.float32)
    # Ties away from zero match ATOM's cache bytes. The E8M0 cap of 240 is
    # what the readers' FP16 decode needs.
    packed, exponents = _mxfp4_quant_op(
        x,
        HEAD_DIM,
        1,
        GROUP,
        SCALING_MODE=1,
        ROUNDING_MODE=1,
        MAX_SCALE_E8M0=240,
    )
    tl.store(
        cache_ptr
        + slot * stride_cache_slot
        + head * stride_cache_head
        + tl.arange(0, HALF),
        tl.reshape(packed, (HALF,)),
    )
    tl.store(
        scale_ptr
        + slot * stride_scale_slot
        + head * stride_scale_head
        + tl.arange(0, GROUPS),
        tl.reshape(exponents, (GROUPS,)),
    )


@triton.jit
def _load_q_row(q_ptr, token, head, stride_q_token, stride_q_head, dims, mask):
    return tl.load(
        q_ptr + token * stride_q_token + head * stride_q_head + dims,
        mask=mask,
        other=0.0,
    ).to(tl.float32)


@triton.jit(repr=_qsa_paged_mqa_logits_fp4_repr)
def _qsa_paged_mqa_logits_fp4_kernel(
    q_ptr,
    k_cache_ptr,
    k_scale_ptr,
    page_table_ptr,
    token_to_request_ptr,
    query_positions_ptr,
    context_lens_ptr,
    visible_groups_ptr,
    row_starts_ptr,
    logits_ptr,
    stride_q_token,
    stride_q_head,
    stride_cache_page,
    stride_cache_token,
    stride_scale_page,
    stride_scale_token,
    stride_table_request,
    stride_table_page,
    stride_logits_token,
    num_tokens,
    num_columns,
    num_cache_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """`_qsa_paged_mqa_logits_kernel` for MXFP4 keys. The even and odd Q dims
    multiply the low and high nibbles of each key byte."""
    token = tl.program_id(0)
    HALF: tl.constexpr = HEAD_DIM // 2
    pairs = tl.arange(0, HALF)

    request = tl.load(token_to_request_ptr + token)
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + token)
    context_len = tl.load(
        context_lens_ptr + safe_request,
        mask=request_valid,
        other=0,
    )
    visible_groups = tl.maximum(
        0,
        tl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            context_len // COMPRESS_RATIO,
        ),
    )
    if tl.program_id(1) == 0:
        tl.store(visible_groups_ptr + token, visible_groups)
        if row_starts_ptr is not None:
            tl.store(row_starts_ptr + token, 0)
    even = 2 * pairs
    m1 = NUM_HEADS > 1
    m2 = NUM_HEADS > 2
    m3 = NUM_HEADS > 3
    q0e = _load_q_row(q_ptr, token, 0, stride_q_token, stride_q_head, even, True)
    q0o = _load_q_row(q_ptr, token, 0, stride_q_token, stride_q_head, even + 1, True)
    q1e = _load_q_row(q_ptr, token, 1, stride_q_token, stride_q_head, even, m1)
    q1o = _load_q_row(q_ptr, token, 1, stride_q_token, stride_q_head, even + 1, m1)
    q2e = _load_q_row(q_ptr, token, 2, stride_q_token, stride_q_head, even, m2)
    q2o = _load_q_row(q_ptr, token, 2, stride_q_token, stride_q_head, even + 1, m2)
    q3e = _load_q_row(q_ptr, token, 3, stride_q_token, stride_q_head, even, m3)
    q3o = _load_q_row(q_ptr, token, 3, stride_q_token, stride_q_head, even + 1, m3)
    tl.static_assert(NUM_HEADS <= 4)
    num_tiles = tl.cdiv(tl.minimum(visible_groups, num_columns), BLOCK_N)
    for tile in range(tl.program_id(1), num_tiles, tl.num_programs(1)):
        columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        logical_page = columns // PAGE_SIZE
        page_offset = columns % PAGE_SIZE
        valid = (
            (token < num_tokens)
            & (columns < num_columns)
            & (columns < visible_groups)
            & request_valid
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        safe_logical_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
        physical_page = tl.load(
            page_table_ptr
            + safe_request * stride_table_request
            + safe_logical_page * stride_table_page,
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_pages)
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        packed = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_page
            + page_offset[:, None] * stride_cache_token
            + pairs[None, :],
            mask=valid[:, None],
            other=0,
        ).to(tl.int32)
        exponent = tl.load(
            k_scale_ptr
            + safe_physical_page[:, None] * stride_scale_page
            + page_offset[:, None] * stride_scale_token
            + (pairs // (GROUP // 2))[None, :],
            mask=valid[:, None],
            other=127,
        )
        group_scale = _e8m0_value_hi(exponent)
        low = _e2m1_value_lo(packed & 15) * group_scale
        high = _e2m1_value_lo(packed >> 4) * group_scale
        score = tl.maximum(
            tl.sum(low * q0e[None, :] + high * q0o[None, :], axis=1), 0.0
        )
        if NUM_HEADS > 1:
            score += tl.maximum(
                tl.sum(low * q1e[None, :] + high * q1o[None, :], axis=1), 0.0
            )
        if NUM_HEADS > 2:
            score += tl.maximum(
                tl.sum(low * q2e[None, :] + high * q2o[None, :], axis=1), 0.0
            )
        if NUM_HEADS > 3:
            score += tl.maximum(
                tl.sum(low * q3e[None, :] + high * q3o[None, :], axis=1), 0.0
            )
        score /= score_divisor
        tl.store(
            logits_ptr + token * stride_logits_token + columns,
            tl.where(valid, score, -float("inf")),
            mask=(token < num_tokens) & (columns < num_columns),
        )


@triton.jit(repr=_qsa_paged_mqa_logits_repr)
def _qsa_paged_mqa_logits_kernel(
    q_ptr,
    k_cache_ptr,
    k_scale_ptr,
    page_table_ptr,
    token_to_request_ptr,
    query_positions_ptr,
    context_lens_ptr,
    visible_groups_ptr,
    row_starts_ptr,
    logits_ptr,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_cache_page,
    stride_cache_token,
    stride_cache_dim,
    stride_scale_page,
    stride_scale_token,
    stride_table_request,
    stride_table_page,
    stride_logits_token,
    num_tokens,
    num_columns,
    num_cache_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    KEY_FP8: tl.constexpr = False,
) -> None:
    """Score BF16 (or per-row-scaled FP8) compressed QSA keys from a paged cache."""
    token = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)

    request = tl.load(token_to_request_ptr + token)
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + token)
    context_len = tl.load(
        context_lens_ptr + safe_request,
        mask=request_valid,
        other=0,
    )
    visible_groups = tl.maximum(
        0,
        tl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            context_len // COMPRESS_RATIO,
        ),
    )
    if tl.program_id(1) == 0:
        tl.store(visible_groups_ptr + token, visible_groups)
        if row_starts_ptr is not None:
            tl.store(row_starts_ptr + token, 0)
    q0 = tl.load(
        q_ptr + token * stride_q_token + dims * stride_q_dim,
        mask=dims < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    q1 = tl.load(
        q_ptr + token * stride_q_token + stride_q_head + dims * stride_q_dim,
        mask=(dims < HEAD_DIM) & (NUM_HEADS > 1),
        other=0.0,
    ).to(tl.float32)
    q2 = tl.load(
        q_ptr + token * stride_q_token + 2 * stride_q_head + dims * stride_q_dim,
        mask=(dims < HEAD_DIM) & (NUM_HEADS > 2),
        other=0.0,
    ).to(tl.float32)
    q3 = tl.load(
        q_ptr + token * stride_q_token + 3 * stride_q_head + dims * stride_q_dim,
        mask=(dims < HEAD_DIM) & (NUM_HEADS > 3),
        other=0.0,
    ).to(tl.float32)
    tl.static_assert(NUM_HEADS <= 4)
    num_tiles = tl.cdiv(tl.minimum(visible_groups, num_columns), BLOCK_N)
    for tile in range(tl.program_id(1), num_tiles, tl.num_programs(1)):
        columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        logical_page = columns // PAGE_SIZE
        page_offset = columns % PAGE_SIZE
        valid = (
            (token < num_tokens)
            & (columns < num_columns)
            & (columns < visible_groups)
            & request_valid
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        safe_logical_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
        physical_page = tl.load(
            page_table_ptr
            + safe_request * stride_table_request
            + safe_logical_page * stride_table_page,
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_pages)
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_page
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        score = tl.maximum(tl.sum(keys * q0[None, :], axis=1), 0.0)
        if NUM_HEADS > 1:
            score += tl.maximum(tl.sum(keys * q1[None, :], axis=1), 0.0)
        if NUM_HEADS > 2:
            score += tl.maximum(tl.sum(keys * q2[None, :], axis=1), 0.0)
        if NUM_HEADS > 3:
            score += tl.maximum(tl.sum(keys * q3[None, :], axis=1), 0.0)
        if KEY_FP8:
            # The row scale is positive, so it commutes with every head's ReLU.
            score *= tl.load(
                k_scale_ptr
                + safe_physical_page * stride_scale_page
                + page_offset * stride_scale_token,
                mask=valid,
                other=0.0,
            )
        score /= score_divisor
        tl.store(
            logits_ptr + token * stride_logits_token + columns,
            tl.where(valid, score, -float("inf")),
            mask=(token < num_tokens) & (columns < num_columns),
        )


@triton.jit(repr=_qsa_paged_mqa_logits_mfma_repr)
def _qsa_paged_mqa_logits_mfma_kernel(
    q_ptr,
    k_cache_ptr,
    k_scale_ptr,
    page_table_ptr,
    token_to_request_ptr,
    query_positions_ptr,
    context_lens_ptr,
    visible_groups_ptr,
    row_starts_ptr,
    logits_ptr,
    stride_q_token,
    stride_q_head,
    stride_cache_page,
    stride_cache_token,
    stride_scale_page,
    stride_scale_token,
    stride_table_request,
    stride_logits_token,
    num_tokens,
    num_columns,
    num_cache_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BMT: tl.constexpr,
    BN: tl.constexpr,
    KEY_FMT: tl.constexpr = 0,
    GROUP: tl.constexpr = 32,
) -> None:
    """`_qsa_paged_mqa_logits_kernel` on matrix cores."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BMT + tl.arange(0, BMT)
    rvalid = rows < num_tokens
    request = tl.load(token_to_request_ptr + rows, mask=rvalid, other=-1)
    request_ok = rvalid & (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    position = tl.load(query_positions_ptr + rows, mask=rvalid, other=-1)
    context = tl.load(context_lens_ptr + safe_request, mask=request_ok, other=0)
    visible = tl.maximum(
        0,
        tl.minimum((position + 1) // COMPRESS_RATIO, context // COMPRESS_RATIO),
    )
    visible = tl.where(request_ok, visible, 0)
    if pid_n == 0:
        tl.store(visible_groups_ptr + rows, visible, mask=rvalid)
        if row_starts_ptr is not None:
            tl.store(row_starts_ptr + rows, 0, mask=rvalid)
    col0 = pid_n * BN
    if col0 >= tl.max(visible, axis=0):
        return

    dims = tl.arange(0, HEAD_DIM)
    q_rows = q_ptr + rows[:, None].to(tl.int64) * stride_q_token + dims[None, :]
    columns = col0 + tl.arange(0, BN)
    logical_page = columns // PAGE_SIZE
    page_offset = columns % PAGE_SIZE
    page_ok = (logical_page < PAGE_TABLE_WIDTH) & (columns < num_columns)
    safe_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
    active = visible > 0
    req_lo = tl.min(tl.where(active, safe_request, num_requests), axis=0)
    req_hi = tl.max(tl.where(active, safe_request, -1), axis=0)
    score = tl.zeros((BMT, BN), dtype=tl.float32)
    HALF: tl.constexpr = HEAD_DIM // 2
    pairs = tl.arange(0, HALF)
    q_even = q_ptr + rows[:, None].to(tl.int64) * stride_q_token + 2 * pairs[None, :]
    for req in range(req_lo, req_hi + 1):
        physical = tl.load(
            page_table_ptr + req.to(tl.int64) * stride_table_request + safe_page,
            mask=page_ok,
            other=-1,
        )
        key_ok = page_ok & (physical >= 0) & (physical < num_cache_pages)
        safe_physical = tl.maximum(physical, 0).to(tl.int64)
        s = tl.zeros((BMT, BN), dtype=tl.float32)
        if KEY_FMT == 2:
            # Decode each nibble plane to Q's dtype (exact). Two dots multiply
            # them by the even and odd Q dims.
            packed = tl.load(
                k_cache_ptr
                + safe_physical[None, :] * stride_cache_page
                + page_offset[None, :] * stride_cache_token
                + pairs[:, None],
                mask=key_ok[None, :],
                other=0,
            ).to(tl.int32)
            exponent = tl.load(
                k_scale_ptr
                + safe_physical[None, :] * stride_scale_page
                + page_offset[None, :] * stride_scale_token
                + (pairs // (GROUP // 2))[:, None],
                mask=key_ok[None, :],
                other=127,
            )
            group_scale = _e8m0_value_hi(exponent)
            low_t = (_e2m1_value_lo(packed & 15) * group_scale).to(
                q_ptr.dtype.element_ty
            )
            high_t = (_e2m1_value_lo(packed >> 4) * group_scale).to(
                q_ptr.dtype.element_ty
            )
            for head in tl.static_range(NUM_HEADS):
                qe = tl.load(
                    q_even + head * stride_q_head, mask=rvalid[:, None], other=0.0
                )
                qo = tl.load(
                    q_even + head * stride_q_head + 1, mask=rvalid[:, None], other=0.0
                )
                s += tl.maximum(tl.dot(qo, high_t, acc=tl.dot(qe, low_t)), 0.0)
        else:
            keys_t = tl.load(
                k_cache_ptr
                + safe_physical[None, :] * stride_cache_page
                + page_offset[None, :] * stride_cache_token
                + dims[:, None],
                mask=key_ok[None, :],
                other=0.0,
            ).to(q_ptr.dtype.element_ty)
            for head in tl.static_range(NUM_HEADS):
                qh = tl.load(
                    q_rows + head * stride_q_head, mask=rvalid[:, None], other=0.0
                )
                s += tl.maximum(tl.dot(qh, keys_t), 0.0)
            if KEY_FMT == 1:
                # The row scale is positive, so it commutes with every ReLU.
                s *= tl.load(
                    k_scale_ptr
                    + safe_physical * stride_scale_page
                    + page_offset * stride_scale_token,
                    mask=key_ok,
                    other=0.0,
                )[None, :]
        score = tl.where((safe_request == req)[:, None], s, score)
    score = score / score_divisor
    tl.store(
        logits_ptr
        + rows[:, None].to(tl.int64) * stride_logits_token
        + columns[None, :],
        tl.where(columns[None, :] < visible[:, None], score, -float("inf")),
        mask=rvalid[:, None] & (columns < num_columns)[None, :],
    )


@triton.jit(repr=_qsa_paged_mqa_logits_stacked_repr)
def _qsa_paged_mqa_logits_stacked_kernel(
    q_ptr,
    k_cache_ptr,
    k_scale_ptr,
    page_table_ptr,
    token_to_request_ptr,
    query_positions_ptr,
    context_lens_ptr,
    visible_groups_ptr,
    row_starts_ptr,
    logits_ptr,
    stride_q_token,
    stride_q_head,
    stride_cache_page,
    stride_cache_token,
    stride_scale_page,
    stride_scale_token,
    stride_table_request,
    stride_logits_token,
    num_tokens,
    num_columns,
    num_cache_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BMT: tl.constexpr,
    BN: tl.constexpr,
    KEY_FMT: tl.constexpr,
    GROUP: tl.constexpr,
) -> None:
    """`_qsa_paged_mqa_logits_mfma_kernel` with the heads stacked into M.

    BMT rows times NUM_HEADS heads make one M operand, so one dot reads each
    key tile for all heads of all rows. The ReLU is applied to each head
    before the heads are added. Keys are BF16, FP8 or MXFP4.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BMT + tl.arange(0, BMT)
    rvalid = rows < num_tokens
    request = tl.load(token_to_request_ptr + rows, mask=rvalid, other=-1)
    request_ok = rvalid & (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    position = tl.load(query_positions_ptr + rows, mask=rvalid, other=-1)
    context = tl.load(context_lens_ptr + safe_request, mask=request_ok, other=0)
    visible = tl.maximum(
        0,
        tl.minimum((position + 1) // COMPRESS_RATIO, context // COMPRESS_RATIO),
    )
    visible = tl.where(request_ok, visible, 0)
    if pid_n == 0:
        tl.store(visible_groups_ptr + rows, visible, mask=rvalid)
        if row_starts_ptr is not None:
            tl.store(row_starts_ptr + rows, 0, mask=rvalid)
    # Each program scores the column tiles pid_n, pid_n + G, ... (G = grid dim 1).
    num_tiles = tl.cdiv(tl.minimum(tl.max(visible, axis=0), num_columns), BN)
    if pid_n >= num_tiles:
        return

    M: tl.constexpr = BMT * NUM_HEADS
    stacked = tl.arange(0, M)
    stacked_row = pid_m * BMT + stacked // NUM_HEADS
    q_base = (
        q_ptr
        + stacked_row.to(tl.int64) * stride_q_token
        + (stacked % NUM_HEADS) * stride_q_head
    )
    q_ok = (stacked_row < num_tokens)[:, None]
    HALF: tl.constexpr = HEAD_DIM // 2
    pairs = tl.arange(0, HALF)
    dims = tl.arange(0, HEAD_DIM)
    if KEY_FMT == 2:
        q_lo = tl.load(q_base[:, None] + 2 * pairs[None, :], mask=q_ok, other=0.0)
        q_hi = tl.load(q_base[:, None] + 2 * pairs[None, :] + 1, mask=q_ok, other=0.0)
    else:
        q_all = tl.load(q_base[:, None] + dims[None, :], mask=q_ok, other=0.0)

    active = visible > 0
    req_lo = tl.min(tl.where(active, safe_request, num_requests), axis=0)
    req_hi = tl.max(tl.where(active, safe_request, -1), axis=0)
    for tile in range(pid_n, num_tiles, tl.num_programs(1)):
        columns = tile * BN + tl.arange(0, BN)
        logical_page = columns // PAGE_SIZE
        page_offset = columns % PAGE_SIZE
        page_ok = (logical_page < PAGE_TABLE_WIDTH) & (columns < num_columns)
        safe_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
        score = tl.zeros((BMT, BN), dtype=tl.float32)
        for req in range(req_lo, req_hi + 1):
            physical = tl.load(
                page_table_ptr + req.to(tl.int64) * stride_table_request + safe_page,
                mask=page_ok,
                other=-1,
            )
            key_ok = page_ok & (physical >= 0) & (physical < num_cache_pages)
            safe_physical = tl.maximum(physical, 0).to(tl.int64)
            if KEY_FMT == 2:
                packed = tl.load(
                    k_cache_ptr
                    + safe_physical[None, :] * stride_cache_page
                    + page_offset[None, :] * stride_cache_token
                    + pairs[:, None],
                    mask=key_ok[None, :],
                    other=0,
                ).to(tl.int32)
                exponent = tl.load(
                    k_scale_ptr
                    + safe_physical[None, :] * stride_scale_page
                    + page_offset[None, :] * stride_scale_token
                    + (pairs // (GROUP // 2))[:, None],
                    mask=key_ok[None, :],
                    other=127,
                )
                group_scale = _e8m0_value_hi(exponent)
                low_t = (_e2m1_value_lo(packed & 15) * group_scale).to(
                    q_ptr.dtype.element_ty
                )
                high_t = (_e2m1_value_lo(packed >> 4) * group_scale).to(
                    q_ptr.dtype.element_ty
                )
                s = tl.dot(q_hi, high_t, acc=tl.dot(q_lo, low_t))
            else:
                keys_t = tl.load(
                    k_cache_ptr
                    + safe_physical[None, :] * stride_cache_page
                    + page_offset[None, :] * stride_cache_token
                    + dims[:, None],
                    mask=key_ok[None, :],
                    other=0.0,
                ).to(q_ptr.dtype.element_ty)
                s = tl.dot(q_all, keys_t)
            s = tl.sum(tl.reshape(tl.maximum(s, 0.0), (BMT, NUM_HEADS, BN)), axis=1)
            if KEY_FMT == 1:
                # The row scale is positive, so it commutes with every ReLU.
                s *= tl.load(
                    k_scale_ptr
                    + safe_physical * stride_scale_page
                    + page_offset * stride_scale_token,
                    mask=key_ok,
                    other=0.0,
                )[None, :]
            score = tl.where((safe_request == req)[:, None], s, score)
        score = score / score_divisor
        tl.store(
            logits_ptr
            + rows[:, None].to(tl.int64) * stride_logits_token
            + columns[None, :],
            tl.where(columns[None, :] < visible[:, None], score, -float("inf")),
            mask=rvalid[:, None] & (columns < num_columns)[None, :],
        )


@triton.jit(repr=_expand_selected_groups_repr)
def _expand_selected_groups(
    Groups,
    Context,
    Positions,
    Requests,
    Out,
    num_requests,
    n_groups,
    out_cols,
    group_s0,
    group_s1,
    out_s0,
    out_s1,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    req = tl.load(Requests + row)
    valid_req = (req >= 0) & (req < num_requests)
    context = tl.load(Context + req, valid_req, 0)
    pos = tl.load(Positions + row)
    length = tl.where(valid_req & (pos >= 0) & (pos < context), pos + 1, 0)
    full_groups = length // RATIO
    tail_start = full_groups * RATIO
    history_end = tl.minimum(full_groups, n_groups) * RATIO
    is_history = cols < history_end
    group = tl.load(
        Groups + row * group_s0 + (cols // RATIO) * group_s1,
        (cols < out_cols) & is_history,
        -1,
    )
    history = tl.where(
        (group >= 0) & (group < full_groups), group * RATIO + cols % RATIO, -1
    )
    tail_offset = cols - history_end
    tail = tl.where(
        (tail_offset >= 0) & (tail_offset < length - tail_start),
        tail_start + tail_offset,
        -1,
    )
    tl.store(
        Out + row * out_s0 + cols * out_s1,
        tl.where(is_history, history, tail).to(tl.int32),
        cols < out_cols,
    )


@triton.jit(repr=_qsa_sparse_paged_gqa_repr)
def _qsa_sparse_paged_gqa_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    logical_indices_ptr,
    block_table_ptr,
    token_to_request_ptr,
    output_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    widths_ptr,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_k_page,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_page,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_scale_page,
    stride_scale_token,
    stride_scale_head,
    stride_indices_token,
    stride_indices_column,
    stride_table_request,
    stride_table_page,
    stride_output_token,
    stride_output_head,
    stride_output_dim,
    num_tokens,
    num_cache_pages,
    num_requests,
    softmax_scale,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    KV_SPLITS: tl.constexpr = 1,
    SUB_GROUP: tl.constexpr = 0,
    KV_FP8: tl.constexpr = False,
) -> None:
    """Apply GQA over arbitrary logical tokens in separate paged BF16 or FP8 K/V.

    FP8 K/V have one FP32 scale per (token, KV head). The key scale multiplies
    the scores. The value scale multiplies the probabilities before PV.
    """
    token = tl.program_id(0)
    part = tl.program_id(2)
    ACTIVE_SUB: tl.constexpr = SUB_GROUP if SUB_GROUP > 0 else GROUP_SIZE
    NUM_HEAD_GROUPS: tl.constexpr = (GROUP_SIZE + ACTIVE_SUB - 1) // ACTIVE_SUB
    TAIL: tl.constexpr = GROUP_SIZE % ACTIVE_SUB
    kv_head = tl.program_id(1) // NUM_HEAD_GROUPS
    head_group = tl.program_id(1) % NUM_HEAD_GROUPS
    if TAIL != 0:
        valid_sub = tl.where(head_group < NUM_HEAD_GROUPS - 1, ACTIVE_SUB, TAIL)
    else:
        valid_sub = ACTIVE_SUB
    if KV_SPLITS > 1:  # noqa: SIM102
        if tl.program_id(1) == 0 and part == 0:
            tl.store(widths_ptr + token, TOPK)
    request = tl.load(token_to_request_ptr + token)
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    first_q_head = kv_head * GROUP_SIZE + head_group * ACTIVE_SUB
    query = tl.load(
        q_ptr
        + token * stride_q_token
        + (first_q_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :] * stride_q_dim,
        mask=(head_offsets[:, None] < valid_sub) & (dim_offsets[None, :] < HEAD_DIM),
        other=0.0,
    )
    query = (query * softmax_scale * 1.4426950408889634).to(query.dtype)

    running_max = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    column_offsets = tl.arange(0, BLOCK_N)

    partition_size = tl.cdiv(TOPK, KV_SPLITS * BLOCK_N) * BLOCK_N
    if part * partition_size >= TOPK:
        return
    for start in tl.range(
        part * partition_size, tl.minimum((part + 1) * partition_size, TOPK), BLOCK_N
    ):
        columns = start + column_offsets
        logical_token = tl.load(
            logical_indices_ptr
            + token * stride_indices_token
            + columns * stride_indices_column,
            mask=columns < TOPK,
            other=-1,
        )
        safe_logical_token = tl.maximum(logical_token, 0)
        logical_page = safe_logical_token // PAGE_SIZE
        page_offset = safe_logical_token % PAGE_SIZE
        valid = (
            (token < num_tokens)
            & request_valid
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_request
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_pages)
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)

        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[None, :] * stride_k_page
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None] * stride_k_dim,
            mask=(dim_offsets[:, None] < HEAD_DIM) & valid[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        values = tl.load(
            v_cache_ptr
            + safe_physical_page[:, None] * stride_v_page
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :] * stride_v_dim,
            mask=valid[:, None] & (dim_offsets[None, :] < HEAD_DIM),
            other=0.0,
            cache_modifier=".cg",
        )

        if KV_FP8:
            scale_offset = (
                safe_physical_page * stride_scale_page
                + page_offset * stride_scale_token
                + kv_head * stride_scale_head
            )
            key_scale = tl.load(k_scale_ptr + scale_offset, mask=valid, other=0.0)
            value_scale = tl.load(v_scale_ptr + scale_offset, mask=valid, other=0.0)
            qk = tl.dot(query, keys.to(query.dtype))
            scores = tl.where(valid[None, :], qk * key_scale[None, :], -1.0e20)
        else:
            scores = tl.where(valid[None, :], tl.dot(query, keys), -1.0e20)
        next_max = tl.maximum(running_max, tl.max(scores, axis=1))
        alpha = tl.math.exp2(running_max - next_max)
        probabilities = tl.where(
            valid[None, :],
            tl.math.exp2(scores - next_max[:, None]),
            0.0,
        )
        if KV_FP8:
            accumulator = tl.dot(
                (probabilities * value_scale[None, :]).to(query.dtype),
                values.to(query.dtype),
                acc=accumulator * alpha[:, None],
            )
        else:
            accumulator = tl.dot(
                probabilities.to(values.dtype),
                values,
                acc=accumulator * alpha[:, None],
            )
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
        running_max = next_max

    if KV_SPLITS > 1:
        partial_offset = (
            token * NUM_KV_HEADS * GROUP_SIZE + first_q_head + head_offsets
        ) * KV_SPLITS + part
        tl.store(
            partial_max_ptr + partial_offset, running_max, head_offsets < valid_sub
        )
        tl.store(
            partial_sum_ptr + partial_offset, running_sum, head_offsets < valid_sub
        )
        output_ptr += part * HEAD_DIM
        output = accumulator
    else:
        output = tl.where(
            running_sum[:, None] > 0,
            accumulator / tl.maximum(running_sum[:, None], 1.0e-20),
            0.0,
        )
    tl.store(
        output_ptr
        + token * stride_output_token
        + (first_q_head + head_offsets[:, None]) * stride_output_head
        + dim_offsets[None, :] * stride_output_dim,
        output,
        mask=(token < num_tokens)
        & (head_offsets[:, None] < valid_sub)
        & (dim_offsets[None, :] < HEAD_DIM),
    )
