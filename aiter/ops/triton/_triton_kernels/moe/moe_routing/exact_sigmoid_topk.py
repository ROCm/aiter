# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _exact_sigmoid_biased_topk(
    logits_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    stride_logits,
    stride_weights,
    stride_ids,
    n_rows,
    routed_scaling,
    N_EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_RENORM: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = (row < n_rows) & (cols < N_EXPERTS)
    logits = tl.load(logits_ptr + row * stride_logits + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scores = 1.0 / (1.0 + tl.exp(-logits))
    choice = tl.where(mask, scores + bias, float("-inf"))
    # The legacy gfx1250 HIP path ignores NaNs during its strict-greater scan.
    choice = tl.where(choice == choice, choice, float("-inf"))  # noqa: PLR0124

    slots = tl.arange(0, TOPK)
    selected_ids = tl.zeros((TOPK,), dtype=tl.int32)
    selected_weights = tl.zeros((TOPK,), dtype=tl.float32)

    # Match the legacy gfx1250 wave32 float4-LDS reduction order. Vectors are
    # assigned round-robin to lanes and local positions retain source order.
    lane = (cols // 4) % 32
    local_pos = (cols // 128) * 4 + cols % 4
    low2 = (lane & 3) ^ tl.where((lane & 4) != 0, 3, 0)
    bit2 = (((lane >> 2) ^ (lane >> 3)) & 1) << 2
    lane_rank = ((lane ^ 0x38) & 0x38) | bit2 | low2
    # Admitted expert counts use BLOCK_N <= 1024, so local_pos fits in 5 bits.
    tie_rank = lane_rank * 32 + local_pos

    for k in tl.static_range(TOPK):
        max_choice = tl.max(choice, axis=0)
        rank = tl.where(choice == max_choice, tie_rank, 0x7FFFFFFF)
        idx = tl.argmin(rank, axis=0, tie_break_left=True).to(tl.int32)
        value = tl.sum(tl.where(cols == idx, scores, 0.0), axis=0)
        selected_ids = tl.where(slots == k, idx, selected_ids)
        selected_weights = tl.where(slots == k, value, selected_weights)
        choice = tl.where(cols == idx, float("-inf"), choice)

    if NEED_RENORM:
        selected_weights *= routed_scaling / tl.sum(selected_weights, axis=0)
    else:
        selected_weights *= routed_scaling

    row_mask = row < n_rows
    tl.store(ids_ptr + row * stride_ids + slots, selected_ids, mask=row_mask)
    tl.store(
        weights_ptr + row * stride_weights + slots,
        selected_weights,
        mask=row_mask,
    )
