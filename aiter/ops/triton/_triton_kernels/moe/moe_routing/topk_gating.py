# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Caller-owned MoE gating kernels without bitmatrix routing metadata."""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _exact_sigmoid_biased_topk(
    logits_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    num_valid_rows_ptr,  # [1] int32: rows at or past it route to -1
    stride_logits,
    stride_weights,
    stride_ids,
    n_rows,
    routed_scaling,
    N_EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,  # next power of two of TOPK
    BLOCK_N: tl.constexpr,
    NEED_RENORM: tl.constexpr,
    HAS_NUM_VALID_ROWS: tl.constexpr,
):
    """Match the gfx1250 HIP fp32/int32 sigmoid-gating contract."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = (row < n_rows) & (cols < N_EXPERTS)
    logits = tl.load(logits_ptr + row * stride_logits + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scores = 1.0 / (1.0 + tl.exp(-logits))
    choice = tl.where(mask, scores + bias, float("-inf"))
    choice = tl.where(choice == choice, choice, float("-inf"))  # noqa: PLR0124

    slots = tl.arange(0, TOPK_PAD)
    selected_ids = tl.zeros((TOPK_PAD,), dtype=tl.int32)
    selected_weights = tl.zeros((TOPK_PAD,), dtype=tl.float32)

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
        weight_sum = tl.sum(selected_weights, axis=0)
        norm_scale = tl.where(weight_sum > 0.0, routed_scaling / weight_sum, 0.0)
        selected_weights *= norm_scale
    else:
        selected_weights *= routed_scaling
    if HAS_NUM_VALID_ROWS:
        pad = row >= tl.load(num_valid_rows_ptr)
        selected_ids = tl.where(pad, -1, selected_ids)
        selected_weights = tl.where(pad, 0.0, selected_weights)

    slot_mask = (row < n_rows) & (slots < TOPK)
    tl.store(ids_ptr + row * stride_ids + slots, selected_ids, mask=slot_mask)
    tl.store(
        weights_ptr + row * stride_weights + slots,
        selected_weights,
        mask=slot_mask,
    )


@triton.jit
def _sqrtsoftplus(x):
    # compute_score<SCORE_SQRTSOFTPLUS> of the HIP topk_gating kernels.
    x = tl.where(x > 3.4028234663852886e38, 3.4028234663852886e38, x)
    softplus = tl.where(
        x > 20.0,
        x,
        tl.math.log2(1.0 + tl.math.exp2(x * 1.4426950408889634)) * 0.6931471805599453,
    )
    return tl.sqrt_rn(softplus)


@triton.jit
def _sqrtsoftplus_384_slot(row_ptr, bias_ptr, cols, HAS_BIAS: tl.constexpr):
    """Selection value, weight and expert id of one expert in each lane."""
    score = _sqrtsoftplus(tl.load(row_ptr + cols).to(tl.float32))
    weight = tl.where(score != score, 0.0, score)  # noqa: PLR0124
    if HAS_BIAS:
        score = score + tl.load(bias_ptr + cols)
    choice = tl.where(score != score, float("-inf"), score)  # noqa: PLR0124
    return choice, weight, cols


@triton.jit
def _compare_exchange(va, wa, ia, vb, wb, ib):
    # Descending: swap when va < vb. A NaN compares false, as in the HIP sort.
    swap = va < vb
    return (
        tl.where(swap, vb, va),
        tl.where(swap, wb, wa),
        tl.where(swap, ib, ia),
        tl.where(swap, va, vb),
        tl.where(swap, wa, wb),
        tl.where(swap, ia, ib),
    )


@triton.jit
def _pick3(cursor, a0, a1, a2, other):
    return tl.where(
        cursor == 0, a0, tl.where(cursor == 1, a1, tl.where(cursor == 2, a2, other))
    )


@triton.jit
def _pick6(cursor, a0, a1, a2, a3, a4, a5, other):
    return tl.where(
        cursor < 3,
        _pick3(cursor, a0, a1, a2, other),
        _pick3(cursor - 3, a3, a4, a5, other),
    )


_sqrtsoftplus_384_topk_repr = make_kernel_repr(
    "_sqrtsoftplus_384_topk",
    ["TOPK", "HAS_BIAS", "HAS_NUM_VALID_ROWS", "WAVES", "LANES", "EPT"],
)


@triton.jit(repr=_sqrtsoftplus_384_topk_repr)
def _sqrtsoftplus_384_topk(
    logits_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    num_valid_rows_ptr,  # [1] int32: rows at or past it route to -1
    stride_logits,
    stride_weights,
    stride_ids,
    routed_scaling,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,  # next power of two of TOPK
    HAS_BIAS: tl.constexpr,
    HAS_NUM_VALID_ROWS: tl.constexpr,
    WAVES: tl.constexpr,
    LANES: tl.constexpr,
    EPT: tl.constexpr,
):
    """Renormalized sqrtsoftplus top-k of one row of 384 experts.

    Uses the lane layout of the HIP kernel that topk_gating runs for 384
    experts. Lane c of the C = WAVES * LANES lanes holds experts c, c + C, ...
    and sorts them with the same network. Each wave takes TOPK rounds of a lane
    argmax (lowest lane on a tie). The wave lists are then merged with a strict
    compare, wave 0 first. So the ids, the tie order and the renorm sum order
    are the same as there.
    """
    C: tl.constexpr = WAVES * LANES
    row = tl.program_id(0)
    row_ptr = logits_ptr + row * stride_logits
    cols = tl.arange(0, C)
    s0, w0, i0 = _sqrtsoftplus_384_slot(row_ptr, bias_ptr, cols, HAS_BIAS)
    s1, w1, i1 = _sqrtsoftplus_384_slot(row_ptr, bias_ptr, cols + C, HAS_BIAS)
    s2, w2, i2 = _sqrtsoftplus_384_slot(row_ptr, bias_ptr, cols + 2 * C, HAS_BIAS)
    if EPT == 3:
        # sort_network_desc<3>
        s0, w0, i0, s1, w1, i1 = _compare_exchange(s0, w0, i0, s1, w1, i1)
        s0, w0, i0, s2, w2, i2 = _compare_exchange(s0, w0, i0, s2, w2, i2)
        s1, w1, i1, s2, w2, i2 = _compare_exchange(s1, w1, i1, s2, w2, i2)
    else:
        s3, w3, i3 = _sqrtsoftplus_384_slot(row_ptr, bias_ptr, cols + 3 * C, HAS_BIAS)
        s4, w4, i4 = _sqrtsoftplus_384_slot(row_ptr, bias_ptr, cols + 4 * C, HAS_BIAS)
        s5, w5, i5 = _sqrtsoftplus_384_slot(row_ptr, bias_ptr, cols + 5 * C, HAS_BIAS)
        # sort_network_desc<6>: its 12 comparators in order
        s0, w0, i0, s1, w1, i1 = _compare_exchange(s0, w0, i0, s1, w1, i1)
        s2, w2, i2, s3, w3, i3 = _compare_exchange(s2, w2, i2, s3, w3, i3)
        s4, w4, i4, s5, w5, i5 = _compare_exchange(s4, w4, i4, s5, w5, i5)
        s0, w0, i0, s2, w2, i2 = _compare_exchange(s0, w0, i0, s2, w2, i2)
        s1, w1, i1, s4, w4, i4 = _compare_exchange(s1, w1, i1, s4, w4, i4)
        s3, w3, i3, s5, w5, i5 = _compare_exchange(s3, w3, i3, s5, w5, i5)
        s0, w0, i0, s1, w1, i1 = _compare_exchange(s0, w0, i0, s1, w1, i1)
        s2, w2, i2, s3, w3, i3 = _compare_exchange(s2, w2, i2, s3, w3, i3)
        s4, w4, i4, s5, w5, i5 = _compare_exchange(s4, w4, i4, s5, w5, i5)
        s1, w1, i1, s2, w2, i2 = _compare_exchange(s1, w1, i1, s2, w2, i2)
        s3, w3, i3, s4, w4, i4 = _compare_exchange(s3, w3, i3, s4, w4, i4)
        s2, w2, i2, s3, w3, i3 = _compare_exchange(s2, w2, i2, s3, w3, i3)

    cursor = tl.zeros([C], dtype=tl.int32)
    lane = tl.arange(0, LANES)
    slots = tl.arange(0, TOPK_PAD)
    out_w = tl.zeros([TOPK_PAD], dtype=tl.float32)
    out_i = tl.zeros([TOPK_PAD], dtype=tl.int32)
    weight_sum = 0.0
    wave_v = tl.zeros([WAVES, TOPK_PAD], dtype=tl.float32)
    wave_w = tl.zeros([WAVES, TOPK_PAD], dtype=tl.float32)
    wave_i = tl.zeros([WAVES, TOPK_PAD], dtype=tl.int32)
    # Each wave: TOPK rounds of a lane argmax over the heads of the lane lists.
    for k in tl.static_range(TOPK):
        if EPT == 3:
            head_v = _pick3(cursor, s0, s1, s2, float("-inf"))
            head_w = _pick3(cursor, w0, w1, w2, 0.0)
            head_i = _pick3(cursor, i0, i1, i2, 0)
        else:
            head_v = _pick6(cursor, s0, s1, s2, s3, s4, s5, float("-inf"))
            head_w = _pick6(cursor, w0, w1, w2, w3, w4, w5, 0.0)
            head_i = _pick6(cursor, i0, i1, i2, i3, i4, i5, 0)
        head_v = tl.reshape(head_v, [WAVES, LANES])
        head_w = tl.reshape(head_w, [WAVES, LANES])
        head_i = tl.reshape(head_i, [WAVES, LANES])
        wave_cursor = tl.reshape(cursor, [WAVES, LANES])
        best = tl.max(head_v, axis=1)
        tied = head_v == best[:, None]
        win_lane = tl.min(tl.where(tied, lane[None, :], LANES), axis=1)
        # When no lane matches the best value, the HIP argmax keeps lane 0.
        win_lane = tl.where(win_lane == LANES, 0, win_lane)
        idx = tl.sum(tl.where(lane[None, :] == win_lane[:, None], head_i, 0), axis=1)
        won = (wave_cursor < EPT) & (head_i == idx[:, None])
        # The weight comes from the lane that holds expert idx.
        owner = lane[None, :] == (idx & (LANES - 1))[:, None]
        wt = tl.sum(tl.where(owner & won, head_w, 0.0), axis=1)
        cursor = tl.reshape(wave_cursor + won.to(tl.int32), [C])
        if WAVES == 1:
            # One wave: its list is the result, summed in selection order.
            wt1 = tl.sum(wt, axis=0)
            weight_sum = weight_sum + wt1
            out_w = tl.where(slots == k, wt1, out_w)
            out_i = tl.where(slots == k, tl.sum(idx, axis=0), out_i)
        else:
            at_k = slots[None, :] == k
            wave_v = tl.where(at_k, best[:, None], wave_v)
            wave_w = tl.where(at_k, wt[:, None], wave_w)
            wave_i = tl.where(at_k, idx[:, None], wave_i)
    if WAVES > 1:
        # Merge the wave lists: strict compare, wave 0 first.
        waves = tl.arange(0, WAVES)
        pos = tl.zeros([WAVES], dtype=tl.int32)
        for k in tl.static_range(TOPK):
            at_pos = slots[None, :] == pos[:, None]
            head_v = tl.sum(tl.where(at_pos, wave_v, 0.0), axis=1)
            head_w = tl.sum(tl.where(at_pos, wave_w, 0.0), axis=1)
            head_i = tl.sum(tl.where(at_pos, wave_i, 0), axis=1)
            best_v = tl.sum(tl.where(waves == 0, head_v, 0.0), axis=0)
            best_wave = tl.sum(tl.zeros([WAVES], dtype=tl.int32), axis=0)
            for w in tl.static_range(1, WAVES):
                v = tl.sum(tl.where(waves == w, head_v, 0.0), axis=0)
                up = v > best_v
                best_v = tl.where(up, v, best_v)
                best_wave = tl.where(up, w, best_wave)
            taken = waves == best_wave
            wt = tl.sum(tl.where(taken, head_w, 0.0), axis=0)
            eid = tl.sum(tl.where(taken, head_i, 0), axis=0)
            pos = pos + taken.to(tl.int32)
            weight_sum = weight_sum + wt
            out_w = tl.where(slots == k, wt, out_w)
            out_i = tl.where(slots == k, eid, out_i)

    # The sum has the same floor as in the HIP kernels (RENORM_SUM_FLOOR).
    out_w = out_w * tl.math.div_rn(routed_scaling, tl.maximum(weight_sum, 1e-20))
    if HAS_NUM_VALID_ROWS:
        pad = row >= tl.load(num_valid_rows_ptr)
        out_i = tl.where(pad, -1, out_i)
        out_w = tl.where(pad, 0.0, out_w)
    slot_mask = slots < TOPK
    tl.store(ids_ptr + row * stride_ids + slots, out_i, mask=slot_mask)
    tl.store(weights_ptr + row * stride_weights + slots, out_w, mask=slot_mask)
