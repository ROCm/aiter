# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""QSA index-Q preparation and raw/compressed cache updates for gfx942."""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _rstd(x, n_cols, eps, R: tl.constexpr):
    # SGLang's Gemma RMSNorm reduces each 128-wide row with DPP row_shr 8/4/2/1, then row pairs,
    # then the two warp partials; summing index bits 3,2,1,0,4,5,6 in that order is bit-exact.
    s = tl.reshape(x * x, [R, 2, 2, 2, 2, 2, 2, 2])
    s = tl.sum(tl.sum(tl.sum(tl.sum(s, axis=4), axis=4), axis=4), axis=4)
    s = tl.sum(tl.sum(tl.sum(s, axis=3), axis=2), axis=1)
    return 1.0 / tl.sqrt(s / n_cols + eps)


@triton.jit
def _partner(cols, ROT: tl.constexpr):
    return tl.where(
        cols < ROT // 2, cols + ROT // 2, tl.where(cols < ROT, cols - ROT // 2, cols)
    )


@triton.jit
def _norm_rope(
    x,
    partner,
    n_cols,
    eps,
    Weight,
    Cache,
    Axis,
    p0,
    p1,
    p2,
    R: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
):
    # Rows of x are normed like SGLang, then NeoX-rotated with its per-op BF16 roundings; partner holds
    # each rotary column's other half (x[i + ROT/2] or x[i - ROT/2]).
    cols = tl.arange(0, D)
    half = cols % (ROT // 2)
    rot = cols < ROT
    rstd = _rstd(x, n_cols, eps, R)[:, None]
    y = (
        (x * rstd * (1.0 + tl.load(Weight + cols).to(tl.float32))[None, :])
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    yp = (
        partner
        * rstd
        * (1.0 + tl.load(Weight + _partner(cols, ROT)).to(tl.float32))[None, :]
    )
    yp = yp.to(tl.bfloat16).to(tl.float32)
    selector = tl.load(Axis + half, mask=rot, other=0)[None, :]
    pos = tl.where(
        selector == 0, p0[:, None], tl.where(selector == 1, p1[:, None], p2[:, None])
    )
    cache = Cache + pos * CACHE_STRIDE + half[None, :]
    c = tl.load(cache, mask=rot[None, :], other=0.0).to(tl.bfloat16).to(tl.float32)
    s = (
        tl.load(cache + ROT // 2, mask=rot[None, :], other=0.0)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    own = (y * c).to(tl.bfloat16).to(tl.float32)
    other = (yp * s).to(tl.bfloat16).to(tl.float32)
    out = tl.where((cols < ROT // 2)[None, :], own - other, own + other)
    return tl.where(rot[None, :], out, y).to(tl.bfloat16)


@triton.jit(
    do_not_specialize=["position_stride", "tokens"],
    repr=make_kernel_repr("_indexer_q_prep", ["TB", "H", "D", "ROT"]),
)
def _indexer_q_prep(
    QK,
    Q,
    Weight,
    KeyState,
    RopeState,
    Slots,
    Positions,
    position_stride,
    Cache,
    Axis,
    Stats,
    tokens,
    n_cols,
    eps,
    TB: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
):
    # position_stride is 0 for 1-D positions, so all three rope axes read the same row.
    first = tl.program_id(0) * TB
    # Per-row logits (min bits, ~max bits) start all ones; the logits kernel folds into them.
    lanes = tl.arange(0, 2)[None, :]
    rows = first + tl.arange(0, TB)[:, None]
    tl.store(Stats + rows * 2 + lanes, tl.full((TB, 2), -1, tl.int32), rows < tokens)
    cols = tl.arange(0, D)
    row = tl.arange(0, TB * H)
    token = first + row // H
    live = token < tokens
    p0 = tl.load(Positions + token, mask=live, other=0)
    p1 = tl.load(Positions + position_stride + token, mask=live, other=0)
    p2 = tl.load(Positions + 2 * position_stride + token, mask=live, other=0)
    src = QK + (token.to(tl.int64) * (H + 1) * D + row % H * D)[:, None]
    x = tl.load(src + cols[None, :], mask=live[:, None], other=0.0).to(tl.float32)
    partner = tl.load(
        src + _partner(cols, ROT)[None, :], mask=live[:, None], other=0.0
    ).to(tl.float32)
    q = _norm_rope(
        x,
        partner,
        n_cols,
        eps,
        Weight,
        Cache,
        Axis,
        p0,
        p1,
        p2,
        TB * H,
        D,
        ROT,
        CACHE_STRIDE,
    )
    tl.store(
        Q + (token.to(tl.int64) * H * D + row % H * D)[:, None] + cols[None, :],
        q,
        mask=live[:, None],
    )
    token = first + tl.arange(0, TB)
    live = token < tokens
    slot = tl.load(Slots + token, mask=live, other=0)
    key = tl.load(
        QK + (token.to(tl.int64) * (H + 1) * D + H * D)[:, None] + cols[None, :],
        mask=live[:, None],
    )
    tl.store(KeyState + slot[:, None] * D + cols[None, :], key, mask=live[:, None])
    lanes = tl.arange(0, 4)[None, :]
    axes = live[:, None] & (lanes < 3)
    value = tl.load(
        Positions + lanes * position_stride + token[:, None], mask=axes, other=0
    )
    tl.store(RopeState + slot[:, None] * 3 + lanes, value, mask=axes)


@triton.jit
def _group_mean(
    QK, first, step, cols, H: tl.constexpr, D: tl.constexpr, RATIO: tl.constexpr
):
    row = QK + (first * (H + 1) * D + H * D)[:, None] + cols[None, :]
    acc = tl.load(row).to(tl.float32)
    for member in tl.static_range(1, RATIO):
        acc = acc + tl.load(row + (member * step * (H + 1) * D)[:, None]).to(tl.float32)
    return (acc * (1.0 / RATIO)).to(tl.bfloat16).to(tl.float32)


@triton.jit(
    do_not_specialize=["groups"],
    repr=make_kernel_repr("_indexer_k_compress", ["GB", "H", "D", "ROT"]),
)
def _indexer_k_compress(
    QK,
    Rows,
    WriteLocs,
    RopeMatrix,
    Cache,
    Axis,
    Weight,
    Compressed,
    groups,
    n_cols,
    eps,
    GB: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    RATIO: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
):
    group = tl.program_id(0) * GB + tl.arange(0, GB)
    live = group < groups
    loc = tl.load(WriteLocs + group, mask=live, other=0).to(tl.int64)
    # Planner padding writes the inert slot 0; ROCm SGLang gathers member row 0 for it.
    first = tl.where(loc != 0, tl.load(Rows + group, mask=live, other=0), 0)
    step = tl.where(loc != 0, 1, 0)
    cols = tl.arange(0, D)
    x = _group_mean(QK, first, step, cols, H, D, RATIO)
    partner = _group_mean(QK, first, step, _partner(cols, ROT), H, D, RATIO)
    p0 = tl.load(RopeMatrix + first * 3)
    p1 = tl.load(RopeMatrix + first * 3 + 1)
    p2 = tl.load(RopeMatrix + first * 3 + 2)
    key = _norm_rope(
        x,
        partner,
        n_cols,
        eps,
        Weight,
        Cache,
        Axis,
        p0,
        p1,
        p2,
        GB,
        D,
        ROT,
        CACHE_STRIDE,
    )
    tl.store(Compressed + loc[:, None] * D + cols[None, :], key, mask=live[:, None])


@triton.jit
def _ring_member(KeyState, loc, own, key, cols, D: tl.constexpr):
    # SGLang gathers after its ring store: a member in this row's own slot is this token's key.
    return tl.where(own, key, tl.load(KeyState + loc * D + cols)).to(tl.float32)


@triton.jit(
    do_not_specialize=["position_stride", "stat_words"],
    repr=make_kernel_repr(
        "_indexer_decode_prep", ["H", "D", "ROT", "COMPRESS", "STAT_SLOTS"]
    ),
)
def _indexer_decode_prep(
    QK,
    Q,
    QWeight,
    KWeight,
    KeyState,
    RopeState,
    Compressed,
    Slots,
    GroupLocs,
    WriteLocs,
    Positions,
    position_stride,
    Cache,
    Axis,
    Stats,
    stat_words,
    n_cols,
    q_eps,
    k_eps,
    H: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    RATIO: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    COMPRESS: tl.constexpr,
    STAT_SLOTS: tl.constexpr,
):
    # One CUDA-graph decode row: q norm/RoPE, pending-ring store, then SGLang's fixed-shape
    # compression of the group ending at this token (oldest member first, written to WriteLocs).
    row = tl.program_id(0)
    # Long rows' logits (min bits, ~max bits) slots start all ones; the logits kernel folds into them.
    words = tl.arange(0, 2 * STAT_SLOTS)
    tl.store(
        Stats + row * stat_words + words,
        tl.full((2 * STAT_SLOTS,), -1, tl.int32),
        mask=words < stat_words,
    )
    cols = tl.arange(0, D)
    other = _partner(cols, ROT)
    heads = tl.arange(0, H)
    r0 = tl.load(Positions + row)
    r1 = tl.load(Positions + position_stride + row)
    r2 = tl.load(Positions + 2 * position_stride + row)
    src = QK + row.to(tl.int64) * (H + 1) * D
    x = tl.load(src + heads[:, None] * D + cols[None, :]).to(tl.float32)
    partner = tl.load(src + heads[:, None] * D + other[None, :]).to(tl.float32)
    lanes = tl.zeros([H], dtype=tl.int64)
    q = _norm_rope(
        x,
        partner,
        n_cols,
        q_eps,
        QWeight,
        Cache,
        Axis,
        r0 + lanes,
        r1 + lanes,
        r2 + lanes,
        H,
        D,
        ROT,
        CACHE_STRIDE,
    )
    tl.store(Q + row.to(tl.int64) * H * D + heads[:, None] * D + cols[None, :], q)
    key = tl.load(src + H * D + cols)
    key_other = tl.load(src + H * D + other)
    slot = tl.load(Slots + row).to(tl.int64)
    tl.store(KeyState + slot * D + cols, key)
    axes = tl.arange(0, 4)
    tl.store(
        RopeState + slot * 3 + axes,
        tl.load(Positions + axes * position_stride + row, mask=axes < 3, other=0),
        mask=axes < 3,
    )
    if COMPRESS:
        first = tl.load(GroupLocs + row * RATIO).to(tl.int64)
        own = first == slot
        acc = _ring_member(KeyState, first, own, key, cols, D)
        acc_other = _ring_member(KeyState, first, own, key_other, other, D)
        for member in tl.static_range(1, RATIO):
            loc = tl.load(GroupLocs + row * RATIO + member).to(tl.int64)
            acc = acc + _ring_member(KeyState, loc, loc == slot, key, cols, D)
            acc_other = acc_other + _ring_member(
                KeyState, loc, loc == slot, key_other, other, D
            )
        mean = (acc * (1.0 / RATIO)).to(tl.bfloat16).to(tl.float32)
        mean_other = (acc_other * (1.0 / RATIO)).to(tl.bfloat16).to(tl.float32)
        one = tl.zeros([1], dtype=tl.int64)
        c0 = tl.where(own, r0, tl.load(RopeState + first * 3)) + one
        c1 = tl.where(own, r1, tl.load(RopeState + first * 3 + 1)) + one
        c2 = tl.where(own, r2, tl.load(RopeState + first * 3 + 2)) + one
        k = _norm_rope(
            mean[None, :],
            mean_other[None, :],
            n_cols,
            k_eps,
            KWeight,
            Cache,
            Axis,
            c0,
            c1,
            c2,
            1,
            D,
            ROT,
            CACHE_STRIDE,
        )
        tl.store(
            Compressed + tl.load(WriteLocs + row).to(tl.int64) * D + cols[None, :], k
        )


@triton.jit(repr=make_kernel_repr("_indexer_ring_compress", ["D", "ROT", "RATIO"]))
def _indexer_ring_compress(
    KeyState,
    RopeState,
    Compressed,
    GroupLocs,
    WriteLocs,
    Cache,
    Axis,
    KWeight,
    n_cols,
    k_eps,
    D: tl.constexpr,
    ROT: tl.constexpr,
    RATIO: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
):
    # The fused decode compression after every row's ring store has landed (all members read back).
    row = tl.program_id(0)
    cols = tl.arange(0, D)
    other = _partner(cols, ROT)
    first = tl.load(GroupLocs + row * RATIO).to(tl.int64)
    acc = tl.load(KeyState + first * D + cols).to(tl.float32)
    acc_other = tl.load(KeyState + first * D + other).to(tl.float32)
    for member in tl.static_range(1, RATIO):
        loc = tl.load(GroupLocs + row * RATIO + member).to(tl.int64)
        acc = acc + tl.load(KeyState + loc * D + cols).to(tl.float32)
        acc_other = acc_other + tl.load(KeyState + loc * D + other).to(tl.float32)
    mean = (acc * (1.0 / RATIO)).to(tl.bfloat16).to(tl.float32)
    mean_other = (acc_other * (1.0 / RATIO)).to(tl.bfloat16).to(tl.float32)
    one = tl.zeros([1], dtype=tl.int64)
    c0 = tl.load(RopeState + first * 3) + one
    c1 = tl.load(RopeState + first * 3 + 1) + one
    c2 = tl.load(RopeState + first * 3 + 2) + one
    k = _norm_rope(
        mean[None, :],
        mean_other[None, :],
        n_cols,
        k_eps,
        KWeight,
        Cache,
        Axis,
        c0,
        c1,
        c2,
        1,
        D,
        ROT,
        CACHE_STRIDE,
    )
    tl.store(Compressed + tl.load(WriteLocs + row).to(tl.int64) * D + cols[None, :], k)
