# SPDX-License-Identifier: MIT

import triton
import triton.language as tl
from triton.language.extra import libdevice

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit(
    repr=make_kernel_repr(
        "_fused_topk_dispatch_kernel",
        [
            "BS",
            "Q",
            "DRAFT",
            "WRITE_REQUESTS",
            "BLOCK_BS",
            "NO_PADDING",
            "HAS_VALID_ROWS",
        ],
    )
)
def _fused_topk_dispatch_kernel(
    logits,
    weights,
    ids,
    cu,
    requests,
    pad,
    recv,
    valid_rows,
    STRIDE: tl.constexpr,
    BS: tl.constexpr,
    Q: tl.constexpr,
    DRAFT: tl.constexpr,
    WRITE_REQUESTS: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    NO_PADDING: tl.constexpr = False,
    HAS_VALID_ROWS: tl.constexpr = False,
):
    row = tl.program_id(0)
    if row == 0:
        tl.store(recv, 0)
        if WRITE_REQUESTS:
            b = tl.arange(0, BLOCK_BS)
            start = tl.load(cu + b, b < BS, 0)
            end = tl.load(cu + b + 1, b < BS, 0)
            tl.store(requests, tl.sum((end > start).to(tl.int64), 0))
    if HAS_VALID_ROWS:
        limit = tl.load(valid_rows)
    elif NO_PADDING:
        limit = 0
    elif DRAFT:
        limit = tl.load(requests) * Q
    else:
        limit = tl.load(cu + BS)
    invalid = tl.full((), False, tl.int1) if NO_PADDING else row >= limit
    tl.store(pad + row, invalid)
    e = tl.arange(0, 256)
    values = tl.load(logits + row * STRIDE + e).to(tl.float32)
    maximum = tl.max(values, 0)
    slots = tl.arange(0, 8)
    selected_ids = tl.full((8,), 0, tl.int32)
    selected_weights = tl.full((8,), 0, tl.float32)
    # Match AITER: OCML expf, sequential top-k sum, then reciprocal multiply.
    renorm = tl.full((), 0, tl.float32)
    for i in tl.static_range(8):
        v = tl.max(values, 0)
        expert = tl.min(tl.where(values == v, e, 2147483647), 0)
        # AITER returns expert 0 when an all-NaN dummy row has no winner.
        # Never send the argmin sentinel as an expert ID to dispatch.
        expert = tl.where(expert == 2147483647, 0, expert)
        selected_ids = tl.where(slots == i, expert, selected_ids)
        numer = libdevice.exp(v - maximum)
        renorm = renorm + numer
        selected_weights = tl.where(slots == i, numer, selected_weights)
        values = tl.where(e == expert, -float("inf"), values)
    selected_weights *= tl.div_rn(1.0, renorm)
    tl.store(ids + row * 8 + slots, tl.where(invalid, -1, selected_ids))
    tl.store(weights + row * 8 + slots, selected_weights)
