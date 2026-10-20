# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _group_norm_load(
    x_ptr, t_ptr, base, n, g, C, HW, CPG, idx, mask, HAS_TEMB: tl.constexpr
):
    """Load elements idx of group g of sample n in fp32, plus temb[n, channel]."""
    x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
    if HAS_TEMB:
        c = g * CPG + idx // HW
        x += tl.load(t_ptr + n * C + c, mask=mask, other=0.0).to(tl.float32)
    return x


@triton.jit
def _group_norm_ref(x_ptr, t_ptr, base, n, g, C, CPG, HAS_TEMB: tl.constexpr):
    """First element of the group. The statistics are of x - ref, so a large
    mean does not cost precision."""
    ref = tl.load(x_ptr + base).to(tl.float32)
    if HAS_TEMB:
        ref += tl.load(t_ptr + n * C + g * CPG).to(tl.float32)
    return ref


@triton.jit
def _group_norm_chunk_stats(
    x_ptr,
    t_ptr,
    base,
    n,
    g,
    C,
    HW,
    CPG,
    start,
    end,
    ref,
    HAS_TEMB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Mean, M2 and count of x - ref over elements [start, end) of one group.

    Each lane keeps a sum and a sum of squares of x - ref, and the lanes are
    reduced once at the end. Because ref is an element of the group, x - ref
    has a mean near the group spread, so the sums do not cancel.
    """
    s1 = tl.zeros((BLOCK,), dtype=tl.float32)
    s2 = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in tl.range(start, end, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < end
        x = _group_norm_load(x_ptr, t_ptr, base, n, g, C, HW, CPG, idx, mask, HAS_TEMB)
        x = tl.where(mask, x - ref, 0.0)
        s1 += x
        s2 += x * x
    count = (end - start).to(tl.float32)
    sum1 = tl.sum(s1, axis=0)
    mean = sum1 / count
    m2 = tl.maximum(tl.sum(s2, axis=0) - sum1 * mean, 0.0)
    return mean, m2, count


@triton.jit
def _welford_merge(mean_a, m2_a, n_a, mean_b, m2_b, n_b):
    n = n_a + n_b
    safe_n = tl.where(n == 0.0, 1.0, n)
    delta = mean_b - mean_a
    mean = mean_a + delta * n_b / safe_n
    m2 = m2_a + m2_b + delta * delta * n_a * n_b / safe_n
    return mean, m2, n


_group_norm_stats_repr = make_kernel_repr(
    "_group_norm_stats_kernel",
    ["G", "HAS_TEMB", "BLOCK"],
)


@triton.jit(repr=_group_norm_stats_repr)
def _group_norm_stats_kernel(
    x_ptr,
    t_ptr,
    part_ptr,
    C,
    HW,
    CPG,
    GS,
    CHUNK,
    G: tl.constexpr,
    HAS_TEMB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Partial statistics of chunk s of group (n, g): part[(n * G + g) * S + s]."""
    pid = tl.program_id(0)
    s = tl.program_id(1)
    n_split = tl.num_programs(1)
    n = pid // G
    g = pid % G
    base = n.to(tl.int64) * C * HW + g * CPG * HW
    start = s * CHUNK
    end = tl.minimum(start + CHUNK, GS)
    ref = _group_norm_ref(x_ptr, t_ptr, base, n, g, C, CPG, HAS_TEMB)
    mean, m2, count = _group_norm_chunk_stats(
        x_ptr, t_ptr, base, n, g, C, HW, CPG, start, end, ref, HAS_TEMB, BLOCK
    )
    out = part_ptr + (pid * n_split + s) * 3
    tl.store(out, mean)
    tl.store(out + 1, m2)
    tl.store(out + 2, count)


_group_norm_apply_repr = make_kernel_repr(
    "_group_norm_apply_kernel",
    [
        "G",
        "SPLIT_P2",
        "INLINE_STATS",
        "HAS_TEMB",
        "HAS_WEIGHT",
        "HAS_BIAS",
        "activation",
        "use_activation",
        "BLOCK",
    ],
)


@triton.jit(repr=_group_norm_apply_repr)
def _group_norm_apply_kernel(
    x_ptr,
    t_ptr,
    w_ptr,
    b_ptr,
    part_ptr,
    y_ptr,
    C,
    HW,
    CPG,
    GS,
    CHUNK,
    eps,
    G: tl.constexpr,
    SPLIT_P2: tl.constexpr,
    INLINE_STATS: tl.constexpr,
    HAS_TEMB: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    activation: tl.constexpr,
    use_activation: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """y = act(group_norm(x + temb) * weight + bias) for chunk s of group (n, g).

    x is (N, C, *) and contiguous. With INLINE_STATS, the program owns the full
    group and computes its statistics first. Otherwise it merges the partial
    statistics that _group_norm_stats_kernel wrote for the group.
    """
    pid = tl.program_id(0)
    s = tl.program_id(1)
    n_split = tl.num_programs(1)
    n = pid // G
    g = pid % G
    base = n.to(tl.int64) * C * HW + g * CPG * HW

    ref = _group_norm_ref(x_ptr, t_ptr, base, n, g, C, CPG, HAS_TEMB)
    if INLINE_STATS:
        mean, m2, count = _group_norm_chunk_stats(
            x_ptr, t_ptr, base, n, g, C, HW, CPG, 0, GS, ref, HAS_TEMB, BLOCK
        )
    else:
        offs = tl.arange(0, SPLIT_P2)
        pmask = offs < n_split
        part = part_ptr + (pid * n_split + offs) * 3
        means = tl.load(part, mask=pmask, other=0.0)
        m2s = tl.load(part + 1, mask=pmask, other=0.0)
        counts = tl.load(part + 2, mask=pmask, other=0.0)
        mean, m2, count = tl.reduce((means, m2s, counts), 0, _welford_merge)
    rstd = tl.rsqrt(m2 / count + eps)

    start = s * CHUNK
    end = tl.minimum(start + CHUNK, GS)
    for off in tl.range(start, end, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < end
        x = _group_norm_load(x_ptr, t_ptr, base, n, g, C, HW, CPG, idx, mask, HAS_TEMB)
        y = ((x - ref) - mean) * rstd
        if HAS_WEIGHT or HAS_BIAS:
            c = g * CPG + idx // HW
        if HAS_WEIGHT:
            y *= tl.load(w_ptr + c, mask=mask, other=0.0).to(tl.float32)
        if HAS_BIAS:
            y += tl.load(b_ptr + c, mask=mask, other=0.0).to(tl.float32)
        if use_activation:
            y = activation(y)
        tl.store(y_ptr + base + idx, y.to(y_ptr.dtype.element_ty), mask=mask)
