# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Generic FlyDSL paged decode correctness test and perf sweep.

Run with: python3 op_tests/test_flydsl_pa_decode.py [-k fp8 bf16 mxfp4]
CI also runs it as `python3 -m pytest -q`, which collects the test_* functions at the end.
"""

import argparse
import importlib
import importlib.util
import re
import sys
from itertools import product
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch

import aiter
from aiter import per_tensor_quant, pertoken_quant
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.quant import per_1x32_f4_quant
from aiter.test_common import assertAllclose, benchmark, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")

DEVICE = torch.device("cuda")
SUPPORTED_GFX = ["gfx942", "gfx950", "gfx1250"]
KV_DTYPES = ["fp8", "bf16", "mxfp4"]


def _reference(query, key, value, table, lengths, query_length, window, sinks):
    """FP32 attention over logical KV, independent of kernel Q/P rounding."""
    batch = lengths.numel()
    heads, dim = query.shape[1:]
    kv_heads, page_size = key.shape[1:3]
    group = heads // kv_heads
    queries = query.float().reshape(batch, query_length, kv_heads, group, dim)
    output = torch.zeros_like(queries)
    positions = torch.arange(query_length, device=query.device)

    for seq, length in enumerate(lengths.cpu().tolist()):
        if length == 0:
            continue
        tokens = torch.arange(length, device=query.device)
        pages = table[seq, tokens // page_size].long()
        offsets = tokens % page_size
        keys, values = key[pages, :, offsets], value[pages, :, offsets]
        scores = torch.einsum("qhgd,khd->qhgk", queries[seq], keys) * dim**-0.5
        visible = length - query_length + 1 + positions
        masked = tokens[None, :] >= visible[:, None]
        if window > 0:
            masked |= tokens[None, :] < (visible - window)[:, None]
        scores.masked_fill_(masked[:, None, None, :], float("-inf"))
        denominator = torch.logsumexp(scores, dim=-1, keepdim=True)
        if sinks is not None:
            denominator = torch.logaddexp(
                denominator, sinks.float().reshape(1, kv_heads, group, 1)
            )
        probs = torch.exp(scores - denominator)
        # Empty query rows have no attention mass, including with -inf sinks.
        probs.masked_fill_(visible[:, None, None, None] <= 0, 0)
        output[seq] = torch.einsum("qhgk,khd->qhgd", probs, values)
    return output.reshape_as(query)


def _make_inputs(
    query_length,
    kv_heads,
    group_size,
    dim,
    page_size,
    dtype,
    pattern,
    device,
    lengths,
    table_width=None,
):
    if lengths is None:
        lengths = (0, 1, 255, 256, 257, 1027) if pattern == "random" else (1024,)
    batch = len(lengths)
    pages_per_seq = (max(lengths) + page_size - 1) // page_size
    num_pages = batch * pages_per_seq
    table = torch.randperm(num_pages, device=device, dtype=torch.int32).reshape(
        batch, pages_per_seq
    )
    if table_width is not None:
        # Block-table padding beyond the live pages must not reach the kernel.
        table = torch.nn.functional.pad(table, (0, table_width - pages_per_seq))
    query = torch.empty(
        (batch * query_length, kv_heads * group_size, dim), dtype=dtype, device=device
    ).uniform_(-0.5, 0.5)
    key = torch.empty(
        (num_pages, kv_heads, page_size, dim), dtype=dtype, device=device
    ).uniform_(-0.5, 0.5)
    value = torch.empty_like(key).uniform_(-0.5, 0.5)

    if pattern == "constant":
        query.zero_()
        key.zero_()
        value.fill_(0.5)
    elif pattern == "small_query":
        # Tiny Q values must survive FP8 normalization or native BF16 loads.
        query.fill_(2**-12)
        key.fill_(14)
        key[:, :, 1::2] = -14
        value.fill_(3.5)
        value[:, :, 1::2] = 0
    elif pattern == "probability_tail":
        # Individually tiny probabilities carry substantial mass together.
        # Keep the maximum first and use one partition to retain all 1023 tails.
        # These Q/K values give the same scores in E4M3FN and E4M3FNUZ,
        # with tail probabilities near 2**-12. Scaling P preserves the tails
        # in both formats; converting unscaled P to FP8 rounds them to zero.
        query.zero_()
        query[..., 0] = 13.4375
        key.zero_()
        key[..., 0] = -7
        key[..., 2] = -14
        value.fill_(3.5)
        first_page = table[0, 0].item()
        key[first_page, :, 0] = 0
        value[first_page, :, 0] = 0
        # Give both quantization modes the same V absmax; the other channels
        # still isolate the probability-tail mass.
        value[first_page, :, 0, 0] = 3.5
    elif pattern == "varied":
        # Uniform draws quantize to nearly one per-token scale, so a shifted
        # scale still matches the oracle. Token magnitudes spanning about a
        # decade make a wrong scale index fail it.
        key.mul_(torch.randn_like(key[..., :1]).mul_(0.4).exp_())
        value.mul_(torch.randn_like(value[..., :1]).mul_(0.4).exp_())

    elif pattern == "block_scales":
        # MXFP4 scales are per 32 elements: unequal blocks expose a block/scale mix-up.
        blocks = torch.tensor([1.0, 0.5, 0.25, 0.125], device=device)
        key.mul_(blocks.repeat_interleave(dim // 4))
        value.mul_(blocks.repeat_interleave(dim // 4))

    context = torch.tensor(lengths, dtype=torch.int32, device=device)
    return query, key, value, table, context


def _mxfp4_pages(x):
    """x [pages, kv_heads, 128, 128] -> (page pool, f32 dequantization).

    A pool row is one (page, head): [128 x 64 B e2m1][128 x 4 B e8m0].
    """
    pages, heads, _, dim = x.shape
    data, scale = per_1x32_f4_quant(x.reshape(-1, dim))
    data = data.view(torch.uint8).reshape(-1, dim // 2)
    scale = scale.view(torch.uint8).reshape(-1, dim // 32)
    dequant = fp4_utils.mxfp4_to_f32(data.clone()) * fp4_utils.e8m0_to_f32(
        scale
    ).repeat_interleave(32, dim=-1)
    pool = torch.cat(
        [data.reshape(pages, heads, -1), scale.reshape(pages, heads, -1)], -1
    )
    return pool, dequant.reshape(x.shape)


def _round_query_e4m3(query):
    """A bf16 query as the MXFP4 kernel sees it: each head row rounded to e4m3."""
    scale = (query.float().abs().amax(-1, keepdim=True) / 448).clamp_min(1e-30)
    return (query.float() / scale).to(torch.float8_e4m3fn).float() * scale


def _mxfp4_views(pool):
    """The data and scale views of a pool that pa_decode takes (shared strides)."""
    pages, heads, _ = pool.shape
    page_head = pool.stride()[:2]
    data = pool.as_strided((pages, heads, 128, 64), (*page_head, 64, 1))
    scale = pool.as_strided((pages, heads, 128, 4), (*page_head, 4, 1), 128 * 64)
    return data, scale


_COMMON_CASES = [
    ("decode", {"block_size": 16, "max_partitions": 1}),
    (
        "window",
        {
            "query_length": 4,
            "group_size": 16,
            "block_size": 128,
            "max_partitions": 7,
            "window": 257,
            "sink_dtype": torch.float32,
        },
    ),
    (
        "head1024",
        {
            "group_size": 4,
            "head_dim": 1024,
            "block_size": 128,
            "sink_dtype": torch.bfloat16,
        },
    ),
    *[
        (
            f"{pattern}-page{page}-g{group}",
            {"group_size": group, "block_size": page, "pattern": pattern},
        )
        for page, group in ((16, 8), (128, 16))
        for pattern in ("small_query", "probability_tail")
    ],
]
_FP8_CASES = _COMMON_CASES + [
    (
        "packed-mtp4",
        {
            "query_length": 4,
            "kv_heads": 2,
            "group_size": 8,
            "head_dim": 256,
            "block_size": 128,
            "max_partitions": 7,
            "sink_dtype": torch.float32,
        },
    ),
    (
        "mtp",
        {
            "query_length": 3,
            "kv_heads": 2,
            "group_size": 4,
            "head_dim": 64,
            "max_partitions": 3,
            "sink_dtype": torch.float16,
        },
    ),
    (
        "head256",
        {
            "query_length": 2,
            "kv_heads": 2,
            "head_dim": 256,
            "block_size": 16,
            "max_partitions": 7,
        },
    ),
    (
        "default-cap",
        {"group_size": 16, "block_size": 128, "max_partitions": None},
    ),
]
_BF16_CASES = _COMMON_CASES + [
    (
        "mtp",
        {
            "query_length": 4,
            "kv_heads": 2,
            "group_size": 4,
            "head_dim": 64,
            "max_partitions": 3,
            "sink_dtype": torch.float16,
        },
    ),
    (
        "head256",
        {"kv_heads": 2, "head_dim": 256, "block_size": 16, "max_partitions": 7},
    ),
    ("head192-decode", {"group_size": 16, "head_dim": 192, "max_partitions": 4}),
    (
        "head192-mtp",
        {"query_length": 4, "group_size": 16, "head_dim": 192, "max_partitions": 4},
    ),
    # Large BF16 Q/P operands must split MTP rows to fit the LDS budget.
    *[
        (
            f"head{dim}-mtp-split{splits}",
            {
                "query_length": length,
                "group_size": 16,
                "head_dim": dim,
                "max_partitions": 4,
            },
        )
        for length, dim, splits in ((4, 512, 2), (4, 1024, 4), (3, 1024, 3))
    ],
    ("graph-workspace", {"query_length": 4, "max_partitions": 7, "capture": True}),
    (
        "loose-bound",
        {
            "block_size": 16,
            "lengths": (0, 1, 257, 300, 512),
            "max_partitions": None,
            "table_width": 768,
            "max_context": None,
        },
    ),
    (
        "head192-np80",
        {
            "group_size": 16,
            "head_dim": 192,
            "lengths": (80 * 256,),
            "max_partitions": 80,
        },
    ),
    *[
        (
            f"head{dim}-mixed-np96",
            {
                "query_length": 4,
                "kv_heads": 2,
                "group_size": 16,
                "head_dim": dim,
                "lengths": (97 * 256,) + (0, 1, 257, 1027) * 10,
                "max_partitions": 96,
                "budget": 4096,
            },
        )
        for dim in (256, 128)
    ],
]
# FP8 head 256, page 64, plain V, uniform query length with at most 32 query
# rows: the gfx950 pipe, for per-tensor and per-token scales. A plan-independent
# staging bound selects its specialization;
# other architectures run the same cases on the generic path.
_PIPE_BASE = {
    "kv_dtype": "fp8",
    "scale_mode": "per-tensor",
    "trans_v": False,
    "head_dim": 256,
    "block_size": 64,
    "group_size": 8,
    "pipe": True,
}
_PIPE_CASES = [
    # At GQA 8, query length q gives 8 * q rows: one M-tile at 1 and 2 (1 pads
    # it), two at 3 and 4 (3 pads the second). Lengths put some row's causal
    # edge on a page or tile boundary, and lengths below q leave rows empty.
    # P is FP8, so random rows see 0, 1 or at least 6 tokens; the token
    # one-hot token pattern covers short contexts exactly.
    #
    # A task needing more than 4096 page ids cannot stage the whole list.
    (
        "d256-unstaged-q2",
        {
            **_PIPE_BASE,
            "query_length": 2,
            "lengths": (0, 1, 65, 257, 8192),
            "max_partitions": 4,
            "budget": 1,
            "table_width": 20000,
            "max_context": None,
            "staged": False,
        },
    ),
    # max_context_length below the real context used to index past the staged
    # page ids and fault. The tail must still match a full-context reference.
    (
        "d256-page64-short-bound",
        {
            **_PIPE_BASE,
            "lengths": (128 * 1024,),
            "max_partitions": 1,
            "max_context": 8192,
            "budget": 1,
        },
    ),
    # 81 and 130 partitions leave a short tail, and sequences with few or no
    # tasks must not change a channel.
    *(
        (
            f"d256-page64-q4-np{partitions}",
            {
                **_PIPE_BASE,
                "query_length": 4,
                "lengths": (0, 1, 257, 66532),
                "max_partitions": partitions,
                "budget": partitions,
                "pattern": "one_hot_v",
            },
        )
        for partitions in (81, 130)
    ),
    (
        "d256-page64-per-token",
        {
            **_PIPE_BASE,
            "scale_mode": "per-token",
            "lengths": (1, 65, 257, 1027),
            "max_partitions": 4,
            "pattern": "varied",
        },
    ),
    (
        "d256-page64-q4-per-token",
        {
            **_PIPE_BASE,
            "scale_mode": "per-token",
            "query_length": 4,
            "lengths": (0, 1, 65, 257, 2049),
            "max_partitions": 4,
            "pattern": "varied",
        },
    ),
]
# MXFP4 K/V on gfx950: head_dim 128, page 128, one query token a row, an e4m3 or a
# bf16 query. Plan-free runs one task per row; unequal 32-element blocks make a
# block/scale mix-up visible.
_E4M3 = torch.float8_e4m3fn
_MXFP4_SWEEP = (1, 15, 16, 17, 127, 128, 129, 255, 256, 257, 511, 2047, 2048)
_MXFP4_BASE = {
    "kv_dtype": "mxfp4",
    "block_size": 128,
    "group_size": 16,
    "max_partitions": None,
    "pattern": "block_scales",
}
_MXFP4_CASES = [
    *[
        (
            f"mxfp4-g{group}-{'e4m3' if dtype == _E4M3 else 'bf16'}-q",
            {
                **_MXFP4_BASE,
                "group_size": group,
                "query_dtype": dtype,
                "lengths": _MXFP4_SWEEP,
            },
        )
        for group in (16, 8)
        for dtype in (torch.bfloat16, _E4M3)
    ],
    *[
        (
            f"mxfp4-plan-free-{'e4m3' if dtype == _E4M3 else 'bf16'}-q",
            {
                **_MXFP4_BASE,
                "query_dtype": dtype,
                "plan_free": True,
                "lengths": (1, 16, 255, 256, 257, 1000, 2048, 4097, 5000),
            },
        )
        for dtype in (torch.bfloat16, _E4M3)
    ],
    *[
        (
            f"mxfp4-kv{kv_heads}-g{group}",
            {
                **_MXFP4_BASE,
                "query_dtype": _E4M3,
                "kv_heads": kv_heads,
                "group_size": group,
                "lengths": (1, 17, 300, 1025),
            },
        )
        for kv_heads, group in ((2, 16), (4, 4), (1, 1), (1, 3))
    ],
    *[
        (
            f"mxfp4-np{partitions}",
            {
                **_MXFP4_BASE,
                "query_dtype": _E4M3,
                "max_partitions": partitions,
                "lengths": (16, 255, 1000, 4097, 5000),
            },
        )
        for partitions in (1, 2, 3, 8)
    ],
    (
        "mxfp4-one-task-per-row",
        {
            **_MXFP4_BASE,
            "query_dtype": _E4M3,
            "max_partitions": 1,
            "lengths": (2048, 1, 129, 777, 1000, 17, 2048, 300) * 8,
        },
    ),
    *[
        (
            f"mxfp4-{'plan-free-' if free else ''}graph",
            {
                **_MXFP4_BASE,
                "query_dtype": torch.bfloat16,
                "max_partitions": 7,
                "plan_free": free,
                "capture": True,
                "lengths": (2048, 700, 2048, 33),
            },
        )
        for free in (False, True)
    ],
]
_INVALID_CASES = [
    ("fp16-query", NotImplementedError, "BF16 KV requires bfloat16 queries"),
    ("plain-v", ValueError, "BF16 KV requires the vectorized 5D value_cache layout"),
    (
        "scaled-kv",
        ValueError,
        "BF16 KV is unscaled; key_scale and value_scale must be None",
    ),
    ("vector16-k", ValueError, "key_cache shape must be"),
    ("vector16-v", ValueError, "transposed value_cache shape must be"),
]


def _check_output(output, reference):
    """Finite and equal to the FP32 reference; any element off fails (ratio 0)."""
    assert torch.isfinite(output).all()
    assert output.shape == reference.shape
    return assertAllclose(
        output.float(), reference, rtol=5e-3, atol=5e-3, tol_err_ratio=0
    )


def _expect_error(error, match, call):
    try:
        call()
    except error as exc:
        if not re.search(match, str(exc)):
            raise AssertionError(
                f"expected {error.__name__} matching {match!r}, got {exc!r}"
            ) from exc
        return
    raise AssertionError(f"expected {error.__name__} matching {match!r}; none raised")


def _roofline(us, query, context, query_length, window, caches, kv_heads, page):
    """Nominal TFLOPS and TB/s: QK and PV per visible token, one K+V read of it."""
    tokens = sum(
        min(n, window + query_length - 1) if window else n for n in context.tolist()
    )
    caches = [c for c in caches if isinstance(c, torch.Tensor)]
    per_token = sum(c.nbytes for c in caches) / (caches[0].shape[0] * kv_heads * page)
    heads, dim = query.shape[1:]
    flops = 4 * query_length * heads * dim * tokens
    return {
        "TFLOPS": flops / us / 1e6,
        "TB/s": tokens * kv_heads * per_token / us / 1e6,
    }


def _one_hot_kv(key, value, table, context, compute_type, tokens_per_channel=256):
    """Zero K and one-hot V in channel (token // tokens_per_channel) % dim.

    With 256 tokens per channel each tile owns one output channel, so a task
    that drops or repeats a tile changes that channel's mean. With one, a row's
    output marks exactly the tokens it sees, and zero K keeps FP8 P exact.
    Scales are explicit because a zero K must not reach the quantizer.
    """
    dim, page = value.shape[-1], value.shape[2]
    dense = torch.zeros(value.shape, dtype=torch.float32, device=value.device)
    for seq, length in enumerate(context.tolist()):
        tokens = torch.arange(length, device=value.device)
        pages = table[seq, tokens // page].long()
        dense[pages, :, tokens % page, (tokens // tokens_per_channel) % dim] = 224
    scale = torch.ones(1, dtype=torch.float32, device=value.device)
    return (
        torch.zeros_like(key, dtype=compute_type),
        scale,
        dense.to(compute_type),
        scale * 0.25,
    )


def _check_pipe_policy(pa, case, arch, compile_kwargs):
    """Pin the selected specialization so a case cannot drift off its path."""
    from aiter.ops.flydsl.kernels.pa_decode.traits import PaDecodeSchedule

    assert compile_kwargs[-1]["wide_kv_addressing"] == case["wide"]
    if case["pipe"]:
        # The cache key ends with (D256_PIPE, staged page ids).
        schedule_kwargs = dict(compile_kwargs[-1])
        schedule_kwargs.pop("architecture", None)
        schedule_kwargs.pop("direct_output", None)
        pipe, staged = PaDecodeSchedule.select(**schedule_kwargs).cache_key[-2:]
        assert pipe == (arch == "gfx950")
        if pipe:
            assert (staged > 0) == (case["staged"] is not False)


def _poison_unowned(cache, scale, owned, fill):
    """Overwrite K/V rows no sequence owns (and their per-token scales)."""
    rows = ~owned[:, None, :].expand(-1, cache.shape[1], -1)
    if cache.dtype == torch.bfloat16:
        bits = {"zero": 0, "nan": 0x7FC0, "inf": 0x7F80}[fill]
        cache.view(torch.int16)[rows] = bits
    else:
        # E4M3 has no infinity; its largest finite magnitude stands in.
        nan = 0x80 if cache.dtype == torch.float8_e4m3fnuz else 0x7F
        bits = {"zero": 0, "nan": nan, "inf": 0x7E}[fill]
        cache.view(torch.uint8)[rows] = bits
    if scale is not None and scale.dtype == torch.uint8:
        # e8m0: 255 is NaN, 254 is 2**127.
        scale[rows] = {"zero": 0, "nan": 255}.get(fill, 254)
    elif scale is not None and scale.numel() > 1:
        scale.view(scale.shape[:3])[rows] = {"zero": 0.0, "nan": float("nan")}.get(
            fill, float("inf")
        )


def _mxfp4_inputs(device, lengths=(300, 1, 2048, 5)):
    """e4m3 query, MXFP4 K/V views and the dequantized reference, one query token."""
    torch.manual_seed(0)
    query, key, value, table, context = _make_inputs(
        1, 1, 16, 128, 128, torch.bfloat16, "block_scales", device, lengths
    )
    query = query.to(torch.float8_e4m3fn)
    (key_pool, logical_key), (value_pool, logical_value) = (
        _mxfp4_pages(key),
        _mxfp4_pages(value),
    )
    (kd, ks), (vd, vs) = _mxfp4_views(key_pool), _mxfp4_views(value_pool)
    return query, kd, ks, vd, vs, table, context, (logical_key, logical_value)


_CASE_DEFAULTS = {
    "query_length": 1,
    "kv_heads": 1,
    "group_size": 8,
    "head_dim": 128,
    "block_size": 64,
    "max_partitions": 1,
    "window": 0,
    "sink_dtype": None,
    "pattern": "random",
    "kv_dtype": "bf16",
    "query_dtype": torch.bfloat16,
    "scale_mode": "none",
    "trans_v": True,
    "capture": False,
    "lengths": None,
    "budget": None,
    "table_width": None,
    "max_context": "lengths",
    "wide": False,
    "pipe": False,
    "staged": None,
    "plan_free": False,
}
_MXFP4_INVALID_CASES = [
    ("fp16-output", TypeError, "bf16 output"),
    ("fp16-query", TypeError, "e4m3 or bf16 query"),
    ("page-64", ValueError, "uint8 .* view"),
    ("17-heads", NotImplementedError, "1..16 query heads"),
    ("sinks", NotImplementedError, "no sinks"),
    ("sliding-window", NotImplementedError, "no sliding window"),
    ("query-length-2", NotImplementedError, "query_length 1"),
    ("plan-free-strides", ValueError, "output needs the query's strides"),
    ("oversized-task", ValueError, "PAGE_ID_STAGE_LIMIT"),
    ("max-partitions-2", ValueError, "max_partitions may only be 1"),
    # Operands the kernel would read with the wrong layout or plan: each used to
    # produce wrong output (or an AttributeError) instead of an error.
    ("non-contiguous-table", ValueError, "block_tables must be contiguous"),
    ("context-on-cpu", ValueError, "context_lengths must be contiguous on the query"),
    ("context-length-mismatch", ValueError, r"query.shape\[0\] .* must match"),
    ("table-rows-mismatch", ValueError, r"query.shape\[0\] .* must match"),
    ("query-stride-2", ValueError, "contiguous head_dim axis"),
    ("output-stride-2", ValueError, "contiguous head_dim axis"),
    ("output-on-cpu", ValueError, "output must be on the query device"),
    ("scratch-on-cpu", ValueError, "exp_sums must be contiguous .* on cuda"),
    ("key-scale-none", TypeError, "key_scale must be a uint8 torch.Tensor"),
    # A windowed plan is refused with one partition (served without it) and more.
    ("windowed-plan-1", ValueError, "sliding_window must match work_plan"),
    ("windowed-plan-2", ValueError, "sliding_window must match work_plan"),
]
_UNOWNED_GEOMETRIES = [(1, 1, 8, 128), (4, 2, 8, 256), (4, 1, 16, 128)]
_UNOWNED_KV = [
    "fp8-per-tensor",
    "fp8-per-tensor-plain-v",
    "fp8-per-token",
    "bf16",
    "mxfp4",
]
_PLAN_CASES = [
    # (batch, max_parts, budget, window, query_length)
    (1, 1, 1, 0, 1),
    (6, 7, 160, 257, 4),
    (63, 7, 160, 0, 1),
    (65, 7, 512, 1, 4),
    (257, None, 4096, 8193, 4),
    (4096, None, 8192, 0, 1),
]


def _prepare(case):
    """Inputs, cache views, options and plan of one decode case (no launch)."""
    case = {**_CASE_DEFAULTS, **case}
    arch = get_gfx_runtime()
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    query_length, kv_heads, group_size, dim, page, window = (
        case[name]
        for name in (
            "query_length",
            "kv_heads",
            "group_size",
            "head_dim",
            "block_size",
            "window",
        )
    )
    torch.manual_seed(0)
    # An e4m3 query is the bf16 one rounded.
    query, key, value, table, context = _make_inputs(
        query_length,
        kv_heads,
        group_size,
        dim,
        page,
        (
            torch.bfloat16
            if case["query_dtype"] == torch.float8_e4m3fn
            else case["query_dtype"]
        ),
        case["pattern"],
        DEVICE,
        case["lengths"],
        case["table_width"],
    )
    query = query.to(case["query_dtype"])
    compute_type = torch.bfloat16
    key_scale = value_scale = None
    vector = 8
    if case["kv_dtype"] == "fp8":
        compute_type = (
            torch.float8_e4m3fn
            if arch in ("gfx950", "gfx1250")
            else torch.float8_e4m3fnuz
        )
        vector = 16
        mode = case["scale_mode"]
        if case["pattern"] in ("one_hot_v", "one_hot_token"):
            tokens_per_channel = 1 if case["pattern"] == "one_hot_token" else 256
            key, key_scale, value, value_scale = _one_hot_kv(
                key, value, table, context, compute_type, tokens_per_channel
            )
        elif mode in ("per-tensor", "per-token"):
            quantize = pertoken_quant if mode == "per-token" else per_tensor_quant
            key, key_scale = quantize(key, quant_dtype=compute_type)
            value, value_scale = quantize(value, quant_dtype=compute_type)
        else:
            key, value = key.to(compute_type), value.to(compute_type)
            if mode == "python":
                key_scale, value_scale = 0.25, 2.0
    if case["kv_dtype"] == "mxfp4":
        compute_type = torch.uint8
        (key_pool, logical_key), (value_pool, logical_value) = (
            _mxfp4_pages(key),
            _mxfp4_pages(value),
        )
        (key_cache, key_scale), (value_cache, value_scale) = (
            _mxfp4_views(key_pool),
            _mxfp4_views(value_pool),
        )
    else:
        logical_key = key.float() * (1.0 if key_scale is None else key_scale)
        logical_value = value.float() * (1.0 if value_scale is None else value_scale)
        pages = key.shape[0]
        key_cache = (
            key.reshape(pages, kv_heads, page, dim // vector, vector)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        value_cache = (
            value.reshape(pages, kv_heads, page // vector, vector, dim)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
            if case["trans_v"]
            else value.permute(0, 1, 3, 2).contiguous()
        )
    sinks = None
    if case["sink_dtype"] is not None:
        sinks = torch.linspace(
            -2, 2, query.shape[1], dtype=case["sink_dtype"], device=DEVICE
        )
        sinks[0], sinks[-1] = float("-inf"), float("inf")
    options = {
        "compute_type": compute_type,
        "key_scale": key_scale,
        "value_scale": value_scale,
        "sinks": sinks,
        "sliding_window": window,
        "max_context_length": (
            None
            if case["max_context"] is None
            else (
                context.max().item()
                if case["max_context"] == "lengths"
                else case["max_context"]
            )
        ),
    }
    plan = pa.plan_pa_decode(
        context,
        kv_heads,
        max_partitions=case["max_partitions"],
        workgroup_budget=case["budget"],
        query_length=query_length,
        sliding_window=window,
    )
    return SimpleNamespace(
        case=case,
        arch=arch,
        pa=pa,
        query=query,
        context=context,
        table=table,
        sinks=sinks,
        options=options,
        plan=plan,
        logical_key=logical_key,
        logical_value=logical_value,
        caches=(key_cache, value_cache, key_scale, value_scale),
        arguments=(
            query,
            key_cache,
            value_cache,
            context,
            table,
            dim**-0.5,
            query_length,
        ),
    )


def _check_unsupported_input(s, invalid, error, match):
    pa, options = s.pa, dict(s.options)
    query, key, value, *rest = s.arguments
    if invalid == "fp16-query":
        query = query.to(torch.float16)
    elif invalid == "plain-v":
        value = torch.empty((1, 1, 128, 64), dtype=query.dtype, device=query.device)
    elif invalid == "scaled-kv":
        options.update(
            key_scale=torch.ones(1, dtype=torch.float32, device=query.device),
            value_scale=torch.ones(1, dtype=torch.float32, device=query.device),
        )
    elif invalid == "vector16-k":
        key = torch.empty((1, 1, 8, 64, 16), dtype=query.dtype, device=query.device)
    elif invalid == "vector16-v":
        value = torch.empty((1, 1, 4, 128, 16), dtype=query.dtype, device=query.device)

    def unexpected_compile(**_):
        raise AssertionError("invalid BF16 inputs reached kernel compilation")

    with mock.patch.object(pa, "compile_pa_decode_tile", unexpected_compile):
        _expect_error(
            error,
            match,
            lambda: pa.pa_decode(
                torch.empty_like(query),
                query,
                key,
                value,
                *rest,
                work_plan=s.plan,
                **options,
            ),
        )


@benchmark()
def run_pa_decode_plan(batch, max_parts, budget, window, query_length):
    """Packed tasks cover every visible tile and remain valid after graph refresh.

    The planner is a host-side graph-safe kernel with no roofline, so it is
    checked but not timed.
    """
    from aiter.ops.flydsl.kernels.pa_decode_plan import plan_pa_decode

    def check(plan, lengths):
        info = plan.reduce_info.cpu().tolist()
        work = plan.work_info.cpu().tolist()
        cursor = 0
        for seq, (ctx, (start, count)) in enumerate(zip(lengths, info)):
            assert start == cursor
            first = (
                max(max(ctx, 0) - query_length + 1 - window, 0) // 256 if window else 0
            )
            last = (max(ctx, 0) + 255) // 256
            assert 0 <= count <= min(last - first, plan.max_partitions)
            assert (count > 0) == (last > first)
            tile = first
            for record in work[start : start + count]:
                row, begin, end, original_length = record
                assert row == seq and original_length == ctx
                assert begin == tile and begin < end <= last
                tile = end
            assert tile == last
            cursor += count
        assert cursor <= plan.capacity
        assert all(record == [0, 0, 0, 0] for record in work[cursor:])

    pattern = (-1, 0, 1, 255, 256, 257, 65537, 2**31 - 1)
    lengths = [pattern[i % len(pattern)] for i in range(batch)]
    context = torch.tensor(lengths, dtype=torch.int32, device=DEVICE)
    plan = plan_pa_decode(
        context,
        2,
        max_partitions=max_parts,
        workgroup_budget=budget,
        sliding_window=window,
        query_length=query_length,
    )
    check(plan, lengths)
    pointers = (plan.work_info.data_ptr(), plan.reduce_info.data_ptr())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        refreshed = plan_pa_decode(
            context, 2, sliding_window=window, query_length=query_length, plan=plan
        )
    assert refreshed is plan
    # All-empty replay must erase old active records, then mixed lengths must
    # rebuild them using the same buffers without stale padding or readback.
    for values in ([0] * batch, list(reversed(lengths))):
        context.copy_(torch.tensor(values, dtype=torch.int32, device=DEVICE))
        graph.replay()
        check(plan, values)
        assert (plan.work_info.data_ptr(), plan.reduce_info.data_ptr()) == pointers
    return {"gfx": get_gfx_runtime()}


@benchmark()
def run_pa_decode(name, geometry):
    """FP8/BF16/MXFP4 decode shares input, reference, planning and replay checks."""
    s = _prepare(geometry)
    case, pa, query, window = s.case, s.pa, s.query, s.case["window"]
    query_length, kv_heads, group_size, dim, page = (
        case[n]
        for n in ("query_length", "kv_heads", "group_size", "head_dim", "block_size")
    )
    compile_kwargs = []
    compile_tile = pa.compile_pa_decode_tile

    def spy_tile(**kwargs):
        compile_kwargs.append(kwargs)
        return compile_tile(**kwargs)

    output = torch.full_like(
        query,
        float("nan"),
        dtype=torch.bfloat16 if query.dtype == torch.float8_e4m3fn else query.dtype,
    )
    workspace = {}
    if case["capture"]:
        scalar_shape = (kv_heads, s.plan.capacity, query_length * group_size)
        workspace = {
            "exp_sums": torch.empty(scalar_shape, dtype=torch.float32, device=DEVICE),
            "max_logits": torch.empty(scalar_shape, dtype=torch.float32, device=DEVICE),
            "temporary_output": torch.empty(
                (*scalar_shape, dim), dtype=output.dtype, device=DEVICE
            ),
        }

    def launch():
        pa.pa_decode(
            output,
            *s.arguments,
            work_plan=None if case["plan_free"] else s.plan,
            **s.options,
            **workspace,
        )

    def check():
        reference = _reference(
            _round_query_e4m3(query) if case["kv_dtype"] == "mxfp4" else query,
            s.logical_key,
            s.logical_value,
            s.table,
            s.context,
            query_length,
            window,
            s.sinks,
        )
        return _check_output(output, reference)

    with mock.patch.object(pa, "compile_pa_decode_tile", spy_tile):
        launch()
        err = check()
        if case["kv_dtype"] != "mxfp4":
            _check_pipe_policy(pa, case, s.arch, compile_kwargs)
        _, us = run_perftest(launch)
        if case["capture"]:
            # Compile/allocate before capture; replay must observe changed query data.
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                launch()
            torch.cuda.current_stream().wait_stream(capture_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                launch()
            for _ in range(2):
                query.neg_()
                output.fill_(float("nan"))
                graph.replay()
                check()
    return {
        "gfx": s.arch,
        "kv": case["kv_dtype"],
        "q": str(query.dtype).removeprefix("torch."),
        "scale": case["scale_mode"],
        "q_len": query_length,
        "kv_heads": kv_heads,
        "group": group_size,
        "dim": dim,
        "page": page,
        "batch": s.context.numel(),
        "us": us,
        **_roofline(
            us, query, s.context, query_length, window, s.caches, kv_heads, page
        ),
        "err": err,
    }


@benchmark()
def run_pa_decode_rejects(kv, invalid, error, match):
    """Unsupported inputs raise the documented error before any kernel compiles."""
    if kv == "bf16":
        _check_unsupported_input(_prepare({"lengths": (1,)}), invalid, error, match)
        return {"gfx": get_gfx_runtime(), "raised": error.__name__}
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    entry = importlib.import_module("aiter.ops.flydsl.pa_decode_mxfp4")
    query, kd, ks, vd, vs, table, context, _ = _mxfp4_inputs(DEVICE)
    output = torch.empty(query.shape, dtype=torch.bfloat16, device=DEVICE)
    options = {"key_scale": ks, "value_scale": vs, "work_plan": None}
    query_length = 1
    if invalid == "fp16-output":
        output = output.half()
    elif invalid == "fp16-query":
        query = query.to(torch.float16)
    elif invalid == "page-64":
        kd = kd[:, :, :64]
    elif invalid == "17-heads":
        query = torch.cat([query, query[:, :1]], 1)
        output = torch.empty(query.shape, dtype=torch.bfloat16, device=DEVICE)
    elif invalid == "sinks":
        options["sinks"] = torch.zeros(16, device=DEVICE)
    elif invalid == "sliding-window":
        options["sliding_window"] = 256
    elif invalid == "query-length-2":
        query_length = 2
    elif invalid == "plan-free-strides":
        output = torch.empty(4, 32, 128, dtype=torch.bfloat16, device=DEVICE)[:, :16]
    elif invalid == "oversized-task":
        table = torch.zeros(4, 4097, dtype=torch.int32, device=DEVICE)
    elif invalid == "non-contiguous-table":
        table = torch.cat([table, table], 1)[:, ::2]
    elif invalid == "context-on-cpu":
        context = context.cpu()
    elif invalid == "context-length-mismatch":
        context = context[:3]
    elif invalid == "table-rows-mismatch":
        table = table[:3]
    elif invalid == "query-stride-2":
        strided = torch.empty(4, 16, 256, dtype=query.dtype, device=DEVICE)[..., ::2]
        query = strided.copy_(query)
    elif invalid == "output-stride-2":
        output = torch.empty(4, 16, 256, dtype=torch.bfloat16, device=DEVICE)[..., ::2]
    elif invalid == "output-on-cpu":
        output = output.cpu()
    elif invalid == "scratch-on-cpu":
        # Scratch is only used (and checked) when the plan has more than one partition.
        options["work_plan"] = pa.plan_pa_decode(context, 1, max_partitions=2)
        options["exp_sums"] = torch.empty(
            1, options["work_plan"].capacity, 16, device="cpu"
        )
    elif invalid == "key-scale-none":
        options["key_scale"] = None
    elif invalid.startswith("windowed-plan"):
        options["work_plan"] = pa.plan_pa_decode(
            context, 1, max_partitions=int(invalid[-1]), sliding_window=256
        )
    if invalid == "max-partitions-2":
        call = lambda: entry.pa_decode_mxfp4(
            output, query, kd, ks, vd, vs, context, table, 1.0, max_partitions=2
        )
    else:
        call = lambda: pa.pa_decode(
            output, query, kd, vd, context, table, 1.0, query_length, **options
        )
    _expect_error(error, match, call)
    return {"gfx": get_gfx_runtime(), "raised": error.__name__}


@benchmark()
def run_pa_decode_unowned_kv_rows(
    kv, block_size, query_length, kv_heads, group_size, head_dim, fill
):
    """Rows no sequence owns contribute exactly zero, even when NaN or Inf.

    Padded block-table lanes read block 0 and the last page's tail past the
    context is read as is. Both may hold NaN (a recycled page, a checkpoint
    image); P is zero there but 0 * NaN is NaN in PV.
    ``mask_unowned_kv=False`` skips the masking and matches it on clean input.
    """
    arch = get_gfx_runtime()
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    torch.manual_seed(0)
    lengths = tuple(max(n, query_length) for n in (1, 17, 255, 257, 1000))
    query, key, value, table, context = _make_inputs(
        query_length,
        kv_heads,
        group_size,
        head_dim,
        block_size,
        torch.bfloat16,
        "random",
        DEVICE,
        lengths,
    )
    # Page 0 is owned by no sequence; padded block-table lanes point at it.
    spare = torch.zeros_like(key[:1])
    key, value = torch.cat([spare, key]), torch.cat([spare, value])
    table = torch.nn.functional.pad(table + 1, (0, 2))
    owned = torch.zeros(key.shape[0], block_size, dtype=torch.bool, device=DEVICE)
    for seq, length in enumerate(lengths):
        tokens = torch.arange(length, device=DEVICE)
        owned[table[seq, tokens // block_size].long(), tokens % block_size] = True

    key_scale = value_scale = None
    vector = 8
    if kv.startswith("fp8"):
        compute_type = (
            torch.float8_e4m3fn if arch == "gfx950" else torch.float8_e4m3fnuz
        )
        vector = 16
        quantize = pertoken_quant if kv == "fp8-per-token" else per_tensor_quant
        key, key_scale = quantize(key, quant_dtype=compute_type)
        value, value_scale = quantize(value, quant_dtype=compute_type)
    if kv == "mxfp4":
        (key_pool, logical_key), (value_pool, logical_value) = (
            _mxfp4_pages(key),
            _mxfp4_pages(value),
        )
    else:
        logical_key = key.float() * (1.0 if key_scale is None else key_scale)
        logical_value = value.float() * (1.0 if value_scale is None else value_scale)
    reference = _reference(
        query, logical_key, logical_value, table, context, query_length, 0, None
    )
    plan = pa.plan_pa_decode(context, kv_heads, query_length=query_length)

    def prepare(fill, mask_unowned_kv=True):
        if kv == "mxfp4":
            (k, ks), (v, vs) = (
                _mxfp4_views(key_pool.clone()),
                _mxfp4_views(value_pool.clone()),
            )
        else:
            k, v = key.clone(), value.clone()
            ks = None if key_scale is None else key_scale.clone()
            vs = None if value_scale is None else value_scale.clone()
        _poison_unowned(k, ks, owned, fill)
        _poison_unowned(v, vs, owned, fill)
        key_cache, value_cache = k, v
        if kv != "mxfp4":
            pages = k.shape[0]
            key_cache = (
                k.reshape(pages, kv_heads, block_size, head_dim // vector, vector)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )
            if kv == "fp8-per-tensor-plain-v":
                value_cache = v.permute(0, 1, 3, 2).contiguous()
            else:
                value_cache = (
                    v.reshape(pages, kv_heads, block_size // vector, vector, head_dim)
                    .permute(0, 1, 2, 4, 3)
                    .contiguous()
                )
        output = torch.full_like(query, float("nan"))

        def launch():
            pa.pa_decode(
                output,
                query,
                key_cache,
                value_cache,
                context,
                table,
                head_dim**-0.5,
                query_length,
                compute_type=key_cache.dtype,
                key_scale=ks,
                value_scale=vs,
                work_plan=plan,
                max_context_length=max(lengths),
                mask_unowned_kv=mask_unowned_kv,
            )

        return output, launch, (key_cache, value_cache, ks, vs)

    def run(fill, mask_unowned_kv=True):
        output, launch, _ = prepare(fill, mask_unowned_kv)
        launch()
        return output

    clean = run("zero")
    poisoned, launch, caches = prepare(fill)
    launch()
    assert torch.isfinite(poisoned).all()
    # Short FP8 contexts round P coarsely; exactness is the bitwise check.
    tol = 5e-3 if kv == "bf16" else 2.5e-2
    err = assertAllclose(
        poisoned.float(), reference, rtol=tol, atol=tol, tol_err_ratio=0
    )
    assert torch.equal(poisoned, clean)
    if fill == "nan":
        assert torch.equal(run("zero", mask_unowned_kv=False), clean)
    _, us = run_perftest(launch)
    return {
        "gfx": arch,
        "us": us,
        **_roofline(us, query, context, query_length, 0, caches, kv_heads, block_size),
        "err": err,
    }


@benchmark()
def run_pa_decode_mxfp4_one_task_per_row():
    """Plan-free, one-partition plan and the pa_decode_mxfp4 entry run the same task.

    A strided output cannot be written directly and takes the reducer; a context-0
    row is zero-filled on both paths.
    """
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    entry = importlib.import_module("aiter.ops.flydsl.pa_decode_mxfp4")
    query, kd, ks, vd, vs, table, context, (logical_key, logical_value) = _mxfp4_inputs(
        DEVICE
    )
    context = torch.tensor([300, 0, 2048, 5], dtype=torch.int32, device=DEVICE)
    plan = pa.plan_pa_decode(context, 1, max_partitions=1)
    reference = _reference(
        query, logical_key, logical_value, table, context, 1, 0, None
    )
    scale = 128**-0.5

    def run(output, **options):
        pa.pa_decode(
            output,
            query,
            kd,
            vd,
            context,
            table,
            scale,
            1,
            key_scale=ks,
            value_scale=vs,
            **options,
        )
        return output

    free = run(
        torch.full(query.shape, 7.0, dtype=torch.bfloat16, device=DEVICE),
        work_plan=None,
    )
    planned = run(torch.full_like(free, 7.0), work_plan=plan)
    aliased = torch.full_like(free, 7.0)
    entry.pa_decode_mxfp4(
        aliased, query, kd, ks, vd, vs, context, table, scale, max_partitions=1
    )
    assert torch.equal(free, planned) and torch.equal(free, aliased)
    err = _check_output(free, reference)
    padded = torch.full((4, 32, 128), 7.0, dtype=torch.bfloat16, device=DEVICE)
    _check_output(run(padded[:, :16], work_plan=plan), reference)
    assert (padded[:, 16:] == 7.0).all()
    _, us = run_perftest(lambda: run(free, work_plan=None))
    return {
        "gfx": get_gfx_runtime(),
        "us": us,
        **_roofline(us, query, context, 1, 0, (kd, vd, ks, vs), 1, 128),
        "err": err,
    }


def _decode_cases(kv_dtypes):
    """(name, geometry) of every valid decode case, in a stable order."""
    cases = []
    if "fp8" in kv_dtypes:
        cases += [
            (
                f"fp8-{dtype}-{scale_mode}-{'transposed' if trans_v else 'plain'}-v-{name}",
                dict(
                    geometry,
                    kv_dtype="fp8",
                    query_dtype=dtype,
                    scale_mode=scale_mode,
                    trans_v=trans_v,
                ),
            )
            for name, geometry in _FP8_CASES
            for dtype, scale_mode, trans_v in product(
                (torch.bfloat16, torch.float16),
                ("per-tensor", "per-token"),
                (False, True),
            )
        ]
    if "bf16" in kv_dtypes:
        cases += [(f"bf16-{name}", geometry) for name, geometry in _BF16_CASES]
    if "fp8" in kv_dtypes:
        cases += _PIPE_CASES
    if "mxfp4" in kv_dtypes:
        cases += _MXFP4_CASES
    return cases


def _skip_reason(case, arch, num_cu):
    """Why this arch cannot run a decode case, or None."""
    case = {**_CASE_DEFAULTS, **case}
    if case["kv_dtype"] == "mxfp4" and arch != "gfx950":
        return "MXFP4 pa_decode runs on gfx950"
    if case["max_partitions"] is not None and case["max_partitions"] > num_cu:
        return f"plan_pa_decode caps max_partitions at {num_cu} CUs"
    if arch == "gfx1250" and (
        case["kv_dtype"] == "bf16" and case["head_dim"] % 128 and case["head_dim"] != 64
    ):
        return "gfx1250 supports head_dim 64 or multiples of 128"
    return None


def _unowned_skip_reason(kv, block_size, query_length, head_dim, arch):
    if arch == "gfx1250":
        return "gfx1250 pipelines still require finite unowned V rows"
    if kv == "mxfp4" and (
        arch != "gfx950" or (block_size, query_length, head_dim) != (128, 1, 128)
    ):
        return "MXFP4 pa_decode: gfx950, page 128, head_dim 128, one query token"
    return None


def _summarize(title, rows):
    # The geometry dict only drives the case; its summary is the columns beside it.
    df = pd.DataFrame(rows).drop(columns="geometry", errors="ignore")
    aiter.logger.info("%s summary (markdown):\n%s", title, df.to_markdown(index=False))


def _sweep(title, test, cases, failures):
    """Run (tag, args) cases, one failing case must not hide the others."""
    rows = []
    for tag, args in cases:
        try:
            rows.append(test(*args))
        except Exception as exc:  # noqa: BLE001 - record, keep sweeping
            aiter.logger.error(
                "FAILED %s case %s: %s",
                title,
                tag,
                exc,
                exc_info=not isinstance(exc, AssertionError),
            )
            failures.append((title, tag, exc))
            rows.append({"name": tag, "gfx": get_gfx_runtime(), "status": "FAIL"})
    _summarize(title, rows)
    return len(rows)


def _skipping(arch, cases, reason):
    """Drop cases whose reason(case) is set, warning once per distinct reason."""
    kept, skipped = [], {}
    for tag, args in cases:
        why = reason(*args)
        if why:
            skipped.setdefault(why, []).append(tag)
        else:
            kept.append((tag, args))
    for why, tags in skipped.items():
        aiter.logger.warning("%s: skipping %d case(s): %s", arch, len(tags), why)
    return kept


def _unsupported():
    """Why pa_decode cannot run here, or None."""
    arch = get_gfx_runtime()
    if not torch.cuda.is_available() or arch not in SUPPORTED_GFX:
        return f"pa_decode unsupported on {arch}"
    if importlib.util.find_spec("flydsl") is None:
        return "FlyDSL is not installed"
    return None


def _plan_sweep():
    return (
        "pa_decode plan",
        run_pa_decode_plan,
        [(f"plan-{'-'.join(map(str, c))}", c) for c in _PLAN_CASES],
    )


def _sweeps(arch, kv_dtypes):
    """(title, benchmark fn, cases) for the KV cache dtypes, gated on this arch."""
    num_cu = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    decode = _skipping(
        arch,
        [(name, (name, geometry)) for name, geometry in _decode_cases(kv_dtypes)],
        lambda name, geometry: _skip_reason(geometry, arch, num_cu),
    )
    rejects = []
    if "bf16" in kv_dtypes:
        # gfx1250 serves FP16 queries and plain BF16 V caches, so it rejects neither.
        rejects += [
            ("bf16", *c)
            for c in _INVALID_CASES
            if arch != "gfx1250" or c[0] not in ("fp16-query", "plain-v")
        ]
    if "mxfp4" in kv_dtypes and arch == "gfx950":
        rejects += [("mxfp4", *c) for c in _MXFP4_INVALID_CASES]
    elif "mxfp4" in kv_dtypes:
        aiter.logger.warning("%s: skipping MXFP4 cases, which run on gfx950", arch)
    unowned = _skipping(
        arch,
        [
            (
                "unowned-" + "-".join(map(str, (kv, block_size, *geometry, fill))),
                (kv, block_size, *geometry, fill),
            )
            for kv, block_size, geometry, fill in product(
                [kv for kv in _UNOWNED_KV if kv.split("-")[0] in kv_dtypes],
                [16, 64, 128],
                _UNOWNED_GEOMETRIES,
                ["nan", "inf"],
            )
        ],
        lambda kv, block_size, query_length, kv_heads, group_size, head_dim, fill: (
            _unowned_skip_reason(kv, block_size, query_length, head_dim, arch)
        ),
    )
    one_task = []
    if "mxfp4" in kv_dtypes and arch == "gfx950":
        one_task = [("mxfp4-one-task-per-row", ())]
    return [
        ("pa_decode", run_pa_decode, decode),
        (
            "pa_decode rejected inputs",
            run_pa_decode_rejects,
            [
                (f"{kv}-{invalid}", (kv, invalid, error, match))
                for kv, invalid, error, match in rejects
            ],
        ),
        ("pa_decode unowned KV rows", run_pa_decode_unowned_kv_rows, unowned),
        (
            "pa_decode MXFP4 one task per row",
            run_pa_decode_mxfp4_one_task_per_row,
            one_task,
        ),
    ]


def _run(sweeps):
    """Run every non-empty sweep once; returns (case count, failures)."""
    failures, total = [], 0
    # The reference is FP32; a TF32 matmul would loosen it.
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for title, test, cases in sweeps:
            if cases:
                total += _sweep(title, test, cases, failures)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return total, failures


def _failure_report(failures, total):
    lines = "\n".join(f"  [{t}] {tag}\n        {exc}" for t, tag, exc in failures)
    return f"pa_decode: {len(failures)} of {total} cases FAILED\n{lines}"


def _run_as_test(what, sweeps):
    """Shared body of the pytest-collected tests: same sweeps, same checks."""
    import unittest

    why = _unsupported()
    if why:
        raise unittest.SkipTest(why)
    total, failures = _run(sweeps())
    if not total:
        raise unittest.SkipTest(f"no {what} case runs on {get_gfx_runtime()}")
    assert not failures, _failure_report(failures, total)


# `python3 -m pytest op_tests/test_flydsl_pa_decode.py` (what CI runs) collects only
# these zero-argument functions; the argument-taking run_* sweeps are not tests.
def test_pa_decode_plan():
    _run_as_test("plan", lambda: [_plan_sweep()])


def test_pa_decode_fp8():
    _run_as_test("FP8", lambda: _sweeps(get_gfx_runtime(), ["fp8"]))


def test_pa_decode_bf16():
    _run_as_test("BF16", lambda: _sweeps(get_gfx_runtime(), ["bf16"]))


def test_pa_decode_mxfp4():
    _run_as_test("MXFP4", lambda: _sweeps(get_gfx_runtime(), ["mxfp4"]))


def main():
    why = _unsupported()
    if why:
        aiter.logger.warning("%s; skipping", why)
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL paged decode correctness + perf sweep",
    )
    parser.add_argument(
        "-k",
        "--kv",
        type=str,
        nargs="*",
        default=KV_DTYPES,
        choices=KV_DTYPES,
        help="KV cache dtypes to sweep (default: all)",
    )
    args = parser.parse_args()
    total, failures = _run([_plan_sweep(), *_sweeps(get_gfx_runtime(), args.kv)])
    if failures:
        aiter.logger.error(_failure_report(failures, total))
        sys.exit(1)
    aiter.logger.info("pa_decode: all %d cases passed", total)


if __name__ == "__main__":
    main()
