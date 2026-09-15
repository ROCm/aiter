# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance sweep for FlyDSL paged-attention Tile.

For example, BS200/MTP4 with FP8 per-token KV scales and explicit split counts::

    python3 op_tests/test_flydsl_pa_decode.py -d bf16 -b 200 -q 4 \
        -s 16,1,128,200000 --block-size 16 128 --trans-v 0 1 \
        --per-token 1 --num-partitions 3 5

Contexts have equal lengths and include the MTP query tokens. Query position
``p`` attends to ``max(0, context_length - query_length + 1 + p)`` KV tokens.
Timing includes the FlyDSL attention kernel and its native FlyDSL reduction.
"""

import argparse
import importlib
import itertools

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes, per_tensor_quant, pertoken_quant
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest


@pytest.fixture(autouse=True)
def _default_cuda_device():
    # Scoped rather than set at import time: this module now runs in the shared
    # Standard Tests shard, where a global default-device switch would follow
    # every later test in the session.
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    yield
    torch.set_default_device(previous)


SUPPORTED_GFX = ["gfx942", "gfx950"]
KV_COMPUTE_BLOCK = 256
# Fixed-length outputs use the tighter bound, while independently selected
# variable-length rows use the established varlen bound.
FIXED_LENGTH_ACCURACY_TOLERANCE = 5e-3
VARIABLE_LENGTH_ACCURACY_TOLERANCE = 5e-2

# Pairwise coverage of the normal-accuracy axes in FlyDSL's PA regression test:
# batches {3, 81, 128}, Q/KV heads {(4,1), (8,1), (16,1)}, head dims
# {128, 256}, and contexts {1027, 8192}. Keep the original 257-token boundary
# case as well. All supported block sizes are crossed with every case in main().
DEFAULT_BATCH_SIZES = [3, 81, 128]
DEFAULT_SHAPES = [
    (8, 1, 128, 257),
    (4, 1, 128, 1027),
    (8, 1, 128, 1027),
    (8, 1, 256, 1027),
    (16, 1, 128, 8192),
]

try:
    from aiter.ops.flydsl.pa_decode import (
        MAX_CONTEXT_PARTITIONS,
        get_recommended_splits,
        pa_decode,
    )
except (ImportError, AttributeError, RuntimeError, OSError):
    MAX_CONTEXT_PARTITIONS = 256
    get_recommended_splits = None
    pa_decode = None


def _quant_dtype() -> torch.dtype:
    return (
        torch.float8_e4m3fn if get_gfx_runtime() == "gfx950" else torch.float8_e4m3fnuz
    )


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if pa_decode is None:
        pytest.skip("FlyDSL is not available")
    if get_gfx_runtime() not in SUPPORTED_GFX:
        pytest.skip(f"pa_decode is unsupported on {get_gfx_runtime()}")


@pytest.mark.parametrize(
    "batch_size,max_partitions,expected",
    [(1, 8, 8), (1, 256, 256), (16, 256, 32), (64, 256, 8)],
)
def test_recommended_splits_has_configurable_upper_clamp(
    monkeypatch, batch_size, max_partitions, expected
):
    """The default stays at eight while long-context callers may opt into 256."""
    if get_recommended_splits is None:
        pytest.skip("FlyDSL is not available")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_args, **_kwargs: type("Props", (), {"multi_processor_count": 256})(),
    )
    assert (
        get_recommended_splits(
            batch_size,
            num_kv_heads=1,
            split_kv_blocks=2,
            max_partitions=max_partitions,
        )
        == expected
    )


def run_torch(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    query_length: int = 1,
) -> torch.Tensor:
    """Dequantized FP32 reference with sequence-major, causal MTP queries."""
    num_queries, num_query_heads, head_dim = query.shape
    batch_size = context_lengths.numel()
    if query_length < 1 or num_queries != batch_size * query_length:
        raise ValueError("query must have batch_size * query_length rows")
    block_size = key_cache.shape[2]
    num_kv_heads = key_cache.shape[1]
    query_group_size = num_query_heads // num_kv_heads
    softmax_scale = head_dim**-0.5
    output = torch.zeros_like(query)
    queries = query.reshape(
        batch_size, query_length, num_kv_heads, query_group_size, head_dim
    )
    positions = torch.arange(query_length, device=query.device)

    for seq_idx in range(batch_size):
        context_length = int(context_lengths[seq_idx].item())
        if context_length == 0:
            continue
        token_ids = torch.arange(context_length, device=query.device)
        logical_pages = token_ids // block_size
        token_offsets = token_ids % block_size
        physical_pages = block_tables[seq_idx, logical_pages].long()

        keys = key_cache[physical_pages, :, token_offsets, :].float()
        values = value_cache[physical_pages, :, :, token_offsets].float()
        if key_scale.numel() == 1:
            keys = keys * key_scale.float()
            values = values * value_scale.float()
        else:
            token_key_scale = key_scale[physical_pages, :, token_offsets, 0].float()
            token_value_scale = value_scale[physical_pages, :, token_offsets, 0].float()
            keys = keys * token_key_scale.unsqueeze(-1)
            values = values * token_value_scale.unsqueeze(-1)
        # Keep KV heads grouped instead of materializing Hq/Hkv copies of the
        # long-context cache just for the reference (notably BS200/C200k).
        scores = (
            torch.einsum("qhgd,khd->qhgk", queries[seq_idx].float(), keys)
            * softmax_scale
        )
        visible = context_length - query_length + 1 + positions
        masked = token_ids.unsqueeze(0) >= visible.unsqueeze(1)
        scores.masked_fill_(masked[:, None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        # Contexts shorter than QL have leading queries with no visible tokens.
        probs.masked_fill_(visible[:, None, None, None] <= 0, 0)
        first_row = seq_idx * query_length
        output[first_row : first_row + query_length] = torch.einsum(
            "qhgk,khd->qhgd", probs, values
        ).reshape(query_length, num_query_heads, head_dim)
    return output


def _run_flydsl(
    output,
    query,
    key_cache,
    value_cache,
    block_tables,
    context_lengths,
    key_scale,
    value_scale,
    num_partitions,
    softmax_scale,
    pmax,
    psum,
    pout,
):
    torch.ops.aiter.pa_decode_flydsl(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        softmax_scale,
        query.shape[0] // context_lengths.shape[0],
        num_partitions,
        256,
        key_cache.dtype,
        None,
        key_scale,
        value_scale,
        exp_sums=psum,
        max_logits=pmax,
        temporary_output=pout,
        ps=True,
    )
    return output


@benchmark()
def run_pa_decode_tile_case(
    batch_size,
    num_query_heads,
    num_kv_heads,
    head_dim,
    context_length,
    block_size,
    dtype,
    trans_v,
    max_partitions=8,
    per_token=False,
    query_length=1,
    num_partitions=None,
):
    if query_length < 1:
        raise ValueError("query_length must be positive")
    if num_partitions is not None and not 1 <= num_partitions <= MAX_CONTEXT_PARTITIONS:
        raise ValueError(f"num_partitions must be in [1, {MAX_CONTEXT_PARTITIONS}]")
    if batch_size < 1 or context_length < 1:
        raise ValueError("batch_size and context_length must be positive")
    if pa_decode is None or get_recommended_splits is None:
        raise RuntimeError("FlyDSL is not available")
    if dtype not in (dtypes.fp16, dtypes.bf16):
        raise ValueError(f"pa_decode only supports fp16/bf16, got {dtype}")
    if num_query_heads < 1 or num_kv_heads < 1 or num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    # An explicit NP is an exact override, not an upper clamp. In particular,
    # NP=3 and NP=5 must not be rounded by the automatic recommendation.
    if num_partitions is None:
        num_partitions = get_recommended_splits(
            batch_size,
            num_kv_heads,
            split_kv_blocks=KV_COMPUTE_BLOCK // block_size,
            max_partitions=max_partitions,
        )

    torch.manual_seed(0)
    blocks_per_sequence = (context_length + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_sequence

    query = torch.empty(
        batch_size * query_length,
        num_query_heads,
        head_dim,
        dtype=dtype,
    ).uniform_(-0.5, 0.5)
    key = torch.empty(
        num_blocks,
        num_kv_heads,
        block_size,
        head_dim,
        dtype=dtype,
    ).uniform_(-0.5, 0.5)
    value = torch.empty(
        num_blocks,
        num_kv_heads,
        head_dim,
        block_size,
        dtype=dtype,
    ).uniform_(-0.5, 0.5)

    quant_dtype = _quant_dtype()
    if per_token:
        key_quant, key_scale = pertoken_quant(key, quant_dtype=quant_dtype)
        value_token_major = value.permute(0, 1, 3, 2).contiguous()
        value_token_quant, value_scale = pertoken_quant(
            value_token_major, quant_dtype=quant_dtype
        )
        value_quant = value_token_quant.permute(0, 1, 3, 2).contiguous()
        del value_token_major, value_token_quant
    else:
        key_quant, key_scale = per_tensor_quant(key, quant_dtype=quant_dtype)
        value_quant, value_scale = per_tensor_quant(value, quant_dtype=quant_dtype)
    del key, value
    key_cache = (
        key_quant.view(
            num_blocks,
            num_kv_heads,
            block_size,
            head_dim // 16,
            16,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    if trans_v:
        value_cache = (
            value_quant.permute(0, 1, 3, 2)
            .contiguous()
            .view(num_blocks, num_kv_heads, block_size // 16, 16, head_dim)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )
    else:
        value_cache = value_quant.contiguous()
    block_tables = torch.arange(num_blocks, dtype=torch.int32).reshape(
        batch_size, blocks_per_sequence
    )
    context_lengths = torch.full((batch_size,), context_length, dtype=torch.int32)
    output = torch.empty_like(query)

    reference = run_torch(
        query,
        key_quant,
        value_quant,
        block_tables,
        context_lengths,
        key_scale,
        value_scale,
        query_length=query_length,
    )
    # Only the kernel's cache layouts need to survive into allocation rotation.
    del key_quant, value_quant

    query_group_size = num_query_heads // num_kv_heads
    partial_shape = (
        batch_size,
        num_kv_heads,
        num_partitions,
        query_length * query_group_size,
    )
    pmax = torch.empty(partial_shape, dtype=dtypes.fp32)
    psum = torch.empty_like(pmax)
    pout = torch.empty(*partial_shape, head_dim, dtype=dtype)
    softmax_scale = head_dim**-0.5

    candidates = {"flydsl": _run_flydsl}

    # QK and PV each perform one multiply-add per visible query/KV-token pair.
    attended_tokens = sum(
        max(0, context_length - query_length + 1 + position)
        for position in range(query_length)
    )
    flops = 4 * batch_size * num_query_heads * attended_tokens * head_dim
    # Effective bandwidth counts Q + O + referenced K/V tokens and metadata once.
    # It excludes padded tokens, repeated loads, and partition scratch traffic.
    # All MTP positions share the KV cache and its scales: count them once,
    # including only valid tokens rather than a partially padded final page.
    scale_elements = batch_size * num_kv_heads * context_length if per_token else 1
    nbytes = (
        2 * query.numel() * query.element_size()
        + 2
        * batch_size
        * num_kv_heads
        * context_length
        * head_dim
        * key_cache.element_size()
        + block_tables.numel() * block_tables.element_size()
        + context_lengths.numel() * context_lengths.element_size()
        + scale_elements * (key_scale.element_size() + value_scale.element_size())
    )

    ret = {
        "gfx": get_gfx_runtime(),
        "partitions": num_partitions,
        "trans_v": trans_v,
        "per_token": per_token,
    }
    for name, fn in candidates.items():
        # Pass tensors explicitly so perftest can rotate their allocations.
        out, us = run_perftest(
            fn,
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            key_scale,
            value_scale,
            num_partitions,
            softmax_scale,
            pmax,
            psum,
            pout,
        )
        err = checkAllclose(
            reference.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=FIXED_LENGTH_ACCURACY_TOLERANCE,
            atol=FIXED_LENGTH_ACCURACY_TOLERANCE,
            tol_err_ratio=0.0,
            msg=f"{name}: pa_decode",
        )
        if err:
            raise AssertionError(f"{name}: pa_decode mismatch ratio {err}")
        ret[f"{name} us"] = us
        # Amortized time per output token, not the latency of an MTP step.
        ret[f"{name} us/token"] = us / (batch_size * query_length)
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def test_pa_decode_benchmark_reference_causal_mtp():
    """An analytic oracle catches flattened-row, KV-head, and causal mistakes."""
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    dtype = dtypes.bf16
    query_length, query_group_size, head_dim, block_size = 4, 2, 4, 16
    lengths, pages = [4, 2, 0], [1, 0, -1]
    query = torch.zeros(12, 4, head_dim, dtype=dtype)
    key = torch.zeros(2, 2, block_size, head_dim, dtype=dtypes.fp32)
    value = (
        100 * torch.arange(2).reshape(2, 1, 1, 1)
        + 10 * torch.arange(2).reshape(1, 2, 1, 1)
        + torch.arange(head_dim).reshape(1, 1, head_dim, 1)
        + torch.arange(1, block_size + 1).reshape(1, 1, 1, block_size)
    ).float()
    value_scale = (
        torch.arange(1, block_size + 1, dtype=dtypes.fp32)
        .reshape(1, 1, block_size, 1)
        .expand(2, 2, block_size, 1)
        .contiguous()
    )
    key_scale = torch.ones_like(value_scale)
    actual = run_torch(
        query,
        key,
        value,
        torch.tensor(pages, dtype=dtypes.i32).reshape(3, 1),
        torch.tensor(lengths, dtype=dtypes.i32),
        key_scale,
        value_scale,
        query_length=query_length,
    )
    expected = []
    for length, page in zip(lengths, pages):
        for position in range(query_length):
            visible = max(0, length - query_length + 1 + position)
            expected.append(
                [
                    [
                        sum(
                            (100 * page + 10 * (head // query_group_size) + dim + t + 1)
                            * (t + 1)
                            for t in range(visible)
                        )
                        / max(visible, 1)
                        for dim in range(head_dim)
                    ]
                    for head in range(4)
                ]
            )
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=dtype), rtol=0, atol=0
    )


def test_pa_decode_benchmark_cli_defaults_and_bs200():
    defaults = _parse_args([])
    assert defaults.query_length == [1]
    assert defaults.num_partitions == [None]
    assert defaults.max_partitions == 8
    args = _parse_args(
        [
            "-d",
            "bf16",
            "-b",
            "200",
            "-s",
            "16,1,128,200000",
            "-q",
            "4",
            "--block-size",
            "16",
            "128",
            "--trans-v",
            "0",
            "1",
            "--per-token",
            "1",
            "--num-partitions",
            "3",
            "5",
            "8",
        ]
    )
    assert args.dtype == [dtypes.bf16]
    assert args.batch == [200]
    assert args.shapes == [(16, 1, 128, 200000)]
    assert args.query_length == [4]
    assert args.block_size == [16, 128]
    assert args.trans_v == [0, 1]
    assert args.per_token == [1]
    assert args.num_partitions == [3, 5, 8]
    boundaries = _parse_args(
        ["-q", "1", "4", "--num-partitions", "1", "256", "--max-partitions", "256"]
    )
    assert boundaries.query_length == [1, 4]
    assert boundaries.num_partitions == [1, 256]
    assert boundaries.max_partitions == 256


@pytest.mark.parametrize(
    "batch_size,query_length,num_partitions,block_size,trans_v,dtype,per_token",
    [
        pytest.param(
            2, 1, None, 16, False, dtypes.bf16, False, id="legacy-auto-decode"
        ),
        pytest.param(2, 4, 3, 16, True, dtypes.bf16, True, id="mtp4-page16-np3"),
        pytest.param(2, 4, 3, 128, True, dtypes.fp16, False, id="mtp4-fp16-scalar"),
        pytest.param(200, 4, 5, 128, False, dtypes.bf16, True, id="bs200-mtp4-np5"),
    ],
)
def test_pa_decode_benchmark_entry(
    monkeypatch,
    batch_size,
    query_length,
    num_partitions,
    block_size,
    trans_v,
    dtype,
    per_token,
):
    """Exercise the real public kernel once, without a 101-iteration benchmark."""
    _require_gpu()
    context_length, num_query_heads, num_kv_heads, head_dim = 257, 16, 1, 128
    expected_partitions = 4 if num_partitions is None else num_partitions
    recommendations, timed_calls = [], []

    def recommend(*args, **kwargs):
        assert num_partitions is None, "an explicit NP must bypass auto selection"
        recommendations.append((args, kwargs))
        return expected_partitions

    def run_once(fn, *args, **kwargs):
        assert fn is _run_flydsl
        assert not kwargs
        (
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            key_scale,
            value_scale,
            partitions,
            softmax_scale,
            pmax,
            psum,
            pout,
        ) = args
        assert output.shape == query.shape == (batch_size * query_length, 16, 128)
        assert output.dtype == query.dtype == pout.dtype == dtype
        assert key_cache.dtype == value_cache.dtype == _quant_dtype()
        assert value_cache.ndim == (5 if trans_v else 4)
        assert partitions == expected_partitions
        assert softmax_scale == head_dim**-0.5
        partial_shape = (batch_size, 1, partitions, query_length * 16)
        assert pmax.shape == psum.shape == partial_shape
        assert pmax.dtype == psum.dtype == dtypes.fp32
        assert pout.shape == (*partial_shape, 128)
        blocks_per_sequence = (context_length + block_size - 1) // block_size
        assert block_tables.shape == (batch_size, blocks_per_sequence)
        assert context_lengths.shape == (batch_size,)
        assert bool((context_lengths == context_length).all())
        scale_elements = (
            batch_size * blocks_per_sequence * block_size if per_token else 1
        )
        assert key_scale.numel() == value_scale.numel() == scale_elements
        timed_calls.append(partitions)
        return fn(*args), 100.0

    monkeypatch.setitem(globals(), "get_recommended_splits", recommend)
    monkeypatch.setitem(globals(), "run_perftest", run_once)
    overrides = {}
    if query_length != 1:
        overrides["query_length"] = query_length
    if num_partitions is not None:
        overrides["num_partitions"] = num_partitions
    if num_partitions == 5:
        # The exact override must also bypass a smaller automatic clamp.
        overrides["max_partitions"] = 4
    result = run_pa_decode_tile_case(
        batch_size,
        num_query_heads,
        num_kv_heads,
        head_dim,
        context_length,
        block_size,
        dtype,
        trans_v,
        per_token=per_token,
        **overrides,
    )
    assert timed_calls == [expected_partitions]
    assert recommendations == (
        [
            (
                (batch_size, num_kv_heads),
                {
                    "split_kv_blocks": KV_COMPUTE_BLOCK // block_size,
                    "max_partitions": 8,
                },
            )
        ]
        if num_partitions is None
        else []
    )
    assert result["query_length"] == query_length
    assert result["partitions"] == expected_partitions
    assert result["flydsl err"] == 0
    assert result["flydsl us"] == 100.0
    assert result["flydsl us/token"] == pytest.approx(
        100.0 / (batch_size * query_length)
    )
    attended_tokens = (
        query_length * context_length - query_length * (query_length - 1) // 2
    )
    flops = 4 * batch_size * num_query_heads * head_dim * attended_tokens
    blocks_per_sequence = (context_length + block_size - 1) // block_size
    # BF16/FP16 Q/O, FP8 KV, int32 metadata, FP32 scales. KV/scales are not
    # multiplied by MTP length, and the padded part of page128 is excluded.
    logical_bytes = (
        2 * batch_size * query_length * num_query_heads * head_dim * 2
        + 2 * batch_size * num_kv_heads * context_length * head_dim
        + batch_size * blocks_per_sequence * 4
        + batch_size * 4
        + (batch_size * num_kv_heads * context_length if per_token else 1) * 2 * 4
    )
    assert result["flydsl TFLOPS"] == pytest.approx(flops / 100.0 / 1e6)
    assert result["flydsl TB/s"] == pytest.approx(logical_bytes / 100.0 / 1e6)


def _run_accuracy_case(
    lengths,
    *,
    num_query_heads,
    num_kv_heads,
    head_dim,
    block_size,
    query_dtype,
    num_partitions,
    tolerance,
    trans_v=False,
):
    """Run PA against a dequantized-FP8 torch reference without benchmarking."""
    _require_gpu()
    if num_query_heads % num_kv_heads:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    torch.manual_seed(0)
    batch_size = len(lengths)
    pages_per_sequence = [
        (context_length + block_size - 1) // block_size for context_length in lengths
    ]
    max_pages = max(pages_per_sequence)
    num_blocks = sum(pages_per_sequence)

    query = torch.empty(
        batch_size, num_query_heads, head_dim, dtype=query_dtype
    ).uniform_(-0.5, 0.5)
    key = torch.empty(
        num_blocks, num_kv_heads, block_size, head_dim, dtype=query_dtype
    ).uniform_(-0.5, 0.5)
    value = torch.empty(
        num_blocks, num_kv_heads, head_dim, block_size, dtype=query_dtype
    ).uniform_(-0.5, 0.5)

    quant_dtype = _quant_dtype()
    key_quant, key_scale = per_tensor_quant(key, quant_dtype=quant_dtype)
    value_quant, value_scale = per_tensor_quant(value, quant_dtype=quant_dtype)
    key_cache = (
        key_quant.view(num_blocks, num_kv_heads, block_size, head_dim // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    if trans_v:
        value_cache = (
            value_quant.permute(0, 1, 3, 2)
            .contiguous()
            .view(num_blocks, num_kv_heads, block_size // 16, 16, head_dim)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )
    else:
        value_cache = value_quant.contiguous()

    block_tables = torch.zeros(
        batch_size, max_pages, dtype=dtypes.i32, device=query.device
    )
    next_page = 0
    for seq_idx, page_count in enumerate(pages_per_sequence):
        block_tables[seq_idx, :page_count] = torch.arange(
            next_page,
            next_page + page_count,
            dtype=dtypes.i32,
            device=query.device,
        )
        next_page += page_count
    context_lengths = torch.tensor(lengths, dtype=dtypes.i32, device=query.device)

    reference = run_torch(
        query,
        key_quant,
        value_quant,
        block_tables,
        context_lengths,
        key_scale,
        value_scale,
    )
    output = torch.empty_like(query)
    torch.ops.aiter.pa_decode_flydsl(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        head_dim**-0.5,
        1,
        num_partitions,
        KV_COMPUTE_BLOCK,
        quant_dtype,
        None,
        key_scale,
        value_scale,
    )

    torch.testing.assert_close(
        output.float(),
        reference.float(),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize(
    "block_size,query_dtype,num_kv_heads,num_partitions",
    [
        pytest.param(16, dtypes.bf16, 1, 1, id="page16-bf16-hkv1-np1"),
        pytest.param(64, dtypes.fp16, 2, 1, id="page64-fp16-hkv2-np1"),
        pytest.param(128, dtypes.bf16, 2, 4, id="page128-bf16-hkv2-np4"),
        pytest.param(128, dtypes.fp16, 1, 4, id="page128-fp16-hkv1-np4"),
    ],
)
def test_pa_decode_fixed_length_accuracy(
    block_size, query_dtype, num_kv_heads, num_partitions
):
    _run_accuracy_case(
        [257, 257, 257],
        num_query_heads=8,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        block_size=block_size,
        query_dtype=query_dtype,
        num_partitions=num_partitions,
        tolerance=FIXED_LENGTH_ACCURACY_TOLERANCE,
    )


@pytest.mark.parametrize("query_dtype", [dtypes.bf16, dtypes.fp16])
def test_pa_decode_variable_length_accuracy(query_dtype):
    _run_accuracy_case(
        [1, 127, 257, 1027],
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=128,
        block_size=128,
        query_dtype=query_dtype,
        num_partitions=4,
        tolerance=VARIABLE_LENGTH_ACCURACY_TOLERANCE,
    )


def _adversarial_case(
    head_dim=128,
    context_length=257,
    block_size=16,
    query_group_size=8,
    query_length=1,
    per_token=False,
    zero_query=False,
    poison_padding_blocks=False,
    tail_value_scale=None,
    seed=0,
    trans_v=False,
    num_partitions=1,
):
    """Build one case plus its torch reference over the same dequantised fp8 KV.

    ``poison_padding_blocks`` fills every block past the sequence's real extent
    with NaN and pads the block table out to reach them -- the entries a caller
    leaves behind, which the kernel must resolve to block 0 rather than follow.
    """
    quant_dtype = _quant_dtype()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    owned = (context_length + block_size - 1) // block_size
    # A 256-token tile always walks 256/block_size pages, so pad past `owned` to
    # give the block table entries that must not be dereferenced.
    blocks = owned + 256 // block_size if poison_padding_blocks else owned
    tokens = blocks * block_size

    def rand(*shape):
        return torch.rand(*shape, generator=generator, device="cuda") - 0.5

    query = (
        torch.zeros(query_length, query_group_size, head_dim, dtype=dtypes.bf16)
        if zero_query
        else rand(query_length, query_group_size, head_dim).to(dtypes.bf16)
    )
    key = rand(blocks, 1, block_size, head_dim).to(quant_dtype)
    value = rand(blocks, 1, block_size, head_dim).to(quant_dtype)

    invalid = (torch.arange(tokens, device="cuda") >= context_length).view(
        blocks, 1, block_size, 1
    )
    if poison_padding_blocks:
        padding = (torch.arange(tokens, device="cuda") >= owned * block_size).view(
            blocks, 1, block_size, 1
        )
        poison = torch.full_like(value, float("nan"), dtype=dtypes.fp32).to(quant_dtype)
        value = torch.where(padding.expand_as(value), poison, value)
        key = torch.where(padding.expand_as(key), poison, key)

    if per_token:
        key_scale = (
            torch.rand(blocks, 1, block_size, 1, generator=generator, device="cuda")
            * 0.5
            + 0.75
        )
        value_scale = (
            torch.rand(blocks, 1, block_size, 1, generator=generator, device="cuda")
            * 0.5
            + 0.75
        )
        if tail_value_scale is not None:
            value_scale = torch.where(
                invalid, torch.full_like(value_scale, tail_value_scale), value_scale
            )
    else:
        key_scale = torch.ones(1, dtype=dtypes.fp32)
        value_scale = torch.ones(1, dtype=dtypes.fp32)

    key_cache = (
        key.view(blocks, 1, block_size, head_dim // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    if trans_v:
        value_cache = (
            value.view(blocks, 1, block_size // 16, 16, head_dim)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )
    else:
        value_cache = value.permute(0, 1, 3, 2).contiguous()
    block_tables = torch.arange(blocks, dtype=dtypes.i32).reshape(1, blocks)
    context_lengths = torch.full((1,), context_length, dtype=dtypes.i32)

    keys = key.float().reshape(tokens, head_dim)
    values = value.float().reshape(tokens, head_dim)
    if per_token:
        keys = keys * key_scale.float().reshape(tokens, 1)
        values = values * value_scale.float().reshape(tokens, 1)
    reference = torch.empty(
        query_length, query_group_size, head_dim, dtype=dtypes.fp32, device="cuda"
    )
    for position in range(query_length):
        # MTP position `p` sees context_length - (query_length - 1) + p tokens.
        visible = context_length - (query_length - 1) + position
        scores = query[position].float() @ keys[:visible].T * head_dim**-0.5
        reference[position] = torch.softmax(scores, dim=-1) @ values[:visible]

    output = torch.empty_like(query)
    torch.ops.aiter.pa_decode_flydsl(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        head_dim**-0.5,
        query_length,
        num_partitions,
        256,
        quant_dtype,
        None,
        key_scale,
        value_scale,
    )
    return output.float(), reference


def _assert_matches(output, reference, tolerance=0.06):
    assert torch.isfinite(output).all(), (
        f"{int((~torch.isfinite(output)).sum())} of {output.numel()} outputs "
        "are NaN or inf"
    )
    scale = float(reference.abs().max()) or 1.0
    error = float((output - reference).abs().max()) / scale
    assert error < tolerance, f"relative error {error:.3%} exceeds {tolerance:.1%}"


@pytest.mark.parametrize("query_length", [1, 4])
@pytest.mark.parametrize("per_token", [False, True])
def test_zero_query_stays_finite(query_length, per_token):
    """An all-zero query row quantises to scale 0; -inf * 0 must not reach exp2."""
    _require_gpu()
    output, reference = _adversarial_case(
        query_length=query_length, per_token=per_token, zero_query=True
    )
    _assert_matches(output, reference)


@pytest.mark.parametrize("query_dtype", [dtypes.bf16, dtypes.fp16])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize(
    "query_group_size,query_length,block_size,context_length,num_partitions,trans_v,split_values",
    [
        pytest.param(32, 1, 64, 128, 1, False, False, id="decode"),
        pytest.param(16, 3, 128, 257, 4, True, False, id="mtp-partitions"),
        pytest.param(8, 3, 128, 257, 1, True, False, id="padded-m-tile"),
        pytest.param(32, 1, 64, 512, 1, False, True, id="history"),
    ],
)
def test_large_negative_logits_preserve_online_softmax(
    query_group_size,
    query_length,
    block_size,
    context_length,
    num_partitions,
    trans_v,
    split_values,
    per_token,
    query_dtype,
):
    """Large negative logits must preserve both initial and accumulated state."""
    _require_gpu()
    head_dim = 128
    blocks = (context_length + block_size - 1) // block_size
    quant_dtype = _quant_dtype()
    query = torch.full(
        (query_length, query_group_size, head_dim), -3.0, dtype=query_dtype
    )
    key = torch.full((blocks, 1, block_size, head_dim), 3.0).to(quant_dtype)
    value = torch.ones(blocks, 1, block_size, head_dim)
    if split_values:
        # With equal logits, both KV tiles contribute equally. Resetting the
        # correction to zero on every tile would incorrectly return 1, not 0.5.
        value[: KV_COMPUTE_BLOCK // block_size] = 0
    value = value.to(quant_dtype)
    key_cache = (
        key.view(blocks, 1, block_size, head_dim // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    if trans_v:
        value_cache = (
            value.view(blocks, 1, block_size // 16, 16, head_dim)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )
    else:
        value_cache = value.permute(0, 1, 3, 2).contiguous()
    block_tables = torch.arange(blocks, dtype=dtypes.i32).reshape(1, blocks)
    context_lengths = torch.full((1,), context_length, dtype=dtypes.i32)
    scale = torch.ones(
        (blocks, 1, block_size, 1) if per_token else (1,), dtype=dtypes.fp32
    )
    output = torch.full_like(query, float("nan"))
    torch.ops.aiter.pa_decode_flydsl(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        head_dim**-0.5,
        query_length,
        num_partitions,
        KV_COMPUTE_BLOCK,
        quant_dtype,
        None,
        scale,
        scale,
    )
    reference = torch.full_like(output, 0.5 if split_values else 1.0, dtype=dtypes.fp32)
    _assert_matches(output.float(), reference, tolerance=0.02)


@pytest.mark.parametrize("block_size", [16, 64, 128])
@pytest.mark.parametrize("context_length", [1, 257, 1027])
def test_block_table_padding_is_not_dereferenced(block_size, context_length):
    """Entries past the sequence's extent must resolve to block 0.

    A 256-token tile always walks 256/block_size pages, so it reads block-table
    entries the sequence does not own. Those hold whatever the caller left --
    often a stale id pointing at another sequence's live page. The tokens behind
    them get a zero probability, but the PV matmul still multiplies their V
    bytes (``0 * NaN == NaN``), so the page index itself has to be pinned.

    Not covered here, by design: the unwritten tail of the last page the
    sequence does own. Those slots are inside a real block and the caller is
    expected to leave them finite.
    """
    _require_gpu()
    output, reference = _adversarial_case(
        context_length=context_length,
        block_size=block_size,
        poison_padding_blocks=True,
    )
    _assert_matches(output, reference)


@pytest.mark.parametrize("query_length", [1, 4])
@pytest.mark.parametrize("context_length", [1, 257])
def test_extreme_tail_value_scale_is_ignored(context_length, query_length):
    """A huge scale on an unwritten slot must not underflow valid probabilities.

    Only finite tails are covered: scale tensors are allocated zeroed, so a NaN
    scale past ``context_len`` is not a case the kernel has to survive.
    """
    _require_gpu()
    if context_length < query_length:
        pytest.skip("MTP position 0 would see a non-positive causal bound")
    output, reference = _adversarial_case(
        context_length=context_length,
        query_length=query_length,
        per_token=True,
        tail_value_scale=1e9,
    )
    _assert_matches(output, reference)


@pytest.mark.parametrize("num_partitions", [1, 4, 8])
@pytest.mark.parametrize("trans_v", [False, True])
@pytest.mark.parametrize("block_size", [16, 128])
def test_per_token_scale_extremes_with_mixed_lengths(
    block_size, trans_v, num_partitions
):
    """Per-token normalization ignores tails and survives odd split starts.

    K is zero, so attention is uniform and the expected output is simply the
    mean of the valid per-token V scales.  Valid scales vary by 256-token tile
    and 64-token wave, the first tile has an exactly-zero maximum, and unused
    slots in the final physical page carry a hostile 1e9 scale.  Lengths 769
    and 2305 put a one-token tail after three and nine full tiles respectively;
    with four/eight partitions they also exercise odd global tile indices in
    the ping-pong scale buffer.  The empty row checks the neutral result.
    """
    _require_gpu()
    lengths = [0, 769, 2305]
    batch_size, num_query_heads, num_kv_heads, head_dim = 3, 16, 1, 128
    pages_per_sequence = [(length + block_size - 1) // block_size for length in lengths]
    max_pages = max(pages_per_sequence)
    num_pages = sum(pages_per_sequence)
    quant_dtype = _quant_dtype()

    query = torch.ones(batch_size, num_query_heads, head_dim, dtype=dtypes.bf16)
    key_cache = torch.zeros(
        num_pages,
        num_kv_heads,
        head_dim // 16,
        block_size,
        16,
        dtype=quant_dtype,
    )
    value_plain = torch.ones(
        num_pages, num_kv_heads, head_dim, block_size, dtype=quant_dtype
    )
    if trans_v:
        value_cache = (
            value_plain.view(
                num_pages,
                num_kv_heads,
                head_dim,
                block_size // 16,
                16,
            )
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
    else:
        value_cache = value_plain

    block_tables = torch.zeros(batch_size, max_pages, dtype=dtypes.i32)
    key_scale = torch.full(
        (num_pages, num_kv_heads, block_size, 1), 1e9, dtype=dtypes.fp32
    )
    value_scale = torch.full_like(key_scale, 1e9)
    expected_values = []
    next_page = 0
    for seq, (length, page_count) in enumerate(zip(lengths, pages_per_sequence)):
        if length == 0:
            expected_values.append(torch.zeros((), dtype=dtypes.fp32))
            continue
        block_tables[seq, :page_count] = torch.arange(
            next_page, next_page + page_count, dtype=dtypes.i32
        )
        capacity = page_count * block_size
        token = torch.arange(capacity, dtype=dtypes.i32)
        tile = token // KV_COMPUTE_BLOCK
        wave = (token % KV_COMPUTE_BLOCK) // 64
        exponent = ((tile + wave) % 5 - 4).to(dtypes.fp32)
        scales = torch.exp2(exponent)
        scales = torch.where(token < KV_COMPUTE_BLOCK, 0.0, scales)
        scales = torch.where(token < length, scales, 1e9)
        page_scales = scales.reshape(page_count, block_size)
        key_scale[next_page : next_page + page_count, 0, :, 0] = page_scales
        value_scale[next_page : next_page + page_count, 0, :, 0] = page_scales
        expected_values.append(scales[:length].mean())
        next_page += page_count

    context_lengths = torch.tensor(lengths, dtype=dtypes.i32)
    output = torch.full_like(query, float("nan"))
    pa_decode(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        head_dim**-0.5,
        1,
        num_partitions,
        KV_COMPUTE_BLOCK,
        quant_dtype,
        None,
        key_scale,
        value_scale,
    )

    expected = (
        torch.stack(expected_values)
        .reshape(batch_size, 1, 1)
        .expand(batch_size, num_query_heads, head_dim)
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), expected, rtol=0.005, atol=0.005)


@pytest.mark.parametrize("trans_v", [False, True])
@pytest.mark.parametrize("block_size", [16, 128])
def test_per_token_sparse_pages_with_odd_partition_start(block_size, trans_v):
    """Page and scale prefetches stay aligned across odd-starting partitions.

    Ten compute tiles split four ways start at 0, 3, 6 and 9. The partition
    starting at 3 therefore both starts in the odd scale buffer and reuses it.
    Nontrivial K/V and independently varying scales expose mistakes hidden by
    uniform attention. Selected physical pages are shuffled and separated by
    unused pages, so a logical-page/physical-page mixup cannot pass unnoticed.
    """
    _require_gpu()
    context_length, num_partitions = 2305, 4
    num_query_heads, head_dim = 16, 128
    logical_pages = (context_length + block_size - 1) // block_size
    num_pages = 2 * logical_pages + 1
    quant_dtype = _quant_dtype()
    generator = torch.Generator(device="cuda").manual_seed(23)
    pages = 2 * torch.randperm(logical_pages, generator=generator) + 1
    token = torch.arange(logical_pages * block_size)
    tile = token // KV_COMPUTE_BLOCK
    wave = (token % KV_COMPUTE_BLOCK) // 64
    dim = torch.arange(head_dim)
    signs = (1 - 2 * (dim % 2)).float()
    query_factors = 0.5 + 0.25 * (torch.arange(num_query_heads) % 3).float()
    query = (query_factors[:, None] * signs[None, :]).unsqueeze(0).to(dtypes.bf16)

    key = torch.rand(num_pages, 1, block_size, head_dim, generator=generator) * 0.25
    value = torch.rand(num_pages, 1, block_size, head_dim, generator=generator) - 0.5
    # Correlated logits and tile-dependent values make a wrong K scale affect
    # the answer, while variation within pages exercises every token lane.
    logical_key = (0.125 + 0.0625 * ((tile + wave) % 4))[:, None] * signs[None, :]
    logical_key += 0.015625 * ((token[:, None] + dim[None, :]) % 3 - 1)
    logical_value = (
        0.25 * (tile % 3 - 1)[:, None]
        + 0.0625 * (token // 16 % 3 - 1)[:, None]
        + 0.03125 * (dim % 5 - 2)[None, :]
    )
    key[pages] = logical_key.reshape(logical_pages, 1, block_size, head_dim)
    value[pages] = logical_value.reshape(logical_pages, 1, block_size, head_dim)
    key = key.to(quant_dtype)
    value = value.to(quant_dtype)
    key_scale = torch.ones(num_pages, 1, block_size, 1, dtype=dtypes.fp32)
    value_scale = torch.ones_like(key_scale)
    key_scale[pages, 0, :, 0] = torch.exp2(
        ((2 * tile + wave + token % 3) % 4 - 2).float()
    ).reshape(logical_pages, block_size)
    value_scale[pages, 0, :, 0] = torch.exp2(
        ((tile + 2 * wave + token % 5) % 4 - 3).float()
    ).reshape(logical_pages, block_size)

    key_cache = (
        key.view(num_pages, 1, block_size, head_dim // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    value_plain = value.permute(0, 1, 3, 2).contiguous()
    value_cache = (
        value.view(num_pages, 1, block_size // 16, 16, head_dim)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
        if trans_v
        else value_plain
    )
    block_tables = pages.to(dtypes.i32).unsqueeze(0)
    context_lengths = torch.tensor([context_length], dtype=dtypes.i32)
    reference = run_torch(
        query.float(),
        key,
        value_plain,
        block_tables,
        context_lengths,
        key_scale,
        value_scale,
    )
    output = torch.full_like(query, float("nan"))
    torch.ops.aiter.pa_decode_flydsl(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        head_dim**-0.5,
        1,
        num_partitions,
        KV_COMPUTE_BLOCK,
        quant_dtype,
        None,
        key_scale,
        value_scale,
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), reference, rtol=0.005, atol=0.005)


def _run_mtp4_fused_reference_case(
    lengths, num_kv_heads, block_size, trans_v, num_partitions
):
    """Check MTP4 with sparse pages, varying scales, and a causal FP32 reference."""
    torch.manual_seed(37)
    query_length, query_group_size, head_dim = 4, 16, 128
    batch_size = len(lengths)
    num_query_heads = num_kv_heads * query_group_size
    pages_per_sequence = [(length + block_size - 1) // block_size for length in lengths]
    max_pages = max(pages_per_sequence)
    num_pages = sum(pages_per_sequence)
    quant_dtype = _quant_dtype()

    query = torch.empty(
        batch_size * query_length,
        num_query_heads,
        head_dim,
        dtype=dtypes.bf16,
    ).uniform_(-0.5, 0.5)
    key = torch.empty(
        num_pages, num_kv_heads, block_size, head_dim, dtype=dtypes.bf16
    ).uniform_(-0.5, 0.5)
    value = torch.empty(
        num_pages, num_kv_heads, head_dim, block_size, dtype=dtypes.bf16
    ).uniform_(-0.5, 0.5)
    key_quant, key_scale = pertoken_quant(key, quant_dtype=quant_dtype)
    value_token_quant, value_scale = pertoken_quant(
        value.permute(0, 1, 3, 2).contiguous(), quant_dtype=quant_dtype
    )
    value_quant = value_token_quant.permute(0, 1, 3, 2).contiguous()

    token = torch.arange(num_pages * block_size).reshape(num_pages, 1, block_size, 1)
    kv_head = torch.arange(num_kv_heads).reshape(1, num_kv_heads, 1, 1)
    key_scale *= torch.exp2(((2 * token + kv_head) % 4 - 2).float())
    value_scale *= torch.exp2(((token + 2 * kv_head) % 5 - 3).float())

    # Make C1--C4 analytically exact while still detecting causal off-by-one:
    # K=0 gives uniform attention, V progresses as [.25, .5, .75, 1], and
    # scale=1. A C4 row therefore produces [.25, .375, .5, .625].
    data_page = 0
    for length, page_count in zip(lengths, pages_per_sequence):
        if 0 < length <= query_length:
            capacity = page_count * block_size
            token_values = (
                torch.arange(capacity, dtype=dtypes.fp32) % query_length + 1
            ) * 0.25
            key_quant[data_page : data_page + page_count].zero_()
            value_quant[data_page : data_page + page_count] = (
                token_values.reshape(page_count, 1, 1, block_size)
                .expand(page_count, num_kv_heads, head_dim, block_size)
                .to(quant_dtype)
            )
            key_scale[data_page : data_page + page_count].fill_(1.0)
            value_scale[data_page : data_page + page_count].fill_(1.0)
        data_page += page_count

    # Scatter logical pages into a permuted set of odd physical pages. Leave
    # holes and page zero finite so masked/padded token reads remain valid.
    selected_pages = 2 * torch.randperm(num_pages) + 1
    physical_page_count = 2 * num_pages + 1

    def scatter_pages(tensor):
        sparse = torch.zeros(
            (physical_page_count, *tensor.shape[1:]), dtype=tensor.dtype
        )
        sparse[selected_pages] = tensor
        return sparse

    key_quant = scatter_pages(key_quant)
    value_quant = scatter_pages(value_quant)
    key_scale = scatter_pages(key_scale)
    value_scale = scatter_pages(value_scale)
    num_pages = physical_page_count

    key_cache = (
        key_quant.view(
            num_pages,
            num_kv_heads,
            block_size,
            head_dim // 16,
            16,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    if trans_v:
        value_cache = (
            value_quant.permute(0, 1, 3, 2)
            .contiguous()
            .view(
                num_pages,
                num_kv_heads,
                block_size // 16,
                16,
                head_dim,
            )
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )
    else:
        value_cache = value_quant

    block_tables = torch.zeros(batch_size, max_pages, dtype=dtypes.i32)
    next_page = 0
    for seq, page_count in enumerate(pages_per_sequence):
        block_tables[seq, :page_count] = selected_pages[
            next_page : next_page + page_count
        ]
        next_page += page_count
    context_lengths = torch.tensor(lengths, dtype=dtypes.i32)

    reference = torch.zeros_like(query, dtype=dtypes.fp32)
    for seq, length in enumerate(lengths):
        if length == 0:
            continue
        token_ids = torch.arange(length)
        logical_pages = token_ids // block_size
        token_offsets = token_ids % block_size
        physical_pages = block_tables[seq, logical_pages].long()
        keys = key_quant[physical_pages, :, token_offsets, :].float()
        values = value_quant[physical_pages, :, :, token_offsets].float()
        keys *= key_scale[physical_pages, :, token_offsets, 0].float().unsqueeze(-1)
        values *= value_scale[physical_pages, :, token_offsets, 0].float().unsqueeze(-1)
        keys = keys.repeat_interleave(query_group_size, dim=1)
        values = values.repeat_interleave(query_group_size, dim=1)
        for position in range(query_length):
            visible = max(0, length - (query_length - 1) + position)
            if visible == 0:
                continue
            row = seq * query_length + position
            scores = (
                torch.einsum("hd,khd->hk", query[row].float(), keys[:visible])
                * head_dim**-0.5
            )
            probs = torch.softmax(scores, dim=-1)
            reference[row] = torch.einsum("hk,khd->hd", probs, values[:visible])

    output = torch.full_like(query, float("nan"))
    torch.ops.aiter.pa_decode_flydsl(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        head_dim**-0.5,
        query_length,
        num_partitions,
        KV_COMPUTE_BLOCK,
        quant_dtype,
        None,
        key_scale,
        value_scale,
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), reference, rtol=0.005, atol=0.005)


@pytest.mark.parametrize("num_partitions", [1, 8])
@pytest.mark.parametrize("trans_v", [False, True])
@pytest.mark.parametrize("block_size", [16, 128])
@pytest.mark.parametrize(
    "lengths,num_kv_heads",
    [
        pytest.param([4], 1, id="c4-hkv1"),
        pytest.param([257], 1, id="c257-hkv1"),
        pytest.param([259], 1, id="c259-hkv1"),
        pytest.param([2305], 1, id="c2305-hkv1"),
        pytest.param([2307], 1, id="c2307-hkv1"),
        pytest.param(
            [0, 1, 2, 3, 4, 255, 256, 257, 258, 259, 2303, 2304, 2305, 2306, 2307],
            1,
            id="mixed-hkv1",
        ),
        pytest.param([257], 2, id="c257-hkv2"),
        pytest.param([259], 2, id="c259-hkv2"),
    ],
)
def test_mtp4_fused_phase_b_accuracy(
    monkeypatch, lengths, num_kv_heads, block_size, trans_v, num_partitions
):
    """Force the fused path and cover its MTP causal/partition boundaries."""
    _require_gpu()
    pa_decode_module = importlib.import_module("aiter.ops.flydsl.pa_decode")
    compile_tile = pa_decode_module.compile_pa_decode_tile

    def compile_fused_mtp4(**kwargs):
        kwargs["query_splits"] = 1
        return compile_tile(**kwargs)

    monkeypatch.setattr(pa_decode_module, "compile_pa_decode_tile", compile_fused_mtp4)
    _run_mtp4_fused_reference_case(
        lengths, num_kv_heads, block_size, trans_v, num_partitions
    )


@pytest.mark.parametrize("trans_v", [False, True])
@pytest.mark.parametrize("block_size", [16, 128])
def test_mtp4_public_selector_fused_path(monkeypatch, block_size, trans_v):
    """B32/NP8 selects the fused one-CTA-per-sequence MTP4 path."""
    _require_gpu()
    pa_decode_module = importlib.import_module("aiter.ops.flydsl.pa_decode")
    compile_tile = pa_decode_module.compile_pa_decode_tile
    selected_splits = []

    def capture_query_splits(**kwargs):
        selected_splits.append(kwargs.get("query_splits", 1))
        return compile_tile(**kwargs)

    monkeypatch.setattr(
        pa_decode_module, "compile_pa_decode_tile", capture_query_splits
    )
    _run_mtp4_fused_reference_case([257, 259] * 16, 1, block_size, trans_v, 8)
    assert selected_splits and set(selected_splits) == {1}


@pytest.mark.parametrize(
    "head_dim,num_partitions",
    [(64, 1), (256, 1), (1024, 1), (1024, 4)],
)
def test_additional_supported_head_dims_are_accurate(head_dim, num_partitions):
    _require_gpu()
    output, reference = _adversarial_case(
        head_dim=head_dim,
        context_length=64,
        num_partitions=num_partitions,
    )
    _assert_matches(output, reference)


def _constant_page_decode_case(
    lengths,
    block_size,
    num_partitions,
    block_table_width,
    poison_padding=False,
    per_token=False,
    trans_v=False,
):
    """Each sequence's V pages are constant, so its attention output is known."""
    _require_gpu()
    batch_size, query_heads, head_dim = len(lengths), 8, 128
    pages_per_sequence = [(length + block_size - 1) // block_size for length in lengths]
    page_values = [
        float(seq + 1)
        for seq, pages in enumerate(pages_per_sequence)
        for _ in range(pages)
    ]
    num_pages = len(page_values)
    quant_dtype = _quant_dtype()
    query = torch.ones(batch_size, query_heads, head_dim, dtype=dtypes.bf16)
    key_cache = torch.zeros(
        num_pages, 1, head_dim // 16, block_size, 16, dtype=quant_dtype
    )
    value_cache = (
        torch.tensor(page_values, dtype=dtypes.fp32)
        .reshape(num_pages, 1, 1, 1)
        .expand(num_pages, 1, head_dim, block_size)
        .contiguous()
        .to(quant_dtype)
    )
    if trans_v:
        value_cache = (
            value_cache.view(num_pages, 1, head_dim, block_size // 16, 16)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
    block_tables = torch.full(
        (batch_size, block_table_width),
        num_pages + 1024 if poison_padding else 0,
        dtype=dtypes.i32,
    )
    next_page = 0
    for seq, pages in enumerate(pages_per_sequence):
        block_tables[seq, :pages] = torch.arange(
            next_page, next_page + pages, dtype=dtypes.i32
        )
        next_page += pages
    context_lengths = torch.tensor(lengths, dtype=dtypes.i32)
    scale = torch.ones(
        (num_pages, 1, block_size, 1) if per_token else (1,), dtype=dtypes.fp32
    )
    output = torch.full_like(query, float("nan"))
    torch.ops.aiter.pa_decode_flydsl(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        head_dim**-0.5,
        1,
        num_partitions,
        256,
        quant_dtype,
        None,
        scale,
        scale,
    )
    # The reference uses the values actually representable in the FP8 cache.
    expected = (
        torch.tensor(
            [float(seq + 1) if length else 0.0 for seq, length in enumerate(lengths)],
            dtype=dtypes.fp32,
        )
        .to(quant_dtype)
        .to(dtypes.fp32)
        .reshape(batch_size, 1, 1)
        .expand_as(output)
    )
    _assert_matches(output.float(), expected, tolerance=0.02)


@pytest.mark.parametrize("num_partitions", [1, 4])
@pytest.mark.parametrize("block_table_width", [1, 2, 3, 4, 5])
def test_block_table_row_alignment(block_table_width, num_partitions):
    """A contiguous block table need not have four-entry-aligned row starts."""
    _constant_page_decode_case([16, 16, 16], 16, num_partitions, block_table_width)


@pytest.mark.parametrize("num_partitions", [1, 4])
@pytest.mark.parametrize("block_size", [16, 64, 128])
@pytest.mark.parametrize(
    "lengths,per_token",
    [
        pytest.param([64, 0], False, id="mixed"),
        pytest.param([64, 0], True, id="mixed-per-token"),
        pytest.param([0, 0], False, id="all-empty"),
    ],
)
def test_empty_contexts_do_not_read_kv(lengths, per_token, block_size, num_partitions):
    """Empty rows must not follow padding or read an empty KV allocation."""
    width = 2 * KV_COMPUTE_BLOCK // block_size if any(lengths) else 0
    _constant_page_decode_case(
        lengths,
        block_size,
        num_partitions,
        width,
        poison_padding=True,
        per_token=per_token,
    )


@pytest.mark.parametrize("context_length", [63, 64, 65, 127, 128, 129, 257])
@pytest.mark.parametrize("trans_v", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
def test_page128_chunk_boundaries(context_length, trans_v, per_token):
    """Two waves share a page-128, but must load different K/V/scales halves."""
    _require_gpu()
    output, reference = _adversarial_case(
        block_size=128,
        context_length=context_length,
        query_group_size=16,
        trans_v=trans_v,
        per_token=per_token,
        num_partitions=4,
    )
    _assert_matches(output, reference)


@pytest.mark.parametrize("block_size", [16, 128])
@pytest.mark.parametrize("context_length", [2048, 100000, 200000])
@pytest.mark.parametrize("query_length", [1, 3])
@pytest.mark.parametrize("per_token", [False, True])
def test_tp4_contexts(block_size, context_length, query_length, per_token):
    """64 Q / 4 KV heads under TP4: Hq16, Hkv1, D128, including dense MTP.

    Page size decides how many block-table entries a 256-token tile walks --
    two at page-128, sixteen at page-16 -- so both served page sizes are run
    against every context.
    """
    _require_gpu()
    output, reference = _adversarial_case(
        block_size=block_size,
        query_group_size=16,
        context_length=context_length,
        query_length=query_length,
        trans_v=True,
        per_token=per_token,
        num_partitions=8,
    )
    _assert_matches(output, reference)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("block_size", [16, 128])
@pytest.mark.parametrize("context_length", [2048, 100000, 200000])
def test_tp4_decode_batches(batch_size, block_size, context_length):
    """Served TP4 decode grid: Hq16, Hkv1, D128, transposed V.

    Batch is the only axis that moves the CTA count, and at NP=8 the sweep
    straddles the tuned V-prefetch window (256 < batch * 8 <= 512 on a 256-CU
    part), so page-128 batch 64 compiles the prefetch specialization while the
    smaller batches compile the default schedule. Page 16 never enters that
    window -- the gate also requires block_size == 128 -- so it covers the
    sixteen-pages-per-tile block-table walk over the same grids instead.
    """
    _run_accuracy_case(
        [context_length] * batch_size,
        num_query_heads=16,
        num_kv_heads=1,
        head_dim=128,
        block_size=block_size,
        query_dtype=dtypes.bf16,
        num_partitions=8,
        tolerance=FIXED_LENGTH_ACCURACY_TOLERANCE,
        trans_v=True,
    )


@pytest.mark.parametrize("trans_v", [False, True])
def test_page128_cache_above_2gib(trans_v):
    """Large batches at 200k context cross the signed-i32 cache offset limit."""
    _require_gpu()
    block_size, head_dim = 128, 128
    boundary_page = 2**31 // (block_size * head_dim)
    pages = boundary_page + 2
    quant_dtype = _quant_dtype()
    key_cache = torch.empty(pages, 1, 8, block_size, 16, dtype=quant_dtype)
    value_shape = (
        (pages, 1, 8, head_dim, 16) if trans_v else (pages, 1, head_dim, block_size)
    )
    value_cache = torch.empty(value_shape, dtype=quant_dtype)
    key_cache[0].zero_()
    value_cache[0].zero_()
    page_ids = [boundary_page - 1, boundary_page, boundary_page + 1]
    for index, page in enumerate(page_ids):
        key_cache[page].zero_()
        value_cache[page].fill_(index + 1)
    query = torch.ones(1, 16, head_dim, dtype=dtypes.bf16)
    output = torch.empty_like(query)
    scale = torch.ones(1, dtype=dtypes.fp32)
    torch.ops.aiter.pa_decode_flydsl(
        output,
        query,
        key_cache,
        value_cache,
        torch.tensor([257], dtype=dtypes.i32),
        torch.tensor([page_ids], dtype=dtypes.i32),
        head_dim**-0.5,
        1,
        8,
        256,
        quant_dtype,
        None,
        scale,
        scale,
    )
    expected = torch.full_like(output, (128 + 2 * 128 + 3) / 257, dtype=dtypes.fp32)
    _assert_matches(output.float(), expected, tolerance=0.02)


@pytest.mark.parametrize(
    "block_size,per_token,packed",
    [
        (16, False, False),
        (16, True, False),
        (16, False, True),
        (128, False, False),
        (128, True, False),
    ],
)
def test_prepared_sparse_page_tables(block_size, per_token, packed):
    """PA consumes selected pages; each MTP query already has its own table.

    Select 2048 tokens from a 200k-token cache. In the page-16 adapter each
    selected logical page-128 expands to eight physical pages. Packed K/V
    sides share a block and V has a different base pointer and cache extent.
    """
    _require_gpu()
    torch.manual_seed(7)
    rows, query_heads, head_dim = 6, 16, 128  # two requests with three MTP queries
    logical_blocks = (200000 + 127) // 128
    pages_per_logical = 128 // block_size
    page_stride = pages_per_logical * (2 if packed else 1)
    num_pages = logical_blocks * page_stride
    quant_dtype = _quant_dtype()
    key = (torch.rand(num_pages, 1, block_size, head_dim) - 0.5).to(quant_dtype)
    value = (torch.rand_like(key, dtype=dtypes.fp32) - 0.5).to(quant_dtype)
    key_cache = (
        key.view(num_pages, 1, block_size, head_dim // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    value_cache = (
        value.view(num_pages, 1, block_size // 16, 16, head_dim)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )
    if packed:
        # Mirror the adapter's contiguous, overlapping views into interleaved
        # K/V storage. Only K-side page ids are put into the block table.
        value_cache = key_cache.flatten()[
            pages_per_logical * block_size * head_dim :
        ].view(num_pages - pages_per_logical, 1, block_size // 16, head_dim, 16)
        value = value_cache.permute(0, 1, 2, 4, 3).reshape(
            num_pages - pages_per_logical, 1, block_size, head_dim
        )

    query = (torch.rand(rows, query_heads, head_dim) - 0.5).to(dtypes.bf16)
    lengths = [2046, 2047, 2048] * 2
    # Spare entry deliberately leaves page-16 rows non-vector-aligned.
    block_tables = torch.full(
        (rows, 16 * pages_per_logical + 1), num_pages + 1024, dtype=dtypes.i32
    )
    if per_token:
        key_scale = 0.5 + torch.rand(num_pages, 1, block_size, 1)
        value_scale = 0.5 + torch.rand_like(key_scale)
    else:
        key_scale = torch.ones(1, dtype=dtypes.fp32)
        value_scale = torch.ones(1, dtype=dtypes.fp32)
    reference = torch.empty_like(query, dtype=dtypes.fp32)
    for row, length in enumerate(lengths):
        selected = torch.randperm(logical_blocks - 1)[:15]
        # Partial current block comes last; earlier selections differ per query.
        selected = torch.cat((selected, selected.new_tensor([logical_blocks - 1])))
        pages = (
            selected[:, None] * page_stride + torch.arange(pages_per_logical)[None, :]
        ).flatten()
        block_tables[row, : pages.numel()] = pages.to(dtypes.i32)
        keys = key[pages].float()
        values = value[pages].float()
        if per_token:
            keys = keys * key_scale[pages]
            values = values * value_scale[pages]
        keys = keys.reshape(-1, head_dim)[:length]
        values = values.reshape(-1, head_dim)[:length]
        scores = query[row].float() @ keys.T * head_dim**-0.5
        reference[row] = torch.softmax(scores, dim=-1) @ values
    output = torch.empty_like(query)
    torch.ops.aiter.pa_decode_flydsl(
        output,
        query,
        key_cache,
        value_cache,
        torch.tensor(lengths, dtype=dtypes.i32),
        block_tables,
        head_dim**-0.5,
        1,
        8,
        256,
        quant_dtype,
        None,
        key_scale,
        value_scale,
        sliding_window=-1,
    )
    _assert_matches(output.float(), reference)


@pytest.mark.parametrize("context_length", [3, 257, 2051])
@pytest.mark.parametrize("num_partitions", [1, 4])
@pytest.mark.parametrize("trans_v", [False, True])
@pytest.mark.parametrize("zero_query", [False, True])
def test_mtp3_page128_partition_boundaries(
    context_length, num_partitions, trans_v, zero_query
):
    """MTP rows retain independent softmax state across partial/empty partitions."""
    _require_gpu()
    output, reference = _adversarial_case(
        head_dim=128,
        context_length=context_length,
        block_size=128,
        query_group_size=16,
        query_length=3,
        num_partitions=num_partitions,
        trans_v=trans_v,
        zero_query=zero_query,
    )
    _assert_matches(output, reference)


@pytest.mark.parametrize("all_empty", [False, True])
def test_page128_decode_prefetch_mixed_contexts(all_empty):
    """A two-CTA-per-CU decode grid handles empty partitions and poisoned padding."""
    _require_gpu()
    lengths = [0] * 64 if all_empty else [0, 1, 63, 127, 128, 129, 257, 2051] * 8
    _constant_page_decode_case(
        lengths,
        block_size=128,
        num_partitions=8,
        block_table_width=0 if all_empty else 17,
        poison_padding=True,
        trans_v=True,
    )


@pytest.mark.parametrize(
    "query_group_size,query_length", [(8, 3), (16, 2), (16, 4), (16, 5)]
)
@pytest.mark.parametrize("context_length", [129, 1027])
def test_page128_reduction_rows_across_m_tiles(
    query_group_size, query_length, context_length
):
    """Aligned scratch rows stay independent across padded M-tiles and partitions."""
    _require_gpu()
    output, reference = _adversarial_case(
        head_dim=128,
        context_length=context_length,
        block_size=128,
        query_group_size=query_group_size,
        query_length=query_length,
        num_partitions=4,
        trans_v=True,
    )
    _assert_matches(output, reference)


@pytest.mark.parametrize("query_splits", [2, 4])
@pytest.mark.parametrize("num_partitions", [1, 8])
@pytest.mark.parametrize("block_size", [16, 128])
@pytest.mark.parametrize("trans_v", [False, True])
@pytest.mark.parametrize("context_length", [4, 257, 2305])
def test_per_token_mtp_query_splits(
    monkeypatch, query_splits, num_partitions, block_size, trans_v, context_length
):
    """Query-split CTAs preserve MTP causality and the unsplit partial layout."""
    _require_gpu()
    pa_decode_module = importlib.import_module("aiter.ops.flydsl.pa_decode")
    compile_tile = pa_decode_module.compile_pa_decode_tile

    def compile_split_tile(**kwargs):
        kwargs["query_splits"] = query_splits
        kwargs["prefetch_v"] = query_splits == 4
        return compile_tile(**kwargs)

    monkeypatch.setattr(pa_decode_module, "compile_pa_decode_tile", compile_split_tile)
    if context_length == 4:
        # An exact short-prefix case isolates the causal/query mapping. With
        # random V scales, even the unsplit FP8 kernel has >0.005 absolute
        # quantization error at four tokens; do not weaken the assertion.
        quant_dtype = _quant_dtype()
        query = torch.ones(4, 16, 128, dtype=dtypes.bf16)
        key_cache = torch.zeros(1, 1, 8, block_size, 16, dtype=quant_dtype)
        value = torch.zeros(1, 1, block_size, 128, dtype=dtypes.fp32)
        value[0, 0, :4] = (
            (torch.arange(4).float() + 1)[:, None]
            * 0.125
            * (1 + 0.25 * (torch.arange(128) % 4).float())[None, :]
        )
        value = value.to(quant_dtype)
        value_cache = (
            value.view(1, 1, block_size // 16, 16, 128)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
            if trans_v
            else value.permute(0, 1, 3, 2).contiguous()
        )
        key_scale = torch.ones(1, 1, block_size, 1, dtype=dtypes.fp32)
        value_scale = torch.ones_like(key_scale)
        value_scale[0, 0, :4, 0] = torch.tensor([0.5, 1.0, 0.25, 1.0])
        prefix_values = (value[0, 0, :4].float() * value_scale[0, 0, :4]).cumsum(0)
        reference = (prefix_values / (torch.arange(4).float() + 1)[:, None])[
            :, None, :
        ].expand(4, 16, 128)
        output = torch.full_like(query, float("nan"))
        torch.ops.aiter.pa_decode_flydsl(
            output,
            query,
            key_cache,
            value_cache,
            torch.tensor([4], dtype=dtypes.i32),
            torch.zeros(1, 1, dtype=dtypes.i32),
            128**-0.5,
            4,
            num_partitions,
            KV_COMPUTE_BLOCK,
            quant_dtype,
            None,
            key_scale,
            value_scale,
        )
        output = output.float()
    else:
        output, reference = _adversarial_case(
            head_dim=128,
            query_group_size=16,
            query_length=4,
            context_length=context_length,
            block_size=block_size,
            trans_v=trans_v,
            per_token=True,
            num_partitions=num_partitions,
        )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, rtol=0.005, atol=0.005)


@pytest.mark.parametrize("num_partitions", [1, 8])
@pytest.mark.parametrize("block_size", [16, 128])
@pytest.mark.parametrize("trans_v", [False, True])
def test_per_token_mtp_query_split_mixed_contexts(num_partitions, block_size, trans_v):
    """Automatic MTP splitting keeps empty and nonempty sequences independent."""
    _require_gpu()
    lengths = [0, 4, 2305]
    batch_size, query_length, query_heads, head_dim = 3, 4, 16, 128
    pages_per_sequence = [(length + block_size - 1) // block_size for length in lengths]
    page_values = [
        (seq + 1) * 0.125
        for seq, count in enumerate(pages_per_sequence)
        for _ in range(count)
    ]
    num_pages = len(page_values)
    quant_dtype = _quant_dtype()
    query = torch.ones(
        batch_size * query_length, query_heads, head_dim, dtype=dtypes.bf16
    )
    key_cache = torch.zeros(
        num_pages, 1, head_dim // 16, block_size, 16, dtype=quant_dtype
    )
    value_plain = (
        torch.tensor(page_values, dtype=dtypes.fp32)
        .reshape(num_pages, 1, 1, 1)
        .expand(num_pages, 1, head_dim, block_size)
        .contiguous()
        .to(quant_dtype)
    )
    value_cache = (
        value_plain.view(num_pages, 1, head_dim, block_size // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        if trans_v
        else value_plain
    )
    block_tables = torch.full(
        (batch_size, max(pages_per_sequence)), num_pages + 1024, dtype=dtypes.i32
    )
    next_page = 0
    for seq, count in enumerate(pages_per_sequence):
        block_tables[seq, :count] = torch.arange(next_page, next_page + count)
        next_page += count
    scales = torch.ones(num_pages, 1, block_size, 1, dtype=dtypes.fp32)
    output = torch.full_like(query, float("nan"))
    torch.ops.aiter.pa_decode_flydsl(
        output,
        query,
        key_cache,
        value_cache,
        torch.tensor(lengths, dtype=dtypes.i32),
        block_tables,
        head_dim**-0.5,
        query_length,
        num_partitions,
        KV_COMPUTE_BLOCK,
        quant_dtype,
        None,
        scales,
        scales,
    )
    expected = (
        torch.tensor([0.0, 0.25, 0.375], dtype=dtypes.fp32)
        .repeat_interleave(query_length)
        .reshape(batch_size * query_length, 1, 1)
        .expand_as(output)
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), expected, rtol=0.005, atol=0.005)


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL pa_decode correctness + perf sweep",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="""Query/output data type.
        e.g.: -d bf16 fp16""",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=_positive_int,
        nargs="*",
        default=DEFAULT_BATCH_SIZES,
        help="""Batch sizes.
        e.g.: -b 1 3 16""",
    )
    parser.add_argument(
        "-q",
        "--query-length",
        type=_positive_int,
        nargs="+",
        default=[1],
        help="""Query tokens per sequence: 1 for decode, 4 for MTP4.
        e.g.: -q 1 4. MTP positions use dense causal masking.""",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=DEFAULT_SHAPES,
        help="""(num_query_heads,num_kv_heads,head_dim,context_length).
        Contexts are equal-length and include the MTP query tokens.
        e.g.: -s 8,1,128,257""",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        nargs="*",
        choices=[16, 64, 128],
        default=[16, 64, 128],
        help="""KV-cache block sizes.""",
    )
    parser.add_argument(
        "--trans-v",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[0, 1],
        help="""V-cache layouts: 0 is the plain 4-D cache, 1 the transposed 5-D
        cache production serves.""",
    )
    parser.add_argument(
        "--max-partitions",
        type=int,
        default=8,
        help="""Upper clamp passed to get_recommended_splits (4..256).
        Only used when --num-partitions is omitted.""",
    )
    parser.add_argument(
        "--num-partitions",
        type=_positive_int,
        nargs="+",
        default=[None],
        help="""Exact partition counts (1..256), bypassing auto selection.
        e.g.: --num-partitions 3 5. Overrides --max-partitions.""",
    )
    parser.add_argument(
        "--per-token",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[0],
        help="""KV scale layout: 0 for per-tensor, 1 for per-token.""",
    )
    args = parser.parse_args(argv)
    if not 4 <= args.max_partitions <= MAX_CONTEXT_PARTITIONS:
        parser.error(f"--max-partitions must be in [4, {MAX_CONTEXT_PARTITIONS}]")
    if any(
        count is not None and count > MAX_CONTEXT_PARTITIONS
        for count in args.num_partitions
    ):
        parser.error(f"--num-partitions must be in [1, {MAX_CONTEXT_PARTITIONS}]")
    for shape in args.shapes:
        if (
            not isinstance(shape, tuple)
            or len(shape) != 4
            or any(value < 1 for value in shape)
        ):
            parser.error("each --shapes value must contain four positive integers")
        if shape[0] % shape[1]:
            parser.error("num_query_heads must be divisible by num_kv_heads")
    return args


def main():
    # Parse first so --help and invalid options do not initialize the GPU.
    args = _parse_args()
    if not torch.cuda.is_available():
        aiter.logger.warning("ROCm is not available; skipping pa_decode")
        return
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning("pa_decode unsupported on %s; skipping", get_gfx_runtime())
        return
    if pa_decode is None:
        aiter.logger.warning("flydsl is unavailable; skipping pa_decode")
        return
    torch.set_default_device("cuda")

    rows = []
    for (
        dtype,
        batch_size,
        shape,
        block_size,
        trans_v,
        per_token,
        query_length,
        num_partitions,
    ) in itertools.product(
        args.dtype,
        args.batch,
        args.shapes,
        args.block_size,
        args.trans_v,
        args.per_token,
        args.query_length,
        args.num_partitions,
    ):
        num_query_heads, num_kv_heads, head_dim, context_length = shape
        rows.append(
            run_pa_decode_tile_case(
                batch_size,
                num_query_heads,
                num_kv_heads,
                head_dim,
                context_length,
                block_size,
                dtype,
                bool(trans_v),
                args.max_partitions,
                bool(per_token),
                query_length=query_length,
                num_partitions=num_partitions,
            )
        )

    df = pd.DataFrame(rows)
    aiter.logger.info("pa_decode summary (markdown):\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
