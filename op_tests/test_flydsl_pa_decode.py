# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance sweep for FlyDSL paged-attention Tile."""

import argparse
import importlib
import itertools

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes, per_tensor_quant
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.attention import pa_decode_flydsl as public_pa_decode
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
        get_recommended_splits,
        pa_decode,
    )
except (ImportError, AttributeError, RuntimeError, OSError):
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


def test_pa_decode_maps_buffers_and_scale_layout(monkeypatch):
    _require_gpu()

    pa_decode_module = importlib.import_module("aiter.ops.flydsl.pa_decode")
    attention_module = importlib.import_module("aiter.ops.attention")
    captured = {}

    monkeypatch.setattr(
        pa_decode_module,
        "compile_pa_decode_tile",
        lambda **kwargs: {"launch": object()},
    )

    def capture_launch(
        launch,
        output,
        max_logits,
        exp_sums,
        temporary_output,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        key_scale,
        value_scale,
        *args,
    ):
        captured.update(
            max_logits=max_logits,
            exp_sums=exp_sums,
            temporary_output=temporary_output,
            key_scale=key_scale,
            value_scale=value_scale,
        )

    monkeypatch.setattr(pa_decode_module, "_run_compiled", capture_launch)
    monkeypatch.setattr(pa_decode_module, "ptr_arg", lambda tensor, dtype=None: tensor)

    def capture_reduce(
        output_5d, reduce_exp_sums, reduce_max_logits, logits, *args, **kwargs
    ):
        captured.update(
            reduce_output=output_5d,
            reduce_exp_sums=reduce_exp_sums,
            reduce_max_logits=reduce_max_logits,
            reduce_logits=logits,
            reduce_kwargs=kwargs,
        )

    monkeypatch.setattr(pa_decode_module, "launch_pa_decode_ps_reduce", capture_reduce)

    query = torch.empty(1, 8, 128, dtype=torch.bfloat16)
    output = torch.empty_like(query)
    key_cache = torch.empty(1, 1, 8, 16, 16, dtype=_quant_dtype())
    value_cache = torch.empty(1, 1, 128, 16, dtype=_quant_dtype())
    context_lengths = torch.tensor([16], dtype=torch.int32)
    block_tables = torch.tensor([[0]], dtype=torch.int32)
    key_scale = torch.ones(1, 1, 16, 1, dtype=torch.float32)
    value_scale = torch.ones_like(key_scale)
    max_logits = torch.empty(1, 1, 2, 8, dtype=torch.float32)
    exp_sums = torch.empty_like(max_logits)
    temporary_output = torch.empty(1, 1, 2, 8, 128, dtype=query.dtype)

    pa_decode(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        128**-0.5,
        1,
        2,
        compute_type=key_cache.dtype,
        key_scale=key_scale,
        value_scale=value_scale,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
    )

    assert captured["key_scale"].shape == (1, 1, 16)
    assert captured["value_scale"].shape == (1, 1, 16)
    assert captured["max_logits"].data_ptr() == max_logits.data_ptr()
    assert captured["exp_sums"].data_ptr() == exp_sums.data_ptr()
    assert captured["temporary_output"].data_ptr() == temporary_output.data_ptr()
    assert captured["reduce_output"].data_ptr() == output.data_ptr()
    assert captured["reduce_exp_sums"].data_ptr() == exp_sums.data_ptr()
    assert captured["reduce_max_logits"].data_ptr() == max_logits.data_ptr()
    assert captured["reduce_logits"].data_ptr() == temporary_output.data_ptr()
    assert captured["reduce_kwargs"]["context_partition_num"] == 2

    dispatches = []
    monkeypatch.setattr(
        attention_module,
        "_pa_decode_flydsl",
        lambda *args, **kwargs: dispatches.append("flydsl"),
    )
    public_pa_decode(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        128**-0.5,
        1,
        2,
        compute_type=key_cache.dtype,
        key_scale=key_scale,
        value_scale=value_scale,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
    )
    assert dispatches == ["flydsl"]


@pytest.mark.parametrize(
    "batch_size,expected",
    [(32, False), (33, True), (64, True), (65, False)],
)
def test_v_prefetch_workgroup_interval(batch_size, expected):
    """Warmup boundaries cover both compile-time prefetch variants."""
    if pa_decode is None:
        pytest.skip("FlyDSL is not available")
    pa_decode_module = importlib.import_module("aiter.ops.flydsl.pa_decode")
    assert (
        pa_decode_module._workgroup_count_enables_v_prefetch(
            batch_size,
            num_kv_heads=1,
            num_partitions=8,
            num_compute_units=256,
        )
        is expected
    )


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


@pytest.mark.parametrize("max_partitions", [0, 3, 257])
def test_recommended_splits_rejects_invalid_upper_clamp(max_partitions):
    if get_recommended_splits is None:
        pytest.skip("FlyDSL is not available")
    with pytest.raises(ValueError, match="max_partitions"):
        get_recommended_splits(1, 1, max_partitions=max_partitions)


def run_torch(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_query_heads, head_dim = query.shape
    block_size = key_cache.shape[2]
    num_kv_heads = key_cache.shape[1]
    query_group_size = num_query_heads // num_kv_heads
    softmax_scale = head_dim**-0.5
    output = torch.empty_like(query)

    for seq_idx in range(batch_size):
        context_length = int(context_lengths[seq_idx].item())
        token_ids = torch.arange(context_length, device=query.device)
        logical_pages = token_ids // block_size
        token_offsets = token_ids % block_size
        physical_pages = block_tables[seq_idx, logical_pages].long()

        keys = (
            key_cache[physical_pages, :, token_offsets, :].float() * key_scale.float()
        )
        values = (
            value_cache[physical_pages, :, :, token_offsets].float()
            * value_scale.float()
        )
        keys = keys.repeat_interleave(query_group_size, dim=1)
        values = values.repeat_interleave(query_group_size, dim=1)
        scores = (
            torch.einsum("hd,khd->hk", query[seq_idx].float(), keys) * softmax_scale
        )
        probs = torch.softmax(scores, dim=-1)
        output[seq_idx] = torch.einsum("hk,khd->hd", probs, values).to(query.dtype)
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
):
    if pa_decode is None or get_recommended_splits is None:
        raise RuntimeError("FlyDSL is not available")
    if dtype not in (dtypes.fp16, dtypes.bf16):
        raise ValueError(f"pa_decode only supports fp16/bf16, got {dtype}")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    torch.manual_seed(0)
    blocks_per_sequence = (context_length + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_sequence

    query = torch.empty(
        batch_size,
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
    key_quant, key_scale = per_tensor_quant(key, quant_dtype=quant_dtype)
    value_quant, value_scale = per_tensor_quant(value, quant_dtype=quant_dtype)
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
    )

    query_group_size = num_query_heads // num_kv_heads
    num_partitions = get_recommended_splits(
        batch_size,
        num_kv_heads,
        split_kv_blocks=KV_COMPUTE_BLOCK // block_size,
        max_partitions=max_partitions,
    )
    partial_shape = (
        batch_size,
        num_kv_heads,
        num_partitions,
        query_group_size,
    )
    pmax = torch.empty(partial_shape, dtype=dtypes.fp32)
    psum = torch.empty_like(pmax)
    pout = torch.empty(*partial_shape, head_dim, dtype=dtype)
    softmax_scale = head_dim**-0.5

    candidates = {"flydsl": _run_flydsl}

    # QK and PV each perform one multiply-add per query-head/context pair.
    flops = 4 * batch_size * num_query_heads * context_length * head_dim
    # Effective bandwidth counts Q + O + referenced K/V tokens and metadata once.
    # It excludes padded tokens, repeated loads, and partition scratch traffic.
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
        + key_scale.numel() * key_scale.element_size()
        + value_scale.numel() * value_scale.element_size()
    )

    ret = {
        "gfx": get_gfx_runtime(),
        "partitions": num_partitions,
        "trans_v": trans_v,
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
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


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


def _valid_pa_decode_arguments(head_dim=128, num_partitions=1):
    """Small valid call used by host-contract tests that fail before compile."""
    quant_dtype = _quant_dtype()
    query = torch.empty(1, 8, head_dim, dtype=dtypes.bf16, device="cuda")
    scale = torch.ones(1, dtype=dtypes.fp32, device=query.device)
    return {
        "output": torch.empty_like(query),
        "query": query,
        "key_cache": torch.empty(
            1, 1, head_dim // 16, 16, 16, dtype=quant_dtype, device=query.device
        ),
        "value_cache": torch.empty(
            1, 1, head_dim, 16, dtype=quant_dtype, device=query.device
        ),
        "context_lengths": torch.tensor([16], dtype=dtypes.i32, device=query.device),
        "block_tables": torch.tensor([[0]], dtype=dtypes.i32, device=query.device),
        "softmax_scale": head_dim**-0.5,
        "query_length": 1,
        "max_context_partition_num": num_partitions,
        "compute_type": quant_dtype,
        "key_scale": scale,
        "value_scale": scale,
    }


def test_ps_false_is_rejected():
    _require_gpu()
    with pytest.raises(NotImplementedError, match="ps=False"):
        pa_decode(**_valid_pa_decode_arguments(), ps=False)


@pytest.mark.parametrize(
    "invalid_input,error_match",
    [
        pytest.param("query_rank", "query", id="query-rank"),
        pytest.param("output_rank", "output", id="output-rank"),
        pytest.param("key_cache_rank", "key_cache", id="key-cache-rank"),
        pytest.param("value_cache_rank", "value_cache", id="value-cache-rank"),
        pytest.param("context_lengths_rank", "context_lengths", id="context-rank"),
        pytest.param("block_tables_rank", "block_tables", id="block-table-rank"),
        pytest.param("query_device", "CUDA device", id="query-device"),
        pytest.param("zero_query_heads", "at least one head", id="zero-query-heads"),
        pytest.param("block_table_rows", "block_tables", id="block-table-rows"),
        pytest.param("kv_block_count", "block", id="kv-block-count"),
        pytest.param("kv_head_count", "head", id="kv-head-count"),
    ],
)
def test_invalid_tensor_structure_is_rejected(invalid_input, error_match):
    _require_gpu()
    arguments = _valid_pa_decode_arguments()
    if invalid_input == "query_rank":
        arguments["query"] = arguments["query"].squeeze(0)
    elif invalid_input == "output_rank":
        arguments["output"] = arguments["output"].squeeze(0)
    elif invalid_input == "key_cache_rank":
        arguments["key_cache"] = arguments["key_cache"].squeeze(0)
    elif invalid_input == "value_cache_rank":
        arguments["value_cache"] = arguments["value_cache"].squeeze(0)
    elif invalid_input == "context_lengths_rank":
        arguments["context_lengths"] = arguments["context_lengths"].reshape(1, 1)
    elif invalid_input == "block_tables_rank":
        arguments["block_tables"] = arguments["block_tables"].flatten()
    elif invalid_input == "query_device":
        arguments["query"] = arguments["query"].cpu()
    elif invalid_input == "zero_query_heads":
        arguments["query"] = arguments["query"][:, :0]
        arguments["output"] = arguments["output"][:, :0]
    elif invalid_input == "block_table_rows":
        arguments["block_tables"] = arguments["block_tables"].repeat(2, 1)
    elif invalid_input == "kv_block_count":
        arguments["value_cache"] = arguments["value_cache"].repeat(2, 1, 1, 1)
    elif invalid_input == "kv_head_count":
        arguments["value_cache"] = arguments["value_cache"].repeat(1, 2, 1, 1)
    else:
        raise AssertionError(f"unknown invalid input: {invalid_input}")

    with pytest.raises(ValueError, match=error_match):
        pa_decode(**arguments)


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


@pytest.mark.parametrize("num_partitions", [1, 4])
@pytest.mark.parametrize("head_dim", [64, 128, 256, 1024])
def test_supported_head_dim_boundaries_are_accepted(
    monkeypatch, head_dim, num_partitions
):
    """The public support check is identical before direct and reduced paths."""
    _require_gpu()
    pa_decode_module = importlib.import_module("aiter.ops.flydsl.pa_decode")
    compile_arguments = []

    def capture_compile(**kwargs):
        compile_arguments.append(kwargs)
        return {"launch": object()}

    monkeypatch.setattr(pa_decode_module, "compile_pa_decode_tile", capture_compile)
    monkeypatch.setattr(pa_decode_module, "_run_compiled", lambda *args: None)
    monkeypatch.setattr(pa_decode_module, "ptr_arg", lambda tensor, dtype=None: tensor)
    monkeypatch.setattr(
        pa_decode_module, "launch_pa_decode_ps_reduce", lambda *args, **kwargs: None
    )

    pa_decode(**_valid_pa_decode_arguments(head_dim, num_partitions))
    assert len(compile_arguments) == 1
    assert compile_arguments[0]["head_dim"] == head_dim
    assert compile_arguments[0]["num_partitions"] == num_partitions


@pytest.mark.parametrize("num_partitions", [1, 4])
@pytest.mark.parametrize("head_dim", [96, 192, 1152])
def test_unsupported_head_dim_is_rejected(head_dim, num_partitions):
    """NP=1 and NP>1 expose the same public head-dimension support domain."""
    _require_gpu()
    with pytest.raises(NotImplementedError, match="head_dim"):
        pa_decode(**_valid_pa_decode_arguments(head_dim, num_partitions))


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


def main():
    torch.set_default_device("cuda")
    if not torch.cuda.is_available():
        aiter.logger.warning("ROCm is not available; skipping pa_decode")
        return
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning("pa_decode unsupported on %s; skipping", get_gfx_runtime())
        return
    if pa_decode is None:
        aiter.logger.warning("flydsl is unavailable; skipping pa_decode")
        return

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
        type=int,
        nargs="*",
        default=DEFAULT_BATCH_SIZES,
        help="""Batch sizes.
        e.g.: -b 1 3 16""",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=DEFAULT_SHAPES,
        help="""(num_query_heads,num_kv_heads,head_dim,context_length).
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
        help="""Upper clamp passed to get_recommended_splits (4..256).""",
    )
    args = parser.parse_args()

    rows = []
    for dtype, batch_size, shape, block_size, trans_v in itertools.product(
        args.dtype, args.batch, args.shapes, args.block_size, args.trans_v
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
            )
        )

    df = pd.DataFrame(rows)
    aiter.logger.info("pa_decode summary (markdown):\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
