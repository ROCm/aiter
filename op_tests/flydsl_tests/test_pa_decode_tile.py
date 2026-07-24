# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance sweep for FlyDSL paged-attention Tile."""

import argparse
import importlib
import inspect
import itertools

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes, per_tensor_quant
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.attention import pa_decode_gluon as public_pa_decode
from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
KV_COMPUTE_BLOCK = 256

# Pairwise coverage of the normal-accuracy axes in FlyDSL's PA regression test:
# batches {3, 81, 128}, Q/KV heads {(4,1), (8,1), (16,1)}, head dims
# {128, 256}, and contexts {1027, 8192}. Keep the original 257-token boundary
# case as well. Both supported block sizes are crossed with every case in main().
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


def test_pa_decode_api_matches_gluon():
    if pa_decode is None:
        pytest.skip("FlyDSL is not available")

    flydsl_parameters = inspect.signature(pa_decode).parameters
    gluon_parameters = inspect.signature(pa_decode_gluon).parameters
    public_parameters = inspect.signature(public_pa_decode).parameters

    assert tuple(flydsl_parameters) == tuple(gluon_parameters)
    assert tuple(public_parameters) == tuple(gluon_parameters)
    for name, parameter in flydsl_parameters.items():
        gluon_parameter = gluon_parameters[name]
        assert parameter.kind == gluon_parameter.kind
        assert parameter.default == gluon_parameter.default
        assert public_parameters[name].kind == gluon_parameter.kind
        assert public_parameters[name].default == gluon_parameter.default


def test_pa_decode_maps_gluon_buffers_and_scale_layout(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if pa_decode is None:
        pytest.skip("FlyDSL is not available")

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
    pa_ps_module = importlib.import_module("csrc.cpp_itfs.pa.pa_ps")
    monkeypatch.setattr(
        pa_ps_module,
        "launch_pa_decode_ps_reduce",
        lambda *args, **kwargs: None,
    )

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

    assert attention_module._can_use_flydsl_pa_decode(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        1,
        2,
        256,
        key_cache.dtype,
        None,
        key_scale,
        value_scale,
        exp_sums,
        max_logits,
        temporary_output,
        None,
        None,
        0,
    )
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

    dispatches = []
    monkeypatch.setattr(
        attention_module,
        "_pa_decode_flydsl",
        lambda *args, **kwargs: dispatches.append("flydsl"),
    )
    monkeypatch.setattr(
        attention_module,
        "_pa_decode_gluon_fallback",
        lambda *args, **kwargs: dispatches.append("gluon"),
    )
    common_kwargs = {
        "key_scale": key_scale,
        "value_scale": value_scale,
        "exp_sums": exp_sums,
        "max_logits": max_logits,
        "temporary_output": temporary_output,
    }
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
        **common_kwargs,
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
        query_scale=torch.ones(1, dtype=torch.float32),
        **common_kwargs,
    )
    assert dispatches == ["flydsl", "gluon"]


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
    torch.ops.aiter.pa_decode_gluon(
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

    candidates = {
        "flydsl": lambda: _run_flydsl(
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
    }

    # QK and PV each perform one multiply-add per query-head/context pair.
    flops = 4 * batch_size * num_query_heads * context_length * head_dim
    # Logical tensor traffic: Q + O + the referenced K/V tokens and metadata.
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

    ret = {"gfx": get_gfx_runtime(), "partitions": num_partitions}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            reference.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=3e-2,
            atol=3e-2,
            tol_err_ratio=0.0,
            msg=f"{name}: pa_decode",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@pytest.mark.parametrize("block_size", [16, 64])
def test_pa_decode_tile(block_size):
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if pa_decode is None:
        pytest.skip("FlyDSL is not available")
    if get_gfx_runtime() not in SUPPORTED_GFX:
        pytest.skip(f"pa_decode is unsupported on {get_gfx_runtime()}")

    result = run_pa_decode_tile_case(
        batch_size=3,
        num_query_heads=8,
        num_kv_heads=1,
        head_dim=128,
        context_length=257,
        block_size=block_size,
        dtype=dtypes.bf16,
    )
    assert result["flydsl err"] == 0


def main():
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
        choices=[16, 64],
        default=[16, 64],
        help="""KV-cache block sizes.""",
    )
    args = parser.parse_args()

    rows = []
    for dtype, batch_size, shape, block_size in itertools.product(
        args.dtype, args.batch, args.shapes, args.block_size
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
            )
        )

    df = pd.DataFrame(rows)
    aiter.logger.info("pa_decode summary (markdown):\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
