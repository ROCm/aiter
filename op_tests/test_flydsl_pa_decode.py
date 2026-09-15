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
# All parametrized cases use equal context lengths and the strict bound.
FIXED_LENGTH_ACCURACY_TOLERANCE = 5e-3

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


PA_DECODE_TEST_CASES = [
    pytest.param(
        {
            "batch_size": 3,
            "num_query_heads": 8,
            "num_kv_heads": 1,
            "head_dim": 128,
            "context_length": 257,
            "block_size": 16,
            "dtype": dtypes.bf16,
            "trans_v": False,
            "per_token": False,
            "query_length": 1,
            "num_partitions": None,
        },
        id="decode-auto-page16-bf16-scalar",
    ),
    pytest.param(
        {
            "batch_size": 3,
            "num_query_heads": 8,
            "num_kv_heads": 2,
            "head_dim": 128,
            "context_length": 1027,
            "block_size": 64,
            "dtype": dtypes.fp16,
            "trans_v": True,
            "per_token": False,
            "query_length": 1,
            "num_partitions": 1,
        },
        id="decode-np1-page64-fp16-trans-v",
    ),
    pytest.param(
        {
            "batch_size": 3,
            "num_query_heads": 8,
            "num_kv_heads": 2,
            "head_dim": 128,
            "context_length": 257,
            "block_size": 128,
            "dtype": dtypes.bf16,
            "trans_v": False,
            "per_token": True,
            "query_length": 1,
            "num_partitions": 4,
        },
        id="decode-reduce-page128-per-token",
    ),
    pytest.param(
        {
            "batch_size": 2,
            "num_query_heads": 16,
            "num_kv_heads": 1,
            "head_dim": 128,
            "context_length": 257,
            "block_size": 128,
            "dtype": dtypes.bf16,
            "trans_v": True,
            "per_token": True,
            "query_length": 2,
            "num_partitions": 3,
            "gfx950_query_splits": 2,
        },
        id="mtp2-query-split-page128-per-token",
    ),
    pytest.param(
        {
            "batch_size": 2,
            "num_query_heads": 16,
            "num_kv_heads": 1,
            "head_dim": 128,
            "context_length": 1027,
            "block_size": 128,
            "dtype": dtypes.bf16,
            "trans_v": False,
            "per_token": True,
            "query_length": 3,
            "num_partitions": 4,
            "gfx950_query_splits": 1,
        },
        id="mtp3-fused-page128-per-token",
    ),
    pytest.param(
        {
            "batch_size": 2,
            "num_query_heads": 16,
            "num_kv_heads": 1,
            "head_dim": 128,
            "context_length": 257,
            "block_size": 16,
            "dtype": dtypes.bf16,
            "trans_v": True,
            "per_token": True,
            "query_length": 4,
            "num_partitions": 3,
            "gfx950_query_splits": 4,
        },
        id="mtp4-query-split-page16-per-token",
    ),
    pytest.param(
        {
            "batch_size": 2,
            "num_query_heads": 16,
            "num_kv_heads": 1,
            "head_dim": 128,
            "context_length": 257,
            "block_size": 128,
            "dtype": dtypes.fp16,
            "trans_v": True,
            "per_token": False,
            "query_length": 4,
            "num_partitions": 3,
            "gfx950_query_splits": 1,
        },
        id="mtp4-reduce-page128-fp16-scalar",
    ),
    pytest.param(
        {
            "batch_size": 2,
            "num_query_heads": 8,
            "num_kv_heads": 1,
            "head_dim": 256,
            "context_length": 64,
            "block_size": 16,
            "dtype": dtypes.bf16,
            "trans_v": False,
            "per_token": False,
            "query_length": 1,
            "num_partitions": 1,
        },
        id="decode-head256-np1",
    ),
    pytest.param(
        {
            "batch_size": 1,
            "num_query_heads": 16,
            "num_kv_heads": 1,
            "head_dim": 128,
            "context_length": 200000,
            "block_size": 128,
            "dtype": dtypes.bf16,
            "trans_v": True,
            "per_token": True,
            "query_length": 1,
            "num_partitions": 8,
            "max_partitions": 4,
        },
        id="decode-long-context-200k",
    ),
    pytest.param(
        {
            "batch_size": 200,
            "num_query_heads": 16,
            "num_kv_heads": 1,
            "head_dim": 128,
            "context_length": 257,
            "block_size": 128,
            "dtype": dtypes.bf16,
            "trans_v": False,
            "per_token": True,
            "query_length": 4,
            "num_partitions": None,
            "gfx950_query_splits": 1,
        },
        id="mtp4-bs200-auto-smoke",
    ),
]


@pytest.mark.parametrize("case", PA_DECODE_TEST_CASES)
def test_pa_decode(case, monkeypatch):
    """Run one shared real-kernel/reference path across the supported PA axes."""
    _require_gpu()
    case = case.copy()
    expected_query_splits = case.pop("gfx950_query_splits", None)
    selected_query_splits = []

    if expected_query_splits is not None:
        module = importlib.import_module("aiter.ops.flydsl.pa_decode")
        compile_tile = module.compile_pa_decode_tile

        def capture_compile(**kwargs):
            selected_query_splits.append(kwargs["query_splits"])
            return compile_tile(**kwargs)

        monkeypatch.setattr(module, "compile_pa_decode_tile", capture_compile)

    def run_once(fn, *args, **kwargs):
        assert fn is _run_flydsl
        assert not kwargs
        return fn(*args), 1.0

    monkeypatch.setitem(globals(), "run_perftest", run_once)

    explicit_partitions = case["num_partitions"]
    max_partitions = case.get("max_partitions", 8)
    if explicit_partitions is None:
        expected_partitions = get_recommended_splits(
            case["batch_size"],
            case["num_kv_heads"],
            split_kv_blocks=KV_COMPUTE_BLOCK // case["block_size"],
            max_partitions=max_partitions,
        )
    else:
        expected_partitions = explicit_partitions

    result = run_pa_decode_tile_case(**case)
    assert result["partitions"] == expected_partitions
    assert result["flydsl err"] == 0
    assert result["flydsl us"] == 1.0
    assert result["flydsl us/token"] == pytest.approx(
        1.0 / (case["batch_size"] * case["query_length"])
    )
    assert result["flydsl TFLOPS"] > 0
    assert result["flydsl TB/s"] > 0

    if expected_query_splits is not None:
        expected = expected_query_splits if get_gfx_runtime() == "gfx950" else 1
        assert selected_query_splits
        assert set(selected_query_splits) == {expected}


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
        help="""Query tokens per sequence: 1 for decode, >1 for MTP.
        e.g.: -q 1 2 3 4. MTP positions use dense causal masking.""",
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
