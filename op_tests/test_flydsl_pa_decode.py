# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Generic FlyDSL paged decode correctness test.

Run with: python -m pytest -q op_tests/test_flydsl_pa_decode.py
"""

import importlib.util
from itertools import product

import pytest
import torch

from aiter import per_tensor_quant, pertoken_quant
from aiter.jit.utils.chip_info import get_gfx_runtime


@pytest.fixture
def device(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if get_gfx_runtime() not in ("gfx942", "gfx950", "gfx1250"):
        pytest.skip(f"pa_decode is unsupported on {get_gfx_runtime()}")
    if importlib.util.find_spec("flydsl") is None:
        pytest.skip("FlyDSL is not installed")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    return torch.device("cuda")


@pytest.fixture
def gfx1250_only(device):
    """Keep gfx1250 BF16 regressions gated within the shared PA suite."""
    if get_gfx_runtime() != "gfx1250":
        pytest.skip("requires gfx1250")


@pytest.mark.parametrize(
    "batch,max_parts,budget,window,query_length",
    [
        (1, 1, 1, 0, 1),
        (6, 7, 160, 257, 4),
        (63, 7, 160, 0, 1),
        (65, 7, 512, 1, 4),
        (257, None, 4096, 8193, 4),
        (4096, None, 8192, 0, 1),
    ],
)
def test_pa_decode_plan(device, batch, max_parts, budget, window, query_length):
    """Packed tasks cover every visible tile and remain valid after graph refresh."""
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
    context = torch.tensor(lengths, dtype=torch.int32, device=device)
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
        context.copy_(torch.tensor(values, dtype=torch.int32, device=device))
        graph.replay()
        check(plan, values)
        assert (plan.work_info.data_ptr(), plan.reduce_info.data_ptr()) == pointers


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
    lengths=None,
):
    if lengths is None:
        lengths = (0, 1, 255, 256, 257, 1027) if pattern == "random" else (1024,)
    batch = len(lengths)
    pages_per_seq = (max(lengths) + page_size - 1) // page_size
    num_pages = batch * pages_per_seq
    table = torch.randperm(num_pages, device=device, dtype=torch.int32).reshape(
        batch, pages_per_seq
    )
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

    context = torch.tensor(lengths, dtype=torch.int32, device=device)
    return query, key, value, table, context


def _make_bf16_inputs(
    device,
    length,
    *,
    query_length=1,
    group=1,
    dim=128,
    trans_v=True,
    strides=False,
    page_ids=None,
    pool_pages=None,
    page=128,
    divisor=8,
    cache_dtype=torch.float8_e4m3fn,
):
    pages = (length + page - 1) // page
    if page_ids is None:
        page_ids = torch.randperm(pages, device=device, dtype=torch.int32)
    table = page_ids.reshape(1, pages)
    pool_pages = pages if pool_pages is None else pool_pages
    # Representable values isolate lane/operand mapping from FP8 cache rounding.
    torch.manual_seed(17)
    key = torch.randint(-4, 5, (pages, 1, page, dim), device=device).float() / divisor
    value = torch.randint(-4, 5, key.shape, device=device).float() / divisor
    qshape = (query_length, group, dim)
    query = (torch.randint(-4, 5, qshape, device=device).float() / divisor).bfloat16()
    if strides:
        query_view = torch.empty_strided(
            qshape,
            (group * (dim + 1) + 3, dim + 1, 1),
            dtype=query.dtype,
            device=device,
        )
        query_view.copy_(query)
        query = query_view
    chunk = 16 // torch.empty((), dtype=cache_dtype).element_size()
    cache_key = torch.empty(
        (pool_pages, 1, dim // chunk, page, chunk), dtype=cache_dtype, device=device
    )
    vshape = (
        (pool_pages, 1, page // chunk, dim, chunk)
        if trans_v
        else (pool_pages, 1, dim, page)
    )
    cache_value = torch.empty(vshape, dtype=cache_dtype, device=device)
    packed_key = (
        key.reshape(pages, 1, page, dim // chunk, chunk)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .to(cache_key.dtype)
    )
    packed_value = (
        (
            value.reshape(pages, 1, page // chunk, chunk, dim).permute(0, 1, 2, 4, 3)
            if trans_v
            else value.permute(0, 1, 3, 2)
        )
        .contiguous()
        .to(cache_value.dtype)
    )
    # FP8 index_copy is not implemented by all bring-up PyTorch builds.
    for logical, physical in enumerate(page_ids.cpu().tolist()):
        cache_key[physical].copy_(packed_key[logical])
        cache_value[physical].copy_(packed_value[logical])
    if pool_pages == pages:
        reference_key, reference_value = torch.empty_like(key), torch.empty_like(value)
        reference_key[page_ids.long()] = key
        reference_value[page_ids.long()] = value
        key, value = reference_key, reference_value
    lengths = torch.tensor([length], dtype=torch.int32, device=device)
    return query, key, value, cache_key, cache_value, table, lengths


def _pack_bf16_cache(key, value, trans_v):
    pages, heads, page, dim = key.shape
    kc = (
        key.reshape(pages, heads, page, dim // 8, 8).permute(0, 1, 3, 2, 4).contiguous()
    )
    vc = (
        value.reshape(pages, heads, page // 8, 8, dim).permute(0, 1, 2, 4, 3)
        if trans_v
        else value.permute(0, 1, 3, 2)
    ).contiguous()
    return kc, vc


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
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), reference, atol=5e-3, rtol=5e-3)


def _check_unsupported_input(pa, arguments, options, plan, case, monkeypatch):
    query, key, value, *rest = arguments
    invalid = case["invalid_input"]
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
        pytest.fail("invalid BF16 inputs reached kernel compilation")

    monkeypatch.setattr(pa, "compile_pa_decode_tile", unexpected_compile)
    with pytest.raises(case["error"], match=case["match"]):
        pa.pa_decode(
            torch.empty_like(query),
            query,
            key,
            value,
            *rest,
            work_plan=plan,
            **options,
        )


@pytest.mark.parametrize(
    "case",
    [
        *[
            pytest.param(
                dict(
                    geometry,
                    kv_dtype="fp8",
                    query_dtype=dtype,
                    scale_mode=scale_mode,
                    trans_v=trans_v,
                ),
                id=f"fp8-{dtype}-{scale_mode}-{'transposed' if trans_v else 'plain'}-v-{name}",
            )
            for name, geometry in _FP8_CASES
            for dtype, scale_mode, trans_v in product(
                (torch.bfloat16, torch.float16),
                ("per-tensor", "per-token"),
                (False, True),
            )
        ],
        *[pytest.param(geometry, id=f"bf16-{name}") for name, geometry in _BF16_CASES],
        *[
            pytest.param(
                {
                    "invalid_input": invalid,
                    "error": error,
                    "match": match,
                    "lengths": (1,),
                },
                id=f"invalid-bf16-{invalid}",
            )
            for invalid, error, match in _INVALID_CASES
        ],
    ],
)
def test_pa_decode(case, device, monkeypatch):
    """FP8/BF16 decode shares input, reference, planning and replay checks."""
    case = dict(
        {
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
        },
        **case,
    )
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    if get_gfx_runtime() == "gfx1250":
        if case.get("invalid_input") in ("fp16-query", "plain-v"):
            pytest.skip("gfx1250 supports FP16 queries and plain BF16 V caches")
        if (
            case["kv_dtype"] == "bf16"
            and case["head_dim"] % 128
            and case["head_dim"] != 64
        ):
            pytest.skip("gfx1250 supports head_dim 64 or multiples of 128")
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
    query, key, value, table, context = _make_inputs(
        query_length,
        kv_heads,
        group_size,
        dim,
        page,
        case["query_dtype"],
        case["pattern"],
        device,
        case["lengths"],
    )
    compute_type = torch.bfloat16
    key_scale = value_scale = None
    vector = 8
    if case["kv_dtype"] == "fp8":
        compute_type = (
            torch.float8_e4m3fn
            if get_gfx_runtime() in ("gfx950", "gfx1250")
            else torch.float8_e4m3fnuz
        )
        vector = 16
        mode = case["scale_mode"]
        if mode in ("per-tensor", "per-token"):
            quantize = pertoken_quant if mode == "per-token" else per_tensor_quant
            key, key_scale = quantize(key, quant_dtype=compute_type)
            value, value_scale = quantize(value, quant_dtype=compute_type)
        else:
            key, value = key.to(compute_type), value.to(compute_type)
            if mode == "python":
                key_scale, value_scale = 0.25, 2.0
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
            -2, 2, query.shape[1], dtype=case["sink_dtype"], device=device
        )
        sinks[0], sinks[-1] = float("-inf"), float("inf")
    options = {
        "compute_type": compute_type,
        "key_scale": key_scale,
        "value_scale": value_scale,
        "sinks": sinks,
        "sliding_window": window,
        "max_context_length": context.max().item(),
    }
    arguments = (query, key_cache, value_cache, context, table, dim**-0.5, query_length)
    plan = pa.plan_pa_decode(
        context,
        kv_heads,
        max_partitions=case["max_partitions"],
        query_length=query_length,
        sliding_window=window,
    )
    if "invalid_input" in case:
        _check_unsupported_input(pa, arguments, options, plan, case, monkeypatch)
        return

    output = torch.full_like(query, float("nan"))
    workspace = {}
    if case["capture"]:
        scalar_shape = (kv_heads, plan.capacity, query_length * group_size)
        workspace = {
            "exp_sums": torch.empty(scalar_shape, dtype=torch.float32, device=device),
            "max_logits": torch.empty(scalar_shape, dtype=torch.float32, device=device),
            "temporary_output": torch.empty(
                (*scalar_shape, dim), dtype=query.dtype, device=device
            ),
        }

    def launch():
        pa.pa_decode(output, *arguments, work_plan=plan, **options, **workspace)

    def check():
        reference = _reference(
            query,
            logical_key,
            logical_value,
            table,
            context,
            query_length,
            window,
            sinks,
        )
        _check_output(output, reference)

    launch()
    check()
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "query_length,head_dim",
    [(1, 128), (1, 256), (2, 128), (2, 256), (4, 256)],
    ids=["decode-d128", "decode-d256", "mtp2-d128", "mtp2-d256", "mtp4-d256"],
)
@pytest.mark.parametrize("max_partitions", [1, 7], ids=["direct", "partitioned"])
def test_pa_decode_packed_page_boundaries(
    dtype, query_length, head_dim, max_partitions, device
):
    """Page-sized FP8 tiles must zero-fill tails and survive plan refresh."""
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("requires gfx1250 packed WMMA tail masking")
    torch.manual_seed(13)
    lengths = (
        0,
        1,
        15,
        16,
        17,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
        511,
        512,
        513,
        1027,
    )
    heads, group, dim, page = 2, 8, head_dim, 128
    query, key, value, table, context = _make_inputs(
        query_length, heads, group, dim, page, dtype, "random", device, lengths
    )
    # Keep short-context Q/P rounding within the suite's absolute tolerance,
    # so this test isolates masking, zero-fill, and refreshed scratch reads.
    value.mul_(0.5)
    fp8 = (
        torch.float8_e4m3fnuz if get_gfx_runtime() == "gfx942" else torch.float8_e4m3fn
    )
    key, ks = pertoken_quant(key, quant_dtype=fp8)
    value, vs = pertoken_quant(value, quant_dtype=fp8)
    references = []
    for active in (lengths, tuple(min(length, 129) for length in lengths)):
        reference_lengths = torch.tensor(active, dtype=torch.int32, device=device)
        references.append(
            _reference(
                query,
                key.float() * ks,
                value.float() * vs,
                table,
                reference_lengths,
                query_length,
                0,
                None,
            )
        )
    # Poison bytes/scales outside every original sequence's valid context.
    # Their pages are owned by that sequence; no live token is overwritten.
    poison = torch.tensor(float("nan"), dtype=fp8, device=device).view(torch.uint8)
    for seq, length in enumerate(lengths):
        if length and length % page:
            physical = table[seq, length // page].item()
            key.view(torch.uint8)[physical, :, length % page :] = poison
            value.view(torch.uint8)[physical, :, length % page :] = poison
            ks[physical, :, length % page :] = float("nan")
            vs[physical, :, length % page :] = float("nan")
    pages = key.shape[0]
    kc = (
        key.reshape(pages, heads, page, dim // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    vc = (
        value.reshape(pages, heads, page // 16, 16, dim)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )
    plan = plan_pa_decode(
        context,
        heads,
        max_partitions=max_partitions,
        workgroup_budget=128,
        query_length=query_length,
    )
    output = torch.full_like(query, float("nan"))
    shape = (heads, plan.capacity, query_length * group)
    sums = torch.full(shape, float("nan"), dtype=torch.float32, device=device)
    maxima = torch.full_like(sums, float("nan"))
    partial = torch.full((*shape, dim), float("nan"), dtype=dtype, device=device)

    def launch():
        pa_decode(
            output,
            query,
            kc,
            vc,
            context,
            table,
            dim**-0.5,
            query_length,
            key_scale=ks,
            value_scale=vs,
            work_plan=plan,
            max_context_length=max(lengths),
            exp_sums=sums,
            max_logits=maxima,
            temporary_output=partial,
        )

    launch()
    torch.testing.assert_close(output.float(), references[0], atol=5e-3, rtol=5e-3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    context.copy_(
        torch.tensor(
            [min(length, 129) for length in lengths], dtype=torch.int32, device=device
        )
    )
    plan_pa_decode(context, heads, plan=plan, query_length=query_length)
    sums.fill_(float("nan"))
    maxima.fill_(float("nan"))
    partial.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), references[1], atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize(
    "query_dtype", [torch.bfloat16, torch.float16], ids=["bf16-q", "fp16-q"]
)
@pytest.mark.parametrize("trans_v", [False, True], ids=["plain-v", "transposed-v"])
@pytest.mark.parametrize(
    "ql,heads,group,dim,page,parts,window,pattern",
    [
        (1, 1, 8, 128, 16, 1, 0, "random"),
        (3, 2, 4, 64, 64, 3, 0, "random"),
        (4, 1, 16, 128, 128, 7, 257, "random"),
        (2, 2, 8, 256, 16, 7, 0, "random"),
        (1, 1, 4, 384, 64, 1, 0, "random"),
        (2, 1, 8, 512, 64, 3, 0, "random"),
        (1, 1, 4, 768, 128, 1, 0, "random"),
        (1, 1, 4, 1024, 128, 1, 0, "random"),
        (1, 1, 8, 128, 16, 1, 0, "small_query"),
        (1, 1, 16, 128, 128, 1, 0, "probability_tail"),
    ],
)
@pytest.mark.usefixtures("gfx1250_only")
def test_bf16_cache(
    ql, heads, group, dim, page, parts, window, pattern, query_dtype, trans_v, device
):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    q, k, v, table, lengths = _make_inputs(
        ql, heads, group, dim, page, torch.bfloat16, pattern, device
    )
    q = q.to(query_dtype)
    kc, vc = _pack_bf16_cache(k, v, trans_v)
    sinks = torch.linspace(-2, 2, q.shape[1], dtype=torch.float32, device=device)
    sinks[0], sinks[-1] = float("-inf"), float("inf")
    plan = plan_pa_decode(
        lengths, heads, max_partitions=parts, query_length=ql, sliding_window=window
    )
    out = torch.full_like(q, float("nan"))
    # Default compute_type is inferred from the cache dtype; no scale buffers
    # are allocated or read for native BF16.
    pa_decode(
        out,
        q,
        kc,
        vc,
        lengths,
        table,
        dim**-0.5,
        ql,
        sinks=sinks,
        sliding_window=window,
        work_plan=plan,
    )
    expected = _reference(q, k.float(), v.float(), table, lengths, ql, window, sinks)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), expected, atol=0.002, rtol=0.01)


@pytest.mark.parametrize("trans_v", [False, True])
@pytest.mark.parametrize("max_partitions", [1, 8])
@pytest.mark.usefixtures("gfx1250_only")
def test_bf16_graph_refresh(trans_v, max_partitions, device):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    q, k, v, kc, vc, table, lengths = _make_bf16_inputs(
        device,
        1027,
        query_length=4,
        group=8,
        trans_v=trans_v,
        strides=True,
        cache_dtype=torch.bfloat16,
    )
    plan = plan_pa_decode(
        lengths, 1, max_partitions=max_partitions, sliding_window=257, query_length=4
    )
    rows = 32
    sums = torch.empty((1, plan.capacity, rows), device=device)
    maxima = torch.empty_like(sums)
    partial = torch.empty((*sums.shape, 128), dtype=q.dtype, device=device)
    backing = torch.empty(4 * (8 * 129 + 3) + 1, dtype=q.dtype, device=device)
    out = backing[1:].as_strided(q.shape, (8 * 129 + 3, 129, 1))

    def launch():
        pa_decode(
            out,
            q,
            kc,
            vc,
            lengths,
            table,
            128**-0.5,
            4,
            compute_type=torch.bfloat16,
            work_plan=plan,
            sliding_window=257,
            exp_sums=sums,
            max_logits=maxima,
            temporary_output=partial,
        )

    launch()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    for length in (1027, 1024, 1023, 257, 256, 255, 1, 0, 1027):
        lengths.fill_(length)
        plan_pa_decode(lengths, 1, plan=plan, sliding_window=257, query_length=4)
        sums.fill_(float("nan"))
        maxima.fill_(float("nan"))
        partial.fill_(float("nan"))
        graph.replay()
        expected = _reference(q, k, v, table, lengths, 4, 257, None)
        torch.testing.assert_close(out.float(), expected, atol=0.002, rtol=0.01)


@pytest.mark.parametrize("boundary", [2**31, 2**32], ids=["2GiB", "4GiB"])
@pytest.mark.parametrize("trans_v", [False, True])
@pytest.mark.usefixtures("gfx1250_only")
def test_bf16_wide_addresses(boundary, trans_v, device):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    page_bytes = 128 * 128 * 2
    high_page = boundary // page_bytes
    pool_pages = high_page + 2
    if torch.cuda.mem_get_info(device)[0] < 2 * pool_pages * page_bytes + 2**30:
        pytest.skip("insufficient memory for wide-address test")
    ids = torch.tensor([high_page + 1, high_page], dtype=torch.int32, device=device)
    q, k, v, kc, vc, table, lengths = _make_bf16_inputs(
        device,
        255,
        cache_dtype=torch.bfloat16,
        trans_v=trans_v,
        page_ids=ids,
        pool_pages=pool_pages,
    )
    plan = plan_pa_decode(lengths, 1, max_partitions=1)
    out = torch.empty_like(q)
    pa_decode(out, q, kc, vc, lengths, table, 128**-0.5, 1, work_plan=plan)
    expected = _reference(
        q,
        k,
        v,
        torch.arange(2, device=device, dtype=torch.int32).reshape(1, 2),
        lengths,
        1,
        0,
        None,
    )
    torch.testing.assert_close(out.float(), expected, atol=0.002, rtol=0.01)


@pytest.mark.usefixtures("gfx1250_only")
def test_bf16_reject_scales(device):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    q, _k, _v, kc, vc, table, lengths = _make_bf16_inputs(
        device, 128, cache_dtype=torch.bfloat16
    )
    plan = plan_pa_decode(lengths, 1, max_partitions=1)
    with pytest.raises(ValueError, match="BF16 KV is unscaled"):
        pa_decode(
            torch.empty_like(q),
            q,
            kc,
            vc,
            lengths,
            table,
            128**-0.5,
            1,
            key_scale=torch.ones(1, device=device),
            work_plan=plan,
        )


@pytest.mark.parametrize("trans_v", [False, True])
@pytest.mark.usefixtures("gfx1250_only")
def test_bf16_large_mtp(trans_v, device):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    # BF16 D1024/MTP4/G16 needs one K/V buffer and reuses one Q tile.
    q, k, v, kc, vc, table, lengths = _make_bf16_inputs(
        device,
        257,
        query_length=4,
        group=16,
        dim=1024,
        trans_v=trans_v,
        cache_dtype=torch.bfloat16,
    )
    plan = plan_pa_decode(lengths, 1, max_partitions=1, query_length=4)
    out = torch.empty_like(q)
    pa_decode(out, q, kc, vc, lengths, table, 1024**-0.5, 4, work_plan=plan)
    expected = _reference(q, k, v, table, lengths, 4, 0, None)
    torch.testing.assert_close(out.float(), expected, atol=0.002, rtol=0.01)


@pytest.mark.parametrize("parts", [1, 7])
@pytest.mark.usefixtures("gfx1250_only")
def test_bf16_split_mtp_graph_refresh(parts, device):
    """Split queries preserve causal rows, packed slots, and empty output on replay."""
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    q, k, v, table, lengths = _make_inputs(
        4,
        2,
        16,
        1024,
        128,
        torch.bfloat16,
        "random",
        device,
        lengths=(0, 1, 257),
    )
    kc, vc = _pack_bf16_cache(k, v, True)
    plan = plan_pa_decode(lengths, 2, max_partitions=parts, query_length=4)
    # Matching odd query/output strides exercise the direct vector epilogue.
    strides = (32 * 1025 + 3, 1025, 1)
    query = torch.empty_strided(q.shape, strides, dtype=q.dtype, device=device)
    query.copy_(q)
    output = torch.empty_strided(q.shape, strides, dtype=q.dtype, device=device)
    shape = (2, plan.capacity, 64)
    sums = torch.empty(shape, device=device)
    maxima = torch.empty_like(sums)
    partial = torch.empty((*shape, 1024), dtype=q.dtype, device=device)

    def launch():
        pa_decode(
            output,
            query,
            kc,
            vc,
            lengths,
            table,
            1024**-0.5,
            4,
            work_plan=plan,
            exp_sums=sums,
            max_logits=maxima,
            temporary_output=partial,
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for active in ((0, 1, 257), (257, 0, 129), (257, 129, 0), (0, 0, 0), (1, 257, 256)):
        lengths.copy_(torch.tensor(active, device=device, dtype=torch.int32))
        plan_pa_decode(lengths, 2, plan=plan, query_length=4)
        output.fill_(float("nan"))
        sums.fill_(float("nan"))
        maxima.fill_(float("nan"))
        partial.fill_(float("nan"))
        graph.replay()
        expected = _reference(q, k.float(), v.float(), table, lengths, 4, 0, None)
        torch.testing.assert_close(output.float(), expected, atol=0.002, rtol=0.01)
