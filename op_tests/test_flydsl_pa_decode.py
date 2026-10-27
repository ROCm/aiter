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


@pytest.mark.parametrize(
    "arch,num_seqs,window,max_parts,capacity_per_seq,expected_splits",
    [
        ("gfx950", 1, 1024, 8, 8, 8),
        ("gfx950", 8, 1024, 8, 8, 8),
        ("gfx950", 16, 1024, 8, 8, 4),
        ("gfx950", 32, 1024, 8, 8, 4),
        ("gfx950", 64, 1024, 8, 8, 2),
        ("gfx950", 128, 1024, 8, 8, 1),
        ("gfx942", 8, 1024, 8, 8, 1),
        ("gfx950", 8, 1023, 8, 8, 8),
        ("gfx950", 8, 1025, 8, 8, 8),
        ("gfx950", 1, 512, 8, 8, 8),
        ("gfx950", 8, 512, 8, 8, 8),
        ("gfx950", 32, 512, 8, 8, 4),
        ("gfx950", 128, 512, 8, 8, 1),
        ("gfx942", 8, 512, 8, 8, 1),
        # Limited budgets still need enough split-query CTAs to fill the device.
        ("gfx950", 8, 512, 1, 1, 8),
        ("gfx950", 8, 512, 4, 4, 8),
        ("gfx950", 8, 512, 8, 3, 8),
        ("gfx950", 32, 2048, 8, 8, 1),
        # Neighboring windows follow planned work, not an exact window value.
        ("gfx950", 8, 1024, 1, 1, 8),
        ("gfx950", 8, 1024, 4, 4, 8),
        ("gfx950", 8, 1024, 8, 2, 8),
        ("gfx950", 8, 1023, 1, 1, 8),
        ("gfx950", 8, 1023, 4, 4, 8),
        ("gfx950", 8, 1025, 1, 1, 8),
        ("gfx950", 8, 1025, 4, 4, 8),
        ("gfx950", 8, 2048, 1, 1, 8),
        ("gfx950", 8, 2048, 4, 4, 8),
        ("gfx950", 8, 2048, 8, 8, 4),
        ("gfx950", 8, 2048, 16, 16, 4),
        ("gfx950", 32, 1023, 1, 1, 8),
        ("gfx950", 32, 1024, 1, 1, 8),
        ("gfx950", 32, 1025, 1, 1, 8),
        ("gfx950", 32, 2048, 1, 1, 8),
        ("gfx950", 32, 2048, 4, 4, 2),
        ("gfx950", 64, 1024, 1, 1, 4),
        ("gfx950", 64, 1024, 4, 4, 1),
        ("gfx950", 128, 1024, 1, 1, 2),
        ("gfx950", 128, 1024, 4, 4, 1),
        ("gfx950", 64, 2048, 1, 1, 4),
        ("gfx950", 8, 4096, 1, 1, 8),
        ("gfx950", 8, 4096, 8, 8, 4),
        ("gfx942", 8, 2048, 1, 1, 1),
    ],
)
def test_bf16_swa_query_split_schedule(
    monkeypatch, arch, num_seqs, window, max_parts, capacity_per_seq, expected_splits
):
    """KV work and grid size select splits consistently across window sizes."""
    traits = importlib.import_module("aiter.ops.flydsl.kernels.pa_decode.traits")
    monkeypatch.setattr(traits, "get_rocm_arch", lambda: arch)
    schedule = traits.PaDecodeSchedule.select(
        head_dim=128,
        value_dim=128,
        query_group_size=16,
        block_size=64,
        num_seqs=num_seqs,
        num_kv_heads=1,
        num_compute_units=256,
        work_capacity=num_seqs * capacity_per_seq,
        num_partitions=max_parts,
        query_dtype="bf16",
        kv_dtype="bf16",
        query_length=8,
        trans_v=True,
        max_context_length=8192,
        sliding_window=window,
    )
    assert traits.PaDecodeTraits.create(schedule).query_splits == expected_splits


@pytest.mark.parametrize("window", [2048, 4096, 2**31 + 1])
def test_bf16_swa_schedule_context_bound(monkeypatch, window):
    traits = importlib.import_module("aiter.ops.flydsl.kernels.pa_decode.traits")
    monkeypatch.setattr(traits, "get_rocm_arch", lambda: "gfx950")
    schedule = traits.PaDecodeSchedule.select(
        head_dim=128,
        query_group_size=16,
        block_size=64,
        num_seqs=8,
        num_kv_heads=1,
        num_compute_units=256,
        work_capacity=64,
        num_partitions=8,
        query_dtype="bf16",
        kv_dtype="bf16",
        query_length=8,
        max_context_length=512,
        sliding_window=window,
    )
    assert traits.PaDecodeTraits.create(schedule).query_splits == 8


@pytest.mark.parametrize(
    "kv_dtype,scale_mode,window,expected_window",
    [
        ("fp8", "per-tensor", -1, 0),
        ("fp8", "per-tensor", 0, 0),
        ("fp8", "none", -1, 0),
        ("fp8", "per-token", 128, None),
        ("fp8", "per-tensor", -2, None),
        ("bf16", "none", -1, 0),
        ("bf16", "none", 0, 0),
        ("bf16", "none", -2, None),
        *[
            (dtype, scale, window, window)
            for dtype, scale in (("fp8", "per-tensor"), ("bf16", "none"))
            for window in (1, 16, 17, 127, 128, 129, 1024, 1025, 4097, 2**31 + 1)
        ],
    ],
)
def test_asymmetric_tuning_cli_window(kv_dtype, scale_mode, window, expected_window):
    """The full-attention alias preserves the asymmetric dtype/scale contract."""
    tuning = importlib.import_module("aiter.ops.flydsl.pa_decode_tuning")
    argv = [
        "--head-dim",
        "192",
        "--value-dim",
        "128",
        "--query-length",
        "8",
        "--block-size",
        "64",
        "--kv-dtype",
        kv_dtype,
        "--scale-mode",
        scale_mode,
        "--sliding-window",
        str(window),
    ]
    if expected_window is None:
        with pytest.raises(SystemExit) as error:
            tuning._parse_args(argv)
        assert error.value.code == 2
    else:
        assert tuning._parse_args(argv).sliding_window == expected_window


@pytest.mark.parametrize(
    "kv_dtype,window,accepted",
    [
        (dtype, window, window >= 0)
        for dtype in ("fp8", "bf16")
        for window in (-1, 0, 1, 16, 17, 127, 1025, 4097, 2**31 + 1)
    ],
)
def test_asymmetric_schedule_window_contract(monkeypatch, kv_dtype, window, accepted):
    """Both KV dtypes accept normalized full attention and positive windows."""
    traits = importlib.import_module("aiter.ops.flydsl.kernels.pa_decode.traits")
    monkeypatch.setattr(traits, "get_rocm_arch", lambda: "gfx950")
    kwargs = {
        "head_dim": 192,
        "value_dim": 128,
        "query_group_size": 16,
        "block_size": 64,
        "num_seqs": 1,
        "num_kv_heads": 1,
        "num_compute_units": 256,
        "work_capacity": 8,
        "num_partitions": 8,
        "query_dtype": "bf16",
        "kv_dtype": kv_dtype,
        "query_length": 8,
        "sliding_window": window,
    }
    if accepted:
        traits.PaDecodeSchedule.select(**kwargs)
    else:
        with pytest.raises(ValueError, match="sliding window"):
            traits.PaDecodeSchedule.select(**kwargs)


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize(
    "kv_dtype,scale_mode", [("bf16", "none"), ("fp8", "per-tensor")]
)
def test_asymmetric_tuning_cli_heads(heads, kv_dtype, scale_mode):
    tuning = importlib.import_module("aiter.ops.flydsl.pa_decode_tuning")
    args = tuning._parse_args(
        [
            "--head-dim",
            "192",
            "--value-dim",
            "128",
            "--query-length",
            "8",
            "--num-query-heads",
            str(16 * heads),
            "--num-kv-heads",
            str(heads),
            "--block-size",
            "64",
            "--kv-dtype",
            kv_dtype,
            "--scale-mode",
            scale_mode,
        ]
    )
    assert args.num_query_heads // args.num_kv_heads == 16


@pytest.mark.parametrize("value_dim", [None, 128, 64])
def test_gfx1250_value_width_dispatch(monkeypatch, value_dim):
    """The gfx1250 dispatcher must not silently ignore an asymmetric width."""
    kernel = importlib.import_module("aiter.ops.flydsl.kernels.pa_decode_kernel")
    calls = []

    def compile_gfx1250(**kwargs):
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(kernel, "compile_gfx1250_pa_decode", compile_gfx1250)
    kwargs = {
        "architecture": "gfx1250",
        "head_dim": 128,
        "value_dim": value_dim,
        "query_group_size": 16,
        "block_size": 64,
        "num_seqs": 1,
        "num_kv_heads": 1,
        "num_compute_units": 256,
        "work_capacity": 1,
    }
    if value_dim == 64:
        with pytest.raises(NotImplementedError, match="asymmetric value width"):
            kernel.compile_pa_decode_tile(**kwargs)
        assert not calls
    else:
        assert kernel.compile_pa_decode_tile(**kwargs)["head_dim"] == 128
        assert len(calls) == 1


def _reference(query, key, value, table, lengths, query_length, window, sinks):
    """FP32 attention over logical KV, independent of kernel Q/P rounding."""
    batch = lengths.numel()
    heads, dim = query.shape[1:]
    kv_heads, page_size = key.shape[1:3]
    group = heads // kv_heads
    queries = query.float().reshape(batch, query_length, kv_heads, group, dim)
    value_dim = value.shape[-1]
    output = torch.zeros(
        (*queries.shape[:-1], value_dim), dtype=torch.float32, device=query.device
    )
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
    return output.reshape(batch * query_length, heads, value_dim)


def _make_inputs(
    query_length,
    kv_heads,
    group_size,
    head_dim,
    value_dim,
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
        (batch * query_length, kv_heads * group_size, head_dim),
        dtype=dtype,
        device=device,
    ).uniform_(-0.5, 0.5)
    key = torch.empty(
        (num_pages, kv_heads, page_size, head_dim), dtype=dtype, device=device
    ).uniform_(-0.5, 0.5)
    value = torch.empty(
        (num_pages, kv_heads, page_size, value_dim), dtype=dtype, device=device
    ).uniform_(-0.5, 0.5)

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

    context = torch.tensor(lengths, dtype=torch.int32, device=device)
    return query, key, value, table, context


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
_ASYMMETRIC_WINDOW_CASES = [
    (
        f"asymmetric-window-{window}",
        {
            "query_length": 8,
            "group_size": 16,
            "head_dim": 192,
            "value_dim": 128,
            "block_size": 64,
            "max_partitions": 8,
            "window": window,
            "sink_dtype": torch.float32,
            "capture": True,
            "lengths": (0, 1, 7, 8, 63, 64, 65, 127, 128, 129, 257, 1027, 8192),
        },
    )
    for window in (1, 2, 4, 16, 17, 127, 257, 1024, 1025, 4097, 2**31 + 1)
]
_ASYMMETRIC_MULTIHEAD_CASES = [
    (
        f"asymmetric-kv{heads}-w{window}",
        {
            **_ASYMMETRIC_WINDOW_CASES[0][1],
            "kv_heads": heads,
            "window": window,
            "pattern": "varied",
            "max_partitions": 4,
            "lengths": (0, 1, 7, 8, 16, 17, 255, 257, 1027),
        },
    )
    for heads, window in ((2, 0), (2, 4), (4, 128), (4, 2048))
]
_FP8_CASES = _COMMON_CASES + [
    *_ASYMMETRIC_MULTIHEAD_CASES,
    *_ASYMMETRIC_WINDOW_CASES,
    (
        "asymmetric-large-window-grid",
        {
            **_ASYMMETRIC_WINDOW_CASES[0][1],
            "window": 1024,
            "lengths": (0, 1, 8, 16, 17, 257, 1027, 8192) * 16,
        },
    ),
    *[
        (
            f"asymmetric-probability-rounding-w{window}",
            {
                **_ASYMMETRIC_WINDOW_CASES[0][1],
                "window": window,
                "sink_dtype": sinks,
                "lengths": (
                    0,
                    1,
                    7,
                    8,
                    63,
                    64,
                    65,
                    127,
                    128,
                    129,
                    257,
                    515,
                    1027,
                    2051,
                    4096,
                    8192,
                ),
            },
        )
        for window, sinks in (
            (0, None),
            (2, None),
            (4, torch.float32),
            (128, torch.float32),
        )
    ],
    (
        "qlen8-full-d192-v128",
        {
            "query_length": 8,
            "group_size": 16,
            "head_dim": 192,
            "value_dim": 128,
            "block_size": 64,
            "max_partitions": 8,
            "scale_mode": "per-tensor",
            "trans_v": True,
            "lengths": (1027, 769),
        },
    ),
    (
        "qlen8-full-d192-v128-long",
        {
            "query_length": 8,
            "group_size": 16,
            "head_dim": 192,
            "value_dim": 128,
            "block_size": 64,
            "max_partitions": 8,
            "scale_mode": "per-tensor",
            "trans_v": True,
            "lengths": (16384,) * 8,
        },
    ),
    (
        "qlen8-w128-d192-v128",
        {
            "query_length": 8,
            "group_size": 16,
            "head_dim": 192,
            "value_dim": 128,
            "block_size": 64,
            "max_partitions": 8,
            "scale_mode": "per-tensor",
            "trans_v": True,
            "window": 128,
            "sink_dtype": torch.float32,
            "capture": True,
            "lengths": (127, 128, 129, 257, 515),
        },
    ),
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
    *_ASYMMETRIC_MULTIHEAD_CASES,
    *[
        (
            f"asymmetric-gqa8-kv2-w{geometry['window']}",
            dict(geometry, group_size=8, kv_heads=2),
        )
        for _, geometry in _ASYMMETRIC_MULTIHEAD_CASES
    ],
    *_ASYMMETRIC_WINDOW_CASES,
    (
        "asymmetric-full",
        {**_ASYMMETRIC_WINDOW_CASES[0][1], "window": 0},
    ),
    *[
        (
            f"qlen8-d128-w{window}-np{parts}",
            {
                "query_length": 8,
                "group_size": 16,
                "head_dim": 128,
                "block_size": 64,
                "max_partitions": parts,
                "window": window,
                "sink_dtype": torch.float32,
                "capture": window in (1023, 2048),
                "lengths": (
                    (0, 1, 127, 2047, 2048, 2049, 4097, 8192)
                    if window == 2048
                    else (0, 1, 127, 128, 129, 257, 515, 1027)
                ),
            },
        )
        for window, parts in (
            (512, 4),
            (1023, 8),
            (1025, 8),
            (1024, 1),
            (1024, 4),
            (1023, 1),
            (1023, 4),
            (1025, 1),
            (1025, 4),
            (2048, 1),
            (2048, 4),
            (2048, 8),
            (2048, 16),
        )
    ],
    *[
        (
            f"qlen8-d128-w2048-b32-np{parts}",
            {
                "query_length": 8,
                "group_size": 16,
                "head_dim": 128,
                "block_size": 64,
                "max_partitions": parts,
                "window": 2048,
                "sink_dtype": torch.float32,
                "capture": True,
                "lengths": (0, 1, 127, 2047, 2048, 2049, 4097, 8192) * 4,
            },
        )
        for parts in (4, 8)
    ],
    (
        "qlen8-d128-w2048-context-bound",
        {
            "query_length": 8,
            "group_size": 16,
            "head_dim": 128,
            "block_size": 64,
            "max_partitions": 8,
            "window": 2048,
            "sink_dtype": torch.float32,
            "capture": True,
            "lengths": (0, 1, 127, 128, 129, 257, 511, 512),
        },
    ),
    (
        "qlen8-w128-d192-v128",
        {
            "query_length": 8,
            "group_size": 16,
            "head_dim": 192,
            "value_dim": 128,
            "block_size": 64,
            "max_partitions": 8,
            "window": 128,
            "sink_dtype": torch.float32,
            "lengths": (127, 128, 129, 257, 515),
        },
    ),
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


def _check_output(output, reference, tolerance=5e-3):
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output.float(), reference, atol=tolerance, rtol=tolerance
    )


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
        pytest.fail("invalid inputs reached kernel compilation")

    monkeypatch.setattr(pa, "compile_pa_decode_tile", unexpected_compile)
    value_dim = query.shape[-1] if case["value_dim"] is None else case["value_dim"]
    with pytest.raises(case["error"], match=case["match"]):
        pa.pa_decode(
            torch.empty(
                (*query.shape[:-1], value_dim), dtype=query.dtype, device=query.device
            ),
            query,
            key,
            value,
            *rest,
            work_plan=plan,
            **options,
        )


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
            if geometry.get("value_dim") is None
            or (dtype == torch.bfloat16 and scale_mode == "per-tensor" and trans_v)
        ],
        *[pytest.param(geometry, id=f"bf16-{name}") for name, geometry in _BF16_CASES],
        *[pytest.param(geometry, id=name) for name, geometry in _PIPE_CASES],
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
            "value_dim": None,
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
        },
        **case,
    )
    arch = get_gfx_runtime()
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    if case["max_partitions"] is not None and case["max_partitions"] > num_cu:
        pytest.skip(f"plan_pa_decode caps max_partitions at {num_cu} CUs")
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
    query_length, kv_heads, group_size, head_dim, page, window = (
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
    value_dim = head_dim if case["value_dim"] is None else case["value_dim"]
    if value_dim != head_dim and not (
        arch == "gfx950" or (arch == "gfx942" and case["kv_dtype"] == "bf16")
    ):
        pytest.skip("asymmetric Q/K and V widths require gfx950 or gfx942 BF16")
    torch.manual_seed(0)
    query, key, value, table, context = _make_inputs(
        query_length,
        kv_heads,
        group_size,
        head_dim,
        value_dim,
        page,
        case["query_dtype"],
        case["pattern"],
        device,
        case["lengths"],
        case["table_width"],
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
    logical_key = key.float() * (1.0 if key_scale is None else key_scale)
    logical_value = value.float() * (1.0 if value_scale is None else value_scale)
    pages = key.shape[0]
    key_cache = (
        key.reshape(pages, kv_heads, page, head_dim // vector, vector)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    value_cache = (
        value.reshape(pages, kv_heads, page // vector, vector, value_dim)
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
    arguments = (
        query,
        key_cache,
        value_cache,
        context,
        table,
        head_dim**-0.5,
        query_length,
    )
    plan = pa.plan_pa_decode(
        context,
        kv_heads,
        max_partitions=case["max_partitions"],
        workgroup_budget=case["budget"],
        query_length=query_length,
        sliding_window=window,
    )
    if "invalid_input" in case:
        _check_unsupported_input(pa, arguments, options, plan, case, monkeypatch)
        return

    compile_kwargs = []
    compile_tile = pa.compile_pa_decode_tile

    def spy_tile(**kwargs):
        compile_kwargs.append(kwargs)
        return compile_tile(**kwargs)

    monkeypatch.setattr(pa, "compile_pa_decode_tile", spy_tile)

    output = torch.full(
        (*query.shape[:-1], value_dim),
        float("nan"),
        dtype=query.dtype,
        device=device,
    )
    workspace = {}
    if case["capture"]:
        scalar_shape = (kv_heads, plan.capacity, query_length * group_size)
        workspace = {
            "exp_sums": torch.empty(scalar_shape, dtype=torch.float32, device=device),
            "max_logits": torch.empty(scalar_shape, dtype=torch.float32, device=device),
            "temporary_output": torch.empty(
                (*scalar_shape, value_dim), dtype=query.dtype, device=device
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
        tolerance = (
            1e-2 if case["kv_dtype"] == "fp8" and value_dim != head_dim else 5e-3
        )
        _check_output(output, reference, tolerance)

    launch()
    check()
    _check_pipe_policy(pa, case, arch, compile_kwargs)
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


@pytest.mark.parametrize(
    "kv_dtype,window",
    [
        ("fp8", 4),
        ("fp8", 17),
        ("fp8", 128),
        ("fp8", 1025),
        ("bf16", 128),
        ("bf16", 1),
        ("bf16", 1025),
    ],
)
def test_asymmetric_autotune_plan(kv_dtype, window, device, tmp_path):
    """Tuned and cached plans must decode with the physical V/output width."""
    arch = get_gfx_runtime()
    if arch != "gfx950" and not (arch == "gfx942" and kv_dtype == "bf16"):
        pytest.skip("asymmetric Q/K and V widths require gfx950 or gfx942 BF16")
    tuning = importlib.import_module("aiter.ops.flydsl.pa_decode_tuning")
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    torch.manual_seed(0)
    # B32/NP8 lets native tuning compare different work-plan capacities.
    query, key, value, table, context = _make_inputs(
        8, 1, 16, 192, 128, 64, torch.bfloat16, "random", device, (1027,) * 32
    )
    key_scale = value_scale = None
    vector = 8
    if kv_dtype == "fp8":
        key, key_scale = per_tensor_quant(key, quant_dtype=torch.float8_e4m3fn)
        value, value_scale = per_tensor_quant(value, quant_dtype=torch.float8_e4m3fn)
        vector = 16
    reference = _reference(
        query,
        key.float() * (1.0 if key_scale is None else key_scale),
        value.float() * (1.0 if value_scale is None else value_scale),
        table,
        context,
        8,
        window,
        None,
    )
    pages = key.shape[0]
    key_cache = (
        key.reshape(pages, 1, 64, 192 // vector, vector)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    value_cache = (
        value.reshape(pages, 1, 64 // vector, vector, 128)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )
    arguments = (query, key_cache, value_cache, context, table, 192**-0.5, 8)
    options = {
        "key_scale": key_scale,
        "value_scale": value_scale,
        "sliding_window": window,
        "max_context_length": 1027,
    }
    plan = tuning.prepare_pa_decode_plan(
        *arguments, **options, max_partitions=8, tune=True, config_dir=tmp_path
    )
    assert (tmp_path / arch / "_pa_decode_autotuner.json").is_file()
    tuning._get_pa_decode_autotuner.cache_clear()
    assert tuning._get_pa_decode_autotuner(arch, tmp_path).cache
    cached_plan = tuning.prepare_pa_decode_plan(
        *arguments, **options, max_partitions=8, config_dir=tmp_path
    )
    assert cached_plan.capacity == plan.capacity
    output = torch.empty((*query.shape[:-1], 128), dtype=query.dtype, device=device)
    for selected_plan in (plan, cached_plan):
        output.fill_(float("nan"))
        pa.pa_decode(output, *arguments, **options, work_plan=selected_plan)
        _check_output(output, reference, 1e-2 if kv_dtype == "fp8" else 5e-3)

    if kv_dtype == "fp8":
        # A cached plan must switch between short and ordinary arithmetic as
        # the device lengths change, without recompilation or stale outputs.
        def launch():
            pa.plan_pa_decode(
                context, 1, query_length=8, sliding_window=window, plan=cached_plan
            )
            pa.pa_decode(output, *arguments, **options, work_plan=cached_plan)

        launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        for length in (0, 8, 16, 17, 1027):
            context.fill_(length)
            query.neg_()
            output.fill_(float("nan"))
            graph.replay()
            reference = _reference(
                query,
                key.float() * key_scale,
                value.float() * value_scale,
                table,
                context,
                8,
                window,
                None,
            )
            _check_output(output, reference, 1e-2)
