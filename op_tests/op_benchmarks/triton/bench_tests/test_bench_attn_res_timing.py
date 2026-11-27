# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU checks for attention-residual benchmark timing arguments."""

from types import SimpleNamespace

import pytest

from op_tests.op_benchmarks.triton import bench_attn_res as bench


@pytest.mark.parametrize(
    "warmup_ms,rep_ms,kernel_ms,graph_calls,expected",
    [
        (25, 100, 0.005, 1000, {"warmup_replays": 5, "num_replays": 20}),
        (25, 100, 0.2, 1000, {"warmup_replays": 1, "num_replays": 1}),
        (0, 0.01, 0.005, 1000, {"warmup_replays": 0, "num_replays": 1}),
        (None, 5.1, 0.005, 1000, {"num_replays": 2}),
        (10, None, 0.005, 2000, {"warmup_replays": 1}),
    ],
)
def test_legacy_budgets_remain_milliseconds(
    warmup_ms, rep_ms, kernel_ms, graph_calls, expected
):
    counts = bench._legacy_replay_counts(warmup_ms, rep_ms, kernel_ms, graph_calls)
    assert counts == expected
    graph_ms = kernel_ms * graph_calls
    for name, budget in (("warmup_replays", warmup_ms), ("num_replays", rep_ms)):
        if budget is not None:
            assert counts[name] * graph_ms >= budget
            assert counts[name] == 0 or (counts[name] - 1) * graph_ms < budget


@pytest.mark.parametrize("kernel_ms", [0, -1, float("nan"), float("inf")])
def test_invalid_calibration_is_rejected(kernel_ms):
    with pytest.raises(ValueError, match="positive finite graph duration"):
        bench._legacy_replay_counts(25, 100, kernel_ms, 1000)


def test_default_rotating_controls_are_unchanged():
    args = bench.parse_args([])
    assert args.warmup is None and args.rep is None
    assert (
        args.graph_warmup_iters,
        args.graph_warmup_replays,
        args.graph_kernels_per_graph,
        args.graph_num_replays,
    ) == (3, 3, 1000, 10)


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], None),
        (["--rep", "1.1"], (3, 3)),
        (["--warmup", "2.1"], (5, 10)),
        (["--warmup", "0", "--rep", "1.1"], (0, 3)),
    ],
)
def test_calibration_restores_unspecified_counts(monkeypatch, flags, expected):
    calls = []
    tensor = SimpleNamespace(
        element_size=lambda: 2, shape=(1, 256), stride=lambda: (256, 1), dtype="bf16"
    )

    class Bencher:
        kernels_per_graph = 2

        def build_pool(self, *args):
            return [{"prefix": tensor} for _ in range(5)]

        def bench(self, fn, **kwargs):
            calls.append(kwargs)
            return 0.1

    monkeypatch.setattr(
        bench, "RotatingBufferBench", SimpleNamespace(from_dict=lambda _: Bencher())
    )
    args = bench.parse_args(["--op", "gate", "--graph-kernels-per-graph", "2", *flags])
    bench.bench_attn_res_fn(1, 256, 2, "packed", args)
    if expected is None:
        assert calls == [{}]
    else:
        # The five-entry pool determines graph duration, not the requested two.
        assert calls == [
            {"warmup_replays": 1, "num_replays": 1},
            {
                "warmup_iters": 3,
                "warmup_replays": expected[0],
                "kernels_per_graph": 2,
                "num_replays": expected[1],
            },
        ]


def test_legacy_budgets_accept_floats_and_graph_size():
    args = bench.parse_args(
        ["--warmup", "0", "--rep", "12.5", "--graph-kernels-per-graph", "32"]
    )
    assert args.warmup == 0 and args.rep == 12.5
    assert args.graph_kernels_per_graph == 32
    assert args.graph_warmup_replays == 3 and args.graph_num_replays == 10


@pytest.mark.parametrize(
    "args",
    [
        ["--warmup", "-1"],
        ["--warmup", "nan"],
        ["--warmup", "inf"],
        ["--rep", "0"],
        ["--rep", "-1"],
        ["--rep", "nan"],
        ["--rep", "inf"],
        ["--warmup", "25", "--graph-warmup-replays", "3"],
        ["--rep", "100", "--graph-num-replays", "10"],
        ["--backend", "arch"],
        ["-N", "1"],
        ["--op", "gate", "--out-quant", "fp8"],
    ],
)
def test_invalid_or_conflicting_arguments_fail(args):
    with pytest.raises(SystemExit) as error:
        bench.parse_args(args)
    assert error.value.code == 2


def test_arch_backend_is_available_for_gate():
    args = bench.parse_args(["--op", "gate", "--backend", "arch"])
    assert args.backend == "arch"
