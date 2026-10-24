"""aiter.bench: unified kernel benchmarking.

CPU tests cover configuration, statistics, the result schema and the device
timer's kernel-to-call matching (on synthetic profiler events). GPU tests run
each timing mode on small torch kernels and skip cleanly without a GPU.
"""

import contextlib
import json
import pathlib
import statistics
import sys
from unittest.mock import patch

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

torch = pytest.importorskip("torch")
bench = pytest.importorskip("aiter.bench")

from aiter.bench import (
    BenchResult,
    EmptyGraphCaptureError,
    MeasureConfig,
    run_bench,
    time_all,
    time_fn,
)
from aiter.bench import env as bench_env
from aiter.bench.stats import count_outliers, summarize
from aiter.bench.timers import device as device_timer
from aiter.bench.timers import graph as graph_timer
from aiter.bench.timers.graph import time_graph

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
FAST = {"iters": 20, "warmup_iters": 2, "record_gpu_state": False}


# --------------------------------------------------------------------------- #
# Configuration, statistics, result schema
# --------------------------------------------------------------------------- #


def test_config_defaults_per_mode():
    assert MeasureConfig(mode="device").resolved_cache == "rotate"
    assert MeasureConfig(mode="graph").resolved_cache == "rotate"
    assert MeasureConfig(mode="eager").resolved_cache == "flush"
    assert MeasureConfig(mode="graph", cache="rotate").resolved_cache == "rotate"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "wall"},
        {"cache": "cold"},
        {"metric": "p50"},
        {"iters": 1},
        {"warmup_iters": -1},
        {"graph_min_replay_us": 0},
        {"graph_max_calls": 0},
    ],
)
def test_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        MeasureConfig(**kwargs)


def test_summarize():
    stats = summarize([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 99.0])
    assert stats.median == 5.0
    assert stats.mean == 15.0
    assert (stats.min, stats.max) == (1.0, 99.0)
    assert stats.outliers == 1
    assert summarize([5.0]).std == 0.0
    with pytest.raises(ValueError):
        summarize([])


def test_outliers_are_counted_not_removed():
    samples = [10.0] * 20 + [50.0]
    assert count_outliers(samples) == 1
    assert summarize(samples).max == 50.0


def test_result_json_round_trip():
    result = BenchResult(
        name="k", mode="graph", us=1.5, samples_us=[1.0, 2.0], per_kernel_us={"a": 1.0}
    )
    again = BenchResult.from_json(result.to_json())
    assert again == result
    data = result.to_dict()
    data["field_from_the_future"] = 1
    assert BenchResult.from_dict(data) == result
    assert data["schema_version"] == bench.SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# Device timer: matching GPU operations to calls (synthetic events)
# --------------------------------------------------------------------------- #


class _Event:
    def __init__(self, name, activity, device, start, end, corr=0):
        self._name, self._activity, self._device = name, activity, device
        self._start, self._end, self._corr = start, end, corr

    def name(self):
        return self._name

    def activity_type(self):
        return self._activity

    def device_type(self):
        return self._device

    def correlation_id(self):
        return self._corr

    def start_ns(self):
        return self._start

    def end_ns(self):
        return self._end

    def duration_ns(self):
        return self._end - self._start


CPU = torch.autograd.DeviceType.CPU
GPU = torch.autograd.DeviceType.CUDA
MARK = device_timer._MARKER


def _launch(corr, at):
    return _Event("hipLaunchKernel", "cuda_runtime", CPU, at, at + 1, corr)


def _kernel(name, corr, start, end):
    return _Event(name, "kernel", GPU, start, end, corr)


def test_split_calls_by_launch_range():
    events = [
        _Event(f"{MARK}0", "user_annotation", CPU, 0, 100, corr=1),
        _launch(1, 10),  # correlation ids may collide with CPU op ids
        _launch(2, 20),
        _launch(3, 30),
        _Event(f"{MARK}1", "user_annotation", CPU, 200, 300, corr=2),
        _launch(4, 210),
        _launch(5, 150),  # between ranges: the cache flush, ignored
        _kernel("gemm", 1, 1000, 1010),
        _kernel("reduce", 2, 1010, 1013),
        _kernel("gemm", 3, 1015, 1025),  # same kernel twice in one call
        _kernel("gemm", 4, 2000, 2012),
        _kernel("flush", 5, 1500, 1600),
        _kernel("orphan", 99, 3000, 3001),  # no launch record
        _Event(f"{MARK}0", "gpu_user_annotation", GPU, 1000, 1025),
    ]
    per_call, unmatched = device_timer._split_calls(events, 2)
    assert [op.name() for op in per_call[0]] == ["gemm", "reduce", "gemm"]
    assert [op.name() for op in per_call[1]] == ["gemm"]
    assert unmatched == 1


def test_capture_rejects_zero_nodes_without_warning(monkeypatch):
    class EmptyGraph:
        def num_nodes(self):
            return 0

    monkeypatch.setattr(graph_timer.torch.cuda, "CUDAGraph", EmptyGraph)
    monkeypatch.setattr(
        graph_timer.torch.cuda,
        "graph",
        lambda graph, stream: contextlib.nullcontext(),
    )

    with pytest.raises(EmptyGraphCaptureError):
        graph_timer._capture([lambda: None], 1, object())


def test_captured_node_count_supports_attribute_and_missing_api():
    class GraphWithNodeAttribute:
        _num_nodes = 3

    assert graph_timer._captured_node_count(GraphWithNodeAttribute()) == 3
    assert graph_timer._captured_node_count(object()) is None


# --------------------------------------------------------------------------- #
# Environment snapshot parsing (no GPU, amd-smi output mocked)
# --------------------------------------------------------------------------- #


def test_snapshot_cli_parsing():
    metric = {
        "gpu_data": [
            {
                "clock": {
                    "gfx_0": {"clk": {"value": 2100}, "clk_locked": "DISABLED"},
                    "gfx_1": {"clk": {"value": 2050}, "clk_locked": "DISABLED"},
                    "gfx_2": "N/A",
                    "mem_0": {"clk": {"value": 2000}},
                },
                "usage": {"gfx_activity": {"value": 3}, "umc_activity": {"value": 1}},
                "perf_level": "AMDSMI_DEV_PERF_LEVEL_AUTO",
                "power": {"socket_power": {"value": 120}},
                "temperature": {"hotspot": {"value": 40}},
            }
        ]
    }
    with (
        patch.object(bench_env, "_smi_index", return_value=0),
        patch.object(bench_env, "pci_bdf", return_value="0000:00:00.0"),
        patch.object(bench_env, "_run_smi", return_value=metric),
    ):
        state = bench_env._snapshot_cli(0)
    assert state["gfx_clk_mhz"] == 2100
    assert state["mem_clk_mhz"] == 2000
    assert state["gfx_clk_locked"] == ["DISABLED"]
    assert state["gfx_activity_pct"] == 3
    assert state["perf_level"] == "AMDSMI_DEV_PERF_LEVEL_AUTO"


# --------------------------------------------------------------------------- #
# GPU: every mode on real kernels
# --------------------------------------------------------------------------- #


def _mul(x, y):
    return torch.mul(x, 2.0, out=y)


def _tensors(n=1 << 16):
    x = torch.randn(n, device="cuda")
    return x, torch.empty_like(x)


@needs_gpu
@pytest.mark.parametrize("mode", ["device", "graph", "eager"])
def test_modes_time_a_kernel(mode):
    x, y = _tensors()
    result = time_fn(
        _mul, MeasureConfig(mode=mode, **FAST), args=(x, y), nbytes=2 * x.nbytes
    )
    assert result.mode == mode
    assert len(result.samples_us) == 20
    assert 0 < result.min_us <= result.median_us <= result.max_us
    assert result.us == result.median_us
    assert result.gb_s == pytest.approx(2 * x.nbytes / result.median_us / 1e3)
    assert result.cache_policy == MeasureConfig(mode=mode).resolved_cache
    assert result.compile_ms is not None
    assert result.system["gfx"]


@needs_gpu
def test_metric_mean():
    x, y = _tensors()
    result = time_fn(_mul, MeasureConfig(metric="mean", **FAST), args=(x, y))
    assert result.us == result.mean_us


@needs_gpu
def test_device_counts_every_kernel():
    x, y = _tensors()
    z = torch.empty_like(x)

    def three_kernels(x, y, z):
        torch.mul(x, 2.0, out=y)
        torch.add(y, 1.0, out=z)
        torch.mul(x, 2.0, out=y)

    result = time_fn(
        three_kernels, MeasureConfig(mode="device", **FAST), args=(x, y, z)
    )
    assert result.kernel_count == 3
    assert result.all_kernels_counted
    assert len(result.per_kernel_us) == 2
    assert sum(result.per_kernel_us.values()) == pytest.approx(
        result.median_us, rel=0.2
    )
    assert result.span_us >= result.median_us * 0.99


@needs_gpu
def test_graph_rejects_empty_capture():
    with pytest.raises(EmptyGraphCaptureError):
        time_fn(lambda: None, MeasureConfig(mode="graph", **FAST))


@needs_gpu
def test_graph_converges_in_calls_per_replay():
    x, y = _tensors(1 << 20)
    calls = [lambda: _mul(x, y)]
    small = time_graph(calls, 20, min_replay_us=500, max_calls=4096)
    large = time_graph(calls, 20, min_replay_us=2000, max_calls=4096)
    assert large.info["graph_calls_per_replay"] > small.info["graph_calls_per_replay"]
    a, b = statistics.median(small.samples_us), statistics.median(large.samples_us)
    assert abs(a - b) / b < 0.05


@needs_gpu
def test_graph_is_not_host_bound():
    # One call per replay measures the host's replay rate; K calls must not.
    x, y = _tensors(1 << 10)
    one = time_graph([lambda: _mul(x, y)], 20, min_replay_us=1e-3, max_calls=1)
    many = time_fn(_mul, MeasureConfig(mode="graph", cache="warm", **FAST), args=(x, y))
    assert many.graph_calls_per_replay > 100
    assert many.median_us < statistics.median(one.samples_us) / 2


@needs_gpu
def test_rotate_needs_inputs():
    x, y = _tensors()
    with pytest.raises(ValueError, match="rotate"):
        time_fn(
            lambda: _mul(x, y), MeasureConfig(mode="device", cache="rotate", **FAST)
        )


@needs_gpu
def test_rotate_copies_are_cold():
    x, y = _tensors()
    result = time_fn(
        _mul, MeasureConfig(mode="device", cache="rotate", **FAST), args=(x, y)
    )
    assert result.rotate_copies == 20 + device_timer.PROFILER_WARMUP
    assert not any("partly warm" in note for note in result.notes)


@needs_gpu
@pytest.mark.parametrize(
    "mode, fallback", [("device", "flush"), ("eager", "flush"), ("graph", "warm")]
)
def test_default_cache_on_closure_falls_back(mode, fallback):
    x, y = _tensors()
    result = time_fn(lambda: _mul(x, y), MeasureConfig(mode=mode, **FAST))
    assert result.cache_policy == fallback
    assert (mode == "graph") == any("timed warm" in note for note in result.notes)


@needs_gpu
def test_graph_rotates_by_default():
    x, y = _tensors(1 << 22)  # 32 MB of inputs: 16 copies cover 2x the LLC
    result = time_fn(_mul, MeasureConfig(mode="graph", **FAST), args=(x, y))
    assert result.cache_policy == "rotate"
    assert result.graph_calls_per_replay % result.rotate_copies == 0
    assert not any("partly warm" in note for note in result.notes)


@needs_gpu
def test_graph_cannot_flush():
    x, y = _tensors()
    with pytest.raises(ValueError, match="rotate"):
        time_fn(_mul, MeasureConfig(mode="graph", cache="flush", **FAST), args=(x, y))


@needs_gpu
def test_auto_cache_resolves():
    x, y = _tensors()
    result = time_fn(
        _mul, MeasureConfig(mode="eager", cache="auto", **FAST), args=(x, y)
    )
    assert result.cache_requested == "auto"
    assert result.cache_policy in ("warm", "flush")
    assert result.cache_probe_delta is not None


@needs_gpu
def test_run_bench_matches_run_perftest_convention():
    x, _ = _tensors()

    def double(x):
        return x * 2

    data, result = run_bench(double, x, measure=MeasureConfig(**FAST))
    torch.testing.assert_close(data, x * 2)
    assert isinstance(result, BenchResult)
    assert result.name == "double"


@needs_gpu
def test_time_all_and_json():
    x, y = _tensors()
    results = time_all(_mul, MeasureConfig(**FAST), args=(x, y))
    assert set(results) == {"device", "graph", "eager"}
    assert len({r.compile_ms for r in results.values()}) == 1
    for result in results.values():
        assert BenchResult.from_dict(json.loads(result.to_json())) == result
