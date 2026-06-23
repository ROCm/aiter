# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Vendored kernel-timing primitives for the fp8_mqa_logits A/B bench.

Two timing strategies are provided:

* ``eager`` -- one full closure call per sample, bracketed by GPU events and
  synchronized. Captures device work PLUS the host bubble (per-call dispatch /
  launch round trip). Maps to non-graph per-call serving latency.
* ``graph`` -- captures one closure call into a HIP graph, then times serial
  back-to-back replays divided by the replay count. Strips per-call host
  dispatch overhead; the fair production-style steady-state number.

Report ``median_us`` as the authoritative latency; keep mean/std as spread.
"""
import statistics
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import torch


# --------------------------------------------------------------------------- #
# Config + results
# --------------------------------------------------------------------------- #
@dataclass
class MeasureConfig:
    """Timing knobs shared by every strategy."""

    warmup_iters: int = 10
    bench_iters: int = 20
    replay_iters: int = 50
    graph_replay_iters: int = 50
    strategy: str = "graph"

    def __post_init__(self):
        if self.warmup_iters < 0:
            raise ValueError("warmup_iters must be >= 0")
        if self.bench_iters < 1:
            raise ValueError("bench_iters must be >= 1")
        if self.graph_replay_iters < 1:
            raise ValueError("graph_replay_iters must be >= 1")
        if self.replay_iters < 1:
            raise ValueError("replay_iters must be >= 1")


@dataclass
class TimingStats:
    """Aggregate of timed samples (all microseconds)."""

    samples_us: list = field(default_factory=list)
    strategy: str = ""
    warmup_iters: int = 0

    @property
    def median_us(self):
        return statistics.median(self.samples_us)

    @property
    def mean_us(self):
        return statistics.fmean(self.samples_us)

    @property
    def std_us(self):
        return statistics.pstdev(self.samples_us) if len(self.samples_us) > 1 else 0.0

    @property
    def min_us(self):
        return min(self.samples_us)

    @property
    def max_us(self):
        return max(self.samples_us)


# --------------------------------------------------------------------------- #
# Graph-capture support
# --------------------------------------------------------------------------- #
class EmptyGraphCaptureError(RuntimeError):
    """Graph capture recorded no work, so the timed graph is empty.

    A FlyDSL launcher that does not declare a ``stream: fx.Stream`` parameter
    runs on the HIP NULL stream; ``torch.cuda.graph`` captures a side stream and
    drops those launches, leaving an empty graph that would otherwise time as a
    meaningless ~1 us. Such a kernel also cannot run inside a graph-based
    inference server, so this is a real defect, not a measurement nuisance.
    """

    def __init__(self) -> None:
        super().__init__(
            "graph capture recorded zero nodes: the kernel launched on the NULL "
            "stream and was not captured. Declare a 'stream: fx.Stream' parameter "
            "on the launcher, pass torch.cuda.current_stream() from run(), and "
            "forward it to .launch(..., stream=stream) so the kernel is "
            "graph-capturable."
        )


def _captured_node_count(graph: object):
    """Captured node count if the torch CUDAGraph binding exposes it, else None.

    The attribute is build-dependent (method or plain attr across ROCm torch
    builds), so probe defensively; None means the count cannot be determined and
    the caller falls back to the runtime's empty-graph warning.
    """
    for attr in ("num_nodes", "_num_nodes"):
        probe = getattr(graph, attr, None)
        if probe is None:
            continue
        value = probe() if callable(probe) else probe
        if isinstance(value, int):
            return value
    return None


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
def bench_eager(closure: Callable[[], None], measure: MeasureConfig) -> TimingStats:
    """Per-call eager timing: device work + host bubble.

    Each sample records a start event, runs ONE complete closure call, records an
    end event, and synchronizes. Do not batch many back-to-back calls under one
    event pair -- that keeps the queue full and erases the host bubble eager
    exists to measure.
    """
    for _ in range(measure.warmup_iters):
        closure()
    torch.cuda.synchronize()

    samples: list = []
    for _ in range(measure.bench_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(measure.replay_iters):
            closure()
        end.record()
        torch.cuda.synchronize()
        samples.append(
            (start.elapsed_time(end) / measure.replay_iters) * 1000.0
        )  # ms -> us
    return TimingStats(
        samples_us=samples, strategy="eager", warmup_iters=measure.warmup_iters
    )


def bench_graph(closure: Callable[[], None], measure: MeasureConfig) -> TimingStats:
    """Graph-replay timing: fair production-style steady-state device work.
    """
    # Warm on a dedicated capture stream. The floor of max(3, warmup_iters)
    # ensures at least one call lands after the FlyDSL cache miss (which runs the
    # kernel twice: once for the real output, then flyc.compile runs one canary
    # launch), so the cache is fully populated before capture.
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        for _ in range(max(3, measure.warmup_iters)):
            closure()
    torch.cuda.synchronize()

    # Capture one launcher call. torch.cuda.graph makes capture_stream current
    # for the block, so a launcher that threads torch.cuda.current_stream()
    # enqueues on it and is recorded. A launcher that omits the stream param runs
    # on the NULL stream and is dropped, leaving an empty graph; detect that and
    # raise rather than time ~1 us.
    g = torch.cuda.CUDAGraph()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.cuda.graph(g, stream=capture_stream):
            closure()

    node_count = _captured_node_count(g)
    empty_warning = any("empty" in str(w.message).lower() for w in caught)
    if node_count == 0 or (node_count is None and empty_warning):
        raise EmptyGraphCaptureError()

    # keepalive holds the runner that owns the FlyDSL engine the captured nodes
    # point into. If the caller's reference to closure is the only one and Python
    # GCs it, graph.replay() becomes a use-after-free.
    keepalive = closure  # noqa: F841

    # Each sample brackets graph_replay_iters back-to-back replays under one
    # event pair and divides by the replay count. Replays are serialized by the
    # stream FIFO, so this is serial device time -- not a throughput/overlap
    # measurement. Bracketing removes per-replay event-record overhead and the
    # per-replay host dispatch overhead where the GPU would otherwise wait.
    samples: list = []
    for _ in range(measure.bench_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(measure.graph_replay_iters):
            g.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(
            (start.elapsed_time(end) / measure.graph_replay_iters) * 1000.0
        )  # ms -> us
    return TimingStats(
        samples_us=samples, strategy="graph", warmup_iters=measure.warmup_iters
    )


_STRATEGIES = {"eager": bench_eager, "graph": bench_graph}


def measure(
    closure: Callable[[], None], strategy: str, cfg: MeasureConfig
) -> TimingStats:
    """Dispatch to a timing strategy by name. Raises on an unknown strategy."""
    fn = _STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(
            f"unknown timing strategy {strategy!r}; available: "
            f"{list(_STRATEGIES)}"
        )
    return fn(closure, cfg)
