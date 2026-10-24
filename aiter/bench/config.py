# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""``MeasureConfig``: the one object that configures a timing run."""

import dataclasses
from dataclasses import dataclass
from typing import Literal, get_args

Mode = Literal["device", "graph", "eager"]
CachePolicy = Literal["warm", "rotate", "flush", "auto"]
Metric = Literal["median", "mean"]

MODES: tuple = get_args(Mode)
CACHE_POLICIES: tuple = get_args(CachePolicy)
METRICS: tuple = get_args(Metric)

# Cache policy used when MeasureConfig.cache is None. Every mode defaults to
# cold inputs, so a number never depends on what the previous call left in the
# cache.
#   device: rotate -- matches run_perftest, so device numbers stay comparable.
#   graph:  rotate -- a graph cannot flush between recorded calls, so cold
#                     inputs come from cycling input copies.
#   eager:  flush  -- matches triton.testing.do_bench.
# When the callable takes no inputs (a closure), rotate falls back to flush,
# or to warm in graph mode, with a note in the result.
DEFAULT_CACHE: dict = {"device": "rotate", "graph": "rotate", "eager": "flush"}


def _check_choice(name, value, choices):
    if value not in choices:
        raise ValueError(f"{name}={value!r} is not one of {choices}")


@dataclass(frozen=True)
class MeasureConfig:
    """How to time a callable.

    mode:            device | graph | eager. See aiter/bench/README.md.
    cache:           warm | rotate | flush | auto, or None for the per-mode
                     default in DEFAULT_CACHE.
    warmup_iters:    untimed calls after the first (compile) call.
    iters:           timed samples. Graph mode: one sample per graph replay.
    metric:          which statistic BenchResult.us reports. "median" is the
                     headline; "mean" reproduces run_perftest and the tuners.
    eager_sync_each: eager only. Synchronize after every call so each sample
                     includes the full host round trip, instead of the
                     do_bench pattern (one sync at the end).
    graph_min_replay_us: graph only. Calls per recorded graph (K) are chosen
                     so one replay takes at least this long. Each replay has a
                     fixed cost (host launch, and 5-10 us on the GPU) spread
                     over K calls; at 1000 us it adds about 1% per call.
    graph_max_calls: graph only. Upper bound on K.
    rotate_max_copies: rotate only. Upper bound on input copies.
    record_gpu_state: snapshot clocks, performance level and activity before
                     and after timing (amdsmi or the amd-smi CLI).
    telemetry:       sample GPU metrics on a background thread during the
                     timed window (needs the amdsmi Python binding).
    """

    mode: Mode = "device"
    cache: CachePolicy | None = None
    warmup_iters: int = 10
    iters: int = 100
    metric: Metric = "median"
    eager_sync_each: bool = False
    graph_min_replay_us: float = 1000.0
    graph_max_calls: int = 4096
    rotate_max_copies: int = 4096
    record_gpu_state: bool = True
    telemetry: bool = False

    def __post_init__(self):
        _check_choice("mode", self.mode, MODES)
        if self.cache is not None:
            _check_choice("cache", self.cache, CACHE_POLICIES)
        _check_choice("metric", self.metric, METRICS)
        if self.warmup_iters < 0:
            raise ValueError(f"warmup_iters must be >= 0, got {self.warmup_iters}")
        if self.iters < 2:
            raise ValueError(f"iters must be >= 2, got {self.iters}")
        if self.graph_min_replay_us <= 0:
            raise ValueError("graph_min_replay_us must be > 0")
        if self.graph_max_calls < 1 or self.rotate_max_copies < 1:
            raise ValueError("graph_max_calls and rotate_max_copies must be >= 1")

    @property
    def resolved_cache(self) -> CachePolicy:
        """The cache policy before an ``auto`` probe has run."""
        return self.cache if self.cache is not None else DEFAULT_CACHE[self.mode]

    def replace(self, **changes) -> "MeasureConfig":
        return dataclasses.replace(self, **changes)
