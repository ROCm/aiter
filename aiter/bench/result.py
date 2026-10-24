# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""``BenchResult``: everything one timing run produced, JSON-serializable."""

import dataclasses
import json
from dataclasses import dataclass, field

# Bump when a field is renamed or its meaning changes; adding a field is fine.
SCHEMA_VERSION = 1


@dataclass
class BenchResult:
    # Identity
    name: str | None = None
    backend: str | None = None
    kernel: str | None = None
    config: dict | None = None
    tunables: dict | None = None

    # Timing (microseconds). `us` is the statistic selected by `metric`.
    mode: str = ""
    metric: str = "median"
    us: float = 0.0
    samples_us: list = field(default_factory=list)
    median_us: float = 0.0
    mean_us: float = 0.0
    min_us: float = 0.0
    max_us: float = 0.0
    std_us: float = 0.0
    outliers: int = 0

    # Cache policy: requested (None = per-mode default) and what actually ran
    cache_requested: str | None = None
    cache_policy: str = ""
    cache_probe_delta: float | None = None  # auto: (cold - warm) / cold
    rotate_copies: int | None = None
    rotate_llc_coverage: float | None = None

    # Mode details
    eager_sync_each: bool | None = None
    graph_calls_per_replay: int | None = None
    captured_nodes: int | None = None
    per_kernel_us: dict | None = None
    span_us: float | None = None
    kernel_count: int | None = None
    all_kernels_counted: bool | None = None

    # Cost
    compile_ms: float | None = None
    warmup_iters: int = 0
    iters: int = 0

    # Throughput (from median_us)
    flops: float | None = None
    nbytes: float | None = None
    tflops: float | None = None
    gb_s: float | None = None
    tflops_peak_ratio: float | None = None
    gbps_peak_ratio: float | None = None

    # Correctness: None (not checked), "pass", "fail" or "skipped"
    verified: str | None = None
    err: float | None = None

    # Context
    system: dict | None = None
    gpu_state_before: dict | None = None
    gpu_state_after: dict | None = None
    telemetry: dict | None = None
    notes: list = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), default=str, **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "BenchResult":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, text: str) -> "BenchResult":
        return cls.from_dict(json.loads(text))

    def summary(self) -> dict:
        """Compact columns for @benchmark tables (no samples, no metadata)."""
        return {
            "mode": self.mode,
            "cache": self.cache_policy,
            "us": self.us,
            "median_us": self.median_us,
            "mean_us": self.mean_us,
            "std_us": self.std_us,
            "tflops": self.tflops,
            "gb_s": self.gb_s,
        }
