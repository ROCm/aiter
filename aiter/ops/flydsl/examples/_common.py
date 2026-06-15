# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared helpers for the FlyDSL op examples.

This module is intentionally dependency-light (``torch`` + stdlib + the FlyDSL
op being exercised). Every per-op example folder reuses the same primitives:

* availability checks (ROCm/CUDA + the optional ``flydsl`` package),
* config loading from ``config.json``,
* seeded input generation,
* a small timing/throughput helper built on CUDA events,
* environment info plus pretty-printing and JSON serialization of run outputs.

Keep additions here generic enough to support new ops (attention, MoE, ...)
by adding a new sibling folder rather than changing this file.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional

import torch

# Reuse the canonical availability probe instead of re-implementing it.
from aiter.ops.flydsl.utils import is_flydsl_available

__all__ = [
    "DEFAULT_INPUT_SEED",
    "is_flydsl_available",
    "is_rocm_available",
    "availability",
    "require_backends",
    "TORCH_DTYPES",
    "parse_dtype",
    "load_config",
    "make_matrix",
    "TimingResult",
    "measure_time",
    "gemm_tflops",
    "environment_info",
    "tensor_stats",
    "print_table",
    "dump_json",
]

DEFAULT_INPUT_SEED = 20260401

TORCH_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "f16": torch.float16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "f32": torch.float32,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def parse_dtype(name: str) -> torch.dtype:
    """Map a config dtype string (e.g. ``"bf16"``) to a ``torch.dtype``."""
    key = str(name).strip().lower()
    if key not in TORCH_DTYPES:
        raise ValueError(
            f"Unsupported dtype {name!r}; expected one of {sorted(TORCH_DTYPES)}"
        )
    return TORCH_DTYPES[key]


def is_rocm_available() -> bool:
    """True when a ROCm/CUDA device is usable by torch."""
    return torch.cuda.is_available()


def availability() -> tuple[bool, str]:
    """Return ``(ok, message)`` describing whether examples can run on GPU.

    Callers (plain scripts, not pytest) can print the message and exit cleanly
    instead of crashing when a backend is missing.
    """
    if not is_rocm_available():
        return False, "ROCm/CUDA device not available. Skipping FlyDSL examples."
    if not is_flydsl_available():
        return False, "`flydsl` is not installed. Skipping FlyDSL examples."
    return True, "ROCm device and `flydsl` are available."


def require_backends() -> bool:
    """Print availability status; return True only when GPU + flydsl are usable."""
    ok, message = availability()
    print(f"[examples] {message}")
    return ok


def load_config(path: str) -> dict[str, Any]:
    """Load and return the parsed ``config.json`` for an op."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def make_matrix(
    rows: int,
    cols: int,
    dtype: torch.dtype,
    *,
    seed: int = DEFAULT_INPUT_SEED,
    device: str = "cuda",
    low: float = 0.0,
    high: float = 1.0,
) -> torch.Tensor:
    """Deterministically generate a ``(rows, cols)`` tensor in ``[low, high)``.

    Uses a seeded per-device generator so repeated runs see the same inputs.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    out = torch.rand((rows, cols), generator=gen, device=device, dtype=dtype)
    if low != 0.0 or high != 1.0:
        out = out * (high - low) + low
    return out


@dataclass
class TimingResult:
    label: str
    median_ms: float
    min_ms: float
    max_ms: float
    iters: int
    warmup: int


def measure_time(
    fn: Callable[[], Any],
    *,
    label: str = "",
    warmup: int = 10,
    iters: int = 50,
) -> TimingResult:
    """Time ``fn`` on the active CUDA stream using CUDA events.

    Discards ``warmup`` runs (JIT/compile/cold cache), then records ``iters``
    timed runs and returns median/min/max wall time in milliseconds.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("measure_time requires a CUDA/ROCm device")
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(max(1, iters)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    median = samples[len(samples) // 2]
    return TimingResult(
        label=label,
        median_ms=median,
        min_ms=samples[0],
        max_ms=samples[-1],
        iters=len(samples),
        warmup=max(0, warmup),
    )


def gemm_tflops(m: int, n: int, k: int, ms: float) -> float:
    """Effective TFLOP/s for an ``m x n x k`` GEMM given median time in ms."""
    if ms <= 0.0:
        return float("nan")
    flops = 2.0 * m * n * k
    return flops / (ms * 1e-3) / 1e12


def environment_info() -> dict[str, Any]:
    """Best-effort machine/runtime metadata for reproducibility in JSON output."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "rocm_available": is_rocm_available(),
        "flydsl_available": is_flydsl_available(),
    }
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            info["device_name"] = props.name
            info["device_arch"] = getattr(props, "gcnArchName", None)
        except Exception:
            pass
    return info


def tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    """Return shape/dtype plus basic float stats for a tensor."""
    tf = t.float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": tf.min().item(),
        "max": tf.max().item(),
        "mean": tf.mean().item(),
        "std": tf.std().item(),
    }


def print_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    """Print ``rows`` as an aligned text table.

    ``columns`` is a list of ``(key, header)`` pairs; missing keys render empty.
    """
    headers = [header for _, header in columns]
    widths = [len(h) for h in headers]
    cells: list[list[str]] = []
    for row in rows:
        cell = []
        for i, (key, _) in enumerate(columns):
            value = row.get(key, "")
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            cell.append(text)
            widths[i] = max(widths[i], len(text))
        cells.append(cell)

    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for cell in cells:
        print("  ".join(cell[i].ljust(widths[i]) for i in range(len(headers))))


def dump_json(path: str, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` as pretty JSON (dataclasses are serialized)."""

    def default(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        raise TypeError(f"Not JSON serializable: {type(obj)!r}")

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=default)
        fh.write("\n")
    print(f"[examples] wrote JSON results to {path}")
