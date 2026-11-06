#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""Benchmark gfx1201 packed group-64 INT4 GEMV mappings.

Examples:
    python op_tests/op_benchmarks/hip/bench_q4_group64_gemv.py --shape 512 3584
    python op_tests/op_benchmarks/hip/bench_q4_group64_gemv.py --sweep \
        -o q4.csv --json q4_raw.json
    python op_tests/op_benchmarks/hip/bench_q4_group64_gemv.py --sweep --mappings all

Every sample records synchronized host wall-clock latency as the primary public
API metric and HIP-event latency as a device-timeline supplement. ``integration``
measures one public auto call or allocation-equivalent private control, including
output allocation. ``batched`` measures a calibrated batch of the same calls
and divides by its launch count. Neither is a claim of host-overhead-free kernel
time; use a native direct-launch benchmark for the strict kernel-only boundary.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import statistics
import time
from pathlib import Path

import torch

from aiter import q4_group64_gemv
from aiter.jit.utils.chip_info import _get_pci_chip_id
from aiter.ops.q4_group64_gemv import (
    _MAPPING_IDS,
    _q4_group64_gemv,
    _selected_mapping,
)
from op_tests.q4_group64_reference import pack_group64

SWEEP_SHAPES = [
    (512, 3584),
    (1024, 3072),
    (1024, 4096),
    (3072, 3072),
    (3072, 8192),
    (3584, 3584),
    (3584, 18944),
    (4096, 4096),
    (4096, 12288),
    (4096, 14336),
    (8192, 3072),
    (12288, 4096),
    (14336, 4096),
    (18944, 3584),
]

MIN_PACKED_RING_BYTES = 64 * 1024 * 1024
DEFAULT_TARGET_SAMPLE_MS = 100.0
DEFAULT_CALIBRATION_ITERATIONS = 100
MAX_BATCH_ITERATIONS = 2_000_000
PCI_CHIP_ID_ATTRIBUTE = 10020


def _make_case(
    n: int, k: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    q = torch.randint(-8, 8, (n, k), generator=generator, dtype=torch.int8)
    scales = torch.empty((n, k // 64), dtype=torch.float32).uniform_(
        0.005, 0.05, generator=generator
    )
    scales = scales.to(torch.float16).float()
    x = torch.empty(k, dtype=torch.float32).uniform_(-0.25, 0.25, generator=generator)
    packed = pack_group64(q, scales)

    q_groups = q.float().reshape(n, k // 64, 64)
    x_groups = x.reshape(1, k // 64, 64)
    reference = ((q_groups * x_groups).sum(dim=-1) * scales).sum(dim=-1)
    return x.cuda(), packed.cuda(), reference.cuda()


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _legal_mapping(mapping: str, n: int) -> bool:
    if mapping == "split2":
        return n % 128 == 0
    if mapping == "split4":
        return n % 64 == 0
    return True


def _resolve_requested(mapping: str, n: int, k: int) -> tuple[str, str]:
    if mapping == "selected":
        selected = _selected_mapping(n, k)
        return selected, selected
    if mapping == "auto":
        candidate = _selected_mapping(n, k)
        return "auto", f"runtime-guarded:{candidate}"
    return mapping, mapping


def _rotation_count(cache: str, rotate: int, packed_bytes: int) -> int:
    if cache == "hot":
        return 1
    if rotate > 0:
        return rotate
    return max(2, MIN_PACKED_RING_BYTES // packed_bytes + 1)


def _cyclic_order(requests: list[str], round_index: int) -> list[str]:
    offset = round_index % len(requests)
    return requests[offset:] + requests[:offset]


def _invoke_request(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    requested: str,
    mapping: str,
) -> torch.Tensor:
    if requested == "auto":
        return q4_group64_gemv(x, packed_weight)
    return _q4_group64_gemv(x, packed_weight, mapping=mapping)


def _call_path(requested: str) -> str:
    if requested == "auto":
        return "public:q4_group64_gemv"
    return "private-allocation-equivalent:_q4_group64_gemv(out=None)"


def _time_launches(
    x: torch.Tensor,
    packed_ring: torch.Tensor,
    requested: str,
    mapping: str,
    iterations: int,
    rotation: int,
) -> tuple[float, float, int]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    copies = packed_ring.shape[0]
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        _invoke_request(x, packed_ring[rotation], requested, mapping)
        rotation = (rotation + 1) % copies
    end.record()
    end.synchronize()
    wall_us = (time.perf_counter() - wall_start) * 1.0e6 / float(iterations)
    event_us = start.elapsed_time(end) * 1000.0 / float(iterations)
    return event_us, wall_us, rotation


def _summarize(
    event_latencies_us: list[float],
    wall_latencies_us: list[float],
    traffic_bytes: int,
) -> dict[str, float | list[float]]:
    ordered_event = sorted(event_latencies_us)
    ordered_wall = sorted(wall_latencies_us)
    median_event_us = statistics.median(ordered_event)
    median_wall_us = statistics.median(ordered_wall)
    return {
        "raw_us": event_latencies_us,
        "raw_event_us": event_latencies_us,
        "raw_wall_us": wall_latencies_us,
        "median_us": median_event_us,
        "median_event_us": median_event_us,
        "p10_us": _percentile(ordered_event, 0.10),
        "p90_us": _percentile(ordered_event, 0.90),
        "p10_event_us": _percentile(ordered_event, 0.10),
        "p90_event_us": _percentile(ordered_event, 0.90),
        "median_wall_us": median_wall_us,
        "p10_wall_us": _percentile(ordered_wall, 0.10),
        "p90_wall_us": _percentile(ordered_wall, 0.90),
        "effective_gbps": traffic_bytes / median_event_us / 1.0e3,
        "effective_wall_gbps": traffic_bytes / median_wall_us / 1.0e3,
    }


def _benchmark_shape(
    n: int,
    k: int,
    requests: list[str],
    *,
    cache: str,
    rotate: int,
    warmup: int,
    samples: int,
    seed: int,
    timing: str,
    calibration_iterations: int,
    target_sample_ms: float,
) -> list[dict[str, object]]:
    specs = {requested: _resolve_requested(requested, n, k) for requested in requests}

    x, packed, reference = _make_case(n, k, seed)
    packed_bytes = packed.numel() * packed.element_size()
    copies = _rotation_count(cache, rotate, packed_bytes)
    if copies == 1:
        packed_ring = packed.unsqueeze(0)
    else:
        packed_ring = packed.unsqueeze(0).expand(copies, *packed.shape).clone()
    correctness: dict[str, dict[str, float]] = {}
    for requested in requests:
        mapping, _ = specs[requested]
        actual = _invoke_request(x, packed_ring[0], requested, mapping)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, reference, rtol=5.0e-4, atol=5.0e-3)
        difference = actual - reference
        difference_l2 = float(torch.linalg.vector_norm(difference).item())
        reference_l2 = float(torch.linalg.vector_norm(reference).item())
        correctness[requested] = {
            "max_abs": float(difference.abs().max().item()),
            "relative_l2": (
                difference_l2 / reference_l2
                if reference_l2 != 0.0
                else difference_l2 / torch.finfo(reference.dtype).tiny
            ),
        }

    rotation = 0
    for round_index in range(warmup):
        for requested in _cyclic_order(requests, round_index):
            mapping, _ = specs[requested]
            _invoke_request(x, packed_ring[rotation], requested, mapping)
            rotation = (rotation + 1) % copies
    torch.cuda.synchronize()

    timing_modes = ("integration", "batched") if timing == "both" else (timing,)
    traffic_bytes = packed_bytes + x.numel() * x.element_size() + n * 4
    packed_ring_bytes = packed_bytes * copies
    total_ring_bytes = packed_ring_bytes + x.numel() * x.element_size() + n * 4
    boundary_by_timing = {
        "integration": (
            "single synchronized measurement around one public auto call or "
            "allocation-equivalent private control, including output allocation; "
            "records host wall-clock and HIP-event elapsed time"
        ),
        "batched": (
            "synchronized measurement around repeated public auto calls or "
            "allocation-equivalent private controls, each including output "
            "allocation, divided by iterations; records host wall-clock and "
            "HIP-event elapsed time"
        ),
    }
    rows: list[dict[str, object]] = []
    for timing_mode in timing_modes:
        iterations = {requested: 1 for requested in requests}
        calibration = {requested: None for requested in requests}
        calibration_wall = {requested: None for requested in requests}
        if timing_mode == "batched":
            for requested in _cyclic_order(requests, seed % len(requests)):
                mapping, _ = specs[requested]
                calibration_us, calibration_wall_us, rotation = _time_launches(
                    x,
                    packed_ring,
                    requested,
                    mapping,
                    calibration_iterations,
                    rotation,
                )
                calibration[requested] = calibration_us
                calibration_wall[requested] = calibration_wall_us
                iterations[requested] = min(
                    MAX_BATCH_ITERATIONS,
                    max(
                        10,
                        math.ceil(
                            target_sample_ms * 1000.0 / float(calibration_wall_us)
                        ),
                    ),
                )

        samples_by_request: dict[str, list[float]] = {
            requested: [] for requested in requests
        }
        wall_samples_by_request: dict[str, list[float]] = {
            requested: [] for requested in requests
        }
        records_by_request: dict[str, list[dict[str, object]]] = {
            requested: [] for requested in requests
        }
        for round_index in range(samples):
            order = _cyclic_order(requests, round_index)
            for position, requested in enumerate(order):
                mapping, _ = specs[requested]
                event_us, wall_us, rotation = _time_launches(
                    x,
                    packed_ring,
                    requested,
                    mapping,
                    iterations[requested],
                    rotation,
                )
                samples_by_request[requested].append(event_us)
                wall_samples_by_request[requested].append(wall_us)
                records_by_request[requested].append(
                    {
                        "round": round_index,
                        "position": position,
                        "execution_order": order,
                        "call_path": _call_path(requested),
                        "latency_us": event_us,
                        "event_us": event_us,
                        "wall_us": wall_us,
                    }
                )

        for requested in requests:
            mapping, resolved = specs[requested]
            summary = _summarize(
                samples_by_request[requested],
                wall_samples_by_request[requested],
                traffic_bytes,
            )
            median_us = float(summary["median_us"])
            median_wall_us = float(summary["median_wall_us"])
            rows.append(
                {
                    "n": n,
                    "k": k,
                    "requested": requested,
                    "mapping": mapping,
                    "resolved": resolved,
                    "candidate_mapping": (
                        _selected_mapping(n, k) if requested == "auto" else None
                    ),
                    "call_path": _call_path(requested),
                    "timing": timing_mode,
                    "timing_boundary": boundary_by_timing[timing_mode],
                    "execution_schedule": "cyclic_latin_by_sample_round",
                    "cache": cache,
                    "rotations": copies,
                    "packed_ring_bytes": packed_ring_bytes,
                    "total_ring_bytes": total_ring_bytes,
                    "samples": samples,
                    "iterations_per_sample": iterations[requested],
                    "calibration_us": calibration[requested],
                    "calibration_wall_us": calibration_wall[requested],
                    **summary,
                    "sample_records": records_by_request[requested],
                    "effective_tflops": (2.0 * n * k) / median_us / 1.0e6,
                    "effective_wall_tflops": (2.0 * n * k) / median_wall_us / 1.0e6,
                    "correct": True,
                    "correctness_max_abs": correctness[requested]["max_abs"],
                    "correctness_relative_l2": correctness[requested]["relative_l2"],
                }
            )
    return rows


def _mapping_requests(values: list[str]) -> list[str]:
    if values == ["all"]:
        return list(_MAPPING_IDS)
    if "all" in values:
        raise ValueError("all cannot be combined with other mappings")
    valid = {*_MAPPING_IDS, "selected"}
    invalid = [value for value in values if value not in valid]
    if invalid:
        raise ValueError(f"unknown mappings: {invalid}")
    return values


def _legal_requests_or_raise(requests: list[str], n: int, k: int) -> list[str]:
    legal_requests = [
        requested
        for requested in requests
        if _legal_mapping(_resolve_requested(requested, n, k)[0], n)
    ]
    if not legal_requests:
        raise ValueError(
            f"no legal mapping requests for shape {(n, k)}; requested {requests}"
        )
    return legal_requests


def _annotate_speedups(rows: list[dict[str, object]]) -> None:
    old_medians = {
        (int(row["n"]), int(row["k"]), str(row["timing"])): float(row["median_us"])
        for row in rows
        if row["requested"] == "old"
    }
    for row in rows:
        key = (int(row["n"]), int(row["k"]), str(row["timing"]))
        old_median = old_medians.get(key)
        row["speedup_vs_old"] = (
            old_median / float(row["median_us"]) if old_median is not None else None
        )
    old_wall_medians = {
        (int(row["n"]), int(row["k"]), str(row["timing"])): float(row["median_wall_us"])
        for row in rows
        if row["requested"] == "old"
    }
    for row in rows:
        key = (int(row["n"]), int(row["k"]), str(row["timing"]))
        old_wall_median = old_wall_medians.get(key)
        row["speedup_vs_old_wall"] = (
            old_wall_median / float(row["median_wall_us"])
            if old_wall_median is not None
            else None
        )
    auto_medians = {
        (int(row["n"]), int(row["k"]), str(row["timing"])): float(row["median_us"])
        for row in rows
        if row["requested"] == "auto"
    }
    for row in rows:
        key = (int(row["n"]), int(row["k"]), str(row["timing"]))
        auto_median = auto_medians.get(key)
        row["latency_ratio_vs_auto"] = (
            float(row["median_us"]) / auto_median if auto_median is not None else None
        )
    auto_wall_medians = {
        (int(row["n"]), int(row["k"]), str(row["timing"])): float(row["median_wall_us"])
        for row in rows
        if row["requested"] == "auto"
    }
    for row in rows:
        key = (int(row["n"]), int(row["k"]), str(row["timing"]))
        auto_wall_median = auto_wall_medians.get(key)
        row["latency_ratio_vs_auto_wall"] = (
            float(row["median_wall_us"]) / auto_wall_median
            if auto_wall_median is not None
            else None
        )


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"raw_us", "raw_event_us", "raw_wall_us", "sample_records"}
        }
        for row in rows
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def _write_json(
    path: str,
    rows: list[dict[str, object]],
    args: argparse.Namespace,
    device_identity: dict[str, object],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "aiter-q4-group64-benchmark-v3",
        "configuration": {
            "cache": args.cache,
            "rotate": args.rotate,
            "automatic_minimum_packed_ring_bytes": MIN_PACKED_RING_BYTES,
            "warmup": args.warmup,
            "samples": args.samples,
            "timing": args.timing,
            "calibration_iterations": args.calibration_iterations,
            "target_sample_ms": args.target_sample_ms,
            "mapping_execution_schedule": "cyclic_latin_by_sample_round",
            "primary_latency_metric": "host wall-clock with a synchronized boundary",
            "supplemental_latency_metric": "HIP event elapsed time",
            "output_allocation_per_call": True,
            "call_paths": {
                "auto": "public:q4_group64_gemv",
                "controls": (
                    "private-allocation-equivalent:_q4_group64_gemv(out=None)"
                ),
            },
            "device_identity_requirement": (
                "gfx1201, name AMD Radeon RX 9070 XT, 32 HIP-reported "
                "multiprocessors, PCI chip ID 0x7550; blank name is accepted "
                "only for ROCm runtimes that do not populate it"
            ),
            "device_identity": device_identity,
            "correctness_policy": {
                "check": "torch.testing.assert_close against dequantized FP32 reference",
                "rtol": 5.0e-4,
                "atol": 5.0e-3,
                "diagnostics": ["correctness_max_abs", "correctness_relative_l2"],
            },
        },
        "results": rows,
    }
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _check_arch() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires an available ROCm GPU")
    properties = torch.cuda.get_device_properties(0)
    arch = getattr(properties, "gcnArchName", "")
    normalized_arch = arch.lower().split(":", maxsplit=1)[0]
    if normalized_arch != "gfx1201":
        raise RuntimeError(f"benchmark requires gfx1201, got {arch!r}")
    name = torch.cuda.get_device_name(0) or getattr(properties, "name", "")
    multiprocessor_count = getattr(properties, "multi_processor_count", -1)
    pci_identity = _benchmark_pci_chip_id(0)
    pci_chip_id = int(pci_identity["effective_value"])
    if (
        pci_chip_id != 0x7550
        or multiprocessor_count != 32
        or (name and name != "AMD Radeon RX 9070 XT")
    ):
        raise RuntimeError(
            "14-shape benchmark requires RX 9070 XT identity: gfx1201, PCI chip "
            "ID 0x7550, 32 HIP-reported multiprocessors, and an empty runtime "
            f"name or exact AMD Radeon RX 9070 XT; got arch={arch!r}, "
            f"chip={pci_chip_id:#x}, multiprocessors={multiprocessor_count}, "
            f"name={name!r}"
        )
    return {
        "device_index": 0,
        "arch": normalized_arch,
        "pci_chip_id": pci_chip_id,
        "pci_chip_id_hex": f"0x{pci_chip_id:04x}",
        "pci_chip_id_query": pci_identity,
        "hip_reported_multiprocessors": multiprocessor_count,
        "name": name,
        "blank_name_compatibility_used": name == "",
    }


def _benchmark_pci_chip_id(device_id: int) -> dict[str, object]:
    """Read PCI chip ID with a benchmark-local ROCm 7.2 compatibility path."""

    helper_raw = _get_pci_chip_id(device_id)
    details: dict[str, object] = {
        "aiter_helper_raw": helper_raw,
        "aiter_helper_raw_hex": f"0x{helper_raw:x}",
        "fallback_attribute_id": None,
        "fallback_value": None,
        "fallback_value_hex": None,
        "fallback_used": False,
        "effective_value": helper_raw,
    }
    if 0x1000 <= helper_raw <= 0xFFFF:
        return details

    # AITER's helper currently hard-codes attribute 10019. Both the ROCm 7.2
    # container and host ROCm 7.14 headers compile hipDeviceAttributePciChipId
    # as 10020; under ROCm 7.2, querying 10019 returned the unrelated value
    # 0x100. Keep this version-specific workaround local to this evidence
    # benchmark. The operator's C++ guard uses the symbolic enum directly.
    libhip = ctypes.CDLL("libamdhip64.so")
    fallback = ctypes.c_int(0)
    error = libhip.hipDeviceGetAttribute(
        ctypes.byref(fallback), PCI_CHIP_ID_ATTRIBUTE, device_id
    )
    if error != 0:
        raise RuntimeError(
            "benchmark PCI chip ID fallback failed: "
            f"hipDeviceGetAttribute({PCI_CHIP_ID_ATTRIBUTE}) returned {error}"
        )
    if fallback.value != 0x7550:
        raise RuntimeError(
            "benchmark PCI chip ID fallback expected 0x7550, got "
            f"{fallback.value:#x}"
        )
    details.update(
        fallback_attribute_id=PCI_CHIP_ID_ATTRIBUTE,
        fallback_value=fallback.value,
        fallback_value_hex=f"0x{fallback.value:04x}",
        fallback_used=True,
        effective_value=fallback.value,
    )
    return details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark gfx1201 Q4 group-64 GEMV mappings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape", nargs=2, type=int, metavar=("N", "K"), default=(512, 3584)
    )
    parser.add_argument(
        "--sweep", action="store_true", help="run all 14 measured plain shapes"
    )
    parser.add_argument(
        "--mappings",
        nargs="+",
        default=["old", "auto", "selected"],
        help="private mappings, selected, or all",
    )
    parser.add_argument("--cache", choices=("hot", "rotating"), default="rotating")
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
        help=(
            "packed-weight copies in rotating mode; 0 chooses the minimum count "
            "whose packed ring is strictly larger than 64 MiB"
        ),
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--timing",
        choices=("integration", "batched", "both"),
        default="both",
        help="single-call integration, calibrated batch, or both boundaries",
    )
    parser.add_argument(
        "--calibration-iterations",
        type=int,
        default=DEFAULT_CALIBRATION_ITERATIONS,
    )
    parser.add_argument(
        "--target-sample-ms", type=float, default=DEFAULT_TARGET_SAMPLE_MS
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("-o", type=str, help="optional CSV output path")
    parser.add_argument("--json", type=str, help="optional raw JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rotate < 0 or args.warmup < 0 or args.samples <= 0:
        raise ValueError("rotate/warmup must be non-negative and samples positive")
    if args.calibration_iterations <= 0 or args.target_sample_ms <= 0:
        raise ValueError("calibration iterations and target sample ms must be positive")
    shapes = SWEEP_SHAPES if args.sweep else [tuple(args.shape)]
    requests = _mapping_requests(args.mappings)
    legal_requests_by_shape: list[list[str]] = []
    for n, k in shapes:
        if n <= 0 or n % 32 or k <= 0 or k % 64:
            raise ValueError(f"shape {(n, k)} must satisfy N%32 == 0 and K%64 == 0")
        legal_requests_by_shape.append(_legal_requests_or_raise(requests, n, k))
    device_identity = _check_arch()

    rows: list[dict[str, object]] = []
    header = (
        f"{'N':>7} {'K':>7} {'requested':>10} {'resolved':>11} {'timing':>11} "
        f"{'wall_us':>11} {'event_us':>11} {'wall_GB/s':>10} {'wall_TF':>9}"
    )
    print(header)
    print("-" * len(header))
    for shape_index, ((n, k), legal_requests) in enumerate(
        zip(shapes, legal_requests_by_shape, strict=True)
    ):
        shape_rows = _benchmark_shape(
            n,
            k,
            legal_requests,
            cache=args.cache,
            rotate=args.rotate,
            warmup=args.warmup,
            samples=args.samples,
            seed=args.seed + shape_index,
            timing=args.timing,
            calibration_iterations=args.calibration_iterations,
            target_sample_ms=args.target_sample_ms,
        )
        rows.extend(shape_rows)
        for row in shape_rows:
            print(
                f"{n:7d} {k:7d} {row['requested']!s:>10} {row['resolved']!s:>11} "
                f"{row['timing']!s:>11} {float(row['median_wall_us']):11.3f} "
                f"{float(row['median_event_us']):11.3f} "
                f"{float(row['effective_wall_gbps']):10.2f} "
                f"{float(row['effective_wall_tflops']):9.3f}"
            )
        first = shape_rows[0]
        print(
            f"{'':7} {'':7} {'ring':>10} {int(first['rotations']):>11} copies "
            f"packed={int(first['packed_ring_bytes']) / (1024**2):.2f} MiB "
            f"total={int(first['total_ring_bytes']) / (1024**2):.2f} MiB"
        )
        torch.cuda.empty_cache()

    _annotate_speedups(rows)
    auto_rows = [
        row
        for row in rows
        if row["requested"] == "auto" and row["speedup_vs_old"] is not None
    ]
    if auto_rows:
        print("\nauto / old speedup (host wall primary; HIP event supplemental)")
        for row in auto_rows:
            print(
                f"  {int(row['n'])}x{int(row['k'])} {row['timing']}: "
                f"wall={float(row['speedup_vs_old_wall']):.3f}x "
                f"event={float(row['speedup_vs_old']):.3f}x"
            )
    selected_rows = [
        row
        for row in rows
        if row["requested"] == "selected" and row["latency_ratio_vs_auto"] is not None
    ]
    if selected_rows:
        print(
            "\nselected / auto latency ratio (host wall primary; HIP event supplemental)"
        )
        for row in selected_rows:
            print(
                f"  {int(row['n'])}x{int(row['k'])} {row['timing']}: "
                f"wall={float(row['latency_ratio_vs_auto_wall']):.3f}x "
                f"event={float(row['latency_ratio_vs_auto']):.3f}x"
            )

    if args.o and rows:
        _write_csv(args.o, rows)
        print(f"wrote {Path(args.o).resolve()}")
    if args.json and rows:
        _write_json(args.json, rows, args, device_identity)
        print(f"wrote {Path(args.json).resolve()}")


if __name__ == "__main__":
    main()
