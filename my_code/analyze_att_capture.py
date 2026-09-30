#!/usr/bin/env python3
"""Analyze ATT realtime and occupancy captures and plot per-SE GFX clocks."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from statistics import fmean, median
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIMD_IDS = (0, 1, 2, 3)
EXPECTED_SE_NAMES = ("SE0", "SE1", "SE2", "SE3")
DEFAULT_REFERENCE_HZ = 100_000_000.0
UNWRAPPED_DEFAULT_SIMD_ID = 3
COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red")
LINE_STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D")
Sample = tuple[float, float]


@dataclass(frozen=True)
class RealtimeCapture:
    path: Path
    reference_hz: float
    metadata_hz: Optional[float]
    reference_source: str
    samples_by_se: dict[str, tuple[Sample, ...]]


@dataclass(frozen=True)
class SeriesSummary:
    se_name: str
    sample_count: int
    valid_intervals: int
    dropped_nonpositive_realtime: int
    reference_hz: float
    elapsed_us: tuple[float, ...]
    interval_mhz: tuple[float, ...]
    total_shader_ticks: float
    total_realtime_ticks: float
    weighted_mean_mhz: float
    min_interval_mhz: float
    max_interval_mhz: float


@dataclass(frozen=True)
class CaptureSummary:
    simd_id: Optional[int]
    capture: RealtimeCapture
    series: tuple[SeriesSummary, ...]
    valid_intervals: int
    dropped_nonpositive_realtime: int
    total_shader_ticks: float
    total_realtime_ticks: float
    weighted_mean_mhz: float


@dataclass(frozen=True)
class SelectedCaptureFiles:
    simd_id: int
    realtime_path: Path
    occupancy_path: Path


@dataclass(frozen=True)
class OccupancyEvent:
    shader_timestamp: int
    packed_sa_wgp: int
    hw_simd: int
    wave_slot: int
    start: int
    kernel_pc_index: int


@dataclass(frozen=True)
class OccupancyCapture:
    path: Path
    events_by_se: dict[str, tuple[OccupancyEvent, ...]]
    kernel_pc_labels: dict[int, str]


@dataclass(frozen=True)
class PerSECycleSpan:
    se_name: str
    sample_count: int
    cycle_span: float


@dataclass(frozen=True)
class ObservedCycleMaximum:
    source_name: str
    cycle_span: float
    coordinates: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class WaveLifetime:
    se_name: str
    packed_sa_wgp: int
    hw_simd: int
    wave_slot: int
    kernel_pc_index: int
    start_timestamp: int
    end_timestamp: int

    @property
    def duration(self) -> int:
        return self.end_timestamp - self.start_timestamp


@dataclass(frozen=True)
class WaveReuseSummary:
    se_name: str
    packed_sa_wgp: int
    hw_simd: int
    wave_slot: int
    kernel_pc_index: int
    wave_count: int


@dataclass(frozen=True)
class SlotUsageSummary:
    se_name: str
    packed_sa_wgp: int
    hw_simd: int
    kernel_pc_index: int
    distinct_slot_ids: tuple[int, ...]
    max_concurrent_active_slots: int


@dataclass(frozen=True)
class WGPUsageSummary:
    se_name: str
    kernel_pc_index: int
    packed_sa_wgp_ids: tuple[int, ...]


@dataclass(frozen=True)
class WGPEpisode:
    se_name: str
    packed_sa_wgp: int
    kernel_pc_index: int
    start_timestamp: int
    end_timestamp: int

    @property
    def duration(self) -> int:
        return self.end_timestamp - self.start_timestamp


@dataclass(frozen=True)
class PhysicalWGPSummary:
    se_name: str
    packed_sa_wgp: int
    kernel_pc_index: int
    episode_count: int
    start_timestamp: int
    end_timestamp: int
    envelope_duration: int
    active_duration: int
    idle_gap_duration: int


@dataclass(frozen=True)
class WGPCompletionSummary:
    se_name: str
    kernel_pc_index: int
    wgp_count: int
    episode_count_min: int
    episode_count_max: int
    envelope_min: int
    envelope_median: float
    envelope_max: int
    active_median: float
    idle_gap_median: float
    final_end_min: int
    final_end_max: int
    final_end_span: Optional[int]
    completion_imbalance: Optional[float]


@dataclass(frozen=True)
class OccupancySummary:
    capture: OccupancyCapture
    cycle_spans: tuple[PerSECycleSpan, ...]
    wave_lifetimes: tuple[WaveLifetime, ...]
    wave_reuse: tuple[WaveReuseSummary, ...]
    slot_usage: tuple[SlotUsageSummary, ...]
    wgp_usage: tuple[WGPUsageSummary, ...]
    wgp_episodes: tuple[WGPEpisode, ...]
    physical_wgp_summaries: tuple[PhysicalWGPSummary, ...]
    completion_by_group: tuple[WGPCompletionSummary, ...]


@dataclass(frozen=True)
class DirectoryCaptureSummary:
    simd_id: int
    files: SelectedCaptureFiles
    realtime: CaptureSummary
    realtime_cycle_spans: tuple[PerSECycleSpan, ...]
    occupancy: OccupancySummary


def default_input_path(script_dir: Path) -> Path:
    return (
        script_dir
        / "f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps.asm.d7-2-gpu1.att"
        / "thread_trace"
        / "simd0"
        / "kernel"
        / "rpf_v3"
        / "ui_output_agent_13454_dispatch_44"
        / "realtime.json"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze ATT realtime and occupancy captures, report wave/WGP behavior, "
            "and plot per-SE GFX clocks. With --dir, discover the existing "
            "thread_trace/simdN captures, or a single thread_trace/kernel "
            "capture, and strictly pair one "
            "kernel/rpf_v3/**/realtime.json with occupancy.json in its UI directory."
        ),
        epilog=(
            "Directory-mode output defaults to the ATT directory's parent and is "
            "named <ATT_BASENAME>.simdN.png for every discovered capture. Examples:\n"
            "  python analyze_att_capture.py --dir ./capture.att\n"
            "  python my_code/analyze_att_capture.py --dir my_code/capture.att"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--input",
        type=Path,
        help="Legacy mode: input realtime.json path",
    )
    mode.add_argument(
        "--dir",
        dest="att_dir",
        type=Path,
        help=(
            "Directory mode: ATT root containing any non-empty subset of "
            "thread_trace/simd0..simd3, or a single unwrapped capture at "
            "thread_trace/kernel/rpf_v3; incomplete simdN directories "
            "without realtime.json are skipped; writes one derived PNG "
            "per complete capture outside the trace root"
        ),
    )
    mode.add_argument(
        "--self-test",
        action="store_true",
        help="Run lightweight pure-function and discovery tests, then exit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Legacy mode output image path "
            "(default: my_code/fgfx_all_se.png)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory mode output directory (default: parent of --dir); "
            "must be outside the ATT trace root"
        ),
    )
    parser.add_argument(
        "--reference-hz",
        type=float,
        default=None,
        help=(
            "REALTIME reference clock in Hz. When provided, overrides "
            "metadata.frequency for every selected realtime.json. When "
            "omitted, a valid metadata.frequency is used; if it is missing "
            "or <= 0, the default "
            f"{DEFAULT_REFERENCE_HZ:g} Hz (100 MHz) is used"
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help=(
            "Directory mode: print the analysis report without writing PNG "
            "plots"
        ),
    )
    return parser


def se_sort_key(name: str) -> int:
    match = re.fullmatch(r"SE(\d+)", name)
    if match is None:
        raise ValueError(f"Invalid SE key: {name!r}")
    return int(match.group(1))


def _finite_sample_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return converted


def _optional_metadata_frequency(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _strict_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{label} must be non-negative, got {value!r}")
    return value


def load_and_validate_realtime(
    input_path: Path,
    reference_hz_override: Optional[float] = None,
    required_se_names: Optional[Sequence[str]] = None,
) -> RealtimeCapture:
    """Load one realtime.json and validate its metadata and sample pairs."""
    path = input_path.expanduser()
    if not path.is_file():
        raise ValueError(f"Input realtime.json does not exist: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSON value must be an object: {path}")

    metadata = data.get("metadata")
    metadata_value = (
        metadata.get("frequency") if isinstance(metadata, dict) else None
    )
    metadata_hz = _optional_metadata_frequency(metadata_value)

    if reference_hz_override is not None:
        reference_hz = float(reference_hz_override)
        if not math.isfinite(reference_hz) or reference_hz <= 0:
            raise ValueError("--reference-hz must be finite and greater than zero")
        if metadata_hz is None:
            metadata_description = "missing or invalid"
        else:
            metadata_description = f"{metadata_hz:g} Hz"
        reference_source = (
            f"CLI override; metadata={metadata_description}"
        )
    elif metadata_hz is not None and metadata_hz > 0:
        reference_hz = metadata_hz
        reference_source = "metadata"
    else:
        reference_hz = DEFAULT_REFERENCE_HZ
        reference_source = "default --reference-hz 100 MHz"

    malformed_se_names = sorted(
        name
        for name in data
        if name.startswith("SE") and re.fullmatch(r"SE\d+", name) is None
    )
    if malformed_se_names:
        raise ValueError(
            f"{path}: malformed SE keys: {', '.join(malformed_se_names)}"
        )
    se_names = sorted(
        (
            name
            for name in data
            if re.fullmatch(r"SE\d+", name) is not None
        ),
        key=se_sort_key,
    )
    if not se_names:
        raise ValueError(f"{path}: no SE samples found")

    if required_se_names is not None:
        expected = set(required_se_names)
        actual = set(se_names)
        missing = sorted(expected - actual, key=se_sort_key)
        unexpected = sorted(actual - expected, key=se_sort_key)
        problems = []
        if missing:
            problems.append(f"missing {', '.join(missing)}")
        if unexpected:
            problems.append(f"unexpected {', '.join(unexpected)}")
        if problems:
            raise ValueError(
                f"{path}: expected exactly {', '.join(required_se_names)}; "
                + "; ".join(problems)
            )

    samples_by_se: dict[str, tuple[Sample, ...]] = {}
    for se_name in se_names:
        raw_samples = data[se_name]
        if not isinstance(raw_samples, list):
            raise ValueError(f"{path}: {se_name} samples must be a list")
        if not raw_samples:
            raise ValueError(f"{path}: {se_name} samples are empty")
        if len(raw_samples) < 2:
            raise ValueError(
                f"{path}: {se_name} needs at least two samples, "
                f"found {len(raw_samples)}"
            )

        samples: list[Sample] = []
        for sample_index, raw_sample in enumerate(raw_samples):
            if not isinstance(raw_sample, list) or len(raw_sample) != 2:
                raise ValueError(
                    f"{path}: {se_name} sample {sample_index} must be "
                    "[gfx_clock, realtime_clock]"
                )
            shader_clock = _finite_sample_number(
                raw_sample[0], f"{path}: {se_name} sample {sample_index} gfx_clock"
            )
            realtime_clock = _finite_sample_number(
                raw_sample[1],
                f"{path}: {se_name} sample {sample_index} realtime_clock",
            )
            samples.append((shader_clock, realtime_clock))
        samples_by_se[se_name] = tuple(samples)

    return RealtimeCapture(
        path=path,
        reference_hz=reference_hz,
        metadata_hz=metadata_hz,
        reference_source=reference_source,
        samples_by_se=samples_by_se,
    )


def load_and_validate_occupancy(
    input_path: Path,
    required_se_names: Optional[Sequence[str]] = None,
) -> OccupancyCapture:
    """Load the verified gfx1250 occupancy tuple schema."""
    path = input_path.expanduser()
    if not path.is_file():
        raise ValueError(f"Input occupancy.json does not exist: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSON value must be an object: {path}")

    canonical_numeric_key = re.compile(r"(?:0|[1-9]\d*)")
    numeric_keys = sorted(
        (
            key
            for key in data
            if isinstance(key, str)
            and canonical_numeric_key.fullmatch(key) is not None
        ),
        key=int,
    )
    unexpected_keys = sorted(
        str(key)
        for key in data
        if key not in numeric_keys and key not in {"dispatches", "version"}
    )
    if unexpected_keys:
        raise ValueError(
            f"{path}: unexpected occupancy top-level keys: "
            + ", ".join(repr(key) for key in unexpected_keys)
        )
    if not numeric_keys:
        raise ValueError(f"{path}: no numeric top-level SE keys found")

    se_names = [f"SE{int(key)}" for key in numeric_keys]
    if required_se_names is not None:
        expected = set(required_se_names)
        actual = set(se_names)
        missing = sorted(expected - actual, key=se_sort_key)
        unexpected = sorted(actual - expected, key=se_sort_key)
        problems = []
        if missing:
            problems.append(f"missing {', '.join(missing)}")
        if unexpected:
            problems.append(f"unexpected {', '.join(unexpected)}")
        if problems:
            raise ValueError(
                f"{path}: occupancy SE set must match "
                f"{', '.join(required_se_names)}; "
                + "; ".join(problems)
            )

    raw_kernel_pc_labels = data.get("dispatches", {})
    if not isinstance(raw_kernel_pc_labels, dict):
        raise ValueError(
            f"{path}: profiler 'dispatches' kernel-label map must be an "
            "object when present"
        )
    kernel_pc_labels: dict[int, str] = {}
    for raw_index, label in raw_kernel_pc_labels.items():
        if (
            not isinstance(raw_index, str)
            or canonical_numeric_key.fullmatch(raw_index) is None
        ):
            raise ValueError(
                f"{path}: profiler 'dispatches' kernel-label key must be a "
                f"non-negative integer string, got {raw_index!r}"
            )
        if not isinstance(label, str):
            raise ValueError(
                f"{path}: profiler 'dispatches' kernel label "
                f"[{raw_index!r}] must be a string, got {label!r}"
            )
        kernel_pc_labels[int(raw_index)] = label
    if "version" in data and not isinstance(data["version"], str):
        raise ValueError(f"{path}: version must be a string when present")

    events_by_se: dict[str, tuple[OccupancyEvent, ...]] = {}
    used_kernel_pc_indices: set[int] = set()
    for numeric_key, se_name in zip(numeric_keys, se_names):
        raw_events = data[numeric_key]
        if not isinstance(raw_events, list):
            raise ValueError(f"{path}: {numeric_key} ({se_name}) must be a list")
        if len(raw_events) < 2:
            raise ValueError(
                f"{path}: {numeric_key} ({se_name}) needs at least two events, "
                f"found {len(raw_events)}"
            )

        events: list[OccupancyEvent] = []
        for event_index, raw_event in enumerate(raw_events):
            label = f"{path}: {numeric_key} ({se_name}) event {event_index}"
            if not isinstance(raw_event, list) or len(raw_event) != 6:
                raise ValueError(
                    f"{label} must be [shader_timestamp, packed_sa_wgp, simd, "
                    "wave_slot, start, kernel_pc_index]"
                )
            timestamp = _strict_nonnegative_int(
                raw_event[0], f"{label} shader_timestamp"
            )
            packed_sa_wgp = _strict_nonnegative_int(
                raw_event[1], f"{label} packed_sa_wgp"
            )
            hw_simd = _strict_nonnegative_int(raw_event[2], f"{label} simd")
            wave_slot = _strict_nonnegative_int(
                raw_event[3], f"{label} wave_slot"
            )
            start = _strict_nonnegative_int(raw_event[4], f"{label} start")
            kernel_pc_index = _strict_nonnegative_int(
                raw_event[5], f"{label} kernel_pc_index"
            )
            if packed_sa_wgp > 0xFF:
                raise ValueError(
                    f"{label} packed_sa_wgp has bits outside gfx1250 "
                    f"SA[7]/WGP[6:0]: {packed_sa_wgp}"
                )
            if hw_simd >= 4:
                raise ValueError(
                    f"{label} simd must be in 0..3 for a four-SIMD WGP, "
                    f"got {hw_simd}"
                )
            if start not in (0, 1):
                raise ValueError(f"{label} start must be 0 or 1, got {start}")
            used_kernel_pc_indices.add(kernel_pc_index)
            events.append(
                OccupancyEvent(
                    shader_timestamp=timestamp,
                    packed_sa_wgp=packed_sa_wgp,
                    hw_simd=hw_simd,
                    wave_slot=wave_slot,
                    start=start,
                    kernel_pc_index=kernel_pc_index,
                )
            )
        events_by_se[se_name] = tuple(events)

    if kernel_pc_labels:
        missing_kernel_pc_labels = sorted(
            used_kernel_pc_indices - set(kernel_pc_labels)
        )
        if missing_kernel_pc_labels:
            raise ValueError(
                f"{path}: events reference kernel-PC label indices absent from "
                "the profiler 'dispatches' kernel-label map: "
                + ", ".join(str(index) for index in missing_kernel_pc_labels)
            )

    return OccupancyCapture(
        path=path,
        events_by_se=events_by_se,
        kernel_pc_labels=kernel_pc_labels,
    )


def compute_interval_summary(
    se_name: str,
    samples: Sequence[Sample],
    reference_hz: float,
    realtime_origin: float,
) -> SeriesSummary:
    """Compute interval values and a REALTIME-tick-weighted mean."""
    elapsed_us: list[float] = []
    interval_mhz: list[float] = []
    total_shader_ticks = 0.0
    total_realtime_ticks = 0.0
    dropped_nonpositive_realtime = 0

    for interval_index, (sample_0, sample_1) in enumerate(
        zip(samples, samples[1:])
    ):
        shader_0, realtime_0 = sample_0
        shader_1, realtime_1 = sample_1
        delta_shader = shader_1 - shader_0
        delta_realtime = realtime_1 - realtime_0

        if delta_shader < 0:
            raise ValueError(
                f"{se_name}: gfx_clock decreases at adjacent sample pair "
                f"{interval_index}->{interval_index + 1}"
            )
        if delta_realtime <= 0:
            dropped_nonpositive_realtime += 1
            continue

        midpoint_realtime = (realtime_0 + realtime_1) / 2.0
        elapsed_us.append(
            (midpoint_realtime - realtime_origin) / reference_hz * 1.0e6
        )
        interval_mhz.append(
            reference_hz * delta_shader / delta_realtime / 1.0e6
        )
        total_shader_ticks += delta_shader
        total_realtime_ticks += delta_realtime

    if not interval_mhz:
        raise ValueError(
            f"{se_name}: no valid intervals remain after filtering "
            f"{dropped_nonpositive_realtime} delta_realtime<=0 pairs"
        )

    weighted_mean_mhz = (
        reference_hz
        * total_shader_ticks
        / total_realtime_ticks
        / 1.0e6
    )
    return SeriesSummary(
        se_name=se_name,
        sample_count=len(samples),
        valid_intervals=len(interval_mhz),
        dropped_nonpositive_realtime=dropped_nonpositive_realtime,
        reference_hz=reference_hz,
        elapsed_us=tuple(elapsed_us),
        interval_mhz=tuple(interval_mhz),
        total_shader_ticks=total_shader_ticks,
        total_realtime_ticks=total_realtime_ticks,
        weighted_mean_mhz=weighted_mean_mhz,
        min_interval_mhz=min(interval_mhz),
        max_interval_mhz=max(interval_mhz),
    )


def summarize_capture(
    capture: RealtimeCapture, simd_id: Optional[int] = None
) -> CaptureSummary:
    """Summarize all SE clock series in one SIMD-select ATT capture."""
    realtime_origin = min(
        samples[0][1] for samples in capture.samples_by_se.values()
    )
    series = tuple(
        compute_interval_summary(
            se_name,
            capture.samples_by_se[se_name],
            capture.reference_hz,
            realtime_origin,
        )
        for se_name in sorted(capture.samples_by_se, key=se_sort_key)
    )
    total_shader_ticks = sum(item.total_shader_ticks for item in series)
    total_realtime_ticks = sum(item.total_realtime_ticks for item in series)
    weighted_mean_mhz = (
        capture.reference_hz
        * total_shader_ticks
        / total_realtime_ticks
        / 1.0e6
    )
    return CaptureSummary(
        simd_id=simd_id,
        capture=capture,
        series=series,
        valid_intervals=sum(item.valid_intervals for item in series),
        dropped_nonpositive_realtime=sum(
            item.dropped_nonpositive_realtime for item in series
        ),
        total_shader_ticks=total_shader_ticks,
        total_realtime_ticks=total_realtime_ticks,
        weighted_mean_mhz=weighted_mean_mhz,
    )


def tick_weighted_mean(series: Sequence[SeriesSummary]) -> float:
    total_ticks = sum(item.total_realtime_ticks for item in series)
    if total_ticks <= 0:
        raise ValueError("Cannot compute a mean without positive REALTIME ticks")
    return (
        sum(
            item.weighted_mean_mhz * item.total_realtime_ticks
            for item in series
        )
        / total_ticks
    )


def compute_realtime_cycle_spans(
    capture: RealtimeCapture,
) -> tuple[PerSECycleSpan, ...]:
    """Return raw shader-cycle spans from each realtime.json SE series."""
    spans: list[PerSECycleSpan] = []
    for se_name in sorted(capture.samples_by_se, key=se_sort_key):
        samples = capture.samples_by_se[se_name]
        if len(samples) < 2:
            raise ValueError(
                f"{capture.path}: {se_name} realtime cycle span is not "
                f"computable from {len(samples)} sample(s)"
            )
        shader_timestamps = [sample[0] for sample in samples]
        spans.append(
            PerSECycleSpan(
                se_name=se_name,
                sample_count=len(samples),
                cycle_span=max(shader_timestamps) - min(shader_timestamps),
            )
        )
    return tuple(spans)


def compute_realtime_event_span(
    capture: RealtimeCapture,
) -> tuple[float, float, float]:
    """Return earliest, latest, and span across all REALTIME samples."""
    realtime_ticks = tuple(
        sample[1]
        for samples in capture.samples_by_se.values()
        for sample in samples
    )
    earliest = min(realtime_ticks)
    latest = max(realtime_ticks)
    span = latest - earliest
    if span <= 0:
        raise ValueError(
            f"{capture.path}: all-SE REALTIME event span must be positive, "
            f"got {span:g}"
        )
    return earliest, latest, span


def compute_occupancy_cycle_spans(
    capture: OccupancyCapture,
) -> tuple[PerSECycleSpan, ...]:
    """Return raw shader-cycle spans from each occupancy.json SE event series."""
    spans: list[PerSECycleSpan] = []
    for se_name in sorted(capture.events_by_se, key=se_sort_key):
        events = capture.events_by_se[se_name]
        if len(events) < 2:
            raise ValueError(
                f"{capture.path}: {se_name} occupancy cycle span is not "
                f"computable from {len(events)} event(s)"
            )
        timestamps = [event.shader_timestamp for event in events]
        spans.append(
            PerSECycleSpan(
                se_name=se_name,
                sample_count=len(events),
                cycle_span=float(max(timestamps) - min(timestamps)),
            )
        )
    return tuple(spans)


def decode_packed_sa_wgp(packed_sa_wgp: int) -> tuple[int, int]:
    return ((packed_sa_wgp >> 7) & 1, packed_sa_wgp & 0x7F)


def _event_sort_key(event: OccupancyEvent) -> tuple[int, int, int, int, int]:
    # At an equal timestamp, deallocation (start=0) precedes allocation.
    return (
        event.shader_timestamp,
        event.start,
        event.packed_sa_wgp,
        event.hw_simd,
        event.wave_slot,
    )


def _wave_key(
    event: OccupancyEvent,
) -> tuple[int, int, int, int]:
    return (
        event.packed_sa_wgp,
        event.hw_simd,
        event.wave_slot,
        event.kernel_pc_index,
    )


def _wave_key_text(
    se_name: str, key: tuple[int, int, int, int]
) -> str:
    packed_sa_wgp, hw_simd, wave_slot, kernel_pc_index = key
    sa_id, wgp_id = decode_packed_sa_wgp(packed_sa_wgp)
    return (
        f"{se_name}/packed={packed_sa_wgp}(SA{sa_id}/WGP{wgp_id})/"
        f"hwSIMD{hw_simd}/slot{wave_slot}/kernel-label={kernel_pc_index}"
    )


def pair_wave_lifetimes(
    capture: OccupancyCapture,
) -> tuple[WaveLifetime, ...]:
    """Pair alloc/dealloc events without crossing kernel-PC labels."""
    lifetimes: list[WaveLifetime] = []
    for se_name in sorted(capture.events_by_se, key=se_sort_key):
        active: dict[tuple[int, int, int, int], int] = {}
        for event in sorted(capture.events_by_se[se_name], key=_event_sort_key):
            key = _wave_key(event)
            if event.start == 1:
                if key in active:
                    raise ValueError(
                        f"{capture.path}: double alloc at "
                        f"{event.shader_timestamp} for "
                        f"{_wave_key_text(se_name, key)}"
                    )
                active[key] = event.shader_timestamp
                continue

            if key not in active:
                raise ValueError(
                    f"{capture.path}: unmatched dealloc at "
                    f"{event.shader_timestamp} for "
                    f"{_wave_key_text(se_name, key)}"
                )
            start_timestamp = active.pop(key)
            duration = event.shader_timestamp - start_timestamp
            if duration <= 0:
                raise ValueError(
                    f"{capture.path}: non-positive wave lifetime ({duration}) "
                    f"for {_wave_key_text(se_name, key)}"
                )
            packed_sa_wgp, hw_simd, wave_slot, kernel_pc_index = key
            lifetimes.append(
                WaveLifetime(
                    se_name=se_name,
                    packed_sa_wgp=packed_sa_wgp,
                    hw_simd=hw_simd,
                    wave_slot=wave_slot,
                    kernel_pc_index=kernel_pc_index,
                    start_timestamp=start_timestamp,
                    end_timestamp=event.shader_timestamp,
                )
            )

        if active:
            coordinates = sorted(
                _wave_key_text(se_name, key) for key in active
            )
            preview = ", ".join(coordinates[:5])
            suffix = (
                f", ... ({len(coordinates)} total)"
                if len(coordinates) > 5
                else ""
            )
            raise ValueError(
                f"{capture.path}: unfinished wave allocation(s): "
                f"{preview}{suffix}"
            )
    return tuple(lifetimes)


def summarize_wave_reuse(
    lifetimes: Sequence[WaveLifetime],
) -> tuple[WaveReuseSummary, ...]:
    counts = Counter(
        (
            item.se_name,
            item.packed_sa_wgp,
            item.hw_simd,
            item.wave_slot,
            item.kernel_pc_index,
        )
        for item in lifetimes
    )
    return tuple(
        WaveReuseSummary(
            se_name=key[0],
            packed_sa_wgp=key[1],
            hw_simd=key[2],
            wave_slot=key[3],
            kernel_pc_index=key[4],
            wave_count=count,
        )
        for key, count in sorted(
            counts.items(),
            key=lambda item: (
                se_sort_key(item[0][0]),
                item[0][1],
                item[0][2],
                item[0][3],
                item[0][4],
            ),
        )
    )


def compute_slot_usage(
    capture: OccupancyCapture,
) -> tuple[SlotUsageSummary, ...]:
    grouped: dict[
        tuple[str, int, int, int], list[OccupancyEvent]
    ] = defaultdict(list)
    for se_name, events in capture.events_by_se.items():
        for event in events:
            grouped[
                (
                    se_name,
                    event.packed_sa_wgp,
                    event.hw_simd,
                    event.kernel_pc_index,
                )
            ].append(event)

    results: list[SlotUsageSummary] = []
    for group_key, events in sorted(
        grouped.items(),
        key=lambda item: (
            se_sort_key(item[0][0]),
            item[0][1],
            item[0][2],
            item[0][3],
        ),
    ):
        active_slots: set[int] = set()
        distinct_slots: set[int] = set()
        max_concurrent = 0
        for event in sorted(events, key=_event_sort_key):
            if event.start == 0:
                if event.wave_slot not in active_slots:
                    raise ValueError(
                        f"{capture.path}: negative active-slot count for "
                        f"{group_key}, slot {event.wave_slot}"
                    )
                active_slots.remove(event.wave_slot)
            else:
                if event.wave_slot in active_slots:
                    raise ValueError(
                        f"{capture.path}: slot already active for "
                        f"{group_key}, slot {event.wave_slot}"
                    )
                active_slots.add(event.wave_slot)
                distinct_slots.add(event.wave_slot)
                max_concurrent = max(max_concurrent, len(active_slots))
        if active_slots:
            raise ValueError(
                f"{capture.path}: active slots remain after sweep for "
                f"{group_key}: {sorted(active_slots)}"
            )
        se_name, packed_sa_wgp, hw_simd, kernel_pc_index = group_key
        results.append(
            SlotUsageSummary(
                se_name=se_name,
                packed_sa_wgp=packed_sa_wgp,
                hw_simd=hw_simd,
                kernel_pc_index=kernel_pc_index,
                distinct_slot_ids=tuple(sorted(distinct_slots)),
                max_concurrent_active_slots=max_concurrent,
            )
        )
    return tuple(results)


def compute_wgp_episodes(
    capture: OccupancyCapture,
) -> tuple[WGPEpisode, ...]:
    """Reconstruct physical-WGP episodes from aggregate active-wave count."""
    grouped: dict[
        tuple[str, int, int], list[OccupancyEvent]
    ] = defaultdict(list)
    for se_name, events in capture.events_by_se.items():
        for event in events:
            grouped[
                (se_name, event.packed_sa_wgp, event.kernel_pc_index)
            ].append(event)

    episodes: list[WGPEpisode] = []
    for group_key, events in sorted(
        grouped.items(),
        key=lambda item: (
            se_sort_key(item[0][0]),
            item[0][1],
            item[0][2],
        ),
    ):
        active_wave_count = 0
        episode_start: Optional[int] = None
        for event in sorted(events, key=_event_sort_key):
            if event.start == 0:
                active_wave_count -= 1
                if active_wave_count < 0:
                    raise ValueError(
                        f"{capture.path}: negative active-wave count at "
                        f"{event.shader_timestamp} for {group_key}"
                    )
                if active_wave_count == 0:
                    if episode_start is None:
                        raise ValueError(
                            f"{capture.path}: WGP episode has no start for "
                            f"{group_key}"
                        )
                    duration = event.shader_timestamp - episode_start
                    if duration <= 0:
                        raise ValueError(
                            f"{capture.path}: non-positive WGP episode "
                            f"duration ({duration}) for {group_key}"
                        )
                    episodes.append(
                        WGPEpisode(
                            se_name=group_key[0],
                            packed_sa_wgp=group_key[1],
                            kernel_pc_index=group_key[2],
                            start_timestamp=episode_start,
                            end_timestamp=event.shader_timestamp,
                        )
                    )
                    episode_start = None
            else:
                if active_wave_count == 0:
                    episode_start = event.shader_timestamp
                active_wave_count += 1
        if active_wave_count != 0 or episode_start is not None:
            raise ValueError(
                f"{capture.path}: unfinished WGP active episode for "
                f"{group_key}; active waves={active_wave_count}"
            )
    return tuple(episodes)


def compute_wgp_usage(
    lifetimes: Sequence[WaveLifetime],
) -> tuple[WGPUsageSummary, ...]:
    grouped: dict[tuple[str, int], set[int]] = defaultdict(set)
    for item in lifetimes:
        grouped[(item.se_name, item.kernel_pc_index)].add(
            item.packed_sa_wgp
        )
    return tuple(
        WGPUsageSummary(
            se_name=key[0],
            kernel_pc_index=key[1],
            packed_sa_wgp_ids=tuple(sorted(packed_ids)),
        )
        for key, packed_ids in sorted(
            grouped.items(),
            key=lambda item: (se_sort_key(item[0][0]), item[0][1]),
        )
    )


def summarize_physical_wgps(
    episodes: Sequence[WGPEpisode],
) -> tuple[PhysicalWGPSummary, ...]:
    """Merge all active episodes for each physical WGP and kernel-PC label."""
    grouped: dict[tuple[str, int, int], list[WGPEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[
            (
                episode.se_name,
                episode.packed_sa_wgp,
                episode.kernel_pc_index,
            )
        ].append(episode)

    results: list[PhysicalWGPSummary] = []
    for group_key, group_episodes in sorted(
        grouped.items(),
        key=lambda item: (
            se_sort_key(item[0][0]),
            item[0][2],
            item[0][1],
        ),
    ):
        starts = [episode.start_timestamp for episode in group_episodes]
        ends = [episode.end_timestamp for episode in group_episodes]
        active_duration = sum(episode.duration for episode in group_episodes)
        start_timestamp = min(starts)
        end_timestamp = max(ends)
        envelope_duration = end_timestamp - start_timestamp
        idle_gap_duration = envelope_duration - active_duration
        if envelope_duration <= 0 or active_duration <= 0:
            raise ValueError(
                "Physical-WGP summary has a non-positive duration for "
                f"{group_key}"
            )
        if idle_gap_duration < 0:
            raise ValueError(
                "Physical-WGP episodes overlap while computing idle gaps for "
                f"{group_key}"
            )
        results.append(
            PhysicalWGPSummary(
                se_name=group_key[0],
                packed_sa_wgp=group_key[1],
                kernel_pc_index=group_key[2],
                episode_count=len(group_episodes),
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                envelope_duration=envelope_duration,
                active_duration=active_duration,
                idle_gap_duration=idle_gap_duration,
            )
        )
    return tuple(results)


def compute_completion_imbalance(
    capture: OccupancyCapture,
    physical_wgps: Sequence[PhysicalWGPSummary],
) -> tuple[WGPCompletionSummary, ...]:
    """Compute completion imbalance from unique physical-WGP summaries."""
    grouped: dict[tuple[str, int], list[PhysicalWGPSummary]] = defaultdict(list)
    for physical_wgp in physical_wgps:
        grouped[
            (physical_wgp.se_name, physical_wgp.kernel_pc_index)
        ].append(physical_wgp)

    represented_se_names = {key[0] for key in grouped}
    for se_name in sorted(capture.events_by_se, key=se_sort_key):
        if se_name not in represented_se_names:
            raise ValueError(
                f"{capture.path}: {se_name} has no complete physical-WGP "
                "summary"
            )

    results: list[WGPCompletionSummary] = []
    for group_key, group_wgps in sorted(
        grouped.items(),
        key=lambda item: (se_sort_key(item[0][0]), item[0][1]),
    ):
        envelope_durations = [
            physical_wgp.envelope_duration for physical_wgp in group_wgps
        ]
        active_durations = [
            physical_wgp.active_duration for physical_wgp in group_wgps
        ]
        idle_gap_durations = [
            physical_wgp.idle_gap_duration for physical_wgp in group_wgps
        ]
        if any(duration <= 0 for duration in envelope_durations):
            raise ValueError(
                f"{capture.path}: {group_key} has a non-positive physical-WGP "
                "envelope duration"
            )
        final_ends = [
            physical_wgp.end_timestamp for physical_wgp in group_wgps
        ]
        envelope_median = float(median(envelope_durations))
        if len(group_wgps) < 2:
            final_end_span: Optional[int] = None
            completion_imbalance: Optional[float] = None
        else:
            final_end_span = max(final_ends) - min(final_ends)
            completion_imbalance = final_end_span / envelope_median
        episode_counts = [
            physical_wgp.episode_count for physical_wgp in group_wgps
        ]
        results.append(
            WGPCompletionSummary(
                se_name=group_key[0],
                kernel_pc_index=group_key[1],
                wgp_count=len(group_wgps),
                episode_count_min=min(episode_counts),
                episode_count_max=max(episode_counts),
                envelope_min=min(envelope_durations),
                envelope_median=envelope_median,
                envelope_max=max(envelope_durations),
                active_median=float(median(active_durations)),
                idle_gap_median=float(median(idle_gap_durations)),
                final_end_min=min(final_ends),
                final_end_max=max(final_ends),
                final_end_span=final_end_span,
                completion_imbalance=completion_imbalance,
            )
        )
    return tuple(results)


def summarize_occupancy(capture: OccupancyCapture) -> OccupancySummary:
    wave_lifetimes = pair_wave_lifetimes(capture)
    wgp_episodes = compute_wgp_episodes(capture)
    physical_wgp_summaries = summarize_physical_wgps(wgp_episodes)
    return OccupancySummary(
        capture=capture,
        cycle_spans=compute_occupancy_cycle_spans(capture),
        wave_lifetimes=wave_lifetimes,
        wave_reuse=summarize_wave_reuse(wave_lifetimes),
        slot_usage=compute_slot_usage(capture),
        wgp_usage=compute_wgp_usage(wave_lifetimes),
        wgp_episodes=wgp_episodes,
        physical_wgp_summaries=physical_wgp_summaries,
        completion_by_group=compute_completion_imbalance(
            capture, physical_wgp_summaries
        ),
    )


def _candidate_list(att_root: Path, paths: Sequence[Path]) -> str:
    if not paths:
        return "  - (none)"
    return "\n".join(f"  - {path.relative_to(att_root)}" for path in paths)


def _infer_unwrapped_simd_id(att_root: Path) -> int:
    """Read att_simd_select from packaged input_*.yaml; default is SIMD 3."""
    pattern = re.compile(
        r"^\s*att_simd_select:\s*[\"']?([0-3])[\"']?\s*$",
        re.MULTILINE,
    )
    found: set[int] = set()
    for path in sorted(att_root.glob("input_*.yaml")):
        if not path.is_file():
            continue
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            found.add(int(match.group(1)))
    if len(found) == 1:
        return next(iter(found))
    if len(found) > 1:
        raise ValueError(
            "Unwrapped kernel capture has conflicting att_simd_select "
            f"values: {sorted(found)}"
        )
    return UNWRAPPED_DEFAULT_SIMD_ID


def _find_paired_capture_jsons(
    root: Path,
    search_root: Path,
    scan_root: Path,
    rule: str,
    label: str,
) -> Optional[tuple[Path, Path]]:
    """Return realtime/occupancy paths, or None when this capture is absent."""
    candidates = (
        sorted(
            path
            for path in search_root.rglob("realtime.json")
            if path.is_file()
        )
        if search_root.is_dir()
        else []
    )
    alternate_candidates = (
        sorted(
            path
            for path in scan_root.rglob("realtime.json")
            if path.is_file() and path not in candidates
        )
        if scan_root.is_dir()
        else []
    )
    if not candidates:
        if alternate_candidates:
            raise ValueError(
                f"{label}: expected exactly one realtime.json under {rule}, "
                "found none; files outside kernel/rpf_v3 are intentionally "
                "ignored:\n"
                + _candidate_list(root, alternate_candidates)
            )
        return None
    if len(candidates) != 1:
        raise ValueError(
            f"{label}: multiple realtime.json candidates under {rule}; "
            "refusing to choose silently:\n"
            + _candidate_list(root, candidates)
        )

    realtime_path = candidates[0]
    occupancy_path = realtime_path.parent / "occupancy.json"
    if occupancy_path.is_file():
        return realtime_path, occupancy_path

    occupancy_candidates = (
        sorted(
            path
            for path in search_root.rglob("occupancy.json")
            if path.is_file()
        )
        if search_root.is_dir()
        else []
    )
    raise ValueError(
        f"{label}: selected realtime.json requires occupancy.json "
        f"in the same UI directory, but this file is missing:\n"
        f"  - {occupancy_path.relative_to(root)}\n"
        "occupancy.json candidates under kernel/rpf_v3:\n"
        + _candidate_list(root, occupancy_candidates)
    )


def discover_capture_files(
    att_root: Path,
) -> dict[int, SelectedCaptureFiles]:
    """Discover existing SIMD-select captures and strictly pair both JSONs."""
    root = att_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"ATT root does not exist or is not a directory: {root}")

    thread_trace = root / "thread_trace"
    if not thread_trace.is_dir():
        raise ValueError(f"Missing ATT thread_trace directory: {thread_trace}")

    simd_directories: dict[int, Path] = {}
    invalid_simd_entries: list[Path] = []
    for entry in thread_trace.iterdir():
        match = re.fullmatch(r"simd(\d+)", entry.name)
        if match is None:
            continue
        simd_id = int(match.group(1))
        if (
            simd_id not in SIMD_IDS
            or entry.name != f"simd{simd_id}"
            or not entry.is_dir()
        ):
            invalid_simd_entries.append(entry)
            continue
        simd_directories[simd_id] = entry
    if invalid_simd_entries:
        raise ValueError(
            "Unsupported or non-directory SIMD capture entries "
            "(expected directories simd0..simd3):\n"
            + _candidate_list(root, sorted(invalid_simd_entries))
        )

    discovered: dict[int, SelectedCaptureFiles] = {}
    for simd_id, simd_dir in sorted(simd_directories.items()):
        paired = _find_paired_capture_jsons(
            root,
            simd_dir / "kernel" / "rpf_v3",
            simd_dir,
            f"{simd_dir.relative_to(root)}/kernel/rpf_v3/**/realtime.json",
            f"SIMD{simd_id}",
        )
        if paired is None:
            continue
        realtime_path, occupancy_path = paired
        discovered[simd_id] = SelectedCaptureFiles(
            simd_id=simd_id,
            realtime_path=realtime_path,
            occupancy_path=occupancy_path,
        )

    if not discovered:
        kernel_dir = thread_trace / "kernel"
        paired = _find_paired_capture_jsons(
            root,
            kernel_dir / "rpf_v3",
            kernel_dir,
            f"{kernel_dir.relative_to(root)}/rpf_v3/**/realtime.json",
            "unwrapped kernel capture",
        )
        if paired is not None:
            simd_id = _infer_unwrapped_simd_id(root)
            realtime_path, occupancy_path = paired
            discovered[simd_id] = SelectedCaptureFiles(
                simd_id=simd_id,
                realtime_path=realtime_path,
                occupancy_path=occupancy_path,
            )

    if not discovered:
        raise ValueError(
            f"No usable ATT capture found under {thread_trace}; expected at "
            "least one complete capture in thread_trace/simd0..simd3/"
            "kernel/rpf_v3 or thread_trace/kernel/rpf_v3"
        )

    return discovered


def build_directory_output_paths(
    att_root: Path,
    output_dir: Optional[Path] = None,
    simd_ids: Sequence[int] = SIMD_IDS,
) -> dict[int, Path]:
    root = att_root.expanduser().resolve()
    destination = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else root.parent
    )
    return {
        simd_id: destination / f"{root.name}.simd{simd_id}.png"
        for simd_id in simd_ids
    }


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def validate_directory_output_paths(
    att_root: Path,
    output_paths: Mapping[int, Path],
    source_paths: Sequence[Path],
) -> None:
    """Permit repeat overwrites only for discovered derived PNG destinations."""
    root = att_root.expanduser().resolve()
    resolved_sources = {path.resolve() for path in source_paths}
    resolved_outputs = [path.resolve() for path in output_paths.values()]
    if not output_paths:
        raise ValueError("No derived SIMD output paths were provided")
    if len(set(resolved_outputs)) != len(output_paths):
        raise ValueError("Derived SIMD output paths are not unique")

    for simd_id in sorted(output_paths):
        output = output_paths[simd_id]
        resolved = output.resolve()
        expected_name = f"{root.name}.simd{simd_id}.png"
        if output.name != expected_name:
            raise ValueError(
                f"Unsafe derived output name {output.name!r}; "
                f"expected {expected_name!r}"
            )
        if resolved in resolved_sources:
            raise ValueError(f"Output would overwrite trace source file: {output}")
        if _path_is_within(resolved, root):
            raise ValueError(
                f"Directory-mode output must stay outside the ATT trace root: "
                f"{output}"
            )
        if output.exists() and not output.is_file():
            raise ValueError(
                f"Cannot overwrite non-file output destination: {output}"
            )


def shared_y_limits(
    captures: Sequence[CaptureSummary],
) -> tuple[float, float]:
    maximum = max(
        value
        for capture in captures
        for series in capture.series
        for value in series.interval_mhz
    )
    return (0.0, max(1.0, maximum * 1.05))


def _plot_series(ax: object, series: Sequence[SeriesSummary]) -> None:
    for index, item in enumerate(series):
        ax.plot(
            item.elapsed_us,
            item.interval_mhz,
            color=COLORS[index % len(COLORS)],
            linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            marker=MARKERS[index % len(MARKERS)],
            markevery=20,
            markersize=3,
            linewidth=1.35,
            alpha=0.85,
            label=(
                f"{item.se_name} "
                f"(time-weighted mean {item.weighted_mean_mhz:.1f} MHz)"
            ),
        )


def plot_simd_capture(
    summary: CaptureSummary,
    att_basename: str,
    output: Path,
    y_limits: tuple[float, float],
) -> None:
    """Plot the four SE series observed in one SIMD-select ATT run."""
    if summary.simd_id is None:
        raise ValueError("SIMD id is required for directory-mode plotting")

    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    try:
        _plot_series(ax, summary.series)
        ax.set_title(
            f"{att_basename}\n"
            f"SIMD{summary.simd_id}-select ATT capture x SE clock series\n"
            "per-SE realtime-derived Fgfx\n"
            f"REALTIME reference: "
            f"{summary.capture.reference_hz / 1.0e6:g} MHz "
            f"({summary.capture.reference_source})"
        )
        ax.set_xlabel("Elapsed REALTIME (us)")
        ax.set_ylabel("Interval-average Fgfx (MHz)")
        ax.set_ylim(*y_limits)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.savefig(output, dpi=180)
    finally:
        plt.close(fig)


def plot_legacy_capture(summary: CaptureSummary, output: Path) -> None:
    """Preserve the original single-input plot layout and naming behavior."""
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    try:
        _plot_series(ax, summary.series)
        ax.set_title(
            "ATT-Derived GFX Clock Frequency by Shader Engine\n"
            f"REALTIME reference: "
            f"{summary.capture.reference_hz / 1.0e6:g} MHz; "
            f"{summary.capture.reference_source}; "
            "one value per adjacent sample pair"
        )
        ax.set_xlabel("Elapsed REALTIME (us)")
        ax.set_ylabel("Interval-average Fgfx (MHz)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
    finally:
        plt.close(fig)


def _metadata_text(capture: RealtimeCapture) -> str:
    if capture.metadata_hz is None:
        return "missing/invalid"
    return f"{capture.metadata_hz / 1.0e6:.3f} MHz"


def _format_cycle_value(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.6g}"


def _tie_suffix(tie_count: int) -> str:
    return f" (+{tie_count - 1} tied)" if tie_count > 1 else ""


def _wave_reuse_coordinate(item: WaveReuseSummary) -> str:
    sa_id, wgp_id = decode_packed_sa_wgp(item.packed_sa_wgp)
    return (
        f"{item.se_name}/packed={item.packed_sa_wgp}"
        f"(SA{sa_id}/WGP{wgp_id})/hwSIMD{item.hw_simd}/"
        f"slot{item.wave_slot}/kernel-label={item.kernel_pc_index}"
    )


def _slot_usage_coordinate(item: SlotUsageSummary) -> str:
    sa_id, wgp_id = decode_packed_sa_wgp(item.packed_sa_wgp)
    return (
        f"{item.se_name}/packed={item.packed_sa_wgp}"
        f"(SA{sa_id}/WGP{wgp_id})/hwSIMD{item.hw_simd}/"
        f"kernel-label={item.kernel_pc_index}"
    )


def _format_local_wgp_ranges(packed_ids: Sequence[int]) -> str:
    by_sa: dict[int, list[int]] = defaultdict(list)
    for packed_id in packed_ids:
        sa_id, wgp_id = decode_packed_sa_wgp(packed_id)
        by_sa[sa_id].append(wgp_id)

    groups: list[str] = []
    for sa_id in sorted(by_sa):
        local_ids = sorted(set(by_sa[sa_id]))
        ranges: list[str] = []
        range_start = local_ids[0]
        range_end = local_ids[0]
        for value in local_ids[1:]:
            if value == range_end + 1:
                range_end = value
                continue
            ranges.append(
                str(range_start)
                if range_start == range_end
                else f"{range_start}-{range_end}"
            )
            range_start = range_end = value
        ranges.append(
            str(range_start)
            if range_start == range_end
            else f"{range_start}-{range_end}"
        )
        groups.append(f"SA{sa_id}/WGP{','.join(ranges)}")
    return "; ".join(groups)


def _ui_dispatch_id(path: Path) -> Optional[int]:
    match = re.search(r"(?:^|_)dispatch_(\d+)(?:_|$)", path.parent.name)
    return int(match.group(1)) if match is not None else None


def format_directory_summary(
    summaries: Mapping[int, DirectoryCaptureSummary],
    output_paths: Optional[Mapping[int, Path]] = None,
) -> str:
    """Format every directory-mode result from one consolidated entry point."""
    if not summaries:
        raise ValueError("No directory capture summaries to print")
    simd_ids = sorted(summaries)
    if output_paths is not None and set(output_paths) != set(summaries):
        raise ValueError("Output paths do not match discovered SIMD captures")

    lines: list[str] = []
    frequency_summary_lines: list[str] = []
    observed_cycle_maxima: list[ObservedCycleMaximum] = []
    lines.append(
        "Selected SIMD-select capture files "
        "(strict kernel/rpf_v3 realtime + same-UI occupancy pairing):"
    )
    for simd_id in simd_ids:
        item = summaries[simd_id]
        lines.append(f"  SIMD{simd_id}-select capture:")
        lines.append(f"    realtime:  {item.files.realtime_path.resolve()}")
        lines.append(f"    occupancy: {item.files.occupancy_path.resolve()}")
        ui_dispatch_id = _ui_dispatch_id(item.files.realtime_path)
        if ui_dispatch_id is not None:
            lines.append(
                f"    UI dispatch ID (from directory path): {ui_dispatch_id}"
            )
        used_kernel_pc_indices = sorted(
            {
                event.kernel_pc_index
                for events in item.occupancy.capture.events_by_se.values()
                for event in events
            }
        )
        kernel_pc_descriptions = []
        for kernel_pc_index in used_kernel_pc_indices:
            label = item.occupancy.capture.kernel_pc_labels.get(kernel_pc_index)
            kernel_pc_descriptions.append(
                f"{kernel_pc_index}={label}"
                if label
                else str(kernel_pc_index)
            )
        lines.append(
            "    occupancy kernel-PC label index(es): "
            + ", ".join(kernel_pc_descriptions)
        )

    lines.extend(
        [
            "",
            "Clock-domain interpretation:",
            "  SIMDn names an ATT SIMD-select capture, not an independent "
            "hardware clock domain.",
            "  SIMD-select captures are independent; maxima "
            "below are observations across captures, not simultaneous GPU state.",
            "  A WGP contains four SIMD32s. Shader-cycle counters are not "
            "synchronized across SIMDs.",
            "  Every raw shader span below is a delta within its own capture x "
            "SE series.",
            "  Cross-SE maxima compare numeric span only; occupancy spans are "
            "not wall durations.",
            "  REALTIME is the fixed-frequency counter that permits wall-time "
            "alignment.",
            "",
            "Reference clocks by SIMD-select ATT capture:",
        ]
    )
    for simd_id in simd_ids:
        capture = summaries[simd_id].realtime.capture
        lines.append(
            f"  SIMD{simd_id}-select: "
            f"{capture.reference_hz / 1.0e6:.3f} MHz "
            f"({capture.reference_source}; "
            f"metadata={_metadata_text(capture)})"
        )
    distinct_references = {
        item.realtime.capture.reference_hz for item in summaries.values()
    }
    if len(distinct_references) > 1:
        lines.append(
            "  Note: references differ; each series uses its own reference and "
            "pooled frequency uses raw REALTIME-tick weights."
        )

    lines.extend(
        [
            "",
            "SIMD-select capture x SE frequency series "
            "(delta_realtime<=0 pairs are filtered):",
        f"{'Capture':<12} {'SE':<4} {'Samples':>8} {'Valid':>7} "
        f"{'Dropped':>8} {'Ref MHz':>10} {'Mean MHz':>11} "
            f"{'Min MHz':>10} {'Max MHz':>10}",
            "-" * 99,
        ]
    )
    all_series: list[tuple[int, SeriesSummary]] = []
    for simd_id in simd_ids:
        for item in summaries[simd_id].realtime.series:
            all_series.append((simd_id, item))
            lines.append(
                f"{f'SIMD{simd_id}-select':<12} {item.se_name:<4} "
                f"{item.sample_count:>8} {item.valid_intervals:>7} "
                f"{item.dropped_nonpositive_realtime:>8} "
                f"{item.reference_hz / 1.0e6:>10.3f} "
                f"{item.weighted_mean_mhz:>11.3f} "
                f"{item.min_interval_mhz:>10.3f} "
                f"{item.max_interval_mhz:>10.3f}"
            )

    frequency_summary_lines.extend(
        [
            "Per-SIMD-select capture means "
            "(REALTIME-tick weighted across SE series):",
        ]
    )
    for simd_id in simd_ids:
        summary = summaries[simd_id].realtime
        frequency_summary_lines.append(
            f"  SIMD{simd_id}-select: "
            f"{summary.weighted_mean_mhz:.3f} MHz across "
            f"{len(summary.series)} SE series; {summary.valid_intervals} valid "
            f"intervals; {summary.dropped_nonpositive_realtime} dropped"
        )

    series_values = [item for _, item in all_series]
    overall_mean_mhz = tick_weighted_mean(series_values)
    minimum_simd, minimum = min(
        all_series, key=lambda entry: entry[1].weighted_mean_mhz
    )
    maximum_simd, maximum = max(
        all_series, key=lambda entry: entry[1].weighted_mean_mhz
    )
    spread_mhz = maximum.weighted_mean_mhz - minimum.weighted_mean_mhz
    frequency_summary_lines.extend(
        [
            "",
            f"Combined mean across {len(all_series)} independent SIMD-select "
            f"capture x SE series (REALTIME-tick weighted): "
            f"{overall_mean_mhz:.3f} MHz",
            f"Spread of the {len(all_series)} series means: "
            f"{spread_mhz:.3f} MHz "
            f"(min SIMD{minimum_simd}-select {minimum.se_name} "
            f"{minimum.weighted_mean_mhz:.3f}; "
            f"max SIMD{maximum_simd}-select {maximum.se_name} "
            f"{maximum.weighted_mean_mhz:.3f})",
        ]
    )

    cycle_sources = (
        (
            "occupancy.json event shader_timestamp",
            "Events",
            {
                simd_id: summaries[simd_id].occupancy.cycle_spans
                for simd_id in simd_ids
            },
        ),
        (
            "realtime.json gfx_clock",
            "Clock samples",
            {
                simd_id: summaries[simd_id].realtime_cycle_spans
                for simd_id in simd_ids
            },
        ),
    )
    for source_name, count_label, spans_by_simd in cycle_sources:
        lines.extend(
            [
                "",
                f"Per-SE raw shader-cycle spans from {source_name}:",
                f"{'Capture':<12} {'SE':<4} {count_label:>14} "
                f"{'per_se_cycle':>16}",
                "-" * 50,
            ]
        )
        all_spans: list[tuple[int, PerSECycleSpan]] = []
        for simd_id in simd_ids:
            for span in spans_by_simd[simd_id]:
                all_spans.append((simd_id, span))
                lines.append(
                    f"{f'SIMD{simd_id}-select':<12} {span.se_name:<4} "
                    f"{span.sample_count:>14} "
                    f"{_format_cycle_value(span.cycle_span):>16}"
                )
        lines.append("Per-capture maximum per_se_cycle:")
        for simd_id in simd_ids:
            capture_spans = spans_by_simd[simd_id]
            maximum_span = max(item.cycle_span for item in capture_spans)
            tied_se_names = [
                item.se_name
                for item in capture_spans
                if item.cycle_span == maximum_span
            ]
            lines.append(
                f"  SIMD{simd_id}-select: "
                f"{_format_cycle_value(maximum_span)} at "
                f"{', '.join(tied_se_names)}"
            )
        observed_maximum = max(item.cycle_span for _, item in all_spans)
        observed_coordinates = tuple(
            (simd_id, item.se_name)
            for simd_id, item in all_spans
            if item.cycle_span == observed_maximum
        )
        observed_cycle_maxima.append(
            ObservedCycleMaximum(
                source_name=source_name,
                cycle_span=observed_maximum,
                coordinates=observed_coordinates,
            )
        )

    lines.extend(
        [
            "",
            "Occupancy wave lifetimes and sequential reuse per physical slot:",
            "  Key = capture/SE/packed-SA-WGP/hwSIMD/slot/kernel-PC-label; "
            "kernel-PC labels are never merged.",
            "  Histogram: wave-lifetimes-per-slot -> physical-slot-key count.",
            "  This is sequential slot reuse, not concurrent occupancy; "
            "concurrent slot peaks are reported separately below.",
            "  A WGP has four hwSIMDs; this histogram remains keyed by each "
            "observed physical slot and is not divided by four.",
        ]
    )
    all_reuse: list[tuple[int, WaveReuseSummary]] = []
    for simd_id in simd_ids:
        occupancy = summaries[simd_id].occupancy
        if not occupancy.wave_reuse:
            raise ValueError(f"SIMD{simd_id} has no wave reuse summaries")
        maximum_reuse = max(item.wave_count for item in occupancy.wave_reuse)
        tied = [
            item
            for item in occupancy.wave_reuse
            if item.wave_count == maximum_reuse
        ]
        representative = tied[0]
        histogram = Counter(
            item.wave_count for item in occupancy.wave_reuse
        )
        histogram_text = ", ".join(
            f"{reuse}:{count}" for reuse, count in sorted(histogram.items())
        )
        lines.append(
            f"  SIMD{simd_id}-select: "
            f"wave lifetimes={len(occupancy.wave_lifetimes)}, "
            f"observed physical slot keys={len(occupancy.wave_reuse)}, "
            f"max sequential wave lifetimes per slot="
            f"{maximum_reuse} at {_wave_reuse_coordinate(representative)}"
            f"{_tie_suffix(len(tied))}; histogram={{{histogram_text}}}"
        )
        all_reuse.extend(
            (simd_id, item) for item in occupancy.wave_reuse
        )
    observed_reuse_max = max(item.wave_count for _, item in all_reuse)
    observed_reuse_ties = [
        (simd_id, item)
        for simd_id, item in all_reuse
        if item.wave_count == observed_reuse_max
    ]
    observed_reuse_simd, observed_reuse_item = observed_reuse_ties[0]
    wave_reuse_summary_line = (
        "Max observed across independent SIMD-select captures: "
        f"{observed_reuse_max} sequential wave lifetimes per slot at "
        f"SIMD{observed_reuse_simd}-select/"
        f"{_wave_reuse_coordinate(observed_reuse_item)}"
        f"{_tie_suffix(len(observed_reuse_ties))}"
    )

    lines.extend(
        [
            "",
            "Occupancy slot-count maxima per "
            "(capture, SE, packed-SA-WGP, hwSIMD, kernel-PC-label):",
            f"{'Capture':<12} {'Physical-SIMD keys':>18} "
            f"{'Max distinct slot IDs per physical-SIMD key':>47} "
            f"{'Coordinate':<64} "
            f"{'Max simultaneously active slot IDs per physical-SIMD key':>61} "
            "Coordinate",
            "-" * 226,
        ]
    )
    all_slot_usage: list[tuple[int, SlotUsageSummary]] = []
    for simd_id in simd_ids:
        usages = summaries[simd_id].occupancy.slot_usage
        if not usages:
            raise ValueError(f"SIMD{simd_id} has no slot usage summaries")
        distinct_maximum = max(len(item.distinct_slot_ids) for item in usages)
        distinct_ties = [
            item
            for item in usages
            if len(item.distinct_slot_ids) == distinct_maximum
        ]
        concurrent_maximum = max(
            item.max_concurrent_active_slots for item in usages
        )
        concurrent_ties = [
            item
            for item in usages
            if item.max_concurrent_active_slots == concurrent_maximum
        ]
        lines.append(
            f"{f'SIMD{simd_id}-select':<12} {len(usages):>18} "
            f"{distinct_maximum:>47} "
            f"{(_slot_usage_coordinate(distinct_ties[0]) + _tie_suffix(len(distinct_ties))):<64} "
            f"{concurrent_maximum:>61} "
            f"{_slot_usage_coordinate(concurrent_ties[0])}"
            f"{_tie_suffix(len(concurrent_ties))}"
        )
        all_slot_usage.extend((simd_id, item) for item in usages)
    observed_distinct = max(
        len(item.distinct_slot_ids) for _, item in all_slot_usage
    )
    observed_distinct_ties = [
        (simd_id, item)
        for simd_id, item in all_slot_usage
        if len(item.distinct_slot_ids) == observed_distinct
    ]
    observed_concurrent = max(
        item.max_concurrent_active_slots for _, item in all_slot_usage
    )
    observed_concurrent_ties = [
        (simd_id, item)
        for simd_id, item in all_slot_usage
        if item.max_concurrent_active_slots == observed_concurrent
    ]
    distinct_simd, distinct_item = observed_distinct_ties[0]
    concurrent_simd, concurrent_item = observed_concurrent_ties[0]
    slot_maximum_summary_lines = [
        (
            "Max distinct slot IDs per physical-SIMD key observed across "
            f"independent captures: {observed_distinct} at "
            f"SIMD{distinct_simd}-select/"
            f"{_slot_usage_coordinate(distinct_item)}"
            f"{_tie_suffix(len(observed_distinct_ties))}"
        ),
        (
            "Max simultaneously active slot IDs per physical-SIMD key "
            f"observed across independent captures: {observed_concurrent} at "
            f"SIMD{concurrent_simd}-select/"
            f"{_slot_usage_coordinate(concurrent_item)}"
            f"{_tie_suffix(len(observed_concurrent_ties))}"
        ),
    ]

    lines.extend(
        [
            "",
            "Distinct physical WGPs per independent capture x SE x "
            "kernel-PC label (packed ID keeps SA distinct):",
            "  Decode: SA=(packed>>7)&1; WGP=packed&0x7f.",
            f"{'Capture':<12} {'SE':<4} {'Kernel label':>12} "
            f"{'WGP count':>10}  Observed packed SA/WGP IDs",
            "-" * 105,
        ]
    )
    all_wgp_usage: list[tuple[int, WGPUsageSummary]] = []
    for simd_id in simd_ids:
        for usage in summaries[simd_id].occupancy.wgp_usage:
            all_wgp_usage.append((simd_id, usage))
            lines.append(
                f"{f'SIMD{simd_id}-select':<12} {usage.se_name:<4} "
                f"{usage.kernel_pc_index:>12} "
                f"{len(usage.packed_sa_wgp_ids):>10}  "
                f"{_format_local_wgp_ranges(usage.packed_sa_wgp_ids)}"
            )
    maximum_wgp_count = max(
        len(item.packed_sa_wgp_ids) for _, item in all_wgp_usage
    )
    maximum_wgp_ties = [
        (simd_id, item)
        for simd_id, item in all_wgp_usage
        if len(item.packed_sa_wgp_ids) == maximum_wgp_count
    ]
    wgp_simd, wgp_item = maximum_wgp_ties[0]
    lines.append(
        "Max distinct packed SA/WGP count observed across independent "
        f"captures: {maximum_wgp_count} at "
        f"SIMD{wgp_simd}-select/{wgp_item.se_name}/"
        f"kernel-label={wgp_item.kernel_pc_index}"
        f"{_tie_suffix(len(maximum_wgp_ties))}"
    )

    lines.extend(
        [
            "",
            "Physical-WGP completion imbalance per independent capture x SE x "
            "kernel-PC label:",
            "  Occupancy tuple field 6 is a kernel-PC label/index, not the UI "
            "directory's dispatch ID.",
            "  Episode = aggregate active-wave count 0 -> 1 through 1 -> 0 for "
            "one (SE, packed-SA-WGP, kernel-PC-label).",
            "  Episode count only diagnoses occupancy gaps; it is not the "
            "load-balance sample count.",
            "  All episodes for one physical WGP are merged first: envelope = "
            "last end - first start; active = sum(episode durations); "
            "idle-gap = envelope - active.",
            "  Episodes use raw profiler shader timestamps across a WGP's "
            "hwSIMDs; they are not REALTIME-aligned wall durations.",
            "  occupancy.json has no logical WG ID. A physical WGP may host "
            "multiple logical WGs, so its envelope is not a single-WG duration.",
            "  Formula: final-end span=max(WGP_end)-min(WGP_end); completion "
            "imbalance=final-end span/median(WGP envelope duration).",
            f"{'Capture':<12} {'SE':<4} {'Kernel label':>12} {'WGPs':>5} "
            f"{'Episodes/WGP':>12} {'Envelope min':>12} "
            f"{'Envelope median':>15} {'Envelope max':>12} "
            f"{'Active median':>13} {'Idle-gap median':>15} "
            f"{'Final-end min':>13} {'Final-end max':>13} "
            f"{'Final-end span':>14} {'Completion imbalance':>20}",
            "-" * 196,
        ]
    )
    calculable_imbalances: list[tuple[int, WGPCompletionSummary]] = []
    for simd_id in simd_ids:
        for item in summaries[simd_id].occupancy.completion_by_group:
            episode_range = (
                str(item.episode_count_min)
                if item.episode_count_min == item.episode_count_max
                else f"{item.episode_count_min}-{item.episode_count_max}"
            )
            final_end_span_text = (
                str(item.final_end_span)
                if item.final_end_span is not None
                else "N/A(<2)"
            )
            imbalance_text = (
                f"{item.completion_imbalance:.6f}"
                if item.completion_imbalance is not None
                else "N/A(<2)"
            )
            lines.append(
                f"{f'SIMD{simd_id}-select':<12} {item.se_name:<4} "
                f"{item.kernel_pc_index:>12} {item.wgp_count:>5} "
                f"{episode_range:>12} {item.envelope_min:>12} "
                f"{item.envelope_median:>15.3f} "
                f"{item.envelope_max:>12} {item.active_median:>13.3f} "
                f"{item.idle_gap_median:>15.3f} "
                f"{item.final_end_min:>13} {item.final_end_max:>13} "
                f"{final_end_span_text:>14} {imbalance_text:>20}"
            )
            if item.completion_imbalance is not None:
                calculable_imbalances.append((simd_id, item))
    completion_summary_lines: list[str] = []
    if calculable_imbalances:
        imbalance_values = [
            item.completion_imbalance
            for _, item in calculable_imbalances
            if item.completion_imbalance is not None
        ]
        maximum_imbalance_simd, maximum_imbalance_item = max(
            calculable_imbalances,
            key=lambda entry: (
                -math.inf
                if entry[1].completion_imbalance is None
                else entry[1].completion_imbalance
            ),
        )
        completion_summary_lines.append(
            "Completion imbalance arithmetic mean "
            f"({len(imbalance_values)} independent capture x SE x "
            f"kernel-PC-label rows): {fmean(imbalance_values):.6f}"
        )
        completion_summary_lines.append(
            f"Completion imbalance median: {median(imbalance_values):.6f}; "
            f"max observed {maximum_imbalance_item.completion_imbalance:.6f} "
            f"at SIMD{maximum_imbalance_simd}-select/"
            f"{maximum_imbalance_item.se_name}/"
            f"kernel-label={maximum_imbalance_item.kernel_pc_index}"
        )
    else:
        completion_summary_lines.append(
            "Completion imbalance arithmetic mean: N/A "
            "(no capture x SE x kernel-PC-label row has at least two "
            "physical WGP summaries)"
        )

    if output_paths is not None:
        lines.extend(
            [
                "",
                f"Saved {len(simd_ids)} SIMD-select capture plot(s) "
                "(shared y-axis across every discovered capture):",
            ]
        )
        for simd_id in simd_ids:
            lines.append(f"  SIMD{simd_id}-select: {output_paths[simd_id]}")

    lines.extend(["", "End-of-report summary:", wave_reuse_summary_line, ""])
    lines.extend(slot_maximum_summary_lines)
    lines.append("")
    lines.extend(completion_summary_lines)
    lines.append("")
    lines.extend(frequency_summary_lines)
    lines.append("")
    for observed in observed_cycle_maxima:
        coordinate_text = ", ".join(
            f"SIMD{simd_id}-select/{se_name}"
            for simd_id, se_name in observed.coordinates
        )
        lines.append(
            "Max observed across independent SIMD-select captures: "
            f"per_se_cycle={_format_cycle_value(observed.cycle_span)} at "
            f"{coordinate_text} (source: {observed.source_name})"
        )

    lines.append("")
    for observed in observed_cycle_maxima:
        simd_id, se_name = observed.coordinates[0]
        capture = summaries[simd_id].realtime.capture
        earliest, latest, realtime_span = compute_realtime_event_span(capture)
        kernel_time_us = realtime_span / capture.reference_hz * 1.0e6
        derived_frequency_mhz = observed.cycle_span / kernel_time_us
        coordinate_note = f"SIMD{simd_id}-select/{se_name}"
        if len(observed.coordinates) > 1:
            coordinate_note += (
                f" (first of {len(observed.coordinates)} tied maxima)"
            )
        lines.append(
            f"Derived frequency from {observed.source_name} max "
            f"per_se_cycle: {derived_frequency_mhz:.3f} MHz "
            f"({_format_cycle_value(observed.cycle_span)} cycles / "
            f"{kernel_time_us:.3f} us at {coordinate_note}; matching "
            f"realtime.json all-SE REALTIME earliest="
            f"{_format_cycle_value(earliest)}, latest="
            f"{_format_cycle_value(latest)}, span="
            f"{_format_cycle_value(realtime_span)} ticks)"
        )

    return "\n".join(lines)


def analyze_directory_capture(
    att_root: Path,
    reference_hz_override: Optional[float] = None,
) -> dict[int, DirectoryCaptureSummary]:
    """Discover, load, and summarize ATT captures. Does not write plots."""
    root = att_root.expanduser().resolve()
    selected_files = discover_capture_files(root)
    summaries: dict[int, DirectoryCaptureSummary] = {}
    for simd_id, files in selected_files.items():
        realtime_capture = load_and_validate_realtime(
            files.realtime_path,
            reference_hz_override=reference_hz_override,
            required_se_names=EXPECTED_SE_NAMES,
        )
        occupancy_capture = load_and_validate_occupancy(
            files.occupancy_path,
            required_se_names=EXPECTED_SE_NAMES,
        )
        realtime_summary = summarize_capture(
            realtime_capture,
            simd_id=simd_id,
        )
        summaries[simd_id] = DirectoryCaptureSummary(
            simd_id=simd_id,
            files=files,
            realtime=realtime_summary,
            realtime_cycle_spans=compute_realtime_cycle_spans(
                realtime_capture
            ),
            occupancy=summarize_occupancy(occupancy_capture),
        )
    return summaries


def run_directory_mode(args: argparse.Namespace) -> list[Path]:
    att_root = args.att_dir.expanduser().resolve()
    summaries = analyze_directory_capture(
        att_root,
        reference_hz_override=args.reference_hz,
    )
    if getattr(args, "no_plot", False):
        print(format_directory_summary(summaries))
        return []

    simd_ids = sorted(summaries)
    output_paths = build_directory_output_paths(
        att_root,
        args.output_dir,
        simd_ids=simd_ids,
    )
    source_paths = [
        path
        for files in (item.files for item in summaries.values())
        for path in (files.realtime_path, files.occupancy_path)
    ]
    validate_directory_output_paths(
        att_root,
        output_paths,
        source_paths,
    )
    y_limits = shared_y_limits(
        [summaries[simd_id].realtime for simd_id in simd_ids]
    )

    output_paths[simd_ids[0]].parent.mkdir(parents=True, exist_ok=True)
    for simd_id in simd_ids:
        plot_simd_capture(
            summaries[simd_id].realtime,
            att_root.name,
            output_paths[simd_id],
            y_limits,
        )
        if (
            not output_paths[simd_id].is_file()
            or output_paths[simd_id].stat().st_size == 0
        ):
            raise RuntimeError(
                f"Plot was not created as a non-empty file: "
                f"{output_paths[simd_id]}"
            )

    print(format_directory_summary(summaries, output_paths))
    return [output_paths[simd_id] for simd_id in simd_ids]


def run_legacy_mode(args: argparse.Namespace) -> Path:
    script_dir = Path(__file__).resolve().parent
    input_path = args.input or default_input_path(script_dir)
    output = args.output or script_dir / "fgfx_all_se.png"
    summary = summarize_capture(
        load_and_validate_realtime(
            input_path,
            reference_hz_override=args.reference_hz,
        )
    )
    plot_legacy_capture(summary, output)
    print(f"Saved plot to {output}")
    return output


def _write_self_test_realtime(
    path: Path,
    frequency: Optional[float] = 100,
) -> None:
    payload = {
        se_name: [[0, 0], [10, 5]]
        for se_name in EXPECTED_SE_NAMES
    }
    metadata: dict[str, object] = {
        "descriptor": "[gfx_clock, realtime_clock]",
    }
    if frequency is not None:
        metadata["frequency"] = frequency
    payload["metadata"] = metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_self_test_occupancy(
    path: Path,
    events_by_numeric_se: Optional[
        Mapping[str, Sequence[Sequence[int]]]
    ] = None,
) -> None:
    if events_by_numeric_se is None:
        events_by_numeric_se = {
            str(se_id): (
                (10 + se_id, se_id, 0, 0, 1, 1),
                (30 + se_id, se_id, 0, 0, 0, 1),
            )
            for se_id in range(len(EXPECTED_SE_NAMES))
        }
    kernel_pc_indices = sorted(
        {
            int(event[5])
            for events in events_by_numeric_se.values()
            for event in events
            if len(event) == 6 and isinstance(event[5], int)
        }
    )
    payload: dict[str, object] = {
        key: [list(event) for event in events]
        for key, events in events_by_numeric_se.items()
    }
    payload["dispatches"] = {
        str(index): f"self_test_kernel_pc_label_{index}"
        for index in kernel_pc_indices
    }
    payload["version"] = "self-test"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_self_test_capture(
    att_root: Path,
    simd_id: int,
    ui_name: Optional[str] = None,
    include_occupancy: bool = True,
) -> Path:
    ui_directory = (
        att_root
        / "thread_trace"
        / f"simd{simd_id}"
        / "kernel"
        / "rpf_v3"
        / (ui_name or f"ui_output_agent_{simd_id}_dispatch_1")
    )
    _write_self_test_realtime(ui_directory / "realtime.json")
    if include_occupancy:
        _write_self_test_occupancy(ui_directory / "occupancy.json")
    return ui_directory


def _expect_value_error(
    label: str,
    expected_text: str,
    action: Callable[[], object],
) -> None:
    try:
        action()
    except ValueError as error:
        if expected_text not in str(error):
            raise AssertionError(
                f"{label}: expected {expected_text!r} in error, got: {error}"
            ) from error
    else:
        raise AssertionError(f"{label}: expected ValueError")


def run_self_test() -> None:
    checks = 0

    weighted = compute_interval_summary(
        "SE0",
        ((0.0, 0.0), (20.0, 2.0), (50.0, 8.0)),
        reference_hz=100.0,
        realtime_origin=0.0,
    )
    if not math.isclose(weighted.weighted_mean_mhz, 0.000625):
        raise AssertionError(
            f"weighted mean mismatch: {weighted.weighted_mean_mhz}"
        )
    print("[PASS] weighted mean uses summed ticks")
    checks += 1

    bad_timestamps = compute_interval_summary(
        "SE0",
        (
            (0.0, 10.0),
            (10.0, 20.0),
            (20.0, 20.0),
            (30.0, 15.0),
            (50.0, 25.0),
        ),
        reference_hz=100.0,
        realtime_origin=10.0,
    )
    if (
        bad_timestamps.valid_intervals != 2
        or bad_timestamps.dropped_nonpositive_realtime != 2
        or not math.isclose(
            bad_timestamps.weighted_mean_mhz, 0.000150
        )
    ):
        raise AssertionError(
            "bad timestamp filtering or accounting produced the wrong result"
        )
    print("[PASS] bad timestamps are filtered and counted")
    checks += 1

    with tempfile.TemporaryDirectory(prefix="att_capture_analysis_self_test_") as temp:
        temp_root = Path(temp)

        zero_freq_path = temp_root / "zero_freq" / "realtime.json"
        _write_self_test_realtime(zero_freq_path, frequency=0)
        zero_freq_capture = load_and_validate_realtime(zero_freq_path)
        if (
            zero_freq_capture.reference_hz != DEFAULT_REFERENCE_HZ
            or zero_freq_capture.metadata_hz != 0
            or zero_freq_capture.reference_source
            != "default --reference-hz 100 MHz"
        ):
            raise AssertionError(
                "zero metadata.frequency did not fall back to 100 MHz"
            )
        missing_freq_path = temp_root / "missing_freq" / "realtime.json"
        _write_self_test_realtime(missing_freq_path, frequency=None)
        missing_freq_capture = load_and_validate_realtime(missing_freq_path)
        if (
            missing_freq_capture.reference_hz != DEFAULT_REFERENCE_HZ
            or missing_freq_capture.metadata_hz is not None
            or missing_freq_capture.reference_source
            != "default --reference-hz 100 MHz"
        ):
            raise AssertionError(
                "missing metadata.frequency did not fall back to 100 MHz"
            )
        override_capture = load_and_validate_realtime(
            zero_freq_path,
            reference_hz_override=50_000_000.0,
        )
        if override_capture.reference_hz != 50_000_000.0:
            raise AssertionError("explicit --reference-hz override was ignored")
        print("[PASS] missing/zero metadata.frequency falls back to 100 MHz")
        checks += 1

        single_root = temp_root / "single_simd2.att"
        _write_self_test_capture(single_root, 2)
        single_discovered = discover_capture_files(single_root)
        if tuple(single_discovered) != (2,):
            raise AssertionError("single simd2 discovery failed")
        selected_simd2 = single_discovered[2]
        if (
            selected_simd2.realtime_path.parent
            != selected_simd2.occupancy_path.parent
        ):
            raise AssertionError("realtime/occupancy UI pairing failed")
        single_outputs = build_directory_output_paths(
            single_root, simd_ids=tuple(single_discovered)
        )
        if (
            tuple(single_outputs) != (2,)
            or single_outputs[2].name != "single_simd2.att.simd2.png"
        ):
            raise AssertionError("single-SIMD output naming failed")
        print("[PASS] only-simd2 discovery, pairing, and output naming")
        checks += 1

        single_realtime_capture = load_and_validate_realtime(
            selected_simd2.realtime_path,
            required_se_names=EXPECTED_SE_NAMES,
        )
        single_occupancy_capture = load_and_validate_occupancy(
            selected_simd2.occupancy_path,
            required_se_names=EXPECTED_SE_NAMES,
        )
        single_realtime_summary = summarize_capture(
            single_realtime_capture, simd_id=2
        )
        earliest, latest, realtime_span = compute_realtime_event_span(
            single_realtime_capture
        )
        if (earliest, latest, realtime_span) != (0.0, 5.0, 5.0):
            raise AssertionError(
                "all-SE REALTIME event span calculation produced the wrong "
                "result"
            )
        print("[PASS] all-SE REALTIME event span uses earliest/latest samples")
        checks += 1

        directory_text = format_directory_summary(
            {
                2: DirectoryCaptureSummary(
                    simd_id=2,
                    files=selected_simd2,
                    realtime=single_realtime_summary,
                    realtime_cycle_spans=compute_realtime_cycle_spans(
                        single_realtime_capture
                    ),
                    occupancy=summarize_occupancy(
                        single_occupancy_capture
                    ),
                )
            },
            single_outputs,
        )
        forbidden_output_fragments = (
            "dispatch" + "1",
            "Group" + "s",
            "Samples/" + "events",
            "Global " + "max",
            "\u251c\u00f9",
            "\u0393\u00e5\u00c6",
        )
        required_output_fragments = (
            "UI dispatch ID (from directory path): 1",
            "occupancy kernel-PC label index(es):",
            "Events",
            "Clock samples",
            "Max observed across independent SIMD-select captures",
            "wave-lifetimes-per-slot -> physical-slot-key count",
            "observed physical slot keys",
            "Physical-SIMD keys",
            "Max distinct slot IDs per physical-SIMD key",
            "Max simultaneously active slot IDs per physical-SIMD key",
            "Observed packed SA/WGP IDs",
            "Completion imbalance",
        )
        if any(
            fragment in directory_text
            for fragment in forbidden_output_fragments
        ):
            raise AssertionError("directory output contains a forbidden label")
        if any(
            fragment not in directory_text
            for fragment in required_output_fragments
        ):
            raise AssertionError("directory output is missing a required label")
        if not directory_text.isascii():
            raise AssertionError("directory output must remain ASCII-safe")
        ordered_footer_fragments = (
            "Max observed across independent SIMD-select captures: "
            "1 sequential wave lifetimes per slot",
            "Max distinct slot IDs per physical-SIMD key observed across",
            "Max simultaneously active slot IDs per physical-SIMD key",
            "Completion imbalance arithmetic mean",
            "Per-SIMD-select capture means",
            "source: occupancy.json event shader_timestamp",
            "source: realtime.json gfx_clock",
            "Derived frequency from occupancy.json event shader_timestamp",
            "Derived frequency from realtime.json gfx_clock",
        )
        footer_start = directory_text.index("End-of-report summary:")
        footer_positions = [
            directory_text.index(fragment, footer_start)
            for fragment in ordered_footer_fragments
        ]
        if footer_positions != sorted(footer_positions):
            raise AssertionError("end-of-report summary order is incorrect")
        if not directory_text.rstrip().endswith("span=5 ticks)"):
            raise AssertionError(
                "derived frequencies must be the final two summary lines"
            )
        print("[PASS] directory labels are precise and ASCII-safe")
        checks += 1

        with redirect_stdout(StringIO()):
            dynamic_outputs = run_directory_mode(
                argparse.Namespace(
                    att_dir=single_root,
                    output_dir=None,
                    reference_hz=None,
                )
            )
        if (
            dynamic_outputs != [single_outputs[2]]
            or not dynamic_outputs[0].is_file()
            or dynamic_outputs[0].stat().st_size == 0
        ):
            raise AssertionError("dynamic one-SIMD directory mode failed")
        print("[PASS] dynamic one-SIMD directory mode writes one plot")
        checks += 1

        legacy_output = temp_root / "legacy.png"
        with redirect_stdout(StringIO()):
            generated_legacy_output = run_legacy_mode(
                argparse.Namespace(
                    input=selected_simd2.realtime_path,
                    output=legacy_output,
                    reference_hz=None,
                )
            )
        if (
            generated_legacy_output != legacy_output
            or not legacy_output.is_file()
            or legacy_output.stat().st_size == 0
        ):
            raise AssertionError("legacy mode failed")
        print("[PASS] legacy input mode writes one plot")
        checks += 1

        four_root = temp_root / "four_simd.att"
        for simd_id in SIMD_IDS:
            _write_self_test_capture(four_root, simd_id)
        discovered = discover_capture_files(four_root)
        if tuple(discovered) != SIMD_IDS:
            raise AssertionError("four-SIMD directory discovery failed")
        print("[PASS] four-SIMD directory discovery")
        checks += 1

        analysis_summaries = analyze_directory_capture(four_root)
        if tuple(analysis_summaries) != SIMD_IDS:
            raise AssertionError(
                "directory analysis did not summarize every SIMD"
            )
        leftover_plots = list(
            four_root.parent.glob(f"{four_root.name}.simd*.png")
        )
        if leftover_plots:
            raise AssertionError(
                "analyze_directory_capture wrote plots: "
                + ", ".join(str(path) for path in leftover_plots)
            )
        print("[PASS] directory analysis summarizes without writing plots")
        checks += 1

        missing_root = temp_root / "missing_occupancy.att"
        _write_self_test_capture(
            missing_root, 0, include_occupancy=False
        )
        _expect_value_error(
            "missing paired occupancy",
            "same UI directory",
            lambda: discover_capture_files(missing_root),
        )
        print("[PASS] missing paired occupancy is rejected")
        checks += 1

        wrong_pair_root = temp_root / "wrong_pair.att"
        _write_self_test_capture(
            wrong_pair_root,
            1,
            ui_name="ui_selected",
            include_occupancy=False,
        )
        _write_self_test_occupancy(
            wrong_pair_root
            / "thread_trace"
            / "simd1"
            / "kernel"
            / "rpf_v3"
            / "ui_other"
            / "occupancy.json"
        )
        _expect_value_error(
            "occupancy in another UI directory",
            "ui_other",
            lambda: discover_capture_files(wrong_pair_root),
        )
        print("[PASS] occupancy is paired to the selected realtime UI")
        checks += 1

        multiple_root = temp_root / "multiple_candidate.att"
        _write_self_test_capture(
            multiple_root, 0, ui_name="ui_a"
        )
        _write_self_test_capture(
            multiple_root, 0, ui_name="ui_b"
        )
        _expect_value_error(
            "multiple realtime candidates",
            "multiple realtime.json candidates",
            lambda: discover_capture_files(multiple_root),
        )
        print("[PASS] multiple realtime candidates are rejected")
        checks += 1

        empty_root = temp_root / "no_simd.att"
        (empty_root / "thread_trace").mkdir(parents=True)
        _expect_value_error(
            "no SIMD directories",
            "No usable ATT capture found",
            lambda: discover_capture_files(empty_root),
        )
        print("[PASS] empty SIMD capture set is rejected")
        checks += 1

        partial_root = temp_root / "partial_simd.att"
        _write_self_test_capture(partial_root, 0)
        (
            partial_root / "thread_trace" / "simd1" / "kernel" / "rpf_v3"
        ).mkdir(parents=True)
        partial_discovered = discover_capture_files(partial_root)
        if tuple(partial_discovered) != (0,):
            raise AssertionError(
                "incomplete sibling SIMD directories should be skipped"
            )
        print("[PASS] incomplete sibling SIMD directories are skipped")
        checks += 1

        unwrapped_root = temp_root / "unwrapped_kernel.att"
        unwrapped_ui = (
            unwrapped_root
            / "thread_trace"
            / "kernel"
            / "rpf_v3"
            / "ui_output_agent_9_dispatch_1"
        )
        _write_self_test_realtime(unwrapped_ui / "realtime.json")
        _write_self_test_occupancy(unwrapped_ui / "occupancy.json")
        unwrapped_discovered = discover_capture_files(unwrapped_root)
        if tuple(unwrapped_discovered) != (UNWRAPPED_DEFAULT_SIMD_ID,):
            raise AssertionError(
                "unwrapped kernel capture should default to SIMD "
                f"{UNWRAPPED_DEFAULT_SIMD_ID}"
            )
        (unwrapped_root / "input_kernel.yaml").write_text(
            "  att_simd_select: \"0\"\n",
            encoding="utf-8",
        )
        labeled_unwrapped = discover_capture_files(unwrapped_root)
        if tuple(labeled_unwrapped) != (0,):
            raise AssertionError(
                "unwrapped kernel capture should read att_simd_select from yaml"
            )
        print("[PASS] unwrapped thread_trace/kernel capture is accepted")
        checks += 1

        bad_length_path = temp_root / "bad_length.json"
        _write_self_test_occupancy(
            bad_length_path,
            {"0": ((0, 0, 0, 0, 1), (10, 0, 0, 0, 0, 1))},
        )
        _expect_value_error(
            "bad occupancy tuple length",
            "must be [shader_timestamp",
            lambda: load_and_validate_occupancy(
                bad_length_path, required_se_names=("SE0",)
            ),
        )

        bad_state_path = temp_root / "bad_state.json"
        _write_self_test_occupancy(
            bad_state_path,
            {"0": ((0, 0, 0, 0, 2, 1), (10, 0, 0, 0, 0, 1))},
        )
        _expect_value_error(
            "bad occupancy state",
            "start must be 0 or 1",
            lambda: load_and_validate_occupancy(
                bad_state_path, required_se_names=("SE0",)
            ),
        )

        bad_type_path = temp_root / "bad_type.json"
        bad_type_path.write_text(
            json.dumps(
                {
                    "0": [
                        [True, 0, 0, 0, 1, 1],
                        [10, 0, 0, 0, 0, 1],
                    ],
                    "dispatches": {"1": "kernel"},
                }
            ),
            encoding="utf-8",
        )
        _expect_value_error(
            "bad occupancy scalar type",
            "must be an integer",
            lambda: load_and_validate_occupancy(
                bad_type_path, required_se_names=("SE0",)
            ),
        )
        print("[PASS] bad occupancy schemas are rejected")
        checks += 1

        sequential_path = temp_root / "sequential.json"
        _write_self_test_occupancy(
            sequential_path,
            {
                "0": (
                    (10, 0, 0, 0, 1, 1),
                    (20, 0, 0, 0, 0, 1),
                    (20, 0, 0, 0, 1, 1),
                    (30, 0, 0, 0, 0, 1),
                )
            },
        )
        sequential = summarize_occupancy(
            load_and_validate_occupancy(
                sequential_path, required_se_names=("SE0",)
            )
        )
        if (
            len(sequential.wave_lifetimes) != 2
            or sequential.wave_reuse[0].wave_count != 2
            or len(sequential.wgp_episodes) != 2
            or len(sequential.physical_wgp_summaries) != 1
            or sequential.physical_wgp_summaries[0].episode_count != 2
            or sequential.slot_usage[0].distinct_slot_ids != (0,)
            or sequential.slot_usage[0].max_concurrent_active_slots != 1
        ):
            raise AssertionError("sequential slot reuse summary is wrong")
        print("[PASS] equal-timestamp dealloc-before-alloc slot reuse")
        checks += 1

        separated_path = temp_root / "separated_episodes.json"
        _write_self_test_occupancy(
            separated_path,
            {
                "0": (
                    (10, 0, 0, 0, 1, 1),
                    (20, 0, 0, 0, 0, 1),
                    (30, 0, 0, 0, 1, 1),
                    (50, 0, 0, 0, 0, 1),
                    (70, 0, 0, 0, 1, 1),
                    (100, 0, 0, 0, 0, 1),
                )
            },
        )
        separated = summarize_occupancy(
            load_and_validate_occupancy(
                separated_path, required_se_names=("SE0",)
            )
        )
        separated_wgp = separated.physical_wgp_summaries[0]
        if (
            len(separated.wgp_episodes) != 3
            or len(separated.physical_wgp_summaries) != 1
            or separated_wgp.episode_count != 3
            or separated_wgp.envelope_duration != 90
            or separated_wgp.active_duration != 60
            or separated_wgp.idle_gap_duration != 30
            or separated.completion_by_group[0].wgp_count != 1
        ):
            raise AssertionError("separated episodes were not merged per WGP")
        print("[PASS] separated episodes merge into one physical-WGP sample")
        checks += 1

        concurrent_path = temp_root / "concurrent.json"
        _write_self_test_occupancy(
            concurrent_path,
            {
                "0": (
                    (10, 0, 0, 0, 1, 1),
                    (11, 0, 0, 1, 1, 1),
                    (20, 0, 0, 0, 0, 1),
                    (21, 0, 0, 1, 0, 1),
                )
            },
        )
        concurrent = summarize_occupancy(
            load_and_validate_occupancy(
                concurrent_path, required_se_names=("SE0",)
            )
        )
        if (
            concurrent.slot_usage[0].distinct_slot_ids != (0, 1)
            or concurrent.slot_usage[0].max_concurrent_active_slots != 2
        ):
            raise AssertionError("concurrent slot summary is wrong")
        print("[PASS] distinct and simultaneously active slot counts differ")
        checks += 1

        multi_label_path = temp_root / "multi_kernel_pc_label.json"
        _write_self_test_occupancy(
            multi_label_path,
            {
                "0": (
                    (0, 0, 0, 0, 1, 1),
                    (0, 1, 0, 0, 1, 1),
                    (0, 0, 0, 0, 1, 2),
                    (0, 1, 0, 0, 1, 2),
                    (100, 0, 0, 0, 0, 1),
                    (110, 1, 0, 0, 0, 1),
                    (200, 0, 0, 0, 0, 2),
                    (240, 1, 0, 0, 0, 2),
                )
            },
        )
        multi_label = summarize_occupancy(
            load_and_validate_occupancy(
                multi_label_path, required_se_names=("SE0",)
            )
        )
        label_rows = {
            item.kernel_pc_index: item
            for item in multi_label.completion_by_group
        }
        if (
            len(multi_label.wave_reuse) != 4
            or any(
                item.wave_count != 1
                for item in multi_label.wave_reuse
            )
            or len(multi_label.slot_usage) != 4
            or len(multi_label.wgp_episodes) != 4
            or tuple(label_rows) != (1, 2)
            or any(item.wgp_count != 2 for item in label_rows.values())
            or not math.isclose(
                label_rows[1].completion_imbalance or -1.0,
                10.0 / 105.0,
            )
            or not math.isclose(
                label_rows[2].completion_imbalance or -1.0,
                40.0 / 220.0,
            )
        ):
            raise AssertionError("kernel-PC-label-separated accounting is wrong")
        print("[PASS] kernel-PC labels produce separate completion rows")
        checks += 1

        formula_path = temp_root / "imbalance_formula.json"
        formula_events: list[tuple[int, ...]] = []
        for packed_sa_wgp in range(16):
            formula_events.extend(
                (
                    (0, packed_sa_wgp, 0, 0, 1, 1),
                    (100 + packed_sa_wgp, packed_sa_wgp, 0, 0, 0, 1),
                )
            )
        _write_self_test_occupancy(
            formula_path,
            {"0": formula_events},
        )
        formula = summarize_occupancy(
            load_and_validate_occupancy(
                formula_path, required_se_names=("SE0",)
            )
        ).completion_by_group[0]
        if (
            formula.wgp_count != 16
            or formula.episode_count_min != 1
            or formula.episode_count_max != 1
            or formula.envelope_min != 100
            or not math.isclose(formula.envelope_median, 107.5)
            or formula.envelope_max != 115
            or formula.final_end_span != 15
            or formula.completion_imbalance is None
            or not math.isclose(
                formula.completion_imbalance, 15.0 / 107.5
            )
        ):
            raise AssertionError("16-WGP completion formula is wrong")
        print("[PASS] 16-WGP synthetic completion formula")
        checks += 1

        lifecycle_path = temp_root / "bad_lifecycle.json"
        _write_self_test_occupancy(
            lifecycle_path,
            {
                "0": (
                    (0, 0, 0, 0, 1, 1),
                    (1, 0, 0, 0, 1, 1),
                    (2, 0, 0, 0, 0, 1),
                )
            },
        )
        lifecycle_capture = load_and_validate_occupancy(
            lifecycle_path, required_se_names=("SE0",)
        )
        _expect_value_error(
            "double allocation",
            "double alloc",
            lambda: summarize_occupancy(lifecycle_capture),
        )
        print("[PASS] invalid occupancy lifecycle is rejected")
        checks += 1

    print(f"Self-test passed: {checks} checks")


def main() -> None:
    if sys.platform == "win32":
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.self_test:
            if (
                args.output is not None
                or args.output_dir is not None
                or args.reference_hz is not None
                or args.no_plot
            ):
                parser.error(
                    "--self-test cannot be combined with output, plot, or "
                    "reference options"
                )
            run_self_test()
            return

        if args.att_dir is not None:
            if args.output is not None:
                parser.error("--output is only valid with legacy --input mode")
            if args.no_plot and args.output_dir is not None:
                parser.error("--output-dir cannot be combined with --no-plot")
            run_directory_mode(args)
            return

        if args.no_plot:
            parser.error("--no-plot requires --dir")
        if args.output_dir is not None:
            parser.error("--output-dir requires --dir")
        run_legacy_mode(args)
    except (AssertionError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
