# SPDX-License-Identifier: MIT
"""Validate one exact single-kernel ticket trace and code-object metadata.

The script deliberately requires an exact kernel name.  It rejects empty or
substring-only matches, reports steady kernel duration from rocprofv3 CSV, and
optionally joins the authoritative AMDHSA resource/private/spill metadata from
``llvm-readelf --notes`` output.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


BASELINES_US = {
    "copy_ticket": 18.80,
    "direct_ready": 25.68,
    "serial_approx": 234.0,
}

RESOURCE_FIELDS = (
    "vgpr_count",
    "sgpr_count",
    "group_segment_fixed_size",
    "private_segment_fixed_size",
    "vgpr_spill_count",
    "sgpr_spill_count",
    "max_flat_workgroup_size",
    "wavefront_size",
)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) & 1:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _trace_summary(
    path: Path, kernel: str, last: int, expect_samples: int
) -> dict[str, object]:
    matches: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "Kernel_Name",
            "Start_Timestamp",
            "End_Timestamp",
            "LDS_Block_Size",
            "Scratch_Size",
            "VGPR_Count",
            "Accum_VGPR_Count",
            "SGPR_Count",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"kernel trace is missing columns: {sorted(missing)}")
        for row in reader:
            if row["Kernel_Name"] == kernel:
                matches.append(row)
    if not matches:
        raise ValueError(f"no exact kernel-trace match for {kernel!r}")
    selected = matches[-last:] if last > 0 else matches
    if expect_samples > 0 and len(selected) != expect_samples:
        raise ValueError(
            f"expected {expect_samples} selected dispatches, found {len(selected)}"
        )

    durations = [
        (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1000.0
        for row in selected
    ]
    trace_resources = {
        "lds_block_size": sorted({int(row["LDS_Block_Size"]) for row in selected}),
        "scratch_size": sorted({int(row["Scratch_Size"]) for row in selected}),
        "vgpr_count": sorted({int(row["VGPR_Count"]) for row in selected}),
        "accum_vgpr_count": sorted(
            {int(row["Accum_VGPR_Count"]) for row in selected}
        ),
        "sgpr_count": sorted({int(row["SGPR_Count"]) for row in selected}),
    }
    return {
        "path": str(path.resolve()),
        "exact_matches_total": len(matches),
        "selected_dispatches": len(selected),
        "durations_us": durations,
        "median_us": _median(durations),
        "min_us": min(durations),
        "max_us": max(durations),
        # rocprof fields are retained as a cross-check.  Register/private/spill
        # sign-off must use the code-object metadata below.
        "rocprof_resources": trace_resources,
    }


def _metadata_blocks(text: str) -> list[str]:
    starts = [
        match.start()
        for match in re.finditer(r"(?m)^\s{2}-\s+\.args:\s*$", text)
    ]
    if not starts:
        return [text]
    return [
        text[start : starts[index + 1] if index + 1 < len(starts) else len(text)]
        for index, start in enumerate(starts)
    ]


def _metadata_name(block: str) -> str | None:
    names = re.findall(r"(?m)^\s+\.name:\s+(.+?)\s*$", block)
    # Argument entries also have .name.  The kernel-level name is conventionally
    # the final .name in one AMDHSA kernel block.
    if not names:
        return None
    return names[-1].strip().strip("'\"")


def _metadata_summary(path: Path, kernel: str) -> dict[str, object]:
    text = path.read_text()
    candidates = [
        block for block in _metadata_blocks(text) if _metadata_name(block) == kernel
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected one exact AMDHSA metadata block for {kernel!r}, "
            f"found {len(candidates)}"
        )
    block = candidates[0]
    values: dict[str, int] = {}
    for field in RESOURCE_FIELDS:
        match = re.search(rf"(?m)^\s+\.{re.escape(field)}:\s+(\d+)\s*$", block)
        if match:
            values[field] = int(match.group(1))
        elif field in ("vgpr_spill_count", "sgpr_spill_count"):
            # Some toolchains omit explicit zero spill fields.
            values[field] = 0
        else:
            raise ValueError(f"missing .{field} in exact AMDHSA metadata block")
    values["spill_free"] = int(
        values["private_segment_fixed_size"] == 0
        and values["vgpr_spill_count"] == 0
        and values["sgpr_spill_count"] == 0
    )
    return {"path": str(path.resolve()), **values}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-trace", type=Path, required=True)
    parser.add_argument("--kernel", required=True, help="Exact unmangled kernel name")
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        help="Use only the final N exact dispatches; 0 uses every exact match.",
    )
    parser.add_argument(
        "--expect-samples",
        type=int,
        default=0,
        help="Fail unless the selected trace contains exactly this many dispatches.",
    )
    parser.add_argument(
        "--metadata-notes",
        type=Path,
        help="Text saved from llvm-readelf --notes on the exact code object.",
    )
    parser.add_argument(
        "--require-spill-free",
        action="store_true",
        help="Fail if private bytes or SGPR/VGPR spill counts are non-zero.",
    )
    args = parser.parse_args()
    if args.last < 0 or args.expect_samples < 0:
        raise ValueError("last and expect-samples must be non-negative")

    trace = _trace_summary(
        args.kernel_trace, args.kernel, args.last, args.expect_samples
    )
    median_us = float(trace["median_us"])
    comparisons = {
        name: {
            "baseline_us": baseline,
            "ticket_over_baseline": median_us / baseline,
            "baseline_over_ticket_speedup": baseline / median_us,
            "delta_pct": (median_us / baseline - 1.0) * 100.0,
        }
        for name, baseline in BASELINES_US.items()
    }
    metadata = (
        _metadata_summary(args.metadata_notes, args.kernel)
        if args.metadata_notes is not None
        else None
    )
    if args.require_spill_free:
        if metadata is None:
            raise ValueError("--require-spill-free requires --metadata-notes")
        if not bool(metadata["spill_free"]):
            raise RuntimeError(
                "single-kernel ticket is not spill-free: "
                f"private={metadata['private_segment_fixed_size']} "
                f"vgpr_spill={metadata['vgpr_spill_count']} "
                f"sgpr_spill={metadata['sgpr_spill_count']}"
            )

    output = {
        "kernel": args.kernel,
        "trace": trace,
        "metadata": metadata,
        "comparisons": comparisons,
    }
    print("MEGAMOE_SINGLE_KERNEL_TICKET " + json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
