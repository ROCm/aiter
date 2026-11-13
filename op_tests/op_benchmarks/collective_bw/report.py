# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Grade gathered records against their roofs, and write the tables.

Per op row:

* ``algbw`` is ``logical_bytes / us``. ``busbw`` is algbw times the rccl-tests
  factor: ``(N-1)/N`` for all-to-all and ``2(N-1)/N`` for all-reduce.
* ``wire GB/s`` is the bytes the op sent to other GPUs (wire format,
  ``fabric_bytes_per_rank``) over ``us``.
* ``op eff`` is ``roof us / op us``: 1.0 is the op's TransferBench roof, which
  moves the op's own (wire-format) bytes in its pattern and memory type.
* ``rccl eff`` is ``rccl roof us / rccl us``. RCCL gets its own roof, because
  it moves different bytes: bf16, not the op's quantized wire format
  (``patterns.rccl_traffic``).
* ``roof cfg`` / ``rccl roof cfg`` name the candidate that set each roof.
* ``op wire/peak`` / ``rccl wire/peak``: the wire rate over the per-rank
  fabric peak (``Roofline.measure_peaks``, push to all peers), the same
  for every implementation. A section after the table sums this up per
  implementation at its largest message (``summary``).

Both roofs are a stock copy kernel's best, not the link's peak, so an
implementation with a better copy loop can reach an eff above 1.0 (see
``roofline``).
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import replace
from pathlib import Path

from measure import RCCL, Record
from patterns import busbw_factor, fabric_bytes_per_rank, rccl_traffic
from roof_settings import DEFAULT, RoofSettings
from roofline import Roof

COLUMNS = (
    "case",
    "op",
    "timing",
    "roof us",
    "op us",
    "op eff",
    "op busbw GB/s",
    "op wire GB/s",
    "op wire/peak",
    "rccl roof us",
    "rccl us",
    "rccl eff",
    "rccl busbw GB/s",
    "rccl wire GB/s",
    "rccl wire/peak",
    "op spread us",
    "variant",
    "roof cfg",
    "rccl roof cfg",
)

_TEXT_COLUMNS = ("case", "op", "timing", "variant", "roof cfg", "rccl roof cfg")


def _gbps(nbytes: float, us: float) -> float:
    return nbytes / us / 1e3 if us > 0 else math.nan


def wire_gbps(r: Record) -> float:
    """The bytes one rank of *r* sent to other GPUs per second: the op's
    wire format, or RCCL's unquantized bytes."""
    t = rccl_traffic(r.traffic) if r.impl == RCCL else r.traffic
    return _gbps(fabric_bytes_per_rank(t), r.us)


def busbw_gbps(r: Record) -> float:
    """rccl-tests bus bandwidth: logical (caller-dtype) bytes, so a
    quantized op can exceed the fabric."""
    t = r.traffic
    return _gbps(t.logical_bytes, r.us) * busbw_factor(t.collective, t.world_size)


def grade(
    records: list[Record],
    roofs: dict,
    rccl_roofs: dict | None = None,
    peak_gbps: float | None = None,
) -> list[dict]:
    """One row per (case, op). RCCL's columns repeat on every op row of
    its case. *roofs* is keyed by the op traffic, *rccl_roofs* by
    ``rccl_traffic`` of it. *peak_gbps* is the per-rank fabric peak."""
    rccl_roofs = rccl_roofs or {}
    peak = peak_gbps or math.nan
    rccl = {r.case: r for r in records if r.impl == RCCL}
    rows = []
    for r in records:
        if r.impl == RCCL:
            continue
        t = r.traffic
        roof: Roof | None = roofs.get(t)
        roof_us = roof.us if roof else math.nan
        base = rccl.get(r.case)
        rroof: Roof | None = (
            rccl_roofs.get(rccl_traffic(base.traffic)) if base else None
        )
        rroof_us = rroof.us if rroof else math.nan
        rows.append(
            {
                "case": r.case,
                "op": r.impl,
                "timing": r.timing,
                "roof us": roof_us,
                "op us": r.us,
                "op eff": roof_us / r.us,
                "op busbw GB/s": busbw_gbps(r),
                "op wire GB/s": wire_gbps(r),
                "op wire/peak": wire_gbps(r) / peak,
                "rccl roof us": rroof_us,
                "rccl us": base.us if base else math.nan,
                "rccl eff": rroof_us / base.us if base else math.nan,
                "rccl busbw GB/s": busbw_gbps(base) if base else math.nan,
                "rccl wire GB/s": wire_gbps(base) if base else math.nan,
                "rccl wire/peak": wire_gbps(base) / peak if base else math.nan,
                "op spread us": r.spread_us,
                "variant": r.variant or "",
                "roof cfg": roof.label if roof else "",
                "rccl roof cfg": rroof.label if rroof else "",
            }
        )
    return rows


def _fmt(v) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def markdown(rows: list[dict]) -> str:
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "|"
        + "|".join(
            "---" if c in _TEXT_COLUMNS else "---:"
            for c in COLUMNS
        )
        + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row[c]) for c in COLUMNS) + " |")
    return "\n".join(lines)


def provenance(
    roofline, extra: list[str] = (), *, settings: RoofSettings = DEFAULT, configs=None
) -> list[str]:
    """Header lines: what changes the numbers. *roofline* is None when the
    roof was skipped on request. *configs* is the TransferBench config index."""
    import os

    if roofline is None:
        tb = "skipped (--no-roof or --tb-dry-run)"
    elif roofline.available:
        tb = (
            f"TransferBench {roofline.version} at {roofline.binary}, "
            f"{settings.iterations} iterations, {settings.warmups} warmup"
        )
    else:
        tb = "TransferBench not found: no roofline"
    return [
        f"- roofline: {tb}",
        *([f"- TransferBench configs: {configs}"] if configs else []),
        f"- HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES', '(unset)')}",
        *extra,
        f"- command: {' '.join(sys.argv)}",
    ]


def add_args(parser) -> None:
    """Options every bandwidth script shares: outputs and the roofline."""
    parser.add_argument("--report", metavar="PATH", help="also write markdown here")
    parser.add_argument("--csv", metavar="PATH", help="also write CSV here")
    parser.add_argument(
        "--tb-bin",
        metavar="PATH",
        help="TransferBench binary (default $TRANSFERBENCH, PATH)",
    )
    parser.add_argument(
        "--tb-iters",
        type=int,
        default=None,
        help=f"TransferBench iterations (default {DEFAULT.iterations})",
    )
    parser.add_argument(
        "--tb-warmup",
        type=int,
        default=None,
        help=f"TransferBench warmups (default {DEFAULT.warmups})",
    )
    parser.add_argument(
        "--link-gbps",
        type=float,
        default=None,
        help="datasheet rate of one link, GB/s per direction, for the nominal\n"
        "per-rank peak (N-1 links on xGMI, 1 on PCIe)",
    )
    parser.add_argument(
        "--tb-dir",
        metavar="DIR",
        help="where to write every roof's TransferBench configs (always written;\n"
        "default: next to --report as <report>.tb/, else ./collective_bw_tb)",
    )
    parser.add_argument(
        "--no-roof",
        action="store_true",
        help="skip TransferBench; report bandwidth only",
    )
    parser.add_argument(
        "--tb-dry-run",
        action="store_true",
        help="write the TransferBench configs of every roof (see --tb-dir)\n"
        "without running them",
    )


def settings_from(args) -> RoofSettings:
    """``RoofSettings`` with the command line's overrides."""
    over = {
        "iterations": args.tb_iters,
        "warmups": args.tb_warmup,
        "link_gbps": args.link_gbps,
    }
    return replace(DEFAULT, **{k: v for k, v in over.items() if v is not None})


def detect_link() -> str:
    """``xgmi`` or ``pcie`` for this host, ``unknown`` if it cannot tell."""
    from aiter.ops.flydsl.allreduce_policy import detect_link as _detect

    try:
        return _detect()
    except ValueError:
        return "unknown"


def nominal_gbps(link: str, world_size: int, settings: RoofSettings) -> float:
    """The datasheet per-rank peak: ``settings.link_gbps`` times the links a
    rank can use at once (N - 1 on an xGMI mesh, 1 on PCIe). NaN without a
    rate or a known link."""
    links = {"xgmi": world_size - 1, "pcie": 1}.get(link)
    if settings.link_gbps is None or links is None:
        return math.nan
    return settings.link_gbps * links


def best_known(records: list[Record], peaks: dict) -> tuple[float, str]:
    """The highest per-rank wire rate seen in this run, the fabric peak
    probe included, and where it was seen."""
    best = (math.nan, "")
    if "all_peers" in peaks:
        best = (peaks["all_peers"].gbps, "TransferBench fabric peak")
    for r in records:
        g = wire_gbps(r)
        if not math.isnan(g) and not g <= best[0]:  # a NaN best is replaced
            best = (g, f"{r.impl}, {r.case}")
    return best


def peak_lines(
    records: list[Record], peaks: dict, link: str, settings: RoofSettings
) -> list[str]:
    """Header lines with the per-rank peaks the summary divides by."""
    n = records[0].traffic.world_size if records else 0
    mib = settings.peak_bytes_per_peer >> 20
    out = [f"- link: {link}, {n} ranks"]
    for kind, what in (
        ("all_peers", f"push to all {n - 1} peers"),
        ("one_peer", "push to the next rank only (one ring)"),
    ):
        pk = peaks.get(kind)
        if pk is None:
            out.append(f"- fabric peak per rank, {what}: not measured")
            continue
        out.append(
            f"- fabric peak per rank, {what}, {mib} MiB per peer: "
            f"{pk.gbps:.1f} GB/s ({pk.per_link_gbps:.1f} per link; "
            f"TransferBench `{pk.roof.label}`)"
        )
    gbps, where = best_known(records, peaks)
    out.append(f"- best known per rank in this run: {_fmt(gbps)} GB/s ({where})")
    nominal = nominal_gbps(link, n, settings)
    if math.isnan(nominal):
        out.append("- nominal per rank: not set (--link-gbps)")
    else:
        out.append(
            f"- nominal per rank: {nominal:.1f} GB/s "
            f"({settings.link_gbps:g} GB/s per link, datasheet)"
        )
    return out


SUMMARY_COLUMNS = (
    "link",
    "collective",
    "ranks",
    "impl",
    "case",
    "wire MB per rank",
    "wire GB/s per rank",
    "wire/peak",
    "wire/best known",
    "wire/nominal",
    "busbw GB/s",
    "busbw/peak",
)


def summary(
    records: list[Record], peaks: dict, link: str, settings: RoofSettings
) -> list[dict]:
    """Per implementation, at its largest message: the per-rank wire rate
    (bytes actually sent to other GPUs, slowest rank) against the per-rank
    peaks, and the logical bus bandwidth against the fabric peak."""
    peak = peaks["all_peers"].gbps if "all_peers" in peaks else math.nan
    best, _where = best_known(records, peaks)
    out = []
    for impl in dict.fromkeys(r.impl for r in records):
        mine = [r for r in records if r.impl == impl]
        r = max(mine, key=lambda x: x.traffic.logical_bytes)
        t = rccl_traffic(r.traffic) if r.impl == RCCL else r.traffic
        nominal = nominal_gbps(link, t.world_size, settings)
        wire, bus = wire_gbps(r), busbw_gbps(r)
        out.append(
            {
                "link": link,
                "collective": t.collective,
                "ranks": t.world_size,
                "impl": impl,
                "case": r.case,
                "wire MB per rank": fabric_bytes_per_rank(t) / 1e6,
                "wire GB/s per rank": wire,
                "wire/peak": wire / peak,
                "wire/best known": wire / best,
                "wire/nominal": wire / nominal,
                "busbw GB/s": bus,
                "busbw/peak": bus / peak,
            }
        )
    return out


def summary_markdown(rows: list[dict]) -> str:
    text = ("link", "collective", "impl", "case")
    lines = [
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "|" + "|".join("---" if c in text else "---:" for c in SUMMARY_COLUMNS) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_fmt(row[c]) for c in SUMMARY_COLUMNS) + " |"
        )
    return "\n".join(lines)


_SUMMARY_NOTE = (
    "Per rank, at each implementation's largest message: `wire` is the bytes "
    "one rank sends to other GPUs (the op's wire format; RCCL's bf16) over the "
    "slowest rank's time. `peak` is the TransferBench fabric peak (push to all "
    "peers); `best known` the highest per-rank wire rate in this run, peak "
    "included; `nominal` the datasheet rate (`--link-gbps`). `busbw` counts "
    "logical bytes, so a quantized op can exceed 1.0 of the peak."
)


def finish(
    records: list[Record], *, title: str, args, header: list[str] = ()
) -> list[dict]:
    """Roofs and peaks for *records* (run while no rank holds the GPUs),
    graded rows, and the report. Returns the rows."""
    import tb_dump
    from roofline import Roofline

    settings = settings_from(args)
    link = detect_link()
    n = records[0].traffic.world_size if records else 2
    roofline, roofs, rccl_roofs, peaks = None, {}, {}, {}
    if not (args.no_roof or args.tb_dry_run):
        roofline = Roofline(args.tb_bin, settings)
        roofs = roofline.measure(r.traffic for r in records if r.impl != RCCL)
        rccl_roofs = roofline.measure_rccl(r.traffic for r in records if r.impl == RCCL)
        peaks = roofline.measure_peaks(n)
    tb_dir = Path(args.tb_dir or tb_dump.default_dir(args.report, args.csv))
    configs = tb_dump.write(
        tb_dir, records, settings, roofs, rccl_roofs, roofline, peaks=peaks
    )
    print(f"TransferBench configs: {configs}", flush=True)
    peak = peaks["all_peers"].gbps if "all_peers" in peaks else None
    rows = grade(records, roofs, rccl_roofs, peak)
    extra = [*header, *peak_lines(records, peaks, link, settings)]
    head = provenance(roofline, extra, settings=settings, configs=configs)
    summ = summary(records, peaks, link, settings)
    write(
        rows,
        head,
        title,
        md=args.report,
        csv_path=args.csv,
        sections=[
            "## Per-rank bandwidth vs peak",
            "",
            _SUMMARY_NOTE,
            "",
            summary_markdown(summ),
        ],
    )
    return rows


def write(
    rows: list[dict],
    header: list[str],
    title: str,
    *,
    md=None,
    csv_path=None,
    sections: list[str] = (),
) -> str:
    """Print the report, and also write *md* and *csv_path* if given.
    *sections* are markdown lines after the table."""
    parts = [f"# {title}", "", *header, "", markdown(rows), ""]
    if sections:
        parts += [*sections, ""]
    text = "\n".join(parts)
    print(text, flush=True)
    if md:
        Path(md).parent.mkdir(parents=True, exist_ok=True)
        Path(md).write_text(text)
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
    return text
