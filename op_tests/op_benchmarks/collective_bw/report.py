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
    "rccl roof us",
    "rccl us",
    "rccl eff",
    "rccl busbw GB/s",
    "op spread us",
    "variant",
    "roof cfg",
    "rccl roof cfg",
)

_TEXT_COLUMNS = ("case", "op", "timing", "variant", "roof cfg", "rccl roof cfg")


def _gbps(nbytes: float, us: float) -> float:
    return nbytes / us / 1e3 if us > 0 else math.nan


def grade(
    records: list[Record], roofs: dict, rccl_roofs: dict | None = None
) -> list[dict]:
    """One row per (case, op). RCCL's columns repeat on every op row of
    its case. *roofs* is keyed by the op traffic, *rccl_roofs* by
    ``rccl_traffic`` of it."""
    rccl_roofs = rccl_roofs or {}
    rccl = {r.case: r for r in records if r.impl == RCCL}
    rows = []
    for r in records:
        if r.impl == RCCL:
            continue
        t = r.traffic
        factor = busbw_factor(t.collective, t.world_size)
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
                "op busbw GB/s": _gbps(t.logical_bytes, r.us) * factor,
                "op wire GB/s": _gbps(fabric_bytes_per_rank(t), r.us),
                "rccl roof us": rroof_us,
                "rccl us": base.us if base else math.nan,
                "rccl eff": rroof_us / base.us if base else math.nan,
                "rccl busbw GB/s": (
                    _gbps(base.traffic.logical_bytes, base.us) * factor
                    if base
                    else math.nan
                ),
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
    over = {"iterations": args.tb_iters, "warmups": args.tb_warmup}
    return replace(DEFAULT, **{k: v for k, v in over.items() if v is not None})


def finish(
    records: list[Record], *, title: str, args, header: list[str] = ()
) -> list[dict]:
    """Roofs for *records* (run while no rank holds the GPUs), graded rows,
    and the report. Returns the rows."""
    import tb_dump
    from roofline import Roofline

    settings = settings_from(args)
    roofline, roofs, rccl_roofs = None, {}, {}
    if not (args.no_roof or args.tb_dry_run):
        roofline = Roofline(args.tb_bin, settings)
        roofs = roofline.measure(r.traffic for r in records if r.impl != RCCL)
        rccl_roofs = roofline.measure_rccl(r.traffic for r in records if r.impl == RCCL)
    tb_dir = Path(args.tb_dir or tb_dump.default_dir(args.report, args.csv))
    configs = tb_dump.write(tb_dir, records, settings, roofs, rccl_roofs, roofline)
    print(f"TransferBench configs: {configs}", flush=True)
    rows = grade(records, roofs, rccl_roofs)
    write(
        rows,
        provenance(roofline, list(header), settings=settings, configs=configs),
        title,
        md=args.report,
        csv_path=args.csv,
    )
    return rows


def write(
    rows: list[dict], header: list[str], title: str, *, md=None, csv_path=None
) -> str:
    """Print the report, and also write *md* and *csv_path* if given."""
    text = "\n".join([f"# {title}", "", *header, "", markdown(rows), ""])
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
