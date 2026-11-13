# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Grade gathered records against their roofs, and write the tables.

Per op row:

* ``algbw`` is ``logical_bytes / us``. ``busbw`` is algbw times the rccl-tests
  factor: ``(N-1)/N`` for all-to-all and ``2(N-1)/N`` for all-reduce.
* ``wire GB/s`` is the bytes the op sent to other GPUs (wire format,
  ``fabric_bytes_per_rank``) over ``us``.
* ``eff`` is ``roof us / us``: 1.0 is the TransferBench roof.
* ``rccl vs roof`` is ``roof us / rccl us``, against the *op's* roof: the same
  bytes and fabric, but RCCL runs its own pattern and memory type.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

from measure import RCCL, Record
from patterns import busbw_factor, fabric_bytes_per_rank
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
    "rccl us",
    "rccl busbw GB/s",
    "rccl vs roof",
    "op spread us",
    "variant",
)


def _gbps(nbytes: float, us: float) -> float:
    return nbytes / us / 1e3 if us > 0 else math.nan


def grade(records: list[Record], roofs: dict) -> list[dict]:
    """One row per (case, op). RCCL's columns repeat on every op row of
    its case."""
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
                "rccl us": base.us if base else math.nan,
                "rccl busbw GB/s": (
                    _gbps(base.traffic.logical_bytes, base.us) * factor
                    if base
                    else math.nan
                ),
                "rccl vs roof": roof_us / base.us if base else math.nan,
                "op spread us": r.spread_us,
                "variant": r.variant or "",
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
            "---" if c in ("case", "op", "timing", "variant") else "---:"
            for c in COLUMNS
        )
        + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row[c]) for c in COLUMNS) + " |")
    return "\n".join(lines)


def provenance(roofline, extra: list[str] = ()) -> list[str]:
    """Header lines: what changes the numbers. *roofline* is None when the
    roof was skipped on request."""
    import os

    if roofline is None:
        tb = "skipped (--no-roof or --tb-dry-run)"
    elif roofline.available:
        tb = (
            f"TransferBench {roofline.version} at {roofline.binary}, "
            f"{roofline.iters} iterations, {roofline.warmup} warmup"
        )
    else:
        tb = "TransferBench not found: no roofline"
    return [
        f"- roofline: {tb}",
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
    parser.add_argument("--tb-iters", type=int, default=20)
    parser.add_argument("--tb-warmup", type=int, default=3)
    parser.add_argument(
        "--no-roof",
        action="store_true",
        help="skip TransferBench; report bandwidth only",
    )
    parser.add_argument(
        "--tb-dry-run",
        action="store_true",
        help="print the TransferBench configs (env + lines) of every roof instead\n"
        "of running them; paste into the stock binary to reproduce a roof",
    )


def dry_run(records: list[Record]) -> str:
    """The TransferBench invocations each roof would run, as shell snippets."""
    import patterns

    out = []
    for traffic in dict.fromkeys(r.traffic for r in records if r.impl != RCCL):
        cases = sorted({r.case for r in records if r.traffic == traffic})
        out.append(f"# {', '.join(cases)}")
        for i, test in enumerate(patterns.tb_tests(traffic)):
            env = " ".join(f"{k}={v}" for k, v in test.env)
            out.append(f"# phase {i}, x{test.repeat}: {env} TransferBench <cfg>")
            out.append(test.line)
    return "\n".join(out)


def finish(
    records: list[Record], *, title: str, args, header: list[str] = ()
) -> list[dict]:
    """Roofs for *records* (run while no rank holds the GPUs), graded rows,
    and the report. Returns the rows."""
    from roofline import Roofline

    if args.tb_dry_run:
        print(dry_run(records), flush=True)
        args.no_roof = True
    roofline, roofs = None, {}
    if not args.no_roof:
        roofline = Roofline(args.tb_bin, iters=args.tb_iters, warmup=args.tb_warmup)
        roofs = roofline.measure(r.traffic for r in records if r.impl != RCCL)
    rows = grade(records, roofs)
    write(
        rows,
        provenance(roofline, list(header)),
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
