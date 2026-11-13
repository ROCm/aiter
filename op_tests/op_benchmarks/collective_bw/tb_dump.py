# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Write every roof's TransferBench configs, for review and reproduction.

Every run writes them, whether TransferBench ran or not::

    <dir>/README.md                       index: every candidate, its time, the roof
    <dir>/<case>/<op>/<candidate>.cfg     the op's roof candidates
    <dir>/<case>/rccl/<candidate>.cfg     the RCCL roof candidates
    <dir>/peak/<kind>/<candidate>.cfg     the fabric-peak probes

Each ``.cfg`` runs as is with the stock binary; its header has the
environment and the command. A candidate whose tests need different
environments (block sizes) is split into ``<candidate>.<i>.cfg``.
"""

from __future__ import annotations

import logging
import re
import shutil
from itertools import groupby
from pathlib import Path

import patterns
from measure import RCCL, Record
from roof_settings import RoofSettings

logger = logging.getLogger("aiter")

# Marks a directory this module wrote, so a rerun may clear stale files.
MARKER = ".collective_bw_tb"

_TIME_RULE = (
    "A candidate's time is the sum over its tests of repeat x the slowest\n"
    '"Transfer" row (timed inside the kernel). A roof is its fastest candidate.'
)


def default_dir(report: str | None, csv: str | None) -> Path:
    """Next to the report (``x.md`` -> ``x.tb/``), else next to the CSV,
    else ``./collective_bw_tb``."""
    for path in (report, csv):
        if path:
            return Path(path).with_suffix(".tb")
    return Path("collective_bw_tb")


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", text).strip("_") or "_"


def _env_str(env) -> str:
    return " ".join(f"{k}={v}" for k, v in env)


def _prepare(root: Path) -> None:
    if (root / MARKER).exists():
        shutil.rmtree(root)
    elif root.exists() and any(root.iterdir()):
        logger.warning(
            "%s is not empty and was not written by this tool: writing the "
            "TransferBench configs into it without clearing old files",
            root,
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER).touch()


def _write_candidate(folder: Path, title: str, cand, us: float | None, best: bool):
    """Write the candidate's ``.cfg`` file(s) into *folder*. Returns their
    names. *us* is the measured time, None if TransferBench did not run."""
    groups = [list(g) for _env, g in groupby(cand.tests, key=lambda t: t.env)]
    names = []
    for i, tests in enumerate(groups):
        suffix = "" if len(groups) == 1 else f".{i}"
        name = f"{slug(cand.label)}{suffix}.cfg"
        if us is None:
            measured = "not measured"
        else:
            measured = f"{us:.3f} us" + (" (the roof)" if best else "")
        part = f", part {i + 1} of {len(groups)}" if len(groups) > 1 else ""
        lines = [
            f"## collective_bw roof candidate: {title} / {cand.label}{part}",
            f"## measured: {measured}",
            *(f"# {s}" for s in _TIME_RULE.splitlines()),
            "# Run:",
            f"#   {_env_str(tests[0].env)} TransferBench {name}",
        ]
        for k, test in enumerate(tests):
            lines += [f"# test {k + 1}: counts x{test.repeat}", test.line]
        (folder / name).write_text("\n".join(lines) + "\n")
        names.append(name)
    return names


def _write_roof(root: Path, rel: Path, case: str, roof_name: str, cands, roof):
    folder = root / rel
    folder.mkdir(parents=True, exist_ok=True)
    times = dict(roof.candidates) if roof else {}
    rows = []
    for cand in cands:
        us = times.get(cand.label)
        best = roof is not None and cand.label == roof.label
        names = _write_candidate(folder, f"{case} / {roof_name}", cand, us, best)
        rows.append((case, roof_name, cand.label, us, best, [rel / n for n in names]))
    return rows


def write(
    root: Path,
    records: list[Record],
    settings: RoofSettings,
    roofs: dict,
    rccl_roofs: dict,
    roofline=None,
    *,
    peaks: dict | None = None,
) -> Path:
    """Write the configs of every op and RCCL roof of *records*, and of the
    fabric-peak probes, under *root*, with the measured times where *roofs*
    / *rccl_roofs* / *peaks* have them. Returns the index path."""
    peaks = peaks or {}
    root = Path(root)
    _prepare(root)
    rows = []
    for r in records:
        if r.impl == RCCL:
            continue
        rows += _write_roof(
            root,
            Path(slug(r.case)) / slug(r.impl),
            r.case,
            f"op {r.impl}",
            patterns.candidates(r.traffic, settings),
            roofs.get(r.traffic),
        )
    for r in records:
        if r.impl != RCCL:
            continue
        rt = patterns.rccl_traffic(r.traffic)
        rows += _write_roof(
            root,
            Path(slug(r.case)) / "rccl",
            r.case,
            "rccl",
            patterns.rccl_candidates(rt, settings),
            rccl_roofs.get(rt),
        )
    if records:
        n = records[0].traffic.world_size
        for kind in patterns.PEAK_KINDS:
            probe = patterns.peak_traffic(n, kind, settings)
            pk = peaks.get(kind)
            rows += _write_roof(
                root,
                Path("peak") / kind,
                "-",
                f"peak {kind}",
                patterns.sweep_candidates(probe, settings),
                pk.roof if pk else None,
            )
    index = root / "README.md"
    index.write_text(_index(rows, settings, roofline))
    return index


def _index(rows, settings: RoofSettings, roofline) -> str:
    if roofline is None:
        tb = "not run (--no-roof or --tb-dry-run)"
    elif roofline.available:
        tb = f"{roofline.version} at {roofline.binary}"
    else:
        tb = "not found: no roofline"
    out = [
        "# TransferBench configs of every roof",
        "",
        f"- TransferBench: {tb}",
        f"- settings: `{settings}`",
        f"- fixed: `{_env_str(patterns.FIXED_ENV)}`",
        "- the tool also sets `OUTPUT_TO_CSV=0 HIDE_ENV=1` for its parser",
        "",
        *_TIME_RULE.splitlines(),
        "Each file's header has its environment and command.",
        "",
        "| case | roof | candidate | us | | files |",
        "|---|---|---|---:|---|---|",
    ]
    for case, roof_name, label, us, best, files in rows:
        us_s = "" if us is None else f"{us:.3f}"
        links = " ".join(f"[{f.name}]({f.as_posix()})" for f in files)
        mark = "**roof**" if best else ""
        out.append(f"| {case} | {roof_name} | {label} | {us_s} | {mark} | {links} |")
    return "\n".join(out) + "\n"
