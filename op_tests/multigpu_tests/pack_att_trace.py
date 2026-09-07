#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pack a WaveScope/rocprofv3 ATT run into the smallest tarball that still analyses.

A finished run is mostly ballast. On a measured MI350P capture of the qr_int4
M=1024 kernel the run directory was 1.1 GB, of which 33 MB was needed and 5 MB
survived gzip -- a 200x saving, because the raw ``.att`` files and the embedded
code objects are only there to re-run the decoder, and ``wstates*.json`` only
feeds the viewer's timeline.

What is kept, and why each is load-bearing:

    wavescope-run.json      the run manifest. Without it the dispatch still
                            loads, but its workload string and profile binding
                            are withheld rather than guessed, so the trace shows
                            up unlabelled.
    code.json               the instruction listing. Annotation and analysis
                            indices are positions in it.
    filenames.json          maps (se, sm, slot, wv) to wave files. Drop a wave
                            file it references and the trace stops satisfying
                            WaveScope's shape contract.
    se*_wv*.json            the measurements themselves.
    wavescope-trace.json    the sentinel: listing hash and measurement digest.
                            Without it provenance is computed rather than
                            managed, and the labels above are withheld.
    occupancy.json          wave begin/end intervals.
    realtime.json           gfx-clock/realtime pairs -- the only way to turn
                            cycles into a duration.
    snapshots.json          source-file mapping.
    source_*                the kernel source as it was compiled, so hot lines
                            resolve to real text.

What is dropped:

    <agent-dir>/            the raw capture: ``*.att`` plus
                            ``*_code_object_id_*.out``. On the same measured run
                            this was 1.0 GB of the 1.1 GB. Needed only to decode
                            again, never to read a decode.
    wstates*.json           viewer timeline state. Verified: with these removed,
                            `att brief/summary/top/waves` all still work and the
                            trace still reports `identity: managed` and
                            `measurements verified`.
    capture.log             the profiler's stdout.
    .wavescope-run-owner    a host-local lock.

Individual wave files are never dropped even though they dominate what is left:
``filenames.json`` names them, and a missing one costs the measurement digest
and with it the verified provenance. Compress instead -- they are numeric arrays
and gzip 6.5x.

USAGE

    python3 pack_att_trace.py <run-dir>                 # pack every usable dispatch
    python3 pack_att_trace.py <run-dir> --dispatch 706  # just one
    python3 pack_att_trace.py <run-dir> --list          # show, pack nothing
    python3 pack_att_trace.py <dispatch-dir>            # a bare decode folder

Writes ``<run-id>.att.tar.gz`` in the current directory unless ``-o`` says
otherwise. Unpack on the far side with

    tar xzf <file>.att.tar.gz -C <workspace>/.wavescope/runs/

Python 3.8+, standard library only, so it runs on whatever box holds the trace.
"""

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

# Dropped outright. Everything else in a dispatch directory is kept, so a decoder
# that starts emitting a new artifact is included by default rather than silently
# lost -- the failure mode of an allowlist here is a trace that will not load on
# the far side, which is expensive to discover.
DROP_PREFIXES = ("wstates",)
DROP_NAMES = {"capture.log", ".wavescope-run-owner"}

RUN_MANIFEST = "wavescope-run.json"
DISPATCH_GLOB = "ui_output_agent_*_dispatch_*"
# A decode folder without these is not a trace WaveScope will load.
REQUIRED = ("code.json", "filenames.json")


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def dir_size(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def keep_file(p: Path) -> bool:
    if p.name in DROP_NAMES:
        return False
    return not any(p.name.startswith(pre) for pre in DROP_PREFIXES)


def wave_files_referenced(dispatch: Path):
    """Wave filenames ``filenames.json`` points at, or None if unreadable.

    Used to refuse a pack that would ship a dispatch whose measurements are
    incomplete: the far side would load it, hash it, and report the provenance
    as unverified with no indication why.
    """
    fn = dispatch / "filenames.json"
    if not fn.is_file():
        return None
    try:
        doc = json.loads(fn.read_text())
    except (OSError, ValueError):
        return None
    names = set()

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            # leaf is [filename, begin, end]
            if node and isinstance(node[0], str):
                names.add(node[0])
            else:
                for v in node:
                    walk(v)

    walk(doc.get("wave_filenames") or {})
    return names


def describe(dispatch: Path):
    """(waves, kernel, ordinal, problems) for one decode folder."""
    problems = []
    for req in REQUIRED:
        if not (dispatch / req).is_file():
            problems.append(f"missing {req}")
    code = dispatch / "code.json"
    if code.is_file() and code.stat().st_size < 1024:
        problems.append(
            f"code.json is only {code.stat().st_size} B -- the decode produced no "
            "instruction listing, so this dispatch traced nothing"
        )
    present = {p.name for p in dispatch.glob("se*_wv*.json")}
    if not present:
        problems.append("no se*_wv*.json wave files -- nothing was captured")
    referenced = wave_files_referenced(dispatch)
    if referenced is not None:
        missing = referenced - present
        if missing:
            problems.append(
                f"{len(missing)} wave file(s) named by filenames.json are absent "
                f"(e.g. {sorted(missing)[0]})"
            )
    kernel, ordinal = None, None
    sentinel = dispatch / "wavescope-trace.json"
    if sentinel.is_file():
        try:
            doc = json.loads(sentinel.read_text())
            kernel, ordinal = doc.get("kernel"), doc.get("ordinal")
        except (OSError, ValueError):
            problems.append("wavescope-trace.json does not parse")
    else:
        problems.append(
            "no wavescope-trace.json -- provenance will be computed, not managed"
        )
    return len(present), kernel, ordinal, problems


def resolve(target: Path):
    """(run_dir_or_None, [dispatch dirs]) for a run directory or a bare decode."""
    if not target.is_dir():
        sys.exit(f"not a directory: {target}")
    dispatches = sorted(target.glob(DISPATCH_GLOB))
    if dispatches:
        return target, dispatches
    # A dispatch directory pointed at directly.
    if (target / "code.json").is_file():
        return (target.parent if (target.parent / RUN_MANIFEST).is_file() else None), [
            target
        ]
    sys.exit(
        f"{target} is neither a run directory (no {DISPATCH_GLOB} inside) nor a "
        "decode folder (no code.json)"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run", nargs="?", help="WaveScope run directory, or one dispatch")
    ap.add_argument("-o", "--output", help="tarball path (default <run-id>.att.tar.gz)")
    ap.add_argument(
        "-d", "--dispatch", action="append", default=[],
        help="ordinal to include; repeatable. Default: every usable dispatch.",
    )
    ap.add_argument(
        "--list", action="store_true", help="report what would be packed, pack nothing"
    )
    ap.add_argument(
        "--keep-empty", action="store_true",
        help="pack dispatches that captured no waves. They cannot be analysed; "
             "this exists for diagnosing a failed capture remotely.",
    )
    args = ap.parse_args(argv)

    # CI runs `python3 <file>` over everything under multigpu_tests/, so a bare
    # invocation has to be a successful no-op rather than a usage error.
    if not args.run:
        ap.print_usage()
        print("\nno run directory given; nothing to do")
        return 0

    target = Path(args.run).expanduser().resolve()
    run_dir, dispatches = resolve(target)

    if args.dispatch:
        want = {str(d) for d in args.dispatch}
        dispatches = [d for d in dispatches if d.name.rsplit("_", 1)[-1] in want]
        if not dispatches:
            sys.exit(f"no dispatch matching {sorted(want)} under {target}")

    run_id = (run_dir or target).name
    print(f"run: {run_id}")
    if run_dir and (run_dir / RUN_MANIFEST).is_file():
        try:
            doc = json.loads((run_dir / RUN_MANIFEST).read_text())
            print(f"  profile:  {doc.get('profileId')}")
            print(f"  workload: {doc.get('workload')}")
        except (OSError, ValueError):
            print(f"  {RUN_MANIFEST} does not parse; packing it anyway")
    else:
        print(f"  no {RUN_MANIFEST} -- the workload label will be withheld on load")

    members = []          # (absolute path, name inside the archive)
    kept_bytes = 0
    usable = 0

    if run_dir and (run_dir / RUN_MANIFEST).is_file():
        p = run_dir / RUN_MANIFEST
        members.append((p, f"{run_id}/{RUN_MANIFEST}"))
        kept_bytes += p.stat().st_size

    print()
    for d in dispatches:
        waves, kernel, ordinal, problems = describe(d)
        files = [p for p in sorted(d.iterdir()) if p.is_file() and keep_file(p)]
        size = sum(p.stat().st_size for p in files)
        full = dir_size(d)
        label = f"  {d.name}"
        detail = f"kernel={kernel} ordinal={ordinal} waves={waves}"
        fatal = waves == 0 or any(m.startswith("missing") for m in problems)
        if fatal and not args.keep_empty:
            print(f"{label}\n      SKIP  {detail}")
            for m in problems:
                print(f"            - {m}")
            continue
        usable += 1
        print(f"{label}\n      pack  {detail}")
        print(f"            {human(full)} on disk -> {human(size)} kept "
              f"({human(full - size)} dropped)")
        for m in problems:
            print(f"            ! {m}")
        for p in files:
            members.append((p, f"{run_id}/{d.name}/{p.name}"))
        kept_bytes += size

    if not usable and not args.keep_empty:
        print("\nnothing usable to pack. Every dispatch here traced no waves --")
        print("usually the capture landed on dispatches too small to reach an")
        print("instrumented CU. Re-capture so the first matching dispatch is the")
        print("real geometry, or pass --keep-empty to ship it for diagnosis.")
        return 1

    print(f"\ntotal to pack: {human(kept_bytes)} in {len(members)} file(s)")
    if args.list:
        return 0

    out = Path(args.output) if args.output else Path(f"{run_id}.att.tar.gz")
    with tarfile.open(out, "w:gz") as tf:
        for src, name in members:
            tf.add(src, arcname=name)
    packed = out.stat().st_size
    ratio = kept_bytes / packed if packed else 0
    print(f"wrote {out}  ({human(packed)}, {ratio:.1f}x)")
    print(f"\nunpack with:\n  tar xzf {out.name} -C <workspace>/.wavescope/runs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
