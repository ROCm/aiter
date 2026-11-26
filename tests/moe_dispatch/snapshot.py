# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Legacy MoE dispatch snapshot: generate, check or update the golden files.

    python tests/moe_dispatch/snapshot.py --check            # compare with golden/
    python tests/moe_dispatch/snapshot.py --update           # rewrite golden/
    python tests/moe_dispatch/snapshot.py --arch gfx950 --out /tmp/snap
    python tests/moe_dispatch/snapshot.py --refresh-inputs   # re-freeze tuned tables

Runs on CPU: GPUs are hidden and dispatch-affecting environment variables are dropped
(arch_sim.prepare_process). Legacy dispatch reads a frozen copy of the tuned tables
(golden/inputs/), so tuning PRs do not change the snapshot; only dispatch logic does.
Re-freeze the inputs when new kinds of tuned rows should be covered, in a PR of its own.

One gzip-compressed JSON-lines file per simulated arch, sorted by key. A call variant
whose result equals the `base` variant is stored as "=base".

Reviewing a change: uncompressed the files are 7-9 MB per arch (26-33k lines, ~33 MB
in all), more than GitHub shows as a diff, so the files stay compressed and the review
reads the key-level diff this script prints: `--check` against the golden files,
`--update` against the files it replaces, every changed, new and missing key with the
golden and the new record. A PR that changes dispatch pastes the `--update` output into
its description. `git diff` shows the files decompressed (`.gitattributes` marks them
`diff=gzip`) once `git config diff.gzip.textconv "gzip -dc"` is set. On a merge
conflict, take either side and rerun `--update`.

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.
"""

import argparse
import gzip
import io
import json
import pathlib
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve().parent
GOLDEN_DIR = _HERE / "golden"
INPUTS_DIR = GOLDEN_DIR / "inputs"
FORMAT_VERSION = 1
FROZEN_TABLES = {
    "AITER_CONFIG_FMOE": "tuned_fmoe.csv",
    "AITER_CONFIG_FHMOE": "tuned_fhmoe.csv",
}
SAME_AS_BASE = "=base"


def _setup(frozen=True):
    # `tests` so `moe_dispatch` imports, and the repository root so `aiter` is
    # this checkout rather than an installed copy, even one earlier on the path.
    front = [str(_HERE.parent), str(_HERE.parents[1])]
    sys.path[:] = front + [p for p in sys.path if p not in front]
    from moe_dispatch import arch_sim

    tables = None
    if frozen:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="moe_dispatch_inputs_"))
        tables = {}
        for env_name, file_name in FROZEN_TABLES.items():
            target = tmp / file_name
            target.write_bytes(
                gzip.decompress((INPUTS_DIR / f"{file_name}.gz").read_bytes())
            )
            tables[env_name] = target
    arch_sim.prepare_process(tables)


def _variant_label(label, variant):
    head, current, rest = label.split("|", 2)
    return f"{head}|{variant}|{rest}" if current != variant else label


def dispatch_keys(gfx, cu_num):
    """`get_2stage_cfgs` keys for one arch; call inside `arch_sim.simulated_arch`."""
    from aiter.utility import dtypes
    from moe_dispatch import keys

    table = keys.tuned_table()
    base = (
        keys.tuned_keys(gfx, cu_num, dtypes.fp8, table)
        + keys.fhmoe_keys(gfx, cu_num, dtypes.fp8)
        + keys.grid_keys(dtypes.fp8, table)
    )
    return base + keys.feature_keys(base, dtypes.fp8)


def resolve_same_as_base(records, label):
    rec = records.get(label)
    return records.get(_variant_label(label, "base")) if rec == SAME_AS_BASE else rec


def generate(arch):
    """Return {label: record} for one simulated arch."""
    import logging

    import aiter.fused_moe as fm
    from moe_dispatch import arch_sim, keys, record

    logging.getLogger("aiter").setLevel(logging.ERROR)
    out = {}
    with arch_sim.simulated_arch(arch) as (gfx, cu_num):
        from aiter.utility import dtypes

        arch_fp8 = dtypes.fp8
        for key in dispatch_keys(gfx, cu_num):
            label = key.label()
            if label in out:
                continue
            # Errors are dispatch results too: record them, do not raise.
            try:
                metadata = fm.get_2stage_cfgs(*key.args, **key.call_kwargs())
                out[label] = record.normalize(metadata)
            except Exception as exc:  # noqa: BLE001
                out[label] = record.error_record(exc)
        for kw in keys.activation_dtype_keys(arch_fp8):
            label = "act_dtype|" + "|".join(
                f"{k}={keys._short(v)}" for k, v in kw.items()
            )
            try:
                out[label] = record.normalize(fm.resolve_activation_dtype(**kw))
            except Exception as exc:  # noqa: BLE001
                out[label] = record.error_record(exc)
    for label, rec in list(out.items()):
        if label.split("|")[1] not in ("base", "") and not label.startswith(
            "act_dtype"
        ):
            base = out.get(_variant_label(label, "base"))
            if base is not None and base == rec:
                out[label] = SAME_AS_BASE
    return out


def golden_path(arch, directory=GOLDEN_DIR):
    return pathlib.Path(directory) / f"{arch}.jsonl.gz"


def _write_gz(path, data):
    # No file name or mtime in the gzip header, so the bytes depend on `data` only and
    # not on the path written through.
    with open(path, "wb") as raw, gzip.GzipFile(
        filename="", fileobj=raw, mode="wb", mtime=0
    ) as f:
        f.write(data)


def write(arch, records, directory=GOLDEN_DIR):
    path = golden_path(arch, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = io.StringIO()
    text.write(json.dumps({"format": FORMAT_VERSION, "arch": arch}) + "\n")
    for label in sorted(records):
        text.write(
            json.dumps({"key": label, "rec": records[label]}, sort_keys=True) + "\n"
        )
    _write_gz(path, text.getvalue().encode())
    return path


def read(arch, directory=GOLDEN_DIR):
    path = golden_path(arch, directory)
    with gzip.open(path, "rt") as f:
        header = json.loads(f.readline())
        if header.get("format") != FORMAT_VERSION:
            raise ValueError(
                f"{path}: format {header.get('format')} != {FORMAT_VERSION}"
            )
        return {(line := json.loads(raw))["key"]: line["rec"] for raw in f}


def diff(expected, actual):
    """Human-readable differences, every key; empty list when identical."""
    lines = []
    counts = {"changed": 0, "new": 0, "missing": 0}
    for label in sorted(set(expected) | set(actual)):
        if label not in actual:
            counts["missing"] += 1
            lines.append(f"- missing: {label}")
            lines.append(f"    golden: {json.dumps(expected[label], sort_keys=True)}")
        elif label not in expected:
            counts["new"] += 1
            lines.append(
                f"+ new:     {label}  {json.dumps(actual[label], sort_keys=True)}"
            )
        elif expected[label] != actual[label]:
            counts["changed"] += 1
            lines.append(f"~ changed: {label}")
            lines.append(f"    golden: {json.dumps(expected[label], sort_keys=True)}")
            lines.append(f"    now:    {json.dumps(actual[label], sort_keys=True)}")
    if lines:
        lines.insert(0, ", ".join(f"{n} {kind}" for kind, n in counts.items()))
    return lines


def refresh_inputs():
    """Freeze the current merged runtime tuned tables as the snapshot inputs."""
    from aiter.jit.core import AITER_CONFIGS

    sources = {
        "AITER_CONFIG_FMOE": AITER_CONFIGS.AITER_CONFIG_FMOE_FILE,
        "AITER_CONFIG_FHMOE": AITER_CONFIGS.AITER_CONFIG_FHMOE_FILE,
    }
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for env_name, file_name in FROZEN_TABLES.items():
        data = pathlib.Path(sources[env_name]).read_bytes()
        target = INPUTS_DIR / f"{file_name}.gz"
        _write_gz(target, data)
        print(f"{env_name}: {sources[env_name]} -> {target}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", action="append", help="simulated arch (default: all)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="compare with golden files")
    mode.add_argument("--update", action="store_true", help="rewrite golden files")
    mode.add_argument(
        "--refresh-inputs", action="store_true", help="re-freeze tuned tables"
    )
    parser.add_argument("--out", help="write snapshots to this directory instead")
    args = parser.parse_args(argv)

    if args.refresh_inputs:
        _setup(frozen=False)
        refresh_inputs()
        return 0

    _setup()
    from moe_dispatch import arch_sim

    archs = args.arch or list(arch_sim.ARCHS)
    failed = False
    for arch in archs:
        records = generate(arch)
        if args.check:
            lines = diff(read(arch), records)
            status = f"DIFFERENT: {lines[0]}" if lines else "identical"
            print(f"{arch}: {len(records)} keys, {status}")
            for line in lines[1:]:
                print("  " + line)
            failed |= bool(lines)
        else:
            directory = args.out or GOLDEN_DIR
            previous, status = None, "new file"
            if golden_path(arch, directory).exists():
                try:
                    previous, status = read(arch, directory), "unchanged"
                except (ValueError, KeyError, TypeError, OSError, EOFError) as error:
                    status = f"replaced an unreadable file ({error})"
            path = write(arch, records, directory)
            lines = diff(previous, records) if previous is not None else []
            status = lines[0] if lines else status
            print(f"{arch}: {len(records)} keys -> {path}, {status}")
            for line in lines[1:]:
                print("  " + line)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
