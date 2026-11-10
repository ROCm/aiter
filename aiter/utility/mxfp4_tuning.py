# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public, read-only source selection for A4W4 GEMM tuning profiles.

Configure before the first A4W4 lookup. Explicit AITER_CONFIG_GEMM_A4W4 paths
replace profiles; within a source list the first row for a shape wins. Kernel
selection and measurements are never rewritten in the source CSVs.

The ``--emit`` option keeps stdout to one value (``path``, ``fingerprint``
``or ``sources``) so shell callers never parse log lines.
"""

import argparse
import csv
import functools
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import threading
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_PROFILE_FILE = _CONFIG_DIR / "mxfp4_profiles.json"
_DEFAULT_FILE = _CONFIG_DIR / "mxfp4_default_sources.json"
_MODEL_DIR = _CONFIG_DIR / "model_configs"
_ENV = "AITER_CONFIG_GEMM_A4W4"
_KEYS = ("gfx", "cu_num", "M", "N", "K")
_REQUIRED = ("cu_num", "M", "N", "K", "kernelName", "splitK")
_configured_sources = None
_frozen_sources = None
_resolved_file = None
_frozen_fingerprint = None
_effective_fingerprint = None
_configured_power_sources = None
_frozen_power_sources = None
_lock = threading.RLock()


@functools.lru_cache(maxsize=1)
def _power_helpers():
    # Load the shared stdlib policy without bootstrapping the aiter package.
    spec = importlib.util.spec_from_file_location(
        "_mxfp4_power_policy", Path(__file__).with_name("tune_result.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _power_sources():
    if os.environ.get(_ENV) or not _power_helpers().power_selection_enabled():
        return ()
    return tuple(
        str(p.resolve())
        for p in sorted(
            (_CONFIG_DIR / _power_helpers().POWER_CONFIG_DIRNAME).glob(
                "*a4w4_blockscale_tuned_gemm*.csv"
            )
        )
        if p.is_file() and "untuned" not in p.name
    )


def _power_policy(sources):
    if sources == _frozen_sources:
        return _frozen_power_sources
    if sources == _configured_sources:
        return _configured_power_sources
    return _power_sources()


def _reset_mxfp4_tuning_for_tuner():
    """Private tuner hook for sequential baseline/candidate measurements."""
    global _configured_sources, _frozen_sources, _resolved_file
    global _frozen_fingerprint, _effective_fingerprint
    global _configured_power_sources, _frozen_power_sources
    with _lock:
        _configured_sources = _frozen_sources = _resolved_file = None
        _frozen_fingerprint = _effective_fingerprint = None
        _configured_power_sources = _frozen_power_sources = None


def _locked(function):
    @functools.wraps(function)
    def synchronized(*args, **kwargs):
        with _lock:
            return function(*args, **kwargs)

    return synchronized


def available_mxfp4_profiles():
    """Return bundled case names; listing profiles does not initialize GEMM."""
    with _PROFILE_FILE.open(encoding="utf-8") as stream:
        return tuple(json.load(stream))


def _profile_sources(profile):
    with _PROFILE_FILE.open(encoding="utf-8") as stream:
        profiles = json.load(stream)
    if profile not in profiles:
        raise ValueError(
            f"Unknown MXFP4 tuning profile {profile!r}; choose {tuple(profiles)}"
        )
    return [_MODEL_DIR / name for name in profiles[profile]]


def _default_sources():
    # Keep existing tuned coverage; new model tables require explicit selection.
    with _DEFAULT_FILE.open(encoding="utf-8") as stream:
        return [_CONFIG_DIR / name for name in json.load(stream)]


def resolve_mxfp4_tuning_sources(profile=None, tuned_config=None, merge_default=True):
    """Preview ordered table paths without changing the process configuration."""
    if profile is not None and tuned_config is not None:
        raise ValueError(
            "Choose either a tuning profile or explicit tuned_config paths"
        )
    override = os.environ.get(_ENV)
    if override:
        paths = [Path(path) for path in override.split(os.pathsep) if path]
    else:
        paths = [Path(path) for path in _power_sources()]
        paths.extend(_profile_sources(profile) if profile is not None else [])
        if tuned_config is not None:
            inputs = (
                [tuned_config]
                if isinstance(tuned_config, (str, os.PathLike))
                else tuned_config
            )
            paths.extend(Path(path) for path in inputs)
        if merge_default:
            paths.extend(_default_sources())
    if not paths:
        raise ValueError("MXFP4 tuning requires at least one source table")
    sources = tuple(dict.fromkeys(str(path.resolve()) for path in paths))
    for path in sources:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    return sources


def _fingerprint(sources, contents=None):
    digest = hashlib.sha256()
    if contents is None:
        contents = tuple(Path(source).read_bytes() for source in sources)
    for source, content in zip(sources, contents):
        digest.update(Path(source).name.encode())
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


@_locked
def mxfp4_tuning_sources():
    """Return the actual ordered input tables, including any bundled defaults."""
    if _frozen_sources is not None:
        return _frozen_sources
    return (
        _configured_sources
        if _configured_sources is not None
        else resolve_mxfp4_tuning_sources()
    )


@_locked
def configure_mxfp4_tuning(profile=None, tuned_config=None, merge_default=True):
    """Select a case or explicit paths before lookup; reject changes after lookup.

    The explicit environment path list wins over the requested profile. Repeating
    an identical configuration is safe; source content changes after lookup are
    rejected rather than silently disagreeing with cached kernel choices.
    """
    global _configured_sources, _configured_power_sources
    sources = resolve_mxfp4_tuning_sources(profile, tuned_config, merge_default)
    fingerprint = _fingerprint(sources)
    power_sources = _power_sources()
    if _frozen_sources is not None and (
        sources != _frozen_sources
        or fingerprint != _frozen_fingerprint
        or power_sources != _frozen_power_sources
    ):
        raise RuntimeError(
            "A4W4 tuning is already resolved; configure it before the first GEMM"
        )
    _configured_sources = sources
    _configured_power_sources = power_sources
    return {
        "sources": sources,
        "fingerprint": fingerprint,
        "resolved_file": _resolved_file,
    }


def _merged_csv(sources, contents=None):
    columns = []
    selected = {}
    power_sources = _power_policy(sources)
    power_rows = {}
    if contents is None:
        contents = tuple(Path(source).read_bytes() for source in sources)
    for source, content in zip(sources, contents):
        with io.StringIO(content.decode("utf-8-sig"), newline="") as stream:
            reader = csv.DictReader(stream)
            missing = set(_REQUIRED) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{source}: missing tuning columns {sorted(missing)}")
            for column in ["gfx", *(reader.fieldnames or ())]:
                if column not in columns:
                    columns.append(column)
            local = {}
            for row in reader:
                if (
                    source in power_sources
                    and _power_helpers().row_select_mode(
                        row.get(_power_helpers().SELECT_COLUMN)
                    )
                    != "power"
                ):
                    continue
                if None in row:
                    raise ValueError(f"{source}: malformed tuning row")
                if not row.get("gfx"):
                    # These historical CU counts have unambiguous legacy owners.
                    arch = {256: "gfx950", 80: "gfx942", 304: "gfx942"}.get(
                        int(row["cu_num"])
                    )
                    if arch is None:
                        raise ValueError(
                            f"{source}: legacy row needs an explicit gfx column"
                        )
                    row["gfx"] = arch
                key = (row["gfx"], *(int(row[column]) for column in _KEYS[1:]))
                if any(value <= 0 for value in key[1:]):
                    raise ValueError(f"{source}: non-positive tuning dimensions {key}")
                if not row["kernelName"] or int(row["splitK"]) < 0:
                    raise ValueError(f"{source}: invalid kernelName or splitK")
                if key in local and local[key] != row:
                    raise ValueError(f"{source}: conflicting rows for {key}")
                if source in power_sources:
                    if key in power_rows and power_rows[key] != row:
                        raise ValueError(f"{source}: conflicting power rows for {key}")
                    power_rows[key] = row
                local[key] = row
                selected.setdefault(key, row)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(selected.values())
    return stream.getvalue().encode("utf-8")


@_locked
def get_mxfp4_config_file():
    """Resolve one immutable derived CSV consumed by the existing public GEMM.

    Only the derived cache is written. Content-addressed paths prevent different
    jobs or profiles from overwriting one shared `/tmp` merge result.
    """
    global _frozen_sources, _frozen_fingerprint, _resolved_file, _effective_fingerprint
    global _frozen_power_sources
    if _resolved_file is not None:
        return _resolved_file
    sources = mxfp4_tuning_sources()
    contents = tuple(Path(source).read_bytes() for source in sources)
    payload = _merged_csv(sources, contents)
    fingerprint = _fingerprint(sources, contents)
    directory = Path(tempfile.gettempdir()) / "aiter_configs" / "mxfp4"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (hashlib.sha256(payload).hexdigest() + ".csv")
    if not target.exists() or target.read_bytes() != payload:
        fd, temporary = tempfile.mkstemp(prefix="mxfp4-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    _frozen_power_sources = _power_policy(sources)
    _frozen_sources = sources
    _frozen_fingerprint = fingerprint
    _effective_fingerprint = hashlib.sha256(payload).hexdigest()
    _resolved_file = str(target)
    return _resolved_file


@_locked
def mxfp4_tuning_fingerprint():
    """Identify effective CSV bytes; once consumed, return the frozen identity."""
    if _effective_fingerprint is not None:
        return _effective_fingerprint
    return hashlib.sha256(_merged_csv(mxfp4_tuning_sources())).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=available_mxfp4_profiles())
    parser.add_argument("--tuned-config", action="append")
    parser.add_argument(
        "--emit",
        choices=("path", "fingerprint", "sources"),
        default="path",
        help="Value printed on stdout; diagnostics stay on stderr.",
    )
    args = parser.parse_args()
    configure_mxfp4_tuning(args.profile, args.tuned_config)
    if args.emit == "fingerprint":
        print(mxfp4_tuning_fingerprint())
    elif args.emit == "sources":
        print("\n".join(mxfp4_tuning_sources()))
    else:
        print(get_mxfp4_config_file())


if __name__ == "__main__":
    main()
