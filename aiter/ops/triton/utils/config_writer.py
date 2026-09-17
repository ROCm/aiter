# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Write side of the config machinery: the atomic, order-stable JSON writer
that tuning tools go through to publish a measured winner.

Reads stay in ``config_utils`` (``resolve_config_dir`` + ``load_config_json``).
This module is the only supported way to *modify* a file in the config tree,
and it exists because three properties have to hold together:

- **Atomic.** A tuning run that dies mid-write must not leave a truncated
  config behind; the next process would read a corrupt tree.
- **Order-stable.** Publishing one winner must produce a one-entry diff, not a
  reshuffled file. Keys already in the file never move.
- **Cache-coherent.** ``load_config_json`` caches per path *including negative
  results*, so a freshly written entry is invisible until the cache is cleared.

Validation happens on write, not at launch: an entry with an unknown key or a
missing required key is rejected here, where the tuning run can still report
it, rather than surfacing as a confusing kernel-launch failure days later.
"""

import json
import os
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from aiter.ops.triton.utils.config_utils import load_config_json
from aiter.ops.triton.utils.logger import AiterTritonLogger

logger = AiterTritonLogger()

__all__ = [
    "invalidate_config_cache",
    "update_config_entry",
    "validate_config_entry",
    "write_config_json",
]


def invalidate_config_cache(fpath: str | None = None) -> None:
    """Drop cached config reads so a just-written file is visible in-process.

    ``functools.lru_cache`` has no per-key eviction, so the whole cache is
    cleared regardless of ``fpath``; the argument documents the caller's
    intent and is used for logging only.
    """
    load_config_json.cache_clear()
    if fpath is not None:
        logger.debug(f"config cache invalidated after writing {fpath}")


def validate_config_entry(
    entry: Mapping[str, Any],
    required: Sequence[str],
    optional: Iterable[str] = (),
    where: str = "config entry",
) -> dict[str, Any]:
    """Return ``entry`` as a plain dict, rejecting unknown or missing keys.

    A config file must fully describe its launch: the loaders deliberately do
    not backfill defaults in Python, so a partial entry is a bug that has to
    fail at write time. Unknown keys are rejected for the same reason -- they
    read as tuned values but no kernel consumes them.
    """
    if not isinstance(entry, Mapping):
        raise TypeError(f"{where} must be a mapping, got {type(entry).__name__}")
    allowed = set(required) | set(optional)
    unknown = sorted(key for key in entry if key not in allowed)
    if unknown:
        raise ValueError(
            f"{where} carries keys no kernel reads: {unknown}; "
            f"allowed keys are {sorted(allowed)}"
        )
    missing = sorted(key for key in required if key not in entry)
    if missing:
        raise ValueError(
            f"{where} is missing required keys: {missing}; a config file must "
            "fully describe its launch (no Python backfill)"
        )
    return dict(entry)


def _assert_json_safe(payload: Any, where: str) -> None:
    """Reject values JSON round-trips badly, before anything touches disk."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if not isinstance(key, str):
                raise TypeError(f"{where}: config keys must be strings, got {key!r}")
            _assert_json_safe(value, f"{where}.{key}")
        return
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _assert_json_safe(value, f"{where}[{index}]")
        return
    if isinstance(payload, float) and payload != payload:
        raise ValueError(f"{where}: NaN is not representable in JSON")
    if isinstance(payload, float) and payload in (float("inf"), float("-inf")):
        raise ValueError(f"{where}: infinity is not representable in JSON")
    if payload is not None and not isinstance(payload, (str, int, float, bool)):
        raise TypeError(f"{where}: {type(payload).__name__} is not JSON-serializable")


def write_config_json(fpath: str, payload: Mapping[str, Any]) -> None:
    """Atomically replace a config file with ``payload``.

    The file is written to a temporary sibling and moved into place with
    ``os.replace``, so a reader either sees the old file or the new one and
    never a partial write. Formatting matches the checked-in tree (2-space
    indent, trailing newline) so the diff is reviewable.
    """
    _assert_json_safe(payload, os.path.basename(fpath))
    directory = os.path.dirname(os.path.abspath(fpath))
    os.makedirs(directory, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        dir=directory, prefix=".config-", suffix=".json.tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.chmod(staged, 0o644)
        os.replace(staged, fpath)
    except BaseException:
        if os.path.exists(staged):
            os.unlink(staged)
        raise
    invalidate_config_cache(fpath)


def _insert_ordered(container: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    """Set ``key`` without disturbing the order of keys already in the file.

    A container that is already sorted stays sorted, so a generated table
    (``shapes``, hardware buckets) grows deterministically no matter what
    order the tuning run happened to measure in. A hand-ordered container
    (``fwd``, ``bkwd_fused``) keeps its authored order and takes an append.
    """
    if key in container:
        container[key] = value
        return container
    existing = list(container)
    if existing == sorted(existing):
        merged = {**container, key: value}
        return {name: merged[name] for name in sorted(merged)}
    container[key] = value
    return container


def update_config_entry(
    fpath: str,
    keypath: Sequence[str],
    entry: Any,
) -> None:
    """Read-modify-write one nested entry, leaving every sibling untouched.

    ``keypath`` is walked from the document root, creating intermediate
    dictionaries as needed; the final element names the entry to set. One
    ``DEFAULT.json`` holds many families and many shapes, so publishing a
    winner must never rewrite the whole document.
    """
    if not keypath:
        raise ValueError("keypath must name at least one key")
    document = load_config_json(fpath, required=False)
    document = dict(document) if document else {}

    node = document
    for step in keypath[:-1]:
        child = node.get(step)
        node[step] = dict(child) if isinstance(child, Mapping) else {}
        node = node[step]

    leaf = keypath[-1]
    updated = _insert_ordered(node, leaf, entry)
    if updated is not node:
        # _insert_ordered rebuilds the container when it re-sorts, so the
        # rebuilt dict has to be spliced back into its parent.
        if len(keypath) == 1:
            document = updated
        else:
            parent = document
            for step in keypath[:-2]:
                parent = parent[step]
            parent[keypath[-2]] = updated
    write_config_json(fpath, document)
