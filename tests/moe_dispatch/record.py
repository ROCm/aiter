# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Turn dispatch results into deterministic, JSON-serializable records.

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import dataclasses
import enum
import functools
import re

import torch

_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
_PATH = re.compile(r"(?:/[\w.+-]+)+/([\w.+-]+)")


@functools.cache
def _legacy_names():
    """Functions moved to aiter.moev2 keep the name the legacy module exports them as."""
    import aiter.fused_moe as fm

    return {
        id(value): f"fused_moe.{name}"
        for name, value in vars(fm).items()
        if callable(value)
        and (getattr(value, "__module__", None) or "").startswith("aiter.moev2.")
    }


def callable_name(fn):
    legacy = _legacy_names().get(id(fn))
    if legacy is not None:
        return legacy
    module = getattr(fn, "__module__", None) or ""
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    if name is None:
        return type(fn).__qualname__
    module = module.removeprefix("aiter.")
    return f"{module}.{name}" if module else name


def normalize(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum) or hasattr(type(value), "__members__"):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, functools.partial):
        out = {"fn": normalize(value.func)}
        if value.args:
            out["args"] = [normalize(a) for a in value.args]
        if value.keywords:
            out["kw"] = {k: normalize(v) for k, v in sorted(value.keywords.items())}
        return out
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dataclass_record(value)
    if isinstance(value, dict):
        return {
            str(k): normalize(v)
            for k, v in sorted(value.items(), key=lambda i: str(i[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [normalize(v) for v in value]
        return sorted(items, key=str) if isinstance(value, (set, frozenset)) else items
    if callable(value):
        return callable_name(value)
    return _ADDRESS.sub("", repr(value))


def _dataclass_record(value):
    """Fields that differ from the dataclass defaults, plus the type name."""
    out = {"type": type(value).__qualname__}
    for f in dataclasses.fields(value):
        current = getattr(value, f.name)
        if f.default is not dataclasses.MISSING and current == f.default:
            continue
        out[f.name] = normalize(current)
    return out


def error_record(exc):
    message = _PATH.sub(r"\1", str(exc))
    return {"error": f"{type(exc).__name__}: {message[:400]}"}
