# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Migration paths: the unit that moves from the legacy MoE code to aiter.moev2.

A path claims a subset of `fused_moe` calls through a predicate over the call arguments.
The legacy entry points ask `route_to_v2(call)` which path, if any, takes a call; with no
path active this costs one cached boolean check.

Predicates run on every call, including during graph capture: they may read dtypes,
shapes and Python values, never tensor contents.
"""

import importlib
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from . import envs

logger = logging.getLogger("aiter")

# Path names agreed for the migration (design doc §13, D1). Only these are accepted in
# AITER_MOEV2, so a typo fails loudly instead of silently running the legacy code.
PLANNED = (
    "a16w16",
    "a8w8_pt",
    "a8w8_ptok",
    "a8w8_blockscale",
    "a4w4",
    "a8w4",
    "a16w4",
    "a16wi4",
    "mxfp8",
    "fhmoe",
    "comm_fused",
    "mega_moe_v2",
    "mega_moe_tp",
    "mega_moe_gfx1250",
)


@dataclass(frozen=True)
class Path:
    """One migration path.

    `claims(call)` / `run(call)`: `call` maps the legacy `fused_moe` parameter names to
    the call's arguments.
    `dispatch_record(args, kwargs)`: for a legacy `get_2stage_cfgs(*args, **kwargs)` key,
    the v2 plan in the legacy snapshot record format, or None if the path does not claim
    the key. Used by the CPU differential test.
    `examples()`: GPU call mappings for the output differential test.
    """

    name: str
    claims: Callable[[Mapping], bool]
    run: Callable[[Mapping], object]
    graduated: bool = False
    dispatch_record: Callable[[tuple, dict], object] | None = None
    examples: Callable[[], Iterable[Mapping]] | None = None


# Modules that register the built-in paths; imported on first use, not at import time.
_BUILTIN_MODULES: tuple = ()

_REGISTRY: dict[str, Path] = {}
_builtin_loaded = False


def _load_builtin():
    global _builtin_loaded
    if not _builtin_loaded:
        _builtin_loaded = True
        for module in _BUILTIN_MODULES:
            importlib.import_module(module)


def register(path: Path):
    _load_builtin()
    if path.name not in PLANNED:
        raise ValueError(f"unknown moev2 path {path.name!r}; add it to PLANNED first")
    if path.name in _REGISTRY:
        raise ValueError(f"moev2 path {path.name!r} is already registered")
    _REGISTRY[path.name] = path
    reload()


def unregister(name: str):
    _REGISTRY.pop(name, None)
    reload()


def registered():
    _load_builtin()
    return dict(_REGISTRY)


_active: tuple | None = None


def active() -> tuple:
    """Registered paths selected by AITER_MOEV2, in registration order."""
    global _active
    if _active is None:
        _active = _select()
    return _active


def _select() -> tuple:
    _load_builtin()
    selection = envs.moev2_selection()
    if selection == "off":
        return ()
    if selection == "on":
        return tuple(_REGISTRY.values())
    if selection == "default":
        return tuple(p for p in _REGISTRY.values() if p.graduated)
    unknown = sorted(selection - set(PLANNED))
    if unknown:
        raise ValueError(
            f"{envs.AITER_MOEV2} names unknown paths {unknown}; known: {', '.join(PLANNED)}"
        )
    missing = sorted(selection - set(_REGISTRY))
    if missing:
        logger.warning(
            "%s: paths %s are not implemented yet", envs.AITER_MOEV2, missing
        )
    return tuple(p for p in _REGISTRY.values() if p.name in selection)


def any_active() -> bool:
    return bool(active())


def route_to_v2(call: Mapping):
    """The active path that claims this call, or None for the legacy code."""
    for path in active():
        if path.claims(call):
            return path
    return None


def reload():
    global _active
    _active = None
