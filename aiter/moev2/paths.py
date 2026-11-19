# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Migration paths: the unit that moves from the legacy MoE code to aiter.moev2.

A path claims a subset of `fused_moe` calls through a predicate over the call arguments.
The legacy entry points ask `route_to_v2(call, entry)` which path, if any, takes a call;
with no path active this costs one cached boolean check. A claimed call may still be
handed back: `run` returns `DECLINED` when the plan it resolves needs a kernel family the
path has not ported, and the legacy code then runs the call as before.

Predicates run on every call, including during graph capture: they may read dtypes,
shapes and Python values, never tensor contents.
"""

import importlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from aiter.jit.utils.chip_info import get_gfx_runtime

from . import envs

logger = logging.getLogger("aiter")

# Path names agreed for the migration. Only these are accepted in
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


class _Declined:
    """`Path.run` result: run this call on the legacy code instead."""

    def __repr__(self):
        return "DECLINED"


DECLINED = _Declined()

# Where the legacy code consults the paths: "api" at the top of `fused_moe` (before the
# FHMoE dispatch, with the `fused_moe` parameters; skipped while torch.compile traces),
# "op" inside the `fused_moe_` custom op (with its parameters: enums as ints, the scatter
# context flattened), so torch.compile records the same op whichever code runs it.
ENTRIES = ("api", "op")


@dataclass(frozen=True)
class Path:
    """One migration path.

    `claims(call)` / `run(call)`: `call` maps the parameter names of the path's `entry`
    function to the call's arguments. `run` returns the output, or `DECLINED`.
    `graduated`: True, or the archs (`get_gfx_runtime()` values) the path runs on by
    default; the default needs a passed GPU gate on that arch.
    `dispatch_record(args, kwargs)`: for a legacy `get_2stage_cfgs(*args, **kwargs)` key,
    the v2 plan in the legacy snapshot record format, or None if the path does not take
    the key (not claimed, or declined). Used by the CPU differential test; the GPU cases
    of each path are in tests/moe_dispatch/gpu_differential.py.
    `segments`: the segments the path resolves plans from; the ranking checks use them.
    `supported_archs`: GPU architectures on which the path can claim calls. An empty set
    means the path does not declare an architecture restriction.
    """

    name: str
    claims: Callable[[Mapping], bool]
    run: Callable[[Mapping], object]
    graduated: bool | frozenset = False
    dispatch_record: Callable[[tuple, dict], object] | None = None
    entry: str = "api"
    segments: tuple = ()
    supported_archs: frozenset[str] = frozenset()

    def __post_init__(self):
        if self.entry not in ENTRIES:
            raise ValueError(f"unknown entry {self.entry!r}; known: {ENTRIES}")

    def graduated_here(self) -> bool:
        if isinstance(self.graduated, bool):
            return self.graduated
        try:
            return get_gfx_runtime() in self.graduated
        except (RuntimeError, KeyError):
            return False


# Modules that register the built-in paths; imported on first use, not at import time.
_BUILTIN_MODULES: tuple = ("aiter.moev2.migration.a8w8_pt",)

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
        return tuple(p for p in _REGISTRY.values() if p.graduated_here())
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


def route_to_v2(call: Mapping, entry: str = "api"):
    """The active path at `entry` that claims this call, or None for the legacy code."""
    for path in active():
        if path.entry == entry and path.claims(call):
            return path
    return None


def reload():
    global _active
    _active = None
