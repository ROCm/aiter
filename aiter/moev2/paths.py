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

The selection is logged once (INFO) when AITER_MOEV2 is set or a path is selected. With
AITER_MOE_EXPLAIN=1, a call no active path claims logs each path's `why_not` reason
once.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import importlib
import logging
import os
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from . import envs
from .dispatch import overrides

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
    `why_not(call)`: why `claims` refuses the call, or None; read only under
    AITER_MOE_EXPLAIN, for the calls no active path claims.
    """

    name: str
    claims: Callable[[Mapping], bool]
    run: Callable[[Mapping], object]
    graduated: bool | frozenset = False
    dispatch_record: Callable[[tuple, dict], object] | None = None
    entry: str = "api"
    segments: tuple = ()
    supported_archs: frozenset[str] = frozenset()
    why_not: Callable[[Mapping], str | None] | None = None

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
_loading = threading.local()

# Guards _REGISTRY, _active and _generation. Never held while importing: a module holds
# its import lock while it runs register(), so a thread importing a path module directly
# would deadlock with a thread waiting for that module under this lock.
_lock = threading.RLock()


def _load_builtin():
    """Import the built-in path modules once, in order.

    A thread arriving while another imports them blocks in `import_module` until each
    module has run, so it returns with every built-in path registered. The modules'
    `register()` calls re-enter here on the importing thread and return at once.
    Loading counts as done only once each module's `PATH` is registered: a module
    imported directly runs this from its own `register()`, before its path is in. A
    failed import raises and leaves loading undone, so the next call imports again
    instead of selecting without the path.
    """
    global _builtin_loaded
    if _builtin_loaded or getattr(_loading, "active", False):
        return
    _loading.active = True
    try:
        for module in _BUILTIN_MODULES:
            importlib.import_module(module)
    finally:
        _loading.active = False
    with _lock:
        _builtin_loaded = all(
            path is not None and _REGISTRY.get(path.name) is path
            for path in (
                getattr(sys.modules[m], "PATH", None) for m in _BUILTIN_MODULES
            )
        )


def register(path: Path):
    _load_builtin()
    with _lock:
        if path.name not in PLANNED:
            raise ValueError(
                f"unknown moev2 path {path.name!r}; add it to PLANNED first"
            )
        if path.name in _REGISTRY:
            raise ValueError(f"moev2 path {path.name!r} is already registered")
        _REGISTRY[path.name] = path
        reload()


def unregister(name: str):
    with _lock:
        _REGISTRY.pop(name, None)
        reload()


def registered():
    _load_builtin()
    with _lock:
        return dict(_REGISTRY)


_active: tuple | None = None
# Bumped by reload(): a selection computed across a reload is returned but not cached.
_generation = 0
# The last selection logged, and the (path, reason) pairs AITER_MOE_EXPLAIN has logged.
# The selection's log record carries the attribute SELECTION_RECORD.
SELECTION_RECORD = "moev2_selection"
_announced = None
_explained: set = set()


def active() -> tuple:
    """Registered paths selected by AITER_MOEV2, in registration order."""
    global _active
    selected = _active
    if selected is not None:
        return selected
    with _lock:
        generation = _generation
    selected = _select()
    with _lock:
        if generation == _generation:
            _active = selected
    _announce(selected)
    return selected


def _announce(selected):
    """Log which paths run on v2, once per selection, when AITER_MOEV2 is set or a
    path is selected (the default with nothing graduated stays silent)."""
    global _announced
    if torch.compiler.is_compiling():
        return
    selection = envs.moev2_selection()
    names = tuple(p.name for p in selected)
    with _lock:
        if (selection == "default" and not names) or _announced == (selection, names):
            return
        _announced = (selection, names)
    from .dispatch.resolver import EXPLAIN_PREFIX

    raw = os.environ.get(envs.AITER_MOEV2, "")
    # Per process, not per call: the differential harness leaves it out of the dispatch
    # lines it compares.
    extra = {SELECTION_RECORD: True}
    if not names:
        logger.info(
            "%s%s=%r: every call on the legacy code",
            EXPLAIN_PREFIX,
            envs.AITER_MOEV2,
            raw,
            extra=extra,
        )
        return
    logger.info(
        "%s%s=%r: %s on aiter.moev2, every other call on the legacy code "
        "(%s=off runs every call on the legacy code)",
        EXPLAIN_PREFIX,
        envs.AITER_MOEV2,
        raw,
        ", ".join(names),
        envs.AITER_MOEV2,
        extra=extra,
    )


def _select() -> tuple:
    selection = envs.moev2_selection()
    # The rollback must work even when a path module fails to import.
    if selection == "off":
        return ()
    registry = registered()
    if selection == "on":
        return tuple(registry.values())
    if selection == "default":
        return tuple(p for p in registry.values() if p.graduated_here())
    unknown = sorted(selection - set(PLANNED))
    if unknown:
        raise ValueError(
            f"{envs.AITER_MOEV2} names unknown paths {unknown}; known: {', '.join(PLANNED)}"
        )
    missing = sorted(selection - set(registry))
    if missing:
        logger.warning(
            "%s: paths %s are not implemented yet", envs.AITER_MOEV2, missing
        )
    return tuple(p for p in registry.values() if p.name in selection)


def any_active() -> bool:
    return bool(active())


def route_to_v2(call: Mapping, entry: str = "api"):
    """The active path at `entry` that claims this call, or None for the legacy code."""
    for path in active():
        if path.entry == entry and path.claims(call):
            return path
    if overrides.explain():
        _explain_unclaimed(call, entry)
    return None


def _explain_unclaimed(call, entry):
    """AITER_MOE_EXPLAIN: log why each active path at `entry` refuses the call, once per
    path and reason."""
    if torch.compiler.is_compiling():
        return
    for path in active():
        if path.entry != entry or path.why_not is None:
            continue
        # Explaining must not break the call it explains.
        try:
            reason = path.why_not(call)
        except Exception as error:  # noqa: BLE001
            reason = f"why_not raised {type(error).__name__}: {error}"
        if reason is None:
            continue
        with _lock:
            if (path.name, reason) in _explained:
                continue
            _explained.add((path.name, reason))
        from .dispatch.resolver import EXPLAIN_PREFIX

        logger.info(
            "%spath %s does not claim a call (%s); the legacy code runs it",
            EXPLAIN_PREFIX,
            path.name,
            reason,
        )


def reload():
    global _active, _generation, _announced
    with _lock:
        _active = None
        _generation += 1
        _announced = None
        _explained.clear()
