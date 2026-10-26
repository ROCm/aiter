"""Validator-owned pytest plugin that records observed Python kernel routes."""

from __future__ import annotations

import importlib
import json
import sys
import threading
from pathlib import Path

_VALIDATION_EXPECTED_ROUTE = ""
_VALIDATION_SHAPE_VARS = ""
_VALIDATION_RECEIPT_PATH = ""


def _normalize(value) -> str:
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    text = str(value)
    for prefix in ("torch.", "flydsl."):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _resolve_route_codes(routes):
    """Map each declared route to the code object it will actually execute.

    A route is written the way a caller reads the source: `module:function`. The profiler
    sees frames, and a frame's identity is `f_globals["__name__"] + ":" + f_code.co_name` --
    which is NOT the same thing whenever the function is wrapped. `functools.wraps` copies
    `__name__` onto the wrapper object but leaves `co_name` as the wrapper's own, so aiter's
    entire `@compile_ops` family executes as `aiter.jit.core:wrapper`: naming the op a
    reviewer actually cares about matched nothing, and the receipt reported an empty route.
    Observed on ROCm/aiter#5081.

    Resolving the declared route to a code object closes that gap without loosening
    anything: the match is still an exact identity, just the right one. Import failures are
    ignored -- the string match below remains, so a route into a module that cannot be
    imported here behaves exactly as it did before.
    """
    codes = {}
    for route in routes:
        module_name, _, attribute = route.partition(":")
        if not module_name or not attribute:
            continue
        try:
            module = importlib.import_module(module_name)
        except BaseException:  # noqa: BLE001 - a target's import must not break the probe
            continue
        target = module
        for part in attribute.split("."):
            target = getattr(target, part, None)
            if target is None:
                break
        if target is None:
            continue
        # Walk the wrapper chain and register every layer: the caller may be intercepted at
        # any of them depending on how the decorator is built.
        seen = set()
        while target is not None and id(target) not in seen:
            seen.add(id(target))
            code = getattr(target, "__code__", None)
            if code is not None:
                codes[code] = route
            target = getattr(target, "__wrapped__", None)
    return codes


def pytest_configure(config):
    expected = _VALIDATION_EXPECTED_ROUTE
    routes = {route.strip() for route in expected.split(",") if route.strip()}
    route_codes = _resolve_route_codes(routes)
    shape_vars = [
        name.strip() for name in _VALIDATION_SHAPE_VARS.split(",") if name.strip()
    ]
    observed_routes = set()
    observed_shapes = []

    # Keyed by the frame OBJECT, never by id(): a freed frame's id is reused, and a
    # second call to the same route would then look like one already recorded. Holding the
    # frame keeps it alive exactly as long as the entry, and the entry is dropped on return.
    pending_frames = {}

    def _try_capture(frame):
        if not shape_vars:
            return True
        if not all(name in frame.f_locals for name in shape_vars):
            return False
        observed_shapes.append(
            ",".join(_normalize(frame.f_locals[name]) for name in shape_vars)
        )
        return True

    def profile(frame, event, arg):
        if event not in ("call", "return"):
            return profile
        route = route_codes.get(frame.f_code)
        if route is None:
            route = f"{frame.f_globals.get('__name__', '')}:{frame.f_code.co_name}"
            if route not in routes:
                return profile
        if event == "call":
            observed_routes.add(route)
            # Sampling only here sees parameters and nothing else, so a route that DERIVES
            # its shapes in its body -- the common case for a kernel whose arguments are
            # tensors -- produced an empty executed_shapes while the receipt still said pass.
            if not _try_capture(frame):
                pending_frames[frame] = True
        else:
            # Body locals are bound by the time the frame returns.
            if pending_frames.pop(frame, False):
                _try_capture(frame)
        return profile

    config._validation_probe = {
        "routes": routes,
        "shape_vars": shape_vars,
        "observed_routes": observed_routes,
        "observed_shapes": observed_shapes,
        "profile": profile,
    }
    sys.setprofile(profile)
    threading.setprofile(profile)


def pytest_sessionfinish(session, exitstatus):
    probe = session.config._validation_probe
    sys.setprofile(None)
    threading.setprofile(None)
    receipt_path = _VALIDATION_RECEIPT_PATH
    if not receipt_path:
        return
    Path(receipt_path).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "route": ",".join(sorted(probe["observed_routes"])),
                "kernel_symbols": sorted(probe["observed_routes"]),
                "executed_shapes": probe["observed_shapes"],
                # Carried into the receipt so the evidence checker can tell "no shapes were
                # asked for" from "shapes were asked for and none were captured".
                "shape_vars": probe["shape_vars"],
                "pytest_exitstatus": int(exitstatus),
                "producer": "validate-kernel-pr.validation_probe",
            }
        )
        + "\n"
    )
