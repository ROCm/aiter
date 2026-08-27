"""Validator-owned pytest plugin that records observed Python kernel routes."""

from __future__ import annotations

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


def pytest_configure(config):
    expected = _VALIDATION_EXPECTED_ROUTE
    routes = {route.strip() for route in expected.split(",") if route.strip()}
    shape_vars = [
        name.strip() for name in _VALIDATION_SHAPE_VARS.split(",") if name.strip()
    ]
    observed_routes = set()
    observed_shapes = []

    def profile(frame, event, arg):
        if event != "call":
            return profile
        route = f"{frame.f_globals.get('__name__', '')}:{frame.f_code.co_name}"
        if route not in routes:
            return profile
        observed_routes.add(route)
        if shape_vars and all(name in frame.f_locals for name in shape_vars):
            observed_shapes.append(
                ",".join(_normalize(frame.f_locals[name]) for name in shape_vars)
            )
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
                "pytest_exitstatus": int(exitstatus),
                "producer": "validate-kernel-pr.validation_probe",
            }
        )
        + "\n"
    )
