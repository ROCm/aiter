# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Let the provider-interface tests run without a built AITER.

RFC #7240 asks for "CPU-only handshake validation for Provider API and routed
op versions" among the checks that block every AITER PR. A gate that needs a
ROCm build and a GPU runner is not that gate: it queues behind hardware and
cannot be required on a documentation PR.

``aiter.plugin`` imports nothing but the standard library, by design, so the
interface *can* be exercised without a build -- but importing it the normal way
runs ``aiter/__init__.py``, which needs torch and a compiled module. When that
import fails, this loads the plugin module on its own and registers it under
its real name.

The fallback is deliberately narrow: it triggers only when the real package is
unavailable, so a built checkout always tests the real thing. If someone gives
``aiter/plugin`` a third-party import, this stops working and the CPU gate goes
red -- which is the property worth protecting.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

_PLUGIN = (
    pathlib.Path(__file__).resolve().parents[1] / "aiter" / "plugin" / "__init__.py"
)


def _load_plugin_standalone() -> None:
    package = types.ModuleType("aiter")
    package.__path__ = [str(_PLUGIN.parent.parent)]
    # Marked so that tests which need the real package can tell. Without this
    # they would see the stub, pass importorskip, and fail instead of skipping.
    package.__aiter_stub__ = True
    sys.modules.setdefault("aiter", package)

    spec = importlib.util.spec_from_file_location("aiter.plugin", _PLUGIN)
    module = importlib.util.module_from_spec(spec)
    sys.modules["aiter.plugin"] = module
    spec.loader.exec_module(module)
    package.plugin = module


try:  # pragma: no cover - which branch runs depends on the environment
    import aiter  # noqa: F401
except Exception:  # noqa: BLE001 - any import failure means "not built here"
    _load_plugin_standalone()


def requires_built_aiter() -> None:
    """Skip the calling module where AITER is not built.

    ``importorskip("aiter")`` is not enough on its own: this file may have put a
    stub under that name so the interface can be tested without a build.
    """

    import pytest

    package = sys.modules.get("aiter")
    if package is None or getattr(package, "__aiter_stub__", False):
        pytest.skip("needs a built AITER", allow_module_level=True)
