# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A routed op must not split the graph it is compiled into.

Routing replaces a direct call with a Python dispatcher, which torch.compile
traces through -- so whatever the dispatcher does on the way to the native
call becomes part of the trace. Building the config snapshot lazily put a lock
acquisition and an ``os.getenv`` on the first dispatch of the process, and
Dynamo can trace neither: that call graph-breaks, and the region compiles as
two graphs instead of one.

A framework that permits one backend invocation per compilation is then handed
a second and asserts -- ATOM's ``VllmBackend can only be called once``. The op
still computes the right answer, which is why the interface's own tests, all
eager, had nothing to say about it.

Each case runs in its own process: the breakage is first-call-only, and
``torch._dynamo`` state is global, so a second case in the same process would
measure a warm interpreter rather than a cold one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_PLUGIN = Path(__file__).resolve().parents[1] / "aiter" / "plugin" / "__init__.py"

# Loaded by path rather than as `aiter.plugin`: the subprocess has no conftest
# to stand up the buildless stub, and the module imports nothing but stdlib.
_PROBE = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("aiter_plugin_probe", {path!r})
plugin = importlib.util.module_from_spec(spec)
sys.modules["aiter_plugin_probe"] = plugin
spec.loader.exec_module(plugin)

# Read BEFORE anything dispatches: a lazily built snapshot would also be set
# by the time the model below has run, so asking afterwards proves nothing.
print("SNAPSHOT_AT_IMPORT", plugin._config_snapshot is not None)

import torch

graphs = []


def counting_backend(gm, example_inputs):
    graphs.append(gm)
    return gm.forward


@plugin.route(op="compile.probe", version=1)
def probe(x):
    return x * 2


@torch.compile(backend=counting_backend, dynamic=False)
def model(x):
    return probe(x) + 1


model(torch.randn(4))
print("GRAPHS", len(graphs))
"""


def _run_probe() -> dict[str, str]:
    source = _PROBE.format(path=str(_PLUGIN))
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,  # the assert below reports the probe's own output
    )
    assert (
        result.returncode == 0
    ), f"probe failed:\n{result.stdout}\n{result.stderr[-2000:]}"
    out = {}
    for line in result.stdout.splitlines():
        if " " in line:
            key, _, value = line.partition(" ")
            out[key] = value
    return out


def test_a_routed_op_compiles_into_a_single_graph():
    """The whole point: one compiled region, one graph.

    Two means the dispatcher broke it, and a framework allowing one backend
    call per compilation fails rather than runs slowly.
    """

    pytest.importorskip("torch")

    seen = _run_probe()
    assert seen.get("GRAPHS") == "1", (
        f"a routed op compiled into {seen.get('GRAPHS')} graphs; the dispatch "
        f"path graph-broke. Anything it does that Dynamo cannot trace -- a "
        f"lock, an environment read -- has to happen before dispatch, not on it"
    )


def test_the_config_snapshot_exists_before_any_dispatch():
    """Where the fix lives, pinned separately from its effect.

    The graph test above would also pass if the snapshot were built lazily but
    Dynamo had learned to trace a lock, which is not something to rely on. This
    one says the dispatch path has nothing left to build.
    """

    pytest.importorskip("torch")

    seen = _run_probe()
    assert seen.get("SNAPSHOT_AT_IMPORT") == "True", (
        "the config snapshot is still built on first dispatch; that call pays "
        "for a lock and an os.getenv, and pays for it inside whatever graph it "
        "happens to land in"
    )
