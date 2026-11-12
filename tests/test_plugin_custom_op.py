# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A routed op can be an opaque node rather than traced Python.

Routing replaces a direct call with a dispatcher, and torch.compile traces
through it -- so this module's internals end up inside the caller's graph,
where anything Dynamo cannot follow splits it. Giving `route` a fake lets it
register the op with torch instead: one node, nothing inside it reachable.

`mutates_args="unknown"` is deliberate: a per-op list has to be right for
every op anyone adds later, and under-declaring is undefined behaviour
rather than an error. "unknown" is never wrong about any op, only
conservative about all of them. It was not observed to change the answer
either way here -- the tests below pin what the mechanism does, not a
failure the declaration prevents.

Subprocesses throughout: `torch._dynamo` state is global, and a library
registration cannot be undone within a process.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_PLUGIN = Path(__file__).resolve().parents[1] / "aiter" / "plugin" / "__init__.py"

_HEAD = """
import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location("aiter_plugin_probe", {path!r})
plugin = importlib.util.module_from_spec(spec)
sys.modules["aiter_plugin_probe"] = plugin
spec.loader.exec_module(plugin)

graphs = []


def backend(gm, example_inputs):
    graphs.append(gm)
    return gm.forward


class _EntryPoint:
    def __init__(self, name, target):
        self.name, self._target = name, target

    def load(self):
        return self._target
"""


def _run(body: str) -> dict[str, str]:
    source = _HEAD.format(path=str(_PLUGIN)) + textwrap.dedent(body)
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"probe failed:\n{result.stdout}\n{result.stderr[-2000:]}"
    out = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition(" ")
        if key.isupper():
            out[key] = value
    return out


def test_a_fake_makes_the_op_opaque_and_the_region_one_graph():
    """The point of the mechanism, stated as what a caller sees."""

    pytest.importorskip("torch")

    seen = _run("""
        @plugin.route(op="probe.one", version=1, fake=lambda x: torch.empty_like(x))
        def probe(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        @torch.compile(backend=backend, dynamic=False)
        def model(x):
            return probe(x) + 1

        x = torch.randn(8, 16)
        out = model(x)
        print("GRAPHS", len(graphs))
        print("CORRECT", bool(torch.allclose(out, x * 2 + 1)))
        """)
    assert (
        seen.get("GRAPHS") == "1"
    ), f"a registered routed op still compiled into {seen.get('GRAPHS')} graphs"
    assert seen.get("CORRECT") == "True"


def test_an_op_that_writes_through_an_argument_is_still_correct():
    """An op is free to write through its arguments and still be correct.

    The compiled region reads the buffer the op wrote, so this pins the
    property a caller depends on: registration does not turn an in-place op
    into one whose writes the compiled code may miss.
    """

    pytest.importorskip("torch")

    seen = _run("""
        @plugin.route(op="probe.mut", version=1, fake=lambda out, x: None)
        def probe(out: torch.Tensor, x: torch.Tensor) -> None:
            out.copy_(x * 2)

        @torch.compile(backend=backend, dynamic=False)
        def model(x):
            out = torch.empty_like(x)
            probe(out, x)
            return out * 3

        x = torch.randn(8, 16)
        got = model(x)
        print("GRAPHS", len(graphs))
        print("CORRECT", bool(torch.allclose(got, x * 2 * 3)))
        """)
    assert seen.get("GRAPHS") == "1"
    assert seen.get("CORRECT") == "True", (
        "the compiled region read a stale buffer: the mutation declaration is "
        "not conservative enough"
    )


def test_a_provider_still_serves_a_registered_op():
    """Registration must not cost what routing is for."""

    pytest.importorskip("torch")

    seen = _run("""
        class Provider:
            name = "probe-provider"
            api_versions = (plugin.API_VERSION,)
            op_versions = {"probe.prov": {1}}

            def select(self, op, args, kwargs):
                return plugin.KernelCandidate(
                    provider="probe-provider",
                    kernel_id="k",
                    artifact_digest="d",
                    run=lambda x: x * 3,
                )

        plugin.configure(enable=("probe-provider",), source="probe")
        plugin._entry_points = lambda: (_EntryPoint("probe-provider", Provider()),)

        @plugin.route(op="probe.prov", version=1, fake=lambda x: torch.empty_like(x))
        def probe(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        @torch.compile(backend=backend, dynamic=False)
        def model(x):
            return probe(x)

        x = torch.randn(8, 16)
        got = model(x)
        print("GRAPHS", len(graphs))
        print("SERVED", bool(torch.allclose(got, x * 3)))
        """)
    assert seen.get("SERVED") == "True", "the provider's kernel did not run"
    assert seen.get("GRAPHS") == "1"


def test_an_op_registration_cannot_describe_stays_exactly_as_it_was():
    """Degrading is an outcome, not a failure.

    No fake, an unannotated parameter, an argument no schema can express --
    each leaves the op traced, which is what it was before this mechanism
    existed. That is what makes it safe to leave on for every op.
    """

    pytest.importorskip("torch")

    seen = _run("""
        @plugin.route(op="probe.nofake", version=1)
        def no_fake(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        @plugin.route(op="probe.noann", version=1, fake=lambda x: torch.empty_like(x))
        def no_annotation(x):
            return x * 2

        x = torch.randn(8, 16)
        print("NOFAKE_TRACED", not hasattr(no_fake, "_schema"))
        print("NOANN_TRACED", not hasattr(no_annotation, "_schema"))
        print("NOFAKE_CORRECT", bool(torch.allclose(no_fake(x), x * 2)))
        print("NOANN_CORRECT", bool(torch.allclose(no_annotation(x), x * 2)))
        """)
    assert (
        seen.get("NOFAKE_TRACED") == "True"
    ), "an op with no fake was registered anyway"
    assert (
        seen.get("NOANN_TRACED") == "True"
    ), "an op whose schema cannot be inferred was registered anyway"
    assert seen.get("NOFAKE_CORRECT") == "True"
    assert seen.get("NOANN_CORRECT") == "True"
