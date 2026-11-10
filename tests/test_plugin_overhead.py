# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Hot-path cost of provider dispatch, measured and gated.

RFC #7240's first milestone asks for hot-path cost to be *measured*, not
designed, so this prints the numbers as well as asserting on them. CPU-only; no
GPU and no provider kernel is involved.

Two gates, both machine-independent, both comparisons measured in this same
process so they scale with the machine rather than against a fixed ceiling.

The first: the whole aiter-side overhead, when a provider serves, must cost
less than a single ``Signature.bind`` of the same signature. Dispatch used to be
dominated by exactly that call, so this fails the moment a generic bind, a
per-call ``getenv``, or a rebuilt provider list comes back.

The second: with nothing enabled -- which is every caller on the day this
merges, and every caller who never turns routing on -- dispatch must cost what
the kill switch costs, not what routing costs. It reached 1.9us once by walking
eligibility and building a frozen Decision for a call that could not be routed
by anyone; that is a cost paid by people who get nothing back for it. The gate
is a multiple rather than an equality because both paths are a handful of
nanoseconds apart and the scheduler owns that difference.

Absolute numbers are printed, not asserted. A hosted runner measures roughly
1.7x what a dedicated machine does: an absolute ceiling set from one of them
goes red on the other, and a gate that goes red for reasons unrelated to the
change teaches people to ignore it. This file had a 4us ceiling for exactly one
CI run before proving that point at 4.212us.

The design budget of 1us is reported for the same reason.
"""

from __future__ import annotations

import timeit
from dataclasses import dataclass
from inspect import signature
from typing import Any, ClassVar

import pytest

from aiter import plugin

BUDGET_US = 1.0
REPEAT = 5
NUMBER = 20_000


@dataclass
class _EntryPoint:
    name: str
    target: Any

    def load(self) -> Any:
        return self.target


def rmsnorm(out, input, weight, epsilon, gemma_norm=False):
    """The signature #7123 routes, so the cost is the real one."""

    return out


class _Provider:
    """Candidate built once, as a real provider would: the budget excludes
    provider time, so select() must not dominate what is being measured."""

    name = "bench"
    api_version = plugin.API_VERSION
    op_versions: ClassVar[dict] = {"bench.rmsnorm": {1}}

    def __init__(self) -> None:
        self.candidate = plugin.KernelCandidate(
            provider="bench",
            kernel_id="k",
            artifact_digest="sha",
            qualification="q",
            run=rmsnorm,
        )

    def select(self, op, args, kwargs):
        return self.candidate


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("AITER_PLUGINS", raising=False)
    plugin._reset_for_testing()
    yield
    plugin._reset_for_testing()


def _microseconds(call) -> float:
    """Minimum per-call cost; the minimum is the one the scheduler cannot inflate."""

    timer = timeit.Timer(call)
    return min(timer.repeat(repeat=REPEAT, number=NUMBER)) / NUMBER * 1e6


def test_dispatch_overhead_is_within_budget(monkeypatch, capsys):
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint("bench", _Provider()),)
    )
    plugin.configure(enable=("bench",), source="test")
    routed = plugin.route(op="bench.rmsnorm", version=1)(rmsnorm)
    args = ("out", "input", "weight", 1e-6)

    native = _microseconds(lambda: rmsnorm(*args))
    served = _microseconds(lambda: routed(*args))

    monkeypatch.setenv("AITER_PLUGINS", "0")
    plugin.reload_config()
    killed = _microseconds(lambda: routed(*args))

    monkeypatch.setenv("AITER_PLUGINS", "1")
    plugin.configure(enable=(), source="test")
    plugin.reload_config()
    unrouted = _microseconds(lambda: routed(*args))

    contract = signature(rmsnorm)

    def bind_once():
        bound = contract.bind(*args)
        bound.apply_defaults()
        return bound.args, bound.kwargs

    generic_bind = _microseconds(bind_once) - _microseconds(lambda: None)
    overhead = served - native

    with capsys.disabled():
        print(f"\n  native call            {native:7.3f}us")
        print(f"  provider serves        {served:7.3f}us   overhead {overhead:7.3f}us")
        print(f"  no provider enabled    {unrouted - native:7.3f}us overhead")
        print(f"  AITER_PLUGINS=0        {killed - native:7.3f}us overhead")
        print(
            f"  nothing enabled is {(unrouted - native) / (killed - native):.2f}"
            f" of the kill switch"
        )
        print(f"  one Signature.bind     {generic_bind:7.3f}us")
        print(f"  budget {BUDGET_US}us: {'met' if overhead <= BUDGET_US else 'MISSED'}")
        print(f"  overhead is {overhead / generic_bind:.2f} of one Signature.bind")

    assert unrouted - native < (killed - native) * 3, (
        f"with nothing enabled dispatch costs {unrouted - native:.3f}us against "
        f"the kill switch's {killed - native:.3f}us; a call no provider can "
        f"serve must not pay for eligibility or for a Decision nobody records"
    )

    assert overhead < generic_bind, (
        f"aiter-side overhead ({overhead:.3f}us) must cost less than a single "
        f"Signature.bind ({generic_bind:.3f}us); dispatch has regressed to "
        f"generic canonicalization or per-call environment reads"
    )


def test_the_generated_binder_is_in_use():
    """The ratio gate above is only meaningful while this holds."""

    assert plugin._canonicalizer_source(signature(rmsnorm)) is not None
