# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Provider discovery through real packaging metadata.

Every other test substitutes ``_entry_points``, so the seam between the plugin
interface and the outside world -- ``importlib.metadata``, the entry-point group
name, loading an instance or a factory -- is the least covered part of it. These
tests install real ``.dist-info`` directories on ``sys.path`` and let discovery
find them the way it finds a wheel.
"""

from __future__ import annotations

import sys
import textwrap
import threading

import pytest

from aiter import plugin

_PROVIDER_MODULE = """
from aiter import plugin


class Provider:
    name = "{name}"
    api_version = plugin.API_VERSION
    op_versions = {{"demo.add": {{1}}}}

    def select(self, op, args, kwargs):
        return plugin.KernelCandidate(
            provider="{name}",
            kernel_id="k1",
            artifact_digest="sha256:demo",
            qualification="https://example.invalid/evidence",
            run=lambda left, right: 100,
        )


instance = Provider()


def factory():
    return Provider()
"""


def _install(tmp_path, monkeypatch, name, target):
    """Write a package and a .dist-info declaring it, and put it on sys.path."""

    site = tmp_path / "site"
    site.mkdir(exist_ok=True)
    (site / f"{name}_pkg.py").write_text(_PROVIDER_MODULE.format(name=name))

    dist = site / f"{name}-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n"
    )
    (dist / "entry_points.txt").write_text(textwrap.dedent(f"""\
            [{plugin.ENTRY_POINT_GROUP}]
            {name} = {name}_pkg:{target}
            """))
    monkeypatch.syspath_prepend(str(site))
    return site


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("AITER_PLUGINS", raising=False)
    plugin._reset_for_testing()
    yield
    plugin._reset_for_testing()
    sys.modules.pop("aklr_pkg", None)
    sys.modules.pop("factoryprov_pkg", None)


@pytest.mark.parametrize("target", ["instance", "factory"])
def test_a_real_entry_point_is_discovered_and_serves(tmp_path, monkeypatch, target):
    """An entry point may name a provider instance or a zero-argument factory."""

    name = "aklr" if target == "instance" else "factoryprov"
    _install(tmp_path, monkeypatch, name, target)
    plugin.configure(enable=(name,), source="test")

    @plugin.route(op="demo.add", version=1)
    def add(left: int, right: int = 1) -> int:
        return left + right

    assert add(2, 3) == 100, "the installed provider should serve"
    decision = plugin.last_decision()
    assert (decision.route, decision.provider) == ("provider", name)
    assert decision.artifact_digest == "sha256:demo"


def test_an_installed_provider_is_still_not_enabled_by_default(tmp_path, monkeypatch):
    """Installation is not opt-in, and that must hold for a real install too."""

    _install(tmp_path, monkeypatch, "aklr", "instance")

    @plugin.route(op="demo.add", version=1)
    def add(left: int, right: int = 1) -> int:
        return left + right

    assert add(2, 3) == 5
    assert plugin.last_decision() is None, "nothing enabled, nothing recorded"
    assert plugin.explain("demo.add", 2, 3).decision.reason == "provider-not-enabled"
    # It was found, just not allowed to serve.
    assert plugin.status()["providers"][0]["name"] == "aklr"


def test_discovery_reads_only_this_group(tmp_path, monkeypatch):
    site = _install(tmp_path, monkeypatch, "aklr", "instance")
    other = site / "stranger-1.0.dist-info"
    other.mkdir()
    (other / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: stranger\nVersion: 1.0\n"
    )
    (other / "entry_points.txt").write_text(
        "[some.other.group]\nstranger = aklr_pkg:instance\n"
    )

    assert [record.provider.name for record in plugin._providers()] == ["aklr"]


_REENTRANT_MODULE = """
from aiter import plugin


class Provider:
    name = "reentrant"
    api_versions = {plugin.API_VERSION}
    op_versions = {"demo.add": {1}}

    def select(self, op, args, kwargs):
        return None


# A provider that touches aiter while importing. Discovery must not be holding
# a lock this call would need: that deadlocks with no traceback and no log, in
# the middle of a kernel call.
plugin.status()
instance = Provider()
"""


def test_a_provider_that_touches_aiter_while_importing_does_not_deadlock(
    tmp_path, monkeypatch
):
    site = tmp_path / "site"
    site.mkdir()
    (site / "reentrant_pkg.py").write_text(_REENTRANT_MODULE)
    dist = site / "reentrant-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: reentrant\nVersion: 1.0\n"
    )
    (dist / "entry_points.txt").write_text(
        f"[{plugin.ENTRY_POINT_GROUP}]\nreentrant = reentrant_pkg:instance\n"
    )
    monkeypatch.syspath_prepend(str(site))

    done = threading.Event()

    def discover():
        plugin._providers()
        done.set()

    worker = threading.Thread(target=discover, daemon=True)
    worker.start()
    # Fails rather than hanging the suite: a deadlock would never set this.
    assert done.wait(timeout=20), "discovery deadlocked on a re-entrant provider"

    # It does not serve, and that is the right answer rather than a hang: the
    # re-entrant call asks for an attribute of a module that is still executing
    # its own top level. The point is that the failure is isolated, reported,
    # and survivable -- a deadlock here is invisible and unrecoverable.
    skipped = {
        e["entry_point"]: e["reason"] for e in plugin.status()["skipped_providers"]
    }
    assert "reentrant" in skipped
    assert "circular import" in skipped["reentrant"]


def test_a_broken_entry_point_scan_does_not_raise_into_the_call(monkeypatch):
    """A malformed wheel belonging to somebody else must not break dispatch."""

    def exploding():
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "some other package is broken")

    monkeypatch.setattr(plugin, "_entry_points", exploding)

    @plugin.route(op="demo.add", version=1)
    def add(left: int, right: int = 1) -> int:
        return left + right

    assert add(2, 3) == 5, "it must fall back to native, not raise"
    assert add(4, 5) == 9, "and must not re-raise on the next call either"
    assert (
        plugin.status()["skipped_providers"][0]["entry_point"] == "<entry point scan>"
    )
