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
    for name in (
        "AITER_PLUGINS",
        "AITER_PLUGIN_DISABLE",
        "AITER_PLUGIN_FORCE",
        "AITER_PLUGIN_ENABLE",
    ):
        monkeypatch.delenv(name, raising=False)
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
    monkeypatch.setenv("AITER_PLUGIN_ENABLE", name)

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
    assert plugin.last_decision().reason == "provider-not-enabled"
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
