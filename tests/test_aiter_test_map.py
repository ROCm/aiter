# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Consistency checks for .github/aiter-test-map.yaml.

The map is what .github/scripts/select_aiter_tests.py falls back on for tests
the import graph cannot place, so a stale entry is a silently unselected test.
These checks make that a red build instead.

They need no GPU: `pytest -q tests` already runs in aiter-test.yaml.
"""

import importlib.util
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
MAP_FILE = ROOT / ".github/aiter-test-map.yaml"
SELECTOR = ROOT / ".github/scripts/select_aiter_tests.py"


@pytest.fixture(scope="module")
def selector():
    spec = importlib.util.spec_from_file_location("select_aiter_tests", SELECTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mapping():
    return yaml.safe_load(MAP_FILE.read_text(encoding="utf-8"))


def test_every_bare_import_test_is_mapped(selector, mapping):
    """A test that only imports bare aiter has no import-graph edges to follow.

    aiter/__init__.py re-exports ~60 modules with `import *`, so the selector
    drops that edge; without an entry here such a test would never be selected
    by a source change. The selector raises and runs the full suite when one is
    missing -- correct, but it silently gives up the saving, so fail here too.
    """
    missing = sorted(selector.bare_import_tests() - set(mapping["tests"]))
    assert not missing, (
        "add these to .github/aiter-test-map.yaml, or give them a specific "
        f"aiter import so the graph can place them: {missing}"
    )


def test_mapped_tests_still_exist(selector, mapping):
    known = set(selector.list_tests())
    stale = sorted(set(mapping["tests"]) - known)
    assert not stale, f"mapped tests that no longer run in this suite: {stale}"


def test_owned_paths_still_exist(mapping):
    """A renamed or split source silently empties the entry that named it."""
    missing = []
    for test, entry in sorted(mapping["tests"].items()):
        for owned in (entry or {}).get("owns") or []:
            if not (ROOT / owned).exists():
                missing.append(f"{test} -> {owned}")
    assert not missing, f"ownership entries pointing at nothing: {missing}"


def test_non_cpu_only_tests_own_something(mapping):
    """`owns: []` is only honest for a test that drives no aiter op."""
    empty = [
        test
        for test, entry in sorted(mapping["tests"].items())
        if not ((entry or {}).get("owns") or []) and not (entry or {}).get("cpu_only")
    ]
    assert not empty, (
        "these own nothing and are not marked cpu_only, so no source change "
        f"selects them: {empty}"
    )


def test_hsa_families_exist(mapping):
    """hsa/ directory names are the map's keys; a rename must not go unnoticed."""
    arch_dirs = [p for p in (ROOT / "hsa").iterdir() if p.is_dir()]
    families = {
        child.name for arch in arch_dirs for child in arch.iterdir() if child.is_dir()
    }
    unknown = sorted(set(mapping.get("hsa") or {}) - families)
    assert not unknown, f"hsa families in the map with no directory: {unknown}"


def test_tuned_config_patterns_compile(mapping):
    for pattern in mapping.get("tuned_configs") or {}:
        re.compile(pattern)


def test_tuned_config_targets_exist(selector, mapping):
    known = set(selector.list_tests())
    stale = sorted(
        {
            test
            for targets in (mapping.get("tuned_configs") or {}).values()
            for test in targets
        }
        - known
    )
    assert not stale, f"tuned_configs naming tests that do not run here: {stale}"


def test_selector_matches_split_tests_collection(selector):
    """The selection is fed to split_tests.sh, which rejects paths it did not
    collect. Keep list_tests() in step with that script's own `find`."""
    script = (ROOT / ".github/scripts/split_tests.sh").read_text(encoding="utf-8")
    for extra in selector.EXTRA_TUNING_TESTS:
        # split_tests.sh spells these "$TEST_DIR/tuning_tests/<file>".
        suffix = extra.split("/", 1)[1]
        assert suffix in script, (
            f"{extra} is in the selector but not in split_tests.sh; the "
            "selection would be rejected as an unknown path"
        )
