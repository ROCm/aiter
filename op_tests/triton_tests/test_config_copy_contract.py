# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The contract every config loader that returns ``dict(entry)`` depends on.

``get_gemm_config()``, ``get_conv_config()`` and the two MHC loaders hand the
caller a **shallow** copy of a dict they took from an ``lru_cache``. That is
only safe while the entry is flat: a nested ``dict``/``list``/``set`` inside one
would be shared between the cache and the caller, and a caller mutating it in
place would silently poison every later lookup of that shape -- wrong tuning
parameters, no exception.

Nothing enforced that. These tests do, in three layers:

* :class:`TestShippedTablesAreFlat` walks the tables each loader resolves
  against, for every architecture, not just the one running the test.
* :class:`TestLoaderCopyContract` drives the loaders themselves, so the
  isolation is exercised rather than inferred from the data.
* :class:`TestLoaderInventory` fails when a *new* shallow-copy loader appears
  without being added here, so the coverage tracks the source instead of
  freezing at the set that existed when this file was written.
"""

import ast
import glob
import json
import os
import re

import pytest

from aiter.ops.triton.utils.conv_config_utils import get_conv_config
from aiter.ops.triton.utils.gemm_config_utils import (
    _get_gemm_config_cached,
    get_gemm_config,
)

AITER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIGS_DIR = os.path.join(AITER_ROOT, "aiter", "ops", "triton", "configs")
UTILS_DIR = os.path.join(AITER_ROOT, "aiter", "ops", "triton", "utils")

SCALARS = (int, float, str, bool, type(None))

# The top-level keys each family's loader actually indexes, mirroring its own
# walk: the bound buckets and "any" everywhere, plus "shapes"/"shapes_<variant>"
# for conv (exact-shape pins, one level deeper) and the MHC post-config tiers.
# Matching the loaders' patterns rather than "every dict" is what keeps conv's
# ``shapes`` container from being mistaken for a bucket, and keeps the test
# looking at the dicts a caller can actually be handed.
BUCKET_RE = r"^(M_LEQ_\d+|M_GEQ_\d+|any)$"
FAMILIES = {
    "gemm": {"buckets": BUCKET_RE, "containers": ()},
    "mhc": {"buckets": rf"{BUCKET_RE[:-2]}|C_\d+|default)$", "containers": ()},
    "conv": {"buckets": BUCKET_RE, "containers": ("shapes",)},
}

# Every loader that returns a shallow copy, and the op directory it resolves
# against. TestLoaderInventory keeps this honest.
SHALLOW_COPY_LOADERS = {
    ("gemm_config_utils.py", "get_gemm_config"): "gemm",
    ("gemm_config_utils.py", "_get_gemm_config_cached"): "gemm",
    ("conv_config_utils.py", "get_conv_config"): "conv",
    ("mhc_config_utils.py", "get_mhc_config"): "mhc",
    ("mhc_config_utils.py", "get_mhc_post_config"): "mhc",
}

# A glob that silently stops matching turns every table test green. These are
# floors well under today's counts, not targets.
MIN_TABLES = {"gemm": 400, "conv": 50, "mhc": 10}


def _tables(op):
    pattern = os.path.join(CONFIGS_DIR, "*", "*", op, "**", "*.json")
    return sorted(glob.glob(pattern, recursive=True))


def _buckets(table, family):
    """Yield (path, entry) for every dict the family's loader can return.

    Also yields ``(key, None)`` for an unrecognised top-level dict, so a new
    bucket shape fails the test instead of silently escaping the walk.
    """
    spec = FAMILIES[family]
    for key, value in table.items():
        is_container = key in spec["containers"] or any(
            key.startswith(c + "_") for c in spec["containers"]
        )
        if is_container:
            for inner_key, inner in (value or {}).items():
                yield f"{key}.{inner_key}", inner
        elif re.match(spec["buckets"], key):
            yield key, value
        elif isinstance(value, dict):
            # Not a bucket the loader indexes, but not a scalar flag either --
            # either a new pattern this test must learn, or a mistake.
            yield key, None


class TestShippedTablesAreFlat:
    @pytest.mark.parametrize("family", sorted(FAMILIES))
    def test_every_bucket_entry_is_flat(self, family):
        tables = _tables(family)
        assert len(tables) >= MIN_TABLES[family], (
            f"only {len(tables)} {family} tables discovered under {CONFIGS_DIR}; "
            "the layout moved and this test is no longer looking at anything"
        )

        offenders, checked = [], 0
        for path in tables:
            rel = os.path.relpath(path, CONFIGS_DIR)
            with open(path) as f:
                table = json.load(f)
            for name, entry in _buckets(table, family):
                if entry is None:
                    offenders.append(
                        f"{rel}: {name} is a dict this walk does not recognise "
                        "as a bucket -- teach the test its pattern or fix the key"
                    )
                    continue
                if not isinstance(entry, dict):
                    offenders.append(f"{rel}: {name} is {type(entry).__name__}")
                    continue
                checked += 1
                for key, value in entry.items():
                    if not isinstance(value, SCALARS):
                        offenders.append(
                            f"{rel}: {name}.{key} is {type(value).__name__}"
                        )

        assert checked > 0, f"no {family} bucket entries were examined"
        assert not offenders, (
            f"{family} loaders return dict(entry), so a non-scalar value is shared "
            "with the lru_cache and a caller mutating it poisons later lookups of "
            "that shape. Keep the entry flat, or give the loader a deep copy:\n  "
            + "\n  ".join(offenders)
        )


class TestLoaderCopyContract:
    """Exercise the isolation, rather than inferring it from the tables."""

    FAMILY, M, N, K = "GEMM-A16W16", 1, 5120, 2880

    def _load(self):
        try:
            return get_gemm_config(self.FAMILY, self.M, self.N, self.K)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no {self.FAMILY} table for the running arch: {exc}")

    def _cached(self):
        # lru_cache keys on the literal argument tuple and does not normalise
        # defaults, so this has to mirror get_gemm_config()'s own call: a
        # shorter one is a different entry and would make the checks vacuous.
        return _get_gemm_config_cached(
            self.FAMILY, self.M, self.N, self.K, None, None, "triton", None
        )

    def test_returned_entry_is_not_the_cached_object(self):
        config, _ = self._load()
        cached, _ = self._cached()
        assert config is not cached
        assert config == cached

    def test_caller_mutation_does_not_reach_the_cache(self):
        first, _ = self._load()
        before = dict(first)
        key = next(iter(first))

        first[key] = "poisoned"
        first["INJECTED_BY_TEST"] = object()
        del first[next(k for k in first if k not in (key, "INJECTED_BY_TEST"))]

        second, _ = self._load()
        assert second == before, (
            "a caller's in-place edits reached the lru_cache entry, so a later "
            "lookup of the same shape returned a corrupted config"
        )

    def test_two_callers_hold_independent_entries(self):
        a, _ = self._load()
        b, _ = self._load()
        assert a is not b
        key = next(iter(a))
        a[key] = "only-in-a"
        assert b[key] != "only-in-a"

    def test_returned_entry_is_flat(self):
        config, _ = self._load()
        nested = {
            k: type(v).__name__ for k, v in config.items() if not isinstance(v, SCALARS)
        }
        assert not nested, f"loader returned non-scalar values: {nested}"

    def test_conv_loader_isolates_its_cache(self):
        names = {
            os.path.basename(os.path.dirname(p)).upper().replace("_", "-")
            for p in _tables("conv")
        }
        for name in sorted(names):
            try:
                first = get_conv_config(name)
            except Exception as exc:  # noqa: BLE001
                print(f"skipping {name}: {exc}")
                continue
            if not isinstance(first, dict) or not first:
                continue
            key = next(iter(first))
            sentinel = object()
            first[key] = sentinel
            second = get_conv_config(name)
            assert second[key] is not sentinel, (
                f"get_conv_config({name!r}) handed back an object a previous "
                "caller had already mutated"
            )
            return
        pytest.skip("no conv config resolves on the running arch")


class TestLoaderInventory:
    """A new shallow-copy loader must arrive with coverage, not silently."""

    def test_every_shallow_copy_loader_is_covered(self):
        found = set()
        for path in sorted(glob.glob(os.path.join(UTILS_DIR, "*_config_utils.py"))):
            with open(path) as f:
                tree = ast.parse(f.read())
            for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
                for node in ast.walk(fn):
                    if not isinstance(node, ast.Return) or node.value is None:
                        continue
                    returned = [node.value] + list(getattr(node.value, "elts", []))
                    for expr in returned:
                        if (
                            isinstance(expr, ast.Call)
                            and isinstance(expr.func, ast.Name)
                            and expr.func.id == "dict"
                            and expr.args
                        ):
                            found.add((os.path.basename(path), fn.name))

        unlisted = found - set(SHALLOW_COPY_LOADERS)
        assert not unlisted, (
            "these loaders return a shallow copy of a cached entry but are not "
            "covered here, so nothing checks that the tables they read stay "
            "flat: " + ", ".join(f"{m}::{f}" for m, f in sorted(unlisted))
        )

        stale = set(SHALLOW_COPY_LOADERS) - found
        assert not stale, (
            "listed as shallow-copy loaders but no longer return dict(entry); "
            "drop them or fix the entry: "
            + ", ".join(f"{m}::{f}" for m, f in sorted(stale))
        )
