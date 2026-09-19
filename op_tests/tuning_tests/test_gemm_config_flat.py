# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Guards the flatness ``get_gemm_config()`` relies on for its shallow copy.

A bucket entry that gained a nested ``dict``/``list``/``set`` would be shared
between the ``lru_cache`` and the caller, and a caller mutating it in place
would poison every later lookup of that shape. ``TestGemmConfigFlat`` walks the
shipped tables; ``TestGemmConfigCopyContract`` drives ``get_gemm_config()``
itself, since reading the tables alone would still pass if the loader returned
the cached dict.

Like ``test_config_shape_collision``, this module is **not wired into any CI
workflow**: ``tuning-tests.yaml`` is schedule/manual and names its modules
explicitly, and the Triton job covers ``op_tests/triton_tests``.

``TestGemmConfigFlat`` reads JSON only and does not import torch;
``TestGemmConfigCopyContract`` needs ``aiter`` and skips without it. Neither
needs a GPU.

Run:
    python3 -m unittest op_tests.tuning_tests.test_gemm_config_flat -v
"""

import glob
import json
import os
import unittest

try:  # importing aiter requires torch; skip cleanly where it is unavailable.
    from aiter.ops.triton.utils.gemm_config_utils import (
        _get_gemm_config_cached,
        get_gemm_config,
    )

    _IMPORT_ERR = None
except Exception as e:  # noqa: BLE001
    get_gemm_config = _get_gemm_config_cached = None
    _IMPORT_ERR = e

AITER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIGS_DIR = os.path.join(AITER_ROOT, "aiter", "ops", "triton", "configs")

# Table-level keys, never returned by get_gemm_config().
NON_BUCKET_KEYS = ("_note", "_comment")

SCALAR_TYPES = (int, float, str, bool, type(None))


def _gemm_config_files():
    """Every JSON table under a ``gemm``/``batched_gemm`` op directory."""
    pattern = os.path.join(CONFIGS_DIR, "*", "*", "*gemm*", "**", "*.json")
    return sorted(glob.glob(pattern, recursive=True))


class TestGemmConfigFlat(unittest.TestCase):
    def test_config_tree_is_present(self):
        self.assertTrue(
            os.path.isdir(CONFIGS_DIR), f"config tree not found at {CONFIGS_DIR}"
        )
        self.assertGreater(
            len(_gemm_config_files()), 0, "no GEMM config tables discovered"
        )

    def test_every_bucket_entry_is_flat(self):
        offenders = []
        buckets = 0
        values = 0

        for path in _gemm_config_files():
            rel = os.path.relpath(path, CONFIGS_DIR)
            with open(path) as f:
                table = json.load(f)

            for bucket, entry in table.items():
                if bucket in NON_BUCKET_KEYS:
                    continue
                if not isinstance(entry, dict):
                    offenders.append(
                        f"{rel}: bucket {bucket!r} is {type(entry).__name__}, "
                        "expected an object of tuning values"
                    )
                    continue

                buckets += 1
                for key, value in entry.items():
                    values += 1
                    if not isinstance(value, SCALAR_TYPES):
                        offenders.append(
                            f"{rel}: {bucket!r}.{key!r} is "
                            f"{type(value).__name__}, expected a scalar"
                        )

        self.assertEqual(
            offenders,
            [],
            "get_gemm_config() returns a shallow copy; keep the entry flat, or "
            "give the loader a deep copy.\n  " + "\n  ".join(offenders),
        )
        self.assertGreater(buckets, 0, "no bucket entries were checked")
        self.assertGreater(values, 0, "no tuning values were checked")


@unittest.skipIf(get_gemm_config is None, f"aiter unavailable: {_IMPORT_ERR}")
class TestGemmConfigCopyContract(unittest.TestCase):
    FAMILY, M, N, K = "GEMM-A16W16", 1, 5120, 2880

    def _load(self):
        try:
            return get_gemm_config(self.FAMILY, self.M, self.N, self.K)
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"no {self.FAMILY} table for the running arch: {e}")

    def _cached(self):
        # Must mirror get_gemm_config()'s own call: lru_cache keys on the
        # literal argument tuple, so a shorter call is a different entry.
        return _get_gemm_config_cached(
            self.FAMILY, self.M, self.N, self.K, None, None, "triton", None
        )

    def test_returned_config_is_not_the_cached_object(self):
        config, _ = self._load()
        cached, _ = self._cached()
        self.assertIsNot(config, cached)
        self.assertEqual(config, cached)

    def test_caller_mutation_does_not_poison_the_cache(self):
        first, _ = self._load()
        before = dict(first)
        key = next(iter(first))

        first[key] = "poisoned"
        first["INJECTED_BY_TEST"] = object()
        del first[next(k for k in first if k != key and k != "INJECTED_BY_TEST")]

        second, _ = self._load()
        self.assertEqual(
            second,
            before,
            "a caller's in-place edits reached the lru_cache entry, so a later "
            "lookup of the same shape returned a corrupted config",
        )

    def test_two_callers_hold_independent_configs(self):
        a, _ = self._load()
        b, _ = self._load()
        self.assertIsNot(a, b)
        key = next(iter(a))
        a[key] = "only-in-a"
        self.assertNotEqual(b[key], "only-in-a")

    def test_is_tuned_flag_survives_the_copy(self):
        _, is_tuned = self._load()
        _, cached_is_tuned = self._cached()
        self.assertIs(is_tuned, cached_is_tuned)


if __name__ == "__main__":
    unittest.main()
