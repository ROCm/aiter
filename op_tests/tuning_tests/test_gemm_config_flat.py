# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Flatness guard for the shipped GEMM config tables.

``get_gemm_config()`` hands the caller a **shallow** copy of the bucket dict it
got from the ``lru_cache``d loader, because every tuning value in the GEMM
tables is a scalar -- so ``dict(entry)`` isolates the cache from caller
mutation exactly as ``copy.deepcopy(entry)`` would, without the per-launch
cost. ``get_conv_config()`` already relies on the same property.

That makes flatness a real invariant rather than an observation: a bucket entry
that ever gained a nested ``dict``/``list``/``set`` would be shared between the
cache and the caller, and a caller mutating it in place would silently poison
every later lookup of that shape. This test fails when such an entry is added,
so the assumption is checked in CI instead of trusted.

It is CPU-only and does not import torch: it reads the JSON tables directly and
walks the same ``<arch>/<backend>/gemm/<d_type>/`` layout the loader resolves.

Run:
    python3 -m unittest op_tests.tuning_tests.test_gemm_config_flat -v
"""

import glob
import json
import os
import unittest

AITER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIGS_DIR = os.path.join(AITER_ROOT, "aiter", "ops", "triton", "configs")

# Keys a table may carry alongside its bucket entries. They are never returned
# by get_gemm_config() -- the loader indexes a bucket by name -- so they are not
# subject to the flatness rule.
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
            "get_gemm_config() returns a shallow copy, so nested values would "
            "be shared with the lru_cache and a caller mutating one in place "
            "would poison every later lookup of that shape. Either keep the "
            "entry flat, or give the affected loader a deep copy.\n  "
            + "\n  ".join(offenders),
        )
        self.assertGreater(buckets, 0, "no bucket entries were checked")
        self.assertGreater(values, 0, "no tuning values were checked")


if __name__ == "__main__":
    unittest.main()
