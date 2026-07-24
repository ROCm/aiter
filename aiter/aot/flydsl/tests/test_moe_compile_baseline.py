# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Contract tests for the gfx950 MoE FlyDSL compile-request baseline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from moe_compile_recorder import canonical_json, record_compile_requests  # noqa: E402

_GOLDEN = _TEST_DIR / "data" / "moe_compile_requests_gfx950.json"
_PRIVATE_CACHE_FIELDS = ("cache_key", "manager_key")
_RECORDER_ENV = (
    "ARCH",
    "COMPILE_ONLY",
    "CUDA_VISIBLE_DEVICES",
    "FLYDSL_GPU_ARCH",
    "GPU_ARCHS",
    "HIP_VISIBLE_DEVICES",
)


class TestMoeCompileBaseline(unittest.TestCase):
    def test_gfx950_requests_match_golden_and_are_deterministic(self) -> None:
        before_env = {name: os.environ.get(name) for name in _RECORDER_ENV}
        before_aiter_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "aiter" or name.startswith("aiter.")
        }

        first = record_compile_requests()
        second = record_compile_requests()
        first_json = canonical_json(first)

        self.assertEqual(first_json, canonical_json(second))
        self.assertEqual(
            before_env,
            {name: os.environ.get(name) for name in _RECORDER_ENV},
        )
        after_aiter_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "aiter" or name.startswith("aiter.")
        }
        self.assertEqual(set(before_aiter_modules), set(after_aiter_modules))
        for name, module in before_aiter_modules.items():
            self.assertIs(module, after_aiter_modules[name])

        golden_text = _GOLDEN.read_text()
        golden = json.loads(golden_text)
        self.assertEqual(golden_text, canonical_json(golden))
        self.assertEqual(first, golden)
        self.assertEqual(first_json, golden_text)

        requests = first["requests"]
        self.assertEqual(first["schema_version"], 1)
        self.assertEqual(first["target"], {"arch": "gfx950", "cu_count": 256})
        ids = [request["id"] for request in requests]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, sorted(ids))
        self.assertGreater(len(ids), 0)
        for request in requests:
            self.assertTrue(request["builder"])
            self.assertTrue(request["kwargs"])
            self.assertEqual(
                set(request["trigger"]),
                {"host_path", "launchers", "scenario"},
            )
            self.assertTrue(request["trigger"]["host_path"])
            self.assertTrue(request["trigger"]["scenario"])

        lowered = first_json.lower()
        for field in _PRIVATE_CACHE_FIELDS:
            self.assertNotIn(field, lowered)


if __name__ == "__main__":
    unittest.main()
