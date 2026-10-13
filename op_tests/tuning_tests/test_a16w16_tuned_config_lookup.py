# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
``get_GEMM_A16W16_tuned_config`` returns the tuned row or None, and
``get_GEMM_A16W16_config`` still builds its default on a miss.

Requires torch (importing ``aiter`` pulls it in); it does not need a GPU.

Run:
    python3 -m unittest op_tests.tuning_tests.test_a16w16_tuned_config_lookup -v
"""

import unittest
from unittest import mock

import torch

import aiter.tuned_gemm as tuned

GFX, CU_NUM = "gfx942", 304
M, N, K = 32, 256, 1024
BF16 = str(torch.bfloat16)
KEY = (GFX, CU_NUM, M, N, K, False, BF16, BF16, False, False)


class TestA16W16TunedConfigLookup(unittest.TestCase):
    def _lookup(self, table):
        patches = [
            mock.patch.object(tuned, "get_GEMM_A16W16_config_", lambda: table),
            mock.patch.object(tuned, "get_gfx", lambda: GFX),
            mock.patch.object(tuned, "get_cu_num", lambda: CU_NUM),
            mock.patch.object(tuned, "get_padded_m", lambda m, _n, _k, _gl: m),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        tuned.get_GEMM_A16W16_config.cache_clear()
        self.addCleanup(tuned.get_GEMM_A16W16_config.cache_clear)
        return (
            tuned.get_GEMM_A16W16_tuned_config(M, N, K, False, BF16, BF16),
            tuned.get_GEMM_A16W16_config(M, N, K, False, BF16, BF16),
        )

    def test_tuned_row_is_returned(self):
        row = {"libtype": "asm", "solidx": 3, "splitK": 1, "kernelName": "k"}
        tuned_config, config = self._lookup({KEY: row})
        self.assertEqual(tuned_config, row)
        self.assertEqual(config, row)

    def test_missing_row_returns_none_and_keeps_default(self):
        tuned_config, config = self._lookup({})
        self.assertIsNone(tuned_config)
        self.assertEqual((config["libtype"], config["solidx"]), ("torch", 0))

    def test_unknown_flydsl_kernel_is_treated_as_missing(self):
        row = {"libtype": "flydsl", "solidx": 0, "kernelName": "not_in_catalog"}
        catalog = mock.Mock()
        catalog.get_flydsl_hgemm_kernel_params.return_value = None
        with mock.patch.object(tuned, "_get_flydsl_gemm_kernels", lambda: catalog):
            tuned_config, config = self._lookup({KEY: row})
        self.assertIsNone(tuned_config)
        self.assertEqual(config["libtype"], "torch")


class TestGfx1250FlydslCandidates(unittest.TestCase):
    """GemmA16W16Tuner's FlyDSL candidates on gfx1250, with the device mocked."""

    def test_candidates_fit_the_gfx1250_kernel_and_round_trip(self):
        from types import SimpleNamespace

        import aiter.ops.flydsl.gemm_a16w16_policy as policy
        from aiter.ops.flydsl.gemm_kernels import (
            flydsl_hgemm_kernel_name,
            get_flydsl_hgemm_kernel_params,
        )

        props = SimpleNamespace(
            multi_processor_count=256,
            warp_size=32,
            max_threads_per_multi_processor=2048,
        )
        with (
            mock.patch.object(policy, "get_gfx", lambda: "gfx1250"),
            mock.patch.object(torch.cuda, "current_device", lambda: 0),
            mock.patch.object(torch.cuda, "get_device_properties", lambda _: props),
        ):
            for m, n, k in ((1, 640, 2880), (2048, 5120, 2880), (8192, 6144, 7168)):
                configs = policy.get_flydsl_a16w16_configs(
                    m, n, k, torch.bfloat16, torch.bfloat16, False
                )
                self.assertTrue(configs, (m, n, k))
                for config in configs:
                    self.assertEqual(config["k_waves"], 1)
                    lds = policy.gfx1250_lds_bytes(
                        config["block_m"],
                        config["block_n"],
                        config["block_k"],
                        config["stages"],
                        config["split_k"],
                    )
                    self.assertLessEqual(lds, 320 * 1024)
                    k_tiles = -(-(k // config["split_k"]) // config["block_k"])
                    self.assertGreaterEqual(k_tiles, config["stages"] - 1)
                    name = flydsl_hgemm_kernel_name(
                        dtype=torch.bfloat16,
                        out_dtype=torch.bfloat16,
                        config=config,
                        has_bias=False,
                        target_gfx="gfx1250",
                    )
                    params = get_flydsl_hgemm_kernel_params(name)
                    self.assertEqual({key: params[key] for key in config}, config, name)


if __name__ == "__main__":
    unittest.main()
