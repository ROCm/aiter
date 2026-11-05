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


class TestGfx1250StrictOpusRow(unittest.TestCase):
    """A gfx1250-strict device reuses gfx1250 rows, but OPUS dispatch keeps
    the strict id, so a row is only usable if that dispatch accepts it."""

    CU = 96
    SHAPE = (4, 2112, 7168)
    KID, SPLIT_K = 0, 5

    def setUp(self):
        if tuned._opus_launch is None:
            self.skipTest("OPUS is not importable")
        from csrc.opus_gemm.opus_gemm_common import get_kernel_instance

        instance = get_kernel_instance("gfx1250", "a16w16", self.KID, cu_num=self.CU)
        row = {
            "libtype": "opus",
            "solidx": self.KID,
            "splitK": self.SPLIT_K,
            "kernelName": instance.name,
        }
        key = ("gfx1250", self.CU, *self.SHAPE, False, BF16, BF16, False, False)
        for p in (
            mock.patch.object(tuned, "get_GEMM_A16W16_config_", lambda: {key: row}),
            mock.patch.object(tuned, "get_padded_m", lambda m, _n, _k, _gl: m),
        ):
            p.start()
            self.addCleanup(p.stop)
        tuned.get_GEMM_A16W16_config.cache_clear()
        self.addCleanup(tuned.get_GEMM_A16W16_config.cache_clear)

    def _lookup(self, gfx):
        args = (*self.SHAPE, False, BF16, BF16, False, False, gfx, self.CU)
        return (
            tuned.get_GEMM_A16W16_tuned_config(*args),
            tuned.get_GEMM_A16W16_config(*args),
        )

    def test_base_id_selects_the_opus_row(self):
        tuned_config, _ = self._lookup("gfx1250")
        self.assertEqual(
            (tuned_config["libtype"], tuned_config["solidx"]), ("opus", self.KID)
        )

    def test_strict_id_skips_a_row_its_dispatch_rejects(self):
        from aiter.ops.opus import dispatch
        from csrc.opus_gemm.opus_gemm_common import KernelKey

        with self.assertRaises(ValueError):
            dispatch._resolve_contract(
                KernelKey("a16w16", "gfx1250-strict", self.CU, self.KID),
                torch.bfloat16,
                torch.bfloat16,
                torch.bfloat16,
                "plain",
                False,
                False,
                False,
                False,
                self.SPLIT_K,
            )
        tuned_config, config = self._lookup("gfx1250-strict")
        self.assertIsNone(tuned_config)
        self.assertEqual(config["libtype"], "triton")


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
                    for target_gfx in ("gfx1250", "gfx1250-strict"):
                        name = flydsl_hgemm_kernel_name(
                            dtype=torch.bfloat16,
                            out_dtype=torch.bfloat16,
                            config=config,
                            has_bias=False,
                            target_gfx=target_gfx,
                        )
                        params = get_flydsl_hgemm_kernel_params(name)
                        self.assertEqual(
                            {key: params[key] for key in config}, config, name
                        )
                        self.assertEqual(params["target_gfx"], target_gfx)


if __name__ == "__main__":
    unittest.main()
