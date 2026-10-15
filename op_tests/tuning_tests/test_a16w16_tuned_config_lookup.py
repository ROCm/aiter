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


class TestGfx1250DefaultDispatch(unittest.TestCase):
    """gfx1250 shapes without a tuned row: Triton where the (N, K) has its own config, else FlyDSL where it beats
    Triton's default config and can write C, else Triton's default config where it can write C, else torch.
    """

    def _default(self, m, n, k, dtype=BF16, otype=BF16, bias=False):
        from aiter.ops.triton.utils._triton import arch_info

        patches = [
            mock.patch.object(tuned, "get_GEMM_A16W16_config_", dict),
            mock.patch.object(tuned, "get_gfx", lambda: "gfx1250"),
            mock.patch.object(tuned, "get_cu_num", lambda: 256),
            mock.patch.object(tuned, "get_padded_m", lambda m_, _n, _k, _gl: m_),
            mock.patch.object(arch_info, "get_arch", lambda: "gfx1250"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        tuned.get_GEMM_A16W16_config.cache_clear()
        self.addCleanup(tuned.get_GEMM_A16W16_config.cache_clear)
        return tuned.get_GEMM_A16W16_config(m, n, k, bias, dtype, otype)

    def test_large_shapes_take_flydsl(self):
        for m, tile in (
            (512, "_t128x128x64x4_"),
            (1024, "_t256x128x64x4_"),
            (4097, "_t256x128x64x4_"),
        ):
            config = self._default(m, 4096, 4096)
            self.assertEqual(config["libtype"], "flydsl")
            self.assertIn(tile, config["kernelName"])
            params = tuned._get_flydsl_gemm_kernels().get_flydsl_hgemm_kernel_params(
                config["kernelName"]
            )
            self.assertIsNotNone(params, config["kernelName"])

    def test_other_shapes_stay_on_triton(self):
        fp16 = str(torch.float16)
        cases = [
            (256, 4096, 4096, BF16),  # M below 512
            (1025, 4096, 4096, BF16),  # 1024 < M <= 4096
            (4096, 4096, 4096, BF16),
            (8192, 2048, 4096, BF16),  # N below 4096
            (8192, 4096, 256, BF16),  # K below 512
            (8192, 5120, 5120, BF16),  # Triton has a gfx1250 config for this (N, K)
            (8192, 4096, 4096, fp16),
        ]
        for m, n, k, dtype in cases:
            config = self._default(m, n, k, dtype, dtype)
            self.assertEqual(config["libtype"], "triton", (m, n, k, dtype))

    def test_outputs_past_the_flydsl_offsets_take_torch(self):
        fp32 = str(torch.float32)
        # The last 256-row tile of C must end below byte 2^32: 14081 rows of 151936 pad to 14336.
        cases = [
            (14080, 151936, 4096, BF16, "flydsl"),
            (14081, 151936, 4096, BF16, "torch"),
            (14134, 151936, 4096, BF16, "torch"),  # C itself is below 2^32
            (16384, 151936, 4096, BF16, "torch"),
            (21248, 50016, 6144, fp32, "flydsl"),
            (21249, 50016, 6144, fp32, "torch"),
        ]
        for m, n, k, otype, libtype in cases:
            config = self._default(m, n, k, otype=otype)
            self.assertEqual(config["libtype"], libtype, (m, n, k, otype))

    def test_outputs_past_the_triton_default_config_leave_it(self):
        fp16, fp32 = str(torch.float16), str(torch.float32)
        # Triton's default config writes C up to byte 2^31: 7067 rows of 151936 x 2 bytes, not 7068.
        cases = [
            (7067, 151936, 4096, fp16, fp16, False, "triton"),
            (7068, 151936, 4096, fp16, fp16, False, "torch"),
            (3533, 151936, 4096, BF16, fp32, False, "triton"),  # 1024 < M <= 4096
            (3534, 151936, 4096, BF16, fp32, False, "flydsl"),
            (8192, 151936, 4096, BF16, fp32, True, "torch"),  # bias with FP32 output
        ]
        for m, n, k, dtype, otype, bias, libtype in cases:
            config = self._default(m, n, k, dtype, otype, bias)
            self.assertEqual(config["libtype"], libtype, (m, n, k, dtype, otype, bias))

    def test_bias_with_fp32_output_stays_on_triton(self):
        fp32 = str(torch.float32)
        self.assertEqual(
            self._default(8192, 4096, 4096, otype=fp32, bias=True)["libtype"], "triton"
        )
        self.assertEqual(
            self._default(8192, 4096, 4096, otype=fp32)["libtype"], "flydsl"
        )
        self.assertEqual(
            self._default(8192, 4096, 4096, bias=True)["libtype"], "flydsl"
        )


if __name__ == "__main__":
    unittest.main()
