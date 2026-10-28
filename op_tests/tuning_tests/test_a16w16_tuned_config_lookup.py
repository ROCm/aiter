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


class TestGfx950FlydslCandidates(unittest.TestCase):
    def test_split_k_workspace_boundary_and_fallbacks(self):
        from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
            _use_split_k_workspace,
        )

        self.assertTrue(
            _use_split_k_workspace(320, 6288, 7, torch.bfloat16, False, False)
        )
        self.assertTrue(
            _use_split_k_workspace(386, 6288, 7, torch.bfloat16, False, False)
        )
        self.assertFalse(
            _use_split_k_workspace(387, 6288, 7, torch.bfloat16, False, False)
        )
        self.assertFalse(
            _use_split_k_workspace(320, 6288, 1, torch.bfloat16, False, False)
        )
        self.assertFalse(
            _use_split_k_workspace(320, 6288, 7, torch.float32, False, False)
        )
        self.assertFalse(
            _use_split_k_workspace(320, 6288, 7, torch.bfloat16, True, False)
        )
        self.assertFalse(
            _use_split_k_workspace(320, 6288, 7, torch.bfloat16, False, True)
        )

    def test_ragged_large_n_keeps_the_required_large_tile(self):
        from types import SimpleNamespace

        import aiter.ops.flydsl.gemm_a16w16_policy as policy

        props = SimpleNamespace(
            multi_processor_count=256,
            warp_size=64,
            max_threads_per_multi_processor=2048,
        )
        with (
            mock.patch.object(policy, "get_gfx", lambda: "gfx950"),
            mock.patch.object(torch.cuda, "current_device", lambda: 0),
            mock.patch.object(torch.cuda, "get_device_properties", lambda _: props),
            mock.patch.object(
                policy, "make_gemm_a16w16_param_and_validate", lambda *args: object()
            ),
        ):
            configs = policy.get_flydsl_a16w16_configs(
                16384, 6288, 7168, torch.bfloat16, torch.bfloat16, False
            )

        self.assertEqual(len(configs), 2)
        self.assertEqual({config["group_m"] for config in configs}, {0, 4})
        for config in configs:
            self.assertEqual(
                (
                    config["block_m"],
                    config["block_n"],
                    config["block_k"],
                    config["stages"],
                    config["m_waves"],
                    config["n_waves"],
                    config["k_waves"],
                ),
                (256, 256, 64, 2, 2, 4, 1),
            )
            self.assertTrue(config["use_half_tile_interleaved"])

    def test_large_gemm_full_search_exposes_multiple_tiles(self):
        from types import SimpleNamespace

        import aiter.ops.flydsl.gemm_a16w16_policy as policy

        props = SimpleNamespace(
            multi_processor_count=256,
            warp_size=64,
            max_threads_per_multi_processor=2048,
        )
        with (
            mock.patch.object(policy, "get_gfx", lambda: "gfx950"),
            mock.patch.object(torch.cuda, "current_device", lambda: 0),
            mock.patch.object(torch.cuda, "get_device_properties", lambda _: props),
            mock.patch.object(
                policy, "make_gemm_a16w16_param_and_validate", lambda *args: object()
            ),
        ):
            configs = policy.get_flydsl_a16w16_configs(
                16384,
                6288,
                7168,
                torch.bfloat16,
                torch.bfloat16,
                False,
                use_large_gemm_fast_path=False,
            )

        self.assertGreater(len(configs), 2)
        self.assertGreater(
            len(
                {
                    (config["block_m"], config["block_n"], config["block_k"])
                    for config in configs
                }
            ),
            1,
        )

    def test_small_bm_half_tile_rejects_ragged_n(self):
        from types import SimpleNamespace

        import aiter.ops.flydsl.gemm_a16w16_policy as policy

        props = SimpleNamespace(
            multi_processor_count=256,
            warp_size=64,
            max_threads_per_multi_processor=2048,
        )
        with (
            mock.patch.object(policy, "get_gfx", lambda: "gfx950"),
            mock.patch.object(torch.cuda, "current_device", lambda: 0),
            mock.patch.object(torch.cuda, "get_device_properties", lambda _: props),
            mock.patch.object(
                policy, "make_gemm_a16w16_param_and_validate", lambda *args: object()
            ),
        ):
            configs = policy.get_flydsl_a16w16_configs(
                1798,
                6288,
                7168,
                torch.bfloat16,
                torch.bfloat16,
                False,
            )

        assert not any(
            config["use_half_tile_interleaved"]
            and config["block_m"] < 128
            and 6288 % config["block_n"] != 0
            for config in configs
        )

    def test_large_m_full_search_keeps_wide_ragged_n_tiles(self):
        from types import SimpleNamespace

        import aiter.ops.flydsl.gemm_a16w16_policy as policy

        props = SimpleNamespace(
            multi_processor_count=256,
            warp_size=64,
            max_threads_per_multi_processor=2048,
        )
        with (
            mock.patch.object(policy, "get_gfx", lambda: "gfx950"),
            mock.patch.object(torch.cuda, "current_device", lambda: 0),
            mock.patch.object(torch.cuda, "get_device_properties", lambda _: props),
            mock.patch.object(
                policy, "make_gemm_a16w16_param_and_validate", lambda *args: object()
            ),
        ):
            configs = policy.get_flydsl_a16w16_configs(
                15360,
                2112,
                7168,
                torch.bfloat16,
                torch.bfloat16,
                False,
                use_large_gemm_fast_path=False,
            )

        assert any(config["block_n"] == 256 for config in configs)

    def test_near_cu_rectangular_tile_survives_pruning(self):
        from types import SimpleNamespace

        import aiter.ops.flydsl.gemm_a16w16_policy as policy

        props = SimpleNamespace(
            multi_processor_count=256,
            warp_size=64,
            max_threads_per_multi_processor=2048,
        )
        with (
            mock.patch.object(policy, "get_gfx", lambda: "gfx950"),
            mock.patch.object(torch.cuda, "current_device", lambda: 0),
            mock.patch.object(torch.cuda, "get_device_properties", lambda _: props),
            mock.patch.object(
                policy, "make_gemm_a16w16_param_and_validate", lambda *args: object()
            ),
        ):
            for shape, tile in (
                ((2078, 6288, 7168), (208, 256, 2, 1, 2, 4, 0)),
                ((2079, 6288, 7168), (224, 256, 2, 1, 2, 4, 0)),
                ((576, 20480, 7168), (192, 256, 2, 1, 4, 4, 0)),
                ((4032, 2880, 7168), (192, 256, 2, 1, 4, 4, 0)),
                ((279, 7168, 35840), (320, 192, 2, 5, 4, 2, 0)),
                ((312, 6288, 7168), (320, 192, 2, 7, 4, 4, 0)),
                ((264, 8448, 7168), (320, 192, 2, 4, 4, 4, 0)),
                ((248, 8448, 7168), (256, 256, 2, 7, 4, 2, 0)),
                ((255, 8448, 7168), (256, 256, 2, 7, 2, 4, 0)),
                ((288, 8448, 7168), (320, 192, 2, 4, 4, 4, 0)),
                ((706, 2880, 7168), (256, 256, 2, 7, 4, 2, 0)),
                ((1152, 2112, 7168), (320, 192, 2, 4, 4, 4, 0)),
                ((386, 6288, 7168), (400, 128, 2, 4, 5, 2, 0)),
                ((322, 6288, 7168), (384, 192, 2, 7, 8, 1, 0)),
                ((4033, 2880, 7168), (256, 192, 2, 1, 4, 4, 0)),
                ((5382, 2112, 7168), (256, 192, 2, 1, 4, 4, 0)),
                ((3079, 8448, 7168), (288, 192, 2, 1, 2, 4, 0)),
                ((4609, 3584, 7168), (320, 224, 2, 1, 4, 2, 2)),
                ((5758, 2880, 7168), (320, 224, 2, 1, 4, 2, 16)),
                ((5762, 2880, 7168), (224, 320, 2, 1, 2, 4, 8)),
                ((6136, 2880, 7168), (224, 320, 2, 1, 2, 4, 8)),
                ((1025, 8448, 7168), (224, 192, 3, 1, 2, 6, 0)),
                ((2686, 2880, 7168), (160, 192, 3, 1, 2, 4, 16)),
                ((2689, 3072, 512), (192, 192, 3, 1, 3, 4, 0)),
                ((5381, 3072, 512), (320, 224, 2, 1, 4, 2, 4)),
            ):
                configs = policy.get_flydsl_a16w16_configs(
                    *shape,
                    torch.bfloat16,
                    torch.bfloat16,
                    False,
                )
                assert any(
                    config["block_m"] == tile[0]
                    and config["block_n"] == tile[1]
                    and config["block_k"] == 64
                    and config["stages"] == tile[2]
                    and config["split_k"] == tile[3]
                    and config["m_waves"] == tile[4]
                    and config["n_waves"] == tile[5]
                    and config["group_m"] == tile[6]
                    for config in configs
                )


if __name__ == "__main__":
    unittest.main()
