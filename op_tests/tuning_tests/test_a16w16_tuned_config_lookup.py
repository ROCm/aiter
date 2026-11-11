# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
``get_GEMM_A16W16_tuned_config`` returns the tuned row or None, and
``get_GEMM_A16W16_config`` still builds its default on a miss.

Requires torch (importing ``aiter`` pulls it in); it does not need a GPU.

Run:
    python3 -m unittest op_tests.tuning_tests.test_a16w16_tuned_config_lookup -v
"""

import csv
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import aiter.tuned_gemm as tuned

GFX, CU_NUM = "gfx942", 304
M, N, K = 32, 256, 1024
BF16 = str(torch.bfloat16)
KEY = (GFX, CU_NUM, M, N, K, False, BF16, BF16, False, False)


def _fake_gfx950_param(m, n, k, kwargs):
    """Stand in for a validated gfx950 param with the real load geometry."""
    from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950_utils import (
        GFX950_DMA_BYTES,
        GFX950_WAVE_SIZE,
    )

    waves = kwargs["m_waves"] * kwargs["n_waves"] * kwargs["k_waves"]
    return SimpleNamespace(
        block_m=kwargs["block_m"],
        block_n=kwargs["block_n"],
        block_k=kwargs["block_k"],
        block_threads=waves * GFX950_WAVE_SIZE,
        async_load_bytes=GFX950_DMA_BYTES,
        in_data_bytes=2,
    )


class TestA16W16TunedConfigLookup(unittest.TestCase):
    def _lookup(self, table):
        patches = [
            mock.patch.object(tuned, "get_GEMM_A16W16_config_", lambda: table),
            mock.patch.object(tuned, "get_gfx_runtime", lambda: GFX),
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
            mock.patch.object(policy, "get_gfx_runtime", lambda: "gfx1250"),
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
                    lds = (
                        max(
                            config["stages"]
                            * (config["block_m"] + config["block_n"])
                            * (config["block_k"] + 8),
                            config["block_m"]
                            * (config["block_n"] + (8 if config["interleave"] else 0)),
                        )
                        * 2
                    )
                    self.assertLessEqual(lds, 320 * 1024)
                    k_tiles = -(-(k // config["split_k"]) // config["block_k"])
                    self.assertGreaterEqual(k_tiles, config["stages"] - 1)
                    for target_gfx in ("gfx1250",):
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


class TestTdmPolicyContract(unittest.TestCase):
    def test_names_include_schedule_and_reduction_and_reject_invalid(self):
        from aiter.ops.flydsl.gemm_kernels import (
            flydsl_hgemm_kernel_name,
            get_flydsl_hgemm_kernel_params,
        )

        config = {
            "block_m": 128,
            "block_n": 128,
            "block_k": 128,
            "stages": 2,
            "split_k": 3,
            "m_waves": 2,
            "n_waves": 2,
            "k_waves": 1,
            "group_m": 0,
        }
        names = []
        for dtype in (torch.bfloat16, torch.float16):
            for policy in ("tdm", "tdmi", "tdma", "tdmia"):
                kwargs = {
                    "dtype": dtype,
                    "out_dtype": dtype,
                    "config": dict(config, policy=policy),
                    "has_bias": True,
                    "target_gfx": "gfx1250",
                }
                if dtype == torch.float16 and policy in ("tdma", "tdmia"):
                    with self.assertRaises(ValueError):
                        flydsl_hgemm_kernel_name(**kwargs)
                    continue
                name = flydsl_hgemm_kernel_name(**kwargs)
                names.append(name)
                parsed = get_flydsl_hgemm_kernel_params(name)
                self.assertEqual(parsed["policy"], policy)
                self.assertEqual(parsed["split_k_atomic"], policy in ("tdma", "tdmia"))
                self.assertEqual(parsed["interleave"], policy in ("tdmi", "tdmia"))
                for invalid in (
                    name.replace("gfx1250", "gfx950"),
                    name.replace("_w2x2x1_", "_w2x2x2_"),
                    name.replace("x128x2_ks", "x128x1_ks"),
                ):
                    self.assertIsNone(get_flydsl_hgemm_kernel_params(invalid))
                if parsed["split_k_atomic"]:
                    self.assertIsNone(
                        get_flydsl_hgemm_kernel_params(name.replace("_ks3_", "_ks1_"))
                    )
        self.assertEqual(len(set(names)), len(names))

    def test_examples_large_policy_and_both_reductions(self):
        from types import SimpleNamespace

        from aiter.ops.flydsl import gemm_a16w16_policy as policy

        props = SimpleNamespace(
            multi_processor_count=256,
            warp_size=32,
            max_threads_per_multi_processor=2048,
        )
        with (
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(torch.cuda, "get_device_properties", return_value=props),
        ):
            for dtype in (torch.bfloat16, torch.float16):
                large = policy._gfx1250_tdm_configs(4096, 4096, 4096, dtype)
                self.assertEqual(len(large), 2)
                for c in large:
                    self.assertEqual(
                        (
                            c["block_m"],
                            c["block_n"],
                            c["block_k"],
                            c["stages"],
                            c["m_waves"],
                            c["n_waves"],
                            c["split_k"],
                            c["interleave"],
                            c["split_k_atomic"],
                        ),
                        (256, 256, 128, 2, 2, 2, 1, True, False),
                    )
                self.assertEqual({c["group_m"] for c in large}, {0, 4})
                small = policy._gfx1250_tdm_configs(32, 32, 256, dtype)
                self.assertTrue(small)
                paired = {}
                for c in small:
                    key = tuple(
                        sorted(
                            (k, v)
                            for k, v in c.items()
                            if k not in ("split_k_atomic", "policy")
                        )
                    )
                    paired.setdefault(key, set()).add(c["split_k_atomic"])
                for key, flags in paired.items():
                    self.assertEqual(
                        flags,
                        (
                            {False, True}
                            if dtype == torch.bfloat16 and dict(key)["split_k"] > 1
                            else {False}
                        ),
                    )
            configs = policy._gfx1250_tdm_configs(128, 128, 768, torch.bfloat16)
            self.assertTrue(any(c["interleave"] for c in configs))
            self.assertTrue(
                all(
                    not c["interleave"]
                    or (c["m_waves"], c["n_waves"]) not in ((2, 4), (4, 2), (4, 4))
                    for c in configs
                )
            )
            odd = policy._gfx1250_tdm_configs(128, 139, 768, torch.bfloat16)
            self.assertTrue(odd)
            self.assertTrue(all(not c["split_k_atomic"] for c in odd))


class TestGfx950FlydslMemoryCandidates(unittest.TestCase):
    """Keep wide decode panels and the existing skinny backend in the catalog."""

    def test_band_names_and_shape_boundaries(self):
        from aiter.ops.flydsl.gemm_a16w16_band import band_stream_config
        from aiter.ops.flydsl.gemm_kernels import (
            flydsl_hgemm_kernel_name,
            get_flydsl_hgemm_kernel_params,
        )

        for m in (1, 16, 17, 32, 33, 48, 49, 64, 65, 96, 97, 128):
            config = band_stream_config(m, 18432, 1536)
            self.assertIsNotNone(config)
            name = flydsl_hgemm_kernel_name(
                dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                config=config,
                has_bias=False,
                target_gfx="gfx950",
            )
            parsed = get_flydsl_hgemm_kernel_params(name)
            self.assertEqual({key: parsed[key] for key in config}, config)
            for invalid in (
                name.replace("_ks1_", "_ks2_"),
                name.replace("_bias0_", "_bias1_"),
                name.replace("_bnt_", "_"),
                name.replace("gfx950", "gfx1250"),
            ):
                self.assertIsNone(get_flydsl_hgemm_kernel_params(invalid), invalid)
        for m, n, k in (
            (0, 18432, 1536),
            (129, 18432, 1536),
            (16, 18440, 1536),
            (16, 18432, 1664),
        ):
            self.assertIsNone(band_stream_config(m, n, k))

    def test_wide_panel_and_skinny_round_trip(self):
        import aiter.ops.flydsl.gemm_a16w16_policy as policy
        from aiter.ops.flydsl.gemm_kernels import (
            flydsl_hgemm_kernel_name,
            get_flydsl_hgemm_kernel_params,
        )

        wide = {
            "block_m": 128,
            "block_n": 96,
            "block_k": 64,
            "stages": 4,
            "split_k": 1,
            "m_waves": 4,
            "n_waves": 1,
            "k_waves": 1,
            "group_m": 0,
            "use_half_tile_interleaved": False,
        }
        with (
            mock.patch.object(policy, "get_gfx_runtime", return_value="gfx950"),
            mock.patch.object(
                policy.itertools,
                "product",
                side_effect=[
                    [tuple(wide.values()), tuple({**wide, "block_n": 80}.values())],
                    [(24, 64, 4, 2)],
                ],
            ),
            mock.patch.object(
                policy.GemmConfigPruner, "prune", side_effect=lambda configs: configs
            ),
            mock.patch.object(policy.torch.cuda, "current_device", return_value=0),
            mock.patch.object(
                policy.torch.cuda, "get_device_properties", return_value=None
            ),
            mock.patch.object(
                policy, "make_gemm_a16w16_param_and_validate", _fake_gfx950_param
            ),
            mock.patch.object(policy, "skinny_config_error", return_value=None),
            mock.patch.object(policy, "band_stream_config", return_value=None),
        ):
            configs = policy.get_flydsl_a16w16_configs(
                112, 18432, 1536, torch.bfloat16, torch.bfloat16, False
            )
        self.assertEqual(configs[0], wide)
        # The fixed rectangular families are only removed by the real pruner.
        derived = [
            config
            for config in configs
            if config.get("policy") == "sk"
            or (config["block_m"], config["block_n"]) == (128, 96)
        ]
        self.assertEqual(len(derived), 7)
        copied = {
            (config["block_m"], config["block_n"])
            for config in configs
            if "k_stagger" in config
        }
        self.assertEqual(copied, {(128, 96)})
        self.assertIn({**wide, "block_n": 80}, configs)
        self.assertEqual(configs[-1]["policy"], "sk")
        self.assertNotIn("k_stagger", configs[-1])
        self.assertNotIn("b_non_temporal", configs[-1])
        for config in configs:
            name = flydsl_hgemm_kernel_name(
                dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                config=config,
                has_bias=False,
                target_gfx="gfx950",
            )
            parsed = get_flydsl_hgemm_kernel_params(name)
            self.assertEqual({key: parsed[key] for key in config}, config)


class TestGfx950FlydslCandidates(unittest.TestCase):
    def test_split_k_workspace_boundary_and_fallbacks(self):
        from aiter.ops.flydsl.gemm_kernels import (
            flydsl_hgemm_kernel_name,
            get_flydsl_hgemm_kernel_params,
        )
        from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
            GEMM_A16W16_DTYPE_BF16,
            GEMM_A16W16_DTYPE_FP32,
            split_k_workspace_supported,
        )

        bf16 = GEMM_A16W16_DTYPE_BF16
        self.assertTrue(split_k_workspace_supported(320, 6288, 7, bf16, False))
        self.assertTrue(split_k_workspace_supported(386, 6288, 7, bf16, False))
        self.assertFalse(split_k_workspace_supported(387, 6288, 7, bf16, False))
        self.assertFalse(split_k_workspace_supported(320, 6288, 1, bf16, False))
        self.assertFalse(
            split_k_workspace_supported(320, 6288, 7, GEMM_A16W16_DTYPE_FP32, False)
        )
        self.assertFalse(split_k_workspace_supported(320, 6288, 7, bf16, True))

        config = {
            "block_m": 320,
            "block_n": 192,
            "block_k": 64,
            "stages": 2,
            "split_k": 7,
            "m_waves": 4,
            "n_waves": 4,
            "k_waves": 1,
            "group_m": 0,
            "use_half_tile_interleaved": False,
        }
        for workspace in (False, True):
            name = flydsl_hgemm_kernel_name(
                dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                config={**config, "use_split_k_workspace": workspace},
                has_bias=False,
                target_gfx="gfx950",
            )
            self.assertEqual("_ks7ws_" in name, workspace, name)
            params = get_flydsl_hgemm_kernel_params(name)
            self.assertEqual(params["use_split_k_workspace"], workspace, name)
        # Names tuned before the marker existed keep the atomic reduction.
        legacy = flydsl_hgemm_kernel_name(
            dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            config=config,
            has_bias=False,
            target_gfx="gfx950",
        )
        self.assertFalse(
            get_flydsl_hgemm_kernel_params(legacy)["use_split_k_workspace"]
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
                policy, "make_gemm_a16w16_param_and_validate", _fake_gfx950_param
            ),
        ):
            configs = policy.get_flydsl_a16w16_configs(
                16384, 6288, 7168, torch.bfloat16, torch.bfloat16, False
            )

        self.assertEqual(len(configs), 6)
        self.assertEqual({config["group_m"] for config in configs}, {0, 1, 2, 4, 8, 16})
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
                policy, "make_gemm_a16w16_param_and_validate", _fake_gfx950_param
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
            raw_configs = policy.get_flydsl_a16w16_configs(
                128,
                256,
                128,
                torch.bfloat16,
                torch.bfloat16,
                False,
                use_large_gemm_fast_path=False,
                apply_policy_pruning=False,
            )
            pruned_configs = policy.get_flydsl_a16w16_configs(
                128,
                256,
                128,
                torch.bfloat16,
                torch.bfloat16,
                False,
                use_large_gemm_fast_path=False,
            )
            no_workspace = [
                policy.get_flydsl_a16w16_configs(
                    128,
                    256,
                    128,
                    torch.bfloat16,
                    out_dtype,
                    has_bias,
                    use_large_gemm_fast_path=False,
                    apply_policy_pruning=False,
                )
                for out_dtype, has_bias in (
                    (torch.float32, False),
                    (torch.bfloat16, True),
                )
            ]

        def without_workspace(config):
            return tuple(
                sorted(
                    (key, value)
                    for key, value in config.items()
                    if key != "use_split_k_workspace"
                )
            )

        # Stagger and B cache-policy variants have no workspace twin.
        atomic = {
            without_workspace(config)
            for config in raw_configs
            if config["split_k"] > 1
            and not config.get("use_split_k_workspace")
            and "k_stagger" not in config
        }
        workspace = {
            without_workspace(config)
            for config in raw_configs
            if config.get("use_split_k_workspace")
        }
        self.assertTrue(atomic)
        self.assertEqual(atomic, workspace)
        self.assertFalse(
            any(
                config.get("use_split_k_workspace") and config["split_k"] == 1
                for config in raw_configs
            )
        )
        for other in no_workspace:
            self.assertFalse(
                any(config.get("use_split_k_workspace") for config in other)
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
        self.assertGreater(len(raw_configs), len(pruned_configs))

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
                policy, "make_gemm_a16w16_param_and_validate", _fake_gfx950_param
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
            grouped_configs = policy.get_flydsl_a16w16_configs(
                2689,
                1536,
                7168,
                torch.bfloat16,
                torch.bfloat16,
                False,
            )
            workspace_configs = policy.get_flydsl_a16w16_configs(
                252,
                20480,
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
        validated_half_tiles = [
            config
            for config in configs
            if config["use_half_tile_interleaved"]
            and (config["block_m"], config["block_n"], config["block_k"])
            == (256, 256, 64)
            and (config["m_waves"], config["n_waves"], config["k_waves"]) == (2, 4, 1)
        ]
        assert {config["group_m"] for config in validated_half_tiles} == {
            0,
            1,
            2,
            4,
            8,
            16,
        }
        assert any(
            config["use_half_tile_interleaved"]
            and (
                config["block_m"],
                config["block_n"],
                config["block_k"],
                config["m_waves"],
                config["n_waves"],
                config["group_m"],
            )
            == (192, 128, 128, 2, 4, 4)
            for config in grouped_configs
        )
        assert any(
            config["use_half_tile_interleaved"]
            and (
                config["block_m"],
                config["block_n"],
                config["block_k"],
                config["split_k"],
                config["m_waves"],
                config["n_waves"],
                config["group_m"],
            )
            == (256, 256, 64, 2, 2, 4, 8)
            for config in workspace_configs
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
                policy, "make_gemm_a16w16_param_and_validate", _fake_gfx950_param
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
                policy, "make_gemm_a16w16_param_and_validate", _fake_gfx950_param
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
                ((522, 8448, 7168), (320, 192, 2, 2, 4, 4, 0)),
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
                ((543, 6288, 7168), (128, 128, 3, 1, 2, 2, 1)),
                ((635, 6288, 7168), (128, 128, 3, 1, 4, 4, 1)),
                # Wide hand-added tiles must not tighten the grid tile limit.
                ((1, 163840, 256), (16, 64, 3, 1, 1, 2, 0)),
                ((13, 163840, 256), (16, 64, 3, 1, 1, 2, 0)),
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
            configs = policy.get_flydsl_a16w16_configs(
                322, 7168, 35840, torch.bfloat16, torch.bfloat16, False
            )

        def accumulator_tiles(config):
            return (config["block_m"] // config["m_waves"] // 16) * (
                config["block_n"] // config["n_waves"] // 16
            )

        # Allowlisted rectangles survive only while their accumulators fit.
        self.assertTrue(
            any((c["block_m"], c["block_n"]) == (256, 256) for c in configs)
        )
        self.assertFalse(
            any(
                not c["use_half_tile_interleaved"] and accumulator_tiles(c) > 64
                for c in configs
            )
        )


class TestLegacyFlydslHgemmConfigs(unittest.TestCase):
    """Every persisted legacy name must keep its defaults and launch fields."""

    @staticmethod
    def _rows():
        root = Path(__file__).resolve().parents[2] / "aiter" / "configs"
        for path in sorted(root.rglob("*.csv")):
            with path.open() as source:
                for row in csv.DictReader(source):
                    name = row.get("kernelName", "")
                    if name.startswith("flydsl_hgemm_"):
                        yield path, row

    def test_persisted_names_and_legacy_defaults(self):
        from aiter.ops.flydsl.gemm_kernels import (
            flydsl_hgemm_kernel_name,
            get_flydsl_hgemm_kernel_params,
        )

        count = 0
        for path, row in self._rows():
            name = row["kernelName"]
            with self.subTest(file=path.name, kernel=name):
                params = get_flydsl_hgemm_kernel_params(name)
                self.assertIsNotNone(params)
                if "_stg" not in name:
                    self.assertEqual(params["k_stagger"], 0)
                if "_bnt_" not in name:
                    self.assertFalse(params["b_non_temporal"])
                rebuilt = flydsl_hgemm_kernel_name(
                    dtype=params["dtype"],
                    out_dtype=params["out_dtype"],
                    config=params,
                    has_bias=params["has_bias"],
                    target_gfx=params["target_gfx"],
                )
                # "hti" and "f16" are legacy spellings accepted by the parser.
                canonical = (
                    name.replace("_phti_", "_pht_")
                    .replace("_af16_", "_afp16_")
                    .replace("_wf16_", "_wfp16_")
                    .replace("_f16_", "_fp16_")
                )
                self.assertEqual(rebuilt, canonical)
            count += 1
        self.assertGreater(count, 0, "No persisted HGEMM rows were checked")

    def test_all_legacy_names_forward_without_memory_options(self):
        import aiter.ops.flydsl.gemm_kernels as kernels

        # The callable is mocked, not the name parser or tuned dispatcher.
        # CPU tensors keep this a host-only check for all architectures.
        seen = set()
        for path, row in self._rows():
            name = row["kernelName"]
            if name in seen or "_stg" in name or "_bnt_" in name:
                continue
            seen.add(name)
            params = kernels.get_flydsl_hgemm_kernel_params(name)
            dtype = torch.bfloat16 if params["dtype"] == "bf16" else torch.float16
            out_dtype = torch.float32 if params["out_dtype"] == "fp32" else dtype
            a = torch.empty((2, 8), dtype=dtype)
            b = torch.empty((8, 8), dtype=dtype)
            bias = torch.zeros(8, dtype=dtype) if params["has_bias"] else None
            out = torch.empty((2, 8), dtype=out_dtype)
            with (
                self.subTest(file=path.name, kernel=name),
                mock.patch.object(kernels, "flydsl_hgemm", return_value=out) as launch,
            ):
                actual = tuned.flydsl_gemm(
                    a,
                    b,
                    0,
                    bias=bias,
                    otype=out_dtype,
                    config={"kernelName": name},
                    out=out,
                )
                self.assertIs(actual, out)
                args = launch.call_args.kwargs
                self.assertEqual(args["k_stagger"], 0)
                self.assertFalse(args["b_non_temporal"])
                self.assertNotEqual(args["policy"], "bs")
                for field in (
                    "block_m",
                    "block_n",
                    "block_k",
                    "stages",
                    "split_k",
                    "m_waves",
                    "n_waves",
                    "k_waves",
                    "group_m",
                ):
                    self.assertEqual(args[field], params[field])
        self.assertGreater(len(seen), 0)

    def test_band_replacement_preserves_public_input_normalization(self):
        from contextlib import nullcontext
        from types import SimpleNamespace

        import aiter.ops.flydsl.gemm_kernels as kernels
        from aiter.ops.flydsl.gemm_a16w16_band import band_stream_config

        config = band_stream_config(16, 18432, 1536)
        config.pop("use_half_tile_interleaved")
        a = torch.empty((16, 3072), dtype=torch.bfloat16)[:, ::2]
        b = torch.empty((18432, 3072), dtype=a.dtype)[:, ::2]
        stream = SimpleNamespace(device=a.device)
        with (
            mock.patch.object(kernels, "get_gfx_runtime", return_value="gfx950"),
            mock.patch.object(
                torch.cuda, "stream", return_value=nullcontext()
            ) as context,
            mock.patch.object(kernels, "gemm_a16w16_band") as launch,
        ):
            kernels.flydsl_hgemm(a, b, stream=stream, **config)
            normalized_a, normalized_b = launch.call_args.args[:2]
            self.assertTrue(normalized_a.is_contiguous())
            self.assertTrue(normalized_b.is_contiguous())
            self.assertEqual(normalized_a.shape, a.shape)
            self.assertEqual(normalized_b.shape, b.shape)
            self.assertIs(launch.call_args.args[5], stream)
            context.assert_called_once_with(stream)

    def test_gfx1250_legacy_wrapper_routing(self):
        from contextlib import nullcontext
        from types import SimpleNamespace

        import aiter.ops.flydsl.gemm_kernels as kernels

        seen = set()
        a = torch.empty((2, 8), dtype=torch.bfloat16)
        b = torch.empty((8, 8), dtype=a.dtype)
        out = torch.empty((2, 8), dtype=a.dtype)
        stream = SimpleNamespace(device=a.device)
        fields = (
            "block_m",
            "block_n",
            "block_k",
            "stages",
            "split_k",
            "m_waves",
            "n_waves",
            "k_waves",
            "group_m",
            "policy",
        )
        for _, row in self._rows():
            name = row["kernelName"]
            if name in seen:
                continue
            params = kernels.get_flydsl_hgemm_kernel_params(name)
            if not params["target_gfx"].startswith("gfx1250"):
                continue
            seen.add(name)
            for gfx in ("gfx1250", "gfx1250-strict"):
                with (
                    self.subTest(gfx=gfx, kernel=name),
                    mock.patch.object(kernels, "get_gfx_runtime", return_value=gfx),
                    mock.patch.object(torch.cuda, "stream", return_value=nullcontext()),
                    mock.patch.object(
                        kernels, "gemm_a16w16_gfx1250", return_value=out
                    ) as launch,
                    mock.patch.object(kernels, "gemm_a16w16_band") as band,
                ):
                    result = kernels.flydsl_hgemm(
                        a,
                        b,
                        out=out,
                        stream=stream,
                        **{field: params[field] for field in fields},
                    )
                    self.assertIs(result, out)
                    band.assert_not_called()
                    args = launch.call_args.kwargs
                    for old, new in (
                        ("tile_m", "block_m"),
                        ("tile_n", "block_n"),
                        ("tile_k", "block_k"),
                        ("num_buffers", "stages"),
                        ("m_warp", "m_waves"),
                        ("n_warp", "n_waves"),
                        ("split_k", "split_k"),
                    ):
                        self.assertEqual(args[old], params[new])
                    self.assertEqual(
                        args["main_loop_unroll"], params["use_half_tile_interleaved"]
                    )
                    self.assertNotIn("k_stagger", args)
                    self.assertNotIn("b_non_temporal", args)
        self.assertGreater(len(seen), 0)


if __name__ == "__main__":
    unittest.main()
