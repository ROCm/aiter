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

    def test_untuned_shape_is_logged_once_unless_asked(self):
        for flag, expected in ((0, 2), (1, 4)):
            with self.subTest(AITER_LOG_TUNED_CONFIG=flag):
                with (
                    mock.patch.object(tuned, "AITER_LOG_TUNED_CONFIG", flag),
                    mock.patch.object(tuned, "_UNTUNED_SHAPES_LOGGED", set()),
                    self.assertLogs(tuned.logger, "INFO") as logs,
                ):
                    self._lookup({})
                    tuned.get_GEMM_A16W16_config(M + 1, N, K, False, BF16, BF16)
                    tuned.get_GEMM_A16W16_config(M, 2 * N, K, False, BF16, BF16)
                    # _lookup clears the lru cache, so this repeat is an evicted M.
                    _, config = self._lookup({})
                self.assertEqual(config["libtype"], "torch")
                untuned = [m for m in logs.output if "not found tuned config" in m]
                self.assertEqual(len(untuned), expected)


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


class TestGfx1250StrictTdmRow(unittest.TestCase):
    """A gfx1250-strict runtime borrows gfx1250 rows. TDM has no strict kernel,
    so the lookup skips a TDM row there instead of letting it raise at launch."""

    CU = 256
    SHAPE = (128, 128, 768)
    NAME = (
        "flydsl_hgemm_abf16_wbf16_bf16_t128x128x128x2_ks3_w2x2x1_"
        "bias0_ktail0_gm0_ptdm_gfx1250"
    )

    def _lookup(self, gfx):
        key = ("gfx1250", self.CU, *self.SHAPE, False, BF16, BF16, False, False)
        row = {"libtype": "flydsl", "solidx": 0, "splitK": 3, "kernelName": self.NAME}
        for p in (
            mock.patch.object(tuned, "get_GEMM_A16W16_config_", lambda: {key: row}),
            mock.patch.object(tuned, "get_padded_m", lambda m, _n, _k, _gl: m),
        ):
            p.start()
            self.addCleanup(p.stop)
        tuned.get_GEMM_A16W16_config.cache_clear()
        self.addCleanup(tuned.get_GEMM_A16W16_config.cache_clear)
        args = (*self.SHAPE, False, BF16, BF16, False, False, gfx, self.CU)
        return (
            tuned.get_GEMM_A16W16_tuned_config(*args),
            tuned.get_GEMM_A16W16_config(*args),
        )

    def test_base_runtime_selects_the_tdm_row(self):
        tuned_config, config = self._lookup("gfx1250")
        self.assertEqual(tuned_config["kernelName"], self.NAME)
        self.assertEqual(config["libtype"], "flydsl")

    def test_strict_runtime_skips_the_tdm_row(self):
        tuned_config, config = self._lookup("gfx1250-strict")
        self.assertIsNone(tuned_config)
        self.assertNotEqual(config["libtype"], "flydsl")


class TestOpusRowOperandLayout(unittest.TestCase):
    """The OPUS launcher only reads K-contiguous A/B, so other layouts take the
    backend the GEMM would use without the OPUS row (Triton on gfx1250)."""

    M, N, K = 16, 4096, 512

    def _backend_for(self, A, B):
        opus_row = {"libtype": "opus", "solidx": 710, "splitK": 0, "kernelName": ""}
        calls = []

        def backend(name):
            def run(inp, weights, *args, **kwargs):
                calls.append(name)
                return torch.zeros(inp.shape[0], weights.shape[0], dtype=inp.dtype)

            return run

        with (
            mock.patch.object(tuned, "get_GEMM_A16W16_config", lambda **_: opus_row),
            mock.patch(
                "aiter.ops.opus._arch._device_arch_and_cu",
                lambda _device: ("gfx1250", 96),
            ),
            mock.patch.dict(
                tuned.solMap, {name: backend(name) for name in ("opus", "triton")}
            ),
        ):
            tuned._gemm_a16w16(A, B)
        return calls

    def _launcher_accepts(self, A, B):
        from aiter.ops.opus.gemm_op_a16w16 import _check_a16w16_launch_layout

        Y = torch.empty(1, A.shape[0], B.shape[0], dtype=A.dtype)
        try:
            _check_a16w16_launch_layout(A.unsqueeze(0), B.unsqueeze(0), Y)
        except NotImplementedError:
            return False
        return True

    def test_backend_matches_the_launcher_layout_contract(self):
        A = torch.randn(self.M, self.K, dtype=torch.bfloat16)
        W = torch.randn(self.N, self.K, dtype=torch.bfloat16)
        wide = torch.randn(self.M, 2 * self.K, dtype=torch.bfloat16)
        cases = {
            "contiguous": (A, W, "opus"),
            "A with padded rows": (wide[:, : self.K], W, "opus"),
            "B transposed": (A, W.t().contiguous().t(), "triton"),
            "A transposed": (A.t().contiguous().t(), W, "triton"),
        }
        for name, (a, b, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(self._launcher_accepts(a, b), expected == "opus")
                self.assertEqual(self._backend_for(a, b), [expected])


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
                            * (
                                config["block_n"]
                                + (8 if config.get("interleave", False) else 0)
                            ),
                        )
                        * 2
                    )
                    if config.get("policy") not in ("tdm", "tdmi"):
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

    def test_gfx1250_catalog_keeps_tdm_and_original_ft_ht(self):
        from types import SimpleNamespace

        from aiter.ops.flydsl import gemm_a16w16_policy as policy
        from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm_kernel_name

        props = SimpleNamespace(
            multi_processor_count=256,
            warp_size=32,
            max_threads_per_multi_processor=2048,
        )
        with (
            mock.patch.object(policy, "get_gfx_runtime", return_value="gfx1250"),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(torch.cuda, "get_device_properties", return_value=props),
        ):
            for m in (6, 12, 18, 24, 30, 36, 42, 48, 72, 96, 128, 144):
                configs = policy.get_flydsl_a16w16_configs(
                    m,
                    32,
                    5120,
                    torch.bfloat16,
                    torch.bfloat16,
                    False,
                )

                def names(cs):
                    return {
                        flydsl_hgemm_kernel_name(
                            dtype="bf16",
                            out_dtype="bf16",
                            config=c,
                            has_bias=False,
                            target_gfx="gfx1250",
                        )
                        for c in cs
                    }

                original = policy._gfx1250_configs(m, 32, 5120)
                tdm = policy._gfx1250_tdm_configs(m, 32, 5120, torch.bfloat16)
                self.assertEqual(names(configs), names(original) | names(tdm))
                self.assertEqual(len(names(configs)), len(configs))
                self.assertFalse(any("policy" in c for c in original))
                self.assertLessEqual({c["policy"] for c in tdm}, {"tdm", "tdmi"})
                if m == 6:
                    # One row of each backend that the DSV4.1 table has used
                    # for this shape.
                    prefix = "flydsl_hgemm_abf16_wbf16_bf16_"
                    suffix = "_w1x1x1_bias0_ktail0_gm0_"
                    self.assertIn(
                        f"{prefix}t16x16x256x5_ks5{suffix}pht_gfx1250", names(configs)
                    )
                    self.assertIn(
                        f"{prefix}t16x16x128x6_ks8{suffix}ptdm_gfx1250", names(configs)
                    )


class TestTdmPolicyContract(unittest.TestCase):
    def test_tdm_names_round_trip_and_removed_names_map(self):
        from aiter.ops.flydsl.gemm_kernels import (
            TDM_POLICIES,
            flydsl_hgemm_kernel_name,
            get_flydsl_hgemm_kernel_params,
        )

        self.assertEqual(TDM_POLICIES, {"tdm": False, "tdmi": True})
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

            def build(policy, dtype=dtype, **extra):
                return flydsl_hgemm_kernel_name(
                    dtype=dtype,
                    out_dtype=dtype,
                    config=dict(config, policy=policy, **extra),
                    has_bias=True,
                    target_gfx="gfx1250",
                )

            for policy, interleave in TDM_POLICIES.items():
                name = build(policy)
                self.assertTrue(name.endswith(f"_gm0_p{policy}_gfx1250"), name)
                names.append(name)
                parsed = get_flydsl_hgemm_kernel_params(name)
                self.assertEqual(parsed["policy"], policy)
                self.assertEqual(parsed["interleave"], interleave)
                for invalid in (
                    name.replace("gfx1250", "gfx950"),
                    name.replace("_w2x2x1_", "_w2x2x2_"),
                    name.replace("x128x2_ks", "x128x1_ks"),
                ):
                    self.assertIsNone(get_flydsl_hgemm_kernel_params(invalid))
                with self.assertRaises(ValueError):
                    build(policy, interleave=not interleave)
                # The removed atomic spellings are never emitted, but a row
                # that still carries one runs the same schedule.
                removed = policy + "a"
                with self.assertRaises(ValueError):
                    build(removed)
                parsed = get_flydsl_hgemm_kernel_params(
                    name.replace(f"_p{policy}_", f"_p{removed}_")
                )
                self.assertEqual(parsed["policy"], policy)
                self.assertEqual(parsed["interleave"], interleave)
        self.assertEqual(len(names), len(set(names)))

    def test_examples_large_policy_and_single_cluster_reduction(self):
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
                        ),
                        (256, 256, 128, 2, 2, 2, 1, True),
                    )
                self.assertEqual({c["group_m"] for c in large}, {0, 4})
                small = policy._gfx1250_tdm_configs(32, 32, 256, dtype)
                self.assertTrue(small)
                self.assertTrue(
                    all(
                        c["policy"] == ("tdmi" if c["interleave"] else "tdm")
                        for c in small
                    )
                )
                self.assertEqual(
                    len(small), len({tuple(sorted(c.items())) for c in small})
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
            self.assertTrue(all(c["policy"] in ("tdm", "tdmi") for c in odd))

    def test_tdm_policy_forwarded_by_tuner_and_production(self):
        import importlib.util

        import aiter.ops.flydsl.gemm_kernels as kernels
        import aiter.tuned_gemm as tuned

        path = (
            Path(__file__).resolve().parents[2] / "csrc/gemm_a16w16/gemm_a16w16_tune.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_hgemm_tuner_schedule_test", path
        )
        tuner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tuner)
        a = torch.empty((128, 768), dtype=torch.bfloat16)
        w = torch.empty((128, 768), dtype=a.dtype)
        out = torch.empty((128, 128), dtype=a.dtype)
        for policy in ("tdm", "tdmi"):
            cfg = {
                "block_m": 128,
                "block_n": 128,
                "block_k": 128,
                "stages": 2,
                "split_k": 3,
                "m_waves": 2,
                "n_waves": 2,
                "k_waves": 1,
                "group_m": 0,
                "policy": policy,
                "interleave": policy == "tdmi",
                "use_half_tile_interleaved": False,
            }
            with mock.patch.object(tuner, "flydsl_hgemm", return_value=out) as launch:
                actual = tuner.run_flydsl_gemm_bf16(
                    a, w, out, otype=a.dtype, config=cfg
                )
                self.assertIs(actual, out)
                self.assertEqual(launch.call_args.kwargs["policy"], policy)
            name = kernels.flydsl_hgemm_kernel_name(
                dtype=a.dtype,
                out_dtype=a.dtype,
                config=cfg,
                has_bias=False,
                target_gfx="gfx1250",
            )
            with mock.patch.object(kernels, "flydsl_hgemm", return_value=out) as launch:
                actual = tuned.flydsl_gemm(
                    a, w, 0, otype=a.dtype, out=out, config={"kernelName": name}
                )
                self.assertIs(actual, out)
                self.assertEqual(launch.call_args.kwargs["policy"], policy)


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
                    [tuple(wide.values())],
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
                policy, "make_gemm_a16w16_param_and_validate", return_value=object()
            ),
            mock.patch.object(policy, "skinny_config_error", return_value=None),
            mock.patch.object(policy, "band_stream_config", return_value=None),
        ):
            configs = policy.get_flydsl_a16w16_configs(
                112, 18432, 1536, torch.bfloat16, torch.bfloat16, False
            )
        self.assertEqual(configs[0], wide)
        self.assertEqual(len(configs), 7)
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
        seen_tdm = False
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
            if params["policy"] in kernels.TDM_POLICIES:
                # gfx1250 routing of TDM rows is checked in the next test. A
                # strict runtime has no TDM kernel: dispatch refuses the row.
                self.assertFalse(
                    kernels.flydsl_hgemm_policy_runs_on(
                        params["policy"], "gfx1250-strict"
                    )
                )
                with (
                    self.subTest(gfx="gfx1250-strict", kernel=name),
                    mock.patch.object(
                        kernels, "get_gfx_runtime", return_value="gfx1250-strict"
                    ),
                    self.assertRaisesRegex(
                        RuntimeError, "no kernel for gfx1250-strict"
                    ),
                ):
                    kernels.flydsl_hgemm(
                        a, b, out=out, stream=stream, **{f: params[f] for f in fields}
                    )
                seen_tdm = True
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
        self.assertTrue(seen_tdm)

    def test_gfx1250_tdm_csv_wrapper_routing(self):
        import importlib
        from contextlib import nullcontext
        from types import SimpleNamespace

        import aiter.ops.flydsl.gemm_kernels as kernels

        tdm = importlib.import_module("aiter.ops.flydsl.gemm_a16w16_tdm")
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
        seen = set()
        for _, row in self._rows():
            name = row["kernelName"]
            params = kernels.get_flydsl_hgemm_kernel_params(name)
            if name in seen or params["policy"] not in kernels.TDM_POLICIES:
                continue
            seen.add(name)
            with (
                self.subTest(kernel=name),
                mock.patch.object(kernels, "get_gfx_runtime", return_value="gfx1250"),
                mock.patch.object(torch.cuda, "stream", return_value=nullcontext()),
                mock.patch.object(tdm, "gemm_a16w16_tdm", return_value=out) as launch,
                mock.patch.object(kernels, "gemm_a16w16_gfx1250") as old,
            ):
                result = kernels.flydsl_hgemm(
                    a,
                    b,
                    out=out,
                    stream=stream,
                    **{f: params[f] for f in fields},
                )
                self.assertIs(result, out)
                old.assert_not_called()
                kw = launch.call_args.kwargs
                self.assertEqual(kw["split_k"], params["split_k"])
                self.assertEqual(kw["interleave"], params["interleave"])
                self.assertIs(kw["stream"], stream)
                self.assertEqual(launch.call_args.args[1].shape, b.t().shape)
        self.assertGreater(len(seen), 0)


class TestHgemmInterleavePreshuffleRouting(unittest.TestCase):
    """Keep the TDM schedule and gfx950 packed-weight APIs independent."""

    def setUp(self):
        from contextlib import nullcontext
        from types import SimpleNamespace

        import aiter.ops.flydsl.gemm_kernels as kernels

        self.kernels = kernels
        self.a = torch.empty((32, 128), dtype=torch.bfloat16)
        self.b = torch.empty((128, 128), dtype=self.a.dtype)
        self.out = torch.empty((32, 128), dtype=self.a.dtype)
        self.stream = SimpleNamespace(device=self.a.device)
        patcher = mock.patch.object(
            torch.cuda, "stream", side_effect=lambda _: nullcontext()
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_tdm_interleave_forwards_with_unshuffled_weights(self):
        import importlib

        tdm = importlib.import_module("aiter.ops.flydsl.gemm_a16w16_tdm")
        with (
            mock.patch.object(self.kernels, "get_gfx_runtime", return_value="gfx1250"),
            mock.patch.object(tdm, "gemm_a16w16_tdm", return_value=self.out) as launch,
        ):
            for policy, interleave in (("tdm", False), ("tdmi", True)):
                with self.subTest(interleave=interleave):
                    result = self.kernels.flydsl_hgemm(
                        self.a,
                        self.b,
                        out=self.out,
                        stream=self.stream,
                        policy=policy,
                        bpreshuffle=False,
                    )
                    self.assertIs(result, self.out)
                    self.assertEqual(launch.call_args.kwargs["interleave"], interleave)
                    self.assertIs(launch.call_args.kwargs["stream"], self.stream)
                    self.assertEqual(
                        launch.call_args.args[1].stride(), self.b.t().stride()
                    )

    def test_gfx950_preshuffle_forwards_to_ft_and_skinny(self):
        with mock.patch.object(self.kernels, "get_gfx_runtime", return_value="gfx950"):
            for policy, target in (("ft", "gemm_a16w16"), ("sk", "gemm_a16w16_skinny")):
                with (
                    self.subTest(policy=policy),
                    mock.patch.object(
                        self.kernels, target, return_value=self.out
                    ) as launch,
                ):
                    result = self.kernels.flydsl_hgemm(
                        self.a,
                        self.b,
                        out=self.out,
                        stream=self.stream,
                        policy=policy,
                        split_k=1,
                        m_waves=1,
                        n_waves=1,
                        k_waves=1,
                        bpreshuffle=True,
                    )
                    self.assertIs(result, self.out)
                    kwargs = launch.call_args.kwargs
                    self.assertTrue(
                        kwargs["user_kwargs"]["bpreshuffle"]
                        if policy == "ft"
                        else kwargs["bpreshuffle"]
                    )

    def test_preshuffle_cannot_enter_tdm_path(self):
        import importlib

        tdm = importlib.import_module("aiter.ops.flydsl.gemm_a16w16_tdm")
        with (
            mock.patch.object(self.kernels, "get_gfx_runtime", return_value="gfx1250"),
            mock.patch.object(tdm, "gemm_a16w16_tdm") as launch,
        ):
            for policy, interleave in (("tdm", False), ("tdmi", True)):
                with (
                    self.subTest(interleave=interleave),
                    self.assertRaisesRegex(ValueError, "BF16 preshuffle requires"),
                ):
                    self.kernels.flydsl_hgemm(
                        self.a,
                        self.b,
                        out=self.out,
                        stream=self.stream,
                        policy=policy,
                        bpreshuffle=True,
                    )
            launch.assert_not_called()

    def test_tdmi_cannot_enter_gfx950_preshuffle_path(self):
        with (
            mock.patch.object(self.kernels, "get_gfx_runtime", return_value="gfx950"),
            mock.patch.object(self.kernels, "gemm_a16w16") as launch,
        ):
            with self.assertRaisesRegex(ValueError, "BF16 preshuffle requires"):
                self.kernels.flydsl_hgemm(
                    self.a,
                    self.b,
                    out=self.out,
                    stream=self.stream,
                    policy="tdmi",
                    bpreshuffle=True,
                )
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
