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
import types
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
            mock.patch.object(policy, "get_flydsl_ring_configs", return_value=[]),
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


class TestFlydslRingPolicyNames(unittest.TestCase):
    """The ring GEMM is a FlyDSL HGEMM policy (``prg``): names, validation and the tuner search space."""

    def _name(self, config, **kw):
        from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm_kernel_name

        return flydsl_hgemm_kernel_name(
            dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            config=config,
            has_bias=kw.get("has_bias", False),
            target_gfx=kw.get("target_gfx", "gfx950"),
        )

    def test_every_tiling_round_trips(self):
        from aiter.ops.flydsl.gemm_a16w16_ring import (
            ring_config,
            ring_name_tiling,
            ring_tilings,
        )
        from aiter.ops.flydsl.gemm_kernels import get_flydsl_hgemm_kernel_params

        seen = set()
        for m in (1, 16, 17, 64, 65, 96, 128):
            for tiling in ring_tilings(m):
                self.assertGreaterEqual(16 * tiling[0] * tiling[1], m)
                config = ring_config(*tiling)
                name = self._name(config)
                self.assertTrue(name.endswith("_prg_gfx950"), name)
                parsed = get_flydsl_hgemm_kernel_params(name)
                self.assertEqual({key: parsed[key] for key in config}, config)
                self.assertEqual(ring_name_tiling(parsed), tiling)
                seen.add(name)
        self.assertEqual(
            len(seen),
            len({t for m in (1, 16, 17, 64, 65, 96, 128) for t in ring_tilings(m)}),
        )

    def test_name_example(self):
        from aiter.ops.flydsl.gemm_a16w16_ring import ring_config

        self.assertEqual(
            self._name(ring_config(4, 1, 4)),
            "flydsl_hgemm_abf16_wbf16_bf16_t64x16x64x4_ks1_w1x4x1_bias0_ktail0_gm0_prg_gfx950",
        )

    def test_unsupported_names_are_rejected(self):
        from aiter.ops.flydsl.gemm_a16w16_ring import ring_config
        from aiter.ops.flydsl.gemm_kernels import get_flydsl_hgemm_kernel_params

        name = self._name(ring_config(2, 2, 3))
        self.assertIsNotNone(get_flydsl_hgemm_kernel_params(name))
        for invalid in (
            name.replace("_ks1_", "_ks2_"),
            name.replace("_bias0_", "_bias1_"),
            name.replace("gfx950", "gfx942"),
            name.replace("_gm0_", "_gm4_"),
            name.replace("_gm0_", "_gm0_stg1_"),
            name.replace("_gm0_", "_gm0_bnt_"),
            name.replace("w2x4x1", "w3x4x1"),
            name.replace("w2x4x1", "w2x2x1"),
            name.replace("w2x4x1", "w2x4x2"),
            name.replace("t64x16x64x3", "t64x32x64x3"),
            name.replace("t64x16x64x3", "t64x16x32x3"),
            name.replace("t64x16x64x3", "t64x16x64x5"),
            name.replace("t64x16x64x3", "t80x16x64x3"),
            name.replace("t64x16x64x3", "t320x16x64x3"),
            name.replace("abf16_wbf16_bf16", "afp16_wfp16_fp16"),
            name.replace("_bf16_t", "_fp32_t"),
        ):
            self.assertIsNone(get_flydsl_hgemm_kernel_params(invalid), invalid)
        for bad in (
            {**ring_config(1, 1, 4), "m_waves": 3},
            {**ring_config(1, 1, 4), "stages": 9},
        ):
            with self.assertRaises(ValueError):
                self._name(bad)
        with self.assertRaises(ValueError):
            self._name(ring_config(1, 1, 4), has_bias=True)

    def test_shape_limits(self):
        from aiter.ops.flydsl.kernels.gemm_a16w16_ring_gfx950 import fits

        self.assertTrue(fits(50016, 6144, 4, 1, 4))
        self.assertFalse(
            fits(50016, 6144, 8, 1, 4)
        )  # 4 slots of an 8-fragment tile exceed the LDS
        self.assertFalse(fits(50016 + 8, 6144, 4, 1, 4))  # N % 16
        self.assertFalse(fits(50016, 6144 + 32, 4, 1, 4))  # K % 64
        self.assertFalse(fits(16 * 255, 6144, 4, 1, 4))  # N / 16 < 256 tiles

    def test_catalog_candidates_need_a_256_cu_gfx950_bf16_no_bias(self):
        import aiter.ops.flydsl.gemm_a16w16_policy as policy

        with mock.patch.object(policy, "ring_supported", return_value=True):
            configs = policy.get_flydsl_ring_configs(
                64, 50016, 6144, torch.bfloat16, torch.bfloat16, False
            )
            self.assertTrue(configs)
            self.assertTrue(
                all(c["policy"] == "rg" and c["block_m"] >= 64 for c in configs)
            )
            for args in (
                (129, 50016, 6144, torch.bfloat16, torch.bfloat16, False),
                (64, 50016, 6144, torch.float16, torch.float16, False),
                (64, 50016, 6144, torch.bfloat16, torch.float32, False),
                (64, 50016, 6144, torch.bfloat16, torch.bfloat16, True),
            ):
                self.assertEqual(policy.get_flydsl_ring_configs(*args), [])
        with mock.patch.object(policy, "ring_supported", return_value=False):
            self.assertEqual(
                policy.get_flydsl_ring_configs(
                    64, 50016, 6144, torch.bfloat16, torch.bfloat16, False
                ),
                [],
            )

    def test_tuned_lookup_accepts_ring_row_for_preshuffled_weight(self):
        from aiter.ops.flydsl import gemm_kernels
        from aiter.ops.flydsl.gemm_a16w16_ring import ring_config

        name = self._name(ring_config(4, 1, 4))
        key = ("gfx950", 256, 64, 50016, 6144, False, BF16, BF16, False, True)
        row = {"libtype": "flydsl", "solidx": 0, "splitK": 1, "kernelName": name}
        with (
            mock.patch.object(tuned, "get_GEMM_A16W16_config_", lambda: {key: row}),
            mock.patch.object(tuned, "get_gfx_runtime", lambda: "gfx950"),
            mock.patch.object(tuned, "get_cu_num", lambda: 256),
            mock.patch.object(tuned, "get_padded_m", lambda m, _n, _k, _gl: m),
            mock.patch.object(tuned, "_get_flydsl_gemm_kernels", lambda: gemm_kernels),
        ):
            got = tuned.get_GEMM_A16W16_tuned_config(
                64, 50016, 6144, False, BF16, BF16, bpreshuffle=True
            )
        self.assertEqual(got, row)

    def test_tuned_gemm_dispatch_passes_the_tiling_and_preshuffle(self):
        from aiter.ops.flydsl import gemm_kernels
        from aiter.ops.flydsl.gemm_a16w16_ring import ring_config

        name = self._name(ring_config(2, 2, 3))
        seen = {}
        with (
            mock.patch.object(tuned, "_get_flydsl_gemm_kernels", lambda: gemm_kernels),
            mock.patch.object(
                gemm_kernels,
                "flydsl_hgemm",
                side_effect=lambda *a, **kw: seen.update(kw) or torch.empty(1),
            ),
        ):
            tuned._flydsl_hgemm_kernel_params.cache_clear()
            tuned.flydsl_gemm(
                torch.empty(1),
                torch.empty(1),
                0,
                otype=torch.bfloat16,
                bpreshuffle=True,
                config={"kernelName": name},
            )
        self.assertEqual(
            (seen["policy"], seen["block_m"], seen["stages"], seen["m_waves"]),
            ("rg", 64, 3, 2),
        )
        self.assertTrue(seen["bpreshuffle"])
        self.assertIsNone(seen["out_dtype"])


class TestFlydslRingTunerCandidates(unittest.TestCase):
    """Tuner side of the ring policies (``csrc/gemm_a16w16/gemm_a16w16_tune.py``) and the committed ring rows."""

    FT = types.MappingProxyType(
        {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "stages": 3,
            "split_k": 1,
            "m_waves": 2,
            "n_waves": 2,
            "k_waves": 1,
            "group_m": 0,
            "use_half_tile_interleaved": False,
        }
    )

    @classmethod
    def _tune(cls):
        from csrc.gemm_a16w16 import gemm_a16w16_tune

        return gemm_a16w16_tune

    def _configs(self):
        from aiter.ops.flydsl.gemm_a16w16_cat_x_ring import ring_splitk_config
        from aiter.ops.flydsl.gemm_a16w16_ring import ring_config

        return [dict(self.FT), ring_config(4, 1, 4), ring_splitk_config(4, 10, 5)]

    def _catalog(self, **kw):
        tune = self._tune()
        tune.get_flydsl_bf16_catalog.cache_clear()
        self.addCleanup(tune.get_flydsl_bf16_catalog.cache_clear)
        with (
            mock.patch.object(
                tune, "get_flydsl_a16w16_configs", return_value=self._configs()
            ),
            mock.patch.object(tune, "get_gfx_runtime", return_value="gfx950"),
        ):
            return tune.get_flydsl_bf16_catalog(
                64, 1536, 30720, torch.bfloat16, False, torch.bfloat16, **kw
            )

    @staticmethod
    def _policies(catalog):
        return sorted(cfg.get("policy", "ft") for _, _, cfg in catalog)

    def test_policies_filter_keeps_only_the_named_policies(self):
        self.assertEqual(self._policies(self._catalog()), ["ft", "rg", "rk"])
        self.assertEqual(self._policies(self._catalog(policies=("rg",))), ["rg"])
        self.assertEqual(
            self._policies(self._catalog(policies=("ft", "rk"))), ["ft", "rk"]
        )
        self.assertEqual(self._policies(self._catalog(policies=("sk",))), [])

    def test_policy_tuple_parsing(self):
        tune = self._tune()
        self.assertIsNone(tune._policy_tuple(None))
        self.assertIsNone(tune._policy_tuple(""))
        self.assertEqual(tune._policy_tuple("rg, rk,"), ("rg", "rk"))

    def test_shuffled_catalog_is_ring_only(self):
        catalog = self._catalog(is_shuffle=True)
        self.assertEqual(self._policies(catalog), ["rg", "rk"])
        for _, name, _ in catalog:
            self.assertRegex(name, r"_p(rg|rk)_gfx950$")
        # The policies filter composes with the shuffle restriction.
        self.assertEqual(
            self._policies(self._catalog(is_shuffle=True, policies=("rk",))), ["rk"]
        )
        self.assertEqual(
            self._policies(self._catalog(is_shuffle=True, policies=("ft",))), []
        )

    def test_b_non_temporal_modes_pass_ring_configs_through_once(self):
        tune = self._tune()
        for mode, ft_count in (("off", 1), ("on", 1), ("both", 2)):
            out = tune._flydsl_b_non_temporal_configs(self._configs(), mode, "gfx950")
            by_policy = [c.get("policy", "ft") for c in out]
            self.assertEqual(by_policy.count("ft"), ft_count, mode)
            self.assertEqual(by_policy.count("rg"), 1, mode)
            self.assertEqual(by_policy.count("rk"), 1, mode)
            for c in out:
                if c.get("policy") in ("rg", "rk"):
                    self.assertFalse(c.get("b_non_temporal", False), mode)

    def test_run_flydsl_gemm_bf16_forwards_bpreshuffle(self):
        from aiter.ops.flydsl.gemm_a16w16_ring import ring_config

        tune = self._tune()

        def run(**kw):
            seen = {}
            with mock.patch.object(
                tune,
                "flydsl_hgemm",
                side_effect=lambda *a, **kwargs: seen.update(kwargs) or torch.empty(1),
            ):
                tune.run_flydsl_gemm_bf16(
                    torch.empty(1),
                    torch.empty(1),
                    torch.empty(1),
                    config=ring_config(4, 1, 4),
                    **kw,
                )
            return seen

        for shuffled in (False, True):
            seen = run(bpreshuffle=shuffled)
            self.assertEqual(seen["policy"], "rg")
            self.assertIs(seen["bpreshuffle"], shuffled)
        seen = run()
        self.assertFalse(seen["bpreshuffle"])  # default stays row-major B

    def test_committed_ring_rows_resolve(self):
        from aiter.ops.flydsl import gemm_kernels
        from aiter.ops.flydsl.gemm_a16w16_cat_x_ring import (
            ring_splitk_config,
            ring_splitk_name_tiling,
        )
        from aiter.ops.flydsl.gemm_a16w16_ring import ring_config, ring_name_tiling

        root = Path(__file__).resolve().parents[2] / "aiter" / "configs"
        rows = []
        for path in sorted(root.rglob("*_tuned_gemm.csv")):
            with open(path, newline="") as f:
                rows += [
                    r
                    for r in csv.DictReader(f)
                    if r.get("libtype") == "flydsl"
                    and r["kernelName"].startswith("flydsl_hgemm_")
                ]
        seen = {"rg": 0, "rk": 0}
        for r in rows:
            match = gemm_kernels._HGEMM_KERNEL_RE.match(r["kernelName"])
            self.assertIsNotNone(match, r["kernelName"])
            policy = match.group("policy")
            if policy not in seen:
                continue
            seen[policy] += 1
            params = gemm_kernels.get_flydsl_hgemm_kernel_params(r["kernelName"])
            self.assertIsNotNone(params, r["kernelName"])
            m = int(r["M"])
            self.assertGreaterEqual(params["block_m"], m, r["kernelName"])
            if policy == "rg":
                tiling = ring_name_tiling(params)
                config = ring_config(*tiling) if tiling else None
            else:
                tiling = ring_splitk_name_tiling(params)
                config = ring_splitk_config(*tiling) if tiling else None
                self.assertEqual(int(r["splitK"]), params["split_k"])
            self.assertIsNotNone(tiling, r["kernelName"])
            # The row's name is exactly what the tuner would emit for its tiling.
            self.assertEqual(
                gemm_kernels.flydsl_hgemm_kernel_name(
                    dtype=torch.bfloat16,
                    out_dtype=torch.bfloat16,
                    config=config,
                    has_bias=False,
                    target_gfx="gfx950",
                ),
                r["kernelName"],
            )
        # lm_head (rg) and DSpark FC (rk) rows are both committed.
        self.assertGreater(seen["rg"], 0)
        self.assertGreater(seen["rk"], 0)

    def test_ring_policy_exports(self):
        self.assertEqual(gemm_kernels_module().RING_POLICY, "rg")
        self.assertEqual(gemm_kernels_module().RING_SPLITK_POLICY, "rk")
        for name in ("RING_POLICY", "RING_SPLITK_POLICY"):
            self.assertIn(name, gemm_kernels_module().__all__)


def gemm_kernels_module():
    from aiter.ops.flydsl import gemm_kernels

    return gemm_kernels


if __name__ == "__main__":
    unittest.main()
