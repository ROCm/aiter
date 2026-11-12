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


class TestPackedDownProjectionCsv(unittest.TestCase):
    """The M128 CSV is explicit; padded consumers keep their prior policy."""

    def test_m128_csv_records_effective_packed_configuration(self):
        from aiter.ops.flydsl.gemm_kernels import get_flydsl_hgemm_kernel_params

        path = (
            Path(__file__).resolve().parents[2]
            / "aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv"
        )
        with path.open() as source:
            rows = [
                row
                for row in csv.DictReader(source)
                if (
                    row["gfx"],
                    row["cu_num"],
                    row["M"],
                    row["N"],
                    row["K"],
                    row["bias"],
                    row["dtype"],
                    row["outdtype"],
                    row["scaleAB"],
                    row["bpreshuffle"],
                )
                == (
                    "gfx950",
                    "256",
                    "128",
                    "3584",
                    "7168",
                    "False",
                    BF16,
                    BF16,
                    "False",
                    "False",
                )
            ]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["libtype"], "flydsl")
        config = get_flydsl_hgemm_kernel_params(row["kernelName"])
        self.assertEqual(
            (
                config["block_m"],
                config["block_n"],
                config["block_k"],
                config["stages"],
                config["split_k"],
            ),
            (64, 64, 64, 5, 2),
        )
        self.assertEqual(int(row["splitK"]), config["split_k"])

    def test_padded_lookup_preserves_old_policy_without_mutating_csv(self):
        from aiter.ops.flydsl.gemm_kernels import get_flydsl_hgemm_kernel_params

        new_name = (
            "flydsl_hgemm_abf16_wbf16_bf16_t64x64x64x5_ks2_"
            "w2x2x1_bias0_ktail0_gm0_pft_gfx950"
        )
        row = {"libtype": "flydsl", "solidx": -1, "splitK": 2, "kernelName": new_name}
        original = row.copy()
        key = ("gfx950", 256, 128, 3584, 7168, False, BF16, BF16, False, False)
        table = {key: row}
        with (
            mock.patch.object(tuned, "get_GEMM_A16W16_config_", lambda: table),
            mock.patch.object(tuned, "get_padded_m", lambda *_: 128),
        ):
            for m in range(65, 129):
                with self.subTest(m=m):
                    result = tuned.get_GEMM_A16W16_tuned_config(
                        m,
                        3584,
                        7168,
                        False,
                        BF16,
                        BF16,
                        gfx="gfx950",
                        cu_num=256,
                    )
                    params = get_flydsl_hgemm_kernel_params(result["kernelName"])
                    self.assertEqual(
                        (params["stages"], params["split_k"], result["splitK"]),
                        (5, 2, 2) if m == 128 else (4, 4, 4),
                    )
                    self.assertEqual(row, original)
            # Even an identical new policy saved at exact M127 must win:
            # compatibility is about borrowing the M128 row, not the shape alone.
            exact_key = (*key[:2], 127, *key[3:])
            table[exact_key] = row.copy()
            exact = tuned.get_GEMM_A16W16_tuned_config(
                127,
                3584,
                7168,
                False,
                BF16,
                BF16,
                gfx="gfx950",
                cu_num=256,
            )
            self.assertIs(exact, table[exact_key])
            self.assertEqual(exact["kernelName"], new_name)


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
            mock.patch.object(policy, "get_gfx", return_value="gfx950"),
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
            mock.patch.object(kernels, "get_gfx", return_value="gfx950"),
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
                    mock.patch.object(kernels, "get_gfx", return_value=gfx),
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
