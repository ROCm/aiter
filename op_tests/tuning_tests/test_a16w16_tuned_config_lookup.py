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


class TestLatentDownProjectionUnchanged(unittest.TestCase):
    """Original tile/split configuration; only exact M128 changes its schedule."""

    def test_down_m128_keeps_original_csv_and_effective_policy(self):
        from aiter.ops.flydsl.gemm_kernels import get_flydsl_hgemm_kernel_params
        from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
            make_gemm_a16w16_param_and_validate,
        )

        path = (
            Path(__file__).resolve().parents[2]
            / "aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv"
        )
        with path.open() as source:
            rows = [
                r
                for r in csv.DictReader(source)
                if (
                    r["gfx"],
                    r["cu_num"],
                    r["M"],
                    r["N"],
                    r["K"],
                    r["bias"],
                    r["dtype"],
                    r["outdtype"],
                    r["scaleAB"],
                    r["bpreshuffle"],
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
        self.assertEqual(
            (row["libtype"], row["solidx"], row["splitK"]), ("flydsl", "3787", "4")
        )
        expected_name = (
            "flydsl_hgemm_abf16_wbf16_bf16_t64x64x64x4_ks4_"
            "w2x2x1_bias0_ktail0_gm0_pft_gfx950"
        )
        self.assertEqual(row["kernelName"], expected_name)
        parsed = get_flydsl_hgemm_kernel_params(row["kernelName"])
        cfg = {
            key: parsed[key]
            for key in (
                "block_m",
                "block_n",
                "block_k",
                "stages",
                "split_k",
                "m_waves",
                "n_waves",
                "k_waves",
                "group_m",
                "use_half_tile_interleaved",
            )
        }
        cfg.update(
            in_dtype_id=2,
            out_dtype_id=2,
            a_is_transposed=False,
            b_is_transposed=True,
            has_bias=False,
        )
        for m in range(65, 129):
            with self.subTest(m=m):
                p = make_gemm_a16w16_param_and_validate(m, 3584, 7168, cfg)
                self.assertEqual(p.stages, 4)
                self.assertTrue(p.is_split_k)
                self.assertFalse(getattr(p, "packed_split", False))
                self.assertFalse(getattr(p, "unroll_lds_ring", False))
                self.assertFalse(p.n_grouped_loads)
                self.assertFalse(p.unroll_hti)
                self.assertEqual(p.split_init_ring, m == 128)


class TestSplitInitRingScope(unittest.TestCase):
    def test_only_exact_validated_policy_and_no_forced_override(self):
        from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
            _use_split_init_ring,
            make_gemm_a16w16_gfx950_kernel_name,
            make_gemm_a16w16_param_and_validate,
        )

        cfg = {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "stages": 4,
            "split_k": 4,
            "m_waves": 2,
            "n_waves": 2,
            "k_waves": 1,
            "group_m": 0,
            "use_half_tile_interleaved": False,
            "in_dtype_id": 2,
            "out_dtype_id": 2,
            "a_is_transposed": False,
            "b_is_transposed": True,
            "has_bias": False,
        }
        for shape in (
            (128, 3584, 7168),
            (127, 3584, 7168),
            (129, 3584, 7168),
            (128, 7168, 3584),
            (16384, 3584, 7168),
            (128, 3584, 3584),
        ):
            want = shape == (128, 3584, 7168)
            self.assertEqual(_use_split_init_ring(*shape, cfg), want)
            p = make_gemm_a16w16_param_and_validate(
                *shape, {**cfg, "split_init_ring": True}
            )
            if shape[0] == 16384:
                # Existing split-K tile-count capacity rejects this small tile.
                self.assertIsNone(p)
                continue
            self.assertIsNotNone(p)
            self.assertEqual(p.split_init_ring, want)
            self.assertEqual(
                make_gemm_a16w16_gfx950_kernel_name(p).endswith("_initring4"), want
            )
        for field, value in (
            ("block_m", 128),
            ("block_n", 128),
            ("block_k", 128),
            ("stages", 5),
            ("split_k", 2),
            ("m_waves", 4),
            ("n_waves", 1),
            ("k_waves", 2),
            ("group_m", 4),
            ("in_dtype_id", 3),
            ("out_dtype_id", 1),
            ("has_bias", True),
            ("a_is_transposed", True),
            ("b_is_transposed", False),
            ("use_half_tile_interleaved", True),
            ("b_non_temporal", True),
            ("k_stagger", 1),
            ("streamk_blocks", 224),
            ("mma_m", 32),
        ):
            with self.subTest(field=field):
                self.assertFalse(
                    _use_split_init_ring(128, 3584, 7168, {**cfg, field: value})
                )


class TestLatentSmallUpCsv(unittest.TestCase):
    def test_up_m128_csv_selects_counter_ws(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv"
        )
        with path.open() as source:
            rows = [
                r
                for r in csv.DictReader(source)
                if (
                    r["gfx"],
                    r["cu_num"],
                    r["M"],
                    r["N"],
                    r["K"],
                    r["bias"],
                    r["dtype"],
                    r["outdtype"],
                    r["scaleAB"],
                    r["bpreshuffle"],
                )
                == (
                    "gfx950",
                    "256",
                    "128",
                    "7168",
                    "3584",
                    "False",
                    BF16,
                    BF16,
                    "False",
                    "False",
                )
            ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            (rows[0]["libtype"], rows[0]["solidx"], rows[0]["splitK"]),
            ("flydsl", "0", "1"),
        )
        self.assertEqual(
            rows[0]["kernelName"],
            "flydsl_hgemm_ws_abf16_wbf16_bf16_t64x64x128x3_ks1_p2_w2x2_pf1_ch1_gfx950",
        )

    def test_ws_name_roundtrip_and_invalid_variants(self):
        import aiter.ops.flydsl.gemm_kernels as kernels

        name = kernels.WS_UP_M128_KERNEL_NAME
        params = kernels.get_flydsl_hgemm_kernel_params(name)
        self.assertEqual(
            kernels.flydsl_hgemm_kernel_name(
                dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                config=params,
                has_bias=False,
                target_gfx="gfx950",
            ),
            name,
        )
        for old, new in (
            ("ch1", "ch0"),
            ("p2_", "p4_"),
            ("ks1", "ks2"),
            ("gfx950", "gfx942"),
            ("x128x3", "x128x2"),
        ):
            self.assertIsNone(
                kernels.get_flydsl_hgemm_kernel_params(name.replace(old, new))
            )
        for field, value in (
            ("block_k", 64),
            ("producer_waves", True),
            ("counter_handshake", 1),
            ("prefetch", False),
        ):
            with self.assertRaises(ValueError):
                kernels.flydsl_hgemm_kernel_name(
                    dtype=torch.bfloat16,
                    out_dtype=torch.bfloat16,
                    config={**params, field: value},
                    has_bias=False,
                    target_gfx="gfx950",
                )

    def test_ws_csv_only_exact_m_and_compatible_keys(self):
        import aiter.ops.flydsl.gemm_kernels as kernels

        key = ("gfx950", 256, 128, 7168, 3584, False, BF16, BF16, False, False)
        row = {
            "libtype": "flydsl",
            "solidx": 0,
            "splitK": 1,
            "kernelName": kernels.WS_UP_M128_KERNEL_NAME,
        }
        with (
            mock.patch.object(tuned, "get_GEMM_A16W16_config_", lambda: {key: row}),
            mock.patch.object(tuned, "get_padded_m", lambda *_: 128),
        ):
            for m in (1, 64, 65, 127, 128, 129):
                actual = tuned.get_GEMM_A16W16_tuned_config(
                    m,
                    7168,
                    3584,
                    False,
                    BF16,
                    BF16,
                    gfx="gfx950",
                    cu_num=256,
                )
                self.assertEqual(
                    actual["kernelName"],
                    (
                        kernels.WS_UP_M128_KERNEL_NAME
                        if m == 128
                        else kernels.WS_UP_M128_LEGACY_KERNEL_NAME
                    ),
                )
            for index, value in (
                (0, "gfx942"),
                (1, 304),
                (3, 3584),
                (5, True),
                (6, "torch.float16"),
                (7, "torch.float32"),
                (8, True),
                (9, True),
            ):
                bad_key = list(key)
                bad_key[index] = value
                with mock.patch.object(
                    tuned,
                    "get_GEMM_A16W16_config_",
                    lambda bad_key=tuple(bad_key): {bad_key: row},
                ):
                    gfx, cu, m, n, k, bias, dtype, outdtype, scale, shuffle = bad_key
                    self.assertIsNone(
                        tuned.get_GEMM_A16W16_tuned_config(
                            m,
                            n,
                            k,
                            bias,
                            dtype,
                            outdtype,
                            scale,
                            shuffle,
                            gfx=gfx,
                            cu_num=cu,
                        )
                    )
        self.assertEqual(row["kernelName"], kernels.WS_UP_M128_KERNEL_NAME)

    def test_ws_ineligible_layout_uses_original_policy(self):
        import aiter.ops.flydsl.gemm_kernels as kernels

        a = torch.empty((2, 8), dtype=torch.bfloat16)
        b = torch.empty((8, 8), dtype=a.dtype)
        out = torch.empty((2, 8), dtype=a.dtype)
        with (
            mock.patch.object(kernels, "flydsl_hgemm", return_value=out) as legacy,
            mock.patch.object(tuned, "_flydsl_ws_launcher") as ws,
        ):
            result = tuned.flydsl_gemm(
                a,
                b,
                0,
                out=out,
                otype=a.dtype,
                config={"kernelName": kernels.WS_UP_M128_KERNEL_NAME},
            )
            self.assertIs(result, out)
            ws.assert_not_called()
            self.assertEqual(legacy.call_args.kwargs["block_k"], 64)
            self.assertEqual(legacy.call_args.kwargs["stages"], 7)


class TestExactLatentLargeCsv(unittest.TestCase):
    def test_new_m16384_row_does_not_change_padded_consumers(self):
        key = ("gfx950", 256, 16384, 7168, 3584, False, BF16, BF16, False, False)
        row = {
            "libtype": "flydsl",
            "solidx": 83,
            "splitK": 1,
            "kernelName": "flydsl_hgemm_abf16_wbf16_bf16_t256x256x64x2_ks1_w2x4x1_bias0_ktail0_gm4_pht_gfx950",
        }
        old_row = {
            "libtype": "asm",
            "solidx": 10,
            "splitK": 1,
            "kernelName": "_ZN5aiter24bf16gemm_bf16_tn_256x256E",
        }
        table = {key: row, (*key[:2], 8192, *key[3:]): old_row}
        saved = row.copy()
        with (
            mock.patch.object(tuned, "get_GEMM_A16W16_config_", lambda: table),
            mock.patch.object(
                tuned, "get_padded_m", lambda m, n, k, gl: 16384 if gl == 0 else 8192
            ),
        ):
            for m in range(16257, 16385):
                with self.subTest(m=m):
                    actual = tuned.get_GEMM_A16W16_tuned_config(
                        m,
                        7168,
                        3584,
                        False,
                        BF16,
                        BF16,
                        gfx="gfx950",
                        cu_num=256,
                    )
                    self.assertIs(actual, row if m == 16384 else old_row)
            # Explicit non-target rows still take precedence.
            exact_key = (*key[:2], 16383, *key[3:])
            table[exact_key] = row.copy()
            actual = tuned.get_GEMM_A16W16_tuned_config(
                16383,
                7168,
                3584,
                False,
                BF16,
                BF16,
                gfx="gfx950",
                cu_num=256,
            )
            self.assertIs(actual, table[exact_key])
        self.assertEqual(row, saved)


class TestGfx950ExtendedHgemm(unittest.TestCase):
    def test_names_and_declined_combinations(self):
        from aiter.ops.flydsl.gemm_kernels import (
            flydsl_hgemm_kernel_name,
            get_flydsl_hgemm_kernel_params,
        )

        base = {
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

        def name(
            config,
            gfx="gfx950",
            dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            bias=False,
        ):
            return flydsl_hgemm_kernel_name(
                config=config,
                target_gfx=gfx,
                dtype=dtype,
                out_dtype=out_dtype,
                has_bias=bias,
            )

        old = name(base)
        self.assertEqual(
            old,
            "flydsl_hgemm_abf16_wbf16_bf16_t64x64x64x3_ks1_w2x2x1_bias0_ktail0_gm0_pft_gfx950",
        )
        self.assertNotIn("mma_m", get_flydsl_hgemm_kernel_params(old))
        self.assertNotIn("streamk_blocks", get_flydsl_hgemm_kernel_params(old))
        for extension in (
            {"mma_m": 32, "mma_n": 32, "mma_k": 16},
            {"streamk_blocks": 7},
            {"mma_m": 32, "mma_n": 32, "mma_k": 16, "streamk_blocks": 13},
        ):
            config = {**base, **extension}
            parsed = get_flydsl_hgemm_kernel_params(name(config))
            self.assertEqual({key: parsed[key] for key in config}, config)
        for cfg, kw in (
            ({**base, "mma_m": 32, "mma_n": 32, "mma_k": 16, "k_waves": 2}, {}),
            ({**base, "mma_m": 32, "mma_n": 32, "mma_k": 16}, {"gfx": "gfx1250"}),
            (
                {**base, "mma_m": 32, "mma_n": 32, "mma_k": 16},
                {"dtype": torch.float16, "out_dtype": torch.float16},
            ),
            ({**base, "mma_m": 16.0}, {}),
            ({**base, "streamk_blocks": True}, {}),
            ({**base, "streamk_blocks": -1}, {}),
            ({**base, "streamk_blocks": 1025}, {}),
            ({**base, "streamk_blocks": 7, "split_k": 2}, {}),
            ({**base, "streamk_blocks": 7, "k_waves": 2}, {}),
            ({**base, "streamk_blocks": 7, "group_m": 4}, {}),
            ({**base, "streamk_blocks": 7, "k_stagger": 1}, {}),
            ({**base, "streamk_blocks": 7, "b_non_temporal": True}, {}),
            ({**base, "streamk_blocks": 7}, {"bias": True}),
            ({**base, "streamk_blocks": 7}, {"out_dtype": torch.float32}),
            ({**base, "streamk_blocks": 7, "policy": "sk"}, {}),
            ({**base, "streamk_blocks": 7, "use_half_tile_interleaved": True}, {}),
        ):
            with (
                self.subTest(config=cfg, kw=kw),
                self.assertRaises((ValueError, TypeError)),
            ):
                name(cfg, **kw)
        for bad in (
            old.replace("_pft", "_mi16x16x32_pft"),
            old.replace("_pft", "_mi32x32x32_pft"),
            old.replace("_pft", "_skg0_pft"),
            old.replace("_pft", "_skg1025_pft"),
            old.replace("_pft", "_mi32x32x16_pht"),
        ):
            self.assertIsNone(get_flydsl_hgemm_kernel_params(bad))

    def test_balanced_interval_and_slot_bound(self):
        # The device inverts floor(w*total/workers). No duplicate or missing
        # K tiles; each output tile's participants occupy contiguous slots.
        for tiles in (1, 2, 3, 7, 8, 32, 112):
            for kt in (1, 2, 3, 5, 14, 28):
                total = tiles * kt
                for workers in sorted(
                    {1, min(7, total), min(13, total), min(224, total), total}
                ):
                    slots = (workers + tiles - 1) // tiles + 1
                    cover = [0] * total
                    participants = [[] for _ in range(tiles)]
                    for w in range(workers):
                        begin = w * total // workers
                        end = (w + 1) * total // workers
                        for i in range(begin, end):
                            cover[i] += 1
                        for tile in range(begin // kt, (end - 1) // kt + 1):
                            participants[tile].append(w)
                    self.assertEqual(set(cover), {1})
                    for tile, ps in enumerate(participants):
                        first = ((tile * kt + 1) * workers - 1) // total
                        last = ((tile + 1) * kt * workers - 1) // total
                        self.assertEqual(ps, list(range(first, last + 1)))
                        self.assertLessEqual(len(ps), slots)

    def test_extra_candidates_validated_and_appended(self):
        import aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 as impl
        from aiter.ops.flydsl import gemm_a16w16_policy as policy

        with mock.patch.object(impl, "get_rocm_arch", return_value="gfx950"):
            cs = policy._gfx950_extended_configs(
                128, 3584, 7168, torch.bfloat16, torch.bfloat16, False
            )
        self.assertTrue(any(c["block_n"] == 112 for c in cs))
        self.assertTrue(any(c["block_n"] == 224 for c in cs))
        self.assertTrue(any(c.get("mma_m") == 32 for c in cs))
        self.assertTrue(any(c.get("streamk_blocks") for c in cs))
        self.assertEqual(len(cs), len({tuple(sorted(c.items())) for c in cs}))
        self.assertEqual(
            policy._gfx950_extended_configs(
                129, 3584, 7168, torch.bfloat16, torch.bfloat16, False
            ),
            [],
        )
        self.assertEqual(
            policy._gfx950_extended_configs(
                128, 3584, 7168, torch.float16, torch.float16, False
            ),
            [],
        )
        with mock.patch.object(impl, "get_rocm_arch", return_value="gfx950"):
            for bias, outdtype in ((True, torch.bfloat16), (False, torch.float32)):
                cs = policy._gfx950_extended_configs(
                    128, 3584, 7168, torch.bfloat16, outdtype, bias
                )
                self.assertFalse(any(c.get("streamk_blocks") for c in cs))

    def test_invalid_types_rejected_after_cache_warmup(self):
        import aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 as impl

        cfg = {
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
            "in_dtype_id": 2,
            "out_dtype_id": 2,
            "streamk_blocks": 1,
            "mma_m": 16,
            "mma_n": 16,
            "mma_k": 32,
        }
        with mock.patch.object(impl, "get_rocm_arch", return_value="gfx950"):
            self.assertIsNotNone(
                impl.make_gemm_a16w16_param_and_validate(128, 256, 256, cfg)
            )
            for field, value in (
                ("streamk_blocks", True),
                ("streamk_blocks", 1.0),
                ("mma_m", 16.0),
                ("mma_n", 16.0),
                ("mma_k", 32.0),
            ):
                with self.subTest(field=field, value=value):
                    self.assertIsNone(
                        impl.make_gemm_a16w16_param_and_validate(
                            128, 256, 256, {**cfg, field: value}
                        )
                    )

    def test_extended_k_layout_rejects(self):
        import aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 as impl

        cfg = {
            "block_m": 64,
            "block_n": 64,
            "block_k": 32,
            "stages": 2,
            "split_k": 1,
            "m_waves": 1,
            "n_waves": 1,
            "k_waves": 1,
            "group_m": 0,
            "use_half_tile_interleaved": False,
            "in_dtype_id": 2,
            "out_dtype_id": 2,
        }
        with mock.patch.object(impl, "get_rocm_arch", return_value="gfx950"):
            for ext in ({"streamk_blocks": 4}, {"mma_m": 32, "mma_n": 32, "mma_k": 16}):
                with self.assertRaisesRegex(ValueError, "block_k divisible by 64"):
                    impl.make_gemm_a16w16_gfx950_param(**cfg, **ext)

    def test_stream_bounds_and_arch_reject(self):
        import aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 as impl

        cfg = {
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
            "in_dtype_id": 2,
            "out_dtype_id": 2,
            "streamk_blocks": 7,
        }
        with mock.patch.object(impl, "get_rocm_arch", return_value="gfx950"):
            self.assertIsNotNone(
                impl.make_gemm_a16w16_param_and_validate(128, 256, 64, cfg)
            )
            for m, n, k in (
                (0, 256, 64),
                (64, 64, 64),
                (128, 256, 72),
                (65536, 65536, 7168),
            ):
                self.assertIsNone(
                    impl.make_gemm_a16w16_param_and_validate(m, n, k, cfg)
                )
        with (
            mock.patch.object(impl, "get_rocm_arch", return_value="gfx942"),
            self.assertRaises(ValueError),
        ):
            impl.make_gemm_a16w16_gfx950_param(**cfg)


class TestWaveSpecializedConfig(unittest.TestCase):
    def test_valid_geometry_and_limits(self):
        from aiter.ops.flydsl.gemm_a16w16_ws import _validate_ws_config

        base = (64, 64, 128, 3, 2, 2, 2, 1, True)
        self.assertEqual(_validate_ws_config(*base), base[:-1])
        self.assertEqual(_validate_ws_config(*base, counter_handshake=True), base[:-1])
        for value in (1, None, "true"):
            with self.assertRaises(ValueError):
                _validate_ws_config(*base, counter_handshake=value)
        for index, value in (
            (0, 63),
            (1, 63),
            (2, 32),
            (3, 1),
            (3, 6),
            (4, 0),
            (4, 3),
            (4, True),
            (5, 3),
            (7, 0),
            (7, 17),
            (8, 1),
        ):
            args = list(base)
            args[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                _validate_ws_config(*args)
        with self.assertRaisesRegex(ValueError, "LDS"):
            _validate_ws_config(256, 256, 128, 5, 4, 2, 4, 1, False)

    def test_dma_coverage_and_sequence_generations(self):
        # Each physical packet belongs to one producer lane; each consumer
        # must release a slot before the next generation reuses it.
        for bm, bn, bk, pr in ((64, 64, 64, 1), (32, 64, 128, 2), (256, 128, 64, 4)):
            for size in (bm * bk, bn * bk):
                offsets = [
                    (p * 64 + lane + i * pr * 64) * 8
                    for i in range(size // (pr * 64 * 8))
                    for p in range(pr)
                    for lane in range(64)
                ]
                self.assertEqual(sorted(offsets), list(range(0, size, 8)))
            for stages in (2, 3, 4, 5):
                for tiles in (1, 2, 3, 7, 28, 112):
                    last = [0] * stages
                    for tile in range(tiles):
                        slot, gen = tile % stages, tile // stages
                        self.assertEqual(last[slot], gen)
                        last[slot] = gen + 1

    def test_cumulative_counter_interleavings(self):
        # A randomized abstract scheduler checks the cumulative thresholds,
        # not GPU memory ordering (covered separately by GPU replay tests).
        import random

        for stages in (2, 3, 4, 5):
            for producers, consumers in ((1, 2), (2, 4), (4, 8)):
                for seed in range(8):
                    rng = random.Random(seed)
                    tiles = 31
                    ready, released = [0] * stages, [0] * stages
                    prod, cons = [0] * producers, [0] * consumers
                    writes, reads = set(), set()
                    while min(cons) < tiles:
                        runnable = []
                        for p, tile in enumerate(prod):
                            slot, gen = tile % stages, tile // stages
                            if tile < tiles and released[slot] >= gen * consumers:
                                runnable.append(("produce", p, tile))
                        for c, tile in enumerate(cons):
                            slot, gen = tile % stages, tile // stages
                            if tile < tiles and ready[slot] >= (gen + 1) * producers:
                                runnable.append(("consume", c, tile))
                        self.assertTrue(runnable, "protocol deadlocked")
                        role, wave, tile = rng.choice(runnable)
                        slot = tile % stages
                        if role == "produce":
                            if tile >= stages:
                                for c in range(consumers):
                                    self.assertIn((tile - stages, c), reads)
                            self.assertNotIn((tile, wave), writes)
                            writes.add((tile, wave))
                            ready[slot] += 1
                            prod[wave] += 1
                        else:
                            for p in range(producers):
                                self.assertIn((tile, p), writes)
                            self.assertNotIn((tile, wave), reads)
                            reads.add((tile, wave))
                            released[slot] += 1
                            cons[wave] += 1
                    self.assertEqual(len(writes), tiles * producers)
                    self.assertEqual(len(reads), tiles * consumers)


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
            mock.patch.object(policy, "_gfx950_extended_configs", return_value=[]),
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
            if name in seen or "_stg" in name or "_bnt_" in name or "hgemm_ws_" in name:
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
