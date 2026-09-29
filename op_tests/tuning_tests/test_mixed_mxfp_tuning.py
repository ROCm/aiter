# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import csv
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch

from aiter.jit.core import AITER_CONFIGS
from aiter.ops import gemm_op_a4w6, gemm_op_a6w4
from aiter.ops.gemm_op_mixed_mxfp import (
    _get_device_gfx_cu,
    _load_mixed_mxfp_configs,
)
from csrc.gemm_mixed_mxfp import gemm_mixed_mxfp_tune
from csrc.gemm_mixed_mxfp.gemm_mixed_mxfp_tune import (
    GemmMixedMxfpTuner,
    candidate_supports_shape,
    choose_guarded_kernel,
    load_mixed_mxfp_candidates,
)

TUNED_HEADER = [
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "kernelId",
    "splitK",
    "us",
    "kernelName",
    "tflops",
    "bw",
    "errRatio",
]
FAMILIES = (
    ("a6w4", gemm_op_a6w4),
    ("a4w6", gemm_op_a4w6),
)
KERNEL_NAMES = {
    "a6w4": (
        "_ZN5aiter40f6f4gemm_bf16_per1x32Fp6Fp4_m32_s0_a4_ntE",
        "_ZN5aiter39f6f4gemm_bf16_per1x32Fp6Fp4_m32_s0_a4_tE",
    ),
    "a4w6": (
        "_ZN5aiter40f4f6gemm_bf16_per1x32Fp4Fp6_m32_s0_a5_ntE",
        "_ZN5aiter39f4f6gemm_bf16_per1x32Fp4Fp6_m32_s0_a4_tE",
    ),
}


class TestMixedMxfpTuningLookup(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        for _family, module in FAMILIES:
            getattr(module, f"clear_gemm_{_family}_config_cache")()

    def tearDown(self):
        for _family, module in FAMILIES:
            getattr(module, f"clear_gemm_{_family}_config_cache")()
        self.tempdir.cleanup()

    def _write_rows(self, family, rows):
        config = os.path.join(self.tempdir.name, f"{family}.csv")
        with open(config, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(TUNED_HEADER)
            writer.writerows(rows)
        _load_mixed_mxfp_configs.cache_clear()
        return config

    def test_exact_match_precedes_padded_match(self):
        for family, module in FAMILIES:
            with self.subTest(family=family):
                exact_kernel, padded_kernel = KERNEL_NAMES[family]
                config = self._write_rows(
                    family,
                    [
                        [
                            "gfx950",
                            256,
                            9450,
                            5120,
                            5120,
                            1,
                            0,
                            1,
                            exact_kernel,
                            1,
                            1,
                            0,
                        ],
                        [
                            "gfx950",
                            256,
                            9472,
                            5120,
                            5120,
                            2,
                            0,
                            2,
                            padded_kernel,
                            1,
                            1,
                            0,
                        ],
                    ],
                )
                with (
                    mock.patch.object(module, "get_gfx", return_value="gfx950"),
                    mock.patch.object(module, "get_cu_num", return_value=256),
                ):
                    record = getattr(module, f"get_GEMM_{family.upper()}_config")(
                        9450, 5120, 5120, config
                    )
                self.assertEqual(record["kernelName"], exact_kernel)

    def test_padded_match_is_used_as_fallback(self):
        for family, module in FAMILIES:
            with self.subTest(family=family):
                padded_kernel = KERNEL_NAMES[family][1]
                config = self._write_rows(
                    family,
                    [
                        [
                            "gfx950",
                            256,
                            9472,
                            5120,
                            5120,
                            1,
                            0,
                            1,
                            padded_kernel,
                            1,
                            1,
                            0,
                        ]
                    ],
                )
                with (
                    mock.patch.object(module, "get_gfx", return_value="gfx950"),
                    mock.patch.object(module, "get_cu_num", return_value=256),
                ):
                    record = getattr(module, f"get_GEMM_{family.upper()}_config")(
                        9450, 5120, 5120, config
                    )
                self.assertEqual(record["kernelName"], padded_kernel)

    def test_device_identity_participates_in_lookup_cache(self):
        for family, module in FAMILIES:
            with self.subTest(family=family):
                gfx950_kernel, gfx942_kernel = KERNEL_NAMES[family]
                config = self._write_rows(
                    family,
                    [
                        [
                            "gfx950",
                            256,
                            256,
                            256,
                            128,
                            0,
                            0,
                            1,
                            gfx950_kernel,
                            1,
                            1,
                            0,
                        ],
                        [
                            "gfx942",
                            304,
                            256,
                            256,
                            128,
                            1,
                            0,
                            1,
                            gfx942_kernel,
                            1,
                            1,
                            0,
                        ],
                    ],
                )
                get_config = getattr(module, f"get_GEMM_{family.upper()}_config")
                with mock.patch.object(
                    module,
                    "_get_device_gfx_cu",
                    side_effect=[("gfx950", 256), ("gfx942", 304)],
                ):
                    first = get_config(
                        256,
                        256,
                        128,
                        config,
                        device=torch.device("cuda:0"),
                    )
                    second = get_config(
                        256,
                        256,
                        128,
                        config,
                        device=torch.device("cuda:1"),
                    )
                self.assertEqual(first["kernelName"], gfx950_kernel)
                self.assertEqual(second["kernelName"], gfx942_kernel)

    def test_device_identity_honors_cu_num_override(self):
        properties = SimpleNamespace(
            gcnArchName="gfx950:sramecc+:xnack-",
            multi_processor_count=256,
        )
        with (
            mock.patch.dict(os.environ, {"CU_NUM": "80"}),
            mock.patch.object(
                torch.cuda, "get_device_properties", return_value=properties
            ),
        ):
            _get_device_gfx_cu.cache_clear()
            try:
                self.assertEqual(_get_device_gfx_cu(0), ("gfx950", 80))
            finally:
                _get_device_gfx_cu.cache_clear()

    def test_nonzero_splitk_and_duplicate_shapes_are_rejected(self):
        for family, _module in FAMILIES:
            with self.subTest(family=family, case="splitK"):
                bad_split = self._write_rows(
                    family,
                    [
                        [
                            "gfx950",
                            256,
                            256,
                            256,
                            128,
                            1,
                            1,
                            1,
                            "kernel",
                            1,
                            1,
                            0,
                        ]
                    ],
                )
                with self.assertRaisesRegex(ValueError, "splitK=0"):
                    _load_mixed_mxfp_configs(bad_split, family.upper())

            with self.subTest(family=family, case="duplicate"):
                row = [
                    "gfx950",
                    256,
                    256,
                    256,
                    128,
                    1,
                    0,
                    1,
                    "kernel",
                    1,
                    1,
                    0,
                ]
                duplicate = self._write_rows(family, [row, row])
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    _load_mixed_mxfp_configs(duplicate, family.upper())

    def test_fractional_and_empty_values_are_rejected(self):
        for family, _module in FAMILIES:
            valid_kernel = KERNEL_NAMES[family][0]
            cases = (
                ("fractional", 0.5, valid_kernel, "non-integral"),
                ("empty", 0, "", "empty"),
            )
            for case, split_k, kernel_name, message in cases:
                with self.subTest(family=family, case=case):
                    config = self._write_rows(
                        family,
                        [
                            [
                                "gfx950",
                                256,
                                256,
                                256,
                                128,
                                1,
                                split_k,
                                1,
                                kernel_name,
                                1,
                                1,
                                0,
                            ]
                        ],
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        _load_mixed_mxfp_configs(config, family.upper())

            with self.subTest(family=family, case="nonpositive"):
                config = self._write_rows(
                    family,
                    [
                        [
                            "gfx950",
                            256,
                            0,
                            256,
                            128,
                            1,
                            0,
                            1,
                            valid_kernel,
                            1,
                            1,
                            0,
                        ]
                    ],
                )
                with self.assertRaisesRegex(ValueError, "non-positive M"):
                    _load_mixed_mxfp_configs(config, family.upper())

    def test_selector_priority_and_low_level_default(self):
        for family, module in FAMILIES:
            with self.subTest(family=family):
                select = getattr(module, f"_select_gemm_{family}_kernel")
                self.assertEqual(select(1, 1, 1, "explicit_kernel"), "explicit_kernel")
                get_config = f"get_GEMM_{family.upper()}_config"
                with mock.patch.object(
                    module,
                    get_config,
                    return_value={"kernelName": "tuned_kernel"},
                ):
                    self.assertEqual(select(256, 256, 128, None), "tuned_kernel")

                packed = torch.empty(0, dtype=torch.uint8, device="meta")
                out = torch.empty((512, 512), dtype=torch.bfloat16, device="meta")
                launch_name = f"_gemm_{family}_asm"
                public_asm = getattr(module, f"gemm_{family}_asm")
                default = getattr(module, f"_default_gemm_{family}_kernel")
                with mock.patch.object(module, launch_name) as launch:
                    public_asm(packed, packed, packed, packed, out, 128)
                self.assertEqual(launch.call_args.args[6], default(512, 512, 128))


class TestMixedMxfpCandidates(unittest.TestCase):
    def test_manifests_enumerate_compatible_candidates(self):
        for family, _module in FAMILIES:
            with self.subTest(family=family):
                candidates = load_mixed_mxfp_candidates(family)
                self.assertGreaterEqual(len(candidates), 4)
                self.assertEqual(
                    len({candidate["kernel_name"] for candidate in candidates}),
                    len(candidates),
                )

    def test_candidate_swizzle_bounds(self):
        bounded = {
            "kernel_id": 0,
            "kernel_name": "bounded",
            "tile_m": 256,
            "tile_n": 256,
            "swizzle_max_m": 131072,
            "swizzle_max_n": 16384,
            "swizzle_max_k": 6144,
        }
        self.assertTrue(candidate_supports_shape(bounded, 9450, 13824, 5120))
        self.assertFalse(candidate_supports_shape(bounded, 9450, 55296, 5120))
        self.assertTrue(candidate_supports_shape(bounded, 9450, 55296, 13824))


class TestMixedMxfpMinimumGainGuard(unittest.TestCase):
    def test_keeps_only_stable_exact_improvements(self):
        selected, gain_pct = choose_guarded_kernel(
            "default", "candidate", [1.025, 1.03, 1.035], 2.0, True
        )
        self.assertEqual(selected, "candidate")
        self.assertAlmostEqual(gain_pct, 3.0)

        selected, _gain_pct = choose_guarded_kernel(
            "default", "candidate", [1.02, 0.999, 1.03], 0.5, True
        )
        self.assertEqual(selected, "default")

        selected, gain_pct = choose_guarded_kernel(
            "default", "candidate", [1.5, 1.6], 0.5, False
        )
        self.assertEqual(selected, "default")
        self.assertEqual(gain_pct, float("-inf"))


class TestMixedMxfpTunerState(unittest.TestCase):
    def test_empty_config_environment_uses_canonical_default(self):
        for family, _module in FAMILIES:
            with self.subTest(family=family):
                tuner = GemmMixedMxfpTuner(family, f"test_{family}")
                env_name = tuner.spec["config_env_name"]
                property_name = f"AITER_CONFIG_GEMM_{family.upper()}_ASM_FILE"
                old_value = os.environ.get(env_name)
                try:
                    os.environ[env_name] = "  "
                    type(AITER_CONFIGS).get_config_file.cache_clear()
                    resolved = getattr(AITER_CONFIGS, property_name)
                    self.assertTrue(resolved.endswith(f"{family}_asm_tuned_gemm.csv"))
                finally:
                    if old_value is None:
                        os.environ.pop(env_name, None)
                    else:
                        os.environ[env_name] = old_value
                    type(AITER_CONFIGS).get_config_file.cache_clear()

    def test_bare_run_config_retains_canonical_untuned_shapes(self):
        for family, _module in FAMILIES:
            with self.subTest(family=family):
                tuner = GemmMixedMxfpTuner(family, f"test_{family}")
                defaults = tuner.get_arg_defaults()
                with (
                    mock.patch.dict(os.environ, {"CU_NUM": "80"}),
                    mock.patch.object(tuner, "get_gfx", return_value="gfx950"),
                    mock.patch.object(
                        torch.cuda,
                        "current_device",
                        side_effect=AssertionError("CPU-only test queried CUDA"),
                    ),
                ):
                    tuner.pre_process(
                        SimpleNamespace(
                            run_config=True,
                            untune_file=defaults["untune_file"],
                            tune_file=defaults["tune_file"],
                        )
                    )
                self.assertEqual(len(tuner.untunedf), 57)
                self.assertEqual(set(tuner.untunedf["cu_num"]), {80})

    def test_restore_config_env_clears_runtime_caches(self):
        for family, module in FAMILIES:
            with self.subTest(family=family):
                tuner = GemmMixedMxfpTuner(family, f"test_{family}")
                env_name = tuner.spec["config_env_name"]
                property_name = f"AITER_CONFIG_GEMM_{family.upper()}_ASM_FILE"
                old_value = os.environ.get(env_name)
                temporary_path = os.path.join(self.temp_path, f"{family}.csv")
                try:
                    os.environ[env_name] = temporary_path
                    type(AITER_CONFIGS).get_config_file.cache_clear()
                    self.assertEqual(
                        getattr(AITER_CONFIGS, property_name), temporary_path
                    )
                    get_config = getattr(module, f"get_GEMM_{family.upper()}_config")
                    cached_get_config = getattr(
                        module, f"_get_GEMM_{family.upper()}_config_cached"
                    )
                    cached_get_config.cache_clear()
                    with (
                        mock.patch.object(module, "get_gfx", return_value="gfx950"),
                        mock.patch.object(module, "get_cu_num", return_value=256),
                        mock.patch.object(
                            torch.cuda,
                            "current_device",
                            side_effect=AssertionError("CPU-only test queried CUDA"),
                        ),
                    ):
                        get_config(1, 1, 1, temporary_path)
                    self.assertGreater(cached_get_config.cache_info().currsize, 0)

                    tuner._restore_config_env(env_name, old_value, 0)
                    self.assertEqual(cached_get_config.cache_info().currsize, 0)
                    self.assertNotEqual(
                        getattr(AITER_CONFIGS, property_name), temporary_path
                    )
                finally:
                    if old_value is None:
                        os.environ.pop(env_name, None)
                    else:
                        os.environ[env_name] = old_value
                    type(AITER_CONFIGS).get_config_file.cache_clear()
                    getattr(module, f"clear_gemm_{family}_config_cache")()

    @property
    def temp_path(self):
        if not hasattr(self, "_tempdir"):
            self._tempdir = tempfile.TemporaryDirectory()
            self.addCleanup(self._tempdir.cleanup)
        return self._tempdir.name

    def test_gain_guard_remeasures_default_after_candidate_error(self):
        tuner = GemmMixedMxfpTuner("a6w4", "test_a6w4")
        candidate = next(
            candidate
            for candidate in tuner.candidates
            if candidate["kernel_name"]
            != gemm_op_a6w4._default_gemm_a6w4_kernel(9450, 5120, 5120)
        )
        row = pd.DataFrame(
            [
                {
                    "gfx": "gfx950",
                    "cu_num": 256,
                    "M": 9450,
                    "N": 5120,
                    "K": 5120,
                    "kernelId": candidate["kernel_id"],
                    "splitK": 0,
                    "us": 1.0,
                    "kernelName": candidate["kernel_name"],
                    "tflops": 1.0,
                    "bw": 1.0,
                    "errRatio": 0.0,
                }
            ]
        )
        fallback = {
            "default_us": 125.0,
            "candidate_us": 125.0,
            "paired_speedups": [1.0],
            "exact_match": True,
        }
        args = SimpleNamespace(
            disable_gain_guard=False,
            min_gain_pct=0.5,
            gain_guard_warmup=1,
            gain_guard_iters=1,
            gain_guard_reps=1,
        )
        with mock.patch.object(
            gemm_mixed_mxfp_tune,
            "benchmark_candidate_gain",
            side_effect=[RuntimeError("candidate failed"), fallback],
        ):
            guarded = tuner._apply_min_gain_guard(row, args)

        self.assertEqual(
            guarded.iloc[0]["kernelName"],
            gemm_op_a6w4._default_gemm_a6w4_kernel(9450, 5120, 5120),
        )
        self.assertEqual(float(guarded.iloc[0]["us"]), 125.0)
        self.assertNotEqual(float(guarded.iloc[0]["tflops"]), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
