# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import aiter.tuned_gemm as tuned
from aiter.execution_context import (
    execution_phase,
    execution_role,
    get_execution_phase,
    get_execution_role,
)


class TestExecutionContext(unittest.TestCase):
    def test_nested_context_and_exception_restore_defaults(self):
        self.assertIsNone(get_execution_role())
        self.assertIsNone(get_execution_phase())
        with execution_role("target"), execution_phase("prefill"):
            with self.assertRaises(RuntimeError), execution_role(
                "draft"
            ), execution_phase("decode"):
                self.assertEqual(get_execution_role(), "draft")
                self.assertEqual(get_execution_phase(), "decode")
                raise RuntimeError("restore context")
            self.assertEqual(get_execution_role(), "target")
            self.assertEqual(get_execution_phase(), "prefill")

            def other_runner():
                with execution_role("draft"), execution_phase("verify"):
                    return get_execution_role(), get_execution_phase()

            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertEqual(
                    pool.submit(other_runner).result(), ("draft", "verify")
                )
            self.assertEqual(get_execution_role(), "target")
            self.assertEqual(get_execution_phase(), "prefill")
        self.assertIsNone(get_execution_role())
        self.assertIsNone(get_execution_phase())

    def test_dispatch_cache_separates_role_and_phase(self):
        tuned.get_GEMM_A16W16_config.cache_clear()
        self.addCleanup(tuned.get_GEMM_A16W16_config.cache_clear)
        args = {
            "M": 96,
            "N": 4096,
            "K": 1536,
            "bias": False,
            "dtype": "torch.bfloat16",
            "otype": "torch.bfloat16",
        }
        ordinary = {"libtype": "triton", "solidx": 0}
        with mock.patch.object(
            tuned, "get_gfx", return_value="gfx950"
        ), mock.patch.object(tuned, "get_cu_num", return_value=256), mock.patch.object(
            tuned, "_opus_launch", mock.Mock()
        ), mock.patch.object(
            tuned, "get_GEMM_A16W16_tuned_config", return_value=ordinary
        ):
            for _ in range(2):
                for role in (None, "target", "draft"):
                    for phase in (None, "decode", "verify", "prefill"):
                        with self.subTest(role=role, phase=phase):
                            config = tuned.get_GEMM_A16W16_config(
                                **args, execution_role=role, execution_phase=phase
                            )
                            if (role, phase) == ("target", "prefill"):
                                self.assertEqual(config["libtype"], "opus")
                                self.assertEqual(config["solidx"], 138)
                                self.assertEqual(config["splitK"], 0)
                            else:
                                self.assertEqual(config, ordinary)

    def test_phase_override_is_limited_to_supported_family(self):
        args = {
            "M": 96,
            "N": 4096,
            "K": 1536,
            "bias": False,
            "dtype": "torch.bfloat16",
            "otype": "torch.bfloat16",
            "scaleAB": False,
            "bpreshuffle": False,
            "execution_role": "target",
            "execution_phase": "prefill",
            "gfx": "gfx950",
            "cu_num": 256,
        }
        self.assertIsNotNone(tuned._get_execution_phase_invariant_config(**args))
        for changes in (
            {"N": 4097},
            {"K": 1537},
            {"bias": True},
            {"dtype": "torch.float16"},
            {"otype": "torch.float32"},
            {"scaleAB": True},
            {"bpreshuffle": True},
            {"gfx": "gfx942"},
            {"cu_num": 304},
        ):
            with self.subTest(changes=changes):
                self.assertIsNone(
                    tuned._get_execution_phase_invariant_config(**(args | changes))
                )

    def test_prefill_tile_boundaries_and_asm_fallback(self):
        from csrc.opus_gemm.opus_gemm_common import DEFAULT_COMPILED_KIDS_GFX950

        args = {
            "N": 4096,
            "K": 1536,
            "bias": False,
            "dtype": "torch.bfloat16",
            "otype": "torch.bfloat16",
            "scaleAB": False,
            "bpreshuffle": False,
            "execution_role": "target",
            "execution_phase": "prefill",
            "gfx": "gfx950",
            "cu_num": 256,
        }
        self.assertTrue({137, 138} <= DEFAULT_COMPILED_KIDS_GFX950)
        with mock.patch.object(tuned, "_opus_launch", mock.Mock()):
            for m, backend, kid in (
                (1, "opus", 138),
                (1024, "opus", 138),
                (1025, "opus", 137),
                (1536, "opus", 137),
                (1537, "asm", 10),
                (8192, "asm", 10),
            ):
                with self.subTest(m=m):
                    config = tuned._get_execution_phase_invariant_config(M=m, **args)
                    self.assertEqual(
                        (config["libtype"], config["solidx"]), (backend, kid)
                    )

        with mock.patch.object(tuned, "_opus_launch", None):
            config = tuned._get_execution_phase_invariant_config(M=1, **args)
            self.assertEqual((config["libtype"], config["solidx"]), ("asm", 10))
        with mock.patch.object(tuned, "_opus_launch", mock.Mock()), mock.patch(
            "aiter.ops.opus.policy.resolve_a16w16_tuned_candidate", return_value=None
        ):
            config = tuned._get_execution_phase_invariant_config(M=1025, **args)
            self.assertEqual((config["libtype"], config["solidx"]), ("asm", 10))

    def test_prefill_uses_explicit_device_architecture(self):
        tuned.get_GEMM_A16W16_config.cache_clear()
        self.addCleanup(tuned.get_GEMM_A16W16_config.cache_clear)
        args = {
            "M": 96,
            "N": 4096,
            "K": 1536,
            "bias": False,
            "dtype": "torch.bfloat16",
            "otype": "torch.bfloat16",
            "execution_role": "target",
            "execution_phase": "prefill",
        }
        ordinary = {"libtype": "triton", "solidx": 0}
        with mock.patch.object(
            tuned, "get_gfx", side_effect=AssertionError
        ), mock.patch.object(
            tuned, "get_cu_num", side_effect=AssertionError
        ), mock.patch.object(
            tuned, "_opus_launch", mock.Mock()
        ), mock.patch.object(
            tuned, "get_GEMM_A16W16_tuned_config", return_value=ordinary
        ):
            config = tuned.get_GEMM_A16W16_config(**args, gfx="gfx950", cu_num=256)
            self.assertEqual((config["libtype"], config["solidx"]), ("opus", 138))
            self.assertEqual(
                tuned.get_GEMM_A16W16_config(**args, gfx="gfx950", cu_num=304),
                ordinary,
            )


if __name__ == "__main__":
    unittest.main()
