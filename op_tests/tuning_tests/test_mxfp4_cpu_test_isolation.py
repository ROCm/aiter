# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression checks for MXMOE CPU tests sharing a process with other tests."""

import contextlib
import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from op_tests.tuning_tests import mxfp4_cpu_test_utils as cpu_utils


class TestMxfp4CpuIsolation(unittest.TestCase):
    def test_input_and_result_cases_restore_state_in_either_order(self) -> None:
        from op_tests.tuning_tests.test_mxfp4_flydsl_results import (
            TestCoupledTuningResults,
        )
        from op_tests.tuning_tests.test_mxfp4_flydsl_tuning import (
            TestMxfp4FlydslInputsAndCandidates,
        )

        previous = cpu_utils.get_configured_default_device()
        self.addCleanup(torch.set_default_device, previous)
        environment = dict(os.environ)
        initialized = torch.cuda.is_initialized()
        cases = (
            (
                TestMxfp4FlydslInputsAndCandidates,
                "test_missing_required_column_reports_the_input_file",
            ),
            (
                TestCoupledTuningResults,
                "test_csv_saves_fastest_valid_winner_and_all_candidate_observations",
            ),
        )
        for device in ("cpu", "cuda", "cuda:3"):
            for order in (cases, tuple(reversed(cases))):
                with self.subTest(
                    device=device, order=[case[0].__name__ for case in order]
                ):
                    torch.set_default_device(device)
                    suite = unittest.TestSuite(cls(method) for cls, method in order)
                    stream = io.StringIO()
                    with contextlib.redirect_stdout(stream):
                        result = unittest.TextTestRunner(stream=stream).run(suite)
                    self.assertTrue(result.wasSuccessful(), stream.getvalue())
                    self.assertEqual(
                        cpu_utils.get_configured_default_device(),
                        torch.device(device),
                    )
                    self.assertEqual(dict(os.environ), environment)
                    self.assertEqual(torch.cuda.is_initialized(), initialized)
        torch.set_default_device("cpu")
        self.assertEqual(torch.empty(1).device, torch.device("cpu"))

    def test_default_device_query_uses_public_fallback_without_private_context(
        self,
    ) -> None:
        for private in ({}, {"_GLOBAL_DEVICE_CONTEXT": SimpleNamespace()}):
            query = Mock(return_value=torch.device("cuda:3"))
            fake_torch = SimpleNamespace(
                **private,
                device=torch.device,
                get_default_device=query,
                cuda=SimpleNamespace(_lazy_init=lambda: None),
            )
            with self.subTest(private=private), patch.object(
                cpu_utils, "torch", fake_torch
            ):
                self.assertEqual(
                    cpu_utils.get_configured_default_device(), torch.device("cuda:3")
                )
            query.assert_called_once_with()

    def test_unreadable_default_device_fails_before_environment_changes(self) -> None:
        def query():
            fake_torch.cuda._lazy_init()
            return torch.device("cuda")

        fake_torch = SimpleNamespace(
            device=torch.device,
            get_default_device=query,
            set_default_device=Mock(),
            cuda=SimpleNamespace(_lazy_init=lambda: None),
        )
        environment = dict(os.environ)
        with (
            patch.object(cpu_utils, "torch", fake_torch),
            self.assertRaisesRegex(
                RuntimeError, "cannot safely read.*no device setting was changed"
            ),
            cpu_utils.cpu_tuner_environment(),
        ):
            self.fail("unreadable device must reject the CPU context before entry")
        fake_torch.set_default_device.assert_not_called()
        self.assertEqual(dict(os.environ), environment)


if __name__ == "__main__":
    unittest.main()
