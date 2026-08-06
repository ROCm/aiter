#!/usr/bin/env python3
"""CPU-only unit tests for tdm_adapter capture and comparison helpers."""

import inspect
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from tdm_adapter import (
    _POISON,
    _compare_outputs,
    _find_output_tensor,
    _iqr_trimmed_median_us,
    Capture,
    IsaRunnerError,
    replay,
)


class TdmAdapterInterfaceTest(unittest.TestCase):
    def test_replay_has_no_optional_check_or_reference_isa(self):
        parameters = inspect.signature(replay).parameters

        self.assertNotIn("check", parameters)
        self.assertNotIn("reference_isa", parameters)

    def test_capture_has_no_reference_isa_state(self):
        self.assertNotIn("reference_isa", Capture.__dataclass_fields__)

    def test_replay_defaults_to_flydsl_style_l2_flush(self):
        parameters = inspect.signature(replay).parameters

        self.assertIn("flush_l2", parameters)
        self.assertTrue(parameters["flush_l2"].default)

    def test_iqr_trimmed_median_rejects_outlier(self):
        median, raw_count, filtered_count = _iqr_trimmed_median_us(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0]
        )

        self.assertEqual(median, 1.0)
        self.assertEqual(raw_count, 8)
        self.assertEqual(filtered_count, 7)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TdmAdapterHelpersTest(unittest.TestCase):
    def test_bfloat16_poison_padding_is_masked_before_float_conversion(self):
        production = torch.full((4,), _POISON, dtype=torch.bfloat16)
        production[:2] = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        candidate = production.clone()

        self.assertNotEqual(
            torch.tensor(_POISON, dtype=torch.bfloat16).item(), _POISON
        )
        report = _compare_outputs(production, candidate)

        self.assertTrue(report["passed"])
        self.assertEqual(report["production_wrote_elems"], 2)
        self.assertEqual(report["skipped_padding_elems"], 2)
        self.assertEqual(report["missing_writes"], 0)
        self.assertEqual(report["unexpected_writes"], 0)

    def test_candidate_missing_production_write_fails(self):
        production = torch.full((3,), _POISON, dtype=torch.bfloat16)
        production[0] = 3.0
        candidate = production.clone()
        candidate[0] = torch.tensor(_POISON, dtype=candidate.dtype)

        report = _compare_outputs(production, candidate)

        self.assertFalse(report["passed"])
        self.assertEqual(report["missing_writes"], 1)
        self.assertEqual(report["still_poisoned"], 1)
        self.assertEqual(report["unexpected_writes"], 0)

    def test_candidate_write_in_production_padding_fails(self):
        production = torch.full((3,), _POISON, dtype=torch.bfloat16)
        production[0] = 3.0
        candidate = production.clone()
        candidate[1] = 0.0

        report = _compare_outputs(production, candidate)

        self.assertFalse(report["passed"])
        self.assertEqual(report["missing_writes"], 0)
        self.assertEqual(report["unexpected_writes"], 1)

    def test_numeric_difference_fails(self):
        production = torch.tensor([1.0, 2.0], dtype=torch.float32)
        candidate = torch.tensor([1.0, 2.25], dtype=torch.float32)

        report = _compare_outputs(production, candidate)

        self.assertFalse(report["passed"])
        self.assertGreater(report["rel_l2"], 0.0)
        self.assertEqual(report["missing_writes"], 0)
        self.assertEqual(report["unexpected_writes"], 0)

    def test_output_tensor_resolves_from_arg_or_keepalive(self):
        output = torch.empty(4)

        class Wrapper:
            def __init__(self, tensor):
                self.tensor = tensor

        self.assertIs(_find_output_tensor(Wrapper(output), []), output)
        self.assertIs(
            _find_output_tensor(output.data_ptr(), [torch.empty(1), output]),
            output,
        )

    def test_missing_output_tensor_is_explicit(self):
        output = torch.empty(1)
        missing_ptr = output.data_ptr() + output.element_size()

        with self.assertRaisesRegex(
            IsaRunnerError, "cannot find a live torch output tensor"
        ):
            _find_output_tensor(missing_ptr, [output])


if __name__ == "__main__":
    unittest.main()
