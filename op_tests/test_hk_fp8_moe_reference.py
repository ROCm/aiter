# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only oracle regressions: python -m unittest op_tests.test_hk_fp8_moe_reference."""

import unittest

import torch

from op_tests.hk_fp8_moe_utils import activation_roundtrip, reference


class TestQuantizedReference(unittest.TestCase):
    def fixture(self):
        x = torch.zeros(2, 128, dtype=torch.bfloat16)
        x[:, 0] = 1
        w13 = torch.zeros(2, 256, 128)
        w13[0, :, 0] = 2
        # A negative gate stays below -1: only the up projection is clamped below.
        w13[1, :128, 0] = -2
        w13[1, 128:, 0] = 2
        w2 = torch.zeros(2, 128, 128)
        w2[:, :, 0] = 1.125
        w2[:, :, 1] = 1 / 128
        ids = torch.tensor([[0, 0, 1, -1, 2**31 - 1], [1, 0, 1, 1, -1]])
        tw = torch.tensor([[0.2, 0.3, 0.4, 10.0, 20.0], [0.1, 0.2, 0.3, 0.4, 30.0]])
        return x, w13, w2, tw, ids

    def combine(self, expert_values, tw, ids, *, bf16=False):
        # Deliberately accumulate by slot, independently of the oracle's expert loop.
        out = torch.zeros(ids.shape[0], 128)
        for tok in range(ids.shape[0]):
            for slot in range(ids.shape[1]):
                expert = ids[tok, slot].item()
                if 0 <= expert < len(expert_values):
                    out[tok] += expert_values[expert] * tw[tok, slot]
        return out.to(torch.bfloat16).float() if bf16 else out

    def test_round_even_subnormals_and_independent_row_blocks(self):
        x = torch.zeros(3, 256)
        x[0, :8] = torch.tensor(
            [448, 1.0625, 1.1875, -1.0625, -1.1875, 1 / 1024, 3 / 1024, 1 / 512]
        )
        x[0, 128:132] = torch.tensor([896, 2.125, 1 / 512, 3 / 512])
        x[1, :3] = torch.tensor([224, 0.53125, 1 / 1024])
        expected = torch.zeros_like(x)
        expected[0, :8] = torch.tensor([448, 1, 1.25, -1, -1.25, 0, 1 / 256, 1 / 512])
        expected[0, 128:132] = torch.tensor([896, 2, 0, 1 / 128])
        expected[1, :3] = torch.tensor([224, 0.5, 1 / 1024])
        torch.testing.assert_close(activation_roundtrip(x), expected, rtol=0, atol=0)

    def test_default_remains_ideal_fp32(self):
        x, w13, w2, tw, ids = self.fixture()
        scale = 1.125 + 1 / 128
        values = [
            torch.nn.functional.silu(torch.tensor(1.0)) * scale,
            torch.nn.functional.silu(torch.tensor(-2.0)) * scale,
        ]
        expected = self.combine(values, tw, ids)
        torch.testing.assert_close(
            reference(x, w13, w2, tw, ids, 1.0), expected, rtol=1e-6, atol=1e-7
        )

    def test_quantized_clamp_expert_store_and_weighted_combine(self):
        x, w13, w2, tw, ids = self.fixture()
        # silu(1) -> FP8 0.75; silu(-2) -> FP8 -0.234375. The two
        # nonzero down weights give 0.849609375 and -0.2655029296875,
        # which must round to BF16 before routing weights are applied.
        expected = self.combine([0.8515625, -0.265625], tw, ids, bf16=True)
        actual = reference(x, w13, w2, tw, ids, 1.0, quantized=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_input_quantization_precedes_gate_projection(self):
        x = torch.zeros(1, 128, dtype=torch.bfloat16)
        x[0, 0] = 1.0625  # E4M3 tie rounds to 1 before the first GEMM.
        w13 = torch.zeros(1, 256, 128)
        w13[:, :, 0] = 0.5
        w2 = torch.zeros(1, 128, 128)
        w2[:, :, 0] = 1
        ids, tw = torch.tensor([[0]]), torch.ones(1, 1)
        actual = reference(x, w13, w2, tw, ids, 0.0, quantized=True)
        # silu(0.5)*0.5 -> FP8 0.15625. Omitting the input round changes this.
        torch.testing.assert_close(
            actual, torch.full_like(actual, 0.15625), rtol=0, atol=0
        )
        ideal = reference(x, w13, w2, tw, ids, 0.0)
        self.assertGreater(((actual - ideal).norm() / ideal.norm()).item(), 0.1)

    def test_clamp_bypass_and_symmetric_gate_clamp_are_visible(self):
        x, w13, w2, tw, ids = self.fixture()
        for expert in (0, 1):
            routing = torch.full_like(ids, expert)
            clamped = reference(x, w13, w2, tw, routing, 1.0, quantized=True)
            if expert == 0:
                wrong = reference(x, w13, w2, tw, routing, 0.0, quantized=True)
            else:
                symmetric = w13.clone()
                symmetric[1, :128, 0] = -1
                wrong = reference(x, symmetric, w2, tw, routing, 1.0, quantized=True)
            self.assertGreater(((wrong - clamped).norm() / clamped.norm()).item(), 0.12)

    def test_invalid_and_empty_routing_are_zero(self):
        x, w13, w2, tw, ids = self.fixture()
        for empty in (False, True):
            inputs = x[:0] if empty else x
            routing = ids[:0] if empty else torch.full_like(ids, -1)
            weights = tw[:0] if empty else tw
            for quantized in (False, True):
                actual = reference(
                    inputs, w13, w2, weights, routing, 1.0, quantized=quantized
                )
                self.assertEqual(actual.shape, inputs.shape)
                self.assertEqual(torch.count_nonzero(actual).item(), 0)
                self.assertTrue(torch.isfinite(actual).all().item())


if __name__ == "__main__":
    unittest.main()
