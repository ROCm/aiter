# SPDX-License-Identifier: MIT

"""CPU regressions for the PyTorch HSTU reference; no AITER kernels required."""

import unittest

import torch
from triton_tests.utils.hstu_attention_ref import pad_sequence, qkv_to_padded_dense


class TestHstuReferencePadding(unittest.TestCase):
    def test_padding_value_and_ragged_sequences(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for lengths in ((2, 0, 1), (0, 0), (3, 3)):
                offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
                q = torch.arange(sum(lengths) * 2, dtype=dtype).reshape(-1, 2)
                for value in (0.0, -3.5, float("nan")):
                    with self.subTest(dtype=dtype, lengths=lengths, value=value):
                        actual = pad_sequence(q, offsets, 3, value)
                        expected = torch.full((len(lengths), 3, 2), value, dtype=dtype)
                        start = 0
                        for batch, length in enumerate(lengths):
                            expected[batch, :length] = q[start : start + length]
                            start += length
                        torch.testing.assert_close(
                            actual, expected, rtol=0, atol=0, equal_nan=True
                        )
                        self.assertEqual(actual.dtype, q.dtype)
                        self.assertEqual(actual.device, q.device)

    def test_existing_qkv_zero_padding(self):
        offsets = torch.tensor([0, 2, 2, 3])
        q = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
        k = q + 1
        v = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
        for actual, original in zip(
            qkv_to_padded_dense(q, k, v, offsets, 3), (q, k, v)
        ):
            expected = torch.zeros(3, 3, *original.shape[1:])
            expected[0, :2] = original[:2]
            expected[2, :1] = original[2:]
            torch.testing.assert_close(actual, expected.transpose(1, 2), rtol=0, atol=0)

    def test_gradients_only_flow_through_input_values(self):
        q = torch.arange(6, dtype=torch.float32).reshape(3, 2).requires_grad_()
        offsets = torch.tensor([0, 2, 2, 3])
        padded = pad_sequence(q, offsets, 3, -3.5)
        padded.sum().backward()
        torch.testing.assert_close(q.grad, torch.ones_like(q), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
