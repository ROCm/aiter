# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import unittest

import torch
from aiter.ops.gfx1201.hip_attention import launch_hip_sage_core
from aiter.ops.gfx1201.prepare import prepare_sage


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise unittest.SkipTest("ROCm GPU required")

    def inputs(self, batch=1, sequence=96, heads=28):
        torch.manual_seed(42)
        query = torch.randn((batch, sequence, heads, 128), device="cuda", dtype=torch.bfloat16)
        return query, torch.randn_like(query), torch.randn_like(query)

    def test_dispatch_and_reference(self):
        for batch, sequence, heads in ((1, 32, 28), (1, 96, 28), (2, 1024, 3)):
            query, key, value = self.inputs(batch, sequence, heads)
            actual = torch.ops.aiter.gfx1201_sage_attention(query, key, value)
            query_int8, query_scale, key_int8, key_scale, value_fp8, value_scale = prepare_sage(query, key, value)
            expected = torch.empty_like(query)
            launch_hip_sage_core(query_int8, key_int8, value_fp8, query_scale, key_scale, value_scale,
                                expected, batch, sequence, sequence, heads)
            # ASM core is within 1e-4 of the HIP core (softmax without int->float converts), not bitwise
            self.assertTrue(torch.isfinite(actual).all())
            self.assertLessEqual(float((actual.float() - expected.float()).abs().mean()), 1e-4)
            self.assertNotEqual(actual.data_ptr(), query.data_ptr())

    def test_tails(self):
        for sequence in (1, 4, 15, 16, 17, 31, 32, 33, 63, 64, 65, 97, 511, 513):
            with self.subTest(sequence=sequence):
                query, key, value = self.inputs(batch=2, sequence=sequence, heads=3)
                actual = torch.ops.aiter.gfx1201_sage_attention(query, key, value)
                expected = torch.nn.functional.scaled_dot_product_attention(
                    query.transpose(1, 2).float(), key.transpose(1, 2).float(),
                    value.transpose(1, 2).float()).transpose(1, 2)
                self.assertEqual(actual.shape, query.shape)
                self.assertTrue(actual.is_contiguous())
                self.assertTrue(torch.isfinite(actual).all())
                cosine = torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.flatten(), dim=0)
                self.assertGreater(cosine.item(), 0.998)
                relative_error = (actual.float() - expected).square().mean().sqrt() / expected.square().mean().sqrt()
                self.assertLess(relative_error.item(), 0.06)
                query.zero_()
                key.zero_()
                value.fill_(1)
                actual = torch.ops.aiter.gfx1201_sage_attention(query, key, value)
                torch.testing.assert_close(actual, value, atol=0.008, rtol=0)

    def test_registration(self):
        results = torch.library.opcheck(torch.ops.aiter.gfx1201_sage_attention.default, self.inputs(batch=2, sequence=97, heads=3))
        self.assertTrue(all(result == "SUCCESS" for result in results.values()))

    def test_invalid_inputs(self):
        for sequence in (0,):
            with self.assertRaises(ValueError):
                torch.ops.aiter.gfx1201_sage_attention(*self.inputs(sequence=sequence))
        query, key, value = self.inputs()
        with self.assertRaises(ValueError):
            torch.ops.aiter.gfx1201_sage_attention(query.float(), key.float(), value.float())
        with self.assertRaises(ValueError):
            torch.ops.aiter.gfx1201_sage_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
        with self.assertRaises(ValueError):
            torch.ops.aiter.gfx1201_sage_attention(query.requires_grad_(), key, value)

    def test_nondefault_stream(self):
        inputs = self.inputs(batch=2, sequence=97, heads=3)
        expected = torch.ops.aiter.gfx1201_sage_attention(*inputs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = torch.ops.aiter.gfx1201_sage_attention(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        self.assertTrue(torch.equal(actual, expected))



if __name__ == "__main__":
    unittest.main()
