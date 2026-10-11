# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import unittest

import torch
from aiter.ops.gfx1201 import qk_norm
from aiter.ops.gfx1201 import rope as _rope
from aiter.ops.gfx1201.norm_rope_prepare import norm_rope_prepare_sage
from aiter.ops.gfx1201.prepare import prepare_sage


def make_inputs(rows, amplitude=1.0, seed=7):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    query, key, value = ((torch.randn((rows, 28, 128), device="cuda", generator=generator) * amplitude).bfloat16()
                         for _ in range(3))
    weights = [(1 + 0.3 * torch.randn(128, device="cuda", generator=generator)).bfloat16() for _ in range(2)]
    angle = torch.rand((rows, 96), device="cuda", generator=generator) * 6.283
    return query, key, value, *weights, angle.cos().contiguous(), angle.sin().contiguous()


def bitwise(left, right):
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


class NormRopeAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise unittest.SkipTest("ROCm GPU required")

    def test_prepare_bitwise(self):
        for rows, amplitude in ((33, 1.0), (1000, 0.0), (1000, 1e-20), (1000, 1e18), (4097, 1.0), (114660, 1.0)):
            query, key, value, query_weight, key_weight, cosine, sine = make_inputs(rows, amplitude)
            rotated_query, rotated_key = _rope.apply(qk_norm(query, query_weight), qk_norm(key, key_weight), cosine, sine)
            expected = prepare_sage(rotated_query.unsqueeze(0), rotated_key.unsqueeze(0), value.unsqueeze(0))
            actual = norm_rope_prepare_sage(query, key, value, query_weight, key_weight, cosine, sine)
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.assertTrue(bitwise(left, right), f"rows={rows} amplitude={amplitude} output={index}")

    def test_attention_bitwise(self):
        for rows in (33, 4097, 114660):
            query, key, value, query_weight, key_weight, cosine, sine = make_inputs(rows)
            rotated_query, rotated_key = _rope.apply(qk_norm(query, query_weight), qk_norm(key, key_weight), cosine, sine)
            expected = torch.ops.aiter.gfx1201_sage_attention(rotated_query.unsqueeze(0), rotated_key.unsqueeze(0),
                                                             value.unsqueeze(0))[0]
            actual = torch.ops.aiter.gfx1201_norm_rope_attention(query, key, value, query_weight, key_weight,
                                                                     cosine, sine)
            self.assertEqual(actual.shape, (rows, 28, 128))
            self.assertTrue(bitwise(expected, actual), f"rows={rows}")


if __name__ == "__main__":
    unittest.main()
