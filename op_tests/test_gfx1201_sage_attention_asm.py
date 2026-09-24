# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import unittest

import torch
from aiter.ops.gfx1201.asm_attention import launch_hip_sage_core as launch_asm
from aiter.ops.gfx1201.hip_attention import launch_hip_sage_core as launch_hip
from aiter.ops.gfx1201.prepare import prepare_sage


def bitwise(left, right):
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def fp32_reference(prepared, sequence, rows):
    """Exact exp2-softmax attention on the dequantized kernel inputs (scales already include log2(e)/sqrt(d))."""
    query_int8, query_scale, key_int8, key_scale, value_fp8, value_scale = prepared
    batch, padded, heads, _ = query_int8.shape
    keys = torch.arange(sequence, device="cuda")
    out = torch.empty((batch, len(rows), heads, 128), device="cuda", dtype=torch.float32)
    for b in range(batch):
        for h in range(heads):
            q = query_int8[b, rows, h].float() * query_scale[b, h, rows // 32].unsqueeze(-1)
            k = key_int8[b, :sequence, h].float() * key_scale[b, h, keys // 32].unsqueeze(-1)
            v = value_fp8[b, h, :, :sequence].float().t() * value_scale[b, h].unsqueeze(0)
            s = (q @ k.t()).double()
            p = torch.exp2(s - s.max(dim=-1, keepdim=True).values)
            out[b, :, h] = ((p @ v.double()) / p.sum(dim=-1, keepdim=True)).float()
    return out


class AsmAttentionAnyShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise unittest.SkipTest("ROCm GPU required")

    def run_both(self, batch, sequence, heads, key_amplitude=1.0):
        generator = torch.Generator(device="cuda").manual_seed(sequence * 131 + heads)
        query, key, value = (torch.randn((batch, sequence, heads, 128), device="cuda", generator=generator).bfloat16()
                             for _ in range(3))
        key = (key.float() * key_amplitude).bfloat16()
        prepared = prepare_sage(query, key, value)
        query_int8, query_scale, key_int8, key_scale, value_fp8, value_scale = prepared
        padded = query_int8.shape[1]
        outputs = []
        for launch in (launch_hip, launch_asm):
            output = torch.full((batch, padded, heads, 128), float("nan"), device="cuda", dtype=torch.bfloat16)
            launch(query_int8, key_int8, value_fp8, query_scale, key_scale, value_scale,
                   output, batch, padded, sequence, heads)
            outputs.append(output[:, :sequence])
        self.assertTrue(bool(torch.isfinite(outputs[1]).all()))
        self.assertTrue(bitwise(torch.ops.aiter.gfx1201_sage_attention(query, key, value), outputs[1]))
        return prepared, outputs

    def test_precision_matches_hip(self):
        for batch, sequence, heads in ((1, 1, 28), (1, 33, 28), (2, 96, 3), (1, 513, 28), (3, 2049, 5),
                                       (1, 4097, 28), (2, 8191, 56), (1, 16389, 7), (1, 57330, 28),
                                       (1, 114660, 28), (1, 150001, 28)):
            with self.subTest(batch=batch, sequence=sequence, heads=heads):
                prepared, (hip, asm) = self.run_both(batch, sequence, heads)
                rows = torch.randperm(sequence, device="cuda")[:256].sort().values
                reference = fp32_reference(prepared, sequence, rows)
                error = lambda x: float((x[:, rows].float() - reference).norm() / reference.norm())
                self.assertLessEqual(error(asm), error(hip) * 1.02 + 1e-6)
                self.assertLessEqual(float((asm.float() - hip.float()).abs().mean()), 1e-4)

    def test_large_scale_falls_back_exactly(self):
        # large key scales route every tile through the exact convert path
        for batch, sequence, heads in ((1, 4096, 28), (1, 4097, 7)):
            with self.subTest(batch=batch, sequence=sequence, heads=heads):
                _, (hip, asm) = self.run_both(batch, sequence, heads, key_amplitude=300.0)
                self.assertTrue(bitwise(hip, asm))


if __name__ == "__main__":
    unittest.main()
