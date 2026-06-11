# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the block_cat_fused custom op.

The op packs ``cat([leaky_relu(x[..., :W1], slope), x[..., W1:] * y])``
into one ``torch.ops.aiter.block_cat_fused`` call so AOTAutograd sees one
op (and dispatches the registered fused backward) instead of decomposing
into per-half pointwise backwards.

These tests cover forward / backward correctness vs an eager torch
reference, on both the CPU fp32 fallback path and the GPU bf16 Triton
path.

Run with::

    pytest op_tests/test_block_cat_fused.py
"""

import unittest

import torch
import torch.nn.functional as F

import aiter.ops.block_cat_fused  # noqa: F401  - registers the op

HAS_GPU = torch.cuda.is_available()


def _eager_block_cat(l12, l4, slope, W1):
    lo = F.leaky_relu(l12[..., :W1], slope)
    hi = l12[..., W1:] * l4
    return torch.cat([lo, hi], dim=-1)


class TestBlockCatFusedOp(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_forward_matches_eager_cpu_fp32(self):
        l12 = torch.randn(8, 16, dtype=torch.float32)
        l4 = torch.randn(8, 10, dtype=torch.float32)
        out = torch.ops.aiter.block_cat_fused(l12, l4, 0.01, 6)
        ref = _eager_block_cat(l12, l4, 0.01, 6)
        self.assertTrue(torch.allclose(out, ref, rtol=1e-6, atol=1e-6))

    def test_backward_matches_eager_cpu_fp32(self):
        l12 = torch.randn(8, 16, dtype=torch.float32, requires_grad=True)
        l4 = torch.randn(8, 10, dtype=torch.float32, requires_grad=True)
        out = torch.ops.aiter.block_cat_fused(l12, l4, 0.01, 6)
        out.sum().backward()
        g12_op, g4_op = l12.grad.clone(), l4.grad.clone()
        l12.grad = None
        l4.grad = None
        ref = _eager_block_cat(l12, l4, 0.01, 6)
        ref.sum().backward()
        self.assertTrue(torch.allclose(l12.grad, g12_op, rtol=1e-6, atol=1e-6))
        self.assertTrue(torch.allclose(l4.grad, g4_op, rtol=1e-6, atol=1e-6))

    @unittest.skipIf(not HAS_GPU, "GPU required")
    def test_forward_matches_eager_gpu_bf16(self):
        l12 = torch.randn(64, 96, device="cuda", dtype=torch.bfloat16)
        l4 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        out = torch.ops.aiter.block_cat_fused(l12, l4, 0.01, 32)
        ref = _eager_block_cat(l12, l4, 0.01, 32)
        # bf16 + Triton: small ULP drift is expected
        self.assertTrue(torch.allclose(out, ref, rtol=1e-2, atol=1e-2))

    @unittest.skipIf(not HAS_GPU, "GPU required")
    def test_backward_matches_eager_gpu_bf16(self):
        l12 = torch.randn(
            64, 96, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        l4 = torch.randn(
            64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        out = torch.ops.aiter.block_cat_fused(l12, l4, 0.01, 32)
        out.sum().backward()
        g12_op, g4_op = l12.grad.clone(), l4.grad.clone()
        l12.grad = None
        l4.grad = None
        ref = _eager_block_cat(l12, l4, 0.01, 32)
        ref.sum().backward()
        self.assertTrue(torch.allclose(l12.grad, g12_op, rtol=2e-2, atol=2e-2))
        self.assertTrue(torch.allclose(l4.grad, g4_op, rtol=2e-2, atol=2e-2))

    @unittest.skipIf(not HAS_GPU, "GPU required")
    def test_forward_fp16_stores_in_input_dtype(self):
        # fp16 in must give fp16 out with no silent bf16 downcast. The kernel
        # computes in fp32 and stores in the input dtype, so the reference is
        # the fp32 result rounded to fp16. A bf16 store (8 mantissa bits) would
        # miss this tol; fp16 (~11 bits) clears it.
        l12 = torch.randn(64, 96, device="cuda", dtype=torch.float16)
        l4 = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        out = torch.ops.aiter.block_cat_fused(l12, l4, 0.01, 32)
        ref = _eager_block_cat(l12.float(), l4.float(), 0.01, 32).to(torch.float16)
        self.assertEqual(out.dtype, torch.float16)
        self.assertTrue(torch.allclose(out, ref, rtol=1e-3, atol=1e-3))

    @unittest.skipIf(not HAS_GPU, "GPU required")
    def test_backward_fp16_stores_in_input_dtype(self):
        l12 = torch.randn(
            64, 96, device="cuda", dtype=torch.float16, requires_grad=True
        )
        l4 = torch.randn(64, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        out = torch.ops.aiter.block_cat_fused(l12, l4, 0.01, 32)
        out.sum().backward()
        g12_op, g4_op = l12.grad.clone(), l4.grad.clone()
        l12f = l12.detach().float().requires_grad_(True)
        l4f = l4.detach().float().requires_grad_(True)
        ref = _eager_block_cat(l12f, l4f, 0.01, 32)
        ref.sum().backward()
        self.assertEqual(g12_op.dtype, torch.float16)
        self.assertEqual(g4_op.dtype, torch.float16)
        self.assertTrue(
            torch.allclose(g12_op, l12f.grad.to(torch.float16), rtol=2e-3, atol=2e-3)
        )
        self.assertTrue(
            torch.allclose(g4_op, l4f.grad.to(torch.float16), rtol=2e-3, atol=2e-3)
        )


if __name__ == "__main__":
    unittest.main()
