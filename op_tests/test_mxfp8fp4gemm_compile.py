# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU tests of the public MXFP8 GEMM tracing contract, with native calls mocked.

These exercise real custom-op registration, FakeTensor and fullgraph compilation;
they do not validate the GPU kernels' numerical results.
"""

import unittest
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from aiter.ops import gemm_op_a8w4, gemm_op_a8w8, mxfp8fp4gemm_common

OPS = (
    (gemm_op_a8w8, "gemm_a8w8_mxfp8", "_mxfp8_mxfp8_gemm_asm", 1),
    (gemm_op_a8w4, "gemm_a8w4_mxfp8", "_mxfp8_mxfp4_gemm_asm", 2),
)


def inputs(m, n, b_packing):
    return (
        torch.zeros(m, 128),
        torch.zeros(n, 128 // b_packing),
        torch.zeros(m, 4),
        torch.zeros(n, 4),
    )


class TestMxfp8GemmCompile(unittest.TestCase):
    def test_fake_output_does_not_query_or_launch_native_code(self):
        for module, name, native, packing in OPS:
            for apre in (False, True):
                with (
                    self.subTest(op=name, apre=apre),
                    patch.object(mxfp8fp4gemm_common, "mxfp8fp4_gemm_splitk") as query,
                    patch.object(module, native) as launch,
                    FakeTensorMode(),
                ):
                    out = getattr(module, name)(
                        *inputs(4, 16, packing), a_preshuffle=apre
                    )
                    self.assertEqual(out.shape, (4, 16))
                    self.assertEqual(out.dtype, torch.bfloat16)
                    self.assertEqual(out.device, torch.device("cpu"))
                    query.assert_not_called()
                    launch.assert_not_called()

    def test_fullgraph_keeps_runtime_splitk_and_reduction(self):
        def choose(m, n, k, b_is_fp4, apre, kernel):
            # Two shapes choose different counts, so tracing cannot hardcode one.
            return 4 if m == 4 else 2

        def launch(a, b, sa, sb, out, kernel, apre, splitk):
            if splitk == 1:
                self.assertEqual(out.shape, (a.shape[0], b.shape[0]))
                out.fill_(1)
            else:
                self.assertEqual(out.shape, (splitk, a.shape[0], b.shape[0]))
                for i in range(splitk):
                    out[i].fill_(i + 1)

        for module, name, native, packing in OPS:
            for apre in (False, True):
                for splitk in (0, 1, 4):
                    torch._dynamo.reset()
                    with (
                        self.subTest(op=name, apre=apre, splitk=splitk),
                        patch.object(
                            mxfp8fp4gemm_common,
                            "mxfp8fp4_gemm_splitk",
                            side_effect=choose,
                        ) as query,
                        patch.object(module, native, side_effect=launch) as native_call,
                        patch.object(module, "require_gfx1250_asm"),
                    ):
                        compiled = torch.compile(
                            getattr(module, name),
                            backend="aot_eager",
                            fullgraph=True,
                            dynamic=True,
                        )
                        for m, n in ((4, 16), (8, 32)):
                            out = compiled(
                                *inputs(m, n, packing), a_preshuffle=apre, splitk=splitk
                            )
                            chosen = splitk or (4 if m == 4 else 2)
                            expected = torch.full(
                                (m, n), chosen * (chosen + 1) / 2, dtype=torch.bfloat16
                            )
                            torch.testing.assert_close(out, expected, rtol=0, atol=0)
                        self.assertEqual(native_call.call_count, 2)
                        self.assertEqual(query.call_count, 2 if splitk == 0 else 0)
        torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
