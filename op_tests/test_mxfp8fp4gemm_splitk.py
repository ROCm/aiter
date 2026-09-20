# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 regression tests for automatic Split-K and compiled public GEMMs.

Run with a fresh AITER_JIT_DIR after changing the native launch code:
    python -m unittest op_tests.test_mxfp8fp4gemm_splitk -v
"""

import unittest

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops import gemm_op_a8w4, gemm_op_a8w8
from aiter.ops.mxfp8fp4gemm_common import mxfp8fp4_gemm_splitk
from op_tests.test_mxfp8fp4gemm import _prep


@unittest.skipUnless(
    torch.cuda.is_available() and get_gfx_runtime() == "gfx1250", "requires gfx1250"
)
class TestMxfp8SplitKGPU(unittest.TestCase):
    def test_automatic_splitk_matches_explicit_partials_and_compiled_output(self):
        ops = (
            ("a8w8", gemm_op_a8w8.gemm_a8w8_mxfp8, gemm_op_a8w8._mxfp8_mxfp8_gemm_asm),
            ("a8w4", gemm_op_a8w4.gemm_a8w4_mxfp8, gemm_op_a8w4._mxfp8_mxfp4_gemm_asm),
        )
        # The first shape selects split1; the second selects a deeper split.
        for m, n, k in ((512, 2048, 7168), (128, 1280, 8192)):
            for intype, public, native in ops:
                for apre in (0, 1):
                    with self.subTest(shape=(m, n, k), intype=intype, apre=apre):
                        chosen = mxfp8fp4_gemm_splitk(
                            m, n, k, int(intype == "a8w4"), apre
                        )
                        self.assertEqual(chosen, 1 if m == 512 else 4)
                        inp, partial_ref, _ = _prep(
                            intype,
                            m,
                            n,
                            k,
                            apre,
                            "constant",
                            "constant",
                            torch.Generator(device="cuda").manual_seed(0),
                            reference_splitk=chosen,
                        )
                        args = tuple(inp[key] for key in ("A", "B", "sA", "sB"))
                        explicit = torch.empty_like(partial_ref, dtype=torch.bfloat16)
                        native(*args, explicit, None, apre, chosen)
                        torch.testing.assert_close(
                            explicit, partial_ref.to(torch.bfloat16), rtol=0, atol=0
                        )
                        for automatic in (0, -1):
                            actual = torch.empty_like(explicit)
                            native(*args, actual, None, apre, automatic)
                            torch.testing.assert_close(actual, explicit, rtol=0, atol=0)

                        expected = (
                            explicit.sum(0, dtype=torch.bfloat16)
                            if chosen > 1
                            else explicit
                        )
                        eager = public(*args, a_preshuffle=bool(apre))
                        torch.testing.assert_close(eager, expected, rtol=0, atol=0)
                        compiled = torch.compile(public, fullgraph=True)
                        actual = compiled(*args, a_preshuffle=bool(apre))
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        # Reuse the compiled graph, including the automatically
                        # sized partial buffer and reduction inside the public op.
                        torch.testing.assert_close(
                            compiled(*args, a_preshuffle=bool(apre)),
                            expected,
                            rtol=0,
                            atol=0,
                        )
                        torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
