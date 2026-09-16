# SPDX-License-Identifier: MIT
"""JIT import regressions; no gfx1250 device or kernel launch is required."""

import importlib
import inspect
import itertools
import unittest


class Mxfp4JitImportTest(unittest.TestCase):
    def test_decorated_kernels_and_heuristics(self):
        cases = {
            "aiter.ops.triton._gluon_kernels.gfx1250.quant.fused_mxfp4_quant": {
                "_gluon_fused_rms_mxfp4_quant_kernel": ("ROWS_PER_CTA", False),
                "_gluon_fused_reduce_rms_mxfp4_quant_kernel": ("BLOCK_SIZE_M", False),
            },
            "aiter.ops.triton._triton_kernels.quant.fused_mxfp4_quant": {
                "_fused_rms_mxfp4_quant_kernel": ("BLOCK_SIZE_M", False),
                "_fused_reduce_act_mul_and_dynamic_mxfp4_quant_kernel": (
                    "BLOCK_SIZE_M1",
                    True,
                ),
                "_fused_reduce_rms_mxfp4_quant_kernel": ("BLOCK_SIZE_M", False),
            },
        }
        for module_name, kernels in cases.items():
            module = importlib.import_module(module_name)
            for name, (block_m, iterative) in kernels.items():
                kernel = getattr(module, name)
                # Heuristics -> JITFunction -> original Python function.
                fn = kernel.fn.fn
                with self.subTest(kernel=name):
                    self.assertIn("def " + name, inspect.getsource(fn))
                    self.assertEqual(
                        set(kernel.values),
                        (
                            {"EVEN_M_N"}
                            if iterative
                            or name == "_gluon_fused_rms_mxfp4_quant_kernel"
                            else (
                                {"EVEN_M_N", "EVEN_M_N2"}
                                if name == "_fused_rms_mxfp4_quant_kernel"
                                else {"EVEN_M_N", "EVEN_M_N2", "EVEN_M_N3"}
                            )
                        ),
                    )
                    for dims in itertools.product((32, 33, 48), repeat=4):
                        for num_iter in (1, 3):
                            args = dict(zip(("M", "N1", "N2", "N3"), dims))
                            args.update(
                                BLOCK_SIZE_M=16,
                                BLOCK_SIZE_M1=32,
                                ROWS_PER_CTA=8,
                                BLOCK_SIZE_N=16,
                                BLOCK_SIZE_N1=8,
                                BLOCK_SIZE_N2=32,
                                BLOCK_SIZE_N3=48,
                                NUM_ITER=num_iter,
                            )
                            for key, predicate in kernel.values.items():
                                suffix = key[-1] if key[-1].isdigit() else ""
                                n = "N" + (suffix or "1")
                                divisor = (
                                    args["BLOCK_SIZE_N1"] * num_iter
                                    if iterative
                                    else args["BLOCK_SIZE_N" + suffix]
                                )
                                expected = (
                                    args["M"] % args[block_m] == 0
                                    and args[n] % divisor == 0
                                )
                                self.assertEqual(predicate(args), expected)


if __name__ == "__main__":
    unittest.main()
