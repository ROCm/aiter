# SPDX-License-Identifier: MIT
"""Import/source inspection regressions; no gfx1250 device is required."""

import importlib
import inspect
import itertools
import unittest


class Mxfp4JitImportTest(unittest.TestCase):
    def test_decorated_kernels_and_heuristics(self):
        modules = [
            "aiter.ops.triton.quant.fused_mxfp4_quant",
        ]
        checked = 0
        for name in modules:
            module = importlib.import_module(name)
            for kernel in vars(module).values():
                values = getattr(kernel, "values", None)
                if not isinstance(values, dict) or "EVEN_M_N" not in values:
                    continue
                checked += 1
                # Heuristics wraps a JIT function, which wraps the Python function.
                fn = kernel.fn.fn
                with self.subTest(module=name, kernel=fn.__name__):
                    self.assertIn("def " + fn.__name__, inspect.getsource(fn))
                    for dims in itertools.product((32, 33, 48), repeat=4):
                        args = dict(zip(("M", "N1", "N2", "N3"), dims))
                        args.update(
                            BLOCK_SIZE_M=16,
                            BLOCK_SIZE_M1=32,
                            ROWS_PER_CTA=16,
                            BLOCK_SIZE_N=16,
                            BLOCK_SIZE_N1=8,
                            NUM_ITER=3,
                            BLOCK_SIZE_N2=16,
                            BLOCK_SIZE_N3=16,
                        )
                        for key, predicate in values.items():
                            n = "N" + (key[-1] if key[-1].isdigit() else "1")
                            expected = args["M"] % 16 == 0 and args[n] % 16 == 0
                            if (
                                fn.__name__
                                == "_fused_reduce_act_mul_and_dynamic_mxfp4_quant_kernel"
                            ):
                                expected = args["M"] % 32 == 0 and args["N1"] % 24 == 0
                            self.assertEqual(predicate(args), expected)
        self.assertEqual(checked, 5)


if __name__ == "__main__":
    unittest.main()
