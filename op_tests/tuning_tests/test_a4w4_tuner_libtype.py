# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 1: --libtype of the a4w4_blockscale tuner (argument parsing and candidate sets).

No GPU is needed: the chip, the CU count and mp_tuner are stubbed, so tune() only
builds its task list, and the ASM kernel list is a temp CSV.

Run:
    python3 -m unittest op_tests.tuning_tests.test_a4w4_tuner_libtype -v
"""

import argparse
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

_TUNER_DIR = Path(__file__).resolve().parents[2] / "csrc" / "ck_gemm_a4w4_blockscale"
sys.path.insert(0, str(_TUNER_DIR))  # the tuner imports its sibling by bare name

import gemm_a4w4_blockscale_tune as tune_mod

ASM_NAMES = {
    (256, 256, 1, 0): "_ZN5aiter_fake_256x256_sk",
    (128, 128, 0, 0): "_ZN5aiter_fake_128x128",
    (32, 128, 0, 1): "_ZN5aiter_fake_32x128_ps",
}
SHAPE = (1, 512, 4096)  # M < 32: no Triton/FlyDSL candidates, so no optional deps
# Shape with real Triton/FlyDSL candidates: M >= 32, N % 16 == 0, K % 256 == 0.
PRESHUFFLE_SHAPE = (32, 1024, 4096)
# An invalid backend name that no PR is going to add (unlike "opus", see #7979).
NO_SUCH_BACKEND = "no_such_backend"


def _parse(argv):
    tuner = tune_mod.GemmA4W4BlockScaleTuner("t", description="t")
    return tuner, tuner.parser.parse_args(argv)


class TestLibtypeArgs(unittest.TestCase):
    def test_libtype_list(self):
        self.assertEqual(tune_mod.libtype_list("ck,asm"), ["ck", "asm"])
        self.assertEqual(tune_mod.libtype_list("all"), ["all"])
        for bad in (NO_SUCH_BACKEND, "ck,nope", "", "cktile"):
            with self.assertRaises(argparse.ArgumentTypeError):
                tune_mod.libtype_list(bad)

    def test_cktile_error_names_the_missing_dispatch(self):
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "no CK Tile runtime"):
            tune_mod.libtype_list("cktile")

    def test_resolve_default_is_ck_asm(self):
        self.assertEqual(tune_mod.resolve_libtypes(), {"ck", "asm"})
        self.assertEqual(tune_mod.resolve_libtypes(None, None), {"ck", "asm"})

    def test_resolve_libtype(self):
        self.assertEqual(tune_mod.resolve_libtypes(["asm"]), {"asm"})
        self.assertEqual(tune_mod.resolve_libtypes(["ck", "asm"]), {"ck", "asm"})
        self.assertEqual(
            tune_mod.resolve_libtypes(["all"]), set(tune_mod.A4W4_LIBTYPES)
        )
        self.assertEqual(
            tune_mod.resolve_libtypes(["ck", "all"]), set(tune_mod.A4W4_LIBTYPES)
        )

    def test_resolve_backends_alias(self):
        self.assertEqual(tune_mod.resolve_libtypes(None, "ck"), {"ck"})
        self.assertEqual(
            tune_mod.resolve_libtypes(None, "ck,asm,triton,flydsl"),
            {"ck", "asm", "triton", "flydsl"},
        )
        for bad in ("all", NO_SUCH_BACKEND, "ck,", ""):
            with self.assertRaisesRegex(ValueError, "Unknown --backends"):
                tune_mod.resolve_libtypes(None, bad)

    def test_libtype_and_backends_conflict(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            tune_mod.resolve_libtypes(["ck"], "ck")

    def test_parser_defaults_and_values(self):
        _, args = _parse([])
        self.assertIsNone(args.libtype)
        self.assertIsNone(args.backends)
        _, args = _parse(["--libtype", "asm,ck"])
        self.assertEqual(args.libtype, ["asm", "ck"])
        _, args = _parse(["--backends", "asm"])
        self.assertEqual(args.backends, "asm")

    def test_parse_args_rejects_conflict_and_bad_values(self):
        for argv in (
            ["--libtype", "ck", "--backends", "ck"],
            ["--backends", NO_SUCH_BACKEND],
            ["--libtype", "cktile"],
        ):
            tuner = tune_mod.GemmA4W4BlockScaleTuner("t", description="t")
            with (
                mock.patch.object(sys, "argv", ["tune.py", *argv]),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as ctx,
            ):
                tuner.parse_args()
            self.assertEqual(ctx.exception.code, 2, argv)


class TunerFixture(unittest.TestCase):
    """Runs tune() without a GPU: stubs the chip, mp_tuner and the ASM kernel list."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        asm_dir = Path(self.tmp.name, "f4gemm")
        asm_dir.mkdir()
        rows = [
            {
                "tile_M": m,
                "tile_N": n,
                "splitK": sk,
                "bpreshuffle": 1,
                "ps": ps,
                "knl_name": name,
            }
            for (m, n, sk, ps), name in ASM_NAMES.items()
        ]
        # A non-preshuffle row must stay excluded.
        rows.append(
            {
                "tile_M": 64,
                "tile_N": 64,
                "splitK": 0,
                "bpreshuffle": 0,
                "ps": 0,
                "knl_name": "_ZN5aiter_fake_nobps",
            }
        )
        pd.DataFrame(rows).to_csv(asm_dir / "f4gemm_bf16_per1x32Fp4.csv", index=False)
        self.asm_dir = self.tmp.name

    def enumerate(self, shape=SHAPE, **opts):
        """[(backend, kernelId, kernelName)] from tune() for `shape`; opts go to args."""
        args = SimpleNamespace(
            splitK=False,
            mp=1,
            shape_grouped=False,
            errRatio=0.05,
            timeout=None,
            verbose=False,
            libtype=None,
            backends=None,
        )
        vars(args).update(opts)
        tuner = tune_mod.GemmA4W4BlockScaleTuner("t", description="t")
        untuned = pd.DataFrame([dict(zip("MNK", shape))])
        captured = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch(
                    "aiter.jit.utils.chip_info.get_gfx_runtime", return_value="gfx950"
                )
            )
            stack.enter_context(mock.patch.object(tuner, "get_cu_num", lambda: 256))
            stack.enter_context(mock.patch.object(tuner, "get_gfx", lambda: "gfx950"))
            stack.enter_context(
                mock.patch.object(tune_mod, "get_asm_dir", lambda: self.asm_dir)
            )
            stack.enter_context(
                mock.patch.object(
                    tune_mod,
                    "mp_tuner",
                    lambda tasks, *a, **kw: captured.extend(tasks) or [],
                )
            )
            tuner.tune(untuned, None, args)
        by_fn = {
            tune_mod.run_gemm_a4w4_blockscale: "ck",
            tune_mod.run_gemm_a4w4_blockscale_asm: "asm",
            tune_mod.run_gemm_preshuffle: "preshuffle",
        }
        out = []
        for task in captured:
            (_shape, kernel_id, _split_k, name), fn = task[0], task[3]
            self.assertEqual(task[0][0], ("gfx950", 256, *shape))
            out.append((by_fn[fn], kernel_id, name))
        return out

    def backends(self, rows):
        return {backend for backend, _, _ in rows}


class TestCandidateSets(TunerFixture):
    """Candidates tune() enumerates for one shape, per libtype."""

    def test_default_is_ck_and_asm(self):
        rows = self.enumerate()
        self.assertEqual(self.backends(rows), {"ck", "asm"})
        self.assertEqual(
            sum(1 for r in rows if r[0] == "ck"), len(tune_mod.kernels_list)
        )
        self.assertEqual(
            sorted(name for b, _, name in rows if b == "asm"),
            sorted(ASM_NAMES.values()),
        )

    def test_default_equals_explicit_ck_asm_and_backends_alias(self):
        default = self.enumerate()
        self.assertEqual(default, self.enumerate(libtype=["ck", "asm"]))
        self.assertEqual(default, self.enumerate(backends="ck,asm"))

    def test_ck_only(self):
        rows = self.enumerate(libtype=["ck"])
        self.assertEqual(self.backends(rows), {"ck"})
        self.assertEqual(
            [r for r in self.enumerate() if r[0] == "ck"],
            rows,
        )

    def test_asm_only_keeps_kernel_ids(self):
        rows = self.enumerate(libtype=["asm"])
        self.assertEqual(self.backends(rows), {"asm"})
        # ids do not shift when CK is not enumerated: they are stable runtime identifiers
        self.assertEqual([r for r in self.enumerate() if r[0] == "asm"], rows)

    def test_ck_only_does_not_read_asm_kernel_list(self):
        with mock.patch.object(
            tune_mod.GemmA4W4BlockScaleTuner, "get_asm_kernels", return_value={}
        ) as get_asm:
            self.enumerate(libtype=["ck"])
        get_asm.assert_not_called()

    def test_preshuffle_backends_gate_candidates(self):
        calls = []

        def fake(M, N, K, backends, split_k):
            calls.append(set(backends))
            return [
                f"triton_preshuffle_fake_{b}" for b in sorted(backends - {"ck", "asm"})
            ]

        with mock.patch.object(tune_mod, "preshuffle_candidates", fake):
            rows = self.enumerate(libtype=["triton"])
            self.assertEqual(calls[-1], {"triton"})
            self.assertEqual(self.backends(rows), {"preshuffle"})
            rows = self.enumerate(libtype=["all"])
            self.assertEqual(calls[-1], set(tune_mod.A4W4_LIBTYPES))
            self.assertEqual(self.backends(rows), {"ck", "asm", "preshuffle"})

    def test_kernel_names_route_in_the_runtime(self):
        """gemm_a4w4 routes a tuned row by kernelName: `_ZN*` to ASM, others to CK."""
        rows = self.enumerate()
        for backend, kernel_id, name in rows:
            if backend == "asm":
                self.assertTrue(name.startswith("_ZN"), name)
            else:
                ck_name = tune_mod.GemmA4W4BlockScaleTuner.getKernelName(
                    None, kernel_id
                )
                self.assertTrue(ck_name and not ck_name.startswith("_ZN"), ck_name)


class TestRealPreshuffleCandidates(TunerFixture):
    """The names `--libtype triton|flydsl` generate must be launchable by the runtime.

    Uses the real `preshuffle_candidates` for an M >= 32 shape and checks every name
    with the parsers `gemm_a4w4` applies to a tuned row (aiter/ops/gemm_op_a4w4.py).
    Nothing is launched, so no GPU kernel runs; Triton and FlyDSL must be importable.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from aiter.ops import gemm_op_a4w4
            from aiter.ops.flydsl.gemm_tune import (
                flydsl_gemm_mxscale_preshuffle_common as fly,
            )
        except ImportError as e:  # optional backends are absent on this host
            raise unittest.SkipTest(f"Triton/FlyDSL not importable: {e}") from e
        cls.runtime = gemm_op_a4w4
        cls.fly = fly

    def candidates(self, backends, split_k=False):
        return tune_mod.preshuffle_candidates(*PRESHUFFLE_SHAPE, backends, split_k)

    def check_triton(self, name, split_k):
        params = self.runtime._parse_triton_a4w4_kernel(name)  # raises if invalid
        if not split_k:
            self.assertEqual(params["NUM_KSPLIT"], 1, name)

    def check_flydsl(self, name):
        params = self.fly.parse_kernel_name(name)
        self.assertIsNotNone(params, name)
        self.assertEqual((params["a_dtype"], params["b_dtype"]), ("fp4", "fp4"), name)
        self.assertFalse(params["fused_split_k"], name)
        inst = self.fly.kernelInstance(**params)
        self.assertTrue(self.fly.instance_valid(inst), name)
        self.assertTrue(
            self.fly.fits_shape(
                inst,
                *PRESHUFFLE_SHAPE,
                a_scale_block="1x32",
                b_scale_block="1x32",
                a_scale_layout="mfma_tile",
                b_scale_layout="mfma_tile",
            ),
            name,
        )

    def test_triton_names_parse_in_the_runtime(self):
        for split_k in (False, True):
            names = self.candidates({"triton"}, split_k)
            self.assertTrue(names)
            for name in names:
                self.assertTrue(name.startswith("triton_preshuffle_"), name)
                self.check_triton(name, split_k)

    def test_flydsl_names_parse_in_the_runtime(self):
        for split_k in (False, True):
            names = self.candidates({"flydsl"}, split_k)
            self.assertTrue(names)
            for name in names:
                self.assertTrue(name.startswith("flydsl_"), name)
                self.check_flydsl(name)

    def test_backends_gate_the_generated_names(self):
        self.assertEqual(self.candidates({"ck", "asm"}), [])
        both = self.candidates({"triton", "flydsl"})
        self.assertEqual(
            sorted(both),
            sorted(self.candidates({"triton"}) + self.candidates({"flydsl"})),
        )

    def test_tune_enumerates_launchable_names_per_libtype(self):
        """End to end through tune(): `--libtype` decides which families appear."""
        for libtype, prefix in (("triton", "triton_"), ("flydsl", "flydsl_")):
            rows = self.enumerate(shape=PRESHUFFLE_SHAPE, libtype=[libtype])
            self.assertEqual({b for b, _, _ in rows}, {"preshuffle"}, libtype)
            self.assertTrue(rows)
            for _, _, name in rows:
                self.assertTrue(name.startswith(prefix), name)
        rows = self.enumerate(shape=PRESHUFFLE_SHAPE, libtype=["all"])
        self.assertEqual({b for b, _, _ in rows}, {"ck", "asm", "preshuffle"})


if __name__ == "__main__":
    unittest.main()
