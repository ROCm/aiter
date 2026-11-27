# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 1: shape legality of the gfx950 a4w4 CK and ASM candidates, and
``--run_config <tuned_csv>`` without the default untuned file (no GPU needed).

The a4w4_blockscale tuner and ``gemm_a4w4`` share the predicates tested here, so the
tuner never launches a candidate that run time would refuse. Narrow N (N=96, K=5120
is Qwen3.8-27B in_proj_ba) is the motivating shape: a tile wider than N reads past B.

Run:
    python3 -m unittest op_tests.tuning_tests.test_a4w4_shape_legality -v
"""

import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ASM_CSV = REPO / "hsa" / "gfx950" / "f4gemm" / "f4gemm_bf16_per1x32Fp4.csv"
TUNED_CSVS = sorted(
    p
    for d in (REPO / "aiter" / "configs", REPO / "aiter" / "configs" / "model_configs")
    for p in d.glob("*a4w4_blockscale*tuned_gemm.csv")
    if "untuned" not in p.name
)
sys.path.insert(0, str(REPO / "csrc" / "ck_gemm_a4w4_blockscale"))

from aiter.ops import gemm_op_a4w4 as op

# Measured on gfx950 (all 35 bpreshuffle kernels at N = 32..1536, K = 1024 and 5120): only these
# two give wrong results (relative L2 error 0.3-0.9) unless N % tile_N == 0. The split-K 256x256 and
# 128x512 kernels and the persistent 256x256 kernel stay at the bf16 split-K noise floor (4e-3).
FULL_N_TILE = {(192, 256, 0, 0), (224, 256, 0, 0)}


class TestPredicates(unittest.TestCase):
    def test_ck_narrow_n_is_illegal(self):
        # every CK instance has NPerBlock >= 128, so N=96 has no legal CK instance
        for n_per_block in (128, 256, 384, 512):
            self.assertFalse(op.a4w4_ck_shape_legal(96, 5120, n_per_block))
        for n in (16, 32, 48, 64, 96, 112):
            self.assertFalse(op.a4w4_ck_shape_legal(n, 5120, 128))

    def test_ck_wide_n_is_legal(self):
        self.assertTrue(op.a4w4_ck_shape_legal(128, 5120, 128))
        self.assertTrue(op.a4w4_ck_shape_legal(7168, 2048, 384))  # partial last tile
        self.assertTrue(op.a4w4_ck_shape_legal(96, 5120, 64))

    def test_ck_alignment(self):
        self.assertFalse(op.a4w4_ck_shape_legal(1000, 5120, 128))  # N % 16
        self.assertFalse(op.a4w4_ck_shape_legal(4096, 2176, 128))  # K % 256
        # KBatch = 2 ** log2_k_split must divide K / 256
        self.assertTrue(op.a4w4_ck_shape_legal(4096, 5120, 128, 2))
        self.assertFalse(op.a4w4_ck_shape_legal(4096, 5120, 128, 3))

    def test_asm_narrow_n(self):
        # kernels without the flag handle N < tile_N (committed N=96 rows use 32x128)
        self.assertTrue(op.a4w4_asm_shape_legal(96, 5120, 128, False))
        self.assertTrue(op.a4w4_asm_shape_legal(96, 5120, 256, False))
        # full_n_tile kernels need whole N tiles
        self.assertFalse(op.a4w4_asm_shape_legal(96, 5120, 256, True))
        self.assertFalse(op.a4w4_asm_shape_legal(800, 5120, 512, True))
        self.assertTrue(op.a4w4_asm_shape_legal(1024, 5120, 512, True))
        self.assertTrue(op.a4w4_asm_shape_legal(256, 5120, 256, True))

    def test_asm_alignment(self):
        self.assertFalse(op.a4w4_asm_shape_legal(100, 5120, 128, False))  # N % 16
        self.assertFalse(op.a4w4_asm_shape_legal(96, 5152, 128, False))  # K % 64
        self.assertTrue(op.a4w4_asm_shape_legal(96, 2176, 128, False))  # K = 17 * 128

    def test_other_names_pass(self):
        for name in (
            "",
            "0",
            "triton_preshuffle_bm32_bn64_bk256_ks1_w4_wpe2",
            "flydsl_x",
        ):
            self.assertTrue(op.a4w4_kernel_shape_legal(name, 96, 5120), name)

    def test_name_dispatch(self):
        ck = "a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3"
        self.assertFalse(op.a4w4_kernel_shape_legal(ck, 96, 5120))
        self.assertTrue(op.a4w4_kernel_shape_legal(ck, 128, 5120))
        asm = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_192x256E"
        self.assertFalse(op.a4w4_kernel_shape_legal(asm, 96, 5120, 0, str(ASM_CSV)))
        self.assertTrue(op.a4w4_kernel_shape_legal(asm, 768, 5120, 0, str(ASM_CSV)))
        small = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"
        self.assertTrue(op.a4w4_kernel_shape_legal(small, 96, 5120, 0, str(ASM_CSV)))
        # unknown ASM names are left to the launcher
        self.assertTrue(
            op.a4w4_kernel_shape_legal("_ZN5aiter3fooE", 96, 5120, 0, str(ASM_CSV))
        )


class TestKernelTable(unittest.TestCase):
    def test_full_n_tile_flags(self):
        df = pd.read_csv(ASM_CSV, comment="#")
        flagged = {
            (r.tile_M, r.tile_N, r.splitK, r.ps)
            for r in df.itertuples()
            if r.full_n_tile and r.bpreshuffle
        }
        self.assertEqual(flagged, FULL_N_TILE)
        self.assertEqual(set(df["full_n_tile"]), {0, 1})

    def test_loader(self):
        table = op.f4gemm_asm_kernels(str(ASM_CSV))
        self.assertEqual(
            table["_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_192x256E"], (256, True)
        )
        self.assertEqual(
            table["_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E"],
            (512, False),
        )
        self.assertEqual(
            table["_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"], (128, False)
        )

    def test_loader_without_flag_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "k.csv")
            pd.DataFrame(
                [{"tile_M": 32, "tile_N": 128, "splitK": 0, "knl_name": "_ZN5aiter1aE"}]
            ).to_csv(path, index=False)
            self.assertEqual(
                op.f4gemm_asm_kernels(path), {"_ZN5aiter1aE": (128, False)}
            )

    def test_committed_rows_are_legal(self):
        """Every committed CK and ASM row stays selectable."""
        illegal = []
        for path in TUNED_CSVS:
            df = pd.read_csv(path)
            if "kernelName" not in df.columns:
                continue
            for r in df.itertuples():
                split = 0 if pd.isna(r.splitK) else int(r.splitK)
                if not op.a4w4_kernel_shape_legal(
                    r.kernelName, r.N, r.K, split, str(ASM_CSV)
                ):
                    illegal.append((path.name, r.M, r.N, r.K, r.kernelName[-8:]))
        self.assertEqual(illegal, [])


class TestTunerCandidates(unittest.TestCase):
    """Candidates tune() builds for one shape; mp_tuner and the chip are stubbed."""

    def setUp(self):
        import gemm_a4w4_blockscale_tune as tune_mod

        self.tune_mod = tune_mod
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        (Path(self.tmp.name) / "f4gemm").mkdir()
        shutil_copy = (Path(self.tmp.name) / "f4gemm" / ASM_CSV.name).write_text
        shutil_copy(ASM_CSV.read_text())

    def candidates(self, n, k=5120, m=32, backends="ck,asm", split_k=False):
        tuner = self.tune_mod.GemmA4W4BlockScaleTuner("t", description="t")
        args = SimpleNamespace(
            splitK=split_k,
            mp=1,
            shape_grouped=False,
            errRatio=0.05,
            timeout=None,
            verbose=False,
            backends=backends,
            libtype=None,
        )
        untuned = pd.DataFrame([{"M": m, "N": n, "K": k}])
        seen = {}

        def fake_mp_tuner(task, *a, **kw):
            seen["task"] = task
            return []

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch(
                    "aiter.jit.utils.chip_info.get_gfx_runtime", return_value="gfx950"
                )
            )
            stack.enter_context(mock.patch.object(tuner, "get_cu_num", lambda: 256))
            stack.enter_context(mock.patch.object(tuner, "get_gfx", lambda: "gfx950"))
            stack.enter_context(
                mock.patch.object(self.tune_mod, "get_asm_dir", lambda: self.tmp.name)
            )
            stack.enter_context(
                mock.patch.object(self.tune_mod, "mp_tuner", fake_mp_tuner)
            )
            # the real one reads the CU count from the GPU; it returns 3 on gfx950
            stack.enter_context(
                mock.patch.object(
                    self.tune_mod.aiter, "compute_gemm_SplitK", lambda *a: 3
                )
            )
            tuner.tune(untuned, pd.DataFrame(), args)
        ck, asm = set(), set()
        for t in seen.get("task", []):
            (info,) = (t[0],)
            kernel_id, split, name = info[1], info[2], info[3]
            (asm if name else ck).add((name or kernel_id, split))
        return ck, asm

    def test_narrow_n_skips_ck_and_full_tile_asm(self):
        ck, asm = self.candidates(96)
        self.assertEqual(ck, set())  # every CK instance has NPerBlock >= 128
        names = {n for n, _ in asm}
        self.assertIn("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", names)
        for tile in ("192x256", "224x256"):
            self.assertFalse([n for n in names if n.endswith(f"_{tile}E")], tile)
        # the split-K 256x256 and 128x512 kernels and the persistent 256x256 kernel
        # handle a partial last tile (measured), so they stay candidates
        for tile in ("256x256", "128x512", "256x256_ps"):
            self.assertTrue([n for n in names if n.endswith(f"_{tile}E")], tile)

    def test_wide_n_keeps_every_candidate(self):
        from gemm_a4w4_blockscale_common import kernels_list

        ck, asm = self.candidates(4096, backends="ck,asm")
        self.assertEqual(len(ck), len(kernels_list))
        asm_rows = pd.read_csv(ASM_CSV, comment="#")
        self.assertEqual(len(asm), int((asm_rows["bpreshuffle"] == 1).sum()))

    def test_split_k_needs_a_divisible_k_for_ck(self):
        ck, _ = self.candidates(4096, k=5120, backends="ck", split_k=True)
        self.assertEqual({s for _, s in ck}, {0, 1, 2})  # K % (256 << 3) != 0

    def test_unaligned_n_has_no_candidates(self):
        ck, asm = self.candidates(100)
        self.assertEqual((ck, asm), (set(), set()))


class TestRunConfigWithoutUntunedFile(unittest.TestCase):
    """--run_config <tuned_csv> must not need the default untuned CSV."""

    def make_tuner(self):
        from aiter.utility.base_tuner import GemmCommonTuner

        calls = {"pre_process": 0, "run_config": 0}

        class Stub(GemmCommonTuner):
            ARG_DEFAULTS: ClassVar[dict] = {
                **GemmCommonTuner.ARG_DEFAULTS,
                "untune_file": "no/such/untuned.csv",
                "tune_file": "no/such/tuned_out.csv",
            }

            def _setup_specific_arguments(self):
                pass

            def pre_process(self, args):
                calls["pre_process"] += 1
                return super().pre_process(args)

            def tune(self, *a, **kw):
                return []

            def getKernelName(self, kid):
                return f"k{kid}"

            def run_config(self, args):
                calls["run_config"] += 1
                return [
                    {"shape": f"({r.M}, {r.N}, {r.K})", "e2e_us": 1.0, "status": "ok"}
                    for r in self.untunedf.itertuples()
                ]

        tuner = Stub("stub")
        tuner.get_cu_num = lambda: 256
        tuner.get_gfx = lambda: "gfx950"
        return tuner, calls

    def tuned_csv(self, tmp):
        path = os.path.join(tmp, "tuned.csv")
        pd.DataFrame(
            [
                {
                    "gfx": "gfx950",
                    "cu_num": 256,
                    "M": 32,
                    "N": 96,
                    "K": 5120,
                    "kernelId": 0,
                    "splitK": 0,
                    "us": 5.0,
                    "kernelName": "k0",
                    "tflops": 1.0,
                    "bw": 1.0,
                    "errRatio": 0.0,
                }
            ]
        ).to_csv(path, index=False)
        return path

    def parse(self, tuner, *argv):
        return tuner.parser.parse_args(list(argv))

    def test_run_config_file_without_untuned_file(self):
        tuner, calls = self.make_tuner()
        with tempfile.TemporaryDirectory() as tmp:
            args = self.parse(tuner, "--run_config", self.tuned_csv(tmp))
            self.assertFalse(os.path.exists(args.untune_file))
            with mock.patch.object(tuner, "_set_config_env_for_run_config") as env:
                env.return_value = (None, 0)
                tuner.run(args)
        self.assertEqual(calls["run_config"], 1)
        self.assertEqual(calls["pre_process"], 0)

    def test_compare_still_needs_the_untuned_file(self):
        tuner, calls = self.make_tuner()
        with tempfile.TemporaryDirectory() as tmp:
            args = self.parse(tuner, "--run_config", self.tuned_csv(tmp), "--compare")
            with self.assertRaisesRegex(AssertionError, "Not exist untuned file"):
                tuner.run(args)
        self.assertEqual(calls["pre_process"], 1)

    def test_run_config_without_file_still_needs_the_untuned_file(self):
        tuner, calls = self.make_tuner()
        args = self.parse(tuner, "--run_config")
        with self.assertRaisesRegex(AssertionError, "Not exist untuned file"):
            tuner.run(args)
        self.assertEqual(calls["pre_process"], 1)

    def test_existing_untuned_file_still_goes_through_pre_process(self):
        tuner, calls = self.make_tuner()
        with tempfile.TemporaryDirectory() as tmp:
            untuned = os.path.join(tmp, "untuned.csv")
            pd.DataFrame([{"M": 1, "N": 96, "K": 5120}]).to_csv(untuned, index=False)
            tuned = self.tuned_csv(tmp)
            args = self.parse(tuner, "--run_config", tuned, "-i", untuned)
            with mock.patch.object(tuner, "_set_config_env_for_run_config") as env:
                env.return_value = (None, 0)
                tuner.run(args)
        self.assertEqual(calls["pre_process"], 1)
        self.assertEqual(calls["run_config"], 1)


if __name__ == "__main__":
    unittest.main()
