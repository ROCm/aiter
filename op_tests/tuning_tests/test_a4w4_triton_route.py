# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 0: static checks of the a4w4 tuned rows that gemm_a4w4 routes to the
Triton split-K kernel (kernelName "triton_preshuffle_*"). No GPU, no aiter import.

The name pattern and supported archs are read from aiter/ops/gemm_op_a4w4.py with
ast, so the test checks the CSVs against the dispatcher's own definitions.
"""

import ast
import glob
import os
import re
import unittest

import pandas as pd

AITER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIGS_DIR = os.path.join(AITER_ROOT, "aiter", "configs")
GEMM_OP_A4W4 = os.path.join(AITER_ROOT, "aiter", "ops", "gemm_op_a4w4.py")


def _module_constants(path, names):
    """Literal values of module-level assignments, e.g. X = re.compile("...")."""
    with open(path) as f:
        tree = ast.parse(f.read())
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in names:
            continue
        value = node.value
        if isinstance(value, ast.Call):  # re.compile(<literal>)
            value = value.args[0]
        out[target.id] = ast.literal_eval(value)
    return out


def _a4w4_tuned_csvs():
    paths = [os.path.join(CONFIGS_DIR, "a4w4_blockscale_tuned_gemm.csv")]
    paths += sorted(
        glob.glob(
            os.path.join(CONFIGS_DIR, "model_configs", "*a4w4_blockscale_tuned_gemm*")
        )
    )
    return [p for p in paths if os.path.exists(p)]


class TestA4W4TritonRoute(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        consts = _module_constants(
            GEMM_OP_A4W4, {"_TRITON_PRESHUFFLE_RE", "_TRITON_PRESHUFFLE_ARCHS"}
        )
        cls.name_re = re.compile(consts["_TRITON_PRESHUFFLE_RE"])
        cls.archs = consts["_TRITON_PRESHUFFLE_ARCHS"]
        frames = []
        for path in _a4w4_tuned_csvs():
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            df["kernelName"] = df["kernelName"].astype(str).str.strip()
            df["csv_file"] = os.path.basename(path)
            frames.append(df)
        cls.df = pd.concat(frames, ignore_index=True)
        cls.tri = cls.df[cls.df["kernelName"].str.startswith("triton_")]

    def test_names_parse(self):
        bad = [
            f"{r.csv_file}: {r.kernelName}"
            for r in self.tri.itertuples()
            if not self.name_re.match(r.kernelName)
        ]
        self.assertEqual(bad, [], "triton_ names the dispatcher cannot parse")

    def test_rows_meet_kernel_restrictions(self):
        bad = []
        for r in self.tri.itertuples():
            mt = self.name_re.match(r.kernelName)
            if mt is None:
                continue
            bm = int(mt.group(1))
            if r.gfx not in self.archs:
                bad.append(f"{r.csv_file} M={r.M} N={r.N} K={r.K}: arch {r.gfx}")
            # The dispatcher falls back below M=32 and when K % 256 != 0; the
            # kernel needs BLOCK_SIZE_M >= 32 and B viewed as (N // 16, -1).
            if r.M < 32 or r.K % 256 or r.N % 16 or bm < 32:
                bad.append(f"{r.csv_file} M={r.M} N={r.N} K={r.K}: {r.kernelName}")
        self.assertEqual(bad, [], "triton_ rows the dispatcher would not run")

    def test_small_m_does_not_pad_onto_triton_rows(self):
        # M in 17..31 pads to 32. Without an exact row it would land on the
        # Triton M=32 row, which falls back to the untuned default heuristic.
        keys = set(
            zip(self.df.gfx, self.df.cu_num, self.df.M, self.df.N, self.df.K)
        ) - set(zip(self.tri.gfx, self.tri.cu_num, self.tri.M, self.tri.N, self.tri.K))
        missing = []
        for r in self.tri[self.tri["M"] == 32].itertuples():
            for m in range(17, 32):
                if (r.gfx, r.cu_num, m, r.N, r.K) not in keys:
                    missing.append(f"{r.csv_file} M={m} N={r.N} K={r.K}")
        self.assertEqual(missing, [], "no exact non-Triton row below M=32")


if __name__ == "__main__":
    unittest.main()
