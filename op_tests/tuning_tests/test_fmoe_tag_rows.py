# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which tuned fmoe rows survive a write and which count as tuned, by _tag.

Rows tuned with --swiglu_limit carry _tag "swiglu_limit=<limit>". A run with a
limit only replaces and skips rows with its own tag and keeps every other row
as it is. A run without one drops the tagged rows and the _tag column, and
skips any shape already in the file. This pins the current policy; it does not
say it is the right one. CPU only: nothing is timed.
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

KEYS = [
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
]
RESULTS = [
    "block_m",
    "ksplit",
    "us1",
    "kernelName1",
    "err1",
    "us2",
    "kernelName2",
    "err2",
    "us",
    "run_1stage",
    "xbf16",
    "flat",
    "tflops",
    "bw",
    "nt",
]
LIMIT5 = "swiglu_limit=5"
LIMIT7 = "swiglu_limit=7"


def shape(token):
    # bf16 so the fp8 flavor of the running GPU does not rewrite the keys.
    return {
        "gfx": "gfx950",
        "cu_num": 256,
        "token": token,
        "model_dim": 7168,
        "inter_dim": 256,
        "expert": 8,
        "topk": 2,
        "act_type": "ActivationType.Silu",
        "dtype": "torch.bfloat16",
        "q_dtype_a": "torch.bfloat16",
        "q_dtype_w": "torch.bfloat16",
        "q_type": "QuantType.No",
        "use_g1u1": 1,
        "doweight_stage1": 0,
    }


def row(token, kernel, tag, n):
    """A tuned row whose result columns all differ from the other rows'."""
    out = {
        **shape(token),
        "block_m": 16 * (n + 1),
        "ksplit": n,
        "us1": 10.0 + n,
        "kernelName1": f"{kernel}_s1",
        "err1": f"{n}.0%",
        "us2": 5.5 + n,
        "kernelName2": f"{kernel}_s2",
        "err2": f"{n}.5%",
        "us": 15.5 + 2 * n,
        "run_1stage": n % 2,
        "xbf16": (n + 1) % 2,
        "flat": n % 2,
        "tflops": 1.25 + n,
        "bw": 100.5 + n,
        "nt": (n + 1) % 2,
    }
    if tag is not None:
        out["_tag"] = tag
    return out


PLAIN_T1 = row(1, "plain", "", 0)
LIM5_T1 = row(1, "lim5", LIMIT5, 1)
LIM7_T1 = row(1, "lim7", LIMIT7, 2)
LIM5_T2 = row(2, "lim5_t2", LIMIT5, 3)
LIM7_T4 = row(4, "lim7_t4", LIMIT7, 4)
PLAIN_T5 = row(5, "plain_t5", "", 5)
OLD_ROWS = [PLAIN_T1, LIM5_T1, LIM7_T1, LIM5_T2, LIM7_T4, PLAIN_T5]


def normalized(value):
    """Undo the text differences a CSV round trip may add: nan/empty, 1/1.0."""
    text = str(value)
    if text in ("", "nan"):
        return ""
    try:
        return float(text)
    except ValueError:
        return text


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None, "the fmoe tuner needs torch"
)
class TestFmoeTagRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd

        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        cls.pd = pd
        cls.FmoeTuner = gemm_moe_tune.FmoeTuner

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tuned = os.path.join(self.tmp.name, "tuned.csv")
        self.untuned = os.path.join(self.tmp.name, "untuned.csv")
        self.pd.DataFrame(OLD_ROWS).to_csv(self.tuned, index=False)

    def tuner(self, *extra):
        tuner = self.FmoeTuner("fmoeTuner", KEYS, list(RESULTS), "fmoe tuner")
        argv = ["gemm_moe_tune.py", "-i", self.untuned, "-o", self.tuned, *extra]
        with mock.patch.object(sys, "argv", argv):
            args = tuner.parse_args()
        return tuner, args

    def assert_written(self, tuner, expected):
        written = self.pd.read_csv(self.tuned, dtype=str, keep_default_na=False)
        self.assertEqual(list(written.columns), tuner.columns)
        got = [
            {c: normalized(v) for c, v in r.items()} for r in written.to_dict("records")
        ]
        want = [{c: normalized(r.get(c, "")) for c in tuner.columns} for r in expected]
        self.assertEqual(got, want)

    def test_write_with_a_limit_keeps_rows_of_other_tags(self):
        tuner, _ = self.tuner("--swiglu_limit", "7")
        new = row(1, "new", LIMIT7, 6)
        tuner.result_to_csv(self.pd.DataFrame([new], columns=tuner.columns), self.tuned)
        self.assert_written(tuner, [PLAIN_T1, LIM5_T1, LIM5_T2, PLAIN_T5, new, LIM7_T4])

    def test_write_without_a_limit_drops_tagged_rows_and_the_tag(self):
        tuner, _ = self.tuner()
        new = row(1, "new", None, 6)
        tuner.result_to_csv(self.pd.DataFrame([new], columns=tuner.columns), self.tuned)
        self.assertNotIn("_tag", tuner.columns)
        self.assert_written(tuner, [new, PLAIN_T5])

    def pre_process(self, *extra):
        self.pd.DataFrame([shape(t) for t in (1, 2, 3)]).to_csv(
            self.untuned, index=False
        )
        tuner, args = self.tuner(*extra)
        with (
            mock.patch.object(tuner, "get_gfx", return_value="gfx950"),
            mock.patch.object(tuner, "get_cu_num", return_value=256),
        ):
            tuner.pre_process(args)
        return [int(t) for t in tuner.untunedf["token"]]

    def test_with_a_limit_only_rows_of_its_tag_count_as_tuned(self):
        self.assertEqual(self.pre_process("--swiglu_limit", "7"), [2, 3])

    def test_without_a_limit_any_row_counts_as_tuned(self):
        self.assertEqual(self.pre_process(), [3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
