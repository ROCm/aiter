# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch

from aiter.ops.flydsl.mxfp4_kname import (
    _parse_mxfp4_g1_kname,
    parse_flydsl_v2_gemm2_kernel,
)
from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune
from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import Mxfp4FlydslTuner

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


def shape_row(**changes):
    row = {
        "gfx": "gfx950",
        "cu_num": 256,
        "token": 1,
        "model_dim": 1024,
        "inter_dim": 384,
        "expert": 4,
        "topk": 2,
        "act_type": "ActivationType.Silu",
        "dtype": "torch.bfloat16",
        "q_dtype_a": "torch.float4_e2m1fn_x2",
        "q_dtype_w": "torch.float4_e2m1fn_x2",
        "q_type": "QuantType.per_1x32",
        "use_g1u1": True,
        "doweight_stage1": False,
    }
    row.update(changes)
    return row


class TestMxfp4FlydslInputsAndCandidates(unittest.TestCase):
    def setUp(self):
        torch.set_default_device("cpu")
        self.tuner = Mxfp4FlydslTuner.__new__(Mxfp4FlydslTuner)
        self.tuner.keys = KEYS
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(torch.set_default_device, "cuda")

    def read_rows(self, rows):
        path = Path(self.directory.name) / "input.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return self.tuner.get_untuned_gemm_list(str(path))

    def test_mixed_rows_use_their_own_precision_and_default_search(self):
        rows = self.read_rows([shape_row(), shape_row(q_dtype_a="torch.float8_e4m3fn")])
        a4, a8 = [row.to_dict() for _, row in rows.iterrows()]
        a4_default = self.tuner._candidate_rows(a4)
        a8_default = self.tuner._candidate_rows(a8)
        self.assertTrue(a4_default)
        self.assertTrue(a8_default)
        self.assertEqual({c["block_m"] for c in a4_default}, {16})
        self.assertEqual({c["block_m"] for c in a8_default}, {32, 64, 128})
        self.assertLess(
            len(a4_default), len(self.tuner._candidate_rows(a4, full_search=True))
        )
        self.assertEqual(a8_default, self.tuner._candidate_rows(a8, full_search=True))
        for candidate in a8_default:
            g1 = _parse_mxfp4_g1_kname(candidate["kernelName1"])
            g2 = parse_flydsl_v2_gemm2_kernel(candidate["kernelName2"])
            self.assertEqual((g1["a_dtype"], g1["out_dtype"]), ("fp8", "fp8"))
            self.assertEqual(
                (g2["a_dtype"], g2["b_dtype"], g2["out_dtype"]),
                ("fp8", "fp4", "bf16"),
            )
            self.assertEqual(g1["BM"], g2["tile_m"])
            self.assertEqual(g1["BM"], g2["sort_block_m"])
            self.assertFalse(g1["inline_quant"])
            self.assertNotEqual(g1["BN"], 64)
            self.assertEqual(g1["num_waves"], 4)
            self.assertFalse(g2["persist"])

    def test_invalid_csv_reports_the_original_row_and_reason(self):
        invalid = (
            ({"dtype": "torch.float16"}, "dtype"),
            ({"q_dtype_a": "torch.bfloat16"}, "activation dtype"),
            ({"q_dtype_a": "torch.float8_e4m3fnuz"}, "activation dtype"),
            ({"q_dtype_w": "torch.float8_e4m3fn"}, "weight dtype"),
            ({"q_type": "QuantType.per_Token"}, "quantization"),
            ({"act_type": "ActivationType.Gelu"}, "activation"),
            ({"use_g1u1": False}, "use_g1u1"),
            ({"doweight_stage1": True}, "doweight_stage1"),
            ({"token": 0}, "token"),
            ({"model_dim": 1024.5}, "model_dim"),
            ({"inter_dim": -128}, "inter_dim"),
            ({"expert": 0}, "expert"),
            ({"topk": 5}, "topk"),
            ({"gfx": "gfx942"}, "gfx"),
        )
        for changes, reason in invalid:
            # A repeated first shape must not erase the source CSV line.
            with self.subTest(changes=changes), self.assertRaisesRegex(
                ValueError, rf"row 4.*{reason}"
            ):
                self.read_rows([shape_row(), shape_row(), shape_row(**changes)])

    def test_missing_required_column_reports_the_input_file(self):
        row = shape_row()
        del row["q_dtype_a"]
        with self.assertRaisesRegex(ValueError, r"input.csv.*q_dtype_a"):
            self.read_rows([row])

    def test_explicit_search_mode_is_used_in_serial_and_spawned_workers(self):
        a8 = self.read_rows([shape_row(token=64, q_dtype_a="torch.float8_e4m3fn")])
        for mode in (None, "prune", "full"):
            args = SimpleNamespace(mxfp4_search_mode=mode, timeout=0, mp=1)

            # Substitute only the GPU evaluation boundary. Real candidate
            # generation and the shape worker still run in both paths.
            def evaluate(_self, row, candidate, _args):
                candidate["us"] = 128.0 / candidate["block_m"]
                candidate["error"] = 0.0
                return candidate["us"]

            with self.subTest(mode=mode), mock.patch.object(
                Mxfp4FlydslTuner, "_run_candidate", evaluate
            ), mock.patch.object(torch.cuda, "device_count", return_value=2):
                serial = self.tuner.tune(a8, pd.DataFrame(), args)[0]
                args.mp = 2
                # Two distinct workloads exercise the shape worker branch.
                mp_rows = pd.concat([a8, a8.assign(token=65)], ignore_index=True)

                def isolated(payloads, _mp_num, _ctx):
                    with mock.patch.object(torch.cuda, "set_device"):
                        return [
                            gemm_moe_tune._mxfp4_tune_shape_worker((*payload[:3], 0))
                            for payload in payloads
                        ]

                with mock.patch.object(gemm_moe_tune, "_run_shapes_isolated", isolated):
                    spawned = self.tuner.tune(mp_rows, pd.DataFrame(), args)[0]
                self.assertEqual(serial["kernelName1"], spawned["kernelName1"])
                self.assertEqual(serial["kernelName2"], spawned["kernelName2"])
                expected_bm = 64 if mode == "prune" else 128
                self.assertEqual(serial["block_m"], expected_bm)

    def test_cli_preserves_an_omitted_mode_and_explicit_overrides(self):
        tuner = Mxfp4FlydslTuner("test", KEYS, [])
        for mode in (None, "prune", "full"):
            argv = ["gemm_moe_tune.py", "--mxfp4-flydsl", "--mp", "1"]
            if mode is not None:
                argv.extend(["--mxfp4-search-mode", mode])
            with self.subTest(mode=mode), mock.patch.object(sys, "argv", argv):
                args = tuner.parse_args()
                self.assertEqual(args.mxfp4_search_mode, mode)

    def test_full_search_keeps_only_the_coupled_shape_support_domain(self):
        unsupported = (
            {"model_dim": 256},  # GEMM1's pipelined loop needs two K tiles.
            {"model_dim": 768},  # More than one tile, but non-divisible k_wave.
            {"model_dim": 8448},  # B-scale addressing ends after 32 K tiles.
            {"inter_dim": 64},  # GEMM1 alone can emit this; GEMM2 cannot.
        )
        for changes in unsupported:
            row = shape_row(q_dtype_a="torch.float8_e4m3fn", **changes)
            candidates = self.tuner._candidate_rows(row, full_search=True)
            if changes == {"model_dim": 768}:
                self.assertTrue(candidates)
                self.assertEqual(
                    {
                        _parse_mxfp4_g1_kname(c["kernelName1"])["k_wave"]
                        for c in candidates
                    },
                    {1},
                )
            else:
                self.assertFalse(candidates, changes)


if __name__ == "__main__":
    unittest.main()
