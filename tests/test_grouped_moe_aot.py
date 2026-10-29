# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only checks for unified grouped-MoE AOT routing and launcher ABI."""

import ast
import importlib
import inspect
import os
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from aiter.aot.flydsl import moe as driver


class GroupedMoeAotTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, FLYDSL_GPU_ARCH="gfx1250"))

    def csv(self, topk=8, ep_fused=0):
        return (
            "gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,"
            "q_dtype_a,q_dtype_w,q_type,kernelName1,kernelName2,ep_fused\n"
            f"gfx1250,256,8192,7168,2048,64,{topk},ActivationType.Silu,"
            "torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,"
            "QuantType.per_1x32,grouped_gemm1_fp4_gugu,grouped_gemm2_fp4,"
            f"{ep_fused}\n"
        )

    def test_grouped_csv_is_in_unified_defaults(self):
        self.assertIn(
            driver.AITER_CONFIGS.AITER_CONFIG_GROUPED_FMOE_FILE,
            driver.DEFAULT_CSVS,
        )

    def test_grouped_jobs_replay_both_bucket_edges(self):
        with patch("builtins.open", mock_open(read_data=self.csv())):
            jobs = driver.parse_csv("grouped.csv")
        self.assertTrue(jobs)
        self.assertEqual({job["m"] for job in jobs}, {4097, 8192})
        self.assertTrue(all(driver._is_grouped(job) for job in jobs))
        self.assertEqual(
            {job["csv_path"] for job in jobs}, {os.path.abspath("grouped.csv")}
        )

    def test_ep_only_rows_are_explicitly_outside_unified_replay(self):
        # The unified driver cannot infer global experts/real topk from EP rows.
        for topk, ep_fused in ((-1, 0), (8, 1)):
            with (
                self.subTest(topk=topk, ep_fused=ep_fused),
                patch("builtins.open", mock_open(read_data=self.csv(topk, ep_fused))),
            ):
                self.assertEqual(driver.parse_csv("grouped.csv"), [])

    def test_grouped_replay_reads_original_tuning_table(self):
        # A-preshuffle and per-stage tile settings are resolved by runtime,
        # not reconstructed by the retired standalone AOT driver.
        with patch("builtins.open", mock_open(read_data=self.csv())):
            job = driver.parse_csv("grouped.csv")[0]
        with patch.dict(os.environ, AITER_CONFIG_GROUPED_FMOE="original.csv"):
            with driver._runtime_reads(job):
                self.assertEqual(
                    os.environ["AITER_CONFIG_GROUPED_FMOE"], job["csv_path"]
                )
            self.assertEqual(os.environ["AITER_CONFIG_GROUPED_FMOE"], "original.csv")

    def test_runtime_remap_calls_match_launcher_signatures(self):
        module = importlib.import_module("aiter.ops.flydsl.kernels.moe_contiguous_psum")
        source = (
            Path(__file__).resolve().parents[1]
            / "aiter/ops/flydsl/grouped_moe_gfx1250.py"
        )
        tree = ast.parse(source.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "contiguous_psum_remap"
        )
        calls = sorted(
            (
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_run_compiled"
            ),
            key=lambda node: node.lineno,
        )
        self.assertEqual(len(calls), 2)
        for call, builder_name in zip(
            calls,
            (
                "build_moe_contiguous_psum_remap_ep_module",
                "build_moe_contiguous_psum_remap_module",
            ),
        ):
            self.assertFalse(call.keywords)
            signature = inspect.signature(getattr(module, builder_name)().func)
            signature.bind(*[None] * (len(call.args) - 1))

    def test_aux_test_callers_match_remap_signatures(self):
        module = importlib.import_module("aiter.ops.flydsl.kernels.moe_contiguous_psum")
        source = Path(__file__).resolve().parents[1] / "op_tests/test_flydsl_moe_aux.py"
        tree = ast.parse(source.read_text())
        for test_name, builder_name in (
            ("test_psum_remap", "build_moe_contiguous_psum_remap_module"),
            ("test_psum_remap_ep", "build_moe_contiguous_psum_remap_ep_module"),
        ):
            test = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == test_name
            )
            launch = next(
                node
                for node in ast.walk(test)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "launch"
            )
            signature = inspect.signature(getattr(module, builder_name)().func)
            signature.bind(
                *[None] * len(launch.args), **{kw.arg: None for kw in launch.keywords}
            )


if __name__ == "__main__":
    unittest.main()
