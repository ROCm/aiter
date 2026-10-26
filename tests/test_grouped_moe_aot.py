# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only checks for grouped-MoE AOT coverage and launcher ABI wiring."""

import ast
import importlib
import inspect
import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import mock_open, patch

import torch

from aiter.aot.flydsl import grouped_moe as driver


class GroupedMoeAotTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, FLYDSL_GPU_ARCH="gfx1250"))
        self.calls = []

    def compile_aux(self, job):
        """Bind calls to real launcher signatures without compiling or launching."""
        builders = {
            "moe_contiguous_psum": [
                "build_moe_contiguous_psum_remap_module",
                "build_moe_contiguous_psum_remap_ep_module",
            ],
            "moe_apreshuffled_quant_scatter": [
                "build_moe_apre_quant_module",
                "build_moe_apre_scatter_module",
            ],
            "moe_fused_route_quant_scatter": [
                "build_moe_fused_quant_preshuffle_module",
                "build_moe_fused_quant_preshuffle_route_ksplit_module",
                "build_moe_fused_route_quant_scatter_module",
                "build_moe_fused_route_quant_scatter_st_ksplit_module",
            ],
            "moe_gather_reduce": ["build_moe_gather_reduce_module"],
            "moe_route_maps": ["build_moe_topids_to_rows_module"],
        }

        def checked_builder(name, build):
            def make_launcher(*args, **kwargs):
                config = inspect.signature(build).bind(*args, **kwargs)
                config.apply_defaults()
                signature = inspect.signature(build(*args, **kwargs).func)

                def launch(*launch_args, **launch_kwargs):
                    bound = signature.bind(*launch_args, **launch_kwargs)
                    self.calls.append((name, config.arguments, bound.arguments))

                return launch

            return make_launcher

        with ExitStack() as stack:
            for module_name, names in builders.items():
                module = importlib.import_module(
                    f"aiter.ops.flydsl.kernels.{module_name}"
                )
                for name in names:
                    stack.enter_context(
                        patch.object(
                            module, name, checked_builder(name, getattr(module, name))
                        )
                    )
            driver._compile_grouped_moe_aux_kernels(
                job,
                dtype=torch.bfloat16,
                quant_mode="fp4" if job["data_format"] == "fp4" else "fp8",
                wmma_rep=job["tile_m"] // job["m_warp"] // 16,
                contiguous=job["grouped_contiguous_m"],
            )

    def job(self, **overrides):
        return {
            "experts": 64,
            "topk": 8,
            "model_dim": 7168,
            "inter_dim": 2048,
            "max_m": 8192,
            "tile_m": 256,
            "m_warp": 2,
            "tile_m2": 128,
            "m_warp2": 2,
            "token_num": 8192,
            "out_dtype": "bf16",
            "split_k2": 1,
            "data_format": "fp4",
            "grouped_contiguous_m": True,
            "a_preshuffle": True,
            **overrides,
        }

    def test_csv_flag_participates_in_job_identity(self):
        header = "model_dim,inter_dim,expert,token,topk,tile_m,m_warp,tile_m2,m_warp2,a_preshuffle\n"
        shape = "7168,2048,64,8192,8,256,2,128,2,"
        content = header + shape + "0\n" + shape + "1\n" + shape + "1\n"
        with patch("builtins.open", mock_open(read_data=content)):
            jobs = driver.parse_csv("unused.csv")
        self.assertEqual(
            len(jobs), 4
        )  # Two layouts per distinct flag; duplicate removed.
        self.assertEqual({job["a_preshuffle"] for job in jobs}, {False, True})
        self.assertTrue(
            all(job["tile_m2"] == 128 and job["m_warp2"] == 2 for job in jobs)
        )

    def test_missing_csv_fields_keep_legacy_defaults(self):
        content = "model_dim,inter_dim,expert,token,topk,tile_m,m_warp\n7168,2048,64,8192,8,256,2\n"
        with patch("builtins.open", mock_open(read_data=content)):
            jobs = driver.parse_csv("unused.csv")
        for job in jobs:
            self.assertFalse(job["a_preshuffle"])
            self.assertEqual(job["tile_m2"], job["tile_m"])
            self.assertEqual(job["m_warp2"], job["m_warp"])

    def test_apre_variants_and_remap_abi(self):
        for topk in (8, -1):
            with self.subTest(topk=topk):
                self.calls.clear()
                self.compile_aux(self.job(topk=topk))
                names = [name for name, _, _ in self.calls]
                self.assertIn("build_moe_apre_quant_module", names)
                scatters = [
                    config
                    for name, config, _ in self.calls
                    if name == "build_moe_apre_scatter_module"
                ]
                self.assertEqual(
                    {config["src_scale_stride_bytes"] for config in scatters},
                    {224, 256},
                )
                for name, config, args in self.calls:
                    if "psum_remap" in name:
                        self.assertIn("row_to_token", args)
                        self.assertEqual(args["topk"], max(1, topk))
                    if name in (
                        "build_moe_fused_quant_preshuffle_module",
                        "build_moe_fused_quant_preshuffle_route_ksplit_module",
                    ):
                        self.assertTrue(config["payload_preshuffled"])
                        self.assertEqual(config["wmma_rep"], 4)
                        self.assertEqual(config["feat_dim"], 2048)
                self.assertIn("build_moe_fused_quant_preshuffle_module", names)
                self.assertIn("build_moe_contiguous_psum_remap_ep_module", names)

    def test_disabled_and_fp8_do_not_build_apre(self):
        for overrides in ({"a_preshuffle": False}, {"data_format": "a8w4"}):
            with self.subTest(overrides=overrides):
                self.calls.clear()
                self.compile_aux(self.job(**overrides))
                self.assertFalse(
                    any("build_moe_apre_" in name for name, _, _ in self.calls)
                )
                for _, config, _ in self.calls:
                    self.assertFalse(config.get("payload_preshuffled", False))

    def test_default_csv_launcher_signatures(self):
        for job in driver.parse_csv(driver.DEFAULT_CSVS[0]):
            with self.subTest(
                token=job["token_num"], topk=job["topk"], apre=job["a_preshuffle"]
            ):
                self.compile_aux(job)

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
