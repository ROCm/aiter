# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU contract checks using the complete tuner and checked-in ASM manifests."""

import importlib
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]


class FlatOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The tuner sets CUDA as the default for its CLI. These checks exercise
        # host task construction and allocations on real CPU Torch tensors.
        with patch.object(torch, "set_default_device", new=lambda device: None):
            cls.module = importlib.import_module(
                "csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune"
            )
        cls.runtime = importlib.import_module("aiter.fused_moe")

    def setUp(self):
        # Other tuner tests in the shared CI process may use a CUDA default.
        device_context = torch.device("cpu")
        device_context.__enter__()
        self.addCleanup(device_context.__exit__, None, None, None)

    def tasks(self, arch):
        mod = self.module
        asm_dir = ROOT / "hsa" / arch
        key = (
            arch,
            80 if arch == "gfx942" else 256,
            1,
            6144,
            256,
            257,
            9,
            mod.ActivationType.Silu,
            torch.bfloat16,
            mod.dtypes.fp8,
            mod.dtypes.fp8,
            mod.QuantType.per_1x128,
            True,
            False,
        )
        tuner = mod.FmoeTuner("flat-contract", [], [])
        with (
            patch.object(mod, "get_gfx", return_value=arch),
            patch.object(mod, "get_asm_dir", return_value=str(asm_dir)),
        ):
            return tuner.gen_1stage_asm_task(key)

    def manifest(self, arch, mode):
        return pd.read_csv(
            ROOT
            / "hsa"
            / arch
            / "fmoe"
            / "silu"
            / f"fmoe_bf16_blockscale{mode}_g1u1_silu.csv"
        )

    def test_gfx942_manifest_flat1_and_flat2_choose_bf16(self):
        tasks = self.tasks("gfx942")
        fp8 = self.manifest("gfx942", "Fp8")
        bf16 = self.module._manifest_flat_by_kernel(self.manifest("gfx942", "Bf16"))
        observed = set()
        for row in fp8.itertuples():
            if row.flat not in (1, 2) or bf16.get(row.knl_name) != row.flat:
                continue
            task = next(t for t in tasks if t[0][2] == row.knl_name)
            observed.add(row.flat)
            self.assertEqual(task[0][1], "asm_1stage_xbf16")
            self.assertEqual(task[4][0][1], "input")
            self.assertEqual(
                task[4][0][4:8], ["topk_ids", "topk_weights", "topk_ids", "topk_ids"]
            )
            self.assertEqual(task[5], {"flat": row.flat})
        self.assertEqual(observed, {1, 2})

    def test_missing_or_mismatched_bf16_manifest_preserves_fp8(self):
        original_read_csv = pd.read_csv
        source = self.manifest("gfx942", "Bf16")
        symbol = source.loc[source["flat"] == 1, "knl_name"].iloc[0]
        for replacement in (None, 2):
            modified = source.copy()
            if replacement is None:
                modified = modified[modified["knl_name"] != symbol]
            else:
                modified.loc[modified["knl_name"] == symbol, "flat"] = replacement

            def read_csv(path, *args, **kwargs):
                if "blockscaleBf16" in str(path):
                    return modified.copy()
                return original_read_csv(path, *args, **kwargs)

            with patch.object(pd, "read_csv", new=read_csv):
                task = next(t for t in self.tasks("gfx942") if t[0][2] == symbol)
            self.assertEqual(task[0][1], "asm_1stage")
            self.assertEqual(task[4][0][1], "a1_qt")

    def test_gfx942_sorted_fp8_selection_is_preserved(self):
        tasks = [t for t in self.tasks("gfx942") if t[0][-1] == 0]
        self.assertTrue(tasks)
        for task in tasks:
            self.assertEqual(task[0][1], "asm_1stage")
            self.assertEqual(task[4][0][1], "a1_qt")
            self.assertEqual(task[4][0][4], "sorted_ids")

    def test_gfx950_both_manifest_input_modes_are_preserved(self):
        tasks = self.tasks("gfx950")
        counts = {"asm_1stage": 0, "asm_1stage_xbf16": 0}
        for task in tasks:
            stage = task[0][1]
            counts[stage] += 1
            self.assertEqual(
                task[4][0][1], "input" if stage.endswith("xbf16") else "a1_qt"
            )
            self.assertEqual(task[5], {"flat": task[0][-1]})
        self.assertTrue(all(counts.values()), counts)

    def test_real_cpu_tail_storage_matches_runtime(self):
        wrapper = self.module.FmoeTuner.run_1stage_fmoe_fp8_blockscale_g1u1
        accepts_flat = "flat" in inspect.signature(wrapper).parameters
        original_zeros = torch.zeros

        def cpu_zeros(*args, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return original_zeros(*args, **kwargs)

        for m in (1, 2, 9, 12, 16):
            for flat in (1, 2):
                ids = torch.zeros((m, 9), dtype=torch.int32, device="cpu")
                weights = torch.ones((m, 9), dtype=torch.float32, device="cpu")
                activation = torch.zeros((m, 6144), dtype=torch.bfloat16, device="cpu")
                inputs = [
                    None,
                    activation,
                    object(),
                    object(),
                    ids,
                    weights,
                    ids,
                    ids,
                    object(),
                    object(),
                    object(),
                ]
                launches = []
                with (
                    patch.object(
                        self.module.aiter,
                        "fmoe_fp8_blockscale_g1u1",
                        new=lambda *a, **k: launches.append((a, k)),
                    ),
                    patch.object(torch, "zeros", new=cpu_zeros),
                ):
                    output = wrapper(
                        *inputs,
                        dtype=torch.bfloat16,
                        **({"flat": flat} if accepts_flat else {}),
                    )
                self.assertEqual(len(launches), 1)
                self.assertIs(launches[0][0][0], output)
                for idx in range(1, 8):
                    self.assertIs(launches[0][0][idx], inputs[idx])
                runtime = self.runtime._moe_prepare_unsorted_input(
                    ids, weights, 6144, torch.bfloat16
                )[-1]
                self.assertEqual(output.shape, (m, 6144))
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertTrue(output.is_contiguous())
                self.assertEqual(torch.count_nonzero(output).item(), 0)
                self.assertEqual(output.untyped_storage().nbytes(), m * 6144 * 2 + 8)
                self.assertEqual(
                    output.untyped_storage().nbytes(),
                    runtime.untyped_storage().nbytes(),
                )
                self.assertEqual(output.data_ptr(), output.untyped_storage().data_ptr())

    def test_sorted_wrapper_keeps_zeroed_unpadded_allocation(self):
        original_zeros = torch.zeros
        requests = []

        def cpu_zeros(*args, **kwargs):
            requests.append(kwargs.get("device"))
            kwargs["device"] = "cpu"
            return original_zeros(*args, **kwargs)

        activation = original_zeros((2, 6144), dtype=torch.bfloat16, device="cpu")
        launches = []
        with (
            patch.object(torch, "zeros", new=cpu_zeros),
            patch.object(
                self.module.aiter,
                "fmoe_fp8_blockscale_g1u1",
                new=lambda *a, **k: launches.append((a, k)),
            ),
        ):
            output = self.module.FmoeTuner.run_1stage_fmoe_fp8_blockscale_g1u1(
                None,
                activation,
                *([None] * 9),
                dtype=torch.bfloat16,
            )
        self.assertEqual(requests, ["cuda"])
        self.assertEqual(output.untyped_storage().nbytes(), 2 * 6144 * 2)
        self.assertEqual(torch.count_nonzero(output).item(), 0)

    def test_task_arguments_reach_native_launch_with_bf16_and_tail(self):
        # Data generation and native execution need a GPU. Use explicit real
        # CPU tensors to check the production task's argument-name binding.
        mod = self.module
        for task in [t for t in self.tasks("gfx942") if t[0][-1] in (1, 2)]:
            names, *args = task[4]
            self.assertEqual(names[1], "input")
            data = {name: torch.zeros((1,), device="cpu") for name in names}
            data["input"] = torch.zeros((1, 6144), dtype=torch.bfloat16, device="cpu")
            data["a1_qt"] = torch.zeros((1, 6144), dtype=mod.dtypes.fp8, device="cpu")
            launches = []
            with patch.object(
                mod.aiter,
                "fmoe_fp8_blockscale_g1u1",
                new=lambda *a, **k: launches.append((a, k)),
            ):
                output = task[3](*[data[n] for n in names], *args, **task[5])
            self.assertIs(launches[0][0][1], data["input"])
            self.assertEqual(output.untyped_storage().nbytes(), 6144 * 2 + 8)
            self.assertIs(task[1], mod.FmoeTuner.generate_data_1stage)
            self.assertIs(task[6], mod.FmoeTuner.torch_moe_blockscale)


if __name__ == "__main__":
    unittest.main()
