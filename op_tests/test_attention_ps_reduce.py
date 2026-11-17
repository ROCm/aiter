"""CPU: python -m unittest op_tests.test_attention_ps_reduce -v.

GPU: python op_tests/test_attention_ps_reduce.py --gpu -v.

CPU forwarding checks execute only selected functions from the trusted local
attention source with mocked GPU dependencies.
"""

# ruff: noqa: S102

import ast
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "csrc/py_itfs_cu/asm_attention_ps_reduce.cu"
CONFIG = ROOT / "hsa/gfx950/reduce/attention_ps_reduce_asm.csv"


class ReducerDispatchTests(unittest.TestCase):
    def forward_fixture(self):
        source = ast.parse((ROOT / "aiter/ops/attention.py").read_text())
        functions = [
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("pa_persistent_fwd", "_pa_ps_reduce_waves")
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                *functions,
            ],
            type_ignores=[],
        )
        namespace = {
            "torch": mock.Mock(),
            "dtypes": SimpleNamespace(bf16="bf16", fp16="fp16", fp32="fp32", fp8="fp8"),
            "QuantType": SimpleNamespace(per_Token="per_token"),
            "get_gfx_runtime": mock.Mock(return_value="gfx950"),
            "pa_ps_fwd_asm": mock.Mock(),
            "_attention_ps_reduce_asm": mock.Mock(),
            "pa_reduce_v1": mock.Mock(),
        }
        namespace[
            "torch"
        ].cuda.get_device_properties.return_value.multi_processor_count = 256
        exec(
            compile(ast.fix_missing_locations(module), "attention.py", "exec"),
            namespace,
        )

        def tensor(shape, dtype="bf16"):
            result = mock.Mock(shape=shape, dtype=dtype, device="cuda")
            result.numel.return_value = math.prod(shape)
            result.size.side_effect = lambda index: shape[index]
            return result

        arguments = {
            "Q": tensor((4, 16, 128)),
            "K": tensor((64, 1, 8, 16, 16), "fp8"),
            "V": tensor((64, 1, 1, 128, 16), "fp8"),
            "output": tensor((4, 16, 128)),
            "max_qlen": 4,
            "qo_indptr": tensor((2,)),
            "kv_indptr": tensor((2,)),
            "kv_indices": tensor((64,)),
            "context_lens": tensor((1,)),
            "work_indptr": tensor((257,)),
            "work_info": tensor((256, 8)),
            "reduce_indptr": tensor((2,)),
            "reduce_final_map": tensor((1, 2)),
            "reduce_partial_map": tensor((16,)),
            "mask": 1,
        }
        return namespace, arguments

    def test_persistent_forward_preserves_wave_override(self):
        for waves in (0, 1, 2, 4, 8):
            with self.subTest(waves=waves):
                namespace, arguments = self.forward_fixture()
                namespace["pa_persistent_fwd"](**arguments, reduce_waves=waves)
                namespace["pa_ps_fwd_asm"].assert_called_once()
                reducer = namespace["_attention_ps_reduce_asm"]
                reducer.assert_called_once()
                self.assertEqual(reducer.call_args.args[-1], waves)
                self.assertIs(reducer.call_args.args[6], arguments["output"])

    def test_invalid_wave_override_rejected_before_launch(self):
        for waves in (-1, 3, 16):
            with self.subTest(waves=waves):
                namespace, arguments = self.forward_fixture()
                with self.assertRaisesRegex(ValueError, "reduce_waves must be"):
                    namespace["pa_persistent_fwd"](**arguments, reduce_waves=waves)
                namespace["torch"].empty.assert_not_called()
                namespace["pa_ps_fwd_asm"].assert_not_called()

    def test_new_wave_overrides_require_supported_backend(self):
        for waves in (2, 4):
            with self.subTest(waves=waves):
                namespace, arguments = self.forward_fixture()
                namespace["get_gfx_runtime"].return_value = "gfx942"
                with self.assertRaisesRegex(ValueError, "requires the gfx950"):
                    namespace["pa_persistent_fwd"](**arguments, reduce_waves=waves)
                namespace["pa_ps_fwd_asm"].assert_not_called()

    def test_native_wave_selection(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("a C++ compiler is required")
        source = LAUNCHER.read_text()
        function = source[source.index("static int select_ps_reduce_waves(") :]
        function = function[: function.index("\nstruct __attribute__")]
        program = "#include <cstdint>\n#include <cassert>\n" + function + """
int main()
{
    for(int requested : {1, 2, 4, 8})
        for(int rows : {1, 32, 64, 128, 256, 512})
            assert(select_ps_reduce_waves(requested, rows, 256) == requested);
    assert(select_ps_reduce_waves(0, 512, 256) == 1);
    assert(select_ps_reduce_waves(0, 256, 256) == 1);
    assert(select_ps_reduce_waves(0, 255, 256) == 2);
    assert(select_ps_reduce_waves(0, 128, 256) == 2);
    assert(select_ps_reduce_waves(0, 127, 256) == 4);
    assert(select_ps_reduce_waves(0, 64, 256) == 4);
    assert(select_ps_reduce_waves(0, 63, 256) == 8);
    assert(select_ps_reduce_waves(0, 1, 256) == 8);
    assert(select_ps_reduce_waves(0, 76, 304) == 4);
    assert(select_ps_reduce_waves(0, 152, 304) == 2);
    assert(select_ps_reduce_waves(0, 304, 304) == 1);
    assert(select_ps_reduce_waves(0, INT64_C(1) << 40, 256) == 1);
}
"""
        program = "#include <initializer_list>\n" + program
        with tempfile.TemporaryDirectory(prefix="pa-reduce-dispatch-") as directory:
            executable = Path(directory) / "dispatch"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-x",
                    "c++",
                    "-",
                    "-o",
                    str(executable),
                ],
                input=program,
                text=True,
                check=True,
            )
            subprocess.run([str(executable)], check=True)

    def test_registered_wave_variants(self):
        with CONFIG.open(newline="") as stream:
            records = list(csv.DictReader(stream))
        keys = {(record["oType"], int(record["waves"])) for record in records}
        self.assertEqual(
            keys,
            {(dtype, waves) for dtype in ("bf16", "fp16") for waves in (1, 2, 4, 8)},
        )
        self.assertEqual(len(records), len(keys))
        self.assertEqual(
            {record["co_name"] for record in records}, {"pa_p16_d128_reduce_ps.co"}
        )
        for record in records:
            self.assertEqual(int(record["adaptive"]), int(record["waves"]) > 1)
            waves = int(record["waves"])
            stem = (
                f"pa_p16_d128_reduce_ps_{record['oType']}"
                if waves == 1
                else f"pa_p16_d128_{waves}w_reduce_ps_auto_{record['oType']}"
            )
            self.assertEqual(record["knl_name"], f"_ZN5aiter{len(stem)}{stem}E")
            self.assertTrue((CONFIG.parent / record["co_name"]).is_file())

    def test_registered_elf_contract(self):
        try:
            import yaml
        except ModuleNotFoundError:
            self.skipTest("PyYAML is required to parse AMDGPU metadata")
        reader = shutil.which("llvm-readobj")
        if reader is None:
            compiler = shutil.which("amdclang++")
            if compiler is not None:
                candidate = Path(compiler).resolve().parent / "llvm-readobj"
                if candidate.is_file():
                    reader = str(candidate)
        if reader is None:
            self.skipTest("llvm-readobj is required")
        with CONFIG.open(newline="") as stream:
            records = list(csv.DictReader(stream))
        for record in records:
            with self.subTest(dtype=record["oType"], waves=record["waves"]):
                artifact = CONFIG.parent / record["co_name"]
                description = json.loads(
                    subprocess.check_output(
                        [
                            reader,
                            "--elf-output-style=JSON",
                            "--notes",
                            "--symbols",
                            str(artifact),
                        ],
                        text=True,
                    )
                )[0]
                self.assertEqual(description["FileSummary"]["Format"], "elf64-amdgpu")
                symbols = {
                    entry["Symbol"]["Name"]["Name"] for entry in description["Symbols"]
                }
                self.assertIn(record["knl_name"], symbols)
                self.assertIn(record["knl_name"] + ".kd", symbols)
                notes = [
                    note["AMDGPU Metadata"]
                    for section in description["NoteSections"]
                    for note in section["NoteSection"]["Notes"]
                    if "AMDGPU Metadata" in note
                ]
                self.assertEqual(len(notes), 1)
                kernels = yaml.safe_load(notes[0])["amdhsa.kernels"]
                self.assertEqual(len(kernels), len(records))
                by_name = {kernel[".name"]: kernel for kernel in kernels}
                self.assertEqual(set(by_name), {entry["knl_name"] for entry in records})
                kernel = by_name[record["knl_name"]]
                waves = int(record["waves"])
                self.assertEqual(kernel[".name"], record["knl_name"])
                self.assertEqual(kernel[".kernarg_segment_size"], 208)
                self.assertEqual(kernel[".wavefront_size"], 64)
                self.assertEqual(kernel[".max_flat_workgroup_size"], 64 * waves)
                self.assertEqual(
                    kernel[".group_segment_fixed_size"],
                    1024 * waves if waves > 1 else 0,
                )
                self.assertEqual(kernel[".private_segment_fixed_size"], 0)
                self.assertEqual(kernel[".vgpr_count"], 72)
                self.assertEqual(kernel[".sgpr_count"], 84 if waves > 1 else 80)
                self.assertEqual(
                    [argument[".offset"] for argument in kernel[".args"]],
                    list(range(0, 208, 16)),
                )


class ReducerGpuTests(unittest.TestCase):
    enabled = False

    @classmethod
    def setUpClass(cls):
        if not cls.enabled:
            raise unittest.SkipTest("GPU execution requires explicit --gpu")
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("requires a gfx950 GPU")
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        if properties.gcnArchName.split(":")[0] != "gfx950":
            raise unittest.SkipTest("requires a gfx950 GPU")
        from aiter.ops.attention import _attention_ps_reduce_asm

        cls.torch = torch
        cls.reduce = staticmethod(_attention_ps_reduce_asm)

    def test_all_waves_against_reference_and_graph_replay(self):
        torch = self.torch
        split_counts = (0, -1, 1, 2, 4, 8, 9, 16, 17, 32, 33, 64, 65, 129, 257)
        heads, query_length, head_dim = 3, 4, 128
        indptr, partial_map, final_map, groups = [0], [], [], []
        partial_rows = 1
        for tile, count in enumerate(split_counts):
            slots = list(
                range(
                    partial_rows,
                    partial_rows + max(count, 0) * query_length,
                    query_length,
                )
            )
            partial_rows += max(count, 0) * query_length
            if count == -1:
                slots = [-1]
            elif count > 8:
                slots.reverse()
                slots[1] = -2
            partial_map.extend(slots)
            indptr.append(len(partial_map))
            final_map.append((tile * query_length, (tile + 1) * query_length))
            groups.append(slots)
        generator = torch.Generator().manual_seed(7415)
        partial_output = torch.randn(
            (partial_rows, heads, head_dim), generator=generator
        )
        partial_lse = torch.randn((partial_rows, heads), generator=generator) * 5
        partial_output[:, 0] = float("nan")
        partial_lse[:, 0] = -float("inf")
        queries = len(split_counts) * query_length
        expected_output = torch.full(
            (queries, heads, head_dim), 123.0, dtype=torch.float64
        )
        expected_lse = torch.full((queries, heads), 123456.0, dtype=torch.float64)
        for tile, slots in enumerate(groups):
            valid = [slot for slot in slots if slot >= 0]
            if not valid:
                continue
            for query in range(query_length):
                rows = torch.tensor(valid) + query
                values = partial_output[rows].double().nan_to_num()
                lse = partial_lse[rows].double()
                weights = lse.softmax(0).nan_to_num()
                target = tile * query_length + query
                expected_output[target] = (values * weights[..., None]).sum(0)
                expected_lse[target] = lse.logsumexp(0)
        inputs = [
            partial_output.cuda(),
            partial_lse.cuda(),
            torch.tensor(indptr, dtype=torch.int32, device="cuda"),
            torch.tensor(final_map, dtype=torch.int32, device="cuda"),
            torch.tensor(partial_map, dtype=torch.int32, device="cuda"),
        ]
        preserved = [tensor.clone() for tensor in inputs]
        for dtype in (torch.bfloat16, torch.float16):
            reference = expected_output.to(device="cuda", dtype=dtype)
            lse_reference = expected_lse.to(device="cuda", dtype=torch.float32)
            for waves in (0, 1, 2, 4, 8):
                for with_lse in (False, True):
                    with self.subTest(dtype=dtype, waves=waves, with_lse=with_lse):
                        storage = torch.full(
                            (queries + 2, heads, head_dim + 8),
                            123.0,
                            device="cuda",
                            dtype=dtype,
                        )
                        output = storage[1:-1, :, :head_dim]
                        final_lse = (
                            torch.full((queries, heads), 123456.0, device="cuda")
                            if with_lse
                            else None
                        )
                        self.reduce(*inputs, query_length, output, final_lse, waves)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            self.reduce(*inputs, query_length, output, final_lse, waves)
                        for _ in range(2):
                            storage.fill_(123.0)
                            if final_lse is not None:
                                final_lse.fill_(123456.0)
                            graph.replay()
                            torch.cuda.synchronize()
                            torch.testing.assert_close(
                                output,
                                reference,
                                rtol=0.0002,
                                atol=0.008 if dtype == torch.bfloat16 else 0.001,
                            )
                            if final_lse is not None:
                                torch.testing.assert_close(
                                    final_lse, lse_reference, rtol=0, atol=0.0003
                                )
                            self.assertTrue(torch.all(storage[0] == 123.0).item())
                            self.assertTrue(torch.all(storage[-1] == 123.0).item())
                            self.assertTrue(
                                torch.all(storage[:, :, head_dim:] == 123.0).item()
                            )
                        graph.reset()
        for tensor, original in zip(inputs, preserved):
            torch.testing.assert_close(tensor, original, rtol=0, atol=0, equal_nan=True)


if __name__ == "__main__":
    if "--gpu" in sys.argv:
        ReducerGpuTests.enabled = True
        sys.argv.remove("--gpu")
    unittest.main()
