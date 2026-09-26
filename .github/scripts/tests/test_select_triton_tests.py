"""CPU regressions for conservative Triton test selection; no GPU imports."""

import argparse
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "select_triton_tests.py"
SPEC = importlib.util.spec_from_file_location("select_triton_tests", SCRIPT)
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)
SRC = selector.SRC
TESTS = selector.TESTS


class SelectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        patcher = mock.patch.object(selector, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        selector.resolve_module.cache_clear()
        self.addCleanup(selector.resolve_module.cache_clear)
        self.write(SRC + "attention/unrelated.py")
        self.unrelated = self.write(
            TESTS + "attention/test_unrelated.py",
            "from aiter.ops.triton.attention.unrelated import run\n",
        )

    def write(self, path, contents=""):
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents, encoding="utf-8")
        return path

    def select(self, *changed):
        tests, _ = selector.select(list(changed))
        return set(tests)

    def kernel_and_test(self):
        kernel = self.write(SRC + "_triton_kernels/normalization/rmsnorm.py")
        self.write(
            SRC + "normalization/rmsnorm.py",
            "from aiter.ops.triton._triton_kernels.normalization.rmsnorm import run\n",
        )
        direct = self.write(
            TESTS + "normalization/test_rmsnorm.py",
            "from aiter.ops.triton.normalization.rmsnorm import run\n",
        )
        return kernel, direct

    def test_transitive_imports_select_fused_consumers_only(self):
        kernel, direct = self.kernel_and_test()
        self.write(
            SRC + "fusions/fused.py",
            "from aiter.ops.triton.normalization.rmsnorm import run\n",
        )
        fused = self.write(
            TESTS + "fusions/test_fused.py",
            "from aiter.ops.triton.fusions.fused import run\n",
        )
        self.assertEqual(self.select(kernel), {direct, fused})

    def test_graph_hit_avoids_unrelated_tests_in_same_category(self):
        kernel = self.write(SRC + "attention/helper.py")
        fused = self.write(
            TESTS + "fusions/test_consumer.py",
            "from aiter.ops.triton.attention.helper import run\n",
        )
        self.assertEqual(self.select(kernel), {fused})

    def test_relative_imports_follow_modules_and_package_members(self):
        kernel, direct = self.kernel_and_test()
        self.write(SRC + "fusions/__init__.py")
        self.write(
            SRC + "fusions/helper.py",
            "from ..normalization.rmsnorm import run\n",
        )
        self.write(SRC + "fusions/fused.py", "from . import helper\n")
        fused = self.write(
            TESTS + "fusions/test_fused.py",
            "from aiter.ops.triton.fusions.fused import run\n",
        )
        self.assertEqual(self.select(kernel), {direct, fused})

    def test_absolute_from_package_resolves_submodule(self):
        kernel, direct = self.kernel_and_test()
        fused = self.write(
            TESTS + "fusions/test_fused.py",
            "from aiter.ops.triton.normalization import rmsnorm\n",
        )
        self.assertEqual(self.select(kernel), {direct, fused})

    def test_import_cycle_terminates_and_reaches_kernel(self):
        kernel, direct = self.kernel_and_test()
        self.write(
            SRC + "fusions/first.py",
            "from aiter.ops.triton.fusions.second import run\n",
        )
        self.write(
            SRC + "fusions/second.py",
            "from aiter.ops.triton.fusions.first import run\n"
            "from aiter.ops.triton.normalization.rmsnorm import run\n",
        )
        fused = self.write(
            TESTS + "fusions/test_fused.py",
            "from aiter.ops.triton.fusions.first import run\n",
        )
        self.assertEqual(self.select(kernel), {direct, fused})

    def test_named_only_test_inherits_wrapper_dependencies(self):
        kernel, direct = self.kernel_and_test()
        self.write(
            SRC + "fusions/fused.py",
            "from aiter.ops.triton.normalization.rmsnorm import run\n",
        )
        compiled = self.write(TESTS + "torch_compile/test_compile_fused.py")
        self.assertEqual(self.select(kernel), {direct, compiled})

    def test_changed_test_selects_tests_importing_its_helpers(self):
        _, direct = self.kernel_and_test()
        fused = self.write(
            TESTS + "fusions/test_fused.py",
            "from op_tests.triton_tests.normalization.test_rmsnorm import inputs\n",
        )
        self.assertEqual(self.select(direct), {direct, fused})

    def test_deleted_test_runs_remaining_importers_only(self):
        _, direct = self.kernel_and_test()
        fused = self.write(
            TESTS + "fusions/test_fused.py",
            "from op_tests.triton_tests.normalization.test_rmsnorm import inputs\n",
        )
        (self.root / direct).unlink()
        self.assertEqual(self.select(direct), {fused})

    def test_renamed_module_keeps_old_importers_visible(self):
        old = SRC + "normalization/old.py"
        new = self.write(SRC + "fusions/new.py")
        direct = self.write(
            TESTS + "fusions/test_new.py",
            "from aiter.ops.triton.fusions.new import run\n",
        )
        importer = self.write(
            TESTS + "normalization/test_consumer.py",
            "from aiter.ops.triton.normalization.old import run\n"
            "from aiter.ops.triton.attention.unrelated import other\n",
        )
        self.assertEqual(self.select(old, new), {direct, importer})

    def test_dynamic_import_with_static_dependency_remains_conservative(self):
        kernel, direct = self.kernel_and_test()
        dynamic = self.write(
            TESTS + "quant/test_jit_import.py",
            "import importlib\n"
            "from aiter.ops.triton.attention.unrelated import helper\n"
            "def test_dynamic(module_name):\n"
            "    importlib.import_module(module_name)\n",
        )
        self.assertEqual(self.select(kernel), {direct, dynamic})

    def test_dynamic_import_in_helper_selects_reaching_test(self):
        kernel, direct = self.kernel_and_test()
        self.write(
            TESTS + "quant/helper.py",
            "from importlib import import_module as load\n"
            "def get_op(name):\n"
            "    return load(name)\n",
        )
        dynamic = self.write(
            TESTS + "quant/test_dynamic.py",
            "from op_tests.triton_tests.quant.helper import get_op\n"
            "from aiter.ops.triton.attention.unrelated import helper\n",
        )
        self.assertEqual(self.select(kernel), {direct, dynamic})

    def test_unmapped_test_runs_with_relevant_selection(self):
        kernel, direct = self.kernel_and_test()
        unknown = self.write(TESTS + "test_unmapped.py")
        self.assertEqual(self.select(kernel), {direct, unknown})

    def test_documentation_and_benchmarks_do_not_select_tests(self):
        self.assertEqual(
            self.select("README.md", selector.BENCH + "bench_rmsnorm.py"), set()
        )

    def test_config_runs_whole_op_and_consumers_of_other_op_variants(self):
        config = SRC + "configs/gfx950/triton/gemm/first/tuned.json"
        self.write(SRC + "gemm/basic/first.py")
        self.write(SRC + "gemm/basic/second.py")
        first = self.write(
            TESTS + "gemm/basic/test_first.py",
            "from aiter.ops.triton.gemm.basic.first import run\n",
        )
        second = self.write(
            TESTS + "gemm/basic/test_second.py",
            "from aiter.ops.triton.gemm.basic.second import run\n",
        )
        fused = self.write(
            TESTS + "fusions/test_fused.py",
            "from aiter.ops.triton.gemm.basic.second import run\n",
        )
        compiled = self.write(TESTS + "torch_compile/test_compile_second.py")
        self.assertEqual(self.select(config), {first, second, fused, compiled})

    def test_unknown_config_op_raises_even_with_matching_family_stem(self):
        self.kernel_and_test()
        with self.assertRaises(RuntimeError):
            self.select(SRC + "configs/gfx950/triton/unknown/rmsnorm/tuned.json")

    def test_attention_config_includes_chunk_delta_attn_category(self):
        self.write(SRC + "_triton_kernels/chunk_delta_attn/chunk.py")
        chunk = self.write(
            TESTS + "chunk_delta_attn/test_chunk.py",
            "from aiter.ops.triton._triton_kernels.chunk_delta_attn.chunk import run\n",
        )
        config = SRC + "configs/gfx950/triton/attention/mha/tuned.json"
        self.assertEqual(self.select(config), {self.unrelated, chunk})

    def test_mhc_config_runs_fusions_folder_and_external_consumers(self):
        self.write(SRC + "fusions/mhc.py")
        mhc = self.write(
            TESTS + "fusions/test_mhc.py",
            "from aiter.ops.triton.fusions.mhc import run\n",
        )
        self.write(SRC + "fusions/another.py")
        other_fusion = self.write(
            TESTS + "fusions/test_another.py",
            "from aiter.ops.triton.fusions.another import run\n",
        )
        consumer = self.write(
            TESTS + "normalization/test_mhc_consumer.py",
            "from aiter.ops.triton.fusions.mhc import run\n",
        )
        config = SRC + "configs/gfx950/triton/mhc/mhc/tuned.json"
        self.assertEqual(self.select(config), {mhc, other_fusion, consumer})

    def test_unknown_config_layout_raises(self):
        with self.assertRaises(RuntimeError):
            self.select(SRC + "configs/rmsnorm.json")

    def test_package_initializer_uses_full_suite_fallback(self):
        self.kernel_and_test()
        with self.assertRaises(RuntimeError):
            self.select(TESTS + "normalization/__init__.py")

    def test_unmapped_source_raises(self):
        source = self.write(SRC + "new_category/new_op.py")
        with self.assertRaises(RuntimeError):
            self.select(source)

    def test_shared_code_outside_graph_requires_full_suite(self):
        for path in (
            "aiter/ops/shuffle.py",
            "aiter/jit/utils/torch_guard.py",
            "op_tests/test_rope.py",
            "requirements.txt",
        ):
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                self.select(path)

    def test_main_falls_back_to_full_suite_on_selection_failure(self):
        source = self.write(SRC + "new_category/new_op.py")
        args = argparse.Namespace(all=False, output=str(self.root / "selected.list"))
        with (
            mock.patch.object(selector, "parse_args", return_value=args),
            mock.patch.object(selector, "changed_files", return_value=[source]),
            mock.patch.object(selector, "write_outputs") as write_outputs,
        ):
            selector.main()
        tests, _, is_full, output = write_outputs.call_args.args
        self.assertTrue(is_full)
        self.assertEqual(tests, [self.unrelated])
        self.assertEqual(output, args.output)


class ChangedFilesTests(unittest.TestCase):
    def test_diff_disables_renames_and_uses_nul_separated_paths(self):
        old = SRC + "normalization/old.py"
        new = SRC + "fusions/new.py"
        args = argparse.Namespace(merge_ref="HEAD", target=None, source=None)
        result = subprocess.CompletedProcess([], 0, old + "\0" + new + "\0", "")
        with mock.patch.object(selector.subprocess, "run", return_value=result) as run:
            self.assertEqual(selector.changed_files(args), [old, new])
        command = run.call_args.args[0]
        self.assertIn("--no-renames", command)
        self.assertIn("-z", command)
        self.assertEqual(command[-2:], ["HEAD^1", "HEAD"])

    def test_source_mode_uses_merge_base_diff(self):
        args = argparse.Namespace(merge_ref=None, target="origin/main", source="HEAD")
        result = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(selector.subprocess, "run", return_value=result) as run:
            self.assertEqual(selector.changed_files(args), [])
        self.assertIn("origin/main...HEAD", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
