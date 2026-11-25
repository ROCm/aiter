# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Stdlib-only tests of the public tuning module (no AITER device bootstrap)."""

import ast
import builtins
import contextlib
import csv
import functools
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from unittest.mock import patch

_MODULE = Path(__file__).resolve().parents[1] / "aiter/utility/mxfp4_tuning.py"
_HEADER = "gfx,cu_num,M,N,K,kernelName,splitK,us\n"


class TestMXFP4TuningConfig(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("mxfp4_tuning_test", _MODULE)
        self.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.api)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.env = patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("AITER_CONFIG_GEMM_A4W4", None)
        os.environ.pop("AITER_SELECT", None)
        self.temp = patch.object(tempfile, "tempdir", str(self.root))
        self.temp.start()
        self.addCleanup(self.temp.stop)

    def table(self, name, rows):
        path = self.root / name
        path.write_text(_HEADER + rows, encoding="utf-8")
        return path

    def rows(self):
        with open(self.api.get_mxfp4_config_file(), newline="") as stream:
            return list(csv.DictReader(stream))

    def default_rows(self):
        profile_paths = {
            Path(path)
            for profile in self.api.available_mxfp4_profiles()
            for path in self.api._profile_sources(profile)
        }
        paths = [self.api._CONFIG_DIR / "a4w4_blockscale_tuned_gemm.csv"]
        paths.extend(
            sorted(
                path
                for path in (self.api._CONFIG_DIR / "model_configs").glob(
                    "*a4w4_blockscale_tuned_gemm*.csv"
                )
                if path not in profile_paths and "untuned" not in path.name
            )
        )
        rows = {}
        for path in paths:
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    key = (
                        row["gfx"],
                        *(int(row[k]) for k in ("cu_num", "M", "N", "K")),
                    )
                    self.assertNotIn(
                        key, rows, f"ambiguous legacy default key in {path}"
                    )
                    rows[key] = (row["kernelName"], int(row["splitK"]))
        return rows

    def selected_rows(self):
        return {
            (r["gfx"], *(int(r[k]) for k in ("cu_num", "M", "N", "K"))): (
                r["kernelName"],
                int(r["splitK"]),
            )
            for r in self.rows()
        }

    def power_fixture(self):
        stock = self.table(
            "stock.csv",
            "gfx950,256,32,64,128,stock,0,10\n" "gfx950,256,64,64,128,uncovered,0,10\n",
        )
        manifest = self.root / "defaults.json"
        manifest.write_text('["stock.csv"]')
        self.api._CONFIG_DIR = self.root
        self.api._DEFAULT_FILE = manifest
        power_dir = self.root / "power_configs"
        power_dir.mkdir()
        power = power_dir / "example_a4w4_blockscale_tuned_gemm.csv"
        power.write_text(
            "gfx,cu_num,M,N,K,kernelName,splitK,us,select\n"
            "gfx950,256,32,64,128,power,0,12,power\n"
            "gfx950,256,64,64,128,ignore-latency,0,1,latency\n"
        )
        return stock, power

    def test_power_overlay_keeps_uncovered_latency_rows_without_rewriting(self):
        stock, power = self.power_fixture()
        before = {path: path.read_bytes() for path in (stock, power)}
        os.environ["AITER_SELECT"] = "power_efficiency"
        rows = self.selected_rows()
        self.assertEqual(rows[("gfx950", 256, 32, 64, 128)], ("power", 0))
        self.assertEqual(rows[("gfx950", 256, 64, 64, 128)], ("uncovered", 0))
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.assertIn(str(power.resolve()), self.api.mxfp4_tuning_sources())

    def test_explicit_environment_table_bypasses_power_overlay(self):
        stock, _ = self.power_fixture()
        os.environ["AITER_SELECT"] = "power"
        os.environ["AITER_CONFIG_GEMM_A4W4"] = str(stock)
        self.assertEqual(self.rows()[0]["kernelName"], "stock")
        self.assertEqual(self.api.mxfp4_tuning_sources(), (str(stock.resolve()),))

    def test_power_policy_stays_frozen_after_first_lookup(self):
        self.power_fixture()
        os.environ["AITER_SELECT"] = "power"
        self.api.configure_mxfp4_tuning()
        fingerprint = self.api.mxfp4_tuning_fingerprint()
        self.assertEqual(self.rows()[0]["kernelName"], "power")
        os.environ["AITER_SELECT"] = "latency"
        self.assertEqual(self.api.mxfp4_tuning_fingerprint(), fingerprint)
        with self.assertRaisesRegex(RuntimeError, "already resolved"):
            self.api.configure_mxfp4_tuning()

    def test_conflicting_power_tables_are_rejected_without_rewriting(self):
        _, power = self.power_fixture()
        other = power.with_name("other_a4w4_blockscale_tuned_gemm.csv")
        other.write_text(
            power.read_text().replace(",power,0,12,power", ",different,0,9,power")
        )
        before = {path: path.read_bytes() for path in (power, other)}
        os.environ["AITER_SELECT"] = "power"
        with self.assertRaisesRegex(ValueError, "conflicting power"):
            self.rows()
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_default_preserves_all_preexisting_model_selections(self):
        expected = self.default_rows()
        self.assertEqual(self.selected_rows(), expected)

    def test_profiles_preserve_unrelated_default_selections(self):
        expected = self.default_rows()
        for profile in self.api.available_mxfp4_profiles():
            with self.subTest(profile=profile):
                sources = self.api.resolve_mxfp4_tuning_sources(profile=profile)
                model_keys = set()
                for path in self.api._profile_sources(profile):
                    with path.open(newline="") as stream:
                        for row in csv.DictReader(stream):
                            model_keys.add(
                                (
                                    row["gfx"],
                                    *(int(row[k]) for k in ("cu_num", "M", "N", "K")),
                                )
                            )
                payload = self.api._merged_csv(sources).decode()
                actual = {
                    (r["gfx"], *(int(r[k]) for k in ("cu_num", "M", "N", "K"))): (
                        r["kernelName"],
                        int(r["splitK"]),
                    )
                    for r in csv.DictReader(io.StringIO(payload))
                }
                for key in expected.keys() - model_keys:
                    self.assertEqual(actual.get(key), expected[key], key)

    def test_profiles_keep_case_specific_kernel_choices(self):
        expected = {"llama31_8b": "128x512E", "qwen3_8b": "256x256E"}
        for profile, suffix in expected.items():
            self.api.configure_mxfp4_tuning(profile=profile)
            # Inspect ordered profile sources without freezing until selection is final.
            payload = self.api._merged_csv(self.api.mxfp4_tuning_sources()).decode()
            row = next(
                r
                for r in csv.DictReader(payload.splitlines())
                if (r["M"], r["N"], r["K"]) == ("16384", "4096", "6144")
            )
            self.assertTrue(row["kernelName"].endswith(suffix))

    def test_profile_row_counts_include_default_without_collisions(self):
        self.api.configure_mxfp4_tuning(profile="qwen3_fsdp")
        rows = self.rows()
        # The small profile overrides one existing Qwen3 VL row.
        self.assertEqual(len(rows), len(self.default_rows()) + 8 + 13 - 1)
        key = lambda r: tuple(r[k] for k in ("gfx", "cu_num", "M", "N", "K"))
        self.assertEqual(len(rows), len({key(r) for r in rows}))
        ck = next(
            r for r in rows if (r["M"], r["N"], r["K"]) == ("8192", "1024", "1024")
        )
        self.assertTrue(ck["kernelName"].startswith("a4w4_blockscale_"))

    def test_profiles_do_not_merge_unrelated_model_tables(self):
        sources = self.api.resolve_mxfp4_tuning_sources(profile="qwen3_8b")
        self.assertEqual(
            {Path(source).name for source in sources},
            {"qwen3_8b_a4w4_blockscale_tuned_gemm.csv"}
            | {Path(path).name for path in self.api._default_sources()},
        )

    def test_first_source_wins_without_editing_source_tables(self):
        first = self.table("first.csv", "gfx950,256,32,64,128,first,0,10\n")
        second = self.table("second.csv", "gfx950,256,32,64,128,faster,0,1\n")
        before = [p.read_bytes() for p in (first, second)]
        self.api.configure_mxfp4_tuning(
            tuned_config=[first, second], merge_default=False
        )
        self.assertEqual(self.rows()[0]["kernelName"], "first")
        self.assertEqual(before, [p.read_bytes() for p in (first, second)])

    def test_environment_overrides_bundled_profile(self):
        source = self.table("override.csv", "gfx950,256,32,64,128,override,0,1\n")
        os.environ["AITER_CONFIG_GEMM_A4W4"] = str(source)
        self.api.configure_mxfp4_tuning(profile="qwen3_8b")
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows()[0]["kernelName"], "override")

    def test_same_config_repeats_but_different_profile_after_lookup_fails(self):
        self.api.configure_mxfp4_tuning(profile="llama2_7b")
        path = self.api.get_mxfp4_config_file()
        self.api.configure_mxfp4_tuning(profile="llama2_7b")
        self.assertEqual(self.api.get_mxfp4_config_file(), path)
        with self.assertRaisesRegex(RuntimeError, "already resolved"):
            self.api.configure_mxfp4_tuning(profile="qwen3_8b")

    def test_mutated_source_is_not_accepted_as_idempotent_configuration(self):
        source = self.table("source.csv", "gfx950,256,32,64,128,first,0,1\n")
        self.api.configure_mxfp4_tuning(tuned_config=source, merge_default=False)
        self.rows()
        source.write_text(_HEADER + "gfx950,256,32,64,128,changed,0,1\n")
        with self.assertRaisesRegex(RuntimeError, "already resolved"):
            self.api.configure_mxfp4_tuning(tuned_config=source, merge_default=False)

    def test_conflicting_rows_inside_one_source_are_rejected(self):
        source = self.table(
            "bad.csv",
            "gfx950,256,32,64,128,first,0,1\ngfx950,256,32,64,128,second,0,2\n",
        )
        before = source.read_bytes()
        self.api.configure_mxfp4_tuning(tuned_config=source, merge_default=False)
        with self.assertRaisesRegex(ValueError, "conflicting rows"):
            self.rows()
        self.assertEqual(source.read_bytes(), before)

    def test_missing_profile_and_missing_path_fail_early(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.api.configure_mxfp4_tuning(profile="not-a-case")
        with self.assertRaises(FileNotFoundError):
            self.api.configure_mxfp4_tuning(tuned_config=self.root / "missing.csv")

    def test_legacy_cu_metadata_is_normalized_without_source_rewrite(self):
        source = self.root / "legacy.csv"
        source.write_text("cu_num,M,N,K,kernelName,splitK\n256,32,64,128,legacy,0\n")
        before = source.read_bytes()
        self.api.configure_mxfp4_tuning(tuned_config=source, merge_default=False)
        self.assertNotIn("gfx", self.rows()[0])
        self.assertEqual(source.read_bytes(), before)

    def test_mixed_legacy_228_cu_uses_shared_arch_backfill(self):
        source = self.root / "legacy.csv"
        source.write_text("cu_num,M,N,K,kernelName,splitK\n228,32,64,128,legacy,0\n")
        tagged = self.table("tagged.csv", "gfx950,256,32,64,128,new,0,1\n")
        self.api.configure_mxfp4_tuning(
            tuned_config=[source, tagged], merge_default=False
        )
        self.assertEqual(self.rows()[0]["gfx"], "gfx942")

    def test_shared_legacy_backfill_preserves_runtime_fallback(self):
        infer = self.api._arch_helpers().gfx_from_cu_num
        for cu, gfx in (
            (256, "gfx950"),
            (80, "gfx942"),
            (228, "gfx942"),
            (304, "gfx942"),
        ):
            self.assertEqual(infer(cu), gfx)
        self.assertEqual(infer(128, lambda: "gfx950"), "gfx950")
        self.assertEqual(infer("invalid", lambda: "gfx1250"), "gfx1250")
        with self.assertRaisesRegex(ValueError, "explicit gfx"):
            infer(128)

    def test_frozen_sources_report_actual_selection_after_environment_changes(self):
        self.api.configure_mxfp4_tuning(profile="qwen3_8b")
        before = self.api.mxfp4_tuning_sources()
        self.rows()
        os.environ["AITER_CONFIG_GEMM_A4W4"] = str(self.root / "not-used.csv")
        self.assertEqual(self.api.mxfp4_tuning_sources(), before)

    def test_effective_fingerprint_stays_bound_to_consumed_bytes(self):
        source = self.table("source.csv", "gfx950,256,32,64,128,first,0,1\n")
        self.api.configure_mxfp4_tuning(tuned_config=source, merge_default=False)
        preview = self.api.mxfp4_tuning_fingerprint()
        path = Path(self.api.get_mxfp4_config_file())
        self.assertEqual(preview, hashlib.sha256(path.read_bytes()).hexdigest())
        source.write_text(_HEADER + "gfx950,256,32,64,128,new,0,1\n")
        self.assertEqual(self.api.mxfp4_tuning_fingerprint(), preview)
        self.assertEqual(self.rows()[0]["kernelName"], "first")

    def test_configure_cannot_race_first_resolution(self):
        self.api.configure_mxfp4_tuning(profile="llama2_7b")
        reading = threading.Event()
        release = threading.Event()
        configuring = threading.Event()
        merge = self.api._merged_csv

        def blocked_merge(*args):
            reading.set()
            if not release.wait(5):
                raise AssertionError("test did not release the table read")
            return merge(*args)

        def configure_other():
            configuring.set()
            return self.api.configure_mxfp4_tuning(profile="qwen3_8b")

        with patch.object(self.api, "_merged_csv", blocked_merge), ThreadPoolExecutor(
            2
        ) as pool:
            reader = pool.submit(self.api.get_mxfp4_config_file)
            try:
                self.assertTrue(reading.wait(5))
                writer = pool.submit(configure_other)
                self.assertTrue(configuring.wait(5))
                with self.assertRaises(TimeoutError):
                    writer.result(timeout=0.05)
            finally:
                release.set()
            path = reader.result(timeout=5)
            with self.assertRaisesRegex(RuntimeError, "already resolved"):
                writer.result(timeout=5)
        self.assertEqual(self.api.get_mxfp4_config_file(), path)
        self.assertIn("llama2_7b", self.api.mxfp4_tuning_sources()[0])

    def test_packaging_manifest_includes_profile_resources(self):
        manifest = _MODULE.parents[2] / "MANIFEST.in"
        self.assertIn("graft aiter", manifest.read_text())
        self.assertEqual(
            set(self.api.available_mxfp4_profiles()),
            {"llama2_7b", "llama31_8b", "qwen3_8b", "qwen3_small", "qwen3_fsdp"},
        )

    def test_cli_emit_keeps_stdout_to_one_value(self):
        argv = sys.argv
        try:
            for mode in ("path", "fingerprint"):
                sys.argv = ["mxfp4_tuning", "--profile", "qwen3_8b", "--emit", mode]
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    self.api.main()
                output = buffer.getvalue()
                self.assertEqual(output.count("\n"), 1)
                if mode == "path":
                    self.assertTrue(output.strip().endswith(".csv"))
                else:
                    self.assertEqual(len(output.strip()), 64)

            sys.argv = ["mxfp4_tuning", "--profile", "qwen3_8b", "--emit", "sources"]
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.api.main()
            self.assertEqual(
                len(buffer.getvalue().splitlines()),
                len(self.api.mxfp4_tuning_sources()),
            )
        finally:
            sys.argv = argv


class TestMXFP4PrebuildConfig(unittest.TestCase):
    def setUp(self):
        TestMXFP4TuningConfig.setUp(self)
        # Execute only the stdlib config boundary, without native JIT dependencies.
        source = _MODULE.parents[1] / "jit/core.py"
        tree = ast.parse(source.read_text())
        nodes = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in {"_prebuild_mxfp4_tuning", "AITER_CONFIG"}
        ]
        self.core = {
            "functools": functools,
            "importlib": importlib,
            "os": os,
            "sys": sys,
            "AITER_ROOT_DIR": str(_MODULE.parents[2]),
        }
        exec(  # noqa: S102 - trusted repository AST
            compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"),
            self.core,
        )
        self.config = self.core["AITER_CONFIG"]()
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        sys.modules.pop("aiter", None)

    table = TestMXFP4TuningConfig.table

    def test_prebuild_expands_blob_command_without_package_import(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "aiter" or name.startswith("aiter."):
                raise AssertionError("prebuild must not bootstrap aiter")
            return original_import(name, *args, **kwargs)

        table = self.table("prebuild.csv", "gfx950,256,1,2,3,first,0,1\n")
        os.environ["AITER_CONFIG_GEMM_A4W4"] = str(table)
        config_file = _MODULE.parents[1] / "jit/optCompilerConfig.json"
        entries = json.loads(config_file.read_text())
        commands = [
            entry["blob_gen_cmd"]
            for entry in entries.values()
            if "AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE"
            in entry.get("blob_gen_cmd", "")
        ]
        self.assertTrue(commands)
        with patch("builtins.__import__", side_effect=guarded_import):
            for command in commands:
                expanded = eval(
                    command, {"AITER_CONFIGS": self.config, "AITER_CSRC_DIR": "csrc"}
                )
                self.assertIn(self.config.AITER_CONFIG_GEMM_A4W4_FILE, expanded)
                with open(self.config.AITER_CONFIG_GEMM_A4W4_FILE) as stream:
                    self.assertEqual(
                        next(csv.DictReader(stream))["kernelName"], "first"
                    )
        self.assertNotIn("aiter", sys.modules)

    def test_prebuild_freezes_once_and_runtime_uses_public_state(self):
        first = self.table("first.csv", "gfx950,256,1,2,3,first,0,1\n")
        second = self.table("second.csv", "gfx950,256,1,2,3,second,0,1\n")
        os.environ["AITER_CONFIG_GEMM_A4W4"] = str(first)
        first_path = self.config.AITER_CONFIG_GEMM_A4W4_FILE
        with open(first_path) as stream:
            self.assertEqual(next(csv.DictReader(stream))["kernelName"], "first")
        os.environ["AITER_CONFIG_GEMM_A4W4"] = str(second)
        self.assertEqual(self.config.AITER_CONFIG_GEMM_A4W4_FILE, first_path)
        os.environ.pop("AITER_CONFIG_GEMM_A4W4")
        self.api.configure_mxfp4_tuning(tuned_config=[str(second)], merge_default=False)
        package = types.ModuleType("aiter")
        utility = types.ModuleType("aiter.utility")
        utility.mxfp4_tuning = self.api
        package.utility = utility
        sys.modules["aiter"] = package
        sys.modules["aiter.utility"] = utility
        sys.modules["aiter.utility.mxfp4_tuning"] = self.api
        self.assertEqual(
            self.config.AITER_CONFIG_GEMM_A4W4_FILE, self.api.get_mxfp4_config_file()
        )
        with open(self.config.AITER_CONFIG_GEMM_A4W4_FILE) as stream:
            self.assertEqual(next(csv.DictReader(stream))["kernelName"], "second")


if __name__ == "__main__":
    unittest.main()
