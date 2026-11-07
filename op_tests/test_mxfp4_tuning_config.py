# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Stdlib-only tests of the public tuning module (no AITER device bootstrap)."""

import contextlib
import csv
import hashlib
import importlib.util
import io
import os
import sys
import tempfile
import threading
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
        # Profiles are opt-in: unrelated model tables must not be merged.
        stock = self.api._CONFIG_DIR / "a4w4_blockscale_tuned_gemm.csv"
        with open(stock, newline="") as stream:
            stock_rows = len(list(csv.DictReader(stream)))
        self.assertEqual(len(rows), stock_rows + 8 + 13)
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
            {
                "qwen3_8b_a4w4_blockscale_tuned_gemm.csv",
                "a4w4_blockscale_tuned_gemm.csv",
            },
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
        self.assertEqual(self.rows()[0]["gfx"], "gfx950")
        self.assertEqual(source.read_bytes(), before)

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
                len(buffer.getvalue().splitlines()), 2
            )  # model table + stock
        finally:
            sys.argv = argv


if __name__ == "__main__":
    unittest.main()
