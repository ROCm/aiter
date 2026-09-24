# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Parser edge cases and end-to-end reproducibility of the exported report."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Import the standalone tool without executing aiter/__init__.py or loading HIP.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aiter"))
import smi_plot as plot_smi


def example(label="test_gemm.test_gemm/M=512/N=6144/K=7168/run_gemm_bpreshuffle#1"):
    return {
        "label": label,
        "device": 0,
        "duration_s": 3.0,
        "interval_s": 0.05,
        "launches": 1000,
        "samples": 60,
        "sample_status": "ok",
        "metrics": {
            "gfx_clk_mhz": {
                "min": 1800,
                "mean": 2000,
                "median": 2050,
                "max": 2200,
                "n": 60,
            },
            "fclk_mhz": {
                "min": 1950,
                "mean": 1950,
                "median": 1950,
                "max": 1950,
                "n": 60,
            },
        },
    }


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def read(self):
        return plot_smi.read_records(
            self.root, self.root / "report", ["*.log", "*.jsonl"]
        )

    def test_console_prefix_bare_jsonl_and_duplicates(self):
        data = example()
        raw = json.dumps(data)
        (self.root / "run.log").write_text(
            "ordinary log\n[worker] " + plot_smi.PREFIX + raw + "\n" + raw + "\n"
        )
        records, sources, warnings = self.read()
        self.assertEqual(len(records), 2)
        self.assertEqual([r.line for r in records], [2, 3])
        self.assertEqual(sources[0]["records"], 2)
        self.assertFalse(warnings)
        self.assertEqual(len(set(plot_smi.descriptions(records))), 2)

    def test_recursive_natural_shape_order_and_output_exclusion(self):
        (self.root / "nested").mkdir()
        (self.root / "report").mkdir()
        for name, n in (("z", 2), ("a", 10)):
            (self.root / "nested" / (name + ".log")).write_text(
                json.dumps(example(f"test/M=1/N={n}/kernel#1"))
            )
        (self.root / "report" / "old.log").write_text(json.dumps(example()))
        records, sources, _ = self.read()
        self.assertEqual([dict(r.params)["N"] for r in records], ["2", "10"])
        self.assertEqual(len(sources), 2)

    def test_malformed_marked_record_is_not_silently_dropped(self):
        (self.root / "bad.log").write_text(plot_smi.PREFIX + '{"label":')
        with self.assertRaisesRegex(ValueError, "bad.log:1: invalid SMI JSON"):
            self.read()

    def test_missing_metric_is_omitted_and_poor_samples_retained(self):
        data = example()
        data["metrics"]["fclk_mhz"] = {"mean": None, "min": "N/A", "max": None}
        data["sample_status"] = "insufficient"
        (self.root / "run.log").write_text(json.dumps(data))
        records, _, warnings = self.read()
        self.assertEqual(len(records), 1)
        self.assertNotIn("fclk_mhz", records[0].data["metrics"])
        self.assertEqual(len(warnings), 2)

    def test_invalid_range_fails_with_source_location(self):
        data = example()
        data["metrics"]["gfx_clk_mhz"]["mean"] = 5000
        (self.root / "bad.log").write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "bad.log:1: invalid min/mean/max"):
            self.read()

    def test_labels_keep_distinct_configuration_values(self):
        _test, params, function, occurrence = plot_smi.split_label(
            "test_gemm.test_gemm/dtype=bf16/M=512/dtype=bf16/variant=x/variant=y/run_gemm_abpreshuffle#2"
        )
        self.assertEqual(params.count(("dtype", "bf16")), 1)
        self.assertIn(("variant", "x"), params)
        self.assertIn(("variant", "y"), params)
        self.assertEqual((function, occurrence), ("run_gemm_abpreshuffle", 2))

    def test_axis_limits_use_all_records(self):
        data = example()
        record = plot_smi.Record(
            "run.log", 1, data, *plot_smi.split_label(data["label"])
        )
        limits = plot_smi.axis_limits([record])
        self.assertLess(limits["Clocks (MHz)"][0], 1800)
        self.assertGreater(limits["Clocks (MHz)"][1], 2200)
        self.assertEqual(limits["Activity (%)"], (0, 100))


class ReproducibilityTests(unittest.TestCase):
    def test_directory_report_is_byte_identical_across_hash_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input"
            inputs.mkdir()
            a = example()
            b = copy.deepcopy(a)
            b["label"] = b["label"].replace(
                "run_gemm_bpreshuffle", "run_gemm_abpreshuffle"
            )
            c = example("test_other.test_op/M=8/kernel#1")
            c["sample_status"] = "insufficient"
            (inputs / "z.log").write_text(plot_smi.PREFIX + json.dumps(c) + "\n")
            (inputs / "a.jsonl").write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
            outputs = [root / "one", root / "two"]
            for seed, output in zip(("1", "29"), outputs):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(plot_smi.__file__)),
                        str(inputs),
                        "-o",
                        str(output),
                        "--rows-per-page",
                        "2",
                    ],
                    env={**os.environ, "PYTHONHASHSEED": seed},
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

            def hashes(path):
                return {
                    str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in path.rglob("*")
                    if p.is_file()
                }

            self.assertEqual(hashes(outputs[0]), hashes(outputs[1]))
            manifest = json.loads((outputs[0] / "manifest.json").read_text())
            self.assertEqual(manifest["records"], 3)
            self.assertEqual(
                len([p for p in manifest["plots"] if p["kind"] == "test"]), 2
            )
            self.assertEqual(
                len([p for p in manifest["plots"] if p["kind"] == "case"]), 2
            )
            self.assertEqual(
                len([p for p in manifest["plots"] if p["kind"] == "overview"]), 2
            )
            self.assertIn("insufficient", (outputs[0] / "index.html").read_text())


if __name__ == "__main__":
    unittest.main()
