# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Raw trace sample preservation, parser failures, and report reproducibility."""

import copy
import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

# Import the standalone tool without executing aiter/__init__.py or loading HIP.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aiter"))
import smi_trace_plot


def example(function="run_gemm_bpreshuffle", start=1000.0, count=37):
    """Clearly synthetic data with irregular timing and missing firmware values."""
    samples = []
    elapsed = 0.017
    for index in range(count):
        elapsed += (0.043, 0.067, 0.055, 0.081)[index % 4]
        sample = {
            "timestamp_s": start + elapsed,
            "gfx_clk_mhz": 1900 + 180 * math.sin(index * 0.8),
            "fclk_mhz": 1650 + 75 * math.cos(index * 0.3),
            "soc_clk_mhz": "N/A",
            "power_w": 1700 + 300 * math.sin(index * 0.3),
            "gfx_activity_pct": 85 + index % 13,
            "umc_activity_pct": 22 + index % 7,
            "temp_hotspot_c": 52 + index / 5,
            "vram_used_mb": 20000,
            "synthetic_extra_field": index,
        }
        if index in (8, 9):
            del sample["gfx_clk_mhz"]
        if index == 12:
            sample["gfx_clk_mhz"] = "N/A"
        if index == 21:
            sample["fclk_mhz"] = None
        samples.append(sample)
    return {
        "schema_version": 1,
        "label": "SYNTHETIC_DEVELOPMENT_FIXTURE/dtype=bfloat16/M=512/N=65536/K=1536/seed=0/"
        + function
        + "#1",
        "device": 0,
        "interval_s": 0.05,
        "duration_s": elapsed + 0.1,
        "launches": 0,
        "metrics": {},
        "sample_status": "ok",
        "sample_count": len(samples),
        "window_start_monotonic_s": start,
        "window_end_monotonic_s": start + elapsed + 0.1,
        "window_start_unix_s": 1700000000.0 + start,
        "samples": samples,
    }


class TracePlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "report"

    def write_records(self, name, records, *, prefix=False, preamble=""):
        path = self.root / name
        marker = smi_trace_plot.PREFIX if prefix else ""
        path.write_text(
            preamble
            + "\n".join(marker + json.dumps(record) for record in records)
            + "\n",
            encoding="utf-8",
        )
        return path

    def read(self, input_path):
        return smi_trace_plot.read_inputs(input_path, self.output, ["*.log", "*.jsonl"])

    def test_all_plotting_points_preserve_irregular_timing_and_missing_values(self):
        originals = [example(), example("run_gemm_abpreshuffle", 1200.0)]
        source = self.write_records(
            "samples.jsonl", originals, prefix=True, preamble="Synthetic fixture only\n"
        )
        records, sources, warnings = self.read(source)
        self.assertEqual([record.line for record in records], [2, 3])
        self.assertEqual(sources[0]["samples"], 74)
        self.assertFalse(warnings)
        for record, original in zip(records, originals):
            self.assertEqual(record.data, original)
            self.assertGreater(record.elapsed[0], 0)
            gaps = [b - a for a, b in zip(record.elapsed, record.elapsed[1:])]
            self.assertGreater(max(gaps) - min(gaps), 0.03)
            for metric in smi_trace_plot.METRIC_KEYS:
                x, y = smi_trace_plot.series(record, metric)
                self.assertEqual(len(x), original["sample_count"])
                self.assertEqual(len(y), original["sample_count"])
                for timestamp, value, sample in zip(x, y, original["samples"]):
                    self.assertEqual(
                        timestamp,
                        sample["timestamp_s"] - original["window_start_monotonic_s"],
                    )
                    expected = sample.get(metric)
                    if expected is None or expected == "N/A":
                        self.assertTrue(math.isnan(value))
                    else:
                        self.assertEqual(value, expected)

    def test_exports_preserve_every_raw_sample_and_rerender_identically(self):
        originals = [example(), example("run_gemm_abpreshuffle", 1200.0)]
        source = self.write_records("samples.jsonl", originals, prefix=True)
        manifest = smi_trace_plot.build(source, self.output, synthetic=True)
        self.assertEqual(manifest["case_count"], 1)
        self.assertEqual(manifest["record_count"], 2)
        self.assertEqual(manifest["sample_count"], 74)
        self.assertTrue(manifest["synthetic_development_fixture"])
        self.assertEqual(manifest["charts"][0]["source_lines"], [1, 2])
        self.assertEqual(manifest["runs"][0]["available_values"]["gfx_clk_mhz"], 34)
        self.assertEqual(manifest["runs"][0]["available_values"]["fclk_mhz"], 36)
        self.assertEqual(manifest["runs"][0]["available_values"]["soc_clk_mhz"], 0)
        self.assertEqual(
            manifest["sources"][0]["sha256"],
            hashlib.sha256(source.read_bytes()).hexdigest(),
        )
        with (self.output / "raw_points.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        expected_samples = [
            (record, index, sample)
            for record in originals
            for index, sample in enumerate(record["samples"])
        ]
        self.assertEqual(len(rows), len(expected_samples))
        for row, (record, index, sample) in zip(rows, expected_samples):
            self.assertEqual(int(row["sample_index"]), index)
            self.assertEqual(row["label"], record["label"])
            self.assertEqual(float(row["timestamp_s"]), sample["timestamp_s"])
            self.assertEqual(
                float(row["elapsed_s"]),
                sample["timestamp_s"] - record["window_start_monotonic_s"],
            )
            self.assertEqual(json.loads(row["raw_sample_json"]), sample)
            for metric in smi_trace_plot.METRIC_KEYS:
                expected = sample.get(metric)
                if expected is None:
                    self.assertEqual(row[metric], "")
                elif expected == "N/A":
                    self.assertEqual(row[metric], "N/A")
                else:
                    self.assertEqual(float(row[metric]), expected)
        artifacts = [manifest["charts"][0][kind] for kind in ("png", "svg")] + [
            "raw_points.csv",
            "manifest.json",
            "index.html",
        ]
        before = {name: (self.output / name).read_bytes() for name in artifacts}
        smi_trace_plot.build(source, self.output, synthetic=True)
        after = {name: (self.output / name).read_bytes() for name in artifacts}
        self.assertEqual(before, after)
        self.assertIn(
            "SYNTHETIC DEVELOPMENT FIXTURE", (self.output / "index.html").read_text()
        )

    def test_grouping_retains_source_order_settings_and_repeated_runs(self):
        first = example()
        first["label"] = first["label"].replace("N=65536", "N=8192")
        second = example("run_gemm_abpreshuffle", 1200.0)
        third = copy.deepcopy(second)
        third["label"] = third["label"].replace("seed=0", "seed=1")
        self.write_records("a.jsonl", [first, second, third, first])
        self.write_records("b.log", [first], prefix=True)
        self.output.mkdir()
        (self.output / "generated.jsonl").write_text(json.dumps(first))
        records, sources, _ = self.read(self.root)
        self.assertEqual(len(records), 5)
        self.assertEqual(len(sources), 2)
        self.assertEqual(len({record.case_key for record in records}), 4)
        self.assertEqual(
            [record.line for record in records if record.source == "a.jsonl"],
            [1, 2, 3, 4],
        )
        self.assertEqual(
            [dict(record.params)["N"] for record in records],
            ["8192", "65536", "65536", "8192", "8192"],
        )
        self.assertEqual(records[0].case_key, records[3].case_key)
        self.assertNotEqual(records[0].case_key, records[4].case_key)

    def test_duplicate_parameters_normalize_without_discarding_conflicts(self):
        original = example()
        duplicate = copy.deepcopy(original)
        duplicate["label"] = duplicate["label"].replace(
            "dtype=bfloat16/", "dtype=bfloat16/dtype=bfloat16/"
        )
        conflicting = copy.deepcopy(original)
        conflicting["label"] = conflicting["label"].replace("seed=0/", "seed=0/seed=1/")
        records, _, _ = self.read(
            self.write_records("labels.jsonl", [original, duplicate, conflicting])
        )
        self.assertEqual(records[0].case_key, records[1].case_key)
        self.assertNotEqual(records[0].case_key, records[2].case_key)
        self.assertEqual(records[1].data["label"], duplicate["label"])

    def test_invalid_counts_timestamps_bounds_and_metrics_are_rejected(self):
        invalid_cases = []
        count = example()
        count["sample_count"] += 1
        invalid_cases.append((count, "sample_count"))
        missing = example()
        del missing["samples"][0]["timestamp_s"]
        invalid_cases.append((missing, "timestamp_s"))
        reversed_times = example()
        reversed_times["samples"][1]["timestamp_s"] = (
            reversed_times["samples"][0]["timestamp_s"] - 1
        )
        invalid_cases.append((reversed_times, "nondecreasing"))
        invalid_bounds = example()
        invalid_bounds["window_end_monotonic_s"] = (
            invalid_bounds["window_start_monotonic_s"] - 1
        )
        invalid_cases.append((invalid_bounds, "invalid window bounds"))
        for value in (0, -1, float("inf")):
            interval = example()
            interval["interval_s"] = value
            invalid_cases.append((interval, "interval"))
        for value in ("corrupt measurement", float("inf"), True):
            metric = example()
            metric["samples"][0]["soc_clk_mhz"] = value
            invalid_cases.append((metric, "nonfinite/nonnumeric"))
        for index, (invalid, message) in enumerate(invalid_cases):
            with self.subTest(index=index, message=message):
                source = self.write_records("invalid.jsonl", [invalid], prefix=True)
                with self.assertRaisesRegex(ValueError, message):
                    self.read(source)

    def test_samples_outside_bounds_are_retained_at_original_timestamps(self):
        original = example()
        original["samples"][0]["timestamp_s"] = (
            original["window_start_monotonic_s"] - 0.1
        )
        original["samples"][-1]["timestamp_s"] = (
            original["window_end_monotonic_s"] + 0.1
        )
        records, _, warnings = self.read(
            self.write_records("outside.jsonl", [original])
        )
        self.assertEqual(records[0].data["samples"], original["samples"])
        self.assertLess(records[0].elapsed[0], 0)
        self.assertGreater(records[0].elapsed[-1], original["duration_s"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("2 samples outside recorded bounds", warnings[0])

    def test_empty_run_retained_with_warning(self):
        original = example(count=0)
        records, _, warnings = self.read(self.write_records("empty.jsonl", [original]))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].elapsed, [])
        self.assertIn("no samples", warnings[0])

    def test_aggregate_only_records_cannot_be_used_as_raw_samples(self):
        source = self.root / "aggregate.log"
        source.write_text(
            'AITER_SMI_RESULT {"label":"test", "samples":37, "metrics":{}}\n'
        )
        with self.assertRaisesRegex(ValueError, "No raw AITER_SMI_TRACE records"):
            self.read(source)


if __name__ == "__main__":
    unittest.main()
