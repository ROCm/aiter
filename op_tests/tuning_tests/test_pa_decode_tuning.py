# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure unittest coverage: no Torch, Triton, FlyDSL, aiter runtime or GPU.

Run: python -B -m unittest op_tests.tuning_tests.test_pa_decode_tuning -v
"""

import copy
import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "aiter/ops/flydsl/pa_decode_tuning.py"
SPEC = importlib.util.spec_from_file_location("offline_pa_decode_tuning", MODULE)
tuning = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tuning)
SOURCES = {name: "a" * 64 for name in tuning.SOURCE_FILES}


def passing_candidate(group, samples):
    return {
        **copy.deepcopy(group),
        "status": "PASS",
        "samples_us": samples,
        "errors": [],
        "plan_unchanged": True,
        "accuracy": {
            stage: {
                "passed": True,
                "finite": True,
                "atol": 0.005,
                "rtol": 0.005,
                "elements": 32768,
                "max_abs_error": 0.001,
                "max_tolerance_ratio": 0.2,
            }
            for stage in tuning.ACCURACY_STAGES
        },
    }


def result_fixture():
    shape = tuning.make_shape(tuning.resolve_model(), 4, 129, 4)
    key = tuning.make_key(shape, "gfx950", 256, SOURCES)
    groups = tuning.candidate_groups([128, 256, 512, 1024], 4, 1, 256)
    candidates = [
        passing_candidate(group, [latency - 0.1, latency, latency + 0.1])
        for group, latency in zip(groups, (10.2, 10.0, 12.0, 11.0))
    ]
    kv_bytes = tuning.unique_kv_bytes(shape)
    record = {
        "key": key,
        "key_sha256": tuning.fingerprint(key),
        "status": "PASS",
        "candidates": candidates,
        "selection": tuning.summarize_candidates(candidates, kv_bytes),
        "unique_kv_bytes": kv_bytes,
        "benchmark_input": {k: shape[k] for k in tuning.BENCHMARK_FIELDS},
        "sources_after": SOURCES,
    }
    return {"schema_version": 1, "records": [record]}, shape, key


class ModelsAndShapes(unittest.TestCase):
    def config(self, value):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "config.json"
        path.write_text(json.dumps(value))
        return path

    def test_preset_geometries_and_explicit_dimensions(self):
        expected = {
            "mqa-synthetic": (16, 1, 128),
            "qwen3-235b-a22b": (64, 4, 128),
            "qwen3-32b": (64, 8, 128),
            "qwen3-8b": (32, 8, 128),
            "gemma3-4b": (8, 4, 256),
        }
        for name, heads in expected.items():
            with self.subTest(name=name):
                model = tuning.resolve_model(name)
                self.assertEqual(
                    tuple(
                        model[k]
                        for k in ("num_query_heads", "num_kv_heads", "head_dim")
                    ),
                    heads,
                )
        path = self.config(
            {
                "text_config": {
                    "num_attention_heads": 64,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "hidden_size": 5120,
                }
            }
        )
        self.assertEqual(tuning.resolve_model(config=path)["head_dim"], 128)
        self.assertEqual(
            tuning.resolve_model(config=path, num_query_heads=64)["head_dim"], 128
        )
        self.assertEqual(
            tuning.resolve_model(config=path, head_dim=256)["head_dim"], 256
        )

    def test_nested_config_fallback_and_global_overrides(self):
        path = self.config(
            {
                "text_config": {
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "hidden_size": 4096,
                }
            }
        )
        model = tuning.resolve_model(
            config=path, num_query_heads=64, num_kv_heads=4, head_dim=128, tp_size=2
        )
        self.assertEqual(
            (
                model["num_query_heads"],
                model["num_kv_heads"],
                model["query_group_size"],
                model["head_dim"],
            ),
            (32, 2, 16, 128),
        )
        self.assertEqual(model["global_num_query_heads"], 64)
        self.assertEqual(tuning.resolve_model(config=path)["head_dim"], 128)
        self.assertEqual(len(model["origin"]["config_sha256"]), 64)

    def test_malformed_configs_and_missing_fields(self):
        for config in (
            None,
            [],
            3,
            {"text_config": []},
            {"text_config": None},
            {"num_attention_heads": 8, "num_key_value_heads": None, "head_dim": 128},
            {"num_attention_heads": 3, "num_key_value_heads": 1, "hidden_size": 100},
            {"num_attention_heads": True, "num_key_value_heads": 1, "head_dim": 128},
            {"kv_lora_rank": 512},
        ):
            with self.subTest(config=config), self.assertRaises(
                (TypeError, ValueError)
            ):
                tuning.resolve_model(config=self.config(config))
        path = self.config({"num_attention_heads": 8, "head_dim": 128})
        self.assertEqual(tuning.resolve_model(config=path)["num_kv_heads"], 8)
        self.assertEqual(
            tuning.resolve_model(config=path, num_kv_heads=8)["num_kv_heads"], 8
        )
        with self.assertRaises(ValueError):
            tuning.resolve_model("qwen3-8b", path)

    def test_tp_sharding_and_explicit_even_replication(self):
        model = tuning.resolve_model("qwen3-235b-a22b", tp_size=4)
        self.assertEqual((model["num_query_heads"], model["num_kv_heads"]), (16, 1))
        with self.assertRaises(ValueError):
            tuning.resolve_model("qwen3-235b-a22b", tp_size=8)
        model = tuning.resolve_model(
            "qwen3-235b-a22b", tp_size=8, allow_kv_replication=True
        )
        self.assertEqual(
            (
                model["num_query_heads"],
                model["num_kv_heads"],
                model["query_group_size"],
                model["kv_replication_factor"],
            ),
            (8, 1, 8, 2),
        )
        for options in (
            {"tp_size": 3},
            {"tp_size": True},
            {"tp_size": 0},
            {"allow_kv_replication": 1},
            {"num_query_heads": 16, "num_kv_heads": 3},
            {
                "num_query_heads": 48,
                "num_kv_heads": 6,
                "tp_size": 8,
                "allow_kv_replication": True,
            },
            {"head_dim": 80},
            {"head_dim": 192},
            {"head_dim": 1152},
        ):
            with self.subTest(options=options), self.assertRaises(
                (TypeError, ValueError)
            ):
                tuning.resolve_model(**options)
        self.assertEqual(tuning.resolve_model(head_dim=64)["head_dim"], 64)

    def test_seeded_varlen_and_logical_window_union(self):
        model = tuning.resolve_model()
        shape = tuning.make_shape(model, 8, 129, 4, length_mode="varlen", seed=19)
        self.assertEqual(
            shape, tuning.make_shape(model, 8, 129, 4, length_mode="varlen", seed=19)
        )
        self.assertEqual((shape["lengths"][0], shape["lengths"][-1]), (4, 129))
        self.assertTrue(all(4 <= n <= 129 for n in shape["lengths"]))
        self.assertEqual(shape["lengths_sha256"], tuning.fingerprint(shape["lengths"]))
        self.assertEqual(tuning.unique_kv_bytes(shape), 2 * 128 * sum(shape["lengths"]))
        windowed = tuning.make_shape(
            model, 8, 129, 4, window=32, length_mode="varlen", seed=19
        )
        self.assertEqual(
            tuning.unique_kv_bytes(windowed),
            2 * 128 * sum(min(n, 35) for n in shape["lengths"]),
        )
        single = tuning.make_shape(model, 1, 129, length_mode="varlen")
        self.assertEqual(single["lengths"], [129])

    def test_shape_validation(self):
        model = tuning.resolve_model()
        for options in (
            {"batch_size": 0},
            {"batch_size": 4097},
            {"context_length": 2**31},
            {"context_length": True},
            {"query_length": 130},
            {"page_size": 32},
            {"dtype": "float32"},
            {"per_token": 1},
            {"trans_v": "yes"},
            {"window": -1},
            {"seed": -1},
            {"seed": 2**63},
        ):
            arguments = {"batch_size": 4, "context_length": 129, **options}
            with self.subTest(options=options), self.assertRaises(
                (TypeError, ValueError)
            ):
                tuning.make_shape(model, **arguments)

    def test_static_key_excludes_distribution_and_sample_identity(self):
        model = tuning.resolve_model()
        uniform = tuning.make_shape(model, 8, 129, 4)
        varlen = tuning.make_shape(model, 8, 129, 4, length_mode="varlen", seed=73)
        key = tuning.make_key(uniform, "gfx950", 256, SOURCES)
        self.assertEqual(key, tuning.make_key(varlen, "gfx950", 256, SOURCES))
        self.assertEqual(set(key["shape"]), set(tuning.STATIC_SHAPE_FIELDS))
        self.assertTrue(set(tuning.BENCHMARK_FIELDS).isdisjoint(key["shape"]))
        self.assertEqual(key, tuning.make_key(key["shape"], "gfx950", 256, SOURCES))
        self.assertEqual(
            key,
            tuning.make_key(
                {**uniform, "unexpected_distribution_field": [1, 2]},
                "gfx950",
                256,
                SOURCES,
            ),
        )
        for options in (
            {"batch_size": 4},
            {"context_length": 257},
            {"query_length": 1},
            {"window": 32},
            {"page_size": 16},
            {"dtype": "float16"},
            {"per_token": False},
            {"trans_v": False},
        ):
            arguments = {
                "batch_size": 8,
                "context_length": 129,
                "query_length": 4,
                **options,
            }
            with self.subTest(options=options):
                self.assertNotEqual(
                    key,
                    tuning.make_key(
                        tuning.make_shape(model, **arguments), "gfx950", 256, SOURCES
                    ),
                )
        for options in (
            {"num_query_heads": 8},
            {"num_kv_heads": 2},
            {"head_dim": 256},
            {"tp_size": 2, "allow_kv_replication": True},
        ):
            with self.subTest(options=options):
                other = tuning.make_shape(tuning.resolve_model(**options), 8, 129, 4)
                self.assertNotEqual(key, tuning.make_key(other, "gfx950", 256, SOURCES))
        self.assertNotEqual(key, tuning.make_key(uniform, "gfx942", 256, SOURCES))
        self.assertNotEqual(key, tuning.make_key(uniform, "gfx950", 128, SOURCES))
        self.assertNotEqual(
            key,
            tuning.make_key(
                uniform, "gfx950", 256, {**SOURCES, "pa_decode.py": "b" * 64}
            ),
        )
        with self.assertRaises(ValueError):
            tuning.make_key({**uniform, "num_kv_heads": 3}, "gfx950", 256, SOURCES)
        with self.assertRaises(ValueError):
            tuning.make_key(uniform, "gfx950", 256, {})


class CandidatesAndSelection(unittest.TestCase):
    def test_capacity_floor_cap_ceil_and_baseline_aliases(self):
        groups = tuning.candidate_groups([128, 256, 512, 1024, 2048], 200, 16, 256)
        self.assertEqual(len(groups), 1)
        self.assertEqual(
            (
                groups[0]["capacity"],
                groups[0]["launch_budget"],
                groups[0]["workgroup_budget"],
            ),
            (200, 512, 128),
        )
        groups = tuning.candidate_groups([128, 256, 1024, 4096], 1, 1, 256)
        self.assertEqual([g["capacity"] for g in groups], [128, 256])
        self.assertEqual(groups[1]["budget_aliases"], [256, 512, 1024, 4096])
        self.assertEqual(groups[1]["launch_budget"], 512)
        groups = tuning.candidate_groups([2048], 200, 3, 256)
        self.assertEqual(groups[-1]["capacity"], 683)
        for budgets in ([], [0], [-1], [True], [1.0]):
            with self.subTest(budgets=budgets), self.assertRaises(
                (TypeError, ValueError)
            ):
                tuning.candidate_groups(budgets, 4, 1, 256)

    def test_rotating_orders_are_balanced_in_both_directions(self):
        for count in (1, 2, 3, 5, 8):
            orders = [tuning.rotated_order(count, r) for r in range(2 * count)]
            for order in orders:
                self.assertEqual(sorted(order), list(range(count)))
            for position in range(count):
                self.assertEqual(
                    sorted(order[position] for order in orders),
                    sorted(list(range(count)) * 2),
                )
        self.assertEqual(tuning.rotated_order(2, 0), [0, 1])
        self.assertEqual(tuning.rotated_order(2, 1), [1, 0])

    def test_selection_metrics_and_smallest_within_97_percent(self):
        data, _, _ = result_fixture()
        record = data["records"][0]
        self.assertEqual(record["selection"]["best_budget"], 256)
        self.assertEqual(record["selection"]["conservative_budget"], 128)
        self.assertEqual(record["selection"]["near_optimal_budgets"], [128, 256])
        best = record["candidates"][1]
        self.assertAlmostEqual(best["baseline_speedup"], 1.2)
        self.assertAlmostEqual(best["unique_kv_tb_s"], record["unique_kv_bytes"] / 10e6)
        self.assertEqual(len(best["round_speedups"]), 3)

    def test_accuracy_finiteness_immutability_and_sample_gates(self):
        data, _, _ = result_fixture()
        valid = data["records"][0]["candidates"][0]
        self.assertTrue(tuning.candidate_valid(valid))
        for update in (
            {"status": "FAIL"},
            {"status": "OOM"},
            {"plan_unchanged": False},
            {"samples_us": []},
            {"samples_us": [0]},
            {"samples_us": [-1]},
            {"samples_us": [math.nan]},
            {"samples_us": [math.inf]},
            {"samples_us": [True]},
            {"accuracy": None},
        ):
            with self.subTest(update=update):
                self.assertFalse(tuning.candidate_valid({**valid, **update}))
        for stage in tuning.ACCURACY_STAGES:
            for field, value in (
                ("passed", False),
                ("finite", False),
                ("atol", 0.01),
                ("rtol", 0.01),
                ("max_abs_error", -1),
                ("max_abs_error", "invalid"),
                ("max_tolerance_ratio", 1.001),
                ("max_tolerance_ratio", math.inf),
                ("elements", 0),
            ):
                candidate = copy.deepcopy(valid)
                candidate["accuracy"][stage][field] = value
                with self.subTest(stage=stage, field=field, value=value):
                    self.assertFalse(tuning.candidate_valid(candidate))

    def test_baseline_failure_and_all_fail_have_no_recommendation(self):
        data, _, _ = result_fixture()
        record = data["records"][0]
        candidates = record["candidates"]
        for candidate in candidates:
            if candidate["is_baseline"]:
                candidate["status"] = "FAIL"
        self.assertIsNone(
            tuning.summarize_candidates(candidates, record["unique_kv_bytes"])
        )
        for candidate in candidates:
            candidate["status"] = "OOM"
        self.assertIsNone(
            tuning.summarize_candidates(candidates, record["unique_kv_bytes"])
        )
        self.assertIsNone(tuning.summarize_candidates([], record["unique_kv_bytes"]))

    def test_mismatched_round_counts_are_not_ranked(self):
        data, _, _ = result_fixture()
        record = data["records"][0]
        record["candidates"][1]["samples_us"] = [1.0]
        selection = tuning.summarize_candidates(
            record["candidates"], record["unique_kv_bytes"]
        )
        self.assertEqual(selection["best_budget"], 128)
        self.assertEqual(selection["valid_capacities"], 3)

    def test_nonfinite_derived_metrics_cannot_win(self):
        data, _, _ = result_fixture()
        record = data["records"][0]
        record["candidates"][1]["samples_us"] = [5e-324] * 3
        selection = tuning.summarize_candidates(
            record["candidates"], record["unique_kv_bytes"]
        )
        self.assertEqual(selection["best_budget"], 128)
        baseline = next(c for c in record["candidates"] if c["is_baseline"])
        baseline["samples_us"] = [1e308, 1e308]
        self.assertIsNone(
            tuning.summarize_candidates(record["candidates"], record["unique_kv_bytes"])
        )
        self.assertFalse(tuning._finite_positive(10**1000))


class ArtifactsAndRuntimeBoundaries(unittest.TestCase):
    def test_json_csv_round_trip_and_no_overwrite(self):
        data, shape, key = result_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path, csv_path = (
                Path(directory) / "result.json",
                Path(directory) / "result.csv",
            )
            tuning.save_results(path, data)
            loaded = tuning.load_results(path)
            self.assertEqual(loaded, data)
            self.assertEqual(tuning.lookup_budget(loaded, key), 128)
            self.assertEqual(tuning.lookup_budget(loaded, key, best=True), 256)
            other_sample = tuning.make_shape(
                tuning.resolve_model(),
                shape["batch_size"],
                shape["context_length"],
                shape["query_length"],
                length_mode="varlen",
                seed=93,
            )
            self.assertEqual(
                tuning.lookup_budget(
                    loaded, tuning.make_key(other_sample, "gfx950", 256, SOURCES)
                ),
                128,
            )
            with self.assertRaises(FileExistsError):
                tuning.save_results(path, data)
            tuning.save_results(path, data, overwrite=True)
            self.assertEqual(tuning.load_results(path), data)
            self.assertEqual(
                [p.name for p in Path(directory).iterdir()], ["result.json"]
            )
            tuning.save_csv(csv_path, data)
            with csv_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(json.loads(rows[0]["key_json"]), key)
            self.assertEqual(
                json.loads(rows[0]["benchmark_input_json"])["lengths"], shape["lengths"]
            )
            self.assertEqual(sum(row["is_best"] == "True" for row in rows), 1)
            with self.assertRaises(FileExistsError):
                tuning.save_csv(csv_path, data)

    def test_lookup_fails_closed_for_wrong_key_or_invalid_result(self):
        data, _, key = result_fixture()
        for bad_key in (
            {**key, "num_cu": 128},
            {**key, "architecture": "gfx942"},
            {**key, "sources": {**SOURCES, "pa_decode.py": "b" * 64}},
        ):
            self.assertIsNone(tuning.lookup_budget(data, bad_key))
        for alteration in (
            "baseline",
            "selection",
            "capacity",
            "aliases",
            "nonfinite",
            "status",
            "sample",
            "bytes",
            "duplicate",
            "key_sample",
            "metadata_override",
            "partial_accuracy",
        ):
            bad = copy.deepcopy(data)
            record = bad["records"][0]
            if alteration == "baseline":
                next(c for c in record["candidates"] if c["is_baseline"])[
                    "status"
                ] = "FAIL"
            elif alteration == "selection":
                record["selection"]["conservative_budget"] = 1024
            elif alteration == "capacity":
                record["candidates"][0]["capacity"] += 1
            elif alteration == "aliases":
                record["candidates"][0]["budget_aliases"] = [1]
            elif alteration == "nonfinite":
                record["candidates"][0]["samples_us"][0] = math.nan
            elif alteration == "status":
                record["status"] = "SOURCE_CHANGED"
            elif alteration == "sample":
                record["benchmark_input"]["lengths"][0] += 1
            elif alteration == "bytes":
                record["unique_kv_bytes"] += 1
            elif alteration == "duplicate":
                bad["records"].append(copy.deepcopy(record))
            elif alteration == "metadata_override":
                record["benchmark_input"]["query_length"] = 1
            elif alteration == "partial_accuracy":
                for candidate in record["candidates"]:
                    for stage in tuning.ACCURACY_STAGES:
                        candidate["accuracy"][stage]["elements"] = 1
            else:
                record["key"]["shape"]["seed"] = 1
                record["key_sha256"] = tuning.fingerprint(record["key"])
            with self.subTest(alteration=alteration):
                self.assertIsNone(tuning.lookup_budget(bad, key))

    def test_load_rejects_invalid_schema_keys_nan_and_duplicate_static_keys(self):
        data, _, _ = result_fixture()
        duplicate = copy.deepcopy(data)
        duplicate["records"].append(copy.deepcopy(duplicate["records"][0]))
        duplicate["records"][1]["benchmark_input"]["seed"] = 99
        for value in (
            [],
            None,
            {},
            {"schema_version": 99, "records": []},
            {"schema_version": 1, "records": [None]},
            {**data, "invalid_metric": math.inf},
            duplicate,
        ):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.json"
                path.write_text(json.dumps(value))
                with self.subTest(value=value), self.assertRaises(ValueError):
                    tuning.load_results(path)

    def test_full_fp32_flags_restore_even_on_exception(self):
        torch = types.SimpleNamespace(
            backends=types.SimpleNamespace(
                cuda=types.SimpleNamespace(
                    matmul=types.SimpleNamespace(allow_tf32=False)
                ),
                cudnn=types.SimpleNamespace(allow_tf32=True),
            )
        )
        precision = ["high"]
        torch.get_float32_matmul_precision = lambda: precision[0]

        def set_precision(value):
            precision[0] = value
            torch.backends.cuda.matmul.allow_tf32 = value != "highest"

        torch.set_float32_matmul_precision = set_precision
        with self.assertRaisesRegex(RuntimeError, "sentinel"), tuning._full_fp32(torch):
            self.assertEqual(precision[0], "highest")
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            raise RuntimeError("sentinel")
        self.assertEqual(precision[0], "high")
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(torch.backends.cudnn.allow_tf32)

    def test_kernel_helper_edits_invalidate_saved_budget(self):
        kernel_sources = {
            str(path.relative_to(MODULE.parent))
            for path in (MODULE.parent / "kernels/pa_decode").rglob("*.py")
        }
        helpers = kernel_sources | {
            "kernels/buffer_ops.py",
            "kernels/dpp_utils.py",
            "kernels/kernels_common.py",
            "kernels/tensor_shim.py",
            "kernels/utils.py",
        }
        self.assertTrue(kernel_sources)
        self.assertLessEqual(helpers, set(tuning.SOURCE_FILES))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in tuning.SOURCE_FILES:
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((MODULE.parent / name).read_bytes())
            with patch.object(tuning, "__file__", str(root / MODULE.name)):
                data, shape, _ = result_fixture()
                key = tuning.make_key(shape, "gfx950", 256)
                record = data["records"][0]
                record.update(
                    key=key,
                    key_sha256=tuning.fingerprint(key),
                    sources_after=key["sources"],
                )
                result_path = root / "budgets.json"
                tuning.save_results(result_path, data)
                saved = tuning.load_results(result_path)
                expected = record["selection"]["conservative_budget"]
                self.assertEqual(tuning.lookup_budget(saved, key), expected)

                for name in sorted(helpers):
                    with self.subTest(source=name):
                        path = root / name
                        original = path.read_bytes()
                        try:
                            path.write_bytes(original + b"\n# Changed kernel helper.\n")
                            changed_key = tuning.make_key(shape, "gfx950", 256)
                            changed_sources = {
                                source
                                for source, digest in changed_key["sources"].items()
                                if digest != key["sources"][source]
                            }
                            self.assertEqual(changed_sources, {name})
                            self.assertIsNone(tuning.lookup_budget(saved, changed_key))
                        finally:
                            path.write_bytes(original)
                self.assertEqual(tuning.make_key(shape, "gfx950", 256), key)
                self.assertEqual(tuning.lookup_budget(saved, key), expected)

    def test_local_backend_path_verification(self):
        root = MODULE.parent
        pa = types.SimpleNamespace(__file__=str(root / "pa_decode.py"))

        def local_module(name):
            relative = name.removeprefix("aiter.ops.flydsl.").replace(".", "/")
            path = root / relative
            source = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
            return types.SimpleNamespace(__file__=str(source))

        with patch.object(
            tuning.importlib, "import_module", side_effect=local_module
        ) as imports:
            self.assertEqual(
                tuning._backend_paths(pa),
                {name: str(root / name) for name in tuning.SOURCE_FILES},
            )
            imports.assert_any_call("aiter.ops.flydsl.kernels.pa_decode")
            self.assertTrue(
                all(
                    not call.args[0].endswith(".__init__")
                    for call in imports.call_args_list
                )
            )

        for source in (
            "kernels/pa_decode_plan.py",
            "kernels/pa_decode/__init__.py",
            "kernels/pa_decode/op_softmax.py",
            "kernels/tensor_shim.py",
        ):
            with self.subTest(source=source):

                def relocated_module(name, wrong_source=source):
                    module = local_module(name)
                    if module.__file__ == str(root / wrong_source):
                        module.__file__ = str(Path("/another/tree") / wrong_source)
                    return module

                with patch.object(
                    tuning.importlib, "import_module", side_effect=relocated_module
                ), self.assertRaisesRegex(RuntimeError, "source tree") as error:
                    tuning._backend_paths(pa)
                self.assertIn(source, str(error.exception))

    def run_cli(self, *arguments):
        code = f"""
import importlib.abc, runpy, sys
class NoGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'triton', 'flydsl', 'aiter'):
            raise AssertionError('GPU/runtime import: ' + fullname)
sys.meta_path.insert(0, NoGPUImports())
sys.argv = [{str(MODULE)!r}, *sys.argv[1:]]
runpy.run_path({str(MODULE)!r}, run_name='__main__')
"""
        with tempfile.TemporaryDirectory() as directory:
            return subprocess.run(
                [sys.executable, "-B", "-I", "-c", code, *arguments],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )

    def test_cli_help_and_models_need_no_runtime(self):
        help_result = self.run_cli("--help")
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--budget-cu", help_result.stdout)
        result = self.run_cli("--list-models")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["gemma3-4b"]["D"], 256)

    def test_cli_dry_run_matrix_has_static_keys_and_sample_metadata(self):
        result = self.run_cli(
            "--shape",
            "8,8,128",
            "--batch-sizes",
            "4",
            "--context-lengths",
            "129",
            "--query-lengths",
            "1,4",
            "--windows",
            "0,32",
            "--page-sizes",
            "16,128",
            "--dtypes",
            "bfloat16,float16",
            "--quant-modes",
            "per_token,per_tensor",
            "--trans-v",
            "both",
            "--length-mode",
            "varlen",
            "--seed",
            "7",
            "--budget-cu",
            "0.5,1,2,4,8,16",
            "--architecture",
            "gfx950",
            "--num-cu",
            "256",
            "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        shapes = json.loads(result.stdout)["shapes"]
        self.assertEqual(len(shapes), 64)
        for item in shapes:
            self.assertEqual(set(item["key"]["shape"]), set(tuning.STATIC_SHAPE_FIELDS))
            self.assertEqual(item["benchmark_input"]["seed"], 7)
            self.assertEqual(sum(c["is_baseline"] for c in item["candidates"]), 1)
            self.assertEqual(
                sorted(b for c in item["candidates"] for b in c["budget_aliases"]),
                [128, 256, 512, 1024, 2048, 4096],
            )

    def test_cli_invalid_inputs_fail_before_runtime_import(self):
        for arguments in (
            ("--dry-run",),
            ("--dry-run", "--architecture", "gfx950", "--num-cu", "0"),
            (
                "--dry-run",
                "--architecture",
                "gfx950",
                "--num-cu",
                "256",
                "--budget-cu",
                "nan",
            ),
            (
                "--dry-run",
                "--architecture",
                "gfx950",
                "--num-cu",
                "256",
                "--budget-cu",
                "0.1",
            ),
            ("--shape", "8,8", "--dry-run"),
            ("--budgets", "0", "--dry-run"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_cli(*arguments)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("GPU/runtime import", result.stderr)


if __name__ == "__main__":
    unittest.main()
