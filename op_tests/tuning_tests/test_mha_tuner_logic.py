# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only policy and enumeration tests for the MHA forward tuner."""

import argparse
import collections
import csv
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import triton  # noqa: F401  # isort: skip  # Must precede torch on this ROCm environment.
import torch

from aiter.jit.utils.chip_info import normalize_gpu_model
from aiter.ops import mha
from aiter.ops.mha_fwd_policy import (
    MHA_FWD_RUNTIME_CSV_FIELDS,
    MHA_FWD_TILE_CONFIG_BACKENDS,
    MHA_FWD_TILE_CONFIG_KEYS,
    MhaFwdCandidate,
    MhaFwdPlan,
    MhaFwdProblem,
    MhaFwdResult,
    enumerate_mha_fwd_candidates,
    mha_fwd_candidate_id,
)

_TUNER_PATH = Path(__file__).parents[1] / "tuners" / "tune_mha_fwd.py"
_SPEC = importlib.util.spec_from_file_location("tune_mha_fwd", _TUNER_PATH)
_TUNER = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_TUNER)


def _problem_row():
    return {
        "gfx": "gfx942",
        "gpu_model": "mi325x",
        "cu_num": 304,
        "mode": "varlen",
        "batch": 1,
        "total_q": 4096,
        "total_k": 42700,
        "max_seqlen_q": 4096,
        "max_seqlen_k": 42700,
        "min_seqlen_q": 0,
        "nhead_q": 12,
        "nhead_k": 12,
        "hdim_q": 192,
        "hdim_v": 128,
        "dtype": "bfloat16",
        "causal": 0,
        "window_left": -1,
        "window_right": -1,
        "sink_size": 0,
        "dropout_p": 0.0,
        "logits_soft_cap": 0.0,
        "how_v3_bf16_cvt": 1,
        "return_lse": 0,
        "return_attn_probs": 0,
        "has_bias": 0,
        "has_alibi": 0,
        "has_sink": 0,
        "has_block_table": 0,
        "has_q_descale": 0,
        "has_physical_padding": 0,
        "is_grad": 0,
    }


def _dummy_varlen_tensors():
    q = torch.empty((8, 12, 192), dtype=torch.bfloat16)
    k = torch.empty((16, 12, 192), dtype=torch.bfloat16)
    v = torch.empty((16, 12, 128), dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 8], dtype=torch.int32)
    cu_k = torch.tensor([0, 16], dtype=torch.int32)
    return q, k, v, cu_q, cu_k


class TestMhaHardwareIdentity(unittest.TestCase):
    def test_gpu_model_normalization_distinguishes_gfx942_products(self):
        self.assertEqual(normalize_gpu_model("AMD Instinct MI300X"), "mi300x")
        self.assertEqual(normalize_gpu_model("AMD Instinct MI325X"), "mi325x")

    def test_problem_key_starts_with_arch_model_and_cu(self):
        problem = MhaFwdProblem.from_mapping(_problem_row())
        self.assertEqual(problem.key()[:3], ("gfx942", "mi325x", "304"))

    def test_bf16_spelling_normalizes_to_runtime_dtype(self):
        row = _problem_row()
        row["dtype"] = "bf16"
        self.assertEqual(MhaFwdProblem.from_mapping(row).dtype, "bfloat16")


class TestMhaTypedPolicy(unittest.TestCase):
    def test_candidate_identity_uses_canonical_config_json(self):
        candidate = MhaFwdCandidate(
            "triton", backend_config={"num_warps": 4, "BLOCK_N": 64}
        )
        self.assertEqual(
            candidate.identity,
            ("triton", 0, '{"BLOCK_N":64,"num_warps":4}'),
        )

    def test_only_asm_accepts_external_split_count(self):
        self.assertEqual(MhaFwdCandidate("asm_v3", 3).identity[:2], ("asm_v3", 3))
        with self.assertRaises(ValueError):
            MhaFwdCandidate("triton", 3)

    def test_result_uses_median_of_positive_samples(self):
        result = MhaFwdResult(
            MhaFwdProblem.from_mapping(_problem_row()),
            MhaFwdCandidate("asm_v3", 3),
            "ok",
            0.0,
            (2.3, 2.1, 2.2),
        )
        self.assertEqual(result.median_us, 2.2)

    def test_candidate_id_is_stable_and_problem_specific(self):
        problem = MhaFwdProblem.from_mapping(_problem_row())
        candidate = MhaFwdCandidate("asm_v3", 3)
        self.assertEqual(
            mha_fwd_candidate_id(problem, candidate),
            mha_fwd_candidate_id(problem, candidate),
        )
        self.assertNotEqual(
            mha_fwd_candidate_id(problem, candidate),
            mha_fwd_candidate_id(problem, MhaFwdCandidate("asm_v3", 4)),
        )

    def test_runtime_plan_accepts_every_legal_backend(self):
        self.assertEqual(MhaFwdPlan("ck").backend, "ck")
        self.assertEqual(MhaFwdPlan("triton").backend, "triton")
        self.assertEqual(
            MhaFwdPlan("gluon", backend_config={"BLOCK_M": 64}).backend, "gluon"
        )
        self.assertEqual(MhaFwdPlan("flydsl").backend, "flydsl")
        self.assertEqual(MhaFwdPlan("opus").backend, "opus")
        self.assertEqual(MhaFwdPlan("asm_v3", 3).num_splits, 3)
        with self.assertRaises(ValueError):
            MhaFwdPlan("asm_v3", 0)
        with self.assertRaises(ValueError):
            MhaFwdPlan("ck", backend_config={"BLOCK_M": 64})


class TestMhaProblemBuckets(unittest.TestCase):
    def test_balanced_lengths_preserve_runtime_key_summary(self):
        lengths = _TUNER._balanced_lengths(12, 3, 5)
        self.assertEqual(sum(lengths), 12)
        self.assertEqual(max(lengths), 5)
        self.assertEqual(len(lengths), 3)

    def test_impossible_summary_is_rejected(self):
        with self.assertRaises(ValueError):
            _TUNER._balanced_lengths(16, 3, 5)


class TestMhaCandidateEnumeration(unittest.TestCase):
    def test_gfx942_enumerates_every_split_and_triton_grid(self):
        candidates = enumerate_mha_fwd_candidates("gfx942")
        splits = [
            candidate.num_splits
            for candidate in candidates
            if candidate.backend == "asm_v3"
        ]
        self.assertEqual(splits, list(range(1, 9)))
        self.assertIn("ck", [candidate.backend for candidate in candidates])
        self.assertGreater(
            sum(candidate.backend == "triton" for candidate in candidates), 1
        )

    def test_candidate_identities_are_unique(self):
        for gfx in ("gfx942", "gfx950", "gfx1250"):
            identities = [
                candidate.identity
                for candidate in enumerate_mha_fwd_candidates(gfx)
            ]
            self.assertEqual(len(identities), len(set(identities)))


class TestTileConfigVocabulary(unittest.TestCase):
    def test_tile_keys_match_enumeration(self):
        seen = set()
        for gfx in ("gfx942", "gfx950", "gfx1250"):
            emitted = collections.defaultdict(set)
            for candidate in enumerate_mha_fwd_candidates(gfx):
                if candidate.backend_config:
                    emitted[candidate.backend].update(candidate.backend_config)
            seen.update(emitted)
            for backend, keys in emitted.items():
                with self.subTest(gfx=gfx, backend=backend):
                    self.assertEqual(keys, set(MHA_FWD_TILE_CONFIG_KEYS[backend]))
        self.assertEqual(seen, set(MHA_FWD_TILE_CONFIG_KEYS))
        self.assertEqual(seen, set(MHA_FWD_TILE_CONFIG_BACKENDS))

    def test_unknown_tile_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            MhaFwdPlan(backend="triton", backend_config={"BLOCK_MM": 128})

    def test_gluon_rejects_triton_only_key(self):
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            MhaFwdPlan(backend="gluon", backend_config={"num_stages": 2})

    def test_legal_tile_config_loads(self):
        plan = MhaFwdPlan(backend="triton", backend_config={"BLOCK_M": 128})
        self.assertEqual(plan.backend_config, {"BLOCK_M": 128})

    def test_non_tile_backend_rejects_config(self):
        with self.assertRaisesRegex(ValueError, "does not accept backend_config"):
            MhaFwdPlan(backend="opus", backend_config={"BLOCK_M": 128})


class TestMhaTunedPolicy(unittest.TestCase):
    def _key_args(self):
        q = torch.empty((4096, 12, 192), dtype=torch.bfloat16)
        k = torch.empty((42700, 12, 192), dtype=torch.bfloat16)
        v = torch.empty((42700, 12, 128), dtype=torch.bfloat16)
        return {
            "mode": "varlen",
            "q": q,
            "k": k,
            "v": v,
            "batch": 1,
            "max_seqlen_q": 4096,
            "max_seqlen_k": 42700,
            "min_seqlen_q": 0,
            "causal": False,
            "window_size": (-1, -1, 0),
            "dropout_p": 0.0,
            "logits_soft_cap": 0.0,
            "how_v3_bf16_cvt": 1,
            "return_lse": False,
            "return_attn_probs": False,
            "bias": None,
            "alibi_slopes": None,
            "sink_ptr": None,
            "block_table": None,
            "q_descale": None,
            "cu_seqlens_q_padded": None,
            "cu_seqlens_k_padded": None,
        }

    def test_seeded_kimi_row_is_exact(self):
        config = Path(mha.__file__).parents[1] / "configs" / "tuned_mha_fwd.csv"
        table = mha._load_mha_fwd_tuning_table(os.fspath(config))
        with (
            mock.patch.object(mha, "get_gfx_runtime", return_value="gfx942"),
            mock.patch.object(mha, "get_gpu_model", return_value="mi325x"),
            mock.patch.object(
                mha.torch.cuda,
                "get_device_properties",
                return_value=mock.Mock(multi_processor_count=304),
            ),
        ):
            key = mha._mha_fwd_tuning_key(**self._key_args())
        self.assertEqual(table[key]["backend"], "asm_v3")
        self.assertEqual(table[key]["num_splits"], 3)
        self.assertIsNone(table[key]["backend_config"])

    def test_lookup_returns_tiles_only_for_the_winning_backend(self):
        tiles = {"BLOCK_M": 64, "BLOCK_N": 32}
        plan = {"backend": "triton", "num_splits": 0, "backend_config": tiles}
        with mock.patch.object(mha, "_get_mha_fwd_tuned_plan", return_value=plan):
            self.assertEqual(
                mha.lookup_mha_fwd_tile_config("triton", **self._key_args()),
                tiles,
            )
            self.assertIsNone(
                mha.lookup_mha_fwd_tile_config("gluon", **self._key_args())
            )
        with mock.patch.object(mha, "_get_mha_fwd_tuned_plan", return_value=None):
            self.assertIsNone(
                mha.lookup_mha_fwd_tile_config("triton", **self._key_args())
            )

    def test_measurement_rows_are_rejected_as_runtime_artifacts(self):
        fields = [
            *mha.MHA_FWD_TUNING_KEY_FIELDS,
            "backend",
            "num_splits",
            "backend_config",
            "status",
        ]
        values = {field: "0" for field in fields}
        values.update(
            {
                "gfx": "gfx942",
                "cu_num": "304",
                "mode": "varlen",
                "backend": "asm_v3",
                "num_splits": "3",
                "backend_config": "",
                "status": "failed",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "failed.csv")
            with open(path, "w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerow(values)
            with self.assertRaisesRegex(ValueError, "non-runtime MHA columns"):
                mha._load_mha_fwd_tuning_table(path)

    def test_different_gpu_model_does_not_match(self):
        config = Path(mha.__file__).parents[1] / "configs" / "tuned_mha_fwd.csv"
        table = mha._load_mha_fwd_tuning_table(os.fspath(config))
        with (
            mock.patch.object(mha, "get_gfx_runtime", return_value="gfx942"),
            mock.patch.object(mha, "get_gpu_model", return_value="mi300x"),
            mock.patch.object(
                mha.torch.cuda,
                "get_device_properties",
                return_value=mock.Mock(multi_processor_count=304),
            ),
        ):
            key = mha._mha_fwd_tuning_key(**self._key_args())
        self.assertNotIn(key, table)

    def test_runtime_csv_has_no_measurement_columns(self):
        config = Path(mha.__file__).parents[1] / "configs" / "tuned_mha_fwd.csv"
        with config.open(encoding="utf-8", newline="") as file:
            fields = tuple(csv.DictReader(file).fieldnames or ())
        self.assertEqual(fields, MHA_FWD_RUNTIME_CSV_FIELDS)
        self.assertIn("backend_config", fields)
        self.assertNotIn("us", fields)
        self.assertNotIn("status", fields)


class TestMhaWinnerPromotion(unittest.TestCase):
    def _result(self, backend, us, err_ratio=0.0, num_splits=0, config=""):
        problem = MhaFwdProblem.from_mapping(_problem_row())
        return (
            (problem.key(), backend, num_splits, config),
            us,
            err_ratio,
            "ok",
        )

    def test_faster_triton_row_is_promoted(self):
        tuner = _TUNER.MhaFwdTuner()
        args = argparse.Namespace(profile_file="", errRatio=0.0)
        config = '{"BLOCK_M":64,"BLOCK_N":64}'
        winners = tuner.post_process(
            [
                self._result("asm_v3", 3.0, num_splits=3),
                self._result("triton", 1.5, config=config),
                self._result("ck", 1.0, err_ratio=0.1),
            ],
            args,
        )
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners.iloc[0]["backend"], "triton")
        self.assertEqual(winners.iloc[0]["backend_config"], config)
        self.assertEqual(winners.iloc[0]["num_splits"], 0)

    def test_faster_ck_row_is_promoted(self):
        tuner = _TUNER.MhaFwdTuner()
        args = argparse.Namespace(profile_file="", errRatio=0.0)
        winners = tuner.post_process(
            [
                self._result("asm_v3", 2.2, num_splits=3),
                self._result("ck", 1.8),
            ],
            args,
        )
        self.assertEqual(list(winners["backend"]), ["ck"])

    def test_runtime_csv_keeps_backend_config_and_drops_metrics(self):
        tuner = _TUNER.MhaFwdTuner()
        problem = MhaFwdProblem.from_mapping(_problem_row())
        config = '{"BLOCK_M":64}'
        row = {
            **problem.as_row(),
            "backend": "triton",
            "num_splits": 0,
            "backend_config": config,
            "us": 1.5,
            "errRatio": 0.0,
            "status": "ok",
            "detail": "",
            "samples_us": "[1.5]",
            "tflops": 1.0,
        }
        result = _TUNER.pd.DataFrame([row], columns=tuner.columns)
        with tempfile.TemporaryDirectory() as directory:
            runtime = os.path.join(directory, "runtime.csv")
            tuner._journal_path = os.path.join(directory, "journal.jsonl")
            tuner._evidence_path = os.path.join(directory, "evidence.json")
            tuner._args = argparse.Namespace(
                warmup=1,
                iters=2,
                finalist_rounds=1,
                strategy="exhaustive",
                errRatio=0.0,
                untune_file="catalogue.csv",
                profile_file="measurements.csv",
            )
            tuner._run_started_at = 1.0
            tuner.untunedf = _TUNER.pd.DataFrame([problem.as_row()])
            tuner.success = result.copy()
            tuner._all_results = result.copy()
            with mock.patch.object(
                tuner,
                "_run_fresh_probe",
                return_value={"status": "verified"},
            ):
                tuner.result_to_csv(result, runtime)
            with open(runtime, encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                fields = tuple(reader.fieldnames or ())
                written = next(reader)
        self.assertEqual(fields, MHA_FWD_RUNTIME_CSV_FIELDS)
        self.assertNotIn("us", fields)
        self.assertNotIn("status", fields)
        self.assertEqual(written["backend"], "triton")
        self.assertEqual(written["backend_config"], config)


class TestMhaPublicDispatch(unittest.TestCase):
    def test_csv_asm_plan_launches_splitkv_operator(self):
        q, k, v, cu_q, cu_k = _dummy_varlen_tensors()
        sentinel = (
            torch.empty(1),
            torch.empty(1),
            torch.empty(1),
            torch.empty(1),
        )
        with (
            mock.patch.object(mha, "get_gfx", return_value="gfx942"),
            mock.patch.object(
                mha, "_fmha_v3_varlen_splitkv_fwd", return_value=sentinel
            ) as splitkv,
            mock.patch.object(mha, "fmha_v3_varlen_fwd") as auto,
        ):
            mha._flash_attn_varlen_forward(
                q,
                k,
                v,
                cu_q,
                cu_k,
                None,
                None,
                8,
                16,
                0,
                0.0,
                0.125,
                False,
                num_splits=3,
                selected_backend="asm_v3",
            )
        splitkv.assert_called_once()
        auto.assert_not_called()
        self.assertEqual(splitkv.call_args.args[-1], 3)

    def test_no_plan_uses_public_asm_auto_select(self):
        q, k, v, cu_q, cu_k = _dummy_varlen_tensors()
        sentinel = (
            torch.empty(1),
            torch.empty(1),
            torch.empty(1),
            torch.empty(1),
        )
        with (
            mock.patch.object(mha, "get_gfx", return_value="gfx942"),
            mock.patch.object(mha, "_fmha_v3_varlen_splitkv_fwd") as splitkv,
            mock.patch.object(mha, "fmha_v3_varlen_fwd", return_value=sentinel) as auto,
        ):
            mha._flash_attn_varlen_forward(
                q,
                k,
                v,
                cu_q,
                cu_k,
                None,
                None,
                8,
                16,
                0,
                0.0,
                0.125,
                False,
                num_splits=0,
                selected_backend=None,
            )
        auto.assert_called_once()
        splitkv.assert_not_called()

    def test_flash_attn_varlen_func_routes_csv_backends(self):
        q, k, v, cu_q, cu_k = _dummy_varlen_tensors()
        plan = {
            "backend": "asm_v3",
            "num_splits": 3,
            "backend_config": None,
        }
        with (
            mock.patch.object(mha, "_get_mha_fwd_tuned_plan", return_value=plan),
            mock.patch.object(
                mha.FlashAttnVarlenFunc, "apply", return_value="asm"
            ) as apply,
            mock.patch(
                "aiter.ops.flydsl.fmha_kernels.flydsl_flash_attn_varlen_func",
                return_value="flydsl",
            ),
            mock.patch(
                "aiter.ops.triton.attention.mha.flash_attn_varlen_func",
                return_value="triton",
            ) as triton_entry,
        ):
            result = mha.flash_attn_varlen_func(q, k, v, cu_q, cu_k, 8, 16)
        self.assertEqual(result, "asm")
        apply.assert_called_once()
        self.assertEqual(apply.call_args.args[-3:], (3, "asm_v3", None))
        triton_entry.assert_not_called()

        triton_plan = {
            "backend": "triton",
            "num_splits": 0,
            "backend_config": {"BLOCK_M": 64},
        }
        with (
            mock.patch.object(mha, "_get_mha_fwd_tuned_plan", return_value=triton_plan),
            mock.patch.object(mha.FlashAttnVarlenFunc, "apply") as apply,
            mock.patch(
                "aiter.ops.flydsl.fmha_kernels.flydsl_flash_attn_varlen_func",
                return_value="flydsl",
            ),
            mock.patch(
                "aiter.ops.triton.attention.mha.flash_attn_varlen_func",
                return_value="triton",
            ) as triton_entry,
        ):
            result = mha.flash_attn_varlen_func(q, k, v, cu_q, cu_k, 8, 16)
        self.assertEqual(result, "triton")
        apply.assert_not_called()
        self.assertEqual(triton_entry.call_args.kwargs["backend"], "triton")
        self.assertEqual(triton_entry.call_args.kwargs["config"], {"BLOCK_M": 64})

    def test_triton_public_varlen_reads_csv_tiles_when_config_is_none(self):
        from aiter.ops.triton.attention import mha as triton_mha

        tiles = {
            "BLOCK_M": 64,
            "BLOCK_N": 32,
            "PRELOAD_V": False,
            "num_warps": 4,
            "waves_per_eu": 2,
            "num_stages": 1,
            "num_ctas": 1,
        }
        q, k, v, cu_q, cu_k = _dummy_varlen_tensors()
        with (
            mock.patch.object(
                mha, "lookup_mha_fwd_tile_config", return_value=tiles
            ) as lookup,
            mock.patch.object(
                triton_mha._FlashAttnVarlenFunc, "apply", return_value="ok"
            ) as apply,
        ):
            result = triton_mha.flash_attn_varlen_func(
                q, k, v, cu_q, cu_k, 8, 16, config=None, backend="triton"
            )
        self.assertEqual(result, "ok")
        lookup.assert_called_once()
        self.assertEqual(lookup.call_args.args[0], "triton")
        self.assertEqual(lookup.call_args.kwargs["mode"], "varlen")
        self.assertEqual(apply.call_args.args[-1], tiles)

    def test_explicit_triton_config_skips_csv_lookup(self):
        from aiter.ops.triton.attention import mha as triton_mha

        q, k, v, cu_q, cu_k = _dummy_varlen_tensors()
        explicit = {"BLOCK_M": 16}
        with (
            mock.patch.object(mha, "lookup_mha_fwd_tile_config") as lookup,
            mock.patch.object(
                triton_mha._FlashAttnVarlenFunc, "apply", return_value="ok"
            ) as apply,
        ):
            result = triton_mha.flash_attn_varlen_func(
                q, k, v, cu_q, cu_k, 8, 16, config=explicit, backend="triton"
            )
        self.assertEqual(result, "ok")
        lookup.assert_not_called()
        self.assertEqual(apply.call_args.args[-1], explicit)

    def test_unknown_backend_fails_closed(self):
        q, k, v, cu_q, cu_k = _dummy_varlen_tensors()
        with (
            mock.patch.object(
                mha,
                "_get_mha_fwd_tuned_plan",
                return_value={"backend": "mystery", "num_splits": 0},
            ),
            self.assertRaisesRegex(ValueError, "unknown tuned MHA backend"),
        ):
            mha.flash_attn_varlen_func(q, k, v, cu_q, cu_k, 8, 16)


class TestMhaCheckpointJournal(unittest.TestCase):
    def test_journal_round_trip_and_truncated_tail(self):
        tuner = _TUNER.MhaFwdTuner()
        problem = MhaFwdProblem.from_mapping(_problem_row())
        candidate = MhaFwdCandidate("asm_v3", 3)
        info = (problem.key(), *candidate.identity)
        with tempfile.TemporaryDirectory() as directory:
            tuner._journal_path = os.path.join(directory, "run.jsonl")
            tuner._args = argparse.Namespace(resume=True)
            tuner._append_journal_result("first", (info, 2.1, 0.0, "ok"))
            tuner._append_journal_result(
                "finalist:0",
                (info, float("inf"), 1.0, "timeout", "exceeded 60s after 61.2s"),
            )
            with open(tuner._journal_path, "a", encoding="utf-8") as file:
                file.write('{"incomplete":')
            records = tuner._load_journal()
        key = (mha_fwd_candidate_id(problem, candidate), "first")
        self.assertEqual(records[key][1:], (2.1, 0.0, "ok", ""))
        failed_key = (mha_fwd_candidate_id(problem, candidate), "finalist:0")
        self.assertEqual(
            records[failed_key][1:],
            (float("inf"), 1.0, "timeout", "exceeded 60s after 61.2s"),
        )

    def test_journal_reads_records_written_before_details_existed(self):
        tuner = _TUNER.MhaFwdTuner()
        problem = MhaFwdProblem.from_mapping(_problem_row())
        candidate = MhaFwdCandidate("asm_v3", 3)
        legacy = {
            "schema_version": 1,
            "candidate_id": mha_fwd_candidate_id(problem, candidate),
            "phase": "first",
            "problem": problem.as_row(),
            "candidate": {
                "backend": candidate.backend,
                "num_splits": candidate.num_splits,
                "backend_config": "",
            },
            "status": "crash",
            "us": None,
            "errRatio": 1.0,
            "recorded_at_unix_s": 0.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            tuner._journal_path = os.path.join(directory, "run.jsonl")
            tuner._args = argparse.Namespace(resume=True)
            with open(tuner._journal_path, "w", encoding="utf-8") as file:
                file.write(json.dumps(legacy) + "\n")
            records = tuner._load_journal()
        key = (mha_fwd_candidate_id(problem, candidate), "first")
        self.assertEqual(records[key][1:], (float("inf"), 1.0, "crash", ""))

    def test_fresh_probe_invokes_module_from_repo_root(self):
        tuner = _TUNER.MhaFwdTuner()
        tuner._args = argparse.Namespace(warmup=1, iters=2, timeout=5)
        row = _TUNER.pd.Series(
            {
                **_problem_row(),
                "backend": "asm_v3",
                "num_splits": 3,
                "backend_config": "",
            }
        )
        completed = mock.Mock(returncode=1, stdout="", stderr="boom")
        with mock.patch.object(_TUNER.subprocess, "run", return_value=completed) as run:
            proof = tuner._run_fresh_probe(row, "/tmp/runtime.csv")
        command = run.call_args.args[0]
        self.assertEqual(
            command[1:4],
            ["-m", "op_tests.tuners.tune_mha_fwd", "--_selection_probe"],
        )
        repository_root = Path(_TUNER.__file__).resolve().parents[2]
        self.assertEqual(run.call_args.kwargs["cwd"], str(repository_root))
        self.assertTrue(
            run.call_args.kwargs["env"]["PYTHONPATH"].startswith(str(repository_root))
        )
        self.assertEqual(proof["status"], "failed")
        self.assertEqual(proof["expected"]["backend_config"], "")

    def test_selection_trace_is_append_only_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "proof.jsonl")
            with mock.patch.dict(
                os.environ, {"AITER_MHA_FWD_SELECTION_PROOF_FILE": path}
            ):
                mha._record_mha_fwd_selection("asm_v3", 3)
                mha._record_mha_fwd_selection("ck", 0)
            with open(path, encoding="utf-8") as file:
                rows = [json.loads(line) for line in file]
        self.assertEqual(
            [(row["backend"], row["num_splits"]) for row in rows],
            [("asm_v3", 3), ("ck", 0)],
        )

    def test_runtime_and_evidence_are_written_separately(self):
        tuner = _TUNER.MhaFwdTuner()
        problem = MhaFwdProblem.from_mapping(_problem_row())
        row = {
            **problem.as_row(),
            "backend": "asm_v3",
            "num_splits": 3,
            "backend_config": "",
            "us": 2.1,
            "errRatio": 0.0,
            "status": "ok",
            "detail": "",
            "samples_us": "[2.1]",
            "tflops": 1.0,
        }
        result = _TUNER.pd.DataFrame([row], columns=tuner.columns)
        with tempfile.TemporaryDirectory() as directory:
            runtime = os.path.join(directory, "runtime.csv")
            tuner._journal_path = os.path.join(directory, "journal.jsonl")
            tuner._evidence_path = os.path.join(directory, "evidence.json")
            tuner._args = argparse.Namespace(
                warmup=1,
                iters=2,
                finalist_rounds=1,
                strategy="exhaustive",
                errRatio=0.0,
                untune_file="catalogue.csv",
                profile_file="measurements.csv",
            )
            tuner._run_started_at = 1.0
            tuner.untunedf = _TUNER.pd.DataFrame([problem.as_row()])
            tuner.success = result.copy()
            tuner._all_results = result.copy()
            with mock.patch.object(
                tuner,
                "_run_fresh_probe",
                return_value={"status": "verified"},
            ):
                tuner.result_to_csv(result, runtime)
            with open(runtime, encoding="utf-8", newline="") as file:
                runtime_fields = tuple(csv.DictReader(file).fieldnames or ())
            with open(tuner._evidence_path, encoding="utf-8") as file:
                evidence = json.load(file)
        self.assertEqual(runtime_fields, MHA_FWD_RUNTIME_CSV_FIELDS)
        self.assertEqual(evidence["run_state"], "verified")
        self.assertEqual(evidence["measurement"]["candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
