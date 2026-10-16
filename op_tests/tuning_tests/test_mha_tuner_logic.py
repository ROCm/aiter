# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the MHA forward tuner as a GatedTuner family.

What the family supplies is tested here: the reference, the candidate field,
the evidence rows and the selection hooks, plus one publish through
GatedTuner. The publication rules themselves are tested against a stub
family in test_gated_tuner.py.
"""

import argparse
import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import triton  # noqa: F401  # isort: skip  # Must precede torch on this ROCm environment.
import pandas as pd
import torch

from aiter.ops.mha_fwd_policy import (
    MHA_FWD_RUNTIME_CSV_FIELDS,
    MHA_FWD_TUNER_SCRIPT,
    MHA_FWD_TUNING_KEY_FIELDS,
    MhaFwdProblem,
    config_digest,
)
from aiter.test_mha_common import attention_ref
from aiter.utility.selection_proof import SelectionProof

_TUNER_PATH = Path(__file__).parents[2] / MHA_FWD_TUNER_SCRIPT
_SPEC = importlib.util.spec_from_file_location("tune_mha_fwd", _TUNER_PATH)
_TUNER = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_TUNER)

TILE = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4}


def _problem_row(**overrides):
    return {
        "gfx": "gfx950",
        "gpu_model": "mi355x",
        "cu_num": 256,
        "mode": "varlen",
        "batch": 2,
        "total_q": 512,
        "total_k": 512,
        "max_seqlen_q": 256,
        "max_seqlen_k": 256,
        "min_seqlen_q": 0,
        "nhead_q": 8,
        "nhead_k": 8,
        "hdim_q": 128,
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
        **overrides,
    }


def _naive_attention(q, k, v, cu_q, cu_k, scale, causal, left, right, cap=0.0):
    """softmax(QK^T)V one sequence and one head at a time, masks bottom-right."""
    out = torch.zeros((*q.shape[:-1], v.shape[-1]), dtype=torch.float32)
    lse = torch.zeros((q.shape[1], q.shape[0]), dtype=torch.float32)
    groups = q.shape[1] // k.shape[1]
    for b in range(len(cu_q) - 1):
        qs, qe, ks, ke = int(cu_q[b]), int(cu_q[b + 1]), int(cu_k[b]), int(cu_k[b + 1])
        sq, sk = qe - qs, ke - ks
        row = torch.arange(sq).unsqueeze(1) + (sk - sq)
        col = torch.arange(sk).unsqueeze(0)
        allowed = torch.ones((sq, sk), dtype=torch.bool)
        if causal:
            allowed &= col <= row
        if left >= 0:
            allowed &= col >= row - left
        if right >= 0:
            allowed &= col <= row + right
        for h in range(q.shape[1]):
            scores = q[qs:qe, h].float() @ k[ks:ke, h // groups].float().T * scale
            if cap > 0:
                scores = cap * torch.tanh(scores / cap)
            scores = scores.masked_fill(~allowed, float("-inf"))
            out[qs:qe, h] = (
                torch.softmax(scores, -1).nan_to_num(0.0)
                @ v[ks:ke, h // groups].float()
            )
            lse[h, qs:qe] = torch.logsumexp(scores, -1)
    return out, lse


class TestChunkedReference(unittest.TestCase):
    """The reference every candidate is judged against."""

    def check(self, *, q_lengths, k_lengths, nhead_q=4, nhead_k=4, **mask):
        torch.manual_seed(0)
        total_q, total_k = sum(q_lengths), sum(k_lengths)
        q = torch.randn(total_q, nhead_q, 32)
        k = torch.randn(total_k, nhead_k, 32)
        v = torch.randn(total_k, nhead_k, 16)
        cu_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()])
        cu_k = torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()])
        causal = mask.get("causal", False)
        left = mask.get("left", -1)
        right = mask.get("right", -1)
        cap = mask.get("cap", 0.0)
        out, lse = _TUNER._chunked_reference(
            q, k, v, cu_q, cu_k, 0.125, causal, left, right, True, cap
        )
        want_out, want_lse = _naive_attention(
            q, k, v, cu_q, cu_k, 0.125, causal, left, right, cap
        )
        torch.testing.assert_close(out, want_out, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(lse, want_lse, rtol=1e-4, atol=1e-4)
        # And against an oracle written outside this tuner, one sequence at a
        # time. attention_ref scales by 1/sqrt(head_dim), so q is rescaled.
        rescale = 0.125 * math.sqrt(q.shape[-1])
        for b in range(len(q_lengths)):
            qs, qe = int(cu_q[b]), int(cu_q[b + 1])
            ks, ke = int(cu_k[b]), int(cu_k[b + 1])
            want, _, _ = attention_ref(
                q[None, qs:qe] * rescale,
                k[None, ks:ke],
                v[None, ks:ke],
                causal=causal,
                window_size=(left, right),
                softcap=cap,
            )
            torch.testing.assert_close(out[qs:qe], want[0], rtol=1e-4, atol=1e-4)

    def test_plain_varlen(self):
        self.check(q_lengths=[7, 3], k_lengths=[7, 3])

    def test_logits_soft_cap(self):
        self.check(q_lengths=[7, 3], k_lengths=[7, 3], cap=2.0)

    def test_causal_soft_cap_against_attention_ref(self):
        self.check(q_lengths=[6, 9], k_lengths=[6, 9], causal=True, cap=1.5)

    def test_a_reference_split_into_many_chunks_is_unchanged(self):
        # Chunks of one or two query rows, so every boundary is exercised.
        for budget in (4, 4 * 4 * 11 * 2):
            with (
                self.subTest(budget=budget),
                mock.patch.object(_TUNER, "REFERENCE_CHUNK_BYTES", budget),
            ):
                self.check(q_lengths=[3, 4], k_lengths=[11, 6], causal=True)
                self.check(q_lengths=[12, 5], k_lengths=[12, 5], left=3, right=1)

    def test_the_score_budget_sets_the_chunk_size(self):
        q, k, v = torch.randn(7, 4, 32), torch.randn(17, 4, 32), torch.randn(17, 4, 16)
        cu_q, cu_k = torch.tensor([0, 3, 7]), torch.tensor([0, 11, 17])
        with (
            mock.patch.object(_TUNER, "REFERENCE_CHUNK_BYTES", 4),
            mock.patch.object(torch, "einsum", wraps=torch.einsum) as einsum,
        ):
            _TUNER._chunked_reference(q, k, v, cu_q, cu_k, 0.125, False, -1, -1, False)
        score_products = [
            c for c in einsum.call_args_list if c.args[0] == "qhgd,khd->hgqk"
        ]
        self.assertEqual(len(score_products), 7)  # one query row per chunk

    def test_grouped_query_attention(self):
        self.check(q_lengths=[5, 9], k_lengths=[5, 9], nhead_q=8, nhead_k=2)

    def test_causal_with_more_keys_than_queries(self):
        self.check(q_lengths=[3, 4], k_lengths=[11, 6], causal=True)

    def test_causal_with_more_queries_than_keys(self):
        # Rows above the diagonal see no key; they come out as zeros.
        self.check(q_lengths=[9], k_lengths=[4], causal=True)

    def test_sliding_window(self):
        self.check(q_lengths=[12, 5], k_lengths=[12, 5], left=3, right=1)


def _tuner(backends=""):
    tuner = _TUNER.MhaFwdTuner()
    tuner._args = argparse.Namespace(backends=backends)
    return tuner


class TestCandidateField(unittest.TestCase):
    def field(self, row, incumbents=(), backends=""):
        tuner = _tuner(backends)
        with (
            mock.patch.object(
                tuner, "incumbents", return_value=[{"backend": b} for b in incumbents]
            ),
            mock.patch(
                "aiter.ops.triton.attention.mha.is_gluon_available", return_value=True
            ),
            mock.patch("builtins.print"),
        ):
            return tuner.candidate_field(pd.Series(row))

    def test_every_backend_that_exists_on_gfx950_is_measured(self):
        self.assertEqual(
            self.field(_problem_row()),
            ["asm_v3", "ck", "flydsl", "gluon", "opus", "triton"],
        )

    def test_gfx942_has_no_opus_or_gluon(self):
        self.assertEqual(
            self.field(_problem_row(gfx="gfx942", gpu_model="mi325x", cu_num=304)),
            ["asm_v3", "ck", "flydsl", "triton"],
        )

    def test_triton_is_not_measured_where_it_cannot_stand_in_for_ck(self):
        field = self.field(_problem_row(return_lse=1))
        self.assertNotIn("triton", field)
        self.assertNotIn("gluon", field)

    def test_kernels_without_a_soft_cap_are_not_measured_on_a_capped_shape(self):
        # asm_v3 ignores the cap and would be judged on a different function.
        self.assertEqual(
            self.field(_problem_row(logits_soft_cap=30.0)), ["ck", "flydsl"]
        )

    def test_a_restricted_field_still_measures_the_incumbent(self):
        self.assertEqual(
            self.field(_problem_row(), incumbents=["asm_v3"], backends="ck"),
            ["ck", "asm_v3"],
        )


class TestEvidenceRows(unittest.TestCase):
    def test_a_triton_row_carries_the_digest_of_the_tile_it_ran(self):
        tuner = _tuner()
        key = MhaFwdProblem.from_mapping(_problem_row()).key()
        resolved = json.dumps(TILE, sort_keys=True, separators=(",", ":"))
        frame = tuner.result_to_df(
            [
                ((key, "triton", resolved), 10.0, 0.0, "ok", ""),
                ((key, "ck", ""), 12.0, 0.0, "ok", ""),
            ]
        )
        by_backend = frame.set_index("backend")
        self.assertEqual(by_backend.loc["triton", "config_digest"], config_digest(TILE))
        self.assertEqual(by_backend.loc["ck", "config_digest"], "")
        self.assertEqual(json.loads(by_backend.loc["triton", "resolved_config"]), TILE)

    def test_the_resolved_tile_comes_from_triton_only_for_triton_backends(self):
        problem = MhaFwdProblem.from_mapping(_problem_row())
        with mock.patch.object(_TUNER, "triton_config", return_value=TILE) as resolve:
            self.assertEqual(_TUNER.MhaFwdTuner._resolved_config(problem, "ck"), "")
            resolved = _TUNER.MhaFwdTuner._resolved_config(problem, "gluon")
        self.assertEqual(json.loads(resolved), TILE)
        resolve.assert_called_once_with(problem, "gluon")


class TestSelectionHooks(unittest.TestCase):
    def test_a_runtime_row_must_record_its_backend(self):
        tuner = _tuner()
        row = {**_problem_row(), "backend": "triton"}
        self.assertEqual(tuner.expected_selection(row), {"backend": "triton"})

    def test_a_problem_row_only_has_to_run(self):
        self.assertIsNone(_tuner().expected_selection(_problem_row()))

    def test_auto_select_maps_to_a_backend_candidate(self):
        tuner = _tuner()
        self.assertEqual(
            tuner.candidate_from_selection({"backend": "asm_v3"}),
            {"backend": "asm_v3"},
        )
        self.assertIsNone(tuner.candidate_from_selection({"backend": "cutlass"}))


class TestPreProcess(unittest.TestCase):
    def run_pre_process(self, backends="", gfx="gfx950", **overrides):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        untuned = os.path.join(directory.name, "untuned.csv")
        row = {k: v for k, v in _problem_row(**overrides).items()}
        for field in ("gfx", "gpu_model", "cu_num"):
            row.pop(field)
        pd.DataFrame([row]).to_csv(untuned, index=False)
        tuner = _tuner()
        args = argparse.Namespace(
            untune_file=untuned,
            tune_file=os.path.join(directory.name, "tuned.csv"),
            all=False,
            backends=backends,
        )
        with (
            mock.patch.object(tuner, "get_gfx", return_value=gfx),
            mock.patch.object(tuner, "get_gpu_model", return_value="mi355x"),
            mock.patch.object(tuner, "get_cu_num", return_value=256),
        ):
            tuner.pre_process(args)
        return tuner

    def test_rows_are_keyed_in_the_spelling_dispatch_looks_up(self):
        tuner = self.run_pre_process(dtype="bf16")
        row = tuner.untunedf.iloc[0]
        self.assertEqual(row["dtype"], "bfloat16")
        self.assertEqual(
            tuple(row[field] for field in MHA_FWD_TUNING_KEY_FIELDS[:3]),
            ("gfx950", "mi355x", "256"),
        )

    def test_inputs_the_tuner_cannot_build_are_refused(self):
        for overrides in (
            {"dropout_p": 0.1},
            {"has_bias": 1},
            {"sink_size": 4},
            {"min_seqlen_q": 16},
            # 512 queries over two sequences with one of 512 leaves one empty.
            {"max_seqlen_q": 512},
            {"backends": "ck,cutlass"},
            {"gfx": "gfx1250"},
        ):
            with self.subTest(**overrides), self.assertRaises(ValueError):
                self.run_pre_process(**overrides)

    def test_named_backends_are_accepted(self):
        self.run_pre_process(backends="ck, triton")


class TestMeasurement(unittest.TestCase):
    def tasks(self, **overrides):
        tuner = _tuner()
        args = argparse.Namespace(
            warmup=3, iters=7, mp=1, errRatio=0.0, timeout=60, verbose=False
        )
        row = pd.Series(
            dict(
                zip(
                    MHA_FWD_TUNING_KEY_FIELDS,
                    MhaFwdProblem.from_mapping(_problem_row(**overrides)).key(),
                )
            )
        )
        with (
            mock.patch.object(tuner, "candidate_field", return_value=["ck"]),
            mock.patch.object(_TUNER, "mp_tuner", return_value=[]) as run,
            mock.patch("builtins.print"),
        ):
            tuner.tune(pd.DataFrame([row]), None, args)
        return run.call_args.args[0]

    def test_candidates_are_timed_on_kernel_time(self):
        (task,) = self.tasks()
        self.assertEqual(
            task.kwargs, {"num_warmup": 3, "num_iters": 7, "use_cuda_event": False}
        )

    def test_the_reference_applies_the_soft_cap_the_kernels_run(self):
        (task,) = self.tasks(logits_soft_cap=30.0)
        self.assertEqual(task.ref_args[-1], 30.0)

    def test_a_triton_result_is_copied_into_a_fresh_out_buffer(self):
        q = torch.randn(4, 2, 8)
        cu = torch.tensor([0, 4])
        produced = torch.randn(4, 2, 8)
        with mock.patch(
            "aiter.ops.triton.attention.mha.flash_attn_varlen_func",
            return_value=produced,
        ):
            result = _TUNER._run_candidate(
                q, q, q, cu, cu, "triton", 4, 4, 0, 0.0, 0.1, 0.0, 1, 0, -1, -1, 0
            )
        out = result[0] if isinstance(result, tuple) else result
        self.assertIsNot(out, produced)
        torch.testing.assert_close(out, produced)


def _tiles_args(**overrides):
    return argparse.Namespace(
        **{
            "errRatio": 0.0,
            "triton_runs_dir": "runs",
            **overrides,
        }
    )


class TestTritonTiles(unittest.TestCase):
    """--triton-tiles: the Triton harness tunes the tile, this tuner checks it."""

    def test_the_harness_stands_in_only_for_equal_length_bf16_without_window(self):
        problem = MhaFwdProblem.from_mapping(_problem_row())
        self.assertIsNone(_TUNER.harness_reason(problem))
        for overrides in (
            {"total_q": 500},
            {"window_left": 64},
            {"dtype": "float16"},
        ):
            with self.subTest(**overrides):
                problem = MhaFwdProblem.from_mapping(_problem_row(**overrides))
                self.assertIsNotNone(_TUNER.harness_reason(problem))

    def test_the_harness_runs_as_its_owners_run_it(self):
        tuner = _tuner()
        key = MhaFwdProblem.from_mapping(_problem_row()).key()
        row = pd.Series(dict(zip(MHA_FWD_TUNING_KEY_FIELDS, key)))
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(tuner, "_measure") as measure,
            mock.patch.object(
                _TUNER, "triton_shape_file", return_value=f"{directory}/MHA.json"
            ),
            mock.patch.object(_TUNER.subprocess, "run") as run,
            mock.patch.object(_TUNER, "clear_triton_config_caches"),
            mock.patch("builtins.print"),
        ):
            run.return_value.returncode = 0
            snapshots, installed = tuner._tune_triton_tiles(
                pd.DataFrame([row]), _tiles_args()
            )
        measure.assert_not_called()
        command = run.call_args.args[0]
        self.assertEqual(command[2], "mha_varlen_fwd")
        self.assertEqual(command[command.index("--SK") + 1], "256")
        self.assertNotIn("--search", command)
        self.assertEqual(run.call_args.kwargs["cwd"], _TUNER.TRITON_HARNESS_DIR)
        self.assertEqual(snapshots, {f"{directory}/MHA.json": None})
        self.assertEqual(installed, {f"{directory}/MHA.json": [key]})

    def test_the_snapshot_is_the_file_before_the_first_install(self):
        tuner = _tuner()
        rows = [
            pd.Series(
                dict(
                    zip(
                        MHA_FWD_TUNING_KEY_FIELDS,
                        MhaFwdProblem.from_mapping(_problem_row(**dims)).key(),
                    )
                )
            )
            for dims in (
                {},
                {
                    "max_seqlen_q": 512,
                    "total_q": 1024,
                    "max_seqlen_k": 512,
                    "total_k": 1024,
                },
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "MHA.json")
            path.write_text("old")

            def harness(*_args, **_kwargs):
                path.write_text(path.read_text() + "+install")
                return mock.Mock(returncode=0)

            with (
                mock.patch.object(
                    _TUNER, "triton_shape_file", return_value=str(path)
                ),
                mock.patch.object(_TUNER.subprocess, "run", side_effect=harness),
                mock.patch.object(_TUNER, "clear_triton_config_caches"),
                mock.patch("builtins.print"),
            ):
                snapshots, installed = tuner._tune_triton_tiles(
                    pd.DataFrame(rows), _tiles_args()
                )
        self.assertEqual(snapshots, {str(path): "old"})
        self.assertEqual(len(installed[str(path)]), 2)

    def test_the_harness_keeps_the_callers_pythonpath(self):
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/elsewhere"}):
            path = _TUNER.harness_env()["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(path, [_TUNER.REPO_ROOT, "/elsewhere"])

    def test_rows_sharing_a_shape_file_restore_the_file_from_before_the_run(self):
        """Every row with the same head dims and mask installs into one file,
        so restoring per row would put back another row's install."""
        tuner = _tuner()
        short, long = ("key", 256), ("key", 2048)

        def ok(key):
            return ((key, "triton", ""), 9.0, 0.0, "ok", "")

        def bad(key):
            return ((key, "triton", ""), 9.0, 0.3, "mismatch", "")

        cases = {
            "both fail": ([bad(short), bad(long)], "old"),
            "one fails": ([bad(short), ok(long)], "old"),
            "only ck measured": ([((short, "ck", ""), 9.0, 0.0, "ok", "")], "old"),
            "both pass": ([ok(short), ok(long)], "both installs"),
        }
        for name, (results, expected) in cases.items():
            with (
                self.subTest(name),
                tempfile.TemporaryDirectory() as directory,
                mock.patch.object(_TUNER, "clear_triton_config_caches"),
                mock.patch("builtins.print"),
            ):
                path = Path(directory, "MHA.json")
                path.write_text("both installs")
                tuner._roll_back_wrong_tiles(
                    results,
                    {str(path): "old"},
                    {str(path): [short, long]},
                    _tiles_args(),
                )
                self.assertEqual(path.read_text(), expected)

    def test_a_file_the_run_created_is_removed_when_its_tile_fails(self):
        tuner = _tuner()
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(_TUNER, "clear_triton_config_caches"),
            mock.patch("builtins.print"),
        ):
            path = Path(directory, "MHA.json")
            path.write_text("new")
            tuner._roll_back_wrong_tiles(
                [], {str(path): None}, {str(path): [("key",)]}, _tiles_args()
            )
            self.assertFalse(path.exists())

    def test_a_measurement_that_raises_still_rolls_back(self):
        tuner = _tuner()
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(_TUNER, "clear_triton_config_caches"),
            mock.patch("builtins.print"),
        ):
            path = Path(directory, "MHA.json")
            path.write_text("new")
            with (
                mock.patch.object(
                    tuner,
                    "_tune_triton_tiles",
                    return_value=({str(path): "old"}, {str(path): [("key",)]}),
                ),
                mock.patch.object(tuner, "_measure", side_effect=RuntimeError("boom")),
                self.assertRaises(RuntimeError),
            ):
                tuner.tune(pd.DataFrame(), None, _tiles_args(triton_tiles=True))
            self.assertEqual(path.read_text(), "old")

    def test_triton_tiles_without_measuring_triton_is_refused(self):
        check = _TUNER.MhaFwdTuner._check_triton_tiles
        with self.assertRaisesRegex(ValueError, "--triton-tiles needs triton"):
            check(argparse.Namespace(backends="ck", triton_tiles=True))
        check(argparse.Namespace(backends="ck,triton", triton_tiles=True))
        check(argparse.Namespace(backends="", triton_tiles=True))

    def test_published_rows_whose_tile_moved_are_listed(self):
        tuner = _tuner()
        row = {**_problem_row(), "backend": "triton", "config_digest": "sha256:old"}
        with tempfile.TemporaryDirectory() as directory:
            tune_file = Path(directory, "tuned.csv")
            pd.DataFrame([row]).to_csv(tune_file, index=False)
            with (
                mock.patch.object(_TUNER, "triton_config", return_value=TILE),
                mock.patch("builtins.print") as printed,
            ):
                tuner._report_moved_digests(_tiles_args(tune_file=str(tune_file)))
        text = " ".join(str(c.args) for c in printed.call_args_list)
        self.assertIn("1 published Triton rows", text)


class TestPublish(unittest.TestCase):
    """One shape through GatedTuner, with the fresh-process proof stubbed."""

    def publish(self, timings):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        tune_file = os.path.join(directory.name, "tuned.csv")
        Path(tune_file).write_text(",".join(MHA_FWD_RUNTIME_CSV_FIELDS) + "\n")
        args = argparse.Namespace(
            errRatio=0.0,
            profile_file="",
            min_improvement_pct=3.0,
            all=False,
            tune_file=tune_file,
            backends="",
            warmup=1,
            iters=1,
            timeout=60,
        )
        tuner = _tuner()
        row = pd.Series(
            dict(
                zip(
                    MHA_FWD_TUNING_KEY_FIELDS,
                    MhaFwdProblem.from_mapping(_problem_row()).key(),
                )
            )
        )
        tuner.untunedf = pd.DataFrame([row])
        tuner.load_published(args)

        def prove(row, config_file, args):
            expected = tuner.expected_selection(row)
            observed = [expected or {"backend": "asm_v3"}]
            status = "verified" if expected else "unchecked"
            return SelectionProof(status, expected, observed, 1.0, "", 0.0)

        resolved = json.dumps(TILE, sort_keys=True, separators=(",", ":"))
        key = tuple(row)
        with (
            mock.patch.object(tuner, "prove_selection", side_effect=prove),
            mock.patch("builtins.print"),
        ):
            tuner.incumbents(row)
            rets = [
                ((key, b, resolved if b == "triton" else ""), us, 0.0, "ok", "")
                for b, us in timings.items()
            ]
            winners = tuner.post_process(rets, args)
            tuner.result_to_csv(winners, tune_file)
        return pd.read_csv(tune_file, dtype=str, keep_default_na=False)

    def test_triton_beating_auto_select_writes_a_row_with_its_digest(self):
        table = self.publish({"asm_v3": 100.0, "ck": 120.0, "triton": 80.0})
        self.assertEqual(list(table["backend"]), ["triton"])
        self.assertEqual(table["config_digest"].iloc[0], config_digest(TILE))

    def test_a_win_inside_the_bar_leaves_the_shape_on_auto_select(self):
        table = self.publish({"asm_v3": 100.0, "triton": 98.0})
        self.assertTrue(table.empty)


if __name__ == "__main__":
    unittest.main()
