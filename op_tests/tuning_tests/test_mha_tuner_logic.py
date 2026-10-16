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


def _naive_attention(q, k, v, cu_q, cu_k, scale, causal, left, right):
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
        out, lse = _TUNER._chunked_reference(
            q, k, v, cu_q, cu_k, 0.125, causal, left, right, True
        )
        want_out, want_lse = _naive_attention(
            q, k, v, cu_q, cu_k, 0.125, causal, left, right
        )
        torch.testing.assert_close(out, want_out, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(lse, want_lse, rtol=1e-4, atol=1e-4)

    def test_plain_varlen(self):
        self.check(q_lengths=[7, 3], k_lengths=[7, 3])

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
    def run_pre_process(self, **overrides):
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
            backends="",
        )
        with (
            mock.patch.object(tuner, "get_gfx", return_value="gfx950"),
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
        for overrides in ({"dropout_p": 0.1}, {"has_bias": 1}, {"sink_size": 4}):
            with self.subTest(**overrides), self.assertRaises(ValueError):
                self.run_pre_process(**overrides)


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
