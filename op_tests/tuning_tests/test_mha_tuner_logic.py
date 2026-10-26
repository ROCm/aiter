# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the MHA forward tuner.

The --compare publication rules are tested in test_compare_logic.py.
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

_TUNER_PATH = Path(__file__).parents[2] / MHA_FWD_TUNER_SCRIPT
_SPEC = importlib.util.spec_from_file_location("tune_mha_fwd", _TUNER_PATH)
_TUNER = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_TUNER)

TILE = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4}


def _triton_mha_imports():
    try:
        import aiter.ops.triton.attention.mha  # noqa: F401
    except (ImportError, RuntimeError):
        return False
    return True


needs_triton_mha = unittest.skipUnless(
    _triton_mha_imports(), "Triton's MHA module probes the GPU when imported"
)


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


def _key_row(**overrides):
    key = MhaFwdProblem.from_mapping(_problem_row(**overrides)).key()
    return dict(zip(MHA_FWD_TUNING_KEY_FIELDS, key))


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

    def test_matches_softmax_attention(self):
        cases = (
            ("plain", [7, 3], [7, 3], {}),
            ("soft_cap", [7, 3], [7, 3], {"cap": 2.0}),
            ("causal_soft_cap", [6, 9], [6, 9], {"causal": True, "cap": 1.5}),
            ("gqa", [5, 9], [5, 9], {"nhead_q": 8, "nhead_k": 2}),
            ("causal_more_keys", [3, 4], [11, 6], {"causal": True}),
            # Rows above the diagonal see no key; they come out as zeros.
            ("causal_more_queries", [9], [4], {"causal": True}),
            ("window", [12, 5], [12, 5], {"left": 3, "right": 1}),
        )
        # Whole sequences, then chunks of one or two query rows.
        for budget in (_TUNER.REFERENCE_CHUNK_BYTES, 4, 4 * 4 * 11 * 2):
            for name, q_lengths, k_lengths, extra in cases:
                with (
                    self.subTest(name, budget=budget),
                    mock.patch.object(_TUNER, "REFERENCE_CHUNK_BYTES", budget),
                ):
                    self.check(q_lengths=q_lengths, k_lengths=k_lengths, **extra)

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


def _tuner(backends=""):
    tuner = _TUNER.MhaFwdTuner()
    tuner._args = argparse.Namespace(backends=backends)
    return tuner


@needs_triton_mha
class TestCandidateField(unittest.TestCase):
    def field(self, row, backends=""):
        tuner = _tuner(backends)
        with (
            mock.patch(
                "aiter.ops.triton.attention.mha.is_gluon_available", return_value=True
            ),
            mock.patch("builtins.print"),
        ):
            return tuner.candidate_field(pd.Series(row))

    def test_every_backend_that_exists_on_the_arch_is_measured(self):
        for gfx, want in (
            ("gfx950", ["asm_v3", "ck", "flydsl", "gluon", "opus", "triton"]),
            ("gfx942", ["asm_v3", "ck", "flydsl", "triton"]),
        ):
            with self.subTest(gfx):
                self.assertEqual(self.field(_problem_row(gfx=gfx)), want)

    def test_triton_is_not_measured_where_it_cannot_stand_in_for_ck(self):
        field = self.field(_problem_row(return_lse=1))
        self.assertNotIn("triton", field)
        self.assertNotIn("gluon", field)

    def test_kernels_without_a_soft_cap_are_not_measured_on_a_capped_shape(self):
        # asm_v3, FlyDSL and opus ignore the cap. At production caps the
        # reference cannot tell a capped output from an uncapped one, so only
        # this exclusion keeps them from winning a capped shape.
        self.assertEqual(self.field(_problem_row(logits_soft_cap=30.0)), ["ck"])

    def test_backends_restricts_the_field(self):
        self.assertEqual(
            self.field(_problem_row(), backends="triton,ck"), ["ck", "triton"]
        )


class TestEvidenceRows(unittest.TestCase):
    def test_a_triton_row_carries_the_digest_of_the_tile_it_ran(self):
        tuner = _tuner()
        key = tuple(_key_row().values())
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
    def test_a_runtime_row_must_record_its_backend_and_a_problem_row_only_run(self):
        tuner = _tuner()
        row = {**_problem_row(), "backend": "triton"}
        self.assertEqual(tuner.expected_selection(row), {"backend": "triton"})
        self.assertIsNone(tuner.expected_selection(_problem_row()))

    def test_compare_also_times_auto_select(self):
        self.assertTrue(_TUNER.MhaFwdTuner.COMPARE_AUTOSELECT)


class TestPreProcess(unittest.TestCase):
    def run_pre_process(
        self, backends="", gfx="gfx950", tuned=(), retune_all=False, **overrides
    ):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        untuned = os.path.join(directory.name, "untuned.csv")
        tune_file = os.path.join(directory.name, "tuned.csv")
        row = _problem_row(**overrides)
        for field in ("gfx", "gpu_model", "cu_num"):
            row.pop(field)
        pd.DataFrame([row]).to_csv(untuned, index=False)
        if tuned:
            pd.DataFrame(tuned, columns=MHA_FWD_RUNTIME_CSV_FIELDS).to_csv(
                tune_file, index=False
            )
        tuner = _tuner()
        args = argparse.Namespace(
            untune_file=untuned,
            tune_file=tune_file,
            all=retune_all,
            backends=backends,
        )
        with (
            mock.patch.object(tuner, "get_gfx", return_value=gfx),
            mock.patch.object(tuner, "get_gpu_model", return_value="mi355x"),
            mock.patch.object(tuner, "get_cu_num", return_value=256),
            mock.patch.object(_TUNER, "triton_config", return_value=TILE),
            mock.patch("builtins.print"),
        ):
            tuner.pre_process(args)
        return tuner

    def test_a_current_row_is_skipped_and_a_stale_one_retuned(self):
        current = {
            **_key_row(),
            "backend": "triton",
            "config_digest": config_digest(TILE),
        }
        stale = {**current, "config_digest": "sha256:000000000000"}
        other_gpu = {**current, "gfx": "gfx942"}
        for tuned, retune_all, want in (
            ([current], False, 0),
            ([current], True, 1),
            ([stale], False, 1),
            ([other_gpu], False, 1),
        ):
            row = tuned[0]
            with self.subTest(
                digest=row["config_digest"], gfx=row["gfx"], all=retune_all
            ):
                tuner = self.run_pre_process(tuned=tuned, retune_all=retune_all)
                self.assertEqual(len(tuner.untunedf), want)

    def test_rows_are_keyed_in_the_spelling_dispatch_looks_up(self):
        tuner = self.run_pre_process(dtype="bf16", backends="ck, triton")
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


class TestMeasurement(unittest.TestCase):
    def tasks(self, **overrides):
        tuner = _tuner()
        args = argparse.Namespace(
            warmup=3, iters=7, mp=1, errRatio=0.0, timeout=60, verbose=False
        )
        with (
            mock.patch.object(tuner, "candidate_field", return_value=["ck"]),
            mock.patch.object(_TUNER, "mp_tuner", return_value=[]) as run,
            mock.patch("builtins.print"),
        ):
            tuner.tune(pd.DataFrame([_key_row(**overrides)]), None, args)
        return run.call_args.args[0]

    def test_a_shape_without_candidates_is_left_out_of_the_run(self):
        # mp_tuner asserts one group per shape, so an empty group would abort
        # every other shape too.
        tuner = _tuner()
        args = argparse.Namespace(
            warmup=3, iters=7, mp=1, errRatio=0.0, timeout=60, verbose=False
        )
        rows = pd.DataFrame([_key_row(logits_soft_cap=30.0), _key_row()])
        with (
            mock.patch.object(tuner, "candidate_field", side_effect=[[], ["ck"]]),
            mock.patch.object(_TUNER, "mp_tuner", return_value=[]) as run,
            mock.patch("builtins.print"),
        ):
            tuner.tune(rows, None, args)
        tasks, group_sizes = run.call_args.args[:2]
        self.assertEqual([task.info[1] for task in tasks], ["ck"])
        self.assertEqual(len(group_sizes), 1)

    def test_candidates_are_timed_on_kernel_time(self):
        (task,) = self.tasks()
        self.assertEqual(task.kwargs, {"num_warmup": 3, "num_iters": 7})

    def test_the_selection_probe_is_timed_on_kernel_time_too(self):
        """--compare ranks rows on the probe's latency, so it must be in the
        candidates' domain, not a host-timed loop of public-op calls. The
        inputs are arguments, so run_perftest rotates them as it does the
        candidates'."""
        out = torch.zeros(4)
        data = {name: torch.zeros(1) for name in _TUNER.INPUTS}
        with (
            mock.patch.object(_TUNER, "generate_data", return_value=data),
            mock.patch.object(_TUNER, "_chunked_reference", return_value=out),
            mock.patch.object(_TUNER, "flash_attn_varlen_func"),
            mock.patch.object(_TUNER, "_normalize_result", return_value=out),
            mock.patch.object(
                _TUNER, "run_perftest", return_value=(out, 12.5)
            ) as perftest,
        ):
            result = _TUNER.selection_probe(_problem_row(), warmup=3, iters=1)
        self.assertEqual(result, {"latency_us": 12.5, "iterations": 2})
        self.assertEqual(perftest.call_args.kwargs, {"num_warmup": 3, "num_iters": 2})
        timed = perftest.call_args.args[1:]
        self.assertEqual(len(timed), len(_TUNER.INPUTS))
        for given, name in zip(timed, _TUNER.INPUTS):
            self.assertIs(given, data[name])

    def test_the_absolute_tolerance_shrinks_with_the_keys_a_query_sees(self):
        def atol(**overrides):
            row = _problem_row(**overrides)
            return _TUNER._error_atol(MhaFwdProblem.from_mapping(row))

        long = {"batch": 1, "total_q": 256, "total_k": 16384, "max_seqlen_k": 16384}
        self.assertEqual(atol(), _TUNER.ERROR_ATOL)
        self.assertAlmostEqual(atol(**long), _TUNER.ERROR_ATOL / 4)
        self.assertEqual(
            atol(**long, causal=1, window_left=127, window_right=0), _TUNER.ERROR_ATOL
        )
        (task,) = self.tasks(**long)
        self.assertAlmostEqual(task.atol, _TUNER.ERROR_ATOL / 4)

    def test_the_reference_applies_the_soft_cap_the_kernels_run(self):
        (task,) = self.tasks(logits_soft_cap=30.0)
        self.assertEqual(task.ref_args[-1], 30.0)


class TestWinners(unittest.TestCase):
    RESOLVED = json.dumps(TILE, sort_keys=True, separators=(",", ":"))

    def ret(self, key, backend, us, status="ok", err=0.0):
        resolved = self.RESOLVED if backend == "triton" else ""
        return ((tuple(key.values()), backend, resolved), us, err, status, "")

    def test_the_fastest_correct_candidate_wins_and_a_shape_without_one_fails(self):
        tuned, failed = _key_row(), _key_row(nhead_q=16)
        tuner = _tuner()
        rets = [
            self.ret(tuned, "triton", 5.0, status="mismatch"),
            self.ret(tuned, "flydsl", 6.0, err=0.1),
            self.ret(tuned, "ck", 9.0),
            self.ret(tuned, "asm_v3", 8.0),
            self.ret(failed, "ck", -1.0, status="crash"),
        ]
        args = argparse.Namespace(errRatio=0.0, profile_file="")
        winners = tuner.post_process(rets, args)
        self.assertEqual(list(winners.columns), tuner.columns)
        self.assertEqual(list(winners["backend"]), ["asm_v3"])
        self.assertEqual(list(tuner.success["backend"]), ["asm_v3"])
        self.assertEqual(tuner.failed["nhead_q"].tolist(), ["16"])
        self.assertIn("ck crash", tuner.failed["detail"].iloc[0])

    def test_a_written_row_replaces_its_shape_and_keeps_the_others(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        tune_file = os.path.join(directory.name, "tuned.csv")
        mine = {**_key_row(), "backend": "ck", "us": "20.0", "config_digest": ""}
        others = [{**mine, "gfx": "gfx942"}, {**mine, "nhead_q": "16"}]
        pd.DataFrame([mine, *others]).to_csv(tune_file, index=False)
        tuner = _tuner()
        winners = tuner.result_to_df([self.ret(_key_row(), "triton", 10.0)])
        with (
            mock.patch.object(tuner, "get_gfx", return_value="gfx950"),
            mock.patch.object(tuner, "get_gpu_model", return_value="mi355x"),
            mock.patch.object(tuner, "get_cu_num", return_value=256),
        ):
            tuner.result_to_csv(winners, tune_file)
        table = pd.read_csv(tune_file, dtype=str, keep_default_na=False)
        self.assertEqual(list(table.columns), list(MHA_FWD_RUNTIME_CSV_FIELDS))
        self.assertEqual(table.iloc[:2].to_dict("records"), others)
        self.assertEqual(table["backend"].iloc[2], "triton")
        self.assertEqual(table["config_digest"].iloc[2], config_digest(TILE))
        self.assertEqual(len(table), 3)


if __name__ == "__main__":
    unittest.main()
