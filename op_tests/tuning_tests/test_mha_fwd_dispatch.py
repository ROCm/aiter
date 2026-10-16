# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the MHA forward tuned-row lookup and dispatch.

A tuned row pins the backend for one exact problem on one GPU: a hit routes
the call to that backend, a miss runs what auto-select runs today, and a row
the backend cannot serve falls back to auto-select instead of failing. A
Triton or Gluon row only holds while Triton still resolves the tile it was
tuned with.
"""

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import triton  # noqa: F401  # isort: skip  # Must precede torch on this ROCm environment.
import torch

from aiter.ops import mha
from aiter.ops.mha_fwd_policy import (
    MHA_FWD_CONFIG_ENV,
    MHA_FWD_RUNTIME_CSV_FIELDS,
    MHA_FWD_TUNED_CSV,
    MhaFwdProblem,
)
from aiter.utility import selection_proof

HARDWARE = {"gfx": "gfx942", "gpu_model": "mi325x", "cu_num": 304}


def _problem_row(**overrides):
    return {
        **HARDWARE,
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
        **overrides,
    }


def _tuned_row(backend="ck", us="", config_digest="", **key):
    return {
        **_problem_row(**key),
        "backend": backend,
        "us": us,
        "config_digest": config_digest,
    }


TRITON_TILE = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4}
TRITON_DIGEST = mha.config_digest(TRITON_TILE)


def _write_table(directory, rows):
    path = os.path.join(directory, MHA_FWD_TUNED_CSV)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(MHA_FWD_RUNTIME_CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _key_args(max_seqlen_k=42700):
    q = torch.empty((4096, 12, 192), dtype=torch.bfloat16)
    k = torch.empty((max_seqlen_k, 12, 192), dtype=torch.bfloat16)
    v = torch.empty((max_seqlen_k, 12, 128), dtype=torch.bfloat16)
    return {
        "mode": "varlen",
        "q": q,
        "k": k,
        "v": v,
        "batch": 1,
        "max_seqlen_q": 4096,
        "max_seqlen_k": max_seqlen_k,
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


def _dummy_varlen_tensors():
    q = torch.empty((8, 12, 192), dtype=torch.bfloat16)
    k = torch.empty((16, 12, 192), dtype=torch.bfloat16)
    v = torch.empty((16, 12, 128), dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 8], dtype=torch.int32)
    cu_k = torch.tensor([0, 16], dtype=torch.int32)
    return q, k, v, cu_q, cu_k


def _on(hardware=None):
    return mock.patch.object(
        mha, "get_tuning_hardware", return_value=hardware or HARDWARE
    )


class _Table(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        mha.clear_mha_fwd_tuning_cache()
        self.addCleanup(mha.clear_mha_fwd_tuning_cache)

    def use(self, rows):
        """Point dispatch at a table holding ``rows``."""
        path = _write_table(self.directory, rows)
        patch = mock.patch.object(mha, "_mha_fwd_table_path", return_value=path)
        patch.start()
        self.addCleanup(patch.stop)
        return path


class TestMhaFwdKey(unittest.TestCase):
    def test_problem_key_starts_with_arch_model_and_cu(self):
        problem = MhaFwdProblem.from_mapping(_problem_row())
        self.assertEqual(problem.key()[:3], ("gfx942", "mi325x", "304"))

    def test_bf16_spelling_normalizes_to_runtime_dtype(self):
        problem = MhaFwdProblem.from_mapping(_problem_row(dtype="bf16"))
        self.assertEqual(problem.dtype, "bfloat16")

    def test_the_runtime_key_matches_the_row_key(self):
        with _on():
            key = mha._mha_fwd_tuning_key(**_key_args())
        self.assertEqual(key, MhaFwdProblem.from_mapping(_problem_row()).key())

    def test_int_and_bool_spellings_of_a_call_find_the_same_row(self):
        with _on():
            key = mha._mha_fwd_tuning_key(
                **{
                    **_key_args(),
                    "dropout_p": 0,
                    "logits_soft_cap": 0,
                    "causal": 0,
                    "return_lse": 0,
                    "window_size": [-1, -1, 0],
                }
            )
        self.assertEqual(key, MhaFwdProblem.from_mapping(_problem_row()).key())

    def test_an_unknown_dtype_is_refused(self):
        with self.assertRaisesRegex(ValueError, "dtype"):
            MhaFwdProblem.from_mapping(_problem_row(dtype="bfloat17"))

    def test_a_digest_never_reads_as_a_number(self):
        digest = mha.config_digest({"BLOCK_M": 1})
        self.assertTrue(digest.startswith("sha256:"))
        self.assertEqual(len(digest), len("sha256:") + 12)


class TestMhaFwdTable(_Table):
    def test_the_runtime_key_finds_a_row(self):
        table = mha._load_mha_fwd_tuning_table(self.use([_tuned_row("opus")]))
        with _on():
            key = mha._mha_fwd_tuning_key(**_key_args())
        self.assertEqual(table[key], "opus")

    def test_a_different_gpu_model_does_not_match(self):
        table = mha._load_mha_fwd_tuning_table(self.use([_tuned_row("ck")]))
        with _on({**HARDWARE, "gpu_model": "mi300x"}):
            key = mha._mha_fwd_tuning_key(**_key_args())
        self.assertNotIn(key, table)

    def test_a_bad_row_is_skipped_and_the_rest_still_load(self):
        rows = [
            _tuned_row("cutlass", total_q=2048, max_seqlen_q=2048),
            _tuned_row("ck", nhead_k=5),
            _tuned_row("asm_v3"),
        ]
        with self.assertLogs("aiter", level="WARNING") as logs:
            table = mha._load_mha_fwd_tuning_table(self.use(rows))
        self.assertEqual(list(table.values()), ["asm_v3"])
        self.assertEqual(len(logs.records), 2)

    def test_a_duplicate_key_keeps_the_first_row(self):
        rows = [_tuned_row("ck"), _tuned_row("asm_v3")]
        with self.assertLogs("aiter", level="WARNING"):
            table = mha._load_mha_fwd_tuning_table(self.use(rows))
        self.assertEqual(list(table.values()), ["ck"])

    def test_a_missing_file_named_by_the_env_var_warns(self):
        path = os.path.join(self.directory, "typo.csv")
        with (
            mock.patch.dict(os.environ, {MHA_FWD_CONFIG_ENV: path}),
            self.assertLogs("aiter", level="WARNING") as logs,
        ):
            self.assertEqual(mha._load_mha_fwd_tuning_table(path), {})
        self.assertIn("does not exist", logs.output[0])

    def test_the_shipped_table_has_the_runtime_schema(self):
        config = Path(mha.__file__).parents[1] / "configs" / MHA_FWD_TUNED_CSV
        with config.open(encoding="utf-8", newline="") as file:
            fields = tuple(csv.DictReader(file).fieldnames or ())
        self.assertEqual(fields, MHA_FWD_RUNTIME_CSV_FIELDS)


class TestTritonRows(_Table):
    """A Triton or Gluon row names only the backend; Triton keeps its tiles."""

    def load(self, rows, gfx="gfx942", tile=TRITON_TILE):
        with (
            mock.patch.object(mha, "get_gfx", return_value=gfx),
            mock.patch.object(mha, "triton_config", return_value=tile) as resolve,
        ):
            table = mha._load_mha_fwd_tuning_table(self.use(rows))
        return table, resolve

    def assert_skipped(self, row, reason):
        with self.assertLogs("aiter", level="WARNING") as logs:
            table, _ = self.load([row])
        self.assertEqual(table, {})
        self.assertIn(reason, logs.output[0])

    def test_a_row_tuned_with_the_tile_triton_resolves_loads(self):
        table, resolve = self.load([_tuned_row("triton", config_digest=TRITON_DIGEST)])
        self.assertEqual(list(table.values()), ["triton"])
        self.assertEqual(resolve.call_args.args[1], "triton")

    def test_a_row_is_dropped_once_triton_resolves_another_tile(self):
        retuned = {**TRITON_TILE, "num_warps": 8}
        with self.assertLogs("aiter", level="WARNING") as logs:
            table, _ = self.load(
                [_tuned_row("triton", config_digest=TRITON_DIGEST)], tile=retuned
            )
        self.assertEqual(table, {})
        self.assertIn("retune it", logs.output[0])

    def test_a_row_without_a_digest_is_skipped(self):
        self.assert_skipped(_tuned_row("triton"), "needs config_digest")

    def test_a_row_whose_digest_check_fails_to_run_is_skipped(self):
        # A renamed private resolver, or a row Triton cannot resolve, must
        # cost that row only, not every MHA call.
        rows = [
            _tuned_row("triton", config_digest=TRITON_DIGEST),
            _tuned_row("asm_v3", total_q=2048, max_seqlen_q=2048),
        ]
        for error in (ImportError("no _get_config"), AttributeError("bfloat17")):
            with self.subTest(error=type(error).__name__):
                mha.clear_mha_fwd_tuning_cache()
                with (
                    mock.patch.object(mha, "get_gfx", return_value="gfx942"),
                    mock.patch.object(mha, "triton_config", side_effect=error),
                    self.assertLogs("aiter", level="WARNING"),
                ):
                    table = mha._load_mha_fwd_tuning_table(self.use(rows))
                self.assertEqual(list(table.values()), ["asm_v3"])

    def test_a_row_triton_cannot_serve_is_skipped(self):
        for key, reason in (
            ({"logits_soft_cap": 30.0}, "soft-cap"),
            ({"has_physical_padding": 1}, "padded"),
            ({"has_q_descale": 1}, "FP8"),
            ({"return_lse": 1}, "layout"),
            ({"is_grad": 1}, "backward"),
            ({"has_block_table": 1}, "block_table"),
            ({"has_bias": 1}, "bias"),
            ({"causal": 1, "window_left": 127, "window_right": 0}, "right window"),
        ):
            with self.subTest(**key):
                mha.clear_mha_fwd_tuning_cache()
                self.assert_skipped(
                    _tuned_row("triton", config_digest=TRITON_DIGEST, **key), reason
                )

    def test_a_gluon_row_with_dropout_is_skipped(self):
        with mock.patch(
            "aiter.ops.triton.attention.mha.is_gluon_available", return_value=True
        ):
            self.assert_skipped(
                _tuned_row("gluon", config_digest=TRITON_DIGEST, dropout_p=0.1),
                "dropout",
            )

    def test_rows_for_another_arch_are_not_resolved(self):
        table, resolve = self.load(
            [_tuned_row("triton", config_digest="stale")], gfx="gfx950"
        )
        self.assertEqual(len(table), 1)
        resolve.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "Triton resolves per GPU arch")
    def test_the_digest_is_of_the_tile_triton_itself_resolves(self):
        from aiter.ops.triton._triton_kernels.attention.mha import _get_config

        problem = MhaFwdProblem.from_mapping(_problem_row())
        expected = _get_config(
            False,
            torch.bfloat16,
            has_pe=True,
            head_dim_v=128,
            causal=False,
            max_seqlen_q=4096,
            max_seqlen_k=42700,
            batch_heads=12,
            head_dim_qk=192,
        )
        self.assertEqual(mha.triton_config(problem, "triton"), dict(expected))


class TestMhaFwdLookup(_Table):
    def test_an_empty_table_returns_before_reading_the_hardware_or_the_key(self):
        self.use([])
        with (
            mock.patch.object(mha, "get_tuning_hardware") as hardware,
            mock.patch.object(mha, "_mha_fwd_tuning_key") as key,
        ):
            self.assertIsNone(mha._get_mha_fwd_tuned_backend(**_key_args()))
        hardware.assert_not_called()
        key.assert_not_called()

    def test_an_empty_table_compiles_without_a_graph_break(self):
        self.use([])

        def call(q, k, v):
            backend = mha._get_mha_fwd_tuned_backend(
                **{**_key_args(), "q": q, "k": k, "v": v}
            )
            return q.sum() if backend is None else q.mean()

        args = _key_args()
        torch._dynamo.reset()
        compiled = torch.compile(call, fullgraph=True, dynamic=True)
        compiled(args["q"], args["k"], args["v"])

    @unittest.skipUnless(torch.cuda.is_available(), "probes the real GPU")
    def test_a_table_with_rows_compiles_fullgraph_with_the_real_probe(self):
        # No hardware mock: the probe reads sysfs and runs rocminfo, which
        # dynamo cannot trace, so it has to stay outside the graph.
        self.use([_tuned_row("ck", gpu_model="mi300x")])

        def call(q, k, v):
            backend = mha._get_mha_fwd_tuned_backend(
                **{**_key_args(), "q": q, "k": k, "v": v}
            )
            return q.sum() if backend is None else q.mean()

        args = _key_args()
        torch._dynamo.reset()
        compiled = torch.compile(call, fullgraph=True, dynamic=True)
        compiled(args["q"], args["k"], args["v"])

    def test_other_head_dims_skip_the_key(self):
        self.use([_tuned_row("ck", hdim_q=128, hdim_v=128)])
        with (
            _on(),
            mock.patch.object(mha, "get_gfx", return_value="gfx942"),
            mock.patch.object(mha, "_mha_fwd_tuning_key") as key,
        ):
            self.assertIsNone(mha._get_mha_fwd_tuned_backend(**_key_args()))
        key.assert_not_called()

    def test_rows_for_other_hardware_skip_the_key(self):
        self.use([_tuned_row("ck", gpu_model="mi300x")])
        with (
            _on(),
            mock.patch.object(mha, "get_gfx", return_value="gfx942"),
            mock.patch.object(mha, "_mha_fwd_tuning_key") as key,
        ):
            self.assertIsNone(mha._get_mha_fwd_tuned_backend(**_key_args()))
        key.assert_not_called()

    def test_a_hit_returns_the_pinned_backend(self):
        self.use([_tuned_row("opus")])
        with _on(), mock.patch.object(mha, "get_gfx", return_value="gfx942"):
            self.assertEqual(mha._get_mha_fwd_tuned_backend(**_key_args()), "opus")

    def test_each_compiled_shape_looks_itself_up(self):
        """The table read is constant-folded, so the key must not be.

        Applied to the whole lookup, ``assume_constant_result`` froze the
        first traced shape's row into the graph for every later shape once
        dynamo made the sizes dynamic: a wrong kernel, not a slow one.
        """
        self.use(
            [
                _tuned_row(backend, total_k=max_seqlen_k, max_seqlen_k=max_seqlen_k)
                for backend, max_seqlen_k in (("ck", 8192), ("asm_v3", 16384))
            ]
        )
        resolved = []

        def call(q, k, v, max_seqlen_k):
            resolved.append(
                mha._get_mha_fwd_tuned_backend(
                    **{**_key_args(), "q": q, "k": k, "v": v},
                    max_seqlen_k=max_seqlen_k,
                )
            )
            return q.sum()

        with mock.patch.object(mha, "get_gfx", return_value="gfx942"), _on():
            torch._dynamo.reset()
            compiled = torch.compile(call, dynamic=True)
            for max_seqlen_k in (8192, 16384):
                args = _key_args(max_seqlen_k)
                compiled(args["q"], args["k"], args["v"], max_seqlen_k)
        self.assertEqual(resolved, ["ck", "asm_v3"])


class TestMhaVarlenForward(unittest.TestCase):
    def _varlen_forward(self, entry="fmha_v3_varlen_fwd", gfx="gfx942", **overrides):
        """Dispatch once with ``entry`` mocked, and hand back that mock."""
        q, k, v, cu_q, cu_k = _dummy_varlen_tensors()
        with (
            mock.patch.object(mha, "get_gfx", return_value=gfx),
            mock.patch.object(
                mha, entry, return_value=tuple(torch.empty(1) for _ in range(4))
            ) as backend,
        ):
            mha._flash_attn_varlen_forward(
                q, k, v, cu_q, cu_k, None, None, 8, 16, 0, 0.0, 0.125, False,
                **overrides,
            )  # fmt: skip
        backend.assert_called_once()
        return backend

    def test_no_row_runs_auto_select(self):
        self._varlen_forward(selected_backend=None)

    def test_a_ck_row_displaces_asm_auto_select(self):
        self._varlen_forward(entry="mha_varlen_fwd", selected_backend="ck")

    def test_a_row_the_backend_cannot_serve_runs_auto_select_not_ck(self):
        # opus only serves gfx950, so on gfx942 the row is declined and the
        # call takes what auto-select takes there (asm_v3), not CK.
        mha._warn_declined_mha_fwd_row.cache_clear()
        with self.assertLogs("aiter", level="WARNING"):
            self._varlen_forward(selected_backend="opus")

    def test_dispatch_records_what_it_ran_for_a_selection_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "proof.jsonl")
            with mock.patch.dict(
                os.environ, {selection_proof.SELECTION_PROOF_ENV: path}
            ):
                self._varlen_forward(selected_backend=None)
                self._varlen_forward(entry="mha_varlen_fwd", selected_backend="ck")
            self.assertEqual(
                selection_proof.read_selections(path, "mha_fwd"),
                [{"backend": "asm_v3"}, {"backend": "ck"}],
            )


class TestFlashAttnVarlenFunc(unittest.TestCase):
    def _route(self, backend, flydsl="flydsl"):
        q, k, v, cu_q, cu_k = _dummy_varlen_tensors()
        with (
            mock.patch.object(mha, "_get_mha_fwd_tuned_backend", return_value=backend),
            mock.patch.object(
                mha.FlashAttnVarlenFunc, "apply", return_value="aiter"
            ) as apply,
            mock.patch(
                "aiter.ops.flydsl.fmha_kernels.flydsl_flash_attn_varlen_func",
                return_value=flydsl,
            ),
        ):
            result = mha.flash_attn_varlen_func(q, k, v, cu_q, cu_k, 8, 16)
        return result, apply

    def test_a_pinned_backend_reaches_the_forward(self):
        result, apply = self._route("asm_v3")
        self.assertEqual(result, "aiter")
        self.assertEqual(apply.call_args.args[-1], "asm_v3")

    def test_a_triton_row_calls_triton_without_a_config(self):
        with (
            mock.patch(
                "aiter.ops.triton.attention.mha.flash_attn_varlen_func",
                return_value="triton",
            ) as triton_entry,
            tempfile.TemporaryDirectory() as directory,
        ):
            path = os.path.join(directory, "proof.jsonl")
            with mock.patch.dict(
                os.environ, {selection_proof.SELECTION_PROOF_ENV: path}
            ):
                result, apply = self._route("gluon")
            selections = selection_proof.read_selections(path, "mha_fwd")
        self.assertEqual(result, "triton")
        apply.assert_not_called()
        kwargs = triton_entry.call_args.kwargs
        self.assertEqual(kwargs["backend"], "gluon")
        self.assertNotIn("config", kwargs)
        self.assertEqual(selections, [{"backend": "gluon"}])

    def test_a_triton_row_writes_the_callers_out_buffer(self):
        q, k, v, cu_q, cu_k = _dummy_varlen_tensors()
        out = torch.full((q.shape[0], q.shape[1], v.shape[-1]), float("nan"))
        computed = torch.ones_like(out)
        with (
            mock.patch.object(mha, "_get_mha_fwd_tuned_backend", return_value="triton"),
            mock.patch(
                "aiter.ops.triton.attention.mha.flash_attn_varlen_func",
                return_value=computed,
            ),
        ):
            result = mha.flash_attn_varlen_func(q, k, v, cu_q, cu_k, 8, 16, out=out)
        self.assertIs(result, out)
        self.assertTrue(torch.equal(out, computed))

    def test_a_declined_flydsl_row_continues_as_auto_select(self):
        mha._warn_declined_mha_fwd_row.cache_clear()
        with self.assertLogs("aiter", level="WARNING"):
            result, apply = self._route("flydsl", flydsl=None)
        self.assertEqual(result, "aiter")
        self.assertIsNone(apply.call_args.args[-1])


if __name__ == "__main__":
    unittest.main()
