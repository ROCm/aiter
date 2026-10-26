# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the MHA forward tuned-row lookup and dispatch."""

import contextlib
import csv
import os
import tempfile
import unittest
from unittest import mock

import triton  # noqa: F401  # isort: skip  # Must precede torch on this ROCm environment.
import torch

from aiter.ops import mha
from aiter.ops import mha_fwd_policy as policy
from aiter.ops.mha_fwd_policy import (
    MHA_FWD_CONFIG_ENV,
    MHA_FWD_RUNTIME_CSV_FIELDS,
    MHA_FWD_TUNED_CSV,
    MhaFwdProblem,
)
from aiter.utility import dispatch_verification

HARDWARE = {"gfx": "gfx942", "gpu_model": "mi325x", "cu_num": 304}
HARDWARE_CELLS = ("gfx942", "mi325x", "304")


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


def _tuned_row(backend="ck", **key):
    return {
        **_problem_row(**key),
        "backend": backend,
        "us": "",
    }


def _key_args(max_seqlen_k=42700, **overrides):
    return {
        "mode": "varlen",
        "q": torch.empty((4096, 12, 192), dtype=torch.bfloat16),
        "k": torch.empty((max_seqlen_k, 12, 192), dtype=torch.bfloat16),
        "v": torch.empty((max_seqlen_k, 12, 128), dtype=torch.bfloat16),
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
        **overrides,
    }


def _dummy_varlen_tensors():
    q = torch.empty((8, 12, 192), dtype=torch.bfloat16)
    k = torch.empty((16, 12, 192), dtype=torch.bfloat16)
    v = torch.empty((16, 12, 128), dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 8], dtype=torch.int32)
    cu_k = torch.tensor([0, 16], dtype=torch.int32)
    return q, k, v, cu_q, cu_k


def _on_hardware():
    """Run on HARDWARE with gfx942 as the built arch."""
    return mock.patch.multiple(
        policy,
        get_tuning_hardware=mock.Mock(return_value=HARDWARE),
        get_gfx=mock.Mock(return_value="gfx942"),
    )


@contextlib.contextmanager
def _recorded_selections():
    """Yield a list that receives the backends dispatch recorded."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "trace.jsonl")
        selections = []
        with (
            mock.patch.dict(
                os.environ, {dispatch_verification.DISPATCH_TRACE_ENV: path}
            ),
            mock.patch.object(dispatch_verification, "_recorded", set()),
        ):
            yield selections
        selections.extend(dispatch_verification.read_selections(path, "mha_fwd"))


class _Table(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        policy.clear_mha_fwd_tuning_cache()
        self.addCleanup(policy.clear_mha_fwd_tuning_cache)

    def use(self, rows):
        """Point dispatch at a table holding ``rows``."""
        path = os.path.join(self.directory, MHA_FWD_TUNED_CSV)
        with open(path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(MHA_FWD_RUNTIME_CSV_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
        patch = mock.patch.object(policy, "_mha_fwd_table_path", return_value=path)
        patch.start()
        self.addCleanup(patch.stop)
        return path


class TestMhaFwdKey(unittest.TestCase):
    def test_a_call_and_its_row_share_one_key(self):
        spellings = {
            "plain": (_key_args(), _problem_row()),
            "int and list call": (
                _key_args(
                    dropout_p=0,
                    logits_soft_cap=0,
                    causal=0,
                    return_lse=0,
                    window_size=[-1, -1, 0],
                ),
                _problem_row(),
            ),
            "bf16 row": (_key_args(), _problem_row(dtype="bf16")),
        }
        for name, (call, row) in spellings.items():
            with self.subTest(name):
                self.assertEqual(
                    policy._mha_fwd_tuning_key(HARDWARE_CELLS, **call),
                    MhaFwdProblem.from_mapping(row).key(),
                )


class TestMhaFwdTable(_Table):
    def test_a_bad_row_is_skipped_and_the_rest_still_load(self):
        rows = [
            _tuned_row("cutlass", total_q=2048, max_seqlen_q=2048),
            _tuned_row("ck", nhead_k=5),
            _tuned_row("ck", dtype="bfloat17"),
            _tuned_row("ck", mode="batch"),
            _tuned_row("ck", has_physical_padding=1),
            _tuned_row("ck", has_block_table=1),
            _tuned_row("asm_v3"),
        ]
        with self.assertLogs("aiter", level="WARNING") as logs:
            table = policy._load_mha_fwd_tuning_table(self.use(rows))
        self.assertEqual(list(table.values()), ["asm_v3"])
        self.assertEqual(len(logs.records), 6)

    def test_a_duplicate_key_keeps_the_first_row(self):
        rows = [_tuned_row("ck"), _tuned_row("asm_v3")]
        with self.assertLogs("aiter", level="WARNING"):
            table = policy._load_mha_fwd_tuning_table(self.use(rows))
        self.assertEqual(list(table.values()), ["ck"])

    def test_a_missing_file_named_by_the_env_var_warns(self):
        path = os.path.join(self.directory, "typo.csv")
        with (
            mock.patch.dict(os.environ, {MHA_FWD_CONFIG_ENV: path}),
            self.assertLogs("aiter", level="WARNING") as logs,
        ):
            self.assertEqual(policy._load_mha_fwd_tuning_table(path), {})
        self.assertIn("does not exist", logs.output[0])


class TestMhaFwdLookup(_Table):
    def test_an_empty_table_returns_before_reading_the_hardware_or_the_key(self):
        self.use([])
        with (
            mock.patch.object(policy, "get_tuning_hardware") as hardware,
            mock.patch.object(policy, "_mha_fwd_tuning_key") as key,
        ):
            self.assertIsNone(policy.get_mha_fwd_tuned_backend(**_key_args()))
        hardware.assert_not_called()
        key.assert_not_called()

    def test_compiled_callers_auto_select_without_a_graph_break(self):
        """Traced, the key read sizes dynamo makes symbolic: fullgraph failed
        on the second token count once a row matched the GPU and head dims."""
        self.use([_tuned_row("ck")])

        def call(q, k, v):
            backend = policy.get_mha_fwd_tuned_backend(**_key_args(q=q, k=k, v=v))
            return q.sum() if backend is None else q.mean()

        with _on_hardware():
            self.assertEqual(policy.get_mha_fwd_tuned_backend(**_key_args()), "ck")
            for dynamic in (None, True):
                with self.subTest(dynamic=dynamic):
                    torch._dynamo.reset()
                    compiled = torch.compile(
                        call, fullgraph=True, dynamic=dynamic, backend="eager"
                    )
                    for total_q in (4096, 2048):
                        q = torch.ones((total_q, 12, 192), dtype=torch.bfloat16)
                        args = _key_args(q=q)
                        torch.testing.assert_close(
                            compiled(q, args["k"], args["v"]), q.sum()
                        )

    def test_a_hit_returns_the_pinned_backend(self):
        self.use([_tuned_row("opus")])
        with _on_hardware():
            self.assertEqual(policy.get_mha_fwd_tuned_backend(**_key_args()), "opus")

    def test_rows_for_other_hardware_or_head_dims_skip_the_key(self):
        for row in (
            _tuned_row("ck", hdim_q=128, hdim_v=128),
            _tuned_row("ck", gpu_model="mi300x"),
        ):
            with self.subTest(row=row):
                policy.clear_mha_fwd_tuning_cache()
                self.use([row])
                with (
                    _on_hardware(),
                    mock.patch.object(policy, "_mha_fwd_tuning_key") as key,
                ):
                    self.assertIsNone(policy.get_mha_fwd_tuned_backend(**_key_args()))
                key.assert_not_called()

    def test_padded_or_paged_calls_skip_the_key(self):
        self.use([_tuned_row("opus")])
        for arg in ("block_table", "cu_seqlens_q_padded", "cu_seqlens_k_padded"):
            with (
                self.subTest(arg),
                _on_hardware(),
                mock.patch.object(policy, "_mha_fwd_tuning_key") as key,
            ):
                call = _key_args(**{arg: torch.zeros(2, dtype=torch.int32)})
                self.assertIsNone(policy.get_mha_fwd_tuned_backend(**call))
            key.assert_not_called()


class TestMhaVarlenForward(unittest.TestCase):
    def _varlen_forward(self, entry="fmha_v3_varlen_fwd", **overrides):
        """Dispatch once on gfx942 with ``entry`` mocked; return the selections."""
        q, k, v, cu_q, cu_k = _dummy_varlen_tensors()
        with (
            mock.patch.object(mha, "get_gfx", return_value="gfx942"),
            mock.patch.object(
                mha, entry, return_value=tuple(torch.empty(1) for _ in range(4))
            ) as backend,
            _recorded_selections() as selections,
        ):
            mha._flash_attn_varlen_forward(
                q, k, v, cu_q, cu_k, None, None, 8, 16, 0, 0.0, 0.125, False,
                **overrides,
            )  # fmt: skip
        backend.assert_called_once()
        return selections

    def test_no_row_runs_auto_select(self):
        selections = self._varlen_forward(selected_backend=None)
        self.assertEqual(selections, [{"backend": "asm_v3"}])

    def test_a_ck_row_displaces_asm_auto_select(self):
        selections = self._varlen_forward(entry="mha_varlen_fwd", selected_backend="ck")
        self.assertEqual(selections, [{"backend": "ck"}])

    def test_a_row_the_backend_cannot_serve_runs_auto_select_not_ck(self):
        # opus only serves gfx950, so on gfx942 the row is declined and the
        # call takes what auto-select takes there (asm_v3), not CK.
        mha.warn_declined_mha_fwd_row.cache_clear()
        with self.assertLogs("aiter", level="WARNING") as logs:
            self._varlen_forward(selected_backend="opus")
        self.assertIn("max_seqlen_k=16, nhead_q=12, hdim_q=192", logs.output[0])

    def test_a_declined_row_warns_once_per_shape(self):
        mha.warn_declined_mha_fwd_row.cache_clear()
        with self.assertLogs("aiter", level="WARNING") as logs:
            for hdim_q in (192, 192, 128):
                mha.warn_declined_mha_fwd_row("opus", hdim_q=hdim_q)
        self.assertEqual(len(logs.records), 2)


class TestFlashAttnVarlenFunc(_Table):
    def _route(self, backend, flydsl="flydsl", **kwargs):
        q, k, v, cu_q, cu_k = _dummy_varlen_tensors()
        with (
            mock.patch.object(mha, "get_mha_fwd_tuned_backend", return_value=backend),
            mock.patch.object(
                mha.FlashAttnVarlenFunc, "apply", return_value="aiter"
            ) as apply,
            mock.patch(
                "aiter.ops.flydsl.fmha_kernels.flydsl_flash_attn_varlen_fwd",
                return_value=flydsl,
            ),
        ):
            result = mha.flash_attn_varlen_func(q, k, v, cu_q, cu_k, 8, 16, **kwargs)
        return result, apply

    def test_a_pinned_backend_reaches_the_forward(self):
        result, apply = self._route("asm_v3")
        self.assertEqual(result, "aiter")
        self.assertEqual(apply.call_args.args[-1], "asm_v3")

    def test_the_public_op_finds_its_row(self):
        # Batch 2 and unequal max_seqlen_q/k: an off-by-one batch or swapped
        # lengths at the call site would miss the row.
        self.use(
            [
                _tuned_row(
                    "opus",
                    batch=2,
                    total_q=8,
                    total_k=16,
                    max_seqlen_q=4,
                    max_seqlen_k=8,
                )
            ]
        )
        q = torch.empty((8, 12, 192), dtype=torch.bfloat16)
        k = torch.empty((16, 12, 192), dtype=torch.bfloat16)
        v = torch.empty((16, 12, 128), dtype=torch.bfloat16)
        cu_q = torch.tensor([0, 4, 8], dtype=torch.int32)
        cu_k = torch.tensor([0, 8, 16], dtype=torch.int32)
        with (
            _on_hardware(),
            mock.patch.object(mha.FlashAttnVarlenFunc, "apply") as apply,
        ):
            mha.flash_attn_varlen_func(q, k, v, cu_q, cu_k, 4, 8)
        self.assertEqual(apply.call_args.args[-1], "opus")

    def test_a_declined_flydsl_row_continues_as_auto_select(self):
        for name, flydsl, available in (
            ("declined", None, True),
            ("not installed", "flydsl", False),
        ):
            with self.subTest(name):
                mha.warn_declined_mha_fwd_row.cache_clear()
                with (
                    mock.patch.object(mha, "_FLYDSL_AVAILABLE", available),
                    self.assertLogs("aiter", level="WARNING"),
                ):
                    result, apply = self._route("flydsl", flydsl=flydsl)
                self.assertEqual(result, "aiter")
                self.assertIsNone(apply.call_args.args[-1])


if __name__ == "__main__":
    unittest.main()
