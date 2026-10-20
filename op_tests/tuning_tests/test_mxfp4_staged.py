# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 staged search tests: CLI, planning, screening and worker supervision.

CPU only; GPU work and real shape workers are blocked.
"""

import argparse
import multiprocessing
import os
import sys
import threading
import weakref
from collections.abc import Callable, Iterator
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from op_tests.tuning_tests._mxfp4_test_process import require_gpu_hidden_process

require_gpu_hidden_process()

from op_tests.tuning_tests import test_mxfp4_tuner as lifecycle
from op_tests.tuning_tests.test_mxfp4_tuner import (
    KEYS,
    RESULTS,
    devices,
    moe,
    prohibit_real_shape_workers_and_gpu_data,  # noqa: F401 - imported autouse fixture
    shape,
)

# Captured before the autouse fixture replaces it with a GPU-work guard.
RUN_CANDIDATE = moe.Mxfp4FlydslTuner._run_candidate


def parse_cli(
    monkeypatch: "pytest.MonkeyPatch", *options: "str"
) -> "argparse.Namespace":
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    monkeypatch.setattr(sys, "argv", ["gemm_moe_tune", *options])
    return tuner.parse_args()


def test_cli_accepts_staged_with_default_screening_topk(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    args = parse_cli(monkeypatch, "--mxfp4-flydsl", "--mxfp4-search-mode", "staged")
    assert args.mxfp4_search_mode == "staged"
    assert args.mxfp4_screen_topk == 2


@pytest.mark.parametrize("screen_topk", [1, 2, 7])
def test_cli_accepts_any_positive_staged_screening_topk(
    monkeypatch: "pytest.MonkeyPatch", screen_topk: "int"
) -> "None":
    args = parse_cli(
        monkeypatch,
        "--mxfp4-flydsl",
        "--mxfp4-search-mode",
        "staged",
        "--mxfp4-screen-topk",
        str(screen_topk),
    )
    assert args.mxfp4_screen_topk == screen_topk


@pytest.mark.parametrize("mode", [None, "prune", "full"])
def test_cli_preserves_default_prune_and_full(
    monkeypatch: "pytest.MonkeyPatch", mode: "str | None"
) -> "None":
    options = ["--mxfp4-flydsl"]
    if mode is not None:
        options.extend(["--mxfp4-search-mode", mode])
    args = parse_cli(monkeypatch, *options)
    assert args.mxfp4_search_mode == (mode or "prune")
    assert args.mxfp4_screen_topk == 2


@pytest.mark.parametrize("mode", [None, "prune", "full"])
def test_cli_rejects_explicit_screening_topk_without_explicit_staged(
    monkeypatch: "pytest.MonkeyPatch",
    capsys: "pytest.CaptureFixture[str]",
    mode: "str | None",
) -> "None":
    options = ["--mxfp4-flydsl", "--mxfp4-screen-topk", "2"]
    if mode is not None:
        options.extend(["--mxfp4-search-mode", mode])
    with pytest.raises(SystemExit) as failure:
        parse_cli(monkeypatch, *options)
    assert failure.value.code == 2
    assert "requires explicit --mxfp4-search-mode staged" in capsys.readouterr().err


@pytest.mark.parametrize("screen_topk", ["0", "-1", "2.5"])
def test_cli_rejects_nonpositive_or_noninteger_screening_topk(
    monkeypatch: "pytest.MonkeyPatch", screen_topk: "str"
) -> "None":
    with pytest.raises(SystemExit) as failure:
        parse_cli(
            monkeypatch,
            "--mxfp4-flydsl",
            "--mxfp4-search-mode",
            "staged",
            "--mxfp4-screen-topk",
            screen_topk,
        )
    assert failure.value.code == 2


@pytest.mark.parametrize("options", [[], ["--mxfp4-flydsl", "--grouped-gemm"]])
def test_cli_retains_tuner_and_grouped_combination_checks(
    monkeypatch: "pytest.MonkeyPatch", options: "list[str]"
) -> "None":
    with pytest.raises(SystemExit) as failure:
        parse_cli(monkeypatch, *options, "--mxfp4-search-mode", "staged")
    assert failure.value.code == 2


def test_cli_rejects_fast_scan_with_staged(
    monkeypatch: "pytest.MonkeyPatch", capsys: "pytest.CaptureFixture[str]"
) -> "None":
    with pytest.raises(SystemExit) as failure:
        parse_cli(
            monkeypatch,
            "--mxfp4-flydsl",
            "--fast-scan",
            "--mxfp4-search-mode",
            "staged",
        )
    assert failure.value.code == 2
    assert "--fast-scan applies to prune/full search" in capsys.readouterr().err


@pytest.mark.parametrize("mode", [None, "prune", "full"])
def test_cli_accepts_fast_scan_with_prune_and_full(
    monkeypatch: "pytest.MonkeyPatch", mode: "str | None"
) -> "None":
    options = ["--mxfp4-flydsl", "--fast-scan"]
    if mode is not None:
        options.extend(["--mxfp4-search-mode", mode])
    args = parse_cli(monkeypatch, *options)
    assert args.fast_scan
    assert args.mxfp4_search_mode == (mode or "prune")


def test_tune_rejects_fast_scan_with_staged_before_any_work() -> "None":
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    args = SimpleNamespace(fast_scan=True, mxfp4_search_mode="staged", mp=1)
    with pytest.raises(ValueError, match="--fast-scan applies to prune/full"):
        tuner.tune(moe.pd.DataFrame([shape()]), None, args)


def pair_names(rows: "list[dict[str, object]]") -> "list[tuple[str, str]]":
    return [(row["kernelName1"], row["kernelName2"]) for row in rows]


def test_staged_plan_projects_real_pruned_pairs_and_supports_i384_without_gpu() -> (
    "None"
):
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    row = shape(512, inter_dim=384)
    pruned = tuner._candidate_rows(row, full_search=False)
    full = tuner._candidate_rows(row, full_search=True)
    plan = tuner._staged_candidate_plan(row)
    assert pair_names(pair.candidate_row for pair in plan.pairs) == pair_names(pruned)
    assert {pair.candidate_row["block_m"] for pair in plan.pairs} == {32, 64}
    assert {candidate["block_m"] for candidate in full} == {16, 32, 64, 128}
    assert all("layout_afp4_wfp4_bf16_" in name for _, name in pair_names(pruned))
    assert any("x128_atomic" in name for _, name in pair_names(pruned))


@pytest.mark.parametrize(
    "m_est,expected_bms",
    [
        (3, {16}),
        (4, {16, 32}),
        (15, {16, 32}),
        (16, {32, 64}),
        (32, {32, 64}),
        (33, {32, 64}),
        (63, {32, 64}),
        (64, {32, 64, 128}),
        (65, {32, 64, 128}),
        (128, {32, 64, 128}),
        (129, {64, 128}),
    ],
)
def test_staged_keeps_main_m_est_boundary_rules(
    m_est: "int", expected_bms: "set[int]"
) -> "None":
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    row = shape(m_est, expert=4, topk=4)
    pruned = tuner._candidate_rows(row, full_search=False)
    plan = tuner._staged_candidate_plan(row)
    rows = [pair.candidate_row for pair in plan.pairs]
    assert pair_names(rows) == pair_names(pruned)
    assert {row["block_m"] for row in rows} == expected_bms
    g1_names = {row["kernelName1"] for row in rows}
    if m_est > 32:
        bm32 = [moe._parse_mxfp4_g1_kname(name) for name in g1_names if "_32x" in name]
        assert all(not config["use_nt"] for config in bm32)
        assert all(
            config["num_waves"] == 4 and config["k_wave"] == 1 for config in bm32
        )
    if m_est > 64:
        bm64 = [moe._parse_mxfp4_g1_kname(name) for name in g1_names if "_64x" in name]
        assert all(not config["use_nt"] for config in bm64)


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"gate_mode": "GateMode.INTERLEAVE"}, "gate_mode"),
        ({"act_type": "ActivationType.Gelu"}, "act_type"),
        ({"dtype": "torch.float16"}, "dtype"),
        ({"q_dtype_a": "torch.float8_e4m3fn"}, "q_dtype_a"),
        ({"q_dtype_w": "torch.float8_e4m3fn"}, "q_dtype_w"),
        ({"q_type": "QuantType.per_Tensor"}, "q_type"),
        ({"use_g1u1": 0}, "use_g1u1"),
        ({"doweight_stage1": 1}, "doweight_stage1"),
    ],
)
def test_staged_rejects_actual_shape_contract_before_enumeration(
    monkeypatch: "pytest.MonkeyPatch", updates: "dict[str, object]", reason: "str"
) -> "None":
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)

    def forbid_enumeration(*args: "object", **kwargs: "object") -> "None":
        pytest.fail("unsupported shape entered candidate enumeration")

    monkeypatch.setattr(tuner, "_candidate_rows", forbid_enumeration)
    with pytest.raises(ValueError, match=reason):
        tuner._staged_candidate_plan(shape(**updates))


def plan_fixture(
    monkeypatch: "pytest.MonkeyPatch",
    row: "dict[str, object]",
    pairs: "list[tuple[int, str, str]] | None",
) -> "moe.StagedSearchPlan":
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    candidates = [
        tuner._candidate_row(row, bm, gemm1, gemm2) for bm, gemm1, gemm2 in pairs
    ]

    def enumerated(
        shape_row: "dict[str, object]", full_search: "bool" = False
    ) -> "list[dict[str, object]]":
        assert full_search is False
        return candidates

    monkeypatch.setattr(tuner, "_candidate_rows", enumerated)
    return tuner._staged_candidate_plan(row)


def test_staged_preserves_complete_names_and_groups_reduce_with_scatter(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    g1 = "flydsl_mxmoe_g1_a4w4_128x128x256_swiglu"
    g1_xcd = g1 + "_xcd2"
    atomic = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_atomic_sbm128"
    atomic_nt = (
        "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_atomic_persist_nt_sbm128"
    )
    reduce = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_reduce_sbm128"
    scatter = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_scatter_sbm128"
    names = [
        (g1, atomic),
        (g1, atomic_nt),
        (g1_xcd, atomic),
        (g1, reduce),
        (g1, scatter),
    ]
    plan = plan_fixture(monkeypatch, shape(2048), [(128, *pair) for pair in names])
    assert pair_names(pair.candidate_row for pair in plan.pairs) == names
    assert len({pair.key for pair in plan.pairs}) == 5
    assert len(plan.gemm1_groups) == 1
    assert len(plan.gemm1_groups[0].candidates) == 2
    grouped = [
        pair_names(pair.candidate_row for pair in group.pairs)
        for group in plan.pipeline_groups
    ]
    assert grouped == [names[:3], names[3:]]
    assert len(plan.gemm2_groups) == 2


@pytest.mark.parametrize(
    "bm,g1,g2,reason",
    [
        (
            32,
            "flydsl_mxmoe_g1_a4w4_32x128x256_swiglu",
            "flydsl_mxmoe_g2_a4w4_32x256x256_atomic",
            "layout-v2",
        ),
        (
            64,
            "flydsl_mxmoe_g1_a4w4_64x128x256_swiglu",
            "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm64",
            "block_m",
        ),
        (
            32,
            "flydsl_mxmoe_g1_a4w4_32x128x256_swiglu",
            "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm64",
            "sort block_m",
        ),
        (
            32,
            "flydsl_mxmoe_g1_a4w4_32x128x256_f16in_swiglu",
            "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm32",
            "input quantization",
        ),
        (
            16,
            "flydsl_mxmoe_g1_a4w4_16x128x256_nt_swiglu",
            "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16",
            "input quantization",
        ),
        (
            32,
            "flydsl_mxmoe_g1_a4w4_32x128x256",
            "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm32",
            "activation",
        ),
        (
            32,
            "flydsl_mxmoe_g1_a4w4_32x128x256_swiglu_fp8out",
            "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm32",
            "out_dtype",
        ),
        (
            32,
            "flydsl_mxmoe_g1_a4w4_32x128x256_swiglu",
            "flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_atomic_sbm32",
            "a_dtype",
        ),
        (
            32,
            "flydsl_mxmoe_g1_a4w4_32x128x256_swiglu",
            "flydsl_moe2_layout_afp4_wfp8_bf16_t32x128x128_atomic_sbm32",
            "b_dtype",
        ),
        (
            32,
            "flydsl_mxmoe_g1_a4w4_32x128x256_swiglu",
            "flydsl_moe2_layout_afp4_wfp4_fp8_t32x128x128_atomic_sbm32",
            "out_dtype",
        ),
    ],
)
def test_staged_rejects_incompatible_admitted_pair_contract(
    monkeypatch: "pytest.MonkeyPatch", bm: "int", g1: "str", g2: "str", reason: "str"
) -> "None":
    with pytest.raises(ValueError, match=reason):
        plan_fixture(monkeypatch, shape(), [(bm, g1, g2)])


G1_A = "flydsl_mxmoe_g1_a4w4_32x128x256_swiglu"
G1_B = "flydsl_mxmoe_g1_a4w4_32x256x256_swiglu"
G1_C = "flydsl_mxmoe_g1_a4w4_32x128x256_swiglu_xcd2"
G2_A = "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm32"
G2_B = "flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x128_atomic_sbm32"
G2_C = "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_persist_sbm32"


def test_staged_shape_validates_only_successful_shortlist_product(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    row = shape(128)
    candidates = [
        tuner._candidate_row(row, 32, gemm1, gemm2)
        for gemm1 in (G1_A, G1_B, G1_C)
        for gemm2 in (G2_A, G2_B, G2_C)
    ]
    monkeypatch.setattr(
        tuner, "_candidate_rows", lambda row, full_search=False: candidates
    )
    monkeypatch.setattr(
        tuner, "_check_reference_consumer", lambda *args: True, raising=False
    )
    monkeypatch.setattr(
        tuner, "_check_reference_pair", lambda *args: True, raising=False
    )
    monkeypatch.setattr(
        tuner,
        "_screen_gemm1",
        lambda row, name, args: {G1_A: 3.0, G1_B: 1.0, G1_C: 2.0}[name],
        raising=False,
    )
    monkeypatch.setattr(
        tuner,
        "_screen_gemm2",
        lambda row, producer, name, args: {G2_A: 2.0, G2_B: 3.0, G2_C: 1.0}[name],
        raising=False,
    )
    attempts = []

    def validate(
        row: "dict[str, object]",
        candidate: "dict[str, object]",
        args: "SimpleNamespace",
    ) -> "float":
        attempts.append((candidate["kernelName1"], candidate["kernelName2"]))
        candidate.update(us=9.5, us1=9.5, us2=0)
        return 9.5

    monkeypatch.setattr(tuner, "_run_candidate", validate)
    result = tuner._tune_one_shape(
        row, SimpleNamespace(timeout=0, mxfp4_search_mode="staged", mxfp4_screen_topk=1)
    )
    assert attempts == [(G1_B, G2_C)]
    assert result["kernelName1"] == G1_B and result["kernelName2"] == G2_C
    assert result["us"] == result["us1"] == 9.5 and result["us2"] == 0


def controlled_search(
    monkeypatch: "pytest.MonkeyPatch",
    *,
    pairs: "list[tuple[int, str, str]] | None" = None,
    screen_topk: "int" = 1,
    row: "dict[str, object] | None" = None,
) -> "SimpleNamespace":
    """Provide returned GPU-boundary outcomes, retaining the real search policy."""
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    row = row or shape(128)
    if pairs is None:
        pairs = [(32, g1, g2) for g1 in (G1_B, G1_A, G1_C) for g2 in (G2_B, G2_C, G2_A)]
    state = SimpleNamespace(
        events=[],
        stage1={G1_B: 1.0, G1_A: 3.0, G1_C: 2.0},
        stage2={G2_B: 3.0, G2_C: 1.0, G2_A: 2.0},
        validation={},
        consumers={},
        bootstrap={},
    )
    args = SimpleNamespace(
        timeout=0,
        mxfp4_search_mode="staged",
        mxfp4_screen_topk=screen_topk,
        errRatio=0.1,
        warmup=3,
        iters=5,
    )

    def returned(outcome: "float | bool | Exception") -> "float | bool":
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def enumerate_pairs(
        actual_row: "dict[str, object]", full_search: "bool" = False
    ) -> "list[dict[str, object]]":
        state.events.append(("enumerate", actual_row["token"], full_search))
        assert full_search is False
        return [tuner._candidate_row(actual_row, bm, g1, g2) for bm, g1, g2 in pairs]

    def consumer(
        actual_row: "dict[str, object]", name: "str", actual_args: "SimpleNamespace"
    ) -> "bool":
        state.events.append(("consumer", name))
        assert actual_args is args
        return returned(state.consumers.get(name, True))

    def bootstrap(
        actual_row: "dict[str, object]",
        producer: "object",
        consumer: "object",
        actual_args: "SimpleNamespace",
    ) -> "bool":
        state.events.append(("bootstrap", producer, consumer))
        assert actual_args is args
        return returned(state.bootstrap.get(producer, True))

    def screen_g1(
        actual_row: "dict[str, object]", name: "str", actual_args: "SimpleNamespace"
    ) -> "float":
        state.events.append(("g1", name))
        assert actual_args is args
        return returned(state.stage1.get(name, 1.0))

    def screen_g2(
        actual_row: "dict[str, object]",
        producer: "object",
        name: "str",
        actual_args: "SimpleNamespace",
    ) -> "float":
        state.events.append(("g2", producer, name))
        assert actual_args is args
        return returned(state.stage2.get(name, 1.0))

    def validate(
        actual_row: "dict[str, object]",
        candidate: "dict[str, object]",
        actual_args: "SimpleNamespace",
        **options: "object",
    ) -> "float":
        pair = candidate["kernelName1"], candidate["kernelName2"]
        state.events.append(("validate", *pair))
        assert actual_args is args
        # Prune passes its fast-scan options explicitly; without fast-scan all are None.
        assert all(value is None for value in options.values())
        us = returned(state.validation.get(pair, 10.0))
        candidate.update(us=us, us1=us, us2=0)
        return us

    monkeypatch.setattr(tuner, "_candidate_rows", enumerate_pairs)
    for name, operation in (
        ("_check_reference_consumer", consumer),
        ("_check_reference_pair", bootstrap),
        ("_screen_gemm1", screen_g1),
        ("_screen_gemm2", screen_g2),
        ("_run_candidate", validate),
    ):
        monkeypatch.setattr(tuner, name, operation, raising=False)
    state.run = lambda actual_row=None: tuner._tune_one_shape(actual_row or row, args)
    state.pairs = [(g1, g2) for _, g1, g2 in pairs]
    state.args = args
    return state


def events(state: "SimpleNamespace", kind: "str") -> "list[tuple[object, ...]]":
    return [event[1:] for event in state.events if event[0] == kind]


@pytest.mark.parametrize("screen_topk,count", [(1, 1), (2, 4), (7, 9)])
def test_staged_screening_topk_caps_shortlist_product(
    monkeypatch: "pytest.MonkeyPatch", screen_topk: "int", count: "int"
) -> "None":
    state = controlled_search(monkeypatch, screen_topk=screen_topk)
    state.run()
    attempted = events(state, "validate")
    assert len(attempted) == len(set(attempted)) == count
    if screen_topk == 2:
        assert attempted == [(G1_B, G2_C), (G1_B, G2_A), (G1_C, G2_C), (G1_C, G2_A)]


def test_staged_exact_ties_keep_only_k_in_admitted_order(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    state = controlled_search(monkeypatch, screen_topk=2)
    state.stage1 = dict.fromkeys((G1_A, G1_B, G1_C), 1.0)
    state.stage2 = dict.fromkeys((G2_A, G2_B, G2_C), 1.0)
    state.run()
    assert events(state, "validate") == [
        (G1_B, G2_B),
        (G1_B, G2_C),
        (G1_A, G2_B),
        (G1_A, G2_C),
    ]


@pytest.mark.parametrize(
    "invalid",
    [
        float("nan"),
        float("inf"),
        0.0,
        -1.0,
        RuntimeError("screen fixture"),
        TimeoutError("screen soft timeout"),
    ],
)
def test_staged_excludes_failed_screening_scores_when_a_shortlist_is_valid(
    monkeypatch: "pytest.MonkeyPatch", invalid: "float | Exception"
) -> "None":
    state = controlled_search(monkeypatch, screen_topk=7)
    state.stage1 = {G1_A: invalid, G1_B: 1.0, G1_C: invalid}
    state.stage2 = {G2_A: invalid, G2_B: invalid, G2_C: 1.0}
    state.run()
    assert events(state, "validate") == [(G1_B, G2_C)]


def test_staged_bootstrap_prefers_admitted_reference_and_keeps_it_untimed(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    state = controlled_search(monkeypatch)
    result = state.run()
    assert events(state, "consumer") == [(G2_A,)]
    assert events(state, "bootstrap") == [(G1_A, G2_A)]
    assert {producer for producer, _ in events(state, "g2")} == {G1_A}
    assert events(state, "validate") == [(G1_B, G2_C)]
    assert result["us"] == 10.0


def test_staged_reference_retries_only_remaining_admitted_candidates(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    state = controlled_search(monkeypatch)
    state.consumers = {
        G2_A: RuntimeError("preferred consumer"),
        G2_B: False,
        G2_C: True,
    }
    state.bootstrap = {
        G1_A: TimeoutError("preferred producer"),
        G1_B: False,
        G1_C: True,
    }
    state.run()
    assert events(state, "consumer") == [(G2_A,), (G2_B,), (G2_C,)]
    assert events(state, "bootstrap") == [(G1_A, G2_C), (G1_B, G2_C), (G1_C, G2_C)]
    assert {producer for producer, _ in events(state, "g2")} == {G1_C}
    assert events(state, "enumerate") == [(128, False)]


@pytest.mark.parametrize("failure", ["consumer", "producer", "g1", "g2"])
def test_staged_exhausts_admitted_pairs_after_independent_fixture_failure(
    monkeypatch: "pytest.MonkeyPatch", failure: "str"
) -> "None":
    state = controlled_search(monkeypatch)
    if failure == "consumer":
        state.consumers = dict.fromkeys((G2_A, G2_B, G2_C), False)
    elif failure == "producer":
        state.bootstrap = dict.fromkeys((G1_A, G1_B, G1_C), False)
    elif failure == "g1":
        state.stage1 = dict.fromkeys((G1_A, G1_B, G1_C), float("nan"))
    else:
        state.stage2 = dict.fromkeys((G2_A, G2_B, G2_C), TimeoutError("all G2 screens"))
    state.validation[(G1_C, G2_A)] = 0.25
    result = state.run()
    assert events(state, "validate") == state.pairs
    assert (result["kernelName1"], result["kernelName2"], result["us"]) == (
        G1_C,
        G2_A,
        0.25,
    )
    assert events(state, "enumerate") == [(128, False)]
    if failure in ("consumer", "producer"):
        assert events(state, "g1") == events(state, "g2") == []


@pytest.mark.parametrize(
    "invalid",
    [
        RuntimeError("incorrect output"),
        ValueError("compile failure"),
        TimeoutError("validation soft timeout"),
        float("inf"),
        float("nan"),
        0.0,
        -1.0,
    ],
)
def test_staged_invalid_shortlist_expands_without_rerunning_returned_outcomes(
    monkeypatch: "pytest.MonkeyPatch", invalid: "float | Exception"
) -> "None":
    state = controlled_search(monkeypatch)
    state.validation[(G1_B, G2_C)] = invalid
    state.validation[(G1_B, G2_B)] = 4.0
    state.validation[(G1_C, G2_A)] = 0.25
    result = state.run()
    assert events(state, "validate") == [
        (G1_B, G2_C),
        *[pair for pair in state.pairs if pair != (G1_B, G2_C)],
    ]
    assert len(set(events(state, "validate"))) == 9
    assert (result["kernelName1"], result["kernelName2"], result["us"]) == (
        G1_C,
        G2_A,
        0.25,
    )


def test_staged_screening_failed_candidates_can_win_during_fallback(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    state = controlled_search(monkeypatch)
    state.stage1[G1_C] = RuntimeError("producer screen")
    state.stage2[G2_A] = float("inf")
    state.validation[(G1_B, G2_C)] = RuntimeError("shortlist numerical failure")
    state.validation[(G1_C, G2_A)] = 0.1
    result = state.run()
    assert len(events(state, "validate")) == len(set(events(state, "validate"))) == 9
    assert (result["kernelName1"], result["kernelName2"], result["us"]) == (
        G1_C,
        G2_A,
        0.1,
    )


def test_staged_no_winner_and_empty_universe_return_invalid(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    state = controlled_search(monkeypatch)
    state.validation = dict.fromkeys(state.pairs, RuntimeError("all pairs incorrect"))
    failed = state.run()
    assert failed["us"] < 0
    assert len(events(state, "validate")) == 9
    empty = controlled_search(monkeypatch, pairs=[])
    assert empty.run()["us"] < 0
    assert (
        events(empty, "consumer")
        == events(empty, "g1")
        == events(empty, "validate")
        == []
    )


def test_staged_scores_and_pair_cache_are_fresh_for_each_shape(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    state = controlled_search(monkeypatch)
    state.validation[(G1_B, G2_C)] = RuntimeError("first shape incorrect")
    state.run()
    state.events.clear()
    state.stage1 = {G1_A: 0.1, G1_B: 2.0, G1_C: 3.0}
    state.stage2 = {G2_A: 0.1, G2_B: 2.0, G2_C: 3.0}
    state.validation[(G1_A, G2_A)] = 0.2
    result = state.run(shape(256))
    assert events(state, "enumerate") == [(256, False)]
    assert len(events(state, "g1")) == len(events(state, "g2")) == 3
    assert events(state, "validate") == [(G1_A, G2_A)]
    assert result["token"] == 256 and result["us"] == 0.2


def test_staged_reuses_g1_ranking_across_atomic_reduce_and_scatter_groups(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    g1 = "flydsl_mxmoe_g1_a4w4_128x128x256_swiglu"
    other = "flydsl_mxmoe_g1_a4w4_128x256x256_swiglu_xcd2"
    atomic = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_atomic_sbm128"
    reduce = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_reduce_sbm128"
    scatter = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_scatter_sbm128"
    state = controlled_search(
        monkeypatch,
        pairs=[
            (128, producer, consumer)
            for producer in (other, g1)
            for consumer in (atomic, reduce, scatter)
        ],
        row=shape(2048),
    )
    state.stage1 = {other: 1.0, g1: 2.0}
    state.stage2 = {atomic: 4.0, reduce: 2.0, scatter: 1.0}
    state.validation = {(other, atomic): 4.0, (other, scatter): 0.5}
    result = state.run()
    assert events(state, "g1") == [(other,), (g1,)]
    assert events(state, "g2") == [(g1, atomic), (g1, reduce), (g1, scatter)]
    assert events(state, "validate") == [(other, atomic), (other, scatter)]
    assert result["kernelName2"] == scatter and result["us"] == 0.5


def test_staged_all_atomic_reference_failures_preserve_valid_nonatomic_pair(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    reduce = "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_reduce_sbm32"
    state = controlled_search(
        monkeypatch,
        pairs=[(32, g1, g2) for g1 in (G1_B, G1_A) for g2 in (G2_B, G2_A, reduce)],
    )
    state.consumers = {G2_A: False, G2_B: RuntimeError("atomic reference fixture")}
    state.validation = dict.fromkeys(state.pairs, RuntimeError("normal pair incorrect"))
    state.validation[(G1_A, reduce)] = 0.4
    result = state.run()
    assert (
        events(state, "bootstrap") == events(state, "g1") == events(state, "g2") == []
    )
    assert events(state, "validate") == [
        (G1_B, G2_B),
        (G1_B, G2_A),
        (G1_A, G2_B),
        (G1_A, G2_A),
        (G1_B, reduce),
        (G1_A, reduce),
    ]
    assert (result["kernelName1"], result["kernelName2"], result["us"]) == (
        G1_A,
        reduce,
        0.4,
    )


def test_staged_g2_failure_exhausts_only_affected_compatibility_group(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    reduce = "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_reduce_sbm32"
    state = controlled_search(
        monkeypatch,
        pairs=[(32, g1, g2) for g1 in (G1_B, G1_A) for g2 in (G2_B, G2_A, reduce)],
    )
    state.stage2 = {G2_B: float("nan"), G2_A: 0.0, reduce: 1.0}
    state.validation[(G1_A, G2_A)] = 0.3
    result = state.run()
    assert events(state, "validate") == [
        (G1_B, G2_B),
        (G1_B, G2_A),
        (G1_A, G2_B),
        (G1_A, G2_A),
        (G1_B, reduce),
    ]
    assert result["kernelName1"] == G1_A and result["us"] == 0.3


def test_staged_bootstrap_failure_is_local_to_bm_and_later_groups_still_screen(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    g1_64 = "flydsl_mxmoe_g1_a4w4_64x128x256_swiglu"
    g2_64 = "flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_atomic_sbm64"
    state = controlled_search(
        monkeypatch, pairs=[(32, G1_B, G2_B), (32, G1_A, G2_A), (64, g1_64, g2_64)]
    )
    state.consumers = {G2_B: False, G2_A: False}
    state.validation[(G1_A, G2_A)] = 0.25
    result = state.run()
    assert events(state, "g1") == [(g1_64,)]
    assert events(state, "g2") == [(g1_64, g2_64)]
    assert events(state, "validate") == [(G1_B, G2_B), (G1_A, G2_A), (g1_64, g2_64)]
    assert result["block_m"] == 32 and result["us"] == 0.25


@pytest.mark.parametrize("token,small", [(1, True), (96, True), (97, False)])
def test_staged_small_batch_times_every_pruned_pair_without_screening(
    monkeypatch: "pytest.MonkeyPatch",
    capsys: "pytest.CaptureFixture[str]",
    token: "int",
    small: "bool",
) -> "None":
    # shape(): expert=128, topk=4, so M_est = ceil(token / 32) and 97 is M_est 4.
    state = controlled_search(monkeypatch, row=shape(token))
    state.validation[(G1_A, G2_B)] = 0.5
    result = state.run()
    log = capsys.readouterr().out
    if small:
        assert events(state, "consumer") == events(state, "bootstrap") == []
        assert events(state, "g1") == events(state, "g2") == []
        assert sorted(events(state, "validate")) == sorted(state.pairs)
        assert "small batch M_est=" in log and "exhaustive fallback" not in log
        assert (result["kernelName1"], result["kernelName2"]) == (G1_A, G2_B)
    else:
        assert len(events(state, "g1")) == len(events(state, "g2")) == 3
        assert events(state, "validate") == [(G1_B, G2_C)]
        assert "small batch" not in log


def test_staged_small_batch_matches_prune_mode_selection(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    winners = {}
    for mode in ("staged", "prune"):
        state = controlled_search(monkeypatch, row=shape(8))
        state.validation[(G1_C, G2_C)] = 0.5
        state.validation[(G1_A, G2_B)] = RuntimeError("incorrect output")
        state.args.mxfp4_search_mode = mode
        result = state.run()
        winners[mode] = (
            sorted(events(state, "validate")),
            result["kernelName1"],
            result["kernelName2"],
            result["us"],
        )
    assert winners["staged"] == winners["prune"]
    assert winners["staged"][1:] == (G1_C, G2_C, 0.5)


def test_g1_screening_times_only_direct_kernel_after_untimed_setup(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    events = []

    def prepare(*args: "object") -> "tuple[dict[str, object], tuple[object, ...]]":
        events.append("case")
        return {}, ()

    tensors = (object(), object())

    def prepare_launch(
        *args: "object", **kwargs: "object"
    ) -> "tuple[Callable[..., None], tuple[object, ...], object, object]":
        events.append("wrapper")

        def launch(*received: "object") -> "None":
            assert received == tensors
            events.append("kernel")

        return launch, tensors, object(), object()

    def measure(
        launch: "Callable[..., object]",
        *received: "object",
        num_warmup: "int",
        num_iters: "int",
    ) -> "tuple[None, float]":
        assert (num_warmup, num_iters) == (3, 5)
        assert events == ["case", "wrapper"]
        # Inputs reach run_perftest positionally, so it can rotate them.
        assert received == tensors
        events.append("timing")
        launch(*received)
        return None, 1.25

    monkeypatch.setattr(tuner, "_prepare_screening_case", prepare, raising=False)
    monkeypatch.setattr(
        tuner, "_prepare_gemm1_screening", prepare_launch, raising=False
    )
    monkeypatch.setattr("aiter.test_common.run_perftest", measure)
    score = tuner._screen_gemm1(
        shape(128),
        "flydsl_mxmoe_g1_a4w4_32x128x256_swiglu",
        SimpleNamespace(warmup=3, iters=5),
    )
    assert score == 1.25
    assert events == ["case", "wrapper", "timing", "kernel"]


class FakeTensor:
    def __init__(
        self,
        shape: "tuple[int, ...]",
        dtype: "torch.dtype | None" = None,
        value: "object" = 3,
    ) -> "None":
        self.shape = shape
        self.dtype = dtype or moe.dtypes.bf16
        self.device = "cpu"
        self.value = value

    def view(self, dtype: "torch.dtype | None") -> "FakeTensor":
        return self

    def numel(self) -> "int":
        product = 1
        for dimension in self.shape:
            product *= dimension
        return product

    def zero_(self) -> "FakeTensor":
        self.value = 0
        return self

    def __getitem__(self, index: "int") -> "SimpleNamespace":
        return SimpleNamespace(item=lambda: 128)


def fake_case() -> "tuple[dict[str, object], tuple[object, ...]]":
    data = {
        "input": FakeTensor((4, 3072)),
        "w1_a16": FakeTensor((128, 1024, 1536), torch.uint8),
        "w2_a16": FakeTensor((128, 3072, 256), torch.uint8),
        "w1s_a16": FakeTensor((128, 32), torch.uint8),
        "w2s_a16": FakeTensor((128, 32), torch.uint8),
        "topk_ids": FakeTensor((4, 4), torch.int32),
        "topk_weights": FakeTensor((4, 4), torch.float32),
    }
    for name in ("a1_qt", "w1_qt", "w2_qt", "a1_scale", "w1_scale", "w2_scale"):
        data[name] = object()
    sorting = (
        FakeTensor((256,), torch.int32),
        FakeTensor((256,), torch.float32),
        FakeTensor((16,), torch.int32),
        FakeTensor((2,), torch.int32),
        FakeTensor((4, 3072)),
        FakeTensor((256,), torch.int32),
        FakeTensor((4,), torch.int32),
    )
    return data, sorting


def rotated_copies(tensors: "tuple[object, ...]") -> "tuple[object, ...]":
    """Stand-ins for one run_perftest rotation set; aliases share one copy."""
    copies: dict[int, object] = {}
    return tuple(
        (
            None
            if tensor is None
            else copies.setdefault(
                id(tensor),
                (
                    FakeTensor(tensor.shape, tensor.dtype)
                    if isinstance(tensor, FakeTensor)
                    else object()
                ),
            )
        )
        for tensor in tensors
    )


@pytest.mark.parametrize("bm", [16, 32, 64, 128])
@pytest.mark.parametrize(
    "act,configured_limit,expected_limit",
    [
        ("silu", None, float("inf")),
        ("swiglu", None, 7.0),
        ("situv2", None, float("inf")),
        ("silu", 5, 5.0),
        ("swiglu", 5, 5.0),
    ],
)
def test_g1_screening_matches_bm_input_quantization_and_effective_default_parameters(
    monkeypatch: "pytest.MonkeyPatch",
    bm: "int",
    act: "str",
    configured_limit: "float | None",
    expected_limit: "float",
) -> "None":
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    row = shape(
        act_type=f"ActivationType.{ {'silu': 'Silu', 'swiglu': 'Swiglu', 'situv2': 'Situv2'}[act] }"
    )
    name = f"flydsl_mxmoe_g1_a4w4_{bm}x128x256"
    if bm == 16:
        name += "_f16in_nt"
    if act != "silu":
        name += f"_{act}"
    data, sorting = fake_case()
    packed = FakeTensor((4, 1536), moe.dtypes.fp4x2)
    scales = FakeTensor((256, 96), torch.uint8)
    intermediate = FakeTensor((256, 256), torch.uint8), FakeTensor(
        (256, 16), torch.uint8
    )
    timed = False
    events = []
    wrapper_parameters = {}
    copy_of: dict[int, object] = {}

    def prepare_case(*args: "object") -> "dict[str, object]":
        assert not timed
        events.append("case")
        return data

    def sort(*args: "object", **kwargs: "object") -> "tuple[object, ...]":
        assert not timed
        assert kwargs == {"block_size": bm, "accumulate": False, "output_aux": "opus"}
        events.append("sort")
        return sorting

    def prequant(**kwargs: "object") -> "tuple[object, object]":
        assert not timed
        assert kwargs["input"] is data["input"]
        assert kwargs["num_experts_upper_bound"] == 128
        assert kwargs["block_size"] == bm and kwargs["topk"] == 4
        events.append("prequant")
        return packed, scales

    def prepare_output(
        input: "object", *args: "object", **kwargs: "object"
    ) -> "object":
        assert not timed
        assert input is (data["input"] if bm == 16 else packed)
        assert kwargs["a1_scale"] is (None if bm == 16 else scales)
        assert kwargs["swiglu_limit"] == expected_limit
        wrapper_parameters.update(kwargs)
        events.append("wrapper")
        return intermediate

    def direct(**kwargs: "object") -> "None":
        assert timed
        # The timed launch reads run_perftest's rotated copies, not the originals.
        received = [value for value in kwargs.values() if isinstance(value, FakeTensor)]
        assert received and all(
            any(tensor is copy for copy in copy_of.values()) for tensor in received
        )
        hidden = data["input"] if bm == 16 else packed
        assert kwargs["hidden_states"] is copy_of[id(hidden)]
        if bm != 16:
            assert kwargs["a_quant"] is kwargs["hidden_states"]
        assert kwargs["w1_u8"] is copy_of[id(data["w1_a16"])]
        assert kwargs["inline_quant"] == kwargs["native_scale_layout"] == (bm == 16)
        assert kwargs["interleave"] is False
        assert kwargs["a_dtype"] == kwargs["out_dtype"] == "fp4"
        assert kwargs["swiglu_limit"] == wrapper_parameters["swiglu_limit"]
        assert (
            kwargs["situ_beta"]
            == wrapper_parameters["situ_beta"]
            == (moe.DEFAULT_SITUV2_BETA if act == "situv2" else 1.0)
        )
        assert (
            kwargs["situ_linear_beta"]
            == wrapper_parameters["situ_linear_beta"]
            == (moe.DEFAULT_SITUV2_LINEAR_BETA if act == "situv2" else 1.0)
        )
        assert kwargs["inter_sorted_quant"] is copy_of[id(intermediate[0])]
        assert kwargs["inter_sorted_shuffled_scale"] is copy_of[id(intermediate[1])]
        events.append("kernel")

    def measure(
        call: "Callable[..., object]",
        *tensors: "object",
        num_warmup: "int",
        num_iters: "int",
    ) -> "tuple[None, float]":
        nonlocal timed
        assert (num_warmup, num_iters) == (3, 5)
        # Every kernel input is rotatable: weights, routing and both buffers.
        for tensor in (data["w1_a16"], data["w1s_a16"], *sorting[2:4], *intermediate):
            assert any(tensor is received for received in tensors)
        assert any(
            (data["input"] if bm == 16 else packed) is received for received in tensors
        )
        copies = rotated_copies(tensors)
        copy_of.update(zip(map(id, tensors), copies))
        timed = True
        try:
            call(*copies)
        finally:
            timed = False
        return None, 1.5

    monkeypatch.delenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", raising=False)
    if configured_limit is not None:
        monkeypatch.setenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", str(configured_limit))
    monkeypatch.setattr(tuner, "_prepare_case", prepare_case)
    monkeypatch.setattr(moe, "moe_sorting", sort)
    monkeypatch.setattr(moe.aiter, "fused_dynamic_mxfp4_quant_moe_sort", prequant)
    monkeypatch.setattr(moe, "_mxfp4_a4w4_stage1_fw", prepare_output)
    monkeypatch.setattr(
        torch, "empty", lambda *args, **kwargs: FakeTensor((0,), torch.uint8)
    )
    monkeypatch.setattr(
        "aiter.ops.flydsl.mxfp4_gemm1_kernels.flydsl_mxfp4_gemm1", direct
    )
    monkeypatch.setattr("aiter.test_common.run_perftest", measure)
    assert tuner._screen_gemm1(row, name, SimpleNamespace(warmup=3, iters=5)) == 1.5
    assert events == (
        ["case", "sort", "wrapper", "kernel"]
        if bm == 16
        else ["case", "sort", "prequant", "wrapper", "kernel"]
    )


def test_g2_screening_owns_fresh_intermediate_and_times_complete_output_path(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    data, _ = fake_case()
    names = [
        f"flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_{epilog}_sbm128"
        for epilog in ("atomic", "reduce", "scatter")
    ]
    producer_name = "flydsl_mxmoe_g1_a4w4_128x128x256_swiglu"
    timed = False
    intermediates = []
    measured = []
    output_refs = []
    events = []
    copy_of: dict[int, object] = {}

    def prepare_case(*args: "object") -> "dict[str, object]":
        assert not timed
        return data

    def sort(*args: "object", **kwargs: "object") -> "tuple[object, ...]":
        assert not timed
        _, sorting = fake_case()
        output_refs.append(weakref.ref(sorting[4]))
        return sorting

    def producer(*args: "object", **kwargs: "object") -> "tuple[object, object]":
        assert not timed
        assert all(reference() is None for reference in intermediates)
        assert kwargs["kernelName1"] == producer_name and kwargs["swiglu_limit"] == 7.0
        payload, scale = FakeTensor((256, 256), torch.uint8), FakeTensor(
            (256, 16), torch.uint8
        )
        intermediates.extend((weakref.ref(payload), weakref.ref(scale)))
        events.append("producer")
        return payload, scale

    def consume(
        payload: "FakeTensor",
        w1: "object",
        w2: "object",
        sti: "object",
        sei: "object",
        nvi: "object",
        output: "FakeTensor",
        topk: "int",
        **kwargs: "object",
    ) -> "object":
        assert timed
        # The timed launch reads run_perftest's rotated copies; w1 stays bound.
        assert payload is copy_of[id(intermediates[-2]())]
        assert kwargs["a2_scale"] is copy_of[id(intermediates[-1]())]
        assert w1 is data["w1_a16"] and w2 is copy_of[id(data["w2_a16"])]
        assert output is copy_of[id(output_refs[-1]())]
        assert output.dtype == moe.dtypes.bf16
        assert (
            kwargs["sorted_weights"] is not None
            and kwargs["reverse_sorted"] is not None
        )
        assert topk == 4
        name = kwargs["kernelName2"]
        if "atomic" in name:
            assert output.value == 0
        else:
            assert output.value != 0
        output.value = 7
        measured.append(name)
        events.append("complete-output")
        return output

    def measure(
        call: "Callable[..., object]",
        *tensors: "object",
        num_warmup: "int",
        num_iters: "int",
    ) -> "tuple[None, float]":
        nonlocal timed
        assert (num_warmup, num_iters) == (3, 5)
        # The intermediate, w2 and the output rotate; w1 (shape only) does not.
        for tensor in (intermediates[-2](), intermediates[-1](), data["w2_a16"]):
            assert any(tensor is received for received in tensors)
        assert any(output_refs[-1]() is received for received in tensors)
        assert not any(data["w1_a16"] is received for received in tensors)
        copies = rotated_copies(tensors)
        copy_of.update(zip(map(id, tensors), copies))
        timed = True
        try:
            for _ in range(2):
                assert call(*copies).value == 7
        finally:
            timed = False
        return None, 2.5

    monkeypatch.delenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", raising=False)
    monkeypatch.setattr(tuner, "_prepare_case", prepare_case)
    monkeypatch.setattr(moe, "moe_sorting", sort)
    monkeypatch.setattr(
        moe.aiter,
        "fused_dynamic_mxfp4_quant_moe_sort",
        lambda **kwargs: (
            FakeTensor((4, 1536), moe.dtypes.fp4x2),
            FakeTensor((256, 96), torch.uint8),
        ),
    )
    monkeypatch.setattr(moe, "_mxfp4_a4w4_stage1_fw", producer)
    monkeypatch.setattr(moe, "_mxfp4_a4w4_stage2_fw", consume)
    monkeypatch.setattr(
        torch, "empty", lambda *args, **kwargs: FakeTensor((0,), torch.uint8)
    )
    monkeypatch.setattr("aiter.test_common.run_perftest", measure)
    for name in names:
        assert (
            tuner._screen_gemm2(
                shape(2048), producer_name, name, SimpleNamespace(warmup=3, iters=5)
            )
            == 2.5
        )
        assert all(reference() is None for reference in intermediates)
    assert measured == [name for name in names for _ in range(2)]
    assert events == ["producer", "complete-output", "complete-output"] * 3


def test_pair_validation_passes_pipeline_inputs_positionally_for_rotation(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    data, _ = fake_case()
    kn1 = "flydsl_mxmoe_g1_a4w4_128x128x256_swiglu"
    kn2 = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_atomic_sbm128"
    timed = False
    timed_inputs = []
    rotated = []

    def pair(received: "dict[str, object]", *args: "object", **kwargs: "object"):
        assert args[:2] == (kn1, kn2)
        if timed:
            timed_inputs.append(received)
        else:
            assert received is data
        return object()

    def measure(
        call: "Callable[..., object]",
        *tensors: "object",
        num_warmup: "int",
        num_iters: "int",
        **options: "object",
    ) -> "tuple[None, float]":
        nonlocal timed
        assert (num_warmup, num_iters) == (3, 5)
        # Rotation stays automatic outside fast-scan.
        assert options == {"profile_warm_iters": 1, "num_rotate_args": 0}
        # A zero-argument closure would make run_perftest's rotation a no-op.
        assert tensors == tuple(data[key] for key in tuner._PORT_E2E_INPUTS)
        rotated.extend(rotated_copies(tensors))
        timed = True
        try:
            call(*rotated)
        finally:
            timed = False
        return None, 4.0

    monkeypatch.delenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", raising=False)
    monkeypatch.setattr(tuner, "_prepare_case", lambda *args: data)
    monkeypatch.setattr(tuner, "_port_e2e", pair)
    monkeypatch.setattr(tuner, "_torch_ref", lambda *args, **kwargs: object())
    monkeypatch.setattr(moe, "cosine_diff_compare", lambda *args, **kwargs: 0.01)
    monkeypatch.setattr(tuner, "calculate", lambda *args: (0.0, 0.0))
    monkeypatch.setattr("aiter.test_common.run_perftest", measure)
    candidate = {"kernelName1": kn1, "kernelName2": kn2, "block_m": 128}
    args = SimpleNamespace(errRatio=0.1, warmup=3, iters=5)
    assert RUN_CANDIDATE(tuner, shape(2048), candidate, args) == 4.0
    assert len(timed_inputs) == 1
    # The timed pipeline runs on the rotated copies, keyed like the original data.
    assert timed_inputs[0] == dict(zip(tuner._PORT_E2E_INPUTS, rotated))
    assert all(timed_inputs[0][key] is not data[key] for key in timed_inputs[0])


@pytest.mark.parametrize("limit", [None, 5.0])
def test_reference_consumer_uses_torch_layout_before_untimed_pair_bootstrap(
    monkeypatch: "pytest.MonkeyPatch",
    limit: "float | None",
) -> "None":
    monkeypatch.delenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", raising=False)
    if limit is not None:
        monkeypatch.setenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", str(limit))
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    data, sorting = fake_case()
    torch_intermediate = FakeTensor((256, 256), torch.uint8), FakeTensor(
        (256, 16), torch.uint8
    )
    output, ref1, ref2 = object(), object(), object()
    events = []
    kn1 = "flydsl_mxmoe_g1_a4w4_128x128x256_swiglu"
    kn2 = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_atomic_sbm128"

    def torch_stage1(*args: "object", **kwargs: "object") -> "object":
        assert kwargs["swiglu_limit"] == limit
        assert kwargs["activation"] == moe.ActivationType.Swiglu
        assert kwargs["doweight_stage1"] is False and kwargs["topk"] == 4
        events.append("torch-stage1")
        return ref1

    def populate(
        reference: "object",
        intermediate: "dict[str, object]",
        token: "int",
        topk: "int",
        bm: "int",
    ) -> "None":
        assert reference == {"ref1": ref1, "adtype": "fp4"}
        assert (token, topk, bm) == (2048, 4, 128)
        assert (
            intermediate["sti"] is sorting[0] and intermediate["cumsum"] is sorting[3]
        )
        events.append("layout-population")
        intermediate.update(isq=torch_intermediate[0], iss=torch_intermediate[1])

    def consume(
        payload: "FakeTensor",
        *args: "object",
        **kwargs: "object",
    ) -> "object":
        assert (
            payload is torch_intermediate[0]
            and kwargs["a2_scale"] is torch_intermediate[1]
        )
        events.append("torch-consumer")
        return output

    def torch_stage2(*args: "object", **kwargs: "object") -> "object":
        assert args[0] is ref1
        events.append("torch-stage2")
        return ref2

    def compare(reference: "object", actual: "object", **kwargs: "object") -> "float":
        assert reference is ref2 and actual is output
        events.append("compare")
        return 0.01

    def pair(*args: "object", **kwargs: "object") -> "object":
        assert kwargs["swiglu_limit"] == limit
        assert args[1:3] == (kn1, kn2)
        events.append("normal-pair")
        return output

    def torch_ref(*args: "object", **kwargs: "object") -> "object":
        assert kwargs["swiglu_limit"] == limit
        events.append("normal-reference")
        return ref2

    monkeypatch.setattr(tuner, "_prepare_screening_case", lambda *args: (data, sorting))
    monkeypatch.setattr(tuner, "_prepare_case", lambda *args: data)
    monkeypatch.setattr(moe.FmoeTuner, "run_torch_moe_stage1", torch_stage1)
    monkeypatch.setattr(moe.FmoeTuner, "run_torch_moe_stage2", torch_stage2)
    monkeypatch.setattr(moe, "_v2_populate_stage2", populate)
    monkeypatch.setattr(moe, "_mxfp4_a4w4_stage2_fw", consume)
    monkeypatch.setattr(moe, "cosine_diff_compare", compare)
    monkeypatch.setattr(tuner, "_port_e2e", pair)
    monkeypatch.setattr(tuner, "_torch_ref", torch_ref)
    args = SimpleNamespace(errRatio=0.1)
    assert tuner._check_reference_consumer(shape(2048), kn2, args)
    assert tuner._check_reference_pair(shape(2048), kn1, kn2, args)
    assert events == [
        "torch-stage1",
        "layout-population",
        "torch-consumer",
        "torch-stage2",
        "compare",
        "normal-pair",
        "normal-reference",
        "compare",
    ]


_GET_CONTEXT = multiprocessing.get_context


_PROCESS_START = BaseProcess.start


def cpu_result_worker(
    payload: "tuple[tuple[str, ...], dict[str, object], SimpleNamespace, int]",
    result_tx: "Connection",
) -> "None":
    keys, row, args, gpu = payload
    del keys, args
    try:
        result_tx.send({**row, "gpu": gpu, "pid": os.getpid(), "us": 3.0})
    finally:
        result_tx.close()


def cpu_coordinated_worker(
    payload: "tuple[tuple[str, ...], dict[str, object], SimpleNamespace, int]",
    result_tx: "Connection",
) -> "None":
    keys, row, args, gpu = payload
    del keys
    try:
        if row.get("barrier"):
            args.barrier.wait(timeout=15)
        result_tx.send({**row, "gpu": gpu, "pid": os.getpid(), "us": 3.0})
        if row.get("pause") and not args.allow_exit.wait(timeout=15):
            raise RuntimeError("test did not release CPU worker")
    finally:
        result_tx.close()


def cpu_outcome_worker(
    payload: "tuple[tuple[str, ...], dict[str, object], SimpleNamespace, int]",
    result_tx: "Connection",
) -> "None":
    keys, row, args, gpu = payload
    del keys, args
    try:
        if row.get("outcome") == "fatal":
            os._exit(23)
        if row.get("outcome") == "eof":
            return
        result = {**row, "gpu": gpu, "pid": os.getpid(), "us": 3.0}
        if row.get("outcome") == "python-error":
            result.update(us=-1.0, kernelName1="FAILED")
        result_tx.send(result)
        if row.get("outcome") == "result-fatal":
            os._exit(23)
    finally:
        result_tx.close()


CPU_WORKERS = {cpu_result_worker, cpu_coordinated_worker, cpu_outcome_worker}


def payloads(
    *rows: "object", args: "SimpleNamespace | None" = None
) -> "list[tuple[tuple[str, ...], dict[str, object], SimpleNamespace, None]]":
    return [(tuple(row), row, args or SimpleNamespace(), None) for row in rows]


class TrackingProcess:
    def __init__(
        self, process: "BaseProcess", gpu: "int", context: "TrackingContext"
    ) -> "None":
        self.process, self.gpu, self.context = process, gpu, context
        self.closed = False
        self.joined = False
        self.saved_exitcode = None

    def start(self) -> "None":
        outstanding = [proc for proc in self.context.processes if not proc.closed]
        self.context.handle_counts.append(len(outstanding))
        self.context.starts.append(self.gpu)
        assert all(proc.gpu != self.gpu for proc in outstanding if proc is not self)
        if self.context.start_failure is not None:
            raise self.context.start_failure
        self.process.start()

    def join(self, *args: "object", **kwargs: "object") -> "None":
        self.process.join(*args, **kwargs)
        self.joined = True

    def close(self) -> "None":
        self.saved_exitcode = self.process.exitcode
        self.process.close()
        self.closed = True

    @property
    def exitcode(self) -> "int | None":
        return self.saved_exitcode if self.closed else self.process.exitcode

    def __getattr__(self, name: "str") -> "object":
        return getattr(self.process, name)


class TrackingConnection:
    def __init__(self, connection: "Connection", context: "TrackingContext") -> "None":
        self.connection, self.context = connection, context

    def recv_bytes(self) -> "bytes":
        if self.context.receive_failure is not None:
            raise self.context.receive_failure
        payload = self.connection.recv_bytes()
        with self.context.received:
            self.context.receive_count += 1
            self.context.received.notify_all()
        return b"invalid-pickle" if self.context.invalid_payload else payload

    def close(self) -> "None":
        self.connection.close()
        if self.context.close_failure is not None:
            raise self.context.close_failure

    def __getattr__(self, name: "str") -> "object":
        return getattr(self.connection, name)


class TrackingContext:
    def __init__(self, base: "BaseContext") -> "None":
        self.base = base
        self.processes = []
        self.pipes = []
        self.starts = []
        self.handle_counts = []
        self.received = threading.Condition()
        self.receive_count = 0
        self.receive_failure = None
        self.start_failure = None
        self.close_failure = None
        self.invalid_payload = False

    def Pipe(
        self, *args: "object", **kwargs: "object"
    ) -> "tuple[TrackingConnection, Connection]":
        rx, tx = self.base.Pipe(*args, **kwargs)
        result = TrackingConnection(rx, self), tx
        self.pipes.append(result)
        return result

    def Process(self, *args: "object", **kwargs: "object") -> "TrackingProcess":
        process = TrackingProcess(
            self.base.Process(*args, **kwargs), kwargs["args"][0][3], self
        )
        self.processes.append(process)
        return process

    def wait_until_received(self, count: "int") -> "bool":
        with self.received:
            return self.received.wait_for(
                lambda: self.receive_count >= count, timeout=20
            )

    def assert_closed(self) -> "None":
        assert all(process.closed for process in self.processes)
        assert all(endpoint.closed for pipe in self.pipes for endpoint in pipe)


def run_in_thread(
    operation: "Callable[[], object]",
) -> "tuple[threading.Thread, list[object], list[BaseException]]":
    result = []
    errors = []

    def run() -> "None":
        try:
            result.append(operation())
        except BaseException as error:  # noqa: BLE001 - transport test-thread failures
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, result, errors


class TestStagedSupervisor:
    """Private-Pipe supervisor checks with explicit CPU-only shape workers."""

    # Overrides the module fixture to also allow the CPU workers above.
    @pytest.fixture(autouse=True)
    def prohibit_real_shape_workers_and_gpu_data(
        self, monkeypatch: "pytest.MonkeyPatch"
    ) -> "Iterator[None]":
        lifecycle.block_real_work(monkeypatch)

        def cpu_only_start(process: "BaseProcess") -> "None":
            if process._target not in CPU_WORKERS:
                pytest.fail("supervisor test attempted a non-CPU worker target")
            return _PROCESS_START(process)

        monkeypatch.setattr(BaseProcess, "start", cpu_only_start)
        with lifecycle.use_default_device("cpu"):
            yield

    def test_staged_supervisor_empty_batch_launches_none_and_one_shape_gets_worker_pid(
        self,
    ) -> "None":
        ctx = _GET_CONTEXT("spawn")
        assert (
            moe._run_staged_shapes_isolated([], 1, ctx, entry=cpu_result_worker) == []
        )
        results = moe._run_staged_shapes_isolated(
            payloads({"token": 4}), 1, ctx, entry=cpu_result_worker
        )
        assert results[0]["token"] == 4 and results[0]["gpu"] == 0
        assert results[0]["pid"] != os.getpid()

    def test_result_arrival_keeps_gpu_and_process_owned_until_exit(self) -> "None":
        base = _GET_CONTEXT("spawn")
        args = SimpleNamespace(allow_exit=base.Event())
        ctx = TrackingContext(base)
        thread, results, errors = run_in_thread(
            lambda: moe._run_staged_shapes_isolated(
                payloads({"token": 1, "pause": True}, {"token": 2}, args=args),
                1,
                ctx,
                entry=cpu_coordinated_worker,
            )
        )
        try:
            assert ctx.wait_until_received(1)
            assert ctx.starts == [0]
            assert not ctx.processes[0].closed and not ctx.processes[0].joined
        finally:
            args.allow_exit.set()
            thread.join(timeout=20)
        assert not thread.is_alive() and errors == []
        assert [row["token"] for row in results[0]] == [1, 2]
        assert all(process.joined for process in ctx.processes)
        ctx.assert_closed()

    def test_different_gpu_workers_overlap_and_all_completed_handles_close(
        self,
    ) -> "None":
        base = _GET_CONTEXT("spawn")
        args = SimpleNamespace(allow_exit=base.Event(), barrier=base.Barrier(2))
        ctx = TrackingContext(base)
        rows = [{"token": i, "pause": i < 2, "barrier": i < 2} for i in range(6)]
        thread, results, errors = run_in_thread(
            lambda: moe._run_staged_shapes_isolated(
                payloads(*rows, args=args), 2, ctx, entry=cpu_coordinated_worker
            )
        )
        try:
            assert ctx.wait_until_received(2)
            assert ctx.starts == [0, 1]
            assert all(process.process.is_alive() for process in ctx.processes)
        finally:
            args.allow_exit.set()
            thread.join(timeout=30)
        assert not thread.is_alive() and errors == []
        assert len(results[0]) == 6
        assert len({row["pid"] for row in results[0]}) == 6
        assert max(ctx.handle_counts) <= 2
        assert all(
            process.joined and process.exitcode == 0 for process in ctx.processes
        )
        ctx.assert_closed()

    def test_one_slot_uses_fresh_worker_and_bounded_handles_for_every_shape(
        self,
    ) -> "None":
        ctx = TrackingContext(_GET_CONTEXT("spawn"))
        results = moe._run_staged_shapes_isolated(
            payloads(*[{"token": i} for i in range(4)]), 1, ctx, entry=cpu_result_worker
        )
        assert len({row["pid"] for row in results}) == 4
        assert ctx.starts == [0, 0, 0, 0] and ctx.handle_counts == [1, 1, 1, 1]
        assert all(process.joined for process in ctx.processes)
        ctx.assert_closed()

    def test_fatal_clean_eof_python_failure_and_result_before_fatal_continue_later_rows(
        self,
    ) -> "None":
        ctx = TrackingContext(_GET_CONTEXT("spawn"))
        results = moe._run_staged_shapes_isolated(
            payloads(
                *[
                    {"token": i, "outcome": outcome}
                    for i, outcome in enumerate(
                        ("fatal", "eof", "python-error", "result-fatal", "success")
                    )
                ]
            ),
            1,
            ctx,
            entry=cpu_outcome_worker,
        )
        assert results[0] is results[1] is results[3] is None
        assert results[2]["us"] < 0 and results[2]["kernelName1"] == "FAILED"
        assert results[4]["token"] == 4 and results[4]["us"] > 0
        assert [proc.exitcode for proc in ctx.processes] == [23, 0, 0, 23, 0]
        ctx.assert_closed()

    @pytest.mark.parametrize(
        "failure",
        [
            RuntimeError("receive failed"),
            KeyboardInterrupt("receive cancelled"),
            "deserialize",
        ],
    )
    def test_unexpected_receive_and_deserialization_failure_clean_owned_workers(
        self,
        failure: "Exception | str",
    ) -> "None":
        base = _GET_CONTEXT("spawn")
        args = SimpleNamespace(allow_exit=base.Event())
        ctx = TrackingContext(base)
        if failure == "deserialize":
            ctx.invalid_payload = True
            expected = Exception
        else:
            ctx.receive_failure = failure
            expected = type(failure)
        with pytest.raises(expected) as actual:
            moe._run_staged_shapes_isolated(
                payloads({"token": 1, "pause": True}, {"token": 2}, args=args),
                1,
                ctx,
                entry=cpu_coordinated_worker,
            )
        if failure != "deserialize":
            assert actual.value is failure
        assert ctx.starts == [0]
        assert ctx.processes[0].joined and ctx.processes[0].exitcode is not None
        ctx.assert_closed()

    @pytest.mark.parametrize(
        "failure", [KeyboardInterrupt("parent cancelled"), RuntimeError("wait failed")]
    )
    def test_parent_cancellation_or_wait_failure_stops_launches_and_preserves_exception(
        self, monkeypatch: "pytest.MonkeyPatch", failure: "BaseException"
    ) -> "None":
        base = _GET_CONTEXT("spawn")
        ctx = TrackingContext(base)
        args = SimpleNamespace(allow_exit=base.Event())

        def fail_wait(*args: "object", **kwargs: "object") -> "None":
            raise failure

        monkeypatch.setattr(moe, "wait_for_connections", fail_wait)
        with pytest.raises(type(failure)) as actual:
            moe._run_staged_shapes_isolated(
                payloads({"token": 1, "pause": True}, {"token": 2}, args=args),
                1,
                ctx,
                entry=cpu_coordinated_worker,
            )
        assert actual.value is failure and ctx.starts == [0]
        assert ctx.processes[0].joined and ctx.processes[0].exitcode is not None
        ctx.assert_closed()

    def test_start_failure_closes_created_process_and_both_pipe_endpoints(
        self,
    ) -> "None":
        ctx = TrackingContext(_GET_CONTEXT("spawn"))
        ctx.start_failure = RuntimeError("CPU spawn failed")
        with pytest.raises(RuntimeError) as actual:
            moe._run_staged_shapes_isolated(
                payloads({"token": 1}, {"token": 2}), 1, ctx, entry=cpu_result_worker
            )
        assert actual.value is ctx.start_failure and ctx.starts == [0]
        ctx.assert_closed()

    def test_start_failure_survives_pipe_close_failure_during_cleanup(self) -> "None":
        ctx = TrackingContext(_GET_CONTEXT("spawn"))
        ctx.start_failure = RuntimeError("CPU spawn failed")
        ctx.close_failure = ValueError("receive endpoint close failed")
        with pytest.raises(RuntimeError) as actual:
            moe._run_staged_shapes_isolated(
                payloads({"token": 1}, {"token": 2}), 1, ctx, entry=cpu_result_worker
            )
        assert actual.value is ctx.start_failure and ctx.starts == [0]
        ctx.assert_closed()

    @pytest.mark.parametrize(
        "mp,shape_count,gpu_count,expected",
        [(0, 3, 3, 1), (-1, 3, 3, 1), (8, 1, 3, 1), (8, 3, 2, 2), (2, 3, 3, 2)],
    )
    def test_staged_scheduler_preserves_main_mp_cap_and_uses_preflighted_first_visible_gpus(
        self,
        monkeypatch: "pytest.MonkeyPatch",
        mp: "int",
        shape_count: "int",
        gpu_count: "int",
        expected: "int",
    ) -> "None":
        identities = [("gfx950", 256)] * expected + [("gfx942", 304)] * (
            gpu_count - expected
        )
        with lifecycle.devices(*identities):
            tuner = moe.Mxfp4FlydslTuner("mxfp4", lifecycle.KEYS, lifecycle.RESULTS)
            observed = []

            def schedule(
                batch: "list[tuple[list[str], dict[str, object], SimpleNamespace, None]]",
                slots: "int",
                ctx: "BaseContext",
            ) -> "list[dict[str, object]]":
                observed.append((slots, len(batch)))
                return [dict(row, us=3.0) for _, row, _, _ in batch]

            monkeypatch.setattr(moe, "_run_staged_shapes_isolated", schedule)
            results = tuner.tune(
                pd.DataFrame([lifecycle.shape(i + 1) for i in range(shape_count)]),
                None,
                SimpleNamespace(mp=mp, timeout=0, mxfp4_search_mode="staged"),
            )
        assert observed == [(expected, shape_count)] and len(results) == shape_count

    def test_staged_single_slot_uses_first_visible_gpu_and_empty_batch_never_schedules(
        self,
        monkeypatch: "pytest.MonkeyPatch",
    ) -> "None":
        with lifecycle.devices(("gfx950", 224), ("gfx942", 304), current=1):
            tuner = moe.Mxfp4FlydslTuner("mxfp4", lifecycle.KEYS, lifecycle.RESULTS)
            launches = []

            def schedule(
                batch: "list[tuple[list[str], dict[str, object], SimpleNamespace, None]]",
                slots: "int",
                ctx: "BaseContext",
            ) -> "list[dict[str, object]]":
                launches.append(slots)
                return [
                    dict(row, gfx="gfx950", cu_num=224, us=3.0)
                    for _, row, _, _ in batch
                ]

            monkeypatch.setattr(moe, "_run_staged_shapes_isolated", schedule)
            args = SimpleNamespace(mp=1, timeout=0, mxfp4_search_mode="staged")
            assert tuner.tune(pd.DataFrame(columns=lifecycle.KEYS), None, args) == []
            results = tuner.tune(pd.DataFrame([lifecycle.shape()]), None, args)
        assert launches == [1] and results[0]["cu_num"] == 224

    def test_staged_rejects_unsupported_participant_before_scheduler(
        self,
        monkeypatch: "pytest.MonkeyPatch",
    ) -> "None":
        with lifecycle.devices(("gfx950", 256), ("gfx942", 304)):
            tuner = moe.Mxfp4FlydslTuner("mxfp4", lifecycle.KEYS, lifecycle.RESULTS)

            def forbidden(*args: "object", **kwargs: "object") -> "None":
                pytest.fail("unsupported staged participant reached scheduling")

            monkeypatch.setattr(moe, "_run_staged_shapes_isolated", forbidden)
            with pytest.raises(ValueError, match="GPU1.*gfx942"):
                tuner.tune(
                    pd.DataFrame([lifecycle.shape(), lifecycle.shape(8)]),
                    None,
                    SimpleNamespace(mp=2, timeout=0, mxfp4_search_mode="staged"),
                )

    def test_staged_result_remains_readable_when_process_exit_precedes_receive(
        self,
        monkeypatch: "pytest.MonkeyPatch",
    ) -> "None":
        ctx = TrackingContext(_GET_CONTEXT("spawn"))
        observed = []

        def wait_after_exit(waitables: "list[object]") -> "list[object]":
            from multiprocessing.connection import wait

            process = next(proc for proc in ctx.processes if not proc.closed)
            wait([process.sentinel])
            process.process.join()
            assert process.process.exitcode == 0
            observed.append(ctx.receive_count)
            return wait(waitables)

        monkeypatch.setattr(moe, "wait_for_connections", wait_after_exit)
        results = moe._run_staged_shapes_isolated(
            payloads({"token": 1}, {"token": 2}), 1, ctx, entry=cpu_result_worker
        )
        assert observed == [0, 1]
        assert [row["token"] for row in results] == [1, 2]
        assert all(process.joined for process in ctx.processes)
        ctx.assert_closed()
