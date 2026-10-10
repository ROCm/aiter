# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Controlled MXFP4 search checks at the CLI and shape-planning seams.

The normal ROCm/FlyDSL toolchain is needed for imports. These checks prohibit
GPU tensor preparation, profiling and real shape-worker launches.
"""

import sys
from types import SimpleNamespace

import pytest

from op_tests.tuning_tests.test_mxfp4_tuner_lifecycle import (
    KEYS,
    RESULTS,
    devices,
    moe,
    prohibit_real_shape_workers_and_gpu_data,  # noqa: F401 - imported autouse fixture
    shape,
)


def parse_cli(monkeypatch, *options):
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    monkeypatch.setattr(sys, "argv", ["gemm_moe_tune", *options])
    return tuner.parse_args()


def test_cli_accepts_staged_with_default_screening_topk(monkeypatch):
    args = parse_cli(monkeypatch, "--mxfp4-flydsl", "--mxfp4-search-mode", "staged")
    assert args.mxfp4_search_mode == "staged"
    assert args.mxfp4_screen_topk == 2


@pytest.mark.parametrize("screen_topk", [1, 2, 7])
def test_cli_accepts_any_positive_staged_screening_topk(monkeypatch, screen_topk):
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
def test_cli_preserves_default_prune_and_full(monkeypatch, mode):
    options = ["--mxfp4-flydsl"]
    if mode is not None:
        options.extend(["--mxfp4-search-mode", mode])
    args = parse_cli(monkeypatch, *options)
    assert args.mxfp4_search_mode == (mode or "prune")
    assert args.mxfp4_screen_topk == 2


@pytest.mark.parametrize("mode", [None, "prune", "full"])
def test_cli_rejects_explicit_screening_topk_without_explicit_staged(
    monkeypatch, capsys, mode
):
    options = ["--mxfp4-flydsl", "--mxfp4-screen-topk", "2"]
    if mode is not None:
        options.extend(["--mxfp4-search-mode", mode])
    with pytest.raises(SystemExit) as failure:
        parse_cli(monkeypatch, *options)
    assert failure.value.code == 2
    assert "requires explicit --mxfp4-search-mode staged" in capsys.readouterr().err


@pytest.mark.parametrize("screen_topk", ["0", "-1", "2.5"])
def test_cli_rejects_nonpositive_or_noninteger_screening_topk(monkeypatch, screen_topk):
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
def test_cli_retains_tuner_and_grouped_combination_checks(monkeypatch, options):
    with pytest.raises(SystemExit) as failure:
        parse_cli(monkeypatch, *options, "--mxfp4-search-mode", "staged")
    assert failure.value.code == 2


def pair_names(rows):
    return [(row["kernelName1"], row["kernelName2"]) for row in rows]


def test_staged_plan_projects_real_pruned_pairs_and_supports_i384_without_gpu():
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
def test_staged_keeps_main_m_est_boundary_rules(m_est, expected_bms):
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
    monkeypatch, updates, reason
):
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)

    def forbid_enumeration(*args, **kwargs):
        pytest.fail("unsupported shape entered candidate enumeration")

    monkeypatch.setattr(tuner, "_candidate_rows", forbid_enumeration)
    with pytest.raises(ValueError, match=reason):
        tuner._staged_candidate_plan(shape(**updates))


def plan_fixture(monkeypatch, row, pairs):
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    candidates = [
        tuner._candidate_row(row, bm, gemm1, gemm2) for bm, gemm1, gemm2 in pairs
    ]

    def enumerated(shape_row, full_search=False):
        assert full_search is False
        return candidates

    monkeypatch.setattr(tuner, "_candidate_rows", enumerated)
    return tuner._staged_candidate_plan(row)


def test_staged_preserves_complete_names_and_groups_reduce_with_scatter(monkeypatch):
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
    monkeypatch, bm, g1, g2, reason
):
    with pytest.raises(ValueError, match=reason):
        plan_fixture(monkeypatch, shape(), [(bm, g1, g2)])


G1_A = "flydsl_mxmoe_g1_a4w4_32x128x256_swiglu"
G1_B = "flydsl_mxmoe_g1_a4w4_32x256x256_swiglu"
G1_C = "flydsl_mxmoe_g1_a4w4_32x128x256_swiglu_xcd2"
G2_A = "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm32"
G2_B = "flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x128_atomic_sbm32"
G2_C = "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_persist_sbm32"


def test_staged_shape_validates_only_successful_shortlist_product(monkeypatch):
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

    def validate(row, candidate, args):
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


def controlled_search(monkeypatch, *, pairs=None, screen_topk=1, row=None):
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

    def returned(outcome):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def enumerate_pairs(actual_row, full_search=False):
        state.events.append(("enumerate", actual_row["token"], full_search))
        assert full_search is False
        return [tuner._candidate_row(actual_row, bm, g1, g2) for bm, g1, g2 in pairs]

    def consumer(actual_row, name, actual_args):
        state.events.append(("consumer", name))
        assert actual_args is args
        return returned(state.consumers.get(name, True))

    def bootstrap(actual_row, producer, consumer, actual_args):
        state.events.append(("bootstrap", producer, consumer))
        assert actual_args is args
        return returned(state.bootstrap.get(producer, True))

    def screen_g1(actual_row, name, actual_args):
        state.events.append(("g1", name))
        assert actual_args is args
        return returned(state.stage1.get(name, 1.0))

    def screen_g2(actual_row, producer, name, actual_args):
        state.events.append(("g2", producer, name))
        assert actual_args is args
        return returned(state.stage2.get(name, 1.0))

    def validate(actual_row, candidate, actual_args):
        pair = candidate["kernelName1"], candidate["kernelName2"]
        state.events.append(("validate", *pair))
        assert actual_args is args
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
    return state


def events(state, kind):
    return [event[1:] for event in state.events if event[0] == kind]


@pytest.mark.parametrize("screen_topk,count", [(1, 1), (2, 4), (7, 9)])
def test_staged_screening_topk_caps_shortlist_product(monkeypatch, screen_topk, count):
    state = controlled_search(monkeypatch, screen_topk=screen_topk)
    state.run()
    attempted = events(state, "validate")
    assert len(attempted) == len(set(attempted)) == count
    if screen_topk == 2:
        assert attempted == [(G1_B, G2_C), (G1_B, G2_A), (G1_C, G2_C), (G1_C, G2_A)]


def test_staged_exact_ties_keep_only_k_in_admitted_order(monkeypatch):
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
    monkeypatch, invalid
):
    state = controlled_search(monkeypatch, screen_topk=7)
    state.stage1 = {G1_A: invalid, G1_B: 1.0, G1_C: invalid}
    state.stage2 = {G2_A: invalid, G2_B: invalid, G2_C: 1.0}
    state.run()
    assert events(state, "validate") == [(G1_B, G2_C)]


def test_staged_bootstrap_prefers_admitted_reference_and_keeps_it_untimed(monkeypatch):
    state = controlled_search(monkeypatch)
    result = state.run()
    assert events(state, "consumer") == [(G2_A,)]
    assert events(state, "bootstrap") == [(G1_A, G2_A)]
    assert {producer for producer, _ in events(state, "g2")} == {G1_A}
    assert events(state, "validate") == [(G1_B, G2_C)]
    assert result["us"] == 10.0


def test_staged_reference_retries_only_remaining_admitted_candidates(monkeypatch):
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
    monkeypatch, failure
):
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
    monkeypatch, invalid
):
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


def test_staged_screening_failed_candidates_can_win_during_fallback(monkeypatch):
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


def test_staged_no_winner_and_empty_universe_return_invalid(monkeypatch):
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


def test_staged_scores_and_pair_cache_are_fresh_for_each_shape(monkeypatch):
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


def test_staged_reuses_g1_ranking_across_atomic_reduce_and_scatter_groups(monkeypatch):
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
    monkeypatch,
):
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


def test_staged_g2_failure_exhausts_only_affected_compatibility_group(monkeypatch):
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
    monkeypatch,
):
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
