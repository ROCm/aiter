# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Controlled MXFP4 search checks at the CLI and shape-planning seams.

The normal ROCm/FlyDSL toolchain is needed for imports. These checks prohibit
GPU tensor preparation, profiling and real shape-worker launches.
"""

import sys

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
