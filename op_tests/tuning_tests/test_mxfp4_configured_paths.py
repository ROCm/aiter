# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Configured tuning behavior at the approved invocation/production seams.

Normal ROCm/FlyDSL imports, offline discovery and fail-closed GPU entry points.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from op_tests.tuning_tests.test_mxfp4_tuner_lifecycle import (
    KEYS,
    RESULTS,
    devices,
    moe,
    prohibit_real_shape_workers_and_gpu_data,  # noqa: F401
    shape,
)

PORT_E2E = moe.Mxfp4FlydslTuner._port_e2e


def test_invocation_captures_limit_and_stage2_settings_for_every_shape(monkeypatch):
    monkeypatch.setenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", "5")
    monkeypatch.setenv("AITER_FLYDSL_STAGE2_FP8", "1")
    monkeypatch.setenv("AITER_MXFP4_INTERMEDIATE", "1")
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        monkeypatch.setattr("sys.argv", ["tuner", "--mxfp4-flydsl"])
        args = tuner.parse_args()
        snapshots = []

        def tune_shape(row, args):
            snapshots.append(args.mxfp4_configuration)
            return {**row, "us": 1.0}

        monkeypatch.setattr(tuner, "_tune_one_shape", tune_shape)
        tuner.tune(pd.DataFrame([shape(128), shape(2048)]), None, args)
    assert (
        snapshots
        == [{"swiglu_limit": 5.0, "stage2_fp8": True, "fp4_scatter": True}] * 2
    )
    assert snapshots[0] is snapshots[1]


@pytest.mark.parametrize("value", ["0", "-5", "nan", "inf"])
def test_invalid_activation_limit_rejects_invocation_before_gpu_work(
    monkeypatch, value
):
    monkeypatch.setenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", value)
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        monkeypatch.setattr("sys.argv", ["tuner", "--mxfp4-flydsl"])
        with pytest.raises(ValueError, match="finite and positive"):
            tuner.parse_args()


def test_custom_limit_rejects_situv2_even_if_same_shape_is_already_tuned(monkeypatch):
    monkeypatch.setenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", "5")
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    row = shape(act_type="ActivationType.Situv2")
    tuner.untunedf = pd.DataFrame([row])
    tuner.tunedf = pd.DataFrame([{**row, "us": 1.0}])
    assert not tuner._already_tuned_mask(KEYS).iloc[0]
    result = tuner._tune_one_shape(row, SimpleNamespace(timeout=0))
    assert result["us"] == -1 and "Situv2" in result["kernelName1"]


def test_separate_invocations_on_one_tuner_capture_current_settings(monkeypatch):
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    snapshots = []
    monkeypatch.setattr(
        tuner,
        "_tune_one_shape",
        lambda row, args: snapshots.append(args.mxfp4_configuration)
        or {**row, "us": 1.0},
    )
    for limit in ("7", "5"):
        monkeypatch.setenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", limit)
        with devices(("gfx950", 256)):
            tuner.tune(pd.DataFrame([shape()]), None, SimpleNamespace(mp=1))
    assert [item["swiglu_limit"] for item in snapshots] == [7.0, 5.0]


def test_fp8_stage2_pair_and_screening_receive_original_route_weights(monkeypatch):
    from op_tests.tuning_tests.test_mxfp4_staged_execution import fake_case

    data, sorting = fake_case()
    monkeypatch.setattr(moe, "moe_sorting", lambda *args, **kwargs: sorting)
    monkeypatch.setattr(
        moe, "_mxfp4_a4w4_stage1_fw", lambda *args, **kwargs: (object(), object())
    )
    monkeypatch.setattr(
        moe.aiter,
        "fused_dynamic_mxfp4_quant_moe_sort",
        lambda **kwargs: (object(), object()),
    )
    observed = []

    def stage2(*args, **kwargs):
        observed.append(kwargs["topk_weights"])
        return sorting[4]

    monkeypatch.setattr(moe, "_mxfp4_a4w4_stage2_fw", stage2)
    name = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_reduce_sbm128"
    PORT_E2E(
        data,
        "flydsl_mxmoe_g1_a4w4_128x128x256_swiglu",
        name,
        4,
        128,
        3072,
        moe.dtypes.bf16,
        swiglu_limit=5,
    )
    moe.Mxfp4FlydslTuner._gemm2_screening_call(
        data, sorting, object(), object(), name, 4
    )()
    assert observed == [data["topk_weights"], data["topk_weights"]]


@pytest.mark.parametrize(
    "fp8,fp4", [(None, None), ("", ""), ("1", None), (None, "1"), ("1", "1")]
)
@pytest.mark.parametrize(
    "epilog,bn", [("atomic", 128), ("reduce", 128), ("scatter", 128), ("scatter", 256)]
)
def test_production_stage2_path_matches_invocation_configuration(
    monkeypatch, fp8, fp4, epilog, bn
):
    import importlib

    from op_tests.tuning_tests.test_mxfp4_staged_execution import FakeTensor, fake_case

    monkeypatch.setattr(FakeTensor, "element_size", lambda self: 1, raising=False)
    fm = importlib.import_module("aiter.fused_moe")
    dispatcher = importlib.import_module("aiter.ops.flydsl.kernels.mxmoe_dispatcher")
    kernels = importlib.import_module("aiter.ops.flydsl.moe_kernels")
    for name, value in (
        ("AITER_FLYDSL_STAGE2_FP8", fp8),
        ("AITER_MXFP4_INTERMEDIATE", fp4),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    data, sorting = fake_case()
    config = tuner._invocation_configuration()
    expected = (
        "fp8"
        if epilog == "reduce" and config["stage2_fp8"]
        else "fp4" if epilog == "scatter" and config["fp4_scatter"] else "bf16"
    )
    observed = {}
    finished = []
    monkeypatch.setattr(
        moe.torch,
        "empty",
        lambda dimensions, **kwargs: FakeTensor(dimensions, kwargs["dtype"]),
    )

    def gemm(**kwargs):
        observed.update(kwargs)

    def reduction(target, out, token, topk, hidden, **kwargs):
        assert kwargs["is_fp8"] == (expected == "fp8")
        assert kwargs["topk_weights"] is (
            data["topk_weights"] if expected == "fp8" else None
        )
        finished.append("reduce")

    def scatter(**kwargs):
        assert kwargs["out"] is sorting[4]
        finished.append("scatter")

    def quantized_scatter(**kwargs):
        assert kwargs["flat_out_q"].shape == (256, 1536)
        assert kwargs["flat_out_scale"].shape == (256, 96)
        assert (
            kwargs["flat_out_q"].dtype
            == kwargs["flat_out_scale"].dtype
            == moe.torch.uint8
        )
        finished.append("fp4-scatter")

    monkeypatch.setattr(dispatcher, "mxfp4_moe_gemm2", gemm)
    monkeypatch.setattr(kernels, "_run_moe_reduction", reduction)
    monkeypatch.setattr(moe.aiter, "mxfp4_moe_scatter_reduce", scatter)
    monkeypatch.setattr(moe.aiter, "mxfp4_moe_scatter_reduce_q", quantized_scatter)
    result = fm._flydsl_v2_stage2_wrapper(
        FakeTensor((256, 256), moe.torch.uint8),
        None,
        data["w2_a16"],
        sorting[0],
        sorting[2],
        sorting[3],
        sorting[4],
        4,
        kernelName=f"flydsl_moe2_layout_afp4_wfp4_bf16_t128x{bn}x128_{epilog}_sbm128",
        model_dim=3072,
        inter_dim=512,
        num_experts=128,
        a2_scale=FakeTensor((256, 16), moe.torch.uint8),
        w2_scale=data["w2s_a16"],
        sorted_weights=sorting[1],
        topk_weights=data["topk_weights"],
        reverse_sorted=sorting[6],
    )
    assert result is sorting[4] and observed["out_dtype"] == expected
    assert finished == (
        []
        if epilog == "atomic"
        else (
            ["reduce"]
            if epilog == "reduce"
            else ["fp4-scatter"] if expected == "fp4" else ["scatter"]
        )
    )


import multiprocessing
import os
from multiprocessing.process import BaseProcess

_GET_CONTEXT = multiprocessing.get_context
_PROCESS_START = BaseProcess.start


def cpu_configuration_worker(payload, result_tx):
    import importlib

    from op_tests.tuning_tests._controlled_tuner_import import import_moe_tuner

    imported = import_moe_tuner()
    fm = importlib.import_module("aiter.fused_moe")
    keys, row, args, gpu = payload
    del keys, gpu
    try:
        result_tx.send(
            {
                **row,
                "snapshot": args.mxfp4_configuration,
                "production_settings": {
                    "stage2_fp8": fm._flydsl_stage2_fp8_enabled(),
                    "fp4_scatter": fm._mxfp4_intermediate_enabled(),
                    "swiglu_limit": imported.runtime_swiglu_limit(
                        args.mxfp4_configuration["swiglu_limit"], "swiglu"
                    ),
                },
                "environment": [
                    os.environ.get(name)
                    for name in (
                        "AITER_MXFP4_TUNE_SWIGLU_LIMIT",
                        "AITER_FLYDSL_STAGE2_FP8",
                        "AITER_MXFP4_INTERMEDIATE",
                    )
                ],
                "us": 1.0,
            }
        )
    finally:
        result_tx.close()


@pytest.mark.parametrize(
    "limit,fp8,fp4",
    [
        (None, None, None),
        ("", "", ""),
        (None, "1", None),
        (None, None, "1"),
        (None, "1", "1"),
        ("5", "1", "1"),
    ],
)
def test_spawned_shape_inherits_snapshot_and_stable_production_settings(
    monkeypatch, limit, fp8, fp4
):
    for name, value in zip(
        (
            "AITER_MXFP4_TUNE_SWIGLU_LIMIT",
            "AITER_FLYDSL_STAGE2_FP8",
            "AITER_MXFP4_INTERMEDIATE",
        ),
        (limit, fp8, fp4),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        monkeypatch.setattr(
            "sys.argv", ["tuner", "--mxfp4-flydsl", "--mxfp4-search-mode", "staged"]
        )
        args = tuner.parse_args()

        def cpu_only_start(process):
            assert process._target is cpu_configuration_worker
            return _PROCESS_START(process)

        monkeypatch.setattr(BaseProcess, "start", cpu_only_start)
        monkeypatch.setattr(multiprocessing, "get_context", _GET_CONTEXT)
        isolated = moe._run_staged_shapes_isolated
        monkeypatch.setattr(
            moe,
            "_run_staged_shapes_isolated",
            lambda payloads, slots, ctx: isolated(
                payloads, slots, ctx, entry=cpu_configuration_worker
            ),
        )
        results = tuner.tune(pd.DataFrame([shape(128), shape(2048)]), None, args)
    for result in results:
        assert result["environment"] == [limit, fp8, fp4]
        assert result["snapshot"] == args.mxfp4_configuration
        assert result["production_settings"] == {
            "stage2_fp8": fp8 == "1",
            "fp4_scatter": fp4 == "1",
            "swiglu_limit": 5.0 if limit == "5" else 7.0,
        }
