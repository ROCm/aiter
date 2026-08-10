# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes


@pytest.mark.parametrize(
    "q_dtype_a,q_dtype_w,expected_input,expected_out,expected_scale_one",
    [
        (dtypes.fp4x2, dtypes.fp4x2, "a1_qt", "fp4", False),
        (dtypes.fp8, dtypes.fp4x2, "a1_qt_fp8_cast", "fp8", True),
        (dtypes.fp8, dtypes.fp8, "a1_qt", "fp8", False),
    ],
)
def test_v2_stage1_tasks_keep_main_payload_only_contract(
    q_dtype_a,
    q_dtype_w,
    expected_input,
    expected_out,
    expected_scale_one,
):
    from aiter.ops.flydsl.mxfp4_v2_tune_utils import (
        v2_stage1_dequant_cosine_err,
    )
    from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import FmoeTuner

    info = (
        "gfx950",
        256,
        8,
        6144,
        384,
        129,
        5,
        ActivationType.Swiglu,
        dtypes.bf16,
        q_dtype_a,
        q_dtype_w,
        QuantType.per_1x32,
        True,
        False,
    )
    tasks = object.__new__(FmoeTuner).gen_flydsl_v2_2stages_task(info, [32])
    stage1_tasks = [task for task in tasks if task[0][1] == "stage1"]

    assert stage1_tasks
    for task in stage1_tasks:
        assert len(task) == 13
        assert task[6] is FmoeTuner.run_v2_stage1_sorted_ref
        assert task[7][0] == ["ref1", "topk_ids", "sti", "sei", "n"]
        assert task[4][0][0] == expected_input
        params = task[4][-1]
        assert params["out_dtype"] == expected_out
        assert params.get("a_scale_one", False) is expected_scale_one
        assert isinstance(task[12], functools.partial)
        assert task[12].func is v2_stage1_dequant_cosine_err


def test_v2_stage1_tuner_runner_returns_payload_only(monkeypatch):
    from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

    payload = torch.arange(8, dtype=torch.uint8).view(2, 4)
    scale = torch.ones((2, 1), dtype=torch.uint8)
    monkeypatch.setattr(
        gemm_moe_tune,
        "flydsl_moe_stage1",
        lambda **_kwargs: (payload, scale),
    )
    result = gemm_moe_tune.FmoeTuner.run_flydsl_v2_stage1_out(
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty_like(payload),
        0,
        4,
        4,
        1,
        1,
        32,
        "fp8",
        ActivationType.Swiglu,
        {
            "tile_m": 32,
            "tile_n": 64,
            "tile_k": 128,
            "a_dtype": "fp8",
            "b_dtype": "fp8",
            "out_dtype": "fp8",
        },
    )

    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, payload)


@pytest.mark.l2_device
@pytest.mark.parametrize(
    "q_dtype_a,q_dtype_w",
    [
        pytest.param(dtypes.fp4x2, dtypes.fp4x2, id="q4_a4w4"),
        pytest.param(dtypes.fp8, dtypes.fp4x2, id="q7_a8w4"),
        pytest.param(dtypes.fp8, dtypes.fp8, id="q9_a8w8"),
    ],
)
def test_v2_stage1_real_kernel_payload_correctness(q_dtype_a, q_dtype_w):
    from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import FmoeTuner

    info = (
        "gfx950",
        256,
        8,
        256,
        128,
        4,
        2,
        ActivationType.Swiglu,
        dtypes.bf16,
        q_dtype_a,
        q_dtype_w,
        QuantType.per_1x32,
        True,
        False,
    )
    tasks = object.__new__(FmoeTuner).gen_flydsl_v2_2stages_task(info, [32])
    task = next(task for task in tasks if task[0][1] == "stage1")
    data = task[1](*task[2], device="cuda")
    arg_keys, *arg_rest = task[4]
    result = task[3](
        *(tuple(data[key] for key in arg_keys) + tuple(arg_rest)),
        **task[5],
    )
    ref_keys, *ref_rest = task[7]
    reference = task[6](
        *(tuple(data[key] for key in ref_keys) + tuple(ref_rest)),
        **task[8],
    )
    torch.cuda.synchronize()
    err = task[12](reference, result, printLog=False)

    assert isinstance(result, torch.Tensor)
    assert err <= 0.1
