# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import importlib

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_stage1_ready import (
    PreparedStage1Input,
    Stage1ReadyPlan,
)


def _make_plan(**overrides):
    tile_count = overrides.pop("tile_count", 8)
    partitions = overrides.pop("partitions", 2)
    state = torch.zeros(
        2 * partitions + tile_count * partitions,
        dtype=torch.int32,
    )
    values = {
        "ready": torch.zeros(1, dtype=torch.int32),
        "tile_source_masks": torch.ones(tile_count, dtype=torch.int32),
        "expert_cursor": state[:partitions],
        "completed_tiles": state[partitions : 2 * partitions],
        "tile_claimed": state[2 * partitions :].view(tile_count, partitions),
        "source_count": 2,
        "chunks_per_source": 2,
        "queue_workers": 4,
    }
    values.update(overrides)
    return Stage1ReadyPlan(**values)


def _dummy_stage1_args():
    return {
        "a": torch.empty((4, 32), dtype=torch.uint8),
        "w1": torch.empty((2, 64, 16), dtype=torch.uint8),
        "sorted_token_ids": torch.arange(8, dtype=torch.int32),
        "sorted_expert_ids": torch.arange(8, dtype=torch.int32),
        "num_valid_ids": torch.tensor([8], dtype=torch.int32),
    }


def test_stage1_ready_plan_tracks_dependency_geometry():
    plan = _make_plan()

    assert plan.dependency_count == 4


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"chunks_per_source": 17}, "fit one int32 mask"),
        ({"queue_workers": 0}, "queue_workers must be positive"),
        (
            {"tile_claimed": torch.zeros((7, 2), dtype=torch.int32)},
            "tile_claimed must have shape",
        ),
        (
            {
                "completed_tiles": torch.zeros(2, dtype=torch.int32),
                "tile_claimed": torch.zeros((8, 2), dtype=torch.int32),
            },
            "packed ABI",
        ),
    ],
)
def test_stage1_ready_plan_rejects_invalid_geometry(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_plan(**overrides)


def test_prepared_stage1_input_keeps_physical_load_ids_separate():
    plan = _make_plan()
    physical_load_ids = torch.arange(7, -1, -1, dtype=torch.int32)
    prepared = PreparedStage1Input(
        values=torch.empty((8, 32), dtype=torch.uint8),
        scales=torch.empty((8, 1), dtype=torch.uint8),
        load_sorted_ids=physical_load_ids,
        ready_plan=plan,
    )

    assert prepared.load_sorted_ids is physical_load_ids
    assert prepared.ready_plan is plan


def test_flydsl_stage1_forwards_structured_plan(monkeypatch):
    moe_kernels = importlib.import_module("aiter.ops.flydsl.moe_kernels")

    monkeypatch.setattr(
        moe_kernels,
        "_flydsl_moe_stage1_impl",
        lambda **kwargs: kwargs,
    )
    args = _dummy_stage1_args()
    plan = _make_plan()

    fused_kwargs = moe_kernels.flydsl_moe_stage1(**args, ready_plan=plan)
    assert fused_kwargs["ready_plan"] is plan

    plain_kwargs = moe_kernels.flydsl_moe_stage1(**args)
    assert plain_kwargs["ready_plan"] is None


def test_flydsl_fused_moe_wrapper_forwards_ready_plan(monkeypatch):
    fused_moe = importlib.import_module("aiter.fused_moe")
    moe_kernels = importlib.import_module("aiter.ops.flydsl.moe_kernels")
    flydsl_ops = importlib.import_module("aiter.ops.flydsl")
    plan = _make_plan()
    captured = {}

    monkeypatch.setattr(
        moe_kernels,
        "get_flydsl_kernel_params",
        lambda _name: {
            "tile_m": 32,
            "tile_n": 256,
            "tile_k": 256,
            "a_dtype": "fp8",
            "b_dtype": "fp4",
            "out_dtype": "bf16",
            "gate_mode": "interleave",
            "persist_m": 7,
        },
    )
    monkeypatch.setattr(
        flydsl_ops,
        "flydsl_moe_stage1",
        lambda **kwargs: captured.update(kwargs) or kwargs["a"],
    )
    args = _dummy_stage1_args()
    fused_moe._flydsl_stage1_wrapper(
        args["a"],
        args["w1"],
        args["w1"],
        args["sorted_token_ids"],
        args["sorted_expert_ids"],
        args["num_valid_ids"],
        None,
        2,
        kernelName="test_stage1",
        activation=ActivationType.Silu,
        ready_plan=plan,
    )

    assert captured["ready_plan"] is plan
    assert captured["persist_m"] == 7

    fused_moe._flydsl_stage1_wrapper(
        args["a"],
        args["w1"],
        args["w1"],
        args["sorted_token_ids"],
        args["sorted_expert_ids"],
        args["num_valid_ids"],
        None,
        2,
        kernelName="test_stage1",
        activation=ActivationType.Silu,
    )
    assert captured["ready_plan"] is None
    assert captured["persist_m"] == 0


def test_fused_moe_2stages_uses_physical_ids_only_for_stage1(monkeypatch):
    fused_moe = importlib.import_module("aiter.fused_moe")
    moe_kernels = importlib.import_module("aiter.ops.flydsl.moe_kernels")
    flydsl_ops = importlib.import_module("aiter.ops.flydsl")

    token_num, model_dim, inter_dim, experts, topk = 4, 32, 32, 2, 2
    logical_sorted_ids = torch.arange(token_num * topk, dtype=torch.int32)
    physical_load_ids = torch.tensor([3, 2, 1, 0, 3, 2, 1, 0], dtype=torch.int32)
    sorted_expert_ids = torch.zeros(token_num * topk, dtype=torch.int32)
    num_valid_ids = torch.tensor([token_num * topk], dtype=torch.int32)
    sorted_weights = torch.ones(token_num * topk, dtype=torch.float32)
    plan = _make_plan()
    prepared = PreparedStage1Input(
        values=torch.empty((token_num, model_dim), dtype=dtypes.fp8),
        scales=torch.empty((token_num, model_dim // 32), dtype=dtypes.fp8_e8m0),
        load_sorted_ids=physical_load_ids,
        ready_plan=plan,
    )
    hidden_states = torch.empty((token_num, model_dim), dtype=torch.bfloat16)
    w1 = torch.empty((experts, 2 * inter_dim, model_dim // 2), dtype=dtypes.fp4x2)
    w2 = torch.empty((experts, model_dim, inter_dim // 2), dtype=dtypes.fp4x2)
    w1_scale = torch.empty(
        (experts, 2 * inter_dim, model_dim // 32), dtype=dtypes.fp8_e8m0
    )
    w2_scale = torch.empty(
        (experts, model_dim, inter_dim // 32), dtype=dtypes.fp8_e8m0
    )
    moe_out = torch.empty((token_num, model_dim), dtype=torch.bfloat16)
    stage1_out = torch.empty((token_num, topk, inter_dim), dtype=dtypes.fp8)
    stage1_out_scale = torch.empty(
        (token_num * topk, inter_dim // 32), dtype=dtypes.fp8_e8m0
    )
    captured = {}

    def capture_stage2(
        inter_states,
        _w1,
        _w2,
        sorted_token_ids,
        _sorted_expert_ids,
        _num_valid_ids,
        out,
        _topk,
        **_kwargs,
    ):
        captured["stage2_inter"] = inter_states
        captured["stage2_ids"] = sorted_token_ids
        captured["stage2_out"] = out

    metadata = fused_moe.MOEMetadata(
        stage1=functools.partial(
            fused_moe._flydsl_stage1_wrapper,
            kernelName="prepared_stage1_test",
        ),
        stage2=capture_stage2,
        block_m=32,
        ksplit=1,
        fuse_quant="fp8",
        skip_inter_quant=True,
    )
    monkeypatch.setattr(fused_moe, "get_2stage_cfgs", lambda *_args, **_kwargs: metadata)
    monkeypatch.setattr(
        moe_kernels,
        "get_flydsl_kernel_params",
        lambda _name: {
            "tile_m": 32,
            "tile_n": 256,
            "tile_k": 256,
            "a_dtype": "fp8",
            "b_dtype": "fp4",
            "out_dtype": "fp8",
            "gate_mode": "interleave",
        },
    )
    monkeypatch.setattr(
        flydsl_ops,
        "flydsl_moe_stage1",
        lambda **kwargs: captured.setdefault("stage1", kwargs)
        and (stage1_out, stage1_out_scale),
    )
    monkeypatch.setattr(
        fused_moe,
        "fused_dynamic_mxfp8_quant_moe_sort",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prepared Stage1 must skip activation quantization")
        ),
    )

    result = fused_moe.fused_moe_2stages(
        hidden_states,
        w1,
        w2,
        topk,
        logical_sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_out,
        True,
        32,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        q_dtype_a=dtypes.fp8,
        q_dtype_w=dtypes.fp4x2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        _prepared_stage1=prepared,
    )

    assert result is moe_out
    assert captured["stage1"]["a"] is prepared.values
    assert captured["stage1"]["a1_scale"] is prepared.scales
    assert captured["stage1"]["sorted_token_ids"] is physical_load_ids
    assert captured["stage1"]["ready_plan"] is plan
    assert captured["stage2_inter"] is stage1_out
    assert captured["stage2_ids"] is logical_sorted_ids
    assert captured["stage2_out"] is moe_out
