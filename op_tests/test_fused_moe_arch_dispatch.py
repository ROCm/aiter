# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from importlib import import_module

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes

fused_moe = import_module("aiter.fused_moe")


def _dsv4_mxfp4_cfg(**overrides):
    cfg = {
        "token": 1,
        "model_dim": 4096,
        "inter_dim": 2048,
        "expert": 256,
        "topk": 6,
        "dtype": dtypes.bf16,
        "q_dtype_a": dtypes.bf16,
        "q_dtype_w": dtypes.fp4x2,
        "q_type": QuantType.per_1x32,
        "use_g1u1": True,
        "activation": ActivationType.Silu,
        "doweight_stage1": False,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "is_shuffled": False,
        "gate_mode": fused_moe.GateMode.SEPARATED,
        "is_ep": False,
        "has_stage2_bias": False,
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.parametrize("gfx", ["gfx942", "gfx1100", "gfx1201", "gfx1250"])
def test_ck_2stage_per_1x32_fails_closed(monkeypatch, gfx):
    monkeypatch.setattr(fused_moe, "get_gfx", lambda: gfx)

    with pytest.raises(
        RuntimeError,
        match=rf"only executable on gfx950.*architecture-empty kernel on {gfx}",
    ):
        fused_moe._require_ck_2stage_quant_arch(QuantType.per_1x32)


def test_ck_2stage_per_1x32_admits_gfx950(monkeypatch):
    monkeypatch.setattr(fused_moe, "get_gfx", lambda: "gfx950")

    fused_moe._require_ck_2stage_quant_arch(QuantType.per_1x32)


@pytest.mark.parametrize("gfx", ["gfx942", "gfx1100", "gfx1201", "gfx1250"])
def test_ck_2stage_other_quantization_is_unchanged(monkeypatch, gfx):
    monkeypatch.setattr(fused_moe, "get_gfx", lambda: gfx)

    fused_moe._require_ck_2stage_quant_arch(QuantType.per_128x128)


@pytest.mark.parametrize("inter_dim", [2048, 256])
@pytest.mark.parametrize("token", [1, 64, 65, 2048, 4096])
def test_dsv4_unshuffled_mxfp4_dispatch_selects_gfx1100_triton(
    monkeypatch, inter_dim, token
):
    monkeypatch.setattr(fused_moe, "get_gfx", lambda: "gfx1100")
    fused_moe.get_2stage_cfgs.cache_clear()

    metadata = fused_moe.get_2stage_cfgs(
        **_dsv4_mxfp4_cfg(token=token, inter_dim=inter_dim)
    )

    assert metadata.stage1.func is fused_moe._triton_mxfp4_a16w4_stage1
    assert metadata.stage2.func is fused_moe._triton_mxfp4_a16w4_stage2
    assert metadata.block_m == 16
    assert metadata.ksplit == 1
    assert metadata.prequant is False
    assert metadata.skip_inter_quant is True
    assert metadata.stage1.keywords["config"] == fused_moe._GFX1100_TRITON_MXFP4_CONFIG
    assert metadata.stage2.keywords["config"] == fused_moe._GFX1100_TRITON_MXFP4_CONFIG


@pytest.mark.parametrize(
    "override",
    [
        {"token": 0},
        {"token": 4097},
        {"model_dim": 7168},
        {"inter_dim": 1024},
        {"expert": 8},
        {"expert": 32},
        {"topk": 5},
        {"dtype": dtypes.fp16},
        {"q_dtype_a": dtypes.fp4x2},
        {"q_dtype_w": dtypes.fp8},
        {"q_type": QuantType.per_1x128},
        {"use_g1u1": False},
        {"activation": ActivationType.Gelu},
        {"doweight_stage1": True},
        {"hidden_pad": 64},
        {"intermediate_pad": 128},
        {"inter_dim": 256, "intermediate_pad": 64},
        {"is_shuffled": True},
        {"gate_mode": fused_moe.GateMode.INTERLEAVE},
        {"is_ep": True},
        {"has_stage2_bias": True},
    ],
)
def test_gfx1100_triton_mxfp4_contract_rejects_nonmatching_inputs(
    monkeypatch, override
):
    monkeypatch.setattr(fused_moe, "get_gfx", lambda: "gfx1100")

    assert (
        fused_moe._gfx1100_triton_mxfp4_metadata(**_dsv4_mxfp4_cfg(**override)) is None
    )


def test_gfx1100_triton_mxfp4_contract_does_not_claim_gfx950(monkeypatch):
    monkeypatch.setattr(fused_moe, "get_gfx", lambda: "gfx950")

    assert fused_moe._gfx1100_triton_mxfp4_metadata(**_dsv4_mxfp4_cfg()) is None


def test_dsv4_unshuffled_mxfp4_dispatch_still_rejects_a4w4_gfx1100(monkeypatch):
    monkeypatch.setattr(fused_moe, "get_gfx", lambda: "gfx1100")
    monkeypatch.setattr(fused_moe, "get_cu_num", lambda: 96)
    monkeypatch.setattr(fused_moe, "is_flydsl_available", lambda: False)
    fused_moe.get_2stage_cfgs.cache_clear()
    fused_moe.get_block_size_M.cache_clear()

    with pytest.raises(RuntimeError, match="architecture-empty kernel on gfx1100"):
        fused_moe.get_2stage_cfgs(**_dsv4_mxfp4_cfg(q_dtype_a=dtypes.fp4x2))


def test_dsv4_gfx1100_entrypoint_selects_bf16_activation(monkeypatch):
    class FakeTensor:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    captured = {}

    def capture_cfg(*args, **_kwargs):
        captured["q_dtype_a"] = args[6]
        raise LookupError("dispatch captured")

    monkeypatch.setattr(fused_moe, "get_gfx", lambda: "gfx1100")
    monkeypatch.setattr(fused_moe, "is_flydsl_available", lambda: False)
    monkeypatch.setattr(fused_moe, "get_2stage_cfgs", capture_cfg)

    with pytest.raises(LookupError, match="dispatch captured"):
        fused_moe._fused_moe_impl(
            hidden_states=FakeTensor((1, 4096), dtypes.bf16),
            w1=FakeTensor((256, 4096, 2048), dtypes.fp4x2),
            w2=FakeTensor((256, 4096, 1024), dtypes.fp4x2),
            topk_weight=FakeTensor((1, 6), dtypes.fp32),
            topk_ids=FakeTensor((1, 6), dtypes.i32),
            quant_type=QuantType.per_1x32.value,
        )

    assert captured["q_dtype_a"] == dtypes.bf16


def test_triton_mxfp4_unit_scales_are_cached_per_device_and_expert_count():
    fused_moe._TRITON_MXFP4_UNIT_SCALES.clear()
    tensor = torch.empty(1)

    a_scale_1, b_scale_1 = fused_moe._triton_mxfp4_unit_scales(tensor, 256)
    a_scale_2, b_scale_2 = fused_moe._triton_mxfp4_unit_scales(tensor, 256)
    _, b_scale_3 = fused_moe._triton_mxfp4_unit_scales(tensor, 8)

    assert a_scale_1 is a_scale_2
    assert b_scale_1 is b_scale_2
    assert a_scale_1.shape == (1,)
    assert b_scale_1.shape == (256,)
    assert b_scale_3.shape == (8,)
    assert torch.equal(a_scale_1, torch.ones_like(a_scale_1))
    assert torch.equal(b_scale_1, torch.ones_like(b_scale_1))


@pytest.mark.parametrize(
    "extra_kwargs, expected",
    [
        ({}, (False, 1, 0.0)),
        (
            {
                "sorted_token_ids_are_packed": True,
                "sorted_top_k": 6,
                "swiglu_limit": 10.0,
            },
            (True, 6, 10.0),
        ),
    ],
)
def test_triton_mxfp4_silu_forwards_sorting_and_clamp_contract(
    monkeypatch, extra_kwargs, expected
):
    triton_moe = import_module("aiter.ops.triton.moe.moe_op_mxfp4_silu_fused")
    captured = {}

    class FakeKernel:
        def __getitem__(self, _grid):
            def launch(*_args, **kwargs):
                captured.update(kwargs)

            return launch

    monkeypatch.setattr(triton_moe, "_fused_moe_kernel_mxfp4_silu", FakeKernel())
    triton_moe.fused_moe_mxfp4_silu(
        torch.empty((1, 128), dtype=torch.bfloat16),
        torch.empty((1, 64, 64), dtype=torch.uint8),
        torch.empty((1, 32), dtype=torch.bfloat16),
        torch.ones((1,), dtype=torch.float32),
        torch.ones((1,), dtype=torch.float32),
        None,
        torch.ones((1, 64, 4), dtype=torch.uint8),
        torch.ones((1, 1), dtype=torch.float32),
        torch.zeros((1, 1), dtype=torch.int64),
        torch.zeros((16,), dtype=torch.int32),
        torch.zeros((1,), dtype=torch.int32),
        torch.tensor([16], dtype=torch.int32),
        False,
        1,
        False,
        False,
        fused_moe._GFX1100_TRITON_MXFP4_CONFIG,
        triton_moe.tl.bfloat16,
        **extra_kwargs,
    )

    assert captured["SORTED_IDS_PACKED"] is expected[0]
    assert captured["SORTED_TOP_K"] == expected[1]
    assert captured["SWIGLU_LIMIT"] == expected[2]


@pytest.mark.parametrize(
    "extra_kwargs, expected",
    [
        ({}, (False, 1)),
        (
            {"sorted_token_ids_are_packed": True, "sorted_top_k": 6},
            (True, 6),
        ),
    ],
)
def test_triton_mxfp4_forwards_sorting_contract(monkeypatch, extra_kwargs, expected):
    triton_moe = import_module("aiter.ops.triton.moe.moe_op_mxfp4")
    captured = {}

    class FakeKernel:
        def __getitem__(self, _grid):
            def launch(*_args, **kwargs):
                captured.update(kwargs)

            return launch

    monkeypatch.setattr(triton_moe, "_fused_moe_kernel_mxfp4", FakeKernel())
    triton_moe.fused_moe_mxfp4(
        torch.empty((1, 128), dtype=torch.bfloat16),
        torch.empty((1, 64, 64), dtype=torch.uint8),
        torch.empty((1, 1, 64), dtype=torch.bfloat16),
        torch.ones((1,), dtype=torch.float32),
        torch.ones((1,), dtype=torch.float32),
        None,
        torch.ones((1, 64, 4), dtype=torch.uint8),
        torch.ones((1, 1), dtype=torch.float32),
        torch.zeros((1, 1), dtype=torch.int64),
        torch.zeros((16,), dtype=torch.int32),
        torch.zeros((1,), dtype=torch.int32),
        torch.tensor([16], dtype=torch.int32),
        True,
        1,
        False,
        False,
        fused_moe._GFX1100_TRITON_MXFP4_CONFIG,
        triton_moe.tl.bfloat16,
        **extra_kwargs,
    )

    assert captured["SORTED_IDS_PACKED"] is expected[0]
    assert captured["SORTED_TOP_K"] == expected[1]


def test_reduce_topk_launches_contiguous_fp32_group_reduction(monkeypatch):
    triton_reduce = import_module("aiter.ops.triton.moe.reduce")
    captured = {}

    class FakeKernel:
        def __getitem__(self, grid):
            captured["grid"] = grid

            def launch(*args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs

            return launch

    monkeypatch.setattr(triton_reduce, "_reduce_grouped", FakeKernel())
    x = torch.empty((3, 6, 1024), dtype=torch.bfloat16)
    out = torch.empty((3, 1024), dtype=torch.bfloat16)

    returned = triton_reduce.reduce_topk(x, out)

    assert returned is out
    assert captured["grid"] == (6,)
    assert captured["args"][8:12] == (1, 18, 1024, 2)
    assert captured["kwargs"]["K"] == 6
    assert captured["kwargs"]["EVEN_N"] is True
    assert captured["kwargs"]["USE_TDM"] is False
    assert captured["kwargs"]["HAS_EXT_RESIDUAL"] is False
