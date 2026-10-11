# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch and correctness coverage for MXFP4 MoE shapes."""

import functools
from types import SimpleNamespace

import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import cktile_moe_stage2, fused_moe, get_2stage_cfgs
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale, shuffle_weight
from aiter.ops.triton.quant import dynamic_mxfp4_quant

_SKIP = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"),
    reason="CDNA (gfx942/gfx950) required for MXFP4 MoE dispatch",
)
_GFX950_ONLY = pytest.mark.skipif(
    get_gfx() != "gfx950", reason="gfx950 CK-Tile MXFP4 MoE required"
)

MODEL_DIM = 6144
INTER_DIM = 384
E = 32
TOPK = 4
TOKEN = 32


def _stage_backend(fn):
    target = fn.func if isinstance(fn, functools.partial) else fn
    name = getattr(target, "__name__", str(target))
    if "flydsl" in name:
        return "flydsl"
    if "cktile" in name:
        return "cktile"
    return name


def _dispatch(
    *,
    model_dim=MODEL_DIM,
    inter_dim=INTER_DIM,
    gate_mode=GateMode.SEPARATED,
    q_dtype_a=dtypes.bf16,
    both_weights_shuffled=True,
    dtype=dtypes.bf16,
):
    get_2stage_cfgs.cache_clear()
    return get_2stage_cfgs(
        TOKEN,
        model_dim,
        inter_dim,
        E,
        TOPK,
        dtype,
        q_dtype_a,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Swiglu,
        False,
        0,
        0,
        is_shuffled=True,
        gate_mode=gate_mode,
        opus_weights_shuffled=both_weights_shuffled,
    )


@_SKIP
@pytest.mark.parametrize(
    ("inter_dim", "want"),
    [
        (128, "flydsl"),
        (192, "cktile"),
        (256, "cktile"),
        (320, "cktile"),
        (384, "flydsl"),
        (512, "cktile"),
        (640, "flydsl"),
        (768, "cktile"),
    ],
)
def test_mxfp4_swiglu_dispatch_by_inter_dim_alignment(inter_dim, want):
    meta = _dispatch(inter_dim=inter_dim)
    got = (_stage_backend(meta.stage1), _stage_backend(meta.stage2))
    assert got == (want, want)


@_SKIP
@pytest.mark.parametrize(
    "kwargs",
    [
        {"gate_mode": GateMode.INTERLEAVE},
        {"model_dim": 1152},
    ],
)
def test_mxfp4_swiglu_does_not_reroute_to_unsupported_a16w4(kwargs):
    meta = _dispatch(**kwargs)
    assert _stage_backend(meta.stage2) == "cktile"


@_SKIP
@pytest.mark.parametrize(
    ("q_dtype_a", "gate_mode", "want"),
    [
        (dtypes.bf16, GateMode.SEPARATED, "flydsl"),
        (dtypes.bf16, GateMode.INTERLEAVE, "cktile"),
        (dtypes.fp4x2, GateMode.SEPARATED, "flydsl"),
        (dtypes.fp4x2, GateMode.INTERLEAVE, "cktile"),
        (dtypes.fp8, GateMode.SEPARATED, "cktile"),
        (dtypes.fp8, GateMode.INTERLEAVE, "cktile"),
    ],
)
def test_mxfp4_swiglu_fallback_matches_weight_layout(q_dtype_a, gate_mode, want):
    meta = _dispatch(q_dtype_a=q_dtype_a, gate_mode=gate_mode)
    assert _stage_backend(meta.stage2) == want


@_SKIP
def test_mxfp4_swiglu_fp16_output_does_not_use_a16w4():
    meta = _dispatch(dtype=dtypes.fp16)
    assert _stage_backend(meta.stage2) == "cktile"


@_SKIP
def test_mxfp4_swiglu_split_k_does_not_bypass_reroute(monkeypatch):
    monkeypatch.setenv("AITER_KSPLIT", "2")
    fused_moe_module.get_ksplit.cache_clear()
    get_2stage_cfgs.cache_clear()
    try:
        meta = _dispatch()
    finally:
        fused_moe_module.get_ksplit.cache_clear()
        get_2stage_cfgs.cache_clear()

    assert _stage_backend(meta.stage2) == "flydsl"


@_SKIP
def test_mxfp4_takeover_requires_both_weights_shuffled():
    meta = _dispatch(both_weights_shuffled=False)
    assert _stage_backend(meta.stage2) == "cktile"


@_SKIP
@pytest.mark.parametrize(
    "kernel_name2",
    ["cktile_test", "swiglu_mxfp4_bf16_cktile"],
)
def test_unsafe_tuned_cktile_stage2_is_discarded(monkeypatch, kernel_name2):
    key = (
        fused_moe_module.get_gfx_runtime(),
        fused_moe_module.get_cu_num(),
        TOKEN,
        MODEL_DIM,
        INTER_DIM,
        E,
        TOPK,
        str(ActivationType.Swiglu),
        str(dtypes.bf16),
        str(dtypes.bf16),
        str(dtypes.fp4x2),
        str(QuantType.per_1x32),
        True,
        False,
    )
    cfg = {
        "block_m": 32,
        "ksplit": 2,
        "kernelName1": "flydsl_test",
        "kernelName2": kernel_name2,
        "run_1stage": False,
    }
    monkeypatch.setattr(fused_moe_module, "cfg_2stages", ({key: cfg}, {}))
    warnings = []
    monkeypatch.setattr(
        fused_moe_module.logger,
        "warning",
        lambda message: warnings.append(str(message)),
    )
    get_2stage_cfgs.cache_clear()
    try:
        meta = _dispatch()
    finally:
        get_2stage_cfgs.cache_clear()

    assert any("discarding unsafe CK-Tile MXFP4 stage2 config" in w for w in warnings)
    assert _stage_backend(meta.stage2) == "flydsl"


@_SKIP
def test_cktile_stage2_rejects_unsafe_mxfp4_shape(monkeypatch):
    monkeypatch.setattr(
        aiter,
        "moe_cktile2stages_gemm2",
        lambda *args, **kwargs: pytest.fail("unsafe CK-Tile kernel was called"),
    )
    a2 = SimpleNamespace(shape=(TOKEN, TOPK, INTER_DIM // 2))
    w1 = SimpleNamespace(shape=(E, INTER_DIM * 2, MODEL_DIM // 2))
    w2 = SimpleNamespace(
        dtype=dtypes.fp4x2,
        shape=(E, MODEL_DIM, INTER_DIM // 2),
    )

    with pytest.raises(NotImplementedError, match="inter_dim.*multiple of 256"):
        cktile_moe_stage2(
            a2,
            w1,
            w2,
            None,
            None,
            None,
            None,
            TOPK,
            None,
            None,
            32,
        )


@_SKIP
def test_cktile_stage2_allows_aligned_mxfp4_shape(monkeypatch):
    called = []
    monkeypatch.setattr(
        aiter,
        "moe_cktile2stages_gemm2",
        lambda *args, **kwargs: called.append(True),
    )
    a2 = SimpleNamespace(shape=(TOKEN, TOPK, 128))
    w1 = SimpleNamespace(shape=(E, 512, MODEL_DIM // 2))
    w2 = SimpleNamespace(dtype=dtypes.fp4x2, shape=(E, MODEL_DIM, 128))

    cktile_moe_stage2(
        a2,
        w1,
        w2,
        None,
        None,
        None,
        None,
        TOPK,
        None,
        None,
        32,
    )
    assert called == [True]


@_GFX950_ONLY
def test_cktile_moe_missing_silu_bias_specialization_raises():
    x = torch.zeros((1, 256), dtype=torch.bfloat16, device="cuda")
    w1 = torch.zeros((1, 256, 128), dtype=torch.uint8, device="cuda").view(
        torch.float4_e2m1fn_x2
    )
    out = torch.empty((1, 1, 128), dtype=torch.bfloat16, device="cuda")
    sorted_ids = torch.zeros(16, dtype=torch.int32, device="cuda")
    expert_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    num_valid_ids = torch.ones(1, dtype=torch.int32, device="cuda")
    bias = torch.zeros((1, 256), dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError) as error:
        aiter.moe_cktile2stages_gemm1(
            x,
            w1,
            out,
            sorted_ids,
            expert_ids,
            num_valid_ids,
            topk=1,
            exp_bias=bias,
            activation=ActivationType.Silu,
            block_m=16,
            split_k=1,
        )
    assert "activation=0, has_bias=1, split_k_gt_1=0" in str(error.value)
    assert (
        "Consider adding the specialization in "
        "csrc/ck_tile_gemm_moe_2stages/gen_instances.py"
    ) in str(error.value)


@_GFX950_ONLY
def test_mxfp4_bf16_swiglu_interleaved_matches_separated(monkeypatch):
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "2147483647")
    monkeypatch.setenv("GPTOSS_SWIGLU_MXFP4_BF16_BOUND", "2147483647")
    get_2stage_cfgs.cache_clear()
    model_dim, inter_dim = 1024, 256
    assert (
        _stage_backend(
            _dispatch(
                model_dim=model_dim,
                inter_dim=inter_dim,
                gate_mode=GateMode.INTERLEAVE,
            ).stage1
        )
        == "cktile"
    )

    torch.manual_seed(0)
    x = torch.randn((TOKEN, model_dim), dtype=torch.bfloat16, device="cuda")
    w13 = torch.randn(
        (E, inter_dim * 2, model_dim), dtype=torch.bfloat16, device="cuda"
    ).mul_(0.1)
    w2 = torch.randn(
        (E, model_dim, inter_dim), dtype=torch.bfloat16, device="cuda"
    ).mul_(0.05)
    topk_ids = (
        torch.arange(TOKEN * TOPK, dtype=torch.int32, device="cuda")
        .reshape(TOKEN, TOPK)
        .remainder_(E)
    )
    topk_weights = torch.full(
        (TOKEN, TOPK), 1 / TOPK, dtype=torch.float32, device="cuda"
    )

    def quantize(weight):
        cols = weight.shape[-1]
        packed, scales = dynamic_mxfp4_quant(weight.reshape(-1, cols))
        packed = packed.view(torch.uint8).reshape(
            weight.shape[0], weight.shape[1], cols // 2
        )
        scales = scales.view(torch.uint8).reshape(-1, cols // 32)
        return packed.view(torch.float4_e2m1fn_x2), scales

    w13_q, w13_scale = quantize(w13)
    w2_q, w2_scale = quantize(w2)

    def run(gate_mode):
        interleave = gate_mode == GateMode.INTERLEAVE
        return fused_moe(
            x,
            shuffle_weight(w13_q, is_guinterleave=interleave, gate_up=True),
            shuffle_weight(w2_q, is_guinterleave=interleave, gate_up=False),
            topk_weights,
            topk_ids,
            activation=ActivationType.Swiglu,
            quant_type=QuantType.per_1x32,
            w1_scale=shuffle_scale(w13_scale, E, interleave, True),
            w2_scale=shuffle_scale(w2_scale, E, interleave, False),
            swiglu_limit=7.0,
            gate_mode=gate_mode.value,
        )

    try:
        with torch.inference_mode():
            separated = run(GateMode.SEPARATED)
            interleaved = run(GateMode.INTERLEAVE)
        assert torch.count_nonzero(separated) > 0
        torch.testing.assert_close(
            interleaved.float(), separated.float(), rtol=0, atol=0.1
        )
    finally:
        get_2stage_cfgs.cache_clear()


def _undo_shuffle_scale(scale):
    sm, sn = scale.shape
    return (
        scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
        .permute(0, 5, 3, 1, 4, 2)
        .contiguous()
        .view(sm, sn)
    )


@pytest.mark.parametrize(
    ("dtype", "src_value", "pad_value"),
    [
        (torch.uint8, 0x22, 0x7F),
        (torch.bfloat16, 2.0, 1.0),
    ],
)
def test_shuffle_scale_standard_layout_initializes_padding(dtype, src_value, pad_value):
    src = torch.tensor([[src_value]], dtype=dtype)
    padded = _undo_shuffle_scale(shuffle_scale(src))

    assert padded[0, 0].item() == src_value
    padded[0, 0] = pad_value
    assert torch.all(padded == pad_value)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
