# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch-only coverage for non-256-aligned MXFP4 MoE shapes."""

import functools
from types import SimpleNamespace

import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    _mxmoe_cfg_shape_reason,
    cktile_moe_stage2,
    get_2stage_cfgs,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale

_SKIP = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"),
    reason="CDNA (gfx942/gfx950) required for MXFP4 MoE dispatch",
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


# Tuned MXMOE rows whose kernels cannot run the runtime shape. Kimi-K3 TP=8 has
# model_dim 3584 and inter_dim 384, which only a BK=128 gemm2 contracts.
_MXMOE_G1 = "flydsl_mxmoe_g1_a4w4_16x128x256_f16in_hpf_nt_situv2"
_MXMOE_G1_BN256 = "flydsl_mxmoe_g1_a4w4_16x256x256_situv2"
_MXMOE_G2_BK128 = "flydsl_mxmoe_g2_a4w4_16x256x128_atomic"
_MXMOE_G2_BK256 = "flydsl_mxmoe_g2_a4w4_16x256x256_atomic"
_V2_G2_K128 = "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16"
_V2_G2_K256 = "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16"
_V2_G2_N256 = "flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_atomic_sbm16"


@pytest.mark.parametrize(
    ("kn1", "kn2", "model_dim", "inter_dim", "reason"),
    [
        # Valid rows are kept.
        (_MXMOE_G1, _MXMOE_G2_BK128, 3584, 384, None),
        (_MXMOE_G1, _MXMOE_G2_BK256, 3584, 512, None),
        (_MXMOE_G1, _V2_G2_K128, 3584, 384, None),
        ("", _V2_G2_N256, 3584, 384, None),
        # Native gemm1 (_MXMOE_G1 is BN=128, BK=256): BK must be 256, K = model_dim
        # by BK with 2..32 K tiles, and N = 2 * inter_dim by BN.
        (_MXMOE_G1, _MXMOE_G2_BK128, 3712, 384, "model_dim % 256"),
        (_MXMOE_G1, "", 8448, 384, "model_dim // 256 <= 32"),
        (_MXMOE_G1, "", 256, 384, "2 * k_wave <= model_dim // 256"),
        ("flydsl_mxmoe_g1_a4w4_16x128x128_f16in_situv2", "", 3584, 384, "BK=128"),
        (_MXMOE_G1, "", 3584, 320, None),
        (_MXMOE_G1_BN256, "", 3584, 320, "2*inter_dim % 256"),
        (_MXMOE_G1, _MXMOE_G2_BK128, 3584, 320, "inter_dim % 128"),
        # Native gemm2: inter_dim must be a multiple of its BK, model_dim of its
        # BN, and BN must be 256.
        (_MXMOE_G1, _MXMOE_G2_BK256, 3584, 384, "inter_dim % 256"),
        (_MXMOE_G1, _MXMOE_G2_BK128, 3584, 448, "inter_dim % 128"),
        ("", _MXMOE_G2_BK128, 3584, 448, "inter_dim % 128"),
        ("", _MXMOE_G2_BK128, 3712, 384, "model_dim % 256"),
        ("", "flydsl_mxmoe_g2_a4w4_16x128x128_atomic", 3584, 384, "unsupported BN/BK"),
        # Layout-v2 gemm2: K = inter_dim by tile_k and N = model_dim by tile_n,
        # with N capped by the launcher's HIDDEN_MAX.
        (_MXMOE_G1, _V2_G2_K256, 3584, 384, "inter_dim % 256"),
        ("", _V2_G2_N256, 3712, 384, "model_dim % 256"),
        ("", _V2_G2_K128, 8192, 384, None),
        ("", _V2_G2_K128, 8448, 384, "model_dim <= 8192"),
    ],
)
def test_mxmoe_cfg_shape_reason(kn1, kn2, model_dim, inter_dim, reason):
    got = _mxmoe_cfg_shape_reason(kn1, kn2, model_dim, inter_dim)
    if reason is None:
        assert got is None, got
    else:
        assert got is not None and reason in got, got


@_SKIP
@pytest.mark.parametrize(
    ("kernel_name2", "kept"),
    [(_MXMOE_G2_BK128, True), (_MXMOE_G2_BK256, False)],
)
def test_stale_tuned_mxmoe_row_is_discarded(monkeypatch, kernel_name2, kept):
    """A tuned row whose gemm2 cannot contract inter_dim falls back to the
    default heuristics instead of failing at compile time; a valid row is kept."""
    token, model_dim, inter_dim, experts, topk = 8, 3584, 384, 896, 16
    key = (
        fused_moe_module.get_gfx_runtime(),
        fused_moe_module.get_cu_num(),
        token,
        model_dim,
        inter_dim,
        experts,
        topk,
        str(ActivationType.Situv2),
        str(dtypes.bf16),
        str(dtypes.fp4x2),
        str(dtypes.fp4x2),
        str(QuantType.per_1x32),
        True,
        False,
    )
    cfg = {
        "block_m": 16,
        "ksplit": 0,
        "kernelName1": _MXMOE_G1,
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
        meta = get_2stage_cfgs(
            token,
            model_dim,
            inter_dim,
            experts,
            topk,
            dtypes.bf16,
            dtypes.fp4x2,
            dtypes.fp4x2,
            QuantType.per_1x32,
            True,
            ActivationType.Situv2,
            False,
            0,
            0,
            is_shuffled=True,
            gate_mode=GateMode.SEPARATED,
            opus_weights_shuffled=True,
        )
    finally:
        get_2stage_cfgs.cache_clear()

    used = (meta.stage2.keywords or {}).get("kernelName2")
    discarded = any("discarding tuned config" in w for w in warnings)
    assert (used == kernel_name2) == kept, used
    assert discarded != kept, warnings


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
