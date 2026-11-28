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
from aiter.fused_moe import cktile_moe_stage2, get_2stage_cfgs
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


def _tuned_key(
    inter_dim=INTER_DIM,
    activation=ActivationType.Swiglu,
    q_dtype_a=dtypes.bf16,
    *,
    model_dim=MODEL_DIM,
    experts=E,
    topk=TOPK,
):
    return (
        fused_moe_module.get_gfx_runtime(),
        fused_moe_module.get_cu_num(),
        TOKEN,
        model_dim,
        inter_dim,
        experts,
        topk,
        str(activation),
        str(dtypes.bf16),
        str(q_dtype_a),
        str(dtypes.fp4x2),
        str(QuantType.per_1x32),
        True,
        False,
    )


def _dispatch(
    *,
    model_dim=MODEL_DIM,
    inter_dim=INTER_DIM,
    gate_mode=GateMode.SEPARATED,
    q_dtype_a=dtypes.bf16,
    both_weights_shuffled=True,
    dtype=dtypes.bf16,
    activation=ActivationType.Swiglu,
    has_stage1_bias=False,
    has_stage2_bias=False,
    gate_up_interleave=None,
    block_size_M=None,
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
        activation,
        False,
        0,
        0,
        is_shuffled=True,
        gate_mode=gate_mode,
        has_stage1_bias=has_stage1_bias,
        has_stage2_bias=has_stage2_bias,
        gate_up_interleave=gate_up_interleave,
        opus_weights_shuffled=both_weights_shuffled,
        block_size_M=block_size_M,
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
    cfg = {
        "block_m": 32,
        "ksplit": 2,
        "kernelName1": "flydsl_test",
        "kernelName2": kernel_name2,
        "run_1stage": False,
    }
    monkeypatch.setattr(fused_moe_module, "cfg_2stages", ({_tuned_key(): cfg}, {}))
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


# Tuned FlyDSL a16w4 rows: the a16w4 port ignores per-expert bias and reads W1 in the
# standard gate/up layout, and neither is part of the row key.
A16W4_INTER = 1536
A16W4_ROW = {
    "block_m": 16,
    "ksplit": 0,
    "kernelName1": "flydsl_moe1_abf16_wfp4_bf16_t16x64x256_w2_xcd1_kw2",
    "kernelName2": "flydsl_moe2_abf16_wfp4_bf16_t16x256x256_atomic_bnt2",
    "run_1stage": False,
}
A4W4_ROW = {
    "block_m": 32,
    "ksplit": 0,
    "kernelName1": "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_bnt0_xcd4_kw4_fp4",
    "kernelName2": "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_sbm32",
    "run_1stage": False,
}


def _collect_warnings(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        fused_moe_module.logger,
        "warning",
        lambda message: warnings.append(str(message)),
    )
    return warnings


def _a16w4_discards(warnings):
    return [w for w in warnings if "discarding a16w4 FlyDSL tuned config" in w]


def _a16w4_call_tensors(tokens=TOKEN, experts=4, topk=2, model_dim=256, inter_dim=256):
    """Uninitialized CPU inputs of a small a16w4 MoE call: bf16 activations and
    preshuffled fp4x2 weights, enough to reach the tuned-row lookup."""
    w1 = torch.empty((experts, 2 * inter_dim, model_dim // 2), dtype=dtypes.fp4x2)
    w2 = torch.empty((experts, model_dim, inter_dim // 2), dtype=dtypes.fp4x2)
    w1.is_shuffled = w2.is_shuffled = True
    return (
        torch.empty((tokens, model_dim), dtype=dtypes.bf16),
        w1,
        w2,
        torch.ones((tokens, topk), dtype=torch.float32),
        torch.zeros((tokens, topk), dtype=torch.int32),
    )


def _dispatch_tuned(
    monkeypatch,
    row,
    *,
    activation=ActivationType.Swiglu,
    q_dtype_a=dtypes.bf16,
    **call,
):
    key = _tuned_key(A16W4_INTER, activation, q_dtype_a)
    monkeypatch.setattr(fused_moe_module, "cfg_2stages", ({key: row}, {}))
    warnings = _collect_warnings(monkeypatch)
    try:
        meta = _dispatch(
            inter_dim=A16W4_INTER, activation=activation, q_dtype_a=q_dtype_a, **call
        )
    finally:
        get_2stage_cfgs.cache_clear()
    return meta, _a16w4_discards(warnings)


@_SKIP
@pytest.mark.parametrize("block_size_M", [None, 16], ids=["no-override", "same-size"])
def test_a16w4_flydsl_tuned_row_is_kept_for_plain_calls(monkeypatch, block_size_M):
    meta, discarded = _dispatch_tuned(monkeypatch, A16W4_ROW, block_size_M=block_size_M)
    assert not discarded
    assert (_stage_backend(meta.stage1), _stage_backend(meta.stage2)) == (
        "flydsl",
        "flydsl",
    )
    assert meta.block_m == A16W4_ROW["block_m"]


@_SKIP
@pytest.mark.parametrize(
    ("call", "want_bias", "want_backend"),
    [
        ({"has_stage1_bias": True}, (True, False), ("cktile", "cktile")),
        ({"has_stage2_bias": True}, (False, True), ("cktile", "cktile")),
        (
            {"gate_mode": GateMode.INTERLEAVE, "gate_up_interleave": True},
            (False, False),
            ("cktile", "cktile"),
        ),
        ({"gate_mode": GateMode.INTERLEAVE}, (False, False), ("cktile", "cktile")),
        ({"gate_up_interleave": True}, (False, False), ("cktile", "cktile")),
        (
            {"has_stage1_bias": True, "block_size_M": 32},
            (True, False),
            ("cktile", "cktile"),
        ),
        # The a16w4 stage 1 launches with w_layout="standard" and no bias
        # argument for every activation, so non-Swiglu callers discard too.
        # Each keeps the default route it had without a row (the SiTUv2
        # default is FlyDSL with default block_m and no bias in the metadata;
        # the kernel-name assertion distinguishes the discard from a kept row).
        (
            {"activation": ActivationType.Situv2, "has_stage1_bias": True},
            (False, False),
            ("flydsl", "flydsl"),
        ),
        (
            {"activation": ActivationType.Silu, "gate_up_interleave": True},
            (False, False),
            ("ck_moe_stage1", "ck_moe_stage2_fwd"),
        ),
    ],
    ids=[
        "bias1",
        "bias2",
        "interleave",
        "interleave-mode-only",
        "interleave-tag-only",
        "bias1-and-override",
        "situv2-bias1",
        "silu-interleave",
    ],
)
def test_a16w4_flydsl_tuned_row_is_discarded_for_bias_or_interleave(
    monkeypatch, call, want_bias, want_backend
):
    meta, discarded = _dispatch_tuned(monkeypatch, A16W4_ROW, **call)
    assert len(discarded) == 1
    assert (_stage_backend(meta.stage1), _stage_backend(meta.stage2)) == want_backend
    stage1_kn = getattr(meta.stage1, "keywords", {}).get("kernelName", "")
    assert stage1_kn != A16W4_ROW["kernelName1"]
    assert (meta.has_bias, meta.stage2_has_bias) == want_bias


@_SKIP
@pytest.mark.parametrize(
    ("row", "activation", "q_dtype_a", "call"),
    [
        (A4W4_ROW, ActivationType.Swiglu, dtypes.fp4x2, {"has_stage1_bias": True}),
        (A4W4_ROW, ActivationType.Swiglu, dtypes.fp4x2, {"block_size_M": 16}),
    ],
    ids=[
        "swiglu-a4w4-bias",
        "swiglu-a4w4-block",
    ],
)
def test_flydsl_tuned_row_outside_guarded_a16w4_prefix_is_kept(
    monkeypatch, row, activation, q_dtype_a, call
):
    meta, discarded = _dispatch_tuned(
        monkeypatch, row, activation=activation, q_dtype_a=q_dtype_a, **call
    )
    assert not discarded
    assert meta.block_m == row["block_m"]
    assert _stage_backend(meta.stage1) == "flydsl"


# A caller's block_size_M sizes moe_sorting and both gemms, but the row key does not
# carry it: the a16w4 FlyDSL stage 1 would run 16-row tiles on a 32-row layout.
@_SKIP
@pytest.mark.parametrize(
    ("activation", "want_backend"),
    [
        (ActivationType.Swiglu, ("cktile", "cktile")),
        (ActivationType.Situv2, ("flydsl", "flydsl")),
    ],
    ids=["swiglu", "situv2"],
)
def test_a16w4_flydsl_tuned_row_is_discarded_for_other_block_size(
    monkeypatch, activation, want_backend
):
    meta, discarded = _dispatch_tuned(
        monkeypatch, A16W4_ROW, activation=activation, block_size_M=32
    )
    assert len(discarded) == 1 and "block_size_M=32" in discarded[0]
    assert (_stage_backend(meta.stage1), _stage_backend(meta.stage2)) == want_backend


class _Lookup(Exception):
    pass


# The override must reach the row lookup: fused_moe sorts with it right after.
@_SKIP
@pytest.mark.parametrize(
    ("block_size_M", "want"), [(-1, None), (16, 16), (32, 32)], ids=["none", "16", "32"]
)
def test_fused_moe_impl_passes_block_size_to_the_row_lookup(
    monkeypatch, block_size_M, want
):
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("block_size_M", "not passed"))
        raise _Lookup

    monkeypatch.setattr(fused_moe_module, "get_2stage_cfgs", spy)
    hidden, w1, w2, weights, ids = _a16w4_call_tensors(tokens=4)
    with pytest.raises(_Lookup):
        fused_moe_module._fused_moe_impl(
            hidden,
            w1,
            w2,
            weights,
            ids,
            activation=ActivationType.Swiglu.value,
            quant_type=QuantType.per_1x32.value,
            block_size_M=block_size_M,
            dtype=dtypes.bf16,
        )
    assert seen == [want]


# fused_moe_2stages takes the sorted inputs and their block size from the caller, so
# its own lookup must see that size, with no fused_moe and no metadata transform.
@_SKIP
@pytest.mark.parametrize(
    ("block_size_M", "want", "n_discards"),
    [(32, "cktile", 1), (16, "flydsl", 0)],
    ids=["32", "16"],
)
def test_fused_moe_2stages_selects_by_block_size(
    monkeypatch, block_size_M, want, n_discards
):
    experts, topk, model_dim, inter_dim = 4, 2, 256, 256
    key = _tuned_key(
        inter_dim,
        model_dim=model_dim,
        experts=experts,
        topk=topk,
    )
    monkeypatch.setattr(fused_moe_module, "cfg_2stages", ({key: A16W4_ROW}, {}))
    warnings = _collect_warnings(monkeypatch)
    seen = []

    def lookup(*args, **kwargs):
        meta = get_2stage_cfgs(*args, **kwargs)
        seen.append((_stage_backend(meta.stage1), _stage_backend(meta.stage2)))
        raise _Lookup

    monkeypatch.setattr(fused_moe_module, "get_2stage_cfgs", lookup)
    get_2stage_cfgs.cache_clear()
    hidden, w1, w2, weights, ids = _a16w4_call_tensors(
        experts=experts, topk=topk, model_dim=model_dim, inter_dim=inter_dim
    )
    try:
        # `ids` stands in for the sorted inputs: the lookup raises before they are read.
        with pytest.raises(_Lookup):
            fused_moe_module.fused_moe_2stages(
                hidden,
                w1,
                w2,
                topk,
                ids,
                weights,
                ids,
                ids,
                torch.empty_like(hidden),
                True,
                block_size_M,
                activation=ActivationType.Swiglu,
                quant_type=QuantType.per_1x32,
                q_dtype_a=dtypes.bf16,
                q_dtype_w=dtypes.fp4x2,
            )
    finally:
        get_2stage_cfgs.cache_clear()
    assert seen == [(want, want)]
    assert len(_a16w4_discards(warnings)) == n_discards


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


def _load_fallback_tables(monkeypatch, csv_name):
    """Load primary/fallback dicts from one model CSV via the merged-config path.

    AITER_CONFIGS.get_config_file() is LRU-cached with the env var read inside
    the body (aiter/jit/core.py), so an AITER_CONFIG_FMOE env override is
    ignored once another test resolved the path first. Replace the module's
    AITER_CONFIGS with a stub pointing at the target CSV instead.
    """
    import os
    import types

    tune_file = os.path.join(
        os.path.dirname(aiter.__file__), "configs", "model_configs", csv_name
    )
    if not os.path.exists(tune_file):
        pytest.skip(f"{tune_file} not found")

    monkeypatch.setattr(
        fused_moe_module,
        "AITER_CONFIGS",
        types.SimpleNamespace(AITER_CONFIG_FMOE_FILE=tune_file),
    )
    monkeypatch.setattr(fused_moe_module, "cfg_2stages", None)
    get_2stage_cfgs.cache_clear()

    try:
        # Trigger global config loading with config_file=None
        get_2stage_cfgs(
            16,
            6144,
            1536,
            129,
            5,
            dtypes.bf16,
            dtypes.bf16,
            dtypes.fp4x2,
            QuantType.per_1x32,
            True,
            ActivationType.Swiglu,
            False,
            0,
            0,
            config_file=None,
        )
    finally:
        get_2stage_cfgs.cache_clear()

    assert fused_moe_module.cfg_2stages is not None
    return fused_moe_module.cfg_2stages


def _fallback_rows_with_kn1_prefix(fallback, *prefixes):
    return [
        (k, cfg.get("kernelName1"), cfg.get("kernelName2"))
        for k, cfg in fallback.items()
        if isinstance(cfg.get("kernelName1"), str)
        and cfg["kernelName1"].startswith(prefixes)
    ]


# _flydsl_stage1_wrapper parses tile parameters from the kernel name and compiles
# with the caller's activation, so tuned FlyDSL rows legitimately serve
# cross-activation fallback matches — the only row property a caller can violate
# is the sorting layout (block_size_M vs the row's block_m).
@_SKIP
@pytest.mark.parametrize(
    ("block_size_M", "want_discards"),
    [(16, 0), (32, 1)],
    ids=["matching-block", "mismatched-block"],
)
def test_fallback_cross_activation_match_validates_block_size(
    monkeypatch, block_size_M, want_discards
):
    # The tuned CSV only has gfx950/cu_num=256 rows; pin the runtime to match.
    monkeypatch.setattr(fused_moe_module, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(fused_moe_module, "get_cu_num", lambda: 256)

    _, fallback = _load_fallback_tables(
        monkeypatch, "minimax_m3_a16w4_tuned_fmoe.csv"
    )
    assert _fallback_rows_with_kn1_prefix(
        fallback, "flydsl_moe1_abf16_wfp4_"
    ), "expected FlyDSL a16w4 rows in the activation-disabled fallback map"

    # A SiTUv2 call at the tuned Swiglu key (token=16, H=6144, I=1536, E=129,
    # topk=5, block_m=16) matches only through the fallback map.
    meta, discarded = _fallback_lookup(
        monkeypatch, 16, ActivationType.Situv2, block_size_M=block_size_M
    )
    assert len(discarded) == want_discards, f"discards: {discarded}"
    if want_discards:
        assert "block_size_M=32" in discarded[0]
    else:
        assert (_stage_backend(meta.stage1), _stage_backend(meta.stage2)) == (
            "flydsl",
            "flydsl",
        )
        assert meta.block_m == 16


def _fallback_lookup(monkeypatch, token, activation, **kwargs):
    """Look up the tuned a16w4 key for `token` after the tables are loaded."""
    warnings = _collect_warnings(monkeypatch)
    get_2stage_cfgs.cache_clear()
    try:
        meta = get_2stage_cfgs(
            token,
            6144,
            1536,
            129,
            5,
            dtypes.bf16,
            dtypes.bf16,
            dtypes.fp4x2,
            QuantType.per_1x32,
            True,
            activation,
            False,
            0,
            0,
            is_shuffled=True,
            opus_weights_shuffled=True,
            **kwargs,
        )
    finally:
        get_2stage_cfgs.cache_clear()
    return meta, _a16w4_discards(warnings)


# The a16w4 stage 1 launches with w_layout="standard" and no bias argument for
# every activation, so a cross-activation fallback match must be rejected when
# the caller's weights are interleaved -- and kept when the layout matches.
@_SKIP
@pytest.mark.parametrize(
    ("call", "want_discards"),
    [({}, 0), ({"gate_up_interleave": True, "gate_mode": GateMode.INTERLEAVE}, 1)],
    ids=["plain-kept", "interleave-discarded"],
)
def test_fallback_cross_activation_match_validates_layout(
    monkeypatch, call, want_discards
):
    monkeypatch.setattr(fused_moe_module, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(fused_moe_module, "get_cu_num", lambda: 256)
    _load_fallback_tables(monkeypatch, "minimax_m3_a16w4_tuned_fmoe.csv")

    # A SiLU call at the tuned Swiglu token-1 key (H=6144, I=1536, E=129,
    # topk=5) matches only through the fallback map.
    meta, discarded = _fallback_lookup(monkeypatch, 1, ActivationType.Silu, **call)
    assert len(discarded) == want_discards, f"discards: {discarded}"
    if want_discards:
        assert "bias or interleaved" in discarded[0]
    else:
        assert (_stage_backend(meta.stage1), _stage_backend(meta.stage2)) == (
            "flydsl",
            "flydsl",
        )
        assert meta.block_m == 16


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
