# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch-only coverage for non-256-aligned MXFP4 MoE shapes."""

import functools
from types import SimpleNamespace

import pytest

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import cktile_moe_stage2, get_2stage_cfgs
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode

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
):
    get_2stage_cfgs.cache_clear()
    return get_2stage_cfgs(
        TOKEN,
        model_dim,
        inter_dim,
        E,
        TOPK,
        dtypes.bf16,
        dtypes.bf16,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Swiglu,
        False,
        0,
        0,
        is_shuffled=True,
        gate_mode=gate_mode,
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
def test_unsafe_tuned_cktile_stage2_is_discarded(monkeypatch):
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
        "kernelName2": "cktile_test",
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
    a2 = SimpleNamespace(shape=(TOKEN, TOPK, INTER_DIM))
    w2 = SimpleNamespace(dtype=dtypes.fp4x2)

    with pytest.raises(NotImplementedError, match="inter_dim.*multiple of 256"):
        cktile_moe_stage2(
            a2,
            None,
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
    a2 = SimpleNamespace(shape=(TOKEN, TOPK, 512))
    w2 = SimpleNamespace(dtype=dtypes.fp4x2)

    cktile_moe_stage2(
        a2,
        None,
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
