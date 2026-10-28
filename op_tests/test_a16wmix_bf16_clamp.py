# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""a16wmix BF16×BF16 SwiGLU must honour swiglu_limit once the dtype gate is open.

SiLU coverage lives in a follow-up commit that teaches the epilogue to clamp.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    _can_reroute_bf16_to_a16wmix,
    _flydsl_stage1_wrapper,
    fused_moe,
    get_2stage_cfgs,
)
from aiter.ops.shuffle import shuffle_weight

_A16W16_S1 = "flydsl_moe1_abf16_wbf16_bf16_t16x128x128"
_A16W16_S2 = "flydsl_moe2_abf16_wbf16_bf16_t16x128x128_atomic"
_INACTIVE = (None, 0.0, float("inf"))

pytestmark_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="a16wmix clamp check needs a ROCm GPU",
)


@pytest.fixture
def gfx942(monkeypatch):
    monkeypatch.setattr("aiter.fused_moe.get_gfx", lambda: "gfx942")


def _a16w16_cfg_kwargs(token=4, *, shuffled=True, limit=7.0, g1u1=True):
    return dict(
        token=token,
        model_dim=256,
        inter_dim=128,
        expert=4,
        topk=2,
        dtype=dtypes.bf16,
        q_dtype_a=dtypes.bf16,
        q_dtype_w=dtypes.bf16,
        q_type=QuantType.No,
        use_g1u1=g1u1,
        activation=ActivationType.Swiglu,
        doweight_stage1=False,
        hidden_pad=0,
        intermediate_pad=0,
        is_shuffled=shuffled,
        opus_weights_shuffled=shuffled,
        swiglu_limit=limit,
    )


def _reroute_base(**overrides):
    base = dict(
        model_dim=256,
        inter_dim=128,
        dtype=dtypes.bf16,
        q_dtype_a=dtypes.bf16,
        q_dtype_w=dtypes.bf16,
        q_type=QuantType.No,
        activation=ActivationType.Swiglu,
        is_shuffled=True,
        use_g1u1=True,
        doweight_stage1=False,
        swiglu_limit=7.0,
        has_stage1_bias=False,
        hidden_pad=0,
        intermediate_pad=0,
    )
    base.update(overrides)
    return base


def test_a16w16_kernel_names_are_registered():
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

    p1 = get_flydsl_kernel_params(_A16W16_S1)
    p2 = get_flydsl_kernel_params(_A16W16_S2)
    assert p1 is not None, _A16W16_S1
    assert p2 is not None, _A16W16_S2
    assert p1["a_dtype"] == "bf16" and p1["b_dtype"] == "bf16"
    assert p2["mode"] == "atomic"


def test_silu_mul_batch_shared_helper_is_unclamped():
    from aiter.ops.flydsl.kernels import mixed_moe_gemm_2stage_common as mm
    from aiter.ops.flydsl.kernels.act import silu_mul_batch

    helper_src = inspect.getsource(silu_mul_batch)
    assert "clamp" not in helper_src
    mixed_src = inspect.getsource(mm)
    assert "clamp_gate_up" in mixed_src
    assert "silu_mul_batch" in mixed_src


def test_can_reroute_swiglu_requires_finite_limit_and_shuffle():
    base = _reroute_base()
    assert _can_reroute_bf16_to_a16wmix(**base)
    assert not _can_reroute_bf16_to_a16wmix(**{**base, "swiglu_limit": None})
    assert not _can_reroute_bf16_to_a16wmix(**{**base, "swiglu_limit": 0.0})
    assert not _can_reroute_bf16_to_a16wmix(**{**base, "swiglu_limit": float("inf")})
    assert not _can_reroute_bf16_to_a16wmix(**{**base, "is_shuffled": False})
    assert not _can_reroute_bf16_to_a16wmix(
        **{**base, "activation": ActivationType.Silu}
    )


def test_can_reroute_force_env_ignores_sentinel(monkeypatch):
    monkeypatch.setenv("AITER_FORCE_A16WMIX_BF16", "1")
    base = _reroute_base(swiglu_limit=None)
    assert _can_reroute_bf16_to_a16wmix(**base)
    monkeypatch.setenv("AITER_FORCE_A16WMIX_BF16", "0")
    assert not _can_reroute_bf16_to_a16wmix(**base)


def test_bf16_swiglu_limit_selects_a16wmix(gfx942, monkeypatch):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    get_2stage_cfgs.cache_clear()
    try:
        metadata = get_2stage_cfgs(**_a16w16_cfg_kwargs())
        stage1 = getattr(metadata.stage1, "func", metadata.stage1)
        assert stage1 is _flydsl_stage1_wrapper
        assert metadata.stage1.keywords["kernelName"] == _A16W16_S1
        assert metadata.stage2.keywords["kernelName"] == _A16W16_S2
    finally:
        get_2stage_cfgs.cache_clear()


@pytest.mark.parametrize("limit", _INACTIVE)
def test_swiglu_sentinels_do_not_select_a16wmix(gfx942, monkeypatch, limit):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    get_2stage_cfgs.cache_clear()
    try:
        metadata = get_2stage_cfgs(**_a16w16_cfg_kwargs(limit=limit))
        stage1 = getattr(metadata.stage1, "func", metadata.stage1)
        assert stage1 is not _flydsl_stage1_wrapper
    finally:
        get_2stage_cfgs.cache_clear()


def test_force_env_selects_a16wmix_with_sentinel(gfx942, monkeypatch):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    monkeypatch.setenv("AITER_FORCE_A16WMIX_BF16", "1")
    get_2stage_cfgs.cache_clear()
    try:
        metadata = get_2stage_cfgs(**_a16w16_cfg_kwargs(limit=None))
        stage1 = getattr(metadata.stage1, "func", metadata.stage1)
        assert stage1 is _flydsl_stage1_wrapper
    finally:
        get_2stage_cfgs.cache_clear()


@pytest.mark.parametrize("tokens", [4, 256])
@pytestmark_gpu
def test_a16wmix_bf16_swiglu_clamp_binds(monkeypatch, tokens):
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() not in ("gfx942", "gfx950"):
        pytest.skip("a16wmix a16w16 not asserted on this arch")
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    torch.manual_seed(0)
    limit = 1.0
    experts, model_dim, inter_dim, topk = 4, 256, 128, 2
    x = torch.randn(tokens, model_dim, device="cuda", dtype=dtypes.bf16)
    w1 = torch.randn(experts, 2 * inter_dim, model_dim, device="cuda", dtype=dtypes.bf16)
    w2 = torch.randn(experts, model_dim, inter_dim, device="cuda", dtype=dtypes.bf16)
    with torch.no_grad():
        probe = F.linear(x[:1].float(), w1[0].float()).abs().max().item()
        w1.mul_(max(4.0 * limit / max(probe, 1e-3), 1.0))
        gate0, up0 = F.linear(x.float(), w1[0].float()).chunk(2, dim=-1)
        gate_min, gate_max = gate0.min().item(), gate0.max().item()
        up_min, up_max = up0.min().item(), up0.max().item()
    weights, ids = torch.softmax(
        torch.randn(tokens, experts, device="cuda"), dim=-1
    ).topk(topk, dim=-1)
    ids = ids.to(torch.int32)

    get_2stage_cfgs.cache_clear()
    try:
        metadata = get_2stage_cfgs(**_a16w16_cfg_kwargs(token=tokens, limit=limit))
        stage1 = getattr(metadata.stage1, "func", metadata.stage1)
        assert stage1 is _flydsl_stage1_wrapper
        kn = metadata.stage1.keywords["kernelName"]
        actual = fused_moe(
            x,
            shuffle_weight(w1),
            shuffle_weight(w2),
            weights,
            ids,
            activation=ActivationType.Swiglu,
            quant_type=QuantType.No,
            swiglu_limit=limit,
        )
    finally:
        get_2stage_cfgs.cache_clear()

    def _ref(*, lim):
        expected = torch.zeros(tokens, model_dim, device=x.device, dtype=torch.float32)
        for expert in range(experts):
            rows, slots = torch.where(ids == expert)
            if rows.numel() == 0:
                continue
            gate, up = F.linear(x[rows].float(), w1[expert].float()).chunk(2, dim=-1)
            if lim is not None:
                gate = gate.clamp(max=lim)
                up = up.clamp(min=-lim, max=lim)
            hidden = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
            y = F.linear(hidden, w2[expert].float())
            expected.index_add_(0, rows, y * weights[rows, slots, None].float())
        return expected

    clamped = _ref(lim=limit)
    unclamped = _ref(lim=None)
    err_c = (actual.float() - clamped).abs().max().item()
    err_u = (actual.float() - unclamped).abs().max().item()
    stats = (
        f"kn={kn} tokens={tokens} "
        f"gate_min={gate_min:.4g} gate_max={gate_max:.4g} "
        f"up_min={up_min:.4g} up_max={up_max:.4g} "
        f"err_clamped={err_c:.4g} err_unclamped={err_u:.4g}"
    )
    print(stats)
    assert max(abs(gate_min), abs(gate_max), abs(up_min), abs(up_max)) > limit, stats
    assert err_c < err_u, stats
