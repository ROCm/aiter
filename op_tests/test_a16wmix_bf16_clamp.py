# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""a16wmix BF16×BF16 must honour swiglu_limit for SiLU and SwiGLU.

Dispatch tests do not need a GPU. Bind / asymmetry / force-override launches
need ROCm.
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
    ck_moe_stage1,
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


def _arch():
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx()


def _a16w16_cfg_kwargs(
    token=4,
    *,
    shuffled=True,
    limit=7.0,
    g1u1=True,
    activation=ActivationType.Silu,
):
    return {
        "token": token,
        "model_dim": 256,
        "inter_dim": 128,
        "expert": 4,
        "topk": 2,
        "dtype": dtypes.bf16,
        "q_dtype_a": dtypes.bf16,
        "q_dtype_w": dtypes.bf16,
        "q_type": QuantType.No,
        "use_g1u1": g1u1,
        "activation": activation,
        "doweight_stage1": False,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "is_shuffled": shuffled,
        "opus_weights_shuffled": shuffled,
        "swiglu_limit": limit,
    }


def _reroute_base(**overrides):
    base = {
        "model_dim": 256,
        "inter_dim": 128,
        "dtype": dtypes.bf16,
        "q_dtype_a": dtypes.bf16,
        "q_dtype_w": dtypes.bf16,
        "q_type": QuantType.No,
        "activation": ActivationType.Swiglu,
        "is_shuffled": True,
        "use_g1u1": True,
        "doweight_stage1": False,
        "swiglu_limit": 7.0,
        "has_stage1_bias": False,
        "hidden_pad": 0,
        "intermediate_pad": 0,
    }
    base.update(overrides)
    return base


def _gate_up_stats(x, w1):
    with torch.no_grad():
        gate0, up0 = F.linear(x.float(), w1[0].float()).chunk(2, dim=-1)
    return (
        gate0.min().item(),
        gate0.max().item(),
        up0.min().item(),
        up0.max().item(),
    )


def _silu_hidden(gate, up, *, limit=None, symmetric=False):
    if limit is not None:
        if symmetric:
            gate = gate.clamp(min=-limit, max=limit)
        else:
            gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def _swiglu_hidden(gate, up, *, limit=None):
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(1.702 * gate) * (up + 1.0)


def _moe_ref(x, w1, w2, ids, weights, *, act, limit=None, symmetric=False):
    tokens, model_dim = x.shape
    experts = w1.shape[0]
    expected = torch.zeros(tokens, model_dim, device=x.device, dtype=torch.float32)
    for expert in range(experts):
        rows, slots = torch.where(ids == expert)
        if rows.numel() == 0:
            continue
        gate, up = F.linear(x[rows].float(), w1[expert].float()).chunk(2, dim=-1)
        if act == "silu":
            hidden = _silu_hidden(gate, up, limit=limit, symmetric=symmetric)
        else:
            hidden = _swiglu_hidden(gate, up, limit=limit)
        y = F.linear(hidden, w2[expert].float())
        expected.index_add_(0, rows, y * weights[rows, slots, None].float())
    return expected


def _run_fused(x, w1, w2, weights, ids, *, limit, activation):
    get_2stage_cfgs.cache_clear()
    try:
        return fused_moe(
            x,
            shuffle_weight(w1),
            shuffle_weight(w2),
            weights,
            ids,
            activation=activation,
            quant_type=QuantType.No,
            swiglu_limit=limit,
        )
    finally:
        get_2stage_cfgs.cache_clear()


def _bind_inputs(tokens, *, seed=0, limit=1.0, negative_gate=False):
    torch.manual_seed(seed)
    experts, model_dim, inter_dim, topk = 4, 256, 128, 2
    x = torch.randn(tokens, model_dim, device="cuda", dtype=dtypes.bf16)
    w1 = torch.randn(
        experts, 2 * inter_dim, model_dim, device="cuda", dtype=dtypes.bf16
    )
    w2 = torch.randn(experts, model_dim, inter_dim, device="cuda", dtype=dtypes.bf16)
    if negative_gate:
        # Expert 0: gate ≈ -25 (must stay unclamped on the below-only side).
        # Expert 1: gate ≈ +8 so the upper bound still binds.
        w1[:, :inter_dim, :] = 0
        w1[:, inter_dim:, :] = 0
        w1[0, :inter_dim, :] = -25.0 / model_dim
        w1[0, inter_dim:, :] = 3.0 / model_dim
        w1[1, :inter_dim, :] = 8.0 / model_dim
        w1[1, inter_dim:, :] = 3.0 / model_dim
        x = torch.ones(tokens, model_dim, device="cuda", dtype=dtypes.bf16)
        ids = torch.zeros(tokens, topk, device="cuda", dtype=torch.int32)
        ids[:, 1] = 1
        weights = torch.full((tokens, topk), 0.5, device="cuda", dtype=torch.float32)
    else:
        with torch.no_grad():
            probe = F.linear(x[:1].float(), w1[0].float()).abs().max().item()
            scale = max(4.0 * limit / max(probe, 1e-3), 1.0)
            w1.mul_(scale)
        weights, ids = torch.softmax(
            torch.randn(tokens, experts, device="cuda"), dim=-1
        ).topk(topk, dim=-1)
        ids = ids.to(torch.int32)
    return x, w1, w2, weights, ids


def test_a16w16_kernel_names_are_registered():
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

    p1 = get_flydsl_kernel_params(_A16W16_S1)
    p2 = get_flydsl_kernel_params(_A16W16_S2)
    assert p1 is not None, _A16W16_S1
    assert p2 is not None, _A16W16_S2
    assert p1["a_dtype"] == "bf16" and p1["b_dtype"] == "bf16"
    assert p1["tile_m"] == 16 and p1["tile_n"] == 128 and p1["tile_k"] == 128
    assert p2["mode"] == "atomic"


def test_a16w16_force_reduce_stays_atomic(monkeypatch):
    """BF16 stage2 has no reduce epilogue, so FORCE_REDUCE must keep the zero-init."""
    import functools

    from aiter.fused_moe import (
        _flydsl_stage2_wrapper,
        _stage2_honors_runtime_reduce_override,
        _stage2_runtime_uses_route_reduce,
    )

    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "1")
    stage2 = functools.partial(_flydsl_stage2_wrapper, kernelName=_A16W16_S2)
    assert not _stage2_honors_runtime_reduce_override(stage2)
    assert not _stage2_runtime_uses_route_reduce(stage2, 4, 8192)


def test_silu_mul_batch_shared_helper_is_unclamped():
    """mixed_moe clamps before calling silu_mul_batch; do not put the clamp in
    the shared helper (that would double-clamp mixed_moe)."""
    from aiter.ops.flydsl.kernels import mixed_moe_gemm_2stage_common as mm
    from aiter.ops.flydsl.kernels.act import silu_mul_batch

    helper_src = inspect.getsource(silu_mul_batch)
    assert "clamp" not in helper_src
    mixed_src = inspect.getsource(mm)
    assert "clamp_gate_up" in mixed_src
    assert "silu_mul_batch" in mixed_src


@pytest.mark.parametrize(
    "activation",
    [ActivationType.Swiglu, ActivationType.Silu],
)
def test_can_reroute_requires_finite_limit_and_shuffle(activation):
    base = _reroute_base(activation=activation)
    assert _can_reroute_bf16_to_a16wmix(**base)
    assert not _can_reroute_bf16_to_a16wmix(**{**base, "swiglu_limit": None})
    assert not _can_reroute_bf16_to_a16wmix(**{**base, "swiglu_limit": 0.0})
    assert not _can_reroute_bf16_to_a16wmix(**{**base, "swiglu_limit": float("inf")})
    assert not _can_reroute_bf16_to_a16wmix(**{**base, "is_shuffled": False})
    assert not _can_reroute_bf16_to_a16wmix(**{**base, "use_g1u1": False})


def test_can_reroute_force_env_ignores_sentinel(monkeypatch):
    monkeypatch.setenv("AITER_FORCE_A16WMIX_BF16", "1")
    base = _reroute_base(swiglu_limit=None, activation=ActivationType.Silu)
    assert _can_reroute_bf16_to_a16wmix(**base)
    monkeypatch.setenv("AITER_FORCE_A16WMIX_BF16", "0")
    assert not _can_reroute_bf16_to_a16wmix(**base)


@pytest.mark.parametrize(
    "activation",
    [ActivationType.Swiglu, ActivationType.Silu],
)
def test_bf16_limit_selects_a16wmix(gfx942, monkeypatch, activation):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    get_2stage_cfgs.cache_clear()
    try:
        metadata = get_2stage_cfgs(**_a16w16_cfg_kwargs(activation=activation))
        stage1 = getattr(metadata.stage1, "func", metadata.stage1)
        assert stage1 is _flydsl_stage1_wrapper
        assert metadata.stage1.keywords["kernelName"] == _A16W16_S1
        assert metadata.stage2.keywords["kernelName"] == _A16W16_S2
    finally:
        get_2stage_cfgs.cache_clear()


@pytest.mark.parametrize("limit", _INACTIVE)
@pytest.mark.parametrize(
    "activation",
    [ActivationType.Swiglu, ActivationType.Silu],
)
def test_sentinels_do_not_select_a16wmix(gfx942, monkeypatch, limit, activation):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    get_2stage_cfgs.cache_clear()
    try:
        metadata = get_2stage_cfgs(
            **_a16w16_cfg_kwargs(limit=limit, activation=activation)
        )
        stage1 = getattr(metadata.stage1, "func", metadata.stage1)
        assert stage1 is ck_moe_stage1, stage1
        assert not metadata.run_1stage
    finally:
        get_2stage_cfgs.cache_clear()


def test_force_env_selects_a16wmix_with_sentinel(gfx942, monkeypatch):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    monkeypatch.setenv("AITER_FORCE_A16WMIX_BF16", "1")
    get_2stage_cfgs.cache_clear()
    try:
        metadata = get_2stage_cfgs(
            **_a16w16_cfg_kwargs(limit=None, activation=ActivationType.Silu)
        )
        stage1 = getattr(metadata.stage1, "func", metadata.stage1)
        assert stage1 is _flydsl_stage1_wrapper
        assert metadata.stage1.keywords["kernelName"].startswith(
            "flydsl_moe1_abf16_wbf16_"
        )
    finally:
        get_2stage_cfgs.cache_clear()


def test_unshuffled_does_not_select_a16wmix(gfx942, monkeypatch):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    get_2stage_cfgs.cache_clear()
    try:
        metadata = get_2stage_cfgs(**_a16w16_cfg_kwargs(shuffled=False))
        stage1 = getattr(metadata.stage1, "func", metadata.stage1)
        assert stage1 is not _flydsl_stage1_wrapper
    finally:
        get_2stage_cfgs.cache_clear()


def _assert_route_a16w16(tokens, activation, limit):
    get_2stage_cfgs.cache_clear()
    metadata = get_2stage_cfgs(
        **_a16w16_cfg_kwargs(token=tokens, activation=activation, limit=limit)
    )
    stage1 = getattr(metadata.stage1, "func", metadata.stage1)
    assert stage1 is _flydsl_stage1_wrapper, stage1
    kn = metadata.stage1.keywords["kernelName"]
    assert kn.startswith("flydsl_moe1_abf16_wbf16_"), kn
    return kn


@pytest.mark.parametrize("tokens", [4, 256])
@pytest.mark.parametrize(
    "activation,act_name",
    [
        (ActivationType.Silu, "silu"),
        (ActivationType.Swiglu, "swiglu"),
    ],
)
@pytestmark_gpu
def test_a16wmix_bf16_clamp_binds(monkeypatch, tokens, activation, act_name):
    arch = _arch()
    if arch not in ("gfx942", "gfx950"):
        pytest.skip(f"a16wmix a16w16 not asserted on {arch}")

    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    limit = 1.0
    x, w1, w2, weights, ids = _bind_inputs(tokens, limit=limit)
    kn = _assert_route_a16w16(tokens, activation, limit)
    actual = _run_fused(x, w1, w2, weights, ids, limit=limit, activation=activation)
    clamped = _moe_ref(x, w1, w2, ids, weights, act=act_name, limit=limit)
    unclamped = _moe_ref(x, w1, w2, ids, weights, act=act_name, limit=None)
    err_c = (actual.float() - clamped).abs().max().item()
    err_u = (actual.float() - unclamped).abs().max().item()
    gate_min, gate_max, up_min, up_max = _gate_up_stats(x, w1)
    stats = (
        f"kn={kn} tokens={tokens} act={act_name} "
        f"gate_min={gate_min:.4g} gate_max={gate_max:.4g} "
        f"up_min={up_min:.4g} up_max={up_max:.4g} "
        f"err_clamped={err_c:.4g} err_unclamped={err_u:.4g}"
    )
    print(stats)
    assert max(abs(gate_min), abs(gate_max), abs(up_min), abs(up_max)) > limit, stats
    assert err_c < err_u, stats
    huge = _run_fused(x, w1, w2, weights, ids, limit=1e9, activation=activation)
    assert not torch.equal(actual, huge), stats + " limit=1.0 vs 1e9 identical"


@pytestmark_gpu
def test_a16wmix_silu_asymmetric_clamp_leaves_negative_gate(monkeypatch):
    """gate.clamp(max=L) must NOT floor large negative gates at -L."""
    arch = _arch()
    if arch not in ("gfx942", "gfx950"):
        pytest.skip(f"not asserted on {arch}")
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    limit = 1.0
    tokens = 8
    x, w1, w2, weights, ids = _bind_inputs(tokens, limit=limit, negative_gate=True)
    _assert_route_a16w16(tokens, ActivationType.Silu, limit)
    gate_min, gate_max, up_min, up_max = _gate_up_stats(x, w1)
    assert gate_min <= -20.0, (gate_min, gate_max, up_min, up_max)
    actual = _run_fused(
        x, w1, w2, weights, ids, limit=limit, activation=ActivationType.Silu
    )
    asym = _moe_ref(x, w1, w2, ids, weights, act="silu", limit=limit, symmetric=False)
    sym = _moe_ref(x, w1, w2, ids, weights, act="silu", limit=limit, symmetric=True)
    err_as = (actual.float() - asym).abs().max().item()
    err_sy = (actual.float() - sym).abs().max().item()
    stats = (
        f"gate_min={gate_min:.4g} gate_max={gate_max:.4g} "
        f"up_min={up_min:.4g} up_max={up_max:.4g} "
        f"err_asymmetric={err_as:.4g} err_symmetric={err_sy:.4g}"
    )
    print(stats)
    assert err_as < err_sy, stats
    assert err_sy > 1.0, stats + " symmetric ref did not diverge; case is too weak"


@pytestmark_gpu
def test_force_env_same_kernel_with_and_without_limit(monkeypatch):
    """P1 isolation: force a16wmix even when the sentinel would skip it."""
    arch = _arch()
    if arch not in ("gfx942", "gfx950"):
        pytest.skip(f"not asserted on {arch}")
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    monkeypatch.setenv("AITER_FORCE_A16WMIX_BF16", "1")
    tokens = 4
    x, w1, w2, weights, ids = _bind_inputs(tokens, limit=1.0)
    get_2stage_cfgs.cache_clear()
    try:
        md_none = get_2stage_cfgs(
            **_a16w16_cfg_kwargs(
                token=tokens, limit=None, activation=ActivationType.Silu
            )
        )
        md_lim = get_2stage_cfgs(
            **_a16w16_cfg_kwargs(
                token=tokens, limit=1.0, activation=ActivationType.Silu
            )
        )
    finally:
        get_2stage_cfgs.cache_clear()
    kn_none = md_none.stage1.keywords["kernelName"]
    kn_lim = md_lim.stage1.keywords["kernelName"]
    assert kn_none == kn_lim, (kn_none, kn_lim)
    assert kn_none.startswith("flydsl_moe1_abf16_wbf16_")
    out_none = _run_fused(
        x, w1, w2, weights, ids, limit=None, activation=ActivationType.Silu
    )
    out_lim = _run_fused(
        x, w1, w2, weights, ids, limit=1.0, activation=ActivationType.Silu
    )
    assert not torch.equal(out_none, out_lim), "force-path limit=None vs 1.0 identical"


# Per-rank Command A shape the clamped server ran: H=4096, I=512, E=128, top-8.
_SERVER = {"model_dim": 4096, "inter_dim": 512, "expert": 128, "topk": 8}
_SERVER_TOKENS = 32
_SERVER_S1 = "flydsl_moe1_abf16_wbf16_bf16_t16x128x128"


def _server_route(limit):
    get_2stage_cfgs.cache_clear()
    try:
        metadata = get_2stage_cfgs(
            **{
                **_a16w16_cfg_kwargs(
                    token=_SERVER_TOKENS, limit=limit, activation=ActivationType.Silu
                ),
                **_SERVER,
            }
        )
    finally:
        get_2stage_cfgs.cache_clear()
    stage1 = getattr(metadata.stage1, "func", metadata.stage1)
    assert stage1 is _flydsl_stage1_wrapper, stage1
    return metadata.stage1.keywords["kernelName"]


def _run_fused_shuffled(x, w1s, w2s, weights, ids, *, limit):
    get_2stage_cfgs.cache_clear()
    try:
        return fused_moe(
            x,
            w1s,
            w2s,
            weights,
            ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.No,
            swiglu_limit=limit,
        )
    finally:
        get_2stage_cfgs.cache_clear()


def _routed_gate_up_absmax(x, w1, ids):
    gate_max, up_absmax = float("-inf"), 0.0
    for expert in ids.unique().tolist():
        rows, _ = torch.where(ids == expert)
        gate, up = F.linear(x[rows].float(), w1[expert].float()).chunk(2, dim=-1)
        gate_max = max(gate_max, gate.max().item())
        up_absmax = max(up_absmax, up.abs().max().item())
    return gate_max, up_absmax


@pytestmark_gpu
def test_a16wmix_server_shape_clamp_binds(monkeypatch):
    arch = _arch()
    if arch not in ("gfx942", "gfx950"):
        pytest.skip(f"not asserted on {arch}")
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    limit = 7.0
    hidden, inter, experts, topk = (
        _SERVER[k] for k in ("model_dim", "inter_dim", "expert", "topk")
    )
    torch.manual_seed(20261016)
    x = torch.randn(_SERVER_TOKENS, hidden, device="cuda", dtype=dtypes.bf16)
    # Gate/up std ~50 against a bound of 7, as in the served checkpoint.
    w1 = (torch.randn(experts, 2 * inter, hidden, device="cuda") * 0.75).to(dtypes.bf16)
    w2 = (torch.randn(experts, hidden, inter, device="cuda") / inter**0.5).to(
        dtypes.bf16
    )
    weights, ids = torch.softmax(
        torch.randn(_SERVER_TOKENS, experts, device="cuda"), dim=-1
    ).topk(topk, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    ids = ids.to(torch.int32)

    kn = _server_route(limit)
    assert kn == _SERVER_S1, kn
    gate_max, up_absmax = _routed_gate_up_absmax(x, w1, ids)
    assert gate_max > 2 * limit and up_absmax > 2 * limit, (gate_max, up_absmax)

    w1s, w2s = shuffle_weight(w1), shuffle_weight(w2)
    actual = _run_fused_shuffled(x, w1s, w2s, weights, ids, limit=limit).float()
    loose = _run_fused_shuffled(x, w1s, w2s, weights, ids, limit=1e9).float()
    clamped = _moe_ref(x, w1, w2, ids, weights, act="silu", limit=limit)
    unclamped = _moe_ref(x, w1, w2, ids, weights, act="silu", limit=None)

    scale_c = clamped.abs().max().item()
    scale_u = unclamped.abs().max().item()
    err_c = (actual - clamped).abs().max().item()
    err_u = (actual - unclamped).abs().max().item()
    err_loose_u = (loose - unclamped).abs().max().item()
    stats = (
        f"kn={kn} gate_max={gate_max:.4g} up_absmax={up_absmax:.4g} "
        f"|clamped|={scale_c:.4g} |unclamped|={scale_u:.4g} "
        f"err_clamped={err_c:.4g} err_unclamped={err_u:.4g} "
        f"limit1e9_err_unclamped={err_loose_u:.4g}"
    )
    print(stats)
    assert err_c <= 2e-2 * scale_c, stats
    assert err_u >= 10 * scale_c, stats
    assert not torch.equal(actual, loose), stats + " limit=7 vs 1e9 identical"
    assert err_loose_u <= 2e-2 * scale_u, stats


@pytestmark_gpu
def test_a16wmix_server_shape_asymmetric_negative_gate(monkeypatch):
    """Gate -25 at limit 1.0 must pass through silu(-25), not silu(-1)."""
    arch = _arch()
    if arch not in ("gfx942", "gfx950"):
        pytest.skip(f"not asserted on {arch}")
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    limit = 1.0
    hidden, inter, experts, topk = (
        _SERVER[k] for k in ("model_dim", "inter_dim", "expert", "topk")
    )
    w1 = torch.zeros(experts, 2 * inter, hidden, device="cuda", dtype=dtypes.bf16)
    w2 = torch.zeros(experts, hidden, inter, device="cuda", dtype=dtypes.bf16)
    # Expert 0: gate -25 (below-only side must stay unclamped); expert 1: gate +8
    # so the upper bound binds too. Up is 3 on both, clamped to 1. k/4096 is exact
    # in bf16, so x=1 gives the pre-activations exactly.
    w1[0, :inter, :] = -25.0 / hidden
    w1[0, inter:, :] = 3.0 / hidden
    w1[1, :inter, :] = 8.0 / hidden
    w1[1, inter:, :] = 3.0 / hidden
    w2[:2] = 1.0 / 64
    x = torch.ones(_SERVER_TOKENS, hidden, device="cuda", dtype=dtypes.bf16)
    ids = (
        torch.arange(topk, device="cuda", dtype=torch.int32)
        .unsqueeze(0)
        .repeat(_SERVER_TOKENS, 1)
    )
    weights = torch.full((_SERVER_TOKENS, topk), 0.5, device="cuda")

    assert _server_route(limit) == _SERVER_S1
    gate0 = F.linear(x[:1].float(), w1[0].float())[0, :inter]
    assert gate0.max().item() <= -20.0, gate0.max().item()

    actual = _run_fused_shuffled(
        x, shuffle_weight(w1), shuffle_weight(w2), weights, ids, limit=limit
    ).float()
    asym = _moe_ref(x, w1, w2, ids, weights, act="silu", limit=limit)
    sym = _moe_ref(x, w1, w2, ids, weights, act="silu", limit=limit, symmetric=True)
    err_as = (actual - asym).abs().max().item()
    err_sy = (actual - sym).abs().max().item()
    stats = f"err_asymmetric={err_as:.4g} err_symmetric={err_sy:.4g}"
    print(stats)
    assert err_as < 0.05, stats
    assert err_sy > 0.5, stats
