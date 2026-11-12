# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 per_Token / per_Tensor MoE must apply swiglu_limit.

Before the fix the limit was dropped silently: ASM 1-stage and CK 2-stage
ran an unclamped SiLU. A finite limit now routes stage1 to the CK
runtime_swiglu_limit module shared with per_1x128, which clamps gate/up
before SiLU. None, 0.0 and +inf take the unclamped path.
"""

import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    asm_stage1,
    ck_moe_stage1,
    fused_moe,
    fused_topk,
    get_2stage_cfgs,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.version.hip
    or "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="Exercises the gfx942 FP8 per_Token / per_Tensor MoE dispatch",
)

MODEL_DIM, INTER_DIM, EXPERTS, TOPK = 2048, 768, 128, 8
# Gate/up pre-activations need |x| well above the limit for the clamp to matter.
GATE_STD = 8.0


@pytest.fixture(autouse=True)
def fresh_dispatch():
    get_2stage_cfgs.cache_clear()
    yield
    get_2stage_cfgs.cache_clear()


@pytest.fixture(scope="module", params=[QuantType.per_Token, QuantType.per_Tensor])
def weights(request):
    q = request.param
    torch.manual_seed(0)
    w1 = torch.randn(
        EXPERTS, 2 * INTER_DIM, MODEL_DIM, device="cuda", dtype=dtypes.bf16
    ) * (GATE_STD / MODEL_DIM**0.5)
    w2 = (
        torch.randn(EXPERTS, MODEL_DIM, INTER_DIM, device="cuda", dtype=dtypes.bf16)
        / INTER_DIM**0.5
    )
    if q == QuantType.per_Tensor:
        w1_qt, w1_scale = aiter.pertoken_quant(
            w1.view(EXPERTS, -1), quant_dtype=dtypes.fp8
        )
        w2_qt, w2_scale = aiter.pertoken_quant(
            w2.view(EXPERTS, -1), quant_dtype=dtypes.fp8
        )
        w1_qt, w2_qt = w1_qt.view(w1.shape), w2_qt.view(w2.shape)
    else:
        torch_quant = aiter.get_torch_quant(q)
        w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp8)
        w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp8)
    return {
        "quant_type": q,
        "w1_qt": w1_qt,
        "w2_qt": w2_qt,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "w1_sh": shuffle_weight(w1_qt, layout=(16, 16)),
        "w2_sh": shuffle_weight(w2_qt, layout=(16, 16)),
    }


def _calc_diff(x, y):
    x, y = x.double(), y.double()
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())


def _reference(w, x, topk_weights, topk_ids, limit):
    q = w["quant_type"]
    torch_quant = aiter.get_torch_quant(q)
    a1_qt, a1_scale = torch_quant(x, quant_dtype=dtypes.fp8)
    out1 = torch_moe_stage1(
        a1_qt,
        w["w1_qt"],
        w["w2_qt"],
        topk_weights,
        topk_ids,
        dtype=dtypes.bf16,
        activation=ActivationType.Silu,
        quant_type=q,
        a1_scale=a1_scale,
        w1_scale=w["w1_scale"],
        swiglu_limit=limit,
    )
    a2_qt, a2_scale = torch_quant(out1, quant_dtype=dtypes.fp8)
    return torch_moe_stage2(
        a2_qt.view(x.shape[0], TOPK, -1),
        w["w1_qt"],
        w["w2_qt"],
        topk_weights,
        topk_ids,
        dtype=dtypes.bf16,
        quant_type=q,
        w2_scale=w["w2_scale"],
        a2_scale=a2_scale,
    )


def _inputs(tokens):
    torch.manual_seed(tokens)
    x = torch.randn(tokens, MODEL_DIM, device="cuda", dtype=dtypes.bf16)
    score = torch.randn(tokens, EXPERTS, device="cuda", dtype=dtypes.bf16)
    topk_weights, topk_ids = fused_topk(x, score, TOPK, True)
    return x, topk_weights, topk_ids


def _fused(w, x, topk_weights, topk_ids, limit):
    return fused_moe(
        x,
        w["w1_sh"],
        w["w2_sh"],
        topk_weights,
        topk_ids,
        w1_scale=w["w1_scale"],
        w2_scale=w["w2_scale"],
        quant_type=w["quant_type"],
        activation=ActivationType.Silu,
        swiglu_limit=limit,
    )


# 64+ tokens selects ASM 1-stage for per_Token without a limit.
@pytest.mark.parametrize("limit", [7.0, None, 5.0, 3.0])
@pytest.mark.parametrize("tokens", [1, 32, 64, 1024])
def test_fused_moe_plain_fp8_swiglu_limit(weights, tokens, limit):
    x, topk_weights, topk_ids = _inputs(tokens)
    out = _fused(weights, x, topk_weights, topk_ids, limit)
    torch.cuda.synchronize()
    assert not out.isnan().any()

    expected = _reference(weights, x, topk_weights, topk_ids, limit)
    logits_diff = _calc_diff(expected, out)
    assert logits_diff < 1e-2, f"logits_diff={logits_diff}"
    if limit:
        unclamped = _reference(weights, x, topk_weights, topk_ids, None)
        assert (
            _calc_diff(unclamped, out) > 5e-2
        ), "output matches the unclamped reference; swiglu_limit was not applied"


@pytest.mark.parametrize("limit", [7.0, 5.0, 3.0])
def test_stage1_clamps_before_silu(weights, limit, monkeypatch):
    """Gate-clamped elements give out/up = silu(L) before SiLU, L after."""
    captured = {}
    stage1_fwd = aiter.ck_moe_stage1_fwd

    def capture(hidden_states, w1, w2, sti, sei, nvi, out, topk, *args, **kwargs):
        ret = stage1_fwd(
            hidden_states, w1, w2, sti, sei, nvi, out, topk, *args, **kwargs
        )
        captured.update(a1=hidden_states.clone(), a1_scale=args[2], out=out.clone())
        captured["limit"] = args[10]
        return ret

    monkeypatch.setattr(aiter, "ck_moe_stage1_fwd", capture)
    x, topk_weights, topk_ids = _inputs(64)
    _fused(weights, x, topk_weights, topk_ids, limit)
    torch.cuda.synchronize()
    assert captured["limit"] == limit

    a = captured["a1"].float() * captured["a1_scale"].float().view(-1, 1)
    w1_scale = weights["w1_scale"].float().view(EXPERTS, -1, 1)
    gate_up = torch.zeros(*topk_ids.shape, 2 * INTER_DIM, device="cuda")
    for e in topk_ids.unique().tolist():
        t, k = (topk_ids == e).nonzero(as_tuple=True)
        gate_up[t, k] = a[t] @ (weights["w1_qt"][e].float() * w1_scale[e]).t()
    g, u = gate_up[..., :INTER_DIM], gate_up[..., INTER_DIM:]
    o = captured["out"].float()
    sel = (g > limit + 1) & (u.abs() > 1) & (u.abs() < limit - 1)
    assert sel.sum() > 1000, "the limit was not exercised"
    implied = float((o[sel] / u[sel]).median())
    silu_limit = float(F.silu(torch.tensor(limit)))
    assert abs(implied - silu_limit) < 0.25 * (limit - silu_limit), (
        f"implied activation {implied}: silu(L)={silu_limit} (before SiLU), "
        f"L={limit} (after SiLU)"
    )


@pytest.mark.parametrize("limit", [None, 0.0, float("inf")])
@pytest.mark.parametrize("tokens", [1, 32, 64, 1024])
def test_fused_moe_plain_fp8_nonfinite_limit_matches_absent(weights, tokens, limit):
    x, topk_weights, topk_ids = _inputs(tokens)
    out_absent = _fused(weights, x, topk_weights, topk_ids, None)
    out = _fused(weights, x, topk_weights, topk_ids, limit)
    torch.cuda.synchronize()
    assert not out.isnan().any()
    # AITER atomics: same-input repeats sit at 0.0024-0.0077 relative L2.
    # A real clamp moves the output by ~0.65. Stay well below that.
    x64, y64 = out.double(), out_absent.double()
    rel = float((x64 - y64).norm() / y64.norm().clamp_min(1e-30))
    assert rel < 2e-2, f"no-clamp limit {limit!r} diverged from absent: rel_l2={rel}"


def _dispatch(token, quant_type, limit, inter_dim=INTER_DIM, w_dtype=dtypes.fp8):
    return get_2stage_cfgs(
        token,
        MODEL_DIM,
        inter_dim,
        EXPERTS,
        TOPK,
        dtypes.bf16,
        w_dtype,
        w_dtype,
        quant_type,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        swiglu_limit=limit,
    )


@pytest.mark.parametrize("limit", [0.0, float("inf")])
@pytest.mark.parametrize("tokens", [32, 1024])
@pytest.mark.parametrize("quant_type", [QuantType.per_Token, QuantType.per_Tensor])
def test_no_clamp_limit_matches_absent_dispatch(quant_type, tokens, limit):
    absent = _dispatch(tokens, quant_type, None)
    other = _dispatch(tokens, quant_type, limit)
    assert bool(other.run_1stage) == bool(absent.run_1stage)
    assert int(other.ksplit or 0) == int(absent.ksplit or 0)
    if not absent.run_1stage:
        assert (
            other.stage1.keywords["kernelName"] == absent.stage1.keywords["kernelName"]
        )


@pytest.mark.parametrize("limit", [7.0, 5.0])
@pytest.mark.parametrize("tokens", [1, 64, 1024])
@pytest.mark.parametrize("quant_type", [QuantType.per_Token, QuantType.per_Tensor])
def test_limit_dispatches_non_split_ck_stage1(quant_type, tokens, limit):
    metadata = _dispatch(tokens, quant_type, limit)
    assert not metadata.run_1stage
    assert int(metadata.ksplit or 0) == 0
    assert metadata.stage1.func is ck_moe_stage1
    kernel_name = metadata.stage1.keywords["kernelName"]
    assert not kernel_name or ("_silu_" in kernel_name and "Splitk" not in kernel_name)


@pytest.mark.parametrize("tokens", [1, 64, 1024])
def test_tagged_per_token_row_is_used(tokens):
    # The shipped per_Token rows are tagged swiglu_limit=7.
    metadata = _dispatch(tokens, QuantType.per_Token, 7.0)
    kernel_name = metadata.stage1.keywords["kernelName"]
    assert "ck2stages" in kernel_name and "_silu_" in kernel_name


@pytest.mark.parametrize("limit", [7.0, 5.0])
@pytest.mark.parametrize("quant_type", [QuantType.per_Token, QuantType.per_Tensor])
def test_limit_without_clamped_kernel_raises(quant_type, limit):
    with pytest.raises(NotImplementedError, match="swiglu_limit"):
        _dispatch(64, quant_type, limit, inter_dim=192)
    if quant_type == QuantType.per_Token:
        with pytest.raises(NotImplementedError, match="swiglu_limit"):
            _dispatch(64, quant_type, limit, w_dtype=dtypes.i8)


@pytest.mark.parametrize("quant_type", [QuantType.per_Token, QuantType.per_Tensor])
def test_asm_stage1_rejects_limit(quant_type):
    with pytest.raises(NotImplementedError, match="swiglu_limit"):
        asm_stage1(
            *([None] * 6),
            torch.empty(1, 1, 1, device="cuda"),
            TOPK,
            32,
            ksplit=0,
            activation=ActivationType.Silu,
            quant_type=quant_type,
            swiglu_limit=7.0,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
