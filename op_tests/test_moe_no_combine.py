# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for `fused_moe(..., no_combine=True)`.

`no_combine=True` returns per-route unweighted output of shape
`(M, topk, model_dim)` instead of the default combined `(M, model_dim)`.
Downstream consumers (e.g. GPT-OSS-style pipelines) apply their own topk
weighting and mixing after the kernel.

Scope (MI350X / gfx950):
- CK 2-stage backend only.
- bf16 inputs / weights / output, QuantType.No only.
- Other configurations raise NotImplementedError.

Hardware-running tests require the patched CK header that adds the
`bool NoCombine = false` template parameter to `DeviceMoeGemm` and gate
the stage2 epilogue on `MulRoutedWeight && !NoCombine`. Build aiter with
`AITER_CK_HAS_NO_COMBINE_TEMPLATE=1` to enable. Without the patch, the
hardware tests skip via `_NEEDS_CK_NO_COMBINE`.
"""

import os

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import (
    MOEMetadata,
    asm_stage1,
    cktile_moe_stage1,
    cktile_moe_stage2,
    fused_moe,
    fused_moe_fake,
    torch_moe_stage1,
    _flydsl_stage1_wrapper,
    _flydsl_stage2_wrapper,
)
from aiter.ops.shuffle import shuffle_weight

torch.set_default_device("cuda")


_NEEDS_CK_NO_COMBINE = pytest.mark.skipif(
    not torch.cuda.is_available()
    or os.environ.get("AITER_CK_HAS_NO_COMBINE_TEMPLATE") != "1",
    reason=(
        "CK NoCombine template not available. Build aiter with "
        "AITER_CK_HAS_NO_COMBINE_TEMPLATE=1 against a CK that carries the "
        "DeviceMoeGemm NoCombine patch."
    ),
)


# ---------------------------------------------------------------------------
# Reference and fixtures
# ---------------------------------------------------------------------------


def _torch_reference_route_outputs(hidden, w1, w2, topk_ids, topk_w):
    """Per-route unweighted reference matching `fused_moe(no_combine=True)`.

    Pipeline: `torch_moe_stage1(doweight=False)` then an inline per-route
    stage2 GEMM that keeps the topk axis intact. `torch_moe_stage2` collapses
    the topk axis via `.sum(1)`, which is why we re-implement its inner GEMM
    here. Returns dtype bf16 to match the kernel output.
    """
    stage1_out = torch_moe_stage1(
        hidden, w1, w2, topk_w, topk_ids,
        dtype=torch.bfloat16,
        activation=ActivationType.Silu,
        quant_type=QuantType.No,
        doweight=False,
    )
    token_num, topk = topk_ids.shape
    model_dim = w2.shape[1]
    stage1_fp32 = stage1_out.to(torch.float32)
    w2_fp32 = w2.to(torch.float32)
    out = torch.zeros(
        (token_num, topk, model_dim),
        dtype=torch.float32,
        device=stage1_out.device,
    )
    for e in range(w2.shape[0]):
        mask = topk_ids == e
        if not mask.any():
            continue
        sub = stage1_fp32[mask]
        out[mask] = sub @ w2_fp32[e].transpose(0, 1)
    return out.to(torch.bfloat16)


def _make_bf16_case(
    M, topk, model_dim, inter_dim, expert_num,
    *, seed=0, topk_id_override=None,
):
    """Deterministic case builder.

    `topk_ids` defaults to a per-token round-robin over experts so different
    tokens land on different expert mixes without invoking the optional
    fused_topk ASM kernel.
    """
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    hidden_states = torch.randn((M, model_dim), dtype=dtype)
    w1 = torch.randn((expert_num, inter_dim * 2, model_dim), dtype=dtype) * 0.02
    w2 = torch.randn((expert_num, model_dim, inter_dim), dtype=dtype) * 0.02
    if topk_id_override is not None:
        topk_ids = topk_id_override.to(torch.int32)
    else:
        base = torch.arange(M, dtype=torch.int32).unsqueeze(1)
        offset = torch.arange(topk, dtype=torch.int32).unsqueeze(0)
        topk_ids = ((base + offset) % expert_num).contiguous()
    topk_weights = torch.ones((M, topk), dtype=torch.float32) / topk
    return hidden_states, w1, w2, topk_ids, topk_weights


def _shuffled(*tensors):
    """CK 2-stage expects pre-shuffled weights (layout=(16, 16)).

    Mirrors the convention used by `op_tests/test_moe_2stage.py`.
    """
    return tuple(shuffle_weight(t, layout=(16, 16)) for t in tensors)


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------


def test_default_returns_combined_shape():
    """Regression guard: default call (no no_combine kwarg) keeps (M, D)."""
    M, topk, model_dim, inter_dim, expert_num = 64, 2, 128, 256, 8
    hidden, w1, w2, topk_ids, topk_w = _make_bf16_case(
        M, topk, model_dim, inter_dim, expert_num
    )
    out = fused_moe(
        hidden_states=hidden, w1=w1, w2=w2,
        topk_weight=topk_w, topk_ids=topk_ids,
        activation=ActivationType.Silu, quant_type=QuantType.No,
    )
    assert out.shape == (M, model_dim)
    assert out.shape != (M, topk, model_dim)


@_NEEDS_CK_NO_COMBINE
@pytest.mark.parametrize(
    "M,topk,model_dim,inter_dim",
    [
        (64, 2, 128, 256),
        (64, 4, 512, 512),
        (64, 8, 2048, 1024),
    ],
)
def test_no_combine_returns_per_route_shape(M, topk, model_dim, inter_dim):
    """no_combine=True returns contiguous (M, topk, model_dim)."""
    expert_num = 16
    hidden, w1, w2, topk_ids, topk_w = _make_bf16_case(
        M, topk, model_dim, inter_dim, expert_num
    )
    w1, w2 = _shuffled(w1, w2)
    out = fused_moe(
        hidden_states=hidden, w1=w1, w2=w2,
        topk_weight=topk_w, topk_ids=topk_ids,
        activation=ActivationType.Silu, quant_type=QuantType.No,
        no_combine=True,
    )
    assert out.shape == (M, topk, model_dim)
    assert out.shape != (M, model_dim)
    assert out.is_contiguous()
    assert out.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


@_NEEDS_CK_NO_COMBINE
@pytest.mark.parametrize("topk", [2, 4, 8])
@pytest.mark.parametrize("model_dim", [128, 512, 2048])
def test_no_combine_matches_torch_reference(topk, model_dim):
    """fused_moe(no_combine=True) matches the per-route torch reference."""
    M, inter_dim, expert_num = 64, 1024, 16
    hidden, w1_orig, w2_orig, topk_ids, topk_w = _make_bf16_case(
        M, topk, model_dim, inter_dim, expert_num
    )
    w1, w2 = _shuffled(w1_orig, w2_orig)
    out = fused_moe(
        hidden_states=hidden, w1=w1, w2=w2,
        topk_weight=topk_w, topk_ids=topk_ids,
        activation=ActivationType.Silu, quant_type=QuantType.No,
        no_combine=True,
    )
    ref = _torch_reference_route_outputs(hidden, w1_orig, w2_orig, topk_ids, topk_w)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@_NEEDS_CK_NO_COMBINE
@pytest.mark.parametrize("topk", [2, 4, 8])
@pytest.mark.parametrize("model_dim", [128, 512, 2048])
def test_no_combine_sums_to_combined(topk, model_dim):
    """Weighted sum of per-route output equals the combined kernel output."""
    M, inter_dim, expert_num = 64, 1024, 16
    hidden, w1, w2, topk_ids, topk_w = _make_bf16_case(
        M, topk, model_dim, inter_dim, expert_num
    )
    w1, w2 = _shuffled(w1, w2)
    out_nc = fused_moe(
        hidden_states=hidden, w1=w1, w2=w2,
        topk_weight=topk_w, topk_ids=topk_ids,
        activation=ActivationType.Silu, quant_type=QuantType.No,
        no_combine=True,
    )
    out_c = fused_moe(
        hidden_states=hidden, w1=w1, w2=w2,
        topk_weight=topk_w, topk_ids=topk_ids,
        activation=ActivationType.Silu, quant_type=QuantType.No,
        no_combine=False,
    )
    manual = (out_nc.float() * topk_w.unsqueeze(-1)).sum(dim=1).to(out_c.dtype)
    is_close = torch.isclose(manual, out_c, rtol=1e-2, atol=5e-2)
    # Up to 5% of elements may fall outside the band — Python re-cast to bf16
    # introduces an extra rounding step relative to the in-kernel atomic_add.
    assert (~is_close).float().mean().item() <= 0.05


@_NEEDS_CK_NO_COMBINE
def test_no_combine_high_numbered_experts():
    """Exercises the full expert id range (experts {5, 7} of 8)."""
    M, topk, model_dim, inter_dim, expert_num = 16, 2, 128, 256, 8
    topk_ids = torch.tensor([[5, 7]] * M, dtype=torch.int32)
    hidden, w1_orig, w2_orig, _, topk_w = _make_bf16_case(
        M, topk, model_dim, inter_dim, expert_num,
        topk_id_override=topk_ids,
    )
    w1, w2 = _shuffled(w1_orig, w2_orig)
    out = fused_moe(
        hidden_states=hidden, w1=w1, w2=w2,
        topk_weight=topk_w, topk_ids=topk_ids,
        activation=ActivationType.Silu, quant_type=QuantType.No,
        no_combine=True,
    )
    ref = _torch_reference_route_outputs(hidden, w1_orig, w2_orig, topk_ids, topk_w)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Rejection guards
# ---------------------------------------------------------------------------


def test_no_combine_rejects_non_bf16_output_dtype():
    M, topk, model_dim, inter_dim, expert_num = 16, 2, 128, 256, 8
    hidden, w1, w2, topk_ids, topk_w = _make_bf16_case(
        M, topk, model_dim, inter_dim, expert_num
    )
    with pytest.raises(NotImplementedError, match=r"bf16 output dtype"):
        fused_moe(
            hidden_states=hidden, w1=w1, w2=w2,
            topk_weight=topk_w, topk_ids=topk_ids,
            activation=ActivationType.Silu, quant_type=QuantType.No,
            dtype=torch.float16,
            no_combine=True,
        )


def test_no_combine_rejects_doweight_stage1():
    M, topk, model_dim, inter_dim, expert_num = 16, 2, 128, 256, 8
    hidden, w1, w2, topk_ids, topk_w = _make_bf16_case(
        M, topk, model_dim, inter_dim, expert_num
    )
    with pytest.raises(NotImplementedError, match=r"doweight_stage1"):
        fused_moe(
            hidden_states=hidden, w1=w1, w2=w2,
            topk_weight=topk_w, topk_ids=topk_ids,
            activation=ActivationType.Silu, quant_type=QuantType.No,
            doweight_stage1=True, no_combine=True,
        )


@pytest.mark.parametrize(
    "qt",
    [QuantType.per_Tensor, QuantType.per_Token, QuantType.per_1x128, QuantType.per_1x32],
)
def test_no_combine_rejects_non_no_quant(qt):
    M, topk, model_dim, inter_dim, expert_num = 16, 2, 128, 256, 8
    hidden, w1, w2, topk_ids, topk_w = _make_bf16_case(
        M, topk, model_dim, inter_dim, expert_num
    )
    with pytest.raises(NotImplementedError, match=r"quant_type"):
        fused_moe(
            hidden_states=hidden, w1=w1, w2=w2,
            topk_weight=topk_w, topk_ids=topk_ids,
            activation=ActivationType.Silu, quant_type=qt, no_combine=True,
        )


def _identity_stage1(*args, **kwargs):
    raise AssertionError("stage1 must not be called when guard fires")


def _identity_stage2(*args, **kwargs):
    raise AssertionError("stage2 must not be called when guard fires")


def _patch_metadata(monkeypatch, stage1_func, stage2_func, *, run_1stage=False):
    """Inject a fake MOEMetadata so the no_combine guards see the requested
    stage1/stage2 funcs without triggering the real heuristic dispatch.
    """
    def fake(*args, **kwargs):
        return MOEMetadata(
            stage1=stage1_func, stage2=stage2_func,
            block_m=32, ksplit=1, run_1stage=run_1stage,
        )
    import aiter.fused_moe as fm_mod
    monkeypatch.setattr(fm_mod, "get_2stage_cfgs", fake)


def test_no_combine_rejects_1stage_dispatch(monkeypatch):
    _patch_metadata(monkeypatch, _identity_stage1, _identity_stage2, run_1stage=True)
    M, topk, model_dim, inter_dim, expert_num = 16, 2, 128, 256, 8
    hidden, w1, w2, topk_ids, topk_w = _make_bf16_case(
        M, topk, model_dim, inter_dim, expert_num
    )
    with pytest.raises(NotImplementedError, match=r"1-stage dispatch"):
        fused_moe(
            hidden_states=hidden, w1=w1, w2=w2,
            topk_weight=topk_w, topk_ids=topk_ids,
            activation=ActivationType.Silu, quant_type=QuantType.No,
            no_combine=True,
        )


@pytest.mark.parametrize(
    "stage1_func",
    [_flydsl_stage1_wrapper, cktile_moe_stage1, asm_stage1],
    ids=["flydsl", "cktile", "asm"],
)
def test_no_combine_rejects_non_ck_stage1(monkeypatch, stage1_func):
    _patch_metadata(monkeypatch, stage1_func, aiter.ck_moe_stage2_fwd)
    M, topk, model_dim, inter_dim, expert_num = 16, 2, 128, 256, 8
    hidden, w1, w2, topk_ids, topk_w = _make_bf16_case(
        M, topk, model_dim, inter_dim, expert_num
    )
    with pytest.raises(NotImplementedError, match=r"CK 2-stage stage1"):
        fused_moe(
            hidden_states=hidden, w1=w1, w2=w2,
            topk_weight=topk_w, topk_ids=topk_ids,
            activation=ActivationType.Silu, quant_type=QuantType.No,
            no_combine=True,
        )


@pytest.mark.parametrize(
    "stage2_func",
    [_flydsl_stage2_wrapper, cktile_moe_stage2],
    ids=["flydsl", "cktile"],
)
def test_no_combine_rejects_non_ck_stage2(monkeypatch, stage2_func):
    def ck_stage1_stub(*args, **kwargs):
        raise AssertionError("stage1 must not run when stage2 guard fires")
    _patch_metadata(monkeypatch, ck_stage1_stub, stage2_func)
    M, topk, model_dim, inter_dim, expert_num = 16, 2, 128, 256, 8
    hidden, w1, w2, topk_ids, topk_w = _make_bf16_case(
        M, topk, model_dim, inter_dim, expert_num
    )
    with pytest.raises(NotImplementedError, match=r"CK 2-stage stage2"):
        fused_moe(
            hidden_states=hidden, w1=w1, w2=w2,
            topk_weight=topk_w, topk_ids=topk_ids,
            activation=ActivationType.Silu, quant_type=QuantType.No,
            no_combine=True,
        )


# ---------------------------------------------------------------------------
# EP semantics: pre-zero'd buffer guarantees deterministic zeros where the
# kernel does not write.
# ---------------------------------------------------------------------------


@_NEEDS_CK_NO_COMBINE
def test_no_combine_expert_mask_zero_fills_non_local_routes():
    """Routes to non-local experts (expert_mask=0) stay exactly zero."""
    M, topk, model_dim, inter_dim, expert_num = 32, 2, 128, 256, 8
    topk_ids = torch.zeros((M, topk), dtype=torch.int32)
    topk_ids[:, 0] = 0   # local
    topk_ids[:, 1] = 7   # non-local
    hidden, w1, w2, _, topk_w = _make_bf16_case(
        M, topk, model_dim, inter_dim, expert_num,
        topk_id_override=topk_ids,
    )
    w1, w2 = _shuffled(w1, w2)
    expert_mask = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0], dtype=torch.int32)
    out = fused_moe(
        hidden_states=hidden, w1=w1, w2=w2,
        topk_weight=topk_w, topk_ids=topk_ids,
        activation=ActivationType.Silu, quant_type=QuantType.No,
        expert_mask=expert_mask, no_combine=True,
    )
    assert torch.all(out[:, 1, :] == 0)
    assert torch.any(out[:, 0, :] != 0)


@_NEEDS_CK_NO_COMBINE
def test_no_combine_num_local_tokens_zero_fills_tail():
    """Tokens [k:M] are exactly zero when num_local_tokens=k.

    Enforced in Python (after stage2) by aiter; the CK kernel itself may emit
    block-padding leakage past the boundary.
    """
    M, topk, model_dim, inter_dim, expert_num = 32, 2, 128, 256, 8
    k_local = 16
    hidden, w1, w2, topk_ids, topk_w = _make_bf16_case(
        M, topk, model_dim, inter_dim, expert_num
    )
    w1, w2 = _shuffled(w1, w2)
    num_local_tokens = torch.tensor([k_local], dtype=torch.int32)
    out = fused_moe(
        hidden_states=hidden, w1=w1, w2=w2,
        topk_weight=topk_w, topk_ids=topk_ids,
        activation=ActivationType.Silu, quant_type=QuantType.No,
        num_local_tokens=num_local_tokens, no_combine=True,
    )
    assert torch.all(out[k_local:, :, :] == 0)
    assert torch.any(out[:k_local, :, :] != 0)


@_NEEDS_CK_NO_COMBINE
def test_no_combine_all_experts_masked_produces_zero_output():
    """All-zero expert_mask produces an all-zero output, no NaNs."""
    M, topk, model_dim, inter_dim, expert_num = 16, 2, 128, 256, 8
    hidden, w1, w2, topk_ids, topk_w = _make_bf16_case(
        M, topk, model_dim, inter_dim, expert_num
    )
    w1, w2 = _shuffled(w1, w2)
    expert_mask = torch.zeros(expert_num, dtype=torch.int32)
    out = fused_moe(
        hidden_states=hidden, w1=w1, w2=w2,
        topk_weight=topk_w, topk_ids=topk_ids,
        activation=ActivationType.Silu, quant_type=QuantType.No,
        expert_mask=expert_mask, no_combine=True,
    )
    assert torch.all(out == 0)
    assert not torch.any(torch.isnan(out))


# ---------------------------------------------------------------------------
# torch.compile / fake-tensor path
# ---------------------------------------------------------------------------


def test_no_combine_fake_returns_per_route_shape():
    """fused_moe_fake(..., no_combine=True) returns (M, topk, model_dim)."""
    M, topk, model_dim, inter_dim, expert_num = 16, 2, 128, 256, 8
    hidden = torch.empty((M, model_dim), dtype=torch.bfloat16)
    w1 = torch.empty((expert_num, inter_dim * 2, model_dim), dtype=torch.bfloat16)
    w2 = torch.empty((expert_num, model_dim, inter_dim), dtype=torch.bfloat16)
    topk_ids = torch.zeros((M, topk), dtype=torch.int32)
    topk_w = torch.ones((M, topk), dtype=torch.float32) / topk
    out = fused_moe_fake(
        hidden_states=hidden, w1=w1, w2=w2,
        topk_weight=topk_w, topk_ids=topk_ids,
        activation=ActivationType.Silu, quant_type=QuantType.No,
        no_combine=True,
    )
    assert out.shape == (M, topk, model_dim)
    assert out.dtype == torch.bfloat16


@_NEEDS_CK_NO_COMBINE
def test_no_combine_torch_compile_traces_fused_moe():
    """torch.compile(fused_moe)(..., no_combine=True) traces and runs."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available in this torch build")
    M, topk, model_dim, inter_dim, expert_num = 16, 2, 128, 256, 8
    hidden, w1, w2, topk_ids, topk_w = _make_bf16_case(
        M, topk, model_dim, inter_dim, expert_num
    )
    w1, w2 = _shuffled(w1, w2)

    def call(h, _w1, _w2, ids, w):
        return fused_moe(
            hidden_states=h, w1=_w1, w2=_w2,
            topk_weight=w, topk_ids=ids,
            activation=ActivationType.Silu, quant_type=QuantType.No,
            no_combine=True,
        )

    compiled = torch.compile(call, fullgraph=False, dynamic=False)
    out = compiled(hidden, w1, w2, topk_ids, topk_w)
    assert out.shape == (M, topk, model_dim)
    assert out.dtype == torch.bfloat16
