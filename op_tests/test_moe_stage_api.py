# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""fused_moe_prepare_stage2 / fused_moe_run_stage2 against fused_moe on MXFP4
experts."""

import dataclasses
import inspect

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    fused_moe,
    fused_moe_prepare_stage2,
    fused_moe_run_stage2,
    fused_topk,
    get_2stage_cfgs,
    get_padded_M,
    supports_stage2_overrides,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="MXFP4 MoE requires gfx950",
)

# One TP8 rank of a 256-expert top-8 model, hidden 4096. The repeated-input
# call (topk 1) has tuned FlyDSL rows for this shape.
E, TOPK, MODEL_DIM, INTER_DIM = 256, 8, 4096, 512


def _quant(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[E, rows, cols] bf16 -> shuffled MXFP4 weight and E8M0 scale."""
    quant = aiter.get_torch_quant(QuantType.per_1x32)
    qt, scale = quant(w, quant_dtype=dtypes.fp4x2)
    qt = shuffle_weight(qt.view(w.shape[0], w.shape[1], -1), layout=(16, 16))
    return qt, fp4_utils.e8m0_shuffle(scale)


@pytest.fixture(scope="module")
def experts():
    torch.manual_seed(0)
    w1 = torch.randn(E, 2 * INTER_DIM, MODEL_DIM, dtype=dtypes.bf16, device="cuda")
    w2 = torch.randn(E, MODEL_DIM, INTER_DIM, dtype=dtypes.bf16, device="cuda")
    w1 = w1 * MODEL_DIM**-0.5
    w2 = w2 * INTER_DIM**-0.5
    half = MODEL_DIM // 2
    return {
        "w1": _quant(w1),
        "w2": _quant(w2),
        "w2_rows": (
            _quant(w2[:, :half].contiguous()),
            _quant(w2[:, half:].contiguous()),
        ),
    }


def _routes(token: int):
    torch.manual_seed(token)
    x = torch.randn(token, MODEL_DIM, dtype=dtypes.bf16, device="cuda")
    score = torch.randn(token, E, dtype=dtypes.bf16, device="cuda")
    topk_weights, topk_ids = fused_topk(x, score, TOPK, True)
    return x, topk_weights, topk_ids


def _kwargs(experts) -> dict:
    (w1, w1_scale), (w2, w2_scale) = experts["w1"], experts["w2"]
    return {
        "w1": w1,
        "w2": w2,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "activation": ActivationType.Silu,
        "quant_type": QuantType.per_1x32,
    }


def _repeated_input_config(token: int):
    """The config of the equivalent call on the input repeated TOPK times."""
    return get_2stage_cfgs(
        get_padded_M(token * TOPK),
        MODEL_DIM,
        INTER_DIM,
        E,
        1,
        dtypes.bf16,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
        opus_weights_shuffled=True,
        input_dtype=dtypes.bf16,
    )


def _check(ref: torch.Tensor, out: torch.Tensor, exact: bool = False):
    if exact:
        assert torch.equal(ref, out)
    else:
        torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("token", [256, 2048])
def test_staged_matches_fused_moe(experts, token):
    x, topk_weights, topk_ids = _routes(token)
    kwargs = _kwargs(experts)
    try:
        inputs = fused_moe_prepare_stage2(
            x, topk_weight=topk_weights, topk_ids=topk_ids, **kwargs
        )
    except RuntimeError:
        pytest.skip("selected config has no separate stage2")
    staged = fused_moe_run_stage2(inputs)
    _check(
        fused_moe(x, topk_weight=topk_weights, topk_ids=topk_ids, **kwargs),
        staged,
    )
    # The sorting's buffer holds the result: a second summed run needs `out`.
    with pytest.raises(RuntimeError):
        fused_moe_run_stage2(inputs)
    _check(staged, fused_moe_run_stage2(inputs, out=torch.empty_like(staged)))


def test_prepare_takes_fused_moe_arguments():
    """fused_moe_prepare_stage2 has fused_moe's parameters (order and
    defaults) up to gate_mode."""
    prepare = list(inspect.signature(fused_moe_prepare_stage2).parameters.values())
    fused = list(inspect.signature(fused_moe).parameters.values())
    n = [p.name for p in fused].index("gate_mode") + 1
    assert [(p.name, p.default) for p in prepare[:n]] == [
        (p.name, p.default) for p in fused[:n]
    ]
    for name in ("quant_type_a", "quant_dtype_a", "quant_dtype_a2", "deterministic"):
        assert inspect.signature(fused_moe_prepare_stage2).parameters[name].default == (
            inspect.signature(fused_moe).parameters[name].default
        )


@pytest.mark.parametrize("token", [256, 2048])
def test_per_slot_matches_repeated_input(experts, token):
    if not supports_stage2_overrides(_repeated_input_config(token)):
        pytest.skip("no FlyDSL MXFP4 stage2 for this shape")
    x, _, topk_ids = _routes(token)
    kwargs = _kwargs(experts)
    ones = torch.ones(token, TOPK, dtype=dtypes.fp32, device="cuda")
    inputs = fused_moe_prepare_stage2(
        x,
        topk_weight=ones,
        topk_ids=topk_ids,
        config_shape=(token * TOPK, 1),
        **kwargs,
    )
    per_slot = fused_moe_run_stage2(inputs, per_slot=True)
    assert per_slot.shape == (token, TOPK, MODEL_DIM)
    # Each route of the repeated input is a top-1 route of weight one.
    ref = fused_moe(
        x.repeat_interleave(TOPK, dim=0),
        topk_weight=ones.view(-1, 1),
        topk_ids=topk_ids.view(-1, 1),
        **kwargs,
    )
    _check(ref.view(token, TOPK, MODEL_DIM), per_slot)


@pytest.mark.parametrize("token", [256, 2048])
def test_per_slot_sums_to_default(experts, token):
    """On the default top-8 config (v1 FlyDSL stage2 on gfx950), the
    per-slot rows sum to the default output."""
    x, topk_weights, topk_ids = _routes(token)
    inputs = fused_moe_prepare_stage2(
        x, topk_weight=topk_weights, topk_ids=topk_ids, **_kwargs(experts)
    )
    try:
        per_slot = fused_moe_run_stage2(inputs, per_slot=True)
    except NotImplementedError:
        pytest.skip("selected stage2 has no per-slot output")
    assert per_slot.shape == (token, TOPK, MODEL_DIM)
    _check(fused_moe_run_stage2(inputs), per_slot.float().sum(dim=1))


@pytest.mark.parametrize("per_slot", [False, True])
def test_replaced_w2_rows(experts, per_slot):
    token = 2048
    if not supports_stage2_overrides(_repeated_input_config(token)):
        pytest.skip("no FlyDSL MXFP4 stage2 for this shape")
    x, topk_weights, topk_ids = _routes(token)
    kwargs = _kwargs(experts)

    def stage2_on(w2=None, w2_scale=None):
        inputs = fused_moe_prepare_stage2(
            x,
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            config_shape=(token * TOPK, 1),
            **kwargs,
        )
        return fused_moe_run_stage2(inputs, per_slot=per_slot, w2=w2, w2_scale=w2_scale)

    full = stage2_on()
    # vLLM passes the shuffled scale viewed as [E, rows, groups].
    halves = [
        stage2_on(w2, scale.view(E, w2.shape[1], -1))
        for w2, scale in experts["w2_rows"]
    ]
    _check(full, torch.cat(halves, dim=-1), exact=per_slot)


def test_replaced_a2(experts):
    token = 2048
    if not supports_stage2_overrides(_repeated_input_config(token)):
        pytest.skip("no FlyDSL MXFP4 stage2 for this shape")
    x, topk_weights, topk_ids = _routes(token)
    inputs = fused_moe_prepare_stage2(
        x,
        topk_weight=topk_weights,
        topk_ids=topk_ids,
        config_shape=(token * TOPK, 1),
        **_kwargs(experts),
    )
    zeros = torch.zeros_like(inputs.a2.view(torch.uint8)).view(inputs.a2.dtype)
    out = fused_moe_run_stage2(inputs, per_slot=True, a2=zeros)
    assert not out.any()


def test_rejected_operand_mismatch(experts):
    """Mismatched replacing operands raise ValueError before any kernel
    runs."""
    token = 2048
    if not supports_stage2_overrides(_repeated_input_config(token)):
        pytest.skip("no FlyDSL MXFP4 stage2 for this shape")
    x, topk_weights, topk_ids = _routes(token)
    inputs = fused_moe_prepare_stage2(
        x,
        topk_weight=topk_weights,
        topk_ids=topk_ids,
        config_shape=(token * TOPK, 1),
        **_kwargs(experts),
    )
    a2, a2_scale = inputs.a2, inputs.a2_scale
    (w2, w2_scale), _ = experts["w2_rows"]
    a2_u8 = a2.view(torch.uint8)
    wide_a2 = torch.cat((a2_u8, a2_u8), dim=-1).view(a2.dtype)
    w2_u8 = w2.view(torch.uint8)
    wide_w2 = torch.cat((w2_u8, w2_u8), dim=-1).view(w2.dtype)
    cases = {
        "a2 device": {"a2": a2.cpu()},
        "a2 rows": {"a2": a2[:-1]},
        "a2 dtype": {"a2": a2.view(torch.int8)},
        "a2 width without a2_scale": {"a2": wide_a2},
        "a2_scale device": {"a2_scale": a2_scale.cpu()},
        "a2_scale rows": {"a2_scale": a2_scale[:-32]},
        "a2_scale dtype": {"a2_scale": a2_scale.view(torch.int16)},
        "w2 experts": {"w2": w2[:-1], "w2_scale": w2_scale},
        "w2 without w2_scale": {"w2": w2},
        "w2 and a2 widths": {"w2": w2[..., :-16].contiguous(), "w2_scale": w2_scale},
        "w2 not contiguous": {
            "w2": experts["w2"][0][:, : MODEL_DIM // 2],
            "w2_scale": w2_scale,
        },
        "w2_scale rows": {"w2": w2, "w2_scale": w2_scale[:256]},
        "w2_scale groups": {"w2": w2, "w2_scale": w2_scale[:, :8].contiguous()},
        "a2_scale groups": {
            "a2": wide_a2,
            "a2_scale": a2_scale,
            "w2": wide_w2,
            "w2_scale": torch.cat((w2_scale, w2_scale), dim=-1),
        },
        "w2_scale alone of another shape": {"w2_scale": w2_scale},
        "out overlapping a2": {"out": a2},
    }
    for name, overrides in cases.items():
        with pytest.raises(ValueError):
            fused_moe_run_stage2(inputs, per_slot=True, **overrides)
            pytest.fail(f"{name}: accepted")


def test_rejected_unsupported_capture(experts):
    """Like supports_stage2_overrides, a captured stage2 after split-K
    (no a2_scale), with bias2 or after the aux sort refuses the overrides."""
    token = 2048
    if not supports_stage2_overrides(_repeated_input_config(token)):
        pytest.skip("no FlyDSL MXFP4 stage2 for this shape")
    x, topk_weights, topk_ids = _routes(token)
    inputs = fused_moe_prepare_stage2(
        x,
        topk_weight=topk_weights,
        topk_ids=topk_ids,
        config_shape=(token * TOPK, 1),
        **_kwargs(experts),
    )
    marker = torch.zeros(1, device="cuda")
    cases = {
        "split-K": dataclasses.replace(inputs, a2_scale=None),
        "bias2": dataclasses.replace(
            inputs, _kwargs={**inputs._kwargs, "bias2": marker}
        ),
        "aux sort": dataclasses.replace(
            inputs, _kwargs={**inputs._kwargs, "reverse_sorted": marker}
        ),
    }
    for name, captured in cases.items():
        with pytest.raises(NotImplementedError):
            fused_moe_run_stage2(captured, per_slot=True)
            pytest.fail(f"{name}: accepted")


def _cat_scale_columns(scales: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate shuffled E8M0 scales [rows, groups] along groups: each
    32-row band stores its 32 x 8 tiles contiguously in group order."""
    rows = scales[0].shape[0]
    bands = [s.view(torch.uint8).reshape(rows // 32, -1) for s in scales]
    return torch.cat(bands, dim=1).view(rows, -1).view(scales[0].dtype)


@pytest.mark.parametrize("token", [256, 2048])
def test_split_intermediate_gather(token):
    """Two ranks each hold half of the intermediate dim. Stage2 on the
    gathered a2 and half of the w2 rows equals the sum of both ranks'
    per-slot outputs on those rows."""
    if not supports_stage2_overrides(_repeated_input_config(token)):
        pytest.skip("no FlyDSL MXFP4 stage2 for this shape")
    torch.manual_seed(1)
    inter_full, half = 2 * INTER_DIM, MODEL_DIM // 2
    w1 = torch.randn(E, 2 * inter_full, MODEL_DIM, dtype=dtypes.bf16, device="cuda")
    w2 = torch.randn(E, MODEL_DIM, inter_full, dtype=dtypes.bf16, device="cuda")
    w1, w2 = w1 * MODEL_DIM**-0.5, w2 * inter_full**-0.5
    x, _, topk_ids = _routes(token)
    ones = torch.ones(token, TOPK, dtype=dtypes.fp32, device="cuda")
    ranks = []
    for r in range(2):
        cols = slice(r * INTER_DIM, (r + 1) * INTER_DIM)
        w1_r = torch.cat((w1[:, cols], w1[:, inter_full:][:, cols]), dim=1)
        (w1q, w1s), (w2q, w2s) = _quant(w1_r.contiguous()), _quant(
            w2[..., cols].contiguous()
        )
        ranks.append(
            fused_moe_prepare_stage2(
                x,
                w1q,
                w2q,
                ones,
                topk_ids,
                activation=ActivationType.Silu,
                quant_type=QuantType.per_1x32,
                w1_scale=w1s,
                w2_scale=w2s,
                config_shape=(token * TOPK, 1),
            )
        )
    ref = sum(fused_moe_run_stage2(inputs, per_slot=True).float() for inputs in ranks)
    a2 = torch.cat([inputs.a2.view(torch.uint8) for inputs in ranks], dim=-1)
    a2_scale = _cat_scale_columns([inputs.a2_scale for inputs in ranks])
    w2_rows, w2_rows_scale = _quant(w2[:, :half].contiguous())
    out = fused_moe_run_stage2(
        ranks[0],
        per_slot=True,
        w2=w2_rows,
        w2_scale=w2_rows_scale,
        a2=a2.view(ranks[0].a2.dtype),
        a2_scale=a2_scale,
    )
    assert out.shape == (token, TOPK, half)
    ref = ref[..., :half]
    err = ((out.float() - ref).norm() / ref.norm()).item()
    assert err < 1e-2, f"relative error {err:.3g}"


@pytest.mark.parametrize("token", [32, 512])
def test_bf16_default_and_rejected_overrides(token):
    """On a bf16 (non-FlyDSL) config the default stage2 equals fused_moe and
    the overrides are refused."""
    torch.manual_seed(token)
    e, model_dim, inter_dim, topk = 32, 1024, 512, 4
    w1 = torch.randn(e, 2 * inter_dim, model_dim, dtype=dtypes.bf16, device="cuda")
    w2 = torch.randn(e, model_dim, inter_dim, dtype=dtypes.bf16, device="cuda")
    w1, w2 = w1 * model_dim**-0.5, w2 * inter_dim**-0.5
    x = torch.randn(token, model_dim, dtype=dtypes.bf16, device="cuda")
    score = torch.randn(token, e, dtype=dtypes.bf16, device="cuda")
    topk_weights, topk_ids = fused_topk(x, score, topk, True)
    try:
        inputs = fused_moe_prepare_stage2(x, w1, w2, topk_weights, topk_ids)
    except RuntimeError:
        pytest.skip("selected config has no separate stage2")
    with pytest.raises(NotImplementedError):
        fused_moe_run_stage2(inputs, per_slot=True)
    _check(fused_moe(x, w1, w2, topk_weights, topk_ids), fused_moe_run_stage2(inputs))
