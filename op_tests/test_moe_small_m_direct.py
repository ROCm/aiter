# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import (
    _can_use_bf16_g1u1_small_m_direct,
    fused_moe,
    torch_moe,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose

# The direct kernel materializes the Silu(gate) * up workspace in bf16, while
# torch_moe keeps the reference intermediate in fp32. On gfx1201/R9700 across
# the 36-case matrix below, max abs diff was 256.0 and max required atol after
# rtol=1e-2 was 62.42, so 64 is the tight power-of-two margin.
_SMALL_M_DIRECT_RTOL = 1e-2
_SMALL_M_DIRECT_ATOL = 64

pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx1201",
    reason="gfx1201 small-M direct MoE coverage",
)


def _make_balanced_topk(token: int, topk: int, expert: int):
    rows = torch.arange(token, device="cuda", dtype=torch.int32)[:, None]
    slots = torch.arange(topk, device="cuda", dtype=torch.int32)[None, :]
    topk_ids = (rows * topk + slots) % expert
    topk_weights = torch.rand((token, topk), dtype=torch.float32, device="cuda")
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    return topk_ids.contiguous(), topk_weights.contiguous()


def _make_case(
    token: int = 4,
    inter_dim: int = 128,
    model_dim: int = 2048,
    expert: int = 256,
    topk: int = 8,
):
    torch.manual_seed(2026 + token + inter_dim + model_dim + expert + topk)
    hidden_states = torch.randn((token, model_dim), dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn(
        (expert, inter_dim * 2, model_dim), dtype=dtypes.bf16, device="cuda"
    )
    w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtypes.bf16, device="cuda")
    topk_ids, topk_weights = _make_balanced_topk(token, topk, expert)
    return hidden_states, w1, w2, topk_weights, topk_ids


def _make_gate_kwargs(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
):
    return dict(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weight=topk_weights,
        topk_ids=topk_ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
        dtype=dtypes.bf16,
        doweight_stage1=False,
        w1_scale=None,
        w2_scale=None,
        a1_scale=None,
        a2_scale=None,
        num_local_tokens=None,
        moe_sorting_dispatch_policy=0,
        expert_mask=None,
        hidden_pad=0,
        intermediate_pad=0,
        bias1=None,
        bias2=None,
        gate_mode=GateMode.SEPARATED.value,
    )


def _case_as_gate_kwargs(case):
    return _make_gate_kwargs(*case)


@pytest.mark.parametrize("token", [1, 2, 3, 4, 8, 16])
@pytest.mark.parametrize("inter_dim", [128, 256])
@pytest.mark.parametrize(
    "preshuffle_w1,preshuffle_w2",
    [(False, False), (True, False), (True, True)],
)
def test_bf16_g1u1_small_m_direct(token, inter_dim, preshuffle_w1, preshuffle_w2):
    hidden_states, w1, w2, topk_weights, topk_ids = _make_case(token, inter_dim)
    w1_input = shuffle_weight(w1, layout=(16, 16)) if preshuffle_w1 else w1
    w2_input = shuffle_weight(w2, layout=(16, 16)) if preshuffle_w2 else w2

    actual = fused_moe(
        hidden_states,
        w1_input,
        w2_input,
        topk_weights,
        topk_ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
        dtype=dtypes.bf16,
        gate_mode=GateMode.SEPARATED.value,
    )
    expected = torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=aiter.ActivationType.Silu,
    )

    assert (
        checkAllclose(
            actual,
            expected,
            rtol=_SMALL_M_DIRECT_RTOL,
            atol=_SMALL_M_DIRECT_ATOL,
        )
        == 0
    )


def test_bf16_g1u1_small_m_direct_output_buffer():
    hidden_states, w1, w2, topk_weights, topk_ids = _make_case()
    expected = torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=aiter.ActivationType.Silu,
    )
    output = torch.empty_like(expected)
    output.fill_(float("nan"))

    actual = fused_moe(
        hidden_states,
        shuffle_weight(w1, layout=(16, 16)),
        shuffle_weight(w2, layout=(16, 16)),
        topk_weights,
        topk_ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
        dtype=dtypes.bf16,
        gate_mode=GateMode.SEPARATED.value,
        output=output,
    )

    assert actual is output
    assert (
        checkAllclose(
            output,
            expected,
            rtol=_SMALL_M_DIRECT_RTOL,
            atol=_SMALL_M_DIRECT_ATOL,
        )
        == 0
    )


def test_bf16_g1u1_small_m_direct_gate_rejects_unsupported_integration():
    hidden_states, w1, w2, topk_weights, topk_ids = _make_case()
    gate_kwargs = _make_gate_kwargs(hidden_states, w1, w2, topk_weights, topk_ids)

    assert _can_use_bf16_g1u1_small_m_direct(**gate_kwargs)
    assert not _can_use_bf16_g1u1_small_m_direct(
        **gate_kwargs, stage2_scatter=object()
    )
    assert not _can_use_bf16_g1u1_small_m_direct(**gate_kwargs, splitk=2)

    w1_bad_layout = shuffle_weight(w1, layout=(32, 16))
    assert not _can_use_bf16_g1u1_small_m_direct(
        **{**gate_kwargs, "w1": w1_bad_layout}
    )

    w1_guinterleave = shuffle_weight(
        w1, layout=(16, 16), is_guinterleave=True, gate_up=True
    )
    assert not _can_use_bf16_g1u1_small_m_direct(
        **{**gate_kwargs, "w1": w1_guinterleave}
    )


def test_bf16_g1u1_small_m_direct_gate_shape_contract():
    hidden_states, w1, w2, topk_weights, topk_ids = _make_case()
    gate_kwargs = _make_gate_kwargs(hidden_states, w1, w2, topk_weights, topk_ids)
    assert _can_use_bf16_g1u1_small_m_direct(**gate_kwargs)

    for case in (
        _make_case(token=32),
        _make_case(inter_dim=64),
        _make_case(model_dim=1024),
        _make_case(expert=128),
        _make_case(topk=4),
    ):
        assert not _can_use_bf16_g1u1_small_m_direct(**_case_as_gate_kwargs(case))


def test_bf16_g1u1_small_m_direct_plain_shuffle_metadata_contract():
    hidden_states, w1, w2, topk_weights, topk_ids = _make_case()
    w1_shuffled = shuffle_weight(w1, layout=(16, 16))
    assert getattr(w1_shuffled, "aiter_shuffle_layout", None) == (16, 16)
    assert getattr(w1_shuffled, "aiter_shuffle_is_guinterleave", None) is False
    assert not hasattr(w1_shuffled, "gate_up_interleave")

    assert _can_use_bf16_g1u1_small_m_direct(
        **_make_gate_kwargs(hidden_states, w1_shuffled, w2, topk_weights, topk_ids)
    )
