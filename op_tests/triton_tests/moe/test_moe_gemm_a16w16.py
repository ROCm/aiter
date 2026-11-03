# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe, torch_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import checkAllclose

pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx1201",
    reason="gfx1201 selected large-M BF16 G1U1 Triton MoE coverage",
)

_RTOL = 1e-2
_ATOL = 128
_RANGE_TOKENS = [
    1024,
    1088,
    1151,
    1152,
    1153,
    1542,
    1543,
    1544,
    1727,
    1728,
    1729,
    1799,
    1803,
    1815,
    1837,
    1944,
    1945,
    2047,
    2048,
    2049,
    2304,
    2560,
]
_OUT_OF_RANGE_TOKENS = [1023, 2561]


def _make_case(tokens: int):
    torch.manual_seed(2026 + tokens)
    model_dim = 2048
    inter_dim = 128
    expert = 256
    topk = 8
    hidden = torch.randn((tokens, model_dim), dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn(
        (expert, 2 * inter_dim, model_dim), dtype=dtypes.bf16, device="cuda"
    )
    w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtypes.bf16, device="cuda")
    ids = torch.randint(0, expert, (tokens, topk), dtype=torch.int32, device="cuda")
    weights = torch.rand((tokens, topk), dtype=torch.float32, device="cuda")
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return hidden, w1.contiguous(), w2.contiguous(), weights.contiguous(), ids.contiguous()


def _large_m_gate_inputs(tokens: int):
    fused_moe_mod = importlib.import_module("aiter.fused_moe")
    hidden, w1, w2, weights, ids = _make_case(tokens)
    return {
        "hidden_states": hidden,
        "w1": w1,
        "w2": w2,
        "topk_weight": weights,
        "topk_ids": ids,
        "activation": aiter.ActivationType.Silu,
        "quant_type": aiter.QuantType.No,
        "dtype": dtypes.bf16,
        "q_dtype_a": dtypes.bf16,
        "q_dtype_w": dtypes.bf16,
        "is_g1u1": True,
        "doweight_stage1": False,
        "w1_scale": None,
        "w2_scale": None,
        "a1_scale": None,
        "a2_scale": None,
        "num_local_tokens": None,
        "moe_sorting_dispatch_policy": 0,
        "expert_mask": None,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "bias1": None,
        "bias2": None,
        "swiglu_limit": None,
        "beta": None,
        "linear_beta": None,
        "gate_mode": fused_moe_mod.GateMode.SEPARATED.value,
        "stage2_scatter": None,
    }


def _large_m_gate(**kwargs):
    fused_moe_mod = importlib.import_module("aiter.fused_moe")
    return fused_moe_mod._can_use_bf16_g1u1_gfx1201_triton_large_m(**kwargs)


@pytest.fixture(autouse=True)
def _force_validated_gfx1201_target(monkeypatch):
    fused_moe_mod = importlib.import_module("aiter.fused_moe")
    monkeypatch.setattr(fused_moe_mod, "get_gfx_runtime", lambda: "gfx1201")
    monkeypatch.setattr(fused_moe_mod, "get_cu_num", lambda: 64)


@pytest.mark.parametrize("tokens", [1024, 1152, 1544, 1728, 1944, 2048, 2560])
def test_public_fused_moe_bf16_g1u1_selected_large_m(tokens):
    hidden, w1, w2, weights, ids = _make_case(tokens)

    actual = fused_moe(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
        dtype=dtypes.bf16,
    )
    expected = torch_moe(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation=aiter.ActivationType.Silu,
    )
    assert checkAllclose(actual, expected, rtol=_RTOL, atol=_ATOL) == 0


def test_public_fused_moe_bf16_g1u1_selected_large_m_output_buffer():
    hidden, w1, w2, weights, ids = _make_case(1152)
    expected = torch_moe(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation=aiter.ActivationType.Silu,
    )
    output = torch.empty_like(expected)
    actual = fused_moe(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
        dtype=dtypes.bf16,
        output=output,
    )
    assert actual is output
    assert checkAllclose(output, expected, rtol=_RTOL, atol=_ATOL) == 0


@pytest.mark.parametrize("tokens", _RANGE_TOKENS)
def test_large_m_gate_accepts_validated_range_rows(tokens):
    assert _large_m_gate(**_large_m_gate_inputs(tokens))


@pytest.mark.parametrize("tokens", _OUT_OF_RANGE_TOKENS)
def test_large_m_gate_rejects_out_of_range_rows(tokens):
    assert not _large_m_gate(**_large_m_gate_inputs(tokens))


def test_large_m_gate_rejects_unsupported_modes(monkeypatch):
    fused_moe_mod = importlib.import_module("aiter.fused_moe")
    case = _large_m_gate_inputs(1152)

    monkeypatch.setattr(fused_moe_mod, "get_gfx_runtime", lambda: "gfx950")
    assert not _large_m_gate(**case)
    monkeypatch.setattr(fused_moe_mod, "get_gfx_runtime", lambda: "gfx1201")

    monkeypatch.setattr(fused_moe_mod, "get_cu_num", lambda: 60)
    assert not _large_m_gate(**case)
    monkeypatch.setattr(fused_moe_mod, "get_cu_num", lambda: 64)

    assert not _large_m_gate(**{**case, "quant_type": aiter.QuantType.per_Token})
    assert not _large_m_gate(**{**case, "is_g1u1": False})
    assert not _large_m_gate(**{**case, "expert_mask": torch.ones(256, device="cuda")})
    assert not _large_m_gate(**{**case, "stage2_scatter": object()})

    shuffled_w1 = case["w1"].clone()
    shuffled_w1.is_shuffled = True
    assert not _large_m_gate(**{**case, "w1": shuffled_w1})

    noncontig_w1 = case["w1"].transpose(1, 2)
    assert not _large_m_gate(**{**case, "w1": noncontig_w1})
