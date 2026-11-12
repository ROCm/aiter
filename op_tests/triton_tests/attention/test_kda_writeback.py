# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.attention import kda
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(get_arch() != "gfx1250", reason="gfx1250 only")


def make_writeback_inputs(tokens, padded=False):
    """Make BF16 raw gates and FP32 paged state for one-token KDA decode."""
    h, d = 12, 128
    values = {
        name: torch.randn((1, tokens, h, d), dtype=torch.bfloat16, device="cuda")
        for name in ("q", "k", "v", "g")
    }
    beta_storage = torch.randn(
        (1, tokens, 6284 if padded else h), dtype=torch.bfloat16, device="cuda"
    )
    values["beta"] = beta_storage[..., :h]
    state_storage = torch.randn(
        (tokens + 2, h * d * d + (d if padded else 0)),
        dtype=torch.float32,
        device="cuda",
    )
    values.update(
        A_log=torch.randn((h,), dtype=torch.float32, device="cuda") * 0.1,
        dt_bias=torch.randn((h, d), dtype=torch.float32, device="cuda") * 0.1,
        initial_state=state_storage[:, : h * d * d].view(tokens + 2, h, d, d),
        cu_seqlens=torch.arange(tokens + 1, dtype=torch.int32, device="cuda"),
        ssm_state_indices=torch.arange(tokens, dtype=torch.int32, device="cuda"),
        out=torch.full((1, tokens, h, d), 7.0, dtype=torch.bfloat16, device="cuda"),
    )
    return values


def run_writeback(inputs, **kwargs):
    return kda.fused_recurrent_kda(
        **inputs,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=-5.0,
        **kwargs,
    )


def reference_step(inputs, state, out):
    starts = inputs["cu_seqlens"][:-1]
    slots = inputs["ssm_state_indices"]
    valid = (slots >= 0) & (inputs["cu_seqlens"][1:] > starts)
    tokens, slots = starts[valid].long(), slots[valid].long()
    q, k, v, g = (inputs[name][0, tokens].float() for name in ("q", "k", "v", "g"))
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * (128**-0.5)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    decay = (
        -5.0 * torch.sigmoid(inputs["A_log"].exp()[:, None] * (g + inputs["dt_bias"]))
    ).exp()
    s = state[slots] * decay[:, :, None, :]
    beta = inputs["beta"][0, tokens].float().sigmoid()
    err = (v - (s * k[:, :, None, :]).sum(-1)) * beta[:, :, None]
    s = s + err[:, :, :, None] * k[:, :, None, :]
    state[slots] = s
    out[0, tokens] = (s * q[:, :, None, :]).sum(-1).to(out.dtype)


@pytest.mark.parametrize("tokens", [1, 8, 72, 256])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("steps", [1, 16])
def test_writeback_decode(tokens, padded, steps, monkeypatch):
    torch.manual_seed(42)
    inputs = make_writeback_inputs(tokens, padded)
    expected_state = inputs["initial_state"].clone()
    expected_out = inputs["out"].clone()
    selected = []
    get_config = kda.get_kda_config

    def observe_config(*args, **kwargs):
        selected.append(kwargs.get("writeback_decode", False))
        return get_config(*args, **kwargs)

    monkeypatch.setattr(kda, "get_kda_config", observe_config)
    for step in range(steps):
        if step:
            for name in ("q", "k", "v", "g", "beta"):
                inputs[name].normal_()
        reference_step(inputs, expected_state, expected_out)
        out, final_state = run_writeback(inputs)
        assert final_state is inputs["initial_state"]
        torch.testing.assert_close(out, expected_out, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(final_state, expected_state, atol=2e-5, rtol=3e-3)
    assert all(selected), "the public wrapper must select write-back decode"


def test_writeback_empty_and_negative_slots():
    inputs = make_writeback_inputs(8, padded=True)
    inputs["cu_seqlens"][2:] -= 1
    inputs["ssm_state_indices"][3] = -1
    expected_state = inputs["initial_state"].clone()
    expected_out = inputs["out"].clone()
    reference_step(inputs, expected_state, expected_out)
    out, state = run_writeback(inputs)
    torch.testing.assert_close(out, expected_out, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(state, expected_state, atol=2e-5, rtol=3e-3)


def test_writeback_allocated_output():
    inputs = make_writeback_inputs(8)
    expected_out = inputs.pop("out")
    expected_state = inputs["initial_state"].clone()
    reference_step(inputs, expected_state, expected_out)
    out, state = run_writeback(inputs)
    torch.testing.assert_close(out, expected_out, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(state, expected_state, atol=2e-5, rtol=3e-3)


@pytest.mark.parametrize(
    "fallback", ["int64", "strided_q", "strided_g", "out_of_place"]
)
def test_writeback_fallback(fallback, monkeypatch):
    inputs = make_writeback_inputs(8)
    kwargs = {}
    if fallback == "int64":
        for name in ("cu_seqlens", "ssm_state_indices"):
            inputs[name] = inputs[name].to(torch.int64)
    elif fallback in ("strided_q", "strided_g"):
        name = fallback[-1]
        storage = torch.empty((1, 8, 24, 128), dtype=torch.bfloat16, device="cuda")
        view = storage[:, :, :12, :]
        view.copy_(inputs[name])
        inputs[name] = view
    else:
        kwargs = {
            "inplace_final_state": False,
            "state_out": torch.empty_like(inputs["initial_state"]),
        }
        inputs["ssm_state_indices"] += 1
    expected_state = inputs["initial_state"].clone()
    expected_out = inputs["out"].clone()
    reference_step(inputs, expected_state, expected_out)
    selected = []
    get_config = kda.get_kda_config

    def observe_config(*args, **kwargs):
        selected.append(kwargs.get("writeback_decode", False))
        return get_config(*args, **kwargs)

    monkeypatch.setattr(kda, "get_kda_config", observe_config)
    out, state = run_writeback(inputs, **kwargs)
    assert selected == [False]
    torch.testing.assert_close(out, expected_out, atol=2e-3, rtol=2e-2)
    if fallback == "out_of_place":
        torch.testing.assert_close(state[:8], expected_state[1:9], atol=2e-5, rtol=3e-3)
    else:
        torch.testing.assert_close(state, expected_state, atol=2e-5, rtol=3e-3)
