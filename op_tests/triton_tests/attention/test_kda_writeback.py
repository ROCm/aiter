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
@pytest.mark.parametrize("steps", [1, 32])
def test_writeback_decode(tokens, padded, steps, monkeypatch):
    torch.manual_seed(42)
    inputs = make_writeback_inputs(tokens, padded)
    expected_state = inputs["initial_state"].clone()
    expected_out = inputs["out"].clone()
    selected = []
    dispatch = kda.try_writeback

    def observe_dispatch(*args, **kwargs):
        result = dispatch(*args, **kwargs)
        selected.append(result is not None)
        return result

    monkeypatch.setattr(kda, "try_writeback", observe_dispatch)
    for step in range(steps):
        if step:
            for name in ("q", "k", "v", "g", "beta"):
                inputs[name].normal_()
        reference_step(inputs, expected_state, expected_out)
        out, final_state = run_writeback(inputs)
        assert final_state is inputs["initial_state"]
        torch.testing.assert_close(out, expected_out, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(final_state, expected_state, atol=1e-5, rtol=1e-4)
    assert len(selected) == steps and all(
        selected
    ), "the public wrapper must select write-back decode"


def test_writeback_empty_and_negative_slots():
    inputs = make_writeback_inputs(8, padded=True)
    inputs["cu_seqlens"][2:] -= 1
    inputs["ssm_state_indices"][3] = -1
    expected_state = inputs["initial_state"].clone()
    expected_out = inputs["out"].clone()
    reference_step(inputs, expected_state, expected_out)
    out, state = run_writeback(inputs)
    torch.testing.assert_close(out, expected_out, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(state, expected_state, atol=1e-5, rtol=1e-4)


def test_writeback_allocated_output():
    inputs = make_writeback_inputs(8)
    expected_out = inputs.pop("out")
    expected_state = inputs["initial_state"].clone()
    reference_step(inputs, expected_state, expected_out)
    out, state = run_writeback(inputs)
    torch.testing.assert_close(out, expected_out, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(state, expected_state, atol=1e-5, rtol=1e-4)


def test_writeback_cached_launch_strides_and_sizes():
    for tokens, padded, index_stride in [
        (1, False, 1),
        (72, True, 2),
        (8, False, 1),
        (72, True, 1),
    ]:
        inputs = make_writeback_inputs(tokens, padded)
        indices = torch.empty(
            (tokens * index_stride,), dtype=torch.int32, device="cuda"
        )
        indices[::index_stride] = inputs["ssm_state_indices"]
        inputs["ssm_state_indices"] = indices[::index_stride]
        expected_state = inputs["initial_state"].clone()
        expected_out = inputs["out"].clone()
        for _ in range(3):
            reference_step(inputs, expected_state, expected_out)
            out, state = run_writeback(inputs)
            torch.testing.assert_close(out, expected_out, atol=2e-3, rtol=2e-2)
            torch.testing.assert_close(state, expected_state, atol=1e-5, rtol=1e-4)


def test_writeback_cached_launch_current_stream():
    inputs = make_writeback_inputs(8, padded=True)
    run_writeback(inputs)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for name in ("q", "k", "v", "g", "beta"):
            inputs[name].normal_()
        expected_state = inputs["initial_state"].clone()
        expected_out = inputs["out"].clone()
        reference_step(inputs, expected_state, expected_out)
        out, state = run_writeback(inputs)
        torch.testing.assert_close(out, expected_out, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(state, expected_state, atol=1e-5, rtol=1e-4)
    stream.synchronize()


def test_writeback_cached_launch_hook():
    from triton import knobs

    inputs = make_writeback_inputs(8)
    run_writeback(inputs)
    calls = []

    def observe(*args, **kwargs):
        calls.append(True)

    knobs.runtime.launch_enter_hook.add(observe)
    try:
        run_writeback(inputs)
        run_writeback(inputs)
        torch.cuda.synchronize()
    finally:
        knobs.runtime.launch_enter_hook.remove(observe)
    assert len(calls) == 2


def test_writeback_large_slot_offset():
    inputs = make_writeback_inputs(1)
    state = torch.empty_strided(
        (2, 12, 128, 128), (2**30, 16384, 128, 1), dtype=torch.float32, device="cuda"
    )
    state[0].normal_()
    state[1].normal_()
    inputs["initial_state"] = state
    inputs["ssm_state_indices"].fill_(1)
    expected_state = state.clone()
    expected_out = inputs["out"].clone()
    reference_step(inputs, expected_state, expected_out)
    out, actual = run_writeback(inputs)
    torch.testing.assert_close(out, expected_out, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(actual, expected_state, atol=1e-5, rtol=1e-4)


def test_writeback_multiple_tokens_reuse_generic_recurrence():
    inputs = make_writeback_inputs(8)
    inputs["cu_seqlens"][1] = 0
    inputs["ssm_state_indices"][2] = -1
    expected_state = inputs["initial_state"].clone()
    expected_out = inputs["out"].clone()
    reference_step(inputs, expected_state, expected_out)
    second = dict(inputs)
    for name in ("q", "k", "v", "g", "beta"):
        second[name] = inputs[name][:, 1:2]
    second["cu_seqlens"] = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    second["ssm_state_indices"] = torch.tensor([1], dtype=torch.int32, device="cuda")
    second_out = torch.empty((1, 1, 12, 128), dtype=torch.bfloat16, device="cuda")
    reference_step(second, expected_state.clone(), second_out)
    expected_out[:, 1:2] = second_out
    out, state = run_writeback(inputs)
    torch.testing.assert_close(out, expected_out, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(state, expected_state, atol=1e-5, rtol=1e-4)


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
        torch.testing.assert_close(state[:8], expected_state[1:9], atol=1e-5, rtol=1e-4)
    else:
        torch.testing.assert_close(state, expected_state, atol=1e-5, rtol=1e-4)
