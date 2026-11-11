# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.attention.kda import fused_recurrent_kda
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(
    get_arch() not in ("gfx1250", "gfx1250-strict"), reason="gfx1250 decode only"
)


def make_paged_decode_inputs(num_seqs, strided=False, negative_slot=True):
    shape = (1, num_seqs, 12, 128)
    q, k, v, g = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)
    )
    beta = torch.randn((1, num_seqs, 6284), device="cuda", dtype=torch.bfloat16)
    beta = beta[..., 6144:6156]
    A_log = torch.empty((12,), device="cuda", dtype=torch.float32).uniform_(1, 16).log()
    dt_bias = torch.randn((12, 128), device="cuda", dtype=torch.float32) * 0.5
    pool_shape = (2 * num_seqs + 3, 12, 128, 128)
    state = torch.empty_strided(
        pool_shape,
        (196608 + (128 if strided else 0), 16384, 128, 1),
        device="cuda",
        dtype=torch.float32,
    ).normal_(std=0.1)
    slots = torch.randperm(pool_shape[0] - 1, device="cuda")[:num_seqs] + 1
    slots[0] = 0
    if negative_slot and num_seqs > 1:
        slots[num_seqs // 2] = -1
    if strided:
        backing = torch.empty((2 * num_seqs,), device="cuda", dtype=slots.dtype)
        backing[::2] = slots
        slots = backing[::2]
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "initial_state": state,
        "cu_seqlens": torch.arange(num_seqs + 1, device="cuda"),
        "ssm_state_indices": slots,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "use_qk_l2norm_in_kernel": True,
        "state_v_first": True,
        "lower_bound": -5.0,
        "skip_negative_state_slots": True,
        "out": torch.full_like(v, 7.0),
    }


@pytest.mark.parametrize("num_seqs", [1, 8, 72, 256])
@pytest.mark.parametrize("strided", [False, True])
def test_paged_decode_negative_slots(num_seqs, strided):
    torch.manual_seed(19)
    inputs = make_paged_decode_inputs(num_seqs, strided)
    state = inputs["initial_state"]
    initial = state.clone()
    expected_state = initial.clone()
    slots = inputs["ssm_state_indices"]
    valid = slots >= 0
    live_slots = slots[valid]
    untouched = torch.ones(state.shape[0], device="cuda", dtype=torch.bool)
    untouched[live_slots] = False
    q, k, v, g = (inputs[name][0, valid].float() for name in ("q", "k", "v", "g"))
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * 128**-0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    decay = (
        -5.0 * torch.sigmoid(inputs["A_log"].exp()[:, None] * (g + inputs["dt_bias"]))
    ).exp()
    beta = torch.sigmoid(inputs["beta"][0, valid].float())
    for _ in range(32 if num_seqs == 72 else 1):
        decayed = expected_state[live_slots] * decay[:, :, None, :]
        update = (v - (decayed * k[:, :, None, :]).sum(-1)) * beta[:, :, None]
        updated = decayed + update[..., None] * k[:, :, None, :]
        expected_output = (updated * q[:, :, None, :]).sum(-1).to(v.dtype)
        expected_state[live_slots] = updated
        output, final_state = fused_recurrent_kda(**inputs)
        assert final_state is state and output is inputs["out"]
        torch.testing.assert_close(
            output[0, valid].float(), expected_output, atol=2e-3, rtol=2e-2
        )
        torch.testing.assert_close(state, expected_state, atol=1e-5, rtol=1e-4)
        assert torch.equal(state[untouched], initial[untouched])
        assert torch.equal(output[0, ~valid], torch.full_like(output[0, ~valid], 7.0))


def test_negative_slot_guard_rejects_conflicting_padding():
    inputs = make_paged_decode_inputs(8)
    with pytest.raises(AssertionError):
        fused_recurrent_kda(**inputs, pad_slot_guard=True)


def test_paged_decode_all_negative_slots():
    inputs = make_paged_decode_inputs(8)
    inputs["ssm_state_indices"].fill_(-1)
    state = inputs["initial_state"].clone()
    output, final_state = fused_recurrent_kda(**inputs)
    assert torch.equal(final_state, state)
    assert torch.equal(output, torch.full_like(output, 7.0))
