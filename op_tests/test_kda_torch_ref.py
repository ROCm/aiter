# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pins the KDA torch reference, which is the oracle everything else trusts.

An error here would be invisible and would propagate into every test that uses
it. Two independent anchors: the gate against the formula vLLM asserts on, and
the recurrence against today's ``flydsl_gdr_decode``.

That second anchor must use the *scalar* path only. Checking the reference
against the per-channel kernel it exists to judge would close the loop and pin
nothing.
"""

import pytest
import torch

from aiter.ops.flydsl import is_flydsl_available
from aiter.ops.torch_ref.kda import kda_gate, l2norm, naive_recurrent_kda

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm not available."
)

DEVICE = "cuda"


def test_kda_gate_matches_vllm_formula():
    """Compare against vLLM's expected_gate (tests/models/kimi_k3/test_kda.py).

    vLLM flattens to ``(T, H*D)``; this takes ``(B, T, H, D)``. Writing both
    pins the broadcast: A_log is per head and must spread across channels.
    """
    torch.manual_seed(0)
    T, H, D, g_min = 4, 6, 16, -5.0

    a_flat = torch.randn(T, H * D, dtype=torch.float32, device=DEVICE)
    A_log = torch.randn(H, dtype=torch.float32, device=DEVICE) * 0.5
    dt_bias_flat = torch.randn(H * D, dtype=torch.float32, device=DEVICE) * 0.1

    expected = g_min * torch.sigmoid(
        A_log.exp()[None, :, None] * (a_flat.view(T, H, D) + dt_bias_flat.view(H, D))
    )
    got = kda_gate(a_flat.view(1, T, H, D), A_log, dt_bias_flat.view(H, D), g_min=g_min)

    torch.testing.assert_close(got[0], expected)


def test_scalar_gate_is_the_gdr_softplus_form():
    torch.manual_seed(0)
    T, H = 4, 6
    a = torch.randn(1, T, H, dtype=torch.float32, device=DEVICE)
    A_log = torch.randn(H, dtype=torch.float32, device=DEVICE) * 0.5
    dt_bias = torch.rand(H, dtype=torch.float32, device=DEVICE) + 1.0

    expected = -A_log.exp() * torch.nn.functional.softplus(a + dt_bias)

    torch.testing.assert_close(
        kda_gate(a, A_log, dt_bias), expected, atol=1e-6, rtol=1e-5
    )


@pytest.mark.skipif(not is_flydsl_available(), reason="flydsl is not installed")
@pytest.mark.parametrize(
    ("B", "H", "dt"),
    [(1, 8, torch.bfloat16), (4, 8, torch.bfloat16), (2, 32, torch.float16)],
    ids=["b1_h8_bf16", "b4_h8_bf16", "b2_h32_fp16"],
)
def test_reference_reproduces_flydsl_scalar_decode(B, H, dt):
    """The reference must reproduce the shipping kernel on the scalar path.

    Identity shuffle indices, isolating the gate and recurrence from the state
    gather. The tight tolerance is deliberate: agreement is near-exact (worst
    over a 36-run sweep, 4e-6 output / 5e-7 state), while swapping in KDA's
    g_min form moves the output only 2.6e-2 -- a loose bound would pass it.
    """
    from aiter.ops.flydsl import flydsl_gdr_decode

    torch.manual_seed(0)
    T, K, V = 1, 128, 128

    q = torch.randn(B, T, H, K, dtype=dt, device=DEVICE)
    k = torch.randn(B, T, H, K, dtype=dt, device=DEVICE)
    v = torch.randn(B, T, H, V, dtype=dt, device=DEVICE)
    a = torch.randn(B, T, H, dtype=dt, device=DEVICE)
    b = torch.randn(B, T, H, dtype=dt, device=DEVICE)
    dt_bias = torch.empty(H, dtype=dt, device=DEVICE).uniform_(1, 2)
    A_log = torch.empty(H, dtype=torch.float32, device=DEVICE).uniform_(0, 16)
    indices = torch.arange(B, dtype=torch.int32, device=DEVICE)
    state = torch.randn(B, H, K, V, dtype=torch.float32, device=DEVICE)
    out = torch.zeros(B, T, H, V, dtype=dt, device=DEVICE)

    kernel_state = state.clone()
    flydsl_gdr_decode(
        q,
        k,
        v,
        a,
        b,
        dt_bias,
        A_log,
        indices,
        kernel_state,
        out,
        use_qk_l2norm=True,
        need_shuffle_state=True,
    )

    ref_out, ref_state = naive_recurrent_kda(
        l2norm(q),
        l2norm(k),
        v,
        kda_gate(a, A_log, dt_bias),
        b.float().sigmoid(),
        scale=K**-0.5,
        initial_state=state,
        output_final_state=True,
    )

    torch.testing.assert_close(out.float(), ref_out.float(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(kernel_state, ref_state, atol=1e-4, rtol=1e-4)
