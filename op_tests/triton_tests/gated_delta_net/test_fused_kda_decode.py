# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for fused KDA decode kernel (conv1d + recurrence + gated RMSNorm)."""

import pytest
import torch
from einops import rearrange

from aiter.ops.triton.gated_delta_net.causal_conv1d_decode import (
    causal_conv1d_update_split_qkv,
)
from aiter.ops.triton.gated_delta_net.fused_kda_decode import fused_kda_decode
from op_tests.triton_tests.gated_delta_net.test_fused_kda_decode_unified import (
    _ref_decode as _shared_ref_decode,
)

device = "cuda"
ATOL = 0.05


def _ref_kda_decode(
    mixed_qkv,
    conv_state,
    conv_weight,
    gate,
    beta,
    out_gate,
    A_log,
    dt_bias,
    ssm_state,
    ssm_state_indices,
    cu_seqlens,
    norm_weight,
    norm_eps,
    head_dim,
    num_local_heads,
    lower_bound,
):
    """Reference: 3 separate aiter kernel calls."""
    T = mixed_qkv.shape[0]
    lp = num_local_heads * head_dim
    D = head_dim
    H = num_local_heads

    # Kernel 1: conv1d
    q, k, v = causal_conv1d_update_split_qkv(
        mixed_qkv,
        conv_state,
        conv_weight,
        lp,
        lp,
        bias=None,
        activation="silu",
        conv_state_indices=ssm_state_indices,
        use_gluon=False,
    )

    # Kernel 2: recurrence (aiter uses softplus gating, not KDA lower-bound)
    # aiter's API: fused_sigmoid_gating_delta_rule_update(A_log, a, dt_bias,
    #   softplus_beta, softplus_threshold, q, k, v, b, state, indices, ...)
    # This uses softplus gating, not KDA's lower-bounded sigmoid.
    # For a proper reference we need the KDA variant.
    # Use a simple PyTorch reference instead.

    q = rearrange(q, "t (h d) -> 1 t h d", d=D)
    k = rearrange(k, "t (h d) -> 1 t h d", d=D)
    v = rearrange(v, "t (h d) -> 1 t h d", d=D)

    # QK L2 norm
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6) * (D**-0.5)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

    # Per-token recurrence (PyTorch reference)
    out = torch.empty(T, H, D, dtype=torch.float32, device=device)
    for t_idx in range(T):
        slot = ssm_state_indices[t_idx].item()
        if slot < 0:
            out[t_idx] = 0
            continue
        for h in range(H):
            qt = q[0, t_idx, h]  # [D]
            kt = k[0, t_idx, h]  # [D]
            vt = v[0, t_idx, h]  # [D]
            state = ssm_state[slot, h].float()  # [D, D]

            # Gate
            a_val = gate[0, t_idx, h].float()  # [D]
            dt_val = dt_bias[h * D : (h + 1) * D].float()
            A_val = A_log[h].float()
            g = lower_bound * torch.sigmoid(torch.exp(A_val) * (a_val + dt_val))

            # Beta
            beta_val = torch.sigmoid(beta[0, t_idx, h].float())

            # Decay
            state = state * torch.exp(g[None, :])

            # Delta rule
            dot = (state * kt[None, :]).sum(dim=1)
            delta_v = (vt - dot) * beta_val
            state = state + delta_v[:, None] * kt[None, :]

            # Output
            o_val = (state * qt[None, :]).sum(dim=1)
            out[t_idx, h] = o_val
            ssm_state[slot, h] = state.to(ssm_state.dtype)

    # Round to bf16 (match kernel behavior)
    out_bf16 = out.to(torch.bfloat16).float()

    # RMSNorm + gate
    sumsq = (out_bf16**2).sum(dim=-1, keepdim=True)
    rstd = torch.rsqrt(sumsq / D + norm_eps)
    out_gate_3d = rearrange(out_gate[:T], "t (h d) -> t h d", d=D).float()
    normed = (
        out_bf16
        * rstd
        * norm_weight.float()[None, None, :]
        * torch.sigmoid(out_gate_3d)
    )

    return rearrange(normed.to(torch.bfloat16), "t h d -> t (h d)")


def _make_inputs(batch, Hloc, D, W=4, dtype=torch.bfloat16):
    lp = Hloc * D
    num_slots = batch + 2
    return {
        "mixed_qkv": torch.randn(batch, 3 * lp, dtype=dtype, device=device),
        "conv_weight": torch.randn(3 * lp, W, dtype=dtype, device=device) * 0.1,
        "conv_state": torch.randn(num_slots, 3 * lp, W - 1, dtype=dtype, device=device)
        * 0.1,
        "gate": torch.randn(1, batch, Hloc, D, dtype=dtype, device=device) * 0.5,
        "beta": torch.randn(1, batch, Hloc, dtype=dtype, device=device),
        "out_gate": torch.randn(batch, lp, dtype=dtype, device=device),
        "A_log": torch.randn(Hloc, dtype=dtype, device=device) * 0.1,
        "dt_bias": torch.randn(lp, dtype=dtype, device=device) * 0.1,
        "ssm_state": torch.randn(
            num_slots, Hloc, D, D, dtype=torch.float32, device=device
        )
        * 0.01,
        "norm_weight": torch.ones(D, dtype=dtype, device=device),
        # Zero-based slots: normal decode treats slot 0 as a real cache slot.
        "ssm_state_indices": torch.arange(batch, dtype=torch.int32, device=device),
        "cu_seqlens": torch.arange(batch + 1, dtype=torch.int64, device=device),
    }


@pytest.mark.parametrize(
    "batch,Hloc,D,strided_indices",
    [
        *((batch, Hloc, 128, False) for batch in (1, 4, 32, 64) for Hloc in (2, 8)),
        (4, 2, 128, True),
    ],
)
def test_fused_kda_decode_correctness(batch, Hloc, D, strided_indices):
    """Fused kernel output matches PyTorch reference."""
    torch.manual_seed(42)
    inp = _make_inputs(batch, Hloc, D)
    if strided_indices:
        base = torch.zeros(batch * 2, dtype=torch.int32, device=device)
        base[::2] = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
        inp["ssm_state_indices"] = base[::2]
        assert inp["ssm_state_indices"].stride(0) == 2

    ref_cs, ref_ss = inp["conv_state"].clone(), inp["ssm_state"].clone()
    ref = _ref_kda_decode(
        inp["mixed_qkv"],
        ref_cs,
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        ref_ss,
        inp["ssm_state_indices"],
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        D,
        Hloc,
        -5.0,
    )

    fused_cs, fused_ss = inp["conv_state"].clone(), inp["ssm_state"].clone()
    out = fused_kda_decode(
        inp["mixed_qkv"],
        fused_cs,
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        fused_ss,
        inp["ssm_state_indices"],
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        D,
        Hloc,
        -5.0,
    )

    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=0.02)
    torch.testing.assert_close(fused_cs, ref_cs, atol=0, rtol=0)
    torch.testing.assert_close(fused_ss, ref_ss, atol=ATOL, rtol=0.02)


def test_fused_kda_decode_determinism():
    """Same input produces identical output across multiple runs."""
    torch.manual_seed(42)
    batch, Hloc, D = 128, 8, 128
    inp = _make_inputs(batch, Hloc, D)

    results = []
    for _ in range(5):
        conv_state = inp["conv_state"].clone()
        ssm_state = inp["ssm_state"].clone()
        out = fused_kda_decode(
            inp["mixed_qkv"],
            conv_state,
            inp["conv_weight"],
            inp["gate"],
            inp["beta"],
            inp["out_gate"],
            inp["A_log"],
            inp["dt_bias"],
            ssm_state,
            inp["ssm_state_indices"],
            inp["cu_seqlens"],
            inp["norm_weight"],
            1e-6,
            D,
            Hloc,
            -5.0,
        )
        results.append((out.clone(), conv_state, ssm_state))

    for i in range(1, len(results)):
        for name, expected, actual in zip(
            ("output", "conv_state", "ssm_state"), results[0], results[i]
        ):
            assert torch.equal(expected, actual), (
                f"{name}, run 0 vs run {i}: max diff = "
                f"{(expected.float() - actual.float()).abs().max().item()}"
            )


def test_fused_kda_decode_pad_slot():
    """PAD_SLOT_ID (-1) sequences are skipped without modifying state."""
    torch.manual_seed(42)
    batch, Hloc, D = 1, 8, 128
    inp = _make_inputs(batch, Hloc, D)

    ssm_before = inp["ssm_state"].clone()
    conv_before = inp["conv_state"].clone()
    inp["ssm_state_indices"] = torch.tensor([-1], dtype=torch.int32, device=device)

    fused_kda_decode(
        inp["mixed_qkv"],
        inp["conv_state"],
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        inp["ssm_state"],
        inp["ssm_state_indices"],
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        D,
        Hloc,
        -5.0,
    )

    assert torch.equal(
        inp["ssm_state"], ssm_before
    ), "SSM state modified for PAD_SLOT_ID"
    assert torch.equal(
        inp["conv_state"], conv_before
    ), "Conv state modified for PAD_SLOT_ID"


@pytest.mark.parametrize(
    "invalid_out,match",
    [
        ("noncontiguous", "contiguous"),
        ("dtype", "torch.bfloat16"),
        ("device", "device"),
    ],
)
def test_fused_kda_decode_rejects_invalid_out(invalid_out, match):
    batch, Hloc, D = 2, 2, 128
    inp = _make_inputs(batch, Hloc, D)
    if invalid_out == "noncontiguous":
        out = torch.zeros(Hloc * D, batch, dtype=torch.bfloat16, device=device).t()
    elif invalid_out == "dtype":
        out = torch.zeros(batch, Hloc * D, dtype=torch.float32, device=device)
    else:
        out = torch.zeros(batch, Hloc * D, dtype=torch.bfloat16, device="cpu")
    with pytest.raises(ValueError, match=match):
        fused_kda_decode(
            inp["mixed_qkv"],
            inp["conv_state"],
            inp["conv_weight"],
            inp["gate"],
            inp["beta"],
            inp["out_gate"],
            inp["A_log"],
            inp["dt_bias"],
            inp["ssm_state"],
            inp["ssm_state_indices"],
            inp["cu_seqlens"],
            inp["norm_weight"],
            1e-6,
            D,
            Hloc,
            -5.0,
            out=out,
        )


SPEC_D = 128
SPEC_W = 4


class _SpecKernelSpy:
    """Record grids the parallel spec kernel is launched with.

    Numerical agreement alone cannot tell the two paths apart, since the generic
    fallback computes the same result. Without this the suite passes while the
    optimized kernel never runs.
    """

    def __init__(self, module):
        self._module = module
        self._name = "fused_kda_spec_parallel_v_kernel"
        self.grids = []

    def __enter__(self):
        self._orig = getattr(self._module, self._name)
        grids = self.grids
        orig = self._orig

        class _Launcher:
            def __getitem__(self, grid):
                grids.append(grid)
                return orig[grid]

        setattr(self._module, self._name, _Launcher())
        return self

    def __exit__(self, *exc):
        setattr(self._module, self._name, self._orig)
        return False


def _ref_spec_decode(*args, **kwargs):
    return _shared_ref_decode(
        *args,
        lower_bound=-5.0,
        full_spec_sequence=True,
        **kwargs,
    )


def _make_spec_inputs(
    batch,
    Hloc,
    num_spec,
    seq_lens=None,
):
    lp = Hloc * SPEC_D
    if seq_lens is not None:
        seq_lens = tuple(seq_lens)
        if len(seq_lens) != batch:
            raise ValueError(f"seq_lens has {len(seq_lens)} entries for batch {batch}")
        total_tokens = sum(seq_lens)
        cu = [0]
        for length in seq_lens:
            cu.append(cu[-1] + length)
        cu_seqlens = torch.tensor(cu, dtype=torch.int64, device=device)
    else:
        seq_len = 1 + num_spec
        total_tokens = batch * seq_len
        cu_seqlens = torch.arange(
            0, total_tokens + 1, seq_len, dtype=torch.int64, device=device
        )
    state_len = SPEC_W - 1 + num_spec
    num_slots = batch + num_spec * batch + 4
    torch.manual_seed(42)
    dtype = torch.bfloat16
    inp = {
        "mixed_qkv": torch.randn(total_tokens, 3 * lp, dtype=dtype, device=device)
        * 0.1,
        "conv_weight": torch.randn(3 * lp, SPEC_W, dtype=dtype, device=device) * 0.1,
        "conv_state": torch.randn(
            num_slots, 3 * lp, state_len, dtype=dtype, device=device
        )
        * 0.1,
        "gate": torch.randn(1, total_tokens, Hloc, SPEC_D, dtype=dtype, device=device)
        * 0.5,
        "beta": torch.randn(1, total_tokens, Hloc, dtype=dtype, device=device),
        "out_gate": torch.randn(total_tokens, lp, dtype=dtype, device=device),
        "A_log": torch.randn(Hloc, dtype=dtype, device=device) * 0.1,
        "dt_bias": torch.randn(lp, dtype=dtype, device=device) * 0.1,
        "state": torch.randn(
            num_slots, Hloc, SPEC_D, SPEC_D, dtype=torch.float32, device=device
        )
        * 0.01,
        "norm_weight": torch.ones(SPEC_D, dtype=dtype, device=device),
        "cu_seqlens": cu_seqlens,
    }
    inp["state_indices"] = torch.arange(
        1, batch * (1 + num_spec) + 1, dtype=torch.int32, device=device
    ).reshape(batch, 1 + num_spec)
    inp["num_accepted_tokens"] = torch.ones(batch, dtype=torch.int32, device=device)
    inp["conv_state_indices"] = torch.arange(
        1, batch + 1, dtype=torch.int32, device=device
    )
    return inp


@pytest.mark.parametrize(
    "weight_layout,batch,Hloc,num_spec",
    [
        ("channels_width", 1, 2, 7),
        ("group_width_channels", 4, 12, 7),
        ("group_width_channels", 1, 12, 3),
    ],
)
def test_optimized_fused_spec_decode(weight_layout, batch, Hloc, num_spec):
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if get_arch() != "gfx950":
        pytest.skip("parallel spec kernel is only dispatched on gfx950")

    spec_tokens = num_spec + 1
    inp = _make_spec_inputs(batch, Hloc, num_spec=num_spec)
    inp["num_accepted_tokens"] = torch.tensor(
        [1 + (i * 3) % spec_tokens for i in range(batch)],
        dtype=torch.int32,
        device=device,
    )
    fused_weight = inp["conv_weight"]
    if weight_layout == "group_width_channels":
        fused_weight = fused_weight.reshape(3, Hloc * SPEC_D, SPEC_W).transpose(1, 2)
        fused_weight = fused_weight.contiguous()

    _assert_spec_matches(inp, expect_parallel=True, fused_weight=fused_weight)


def test_optimized_fused_spec_decode_uses_real_cu_seqlens_with_padded_tokens():
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if get_arch() != "gfx950":
        pytest.skip("parallel spec-7 kernel is only dispatched on gfx950")

    batch, Hloc, num_spec = 1, 2, 7
    inp = _make_spec_inputs(batch, Hloc, num_spec=num_spec)
    inp["num_accepted_tokens"] = torch.tensor([4], dtype=torch.int32, device=device)
    ref_cs, ref_ss = inp["conv_state"].clone(), inp["state"].clone()
    ref = _ref_spec_decode(
        inp["mixed_qkv"],
        ref_cs,
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        ref_ss,
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        Hloc,
        state_indices=inp["state_indices"],
        num_accepted_tokens=inp["num_accepted_tokens"],
        conv_state_indices=inp["conv_state_indices"],
    )

    pad = 8
    mixed_qkv = torch.cat(
        [inp["mixed_qkv"], torch.zeros_like(inp["mixed_qkv"][:pad])], dim=0
    )
    gate = torch.cat([inp["gate"], torch.zeros_like(inp["gate"][:, :pad])], dim=1)
    beta = torch.cat([inp["beta"], torch.zeros_like(inp["beta"][:, :pad])], dim=1)
    out_gate = torch.cat(
        [inp["out_gate"], torch.zeros_like(inp["out_gate"][:pad])], dim=0
    )
    fused_cs, fused_ss = inp["conv_state"].clone(), inp["state"].clone()
    out = torch.zeros(
        mixed_qkv.shape[0], Hloc * SPEC_D, dtype=torch.bfloat16, device=device
    )
    fused_kda_decode(
        mixed_qkv,
        fused_cs,
        inp["conv_weight"],
        gate,
        beta,
        out_gate,
        inp["A_log"],
        inp["dt_bias"],
        fused_ss,
        inp["state_indices"],
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        SPEC_D,
        Hloc,
        -5.0,
        num_accepted_tokens=inp["num_accepted_tokens"],
        conv_state_indices=inp["conv_state_indices"],
        out=out,
    )
    torch.testing.assert_close(out[:8], ref, atol=0.15, rtol=0.1)
    torch.testing.assert_close(out[8:], torch.zeros_like(out[8:]), atol=0, rtol=0)
    torch.testing.assert_close(fused_cs, ref_cs, atol=0, rtol=0)
    torch.testing.assert_close(fused_ss, ref_ss, atol=0.05, rtol=0.02)


# num_spec 7 dispatches to the parallel kernels, 1 stays on the generic kernel
# at this batch. Each path guards the rollback in a different place: the
# parallel path in finalize, the generic path in the kernel itself.
@pytest.mark.parametrize("num_spec", [7, 1])
@pytest.mark.parametrize("null_pos", ["last", "first"])
def test_spec_decode_null_token_does_not_write_output(null_pos, num_spec):
    """A live sequence with one slot-0 token must leave that output row and
    physical slot 0 untouched. The accepted checkpoint stays valid, so the
    sequence-level NULL return does not cover this token.

    "last" is the shape vLLM produces: NULL slots are the padded tail of an
    index row.

    "first" stops the sequence. Tokens after the first invalid slot are not
    computed, and the convolution cache is left unchanged."""
    inp = _make_spec_inputs(1, 2, num_spec=num_spec)
    if null_pos == "last":
        null_tok = inp["state_indices"].shape[1] - 1
    else:
        null_tok = 0
        inp["num_accepted_tokens"][0] = 2
    inp["state_indices"][0, null_tok] = 0
    cidx = int(inp["conv_state_indices"][0].item())
    before_slot0 = inp["state"][0].clone()
    before_conv = inp["conv_state"][cidx].clone()
    out = _run_spec(inp)
    torch.testing.assert_close(
        out[null_tok], torch.zeros_like(out[null_tok]), atol=0, rtol=0
    )
    torch.testing.assert_close(inp["state"][0], before_slot0, atol=0, rtol=0)
    if null_pos == "last":
        assert out[:null_tok].abs().max().item() > 0
    else:
        torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
        torch.testing.assert_close(inp["conv_state"][cidx], before_conv, atol=0, rtol=0)


def _run_spec(inp, conv_weight=None, out=None):
    return fused_kda_decode(
        inp["mixed_qkv"],
        inp["conv_state"],
        inp["conv_weight"] if conv_weight is None else conv_weight,
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        inp["state"],
        inp["state_indices"],
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        SPEC_D,
        inp["state"].shape[1],
        -5.0,
        num_accepted_tokens=inp["num_accepted_tokens"],
        conv_state_indices=inp["conv_state_indices"],
        out=out,
    )


@pytest.mark.parametrize(
    "invalid_accepted,num_spec",
    [
        ("null_state", 7),
        ("null_state_and_conv", 7),
        ("count_past_width", 7),
        ("count_past_width", 1),
    ],
)
def test_fused_spec_decode_skips_invalid_accepted_state(invalid_accepted, num_spec):
    """Invalid accepted metadata must not commit state or convolution carry."""
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if get_arch() != "gfx950":
        pytest.skip("parallel spec-7 kernel is only dispatched on gfx950")

    inp = _make_spec_inputs(1, 2, num_spec=num_spec)
    if invalid_accepted == "count_past_width":
        inp["num_accepted_tokens"][0] = inp["state_indices"].shape[1] + 1
    else:
        inp["state_indices"].zero_()
        if invalid_accepted == "null_state_and_conv":
            inp["conv_state_indices"].zero_()
    before_cs, before_ss = inp["conv_state"].clone(), inp["state"].clone()
    out = _run_spec(inp)
    torch.testing.assert_close(inp["conv_state"], before_cs, atol=0, rtol=0)
    torch.testing.assert_close(inp["state"], before_ss, atol=0, rtol=0)
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)


def test_fused_spec_decode_rejects_unsupported_width():
    """Both speculative paths keep exactly W - 1 = 3 history taps."""
    inp = _make_spec_inputs(1, 2, num_spec=3)
    w3 = inp["conv_weight"][:, :3].contiguous()
    with pytest.raises(NotImplementedError, match="W == 4"):
        _run_spec(inp, conv_weight=w3)


def _assert_spec_matches(inp, *, expect_parallel, fused_weight=None):
    import aiter.ops.triton.gated_delta_net.fused_kda_decode as fkd

    ref_cs, ref_ss = inp["conv_state"].clone(), inp["state"].clone()
    ref = _ref_spec_decode(
        inp["mixed_qkv"],
        ref_cs,
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        ref_ss,
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        inp["state"].shape[1],
        state_indices=inp["state_indices"],
        num_accepted_tokens=inp["num_accepted_tokens"],
        conv_state_indices=inp["conv_state_indices"],
    )
    fused_cs, fused_ss = inp["conv_state"].clone(), inp["state"].clone()
    fused_weight = inp["conv_weight"] if fused_weight is None else fused_weight
    with _SpecKernelSpy(fkd) as dispatched:
        out = fused_kda_decode(
            inp["mixed_qkv"],
            fused_cs,
            fused_weight,
            inp["gate"],
            inp["beta"],
            inp["out_gate"],
            inp["A_log"],
            inp["dt_bias"],
            fused_ss,
            inp["state_indices"],
            inp["cu_seqlens"],
            inp["norm_weight"],
            1e-6,
            SPEC_D,
            inp["state"].shape[1],
            -5.0,
            num_accepted_tokens=inp["num_accepted_tokens"],
            conv_state_indices=inp["conv_state_indices"],
        )
    if expect_parallel:
        assert dispatched.grids, "specialized spec kernel was not dispatched"
    else:
        assert not dispatched.grids, "specialized spec kernel ran on the fallback path"
    torch.testing.assert_close(out, ref, atol=0.15, rtol=0.1)
    torch.testing.assert_close(fused_cs, ref_cs, atol=0, rtol=0)
    torch.testing.assert_close(fused_ss, ref_ss, atol=0.05, rtol=0.02)


@pytest.mark.parametrize(
    "num_spec,batch,seq_lens,expect_parallel",
    [
        # spec_tokens 2 is the one losing width: 0.77x at batch 1 and 0.94x at
        # batch 8, turning to 1.18x only at batch 16. The batch 8 row stops the
        # threshold dropping to 8 or below; nothing here pins it at exactly 16.
        (1, 2, None, False),
        (1, 8, None, False),
        (1, 16, None, True),
        # spec_tokens 3 pays at every batch (1.03x at batch 1, 1.31x at 8), so
        # it must dispatch even at a small batch. This row holds the width
        # threshold at 3 rather than 4.
        (2, 2, None, True),
        # A short packed sequence still dispatches; only the index-row width,
        # not total token count, defines the static speculative window.
        (7, 2, (3, 8), True),
    ],
)
def test_spec_dispatch_gate(num_spec, batch, seq_lens, expect_parallel):
    """Both gate thresholds are load-bearing, so both are pinned here.
    Either path has to match the reference."""
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    inp = _make_spec_inputs(
        batch,
        2,
        num_spec=num_spec,
        seq_lens=seq_lens,
    )
    _assert_spec_matches(
        inp, expect_parallel=expect_parallel and get_arch() == "gfx950"
    )


# The last row stays on the generic kernel, which reads the same index vector.
@pytest.mark.parametrize(
    "batch,num_spec,expect_parallel", [(4, 7, True), (2, 3, True), (2, 1, False)]
)
def test_spec_decode_honours_non_contiguous_index_strides(
    batch, num_spec, expect_parallel
):
    """A sliced conv_state_indices must select the same slots as a packed one.

    vLLM builds it as ssm_state_indices[:, 0], a column of the [B, S] index
    tensor, so its stride is S rather than 1 at every batch above one. Reading
    it as packed silently picks up another sequence's cache slot; the bug hides
    at batch 1, where a one-element slice is contiguous."""
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    inp = _make_spec_inputs(batch, 2, num_spec=num_spec)
    sliced = inp["state_indices"][:, 0]
    assert not sliced.is_contiguous(), "test would be vacuous on a packed slice"
    inp["conv_state_indices"] = sliced
    _assert_spec_matches(
        inp, expect_parallel=expect_parallel and get_arch() == "gfx950"
    )


@pytest.mark.parametrize(
    "invalid_layout",
    [
        "narrow_conv_cache",
        "zero_sequence_stride",
        "zero_token_stride",
        "zero_beta_token_stride",
        "zero_out_gate_token_stride",
        "empty_spec_row",
        "single_token_spec_row",
        "short_state_indices",
        "short_accepted_tokens",
        "short_conv_indices",
    ],
)
def test_spec_decode_rejects_invalid_layout(invalid_layout):
    num_spec = 7
    batch = 1 if invalid_layout == "narrow_conv_cache" else 2
    inp = _make_spec_inputs(batch, 2, num_spec=num_spec)
    spec_tokens = inp["state_indices"].shape[1]
    if invalid_layout == "narrow_conv_cache":
        required = spec_tokens + SPEC_W - 2
        inp["conv_state"] = inp["conv_state"][:, :, : required - 1].contiguous()
        inp["num_accepted_tokens"][0] = spec_tokens
        match = rf"conv_state\.shape\[2\] >= {required}"
    elif invalid_layout == "zero_sequence_stride":
        packed = inp["state_indices"]
        inp["state_indices"] = packed[:1].expand_as(packed)
        match = "positive sequence stride"
    elif invalid_layout == "zero_token_stride":
        packed = inp["state_indices"]
        inp["state_indices"] = packed[:, :1].expand_as(packed)
        match = "positive token stride"
    elif invalid_layout == "zero_beta_token_stride":
        inp["beta"] = inp["beta"][:, :1].expand_as(inp["beta"])
        match = "beta must have a positive token stride"
    elif invalid_layout == "zero_out_gate_token_stride":
        inp["out_gate"] = inp["out_gate"][:1].expand_as(inp["out_gate"])
        match = "out_gate must have a positive token stride"
    elif invalid_layout == "empty_spec_row":
        inp["state_indices"] = inp["state_indices"][:, :0]
        match = "at least 2 state-index entries"
    elif invalid_layout == "single_token_spec_row":
        inp["state_indices"] = inp["state_indices"][:, :1]
        match = "at least 2 state-index entries"
    else:
        key, name = {
            "short_state_indices": ("state_indices", "ssm_state_indices"),
            "short_accepted_tokens": (
                "num_accepted_tokens",
                "num_accepted_tokens",
            ),
            "short_conv_indices": ("conv_state_indices", "conv_state_indices"),
        }[invalid_layout]
        inp[key] = inp[key][:-1]
        match = rf"{name} must cover every sequence"
    with pytest.raises(ValueError, match=match):
        _run_spec(inp)


def test_spec_decode_span_past_index_width_drops_surplus_tokens():
    """A cu_seqlens span longer than ssm_state_indices is not read past the
    row. Surplus tokens stay at the preallocated output and do not move state."""
    hloc = 2
    inp = _make_spec_inputs(1, hloc, num_spec=7)
    spec_tokens = inp["state_indices"].shape[1]
    ref_cs, ref_ss = inp["conv_state"].clone(), inp["state"].clone()
    ref = _ref_spec_decode(
        inp["mixed_qkv"],
        ref_cs,
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        ref_ss,
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        hloc,
        state_indices=inp["state_indices"],
        num_accepted_tokens=inp["num_accepted_tokens"],
        conv_state_indices=inp["conv_state_indices"],
    )
    extra = 2
    lp = hloc * SPEC_D
    dtype = inp["mixed_qkv"].dtype
    inp["mixed_qkv"] = torch.cat(
        [
            inp["mixed_qkv"],
            torch.randn(extra, 3 * lp, dtype=dtype, device=device),
        ],
        dim=0,
    )
    inp["gate"] = torch.cat(
        [
            inp["gate"],
            torch.randn(1, extra, hloc, SPEC_D, dtype=dtype, device=device),
        ],
        dim=1,
    )
    inp["beta"] = torch.cat(
        [inp["beta"], torch.randn(1, extra, hloc, dtype=dtype, device=device)],
        dim=1,
    )
    inp["out_gate"] = torch.cat(
        [inp["out_gate"], torch.randn(extra, lp, dtype=dtype, device=device)],
        dim=0,
    )
    inp["cu_seqlens"] = inp["cu_seqlens"].clone()
    inp["cu_seqlens"][-1] += extra
    fused_cs, fused_ss = inp["conv_state"].clone(), inp["state"].clone()
    out = torch.zeros(spec_tokens + extra, lp, dtype=torch.bfloat16, device=device)
    # _run_spec writes the caller's state tensors. Swap in the clones.
    saved_cs, saved_ss = inp["conv_state"], inp["state"]
    inp["conv_state"], inp["state"] = fused_cs, fused_ss
    try:
        _run_spec(inp, out=out)
    finally:
        inp["conv_state"], inp["state"] = saved_cs, saved_ss
    torch.testing.assert_close(out[:spec_tokens], ref, atol=0.15, rtol=0.1)
    torch.testing.assert_close(
        out[spec_tokens:], torch.zeros_like(out[spec_tokens:]), atol=0, rtol=0
    )
    torch.testing.assert_close(fused_cs, ref_cs, atol=0, rtol=0)
    torch.testing.assert_close(fused_ss, ref_ss, atol=0.05, rtol=0.02)
