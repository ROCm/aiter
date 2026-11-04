# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 KDA speculative verify (flydsl_kda_verify_with_f_b) against FP32 references."""

import argparse
import itertools
from dataclasses import dataclass

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.linear_attention_kernels import (
    flydsl_kda_decode_with_f_b,
    flydsl_kda_verify_with_f_b,
    is_flydsl_kda_decode_supported,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx950"]

pytestmark = pytest.mark.skipif(
    not is_flydsl_kda_decode_supported(), reason="requires a gfx950 GPU"
)

_H = 12
_D = 128
_C = 3 * _H * _D
_LOWER_BOUND = -5.0
_NORM_EPS = 1e-5
_DEVICE = "cuda"
# Output errors are BF16 roundings of the gated RMSNorm (|out| up to ~4).
_OUT_TOL = {"atol": 2e-2, "rtol": 2e-2}


@dataclass
class VerifyInputs:
    f_a: torch.Tensor
    f_b_weight: torch.Tensor
    x: torch.Tensor
    conv_weight: torch.Tensor
    conv_state: torch.Tensor
    raw_beta: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    state: torch.Tensor
    cache_indices: torch.Tensor
    inter_indices: torch.Tensor
    window: torch.Tensor
    output_gate: torch.Tensor
    norm_weight: torch.Tensor
    inter_states: torch.Tensor | None
    ring: tuple | None


def _make_inputs(
    batch, num_tokens, state_dtype, ring, seed=7, steps=None, sglang_layout=True
):
    g = torch.Generator(device=_DEVICE).manual_seed(seed + 31 * batch + num_tokens)
    tokens = batch * num_tokens
    steps = steps or num_tokens

    def randn(*shape, dtype=torch.bfloat16, scale=1.0):
        return (scale * torch.randn(shape, generator=g, device=_DEVICE)).to(dtype)

    # SGLang slices x / f_a / beta / gate out of one wide in-projection.
    wide = randn(tokens, _C + _D + _H + _H * _D + 24)
    x = wide[:, :_C]
    f_a = wide[:, _C : _C + _D]
    raw_beta = wide[:, _C + _D : _C + _D + _H].unsqueeze(0)
    output_gate = wide[:, _C + _D + _H : _C + _D + _H + _H * _D].view(tokens, _H, _D)

    slots = batch + 3
    if sglang_layout:
        # SGLang stores conv state [slots, 3, 4608] and passes transpose(-1, -2).
        conv_state = randn(slots, 3, _C).transpose(-1, -2)
        window = torch.zeros(
            slots, steps, 3, _C, dtype=torch.bfloat16, device=_DEVICE
        ).transpose(-1, -2)
    else:
        conv_state = randn(slots, _C, 3)
        window = torch.zeros(slots, steps, _C, 3, dtype=torch.bfloat16, device=_DEVICE)
    state = randn(slots, _H, _D, _D, dtype=torch.float32, scale=0.05).to(state_dtype)
    inter_states = None
    ring_tensors = None
    if ring:
        L = steps + 1
        ring_tensors = (
            torch.zeros(slots, _H, L, _D, dtype=torch.bfloat16, device=_DEVICE),
            torch.zeros(slots, _H, L, _D, dtype=torch.bfloat16, device=_DEVICE),
            torch.zeros(slots, _H, L, _D, dtype=torch.float32, device=_DEVICE),
            torch.zeros(slots, _H, L, dtype=torch.float32, device=_DEVICE),
        )
    else:
        inter_states = torch.zeros(
            slots, steps, _H, _D, _D, dtype=state_dtype, device=_DEVICE
        )
    perm = torch.randperm(slots, generator=g, device=_DEVICE)
    return VerifyInputs(
        f_a=f_a,
        f_b_weight=randn(_H, _D, _D, scale=0.05),
        x=x,
        conv_weight=randn(_C, 4, dtype=torch.float32, scale=0.5),
        conv_state=conv_state,
        raw_beta=raw_beta,
        A_log=randn(_H, dtype=torch.float32, scale=0.5),
        dt_bias=randn(_H * _D, dtype=torch.float32, scale=0.5),
        state=state,
        cache_indices=perm[:batch].to(torch.int32),
        inter_indices=perm[slots - batch :].flip(0).to(torch.int32),
        window=window,
        output_gate=output_gate,
        norm_weight=randn(_D),
        inter_states=inter_states,
        ring=ring_tensors,
    )


def _run(
    inp: VerifyInputs, num_tokens: int, out: torch.Tensor | None = None
) -> torch.Tensor:
    return flydsl_kda_verify_with_f_b(
        f_a=inp.f_a,
        f_b_weight=inp.f_b_weight,
        x=inp.x,
        conv_weight=inp.conv_weight,
        conv_bias=None,
        conv_state=inp.conv_state,
        raw_beta=inp.raw_beta,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        lower_bound=_LOWER_BOUND,
        state=inp.state,
        cache_indices=inp.cache_indices,
        intermediate_indices=inp.inter_indices,
        intermediate_conv_window=inp.window,
        output_gate=inp.output_gate,
        norm_weight=inp.norm_weight,
        norm_eps=_NORM_EPS,
        num_tokens=num_tokens,
        intermediate_states=inp.inter_states,
        replay_ring=inp.ring,
        out=out,
    )


def _reference(inp: VerifyInputs, num_tokens: int):
    """Returns (out, windows, states-or-None, ring-or-None) for every sequence."""
    batch = inp.cache_indices.shape[0]
    tokens = batch * num_tokens
    out = torch.zeros(1, tokens, _H, _D, dtype=torch.bfloat16, device=_DEVICE)
    window = inp.window.clone()
    states = None if inp.inter_states is None else inp.inter_states.clone()
    ring = None if inp.ring is None else tuple(t.clone() for t in inp.ring)
    w = inp.conv_weight.float()
    a = inp.A_log.float().exp()[:, None]
    dt = inp.dt_bias.view(_H, _D)
    norm_w = inp.norm_weight.float()
    for b in range(batch):
        slot = int(inp.cache_indices[b])
        line = int(inp.inter_indices[b])
        if not (0 <= slot < inp.state.shape[0] and 0 <= line < window.shape[0]):
            continue
        hist = inp.conv_state[slot].float().clone()  # [C, 3]
        h = inp.state[slot].float().clone()  # [H, V, K]
        for t in range(num_tokens):
            tok = b * num_tokens + t
            cur = inp.x[tok].float()
            conv = (hist * w[:, :3]).sum(-1) + cur * w[:, 3]
            hist = torch.cat((hist[:, 1:], cur[:, None]), dim=-1)
            window[line, t] = hist.to(torch.bfloat16)
            qkv = F.silu(conv).to(torch.bfloat16).view(3, _H, _D)
            q, k, v = (t_.float() for t_ in qkv.unbind(0))
            proj = torch.einsum("k,hok->ho", inp.f_a[tok].float(), inp.f_b_weight.float())
            proj = proj.to(torch.bfloat16).float()
            log_decay = _LOWER_BOUND * torch.sigmoid((proj + dt) * a)  # [H, K]
            beta = torch.sigmoid(inp.raw_beta[0, tok].float())  # [H]
            if ring is not None:
                ring[0][slot, :, t] = qkv[2]
                ring[1][slot, :, t] = qkv[1]
                ring[2][slot, :, t] = log_decay
                ring[3][slot, :, t] = beta
            qn = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * _D**-0.5
            kn = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            h = h * log_decay.exp()[:, None, :]
            v_new = (v - torch.einsum("hvk,hk->hv", h, kn)) * beta[:, None]
            h = h + v_new[:, :, None] * kn[:, None, :]
            rec = torch.einsum("hvk,hk->hv", h, qn).to(torch.bfloat16).float()
            if states is not None:
                states[line, t] = h.to(states.dtype)
            inv_rms = torch.rsqrt(rec.square().mean(-1, keepdim=True) + _NORM_EPS)
            gate = torch.sigmoid(inp.output_gate[tok].float())
            out[0, tok] = (rec * inv_rms * norm_w * gate).to(torch.bfloat16)
    return out, window, states, ring


def _assert_close(ref, act, msg, atol, rtol, max_bad=0.0):
    ref = ref.float()
    act = act.float()
    bad = ~torch.isclose(act, ref, atol=atol, rtol=rtol)
    ratio = bad.float().mean().item()
    assert ratio <= max_bad, (
        f"{msg}: {ratio:.4%} outside atol={atol} rtol={rtol}, "
        f"max abs err {(act - ref).abs().max().item():.3e}"
    )


def _check(inp, num_tokens, max_bad=1e-3):
    conv_before = inp.conv_state.clone()
    state_before = inp.state.clone()
    ref_out, ref_window, ref_states, ref_ring = _reference(inp, num_tokens)
    out = _run(inp, num_tokens)
    torch.cuda.synchronize()
    # Verify never advances the caches.
    assert torch.equal(inp.conv_state, conv_before)
    assert torch.equal(inp.state, state_before)
    assert torch.equal(inp.window, ref_window), "conv windows differ"
    _assert_close(ref_out, out, "output", **_OUT_TOL, max_bad=max_bad)
    if ref_states is not None:
        tol = (1e-4, 1e-3) if inp.state.dtype == torch.float32 else (1e-4, 1e-2)
        _assert_close(ref_states, inp.inter_states, "states", *tol, max_bad=max_bad)
    if ref_ring is not None:
        _assert_close(ref_ring[0], inp.ring[0], "ring v", 1e-2, 1e-2, max_bad)
        _assert_close(ref_ring[1], inp.ring[1], "ring k", 1e-2, 1e-2, max_bad)
        _assert_close(ref_ring[2], inp.ring[2], "ring g", 1e-4, 1e-3, max_bad)
        _assert_close(ref_ring[3], inp.ring[3], "ring beta", 1e-5, 1e-4, 0.0)
    return out


@pytest.mark.parametrize("ring", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 2, 5, 6, 8])
@pytest.mark.parametrize("batch", [1, 3, 17])
def test_kda_verify_matches_reference(batch, num_tokens, state_dtype, ring):
    _check(_make_inputs(batch, num_tokens, state_dtype, ring), num_tokens)


def test_kda_verify_channel_major_conv_and_longer_scratch():
    inp = _make_inputs(4, 5, torch.bfloat16, False, steps=8, sglang_layout=False)
    _check(inp, 5)


@pytest.mark.parametrize("ring", [False, True])
def test_kda_verify_padding_rows(ring):
    inp = _make_inputs(6, 5, torch.bfloat16, ring)
    inp.cache_indices[1] = -1
    inp.inter_indices[3] = -1
    inp.cache_indices[4] = inp.state.shape[0]  # out of range is padding too
    out = _check(inp, 5)
    for b in (1, 3, 4):
        assert torch.count_nonzero(out[0, b * 5 : (b + 1) * 5]) == 0


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_kda_verify_token_zero_matches_fused_decode(state_dtype):
    """Token 0 of a verify equals one decode step from the same caches."""
    inp = _make_inputs(5, 6, state_dtype, False)
    out = _run(inp, 6)
    conv = inp.conv_state.clone()
    state = inp.state.clone()
    rows = torch.arange(5, device=_DEVICE) * 6
    decode_out = flydsl_kda_decode_with_f_b(
        f_a=inp.f_a[rows].contiguous(),
        f_b_weight=inp.f_b_weight,
        x=inp.x[rows].contiguous(),
        conv_weight=inp.conv_weight,
        conv_bias=None,
        conv_state=conv,
        raw_beta=inp.raw_beta[:, rows].contiguous(),
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        lower_bound=_LOWER_BOUND,
        state=state,
        state_indices=inp.cache_indices,
        output_gate=inp.output_gate[rows].contiguous(),
        norm_weight=inp.norm_weight,
        norm_eps=_NORM_EPS,
    )
    torch.testing.assert_close(out[:, rows], decode_out, atol=0, rtol=0)
    for b in range(5):
        line = int(inp.inter_indices[b])
        slot = int(inp.cache_indices[b])
        torch.testing.assert_close(
            inp.inter_states[line, 0], state[slot], atol=0, rtol=0
        )


def test_kda_verify_rejects_bad_arguments():
    inp = _make_inputs(2, 5, torch.bfloat16, False)
    with pytest.raises(ValueError, match="exactly one"):
        flydsl_kda_verify_with_f_b(
            **{**_kwargs(inp, 5), "intermediate_states": None, "replay_ring": None}
        )
    with pytest.raises(ValueError, match="num_tokens"):
        flydsl_kda_verify_with_f_b(**{**_kwargs(inp, 5), "num_tokens": 9})
    with pytest.raises(ValueError, match="intermediate_states"):
        flydsl_kda_verify_with_f_b(
            **{**_kwargs(inp, 5), "intermediate_states": inp.inter_states.float()}
        )


def _kwargs(inp, num_tokens):
    return dict(
        f_a=inp.f_a,
        f_b_weight=inp.f_b_weight,
        x=inp.x,
        conv_weight=inp.conv_weight,
        conv_bias=None,
        conv_state=inp.conv_state,
        raw_beta=inp.raw_beta,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        lower_bound=_LOWER_BOUND,
        state=inp.state,
        cache_indices=inp.cache_indices,
        intermediate_indices=inp.inter_indices,
        intermediate_conv_window=inp.window,
        output_gate=inp.output_gate,
        norm_weight=inp.norm_weight,
        norm_eps=_NORM_EPS,
        num_tokens=num_tokens,
        intermediate_states=inp.inter_states,
        replay_ring=inp.ring,
    )


def _verify_work(batch, num_tokens, state_dtype, ring):
    """FLOPs and useful bytes of one verify launch."""
    tokens = batch * num_tokens
    state_bytes = _H * _D * _D * torch.empty((), dtype=state_dtype).element_size()
    # Per token and head: the recurrence (~7*K*V), the f_b projection (2*K*K)
    # and the width-4 convolution of three channels.
    flops = tokens * _H * (7 * _D * _D + 2 * _D * _D + 3 * 2 * 4 * _D)
    # Read the state and conv history once per sequence, read each token's
    # x/f_a/beta/gate, write its output and its conv window, and either a
    # state snapshot or the ring entries; the weights are read once.
    nbytes = batch * (state_bytes + _C * 3 * 2)
    nbytes += tokens * ((_C + _D + _H + 2 * _H * _D) * 2 + _C * 3 * 2)
    if ring:
        nbytes += tokens * _H * (_D * (2 + 2 + 4) + 4)
    else:
        nbytes += tokens * state_bytes
    nbytes += _H * _D * _D * 2 + _C * 4 * 4 + _H * _D * 4 + _H * 4 + _D * 2
    return flops, nbytes


@benchmark()
def test_flydsl_kda_verify(batch, num_tokens, state_dtype, mode):
    ring = mode == "ring"
    inp = _make_inputs(batch, num_tokens, state_dtype, ring)
    ref_out, ref_window, ref_states, ref_ring = _reference(inp, num_tokens)
    # SGLang's KDA backend takes the kernel's own output allocation.
    candidates = {"flydsl": lambda inputs: _run(inputs, num_tokens)}

    flops, nbytes = _verify_work(batch, num_tokens, state_dtype, ring)
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        # The kernel only writes scratch, so rotated copies serve to keep the
        # state and scratch reads cold, not to restore inputs.
        rotations = max(1, min(32, (1 << 30) // max(1, nbytes)))
        _, us = run_perftest(fn, inp, num_rotate_args=rotations)
        out = fn(inp)
        err = checkAllclose(
            ref_out.to(dtypes.fp32),
            out.to(dtypes.fp32),
            tol_err_ratio=1e-3,
            msg=f"{name}: KDA verify output ",
            **_OUT_TOL,
        )
        assert torch.equal(inp.window, ref_window), f"{name}: conv windows differ"
        if ref_states is not None:
            err = max(
                err,
                checkAllclose(
                    ref_states.to(dtypes.fp32),
                    inp.inter_states.to(dtypes.fp32),
                    atol=1e-4,
                    rtol=1e-2,
                    tol_err_ratio=1e-3,
                    msg=f"{name}: KDA verify state snapshots ",
                ),
            )
        if ref_ring is not None:
            for ref_part, part, label in zip(ref_ring, inp.ring, ("v", "k", "g", "beta")):
                err = max(
                    err,
                    checkAllclose(
                        ref_part.to(dtypes.fp32),
                        part.to(dtypes.fp32),
                        atol=1e-2,
                        rtol=1e-2,
                        tol_err_ratio=1e-3,
                        msg=f"{name}: KDA verify ring {label} ",
                    ),
                )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


test_flydsl_kda_verify.__test__ = False


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("FlyDSL KDA verify unsupported on %s; skipping", get_gfx())
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL Kimi-K3 KDA verify correctness + performance sweep",
    )
    # Kimi-K3 TP8 serving: 12 local heads of 128; DSpark verifies block + 1
    # tokens (5 to 8 in the AgentX recipes) for 1 to 60 running requests.
    parser.add_argument(
        "-b", "--batch", type=int, nargs="*", default=[1, 2, 4, 14, 16, 32, 60]
    )
    parser.add_argument("-t", "--tokens", type=int, nargs="*", default=[5, 6, 8])
    parser.add_argument(
        "--state-dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.fp32, dtypes.bf16],
        nargs="*",
        default=[dtypes.bf16],
        help="Recurrent-state storage. Example: --state-dtype bf16 fp32",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["snapshot", "ring"],
        nargs="*",
        default=["snapshot", "ring"],
        help="Per-step state snapshots, or the ReplaySSM ring.",
    )
    args = parser.parse_args()
    rows = [
        test_flydsl_kda_verify(batch, tokens, state_dtype, mode)
        for state_dtype, mode, tokens, batch in itertools.product(
            args.state_dtype, args.mode, args.tokens, args.batch
        )
    ]
    aiter.logger.info(
        "FlyDSL KDA verify summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
