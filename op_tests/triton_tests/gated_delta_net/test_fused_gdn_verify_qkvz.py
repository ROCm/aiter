# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests of fused_gdn_verify_qkvz, the Gated DeltaNet speculative-verify step on
ReplaySSM state (gfx1250), against a float64 reference of the step. The reference
uses the bf16 roundings of the unfused chain: the conv outputs, beta and the readout
before the norm."""

import types

import pytest
import torch

from aiter.ops.triton.gated_delta_net.fused_gdn_verify_qkvz import (
    fused_gdn_verify_qkvz,
)

K = V = 128
SLOTS = 96
EPS = 1e-6
# (key heads, value heads): Qwen3.5-35B-A3B at TP4 and with DP attention, then
# head ratios 1, 3 and 4.
SHAPES = [(4, 8), (16, 32), (8, 8), (4, 12), (4, 16)]


def _requires_gfx1250():
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if get_arch() != "gfx1250":
        pytest.skip(f"the fused GDN verify kernel is gfx1250-only, got {get_arch()}")


def make_layer(kh, vh, seed, param_dtype=torch.float32, conv_bias=False):
    torch.manual_seed(seed)
    bf16 = {"dtype": torch.bfloat16, "device": "cuda"}
    qkv = 2 * kh * K + vh * V
    return types.SimpleNamespace(
        kh=kh,
        vh=vh,
        qkv=qkv,
        conv_weight=torch.randn((qkv, 4), **bf16) * 0.3,
        conv_bias=torch.randn((qkv,), **bf16) * 0.1 if conv_bias else None,
        A_log=torch.randn((vh,), dtype=torch.float32, device="cuda").to(param_dtype),
        dt_bias=torch.randn((vh,), **bf16).to(param_dtype),
        norm_weight=torch.randn((V,), **bf16).to(param_dtype),
    )


def make_pools(layer, window, cap, seed, num_slots=SLOTS):
    """ReplaySSM pools as a serving stack binds them: the conv windows stored as
    [slot, width, channels] and passed transposed, and K-major checkpoints."""
    torch.manual_seed(seed)
    bf16 = {"dtype": torch.bfloat16, "device": "cuda"}
    n, vh = num_slots, layer.vh
    record_k = torch.nn.functional.normalize(
        torch.randn((n, vh, cap, K), dtype=torch.float32, device="cuda"), dim=-1
    )
    return {
        "conv": torch.randn((n, window + 2, layer.qkv), **bf16),
        "ckpt": torch.randn((n, vh, K, V), **bf16) * 0.1,
        "rk": record_k.to(torch.bfloat16),
        "ru": torch.randn((n, vh, cap, V), **bf16) * 0.1,
        "rg": -torch.rand((n, vh, cap), dtype=torch.float32, device="cuda") * 0.3,
        "wp": torch.randint(
            0, cap - window + 1, (n,), dtype=torch.int32, device="cuda"
        ),
    }


def make_step(layer, seqs, window, seed, slots=None, num_slots=SLOTS, lengths=None):
    """One verify step. Each sequence has ``window`` tokens, or ``lengths[i]``."""
    torch.manual_seed(seed)
    vh = layer.vh
    if lengths is None:
        lengths = torch.full((seqs,), window, dtype=torch.int32, device="cuda")
    else:
        lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    cu = torch.zeros((seqs + 1,), dtype=torch.int32, device="cuda")
    cu[1:] = torch.cumsum(lengths, 0)
    n = int(cu[-1])
    # One projection output [q|k|v|z|b|a] with padding, so the rows are strided.
    width = layer.qkv + vh * V + 2 * vh
    qkvzba = torch.randn((n, width + 8), dtype=torch.bfloat16, device="cuda")
    qkvz = qkvzba[:, : layer.qkv + vh * V]
    ba = qkvzba[:, layer.qkv + vh * V : width]
    if slots is None:
        slots = torch.randperm(num_slots, dtype=torch.int32, device="cuda")[:seqs]
    return types.SimpleNamespace(
        n=n,
        window=window,
        qkvz=qkvz,
        ba=ba,
        mixed_qkv=qkvz[:, : layer.qkv],
        z=qkvz[:, layer.qkv :].view(n, vh, V),
        b=ba[:, :vh],
        a=ba[:, vh:],
        slots=slots,
        accepted=torch.randint(
            1, window + 1, (seqs,), dtype=torch.int32, device="cuda"
        ),
        cu=cu,
    )


def commit(pools, step, cap):
    """The ReplaySSM commit, which the caller runs once for each forward. It skips
    padded sequences, and a cursor of -1 (no records yet) becomes 0."""
    live = step.slots >= 0
    slots = step.slots[live].long()
    cursor = pools["wp"][slots]
    base = torch.where(cursor + 2 * step.window > cap, 0, cursor)
    pools["wp"][slots] = torch.where(cursor < 0, 0, base + step.accepted[live]).to(
        torch.int32
    )


def op_args(layer, step, pools, out):
    return {
        "projected_qkvz": step.qkvz,
        "projected_ba": step.ba,
        "conv_state": pools["conv"].transpose(-1, -2),
        "conv_state_indices": step.slots,
        "num_accepted_tokens": step.accepted,
        "query_start_loc": step.cu,
        "checkpoint": pools["ckpt"],
        "record_k": pools["rk"],
        "record_u": pools["ru"],
        "record_g": pools["rg"],
        "write_pos": pools["wp"],
        "slot_idx": step.slots,
        "conv_weight": layer.conv_weight,
        "conv_bias": layer.conv_bias,
        "A_log": layer.A_log,
        "dt_bias": layer.dt_bias,
        "norm_weight": layer.norm_weight,
        "out": out,
    }


def run_op(layer, step, pools):
    out = torch.empty((step.n, layer.vh, V), dtype=torch.bfloat16, device="cuda")
    args = op_args(layer, step, pools, out)
    fused_gdn_verify_qkvz(
        max_query_len=step.window, scale=K**-0.5, norm_eps=EPS, **args
    )
    return out.double()


def reference(layer, step, pools, cap):
    """The verify step in float64, one sequence at a time, from the pools after the
    commit. Returns the output, with NaN in the rows that the op must not write, and
    the pools that the step must leave. It changes nothing."""
    kh, vh, window = layer.kh, layer.vh, step.window
    ratio = vh // kh
    expected = {name: x.clone() for name, x in pools.items()}
    truth = torch.full(
        (step.n, vh, V), float("nan"), dtype=torch.float64, device="cuda"
    )
    taps = layer.conv_weight.double().t()
    decay_rate = torch.exp(layer.A_log.double())
    dt_bias = layer.dt_bias.double()
    weight = layer.norm_weight.double()
    cu, accepted, cursors = (
        step.cu.tolist(),
        step.accepted.tolist(),
        pools["wp"].tolist(),
    )
    for seq, slot in enumerate(step.slots.tolist()):
        bos, eos = cu[seq], cu[seq + 1]
        if slot < 0 or eos == bos:
            continue
        # The three inputs before the window start at accepted - 1. The new window is
        # [h1, h2, x_0 .. x_{tokens-1}], and the later positions do not change.
        history = pools["conv"][slot, accepted[seq] - 1 : accepted[seq] + 2]
        inputs = step.mixed_qkv[bos:eos]
        expected["conv"][slot, : 2 + eos - bos] = torch.cat([history[1:], inputs])
        full = torch.cat([history, inputs]).double()
        conv = torch.stack([(full[t : t + 4] * taps).sum(0) for t in range(eos - bos)])
        if layer.conv_bias is not None:
            conv = conv + layer.conv_bias.double()
        conv = torch.nn.functional.silu(conv).to(torch.bfloat16).double()
        q, k, v = conv.split([kh * K, kh * K, vh * V], -1)
        q = q.view(-1, kh, K).repeat_interleave(ratio, 1)
        k = k.view(-1, kh, K).repeat_interleave(ratio, 1)
        v = v.view(-1, vh, V)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * K**-0.5
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        g = -decay_rate * torch.nn.functional.softplus(
            step.a[bos:eos].double() + dt_bias
        )
        beta = torch.sigmoid(step.b[bos:eos].double()).to(torch.bfloat16).double()
        # Rebuild the state [VH, K, V] from the checkpoint and the committed records.
        state = pools["ckpt"][slot].double()
        committed = max(cursors[slot], 0)
        for j in range(committed):
            state = torch.exp(pools["rg"][slot, :, j].double())[:, None, None] * state
            state = state + (
                pools["rk"][slot, :, j, :, None].double()
                * pools["ru"][slot, :, j, None, :].double()
            )
        flush = committed + 2 * window > cap
        if flush:
            expected["ckpt"][slot] = state.to(pools["ckpt"].dtype)
        first = 0 if flush else committed
        for t in range(eos - bos):
            state = torch.exp(g[t])[:, None, None] * state
            u = beta[t][:, None] * (v[t] - (state * k[t][:, :, None]).sum(1))
            state = state + k[t][:, :, None] * u[:, None, :]
            core = (state * q[t][:, :, None]).sum(1).to(torch.bfloat16).double()
            z = step.z[bos + t].double()
            rms = torch.rsqrt((core * core).mean(-1, keepdim=True) + EPS)
            truth[bos + t] = core * rms * weight * z * torch.sigmoid(z)
            expected["rk"][slot, :, first + t] = k[t].to(pools["rk"].dtype)
            expected["ru"][slot, :, first + t] = u.to(pools["ru"].dtype)
            expected["rg"][slot, :, first + t] = g[t].to(pools["rg"].dtype)
    return truth, expected


def check_output(actual, truth):
    """The rows that the reference wrote, against float64."""
    rows = ~torch.isnan(truth[:, 0, 0])
    assert rows.any(), "the step has no live tokens"
    err = (actual[rows] - truth[rows]).abs()
    relative = (err / truth[rows].abs().clamp_min(1.0)).max().item()
    assert relative < 1e-2, f"max error {relative:.4f} of max(|x|, 1)"
    assert err.mean().item() < 1e-3, f"mean error {err.mean().item():.5f}"


def check_pools(got, expected):
    """Conv windows and cursors exact, the rest within one bf16 rounding step."""
    assert torch.equal(got["conv"], expected["conv"]), "conv windows differ"
    assert torch.equal(got["wp"], expected["wp"]), "the op must not move write_pos"
    for name, tol in (("ckpt", 4e-3), ("rk", 8e-3), ("ru", 3.2e-2), ("rg", 1e-6)):
        torch.testing.assert_close(
            got[name].float(), expected[name].float(), atol=tol, rtol=tol, msg=name
        )


def one_step(kh, vh, seqs, window, cap, seed, lengths=None, **layer_kwargs):
    layer = make_layer(kh, vh, seed=seed, **layer_kwargs)
    pools = make_pools(layer, window, cap, seed=100 + seed)
    step = make_step(layer, seqs, window, seed=200 + seed, lengths=lengths)
    commit(pools, step, cap)
    truth, expected = reference(layer, step, pools, cap)
    check_output(run_op(layer, step, pools), truth)
    check_pools(pools, expected)


@pytest.mark.parametrize("kh,vh", SHAPES)
@pytest.mark.parametrize("seqs", [1, 3, 64])
def test_fused_gdn_verify(kh, vh, seqs):
    _requires_gfx1250()
    one_step(kh, vh, seqs, window=4, cap=16, seed=seqs)


@pytest.mark.parametrize(
    "kh,vh,window,cap",
    [
        (4, 8, 1, 16),
        (4, 8, 2, 16),
        (4, 8, 3, 8),
        (4, 8, 4, 24),
        (4, 8, 5, 16),
        (4, 8, 8, 16),
        (4, 8, 11, 32),
        (4, 8, 16, 32),
        (8, 8, 16, 64),
        (4, 12, 8, 32),
        (4, 16, 4, 32),
    ],
)
def test_fused_gdn_verify_window_and_cache_lengths(kh, vh, window, cap):
    """Windows and record buffers of several lengths, also lengths that are not a
    power of two."""
    _requires_gfx1250()
    one_step(kh, vh, 8, window=window, cap=cap, seed=window)


@pytest.mark.parametrize("kh,vh", [(4, 8), (4, 12)])
@pytest.mark.parametrize("window", [4, 5])
def test_fused_gdn_verify_ragged_windows(kh, vh, window):
    """Sequences with fewer tokens than max_query_len: the missing tokens write no
    records, no outputs and no conv window positions."""
    _requires_gfx1250()
    lengths = [i % window + 1 for i in range(16)]
    one_step(kh, vh, 16, window=window, cap=16, seed=window, lengths=lengths)


@pytest.mark.parametrize("param_dtype", [torch.float32, torch.bfloat16])
def test_fused_gdn_verify_conv_bias_and_param_dtypes(param_dtype):
    _requires_gfx1250()
    one_step(
        16, 32, 16, window=4, cap=16, seed=5, param_dtype=param_dtype, conv_bias=True
    )


@pytest.mark.parametrize("kh,vh", [(4, 8), (4, 12)])
def test_fused_gdn_verify_flush_boundaries(kh, vh):
    """Cursors on both sides of the flush rule, the cursor -1 of a sequence after its
    prefill, and every count of accepted tokens, in one batch."""
    _requires_gfx1250()
    window, cap = 4, 16
    layer = make_layer(kh, vh, seed=7)
    pools = make_pools(layer, window, cap, seed=8)
    cursors = cap - window + 2
    step = make_step(layer, window * cursors, window, seed=9)
    cursor = torch.arange(-1, cap - window + 1, dtype=torch.int32, device="cuda")
    pools["wp"][step.slots.long()] = cursor.repeat_interleave(window)
    step.accepted = torch.arange(
        1, window + 1, dtype=torch.int32, device="cuda"
    ).repeat(cursors)
    commit(pools, step, cap)
    truth, expected = reference(layer, step, pools, cap)
    check_output(run_op(layer, step, pools), truth)
    check_pools(pools, expected)


def test_fused_gdn_verify_reads_an_empty_cursor():
    """A cursor of -1 means no records. The op reads it as 0, also without the
    commit."""
    _requires_gfx1250()
    window, cap = 4, 16
    layer = make_layer(4, 8, seed=51)
    pools = make_pools(layer, window, cap, seed=52)
    step = make_step(layer, 8, window, seed=53)
    pools["wp"][step.slots.long()] = -1
    truth, expected = reference(layer, step, pools, cap)
    check_output(run_op(layer, step, pools), truth)
    check_pools(pools, expected)


@pytest.mark.parametrize("kh,vh", [(4, 8), (16, 32)])
def test_fused_gdn_verify_many_steps(kh, vh):
    """Twelve steps on the same sequences with random acceptance, so the cursor goes
    through several flushes. Each step is checked from the op's own pools."""
    _requires_gfx1250()
    window, cap = 4, 16
    layer = make_layer(kh, vh, seed=11)
    pools = make_pools(layer, window, cap, seed=12)
    slots = torch.randperm(SLOTS, dtype=torch.int32, device="cuda")[:8]
    flushes = 0
    for i in range(12):
        step = make_step(layer, 8, window, seed=300 + i, slots=slots)
        commit(pools, step, cap)
        flushes += (pools["wp"][slots.long()] + 2 * window > cap).sum().item()
        truth, expected = reference(layer, step, pools, cap)
        check_output(run_op(layer, step, pools), truth)
        check_pools(pools, expected)
    assert flushes > 8, "the steps must cross the flush rule"


def test_fused_gdn_verify_skips_padded_sequences():
    """A sequence without tokens and a sequence with a PAD_SLOT_ID slot change
    nothing, and the live sequences next to them are exact."""
    _requires_gfx1250()
    window, cap = 4, 16
    layer = make_layer(4, 8, seed=21)
    pools = make_pools(layer, window, cap, seed=22)
    step = make_step(layer, 4, window, seed=23, lengths=[4, 0, 2, 3])
    step.slots[3] = -1
    commit(pools, step, cap)
    truth, expected = reference(layer, step, pools, cap)
    out = torch.full((step.n, layer.vh, V), 7.0, dtype=torch.bfloat16, device="cuda")
    fused_gdn_verify_qkvz(
        max_query_len=window,
        scale=K**-0.5,
        norm_eps=EPS,
        **op_args(layer, step, pools, out),
    )
    check_output(out.double(), truth)
    padded = torch.isnan(truth[:, 0, 0])
    assert padded.sum().item() == 3 and torch.all(out[padded] == 7.0)
    check_pools(pools, expected)


def test_fused_gdn_verify_is_deterministic():
    _requires_gfx1250()
    window, cap = 4, 16
    layer = make_layer(16, 32, seed=31)
    pools = make_pools(layer, window, cap, seed=32)
    step = make_step(layer, 16, window, seed=33)
    commit(pools, step, cap)
    outs = []
    for _ in range(2):
        copy = {name: x.clone() for name, x in pools.items()}
        outs.append((run_op(layer, step, copy), copy))
    assert torch.equal(outs[0][0], outs[1][0])
    for name in pools:
        assert torch.equal(outs[0][1][name], outs[1][1][name]), name
