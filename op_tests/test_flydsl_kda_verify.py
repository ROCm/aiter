# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 KDA speculative verify (flydsl_kda_verify_with_f_b) against FP32 references."""

import argparse
import dataclasses
import itertools
import re
from dataclasses import dataclass

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import linear_attention_kernels
from aiter.ops.flydsl.linear_attention_kernels import (
    flydsl_kda_decode_with_f_b,
    flydsl_kda_verify_with_f_b,
    is_flydsl_kda_decode_supported,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx942", "gfx950"]

pytestmark = pytest.mark.skipif(
    not is_flydsl_kda_decode_supported(), reason="requires a gfx942 or gfx950 GPU"
)

_H = 12
_D = 128
_C = 3 * _H * _D
_HDD = _H * _D * _D
# Elements of one step's [4608, 3] conv window.
_WINDOW_STEP = 3 * _C
_LOWER_BOUND = -5.0
_NORM_EPS = 1e-5
_DEVICE = "cuda"
# Output errors are BF16 roundings of the gated RMSNorm (|out| up to ~4).
_OUT_TOL = {"atol": 2e-2, "rtol": 2e-2}
_RING_PARTS = ("raw_v", "raw_k", "g", "beta")


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


def _kwargs(inp, num_tokens):
    return {
        "f_a": inp.f_a,
        "f_b_weight": inp.f_b_weight,
        "x": inp.x,
        "conv_weight": inp.conv_weight,
        "conv_bias": None,
        "conv_state": inp.conv_state,
        "raw_beta": inp.raw_beta,
        "A_log": inp.A_log,
        "dt_bias": inp.dt_bias,
        "lower_bound": _LOWER_BOUND,
        "state": inp.state,
        "cache_indices": inp.cache_indices,
        "intermediate_indices": inp.inter_indices,
        "intermediate_conv_window": inp.window,
        "output_gate": inp.output_gate,
        "norm_weight": inp.norm_weight,
        "norm_eps": _NORM_EPS,
        "num_tokens": num_tokens,
        "intermediate_states": inp.inter_states,
        "replay_ring": inp.ring,
    }


def _run(
    inp: VerifyInputs,
    num_tokens: int,
    out: torch.Tensor | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    return flydsl_kda_verify_with_f_b(
        **_kwargs(inp, num_tokens), out=out, num_warps=num_warps
    )


# Activations SGLang slices out of one wide in-projection, so they are not dense.
_STRIDED_INPUTS = ("x", "f_a", "raw_beta", "output_gate")


def _copy_inputs(inp: VerifyInputs) -> VerifyInputs:
    """Copy every tensor with its strides; clone() would compact the activations.

    Caches stay clone()d: their dense layouts keep strides, and a copy of the
    large-offset test's padded pools would repeat their multi-GiB gaps.
    """

    def copy(name, value):
        if name in _STRIDED_INPUTS:
            out = torch.empty_strided(
                value.shape, value.stride(), dtype=value.dtype, device=value.device
            )
            return out.copy_(value)
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, tuple):
            return tuple(t.clone() for t in value)
        return value

    return VerifyInputs(
        **{f.name: copy(f.name, getattr(inp, f.name)) for f in dataclasses.fields(inp)}
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
    # Padding follows the kernel: an index past any cache it addresses is padding.
    slot_caches = (inp.state, inp.conv_state) + (inp.ring or ())
    num_slots = min(t.shape[0] for t in slot_caches)
    line_caches = (window,) if states is None else (window, states)
    num_lines = min(t.shape[0] for t in line_caches)
    for b in range(batch):
        slot = int(inp.cache_indices[b])
        line = int(inp.inter_indices[b])
        if not (0 <= slot < num_slots and 0 <= line < num_lines):
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
            proj = torch.einsum(
                "k,hok->ho", inp.f_a[tok].float(), inp.f_b_weight.float()
            )
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


def _assert_matches(inp, out, refs, max_bad=1e-3):
    """Check the output and every verify write against :func:`_reference`."""
    ref_out, ref_window, ref_states, ref_ring = refs
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


def _check(inp, num_tokens, max_bad=1e-3, num_warps=None):
    conv_before = inp.conv_state.clone()
    state_before = inp.state.clone()
    refs = _reference(inp, num_tokens)
    out = _run(inp, num_tokens, num_warps=num_warps)
    torch.cuda.synchronize()
    # Verify never advances the caches.
    assert torch.equal(inp.conv_state, conv_before)
    assert torch.equal(inp.state, state_before)
    _assert_matches(inp, out, refs, max_bad)
    return out


def _verify_writes(inp):
    """Every buffer the kernel writes, other than ``out``."""
    written = [inp.window]
    written += [inp.inter_states] if inp.ring is None else list(inp.ring)
    return written


def _bits_equal(a, b):
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("num_warps", flydsl_kda_verify_with_f_b.num_warps)
@pytest.mark.parametrize("ring", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 2, 5, 6, 8])
@pytest.mark.parametrize("batch", [1, 3, 17])
def test_kda_verify_matches_reference(batch, num_tokens, state_dtype, ring, num_warps):
    _check(
        _make_inputs(batch, num_tokens, state_dtype, ring),
        num_tokens,
        num_warps=num_warps,
    )


@pytest.mark.parametrize("cus", [256, 80, 32, 8])
def test_kda_verify_advertises_num_warps(monkeypatch, cus):
    """The tag lists every wave count the automatic policy picks, so a caller
    that pre-builds each one before graph capture covers every batch."""
    monkeypatch.setattr(linear_attention_kernels, "get_num_sms", lambda: cus)
    picks = {
        linear_attention_kernels._kda_verify_num_warps(batch, ring)
        for batch in range(1, 257)
        for ring in (False, True)
    }
    assert picks <= set(flydsl_kda_verify_with_f_b.num_warps)
    if cus >= _H:
        assert picks == set(flydsl_kda_verify_with_f_b.num_warps)


# (CUs, ring, last batch on 8 waves): with 12 workgroups per sequence, the
# ring keeps 8 waves while workgroups <= CUs, snapshots while workgroups <=
# 5/4 of the CUs. 256 is a full MI355X, 32 one CPX partition of it.
@pytest.mark.parametrize(
    "cus, ring, last_8_wave_batch",
    [(256, True, 21), (256, False, 26), (32, True, 2), (32, False, 3)],
)
def test_kda_verify_num_warps_crossover(monkeypatch, cus, ring, last_8_wave_batch):
    monkeypatch.setattr(linear_attention_kernels, "get_num_sms", lambda: cus)
    picks = [
        linear_attention_kernels._kda_verify_num_warps(batch, ring)
        for batch in range(1, 257)
    ]
    assert picks == [8] * last_8_wave_batch + [4] * (256 - last_8_wave_batch)


def test_kda_verify_channel_major_conv_and_longer_scratch():
    inp = _make_inputs(4, 5, torch.bfloat16, False, steps=8, sglang_layout=False)
    _check(inp, 5)


def _pad_slots(tensor, pools):
    """Copy a dense slot-major ``tensor`` into every other slot of a zeroed pool,
    as a [slots, 2, ...] pool sliced on its second axis would give."""
    slot = tensor.stride(0)
    pool = torch.zeros(
        2 * tensor.shape[0] * slot, dtype=tensor.dtype, device=tensor.device
    )
    view = pool.as_strided(tensor.shape, (2 * slot, *tensor.stride()[1:]))
    view.copy_(tensor)
    pools.append((pool, tensor.shape[0]))
    return view


@pytest.mark.parametrize("ring", [False, True])
def test_kda_verify_accepts_padded_pool_slices(ring):
    """Caches and scratch whose slot stride exceeds one slot are accepted, and
    the padding between slots is never written."""
    inp = _make_inputs(4, 5, torch.float32, ring)
    pools = []
    inp.conv_state = _pad_slots(inp.conv_state, pools)
    inp.state = _pad_slots(inp.state, pools)
    inp.window = _pad_slots(inp.window, pools)
    if ring:
        inp.ring = tuple(_pad_slots(t, pools) for t in inp.ring)
    else:
        inp.inter_states = _pad_slots(inp.inter_states, pools)
    _check(inp, 5)
    for pool, slots in pools:
        assert torch.count_nonzero(pool.view(slots, 2, -1)[:, 1]) == 0


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


@pytest.mark.parametrize("ring", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_kda_verify_graph_replay(state_dtype, ring):
    """A captured verify matches eager bit for bit across replays: it reads the
    draft tokens and the committed caches written before each replay."""
    batch, num_tokens = 4, 6
    seed = _make_inputs(batch, num_tokens, state_dtype, ring)
    # Compile outside the capture on a throwaway copy, so capture records launches only.
    _run(_copy_inputs(seed), num_tokens)
    torch.cuda.synchronize()

    graph_inputs, eager_inputs = _copy_inputs(seed), _copy_inputs(seed)
    for name in _STRIDED_INPUTS:
        for inputs in (graph_inputs, eager_inputs):
            assert getattr(inputs, name).stride() == getattr(seed, name).stride(), name
    out = torch.empty(
        (1, batch * num_tokens, _H, _D), dtype=torch.bfloat16, device=_DEVICE
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(graph_inputs, num_tokens, out=out)

    for step in range(3):
        # New draft tokens and a newly committed state, as between two verifies.
        fresh = _make_inputs(batch, num_tokens, state_dtype, ring, seed=101 + step)
        for inputs in (graph_inputs, eager_inputs):
            for name in ("f_a", "x", "raw_beta", "output_gate", "conv_state", "state"):
                getattr(inputs, name).copy_(getattr(fresh, name))
        refs = _reference(graph_inputs, num_tokens)

        graph.replay()
        eager = _run(eager_inputs, num_tokens)
        torch.cuda.synchronize()
        assert torch.equal(out, eager), f"replay {step} output differs from eager"
        for graph_write, eager_write in zip(
            _verify_writes(graph_inputs), _verify_writes(eager_inputs)
        ):
            assert _bits_equal(
                graph_write, eager_write
            ), f"replay {step} scratch differs from eager"
        _assert_matches(graph_inputs, out, refs)


# Byte distance from slot 0 to slot 1 of every cache in the large-offset test.
# It passes 2**32, so a 32-bit byte offset would wrap into the zeroed gap below.
_HIGH_SLOT_BYTES = (1 << 32) + (1 << 24)


@pytest.mark.parametrize("ring", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_kda_verify_large_cache_slot_offsets(state_dtype, ring):
    """Slot and line 1 of every cache sit 4 GiB past slot 0, which only the
    kernel's 64-bit rebase reaches. All caches are strided two-slot views of
    one ~4.3 GB buffer, so the test stays small."""
    num_tokens = 5
    seed = _make_inputs(2, num_tokens, state_dtype, ring)
    caches = {"conv_state": seed.conv_state, "state": seed.state}
    caches["window"] = seed.window
    if ring:
        caches.update(zip(_RING_PARTS, seed.ring))
    else:
        caches["inter_states"] = seed.inter_states
    # Sequence 0 uses cache slot 1 and line 0; sequence 1 slot 0 and line 1.
    cache_indices, inter_indices = (1, 0), (0, 1)

    offsets, low_bytes = {}, 0
    for name, tensor in caches.items():
        offsets[name] = low_bytes
        slot_bytes = tensor[0].numel() * tensor.element_size()
        low_bytes += -(-slot_bytes // 256) * 256
    assert low_bytes < 1 << 24
    storage = torch.zeros(
        _HIGH_SLOT_BYTES + low_bytes, dtype=torch.uint8, device=_DEVICE
    )
    large = {}
    for name, tensor in caches.items():
        elem = tensor.element_size()
        view = storage.view(tensor.dtype).as_strided(
            (2, *tensor.shape[1:]),
            (_HIGH_SLOT_BYTES // elem, *tensor.stride()[1:]),
            offsets[name] // elem,
        )
        by_line = name in ("window", "inter_states")
        old = seed.inter_indices if by_line else seed.cache_indices
        new = inter_indices if by_line else cache_indices
        for b in range(2):
            view[new[b]].copy_(tensor[int(old[b])])
        large[name] = view
    actual = dataclasses.replace(
        seed,
        conv_state=large["conv_state"],
        state=large["state"],
        window=large["window"],
        inter_states=large.get("inter_states"),
        ring=tuple(large[name] for name in _RING_PARTS) if ring else None,
        cache_indices=torch.tensor(cache_indices, dtype=torch.int32, device=_DEVICE),
        inter_indices=torch.tensor(inter_indices, dtype=torch.int32, device=_DEVICE),
    )
    # The reference runs on dense copies of the same two slots.
    expected = _copy_inputs(actual)
    refs = _reference(expected, num_tokens)

    out = _run(actual, num_tokens)
    torch.cuda.synchronize()
    assert torch.equal(actual.conv_state, expected.conv_state)
    assert torch.equal(actual.state, expected.state)
    _assert_matches(actual, out, refs)
    # Nothing outside the addressed slots may be touched by wrapped offsets.
    for view in large.values():
        view.zero_()
    assert torch.count_nonzero(storage.view(torch.int64)) == 0


@pytest.mark.parametrize("ring", [False, True])
def test_kda_verify_slot_check_rejects_duplicate_live_indices(monkeypatch, ring):
    # AITER_KDA_DECODE_SLOT_CHECK is read at import; patch the flag it sets.
    monkeypatch.setattr(linear_attention_kernels, "_KDA_SLOT_CHECK", True)
    inp = _make_inputs(4, 5, torch.bfloat16, ring)

    def set_indices(cache, inter):
        inp.cache_indices.copy_(torch.tensor(cache, dtype=torch.int32))
        inp.inter_indices.copy_(torch.tensor(inter, dtype=torch.int32))

    # Padding may repeat either index, also a live slot or line paired with a
    # padding index in the other array.
    set_indices([-1, -1, 2, 2], [0, 0, -1, 3])
    _check(inp, 5)

    cases = [([1, 2, 3, 4], [5, 2, 2, -1], "intermediate_indices", 2)]
    if ring:
        # Only the ring is written per cache slot.
        cases.append(([3, -1, 0, 3], [0, 1, 2, 4], "cache_indices", 3))
    for cache, inter, name, duplicate in cases:
        set_indices(cache, inter)
        before = [t.clone() for t in _verify_writes(inp)]
        with pytest.raises(
            ValueError, match=rf"`{name}`.*duplicated slots: \[{duplicate}\]"
        ):
            _run(inp, 5)
        torch.cuda.synchronize()
        # The check runs before launch, so no scratch is written.
        for written, original in zip(_verify_writes(inp), before):
            assert torch.equal(written, original)


def test_kda_verify_snapshot_lets_live_sequences_share_a_cache_slot(monkeypatch):
    """Without a ring the caches are only read, so a repeated live cache slot
    is fine as long as every sequence has its own intermediate line."""
    monkeypatch.setattr(linear_attention_kernels, "_KDA_SLOT_CHECK", True)
    inp = _make_inputs(4, 5, torch.bfloat16, False)
    inp.cache_indices.copy_(torch.tensor([3, 3, 1, 3], dtype=torch.int32))
    inp.inter_indices.copy_(torch.tensor([0, 1, 2, 4], dtype=torch.int32))
    _check(inp, 5)


@pytest.mark.parametrize("ring", [False, True])
def test_kda_verify_slot_check_treats_the_padding_slot_as_padding(monkeypatch, ring):
    """Rows on AITER_KDA_DECODE_SLOT_CHECK_PADDING_SLOT are padding for both
    uniqueness checks, so they may share their slot and their scratch line."""
    monkeypatch.setattr(linear_attention_kernels, "_KDA_SLOT_CHECK", True)
    monkeypatch.setattr(linear_attention_kernels, "_KDA_SLOT_CHECK_PADDING_SLOT", None)
    inp = _make_inputs(4, 5, torch.bfloat16, ring)
    # Rows 0 and 1 are DP-attention padding on the reserved slot 0.
    inp.cache_indices.copy_(torch.tensor([0, 0, 1, 2], dtype=torch.int32))
    inp.inter_indices.copy_(torch.tensor([3, 3, 4, 5], dtype=torch.int32))
    with pytest.raises(ValueError, match="duplicated slots"):
        _run(inp, 5)

    monkeypatch.setattr(linear_attention_kernels, "_KDA_SLOT_CHECK_PADDING_SLOT", 0)
    ref_out = _reference(inp, 5)[0]
    out = _run(inp, 5)
    torch.cuda.synchronize()
    # The padding rows race on their shared scratch, so only check the live rows.
    live = slice(2 * 5, 4 * 5)
    _assert_close(ref_out[:, live], out[:, live], "output", **_OUT_TOL, max_bad=1e-3)


@pytest.mark.parametrize("ring", [False, True])
def test_kda_verify_slot_check_rejects_padding_on_a_live_line(monkeypatch, ring):
    """A row on the padding slot still writes its line, so it may not share a
    live sequence's line."""
    _reject_compile(monkeypatch)
    monkeypatch.setattr(linear_attention_kernels, "_KDA_SLOT_CHECK", True)
    monkeypatch.setattr(linear_attention_kernels, "_KDA_SLOT_CHECK_PADDING_SLOT", 0)
    inp = _make_inputs(3, 5, torch.bfloat16, ring)
    inp.cache_indices.copy_(torch.tensor([0, 1, 2], dtype=torch.int32))
    inp.inter_indices.copy_(torch.tensor([4, 4, 5], dtype=torch.int32))
    with pytest.raises(
        ValueError, match=r"line of a live sequence; shared lines: \[4\]"
    ):
        _run(inp, 5)


@pytest.mark.parametrize(
    "scratch, cache",
    [("inter_states", "state"), ("window", "conv_state")],
)
def test_kda_verify_rejects_scratch_on_a_cache(monkeypatch, scratch, cache):
    """Scratch over a cache would commit the verify early, so it is rejected."""
    _reject_compile(monkeypatch)
    inp = _make_inputs(2, 1, torch.bfloat16, False)
    # One token: the scratch's own layout, laid over the cache's slots.
    target = getattr(inp, scratch)
    setattr(inp, scratch, getattr(inp, cache).as_strided(target.shape, target.stride()))
    with pytest.raises(ValueError, match=f"must not overlap `{cache}`"):
        _run(inp, 1)


def _overlay(pool, offset, like):
    """A tensor of ``like``'s shape and dtype at a byte offset into ``pool``."""
    nbytes = like.numel() * like.element_size()
    return pool[offset : offset + nbytes].view(like.dtype).view(like.shape)


@pytest.mark.parametrize(
    "weight", ["f_b_weight", "conv_weight", "A_log", "dt_bias", "norm_weight"]
)
@pytest.mark.parametrize("written", ["window", "out"])
def test_kda_verify_rejects_writes_over_a_weight(monkeypatch, written, weight):
    """Every workgroup reads the weights, so a store into one is a race."""
    _reject_compile(monkeypatch)
    inp = _make_inputs(2, 1, torch.bfloat16, False)
    out = torch.empty(
        (1, 2, _H, _D), dtype=torch.bfloat16, device=_DEVICE
    )  # 2 x 1 token
    victim = getattr(inp, weight)
    target = inp.window if written == "window" else out
    # One allocation holds both, the written buffer over the weight's first bytes.
    size = 2 * max(victim.numel() * victim.element_size(), target.numel() * 4)
    pool = torch.zeros(size, dtype=torch.uint8, device=_DEVICE)
    setattr(inp, weight, _overlay(pool, 0, victim).copy_(victim))
    over = _overlay(pool, 0, target)
    if written == "window":
        inp.window = over
        out = None
    else:
        out = over
    with pytest.raises(ValueError, match=f"must not overlap `{weight}`"):
        _run(inp, 1, out=out)


@pytest.mark.parametrize("written, other", [(1, 0), (3, 2)], ids=["k-v", "beta-g"])
def test_kda_verify_rejects_ring_buffers_that_overlap(monkeypatch, written, other):
    """Two ring buffers over one allocation would overwrite each other's stores."""
    _reject_compile(monkeypatch)
    inp = _make_inputs(2, 5, torch.bfloat16, True)
    ring = list(inp.ring)
    target = ring[written]
    # Same dtype, so the target's own layout lands on the other's first bytes.
    ring[written] = ring[other].as_strided(target.shape, target.stride())
    inp.ring = tuple(ring)
    with pytest.raises(ValueError, match="must not overlap `replay_ring"):
        _run(inp, 5)


def _reject_compile(monkeypatch):
    def unexpected_compile(*args, **kwargs):
        pytest.fail("Invalid arguments must be rejected before kernel compilation.")

    monkeypatch.setattr(
        linear_attention_kernels, "create_kda_verify_kernel", unexpected_compile
    )


def test_kda_verify_rejects_bad_arguments(monkeypatch):
    _reject_compile(monkeypatch)
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
    # A wave count must be an int: 8.0 equals 8 but is rejected.
    for num_warps in (16, 8.0, True):
        with pytest.raises(ValueError, match="num_warps"):
            _run(inp, 5, num_warps=num_warps)


def test_kda_verify_rejects_conv_weight_without_unit_stride(monkeypatch):
    _reject_compile(monkeypatch)
    inp = _make_inputs(1, 2, torch.bfloat16, False)
    inp.conv_weight = torch.empty((_C, 8), dtype=torch.float32, device=_DEVICE)[:, ::2]
    with pytest.raises(
        ValueError, match="`conv_weight` must have an axis with stride 1"
    ):
        _run(inp, 2)


def _set_input(inp, out, name, tensor):
    """Replace ``name`` (an input field, a ring part or ``out``); return ``out``."""
    if name == "out":
        return tensor
    if name in _RING_PARTS:
        ring = list(inp.ring)
        ring[_RING_PARTS.index(name)] = tensor
        inp.ring = tuple(ring)
    else:
        setattr(inp, name, tensor)
    return out


def _get_input(inp, out, name):
    if name == "out":
        return out
    if name in _RING_PARTS:
        return inp.ring[_RING_PARTS.index(name)]
    return getattr(inp, name)


_WRITE_LABELS = {
    "window": "intermediate_conv_window",
    "inter_states": "intermediate_states",
    **{name: f"replay_ring.{name}" for name in _RING_PARTS},
    "out": "out",
}


# Inputs are [5 lines/slots, 5 steps, ...]; the ring holds L = 6 tokens.
@pytest.mark.parametrize(
    "name, strides",
    [
        # zeros(L, 1, 4608, 3).expand(L, 5, 4608, 3): every step aliases one.
        ("window", (_WINDOW_STEP, 0, 3, 1)),
        ("window", (0, _WINDOW_STEP, 1, _C)),
        ("window", (5 * _WINDOW_STEP - 1, _WINDOW_STEP, 3, 1)),
        ("window", (5 * _WINDOW_STEP, _WINDOW_STEP, 3, 0)),
        ("window", (5 * _WINDOW_STEP, _WINDOW_STEP, 1, 1)),
        ("window", (10 * _WINDOW_STEP, 2 * _WINDOW_STEP, 6, 2)),
        ("inter_states", (0, _HDD, _D * _D, _D, 1)),
        ("inter_states", (5 * _HDD - 1, _HDD, _D * _D, _D, 1)),
        ("inter_states", (5 * _HDD, 0, _D * _D, _D, 1)),
        ("inter_states", (5 * _HDD, _HDD, 0, _D, 1)),
        ("raw_v", (0, 6 * _D, _D, 1)),
        ("raw_k", (_H * 6 * _D - 1, 6 * _D, _D, 1)),
        ("g", (_H * 6 * _D, 0, _D, 1)),
        ("g", (_H * 6 * _D, _D, _D, 1)),
        ("beta", (0, 6, 1)),
        ("beta", (_H * 6 - 1, 6, 1)),
        ("beta", (_H * 6, 0, 1)),
        ("out", (0, _H * _D, _D - 1, 1)),
        ("out", (0, 0, _D, 1)),
    ],
)
def test_kda_verify_rejects_unsafe_write_layouts(monkeypatch, name, strides):
    _reject_compile(monkeypatch)
    with FakeTensorMode():
        inp = _make_inputs(2, 5, torch.bfloat16, name in _RING_PARTS)
        out = torch.empty((1, 10, _H, _D), dtype=torch.bfloat16, device=_DEVICE)
        tensor = _get_input(inp, out, name)
        view = torch.empty_strided(
            tensor.shape, strides, dtype=tensor.dtype, device=_DEVICE
        )
        out = _set_input(inp, out, name, view)
        label = re.escape(_WRITE_LABELS[name])
        with pytest.raises(ValueError, match=f"`{label}`.*non-overlapping"):
            _run(inp, 5, out=out)


@pytest.mark.parametrize(
    "name, dim",
    # Cache and scratch inner strides are fixed by the dense-slot checks, so
    # their byte span cannot overflow; unsafe layouts are covered above.
    [
        ("f_a", 0),
        ("f_b_weight", 0),
        ("x", 0),
        ("conv_weight", 0),
        ("conv_weight", 1),
        ("raw_beta", 1),
        ("output_gate", 0),
        ("output_gate", 1),
        ("out", 1),
        ("out", 2),
    ],
)
@pytest.mark.parametrize("stride", [2**30, 2**31])
def test_kda_verify_rejects_i32_address_overflow(monkeypatch, name, dim, stride):
    _reject_compile(monkeypatch)
    # Fake views exercise multi-GiB spans without allocating or accessing them.
    # 2**30 fits the launcher parameter but overflows the resulting byte span;
    # 2**31 also overflows the stride parameter itself.
    with FakeTensorMode():
        inp = _make_inputs(2, 5, torch.bfloat16, False)
        out = torch.empty((1, 10, _H, _D), dtype=torch.bfloat16, device=_DEVICE)
        tensor = _get_input(inp, out, name)
        strides = list(tensor.stride())
        strides[dim] = stride
        view = torch.empty_strided(
            tensor.shape, strides, dtype=tensor.dtype, device=_DEVICE
        )
        out = _set_input(inp, out, name, view)
        with pytest.raises(ValueError, match=f"`{name}`.*int32"):
            _run(inp, 5, out=out)


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


def _checked_close(ref, act, msg, atol, rtol, tol_err_ratio=1e-3):
    """checkAllclose that fails the sweep on a mismatch instead of only logging it."""
    err = checkAllclose(
        ref.to(dtypes.fp32),
        act.to(dtypes.fp32),
        atol=atol,
        rtol=rtol,
        tol_err_ratio=tol_err_ratio,
        msg=msg,
    )
    assert (
        err <= tol_err_ratio
    ), f"{msg}: {err:.4%} of elements outside atol={atol} rtol={rtol}"
    return err


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
        err = _checked_close(ref_out, out, f"{name}: KDA verify output ", **_OUT_TOL)
        assert torch.equal(inp.window, ref_window), f"{name}: conv windows differ"
        if ref_states is not None:
            err = max(
                err,
                _checked_close(
                    ref_states,
                    inp.inter_states,
                    f"{name}: KDA verify state snapshots ",
                    atol=1e-4,
                    rtol=1e-2,
                ),
            )
        if ref_ring is not None:
            for ref_part, part, label in zip(ref_ring, inp.ring, _RING_PARTS):
                err = max(
                    err,
                    _checked_close(
                        ref_part,
                        part,
                        f"{name}: KDA verify ring {label} ",
                        atol=1e-2,
                        rtol=1e-2,
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
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.fp32, dtypes.bf16],
        nargs="*",
        default=[dtypes.bf16],
        help="Recurrent-state (and snapshot) storage dtype; Q/K/V and the output "
        "are always BF16. Example: -d bf16 fp32",
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
            args.dtype, args.mode, args.tokens, args.batch
        )
    ]
    aiter.logger.info(
        "FlyDSL KDA verify summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
