# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for chunk_kimi_delta_attn with batch_invariant=True."""

import itertools

import pytest
import torch

from aiter.ops.triton._triton_kernels.kimi_delta_attn import chunk_fwd
from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn
from op_tests.triton_tests.utils.kda_ref import chunk_kda_ref

H, K = 4, 64
LOWER_BOUND = -5.0
PROBE_LEN = 150
BATCH_SIZES = [1, 2, 7, 16, 64, 255, 256, 1024]


def _make_seq(T, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape, dtype=torch.bfloat16):
        return torch.randn(shape, generator=gen, dtype=dtype, device="cuda")

    return {
        "q": rnd(T, H, K),
        "k": rnd(T, H, K),
        "v": rnd(T, H, K),
        "g": rnd(T, H, K),
        "beta": rnd(T, H, dtype=torch.float32),
        "h0": rnd(H, K, K, dtype=torch.float32) * 0.1,
    }


def _params():
    gen = torch.Generator(device="cuda").manual_seed(1234)
    A_log = torch.rand(H, generator=gen, dtype=torch.float32, device="cuda")
    dt_bias = (
        torch.randn(H * K, generator=gen, dtype=torch.float32, device="cuda") * 0.1
    )
    return A_log, dt_bias


def _run(seqs, safe_gate, batch_invariant=True, **extra):
    """Pack the sequences with cu_seqlens and return per-sequence (o, ht).

    batch_invariant=None does not pass the argument.
    """
    if batch_invariant is not None:
        extra["batch_invariant"] = batch_invariant
    A_log, dt_bias = _params()
    lens = [s["q"].shape[0] for s in seqs]
    cu = torch.tensor([0] + lens, device="cuda").cumsum(0).to(torch.int32)
    cat = {n: torch.cat([s[n] for s in seqs])[None] for n in ("q", "k", "v", "g")}
    o, ht = chunk_kimi_delta_attn(
        cat["q"],
        cat["k"],
        cat["v"],
        cat["g"],
        torch.cat([s["beta"] for s in seqs])[None],
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=torch.stack([s["h0"] for s in seqs]),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=safe_gate,
        lower_bound=LOWER_BOUND,
        cu_seqlens=cu,
        **extra,
    )
    bounds = cu.tolist()
    return [(o[0, a:b], ht[i]) for i, (a, b) in enumerate(itertools.pairwise(bounds))]


def _fillers(n, seed=7):
    lens = torch.randint(1, 131, (n,), generator=torch.Generator().manual_seed(seed))
    return [_make_seq(int(t), seed + 1 + i) for i, t in enumerate(lens)]


def _assert_equal(got, want, what):
    diff = (got.float() - want.float()).abs().max().item()
    assert torch.equal(got, want), f"{what}: max abs diff {diff}"


@pytest.mark.parametrize("safe_gate", [True, False])
def test_matches_reference(safe_gate):
    A_log, dt_bias = _params()
    seq = _make_seq(PROBE_LEN, 0)
    o, ht = _run([seq], safe_gate)[0]
    o_ref, ht_ref = chunk_kda_ref(
        seq["q"][None],
        seq["k"][None],
        seq["v"][None],
        seq["g"][None],
        seq["beta"][None],
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=seq["h0"][None],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=LOWER_BOUND,
    )
    torch.testing.assert_close(o.float(), o_ref[0].float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ht, ht_ref[0], atol=1e-2, rtol=1e-2)


def _with_probe(probe, others):
    pos = len(others) // 2
    return others[:pos] + [probe] + others[pos:], pos


@pytest.mark.parametrize("safe_gate", [True, False])
def test_batch_size_invariance(safe_gate):
    probe = _make_seq(PROBE_LEN, 0)
    fillers = _fillers(max(BATCH_SIZES) - 1)
    seqs, pos = _with_probe(probe, fillers)
    want = _run(seqs, safe_gate)
    want_probe, want_fillers = want.pop(pos), want
    for m in BATCH_SIZES:
        seqs, pos = _with_probe(probe, fillers[: m - 1])
        got = _run(seqs, safe_gate)
        got_probe = got.pop(pos)
        _assert_equal(got_probe[0], want_probe[0], f"probe o, M={m}")
        _assert_equal(got_probe[1], want_probe[1], f"probe ht, M={m}")
        for i, (o_i, ht_i) in enumerate(got):
            _assert_equal(o_i, want_fillers[i][0], f"filler {i} o, M={m}")
            _assert_equal(ht_i, want_fillers[i][1], f"filler {i} ht, M={m}")


@pytest.mark.parametrize("safe_gate", [True, False])
def test_order_invariance(safe_gate):
    seqs = [_make_seq(PROBE_LEN, 0)] + _fillers(63)
    want = _run(seqs, safe_gate)
    for seed in range(3):
        perm = torch.randperm(len(seqs), generator=torch.Generator().manual_seed(seed))
        got = _run([seqs[i] for i in perm.tolist()], safe_gate)
        for j, i in enumerate(perm.tolist()):
            _assert_equal(got[j][0], want[i][0], f"seq {i} o, perm {seed}")
            _assert_equal(got[j][1], want[i][1], f"seq {i} ht, perm {seed}")


@pytest.mark.parametrize("safe_gate", [True, False])
def test_prefix_invariance(safe_gate):
    seq = _make_seq(PROBE_LEN, 0)
    o_full = _run([seq], safe_gate)[0][0]
    for t in range(1, PROBE_LEN):
        part = {n: x[:t] for n, x in seq.items() if n != "h0"}
        part["h0"] = seq["h0"]
        o_t = _run([part], safe_gate)[0][0]
        _assert_equal(o_t, o_full[:t], f"prefix T={t}")


@pytest.mark.parametrize("safe_gate", [True, False])
def test_default_is_off(safe_gate):
    seqs = [_make_seq(PROBE_LEN, 0)] + _fillers(7)
    want = _run(seqs, safe_gate, batch_invariant=None)
    got = _run(seqs, safe_gate, batch_invariant=False)
    for i, (w, g) in enumerate(zip(want, got)):
        _assert_equal(g[0], w[0], f"seq {i} o")
        _assert_equal(g[1], w[1], f"seq {i} ht")


def test_rejects_runtime_autotune(monkeypatch):
    monkeypatch.setattr(chunk_fwd, "_AUTOTUNE_ON", True)
    with pytest.raises(ValueError, match="batch_invariant"):
        _run([_make_seq(16, 0)], safe_gate=True)


def test_rejects_flash_kda_only_args():
    seq = _make_seq(16, 0)
    out = torch.empty(1, 16, H, K, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="FlashKDA"):
        _run([seq], safe_gate=True, out=out)
