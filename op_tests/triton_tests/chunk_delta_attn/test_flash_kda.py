# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for the two-kernel FlashKDA path against the default chunk_delta_attn
pipeline.

Both compute the same forward, so the default pipeline is the reference. They
are not bit-identical: the fused path keeps the recurrent state in registers
across the whole sequence and inverts (I - L) with bf16 multiplicative doubling,
where the default path materializes the state per chunk and uses an exact
forward substitution. The gap is bf16-level, so the checks are on relative
error rather than allclose.
"""

import contextlib
import math

import pytest
import torch

from aiter.ops.triton._triton_kernels.chunk_delta_attn import chunk_delta_attn_fwd
from aiter.ops.triton._triton_kernels.chunk_delta_attn import chunk_fwd as _chunk_fwd
from aiter.ops.triton._triton_kernels.chunk_delta_attn import flash_kda as _flash_kda
from aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_kda import (
    FLASH_KDA_CHUNK,
    flash_kda_fwd,
    flash_kda_supported,
)
from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn
from op_tests.triton_tests.utils.kda_ref import chunk_kda_ref

device = "cuda"
dtype = torch.bfloat16

LOWER_BOUND = -5.0
K_DIM = 128


def make_inputs(B, T, H, seed=42):
    torch.manual_seed(seed)
    K = V = K_DIM
    scale = 1.0 / math.sqrt(K)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    beta = torch.randn(B, T, H, device=device, dtype=torch.float32)
    A_log = torch.randn(H, device=device, dtype=torch.float32).abs() * 0.5
    dt_bias = torch.randn(H * K, device=device, dtype=torch.float32) * 0.1
    return q, k, v, g, beta, A_log, dt_bias, scale


@contextlib.contextmanager
def _force_default_pipeline():
    """Pin the reference to the five-kernel path.

    ``chunk_delta_attn_fwd`` dispatches to ``flash_kda_fwd`` whenever
    CHUNK_DELTA_ATTN_USE_FLASH_KDA is set. Without this, running the suite
    with that variable exported makes every comparison here flash_kda
    against itself, which passes unconditionally.
    """
    saved = _chunk_fwd.CHUNK_DELTA_ATTN_USE_FLASH_KDA
    _chunk_fwd.CHUNK_DELTA_ATTN_USE_FLASH_KDA = False
    try:
        yield
    finally:
        _chunk_fwd.CHUNK_DELTA_ATTN_USE_FLASH_KDA = saved


def run_reference(q, k, v, g, beta, A_log, dt_bias, scale, **kw):
    """Default five-kernel pipeline."""
    with _force_default_pipeline():
        o, final_state, *_ = chunk_delta_attn_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            A_log=A_log,
            dt_bias=dt_bias,
            chunk_size=FLASH_KDA_CHUNK,
            safe_gate=True,
            lower_bound=kw.get("lower_bound", LOWER_BOUND),
            use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            disable_recompute=False,
            initial_state=kw.get("initial_state"),
            output_final_state=kw.get("output_final_state", False),
            state_v_first=kw.get("state_v_first", False),
            cu_seqlens=kw.get("cu_seqlens"),
        )
    return o, final_state


def run_flash(q, k, v, g, beta, A_log, dt_bias, scale, **kw):
    return flash_kda_fwd(
        chunks_per_seg=kw.get("chunks_per_seg"),
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        lower_bound=kw.get("lower_bound", LOWER_BOUND),
        initial_state=kw.get("initial_state"),
        output_final_state=kw.get("output_final_state", False),
        state_v_first=kw.get("state_v_first", False),
        cu_seqlens=kw.get("cu_seqlens"),
    )


def fp32_gold(q, k, v, g, beta, A_log, dt_bias, scale, **kw):
    """Token-at-a-time fp32 recurrence, driven by the same kw dict as the others.

    bf16 inputs upcast exactly, so this is the same call in fp32. A bf16 gold
    would round away the gap the ratio assertions below measure.
    """
    return chunk_kda_ref(
        q=q.float(),
        k=k.float(),
        v=v.float(),
        g=g.float(),
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=kw.get("lower_bound", LOWER_BOUND),
        initial_state=kw.get("initial_state"),
        output_final_state=kw.get("output_final_state", False),
        state_v_first=kw.get("state_v_first", False),
        cu_seqlens=kw.get("cu_seqlens"),
    )


def rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / (b.norm() + 1e-6)).item()


@pytest.mark.parametrize("B,T,H", [(1, 256, 4), (2, 512, 8), (1, 1024, 2)])
def test_batched_matches_reference(B, T, H):
    args = make_inputs(B, T, H)
    o_ref, _ = run_reference(*args)
    o_fkda, _ = run_flash(*args)
    assert rel_err(o_fkda, o_ref) < 2e-2


@pytest.mark.parametrize("T", [192, 200, 65, 320])
def test_tail_chunk(T):
    """T not a multiple of 64 exercises the partial trailing chunk."""
    args = make_inputs(1, T, 4)
    o_ref, _ = run_reference(*args)
    o_fkda, _ = run_flash(*args)
    assert rel_err(o_fkda, o_ref) < 2e-2


@pytest.mark.parametrize(
    "lens",
    [
        [128, 64],
        [100, 156, 33],  # all three end mid-chunk
        [64, 1, 191],  # single-token sequence next to a long one
        [512],
    ],
)
def test_varlen_matches_reference(lens):
    total = sum(lens)
    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0)), device=device, dtype=torch.long
    )
    args = make_inputs(1, total, 4)
    o_ref, _ = run_reference(*args, cu_seqlens=cu_seqlens)
    o_fkda, _ = run_flash(*args, cu_seqlens=cu_seqlens)
    assert rel_err(o_fkda, o_ref) < 2e-2


@pytest.mark.parametrize("state_v_first", [False, True])
def test_final_state(state_v_first):
    B, T, H = 2, 256, 4
    args = make_inputs(B, T, H)
    kw = {"output_final_state": True, "state_v_first": state_v_first}
    o_ref, ht_ref = run_reference(*args, **kw)
    o_fkda, ht_fkda = run_flash(*args, **kw)
    assert ht_fkda is not None and ht_fkda.shape == ht_ref.shape
    assert rel_err(o_fkda, o_ref) < 2e-2
    assert rel_err(ht_fkda, ht_ref) < 2e-2


@pytest.mark.parametrize("state_v_first", [False, True])
def test_initial_state_roundtrip(state_v_first):
    """Carrying a state in and back out, as chunked prefill does."""
    B, T, H = 2, 192, 4
    args = make_inputs(B, T, H)
    shape = (B, H, K_DIM, K_DIM)
    h0 = torch.randn(*shape, device=device, dtype=torch.float32) * 0.1
    kw = {
        "initial_state": h0,
        "output_final_state": True,
        "state_v_first": state_v_first,
    }
    o_ref, ht_ref = run_reference(*args, **kw)
    o_fkda, ht_fkda = run_flash(*args, **kw)
    assert rel_err(o_fkda, o_ref) < 2e-2
    assert rel_err(ht_fkda, ht_ref) < 2e-2


@pytest.mark.parametrize("state_v_first", [False, True])
def test_varlen_initial_state(state_v_first):
    """Packed sequences threading a state, in both state layouts.

    varlen and a V-first state are covered apart above but only together in the
    serving path, where prefill carries a state per sequence and hands it to
    decode in that layout.
    """
    lens = [128, 100, 64]
    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0)), device=device, dtype=torch.long
    )
    args = make_inputs(1, sum(lens), 4)
    h0 = (
        torch.randn(len(lens), 4, K_DIM, K_DIM, device=device, dtype=torch.float32)
        * 0.1
    )
    kw = {
        "cu_seqlens": cu_seqlens,
        "initial_state": h0,
        "output_final_state": True,
        "state_v_first": state_v_first,
    }
    o_ref, ht_ref = run_reference(*args, **kw)
    o_fkda, ht_fkda = run_flash(*args, **kw)
    assert rel_err(o_fkda, o_ref) < 2e-2
    assert rel_err(ht_fkda, ht_ref) < 2e-2


def test_chunked_prefill_equivalence():
    """Splitting a sequence and threading the state must match one shot."""
    B, T, H = 1, 256, 4
    q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(B, T, H)
    o_full, ht_full = run_flash(
        q, k, v, g, beta, A_log, dt_bias, scale, output_final_state=True
    )

    split = 128
    o1, ht1 = run_flash(
        q[:, :split],
        k[:, :split],
        v[:, :split],
        g[:, :split],
        beta[:, :split],
        A_log,
        dt_bias,
        scale,
        output_final_state=True,
    )
    o2, ht2 = run_flash(
        q[:, split:],
        k[:, split:],
        v[:, split:],
        g[:, split:],
        beta[:, split:],
        A_log,
        dt_bias,
        scale,
        initial_state=ht1,
        output_final_state=True,
    )
    o_split = torch.cat([o1, o2], dim=1)
    assert rel_err(o_split, o_full) < 2e-2
    assert rel_err(ht2, ht_full) < 2e-2


# The shapes above are far too short for the heuristic to segment, so the
# segmented path is pinned explicitly here. chunks_per_seg=1 is the extreme
# case: every chunk becomes its own segment, so the scan carries the entire
# recurrence and any error in the affine decomposition shows up immediately.
@pytest.mark.parametrize("chunks_per_seg", [1, 2, 3, 8])
def test_segmented_matches_reference(chunks_per_seg):
    args = make_inputs(2, 512, 4)
    o_ref, _ = run_reference(*args)
    o_seg, _ = run_flash(*args, chunks_per_seg=chunks_per_seg)
    assert rel_err(o_seg, o_ref) < 2e-2


@pytest.mark.parametrize("chunks_per_seg", [1, 3])
def test_segmented_states(chunks_per_seg):
    """Segmenting must not disturb either end of the state threading."""
    B, T, H = 2, 384, 4
    args = make_inputs(B, T, H)
    h0 = torch.randn(B, H, K_DIM, K_DIM, device=device, dtype=torch.float32) * 0.1
    kw = {"initial_state": h0, "output_final_state": True}
    o_ref, ht_ref = run_reference(*args, **kw)
    o_seg, ht_seg = run_flash(*args, **kw, chunks_per_seg=chunks_per_seg)
    assert rel_err(o_seg, o_ref) < 2e-2
    assert rel_err(ht_seg, ht_ref) < 2e-2


def test_segmented_varlen_uneven():
    """Sequences whose last segment is short, and one shorter than a segment."""
    lens = [500, 33, 257, 64]
    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0)), device=device, dtype=torch.long
    )
    args = make_inputs(1, sum(lens), 4)
    h0 = torch.randn(len(lens), 4, K_DIM, K_DIM, device=device, dtype=torch.float32)
    kw = {
        "cu_seqlens": cu_seqlens,
        "initial_state": h0 * 0.1,
        "output_final_state": True,
    }
    o_ref, ht_ref = run_reference(*args, **kw)
    o_seg, ht_seg = run_flash(*args, **kw, chunks_per_seg=3)
    assert rel_err(o_seg, o_ref) < 2e-2
    assert rel_err(ht_seg, ht_ref) < 2e-2


def test_segmented_agrees_with_unsegmented():
    """The segment count is a scheduling choice and must not change the result."""
    args = make_inputs(1, 1024, 4)
    o_plain, ht_plain = run_flash(*args, output_final_state=True, chunks_per_seg=0)
    for m in (2, 7, 16):
        o_seg, ht_seg = run_flash(*args, output_final_state=True, chunks_per_seg=m)
        assert rel_err(o_seg, o_plain) < 5e-3, m
        assert rel_err(ht_seg, ht_plain) < 5e-3, m


def test_tuner_keeps_the_two_schedules_apart():
    """A segmented run must not hand its config to a later unsegmented one.

    K2's grid is ``cdiv(W, BW) * num_segs * H``, so the wide BW that suits a
    segmented sweep leaves the unsegmented scan a quarter of the blocks: at
    H=12 that pick costs 2.3x. The two collide unless the segment count reaches
    the autotune key, and `cache_results` then persists whichever won.
    """
    kern = _flash_kda._flash_kda_segment_kernel
    args = make_inputs(1, 1024, 4)
    # An incoming state is what makes the two output passes otherwise identical:
    # without it the unsegmented one passes h_in=None and the key picks up the
    # difference through the dtypes it appends, hiding the collision.
    kw = {"initial_state": torch.zeros(1, 4, K_DIM, K_DIM, device=device)}
    kern.cache.clear()
    run_flash(*args, chunks_per_seg=4, **kw)
    segmented_keys = set(kern.cache)
    run_flash(*args, chunks_per_seg=0, **kw)
    assert set(kern.cache) - segmented_keys, "unsegmented reused a segmented config"


# A weak gate is the only setting that exposes the intra-chunk inverse. At the
# Kimi default lower_bound=-5 the decay damps L to almost nothing, so even an
# outright wrong inverse stays inside bf16 noise; these three catch it.
@pytest.mark.parametrize("lower_bound", [-1.0, -0.1, -0.01])
@pytest.mark.parametrize("chunks_per_seg", [0, 3])
def test_weak_gate(lower_bound, chunks_per_seg):
    args = make_inputs(1, 512, 4)
    o_ref, _ = run_reference(*args, lower_bound=lower_bound)
    o_fkda, _ = run_flash(*args, lower_bound=lower_bound, chunks_per_seg=chunks_per_seg)
    assert rel_err(o_fkda, o_ref) < 2e-2


# Gaussian keys in 128 dimensions are near-orthogonal, so L's off-diagonal
# entries sit at ~0.02 and every inverse of it looks accurate -- which is the
# regime every other case here feeds. Repeated tokens in real text correlate the
# keys instead, and with beta near 1 and a gate that barely decays L's entries
# approach 1. The powers the inverse is built from then run to 5e7 across a
# 32-wide chunk before cancelling back to a result bounded by 1, which no
# storage precision survives: this case reached the model as 0.2% on gsm8k while
# the rest of this file passed.
@pytest.mark.parametrize("rho", [0.9, 1.0])
def test_correlated_keys_weak_gate(rho):
    q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(1, 512, 4)
    # rho=1 is a token repeated verbatim for the whole sequence.
    k = (rho * k[:, :1] + (1.0 - rho) * k).to(dtype)
    beta = torch.full_like(beta, 4.0)  # sigmoid(4) = 0.98, so L is barely shrunk
    args = (q, k, v, g, beta, A_log, dt_bias, scale)
    kw = {"lower_bound": -0.01}
    gold, _ = fp32_gold(*args, **kw)
    e_fkda = rel_err(run_flash(*args, **kw)[0], gold)
    e_ref = rel_err(run_reference(*args, **kw)[0], gold)
    assert e_fkda < 5e-2, f"flash_kda {e_fkda:.2e} against the fp32 recurrence"
    assert e_fkda < 1.3 * e_ref, f"flash_kda {e_fkda:.2e} vs default {e_ref:.2e}"


@pytest.mark.parametrize("lower_bound", [-5.0, -1.0, -0.01])
def test_matches_fp32_recurrence(lower_bound):
    """Against the token-at-a-time fp32 recurrence both paths blockify.

    The default pipeline is the reference everywhere else here, which cannot
    show the two paths drifting the same way; the absolute bound catches that.
    The second assertion keeps a merely noisier fused path from passing it, and
    covers a gap in the default path's own reference check, which pins
    chunk_size=64 -- a size flash_kda rejects.
    """
    args = make_inputs(1, 512, 4)
    gold, _ = fp32_gold(*args, lower_bound=lower_bound)
    e_fkda = rel_err(run_flash(*args, lower_bound=lower_bound)[0], gold)
    e_ref = rel_err(run_reference(*args, lower_bound=lower_bound)[0], gold)
    assert e_fkda < 2e-2, f"flash_kda {e_fkda:.2e} against the fp32 recurrence"
    assert e_fkda < 1.3 * e_ref, f"flash_kda {e_fkda:.2e} vs default {e_ref:.2e}"


@pytest.mark.parametrize("lower_bound", [-5.0, -1.0, -0.01])
def test_final_state_matches_fp32_recurrence(lower_bound):
    """The same absolute bound as above, on the state rather than the output.

    Decode continues the recurrence from this tensor, so a state that drifts the
    same way in both chunked paths passes every default-pipeline comparison here
    and still ruins generation. varlen with a V-first state is what the serving
    path asks for, and it is the combination the checks above split apart.
    """
    lens = [200, 100, 156]
    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0)), device=device, dtype=torch.long
    )
    args = make_inputs(1, sum(lens), 4)
    kw = {
        "cu_seqlens": cu_seqlens,
        "output_final_state": True,
        "state_v_first": True,
        "lower_bound": lower_bound,
    }
    _, gold = fp32_gold(*args, **kw)
    e_fkda = rel_err(run_flash(*args, **kw)[1], gold)
    e_ref = rel_err(run_reference(*args, **kw)[1], gold)
    assert e_fkda < 2e-2, f"flash_kda state {e_fkda:.2e} against the fp32 recurrence"
    assert e_fkda < 1.3 * e_ref, f"flash_kda state {e_fkda:.2e} vs default {e_ref:.2e}"


def test_supported_rejects_unsupported():
    q = torch.randn(1, 64, 4, 64, device=device, dtype=dtype)
    v = torch.randn(1, 64, 4, 64, device=device, dtype=dtype)
    A_log = torch.randn(4, device=device, dtype=torch.float32)
    base = {
        "chunk_size": FLASH_KDA_CHUNK,
        "safe_gate": True,
        "use_gate_in_kernel": True,
        "use_qk_l2norm_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "lower_bound": LOWER_BOUND,
        "A_log": A_log,
    }
    # K = V = 64 is not supported.
    assert not flash_kda_supported(q=q, v=v, **base)

    q128 = torch.randn(1, 64, 4, 128, device=device, dtype=dtype)
    v128 = torch.randn(1, 64, 4, 128, device=device, dtype=dtype)
    assert flash_kda_supported(q=q128, v=v128, **base)
    assert not flash_kda_supported(q=q128, v=v128, **{**base, "chunk_size": 64})
    assert not flash_kda_supported(q=q128, v=v128, **{**base, "lower_bound": None})
    assert not flash_kda_supported(q=q128, v=v128, **{**base, "safe_gate": False})
    assert not flash_kda_supported(
        q=q128, v=v128, **{**base, "use_gate_in_kernel": False}
    )
    # GVA (HV != H) is not supported.
    v_gva = torch.randn(1, 64, 8, 128, device=device, dtype=dtype)
    assert not flash_kda_supported(q=q128, v=v_gva, **base)


def test_public_wrapper_routes_to_flash_kda(monkeypatch):
    """``chunk_kimi_delta_attn`` must actually reach the fast path.

    Every other test here drives the internal functions, so nothing would catch
    the public wrapper pinning an argument that disqualifies the dispatch: the
    fast path would silently stop being used and the suite would stay green.
    """
    q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(1, 512, 4)
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": scale,
        "chunk_size": FLASH_KDA_CHUNK,
        "safe_gate": True,
        "lower_bound": LOWER_BOUND,
        "use_gate_in_kernel": True,
        "use_qk_l2norm_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "output_final_state": True,
    }

    calls = []
    real = _chunk_fwd.flash_kda_fwd

    def counting(**kw):
        calls.append(1)
        return real(**kw)

    monkeypatch.setattr(_chunk_fwd, "flash_kda_fwd", counting)

    monkeypatch.setattr(_chunk_fwd, "CHUNK_DELTA_ATTN_USE_FLASH_KDA", False)
    o_ref, ht_ref = chunk_kimi_delta_attn(**kwargs)
    assert not calls, "the flag is off, the default pipeline should have served this"

    monkeypatch.setattr(_chunk_fwd, "CHUNK_DELTA_ATTN_USE_FLASH_KDA", True)
    o_fkda, ht_fkda = chunk_kimi_delta_attn(**kwargs)
    assert len(calls) == 1, "the wrapper never reached flash_kda_fwd"

    chunk_kimi_delta_attn(**{**kwargs, "safe_gate": False})
    assert len(calls) == 1, "safe_gate=False must stay on the default intra kernel"

    assert rel_err(o_fkda, o_ref) < 2e-2
    assert rel_err(ht_fkda, ht_ref) < 2e-2


def test_unset_chunk_size_follows_the_dispatch(monkeypatch):
    """``chunk_size=None`` has to resolve the way the dispatch goes.

    Picking 32 without reaching flash_kda would leave the call on the default
    pipeline at 32, which is the slowest of the three configurations rather
    than the fastest, and nothing about the result would reveal it.
    """
    q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(1, 512, 4)
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": scale,
        "safe_gate": True,
        "lower_bound": LOWER_BOUND,
        "use_gate_in_kernel": True,
        "use_qk_l2norm_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
    }

    calls = []
    real = _chunk_fwd.flash_kda_fwd

    def counting(**kw):
        calls.append(1)
        return real(**kw)

    monkeypatch.setattr(_chunk_fwd, "flash_kda_fwd", counting)
    monkeypatch.setattr(_chunk_fwd, "CHUNK_DELTA_ATTN_USE_FLASH_KDA", True)

    chunk_kimi_delta_attn(chunk_size=None, **kwargs)
    assert len(calls) == 1, "an eligible call should have resolved to FLASH_KDA_CHUNK"

    # Ineligible, so it must fall back to 64 rather than to FLASH_KDA_CHUNK.
    # Same pipeline and same chunk size means the two agree bit for bit.
    o_auto, _ = chunk_kimi_delta_attn(chunk_size=None, **{**kwargs, "safe_gate": False})
    o_64, _ = chunk_kimi_delta_attn(chunk_size=64, **{**kwargs, "safe_gate": False})
    assert len(calls) == 1
    assert torch.equal(o_auto, o_64), "an ineligible call should have resolved to 64"
