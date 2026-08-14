# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Non-interference tests for the Triton fused-paged-prefill fast path in
`mha_batch_prefill_func`.

The kernel has its own correctness tests under
`op_tests/triton_tests/attention/test_fused_paged_prefill.py`. This file tests the
*dispatch*: that adding the fast path changed nothing for any call it does not take.

`AITER_FUSED_PAGED_PREFILL=0` skips the fast path entirely and is read per call, so
each case runs both legs in one process against the same allocations. For every call
shape the predicate rejects, the two legs must be **bit-identical** -- not close, equal
-- and the Triton kernel must not be entered at all.
"""

import os

import pytest
import torch

from aiter.ops import mha
from aiter.ops.triton.attention import fused_paged_prefill as fpp

POOL_PAGES = 20000
ENV = "AITER_FUSED_PAGED_PREFILL"


@pytest.fixture(autouse=True)
def _restore_env():
    old = os.environ.get(ENV)
    yield
    if old is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = old


def make(
    bs=2,
    q_len=256,
    kv_len=1024,
    H=8,
    Hkv=1,
    D=256,
    dtype=torch.bfloat16,
    seed=0,
    device="cuda",
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    prefix = kv_len - q_len
    perm = torch.randperm(POOL_PAGES, generator=g)
    shared, tails = perm[:prefix], perm[prefix : prefix + bs * q_len]
    pages = torch.cat(
        [torch.cat([shared, tails[i * q_len : (i + 1) * q_len]]) for i in range(bs)]
    )

    def mk(*sh):
        return (torch.randn(*sh, generator=g) * 0.5).to(dtype).to(device)

    return {
        "q": mk(bs * q_len, H, D),
        "k": mk(POOL_PAGES, Hkv, D),
        "v": mk(POOL_PAGES, Hkv, D),
        "cu_seqlens_q": (torch.arange(bs + 1, dtype=torch.int32) * q_len).to(device),
        "kv_indptr": (torch.arange(bs + 1, dtype=torch.int32) * kv_len).to(device),
        "kv_page_indices": pages.to(torch.int32).to(device),
        "max_seqlen_q": q_len,
        "max_seqlen_k": kv_len,
    }


def run(inp, **kwargs):
    """Call the op, returning ('ok', outputs) or ('raised', 'ExcType: msg')."""
    kwargs.setdefault("causal", True)
    try:
        out = mha.mha_batch_prefill_func(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["cu_seqlens_q"],
            inp["kv_indptr"],
            inp["kv_page_indices"],
            inp["max_seqlen_q"],
            inp["max_seqlen_k"],
            **kwargs,
        )
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - the exception IS the observable behaviour
        return "raised", f"{type(e).__name__}: {e}"
    return "ok", tuple(out) if isinstance(out, tuple) else (out,)


def both_legs(inp, **kwargs):
    """Run with the fast path on and off, counting entries into the Triton kernel.

    The off leg runs twice so each case carries its own control for whether the
    CK-tile path is even reproducible for that call shape -- it is not always, and
    without the control a CK nondeterminism looks identical to interference from the
    patch. A warm-up call first keeps JIT/allocator effects out of both legs.
    """
    os.environ[ENV] = "0"
    run(inp, **kwargs)  # warm-up, discarded
    off1 = run(inp, **kwargs)
    off2 = run(inp, **kwargs)

    hits = {"n": 0}
    orig = fpp.mha_batch_prefill_func

    def counting(*a, **k):
        hits["n"] += 1
        return orig(*a, **k)

    fpp.mha_batch_prefill_func = counting
    try:
        os.environ[ENV] = "1"
        on = run(inp, **kwargs)
    finally:
        fpp.mha_batch_prefill_func = orig
    return on, off1, off2, hits["n"]


def _rms_rel(x, y):
    d = (x.float() - y.float()).pow(2).mean().sqrt()
    n = y.float().pow(2).mean().sqrt()
    return (d / n).item() if n > 0 else d.item()


def assert_no_interference(on, off1, off2, what):
    """The fast path must not change a call it does not take.

    Bit-identical is the assertion wherever CK-tile is itself reproducible. Where it
    is not, the most that can be asserted is that the on-leg sits inside CK-tile's
    own run-to-run spread, and the case says so rather than quietly weakening.
    """
    assert (
        on[0] == off1[0]
    ), f"{what}: outcome differs, on={on[0]} off={off1[0]}\n  on: {on[1]}\n off: {off1[1]}"
    if on[0] == "raised":
        assert (
            on[1] == off1[1]
        ), f"{what}: different exception\n  on: {on[1]}\n off: {off1[1]}"
        return "raised identically"

    a, b, c = on[1], off1[1], off2[1]
    assert len(a) == len(b), f"{what}: returned {len(a)} vs {len(b)} tensors"
    notes = []
    for i, (x, y, z) in enumerate(zip(a, b, c)):
        if x is None and y is None:
            continue
        assert (
            x.shape == y.shape and x.dtype == y.dtype
        ), f"{what}[{i}]: shape/dtype differ"
        ck_reproducible = torch.equal(y, z)
        if ck_reproducible:
            assert torch.equal(x, y), (
                f"{what}[{i}]: CK-tile is bit-reproducible for this call, but the fast "
                f"path being enabled changed the result -- max|delta|="
                f"{(x.float() - y.float()).abs().max().item():.3e}"
            )
            notes.append("bit-identical")
        else:
            ck_spread = _rms_rel(y, z)
            delta = _rms_rel(x, y)
            assert delta <= max(ck_spread * 4, 1e-6), (
                f"{what}[{i}]: differs from the CK-tile leg by rms {delta:.3e}, which "
                f"exceeds CK-tile's own run-to-run spread of {ck_spread:.3e}"
            )
            notes.append(
                f"CK-tile not reproducible here (spread {ck_spread:.2e}); "
                f"on-leg within it ({delta:.2e})"
            )
    return "; ".join(notes)


# --------------------------------------------------------------------------------
# Every call shape the predicate rejects. Each must be bit-identical with the fast
# path on and off, and must not enter the Triton kernel.
# --------------------------------------------------------------------------------


def _alibi(H=8):
    return torch.zeros(H, device="cuda", dtype=torch.float32)


def _scale():
    return torch.ones(1, device="cuda", dtype=torch.float32)


REJECTED = {
    # name: (kwargs for make(), kwargs for the op call)
    "non_causal": ({}, {"causal": False}),
    "logits_soft_cap": ({}, {"logits_soft_cap": 30.0}),
    "sliding_window": ({}, {"window_size": (1024, 0)}),
    "return_lse": ({}, {"return_lse": True}),
    "return_attn_probs": ({}, {"return_attn_probs": True}),
    "alibi_slopes": ({}, {"alibi_slopes": _alibi}),
    "sink_size": ({}, {"sink_size": 4}),
    "q_descale": ({}, {"q_descale": _scale}),
    "k_descale": ({}, {"k_descale": _scale}),
    "v_descale": ({}, {"v_descale": _scale}),
    "dropout": ({}, {"dropout_p": 0.1}),
    "seqlen_k": ({}, {"seqlen_k": 1024}),
    # tensor-shape rejections: the ones that would otherwise be silent
    "two_kv_heads": ({"Hkv": 2}, {}),
    "four_kv_heads": ({"Hkv": 4}, {}),
    "head_dim_128": ({"D": 128}, {}),
}


@pytest.mark.parametrize("name", list(REJECTED))
def test_rejected_calls_are_untouched(name):
    mk_kwargs, call_kwargs = REJECTED[name]
    inp = make(**mk_kwargs)
    # some values need a device tensor built after cuda is up
    call_kwargs = {k: (v() if callable(v) else v) for k, v in call_kwargs.items()}
    on, off1, off2, hits = both_legs(inp, **call_kwargs)
    assert hits == 0, f"{name}: Triton kernel entered {hits}x for a rejected call"
    print(f"\n  {name}: {assert_no_interference(on, off1, off2, name)}")


def test_non_unit_stride_q_falls_back():
    """A q whose head dim is not unit-stride must not be dispatched: the kernel adds
    offs_d to the base pointer without a stride."""
    inp = make()
    q = inp["q"]
    # [T, H, D] -> a view with a non-unit last-dim stride
    inp["q"] = q.transpose(1, 2).contiguous().transpose(1, 2)
    assert inp["q"].stride(-1) != 1
    on, off1, off2, hits = both_legs(inp)
    assert hits == 0, "non-unit-stride q was dispatched"
    assert_no_interference(on, off1, off2, "non_unit_stride_q")


def test_float32_falls_back():
    inp = make(dtype=torch.float32)
    on, off1, off2, hits = both_legs(inp)
    assert hits == 0, "fp32 was dispatched"
    assert_no_interference(on, off1, off2, "float32")


def test_kv_block_descale_falls_back():
    inp = make()
    ok, why = fpp.is_supported(
        inp["q"],
        inp["k"],
        inp["v"],
        kv_block_descale=torch.ones(POOL_PAGES, 1, 2, device="cuda"),
    )
    assert not ok and "kv_block_descale" in why


# --------------------------------------------------------------------------------
# Calls the predicate accepts
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("D", fpp.SUPPORTED_HEAD_DIMS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_accepted_calls_are_dispatched_and_agree(D, dtype):
    inp = make(D=D, dtype=dtype)
    on, off1, _off2, hits = both_legs(inp)
    assert hits == 1, f"D={D} {dtype}: expected 1 Triton entry, got {hits}"
    assert on[0] == "ok" and off1[0] == "ok", f"unexpected raise: {on[1]} / {off1[1]}"
    got, ref = on[1][0], off1[1][0]
    assert torch.isfinite(got).all()
    # different accumulation order than CK-tile; bf16 rounding floor is ~2e-3
    rel = _rms_rel(got, ref)
    assert rel < 2e-2, f"D={D} {dtype}: rms_rel vs CK-tile = {rel:.3e}"


def test_out_is_written_not_ignored():
    """`out=` is an aliasing contract in the CK path -- the caller may read its buffer
    rather than the return value, so the fast path has to write it."""
    inp = make()
    buf = torch.full_like(inp["q"], float("nan"))
    os.environ[ENV] = "1"
    ret = mha.mha_batch_prefill_func(
        inp["q"],
        inp["k"],
        inp["v"],
        inp["cu_seqlens_q"],
        inp["kv_indptr"],
        inp["kv_page_indices"],
        inp["max_seqlen_q"],
        inp["max_seqlen_k"],
        causal=True,
        out=buf,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(buf).all(), "out buffer was left untouched by the fast path"
    assert ret.data_ptr() == buf.data_ptr(), "return value does not alias out"

    os.environ[ENV] = "0"
    ref = torch.empty_like(inp["q"])
    mha.mha_batch_prefill_func(
        inp["q"],
        inp["k"],
        inp["v"],
        inp["cu_seqlens_q"],
        inp["kv_indptr"],
        inp["kv_page_indices"],
        inp["max_seqlen_q"],
        inp["max_seqlen_k"],
        causal=True,
        out=ref,
    )
    torch.cuda.synchronize()
    rel = _rms_rel(buf, ref)
    assert rel < 2e-2, f"out= content disagrees with CK-tile: {rel:.3e}"


def _fp32_reference(inp, sm_scale):
    q, k, v = inp["q"], inp["k"], inp["v"]
    cu_q, kv_indptr, pages = (
        inp["cu_seqlens_q"],
        inp["kv_indptr"],
        inp["kv_page_indices"],
    )
    out = torch.empty_like(q)
    for b in range(cu_q.shape[0] - 1):
        qs, qe = int(cu_q[b]), int(cu_q[b + 1])
        ks, ke = int(kv_indptr[b]), int(kv_indptr[b + 1])
        pg = pages[ks:ke].long()
        kk, vv = k[pg, 0].float(), v[pg, 0].float()
        delta = (ke - ks) - (qe - qs)
        pos = torch.arange(qe - qs, device=q.device)[:, None] + delta
        mask = torch.arange(ke - ks, device=q.device)[None, :] <= pos
        qi = q[qs:qe].float()
        for h in range(q.shape[1]):
            s = (qi[:, h] @ kk.T) * sm_scale
            p = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
            out[qs:qe, h] = (p @ vv).to(out.dtype)
    return out


@pytest.mark.parametrize("dist", ["normal", "spikes"])
def test_accuracy_is_no_worse_than_ck(dist):
    """Both paths measured against the same fp32 reference on the same inputs.

    The claim that matters for a drop-in replacement is not an absolute tolerance but
    that it is not WORSE than what it replaces.
    """
    inp = make(bs=2, q_len=512, kv_len=8192, D=256)
    if dist == "spikes":
        for t in ("k", "v"):
            n = inp[t].shape[0]
            inp[t][int(n * 0.9) :] *= 12.0
    sm_scale = 256**-0.5
    ref = _fp32_reference(inp, sm_scale)

    os.environ[ENV] = "1"
    got = run(inp, softmax_scale=sm_scale)
    os.environ[ENV] = "0"
    ck = run(inp, softmax_scale=sm_scale)
    assert got[0] == "ok" and ck[0] == "ok"

    e_triton, e_ck = _rms_rel(got[1][0], ref), _rms_rel(ck[1][0], ref)
    print(
        f"\n  {dist}: triton {e_triton:.3e} vs ck-tile {e_ck:.3e} "
        f"(ratio {e_triton / e_ck:.3f})"
    )
    assert e_triton < 2e-2, f"triton rms vs fp32 = {e_triton:.3e}"
    # The default scaling mode is exact in bf16, so the error should track CK-tile's
    # rather than merely stay inside a loose budget. This pins that: an earlier default
    # passed a 2e-2 budget while carrying 7x CK-tile's error on the spiked case.
    assert (
        e_triton <= e_ck * 1.25
    ), f"{dist}: triton error {e_triton:.3e} is more than 1.25x CK-tile's {e_ck:.3e}"


def test_kill_switch_prevents_dispatch():
    """A supported call must NOT reach the Triton kernel when the switch is off."""
    inp = make()
    hits = {"n": 0}
    orig = fpp.mha_batch_prefill_func

    def counting(*a, **k):
        hits["n"] += 1
        return orig(*a, **k)

    fpp.mha_batch_prefill_func = counting
    try:
        os.environ[ENV] = "0"
        assert run(inp)[0] == "ok"
        assert hits["n"] == 0, "kill switch did not prevent dispatch"
        os.environ[ENV] = "1"
        assert run(inp)[0] == "ok"
        assert hits["n"] == 1, "switch on: expected exactly one dispatch"
    finally:
        fpp.mha_batch_prefill_func = orig


def test_default_causal_false_is_not_dispatched():
    """The op's own default is causal=False, so a caller that does not ask for causal
    must keep the CK path."""
    inp = make()
    hits = {"n": 0}
    orig = fpp.mha_batch_prefill_func

    def counting(*a, **k):
        hits["n"] += 1
        return orig(*a, **k)

    fpp.mha_batch_prefill_func = counting
    try:
        os.environ[ENV] = "1"
        mha.mha_batch_prefill_func(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["cu_seqlens_q"],
            inp["kv_indptr"],
            inp["kv_page_indices"],
            inp["max_seqlen_q"],
            inp["max_seqlen_k"],
        )
        torch.cuda.synchronize()
    finally:
        fpp.mha_batch_prefill_func = orig
    assert hits["n"] == 0, "default (causal=False) call was dispatched"
