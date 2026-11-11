# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DeepSeek-V4.1 decode-layer contracts and their native reference, on CPU.

No GPU and no AITER build: the reference imports only torch and is loaded from
its file, the contract module is loaded beside the standalone ``aiter.plugin``
(``conftest.py``). The layer runs at reduced shapes -- the reference reads every
width off the tensors -- built from row-major weights and shuffled into the
contract's layouts with copies of aiter's own shuffles, so the inverse the
reference applies is checked against the forward it inverts.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO = Path(__file__).resolve().parents[1]
_OPS = _REPO / "aiter" / "ops"
OPS = ("deepseek_v41_decode_attn_pre", "deepseek_v41_decode_layer_post")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ref = _load("_dsv41_decode_ref", _OPS / "deepseek_v41_decode_ref.py")


def _contract_module():
    """``aiter.ops.deepseek_v41_decode`` -- the real one on a built AITER, else
    loaded beside the standalone ``aiter.plugin`` conftest registered."""
    if "aiter.ops" not in sys.modules:
        package = types.ModuleType("aiter.ops")
        package.__path__ = [str(_OPS)]
        sys.modules["aiter.ops"] = package
    name = "aiter.ops.deepseek_v41_decode"
    if name in sys.modules:
        return sys.modules[name]
    return _load(name, _OPS / "deepseek_v41_decode.py")


# ---------------------------------------------------------------------------
# the contract surface
# ---------------------------------------------------------------------------


def test_both_ops_are_routed_with_the_committed_signature():
    from test_plugin_contracts import routed_contracts

    found = routed_contracts(_REPO)
    baseline = json.loads((_REPO / "tests" / "plugin_contracts.json").read_text())
    for op in OPS:
        assert op in found, f"{op} is not routed"
        assert found[op]["version"] == 1
        assert baseline[op]["fingerprint"] == found[op]["fingerprint"]
        assert found[op]["source"] == "aiter/ops/deepseek_v41_decode.py"


def test_communication_and_step_arguments_are_keyword_only():
    module = _contract_module()
    for op in OPS:
        params = inspect.signature(getattr(module, op)).parameters
        for name in ("group", "stream", "layer_index", "eps", "hc_eps"):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, (op, name)
        assert params["group"].default is None and params["stream"].default is None
        assert params["layer_index"].default is inspect.Parameter.empty


def test_the_reference_takes_the_contract_signature_exactly():
    """The native path is called with every contract name; a drift between the
    two would surface only on the first real call."""
    module = _contract_module()
    for op, impl in zip(OPS, (ref.attn_pre, ref.layer_post)):
        contract = inspect.signature(getattr(module, op)).parameters
        native = inspect.signature(impl).parameters
        assert [(n, p.kind, p.default) for n, p in contract.items()] == [
            (n, p.kind, p.default) for n, p in native.items()
        ]


def test_the_ops_carry_the_route_marker_when_aiter_plugin_is_real():
    module = _contract_module()
    for op in OPS:
        marker = getattr(getattr(module, op), "__aiter_plugin_op__", None)
        assert marker is not None and marker.name == op and marker.version == 1


# ---------------------------------------------------------------------------
# aiter's shuffles (copied verbatim in effect from aiter/ops/shuffle.py) and
# their inverses
# ---------------------------------------------------------------------------


def shuffle_16x16(x):
    """``shuffle_weight(x, (16, 16))`` for 1-byte elements."""
    n, k = x.shape[-2:]
    v = x.view(-1, n // 16, 16, k // 32, 2, 16).permute(0, 1, 3, 4, 2, 5)
    return v.contiguous().view(*x.shape)


def shuffle_moe(x, gate_up):
    """``shuffle_weight(x, (16, 16), is_guinterleave=True, gate_up=gate_up)``."""
    e, n, kb = x.shape
    if gate_up:
        n //= 2
    n0, k0 = n // 16, kb // 64
    if gate_up:
        v = x.view(e, 2, n0, 16, k0, 4, 16).permute(0, 2, 1, 4, 5, 3, 6)
    else:
        v = x.view(e, n0, 16, k0, 4, 16).permute(0, 1, 3, 4, 2, 5)
    return v.contiguous().view(*x.shape)


def shuffle_moe_scale(src, experts, gate_up):
    """``shuffle_scale(src, experts, is_guinterleave=True, gate_up=gate_up)``."""
    rows, k = src.shape
    if k % 8:
        pad = torch.full((rows, 8 - k % 8), 0x7F, dtype=src.dtype)
        src = torch.cat([src, pad], dim=1)
        k = src.shape[1]
    n = rows // experts
    k1, n1 = k // 8, n // 32
    if gate_up:
        v = src.view(experts, 2, n1, 16, k1, 2, 4).permute(0, 2, 4, 6, 3, 5, 1)
    else:
        v = src.view(experts, n1, 2, 16, k1, 2, 4).permute(0, 1, 4, 6, 3, 5, 2)
    return v.contiguous().view(rows, k)


def _bytes(*shape, gen):
    return torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen)


def test_unshuffle_inverts_the_dense_preshuffle():
    gen = torch.Generator().manual_seed(0)
    w = _bytes(2, 64, 96, gen=gen)
    assert torch.equal(ref.unshuffle_16x16(shuffle_16x16(w)), w)
    assert not torch.equal(shuffle_16x16(w), w)


@pytest.mark.parametrize("gate_up", [True, False])
def test_unshuffle_inverts_the_moe_weight_preshuffle(gate_up):
    gen = torch.Generator().manual_seed(1)
    w = _bytes(3, 64, 128, gen=gen)
    assert torch.equal(ref.unshuffle_moe_weight(shuffle_moe(w, gate_up), gate_up), w)


@pytest.mark.parametrize("gate_up", [True, False])
def test_unshuffle_inverts_the_moe_scale_shuffle(gate_up):
    gen = torch.Generator().manual_seed(2)
    s = _bytes(3 * 64, 5, gen=gen)
    back = ref.unshuffle_moe_scale(shuffle_moe_scale(s, 3, gate_up), 3, gate_up)
    assert torch.equal(back[:, :, :5].reshape(3 * 64, 5), s)
    assert bool((back[:, :, 5:] == 0x7F).all())


def test_the_shuffle_copies_match_aiter_when_it_imports():
    """Guards the copies above against aiter's own (skipped without a build)."""
    try:
        from aiter.ops import shuffle as aiter_shuffle  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        pytest.skip("aiter.ops.shuffle needs a built AITER")
    gen = torch.Generator().manual_seed(3)
    w = _bytes(64, 96, gen=gen).view(torch.float8_e4m3fn)
    got = aiter_shuffle.shuffle_weight(w, layout=(16, 16)).view(torch.uint8)
    assert torch.equal(got, shuffle_16x16(w.view(torch.uint8)))
    e = _bytes(3, 64, 128, gen=gen)
    for gate_up in (True, False):
        got = aiter_shuffle.shuffle_weight(
            e, layout=(16, 16), is_guinterleave=True, gate_up=gate_up
        )
        assert torch.equal(got.view(torch.uint8), shuffle_moe(e, gate_up)), gate_up
        s = _bytes(3 * 64, 5, gen=gen)
        got = aiter_shuffle.shuffle_scale(s, 3, is_guinterleave=True, gate_up=gate_up)
        assert torch.equal(
            got.view(torch.uint8), shuffle_moe_scale(s, 3, gate_up)
        ), gate_up


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------


def test_ceil_rule_code_is_the_ceiling_power_of_two():
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(4, 64, generator=gen) * 3.0
    q, code = ref.mxfp8_ceil(x)
    amax = x.view(4, 2, 32).abs().amax(-1)
    expect = torch.ceil(torch.log2(amax / 448.0)).to(torch.int32) + 127
    assert torch.equal(code, expect)
    deq = ref.mx_dequant(q, code)
    assert bool(
        (
            deq.view(4, 2, 32).abs().amax(-1)
            <= 448.0 * torch.ldexp(torch.ones(()), code - 127)
        ).all()
    )
    assert torch.allclose(deq, x, rtol=0.07, atol=0.0)


def test_ceil_rule_floors_an_all_zero_group():
    q, code = ref.mxfp8_ceil(torch.zeros(1, 32))
    assert bool((q.float() == 0).all())
    assert int(code[0, 0]) == int(ref._ceil_code(torch.tensor([1e-10 / 448.0]))[0])


def test_round_rule_scale_leaves_headroom():
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(8, 64, generator=gen) * 100.0
    q, code = ref.mxfp8_round(x)
    assert bool(torch.isfinite(q.float()).all())
    # p2 carries once the mantissa reaches 0.75: |x| / scale stays below 1.75 x 256
    assert q.float().abs().max() <= 448.0
    assert torch.allclose(ref.mx_dequant(q, code), x, rtol=0.07, atol=1e-3)


def test_fp4_unpack_low_nibble_first():
    packed = torch.tensor([[0x21, 0xF9]], dtype=torch.uint8)
    assert ref.fp4_unpack(packed).tolist() == [[0.5, 1.0, -0.5, -6.0]]


def _seam_inputs(s=3, hc=4, h=64, seed=6):
    gen = torch.Generator().manual_seed(seed)
    mix = hc * (hc + 2)
    return dict(
        residual=torch.randn(s, hc, h, generator=gen).bfloat16(),
        pre_mix=torch.rand(s, hc, generator=gen),
        hc_fn=torch.randn(mix, hc * h, generator=gen) * 0.05,
        hc_scale=torch.rand(3, generator=gen) + 0.5,
        hc_base=torch.randn(mix, generator=gen) * 0.1,
        norm_weight=(torch.rand(h, generator=gen) + 0.5).bfloat16(),
    )


def test_seam_gates_are_in_range_and_comb_is_doubly_stochastic():
    a = _seam_inputs()
    rn, normed, pre, post, comb = ref.mhc_seam(**a, eps=1e-20, hc_eps=1e-6)
    assert torch.equal(rn, a["residual"])
    assert bool(((pre > 0) & (pre <= 1 + 1e-6)).all())
    assert bool(((post >= 0) & (post <= 2)).all())
    assert torch.allclose(comb.sum(-2), torch.ones(3, 4), atol=1e-4)
    assert torch.allclose(comb.sum(-1), torch.ones(3, 4), atol=1e-3)
    assert normed.dtype == torch.bfloat16 and normed.shape == (3, 64)


def test_seam_fold_is_post_times_output_plus_comb_mixed_streams():
    a = _seam_inputs()
    gen = torch.Generator().manual_seed(7)
    y = torch.randn(3, 64, generator=gen).bfloat16()
    post = torch.rand(3, 4, generator=gen)
    comb = torch.rand(3, 4, 4, generator=gen)
    rn, *_ = ref.mhc_seam(**a, eps=1e-20, hc_eps=1e-6, sublayer_output=y, post_mix=post,
                          comb_mix=comb)  # fmt: skip
    r = a["residual"].float()
    for t in range(3):
        for j in range(4):
            want = post[t, j] * y[t].float() + sum(
                comb[t, i, j] * r[t, i] for i in range(4)
            )
            assert torch.allclose(rn[t, j].float(), want.bfloat16().float(), atol=1e-2)


def test_inverse_rope_undoes_rope():
    gen = torch.Generator().manual_seed(8)
    x = torch.randn(3, 2, 128, generator=gen)
    angle = torch.rand(16, 32, generator=gen) * 6.28
    cos, sin = torch.cos(angle), torch.sin(angle)
    pos = torch.tensor([0, 5, 15])
    there = ref.rope_tail(x, cos, sin, pos)
    assert torch.equal(there[..., :64], x[..., :64])
    back = ref.rope_tail(there, cos, sin, pos, inverse=True)
    assert torch.allclose(back, x, atol=1e-5)


def test_sparse_attention_is_softmax_with_a_sink_in_the_denominator():
    gen = torch.Generator().manual_seed(9)
    q = torch.randn(2, 3, 64, generator=gen).bfloat16()
    cache = torch.randn(10, 64, generator=gen).bfloat16()
    sink = torch.randn(3, generator=gen)
    keys = [torch.tensor([1, 4, 7]), torch.tensor([], dtype=torch.long)]
    out = ref.sparse_attention(q, cache, keys, sink, 0.125, 64)
    kv = cache[keys[0]].float()
    logits = (q[0].float() @ kv.t()) * 0.125
    full = torch.cat([logits, sink[:, None]], dim=-1).softmax(-1)
    want = full[:, :3] @ kv
    assert torch.allclose(out[0].float(), want, atol=2e-2)
    assert bool((out[1] == 0).all())


# ---------------------------------------------------------------------------
# the layer at reduced shapes
# ---------------------------------------------------------------------------

S, HC, H, QR, D, HEADS, OGROUPS, ORANK = 3, 4, 256, 64, 128, 4, 2, 64
E, TOPK, INTER, SH_INTER, IH, ID = 8, 2, 256, 128, 4, 128
FP8 = torch.float8_e4m3fn


def _codes(*shape, gen, lo=120, hi=124):
    return torch.randint(lo, hi, shape, dtype=torch.uint8, generator=gen).view(
        torch.float8_e8m0fnu
    )


def _fp8_rowmajor(n, k, gen):
    return (torch.randn(n, k, generator=gen) * 4).to(FP8)


class Dense:
    """A row-major FP8 weight with 32x32 scales, and its contract layout."""

    def __init__(self, n, k, gen):
        self.w = _fp8_rowmajor(n, k, gen)
        self.s = _codes(n // 32, k // 32, gen=gen)

    def rows(self, lo, hi):
        out = Dense.__new__(Dense)
        out.w, out.s = (
            self.w[lo:hi].contiguous(),
            self.s[lo // 32 : hi // 32].contiguous(),
        )
        return out

    def cols(self, lo, hi):
        out = Dense.__new__(Dense)
        out.w = self.w[:, lo:hi].contiguous()
        out.s = self.s[:, lo // 32 : hi // 32].contiguous()
        return out

    def contract(self):
        return shuffle_16x16(self.w.view(torch.uint8)).view(FP8), self.s


class Experts:
    """Row-major MXFP4 experts (gate rows, up rows) and their contract layout."""

    def __init__(self, gen, inter=INTER):
        self.inter = inter
        self.w13 = _bytes(E, 2 * inter, H // 2, gen=gen) & 0x77  # |v| <= 6, no -0 noise
        self.s13 = torch.randint(122, 126, (E, 2 * inter, H // 32), dtype=torch.uint8,
                                 generator=gen)  # fmt: skip
        self.w2 = _bytes(E, H, inter // 2, gen=gen) & 0x77
        self.s2 = torch.randint(122, 126, (E, H, inter // 32), dtype=torch.uint8,
                                generator=gen)  # fmt: skip

    def shard(self, rank, ranks, pad_to=128):
        """This rank's slice of every expert's intermediate, padded with zeros."""
        n = self.inter // ranks
        p = -(-n // pad_to) * pad_to
        lo, hi = rank * n, (rank + 1) * n
        w13 = torch.zeros(E, 2 * p, H // 2, dtype=torch.uint8)
        s13 = torch.full((E, 2 * p, H // 32), 127, dtype=torch.uint8)
        w13[:, :n], w13[:, p : p + n] = (
            self.w13[:, lo:hi],
            self.w13[:, self.inter + lo : self.inter + hi],
        )
        s13[:, :n], s13[:, p : p + n] = (
            self.s13[:, lo:hi],
            self.s13[:, self.inter + lo : self.inter + hi],
        )
        w2 = torch.zeros(E, H, p // 2, dtype=torch.uint8)
        s2 = torch.full((E, H, p // 32), 127, dtype=torch.uint8)
        w2[:, :, : n // 2] = self.w2[:, :, lo // 2 : hi // 2]
        s2[:, :, : n // 32] = self.s2[:, :, lo // 32 : hi // 32]
        return (
            shuffle_moe(w13, True),
            shuffle_moe_scale(s13.reshape(E * 2 * p, -1), E, True),
            shuffle_moe(w2, False),
            shuffle_moe_scale(s2.reshape(E * H, -1), E, False),
        )  # fmt: skip


def _layer(seed=10):
    gen = torch.Generator().manual_seed(seed)
    mix = HC * (HC + 2)
    return types.SimpleNamespace(
        hc_attn=(torch.randn(mix, HC * H, generator=gen) * 0.02,
                 torch.rand(3, generator=gen) + 0.5, torch.randn(mix, generator=gen) * 0.1),
        hc_ffn=(torch.randn(mix, HC * H, generator=gen) * 0.02,
                torch.rand(3, generator=gen) + 0.5, torch.randn(mix, generator=gen) * 0.1),
        attn_norm=(torch.rand(H, generator=gen) + 0.5).bfloat16(),
        ffn_norm=(torch.rand(H, generator=gen) + 0.5).bfloat16(),
        wqkv=Dense(QR + D, H, gen), wqb=Dense(HEADS * D, QR, gen),
        iwqb=Dense(IH * ID, QR, gen),
        iproj=(torch.randn(IH, H, generator=gen) * 0.05).bfloat16(),
        qn=(torch.rand(QR, generator=gen) + 0.5).bfloat16(),
        kvn=(torch.rand(D, generator=gen) + 0.5).bfloat16(),
        woa=Dense(OGROUPS * ORANK, HEADS // OGROUPS * D, gen),
        wob=Dense(H, OGROUPS * ORANK, gen),
        sink=torch.randn(HEADS, generator=gen),
        gate=(torch.randn(E, H, generator=gen) * 0.05).bfloat16(),
        bias=torch.randn(E, generator=gen) * 0.01,
        experts=Experts(gen),
        sgu=Dense(2 * SH_INTER, H, gen), sw2=Dense(H, SH_INTER, gen),
        cos=torch.cos(torch.rand(64, 32, generator=gen) * 6).bfloat16(),
        sin=torch.sin(torch.rand(64, 32, generator=gen) * 6).bfloat16(),
    )  # fmt: skip


def _state(seed=11):
    gen = torch.Generator().manual_seed(seed)
    return types.SimpleNamespace(
        residual=torch.randn(S, HC, H, generator=gen).bfloat16(),
        pre=torch.rand(S, HC, generator=gen),
        pend=torch.randn(S, H, generator=gen).bfloat16(),
        post=torch.rand(S, HC, generator=gen),
        comb=torch.rand(S, HC, HC, generator=gen),
        positions=torch.tensor([3, 17, 40], dtype=torch.int64),
    )


def _run_attn_pre(
    lw, st, cache, slots, *, rank=0, ranks=1, index=True, fold=True, group=None
):
    heads = HEADS // ranks
    wqb = lw.wqb.rows(rank * heads * D, (rank + 1) * heads * D)
    out = types.SimpleNamespace(
        pre=torch.empty(S, HC), post=torch.empty(S, HC), comb=torch.empty(S, HC, HC),
        q=torch.empty(S, heads, D, dtype=torch.bfloat16),
        res=torch.empty(S, HC, H, dtype=torch.bfloat16) if fold else None,
        normed=torch.empty(S, H, dtype=torch.bfloat16),
        qr=torch.empty(S, QR, dtype=FP8),
        qrs=torch.empty(S, QR // 32, dtype=torch.float8_e8m0fnu),
        iq=torch.empty(S, IH, ID, dtype=FP8), iqs=torch.empty(S, IH), iw=torch.empty(S, IH),
        aux=torch.empty(S, H, dtype=torch.bfloat16),
    )  # fmt: skip
    idx = {}
    if index:
        iwqb, iwqb_s = lw.iwqb.contract()
        idx = dict(
            normed_out=out.normed, q_latent_out=out.qr, q_latent_scale_out=out.qrs,
            index_wq_b=iwqb, index_wq_b_scale=iwqb_s, index_weights_proj=lw.iproj,
            index_q_out=out.iq, index_q_scale_out=out.iqs, index_weights_out=out.iw,
        )  # fmt: skip
    fold_args = (
        dict(
            sublayer_output=st.pend,
            post_mix=st.post,
            comb_mix=st.comb,
            residual_out=out.res,
        )
        if fold
        else {}
    )
    ref.attn_pre(
        st.residual, st.pre, *lw.hc_attn, lw.attn_norm, *lw.wqkv.contract(), lw.qn, lw.kvn,
        *wqb.contract(), lw.cos, lw.sin, st.positions, cache, slots,
        out.pre, out.post, out.comb, out.q,
        layer_index=0, aux_out=out.aux, group=group, **fold_args, **idx,
    )  # fmt: skip
    return out


def test_attn_pre_writes_every_output_and_only_the_named_cache_rows():
    lw, st = _layer(), _state()
    cache = torch.zeros(16, D, dtype=torch.bfloat16)
    slots = torch.tensor([2, -1, 9], dtype=torch.int32)
    out = _run_attn_pre(lw, st, cache, slots)
    for name in ("pre", "post", "comb", "q", "res", "normed", "iqs", "iw", "aux"):
        assert bool(torch.isfinite(getattr(out, name).float()).all()), name
    assert bool(torch.isfinite(out.iq.float()).all())
    written = (cache != 0).any(-1)
    assert written.nonzero().flatten().tolist() == [2, 9]
    # the window row is the QAT of RoPE(kv): its groups' amax at most 448 x scale
    q8, code = ref.mxfp8_ceil(cache[2].float()[None])
    assert torch.equal(ref.mx_dequant(q8, code).bfloat16()[0], cache[2])
    # rows outside RoPE's 64 tail dims of q are those of the unrotated GEMM
    assert out.q.shape == (S, HEADS, D)


def test_attn_pre_mxfp8_cache_rows_are_values_then_codes():
    lw, st = _layer(), _state()
    bf = torch.zeros(8, D, dtype=torch.bfloat16)
    mx = torch.zeros(8, D + D // 32, dtype=torch.uint8)
    slots = torch.tensor([0, 1, 2], dtype=torch.int32)
    _run_attn_pre(lw, st, bf, slots, index=False)
    _run_attn_pre(lw, st, mx, slots, index=False)
    deq = ref.mx_dequant(mx[:3, :D].view(FP8), mx[:3, D:].to(torch.int32))
    assert torch.equal(deq.bfloat16(), bf[:3])


def test_attn_pre_without_fold_takes_the_residual_as_settled():
    lw, st = _layer(), _state()
    cache = torch.zeros(4, D, dtype=torch.bfloat16)
    slots = torch.full((S,), -1, dtype=torch.int32)
    out = _run_attn_pre(lw, st, cache, slots, fold=False, index=False)
    _, _, pre, _, _ = ref.mhc_seam(
        st.residual, st.pre, *lw.hc_attn, lw.attn_norm, 1e-20, 1e-6
    )
    assert torch.equal(out.pre, pre)
    assert bool((cache == 0).all())


def test_attn_pre_rejects_a_partial_fold_before_writing():
    lw, st = _layer(), _state()
    cache = torch.zeros(4, D, dtype=torch.bfloat16)
    pre_out = torch.zeros(S, HC)
    with pytest.raises(ref.ContractError, match="all or none"):
        ref.attn_pre(
            st.residual, st.pre, *lw.hc_attn, lw.attn_norm, *lw.wqkv.contract(), lw.qn,
            lw.kvn, *lw.wqb.contract(), lw.cos, lw.sin, st.positions, cache,
            torch.zeros(S, dtype=torch.int32), pre_out, torch.zeros(S, HC),
            torch.zeros(S, HC, HC), torch.zeros(S, HEADS, D, dtype=torch.bfloat16),
            layer_index=0, sublayer_output=st.pend,
        )  # fmt: skip
    assert bool((pre_out == 0).all())


def _post_args(lw, st, k1, cache, *, rank=0, ranks=1):
    heads, groups = HEADS // ranks, OGROUPS // ranks
    q = k1.q
    sink = lw.sink[rank * heads : (rank + 1) * heads].contiguous()
    woa = lw.woa.rows(rank * groups * ORANK, (rank + 1) * groups * ORANK)
    wob = lw.wob.cols(rank * groups * ORANK, (rank + 1) * groups * ORANK)
    sh = SH_INTER // ranks
    sgu = Dense.__new__(Dense)
    sgu.w = torch.cat(
        [
            lw.sgu.w[rank * sh : (rank + 1) * sh],
            lw.sgu.w[SH_INTER + rank * sh : SH_INTER + (rank + 1) * sh],
        ]
    )
    sgu.s = torch.cat([lw.sgu.s[rank * sh // 32 : (rank + 1) * sh // 32],
                       lw.sgu.s[(SH_INTER + rank * sh) // 32 : (SH_INTER + (rank + 1) * sh)
                                // 32]])  # fmt: skip
    sw2 = lw.sw2.cols(rank * sh, (rank + 1) * sh)
    topk_rows = torch.tensor([[5, 6, 0], [7, 0, 0], [1, 2, 3]], dtype=torch.int32)
    topk_count = torch.tensor([2, 1, 3], dtype=torch.int32)
    window_meta = torch.tensor(
        [[0, 2, 0, 8], [3, 4, 0, 8], [0, 0, 0, 8]], dtype=torch.int32
    )
    out = types.SimpleNamespace(
        res=torch.empty(S, HC, H, dtype=torch.bfloat16), pre=torch.empty(S, HC),
        post=torch.empty(S, HC), comb=torch.empty(S, HC, HC),
        ffn=torch.empty(S, H, dtype=torch.bfloat16),
        normed=torch.empty(S, H, dtype=torch.bfloat16),
    )  # fmt: skip
    args = (
        q, cache, topk_rows, topk_count, window_meta, sink, st.positions, lw.cos, lw.sin,
        *woa.contract(), *wob.contract(), k1.res, k1.pre, k1.post, k1.comb, *lw.hc_ffn,
        lw.ffn_norm, lw.gate, lw.bias, *lw.experts.shard(rank, ranks),
        *sgu.contract(), *sw2.contract(), out.res, out.pre, out.post, out.comb, out.ffn,
    )  # fmt: skip
    kwargs = dict(softmax_scale=D**-0.5, ring_slots=6, layer_index=0,
                  ffn_normed_out=out.normed, top_k=TOPK)  # fmt: skip
    return args, kwargs, out


def _cache():
    gen = torch.Generator().manual_seed(12)
    return torch.randn(16, D, generator=gen).bfloat16()


def test_layer_post_runs_and_writes_every_output():
    lw, st = _layer(), _state()
    cache = _cache()
    k1 = _run_attn_pre(lw, st, cache, torch.tensor([8, 9, 10], dtype=torch.int32))
    args, kwargs, out = _post_args(lw, st, k1, cache)
    ref.layer_post(*args, **kwargs)
    for name in ("res", "pre", "post", "comb", "ffn", "normed"):
        assert bool(torch.isfinite(getattr(out, name).float()).all()), name
    assert out.ffn.abs().sum() > 0


def test_padded_expert_intermediate_changes_nothing():
    """A shard's zero padding (TP4's 576 -> 640) contributes exactly zero."""
    lw = _layer()
    x = torch.randn(S, H, generator=torch.Generator().manual_seed(13)).bfloat16()
    shared = (*lw.sgu.contract(), *lw.sw2.contract())
    opts = (TOPK, 1.5, 10.0)
    plain = ref.moe_partial(x, lw.gate, lw.bias, *lw.experts.shard(0, 1), shared, opts)
    padded = ref.moe_partial(x, lw.gate, lw.bias, *lw.experts.shard(0, 1, pad_to=384),
                             shared, opts)  # fmt: skip
    assert torch.equal(plain, padded)


def test_layer_post_rejects_a_count_past_the_rows():
    lw, st = _layer(), _state()
    cache = _cache()
    k1 = _run_attn_pre(lw, st, cache, torch.tensor([8, 9, 10], dtype=torch.int32))
    args, kwargs, out = _post_args(lw, st, k1, cache)
    args = list(args)
    args[3] = torch.tensor([2, 4, 3], dtype=torch.int32)
    with pytest.raises(ref.ContractError, match="topk_count"):
        ref.layer_post(*args, **kwargs)


def _tp_worker(rank, ranks, port, results):
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=ranks)
    try:
        lw, st = _layer(), _state()
        cache = _cache()
        k1 = _run_attn_pre(lw, st, cache, torch.tensor([8, 9, 10], dtype=torch.int32),
                           rank=rank, ranks=ranks, group=dist.group.WORLD)  # fmt: skip
        args, kwargs, out = _post_args(lw, st, k1, cache, rank=rank, ranks=ranks)
        ref.layer_post(*args, **kwargs, group=dist.group.WORLD)
        results[rank] = (out.ffn.float(), out.res.float(), out.normed.float())
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not __import__("torch").distributed.is_available(), reason="torch.distributed"
)
def test_tp2_shards_reduce_to_the_tp1_layer():
    """Two gloo ranks, each with its TP shard of every weight, reach the TP 1
    result up to the rounding the split moves (per-rank bf16 partials)."""
    import socket

    import torch.multiprocessing as mp

    lw, st = _layer(), _state()
    cache = _cache()
    k1 = _run_attn_pre(lw, st, cache, torch.tensor([8, 9, 10], dtype=torch.int32))
    args, kwargs, one = _post_args(lw, st, k1, cache)
    ref.layer_post(*args, **kwargs)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    manager = mp.get_context("spawn").Manager()
    results = manager.dict()
    mp.start_processes(_tp_worker, args=(2, port, results), nprocs=2, join=True,
                       start_method="spawn")  # fmt: skip
    for rank in range(2):
        ffn, res, normed = results[rank]
        assert torch.allclose(res, one.res.float(), rtol=2e-2, atol=2e-2)
        assert torch.allclose(normed, one.normed.float(), rtol=3e-2, atol=3e-2)
        assert torch.allclose(
            ffn, one.ffn.float(), rtol=5e-2, atol=5e-2 * one.ffn.abs().max()
        )
    assert torch.equal(results[0][0], results[1][0])
