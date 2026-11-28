# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass, fields

import pytest
import torch

from aiter.ops.triton.attention.mha_fp8 import mha_fwd, mha_varlen_fwd
from aiter.ops.triton.utils._triton.arch_info import get_arch

arch = get_arch()


# ---------------
# reference
# ---------------


def mha_torch(q, k, v, sm_scale, causal, p_scale=1.0):
    # GQA: broadcast each K/V head to the group of query heads that shares it
    group_size = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)
    qk = q @ k.transpose(-2, -1)
    qk = qk * sm_scale
    if causal:
        # bottom-right aligned: query i attends to keys j <= i + (SK - SQ)
        sq, sk = q.shape[2], k.shape[2]
        M = torch.tril(torch.ones((sq, sk), device=q.device), diagonal=sk - sq)
        qk[:, :, M == 0] = float("-inf")
    lse = torch.logsumexp(qk, -1)
    m = qk.amax(-1, keepdim=True).nan_to_num(neginf=0.0)
    p = torch.exp(qk - m)
    l = p.sum(-1, keepdim=True).clamp_min(1e-30)
    p_q = (p * p_scale).to(torch.float8_e4m3fn).to(torch.float32) / p_scale
    o = (p_q @ v) / l
    return o, lse


# ---------------
# initialize data
# ---------------


def quant_fp8(x):
    """Per-tensor fp8 e4m3 quantization; returns ``(fp8 tensor, fp32 scale)``."""
    scale = x.abs().amax().to(torch.float32) / 448.0
    return (x / scale).to(torch.float8_e4m3fn), scale


def init_qkv(b, hq, hk, sq, sk, d, device="cuda"):
    """fp8 q/k/v in the requested layout, plus their per-tensor scales."""
    q_shape = (b, hq, sq, d)
    kv_shape = (b, hk, sk, d)
    q, q_scale = quant_fp8(
        torch.randn(q_shape, device=device, dtype=torch.float32) * 0.5
    )
    k, k_scale = quant_fp8(
        torch.randn(kv_shape, device=device, dtype=torch.float32) * 0.5
    )
    v, v_scale = quant_fp8(
        torch.randn(kv_shape, device=device, dtype=torch.float32) * 0.5
    )
    return q, k, v, q_scale, k_scale, v_scale


# ---------------
# unit tests
# ---------------


@dataclass
class Case:
    b: int
    hq: int
    hk: int
    sq: int
    sk: int
    d: int


@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case))
        for case in [
            # one K/V tile, one M tile: the shortest path through the pipeline
            Case(1, 2, 2, 64, 64, 128),
            Case(1, 1, 1, 128, 128, 128),
            Case(1, 8, 8, 256, 256, 128),
            Case(2, 8, 8, 512, 512, 128),
            Case(2, 8, 8, 512, 512, 64),
            Case(4, 32, 32, 1024, 1024, 128),
            Case(2, 16, 16, 2048, 2048, 128),
            Case(1, 8, 8, 4096, 4096, 128),
            Case(3, 5, 5, 1024, 1024, 128),
            # M not a multiple of BLOCK_M: the last program writes a partial tile
            Case(2, 4, 4, 600, 600, 128),
            # HEAD_DIM != HEAD_DIM_POW2: the descriptors pad the head dim to 128
            Case(2, 8, 8, 512, 512, 96),
            # GQA: several query heads share one K/V head
            Case(2, 8, 2, 512, 512, 128),
            Case(4, 32, 8, 1024, 1024, 128),
            Case(2, 16, 4, 1024, 1024, 64),
            Case(1, 48, 8, 2048, 2048, 128),
            # seqlen_q < seqlen_k
            Case(2, 8, 2, 512, 1024, 128),
            Case(2, 4, 4, 600, 700, 128),
            # seqlen_q > seqlen_k
            Case(1, 8, 8, 2048, 512, 128),
        ]
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_mha_fp8(b, hq, hk, sq, sk, d, causal, device="cuda"):
    if get_arch() not in ("gfx1250", "gfx1250-strict"):
        pytest.skip("Only supported on gfx1250")
    if causal and sq > sk:
        pytest.skip("Causal requires seqlen_k >= seqlen_q")
    torch.manual_seed(0)

    q_tri, k_tri, v_tri, q_scale, k_scale, v_scale = init_qkv(
        b, hq, hk, sq, sk, d, device=device
    )
    q_ref = q_tri.to(torch.float32) * q_scale
    k_ref = k_tri.to(torch.float32) * k_scale
    v_ref = v_tri.to(torch.float32) * v_scale
    sm_scale = 1.0 / (d**0.5)

    o_ref, _ = mha_torch(q_ref, k_ref, v_ref, sm_scale, causal=causal)

    out = torch.empty(q_tri.shape, device=device, dtype=torch.bfloat16)
    o_tri = mha_fwd(
        q_tri,
        k_tri,
        v_tri,
        q_scale,
        k_scale,
        v_scale,
        o=out,
        sm_scale=sm_scale,
        causal=causal,
    )
    o_tri = o_tri.float()

    torch.testing.assert_close(o_tri, o_ref, atol=0.02, rtol=0.02)


VARLEN_P_SCALE = {"gfx950": 448.0, "gfx1250": 1.0}
VARLEN_ARCHS = tuple(VARLEN_P_SCALE)


def generate_mha_varlen_fp8_inputs(
    seqlens_q, seqlens_k, heads, pad=0, device="cuda", d_pe=64
):
    """Packed fp8 q/k/v (MLA prefill dims), cu_seqlens, per-tensor scales, and out/lse
    buffers with ``pad`` extra rows prefilled with 7.0."""
    D_NOPE, D_PE, D_V = 128, d_pe, 128
    cu_q = torch.tensor([0] + seqlens_q, device=device).cumsum(0).to(torch.int32)
    cu_k = torch.tensor([0] + seqlens_k, device=device).cumsum(0).to(torch.int32)
    tq, tk = sum(seqlens_q), sum(seqlens_k)
    q, q_scale = quant_fp8(
        torch.randn(
            (tq + pad, heads, D_NOPE + D_PE), dtype=torch.float32, device=device
        )
        * 0.5
    )
    k, k_scale = quant_fp8(
        torch.randn((tk, heads, D_NOPE + D_PE), dtype=torch.float32, device=device)
        * 0.5
    )
    v, v_scale = quant_fp8(
        torch.randn((tk, heads, D_V), dtype=torch.float32, device=device) * 0.5
    )
    q_scale, k_scale, v_scale = (s.reshape(1) for s in (q_scale, k_scale, v_scale))
    out = torch.full((tq + pad, heads, D_V), 7.0, dtype=torch.bfloat16, device=device)
    lse = torch.full((heads, tq + pad), 7.0, dtype=torch.float32, device=device)
    return q, k, v, cu_q, cu_k, q_scale, k_scale, v_scale, out, lse


@pytest.mark.parametrize(
    "seqlens_q, seqlens_k",
    [
        ([128], [128]),
        # more queries than keys: the first causal rows see no key
        ([128], [64]),
        ([600], [600]),
        ([1024, 77, 300], [1024, 77, 300]),
        ([256, 100], [2048, 900]),
        # short queries
        ([64], [8192]),
        ([16, 130], [4096, 6000]),
    ],
)
@pytest.mark.parametrize("heads", [8, 12, 16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pad", [0, 50])
@pytest.mark.parametrize("d_pe", [64, 0])
def test_mha_varlen_fp8(seqlens_q, seqlens_k, heads, causal, pad, d_pe, device="cuda"):
    if get_arch() not in VARLEN_ARCHS:
        pytest.skip("Only supported on gfx950 and gfx1250")
    if get_arch() == "gfx1250" and d_pe != 64:
        pytest.skip("gfx1250 needs 64 rope dims")
    torch.manual_seed(0)
    q, k, v, cu_q, cu_k, q_scale, k_scale, v_scale, out, lse = (
        generate_mha_varlen_fp8_inputs(seqlens_q, seqlens_k, heads, pad, device, d_pe)
    )
    tq = sum(seqlens_q)
    sm_scale = q.shape[-1] ** -0.5

    mha_varlen_fwd(
        q, k, v, cu_q, cu_k, max(seqlens_q), max(seqlens_k),
        q_scale, k_scale, v_scale,
        softmax_scale=sm_scale, causal=causal, return_lse=True, out=out, lse=lse,
    )  # fmt: skip

    qf, kf, vf = q.float() * q_scale, k.float() * k_scale, v.float() * v_scale
    for i in range(len(seqlens_q)):
        qs, ks = slice(cu_q[i], cu_q[i + 1]), slice(cu_k[i], cu_k[i + 1])
        o_ref, lse_ref = mha_torch(
            *(t.transpose(0, 1)[None] for t in (qf[qs], kf[ks], vf[ks])),
            sm_scale,
            causal,
            p_scale=VARLEN_P_SCALE.get(get_arch(), 1.0),
        )
        torch.testing.assert_close(
            out[qs].float(), o_ref[0].transpose(0, 1), atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(lse[:, qs], lse_ref[0], atol=1e-2, rtol=1e-3)
    assert (out[tq:] == 7.0).all(), "rows outside cu_seqlens_q were written"
    assert (lse[:, tq:] == 7.0).all(), "LSE outside cu_seqlens_q was written"


QK_NOPE_D, ROPE_D = 128, 64
QK_SCALE = (QK_NOPE_D + ROPE_D) ** -0.5


def _cu(lens, device="cuda"):
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(lens, dtype=torch.int32, device=device).cumsum(0)
    return cu


def generate_varlen_pe_fp8_inputs(
    q_lens, k_lens, hq, hk, dv=128, seed=0, device="cuda"
):
    """fp8 q, k/v views of one buffer, and k_pe a strided [Tk, 1, 64] view, as vLLM passes them."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    tq, tk = sum(q_lens), sum(k_lens)

    def rnd(*shape):
        x = torch.randn(shape, dtype=torch.float32, device=device, generator=g)
        return (0.5 * x).to(torch.float8_e4m3fn)

    q = rnd(tq, hq, QK_NOPE_D + ROPE_D)
    k, v = rnd(tk, hk, QK_NOPE_D + dv).split([QK_NOPE_D, dv], dim=-1)
    k_pe = rnd(tk, 512 + ROPE_D)[:, 512:].unsqueeze(1)
    return q, k, v, k_pe


def varlen_pe_ref(q, k, v, k_pe, q_lens, k_lens, causal, sm_scale=QK_SCALE):
    """fp32 out [Tq, H, dv] and lse [H, Tq]; masks are bottom-right aligned."""
    k = torch.cat([k.float(), k_pe.float().expand(-1, k.shape[1], -1)], dim=-1)
    outs, lses = [], []
    qs = ks = 0
    for lq, lk in zip(q_lens, k_lens):
        o, lse = mha_torch(
            *(
                t.float().transpose(0, 1)[None]
                for t in (q[qs : qs + lq], k[ks : ks + lk], v[ks : ks + lk])
            ),
            sm_scale,
            causal,
        )
        outs.append(o[0].transpose(0, 1))
        lses.append(lse[0])
        qs += lq
        ks += lk
    return torch.cat(outs), torch.cat(lses, dim=1)


def check_varlen_pe(o, lse, ref_o, ref_lse):
    """Rows without a visible key: o = 0 and lse = -inf. lse None: check only o."""
    empty = torch.isinf(ref_lse)
    assert (o.float().transpose(0, 1)[empty] == 0).all()
    torch.testing.assert_close(o.float(), ref_o, atol=3e-2, rtol=3e-2)
    if lse is not None:
        assert torch.isinf(lse[empty]).all()
        torch.testing.assert_close(lse[~empty], ref_lse[~empty], atol=1e-2, rtol=1e-3)


def _skip_unless_gfx1250():
    if get_arch() != "gfx1250":
        pytest.skip("Only supported on gfx1250")


def _seq_ids(lens):
    return torch.repeat_interleave(
        torch.arange(len(lens), device="cuda"), torch.tensor(lens, device="cuda")
    )


def generate_varlen_pe_scaled_inputs(q_lens, k_lens, hq, hk, dv, scales, seed=0):
    """fp8 inputs and dequant scales for one ``scales`` mode, and their fp32 values for the reference.

    none: no scales. per_seq: per-sequence amax scales (k_pe shares the k scale).
    zero_softmax: unit scales, softmax scale 0. zero_q: q_descale 0 for the first sequence.
    """
    n = len(q_lens)
    if scales != "per_seq":
        q, k, v, k_pe = generate_varlen_pe_fp8_inputs(q_lens, k_lens, hq, hk, dv, seed)
        if scales == "none":
            return q, k, v, k_pe, (None, None, None), QK_SCALE, (q, k, v, k_pe)
        ones = [
            torch.ones((n, h), dtype=torch.float32, device="cuda") for h in (hq, hk, hk)
        ]
        if scales == "zero_q":
            ones[0][0] = 0.0
        sm_scale = 0.0 if scales == "zero_softmax" else QK_SCALE
        q_ref = q.float() * ones[0][_seq_ids(q_lens)][..., None]
        return q, k, v, k_pe, tuple(ones), sm_scale, (q_ref, k, v, k_pe)

    tq, tk, f8 = sum(q_lens), sum(k_lens), torch.float8_e4m3fn
    seq_q, seq_k = _seq_ids(q_lens), _seq_ids(k_lens)
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)

    def rnd(*shape):
        return 0.5 * torch.randn(shape, dtype=torch.float32, device="cuda", generator=g)

    def amax_per_seq(x, seq):
        """amax over the tokens (and the last dim) of each sequence: [n, x.shape[1]]."""
        out = torch.zeros((n, x.shape[1]), dtype=torch.float32, device="cuda")
        return out.index_reduce_(0, seq, x.abs().amax(-1), "amax")

    q_f, kv_f, kpe_f = (
        rnd(tq, hq, QK_NOPE_D + ROPE_D),
        rnd(tk, hk, QK_NOPE_D + dv),
        rnd(tk, 1, ROPE_D),
    )
    q_scale = amax_per_seq(q_f, seq_q) / 448.0
    k_amax = torch.maximum(
        amax_per_seq(kv_f[..., :QK_NOPE_D], seq_k).amax(1, keepdim=True),
        amax_per_seq(kpe_f, seq_k),
    )
    k_scale = (k_amax / 448.0).expand(-1, hk).contiguous()
    v_scale = amax_per_seq(kv_f[..., QK_NOPE_D:], seq_k) / 448.0
    q = (q_f / q_scale[seq_q][..., None]).to(f8)
    kv = torch.empty((tk, hk, QK_NOPE_D + dv), dtype=f8, device="cuda")
    kv[..., :QK_NOPE_D] = (kv_f[..., :QK_NOPE_D] / k_scale[seq_k][..., None]).to(f8)
    kv[..., QK_NOPE_D:] = (kv_f[..., QK_NOPE_D:] / v_scale[seq_k][..., None]).to(f8)
    k, v = kv.split([QK_NOPE_D, dv], dim=-1)
    pe_buf = torch.empty((tk, 512 + ROPE_D), dtype=f8, device="cuda")
    pe_buf[:, 512:] = (kpe_f[:, 0] / k_scale[seq_k][:, :1]).to(f8)
    k_pe = pe_buf[:, 512:].unsqueeze(1)
    ks = k_scale[seq_k][..., None]
    ref = (
        q.float() * q_scale[seq_q][..., None],
        k.float() * ks,
        v.float() * v_scale[seq_k][..., None],
        k_pe.float().expand(-1, hk, -1) * ks,
    )
    return q, k, v, k_pe, (q_scale, k_scale, v_scale), QK_SCALE, ref


@pytest.mark.parametrize(
    "causal, q_lens, k_lens, hq, hk, dv, scales",
    [
        (True, [1024], [1024], 24, 24, 128, "none"),
        (True, [1000, 37, 2048, 1, 513], [1000, 37, 2048, 1, 513], 24, 24, 128, "none"),
        (True, [300, 1000], [900, 1000], 8, 8, 128, "none"),  # seqlen_k > seqlen_q
        # seqlen_k < seqlen_q: rows without keys
        (True, [300], [100], 4, 4, 128, "none"),
        (False, [256], [256], 24, 24, 128, "none"),  # short keys
        (False, [700], [5000], 24, 4, 128, "none"),  # GQA
        (False, [777, 64], [5000, 130], 16, 16, 128, "none"),
        (False, [512], [65536], 24, 24, 128, "none"),  # split-KV
        # split-KV, two sequences
        (False, [200, 312], [40000, 30000], 16, 16, 128, "per_seq"),
        (True, [600, 100], [600, 100], 8, 8, 64, "none"),
        (False, [300], [5000], 8, 2, 64, "none"),
        (True, [700, 300], [900, 300], 24, 4, 128, "per_seq"),
        (False, [700, 300], [900, 300], 24, 4, 128, "per_seq"),
        (True, [300, 100], [500, 100], 8, 8, 128, "zero_softmax"),
        (False, [300, 100], [500, 100], 8, 8, 128, "zero_softmax"),
        (True, [300, 100], [500, 100], 8, 8, 128, "zero_q"),
        (False, [300, 100], [500, 100], 8, 8, 128, "zero_q"),
    ],
)
@pytest.mark.parametrize("return_lse", [True, False])
def test_mha_varlen_fp8_shared_pe(
    causal, q_lens, k_lens, hq, hk, dv, scales, return_lse
):
    """k_pe shared by the heads, as vLLM passes it. return_lse=False is the default path, with other
    kernel options. A zero softmax or q scale gives equal weights to the visible keys. Rows after
    cu_seqlens_q[-1] must not be written."""
    _skip_unless_gfx1250()
    q, k, v, k_pe, (q_scale, k_scale, v_scale), sm_scale, ref_in = (
        generate_varlen_pe_scaled_inputs(q_lens, k_lens, hq, hk, dv, scales)
    )
    tq, pad = sum(q_lens), 50
    q = torch.cat([q, q[:pad]])
    out = torch.full((tq + pad, hq, dv), 7.0, dtype=torch.bfloat16, device="cuda")
    lse = (
        torch.full((hq, tq + pad), 7.0, dtype=torch.float32, device="cuda")
        if return_lse
        else None
    )
    res = mha_varlen_fwd(
        q, k, v, _cu(q_lens), _cu(k_lens), max(q_lens), max(k_lens),
        q_scale, k_scale, v_scale,
        softmax_scale=sm_scale, causal=causal, return_lse=return_lse, out=out, lse=lse,
        k_pe=k_pe,
    )  # fmt: skip
    o = res[0] if return_lse else res
    ref = varlen_pe_ref(*ref_in, q_lens, k_lens, causal, sm_scale=sm_scale)
    assert torch.isfinite(o[:tq].float()).all()
    check_varlen_pe(o[:tq], None if lse is None else lse[:, :tq], *ref)
    assert (out[tq:] == 7.0).all(), "rows outside cu_seqlens_q were written"
    if lse is not None:
        assert (lse[:, tq:] == 7.0).all(), "LSE outside cu_seqlens_q was written"


@pytest.mark.parametrize("context_first", [True, False])
def test_mha_varlen_fp8_shared_pe_merge(context_first):
    """A split-KV context pass and a causal new-token pass, merged via o_init and lse_init."""
    _skip_unless_gfx1250()
    q_lens, ctx_lens, h = [700, 300], [6000, 20000], 24
    q, k_new, v_new, kpe_new = generate_varlen_pe_fp8_inputs(
        q_lens, q_lens, h, h, seed=1
    )
    _, k_ctx, v_ctx, kpe_ctx = generate_varlen_pe_fp8_inputs(
        q_lens, ctx_lens, h, h, seed=2
    )
    cu_q, cu_ctx = _cu(q_lens), _cu(ctx_lens)
    passes = [
        (k_ctx, v_ctx, kpe_ctx, cu_ctx, max(ctx_lens), False),
        (k_new, v_new, kpe_new, cu_q, max(q_lens), True),
    ]
    if not context_first:
        passes.reverse()
    o = lse = None
    for k, v, k_pe, cu_k, max_k, causal in passes:
        o, lse = mha_varlen_fwd(
            q, k, v, cu_q, cu_k, max(q_lens), max_k, None, None, None,
            softmax_scale=QK_SCALE, causal=causal, return_lse=True, out=o, lse=lse,
            k_pe=k_pe, o_init=o, lse_init=lse,
        )  # fmt: skip
    # The same keys in one call: per sequence, the context keys then the new keys.
    k_all, v_all, kpe_all, k_lens = [], [], [], []
    for s in range(len(q_lens)):
        a0, a1 = cu_ctx[s].item(), cu_ctx[s + 1].item()
        b0, b1 = cu_q[s].item(), cu_q[s + 1].item()
        k_all += [k_ctx[a0:a1], k_new[b0:b1]]
        v_all += [v_ctx[a0:a1], v_new[b0:b1]]
        kpe_all += [kpe_ctx[a0:a1], kpe_new[b0:b1]]
        k_lens.append((a1 - a0) + (b1 - b0))
    ref = varlen_pe_ref(
        q, torch.cat(k_all), torch.cat(v_all), torch.cat(kpe_all), q_lens, k_lens, True
    )
    check_varlen_pe(o, lse, *ref)
