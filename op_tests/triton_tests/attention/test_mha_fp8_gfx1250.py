# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass, fields

import pytest
import torch

from aiter.ops.triton.attention.mha_fp8 import mha_fwd, mha_fwd_mla_varlen
from aiter.ops.triton.utils._triton.arch_info import get_arch

arch = get_arch()


# ---------------
# reference
# ---------------


def mha_torch(q, k, v, sm_scale, causal):
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
    p = torch.exp(qk - qk.amax(-1, keepdim=True))
    l = p.sum(-1, keepdim=True)
    p_q = p.to(torch.float8_e4m3fn).to(torch.float32)
    o = (p_q @ v) / l
    return o


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
    if get_arch() != "gfx1250":
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

    o_ref = mha_torch(q_ref, k_ref, v_ref, sm_scale, causal=causal)

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


# variant "mla_varlen": packed varlen sequences with MLA-shaped heads
MLA_D, MLA_PE, MLA_V = 128, 64, 128
MLA_SCALE = (MLA_D + MLA_PE) ** -0.5


def _cu(lens):
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32, device="cuda")
    cu[1:] = torch.tensor(lens, dtype=torch.int32, device="cuda").cumsum(0)
    return cu


def mla_inputs(q_lens, k_lens, hq, hk, seed):
    """fp8 q [Tq, hq, 192], k [Tk, hk, 128] and v views of one [Tk, hk, 256] buffer,
    and k_pe, a [Tk, 1, 64] view of a [Tk, 576] latent, as vLLM passes them."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    tq, tk = sum(q_lens), sum(k_lens)
    f8 = torch.float8_e4m3fn

    def rnd(*shape):
        return torch.randn(shape, dtype=torch.float32, device="cuda", generator=g)

    q = rnd(tq, hq, MLA_D + MLA_PE).to(f8)
    k, v = rnd(tk, hk, MLA_D + MLA_V).to(f8).split([MLA_D, MLA_V], dim=-1)
    k_pe = rnd(tk, 512 + MLA_PE).to(f8)[:, 512:].unsqueeze(1)
    return q, k, v, k_pe


@torch.no_grad()
def mla_torch(q, k, v, k_pe, q_lens, k_lens, causal):
    """fp32 out [Tq, H, dv] and lse [H, Tq]; masks are bottom-right aligned."""
    hq, hk = q.shape[1], k.shape[1]
    k = torch.cat([k.float(), k_pe.float().expand(-1, hk, -1)], dim=-1)
    k = k.repeat_interleave(hq // hk, dim=1)
    v = v.float().repeat_interleave(hq // hk, dim=1)
    outs, lses = [], []
    qs = ks = 0
    for lq, lk in zip(q_lens, k_lens):
        s = torch.einsum("qhd,khd->hqk", q[qs : qs + lq].float(), k[ks : ks + lk])
        s = s * MLA_SCALE
        if causal:
            rows = torch.arange(lq, device="cuda")[:, None] + (lk - lq)
            s = s.masked_fill(
                torch.arange(lk, device="cuda")[None, :] > rows, float("-inf")
            )
        lse = torch.logsumexp(s, -1)
        p = torch.exp(s - torch.where(torch.isinf(lse), 0.0, lse)[..., None])
        outs.append(torch.einsum("hqk,khd->qhd", p, v[ks : ks + lk]))
        lses.append(lse)
        qs += lq
        ks += lk
    return torch.cat(outs), torch.cat(lses, dim=1)


def check_mla(o, lse, ref_o, ref_lse):
    """Rows without a visible key: o = 0 and lse = -inf."""
    empty = torch.isinf(ref_lse)
    assert torch.isinf(lse[empty]).all()
    assert (o.float().transpose(0, 1)[empty] == 0).all()
    torch.testing.assert_close(o.float(), ref_o, atol=8e-2, rtol=0)
    torch.testing.assert_close(lse[~empty], ref_lse[~empty], atol=1e-2, rtol=0)


@pytest.mark.parametrize(
    "causal, q_lens, k_lens, hq, hk",
    [
        (True, [1024], [1024], 24, 24),
        (True, [1000, 37, 2048, 1, 513], [1000, 37, 2048, 1, 513], 24, 24),
        (True, [300, 1000], [900, 1000], 8, 8),  # seqlen_k > seqlen_q
        (True, [300], [100], 4, 4),  # seqlen_k < seqlen_q: rows without keys
        (False, [256], [256], 24, 24),  # short keys
        (False, [700], [5000], 24, 4),  # GQA
        (False, [777, 64], [5000, 130], 16, 16),
        (False, [512], [65536], 24, 24),  # split-KV
    ],
)
def test_mha_fp8_mla_varlen(causal, q_lens, k_lens, hq, hk):
    if get_arch() != "gfx1250":
        pytest.skip("Only supported on gfx1250")
    q, k, v, k_pe = mla_inputs(q_lens, k_lens, hq, hk, seed=0)
    o, lse = mha_fwd_mla_varlen(
        q,
        k,
        v,
        k_pe,
        _cu(q_lens),
        _cu(k_lens),
        max(q_lens),
        max(k_lens),
        sm_scale=MLA_SCALE,
        causal=causal,
        return_lse=True,
    )
    check_mla(o, lse, *mla_torch(q, k, v, k_pe, q_lens, k_lens, causal))


@pytest.mark.parametrize("causal", [True, False])
def test_mha_fp8_mla_varlen_scales(causal):
    """Dequant scales per sequence (and per head for q and v), made like quant_fp8.
    k_pe is shared by the heads and uses the k scale, so the k scale is per sequence."""
    if get_arch() != "gfx1250":
        pytest.skip("Only supported on gfx1250")
    q_lens, k_lens, hq, hk = [700, 300], [900, 300], 24, 4
    tq, tk, f8 = sum(q_lens), sum(k_lens), torch.float8_e4m3fn
    seq_q = torch.repeat_interleave(
        torch.arange(len(q_lens), device="cuda"), torch.tensor(q_lens, device="cuda")
    )
    seq_k = torch.repeat_interleave(
        torch.arange(len(k_lens), device="cuda"), torch.tensor(k_lens, device="cuda")
    )
    g = torch.Generator(device="cuda")
    g.manual_seed(3)

    def rnd(*shape):
        return 0.5 * torch.randn(shape, dtype=torch.float32, device="cuda", generator=g)

    def amax_per_seq(x, seq, n):
        """amax over the tokens (and the last dim) of each sequence: [n, x.shape[1]]."""
        out = torch.zeros((n, x.shape[1]), dtype=torch.float32, device="cuda")
        return out.index_reduce_(0, seq, x.abs().amax(-1), "amax")

    q_f, kv_f, kpe_f = rnd(tq, hq, MLA_D + MLA_PE), rnd(tk, hk, 256), rnd(tk, 1, MLA_PE)
    q_scale = amax_per_seq(q_f, seq_q, len(q_lens)) / 448.0
    k_amax = torch.maximum(
        amax_per_seq(kv_f[..., :MLA_D], seq_k, len(k_lens)).amax(1, keepdim=True),
        amax_per_seq(kpe_f, seq_k, len(k_lens)),
    )
    k_scale = (k_amax / 448.0).expand(-1, hk).contiguous()
    v_scale = amax_per_seq(kv_f[..., MLA_D:], seq_k, len(k_lens)) / 448.0
    q = (q_f / q_scale[seq_q][..., None]).to(f8)
    kv = torch.empty((tk, hk, 256), dtype=f8, device="cuda")
    kv[..., :MLA_D] = (kv_f[..., :MLA_D] / k_scale[seq_k][..., None]).to(f8)
    kv[..., MLA_D:] = (kv_f[..., MLA_D:] / v_scale[seq_k][..., None]).to(f8)
    k, v = kv.split([MLA_D, MLA_V], dim=-1)
    latent = torch.empty((tk, 512 + MLA_PE), dtype=f8, device="cuda")
    latent[:, 512:] = (kpe_f[:, 0] / k_scale[seq_k][:, :1]).to(f8)
    k_pe = latent[:, 512:].unsqueeze(1)
    o, lse = mha_fwd_mla_varlen(
        q,
        k,
        v,
        k_pe,
        _cu(q_lens),
        _cu(k_lens),
        max(q_lens),
        max(k_lens),
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sm_scale=MLA_SCALE,
        causal=causal,
        return_lse=True,
    )
    ks = k_scale[seq_k][..., None]
    ref = mla_torch(
        q.float() * q_scale[seq_q][..., None],
        k.float() * ks,
        v.float() * v_scale[seq_k][..., None],
        k_pe.float().expand(-1, hk, -1) * ks,
        q_lens,
        k_lens,
        causal,
    )
    check_mla(o, lse, *ref)


def test_mha_fp8_mla_varlen_merge():
    """A context pass (split-KV) merged in place with a causal new-token pass."""
    if get_arch() != "gfx1250":
        pytest.skip("Only supported on gfx1250")
    q_lens, ctx_lens, h = [700, 300], [6000, 20000], 24
    q, k_new, v_new, kpe_new = mla_inputs(q_lens, q_lens, h, h, seed=1)
    _, k_ctx, v_ctx, kpe_ctx = mla_inputs(q_lens, ctx_lens, h, h, seed=2)
    cu_q, cu_ctx = _cu(q_lens), _cu(ctx_lens)
    o, lse = mha_fwd_mla_varlen(
        q,
        k_ctx,
        v_ctx,
        kpe_ctx,
        cu_q,
        cu_ctx,
        max(q_lens),
        max(ctx_lens),
        sm_scale=MLA_SCALE,
        return_lse=True,
    )
    mha_fwd_mla_varlen(
        q,
        k_new,
        v_new,
        kpe_new,
        cu_q,
        cu_q,
        max(q_lens),
        max(q_lens),
        sm_scale=MLA_SCALE,
        causal=True,
        out=o,
        lse_out=lse,
        o_init=o,
        lse_init=lse,
    )
    # The same keys in one call: per sequence, the context keys then the new keys.
    k_all, v_all, kpe_all, k_lens = [], [], [], []
    for s in range(len(q_lens)):
        a0, a1 = cu_ctx[s].item(), cu_ctx[s + 1].item()
        b0, b1 = cu_q[s].item(), cu_q[s + 1].item()
        k_all += [k_ctx[a0:a1], k_new[b0:b1]]
        v_all += [v_ctx[a0:a1], v_new[b0:b1]]
        kpe_all += [kpe_ctx[a0:a1], kpe_new[b0:b1]]
        k_lens.append((a1 - a0) + (b1 - b0))
    ref = mla_torch(
        q, torch.cat(k_all), torch.cat(v_all), torch.cat(kpe_all), q_lens, k_lens, True
    )
    check_mla(o, lse, *ref)
