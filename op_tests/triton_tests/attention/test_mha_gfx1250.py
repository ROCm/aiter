# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests of the gfx1250 Gluon path of flash_attn_varlen_func (MLA prefill shapes:
bf16, qk head dim 192 = 128 + 64 PE, v head dim 128 or 192, Hq == Hk)."""

import math

import pytest
import torch

from aiter.ops.triton.attention.mha import (
    flash_attn_varlen_func,
    gluon_forward_unsupported_reason,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

D_NOPE, D_PE, D_V = 128, 64, 128
D_QK = D_NOPE + D_PE
SCALE = 1.0 / math.sqrt(D_QK)

pytestmark = pytest.mark.skipif(
    get_arch() != "gfx1250", reason="The Gluon MLA-shape MHA kernel is gfx1250 only"
)


def _cu(lens):
    t = torch.zeros(len(lens) + 1, dtype=torch.int32, device="cuda")
    t[1:] = torch.tensor(lens, dtype=torch.int32, device="cuda").cumsum(0)
    return t


def _make_inputs(q_lens, k_lens, H, seed=0, layout="shared_pe", d_v=D_V):
    """q (Tq, H, 192). Returns (q, k, k_pe, v) for flash_attn_varlen_func.

    layout:
      "shared_pe": k (Tk, H, 128) and v (Tk, H, 128) are views of one (Tk, H, 256)
          buffer; k_pe (Tk, 1, 64) is a view of a (Tk, 576) buffer (vLLM MLA).
      "contiguous": contiguous k (Tk, H, 128), v and k_pe (Tk, 1, 64).
      "concat": k = cat(k_nope, k_pe per head) (Tk, H, 192), k_pe is None.
    """
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    Tq, Tk = sum(q_lens), sum(k_lens)

    def rnd(*shape):
        return torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=g)

    q = rnd(Tq, H, D_QK)
    if layout == "shared_pe":
        k, v = rnd(Tk, H, D_NOPE + D_V).split([D_NOPE, D_V], dim=-1)
        k_pe = rnd(Tk, 512 + D_PE)[:, 512:].unsqueeze(1)
    else:
        k, v, k_pe = rnd(Tk, H, D_NOPE), rnd(Tk, H, D_V), rnd(Tk, 1, D_PE)
    if d_v != D_V:
        v = rnd(Tk, H, d_v)
    if layout == "concat":
        k = torch.cat([k, k_pe.expand(-1, H, -1)], dim=-1)
        k_pe = None
    return q, k, k_pe, v


def _full_k(k, k_pe):
    if k_pe is None:
        return k
    return torch.cat([k, k_pe.expand(-1, k.shape[1], -1)], dim=-1)


@torch.no_grad()
def _reference(q, k, v, q_lens, k_lens, causal, row_stride=1, head_chunk=4):
    """fp32 reference on a subset of rows, causal mask bottom-right aligned.
    Returns out (Tq, H, d_v), lse (H, Tq) and the mask of the checked rows."""
    Tq, H, _ = q.shape
    d_v = v.shape[-1]
    out = torch.full((Tq, H, d_v), float("nan"), dtype=torch.float32, device="cuda")
    lse = torch.full((H, Tq), float("nan"), dtype=torch.float32, device="cuda")
    rows_mask = torch.zeros(Tq, dtype=torch.bool, device="cuda")
    qs = ks = 0
    for Lq, Lk in zip(q_lens, k_lens):
        rows = torch.arange(0, Lq, row_stride, device="cuda")
        if rows[-1].item() != Lq - 1:
            rows = torch.cat([rows, torch.tensor([Lq - 1], device="cuda")])
        qq = q[qs + rows].float().transpose(0, 1)
        kk = k[ks : ks + Lk].float().transpose(0, 1)
        vv = v[ks : ks + Lk].float().transpose(0, 1)
        for h0 in range(0, H, head_chunk):
            hs = slice(h0, h0 + head_chunk)
            s = torch.matmul(qq[hs], kk[hs].transpose(1, 2)) * SCALE
            if causal:
                lim = rows + (Lk - Lq)
                kpos = torch.arange(Lk, device="cuda")
                s = s.masked_fill(kpos[None, None, :] > lim[None, :, None], float("-inf"))
            m = s.amax(-1, keepdim=True)
            m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
            p = torch.exp(s - m)
            l = p.sum(-1, keepdim=True)
            out[qs + rows, hs] = (torch.matmul(p, vv[hs]) / l).transpose(0, 1)
            lse[hs, qs + rows] = (m + torch.log(l)).squeeze(-1)
        rows_mask[qs + rows] = True
        qs += Lq
        ks += Lk
    return out, lse, rows_mask


def _check(o, lse, ref_o, ref_lse, rows_mask, o_atol=2e-2, lse_atol=2e-3):
    """lse is (H, Tq). Rows without a visible key: o = 0, lse = -inf."""
    empty = torch.isnan(ref_o).any(-1).any(-1) & rows_mask
    if empty.any():
        assert (o[empty] == 0).all() and torch.isinf(lse[:, empty]).all()
        rows_mask = rows_mask & ~empty
    torch.testing.assert_close(o[rows_mask].float(), ref_o[rows_mask], atol=o_atol, rtol=0)
    torch.testing.assert_close(
        lse[:, rows_mask], ref_lse[:, rows_mask], atol=lse_atol, rtol=0
    )


def _supported(q, k, v, cu_q, cu_k, **kw):
    return gluon_forward_unsupported_reason(
        q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, **kw
    )


# (causal, q_lens, k_lens, H, layout)
CASES = [
    (True, [1], [1], 2, "shared_pe"),
    (True, [77], [77], 3, "shared_pe"),
    (True, [128], [128], 2, "shared_pe"),
    (True, [1000], [1000], 4, "shared_pe"),
    (True, [513, 64, 1, 300], [513, 64, 1, 300], 4, "shared_pe"),
    (True, [2048], [2048], 24, "contiguous"),
    (True, [1024] * 7, [1024] * 7, 24, "shared_pe"),
    (True, [7680], [7680], 24, "shared_pe"),
    (True, [100], [300], 2, "shared_pe"),  # Lk > Lq (bottom-right)
    (True, [300], [100], 2, "shared_pe"),  # Lk < Lq (rows with no keys)
    (False, [128], [128], 2, "shared_pe"),
    (False, [100], [1000], 4, "shared_pe"),
    (False, [1], [5], 2, "shared_pe"),
    (False, [333, 50], [4097, 129], 4, "shared_pe"),
    (False, [2048], [16896], 24, "shared_pe"),
    (False, [7680], [16896], 24, "shared_pe"),
    (False, [512], [131072], 24, "shared_pe"),  # split-KV path
]


@pytest.mark.parametrize("causal, q_lens, k_lens, H, layout", CASES)
def test_mha_varlen_gfx1250(causal, q_lens, k_lens, H, layout):
    q, k, k_pe, v = _make_inputs(q_lens, k_lens, H, layout=layout)
    cu_q, cu_k = _cu(q_lens), _cu(k_lens)
    assert _supported(q, k, v, cu_q, cu_k, k_pe=k_pe) is None
    o, lse = flash_attn_varlen_func(
        q, k, v, cu_q, cu_k, max(q_lens), max(k_lens), softmax_scale=SCALE,
        causal=causal, return_lse=True, k_pe=k_pe,
    )
    assert lse.shape == (sum(q_lens), H)
    ref_o, ref_lse, rows = _reference(
        q, _full_k(k, k_pe), v, q_lens, k_lens, causal,
        row_stride=max(1, sum(q_lens) // 600),
    )
    _check(o, lse.t(), ref_o, ref_lse, rows)


# The call that vLLM makes on gfx1250 today: k = cat(k_nope, k_pe) per head,
# v zero-padded to 192, no backend argument.
# (causal, q_lens, k_lens, H, v_mode)
CONCAT_CASES = [
    (True, [513, 64, 1, 300], [513, 64, 1, 300], 4, "v128"),
    (True, [4096], [4096], 24, "pad192"),
    (True, [1024] * 7, [1024] * 7, 8, "rand192"),
    (False, [333, 50], [4097, 129], 4, "pad192"),
    (False, [2048], [1024], 8, "rand192"),
    (False, [7680], [16896], 24, "v128"),
    (False, [512], [65536], 24, "pad192"),  # split-KV with a 192-wide V
]


@pytest.mark.parametrize("causal, q_lens, k_lens, H, v_mode", CONCAT_CASES)
def test_mha_varlen_gfx1250_concat_k(causal, q_lens, k_lens, H, v_mode):
    q, k, _, v = _make_inputs(
        q_lens, k_lens, H, layout="concat", d_v=D_QK if v_mode == "rand192" else D_V
    )
    if v_mode == "pad192":
        v = torch.nn.functional.pad(v, [0, D_QK - D_V], value=0)
    cu_q, cu_k = _cu(q_lens), _cu(k_lens)
    assert _supported(q, k, v, cu_q, cu_k) is None
    o, lse = flash_attn_varlen_func(
        q, k, v, cu_q, cu_k, max(q_lens), max(k_lens), softmax_scale=SCALE,
        causal=causal, return_lse=True,
    )
    assert o.shape == (sum(q_lens), H, v.shape[-1])
    ref_o, ref_lse, rows = _reference(
        q, k, v, q_lens, k_lens, causal, row_stride=max(1, sum(q_lens) // 600)
    )
    _check(o, lse.t(), ref_o, ref_lse, rows)
    if v_mode == "pad192":
        assert (o[..., D_V:] == 0).all()


# (new_tokens_first, q_lens, ctx_lens, H, d_v, config of the context pass)
MERGE_CASES = [
    (False, [300], [1000], 4, D_V, None),
    (False, [513, 77], [4096, 0], 4, D_V, None),
    (False, [512], [32768], 8, D_V, None),
    (False, [1024], [8192], 24, D_V, dict(NUM_SPLITS=1)),
    (False, [2048], [16896], 24, D_V, None),
    (False, [100], [70000], 4, D_V, None),
    (True, [512], [32768], 8, D_V, None),  # split-KV combine with an initial state
    (True, [300, 200], [5000, 9000], 4, D_V, None),
    (True, [2048], [16896], 24, D_V, None),
    (False, [300, 200], [5000, 9000], 4, D_QK, None),
    (True, [512], [32768], 8, D_QK, None),
]


@pytest.mark.parametrize("new_first, q_lens, ctx_lens, H, d_v, ctx_cfg", MERGE_CASES)
def test_mha_varlen_gfx1250_merge(new_first, q_lens, ctx_lens, H, d_v, ctx_cfg):
    """Context pass + new-token pass merged in place with o_init/lse_init."""
    q, k_new, kpe_new, v_new = _make_inputs(q_lens, q_lens, H, seed=1, d_v=d_v)
    _, k_ctx, kpe_ctx, v_ctx = _make_inputs(q_lens, ctx_lens, H, seed=2, d_v=d_v)
    cu_q, cu_ctx = _cu(q_lens), _cu(ctx_lens)
    Tq = sum(q_lens)
    out = torch.empty((Tq, H, d_v), dtype=torch.bfloat16, device="cuda")
    lse = torch.empty((H, Tq), dtype=torch.float32, device="cuda")
    mx_q, mx_c = max(q_lens), max(max(ctx_lens), 1)

    def ctx(**kw):
        flash_attn_varlen_func(
            q, k_ctx, v_ctx, cu_q, cu_ctx, mx_q, mx_c, softmax_scale=SCALE,
            causal=False, out=out, lse_out=lse, k_pe=kpe_ctx, config=ctx_cfg,
            backend="gluon", **kw,
        )

    def new(**kw):
        flash_attn_varlen_func(
            q, k_new, v_new, cu_q, cu_q, mx_q, mx_q, softmax_scale=SCALE,
            causal=True, out=out, lse_out=lse, k_pe=kpe_new, **kw,
        )

    if new_first:
        new()
        ctx(o_init=out, lse_init=lse)
    else:
        ctx()
        new(o_init=out, lse_init=lse)

    k_all, v_all, k_lens = [], [], []
    k_ctx, k_new = _full_k(k_ctx, kpe_ctx), _full_k(k_new, kpe_new)
    for s in range(len(q_lens)):
        a0, a1 = cu_ctx[s].item(), cu_ctx[s + 1].item()
        b0, b1 = cu_q[s].item(), cu_q[s + 1].item()
        k_all.append(torch.cat([k_ctx[a0:a1], k_new[b0:b1]]))
        v_all.append(torch.cat([v_ctx[a0:a1], v_new[b0:b1]]))
        k_lens.append((a1 - a0) + (b1 - b0))
    ref_o, ref_lse, rows = _reference(
        q, torch.cat(k_all), torch.cat(v_all), q_lens, k_lens, True,
        row_stride=max(1, Tq // 400),
    )
    _check(out, lse, ref_o, ref_lse, rows)


@pytest.mark.parametrize(
    "causal, q_lens, k_lens, H",
    [
        (False, [128], [524288], 4),  # 512K context: 64-bit addressing
        (False, [256, 64], [300000, 224288], 4),
        (True, [16384], [16384], 8),
    ],
)
def test_mha_varlen_gfx1250_long(causal, q_lens, k_lens, H):
    q, k, k_pe, v = _make_inputs(q_lens, k_lens, H)
    cu_q, cu_k = _cu(q_lens), _cu(k_lens)
    lse = torch.empty((H, sum(q_lens)), dtype=torch.float32, device="cuda")
    o, lse_ret = flash_attn_varlen_func(
        q, k, v, cu_q, cu_k, max(q_lens), max(k_lens), softmax_scale=SCALE,
        causal=causal, return_lse=True, k_pe=k_pe, lse_out=lse,
    )
    assert lse_ret is lse
    ref_o, ref_lse, rows = _reference(
        q, _full_k(k, k_pe), v, q_lens, k_lens, causal,
        row_stride=max(1, sum(q_lens) // 128), head_chunk=1,
    )
    _check(o, lse, ref_o, ref_lse, rows)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("v_mode", ["v128", "pad192"])
def test_mha_varlen_gfx1250_default_backend_parity(causal, v_mode):
    """backend=None (Gluon on gfx1250) agrees with backend="triton"."""
    q_lens, k_lens, H = [700, 1, 300], ([700, 1, 300] if causal else [2000, 7, 900]), 8
    q, k, _, v = _make_inputs(q_lens, k_lens, H, layout="concat")
    if v_mode == "pad192":
        v = torch.nn.functional.pad(v, [0, D_QK - D_V], value=0)
    cu_q, cu_k = _cu(q_lens), _cu(k_lens)
    args = (q, k, v, cu_q, cu_k, max(q_lens), max(k_lens))
    o_auto = flash_attn_varlen_func(*args, causal=causal)
    o_tri = flash_attn_varlen_func(*args, causal=causal, backend="triton")
    assert isinstance(o_auto, torch.Tensor) and o_auto.shape == o_tri.shape
    torch.testing.assert_close(o_auto, o_tri, atol=2e-2, rtol=0)
    o_auto, lse_auto = flash_attn_varlen_func(*args, causal=causal, return_lse=True)
    o_tri, lse_tri = flash_attn_varlen_func(
        *args, causal=causal, return_lse=True, backend="triton"
    )
    assert lse_auto.shape == lse_tri.shape == (sum(q_lens), H)
    torch.testing.assert_close(o_auto, o_tri, atol=2e-2, rtol=0)
    torch.testing.assert_close(lse_auto, lse_tri, atol=2e-3, rtol=0)


def test_mha_varlen_gfx1250_out_and_init_on_triton():
    """The new arguments give the same result on the Triton backend."""
    q_lens, ctx_lens, H = [300, 200], [1000, 700], 4
    q, k_new, kpe_new, v_new = _make_inputs(q_lens, q_lens, H, seed=1)
    _, k_ctx, kpe_ctx, v_ctx = _make_inputs(q_lens, ctx_lens, H, seed=2)
    cu_q, cu_ctx = _cu(q_lens), _cu(ctx_lens)
    Tq = sum(q_lens)
    results = []
    for backend in ("gluon", "triton"):
        out = torch.empty((Tq, H, D_V), dtype=torch.bfloat16, device="cuda")
        lse = torch.empty((H, Tq), dtype=torch.float32, device="cuda")
        flash_attn_varlen_func(
            q, k_ctx, v_ctx, cu_q, cu_ctx, max(q_lens), max(ctx_lens),
            out=out, lse_out=lse, k_pe=kpe_ctx, backend=backend,
        )
        o, lse_ret = flash_attn_varlen_func(
            q, k_new, v_new, cu_q, cu_q, max(q_lens), max(q_lens), causal=True,
            return_lse=True, out=out, lse_out=lse, k_pe=kpe_new, o_init=out,
            lse_init=lse, backend=backend,
        )
        assert o is out and lse_ret is lse
        results.append((out, lse))
    torch.testing.assert_close(results[0][0], results[1][0], atol=2e-2, rtol=0)
    torch.testing.assert_close(results[0][1], results[1][1], atol=2e-3, rtol=0)


def test_mha_varlen_gfx1250_fallback():
    """Calls outside the Gluon kernel's shapes use the Triton kernel unchanged."""
    q_lens, H = [300, 100], 4
    cu = _cu(q_lens)
    Tq = sum(q_lens)
    # GQA and a 128-wide qk head: not supported by the gfx1250 Gluon kernel
    for hk, d in ((2, D_QK), (H, 128)):
        q = torch.randn((Tq, H, d), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((Tq, hk, d), dtype=torch.bfloat16, device="cuda")
        v = torch.randn((Tq, hk, D_V), dtype=torch.bfloat16, device="cuda")
        args = (q, k, v, cu, cu, max(q_lens), max(q_lens))
        assert _supported(q, k, v, cu, cu) is not None
        o_auto = flash_attn_varlen_func(*args, causal=True)
        o_tri = flash_attn_varlen_func(*args, causal=True, backend="triton")
        torch.testing.assert_close(o_auto, o_tri, atol=0, rtol=0)
        with pytest.raises(ValueError):
            flash_attn_varlen_func(*args, causal=True, backend="gluon")
    # inputs that need a gradient use the Triton kernel (the Gluon kernel is forward-only)
    q, k, _, v = _make_inputs(q_lens, q_lens, H, layout="concat")
    args = (q, k, v, cu, cu, max(q_lens), max(q_lens))
    o_tri = flash_attn_varlen_func(*args, causal=True, backend="triton")
    q.requires_grad_(True)
    o_auto = flash_attn_varlen_func(*args, causal=True)
    assert o_auto.requires_grad
    torch.testing.assert_close(o_auto.detach(), o_tri, atol=0, rtol=0)
