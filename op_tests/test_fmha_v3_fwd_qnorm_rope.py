# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""fmha_v3_fwd_qnorm_rope vs a Triton RMSNorm + interleaved-RoPE row kernel followed by the plain v3 forward.

Everything is required to be bitwise equal: out, lse, q_n, q_rstd, and dq / dk / dv of the deterministic backward run
on q_n (with the fused forward's out / lse) vs on the reference q (with the reference forward's out / lse). The v3 asm
backward (non-deterministic mode, as training runs it) must give bitwise dk / dv; its dq accumulates with atomics in
16 bits, so it is compared within a tolerance.
The reference row kernel uses 8 rows per program and 4 warps (32 threads per row, 4 columns each), whose reduction
order the fused kernel reproduces.

    python op_tests/test_fmha_v3_fwd_qnorm_rope.py [-b 32] [-s 512] [--heads 24]
    pytest op_tests/test_fmha_v3_fwd_qnorm_rope.py
"""

import argparse

import pytest
import torch
import triton
import triton.language as tl

import aiter
from aiter.ops.mha import _flash_attn_backward, _flash_attn_forward

D = 128


@triton.jit
def _norm_rope_rows(X, W, COS, SIN, OUT, RSTD, M, BH, EPS, SX, SC, D: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M
    mask = mask_m[:, None]
    half: tl.constexpr = D // 2
    cols = tl.arange(0, D)[None, :]
    x = tl.load(X + rows[:, None] * SX + cols, mask=mask, other=0.0).to(tl.float32)
    x_lo, x_hi = tl.split(tl.reshape(x, (BLOCK_M, half, 2)))
    rstd = tl.rsqrt((tl.sum(x_lo * x_lo, axis=1) + tl.sum(x_hi * x_hi, axis=1)) / D + EPS)[:, None]
    w = tl.load(W + tl.arange(0, D)).to(tl.float32)
    w_lo, w_hi = tl.split(tl.reshape(w, (half, 2)))
    n_lo, n_hi = x_lo * rstd * w_lo[None, :], x_hi * rstd * w_hi[None, :]
    s = rows // BH
    c = tl.load(COS + s[:, None] * SC + cols, mask=mask, other=0.0).to(tl.float32)
    c_lo, c_hi = tl.split(tl.reshape(c, (BLOCK_M, half, 2)))
    sn = tl.load(SIN + s[:, None] * SC + cols, mask=mask, other=0.0).to(tl.float32)
    s_lo, s_hi = tl.split(tl.reshape(sn, (BLOCK_M, half, 2)))
    o = tl.reshape(tl.join(n_lo * c_lo - n_hi * s_lo, n_hi * c_hi + n_lo * s_hi), (BLOCK_M, D))
    tl.store(OUT + rows[:, None] * D + cols, o.to(OUT.dtype.element_ty), mask=mask)
    tl.store(RSTD + rows, tl.reshape(rstd, (BLOCK_M,)), mask=mask_m)


def norm_rope_ref(x_sbhd, w, cos, sin, eps, out_sbhd, rstd):
    """x: [S, B, H, D] view with rows evenly strided; writes out (packed [S, B, H, D]) and rstd [S*B*H]."""
    S, B, H, _ = x_sbhd.shape
    M = S * B * H
    _norm_rope_rows[(triton.cdiv(M, 8),)](
        x_sbhd, w, cos, sin, out_sbhd, rstd, M, M // cos.shape[0], eps, x_sbhd.stride(2), cos.stride(0),
        D=D, BLOCK_M=8, num_warps=4, num_stages=2)


def bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def same(a, b):
    return torch.equal(bits(a), bits(b))


def run(B, S, H, streams, eps=1e-6, seed=0):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, device=dev, dtype=torch.bfloat16, generator=g)  # noqa: E731
    qkv = rn(S, B, H, 3 * D)  # fused QKV projection output, sbhd
    q_raw = qkv[..., :D]
    k = rn(S, B, H, D)
    v = qkv[..., 2 * D:]
    sets = []
    for n in streams:
        w = (1.0 + 0.3 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
        ang = torch.rand(n * B, D // 2, device=dev, generator=g) * 6.3
        cos = torch.cos(ang).repeat_interleave(2, -1).to(torch.bfloat16).contiguous()
        sin = torch.sin(ang * 1.7).repeat_interleave(2, -1).to(torch.bfloat16).contiguous()
        sets.append((w, cos, sin, torch.cat([cos[:, 0::2], sin[:, 0::2]], -1).contiguous()))

    # reference: row kernel per stream into one packed q, then the plain v3 forward
    q_ref = torch.empty(S, B, H, D, device=dev, dtype=torch.bfloat16)
    rstd_ref = torch.empty(S * B * H, device=dev)
    s0 = 0
    for n, (w, cos, sin, _) in zip(streams, sets):
        norm_rope_ref(q_raw[s0:s0 + n], w, cos, sin, eps, q_ref[s0:s0 + n], rstd_ref[s0 * B * H:(s0 + n) * B * H])
        s0 += n
    bshd = lambda t: t.transpose(0, 1)  # noqa: E731
    scale = D ** -0.5
    out_ref = torch.empty(S, B, H, D, device=dev, dtype=torch.bfloat16).permute(1, 0, 2, 3)
    _, lse_ref, _, rng_state = _flash_attn_forward(bshd(q_ref), bshd(k), bshd(v), 0.0, scale, False, -1, -1, 0, None, None,
                                           None, None, None, True, False, out=out_ref)

    # fused
    ntile_a = streams[0] // 256 if len(streams) == 2 else S // 256
    (wa, _, _, ta), (wb, _, _, tb) = sets[0], sets[-1]
    assert aiter.fmha_v3_fwd_qnorm_rope_ok(B, S, H, D, ntile_a)
    q_n = torch.full((S, B, H, D), float("nan"), device=dev, dtype=torch.bfloat16)
    q_rstd = torch.full((S * B * H,), float("nan"), device=dev)
    out, lse = aiter.fmha_v3_fwd_qnorm_rope(bshd(q_raw), bshd(k), bshd(v), scale, wa, ta, wb, tb, ntile_a, eps,
                                            bshd(q_n), q_rstd)
    q_rstd2 = torch.empty_like(q_rstd)
    out2, lse2 = aiter.fmha_v3_fwd_qnorm_rope(bshd(q_raw), bshd(k), bshd(v), scale, wa, ta, wb, tb, ntile_a, eps,
                                              None, q_rstd2)
    torch.cuda.synchronize()
    res = dict(out=same(out, out_ref), lse=same(lse, lse_ref), q_n=same(q_n, q_ref), q_rstd=same(q_rstd, rstd_ref),
               layout=out.stride() == out_ref.stride() and lse.shape == lse_ref.shape,
               no_q_n=same(out2, out) and same(lse2, lse) and same(q_rstd2, q_rstd))

    # backward on the fused forward's (q_n, out, lse) vs on the reference's: deterministic mode bitwise; the v3 asm
    # path (non-deterministic mode, 16-bit dq atomics) bitwise for dk / dv, dq within the atomics' rounding
    dout = rn(B, S, H, D)
    for deterministic in (True, False):
        grads = []
        for qq, oo, ll in ((bshd(q_n), out, lse), (bshd(q_ref), out_ref, lse_ref)):
            dq, dk, dv = (torch.empty(B, S, H, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
            _flash_attn_backward(dout, qq, bshd(k), bshd(v), oo, ll, dq, dk, dv, None, 0.0, scale, False, -1, -1,
                                 None, None, deterministic, rng_state, False, 1)
            grads.append((dq, dk, dv))
        torch.cuda.synchronize()
        tag = "" if deterministic else "_v3"
        for i, name in enumerate(("dq", "dk", "dv")):
            a, b = grads[0][i], grads[1][i]
            if deterministic or name != "dq":
                res[name + tag] = same(a, b)
            else:
                res[name + tag] = ((a.float() - b.float()).abs().max() <= 2 ** -6 * b.float().abs().max()).item()
    return res


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("streams", [[512], [256, 256]], ids=["one_stream", "two_streams"])
def test_fmha_v3_fwd_qnorm_rope(streams):
    r = run(32, 512, 24, streams)
    assert all(r.values()), {k: v for k, v in r.items() if not v}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", type=int, default=32)
    ap.add_argument("-s", type=int, default=512)
    ap.add_argument("--heads", type=int, default=24)
    a = ap.parse_args()
    ok = True
    for name, streams in (("one stream", [a.s]), ("two streams", [256, a.s - 256])):
        r = run(a.b, a.s, a.heads, streams)
        print(f"B={a.b} S={a.s} H={a.heads} {name}: " + "  ".join(f"{k} {'OK' if v else 'FAIL'}" for k, v in r.items()))
        ok &= all(r.values())
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
