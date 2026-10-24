# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import sys

import torch
import torch.nn.functional as F

from aiter.jit.utils.chip_info import get_gfx

LOG2E = 1.4426950408889634


def _quant_qk(x, rows, padded, sm_scale=None):
    batch, _, heads, dim = x.shape
    x = F.pad(x.float(), (0, 0, 0, 0, 0, padded - rows))
    if sm_scale is not None:
        x = x * torch.tensor(sm_scale, dtype=torch.float32, device=x.device)
    blocks = x.view(batch, padded // 32, 32, heads, dim)
    amax = blocks.abs().amax(dim=(2, 4))
    scale = torch.div(amax, torch.full_like(amax, 127.0))
    q = torch.div(blocks, scale[:, :, None, :, None])
    q = q + torch.where(q >= 0, 0.5, -0.5)
    q = torch.where(scale[:, :, None, :, None] > 0, q.trunc(), 0.0)
    return (
        q.to(torch.int8).view(batch, padded, heads, dim),
        scale.transpose(1, 2).contiguous(),
    )


def _quant_v(v, rows, padded):
    one = torch.ones((), dtype=torch.float32, device=v.device)
    scale = v.float().abs().amax(dim=1) * (one / 448.0)
    x = torch.where(scale[:, None] > 0, v.float() / scale[:, None], 0.0)
    x = F.pad(x, (0, 0, 0, 0, 0, padded - rows)).to(torch.float8_e4m3fn)
    return x.permute(0, 2, 3, 1).contiguous(), scale


def _norm_rope(x, weight, cosine, sine, eps):
    x = F.rms_norm(x, (x.shape[-1],), weight, eps)
    rope_dim = cosine.shape[-1]
    head = x[..., :rope_dim].float()
    rotated = torch.cat(
        [-head[..., rope_dim // 2 :], head[..., : rope_dim // 2]], dim=-1
    )
    head = head * cosine[None, :, None, :] + rotated * sine[None, :, None, :]
    return torch.cat([head.to(x.dtype), x[..., rope_dim:]], dim=-1)


def reference(q, k, v, sm_scale, qw=None, kw=None, cosine=None, sine=None, eps=1e-6):
    rows = q.shape[1]
    padded = (rows + 31) // 32 * 32
    if qw is not None:
        q = _norm_rope(q, qw, cosine, sine, eps)
        k = _norm_rope(k, kw, cosine, sine, eps)
    qi, qs = _quant_qk(q, rows, padded, sm_scale * LOG2E)
    ki, ks = _quant_qk(k, rows, padded)
    vf, vs = _quant_v(v, rows, padded)
    return qi, qs, ki, ks, vf, vs


def _rope_tables(rows, rope_dim, device):
    inv = 1.0 / (
        10000 ** (torch.arange(0, rope_dim, 2, device=device).float() / rope_dim)
    )
    angle = torch.arange(rows, device=device).float()[:, None] * inv[None, :]
    angle = torch.cat([angle, angle], dim=-1)
    return angle.cos().contiguous(), angle.sin().contiguous()


def check(name, got, want, exact):
    names = ("q_int8", "q_scale", "k_int8", "k_scale", "v_fp8", "v_scale")
    ok = True
    for label, a, b in zip(names, got, want):
        if a.shape != b.shape or a.dtype != b.dtype:
            print(
                f"  {name} {label}: shape/dtype {tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}"
            )
            ok = False
            continue
        if a.dtype in (torch.int8, torch.float8_e4m3fn):
            ai = a.view(torch.int8).int()
            bi = b.view(torch.int8).int()
            diff = (ai - bi).abs()
            bad = diff.max().item() if exact else (diff > 1).sum().item()
        else:
            bad = (
                (a != b).sum().item()
                if exact
                else ((a - b).abs() > 1e-6 * b.abs()).sum().item()
            )
        if bad:
            print(f"  {name} {label}: mismatch {bad}")
            ok = False
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main():
    if get_gfx() != "gfx1201":
        print(f"skip: gfx1201_sage_prepare requires gfx1201, got {get_gfx()}")
        return 0
    from aiter import gfx1201_sage_prepare

    torch.manual_seed(0)
    device = "cuda"
    ok = True
    for batch, rows, heads in (
        (1, 1, 1),
        (1, 33, 7),
        (2, 513, 14),
        (1, 4097, 28),
        (2, 2000, 3),
        (1, 20000, 56),
    ):
        q, k, v = (
            torch.randn(batch, rows, heads, 128, device=device, dtype=torch.bfloat16)
            * 3
            for _ in range(3)
        )
        k[:, :, 0] *= 1e-38
        v[:, :, :, 5] = 0
        if rows > 64:
            q[:, 32:64] = 0
        scale = 128**-0.5
        got = gfx1201_sage_prepare(q, k, v)
        ok &= check(
            f"quant B{batch} S{rows} H{heads}",
            got,
            reference(q, k, v, scale),
            exact=True,
        )
        torch.cuda.synchronize()
    for rows, heads, rope_dim in (
        (33, 7, 96),
        (4097, 28, 96),
        (513, 14, 128),
        (1000, 5, 64),
        (300, 3, 32),
    ):
        q, k, v = (
            torch.randn(1, rows, heads, 128, device=device, dtype=torch.bfloat16) * 3
            for _ in range(3)
        )
        qw, kw = (
            torch.rand(128, device=device, dtype=torch.bfloat16) + 0.5 for _ in range(2)
        )
        cosine, sine = _rope_tables(rows, rope_dim, device)
        got = gfx1201_sage_prepare(q, k, v, None, qw, kw, cosine, sine, 1e-5)
        want = reference(q, k, v, 128**-0.5, qw, kw, cosine, sine, 1e-5)
        ok &= check(f"norm_rope S{rows} H{heads} rope{rope_dim}", got, want, exact=True)
    stream = torch.cuda.Stream()
    q, k, v = (
        torch.randn(1, 777, 4, 128, device=device, dtype=torch.bfloat16)
        for _ in range(3)
    )
    with torch.cuda.stream(stream):
        got = gfx1201_sage_prepare(q, k, v)
    stream.synchronize()
    ok &= check("non-default stream", got, reference(q, k, v, 128**-0.5), exact=True)
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
