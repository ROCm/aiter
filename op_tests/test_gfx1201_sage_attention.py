# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import sys

import torch
import torch.nn.functional as F

from aiter.jit.utils.chip_info import get_gfx
from aiter.test_mha_common import attention_ref

# Same tolerance band as the Triton SageAttention test (op_tests/triton_tests/attention/test_fav3_sage.py).
ATOL, RTOL, MAX_FAIL_PERCENT = 0.3, 0.25, 0.5
MIN_COSINE, MIN_ROW_COSINE_MEDIAN = 0.998, 0.99
BACKENDS = ("asm", "hip")
# The code object rounds Q K^T differently from the HIP kernel on fast-path tiles (see
# gfx1201_sage_attention_fwd); measured asm-vs-hip relative L2 is ~2e-3 on randn inputs.
MAX_ASM_VS_HIP_REL_L2 = 5e-3


def accuracy(out, ref):
    out, ref = out.float(), ref.float()
    abs_diff = (out - ref).abs()
    rel_diff = abs_diff / ref.clamp(min=1e-6).abs()
    fail = ((abs_diff > ATOL) & (rel_diff > RTOL)).float().mean().item() * 100
    cosine = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    row_cosine = F.cosine_similarity(out, ref, dim=-1).flatten()
    rel_l2 = ((out - ref).norm() / ref.norm()).item()
    return {
        "fail%": fail,
        "max_abs": abs_diff.max().item(),
        "cos": cosine,
        "row_cos_median": row_cosine.median().item(),
        "row_cos_p10": (
            row_cosine.quantile(0.1).item()
            if row_cosine.numel() < 2**24
            else float("nan")
        ),
        "rel_l2": rel_l2,
    }


def report(name, stats, ok):
    text = " ".join(f"{k}={v:.4g}" for k, v in stats.items())
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {text}")
    return ok


def check_accuracy(name, out, ref):
    stats = accuracy(out, ref)
    ok = (
        stats["fail%"] <= MAX_FAIL_PERCENT
        and stats["cos"] >= MIN_COSINE
        and stats["row_cos_median"] >= MIN_ROW_COSINE_MEDIAN
        and torch.isfinite(out).all().item()
    )
    return report(name, stats, ok)


def rope_tables(rows, rope_dim, device):
    inv = 1.0 / (
        10000 ** (torch.arange(0, rope_dim, 2, device=device).float() / rope_dim)
    )
    angle = torch.arange(rows, device=device).float()[:, None] * inv[None, :]
    angle = torch.cat([angle, angle], dim=-1)
    return angle.cos().contiguous(), angle.sin().contiguous()


def norm_rope(x, weight, cosine, sine, eps):
    x = F.rms_norm(x, (x.shape[-1],), weight, eps)
    rope_dim = cosine.shape[-1]
    head = x[..., :rope_dim].float()
    rotated = torch.cat(
        [-head[..., rope_dim // 2 :], head[..., : rope_dim // 2]], dim=-1
    )
    head = head * cosine[None, :, None, :] + rotated * sine[None, :, None, :]
    return torch.cat([head.to(x.dtype), x[..., rope_dim:]], dim=-1)


def inputs(batch, rows, heads, scale=1.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        (
            torch.randn(batch, rows, heads, 128, device="cuda", generator=g) * scale
        ).bfloat16()
        for _ in range(3)
    )


def test_accuracy(backend):
    ok = True
    op = torch.ops.aiter.gfx1201_sage_attention
    for batch, rows, heads, scale in (
        (1, 128, 28, 1.0),
        (2, 1024, 3, 1.0),
        (1, 4096, 8, 1.0),
        (1, 4096, 8, 1.5),
        (2, 3000, 5, 2.0),
        (1, 16384, 2, 1.0),
    ):
        q, k, v = inputs(batch, rows, heads, scale)
        out = op(q, k, v, backend=backend)
        ref = attention_ref(q, k, v)[0]
        ok &= out.shape == q.shape and out.is_contiguous()
        ok &= check_accuracy(f"{backend} B{batch} S{rows} H{heads} x{scale}", out, ref)
    q, k, v = inputs(1, 777, 4)
    scale = 0.05
    ref = attention_ref(q * (scale * 128**0.5), k, v)[0]
    ok &= check_accuracy(
        f"{backend} softmax_scale=0.05", op(q, k, v, scale, backend=backend), ref
    )
    return ok


def test_tails(backend):
    ok = True
    op = torch.ops.aiter.gfx1201_sage_attention
    for rows in (1, 4, 15, 16, 17, 31, 32, 33, 63, 64, 65, 97, 511, 512, 513, 1025):
        q, k, v = inputs(2, rows, 3, seed=rows)
        out = op(q, k, v, backend=backend)
        ok &= check_accuracy(f"{backend} tail S{rows}", out, attention_ref(q, k, v)[0])
        q.zero_()
        k.zero_()
        v.fill_(1)
        flat = op(q, k, v, backend=backend)
        ok &= report(
            f"{backend} uniform S{rows}",
            {"max_abs": (flat.float() - 1).abs().max().item()},
            (flat.float() - 1).abs().max() <= 8e-3,
        )
    return ok


def test_fused_norm_rope():
    ok = True
    for rows, heads, rope_dim in ((4097, 28, 96), (513, 7, 128), (300, 3, 64)):
        q, k, v = inputs(1, rows, heads, 3.0, seed=rope_dim)
        qw, kw = (torch.rand(128, device="cuda").add(0.5).bfloat16() for _ in range(2))
        cosine, sine = rope_tables(rows, rope_dim, "cuda")
        fused = torch.ops.aiter.gfx1201_sage_attention(
            q, k, v, None, qw, kw, cosine, sine, 1e-5
        )
        q2, k2 = norm_rope(q, qw, cosine, sine, 1e-5), norm_rope(
            k, kw, cosine, sine, 1e-5
        )
        chain = torch.ops.aiter.gfx1201_sage_attention(q2, k2, v)
        same = torch.equal(fused.view(torch.int16), chain.view(torch.int16))
        ok &= report(
            f"fused == norm+rope+attention S{rows} H{heads} rope{rope_dim}",
            {"bitwise": float(same)},
            same,
        )
        ok &= check_accuracy(
            f"fused vs ref S{rows}", fused, attention_ref(q2, k2, v)[0]
        )
    return ok


def test_asm_vs_hip():
    ok = True
    op = torch.ops.aiter.gfx1201_sage_attention
    # x1: every tile on the fast path; x8: q_scale * k_scale > 1/384 everywhere -> exact path.
    for scale in (1.0, 8.0):
        q, k, v = inputs(2, 1000, 5, seed=7)
        q, k = q * scale, k * scale
        asm = op(q, k, v, backend="asm")
        hip = op(q, k, v, backend="hip")
        rel = ((asm.float() - hip.float()).norm() / hip.float().norm()).item()
        if scale == 1.0:
            ok &= report(
                "asm vs hip, fast path", {"rel_l2": rel}, rel <= MAX_ASM_VS_HIP_REL_L2
            )
        else:
            same = torch.equal(asm, hip)
            ok &= report("asm == hip, exact path", {"bitwise": float(same)}, same)
        same = torch.equal(asm, op(q, k, v, backend="asm"))
        ok &= report(f"asm repeatable x{scale}", {"bitwise": float(same)}, same)
    return ok


def test_registration_and_streams():
    ok = True
    args = inputs(2, 97, 3)
    results = torch.library.opcheck(
        torch.ops.aiter.gfx1201_sage_attention.default, args
    )
    ok &= report(
        "opcheck",
        {"passed": float(all(r == "SUCCESS" for r in results.values()))},
        all(r == "SUCCESS" for r in results.values()),
    )
    expected = torch.ops.aiter.gfx1201_sage_attention(*args)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = torch.ops.aiter.gfx1201_sage_attention(*args)
    torch.cuda.current_stream().wait_stream(stream)
    same = torch.equal(actual, expected)
    ok &= report("non-default stream", {"bitwise": float(same)}, same)
    return ok


def test_invalid_inputs():
    ok = True
    q, k, v = inputs(1, 64, 2)
    cases = {
        "fp32": (q.float(), k.float(), v.float()),
        "non-contiguous": (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)),
        "head dim 64": (
            q[..., :64].contiguous(),
            k[..., :64].contiguous(),
            v[..., :64].contiguous(),
        ),
        "shape mismatch": (q, k[:, :32].contiguous(), v),
        "requires_grad": (q.clone().requires_grad_(), k, v),
    }
    for name, args in cases.items():
        try:
            torch.ops.aiter.gfx1201_sage_attention(*args)
            ok &= report(f"rejects {name}", {}, False)
        except (ValueError, RuntimeError):
            ok &= report(f"rejects {name}", {}, True)
    try:
        torch.ops.aiter.gfx1201_sage_attention(q, k, v, backend="ck")
        ok &= report("rejects backend ck", {}, False)
    except (ValueError, RuntimeError):
        ok &= report("rejects backend ck", {}, True)
    return ok


def main():
    if get_gfx() != "gfx1201":
        print(f"skip: gfx1201_sage_attention requires gfx1201, got {get_gfx()}")
        return 0
    import aiter  # noqa: F401  registers torch.ops.aiter.gfx1201_sage_attention

    ok = True
    for backend in BACKENDS:
        ok &= test_accuracy(backend)
        ok &= test_tails(backend)
    for test in (
        test_asm_vs_hip,
        test_fused_norm_rope,
        test_registration_and_streams,
        test_invalid_inputs,
    ):
        ok &= test()
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
