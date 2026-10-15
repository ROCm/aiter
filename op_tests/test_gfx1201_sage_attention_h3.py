# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Accuracy and contract tests for backend="asm_h3" of gfx1201_sage_attention.

The H3 code object keeps Q in BF16 and quantizes it per 32-row block inside the
kernel, mean-centres K before INT8 quantization and reads V as FP8 e4m3, so its
error profile differs from the "asm"/"hip" backends even though the op signature
is the same. It implements neither causal masking nor the fused norm/RoPE path.
"""

import sys

import torch
import torch.nn.functional as F

from aiter.jit.utils.chip_info import get_gfx
from aiter.test_mha_common import attention_ref

BACKEND = "asm_h3"
# Same tolerance band as test_gfx1201_sage_attention.py.
ATOL, RTOL, MAX_FAIL_PERCENT = 0.3, 0.25, 0.5
MIN_COSINE, MIN_ROW_COSINE_MEDIAN = 0.998, 0.99
MAX_REL_L2_VS_ASM = 1.3


def accuracy(out, ref):
    out, ref = out.float(), ref.float()
    abs_diff = (out - ref).abs()
    rel_diff = abs_diff / ref.clamp(min=1e-6).abs()
    row_cosine = F.cosine_similarity(out, ref, dim=-1).flatten()
    return {
        "fail%": ((abs_diff > ATOL) & (rel_diff > RTOL)).float().mean().item() * 100,
        "max_abs": abs_diff.max().item(),
        "cos": F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item(),
        "row_cos_median": row_cosine.median().item(),
        "rel_l2": ((out - ref).norm() / ref.norm()).item(),
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


def inputs(batch, rows, heads, scale=1.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        (
            torch.randn(batch, rows, heads, 128, device="cuda", generator=g) * scale
        ).bfloat16()
        for _ in range(3)
    )


def run(q, k, v, softmax_scale=None):
    return torch.ops.aiter.gfx1201_sage_attention(
        q, k, v, softmax_scale, backend=BACKEND
    )


def test_accuracy():
    ok = True
    for batch, rows, heads, scale in (
        (1, 256, 28, 1.0),
        (2, 1024, 3, 1.0),
        (1, 4096, 8, 1.0),
        (1, 4096, 8, 1.5),
        (2, 3000, 5, 2.0),
        (1, 16384, 2, 1.0),
    ):
        q, k, v = inputs(batch, rows, heads, scale)
        out = run(q, k, v)
        ok &= report(
            f"shape B{batch} S{rows} H{heads}",
            {"contiguous": float(out.is_contiguous())},
            out.shape == q.shape and out.dtype == q.dtype and out.is_contiguous(),
        )
        ok &= check_accuracy(
            f"B{batch} S{rows} H{heads} x{scale}", out, attention_ref(q, k, v)[0]
        )
    q, k, v = inputs(1, 777, 4)
    scale = 0.05
    ref = attention_ref(q * (scale * 128**0.5), k, v)[0]
    ok &= check_accuracy("softmax_scale=0.05", run(q, k, v, scale), ref)
    return ok


def test_tails():
    ok = True
    # The code object covers 256 query rows per workgroup and 16 key rows per V tile.
    for rows in (1, 15, 16, 17, 63, 64, 65, 255, 256, 257, 511, 1021, 1024, 1027):
        q, k, v = inputs(2, rows, 3, seed=rows)
        ok &= check_accuracy(f"tail S{rows}", run(q, k, v), attention_ref(q, k, v)[0])
        q.zero_()
        k.zero_()
        v.fill_(1)
        flat = run(q, k, v)
        ok &= report(
            f"uniform S{rows}",
            {"max_abs": (flat.float() - 1).abs().max().item()},
            (flat.float() - 1).abs().max() <= 8e-3,
        )
    return ok


def test_determinism_and_streams():
    ok = True
    args = inputs(2, 1000, 5, seed=7)
    expected = run(*args)
    for i in range(2):
        same = torch.equal(run(*args), expected)
        ok &= report(f"repeatable #{i}", {"bitwise": float(same)}, same)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = run(*args)
    torch.cuda.current_stream().wait_stream(stream)
    same = torch.equal(actual, expected)
    ok &= report("non-default stream", {"bitwise": float(same)}, same)
    results = torch.library.opcheck(
        torch.ops.aiter.gfx1201_sage_attention.default,
        inputs(2, 300, 3),
        {"backend": BACKEND},
    )
    passed = all(r == "SUCCESS" for r in results.values())
    ok &= report("opcheck", {"passed": float(passed)}, passed)
    return ok


def test_vs_block_quant_backend():
    """Keep the error gap to the "asm" backend in view.

    asm_h3 is the less accurate of the two: it shares one INT8 scale across 64 key
    rows where "asm" uses 32, and mean-centring K only partly pays that back. The
    measured ratio is ~1.14, so a regression past MAX_REL_L2_VS_ASM means the H3
    quantization changed, not that the two simply differ.
    """
    ok = True
    for rows, heads in ((1024, 5), (4096, 8)):
        q, k, v = inputs(1, rows, heads, seed=rows)
        ref = attention_ref(q, k, v)[0]
        h3 = accuracy(run(q, k, v), ref)
        asm = accuracy(
            torch.ops.aiter.gfx1201_sage_attention(q, k, v, backend="asm"), ref
        )
        ratio = h3["rel_l2"] / asm["rel_l2"]
        ok &= report(
            f"asm_h3 vs asm S{rows}",
            {"h3_rel_l2": h3["rel_l2"], "asm_rel_l2": asm["rel_l2"], "ratio": ratio},
            ratio <= MAX_REL_L2_VS_ASM,
        )
    return ok


def test_invalid_inputs():
    ok = True
    q, k, v = inputs(1, 64, 2)
    rope = torch.ones(64, 128, device="cuda")
    weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    cases = {
        "fp32": ((q.float(), k.float(), v.float()), {}),
        "non-contiguous": (
            (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)),
            {},
        ),
        "head dim 64": (
            (
                q[..., :64].contiguous(),
                k[..., :64].contiguous(),
                v[..., :64].contiguous(),
            ),
            {},
        ),
        "shorter K": ((q, k[:, :32].contiguous(), v[:, :32].contiguous()), {}),
        "requires_grad": ((q.clone().requires_grad_(), k, v), {}),
        "fused norm": ((q, k, v), {"query_weight": weight, "key_weight": weight}),
        "fused rope": ((q, k, v), {"cosine": rope, "sine": rope}),
    }
    for name, (args, kwargs) in cases.items():
        try:
            torch.ops.aiter.gfx1201_sage_attention(*args, backend=BACKEND, **kwargs)
            ok &= report(f"rejects {name}", {}, False)
        except (ValueError, RuntimeError):
            ok &= report(f"rejects {name}", {}, True)
    return ok


def main():
    if get_gfx() != "gfx1201":
        print(f"skip: gfx1201_sage_attention requires gfx1201, got {get_gfx()}")
        return 0
    import aiter  # noqa: F401  registers torch.ops.aiter.gfx1201_sage_attention

    ok = True
    for test in (
        test_accuracy,
        test_tails,
        test_determinism_and_streams,
        test_vs_block_quant_backend,
        test_invalid_inputs,
    ):
        ok &= test()
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
