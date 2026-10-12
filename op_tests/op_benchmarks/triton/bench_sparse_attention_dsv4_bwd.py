# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for the DSv4 sparse MLA training backward (gfx950 / CDNA4).

Reproduces the numbers quoted for `sparse_mla_bwd_dsv4`. The top-k is not uniform random:
it is a sliding window plus a causally-visible compressed pool, the distribution the V4
indexer actually produces. That matters here -- a uniform top-k gives every KV row about the
same number of contributors, while the real one gives the pool rows runs of a few thousand,
and the dKV gather is sized for those runs. Benchmarking on uniform indices flatters the
gather by roughly 2x.

`--breakdown` additionally times each phase on its own, which is how the per-kernel table in
the PR description was produced.

Usage:
  python op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4_bwd.py
  python op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4_bwd.py --breakdown
  python op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4_bwd.py --cfgs 4096,128,512
"""

import argparse

import torch
import triton

from aiter.ops.triton.attention.sparse_attention_dsv4_bwd import (
    _bwd_phases,
    _plan_bwd,
    sparse_mla_bwd_dsv4,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_mha_common import sparse_mla_dsv4_ref

D = 512
SWA = 128  # sliding-window width
CR_POOL = (
    4  # pool compression ratio: block b is visible to token t once (b+1)*CR_POOL-1 <= t
)


# ---------------------------------------------------------------------------
# Bench data builder
# ---------------------------------------------------------------------------
def _build_topk_swa_pool(T, topk, num_pool, device, generator):
    """[T, topk] int32 -- SWA(128) window plus causally-visible pool ranks, -1 padded.

    KV row layout is ``[ per-token 0..T-1 | pool T..T+num_pool-1 ]``. Early tokens see fewer
    than `topk - SWA` pool blocks, so their trailing slots stay -1, exactly as in production.

    ``topk`` must be at least the window width: the window is the floor of a V4 top-k, and a
    narrower request has no meaning here. Returning the full window anyway would report one
    ``topk`` in the results table while timing another.
    """
    assert topk >= SWA, f"topk={topk} is narrower than the SWA({SWA}) window"

    idx = torch.arange(T, device=device)
    off = torch.arange(SWA, device=device)
    swa = idx[:, None] - (SWA - 1) + off[None, :]
    swa = torch.where(swa >= 0, swa, torch.full_like(swa, -1))

    n_pool = topk - SWA
    if n_pool == 0 or num_pool == 0:
        return swa.to(torch.int32).contiguous()
    assert n_pool <= num_pool, f"need {n_pool} pool ranks but only {num_pool} blocks"

    n_visible = torch.clamp((idx + 1) // CR_POOL, min=0, max=num_pool)
    blk = torch.arange(num_pool, device=device)
    visible = blk[None, :] < n_visible[:, None]

    # Pick n_pool visible blocks at random: score the visible ones in [0,1), push the rest to
    # 2.0, take the smallest.
    score = torch.rand(T, num_pool, device=device, generator=generator)
    score = torch.where(visible, score, torch.full_like(score, 2.0))
    sel = score.argsort(dim=1)[:, :n_pool]
    pool = torch.where(torch.gather(visible, 1, sel), T + sel, torch.full_like(sel, -1))
    return torch.cat([swa, pool], dim=1).to(torch.int32).contiguous()


def _build_case(T, H, topk, device, num_pool=1024, seed=0):
    """Inputs for one configuration.

    `o` / `lse` are synthetic. No kernel branches on their values, so timing is unaffected and
    this avoids a reference forward that would dominate the run.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    indices = _build_topk_swa_pool(T, topk, num_pool, device, gen)
    num_kv = T + num_pool if topk > SWA else T

    def _bf16(*shape):
        return torch.randn(*shape, device=device, dtype=torch.bfloat16, generator=gen)

    return {
        "q": _bf16(T, H, D),
        "kv": _bf16(num_kv, D),
        "do": _bf16(T, H, D),
        "o": _bf16(T, H, D),
        "lse": torch.randn(T, H, device=device, dtype=torch.float32, generator=gen),
        "sink": (
            torch.randn(H, device=device, dtype=torch.float32, generator=gen) * 0.1
        ).contiguous(),
        "indices": indices,
        "num_kv": num_kv,
    }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def _flops(T, H, topk):
    # dQ contributes S, dP and dS@kv; dKV-interm contributes dS*Q and P*dO. Five 2*D-flop
    # products per (token, head, top-k slot).
    return 10.0 * D * T * H * topk


def _time_phases(case, scale):
    """Time each phase of the real pipeline.

    `_plan_bwd` and `_bwd_phases` are the same helpers `sparse_mla_bwd_dsv4` runs, so this cannot
    drift from the op: the tile widths, the workspace and the phase order all come from the
    wrapper rather than being restated here. Timing a phase repeatedly re-runs its side effects,
    which is harmless -- only the duration is read.
    """
    plan = _plan_bwd(
        case["q"],
        case["kv"],
        case["do"],
        case["o"],
        case["lse"],
        case["indices"],
        case["sink"],
        scale,
    )
    return [(name, triton.testing.do_bench(run)) for name, run in _bwd_phases(plan)]


# ---------------------------------------------------------------------------
# Correctness gate
# ---------------------------------------------------------------------------
def check_correctness(device):
    """Small-shape gate against autograd, so a broken kernel fails before it is timed."""
    print("\n========== CORRECTNESS ==========")
    T, H, topk = 128, 64, 128
    scale = 1.0 / (D**0.5)
    case = _build_case(T, H, topk, device, num_pool=64)
    q, kv, do, sink, indices = (
        case["q"],
        case["kv"],
        case["do"],
        case["sink"],
        case["indices"],
    )

    qg = q.float().clone().requires_grad_(True)
    kvg = kv.float().clone().requires_grad_(True)
    sg = sink.clone().requires_grad_(True)
    ref_o, lse = sparse_mla_dsv4_ref(qg, kvg, indices, sg, scale)
    o = ref_o.detach().to(torch.bfloat16).contiguous()

    dq, dkv, d_sink = sparse_mla_bwd_dsv4(
        q, kv, do, o, lse, indices, attn_sink=sink, scale=scale
    )
    ref_o.backward(do.float())

    def _cos(a, b):
        return torch.nn.functional.cosine_similarity(
            a.float().reshape(-1), b.float().reshape(-1), dim=0
        ).item()

    for name, got, want in (
        ("dq", dq, qg.grad),
        ("dkv", dkv, kvg.grad),
        ("d_sink", d_sink, sg.grad),
    ):
        cos = _cos(got, want)
        assert cos > 0.999, f"{name} cos={cos:.6f}"
        print(f"  {name:7s}: OK (cos={cos:.6f})")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_table(headers, rows):
    def _fmt(x):
        if isinstance(x, float):
            return f"{x:.3f}" if x >= 1 or x == 0 else f"{x:.4f}"
        return str(x)

    cells = [[_fmt(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(headers)]
    print("| " + " | ".join(h.rjust(widths[i]) for i, h in enumerate(headers)) + " |")
    print("| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |")
    for c in cells:
        print("| " + " | ".join(s.rjust(widths[i]) for i, s in enumerate(c)) + " |")


def run_bwd_bench(args, device):
    print("\n========== BACKWARD ==========")
    rows = []
    for T, H, topk in args.cfgs:
        case = _build_case(T, H, topk, device)
        scale = 1.0 / (D**0.5)
        tflops = _flops(T, H, topk) / 1e12

        ms = triton.testing.do_bench(
            lambda c=case, s=scale: sparse_mla_bwd_dsv4(
                c["q"],
                c["kv"],
                c["do"],
                c["o"],
                c["lse"],
                c["indices"],
                attn_sink=c["sink"],
                scale=s,
            )
        )
        rows.append([T, H, case["num_kv"], topk, ms, tflops / (ms * 1e-3)])

        if args.breakdown:
            print(f"\n  per-kernel, T={T} H={H} topk={topk}:")
            phases = _time_phases(case, scale)
            total = sum(t for _, t in phases)
            for name, t in phases:
                print(f"    {name:10s} {t:7.3f} ms  ({100 * t / total:4.1f}%)")
            print(
                f"    {'SUM':10s} {total:7.3f} ms  -> {tflops / (total * 1e-3):.0f} TFLOPS"
            )

    _print_table(["T", "H", "Kv", "topk", "ms", "TFLOPS"], rows)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cfgs",
        nargs="+",
        type=str,
        default=[  # (T, H, topk)
            "4096,128,512",
            "4096,128,1024",
            "8192,128,512",
        ],
        help="one or more shapes as T,H,topk (e.g. 4096,128,512). topk must be >= the sliding "
        "window (128); the remainder is drawn from the compressed pool.",
    )
    p.add_argument(
        "--breakdown",
        action="store_true",
        help="also time each phase separately (the per-kernel table in the PR description)",
    )
    p.add_argument(
        "--skip-correctness",
        action="store_true",
        help="skip the small-shape autograd check that otherwise runs before any timing.",
    )
    args = p.parse_args()
    args.cfgs = [tuple(int(x) for x in s.split(",")) for s in args.cfgs]
    return args


def main():
    args = _parse_args()
    device = "cuda"
    if arch_info.get_arch() != "gfx950":
        print(f"sparse_mla_bwd_dsv4 is gfx950 only; this is {arch_info.get_arch()}")
        return
    print(
        f"GPU: {torch.cuda.get_device_name(0)}  "
        f"({torch.cuda.get_device_properties(0).multi_processor_count} CUs)"
    )
    print(f"Triton: {triton.__version__}")

    if not args.skip_correctness:
        check_correctness(device)
    run_bwd_bench(args, device)


if __name__ == "__main__":
    main()
