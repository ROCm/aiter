# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Head-to-head benchmark: FlyDSL jagged_dense_bmm (dispatched, skew-aware) vs the
# upstream Meta/HSTU Triton kernel, over the 4 production "shapes of interest" in
# BOTH sequence-length regimes:
#
#   * UNIFORM   M_i == max_seq_len for every group (controlled baseline; the
#               static max-M grid has zero tail-tile waste).
#   * SKEWED    M_i = max_seq_len * U(0,1)**4 with ~20% empty groups + one full
#               and one near-full group (the real HSTU deployment distribution,
#               M_i ~ 0.95*Uniform(1,max_seq_len)). Here the FlyDSL dispatch
#               routes large shapes to the persistent problem-visitor kernel
#               (on-device CUM prefix, no host sync) which skips early-exit waste.
#
# The 4 shapes of interest (HSTU bench naming (B, D, K) -> GEMM: reduction K = D,
# output N = K_bench; max_seq_len is the jagged M envelope):
#   B1024_D256_K256 / B1024_D512_K512 / B120_D256_K256 / B120_D512_K512
#
# Timing = triton.testing.do_bench (CUDA-event wall-clock, many reps, autotuner
# allowed to settle). For FlyDSL the dispatched path includes the on-device CUM
# build on every call in the skew regime, so the reported number is END-TO-END
# (the production cost), not kernel-only.
#
# Run inside the devcontainer:
#   PYTHONPATH=/home/anguyenh/aiter:/home/anguyenh/generative-recommenders \
#     MPLBACKEND=Agg python op_tests/flydsl_tests/bench_jdbba_vs_triton.py -test
#
# Flags: --regime {uniform,skew,both} (default both), --metric {time,throughput,
#   bandwidth} (default time), -test (correctness vs torch eager), --mi N
#   (max_seq_len, default 7680), --seed S (skew RNG).

from __future__ import annotations

import argparse
import sys

import torch
import triton

import flydsl.compiler as flyc
from aiter.ops.flydsl.jagged_dense_bmm_dispatch_v2 import jagged_dense_bmm_dispatched

try:
    from generative_recommenders.ops.triton.triton_jagged import (
        triton_jagged_dense_bmm_add_fwd,
    )

    _HAS_TRITON = True
except Exception as _exc:  # pragma: no cover - environment dependent
    _HAS_TRITON = False
    _TRITON_ERR = _exc


# (B groups, D=reduction K, Kout=output N) — the 4 shapes of interest.
SHAPES = [
    (120, 256, 256),
    (120, 512, 512),
    (1024, 256, 256),
    (1024, 512, 512),
]


def _make_seqlens(B, max_seq_len, regime, seed):
    """Per-group sequence lengths. uniform = every group == max_seq_len; skew =
    max_seq_len*U**4 with ~20% empty groups + a full and a near-full group (the
    deployment distribution used across the jdbba skew experiments)."""
    if regime == "uniform":
        mi = [max_seq_len] * B
    else:
        g = torch.Generator().manual_seed(seed)
        u = torch.rand(B, generator=g)
        t = (max_seq_len * (u**4)).floor().to(torch.int64)
        t[: max(1, B // 5)] = 0          # ~20% empty groups
        t[-1] = max_seq_len              # one full-envelope group
        if B > 1:
            t[-2] = int(0.9 * max_seq_len)
        mi = [int(x) for x in t.tolist()]
    so = torch.zeros(B + 1, dtype=torch.int32)
    for i in range(B):
        so[i + 1] = so[i] + mi[i]
    return so, mi


def _make_inputs(B, D, Kout, max_seq_len, regime, seed, device="cuda"):
    N, K = Kout, D
    seq_offsets, mi = _make_seqlens(B, max_seq_len, regime, seed)
    L = int(seq_offsets[-1])
    torch.manual_seed(0)
    jagged = torch.randn(max(L, 1), K, dtype=torch.bfloat16, device=device)
    dense = torch.randn(B, K, N, dtype=torch.bfloat16, device=device)  # (B, K, N)
    bias = torch.randn(B, N, dtype=torch.bfloat16, device=device)
    return jagged, dense, bias, seq_offsets.to(device), L, N, K, mi


def _flydsl_fn(jagged, dense, bias, seq_offsets, B, max_seq_len, N, K, regime):
    dense_tall = dense.transpose(1, 2).reshape(B * N, K).contiguous()  # (B*N, K)
    bias_flat = bias.reshape(B * N).contiguous()
    L = jagged.shape[0]
    out = torch.zeros(L + 128, N, dtype=torch.bfloat16, device="cuda")
    tA = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)
    uniform = regime == "uniform"

    def fn():
        jagged_dense_bmm_dispatched(
            tC, tA, dense_tall, bias_flat, seq_offsets, B, max_seq_len,
            stream=torch.cuda.current_stream(), uniform_seqlen=uniform,
        )

    return fn, out, L


def _triton_fn(jagged, dense, bias, seq_offsets, max_seq_len):
    so64 = seq_offsets.to(torch.int64)

    def fn():
        return triton_jagged_dense_bmm_add_fwd(max_seq_len, so64, jagged, dense, bias)

    return fn


def _torch_reference(jagged, dense, bias, seq_offsets, N):
    L = jagged.shape[0]
    out = torch.zeros((L, N), dtype=torch.bfloat16, device=jagged.device)
    for b in range(dense.shape[0]):
        s = int(seq_offsets[b].item())
        e = int(seq_offsets[b + 1].item())
        if e > s:
            out[s:e] = (jagged[s:e].float() @ dense[b].float() + bias[b].float()[None, :]).to(torch.bfloat16)
    return out


def run_regime(regime, args):
    providers = ["flydsl"]
    if _HAS_TRITON:
        providers.append("triton")

    rows = []
    for (B, D, Kout) in SHAPES:
        jagged, dense, bias, seq_offsets, L, N, K, mi = _make_inputs(
            B, D, Kout, args.mi, regime, args.seed
        )
        nz = [x for x in mi if x > 0]
        empty_pct = 100.0 * (len(mi) - len(nz)) / len(mi)
        mean_mi = sum(mi) / len(mi)

        # FLOPs/bytes use the ACTUAL packed length L (sum of M_i), so skew vs
        # uniform are compared on the work each truly does.
        flops = 2.0 * L * D * N
        mem = (L * D + B * D * N + B * N + L * N) * 2

        rec = {"B": B, "D": D, "Kout": Kout, "L": L, "meanMi": mean_mi, "empty%": empty_pct}
        for prov in providers:
            if prov == "flydsl":
                fn, out, _ = _flydsl_fn(jagged, dense, bias, seq_offsets, B, args.mi, N, K, regime)
                out_view = out[:L]
            else:
                fn = _triton_fn(jagged, dense, bias, seq_offsets, args.mi)
                out_view = None

            if args.test:
                ref = _torch_reference(jagged, dense, bias, seq_offsets, N)
                if prov == "flydsl":
                    fn(); torch.cuda.synchronize(); got = out_view
                else:
                    got, _, _, _ = fn(); torch.cuda.synchronize()
                cos = torch.nn.functional.cosine_similarity(
                    ref.float().flatten(), got.float().flatten(), dim=0
                ).item()
                tag = "PASS" if cos > 0.999 else "FAIL"
                print(f"  {tag} [{prov:6s}] {regime:7s} (B={B}, D={D}, Kout={Kout})  cos={cos:.4f}")

            ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
            if args.metric == "time":
                rec[prov] = ms
            elif args.metric == "throughput":
                rec[prov] = flops / ms * 1e-9
            else:
                rec[prov] = mem / ms * 1e-6
        rows.append(rec)
    return providers, rows


def _print_table(regime, providers, rows, metric):
    unit = {"time": "ms", "throughput": "TFLOPS", "bandwidth": "GB/s"}[metric]
    print(f"\n=== {regime.upper()}  (metric={metric} [{unit}], lower-time/higher-tput = better) ===")
    hdr = f"{'shape':22s} {'meanMi':>7s} {'empty%':>6s}"
    for p in providers:
        hdr += f" {p+'('+unit+')':>15s}"
    if "flydsl" in providers and "triton" in providers:
        hdr += f" {'speedup(t/f)':>12s}"
    print(hdr)
    for r in rows:
        line = f"B{r['B']}_D{r['D']}_K{r['Kout']:<8d} {r['meanMi']:7.0f} {r['empty%']:6.0f}"
        for p in providers:
            line += f" {r[p]:15.4f}"
        if "flydsl" in providers and "triton" in providers:
            # speedup of flydsl over triton: time -> triton/flydsl; tput/bw -> flydsl/triton
            sp = (r["triton"] / r["flydsl"]) if metric == "time" else (r["flydsl"] / r["triton"])
            line += f" {sp:11.2f}x"
        print(line)


def main(argv=None):
    p = argparse.ArgumentParser(prog="jdbba FlyDSL-vs-Triton comparison (uniform + skew)")
    p.add_argument("--regime", choices=["uniform", "skew", "both"], default="both")
    p.add_argument("--metric", choices=["time", "throughput", "bandwidth"], default="time")
    p.add_argument("--mi", type=int, default=7680, help="max_seq_len envelope")
    p.add_argument("--seed", type=int, default=1234, help="skew RNG seed")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument("-test", action="store_true", help="correctness vs torch eager")
    args = p.parse_args(argv)

    if not _HAS_TRITON:
        print(f"WARNING: upstream Triton kernel unavailable: {_TRITON_ERR}")

    regimes = ["uniform", "skew"] if args.regime == "both" else [args.regime]
    for regime in regimes:
        providers, rows = run_regime(regime, args)
        _print_table(regime, providers, rows, args.metric)


if __name__ == "__main__":
    sys.exit(main())
