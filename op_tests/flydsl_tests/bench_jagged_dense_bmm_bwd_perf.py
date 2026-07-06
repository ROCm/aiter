# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# triton.testing.perf_report-style benchmark for the BACKWARD pass of
# jagged_dense_bmm_broadcast_add (jdbba), the companion to
# bench_jagged_dense_bmm_perf.py. Reports time / throughput / bandwidth for both
# the upstream Meta/HSTU Triton backward kernels and the FlyDSL backward. The
# FlyDSL path is reached through its packaged aiter entrypoint
# (aiter.ops.flydsl.jagged_dense_bmm_bwd_dispatched) for the full backward, plus
# the raw per-component launchers for component-level (jagged / dense_bias) timing.
#
# Given the forward
#     Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]   per group b
# the backward of upstream gradient dOut (L, N) produces, per group b:
#     dJagged[s:e, :] = dOut[s:e, :] @ Dense[b].T        (M_b x K)
#     dDense[b]       = Jagged[s:e, :].T @ dOut[s:e, :]   (K x N)
#     dBias[b]        = sum_m dOut[s:e, :]                (N,)
#
# The upstream Triton path splits this across two functions:
#   triton_jagged_dense_bmm_add_bwd_jagged       -> dJagged
#   triton_jagged_dense_bmm_add_bwd_dense_bias   -> (dDense, dBias)
# which map onto the FlyDSL grad_jagged and the fused grad_dense_bias launcher
# (dBias is folded into the dDense partials pass) respectively. The --component flag
# selects which piece(s) to score so each FlyDSL kernel can be compared against its
# exact Triton counterpart.
#
# HSTU (B,D,K,N) bench naming -> GEMM dims: reduction K = bench D, output N =
# bench K (here Kout), grid M-envelope = bench N (max_seq_len, here Mi cap).
#
# Two sequence-length regimes (--regime): see the forward bench for the rationale
#   uniform : M_i == Mi for every group (controlled baseline; zero tail waste).
#   skew    : M_i = Mi * U(0,1)**4 with ~20% empty groups + one full and one
#             near-full group (the real HSTU deployment distribution).
#
# Run inside the devcontainer (torch/triton in the flydsl_venv):
#   PYTHONPATH=/workspaces/aiter:/workspaces/generative-recommenders:$PYTHONPATH \
#     python op_tests/flydsl_tests/bench_jagged_dense_bmm_bwd_perf.py --metric time -test
#     python op_tests/flydsl_tests/bench_jagged_dense_bmm_bwd_perf.py --regime both -test
#
# This uses triton.testing.do_bench (CUDA-event timing, many reps, L2-flushed).

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton

# Make `aiter` importable when this file is run directly as a script (sys.path[0]
# is then this dir, not the repo root). Repo root = .../meta/aiter (the dir that
# contains the `aiter` package), two levels up from op_tests/flydsl_tests/.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# The FlyDSL backward is imported the aiter way now (no sys.path hack): the
# packaged wrapper for the full backward, plus the kernel module for the raw
# per-component launchers + tiling constants. The kernel module resolves its
# forward constants dual-path, so importing it as an aiter package submodule
# works (the old bare-sibling shim is gone).
try:
    import flydsl.compiler as flyc  # noqa: E402
    from aiter.ops.flydsl import jagged_dense_bmm_bwd_dispatched as _fly_bwd_dispatched  # noqa: E402
    from aiter.ops.flydsl.kernels import jagged_dense_bmm_bwd as _bwd  # noqa: E402

    _HAS_FLYDSL = True
except Exception as _fexc:  # pragma: no cover - environment dependent
    _HAS_FLYDSL = False
    _FLYDSL_ERR = _fexc

try:
    from generative_recommenders.ops.triton.triton_jagged import (
        triton_jagged_dense_bmm_add_bwd_dense_bias,
        triton_jagged_dense_bmm_add_bwd_jagged,
    )

    _HAS_TRITON = True
except Exception as _exc:  # pragma: no cover - environment dependent
    _HAS_TRITON = False
    _TRITON_ERR = _exc


# Headline shapes: (B groups, D=reduction K, Kout=output N). Mi (max_seq_len) is
# a separate axis so it can be swept independently.
def default_benchmark_configs():
    return [
        (120, 256, 256),
        (120, 512, 512),
        (1024, 256, 256),
        (1024, 512, 512),
    ]


def _make_seq_offsets(B, Mi, regime, seed, device, sparsity=0.95):
    """Per-group prefix-sum offsets. uniform: every group == Mi. skew: Mi*U**4
    with ~20% empty groups + one full and one near-full group (deployment dist).
    genrec: M_i ~ Uniform(1, Mi) * sparsity, clamped >=1 -- mirrors the recsys
    harness common.gen_jagged_offsets so this bench is apples-to-apples with the
    genrec/HSTU jdbba sweeps (use the genrec seed, e.g. 1001, for an identical
    M_i instance). Here Mi is the ENVELOPE (max_seq_len), not the per-group len."""
    if regime == "uniform":
        return torch.arange(0, (B + 1) * Mi, Mi, dtype=torch.int32, device=device)
    if regime == "genrec":
        g = torch.Generator(device=device).manual_seed(seed)
        lengths = torch.randint(1, Mi + 1, size=(B,), device=device, generator=g)
        if sparsity < 1.0:
            lengths = (lengths.float() * sparsity).clamp(min=1.0).to(torch.int64)
        so = torch.zeros(B + 1, dtype=torch.int32, device=device)
        so[1:] = torch.cumsum(lengths, dim=0).to(torch.int32)
        return so
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(B, generator=g)
    t = (Mi * (u**4)).floor().to(torch.int64)
    t[: max(1, B // 5)] = 0          # ~20% empty groups
    t[-1] = Mi                       # one full-envelope group
    if B > 1:
        t[-2] = int(0.9 * Mi)        # one near-full group
    so = torch.zeros(B + 1, dtype=torch.int32)
    for i in range(B):
        so[i + 1] = so[i] + int(t[i])
    return so.to(device)


def _make_inputs(B, D, Kout, Mi, regime="uniform", seed=1234, device="cuda", sparsity=0.95):
    """Returns the tensors the backward kernels need for the given regime.

    jagged (L, K), dense (B, K, N), dOut (L, N), seq_offsets (B+1,). Naming mirrors
    the forward bench: reduction K = D, output N = Kout.
    """
    torch.manual_seed(0)
    N, K = Kout, D  # GEMM: output N = Kout, reduction K = D
    seq_offsets = _make_seq_offsets(B, Mi, regime, seed, device, sparsity=sparsity)
    L = int(seq_offsets[-1].item())

    jagged = torch.randn(max(L, 1), K, dtype=torch.bfloat16, device=device)
    dense = torch.randn(B, K, N, dtype=torch.bfloat16, device=device)  # (B, K, N)
    d_out = torch.randn(max(L, 1), N, dtype=torch.bfloat16, device=device)
    return jagged, dense, d_out, seq_offsets, L, N, K


def _triton_fn(jagged, dense, d_out, seq_offsets, B, Mi, N, K, component):
    """Build the do_bench closure for the requested backward component(s).

    component in {"jagged", "dense_bias", "all"}:
      jagged     -> dJagged via triton_jagged_dense_bmm_add_bwd_jagged
      dense_bias -> (dDense, dBias) via triton_jagged_dense_bmm_add_bwd_dense_bias
      all        -> both, the full backward
    """
    so64 = seq_offsets.to(torch.int64)

    def run_jagged():
        return triton_jagged_dense_bmm_add_bwd_jagged(
            Mi, so64, torch.empty_like(jagged), dense, d_out, K, B, N
        )

    def run_dense_bias():
        return triton_jagged_dense_bmm_add_bwd_dense_bias(
            Mi, so64, jagged, torch.empty_like(dense), B, K, N, d_out, False
        )

    if component == "jagged":
        return run_jagged
    if component == "dense_bias":
        return run_dense_bias

    def run_all():
        dj = run_jagged()
        dd, db = run_dense_bias()
        return dj, dd, db

    return run_all


def _flydsl_fn(jagged, dense, d_out, seq_offsets, B, Mi, N, K, component):
    """Build the do_bench closure for the FlyDSL backward component(s).

    D (= K = N) is a compile-time constant the kernels snapshot on the first
    launch, so it is pinned here via configure_dim(K). The bench therefore runs
    ONE D per process (custom -b/-d/-kout); a default multi-D sweep would trip the
    single-D snapshot on the second shape.

    Timing methodology:
      * component == "all" -> the packaged wrapper (jagged_dense_bmm_bwd_dispatched),
        i.e. the production path. It allocates its grad outputs per call, so this
        measures the real end-to-end op cost (alloc + all three launches), NOT the
        steady-state kernel-only time of the per-component closures.
      * component in {"jagged","dense_bias"} -> raw launchers over PRE-ALLOCATED
        buffers reused across iters, for steady-state kernel-only timing that is
        apples-to-apples with the split Triton kernels.
    """
    # Pin the square dim for this process (K == N == D here).
    _bwd.configure_dim(K)
    BLOCK_M = _bwd.BLOCK_M
    SPLIT = _bwd.SPLIT

    device = jagged.device
    total_rows = jagged.shape[0]
    stream = torch.cuda.current_stream()

    if component == "all":
        def run_all():
            return _fly_bwd_dispatched(
                jagged, dense, d_out, seq_offsets, n_groups=B, max_seq_len=Mi, stream=stream
            )

        return run_all

    tDOut = flyc.from_dlpack(d_out).mark_layout_dynamic(leading_dim=1, divisibility=8)

    if component == "jagged":
        # dJagged: RHS is Dense[b] in its plain (K, N) layout, flattened tall.
        dense_kn = dense.reshape(B * K, N).contiguous()
        d_jagged = torch.zeros(total_rows + BLOCK_M, K, dtype=torch.bfloat16, device=device)
        tDJ = flyc.from_dlpack(d_jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)

        def run_jagged():
            _bwd.grad_jagged(tDJ, tDOut, dense_kn, seq_offsets, B, Mi, stream=stream)
            return d_jagged[:total_rows]

        return run_jagged

    # component == "dense_bias": dDense (+ fused dBias) split-reduction.
    d_dense = torch.zeros(B, K, N, dtype=torch.bfloat16, device=device)
    d_dense_v = d_dense.view(B * K, N)
    dense_partials = torch.zeros(B * SPLIT * K, N, dtype=torch.float32, device=device)
    d_bias = torch.zeros(B, N, dtype=torch.bfloat16, device=device)
    bias_partials = torch.zeros(B * SPLIT, N, dtype=torch.float32, device=device)
    tJagged = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)

    def run_dense_bias():
        _bwd.grad_dense_bias(
            d_dense_v, d_bias, tJagged, tDOut, seq_offsets, dense_partials,
            bias_partials, B, Mi, stream=stream
        )
        return d_dense, d_bias

    return run_dense_bias


def _torch_reference(jagged, dense, d_out, seq_offsets, N, K):
    """Eager dJagged (L, K), dDense (B, K, N), dBias (B, N)."""
    L = jagged.shape[0]
    B = dense.shape[0]
    d_jagged = torch.zeros((L, K), dtype=torch.bfloat16, device=jagged.device)
    d_dense = torch.zeros((B, K, N), dtype=torch.bfloat16, device=jagged.device)
    d_bias = torch.zeros((B, N), dtype=torch.bfloat16, device=jagged.device)
    for b in range(B):
        s = int(seq_offsets[b].item())
        e = int(seq_offsets[b + 1].item())
        if e > s:
            go = d_out[s:e].float()
            d_jagged[s:e] = (go @ dense[b].float().t()).to(torch.bfloat16)
            d_dense[b] = (jagged[s:e].float().t() @ go).to(torch.bfloat16)
            d_bias[b] = go.sum(0).to(torch.bfloat16)
    return d_jagged, d_dense, d_bias


def _check(component, got, ref):
    """Cosine-similarity check of the component's output(s) vs eager reference.

    ref is (d_jagged, d_dense, d_bias). Returns min cosine across checked tensors.
    """
    d_jagged_ref, d_dense_ref, d_bias_ref = ref

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        ).item()

    if component == "jagged":
        return cos(got, d_jagged_ref)
    if component == "dense_bias":
        dd, db = got
        return min(cos(dd, d_dense_ref), cos(db, d_bias_ref))
    # all
    dj, dd, db = got
    return min(cos(dj, d_jagged_ref), cos(dd, d_dense_ref), cos(db, d_bias_ref))


def _flops_bytes(component, L, B, D, N):
    """Per-component FLOPs and bf16 byte traffic (using packed length L = sum M_i).

    dJagged: dOut(L,N) @ Dense[b].T -> 2*L*D*N flops; reads dOut + Dense, writes dJagged.
    dDense : Jagged[b].T @ dOut(L,N) -> 2*L*D*N flops; reads Jagged + dOut, writes dDense (+ dBias).
    """
    f_jag = 2.0 * L * D * N
    f_db = 2.0 * L * D * N  # dDense GEMM; dBias reduction (L*N adds) is negligible
    m_jag = (L * N + B * D * N + L * D) * 2          # read dOut + Dense, write dJagged
    m_db = (L * D + L * N + B * D * N + B * N) * 2   # read Jagged + dOut, write dDense + dBias
    if component == "jagged":
        return f_jag, m_jag
    if component == "dense_bias":
        return f_db, m_db
    return f_jag + f_db, m_jag + m_db


def _run_one_regime(custom, args, regime, providers, unit):
    if custom:
        x_vals = [(args.b, args.d, args.kout)]
    else:
        x_vals = default_benchmark_configs()
    x_vals = [(*v, args.mi) for v in x_vals]

    config = triton.testing.Benchmark(
        x_names=["B", "D", "KOUT", "MI"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=providers,
        line_names=providers,
        styles=[("blue", "-"), ("red", "-")][: len(providers)],
        ylabel=unit,
        plot_name=f"bench_jagged_dense_bmm_bwd_{args.component}_{regime}_{args.metric}",
        args={"regime": regime},
    )

    @triton.testing.perf_report([config])
    def bench_fn(B, D, KOUT, MI, provider, regime):
        jagged, dense, d_out, seq_offsets, L, N, K = _make_inputs(
            B, D, KOUT, MI, regime=regime, seed=args.seed, sparsity=args.sparsity
        )

        if provider == "triton":
            fn = _triton_fn(jagged, dense, d_out, seq_offsets, B, MI, N, K, args.component)
        elif provider == "flydsl":
            fn = _flydsl_fn(jagged, dense, d_out, seq_offsets, B, MI, N, K, args.component)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        if args.test:
            ref = _torch_reference(jagged, dense, d_out, seq_offsets, N, K)
            got = fn()
            torch.cuda.synchronize()
            cos = _check(args.component, got, ref)
            tag = f"({regime:7s} B={B}, D={D}, Kout={KOUT}, Mi={MI})"
            print(f"  {'PASS' if cos > 0.999 else 'FAIL'} [{provider:6s}] {tag}  cos={cos:.4f}")

        ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)

        # FLOPs/bytes use the ACTUAL packed length L (sum of M_i), so skew and
        # uniform are each scored on the work they truly do.
        flops, mem = _flops_bytes(args.component, L, B, D, N)

        if args.metric == "time":
            return ms
        elif args.metric == "throughput":
            return flops / ms * 1e-9
        elif args.metric == "bandwidth":
            return mem / ms * 1e-6
        raise ValueError(f"Unknown metric: {args.metric}")

    print(f"\n=== regime={regime}  component={args.component}  (metric={args.metric} [{unit}]) ===")
    bench_fn.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(custom, args):
    providers = []
    if _HAS_FLYDSL and not args.triton_only:
        providers.append("flydsl")
    if _HAS_TRITON and not args.flydsl_only:
        providers.append("triton")
    if not providers:
        print("No providers selected / available.")
        return

    unit = {"time": "ms", "throughput": "TFLOPS", "bandwidth": "GB/s"}[args.metric]
    regimes = ["uniform", "skew"] if args.regime == "both" else [args.regime]
    for regime in regimes:
        _run_one_regime(custom, args, regime, providers, unit)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="Benchmark jagged_dense_bmm backward (jdbba bwd): Triton",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--metric", choices=["time", "throughput", "bandwidth"], default="time")
    p.add_argument("--component", choices=["jagged", "dense_bias", "all"], default="all",
                   help="which backward output(s) to score: dJagged / (dDense,dBias) / both")
    p.add_argument("--regime", choices=["uniform", "skew", "genrec", "both"], default="uniform",
                   help="sequence-length distribution (uniform baseline / skewed deployment / "
                        "genrec Uniform(1,Mi)*sparsity apples-to-apples with the recsys sweeps / both)")
    p.add_argument("--seed", type=int, default=1234, help="skew/genrec RNG seed (use 1001 to match genrec)")
    p.add_argument("--sparsity", type=float, default=0.95,
                   help="genrec regime: scale M_i ~ Uniform(1,Mi) by this factor (clamped >=1)")
    p.add_argument("-b", type=int, default=0, help="B groups (custom shape)")
    p.add_argument("-d", type=int, default=0, help="D = reduction K (custom shape)")
    p.add_argument("-kout", type=int, default=0, help="Kout = output N (custom shape)")
    p.add_argument("-mi", type=int, default=7680, help="max_seq_len (per-group rows in uniform; envelope in skew)")
    p.add_argument("--warmup", type=int, default=25, help="do_bench warmup ms")
    p.add_argument("--rep", type=int, default=100, help="do_bench rep ms")
    p.add_argument("-test", action="store_true", help="correctness check vs torch eager")
    p.add_argument("-o", action="store_true", help="save CSV/plot")
    p.add_argument("--flydsl-only", action="store_true", help="score only the FlyDSL provider")
    p.add_argument("--triton-only", action="store_true", help="score only the Triton provider")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not _HAS_TRITON:
        print(f"WARNING: upstream Triton kernel unavailable: {_TRITON_ERR}")
    if not _HAS_FLYDSL:
        print(f"WARNING: FlyDSL backward kernels unavailable: {_FLYDSL_ERR}")
    custom = bool(args.b or args.d or args.kout)
    if custom:
        assert args.b and args.d and args.kout, "custom shape needs -b, -d, -kout"
    run_benchmark(custom, args)


if __name__ == "__main__":
    sys.exit(main())
