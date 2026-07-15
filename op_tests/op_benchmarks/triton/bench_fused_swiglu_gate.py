#!/usr/bin/env python3
"""Microbenchmark for fused MiniMax / GPT-OSS swiglu gate activation.

Compares the aiter ``fused_swiglu_gate`` kernel against the torch.compile
reference used by sglang (``swiglu_no_interleaved_with_alpha_and_limit``).

Typical MiniMax-M3 TP4 gate-up shape: ``(M * top_k, 768)`` bf16 in,
``(M * top_k, 384)`` bf16 out — one launch per MoE layer (60 layers/pass).
"""

import argparse
import sys

import torch
import triton

from aiter.ops.triton.fusions.fused_swiglu_gate import fused_swiglu_gate

# MiniMax-M3 TP4 local dims
_MINIMAX_LOCAL_D = 384
_MINIMAX_LAST_DIM = 768
_MINIMAX_TOP_K = 8
_MINIMAX_ALPHA = 1.702
_MINIMAX_LIMIT = 7.0


@torch.compile
def _compiled_swiglu_ref(x, alpha, limit):
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(gate * alpha) * (up + 1)


def _make_inputs(n_rows: int, dtype: torch.dtype):
    x = torch.randn(n_rows, _MINIMAX_LAST_DIM, device="cuda", dtype=dtype)
    out = torch.empty(n_rows, _MINIMAX_LOCAL_D, device="cuda", dtype=dtype)
    return x, out


def bench_fn(fn, warmup: int = 25, rep: int = 100) -> float:
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def run_point_bench(args):
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    n_rows = args.M * _MINIMAX_TOP_K
    x, out = _make_inputs(n_rows, dtype)

    compiled = _compiled_swiglu_ref(x, _MINIMAX_ALPHA, _MINIMAX_LIMIT)
    _ = fused_swiglu_gate(x, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, compiled.to(dtype), rtol=1e-2, atol=1e-2)

    ms_compile = bench_fn(
        lambda: _compiled_swiglu_ref(x, _MINIMAX_ALPHA, _MINIMAX_LIMIT),
        warmup=args.warmup,
        rep=args.rep,
    )
    ms_fused = bench_fn(
        lambda: fused_swiglu_gate(x, out),
        warmup=args.warmup,
        rep=args.rep,
    )
    ms_eager = bench_fn(
        lambda: torch_swiglu_eager(x, _MINIMAX_ALPHA, _MINIMAX_LIMIT),
        warmup=args.warmup,
        rep=args.rep,
    )

    us_compile = ms_compile * 1e3
    us_fused = ms_fused * 1e3
    us_eager = ms_eager * 1e3

    elem = x.element_size()
    mem_read = n_rows * _MINIMAX_LAST_DIM * elem
    mem_write = n_rows * _MINIMAX_LOCAL_D * elem
    mem_gb_s_fused = (mem_read + mem_write) / (ms_fused * 1e-3) / 1e9

    print(f"Shape: ({n_rows}, {_MINIMAX_LAST_DIM}) -> ({n_rows}, {_MINIMAX_LOCAL_D})")
    print(f"dtype={dtype}, alpha={_MINIMAX_ALPHA}, limit={_MINIMAX_LIMIT}")
    print(f"warmup={args.warmup}, rep={args.rep}")
    print()
    print(f"  eager torch:     {us_eager:7.2f} µs/call  ({ms_eager:8.4f} ms)")
    print(f"  torch.compile:   {us_compile:7.2f} µs/call  ({ms_compile:8.4f} ms)")
    print(f"  fused_swiglu_gate: {us_fused:7.2f} µs/call  ({ms_fused:8.4f} ms)")
    print(f"  speedup vs compile: {us_compile / us_fused:.2f}x")
    print(f"  bandwidth (fused): {mem_gb_s_fused:.1f} GB/s")
    if args.layers > 0:
        print(
            f"  est. {args.layers} layers/pass: "
            f"{ms_fused * args.layers:.3f} ms (fused) vs "
            f"{ms_compile * args.layers:.3f} ms (compile)"
        )


def torch_swiglu_eager(x, alpha, limit):
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(gate * alpha) * (up + 1)


def run_sweep(args):
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    m_list = [int(x) for x in args.M_list.split(",")]

    print(
        f"{'M':>8} {'rows':>10} {'eager_us':>10} {'compile_us':>12} "
        f"{'fused_us':>10} {'vs_compile':>12}"
    )
    for m in m_list:
        n_rows = m * _MINIMAX_TOP_K
        x, out = _make_inputs(n_rows, dtype)
        ms_eager = bench_fn(
            lambda: torch_swiglu_eager(x, _MINIMAX_ALPHA, _MINIMAX_LIMIT),
            warmup=args.warmup,
            rep=args.rep,
        )
        ms_compile = bench_fn(
            lambda: _compiled_swiglu_ref(x, _MINIMAX_ALPHA, _MINIMAX_LIMIT),
            warmup=args.warmup,
            rep=args.rep,
        )
        ms_fused = bench_fn(
            lambda: fused_swiglu_gate(x, out),
            warmup=args.warmup,
            rep=args.rep,
        )
        print(
            f"{m:8d} {n_rows:10d} {ms_eager * 1e3:10.2f} "
            f"{ms_compile * 1e3:12.2f} {ms_fused * 1e3:10.2f} "
            f"{ms_compile / ms_fused:12.2f}x"
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark fused_swiglu_gate (MiniMax-M3 TP4 gate activation)"
    )
    p.add_argument(
        "-M",
        type=int,
        default=4,
        help="Token count M (rows = M * top_k, default top_k=8)",
    )
    p.add_argument(
        "-M_list",
        type=str,
        default="1,4,32,256,8193,7238",
        help="Comma-separated M values for sweep mode",
    )
    p.add_argument(
        "--dtype",
        choices=["bf16", "fp16"],
        default="bf16",
    )
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument(
        "--layers",
        type=int,
        default=60,
        help="MoE layers per pass for aggregate time estimate (0 to skip)",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep multiple M values",
    )
    return p.parse_args()


def main():
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1
    args = parse_args()
    if args.sweep:
        run_sweep(args)
    else:
        run_point_bench(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
