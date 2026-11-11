"""Microbenchmark for the fp8 MHA forward kernel in mha.py.

Timing uses ``triton.testing.do_bench_cudagraph``: the launch is captured into a
graph and replayed, so the numbers exclude per-call host overhead (the Triton
launcher, the argument marshalling in ``mha_fwd``) and measure the kernel
itself. Everything the timed callable touches is preallocated for the same
reason -- nothing but the launch happens inside the graph.

The mla_varlen variant uses ``RotatingBufferBench``: each captured call reads a
different copy of the inputs, so the inputs are not in the cache.

Usage:
    python bench_mha_fp8.py                          # default seqlen sweep
    python bench_mha_fp8.py --seqlen_q 4096 --batch 2
    python bench_mha_fp8.py --seqlen_q 1024 --seqlen_k 8192  # different q / k lengths
    python bench_mha_fp8.py --seqlen_q 1024 --seqlen_k 8192 --causal  # bottom-right causal
    python bench_mha_fp8.py --q_heads 32 --k_heads 8  # grouped query attention
    python bench_mha_fp8.py --rep 100
    python bench_mha_fp8.py --variant mla_varlen --q_heads 24 --batch 4 --seqlen_q 2048 --causal
        # MLA prefill: q [B * SQ, HQ, 192], k [B * SK, HK, 128] + k_pe [B * SK, 1, 64],
        # v [B * SK, HK, 128], packed varlen sequences
"""

import argparse

import torch
import triton

from aiter.ops.triton.attention.mha_fp8 import mha_fwd, mha_fwd_mla_varlen
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def quant_fp8(x):
    """Per-tensor fp8 e4m3 quantization; returns ``(fp8 tensor, fp32 scale)``."""
    scale = x.abs().amax().to(torch.float32) / FP8_MAX
    return (x / scale).to(FP8_DTYPE), scale


def make_inputs(b, hq, hk, sq, sk, d, device):
    q_shape = (b, hq, sq, d)
    kv_shape = (b, hk, sk, d)
    q, q_scale = quant_fp8(torch.randn(q_shape, device=device, dtype=torch.float32))
    k, k_scale = quant_fp8(torch.randn(kv_shape, device=device, dtype=torch.float32))
    v, v_scale = quant_fp8(torch.randn(kv_shape, device=device, dtype=torch.float32))
    o = torch.empty(q_shape, device=device, dtype=torch.bfloat16)
    return q, k, v, q_scale, k_scale, v_scale, o


def bench_case(b, hq, hk, sq, sk, d, causal, rep, device):
    q, k, v, q_scale, k_scale, v_scale, o = make_inputs(b, hq, hk, sq, sk, d, device)

    def run():
        mha_fwd(q, k, v, q_scale, k_scale, v_scale, o=o, causal=causal)

    ms = triton.testing.do_bench_cudagraph(run, rep=rep)

    # Two SQ x SK x D matmuls per (batch, query head): Q @ K^T and P @ V.
    # With the bottom-right causal mask, query i sees i + (SK - SQ) + 1 keys,
    # so only SQ * SK - SQ * (SQ - 1) / 2 (query, key) pairs are computed.
    # num_pairs = sq * sk - sq * (sq - 1) / 2 if causal else sq * sk
    if causal and sk > sq:
        num_pairs = sq * sk - sq * (sq - 1) / 2
    elif causal:
        num_pairs = sq * sk / 2
    else:
        num_pairs = sq * sk
    flops = 2 * (2.0 * b * hq * num_pairs * d)
    # fp8 q (hq heads, sq) and k/v (hk heads, sk) in, bf16 o (hq heads, sq) out.
    num_bytes = (b * hq * sq * d) + 2 * (b * hk * sk * d) + 2 * (b * hq * sq * d)
    return {
        "us": ms * 1e3,
        "tflops": flops * 1e-12 / (ms * 1e-3),
        "tbps": num_bytes * 1e-12 / (ms * 1e-3),
    }


def num_pairs(sq, sk, causal):
    """(query, key) pairs one sequence computes under the bottom-right causal mask."""
    if causal and sk > sq:
        return sq * sk - sq * (sq - 1) / 2
    if causal:
        return sq * sk / 2
    return sq * sk


def bench_mla_varlen_case(b, hq, hk, sq, sk, causal, kernels_per_graph, device):
    """mha_fwd_mla_varlen on b sequences of sq queries and sk keys."""
    d, d_pe, dv = 128, 64, 128

    def make_inputs():
        q = torch.randn((b * sq, hq, d + d_pe), device=device, dtype=torch.float32)
        q = q.to(FP8_DTYPE)
        # k and v are views of one buffer and k_pe a view of the latent, as in vLLM.
        kv = torch.randn((b * sk, hk, d + dv), device=device, dtype=torch.float32)
        k, v = kv.to(FP8_DTYPE).split([d, dv], dim=-1)
        latent = torch.randn((b * sk, 512 + d_pe), device=device, dtype=torch.float32)
        k_pe = latent.to(FP8_DTYPE)[:, 512:].unsqueeze(1)
        o = torch.empty((b * sq, hq, dv), device=device, dtype=torch.bfloat16)
        return q, k, v, k_pe, o

    cu_q = torch.arange(0, b + 1, device=device, dtype=torch.int32) * sq
    cu_k = torch.arange(0, b + 1, device=device, dtype=torch.int32) * sk
    bencher = RotatingBufferBench()
    bencher.build_pool([make_inputs], [("q", "k", "v", "k_pe", "o")])

    def run(q, k, v, k_pe, o):
        mha_fwd_mla_varlen(q, k, v, k_pe, cu_q, cu_k, sq, sk, causal=causal, out=o)

    ms = bencher.bench(run, kernels_per_graph=kernels_per_graph)
    flops = 2.0 * b * hq * num_pairs(sq, sk, causal) * (d + d_pe + dv)
    num_bytes = (
        b * sq * hq * (d + d_pe)
        + b * sk * (hk * (d + dv) + d_pe)
        + 2 * b * sq * hq * dv
    )
    return {
        "us": ms * 1e3,
        "tflops": flops * 1e-12 / (ms * 1e-3),
        "tbps": num_bytes * 1e-12 / (ms * 1e-3),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--variant",
        choices=("default", "mla_varlen"),
        default="default",
        help="default: mha_fwd on [B, H, S, D]; mla_varlen: mha_fwd_mla_varlen",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q_heads", type=int, default=32, help="number of query heads")
    parser.add_argument(
        "--k_heads",
        type=int,
        default=None,
        help="number of K/V heads for GQA; defaults to --q-_heads (plain MHA)",
    )
    parser.add_argument(
        "--seqlen_q", type=int, default=8192, help="query sequence length"
    )
    parser.add_argument(
        "--seqlen_k",
        type=int,
        default=None,
        help="key/value sequence length; defaults to --seqlen_q",
    )
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--causal", action="store_true", default=False)
    parser.add_argument(
        "--rep", type=int, default=500, help="do_bench_cudagraph replay budget, in ms"
    )
    parser.add_argument(
        "--kernels_per_graph",
        type=int,
        default=100,
        help="mla_varlen: calls in each RotatingBufferBench CUDA graph",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "bench_mha.py needs a GPU"
    torch.manual_seed(0)
    device = torch.device("cuda")

    k_heads = args.q_heads if args.k_heads is None else args.k_heads
    assert (
        args.q_heads % k_heads == 0
    ), f"--q_heads ({args.q_heads}) must be a multiple of --k-heads ({k_heads})"

    seqlen_k = args.seqlen_q if args.seqlen_k is None else args.seqlen_k
    assert (
        not args.causal or seqlen_k >= args.seqlen_q
    ), f"--causal requires --seqlen_k ({seqlen_k}) >= --seqlen_q ({args.seqlen_q})"

    if args.variant == "mla_varlen":
        print("MHA fwd fp8, variant mla_varlen (qk 128 + 64 PE, v 128)")
        print(
            f"{'B':>4} {'HQ':>4} {'HK':>4} {'SQ':>7} {'SK':>7} {'us':>10} {'TFLOP/s':>10} {'TB/s':>9}"
        )
        r = bench_mla_varlen_case(
            args.batch,
            args.q_heads,
            k_heads,
            args.seqlen_q,
            seqlen_k,
            args.causal,
            args.kernels_per_graph,
            device,
        )
        print(
            f"{args.batch:>4} {args.q_heads:>4} {k_heads:>4} {args.seqlen_q:>7} {seqlen_k:>7} "
            f"{r['us']:>10.2f} {r['tflops']:>10.2f} {r['tbps']:>9.2f}"
        )
        return

    print("MHA fwd fp8")
    print(
        f"{'B':>4} {'HQ':>4} {'HK':>4} {'SQ':>7} {'SK':>7} {'D':>5} {'us':>10} {'TFLOP/s':>10} {'TB/s':>9}"
    )
    r = bench_case(
        args.batch,
        args.q_heads,
        k_heads,
        args.seqlen_q,
        seqlen_k,
        args.head_dim,
        args.causal,
        args.rep,
        device,
    )
    print(
        f"{args.batch:>4} {args.q_heads:>4} {k_heads:>4} {args.seqlen_q:>7} {seqlen_k:>7} {args.head_dim:>5} "
        f"{r['us']:>10.2f} {r['tflops']:>10.2f} {r['tbps']:>9.2f}"
    )


if __name__ == "__main__":
    main()
