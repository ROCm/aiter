"""Microbenchmark for the MXFP8 MHA forward kernel in mha_mxfp8.py (gfx1250 only).

Timing uses ``RotatingBufferBench``: the launches are captured into a CUDA graph
that rotates over a pool of input sets, so the numbers exclude per-call host
overhead and do not time cache-resident inputs. The output buffer is
preallocated in each pool entry, so the graph holds only the launches.

Usage:
    python bench_mha_mxfp8.py                          # default shape
    python bench_mha_mxfp8.py --seqlen_q 4096 --batch 2
    python bench_mha_mxfp8.py --seqlen_q 1024 --seqlen_k 8192  # different q / k lengths
    python bench_mha_mxfp8.py --seqlen_q 1024 --seqlen_k 8192 --causal  # bottom-right causal
    python bench_mha_mxfp8.py --q_heads 32 --k_heads 8  # grouped query attention
"""

import argparse
import functools

import torch

from aiter.ops.triton.attention.mha_mxfp8 import mha_fwd
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.attention.test_mha_mxfp8 import init_qkv

_POOL_KEYS = ["q", "k", "v", "q_scale", "k_scale", "v_scale", "o"]


def _generate_inputs(b, hq, hk, sq, sk, d):
    """Quantized Q/K/V from the unit test, and a preallocated bf16 output."""
    q_shape = (b, hq, sq, d)
    kv_shape = (b, hk, sk, d)
    # init_qkv also returns the dequantized input; the kernel does not read it.
    _, q, q_scale = init_qkv(q_shape, -1)
    _, k, k_scale = init_qkv(kv_shape, -1)
    _, v, v_scale = init_qkv(kv_shape, -2)
    o = torch.empty(q_shape, device="cuda", dtype=torch.bfloat16)
    return q, k, v, q_scale, k_scale, v_scale, o


def bench_case(bencher, b, hq, hk, sq, sk, d, causal):
    bencher.build_pool(
        [functools.partial(_generate_inputs, b, hq, hk, sq, sk, d)], [_POOL_KEYS]
    )

    def run(q, k, v, q_scale, k_scale, v_scale, o):
        mha_fwd(q, k, v, q_scale, k_scale, v_scale, o=o, causal=causal)

    ms = bencher.bench(run)

    # Two SQ x SK x D matmuls per (batch, query head): Q @ K^T and P @ V.
    # With the bottom-right causal mask, query i sees i + (SK - SQ) + 1 keys,
    # so only SQ * SK - SQ * (SQ - 1) / 2 (query, key) pairs are computed.
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


def main():
    parser = get_parser(kernel_name="MXFP8 MHA forward")
    parser.description = __doc__
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q_heads", type=int, default=32, help="number of query heads")
    parser.add_argument(
        "--k_heads",
        type=int,
        default=None,
        help="number of K/V heads for GQA; defaults to --q_heads (plain MHA)",
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
    args = parser.parse_args()

    assert torch.cuda.is_available(), "bench_mha_mxfp8.py needs a GPU"
    torch.manual_seed(0)

    k_heads = args.q_heads if args.k_heads is None else args.k_heads
    assert (
        args.q_heads % k_heads == 0
    ), f"--q_heads ({args.q_heads}) must be a multiple of --k_heads ({k_heads})"

    seqlen_k = args.seqlen_q if args.seqlen_k is None else args.seqlen_k
    assert (
        not args.causal or seqlen_k >= args.seqlen_q
    ), f"--causal requires --seqlen_k ({seqlen_k}) >= --seqlen_q ({args.seqlen_q})"

    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    print("MHA fwd mxfp8")
    print(
        f"{'B':>4} {'HQ':>4} {'HK':>4} {'SQ':>7} {'SK':>7} {'D':>5} {'us':>10} {'TFLOP/s':>10} {'TB/s':>9}"
    )
    r = bench_case(
        bencher,
        args.batch,
        args.q_heads,
        k_heads,
        args.seqlen_q,
        seqlen_k,
        args.head_dim,
        args.causal,
    )
    print(
        f"{args.batch:>4} {args.q_heads:>4} {k_heads:>4} {args.seqlen_q:>7} {seqlen_k:>7} {args.head_dim:>5} "
        f"{r['us']:>10.2f} {r['tflops']:>10.2f} {r['tbps']:>9.2f}"
    )


if __name__ == "__main__":
    main()
