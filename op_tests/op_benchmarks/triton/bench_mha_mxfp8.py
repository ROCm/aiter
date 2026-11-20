"""Microbenchmark for the fp8 MHA forward kernel in mha.py.

Timing uses ``triton.testing.do_bench_cudagraph``: the launch is captured into a
graph and replayed, so the numbers exclude per-call host overhead (the Triton
launcher, the argument marshalling in ``mha_fwd``) and measure the kernel
itself. Everything the timed callable touches is preallocated for the same
reason -- nothing but the launch happens inside the graph.

Usage:
    python bench_mha_fp8.py                          # default seqlen sweep
    python bench_mha_fp8.py --seqlen_q 4096 --batch 2
    python bench_mha_fp8.py --seqlen_q 1024 --seqlen_k 8192  # different q / k lengths
    python bench_mha_fp8.py --seqlen_q 1024 --seqlen_k 8192 --causal  # bottom-right causal
    python bench_mha_fp8.py --seqlen_q 8192 --seqlen_k 1024 --causal  # causal, SQ > SK
    python bench_mha_fp8.py --q_heads 32 --k_heads 8  # grouped query attention
    python bench_mha_fp8.py --batch 4 --seqlen_q 2048 --varlen  # varlen, "htd" layout
    python bench_mha_fp8.py --causal --window_left 1024  # causal sliding window
    python bench_mha_fp8.py --causal --sink  # attention sink for each query head
    python bench_mha_fp8.py --rep 100
"""

import argparse

import torch
import triton

from aiter.ops.triton.attention.mha_mxfp8 import mha_fwd
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def init_qkv(shape, quant_dim, device="cuda"):
    x = torch.randn(shape, device=device, dtype=torch.float32)
    x_quant, x_scale = downcast_to_mxfp(
        x.to(torch.bfloat16), torch.float8_e4m3fn, axis=quant_dim
    )
    if quant_dim != -1:
        x_quant = x_quant.contiguous()
        x_scale = x_scale.contiguous()
    return x_quant, x_scale


def num_visible_pairs(sq, sk, causal, window_left):
    # With the bottom-right causal mask, query i sees keys up to i + (SK - SQ),
    # and with a sliding window keys from i + (SK - SQ) - window_left.
    if not causal:
        return sq * sk
    num_pairs = 0
    for i in range(sq):
        hi = min(i + sk - sq, sk - 1)
        lo = max(i + sk - sq - window_left, 0) if window_left > 0 else 0
        num_pairs += max(hi - lo + 1, 0)
    return num_pairs


def bench_case(b, hq, hk, sq, sk, d, causal, window_left, sink, varlen, rep, device):
    if varlen:
        # b sequences of the same length, packed in the "htd" layout
        q_shape = (hq, b * sq, d)
        kv_shape = (hk, b * sk, d)
        cu_seqlens = torch.arange(0, b + 1, device=device, dtype=torch.int32)
        varlen_args = {
            "cu_seqlens_q": cu_seqlens * sq,
            "cu_seqlens_k": cu_seqlens * sk,
            "max_seqlen_q": sq,
            "max_seqlen_k": sk,
        }
    else:
        q_shape = (b, hq, sq, d)
        kv_shape = (b, hk, sk, d)
        varlen_args = {}
    q, q_scale = init_qkv(q_shape, -1, device)
    k, k_scale = init_qkv(kv_shape, -1, device)
    v, v_scale = init_qkv(kv_shape, -2, device)
    o = torch.empty(q_shape, device=device, dtype=torch.bfloat16)
    sink_logits = (
        torch.randn((hq,), device=device, dtype=torch.float32) if sink else None
    )

    def run():
        mha_fwd(
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
            o=o,
            causal=causal,
            window_size=(window_left, -1),
            sink=sink_logits,
            **varlen_args,
        )

    ms = triton.testing.do_bench_cudagraph(run, rep=rep)

    # Two matmuls per visible (query, key) pair and query head: Q @ K^T and P @ V.
    num_pairs = num_visible_pairs(sq, sk, causal, window_left)
    flops = 2 * (2.0 * b * hq * num_pairs * d)
    # fp8 q (hq heads, sq) and k/v (hk heads, sk) in, bf16 o (hq heads, sq) out.
    num_bytes = (b * hq * sq * d) + 2 * (b * hk * sk * d) + 2 * (b * hq * sq * d)
    return {
        "us": ms * 1e3,
        "tflops": flops * 1e-12 / (ms * 1e-3),
        "tbps": num_bytes * 1e-12 / (ms * 1e-3),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
        "--sink",
        action="store_true",
        default=False,
        help="add an attention sink logit for each query head",
    )
    parser.add_argument(
        "--window_left",
        type=int,
        default=0,
        help="left sliding window size; needs --causal. 0 means no window",
    )
    parser.add_argument(
        "--varlen",
        action="store_true",
        default=False,
        help='pack --batch sequences in the varlen "htd" layout',
    )
    parser.add_argument(
        "--rep", type=int, default=500, help="do_bench_cudagraph replay budget, in ms"
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
        args.window_left,
        args.sink,
        args.varlen,
        args.rep,
        device,
    )
    print(
        f"{args.batch:>4} {args.q_heads:>4} {k_heads:>4} {args.seqlen_q:>7} {seqlen_k:>7} {args.head_dim:>5} "
        f"{r['us']:>10.2f} {r['tflops']:>10.2f} {r['tbps']:>9.2f}"
    )


if __name__ == "__main__":
    main()
