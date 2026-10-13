"""Microbenchmark for the fp8 MHA forward kernel in mha.py.

Timing uses ``triton.testing.do_bench_cudagraph``: the launch is captured into a
graph and replayed, so the numbers exclude per-call host overhead (the Triton
launcher, the argument marshalling in ``mha_fwd``) and measure the kernel
itself. Everything the timed callable touches is preallocated for the same
reason -- nothing but the launch happens inside the graph.

Usage:
    python bench_mha.py                          # default seqlen sweep
    python bench_mha.py --seqlen 4096 --batch 2
    python bench_mha.py --rep 100
"""

import argparse

import torch
import triton

from aiter.ops.triton.attention.mha_fp8 import mha_fwd


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def quant_fp8(x):
    """Per-tensor fp8 e4m3 quantization; returns ``(fp8 tensor, fp32 scale)``."""
    scale = x.abs().amax().to(torch.float32) / FP8_MAX
    return (x / scale).to(FP8_DTYPE), scale


def make_inputs(b, h, s, d, device):
    shape = (b, h, s, d) 
    q, q_scale = quant_fp8(torch.randn(shape, device=device, dtype=torch.float32))
    k, k_scale = quant_fp8(torch.randn(shape, device=device, dtype=torch.float32))
    v, v_scale = quant_fp8(torch.randn(shape, device=device, dtype=torch.float32))
    o = torch.empty(shape, device=device, dtype=torch.bfloat16)
    return q, k, v, q_scale, k_scale, v_scale, o


def bench_case(b, h, s, d, causal, rep, device):
    q, k, v, q_scale, k_scale, v_scale, o = make_inputs(b, h, s, d, device)

    def run():
        mha_fwd(q, k, v, q_scale, k_scale, v_scale, o=o, causal=causal)

    ms = triton.testing.do_bench_cudagraph(run, rep=rep)

    # Two S x S x D matmuls per (batch, head): Q @ K^T and P @ V.
    flops = 2 * (2.0 * b * h * s * s * d)
    if causal:
        flops = flops // 2
    # fp8 q/k/v in, bf16 o out.
    num_bytes = 3 * (b * h * s * d) + 2 * (b * h * s * d)
    return {
        "us": ms * 1e3,
        "tflops": flops * 1e-12 / (ms * 1e-3),
        "tbps": num_bytes * 1e-12 / (ms * 1e-3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--causal", action="store_true", default=False)
    parser.add_argument("--rep", type=int, default=500,
                        help="do_bench_cudagraph replay budget, in ms")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "bench_mha.py needs a GPU"
    torch.manual_seed(0)
    device = torch.device("cuda")

    print(f"fp8 MHA fwd")
    print(f"{'B':>4} {'H':>4} {'S':>7} {'D':>5} {'us':>10} {'TFLOP/s':>10} {'TB/s':>9}")
    r = bench_case(args.batch, args.heads, args.seqlen, args.head_dim, args.causal, args.rep, device)
    print(f"{args.batch:>4} {args.heads:>4} {args.seqlen:>7} {args.head_dim:>5} "
            f"{r['us']:>10.2f} {r['tflops']:>10.2f} {r['tbps']:>9.2f}")


if __name__ == "__main__":
    main()
