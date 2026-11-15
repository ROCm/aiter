"""Microbenchmark for the fp8 MHA forward kernels in attention/mha_fp8.py.

Both modes time with ``RotatingBufferBench``: the calls are captured into a CUDA graph
that rotates through a pool of input sets, so the numbers exclude host overhead and
inputs that stay in the cache.

Usage:
    python bench_mha_fp8.py                          # mha_fwd, default shape
    python bench_mha_fp8.py --seqlen_q 4096 --batch 2
    python bench_mha_fp8.py --seqlen_q 1024 --seqlen_k 8192  # different q / k lengths
    python bench_mha_fp8.py --seqlen_q 1024 --seqlen_k 8192 --causal  # bottom-right causal
    python bench_mha_fp8.py --q_heads 32 --k_heads 8  # grouped query attention
    python bench_mha_fp8.py --varlen --q_heads 12 --seqlen_q 7680 --causal  # gfx950 MLA prefill
    python bench_mha_fp8.py --varlen --q_heads 12 --seqlen_q 128 --seqlen_k 16384  # key split

``--varlen`` benchmarks ``mha_varlen_fwd`` (gfx950) with packed q/k ``[T, H, head_dim +
pe_dim]`` and v ``[T, H, head_dim]``. A non-causal call also writes the LSE, as an MLA
context chunk does.
"""

import functools

import torch

from aiter.ops.triton.attention.mha_fp8 import mha_fwd, mha_varlen_fwd
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.attention.test_mha_fp8_e2e import (
    generate_mha_varlen_fp8_inputs,
    init_qkv,
)

MHA_FWD_KEYS = ["q", "k", "v", "q_scale", "k_scale", "v_scale", "o"]
MHA_VARLEN_KEYS = [
    "q", "k", "v", "cu_seqlens_q", "cu_seqlens_k",
    "q_descale", "k_descale", "v_descale", "out", "lse",
]  # fmt: skip


def generate_mha_fwd_inputs(b, hq, hk, sq, sk, d, device):
    q, k, v, q_scale, k_scale, v_scale = init_qkv(b, hq, hk, sq, sk, d, device)
    o = torch.empty(q.shape, dtype=torch.bfloat16, device=device)
    return q, k, v, q_scale, k_scale, v_scale, o


def bench_case(bencher, b, hq, hk, sq, sk, d, d_pe, causal, varlen, device):
    if varlen:
        gen = functools.partial(
            generate_mha_varlen_fp8_inputs, [sq] * b, [sk] * b, hq, 0, device, d_pe
        )
        keys = MHA_VARLEN_KEYS
        kernel_fn = functools.partial(
            mha_varlen_fwd,
            max_seqlen_q=sq,
            max_seqlen_k=sk,
            causal=causal,
            return_lse=not causal,
        )
    else:
        gen = functools.partial(generate_mha_fwd_inputs, b, hq, hk, sq, sk, d, device)
        keys = MHA_FWD_KEYS
        kernel_fn = functools.partial(mha_fwd, causal=causal)

    bencher.build_pool([gen], [keys])
    ms = bencher.bench(kernel_fn)

    num_pairs = sq * sk - sq * (sq - 1) / 2 if causal else sq * sk
    flops = 2.0 * b * hq * num_pairs * (2 * d + d_pe)
    num_bytes = b * (
        hq * sq * (d + d_pe) + hk * sk * (d + d_pe) + hk * sk * d + 2 * hq * sq * d
    )
    return {
        "us": ms * 1e3,
        "tflops": flops * 1e-12 / (ms * 1e-3),
        "tbps": num_bytes * 1e-12 / (ms * 1e-3),
    }


def parse_args():
    parser = get_parser(kernel_name="MHA fwd fp8")
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
    parser.add_argument(
        "--varlen",
        action="store_true",
        default=False,
        help="benchmark mha_varlen_fwd (gfx950 MLA prefill layout)",
    )
    parser.add_argument(
        "--pe_dim", type=int, default=64, help="rope head dim of q/k with --varlen"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "bench_mha_fp8.py needs a GPU"
    torch.manual_seed(0)
    device = torch.device("cuda")

    k_heads = args.q_heads if args.k_heads is None else args.k_heads
    assert (
        args.q_heads % k_heads == 0
    ), f"--q_heads ({args.q_heads}) must be a multiple of --k_heads ({k_heads})"
    seqlen_k = args.seqlen_q if args.seqlen_k is None else args.seqlen_k
    assert (
        not args.causal or seqlen_k >= args.seqlen_q
    ), f"--causal requires --seqlen_k ({seqlen_k}) >= --seqlen_q ({args.seqlen_q})"
    if args.varlen:
        assert k_heads == args.q_heads, "--varlen needs --k_heads == --q_heads"
    pe_dim = args.pe_dim if args.varlen else 0

    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )
    r = bench_case(
        bencher,
        args.batch,
        args.q_heads,
        k_heads,
        args.seqlen_q,
        seqlen_k,
        args.head_dim,
        pe_dim,
        args.causal,
        args.varlen,
        device,
    )

    print("MHA varlen fwd fp8" if args.varlen else "MHA fwd fp8")
    print(
        f"{'B':>4} {'HQ':>4} {'HK':>4} {'SQ':>7} {'SK':>7} {'D':>5} {'D_PE':>5} "
        f"{'us':>10} {'TFLOP/s':>10} {'TB/s':>9}"
    )
    print(
        f"{args.batch:>4} {args.q_heads:>4} {k_heads:>4} {args.seqlen_q:>7} {seqlen_k:>7} "
        f"{args.head_dim:>5} {pe_dim:>5} {r['us']:>10.2f} {r['tflops']:>10.2f} {r['tbps']:>9.2f}"
    )


if __name__ == "__main__":
    main()
