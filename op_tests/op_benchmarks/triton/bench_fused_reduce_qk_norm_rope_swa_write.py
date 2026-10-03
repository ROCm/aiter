"""
Benchmark for the fused split-K reduce + Q/KV RMSNorm + RoPE (+ SWA write) kernel.

Shapes follow DeepSeek-V4 attention: head_dim 512 (448 NoPE + 64 RoPE), SWA window
128, and the head counts / split-K factors the unit test covers. Decode rows (small M)
are bound by host launch overhead under do_bench.
"""

import argparse

import torch
import triton

from aiter.ops.triton.fusions.fused_reduce_qk_norm_rope_swa_write import (
    fused_reduce_qk_norm_rope_swa_write,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.fusions.test_fused_reduce_qk_norm_rope_swa_write import (
    _build_cos_sin,
)

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def get_benchmark_shapes(args):
    """Return [(M, heads, splitk), ...] for the current CLI args."""
    decode_tokens = (1, 2, 4, 8, 16, 32, 64, 128)
    prefill_tokens = (1166, 4096, 8192, 16384)
    if args.M is not None:
        tokens = (args.M,)
    else:
        tokens = ()
        if args.sweep in ("decode", "all"):
            tokens += decode_tokens
        if args.sweep in ("prefill", "all"):
            tokens += prefill_tokens
    heads = (args.heads,) if args.heads is not None else (8, 128)
    splitk = (args.splitk,) if args.splitk is not None else (1, 2)
    return [(m, h, s) for h in heads for s in splitk for m in tokens]


def _positive_int(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def bench_fused_reduce_qk_norm_rope_swa_write_fn(M, heads, splitk, metric, args):
    head_dim = 512
    rope_dim = 64
    win = 128
    max_seq = 32768  # >= the largest M (16384), so every token gets its own position
    dtype = arg_to_torch_dtype[args.dtype]
    device = torch.device("cuda")
    N = heads * head_dim
    q_shape = (M, N) if splitk == 1 else (splitk, M, N)
    q = torch.randn(q_shape, dtype=dtype, device=device)
    kv = torch.randn(M, head_dim, dtype=dtype, device=device)
    q_out = torch.empty(M, heads, head_dim, dtype=dtype, device=device)
    cos, sin = _build_cos_sin(rope_dim, max_seq, dtype, device)
    positions = torch.randperm(max_seq, dtype=torch.int64, device=device)[:M]

    swa_kwargs = {}
    if not args.no_swa:
        swa_kwargs = {
            "write_indices": torch.randperm(M, dtype=torch.int32, device=device),
            "batch_id_per_token": torch.randperm(M, dtype=torch.int32, device=device),
            "state_slot_mapping": torch.randperm(M, dtype=torch.int32, device=device),
            "swa_kv": torch.zeros(M, win, head_dim, dtype=dtype, device=device),
            "win": win,
        }

    def fn():
        return fused_reduce_qk_norm_rope_swa_write(
            q,
            kv,
            None,
            None,
            1e-6,
            1e-6,
            rope_dim,
            cos,
            sin,
            positions,
            q_out=q_out,
            is_neox=args.neox,
            dtype=dtype,
            **swa_kwargs,
        )

    ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
    if metric == "time":
        return ms * 1000  # us

    # q_in read + q_out write + kv read/write (+ SWA row write); cos/sin are tiny.
    elem = torch.tensor([], dtype=dtype).element_size()
    mem = (splitk * M * N + M * N + 2 * M * head_dim) * elem
    if not args.no_swa:
        mem += M * head_dim * elem
    if metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9  # GB/s
    raise ValueError(f"unknown metric {metric}")


def run_benchmark(args):
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    benchmark = triton.testing.Benchmark(
        x_names=["M", "heads", "splitk"],
        x_vals=get_benchmark_shapes(args),
        line_arg="metric",
        line_vals=list(metrics),
        line_names=[f"{m} ({'us' if m == 'time' else 'GB/s'})" for m in metrics],
        styles=[("red", "-"), ("blue", "-")][: len(metrics)],
        ylabel="",
        plot_name=get_caller_name_no_ext() + f"_{args.dtype}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(M, heads, splitk, metric):
        return bench_fused_reduce_qk_norm_rope_swa_write_fn(
            M, heads, splitk, metric, args
        )

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark fused_reduce_qk_norm_rope_swa_write",
        description="Benchmark the Triton fused split-K reduce + Q/KV RMSNorm + RoPE "
        "(+ SWA write) kernel",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-M",
        type=_positive_int,
        default=None,
        help="Number of tokens (overrides --sweep)",
    )
    parser.add_argument(
        "--heads",
        type=_positive_int,
        default=None,
        help="Local query heads (default: 8, 128)",
    )
    parser.add_argument(
        "--splitk",
        type=int,
        choices=[1, 2, 4, 8],
        default=None,
        help="Split-K factor, a power of two (default: 1, 2)",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default="all",
        choices=["prefill", "decode", "all"],
        help="Token counts to sweep: prefill sizes, the decode range (1-128), or both",
    )
    parser.add_argument(
        "--no-swa",
        dest="no_swa",
        action="store_true",
        default=False,
        help="Skip the SWA KV write",
    )
    parser.add_argument(
        "--neox", action="store_true", default=False, help="NeoX-style RoPE"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=list(arg_to_torch_dtype),
        help="Input/output dtype (default: bf16)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=["all", "time", "bandwidth"],
        help="Metric to report (default: all)",
    )
    parser.add_argument(
        "--warmup", type=int, default=25, help="do_bench warmup budget in ms"
    )
    parser.add_argument(
        "--rep", type=int, default=100, help="do_bench measurement budget in ms"
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels",
    )
    parser.add_argument(
        "-o",
        action="store_true",
        default=False,
        help="Write CSV/PNG/HTML results to the current directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.print_vgpr:
        print(
            "Retrieving VGPR usage for fused_reduce_qk_norm_rope_swa_write kernels..."
        )
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return
    run_benchmark(args)


if __name__ == "__main__":
    main()
