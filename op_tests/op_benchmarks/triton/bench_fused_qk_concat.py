"""
Benchmark for the fused q/k concat kernels: fused_qk_cat (concat only) and
fused_qk_rope_cat (RoPE on pe part, then concat).
"""

import functools

import triton
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

from aiter.ops.triton.fusions.fused_qk_concat import fused_qk_cat, fused_qk_rope_cat
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
    positive_int,
    pow2_int,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.fusions.test_fused_qk_concat import (
    generate_qk_inputs,
    generate_rope_cached_freqs,
)


def _qk_cat_fn(q_nope, q_pe, k_nope, k_pe, **_):
    return fused_qk_cat(q_nope, q_pe, k_nope, k_pe)


def _generate_qk_rope_cat_pool_inputs(
    B, qh_per_kh, KH, D_nope, D_pe, dtype, d_freq, max_embed_positions
):
    q_nope, q_pe, k_nope, k_pe = generate_qk_inputs(
        B, qh_per_kh, KH, D_nope, D_pe, dtype
    )
    pos, _, cos, sin = generate_rope_cached_freqs(B, max_embed_positions, d_freq, dtype)
    return q_nope, q_pe, k_nope, k_pe, pos, cos, sin


def get_benchmark_shapes(args):
    """Return [(B, QH, KH, D_pe), ...] for the current CLI args."""
    tokens = args.B if args.B is not None else (1, 4, 8, 16, 32, 4096)
    qh_per_kh = args.qh_per_kh if args.qh_per_kh is not None else (1, 2, 4, 8, 16)
    kv_heads = args.kh if args.kh is not None else (1, 4)
    pe_dims = args.d_pe if args.d_pe is not None else (64, 128)
    return [
        (b, r * kh, kh, d_pe)
        for d_pe in pe_dims
        for kh in kv_heads
        for r in qh_per_kh
        for b in tokens
    ]


def bench_fused_qk_concat_fn(bencher, B, QH, KH, D_pe, metric, args):
    dtype = str_to_torch_dtype[args.dtype]
    D_nope = args.d_nope

    if args.op == "qk_cat":
        freq_bytes = 0
        _p_gen = functools.partial(
            generate_qk_inputs, B, QH // KH, KH, D_nope, D_pe, dtype
        )
        _out_val_keys = ["q_nope", "q_pe", "k_nope", "k_pe"]
        rot_bufs = bencher.build_pool([_p_gen], [_out_val_keys])
        kernel_fn = _qk_cat_fn
    else:
        d_freq = D_pe // 2 if args.reuse_freqs_front_part else D_pe
        _p_gen = functools.partial(
            _generate_qk_rope_cat_pool_inputs,
            B,
            QH // KH,
            KH,
            D_nope,
            D_pe,
            dtype,
            d_freq,
            args.max_embed_positions,
        )
        _out_val_keys = ["q_nope", "q_pe", "k_nope", "k_pe", "pos", "cos", "sin"]
        rot_bufs = bencher.build_pool([_p_gen], [_out_val_keys])
        freq_bytes = 2 * B * d_freq * rot_bufs[0]["cos"].element_size()
        kernel_fn = functools.partial(fused_qk_rope_cat, is_neox=args.neox)

    ms = bencher.bench(kernel_fn)
    if metric == "time":
        return ms

    # Bytes: q/k nope and pe reads, q_out/k_out writes, and for RoPE one cos and one sin
    # row per token
    mem = (
        2 * B * (QH + KH) * (D_nope + D_pe) * rot_bufs[0]["q_nope"].element_size()
        + freq_bytes
    )
    return mem / (ms * 1e-3) * 1e-9  # GB/s


def run_benchmark(args):
    units = {"time": "ms", "bandwidth": "GB/s"}
    benchmark = triton.testing.Benchmark(
        x_names=["B", "QH", "KH", "D_pe"],
        x_vals=get_benchmark_shapes(args),
        line_arg="metric",
        line_vals=[args.metric],
        line_names=[args.metric],
        ylabel=units[args.metric],
        plot_name=f"{get_caller_name_no_ext()}_{args.op}_{args.dtype}",
        args={},
    )

    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(B, QH, KH, D_pe, metric):
        return bench_fused_qk_concat_fn(bencher, B, QH, KH, D_pe, metric, args)

    df = bench_fn.run(print_data=True, return_df=True)[0]
    if args.o:
        df.to_csv(f"{benchmark.plot_name}.csv", index=False)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="fused_qk_concat")
    parser.add_argument(
        "--op",
        type=str,
        default="qk_cat",
        choices=["qk_cat", "qk_rope_cat"],
        help="fused_qk_cat (concat) or fused_qk_rope_cat (RoPE on the pe part, then "
        "concat) (default: qk_cat)",
    )
    parser.add_argument(
        "-B",
        type=positive_int,
        nargs="+",
        default=None,
        help="Number of tokens (default: 1 4 8 16 32 4096)",
    )
    parser.add_argument(
        "--kh",
        type=positive_int,
        nargs="+",
        default=None,
        help="Number of KV heads (default: 1 4)",
    )
    parser.add_argument(
        "--qh-per-kh",
        type=positive_int,
        nargs="+",
        default=None,
        help="Query heads per KV head (default: 1 2 4 8 16)",
    )
    parser.add_argument(
        "--d-nope",
        type=pow2_int,
        default=512,
        help="nope head dim (default: 512)",
    )
    parser.add_argument(
        "--d-pe",
        type=pow2_int,
        nargs="+",
        default=None,
        help="pe (RoPE) head dim (default: 64 128)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Input dtype (default: bf16)",
    )
    parser.add_argument(
        "--neox",
        action="store_true",
        default=False,
        help="NeoX-style RoPE (qk_rope_cat only, default: GPT-J style)",
    )
    parser.add_argument(
        "--reuse-freqs-front-part",
        action="store_true",
        default=False,
        help="cos/sin hold D_pe // 2 frequencies (qk_rope_cat only, default: D_pe)",
    )
    parser.add_argument(
        "--max-embed-positions",
        type=positive_int,
        default=131072,
        help="Rows in the cos/sin cache (qk_rope_cat only, default: 131072)",
    )
    override_parser_action(
        parser,
        action="--metric",
        choices=["time", "bandwidth"],
        default="time",
        help="Metric to report (default: time)",
    )
    parser.add_argument(
        "--rep",
        type=positive_int,
        default=20,
        help="Unused: RotatingBufferBench.bench's replay count is a fixed "
        "constant, not an adaptive rep target. Kept for CLI compatibility.",
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
        help="Write performance results to CSV file",
    )
    return parser.parse_args(args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for fused_qk_concat kernels...")
        print_vgpr(lambda: run_benchmark(parsed_args), get_caller_name_no_ext())
        return
    run_benchmark(parsed_args)


if __name__ == "__main__":
    main()
