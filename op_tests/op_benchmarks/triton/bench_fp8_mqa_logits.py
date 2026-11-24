import functools

import torch
import triton

from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.utils.types import e4m3_dtype
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.attention.test_fp8_mqa_logits import (
    per_custom_dims_cast_to_fp8,
)

# (batch_size, seq_q_l, seq_kv_l, clean_logits) for --long_kv_cases, with 32
# heads of size 128. The comments name the gfx950 kernel of each group.
# fmt: off
LONG_KV_CASES = (
    # Existing kernel: a small eight-row grid, a row count that is not a
    # multiple of 8, a short KV, or clean_logits.
    (1, 8, 4095, 0), (1, 8, 4096, 0), (1, 8, 4097, 0), (1, 9, 4097, 0),
    (1, 24, 4223, 0), (1, 8, 4097, 1), (1, 4098, 8192, 0), (1, 2040, 4096, 0),
    (1, 1024, 9216, 0), (1, 64, 193536, 0),
    # Eight-row kernel at the smallest grids of the gate: 2 to 64 KV splits.
    (1, 2048, 4096, 0), (1, 1024, 9217, 0), (1, 256, 46081, 0), (1, 64, 193537, 0),
    # Eight-row kernel, one request in a chunk: 3 to 342 KV splits.
    (1, 4096, 8192, 0), (1, 4104, 8192, 0), (1, 4088, 8192, 0), (1, 2048, 16384, 0),
    (1, 2048, 32768, 0), (1, 1024, 49152, 0), (1, 1024, 65536, 0), (1, 4096, 32768, 0),
    (1, 2048, 65536, 0), (1, 1024, 131072, 0), (1, 1856, 116480, 0),
    (1, 512, 262144, 0), (1, 256, 524288, 0), (1, 128, 1048576, 0), (1, 64, 573440, 0),
    (1, 64, 1048576, 0), (1, 936, 573440, 0),
    # Eight-row kernel, equal requests in a chunk: resumed ones whose row
    # count is and is not a multiple of 8, and fresh ones.
    (4, 1024, 8192, 0), (4, 1022, 8192, 0), (4, 256, 32768, 0), (8, 511, 4096, 0),
    (4, 2046, 2046, 0),
)
# fmt: on


def generate_fp8_mqa_logits_inputs(
    batch_size, seq_q_l, seq_kv_l, num_heads_q, head_dim
):
    s_q = batch_size * seq_q_l
    s_k = batch_size * seq_kv_l

    q = torch.randn(s_q, num_heads_q, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv = (kv_fp8.to(torch.float32) * scales[:, None]).to(torch.bfloat16)
    weights = torch.randn(s_q, num_heads_q, device="cuda", dtype=torch.float32)

    ks = torch.zeros(s_q, dtype=torch.int, device="cuda")
    ke = torch.zeros(s_q, dtype=torch.int, device="cuda")
    arange_q = torch.arange(seq_q_l, dtype=torch.int, device="cuda")
    for b in range(batch_size):
        qs = b * seq_q_l
        kvs = b * seq_kv_l
        ks[qs : qs + seq_q_l] = kvs
        ke[qs : qs + seq_q_l] = kvs + (seq_kv_l - seq_q_l) + arange_q + 1

    q_fp8 = q.to(e4m3_dtype)
    return q_fp8, kv_fp8, scales, weights, ks, ke


def calculate_tflops(start_inds, end_inds, num_heads_q, head_dim, time_ms):
    time_s = time_ms * 1e-3
    start_inds = start_inds.to("cpu").numpy()
    end_inds = end_inds.to("cpu").numpy()
    total_flops = 0.0
    for i in range(len(start_inds)):
        start = start_inds[i]
        end = end_inds[i]
        total_flops += 2.0 * num_heads_q * head_dim * (end - start)
    # TFLOPs = total FLOPs / (time in seconds * 1e12)
    tflops = total_flops / (time_s * 1e12)

    return tflops


def run_benchmark(args):
    if not 0 < args.seq_q_l <= args.seq_kv_l:
        raise ValueError("query length must be positive and no larger than key length")
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )
    x_names = [
        "batch_size",
        "seq_q_l",
        "seq_kv_l",
        "num_heads_q",
        "head_dim",
        "clean_logits",
        "layout",
    ]
    x_vals_list = [
        [
            args.batch_size,
            args.seq_q_l,
            args.seq_kv_l,
            args.num_heads_q,
            args.head_dim,
            args.clean_logits,
            args.layout,
        ]
    ]
    if args.long_kv_cases:
        x_vals_list = [
            [batch, rows, keys, 32, 128, clean, args.layout]
            for batch, rows, keys, clean in LONG_KV_CASES
        ]
    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "throughput":
        ylabel = "TFLOPs"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    line_names = [ylabel]
    line_vals = [ylabel]
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="unit",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fp8_mqa_logits(
        batch_size,
        seq_q_l,
        seq_kv_l,
        num_heads_q,
        head_dim,
        clean_logits,
        layout,
        metric,
        **kwargs,
    ):
        _p_gen = functools.partial(
            generate_fp8_mqa_logits_inputs,
            batch_size,
            seq_q_l,
            seq_kv_l,
            num_heads_q,
            head_dim,
        )
        _out_val_keys = ["q_fp8", "kv_fp8", "scales", "weights", "ks", "ke"]
        rot_bufs = bencher.build_pool([_p_gen], [_out_val_keys])

        def _run(q_fp8, kv_fp8, scales, weights, ks, ke):
            fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits)

        time_ms = bencher.bench(_run)
        sample = rot_bufs[0]
        tflops = calculate_tflops(
            sample["ks"], sample["ke"], num_heads_q, head_dim, time_ms
        )

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return time_ms
        elif metric == "throughput":
            return tflops
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_fp8_mqa_logits.run(save_path="." if args.o else None, print_data=True)


def main():
    parser = get_parser(kernel_name="FP8 MQA Logits")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_heads_q", type=int, default=64, help="num. q heads")
    parser.add_argument("--head_dim", type=int, default=128, help="head dim size")
    parser.add_argument(
        "--layout",
        choices=["allocated", "dense", "packed"],
        default="allocated",
        help="Logits allocation and row layout",
    )
    parser.add_argument(
        "--seq_q_l", type=int, default=4096, help="Input sequence length"
    )
    parser.add_argument(
        "--seq_kv_l", type=int, default=4096, help="Output sequence length"
    )
    parser.add_argument(
        "--clean_logits", type=int, default=1, help="Clean the untouched logits"
    )
    parser.add_argument(
        "--long_kv_cases",
        action="store_true",
        help="Benchmark the H32/D128 shapes of LONG_KV_CASES: calls on each side of "
        "the gfx950 eight-row gate, and prefill chunks with 2 to 342 KV splits. "
        "Ignores the other shape options",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    override_parser_action(
        parser, action="--metric", choices=["time", "throughput"], default="throughput"
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
