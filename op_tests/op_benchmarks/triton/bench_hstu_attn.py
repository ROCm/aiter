import functools
import itertools
import sys

import torch
import triton

from aiter.ops.triton.attention.hstu_attention import _AttentionFunction
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.attention.test_hstu_attn import (
    apply_SL,
    generate_sparse_seq_len,
    get_bytes,
    get_flops,
    sanity_check_attention,
    switch_to_contiguous_if_needed,
)

_TARGET_SIZE = 20
_POOL_FIELDS = ["q", "k", "v", "seq_offsets", "num_targets", "o", "do"]


def _generate_pool_entry(
    batch_size,
    max_seq_len,
    sparsity,
    heads,
    attn_dim,
    hidden_dim,
    sl_alpha,
    dtype,
    provider,
    mode,
    alpha,
    causal,
    sort_by_length,
):
    lengths = generate_sparse_seq_len(
        size=batch_size,
        max_seq_len=max_seq_len,
        sparsity=sparsity,
        device=torch.device("cuda"),
    )
    lengths = apply_SL(lengths, sl_alpha, max_seq_len=max_seq_len)
    num_targets = torch.randint(
        1,
        _TARGET_SIZE + 1,
        (batch_size,),
        device=lengths.device,
        dtype=lengths.dtype,
    )
    num_targets = torch.where(num_targets > lengths, lengths, num_targets)
    seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
    )
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    L = int(seq_offsets[-1].item())
    x = torch.empty(
        (L, heads, attn_dim * 2 + hidden_dim),
        dtype=dtype,
        device=torch.device("cuda"),
    ).uniform_(-0.01, 0.01)
    q, k, v = torch.split(x, [attn_dim, attn_dim, hidden_dim], dim=-1)

    q = switch_to_contiguous_if_needed(q)
    k = switch_to_contiguous_if_needed(k)
    v = switch_to_contiguous_if_needed(v)

    o = do = None
    if mode == "bwd":
        if provider == "aiter_triton":
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)
            o = _AttentionFunction.apply(
                max_seq_len,
                alpha,
                q,
                k,
                v,
                seq_offsets,
                causal,
                num_targets,
                0,
                0,
                sort_by_length,
            )
            do = torch.randn_like(o)
        else:  # flydsl
            do = torch.randn_like(v)

    return q, k, v, seq_offsets, num_targets, o, do


def _make_kernel_fn(provider, mode, max_seq_len, alpha, causal, sort_by_length):
    if provider == "flydsl":
        from aiter.ops.flydsl.hstu_attention import (
            flydsl_hstu_attention_bwd,
            flydsl_hstu_attention_fwd,
        )

        if mode == "fwd":

            def kernel_fn(q, k, v, seq_offsets, num_targets, **_):
                return flydsl_hstu_attention_fwd(
                    max_seq_len, alpha, q, k, v, seq_offsets, causal, num_targets, 0, 0
                )

        else:

            def kernel_fn(q, k, v, seq_offsets, num_targets, do, **_):
                return flydsl_hstu_attention_bwd(
                    max_seq_len,
                    alpha,
                    q,
                    k,
                    v,
                    do,
                    seq_offsets,
                    causal,
                    num_targets,
                    0,
                    0,
                    sort_by_length=sort_by_length,
                )

    else:  # aiter_triton
        if mode == "fwd":

            def kernel_fn(q, k, v, seq_offsets, num_targets, **_):
                return _AttentionFunction.apply(
                    max_seq_len,
                    alpha,
                    q,
                    k,
                    v,
                    seq_offsets,
                    causal,
                    num_targets,
                    0,
                    0,
                    sort_by_length,
                )

        else:
            # torch.autograd.grad, not o.backward(): .backward() accumulates
            # into q/k/v.grad on every call, which do_bench_aiter_triton's
            # thousands of graph replays would grow unbounded. grad() returns
            # fresh tensors each call instead.
            def kernel_fn(q, k, v, o, do, **_):
                return torch.autograd.grad(o, (q, k, v), do, retain_graph=True)

    return kernel_fn


def _rotating_call(fn, rot_bufs):
    """Manual pool rotation for candidates that can't be CUDA-graph captured.

    aiter_triton/bwd's autograd graph is built from ``o = _AttentionFunction
    .apply(...)`` computed once per pool entry, outside any graph capture
    region. Replaying ``torch.autograd.grad`` against it under
    ``do_bench_aiter_triton``'s capture fails: the autograd node holds a
    stale reference to the (non-capturing) stream it was built on. It still
    benefits from rotating through independent input sets, just not via
    CUDA graph replay.
    """
    pool_len = len(rot_bufs)
    counter = itertools.count()

    def run():
        return fn(**rot_bufs[next(counter) % pool_len])

    return run


def get_x_values():
    x_vals = [
        (32, 1024, 0.5, 4, 128, 128),
        (32, 2048, 0.5, 4, 128, 128),
        (32, 4096, 0.5, 4, 128, 128),
        (32, 8192, 0.5, 4, 128, 128),
        (32, 16384, 0.5, 4, 128, 128),
        (512, 512, 0.97, 4, 128, 128),
        (512, 3072, 0.366, 4, 128, 128),
        (1024, 1024, 0.768, 4, 128, 128),
    ]

    return x_vals


def run_benchmark(args):

    x_names = [
        "batch_size",
        "max_seq_len",
        "sparsity",
        "heads",
        "attn_dim",
        "hidden_dim",
    ]
    if args.user_input:
        x_val_list = [
            (
                args.b,
                args.max_seq_len,
                args.sparsity,
                args.heads,
                args.head_dim,
                args.hidden_dim,
            )
        ]
    else:
        x_val_list = get_x_values()

    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "throughput":
        ylabel = "Throughput (TFLOPS)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth (GBs)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    evaluation_metric_to_unit = {
        "throughput": "TFLOPS",
        "time": "Time_(ms)",
        "bandwidth": "Bandwidth_(GB/s)",  # spaces break prettytable parsing
    }
    modes = [args.mode]
    if args.mode == "both":
        modes = ["fwd", "bwd"]

    configs = []
    for mode in modes:
        metric = args.metric
        # backward only supports the time metric; force it (and the column
        # label + ylabel) so the reported values are never mislabeled.
        if mode == "bwd":
            metric = "time"
        if metric == "time":
            ylabel = "Time (ms)"
        elif metric == "throughput":
            ylabel = "Throughput (TFLOPS)"
        else:
            ylabel = "Bandwidth (GBs)"

        unit = evaluation_metric_to_unit[metric]
        configs.append(
            triton.testing.Benchmark(
                x_names=x_names,
                x_vals=x_val_list,
                line_arg="unit",
                line_vals=[unit],
                line_names=[unit],
                styles=[("green", "-")],
                ylabel=ylabel,
                plot_name=get_caller_name_no_ext(),
                args={"metric": metric, "mode": mode},
            )
        )

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report(configs)
    def bench_hstu_attn(
        batch_size,
        max_seq_len,
        sparsity,
        heads,
        attn_dim,
        hidden_dim,
        metric,
        mode,
        **kwargs,
    ):
        type_str = args.dtype
        assert type_str in [
            "fp16",
            "bf16",
        ], "only fp16 or bf16 data types are supported!"
        dropout_pr = 0.0
        # apply_SL clamps long sequences to ~N^(sl_alpha/2). sl_alpha=2.0 is
        # effectively no clamp; 1.7 matches the HSTU length distribution.
        sl_alpha: float = args.sl_alpha
        dtype = str_to_torch_dtype[type_str]

        invalid_attn_mask_type = "lower_triangular"
        causal = True
        alpha = 1.0 / attn_dim * 10000
        provider = args.provider

        torch.manual_seed(1001)  # for reproducibility
        pool_gen = functools.partial(
            _generate_pool_entry,
            batch_size,
            max_seq_len,
            sparsity,
            heads,
            attn_dim,
            hidden_dim,
            sl_alpha,
            dtype,
            provider,
            mode,
            alpha,
            causal,
            args.sort_by_length,
        )
        rot_bufs = generate_rotating_buffers_pool(
            [pool_gen], [_POOL_FIELDS], **kwargs_rot_bufs
        )

        sanity_check_attention(
            max_seq_len=max_seq_len,
            q=rot_bufs[0]["q"],
            k=rot_bufs[0]["k"],
            v=rot_bufs[0]["v"],
            seq_offsets=rot_bufs[0]["seq_offsets"],
            invalid_attn_mask_type=invalid_attn_mask_type,
            dropout_pr=dropout_pr,
            attn_bias=None,
            max_attn_len=None,
            contextual_seq_len=0,
        )

        kernel_fn = _make_kernel_fn(
            provider, mode, max_seq_len, alpha, causal, args.sort_by_length
        )
        if provider == "aiter_triton" and mode == "bwd":
            ms = triton.testing.do_bench(
                _rotating_call(kernel_fn, rot_bufs), warmup=5, rep=20
            )
        else:
            ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return ms
        elif metric == "throughput":
            flops = get_flops(
                rot_bufs[0]["seq_offsets"].cpu().numpy(), heads, attn_dim, hidden_dim
            )
            tflops = flops / ms * 1e-9
            return tflops
        elif metric == "bandwidth":
            elem_size = rot_bufs[0]["q"].element_size()
            bytes = get_bytes(
                rot_bufs[0]["seq_offsets"].cpu().numpy(),
                heads,
                attn_dim,
                hidden_dim,
                elem_size,
            )
            bandwidth = bytes / (ms * 1e-3) * 1e-9  # GB/s
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_hstu_attn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = get_parser(kernel_name="HSTU Attention")
    parser.add_argument(
        "--b",
        type=int,
        default=512,
        help="Batch dim of input sequences",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="max sequence length",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="sparsity of input sequence lengths",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="number of heads",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=128,
        help="head dimension",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="hidden dimension",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        help="data type, default (bfloat16)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=["aiter_triton", "flydsl"],
        default="aiter_triton",
        help="kernel implementation to benchmark",
    )

    parser.add_argument(
        "--sl_alpha",
        type=float,
        default=2.0,
        help=(
            "apply_SL clamp exponent (~N^(sl_alpha/2)). 2.0 = near-unclamped "
            "(heavy); 1.7 = HSTU deployment distribution."
        ),
    )

    parser.add_argument(
        "--sort_by_length",
        action="store_true",
        default=False,
        help=(
            "Enable sort-by-length load balancing in the bwd (FlyDSL: group-aware "
            "batch remap; AITER-Triton: its native sort_by_length). fwd unaffected."
        ),
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="fwd",
        choices=["fwd", "bwd", "both"],
        help="indicate run forward, backward, or both",
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Prints the VGPR usage of the compiled triton kernel.",
    )
    parser.add_argument(
        "--user_input",
        action="store_true",
        default=False,
        help="Run user input info",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")

        def fun():
            run_benchmark(args)

        print_vgpr(fun, get_caller_name_no_ext())
        return 0
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
