import argparse
import sys
import triton
import torch
from aiter.ops.triton.hstu_attention import _AttentionFunction
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.triton_tests.test_hstu_attn import (
    switch_to_contiguous_if_needed,
    generate_sparse_seq_len,
    apply_SL,
    sanity_check_attention,
    get_flops,
    get_bytes,
)


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
        # (256, 4096, 0.5, 4, 128, 128),
        # (320, 4096, 0.5, 4, 128, 128),
        # (384, 4096, 0.5, 4, 128, 128),
        # (448, 4096, 0.5, 4, 128, 128),
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

    line_names = ["Triton"]
    line_vals = ["triton"]
    modes = [args.mode]
    if args.mode == "both":
        modes = ["fwd", "bwd"]

    configs = []
    for mode in modes:
        metric = args.metric
        # backward only support time metric
        if mode == "bwd":
            metric = "time"
            ylabel = "Time (ms)"

        configs.append(
            triton.testing.Benchmark(
                x_names=x_names,
                x_vals=x_val_list,
                line_arg="provider",
                line_vals=line_vals,
                line_names=line_names,
                styles=[("green", "-")],
                ylabel=ylabel,
                plot_name=f"HSTU attention Benchmark ({mode}), {ylabel}",
                args={"metric": metric, "mode": mode},
            )
        )

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
        provider,
    ):
        type_str = args.dtype
        assert type_str in [
            "fp16",
            "bf16",
        ], "only fp16 or bf16 data types are supported!"
        dropout_pr = 0.0
        target_size: int = 20
        sl_alpha: float = 2.0
        dtype = str_to_torch_dtype[type_str]

        invalid_attn_mask_type = "lower_triangular"
        causal = True
        alpha = 1.0 / attn_dim * 10000

        # generate inputs
        torch.manual_seed(1001)  # for reproducibility
        lengths = generate_sparse_seq_len(
            size=batch_size,
            max_seq_len=max_seq_len,
            sparsity=sparsity,
            device=torch.device("cuda"),
        )
        lengths = apply_SL(lengths, sl_alpha, max_seq_len=max_seq_len)
        num_targets = torch.randint(
            1,
            target_size + 1,
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

        sanity_check_attention(
            max_seq_len=max_seq_len,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            invalid_attn_mask_type=invalid_attn_mask_type,
            dropout_pr=dropout_pr,
            attn_bias=None,
            max_attn_len=None,
            contextual_seq_len=0,
        )

        def attn_fwd():
            return _AttentionFunction.apply(
                max_seq_len,
                alpha,
                q,
                k,
                v,
                seq_offsets,
                causal,
                num_targets,
                0,  # max_attn_len,
                0,  # contextual_seq_len
                True,  # sort_by_length,
            )

        if mode == "fwd":
            ms = triton.testing.do_bench(
                attn_fwd,
                warmup=25,
                rep=100,
            )
        else:
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)

            o = attn_fwd()
            do = torch.randn_like(o)

            def attn_bwd():
                return o.backward(do, retain_graph=True)

            ms = triton.testing.do_bench(
                attn_bwd,
                warmup=25,
                rep=100,
            )

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return ms
        elif metric == "throughput":
            flops = get_flops(seq_offsets.cpu().numpy(), heads, attn_dim, hidden_dim)
            tflops = flops / ms * 1e-9
            return tflops
        elif metric == "bandwidth":
            elem_size = q.element_size()
            bytes = get_bytes(
                seq_offsets.cpu().numpy(), heads, attn_dim, hidden_dim, elem_size
            )
            bandwidth = bytes / (ms * 1e-3) * 1e-9  # GB/s
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_hstu_attn.run(save_path=".", print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark HSTU Attention",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        "--metric",
        type=str,
        choices=["time", "throughput", "bandwidth"],
        default="throughput",
        help="metric to plot",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="fwd",
        choices=["fwd", "bwd", "both"],
        help="indicate run forward, backward, or both",
    )

    parser.add_argument(
        "--user_input",
        action="store_true",
        default=False,
        help="Run user input info",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
