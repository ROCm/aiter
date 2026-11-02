"""
Benchmark for the fused split-K reduce + Q/KV RMSNorm + RoPE (+ SWA write) kernel.

Shapes follow DeepSeek-V4 attention: head_dim 512 (448 NoPE + 64 RoPE), SWA window
128, and the head counts / split-K factors the unit test covers. Decode rows (small M)
are bound by host launch overhead under do_bench.
"""

import argparse
import functools

import torch
import triton

from aiter.ops.triton.fusions.fused_reduce_qk_norm_rope_swa_write import (
    fused_reduce_qk_norm_rope_swa_write,
)
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.fusions.test_fused_reduce_qk_norm_rope_swa_write import (
    _build_cos_sin,
)


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
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer, got {value}")
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def _generate_inputs(
    M, heads, splitk, head_dim, rope_dim, win, max_seq, dtype, device, no_swa
):
    N = heads * head_dim
    q_shape = (M, N) if splitk == 1 else (splitk, M, N)
    q = torch.randn(q_shape, dtype=dtype, device=device)
    kv = torch.randn(M, head_dim, dtype=dtype, device=device)
    q_out = torch.empty(M, heads, head_dim, dtype=dtype, device=device)
    cos, sin = _build_cos_sin(rope_dim, max_seq, dtype, device)
    positions = torch.randperm(max_seq, dtype=torch.int64, device=device)[:M]
    write_indices = torch.randperm(M, dtype=torch.int32, device=device)
    batch_id_per_token = torch.randperm(M, dtype=torch.int32, device=device)
    state_slot_mapping = torch.randperm(M, dtype=torch.int32, device=device)
    swa_kv = None
    if not no_swa:
        swa_kv = torch.zeros(M, win, head_dim, dtype=dtype, device=device)
    return (
        q,
        kv,
        q_out,
        cos,
        sin,
        positions,
        write_indices,
        batch_id_per_token,
        state_slot_mapping,
        swa_kv,
        win,
    )


def bench_fused_reduce_qk_norm_rope_swa_write_fn(M, heads, splitk, metric, args):
    head_dim = 512
    rope_dim = 64
    win = 128
    max_seq = max(32768, M)  # Ensure every requested token has a valid position.
    dtype = str_to_torch_dtype[args.dtype]
    device = torch.device("cuda")
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    torch.manual_seed(0)
    _p_gen = functools.partial(
        _generate_inputs,
        M,
        heads,
        splitk,
        head_dim,
        rope_dim,
        win,
        max_seq,
        dtype,
        device,
        args.no_swa,
    )
    _out_val_keys = [
        "q",
        "kv",
        "q_out",
        "cos",
        "sin",
        "positions",
        "write_indices",
        "batch_id_per_token",
        "state_slot_mapping",
        "swa_kv",
        "win",
    ]
    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [_out_val_keys], **kwargs_rot_bufs
    )
    sample = rot_bufs[0]

    def _run(
        q,
        kv,
        q_out,
        cos,
        sin,
        positions,
        write_indices,
        batch_id_per_token,
        state_slot_mapping,
        swa_kv,
        win,
    ):
        swa_kwargs = {}
        if not args.no_swa:
            swa_kwargs = {
                "write_indices": write_indices,
                "batch_id_per_token": batch_id_per_token,
                "state_slot_mapping": state_slot_mapping,
                "swa_kv": swa_kv,
                "win": win,
            }
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

    ms = do_bench_aiter_triton(_run, rot_bufs, **kwargs_do_bench_graph)
    if metric == "time":
        return ms

    # q_in read + q_out write + kv read/write (+ SWA row write); cos/sin are tiny.
    elem = sample["q"].element_size()
    N = sample["q"].shape[-1]
    mem = (splitk * M * N + M * N + 2 * M * head_dim) * elem
    if not args.no_swa:
        mem += M * head_dim * elem
    if metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9  # GB/s
    raise ValueError(f"unknown metric {metric}")


def run_benchmark(args):
    units = {"time": "ms", "bandwidth": "GB/s"}
    benchmark = triton.testing.Benchmark(
        x_names=["M", "heads", "splitk"],
        x_vals=get_benchmark_shapes(args),
        line_arg="metric",
        line_vals=[args.metric],
        line_names=[args.metric],
        styles=[("red", "-")],
        ylabel=units[args.metric],
        plot_name=get_caller_name_no_ext() + f"_{args.dtype}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(M, heads, splitk, metric):
        return bench_fused_reduce_qk_norm_rope_swa_write_fn(
            M, heads, splitk, metric, args
        )

    df = bench_fn.run(print_data=True, return_df=True)[0]
    if args.o:
        df.to_csv(f"{benchmark.plot_name}.csv", index=False)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="fused_reduce_qk_norm_rope_swa_write")
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
        choices=["bf16", "fp16"],
        help="Input/output dtype (default: bf16)",
    )
    override_parser_action(
        parser, action="--metric", choices=["time", "bandwidth"], default="time"
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
        print(
            "Retrieving VGPR usage for fused_reduce_qk_norm_rope_swa_write kernels..."
        )
        print_vgpr(lambda: run_benchmark(parsed_args), get_caller_name_no_ext())
        return
    run_benchmark(parsed_args)


if __name__ == "__main__":
    main()
