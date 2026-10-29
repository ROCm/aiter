import argparse
from collections.abc import Iterable
from typing import TypeVar

_T = TypeVar("_T")


from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_available_models,
)


def positive_int(value: str) -> int:
    error = argparse.ArgumentTypeError(f"'{value}' is not a positive integer")
    try:
        float_value = float(value)
        if float_value != int(float_value):
            raise error
        int_value = int(float_value)
    except (ValueError, OverflowError):
        raise error
    if int_value <= 0:
        raise error
    return int_value


def pow2_int(value: str) -> int:
    int_value = positive_int(value)
    if int_value & (int_value - 1):
        raise argparse.ArgumentTypeError(f"'{value}' is not a power of 2")
    return int_value


def override_parser_action(
    parser: argparse.ArgumentParser,
    action: str,
    choices: Iterable[_T] | None = None,
    default: _T | None = None,
    help: str | None = None,
):
    act = parser._option_string_actions[action]
    if default is not None:
        act.default = default
    if choices is not None:
        act.choices = choices
    if help is not None:
        act.help = help


def get_parser(kernel_name: str) -> argparse.ArgumentParser:
    """
    Builds an argparser with default flags for Triton kernel benchmarking.

    Args:
        - kernel_name: The name of the Triton kernel to benchmark -> appears in --help
    """
    parser = argparse.ArgumentParser(
        prog=f"Benchmark {kernel_name}",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    available_models = get_available_models()  # Dynamically load model names
    model_help = (
        "Model name to benchmark. Select from: ["
        + ", ".join(available_models)
        + "]. Use 'all' to benchmark all models or leave blank for the default benchmark script."
    )
    parser.add_argument(
        "--model-configs",
        type=str,
        default="utils/model_configs.json",
        help="Model config json file.",
    )
    parser.add_argument("--model", type=str, help=model_help)

    def to_backend_arg(s: str) -> str | None:
        if s == "auto":
            return None
        else:
            return s

    parser.add_argument(
        "--backend",
        type=to_backend_arg,
        choices=["triton", "gluon", None],
        metavar="{triton,gluon,auto}",
        default="auto",
        help="kernel backend to use",
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "throughput", "bandwidth"],
        default="throughput",
        help="metric to plot",
    )

    parser.add_argument(
        "--graph-warmup-iters",
        type=positive_int,
        default=3,
        help="do_bench_aiter_triton: untimed iterations before the CUDA graph is captured",
    )
    parser.add_argument(
        "--graph-warmup-replays",
        type=positive_int,
        default=3,
        help="do_bench_aiter_triton: untimed graph replays before the timed ones",
    )
    parser.add_argument(
        "--graph-kernels-per-graph",
        type=positive_int,
        default=1000,
        help="do_bench_aiter_triton: kernel calls captured per CUDA graph",
    )
    parser.add_argument(
        "--graph-num-replays",
        type=positive_int,
        default=10,
        help="do_bench_aiter_triton: timed graph replays",
    )
    parser.add_argument(
        "--rot-bufs-min-pool-len",
        type=positive_int,
        default=4,
        help="rotating buffers minimum pool length",
    )
    parser.add_argument(
        "--rot-bufs-target-mb",
        type=positive_int,
        default=256,
        help="rotating buffers target size in MB",
    )
    parser.add_argument(
        "--rot-bufs-print-info",
        action="store_true",
        default=False,
        help="Print rotating buffers pool info (pool length, per-set and total bytes)",
    )

    return parser


def get_do_bench_graph_kwargs(args: argparse.Namespace) -> dict:
    """do_bench_aiter_triton kwargs from the --graph-* flags added by get_parser()."""
    return {
        "warmup_iters": args.graph_warmup_iters,
        "warmup_replays": args.graph_warmup_replays,
        "kernels_per_graph": args.graph_kernels_per_graph,
        "num_replays": args.graph_num_replays,
    }


def get_rot_bufs_kwargs(args: argparse.Namespace) -> dict:
    """generate_rotating_buffers_pool kwargs from the --rot-bufs-* flags and added by get_parser().
    Note: max_pool_len is set to --graph-kernels-per-graph
    """
    return {
        "target_mb": args.rot_bufs_target_mb,
        "min_pool_len": args.rot_bufs_min_pool_len,
        "print_info": args.rot_bufs_print_info,
        # By default keep the rotating buffer pool length <= graph_kernels_per_graph
        # Otherwise, in case per-set bytes are too small, then covering target_mb might
        # blow up the pool_len. This is not ideal for capturing graph
        "max_pool_len": args.graph_kernels_per_graph,
    }


def get_ff_args(
    parser: argparse.ArgumentParser, args: list[str] | None = None
) -> tuple[dict, dict]:
    """
    Does additional processing on parser args for feed-forward blocks.
    """
    args = parser.parse_args(args=args)
    if args.shape is not None:
        if len(args.shape) == 3:
            args.M, args.N, args.K = args.shape
        elif len(args.shape) == 4:
            args.B, args.M, args.N, args.K = args.shape
    defaults = parser.parse_args([])  # get default arguments

    return args, defaults


def add_argparse_ff(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Adds argparse flags for benchmarking Triton kernels for feed-forward layers.
    """
    parser.add_argument(
        "-tp",
        type=int,
        default=1,
        help="Tensor parallel (divides intermediate_size)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["TT", "TN", "NT", "NN"],
        default="TN",
        help="Layout of input and weight matrix",
    )
    parser.add_argument(
        "-fc1",
        action="store_true",
        help="Benchmark the up-projection (hidden dim to intermediate dim in the feed-forward layer)",
    )
    parser.add_argument(
        "-fc2",
        action="store_true",
        help="Benchmark the down-projection (intermediate dim to hidden dim in the feed-forward layer)",
    )
    parser.add_argument(
        "-no-glu",
        action="store_true",
        help="Benchmark the feed-forward layer without GLU activation (default is with GLU)",
    )
    parser.add_argument(
        "-M",
        type=int,
        default=None,
        help="M dim of model benchmark if only one model is under test",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        metavar=("DIM"),
        help="user-defined shape to benchmark. Can be 3D (M, N, K) or 4D (B, M, N, K) for batched kernels.",
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        help="Print VGPR usage for Triton kernels.",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    return parser
