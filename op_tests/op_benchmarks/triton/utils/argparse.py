import argparse
from typing import Tuple
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_available_models,
)


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
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "throughput", "bandwidth"],
        default="throughput",
        help="metric to plot",
    )
    return parser


def get_ff_args(parser: argparse.ArgumentParser) -> Tuple[dict, dict]:
    """
    Does additional processing on parser args for feed-forward blocks.
    """
    args = parser.parse_args()
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
        "-no_glu",
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
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    return parser
