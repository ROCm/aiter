# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
import argparse
import functools
import logging

# PyTorch
import torch

# Triton
import triton
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

# AITER: Triton kernel wrappers
from aiter.ops.triton.gemm.grouped.gmm import (
    gmm as triton_gmm,
)
from aiter.ops.triton.gemm.grouped.gmm import (
    nptgmm as triton_nptgmm,
)
from aiter.ops.triton.gemm.grouped.gmm import (
    ptgmm as triton_ptgmm,
)

# AITER: GMM defaults and utility functions
from aiter.ops.triton.utils.gmm_common import (
    DTYPE,
    DTYPE_STR,
    GROUP_SIZES_DTYPE,
    GROUP_SIZES_DTYPE_STR,
    NUM_GROUP_SIZES,
    RNG_SEED,
    SUPPORTED_DTYPES_STR,
    SUPPORTED_GROUP_SIZES_DTYPES_STR,
    TRANS_LHS,
    TRANS_RHS,
    dtype_from_str,
    gen_gmm_input,
    gen_gmm_output,
    gen_multiple_group_sizes,
    gen_tgmm_bias_grad,
    gen_tgmm_input,
    gen_tgmm_output,
    group_sizes_dtype_from_str,
    str_from_dtype,
    str_from_group_sizes_dtype,
)

# AITER: shared argparse helper
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
    positive_int,
)

logger = logging.getLogger(__name__)

# Benchmark.
# ------------------------------------------------------------------------------


# GMM variants.

GMM_TYPES: set[str] = {"gmm", "ptgmm", "nptgmm"}

DEFAULT_GMM: str = "gmm"
assert DEFAULT_GMM in GMM_TYPES, "Default GMM type isn't in set of GMM variants."


# Real shapes, used by real models.
# fmt: off
REAL_SHAPES: list[tuple[int, int, int, int]] = [
    #      M,     K,     N,   G
    (  49152,  1408,  2048,  64),  # deepseekv2-16B
    (3145728,  2048,  1408,   8),  # deepseekv2-16B
    ( 393216,  2048,  1408,  64),  # deepseekv2-16B
    (  32768,  6144, 16384,   8),  # Mixtral 8x22B
    (  32768, 16384,  6144,   8),  # Mixtral 8x22B
    ( 267424,  1280,  2560,  32),  # real workload, but unknown model
]
# fmt: on


# Benchmark metrics.

METRICS: set[str] = {"time", "throughput", "bandwidth"}

DEFAULT_METRIC: str = "throughput"
assert (
    DEFAULT_METRIC in METRICS
), "Default benchmark metric isn't in set of supported metrics."

METRIC_UNITS: dict[str, str] = {
    "time": "ms",
    "throughput": "tflops",
    "bandwidth": "gbps",
}
assert (
    METRIC_UNITS.keys() == METRICS
), "Mismatch between available benchmark metrics and respective units."


def select_triton_kernel(gmm_type: str):
    assert gmm_type in GMM_TYPES, "Invalid GMM type."
    if gmm_type == "gmm":
        desc, kernel_wrapper = "GMM", triton_gmm
    if gmm_type == "ptgmm":
        desc, kernel_wrapper = "persistent TGMM", triton_ptgmm
    if gmm_type == "nptgmm":
        desc, kernel_wrapper = "non-persistent TGMM", triton_nptgmm
    return desc, kernel_wrapper


def _generate_rotating_lhs_rhs(
    gmm_type: str,
    M: int,
    K: int,
    N: int,
    G: int,
    in_dtype: torch.dtype,
    group_sizes_dtype: torch.dtype,
    trans_lhs: bool,
    trans_rhs: bool,
    rng_seed: int,
    unif_group_sizes: bool,
):
    """lhs/rhs only -- the large GEMM operands, regenerated per pool entry so
    replaying through the pool defeats cache residency. The group_sizes_0
    this also returns is discarded: group_sizes variants are generated once
    (see _generate_multiple_group_sizes) and tested outside the pool, not
    per entry.
    """
    if gmm_type == "gmm":
        lhs, rhs, _ = gen_gmm_input(
            M,
            K,
            N,
            G,
            preferred_element_type=in_dtype,
            group_sizes_dtype=group_sizes_dtype,
            trans_rhs=trans_rhs,
            rng_seed=rng_seed,
            unif_group_sizes=unif_group_sizes,
        )
    else:
        lhs, rhs, _ = gen_tgmm_input(
            M,
            K,
            N,
            G,
            preferred_element_type=in_dtype,
            group_sizes_dtype=group_sizes_dtype,
            trans_lhs=trans_lhs,
            rng_seed=rng_seed,
            unif_group_sizes=unif_group_sizes,
        )
    return lhs, rhs


def _generate_multiple_group_sizes(
    gmm_type: str,
    M: int,
    K: int,
    N: int,
    G: int,
    in_dtype: torch.dtype,
    group_sizes_dtype: torch.dtype,
    trans_lhs: bool,
    trans_rhs: bool,
    rng_seed: int,
    num_group_sizes: int,
    unif_group_sizes: bool,
) -> list[torch.Tensor]:
    """The num_group_sizes variants to test against the same rotating lhs/rhs
    pool, generated once per shape -- not per pool entry, since these test
    load-balancing scenarios, not cache residency."""
    if gmm_type == "gmm":
        _, _, group_sizes_0 = gen_gmm_input(
            M,
            K,
            N,
            G,
            preferred_element_type=in_dtype,
            group_sizes_dtype=group_sizes_dtype,
            trans_rhs=trans_rhs,
            rng_seed=rng_seed,
            unif_group_sizes=unif_group_sizes,
        )
    else:
        _, _, group_sizes_0 = gen_tgmm_input(
            M,
            K,
            N,
            G,
            preferred_element_type=in_dtype,
            group_sizes_dtype=group_sizes_dtype,
            trans_lhs=trans_lhs,
            rng_seed=rng_seed,
            unif_group_sizes=unif_group_sizes,
        )
    return gen_multiple_group_sizes(
        num_group_sizes,
        M,
        G,
        group_sizes_dtype=group_sizes_dtype,
        rng_seed=None,
        group_sizes_0=group_sizes_0,
    )


def benchmark_gmm(
    bencher: RotatingBufferBench,
    gmm_type: str,
    bench_shape: tuple[int, int, int, int] | None = None,
    in_dtype: torch.dtype = DTYPE,
    out_dtype: torch.dtype = DTYPE,
    group_sizes_dtype: torch.dtype = GROUP_SIZES_DTYPE,
    trans_lhs: bool = TRANS_LHS,
    trans_rhs: bool = TRANS_RHS,
    rng_seed: int = RNG_SEED,
    num_group_sizes: int = NUM_GROUP_SIZES,
    unif_group_sizes: bool = False,
    metric: str = DEFAULT_METRIC,
    use_bias: bool = False,
    accumulate: bool = False,
    grid_dim: int | None = None,
    work_stealing: bool = False,
    output: bool = False,
) -> None:
    assert gmm_type in GMM_TYPES, "Invalid GMM type."
    assert metric in METRICS, "Invalid benchmark metric."

    desc, kernel_wrapper = select_triton_kernel(gmm_type)

    in_dtype_str: str = str_from_dtype(in_dtype)
    out_dtype_str: str = str_from_dtype(out_dtype)
    group_sizes_dtype_str: str = str_from_group_sizes_dtype(group_sizes_dtype)
    unit: str = METRIC_UNITS[metric]
    has_grid_dim: bool = gmm_type != "nptgmm" and grid_dim is not None
    has_work_stealing: bool = gmm_type == "gmm" and work_stealing
    triton_provider_desc: list[str] = [
        "triton",
        gmm_type,
        # data types
        f"i{in_dtype_str}",
        f"o{out_dtype_str}",
        f"g{group_sizes_dtype_str}",
        # layout description
        "".join("t" if trans else "n" for trans in (trans_lhs, trans_rhs)) + "n",
    ]
    # grid dim
    if has_grid_dim:
        triton_provider_desc.append(f"gd{grid_dim}")
    # work stealing
    if has_work_stealing:
        triton_provider_desc.append("ws")
    # unit of benchmark metric
    triton_provider_desc.append(unit)
    triton_provider: str = "_".join(triton_provider_desc)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "K", "N", "G"],
            x_vals=[bench_shape] if bench_shape is not None else REAL_SHAPES,
            line_arg="provider",
            line_vals=[triton_provider],
            line_names=[triton_provider],
            plot_name=triton_provider,
            args={},
            ylabel=unit,
        )
    )
    def benchmark(M: int, K: int, N: int, G: int, provider: str):
        logger.info("    (M, K, N, G) = (%d, %d, %d, %d)", M, K, N, G)

        _p_gen = functools.partial(
            _generate_rotating_lhs_rhs,
            gmm_type,
            M,
            K,
            N,
            G,
            in_dtype,
            group_sizes_dtype,
            trans_lhs,
            trans_rhs,
            rng_seed,
            unif_group_sizes,
        )

        if gmm_type == "gmm":
            out = gen_gmm_output(M, N, preferred_element_type=out_dtype)
            bias = None
            if use_bias:
                torch.manual_seed(rng_seed + 1000)  # Different seed for bias
                bias = torch.randn(G, N, dtype=in_dtype, device="cuda")
            bias_key = "bias"
        else:
            out = gen_tgmm_output(K, N, G, preferred_element_type=out_dtype)
            bias = gen_tgmm_bias_grad(K, G, with_bias_grad=True) if use_bias else None
            bias_key = "bias_grad"

        def _gen_fixed():
            return out, bias

        rot_bufs = bencher.build_pool(
            [_p_gen],
            [["lhs", "rhs"]],
            inp_gen_fixed_fns=[_gen_fixed],
            inp_fixed_keys=[["existing_out", bias_key]],
        )

        # Generated once per shape, not per pool entry: these test
        # load-balancing scenarios, not cache residency.
        multiple_group_sizes = _generate_multiple_group_sizes(
            gmm_type,
            M,
            K,
            N,
            G,
            in_dtype,
            group_sizes_dtype,
            trans_lhs,
            trans_rhs,
            rng_seed,
            num_group_sizes,
            unif_group_sizes,
        )

        tops_sum = 0.0
        gb_sum = 0.0
        ms_sum = 0.0

        for group_sizes in multiple_group_sizes:
            logger.debug(
                "      group_sizes (first 5) = %s", str(group_sizes[:5].tolist())
            )

            extra_kwargs = {
                "group_sizes": group_sizes,
                "preferred_element_type": out_dtype,
            }
            if gmm_type == "gmm":
                extra_kwargs["work_stealing"] = work_stealing
            elif gmm_type == "ptgmm" or gmm_type == "nptgmm":
                extra_kwargs["accumulate"] = accumulate
            if has_grid_dim:
                extra_kwargs["grid_dim"] = grid_dim

            kernel_fn = functools.partial(kernel_wrapper, **extra_kwargs)
            ms = bencher.bench(kernel_fn)
            ms_sum += ms

            m = torch.sum(group_sizes).item()

            # Aggregate operations, in TOps. The math is the same for GMM and TGMM.
            tops_sum += 1e-12 * 2 * m * N * K

            # Aggregate memory, in GB.
            lhs, rhs = rot_bufs[0]["lhs"], rot_bufs[0]["rhs"]
            if "tgmm" in gmm_type:
                # TGMM case.
                read_bytes = K * m * lhs.element_size() + m * N * rhs.element_size()
                write_bytes = G * K * N * out.element_size()
            else:
                # GMM case.
                read_bytes = m * K * lhs.element_size() + G * K * N * rhs.element_size()
                write_bytes = m * N * out.element_size()
            gb_sum += 1e-9 * (read_bytes + write_bytes)

        # Aggregate across all group_sizes variants: milliseconds of all
        # group sizes / number of group sizes.
        ms = round(ms_sum / num_group_sizes, 4)
        s_sum = 1e-3 * ms_sum

        # Compute TFLOPS: TOps of all group sizes / seconds of all group sizes.
        tflops = round(tops_sum / s_sum, 2)

        # Compute GB/s: GB of all group sizes / seconds of all group sizes.
        gbps = round(gb_sum / s_sum, 2)

        logger.info("      ms: %7.4f", ms)
        logger.info("      TFLOPS: %6.2f", tflops)
        logger.info("      GB/s: %6.2f", gbps)

        if metric == "time":
            return ms
        if metric == "throughput":
            return tflops
        if metric == "bandwidth":
            return gbps

    logger.info("Benchmarking Triton %s kernel:", desc)
    logger.info(
        "  input_type = %s, output_type = %s, group_sizes_type = %s, rng_seed = %d, use_bias = %s, accumulate = %s",
        in_dtype_str,
        out_dtype_str,
        group_sizes_dtype_str,
        rng_seed,
        use_bias,
        accumulate,
    )
    logger.info(
        "  trans_lhs = %s, trans_rhs = %s",
        trans_lhs,
        trans_rhs,
    )
    logger.info(
        "  num_group_sizes = %d, unif_group_sizes = %s",
        num_group_sizes,
        unif_group_sizes,
    )
    if has_grid_dim:
        logger.info("  overridden persistent grid_dim = %d", grid_dim)
    if has_work_stealing:
        logger.info("  work stealing enabled")
    logger.info(
        "  metric = %s (in %s)",
        metric,
        unit,
    )
    benchmark.run(show_plots=False, print_data=True, save_path="." if output else "")


# Command line interface parsing.
# ------------------------------------------------------------------------------


def add_trans_arg(
    parser: argparse.ArgumentParser, arg: str, default_trans: bool
) -> None:
    if default_trans:
        parser.add_argument(
            f"--no-trans-{arg}",
            action="store_false",
            dest=f"trans_{arg}",
            help=f"don't transpose {arg}, i.e. row-major {arg}",
        )
    else:
        parser.add_argument(
            f"--trans-{arg}",
            action="store_true",
            dest=f"trans_{arg}",
            help=f"transpose {arg}, i.e. column-major {arg}",
        )


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    shape_args = [args.M, args.K, args.N, args.G]
    all_none = all(arg is None for arg in shape_args)
    all_provided = all(arg is not None for arg in shape_args)

    if not all_none and not all_provided:
        raise argparse.ArgumentError(
            None,
            "M, K, N, and G must be either all provided or all absent",
        )

    if args.unif_group_sizes and args.num_group_sizes != 1:
        raise argparse.ArgumentError(
            None,
            "number of distinct group sizes must be 1 when --unif-group-sizes is used",
        )

    return args


def parse_args() -> argparse.Namespace:
    parser = get_parser(kernel_name="GMM")

    # Shape
    parser.add_argument("M", type=positive_int, nargs="?", help="number of rows")
    parser.add_argument("K", type=positive_int, nargs="?", help="shared dimension")
    parser.add_argument("N", type=positive_int, nargs="?", help="number of columns")
    parser.add_argument("G", type=positive_int, nargs="?", help="number of groups")

    # GMM type
    parser.add_argument(
        "--gmm-type",
        type=str.lower,
        choices=GMM_TYPES,
        default=DEFAULT_GMM,
        help=f"GMM variant to run: GMM, persistent TGMM, non-persistent TGMM (default: {DEFAULT_GMM})",
    )

    # Data type
    parser.add_argument(
        "--input-type",
        type=str.lower,
        choices=SUPPORTED_DTYPES_STR,
        default=DTYPE_STR,
        help=f"input data type (default: {DTYPE_STR})",
    )
    parser.add_argument(
        "--output-type",
        type=str.lower,
        choices=SUPPORTED_DTYPES_STR,
        default=DTYPE_STR,
        help=f"output data type (default: {DTYPE_STR})",
    )
    parser.add_argument(
        "--group-sizes-type",
        type=str.lower,
        choices=SUPPORTED_GROUP_SIZES_DTYPES_STR,
        default=GROUP_SIZES_DTYPE_STR,
        help=f"group sizes data type (default: {GROUP_SIZES_DTYPE_STR})",
    )

    # Transposition
    add_trans_arg(parser, "lhs", TRANS_LHS)
    add_trans_arg(parser, "rhs", TRANS_RHS)

    # Input generation
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=RNG_SEED,
        help=f"seed for random input generation (default: {RNG_SEED})",
    )
    parser.add_argument(
        "--num-group-sizes",
        type=positive_int,
        default=NUM_GROUP_SIZES,
        help=f"number of distinct random group sizes to use (default: {NUM_GROUP_SIZES})",
    )
    parser.add_argument(
        "--unif-group-sizes",
        action="store_true",
        help="evenly distributes tokens among all groups",
    )
    override_parser_action(
        parser, action="--metric", choices=METRICS, default=DEFAULT_METRIC
    )
    # Other arguments
    parser.add_argument(
        "--use-bias",
        action="store_true",
        help="use bias",
    )
    parser.add_argument(
        "--accumulate",
        action="store_true",
        help="accumulate bias gradient",
    )
    parser.add_argument(
        "--grid-dim",
        type=positive_int,
        default=None,
        help="override grid dimension config of persistent kernels",
    )
    parser.add_argument(
        "--work-stealing",
        action="store_true",
        help="enable work stealing dynamic load-balancing of persistent kernels",
    )
    parser.add_argument(
        "-o",
        action="store_true",
        help="write performance results to CSV file in the current directory",
    )
    parser.add_argument("--verbose", action="store_true", help="enable verbose output")

    try:
        return validate_args(parser.parse_args())
    except argparse.ArgumentError as arg_error:
        import sys

        parser.print_usage()
        print(f"{sys.argv[0]}: error: {arg_error}")
        sys.exit(1)


# Main function: script entry point.
# ------------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s > %(message)s",
        level=logging.INFO if args.verbose else logging.ERROR,
    )
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL + 1)

    shape = (args.M, args.K, args.N, args.G)
    in_dtype = dtype_from_str(args.input_type)
    out_dtype = dtype_from_str(args.output_type)
    group_sizes_dtype = group_sizes_dtype_from_str(args.group_sizes_type)
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    benchmark_gmm(
        bencher,
        args.gmm_type,
        bench_shape=None if all(arg is None for arg in shape) else shape,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        group_sizes_dtype=group_sizes_dtype,
        trans_lhs=args.trans_lhs,
        trans_rhs=args.trans_rhs,
        rng_seed=args.rng_seed,
        num_group_sizes=args.num_group_sizes,
        unif_group_sizes=args.unif_group_sizes,
        metric=args.metric,
        use_bias=args.use_bias,
        accumulate=args.accumulate,
        grid_dim=args.grid_dim,
        work_stealing=args.work_stealing,
        output=args.o,
    )


if __name__ == "__main__":
    main()
