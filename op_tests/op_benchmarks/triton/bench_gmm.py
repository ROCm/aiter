# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
import argparse
import logging

# PyTorch
import torch

# Triton
import triton

# AITER: GMM defaults and utility functions
from aiter.ops.triton.utils.gmm_common import (
    SUPPORTED_DTYPES_STR,
    DTYPE_STR,
    dtype_from_str,
    DTYPE,
    str_from_dtype,
    TRANS_LHS,
    TRANS_RHS,
    RNG_SEED,
    NUM_GROUP_SIZES,
    gen_gmm_tensors,
    gen_tgmm_tensors,
)

# AITER: Triton kernel wrappers
from aiter.ops.triton.gmm import (
    gmm as triton_gmm,
    ptgmm as triton_ptgmm,
    nptgmm as triton_nptgmm,
)


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
        desc, gen_tensors, kernel_wrapper = (
            "GMM",
            gen_gmm_tensors,
            triton_gmm,
        )
    if gmm_type == "ptgmm":
        desc, gen_tensors, kernel_wrapper = (
            "persistent TGMM",
            gen_tgmm_tensors,
            triton_ptgmm,
        )
    if gmm_type == "nptgmm":
        desc, gen_tensors, kernel_wrapper = (
            "non-persistent TGMM",
            gen_tgmm_tensors,
            triton_nptgmm,
        )
    return (
        desc,
        gen_tensors,
        kernel_wrapper,
    )


def benchmark_gmm(
    gmm_type: str,
    bench_shape: tuple[int, int, int, int] | None = None,
    in_dtype: torch.dtype = DTYPE,
    out_dtype: torch.dtype = DTYPE,
    trans_lhs: bool = TRANS_LHS,
    trans_rhs: bool = TRANS_RHS,
    rng_seed: int = RNG_SEED,
    num_group_sizes: int = NUM_GROUP_SIZES,
    unif_group_sizes: bool = False,
    metric: str = DEFAULT_METRIC,
    use_bias: bool = False,
    accumulate: bool = False,
) -> None:
    assert gmm_type in GMM_TYPES, "Invalid GMM type."
    assert metric in METRICS, "Invalid benchmark metric."

    desc, gen_tensors, kernel_wrapper = select_triton_kernel(gmm_type)

    in_dtype_str = str_from_dtype(in_dtype)
    out_dtype_str = str_from_dtype(out_dtype)
    dtypes_desc = f"i{in_dtype_str}_o{out_dtype_str}"
    layout_desc = (
        "".join("t" if trans else "n" for trans in (trans_lhs, trans_rhs)) + "n"
    )
    unit = METRIC_UNITS[metric]
    triton_provider = f"triton_{gmm_type}_{dtypes_desc}_{layout_desc}_{unit}"

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
        logging.info("    (M, K, N, G) = (%d, %d, %d, %d)", M, K, N, G)

        lhs, rhs, multiple_group_sizes, out, bias = gen_tensors(
            M,
            K,
            N,
            G,
            num_group_sizes,
            input_type=in_dtype,
            output_type=out_dtype,
            trans_lhs=trans_lhs,
            trans_rhs=trans_rhs,
            rng_seed=rng_seed,
            unif_group_sizes=unif_group_sizes,
            use_bias=use_bias,
        )

        quantiles = [0.5, 0.2, 0.8]
        p50_ms_sum = 0.0
        p20_ms_sum = 0.0
        p80_ms_sum = 0.0
        tops_sum = 0.0
        gb_sum = 0.0

        for group_sizes in multiple_group_sizes:
            logging.debug(
                "      group_sizes (first 5) = %s", str(group_sizes[:5].tolist())
            )

            kwargs = {
                "lhs": lhs,
                "rhs": rhs,
                "group_sizes": group_sizes,
                "preferred_element_type": out_dtype,
                "existing_out": out,
            }

            if gmm_type == "gmm":
                kwargs["bias"] = bias
            elif gmm_type == "ptgmm" or gmm_type == "nptgmm":
                kwargs["bias_grad"] = bias
                kwargs["accumulate"] = accumulate

            p50_ms, p20_ms, p80_ms = triton.testing.do_bench(
                lambda: kernel_wrapper(**kwargs),
                quantiles=quantiles,
            )

            # Aggregate time, in milliseconds.
            p50_ms_sum += p50_ms
            p20_ms_sum += p20_ms
            p80_ms_sum += p80_ms

            m = torch.sum(group_sizes).item()

            # Aggregate operations, in TOps. The math is the same for GMM and TGMM.
            tops_sum += 1e-12 * 2 * m * N * K

            # Aggregate memory, in GB.
            if "tgmm" in gmm_type:
                # TGMM case.
                read_bytes = K * m * lhs.element_size() + m * N * rhs.element_size()
                write_bytes = G * K * N * out.element_size()
            else:
                # GMM case.
                read_bytes = m * K * lhs.element_size() + G * K * N * rhs.element_size()
                write_bytes = m * N * out.element_size()
            gb_sum += 1e-9 * (read_bytes + write_bytes)

        # Compute milliseconds: milliseconds of all group sizes / number of group sizes.
        p50_ms = round(p50_ms_sum / G, 4)
        p20_ms = round(p20_ms_sum / G, 4)
        p80_ms = round(p80_ms_sum / G, 4)

        # Compute total seconds.
        p50_s_sum = 1e-3 * p50_ms_sum
        p20_s_sum = 1e-3 * p20_ms_sum
        p80_s_sum = 1e-3 * p80_ms_sum

        # Compute TFLOPS: TOps of all group sizes / seconds of all group sizes.
        p50_tflops = round(tops_sum / p50_s_sum, 2)
        p20_tflops = round(tops_sum / p80_s_sum, 2)
        p80_tflops = round(tops_sum / p20_s_sum, 2)

        # Compute GB/s: GB of all group sizes / seconds of all group sizes.
        p50_gbps = round(gb_sum / p50_s_sum, 2)
        p20_gbps = round(gb_sum / p80_s_sum, 2)
        p80_gbps = round(gb_sum / p20_s_sum, 2)

        logging.info(
            "      ms: p20 = %7.4f, p50 = %7.4f, p80 = %7.4f",
            p20_ms,
            p50_ms,
            p80_ms,
        )
        logging.info(
            "      TFLOPS: p20 = %6.2f, p50 = %6.2f, p80 = %6.2f",
            p20_tflops,
            p50_tflops,
            p80_tflops,
        )
        logging.info(
            "      GB/s: p20 = %6.2f, p50 = %6.2f, p80 = %6.2f",
            p20_gbps,
            p50_gbps,
            p80_gbps,
        )

        if metric == "time":
            return p50_ms, p20_ms, p80_ms
        if metric == "throughput":
            return p50_tflops, p20_tflops, p80_tflops
        if metric == "bandwidth":
            return p50_gbps, p20_gbps, p80_gbps

    logging.info("Benchmarking Triton %s kernel:", desc)
    logging.info(
        "  input_type = %s, output_type = %s, rng_seed = %d, use_bias = %s, accumulate = %s",
        in_dtype_str,
        out_dtype_str,
        rng_seed,
        use_bias,
        accumulate,
    )
    logging.info(
        "  trans_lhs = %s, trans_rhs = %s",
        trans_lhs,
        trans_rhs,
    )
    logging.info(
        "  num_group_sizes = %d, unif_group_sizes = %s",
        num_group_sizes,
        unif_group_sizes,
    )
    logging.info(
        "  metric = %s (in %s)",
        metric,
        unit,
    )
    benchmark.run(show_plots=False, print_data=True)


# Command line interface parsing.
# ------------------------------------------------------------------------------


def positive_int(value: str) -> int:
    error = argparse.ArgumentTypeError(f"'{value}' is not a positive integer")
    try:
        # First try to convert to float to handle ".0" decimal notation.
        float_value = float(value)
        # Check if it's a whole number (no fractional part).
        if float_value != int(float_value):
            raise error
        int_value = int(float_value)
    except ValueError:
        raise error
    if int_value <= 0:
        raise error
    return int_value


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
    parser = argparse.ArgumentParser(description="benchmark GMM Triton kernels")

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

    # Benchmark metric
    parser.add_argument(
        "--metric",
        type=str.lower,
        choices=METRICS,
        default=DEFAULT_METRIC,
        help=f"benchmark metric (default: {DEFAULT_METRIC})",
    )

    # Other arguments
    parser.add_argument("--verbose", action="store_true", help="enable verbose output")
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

    benchmark_gmm(
        args.gmm_type,
        bench_shape=None if all(arg is None for arg in shape) else shape,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        trans_lhs=args.trans_lhs,
        trans_rhs=args.trans_rhs,
        rng_seed=args.rng_seed,
        num_group_sizes=args.num_group_sizes,
        unif_group_sizes=args.unif_group_sizes,
        metric=args.metric,
        use_bias=args.use_bias,
        accumulate=args.accumulate,
    )


if __name__ == "__main__":
    main()
