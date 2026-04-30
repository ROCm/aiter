# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for mHC (manifold-constrained Hyper Connection) fused kernels.

Measures performance of Triton `mhc()` and `mhc_post()` implementations across
various input shapes and configurations, reporting time, throughput (TFLOPS),
and bandwidth.

Usage:
  python bench_mhc.py              # Benchmark mhc (pre-transformer)
  python bench_mhc.py --op post    # Benchmark mhc_post (post-transformer)
  python bench_mhc.py --op e2e     # Benchmark full pipeline (mhc → mhc_post)

- `--with-hip`: adds the HIP kernel alongside the Triton kernel.
  When passed, a silent Triton-vs-HIP correctness check runs as the first
  step of each `(M, n, C)` row, once per config.
  AssertionError on mismatch aborts the benchmark.
"""

import argparse
import logging
import sys
from itertools import product

import torch
import triton

from aiter.ops.triton.fusions.mhc import mhc, mhc_post
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.utils.mhc_ref import (
    generate_mhc_inputs,
    generate_mhc_post_inputs,
)

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

# Configure logging before importing aiter modules
logging.basicConfig(
    level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s", force=True
)


def get_benchmark_configs(args):
    """Generate list of benchmark configurations based on args."""
    configs = []

    if args.M and args.n and args.C:
        configs.append((args.M, args.n, args.C))
    else:
        # Default configurations - typical mHC usage patterns
        # Format: (M, n, C) where:
        #   M: batch/sequence length
        #   n: stream parameter (manifold dimension)
        #   C: hidden dimension per stream
        Ms = [2**i for i in range(10, 15)]
        n = 4
        Cs = [512, 4096, 2**15]
        # Sort by C (hidden dimension) to show scaling across C values
        configs = sorted(list(product(Ms, [n], Cs)), key=lambda x: (x[2], x[0]))

    return configs


def _compute_metrics(
    operation,
    M,
    n=None,
    C=None,
    sinkhorn_iters=None,
    hc_mult=None,
    hidden_size=None,
    elem_bytes=None,
):
    """Compute FLOPs and memory traffic for any operation type.

    Args:
        operation: "pre", "post", or "e2e"
        M: Batch/sequence dimension
        n: Stream parameter (for pre/e2e)
        C: Hidden dimension per stream (for pre/e2e)
        sinkhorn_iters: Sinkhorn iterations (for pre/e2e)
        hc_mult: Stream multiplier (for post, typically 4)
        hidden_size: Total hidden dimension (for post)
        elem_bytes: Element size in bytes (for post)

    Returns:
        tuple: (total_flops, total_memory)
               For operations without FLOP metric, returns (None, total_memory)
    """
    if operation == "pre":
        nC = n * C
        n_squared = n * n
        N = n_squared + 2 * n

        # FLOPs computation
        # Eq 14: matmul for 3 streams
        flops_matmul = 2.0 * M * nC * n + 2.0 * M * nC * n + 2.0 * M * nC * n_squared
        # Eq 15: RMS normalization
        flops_rms = 4.0 * M * nC
        # Apply-pre step
        flops_apply_pre = 2.0 * M * n * C
        # Eq 19: Sinkhorn-Knopp
        flops_sinkhorn = 10.0 * M * n_squared * sinkhorn_iters
        total_flops = flops_matmul + flops_rms + flops_apply_pre + flops_sinkhorn

        # Memory computation
        elem_size = 2  # bf16/fp16 = 2 bytes
        bias_size = 4  # bias is fp32 = 4 bytes
        mem_read = (
            M * nC * elem_size  # x (GEMM)
            + M * nC * elem_size  # x (apply-pre re-read)
            + nC * n * elem_size  # phi_pre
            + nC * n * elem_size  # phi_post
            + nC * n_squared * elem_size  # phi_res
            + N * bias_size  # bias
        )
        mem_write = (
            M * n * elem_size  # post_mix
            + M * n_squared * elem_size  # comb_mix
            + M * C * elem_size  # layer_input
        )
        total_memory = mem_read + mem_write

    elif operation == "post":
        # Post operation has no FLOP metric, only memory
        mix_bytes = 4  # fp32
        total_flops = None
        total_memory = (
            M * hidden_size * elem_bytes  # read x
            + M * hc_mult * hidden_size * elem_bytes  # read residual
            + M * hc_mult * mix_bytes  # read post_layer_mix
            + M * hc_mult * hc_mult * mix_bytes  # read comb_res_mix
            + M * hc_mult * hidden_size * elem_bytes  # write out
        )

    else:  # e2e
        # E2E uses rough approximations
        K = n * C
        total_flops = 2 * M * K * K * 2  # Approximate
        element_size = 2  # bf16
        total_memory = M * K * element_size * 4  # rough estimate

    return total_flops, total_memory


def _get_benchmark_config(args, operation):
    """
    Build unified benchmark configuration for any operation type.

    Args:
        args: Command-line arguments
        operation: "pre", "post", or "e2e"

    Returns:
        dict with keys: x_names, x_vals_list, metrics, line_vals, benchmark_name, palette
    """
    configs = get_benchmark_configs(args)

    # Operation-specific x_names and x_vals
    if operation == "post":
        x_names = ["M", "hidden_size"]
        x_vals_list = [(M, C) for M, n, C in configs]
    else:  # "pre" or "e2e"
        x_names = ["M", "n", "C"]
        x_vals_list = configs

    # Determine metrics based on operation and args
    if args.metric == "all" or args.metric is None:
        if operation == "pre":
            metrics = [
                "time(ms)",
                "throughput(TFLOPS)",
                "bandwidth(GB/s)",
                "arithmetic_intensity(FLOP/byte)",
            ]
        elif operation == "post":
            metrics = ["time(ms)", "bandwidth(GB/s)"]
        else:  # e2e
            metrics = ["time(ms)", "throughput(TFLOPS)", "bandwidth(GB/s)"]
    else:
        metric_map = {
            "time": "time(ms)",
            "throughput": "throughput(TFLOPS)",
            "bandwidth": "bandwidth(GB/s)",
            "arithmetic_intensity": "arithmetic_intensity(FLOP/byte)",
        }
        metrics = [metric_map.get(args.metric, "throughput(TFLOPS)")]

    # Build provider names
    backends = ["triton"] + (["hip"] if args.with_hip else [])
    if args.with_hip:
        line_vals = [f"{b}_{m}" for m in metrics for b in backends]
    else:
        line_vals = list(metrics)

    # Benchmark name
    if operation == "pre":
        benchmark_name = get_caller_name_no_ext()
        benchmark_name += f"_sinkhorn-{args.sinkhorn_iters}iters"
    elif operation == "post":
        benchmark_name = f"bench_mhc_post_{args.dtype}"
    else:  # e2e
        benchmark_name = f"mhc-e2e-{args.dtype}"

    if args.with_hip:
        benchmark_name += "_triton+hip"

    # Palette
    _palette = [
        ("red", "-"),
        ("blue", "-"),
        ("yellow", "-"),
        ("green", "-"),
        ("red", "--"),
        ("blue", "--"),
        ("yellow", "--"),
        ("green", "--"),
    ]

    return {
        "x_names": x_names,
        "x_vals_list": x_vals_list,
        "metrics": metrics,
        "line_vals": line_vals,
        "benchmark_name": benchmark_name,
        "palette": _palette[: len(line_vals)],
    }


def _triton_to_hip_pre_inputs(x, phi, alpha_pre, alpha_post, alpha_res, bias, n):
    """Convert Triton-convention mhc inputs to HIP aiter.mhc_pre conventions.

    Mapping:
      M <-> m                  n <-> hc_mult              C <-> hidden_size
      x (M, K=n*C)             <-> residual (m, hc_mult, hidden_size) bf16
      phi (K, 2n+n^2)          <-> fn.T (fn is (hc_mult3, hc_hidden_size)) fp32
      (alpha_pre/post/res)     <-> hc_scale (3,) fp32
      bias                     <-> hc_base (hc_mult3,) fp32
    """
    M, K = x.shape
    C = K // n
    residual = x.view(M, n, C).contiguous().to(torch.bfloat16)
    fn_hip = phi.T.contiguous().float()
    hc_scale = torch.tensor(
        [alpha_pre, alpha_post, alpha_res], dtype=torch.float32, device=x.device
    )
    hc_base = bias.to(torch.float32).contiguous()
    return residual, fn_hip, hc_scale, hc_base


def _create_benchmark_kernel(args, operation):
    """
    Factory function that creates the appropriate benchmark kernel based on operation type.

    Args:
        args: Command-line arguments
        operation: "pre", "post", or "e2e"

    Returns:
        Decorated benchmark function ready to run
    """
    dtype = arg_to_torch_dtype[args.dtype]
    sinkhorn_iters = args.sinkhorn_iters
    hc_mult = 4  # Always 4 for mHC

    # Get benchmark configuration
    config = _get_benchmark_config(args, operation)

    # Create Benchmark object
    benchmark = triton.testing.Benchmark(
        x_names=config["x_names"],
        x_vals=config["x_vals_list"],
        line_arg="provider",
        line_vals=config["line_vals"],
        line_names=config["line_vals"],
        styles=config["palette"],
        ylabel="",
        plot_name=config["benchmark_name"],
        args={},
    )

    # Track configs that have passed correctness check
    _checked_configs = set()

    @triton.testing.perf_report([benchmark])
    def bench_mhc_kernel(provider, **benchmark_params):
        """Unified benchmark kernel for all mHC operations."""

        # === Operation-specific setup ===
        if operation == "pre":
            M = benchmark_params["M"]
            n = benchmark_params["n"]
            C = benchmark_params["C"]
            (
                x,
                phi,
                alpha_pre,
                alpha_post,
                alpha_res,
                bias,
                n_streams,
            ) = generate_mhc_inputs(M, n, C, dtype)

            # Compute metrics
            total_flops, total_mem = _compute_metrics(
                "pre", M, n=n, C=C, sinkhorn_iters=sinkhorn_iters
            )

            def triton_fn():
                return mhc(
                    x,
                    phi,
                    alpha_pre,
                    alpha_post,
                    alpha_res,
                    bias,
                    n_streams,
                    sinkhorn_iters=sinkhorn_iters,
                )

            hip_fn = None
            if args.with_hip:
                import aiter  # noqa: F401

                residual, fn_hip, hc_scale, hc_base = _triton_to_hip_pre_inputs(
                    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams
                )
                hip_device_ctx = torch.device(residual.device)

                def hip_fn():
                    with hip_device_ctx:
                        return aiter.mhc_pre(
                            residual,
                            fn_hip,
                            hc_scale,
                            hc_base,
                            rms_eps=1e-6,
                            hc_pre_eps=0.0,
                            hc_sinkhorn_eps=0.0,
                            hc_post_mult_value=2.0,
                            sinkhorn_repeat=sinkhorn_iters,
                        )

            config_key = (M, n, C)
            if args.with_hip and config_key not in _checked_configs:
                _assert_triton_matches_hip(triton_fn(), hip_fn(), "pre", M=M, n=n, C=C)
                _checked_configs.add(config_key)

        elif operation == "post":
            M = benchmark_params["M"]
            hidden_size = benchmark_params["hidden_size"]
            x, residual, post_mix, comb_mix = generate_mhc_post_inputs(
                M, hc_mult, hidden_size, dtype
            )

            # Compute metrics
            elem_bytes = x.element_size()
            total_flops, total_mem = _compute_metrics(
                "post",
                M,
                hc_mult=hc_mult,
                hidden_size=hidden_size,
                elem_bytes=elem_bytes,
            )

            def triton_fn():
                return mhc_post(None, x, residual, post_mix, comb_mix)

            hip_fn = None
            if args.with_hip:
                import aiter

                out_hip = torch.empty(
                    M, hc_mult, hidden_size, dtype=dtype, device=x.device
                )

                def hip_fn():
                    return aiter.mhc_post(out_hip, x, residual, post_mix, comb_mix)

            config_key = (M, hidden_size)
            if args.with_hip and config_key not in _checked_configs:
                _assert_triton_matches_hip(
                    triton_fn(), hip_fn(), "post", M=M, hidden_size=hidden_size
                )
                _checked_configs.add(config_key)

        else:  # e2e
            M = benchmark_params["M"]
            n = benchmark_params["n"]
            C = benchmark_params["C"]
            x_l_flat, phi, alpha_pre, alpha_post, alpha_res, bias, _ = (
                generate_mhc_inputs(M, n, C, dtype)
            )

            # Compute metrics
            total_flops, total_mem = _compute_metrics("e2e", M, n=n, C=C)

            def triton_fn():
                h_post, h_res, layer_input = mhc(
                    x_l_flat,
                    phi,
                    alpha_pre,
                    alpha_post,
                    alpha_res,
                    bias,
                    n,
                    1e-6,  # eps
                    1e-6,  # hc_pre_eps
                    1.0,  # hc_post_mult_value
                    sinkhorn_iters,
                    None,  # config
                )
                x_l = x_l_flat.view(M, n, C)
                x_l_plus_1 = mhc_post(None, layer_input, x_l, h_post, h_res, None)
                return x_l_plus_1

            hip_fn = None  # E2E doesn't support HIP yet

        # === Common benchmark execution ===
        backend = "hip" if provider.startswith("hip_") else "triton"
        fn = hip_fn if backend == "hip" else triton_fn

        # Benchmark
        ms = triton.testing.do_bench(fn)

        # Return requested metric based on provider string
        if "ms" in provider or "time" in provider:
            return ms
        elif "TFLOPS" in provider or "tflops" in provider or "throughput" in provider:
            if total_flops is not None:
                return total_flops / (ms * 1e-3) / 1e12
            return 0.0
        elif "GB/s" in provider or "gbps" in provider or "bandwidth" in provider:
            return total_mem / (ms * 1e-3) / 1e9
        elif "arithmetic_intensity" in provider:
            return total_flops / total_mem
        return ms

    return bench_mhc_kernel


def run_benchmark(args, operation):
    """Run mHC benchmark with specified configuration and operation type."""
    bench_fn = _create_benchmark_kernel(args, operation=operation)
    bench_fn.run(save_path="." if args.o else None, print_data=True)


# Benchmark - Validation
# =============================================================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="Benchmark mHC",
        description="Benchmark mHC (manifold-constrained Hyper Connection) kernels",
        allow_abbrev=False,
    )

    # Operation selection
    parser.add_argument(
        "--op",
        type=str,
        default="pre",
        choices=["pre", "post", "e2e"],
        help="Which operation to benchmark: 'pre' (mhc), 'post' (mhc_post), or 'e2e' (full pipeline)",
    )

    # Shape parameters
    parser.add_argument(
        "-M",
        type=int,
        default=None,
        help="Batch/sequence dimension (default: run suite of configs)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Stream parameter (manifold dimension, typically 4)",
    )
    parser.add_argument(
        "-C",
        type=int,
        default=None,
        help="Hidden dimension per stream (typically 1024)",
    )

    # Kernel configuration
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for computation",
    )
    parser.add_argument(
        "-sinkhorn_iters",
        type=int,
        default=20,
        help="Number of Sinkhorn-Knopp iterations for mhc (default: 20)",
    )
    parser.add_argument(
        "--with-hip",
        dest="with_hip",
        action="store_true",
        default=False,
        help=(
            "Also benchmark the HIP kernel alongside Triton. "
            "For --op pre: requires --dtype bf16. "
            "Runs a silent Triton-vs-HIP correctness check once per (M, n, C) "
            "before timing that row; aborts on mismatch."
        ),
    )

    # Output options
    parser.add_argument(
        "-metric",
        nargs="?",
        const="throughput",
        choices=["all", "time", "throughput", "bandwidth", "arithmetic_intensity"],
        default=None,
        help="Metrics for the kernel benchmark (default: all for pre, time+bandwidth for post)",
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

    return parser.parse_args()


def _assert_triton_matches_hip(triton_out, hip_out, operation, **params):
    """Unified Triton-vs-HIP parity checker for all operations.

    Args:
        triton_out: Triton kernel output
        hip_out: HIP kernel output
        operation: "pre" or "post"
        **params: Config parameters (M, n, C for pre; M, hidden_size for post)

    Tolerances from op_tests/triton_tests/fusions/test_mhc.py:
        pre - post_mix:    atol=4e-2, rtol=1e-2
        pre - comb_mix:    atol=4e-2, rtol=1e-2
        pre - layer_input: atol=8e-2, rtol=2e-2
        post - output:     atol=2e-2, rtol=1e-2
    """
    if operation == "pre":
        M, n, C = params["M"], params["n"], params["C"]
        post_t, comb_t, li_t = triton_out
        post_h, comb_h, li_h = hip_out

        cfg = f"(M={M}, n={n}, C={C})"
        torch.testing.assert_close(
            post_t.float(),
            post_h.float(),
            atol=4e-2,
            rtol=1e-2,
            msg=f"post_mix mismatch between Triton and HIP at {cfg}",
        )
        torch.testing.assert_close(
            comb_t.float(),
            comb_h.float(),
            atol=4e-2,
            rtol=1e-2,
            msg=f"comb_mix mismatch between Triton and HIP at {cfg}",
        )
        torch.testing.assert_close(
            li_t.float(),
            li_h.float(),
            atol=8e-2,
            rtol=2e-2,
            msg=f"layer_input mismatch between Triton and HIP at {cfg}",
        )
    elif operation == "post":
        M, hidden_size = params["M"], params["hidden_size"]
        cfg = f"(M={M}, hidden_size={hidden_size})"
        torch.testing.assert_close(
            triton_out.float(),
            hip_out.float(),
            atol=2e-2,
            rtol=1e-2,
            msg=f"mhc_post output mismatch between Triton and HIP at {cfg}",
        )


def _validate_with_hip_pre(args):
    """Validate --with-hip arguments for mhc_pre operation.

    Returns 0 if args are acceptable, or a non-zero exit code suitable for sys.exit().
    """
    for M_, n_, C_ in get_benchmark_configs(args):
        hc_hidden_size = n_ * C_
        # aiter.mhc_pre_gemm_sqrsum: hc_hidden_size % tile_k == 0 for tile_k
        # in {64, 128}. The Python dispatcher always selects a valid tile_k
        # iff hc_hidden_size is at least 64-aligned.
        if hc_hidden_size % 64 != 0:
            logging.error(
                "--with-hip requires n*C (hc_hidden_size) divisible by 64 "
                "(got n=%d, C=%d, n*C=%d for M=%d). aiter.mhc_pre_gemm_sqrsum "
                "requires hc_hidden_size %% tile_k == 0 for tile_k in {64, 128}.",
                n_,
                C_,
                hc_hidden_size,
                M_,
            )
            return 1

        # aiter.mhc_pre_big_fuse MHC_PRE_BIG_FUSE_KERNEL_DISPATCH:
        #   m <= cu_num*12  -> residual_block = 256 (needs C % 256 == 0, C >= 512)
        #   m >  cu_num*12  -> residual_block = 128 (needs C % 128 == 0, C >= 256)
        # Use the strictest condition so the check is independent of cu_num
        # (validated by the TORCH_CHECK inside MHC_PRE_BIG_FUSE_KERNEL_IMPL).
        if C_ % 128 != 0 or C_ < 512:
            logging.error(
                "--with-hip requires C (hidden_size) divisible by 128 and "
                ">= 512 (got C=%d for M=%d, n=%d). aiter.mhc_pre_big_fuse "
                "dispatches with residual_block in {128, 256} and enforces "
                "hidden_size %% residual_block == 0 && hidden_size >= "
                "residual_block * 2 via TORCH_CHECK.",
                C_,
                M_,
                n_,
            )
            return 1

    return 0


def _validate_with_hip_post(args):
    """Validate --with-hip arguments for mhc_post operation.

    Returns 0 if args are acceptable, or a non-zero exit code suitable for sys.exit().
    """
    for M_, n_, C_ in get_benchmark_configs(args):
        # aiter.mhc_post MHC_POST_KERNEL_DISPATCH requirements:
        # - non-gfx942 + hidden_size % 1024 == 0 -> residual_block=1024, needs hidden_size >= 2048
        # - hidden_size % 512 == 0 -> residual_block=512, needs hidden_size >= 1024
        # - hidden_size % 256 == 0 -> residual_block=256, needs hidden_size >= 512
        # The macro enforces: hidden_size % residual_block == 0 && hidden_size >= residual_block * 2
        if C_ % 256 != 0:
            logging.error(
                "--with-hip requires C (hidden_size) divisible by 256 "
                "(got C=%d for M=%d, n=%d). aiter.mhc_post dispatches with "
                "residual_block in {256, 512, 1024}.",
                C_,
                M_,
                n_,
            )
            return 1

        # Check the residual_block * 2 constraint based on dispatch logic
        import aiter.jit.utils.chip_info as chip_info

        arch_id = chip_info.get_gpu_arch()

        if arch_id != "gfx942" and C_ % 1024 == 0:
            # Will use residual_block=1024
            if C_ < 2048:
                logging.error(
                    "--with-hip: hidden_size %% 1024 == 0 on %s selects residual_block=1024, "
                    "requires hidden_size >= 2048 (got C=%d for M=%d, n=%d).",
                    arch_id,
                    C_,
                    M_,
                    n_,
                )
                return 1
        elif C_ % 512 == 0:
            # Will use residual_block=512
            if C_ < 1024:
                logging.error(
                    "--with-hip: hidden_size %% 512 == 0 selects residual_block=512, "
                    "requires hidden_size >= 1024 (got C=%d for M=%d, n=%d).",
                    C_,
                    M_,
                    n_,
                )
                return 1
        else:
            # Will use residual_block=256
            if C_ < 512:
                logging.error(
                    "--with-hip: hidden_size %% 256 == 0 selects residual_block=256, "
                    "requires hidden_size >= 512 (got C=%d for M=%d, n=%d).",
                    C_,
                    M_,
                    n_,
                )
                return 1

    return 0


def _validate_with_hip(args, operation="pre"):
    """Validate --with-hip arguments for any operation type.

    Returns 0 if args are acceptable, or a non-zero exit code suitable for sys.exit().
    """
    if not args.with_hip:
        return 0

    # Common validations
    if args.dtype != "bf16":
        kernel_name = "aiter.mhc_pre" if operation == "pre" else "aiter.mhc_post"
        logging.error(
            "--with-hip only supports --dtype bf16 (got %r). "
            "%s kernel is template-instantiated for bf16 "
            "residual with fp32 parameters.",
            args.dtype,
            kernel_name,
        )
        return 1

    # Check n == 4 for all operations
    for M_, n_, C_ in get_benchmark_configs(args):
        if n_ != 4:
            kernel_name = (
                "aiter.mhc_pre_big_fuse" if operation == "pre" else "aiter.mhc_post"
            )
            logging.error(
                "--with-hip requires n == 4 (got n=%d for M=%d, C=%d). "
                "%s hardcodes hc_mult == 4 via TORCH_CHECK.",
                n_,
                M_,
                C_,
                kernel_name,
            )
            return 1

    # Operation-specific validations
    if operation == "pre":
        return _validate_with_hip_pre(args)
    elif operation == "post":
        return _validate_with_hip_post(args)
    else:  # e2e
        return 0  # No HIP validation for E2E yet


# Main Entry Points
# =============================================================================
def main():
    """Main entry point."""
    args = parse_args()

    # Dispatch table for operation-specific handlers
    operation_dispatch = {
        "pre": {"vgpr_msg": "Retrieving VGPR usage for mHC Triton kernels..."},
        "post": {"vgpr_msg": "Retrieving VGPR usage for mhc_post Triton kernels..."},
        "e2e": {"vgpr_msg": "Retrieving VGPR usage for mhc E2E Triton kernels..."},
    }

    op_config = operation_dispatch[args.op]

    # Validate HIP arguments (unified validator)
    rc = _validate_with_hip(args, operation=args.op)
    if rc != 0:
        return rc

    # Handle VGPR printing
    if args.print_vgpr:
        print(op_config["vgpr_msg"])
        print_vgpr(lambda: run_benchmark(args, args.op), get_caller_name_no_ext())
        return 0

    # Run the benchmark
    run_benchmark(args, args.op)
    return 0


if __name__ == "__main__":
    sys.exit(main())
