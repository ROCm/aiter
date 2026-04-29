# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for mHC (manifold-constrained Hyper Connection) fused kernel.

Measures performance of the Triton `mhc()` implementation across various
input shapes and configurations, reporting time, throughput (TFLOPS), and
bandwidth.

- `--with-hip`: adds the HIP kernel aiter.mhc_pre alongside the Triton kernel.
  When passed, a silent Triton-vs-HIP correctness check runs as the first
  step of each `(M, n, C)` row, once per config.
  AssertionError on mismatch aborts the benchmark.
"""

import sys
import argparse
import logging
from itertools import product
import torch
import triton

from aiter.ops.triton.fusions.mhc import mhc
from aiter.test_common import checkAllclose
from op_tests.triton_tests.utils.mhc_ref import generate_mhc_inputs
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    print_vgpr,
    get_caller_name_no_ext,
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


def run_benchmark(args):
    """Run mHC benchmark with specified configuration."""
    dtype = arg_to_torch_dtype[args.dtype]
    sinkhorn_iters = args.sinkhorn_iters

    configs = get_benchmark_configs(args)
    x_vals_list = configs
    x_names = ["M", "n", "C"]

    # Determine which metrics to report (following bench_diffusion_attention pattern)
    if args.metric == "all" or args.metric is None:
        metrics = [
            "time(ms)",
            "throughput(TFLOPS)",
            "bandwidth(GB/s)",
            "arithmetic_intensity(FLOP/byte)",
        ]
    else:
        metric_map = {
            "time": "time(ms)",
            "throughput": "throughput(TFLOPS)",
            "bandwidth": "bandwidth(GB/s)",
            "arithmetic_intensity": "arithmetic_intensity(FLOP/byte)",
        }
        metrics = [metric_map.get(args.metric, "throughput(TFLOPS)")]

    backends = ["triton"] + (["hip"] if args.with_hip else [])
    if args.with_hip:
        line_vals = [f"{b}_{m}" for m in metrics for b in backends]
    else:
        line_vals = list(metrics)

    benchmark_name = get_caller_name_no_ext()
    benchmark_name += f"_sinkhorn-{sinkhorn_iters}iters"
    if args.with_hip:
        benchmark_name += "_triton+hip"

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
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=_palette[: len(line_vals)],
        ylabel="",
        plot_name=benchmark_name,
        args={},
    )

    _checked_configs: set = set()

    @triton.testing.perf_report([benchmark])
    def bench_mhc_kernel(M, n, C, provider):
        """Benchmark mHC kernel for given configuration."""
        (
            x,
            phi,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            n_streams,
        ) = generate_mhc_inputs(M, n, C, dtype)

        # Compute FLOPs for mHC operation
        nC = n * C
        n_squared = n * n
        N = n_squared + 2 * n

        # Standard GEMM FLOPs (2*M*N*K for matrix multiply)
        # Eq 14: x @ phi for 3 streams
        # - x @ phi_pre: (M, nC) @ (nC, n) = 2*M*nC*n
        # - x @ phi_post: (M, nC) @ (nC, n) = 2*M*nC*n
        # - x @ phi_res: (M, nC) @ (nC, n²) = 2*M*nC*n²
        flops_matmul = 2.0 * M * nC * n + 2.0 * M * nC * n + 2.0 * M * nC * n_squared

        # Eq 15: RMS normalization - M rows, each with nC elements
        # sqrt(sum(x^2)/K) requires: square + sum + divide + sqrt ≈ 4*nC ops per row
        flops_rms = 4.0 * M * nC

        # Apply-pre step:
        # layer_input[m, c] = Σᵢ pre_mix[m, i] * x[m, i*C + c]
        # = 2*M*n*C FLOPs (multiply-add for each (m, i, c))
        flops_apply_pre = 2.0 * M * n * C

        # Eq 19: Sinkhorn-Knopp
        # Each iteration: 2 normalizations (row + col) on M matrices of size (n, n)
        # Simplified: ~10*n² per iteration (accounting for expensive exp/log ops)
        flops_sinkhorn = 10.0 * M * n_squared * sinkhorn_iters

        total_flops = flops_matmul + flops_rms + flops_apply_pre + flops_sinkhorn

        # Compute memory traffic
        elem_size = 2  # bf16/fp16 = 2 bytes
        bias_size = 4  # bias is fp32 = 4 bytes

        # Memory reads:
        # - x: (M, nC) - read once for the GEMM
        # - x re-read by the apply-pre step: (M, n*C) = (M, nC) again
        # - phi_pre, phi_post, phi_res: (nC, n), (nC, n), (nC, n_res)
        # - bias: (N,) in fp32
        mem_read = (
            M * nC * elem_size  # x (GEMM)
            + M * nC * elem_size  # x (apply-pre re-read)
            + nC * n * elem_size  # phi_pre
            + nC * n * elem_size  # phi_post
            + nC * n_squared * elem_size  # phi_res
            + N * bias_size  # bias
        )

        # Memory writes:
        # - post_mix: (M, n)
        # - comb_mix doubly-stochastic Sinkhorn output: (M, n_squared)
        # - layer_input: (M, C) - replaces the old H^pre write
        mem_write = (
            M * n * elem_size  # post_mix
            + M * n_squared * elem_size  # comb_mix
            + M * C * elem_size  # layer_input
        )

        total_mem = mem_read + mem_write

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
            import aiter  # noqa: F401  (side-effect: register module_mhc ops)

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
                        hc_post_mult_value=2.0,  # parity with Triton's 2*sigmoid(H_post)
                        sinkhorn_repeat=sinkhorn_iters,
                    )

        if args.with_hip and (M, n, C) not in _checked_configs:
            post_t, comb_t, li_t = triton_fn()
            post_h, comb_h, li_h = hip_fn()
            cfg = f"(M={M}, n={n}, C={C})"
            for name, t, h, atol, rtol in (
                ("post_mix", post_t, post_h, 4e-2, 1e-2),
                ("comb_mix", comb_t, comb_h, 4e-2, 1e-2),
                ("layer_input", li_t, li_h, 8e-2, 2e-2),
            ):
                msg = f"{name} mismatch between Triton and HIP at {cfg}"
                pct = checkAllclose(
                    t.detach().cpu().float(),
                    h.detach().cpu().float(),
                    atol=atol,
                    rtol=rtol,
                    tol_err_ratio=0.05,
                    msg=msg,
                    printLog=True,
                )
                assert pct <= 0.05, (
                    f"{msg} (atol={atol:g}, rtol={rtol:g}, bad_element_ratio={pct:.2%})"
                )
            _checked_configs.add((M, n, C))

        backend = "hip" if provider.startswith("hip_") else "triton"
        fn = hip_fn if backend == "hip" else triton_fn

        # Benchmark
        ms = triton.testing.do_bench(fn)

        # Return requested metric based on provider string (following bench_diffusion_attention pattern)
        if "ms" in provider:
            return ms
        elif "TFLOPS" in provider:
            return total_flops / (ms * 1e-3) / 1e12
        elif "GB/s" in provider:
            return total_mem / (ms * 1e-3) / 1e9
        elif "arithmetic_intensity" in provider:
            return total_flops / total_mem
        return ms

    bench_mhc_kernel.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="Benchmark mHC",
        description="Benchmark mHC (manifold-constrained Hyper Connection) kernel",
        allow_abbrev=False,
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
        help="Number of Sinkhorn-Knopp iterations (default: 20)",
    )
    parser.add_argument(
        "--with-hip",
        dest="with_hip",
        action="store_true",
        default=False,
        help=(
            "Also benchmark the HIP aiter.mhc_pre kernel "
            "(mhc_pre_gemm_sqrsum + mhc_pre_big_fuse) alongside Triton. "
            "Requires --dtype bf16. Also runs a silent Triton-vs-HIP "
            "correctness check once per (M, n, C) before timing that row; "
            "aborts on mismatch."
        ),
    )

    # Output options
    parser.add_argument(
        "-metric",
        nargs="?",
        const="throughput",
        choices=["all", "time", "throughput", "bandwidth", "arithmetic_intensity"],
        default=None,
        help="Metrics for the kernel benchmark (default: all)",
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


def _validate_with_hip(args):
    """Reject unsupported --with-hip combinations up-front with kernel-named errors.

    Returns 0 if args are acceptable, or a non-zero exit code suitable for sys.exit().
    """
    if not args.with_hip:
        return 0

    if args.dtype != "bf16":
        logging.error(
            "--with-hip only supports --dtype bf16 (got %r). "
            "aiter.mhc_pre_gemm_sqrsum is template-instantiated for bf16 "
            "residual only (with fp32 fn/hc_scale/hc_base).",
            args.dtype,
        )
        return 1

    # Validate every (M, n, C) that will actually be benchmarked. This
    # covers both the explicit "-M -n -C" case and the default suite, so we
    # catch e.g. the default C=128 case (hidden_size below the HIP kernel's
    # residual_block * 2 = 512 lower bound) before any kernel launch.
    for M_, n_, C_ in get_benchmark_configs(args):
        # aiter.mhc_pre_big_fuse hardcodes hc_mult == 4 via TORCH_CHECK.
        if n_ != 4:
            logging.error(
                "--with-hip requires n == 4 (got n=%d for M=%d, C=%d). "
                "aiter.mhc_pre_big_fuse hardcodes hc_mult == 4 "
                "(static_assert in mhc_pre_big_fuse_kernel + runtime "
                "TORCH_CHECK in mhc_pre_big_fuse).",
                n_,
                M_,
                C_,
            )
            return 1

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


def main():
    """Main entry point."""
    args = parse_args()

    rc = _validate_with_hip(args)
    if rc != 0:
        return rc

    if args.print_vgpr:
        print("Retrieving VGPR usage for mHC Triton kernels...")

        def fun():
            return run_benchmark(args)

        print_vgpr(fun, get_caller_name_no_ext())
        return 0

    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
