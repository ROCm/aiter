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

- `--with-hip`: adds the HIP kernel alongside the Triton kernel.
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

    # Tracks (configs that have already passed the silent Triton-vs-HIP
    # parity check within this benchmark run, so each
    # config is checked at most once even when multiple providers/metrics
    # are reported per row.
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

        # Apply-pre step (fused in both Triton and HIP):
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
            _assert_triton_matches_hip(triton_fn(), hip_fn(), M=M, n=n, C=C)
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
        description="Benchmark mHC (manifold-constrained Hyper Connection) kernels",
        allow_abbrev=False,
    )

    # Operation selection
    parser.add_argument(
        "--op",
        type=str,
        default="pre",
        choices=["pre", "post"],
        help="Which operation to benchmark: 'pre' (mhc) or 'post' (mhc_post)",
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


def _assert_triton_matches_hip(triton_out, hip_out, *, M, n, C):
    """Silent Triton-vs-HIP parity assertions on already-computed output tuples.

    Runs only the three `torch.testing.assert_close` calls; callers are
    responsible for invoking `mhc()` and `aiter.mhc_pre()` with parity-aligned
    parameters and passing the resulting 3-tuples here. Silent on success;
    raises AssertionError on mismatch with a descriptive `(M, n, C)` tag.

    Tolerances are reused verbatim from the parity test
    `test_triton_mhc_matches_hip` in
    `op_tests/triton_tests/fusions/test_mhc.py`, where they were
    empirically calibrated to be robust across 10 seeds:
        post_mix:    atol=4e-2, rtol=1e-2
        comb_mix:    atol=4e-2, rtol=1e-2
        layer_input: atol=8e-2, rtol=2e-2  (widest because pre_mix is
                                            bf16-quantized in HIP before
                                            the apply-pre step).
    """
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


# =============================================================================
# mhc_post Benchmark
# =============================================================================


def run_post_benchmark(args):
    """Run mhc_post benchmark with specified configuration."""
    from aiter.ops.triton.fusions.mhc import mhc_post
    from op_tests.triton_tests.utils.mhc_ref import generate_mhc_post_inputs

    dtype = arg_to_torch_dtype[args.dtype]
    hc_mult = 4  # Always 4 for mHC

    configs = get_benchmark_configs(args)
    x_vals_list = [(M, C) for M, n, C in configs]  # Use (M, hidden_size)
    x_names = ["M", "hidden_size"]

    # Determine metrics
    if args.metric == "all" or args.metric is None:
        metrics = ["time(ms)", "bandwidth(GB/s)"]
    else:
        metric_map = {
            "time": "time(ms)",
            "bandwidth": "bandwidth(GB/s)",
        }
        metrics = [metric_map.get(args.metric, "bandwidth(GB/s)")]

    backends = ["triton"] + (["hip"] if args.with_hip else [])
    if args.with_hip:
        line_vals = [f"{b}_{m}" for m in metrics for b in backends]
    else:
        line_vals = list(metrics)

    benchmark_name = f"bench_mhc_post_{args.dtype}"
    if args.with_hip:
        benchmark_name += "_triton+hip"

    _palette = [
        ("blue", "-"),
        ("green", "-"),
        ("blue", "--"),
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

    # Track configs that have already passed correctness check
    _checked_configs: set = set()

    @triton.testing.perf_report([benchmark])
    def bench_mhc_post_kernel(M, hidden_size, provider):
        """Benchmark mhc_post kernel for given configuration."""
        # Generate inputs
        x, residual, post_mix, comb_mix = generate_mhc_post_inputs(
            M, hc_mult, hidden_size, dtype
        )

        # Memory traffic calculation
        elem_bytes = x.element_size()  # 2 for bf16/fp16
        mix_bytes = 4  # fp32
        bytes_io = (
            M * hidden_size * elem_bytes  # read x
            + M * hc_mult * hidden_size * elem_bytes  # read residual
            + M * hc_mult * mix_bytes  # read post_layer_mix
            + M * hc_mult * hc_mult * mix_bytes  # read comb_res_mix
            + M * hc_mult * hidden_size * elem_bytes  # write out
        )

        def triton_fn():
            return mhc_post(None, x, residual, post_mix, comb_mix)

        hip_fn = None
        if args.with_hip:
            import aiter

            out_hip = torch.empty(M, hc_mult, hidden_size, dtype=dtype, device=x.device)

            def hip_fn():
                return aiter.mhc_post(out_hip, x, residual, post_mix, comb_mix)

        # Silent correctness check (once per config)
        if args.with_hip and (M, hidden_size) not in _checked_configs:
            _assert_triton_matches_hip_post(
                triton_fn(), hip_fn(), M=M, hidden_size=hidden_size
            )
            _checked_configs.add((M, hidden_size))

        backend = "hip" if provider.startswith("hip_") else "triton"
        fn = hip_fn if backend == "hip" else triton_fn

        # Benchmark
        ms = triton.testing.do_bench(fn)

        # Return requested metric based on provider string
        if "ms" in provider:
            return ms
        elif "GB/s" in provider:
            return bytes_io / (ms * 1e-3) / 1e9
        return ms

    bench_mhc_post_kernel.run(save_path="." if args.o else None, print_data=True)


def _validate_with_hip_post(args):
    """Validate args for --with-hip in mhc_post benchmark.

    Returns 0 if args are acceptable, or a non-zero exit code for sys.exit().
    """
    if not args.with_hip:
        return 0

    if args.dtype != "bf16":
        logging.error(
            "--with-hip only supports --dtype bf16 (got %r). "
            "aiter.mhc_post kernel is template-instantiated for bf16/fp16 "
            "with fp32 mixing coefficients.",
            args.dtype,
        )
        return 1

    # Validate every (M, n, C) that will be benchmarked
    for M_, n_, C_ in get_benchmark_configs(args):
        # aiter.mhc_post hardcodes hc_mult == 4 via TORCH_CHECK
        if n_ != 4:
            logging.error(
                "--with-hip requires n == 4 (got n=%d for M=%d, C=%d). "
                "aiter.mhc_post hardcodes hc_mult == 4 via TORCH_CHECK.",
                n_,
                M_,
                C_,
            )
            return 1

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


def _assert_triton_matches_hip_post(triton_out, hip_out, *, M, hidden_size):
    """Silent Triton-vs-HIP parity assertions for mhc_post.

    Runs torch.testing.assert_close on the output tensors. Silent on success;
    raises AssertionError on mismatch with a descriptive (M, hidden_size) tag.

    Tolerances match the test_mhc_post_matches_hip test in
    op_tests/triton_tests/fusions/test_mhc.py:
        out: atol=2e-2, rtol=1e-2
    """
    cfg = f"(M={M}, hidden_size={hidden_size})"
    torch.testing.assert_close(
        triton_out.float(),
        hip_out.float(),
        atol=2e-2,
        rtol=1e-2,
        msg=f"mhc_post output mismatch between Triton and HIP at {cfg}",
    )


# =============================================================================
# Main Entry Points
# =============================================================================


def main():
    """Main entry point."""
    args = parse_args()

    # Route to appropriate benchmark based on --op flag
    if args.op == "post":
        return main_post(args)
    else:
        return main_pre(args)


def main_pre(args):
    """Main entry point for mhc (pre-transformer) benchmark."""
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


def main_post(args):
    """Main entry point for mhc_post (post-transformer) benchmark."""
    rc = _validate_with_hip_post(args)
    if rc != 0:
        return rc

    if args.print_vgpr:
        print("Retrieving VGPR usage for mhc_post Triton kernels...")

        def fun():
            return run_post_benchmark(args)

        print_vgpr(fun, get_caller_name_no_ext())
        return 0

    run_post_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
