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
  When passed, a Triton-vs-HIP correctness check runs as the first step of
  each `(M, n, C)` row (once per config). Requires `--dtype bf16` and
  `n == 4` for all ops. Mismatch behavior is operation-specific:
    - `--op pre`:  AssertionError aborts the benchmark.
    - `--op post`: WARNING is logged with the max-abs diff and timing
                   continues. (Allows profiling shapes where the HIP
                   kernel is known to have a non-determinism / race issue
                   at large M+H without losing the rest of the matrix.)
    - `--op e2e`:  WARNING is logged; matches `--op post` (the e2e
                   pipeline includes the post kernel). The HIP path
                   chains `aiter.mhc_pre` -> `aiter.mhc_post` on the
                   same shared input that feeds the Triton path.
"""

from __future__ import annotations

import argparse
import enum
import logging
import sys
from itertools import product
from typing import Any

import torch
import triton

from aiter.ops.triton.fusions.mhc import mhc, mhc_post
from aiter.ops.triton.utils.mhc_config_utils import hip_post_dispatch_block
from aiter.test_common import checkAllclose
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.utils.mhc_ref import (
    generate_mhc_inputs,
    mhc_e2e_ref,
)

# Optional HIP imports; --with-hip code paths fail loudly at runtime via
# _validate_with_hip when these are None.
try:  # pragma: no cover
    import aiter as _aiter
    import aiter.jit.utils.chip_info as _aiter_chip_info
except ImportError:  # pragma: no cover
    _aiter = None
    _aiter_chip_info = None

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


class Metric(enum.Enum):
    """Benchmark output metrics; ``value`` matches the perf_report column header."""

    TIME = "time(ms)"
    THROUGHPUT = "throughput(TFLOPS)"
    BANDWIDTH = "bandwidth(GB/s)"
    ARITHMETIC_INTENSITY = "arithmetic_intensity(FLOP/byte)"

    @property
    def column_label(self) -> str:
        return self.value

    def compute(self, ms: float, flops: float | None, mem: int) -> float:
        if self is Metric.TIME:
            return ms
        if self is Metric.THROUGHPUT:
            return 0.0 if flops is None else flops / (ms * 1e-3) / 1e12
        if self is Metric.BANDWIDTH:
            return mem / (ms * 1e-3) / 1e9
        if self is Metric.ARITHMETIC_INTENSITY:
            return 0.0 if flops is None else flops / mem
        raise ValueError(f"Unhandled metric: {self}")


# CLI --metric alias -> Metric.
_METRIC_BY_CLI = {
    "time": Metric.TIME,
    "throughput": Metric.THROUGHPUT,
    "bandwidth": Metric.BANDWIDTH,
    "arithmetic_intensity": Metric.ARITHMETIC_INTENSITY,
}


def _metric_from_provider(provider: str) -> Metric:
    """Recover the ``Metric`` from a ``[backend_]<column_label>`` provider string."""
    for m in Metric:
        if m.column_label in provider:
            return m
    raise ValueError(f"Unrecognized provider string: {provider!r}")


# Color/linestyle pairs for `triton.testing.Benchmark.styles`, sliced by line_vals count.
_PALETTE = [
    ("red", "-"),
    ("blue", "-"),
    ("yellow", "-"),
    ("green", "-"),
    ("red", "--"),
    ("blue", "--"),
    ("yellow", "--"),
    ("green", "--"),
]


def get_benchmark_configs(args):
    """Return [(M, n, C), ...] for the current CLI args."""
    if args.M and args.n and args.C:
        return [(args.M, args.n, args.C)]

    Ms = [2**i for i in range(10, 15)]
    n = 4
    # --with-hip: C=512 is excluded because the greedy post dispatcher selects
    # block=512 and then fails the kernel's hidden_size >= residual_block * 2 check.
    Cs = [1280, 2560, 4096, 7168]
    return sorted(list(product(Ms, [n], Cs)), key=lambda x: (x[2], x[0]))


def _compute_metrics(
    operation: str,
    M: int,
    n: int,
    C: int,
    sinkhorn_iters: int | None = None,
    elem_bytes: int | None = None,
) -> tuple[float, int]:
    """Analytic FLOPs and memory traffic; returns ``(flops, bytes)``.

    For ``operation == "e2e"`` the pre and post stages are accumulated
    in-place. ``elem_bytes`` defaults to 2 (bf16/fp16) when omitted.
    """
    if operation not in ("pre", "post", "e2e"):
        raise ValueError(f"Unsupported operation: {operation!r}")

    elem_size = elem_bytes if elem_bytes is not None else 2
    total_flops = 0.0
    total_memory = 0

    if operation in ("pre", "e2e"):
        nC = n * C
        n_squared = n * n
        N = n_squared + 2 * n

        # Eq 14: matmul for 3 streams
        flops_matmul = 2.0 * M * nC * n + 2.0 * M * nC * n + 2.0 * M * nC * n_squared
        # Eq 15: RMS normalization
        flops_rms = 4.0 * M * nC
        flops_apply_pre = 2.0 * M * n * C
        # Eq 19: Sinkhorn-Knopp
        flops_sinkhorn = 10.0 * M * n_squared * sinkhorn_iters
        total_flops += flops_matmul + flops_rms + flops_apply_pre + flops_sinkhorn

        bias_size = 4  # bias is always fp32 regardless of activation dtype
        total_memory += (
            M * nC * elem_size  # x (GEMM)
            + M * nC * elem_size  # x (apply-pre re-read)
            + nC * n * elem_size  # phi_pre
            + nC * n * elem_size  # phi_post
            + nC * n_squared * elem_size  # phi_res
            + N * bias_size
            + M * n * elem_size  # post_mix write
            + M * n_squared * elem_size  # comb_mix write
            + M * C * elem_size  # layer_input write
        )

    if operation in ("post", "e2e"):
        # out[M, j, c] = post_mix[M, j] * layer_input[M, c]
        #              + sum_h comb_mix[M, h, j] * residual[M, h, c]
        # 2 * n * (n+1) FLOPs per (M, c) under the GEMV "2 FLOPs per MAC"
        # convention.
        mix_bytes = 4  # fp32 mixes
        total_flops += 2.0 * M * n * (n + 1) * C
        total_memory += (
            M * C * elem_size  # read layer_input
            + M * n * C * elem_size  # read residual
            + M * n * mix_bytes  # read post_mix
            + M * n * n * mix_bytes  # read comb_mix
            + M * n * C * elem_size  # write out
        )

    return float(total_flops), int(total_memory)


def _get_benchmark_config(args, operation):
    """Build the dict consumed by ``_create_benchmark_kernel`` via the op handler."""
    handler = _OP_HANDLERS[operation]
    configs = get_benchmark_configs(args)

    x_vals_list = handler.x_vals(configs)

    if args.metric == "all" or args.metric is None:
        metrics = [m.column_label for m in Metric]
    else:
        metrics = [_METRIC_BY_CLI.get(args.metric, Metric.THROUGHPUT).column_label]

    backends = ["triton"] + (["hip"] if args.with_hip else [])
    if args.with_hip:
        line_vals = [f"{b}_{m}" for m in metrics for b in backends]
    else:
        line_vals = list(metrics)

    benchmark_name = handler.benchmark_name(args)
    if args.with_hip:
        benchmark_name += "_triton+hip"

    return {
        "x_names": handler.x_names,
        "x_vals_list": x_vals_list,
        "metrics": metrics,
        "line_vals": line_vals,
        "benchmark_name": benchmark_name,
        "palette": _PALETTE[: len(line_vals)],
    }


class _MhcHandler:
    """Single handler for {pre, post, e2e}: three timing slices of the same
    pre -> post pipeline. ``self.op`` selects the slice."""

    _X_NAMES = {
        "pre": ["M", "n", "C"],
        "post": ["M", "C"],
        "e2e": ["M", "n", "C"],
    }
    POST_N = 4

    def __init__(self, op: str):
        self.op = op
        self.name = op
        self.x_names = self._X_NAMES[op]

    def x_vals(self, configs: list[tuple[int, int, int]]) -> list[tuple]:
        return [(M, C) for M, _n, C in configs] if self.op == "post" else configs

    def benchmark_name(self, args: argparse.Namespace) -> str:
        if self.op == "pre":
            return get_caller_name_no_ext() + f"_sinkhorn-{args.sinkhorn_iters}iters"
        if self.op == "post":
            return f"bench_mhc_post_{args.dtype}"
        return f"mhc-e2e-{args.dtype}"

    def vgpr_msg(self) -> str:
        return f"Retrieving VGPR usage for mhc {self.op} Triton kernels..."

    def setup_call(
        self,
        params: dict[str, int],
        args: argparse.Namespace,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        op = self.op
        M = params["M"]
        n = self.POST_N if op == "post" else params["n"]
        C = params["C"]
        sinkhorn_iters = args.sinkhorn_iters
        if op == "post":
            torch.manual_seed(0)

        x, phi, alpha_pre, alpha_post, alpha_res, bias, _ = generate_mhc_inputs(
            M, n, C, dtype
        )

        flops, mem_bytes = _compute_metrics(
            op,
            M,
            n=n,
            C=C,
            sinkhorn_iters=sinkhorn_iters,
            elem_bytes=x.element_size(),
        )

        def _call_triton_pre():
            return mhc(
                x,
                phi,
                alpha_pre,
                alpha_post,
                alpha_res,
                bias,
                n,
                hc_pre_eps=1e-6,
                sinkhorn_iters=sinkhorn_iters,
            )

        if op == "post":
            h_post, h_res, layer_input = _call_triton_pre()
            residual = x.view(M, n, C)

        if op == "pre":
            triton_fn = _call_triton_pre

        elif op == "post":

            def triton_fn():
                return mhc_post(None, layer_input, residual, h_post, h_res)

        else:  # e2e

            def triton_fn():
                hp, hr, li = _call_triton_pre()
                return mhc_post(None, li, x.view(M, n, C), hp, hr, None)

        hip_fn = None
        check_payload = None
        if args.with_hip:
            residual_b, fn_hip, hc_scale, hc_base = _triton_to_hip_pre_inputs(
                x, phi, alpha_pre, alpha_post, alpha_res, bias, n
            )
            hip_dev = torch.device(residual_b.device)

            # All HIP mhc_pre invocations share these kwargs.
            def _call_aiter_pre():
                return _aiter.mhc_pre(
                    residual_b,
                    fn_hip,
                    hc_scale,
                    hc_base,
                    rms_eps=1e-6,
                    hc_pre_eps=1e-6,
                    hc_sinkhorn_eps=0.0,
                    hc_post_mult_value=2.0,
                    sinkhorn_repeat=sinkhorn_iters,
                )

            if op == "pre":

                def hip_fn():
                    with hip_dev:
                        return _call_aiter_pre()

            elif op == "post":
                with hip_dev:
                    post_mix_hip, comb_mix_hip, layer_input_hip = _call_aiter_pre()
                out_hip = torch.empty(M, n, C, dtype=dtype, device=residual_b.device)

                def hip_fn():
                    with hip_dev:
                        _aiter.mhc_post(
                            out_hip,
                            layer_input_hip,
                            residual_b,
                            post_mix_hip,
                            comb_mix_hip,
                        )
                        return out_hip

                check_payload = (
                    x.view(M, n, C),
                    phi,
                    alpha_pre,
                    alpha_post,
                    alpha_res,
                    bias,
                    n,
                    sinkhorn_iters,
                )
            else:  # e2e
                out_hip = torch.empty(M, n, C, dtype=dtype, device=residual_b.device)

                def hip_fn():
                    with hip_dev:
                        pm, cm, li_h = _call_aiter_pre()
                        _aiter.mhc_post(out_hip, li_h, residual_b, pm, cm)
                        return out_hip

                check_payload = (
                    x.view(M, n, C),
                    phi,
                    alpha_pre,
                    alpha_post,
                    alpha_res,
                    bias,
                    n,
                    sinkhorn_iters,
                )

        return {
            "triton_fn": triton_fn,
            "hip_fn": hip_fn,
            "metrics": (flops, mem_bytes),
            "check_payload": check_payload,
        }

    def correctness_check(self, setup: dict[str, Any], params: dict[str, int]) -> None:
        if setup["hip_fn"] is None:
            return
        op = self.op
        if op == "pre":
            _assert_triton_matches_hip(
                "pre",
                setup["triton_fn"](),
                setup["hip_fn"](),
                M=params["M"],
                n=params["n"],
                C=params["C"],
            )
            return
        xl, phi, ap, apo, ar, bs, n_, si = setup["check_payload"]
        _, x_l_plus_1, _, _ = mhc_e2e_ref(
            xl,
            phi,
            ap,
            apo,
            ar,
            bs,
            n_,
            hc_pre_eps=1e-6,
            sinkhorn_iters=si,
        )
        t_out = setup["triton_fn"]()
        h_out = setup["hip_fn"]()
        t_vs_ref = (t_out.float() - x_l_plus_1).abs().max().item()
        h_vs_ref = (h_out.float() - x_l_plus_1).abs().max().item()
        if op == "post":
            logging.info(
                "mhc_post (M=%d, C=%d): triton-vs-ref max=%.4g  hip-vs-ref max=%.4g",
                params["M"],
                params["C"],
                t_vs_ref,
                h_vs_ref,
            )
            _assert_triton_matches_hip(
                "post", t_out, h_out, M=params["M"], C=params["C"]
            )
        else:  # e2e
            logging.info(
                "mhc_e2e (M=%d, n=%d, C=%d): triton-vs-ref max=%.4g  hip-vs-ref max=%.4g",
                params["M"],
                params["n"],
                params["C"],
                t_vs_ref,
                h_vs_ref,
            )
            _assert_triton_matches_hip(
                "e2e",
                t_out,
                h_out,
                M=params["M"],
                n=params["n"],
                C=params["C"],
            )

    def validate_hip(self, args: argparse.Namespace) -> None:
        if self.op != "post":
            _validate_with_hip_pre(args)
        if self.op != "pre":
            _validate_with_hip_post(args)


_OP_HANDLERS: dict[str, _MhcHandler] = {
    op: _MhcHandler(op) for op in ("pre", "post", "e2e")
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
    """Build a perf_report-decorated bench function for one op."""
    handler = _OP_HANDLERS[operation]
    dtype = arg_to_torch_dtype[args.dtype]

    config = _get_benchmark_config(args, operation)

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

    # Per-shape correctness check runs only on first occurrence of each shape.
    _checked_configs: set[tuple] = set()

    @triton.testing.perf_report([benchmark])
    def bench_mhc_kernel(provider, **benchmark_params):
        setup = handler.setup_call(benchmark_params, args, dtype)
        triton_fn = setup["triton_fn"]
        hip_fn = setup["hip_fn"]
        flops, mem_bytes = setup["metrics"]

        if args.with_hip and hip_fn is not None:
            config_key = tuple(benchmark_params[k] for k in handler.x_names)
            if config_key not in _checked_configs:
                handler.correctness_check(setup, benchmark_params)
                _checked_configs.add(config_key)

        backend = "hip" if provider.startswith("hip_") else "triton"
        fn = hip_fn if backend == "hip" else triton_fn

        ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)

        return _metric_from_provider(provider).compute(ms, flops, mem_bytes)

    return bench_mhc_kernel


def run_benchmark(args, operation):
    bench_fn = _create_benchmark_kernel(args, operation=operation)
    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Benchmark mHC",
        description="Benchmark mHC (manifold-constrained Hyper Connection) kernels",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--op",
        type=str,
        default="pre",
        choices=["pre", "post", "e2e"],
        help="Which operation to benchmark: 'pre' (mhc), 'post' (mhc_post), or 'e2e' (full pipeline)",
    )

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
            "Also benchmark the HIP kernel alongside Triton. Requires "
            "--dtype bf16 and n == 4 for all ops. Runs a Triton-vs-HIP "
            "correctness check once per (M, n, C) before timing that row. "
            "Mismatch policy: --op pre aborts; --op post and --op e2e log "
            "a WARNING and continue. For --op e2e the HIP path chains "
            "aiter.mhc_pre -> aiter.mhc_post on the shared input."
        ),
    )

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
        help=(
            "Write performance results to a CSV file in the current "
            "directory. Filename pattern (per --op):\n"
            "  pre  -> bench_mhc_sinkhorn-<sinkhorn_iters>iters[+_triton+hip].csv\n"
            "  post -> bench_mhc_post_<dtype>[+_triton+hip].csv\n"
            "  e2e  -> mhc-e2e-<dtype>.csv\n"
            "(The '_triton+hip' suffix is appended when --with-hip is set.)"
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help=(
            "Warmup iterations passed to triton.testing.do_bench (default: "
            "25). Increase for sub-50us kernels where the default produces "
            "noisy timings; decrease to speed up sweeps over large shapes."
        ),
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=100,
        help=(
            "Measurement iterations passed to triton.testing.do_bench "
            "(default: 100). The reported time is the median over rep "
            "iterations."
        ),
    )

    args = parser.parse_args()

    if args.op == "post" and args.n is not None and args.n != 4:
        parser.error(
            f"--op post requires n == 4 (got -n {args.n}). "
            "The post bench config pins n=4 to match aiter.mhc_post, which "
            "hardcodes hc_mult == 4 via TORCH_CHECK."
        )

    return args


def _assert_triton_matches_hip(
    op: str,
    triton_out,
    hip_out,
    *,
    M: int,
    C: int,
    n: int | None = None,
) -> None:
    """Triton-vs-HIP parity via ``checkAllclose`` for all ops.

    pre  -> per-tensor (post_mix, comb_mix, layer_input) check with op-specific
            atol/rtol from op_tests/triton_tests/fusions/test_mhc.py;
            fails assert when bad-element ratio > 5%.
    post -> single-tensor check with atol=4e-2, rtol=1e-2; SOFT-WARNS at >5%.
    e2e  -> single-tensor check with atol=4e-2, rtol=1e-2; SOFT-WARNS at >5%.
    """
    cfg = f"(M={M}, C={C})" if n is None else f"(M={M}, n={n}, C={C})"
    if op == "pre":
        post_t, comb_t, li_t = triton_out
        post_h, comb_h, li_h = hip_out
        checks = (
            ("post_mix", post_t, post_h, 4e-2, 1e-2),
            ("comb_mix", comb_t, comb_h, 4e-2, 1e-2),
            ("layer_input", li_t, li_h, 8e-2, 2e-2),
        )
    else:
        checks = ((f"mhc_{op}", triton_out, hip_out, 4e-2, 1e-2),)

    for name, t, h, atol, rtol in checks:
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
        ok = pct <= 0.05
        logging.log(
            logging.INFO if ok else logging.WARNING,
            "%s correctness %s at %s: bad_element_ratio=%.2f%% (atol=%g, rtol=%g)",
            name,
            "OK" if ok else "MISMATCH",
            cfg,
            pct * 100,
            atol,
            rtol,
        )
        if op == "pre":
            assert (
                ok
            ), f"{msg} (atol={atol:g}, rtol={rtol:g}, bad_element_ratio={pct:.2%})"


def _validate_with_hip_pre(args) -> None:
    """Validate --with-hip args for mhc_pre; raises AssertionError on failure."""
    for M_, n_, C_ in get_benchmark_configs(args):
        hc_hidden_size = n_ * C_
        assert hc_hidden_size % 64 == 0, (
            f"--with-hip requires n*C (hc_hidden_size) divisible by 64 "
            f"(got n={n_}, C={C_}, n*C={hc_hidden_size} for M={M_}). "
            f"aiter.mhc_pre_gemm_sqrsum requires hc_hidden_size % tile_k == 0 "
            f"for tile_k in {{64, 128}}."
        )
        assert C_ % 128 == 0 and C_ >= 512, (
            f"--with-hip requires C (hidden_size) divisible by 128 and >= 512 "
            f"(got C={C_} for M={M_}, n={n_}). aiter.mhc_pre_big_fuse dispatches "
            f"with residual_block in {{128, 256}} and enforces "
            f"hidden_size % residual_block == 0 && hidden_size >= "
            f"residual_block * 2 via TORCH_CHECK."
        )


def _validate_with_hip_post(args) -> None:
    """Validate --with-hip args for mhc_post; raises AssertionError on failure."""
    arch_id = _aiter_chip_info.get_gfx()
    for M_, n_, C_ in get_benchmark_configs(args):
        block = hip_post_dispatch_block(C_, arch_id)
        assert block is not None, (
            f"--with-hip requires C (hidden_size) divisible by 256 "
            f"(got C={C_} for M={M_}, n={n_}). aiter.mhc_post dispatches with "
            f"residual_block in {{256, 512, 1024}}."
        )
        assert C_ >= 2 * block, (
            f"--with-hip on {arch_id} selects residual_block={block} for C={C_}; "
            f"requires hidden_size >= {2 * block} (got C={C_} for M={M_}, n={n_})."
        )


def _validate_with_hip(args, operation: str = "pre") -> None:
    """
    Common --with-hip validation (dtype/n); raises AssertionError on failure.
    """
    if not args.with_hip:
        return

    assert args.dtype == "bf16", (
        f"--with-hip only supports --dtype bf16 (got {args.dtype!r}). "
        f"{'aiter.mhc_pre' if operation == 'pre' else 'aiter.mhc_post'} kernel "
        f"is template-instantiated for bf16 residual with fp32 parameters."
    )

    n_kernel = "aiter.mhc_pre_big_fuse" if operation == "pre" else "aiter.mhc_post"
    for M_, n_, C_ in get_benchmark_configs(args):
        assert n_ == 4, (
            f"--with-hip requires n == 4 (got n={n_} for M={M_}, C={C_}). "
            f"{n_kernel} hardcodes hc_mult == 4 via TORCH_CHECK."
        )

    _OP_HANDLERS[operation].validate_hip(args)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s: %(message)s",
        force=True,
    )

    args = parse_args()
    handler = _OP_HANDLERS[args.op]

    _validate_with_hip(args, operation=args.op)

    if args.print_vgpr:
        print(handler.vgpr_msg())
        print_vgpr(lambda: run_benchmark(args, args.op), get_caller_name_no_ext())
        return 0

    run_benchmark(args, args.op)
    return 0


if __name__ == "__main__":
    sys.exit(main())
