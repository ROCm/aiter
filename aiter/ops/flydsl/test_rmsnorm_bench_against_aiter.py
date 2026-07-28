# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests and benchmark for FlyDSL RMSNorm vs AITER RMSNorm.

Tests:
  - FlyDSL RMSNorm correctness against PyTorch reference
  - AITER RMSNorm correctness against PyTorch reference
  - AITER vs FlyDSL forward benchmark
  - Multiple shapes and dtypes
  - Includes aligned and unaligned hidden sizes

Usage:
    python op_tests/test_flydsl_rmsnorm.py
    python op_tests/test_flydsl_rmsnorm.py --shapes 64,256,f32 128,1024,f16
    python op_tests/test_flydsl_rmsnorm.py --eps 1e-5
    python op_tests/test_flydsl_rmsnorm.py --warmup 20 --iters 100
"""

import argparse
import sys

import torch

torch.set_default_device("cuda")

EPS_DEFAULT = 1e-5


def _parse_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "f32":
        return torch.float32
    if dtype_str == "f16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype string: {dtype_str}")


def _default_tol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-4, 1e-4
    if dtype == torch.float16:
        return 1e-2, 1e-2
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    raise ValueError(f"unsupported dtype: {dtype}")


def _torch_reference_rmsnorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # Compute in FP32 for stable reference, matching the FlyDSL test style.
    x_fp32 = x.float()
    gamma_fp32 = gamma.float()
    sq_mean = (x_fp32 * x_fp32).mean(dim=1, keepdim=True)
    rms = torch.sqrt(sq_mean + eps)
    out = (x_fp32 / rms) * gamma_fp32
    return out


def _generate_data(
    M: int,
    N: int,
    dtype: torch.dtype,
    eps: float,
):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    x = torch.randn((M, N), device="cuda", dtype=dtype)
    gamma = torch.rand((N,), device="cuda", dtype=dtype)
    ref = _torch_reference_rmsnorm(x, gamma, eps)
    return x, gamma, ref


def _check_result(
    ref_out: torch.Tensor,
    test_out: torch.Tensor,
    label: str,
    atol: float,
    rtol: float,
    pass_pct: float = 100.0,
):
    ref_f = ref_out.float()
    test_f = test_out.float()

    max_delta = (ref_f - test_f).abs().max().item()
    close_mask = torch.isclose(ref_f, test_f, atol=atol, rtol=rtol)
    pct_close = close_mask.float().mean().item() * 100.0
    passed = close_mask.all().item()

    print(
        f"  max_delta={max_delta:.6f}, {pct_close:.2f}% close (atol={atol}, rtol={rtol}) "
    )
    print(f"  ref  sample:  {ref_f.reshape(-1)[:8]}")
    print(f"  test sample: {test_f.reshape(-1)[:8]}")
    print(f"  --> {'PASS' if passed else 'FAIL'}")

    return passed, max_delta, pct_close


def _bench_gpu_us(fn, warmup=20, iters=500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) * 1000.0 / iters  # ms -> us


def test_flydsl_rmsnorm_case(
    M: int = 64,
    N: int = 256,
    dtype: torch.dtype = torch.float32,
    eps: float = EPS_DEFAULT,
    atol: float | None = None,
    rtol: float | None = None,
):
    from aiter.ops.flydsl.rmsnorm import flydsl_rmsnorm

    if atol is None or rtol is None:
        atol, rtol = _default_tol(dtype)

    print(f"\n{'=' * 70}")
    print(f"[TEST] FlyDSL RMSNorm: M={M}, N={N}, dtype={dtype}, eps={eps}")
    print(f"{'=' * 70}")

    x, gamma, ref = _generate_data(M=M, N=N, dtype=dtype, eps=eps)

    out = flydsl_rmsnorm(x, gamma, eps)
    torch.cuda.synchronize()

    return _check_result(
        ref_out=ref,
        test_out=out,
        label=f"flydsl_rmsnorm_M{M}_N{N}_{dtype}",
        atol=atol,
        rtol=rtol,
        pass_pct=100.0,
    )


def test_aiter_rmsnorm_case(
    M: int = 64,
    N: int = 256,
    dtype: torch.dtype = torch.float32,
    eps: float = EPS_DEFAULT,
    atol: float | None = None,
    rtol: float | None = None,
):
    from aiter import rmsnorm2d_fwd

    if atol is None or rtol is None:
        atol, rtol = _default_tol(dtype)

    print(f"\n{'=' * 70}")
    print(f"[TEST] AITER RMSNorm: M={M}, N={N}, dtype={dtype}, eps={eps}")
    print(f"{'=' * 70}")

    x, gamma, ref = _generate_data(M=M, N=N, dtype=dtype, eps=eps)

    out = rmsnorm2d_fwd(x, gamma, eps)
    torch.cuda.synchronize()

    return _check_result(
        ref_out=ref,
        test_out=out,
        label=f"aiter_rmsnorm_M{M}_N{N}_{dtype}",
        atol=atol,
        rtol=rtol,
        pass_pct=100.0,
    )


def bench_aiter_vs_flydsl(
    M: int,
    N: int,
    dtype: torch.dtype,
    eps: float,
    warmup: int = 20,
    iters: int = 100,
):
    from aiter import rmsnorm2d_fwd
    from aiter.ops.flydsl.rmsnorm import flydsl_rmsnorm

    print(f"\n{'=' * 70}")
    print(f"[BENCH] RMSNorm AITER vs FlyDSL: M={M}, N={N}, dtype={dtype}, eps={eps}")
    print(f"{'=' * 70}")

    x, gamma, _ = _generate_data(M=M, N=N, dtype=dtype, eps=eps)

    def run_aiter():
        rmsnorm2d_fwd(x, gamma, eps)

    def run_flydsl():
        flydsl_rmsnorm(x, gamma, eps)

    with torch.inference_mode():
        aiter_us = _bench_gpu_us(run_aiter, warmup=warmup, iters=iters)
        flydsl_us = _bench_gpu_us(run_flydsl, warmup=warmup, iters=iters)
    torch.cuda.synchronize()
    speedup = aiter_us / flydsl_us if flydsl_us > 0 else float("inf")

    print(f"  AITER  : {aiter_us:.2f} us")
    print(f"  FlyDSL : {flydsl_us:.2f} us")
    print(f"  Speedup(FlyDSL vs AITER): {speedup:.3f}x")

    return aiter_us, flydsl_us, speedup


def _parse_shape_specs(shape_specs: list[str]):
    parsed = []
    for spec in shape_specs:
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"invalid shape spec '{spec}', expected format M,N,dtype "
                f"(example: 64,256,f32)"
            )
        M = int(parts[0])
        N = int(parts[1])
        dtype = _parse_dtype(parts[2])
        parsed.append((M, N, dtype))
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="FlyDSL RMSNorm unit tests and AITER benchmark",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--shapes",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Shapes to test in format M,N,dtype.\n"
            "Examples:\n"
            "  --shapes 64,256,f32 128,1024,f16 16,512,bf16"
        ),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=EPS_DEFAULT,
        help=f"RMSNorm epsilon (default: {EPS_DEFAULT})",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Override atol for all tests",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Override rtol for all tests",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Benchmark warmup iterations (default: 20)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Benchmark timed iterations (default: 100)",
    )
    args = parser.parse_args()

    from aiter.ops.flydsl.utils import is_flydsl_available

    if not is_flydsl_available():
        print("[SKIP] FlyDSL is not available. Install flydsl package first.")
        sys.exit(0)

    if args.shapes is not None:
        configs = _parse_shape_specs(args.shapes)
    else:
        configs = [
            # GPT OSS Input Dimensions
            (3000, 2880, torch.bfloat16),
            (4000, 2880, torch.bfloat16),
            (5000, 2880, torch.bfloat16),
            (7000, 2880, torch.bfloat16),
            (3072, 2880, torch.bfloat16),
            (4096, 2880, torch.bfloat16),
            (7168, 2880, torch.bfloat16),
            (8192, 2880, torch.bfloat16),
            # --- Stress / max throughput ---
            (16384, 2880, torch.bfloat16),
        ]

    results = []

    for M, N, dtype in configs:
        try:
            atol = args.atol
            rtol = args.rtol
            if atol is None or rtol is None:
                def_atol, def_rtol = _default_tol(dtype)
                atol = def_atol if atol is None else atol
                rtol = def_rtol if rtol is None else rtol

            # Benchmark first so both result rows can include the same perf numbers
            aiter_us, flydsl_us, speedup = bench_aiter_vs_flydsl(
                M=M,
                N=N,
                dtype=dtype,
                eps=args.eps,
                warmup=args.warmup,
                iters=args.iters,
            )

            # FlyDSL correctness
            passed, max_delta, pct = test_flydsl_rmsnorm_case(
                M=M,
                N=N,
                dtype=dtype,
                eps=args.eps,
                atol=atol,
                rtol=rtol,
            )
            results.append(
                (
                    f"flydsl_rmsnorm_M{M}_N{N}_{dtype}",
                    "PASS" if passed else "FAIL",
                    max_delta,
                    pct,
                    aiter_us,
                    flydsl_us,
                    speedup,
                )
            )

            # AITER correctness
            passed, max_delta, pct = test_aiter_rmsnorm_case(
                M=M,
                N=N,
                dtype=dtype,
                eps=args.eps,
                atol=atol,
                rtol=rtol,
            )
            results.append(
                (
                    f"aiter_rmsnorm_M{M}_N{N}_{dtype}",
                    "PASS" if passed else "FAIL",
                    max_delta,
                    pct,
                    aiter_us,
                    flydsl_us,
                    speedup,
                )
            )

        except Exception:
            import traceback

            traceback.print_exc()
            results.append(
                (f"rmsnorm_M{M}_N{N}_{dtype}", "ERROR", 0.0, 0.0, 0.0, 0.0, 0.0)
            )

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for name, status, delta, pct, aiter_us, flydsl_us, speedup in results:
        print(
            f"  {status:>5s}  {name:<40s}  "
            f"max_delta={delta:>10.6f}  close={pct:>6.2f}%  "
            f"AITER={aiter_us:>8.2f}us  "
            f"FlyDSL={flydsl_us:>8.2f}us  "
            f"speedup={speedup:>6.3f}x"
        )

    n_pass = sum(1 for _, s, _, _, _, _, _ in results if s == "PASS")
    print(f"\n  {n_pass}/{len(results)} passed")

    if any(s in ("FAIL", "ERROR") for _, s, _, _, _, _, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
