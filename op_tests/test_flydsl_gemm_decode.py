# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and benchmark coverage for the gfx950 BF16 decode GEMM."""

import argparse
import statistics

import pytest
import torch

pytest.importorskip("flydsl")

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, gpu
from flydsl.expr.typing import T

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.gemm_decode import (
    OutputRounding,
    _convert_bf16,
    gemm_decode_bf16,
)

pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx950",
    reason="gemm_decode_bf16 requires gfx950",
)

ATOL = 0.125
RTOL = 0.01
BENCHMARK_PROVIDERS = (
    "flydsl",
    "wvsplitk",
    "wvsplitk_small",
    "hipblaslt",
)

CORRECTNESS_CASES = [
    (1, 1, 1),
    (2, 3, 7),
    (3, 63, 127),
    (4, 65, 128),
    (1, 64, 511),
    (2, 65, 512),
    (3, 63, 513),
    (4, 65, 768),
    (1, 33, 4224),
    (4, 17, 8448),
    (2, 128, 7168),
    (4, 6288, 7168),
]

def _launcher(rounding):
    if rounding != OutputRounding.RNE:
        raise ValueError("the production GEMM launcher is fixed to RNE")
    return gemm_decode_bf16


def _run_case(M, N, K, rounding=OutputRounding.RNE, seed=0):
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.full((M, N), torch.nan, dtype=torch.bfloat16, device="cuda")
    _launcher(rounding)(A, B, C, M, N, K, stream=fx.Stream(None))
    ref = (A.float() @ B.float().T).bfloat16()
    torch.cuda.synchronize()
    return C, ref


@pytest.mark.parametrize(("M", "N", "K"), CORRECTNESS_CASES)
def test_gemm_decode_bf16(M, N, K):
    out, ref = _run_case(M, N, K)
    assert torch.isfinite(out).all(), "output sentinel was not fully overwritten"
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


def test_gemm_decode_rejects_unsupported_m():
    A = torch.zeros((5, 8), dtype=torch.bfloat16, device="cuda")
    B = torch.zeros((2, 8), dtype=torch.bfloat16, device="cuda")
    C = torch.zeros((5, 2), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match=r"M in \[1, 4\]"):
        gemm_decode_bf16(A, B, C, 5, 2, 8, stream=fx.Stream(None))


def _make_conversion_launcher(rounding):
    @flyc.kernel
    def convert_kernel(
        src: fx.Tensor,
        dst: fx.Tensor,
        N: fx.Constexpr[int],
    ):
        idx = gpu.block_idx.x * fx.Int32(64) + gpu.thread_idx.x
        if idx < fx.Int32(N):
            src_rsrc = buffer_ops.create_buffer_resource(src)
            dst_rsrc = buffer_ops.create_buffer_resource(dst)
            value = buffer_ops.buffer_load(src_rsrc, idx, vec_width=1, dtype=T.f32)
            converted = _convert_bf16(value, idx, rounding)
            buffer_ops.buffer_store(converted, dst_rsrc, idx)

    @flyc.jit
    def convert(
        src: fx.Tensor,
        dst: fx.Tensor,
        N: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        convert_kernel(src, dst, N).launch(
            grid=((N + 63) // 64, 1, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    return convert


def _convert_values(values, rounding):
    src = values.to(device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src, dtype=torch.bfloat16)
    _make_conversion_launcher(rounding)(src, dst, src.numel(), stream=fx.Stream(None))
    torch.cuda.synchronize()
    return dst.cpu()


def _rtz_reference(values):
    bits = values.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    upper = bits >> 16
    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
    upper = torch.where(is_nan, upper | 0x40, upper)
    return upper.to(torch.uint16).view(torch.bfloat16)


def test_rne_conversion_halfway_and_special_values():
    values = torch.tensor(
        [
            1.0,
            1.00390625,
            1.01171875,
            -1.00390625,
            0.0,
            -0.0,
            2.0**-133,
            float("inf"),
            float("-inf"),
            float("nan"),
        ],
        dtype=torch.float32,
    )
    actual = _convert_values(values, OutputRounding.RNE)
    torch.testing.assert_close(actual, values.bfloat16(), rtol=0.0, atol=0.0, equal_nan=True)


def test_rtz_conversion_special_values():
    values = torch.tensor(
        [1.00390625, -1.00390625, 2.0**-133, float("inf"), float("nan")],
        dtype=torch.float32,
    )
    actual = _convert_values(values, OutputRounding.RTZ)
    torch.testing.assert_close(actual, _rtz_reference(values), rtol=0.0, atol=0.0, equal_nan=True)


def test_truncation_conversion_finite_values():
    values = torch.tensor(
        [1.00390625, -1.00390625, 2.0**-133, 0.0, float("inf")],
        dtype=torch.float32,
    )
    actual = _convert_values(values, OutputRounding.TRUNCATE)
    torch.testing.assert_close(actual, _rtz_reference(values), rtol=0.0, atol=0.0)


def test_stochastic_conversion_is_repeatable_and_bounded():
    midpoint = 1.00390625
    values = torch.full((64,), midpoint, dtype=torch.float32)
    first = _convert_values(values, OutputRounding.STOCHASTIC)
    second = _convert_values(values, OutputRounding.STOCHASTIC)
    assert torch.equal(first, second)
    allowed = {1.0, 1.0078125}
    assert set(first.float().tolist()) <= allowed


def _ordered_bf16_bits(values):
    bits = values.view(torch.uint16).to(torch.int32)
    sign = (bits & 0x8000) != 0
    return torch.where(sign, 0x8000 - (bits & 0x7FFF), 0x8000 + bits)


def _error_metrics(out, ref):
    out_f32 = out.float()
    ref_f32 = ref.float()
    abs_error = (out_f32 - ref_f32).abs()
    nonzero = ref_f32 != 0
    relative = torch.where(nonzero, abs_error / ref_f32.abs(), torch.zeros_like(abs_error))
    ulp = (_ordered_bf16_bits(out) - _ordered_bf16_bits(ref)).abs()
    return {
        "max_abs": abs_error.max().item(),
        "max_rel": relative.max().item(),
        "mismatch": (out != ref).float().mean().item(),
        "max_ulp": ulp.max().item(),
    }


def _measure_us(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeat)
    return statistics.median(samples), min(samples), max(samples)


def _prepare_provider(provider, A, B, C, M, N, K):
    if provider == "flydsl":
        return (
            lambda: gemm_decode_bf16(A, B, C, M, N, K, stream=fx.Stream(None)),
            "",
        )

    if provider in {"wvsplitk", "wvsplitk_small"}:
        if K % 8 != 0:
            return None, "requires K divisible by 8"
        if provider == "wvsplitk" and M > 4:
            return None, "supports M in [1, 4]"
        from aiter.ops.custom import wvSpltK, wv_splitk_small_fp16_bf16

        cu_count = torch.cuda.get_device_properties(0).multi_processor_count
        op = wvSpltK if provider == "wvsplitk" else wv_splitk_small_fp16_bf16
        return lambda: op(B, A, C, M, cu_count), ""

    if provider == "hipblaslt":
        torch.backends.cuda.preferred_blas_library("hipblaslt")
        B_t = B.T
        return lambda: torch.mm(A, B_t, out=C), ""

    raise ValueError(f"unknown benchmark provider: {provider}")


def _benchmark(M, N, K, rounding, providers, warmup, repeat):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    ref = (A.float() @ B.float().T).bfloat16()
    bytes_transferred = (M * K + N * K + M * N) * 2
    results = {}
    for provider in providers:
        C = torch.full((M, N), torch.nan, dtype=torch.bfloat16, device="cuda")
        run, skip_reason = _prepare_provider(provider, A, B, C, M, N, K)
        if run is None:
            print(f"M={M} N={N} K={K} provider={provider}: SKIP {skip_reason}")
            continue

        # Compile/JIT and validate outside the timed region.
        run()
        torch.cuda.synchronize()
        assert torch.isfinite(C).all(), f"{provider} did not fully write its output"
        torch.testing.assert_close(C, ref, atol=ATOL, rtol=RTOL)

        median, minimum, maximum = _measure_us(run, warmup, repeat)
        metrics = _error_metrics(C, ref)
        bandwidth = bytes_transferred / (median * 1e-6) / 1e9
        tflops = 2 * M * N * K / (median * 1e-6) / 1e12
        results[provider] = {
            "median": median,
            "minimum": minimum,
            "maximum": maximum,
            "bandwidth": bandwidth,
            "tflops": tflops,
            **metrics,
        }

    baseline = results.get("wvsplitk")
    for provider in providers:
        if provider not in results:
            continue
        result = results[provider]
        relative = ""
        if baseline is not None:
            relative = f" speedup_vs_wvsplitk={baseline['median'] / result['median']:.3f}x"
        print(
            f"M={M} N={N} K={K} provider={provider} rounding={rounding.value}: "
            f"median={result['median']:.2f} us "
            f"range=[{result['minimum']:.2f}, {result['maximum']:.2f}] us "
            f"bandwidth={result['bandwidth']:.0f} GB/s "
            f"throughput={result['tflops']:.2f} TFLOP/s{relative} "
            f"max_abs={result['max_abs']:.6g} max_rel={result['max_rel']:.6g} "
            f"mismatch={result['mismatch']:.3%} max_ulp={result['max_ulp']}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-M", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("-N", type=int, default=16384)
    parser.add_argument("-K", type=int, default=7168)
    parser.add_argument(
        "--rounding",
        choices=[OutputRounding.RNE.value],
        nargs="+",
        default=[OutputRounding.RNE.value],
    )
    parser.add_argument(
        "--providers",
        choices=BENCHMARK_PROVIDERS,
        nargs="+",
        default=list(BENCHMARK_PROVIDERS),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    for rounding_name in args.rounding:
        rounding = OutputRounding(rounding_name)
        for M in args.M:
            _benchmark(
                M,
                args.N,
                args.K,
                rounding,
                args.providers,
                args.warmup,
                args.repeat,
            )


if __name__ == "__main__":
    main()
