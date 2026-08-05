# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and benchmark coverage for gfx942/gfx950 BF16 decode GEMM."""

import argparse
import statistics
from pathlib import Path

import pytest
import torch

pytest.importorskip("flydsl")

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, gpu
from flydsl.expr.typing import T

import aiter.ops.flydsl.kernels.gemm_decode_common as gemm_decode_common
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.gemm_decode import (
    ContractionMode,
    GemmDecodeConfig,
    OutputRounding,
    ReductionMode,
    _convert_bf16,
    _launch_gemm_decode_bf16,
    gemm_decode_bf16,
    gemm_decode_bf16_configured,
    pack_bf16x2,
    select_gemm_decode_config,
    unpack_bf16x2_f32,
)

pytestmark = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"),
    reason="gemm_decode_bf16 requires gfx942 or gfx950",
)

ATOL = 0.125
RTOL = 0.01
BENCHMARK_PROVIDERS = (
    "flydsl",
    "aiter_wvsplitk",
    "vllm_wvsplitk",
    "wvsplitk_small",
    "hipblaslt",
)
TP_RELEVANT_SHAPES = (
    (16384, 7168),
    (4096, 7168),
    (2048, 7168),
    (7168, 4096),
    (7168, 2048),
    (7168, 1792),
    (7168, 896),
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


def test_odd_k_uses_flydsl_tail(monkeypatch):
    M, N, K = 3, 63, 127
    torch.manual_seed(2)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    ref = (A.float() @ B.float().T).bfloat16()
    C = torch.full((M, N), torch.nan, dtype=torch.bfloat16, device="cuda")

    def reject_library_fallback(*args, **kwargs):
        raise AssertionError("odd K must use the FlyDSL tail path")

    monkeypatch.setattr(torch, "mm", reject_library_fallback)
    gemm_decode_bf16(A, B, C, M, N, K, stream=fx.Stream(None))
    torch.cuda.synchronize()
    assert torch.isfinite(C).all(), "output sentinel was not fully overwritten"
    torch.testing.assert_close(C, ref, atol=ATOL, rtol=RTOL)


def test_gemm_decode_rejects_unsupported_m():
    A = torch.zeros((5, 8), dtype=torch.bfloat16, device="cuda")
    B = torch.zeros((2, 8), dtype=torch.bfloat16, device="cuda")
    C = torch.zeros((5, 2), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match=r"M in \[1, 4\]"):
        gemm_decode_bf16(A, B, C, 5, 2, 8, stream=fx.Stream(None))


def test_gemm_decode_validates_tensor_contract(monkeypatch):
    M, N, K = 2, 8, 8
    A = torch.zeros((M, K), dtype=torch.bfloat16, device="cuda")
    B = torch.zeros((N, K), dtype=torch.bfloat16, device="cuda")
    C = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(TypeError, match="A must be a torch.Tensor"):
        gemm_decode_bf16(object(), B, C, M, N, K)
    with pytest.raises(ValueError, match="A must be rank 2"):
        gemm_decode_bf16(A.flatten(), B, C, M, N, K)
    with pytest.raises(ValueError, match="A must have dtype"):
        gemm_decode_bf16(A.float(), B, C, M, N, K)
    with pytest.raises(ValueError, match="A must be on"):
        gemm_decode_bf16(A.cpu(), B, C, M, N, K)
    with pytest.raises(ValueError, match="B must have shape"):
        gemm_decode_bf16(A, B[:-1], C, M, N, K)
    with pytest.raises(ValueError, match="packed row-major"):
        gemm_decode_bf16(A, B.T, C, M, N, K)
    with pytest.raises(ValueError, match="C must not overlap"):
        gemm_decode_bf16(A, B, A, M, N, K)

    monkeypatch.setattr(gemm_decode_common, "get_rocm_arch", lambda: "gfx90a")
    with pytest.raises(ValueError, match="requires gfx942 or gfx950"):
        gemm_decode_bf16(A, B, C, M, N, K)


@pytest.mark.parametrize(
    "config",
    [
        GemmDecodeConfig(kvec=2, m_per_wave=1, n_per_wave=1),
        GemmDecodeConfig(kvec=4, m_per_wave=2, n_per_wave=2),
        GemmDecodeConfig(kvec=8, m_per_wave=4, n_per_wave=4),
        GemmDecodeConfig(
            kvec=8,
            m_per_wave=4,
            n_per_wave=2,
            waves_per_eu=1,
            b_cache_modifier=0,
            reduction=ReductionMode.BPERMUTE_REFERENCE,
        ),
        GemmDecodeConfig(
            kvec=8,
            m_per_wave=1,
            n_per_wave=1,
            contraction=ContractionMode.PACKED_F32,
        ),
    ],
)
def test_gemm_decode_compile_time_axes(config):
    M, N, K = 4, 64, 128
    torch.manual_seed(1)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.full((M, N), torch.nan, dtype=torch.bfloat16, device="cuda")
    gemm_decode_bf16_configured(
        A,
        B,
        C,
        M,
        N,
        K,
        config,
        stream=fx.Stream(None),
    )
    torch.cuda.synchronize()
    ref = (A.float() @ B.float().T).bfloat16()
    assert torch.isfinite(C).all()
    torch.testing.assert_close(C, ref, atol=ATOL, rtol=RTOL)
    if config.contraction == ContractionMode.PACKED_F32:
        source_ir = _launch_gemm_decode_bf16._last_compiled[1].source_ir
        assert "v_pk_fma_f32" in source_ir


def test_gemm_decode_config_selection():
    if get_gfx() == "gfx942":
        assert select_gemm_decode_config(1, 7168, 896) == GemmDecodeConfig(
            kvec=8,
            m_per_wave=1,
            n_per_wave=1,
            waves_per_eu=4,
            contraction=ContractionMode.SCALAR_F32,
        )
        assert select_gemm_decode_config(4, 16384, 7168) == GemmDecodeConfig(
            kvec=8,
            m_per_wave=4,
            n_per_wave=2,
            waves_per_eu=4,
            contraction=ContractionMode.PACKED_F32,
        )
        assert select_gemm_decode_config(4, 65, 128).n_per_wave == 1
        assert (
            select_gemm_decode_config(1, 16384, 4096).b_cache_modifier
            == 0x2
        )
        assert (
            select_gemm_decode_config(4, 16384, 7168).b_cache_modifier
            == 0
        )
        return
    assert select_gemm_decode_config(1, 7168, 896) == GemmDecodeConfig(
        kvec=2,
        m_per_wave=1,
        n_per_wave=4,
    )
    assert select_gemm_decode_config(4, 16384, 7168) == GemmDecodeConfig(
        kvec=8,
        m_per_wave=4,
        n_per_wave=4,
        reduction=ReductionMode.BPERMUTE_REFERENCE,
    )
    assert select_gemm_decode_config(4, 65, 128).n_per_wave == 1


def test_gemm_decode_rejects_unsupported_cache_policy():
    with pytest.raises(ValueError, match="unsupported cache policy"):
        GemmDecodeConfig(b_cache_modifier=0x2000).validate()


def test_gfx942_rejects_bf16_dot2():
    if get_gfx() != "gfx942":
        pytest.skip("gfx942-specific instruction guard")
    M, N, K = 1, 64, 128
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    config = GemmDecodeConfig(
        kvec=2,
        m_per_wave=1,
        n_per_wave=1,
        contraction=ContractionMode.DOT2_BF16,
    )
    with pytest.raises(ValueError, match="dot2_bf16 contraction requires gfx950"):
        gemm_decode_bf16_configured(A, B, C, M, N, K, config)


def test_bf16_pair_expansion_is_bit_exact():
    @flyc.kernel
    def expand_kernel(
        src: fx.Tensor,
        dst: fx.Tensor,
        pairs: fx.Constexpr[int],
    ):
        pair = gpu.thread_idx.x
        if pair < fx.Int32(pairs):
            src_rsrc = buffer_ops.create_buffer_resource(src)
            dst_rsrc = buffer_ops.create_buffer_resource(dst)
            lo = buffer_ops.buffer_load(
                src_rsrc,
                pair * fx.Int32(2),
                vec_width=1,
                dtype=T.bf16,
            )
            hi = buffer_ops.buffer_load(
                src_rsrc,
                pair * fx.Int32(2) + fx.Int32(1),
                vec_width=1,
                dtype=T.bf16,
            )
            expanded = unpack_bf16x2_f32(pack_bf16x2(lo, hi))
            buffer_ops.buffer_store(
                expanded[0],
                dst_rsrc,
                pair * fx.Int32(2),
            )
            buffer_ops.buffer_store(
                expanded[1],
                dst_rsrc,
                pair * fx.Int32(2) + fx.Int32(1),
            )

    @flyc.jit
    def expand(
        src: fx.Tensor,
        dst: fx.Tensor,
        pairs: fx.Constexpr[int],
    ):
        expand_kernel(src, dst, pairs).launch(
            grid=(1, 1, 1),
            block=(64, 1, 1),
        )

    bits = torch.tensor(
        [0x0000, 0x8000, 0x0001, 0x8001, 0x3F80, 0x7F80, 0xFF80, 0x7FC1],
        dtype=torch.uint16,
    )
    src = bits.view(torch.bfloat16).cuda()
    dst = torch.empty(bits.numel(), dtype=torch.float32, device="cuda")
    expand(src, dst, bits.numel() // 2)
    torch.cuda.synchronize()
    expected_bits = (bits.to(torch.int32) << 16).view(torch.float32)
    assert torch.equal(dst.cpu().view(torch.int32), expected_bits.view(torch.int32))


def test_waves_per_eu_is_attached_to_direct_gpu_func():
    M, N, K = 2, 64, 128
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    config = GemmDecodeConfig(
        kvec=2,
        m_per_wave=2,
        n_per_wave=2,
        waves_per_eu=1,
    )
    gemm_decode_bf16_configured(A, B, C, M, N, K, config)
    torch.cuda.synchronize()

    source_ir = _launch_gemm_decode_bf16._last_compiled[1].source_ir
    assert source_ir.count("rocdl.waves_per_eu") == 1
    attr_offset = source_ir.index("rocdl.waves_per_eu")
    assert source_ir.rfind("gpu.func", 0, attr_offset) > source_ir.rfind(
        "gpu.launch_func",
        0,
        attr_offset,
    )
    if get_gfx() == "gfx942":
        assert "llvm.intr.fma" in source_ir
        assert "v_dot2_f32_bf16" not in source_ir


def test_gemm_decode_graph_replay_on_non_default_stream():
    M, N, K = 3, 63, 127
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.full((M, N), torch.nan, dtype=torch.bfloat16, device="cuda")
    ref = (A.float() @ B.float().T).bfloat16()
    current = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(current)

    gemm_decode_bf16(A, B, C, M, N, K, stream=fx.Stream(side))
    side.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        gemm_decode_bf16(A, B, C, M, N, K, stream=fx.Stream(side))
    side.synchronize()

    C.fill_(torch.nan)
    side.wait_stream(current)
    with torch.cuda.stream(side):
        graph.replay()
    current.wait_stream(side)
    current.synchronize()
    assert torch.isfinite(C).all(), "graph replay did not overwrite the output"
    torch.testing.assert_close(C, ref, atol=ATOL, rtol=RTOL)


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
    if get_gfx() != "gfx950":
        pytest.skip("stochastic BF16 conversion requires gfx950")
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


def _load_vllm_wvsplitk(library):
    if not hasattr(torch.ops, "_rocm_C") or not hasattr(torch.ops._rocm_C, "wvSplitK"):
        if library is None:
            raise ValueError(
                "vllm_wvsplitk requires --vllm-rocm-library when _rocm_C "
                "is not already loaded"
            )
        library = Path(library)
        if not library.is_file():
            raise ValueError(f"vLLM ROCm library does not exist: {library}")
        torch.ops.load_library(str(library))
    return torch.ops._rocm_C.wvSplitK


def _prepare_provider(provider, A, B, C, M, N, K, vllm_rocm_library=None):
    if provider == "flydsl":

        def run_flydsl():
            gemm_decode_bf16(A, B, C, M, N, K, stream=fx.Stream(None))
            return C

        return run_flydsl, ""

    if provider in {"aiter_wvsplitk", "wvsplitk_small"}:
        if K % 8 != 0:
            return None, "requires K divisible by 8"
        if provider == "aiter_wvsplitk" and M > 4:
            return None, "supports M in [1, 4]"
        from aiter.ops.custom import wvSpltK, wv_splitk_small_fp16_bf16

        cu_count = torch.cuda.get_device_properties(0).multi_processor_count
        op = wvSpltK if provider == "aiter_wvsplitk" else wv_splitk_small_fp16_bf16

        def run_aiter():
            op(B, A, C, M, cu_count)
            return C

        return run_aiter, ""

    if provider == "vllm_wvsplitk":
        if K % 8 != 0:
            return None, "requires K divisible by 8"
        if M > 5:
            return None, "supports M in [1, 5]"
        try:
            op = _load_vllm_wvsplitk(vllm_rocm_library)
        except ValueError as error:
            return None, str(error)
        cu_count = torch.cuda.get_device_properties(0).multi_processor_count
        return lambda: op(B, A, None, cu_count), ""

    if provider == "hipblaslt":
        torch.backends.cuda.preferred_blas_library("hipblaslt")
        B_t = B.T
        return lambda: torch.mm(A, B_t, out=C), ""

    raise ValueError(f"unknown benchmark provider: {provider}")


def _benchmark(
    M,
    N,
    K,
    rounding,
    providers,
    warmup,
    repeat,
    vllm_rocm_library=None,
):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    ref = (A.float() @ B.float().T).bfloat16()
    bytes_transferred = (M * K + N * K + M * N) * 2
    results = {}
    for provider in providers:
        C = torch.full((M, N), torch.nan, dtype=torch.bfloat16, device="cuda")
        run, skip_reason = _prepare_provider(
            provider,
            A,
            B,
            C,
            M,
            N,
            K,
            vllm_rocm_library,
        )
        if run is None:
            print(f"M={M} N={N} K={K} provider={provider}: SKIP {skip_reason}")
            continue

        # Compile/JIT and validate outside the timed region.
        out = run()
        torch.cuda.synchronize()
        assert torch.isfinite(out).all(), f"{provider} did not fully write its output"
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

        median, minimum, maximum = _measure_us(run, warmup, repeat)
        metrics = _error_metrics(out, ref)
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

    baseline = results.get("aiter_wvsplitk")
    for provider in providers:
        if provider not in results:
            continue
        result = results[provider]
        relative = ""
        if baseline is not None:
            relative = (
                f" speedup_vs_aiter_wvsplitk="
                f"{baseline['median'] / result['median']:.3f}x"
            )
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
        "--shape",
        type=int,
        nargs=2,
        action="append",
        metavar=("N", "K"),
        help="benchmark an (N, K) pair; repeat for multiple shapes",
    )
    parser.add_argument(
        "--tp-shapes",
        action="store_true",
        help="benchmark representative unsharded, column-TP, and row-TP shapes",
    )
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
    parser.add_argument(
        "--vllm-rocm-library",
        help="path to vLLM's built _rocm_C shared library",
    )
    args = parser.parse_args()
    if args.tp_shapes and args.shape:
        parser.error("--tp-shapes cannot be combined with --shape")
    shapes = (
        TP_RELEVANT_SHAPES
        if args.tp_shapes
        else (args.shape or [(args.N, args.K)])
    )
    for rounding_name in args.rounding:
        rounding = OutputRounding(rounding_name)
        for N, K in shapes:
            for M in args.M:
                _benchmark(
                    M,
                    N,
                    K,
                    rounding,
                    args.providers,
                    args.warmup,
                    args.repeat,
                    args.vllm_rocm_library,
                )


if __name__ == "__main__":
    main()
