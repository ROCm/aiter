# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Manual gfx11 FlyDSL A8W8 correctness and timing check.

Run with the active ROCm Python after building/installing the FlyDSL runtime:
    python op_tests/test_gemm_a8w8_flydsl_runtime.py
"""

import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch


def _measure_ms(fn, repeats=50):
    fn()
    torch.cuda.synchronize()
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def _make_inputs(m, n, k, *, padded=False):
    device = torch.device("cuda")
    lda = k + 16 if padded else k
    ldb = k + 16 if padded else k
    x = torch.empty_strided((m, k), (lda, 1), dtype=torch.int8, device=device)
    w = torch.empty_strided((n, k), (ldb, 1), dtype=torch.int8, device=device)
    x.random_(-17, 18)
    w.random_(-17, 18)
    x_scale = torch.linspace(0.007, 0.19, m, dtype=torch.float32, device=device)
    w_scale = torch.linspace(0.011, 0.13, n, dtype=torch.float32, device=device)
    return x, w, x_scale, w_scale


def _check_correctness(dtype, gemm_op_a8w8, gemm_a8w8_triton):
    m, n, k = 70, 256, 256
    x, w, x_scale, w_scale = _make_inputs(m, n, k, padded=True)
    dispatch = []
    original = gemm_op_a8w8._try_flydsl_rdna3_a8w8

    def track_real_dispatch(*args):
        result = original(*args)
        dispatch.append(result is not None)
        return result

    gemm_op_a8w8._try_flydsl_rdna3_a8w8 = track_real_dispatch
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        actual = gemm_op_a8w8.gemm_a8w8(x, w, x_scale, w_scale, dtype=dtype)
        torch.cuda.synchronize()
        first_call_ms = (time.perf_counter() - start) * 1000
    finally:
        gemm_op_a8w8._try_flydsl_rdna3_a8w8 = original

    assert dispatch == [True], f"public A8W8 call did not take FlyDSL: {dispatch}"
    triton = gemm_a8w8_triton(x, w, x_scale, w_scale, dtype=dtype)
    reference = (x.float() @ w.float().T) * (x_scale[:, None] * w_scale[None, :])
    reference = reference.to(dtype)
    if dtype == torch.float32:
        rtol, atol = 1e-5, 1e-4
    elif dtype == torch.float16:
        rtol, atol = 2e-3, 2e-3
    else:
        rtol, atol = 2e-2, 2e-2
    torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
    torch.testing.assert_close(triton, reference, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual, triton, rtol=rtol, atol=atol)
    error = (actual.float() - reference.float()).abs()
    triton_error = (actual.float() - triton.float()).abs()
    return {
        "dtype": str(dtype),
        "shape_mnk": [m, n, k],
        "x_stride": list(x.stride()),
        "w_stride": list(w.stride()),
        "x_scale_shape": list(x_scale.shape),
        "w_scale_shape": list(w_scale.shape),
        "x_scale_range": [x_scale.min().item(), x_scale.max().item()],
        "w_scale_range": [w_scale.min().item(), w_scale.max().item()],
        "max_abs_error_vs_reference": error.max().item(),
        "rmse_vs_reference": error.square().mean().sqrt().item(),
        "max_abs_error_vs_triton": triton_error.max().item(),
        "first_public_call_ms_including_jit_or_cache_load": first_call_ms,
        "real_flydsl_dispatch_proven": dispatch == [True],
    }


def _check_extreme_scale_correctness(gemm_op_a8w8, gemm_a8w8_triton):
    m, n, k = 1, 64, 128
    device = torch.device("cuda")
    x = torch.full((m, k), 127, dtype=torch.int8, device=device)
    w = torch.full((n, k), 127, dtype=torch.int8, device=device)
    x_scale = torch.full((m,), 1e33, dtype=torch.float32, device=device)
    w_scale = torch.full((n,), 1e-33, dtype=torch.float32, device=device)
    dispatch = []
    original = gemm_op_a8w8._try_flydsl_rdna3_a8w8

    def track_real_dispatch(*args):
        result = original(*args)
        dispatch.append(result is not None)
        return result

    gemm_op_a8w8._try_flydsl_rdna3_a8w8 = track_real_dispatch
    try:
        actual = gemm_op_a8w8.gemm_a8w8(x, w, x_scale, w_scale, dtype=torch.float32)
        torch.cuda.synchronize()
    finally:
        gemm_op_a8w8._try_flydsl_rdna3_a8w8 = original

    assert dispatch == [True], f"public A8W8 call did not take FlyDSL: {dispatch}"
    triton = gemm_a8w8_triton(x, w, x_scale, w_scale, dtype=torch.float32)
    accumulator = x.float() @ w.float().T
    reference = accumulator * (x_scale[:, None] * w_scale[None, :])
    old_grouping = accumulator * x_scale[:, None] * w_scale[None, :]
    expected = torch.full((m, n), 2064512.0, dtype=torch.float32, device=device)
    assert not torch.isfinite(old_grouping).all()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(triton, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    return {
        "dtype": str(torch.float32),
        "shape_mnk": [m, n, k],
        "x_scale": 1e33,
        "w_scale": 1e-33,
        "expected": 2064512.0,
        "old_left_associated_grouping_is_finite": bool(
            torch.isfinite(old_grouping).all().item()
        ),
        "real_flydsl_dispatch_proven": dispatch == [True],
    }


def _benchmark_shape(m, n, k, gemm_op_a8w8, gemm_a8w8_triton):
    x, w, x_scale, w_scale = _make_inputs(m, n, k)
    flydsl_dispatches = []
    original = gemm_op_a8w8._try_flydsl_rdna3_a8w8

    def track_real_dispatch(*args):
        result = original(*args)
        flydsl_dispatches.append(result is not None)
        return result

    gemm_op_a8w8._try_flydsl_rdna3_a8w8 = track_real_dispatch
    try:
        # Separate first-call latency from the warmed measurements below.
        torch.cuda.synchronize()
        start = time.perf_counter()
        gemm_op_a8w8.gemm_a8w8(x, w, x_scale, w_scale, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        first_call_ms = (time.perf_counter() - start) * 1000
        flydsl_ms = _measure_ms(
            lambda: gemm_op_a8w8.gemm_a8w8(x, w, x_scale, w_scale, dtype=torch.bfloat16)
        )
    finally:
        gemm_op_a8w8._try_flydsl_rdna3_a8w8 = original

    assert all(flydsl_dispatches), "a timed public call fell back from FlyDSL"
    triton_ms = _measure_ms(
        lambda: gemm_a8w8_triton(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    )
    return {
        "dtype": str(torch.bfloat16),
        "shape_mnk": [m, n, k],
        "x_stride": list(x.stride()),
        "w_stride": list(w.stride()),
        "x_scale_shape": list(x_scale.shape),
        "w_scale_shape": list(w_scale.shape),
        "x_scale_range": [x_scale.min().item(), x_scale.max().item()],
        "w_scale_range": [w_scale.min().item(), w_scale.max().item()],
        "first_public_call_ms_including_jit_or_cache_load": first_call_ms,
        "flydsl_warm_event_ms": flydsl_ms,
        "triton_warm_event_ms": triton_ms,
        "flydsl_over_triton_ratio": flydsl_ms / triton_ms,
        "public_calls_proven_flydsl": len(flydsl_dispatches),
    }


def main():
    if not torch.cuda.is_available():
        print("SKIP: no HIP/CUDA device is available")
        return
    arch = str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0]
    if not arch.startswith("gfx11"):
        print(f"SKIP: RDNA3 FlyDSL A8W8 requires gfx11*, got {arch}")
        return
    try:
        import flydsl  # noqa: F401
    except (ImportError, OSError) as exc:
        print(f"SKIP: FlyDSL runtime is not importable: {exc}")
        return

    # Keep Aiter imports after the device check so CPU runners skip before JIT bootstrap.
    os.environ["AITER_GEMM_A8W8_BACKEND"] = "flydsl"
    from aiter.ops import gemm_op_a8w8
    from aiter.ops.triton.gemm.basic.gemm_a8w8 import (
        gemm_a8w8 as gemm_a8w8_triton,
    )

    torch.manual_seed(4340)
    cache_value = os.getenv("FLYDSL_RUNTIME_CACHE_DIR")
    cache_dir = Path(cache_value) if cache_value else None
    cache_existed = cache_dir.is_dir() if cache_dir is not None else False
    aiter_cache_value = os.getenv("AITER_JIT_DIR")
    aiter_cache_dir = Path(aiter_cache_value) if aiter_cache_value else None
    report = {
        "environment": {
            "python": sys.executable,
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "gpu_name": torch.cuda.get_device_properties(0).name,
            "gpu_arch": arch,
            "aiter_module": gemm_op_a8w8.__file__,
            "aiter_flydsl_kernel_module": importlib.import_module(
                "aiter.ops.flydsl.kernels.rdna3_int8_gemm"
            ).__file__,
            "flydsl_module": flydsl.__file__,
            "triton_gemm_module": importlib.import_module(
                "aiter.ops.triton.gemm.basic.gemm_a8w8"
            ).__file__,
            "aiter_jit_dir": aiter_cache_value,
            "aiter_jit_dir_existed_before_run": (
                aiter_cache_dir.is_dir() if aiter_cache_dir is not None else False
            ),
            "flydsl_runtime_cache_dir": cache_value,
            "flydsl_runtime_cache_dir_existed_before_run": cache_existed,
            "a8w8_backend": os.environ["AITER_GEMM_A8W8_BACKEND"],
            "input_seed": 4340,
        },
        "reference": "float32 torch.mm(x, w.T) * (row_scale * column_scale), cast to output dtype",
        "comparison_backend": "Triton; Aiter CK/asm A8W8 is gated to gfx9, so it is unavailable on gfx11",
        "first_call_note": (
            "Includes JIT compilation or persistent-cache restore; this number alone "
            "does not establish a cold compile."
        ),
        "warm_timing_note": "GPU event averages after first call, synchronization, and 5 warmups; JIT excluded.",
        "correctness": [],
        "performance": [],
    }

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        result = _check_correctness(dtype, gemm_op_a8w8, gemm_a8w8_triton)
        report["correctness"].append(result)
        print(f"correctness {dtype}: passed")

    result = _check_extreme_scale_correctness(gemm_op_a8w8, gemm_a8w8_triton)
    report["correctness"].append(result)
    print("correctness extreme scales: passed")

    for shape in ((1, 1024, 1024), (32, 1024, 1024), (256, 2048, 1024)):
        result = _benchmark_shape(*shape, gemm_op_a8w8, gemm_a8w8_triton)
        report["performance"].append(result)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
