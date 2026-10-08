# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Manual gfx11 FlyDSL A8W8 correctness and timing check.

Run with the active ROCm Python after building/installing the FlyDSL runtime:
    python op_tests/test_gemm_a8w8_flydsl_runtime.py
"""

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
    x.random_(from_=-17, to=18)
    w.random_(from_=-17, to=18)
    x_scale = torch.linspace(0.007, 0.19, m, dtype=torch.float32, device=device)
    w_scale = torch.linspace(0.011, 0.13, n, dtype=torch.float32, device=device)
    return x, w, x_scale, w_scale


def _check_correctness(dtype):
    from aiter.ops import gemm_op_a8w8
    from aiter.ops.triton.gemm.basic.gemm_a8w8 import (
        gemm_a8w8 as gemm_a8w8_triton,
    )

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
        actual = gemm_op_a8w8.gemm_a8w8(
            x, w, x_scale, w_scale, dtype=dtype
        )
    finally:
        gemm_op_a8w8._try_flydsl_rdna3_a8w8 = original

    assert dispatch == [True], f"public A8W8 call did not take FlyDSL: {dispatch}"
    triton = gemm_a8w8_triton(x, w, x_scale, w_scale, dtype=dtype)
    reference = (x.float() @ w.float().T) * x_scale[:, None] * w_scale[None, :]
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
    return actual


def _benchmark_shape(m, n, k):
    from aiter.ops import gemm_op_a8w8
    from aiter.ops.triton.gemm.basic.gemm_a8w8 import (
        gemm_a8w8 as gemm_a8w8_triton,
    )

    x, w, x_scale, w_scale = _make_inputs(m, n, k)
    flydsl_dispatches = []
    original = gemm_op_a8w8._try_flydsl_rdna3_a8w8

    def track_real_dispatch(*args):
        result = original(*args)
        flydsl_dispatches.append(result is not None)
        return result

    gemm_op_a8w8._try_flydsl_rdna3_a8w8 = track_real_dispatch
    try:
        # First call triggers JIT compilation; it is deliberately excluded from timing.
        gemm_op_a8w8.gemm_a8w8(x, w, x_scale, w_scale, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        flydsl_ms = _measure_ms(
            lambda: gemm_op_a8w8.gemm_a8w8(
                x, w, x_scale, w_scale, dtype=torch.bfloat16
            )
        )
    finally:
        gemm_op_a8w8._try_flydsl_rdna3_a8w8 = original

    assert all(flydsl_dispatches), "a timed public call fell back from FlyDSL"
    triton_ms = _measure_ms(
        lambda: gemm_a8w8_triton(
            x, w, x_scale, w_scale, dtype=torch.bfloat16
        )
    )
    return flydsl_ms, triton_ms


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

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        result = _check_correctness(dtype)
        print(f"correctness {dtype}: passed, output={tuple(result.shape)}")

    for shape in ((1, 1024, 1024), (32, 1024, 1024), (256, 2048, 1024)):
        flydsl_ms, triton_ms = _benchmark_shape(*shape)
        print(
            f"timing BF16 M/N/K={shape}: FlyDSL={flydsl_ms:.4f} ms, "
            f"Triton={triton_ms:.4f} ms, FlyDSL/Triton={flydsl_ms / triton_ms:.3f}x"
        )


if __name__ == "__main__":
    main()
