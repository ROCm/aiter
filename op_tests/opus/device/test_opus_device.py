# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Test OPUS device kernels via a single PyTorch extension (opus_device_test).
Covers:
  - MFMA 32x32x8   fp16/bf16 (gfx942 only)
  - MFMA 16x16x16  fp16/bf16 (gfx942 only)
  - MFMA 32x32x16  fp16/bf16 (gfx942 + gfx950)
  - MFMA 16x16x32  fp16/bf16 (gfx942 + gfx950)
  - MFMA 32x32x16  fp8/bf8  (gfx942 + gfx950, fp32 output)
  - MFMA 16x16x32  fp8/bf8  (gfx942 + gfx950, fp32 output)
  - vector_add (all GPUs)
  - async_load (all GPUs)
  - dtype_convert: FP32<->BF16, FP32<->FP16, FP32<->FP8 round-trips (all GPUs)
"""

import glob
import os
import subprocess
import sys

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: F401
except ImportError as e:
    print(f"SKIP: PyTorch or C++ extension support not available ({e})")
    sys.exit(0)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_NAME = "opus_device_test"


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _clean_previous_extension():
    """Remove previously built extension and build dir for a fresh build."""
    removed = []
    for pattern in (f"{_MODULE_NAME}*.so", f"{_MODULE_NAME}*.pyd"):
        for path in glob.glob(os.path.join(_THIS_DIR, pattern)):
            try:
                os.remove(path)
                removed.append(path)
            except OSError as e:
                print(f"WARNING: could not remove {path}: {e}", file=sys.stderr)
    build_dir = os.path.join(_THIS_DIR, "build")
    if os.path.isdir(build_dir):
        try:
            import shutil

            shutil.rmtree(build_dir)
            removed.append(build_dir)
        except OSError as e:
            print(f"WARNING: could not remove {build_dir}: {e}", file=sys.stderr)
    if removed:
        print(
            "Cleaned previous extension:",
            " ".join(os.path.basename(p) for p in removed),
        )


def _ensure_extension_built():
    """Build extension with setup.py if not already importable."""
    try:
        __import__(_MODULE_NAME)
        return True
    except ModuleNotFoundError:
        pass
    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)
    try:
        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=_THIS_DIR,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"FAIL: Build exited with code {e.returncode}", file=sys.stderr)
        return False
    return True


def _get_gpu_arch():
    """Return the GCN architecture name of the current GPU, e.g. 'gfx942'."""
    props = torch.cuda.get_device_properties(0)
    return getattr(props, "gcnArchName", "").split(":")[0]


# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------

# Arch sets for runtime gating
_MFMA_ARCHS_GFX942 = {"gfx942"}  # 32x32x8, 16x16x16
_MFMA_ARCHS_GFX942_GFX950 = {"gfx942", "gfx950"}  # 32x32x16, 16x16x32


# FP8/BF8 torch dtypes differ by architecture:
#   gfx942: float8_e4m3fnuz / float8_e5m2fnuz  (OCP "fnuz" format)
#   gfx950: float8_e4m3fn   / float8_e5m2       (OCP non-"nuz" format)
_FP8_LIKE_DTYPES = {
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
}


def _get_fp8_dtype():
    """Return the correct FP8 (e4m3) torch dtype for the current GPU arch."""
    arch = _get_gpu_arch()
    if arch == "gfx950":
        return torch.float8_e4m3fn
    return torch.float8_e4m3fnuz  # gfx942 default


def _get_bf8_dtype():
    """Return the correct BF8 (e5m2) torch dtype for the current GPU arch."""
    arch = _get_gpu_arch()
    if arch == "gfx950":
        return torch.float8_e5m2
    return torch.float8_e5m2fnuz  # gfx942 default


def _test_mfma_variant(mod, variant, M, N, K, dtype, supported_archs):
    """Test a single MFMA variant. Returns 0 on pass, 1 on fail.

    For fp8/bf8 dtypes the kernel outputs raw fp32 accumulator (no cast back),
    so C is float32 and we compare against an fp32 reference.
    """
    arch = _get_gpu_arch()
    if arch not in supported_archs:
        print(f"  SKIP: mfma_{variant} requires {supported_archs}, got '{arch}'")
        return 0

    device = torch.device("cuda")
    is_fp8_like = dtype in _FP8_LIKE_DTYPES

    torch.manual_seed(12345)
    if is_fp8_like:
        # Use small integers that are exactly representable in fp8/bf8
        A = torch.randint(-3, 4, (M, K), device=device).float().to(dtype)
        B = torch.randint(-3, 4, (N, K), device=device).float().to(dtype)
        out_dtype = torch.float32  # kernel stores fp32 accumulator
    else:
        A = torch.randint(-10, 11, (M, K), device=device).to(dtype)
        B = torch.randint(-10, 11, (N, K), device=device).to(dtype)
        out_dtype = dtype

    C = torch.empty(M, N, device=device, dtype=out_dtype)

    mod.run_mfma(A, B, C, variant)

    # swap_ab net result in row-major memory: C = A @ B^T
    # Kernel uses opus::cast with RNE for bf16, matching PyTorch .to(bfloat16).
    # For fp8/bf8: inputs are exact small ints, accumulator is fp32 -> exact result.
    C_ref = torch.mm(A.float(), B.float().t()).to(out_dtype)

    atol, rtol = 1e-3, 1e-3
    ok = torch.allclose(C.float(), C_ref.float(), atol=atol, rtol=rtol)
    max_diff = (C.float() - C_ref.float()).abs().max().item()
    if not ok:
        diff_count = (
            (C.float() - C_ref.float())
            .abs()
            .gt(atol + rtol * C_ref.float().abs())
            .sum()
            .item()
        )
        print(
            f"  FAIL: mfma_{variant} max_diff={max_diff:.4f}, "
            f"{diff_count} elements outside tol"
        )
        return 1
    print(f"  PASS: mfma_{variant}, max_diff={max_diff:.4f}")
    return 0


def test_mfma_32x32x8_f16(mod):
    """Test MFMA 32x32x8 fp16 kernel (gfx942 only)."""
    return _test_mfma_variant(
        mod, "32x32x8_f16", 32, 32, 8, torch.float16, _MFMA_ARCHS_GFX942
    )


def test_mfma_32x32x8_bf16(mod):
    """Test MFMA 32x32x8 bf16 kernel (gfx942 only)."""
    return _test_mfma_variant(
        mod, "32x32x8_bf16", 32, 32, 8, torch.bfloat16, _MFMA_ARCHS_GFX942
    )


def test_mfma_16x16x16_f16(mod):
    """Test MFMA 16x16x16 fp16 kernel (gfx942 only)."""
    return _test_mfma_variant(
        mod, "16x16x16_f16", 16, 16, 16, torch.float16, _MFMA_ARCHS_GFX942
    )


def test_mfma_16x16x16_bf16(mod):
    """Test MFMA 16x16x16 bf16 kernel (gfx942 only)."""
    return _test_mfma_variant(
        mod, "16x16x16_bf16", 16, 16, 16, torch.bfloat16, _MFMA_ARCHS_GFX942
    )


def test_mfma_32x32x16_f16(mod):
    """Test MFMA 32x32x16 fp16 kernel (gfx942 step_k + gfx950 native)."""
    return _test_mfma_variant(
        mod, "32x32x16_f16", 32, 32, 16, torch.float16, _MFMA_ARCHS_GFX942_GFX950
    )


def test_mfma_32x32x16_bf16(mod):
    """Test MFMA 32x32x16 bf16 kernel (gfx942 step_k + gfx950 native)."""
    return _test_mfma_variant(
        mod, "32x32x16_bf16", 32, 32, 16, torch.bfloat16, _MFMA_ARCHS_GFX942_GFX950
    )


def test_mfma_16x16x32_f16(mod):
    """Test MFMA 16x16x32 fp16 kernel (gfx942 step_k + gfx950 native)."""
    return _test_mfma_variant(
        mod, "16x16x32_f16", 16, 16, 32, torch.float16, _MFMA_ARCHS_GFX942_GFX950
    )


def test_mfma_16x16x32_bf16(mod):
    """Test MFMA 16x16x32 bf16 kernel (gfx942 step_k + gfx950 native)."""
    return _test_mfma_variant(
        mod, "16x16x32_bf16", 16, 16, 32, torch.bfloat16, _MFMA_ARCHS_GFX942_GFX950
    )


def test_mfma_32x32x16_fp8(mod):
    """Test MFMA 32x32x16 fp8 kernel (native, fp32 output)."""
    return _test_mfma_variant(
        mod,
        "32x32x16_fp8",
        32,
        32,
        16,
        _get_fp8_dtype(),
        _MFMA_ARCHS_GFX942_GFX950,
    )


def test_mfma_32x32x16_bf8(mod):
    """Test MFMA 32x32x16 bf8 kernel (native, fp32 output)."""
    return _test_mfma_variant(
        mod,
        "32x32x16_bf8",
        32,
        32,
        16,
        _get_bf8_dtype(),
        _MFMA_ARCHS_GFX942_GFX950,
    )


def test_mfma_16x16x32_fp8(mod):
    """Test MFMA 16x16x32 fp8 kernel (native, fp32 output)."""
    return _test_mfma_variant(
        mod,
        "16x16x32_fp8",
        16,
        16,
        32,
        _get_fp8_dtype(),
        _MFMA_ARCHS_GFX942_GFX950,
    )


def test_mfma_16x16x32_bf8(mod):
    """Test MFMA 16x16x32 bf8 kernel (native, fp32 output)."""
    return _test_mfma_variant(
        mod,
        "16x16x32_bf8",
        16,
        16,
        32,
        _get_bf8_dtype(),
        _MFMA_ARCHS_GFX942_GFX950,
    )


def test_vector_add(mod):
    """Test vector addition kernel."""
    n = 1310720
    device = torch.device("cuda")
    dtype = torch.float32

    torch.manual_seed(42)
    A = torch.randn(n, device=device, dtype=dtype)
    B = torch.randn(n, device=device, dtype=dtype)
    Result = torch.empty(n, device=device, dtype=dtype)

    mod.run_vector_add(A, B, Result)

    Ref = A + B

    atol, rtol = 1e-5, 1e-5
    ok = torch.allclose(Result, Ref, atol=atol, rtol=rtol)
    max_diff = (Result - Ref).abs().max().item()
    if not ok:
        diff_count = (Result - Ref).abs().gt(atol + rtol * Ref.abs()).sum().item()
        print(
            f"  FAIL: vector_add max_diff={max_diff:.6e}, "
            f"{diff_count} elements outside tol"
        )
        return 1
    print(f"  PASS: vector_add (n={n}), max_diff={max_diff:.6e}")
    return 0


def test_async_load(mod):
    """Test async_load: copy data through LDS and verify integrity."""
    # n should be a multiple of BLOCK_SIZE (256)
    n = 1048576  # 1M elements
    device = torch.device("cuda")
    dtype = torch.float32

    torch.manual_seed(99)
    Src = torch.randn(n, device=device, dtype=dtype)
    Dst = torch.empty(n, device=device, dtype=dtype)

    mod.run_async_load(Src, Dst)

    # Output should be bit-identical to input (float copy, no arithmetic)
    ok = torch.equal(Src, Dst)
    if not ok:
        diff = (Src - Dst).abs()
        max_diff = diff.max().item()
        diff_count = diff.gt(0).sum().item()
        print(
            f"  FAIL: async_load max_diff={max_diff:.6e}, "
            f"{diff_count}/{n} elements differ"
        )
        return 1
    print(f"  PASS: async_load (n={n}), bit-exact copy")
    return 0


def test_dtype_convert_fp32_bf16(mod):
    """Test FP32 -> BF16 -> FP32 round-trip via OPUS cast."""
    n = 1048576  # 1M elements
    device = torch.device("cuda")

    torch.manual_seed(200)
    In = torch.randn(n, device=device, dtype=torch.float32)
    Out = torch.empty(n, device=device, dtype=torch.float32)

    mod.run_dtype_convert(In, Out, "fp32_bf16")

    arch = _get_gpu_arch()
    if arch == "gfx950":
        # gfx950 uses native hardware bf16 conversion (round-to-nearest-even),
        # which matches PyTorch's .to(bfloat16) behaviour.
        Ref = In.to(torch.bfloat16).to(torch.float32)
    else:
        # Pre-gfx950: OPUS default bf16 conversion is truncation (rm=2):
        # simply discard the lower 16 bits of the float32 representation.
        # PyTorch .to(bfloat16) uses round-to-nearest-even which differs,
        # so we replicate the truncation in Python via bitwise ops.
        bits = In.view(dtype=torch.int32) & 0xFFFF0000
        Ref = bits.view(dtype=torch.float32)

    ok = torch.equal(Out, Ref)
    if not ok:
        diff = (Out - Ref).abs()
        max_diff = diff.max().item()
        diff_count = diff.gt(0).sum().item()
        print(
            f"  FAIL: dtype_convert fp32<->bf16 max_diff={max_diff:.6e}, "
            f"{diff_count}/{n} elements differ"
        )
        return 1
    print(f"  PASS: dtype_convert fp32<->bf16 (n={n}), bit-exact")
    return 0


def test_dtype_convert_fp32_fp16(mod):
    """Test FP32 -> FP16 -> FP32 round-trip via OPUS cast."""
    n = 1048576  # 1M elements
    device = torch.device("cuda")

    torch.manual_seed(201)
    # Use a smaller range to stay within fp16 representable values
    In = torch.randn(n, device=device, dtype=torch.float32)
    Out = torch.empty(n, device=device, dtype=torch.float32)

    mod.run_dtype_convert(In, Out, "fp32_fp16")

    # Reference: PyTorch fp32 -> fp16 -> fp32 round-trip
    Ref = In.to(torch.float16).to(torch.float32)

    atol, rtol = 1e-4, 1e-4
    ok = torch.allclose(Out, Ref, atol=atol, rtol=rtol)
    max_diff = (Out - Ref).abs().max().item()
    if not ok:
        diff_count = (Out - Ref).abs().gt(atol + rtol * Ref.abs()).sum().item()
        print(
            f"  FAIL: dtype_convert fp32<->fp16 max_diff={max_diff:.6e}, "
            f"{diff_count}/{n} elements outside tol"
        )
        return 1
    print(f"  PASS: dtype_convert fp32<->fp16 (n={n}), max_diff={max_diff:.6e}")
    return 0


_FP8_SUPPORTED_ARCHS = {"gfx942"}


def test_dtype_convert_fp32_fp8(mod):
    """Test FP32 -> FP8 (e4m3) -> FP32 round-trip via OPUS packed cast."""
    arch = _get_gpu_arch()
    if arch not in _FP8_SUPPORTED_ARCHS:
        print(
            f"  SKIP: dtype_convert fp32<->fp8 requires {_FP8_SUPPORTED_ARCHS}, got '{arch}'"
        )
        return 0

    # n must be a multiple of BLOCK_SIZE * 4 = 1024
    n = 1048576  # 1M elements
    device = torch.device("cuda")

    torch.manual_seed(202)
    # FP8 e4m3 range is approx [-448, 448]; keep values small to avoid overflow
    In = torch.randn(n, device=device, dtype=torch.float32) * 2.0
    Out = torch.empty(n, device=device, dtype=torch.float32)

    mod.run_dtype_convert(In, Out, "fp32_fp8")

    # Reference: PyTorch fp32 -> fp8 -> fp32 round-trip (arch-dependent dtype)
    fp8_dtype = _get_fp8_dtype()
    Ref = In.to(fp8_dtype).to(torch.float32)

    # FP8 has low precision (3 mantissa bits) so tolerance is larger
    atol, rtol = 0.5, 0.25
    ok = torch.allclose(Out, Ref, atol=atol, rtol=rtol)
    max_diff = (Out - Ref).abs().max().item()
    if not ok:
        diff_count = (Out - Ref).abs().gt(atol + rtol * Ref.abs()).sum().item()
        print(
            f"  FAIL: dtype_convert fp32<->fp8 max_diff={max_diff:.6e}, "
            f"{diff_count}/{n} elements outside tol"
        )
        return 1
    print(f"  PASS: dtype_convert fp32<->fp8 (n={n}), max_diff={max_diff:.6e}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0

    _clean_previous_extension()
    arch = _get_gpu_arch()
    print(f"GPU arch: {arch}")
    print(f"Building {_MODULE_NAME} extension ...")
    if not _ensure_extension_built():
        return 1

    mod = __import__(_MODULE_NAME)

    failures = 0
    failures += test_mfma_32x32x8_f16(mod)
    failures += test_mfma_32x32x8_bf16(mod)
    failures += test_mfma_16x16x16_f16(mod)
    failures += test_mfma_16x16x16_bf16(mod)
    failures += test_mfma_32x32x16_f16(mod)
    failures += test_mfma_32x32x16_bf16(mod)
    failures += test_mfma_16x16x32_f16(mod)
    failures += test_mfma_16x16x32_bf16(mod)
    failures += test_mfma_32x32x16_fp8(mod)
    failures += test_mfma_32x32x16_bf8(mod)
    failures += test_mfma_16x16x32_fp8(mod)
    failures += test_mfma_16x16x32_bf8(mod)
    failures += test_vector_add(mod)
    failures += test_async_load(mod)
    failures += test_dtype_convert_fp32_bf16(mod)
    failures += test_dtype_convert_fp32_fp16(mod)
    failures += test_dtype_convert_fp32_fp8(mod)

    if failures:
        print(f"\n{failures} test(s) FAILED")
    else:
        print("\nAll device tests PASSED")
    return failures


if __name__ == "__main__":
    sys.exit(main())
