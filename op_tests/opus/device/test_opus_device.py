# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Test OPUS device kernels via a single PyTorch extension (opus_device_test).
Covers:
  - MFMA 32x32x8 fp16 (gfx942 only)
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

_MFMA_SUPPORTED_ARCHS = {"gfx942"}


def test_mfma(mod):
    """Test MFMA 32x32x8 fp16 kernel (gfx942 only)."""
    arch = _get_gpu_arch()
    if arch not in _MFMA_SUPPORTED_ARCHS:
        print(f"  SKIP: mfma requires {_MFMA_SUPPORTED_ARCHS}, got '{arch}'")
        return 0

    M, N, K = 32, 32, 8
    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(12345)
    A = torch.randint(-10, 11, (M, K), device=device).to(dtype)
    B = torch.randint(-10, 11, (N, K), device=device).to(dtype)
    C = torch.empty(M, N, device=device, dtype=dtype)

    mod.run_mfma_32x32x8_f16(A, B, C)

    # swap_ab net result in row-major memory: C = A @ B^T
    C_ref = torch.mm(A.float(), B.float().t()).to(dtype)

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
            f"  FAIL: mfma_32x32x8_f16 max_diff={max_diff:.4f}, "
            f"{diff_count} elements outside tol"
        )
        return 1
    print(f"  PASS: mfma_32x32x8_f16, max_diff={max_diff:.4f}")
    return 0


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

    # Reference: OPUS default bf16 conversion is truncation (rm=2):
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


def test_dtype_convert_fp32_fp8(mod):
    """Test FP32 -> FP8 (e4m3) -> FP32 round-trip via OPUS packed cast."""
    # n must be a multiple of BLOCK_SIZE * 4 = 1024
    n = 1048576  # 1M elements
    device = torch.device("cuda")

    torch.manual_seed(202)
    # FP8 e4m3 range is approx [-448, 448]; keep values small to avoid overflow
    In = torch.randn(n, device=device, dtype=torch.float32) * 2.0
    Out = torch.empty(n, device=device, dtype=torch.float32)

    mod.run_dtype_convert(In, Out, "fp32_fp8")

    # Reference: PyTorch fp32 -> fp8_e4m3fnuz -> fp32 round-trip
    Ref = In.to(torch.float8_e4m3fnuz).to(torch.float32)

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
    failures += test_mfma(mod)
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
