# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Test OPUS device kernels via a single PyTorch extension (opus_device_test).
Covers:
  - MFMA 32x32x8 fp16 (gfx942 only)
  - vector_add (all GPUs)
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

    if failures:
        print(f"\n{failures} test(s) FAILED")
    else:
        print("\nAll device tests PASSED")
    return failures


if __name__ == "__main__":
    sys.exit(main())
