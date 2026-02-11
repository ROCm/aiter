# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Test OPUS 32x32x8 fp16 MFMA via PyTorch extension: random A, B, compare with torch GEMM.
Uses BuildExtension + CUDAExtension (see setup.py). Cleans previous build each run, then builds.
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


def _clean_previous_extension():
    """Remove any previously built opus_mfma extension and build dir so each run does a fresh build."""
    removed = []
    for pattern in ("opus_mfma*.so", "opus_mfma*.pyd"):
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
    """Build extension with setup.py if opus_mfma is not importable."""
    try:
        import opus_mfma  # noqa: F401

        return True
    except ModuleNotFoundError:
        pass
    # Build in-place from this directory
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


_SUPPORTED_ARCHS = {"gfx942"}


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0

    arch = _get_gpu_arch()
    if arch not in _SUPPORTED_ARCHS:
        print(f"SKIP: GPU arch '{arch}' not supported (need one of {_SUPPORTED_ARCHS})")
        return 0

    _clean_previous_extension()
    print(f"GPU arch: {arch}")
    print("Building OPUS MFMA extension (BuildExtension + CUDAExtension)...")
    if not _ensure_extension_built():
        return 1

    import opus_mfma

    M, N, K = 32, 32, 8
    device = torch.device("cuda")
    dtype = torch.float16

    # Random integer A (MxK), B (NxK) in [-10, 10]
    torch.manual_seed(12345)
    A = torch.randint(-10, 11, (M, K), device=device).to(dtype)
    B = torch.randint(-10, 11, (N, K), device=device).to(dtype)
    C = torch.empty(M, N, device=device, dtype=dtype)

    opus_mfma.run_mfma_32x32x8_f16(A, B, C)

    # Kernel uses mfma_adaptor_swap_ab which internally swaps A/B in the MFMA
    # instruction and transposes the C register layout. The net result stored
    # in row-major memory is C = A @ B^T (same as gemm_rcr in the reference).
    C_ref = torch.mm(A.float(), B.float().t()).to(dtype)

    atol = 1e-3
    rtol = 1e-3
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
            f"FAIL: MFMA vs reference max_diff={max_diff:.4f}, "
            f"{diff_count} elements outside tol (atol={atol}, rtol={rtol})"
        )
        return 1
    print(f"PASS: MFMA 32x32x8 fp16 (OPUS ext, block_v2), max_diff={max_diff:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
