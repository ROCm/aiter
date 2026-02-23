# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Test OPUS device kernels via a single PyTorch extension (opus_device_test).
Covers:
  - MFMA 32x32x2   fp32      (gfx942 + gfx950)
  - MFMA 16x16x4   fp32      (gfx942 + gfx950)
  - MFMA 32x32x8   fp16/bf16 (gfx942 only)
  - MFMA 16x16x16  fp16/bf16 (gfx942 only)
  - MFMA 32x32x16  fp16/bf16 (gfx942 + gfx950)
  - MFMA 16x16x32  fp16/bf16 (gfx942 + gfx950)
  - MFMA 32x32x16  fp8/bf8  (gfx942 + gfx950, fp32 output)
  - MFMA 16x16x32  fp8/bf8  (gfx942 + gfx950, fp32 output)
  - MXFP8 32x32x64  fp8*fp8 (gfx950 only, fp32 output)
  - MXFP8 16x16x128 fp8*fp8 (gfx950 only, fp32 output)
  - MXFP4 32x32x64  fp4*fp4 (gfx950 only, fp32 output)
  - MXFP4 16x16x128 fp4*fp4 (gfx950 only, fp32 output)
  - vector_add (all GPUs)
  - async_load (all GPUs)
  - dtype_convert: FP32<->BF16, FP32<->FP16, FP32<->FP8 round-trips (all GPUs)
  - dtype_convert: FP32<->FP4 (e2m1) round-trip (gfx950 only, packed x8)
  - predicated_copy: gmem load_if/store_if with boundary predicate (all GPUs)
  - free_func_vector_add: opus::load/store free function API (all GPUs)
  - predicated_async_load: gmem async_load_if with boundary predicate (all GPUs)
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


def test_mfma_32x32x2_f32(mod):
    """Test MFMA 32x32x2 fp32 kernel (gfx942 + gfx950)."""
    return _test_mfma_variant(
        mod, "32x32x2_f32", 32, 32, 2, torch.float32, _MFMA_ARCHS_GFX942_GFX950
    )


def test_mfma_16x16x4_f32(mod):
    """Test MFMA 16x16x4 fp32 kernel (gfx942 + gfx950)."""
    return _test_mfma_variant(
        mod, "16x16x4_f32", 16, 16, 4, torch.float32, _MFMA_ARCHS_GFX942_GFX950
    )


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


# ---------------------------------------------------------------------------
# MXFP tests (gfx950 only)
# ---------------------------------------------------------------------------

_MXFP_SUPPORTED_ARCHS = {"gfx950"}

# FP4 E2M1 representable magnitudes (3-bit unsigned magnitude code)
_FP4_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_FP4_ALL_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] + [
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _fp32_to_fp4_nibble(val):
    """Convert a single fp32 value to a 4-bit FP4 E2M1 code."""
    sign_bit = 0
    v = float(val)
    if v < 0.0:
        sign_bit = 8  # bit 3
        v = -v
    elif v == 0.0:
        return 0
    for code, mag in enumerate(_FP4_MAGNITUDES):
        if abs(v - mag) < 1e-6:
            return sign_bit | code
    raise ValueError(f"Value {val} is not representable in FP4 E2M1")


def _pack_fp4_tensor(fp32_tensor):
    """Pack fp32 tensor of FP4-representable values into uint8 (2 fp4 per byte).

    Packing is along the last dimension (which must be even):
      byte[..., j] = fp4_encode(tensor[..., 2*j]) | (fp4_encode(tensor[..., 2*j+1]) << 4)
    """
    assert fp32_tensor.shape[-1] % 2 == 0, "Last dim must be even for fp4 packing"
    flat = fp32_tensor.contiguous().view(-1).tolist()
    packed = []
    for i in range(0, len(flat), 2):
        lo = _fp32_to_fp4_nibble(flat[i])
        hi = _fp32_to_fp4_nibble(flat[i + 1])
        packed.append(lo | (hi << 4))
    shape = list(fp32_tensor.shape)
    shape[-1] //= 2
    return torch.tensor(packed, dtype=torch.uint8).reshape(shape)


def _test_mxfp8(mod, variant, M, N, K):
    """Test an MXFP8 variant. Returns 0 on pass/skip, 1 on fail."""
    arch = _get_gpu_arch()
    if arch not in _MXFP_SUPPORTED_ARCHS:
        print(f"  SKIP: {variant} requires {_MXFP_SUPPORTED_ARCHS}, " f"got '{arch}'")
        return 0

    device = torch.device("cuda")
    fp8_dtype = _get_fp8_dtype()

    torch.manual_seed(42)
    # Small integers exactly representable in fp8
    A = torch.randint(-3, 4, (M, K), device=device).float().to(fp8_dtype)
    B = torch.randint(-3, 4, (K, N), device=device).float().to(fp8_dtype)
    C = torch.empty(M, N, device=device, dtype=torch.float32)

    # scale=127 -> 2^0 = 1.0 (no scaling)
    mod.run_mxfp(A, B, C, variant, 127, 127)

    # Reference: standard matmul (C = A @ B, no transpose)
    C_ref = torch.mm(A.float(), B.float())

    atol, rtol = 1e-3, 1e-3
    ok = torch.allclose(C, C_ref, atol=atol, rtol=rtol)
    max_diff = (C - C_ref).abs().max().item()
    if not ok:
        diff_count = (C - C_ref).abs().gt(atol + rtol * C_ref.abs()).sum().item()
        print(
            f"  FAIL: {variant} max_diff={max_diff:.4f}, "
            f"{diff_count} elements outside tol"
        )
        return 1
    print(f"  PASS: {variant}, max_diff={max_diff:.4f}")
    return 0


def _test_mxfp4(mod, variant, M, N, K):
    """Test an MXFP4 variant. Returns 0 on pass/skip, 1 on fail."""
    arch = _get_gpu_arch()
    if arch not in _MXFP_SUPPORTED_ARCHS:
        print(f"  SKIP: {variant} requires {_MXFP_SUPPORTED_ARCHS}, " f"got '{arch}'")
        return 0

    device = torch.device("cuda")

    # Generate random FP4-representable values
    fp4_values = torch.tensor(_FP4_ALL_VALUES, dtype=torch.float32)
    torch.manual_seed(42)
    A_fp32 = fp4_values[torch.randint(0, len(fp4_values), (M, K))]
    B_fp32 = fp4_values[torch.randint(0, len(fp4_values), (K, N))]

    # Pack into fp4 bytes (2 values per byte)
    A_packed = _pack_fp4_tensor(A_fp32).to(device)  # [M, K//2] uint8
    B_packed = _pack_fp4_tensor(B_fp32).to(device)  # [K, N//2] uint8
    C = torch.empty(M, N, device=device, dtype=torch.float32)

    # scale=127 -> 2^0 = 1.0 (no scaling)
    mod.run_mxfp(A_packed, B_packed, C, variant, 127, 127)

    # Reference: standard matmul in fp32
    C_ref = torch.mm(A_fp32.to(device), B_fp32.to(device))

    atol, rtol = 1e-3, 1e-3
    ok = torch.allclose(C, C_ref, atol=atol, rtol=rtol)
    max_diff = (C - C_ref).abs().max().item()
    if not ok:
        diff_count = (C - C_ref).abs().gt(atol + rtol * C_ref.abs()).sum().item()
        print(
            f"  FAIL: {variant} max_diff={max_diff:.4f}, "
            f"{diff_count} elements outside tol"
        )
        return 1
    print(f"  PASS: {variant}, max_diff={max_diff:.4f}")
    return 0


def test_mxfp8_32x32x64(mod):
    """Test MXFP8 32x32x64 fp8*fp8 (gfx950 only)."""
    return _test_mxfp8(mod, "mxfp8_32x32x64", 32, 32, 64)


def test_mxfp8_16x16x128(mod):
    """Test MXFP8 16x16x128 fp8*fp8 (gfx950 only)."""
    return _test_mxfp8(mod, "mxfp8_16x16x128", 16, 16, 128)


def test_mxfp4_32x32x64(mod):
    """Test MXFP4 32x32x64 fp4*fp4 (gfx950 only)."""
    return _test_mxfp4(mod, "mxfp4_32x32x64", 32, 32, 64)


def test_mxfp4_16x16x128(mod):
    """Test MXFP4 16x16x128 fp4*fp4 (gfx950 only)."""
    return _test_mxfp4(mod, "mxfp4_16x16x128", 16, 16, 128)


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
    """Test FP32 -> BF16 -> FP32 round-trip via OPUS cast (RNE on all archs).

    The kernel uses round-to-nearest-even for FP32->BF16 on every architecture:
      - gfx942: opus::cast<bf16_t>(val, 0_I) -- explicit RNE via 2nd parameter
      - gfx950: opus::cast<bf16_t>(val)       -- hardware default is RNE
    PyTorch .to(bfloat16) also uses RNE, so we compare against that directly.
    """
    n = 1048576  # 1M elements
    device = torch.device("cuda")

    torch.manual_seed(200)
    In = torch.randn(n, device=device, dtype=torch.float32)
    Out = torch.empty(n, device=device, dtype=torch.float32)

    mod.run_dtype_convert(In, Out, "fp32_bf16")

    # Both gfx942 (with 0_I) and gfx950 (native) use RNE,
    # which matches PyTorch's .to(bfloat16).
    Ref = In.to(torch.bfloat16).to(torch.float32)

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


_FP8_SUPPORTED_ARCHS = {"gfx942", "gfx950"}
_FP4_SUPPORTED_ARCHS = {"gfx950"}


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


def test_dtype_convert_fp32_fp4(mod):
    """Test FP32 -> FP4 (e2m1) -> FP32 round-trip via OPUS packed cast (gfx950 only)."""
    arch = _get_gpu_arch()
    if arch not in _FP4_SUPPORTED_ARCHS:
        print(
            f"  SKIP: dtype_convert fp32<->fp4 requires {_FP4_SUPPORTED_ARCHS}, got '{arch}'"
        )
        return 0

    # n must be a multiple of BLOCK_SIZE * 8 = 2048
    n = 1048576  # 1M elements
    device = torch.device("cuda")

    # FP4 E2M1 representable values (with scale=1.0):
    #   +-{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    # Use exactly representable values so the round-trip is bit-exact.
    fp4_values = torch.tensor(
        [
            -6.0,
            -4.0,
            -3.0,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
        ],
        dtype=torch.float32,
    )

    torch.manual_seed(203)
    indices = torch.randint(0, len(fp4_values), (n,))
    In = fp4_values[indices].to(device=device)
    Out = torch.empty(n, device=device, dtype=torch.float32)

    mod.run_dtype_convert(In, Out, "fp32_fp4")

    # Reference: input values are exactly representable in FP4,
    # so the round-trip should be bit-exact.
    Ref = In

    ok = torch.equal(Out, Ref)
    if not ok:
        diff = (Out - Ref).abs()
        max_diff = diff.max().item()
        diff_count = diff.gt(0).sum().item()
        print(
            f"  FAIL: dtype_convert fp32<->fp4 max_diff={max_diff:.6e}, "
            f"{diff_count}/{n} elements differ"
        )
        return 1
    print(f"  PASS: dtype_convert fp32<->fp4 (n={n}), bit-exact")
    return 0


# ---------------------------------------------------------------------------
# Predicated load/store and free function API tests (all GPUs)
# ---------------------------------------------------------------------------


def test_predicated_copy(mod):
    """Test gmem load_if/store_if via free function wrappers (boundary predicate)."""
    # Use n not aligned to block*4 to create a partial boundary condition
    n = 1001
    BLOCK_SIZE = 256
    ELEMS = 4
    total_threads = (n + ELEMS - 1) // ELEMS
    n_padded = ((total_threads + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE * ELEMS

    device = torch.device("cuda")

    torch.manual_seed(42)
    Src = torch.randn(n, device=device, dtype=torch.float32)
    Dst = torch.full((n_padded,), -1.0, device=device, dtype=torch.float32)

    mod.run_predicated_copy(Src, Dst)

    # In-bounds elements should match source
    ok_data = torch.equal(Dst[:n], Src)
    # Out-of-bounds elements should be untouched (sentinel = -1.0)
    ok_sentinel = torch.equal(Dst[n:], torch.full((n_padded - n,), -1.0, device=device))

    if not ok_data or not ok_sentinel:
        if not ok_data:
            diff = (Dst[:n] - Src).abs()
            print(
                f"  FAIL: predicated_copy data mismatch, "
                f"max_diff={diff.max().item():.6e}, "
                f"{diff.gt(0).sum().item()}/{n} elements differ"
            )
        if not ok_sentinel:
            bad = (Dst[n:] != -1.0).sum().item()
            print(
                f"  FAIL: predicated_copy sentinel corrupted, "
                f"{bad}/{n_padded - n} sentinel elements modified"
            )
        return 1
    print(f"  PASS: predicated_copy (n={n}), bit-exact with boundary predicate")
    return 0


def test_free_func_vector_add(mod):
    """Test opus::load / opus::store free function wrappers (vector add)."""
    n = 1310720  # same as regular vector_add test
    device = torch.device("cuda")
    dtype = torch.float32

    torch.manual_seed(99)
    A = torch.randn(n, device=device, dtype=dtype)
    B = torch.randn(n, device=device, dtype=dtype)
    Result = torch.empty(n, device=device, dtype=dtype)

    mod.run_free_func_add(A, B, Result)

    Ref = A + B

    atol, rtol = 1e-5, 1e-5
    ok = torch.allclose(Result, Ref, atol=atol, rtol=rtol)
    max_diff = (Result - Ref).abs().max().item()
    if not ok:
        diff_count = (Result - Ref).abs().gt(atol + rtol * Ref.abs()).sum().item()
        print(
            f"  FAIL: free_func_vector_add max_diff={max_diff:.6e}, "
            f"{diff_count} elements outside tol"
        )
        return 1
    print(f"  PASS: free_func_vector_add (n={n}), max_diff={max_diff:.6e}")
    return 0


def test_predicated_async_load(mod):
    """Test gmem async_load_if via free function wrapper (boundary predicate)."""
    n = 1001
    BLOCK_SIZE = 256
    n_padded = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE  # 1024

    device = torch.device("cuda")

    torch.manual_seed(77)
    Src = torch.randn(n, device=device, dtype=torch.float32)
    Dst = torch.full((n_padded,), -1.0, device=device, dtype=torch.float32)

    mod.run_predicated_async_load(Src, Dst, n_padded)

    # In-bounds: should match source
    ok_data = torch.equal(Dst[:n], Src)
    # Out-of-bounds: async_load_if zero-fills smem, so dst[n:] should be 0.0
    ok_zeros = torch.equal(Dst[n:], torch.zeros(n_padded - n, device=device))

    if not ok_data or not ok_zeros:
        if not ok_data:
            diff = (Dst[:n] - Src).abs()
            print(
                f"  FAIL: predicated_async_load data mismatch, "
                f"max_diff={diff.max().item():.6e}, "
                f"{diff.gt(0).sum().item()}/{n} elements differ"
            )
        if not ok_zeros:
            bad = (Dst[n:] != 0.0).sum().item()
            print(
                f"  FAIL: predicated_async_load OOB not zeroed, "
                f"{bad}/{n_padded - n} elements non-zero"
            )
        return 1
    print(f"  PASS: predicated_async_load (n={n}), bit-exact with boundary predicate")
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
    failures += test_mfma_32x32x2_f32(mod)
    failures += test_mfma_16x16x4_f32(mod)
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
    failures += test_mxfp8_32x32x64(mod)
    failures += test_mxfp8_16x16x128(mod)
    failures += test_mxfp4_32x32x64(mod)
    failures += test_mxfp4_16x16x128(mod)
    failures += test_vector_add(mod)
    failures += test_async_load(mod)
    failures += test_dtype_convert_fp32_bf16(mod)
    failures += test_dtype_convert_fp32_fp16(mod)
    failures += test_dtype_convert_fp32_fp8(mod)
    failures += test_dtype_convert_fp32_fp4(mod)
    failures += test_predicated_copy(mod)
    failures += test_free_func_vector_add(mod)
    failures += test_predicated_async_load(mod)

    if failures:
        print(f"\n{failures} test(s) FAILED")
    else:
        print("\nAll device tests PASSED")
    return failures


if __name__ == "__main__":
    sys.exit(main())
