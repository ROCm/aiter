# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Test OPUS device kernels via a ctypes wrapper around opus_device_test.so.

The .so is built by setup.py using hipcc directly (no torch/pybind11 headers
needed at compile time). Python loads it via ctypes and calls the extern "C"
launcher functions, passing device pointers obtained from torch tensors.

Covers:
  - MFMA variants (fp32, fp16, bf16, fp8, bf8)
  - MXFP variants (fp8, fp4) -- gfx950 only
  - vector_add, async_load, tr_load_f16, dtype_convert, predicated_copy,
    free_func_add, predicated_async_load, numeric_limits, finfo, mdiv,
    workgroup_barrier
"""

import ctypes
import os
import subprocess
import sys

try:
    import torch
except ImportError as e:
    print(f"SKIP: PyTorch not available ({e})")
    sys.exit(0)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SO_NAME = "opus_device_test.so"


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _build_so():
    """Build opus_device_test.so via setup.py (hipcc, no torch/pybind11)."""
    so_path = os.path.join(_THIS_DIR, _SO_NAME)
    try:
        subprocess.run(
            [sys.executable, "setup.py"],
            cwd=_THIS_DIR,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"FAIL: Build exited with code {e.returncode}", file=sys.stderr)
        return None
    if not os.path.isfile(so_path):
        print(f"FAIL: {_SO_NAME} not found after build", file=sys.stderr)
        return None
    return so_path


# ---------------------------------------------------------------------------
# ctypes wrapper -- same interface as the old pybind11 module
# ---------------------------------------------------------------------------

_VP = ctypes.c_void_p
_I = ctypes.c_int


class OpusDeviceLib:
    """Thin ctypes wrapper that provides the same call interface as the old
    pybind11 module so test functions don't need to change."""

    def __init__(self, so_path):
        self._lib = ctypes.CDLL(so_path)

    @staticmethod
    def _ptr(tensor):
        return ctypes.c_void_p(tensor.data_ptr())

    # -- MFMA --
    def run_mfma(self, A, B, C, variant):
        fn = getattr(self._lib, f"run_mfma_{variant}")
        fn.restype = None
        fn.argtypes = [_VP, _VP, _VP, _I, _I, _I]
        fn(
            self._ptr(A),
            self._ptr(B),
            self._ptr(C),
            int(A.stride(0)),
            int(B.stride(0)),
            int(C.stride(0)),
        )

    # -- WMMA --
    def run_wmma(self, A, B, C, variant):
        fn = getattr(self._lib, f"run_wmma_{variant}")
        fn.restype = None
        fn.argtypes = [_VP, _VP, _VP, _I, _I, _I]
        fn(
            self._ptr(A),
            self._ptr(B),
            self._ptr(C),
            int(A.stride(0)),
            int(B.stride(0)),
            int(C.stride(0)),
        )

    # -- MXFP --
    def run_mxfp(self, A, B, C, variant, scale_a=127, scale_b=127):
        fn = getattr(self._lib, f"run_{variant}")
        fn.restype = None
        fn.argtypes = [_VP, _VP, _VP, _I, _I]
        fn(self._ptr(A), self._ptr(B), self._ptr(C), int(scale_a), int(scale_b))

    # -- WMMA Scale (BX32: int scale, BX16: long scale) --
    def run_wmma_scale_bx32(
        self, A, B, C, variant, scale_a=0x7F7F7F7F, scale_b=0x7F7F7F7F
    ):
        fn = getattr(self._lib, f"run_{variant}")
        fn.restype = None
        fn.argtypes = [_VP, _VP, _VP, _I, _I]
        fn(self._ptr(A), self._ptr(B), self._ptr(C), int(scale_a), int(scale_b))

    def run_wmma_scale_bx16(
        self, A, B, C, variant, scale_a=0x7F7F7F7F7F7F7F7F, scale_b=0x7F7F7F7F7F7F7F7F
    ):
        _L = ctypes.c_long
        fn = getattr(self._lib, f"run_{variant}")
        fn.restype = None
        fn.argtypes = [_VP, _VP, _VP, _L, _L]
        fn(self._ptr(A), self._ptr(B), self._ptr(C), _L(scale_a), _L(scale_b))

    def run_wmma_scale_perlane(self, A, B, C, variant, scale_a_buf, scale_b_buf):
        """Run WMMA scale with per-lane scale buffers (int32[32] each)."""
        fn = getattr(self._lib, f"run_{variant}")
        fn.restype = None
        fn.argtypes = [_VP, _VP, _VP, _VP, _VP]
        fn(
            self._ptr(A),
            self._ptr(B),
            self._ptr(C),
            self._ptr(scale_a_buf),
            self._ptr(scale_b_buf),
        )

    def run_tiled_wmma_scale(
        self, A, B, C, variant, scale_a=0x7F7F7F7F, scale_b=0x7F7F7F7F
    ):
        fn = getattr(self._lib, f"run_{variant}")
        fn.restype = None
        fn.argtypes = [_VP, _VP, _VP, _I, _I, _I, _I, _I]
        fn(
            self._ptr(A),
            self._ptr(B),
            self._ptr(C),
            int(A.stride(0)),
            int(B.stride(0)),
            int(C.stride(0)),
            int(scale_a),
            int(scale_b),
        )

    # -- vector_add --
    def run_vector_add(self, A, B, Result):
        fn = self._lib.run_vector_add
        fn.restype = None
        fn.argtypes = [_VP, _VP, _VP, _I]
        fn(self._ptr(A), self._ptr(B), self._ptr(Result), int(A.numel()))

    # -- async_load --
    def run_async_load(self, Src, Dst):
        fn = self._lib.run_async_load
        fn.restype = None
        fn.argtypes = [_VP, _VP, _I]
        fn(self._ptr(Src), self._ptr(Dst), int(Src.numel()))

    # -- tr_load_f16 --
    def run_tr_load_f16(self, Src, Dst):
        fn = self._lib.run_tr_load_f16
        fn.restype = None
        fn.argtypes = [_VP, _VP]
        fn(self._ptr(Src), self._ptr(Dst))

    # -- dtype_convert --
    def run_dtype_convert(self, In, Out, variant):
        fn = getattr(self._lib, f"run_dtype_convert_{variant}")
        fn.restype = None
        fn.argtypes = [_VP, _VP, _I]
        fn(self._ptr(In), self._ptr(Out), int(In.numel()))

    # -- predicated_copy --
    def run_predicated_copy(self, Src, Dst):
        fn = self._lib.run_predicated_copy
        fn.restype = None
        fn.argtypes = [_VP, _VP, _I]
        fn(self._ptr(Src), self._ptr(Dst), int(Src.numel()))

    # -- free_func_add --
    def run_free_func_add(self, A, B, Result):
        fn = self._lib.run_free_func_add
        fn.restype = None
        fn.argtypes = [_VP, _VP, _VP, _I]
        fn(self._ptr(A), self._ptr(B), self._ptr(Result), int(A.numel()))

    # -- predicated_async_load --
    def run_predicated_async_load(self, Src, Dst, n_padded):
        fn = self._lib.run_predicated_async_load
        fn.restype = None
        fn.argtypes = [_VP, _VP, _I, _I]
        fn(self._ptr(Src), self._ptr(Dst), int(Src.numel()), int(n_padded))

    # -- numeric_limits --
    def run_numeric_limits(self, Out):
        fn = self._lib.run_numeric_limits
        fn.restype = None
        fn.argtypes = [_VP]
        fn(self._ptr(Out))

    # -- finfo --
    def run_finfo(self, Out):
        fn = self._lib.run_finfo
        fn.restype = None
        fn.argtypes = [_VP]
        fn(self._ptr(Out))

    # -- mdiv --
    def run_mdiv(self, Dividends, OutQ, OutR, divisor):
        fn = self._lib.run_mdiv
        fn.restype = None
        fn.argtypes = [_VP, _VP, _VP, _I, _I]
        fn(
            self._ptr(Dividends),
            self._ptr(OutQ),
            self._ptr(OutR),
            int(divisor),
            int(Dividends.numel()),
        )

    # -- workgroup_barrier --
    def run_wb_cumulative(self, Accum, n_workgroups):
        fn = self._lib.run_workgroup_barrier_cumulative
        fn.restype = None
        fn.argtypes = [_VP, _I]
        fn(self._ptr(Accum), int(n_workgroups))

    def run_wb_streamk_reduce(self, Input, Workspace, Result, n_chunks):
        fn = self._lib.run_workgroup_barrier_streamk_reduce
        fn.restype = None
        fn.argtypes = [_VP, _VP, _VP, _I]
        fn(self._ptr(Input), self._ptr(Workspace), self._ptr(Result), int(n_chunks))


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
_MFMA_ALL = _MFMA_ARCHS_GFX942 | _MFMA_ARCHS_GFX942_GFX950  # all archs with MFMA
_WMMA_ARCHS_GFX1250 = {"gfx1250"}  # WMMA 16x16x32 (wave32)
_TR_LOAD_ARCHS_GFX950 = {"gfx950"}  # smem tr_load


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
    if arch in ("gfx950", "gfx1250"):
        return torch.float8_e4m3fn
    return torch.float8_e4m3fnuz  # gfx942 default


def _get_bf8_dtype():
    """Return the correct BF8 (e5m2) torch dtype for the current GPU arch."""
    arch = _get_gpu_arch()
    if arch in ("gfx950", "gfx1250"):
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
# WMMA tests (gfx1250 only)
# ---------------------------------------------------------------------------


def _test_wmma_variant(mod, variant, M, N, K, dtype, supported_archs):
    """Test a single WMMA variant. Returns 0 on pass, 1 on fail.

    WMMA kernel outputs fp32 accumulator (C = A @ B^T via swap_ab).
    """
    arch = _get_gpu_arch()
    if arch not in supported_archs:
        print(f"  SKIP: wmma_{variant} requires {supported_archs}, got '{arch}'")
        return 0

    device = torch.device("cuda")

    torch.manual_seed(12345)
    A = torch.randint(-10, 11, (M, K), device=device).to(dtype)
    B = torch.randint(-10, 11, (N, K), device=device).to(dtype)
    C = torch.empty(M, N, device=device, dtype=torch.float32)

    mod.run_wmma(A, B, C, variant)

    # swap_ab net result: C = A @ B^T
    C_ref = torch.mm(A.float(), B.float().t())

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
            f"  FAIL: wmma_{variant} max_diff={max_diff:.4f}, "
            f"{diff_count} elements outside tol"
        )
        return 1
    print(f"  PASS: wmma_{variant}, max_diff={max_diff:.4f}")
    return 0


def _test_wmma_variant_generic(
    mod, variant, M, N, K, in_dtype, out_dtype, supported_archs, atol=1e-3, rtol=1e-3
):
    """Test a WMMA variant with explicit input and output dtypes."""
    arch = _get_gpu_arch()
    if arch not in supported_archs:
        print(f"  SKIP: wmma_{variant} requires {supported_archs}, got '{arch}'")
        return 0

    device = torch.device("cuda")
    torch.manual_seed(12345)

    # For fp8 types, use small integer range to avoid overflow
    is_fp8 = in_dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    )
    if is_fp8:
        A = torch.randint(-3, 4, (M, K), device=device).to(torch.float32).to(in_dtype)
        B = torch.randint(-3, 4, (N, K), device=device).to(torch.float32).to(in_dtype)
    else:
        A = torch.randint(-10, 11, (M, K), device=device).to(in_dtype)
        B = torch.randint(-10, 11, (N, K), device=device).to(in_dtype)

    C = torch.empty(M, N, device=device, dtype=out_dtype)

    mod.run_wmma(A, B, C, variant)

    # swap_ab net result: C = A @ B^T
    C_ref = torch.mm(A.float(), B.float().t())

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
            f"  FAIL: wmma_{variant} max_diff={max_diff:.4f}, "
            f"{diff_count} elements outside tol"
        )
        return 1
    print(f"  PASS: wmma_{variant}, max_diff={max_diff:.4f}")
    return 0


def test_wmma_16x16x32_f16(mod):
    return _test_wmma_variant(
        mod, "16x16x32_f16", 16, 16, 32, torch.float16, _WMMA_ARCHS_GFX1250
    )


def test_wmma_16x16x32_bf16(mod):
    return _test_wmma_variant(
        mod, "16x16x32_bf16", 16, 16, 32, torch.bfloat16, _WMMA_ARCHS_GFX1250
    )


def test_wmma_16x16x32_f16_f16(mod):
    return _test_wmma_variant_generic(
        mod,
        "16x16x32_f16_f16",
        16,
        16,
        32,
        torch.float16,
        torch.float16,
        _WMMA_ARCHS_GFX1250,
    )


def test_wmma_16x16x32_bf16_bf16(mod):
    return _test_wmma_variant_generic(
        mod,
        "16x16x32_bf16_bf16",
        16,
        16,
        32,
        torch.bfloat16,
        torch.bfloat16,
        _WMMA_ARCHS_GFX1250,
        atol=4.0,
        rtol=1e-2,  # bf16 output has ~3 decimal digits precision
    )


def test_wmma_16x16x4_f32(mod):
    return _test_wmma_variant_generic(
        mod, "16x16x4_f32", 16, 16, 4, torch.float32, torch.float32, _WMMA_ARCHS_GFX1250
    )


def test_wmma_16x16x64_fp8_f32(mod):
    return _test_wmma_variant_generic(
        mod,
        "16x16x64_fp8_f32",
        16,
        16,
        64,
        _get_fp8_dtype(),
        torch.float32,
        _WMMA_ARCHS_GFX1250,
    )


def test_wmma_16x16x64_bf8_f32(mod):
    return _test_wmma_variant_generic(
        mod,
        "16x16x64_bf8_f32",
        16,
        16,
        64,
        _get_bf8_dtype(),
        torch.float32,
        _WMMA_ARCHS_GFX1250,
    )


def test_wmma_16x16x64_fp8_f16(mod):
    return _test_wmma_variant_generic(
        mod,
        "16x16x64_fp8_f16",
        16,
        16,
        64,
        _get_fp8_dtype(),
        torch.float16,
        _WMMA_ARCHS_GFX1250,
    )


def test_wmma_16x16x128_fp8_f32(mod):
    return _test_wmma_variant_generic(
        mod,
        "16x16x128_fp8_f32",
        16,
        16,
        128,
        _get_fp8_dtype(),
        torch.float32,
        _WMMA_ARCHS_GFX1250,
    )


def test_wmma_16x16x128_bf8_f32(mod):
    return _test_wmma_variant_generic(
        mod,
        "16x16x128_bf8_f32",
        16,
        16,
        128,
        _get_bf8_dtype(),
        torch.float32,
        _WMMA_ARCHS_GFX1250,
    )


def test_wmma_16x16x128_fp8_f16(mod):
    return _test_wmma_variant_generic(
        mod,
        "16x16x128_fp8_f16",
        16,
        16,
        128,
        _get_fp8_dtype(),
        torch.float16,
        _WMMA_ARCHS_GFX1250,
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


# ── WMMA Scale tests (gfx1250 only) ─────────────────────────────────────────

_WMMA_SCALE_ARCHS = {"gfx1250"}


def _test_wmma_scale_fp8(mod, variant, bx16=False):
    """Test WMMA scale fp8 (16x16x128 f8f6f4). Returns 0 on pass, 1 on fail."""
    arch = _get_gpu_arch()
    if arch not in _WMMA_SCALE_ARCHS:
        print(f"  SKIP: {variant} requires {_WMMA_SCALE_ARCHS}, got '{arch}'")
        return 0
    M, N, K = 16, 16, 128
    device = torch.device("cuda")
    fp8_dtype = _get_fp8_dtype()
    torch.manual_seed(12345)
    A = torch.randint(-15, 16, (M, K), device=device).float().to(fp8_dtype)
    B = torch.randint(-15, 16, (N, K), device=device).float().to(fp8_dtype)
    C = torch.empty(M, N, device=device, dtype=torch.float32)
    # scale=no-scaling (127 packed)
    if bx16:
        mod.run_wmma_scale_bx16(A, B, C, variant)
    else:
        mod.run_wmma_scale_bx32(A, B, C, variant)
    C_ref = torch.mm(A.float(), B.float().t())
    ok = torch.equal(C, C_ref)
    max_diff = (C - C_ref).abs().max().item()
    if not ok:
        diff_count = (C != C_ref).sum().item()
        print(f"  FAIL: {variant} max_diff={max_diff:.4f}, {diff_count} mismatches")
        return 1
    print(f"  PASS: {variant}, max_diff={max_diff:.4f}")
    return 0


def _test_wmma_scale_fp4(mod, variant, M, N, K, bx16=False):
    """Test WMMA scale fp4. Returns 0 on pass, 1 on fail."""
    arch = _get_gpu_arch()
    if arch not in _WMMA_SCALE_ARCHS:
        print(f"  SKIP: {variant} requires {_WMMA_SCALE_ARCHS}, got '{arch}'")
        return 0
    device = torch.device("cuda")
    torch.manual_seed(54321)
    # fp4 E2M1 representable: {0, 0.5, 1, 1.5, 2, 3, 4, 6} + negatives
    fp4_vals = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    # Generate random indices and pack as fp4x2 bytes
    a_idx = torch.randint(0, 16, (M, K), device="cpu")
    b_idx = torch.randint(0, 16, (N, K), device="cpu")
    a_vals = fp4_vals[a_idx]
    b_vals = fp4_vals[b_idx]
    # Pack fp4 nibbles into bytes
    a_packed = ((a_idx[:, 1::2] << 4) | a_idx[:, 0::2]).to(torch.uint8)
    b_packed = ((b_idx[:, 1::2] << 4) | b_idx[:, 0::2]).to(torch.uint8)
    A = a_packed.to(device)
    B = b_packed.to(device)
    C = torch.empty(M, N, device=device, dtype=torch.float32)
    if bx16:
        mod.run_wmma_scale_bx16(A, B, C, variant)
    else:
        mod.run_wmma_scale_bx32(A, B, C, variant)
    C_ref = torch.mm(a_vals.float(), b_vals.float().t()).to(device)
    ok = torch.equal(C, C_ref)
    max_diff = (C - C_ref).abs().max().item()
    if not ok:
        diff_count = (C != C_ref).sum().item()
        print(f"  FAIL: {variant} max_diff={max_diff:.4f}, {diff_count} mismatches")
        return 1
    print(f"  PASS: {variant}, max_diff={max_diff:.4f}")
    return 0


def _test_tiled_wmma_scale_fp8(mod, variant):
    """Test tiled WMMA scale fp8 via make_tiled_mma (C = A @ B^T)."""
    arch = _get_gpu_arch()
    if arch not in _WMMA_SCALE_ARCHS:
        print(f"  SKIP: {variant} requires {_WMMA_SCALE_ARCHS}, got '{arch}'")
        return 0
    M, N, K = 16, 16, 128
    device = torch.device("cuda")
    fp8_dtype = _get_fp8_dtype()
    torch.manual_seed(99999)
    A = torch.randint(-15, 16, (M, K), device=device).float().to(fp8_dtype)
    B = torch.randint(-15, 16, (N, K), device=device).float().to(fp8_dtype)
    C = torch.empty(M, N, device=device, dtype=torch.float32)
    mod.run_tiled_wmma_scale(A, B, C, variant)
    C_ref = torch.mm(A.float(), B.float().t())
    ok = torch.equal(C, C_ref)
    max_diff = (C - C_ref).abs().max().item()
    if not ok:
        diff_count = (C != C_ref).sum().item()
        print(f"  FAIL: {variant} max_diff={max_diff:.4f}, {diff_count} mismatches")
        return 1
    print(f"  PASS: {variant}, max_diff={max_diff:.4f}")
    return 0


def test_wmma_scale_16x16x128_fp8_bx32(mod):
    return _test_wmma_scale_fp8(mod, "wmma_scale_16x16x128_fp8_bx32")


def test_wmma_scale16_16x16x128_fp8_bx16(mod):
    return _test_wmma_scale_fp8(mod, "wmma_scale16_16x16x128_fp8_bx16", bx16=True)


def test_wmma_scale_16x16x128_fp4_bx32(mod):
    return _test_wmma_scale_fp4(mod, "wmma_scale_16x16x128_fp4_bx32", 16, 16, 128)


def test_wmma_scale16_16x16x128_fp4_bx16(mod):
    return _test_wmma_scale_fp4(
        mod, "wmma_scale16_16x16x128_fp4_bx16", 16, 16, 128, bx16=True
    )


def test_wmma_scale_32x16x128_fp4_bx32(mod):
    return _test_wmma_scale_fp4(mod, "wmma_scale_32x16x128_fp4_bx32", 32, 16, 128)


def test_wmma_scale16_32x16x128_fp4_bx16(mod):
    return _test_wmma_scale_fp4(
        mod, "wmma_scale16_32x16x128_fp4_bx16", 32, 16, 128, bx16=True
    )


def test_tiled_wmma_scale_16x16x128_fp8(mod):
    return _test_tiled_wmma_scale_fp8(mod, "tiled_wmma_scale_16x16x128_fp8")


def _test_tiled_wmma_scale_fp8_multi(mod, variant, M, N, K):
    """Test tiled WMMA scale fp8 with multiple waves (C = A @ B^T)."""
    arch = _get_gpu_arch()
    if arch not in _WMMA_SCALE_ARCHS:
        print(f"  SKIP: {variant} requires {_WMMA_SCALE_ARCHS}, got '{arch}'")
        return 0
    device = torch.device("cuda")
    fp8_dtype = _get_fp8_dtype()
    torch.manual_seed(11111)
    A = torch.randint(-15, 16, (M, K), device=device).float().to(fp8_dtype)
    B = torch.randint(-15, 16, (N, K), device=device).float().to(fp8_dtype)
    C = torch.empty(M, N, device=device, dtype=torch.float32)
    mod.run_tiled_wmma_scale(A, B, C, variant)
    C_ref = torch.mm(A.float(), B.float().t())
    ok = torch.equal(C, C_ref)
    max_diff = (C - C_ref).abs().max().item()
    if not ok:
        diff_count = (C != C_ref).sum().item()
        print(
            f"  FAIL: {variant} max_diff={max_diff:.4f}, {diff_count}/{M*N} mismatches"
        )
        return 1
    print(f"  PASS: {variant} ({M}x{N}x{K}), max_diff={max_diff:.4f}")
    return 0


def test_tiled_wmma_scale_fp8_2x2(mod):
    """T_M=2, T_N=2: 4 waves, 32x32 block."""
    return _test_tiled_wmma_scale_fp8_multi(
        mod, "tiled_wmma_scale_16x16x128_fp8_2x2", 32, 32, 128
    )


def test_tiled_wmma_scale_fp8_4x1(mod):
    """T_M=4, T_N=1: 4 waves, 64x16 block."""
    return _test_tiled_wmma_scale_fp8_multi(
        mod, "tiled_wmma_scale_16x16x128_fp8_4x1", 64, 16, 128
    )


# ── WMMA Scale tests with random per-row/col scale values ───────────────────


def _pack_bx32_scales(exponents):
    """Pack per-lane E8M0 exponents into BX32 int: same byte repeated 4x.
    Returns values as unsigned 32-bit integers to avoid signed overflow."""
    import struct

    result = []
    for e in exponents:
        e = int(e) & 0xFF
        u32 = e | (e << 8) | (e << 16) | (e << 24)
        # Reinterpret as signed int32 for torch
        (i32,) = struct.unpack("i", struct.pack("I", u32))
        result.append(i32)
    return result


def _test_wmma_scale_fp8_with_scaling(mod, variant, bx16=False):
    """Test WMMA scale fp8 with random per-row/col E8M0 scale values."""
    arch = _get_gpu_arch()
    if arch not in _WMMA_SCALE_ARCHS:
        print(f"  SKIP: {variant} requires {_WMMA_SCALE_ARCHS}, got '{arch}'")
        return 0
    M, N, K = 16, 16, 128
    device = torch.device("cuda")
    fp8_dtype = _get_fp8_dtype()
    torch.manual_seed(77777)

    A = torch.randint(-15, 16, (M, K), device=device).float().to(fp8_dtype)
    B = torch.randint(-15, 16, (N, K), device=device).float().to(fp8_dtype)
    C = torch.empty(M, N, device=device, dtype=torch.float32)

    # Random per-row/col scale exponents in [122..133] (2^-5 to 2^6)
    sa_exps = torch.randint(122, 134, (M,))  # per m-row
    sb_exps = torch.randint(122, 134, (N,))  # per n-col

    # Build per-lane scale arrays (32 lanes: lanes 0-15 = data, 16-31 = k-group1)
    # Scale comes from lanes 0-15 (scale_sel=0), so lanes 16-31 don't matter.
    sa_per_lane = _pack_bx32_scales([sa_exps[lane % 16] for lane in range(32)])
    sb_per_lane = _pack_bx32_scales([sb_exps[lane % 16] for lane in range(32)])

    sa_buf = torch.tensor(sa_per_lane, dtype=torch.int32, device=device)
    sb_buf = torch.tensor(sb_per_lane, dtype=torch.int32, device=device)

    mod.run_wmma_scale_perlane(A, B, C, variant, sa_buf, sb_buf)

    # Reference: C[m][n] = 2^(sa[m]-127) * 2^(sb[n]-127) * dot(A[m,:], B[n,:])
    dot = torch.mm(A.float(), B.float().t())  # on device
    sa_scale = (2.0 ** (sa_exps.float() - 127)).unsqueeze(1).to(device)  # [M,1]
    sb_scale = (2.0 ** (sb_exps.float() - 127)).unsqueeze(0).to(device)  # [1,N]
    C_ref = dot * sa_scale * sb_scale

    ok = torch.equal(C, C_ref)
    max_diff = (C - C_ref).abs().max().item()
    if not ok:
        diff_count = (C != C_ref).sum().item()
        print(f"  FAIL: {variant} max_diff={max_diff:.4f}, {diff_count} mismatches")
        return 1
    print(f"  PASS: {variant}, max_diff={max_diff:.4f}")
    return 0


def test_wmma_scale_16x16x128_fp8_bx32_scaled(mod):
    return _test_wmma_scale_fp8_with_scaling(
        mod, "wmma_scale_16x16x128_fp8_bx32_perlane"
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


def test_tr_load_f16(mod):
    """16x32 row-major fp16 -> LDS -> tr_load -> MFMA B layout store; expect 32x16 = input.T."""
    arch = _get_gpu_arch()
    if arch not in _TR_LOAD_ARCHS_GFX950:
        print(f"  SKIP: tr_load_f16 requires {_TR_LOAD_ARCHS_GFX950}, got '{arch}'")
        return 0

    device = torch.device("cuda")
    torch.manual_seed(42)
    inp = torch.randn(16, 32, device=device, dtype=torch.float16)
    out = torch.empty(32, 16, device=device, dtype=torch.float16)
    ref = inp.t().contiguous()

    mod.run_tr_load_f16(inp, out)

    ok = torch.equal(out, ref)
    max_diff = (out.float() - ref.float()).abs().max().item()
    if not ok:
        diff_count = (out != ref).sum().item()
        print(f"  FAIL: tr_load_f16 max_diff={max_diff:.4f}, {diff_count} mismatches")
        return 1
    print(f"  PASS: tr_load_f16 16x32 -> 32x16, max_diff={max_diff:.4f}")
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


def test_dtype_convert_fp32_bf16_vec4(mod):
    """Test vectorized FP32x4 -> BF16x4 -> FP32x4 round-trip via opus::cast<bf16_t>(fp32x4_t).

    Exercises the generic vectorized cast() entry point (element-wise bf16 conversion
    applied to a 4-wide vector), unlike the scalar test which converts one element at a time.
    """
    n = 1048576
    device = torch.device("cuda")

    torch.manual_seed(210)
    In = torch.randn(n, device=device, dtype=torch.float32)
    Out = torch.empty(n, device=device, dtype=torch.float32)

    mod.run_dtype_convert(In, Out, "fp32_bf16_vec4")

    Ref = In.to(torch.bfloat16).to(torch.float32)

    ok = torch.equal(Out, Ref)
    if not ok:
        diff = (Out - Ref).abs()
        max_diff = diff.max().item()
        diff_count = diff.gt(0).sum().item()
        print(
            f"  FAIL: dtype_convert fp32<->bf16 vec4 max_diff={max_diff:.6e}, "
            f"{diff_count}/{n} elements differ"
        )
        return 1
    print(f"  PASS: dtype_convert fp32<->bf16 vec4 (n={n}), bit-exact")
    return 0


def test_dtype_convert_fp32_fp16_vec4(mod):
    """Test vectorized FP32x4 -> FP16x4 -> FP32x4 round-trip via opus::cast<fp16_t>(fp32x4_t).

    Exercises the generic vectorized cast() entry point for fp16.
    """
    n = 1048576
    device = torch.device("cuda")

    torch.manual_seed(211)
    In = torch.randn(n, device=device, dtype=torch.float32)
    Out = torch.empty(n, device=device, dtype=torch.float32)

    mod.run_dtype_convert(In, Out, "fp32_fp16_vec4")

    Ref = In.to(torch.float16).to(torch.float32)

    atol, rtol = 1e-4, 1e-4
    ok = torch.allclose(Out, Ref, atol=atol, rtol=rtol)
    max_diff = (Out - Ref).abs().max().item()
    if not ok:
        diff_count = (Out - Ref).abs().gt(atol + rtol * Ref.abs()).sum().item()
        print(
            f"  FAIL: dtype_convert fp32<->fp16 vec4 max_diff={max_diff:.6e}, "
            f"{diff_count}/{n} elements outside tol"
        )
        return 1
    print(f"  PASS: dtype_convert fp32<->fp16 vec4 (n={n}), max_diff={max_diff:.6e}")
    return 0


_FP8_SUPPORTED_ARCHS = {"gfx942", "gfx950", "gfx1250"}
_FP4_SUPPORTED_ARCHS = {"gfx950", "gfx1250"}


def test_dtype_convert_fp32_fp8_scalar(mod):
    """Test scalar FP32 -> FP8 (e4m3) -> FP32 round-trip via opus::cast (one element per thread)."""
    arch = _get_gpu_arch()
    if arch not in _FP8_SUPPORTED_ARCHS:
        print(
            f"  SKIP: dtype_convert fp32<->fp8 scalar requires {_FP8_SUPPORTED_ARCHS}, got '{arch}'"
        )
        return 0

    n = 1048576
    device = torch.device("cuda")

    torch.manual_seed(220)
    In = torch.randn(n, device=device, dtype=torch.float32) * 2.0
    Out = torch.empty(n, device=device, dtype=torch.float32)

    mod.run_dtype_convert(In, Out, "fp32_fp8_scalar")

    fp8_dtype = _get_fp8_dtype()
    Ref = In.to(fp8_dtype).to(torch.float32)

    atol, rtol = 0.5, 0.25
    ok = torch.allclose(Out, Ref, atol=atol, rtol=rtol)
    max_diff = (Out - Ref).abs().max().item()
    if not ok:
        diff_count = (Out - Ref).abs().gt(atol + rtol * Ref.abs()).sum().item()
        print(
            f"  FAIL: dtype_convert fp32<->fp8 scalar max_diff={max_diff:.6e}, "
            f"{diff_count}/{n} elements outside tol"
        )
        return 1
    print(f"  PASS: dtype_convert fp32<->fp8 scalar (n={n}), max_diff={max_diff:.6e}")
    return 0


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


def test_dtype_convert_fp32_fp8_x2(mod):
    """Test FP32x2 -> FP8x2 -> FP32x2 round-trip via opus::cast packed x2."""
    arch = _get_gpu_arch()
    if arch not in _FP8_SUPPORTED_ARCHS:
        print(
            f"  SKIP: dtype_convert fp32<->fp8 x2 requires {_FP8_SUPPORTED_ARCHS}, got '{arch}'"
        )
        return 0

    n = 1048576
    device = torch.device("cuda")

    torch.manual_seed(212)
    In = torch.randn(n, device=device, dtype=torch.float32) * 2.0
    Out = torch.empty(n, device=device, dtype=torch.float32)

    mod.run_dtype_convert(In, Out, "fp32_fp8_x2")

    fp8_dtype = _get_fp8_dtype()
    Ref = In.to(fp8_dtype).to(torch.float32)

    atol, rtol = 0.5, 0.25
    ok = torch.allclose(Out, Ref, atol=atol, rtol=rtol)
    max_diff = (Out - Ref).abs().max().item()
    if not ok:
        diff_count = (Out - Ref).abs().gt(atol + rtol * Ref.abs()).sum().item()
        print(
            f"  FAIL: dtype_convert fp32<->fp8 x2 max_diff={max_diff:.6e}, "
            f"{diff_count}/{n} elements outside tol"
        )
        return 1
    print(f"  PASS: dtype_convert fp32<->fp8 x2 (n={n}), max_diff={max_diff:.6e}")
    return 0


def test_dtype_convert_fp32_fp8_vec8(mod):
    """Test FP32x8 -> FP8(auto-fold 2x x4) -> FP32x8 round-trip.

    Exercises the generic vectorized cast() auto-folding path for fp8 with 8-wide input.
    """
    arch = _get_gpu_arch()
    if arch not in _FP8_SUPPORTED_ARCHS:
        print(
            f"  SKIP: dtype_convert fp32<->fp8 vec8 requires {_FP8_SUPPORTED_ARCHS}, got '{arch}'"
        )
        return 0

    n = 1048576
    device = torch.device("cuda")

    torch.manual_seed(213)
    In = torch.randn(n, device=device, dtype=torch.float32) * 2.0
    Out = torch.empty(n, device=device, dtype=torch.float32)

    mod.run_dtype_convert(In, Out, "fp32_fp8_vec8")

    fp8_dtype = _get_fp8_dtype()
    Ref = In.to(fp8_dtype).to(torch.float32)

    atol, rtol = 0.5, 0.25
    ok = torch.allclose(Out, Ref, atol=atol, rtol=rtol)
    max_diff = (Out - Ref).abs().max().item()
    if not ok:
        diff_count = (Out - Ref).abs().gt(atol + rtol * Ref.abs()).sum().item()
        print(
            f"  FAIL: dtype_convert fp32<->fp8 vec8 max_diff={max_diff:.6e}, "
            f"{diff_count}/{n} elements outside tol"
        )
        return 1
    print(f"  PASS: dtype_convert fp32<->fp8 vec8 (n={n}), max_diff={max_diff:.6e}")
    return 0


def test_dtype_convert_fp32_fp4_x2(mod):
    """Test FP32x2 -> FP4(x2) -> FP32x2 round-trip via opus::cast packed x2 (gfx950 only)."""
    arch = _get_gpu_arch()
    if arch not in _FP4_SUPPORTED_ARCHS:
        print(
            f"  SKIP: dtype_convert fp32<->fp4 x2 requires {_FP4_SUPPORTED_ARCHS}, got '{arch}'"
        )
        return 0

    n = 1048576
    device = torch.device("cuda")
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
    torch.manual_seed(214)
    indices = torch.randint(0, len(fp4_values), (n,))
    In = fp4_values[indices].to(device=device)
    Out = torch.empty(n, device=device, dtype=torch.float32)

    mod.run_dtype_convert(In, Out, "fp32_fp4_x2")

    Ref = In
    ok = torch.equal(Out, Ref)
    if not ok:
        diff = (Out - Ref).abs()
        max_diff = diff.max().item()
        diff_count = diff.gt(0).sum().item()
        print(
            f"  FAIL: dtype_convert fp32<->fp4 x2 max_diff={max_diff:.6e}, "
            f"{diff_count}/{n} elements differ"
        )
        return 1
    print(f"  PASS: dtype_convert fp32<->fp4 x2 (n={n}), bit-exact")
    return 0


def test_dtype_convert_fp32_fp4_x4(mod):
    """Test FP32x4 -> FP4(x4) -> FP32x4 round-trip via opus::cast packed x4 (gfx950 only)."""
    arch = _get_gpu_arch()
    if arch not in _FP4_SUPPORTED_ARCHS:
        print(
            f"  SKIP: dtype_convert fp32<->fp4 x4 requires {_FP4_SUPPORTED_ARCHS}, got '{arch}'"
        )
        return 0

    n = 1048576
    device = torch.device("cuda")
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
    torch.manual_seed(215)
    indices = torch.randint(0, len(fp4_values), (n,))
    In = fp4_values[indices].to(device=device)
    Out = torch.empty(n, device=device, dtype=torch.float32)

    mod.run_dtype_convert(In, Out, "fp32_fp4_x4")

    Ref = In
    ok = torch.equal(Out, Ref)
    if not ok:
        diff = (Out - Ref).abs()
        max_diff = diff.max().item()
        diff_count = diff.gt(0).sum().item()
        print(
            f"  FAIL: dtype_convert fp32<->fp4 x4 max_diff={max_diff:.6e}, "
            f"{diff_count}/{n} elements differ"
        )
        return 1
    print(f"  PASS: dtype_convert fp32<->fp4 x4 (n={n}), bit-exact")
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
# mdiv tests
# ---------------------------------------------------------------------------


def test_mdiv(mod):
    """Test opus::mdiv magic division against native integer division."""
    device = torch.device("cuda")

    divisors = [1, 2, 3, 7, 13, 32, 64, 127, 255, 1024, 65537]
    n = 100000

    torch.manual_seed(99)
    # Unsigned 32-bit range: use int32 tensor with values in [0, 2^31-1)
    dividends = torch.randint(0, 2**31 - 1, (n,), device=device, dtype=torch.int32)

    fails = 0
    for d in divisors:
        out_q = torch.empty(n, device=device, dtype=torch.int32)
        out_r = torch.empty(n, device=device, dtype=torch.int32)
        mod.run_mdiv(dividends, out_q, out_r, d)

        # Reference: unsigned division via Python
        vals = dividends.to(torch.int64)  # avoid overflow
        ref_q = (vals // d).to(torch.int32)
        ref_r = (vals % d).to(torch.int32)

        q_ok = torch.equal(out_q, ref_q)
        r_ok = torch.equal(out_r, ref_r)
        if not q_ok or not r_ok:
            q_bad = (out_q != ref_q).sum().item()
            r_bad = (out_r != ref_r).sum().item()
            print(f"  FAIL: mdiv divisor={d}, q_mismatch={q_bad}, r_mismatch={r_bad}")
            fails += 1
        else:
            print(f"  PASS: mdiv divisor={d}")

    if fails:
        print(f"  mdiv: {fails}/{len(divisors)} divisors FAILED")
        return 1
    print(f"  PASS: mdiv all {len(divisors)} divisors correct")
    return 0


def test_numeric_limits(mod):
    """Test opus::numeric_limits against torch reference values (bitwise comparison)."""
    import struct

    arch = _get_gpu_arch()
    device = torch.device("cuda")

    N = 55  # 11 types * 5 fields
    out = torch.zeros(N, device=device, dtype=torch.int32)
    mod.run_numeric_limits(out)
    raw = [(out[i].item()) & 0xFFFFFFFF for i in range(N)]

    fails = 0
    fields = ["min", "max", "lowest", "quiet_nan", "infinity"]

    def float_to_u32(f):
        return struct.unpack("I", struct.pack("f", float(f)))[0]

    def tensor_to_bits(t, size):
        if size == 4:
            return t.reshape(1).view(torch.int32).item() & 0xFFFFFFFF
        elif size == 2:
            return t.reshape(1).view(torch.int16).item() & 0xFFFF
        else:
            return t.reshape(1).view(torch.uint8).item()

    def ref_float(dtype, size, has_inf):
        fi = torch.finfo(dtype)
        ref = {}
        for field in fields:
            if field == "min":
                ref[field] = tensor_to_bits(torch.tensor(fi.tiny, dtype=dtype), size)
            elif field == "max":
                ref[field] = tensor_to_bits(torch.tensor(fi.max, dtype=dtype), size)
            elif field == "lowest":
                ref[field] = tensor_to_bits(torch.tensor(fi.min, dtype=dtype), size)
            elif field == "quiet_nan":
                ref[field] = tensor_to_bits(
                    torch.tensor(float("nan"), dtype=dtype), size
                )
            elif field == "infinity":
                if not has_inf:
                    ref[field] = 0
                    continue
                ref[field] = tensor_to_bits(
                    torch.tensor(float("inf"), dtype=dtype), size
                )
        return ref

    def ref_int(dtype, size):
        ii = torch.iinfo(dtype)
        mask = (1 << (size * 8)) - 1
        return {
            "min": ii.min & mask,
            "max": ii.max & mask,
            "lowest": ii.min & mask,
            "quiet_nan": 0,
            "infinity": 0,
        }

    fp8_dtype = _get_fp8_dtype()
    bf8_dtype = _get_bf8_dtype()

    # (name, offset, byte_size, is_float, torch_dtype, has_infinity)
    type_table = [
        ("fp32", 0, 4, True, torch.float32, True),
        ("fp16", 5, 2, True, torch.float16, True),
        ("bf16", 10, 2, True, torch.bfloat16, True),
        ("fp8", 15, 1, True, fp8_dtype, False),
        ("bf8", 20, 1, True, bf8_dtype, arch in ("gfx950", "gfx1250")),
        ("i32", 25, 4, False, torch.int32, False),
        ("i16", 35, 2, False, torch.int16, False),
        ("i8", 45, 1, False, torch.int8, False),
        ("u8", 50, 1, False, torch.uint8, False),
    ]

    for name, offset, size, is_float, dtype, has_inf in type_table:
        if is_float:
            ref = ref_float(dtype, size, has_inf)
        else:
            ref = ref_int(dtype, size)
        mask = (1 << (size * 8)) - 1
        width = size * 2
        type_fails = 0
        for j, field in enumerate(fields):
            actual = raw[offset + j] & mask
            expected = ref[field]
            if actual != expected:
                print(
                    f"    {name}.{field}: 0x{actual:0{width}X} "
                    f"!= expected 0x{expected:0{width}X}"
                )
                type_fails += 1
        if type_fails == 0:
            print(f"  PASS: numeric_limits<{name}> (all {len(fields)} fields)")
        else:
            print(f"  FAIL: numeric_limits<{name}> ({type_fails} field(s) wrong)")
            fails += type_fails

    if fails:
        print(f"  numeric_limits: {fails} field(s) FAILED")
        return 1
    print("  PASS: numeric_limits all types correct")
    return 0


def test_finfo(mod):
    """Test opus::finfo against torch.finfo reference values (bitwise comparison)."""
    import struct

    device = torch.device("cuda")

    N_TYPES = 7  # fp32, fp16, bf16, fp8, bf8, fp4, e8m0
    FIELDS_PER_TYPE = 5  # eps, max, min, tiny, bits
    N = N_TYPES * FIELDS_PER_TYPE
    out = torch.zeros(N, device=device, dtype=torch.float32)
    mod.run_finfo(out)
    raw = out.cpu()

    fails = 0
    fields = ["eps", "max", "min", "tiny", "bits"]

    def float_to_u32(f):
        return struct.unpack("I", struct.pack("f", float(f)))[0]

    def u32_to_float(u):
        return struct.unpack("f", struct.pack("I", u))[0]

    def ref_from_torch_finfo(dtype):
        fi = torch.finfo(dtype)
        return {
            "eps": fi.eps,
            "max": fi.max,
            "min": fi.min,
            "tiny": fi.tiny,
        }

    fp8_dtype = _get_fp8_dtype()
    bf8_dtype = _get_bf8_dtype()

    # (name, offset, torch_dtype_or_None, manual_ref_or_None)
    # For fp4 and e8m0 there is no torch.finfo, so we provide manual reference.
    fp4_ref = {"eps": 0.5, "max": 6.0, "min": -6.0, "tiny": 1.0, "bits": 4}
    e8m0_ref = {
        "eps": 1.0,
        "max": 2.0**127,
        "min": 2.0**-127,
        "tiny": 2.0**-127,
        "bits": 8,
    }

    type_table = [
        ("fp32", 0, torch.float32, 32, None),
        ("fp16", 5, torch.float16, 16, None),
        ("bf16", 10, torch.bfloat16, 16, None),
        ("fp8", 15, fp8_dtype, 8, None),
        ("bf8", 20, bf8_dtype, 8, None),
        ("fp4", 25, None, 4, fp4_ref),
        ("e8m0", 30, None, 8, e8m0_ref),
    ]

    for name, offset, dtype, expected_bits, manual_ref in type_table:
        if dtype is not None:
            ref = ref_from_torch_finfo(dtype)
            ref["bits"] = expected_bits
        else:
            ref = manual_ref

        type_fails = 0
        for j, field in enumerate(fields):
            actual_f32 = raw[offset + j].item()
            if field == "bits":
                # bits is stored as __int_as_float(bits), extract the int
                actual_val = struct.unpack("I", struct.pack("f", actual_f32))[0]
                expected_val = ref["bits"]
                if actual_val != expected_val:
                    print(
                        f"    {name}.{field}: {actual_val} != expected {expected_val}"
                    )
                    type_fails += 1
            else:
                expected_f32 = float(ref[field])
                actual_bits = float_to_u32(actual_f32)
                expected_bits_val = float_to_u32(expected_f32)
                if actual_bits != expected_bits_val:
                    print(
                        f"    {name}.{field}: 0x{actual_bits:08X} ({actual_f32}) "
                        f"!= expected 0x{expected_bits_val:08X} ({expected_f32})"
                    )
                    type_fails += 1
        if type_fails == 0:
            print(f"  PASS: finfo<{name}> (all {len(fields)} fields)")
        else:
            print(f"  FAIL: finfo<{name}> ({type_fails} field(s) wrong)")
            fails += type_fails

    if fails:
        print(f"  finfo: {fails} field(s) FAILED")
        return 1
    print("  PASS: finfo all types correct")
    return 0


def test_wb_cumulative(mod):
    """Test workgroup_barrier wait_lt + inc: N workgroups contribute i+1 sequentially."""
    device = torch.device("cuda")
    n_workgroups = 128
    accum = torch.zeros(1, device=device, dtype=torch.int32)
    mod.run_wb_cumulative(accum, n_workgroups)
    expected = n_workgroups * (n_workgroups + 1) // 2
    actual = accum.item()
    if actual != expected:
        print(f"  FAIL: wb_cumulative expected={expected}, got={actual}")
        return 1
    print(f"  PASS: wb_cumulative (n={n_workgroups}, accum={actual})")
    return 0


def test_wb_streamk_reduce(mod):
    """Test workgroup_barrier stream-K reduce: N producers + 1 consumer sum an array."""
    device = torch.device("cuda")
    for n_chunks in [1, 4, 16, 64]:
        n_elems = 256 * n_chunks
        torch.manual_seed(42)
        inp = torch.randn(n_elems, device=device, dtype=torch.float32)
        workspace = torch.empty(n_chunks, device=device, dtype=torch.float32)
        result = torch.empty(1, device=device, dtype=torch.float32)
        mod.run_wb_streamk_reduce(inp, workspace, result, n_chunks)
        expected = inp.sum().item()
        actual = result.item()
        rtol = 1e-4
        if abs(expected) > 0:
            rel_err = abs(actual - expected) / abs(expected)
        else:
            rel_err = abs(actual - expected)
        if rel_err > rtol:
            print(
                f"  FAIL: wb_streamk_reduce n_chunks={n_chunks}, "
                f"expected={expected:.6f}, got={actual:.6f}, rel_err={rel_err:.2e}"
            )
            return 1
        print(f"  PASS: wb_streamk_reduce n_chunks={n_chunks} (rel_err={rel_err:.2e})")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0

    arch = _get_gpu_arch()
    print(f"GPU arch: {arch}")
    print(f"Building {_SO_NAME} ...")
    so_path = _build_so()
    if so_path is None:
        return 1

    mod = OpusDeviceLib(so_path)

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
    failures += test_wmma_16x16x32_f16(mod)
    failures += test_wmma_16x16x32_bf16(mod)
    failures += test_wmma_16x16x32_f16_f16(mod)
    failures += test_wmma_16x16x32_bf16_bf16(mod)
    failures += test_wmma_16x16x4_f32(mod)
    failures += test_wmma_16x16x64_fp8_f32(mod)
    failures += test_wmma_16x16x64_bf8_f32(mod)
    failures += test_wmma_16x16x64_fp8_f16(mod)
    failures += test_wmma_16x16x128_fp8_f32(mod)
    failures += test_wmma_16x16x128_bf8_f32(mod)
    failures += test_wmma_16x16x128_fp8_f16(mod)
    failures += test_mxfp8_32x32x64(mod)
    failures += test_mxfp8_16x16x128(mod)
    failures += test_mxfp4_32x32x64(mod)
    failures += test_mxfp4_16x16x128(mod)
    failures += test_wmma_scale_16x16x128_fp8_bx32(mod)
    failures += test_wmma_scale16_16x16x128_fp8_bx16(mod)
    failures += test_wmma_scale_16x16x128_fp4_bx32(mod)
    failures += test_wmma_scale16_16x16x128_fp4_bx16(mod)
    failures += test_wmma_scale_32x16x128_fp4_bx32(mod)
    failures += test_wmma_scale16_32x16x128_fp4_bx16(mod)
    failures += test_tiled_wmma_scale_16x16x128_fp8(mod)
    failures += test_tiled_wmma_scale_fp8_2x2(mod)
    failures += test_tiled_wmma_scale_fp8_4x1(mod)
    failures += test_wmma_scale_16x16x128_fp8_bx32_scaled(mod)
    failures += test_vector_add(mod)
    failures += test_async_load(mod)
    failures += test_tr_load_f16(mod)
    failures += test_dtype_convert_fp32_bf16(mod)
    failures += test_dtype_convert_fp32_fp16(mod)
    failures += test_dtype_convert_fp32_bf16_vec4(mod)
    failures += test_dtype_convert_fp32_fp16_vec4(mod)
    failures += test_dtype_convert_fp32_fp8_scalar(mod)
    failures += test_dtype_convert_fp32_fp8(mod)
    failures += test_dtype_convert_fp32_fp8_x2(mod)
    failures += test_dtype_convert_fp32_fp8_vec8(mod)
    failures += test_dtype_convert_fp32_fp4(mod)
    failures += test_dtype_convert_fp32_fp4_x2(mod)
    failures += test_dtype_convert_fp32_fp4_x4(mod)
    failures += test_predicated_copy(mod)
    failures += test_free_func_vector_add(mod)
    failures += test_predicated_async_load(mod)
    failures += test_numeric_limits(mod)
    failures += test_finfo(mod)
    failures += test_mdiv(mod)
    failures += test_wb_cumulative(mod)
    failures += test_wb_streamk_reduce(mod)

    if failures:
        print(f"\n{failures} test(s) FAILED")
    else:
        print("\nAll device tests PASSED")
    return failures


if __name__ == "__main__":
    sys.exit(main())
