# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Precision + perf test for the FlyDSL ``dynamic_per_tensor_quant`` kernel.

Goals
-----
1. Verify FlyDSL output is bit-comparable (within tolerance) to the torch
   reference and the existing aiter HIP/Triton paths.
2. Compare runtime of FlyDSL vs HIP and Triton across a representative grid
   of (m, n, dtype) shapes (same defaults as ``op_tests/test_quant.py``).

Run inside the docker container (where ``flydsl`` is installed):
    python aiter/ops/flydsl/test_flydsl_dynamic_quant.py            # quick smoke + bench
    python aiter/ops/flydsl/test_flydsl_dynamic_quant.py --m 1 2 16 128 1024 16384 \
                                                          --n 4096 8192          # full sweep
    pytest -q aiter/ops/flydsl/test_flydsl_dynamic_quant.py         # CI-style precision only

Notes
-----
* The FlyDSL kernel only supports fp8 (E4M3) output today; we always use
  ``dtypes.fp8`` (which is e4m3fnuz on MI300, e4m3fn on MI350).
* Tolerance is loose because fp8 quant has ~7-bit precision; what we really
  care about is that the FlyDSL/HIP/Triton paths match each other.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from typing import Callable, Dict, Tuple

import pytest
import torch

# Skip the whole module when prerequisites aren't there. We do this BEFORE any
# heavy imports so pytest collection on a CPU box doesn't blow up.
if not torch.cuda.is_available():
    pytest.skip("ROCm/CUDA not available; skipping.", allow_module_level=True)

from aiter.ops.flydsl.utils import is_flydsl_available  # noqa: E402

if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed; skipping FlyDSL dynamic_per_tensor_quant tests.",
        allow_module_level=True,
    )

import aiter  # noqa: E402
from aiter import dtypes  # noqa: E402
from aiter import get_hip_quant, get_torch_quant, get_triton_quant  # noqa: E402
from aiter.ops.flydsl.quant_kernels import (  # noqa: E402
    flydsl_dynamic_per_tensor_quant,
    flydsl_per_1x32_fp4_quant,
    flydsl_per_1x32_fp4_quant_hadamard,
    flydsl_per_1x32_fp4_quant_block_rotation,
    flydsl_per_1x32_fp4_quant_block_rotation_mfma,
)
from aiter.test_common import checkAllclose, run_perftest  # noqa: E402
from aiter.utility import fp4_utils  # noqa: E402

torch.set_default_device("cuda")


def _flydsl_per_tensor_quant(
    x: torch.Tensor,
    scale: torch.Tensor | None = None,
    quant_dtype: torch.dtype = dtypes.fp8,
):
    """Adapter that mirrors the signature used by ``test_quant.py`` baselines."""
    assert scale is None, "FlyDSL kernel only supports dynamic quant for now"
    assert quant_dtype == dtypes.fp8, "FlyDSL kernel only supports fp8 output"
    y = torch.empty(x.shape, dtype=quant_dtype, device=x.device)
    s = torch.empty(1, dtype=dtypes.fp32, device=x.device)
    flydsl_dynamic_per_tensor_quant(y, x, s)
    return y, s.view(1)


def _flydsl_per_1x32_fp4_quant(
    x: torch.Tensor,
    scale: torch.Tensor | None = None,
    quant_dtype: torch.dtype = dtypes.fp4x2,
    shuffle: bool = False,
):
    """Adapter matching ``per_1x32_f4_quant_{hip,triton}`` signature."""
    assert scale is None, "FlyDSL kernel only supports dynamic quant"
    assert quant_dtype == dtypes.fp4x2, "FlyDSL kernel only supports fp4x2 output"
    assert not shuffle, "FlyDSL kernel does not support shuffle layout yet"
    shape = x.shape
    x_view = x.reshape(-1, shape[-1])
    m, n = x_view.shape
    y = torch.empty(m, n // 2, dtype=torch.uint8, device=x.device)
    s = torch.empty(m, n // 32, dtype=torch.uint8, device=x.device)
    flydsl_per_1x32_fp4_quant(y, x_view, s)
    out_shape = (*shape[:-1], n // 2)
    return y.view(*out_shape).view(dtypes.fp4x2), s.view(dtypes.fp8_e8m0)


def _flydsl_per_1x32_fp4_quant_hadamard(
    x: torch.Tensor,
    scale: torch.Tensor | None = None,
    quant_dtype: torch.dtype = dtypes.fp4x2,
    shuffle: bool = False,
):
    """Adapter for the H_32-rotated MXFP4 kernel.

    Same call shape as ``_flydsl_per_1x32_fp4_quant`` -- the rotation happens
    inside the kernel, so the inputs/outputs match the no-rotation API.
    """
    assert scale is None, "FlyDSL kernel only supports dynamic quant"
    assert quant_dtype == dtypes.fp4x2, "FlyDSL kernel only supports fp4x2 output"
    assert not shuffle, "FlyDSL kernel does not support shuffle layout yet"
    shape = x.shape
    x_view = x.reshape(-1, shape[-1])
    m, n = x_view.shape
    y = torch.empty(m, n // 2, dtype=torch.uint8, device=x.device)
    s = torch.empty(m, n // 32, dtype=torch.uint8, device=x.device)
    flydsl_per_1x32_fp4_quant_hadamard(y, x_view, s)
    out_shape = (*shape[:-1], n // 2)
    return y.view(*out_shape).view(dtypes.fp4x2), s.view(dtypes.fp8_e8m0)


# -----------------------------------------------------------------------------
# Precision check (pytest-friendly).
# -----------------------------------------------------------------------------
_DEFAULT_PRECISION_CASES = [
    # (m, n, dtype)
    (1, 4096, torch.bfloat16),
    (1, 7168, torch.bfloat16),
    (128, 4096, torch.bfloat16),
    (1024, 4096, torch.bfloat16),
    (1024, 7168, torch.bfloat16),
    (16384, 4096, torch.bfloat16),
    (128, 4096, torch.float16),
    (1024, 7168, torch.float16),
]


@pytest.mark.parametrize("m,n,dtype", _DEFAULT_PRECISION_CASES)
def test_flydsl_per_tensor_quant_precision(m: int, n: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x = torch.randn((m, n), dtype=dtype) * 5.0  # widen range so scale isn't trivial

    # Reference: torch per_tensor_quant.
    ref_y, ref_scale = get_torch_quant(aiter.QuantType.per_Tensor)(
        x, quant_dtype=dtypes.fp8
    )

    fd_y, fd_scale = _flydsl_per_tensor_quant(x, quant_dtype=dtypes.fp8)

    # FlyDSL produces dequant scale = max(|x|)/448 (same as torch ref).
    checkAllclose(
        ref_scale.to(dtypes.fp32),
        fd_scale.to(dtypes.fp32),
        rtol=1e-3,
        atol=1e-3,
        msg=f"flydsl scale m={m} n={n} dt={dtype}",
    )
    checkAllclose(
        ref_y.to(dtypes.fp32),
        fd_y.to(dtypes.fp32),
        rtol=1e-3,
        atol=1e-3,
        msg=f"flydsl quant m={m} n={n} dt={dtype}",
    )


# -----------------------------------------------------------------------------
# MXFP4 (per_1x32) precision check (pytest-friendly).
# -----------------------------------------------------------------------------
_FP4_PRECISION_CASES = [
    # (m, n, dtype) -- n must be a multiple of 32
    (1, 4096, torch.bfloat16),
    (1, 7168, torch.bfloat16),
    (16, 4096, torch.bfloat16),
    (128, 4096, torch.bfloat16),
    (1024, 4096, torch.bfloat16),
    (1024, 7168, torch.bfloat16),
    (16384, 4096, torch.bfloat16),
    (128, 4096, torch.float16),
    (1024, 7168, torch.float16),
]


def _dequant_mxfp4(y_packed: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Dequantize an MXFP4 (uint8 packed) tensor + E8M0 scale into f32.

    ``y_packed`` is ``(*, n // 2)`` of uint8 (low nibble = element 2k,
    high nibble = element 2k+1). ``scale_e8m0`` is ``(*, n // 32)`` of uint8
    storing the biased exponent (E8M0).
    """
    # E2M1 nibble -> f32 LUT.
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=y_packed.device,
    )
    y_bytes = y_packed.view(torch.uint8).to(torch.int64)
    low = y_bytes & 0xF
    high = (y_bytes >> 4) & 0xF
    nibbles = torch.stack([low, high], dim=-1).reshape(*y_bytes.shape[:-1], -1)
    fp4_vals = lut[nibbles]
    # E8M0 -> f32 scale: 2^(byte - 127); guard for 0.
    s_bytes = scale_e8m0.view(torch.uint8).to(torch.int32)
    s_f32 = torch.pow(
        torch.tensor(2.0, device=s_bytes.device),
        (s_bytes - 127).to(torch.float32),
    )
    # Broadcast scale[..., g] to fp4_vals[..., g * 32 : (g+1) * 32].
    fp4_grouped = fp4_vals.reshape(*fp4_vals.shape[:-1], -1, 32)
    return (fp4_grouped * s_f32.unsqueeze(-1)).reshape(fp4_vals.shape)


@pytest.mark.parametrize("m,n,dtype", _FP4_PRECISION_CASES)
def test_flydsl_per_1x32_fp4_precision(m: int, n: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x = torch.randn((m, n), dtype=dtype) * 5.0

    ref_y, ref_scale = get_torch_quant(aiter.QuantType.per_1x32)(
        x, quant_dtype=dtypes.fp4x2, shuffle=False
    )
    fd_y, fd_scale = _flydsl_per_1x32_fp4_quant(
        x, quant_dtype=dtypes.fp4x2, shuffle=False
    )

    # 1) Scale bytes (E8M0) must match exactly -- both use the same RNE
    # round-up-to-pow2 rule.
    ref_s_u8 = ref_scale.view(torch.uint8).reshape(m, n // 32)
    fd_s_u8 = fd_scale.view(torch.uint8).reshape(m, n // 32)
    n_mismatch = (ref_s_u8 != fd_s_u8).sum().item()
    assert n_mismatch == 0, (
        f"scale byte mismatch: {n_mismatch}/{ref_s_u8.numel()} bytes differ "
        f"for m={m} n={n} dt={dtype}; first diff at "
        f"{(ref_s_u8 != fd_s_u8).nonzero()[0].tolist()}"
    )

    # 2) Dequantized output should match within fp4 quant granularity.
    ref_deq = _dequant_mxfp4(ref_y.view(torch.uint8), ref_s_u8)
    fd_deq = _dequant_mxfp4(fd_y.view(torch.uint8), fd_s_u8)
    # FP4 buckets are widely spaced -- a max-abs-error tolerance of 0 is
    # achievable because both impls use HW v_cvt_scalef32_pk_fp4_* (or its
    # bit-exact SW emulation).
    max_err = (ref_deq - fd_deq).abs().max().item()
    assert max_err == 0.0, (
        f"fp4 dequant mismatch m={m} n={n} dt={dtype}: max_err={max_err}"
    )


# -----------------------------------------------------------------------------
# MXFP4 (per_1x32) + H_32 Hadamard rotation precision check.
# -----------------------------------------------------------------------------
_FP4_HADAMARD_CASES = [
    # (m, n, dtype) -- n must be a multiple of 32
    (1, 4096, torch.bfloat16),
    (1, 7168, torch.bfloat16),
    (16, 4096, torch.bfloat16),
    (128, 4096, torch.bfloat16),
    (1024, 4096, torch.bfloat16),
    (1024, 7168, torch.bfloat16),
    (16384, 4096, torch.bfloat16),
    (128, 4096, torch.float16),
    (1024, 7168, torch.float16),
]


def _sylvester_hadamard(n: int, device, dtype=torch.float32) -> torch.Tensor:
    """Build H_n by Sylvester recursion (n must be a power of 2).

    Entries are +-1; the rotation actually applied by the kernel is H_n /
    sqrt(n). The kernel butterfly stages stride = 1, 2, 4, ..., n/2 emit the
    natural-order Sylvester matrix, so this reference matches the kernel
    exactly.
    """
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    return H


def _ref_hadamard_rotate_f32(x: torch.Tensor) -> torch.Tensor:
    """f32 reference for the in-kernel rotation y = (H_32 @ x.T).T / sqrt(32).

    Per group of 32 elements along the last dim.
    """
    m, n = x.shape
    assert n % 32 == 0, "n must be a multiple of 32"
    H = _sylvester_hadamard(32, x.device, torch.float32)
    x_f32 = x.float().reshape(m, n // 32, 32)
    # y[..., i] = sum_j H[i, j] * x[..., j] = (x @ H.T)[..., i].
    y = x_f32 @ H.T
    return (y / math.sqrt(32.0)).reshape(m, n)


def _ref_e8m0_scale_from_amax(amax_f32: torch.Tensor) -> torch.Tensor:
    """Reproduce the kernel's RNE-pow2 -> E8M0 byte calculation.

    Bit-exact replica of the FlyDSL bit-trick; produces uint8 scale bytes.
    """
    u = amax_f32.view(torch.int32)
    exp = (u >> 23) & 0xFF
    bit22 = (u >> 22) & 1
    bit21 = (u >> 21) & 1
    lo21 = u & 0x1FFFFF
    round_up = (bit22 != 0) & ((bit21 != 0) | (lo21 != 0) | (exp != 0))
    exp_rounded = exp + round_up.to(torch.int32)
    is_inf_nan = exp == 0xFF
    exp_final = torch.where(
        is_inf_nan,
        torch.tensor(0xFF, dtype=torch.int32, device=exp.device),
        exp_rounded,
    )
    next_pow2 = (exp_final << 23).view(torch.float32)
    inv_scale = next_pow2 * 0.25
    return ((inv_scale.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


@pytest.mark.parametrize("m,n,dtype", _FP4_HADAMARD_CASES)
def test_flydsl_per_1x32_fp4_hadamard_precision(
    m: int, n: int, dtype: torch.dtype
):
    """Validate the fused Hadamard + MXFP4 kernel.

    Three checks, in order of strictness:
      1) Scale bytes (E8M0) must match a pure-torch f32 reference exactly.
      2) Dequantizing the kernel output and applying the inverse rotation
         must recover the original input within ~max-bucket / 2 per element.
      3) Sanity: the rotation actually decorrelates -- per-group amax of the
         rotated values is on average <= the per-group amax of the original
         (within sqrt(n) bound).
    """
    torch.manual_seed(0)
    x = torch.randn((m, n), dtype=dtype) * 5.0

    # --- Reference: do the rotation in f32, then compute the expected E8M0
    # scale bytes via the same bit-trick the kernel uses.
    x_rot_f32 = _ref_hadamard_rotate_f32(x)  # (m, n) f32
    amax = x_rot_f32.abs().reshape(m, n // 32, 32).amax(dim=-1)  # (m, n//32)
    ref_scale_bytes = _ref_e8m0_scale_from_amax(amax)  # (m, n//32) u8

    # --- Run the kernel.
    fd_y, fd_scale = _flydsl_per_1x32_fp4_quant_hadamard(
        x, quant_dtype=dtypes.fp4x2, shuffle=False
    )
    fd_s_u8 = fd_scale.view(torch.uint8).reshape(m, n // 32)

    # (1) Bit-exact scale match. Same Hadamard order, same RNE bit-trick, so
    # the kernel and the f32 reference must produce identical bytes.
    n_mismatch = (ref_scale_bytes != fd_s_u8).sum().item()
    assert n_mismatch == 0, (
        f"scale byte mismatch: {n_mismatch}/{ref_scale_bytes.numel()} bytes "
        f"differ for m={m} n={n} dt={dtype}; first diff at "
        f"{(ref_scale_bytes != fd_s_u8).nonzero()[0].tolist()}"
    )

    # (2) Dequantize and check the residual is bounded.
    # Per-bucket spacing in MXFP4 = scale (the gap between fp4 codes 0.5/1.0,
    # 1.0/1.5, 1.5/2.0 etc. is 0.5, then 1.0, then 1.0, then 2.0 in the
    # *unscaled* fp4 space). Worst-case per-bucket rounding error = scale.
    # We accept |dequant - x_rot_f32| <= 1.01 * scale_f32 per element.
    fd_deq = _dequant_mxfp4(fd_y.view(torch.uint8), fd_s_u8)  # (m, n) f32
    scale_f32 = torch.pow(
        torch.tensor(2.0, device=fd_s_u8.device),
        (fd_s_u8.to(torch.int32) - 127).float(),
    )  # (m, n//32)
    per_elem_scale = scale_f32.unsqueeze(-1).expand(-1, -1, 32).reshape(m, n)
    elem_err = (fd_deq - x_rot_f32).abs()
    bound = 1.01 * per_elem_scale
    n_bad = (elem_err > bound).sum().item()
    assert n_bad == 0, (
        f"dequant residual exceeded per-bucket spacing at {n_bad}/{m*n} "
        f"elements for m={m} n={n} dt={dtype}; "
        f"max ratio={(elem_err / per_elem_scale).max().item():.3f}"
    )

    # (3) Rotation sanity: average per-group amax of the rotated tensor is
    # comparable to the original (Hadamard is orthonormal -- preserves
    # ell2 -- so for random inputs amax should be of similar order). This
    # mostly catches "I accidentally forgot to normalize / used the wrong
    # H matrix" bugs.
    amax_orig = x.float().abs().reshape(m, n // 32, 32).amax(dim=-1).mean()
    amax_rot = amax.mean()
    # Rough check: rotated amax shouldn't blow up by more than sqrt(32) ~ 5.66.
    assert amax_rot < amax_orig * math.sqrt(32) * 1.1, (
        f"rotated amax too large: orig={amax_orig:.3f} rot={amax_rot:.3f}"
    )


# -----------------------------------------------------------------------------
# MXFP4 (per_1x32) + per-block rotation precision check.
# -----------------------------------------------------------------------------
_FP4_BLOCK_ROT_CASES = [
    # (m, n, dtype, rot_dtype)
    (1,     4096, torch.bfloat16, torch.bfloat16),
    (16,    4096, torch.bfloat16, torch.bfloat16),
    (128,   4096, torch.bfloat16, torch.bfloat16),
    (1024,  4096, torch.bfloat16, torch.bfloat16),
    (1024,  7168, torch.bfloat16, torch.bfloat16),
    (16384, 4096, torch.bfloat16, torch.bfloat16),
    (128,   4096, torch.float16,  torch.float16),
    (1024,  7168, torch.float16,  torch.float16),
    # f32 R: caller may keep R in fp32 for precision-critical paths.
    (128,   4096, torch.bfloat16, torch.float32),
    (1024,  4096, torch.bfloat16, torch.float32),
]


def _apply_block_rotation_f32(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """f32 reference for the in-kernel rotation -- mirrors the host helper

    .. code-block:: python

        def apply_block_rotation(x, R):
            *lead, N = x.shape
            B, g, _ = R.shape
            xb = x.reshape(*lead, B, g)
            return torch.einsum("...bg,bhg->...bh", xb, R).reshape(*lead, N)

    Promotes both x and R to f32 before the einsum so we get a bit-for-bit
    match with the kernel's internal f32 matmul.
    """
    *lead, N = x.shape
    B, g, _ = R.shape
    assert B * g == N
    x_f32 = x.float().reshape(*lead, B, g)
    R_f32 = R.float()
    y_f32 = torch.einsum("...bg,bhg->...bh", x_f32, R_f32)
    y_f32_new = torch.einsum("...bg,bhg->...bg", x_f32, R_f32.transpose(1, 2))
    return y_f32.reshape(*lead, N)


@pytest.mark.parametrize("m,n,dtype,rot_dtype", _FP4_BLOCK_ROT_CASES)
def test_flydsl_per_1x32_fp4_block_rotation_precision(
    m: int, n: int, dtype: torch.dtype, rot_dtype: torch.dtype
):
    """Validate the per-block-rotated MXFP4 kernel.

    Reference: apply ``apply_block_rotation(x, R)`` in pure f32, then re-derive
    the E8M0 scale via the same bit-trick the kernel uses. The kernel reads
    bf16/fp16 inputs (and possibly bf16 R) and extf's everything to f32
    internally, so we expect a bit-exact scale match against an f32 reference
    that promotes the same way.
    """
    torch.manual_seed(0)
    x = torch.randn((m, n), dtype=dtype) * 5.0
    g, B = 32, n // 32
    R = torch.randn(B, g, g, dtype=rot_dtype, device=x.device)
    # Make R closer to orthonormal so amax of rotated values doesn't blow up
    # (this matches realistic SpinQuant-style rotation matrices). The
    # correctness check itself doesn't require R to be orthogonal.
    Q, _ = torch.linalg.qr(R.float())
    R = Q.to(rot_dtype).contiguous()

    # --- Reference: rotate in f32, then derive E8M0 scale bit-exactly.
    y_rot_f32 = _apply_block_rotation_f32(x, R)  # (m, n)
    amax = y_rot_f32.abs().reshape(m, B, g).amax(dim=-1)  # (m, B)
    ref_scale_bytes = _ref_e8m0_scale_from_amax(amax)  # (m, B)

    # --- Run the kernel.
    y_out = torch.empty(m, n // 2, dtype=torch.uint8, device=x.device)
    s_out = torch.empty(m, B, dtype=torch.uint8, device=x.device)
    flydsl_per_1x32_fp4_quant_block_rotation(y_out, x, R, s_out)
    torch.cuda.synchronize()

    # (1) Bit-exact scale match: same rotation in f32, same RNE-pow2 trick.
    n_mismatch = (ref_scale_bytes != s_out).sum().item()
    assert n_mismatch == 0, (
        f"scale byte mismatch: {n_mismatch}/{ref_scale_bytes.numel()} bytes "
        f"differ for m={m} n={n} dt={dtype} R_dt={rot_dtype}; "
        f"first diff at {(ref_scale_bytes != s_out).nonzero()[0].tolist()}"
    )

    # (2) Dequant residual bounded by per-bucket spacing (= scale).
    deq = _dequant_mxfp4(y_out, s_out)
    scale_f32 = torch.pow(
        torch.tensor(2.0, device=s_out.device),
        (s_out.to(torch.int32) - 127).float(),
    )  # (m, B)
    per_elem_scale = scale_f32.unsqueeze(-1).expand(-1, -1, g).reshape(m, n)
    elem_err = (deq - y_rot_f32).abs()
    bound = 1.01 * per_elem_scale
    n_bad = (elem_err > bound).sum().item()
    assert n_bad == 0, (
        f"dequant residual exceeded per-bucket spacing at {n_bad}/{m*n} "
        f"elements for m={m} n={n} dt={dtype} R_dt={rot_dtype}; "
        f"max ratio={(elem_err / per_elem_scale).max().item():.3f}"
    )


# -----------------------------------------------------------------------------
# MXFP4 per-block-rotation **MFMA** precision check.
# -----------------------------------------------------------------------------
# Same math as the scalar variant but with v_mfma_f32_16x16x16_{bf16,f16}_1k
# in the matmul. The MFMA path differs from the scalar fma chain in
# accumulation order (tree-reduction inside each 16-K MFMA, then K-tile sum
# vs. left-to-right scalar fma), so amax can land 1 ULP on either side of a
# power-of-2 rounding boundary. We allow at most a tiny fraction of E8M0
# scale bytes to differ from the f32 reference by exactly 1 (per the bit
# trick: that means the amax sits exactly at a 2^k boundary), and the
# dequant residual gate is identical to the scalar test (1.01 * bucket spacing).
# MFMA path requires bf16 input + bf16 R, or fp16 input + fp16 R (no f32 R).
_FP4_BLOCK_ROT_MFMA_CASES = [
    # (m, n, dtype)
    (1,     4096, torch.bfloat16),
    (16,    4096, torch.bfloat16),
    (128,   4096, torch.bfloat16),
    (1024,  4096, torch.bfloat16),
    (1024,  7168, torch.bfloat16),
    (16384, 4096, torch.bfloat16),
    (128,   4096, torch.float16),
    (1024,  7168, torch.float16),
]


@pytest.mark.parametrize(
    "mfma_variant",
    [
        pytest.param("v1", id="mfma_v1"),
        pytest.param("v1_transposed", id="mfma_v1_transposed"),
    ],
)
@pytest.mark.parametrize(
    "m,n,dtype",
    [
        (1, 4096, torch.bfloat16),
        (64, 4096, torch.bfloat16),
        (1024, 4096, torch.bfloat16),
        (1024, 7168, torch.bfloat16),
        (64, 4096, torch.float16),
    ],
)
def test_flydsl_per_1x32_fp4_block_rotation_mfma_identity_R(
    m: int, n: int, dtype: torch.dtype, mfma_variant: str
):
    """Hard sanity check: with R = I, the MFMA fused kernel must produce
    **bit-exact** the same output as the no-rotation FlyDSL MXFP4 kernel.

    This isolates the MFMA matmul path. If the kernel were doing the wrong
    matmul (transposed R, wrong column offset, wrong K-tile sum, etc.),
    R = I would still produce ``y_rot == x`` mathematically, but any
    indexing error inside the matmul would shift X values across columns
    and the output would mismatch plain quant byte-by-byte.

    The ``v1_transposed`` variant exercises the ``rot_transposed=True``
    code path. Since ``I^T == I``, the same identity tensor is the
    physical transpose of itself, so we can reuse the same R but flip
    the flag -- if the transposed coop-load is wrong, it will land
    elements in wrong LDS positions and the output won't match plain
    quant.
    """
    torch.manual_seed(0)
    x = torch.randn((m, n), dtype=dtype) * 5.0
    g, B = 32, n // 32

    R = torch.zeros(B, g, g, dtype=dtype, device=x.device)
    R[:, torch.arange(g), torch.arange(g)] = 1  # R[b] = I_g for every b

    y_rot = torch.empty(m, n // 2, dtype=torch.uint8, device=x.device)
    s_rot = torch.empty(m, B, dtype=torch.uint8, device=x.device)
    if mfma_variant == "v1":
        flydsl_per_1x32_fp4_quant_block_rotation_mfma(y_rot, x, R, s_rot)
    elif mfma_variant == "v1_transposed":
        flydsl_per_1x32_fp4_quant_block_rotation_mfma(
            y_rot, x, R, s_rot, rot_transposed=True
        )
    else:
        raise AssertionError(f"unknown mfma_variant {mfma_variant!r}")

    y_plain = torch.empty(m, n // 2, dtype=torch.uint8, device=x.device)
    s_plain = torch.empty(m, B, dtype=torch.uint8, device=x.device)
    flydsl_per_1x32_fp4_quant(y_plain, x, s_plain)
    torch.cuda.synchronize()

    # Scale bytes: allow 1-ULP drift on power-of-2 amax boundaries because
    # the MFMA path always rounds through an f32 acc tree (even with R=I the
    # acc value is bitcast through a 16x16 _1k mfma reduce), while the
    # plain quant path goes bf16 -> f32 directly. The fraction must be tiny.
    s_diff = (s_rot.to(torch.int32) - s_plain.to(torch.int32)).abs()
    assert (s_diff <= 1).all().item(), (
        f"R=I scale mismatch >1 ULP for m={m} n={n} dt={dtype} "
        f"variant={mfma_variant}: max diff={s_diff.max().item()}"
    )
    s_mismatch = (s_diff != 0).sum().item()
    assert s_mismatch / s_rot.numel() <= 5e-3, (
        f"R=I scale 1-ULP fraction too high m={m} n={n} dt={dtype} "
        f"variant={mfma_variant}: {s_mismatch}/{s_rot.numel()}"
    )

    # fp4 bytes: dequant both and require per-element delta no larger than 1
    # E2M1 bucket-spacing. A real matmul bug would shift values across cols
    # and explode this gate.
    deq_rot = _dequant_mxfp4(y_rot, s_rot)
    deq_plain = _dequant_mxfp4(y_plain, s_plain)
    per_elem_scale_rot = torch.pow(
        torch.tensor(2.0, device=s_rot.device),
        (s_rot.to(torch.int32) - 127).float(),
    ).unsqueeze(-1).expand(-1, -1, g).reshape(m, n)
    delta = (deq_rot - deq_plain).abs()
    n_bad = (delta > per_elem_scale_rot * 1.01).sum().item()
    assert n_bad == 0, (
        f"R=I fp4 byte mismatch beyond 1 bucket for m={m} n={n} dt={dtype} "
        f"variant={mfma_variant}: {n_bad}/{m*n} elements off; "
        f"max ratio={(delta / per_elem_scale_rot).max().item():.3f}"
    )


@pytest.mark.parametrize(
    "mfma_variant",
    [
        pytest.param("v1", id="mfma_v1"),
        pytest.param("v1_transposed", id="mfma_v1_transposed"),
    ],
)
@pytest.mark.parametrize("m,n,dtype", _FP4_BLOCK_ROT_MFMA_CASES)
def test_flydsl_per_1x32_fp4_block_rotation_mfma_precision(
    m: int, n: int, dtype: torch.dtype, mfma_variant: str
):
    """Validate the MFMA-accelerated block-rotated MXFP4 kernel.

    The ``v1_transposed`` variant exercises the ``rot_transposed=True``
    code path: we pass ``R.transpose(-1, -2).contiguous()`` to the
    kernel, which means the kernel reads ``rot_R[b, g, h]`` and
    internally lays it back into the canonical (h, g) LDS layout. The
    arithmetic result must match the *same* reference rotation used by
    the non-transposed variant.
    """
    torch.manual_seed(0)
    x = torch.randn((m, n), dtype=dtype) * 5.0
    g, B = 32, n // 32
    R = torch.randn(B, g, g, dtype=dtype, device=x.device)
    Q, _ = torch.linalg.qr(R.float())
    R = Q.to(dtype).contiguous()

    # f32 reference: same as the scalar variant. Reference uses R in
    # (B, h, g) interpretation always; the transposed kernel variant
    # below is fed R.transpose(-1, -2) so its internal math collapses
    # to the same answer.
    y_rot_f32 = _apply_block_rotation_f32(x, R)
    amax = y_rot_f32.abs().reshape(m, B, g).amax(dim=-1)
    ref_scale_bytes = _ref_e8m0_scale_from_amax(amax)  # (m, B)

    y_out = torch.empty(m, n // 2, dtype=torch.uint8, device=x.device)
    s_out = torch.empty(m, B, dtype=torch.uint8, device=x.device)
    if mfma_variant == "v1":
        flydsl_per_1x32_fp4_quant_block_rotation_mfma(y_out, x, R, s_out)
    elif mfma_variant == "v1_transposed":
        R_T = R.transpose(-1, -2).contiguous()
        flydsl_per_1x32_fp4_quant_block_rotation_mfma(
            y_out, x, R_T, s_out, rot_transposed=True
        )
    else:
        raise AssertionError(f"unknown mfma_variant {mfma_variant!r}")
    torch.cuda.synchronize()

    # (1) Scale bytes: bit-exact in the common case; allow rare 1-ULP drift
    #     near power-of-2 amax boundaries (MFMA tree-reduction vs. scalar
    #     left-to-right fma can flip RNE direction on the boundary).
    diff = (ref_scale_bytes.to(torch.int32) - s_out.to(torch.int32)).abs()
    n_mismatch = (diff != 0).sum().item()
    n_total = ref_scale_bytes.numel()
    # Allow at most 0.5% of scales to differ, and each by at most 1 ULP.
    assert (diff <= 1).all().item(), (
        f"scale byte off by >1 for m={m} n={n} dt={dtype} variant={mfma_variant}: "
        f"max diff={diff.max().item()} at "
        f"{(diff > 1).nonzero()[0].tolist()}"
    )
    assert n_mismatch / n_total <= 5e-3, (
        f"scale 1-ULP mismatch fraction too high for m={m} n={n} dt={dtype} "
        f"variant={mfma_variant}: {n_mismatch}/{n_total} = {n_mismatch/n_total:.4%}"
    )

    # (2) Same dequant residual bound as the scalar test.
    deq = _dequant_mxfp4(y_out, s_out)
    scale_f32 = torch.pow(
        torch.tensor(2.0, device=s_out.device),
        (s_out.to(torch.int32) - 127).float(),
    )
    per_elem_scale = scale_f32.unsqueeze(-1).expand(-1, -1, g).reshape(m, n)
    elem_err = (deq - y_rot_f32).abs()
    bound = 1.01 * per_elem_scale
    n_bad = (elem_err > bound).sum().item()
    assert n_bad == 0, (
        f"dequant residual exceeded per-bucket spacing at {n_bad}/{m*n} "
        f"elements for m={m} n={n} dt={dtype} variant={mfma_variant}; "
        f"max ratio={(elem_err / per_elem_scale).max().item():.3f}"
    )


# -----------------------------------------------------------------------------
# Benchmark (script-style; mirrors op_tests/test_quant.py structure).
# -----------------------------------------------------------------------------
def _bench_case_per_tensor(m: int, n: int, dtype: torch.dtype) -> Dict[str, float]:
    """Return per-impl us + max-err for a single per-tensor (m, n, dtype) case."""
    x = torch.randn((m, n), dtype=dtype) * 5.0

    ref_y, ref_scale = get_torch_quant(aiter.QuantType.per_Tensor)(
        x, quant_dtype=dtypes.fp8
    )

    impls: Dict[str, Callable] = {
        "flydsl": _flydsl_per_tensor_quant,
        "hip": get_hip_quant(aiter.QuantType.per_Tensor),
        "triton": get_triton_quant(aiter.QuantType.per_Tensor),
    }

    res: Dict[str, float] = {"m": m, "n": n, "dt": str(dtype).split(".")[-1]}
    for name, fn in impls.items():
        try:
            (out, scale), us = run_perftest(fn, x, quant_dtype=dtypes.fp8)
        except Exception as exc:
            res[f"{name}_us"] = float("nan")
            res[f"{name}_err"] = float("nan")
            res[f"{name}_msg"] = f"FAILED: {exc!s}"
            continue
        err = (ref_y.to(dtypes.fp32) - out.to(dtypes.fp32)).abs().max().item()
        scale_err = (
            (ref_scale.to(dtypes.fp32) - scale.to(dtypes.fp32)).abs().max().item()
        )
        res[f"{name}_us"] = us
        res[f"{name}_err"] = err
        res[f"{name}_scale_err"] = scale_err
    return res


def _bench_case_fp4(m: int, n: int, dtype: torch.dtype) -> Dict[str, float]:
    """Return per-impl us for MXFP4 per-1x32 quant at (m, n, dtype)."""
    x = torch.randn((m, n), dtype=dtype) * 5.0

    impls: Dict[str, Callable] = {
        "flydsl": _flydsl_per_1x32_fp4_quant,
        "hip": get_hip_quant(aiter.QuantType.per_1x32),
        "triton": get_triton_quant(aiter.QuantType.per_1x32),
    }

    res: Dict[str, float] = {"m": m, "n": n, "dt": str(dtype).split(".")[-1]}
    for name, fn in impls.items():
        try:
            (out, scale), us = run_perftest(
                fn, x, quant_dtype=dtypes.fp4x2, shuffle=False
            )
        except Exception as exc:
            res[f"{name}_us"] = float("nan")
            res[f"{name}_msg"] = f"FAILED: {exc!s}"
            continue
        res[f"{name}_us"] = us
    return res


def _bench_case_fp4_block_rotation(
    m: int, n: int, dtype: torch.dtype
) -> Dict[str, float]:
    """Per-block-rotated MXFP4 bench: FlyDSL fused vs HIP/Triton 2-step.

    The HIP / Triton baselines pay an extra ``apply_block_rotation`` pass in
    bf16/fp16 plus the existing MXFP4 quant; that's what users pay today
    without the fusion.
    """
    x = torch.randn((m, n), dtype=dtype) * 5.0
    g, B = 32, n // 32
    R = torch.randn(B, g, g, dtype=dtype, device=x.device)
    Q, _ = torch.linalg.qr(R.float())
    R = Q.to(dtype).contiguous()

    def _fd_fused(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        y = torch.empty(m, n // 2, dtype=torch.uint8, device=x_in.device)
        s = torch.empty(m, B, dtype=torch.uint8, device=x_in.device)
        flydsl_per_1x32_fp4_quant_block_rotation(y, x_in, R, s)
        return (
            y.view(dtypes.fp4x2),
            s.view(dtypes.fp8_e8m0),
        )

    def _torch_rotate(x_in):
        xb = x_in.reshape(-1, B, g)
        yb = torch.einsum("...bg,bhg->...bh", xb, R)
        return yb.reshape(m, n)

    def _hip_baseline(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        x_rot = _torch_rotate(x_in)
        return get_hip_quant(aiter.QuantType.per_1x32)(
            x_rot, quant_dtype=quant_dtype, shuffle=shuffle
        )

    def _triton_baseline(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        x_rot = _torch_rotate(x_in)
        return get_triton_quant(aiter.QuantType.per_1x32)(
            x_rot, quant_dtype=quant_dtype, shuffle=shuffle
        )

    impls: Dict[str, Callable] = {
        "flydsl": _fd_fused,
        "hip": _hip_baseline,
        "triton": _triton_baseline,
    }

    res: Dict[str, float] = {"m": m, "n": n, "dt": str(dtype).split(".")[-1]}
    for name, fn in impls.items():
        try:
            (out, scale), us = run_perftest(
                fn, x, quant_dtype=dtypes.fp4x2, shuffle=False
            )
        except Exception as exc:
            res[f"{name}_us"] = float("nan")
            res[f"{name}_msg"] = f"FAILED: {exc!s}"
            continue
        res[f"{name}_us"] = us
    return res


def _bench_case_fp4_block_rotation_mfma(
    m: int, n: int, dtype: torch.dtype
) -> Dict[str, float]:
    """Per-block-rotated MXFP4 bench: scalar FlyDSL vs MFMA FlyDSL vs HIP/Triton 2-step.

    Same baselines as ``_bench_case_fp4_block_rotation`` plus a 4th column
    (``mfma_us``) for the MFMA-accelerated FlyDSL fused kernel. Reported
    ratios are computed against the scalar FlyDSL kernel so we can directly
    quantify the MFMA speedup at each (m, n).
    """
    x = torch.randn((m, n), dtype=dtype) * 5.0
    g, B = 32, n // 32
    R = torch.randn(B, g, g, dtype=dtype, device=x.device)
    Q, _ = torch.linalg.qr(R.float())
    R = Q.to(dtype).contiguous()

    def _fd_scalar(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        y = torch.empty(m, n // 2, dtype=torch.uint8, device=x_in.device)
        s = torch.empty(m, B, dtype=torch.uint8, device=x_in.device)
        flydsl_per_1x32_fp4_quant_block_rotation(y, x_in, R, s)
        return y.view(dtypes.fp4x2), s.view(dtypes.fp8_e8m0)

    def _fd_mfma(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        y = torch.empty(m, n // 2, dtype=torch.uint8, device=x_in.device)
        s = torch.empty(m, B, dtype=torch.uint8, device=x_in.device)
        flydsl_per_1x32_fp4_quant_block_rotation_mfma(y, x_in, R, s)
        return y.view(dtypes.fp4x2), s.view(dtypes.fp8_e8m0)

    def _torch_rotate(x_in):
        xb = x_in.reshape(-1, B, g)
        yb = torch.einsum("...bg,bhg->...bh", xb, R)
        return yb.reshape(m, n)

    def _hip_baseline(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        x_rot = _torch_rotate(x_in)
        return get_hip_quant(aiter.QuantType.per_1x32)(
            x_rot, quant_dtype=quant_dtype, shuffle=shuffle
        )

    def _triton_baseline(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        x_rot = _torch_rotate(x_in)
        return get_triton_quant(aiter.QuantType.per_1x32)(
            x_rot, quant_dtype=quant_dtype, shuffle=shuffle
        )

    impls: Dict[str, Callable] = {
        "flydsl": _fd_scalar,
        "mfma": _fd_mfma,
        "hip": _hip_baseline,
        "triton": _triton_baseline,
    }

    res: Dict[str, float] = {"m": m, "n": n, "dt": str(dtype).split(".")[-1]}
    for name, fn in impls.items():
        try:
            (out, scale), us = run_perftest(
                fn, x, quant_dtype=dtypes.fp4x2, shuffle=False
            )
        except Exception as exc:
            res[f"{name}_us"] = float("nan")
            res[f"{name}_msg"] = f"FAILED: {exc!s}"
            continue
        res[f"{name}_us"] = us
    return res


def _bench_case_fp4_hadamard(
    m: int, n: int, dtype: torch.dtype
) -> Dict[str, float]:
    """Return per-impl us for the H_32-fused MXFP4 path.

    Only FlyDSL has a fused-Hadamard kernel today; the HIP / Triton columns
    measure ``rotate_in_f32 + quant`` as a two-step baseline so we can
    quantify the value of the fusion.
    """
    x = torch.randn((m, n), dtype=dtype) * 5.0

    def _hip_baseline(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        x_rot = _ref_hadamard_rotate_f32(x_in).to(x_in.dtype)
        return get_hip_quant(aiter.QuantType.per_1x32)(
            x_rot, quant_dtype=quant_dtype, shuffle=shuffle
        )

    def _triton_baseline(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        x_rot = _ref_hadamard_rotate_f32(x_in).to(x_in.dtype)
        return get_triton_quant(aiter.QuantType.per_1x32)(
            x_rot, quant_dtype=quant_dtype, shuffle=shuffle
        )

    impls: Dict[str, Callable] = {
        "flydsl": _flydsl_per_1x32_fp4_quant_hadamard,
        "hip": _hip_baseline,      # rotate + HIP MXFP4 (unfused)
        "triton": _triton_baseline,  # rotate + Triton MXFP4 (unfused)
    }

    res: Dict[str, float] = {"m": m, "n": n, "dt": str(dtype).split(".")[-1]}
    for name, fn in impls.items():
        try:
            (out, scale), us = run_perftest(
                fn, x, quant_dtype=dtypes.fp4x2, shuffle=False
            )
        except Exception as exc:
            res[f"{name}_us"] = float("nan")
            res[f"{name}_msg"] = f"FAILED: {exc!s}"
            continue
        res[f"{name}_us"] = us
    return res


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        nargs="*",
        default=[1, 16, 128, 1024, 16384, 163840],
        help="Token / row counts to sweep.",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        nargs="*",
        default=[4096, 7168, 8192],
        help="Hidden dims to sweep.",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        nargs="*",
        default=["bf16"],
        choices=["bf16", "fp16"],
    )
    parser.add_argument(
        "-q",
        "--quant",
        type=str,
        nargs="*",
        default=["per_tensor"],
        choices=[
            "per_tensor",
            "fp4",
            "fp4_hadamard",
            "fp4_block_rotation",
            "fp4_block_rotation_mfma",
        ],
        help="Which quantization flavor(s) to benchmark.",
    )
    args = parser.parse_args(argv)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    bench_dispatch = {
        "per_tensor": _bench_case_per_tensor,
        "fp4": _bench_case_fp4,
        "fp4_hadamard": _bench_case_fp4_hadamard,
        "fp4_block_rotation": _bench_case_fp4_block_rotation,
        "fp4_block_rotation_mfma": _bench_case_fp4_block_rotation_mfma,
    }

    has_mfma_col = "fp4_block_rotation_mfma" in args.quant
    if has_mfma_col:
        header = (
            f"{'qtype':>22} {'dt':>10} {'m':>6} {'n':>6} "
            f"{'flydsl_us':>10} {'mfma_us':>9} "
            f"{'hip_us':>8} {'triton_us':>10} "
            f"{'mfma/fd':>8} "
            f"{'fd/hip':>7} {'fd/triton':>10}"
        )
    else:
        header = (
            f"{'qtype':>22} {'dt':>10} {'m':>6} {'n':>6} "
            f"{'flydsl_us':>10} {'hip_us':>8} {'triton_us':>10} "
            f"{'fd/hip':>7} {'fd/triton':>10}"
        )
    print(header)
    print("-" * len(header))

    for qt in args.quant:
        bench_fn = bench_dispatch[qt]
        for dt_str, n, m in itertools.product(args.dtype, args.n, args.m):
            if m * n > 1_500_000_000:
                print(f"[skip] m={m} n={n} too large", file=sys.stderr)
                continue
            r = bench_fn(m, n, dtype_map[dt_str])
            fd = r.get("flydsl_us", float("nan"))
            hp = r.get("hip_us", float("nan"))
            tr = r.get("triton_us", float("nan"))
            fd_hip = (fd / hp) if (hp == hp and hp > 0) else float("nan")
            fd_tri = (fd / tr) if (tr == tr and tr > 0) else float("nan")
            if has_mfma_col:
                mfma_us = r.get("mfma_us", float("nan"))
                mfma_ratio = (
                    (mfma_us / fd) if (fd == fd and fd > 0 and mfma_us == mfma_us) else float("nan")
                )
                print(
                    f"{qt:>22} {r['dt']:>10} {r['m']:>6} {r['n']:>6} "
                    f"{fd:>10.2f} {mfma_us:>9.2f} "
                    f"{hp:>8.2f} {tr:>10.2f} "
                    f"{mfma_ratio:>8.2f} "
                    f"{fd_hip:>7.2f} {fd_tri:>10.2f}"
                )
            else:
                print(
                    f"{qt:>22} {r['dt']:>10} {r['m']:>6} {r['n']:>6} "
                    f"{fd:>10.2f} {hp:>8.2f} {tr:>10.2f} "
                    f"{fd_hip:>7.2f} {fd_tri:>10.2f}"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
