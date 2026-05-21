# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.quant.quant_mxfp8 import (
    per_1x32_mxfp8_quant_triton,
    fp8_legacy_to_mxfp8,
    rmsnorm_mxfp8_quant,
    dual_rmsnorm_mxfp8_quant,
    fused_flatten_mxfp8_quant,
)
import aiter.ops.triton.utils._triton.arch_info as arch_info

QUANT_BLOCK_SIZE = 32
LEGACY_BLOCK_SIZE = 128
# 0xFF800000 in two's complement int32. Mask keeps sign + 8-bit exponent + top mantissa bit.
_E8M0_MASK_INT32 = -8388608


def torch_mxfp8_quant_from_fp32(x_fp32: torch.Tensor):
    """Bit-faithful port of `_mxfp8_quant_kernel` quant logic, taking fp32 input.

    Computes per-1x32 e8m0 scale (uint8) and FP8 e4m3fn values.
    """
    assert x_fp32.dim() == 2, f"x_fp32 must be 2D, got {x_fp32.dim()}"
    M, K = x_fp32.shape
    assert K % QUANT_BLOCK_SIZE == 0
    Ng = K // QUANT_BLOCK_SIZE
    x_2d = x_fp32.reshape(M, Ng, QUANT_BLOCK_SIZE).to(torch.float32)
    amax = torch.amax(torch.abs(x_2d), dim=-1, keepdim=True)  # (M, Ng, 1)

    # Same bit-level "round up to e8m0-representable pow-2" as the kernel.
    amax_i32 = amax.contiguous().view(torch.int32)
    amax_i32 = (amax_i32 + 0x200000) & _E8M0_MASK_INT32
    amax_p2 = amax_i32.view(torch.float32)

    scale_unbiased = torch.log2(amax_p2).floor() - 8
    scale_unbiased = torch.clamp(scale_unbiased, min=-127, max=127)
    scale_e8m0 = (scale_unbiased.to(torch.int32) + 127).to(torch.uint8)
    quant_scale = torch.exp2(-scale_unbiased)

    qx_2d = x_2d * quant_scale  # broadcast over inner-32
    qx = qx_2d.reshape(M, K)
    y_fp8 = qx.to(torch.float8_e4m3fn)
    s = scale_e8m0.reshape(M, Ng)
    return y_fp8, s


def e8m0_to_f32(x: torch.Tensor) -> torch.Tensor:
    return torch.exp2((x.to(torch.int32) - 127).to(torch.float32))


# -----------------------------------------------------------------------------
# per_1x32_mxfp8_quant_triton
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M, K",
    [
        (1, 32),
        (1, 64),
        (1, 128),
        (2, 32),
        (8, 64),
        (16, 128),
        (32, 256),
        (64, 512),
        (128, 1024),
        (137, 64),  # non-power-of-2 M
        (256, 32),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_per_1x32_mxfp8_quant(M: int, K: int, dtype: torch.dtype):
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 not supported on this arch")
    torch.cuda.empty_cache()
    torch.manual_seed(20)

    x = torch.randn((M, K), dtype=dtype, device="cuda") * 4.0

    # Reference path: emulate the kernel in fp32 (matching its precision).
    x_fp32 = x.to(torch.float32)
    y_ref, s_ref = torch_mxfp8_quant_from_fp32(x_fp32)

    # Triton path.
    y_kern, s_kern = per_1x32_mxfp8_quant_triton(x)

    # Scales must be bit-exact: the e8m0 derivation is integer-only after
    # the fp32 cast, and amax is order-independent.
    torch.testing.assert_close(s_kern, s_ref)

    # Quantized values: compare via the uint8 view (allow off-by-1 for any
    # rounding-mode subtlety in the fp32→fp8 cast).
    torch.testing.assert_close(
        y_kern.view(torch.uint8).to(torch.int32),
        y_ref.view(torch.uint8).to(torch.int32),
        atol=1,
        rtol=0,
    )


def test_per_1x32_mxfp8_quant_preallocated_scale():
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 not supported on this arch")
    torch.cuda.empty_cache()
    torch.manual_seed(20)

    M, K = 64, 256
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    scale_pre = torch.empty(
        (M, K // QUANT_BLOCK_SIZE), dtype=torch.uint8, device="cuda"
    )
    y, s = per_1x32_mxfp8_quant_triton(x, scale=scale_pre)
    assert s.data_ptr() == scale_pre.data_ptr()

    y_ref, s_ref = torch_mxfp8_quant_from_fp32(x.to(torch.float32))
    torch.testing.assert_close(s, s_ref)


def test_per_1x32_mxfp8_quant_multidim():
    """Wrapper folds higher dims into M; sanity-check 3D input."""
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 not supported on this arch")
    torch.cuda.empty_cache()
    torch.manual_seed(0)

    B, M, K = 4, 8, 128
    x = torch.randn((B, M, K), dtype=torch.bfloat16, device="cuda")
    y, s = per_1x32_mxfp8_quant_triton(x)
    assert y.shape == (B, M, K)
    assert s.shape == (B, M, K // QUANT_BLOCK_SIZE)

    y_ref, s_ref = torch_mxfp8_quant_from_fp32(x.reshape(-1, K).to(torch.float32))
    torch.testing.assert_close(s.reshape(-1, K // QUANT_BLOCK_SIZE), s_ref)


# -----------------------------------------------------------------------------
# fp8_legacy_to_mxfp8
# -----------------------------------------------------------------------------


def torch_fp8_legacy_to_mxfp8(x_fnuz: torch.Tensor, x_scale_fp32: torch.Tensor):
    """Reference: dequantize fnuz fp8 with the 1x128 fp32 scale, then run
    the standard mxfp8 1x32 quant on the result."""
    M, N = x_fnuz.shape
    x_dq = x_fnuz.to(torch.float32) * x_scale_fp32.repeat_interleave(
        LEGACY_BLOCK_SIZE, dim=1
    )
    return torch_mxfp8_quant_from_fp32(x_dq)


@pytest.mark.parametrize(
    "M, N",
    [
        (1, 128),
        (8, 128),
        (16, 256),
        (32, 512),
        (64, 1024),
        (128, 256),
        (37, 256),  # non-pow-2 M
    ],
)
def test_fp8_legacy_to_mxfp8(M: int, N: int):
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 not supported on this arch")
    torch.cuda.empty_cache()
    torch.manual_seed(5)

    # Random values within e4m3fnuz range, then cast to fnuz fp8.
    x_f32 = (torch.randn((M, N), dtype=torch.float32, device="cuda")).clamp(-200, 200)
    x_fnuz = x_f32.to(torch.float8_e4m3fnuz)
    # Random fp32 1x128 scales in a moderate range so the dequant stays within fp8.
    x_scale_fp32 = (
        torch.rand((M, N // LEGACY_BLOCK_SIZE), dtype=torch.float32, device="cuda")
        * 0.5
        + 0.25
    )

    y_ref, s_ref = torch_fp8_legacy_to_mxfp8(x_fnuz, x_scale_fp32)
    y_kern, s_kern = fp8_legacy_to_mxfp8(x_fnuz, x_scale_fp32)

    torch.testing.assert_close(s_kern, s_ref)
    torch.testing.assert_close(
        y_kern.view(torch.uint8).to(torch.int32),
        y_ref.view(torch.uint8).to(torch.int32),
        atol=1,
        rtol=0,
    )


def test_fp8_legacy_to_mxfp8_preallocated():
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 not supported on this arch")
    torch.cuda.empty_cache()
    torch.manual_seed(5)

    M, N = 16, 256
    x_fnuz = (torch.randn((M, N), device="cuda") * 4).to(torch.float8_e4m3fnuz)
    x_scale_fp32 = torch.rand((M, N // LEGACY_BLOCK_SIZE), device="cuda") * 0.5 + 0.25
    y_pre = torch.empty((M, N), dtype=torch.float8_e4m3fn, device="cuda")
    s_pre = torch.empty((M, N // QUANT_BLOCK_SIZE), dtype=torch.uint8, device="cuda")
    y, s = fp8_legacy_to_mxfp8(x_fnuz, x_scale_fp32, y_fn=y_pre, y_scale=s_pre)
    assert y.data_ptr() == y_pre.data_ptr()
    assert s.data_ptr() == s_pre.data_ptr()

    y_ref, s_ref = torch_fp8_legacy_to_mxfp8(x_fnuz, x_scale_fp32)
    torch.testing.assert_close(s, s_ref)


# -----------------------------------------------------------------------------
# rmsnorm_mxfp8_quant
# -----------------------------------------------------------------------------


def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f32 = x.to(torch.float32)
    g_f32 = weight.to(torch.float32)
    rstd = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps)
    return x_f32 * rstd * g_f32


def torch_rmsnorm_mxfp8_quant(x, weight, eps):
    y_fp32 = torch_rmsnorm(x, weight, eps)
    return torch_mxfp8_quant_from_fp32(y_fp32)


@pytest.mark.parametrize(
    "M, K",
    [
        (1, 32),
        (1, 128),
        (8, 128),
        (16, 256),
        (32, 512),
        (64, 1024),
        (128, 2048),
        (97, 64),  # non-pow-2 M, K=64
        (200, 192),  # non-pow-2 K (still multiple of 32)
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rmsnorm_mxfp8_quant(M: int, K: int, dtype: torch.dtype):
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 not supported on this arch")
    torch.cuda.empty_cache()
    torch.manual_seed(11)

    x = torch.randn((M, K), dtype=dtype, device="cuda")
    weight = torch.randn((K,), dtype=dtype, device="cuda") * 0.5 + 1.0
    eps = 1e-5

    y_ref, s_ref = torch_rmsnorm_mxfp8_quant(x, weight, eps)
    y_kern, s_kern = rmsnorm_mxfp8_quant(x, weight, eps)

    # Hardware rsqrt vs torch.rsqrt can disagree by a ULP; that may flip a single
    # e8m0 bin near a power-of-2 boundary. Compare dequantized values instead.
    s_ref_f32 = e8m0_to_f32(s_ref).repeat_interleave(QUANT_BLOCK_SIZE, dim=1)
    s_kern_f32 = e8m0_to_f32(s_kern).repeat_interleave(QUANT_BLOCK_SIZE, dim=1)
    y_ref_dq = y_ref.to(torch.float32) * s_ref_f32
    y_kern_dq = y_kern.to(torch.float32) * s_kern_f32

    torch.testing.assert_close(y_kern_dq, y_ref_dq, atol=5e-2, rtol=5e-2)


def test_rmsnorm_mxfp8_quant_preallocated():
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 not supported on this arch")
    torch.cuda.empty_cache()
    torch.manual_seed(11)

    M, K = 32, 256
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((K,), dtype=torch.bfloat16, device="cuda")
    y_pre = torch.empty((M, K), dtype=torch.float8_e4m3fn, device="cuda")
    s_pre = torch.empty((M, K // QUANT_BLOCK_SIZE), dtype=torch.uint8, device="cuda")
    y, s = rmsnorm_mxfp8_quant(x, weight, 1e-5, y=y_pre, scale=s_pre)
    assert y.data_ptr() == y_pre.data_ptr()
    assert s.data_ptr() == s_pre.data_ptr()


# -----------------------------------------------------------------------------
# dual_rmsnorm_mxfp8_quant
# -----------------------------------------------------------------------------


def torch_dual_rmsnorm_mxfp8_quant(q, k, q_weight, k_weight, eps_q, eps_k):
    yq_fp32 = torch_rmsnorm(q, q_weight, eps_q)
    yq, sq = torch_mxfp8_quant_from_fp32(yq_fp32)
    yk_fp32 = torch_rmsnorm(k, k_weight, eps_k)
    yk = yk_fp32.to(k.dtype)
    return yq, sq, yk


@pytest.mark.parametrize(
    "M, KQ, KK",
    [
        (1, 32, 32),
        (1, 128, 64),
        (8, 256, 128),
        (16, 512, 256),
        (32, 1024, 512),
        (64, 2048, 1024),
        (47, 96, 80),  # non-pow-2 sizes
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dual_rmsnorm_mxfp8_quant(M: int, KQ: int, KK: int, dtype: torch.dtype):
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 not supported on this arch")
    torch.cuda.empty_cache()
    torch.manual_seed(13)

    q = torch.randn((M, KQ), dtype=dtype, device="cuda")
    k = torch.randn((M, KK), dtype=dtype, device="cuda")
    q_weight = torch.randn((KQ,), dtype=dtype, device="cuda") * 0.5 + 1.0
    k_weight = torch.randn((KK,), dtype=dtype, device="cuda") * 0.5 + 1.0
    eps_q, eps_k = 1e-5, 2e-5

    yq_ref, sq_ref, yk_ref = torch_dual_rmsnorm_mxfp8_quant(
        q, k, q_weight, k_weight, eps_q, eps_k
    )
    yq_kern, sq_kern, yk_kern = dual_rmsnorm_mxfp8_quant(
        q, k, q_weight, k_weight, eps_q, eps_k
    )

    # Q side: compare dequantized values (rsqrt jitter -> tolerate e8m0 ULP flips).
    sq_ref_f32 = e8m0_to_f32(sq_ref).repeat_interleave(QUANT_BLOCK_SIZE, dim=1)
    sq_kern_f32 = e8m0_to_f32(sq_kern).repeat_interleave(QUANT_BLOCK_SIZE, dim=1)
    yq_ref_dq = yq_ref.to(torch.float32) * sq_ref_f32
    yq_kern_dq = yq_kern.to(torch.float32) * sq_kern_f32
    torch.testing.assert_close(yq_kern_dq, yq_ref_dq, atol=5e-2, rtol=5e-2)

    # K side: bf16/fp16 RMSNorm output.
    torch.testing.assert_close(yk_kern, yk_ref, atol=5e-3, rtol=5e-3)


def test_dual_rmsnorm_mxfp8_quant_default_eps_k():
    """eps_k defaults to eps_q when not provided."""
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 not supported on this arch")
    torch.cuda.empty_cache()
    torch.manual_seed(13)

    M, KQ, KK = 16, 128, 96
    dtype = torch.bfloat16
    q = torch.randn((M, KQ), dtype=dtype, device="cuda")
    k = torch.randn((M, KK), dtype=dtype, device="cuda")
    q_weight = torch.randn((KQ,), dtype=dtype, device="cuda")
    k_weight = torch.randn((KK,), dtype=dtype, device="cuda")
    eps = 1e-5

    yq_a, sq_a, yk_a = dual_rmsnorm_mxfp8_quant(q, k, q_weight, k_weight, eps)
    yq_b, sq_b, yk_b = dual_rmsnorm_mxfp8_quant(
        q, k, q_weight, k_weight, eps, eps_k=eps
    )
    torch.testing.assert_close(yq_a.view(torch.uint8), yq_b.view(torch.uint8))
    torch.testing.assert_close(sq_a, sq_b)
    torch.testing.assert_close(yk_a, yk_b)


# -----------------------------------------------------------------------------
# fused_flatten_mxfp8_quant
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M, N1, N2",
    [
        (1, 1, 32),
        (1, 4, 64),
        (8, 2, 128),
        (16, 3, 256),
        (32, 4, 512),
        (64, 1, 1024),
        (37, 5, 64),  # non-pow-2 M
        (128, 8, 32),
        (64, 8, 7168),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_flatten_mxfp8_quant(M: int, N1: int, N2: int, dtype: torch.dtype):
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 not supported on this arch")
    torch.cuda.empty_cache()
    torch.manual_seed(17)

    # x = torch.randn((M, N1, N2), dtype=dtype, device="cuda") * 4.0
    x = torch.randn((N1, M, N2), dtype=dtype, device="cuda").transpose(0, 1) * 4.0

    # Reference: flatten (M, N1, N2) -> (M, N1 * N2), then MXFP8 quant in fp32.
    x_flat_fp32 = x.reshape(M, N1 * N2).to(torch.float32)
    y_ref, s_ref = torch_mxfp8_quant_from_fp32(x_flat_fp32)

    y_kern, s_kern = fused_flatten_mxfp8_quant(x)

    assert y_kern.shape == (M, N1 * N2)
    assert s_kern.shape == (M, (N1 * N2) // QUANT_BLOCK_SIZE)

    # Scales must be bit-exact (integer-only after fp32 cast).
    torch.testing.assert_close(s_kern, s_ref)

    # Quantized values: compare via uint8 view, allow off-by-1 for fp32->fp8
    # rounding-mode subtlety.
    torch.testing.assert_close(
        y_kern.view(torch.uint8).to(torch.int32),
        y_ref.view(torch.uint8).to(torch.int32),
        atol=1,
        rtol=0,
    )


def test_fused_flatten_mxfp8_quant_matches_per_1x32_after_flatten():
    """Sanity: the flatten+quant path should match per_1x32_mxfp8_quant_triton
    applied to the pre-flattened (M, N1 * N2) tensor."""
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 not supported on this arch")
    torch.cuda.empty_cache()
    torch.manual_seed(19)

    M, N1, N2 = 16, 4, 128
    x = torch.randn((M, N1, N2), dtype=torch.bfloat16, device="cuda")

    y_flat, s_flat = fused_flatten_mxfp8_quant(x)
    y_ref, s_ref = per_1x32_mxfp8_quant_triton(x.reshape(M, N1 * N2).contiguous())

    torch.testing.assert_close(s_flat, s_ref)
    torch.testing.assert_close(
        y_flat.view(torch.uint8).to(torch.int32),
        y_ref.view(torch.uint8).to(torch.int32),
        atol=1,
        rtol=0,
    )
