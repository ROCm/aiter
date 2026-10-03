# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.quant.fused_mxfp8_quant import (
    fused_deepseek_v4_compress_norm_rope_store,
    fused_deepseek_v4_compress_norm_rope_store_two_stage,
    fused_deepseek_v4_dequantize_and_gather_k_cache,
    fused_deepseek_v4_mxfp8_quant_q_pack,
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
    fused_deepseek_v4_quantize_and_insert_k_cache,
    fused_dual_rmsnorm_mxfp8_quant,
    fused_flatten_mxfp8_quant,
    fused_rms_mxfp8_quant,
)
from aiter.ops.triton.quant.quant import (
    dynamic_mxfp8_quant,
    fp8_legacy_to_mxfp8,
)
from aiter.ops.triton.utils._triton import arch_info

QUANT_BLOCK_SIZE = 32
LEGACY_BLOCK_SIZE = 128
# 0xFF800000 in two's complement int32. Mask keeps sign + 8-bit exponent + top mantissa bit.
_E8M0_MASK_INT32 = -8388608


def torch_mxfp8_quant_from_fp32(x_fp32: torch.Tensor):
    """Bit-faithful port of `_dynamic_mxfp8_quant_kernel` quant logic, taking fp32 input.

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
# dynamic_mxfp8_quant
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
    y_kern, s_kern = dynamic_mxfp8_quant(x)

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
    _y, s = dynamic_mxfp8_quant(x, scale=scale_pre)
    assert s.data_ptr() == scale_pre.data_ptr()

    _y_ref, s_ref = torch_mxfp8_quant_from_fp32(x.to(torch.float32))
    torch.testing.assert_close(s, s_ref)


def test_per_1x32_mxfp8_quant_multidim():
    """Wrapper folds higher dims into M; sanity-check 3D input."""
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 not supported on this arch")
    torch.cuda.empty_cache()
    torch.manual_seed(0)

    B, M, K = 4, 8, 128
    x = torch.randn((B, M, K), dtype=torch.bfloat16, device="cuda")
    y, s = dynamic_mxfp8_quant(x)
    assert y.shape == (B, M, K)
    assert s.shape == (B, M, K // QUANT_BLOCK_SIZE)

    _y_ref, s_ref = torch_mxfp8_quant_from_fp32(x.reshape(-1, K).to(torch.float32))
    torch.testing.assert_close(s.reshape(-1, K // QUANT_BLOCK_SIZE), s_ref)


# -----------------------------------------------------------------------------
# fp8_legacy_to_mxfp8
# -----------------------------------------------------------------------------


def torch_fp8_legacy_to_mxfp8(x_fnuz: torch.Tensor, x_scale_fp32: torch.Tensor):
    """Reference: dequantize fnuz fp8 with the 1x128 fp32 scale, then run
    the standard mxfp8 1x32 quant on the result."""
    _M, _N = x_fnuz.shape
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

    _y_ref, s_ref = torch_fp8_legacy_to_mxfp8(x_fnuz, x_scale_fp32)
    torch.testing.assert_close(s, s_ref)


# -----------------------------------------------------------------------------
# fused_rms_mxfp8_quant
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
    y_kern, s_kern = fused_rms_mxfp8_quant(x, weight, eps)

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
    y, s = fused_rms_mxfp8_quant(x, weight, 1e-5, y=y_pre, scale=s_pre)
    assert y.data_ptr() == y_pre.data_ptr()
    assert s.data_ptr() == s_pre.data_ptr()


# -----------------------------------------------------------------------------
# fused_dual_rmsnorm_mxfp8_quant
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
    yq_kern, sq_kern, yk_kern = fused_dual_rmsnorm_mxfp8_quant(
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

    yq_a, sq_a, yk_a = fused_dual_rmsnorm_mxfp8_quant(q, k, q_weight, k_weight, eps)
    yq_b, sq_b, yk_b = fused_dual_rmsnorm_mxfp8_quant(
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
    """Sanity: the flatten+quant path should match dynamic_mxfp8_quant
    applied to the pre-flattened (M, N1 * N2) tensor."""
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 not supported on this arch")
    torch.cuda.empty_cache()
    torch.manual_seed(19)

    M, N1, N2 = 16, 4, 128
    x = torch.randn((M, N1, N2), dtype=torch.bfloat16, device="cuda")

    y_flat, s_flat = fused_flatten_mxfp8_quant(x)
    y_ref, s_ref = dynamic_mxfp8_quant(x.reshape(M, N1 * N2).contiguous())

    torch.testing.assert_close(s_flat, s_ref)
    torch.testing.assert_close(
        y_flat.view(torch.uint8).to(torch.int32),
        y_ref.view(torch.uint8).to(torch.int32),
        atol=1,
        rtol=0,
    )


# =========================================================================== #
# DSv4 a8w8 producer kernels: the Q pack, and the fused Q+KV producer.
#
# References are torch implementations of the FORMAT, written from its
# definition rather than paraphrased from the kernels, so a layout mistake
# shows up as a byte difference rather than cancelling out.
#
# The aligned record, per token, 640 B:
#
#     [  0, 448)  NoPE, e4m3, one UE8M0 scale per 64-element group
#     [448, 462)  those 7 scale bytes, EACH WRITTEN TWICE
#     [462, 512)  pad
#     [512, 640)  RoPE, bf16, never quantized
#
# Bytes [0, 512) are also exactly a Q row. The duplication is not redundancy:
# the decode kernel's scaled-MMA blocks are 32 elements wide while the quant
# group is 64, so each group's scale is read twice.
#
# Two tests per kernel, not one. The second covers the API contract -- what the
# kernel REFUSES and what it returns -- which asserts on exceptions and shapes
# rather than values, and must not be dragged through the numerical
# parametrization that would re-run it for every shape.
# =========================================================================== #
_NOPE, _ROPE, _QK = 448, 64, 512
_GROUP = 64
_NUM_TILES = _NOPE // _GROUP  # 7
_REC, _SC_IN_REC, _ROPE_IN_REC = 640, 448, 512

# One ULP of bf16 is a relative 2**-8; the bound is 2**-7 to cover a rounding
# that crosses a binade, with an absolute floor so values near zero are not
# judged on a relative scale.
_ULP = dict(rtol=2**-7, atol=1e-5)
_EXACT = dict(rtol=0, atol=0)


def _skip_without_fp8():
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 not supported on this arch")


# --------------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------------- #
def torch_pack_q_ref(q: torch.Tensor):
    """The torch implementation the Q-pack kernel replaced, kept verbatim."""
    lead = q.shape[:-1]
    nope = q[..., :_NOPE].float()
    rope = q[..., _NOPE:].contiguous()
    tiled = nope.reshape(*lead, _NUM_TILES, _GROUP)
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    # amax/fp8_max rounded UP to a power of two, exactly as E8M0 stores it
    scale = torch.pow(
        2.0, torch.clamp_min(tiled.abs().amax(dim=-1) / fp8_max, 1e-4).log2().ceil()
    )
    nope_fp8 = (tiled / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    e8m0 = (scale.log2().round().to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)
    packed = torch.zeros((*lead, _QK), dtype=torch.uint8, device=q.device)
    packed[..., :_NOPE] = nope_fp8.reshape(*lead, _NOPE).view(torch.uint8)
    packed[..., _NOPE : _NOPE + 2 * _NUM_TILES] = e8m0.repeat_interleave(2, dim=-1)
    return packed.view(torch.float8_e4m3fn), rope


def torch_qnorm_rope_ref(q, positions, cos_sin, eps, padded_heads, apply_norm):
    """``[T, H, 512]`` bf16 -> the padded, normed, rotated Q, in fp32."""
    t, h, _ = q.shape
    x = q.float()
    if apply_norm:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    half = _ROPE // 2
    cs = cos_sin[positions.long()]
    cos, sin = cs[:, :half], cs[:, half:]
    e, o = x[..., _NOPE::2], x[..., _NOPE + 1 :: 2]
    ye = e * cos[:, None, :] - o * sin[:, None, :]
    yo = e * sin[:, None, :] + o * cos[:, None, :]
    out = x.clone()
    out[..., _NOPE::2], out[..., _NOPE + 1 :: 2] = ye, yo
    padded = torch.zeros(t, padded_heads, _QK, dtype=torch.float32, device=q.device)
    padded[:, :h] = out
    return padded


def torch_kv_record_ref(kv, positions, cos_sin):
    """``[T, 512]`` bf16 -> ``[T, 640]`` uint8, the aligned record, pad zero.

    The KV side clamps the ABSMAX before dividing, where the Q side clamps the
    ratio. That asymmetry is inherited from the .cu producer this replaces, so
    the reference reproduces it rather than tidying it.
    """
    t = kv.shape[0]
    half = _ROPE // 2
    cs = cos_sin[positions.long()]
    cos, sin = cs[:, :half], cs[:, half:]
    e, o = kv[:, _NOPE::2].float(), kv[:, _NOPE + 1 :: 2].float()
    ye, yo = e * cos - o * sin, e * sin + o * cos
    rope = torch.empty(t, _ROPE, dtype=torch.float32, device=kv.device)
    rope[:, 0::2], rope[:, 1::2] = ye, yo

    nope = kv[:, :_NOPE].float().reshape(t, _NUM_TILES, _GROUP)
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    amax = torch.clamp_min(nope.abs().amax(-1), 1e-4)
    scale = torch.pow(2.0, (amax / fp8_max).log2().ceil())
    q8 = torch.clamp(nope / scale.unsqueeze(-1), -fp8_max, fp8_max).to(
        torch.float8_e4m3fn
    )
    e8m0 = (scale.log2().round().to(torch.int32) + 127).clamp(0, 255).to(torch.uint8)

    rec = torch.zeros(t, _REC, dtype=torch.uint8, device=kv.device)
    rec[:, :_NOPE] = q8.reshape(t, _NOPE).view(torch.uint8)
    rec[:, _SC_IN_REC : _SC_IN_REC + 2 * _NUM_TILES] = e8m0.repeat_interleave(2, dim=-1)
    rec[:, _ROPE_IN_REC:] = rope.to(torch.bfloat16).view(torch.uint8)
    return rec


def _make_inputs(T, H, padded_heads, nb, block, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    q = (torch.randn(T, H, _QK, device=dev) * 0.125).to(torch.bfloat16)
    kv = (torch.randn(T, _QK, device=dev) * 0.4).to(torch.bfloat16)
    # 0xCD, not zero: a record the kernel fails to write shows up as garbage
    # rather than silently matching a zeroed reference.
    cache = torch.full((nb, block, _REC), 0xCD, dtype=torch.uint8, device=dev)
    slot = torch.randperm(nb * block, device=dev, dtype=torch.int32)[:T]
    positions = torch.randint(0, 128, (T,), device=dev, dtype=torch.int64)
    cos_sin = torch.randn(256, _ROPE, device=dev, dtype=torch.float32)
    return q, kv, cache, slot, positions, cos_sin


# --------------------------------------------------------------------------- #
# fused_deepseek_v4_mxfp8_quant_q_pack
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T,H", [(1, 16), (4, 32), (64, 16), (512, 128)])
@pytest.mark.parametrize("mag", [0.0, 1e-3, 1.0, 1e3])
def test_fused_deepseek_v4_mxfp8_quant_q_pack(T, H, mag):
    """Byte-for-byte against the reference, plus the format's invariants.

    Exact, not close-to: the packing is integer work once the exponent is
    chosen, so any difference is a bug rather than rounding. The magnitude
    sweep is what exercises the exponent path, and ``mag = 0`` is the padded /
    empty head: its scale bytes must carry the 1e-4 floor exponent, never 0xFF,
    which is E8M0 NaN -- the decode kernel multiplies a masked score by it and
    0 * NaN would poison the row.
    """
    _skip_without_fp8()
    torch.manual_seed(0)
    q = (torch.randn(T, H, _QK, device="cuda") * mag).to(torch.bfloat16)
    got_p, got_r = fused_deepseek_v4_mxfp8_quant_q_pack(q)
    exp_p, exp_r = torch_pack_q_ref(q)

    u8, exp_u8 = got_p.view(torch.uint8), exp_p.view(torch.uint8)
    torch.testing.assert_close(u8, exp_u8, **_EXACT)
    torch.testing.assert_close(got_r, exp_r, **_EXACT)

    sc = u8[..., _NOPE : _NOPE + 2 * _NUM_TILES]
    assert (sc != 0xFF).all(), "a scale byte is E8M0 NaN"
    # each group's scale appears twice, adjacently -- the operand layout the
    # decode kernel indexes
    torch.testing.assert_close(sc[..., 0::2], sc[..., 1::2], **_EXACT)
    assert (u8[..., _NOPE + 2 * _NUM_TILES :] == 0).all(), "tail not zeroed"


def test_fused_deepseek_v4_mxfp8_quant_q_pack_contract():
    """A row that is not 448 NoPE + 64 RoPE is refused, not silently reshaped."""
    _skip_without_fp8()
    q = torch.zeros(2, 4, 256, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="448 NoPE"):
        fused_deepseek_v4_mxfp8_quant_q_pack(q)


# --------------------------------------------------------------------------- #
# fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T,H,padded", [(1, 16, 16), (8, 16, 32), (37, 128, 128)])
@pytest.mark.parametrize("apply_norm", [False, True])
def test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    T, H, padded, apply_norm
):
    """Q, its pack, and the KV record, against per-side references.

    With the norm OFF both sides do the identical arithmetic, so everything the
    LAYOUT determines is compared exactly. Only the rotated half gets a one-ULP
    bound, because ``a*cos - b*sin`` may or may not contract into an FMA and
    that is not this kernel's choice. With the norm ON the reduction order for
    sum(x*x) differs from torch's as well, so every value moves in its last
    bits and the whole comparison relaxes to one ULP -- and the pack, whose
    exponent could tip at a boundary, is checked by dequantizing instead.

    Also covers the two behaviours that have no reference of their own: a
    padded head slot must be zero-filled, and ``slot == -1`` must leave its
    record untouched.
    """
    _skip_without_fp8()
    q, kv, cache, slot, pos, cs = _make_inputs(T, H, padded, nb=4, block=64)
    slot[1::2] = -1  # every other token has no cache row; token 0 stays live
    # (odd indices, so the T=1 case still exercises a real insert)
    before = cache.clone()

    q_out, q_packed, q_rope = (
        fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q, kv, cache, slot, pos, cs, 64, 1e-6, padded, apply_q_norm=apply_norm
        )
    )

    want_f32 = torch_qnorm_rope_ref(q, pos, cs, 1e-6, padded, apply_norm)
    want_q = want_f32.to(torch.bfloat16)
    nope_tol = _ULP if apply_norm else _EXACT
    torch.testing.assert_close(q_out[..., :_NOPE], want_q[..., :_NOPE], **nope_tol)
    torch.testing.assert_close(q_out[..., _NOPE:], want_q[..., _NOPE:], **_ULP)

    # padded slots: zero everywhere, so no 0xFF reaches the decode kernel
    assert (q_packed.view(torch.uint8)[:, H:] == 0).all()
    assert (q_rope[:, H:] == 0).all()

    if apply_norm:
        # dequantized, since a last-bit difference in Q can tip an exponent
        u8 = q_packed.view(torch.uint8)[:, :H]
        nope = (
            u8[..., :_NOPE]
            .view(torch.float8_e4m3fn)
            .float()
            .reshape(T, H, _NUM_TILES, _GROUP)
        )
        exps = u8[..., _NOPE : _NOPE + 2 * _NUM_TILES : 2].to(torch.int32)
        deq = (nope * torch.pow(2.0, (exps - 127).float()).unsqueeze(-1)).reshape(
            T, H, _NOPE
        )
        ref = want_q[:, :H, :_NOPE].float()
        rel = (deq - ref).norm() / ref.norm()
        # a UE8M0 scale rounded UP to a power of two costs up to a mantissa
        # bit, which puts e4m3's RMS in the low percent
        assert rel < 4e-2, f"packed Q drifts from the Q it came from: {rel:.2e}"
    else:
        exp_p, exp_r = torch_pack_q_ref(want_f32[:, :H])
        torch.testing.assert_close(
            q_packed.view(torch.uint8)[:, :H], exp_p.view(torch.uint8), **_EXACT
        )
        torch.testing.assert_close(q_rope[:, :H], exp_r.to(torch.bfloat16), **_ULP)

    rows = cache.reshape(-1, _REC)
    live = slot >= 0
    got = rows[slot[live].long()]
    want_kv = torch_kv_record_ref(kv, pos, cs)[live]
    torch.testing.assert_close(got[:, :_NOPE], want_kv[:, :_NOPE], **_EXACT)
    sl = slice(_SC_IN_REC, _SC_IN_REC + 2 * _NUM_TILES)
    torch.testing.assert_close(got[:, sl], want_kv[:, sl], **_EXACT)
    torch.testing.assert_close(
        got[:, _ROPE_IN_REC:].view(torch.bfloat16),
        want_kv[:, _ROPE_IN_REC:].view(torch.bfloat16),
        **_ULP,
    )

    untouched = torch.ones(rows.shape[0], dtype=torch.bool, device=cache.device)
    untouched[slot[live].long()] = False
    torch.testing.assert_close(
        rows[untouched], before.reshape(-1, _REC)[untouched], **_EXACT
    )


def test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_contract():
    """What it returns without the pack, and what it refuses."""
    _skip_without_fp8()
    q, kv, cache, slot, pos, cs = _make_inputs(4, 16, 16, nb=2, block=64)
    out = fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q, kv, cache, slot, pos, cs, 64, 1e-6, 16, pack_q=False
    )
    assert isinstance(out, torch.Tensor) and out.shape == (4, 16, _QK)

    bad = torch.zeros(2, 64, 584, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="640"):
        fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q, kv, bad, slot, pos, cs, 64, 1e-6, 16
        )


# --------------------------------------------------------------------------- #
# fused_deepseek_v4_quantize_and_insert_k_cache
# fused_deepseek_v4_dequantize_and_gather_k_cache
# --------------------------------------------------------------------------- #
# The writer and the reader are exact inverses up to the UE8M0 quantization,
# and they are the only pair that has to agree on the record layout. Both
# layouts are covered here: the 584-byte packed record, whose scales sit in a
# per-block region after the block's token data, and the 640-byte aligned one,
# whose scales sit inside each record with every group's byte written twice.
# Only the aligned record is used in production, and before these tests it had
# no coverage below the end-to-end level.
_PACKED_REC = 584
_SCALES_PER_REC = _NUM_TILES  # 7 groups of 64 over the 448 NoPE bytes


def _build_paged_k(seq_lens, rec_bytes, block=64, seed=0):
    """K rows plus the paged cache, block table and slot mapping addressing it.

    Physical pages are shuffled and the padding block-table entries are left
    wildly out of range, so a kernel that ignored the translation would fault
    or read garbage rather than quietly pass.
    """
    torch.manual_seed(seed)
    dev = "cuda"
    nblocks = [(s + block - 1) // block for s in seq_lens]
    nb, max_blocks = sum(nblocks), max(nblocks)
    num_tokens = sum(seq_lens)

    k = torch.randn(num_tokens, _QK, dtype=torch.bfloat16, device=dev)
    pages = torch.randperm(nb, device=dev)
    block_table = torch.full(
        (len(seq_lens), max_blocks), -(10**6), dtype=torch.int32, device=dev
    )
    slot = torch.empty(num_tokens, dtype=torch.int64, device=dev)
    b = t = 0
    for r, s in enumerate(seq_lens):
        n = nblocks[r]
        block_table[r, :n] = pages[b : b + n]
        b += n
        pos = torch.arange(s, device=dev)
        blk = block_table[r, pos // block].to(torch.int64)
        slot[t : t + s] = blk * block + pos % block
        t += s

    cache = torch.zeros(nb, block, rec_bytes, dtype=torch.uint8, device=dev)
    return k, cache, slot, block_table


def _torch_decode_cache(cache, block, rec_bytes):
    """Decode every slot of the paged cache to bf16, indexed by physical slot.

    This is the reader's job done in torch, straight off the stored bytes, so
    it checks the reader without reproducing the writer's choice of scale.
    """
    nb = cache.shape[0]
    raw = cache.reshape(nb, -1)
    if rec_bytes == _REC:  # 640 aligned: scales and RoPE inside the record
        rec = raw.reshape(nb, block, _REC)
        fp8_b = rec[..., :_NOPE]
        # one byte per 32 columns, so every 64-element group's scale is twice
        scale_b = rec[..., _SC_IN_REC : _SC_IN_REC + 2 * _SCALES_PER_REC][..., 0::2]
        rope_b = rec[..., _ROPE_IN_REC:_REC]
    else:  # 584 packed: token data first, then a per-block scale region
        data = raw[:, : block * 576].reshape(nb, block, 576)
        fp8_b = data[..., :_NOPE]
        rope_b = data[..., _NOPE:576]
        scale_b = raw[:, block * 576 : block * 576 + block * 8].reshape(nb, block, 8)[
            ..., :_SCALES_PER_REC
        ]

    fp8 = fp8_b.contiguous().view(torch.float8_e4m3fn).to(torch.float32)
    scale = torch.exp2(scale_b.to(torch.int32).to(torch.float32) - 127.0)
    nope = (fp8.reshape(nb, block, _SCALES_PER_REC, _GROUP) * scale[..., None]).reshape(
        nb, block, _NOPE
    )
    rope = rope_b.contiguous().view(torch.bfloat16).to(torch.float32)
    return torch.cat([nope, rope], dim=-1).to(torch.bfloat16).reshape(nb * block, _QK)


def _gather_reference(decoded, seq_lens, slot, gather_lens, offset, out_shape):
    """Lay the decoded slots out the way the reader is asked to gather them.

    ``gather_lens`` is a sliding window, not a prefix: a request gathers its
    LAST ``gather_len`` positions, starting at ``seq_len - gather_len``. The
    SWA caller relies on that, and taking the prefix instead still produces
    well-formed output, so only a reference that gets this right can catch it.
    """
    want = torch.zeros(out_shape, dtype=torch.bfloat16, device=decoded.device)
    t = 0
    for r, s in enumerate(seq_lens):
        n = s if gather_lens is None else gather_lens[r]
        want[r, offset : offset + n] = decoded[slot[t + s - n : t + s]]
        t += s
    return want


@pytest.mark.parametrize("rec_bytes", [_PACKED_REC, _REC], ids=["packed584", "aligned640"])
@pytest.mark.parametrize(
    "seq_lens,gather_lens,offset",
    [
        ([1], None, 0),
        ([17, 1, 128], None, 0),
        ([130, 65, 3], [64, 32, 1], 16),
        ([255, 256], [255, 256], 2),
    ],
)
def test_fused_deepseek_v4_k_cache_roundtrip(rec_bytes, seq_lens, gather_lens, offset):
    """Write K into the paged cache, read it back, on both record layouts.

    Two independent claims. The reader must reproduce what torch decodes from
    the stored bytes -- that is the layout and the gather. And the stored
    bytes must still be the input to within one UE8M0-scaled FP8 step -- that
    is the writer. Splitting them means a layout bug cannot hide behind the
    quantization tolerance.
    """
    _skip_without_fp8()
    block = 64
    k, cache, slot, block_table = _build_paged_k(seq_lens, rec_bytes, block)
    fused_deepseek_v4_quantize_and_insert_k_cache(k, cache, slot, block)

    max_len = max(gather_lens or seq_lens)
    shape = (len(seq_lens), offset + max_len + 3, _QK)
    got = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    fused_deepseek_v4_dequantize_and_gather_k_cache(
        got,
        cache,
        torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        None
        if gather_lens is None
        else torch.tensor(gather_lens, dtype=torch.int32, device="cuda"),
        block_table,
        block,
        offset,
    )

    decoded = _torch_decode_cache(cache, block, rec_bytes)
    want = _gather_reference(decoded, seq_lens, slot, gather_lens, offset, shape)
    for r, s in enumerate(seq_lens):
        n = s if gather_lens is None else gather_lens[r]
        torch.testing.assert_close(
            got[r, offset : offset + n], want[r, offset : offset + n], **_ULP
        )

    # The writer: every NoPE group must survive to within one FP8 step at that
    # group's own scale, and the RoPE half is bf16 stored verbatim.
    live = decoded[slot]
    groups = k[:, :_NOPE].reshape(-1, _SCALES_PER_REC, _GROUP).to(torch.float32)
    amax = groups.abs().amax(dim=-1, keepdim=True)
    err = (live[:, :_NOPE].to(torch.float32).reshape_as(groups) - groups).abs()
    # e4m3 keeps 3 mantissa bits, so one step at the group's scale is amax/2^3;
    # allow a half-step of rounding on top of it.
    assert (err <= amax / 8 + 1e-6).all(), (err / amax.clamp(min=1e-9)).max().item()
    torch.testing.assert_close(live[:, _NOPE:], k[:, _NOPE:], **_EXACT)


@pytest.mark.parametrize("seq_lens", [[1], [130, 65, 3]])
def test_fused_deepseek_v4_k_cache_layouts_agree(seq_lens):
    """The two record layouts must decode to the same values, bit for bit.

    They differ only in where the bytes sit, so any disagreement is a geometry
    bug in the writer or the reader -- and this compares them with no
    tolerance at all, which the round-trip test cannot do.
    """
    _skip_without_fp8()
    block = 64
    outs = []
    for rec_bytes in (_PACKED_REC, _REC):
        k, cache, slot, block_table = _build_paged_k(seq_lens, rec_bytes, block, seed=7)
        fused_deepseek_v4_quantize_and_insert_k_cache(k, cache, slot, block)
        out = torch.zeros(
            (len(seq_lens), max(seq_lens), _QK), dtype=torch.bfloat16, device="cuda"
        )
        fused_deepseek_v4_dequantize_and_gather_k_cache(
            out,
            cache,
            torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
            None,
            block_table,
            block,
            0,
        )
        outs.append(out)
    for r, s in enumerate(seq_lens):
        torch.testing.assert_close(outs[0][r, :s], outs[1][r, :s], **_EXACT)


def test_fused_deepseek_v4_k_cache_contract():
    """What the writer refuses."""
    _skip_without_fp8()
    cache = torch.zeros(2, 64, _REC, dtype=torch.uint8, device="cuda")
    slot = torch.zeros(4, dtype=torch.int64, device="cuda")
    with pytest.raises(AssertionError, match="512"):
        fused_deepseek_v4_quantize_and_insert_k_cache(
            torch.zeros(4, 256, dtype=torch.bfloat16, device="cuda"), cache, slot, 64
        )
    with pytest.raises(AssertionError, match="bf16"):
        fused_deepseek_v4_quantize_and_insert_k_cache(
            torch.zeros(4, _QK, dtype=torch.float16, device="cuda"), cache, slot, 64
        )


# --------------------------------------------------------------------------- #
# fused_deepseek_v4_compress_norm_rope_store / fused_deepseek_v4_compress_norm_rope_store_two_stage
# --------------------------------------------------------------------------- #
# These two launchers must produce the same cache, by different routes: the
# single-pass kernel does compress+norm+RoPE+quant+store in one go, while the
# two-stage one fans the compression across CUs into a scratch buffer and then
# finalises it. Both carry their own copy of the full-width GPT-J RoPE, which
# is aiter's and has no counterpart in vLLM -- so nothing but this compares
# them. A/B rather than a reference: the two routes agreeing bit for bit
# pins the rotation, the normalisation and the record layout at once.
#
# The index pattern (positions, block_table, token_to_req_indices) is the
# benchmark's, which is known in-bounds; only the DATA is randomised, plus
# distinct slots so the tokens do not race for one record. Varying positions
# would need the compress gather's addressing pinned down first, and an
# out-of-bounds read here costs a GPU reset, so that is left alone.
_CR128 = dict(
    head_dim=512, rope_dim=64, quant_block=64, token_stride=_REC, scale_dim=8,
    state_width=512, compress_ratio=128, overlap=False,
    block_size=8, state_mid=8, kv_page=2, rows_per_block=64,
    block_table_width=64, cos_sin_rows=4096, rms_eps=1e-6,
)



def _emitting_positions(tokens, cfg, dev):
    """Positions that actually produce a record, and stay in bounds doing it.

    A token is skipped unless ``(position + 1) % compress_ratio == 0``, which
    is why an all-zero positions vector -- the benchmark's -- writes nothing
    at all. The gather then walks ``position - compress_ratio + 1 .. position``
    and indexes ``block_table[req, pos // block_size]``, so the largest
    position usable here is ``block_table_width * block_size - 1`` = 511.
    Cycling the four emitting positions below keeps every load inside both
    the block table and state_cache[0], which block_table being zero selects.
    """
    cr, bs, w = cfg["compress_ratio"], cfg["block_size"], cfg["block_table_width"]
    valid = [p for p in range(cr - 1, w * bs, cr)]
    assert valid, "no emitting position fits the block table"
    return torch.tensor(
        [valid[i % len(valid)] for i in range(tokens)],
        dtype=torch.int64, device=dev,
    )

def _compressor_inputs(tokens, cfg, seed=0):
    from types import SimpleNamespace

    torch.manual_seed(seed)
    dev = "cuda"
    hd, sw = cfg["head_dim"], cfg["state_width"]

    state_cache = torch.randn(
        tokens, cfg["state_mid"], 2 * sw, dtype=torch.float32, device=dev
    )
    kv_backing = torch.zeros(
        tokens, cfg["rows_per_block"], cfg["token_stride"],
        dtype=torch.uint8, device=dev,
    )
    kv_cache = kv_backing[:, : cfg["kv_page"], :]

    # a real rotation, not the identity the benchmark uses
    ang = torch.randn(cfg["cos_sin_rows"], cfg["rope_dim"] // 2, device=dev)
    cos_sin_cache = torch.cat([ang.cos(), ang.sin()], dim=-1).to(torch.float32)

    # distinct records: slot s lands in block s // kv_page, row s % kv_page,
    # and s < tokens <= tokens * kv_page keeps every one inside kv_cache.
    slot_mapping = torch.arange(tokens, dtype=torch.int64, device=dev)

    return kv_backing, SimpleNamespace(
        state_cache=state_cache,
        num_actual=tokens,
        token_to_req_indices=torch.zeros(tokens, dtype=torch.int32, device=dev),
        positions=_emitting_positions(tokens, cfg, dev),
        slot_mapping=slot_mapping,
        block_table=torch.zeros(
            tokens, cfg["block_table_width"], dtype=torch.int32, device=dev
        ),
        block_size=cfg["block_size"],
        state_width=sw,
        cos_sin_cache=cos_sin_cache,
        kv_cache=kv_cache,
        k_cache_metadata=SimpleNamespace(slot_mapping=slot_mapping),
        pdl_kwargs={},
        head_dim=hd,
        rope_head_dim=cfg["rope_dim"],
        compress_ratio=cfg["compress_ratio"],
        overlap=cfg["overlap"],
        use_fp4_cache=False,
        rms_norm_weight=torch.randn(hd, dtype=torch.bfloat16, device=dev),
        rms_norm_eps=cfg["rms_eps"],
        quant_block=cfg["quant_block"],
        token_stride=cfg["token_stride"],
        scale_dim=cfg["scale_dim"],
    )


@pytest.mark.parametrize("tokens", [1, 16, 64])
def test_compress_norm_rope_store_two_stage_matches_single_pass(tokens):
    """The split compressor must write exactly what the single-pass one does."""
    _skip_without_fp8()
    cfg = _CR128

    backing_a, a = _compressor_inputs(tokens, cfg)
    fused_deepseek_v4_compress_norm_rope_store(**vars(a))

    backing_b, b = _compressor_inputs(tokens, cfg)
    torch.testing.assert_close(b.state_cache, a.state_cache, **_EXACT)
    fused_deepseek_v4_compress_norm_rope_store_two_stage(
        **vars(b),
        # 0 decode tokens, so every token takes the two-stage route rather
        # than falling through to the single-pass launcher under test.
        num_decode_tokens=0,
        compress_scratch=torch.zeros(
            tokens, cfg["head_dim"], dtype=torch.float32, device="cuda"
        ),
    )
    torch.cuda.synchronize()

    live = a.slot_mapping.cpu().tolist()
    ra = backing_a.reshape(-1, cfg["token_stride"])
    rb = backing_b.reshape(-1, cfg["token_stride"])
    for s in live:
        blk, row = divmod(s, cfg["kv_page"])
        i = blk * cfg["rows_per_block"] + row
        torch.testing.assert_close(rb[i], ra[i], **_EXACT)

    # and something was actually written, so an all-zero pass cannot "agree"
    assert ra[[divmod(s, cfg["kv_page"])[0] * cfg["rows_per_block"]
               + divmod(s, cfg["kv_page"])[1] for s in live]].any()
