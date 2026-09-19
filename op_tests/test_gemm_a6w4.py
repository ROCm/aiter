# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.gemm_op_a6w4 import (
    _select_gemm_a6w4_kernel,
    gemm_a6w4,
    gemm_a6w4_asm,
    mxfp4_gemm_pack_size,
    quant_mxfp4_gemm,
)
from aiter.ops.gemm_op_a6w6 import (
    _ceil,
    _rotate_k32_torch,
    dequant_mxfp6_torch,
    mxfp6_gemm_pack_size,
    quant_mxfp6_gemm,
    quant_mxfp6_torch,
)
from aiter.ops.quant import per_1x32_f4_quant
from aiter.utility import dtypes, fp4_utils
from aiter.utility.mx_types import MX_DEFAULT_ROUND_MODE

MFMA32_SMALL_KERNEL = "f6f4gemm_m32_s0_a4_nt_kernel_func"
MFMA32_SWZ0_KERNEL = "f6f4gemm_m32_s0_a4_t_kernel_func"
MFMA32_GROUPED_KERNEL = "f6f4gemm_m32_s3_a4_t_kernel_func"
MFMA32_LONG_K_KERNEL = "f6f4gemm_m32_s3_a5_t_kernel_func"


def _is_gfx950() -> bool:
    try:
        return torch.cuda.is_available() and get_gfx_runtime() == "gfx950"
    except (KeyError, RuntimeError):
        return False


requires_gfx950 = pytest.mark.skipif(not _is_gfx950(), reason="A6W4 requires gfx950")


def _quantized_reference(
    x: torch.Tensor,
    w: torch.Tensor,
    round_mode: int,
) -> torch.Tensor:
    K = x.shape[1]
    padK = _ceil(K, 128)
    x = F.pad(x, (0, padK - K))
    w = F.pad(w, (0, padK - K))

    x_codes, x_scales = quant_mxfp6_torch(x)
    x_dequant = dequant_mxfp6_torch(x_codes, x_scales)

    w_rotated = _rotate_k32_torch(w).to(torch.bfloat16)
    w_codes, w_scales = per_1x32_f4_quant(
        w_rotated,
        quant_dtype=dtypes.fp4x2,
        round_mode=round_mode,
    )
    w_dequant = fp4_utils.mxfp4_to_f32(w_codes)
    w_scale_f32 = fp4_utils.e8m0_to_f32(w_scales.view(torch.uint8))
    w_dequant *= w_scale_f32.repeat_interleave(32, dim=1)
    return x_dequant @ w_dequant.T


@pytest.mark.parametrize(
    "shape,kernel_name,round_mode",
    [
        ((256, 256, 128), None, MX_DEFAULT_ROUND_MODE),
        ((256, 256, 256), MFMA32_SMALL_KERNEL, 2),
        ((512, 768, 256), MFMA32_SMALL_KERNEL, MX_DEFAULT_ROUND_MODE),
        ((513, 769, 257), MFMA32_SMALL_KERNEL, MX_DEFAULT_ROUND_MODE),
        ((257, 513, 129), MFMA32_SMALL_KERNEL, MX_DEFAULT_ROUND_MODE),
        ((257, 513, 513), MFMA32_SMALL_KERNEL, MX_DEFAULT_ROUND_MODE),
        ((2048, 512, 256), None, MX_DEFAULT_ROUND_MODE),
        ((2048, 256, 512), None, MX_DEFAULT_ROUND_MODE),
        ((9450, 5120, 5120), None, MX_DEFAULT_ROUND_MODE),
        ((9450, 13824, 5120), None, MX_DEFAULT_ROUND_MODE),
        ((9450, 5120, 13824), None, MX_DEFAULT_ROUND_MODE),
    ],
)
@torch.no_grad()
@requires_gfx950
def test_gemm_a6w4_matches_quantized_reference(shape, kernel_name, round_mode):
    M, N, K = shape
    torch.manual_seed(M + N + K)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")

    x_packed, x_scales = quant_mxfp6_gemm(x)
    w_packed, w_scales = quant_mxfp4_gemm(w, round_mode=round_mode)
    out = gemm_a6w4(
        x_packed,
        w_packed,
        x_scales,
        w_scales,
        M,
        N,
        K,
        kernelName=kernel_name,
    )

    quantized_ref = _quantized_reference(x, w, round_mode)
    bf16_ref = x.float() @ w.float().T
    kernel_cosine = F.cosine_similarity(
        out.float().flatten(),
        quantized_ref.flatten(),
        dim=0,
    ).item()
    bf16_cosine = F.cosine_similarity(
        out.float().flatten(),
        bf16_ref.flatten(),
        dim=0,
    ).item()
    relative_l2 = ((out.float() - quantized_ref).norm() / quantized_ref.norm()).item()
    norm_ratio = (out.float().norm() / quantized_ref.norm()).item()

    assert out.shape == (M, N)
    assert out.is_contiguous() == (N % 256 == 0)
    assert torch.isfinite(out).all()
    assert kernel_cosine > 0.9999
    assert relative_l2 < 0.01
    assert abs(norm_ratio - 1.0) < 0.01
    assert bf16_cosine > 0.985


@torch.no_grad()
@requires_gfx950
def test_a6w4_mfma32_variants_match_swizzle0_bitwise():
    M, N, K = 257, 513, 129
    torch.manual_seed(M + N + K)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    x_packed, x_scales = quant_mxfp6_gemm(x)
    w_packed, w_scales = quant_mxfp4_gemm(w)

    padM, padN, padK = _ceil(M, 256), _ceil(N, 256), _ceil(K, 128)
    baseline = torch.empty((padM, padN), dtype=torch.bfloat16, device="cuda")
    gemm_a6w4_asm(
        x_packed,
        w_packed,
        x_scales,
        w_scales,
        baseline,
        padK,
        MFMA32_SWZ0_KERNEL,
    )
    for kernel_name in (
        MFMA32_SMALL_KERNEL,
        MFMA32_GROUPED_KERNEL,
        MFMA32_LONG_K_KERNEL,
    ):
        actual = torch.empty_like(baseline)
        gemm_a6w4_asm(
            x_packed,
            w_packed,
            x_scales,
            w_scales,
            actual,
            padK,
            kernel_name,
        )
        assert torch.equal(actual[:M, :N], baseline[:M, :N]), kernel_name


@torch.no_grad()
@requires_gfx950
def test_a6w4_long_k_path_matches_swizzle0_bitwise():
    M, N, K = 256, 256, 6272
    torch.manual_seed(K)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    x_packed, x_scales = quant_mxfp6_gemm(x)
    w_packed, w_scales = quant_mxfp4_gemm(w)
    baseline = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty_like(baseline)
    gemm_a6w4_asm(
        x_packed, w_packed, x_scales, w_scales, baseline, K, MFMA32_SWZ0_KERNEL
    )
    gemm_a6w4_asm(
        x_packed,
        w_packed,
        x_scales,
        w_scales,
        actual,
        K,
        MFMA32_LONG_K_KERNEL,
    )
    assert torch.equal(actual, baseline)


def test_a6w4_dispatch_respects_grouped_kernel_bounds():
    assert _select_gemm_a6w4_kernel(512, 5120, 5120, None) == MFMA32_SMALL_KERNEL
    assert _select_gemm_a6w4_kernel(9450, 5120, 5120, None) == MFMA32_GROUPED_KERNEL
    assert _select_gemm_a6w4_kernel(9450, 13824, 5120, None) == MFMA32_GROUPED_KERNEL
    assert _select_gemm_a6w4_kernel(9450, 5120, 13824, None) == MFMA32_LONG_K_KERNEL
    assert _select_gemm_a6w4_kernel(9450, 27648, 5120, None) == MFMA32_SWZ0_KERNEL
    assert _select_gemm_a6w4_kernel(131073, 13824, 5120, None) == MFMA32_SWZ0_KERNEL


@torch.no_grad()
@requires_gfx950
def test_a6w4_asm_rejects_malformed_or_misaligned_buffers():
    M = N = 256
    K = 128
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    x_packed, x_scales = quant_mxfp6_gemm(x)
    w_packed, w_scales = quant_mxfp4_gemm(w)
    out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(RuntimeError, match="buffer sizes"):
        gemm_a6w4_asm(x_packed[:-1], w_packed, x_scales, w_scales, out, K)
    overallocated_x = torch.empty(
        x_packed.numel() + 16, dtype=torch.uint8, device=x_packed.device
    )
    with pytest.raises(RuntimeError, match="buffer sizes"):
        gemm_a6w4_asm(overallocated_x, w_packed, x_scales, w_scales, out, K)

    storage = torch.empty(
        x_packed.numel() + 1, dtype=torch.uint8, device=x_packed.device
    )
    misaligned_x = storage[1:]
    misaligned_x.copy_(x_packed)
    with pytest.raises(RuntimeError, match="aligned to 16 bytes"):
        gemm_a6w4_asm(misaligned_x, w_packed, x_scales, w_scales, out, K)


@torch.no_grad()
@requires_gfx950
def test_a6w4_compiles_fullgraph():
    def quantize_and_gemm(activation, packed_weight, weight_scale, M, N, K):
        packed_activation, activation_scale = quant_mxfp6_gemm(activation)
        return gemm_a6w4(
            packed_activation,
            packed_weight,
            activation_scale,
            weight_scale,
            M,
            N,
            K,
        )

    compiled = torch.compile(quantize_and_gemm, dynamic=True, fullgraph=True)
    for M, N, K in (
        (257, 513, 129),
        (2048, 512, 256),
        (300, 513, 129),
        (512, 5120, 5120),
        (511, 5120, 5120),
    ):
        x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
        w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
        w_packed, w_scales = quant_mxfp4_gemm(w)
        args = (x, w_packed, w_scales, M, N, K)
        eager = quantize_and_gemm(*args)
        actual = compiled(*args)
        assert torch.equal(actual, eager)


def test_a6w4_rejects_unsupported_launch_contracts_before_allocation():
    placeholder = torch.empty(0, dtype=torch.uint8)
    args = (placeholder,) * 4
    with pytest.raises(ValueError, match="alpha=1.0"):
        gemm_a6w4(*args, 1, 1, 1, alpha=0.5)
    with pytest.raises(ValueError, match="2 GiB"):
        gemm_a6w4(*args, 65792, 16384, 128)
    with pytest.raises(ValueError, match="int32"):
        gemm_a6w4(*args, 1, 1, (1 << 32) + 128)
    with pytest.raises(ValueError, match="padded K"):
        gemm_a6w4(*args, 1, 1, (1 << 31) - 1)


@pytest.mark.parametrize("pack_size", [mxfp4_gemm_pack_size, mxfp6_gemm_pack_size])
def test_mixed_pack_sizes_reject_invalid_or_oversized_shapes(pack_size):
    with pytest.raises(ValueError, match="positive dimensions"):
        pack_size(0, 128)
    with pytest.raises(ValueError, match="2 GiB"):
        pack_size(65536, 65536)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
