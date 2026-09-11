# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale
from aiter.ops.triton.quant.quant_fp8_blockwise import quant_fp8_blockwise

# DeepSeek-V4.1-Flash dense-weight format: fp8 e4m3 values, 2-D [32, 32] weight
# block scales and per-token 1x32 activation scales, both ue8m0.
BLOCK_SIZE = 32
FP8_MAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn
E8M0_DTYPE = torch.float8_e8m0fnu


def get_x_vals():
    return [
        (1280, 5120),
        (32768, 1280),
        (512, 5120),
        (5120, 8192),
        (4096, 1280),
        (2304, 5120),
        (5120, 2304),
        (25600, 6144),
    ]


def e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    return torch.exp2(scale.view(torch.uint8).to(torch.float32) - 127.0)


def torch_act_quant(x: torch.Tensor, block_size: int = BLOCK_SIZE):
    """Reference for DeepSeek-V4.1-Flash inference/kernel.py::act_quant (ue8m0)."""
    M, K = x.shape
    pad = (-K) % block_size
    xb = torch.nn.functional.pad(x.float(), (0, pad)).reshape(M, -1, block_size)
    amax = xb.abs().amax(dim=-1).clamp(min=1e-4)
    s = torch.exp2(torch.ceil(torch.log2(amax * (1.0 / FP8_MAX))))
    xq = (xb / s[:, :, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return xq.reshape(M, -1)[:, :K].contiguous(), s


def run_torch(x_q, x_scale, w, w_scale, dtype=torch.bfloat16):
    """Dequantize both operands to fp32 and matmul: out = x @ w.T."""
    _M, K = x_q.shape
    N = w.shape[0]
    xs = e8m0_to_fp32(x_scale).repeat_interleave(BLOCK_SIZE, dim=1)[:, :K]
    ws = e8m0_to_fp32(w_scale)
    ws = ws.repeat_interleave(BLOCK_SIZE, dim=0).repeat_interleave(BLOCK_SIZE, dim=1)
    out = (x_q.float() * xs) @ (w.float() * ws[:N, :K]).T
    return out.to(dtype)


def generate_gemm_a8w8_blockscale_ue8m0_inputs(M: int, N: int, K: int, seed: int = 0):
    """Returns (x, x_q, x_scale, w, w_scale) with ue8m0 scales on both sides."""
    torch.manual_seed(seed)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = (torch.randn((N, K), dtype=torch.bfloat16, device="cuda") / 8).to(FP8_DTYPE)
    scale_n = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    scale_k = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    w_scale = torch.randint(
        120, 132, (scale_n, scale_k), dtype=torch.uint8, device="cuda"
    ).view(E8M0_DTYPE)
    x_q, x_scale = quant_fp8_blockwise(
        x,
        block_size=BLOCK_SIZE,
        fp8_max=FP8_MAX,
        quant_dtype=FP8_DTYPE,
        scale_fmt="ue8m0",
    )
    return x, x_q, x_scale, w, w_scale


@pytest.mark.parametrize("M", [1, 4, 64, 1024, 4096])
@pytest.mark.parametrize("N, K", get_x_vals())
def test_gemm_a8w8_blockscale_ue8m0(M, N, K):
    torch.cuda.empty_cache()
    _, x_q, x_scale, w, w_scale = generate_gemm_a8w8_blockscale_ue8m0_inputs(M, N, K)

    ref = run_torch(x_q, x_scale, w, w_scale, torch.float32)
    out = gemm_a8w8_blockscale(x_q, w, x_scale, w_scale, torch.bfloat16)

    atol = 1e-2 * ref.abs().max().item()
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=2e-2)


@pytest.mark.parametrize("M", [1, 64, 1024])
@pytest.mark.parametrize("K", [1280, 5120])
def test_act_quant_ue8m0(M, K):
    """The ue8m0 activation quant must match the reference bit for bit."""
    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda") * 3
    x_q, x_scale = quant_fp8_blockwise(
        x,
        block_size=BLOCK_SIZE,
        fp8_max=FP8_MAX,
        quant_dtype=FP8_DTYPE,
        scale_fmt="ue8m0",
    )
    x_q_ref, x_scale_ref = torch_act_quant(x)

    assert x_scale.dtype == E8M0_DTYPE
    torch.testing.assert_close(e8m0_to_fp32(x_scale), x_scale_ref, atol=0, rtol=0)
    torch.testing.assert_close(
        x_q.view(torch.uint8), x_q_ref.view(torch.uint8), atol=0, rtol=0
    )


@pytest.mark.parametrize("M, N, K", [(64, 1280, 5120), (1024, 512, 5120)])
def test_gemm_a8w8_blockscale_32_fp32_scales(M, N, K):
    """Same 32-wide groups as fp32 scales: several scale steps per K tile."""
    torch.cuda.empty_cache()
    _, x_q, x_scale, w, w_scale = generate_gemm_a8w8_blockscale_ue8m0_inputs(M, N, K)

    ref = run_torch(x_q, x_scale, w, w_scale, torch.float32)
    out = gemm_a8w8_blockscale(
        x_q, w, e8m0_to_fp32(x_scale), e8m0_to_fp32(w_scale), torch.bfloat16
    )

    atol = 1e-2 * ref.abs().max().item()
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=2e-2)


@pytest.mark.parametrize("M, N, K", [(64, 1280, 1296), (1024, 2304, 5152)])
def test_gemm_a8w8_blockscale_ue8m0_partial_k_group(M, N, K):
    """K need not be a multiple of 32: the tail group is masked on both sides."""
    torch.cuda.empty_cache()
    _, x_q, x_scale, w, w_scale = generate_gemm_a8w8_blockscale_ue8m0_inputs(M, N, K)

    ref = run_torch(x_q, x_scale, w, w_scale, torch.float32)
    out = gemm_a8w8_blockscale(x_q, w, x_scale, w_scale, torch.bfloat16)

    atol = 1e-2 * ref.abs().max().item()
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=2e-2)


def test_gemm_a8w8_blockscale_ue8m0_rejects_gluon():
    _, x_q, x_scale, w, w_scale = generate_gemm_a8w8_blockscale_ue8m0_inputs(
        64, 512, 1280
    )
    with pytest.raises(AssertionError):
        gemm_a8w8_blockscale(x_q, w, x_scale, w_scale, torch.bfloat16, backend="gluon")
