# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import _is_gluon_available
from aiter.ops.triton.gemm.batched.batched_gemm_afp8wfp8 import (
    batched_gemm_afp8wfp8_preshuffle,
)

SCALE_GROUP_SIZE = 128

_requires_gluon_preshuffle = pytest.mark.skipif(
    not _is_gluon_available(),
    reason="batched_gemm_afp8wfp8_preshuffle is gfx1250 (gluon) only",
)


def generate_batched_gemm_afp8wfp8_preshuffle_inputs(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    token_major: bool = True,
    x_scale_transposed: bool = False,
    output: bool = False,
    seed: int = 0,
):
    """
    Returns:
        - x: shape (B, M, K), a view of (M, B, K) when token_major
        - w: shape (B, N, K)
        - w_shuffled: shape (B, N // 16, K * 16)
        - x_scales: shape (B, M, K // 128), or (B, K // 128, M) when x_scale_transposed
        - w_scales: shape (B, N // 128, K // 128)
        - y: shape (B, M, N), a view of (M, B, N) when token_major, or None
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape_x = (M, B, K) if token_major else (B, M, K)
    x = (torch.rand(shape_x, device="cuda", generator=g) * 4 - 2).to(
        torch.float8_e4m3fn
    )
    x_scales = torch.randint(
        124,
        129,
        shape_x[:2] + (K // SCALE_GROUP_SIZE,),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    w = (torch.rand((B, N, K), device="cuda", generator=g) * 4 - 2).to(
        torch.float8_e4m3fn
    )
    w_scales = torch.randint(
        124,
        129,
        (B, -(-N // SCALE_GROUP_SIZE), K // SCALE_GROUP_SIZE),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    if token_major:
        x, x_scales = x.transpose(0, 1), x_scales.transpose(0, 1)
    if x_scale_transposed:
        x_scales = x_scales.transpose(1, 2).contiguous()
    w_shuffled = shuffle_weight(w.view(torch.uint8)).view(B, N // 16, K * 16)

    y = None
    if output:
        shape_y = (M, B, N) if token_major else (B, M, N)
        y = torch.empty(shape_y, dtype=dtype, device=x.device)
        if token_major:
            y = y.transpose(0, 1)

    return x, w, w_shuffled.view(torch.float8_e4m3fn), x_scales, w_scales, y


def run_torch(x, w, x_scales, w_scales, x_scale_transposed=False):
    N = w.shape[1]
    if x_scale_transposed:
        x_scales = x_scales.transpose(1, 2)
    x_f32 = x.float() * torch.exp2(x_scales.float() - 127).repeat_interleave(
        SCALE_GROUP_SIZE, -1
    )
    w_scales_f32 = (
        torch.exp2(w_scales.float() - 127)
        .repeat_interleave(SCALE_GROUP_SIZE, 1)[:, :N]
        .repeat_interleave(SCALE_GROUP_SIZE, 2)
    )
    return torch.bmm(x_f32, (w.float() * w_scales_f32).transpose(1, 2))


def preshuffle_check(y, ref):
    err = ((y.float() - ref).norm() / ref.norm()).item()
    assert err < 1e-2, f"rel_l2 {err:.3e}"
    tol = 2e-2 * ref.abs().max().item()
    assert (y.float() - ref).abs().max().item() <= tol


def get_x_vals():
    return [
        (2, 1, 1024, 256),
        (3, 31, 1024, 512),
        (3, 150, 2048, 1024),
        (4, 1000, 1024, 4096),
        (16, 257, 1024, 4096),
        (16, 512, 1024, 4096),
        (16, 2048, 1024, 4096),
        (16, 16384, 1024, 4096),
    ]


@_requires_gluon_preshuffle
@pytest.mark.parametrize("B, M, N, K", get_x_vals())
@pytest.mark.parametrize("token_major", [True, False])
@pytest.mark.parametrize("x_scale_transposed", [False, True])
@pytest.mark.parametrize("output", [True, False])
def test_batched_gemm_afp8wfp8_preshuffle(
    B, M, N, K, token_major, x_scale_transposed, output
):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, w_shuffled, x_scales, w_scales, y = (
        generate_batched_gemm_afp8wfp8_preshuffle_inputs(
            B,
            M,
            N,
            K,
            torch.bfloat16,
            token_major=token_major,
            x_scale_transposed=x_scale_transposed,
            output=output,
        )
    )
    out = batched_gemm_afp8wfp8_preshuffle(
        x,
        w_shuffled,
        x_scales,
        w_scales,
        y=y,
        is_x_scale_transposed=x_scale_transposed,
    )
    if output:
        assert out.data_ptr() == y.data_ptr()
    preshuffle_check(out, run_torch(x, w, x_scales, w_scales, x_scale_transposed))


@_requires_gluon_preshuffle
def test_batched_gemm_afp8wfp8_preshuffle_rejects_bad_args():
    x, _, w_shuffled, x_scales, w_scales, _ = (
        generate_batched_gemm_afp8wfp8_preshuffle_inputs(
            2, 64, 1024, 512, torch.bfloat16
        )
    )
    with pytest.raises(AssertionError):  # row-major x scales need the flag
        batched_gemm_afp8wfp8_preshuffle(x, w_shuffled, x_scales, w_scales)
    with pytest.raises(ValueError):  # w scales of the wrong shape
        batched_gemm_afp8wfp8_preshuffle(
            x,
            w_shuffled,
            x_scales,
            w_scales[:, :, :1],
            is_x_scale_transposed=False,
        )
    with pytest.raises(AssertionError):  # w not viewed as (B, N // 16, K * 16)
        batched_gemm_afp8wfp8_preshuffle(
            x,
            w_shuffled.reshape(2, 1024, 512),
            x_scales,
            w_scales,
            is_x_scale_transposed=False,
        )
