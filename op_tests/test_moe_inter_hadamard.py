# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Block Hadamard of the SwiGLU output between the two MoE GEMMs.

An online-rotated MXFP4 checkpoint stores both expert GEMMs pre-rotated along
K. ``inter_hadamard`` rotates the stage-1 activation before stage 2 quantizes
or consumes it, so ``(y H)(w H)^T = y w^T``.
"""

import inspect

import pytest
import torch

from aiter.ops.hadamard import block_hadamard_rotate, hadamard_matrix


def test_rotation_preserves_gemm():
    y = torch.randn(4, 256, dtype=torch.float32)
    w = torch.randn(64, 256, dtype=torch.float32)
    y_rot = block_hadamard_rotate(y, 128)
    w_rot = block_hadamard_rotate(w, 128)
    torch.testing.assert_close(y_rot @ w_rot.T, y @ w.T, atol=1e-4, rtol=1e-4)


def test_hadamard_is_orthonormal():
    h = hadamard_matrix(128, torch.device("cpu"))
    eye = torch.eye(128)
    torch.testing.assert_close(h.T @ h, eye, atol=1e-5, rtol=1e-5)


def test_rejects_non_power_of_two_and_indivisible_width():
    with pytest.raises(ValueError, match="power of two"):
        hadamard_matrix(96, torch.device("cpu"))
    x = torch.randn(2, 100)
    with pytest.raises(ValueError, match="not divisible"):
        block_hadamard_rotate(x, 128)


def test_inter_hadamard_is_plumbed_to_stage1_output():
    import aiter.fused_moe

    for name in (
        "fused_moe",
        "fused_moe_fake",
        "fused_moe_",
        "_fused_moe_impl",
        "fused_moe_2stages",
    ):
        fn = getattr(aiter.fused_moe, name)
        assert "inter_hadamard" in inspect.signature(fn).parameters, name

    # The custom op and its fake must accept the same arguments.
    impl = list(inspect.signature(aiter.fused_moe.fused_moe_).parameters)
    fake = list(inspect.signature(aiter.fused_moe.fused_moe_fake).parameters)
    assert impl == fake

    source = inspect.getsource(aiter.fused_moe.fused_moe_2stages)
    rotate_at = source.index("block_hadamard_rotate")
    assert source.index("a2 = _stage1_call()") < rotate_at
    assert source.index("fused_dynamic_mxfp4_quant_moe_sort", rotate_at) > rotate_at
    impl_source = inspect.getsource(aiter.fused_moe._fused_moe_impl)
    assert "1-stage kernel never exposes" in impl_source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_rotation_matches_on_gpu():
    x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    got = block_hadamard_rotate(x, 128)
    h = hadamard_matrix(128, x.device)
    ref = (x.float().unflatten(-1, (-1, 128)) @ h).flatten(-2).to(x.dtype)
    torch.testing.assert_close(got, ref)
