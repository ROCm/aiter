# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.activation import silu_and_mul_backward
from aiter.ops.triton.utils import config_utils
from aiter.ops.triton.utils._triton.arch_info import get_arch

_SUPPORTED_ARCHS = ("gfx950",)
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and get_arch() not in _SUPPORTED_ARCHS,
        reason="silu_and_mul_backward supports gfx950",
    ),
]


def _torch_reference(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x_ref = x.float().detach().requires_grad_(True)
    gate, up = x_ref.chunk(2, dim=-1)
    out = torch.nn.functional.silu(gate) * up
    (grad_input,) = torch.autograd.grad(out, x_ref, grad_output.float())
    return grad_input.to(x.dtype)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((3, 2), id="width_1"),
        (4, 64),
        (31, 500),
        (2, 16, 128),
        (3, 1024),
        (3, 1026),
        pytest.param((2, 8192), id="qwen3_tp3_like"),
        pytest.param((2, 24576), id="qwen3_tp1"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("noncontiguous_grad", [False, True])
def test_silu_and_mul_backward(shape, dtype, noncontiguous_grad):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    grad_shape = (*shape[:-1], shape[-1] // 2)
    if noncontiguous_grad:
        grad_storage = torch.randn(*shape[:-1], shape[-1], dtype=dtype, device=x.device)
        grad_output = grad_storage[..., ::2]
        assert not grad_output.is_contiguous()
    else:
        grad_output = torch.randn(grad_shape, dtype=dtype, device=x.device)

    ref = _torch_reference(grad_output, x)
    out = silu_and_mul_backward(grad_output, x)
    atol = rtol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


def test_silu_and_mul_backward_explicit_out():
    x = torch.randn((7, 256), dtype=torch.bfloat16, device="cuda")
    grad_output = torch.randn((7, 128), dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    result = silu_and_mul_backward(grad_output, x, out=out)
    assert result is out
    torch.testing.assert_close(
        out, _torch_reference(grad_output, x), rtol=1e-2, atol=1e-2
    )


def test_silu_and_mul_backward_empty_rows():
    x = torch.empty((0, 64), dtype=torch.bfloat16, device="cuda")
    grad_output = torch.empty((0, 32), dtype=x.dtype, device=x.device)
    out = silu_and_mul_backward(grad_output, x)
    assert out.shape == x.shape
    assert out.numel() == 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_silu_and_mul_backward_non_current_device():
    current_device = torch.cuda.current_device()
    input_device = (current_device + 1) % torch.cuda.device_count()
    x = torch.randn((3, 128), dtype=torch.bfloat16, device=input_device)
    grad_output = torch.randn((3, 64), dtype=x.dtype, device=x.device)

    out = silu_and_mul_backward(grad_output, x)
    torch.testing.assert_close(
        out, _torch_reference(grad_output, x), rtol=1e-2, atol=1e-2
    )
    assert out.device == x.device
    assert torch.cuda.current_device() == current_device


def test_silu_and_mul_backward_unsupported_arch(monkeypatch):
    x = torch.randn((2, 64), dtype=torch.bfloat16, device="cuda")
    grad_output = torch.randn((2, 32), dtype=x.dtype, device=x.device)
    monkeypatch.setattr(config_utils.arch_info, "get_arch", lambda: "gfx942")

    with pytest.raises(FileNotFoundError, match="gfx942.*silu_and_mul_backward"):
        silu_and_mul_backward(grad_output, x)


def test_silu_and_mul_backward_validation():
    x = torch.randn((2, 64), dtype=torch.bfloat16, device="cuda")
    grad_output = torch.randn((2, 32), dtype=x.dtype, device=x.device)
    with pytest.raises(AssertionError, match="non-zero"):
        silu_and_mul_backward(
            torch.empty((2, 0), device=x.device), torch.empty((2, 0), device=x.device)
        )
    with pytest.raises(AssertionError, match="even"):
        silu_and_mul_backward(grad_output, torch.randn((2, 63), device=x.device))
    with pytest.raises(AssertionError, match="shape"):
        silu_and_mul_backward(grad_output[:, :-1], x)
    with pytest.raises(AssertionError, match="dtype"):
        silu_and_mul_backward(grad_output.float(), x)
    with pytest.raises(AssertionError, match="contiguous"):
        silu_and_mul_backward(grad_output, x[:, ::2])
    with pytest.raises(AssertionError, match="out shape"):
        silu_and_mul_backward(grad_output, x, out=torch.empty_like(grad_output))
