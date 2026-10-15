# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

from aiter.ops.triton.fusions.block_cat_fused import (
    block_cat_fused,
    block_cat_fused_bwd,
    block_cat_fused_fwd,
)

SLOPE = 0.01

# (rows, W12, W1); includes the W1=1 / W4=1 edges, odd widths, and row counts
# whose element count is not a multiple of the launch block
SHAPES = [
    (64, 96, 32),
    (31, 500, 123),
    (128, 33, 1),
    (128, 33, 32),
    (1, 576, 128),
    (4096, 576, 128),
    (4096, 576, 512),
]


def eager_block_cat(l12, l4, slope, l1_width):
    return torch.cat(
        [F.leaky_relu(l12[..., :l1_width], slope), l12[..., l1_width:] * l4], dim=-1
    )


def generate_block_cat_fused_inputs(shape, l1_width, dtype, device="cuda"):
    """(l12, l4) for l12 of the given shape. Shared with the benchmark."""
    torch.manual_seed(0)
    l12 = torch.randn(shape, dtype=dtype, device=device)
    l4 = torch.randn((*shape[:-1], shape[-1] - l1_width), dtype=dtype, device=device)
    return l12, l4


def _fp32_grads(l12, l4, grad_out, l1_width):
    """Eager autograd in fp32, rounded once to the input dtype (what the kernel does)."""
    l12f = l12.detach().float().requires_grad_(True)
    l4f = l4.detach().float().requires_grad_(True)
    out = eager_block_cat(l12f, l4f, SLOPE, l1_width)
    out.backward(grad_out.float())
    return out.to(l12.dtype), l12f.grad.to(l12.dtype), l4f.grad.to(l4.dtype)


def _run_op(l12, l4, grad_out, l1_width):
    l12 = l12.detach().requires_grad_(True)
    l4 = l4.detach().requires_grad_(True)
    out = block_cat_fused(l12, l4, SLOPE, l1_width)
    out.backward(grad_out)
    return out, l12.grad, l4.grad


@pytest.mark.parametrize("rows, W12, W1", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_block_cat_fused(rows, W12, W1, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    l12, l4 = generate_block_cat_fused_inputs((rows, W12), W1, dtype)
    # random upstream grad: an all-ones grad cannot catch a dropped grad factor
    grad_out = torch.randn_like(l12)

    out, grad_l12, grad_l4 = _run_op(l12, l4, grad_out, W1)
    ref_out, ref_grad_l12, ref_grad_l4 = _fp32_grads(l12, l4, grad_out, W1)

    for t in (out, grad_l12, grad_l4):
        assert t.dtype == dtype
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(grad_l12, ref_grad_l12)
    torch.testing.assert_close(grad_l4, ref_grad_l4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_block_cat_fused_matches_eager_autograd(dtype):
    """Agreement with eager autograd run in the input dtype."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    l12, l4 = generate_block_cat_fused_inputs((256, 576), 128, dtype)
    grad_out = torch.randn_like(l12)
    out, grad_l12, grad_l4 = _run_op(l12, l4, grad_out, 128)

    l12_ref = l12.detach().requires_grad_(True)
    l4_ref = l4.detach().requires_grad_(True)
    ref = eager_block_cat(l12_ref, l4_ref, SLOPE, 128)
    ref.backward(grad_out)
    torch.testing.assert_close(out, ref.detach())
    torch.testing.assert_close(grad_l12, l12_ref.grad)
    torch.testing.assert_close(grad_l4, l4_ref.grad)


@pytest.mark.parametrize("transposed", ["l4", "l12", "both"])
def test_block_cat_fused_noncontiguous_inputs(transposed):
    """Transposed views in: outputs and grads still match eager and come back contiguous."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dtype = torch.bfloat16
    rows, W12, W1 = 64, 96, 32
    torch.manual_seed(0)

    def make(cols, t):
        shape = (cols, rows) if t else (rows, cols)
        leaf = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
        return leaf, leaf.t() if t else leaf

    leaf12, l12 = make(W12, transposed in ("l12", "both"))
    leaf4, l4 = make(W12 - W1, transposed in ("l4", "both"))
    grad_out = torch.randn(rows, W12, dtype=dtype, device="cuda")

    out = block_cat_fused(l12, l4, SLOPE, W1)
    out.backward(grad_out)
    grad12, grad4 = leaf12.grad.clone(), leaf4.grad.clone()
    leaf12.grad = None
    leaf4.grad = None

    ref = eager_block_cat(l12, l4, SLOPE, W1)
    ref.backward(grad_out)
    assert out.is_contiguous()
    torch.testing.assert_close(out, ref.detach())
    torch.testing.assert_close(grad12, leaf12.grad)
    torch.testing.assert_close(grad4, leaf4.grad)


def test_block_cat_fused_fake_matches_real_layout():
    """The fake impls must report the contiguous layout the real ops return."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    rows, W12, W1 = 64, 96, 32
    l12 = torch.randn(W12, rows, dtype=torch.bfloat16, device="cuda").t()
    l4 = torch.randn(W12 - W1, rows, dtype=torch.bfloat16, device="cuda").t()
    grad_out = torch.randn(W12, rows, dtype=torch.bfloat16, device="cuda").t()

    real = [block_cat_fused_fwd(l12, l4, SLOPE, W1)]
    real += block_cat_fused_bwd(grad_out, l12, l4, SLOPE, W1)
    with FakeTensorMode() as mode:
        f12, f4, fg = (mode.from_tensor(t) for t in (l12, l4, grad_out))
        fake = [block_cat_fused_fwd(f12, f4, SLOPE, W1)]
        fake += block_cat_fused_bwd(fg, f12, f4, SLOPE, W1)

    for r, f in zip(real, fake):
        assert r.is_contiguous()
        assert f.shape == r.shape
        assert f.dtype == r.dtype
        assert f.stride() == r.stride()


def test_block_cat_fused_torch_compile():
    """fwd + bwd under torch.compile(fullgraph=True) give the eager op's results."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    l12, l4 = generate_block_cat_fused_inputs((512, 576), 128, torch.bfloat16)
    grad_out = torch.randn_like(l12)

    def fn(a, b):
        return block_cat_fused(a, b, SLOPE, 128)

    compiled = torch.compile(fn, fullgraph=True)
    a = l12.detach().requires_grad_(True)
    b = l4.detach().requires_grad_(True)
    out = compiled(a, b)
    out.backward(grad_out)

    ref_out, ref_grad_l12, ref_grad_l4 = _run_op(l12, l4, grad_out, 128)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(a.grad, ref_grad_l12, rtol=0, atol=0)
    torch.testing.assert_close(b.grad, ref_grad_l4, rtol=0, atol=0)


def test_block_cat_fused_3d():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    l12, l4 = generate_block_cat_fused_inputs((4, 33, 96), 32, torch.bfloat16)
    grad_out = torch.randn_like(l12)
    out, grad_l12, grad_l4 = _run_op(l12, l4, grad_out, 32)
    ref_out, ref_grad_l12, ref_grad_l4 = _fp32_grads(l12, l4, grad_out, 32)
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(grad_l12, ref_grad_l12)
    torch.testing.assert_close(grad_l4, ref_grad_l4)


def test_block_cat_fused_empty():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    l12, l4 = generate_block_cat_fused_inputs((0, 96), 32, torch.bfloat16)
    out, grad_l12, grad_l4 = _run_op(l12, l4, torch.empty_like(l12), 32)
    assert out.shape == (0, 96)
    assert grad_l12.shape == (0, 96)
    assert grad_l4.shape == (0, 64)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_block_cat_fused_fp32_eager_path(device):
    """FP32 (and CPU) inputs take the eager path, gradients included."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    l12, l4 = generate_block_cat_fused_inputs((8, 16), 6, torch.float32, device)
    grad_out = torch.randn_like(l12)
    out, grad_l12, grad_l4 = _run_op(l12, l4, grad_out, 6)

    l12_ref = l12.detach().requires_grad_(True)
    l4_ref = l4.detach().requires_grad_(True)
    ref = eager_block_cat(l12_ref, l4_ref, SLOPE, 6)
    ref.backward(grad_out)
    torch.testing.assert_close(out, ref.detach())
    torch.testing.assert_close(grad_l12, l12_ref.grad)
    torch.testing.assert_close(grad_l4, l4_ref.grad)


@pytest.mark.parametrize(
    "bad,match",
    [
        ("dtype", "dtype mismatch"),
        ("device", "same device"),
        ("shape", "l4 shape"),
    ],
)
def test_block_cat_fused_rejects_bad_inputs(bad, match):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    l12, l4 = generate_block_cat_fused_inputs((8, 96), 32, torch.bfloat16)
    if bad == "dtype":
        l4 = l4.float()
    elif bad == "device":
        l4 = l4.cpu()
    else:
        l4 = l4[:, :-1]
    with pytest.raises(ValueError, match=match):
        block_cat_fused(l12, l4, SLOPE, 32)
