# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.fused_sigmoid_mul import fused_sigmoid_mul

LOG2_E = 1.44269504089
_K3_TP8_WIDTH = 4096


def sigmoid_exp2_ref(t: torch.Tensor) -> torch.Tensor:
    """Match ``_sigmoid_exp2`` in Triton (exp2 form, computed in fp32)."""
    x = t.float()
    return 1.0 / (1.0 + torch.exp2(-(x * LOG2_E)))


def torch_sigmoid_mul_ref(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """fp32 product, rounded once -- what the kernel does."""
    return (x.float() * sigmoid_exp2_ref(gate)).to(x.dtype)


def generate_fused_sigmoid_mul_inputs(shape, dtype, device="cuda"):
    """(x, gate) for the given shape/dtype. Shared with the benchmark."""
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=device)
    gate = torch.randn(shape, dtype=dtype, device=device)
    return x, gate


@pytest.mark.parametrize(
    "shape",
    [
        (4, 64),
        (128, 256),
        (31, 500),
        (2, 16, 128),
        (1, 3, 7, 32),
        (1, _K3_TP8_WIDTH),
        (8192, 512),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_explicit_out", [False, True])
def test_fused_sigmoid_mul(shape, dtype, use_explicit_out):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x, gate = generate_fused_sigmoid_mul_inputs(shape, dtype)
    ref = torch_sigmoid_mul_ref(x, gate)

    if use_explicit_out:
        out = torch.empty_like(x)
        ret = fused_sigmoid_mul(x, gate, out)
        assert ret is out
        torch.testing.assert_close(
            x, generate_fused_sigmoid_mul_inputs(shape, dtype)[0]
        )
    else:
        ret = fused_sigmoid_mul(x, gate)
        assert ret is x

    torch.testing.assert_close(ret, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "n_tokens",
    [1, 2, 4, 8, 16, 32, 64, 128, 1166, 8192],
    ids=lambda n: f"tokens{n}",
)
def test_fused_sigmoid_mul_k3_mla_gate_shapes(n_tokens):
    """MLA output gating at Kimi-K3 TP8 width, across decode and prefill."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dtype = torch.bfloat16
    x, gate = generate_fused_sigmoid_mul_inputs((n_tokens, _K3_TP8_WIDTH), dtype)
    ref = torch_sigmoid_mul_ref(x, gate)
    torch.testing.assert_close(fused_sigmoid_mul(x, gate), ref, rtol=1e-2, atol=1e-2)


def test_fused_sigmoid_mul_matches_eager_within_dtype_noise():
    """Agreement with the eager form the fusion replaces."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dtype = torch.bfloat16
    x, gate = generate_fused_sigmoid_mul_inputs((4096, 1024), dtype)
    eager = x * gate.sigmoid()
    torch.testing.assert_close(
        fused_sigmoid_mul(x.clone(), gate), eager, rtol=2e-2, atol=2e-2
    )


def test_fused_sigmoid_mul_empty():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.empty(0, dtype=torch.bfloat16, device="cuda")
    gate = torch.empty(0, dtype=torch.bfloat16, device="cuda")
    assert fused_sigmoid_mul(x, gate).numel() == 0


@pytest.mark.parametrize(
    "bad,match",
    [
        ("shape", "shape mismatch"),
        ("dtype", "dtype mismatch"),
        ("contiguous", "must be contiguous"),
    ],
)
def test_fused_sigmoid_mul_rejects_bad_inputs(bad, match):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
    if bad == "shape":
        gate = torch.randn(8, 32, dtype=torch.bfloat16, device="cuda")
    elif bad == "dtype":
        gate = torch.randn(8, 16, dtype=torch.float16, device="cuda")
    else:
        gate = torch.randn(16, 8, dtype=torch.bfloat16, device="cuda").t()
    with pytest.raises(AssertionError, match=match):
        fused_sigmoid_mul(x, gate)


# Kimi-K3 TP8 MLA output gate, per rank: 12 heads x v_head_dim 128.
_K3_GATE_WIDTH = 1536
# The gate is the last shard of the merged [q_a | kv_a | gate] GEMM output.
_K3_MERGED_ROW = 1536 + 576 + _K3_GATE_WIDTH
# attn_out kept at the 16 heads the ASM MLA kernels run at (16 x 128).
_K3_PADDED_ATTN_ROW = 2048


@pytest.mark.parametrize(
    "n_tokens", [1, 7, 56, 64, 1166, 7827], ids=lambda n: f"tokens{n}"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("x_layout", ["contiguous", "padded_heads"])
@pytest.mark.parametrize("use_explicit_out", [False, True])
def test_fused_sigmoid_mul_row_strided(n_tokens, dtype, x_layout, use_explicit_out):
    """Row-strided 2-D views, as the K3 MLA gate passes them: the gate is a column
    slice of the merged GEMM output, attn_out optionally a slice of a padded buffer."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    merged = torch.randn(n_tokens, _K3_MERGED_ROW, dtype=dtype, device="cuda")
    gate = merged[:, _K3_MERGED_ROW - _K3_GATE_WIDTH :]
    if x_layout == "padded_heads":
        x_buf = torch.randn(n_tokens, _K3_PADDED_ATTN_ROW, dtype=dtype, device="cuda")
        x = x_buf[:, :_K3_GATE_WIDTH]
    else:
        x_buf = torch.randn(n_tokens, _K3_GATE_WIDTH, dtype=dtype, device="cuda")
        x = x_buf
    ref = torch_sigmoid_mul_ref(x, gate)
    merged_before, x_buf_before = merged.clone(), x_buf.clone()

    if use_explicit_out:
        out = torch.empty(n_tokens, _K3_GATE_WIDTH, dtype=dtype, device="cuda")
        ret = fused_sigmoid_mul(x, gate, out)
        assert ret is out
        assert torch.equal(x_buf, x_buf_before), "x must be left untouched"
    else:
        ret = fused_sigmoid_mul(x, gate)
        assert ret is x
        assert torch.equal(
            x_buf[:, _K3_GATE_WIDTH:], x_buf_before[:, _K3_GATE_WIDTH:]
        ), "in-place write must stay inside the x view"

    torch.testing.assert_close(ret, ref, rtol=1e-2, atol=1e-2)
    assert torch.equal(merged, merged_before), "the gate source must be untouched"


def test_fused_sigmoid_mul_rejects_non_2d_strided():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(4, 8, 32, device="cuda")[:, :, :16]
    gate = torch.randn(4, 8, 16, device="cuda")
    with pytest.raises(AssertionError, match="2-D"):
        fused_sigmoid_mul(x, gate)
