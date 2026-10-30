# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.fused_situ_mul import fused_situ_mul

# Kimi-K3 shared experts: beta 4, linear_beta 25, d = 768 per TP rank.
_BETA = 4.0
_LINEAR_BETA = 25.0
_K3_D = 768


def torch_situ_mul_ref(x: torch.Tensor, beta: float, linear_beta: float | None):
    """fp32 SituGLU (vLLM SituAndMul.forward_native), rounded once."""
    d = x.shape[-1] // 2
    gate, up = x[..., :d].float(), x[..., d:].float()
    act = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (act * up).to(x.dtype)


def generate_fused_situ_mul_inputs(shape, dtype, scale=3.0, device="cuda"):
    """x = [gate | up] on the last dim. Shared with the benchmark."""
    torch.manual_seed(0)
    return (torch.randn(shape, dtype=torch.float32, device=device) * scale).to(dtype)


@pytest.mark.parametrize(
    "shape",
    [(4, 64), (128, 256), (31, 1000), (2, 16, 128), (1, 3, 7, 32), (8192, 1536)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("linear_beta", [_LINEAR_BETA, None])
def test_fused_situ_mul(shape, dtype, linear_beta):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = generate_fused_situ_mul_inputs(shape, dtype)
    ref = torch_situ_mul_ref(x, _BETA, linear_beta)
    torch.testing.assert_close(
        fused_situ_mul(x, _BETA, linear_beta), ref, rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize(
    "n_tokens",
    [1, 5, 10, 15, 16, 64, 320, 1112, 2136, 7680],
    ids=lambda n: f"tokens{n}",
)
def test_fused_situ_mul_k3_shared_expert_shapes(n_tokens):
    """Kimi-K3 shared-expert SituGLU at TP8 width, decode to prefill."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = generate_fused_situ_mul_inputs((n_tokens, 2 * _K3_D), torch.bfloat16)
    ref = torch_situ_mul_ref(x, _BETA, _LINEAR_BETA)
    torch.testing.assert_close(
        fused_situ_mul(x, _BETA, _LINEAR_BETA), ref, rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize("scale", [10.0, 100.0, 1e4])
def test_fused_situ_mul_large_inputs_stay_finite(scale):
    """Saturated tanh / sigmoid: the capped exponents keep the result finite."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = generate_fused_situ_mul_inputs((256, 2 * _K3_D), torch.float32, scale=scale)
    out = fused_situ_mul(x, _BETA, _LINEAR_BETA)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out, torch_situ_mul_ref(x, _BETA, _LINEAR_BETA), rtol=2e-2, atol=2e-2
    )


def test_fused_situ_mul_row_strided():
    """A 2-D view with a dense last dim and padded rows, as a GEMM output slice."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    buf = generate_fused_situ_mul_inputs((64, 4 * _K3_D), torch.bfloat16)
    x = buf[:, : 2 * _K3_D]
    before = buf.clone()
    out = fused_situ_mul(x, _BETA, _LINEAR_BETA)
    torch.testing.assert_close(
        out, torch_situ_mul_ref(x, _BETA, _LINEAR_BETA), rtol=2e-2, atol=2e-2
    )
    assert torch.equal(buf, before), "input must be left untouched"


def test_fused_situ_mul_empty():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.empty(0, 2 * _K3_D, dtype=torch.bfloat16, device="cuda")
    assert fused_situ_mul(x, _BETA, _LINEAR_BETA).shape == (0, _K3_D)


@pytest.mark.parametrize("bad,match", [("odd", "must be even"), ("layout", "2-D")])
def test_fused_situ_mul_rejects_bad_inputs(bad, match):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if bad == "odd":
        x = torch.randn(8, 15, dtype=torch.bfloat16, device="cuda")
    else:
        x = torch.randn(4, 8, 64, dtype=torch.bfloat16, device="cuda")[:, :, :32]
    with pytest.raises(AssertionError, match=match):
        fused_situ_mul(x, _BETA, _LINEAR_BETA)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_situ_mul_special_values(dtype):
    """NaN propagates and +-inf / +-0 / dtype max behave like the eager form (K3 clips up)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    big = torch.finfo(dtype).max
    vals = [0.0, -0.0, 1.0, -1.0, 40.0, -40.0, 1e4, -1e4, big, -big]
    vals += [float("inf"), float("-inf"), float("nan")]
    sp = torch.tensor(vals, dtype=torch.float32, device="cuda")
    gate, up = sp.repeat_interleave(len(sp)), sp.repeat(len(sp))
    x = torch.cat([gate, up]).view(1, -1).to(dtype)
    out = fused_situ_mul(x, _BETA, _LINEAR_BETA)
    ref = torch_situ_mul_ref(x, _BETA, _LINEAR_BETA)
    assert torch.equal(out.isnan(), ref.isnan())
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2, equal_nan=True)
