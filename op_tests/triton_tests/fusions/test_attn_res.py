# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.attn_res import attn_res_fwd, attn_res_gate

# (dtype -> (atol, rtol)) for comparing against the fp32 torch reference.
_TOL = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (2e-2, 2e-2),
}


def generate_attn_res_inputs(N, D, L, dtype, with_onorm, seed=33):
    torch.manual_seed(seed)
    residuals = [torch.randn(N, D, dtype=dtype, device="cuda") for _ in range(L)]
    query = torch.randn(D, dtype=dtype, device="cuda")
    rms_weight = torch.randn(D, dtype=dtype, device="cuda")
    output_rms_weight = (
        torch.randn(D, dtype=dtype, device="cuda") if with_onorm else None
    )
    return query, residuals, rms_weight, output_rms_weight


def run_torch(query, residuals, rms_weight, output_rms_weight, rms_eps, scale):
    D = residuals[0].shape[-1]
    v = torch.stack([r.reshape(-1, D).float() for r in residuals], dim=0)  # [L, N, D]
    qw = query.flatten().float() * rms_weight.flatten().float()
    rstd = torch.rsqrt((v * v).mean(-1) + rms_eps)  # [L, N]
    logit = rstd * (v * qw).sum(-1)  # [L, N]
    probs = torch.softmax(logit * scale, dim=0)  # [L, N]
    o_pre = (probs.unsqueeze(-1) * v).sum(0)  # [N, D]
    if output_rms_weight is not None:
        o_rstd = torch.rsqrt((o_pre * o_pre).mean(-1, keepdim=True) + rms_eps)
        o = o_pre * o_rstd * output_rms_weight.flatten().float()
    else:
        o = o_pre
    return o, o_pre, rstd, logit, probs


@pytest.mark.parametrize("layout", ["sequence", "packed"])
@pytest.mark.parametrize("shape", [(64, 256), (128, 512), (37, 1024)])
@pytest.mark.parametrize("L", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("with_onorm", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attn_res(layout, shape, L, with_onorm, dtype):
    N, D = shape
    rms_eps, scale = 1e-6, 0.7
    query, residuals, rms_weight, output_rms_weight = generate_attn_res_inputs(
        N, D, L, dtype, with_onorm
    )

    o_ref, *_ = run_torch(
        query, residuals, rms_weight, output_rms_weight, rms_eps, scale
    )
    o = attn_res_fwd(
        query,
        residuals,
        rms_weight,
        output_rms_weight,
        rms_eps,
        scale,
        layout=layout,
    )

    atol, rtol = _TOL[dtype]
    torch.testing.assert_close(o.float(), o_ref, atol=atol, rtol=rtol)


def test_attn_res_packed_tensor_input():
    """The packed layout also accepts a pre-stacked [N, L, D] tensor."""
    N, D, L = 64, 512, 4
    rms_eps, scale = 1e-6, 1.0
    dtype = torch.bfloat16
    query, residuals, rms_weight, _ = generate_attn_res_inputs(
        N, D, L, dtype, with_onorm=False
    )
    packed = torch.stack(residuals, dim=-2).contiguous()  # [N, L, D]

    o_list = attn_res_fwd(
        query, residuals, rms_weight, None, rms_eps, scale, layout="packed"
    )
    o_packed = attn_res_fwd(
        query, packed, rms_weight, None, rms_eps, scale, layout="packed"
    )

    torch.testing.assert_close(o_packed, o_list, atol=0, rtol=0)


def generate_attn_res_gate_inputs(N, D, B, dtype, with_add, seed=33):
    torch.manual_seed(seed)
    prefix = torch.randn(N, D, dtype=dtype, device="cuda")
    block_residual = torch.randn(N, B, D, dtype=dtype, device="cuda")
    score_weight = torch.randn(D, dtype=dtype, device="cuda")
    add_hidden = torch.randn(N, D, dtype=dtype, device="cuda") if with_add else None
    return prefix, block_residual, score_weight, add_hidden


def run_torch_gate(
    prefix,
    block_residual,
    score_weight,
    eps,
    add_hidden,
    output_rms_weight=None,
    scale=1.0,
):
    """Reference for attn_res_gate.

    Mirrors the kernel's precision: the prefix add accumulates in fp32 and that
    fp32 value is what feeds the candidate, while the written-back prefix is
    rounded to the tensor dtype.
    """
    ps = prefix.float()
    if add_hidden is not None:
        ps = ps + add_hidden.float()
        prefix_out = ps.to(prefix.dtype)
    else:
        prefix_out = prefix
    v = torch.cat([block_residual.float(), ps.unsqueeze(-2)], dim=-2)  # [N, B+1, D]
    rstd = torch.rsqrt((v * v).mean(-1) + eps)
    logit = rstd * (v * score_weight.float()).sum(-1)
    probs = torch.softmax(logit * scale, dim=-1)
    y = (probs.unsqueeze(-1) * v).sum(-2)
    if output_rms_weight is not None:
        y_rstd = torch.rsqrt((y * y).mean(-1, keepdim=True) + eps)
        y = y * y_rstd * output_rms_weight.flatten().float()
    return y, prefix_out


@pytest.mark.parametrize("shape", [(64, 256), (128, 512), (37, 1024)])
@pytest.mark.parametrize("B", [1, 2, 3, 7])
@pytest.mark.parametrize("with_add", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attn_res_gate(shape, B, with_add, dtype):
    N, D = shape
    eps = 1e-6
    prefix, block_residual, score_weight, add_hidden = generate_attn_res_gate_inputs(
        N, D, B, dtype, with_add
    )

    y_ref, prefix_ref = run_torch_gate(
        prefix, block_residual, score_weight, eps, add_hidden
    )
    y, prefix_out = attn_res_gate(prefix, block_residual, score_weight, eps, add_hidden)

    atol, rtol = _TOL[dtype]
    torch.testing.assert_close(y.float(), y_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(prefix_out.float(), prefix_ref.float(), atol=0, rtol=0)


@pytest.mark.parametrize("B", [1, 3, 7])
@pytest.mark.parametrize("with_add", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attn_res_gate_output_rmsnorm(B, with_add, dtype):
    """output_rms_weight fuses the following prenorm into the gate."""
    N, D = 128, 512
    eps = 1e-6
    prefix, block_residual, score_weight, add_hidden = generate_attn_res_gate_inputs(
        N, D, B, dtype, with_add
    )
    output_rms_weight = torch.randn(D, dtype=dtype, device="cuda")

    y_ref, prefix_ref = run_torch_gate(
        prefix, block_residual, score_weight, eps, add_hidden, output_rms_weight
    )
    y, prefix_out = attn_res_gate(
        prefix,
        block_residual,
        score_weight,
        eps,
        add_hidden,
        output_rms_weight=output_rms_weight,
    )

    atol, rtol = _TOL[dtype]
    torch.testing.assert_close(y.float(), y_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(prefix_out.float(), prefix_ref.float(), atol=0, rtol=0)


def test_attn_res_gate_output_rmsnorm_matches_unfused():
    """Fusing the prenorm matches gate + a separate RMSNorm on its output."""
    N, D, B = 64, 512, 4
    eps = 1e-6
    dtype = torch.float32
    prefix, block_residual, score_weight, _ = generate_attn_res_gate_inputs(
        N, D, B, dtype, with_add=False
    )
    output_rms_weight = torch.randn(D, dtype=dtype, device="cuda")

    y_fused, _ = attn_res_gate(
        prefix,
        block_residual,
        score_weight,
        eps,
        output_rms_weight=output_rms_weight,
    )
    y_pre, _ = attn_res_gate(prefix, block_residual, score_weight, eps)
    y_unfused = torch.nn.functional.rms_norm(y_pre, (D,), output_rms_weight, eps)

    atol, rtol = _TOL[dtype]
    torch.testing.assert_close(y_fused, y_unfused, atol=atol, rtol=rtol)


def test_attn_res_gate_no_add_returns_prefix_unchanged():
    """Without add_hidden the prefix is passed through untouched."""
    prefix, block_residual, score_weight, _ = generate_attn_res_gate_inputs(
        64, 512, 3, torch.float32, with_add=False
    )
    prefix_copy = prefix.clone()

    _y, prefix_out = attn_res_gate(prefix, block_residual, score_weight, 1e-6)

    assert prefix_out is prefix
    torch.testing.assert_close(prefix, prefix_copy, atol=0, rtol=0)


@pytest.mark.parametrize("B", [1, 4])
def test_attn_res_gate_matches_attn_res_fwd(B):
    """The gate is attn_res_fwd on the packed layout with prefix as last candidate."""
    N, D = 64, 512
    eps, scale = 1e-6, 0.8
    dtype = torch.float32
    prefix, block_residual, score_weight, _ = generate_attn_res_gate_inputs(
        N, D, B, dtype, with_add=False
    )

    y_gate, _ = attn_res_gate(prefix, block_residual, score_weight, eps, scale=scale)
    # attn_res_fwd folds query * rms_weight, so feed the folded vector as query
    # and a unit rms_weight, with the prefix materialized as the last candidate.
    packed = torch.cat([block_residual, prefix.unsqueeze(-2)], dim=-2).contiguous()
    ones = torch.ones(D, dtype=dtype, device=prefix.device)
    y_fwd = attn_res_fwd(score_weight, packed, ones, None, eps, scale, layout="packed")

    torch.testing.assert_close(y_gate, y_fwd, atol=1e-5, rtol=1e-5)


def test_attn_res_sequence_requires_d_multiple_of_16():
    """The sequence gather hints 16-element alignment, so D must be a multiple of 16."""
    query, residuals, rms_weight, _ = generate_attn_res_inputs(
        16, 40, 2, torch.float32, with_onorm=False
    )
    with pytest.raises(AssertionError, match="multiple of 16"):
        attn_res_fwd(query, residuals, rms_weight, layout="sequence")
