# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.attn_res import attn_res_fwd, attn_res_gate
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

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


def generate_attn_res_gate_inputs(N, D, B, dtype, with_add, seed=33, with_add2=False):
    torch.manual_seed(seed)
    prefix = torch.randn(N, D, dtype=dtype, device="cuda")
    block_residual = torch.randn(N, B, D, dtype=dtype, device="cuda")
    score_weight = torch.randn(D, dtype=dtype, device="cuda")
    add_hidden = torch.randn(N, D, dtype=dtype, device="cuda") if with_add else None
    add_hidden2 = (
        torch.randn(N, D, dtype=dtype, device="cuda")
        if (with_add and with_add2)
        else None
    )
    return prefix, block_residual, score_weight, add_hidden, add_hidden2


def run_torch_gate(
    prefix,
    block_residual,
    score_weight,
    eps,
    add_hidden,
    add_hidden2=None,
    *,
    output_rms_weight=None,
    output_rms_eps=None,
    scale=1.0,
):
    """Reference for attn_res_gate.

    Mirrors the kernel's precision: the prefix add accumulates in fp32 and that
    fp32 value is what feeds the candidate, while the written-back prefix is
    rounded to the tensor dtype.
    """
    if output_rms_eps is None:
        output_rms_eps = eps
    ps = prefix.float()
    if add_hidden is not None:
        ps = ps + add_hidden.float()
        if add_hidden2 is not None:
            ps = ps + add_hidden2.float()
        prefix_out = ps.to(prefix.dtype)
    else:
        prefix_out = prefix
    v = torch.cat([block_residual.float(), ps.unsqueeze(-2)], dim=-2)  # [N, B+1, D]
    rstd = torch.rsqrt((v * v).mean(-1) + eps)
    logit = rstd * (v * score_weight.float()).sum(-1)
    probs = torch.softmax(logit * scale, dim=-1)
    y = (probs.unsqueeze(-1) * v).sum(-2)
    if output_rms_weight is not None:
        y_rstd = torch.rsqrt((y * y).mean(-1, keepdim=True) + output_rms_eps)
        y = y * y_rstd * output_rms_weight.flatten().float()
    return y, prefix_out


@pytest.mark.parametrize("shape", [(64, 256), (128, 512), (37, 1024)])
@pytest.mark.parametrize("B", [1, 2, 3, 7])
@pytest.mark.parametrize("with_add", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attn_res_gate(shape, B, with_add, dtype):
    N, D = shape
    eps = 1e-6
    prefix, block_residual, score_weight, add_hidden, _ = generate_attn_res_gate_inputs(
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
    prefix, block_residual, score_weight, add_hidden, _ = generate_attn_res_gate_inputs(
        N, D, B, dtype, with_add
    )
    output_rms_weight = torch.randn(D, dtype=dtype, device="cuda")

    y_ref, prefix_ref = run_torch_gate(
        prefix,
        block_residual,
        score_weight,
        eps,
        add_hidden,
        output_rms_weight=output_rms_weight,
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


@pytest.mark.parametrize("B", [1, 3, 7])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attn_res_gate_add_hidden2(B, dtype):
    """add_hidden2 folds a SECOND addend into the prefix (mirrors ATOM's
    routed + shared MoE expert output fold)."""
    N, D = 128, 512
    eps = 1e-6
    prefix, block_residual, score_weight, add_hidden, add_hidden2 = (
        generate_attn_res_gate_inputs(N, D, B, dtype, with_add=True, with_add2=True)
    )

    y_ref, prefix_ref = run_torch_gate(
        prefix, block_residual, score_weight, eps, add_hidden, add_hidden2
    )
    y, prefix_out = attn_res_gate(
        prefix, block_residual, score_weight, eps, add_hidden, add_hidden2
    )

    atol, rtol = _TOL[dtype]
    torch.testing.assert_close(y.float(), y_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(prefix_out.float(), prefix_ref.float(), atol=0, rtol=0)


def test_attn_res_gate_add_hidden2_requires_add_hidden():
    """add_hidden2 without add_hidden is rejected (mirrors ATOM's validation)."""
    prefix, block_residual, score_weight, _, _ = generate_attn_res_gate_inputs(
        32, 256, 2, torch.float32, with_add=False
    )
    add_hidden2 = torch.randn(32, 256, device="cuda")
    with pytest.raises(ValueError, match="add_hidden2 requires add_hidden"):
        attn_res_gate(prefix, block_residual, score_weight, 1e-6, None, add_hidden2)


@pytest.mark.parametrize("B", [1, 3, 7])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attn_res_gate_output_rms_eps_independent_of_eps(B, dtype):
    """output_rms_eps can differ from the per-candidate eps."""
    N, D = 128, 512
    # Deliberately far apart (mean_sq of the output is O(1), so out_eps=1.0
    # noticeably changes the RMSNorm denominator) -- large enough to clear
    # even bf16's loose comparison tolerance in the negative check below.
    eps, output_rms_eps = 1e-6, 1.0
    prefix, block_residual, score_weight, add_hidden, _ = generate_attn_res_gate_inputs(
        N, D, B, dtype, with_add=True
    )
    output_rms_weight = torch.randn(D, dtype=dtype, device="cuda")

    y_ref, prefix_ref = run_torch_gate(
        prefix,
        block_residual,
        score_weight,
        eps,
        add_hidden,
        output_rms_weight=output_rms_weight,
        output_rms_eps=output_rms_eps,
    )
    y, prefix_out = attn_res_gate(
        prefix,
        block_residual,
        score_weight,
        eps,
        add_hidden,
        output_rms_weight=output_rms_weight,
        output_rms_eps=output_rms_eps,
    )

    atol, rtol = _TOL[dtype]
    torch.testing.assert_close(y.float(), y_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(prefix_out.float(), prefix_ref.float(), atol=0, rtol=0)

    # Sanity: using eps for both (the old behavior) must NOT match, so the test
    # would actually catch a regression back to a single shared eps.
    y_shared_eps, _ = run_torch_gate(
        prefix,
        block_residual,
        score_weight,
        eps,
        add_hidden,
        output_rms_weight=output_rms_weight,
        output_rms_eps=eps,
    )
    assert not torch.allclose(y_ref, y_shared_eps, atol=atol, rtol=rtol)


def test_attn_res_gate_output_rmsnorm_matches_unfused():
    """Fusing the prenorm matches gate + a separate RMSNorm on its output."""
    N, D, B = 64, 512, 4
    eps = 1e-6
    dtype = torch.float32
    prefix, block_residual, score_weight, _, _ = generate_attn_res_gate_inputs(
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
    prefix, block_residual, score_weight, _, _ = generate_attn_res_gate_inputs(
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
    prefix, block_residual, score_weight, _, _ = generate_attn_res_gate_inputs(
        N, D, B, dtype, with_add=False
    )

    y_gate, _ = attn_res_gate(prefix, block_residual, score_weight, eps, scale=scale)
    # attn_res_fwd folds query * rms_weight, so feed the folded vector as query
    # and a unit rms_weight, with the prefix materialized as the last candidate.
    packed = torch.cat([block_residual, prefix.unsqueeze(-2)], dim=-2).contiguous()
    ones = torch.ones(D, dtype=dtype, device=prefix.device)
    y_fwd = attn_res_fwd(score_weight, packed, ones, None, eps, scale, layout="packed")

    torch.testing.assert_close(y_gate, y_fwd, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("B", [1, 3, 7])
@pytest.mark.parametrize("with_add", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attn_res_gate_close_block(B, with_add, dtype):
    """close_block fuses cat([block_residual, prefix_out], -2) into the kernel
    (mirrors ATOM's AttnRes.maybe_close_block); must not perturb (y, prefix_out)
    and block_out must match a manual torch.cat exactly (pure relocation, not a
    new computation)."""
    N, D = 128, 512
    eps = 1e-6
    prefix, block_residual, score_weight, add_hidden, _ = generate_attn_res_gate_inputs(
        N, D, B, dtype, with_add
    )

    y0, prefix_out0 = attn_res_gate(
        prefix, block_residual, score_weight, eps, add_hidden, close_block=False
    )
    y1, prefix_out1, block_out = attn_res_gate(
        prefix, block_residual, score_weight, eps, add_hidden, close_block=True
    )

    torch.testing.assert_close(y1, y0, atol=0, rtol=0)
    torch.testing.assert_close(prefix_out1.float(), prefix_out0.float(), atol=0, rtol=0)
    assert block_out.shape == (N, B + 1, D)

    expected = torch.cat([block_residual, prefix_out1.unsqueeze(-2)], dim=-2)
    torch.testing.assert_close(block_out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("B", [1, 3, 7])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attn_res_gate_close_block_composes_with_add2_and_onorm(B, dtype):
    """close_block composes with add_hidden2 and output_rms_weight (all three
    flags fold into the same single kernel launch)."""
    N, D = 128, 512
    eps = 1e-6
    prefix, block_residual, score_weight, add_hidden, add_hidden2 = (
        generate_attn_res_gate_inputs(N, D, B, dtype, with_add=True, with_add2=True)
    )
    output_rms_weight = torch.randn(D, dtype=dtype, device="cuda")

    y_ref, prefix_ref = run_torch_gate(
        prefix,
        block_residual,
        score_weight,
        eps,
        add_hidden,
        add_hidden2,
        output_rms_weight=output_rms_weight,
    )
    y, prefix_out, block_out = attn_res_gate(
        prefix,
        block_residual,
        score_weight,
        eps,
        add_hidden,
        add_hidden2,
        output_rms_weight=output_rms_weight,
        close_block=True,
    )

    atol, rtol = _TOL[dtype]
    torch.testing.assert_close(y.float(), y_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(prefix_out.float(), prefix_ref.float(), atol=0, rtol=0)
    expected_block = torch.cat([block_residual, prefix_out.unsqueeze(-2)], dim=-2)
    torch.testing.assert_close(block_out, expected_block, atol=0, rtol=0)


def _dequant_per_token(y_fp8, y_scale):
    return y_fp8.float() * y_scale.float()


def run_torch_per_token_quant(x, fp8_dtype):
    """Reference for the fused per-token FP8 quant, in aiter's convention
    (``_dynamic_per_token_quant_fp8_i8_kernel``): one fp32 scale per row, taken
    as ``amax / finfo(dtype).max``, applied as a reciprocal multiply."""
    scale = x.float().abs().amax(-1, keepdim=True) / torch.finfo(fp8_dtype).max
    return (x.float() * (1.0 / scale)).to(fp8_dtype), scale


@pytest.mark.parametrize("B", [1, 3, 7])
@pytest.mark.parametrize("with_add", [False, True])
@pytest.mark.parametrize("close_block", [False, True])
def test_attn_res_gate_out_quant(B, with_add, close_block):
    """out_quant_dtype folds the per-token FP8 quant of the output RMSNorm into
    the kernel; dequantizing must recover the BF16 result, and the block-banking
    cat must stay unquantized and byte-identical to torch.cat."""
    N, D = 128, 512
    eps, dtype = 1e-6, torch.bfloat16
    fp8_dtype = get_fp8_e4m3_dtype()
    prefix, block_residual, score_weight, add_hidden, _ = generate_attn_res_gate_inputs(
        N, D, B, dtype, with_add
    )
    output_rms_weight = torch.randn(D, dtype=dtype, device="cuda")

    out_bf16 = attn_res_gate(
        prefix,
        block_residual,
        score_weight,
        eps,
        add_hidden,
        output_rms_weight=output_rms_weight,
        close_block=close_block,
    )
    out_quant = attn_res_gate(
        prefix,
        block_residual,
        score_weight,
        eps,
        add_hidden,
        output_rms_weight=output_rms_weight,
        close_block=close_block,
        out_quant_dtype=fp8_dtype,
    )
    y_bf16, prefix_bf16 = out_bf16[0], out_bf16[1]
    (y_fp8, y_scale), prefix_quant = out_quant[0], out_quant[1]

    assert y_fp8.dtype == fp8_dtype
    assert y_fp8.shape == (N, D)
    assert y_scale.shape == (N, 1) and y_scale.dtype == torch.float32

    # e4m3 keeps 3 mantissa bits, so a value survives to one relative step of
    # 2^-3, plus a floor of one scale unit for the fp8-subnormal elements.
    ref = y_bf16.float()
    err = (_dequant_per_token(y_fp8, y_scale) - ref).abs()
    assert (err <= ref.abs() * 2**-3 + y_scale.float()).all()
    # The scale is the row amax, which pins the convention (not just closeness).
    torch.testing.assert_close(
        y_scale,
        ref.abs().amax(-1, keepdim=True) / torch.finfo(fp8_dtype).max,
        atol=0.0,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        prefix_quant.float(), prefix_bf16.float(), atol=0, rtol=0
    )

    if close_block:
        block_out = out_quant[2]
        assert block_out.dtype == dtype
        expected = torch.cat([block_residual, prefix_quant.unsqueeze(-2)], dim=-2)
        torch.testing.assert_close(block_out, expected, atol=0, rtol=0)


def test_attn_res_gate_out_quant_matches_unfused_quant():
    """The fused quant is bit-identical to gate() followed by a separate
    per-token FP8 quant of its output.

    Run in fp32 so the unfused leg's intermediate is the kernel's own fp32
    result rather than a bf16 rounding of it; the two then have to agree
    exactly, which pins scale derivation and rounding, not just closeness.
    """
    N, D, B = 64, 512, 3
    eps, dtype = 1e-6, torch.float32
    fp8_dtype = get_fp8_e4m3_dtype()
    prefix, block_residual, score_weight, _, _ = generate_attn_res_gate_inputs(
        N, D, B, dtype, with_add=False
    )
    output_rms_weight = torch.randn(D, dtype=dtype, device="cuda")

    y_fp32, _ = attn_res_gate(
        prefix,
        block_residual,
        score_weight,
        eps,
        output_rms_weight=output_rms_weight,
    )
    (y_fp8, y_scale), _ = attn_res_gate(
        prefix,
        block_residual,
        score_weight,
        eps,
        output_rms_weight=output_rms_weight,
        out_quant_dtype=fp8_dtype,
    )

    _qx, scale = run_torch_per_token_quant(y_fp32, fp8_dtype)

    # The scale agrees to within one fp32 ulp rather than bit-exactly: Triton
    # lowers the fp32 divide to a reciprocal plus refinement on AMD.
    torch.testing.assert_close(y_scale, scale, atol=0.0, rtol=1e-6)
    # Quantizing with the kernel's own scale takes that divide out of the
    # comparison, leaving the rounding itself, which must match exactly.
    qx = (y_fp32.float() * (1.0 / y_scale)).to(fp8_dtype)
    torch.testing.assert_close(y_fp8.float(), qx.float(), atol=0, rtol=0)


def test_attn_res_gate_out_quant_requires_output_rms_weight():
    """Quantizing without an output RMSNorm has no defined input, so it's rejected."""
    prefix, block_residual, score_weight, _, _ = generate_attn_res_gate_inputs(
        32, 256, 2, torch.bfloat16, with_add=False
    )
    with pytest.raises(ValueError, match="out_quant_dtype requires output_rms_weight"):
        attn_res_gate(
            prefix,
            block_residual,
            score_weight,
            1e-6,
            out_quant_dtype=get_fp8_e4m3_dtype(),
        )


def test_attn_res_sequence_requires_d_multiple_of_16():
    """The sequence gather hints 16-element alignment, so D must be a multiple of 16."""
    query, residuals, rms_weight, _ = generate_attn_res_inputs(
        16, 40, 2, torch.float32, with_onorm=False
    )
    with pytest.raises(AssertionError, match="multiple of 16"):
        attn_res_fwd(query, residuals, rms_weight, layout="sequence")
