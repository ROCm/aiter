# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter.ops.triton.fusions.attn_res as attn_res_module
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
    (``absMax * inverted_DTYPE_MAX`` in csrc/kernels/quant_kernels.cu, i.e. what
    ``get_hip_quant(QuantType.per_Token)`` emits): one fp32 scale per row, taken
    as ``amax * (1 / finfo(dtype).max)``, applied as a reciprocal multiply.

    torch computes the multiply and the divide identically here; the kernel does
    not, which is why it spells this one out (see ``QUANT_FP8`` in the kernel).
    """
    scale = x.float().abs().amax(-1, keepdim=True) * (1.0 / torch.finfo(fp8_dtype).max)
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


@pytest.mark.parametrize("N", [64, 512])
def test_attn_res_gate_out_quant_matches_unfused_quant(N):
    """The fused quant is bit-identical to gate() followed by a separate
    per-token FP8 quant of its output.

    Run in fp32 so the unfused leg's intermediate is the kernel's own fp32
    result rather than a bf16 rounding of it; the two then have to agree
    exactly, which pins scale derivation and rounding, not just closeness.

    This is also what keeps the kernel tied to the HIP per-token quant, which a
    consumer runs on this same activation whenever the fold is off. That kernel
    takes fp16/bf16 only, so it cannot be fed the fp32 tensor a bit-exact
    comparison needs -- but it derives its scale by the identical reciprocal
    multiply, so matching the torch reference below at rtol=0 matches it too.
    Spelling the kernel's scale as a divide instead breaks this test: Triton
    lowers an fp32 divide on AMD to a reciprocal plus refinement, which lands
    about half the rows 1 ulp away from both torch and HIP.

    ``N`` straddles ``_ATTN_RES_PREFILL_T``: the separated prefill path inlines its
    own copy of the quant epilogue, so both have to be pinned.
    """
    D, B = 512, 3
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

    # Bit-exact, not within-an-ulp: the kernel derives the scale by the same
    # reciprocal multiply the reference does, so Triton's fp32 divide (which is
    # a reciprocal plus refinement on AMD, and 1 ulp off) never enters.
    torch.testing.assert_close(y_scale, scale, atol=0.0, rtol=0.0)
    qx = (y_fp32.float() * (1.0 / y_scale)).to(fp8_dtype)
    torch.testing.assert_close(y_fp8.float(), qx.float(), atol=0, rtol=0)


# Straddles _ATTN_RES_PREFILL_T: the separated prefill path carries its own copy of
# the quant epilogue, so the convention has to be asserted on both.
@pytest.mark.parametrize("N", [8, 512])
@pytest.mark.parametrize("close_block", [False, True])
def test_attn_res_gate_out_quant_all_zero_row(N, close_block):
    """An all-zero row gets scale 0 (the HIP convention) and quantizes to zeros.

    The reciprocal of that scale is forced to 0 in the kernel; left as inf it
    would store NaN, and a nonzero placeholder scale would disagree with every
    other per-token quant in the tree on a row that occurs in real prefill
    padding.
    """
    D, B = 256, 3
    dtype, fp8_dtype = torch.bfloat16, get_fp8_e4m3_dtype()
    prefix = torch.zeros(N, D, dtype=dtype, device="cuda")
    block_residual = torch.zeros(N, B, D, dtype=dtype, device="cuda")
    score_weight = torch.randn(D, dtype=dtype, device="cuda")
    output_rms_weight = torch.randn(D, dtype=dtype, device="cuda")

    out = attn_res_gate(
        prefix,
        block_residual,
        score_weight,
        1e-6,
        output_rms_weight=output_rms_weight,
        close_block=close_block,
        out_quant_dtype=fp8_dtype,
    )
    y_fp8, y_scale = out[0]

    assert torch.equal(y_scale, torch.zeros_like(y_scale))
    assert torch.equal(y_fp8.float(), torch.zeros_like(y_fp8, dtype=torch.float32))


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


def _gate_all_variants(prefix, block_residual, score_weight, orw, fp8_dtype):
    """Every axis attn_res_gate keys its launch cache on, one call per setting.

    Deliberately includes settings that do not change the result but DO change
    the kernel Triton compiles: eps/out_eps/scale values (a scalar that happens
    to be 1.0 specializes differently from one that does not), and an unaligned
    prefix (which drops the 16-byte divisibility hint).
    """
    N, D = prefix.shape
    add = torch.randn_like(prefix)
    add2 = torch.randn_like(prefix)
    # A row-sliced view: same shape and dtype, but offset off the 16-byte grid.
    wide = torch.randn(N, D + 1, dtype=prefix.dtype, device=prefix.device)
    unaligned = wide[:, 1:]
    for kwargs in (
        {},
        {"add_hidden": add},
        {"add_hidden": add, "add_hidden2": add2},
        {"output_rms_weight": orw},
        {"output_rms_weight": orw, "out_quant_dtype": fp8_dtype},
        {"close_block": True},
        {"add_hidden": add, "close_block": True},
        {"output_rms_weight": orw, "out_quant_dtype": fp8_dtype, "close_block": True},
        {"eps": 1.0},
        {"output_rms_weight": orw, "output_rms_eps": 1.0},
        {"scale": 0.8},
        {"scale": 1.0},
    ):
        eps = kwargs.pop("eps", 1e-6)
        add_hidden = kwargs.pop("add_hidden", None)
        add_hidden2 = kwargs.pop("add_hidden2", None)
        yield (
            prefix,
            block_residual,
            score_weight,
            eps,
            add_hidden,
            add_hidden2,
        ), kwargs
        yield (
            unaligned,
            block_residual,
            score_weight,
            eps,
            add_hidden,
            add_hidden2,
        ), kwargs


@pytest.mark.parametrize("B", [1, 8])
@pytest.mark.parametrize("N", [1, 128])
def test_attn_res_gate_launch_cache_matches_triton(monkeypatch, B, N):
    """The cached launch must resolve to the kernel Triton itself would pick.

    attn_res_gate skips Triton's per-launch argument specialization by caching
    the resolved kernel under a key it derives itself (decode is host-bound, and
    that specialization is the single largest cost in the launch). Getting the
    key too coarse would silently run a kernel compiled for different arguments,
    so the module can re-resolve through Triton on every hit and assert it got
    the same object back; this turns that check on across the flag matrix.
    """
    monkeypatch.setattr(attn_res_module, "_LAUNCH_CACHE_VERIFY", True)
    D = 256
    dtype, fp8_dtype = torch.bfloat16, get_fp8_e4m3_dtype()
    prefix, block_residual, score_weight, _, _ = generate_attn_res_gate_inputs(
        N, D, B, dtype, with_add=False
    )
    orw = torch.randn(D, dtype=dtype, device="cuda")

    for args, kwargs in _gate_all_variants(
        prefix, block_residual, score_weight, orw, fp8_dtype
    ):
        # Twice: the first call populates the cache, the second is the hit that
        # the verification actually checks.
        attn_res_gate(*args, **kwargs)
        attn_res_gate(*args, **kwargs)


@pytest.mark.parametrize("close_block", [False, True])
@pytest.mark.parametrize("quant", [False, True])
def test_attn_res_gate_launch_cache_bit_identical(monkeypatch, quant, close_block):
    """Cached and uncached launches must produce bit-identical results."""
    N, D, B = 64, 256, 3
    dtype, fp8_dtype = torch.bfloat16, get_fp8_e4m3_dtype()
    prefix, block_residual, score_weight, add_hidden, _ = generate_attn_res_gate_inputs(
        N, D, B, dtype, with_add=True
    )
    orw = torch.randn(D, dtype=dtype, device="cuda")
    kwargs = {
        "output_rms_weight": orw,
        "output_rms_eps": 1e-5,
        "close_block": close_block,
        "out_quant_dtype": fp8_dtype if quant else None,
    }

    def run():
        return attn_res_gate(
            prefix, block_residual, score_weight, 1e-6, add_hidden, **kwargs
        )

    monkeypatch.setattr(attn_res_module, "_LAUNCH_CACHE_ENABLED", False)
    uncached = run()
    monkeypatch.setattr(attn_res_module, "_LAUNCH_CACHE_ENABLED", True)
    run()  # populate
    cached = run()

    def flat(out):
        y = out[0]
        tensors = list(y) if isinstance(y, tuple) else [y]
        return tensors + [t for t in out[1:] if t is not None]

    for a, b in zip(flat(uncached), flat(cached)):
        torch.testing.assert_close(a.float(), b.float(), atol=0, rtol=0)


def test_attn_res_gate_launch_cache_is_bounded_in_token_count():
    """Token count must not be a cache axis, or decode would leak an entry a step.

    N reaches the key only through the properties Triton specializes on
    (16-divisibility, being 1, fitting in i32), not by value.
    """
    D, B = 256, 3
    dtype = torch.bfloat16
    attn_res_module._LAUNCH_CACHE.clear()
    for N in range(17, 49):
        prefix, block_residual, score_weight, _, _ = generate_attn_res_gate_inputs(
            N, D, B, dtype, with_add=False
        )
        attn_res_gate(prefix, block_residual, score_weight, 1e-6)

    entries = sum(len(v) for v in attn_res_module._LAUNCH_CACHE.values())
    # N in [17, 48] spans both 16-divisibility classes and two launch-config
    # buckets (<=64 vs the N<=8 bucket is not reached here), so a handful of
    # entries is expected -- 32 would mean N leaked in by value.
    assert entries <= 8, f"launch cache grew to {entries} entries over 32 token counts"


def test_attn_res_separate_bl_table_matches_documented_buckets():
    """Lock the tuned SEPARATE BL buckets, which key on the candidate count, not N.

    The invariant the table encodes is that the separated loop must run at least
    two iterations: BL = l2 // 2, capped at 4 by the register file. Every
    measured loss sits at BL == l2, where the loop collapses to one iteration
    and the wide tile buys registers with nothing to overlap. So a bucket that
    drifts up to l2 is a real regression (-2% at B=4, -10% at B=8), not a
    mistuning, and that is what these assertions pin.
    """

    def pick(tokens, l2, close_block=True):
        return attn_res_module._pick_attn_res_separate_bl(tokens, l2, close_block)

    mid = 4096  # inside the N range where BL>1 pays at all
    assert pick(mid, 1) == 1
    assert pick(mid, 2) == 1
    assert pick(mid, 4) == 2
    assert pick(mid, 8) == 4
    assert pick(mid, 16) == 4  # capped, not 8
    # BL>1 washes out above _ATTN_RES_SEPARATE_BL_MAX_T and turns into a ~3% loss.
    assert pick(16384, 8) == 4
    assert pick(16385, 8) == 1
    assert pick(65536, 16) == 1
    # The separated loop only covers L-1 rows, so BL must never exceed them.
    assert pick(257, 1) == 1


def test_attn_res_separate_bl_is_pinned_to_one_without_close_block():
    """Without the block_out write, every BL>1 bucket has to collapse to 1.

    This is the whole table's precondition, not a corner case: block_out is
    roughly half the kernel's traffic at B=8, and with it gone the same buckets
    swing from neutral to a 1.4-1.6x *regression* (BL=1 over BL=4 measures
    0.60-0.68 at B=4..15). Swept with close_block off over the canonical grid
    plus N=65536, BL=1 won every cell, so there is no second table to consult --
    which is exactly why an accidental un-gating would be easy to miss.
    """
    pick = attn_res_module._pick_attn_res_separate_bl
    for tokens in (257, 2048, 4096, 16384, 65536):
        for l2 in (1, 2, 4, 8, 16):
            assert pick(tokens, l2, False) == 1, (tokens, l2)
            # ... while the gated-on path still has live BL>1 buckets, so this
            # test cannot pass just because the table went flat everywhere.
    assert pick(4096, 8, True) == 4


@pytest.mark.parametrize("N", [512, 4096])
@pytest.mark.parametrize("B", [1, 3, 8])
def test_attn_res_gate_separate_bl_buckets_match_reference(N, B):
    """The BL>1 SEPARATE branch has to be as accurate as the BL=1 one it replaced.

    Nothing exercised that branch above N=256 before: SEPARATE pinned BL=1. It folds
    the online softmax over a [BL, BD] tile instead of one row at a time, so the
    summation order differs and results are *not* bit-identical to BL=1 -- the
    fp8 scale in particular shifts on almost every row. That is reordering, not
    error, so the reference is the fp32 torch one. B is what selects the bucket,
    so the three values here cover all of it: BL=1, BL=2 and BL=4.
    """
    D, eps = 256, 1e-6
    dtype = torch.bfloat16
    prefix, block_residual, score_weight, add_hidden, add_hidden2 = (
        generate_attn_res_gate_inputs(N, D, B, dtype, with_add=True, with_add2=True)
    )
    orw = torch.randn(D, dtype=dtype, device="cuda")

    y_ref, prefix_ref = run_torch_gate(
        prefix,
        block_residual,
        score_weight,
        eps,
        add_hidden,
        add_hidden2,
        output_rms_weight=orw,
        output_rms_eps=1e-5,
    )
    y, prefix_out, block_out = attn_res_gate(
        prefix,
        block_residual,
        score_weight,
        eps,
        add_hidden,
        add_hidden2,
        output_rms_weight=orw,
        output_rms_eps=1e-5,
        close_block=True,
    )

    atol, rtol = _TOL[dtype]
    torch.testing.assert_close(y.float(), y_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(prefix_out.float(), prefix_ref.float(), atol=0, rtol=0)
    # close_block is cat([block_residual, prefix_out], -2) -- a pure relocation,
    # and the BL tiling is exactly what walks those rows, so it is the piece most
    # likely to go wrong if a wider tile mismaps its lanes.
    expected_block = torch.cat([block_residual, prefix_out.unsqueeze(-2)], dim=-2)
    torch.testing.assert_close(block_out, expected_block, atol=0, rtol=0)


@pytest.mark.parametrize("N", [16384, 32768])
def test_attn_res_gate_launch_cache_separates_separate_bl_buckets(monkeypatch, N):
    """Two token counts in different BL buckets must not share a cache entry.

    BL is a constexpr, and the only trace N leaves in the key is _int_spec,
    which records 16-divisibility rather than a value. Now that the table keys
    BL on the candidate count, _ATTN_RES_SEPARATE_BL_MAX_T is the only boundary N
    still crosses, so 16384 and 32768 are the pair that isolates it: both land
    in the _ATTN_RES_PACKED_CONFIGS catchall (same num_warps/num_stages) and
    both are 16-divisible, leaving BL as the sole difference.

    close_block must be on: BL>1 is gated on it, so with the default off both N
    would pick BL=1 and this would pass without testing anything.

    Verification mode re-resolves through Triton on every hit, so a key that
    dropped BL would fail here rather than silently launching a kernel compiled
    for the other tile width.
    """
    monkeypatch.setattr(attn_res_module, "_LAUNCH_CACHE_VERIFY", True)
    D, B = 256, 8
    dtype = torch.bfloat16
    # The separated loop covers L-1 = B rows, matching the wrapper's l2 argument.
    l2 = B  # already a power of two
    assert attn_res_module._pick_attn_res_separate_bl(
        16384, l2, True
    ) != attn_res_module._pick_attn_res_separate_bl(
        32768, l2, True
    ), "N pair no longer spans a BL bucket boundary"

    prefix, block_residual, score_weight, _, _ = generate_attn_res_gate_inputs(
        N, D, B, dtype, with_add=False
    )
    # Prime with the other bucket first, so a too-coarse key would hit that
    # entry rather than compiling fresh.
    other_n = 32768 if N == 16384 else 16384
    other = generate_attn_res_gate_inputs(other_n, D, B, dtype, with_add=False)
    attn_res_gate(other[0], other[1], other[2], 1e-6, close_block=True)

    attn_res_gate(prefix, block_residual, score_weight, 1e-6, close_block=True)
    attn_res_gate(prefix, block_residual, score_weight, 1e-6, close_block=True)


def test_attn_res_sequence_requires_d_multiple_of_16():
    """The sequence gather hints 16-element alignment, so D must be a multiple of 16."""
    query, residuals, rms_weight, _ = generate_attn_res_inputs(
        16, 40, 2, torch.float32, with_onorm=False
    )
    with pytest.raises(AssertionError, match="multiple of 16"):
        attn_res_fwd(query, residuals, rms_weight, layout="sequence")
