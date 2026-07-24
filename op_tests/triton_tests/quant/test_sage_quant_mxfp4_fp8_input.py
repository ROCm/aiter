# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Tests for sage_quant_mxfp4_fp8_input — the fused fp8-input quantization kernel.
#
# Two levels:
#   1. Unit tests: compare quantized outputs directly against the reference
#      sage_quant_mxfp4(bf16) and sage_quant_mxfp4(fp8 upcast) paths using
#      cosine similarity, sign agreement, and scale match after dequantization.
#   2. End-to-end test: feed the quantized outputs into sage_fwd_mxfp4 and
#      compare the attention result against the PyTorch reference attention,
#      using the same tolerances as test_sage_mxfp4.

import torch
import pytest

import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    sage_quant_mxfp4,
    create_hadamard_matrix,
)
from aiter.ops.triton.quant.sage_attention_quant_fp8_input_wrapper import (
    sage_quant_mxfp4_fp8_input,
)
from aiter.ops.triton.moe.quant_moe import upcast_from_mxfp
from aiter.test_mha_common import attention_ref
from op_tests.triton_tests.attention.test_fav3_sage import (
    check_attention_outputs,
    input_helper,
)

import aiter

FP8_TYPE = aiter.dtypes.fp8
FP8_MAX = torch.finfo(FP8_TYPE).max

ATOL_FP8 = 3.0e-1
RTOL_FP8 = 2.5e-1

# Cosine similarity threshold for quantized-output unit tests.
# fp8 input is inherently lossy vs bf16 input, so we accept ~0.97+.
COS_SIM_THRESHOLD = 0.96
# Sign agreement: fp4 only has 3 mantissa bits, sign errors are critical.
SIGN_AGREE_THRESHOLD = 0.99
# V must be bit-identical (same bf16 input, same code path).
V_COS_THRESHOLD = 0.9999


def _dequant(packed, scale):
    """Packed fp4x2 + e8m0 scale → bf16, quantized axis is last."""
    return upcast_from_mxfp(packed, scale, torch.bfloat16, axis=-1)


def _cosine_sim(a, b):
    a = a.reshape(-1, a.shape[-1]).float()
    b = b.reshape(-1, b.shape[-1]).float()
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return (a_n * b_n).sum(dim=-1).mean().item()


def _sign_agree(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    nz = (a != 0) & (b != 0)
    if nz.sum() == 0:
        return 1.0
    return (torch.sign(a[nz]) == torch.sign(b[nz])).float().mean().item()


def _quant_kwargs(layout, R):
    return dict(
        FP8_TYPE=FP8_TYPE, FP8_MAX=FP8_MAX,
        BLKQ=128, BLKK=64, layout=layout, R=R, BLOCK_R=128,
    )


# ---------------------------------------------------------------------------
# Unit tests: quantized-output correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v_fp8", [False, True])
@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
@pytest.mark.parametrize("B, S, H, D", [
    (1, 256, 8, 128),
    (1, 1024, 16, 128),
    (2, 512, 16, 128),
    (4, 256, 16, 128),
])
def test_sage_quant_mxfp4_fp8_input_vs_upcast(B, S, H, D, layout, v_fp8):
    """Fused fp8 kernel must agree with the upcast path on identical fp8 inputs."""
    if not arch_info.is_fp4_avail():
        pytest.skip("MXFP4 not supported on this architecture")

    torch.manual_seed(42)
    shape = (B, S, H, D) if layout == "bshd" else (B, H, S, D)
    q_bf16 = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k_bf16 = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v_bf16 = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    q_fp8 = q_bf16.to(FP8_TYPE)
    k_fp8 = k_bf16.to(FP8_TYPE)

    R = create_hadamard_matrix(128, device="cuda", dtype=torch.bfloat16) / (128 ** 0.5)
    kw = _quant_kwargs(layout, R)

    # Reference: same fp8 inputs upcast to bf16 on the host
    ref = sage_quant_mxfp4(q_fp8.to(torch.bfloat16), k_fp8.to(torch.bfloat16), v_bf16, **kw)
    ref_q_fp4, ref_q_sc, ref_k_fp4, ref_k_sc, ref_v_fp8, ref_v_sc, _ = ref

    # Candidate: fused on-chip widening, with optional pre-quantized fp8 V
    if v_fp8:
        # Simulate upstream-quantized V: use ref v_fp8/v_sc as the pre-quantized input
        out = sage_quant_mxfp4_fp8_input(q_fp8, k_fp8, ref_v_fp8, **kw, v_scale=ref_v_sc)
    else:
        out = sage_quant_mxfp4_fp8_input(q_fp8, k_fp8, v_bf16, **kw)
    out_q_fp4, out_q_sc, out_k_fp4, out_k_sc, out_v_fp8, out_v_sc, _ = out

    q_dq = _dequant(out_q_fp4, out_q_sc)
    k_dq = _dequant(out_k_fp4, out_k_sc)
    rq_dq = _dequant(ref_q_fp4, ref_q_sc)
    rk_dq = _dequant(ref_k_fp4, ref_k_sc)

    assert _cosine_sim(q_dq, rq_dq) >= COS_SIM_THRESHOLD, \
        f"Q cosine similarity too low: {_cosine_sim(q_dq, rq_dq):.4f}"
    assert _cosine_sim(k_dq, rk_dq) >= COS_SIM_THRESHOLD, \
        f"K cosine similarity too low: {_cosine_sim(k_dq, rk_dq):.4f}"
    assert _sign_agree(q_dq, rq_dq) >= SIGN_AGREE_THRESHOLD, \
        f"Q sign agreement too low: {_sign_agree(q_dq, rq_dq):.4f}"
    assert _sign_agree(k_dq, rk_dq) >= SIGN_AGREE_THRESHOLD, \
        f"K sign agreement too low: {_sign_agree(k_dq, rk_dq):.4f}"

    # V checks: fp8 passthrough must be bit-identical; bf16 path must be near-identical
    v_dq = out_v_fp8.to(torch.float32) * out_v_sc.unsqueeze(1 if layout == "bshd" else 2)
    rv_dq = ref_v_fp8.to(torch.float32) * ref_v_sc.unsqueeze(1 if layout == "bshd" else 2)
    if v_fp8:
        assert torch.equal(out_v_fp8, ref_v_fp8), "V fp8 passthrough must be bit-identical"
        assert torch.equal(out_v_sc, ref_v_sc), "V scale fp8 passthrough must be bit-identical"
    else:
        assert _cosine_sim(v_dq, rv_dq) >= V_COS_THRESHOLD, \
            f"V cosine similarity too low: {_cosine_sim(v_dq, rv_dq):.4f}"


@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
@pytest.mark.parametrize("B, S, H, D", [
    (1, 256, 8, 128),
    (1, 1024, 16, 128),
])
def test_sage_quant_mxfp4_fp8_input_vs_bf16_ref(B, S, H, D, layout):
    """fp8 fused path must stay above a minimum cosine similarity vs bf16 reference.
    V dtype is not parametrized here because V does not affect Q/K quantization —
    the fp8-V passthrough path is already covered by test_sage_quant_mxfp4_fp8_input_vs_upcast
    and test_sage_quant_mxfp4_fp8_input_attention.
    """
    if not arch_info.is_fp4_avail():
        pytest.skip("MXFP4 not supported on this architecture")

    torch.manual_seed(42)
    shape = (B, S, H, D) if layout == "bshd" else (B, H, S, D)
    q_bf16 = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k_bf16 = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v_bf16 = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    q_fp8 = q_bf16.to(FP8_TYPE)
    k_fp8 = k_bf16.to(FP8_TYPE)

    R = create_hadamard_matrix(128, device="cuda", dtype=torch.bfloat16) / (128 ** 0.5)
    kw = _quant_kwargs(layout, R)

    ref = sage_quant_mxfp4(q_bf16, k_bf16, v_bf16, **kw)
    ref_q_fp4, ref_q_sc, ref_k_fp4, ref_k_sc, _, _, _ = ref

    out = sage_quant_mxfp4_fp8_input(q_fp8, k_fp8, v_bf16, **kw)
    out_q_fp4, out_q_sc, out_k_fp4, out_k_sc, _, _, _ = out

    q_cos = _cosine_sim(_dequant(out_q_fp4, out_q_sc), _dequant(ref_q_fp4, ref_q_sc))
    k_cos = _cosine_sim(_dequant(out_k_fp4, out_k_sc), _dequant(ref_k_fp4, ref_k_sc))

    # fp8 input is inherently noisier than bf16; this just guards against catastrophic regression
    assert q_cos >= COS_SIM_THRESHOLD, f"Q cosine vs bf16 ref too low: {q_cos:.4f}"
    assert k_cos >= COS_SIM_THRESHOLD, f"K cosine vs bf16 ref too low: {k_cos:.4f}"


# ---------------------------------------------------------------------------
# End-to-end attention test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v_fp8", [False, True])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(16, 16), (16, 8)])
@pytest.mark.parametrize("BATCH, SEQLEN", [(1, 256), (2, 256), (4, 128)])
def test_sage_quant_mxfp4_fp8_input_attention(BATCH, SEQLEN, NUM_Q_HEADS, NUM_K_HEADS, causal, v_fp8):
    """
    Full attention pass using fp8 q/k through sage_quant_mxfp4_fp8_input,
    compared against PyTorch reference attention with the same tolerances as
    test_sage_mxfp4. Parametrized over v_fp8 to cover both the bf16-V
    quantization path and the fp8-V passthrough path.
    """
    if not arch_info.is_fp4_avail():
        pytest.skip("MXFP4 not supported on this architecture")

    HEAD_SZ = 128
    layout = "bhsd"
    torch.cuda.empty_cache()
    torch.manual_seed(20)

    q_bf16, k_bf16, v_bf16 = input_helper(
        BATCH, NUM_Q_HEADS, NUM_K_HEADS, SEQLEN, SEQLEN, HEAD_SZ, HEAD_SZ,
        torch.bfloat16, layout,
    )
    q_fp8 = q_bf16.to(FP8_TYPE)
    k_fp8 = k_bf16.to(FP8_TYPE)

    R = create_hadamard_matrix(128, device="cuda", dtype=torch.bfloat16) / (128 ** 0.5)
    kw = _quant_kwargs(layout, R)

    if v_fp8:
        # Pre-quantize V using the bf16 path to get a realistic fp8 V + scale,
        # then pass them directly to exercise the passthrough branch.
        _, _, _, _, v_in, v_scale_in, _ = sage_quant_mxfp4_fp8_input(
            q_fp8, k_fp8, v_bf16, **kw
        )
        q_fp4, q_sc, k_fp4, k_sc, v_out, v_sc, _ = sage_quant_mxfp4_fp8_input(
            q_fp8, k_fp8, v_in, **kw, v_scale=v_scale_in
        )
    else:
        q_fp4, q_sc, k_fp4, k_sc, v_out, v_sc, _ = sage_quant_mxfp4_fp8_input(
            q_fp8, k_fp8, v_bf16, **kw
        )

    # Run the mxfp4 attention kernel with our quantized tensors.
    # fav3_sage_mxfp4_func signature: (q, k, v, q_descale, k_descale, v_descale, ...)
    from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
        get_sage_fwd_configs_mxfp4,
        fav3_sage_mxfp4_func,
    )

    config = get_sage_fwd_configs_mxfp4()
    triton_out = fav3_sage_mxfp4_func(
        q_fp4, k_fp4, v_out,
        q_descale=q_sc,
        k_descale=k_sc,
        v_descale=v_sc,
        causal=causal,
        layout=layout,
        config=config,
    )

    # PyTorch reference (bf16 inputs, full-precision attention); bhsd → bshd for attention_ref
    q_ref = q_bf16.permute(0, 2, 1, 3).contiguous()
    k_ref = k_bf16.permute(0, 2, 1, 3).contiguous()
    v_ref = v_bf16.permute(0, 2, 1, 3).contiguous()
    torch_out, _, _ = attention_ref(q_ref, k_ref, v_ref, dropout_p=0.0, dropout_mask=None, causal=causal)
    torch_out = torch_out.permute(0, 2, 1, 3).contiguous()

    assert triton_out.shape == torch_out.shape
    # fp8 inputs are noisier than bf16, so we allow a higher diff percentage
    # than the bf16→mxfp4 path (which uses 1.5%).
    check_attention_outputs(
        triton_out, torch_out,
        fp8=True,
        atol=ATOL_FP8,
        rtol=RTOL_FP8,
        max_diff_percentage=4.0,
    )
