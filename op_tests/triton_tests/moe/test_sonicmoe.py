# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.moe.sonicmoe import (
    SonicMoEActivationType,
    grouped_gemm,
    moe_general_routing_inputs,
    moe_pre_routed_inputs,
    moe_TC_softmax_topk_layer,
    sonicmoe_is_glu,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

_ACTIVATIONS = list(SonicMoEActivationType)


def _activation(x, activation):
    if sonicmoe_is_glu(activation):
        gate, up = x[..., ::2], x[..., 1::2]
        if activation == SonicMoEActivationType.SWIGLU:
            return F.silu(gate) * up
        if activation == SonicMoEActivationType.GEGLU:
            return F.gelu(gate.float(), approximate="tanh").to(x.dtype) * up
        return F.relu(gate) * up
    if activation == SonicMoEActivationType.GELU:
        return F.gelu(x.float(), approximate="tanh").to(x.dtype)
    if activation == SonicMoEActivationType.RELU:
        return F.relu(x)
    if activation == SonicMoEActivationType.SILU:
        return F.silu(x)
    return F.relu(x).square()


def _reference_topk_moe(x, router_w, w1, w2, top_k, activation):
    logits = F.linear(x, router_w)
    selected = logits.topk(top_k, dim=-1)
    scores = selected.values.softmax(dim=-1, dtype=torch.float32)
    out = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(router_w.shape[0]):
        token, slot = (selected.indices == expert).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        hidden = F.linear(x[token], w1[expert])
        expert_out = F.linear(_activation(hidden, activation), w2[expert])
        out.index_add_(0, token, expert_out.float() * scores[token, slot, None])
    return out.to(x.dtype), logits


@pytest.mark.parametrize("activation", _ACTIVATIONS, ids=lambda x: x.value)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sonicmoe_topk_forward_backward(activation, dtype):
    torch.manual_seed(7)
    tokens, hidden, intermediate, experts, top_k = 32, 64, 32, 4, 2
    full_intermediate = intermediate * (2 if sonicmoe_is_glu(activation) else 1)

    x = (torch.randn(tokens, hidden, device="cuda", dtype=dtype) * 0.1).requires_grad_()
    router_w = (
        torch.randn(experts, hidden, device="cuda", dtype=dtype) * 0.02
    ).requires_grad_()
    w1_ref = (
        torch.randn(experts, full_intermediate, hidden, device="cuda", dtype=dtype)
        * 0.02
    ).requires_grad_()
    w2_ref = (
        torch.randn(experts, hidden, intermediate, device="cuda", dtype=dtype) * 0.02
    ).requires_grad_()

    x_ref = x.detach().clone().requires_grad_()
    router_ref = router_w.detach().clone().requires_grad_()
    w1 = w1_ref.detach().permute(1, 2, 0).contiguous().requires_grad_()
    w2 = w2_ref.detach().permute(1, 2, 0).contiguous().requires_grad_()
    out, logits, _ = moe_TC_softmax_topk_layer(
        x,
        router_w,
        w1,
        None,
        w2,
        None,
        top_k,
        torch.cuda.current_stream().cuda_stream,
        activation,
    )
    ref, ref_logits = _reference_topk_moe(
        x_ref, router_ref, w1_ref, w2_ref, top_k, activation
    )

    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(logits, ref_logits, rtol=2e-2, atol=2e-2)
    grad = torch.randn_like(out)
    grads = torch.autograd.grad(out, (x, router_w, w1, w2), grad)
    ref_grads = torch.autograd.grad(ref, (x_ref, router_ref, w1_ref, w2_ref), grad)
    torch.testing.assert_close(grads[0], ref_grads[0], rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(grads[1], ref_grads[1], rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(
        grads[2], ref_grads[2].permute(1, 2, 0), rtol=7e-2, atol=7e-2
    )
    torch.testing.assert_close(
        grads[3], ref_grads[3].permute(1, 2, 0), rtol=7e-2, atol=7e-2
    )


@pytest.mark.parametrize("activation", _ACTIVATIONS, ids=lambda x: x.value)
def test_sonicmoe_pre_routed_forward_backward(activation):
    torch.manual_seed(11)
    counts = [5, 0, 7]
    tokens, experts, hidden, intermediate = sum(counts), len(counts), 32, 16
    full_intermediate = intermediate * (2 if sonicmoe_is_glu(activation) else 1)
    x = (
        torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    ).requires_grad_()
    scores = torch.rand(tokens, device="cuda", dtype=torch.float32).requires_grad_()
    w1 = (
        torch.randn(
            experts, hidden, full_intermediate, device="cuda", dtype=torch.bfloat16
        )
        * 0.02
    ).requires_grad_()
    w2 = (
        torch.randn(experts, intermediate, hidden, device="cuda", dtype=torch.bfloat16)
        * 0.02
    ).requires_grad_()

    out, _ = moe_pre_routed_inputs(
        x,
        scores,
        torch.tensor(counts, dtype=torch.int32),
        w1,
        None,
        w2,
        None,
        torch.cuda.current_stream().cuda_stream,
        activation,
    )
    ref_chunks = []
    offset = 0
    for expert, count in enumerate(counts):
        if count:
            hidden_state = x[offset : offset + count] @ w1[expert]
            ref_chunks.append(
                (_activation(hidden_state, activation) @ w2[expert])
                * scores[offset : offset + count, None]
            )
        offset += count
    ref = torch.cat(ref_chunks).to(out.dtype)
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)
    grad = torch.randn_like(out)
    actual_grads = torch.autograd.grad(
        out, (x, scores, w1, w2), grad, retain_graph=True
    )
    ref_grads = torch.autograd.grad(ref, (x, scores, w1, w2), grad)
    for actual, expected in zip(actual_grads, ref_grads):
        torch.testing.assert_close(actual, expected, rtol=7e-2, atol=7e-2)


@pytest.mark.skipif(get_arch() != "gfx942", reason="fnuz blockwise FP8 is gfx942-only")
def test_grouped_gemm_blockwise_fp8_matches_dequantized_torch():
    torch.manual_seed(17)
    fp8 = torch.float8_e4m3fnuz
    counts = [128, 128]
    experts, m, k, n = len(counts), sum(counts), 128, 128
    a_scale = torch.rand(m, 1, device="cuda", dtype=torch.float32) + 0.5
    b_scale = torch.rand(experts, 1, 1, device="cuda", dtype=torch.float32) + 0.5
    a = torch.randn(m, k, device="cuda").clamp(-2, 2).to(fp8)
    b = torch.randn(experts, k, n, device="cuda").clamp(-2, 2).to(fp8)
    offsets = torch.tensor([0, 128, 256], dtype=torch.int32, device="cuda")
    actual = grouped_gemm(
        a,
        b,
        offsets,
        A_scale=a_scale,
        B_scale=b_scale,
        out_dtype=torch.bfloat16,
    )
    expected = torch.cat(
        [
            (
                a[sum(counts[:expert]) : sum(counts[: expert + 1])].float()
                * a_scale[sum(counts[:expert]) : sum(counts[: expert + 1])]
            )
            @ (b[expert].float() * b_scale[expert])
            for expert in range(experts)
        ]
    )
    torch.testing.assert_close(actual.float(), expected, rtol=6e-2, atol=1.0)


def test_grouped_wgrad_validates_scale_rows_per_expert():
    a = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    a_scale = torch.ones(1, 128, device="cuda")
    b_scale = torch.ones(1, 128, device="cuda")

    with pytest.raises(ValueError, match=r"A_scale must have shape \[2, 128\]"):
        grouped_gemm(
            a,
            b,
            offsets,
            A_is_transposed=True,
            A_scale=a_scale,
            B_scale=b_scale,
        )


@pytest.mark.parametrize("with_bias", [False, True])
def test_general_routing_grouped_weights_match_legacy_layout(with_bias):
    torch.manual_seed(43)
    device = torch.device("cuda")
    tokens, hidden, intermediate, experts = 64, 64, 64, 2
    token_indices = torch.arange(tokens, dtype=torch.int32, device=device)
    expert_indices = torch.repeat_interleave(
        torch.arange(experts, dtype=torch.int32, device=device),
        torch.tensor([32, 32], dtype=torch.int32, device=device),
    )

    x_grouped = torch.randn(
        tokens, hidden, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    x_legacy = x_grouped.detach().clone().requires_grad_(True)
    scores_grouped = torch.rand(
        tokens, dtype=torch.float32, device=device, requires_grad=True
    )
    scores_legacy = scores_grouped.detach().clone().requires_grad_(True)
    w1_grouped = torch.randn(
        experts,
        hidden,
        2 * intermediate,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    w2_grouped = torch.randn(
        experts,
        intermediate,
        hidden,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    w1_legacy = w1_grouped.detach().permute(2, 1, 0).contiguous().requires_grad_(True)
    w2_legacy = w2_grouped.detach().permute(2, 1, 0).contiguous().requires_grad_(True)
    b1_grouped = (
        torch.randn(
            experts, 2 * intermediate, dtype=torch.bfloat16, device=device
        ).requires_grad_(True)
        if with_bias
        else None
    )
    b2_grouped = (
        torch.randn(
            experts, hidden, dtype=torch.bfloat16, device=device
        ).requires_grad_(True)
        if with_bias
        else None
    )
    b1_legacy = b1_grouped.detach().clone().requires_grad_(True) if with_bias else None
    b2_legacy = b2_grouped.detach().clone().requires_grad_(True) if with_bias else None

    common = (
        token_indices,
        expert_indices,
        experts,
        torch.cuda.current_stream().cuda_stream,
        SonicMoEActivationType.SWIGLU,
        False,
        True,
    )
    output_grouped, _ = moe_general_routing_inputs(
        x_grouped,
        scores_grouped,
        common[0],
        common[1],
        w1_grouped,
        b1_grouped,
        w2_grouped,
        b2_grouped,
        *common[2:],
        grouped_weight_layout=True,
    )
    output_legacy, _ = moe_general_routing_inputs(
        x_legacy,
        scores_legacy,
        common[0],
        common[1],
        w1_legacy,
        b1_legacy,
        w2_legacy,
        b2_legacy,
        *common[2:],
    )

    torch.testing.assert_close(output_grouped, output_legacy)
    grad = torch.randn_like(output_grouped)
    output_grouped.backward(grad)
    output_legacy.backward(grad)
    torch.testing.assert_close(x_grouped.grad, x_legacy.grad)
    torch.testing.assert_close(scores_grouped.grad, scores_legacy.grad)
    torch.testing.assert_close(w1_grouped.grad, w1_legacy.grad.permute(2, 1, 0))
    torch.testing.assert_close(w2_grouped.grad, w2_legacy.grad.permute(2, 1, 0))
    if with_bias:
        torch.testing.assert_close(b1_grouped.grad, b1_legacy.grad)
        torch.testing.assert_close(b2_grouped.grad, b2_legacy.grad)
