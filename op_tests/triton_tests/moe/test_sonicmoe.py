# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton._triton_kernels.moe.sonicmoe.grouped_gemm_triton import (
    grouped_gemm,
)
from aiter.ops.triton._triton_kernels.moe.sonicmoe.reduction_over_k_gather import (
    token_gather_and_sum_varlen_K_triton,
)
from aiter.ops.triton._triton_kernels.moe.sonicmoe.routing import (
    TC_topk_router_metadata_triton,
    general_routing_router_metadata_triton,
)
from aiter.ops.triton.sonicmoe import (
    SonicMoEActivationType,
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


def test_topk_routing_metadata_matches_torch():
    indices = torch.tensor(
        [[2, 0], [1, 2], [0, 1], [2, 1]], dtype=torch.int32, device="cuda"
    )
    tokens, top_k = indices.shape
    experts = 3
    total = tokens * top_k
    freq = torch.empty(experts, dtype=torch.int32, device="cuda")
    offsets = torch.empty(experts + 1, dtype=torch.int32, device="cuda")
    gather = torch.empty(total, dtype=torch.int32, device="cuda")
    scatter = torch.empty(total, dtype=torch.int32, device="cuda")
    reverse = torch.empty(total, dtype=torch.int32, device="cuda")
    TC_topk_router_metadata_triton(
        indices, experts, freq, offsets, gather, scatter, reverse
    )

    flat = indices.flatten()
    order = torch.argsort(flat, stable=True)
    expected_freq = torch.bincount(flat, minlength=experts).to(torch.int32)
    torch.testing.assert_close(freq, expected_freq)
    torch.testing.assert_close(
        offsets,
        torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device="cuda"),
                expected_freq.cumsum(0).to(torch.int32),
            )
        ),
    )
    torch.testing.assert_close(gather, order.div(top_k, rounding_mode="floor").int())
    torch.testing.assert_close(scatter, order.int())
    torch.testing.assert_close(reverse[order], torch.arange(total, device="cuda").int())


def test_general_routing_metadata_matches_torch():
    token = torch.tensor([0, 0, 2, 3, 3, 3], dtype=torch.int32, device="cuda")
    expert = torch.tensor([2, 0, 1, 2, 1, 0], dtype=torch.int32, device="cuda")
    tokens, experts, total = 4, 3, expert.numel()
    freq = torch.empty(experts, dtype=torch.int32, device="cuda")
    offsets = torch.empty(experts + 1, dtype=torch.int32, device="cuda")
    gather = torch.empty(total, dtype=torch.int32, device="cuda")
    scatter = torch.empty(total, dtype=torch.int32, device="cuda")
    reverse = torch.empty(total, dtype=torch.int32, device="cuda")
    token_offsets = torch.empty(tokens + 1, dtype=torch.int32, device="cuda")
    general_routing_router_metadata_triton(
        token,
        expert,
        tokens,
        experts,
        freq,
        offsets,
        gather,
        scatter,
        reverse,
        token_offsets,
    )
    order = torch.argsort(expert, stable=True)
    expected_freq = torch.bincount(expert, minlength=experts).int()
    torch.testing.assert_close(freq, expected_freq)
    torch.testing.assert_close(gather, token[order])
    torch.testing.assert_close(scatter, order.int())
    torch.testing.assert_close(reverse[order], torch.arange(total, device="cuda").int())
    torch.testing.assert_close(
        token_offsets,
        torch.searchsorted(
            token, torch.arange(tokens + 1, dtype=torch.int32, device="cuda")
        ).int(),
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_token_gather_sum_matches_torch(weighted):
    torch.manual_seed(13)
    offsets = torch.tensor([0, 2, 2, 5], dtype=torch.int32, device="cuda")
    permutation = torch.tensor([4, 1, 3, 0, 2], dtype=torch.int32, device="cuda")
    x = torch.randn(5, 33, dtype=torch.bfloat16, device="cuda")
    weights = torch.rand(5, dtype=torch.float32, device="cuda") if weighted else None
    out = torch.empty(3, 33, dtype=x.dtype, device="cuda")
    token_gather_and_sum_varlen_K_triton(
        x, weights, out, permutation, offsets, 3, 3, 33, True
    )
    expected = []
    for token in range(3):
        rows = permutation[offsets[token] : offsets[token + 1]].long()
        values = x[rows].float()
        if weights is not None:
            values *= weights[offsets[token] : offsets[token + 1], None]
        expected.append(values.sum(0))
    torch.testing.assert_close(
        out, torch.stack(expected).to(out.dtype), rtol=2e-2, atol=2e-2
    )


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
