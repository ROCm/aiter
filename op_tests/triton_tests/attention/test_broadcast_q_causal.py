# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.attention.broadcast_q_causal import (
    broadcast_q_causal_varlen_func,
)


def _make_inputs(q_len, lengths, heads, head_dim, dtype, seed=1234):
    torch.manual_seed(seed)
    total_kv = sum(lengths)
    q = torch.randn(
        (q_len, heads, head_dim), device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        (total_kv, heads, head_dim), device="cuda", dtype=dtype, requires_grad=True
    )
    v = torch.randn_like(k, requires_grad=True)
    cu_seqlens_k = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    return q, k, v, cu_seqlens_k


def _reference(q, k, v, lengths, softmax_scale):
    q_ref = q.detach().float().requires_grad_()
    k_ref = k.detach().float().requires_grad_()
    v_ref = v.detach().float().requires_grad_()
    q_len, heads, head_dim = q.shape
    out_parts = []
    lse_parts = []
    start = 0

    for length in lengths:
        active_len = min(q_len, length)
        if active_len == 0:
            out_parts.append(torch.zeros((q_len, heads, head_dim), device=q.device))
            lse_parts.append(
                torch.full(
                    (q_len, heads), -torch.inf, device=q.device, dtype=torch.float32
                )
            )
            start += length
            continue

        scores = (
            torch.einsum(
                "mhd,nhd->hmn",
                q_ref[:active_len],
                k_ref[start : start + active_len],
            )
            * softmax_scale
        )
        causal = torch.tril(
            torch.ones((active_len, active_len), device=q.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal, -torch.inf)
        lse = torch.logsumexp(scores, dim=-1).T
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum(
            "hmn,nhd->mhd",
            probs,
            v_ref[start : start + active_len],
        )
        out_parts.append(
            torch.cat(
                (
                    out,
                    torch.zeros((q_len - active_len, heads, head_dim), device=q.device),
                ),
                dim=0,
            )
        )
        lse_parts.append(
            torch.cat(
                (
                    lse,
                    torch.full(
                        (q_len - active_len, heads),
                        -torch.inf,
                        device=q.device,
                        dtype=torch.float32,
                    ),
                ),
                dim=0,
            )
        )
        start += length

    return (
        torch.stack(out_parts),
        torch.stack(lse_parts),
        (q_ref, k_ref, v_ref),
    )


def _assert_forward_and_backward(
    q,
    k,
    v,
    cu_seqlens_k,
    lengths,
    softmax_scale=None,
):
    out, lse = broadcast_q_causal_varlen_func(
        q, k, v, cu_seqlens_k, softmax_scale=softmax_scale
    )
    grad_out = torch.randn_like(out)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), grad_out)

    out_ref, lse_ref, refs = _reference(
        q, k, v, lengths, softmax_scale or q.shape[2] ** -0.5
    )
    grad_out_ref = grad_out.float().reshape_as(out_ref)
    ref_grads = torch.autograd.grad(out_ref, refs, grad_out_ref)

    torch.testing.assert_close(
        out.float().reshape_as(out_ref), out_ref, atol=0.05, rtol=0.05
    )
    finite = torch.isfinite(lse_ref)
    assert torch.equal(lse.float().reshape_as(lse_ref)[~finite], lse_ref[~finite])
    torch.testing.assert_close(
        lse.float().reshape_as(lse_ref)[finite],
        lse_ref[finite],
        atol=0.02,
        rtol=0.02,
    )
    for actual, expected in zip((dq, dk, dv), ref_grads):
        torch.testing.assert_close(actual.float(), expected, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_forward_and_backward(dtype, head_dim):
    lengths = [0, 3, 7]
    q, k, v, cu_seqlens_k = _make_inputs(5, lengths, 2, head_dim, dtype)
    _assert_forward_and_backward(q, k, v, cu_seqlens_k, lengths)


def test_multiple_query_and_key_blocks():
    lengths = [512, 37, 0, 300, 128, 451, 1]
    q, k, v, cu_seqlens_k = _make_inputs(300, lengths, 2, 64, torch.bfloat16, seed=2468)
    _assert_forward_and_backward(q, k, v, cu_seqlens_k, lengths)


def test_one_head_and_non_default_scale():
    lengths = [19, 3]
    q, k, v, cu_seqlens_k = _make_inputs(17, lengths, 1, 64, torch.bfloat16, seed=1357)
    _assert_forward_and_backward(q, k, v, cu_seqlens_k, lengths, softmax_scale=0.07)


def test_non_contiguous_inputs_and_upstream_gradient():
    lengths = [9, 2]
    q_len, heads, head_dim = 7, 2, 64
    _, _, _, cu_seqlens_k = _make_inputs(
        q_len, lengths, heads, head_dim, torch.bfloat16, seed=8642
    )
    q_storage = torch.randn(
        (q_len, heads, head_dim, 2), device="cuda", dtype=torch.bfloat16
    )
    k_storage = torch.randn(
        (sum(lengths), heads, head_dim, 2), device="cuda", dtype=torch.bfloat16
    )
    v_storage = torch.randn_like(k_storage)
    q_noncontig = q_storage[..., 0].requires_grad_()
    k_noncontig = k_storage[..., 0].requires_grad_()
    v_noncontig = v_storage[..., 0].requires_grad_()
    out, _ = broadcast_q_causal_varlen_func(
        q_noncontig, k_noncontig, v_noncontig, cu_seqlens_k
    )
    grad_storage = torch.randn(
        (out.shape[0], heads, head_dim, 2), device="cuda", dtype=torch.float32
    )
    grad_out = grad_storage[..., 0]
    dq, dk, dv = torch.autograd.grad(
        out, (q_noncontig, k_noncontig, v_noncontig), grad_out
    )
    out_ref, _, refs = _reference(
        q_noncontig, k_noncontig, v_noncontig, lengths, head_dim**-0.5
    )
    grad_out_ref = grad_out.float().reshape_as(out_ref)
    ref_grads = torch.autograd.grad(out_ref, refs, grad_out_ref)
    for actual, expected in zip((dq, dk, dv), ref_grads):
        torch.testing.assert_close(actual.float(), expected, atol=0.05, rtol=0.05)


def test_non_contiguous_sliced_backward():
    lengths = [9, 2]
    q_len, heads, head_dim = 7, 2, 64
    _, _, _, cu_seqlens_k = _make_inputs(
        q_len, lengths, heads, head_dim, torch.bfloat16, seed=7531
    )
    q_storage = torch.randn(
        (q_len * 2, heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    k_storage = torch.randn(
        (sum(lengths) * 2, heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    v_storage = torch.randn_like(k_storage)
    q = q_storage[::2].requires_grad_()
    k = k_storage[::2].requires_grad_()
    v = v_storage[::2].requires_grad_()
    out, _ = broadcast_q_causal_varlen_func(q, k, v, cu_seqlens_k)
    grad_out = torch.randn_like(out)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), grad_out)

    out_ref, _, refs = _reference(q, k, v, lengths, head_dim**-0.5)
    grad_out_ref = grad_out.float().reshape_as(out_ref)
    ref_grads = torch.autograd.grad(out_ref, refs, grad_out_ref)
    for actual, expected in zip((dq, dk, dv), ref_grads):
        torch.testing.assert_close(actual.float(), expected, atol=0.05, rtol=0.05)


def test_return_lse_false():
    lengths = [4, 2]
    q, k, v, cu_seqlens_k = _make_inputs(5, lengths, 2, 64, torch.bfloat16, seed=9753)
    out = broadcast_q_causal_varlen_func(q, k, v, cu_seqlens_k, return_lse=False)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (len(lengths) * 5, 2, 64)


@pytest.mark.parametrize("input_name", ["q", "k", "v"])
def test_partial_gradients(input_name):
    lengths = [6, 2]
    q, k, v, cu_seqlens_k = _make_inputs(5, lengths, 2, 64, torch.bfloat16, seed=3141)
    inputs = (q, k, v)
    for tensor in inputs:
        tensor.requires_grad_(False)
    index = ("q", "k", "v").index(input_name)
    inputs[index].requires_grad_(True)
    out, _ = broadcast_q_causal_varlen_func(q, k, v, cu_seqlens_k)
    grad_out = torch.randn_like(out)
    actual = torch.autograd.grad(out, inputs[index], grad_out)[0]

    out_ref, _, refs = _reference(q, k, v, lengths, 64**-0.5)
    grad_out_ref = grad_out.float().reshape_as(out_ref)
    expected = torch.autograd.grad(out_ref, refs[index], grad_out_ref)[0]
    torch.testing.assert_close(actual.float(), expected, atol=0.05, rtol=0.05)


def test_lse_is_non_differentiable():
    lengths = [3]
    q, k, v, cu_seqlens_k = _make_inputs(4, lengths, 1, 64, torch.bfloat16)
    _, lse = broadcast_q_causal_varlen_func(q, k, v, cu_seqlens_k)
    assert lse.requires_grad is False
    with pytest.raises(RuntimeError):
        lse.backward()


def test_backward_rejects_nonzero_first_offset():
    q, k, v, _ = _make_inputs(4, [4, 4], 1, 64, torch.bfloat16)
    cu_seqlens_k = torch.tensor([1, 4, 8], device="cuda", dtype=torch.int32)
    out, _ = broadcast_q_causal_varlen_func(q, k, v, cu_seqlens_k)
    with pytest.raises(ValueError, match="zero"):
        torch.autograd.grad(out, (q, k, v), torch.randn_like(out))


def test_empty_batch_backward():
    q, k, v, _ = _make_inputs(4, [], 1, 64, torch.bfloat16)
    cu_seqlens_k = torch.zeros(1, device="cuda", dtype=torch.int32)
    out, lse = broadcast_q_causal_varlen_func(q, k, v, cu_seqlens_k)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), torch.empty_like(out))
    assert out.shape == (0, 1, 64)
    assert lse.shape == (0, 1)
    assert torch.count_nonzero(dq) == 0
    assert torch.count_nonzero(dk) == 0
    assert torch.count_nonzero(dv) == 0


def test_int64_offsets():
    lengths = [3, 2]
    q, k, v, _ = _make_inputs(4, lengths, 1, 64, torch.bfloat16)
    cu_seqlens_k = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int64)
    _assert_forward_and_backward(q, k, v, cu_seqlens_k, lengths)


def test_rejects_unsupported_head_dim():
    q, k, v, cu_seqlens_k = _make_inputs(4, [2], 1, 32, torch.bfloat16)
    with pytest.raises(ValueError, match="head_dim"):
        broadcast_q_causal_varlen_func(q, k, v, cu_seqlens_k)
