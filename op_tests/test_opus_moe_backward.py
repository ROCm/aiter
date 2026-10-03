# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Native backward contracts, checked against independent FP32 equations."""
import pytest
import torch
from dataclasses import replace

from aiter.ops.opus import moe_backward as opus

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.version.hip
    or 'gfx950' not in torch.cuda.get_device_properties(0).gcnArchName,
    reason='Opus BF16 backward requires gfx950',
)


def make_case(t=67, d=512, i=256, e=8, k=4, routing='balanced'):
    torch.manual_seed(20261003)
    active = e // 2 if routing == 'empty' else e
    ids = torch.rand(t, active).argsort(1)[:, :k].contiguous()
    if routing == 'skew':
        ids[:, 0] = 0
        for slot in range(1, k):
            ids[:, slot] = (torch.arange(t) + slot - 1) % (e - 1) + 1
    flat = ids.flatten()
    counts = torch.bincount(flat, minlength=e)
    padded = ((counts + 31) // 32) * 32
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), padded.cumsum(0)))
    live = int(padded.sum())
    # Exercise unused capacity after the valid padded prefix, as in the sorter.
    capacity = live + 64
    packed = torch.full((capacity,), t | (k << 24), dtype=torch.int32)
    experts = torch.zeros(capacity // 32, dtype=torch.int32)
    reverse = torch.empty(t * k, dtype=torch.int32)
    for expert in range(e):
        routes = (flat == expert).nonzero().flatten()
        base = int(offsets[expert])
        packed[base:base + routes.numel()] = ((routes // k) | ((routes % k) << 24)).int()
        reverse[routes] = torch.arange(base, base + routes.numel(), dtype=torch.int32)
        experts[base // 32:int(offsets[expert + 1]) // 32] = expert
    gpu = lambda value: value.cuda().contiguous()
    metadata = opus.OpusMoeFixedMetadata(gpu(packed), gpu(experts),
        gpu(torch.tensor([live, t], dtype=torch.int32)), gpu(reverse), gpu(offsets.int()), 32)
    rand = lambda *shape: torch.randn(*shape, device='cuda', dtype=torch.bfloat16)
    x, dout = rand(t, d), rand(t, d)
    w1, w2 = rand(e, 2 * i, d), rand(e, d, i)
    w1.mul_(d ** -0.5)
    w2.mul_(i ** -0.5)
    scores = torch.softmax(torch.randn(t, k, device='cuda'), -1)
    z = torch.zeros(capacity, 2 * i, device='cuda', dtype=torch.bfloat16)
    z[metadata.reverse_sorted.long()] = rand(t * k, 2 * i)
    return x, dout, w1, w2, scores, z, metadata, ids.cuda().int()


def reference(case, b2=None):
    x, dout, w1, w2, scores, z, metadata, ids = case
    t, d = x.shape
    e, _, i = w2.shape
    k = scores.shape[1]
    dxr = torch.zeros(t * k, d, device=x.device, dtype=x.dtype)
    dz_sorted = torch.zeros_like(z)
    scaled_sorted = torch.zeros(z.shape[0], i, device=x.device, dtype=x.dtype)
    dw1, dw2 = torch.zeros_like(w1), torch.zeros_like(w2)
    db1 = torch.zeros(e, 2 * i, device=x.device, dtype=x.dtype)
    db2 = torch.zeros(e, d, device=x.device, dtype=x.dtype)
    ds = torch.zeros_like(scores).flatten()
    for expert in range(e):
        routes = (ids.flatten() == expert).nonzero().flatten()
        tokens = routes // k
        rows = metadata.reverse_sorted.long()[routes]
        gate, up = z[rows].float().chunk(2, -1)
        sigmoid = gate.sigmoid()
        silu = gate * sigmoid
        activation = silu * up
        acc = dout[tokens].float() @ w2[expert].float()
        q = acc * scores.flatten()[routes, None]
        dz = torch.cat((q * up * sigmoid * (1 + gate * (1 - sigmoid)), q * silu), 1).bfloat16()
        scaled = (activation * scores.flatten()[routes, None]).bfloat16()
        dz_sorted[rows] = dz
        scaled_sorted[rows] = scaled
        ds[routes] = (acc * activation).sum(1)
        if b2 is not None:
            ds[routes] += (dout[tokens].float() * b2[expert].float()).sum(1)
        dxr[routes] = (dz.float() @ w1[expert].float()).bfloat16()
        dw1[expert] = (dz.float().T @ x[tokens].float()).bfloat16()
        dw2[expert] = (dout[tokens].float().T @ scaled.float()).bfloat16()
        db1[expert] = dz.float().sum(0).bfloat16()
        db2[expert] = (dout[tokens].float() * scores.flatten()[routes, None]).sum(0).bfloat16()
    return opus.OpusMoeBackwardOutput(dxr.reshape(t, k, d).float().sum(1).bfloat16(),
        dw1, dw2, ds.reshape(t, k), dz_sorted, scaled_sorted, db1, db2)


def assert_output(actual, expected, bias=False):
    names = ['d_x', 'd_w1', 'd_w2', 'd_scores']
    if bias:
        names += ['d_b1', 'd_b2']
    for name in names:
        a, b = getattr(actual, name).float(), getattr(expected, name).float()
        assert torch.isfinite(a).all(), name
        # BF16 output/intermediate rounding is part of the native contract.
        relative_l2 = (a - b).norm() / b.norm().clamp_min(1e-12)
        assert relative_l2 < 0.02, (name, relative_l2.item())
        torch.testing.assert_close(a, b, rtol=0.02, atol=0.02)


@pytest.mark.parametrize('k,routing', [(1, 'balanced'), (2, 'balanced'), (4, 'empty'), (8, 'balanced'), (4, 'skew')])
def test_fixed_gradients_and_padding(k, routing):
    case = make_case(k=k, routing=routing)
    x, dout, w1, w2, scores, z, md, ids = case
    result = opus.opus_moe_backward(dout, x, z, w1, w2, scores, md)
    expected = reference(case)
    assert_output(result, expected)
    live_rows = md.reverse_sorted.long()
    torch.testing.assert_close(result.d_z_sorted[live_rows].float(), expected.d_z_sorted[live_rows].float(), rtol=0.02, atol=0.002)
    empty = torch.bincount(ids.flatten().long(), minlength=w1.shape[0]) == 0
    assert torch.count_nonzero(result.d_w1[empty]) == 0
    assert torch.count_nonzero(result.d_w2[empty]) == 0


def test_saved_caches_and_repeated_graph():
    case = make_case(t=1024)
    x, dout, w1, w2, scores, z, md, _ = case
    expected = reference(case)
    # The independent oracle constructs the forward saved activation cache.
    xs = opus.opus_moe_gather_x_blocked_g2(x, md.sorted_token_ids, md.num_valid_ids, block_m=32)
    def launch():
        return opus.opus_moe_backward(dout, x, z, w1, w2, scores, md,
            saved_a_scaled=expected.a_scaled, saved_x_sorted=xs, saved_x_sorted_blocked_g2=True)
    actual = launch()
    assert_output(actual, expected)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = launch()
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    assert_output(captured, expected)
    for name in ('d_x', 'd_w1', 'd_w2', 'd_scores'):
        assert torch.equal(getattr(captured, name), getattr(actual, name))


@pytest.mark.parametrize('requested', ['all', 'x_only', 'w2_only', 'scores_only', 'bias_only'])
def test_fixed_bias_and_autograd_attachment(requested):
    case = make_case()
    x, dout, w1, w2, scores, z, md, _ = case
    b1 = torch.randn(w1.shape[:2], device=x.device, dtype=x.dtype)
    b2 = torch.randn(w2.shape[:2], device=x.device, dtype=x.dtype)
    expected = reference(case, b2)
    actual = opus.opus_moe_backward(dout, x, z, w1, w2, scores, md, b1=b1, b2=b2)
    assert_output(actual, expected, bias=True)
    tensors = {'d_x': x, 'd_w1': w1, 'd_w2': w2, 'd_scores': scores, 'd_b1': b1, 'd_b2': b2}
    requested_names = {'all': tuple(tensors), 'x_only': ('d_x',), 'w2_only': ('d_w2',),
        'scores_only': ('d_scores',), 'bias_only': ('d_b1', 'd_b2')}[requested]
    for name, tensor in tensors.items():
        tensor.requires_grad_(name in requested_names)
    a = torch.zeros(z.shape[0], w2.shape[2], device=x.device, dtype=x.dtype)
    out = opus.opus_moe_attach_backward(torch.zeros_like(x), a, z, x, w1, w2, scores, md, b1=b1, b2=b2)
    out.backward(dout)
    for name, tensor in tensors.items():
        if name in requested_names:
            torch.testing.assert_close(tensor.grad.float(), getattr(expected, name).float(), rtol=0.02, atol=0.02)
        else:
            assert tensor.grad is None


@pytest.mark.parametrize('k', [1, 2, 4, 8])
def test_router_selected_softmax(k):
    torch.manual_seed(42)
    logits = torch.randn(37, 13, device='cuda', requires_grad=True)
    ids = torch.rand(37, 13, device='cuda').argsort(1)[:, :k].int().contiguous()
    grad = torch.randn(37, k, device='cuda')
    expected = logits.gather(1, ids.long()).softmax(1)
    d_expected = torch.autograd.grad(expected, logits, grad)[0]
    actual = opus.opus_moe_selected_softmax(logits, ids)
    d_actual = torch.autograd.grad(actual, logits, grad)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(d_actual, d_expected, rtol=1e-5, atol=1e-6)


def test_compact_routes_and_bias():
    case = make_case(k=4, routing='empty')
    x, dout, w1, w2, scores, z, md, ids = case
    # Drop a different number of routes per token, including tokens with none.
    t, k = scores.shape
    mask = torch.arange(k, device=x.device)[None, :] < (torch.arange(t, device=x.device) % (k + 1))[:, None]
    routes = mask.flatten().nonzero().flatten()
    to_token = (routes // k).int()
    route_experts = ids.flatten()[routes]
    counts = torch.bincount(route_experts.long(), minlength=w1.shape[0])
    padded = ((counts + 31) // 32) * 32
    offsets = torch.cat((torch.zeros(1, device=x.device, dtype=torch.int64), padded.cumsum(0))).int()
    capacity = int(offsets[-1]) + 32
    sorted_routes = torch.full((capacity,), routes.numel(), device=x.device, dtype=torch.int32)
    sorted_experts = torch.zeros(capacity // 32, device=x.device, dtype=torch.int32)
    zs = torch.zeros(capacity, z.shape[1], device=x.device, dtype=x.dtype)
    for expert in range(w1.shape[0]):
        selected = (route_experts == expert).nonzero().flatten()
        base = int(offsets[expert])
        sorted_routes[base:base + selected.numel()] = selected.int()
        zs[base:base + selected.numel()] = z[md.reverse_sorted.long()[routes[selected]]]
        sorted_experts[base // 32:int(offsets[expert + 1]) // 32] = expert
    token_offsets = torch.cat((torch.zeros(1, device=x.device, dtype=torch.int64), mask.sum(1).cumsum(0))).int()
    vm = opus.OpusMoeVarlenMetadata(sorted_routes, sorted_experts,
        torch.tensor([int(offsets[-1]), t], device=x.device, dtype=torch.int32), to_token, token_offsets, offsets, 32)
    b1 = torch.randn(w1.shape[:2], device=x.device, dtype=x.dtype)
    b2 = torch.randn(w2.shape[:2], device=x.device, dtype=x.dtype)
    actual = opus.opus_moe_varlen_backward(dout, x, zs, w1, w2, scores.flatten()[routes].contiguous(), vm, b1=b1, b2=b2)
    zero_scores = scores.clone()
    zero_scores[~mask] = 0
    expected = reference((x, dout, w1, w2, zero_scores, z, md, ids), b2)
    expected = replace(expected, d_scores=expected.d_scores.flatten()[routes])
    assert_output(actual, expected, bias=True)


def test_invalid_cache_contract():
    x, dout, w1, w2, scores, z, md, _ = make_case()
    with pytest.raises(ValueError, match='requires saved_x_sorted'):
        opus.opus_moe_backward(dout, x, z, w1, w2, scores, md, saved_x_sorted_blocked_g2=True)


@pytest.mark.parametrize('kid', [13, 16, 18, 19])
def test_no_store_down_requires_cache(kid):
    x, dout, w1, w2, scores, z, md, _ = make_case()
    with pytest.raises(ValueError, match='requires saved_a_scaled'):
        opus.opus_moe_backward(dout, x, z, w1, w2, scores, md, down_kernel_id=kid)


@pytest.mark.parametrize('kid', [5, 6, 8, 14, 15, 17])
def test_retired_down_kernel_is_rejected(kid):
    x, dout, w1, w2, scores, z, md, _ = make_case()
    with pytest.raises((ValueError, RuntimeError)):
        opus.opus_moe_backward(dout, x, z, w1, w2, scores, md, down_kernel_id=kid)


def test_large_partial_autograd_route_policy_is_retained():
    # Meta tensors cover the large-working-set policy without allocating a
    # synthetic multi-GB case; GPU output checks separately exercise kid16.
    dz = torch.empty(131072, 1024, device='meta', dtype=torch.bfloat16)
    w1 = torch.empty(64, 1024, 512, device='meta', dtype=torch.bfloat16)
    assert opus._select_internal_fixed_route_pair(dz, w1, 8) == (16, 1)
    case = make_case(t=1024)
    x, dout, w1, w2, scores, z, md, _ = case
    expected = reference(case)
    actual = opus.opus_moe_backward(dout, x, z, w1, w2, scores, md,
        route_dx_kernel_id=16, route_reduce_kernel_id=1)
    assert_output(actual, expected)


def test_compact_router_and_empty_token_segments():
    torch.manual_seed(42)
    logits = torch.randn(9, 7, device='cuda', requires_grad=True)
    counts = torch.tensor([0, 1, 2, 3, 0, 4, 1, 0, 2], device='cuda')
    token_offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0))).int()
    to_token = torch.repeat_interleave(torch.arange(9, device='cuda'), counts).int()
    experts = torch.cat([torch.randperm(7, device='cuda')[:int(n)] for n in counts]).int()
    expected = torch.cat([logits[token, experts[int(token_offsets[token]):int(token_offsets[token + 1])].long()].softmax(0) for token in range(9)])
    grad = torch.randn_like(expected)
    d_expected = torch.autograd.grad(expected, logits, grad)[0]
    actual = opus.opus_moe_varlen_selected_softmax(logits, experts, to_token, token_offsets)
    d_actual = torch.autograd.grad(actual, logits, grad)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(d_actual, d_expected, rtol=1e-5, atol=1e-6)
