# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.opus.megatron_moe import (
    _can_cache_activated_scaled,
    _opus_megatron_moe_metadata_raw,
    _opus_megatron_moe_poison_invalid_backward_raw,
    build_megatron_local_metadata,
    opus_megatron_grouped_mlp_attach_backward,
    pad_megatron_local_rows,
)
from aiter.ops.opus.moe_backward import opus_moe_backward

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.version.hip
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="Opus BF16 backward requires gfx950",
)


def _reference(x, w1, w2, probs, counts):
    outputs = []
    zs = []
    acts = []
    base = 0
    for expert, count in enumerate(counts):
        xe = x[base : base + count]
        z = xe @ w1[expert].transpose(0, 1)
        gate, up = z.chunk(2, dim=-1)
        act = torch.nn.functional.silu(gate) * up
        scaled = (act.float() * probs[base : base + count, None]).bfloat16()
        outputs.append(scaled @ w2[expert].transpose(0, 1))
        zs.append(z)
        acts.append(scaled)
        base += count
    return torch.cat(outputs), torch.cat(zs), torch.cat(acts)


def _reference_backward(x, w1, w2, probs, counts, z, grad):
    dx = torch.empty_like(x)
    dw1 = torch.zeros_like(w1)
    dw2 = torch.zeros_like(w2)
    ds = torch.empty_like(probs)
    base = 0
    for expert, count in enumerate(counts):
        rows = slice(base, base + count)
        gate, up = z[rows].float().chunk(2, dim=-1)
        sigmoid = gate.sigmoid()
        silu = gate * sigmoid
        activation = silu * up
        acc = grad[rows].float() @ w2[expert].float()
        q = acc * probs[rows, None]
        dz = torch.cat(
            (q * up * sigmoid * (1 + gate * (1 - sigmoid)), q * silu), dim=1
        ).bfloat16()
        scaled = (activation * probs[rows, None]).bfloat16()
        dx[rows] = (dz.float() @ w1[expert].float()).bfloat16()
        dw1[expert] = (dz.float().T @ x[rows].float()).bfloat16()
        dw2[expert] = (grad[rows].float().T @ scaled.float()).bfloat16()
        ds[rows] = (acc * activation).sum(dim=1)
        base += count
    return dx, dw1, dw2, ds


@pytest.mark.parametrize("counts", [[17, 0, 35, 5], [1, 33, 2, 64]])
@pytest.mark.parametrize("prob_dtype", [torch.bfloat16, torch.float32])
def test_megatron_adapter_gradients(counts, prob_dtype):
    torch.manual_seed(20261004)
    device = "cuda"
    d, i, e = 256, 256, len(counts)
    routes = sum(counts)
    x = torch.randn(routes, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    w1 = (
        torch.randn(e, 2 * i, d, device=device, dtype=torch.bfloat16) / d**0.5
    ).requires_grad_()
    w2 = (
        torch.randn(e, d, i, device=device, dtype=torch.bfloat16) / i**0.5
    ).requires_grad_()
    probs = torch.rand(routes, device=device, dtype=prob_dtype, requires_grad=True)
    counts_gpu = torch.tensor(counts, device=device, dtype=torch.int32)

    ref_out, z, _ = _reference(x, w1, w2, probs, counts)
    # Detach the existing forward graph: the adapter must supply every expert
    # gradient itself, exactly as it will when replacing TE autograd.
    out = ref_out.detach()
    attached = opus_megatron_grouped_mlp_attach_backward(
        out,
        z.detach(),
        x,
        w1,
        w2,
        probs,
        counts_gpu,
    )
    grad = torch.randn_like(out)
    attached.backward(grad)
    actual = (x.grad.clone(), w1.grad.clone(), w2.grad.clone(), probs.grad.clone())
    expected = _reference_backward(x, w1, w2, probs, counts, z, grad)
    for name, got, want in zip(("dx", "dw1", "dw2", "dprob"), actual, expected):
        rel = (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12)
        assert rel < 0.02, (name, rel.item())
        torch.testing.assert_close(got.float(), want.float(), rtol=0.02, atol=0.02)


def test_megatron_adapter_severs_replaced_te_graph():
    torch.manual_seed(20261012)
    routes, experts, model_dim, inter_dim = 8192, 16, 256, 1536
    counts = torch.full((experts,), routes // experts, device="cuda", dtype=torch.int32)
    x = torch.randn(
        routes, model_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    w1 = torch.randn(
        experts,
        2 * inter_dim,
        model_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    w2 = torch.randn(
        experts,
        model_dim,
        inter_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    probs = torch.rand(routes, device="cuda", dtype=torch.float32, requires_grad=True)

    te_output_source = torch.randn_like(x, requires_grad=True)
    te_fc1_source = torch.randn(
        routes,
        2 * inter_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    te_activation_source = torch.randn(
        routes,
        inter_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    attached = opus_megatron_grouped_mlp_attach_backward(
        te_output_source * 2,
        te_fc1_source * 3,
        x,
        w1,
        w2,
        probs,
        counts,
        activated_scaled=te_activation_source * 4,
    )

    attached.backward(torch.randn_like(attached))

    assert te_output_source.grad is None
    assert te_fc1_source.grad is None
    assert te_activation_source.grad is None
    assert x.grad is not None
    assert w1.grad is not None
    assert w2.grad is not None
    assert probs.grad is not None


def test_megatron_adapter_caches_only_supported_activation_geometry():
    small_counts = torch.tensor([2, 2], device="cuda", dtype=torch.int32)
    small_metadata = build_megatron_local_metadata(small_counts, 4)
    assert not _can_cache_activated_scaled(256, small_metadata)

    qwen_counts = torch.full((16,), 512, device="cuda", dtype=torch.int32)
    qwen_metadata = build_megatron_local_metadata(qwen_counts, 8192)
    assert _can_cache_activated_scaled(1536, qwen_metadata)


def test_megatron_adapter_validates_uncached_activation():
    routes, experts, model_dim, inter_dim = 4, 2, 256, 256
    x = torch.randn(routes, model_dim, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(
        experts,
        2 * inter_dim,
        model_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        experts,
        model_dim,
        inter_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    probs = torch.rand(routes, device="cuda", dtype=torch.float32)
    counts = torch.tensor([2, 2], device="cuda", dtype=torch.int32)
    output, z, _ = _reference(x, w1, w2, probs, counts.tolist())
    noncontiguous_activation = torch.empty(
        routes, 2 * inter_dim, device="cuda", dtype=torch.bfloat16
    )[:, ::2]

    with pytest.raises(ValueError, match="activated_scaled must be contiguous"):
        opus_megatron_grouped_mlp_attach_backward(
            output,
            z,
            x,
            w1,
            w2,
            probs,
            counts,
            activated_scaled=noncontiguous_activation,
        )


def test_grouped_metadata_has_no_general_sort():
    counts = torch.tensor([17, 0, 35, 5], device="cuda", dtype=torch.int32)
    md = build_megatron_local_metadata(counts, 57)
    assert md.expert_padded_offsets.cpu().tolist() == [0, 32, 32, 96, 128]
    assert md.num_valid_ids.cpu().tolist() == [128, 57, 64]
    assert md.sorted_expert_ids[:4].cpu().tolist() == [0, 2, 2, 3]
    torch.testing.assert_close(
        md.sorted_token_ids[md.reverse_sorted.long()].cpu(),
        torch.arange(57, dtype=torch.int32),
    )
    live_rows = torch.zeros(
        md.sorted_token_ids.numel(), dtype=torch.bool, device="cuda"
    )
    live_rows[md.reverse_sorted.long()] = True
    assert torch.all(md.sorted_token_ids[~live_rows] & (1 << 24))


def test_grouped_metadata_supports_all_empty_experts():
    counts = torch.zeros(4, device="cuda", dtype=torch.int32)
    md = build_megatron_local_metadata(counts, 0)
    assert md.num_valid_ids.cpu().tolist() == [0, 0, 0]
    assert md.expert_padded_offsets.cpu().tolist() == [0, 0, 0, 0, 0]
    assert md.reverse_sorted.numel() == 0
    assert torch.all(md.sorted_token_ids & (1 << 24))


@pytest.mark.parametrize(
    ("counts", "route_count"),
    [([-1, 2, 3], 4), ([3, 3], 4), ([1, 1], 4)],
)
def test_grouped_metadata_sanitizes_invalid_counts(counts, route_count):
    counts_gpu = torch.tensor(counts, device="cuda", dtype=torch.int32)
    experts = len(counts)
    capacity = route_count + experts * 32 - 1
    blocks = (capacity + 31) // 32
    options = {"device": "cuda", "dtype": torch.int32}
    sorted_token_ids = torch.empty(capacity, **options)
    sorted_expert_ids = torch.empty(blocks, **options)
    num_valid_ids = torch.empty(3, **options)
    reverse_sorted = torch.empty(route_count, **options)
    expert_padded_offsets = torch.empty(experts + 1, **options)
    _opus_megatron_moe_metadata_raw(
        counts_gpu,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        reverse_sorted,
        expert_padded_offsets,
        route_count,
        32,
    )
    assert num_valid_ids.cpu().tolist() == [0, -1, 0]
    assert expert_padded_offsets.cpu().tolist() == [0] * (experts + 1)
    assert reverse_sorted.cpu().tolist() == [0] * route_count
    assert torch.all(sorted_token_ids & (1 << 24))


def test_grouped_metadata_synchronous_validation_is_catchable():
    counts = torch.tensor([3, 3], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="sum to route_count"):
        build_megatron_local_metadata(counts, 4, synchronize_validation=True)


def test_invalid_metadata_poison_reaches_every_backward_output():
    torch.manual_seed(20261012)
    routes, experts, model_dim, inter_dim = 4, 2, 128, 128
    counts = torch.tensor([3, 3], device="cuda", dtype=torch.int32)
    x = torch.randn(
        routes,
        model_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    w1 = torch.randn(
        experts,
        2 * inter_dim,
        model_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    w2 = torch.randn(
        experts,
        model_dim,
        inter_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    probs = torch.rand(routes, device="cuda", dtype=torch.float32, requires_grad=True)
    output = torch.zeros_like(x)
    z = torch.zeros(routes, 2 * inter_dim, device="cuda", dtype=torch.bfloat16)
    attached = opus_megatron_grouped_mlp_attach_backward(
        output, z, x, w1, w2, probs, counts
    )
    attached.backward(torch.ones_like(output))
    for tensor in (x.grad, w1.grad, w2.grad, probs.grad):
        assert torch.isnan(tensor).all()


def test_grouped_row_padding_supports_unaligned_contiguous_view():
    counts = torch.tensor([3, 2], device="cuda", dtype=torch.int32)
    md = build_megatron_local_metadata(counts, 5)
    columns = 128
    storage = torch.arange(5 * columns + 1, device="cuda", dtype=torch.bfloat16)
    compact = storage[1:].view(5, columns)
    assert compact.is_contiguous()
    assert compact.data_ptr() % 16 != 0
    padded = pad_megatron_local_rows(compact, md)
    torch.testing.assert_close(padded[md.reverse_sorted.long()], compact)
    live = torch.zeros(padded.shape[0], device="cuda", dtype=torch.bool)
    live[md.reverse_sorted.long()] = True
    assert torch.count_nonzero(padded[~live]) == 0


def test_megatron_metadata_full_backward_cuda_graph_replay():
    torch.manual_seed(20261012)
    counts_cpu = [17, 0, 35, 5]
    routes, experts = sum(counts_cpu), len(counts_cpu)
    model_dim = inter_dim = 256
    counts = torch.tensor(counts_cpu, device="cuda", dtype=torch.int32)
    x = torch.randn(routes, model_dim, device="cuda", dtype=torch.bfloat16)
    d_out = torch.randn_like(x)
    w1 = torch.randn(
        experts,
        2 * inter_dim,
        model_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        experts,
        model_dim,
        inter_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    probs = torch.rand(routes, 1, device="cuda", dtype=torch.float32)
    _, compact_z, _ = _reference(x, w1, w2, probs[:, 0], counts_cpu)

    def launch():
        metadata = build_megatron_local_metadata(counts, routes)
        padded_z = pad_megatron_local_rows(compact_z, metadata)
        result = opus_moe_backward(d_out, x, padded_z, w1, w2, probs, metadata)
        _opus_megatron_moe_poison_invalid_backward_raw(
            metadata.num_valid_ids,
            result.d_x,
            result.d_w1,
            result.d_w2,
            result.d_scores,
            result.d_z_sorted,
            result.a_scaled,
        )
        return result

    eager = launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = launch()
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    for name in ("d_x", "d_w1", "d_w2", "d_scores"):
        torch.testing.assert_close(
            getattr(captured, name), getattr(eager, name), rtol=0, atol=0
        )


def test_full_backward_supports_zero_route_rank():
    experts, model_dim, inter_dim = 4, 128, 128
    counts = torch.zeros(experts, device="cuda", dtype=torch.int32)
    md = build_megatron_local_metadata(counts, 0)
    x = torch.empty(0, model_dim, device="cuda", dtype=torch.bfloat16)
    d_out = torch.empty_like(x)
    compact_z = torch.empty(0, 2 * inter_dim, device="cuda", dtype=torch.bfloat16)
    z = pad_megatron_local_rows(compact_z, md)
    w1 = torch.randn(
        experts, 2 * inter_dim, model_dim, device="cuda", dtype=torch.bfloat16
    )
    w2 = torch.randn(experts, model_dim, inter_dim, device="cuda", dtype=torch.bfloat16)
    scores = torch.empty(0, 1, device="cuda", dtype=torch.float32)
    result = opus_moe_backward(d_out, x, z, w1, w2, scores, md)
    assert result.d_x.shape == x.shape
    assert result.d_scores.shape == scores.shape
    assert torch.count_nonzero(result.d_w1) == 0
    assert torch.count_nonzero(result.d_w2) == 0
    assert torch.count_nonzero(result.d_z_sorted) == 0
    assert torch.count_nonzero(result.a_scaled) == 0


@pytest.mark.parametrize("main_grad_dtype", [torch.bfloat16, torch.float32])
def test_megatron_adapter_accumulates_repeated_backward_into_main_grad(
    main_grad_dtype,
):
    torch.manual_seed(20261012)
    counts = [17, 0, 35, 5]
    routes, experts = sum(counts), len(counts)
    model_dim = inter_dim = 256
    w1 = (
        torch.randn(
            experts,
            2 * inter_dim,
            model_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / model_dim**0.5
    ).requires_grad_()
    w2 = (
        torch.randn(
            experts,
            model_dim,
            inter_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / inter_dim**0.5
    ).requires_grad_()
    w1.main_grad = torch.zeros_like(w1, dtype=main_grad_dtype)
    w2.main_grad = torch.zeros_like(w2, dtype=main_grad_dtype)
    w1.grad_added_to_main_grad = False
    w2.grad_added_to_main_grad = False
    w1.zero_out_wgrad = True
    w2.zero_out_wgrad = True
    counts_gpu = torch.tensor(counts, device="cuda", dtype=torch.int32)
    expected_w1 = torch.zeros_like(w1, dtype=main_grad_dtype)
    expected_w2 = torch.zeros_like(w2, dtype=main_grad_dtype)

    for _ in range(2):
        x = torch.randn(
            routes,
            model_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        probs = torch.rand(
            routes, device="cuda", dtype=torch.float32, requires_grad=True
        )
        ref_out, z, _ = _reference(x, w1, w2, probs, counts)
        grad = torch.randn_like(ref_out)
        expected = _reference_backward(x, w1, w2, probs, counts, z, grad)
        expected_w1.add_(expected[1])
        expected_w2.add_(expected[2])
        attached = opus_megatron_grouped_mlp_attach_backward(
            ref_out.detach(),
            z.detach(),
            x,
            w1,
            w2,
            probs,
            counts_gpu,
            accumulate_into_main_grad=True,
        )
        attached.backward(grad)
        torch.testing.assert_close(
            x.grad.float(), expected[0].float(), rtol=0.02, atol=0.02
        )
        torch.testing.assert_close(probs.grad, expected[3], rtol=0.02, atol=0.02)

    torch.testing.assert_close(
        w1.main_grad.float(), expected_w1.float(), rtol=0.02, atol=0.02
    )
    torch.testing.assert_close(
        w2.main_grad.float(), expected_w2.float(), rtol=0.02, atol=0.02
    )
    assert w1.grad_added_to_main_grad and w2.grad_added_to_main_grad
    assert torch.count_nonzero(w1.grad) == 0
    assert torch.count_nonzero(w2.grad) == 0


def test_megatron_adapter_rejects_unimplemented_distributed_contracts():
    routes, model_dim, inter_dim = 1, 128, 128
    counts = torch.ones(1, device="cuda", dtype=torch.int32)
    x = torch.zeros(routes, model_dim, device="cuda", dtype=torch.bfloat16)
    z = torch.zeros(routes, 2 * inter_dim, device="cuda", dtype=torch.bfloat16)
    w1 = torch.zeros(1, 2 * inter_dim, model_dim, device="cuda", dtype=torch.bfloat16)
    w2 = torch.zeros(1, model_dim, inter_dim, device="cuda", dtype=torch.bfloat16)
    probs = torch.ones(routes, device="cuda", dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="tensor parallel"):
        opus_megatron_grouped_mlp_attach_backward(
            x, z, x, w1, w2, probs, counts, expert_tensor_parallel_size=2
        )
    with pytest.raises(NotImplementedError, match="activation recompute"):
        opus_megatron_grouped_mlp_attach_backward(
            x, z, x, w1, w2, probs, counts, activation_recompute=True
        )
