# SPDX-License-Identifier: MIT
"""Bitwise routing parity with AITER, including graph replay and DP padding."""

import pytest
import torch


@pytest.mark.parametrize("bs,q", [(1, 1), (3, 4), (32, 1), (64, 4), (512, 4)])
@pytest.mark.parametrize("scale", [0, 0.01, 1, 16, 64, 256])
def test_bitwise_weights_and_ids_on_replay(bs, q, scale):
    from aiter import topk_softmax
    from aiter.ops.triton.moe.fused_topk_dispatch import fused_topk_dispatch

    torch.manual_seed(123)
    n = bs * q
    logits = (
        torch.randn(n, 257, device="cuda", dtype=torch.float32) * scale
    ).bfloat16()[:, :256]
    weights = torch.empty(n, 8, device="cuda", dtype=torch.float32)
    ids = torch.empty(n, 8, device="cuda", dtype=torch.int32)
    reference_weights, reference_ids = torch.empty_like(weights), torch.empty_like(ids)
    topk_softmax(reference_weights, reference_ids, torch.empty_like(ids), logits, True)
    cu = torch.arange(bs + 1, device="cuda", dtype=torch.int32) * q
    requests = torch.empty((), device="cuda", dtype=torch.int64)
    pad = torch.empty(n, device="cuda", dtype=torch.bool)
    recv = torch.empty((), device="cuda", dtype=torch.int32)

    def launch(draft):
        fused_topk_dispatch(
            logits,
            weights,
            ids,
            cu,
            requests,
            pad,
            recv,
            batch_size=bs,
            is_draft=draft,
            write_requests=not draft,
        )

    launch(False)
    launch(True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(False)
        launch(True)
    for live in sorted({0, 1, max(0, bs - 1), bs}):
        cu.copy_(
            torch.arange(bs + 1, device="cuda", dtype=torch.int32).clamp(max=live) * q
        )
        recv.fill_(12345)
        requests.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert requests.item() == live
        assert recv.item() == 0
        assert torch.equal(ids[: live * q], reference_ids[: live * q])
        assert (ids[live * q :] == -1).all()
        assert torch.equal(pad, torch.arange(n, device="cuda") >= live * q)
        assert torch.equal(
            weights[: live * q].view(torch.int32),
            reference_weights[: live * q].view(torch.int32),
        )


@pytest.mark.parametrize("n,bs,real_requests", [(4, 2, 1), (8, 4, 2)])
def test_eager_draft_real_rows_with_padded_request_count(n, bs, real_requests):
    """MTP step 0 has four real rows/request, even if running_bs is padded."""
    from aiter import topk_softmax
    from aiter.ops.triton.moe.fused_topk_dispatch import fused_topk_dispatch

    logits = torch.randn(n, 256, device="cuda", dtype=torch.bfloat16)
    weights = torch.empty(n, 8, device="cuda", dtype=torch.float32)
    ids = torch.empty(n, 8, device="cuda", dtype=torch.int32)
    ref_weights, ref_ids = torch.empty_like(weights), torch.empty_like(ids)
    topk_softmax(ref_weights, ref_ids, torch.empty_like(ids), logits, True)
    cu = (
        torch.arange(bs + 1, device="cuda", dtype=torch.int32).clamp(max=real_requests)
        * 4
    )
    requests = torch.tensor(real_requests, device="cuda", dtype=torch.int64)
    pad = torch.empty(n, device="cuda", dtype=torch.bool)
    recv = torch.ones((), device="cuda", dtype=torch.int32)
    fused_topk_dispatch(
        logits,
        weights,
        ids,
        cu,
        requests,
        pad,
        recv,
        batch_size=bs,
        is_draft=True,
        write_requests=False,
        no_padding=True,
    )
    assert not pad.any()
    assert torch.equal(ids, ref_ids)
    assert torch.equal(weights.view(torch.int32), ref_weights.view(torch.int32))
    assert recv.item() == 0


def test_nan_dummy_row_uses_aiter_expert_id_fallback():
    from aiter import topk_softmax
    from aiter.ops.triton.moe.fused_topk_dispatch import fused_topk_dispatch

    logits = torch.full((1, 256), float("nan"), device="cuda", dtype=torch.bfloat16)
    weights = torch.empty(1, 8, device="cuda", dtype=torch.float32)
    ids = torch.empty(1, 8, device="cuda", dtype=torch.int32)
    ref_weights, ref_ids = torch.empty_like(weights), torch.empty_like(ids)
    topk_softmax(ref_weights, ref_ids, torch.empty_like(ids), logits, True)
    cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    requests = torch.ones((), device="cuda", dtype=torch.int64)
    pad = torch.empty(1, device="cuda", dtype=torch.bool)
    recv = torch.ones((), device="cuda", dtype=torch.int32)
    fused_topk_dispatch(
        logits,
        weights,
        ids,
        cu,
        requests,
        pad,
        recv,
        batch_size=1,
        is_draft=False,
        write_requests=True,
        no_padding=True,
    )
    assert torch.equal(ids, ref_ids)
    assert weights.isnan().all() and ref_weights.isnan().all()


@pytest.mark.parametrize("count_dtype", [torch.int32, torch.int64])
def test_explicit_valid_rows_on_graph_replay(count_dtype):
    from aiter import topk_softmax
    from aiter.ops.triton.moe.fused_topk_dispatch import fused_topk_dispatch

    n = 12
    logits = torch.randn((n, 256), device="cuda", dtype=torch.bfloat16)
    weights = torch.empty((n, 8), device="cuda", dtype=torch.float32)
    ids = torch.empty((n, 8), device="cuda", dtype=torch.int32)
    ref_weights, ref_ids = torch.empty_like(weights), torch.empty_like(ids)
    topk_softmax(ref_weights, ref_ids, torch.empty_like(ids), logits, True)
    pad = torch.empty(n, device="cuda", dtype=torch.bool)
    recv = torch.ones((), device="cuda", dtype=torch.int32)
    count = torch.tensor(n, device="cuda", dtype=count_dtype)

    def launch():
        fused_topk_dispatch(
            logits,
            weights,
            ids,
            None,
            None,
            pad,
            recv,
            batch_size=1,
            is_draft=False,
            write_requests=False,
            num_valid_rows=count,
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for live in (0, 1, 7, n):
        count.fill_(live)
        recv.fill_(123)
        graph.replay()
        torch.cuda.synchronize()
        assert recv.item() == 0
        assert torch.equal(ids[:live], ref_ids[:live])
        assert (ids[live:] == -1).all()
        assert torch.equal(pad, torch.arange(n, device="cuda") >= live)
        assert torch.equal(weights.view(torch.int32), ref_weights.view(torch.int32))
    fused_topk_dispatch(
        logits,
        weights,
        ids,
        None,
        None,
        pad,
        recv,
        batch_size=1,
        is_draft=False,
        write_requests=False,
        no_padding=True,
    )
    assert not pad.any()
    assert torch.equal(ids, ref_ids)
