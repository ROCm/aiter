"""Padded graph rows must not turn NaN router scores into invalid expert IDs."""

import pytest
import torch

from aiter import biased_grouped_topk


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 4, 16, 256])
@pytest.mark.parametrize("finite_experts", [0, 1, 7, 8])
def test_biased_grouped_topk_nan_rows(dtype, tokens, finite_experts):
    torch.manual_seed(190)
    logits = torch.randn(tokens, 256, device="cuda", dtype=dtype)
    bias = torch.randn(256, device="cuda", dtype=dtype) * 0.01
    ids = torch.full((tokens, 9), 256, device="cuda", dtype=torch.int32)
    weights = torch.ones((tokens, 9), device="cuda", dtype=torch.float32)

    def run():
        biased_grouped_topk(logits, bias, weights[:, :8], ids[:, :8], 1, 1, True, 2.5)

    run()
    reference_ids, reference_weights = ids.clone(), weights.clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    logits[-1, finite_experts:] = float("nan")
    for replay in [False, True]:
        ids[:, :8] = -1
        if replay:
            graph.replay()
        else:
            run()
        assert bool(((ids[:, :8] >= 0) & (ids[:, :8] < 256)).all())
        assert bool((ids[:, 8] == 256).all())
        assert torch.equal(ids[:-1], reference_ids[:-1])
        torch.testing.assert_close(weights[:-1], reference_weights[:-1], rtol=0, atol=0)
        if finite_experts == 0:
            # All-NaN input remains visible; only index safety is repaired.
            assert torch.equal(ids[-1, :8], torch.arange(8, device="cuda"))
            assert bool(torch.isnan(weights[-1, :8]).all())
