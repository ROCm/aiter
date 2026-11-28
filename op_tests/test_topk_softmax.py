# op_tests/test_topk_softmax.py
import pytest
import torch

from aiter.ops.moe_op import topk_softmax

torch.set_default_device("cuda")


def topk_softmax_ref(gating_output, topk, need_renorm):
    """Pure PyTorch reference: full softmax + topk."""
    scores = torch.nn.functional.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = scores.topk(k=topk, dim=-1, largest=True, sorted=True)
    if need_renorm:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids.to(torch.int32)


def _run_topk_softmax(gating_output, topk, need_renorm):
    num_tokens = gating_output.shape[0]
    topk_weights = torch.empty(num_tokens, topk, dtype=torch.float32)
    topk_indices = torch.empty(num_tokens, topk, dtype=torch.int32)
    token_expert_indices = torch.empty(num_tokens, topk, dtype=torch.int32)
    topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        need_renorm,
    )
    return topk_weights, topk_indices


@pytest.mark.parametrize("num_tokens", [1, 32, 512, 4096])
@pytest.mark.parametrize("num_experts", [8, 64, 128])
@pytest.mark.parametrize("topk", [1, 2, 8])
@pytest.mark.parametrize("need_renorm", [True, False])
def test_topk_softmax(num_tokens, num_experts, topk, need_renorm):
    if topk > num_experts:
        pytest.skip("topk > num_experts")

    torch.manual_seed(42)
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float32)

    weights_ref, ids_ref = topk_softmax_ref(gating_output, topk, need_renorm)
    topk_weights, topk_indices = _run_topk_softmax(gating_output, topk, need_renorm)

    torch.testing.assert_close(topk_weights, weights_ref, atol=1e-4, rtol=1e-4)

    for row in range(min(num_tokens, 16)):
        assert set(topk_indices[row].tolist()) == set(
            ids_ref[row].tolist()
        ), f"Row {row}: expert sets differ"


def _nan_rows(num_experts, topk, dtype):
    """Row 0 all NaN and row 3 all -inf (no valid expert); row 1 NaN with
    topk - 1 valid experts; row 2 -inf with one valid expert."""
    torch.manual_seed(42)
    gating_output = torch.randn(4, num_experts, dtype=torch.float32)
    gating_output[0] = float("nan")
    gating_output[1] = float("nan")
    valid = list(range(5, 5 + topk - 1))
    gating_output[1, valid] = torch.arange(topk - 1, dtype=torch.float32) + 1.0
    gating_output[2] = float("-inf")
    gating_output[2, 3] = 0.0
    gating_output[3] = float("-inf")
    return gating_output.to(dtype), {1: set(valid), 2: {3}}


def _check_nan_rows(topk_weights, topk_indices, num_experts, topk, valid):
    assert topk_weights.isfinite().all()
    for row in range(4):
        ids = topk_indices[row].tolist()
        assert len(set(ids)) == topk, f"Row {row}: repeated ids {ids}"
        assert all(0 <= i < num_experts for i in ids), f"Row {row}: {ids}"
    for row in (0, 3):
        assert (topk_weights[row] == 0).all(), f"Row {row}: {topk_weights[row]}"
    for row, experts in valid.items():
        ids = topk_indices[row].tolist()
        assert experts <= set(ids), f"Row {row}: {ids}"
        weighted = {i for i, w in zip(ids, topk_weights[row].tolist()) if w != 0}
        assert weighted == experts, f"Row {row}: weighted {weighted}"
        torch.testing.assert_close(
            topk_weights[row].sum(), torch.tensor(1.0), atol=1e-5, rtol=0
        )


@pytest.mark.parametrize("num_experts", [64, 256, 512, 160])
@pytest.mark.parametrize("need_renorm", [True, False])
def test_topk_softmax_nan_rows_distinct_ids(num_experts, need_renorm):
    """A row with fewer than topk valid experts must still yield topk distinct
    ids: the fused MoE kernels queue every (token, expert) pair, weight 0 or not.
    A row with no valid expert routes nowhere (weight 0), as in topk_gating,
    rather than to topk experts at the uniform 1/E its clamped logits would
    softmax to. E=160 takes the moeTopK fallback, the others the fused kernel."""
    topk = 8
    gating_output, valid = _nan_rows(num_experts, topk, torch.float32)
    topk_weights, topk_indices = _run_topk_softmax(gating_output, topk, need_renorm)
    _check_nan_rows(topk_weights, topk_indices, num_experts, topk, valid)


def test_topk_softmax_n_zero():
    pytest.skip(
        "topk_softmax ASM kernel does not support 0 tokens (invalid HIP launch)"
    )


def test_topk_softmax_bf16_input():
    """ASM topk_softmax expects fp32 logits; caller must cast bf16."""
    torch.manual_seed(42)
    gating_bf16 = torch.randn(32, 8, dtype=torch.bfloat16)
    gating_fp32 = gating_bf16.float()

    weights_ref, ids_ref = topk_softmax_ref(gating_fp32, 2, True)
    topk_weights, topk_indices = _run_topk_softmax(gating_fp32, 2, True)

    torch.testing.assert_close(topk_weights, weights_ref, atol=2e-3, rtol=2e-3)
    for row in range(16):
        assert set(topk_indices[row].tolist()) == set(
            ids_ref[row].tolist()
        ), f"Row {row}: expert sets differ"
