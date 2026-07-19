# op_tests/test_softmax_topk.py
import torch
import pytest

torch.set_default_device("cuda")


def softmax_topk_ref(gating_output, topk, need_renorm):
    """Pure PyTorch reference: full softmax + topk."""
    scores = torch.nn.functional.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = scores.topk(k=topk, dim=-1, largest=True, sorted=True)
    if need_renorm:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return scores, topk_weights, topk_ids.to(torch.int32)


@pytest.mark.parametrize("num_tokens", [1, 32, 512, 4096])
@pytest.mark.parametrize("num_experts", [8, 64, 128])
@pytest.mark.parametrize("topk", [1, 2, 8])
@pytest.mark.parametrize("need_renorm", [True, False])
def test_softmax_topk(num_tokens, num_experts, topk, need_renorm):
    if topk > num_experts:
        pytest.skip("topk > num_experts")

    from aiter.ops.moe_op import softmax_topk

    torch.manual_seed(42)
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float32)

    scores_ref, weights_ref, ids_ref = softmax_topk_ref(
        gating_output, topk, need_renorm
    )

    scores = torch.empty(num_tokens, num_experts, dtype=torch.float32)
    topk_weights = torch.empty(num_tokens, topk, dtype=torch.float32)
    topk_indices = torch.empty(num_tokens, topk, dtype=torch.int32)
    token_expert_indices = torch.empty(num_tokens, topk, dtype=torch.int32)

    softmax_topk(
        scores, topk_weights, topk_indices, token_expert_indices,
        gating_output, topk, need_renorm,
    )

    torch.testing.assert_close(scores, scores_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(topk_weights, weights_ref, atol=1e-4, rtol=1e-4)

    for row in range(min(num_tokens, 16)):
        assert set(topk_indices[row].tolist()) == set(ids_ref[row].tolist()), (
            f"Row {row}: expert sets differ"
        )


def test_softmax_topk_n_zero():
    from aiter.ops.moe_op import softmax_topk

    scores = torch.empty(0, 8, dtype=torch.float32)
    topk_weights = torch.empty(0, 2, dtype=torch.float32)
    topk_indices = torch.empty(0, 2, dtype=torch.int32)
    token_expert_indices = torch.empty(0, 2, dtype=torch.int32)
    gating_output = torch.empty(0, 8, dtype=torch.float32)

    softmax_topk(
        scores, topk_weights, topk_indices, token_expert_indices,
        gating_output, 2, False,
    )
    assert scores.shape == (0, 8)


def test_softmax_topk_bf16_input():
    """Verify bf16 inputs are handled (caller should cast to fp32)."""
    from aiter.ops.moe_op import softmax_topk

    torch.manual_seed(42)
    gating_fp32 = torch.randn(32, 8, dtype=torch.float32)

    scores = torch.empty(32, 8, dtype=torch.float32)
    topk_weights = torch.empty(32, 2, dtype=torch.float32)
    topk_indices = torch.empty(32, 2, dtype=torch.int32)
    token_expert_indices = torch.empty(32, 2, dtype=torch.int32)

    softmax_topk(
        scores, topk_weights, topk_indices, token_expert_indices,
        gating_fp32, 2, True,
    )

    scores_ref, _, _ = softmax_topk_ref(gating_fp32, 2, True)
    torch.testing.assert_close(scores, scores_ref, atol=1e-5, rtol=1e-5)
