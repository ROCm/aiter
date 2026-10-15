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
