# op_tests/test_topk_sigmoid.py
import torch

from aiter.ops.moe_op import topk_sigmoid

torch.set_default_device("cuda")


def test_topk_sigmoid():
    """Ensure topk_sigmoid matches torch.sigmoid."""
    torch.manual_seed(0)
    num_tokens, num_experts, topk = 2, 4, 2
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float16)
    topk_weights = torch.zeros(num_tokens, topk, dtype=torch.float32)
    topk_ids = torch.zeros(num_tokens, topk, dtype=torch.int32)
    topk_sigmoid(topk_weights, topk_ids, gating_output)
    ref_scores, ref_ids = torch.topk(gating_output, topk, dim=-1)
    ref_weights = torch.sigmoid(ref_scores.float())
    torch.testing.assert_close(topk_ids.long(), ref_ids.long())
    torch.testing.assert_close(topk_weights, ref_weights, atol=2e-2, rtol=2e-2)
