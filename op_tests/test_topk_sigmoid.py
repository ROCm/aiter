# op_tests/test_topk_sigmoid.py
import torch

from aiter.ops.moe_op import topk_sigmoid

torch.set_default_device("cuda")


def test_topk_sigmoid_binding_loads():
    """Ensure pybind can convert torch.Tensor to at::Tensor."""
    gating_output = torch.randn(2, 2, dtype=torch.float16)
    topk_weights = torch.zeros(2, 2, dtype=torch.float32)
    topk_ids = torch.zeros(2, 2, dtype=torch.int32)
    topk_sigmoid(topk_weights, topk_ids, gating_output)
