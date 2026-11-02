# SPDX-License-Identifier: MIT

import torch

from atom.models.deepseek_v4_mono import linear, runner


def test_weight_view_does_not_mutate_native_children():
    original = torch.nn.Module()
    original.child = torch.nn.Linear(4, 4)
    view = runner.module_view(original)
    assert view.child.weight.data_ptr() == original.child.weight.data_ptr()
    view.child = torch.nn.Identity()
    assert isinstance(original.child, torch.nn.Linear)


def test_linear_preserves_padding_bias_and_tp_reduction(monkeypatch):
    class Native(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, 4))
            self.weight_scale = torch.ones(4, 1)
            self.bias = torch.ones(4)
            self.is_output_padded = True
            self._output_size_before_padding = 3
            self.tp_dim = 1
            self.tp_size = 4
            self.reduce_results = True

    native = Native()
    calls = []
    monkeypatch.setattr(linear, "projection", lambda *args: torch.full((2, 4), 2.0))
    from atom.model_ops import communication_op

    monkeypatch.setattr(
        communication_op,
        "tensor_model_parallel_all_reduce",
        lambda y: calls.append(y.clone()) or y * 4,
    )
    out = linear.HybridLinear(native, "pro_tp4", "wo_b")(
        torch.ones(2, 4), x_scale=torch.ones(2, 1)
    )
    assert out.shape == (2, 3)
    assert torch.equal(out, torch.full((2, 3), 12.0))
    assert len(calls) == 1
