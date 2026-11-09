"""Torch references of the activations in _triton_kernels/common/activation.py."""

import torch

LOG2_E = 1.44269504089


def silu_exp2_ref(t: torch.Tensor) -> torch.Tensor:
    """Match ``_silu_exp2`` in Triton (same as MoE silu-fused path)."""
    x = t.float()
    return x / (1.0 + torch.exp2(-(x * LOG2_E)))
