"""
aiter.layers - Neural Network Layer Implementations

This module provides high-level layer implementations built on top of aiter's
operator library. These layers can be used to construct complete models or
integrated into existing frameworks.

Layers:
    - Qwen3GatedDeltaNet: Linear attention layer with gated delta rule

Author: AIter Team
License: Apache 2.0
"""

from aiter.ops.triton._triton_kernels.gdn_block_sglang import Qwen3GatedDeltaNet

__all__ = [
    "Qwen3GatedDeltaNet",
]

