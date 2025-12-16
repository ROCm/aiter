"""
GDN Block - Gated Delta Network Block Implementation

This module provides the complete Qwen3GatedDeltaNet layer implementation
built on top of the gdr_sglang kernels.

Author: AIter Team
License: Apache 2.0
"""

from aiter.ops.triton._triton_kernels.gdn_block_sglang.linear_attention import (
    Qwen3GatedDeltaNet,
    RMSNormGated,
)

__all__ = [
    "Qwen3GatedDeltaNet",
    "RMSNormGated",
]


