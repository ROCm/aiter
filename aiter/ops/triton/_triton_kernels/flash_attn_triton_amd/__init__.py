"""Compatibility imports for FlashAttention consumers using the former Dao-AI path."""

import sys

from aiter.ops.triton._triton_kernels.attention.dao_ai import (
    bwd,
    flash_attn_2,
    flash_attn_3,
    interface_v2,
    utils,
)

# FlashAttention main imports the utilities as a submodule, not a package attribute.
sys.modules[__name__ + ".utils"] = utils

__all__ = ["bwd", "flash_attn_2", "flash_attn_3", "interface_v2", "utils"]
