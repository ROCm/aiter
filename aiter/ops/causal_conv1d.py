# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from ..jit.core import compile_ops


MD_NAME = "module_causal_conv1d"


@compile_ops("module_causal_conv1d")
def causal_conv1d_fwd(
    out: Tensor,      # [batch, dim, seqlen]
    x: Tensor,        # [batch, dim, seqlen]
    weight: Tensor,   # [dim, width]
    bias: Tensor,     # [dim] or empty tensor
    use_silu: bool    # whether to apply SiLU activation
) -> None:
    """
    Causal 1D convolution forward pass.
    
    Args:
        out: Output tensor [batch, dim, seqlen]
        x: Input tensor [batch, dim, seqlen]
        weight: Weight tensor [dim, width]
        bias: Bias tensor [dim] (can be empty)
        use_silu: Whether to apply SiLU activation
    
    Note:
        - Supports fp16, bf16, and fp32 data types
        - Implements causal convolution (only looks at current and past)
        - Uses optimized BlockLoad/BlockStore for memory coalescing
        - Supports kernel sizes: 2, 3, 4 (optimized), others (naive fallback)
    """
    ...

