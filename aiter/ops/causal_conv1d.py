# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
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


@compile_ops("module_causal_conv1d_update")
def causal_conv1d_update(
    x: Tensor,                    # [batch, dim, seqlen] - new input (typically seqlen=1)
    conv_state: Tensor,           # [batch, dim, state_len] - state buffer (updated in-place)
    weight: Tensor,               # [dim, width]
    bias: Tensor,                 # [dim] or empty
    out: Tensor,                  # [batch, dim, seqlen] - output
    use_silu: bool,
    cache_seqlens: Tensor,        # [batch] or empty tensor - for circular buffer
    conv_state_indices: Tensor    # [batch] or empty tensor - for continuous batching
) -> None:
    """
    Causal 1D convolution update with state management (for inference/decoding).
    
    This function is designed for autoregressive generation where we process one (or a few)
    new tokens at a time and maintain a sliding window state buffer.
    
    Args:
        x: Input tensor [batch, dim, seqlen] - typically seqlen=1 for decoding
        conv_state: State buffer [batch, dim, state_len] - updated in-place
                   state_len >= width-1 required
        weight: Weight tensor [dim, width] - convolution weights
        bias: Bias tensor [dim] or empty tensor
        out: Output tensor [batch, dim, seqlen] - will be written
        use_silu: Whether to apply SiLU activation
        cache_seqlens: [batch] int32 tensor or empty for circular buffer mode.
                      If not empty, enables circular buffer indexing for state management.
        conv_state_indices: [batch] int32 tensor or empty for continuous batching.
                           Maps logical batch indices to physical conv_state indices.
                           Negative values indicate padding tokens (outputs zeros).
    
    Modes:
        - Non-circular mode (cache_seqlens empty): Shifts state buffer linearly
        - Circular mode (cache_seqlens not empty): Uses circular indexing (more efficient)
    
    Features:
        - Continuous batching: Different sequences can use different state slots
        - Padding token handling: Negative indices in conv_state_indices → zero output
        - In-place state update: conv_state is modified during execution
        - Optimized for small seqlen (1-4 tokens), typical for decoding
    
    Note:
        - Supports fp16, bf16, and fp32 data types
        - Kernel width support: 2, 3, 4
        - Optimized for AMD MI308 GPU (64 threads per wavefront)
        - Uses register-based sliding window for efficiency
        - Pass empty tensors (torch.empty(0, ...)) for optional parameters
    """
    ...
