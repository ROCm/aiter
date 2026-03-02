# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from ..jit.core import compile_ops

MD_NAME = "module_causal_conv1d_update"


@compile_ops("module_causal_conv1d_update")
def causal_conv1d_update(
    x: Tensor,
    conv_state: Tensor,
    weight: Tensor,
    bias: Tensor,
    out: Tensor,
    use_silu: bool,
    cache_seqlens: Tensor,
    conv_state_indices: Tensor,
    pad_slot_id: int,
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
             IMPORTANT: Initialize with torch.zeros_like() instead of torch.empty_like()
             when using padding (pad_slot_id) to ensure padded outputs are zero.
        use_silu: Whether to apply SiLU activation
        cache_seqlens: [batch] int32 tensor or empty for circular buffer mode.
                      If not empty, enables circular buffer indexing for state management.
        conv_state_indices: [batch] int32 tensor or empty for continuous batching.
                           Maps logical batch indices to physical conv_state indices.
        pad_slot_id: Padding slot ID. If conv_state_indices[i] == pad_slot_id, skip processing.

    Modes:
        - Non-circular mode (cache_seqlens empty): Shifts state buffer linearly
        - Circular mode (cache_seqlens not empty): Uses circular indexing (more efficient)

    Features:
        - Continuous batching: Different sequences can use different state slots
        - Padding token handling: conv_state_indices[i] == pad_slot_id → skip processing
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
