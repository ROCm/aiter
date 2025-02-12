# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR
import torch.nn.functional as F


MD_NAME = "module_rope"


@compile_ops("module_rope")
def rope_fwd(
    output: Tensor,
    input: Tensor,
    freqs: Tensor
): 
    '''
    Forward propagation of traditional RoPE (Rotary Position Embedding).
    Input and output should be in "sbhd" format and freqs should be in shape of [s, 1, 1, d] and in float.
    This implemenation rotates the 2nd half of elements.
    '''
    ...

@compile_ops("module_rope")
def rope_bwd(
    input_grads: Tensor,
    output_grads: Tensor,
    freqs: Tensor
): 
    '''
    Backward propagation of traditional RoPE (Rotary Position Embedding).
    Input and output should be in "sbhd" format and freqs should be in shape of [s, 1, 1, d] and in float.
    This implemenation rotates the 2nd half of elements.
    '''
    ...

@compile_ops("module_rope")
def rope_cached_fwd(
    output: Tensor,
    input: Tensor,
    cos: Tensor,
    sin: Tensor
): 
    '''
    Forward propagation of RoPE (Rotary Position Embedding) with cached cos and sin.
    Input and output should be in "sbhd" format, and cos and sin should be in shape of [s, 1, 1, d] and in float.
    This implemenation rotates the 2nd half of elements.
    '''
    ...

@compile_ops("module_rope")
def rope_cached_bwd(
    input_grads: Tensor,
    output_grads: Tensor,
    cos: Tensor,
    sin: Tensor
): 
    '''
    Backward propagation of RoPE (Rotary Position Embedding) with cached cos and sin.
    Input and output should be in "sbhd" format, and cos and sin should be in shape of [s, 1, 1, d] and in float.
    This implemenation rotates the 2nd half of elements.
    '''
    ...

@compile_ops("module_rope")
def rope_thd_fwd(
    output: Tensor,
    input: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor
):
    '''
    Forward propagation of RoPE (Rotary Position Embedding) with input sizes: (t, h, d).
    where t is cumulative sum of sequence lengths.
    This implemenation rotates the 2nd half of elements.
    '''
    ...

@compile_ops("module_rope")
def rope_thd_bwd(
    input_grads: Tensor,
    output_grads: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor
):
    '''
    Backward propagation of RoPE (Rotary Position Embedding) with input sizes: (t, h, d).
    where t is cumulative sum of sequence lengths.
    This implemenation rotates the 2nd half of elements.
    '''
    ...

@compile_ops("module_rope")
def rope_2d_fwd(
    output: Tensor,
    input: Tensor,
    cos_height: Tensor,
    sin_height: Tensor,
    cos_width: Tensor,
    sin_width: Tensor
):
    '''
    Forward propagation of RoPE (Rotary Position Embedding) with 2D image as input.
    This implemenation rotates the 2nd half of elements.
    '''
    ...

@compile_ops("module_rope")
def rope_2d_bwd(
    input_grads: Tensor,
    output_grads: Tensor,
    cos_height: Tensor,
    sin_height: Tensor,
    cos_width: Tensor,
    sin_width: Tensor
):
    '''
    Backward propagation of RoPE (Rotary Position Embedding) with 2D image as input.
    This implemenation rotates the 2nd half of elements.
    '''
    ...