# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor, empty, empty_like
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR
import torch.nn.functional as F


MD_NAME = "module_rope"


@compile_ops("module_rope")
def rope_fwd_impl(
    output: Tensor,
    input: Tensor,
    freqs: Tensor
): 
    '''
    Forward propagation of traditional RoPE (Rotary Position Embedding).
    Input and output should be in "sbhd" format and freqs should be in shape of [s, 1, 1, d].
    This implemenation rotates the 2nd half of elements.
    '''
    ...

def rope_fwd(
    input: Tensor,
    freqs: Tensor,
    transpose_output: bool = False
) -> Tensor :
    s, b, h, d = input.shape
    output = empty((b, s, h, d), dtype=input.dtype, device=input.device, requires_grad=False).transpose(0, 1)\
        if transpose_output else empty_like(input, requires_grad=False)
    rope_fwd_impl(output, input, freqs)
    return output

@compile_ops("module_rope")
def rope_bwd_impl(
    input_grads: Tensor,
    output_grads: Tensor,
    freqs: Tensor
): 
    '''
    Backward propagation of traditional RoPE (Rotary Position Embedding).
    Input and output should be in "sbhd" format and freqs should be in shape of [s, 1, 1, d].
    This implemenation rotates the 2nd half of elements.
    '''
    ...

def rope_bwd(
    output_grads: Tensor,
    freqs: Tensor,
    transpose_output: bool = False
) -> Tensor :
    s, b, h, d = output_grads.shape
    input_grads = empty((b, s, h, d), dtype=output_grads.dtype, device=output_grads.device, requires_grad=False).transpose(0, 1)\
        if transpose_output else empty_like(output_grads, requires_grad=False)
    rope_bwd_impl(input_grads, output_grads, freqs)
    return input_grads

@compile_ops("module_rope")
def rope_cached_fwd_impl(
    output: Tensor,
    input: Tensor,
    cos: Tensor,
    sin: Tensor
): 
    '''
    Forward propagation of RoPE (Rotary Position Embedding) with cached cos and sin.
    Input and output should be in "sbhd" format, and cos and sin should be in shape of [s, 1, 1, d].
    This implemenation rotates the 2nd half of elements.
    '''
    ...

def rope_cached_fwd(
    input: Tensor,
    cos: Tensor,
    sin: Tensor,
    transpose_output: bool = False
) -> Tensor :
    s, b, h, d = input.shape
    output = empty((b, s, h, d), dtype=input.dtype, device=input.device, requires_grad=False).transpose(0, 1)\
        if transpose_output else empty_like(input, requires_grad=False)
    rope_cached_fwd_impl(output, input, cos, sin)
    return output

@compile_ops("module_rope")
def rope_cached_bwd_impl(
    input_grads: Tensor,
    output_grads: Tensor,
    cos: Tensor,
    sin: Tensor
): 
    '''
    Backward propagation of RoPE (Rotary Position Embedding) with cached cos and sin.
    Input and output should be in "sbhd" format, and cos and sin should be in shape of [s, 1, 1, d].
    This implemenation rotates the 2nd half of elements.
    '''
    ...

def rope_cached_bwd(
    output_grads: Tensor,
    cos: Tensor,
    sin: Tensor,
    transpose_output: bool = False
) -> Tensor :
    s, b, h, d = output_grads.shape
    input_grads = empty((b, s, h, d), dtype=output_grads.dtype, device=output_grads.device, requires_grad=False).transpose(0, 1)\
        if transpose_output else empty_like(output_grads, requires_grad=False)
    rope_cached_bwd_impl(input_grads, output_grads, cos, sin)
    return input_grads

@compile_ops("module_rope")
def rope_thd_fwd_impl(
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

def rope_thd_fwd(
    input: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor
) -> Tensor :
    output = empty_like(input, requires_grad=False)
    rope_thd_fwd_impl(output, input, cu_seqlens, freqs)
    return output

@compile_ops("module_rope")
def rope_thd_bwd_impl(
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

def rope_thd_bwd(
    output_grads: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor
) -> Tensor :
    input_grads = empty_like(output_grads, requires_grad=False)
    rope_thd_bwd_impl(input_grads, output_grads, cu_seqlens, freqs)
    return input_grads

@compile_ops("module_rope")
def rope_2d_fwd_impl(
    output: Tensor,
    input: Tensor,
    cos_h: Tensor,
    sin_h: Tensor,
    cos_w: Tensor,
    sin_w: Tensor
):
    '''
    Forward propagation of RoPE (Rotary Position Embedding) with 2D image as input.
    Input size should be (b, H, W, h, d) while output should be in (b, s, h, d) where s = H * W.
    cos_h and sin_h are in (1, H', 1, h, d // 2) where H' >= H.
    cos_w and sin_w are in (1, 1, W', h, d // 2) where W' >= W.
    This implemenation rotates the 2nd half of elements.
    '''
    ...

def rope_2d_fwd(
    input: Tensor,
    height: int,
    width: int,
    cos_h: Tensor,
    sin_h: Tensor,
    cos_w: Tensor,
    sin_w: Tensor
) -> Tensor :
    '''
    Input and output are in (b, s, h, d) where s = H * W.
    '''
    b, s, h, d = input.shape
    input = input.view(b, height, width, h, d)
    output = empty((b, s, h, d), dtype=input.dtype, device=input.device, requires_grad=False)
    rope_2d_fwd_impl(output, input, cos_h, sin_h, cos_w, sin_w)
    return output

@compile_ops("module_rope")
def rope_2d_bwd_impl(
    input_grads: Tensor,
    output_grads: Tensor,
    cos_h: Tensor,
    sin_h: Tensor,
    cos_w: Tensor,
    sin_w: Tensor
):
    '''
    Backward propagation of RoPE (Rotary Position Embedding) with 2D image as input.
    output_grads size should be (b, H, W, h, d) while input_grads should be in (b, s, h, d) where s = H * W.
    cos_h and sin_h are in (1, H', 1, h, d // 2) where H' >= H.
    cos_w and sin_w are in (1, 1, W', h, d // 2) where W' >= W.
    This implemenation rotates the 2nd half of elements.
    '''
    ...

def rope_2d_bwd(
    output_grads: Tensor,
    height: int,
    width: int,
    cos_h: Tensor,
    sin_h: Tensor,
    cos_w: Tensor,
    sin_w: Tensor
) -> Tensor :
    '''
    output_grads and input_grads are in (b, s, h, d) where s = H * W.
    '''
    b, s, h, d = output_grads.shape
    output_grads = output_grads.view(b, height, width, h, d)
    input_grads = empty((b, s, h, d), dtype=output_grads.dtype, device=output_grads.device, requires_grad=False)
    rope_2d_bwd_impl(input_grads, output_grads, cos_h, sin_h, cos_w, sin_w)
    return input_grads
