# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rotary Positional Embeddings."""
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from aiter import dtypes

# from custom_op import CustomOp


def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


# class RotaryEmbedding(CustomOp):
class RotaryEmbedding(nn.Module):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cos, sin = self._compute_cos_sin_cache()
        cos = cos.to(dtype)
        sin = sin.to(dtype)
        self.cos_cache: torch.Tensor
        self.sin_cache: torch.Tensor
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=dtypes.fp32) / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=dtypes.fp32)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos().unsqueeze(-2).unsqueeze(-2)
        sin = freqs.sin().unsqueeze(-2).unsqueeze(-2)
        return cos, sin

    def forward(self, *args, **kwargs):
        if torch.compiler.is_compiling():
            return self.forward_native(*args, **kwargs)
        else:
            return self.forward_hip(*args, **kwargs)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        is_nope_first=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets.view_as(positions)
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        # cos_sin = self.cos_sin_cache.index_select(0, positions)
        # cos, sin = cos_sin.chunk(2, dim=-1)
        cos = self.cos_cache.index_select(0, positions).squeeze(-2).squeeze(-2)
        sin = self.sin_cache.index_select(0, positions).squeeze(-2).squeeze(-2)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = (
            query[..., : self.rotary_dim]
            if not is_nope_first
            else query[..., -self.rotary_dim :]
        )
        query_pass = (
            query[..., self.rotary_dim :]
            if not is_nope_first
            else query[..., : -self.rotary_dim]
        )
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = (
            torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
            if not is_nope_first
            else torch.cat((query_pass, query_rot), dim=-1).reshape(query_shape)
        )
        
        if key is None:
            return query

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = (
            key[..., : self.rotary_dim]
            if not is_nope_first
            else key[..., -self.rotary_dim :]
        )
        key_pass = (
            key[..., self.rotary_dim :]
            if not is_nope_first
            else key[..., : -self.rotary_dim]
        )
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = (
            torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
            if not is_nope_first
            else torch.cat((key_pass, key_rot), dim=-1).reshape(key_shape)
        )
        return query, key

    # def forward_cuda(
    def forward_old(
        self,
        positions: torch.Tensor,
        # if     is_nope_first
        # [num_tokens, num_heads, nope_size+rope_size]
        # if NOT is_nope_first
        # [num_tokens, num_heads, rope_size+nope_size],
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        is_nope_first=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # from vllm import _custom_ops as ops
        import aiter as ops

        self.cos_cache = self.cos_cache.to(query.device, dtype=query.dtype)
        self.sin_cache = self.sin_cache.to(query.device, dtype=query.dtype)
        # ops.rotary_embedding()/batched_rotary_embedding()
        # are in-place operations that update the query and key tensors.
        if offsets is not None:
            ops.batched_rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                self.cos_cache,
                self.sin_cache,
                self.is_neox_style,
                is_nope_first,
                self.rotary_dim,
                offsets,
            )
        else:
            ops.rotary_embedding_fwd(
                positions,
                query,
                key,
                self.head_size,
                self.cos_cache,
                self.sin_cache,
                self.is_neox_style,
                is_nope_first,
            )
        return query, key

    def forward_hip(
        self,
        positions: torch.Tensor,
        # if     is_nope_first
        # [[batch_size, seq_len, num_heads, nope_size+rope_size]
        # if NOT is_nope_first
        # [[batch_size, seq_len, num_heads, rope_size+nope_size],
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        is_nope_first=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # import aiter as ops
        import aiter.ops.triton.rope as ops

        self.cos_cache = self.cos_cache.to(query.device, dtype=query.dtype)
        self.sin_cache = self.sin_cache.to(query.device, dtype=query.dtype)
        cos, sin = self.cos_cache, self.sin_cache

        rotate_style = 0 if self.is_neox_style else 1

        num_tokens = positions.numel()

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size) # fixed to thd layout
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size) # fixed to thd layout

        positions = positions.view(*query.shape[:2])
        if offsets is not None:
            offsets = offsets.view(*query.shape[:2])

        if not is_nope_first:
            query_ = query[..., : self.rotary_dim]
            key_ = key[..., : self.rotary_dim] if key is not None else None
        else:
            query_ = query[..., -self.rotary_dim :]
            key_ = key[..., -self.rotary_dim :] if key is not None else None

        if key_ is not None:
            if offsets is None:
                # ops.rope_cached_positions_2c_fwd_inplace(
                ops.rope_cached_thd_positions_2c_fwd_inplace(
                    query_,
                    key_,
                    cos,
                    sin,
                    positions,
                    rotate_style,
                    reuse_freqs_front_part=True,
                    nope_first=is_nope_first,
                )
            else:
                raise NotImplementedError("RoPE style not implemented yet")
                ops.rope_cached_positions_offsets_2c_fwd_inplace(
                    query_,
                    key_,
                    cos,
                    sin,
                    positions,
                    offsets,
                    rotate_style,
                    reuse_freqs_front_part=True,
                    nope_first=is_nope_first,
                )
            return query.view(query_shape), key.view(key_shape)
        else:
            raise NotImplementedError("RoPE style not implemented yet")
            if offsets is None:
                ops.rope_cached_positions_fwd_inplace(
                    query_,
                    cos,
                    sin,
                    positions,
                    rotate_style,
                    reuse_freqs_front_part=True,
                    nope_first=is_nope_first,
                )
            else:
                ops.rope_cached_positions_offsets_fwd_inplace(
                    query_,
                    cos,
                    sin,
                    positions,
                    offsets,
                    rotate_style,
                    reuse_freqs_front_part=True,
                    nope_first=is_nope_first,
                )
            return query.view(query_shape)

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s

_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}

def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if rope_scaling is None:
        rotary_emb = RotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype
        )
    else:
        raise ValueError(f"Unknown RoPE")
        
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb

def get_rope_wrapper(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
    device: Optional[str] = None,
):
    if device != "cpu":
        return get_rope(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            rope_scaling,
            dtype,
            partial_rotary_factor,
        )

    assert False, "get_rope_wrapper in AITER is not implemented for cpu device"