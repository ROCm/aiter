# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from ..jit.core import compile_ops

MD_NAME = "module_pos_encoding"


@compile_ops("module_pos_encoding")
def rotary_embedding_fwd(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    cos_cache: Tensor,
    sin_cache: Tensor,
    is_neox: bool,
    is_nope_first: bool,
): ...


@compile_ops("module_pos_encoding")
def batched_rotary_embedding(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    cos_cache: Tensor,
    sin_cache: Tensor,
    is_neox: bool,
    is_nope_first: bool,
    rot_dim: int,
    cos_sin_cache_offsets: Tensor,
): ...
