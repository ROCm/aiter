# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR, ATER_ROOT_DIR
import torch.nn.functional as F

MD_NAME = "module_pos_encoding"


@compile_ops("module_pos_encoding")
def rotary_embedding_fwd(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    cos_sin_cache: Tensor,
    is_neox: bool,
): ...


@compile_ops("module_pos_encoding")
def batched_rotary_embedding(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    cos_sin_cache: Tensor,
    is_neox: bool,
    rot_dim: int,
    cos_sin_cache_offsets: Tensor,
): ...
