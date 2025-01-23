# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor, Generator
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR, AITER_ROOT_DIR
import torch.nn.functional as F

MD_NAME = "module_mha_fwd"

@compile_ops("module_mha_fwd", fc_name="mha_fwd")
def mha_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...

