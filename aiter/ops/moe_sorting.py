# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR
import torch.nn.functional as F

MD_NAME = "module_moe_sorting"


@compile_ops("module_moe_sorting")
def moe_sorting_fwd(topk_ids: Tensor,
                    topk_weights: Tensor,
                    sorted_token_ids: Tensor,
                    sorted_weights: Tensor,
                    sorted_expert_ids: Tensor,
                    total_tokens_post_pad: Tensor,
                    moe_buf: Tensor,
                    num_experts: int,
                    unit_size: int): ...
