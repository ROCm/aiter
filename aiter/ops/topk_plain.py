# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

# user interface

from typing import Tuple
import torch
from ..jit.core import (
    compile_ops,
)
from ..utility import dtypes
from ..jit.utils.chip_info import get_cu_num

@compile_ops("module_topk_plain")
def topk_plain(
    output: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_num: int,
) -> None: ...
