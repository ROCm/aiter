# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

# user interface

import torch
from ..jit.core import (
    compile_ops,
)


@compile_ops("module_topk_plain")
def topk_plain(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_out: torch.Tensor,
    topk: int,
    largest: bool = True,
    rowStarts: torch.Tensor = None,
    rowEnds: torch.Tensor = None,
    stride0: int = -1,
    stride1: int = 1,
) -> None:
    pass
