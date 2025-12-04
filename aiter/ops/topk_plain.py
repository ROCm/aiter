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
    topk: int,
    largest: bool,
) -> None:
    pass
