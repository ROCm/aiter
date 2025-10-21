# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from ..jit.core import compile_ops

@compile_ops("module_mqa_logits")
def paged_mqa_logits_metadata(
    context_lens: Tensor,
    schedule_metadata: Tensor,
    batch_size: int,
    block_size: int,
    cu_num: int,
) -> None: ...


def get_paged_mqa_logits_metadata(
    context_lens: Tensor,
    block_size: int = 128,
    cu_num: int = 80,
) -> Tensor:
    batch_size = context_lens.size(0)

    schedule_metadata = torch.empty([cu_num + 1, 2], dtype=torch.int32, device=context_lens.device)

    paged_mqa_logits_metadata(context_lens,
                              schedule_metadata,
                              batch_size,
                              block_size,
                              cu_num,
                              )

    return schedule_metadata
