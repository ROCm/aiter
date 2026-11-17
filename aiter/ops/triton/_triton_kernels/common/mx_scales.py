# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def load_e8m0_scales_k_tail(scale_ptrs, offs_scale_k, num_groups_left):
    """Load a block of e8m0 scales whose K groups can run past the end of the tensor.

    `offs_scale_k` indexes the scale groups along the last axis of `scale_ptrs`, and
    `num_groups_left` is the number of groups left in K from the start of this block.
    The groups past it read as 127, a scale of 2^0. They are not part of the tensor,
    and 0xFF there is the e8m0 NaN, which `tl.dot_scaled` propagates even where the
    data of that group is masked to zero.
    """
    return tl.load(scale_ptrs, mask=offs_scale_k[None, :] < num_groups_left, other=127)
