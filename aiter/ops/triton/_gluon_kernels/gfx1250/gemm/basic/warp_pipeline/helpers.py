# SPDX-License-Identifier: MIT
import triton.experimental.gluon.language as ttgl
from triton.experimental import gluon


@gluon.jit
def chiplet_transform(pid, num_workgroups, num_xcds: ttgl.constexpr):
    """
    Basic chiplet transformation for multi-XCD AMD GPUs.

    Transforms program ID to distribute work evenly across chiplets (XCDs).
    Each XCD gets a contiguous range of work items.
    """
    xcd = pid % num_xcds
    pos_in_xcd = pid // num_xcds
    min_per_xcd = num_workgroups // num_xcds
    extra_sms = num_workgroups % num_xcds
    offset = xcd * min_per_xcd + min(xcd, extra_sms)
    return offset + pos_in_xcd
