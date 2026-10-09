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


def pack_scale(x):
    if x is None:
        return x
    NON_K, K_SCALE = x.shape
    preshuffle_factor = 128
    num_chunk_m = NON_K // preshuffle_factor
    SCALE_KWIDTH = min(4, K_SCALE)
    num_chunk_k = K_SCALE // SCALE_KWIDTH

    x = x.view(num_chunk_m, 4, preshuffle_factor // 4, num_chunk_k, SCALE_KWIDTH)
    x = x.permute(0, 3, 2, 1, 4).contiguous()
    return x.view(NON_K // preshuffle_factor, K_SCALE * preshuffle_factor)
