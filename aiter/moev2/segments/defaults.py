# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Default kernel parameters used when no tuned row applies.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
This module moved here from aiter/fused_moe.py and is shared with the legacy MoE path:
aiter.fused_moe re-exports it and legacy MoE calls run it. Changes must keep legacy
behavior (`python tests/moe_dispatch/snapshot.py --check`) and should loop in the moev2
owners (@qli88). See aiter/moev2/README.md.
"""

import functools
import os

from aiter.jit.utils.chip_info import get_cu_num

BLOCK_SIZE_M = 32


@functools.lru_cache(maxsize=2048)
def get_block_size_M(token, topk, expert, inter_dim):
    cu_num = get_cu_num()
    tileN = 128
    tgN = (inter_dim + tileN - 1) // tileN
    support_list = [32, 64, 128]

    tmp = []
    for el in support_list:
        max_num_tokens = token * topk + expert * el - topk
        tg_num = tgN * (max_num_tokens + el - 1) // el
        rnd = (tg_num + cu_num - 1) // cu_num
        empty = cu_num - tg_num % cu_num
        tmp.append((rnd, empty, el))
    return min(tmp, key=lambda x: x[:2])[-1]


@functools.lru_cache(maxsize=2048)
def use_nt(token, topk, e):
    use_nt = int(os.environ.get("AITER_USE_NT", "-1"))
    if use_nt != -1:
        return bool(use_nt)
    return (token * topk // e) < 64


@functools.lru_cache(maxsize=2048)
def get_ksplit(token, topk, expert, inter_dim, model_dim):
    aiter_ksplit = int(os.environ.get("AITER_KSPLIT", "0"))
    if aiter_ksplit != 0:
        return aiter_ksplit
    # only for moe_blk gemm1 a8w8 decode scenario
    if token * topk > expert:
        return 0
    cu_num = get_cu_num()
    tileN = 128

    tgM = token * topk  # decode tile num
    tgN = (inter_dim + tileN - 1) // tileN

    tg_num = tgN * tgM
    # if all cu already active
    if tg_num >= cu_num:
        return 0
    tilek = 256
    split_max = (cu_num + tg_num - 1) // tg_num
    # at least split = 2
    for i in reversed(range(2, split_max + 1)):
        if (model_dim % i == 0) and ((model_dim // i) % tilek == 0):
            return i
    return 0
