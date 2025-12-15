# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import List, Dict, Any

import torch
import triton
import json

def prev_power_of_2(x: int) -> int:
    out = triton.next_power_of_2(x)
    return out // 2 if out > x else out


STATIC_MAX_SEQ_LENS: List[int] = []
USE_RUNTIME_MAX_SEQ_LEN: bool = False


def autotune_max_seq_len(runtime_max_seq_len: int) -> int:
    global USE_RUNTIME_MAX_SEQ_LEN

    if USE_RUNTIME_MAX_SEQ_LEN:
        return prev_power_of_2(runtime_max_seq_len)
    else:
        if STATIC_MAX_SEQ_LENS == []:
            return 1
        for max_len in STATIC_MAX_SEQ_LENS:
            if max_len >= runtime_max_seq_len:
                return max_len
        return STATIC_MAX_SEQ_LENS[-1]


def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if x.stride(-1) == 1:
        return x
    return x.contiguous()

def serialize_dict(d: Dict[str, Any]) -> str:
    items_list = list(d.items())
    sorted_items = sorted(items_list)
    return json.dumps(sorted_items)

def deserialize_dict(s: str) -> Dict[str, Any]:
    items_list = json.loads(s)
    return dict(items_list)
