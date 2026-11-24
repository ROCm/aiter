# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.


import json

import torch
import triton


def prev_power_of_2(x: int) -> int:
    out = triton.next_power_of_2(x)
    return out // 2 if out > x else out


STATIC_MAX_SEQ_LENS: list[int] = []
USE_RUNTIME_MAX_SEQ_LEN: bool = False


def autotune_max_seq_len(runtime_max_seq_len: int) -> int:

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


def max_addressable_bytes(t: torch.Tensor) -> int:
    span = 1
    for size, stride in zip(t.shape, t.stride()):
        if size > 1:
            span += (size - 1) * stride
    return span * t.element_size()


def serialize_dict(d: dict) -> str:
    return json.dumps(d)


def deserialize_str(s: str) -> dict:
    return json.loads(s)


def strip_annotate(cls):
    """
    Neutralize '__annotate__' so Triton's aggregate hash walker skips it.

    Triton's '@aggregate' builds 'hash_attrs' from 'inspect.getmembers'.
    On Python 3.14 (PEP 649) that yields the compiler-generated annotate
    function, and the JIT's 'record_reference' rejects it with
    "Unsupported function referenced".

    'strip_annotate' can be used like below to remove the problematic attribute:

        @aggregate
        @strip_annotate
        class MyLoader:
            field: gl.constexpr

    No-op on Python < 3.14.

    Fixed upstream in triton-lang/triton main (PR #9529), but NOT in any 3.7.x release.
    """
    _ = cls.__annotations__
    cls.__annotate__ = None
    return cls
