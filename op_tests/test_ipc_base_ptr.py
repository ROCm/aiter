# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Single-GPU guard for the IPC handle offset used by CustomAllreduce.

``hipIpcGetMemHandle`` hands out a handle for the whole allocation containing a
pointer, so a buffer that is a sub-block of a larger allocation must travel
with its offset or importing ranks address the wrong region. That failure is
silent -- wrong all-reduce results, no error -- so pin the arithmetic here,
where it needs only one GPU.
"""

import pytest
import torch

from aiter.dist.device_communicators.custom_all_reduce import _ipc_base_ptr

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a device allocation to query"
)

_MiB = 1024 * 1024


def test_sub_blocks_of_one_allocation_share_a_base():
    """The regression: distinct sub-blocks must not all report offset 0."""
    buf = torch.empty(4 * _MiB, dtype=torch.uint8, device="cuda")
    head, tail = buf[:16], buf[2 * _MiB :]

    base = _ipc_base_ptr(head.data_ptr())
    assert _ipc_base_ptr(tail.data_ptr()) == base
    assert tail.data_ptr() - base == (head.data_ptr() - base) + 2 * _MiB


def test_base_lies_at_or_below_the_pointer():
    buf = torch.empty(4 * _MiB, dtype=torch.uint8, device="cuda")
    view = buf[1024:]
    assert _ipc_base_ptr(view.data_ptr()) <= buf.data_ptr() <= view.data_ptr()


def test_raw_hipmalloc_buffer_is_its_own_base():
    """Why AITER_CUSTOM_AR_RAW_INPUT_POOL masked the bug: offset is always 0."""
    import aiter as ops

    ptr = ops.allocate_data_buffer(_MiB)
    try:
        assert _ipc_base_ptr(ptr) == ptr
    finally:
        ops.free_meta_buffer(ptr)
