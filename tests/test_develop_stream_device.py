# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A develop=True op takes the stream of its first GPU tensor's device.

The C++ side guards to the input's device, so the bridge must not hand it the
current device's stream. The end-to-end two-GPU case is
op_tests/multigpu_tests/test_develop_op_stream_device.py.

Needs a built AITER, so it is not part of the CPU-only gate.
"""

from __future__ import annotations

from conftest import requires_built_aiter

requires_built_aiter()


class FakeTensor:
    def __init__(self, device: int):
        self.device = device

    def get_device(self) -> int:
        return self.device


def pick(*args, **kwargs) -> int:
    from aiter.jit.core import _develop_device

    return _develop_device(args, kwargs, FakeTensor)


def test_first_gpu_tensor_wins():
    assert pick(1, FakeTensor(3), FakeTensor(0)) == 3


def test_cpu_tensors_are_skipped():
    assert pick(FakeTensor(-1), FakeTensor(2)) == 2


def test_tensors_inside_lists_count():
    assert pick(0.5, [FakeTensor(-1), FakeTensor(5)], FakeTensor(1)) == 5


def test_keyword_tensors_count():
    assert pick(None, out=FakeTensor(4)) == 4


def test_no_gpu_tensor_means_current_device():
    assert pick(1, "x", [2, 3], FakeTensor(-1)) == -1
